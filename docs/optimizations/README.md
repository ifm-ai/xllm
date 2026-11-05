# Gekko training optimizations: B200, H100, H200

The Gekko kernel / training-step optimization work for Blackwell and Hopper, with one report per architecture:

- [SUMMARY.md](SUMMARY.md): one page - the top 5 kernel optimizations and the model speed-ups on the three architectures.
- [KERNELS.md](KERNELS.md): the full kernel-by-kernel table (every item, all three architectures).
- [B200.md](B200.md): Blackwell (sm_100a) results and recipes - done.
- [H100.md](H100.md): Hopper (sm_90a) results and recipes - done; open decisions listed at the end.
- [H200.md](H200.md): H200 (sm_90a, 141 GB HBM3e) results and recipes - one pass on a KVM node whose host clock is
  bimodal; open items at the end.
- `patches/`: the local xattn 1.1.1 patches the H100 / H200 numbers depend on.

How the numbers were measured: `build_model(cfg, dtype="bf16", ...)` under FSDP2 on 8 GPUs with the preset's
`ModelConf` and the fields below, random tokens with an EOS every ~3000 positions (`multi_segments=True`), the step
= forward + backward + `clip_grad_norm_` + fused AdamW, timed with `torch.cuda.synchronize()` around each part;
per-kernel times from `torch.profiler` traces of one step.

The code selects the path per architecture at runtime (`torch.cuda.get_device_capability`): Blackwell uses the
CuTe DSL tcgen05 / TMEM kernels and MXFP8 GEMMs, Hopper the wgmma kernels, blockwise fp8 GEMMs and the patched
xattn / swift attention. Everything is opt-in or falls back to the previous implementation (torch / CUB / cuBLAS /
Triton) when a shape or device is not handled.

Environment used for the numbers: torch 2.14.1+cu130, nvidia-cutlass-dsl 4.8.1, NCCL 2.30.7, CUDA 13.1; the Gekko
presets `gekko-0.9B` (B2 L16384), `gekko-4B` (B1 L8192), `gekko-7.1B` (B1 L16384 on B200, L8192 on H100 / H200, L16384 also on H200),
`fused_block=True`, FSDP2 on 8 GPUs; per-step time = forward + backward (clip and optimizer reported separately).

## What is in the branch

Kernels (all `xllm/modules/fused_ops/cute/`, bf16 in / out, fp32 math, CuTe DSL):

| kernel | B200 | H100 / H200 | replaces |
|---|---|---|---|
| timestep decay norm fwd / bwd (`timestep_decay_norm.py`) | tcgen05-free, f32x2 math | scalar fallback | CUB kernels |
| adaptive working memory (`adaptive_working_memory*.py`) | tcgen05 / TMEM recurrence, chunk and output GEMMs | wgmma kernels (`_sm90.py`) | torch / cuBLAS |
| sliding chunk attention (`sliding_chunk_attention.py`) | flash-style fwd, dV / dK / dQ with TMEM-A operands, materialized-P path for V > 256, optional MXFP8 dP | not supported (xattn / swift with local patches) | swift (cuBLAS + materialized attention) |
| rotary embedding, q / k prep (rmsnorm + rope + softmax), attention gate, SwiGLU, group LN / RMS norm, cross-entropy row kernels (`rope.py`, `qk_prep.py`, `attn_gate.py`, `swiglu.py`, `layer_norm.py`, `group_norm.py`, `cross_entropy.py`, `softmax_bwd.py`) | yes | yes | torch elementwise / norms |
| causal conv (`causal_conv.py`, `causal_conv_sm90.py`) | fwd (bwd stays Triton) | fwd + bwd | Triton |
| conv + q / k prep fused (`conv_qk_prep.py`) | yes (written on H100, measured faster on B200 too) | yes | conv + qk prep |
| MXFP8 GEMM path (`mxfp8.py`, `ModelConf.fp8_gemm`) | e4m3 + e8m0 per 32, cuBLASLt | - | bf16 cuBLAS |
| blockwise fp8 GEMM path (`fp8_sm90.py`, `fp8_fsdp.py`; opt-in DeepGEMM backend `XLLM_FP8_DEEPGEMM=1`) | - | 1 x 128 / 128 x 128 e4m3, cuBLASLt or DeepGEMM | bf16 cuBLAS |
| CUDA-graph replay of the fused block (`xllm/models/fused_blocks/gekko/graphed.py`, `ModelConf.cuda_graph_block`) | yes | H200: runs without EOS-packed batches only (xattn's host-side BOS plan), see H200.md | eager launches |

Training-step changes (`xllm/models/fused_blocks/gekko/gekko.py`, `utils.py`, `xllm/distributed/fsdp_patches.py`):
fused residual adds / GEMM epilogues, `recompute_conv_qk=False` (keep the conv / q-k prep outputs), FSDP2
`fsdp_layers_per_group`, `reshard_after_forward=False`, fp8 all-gather (`XLLM_FP8_AG`), symmetric-memory
collectives (`XLLM_FSDP_SYMM`, H100), gradient-divide folded into the reduce-scatter copy-in, fused AdamW.

## Configuration and environment switches

`ModelConf` (xllm/config.py):

| field | default | effect |
|---|---|---|
| `timenorm_backend` | `cub` | `cute`: CuTe TSDN kernels (B200 / H100) |
| `awm_backend` | `torch` | `cute`: CuTe AWM kernels (B200 tcgen05 / H100 wgmma) |
| `sca_backend` | `swift` | `cute`: CuTe sliding chunk attention (B200 only; falls back to swift elsewhere); `xattn` on H100 |
| `recompute_conv_qk` | True | False: keep the conv / q-k prep / norm outputs from the forward (~5-10% faster, more memory) |
| `fsdp_layers_per_group` | 1 | layers per FSDP2 all-gather group (B200: 4; H200: 4 / 6 / 2 per preset) |
| `fp8_gemm` | False | projection / FFN GEMMs in fp8 (MXFP8 on B200, blockwise on H100) |
| `fp8_attn_dp` | False | B200, v_head_dim > 256: the attention backward's dP GEMMs in MXFP8 |
| `cuda_graph_block`, `cuda_graph_warmup` | False, 2 | CUDA-graph replay of each fused block (needs dropout 0, no context parallelism, `reshard_after_forward=False`) |
| `balance_dp_rows` (training config) | False | redistribute the rows of a step across DP ranks to even out the attention work of EOS-packed rows (used in the H100 recipes) |

Environment switches (all optional; `=0` disables a kernel that is on by default, `=1` enables an opt-in):

- B200 kernels: `XLLM_CUTE_ROPE / QK / GATE / SWIGLU / CE / LN / CONV=0` (fall back to torch / Triton),
  `XLLM_CUTE_CONV_BWD=1` (CuTe conv backward, slower than Triton), `SCA_DV_PMAT=0/1` (force the kernel / materialized-P
  backward path for V > 256; default: by attended fraction), `SCA_DP_FP8=0`, `SCA_TRIL_N`, `XLLM_CG_DEBUG=1`.
- H100 / H200: `XLLM_CONV_QK=0` (fused conv + q/k prep off), `XLLM_CONV_SM90=0`, `XLLM_QUACK_FFN / CE=0`,
  `XLLM_FSDP_SYMM=1`, `NCCL_CTA_POLICY=2`, `XLLM_FP8_AG=1`, `XLLM_FP8_KEEP_X=1`, `XLLM_GEKKO_SAVE_RMX=1`,
  `XLLM_GEKKO_SAVE_QK_PREP=1`, `XLLM_FP8_DEEPGEMM=1` (DeepGEMM backend for the fp8 GEMMs; `XLLM_FP8_DEEPGEMM_SMS` caps its SMs, `XLLM_FP8_SM90_WGRAD_TILE`), `XLLM_SWIFT_*`, `XLLM_XATTN_DPSUM`,
  `XLLM_CE_DGRAD_FP32=1` (fp32 scores for the CE gradient GEMM), `XLLM_FSDP_FOLD_DIV`, `XLLM_FSDP_DIRECT_AG`.
- `reshard_after_forward=False` is the `build_model(...)` argument.

## Health metrics

`health_metrics=True` (the `HealthMetricCollector` hooks inside the fused Gekko block, the AWM forward and the
chunked cross entropy) needs the pre-RoPE q / k, the gate activation, the memory update ratio and the logits, which
the fused conv + q / k prep, q / k prep, gate, AWM and cross-entropy kernels never materialise. A sampled logging
step (every `health_log_freq` steps) therefore runs the reference path of those stages in the forward and the
backward (`ctx.health`); every other step is unaffected. `cuda_graph_block` cannot record health metrics: a replayed
graph would skip the hooks, so the block asserts when a collector is active.

## Tests

- B200: `tests/test_cute_sliding_chunk_attention.py`, `test_cute_adaptive_working_memory.py`,
  `tests/normalization/test_cute_timestep_decay_norm.py`, `test_cute_{rope,qk_prep,attn_gate,swiglu,layer_norm,
  causal_conv,cross_entropy,fused_elementwise}.py`, `test_mxfp8.py`, `test_fp8_gekko_block.py` (small Gekko model
  fp8 vs bf16 over the recompute knobs, and `fp8_attn_dp`), `test_fsdp_copy_in.py`, `test_health_fused_gekko.py` (a health-logging step against the fused step, and
  `cuda_graph_block` refusing one).
- H100 / H200: `test_fp8_sm90.py` (`test_mm_deepgemm` needs `deep_gemm` installed, skipped otherwise), `test_fp8_gekko_block_sm90.py`, `test_fp8_fsdp.py`, `test_cute_causal_conv_sm90.py`,
  `test_cute_conv_qk_prep.py`, `test_quack_{swiglu_ffn,linear_ce}.py`, `test_swift_attention.py`,
  `tests/attn/test_swift_blocks.py`, `tests/attn/test_attention_regressions.py`.
- Run with the extension on the path: `PYTHONPATH=<xllm_extension build dir> python -m pytest tests/<file>`; the
  CuTe tests load their module from the file and skip on other architectures.
- `test_cute_timestep_decay_norm.py::test_backward_residual_accumulate` allows one ulp of the output dtype on the
  gamma / beta gradients: they come from fp32 atomics whose summation order is not reproducible.
