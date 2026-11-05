# Gekko training optimizations: full kernel table (B200 / H100 / H200)

The per-architecture reports ([B200.md](B200.md), [H100.md](H100.md), [H200.md](H200.md)) hold the recipes,
methodology and caveats. "x" = speed-up of the item
against what it replaced, measured on the same shape; per-call numbers are GPU time per kernel call, per-step
numbers are the contribution to one training step.

## 1. Kernels

| kernel (what it replaced) | B200 (sm_100a, CuTe tcgen05 / TMEM) | H100 (sm_90a, CuTe wgmma / patched xattn) | H200 (same code as H100) |
|---|---|---|---|
| Timestep decay norm fwd / bwd (CUB) | 1488 / 3104 -> 102 / 156 us per call (**14.6x / 19.9x**, ~85% of HBM roofline); 0.9B step 120 -> 2.3 ms | 46.6 -> 2.3 ms / step (**~20x**) | bwd + residual 161 -> 117 us / layer vs H100 (1.38x) |
| Adaptive working memory forward (torch / cuBLAS) | 0.9B 1376 -> 125 us, 4B 729 -> 91, 7.1B 1259 -> 207 (**6-11x**) | 1796 -> 244 us (**7.4x**) | same |
| Adaptive working memory backward (torch) | per step 0.9B 99 -> 6 ms, 7.1B 98 -> 11.4 ms (**9-16x**) | chunk backward 432 -> 109 us (**4.0x**) | same |
| Sliding chunk attention (swift: materialised attention matrix) | 0.9B step 1151 -> 590 ms, peak memory 139 -> 40 GB; 7.1B backward 6.13 -> 2.77 ms / layer (**2.2x**); forward at ~95% of the tensor-pipe peak | not ported (xattn / swift kept): swift 7.1B fwd + bwd 3796 -> 2174 us / layer (**1.75x**), xattn D = 256 bwd 4.8 -> 3.9 ms / layer (1.23x) | xattn bwd 4025 -> 3908 us (L2-bound, no gain) |
| RoPE, q / k prep (rmsnorm + rope + softmax), conv + q / k prep fused (torch ops) | 0.9B step -58 ms (RoPE) and -54 ms (q / k prep); conv + q / k prep fused kernel also used | fused RoPE 72 -> 12 ms / step (**6x**); conv + q / k prep fwd 264 -> 153, bwd 643 -> 262 us / layer (**1.7x / 2.5x**) | 106 / 210 us (1.44x / 1.26x vs H100) |
| Attention gate chain (add + rmsnorm + softplus + mul, fwd + bwd) | steps 0.9B 379 -> 356, 4B 368 -> 334, 7.1B 893 -> 852 ms; 7.1B gate bwd 385 -> 322 us / layer | block elementwise (softmax bwd, bf16 softmaxes, gate) 145 -> 41 ms / step (**3.5x**); gate straight into fp8 4B 143 -> 104 us / layer | 252 -> 241 us (1.05x) |
| SwiGLU elementwise (3 passes -> 1; fp8: writes the MX / fp8 operands directly) | 4B step 334 -> 323 ms; fp8 path bwd 388 -> 320 us | quack fused SwiGLU FFN 5613 -> 5148 us / layer (1.08x) | same |
| Group layer norm fwd / bwd (extension kernels) | bwd 7.1B 153 -> 102 us, 0.9B 117 -> 86 (**1.4-1.5x**); gamma / beta bwd row-split (was 15-30 ms / step) | 56 -> 32 ms / step (**1.75x**) | same |
| Cross entropy (row kernels, gradient GEMM) | fp32 gradient GEMM 64 -> ~5 ms / step (TF32); row kernels 0.9B step 352 -> 338 ms | gradient GEMM 581 -> 246 us / chunk (**2.4x**); row kernels 43 -> 25 ms / step (1.7x) | same |
| Causal conv1d (Triton) | fwd 128 / 126 / 207 -> 75 / 59 / 164 us (**1.3-2.1x**); bwd small shapes 97 -> 46 us | 4B v conv fwd 19 -> 11, bwd 89 -> 36 us (**1.7x / 2.5x**) | same |
| FSDP2 reduce-scatter copy-in (torch._chunk_cat), clipping on flat buffers, fused AdamW on local shards | 8 GPUs 7.1B 828 -> 818, 0.9B 348 -> 326 ms / step; clip 9 -> 4, optimizer 13 -> 3 ms | fused AdamW 13 -> 7 ms (1.9x) | same |
| Linear GEMMs in fp8 (bf16 cuBLAS) | MXFP8: GEMM time / layer 7.1B 12.9 -> 7.0 ms, 4B 3.46 -> 1.97, 0.9B 4.04 -> 2.33 (**1.75-1.84x**) | blockwise e4m3: 755-790 TFLOP/s -> 1.0-1.24 PFLOP/s (**1.5-1.7x**) | cuBLASLt 1.08-1.32 PF; DeepGEMM another **1.10-1.35x** per GEMM (opt-in) |
| Attention backward dP = dO V^T in MXFP8 (7.1B, opt-in) | 1113 -> 972 us / layer (1.15x) | - | - |
| CUDA-graph replay of the fused block | 4B: removes the bimodal step (251 / 280 -> 251 ms); 0.9B / 7.1B no gain (power-capped / GPU-bound) | memory-bound on 80 GB | 4B 468-478 -> 431-454 ms without EOS-packed batches (needs an xattn change with them) |

## 2. Models (8 GPUs, FSDP2, ms per training step fwd + bwd, medians)

| preset | B200: start -> best | | H100: start -> best | | H200: start -> best | |
|---|---|---|---|---|---|---|
| gekko-0.9B B2 L16384 | 1258 -> **270** ms | **4.7x** | 865 -> **465** | **1.86x** | 1075 -> **444** | **2.42x** |
| gekko-4B B1 L8192 | 967 -> **250** | **3.9x** | 667 -> **393-405** | **1.68x** | 762 -> **419** | **1.82x** |
| gekko-7.1B B1 L8192 (H100 / H200), B1 L16384 (B200) | 1150 -> **550** | **2.1x** | 1003 -> **584** | **1.71x** | 1082 -> **536** | **2.02x** |
| tokens / s / GPU at best (from the fwd + bwd median) | 121k / 33k / 30k | | 70k / 20k / 14k | | 74k / 20k / 15k | |
| peak memory at best (GiB) | 73 / 82 / 134 | | 69 / 55 / 68 | | 69 / 55 / 86 | |

- "start" = CUB TSDN, torch AWM, swift / upstream xattn attention, torch elementwise, bf16 GEMMs. On B200 the
  start was measured on 1 GPU (per-rank batch identical; 8-GPU communication would only add to it), the rest on 8.
- "best" = the recommended recipe of each report: all kernels on, `recompute_conv_qk=False`,
  `fsdp_layers_per_group` > 1, `reshard_after_forward=False`, `fp8_gemm=True`; plus `cuda_graph_block` (B200 4B),
  `fp8_attn_dp` (B200 7.1B), `XLLM_FSDP_SYMM=1` and `balance_dp_rows` (H100 / H200), DeepGEMM (H200 7.1B).
  bf16-only (no fp8) B200 steps: 308 / 270 / 771 ms; H100 DeepGEMM rows: 454 / 388 / 574.
- H200 7.1B at B1 L16384 fits without the FFN recompute (119 GiB): 1048 ms, 15.4k tokens / s / GPU (OOM on H100).
- Numerics: every kernel is unit-tested against an fp32 / fp64 reference; end to end, all-on vs all-off on the same
  batches: bf16 items within 2e-4 of loss, fp8 within ~1e-3 (soaks of 100-200 steps on each architecture).
