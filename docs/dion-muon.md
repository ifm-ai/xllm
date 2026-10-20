---
date: 2026-10-20
---

# Hybrid Gekko + Dion-based Muon

This branch stacks Muon on `yash.akhauri/training-dynamics-v1.0.2-hybrid-gekko`,
which includes `yash.akhauri/training-dynamics-v1.0.2-attention-fixes`.
The shared base is [`training-dynamics-branch` / `release-v1.0.2` at `9f59843`](https://github.com/ifm-ai/xllm/commit/9f59843b1086719a0f5a8b4c844921870a15b964);
the hybrid parent is `7cf5c55`.
The implementation uses [Dion's Muon](https://github.com/microsoft/dion/tree/7692479288f928a3d61d1cc58137b4dbe4f71000),
not the separate Dion optimizer algorithm.

## New arguments

Arguments below are added by this stack; defaults are defined in [config.py](../xllm/config.py).

| Argument | Default | Meaning |
| --- | --- | --- |
| `--model.full_attention_interval` | `0` | Gekko: `0` = pure; `4` = full/Gekko/Gekko/Gekko, starting at layer 0. Other integer intervals >=2 are allowed. |
| `--optim.optimizer` | `adamw` | Select `dion_muon` to enable Muon. |
| `--optim.muon_lr` | `0.01` | Muon LR before matrix-size adjustment. Existing `--optim.lr` controls the AdamW group. |
| `--optim.muon_momentum` | `0.95` | Muon momentum. |
| `--optim.muon_nesterov` | `false` | Enable Nesterov momentum. |
| `--optim.muon_adjust_lr` | `spectral_norm` | Matrix-size LR adjustment: `spectral_norm` or `rms_norm`; unrelated to activation RMSNorm. |
| `--optim.muon_orthogonalizer` | `newton_schulz5` | `newton_schulz5` or `polar_express`. |
| `--optim.muon_use_gram_newton_schulz` | `false` | Select Dion's Gram Newton-Schulz backend; requires the optional GNS dependency. |
| `--optim.muon_use_triton` | `false` | Enable accelerated orthogonalization kernels. With GNS, forwards to its `ns_use_kernels` setting. |

## Use

Install xLLM and xattn as described in the [README](../README.md), then install
the optional pinned dependency:

```bash
python -m pip install -r requirements-dion.txt
```

For optional accelerated backends, install the matching extras:

```bash
python -m pip install "dion[gns,triton] @ git+https://github.com/microsoft/dion.git@7692479288f928a3d61d1cc58137b4dbe4f71000"
```

The GNS extra pins CUTLASS/Quack versions; check compatibility with the
attention backend before changing an existing container.

GNS takes precedence over `muon_orthogonalizer` in Dion; that setting selects
Newton-Schulz5/Polar Express only when GNS is off. Both new flags default to
`false`, including when loading older configs. Kernel compilation/autotuning can
make the first steps slow; enabling a backend does not guarantee a speedup.

Append these settings to an otherwise complete training command:

```bash
--model.arch gekko --model.full_attention_interval 4 \
--model.ddp_backend fsdp2 \
--model.causal_attn_backend xattn --model.sca_backend xattn \
--model.memory_efficient_norm false \
--optim.optimizer dion_muon --optim.muon_lr 0.01 --optim.lr 0.001 \
--dcp_for_optimizer true --async_checkpointing false \
--model_parallel_size 1
```

For pure Gekko, set the interval to `0`. For a dense Transformer, use
`--model.arch transformer --model.full_attention_interval 0`.
Regular and fused blocks are supported; `--model.fused_block true` is optional.
These example LRs are not a tuned recipe.

Muon updates trainable hidden linear weights, including Gekko projections.
Embeddings, final output, biases, norms and short-convolution kernels use AdamW.
Existing scheduler settings apply the same multiplier to both LR groups;
existing betas/epsilon configure AdamW, and weight decay applies to both groups.
Training logs include `optim/lr_muon` and `optim/lr_adamw`.

## Limits and checkpoints

- FSDP2, TP=1, dense FFNs only. No MoE/MoVA or tied parameters.
- CP uses native FSDP2 gradient reduction. Muon synchronizes weights and optimizer
  tensor states across CP/HSDP replicas after each update; it does not reduce gradients twice.
- Set `--context_parallel_size` as usual. Keep a nontrivial FSDP shard group and
  make `seq_len` divisible by `context_parallel_size * model.chunk_size`.
- Keep `memory_efficient_norm=false`: its separate zero-division issue is not fixed here.
- Hybrid layers share head dimensions and RoPE settings; `v_head_dim=head_dim`.
  Packed generation with a cache is unsupported.
- Normal Muon resume requires unchanged parameter groups, recipe, scheduler horizon
  and mesh layout. Both saved LR trajectories are preserved. Drift is rejected.
  Older checkpoints without kernel flags resume with both flags `false`;
  switching backends requires a fresh optimizer, not a normal resume.
- To switch from AdamW, use a fresh `dump_dir` plus `base_model_dir`, with
  `reload_base_optim_state=false`. This loads weights, not the old optimizer/data cursor.
  Changing the hybrid layer pattern additionally requires model checkpoint conversion.

Focused checks (run on a CUDA worker with installed backends):

```bash
python -m pytest -q tests/optim/test_dion_muon.py \
  tests/test_hybrid_gekko.py tests/attn/test_attention_regressions.py
```
