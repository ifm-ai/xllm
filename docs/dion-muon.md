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
| `--optim.muon_weight_decay` | unset | Muon weight decay; unset uses `--optim.weight_decay`. |
| `--optim.no_decay_embed_readout` | `false` | Also exclude the token embedding and readout from AdamW weight decay. |
| `--optim.muon_replica_sync` | `false` | Broadcast weights and optimizer state across CP/HSDP replicas after each step. |

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
Other parameters use AdamW in two groups:

| Group | Parameters | Weight decay |
| --- | --- | --- |
| `muon` | Hidden linear weights | `muon_weight_decay`, else `weight_decay` |
| `adamw` | Embedding, readout, short-convolution kernels | `weight_decay` |
| `adamw_no_decay` | Norms, biases and other vectors; embedding and readout with `no_decay_embed_readout` | 0 |

Set `--optim.weight_decay 0` to turn off AdamW weight decay while keeping it
for Muon via `--optim.muon_weight_decay`. Empty groups are dropped.
Existing scheduler settings apply the same multiplier to every group;
existing betas/epsilon configure AdamW. Training logs include one
`optim/lr_<group>` per group.

## Limits and checkpoints

- FSDP2, TP=1, dense FFNs only. No MoE/MoVA or tied parameters.
- CP uses native FSDP2 gradient reduction, so every CP/HSDP replica receives
  identical gradients and computes the same update; Dion itself does not sync
  replicas. `--optim.muon_replica_sync true` adds a per-step broadcast of local
  weight shards and optimizer state across replicas, as a guard against
  nondeterministic kernels.
- Set `--context_parallel_size` as usual. Keep a nontrivial FSDP shard group and
  make `seq_len` divisible by `context_parallel_size * model.chunk_size`.
- Keep `memory_efficient_norm=false`: its separate zero-division issue is not fixed here.
- Hybrid layers share head dimensions and RoPE settings; `v_head_dim=head_dim`.
  Packed generation with a cache is unsupported.
- Normal Muon resume requires unchanged parameter groups and mesh layout;
  other drift is rejected. LRs, weight decay, momentum, betas, epsilon,
  orthogonalization backend and the scheduler horizon come from the config on
  resume, so a run can be extended or its LRs lowered without losing optimizer
  state. The scheduler keeps its position; each group's base LR is the
  configured one.
- To switch from AdamW, use a fresh `dump_dir` plus `base_model_dir`, with
  `reload_base_optim_state=false`. This loads weights, not the old optimizer/data cursor.
  Changing the hybrid layer pattern additionally requires model checkpoint conversion.

Focused checks (run on a CUDA worker with installed backends):

```bash
python -m pytest -q tests/optim/test_dion_muon.py \
  tests/test_hybrid_gekko.py tests/attn/test_attention_regressions.py
```
