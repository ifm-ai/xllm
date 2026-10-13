# Dion-based Muon

This integration uses `dion.Muon`, not the separate Dion optimizer algorithm.
Install the pinned optional dependency:

```bash
python -m pip install -r requirements-dion.txt
```

Set these options in a normal training command:

```text
--model.ddp_backend fsdp2
--optim.optimizer dion_muon
--optim.muon_lr 0.01
--optim.lr 0.001
--model.memory_efficient_norm false
--dcp_for_optimizer true
--async_checkpointing false
```

Muon handles trainable hidden linear projection weights. Dion's AdamW handles
embeddings, final output weights, all biases and norms, and Gekko's 2-D
depthwise convolution kernels. Frozen weights are excluded. MoE/MoVA, tied
weights and TP are currently rejected. The existing AdamW optimizer remains
the default and does not require installing Dion.

## Recipe controls

| Option | Default | Meaning |
| --- | --- | --- |
| `optim.muon_lr` | 0.01 | Muon LR before matrix-size adjustment |
| `optim.lr` | 0.001 | AdamW LR |
| `optim.muon_momentum` | 0.95 | Muon momentum |
| `optim.muon_nesterov` | false | Nesterov momentum |
| `optim.muon_adjust_lr` | spectral_norm | spectral_norm or rms_norm |
| `optim.muon_orthogonalizer` | newton_schulz5 | newton_schulz5 or polar_express |

`optim.weight_decay` and `optim.epsilon` apply to both groups; `beta1` and
`beta2` apply to AdamW. Both LRs follow the configured warmup/decay curve.
Training logs expose `optim/lr_muon` and `optim/lr_adamw`.

## Hybrid Gekko

`model.arch: gekko` with `model.full_attention_interval: 4` builds
`full, Gekko, Gekko, Gekko, ...`, with full attention in the first layer.
Zero, the default, retains pure Gekko and its original output layer.
The hybrid reuses both existing block implementations and the Transformer
output norm. Head dimensions and RoPE settings are shared; full-attention
layers require equal Q/K and V head dimensions.

Use `model.causal_attn_backend: xattn` for full layers and
`model.sca_backend: xattn` for Gekko layers. Regular and fused blocks retain
their existing recomputation settings. Packed training supplies full-attention
segment metadata and Gekko's previous-chunk metadata independently.
Full-attention recomputation and backward retain the selected FlashAttention
or xattn backend, including its packed/CP metadata.
Unpacked inference uses the corresponding cache type per layer; packed
inference with a cache is not supported by the hybrid.

## Context parallelism and checkpoints

The release's FSDP2 mesh shards across `dp` and replicates across `cp`
(including any additional HSDP replicas). FSDP2 performs gradient reduction;
Muon receives only the 1-D `dp` shard mesh for matrix orthogonalization.
There is no second optimizer-level CP gradient reduction. After the optimizer
step, a coalesced broadcast keeps parameter shards and optimizer tensors
identical across replicas; independent BF16 orthogonalizations can otherwise
drift. This adds replica communication but avoids inconsistent checkpoints.

Use `context_parallel_size > 1` with sequence length divisible by
`context_parallel_size * model.chunk_size`. The release requires at least two
FSDP shard ranks when CP is enabled; CP=2 therefore needs at least four GPUs.
For packed FlashAttention full layers, the release additionally requires
per-DP batch size 1. xattn is used by the qualification below.

Normal resume preserves both optimizer groups and their saved LRs. A small
training-state contract checks optimizer choice, parameter membership, recipe,
orthogonalizer and mesh shape before loading state. Changing these requires a
fresh optimizer (`base_model_dir` with `reload_base_optim_state: false`).
Optimizer transfer across topologies is not qualified.

## Tests

Focused routing, configuration and scheduler tests:

```bash
python -m pytest -q tests/optim/test_dion_muon.py
```

Real-kernel distributed tests use the same seeded batches, DP size and global
token count at CP=1 and CP=2. They include document boundaries inside chunks,
masked tokens, gradient checks, parameter-update parity and exact DCP resume.
BF16 gradient parity requires each tensor's RMS error to stay within 1%
of its reference gradient RMS (plus an absolute per-entry floor of 1e-6),
and its largest error within 3% of that
tensor's maximum gradient magnitude (plus 1e-5). Near-zero entries can cancel
differently when CP partitions BF16 matrix products.
Trained weights use 3% tensor-RMS tolerance plus 10% of the AdamW LR, and a
maximum-entry bound of 3% of the reference tensor's largest weight plus the
AdamW LR. This allows
rare AdamW sign changes near zero without dropping the whole-tensor check.
Checkpoint state and the resumed update still require bit-exact equality.
The following example tests the fused 1:3 hybrid:

```bash
torchrun --standalone --nproc_per_node=2 tests/optim/distributed_dion_muon.py \
  --arch hybrid --fused --cp 1 --save-reference /tmp/hybrid-cp1.pt \
  --output /tmp/hybrid-cp1.json
torchrun --standalone --nproc_per_node=4 tests/optim/distributed_dion_muon.py \
  --arch hybrid --fused --cp 2 --reference /tmp/hybrid-cp1.pt \
  --output /tmp/hybrid-cp2.json
```

Repeat with `--arch transformer` and `--arch gekko`, and without `--fused`.
Use `--full-attn-backend flash` to test the installed FlashAttention variant,
`--orthogonalizer polar_express` for that recipe, and `--optimizer adamw` to
check the unchanged default optimizer.
CP=4 needs eight ranks with the same DP size. Each output includes loss,
gradient norm, throughput, peak allocated memory and checkpoint-resume result.
Passing these tests qualifies only the tested runtime and topology; run them
again after changing xattn, PyTorch, CUDA or GPU architecture.

## Qualification status (2026-10-13)

Rebased onto release `5cc29a2`, with model code tested at `6ef4996` on
H200, PyTorch 2.11.0+cu128,
CUDA toolkit 12.9, xattn `9bfc56a`, FlashAttention 4.0.0b34 and the pinned
Dion revision. All 24 focused tests pass, including partial-RoPE preservation
and packed-xattn metadata trimming with either BOS masks or segment indices.
Runs are recorded in the
[qualification W&B project](https://wandb.ai/mbzuai-llm/xllm-release-v1.01-muon).

| Architecture | Attention backends | Passed CP sizes, DP=2 | Blocks |
| --- | --- | --- | --- |
| Transformer | xattn | 1, 2, 4 | Regular and fused |
| Pure Gekko | xattn | 1, 2, 4 | Regular and fused |
| 1:3 hybrid | xattn + xattn | 1, 2, 4 | Fused |
| 1:3 hybrid | xattn + xattn | 1, 2 | Regular |

Passing cases include exact model/optimizer checkpoint restore and an exact
resumed update. Hybrid CP1 also tests cache prefill/decode. For the all-xattn
hybrid, regular CP4 exceeded the first-gradient budget on
`layers.2.gda.v_conv.weight` (3.04% relative L2; 2.24e-6 absolute RMS).
Its loss remained close to CP1, and whole-model gradient relative L2 error
was 0.45%. The discrepancy occurs before the first optimizer update and
reproduces unchanged after the rebase; it is not yet proven to be rounding.
Probes with the existing `attn_stability_control` settings 1 and 2 are queued.
No gradient thresholds were relaxed during this diagnosis.

32K fused Gekko and hybrid CP1 tests passed before the rebase. Fresh CP1
references and cross-node CP4 tests on four 2-GPU nodes are queued; this
layout exercises both CP and FSDP/Muon communication across nodes.
Do not treat these pending checks, regular hybrid CP4, or CP16 as qualified.
FlashAttention 4 dense and FlashAttention 4 + Swift hybrid/Gekko CP1/2/4
passed on the earlier base; focused FA4 regressions after the rebase are queued.

These are small, seeded four-layer correctness tests, not convergence or
performance claims. The cold three-step throughput includes JIT startup and
must not be used to compare attention speed. The updated release's packed
xattn changes required trimming metadata on the sequence axis and keeping
fused recomputation/backward metadata aligned with trimmed keys.
