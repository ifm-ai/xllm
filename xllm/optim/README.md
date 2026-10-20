# Complete(d)P width/depth scaling

Tune hyperparameters on a small base model, then train wider and deeper models
with the same `optim.lr`, `optim.epsilon` and `optim.weight_decay`. The `mup`
config scales init, learning rate, Adam epsilon and weight decay per parameter
role, and scales residual branches with depth, following Complete(d)P
([arXiv:2512.22382](https://arxiv.org/abs/2512.22382), Table 1). Without a
`mup` block, training is unchanged.

```bash
torchrun ... train.py \
  --model.init_mode gaussian --model.model_dim 2048 --model.num_layers 32 \
  --mup.base_model_dim 256 \
  --mup.base_init_embed_std 1.0 \
  --mup.base_init_hidden_std 0.02 \
  --mup.base_init_unembed_std 0.02 \
  --mup.base_num_layers 8 --mup.alpha 1.0 \
  --optim.lr 1e-2 --optim.epsilon 1e-8 --optim.weight_decay 0.1
```

`base_num_layers` and `alpha` (in [0.5, 1]) are optional and enable depth
scaling. With `m_N = model_dim / base_model_dim` and
`m_L = num_layers / base_num_layers`:

| Parameter role | Init std | Learning rate | Adam epsilon | Weight decay |
| --- | --- | --- | --- | --- |
| Input embedding | `base_init_embed_std` | `1` | `m_N^-1` | `1` |
| Hidden matrix | `base_init_hidden_std * m_N^-1/2` | `m_N^-1 m_L^(alpha-1)` | `m_N^-1 m_L^-alpha` | `m_N` |
| Hidden bias/norm | `1` (norm weights) | `m_L^(alpha-1)` | `m_N^-1 m_L^-alpha` | `1` |
| Final norm | `1` | `1` | `1` | `1` |
| Readout | `base_init_unembed_std * m_N^-1` | `m_N^-1` | `1` | `m_N` |

With `mup`, set the init through `mup.base_init_*` (values at the base width).
`model.init_std`, `model.init_embed_std` and `model.init_logits_std` are set
from them; passing a different value is an error, since a fixed std would undo
the width scaling for that layer.

Attention and FFN branch outputs are multiplied by `m_L^-alpha` before the
residual add. Roles are assigned by `xllm.optim.param_roles`; the per-role LRs
are logged as `lr/<role>` in the optim metrics. xLLM's QK-norm weights are per
channel, so they follow the hidden bias/norm row (Complete(d)P's separate QK-norm
epsilon rule is for gains shared across heads).

**Supported models.** Dense Transformer (`arch transformer`, no MoE/MoVA),
`ddp_backend fsdp2`, `init_mode gaussian`, `scale_emb false`, `residual_func base`.
Depth scaling also needs `fused_block false`. Across a width
family, keep `head_dim` and the vocabulary fixed and scale `num_heads`,
`num_kv_heads` and `ffn_hidden_dim` with `model_dim`. Training only knows
`base_model_dim`, so it cannot check this; call
`xllm.config.validate_mup_width_family(base_model, model)` on the base and target
model configs, e.g. in a sweep script.

**Options.**
- `mup.zero_query`: zero-init the query weights. Not supported with
  `model.qknorm`: QK-norm rescales each query head to unit size, so a zero-init
  query is full size after the first step whatever the learning rate.
- `mup.zero_readout`: zero-init the readout weights; works with QK-norm.
- `optim.no_decay_embed_readout`: no weight decay on the token embedding and
  readout matrix; works with and without `mup`.
- `optim.tau_epoch`: set the base weight decay to
  `1 / (optim.lr * tau_epoch * steps)` instead of `optim.weight_decay`, keeping
  the AdamW timescale fixed as a fraction of training (as in the CompleteP
  experiments, [arXiv:2505.01618](https://arxiv.org/abs/2505.01618)). The
  per-role factors above are applied on top.

**Gradient clipping.** Complete(d)P has no rule for clipping, and the global
gradient norm grows with width (the readout dominates it), so a fixed
`optim.clip` clips wide models more than narrow ones. Use `--optim.clip 0` to
disable clipping, e.g. for learning-rate transfer sweeps.

**Resume.** On resume, the per-role learning rates follow the config (as the
single learning rate does without `mup`), and with `mup` so do the per-role
epsilon and weight decay.
