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

The width factors cancel in `lr * weight_decay`, so the AdamW timescale
`1 / (lr * weight_decay * steps)` of every role is the same at every width. With
depth scaling, the hidden-matrix and hidden-norm learning rates carry
`m_L^(alpha-1)` and their weight decay has no depth factor, so their timescale
is depth-invariant only at `alpha = 1`; at `alpha < 1` it grows as
`m_L^(1-alpha)`. A different number of steps changes the timescale of every
role; `optim.tau_epoch` (below) keeps it fixed.

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

**Export and serving.** With depth scaling, the `m_L^-alpha` branch multiplier
is part of the model's forward pass (`model.residual_multiplier` in
`config.json`), but the xBridges Hugging Face and vLLM models add the attention
and FFN outputs without it. Before serving a depth-scaled model, fold the
multiplier into the attention output projection (`wo`) and the FFN down
projection (`fc2`), which both feed the residual add and have no bias. Width-only
models have multiplier 1 and export as is.

**Resume.** On resume, the per-role learning rates follow the config (as the
single learning rate does without `mup`), and with `mup` so do the per-role
epsilon and weight decay. Optimizer state can only be reloaded with the param
group layout it was saved with (`mup`, `optim.no_decay_embed_readout`, or
neither); switching layouts raises an error. Model weights load across layouts,
e.g. from `base_model_dir` with `reload_base_optim_state false`.
