Part of #7 (the coordinate check follows in a separate PR). Depends on #17.

## Summary

Adds an optional `mup` config that scales init, learning rate, Adam epsilon and weight decay per parameter role relative to a base model, and multiplies the attention/FFN residual branches by `m_L^-alpha`, following Complete(d)P ([arXiv:2512.22382](https://arxiv.org/abs/2512.22382), Table 1). Without a `mup` block, training is unchanged (checked bitwise against `main`, see Testing).

With `m_N = model_dim / base_model_dim` and `m_L = num_layers / base_num_layers`:

| Role | Init std | LR | Adam eps | WD |
| --- | --- | --- | --- | --- |
| Input embedding | `base_init_embed_std` | 1 | `m_N^-1` | 1 |
| Hidden matrix | `base_init_hidden_std * m_N^-1/2` | `m_N^-1 m_L^(alpha-1)` | `m_N^-1 m_L^-alpha` | `m_N` |
| Hidden bias/norm (incl. QK-norm) | 1 | `m_L^(alpha-1)` | `m_N^-1 m_L^-alpha` | 1 |
| Final norm | 1 | 1 | 1 | 1 |
| Readout | `base_init_unembed_std * m_N^-1` | `m_N^-1` | 1 | `m_N` |

`base_num_layers` and `alpha` (in [0.5, 1]) are optional; without them only width scaling is applied. Usage and options are documented in `xllm/optim/README.md`.

## Changes

- `MuPConf` (`--mup.*`): base width, base init stds, optional base depth and alpha. `resolve()` validates the supported setup (dense Transformer, FSDP2, Gaussian init, `scale_emb false`, `residual_func base`; depth scaling also needs `fused_block false`) and sets the model's init stds and residual multiplier. An explicit `model.init_std` / `init_embed_std` / `init_logits_std` that conflicts with `mup` is an error (set `mup.base_init_*` instead).
- `xllm.optim.param_roles()`: assigns every trainable parameter to exactly one of the five roles (raises otherwise). xLLM's QK-norm weights are per channel, so they follow the hidden bias/norm row.
- `xllm.config.validate_mup_width_family()`: checks that a width family keeps `head_dim` and the vocabulary fixed and scales heads, KV heads and FFN width with `model_dim`.
- `BaseResidual` takes a branch multiplier (`m_L^-alpha`), passed through attention and FFN.
- `mup.zero_query` / `mup.zero_readout`: optional zero init of the query and of the readout weights. `zero_query` is rejected with `model.qknorm`, since QK-norm rescales the query to unit size after the first step.
- `optim.no_decay_embed_readout`: no weight decay on the token embedding and readout matrix, with or without `mup`.
- `optim.tau_epoch`: optional; base weight decay `1 / (lr * tau_epoch * steps)` (CompleteP's fixed AdamW timescale), then the per-role factors above. Unset by default.
- Per-role LRs are logged as `lr/<role>`.
- Reloading optimizer state (resume, or `base_model_dir` with `reload_base_optim_state`) checks that the checkpoint was saved with the same param group layout (`default`, `no_decay_embed_readout` or `mup`, from its `config.json`) and raises a clear error otherwise. Model weights still load across layouts.

## Bug fixes from #17

This PR still contains the two `train.py` fixes from #17 (`optim.clip <= 0` disables clipping; resume keeps one base LR per param group), because it depends on them: `mup` has five param groups, and the README recommends `--optim.clip 0` for sweeps. They will drop out of this PR once #17 is merged into the base branch. With `mup`, per-role eps and weight decay also follow the config after a reload (resume and `base_model_dir` with `reload_base_optim_state`).

## Decisions from #7

- **Clipping:** Complete(d)P gives no clipping rule. Vetting runs (coordinate checks, LR-transfer sweeps) run with `--optim.clip 0`. For training runs, we will run the LR-transfer sweep with and without `optim.clip 1.0` and keep clipping if the optima match, otherwise scale the threshold with width.
- **`tau_epoch`:** optional and unset by default, CompleteP-style as above. Complete(d)P's full horizon rule (LR and WD scaled by `1/sqrt(kappa)`, plus eps/betas) is left to a separate issue.
- **Coordinate check:** out of scope here; it will follow once the `logging` branch is merged.

## Testing

`tests/test_completedp.py` (52 tests, single H100, torch 2.12):

- config: width/depth resolution, rejected setups (including explicit init stds that conflict with mup), CLI round trip with and without `mup`, width family checks
- model: role counts, init stds, zero init of query and readout (query rejected with QK-norm), residual multiplier (multiplier 0 makes each block the identity)
- optimizer: per-role groups match the table for alpha 0.5 and 1; `no_decay_embed_readout`; `tau_epoch` (at alpha 1, `lr * wd * steps = 1/tau` for every role); without `mup` the optimizer equals a plain `AdamW(model.parameters(), ...)`
- clipping: `get_clip_max_norm` gives an infinite threshold for `clip <= 0`, and an infinite threshold leaves gradients unchanged
- resume: a resumed run matches an uninterrupted one exactly, and configured per-role hparams are restored; reloading optimizer state saved with a different param group layout raises an error

```
pytest tests/test_completedp.py
```

Requires CUDA and `fla-core` (imported by `xllm.modules`).

Without `mup`, 5 training steps on this branch and on `main` give identical parameters (21/21 tensors bitwise equal), param groups and grad norms.

Sanity check of update sizes after 5 steps (change in logits / last block output), depth 2:

| width | SP dlogits | SP dh | mup dlogits | mup dh |
| --- | --- | --- | --- | --- |
| 128 | 1.44 | 0.58 | 0.69 | 0.93 |
| 512 | 2.90 | 13.1 | 0.51 | 0.79 |
| 2048 | 9.37 | 683 | 0.45 | 0.69 |

Depth 2 to 32 at width 256 with `mup` stays flat for alpha 1 (dlogits 1.21 to 1.04, dh 1.76 to 1.52) and alpha 0.5 (1.21 to 1.10, 1.76 to 1.38); SP grows (dh 2.0 to 136).

Multi-GPU (FSDP2), on 1 node x 2 GPUs and 2 nodes x 2 GPUs, all passing:

- without `mup`, 5 training steps on this branch and on `main` give bitwise identical parameters, param groups and grad norms
- with `mup` (width x4, depth x4, alpha 0.5), per-role lr/eps/wd match the table on every rank, and each role's shards add up to the full parameters
- `clip 0` leaves gradients unchanged and reports the global gradient norm
- resuming after 4 steps matches an uninterrupted run bitwise, with DCP and with torch-format optimizer state
- the optimizer layout check raises on every rank

## Notes

- The QK-norm kernel (`GroupRMSNorm`) supports at most 32 groups, so with QK-norm a width family is limited to 32 heads; use a larger fixed `head_dim` for wide models. This limit is pre-existing.
- On resume, the first step after reload still uses the checkpoint's per-group `lr`; the scheduler resets it after that step. This one-step lag is pre-existing behavior on `main`.
- Depth-scaled models need the residual multiplier folded into `wo` and `fc2` before serving with xBridges/vLLM (documented in the README; the xBridges change is pending). Width-only models export as is.
- At `alpha < 1`, the hidden roles' AdamW timescale `1 / (lr * wd * steps)` changes with depth as `m_L^(1-alpha)`, as in Complete(d)P's Table 1; it is depth-invariant at `alpha = 1` (documented in the README).
- xLLM norms parameterize the gain as `1 + weight` with `weight` initialized to 0, so weight decay pulls gains toward 1 rather than 0. This is existing behavior; the table's WD factor of 1 for norms applies as is.
