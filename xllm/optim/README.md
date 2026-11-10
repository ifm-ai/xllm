# Optimizers

`--optim.optimizer` selects `adamw` (default) or `muon`.

## Muon on FSDP2

`xllm/optim/muon.py`. Requires `--model.ddp_backend fsdp2`, no MoE (experts are registered, not yet run) and
`--dcp_for_optimizer true` (the optimizer state is sharded like the parameters).

- **Parameters.** A module registers the weights Muon updates in its `__init__` with
  `register_muon`; they use `--optim.muon_lr`. Everything else (embedding, output layer,
  norms, biases, convolutions, the MoE router) uses AdamW with `--optim.lr`. Both groups
  follow the same LR schedule. AdamW uses `--optim.weight_decay`; the Muon matrices use
  `--optim.muon_weight_decay` (default: the same value). Attention, the FFN, MoE experts and
  shared experts register their projections; linear layers in the blocks that are not
  registered are listed in a warning when the optimizer is built.
- **Registration API.** Import `register_muon` from `xllm.optim.muon_spec`; model construction does not import the optimizer runtime. Expert layouts and arbitrary block grids describe internal geometry; this release supports dense training with any TP and CP.
- **Blocks.** A registration names the axes of the weight's two dims as the forward
  rearranges them, which axes form the Newton-Schulz matrix, and the sizes of the inner axes:

  ```python
  register_muon(self.wo, tp_dim=1)                           # a linear layer: one matrix
  register_muon(self.wq, layout="(h k d) m", fan_out="k d", fan_in="m", k=nq, d=self.head_dim, tp_dim=0)
  register_muon(self, "weight2", "(n s) d", fan_out="s", fan_in="d", s=self.model_dim, tp_dim=0)
  ```

  Parentheses flatten several axes into one dim, outer first. The `fan_out` / `fan_in` axes
  (innermost in their dim) are the matrix rows / columns; every other axis indexes separate
  blocks, each orthogonalized and scaled on its own: here one block per head (both softdelta
  queries of a head together) and one per expert. Each dim's outermost axis is inferred from
  the global shape and cannot be given, so a tensor-parallel local count cannot slip in.
  Putting an axis in `fan_out` / `fan_in` makes it part of the matrix (`fan_out="h d"` on `wq`:
  all heads as one matrix). `tp_dim` is the dim tensor parallelism splits (0 for a
  column-parallel layer, 1 for a row-parallel one, 0 for experts, None if not split); it is
  required when TP > 1 and ignored otherwise. The spec is kept on the module (FSDP2 replaces the parameter tensors);
  the optimizer resolves it once into boxes and logs one line per weight.
- **Tensor parallelism.** TP is one more way the weight is sharded, independent of the block
  layout. Each TP rank holds its TP-local weights (FSDP2 shards them over its own data-parallel
  group); a weight's `tp_dim`, passed to `register_muon` (0 for column-parallel layers, 1 for row-parallel ones,
  required when TP > 1), says which dim TP splits. Blocks
  are defined on the whole weight and each rank's region is a box of it, so a block may sit inside
  one TP rank (whole heads), be cut by TP (the whole-matrix `wo`, FFN) or straddle a TP boundary.
  The same two all-to-alls run over the plane of the dp mesh x TP group, every block is
  orthogonalized once by its owner, and update scales use the whole weight's shape. Optimizer state
  is per TP rank, like the parameters.
- **Context parallelism and replicas.** Weights are replicated over context-parallel ranks (they sit on the
  replicate dim of the FSDP mesh, with any pure data-parallel replicas), and FSDP2 all-reduces the gradients
  over that dim. Each replica group therefore runs the same Muon step on the same gradients over its own
  shard group (`mesh["dp"]`); the optimizer needs nothing else for CP.
- **Mesh.** The optimizer takes the mesh from the FSDP2 parameters themselves
  (`param.device_mesh`) and works over its shard dimension.
- **Step.**
  1. Each rank updates the momentum of its own shard (`m = mu * m + g`; Nesterov by default, `--optim.muon_nesterov`)
     and casts the update direction to bf16.
  2. A fixed plan assigns every block to one rank (the one storing most of it, then the least loaded). One
     all-to-all sends each block's pieces to its owner; meanwhile the blocks that lie whole in their owner's
     shard (heads, experts) are orthogonalized without communication, one buffer per block shape at a time.
  3. The owner stacks its blocks by shape and runs one batched
     Newton-Schulz per shape (Polar Express coefficients), then scales each block by its own shape
     (`--optim.muon_block_scale block`, default) or its whole matrix's shape (`matrix`):
     `sqrt(rows / cols)` for `--optim.muon_adjust_lr spectral_norm` (default),
     `0.2 * sqrt(max(rows, cols))` for `rms_norm`.
  4. A second all-to-all returns the pieces; each rank applies
     `p = p - lr * (wd * p + update)` to its shard.
- **Checkpoints.** Momentum and AdamW moments are DTensors like the parameters, so DCP
  reshards them if the mesh changes; the block plan is rebuilt from the global shapes.
- **Resume.** `state_dict()` includes momentum, moments and algorithm hyperparameters in
  the parameter groups. `load_state_dict()` restores checkpoint values. The training
  entry point explicitly reapplies the configured hyperparameters via
  `restore_optimizer_config`; the scheduler keeps its position and uses configured base
  LRs, including on the first resumed step. Loading a base model with optimizer state
  uses the new schedule's initial LR. Standalone AdamW retains upstream's resume policy.
- **Missing gradients.** `grad=None` skips momentum, parameter updates and weight decay;
  an actual zero gradient still updates state. The fixed collective sequence is preserved.

```bash
--optim.optimizer muon --optim.muon_lr 0.01 --optim.lr 2e-3 \
--dcp_for_optimizer true
```

## Parameter roles (`--roles.*`)

A module declares the role of each parameter where it creates it, `register_role(module, name, role)`
(`xllm/optim/roles.py`; the tag lives on the module, so it survives FSDP2). The Transformer's and
gekko's roles: `embed`, `hidden_matrix` (the Muon matrices in the blocks but `residual_out`),
`residual_out` (attention / gated delta attention `wo`, FFN `fc2`: the residual branches' outputs),
`hidden_vector` (norm gains, biases and gekko's causal-conv kernels in the blocks), `final_norm`,
`readout`. A parameter no module tags is in `other` (MoE layers' routers and experts).

Each role (`RolesConf` in `xllm/config.py`) has, unset by default (unset changes nothing):
- `lr`, `eps`, `weight_decay`: factors on the configured values. AdamW steps one param group per
  role; Muon scales each matrix's lr and weight decay (`lr_scales`, `wd_scales`), its decay
  `lr * weight_decay` taking both factors, as AdamW's. `eps` is AdamW's only.
- `init_std`: a truncated normal init (at 3 std; 0: zeros) as the model is built (`init_roles`).
- `clip_scale`: a factor on the role's gradients in the clipping norm; every gradient is still
  clipped by the same factor.

Roles need `--optim.optimizer muon`, and `init_std` needs `--model_parallel_size 1` (a
tensor-parallel shard would draw the same values on every rank); values are checked with the config.
- **Param groups.** Only a role `lr`, `eps` or `weight_decay` other than 1 makes per-role groups;
  `init_std` and `clip_scale` alone keep Muon's two. Checkpoints record their groups, and a resume
  (or `--reload_base_optim_state`) into other groups fails before loading: resume with the roles a
  checkpoint was saved with, or load its weights only.
- **Logs.** `lr` is Muon's; then `adamw_lr`, or with role groups one `<role>_lr` per role (a Muon
  role's: `lr` x its factor). `gnorm` / `g_norm` are the gradient norm; with clip scales,
  `gnorm_clip` / `g_norm_clip` are the norm clipping compares.

E.g. `--roles.readout.lr 0.5 --roles.readout.weight_decay 0`.

## Complete(d)P with Muon (`--mup.*`, experimental)

Width / depth transfer of the hyperparameters of a base model (arXiv:2512.22382; `MuPConf` in
`xllm/config.py`), as a preset of the roles. Scale `num_heads`, `num_kv_heads` and `ffn_hidden_dim`
with `model_dim` and keep `head_dim` fixed (`--mup.base_head_dim` checks it).
m_N = model_dim / base_model_dim, m_L = num_layers / base_num_layers, c = m_L^-alpha,
sigma_h = base_init_hidden_std m_N^-1/2:

| role            | lr              | eps                | weight_decay | init_std                    | clip_scale |
|-----------------|-----------------|--------------------|--------------|-----------------------------|------------|
| `embed`         | 1               | m_N^-1             | 1            | base_init_embed_std         | 1          |
| `hidden_matrix` | m_L^(alpha-1)   | 1                  | 1            | sigma_h                     | 1          |
| `residual_out`  | m_L^(alpha-1) c | 1                  | 1/c          | sigma_h c                   | c          |
| `hidden_vector` | m_L^(alpha-1)   | m_N^-1 m_L^-alpha  | 1            | -                           | 1          |
| `final_norm`    | 1               | 1                  | 1            | -                           | 1          |
| `readout`       | m_N^-1          | 1                  | m_N          | base_init_unembed_std / m_N | 1          |

Every parameter needs one of these roles (none in `other`), and a `--roles` value set by hand must
equal the table's. Where this departs from the paper, whose rules are for AdamW:
- **Hidden matrices** get no width factor: the paper's lr x m_N^-1 and weight decay x m_N (their
  product unchanged) are left to Muon's spectral-norm update scale, which already transfers across
  widths. This is our choice for Muon, not the paper's.
- **The depth scale c** is folded into `residual_out` instead of a residual multiplier: its init
  and lr x c, weight decay x 1/c (lr * weight decay unchanged) and gradients x c in the clipping
  norm (`clip_scale`; the paper does not discuss clipping). Training equals the multiplier version
  exactly, and the forward is a plain Transformer, so checkpoints serve as they are.
- **Batch size and training duration** (the paper's m_B, m_D) are not covered: the hyperparameters
  transfer across width and depth only.

Supported: dense Transformers (no MoE, value heads or `scale_emb`; a plain residual add),
`init_mode gaussian` with the roles' init stds (no `model.init_std`, `init_embed_std`,
`init_logits_std`), `muon_adjust_lr spectral_norm`, `muon_block_scale block`, `model_parallel_size 1`.

```bash
--optim.optimizer muon --dcp_for_optimizer true --model.init_mode gaussian \
--mup.base_model_dim 1024 --mup.base_num_layers 8 --mup.alpha 1.0 \
--mup.base_init_hidden_std 0.02 --mup.base_init_embed_std 0.02 --mup.base_init_unembed_std 0.02
```
