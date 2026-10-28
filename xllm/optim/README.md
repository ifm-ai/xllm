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
  2. A fixed plan assigns every block to one rank (balanced by size). One all-to-all sends
     each block's pieces to its owner.
  3. The owner stacks its blocks by shape (tall blocks transposed) and runs one batched
     Newton-Schulz 5 per shape, then scales each block by its own shape
     (`--optim.muon_block_scale block`, default) or its whole matrix's shape (`matrix`):
     `sqrt(rows / cols)` for `--optim.muon_adjust_lr spectral_norm` (default),
     `0.2 * sqrt(max(rows, cols))` for `rms_norm`.
  4. A second all-to-all returns the pieces; each rank applies
     `p = p * (1 - lr * wd) - lr * update` to its shard.
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

## Complete(d)P with Muon (`--mup.*`, experimental)

Width / depth scaling relative to a base model (arXiv:2512.22382; `MuPConf` in `xllm/config.py`,
groups in `xllm/optim/mup.py`). Scale `num_heads`, `num_kv_heads` and `ffn_hidden_dim` with
`model_dim`, keep `head_dim` fixed; the hyperparameters are the base model's.

- m_N = model_dim / base_model_dim, m_L = num_layers / base_num_layers, c = m_L^-alpha.
- Init: hidden std x m_N^-1/2, readout x m_N^-1, embedding unchanged; attention `wo` and FFN
  `fc2` x c on top of the hidden std (`model.init_out_std`). Explicit init stds that disagree are rejected.
- Muon: lr x m_L^(alpha-1); `wo` / `fc2` updates x c, their decay `lr * wd` unchanged.
- AdamW per role (lr, eps, weight decay): embed (1, m_N^-1, 1), hidden_vector (m_L^(alpha-1),
  m_N^-1 m_L^-alpha, 1), final_norm (1, 1, 1), readout (m_N^-1, 1, m_N).
- Clipping counts `wo` / `fc2` gradients x c.

The depth scale is folded into `wo` / `fc2` instead of a residual multiplier: the forward is a
plain Transformer (checkpoints serve as they are), and training equals the multiplier version
(`tests/test_completedp_muon.py`, `tests/test_completedp_muon_gpu.py`). Dense Transformers only,
`init_mode gaussian`, `muon_adjust_lr spectral_norm`, `muon_block_scale block`.

```bash
--optim.optimizer muon --dcp_for_optimizer true --model.init_mode gaussian \
--mup.base_model_dim 1024 --mup.base_num_layers 8 --mup.alpha 1.0 \
--mup.base_init_hidden_std 0.02 --mup.base_init_embed_std 0.02 --mup.base_init_unembed_std 0.02
```
