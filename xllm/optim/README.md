# Optimizers

`--optim.optimizer` selects `adamw` (default) or `muon`.

## Muon on FSDP2

`xllm/optim/muon.py`. Requires `--model.ddp_backend fsdp2`, CP = 1, no MoE (experts are registered, not yet run) and
`--dcp_for_optimizer true` (the optimizer state is sharded like the parameters).

- **Parameters.** A module registers the weights Muon updates in its `__init__` with
  `register_muon`; they use `--optim.muon_lr`. Everything else (embedding, output layer,
  norms, biases, convolutions, the MoE router) uses AdamW with `--optim.lr`. Both groups
  follow the same LR schedule. AdamW uses `--optim.weight_decay`; the Muon matrices use
  `--optim.muon_weight_decay` (default: the same value). Attention, the FFN, MoE experts and
  shared experts register their projections; linear layers in the blocks that are not
  registered are listed in a warning when the optimizer is built.
- **Registration API.** Import `register_muon` from `xllm.optim.muon_spec`; model construction does not import the optimizer runtime. Expert layouts and arbitrary block grids describe internal geometry; this release supports dense training with CP=1 and any TP.
- **Blocks.** A registration names the axes of the weight's two dims as the forward
  rearranges them, which axes form the Newton-Schulz matrix, and the sizes of the inner axes:

  ```python
  register_muon(self.wo)                                     # a linear layer: one matrix
  register_muon(self.wq, layout="(h k d) m", fan_out="k d", fan_in="m", k=nq, d=self.head_dim)
  register_muon(self, "weight2", "(n s) d", fan_out="s", fan_in="d", s=self.model_dim)
  ```

  Parentheses flatten several axes into one dim, outer first. The `fan_out` / `fan_in` axes
  (innermost in their dim) are the matrix rows / columns; every other axis indexes separate
  blocks, each orthogonalized and scaled on its own: here one block per head (both softdelta
  queries of a head together) and one per expert. Each dim's outermost axis is inferred from
  the global shape and cannot be given, so a tensor-parallel local count cannot slip in.
  Putting an axis in `fan_out` / `fan_in` makes it part of the matrix (`fan_out="h d"` on `wq`:
  all heads as one matrix). A module that splits a weight over the tensor-parallel group
  declares the split dim (`self.tp_dims = {"weight": 0}`, None if not split; the parallel
  linear layers and the experts do); registering a weight without it is an error. The spec
  is kept on the module (FSDP2 replaces the parameter tensors);
  the optimizer resolves it once into boxes and logs one line per weight.
- **Tensor parallelism.** TP is one more way the weight is sharded, independent of the block
  layout. Each TP rank holds its TP-local weights (FSDP2 shards them over its own data-parallel
  group); a weight's `tp_dim` (declared by the parallel layer) says which dim TP splits. Blocks
  are defined on the whole weight and each rank's region is a box of it, so a block may sit inside
  one TP rank (whole heads), be cut by TP (the whole-matrix `wo`, FFN) or straddle a TP boundary.
  The same two all-to-alls run over the plane of the dp mesh x TP group, every block is
  orthogonalized once by its owner, and update scales use the whole weight's shape. Optimizer state
  is per TP rank, like the parameters.
- **Mesh.** `train.py` builds the FSDP mesh once (`build_fsdp_mesh`) and passes the same
  mesh to `fully_shard` and to the optimizer. Muon works over the shard dimension.
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
