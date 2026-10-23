# Optimizers

`--optim.optimizer` selects `adamw` (default) or `muon`.

## Muon on FSDP2

`xllm/optim/muon.py`. Requires `--model.ddp_backend fsdp2`, TP = CP = 1, no MoE (experts are registered, not yet run) and
`--dcp_for_optimizer true` (the optimizer state is sharded like the parameters).

- **Parameters.** A module registers the weights Muon updates in its `__init__` with
  `register_muon`; they use `--optim.muon_lr`. Everything else (embedding, output layer,
  norms, biases, convolutions, the MoE router) uses AdamW with `--optim.lr`. Both groups
  follow the same LR schedule. AdamW uses `--optim.weight_decay`; the Muon matrices use
  `--optim.muon_weight_decay` (default: the same value). Attention, the FFN, MoE experts and
  shared experts register their projections; linear layers in the blocks that are not
  registered are listed in a warning when the optimizer is built.
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
- **Resume.** The config wins over the checkpoint: the scheduler keeps its position, but its
  base LRs (`--optim.muon_lr`, `--optim.lr`) and the groups' weight decay and epsilon come from
  the config, so a run can be continued with new values. Momentum, Nesterov, betas and the
  update scale are always taken from the config.

```bash
--optim.optimizer muon --optim.muon_lr 0.01 --optim.lr 2e-3 \
--dcp_for_optimizer true
```
