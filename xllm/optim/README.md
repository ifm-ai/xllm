# Optimizers

`--optim.optimizer` selects `adamw` (default) or `muon`.

## Muon on FSDP2

`xllm/optim/muon.py`. Requires `--model.ddp_backend fsdp2`, TP = CP = 1, no MoE and
`--dcp_for_optimizer true` (the optimizer state is sharded like the parameters).

- **Parameters.** The weights of `ColumnParallelLinear` / `RowParallelLinear` inside
  `model.layers` are updated by Muon with `--optim.muon_lr`. Everything else (embedding,
  output layer, norms, biases, convolutions) uses AdamW with `--optim.lr`. Both groups
  follow the same LR schedule and `--optim.weight_decay`.
- **Blocks.** Each Muon matrix is a list of blocks that are orthogonalized independently.
  By default a matrix is one block. With `--optim.muon_per_head true` the attention
  projections are split into heads: rows of `wq` / `wk` (`head_dim`) and `wv` / `wr`
  (`v_head_dim`), columns of `wo` (`v_head_dim`).
- **Mesh.** `train.py` builds the FSDP mesh once (`build_fsdp_mesh`) and passes the same
  mesh to `fully_shard` and to the optimizer. Muon works over the shard dimension.
- **Step.**
  1. Each rank updates the momentum of its own shard (`m = mu * m + g`, optional Nesterov)
     and casts the update direction to bf16.
  2. A fixed plan assigns every block to one rank (balanced by size). One all-to-all sends
     each block's pieces to its owner.
  3. The owner stacks its blocks by shape (tall blocks transposed) and runs one batched
     Newton-Schulz 5 per shape, then scales each block by its own shape:
     `sqrt(rows / cols)` for `--optim.muon_adjust_lr spectral_norm` (default),
     `0.2 * sqrt(max(rows, cols))` for `rms_norm`.
  4. A second all-to-all returns the pieces; each rank applies
     `p = p * (1 - lr * wd) - lr * update` to its shard.
- **Checkpoints.** Momentum and AdamW moments are DTensors like the parameters, so DCP
  reshards them if the mesh changes; the block plan is rebuilt from the global shapes.

```bash
--optim.optimizer muon --optim.muon_lr 0.01 --optim.lr 2e-3 --optim.muon_per_head true \
--dcp_for_optimizer true
```
