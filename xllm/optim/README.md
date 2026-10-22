# Optimizers

`--optim.optimizer` selects `adamw` (default) or `muon`.

## Muon on FSDP2

`xllm/optim/muon.py`. Requires `--model.ddp_backend fsdp2`, TP = CP = 1, no MoE and
`--dcp_for_optimizer true` (the optimizer state is sharded like the parameters).

- **Parameters.** A model marks the linear layers whose weights Muon updates with a
  `muon` attribute (`layer.muon = n`); they use `--optim.muon_lr`. Everything else
  (embedding, output layer, norms, biases, convolutions) uses AdamW with `--optim.lr`.
  Both groups follow the same LR schedule. AdamW uses `--optim.weight_decay`; the Muon
  matrices use `--optim.muon_weight_decay` (default: the same value). The attention
  layers and the FFN mark their projections.
- **Blocks.** `n` is the number of equal row blocks the matrix is cut into, each
  orthogonalized on its own: 1 = whole matrix, the number of heads = one block per head,
  in between = blocks of adjacent heads. Attention marks `wq` / `wr` with its heads and
  `wk` / `wv` with its KV heads; `wo` and the FFN are 1.
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
