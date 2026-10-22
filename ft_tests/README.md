# Fault-tolerance tests (hot swap and checkpointing)

Scripts used to measure the xLLM hot-swap design. Each prints machine-readable lines (`RESULT`, `STEPTIME`,
`PGSCALE`, `STORELOAD`, `DISKRESTORE`, `BENCH`, `PROCSTART`) that are the results to report.

The `*.sbatch` files were written for one SLURM cluster and need adapting on another: `--account`, `--partition`,
`--qos`, the paths `J=` (this folder) and `B=` (the xLLM checkout), the container (`--container-image` for
pyxis/enroot, or drop `--container-*` and use a Python env with torch 2.12+ and this checkout on `PYTHONPATH`).
Keep `--propagate=NONE` on clusters where SLURM memlock limits break multi-node NCCL.

| Script | What it measures | Typical size |
| --- | --- | --- |
| `async_multi_gpu.py` | async checkpoint resume is bitwise equal (DCP and torch formats) | 1-2 nodes |
| `ckpt_bench.py` | sync vs async checkpoint stall, 1B-70B | 1-4 nodes |
| `swap_probe.py` | destroy and re-create all process groups in one process, rebuild, restore; bitwise | 1 node |
| `swap_kill_test.py` | in-process recovery after a real rank death (expected to FAIL: survivors stay hung) | 1 node, 5 GPUs |
| `restart_all_test.py` / `restart_all_tp_test.py` | per-node agent kills and respawns all trainers; state from /dev/shm; FSDP, TP (FT_MP), MoE (FT_MOE) | 1 node, 4 GPUs |
| `node_death_test.py` | a whole node dies; an idle spare node takes over with state from peer copies | 3 nodes |
| `scale_test.py` | node death at 8B (FT_SIZE=8b) with per-step copies | 3 nodes x 8 GPUs |
| `cadence_test.py`, `async_copy_test.py`, `own_copy_test.py` | training-time cost of the copies vs cadence (FT_K, FT_SKIP) | 3 nodes x 8 GPUs |
| `pg_scale_bench.py` | process-group init, xLLM group creation, first all-reduce, destroy, TCPStore load vs node count | 1-16+ nodes |
| `disk_restore_bench.py` | restore time from a DCP checkpoint on the shared filesystem at 8B | 2 nodes |
| `proc_start.sbatch` | trainer process start (imports + CUDA init) for 8 trainers on a node | 1 node |
