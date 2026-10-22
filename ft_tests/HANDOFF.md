# Handoff: xLLM hot-swap work on cluster m4

## Goal
xLLM training that never stops: on a process, GPU or node failure, a per-node agent kills and restarts all trainers inside the
same allocation (no requeue), state comes back from host-memory copies (own /dev/shm + a peer node), and a dead node is replaced
by an idle node requested on demand. Async disk checkpoints with bounded retention are the safety net. Must work for any model
size (up to trillion-parameter, dense and MoE) and any cluster size, NCCL only (no gloo), mostly NVIDIA clusters, and must not
break xLLM (everything off by default, bitwise-unchanged when off).

## Decided (with evidence on the first cluster, H100, torch 2.12)
- Restart-all recovery works bitwise: process death 1B FSDP / +TP / +MoE 6.4 / 8.2 / 10.7 s; node death 1B 12.2 s, 8B on 16 GPUs
  32.6 s; plus trainer process start 7.4-7.8 s. In-process recovery (abort a hung collective) failed in 3 configurations: rejected.
- Copy cost at 8B (6 GB/rank): full copy own+peer 3.0-3.3 s synchronous, 2.3-2.6 s asynchronous, own only 1.4-1.5 s of training
  time per copy. So copies are spaced: k* = sqrt(2*C*50700*3600/(W*t_step^2)), capped at the checkpoint interval.
- Restore at 8B/16 GPUs: in-memory 3.4 s vs DCP from Lustre 47-53 s.
- Heartbeats must be counters judged on the local clock (node clocks differed by ~100 s).

## What to run on m4 (see README.md in this folder)
1. Describe m4 (scheduler, account/partition/QOS, GPUs/node, network, free nodes, container runtime, /dev/shm, RAM).
2. restart_all.sbatch (1 node) as a sanity check, proc_start.sbatch.
3. pg_scale.sbatch at 1, 2, 4, 8, 16+ nodes (most important: calibrates projections to 128-1024 nodes).
4. scale.sbatch (3 x 8 GPUs, FT_SIZE=8b), disk_restore.sbatch (2 nodes).
Later, if nodes allow: 70B multi-node, xLLM's real data loader across a swap, two close failures, spare from the queue.

## Rules
- Report exact result lines (RESULT, STEPTIME, PGSCALE, STORELOAD, DISKRESTORE, PROCSTART) with job ids, to the session "Xllm"
  via SendMessage, or as a text file on branch ahmad.chamma/ft-results-m4 in ifm-ai/xllm.
- Commits: one change each, plain messages, no AI attribution lines. No other pushes, no PRs.
- Cancel only jobs you submitted, by job id; never user-wide scancel.
- Keep replies short; record every number with its job id; never present an estimate as a measurement.
