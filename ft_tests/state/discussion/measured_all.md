# All moderator measurements (GPU runs on this cluster, H100, torch 2.12, xLLM main) - the only numbers you may treat as measured
Restart-all recovery (process start excluded): 1B FSDP 4 GPUs 6.4 s; 1B FSDP+TP2 8.2 s; 1B FSDP+TP2+MoE(8 experts) 10.7 s;
  node death 1B 3x2 GPUs 12.2 s (detect 6.0 = deadline); node death 8B 3x8 GPUs 32.6 s (detect 10.0, kill 10.54, PG init 5.41,
  build 0.58, restore 3.42, first step 2.68). All resumed at the failed step, bitwise equal. Jobs 33027/33030/33032/33035/33036.
Trainer process start (spawn + imports from Lustre + CUDA init), 8 in parallel on a node: 7.4-7.8 s (job 33052).
In-process recovery after a real rank death: failed in 3 NCCL configurations (survivors never leave the hung collective).
Copy cost at 8B (2 nodes x 8 GPUs, 6.04 GB/rank, baseline 0.70 s/step): full copy own+peer: synchronous 3.0-3.3 s, asynchronous
  2.3-2.6 s of training time per copy (async hides ~20-25%); own copy only 1.4-1.5 s per copy.
Disk restore at 8B / 16 GPUs from a Lustre DCP checkpoint: 47-53 s (save 24.6 s) vs in-memory restore 3.4 s (job 33053).
Async DCP checkpoint stall (1B-70B, excl. a 10 s folder-wait bug): 1-22 s; sync 4-64 s (jobs 33015-33018).
PG scaling (1-2 nodes): re-init 0.4 s, xLLM groups 0.5 s, first all-reduce 0.65-0.87 s, destroy 1.1 s; TCPStore heartbeat p99 <= 0.3 ms.
  4-16 node points are queued (blocked by cluster reservations); keep that term as a band until they land.
Node facts: /dev/shm = 504 GB per node; 8 H100 80 GB per node; ~1 TB RAM per node. Clock skew between nodes ~100 s observed.
GPU memory per rank, xLLM FSDP2, 8B on 16 GPUs (0.503B local params; job 33054): after build 2.0 GB (4.0 B/param);
  between steps after zero_grad 6.1 GB (12.2 B/param); during a step with grads 8.1 GB (16.2 B/param); peak with activations
  (seq 2048, no activation checkpointing) 47.8 GB (95 B/param). Verifier: +4 B/param foreach AdamW temporary inside optimizer.step().
Second cluster m4 (described, not yet measured): 8x H100 80 GB per node, ~2 TB RAM, /dev/shm 1008 GB, IB 400 Gb/s.
