"""Scaling micro-benchmark for the recovery path: how long do process-group (re)creation and xLLM's parallel-group
setup take as the world grows? One srun task per GPU. Measures, max over ranks:
  init      init_process_group on a TCPStore generation (eager NCCL init via device_id)
  groups    xLLM initialize_model_parallel(mp, 1) (its new_group calls), as in xllm/distributed/initialize.py
  groups_ls the same groups with new_group(use_local_synchronization=True)
  first     first all_reduce on the world group (communicator warm-up)
  destroy   destroy_model_parallel + destroy_process_group
Repeated for 2 generations to also time a re-rendezvous. Also measures TCPStore heartbeat round-trip under load:
every rank sets/gets a key once per 0.2 s and logs the p50/p99 latency.
"""
import os
import statistics
import sys
import threading
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from xllm.distributed import initialize_model_parallel
from xllm.distributed.initialize import destroy_model_parallel

RANK, WORLD, LOCAL = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"]), int(os.environ["SLURM_LOCALID"])
MP = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def gmax(x):
    t = torch.tensor([float(x)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def main():
    torch.cuda.set_device(LOCAL)
    store = dist.TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]), WORLD, is_master=RANK == 0,
                          timeout=timedelta(seconds=900))
    orig_new_group = dist.new_group
    for gen, local_sync in ((0, False), (1, False), (2, True)):
        t = time.perf_counter()
        dist.init_process_group("nccl", store=dist.PrefixStore(f"g{gen}", store), rank=RANK, world_size=WORLD,
                                timeout=timedelta(seconds=600), device_id=torch.device("cuda", LOCAL))
        t_init = time.perf_counter() - t
        if local_sync:
            dist.new_group = lambda *a, **k: orig_new_group(*a, use_local_synchronization=True, **k)
        t = time.perf_counter()
        initialize_model_parallel(MP, 1, 600)
        torch.cuda.synchronize()
        t_groups = time.perf_counter() - t
        dist.new_group = orig_new_group
        t = time.perf_counter()
        x = torch.ones(1, device="cuda")
        dist.all_reduce(x)
        torch.cuda.synchronize()
        t_first = time.perf_counter() - t
        vals = [gmax(v) for v in (t_init, t_groups, t_first)]
        t = time.perf_counter()
        destroy_model_parallel()
        dist.destroy_process_group()
        t_destroy = time.perf_counter() - t
        # report through a fresh tiny group would need NCCL again; use the store instead
        store.set(f"destroy_{gen}_{RANK}", str(t_destroy))
        if RANK == 0:
            ds = [float(store.get(f"destroy_{gen}_{r}").decode()) for r in range(WORLD)]
            print(f"PGSCALE world={WORLD} nodes={os.environ.get('SLURM_NNODES')} mp={MP} gen={gen} "
                  f"local_sync={local_sync} init={vals[0]:.2f}s groups={vals[1]:.2f}s first_allreduce={vals[2]:.2f}s "
                  f"destroy={max(ds):.2f}s", flush=True)

    # TCPStore load: every rank heartbeats (set + get of a peer's key) every 0.2 s for 20 s
    lat, stop = [], time.time() + 20
    while time.time() < stop:
        t = time.perf_counter()
        store.set(f"hb_{RANK}", str(len(lat)))
        store.get(f"hb_{(RANK + 1) % WORLD}")
        lat.append(time.perf_counter() - t)
        time.sleep(0.2)
    lat.sort()
    store.set(f"lat_{RANK}", f"{statistics.median(lat)},{lat[int(0.99 * (len(lat) - 1))]}")
    if RANK == 0:
        pairs = [tuple(map(float, store.get(f"lat_{r}").decode().split(","))) for r in range(WORLD)]
        print(f"STORELOAD world={WORLD} clients={WORLD} rate={2 * WORLD / 0.2:.0f} ops/s "
              f"p50_max={max(p[0] for p in pairs) * 1e3:.1f}ms p99_max={max(p[1] for p in pairs) * 1e3:.1f}ms", flush=True)


if __name__ == "__main__":
    main()
