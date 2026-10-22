"""Benchmark: how long training stalls per checkpoint, synchronous vs async (NCCL-only draft).

One srun task per GPU. Usage: ckpt_bench.py <workdir> <size> [repeats]
Per mode it measures, as the max over ranks:
  stall      wall time training is blocked by the save call (sync: the whole write)
  write      async only: background write time, from submit to done
  slowdown   async only: extra time of the training steps that overlap the write
  finalize   async only: time of the finalize collective at close
plus baseline step time, checkpoint size and peak host RSS.
"""
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

import xllm.checkpointing as xc

from xllm.checkpointing import AsyncCheckpointer, SyncCheckpointer, get_checkpoint_dir
from xllm.config import ModelConf, OptimConf
from xllm.distributed import initialize_model_parallel
from xllm.models import build_model
from xllm.optim import build_optimizer, rescale_grads
from xllm.utils import clip_grad_norm_, get_model_state, get_sharded_optimizer_state

RANK, WORLD, LOCAL = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"]), int(os.environ["SLURM_LOCALID"])
VOCAB, SEQ = 32000, 2048
SIZES = {  # dim, layers, heads, ffn, head_dim (<= 32 heads: a kernel limit hit with 52 heads)
    "1b": (2048, 16, 16, 5632, 128),
    "8b": (4096, 32, 32, 14336, 128),
    "30b": (6144, 64, 24, 20480, 256),
    "70b": (8192, 80, 32, 28672, 256),
}


PHASES = {}


def timed(name, fn):
    def wrapper(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            torch.cuda.synchronize(); PHASES[name] = PHASES.get(name, 0.0) + time.perf_counter() - t
    return wrapper


xc.wait_for_folder_to_exists = timed("folder_wait", xc.wait_for_folder_to_exists)
xc._stage_state_dict = timed("stage", xc._stage_state_dict)
xc._AsyncDCPSave.plan = timed("plan", xc._AsyncDCPSave.plan)


def phases():
    out = " ".join(f"{k}={gmax(PHASES.get(k, 0.0)):.1f}s" for k in ("state_dict", "folder_wait", "stage", "plan"))
    PHASES.clear()
    return out


class Tok:
    vocab_size, bos_id, eos_id, pad_id = VOCAB, 1, 2, 0


def log(msg):
    if RANK == 0:
        print(msg, flush=True)


def gmax(x):
    t = torch.tensor([float(x)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def build(size):
    dim, layers, heads, ffn, hd = SIZES[size]
    cfg = ModelConf(arch="transformer", ddp_backend="fsdp2", init_mode="gaussian", init_std=0.02,
                    num_layers=layers, model_dim=dim, num_heads=heads, head_dim=hd, ffn_hidden_dim=ffn,
                    chunk_size=SEQ, vocab_size=VOCAB, apply_rmsnorm=True, qknorm=True, swiglu=True,
                    rope_head_dim=hd, layerwise_ckpt=size in ("30b", "70b"))
    torch.manual_seed(0)
    model = build_model(cfg, dtype="bf16", fully_sharded_size=None, fp32_reduce_scatter=True,
                        reshard_after_forward=True, forward_prefetch=False, tokenizer=Tok())
    opt, sched = build_optimizer(model, OptimConf(lr=1e-4, warmup=2, clip=1.0), 1000, "bf16")
    return model, opt, sched


def step(model, opt, sched, i):
    torch.cuda.synchronize()
    t = time.perf_counter()
    g = torch.Generator(device="cuda").manual_seed(1000 * i + RANK)
    tokens = torch.randint(3, VOCAB, (1, SEQ + 1), device="cuda", generator=g)
    loss, _, _ = model(tokens[:, :-1], multi_segments=False, targets=tokens[:, 1:])
    loss.mean().backward()
    clip_grad_norm_(fsdp_module=model, max_norm=1.0)
    rescale_grads(model, 1.0)
    opt.step()
    sched.step()
    opt.zero_grad()
    torch.cuda.synchronize()
    return time.perf_counter() - t


def states(model, opt, i):
    torch.cuda.synchronize(); t = time.perf_counter()
    try:
        return _states(model, opt, i)
    finally:
        PHASES["state_dict"] = PHASES.get("state_dict", 0.0) + time.perf_counter() - t


def _states(model, opt, i):
    return dict(model_state=get_model_state(model, full_state=False),
                optimizer_state=get_sharded_optimizer_state(model, opt, True),
                training_state={"step": i}, step=i)


def du(path):
    if RANK != 0:
        return 0.0
    out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True).stdout.split()
    return int(out[0]) / 1e9 if out else 0.0


def main():
    torch.cuda.set_device(LOCAL)
    dist.init_process_group("nccl", rank=RANK, world_size=WORLD)
    initialize_model_parallel(1, 1, 1800)
    workdir, size = Path(sys.argv[1]), sys.argv[2]
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    model, opt, sched = build(size)
    nparams = sum(p.numel() for p in model.parameters()) / 1e9  # DTensor numel is the global size
    i = 0
    for _ in range(3):
        step(model, opt, sched, i); i += 1
    base = [step(model, opt, sched, i + k) for k in range(5)]; i += 5
    base_t = gmax(sum(base) / len(base))
    log(f"BENCH size={size} params~{nparams:.1f}B gpus={WORLD} nodes={os.environ.get('SLURM_NNODES')} "
        f"baseline_step={base_t:.3f}s")
    cfg = SimpleNamespace(dcp_for_optimizer=True, fully_sharded_size=None, to_json=lambda: "{}")

    for r in range(repeats):
        # ---- synchronous ----
        d = workdir / f"sync{r}"
        ck = SyncCheckpointer(dump_dir=str(d), cfg=cfg, global_rank=RANK, world_size=WORLD)
        dist.barrier(); torch.cuda.synchronize(); t = time.perf_counter()
        ck.save_latest_checkpoint(**states(model, opt, i))
        dist.barrier(); stall = gmax(time.perf_counter() - t)
        gb = du(get_checkpoint_dir(d, i))
        log(f"RESULT size={size} mode=sync rep={r} stall={stall:.1f}s size={gb:.0f}GB "
            f"({gb / stall if stall else 0:.1f} GB/s) {phases()}")
        if RANK == 0:
            shutil.rmtree(d, ignore_errors=True)

        # ---- async ----
        d = workdir / f"async{r}"
        ck = AsyncCheckpointer(dump_dir=str(d), cfg=cfg, global_rank=RANK, world_size=WORLD)
        dist.barrier(); torch.cuda.synchronize(); t = time.perf_counter()
        ck.save_latest_checkpoint(**states(model, opt, i))
        torch.cuda.synchronize(); stall = gmax(time.perf_counter() - t)
        ph = phases()
        done_at = []
        ck._pending.future.add_done_callback(lambda f: done_at.append(time.perf_counter()))
        overlap, n = [], 0
        while True:                                  # train until every rank's write is done
            st = step(model, opt, sched, i + 1 + n); n += 1
            if not done_at:
                overlap.append(st)
            flag = torch.tensor([1.0 if done_at else 0.0], device="cuda")
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if flag.item() == 1.0:
                break
        write = gmax(done_at[0] - t)
        slow = gmax(sum(overlap) - len(overlap) * base_t) if overlap else gmax(0.0)
        f0 = time.perf_counter(); ck.close(); dist.barrier(); fin = gmax(time.perf_counter() - f0)
        i += 1 + n
        gb = du(get_checkpoint_dir(d, i - n - 1))
        rss = gmax(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
        log(f"RESULT size={size} mode=async rep={r} stall={stall:.1f}s write={write:.1f}s "
            f"overlap_steps={len(overlap)} slowdown={slow:.1f}s finalize={fin:.1f}s size={gb:.0f}GB "
            f"peak_rss_per_rank={rss:.0f}GB {ph}")
        if RANK == 0:
            shutil.rmtree(d, ignore_errors=True)
        dist.barrier()
    log("BENCH DONE")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
