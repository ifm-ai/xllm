"""Restart-all prototype: a per-node agent restarts every trainer after a failure; state comes from /dev/shm.

  python restart_all_test.py agent <workdir> <ngpus>     # one per node (srun --ntasks-per-node=1)
  python restart_all_test.py worker                      # spawned by the agent

The agent first runs an uninterrupted reference (generation "ref"), then a run where rank KILL_RANK dies at step
KILL_AT of generation 0. On any trainer exit != 0 it SIGKILLs the rest (stuck in a dead collective), waits for the
GPUs to be released, and respawns all trainers as generation g+1. Each trainer writes, after every step, its local
shards + optimizer state into a double-buffered /dev/shm file plus a small meta file written last (completion
marker). On restart, ranks agree on the newest step they all hold and resume from it.
Check: final shards bitwise equal to the reference; timings of each phase.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

N_STEPS, KILL_RANK, KILL_AT = 9, 3, 5
VOCAB, SEQ = 32000, 2048
SHM = Path("/dev/shm/xllm_ft_test")


# ------------------------------------------------------------------ agent
def agent(workdir: Path, ngpus: int):
    SHM.mkdir(parents=True, exist_ok=True)
    for f in SHM.glob("*"):
        f.unlink()
    base_port = int(os.environ["MASTER_PORT"])

    def launch(gen, mode):
        procs = []
        for r in range(ngpus):
            env = dict(os.environ, RANK=str(r), WORLD_SIZE=str(ngpus), LOCAL_RANK=str(r), MASTER_ADDR="127.0.0.1",
                       MASTER_PORT=str(base_port + (gen if isinstance(gen, int) else 50)), FT_GEN=str(gen),
                       FT_MODE=mode, FT_WORKDIR=str(workdir), CUDA_VISIBLE_DEVICES=str(r))
            procs.append(subprocess.Popen([sys.executable, __file__, "worker"], env=env))
        return procs

    def gpus_free(timeout=120):
        t = time.time()
        while time.time() - t < timeout:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout.strip()
            if not out:
                return time.time() - t
            time.sleep(0.2)
        return -1.0

    print("agent: reference run", flush=True)
    for p in launch("ref", "ref"):
        assert p.wait() == 0, "reference run failed"

    gen = 0
    procs = launch(gen, "train")
    while True:
        codes = [p.poll() for p in procs]
        if all(c == 0 for c in codes):
            break
        bad = [r for r, c in enumerate(codes) if c not in (None, 0)]
        if bad:
            t_fail = time.time()
            print(f"agent: rank(s) {bad} exited (codes {[codes[r] for r in bad]}) in gen {gen}; killing all", flush=True)
            for p in procs:
                if p.poll() is None:
                    p.send_signal(signal.SIGKILL)
            for p in procs:
                p.wait()
            t_killed = time.time() - t_fail
            t_free = gpus_free()
            print(f"agent: all trainers dead after {t_killed:.2f}s, GPUs free after {t_free:.2f}s more", flush=True)
            (workdir / "t_fail").write_text(str(t_fail))
            gen += 1
            procs = launch(gen, "train")
        time.sleep(0.05)
    print(f"agent: run finished after {gen} restart(s)", flush=True)


# ------------------------------------------------------------------ worker
def worker():
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    from xllm.config import ModelConf, OptimConf
    from xllm.distributed import initialize_model_parallel
    from xllm.models import build_model
    from xllm.optim import build_optimizer, rescale_grads
    from xllm.utils import clip_grad_norm_

    rank, world, gen, mode = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), os.environ["FT_GEN"], os.environ["FT_MODE"]
    workdir = Path(os.environ["FT_WORKDIR"])
    t_start = time.time()
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", 0))
    MP = int(os.environ.get("FT_MP", "1")); MOE = int(os.environ.get("FT_MOE", "0"))
    initialize_model_parallel(MP, 1, 300)
    t_init = time.time() - t_start

    class Tok:
        vocab_size, bos_id, eos_id, pad_id = VOCAB, 1, 2, 0

    t0 = time.time()
    kw = dict(arch="transformer", ddp_backend="fsdp2", init_mode="gaussian", init_std=0.02, num_layers=16,
              model_dim=2048, num_heads=16, head_dim=128, ffn_hidden_dim=5632, chunk_size=SEQ, vocab_size=VOCAB,
              apply_rmsnorm=True, qknorm=True, swiglu=True, rope_head_dim=128)
    if MOE:
        kw.update(num_experts=8, num_activated_experts=2, expert_inter_dim=1408)
    cfg = ModelConf(**kw)
    torch.manual_seed(1)
    model = build_model(cfg, dtype="bf16", fully_sharded_size=None, fp32_reduce_scatter=True,
                        reshard_after_forward=True, forward_prefetch=False, tokenizer=Tok())
    opt, sched = build_optimizer(model, OptimConf(lr=1e-4, warmup=2, clip=1.0), 100, "bf16")
    loc = lambda t: t.to_local() if isinstance(t, DTensor) else t
    params = list(model.parameters())
    bufs = [b for b in model.buffers() if b.is_floating_point()]
    sizes = [loc(p).numel() for p in params]
    total = 3 * sum(sizes) + sum(loc(b).numel() for b in bufs)
    torch.cuda.synchronize()
    t_build = time.time() - t0

    def step(i):
        g = torch.Generator(device="cuda").manual_seed(1000 * i + rank // MP)   # same batch inside a TP group
        tokens = torch.randint(3, VOCAB, (1, SEQ + 1), device="cuda", generator=g)
        loss, _, _ = model(tokens[:, :-1], multi_segments=False, targets=tokens[:, 1:])
        loss.mean().backward()
        clip_grad_norm_(fsdp_module=model, max_norm=1.0)
        rescale_grads(model, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()

    def flat():
        parts = []
        for p in params:
            st = opt.state[p]
            parts += [loc(p).reshape(-1), loc(st["exp_avg"]).reshape(-1), loc(st["exp_avg_sq"]).reshape(-1)]
        parts += [loc(b).reshape(-1) for b in bufs]                 # e.g. MoE router bias: state outside the optimizer
        return torch.cat([t.detach().float() for t in parts])

    # double-buffered /dev/shm slots, pinned host staging buffer allocated once
    host = torch.empty(total, dtype=torch.float32, pin_memory=True)
    slots = [SHM / f"rank{rank}_slot{s}.bin" for s in (0, 1)]
    metas = [SHM / f"rank{rank}_slot{s}.json" for s in (0, 1)]
    mm = []
    for s in slots:
        if not s.exists() or s.stat().st_size != total * 4:
            with open(s, "wb") as f:
                f.truncate(total * 4)
        mm.append(torch.from_file(str(s), shared=True, size=total, dtype=torch.float32))

    def save_replica(i):
        s = i % 2
        metas[s].unlink(missing_ok=True)                      # invalidate the slot before overwriting it
        host.copy_(flat(), non_blocking=True)
        torch.cuda.synchronize()
        mm[s].copy_(host)
        meta = {"step": i, "opt_step": float(next(iter(opt.state.values()))["step"]),
                "groups": [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups],
                "sched": sched.state_dict()}
        tmp = metas[s].with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, default=float))
        os.replace(tmp, metas[s])                             # written last: the slot is complete

    @torch.no_grad()
    def load_replica(s):
        meta = json.loads(metas[s].read_text())
        buf = mm[s].cuda()
        off = 0
        for p, n in zip(params, sizes):
            loc(p).copy_(buf[off:off + n].view_as(loc(p))); off += n
            st = {}
            for k in ("exp_avg", "exp_avg_sq"):
                t = buf[off:off + n].view_as(loc(p)).clone().to(loc(p).dtype); off += n
                st[k] = DTensor.from_local(t, p.device_mesh, p.placements, run_check=False, shape=p.shape,
                                           stride=p.stride()) if isinstance(p, DTensor) else t
            st["step"] = torch.tensor(meta["opt_step"])
            opt.state[p] = st
        for b in bufs:
            n = loc(b).numel(); loc(b).copy_(buf[off:off + n].view_as(loc(b))); off += n
        for g, saved in zip(opt.param_groups, meta["groups"]):
            g.update(saved)
        sched.load_state_dict(meta["sched"])
        return meta["step"]

    start, t_load = 0, 0.0
    if mode == "train" and gen != "0":
        t0 = time.time()
        have = {}
        for s in (0, 1):
            if metas[s].exists():
                have[json.loads(metas[s].read_text())["step"]] = s
        # newest step every rank holds: max over my steps, then min across ranks
        mine = torch.tensor([max(have) if have else -1], device="cuda")
        dist.all_reduce(mine, op=dist.ReduceOp.MIN)
        common = int(mine.item())
        assert common in have, (rank, common, have)
        last = load_replica(have[common])
        start = last + 1
        torch.cuda.synchronize()
        t_load = time.time() - t0

    t_rep, n_rep = 0.0, 0
    t_first = None
    for i in range(start, N_STEPS):
        if mode == "train" and gen == "0" and rank == KILL_RANK and i == KILL_AT:
            os._exit(1)
        t0 = time.time()
        step(i)
        torch.cuda.synchronize()
        if t_first is None:
            t_first = time.time() - t0
        if mode == "train":
            t1 = time.time()
            save_replica(i)
            t_rep += time.time() - t1; n_rep += 1

    final = flat().cpu()
    if mode == "ref":
        torch.save(final, workdir / f"ref_{rank}.pt")
    else:
        ok = torch.equal(final, torch.load(workdir / f"ref_{rank}.pt"))
        flag = torch.tensor([1 if ok else 0], device="cuda")
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        tm = torch.tensor([t_init, t_build, t_load, t_first or 0.0, t_rep / max(n_rep, 1)], device="cuda")
        dist.all_reduce(tm, op=dist.ReduceOp.MAX)
        if rank == 0:
            since = time.time() - float((workdir / "t_fail").read_text()) if (workdir / "t_fail").exists() else 0.0
            a, b, c, d, e = tm.tolist()
            print(f"RESULT mp={MP} moe={MOE} n_params={len(params)} n_float_buffers={len(bufs)} gen={gen} resumed_at_step={start} init_pg={a:.2f}s build={b:.2f}s load_from_shm={c:.2f}s "
                  f"first_step={d:.2f}s replica_per_step={e:.2f}s shard_GB_per_rank={total * 4 / 1e9:.2f} "
                  f"bitwise_equal={'yes' if flag.item() else 'NO'}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    if sys.argv[1] == "agent":
        agent(Path(sys.argv[2]), int(sys.argv[3]))
    else:
        worker()
