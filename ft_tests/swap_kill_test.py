"""Real-failure hot-swap test on one node: 4 training ranks + 1 parked spare (5 srun tasks, 5 GPUs).

Every step each rank keeps a host copy of its own shard and sends a copy to its peer ((r+1) % 4) over NCCL.
At step KILL_AT, rank 3 dies (os._exit) before its forward pass. Survivors detect the silence through heartbeats
on a TCPStore that outlives the process groups, abort the hung NCCL group, destroy all groups and re-rendezvous on
a new store generation together with the spare, which takes logical rank 3. The spare gets rank 3's state from
rank 0 (its peer) over NCCL, survivors restore their own copy, everyone rebuilds the model and continues.
Check: final shards equal an uninterrupted 4-rank run bitwise; log each phase's time.
"""
import os
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from xllm.config import ModelConf, OptimConf
from xllm.distributed import initialize_model_parallel
from xllm.distributed.initialize import destroy_model_parallel
from xllm.models import build_model
from xllm.optim import build_optimizer, rescale_grads
from xllm.utils import clip_grad_norm_

TASK, LOCAL = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_LOCALID"])
N, SPARE, DEAD, KILL_AT, STEPS = 4, 4, 3, 5, 9
VOCAB, SEQ = 32000, 2048
WORK = Path(sys.argv[1])
HB_DEADLINE = 15.0  # seconds without progress before survivors declare a rank dead


class Tok:
    vocab_size, bos_id, eos_id, pad_id = VOCAB, 1, 2, 0


def log(msg, rank=0, me=None):
    if me == rank:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build(seed):
    cfg = ModelConf(arch="transformer", ddp_backend="fsdp2", init_mode="gaussian", init_std=0.02, num_layers=16,
                    model_dim=2048, num_heads=16, head_dim=128, ffn_hidden_dim=5632, chunk_size=SEQ, vocab_size=VOCAB,
                    apply_rmsnorm=True, qknorm=True, swiglu=True, rope_head_dim=128)
    torch.manual_seed(seed)
    model = build_model(cfg, dtype="bf16", fully_sharded_size=None, fp32_reduce_scatter=True,
                        reshard_after_forward=True, forward_prefetch=False, tokenizer=Tok())
    opt, sched = build_optimizer(model, OptimConf(lr=1e-4, warmup=2, clip=1.0), 100, "bf16")
    return model, opt, sched


def one_step(model, opt, sched, i, rank):
    g = torch.Generator(device="cuda").manual_seed(1000 * i + rank)       # data keyed on the logical rank
    tokens = torch.randint(3, VOCAB, (1, SEQ + 1), device="cuda", generator=g)
    loss, _, _ = model(tokens[:, :-1], multi_segments=False, targets=tokens[:, 1:])
    loss.mean().backward()
    clip_grad_norm_(fsdp_module=model, max_norm=1.0)
    rescale_grads(model, 1.0)
    opt.step()
    sched.step()
    opt.zero_grad()


def tensors(model, opt):
    """This rank's local shards in a fixed order: params, then exp_avg, exp_avg_sq of each param."""
    out = []
    for p in model.parameters():
        out.append(p.to_local())
        st = opt.state[p]
        out += [st["exp_avg"].to_local(), st["exp_avg_sq"].to_local()]
    return out


def flat(model, opt):
    return torch.cat([t.detach().reshape(-1).float() for t in tensors(model, opt)])


@torch.no_grad()
def unflat(model, opt, buf, step_t, groups):
    """Restore from a flat fp32 buffer into a freshly built model (new mesh): plain tensors -> new DTensors."""
    off = 0
    params = list(model.parameters())
    for p in params:
        n = p.to_local().numel()
        p.to_local().copy_(buf[off:off + n].view_as(p.to_local())); off += n
        st = {}
        for k in ("exp_avg", "exp_avg_sq"):
            loc = buf[off:off + n].view_as(p.to_local()).clone(); off += n
            st[k] = DTensor.from_local(loc, p.device_mesh, p.placements, run_check=False, shape=p.shape, stride=p.stride())
        st["step"] = step_t.clone()
        opt.state[p] = st
    for g, saved in zip(opt.param_groups, groups):
        g.update(saved)


def main():
    torch.cuda.set_device(LOCAL)
    store = dist.TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]), N + 1, is_master=TASK == 0,
                          timeout=timedelta(seconds=900))

    # ---------------- reference: uninterrupted run on ranks 0..3 (the spare waits) ----------------
    if TASK < N:
        dist.init_process_group("nccl", store=dist.PrefixStore("ref", store), rank=TASK, world_size=N,
                                timeout=timedelta(seconds=120), device_id=torch.device("cuda", LOCAL))
        initialize_model_parallel(1, 1, 300)
        model, opt, sched = build(1)
        for i in range(STEPS):
            one_step(model, opt, sched, i, TASK)
        torch.save(flat(model, opt).cpu(), WORK / f"ref_{TASK}.pt")
        del model, opt, sched
        destroy_model_parallel()
        dist.destroy_process_group()
        store.set(f"ref_done_{TASK}", "1")
    store.wait([f"ref_done_{r}" for r in range(N)])

    if TASK == SPARE:
        spare(store)
    else:
        worker(store, TASK)


def worker(store, rank):
    me = rank
    dist.init_process_group("nccl", store=dist.PrefixStore("gen0", store), rank=rank, world_size=N,
                            timeout=timedelta(seconds=120), device_id=torch.device("cuda", LOCAL))
    initialize_model_parallel(1, 1, 300)
    log(f"gen0 up, world={N}, spare parked on task {SPARE}", me=me)
    model, opt, sched = build(1)

    # heartbeat publisher + watcher threads: they talk only to the store, never to NCCL
    progress = {"step": -1}
    dead = threading.Event()
    stop = threading.Event()

    def heartbeat():
        while not stop.is_set():
            store.set(f"hb_{rank}", f"{progress['step']}:{time.time()}")
            time.sleep(1.0)

    def watcher():
        while not stop.is_set():
            time.sleep(1.0)
            for r in range(N):
                if r == rank:
                    continue
                try:
                    _, ts = store.get(f"hb_{r}").decode().split(":")
                except Exception:
                    continue
                if time.time() - float(ts) > HB_DEADLINE:
                    store.set("failed_rank", str(r))
                    store.set("t_detect", str(time.time()))
                    dead.set()
                    dist.distributed_c10d._abort_process_group()        # unblock our hung collective
                    return

    threading.Thread(target=heartbeat, daemon=True).start()
    threading.Thread(target=watcher, daemon=True).start()

    own_copy = peer_copy = None
    step_t = groups = None
    t_fail = None
    i = 0
    try:
        while i < STEPS:
            if rank == DEAD and i == KILL_AT:
                print(f"[{time.strftime('%H:%M:%S')}] rank {rank}: dying at step {i}", flush=True)
                os._exit(1)
            if i == KILL_AT:
                t_fail = time.time()
            one_step(model, opt, sched, i, rank)
            # per-step replica: own copy on host + peer copy over NCCL (r sends to r+1)
            buf = flat(model, opt)
            recv = torch.empty_like(buf)
            ops = [dist.P2POp(dist.isend, buf, (rank + 1) % N), dist.P2POp(dist.irecv, recv, (rank - 1) % N)]
            for w in dist.batch_isend_irecv(ops):
                w.wait()
            torch.cuda.synchronize()
            own_copy, peer_copy = buf.cpu(), recv.cpu()
            step_t = next(iter(opt.state.values()))["step"].clone()
            groups = [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups]
            sched_state = sched.state_dict()
            i += 1
            progress["step"] = i
    except Exception as e:
        log(f"collective aborted after failure ({type(e).__name__}); last completed step {i}", me=me)
    if not dead.is_set():
        dead.wait(timeout=120)
    t_det = float(store.get("t_detect").decode())
    stop.set()
    failed = int(store.get("failed_rank").decode())
    log(f"rank {failed} declared dead {t_det - t_fail:.1f}s after it stopped (deadline {HB_DEADLINE:.0f}s)", me=me)

    # ---------------- swap ----------------
    t0 = time.time()
    del model, opt, sched
    destroy_model_parallel()
    try:
        dist.destroy_process_group()
    except Exception as e:
        log(f"destroy after abort raised {type(e).__name__}, continuing", me=me)
    torch.cuda.empty_cache()
    t_destroy = time.time() - t0
    store.set("resume_step", str(i))                 # every survivor completed the same step i
    store.set("swap_go", "1")
    t1 = time.time()
    dist.init_process_group("nccl", store=dist.PrefixStore("gen1", store), rank=rank, world_size=N,
                            timeout=timedelta(seconds=120), device_id=torch.device("cuda", LOCAL))
    initialize_model_parallel(1, 1, 300)
    t_init = time.time() - t1
    t2 = time.time()
    model, opt, sched = build(7)
    torch.cuda.synchronize()
    t_build = time.time() - t2
    t3 = time.time()
    if rank == (failed + 1) % N:                     # holder of the dead rank's replica sends it to the spare
        dist.send(peer_copy.cuda(), failed)
        meta = [step_t, groups, sched_state]
        dist.send_object_list(meta, dst=failed)
    unflat(model, opt, own_copy.cuda(), step_t, groups)
    sched.load_state_dict(sched_state)
    torch.cuda.synchronize()
    t_restore = time.time() - t3
    finish(model, opt, sched, rank, i, me, t_destroy, t_init, t_build, t_restore)


def spare(store):
    store.wait(["swap_go"], timedelta(seconds=600))
    failed = int(store.get("failed_rank").decode())
    resume = int(store.get("resume_step").decode())
    rank = failed                                     # the spare takes the dead rank's logical rank
    print(f"[{time.strftime('%H:%M:%S')}] spare (task {TASK}, GPU {LOCAL}) joining as logical rank {rank}", flush=True)
    t1 = time.time()
    dist.init_process_group("nccl", store=dist.PrefixStore("gen1", store), rank=rank, world_size=N,
                            timeout=timedelta(seconds=120), device_id=torch.device("cuda", LOCAL))
    initialize_model_parallel(1, 1, 300)
    t_init = time.time() - t1
    t2 = time.time()
    model, opt, sched = build(7)
    torch.cuda.synchronize()
    t_build = time.time() - t2
    t3 = time.time()
    buf = torch.empty_like(flat_zero(model))
    dist.recv(buf, (rank + 1) % N)
    meta = [None, None, None]
    dist.recv_object_list(meta, src=(rank + 1) % N)
    step_t, groups, sched_state = meta
    unflat(model, opt, buf, step_t, groups)
    sched.load_state_dict(sched_state)
    torch.cuda.synchronize()
    t_restore = time.time() - t3
    finish(model, opt, sched, rank, resume, None, 0.0, t_init, t_build, t_restore, spare_rank=rank)


def flat_zero(model):
    n = sum(3 * p.to_local().numel() for p in model.parameters())
    return torch.empty(n, device="cuda", dtype=torch.float32)


def finish(model, opt, sched, rank, start, me, t_destroy, t_init, t_build, t_restore, spare_rank=None):
    t4 = time.time()
    one_step(model, opt, sched, start, rank)
    torch.cuda.synchronize()
    t_first = time.time() - t4
    for i in range(start + 1, STEPS):
        one_step(model, opt, sched, i, rank)
    final = flat(model, opt).cpu()
    ref = torch.load(WORK / f"ref_{rank}.pt")
    ok = torch.equal(final, ref)
    flag = torch.tensor([1 if ok else 0], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    times = torch.tensor([t_destroy, t_init, t_build, t_restore, t_first], device="cuda")
    dist.all_reduce(times, op=dist.ReduceOp.MAX)
    if rank == 0:
        d, ini, b, r, f = times.tolist()
        print(f"RESULT resumed_at_step={start} destroy={d:.2f}s reinit_groups={ini:.2f}s rebuild_model={b:.2f}s "
              f"restore_state={r:.2f}s first_step={f:.2f}s bitwise_equal_all_ranks={'yes' if flag.item() else 'NO'}",
              flush=True)
    if spare_rank is not None:
        print(f"spare as rank {spare_rank}: matches reference = {ok}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
