"""Node-death hot swap: 2 active nodes + 1 spare node, one agent per node (srun --ntasks-per-node=1), G GPUs each.

Every step each trainer writes its own state into a double-buffered /dev/shm slot and sends a copy to its peer on the
other node ((r + G) % W), which writes it into its own /dev/shm "peer" slot. At step KILL_AT one trainer on node 1
kills its whole process group (agent + trainers): the node is gone, including its /dev/shm. Node 0's agent sees node 1's
heartbeat stop on the TCPStore, kills its own (hung) trainers and hands logical node 1 to the spare node. Generation 1:
node 0 trainers reload their own slot; spare trainers receive node 1's state from node 0's peer slots over NCCL.
Check: every rank's final state is bitwise equal to an uninterrupted reference; phase timings.
"""
import json
import os
import signal
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

N_STEPS, KILL_AT, HB_DEADLINE = 26, 10**9, 10.0
K = int(os.environ.get('FT_K', '1')); SKIP = os.environ.get('FT_SKIP', '0') == '1'
SIZES = {'1b': (2048, 16, 16, 5632), '8b': (4096, 32, 32, 14336)}
VOCAB, SEQ = 32000, 2048
SHM = Path("/dev/shm/xllm_ft_node")


def agent(workdir: Path, G: int):
    import torch.distributed as dist
    os.setpgrp()                                              # trainers share this group: killpg = node death
    node = int(os.environ["SLURM_NODEID"])
    head, port = os.environ["FT_HEAD"], int(os.environ["MASTER_PORT"])
    store = dist.TCPStore(head, port, 3, is_master=node == 0, timeout=timedelta(seconds=900))
    SHM.mkdir(parents=True, exist_ok=True)
    for f in SHM.glob("*"):
        f.unlink()

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] agent node{node}: {m}", flush=True)

    def launch(gen, mode, lnode, restore):
        return [subprocess.Popen([sys.executable, __file__, "worker"], env=dict(
            os.environ, RANK=str(lnode * G + l), WORLD_SIZE=str(2 * G), MASTER_ADDR=head,
            MASTER_PORT=str(port + 1 + (gen if isinstance(gen, int) else 90)), FT_GEN=str(gen), FT_MODE=mode,
            FT_RESTORE=restore, FT_WORKDIR=str(workdir), FT_G=str(G), CUDA_VISIBLE_DEVICES=str(l)))
            for l in range(G)]

    def beat():
        import threading
        def run():
            n = 0
            while True:                                       # a counter, never a timestamp: node clocks differ
                n += 1; store.set(f"hb_{node}", str(n)); time.sleep(1.0)
        threading.Thread(target=run, daemon=True).start()

    beat()
    # reference run on nodes 0 and 1
    if node < 2:
        for p in launch("ref", "ref", node, "none"):
            assert p.wait() == 0
        store.set(f"ref_done_{node}", "1")
    store.wait(["ref_done_0", "ref_done_1"])

    if node == 2:                                             # spare: parked until a node is handed to it
        while not store.check(["swap_go"]):
            if store.check(["done_0"]):
                log("run finished without a failure; spare exits"); return
            time.sleep(1.0)
        lnode = int(store.get("swap_lnode").decode())
        log(f"activated as logical node {lnode}")
        procs = launch(1, "train", lnode, "peer")
        codes = [p.wait() for p in procs]
        log(f"trainers finished with codes {codes}")
        return

    procs = launch(0, "train", node, "none")
    last_val, last_change = None, time.time()               # judged on this node's own clock only
    while True:
        codes = [p.poll() for p in procs]
        if all(c == 0 for c in codes):
            log("run finished"); store.set(f"done_{node}", "1"); return
        other = 1 - node
        val = store.get(f"hb_{other}").decode()
        if val != last_val:
            last_val, last_change = val, time.time()
        if node == 0 and time.time() - last_change > HB_DEADLINE and not store.check(["swap_go"]):
            t_detect = time.time()
            log(f"node {other} heartbeat unchanged for {time.time() - last_change:.1f}s: declared dead; killing local trainers")
            for p in procs:
                p.send_signal(signal.SIGKILL)
            for p in procs:
                p.wait()
            store.set("t_detect", str(t_detect))
            store.set("swap_lnode", str(other))
            store.set("swap_go", "1")
            log(f"local trainers dead after {time.time() - t_detect:.2f}s; logical node {other} -> spare node 2")
            procs = launch(1, "train", node, "own")
            codes = [p.wait() for p in procs]
            log(f"trainers finished with codes {codes}")
            return
        time.sleep(0.2)


def worker():
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor
    from xllm.config import ModelConf, OptimConf
    from xllm.distributed import initialize_model_parallel
    from xllm.models import build_model
    from xllm.optim import build_optimizer, rescale_grads
    from xllm.utils import clip_grad_norm_

    rank, W, G = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["FT_G"])
    gen, mode, restore, workdir = os.environ["FT_GEN"], os.environ["FT_MODE"], os.environ["FT_RESTORE"], Path(os.environ["FT_WORKDIR"])
    peer_of, peer_src = (rank + G) % W, (rank - G) % W
    t_start = time.time()
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=rank, world_size=W, timeout=timedelta(seconds=300), device_id=torch.device("cuda", 0))
    initialize_model_parallel(1, 1, 300)
    t_init = time.time() - t_start

    class Tok:
        vocab_size, bos_id, eos_id, pad_id = VOCAB, 1, 2, 0

    t0 = time.time()
    torch.manual_seed(1)
    model = build_model(ModelConf(arch="transformer", ddp_backend="fsdp2", init_mode="gaussian", init_std=0.02,
                                  num_layers=SIZES[os.environ['FT_SIZE']][1], model_dim=SIZES[os.environ['FT_SIZE']][0],
                                  num_heads=SIZES[os.environ['FT_SIZE']][2], head_dim=128, ffn_hidden_dim=SIZES[os.environ['FT_SIZE']][3],
                                  chunk_size=SEQ, vocab_size=VOCAB, apply_rmsnorm=True, qknorm=True, swiglu=True,
                                  rope_head_dim=128),
                        dtype="bf16", fully_sharded_size=None, fp32_reduce_scatter=True, reshard_after_forward=True,
                        forward_prefetch=False, tokenizer=Tok())
    opt, sched = build_optimizer(model, OptimConf(lr=1e-4, warmup=2, clip=1.0), 100, "bf16")
    params = list(model.parameters())
    sizes = [p.to_local().numel() for p in params]
    total = 3 * sum(sizes)
    torch.cuda.synchronize()
    t_build = time.time() - t0

    def step(i):
        g = torch.Generator(device="cuda").manual_seed(1000 * i + rank)
        tok = torch.randint(3, VOCAB, (1, SEQ + 1), device="cuda", generator=g)
        loss, _, _ = model(tok[:, :-1], multi_segments=False, targets=tok[:, 1:])
        loss.mean().backward()
        clip_grad_norm_(fsdp_module=model, max_norm=1.0)
        rescale_grads(model, 1.0)
        opt.step(); sched.step(); opt.zero_grad()

    def flat():
        parts = []
        for p in params:
            st = opt.state[p]
            parts += [p.to_local().reshape(-1), st["exp_avg"].to_local().reshape(-1), st["exp_avg_sq"].to_local().reshape(-1)]
        return torch.cat([t.detach().float() for t in parts])

    def slot(kind, r, s):
        f = SHM / f"{kind}_rank{r}_slot{s}.bin"
        if not f.exists() or f.stat().st_size != total * 4:
            with open(f, "wb") as fh:
                fh.truncate(total * 4)
        return f, torch.from_file(str(f), shared=True, size=total, dtype=torch.float32)

    def meta_path(kind, r, s):
        return SHM / f"{kind}_rank{r}_slot{s}.json"

    host = torch.empty(total, dtype=torch.float32, pin_memory=True)
    host_peer = torch.empty(total, dtype=torch.float32, pin_memory=True)

    def meta_now(i):
        return {"step": i, "opt_step": float(next(iter(opt.state.values()))["step"]),
                "groups": [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups],
                "sched": sched.state_dict()}

    import threading
    copy_stream = torch.cuda.Stream()
    gpu_mine = torch.empty(total, device="cuda"); gpu_recv = torch.empty(total, device="cuda")
    pending = {"thread": None, "event": None}

    skipped = {"n": 0, "saved": 0}

    peer_pg = dist.new_group(list(range(W)))                 # own communicator: never queued behind training collectives
    state = {"busy": False}

    def save(i):
        """Fully asynchronous: the training thread only packs on the GPU and enqueues; nothing waits."""
        if i % K:
            return
        flag = torch.tensor([1 if state["busy"] else 0], device="cuda"); dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            skipped["n"] += 1
            return
        skipped["saved"] += 1
        state["busy"] = True
        s = i % 2
        meta = meta_now(i)
        torch.cat([t.detach().float().reshape(-1) for t in sum(([p.to_local(), opt.state[p]["exp_avg"].to_local(),
                   opt.state[p]["exp_avg_sq"].to_local()] for p in params), [])], out=gpu_mine)
        ev = torch.cuda.Event(); ev.record()
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(ev)
            host.copy_(gpu_mine, non_blocking=True)
            works = dist.batch_isend_irecv([dist.P2POp(dist.isend, gpu_mine, peer_of, group=peer_pg),
                                            dist.P2POp(dist.irecv, gpu_recv, peer_src, group=peer_pg)])
            done_own = torch.cuda.Event(); done_own.record(copy_stream)

        def write():
            with torch.cuda.stream(copy_stream):
                for w in works:
                    w.wait()
                host_peer.copy_(gpu_recv, non_blocking=True)
                done_peer = torch.cuda.Event(); done_peer.record(copy_stream)
            done_own.synchronize(); done_peer.synchronize()
            for kind, r, h in (("own", rank, host), ("peer", peer_src, host_peer)):
                mp = meta_path(kind, r, s); mp.unlink(missing_ok=True)
                slot(kind, r, s)[1].copy_(h)
                tmp = mp.with_suffix(".tmp"); tmp.write_text(json.dumps(meta, default=float)); os.replace(tmp, mp)
            state["busy"] = False
        pending["thread"] = threading.Thread(target=write, daemon=True); pending["thread"].start()

    def save_sync_unused(i):
        s = i % 2
        mine = flat()
        recv = torch.empty_like(mine)
        for w in dist.batch_isend_irecv([dist.P2POp(dist.isend, mine, peer_of), dist.P2POp(dist.irecv, recv, peer_src)]):
            w.wait()
        for kind, r, src, h in (("own", rank, mine, host), ("peer", peer_src, recv, host_peer)):
            mp = meta_path(kind, r, s); mp.unlink(missing_ok=True)
            h.copy_(src, non_blocking=True); torch.cuda.synchronize()
            slot(kind, r, s)[1].copy_(h)
            tmp = mp.with_suffix(".tmp"); tmp.write_text(json.dumps(meta_now(i), default=float)); os.replace(tmp, mp)

    @torch.no_grad()
    def apply(buf, meta):
        off = 0
        for p, n in zip(params, sizes):
            p.to_local().copy_(buf[off:off + n].view_as(p.to_local())); off += n
            st = {}
            for k in ("exp_avg", "exp_avg_sq"):
                t = buf[off:off + n].view_as(p.to_local()).clone(); off += n
                st[k] = DTensor.from_local(t, p.device_mesh, p.placements, run_check=False, shape=p.shape, stride=p.stride())
            st["step"] = torch.tensor(meta["opt_step"])
            opt.state[p] = st
        for g, saved in zip(opt.param_groups, meta["groups"]):
            g.update(saved)
        sched.load_state_dict(meta["sched"])

    start, t_load = 0, 0.0
    if restore != "none":
        t0 = time.time()
        def steps(kind, r):
            out = {}
            for s in (0, 1):
                mp = meta_path(kind, r, s)
                if mp.exists():
                    out[json.loads(mp.read_text())["step"]] = s
            return out
        if restore == "own":   # survivor node: needs its own slot and the peer slot it holds for the dead node
            own, held = steps("own", rank), steps("peer", peer_src)
            cand = max(set(own) & set(held)) if set(own) & set(held) else -1
        else:
            cand = 1 << 30     # spare: no local state, defers to its peer holder
        c = torch.tensor([cand], device="cuda"); dist.all_reduce(c, op=dist.ReduceOp.MIN); common = int(c.item())
        if restore == "own":
            s_own, s_held = steps("own", rank)[common], steps("peer", peer_src)[common]
            dist.send(slot("peer", peer_src, s_held)[1].cuda(), peer_src)          # node 1's state -> its new owner
            meta = json.loads(meta_path("own", rank, s_own).read_text())
            apply(slot("own", rank, s_own)[1].cuda(), meta)
        else:
            buf = torch.empty(total, device="cuda"); dist.recv(buf, peer_of)
            meta = None
        # metadata (lr, scheduler, optimizer step) is identical on every rank: broadcast it from rank 0
        obj = [meta if rank == 0 else None]
        dist.broadcast_object_list(obj, src=0)
        if restore != "own":
            apply(buf, obj[0])
        start = common + 1
        torch.cuda.synchronize(); t_load = time.time() - t0

    t_first = None
    step_times = []
    if mode == 'ref':
        skipped = {'n': 0, 'saved': 0}
    for i in range(start, N_STEPS):
        if gen == "0" and rank == G and i == KILL_AT:        # first trainer of node 1: the node dies now
            os.killpg(os.getpgid(0), signal.SIGKILL)
        t0 = time.time(); step(i)
        if mode == "train":
            save(i)
        torch.cuda.current_stream().synchronize()           # training stream only: the copy stream runs on
        dt = time.time() - t0
        if t_first is None:
            t_first = dt
        elif i >= start + 2:
            step_times.append(dt)

    if mode == "train" and pending["thread"] is not None:
        pending["thread"].join()
    avg = torch.tensor([sum(step_times) / max(len(step_times), 1)], device="cuda"); dist.all_reduce(avg, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"STEPTIME async_peer=1 mode={mode} k={K} skip_if_busy={SKIP} saved={skipped['saved'] if mode == 'train' else 0} skipped={skipped['n'] if mode == 'train' else 0} size={os.environ['FT_SIZE']} world={W} avg_step={avg.item():.3f}s "
              f"(steady steps, {'with overlapped replica + peer copy' if mode == 'train' else 'no replica'})", flush=True)
    final = flat().cpu()
    if mode == "ref":
        torch.save(final, workdir / f"ref_{rank}.pt")
    else:
        ok = torch.equal(final, torch.load(workdir / f"ref_{rank}.pt"))
        f = torch.tensor([1 if ok else 0], device="cuda"); dist.all_reduce(f, op=dist.ReduceOp.MIN)
        tm = torch.tensor([t_init, t_build, t_load, t_first or 0.0], device="cuda"); dist.all_reduce(tm, op=dist.ReduceOp.MAX)
        if rank == 0 and gen == "1":
            a, b, c2, d = tm.tolist()
            print(f"RESULT node_death resumed_at_step={start} init_pg={a:.2f}s build={b:.2f}s restore={c2:.2f}s "
                  f"first_step={d:.2f}s size={os.environ['FT_SIZE']} world={W} shard_GB_per_rank={total * 4 / 1e9:.2f} bitwise_equal_all_ranks={'yes' if f.item() else 'NO'}",
                  flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    if sys.argv[1] == "agent":
        agent(Path(sys.argv[2]), int(sys.argv[3]))
    else:
        worker()
