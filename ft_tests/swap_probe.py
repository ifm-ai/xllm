"""Probe of the hot-swap core: destroy and re-create all NCCL process groups in a running process, rebuild the FSDP2
model, restore its state from host memory, and continue training. One srun task per GPU.

gen0: reference run of 6 steps; then a second run of 3 steps whose model+optimizer state is snapshotted to CPU.
swap: destroy_process_group + xLLM group reset, re-init on a new store generation, rebuild, restore, train steps 3-5.
Check: final parameters equal the reference bitwise; log the time of each phase (max over ranks).
"""
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from xllm.config import ModelConf, OptimConf
from xllm.distributed import initialize_model_parallel
from xllm.distributed.initialize import destroy_model_parallel
from xllm.models import build_model
from xllm.optim import build_optimizer, rescale_grads
from torch.distributed.tensor import DTensor

from xllm.utils import clip_grad_norm_

RANK, WORLD, LOCAL = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"]), int(os.environ["SLURM_LOCALID"])
SIZE = sys.argv[1] if len(sys.argv) > 1 else "1b"
DIMS = {"1b": (2048, 16, 16, 5632, 128), "8b": (4096, 32, 32, 14336, 128)}
VOCAB, SEQ = 32000, 2048


class Tok:
    vocab_size, bos_id, eos_id, pad_id = VOCAB, 1, 2, 0


def log(msg):
    if RANK == 0:
        print(msg, flush=True)


def gmax(x):
    t = torch.tensor([float(x)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def init_generation(store, gen):
    dist.init_process_group("nccl", store=dist.PrefixStore(f"gen{gen}", store), rank=RANK, world_size=WORLD,
                            timeout=timedelta(seconds=600), device_id=torch.device("cuda", LOCAL))
    initialize_model_parallel(1, 1, 600)


def build(seed):
    dim, layers, heads, ffn, hd = DIMS[SIZE]
    cfg = ModelConf(arch="transformer", ddp_backend="fsdp2", init_mode="gaussian", init_std=0.02, num_layers=layers,
                    model_dim=dim, num_heads=heads, head_dim=hd, ffn_hidden_dim=ffn, chunk_size=SEQ, vocab_size=VOCAB,
                    apply_rmsnorm=True, qknorm=True, swiglu=True, rope_head_dim=hd)
    torch.manual_seed(seed)
    model = build_model(cfg, dtype="bf16", fully_sharded_size=None, fp32_reduce_scatter=True,
                        reshard_after_forward=True, forward_prefetch=False, tokenizer=Tok())
    opt, sched = build_optimizer(model, OptimConf(lr=1e-4, warmup=2, clip=1.0), 100, "bf16")
    return model, opt, sched


def train(model, opt, sched, steps):
    for i in steps:
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


def local_shards(model):
    return {k: v.to_local().detach().float().cpu().clone() for k, v in model.state_dict().items()}


def snapshot(model, opt):
    """Host copy of this rank's shards only (plain tensors: a DTensor would keep the old, destroyed mesh)."""
    params = []
    for p in model.parameters():
        st = opt.state.get(p, {})
        params.append({"param": p.to_local().detach().cpu().clone(),
                       "state": {k: (v.to_local() if isinstance(v, DTensor) else v).detach().cpu().clone()
                                 for k, v in st.items()}})
    return params


@torch.no_grad()
def restore(model, opt, snap):
    for p, s in zip(model.parameters(), snap):
        p.to_local().copy_(s["param"])
        opt.state[p] = {k: (DTensor.from_local(v.cuda(), p.device_mesh, p.placements, run_check=False,
                                               shape=p.shape, stride=p.stride()) if v.dim() > 0 else v.clone())
                        for k, v in s["state"].items()}


def main():
    torch.cuda.set_device(LOCAL)
    host, port = os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"])
    # The store outlives the process groups: a new generation is a new key prefix on it.
    store = dist.TCPStore(host, port, WORLD, is_master=RANK == 0, timeout=timedelta(seconds=600))
    init_generation(store, 0)
    log(f"swap probe: size={SIZE} world={WORLD} nodes={os.environ.get('SLURM_NNODES')} torch={torch.__version__}")

    model, opt, sched = build(1)
    train(model, opt, sched, range(6))
    reference = local_shards(model)
    del model, opt, sched

    model, opt, sched = build(1)
    train(model, opt, sched, range(3))
    t = time.perf_counter()
    snap = snapshot(model, opt)
    snap_sched = sched.state_dict()
    snap_groups = [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups]
    t_snap = gmax(time.perf_counter() - t)

    # ---- the swap: tear down every group, keep only host-memory state ----
    del model, opt, sched
    t0 = time.perf_counter()
    destroy_model_parallel()
    dist.destroy_process_group()
    torch.cuda.empty_cache()
    t_destroy = time.perf_counter() - t0
    t1 = time.perf_counter()
    init_generation(store, 1)
    t_init = time.perf_counter() - t1
    t2 = time.perf_counter()
    model, opt, sched = build(7)                                       # different seed: state must come from the snapshot
    torch.cuda.synchronize()
    t_build = time.perf_counter() - t2
    t3 = time.perf_counter()
    restore(model, opt, snap)
    for g, saved in zip(opt.param_groups, snap_groups):
        g.update(saved)                                                 # current lr lives here, not in the scheduler
    sched.load_state_dict(snap_sched)
    torch.cuda.synchronize()
    t_restore = time.perf_counter() - t3
    t4 = time.perf_counter()
    train(model, opt, sched, range(3, 4))
    t_first = time.perf_counter() - t4
    train(model, opt, sched, range(4, 6))

    resumed = local_shards(model)
    same = all(torch.equal(reference[k], resumed[k]) for k in reference)
    flag = torch.tensor([1 if same else 0], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    log(f"RESULT size={SIZE} snapshot={t_snap:.2f}s destroy={gmax(t_destroy):.2f}s reinit_groups={gmax(t_init):.2f}s "
        f"rebuild_model={gmax(t_build):.2f}s restore_state={gmax(t_restore):.2f}s first_step={gmax(t_first):.2f}s "
        f"bitwise_equal={'yes' if flag.item() else 'NO'}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
