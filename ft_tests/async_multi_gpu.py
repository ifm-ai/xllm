"""Multi-GPU / multi-node check of the async checkpointing draft, one srun task per GPU.

Run A: train 8 steps; async save at step 4 while training continues; checkpoint finalized at close.
Run B: fresh model, resume from that checkpoint exactly as train.py does, train steps 5-8.
Check: B's final parameters equal A's bitwise (DCP and torch-format optimizer), the checkpoint was
not marked complete before finalization, and its layout matches a synchronous save.
"""
import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_ckpt

from xllm.checkpointing import AsyncCheckpointer, SyncCheckpointer, get_checkpoint_dir, sharded_model_dir, \
    sharded_optim_dir, sharded_optim_name, training_state_dir, training_state_name
from xllm.config import ModelConf, OptimConf
from xllm.distributed import get_hybrid_shard_data_parallel_group, initialize_model_parallel
from xllm.models import build_model
from xllm.optim import build_optimizer, rescale_grads
from xllm.utils import clip_grad_norm_, get_master_rank_in_replicated_group, get_model_state, get_sharded_optimizer_state, set_model_state, \
    set_sharded_optimizer_state

RANK, WORLD, LOCAL = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"]), int(os.environ["SLURM_LOCALID"])


class Tok:
    vocab_size, bos_id, eos_id, pad_id = 512, 1, 2, 0


def log(msg):
    if RANK == 0:
        print(msg, flush=True)


def check(name, ok, detail=""):
    flag = torch.tensor([1 if ok else 0], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    log(f"  [{'PASS' if flag.item() else 'FAIL'}] {name} {detail}")
    return bool(flag.item())


def build(seed):
    cfg = ModelConf(arch="transformer", ddp_backend="fsdp2", init_mode="gaussian", init_std=0.02, num_layers=4,
                    model_dim=256, num_heads=16, head_dim=16, ffn_hidden_dim=512, chunk_size=16, vocab_size=512,
                    apply_rmsnorm=True, qknorm=True, swiglu=True, rope_head_dim=16)
    torch.manual_seed(seed)
    model = build_model(cfg, dtype="bf16", fully_sharded_size=None, fp32_reduce_scatter=True,
                        reshard_after_forward=True, forward_prefetch=False, tokenizer=Tok())
    opt, sched = build_optimizer(model, OptimConf(lr=3e-3, warmup=2, clip=1.0), 20, "bf16")
    return model, opt, sched


def train(model, opt, sched, steps):
    for step in steps:
        g = torch.Generator(device="cuda").manual_seed(1000 * step + RANK)
        tokens = torch.randint(3, 512, (4, 65), device="cuda", generator=g)
        loss, _, _ = model(tokens[:, :-1], multi_segments=False, targets=tokens[:, 1:])
        loss.mean().backward()
        clip_grad_norm_(fsdp_module=model, max_norm=1.0)
        rescale_grads(model, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()


def full(model):
    return {k: v.full_tensor().float().cpu() for k, v in model.state_dict().items()}


def run(workdir: Path, dcp: bool):
    dump = workdir / ("dcp" if dcp else "torch")
    cfg = SimpleNamespace(dcp_for_optimizer=dcp, fully_sharded_size=None, to_json=lambda: "{}")
    results = []

    # ---- run A: uninterrupted, async save at step 4 ----
    model, opt, sched = build(1)
    train(model, opt, sched, range(4))
    ck = AsyncCheckpointer(dump_dir=str(dump), cfg=cfg, global_rank=RANK, world_size=WORLD)
    live_step = next(iter(opt.state.values()))["step"]
    snap = get_sharded_optimizer_state(model, opt, dcp)
    snap_step = next(v["step"] for v in snap["state"].values() if isinstance(v, dict) and "step" in v)
    log(f"     live optimizer step: {live_step!r} device={getattr(live_step,'device',None)}; "
        f"state-dict step shares storage with it: {getattr(snap_step,'data_ptr',lambda:0)() == getattr(live_step,'data_ptr',lambda:1)()}")
    training_state = {"scheduler": copy.deepcopy(sched.state_dict()), "step": 4}
    ck.save_latest_checkpoint(model_state=get_model_state(model, full_state=False),
                              optimizer_state=get_sharded_optimizer_state(model, opt, dcp),
                              training_state=training_state, step=4)
    ckdir = get_checkpoint_dir(dump, 4)
    results.append(check("not marked complete while in flight", not (ckdir / "config.json").exists()))
    train(model, opt, sched, range(4, 8))           # training continues during the background write
    ck.check_ok()
    ck.close()                                      # finalize: write collectives + config.json
    dist.barrier()
    results.append(check("complete after close", (ckdir / "config.json").exists()))
    reference = full(model)

    # ---- sync save of a fresh model at step 4, to compare the on-disk layout ----
    m2, o2, s2 = build(1)
    train(m2, o2, s2, range(4))
    sck = SyncCheckpointer(dump_dir=str(dump), cfg=cfg, global_rank=RANK, world_size=WORLD)
    sck.save_latest_checkpoint(model_state=get_model_state(m2, full_state=False),
                               optimizer_state=get_sharded_optimizer_state(m2, o2, dcp),
                               training_state={"scheduler": copy.deepcopy(s2.state_dict()), "step": 4}, step=5)
    dist.barrier()
    files = lambda d: sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())
    same = files(ckdir) == files(get_checkpoint_dir(dump, 5))
    results.append(check("same file layout as a synchronous save", same))
    del m2, o2, s2

    # ---- run B: resume from the async checkpoint as train.py does ----
    model, opt, sched = build(7)
    hsdp = get_hybrid_shard_data_parallel_group()
    ms = get_model_state(model, full_state=False)
    dist_ckpt.load(ms, checkpoint_id=ckdir / sharded_model_dir(0), process_group=hsdp)
    set_model_state(model, ms, full_state=False)
    if dcp:
        os_ = get_sharded_optimizer_state(model, opt, True)
        dist_ckpt.load(os_, checkpoint_id=ckdir / sharded_optim_dir(0), process_group=hsdp)
    else:
        master = get_master_rank_in_replicated_group(RANK, None)
        os_ = torch.load(ckdir / sharded_optim_dir(0) / sharded_optim_name(master), map_location="cpu",
                         weights_only=False)
    steps_in_ckpt = sorted({float(v["step"]) for v in os_["state"].values() if isinstance(v, dict) and "step" in v})
    log(f"     optimizer 'step' values stored in the step-4 checkpoint: {steps_in_ckpt}")
    set_sharded_optimizer_state(model, opt, os_, dcp)
    ts = torch.load(ckdir / training_state_dir() / training_state_name(RANK), map_location="cpu", weights_only=False)
    sts = ts["scheduler"]
    sts["base_lrs"] = list(sched.base_lrs)
    sched.load_state_dict(sts)
    train(model, opt, sched, range(4, 8))
    resumed = full(model)
    diff = [k for k in reference if not torch.equal(reference[k], resumed[k])]
    results.append(check("resume from async checkpoint == uninterrupted run, bitwise", not diff,
                         f"(differing tensors: {len(diff)}/{len(reference)})"))
    return results


def main():
    torch.cuda.set_device(LOCAL)
    dist.init_process_group("nccl", rank=RANK, world_size=WORLD)     # NCCL only, as xLLM
    initialize_model_parallel(1, 1, 600)
    log(f"async checks: world={WORLD} nodes={os.environ.get('SLURM_NNODES')} "
        f"backends={dist.get_backend()} torch={torch.__version__}")
    workdir = Path(sys.argv[1])
    allres = []
    for dcp in (True, False):
        log(f"-- optimizer format: {'DCP' if dcp else 'torch per-rank'}")
        allres += run(workdir, dcp)
    log(f"SUMMARY world={WORLD}: {sum(allres)}/{len(allres)} passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
