"""Optimizer step time on routed experts: AdamW against Muon, over block layouts and parallel layouts.

    TP=2 torchrun --nproc_per_node=8 tests/test_muon_moe_speed.py     # TP = expert-parallel size, dp = 8 / TP

Experts of k2-horizon-moe-36B-a4B (128 experts, model 2560, inter 768), LAYERS Expert modules sharded by
FSDP2 like models/build.py, random gradients. Compared per step:

- adamw:        torch's AdamW (foreach) on the same sharded weights
- muon expert:  one Newton-Schulz block per expert (what Expert registers)
- muon whole:   each weight one block (the registration overridden)
"""
import os
import time

import torch
import torch.distributed as dist
from torch.optim.adamw import AdamW

from xllm.config import OptimConf
from xllm.distributed import initialize_model_parallel
from xllm.distributed.utils import get_device_mesh
from xllm.distributed.wrap import enable_wrap, wrap
from xllm.modules.moe.expert import Expert
from xllm.optim.muon import build_muon
from xllm.optim.muon_spec import muon_spec

TP = int(os.environ.get("TP", "1"))
LAYERS = int(os.environ.get("LAYERS", "4"))
EXPERTS, INTER, MODEL = 128, 768, 2560
WARMUP, ITERS = 3, 10
BATCH = int(os.environ.get("BATCH", "250000000"))  # muon_batch_numel, elements of a rank; one layer's experts are 3 * 128 * 768 * 2560 / 8 = 94M at TP x dp = 8
CFG = OptimConf(optimizer="muon", lr=1e-3, muon_lr=0.02, muon_momentum=0.95, weight_decay=0.01,
                beta1=0.9, beta2=0.95, epsilon=1e-8, muon_batch_numel=BATCH)


def build(mesh, whole):
    layers = []
    for i in range(LAYERS):
        torch.manual_seed(i)
        with enable_wrap(mesh=mesh):
            e = wrap(Expert(MODEL, INTER, EXPERTS, backend="sequential").cuda())
        if whole:  # one block per weight; the weights stay split over the expert-parallel group
            for name in e.muon_specs:
                e.muon_specs[name] = muon_spec(tp_dim=0)
        layers.append(e)
    return torch.nn.ModuleList(layers)


def set_grads(model):
    for p in model.parameters():
        p.grad = torch.randn_like(p)  # a DTensor with the weight's placement


def timed(step):
    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    dist.barrier()
    t = time.perf_counter()
    for _ in range(ITERS):
        step()
    torch.cuda.synchronize()
    ms = torch.tensor((time.perf_counter() - t) / ITERS * 1e3, device="cuda")
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)
    return ms.item()


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    initialize_model_parallel(TP, 1, timeout=600)
    torch._dynamo.config.cache_size_limit = 64
    dp = dist.get_world_size() // TP
    mesh = get_device_mesh(dp, 1, TP)
    rows = []
    for name in ("adamw", "muon expert", "muon whole"):
        model = build(mesh, whole=name == "muon whole")
        set_grads(model)
        if name == "adamw":
            opt = AdamW(list(model.parameters()), lr=CFG.lr, betas=(CFG.beta1, CFG.beta2), eps=CFG.epsilon,
                        weight_decay=CFG.weight_decay, foreach=True)
            blocks, remote = 0, 0
        else:
            opt = build_muon(model, CFG)
            blocks, remote = len(opt.blocks), len(opt.remote)
        rows.append((name, blocks, remote, timed(opt.step)))
        del opt, model
        torch.cuda.empty_cache()
    if dist.get_rank() == 0:
        base = rows[0][3]
        print(f"EP(TP)={TP} dp={dp} layers={LAYERS} experts={EXPERTS} local weights/layer="
              f"{3 * EXPERTS // TP * INTER * MODEL / dp / 1e6:.1f}M params per rank")
        for name, blocks, remote, ms in rows:
            print(f"SPEED TP={TP} dp={dp} {name:12s} blocks={blocks:5d} remote matrices={remote:3d} step {ms:8.2f} ms  "
                  f"({ms / base:5.2f}x adamw)", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
