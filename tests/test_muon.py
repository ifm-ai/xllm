"""Muon on FSDP2-sharded parameters against a dense reference.

    torchrun --nproc_per_node=8 tests/test_muon.py

Matrices are sharded on dim 0 like FSDP2 (uneven shards, heads that span several
ranks, row and column heads). After a few steps the sharded optimizer must match
per-block Newton-Schulz on the full matrices; whole-matrix mode must match
microsoft/dion, and the AdamW group must match torch.optim.AdamW.
"""
import math
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from xllm.config import OptimConf
from xllm.optim.muon import Muon, newton_schulz5

# (shape, block spec): None = whole matrix, (0, n) = row heads, (1, n) = column heads
MATRICES = [
    ((2048, 1536), (0, 256)),  # wq
    ((1024, 1536), (0, 256)),  # wk, GQA
    ((1536, 2048), (1, 256)),  # wo
    ((4096, 1536), None),      # fc1
    ((96, 64), (0, 32)),       # heads across shard boundaries
    ((100, 37), None),         # uneven shards
    ((13, 300), (1, 60)),      # fewer rows than ranks
]
# no weight decay in the Muon checks: the comparison is on the orthogonalized update only
CFG = OptimConf(optimizer="muon", lr=1e-3, muon_lr=0.02, muon_momentum=0.95, weight_decay=0.0,
                beta1=0.9, beta2=0.95, epsilon=1e-8)
CFG_NESTEROV_RMS = OptimConf(optimizer="muon", lr=1e-3, muon_lr=0.02, muon_momentum=0.95, weight_decay=0.0,
                             beta1=0.9, beta2=0.95, epsilon=1e-8, muon_nesterov=True, muon_adjust_lr="rms_norm")
CFG_ADAMW = OptimConf(optimizer="muon", lr=1e-3, muon_lr=0.02, weight_decay=0.1, beta1=0.9, beta2=0.95,
                      epsilon=1e-8)
STEPS = 3


def make(mesh, seed):
    torch.manual_seed(seed)
    full = [torch.randn(s, device="cuda") for s, _ in MATRICES]
    vecs = [torch.randn(64, device="cuda"), torch.randn(10, 7, device="cuda")]
    grads = [[torch.randn_like(t) for t in full + vecs] for _ in range(STEPS)]
    params = [nn.Parameter(distribute_tensor(t.clone(), mesh, [Shard(0)])) for t in full + vecs]
    return full, vecs, grads, params


def dense_muon(full, specs, grads, cfg):
    """Reference: every block orthogonalized on its own, on full matrices."""
    mom = [torch.zeros_like(x) for x in full]
    for step_grads in grads:
        for x, m, g, spec in zip(full, mom, step_grads, specs):
            m.mul_(cfg.muon_momentum).add_(g)
            u = (g + cfg.muon_momentum * m if cfg.muon_nesterov else m).to(torch.bfloat16)
            dim, size = spec if spec is not None else (0, x.shape[0])
            out = torch.empty_like(u)
            for s in range(0, x.shape[dim], size):
                blk = u.narrow(dim, s, min(size, x.shape[dim] - s))
                o = newton_schulz5((blk if blk.shape[0] <= blk.shape[1] else blk.mT)[None], cfg.epsilon)[0]
                o = o if blk.shape[0] <= blk.shape[1] else o.mT
                rows, cols = blk.shape
                scale = 0.2 * math.sqrt(max(rows, cols)) if cfg.muon_adjust_lr == "rms_norm" else math.sqrt(rows / cols)
                out.narrow(dim, s, blk.shape[dim]).copy_(o * scale)
            x.mul_(1 - cfg.muon_lr * cfg.weight_decay).add_(out.float(), alpha=-cfg.muon_lr)
    return full


def run_sharded(opt_cls, grad_mesh, model_params, grads, **kw):
    opt = opt_cls(**kw)
    for step_grads in grads:
        for p, g in zip(model_params, step_grads):
            p.grad = distribute_tensor(g, grad_mesh, [Shard(0)])
        opt.step()
    return [p.full_tensor() for p in model_params]


def update_error(got, want, init):
    """Max relative difference of the parameter updates (final - initial)."""
    return max(((a - x) - (b - x)).abs().max().item() / ((b - x).abs().max().item() + 1e-12)
               for a, b, x in zip(got, want, init))


def check(name, got, want, init, tol):
    err = update_error(got, want, init)
    if dist.get_rank() == 0:
        print(f"{name}: max relative error {err:.2e} (tol {tol:.0e})")
    assert err < tol, name


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    n = len(MATRICES)

    for per_head, cfg in ((True, CFG), (False, CFG), (True, CFG_NESTEROV_RMS)):
        specs = [spec if per_head else None for _, spec in MATRICES]
        full, vecs, grads, params = make(mesh, seed=0)
        got = run_sharded(Muon, mesh, params[:n], [g[:n] for g in grads], muon_params=params[:n],
                          block_sizes=specs, adamw_params=[], mesh=mesh, cfg=cfg)
        want = dense_muon([x.clone() for x in full], specs, [g[:n] for g in grads], cfg)
        check(f"muon per_head={per_head} nesterov={cfg.muon_nesterov} adjust={cfg.muon_adjust_lr} "
              f"vs dense reference", got, want, full, 3e-2)
        if per_head and not cfg.muon_nesterov:
            # negative control: per-head and whole-matrix updates must differ far beyond the tolerance
            whole = dense_muon([x.clone() for x in full], [None] * n, [g[:n] for g in grads], cfg)
            err = update_error(got[:4], whole[:4], full[:4])
            if dist.get_rank() == 0:
                print(f"negative control, per-head vs whole-matrix reference: {err:.2e}")
            assert err > 0.3, "per-head blocks are indistinguishable from whole matrices"

    full, vecs, grads, params = make(mesh, seed=0)
    got = run_sharded(Muon, mesh, params, grads, muon_params=params[:n], block_sizes=[None] * n,
                      adamw_params=params[n:], mesh=mesh, cfg=CFG_ADAMW)
    ref = [nn.Parameter(v.clone()) for v in vecs]
    adamw = torch.optim.AdamW(ref, lr=CFG_ADAMW.lr, betas=(CFG_ADAMW.beta1, CFG_ADAMW.beta2),
                              eps=CFG_ADAMW.epsilon, weight_decay=CFG_ADAMW.weight_decay)
    for step_grads in grads:
        for p, g in zip(ref, step_grads[n:]):
            p.grad = g
        adamw.step()
    check("adamw group vs torch.optim.AdamW", got[n:], [p.detach() for p in ref], vecs, 1e-4)

    try:
        from dion import Muon as DionMuon
    except ImportError:
        DionMuon = None
    if DionMuon is not None:
        full, vecs, grads, params = make(mesh, seed=1)
        got = run_sharded(Muon, mesh, params[:n], [g[:n] for g in grads], muon_params=params[:n],
                          block_sizes=[None] * n, adamw_params=[], mesh=mesh, cfg=CFG)
        _, _, _, dparams = make(mesh, seed=1)
        want = run_sharded(
            lambda **kw: DionMuon(**kw), mesh, dparams[:n], [g[:n] for g in grads],
            params=[{"params": dparams[:n], "algorithm": "muon", "lr": CFG.muon_lr}],
            distributed_mesh=mesh, mu=CFG.muon_momentum, weight_decay=CFG.weight_decay,
            epsilon=CFG.epsilon, adjust_lr="spectral_norm", use_polar_express=False,
        )
        check("muon whole-matrix vs dion.Muon", got, want, full, 3e-2)

    if dist.get_rank() == 0:
        print("MUON_TEST_OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
