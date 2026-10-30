"""Muon on the routed experts (xllm/modules/moe/expert.py) against a per-expert reference.

    torchrun --nproc_per_node=8 tests/test_muon_moe.py            # FSDP2 only (TP=1)
    TP=2 torchrun --nproc_per_node=8 tests/test_muon_moe.py       # experts split over a 2-way model-parallel group

The reference is written without MuonSpec: it cuts the full weight into n experts with a plain
reshape and orthogonalizes each one on its own, so a wrong layout, fan_out / fan_in or tp_dim in
register_muon cannot cancel against itself. The weights are the real Expert's, sharded by FSDP2:

1. synthetic gradients: weight1 / weight3 are n [inter, model] experts, weight2 n [model, inter] (tall);
   n below, equal to and above the number of ranks, so experts straddle shards and ranks hold none;
2. a real forward / backward with unrouted experts (zero gradient): FSDP2's averaged gradients are
   orthogonalized, unrouted experts stay put and nothing turns NaN (TP=1 only: the all-to-all of
   expert parallelism is outside this test).
"""
import math
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import Shard, distribute_tensor

from xllm.config import OptimConf
from xllm.distributed import get_model_parallel_group, get_model_parallel_rank, initialize_model_parallel
from xllm.distributed.utils import get_device_mesh
from xllm.distributed.wrap import enable_wrap, wrap
from xllm.modules.moe.expert import Expert
from xllm.optim.muon import build_muon, newton_schulz

TP = int(os.environ.get("TP", "1"))
STEPS = 3
TOL = 3e-2
CFG = OptimConf(optimizer="muon", lr=1e-3, muon_lr=0.02, muon_momentum=0.95, weight_decay=0.0,
                beta1=0.9, beta2=0.95, epsilon=1e-8)
# (experts, inter dim, model dim): inter != model, so every expert is a non-square matrix
SHAPES = [(8, 96, 160), (16, 96, 160), (6, 64, 96), (2, 64, 96)]


def log(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def build_expert(n, inter, model, dp_mesh):
    """The real Expert, sharded like models/build.py; the same seed gives every rank the same weights."""
    torch.manual_seed(0)
    with enable_wrap(mesh=dp_mesh):
        return wrap(Expert(model, inter, n, backend="sequential").cuda())


def gather(p):
    """The whole weight (all experts): FSDP shards gathered, then the expert-parallel chunks in rank order."""
    local = p.full_tensor().contiguous()
    if TP == 1:
        return local
    parts = [torch.empty_like(local) for _ in range(TP)]
    dist.all_gather(parts, local, group=get_model_parallel_group())
    return torch.cat(parts, dim=0)


def per_expert_muon(w, grads, n, cfg):
    """n experts stacked on dim 0 of w [n * rows, cols], each orthogonalized on its own; the update is
    scaled by sqrt(rows / cols) of one expert. Returns the weight after len(grads) steps."""
    rows = w.shape[0] // n
    w = w.clone()
    mom = torch.zeros_like(w)
    for g in grads:
        mom.mul_(cfg.muon_momentum).add_(g)
        u = (g + cfg.muon_momentum * mom).to(torch.bfloat16)  # nesterov
        for e in range(n):
            x = u[e * rows:(e + 1) * rows]
            wide = x.shape[0] <= x.shape[1]
            o = newton_schulz((x if wide else x.mT)[None], cfg.epsilon)[0]
            o = (o if wide else o.mT).float()
            w[e * rows:(e + 1) * rows] += -cfg.muon_lr * math.sqrt(x.shape[0] / x.shape[1]) * o
    return w


def whole_matrix_muon(w, grads, cfg):
    """The same weight orthogonalized as one matrix (the negative control)."""
    return per_expert_muon(w, grads, 1, cfg)


def update_error(got, want, init):
    return (got - want).abs().max().item() / ((want - init).abs().max().item() + 1e-12)


def check_layout(model, n, inter, dim):
    """What build_muon will read from the registered specs: n Blocks per weight, each one expert."""
    tp = TP
    for name, rows, cols in (("weight1", inter, dim), ("weight3", inter, dim), ("weight2", dim, inter)):
        spec = model.muon_specs[name]
        shape = tuple(getattr(model, name).shape)  # FSDP2 DTensor: global over the dp mesh, this rank's expert chunk
        blocks = spec.blocks(spec.global_shape(shape, tp))
        assert len(blocks) == n, (name, len(blocks), n)
        assert all(b.shape == (rows, cols) for b in blocks), (name, blocks[0].shape, (rows, cols))
        assert spec.fans(blocks[0].shape) == (rows, cols) and spec.tp_dim == 0, (name, spec)


def check_synthetic(n, inter, dim, dp_mesh):
    """Random gradients, STEPS Muon steps: the sharded optimizer against the per-expert reference."""
    model = build_expert(n, inter, dim, dp_mesh)
    check_layout(model, n, inter, dim)
    names = ("weight1", "weight2", "weight3")
    params = [getattr(model, k) for k in names]
    init = [gather(p) for p in params]
    opt = build_muon(model, CFG)
    assert len(opt.blocks) == 3 * n, (len(opt.blocks), n)
    # a weight's blocks are all local on a rank (no communication) if the rank's shard holds whole experts only
    def whole(rows):
        total, i = n // TP * rows, dp_mesh.get_local_rank()
        step = math.ceil(total / dp_mesh.size())  # FSDP2's torch.chunk of the TP-local weight
        r0, r1 = min(i * step, total), min((i + 1) * step, total)
        return r1 > r0 and r0 % rows == 0 and r1 % rows == 0
    expert_rows = {id(model.weight1): inter, id(model.weight2): dim, id(model.weight3): inter}
    name_of = {id(p): name for name, p in zip(names, params)}
    for i, p in enumerate(opt.param_groups[0]["params"]):  # the blocks I store are all local iff my shard holds whole experts
        spec = model.muon_specs[name_of[id(p)]]
        region = spec.regions(spec.global_shape(tuple(p.shape), TP), opt.coords, opt.dp, opt.tp)[opt.rank]
        stored = [b for b in opt.blocks if b.param == i and b.overlap(region)]
        assert stored and all(b.local for b in stored) == whole(expert_rows[id(p)]), (n, inter, dim, name_of[id(p)])
    everywhere = all(math.ceil(n // TP * rows / dp_mesh.size()) % rows == 0 for rows in expert_rows.values())
    assert opt.has_remote == (not everywhere), (n, opt.has_remote)
    log(f"TP={TP} experts={n}: rank 0 local blocks {[len(X) for X in opt.local]}, "
        f"{len(opt.remote)} buffers of remote blocks owned; all-to-alls: {opt.has_remote}")

    torch.manual_seed(1)  # the same on every rank
    grads = [[torch.randn_like(w) for w in init] for _ in range(STEPS)]
    tp_rank = get_model_parallel_rank() if TP > 1 else 0
    for step_grads in grads:
        for p, g in zip(params, step_grads):
            p.grad = distribute_tensor(g.chunk(TP, dim=0)[tp_rank].contiguous(), dp_mesh, [Shard(0)])
        opt.step()
    got = [gather(p) for p in params]

    errs = []
    for k, (w0, g, got_w) in enumerate(zip(init, zip(*grads), got)):
        want = per_expert_muon(w0, g, n, CFG)
        errs.append(update_error(got_w, want, w0))
        whole = whole_matrix_muon(w0, g, CFG)
        control = update_error(got_w, whole, w0)
        assert control > 0.1, f"{names[k]}: per-expert update is indistinguishable from a whole-matrix one ({control:.2e})"
    log(f"TP={TP} experts={n} inter={inter} model={dim}: max relative error {max(errs):.2e} (tol {TOL:.0e}); "
        f"whole-matrix control differs by > 0.1")
    assert max(errs) < TOL, (n, inter, dim, errs)


def check_end_to_end(dp_mesh, n=6, inter=64, dim=96):
    """A real forward / backward through the experts, some of which no token reaches."""
    assert TP == 1
    group_sizes = [20, 0, 15, 13, 0, 0]  # experts 1, 4, 5 are unrouted: their gradient is exactly zero
    assert len(group_sizes) == n
    tokens = sum(group_sizes)
    world, rank = dist.get_world_size(), dist.get_rank()
    xs = [torch.randn(tokens, dim, device="cuda", generator=torch.Generator("cuda").manual_seed(100 + r))
          for r in range(world)]

    model = build_expert(n, inter, dim, dp_mesh)
    names = ("weight1", "weight2", "weight3")
    params = [getattr(model, k) for k in names]
    init = [gather(p) for p in params]
    opt = build_muon(model, CFG)

    loss = model(xs[rank], group_sizes).square().mean()
    loss.backward()
    opt.step()
    got = [gather(p) for p in params]

    # reference: the same Expert unsharded; every rank's batch, gradients averaged like FSDP2's reduce-scatter
    torch.manual_seed(0)
    ref = Expert(dim, inter, n, backend="sequential").cuda()
    for k, w0 in zip(names, init):
        assert torch.equal(getattr(ref, k).detach(), w0), f"reference {k} does not start from the sharded weights"
    for x in xs:
        (ref(x, group_sizes).square().mean() / world).backward()

    for k, w0, got_w in zip(names, init, got):
        assert torch.isfinite(got_w).all(), f"{k}: non-finite after the step"
        g = getattr(ref, k).grad
        rows = w0.shape[0] // n
        want = per_expert_muon(w0, [g], n, CFG)
        err = update_error(got_w, want, w0)
        for e in (1, 4, 5):
            assert torch.equal(got_w[e * rows:(e + 1) * rows], w0[e * rows:(e + 1) * rows]), f"{k}: unrouted expert {e} moved"
        for e in (0, 2, 3):
            assert not torch.equal(got_w[e * rows:(e + 1) * rows], w0[e * rows:(e + 1) * rows]), f"{k}: routed expert {e} did not move"
        log(f"end to end {k}: max relative error {err:.2e} (tol {TOL:.0e}); unrouted experts unchanged, routed ones moved")
        assert err < TOL, (k, err)


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    initialize_model_parallel(TP, 1, timeout=600)
    dp = dist.get_world_size() // TP
    dp_mesh = get_device_mesh(dp, 1, TP)
    for n, inter, dim in SHAPES:
        if n % TP == 0:
            check_synthetic(n, inter, dim, dp_mesh)
    if TP == 1:
        check_end_to_end(dp_mesh)
    log(f"MUON_MOE_TEST_OK TP={TP}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
