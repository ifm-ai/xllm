"""Muon on FSDP2-sharded parameters against a dense reference.

    torchrun --nproc_per_node=8 tests/test_muon.py

Matrices are sharded on dim 0 like FSDP2 (uneven shards, empty shards, blocks that span
several ranks, column blocks and block grids that cut every shard). After a few steps the
sharded optimizer must match per-block Newton-Schulz on the full matrices; whole-matrix mode
must match microsoft/dion, and the AdamW group must match torch.optim.AdamW.
"""
import copy
import math
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from xllm.config import OptimConf
from xllm.distributed import initialize_model_parallel
from xllm.distributed.wrap import enable_wrap, wrap
from xllm.optim.muon import Muon, newton_schulz5, split_muon_params
from xllm.optim.muon_spec import muon_spec, register_muon
from xllm.optim import restore_optimizer_config

# (shape, how the test weight is cut into Newton-Schulz blocks)
MATRICES = [
    ((2048, 1536), muon_spec("(h d) m", fan_out="d", fan_in="m", d=256)),       # wq, one block per head
    ((1024, 1536), muon_spec("(h d) m", fan_out="d", fan_in="m", d=512)),       # wk, blocks of 2 heads
    ((1536, 2048), muon_spec()),                                               # wo
    ((4096, 1536), muon_spec()),                                               # fc1
    ((96, 64), muon_spec("(h d) m", fan_out="d", fan_in="m", d=32)),           # blocks across shard boundaries
    ((100, 37), muon_spec()),                                                  # uneven shards
    ((6, 300), muon_spec("(h d) m", fan_out="d", fan_in="m", d=2)),            # fewer rows than ranks: empty shards
    ((1536, 2048), muon_spec("o (h i)", fan_out="o", fan_in="i", i=256)),      # wo per head: blocks span every rank
    ((96, 64), muon_spec("(a r) (b c)", fan_out="r", fan_in="c", r=32, c=32)),  # 3 x 2 boxes cut both ways
    ((48, 160), muon_spec("(h i) o", fan_out="o", fan_in="i", i=24)),          # stored [in, out]: fan_out 160
]
WHOLE = muon_spec()


def row_boxes(shape, n):
    """n equal row ranges, all columns."""
    rows, cols = shape
    size = rows // n
    return [(r, r + size, 0, cols) for r in range(0, rows, size)]


# no weight decay in the Muon checks: the comparison is on the orthogonalized update only
CFG = OptimConf(optimizer="muon", lr=1e-3, muon_lr=0.02, muon_momentum=0.95, weight_decay=0.0,
                beta1=0.9, beta2=0.95, epsilon=1e-8)
CFG_PLAIN_RMS = OptimConf(optimizer="muon", lr=1e-3, muon_lr=0.02, muon_momentum=0.95, weight_decay=0.0,
                          beta1=0.9, beta2=0.95, epsilon=1e-8, muon_nesterov=False, muon_adjust_lr="rms_norm")
CFG_MATRIX_SCALE = OptimConf(optimizer="muon", lr=1e-3, muon_lr=0.02, muon_momentum=0.95, weight_decay=0.0,
                             beta1=0.9, beta2=0.95, epsilon=1e-8, muon_block_scale="matrix")
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
    """Reference: every block orthogonalized on its own, on full matrices; the update is scaled in
    fp32 by a block's fans (muon_block_scale=block) or by the whole weight's (matrix)."""
    mom = [torch.zeros_like(x) for x in full]
    for step_grads in grads:
        for x, m, g, spec in zip(full, mom, step_grads, specs):
            m.mul_(cfg.muon_momentum).add_(g)
            u = (g + cfg.muon_momentum * m if cfg.muon_nesterov else m).to(torch.bfloat16)
            out = torch.empty_like(u)
            for b in spec.blocks(tuple(x.shape)):
                blk = u[b.r0:b.r1, b.c0:b.c1]
                o = newton_schulz5((blk if blk.shape[0] <= blk.shape[1] else blk.mT)[None], cfg.epsilon)[0]
                out[b.r0:b.r1, b.c0:b.c1] = o if blk.shape[0] <= blk.shape[1] else o.mT
            fan_out, fan_in = spec.fans(tuple(x.shape), per_block=cfg.muon_block_scale == "block")
            scale = 0.2 * math.sqrt(max(fan_out, fan_in)) if cfg.muon_adjust_lr == "rms_norm" else math.sqrt(fan_out / fan_in)
            x.mul_(1 - cfg.muon_lr * cfg.weight_decay).add_(out.float(), alpha=-cfg.muon_lr * scale)
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


def check_registration(mesh):
    """Specs registered on layers survive fully_shard and .cuda() and decide the Muon / AdamW split."""
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.wq = nn.Linear(64, 128, bias=False)
            self.wo = nn.Linear(128, 64, bias=False)
            self.fc = nn.Linear(64, 64)
            for lin in (self.wq, self.wo, self.fc):
                lin.tp_dims = {"weight": None}  # plain nn.Linear: not split over tensor parallelism
            register_muon(self.wq, layout="(h d) m", fan_out="d", fan_in="m", d=32)
            register_muon(self.wo)
            register_muon(self.fc)

    # same as models/build.py: each layer wrapped, then .cuda() (swaps the DTensor params), then the root
    with enable_wrap(mesh=mesh):
        model = wrap(nn.Sequential(wrap(Layer().cuda()), nn.Linear(64, 8)).cuda())
    muon, specs, names, adamw = split_muon_params(model)
    layer = model[0]
    assert muon == [layer.wq.weight, layer.wo.weight, layer.fc.weight], "wrong Muon params"
    assert all(hasattr(p, "to_local") for p in muon), "Muon params are not the sharded ones"
    assert names == ["*.wq.weight", "*.wo.weight", "*.fc.weight"] and [str(s) for s in specs] == ["(h d) m", "o i", "o i"], (names, specs)
    assert [len(s.blocks(tuple(p.shape))) for s, p in zip(specs, muon)] == [4, 1, 1]
    assert len(adamw) == 3  # fc.bias, unregistered Linear weight and bias
    if dist.get_rank() == 0:
        print("registered specs survive fully_shard and .cuda() and split params as expected")


def check_model_blocks():
    """register_muon in the real blocks gives the Blocks of the marks it replaced: wq / wr one row
    block per head, wk / wv one per KV head, wo / wg / FFN whole; experts one per expert, shared
    experts whole."""
    import dataclasses
    from xllm.configuration import ConfStore
    from xllm.models.gekko import GekkoBlock
    from xllm.models.transformer import TransformerBlock, TransformerMoEBlock

    def expected(parent, attr, shape):
        heads = getattr(parent, "n_heads", None) or getattr(parent, "num_heads", None)
        kv_heads = getattr(parent, "n_kv_heads", None) or getattr(parent, "num_kv_heads", None)
        n = {"wq": heads, "wr": heads, "wk": kv_heads, "wv": kv_heads}.get(attr, 1)
        if type(parent).__name__ == "Expert":
            n = parent.n_experts
        return row_boxes(shape, n)

    # (config, block, overrides, registered weights: the marked ones, plus experts and shared experts)
    cases = [("k2-horizon-0.9B", TransformerBlock, {}, 7), ("k2-horizon-0.9B", TransformerBlock, {"attn_act_func": "softdelta"}, 8),
             ("gekko-0.9B", GekkoBlock, {}, 8), ("gekko-0.9B", GekkoBlock, {"attn_act_func": "softdelta"}, 9),
             ("gekko-7.1B", GekkoBlock, {}, 8), ("k2-horizon-moe-36B-a4B", TransformerMoEBlock, {}, 11)]
    for name, cls, overrides, n_registered in cases:
        block = cls(dataclasses.replace(ConfStore[name], **overrides), layer_id=0)
        checked = []
        for parent in block.modules():
            for attr in ("wq", "wk", "wv", "wr", "wo", "wg", "fc1", "fc2", "fc3", "weight1", "weight2", "weight3"):
                child = getattr(parent, attr, None)
                owner, pname = (parent, attr) if isinstance(child, nn.Parameter) else (child, "weight")
                spec = (getattr(owner, "muon_specs", None) or {}).get(pname) if owner is not None else None
                if spec is None:
                    continue
                shape = tuple(getattr(owner, pname).shape)
                blocks = spec.blocks(shape)
                assert [(b.r0, b.r1, b.c0, b.c1) for b in blocks] == expected(parent, attr, shape), (name, overrides, attr, spec)
                assert spec.fans(shape, per_block=True) == blocks[0].shape, (name, attr, spec)  # all stored [out, in]
                checked.append(attr)
        n_specs = sum(len(getattr(m, "muon_specs", None) or {}) for m in block.modules())
        assert len(checked) == n_specs == n_registered, (name, overrides, checked, n_specs)
        if dist.get_rank() == 0:
            print(f"{name} {overrides or ''}: Blocks as before for {sorted(set(checked))} ({len(checked)} weights)")


def check_resume_config_wins(mesh):
    """State loading restores checkpoint settings; training explicitly applies config overrides."""
    n = len(MATRICES)
    _, _, grads, params = make(mesh, seed=2)
    old = Muon(params[:n], [WHOLE] * n, params[n:], mesh, CFG_ADAMW)  # weight decay 0.1, eps 1e-8
    for step_grads in grads:
        for p, g in zip(params, step_grads):
            p.grad = distribute_tensor(g, mesh, [Shard(0)])
        old.step()
    saved = copy.deepcopy(old.state_dict())
    assert [g["weight_decay"] for g in saved["param_groups"]] == [0.1, 0.1]
    cfg = OptimConf(optimizer="muon", lr=1e-3, muon_lr=0.02, weight_decay=0.05, muon_weight_decay=0.2,
                    beta1=0.9, beta2=0.95, epsilon=1e-6)
    new = Muon(params[:n], [WHOLE] * n, params[n:], mesh, cfg)
    new.load_state_dict(saved)
    assert [g["weight_decay"] for g in new.param_groups] == [0.1, 0.1]
    restore_optimizer_config(new, cfg)
    assert [g["weight_decay"] for g in new.param_groups] == [0.2, 0.05], new.param_groups
    assert [g["eps"] for g in new.param_groups] == [1e-6, 1e-6]
    for p in params[:n]:
        assert torch.equal(new.state[p]["momentum"].to_local(), old.state[p]["momentum"].to_local())
    for p in params[n:]:
        for k in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(new.state[p][k].to_local(), old.state[p][k].to_local()), k
    if dist.get_rank() == 0:
        print("resume: momentum and moments reloaded, configured weight decay and eps kept")


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    n = len(MATRICES)

    for heads, cfg in ((True, CFG), (False, CFG), (True, CFG_PLAIN_RMS), (True, CFG_MATRIX_SCALE)):
        specs = [spec if heads else WHOLE for _, spec in MATRICES]  # heads=False: every matrix one block
        full, vecs, grads, params = make(mesh, seed=0)
        got = run_sharded(Muon, mesh, params[:n], [g[:n] for g in grads], muon_params=params[:n],
                          specs=specs, adamw_params=[], mesh=mesh, cfg=cfg)
        want = dense_muon([x.clone() for x in full], specs, [g[:n] for g in grads], cfg)
        check(f"muon heads={heads} scale={cfg.muon_block_scale} nesterov={cfg.muon_nesterov} "
              f"adjust={cfg.muon_adjust_lr} vs dense reference", got, want, full, 3e-2)
        if heads and cfg is CFG:
            # negative controls: per-head vs whole-matrix, and block vs matrix scale, must differ far beyond the tolerance
            whole_ref = dense_muon([x.clone() for x in full], [WHOLE] * n, [g[:n] for g in grads], cfg)
            other_scale = dense_muon([x.clone() for x in full], specs, [g[:n] for g in grads], CFG_MATRIX_SCALE)
            err_whole = update_error(got[:4], whole_ref[:4], full[:4])
            err_scale = update_error(got[:4], other_scale[:4], full[:4])
            swapped = specs[:-1] + [muon_spec("(h o) i", fan_out="o", fan_in="i", o=24)]  # same boxes, fans swapped
            as_out_in = dense_muon([x.clone() for x in full], swapped, [g[:n] for g in grads], cfg)
            err_fan = update_error(got[-1:], as_out_in[-1:], full[-1:])
            if dist.get_rank() == 0:
                print(f"negative controls: per-head vs whole-matrix {err_whole:.2e}, block vs matrix scale {err_scale:.2e}, "
                      f"[in, out] weight read as [out, in] {err_fan:.2e}")
            assert err_whole > 0.3, "per-head blocks are indistinguishable from whole matrices"
            assert err_scale > 0.3, "block and matrix scales are indistinguishable"
            assert err_fan > 0.3, "the fan_out dim does not change the update scale"

    full, vecs, grads, params = make(mesh, seed=0)
    got = run_sharded(Muon, mesh, params, grads, muon_params=params[:n], specs=[WHOLE] * n,
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
        # the original matrices only: the [in, out] one is for the fan_out-dim control above; against dion
        # (both reading it as [out, in]) bf16 Newton-Schulz rounding gives it 3.5e-2 at seed 1, 1.7e-2 at seeds 2-3
        k = n - 1
        full, vecs, grads, params = make(mesh, seed=1)
        got = run_sharded(Muon, mesh, params[:k], [g[:k] for g in grads], muon_params=params[:k],
                          specs=[WHOLE] * k, adamw_params=[], mesh=mesh, cfg=CFG)
        _, _, _, dparams = make(mesh, seed=1)
        want = run_sharded(
            lambda **kw: DionMuon(**kw), mesh, dparams[:k], [g[:k] for g in grads],
            params=[{"params": dparams[:k], "algorithm": "muon", "lr": CFG.muon_lr}],
            distributed_mesh=mesh, mu=CFG.muon_momentum, nesterov=CFG.muon_nesterov, weight_decay=CFG.weight_decay,
            epsilon=CFG.epsilon, adjust_lr="spectral_norm", use_polar_express=False,
        )
        check("muon whole-matrix vs dion.Muon", got, want, full[:k], 3e-2)

    check_registration(mesh)
    initialize_model_parallel(1, 1, timeout=600)
    check_model_blocks()
    check_resume_config_wins(mesh)

    if dist.get_rank() == 0:
        print("MUON_TEST_OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
