"""Check the MoE dispatch at EP > TP against a reference and against EP = TP.

Every data-parallel replica gets its own tokens and the global weights are the same everywhere. At
EP = TP x E_out, the fused path (moe_fwd / moe_bwd) and the unfused NormalizedMoE are compared, per replica,
on the output, the input gradient and the router and shared-expert gradients, and per expert group on the
routed experts' gradients gathered by global expert index, with:

  ref    an fp32 reference computing every expert for every token of the expert group (same routing);
  ep=tp  the same path at EP = TP with the same global weights, its expert gradients summed over the
         replicas of the expert group.

It also checks that two runs are bitwise equal, the routing is the same at both EP, and every result is
finite. Launch one process per rank, e.g.:

    torchrun --nproc_per_node 8 tests/moe/test_ep_dispatch.py --tp 2 --ep 4
    torchrun --nproc_per_node 8 tests/moe/test_ep_dispatch.py --tp 2 --ep 8 --dtype fp32
    torchrun --nproc_per_node 8 tests/moe/test_ep_dispatch.py --tp 1 --ep 4 --skew --topk 4

--skew routes every token to the experts of the GPU at TP rank 0 of the first replica of each expert
group, so the other GPUs, and the other replicas, receive no rows. --shared_dim 0 disables the shared
expert; --router_bias False drops the router bias.
"""
import os
from typing import Dict, List, Optional

import fire
import torch
from torch import Tensor

from xllm.distributed import (
    initialize_model_parallel,
    destroy_model_parallel,
    get_model_parallel_rank,
    get_model_parallel_world_size,
)
from xllm.models.fused_blocks.moe import moe_fwd, moe_bwd
from xllm.modules.moe.dispatch import build_dispatcher
from xllm.modules.moe.layout import ExpertLayout
from xllm.modules.moe.moe import NormalizedMoE
from tests.moe.test_dedup_dispatch import reference, rel_err, SCALING_FACTOR, EXPERT_KEYS

SHARED_KEYS = ('s1', 's2', 's3')


def local_weights(weights: Dict[str, Optional[Tensor]], layout: ExpertLayout, cols: slice) -> Dict[str, Optional[Tensor]]:
    """This rank's part of every weight: its experts, and its TP slice of the router and shared expert."""
    n = layout.n_experts
    local = {k: weights[k].view(n, -1, weights[k].size(-1))[layout.start:layout.end].reshape(-1, weights[k].size(-1))
             for k in EXPERT_KEYS}
    local['router_w'] = weights['router_w'][:, cols]
    if weights['s1'] is not None:
        local.update(s1=weights['s1'][:, cols], s2=weights['s2'][cols], s3=weights['s3'][:, cols])
    else:
        local.update(s1=None, s2=None, s3=None)
    return {k: (v.contiguous() if v is not None else None) for k, v in local.items()}


def run_fused(x, out_grad, weights, router_bias, layout, topk, backend):
    rank, tp = get_model_parallel_rank(), get_model_parallel_world_size()
    dim = x.size(-1) // tp
    cols = slice(rank * dim, (rank + 1) * dim)
    w = local_weights(weights, layout, cols)
    dispatcher = build_dispatcher(layout, topk, backend)
    xf = x[..., cols].contiguous()
    bias = None if router_bias is None else router_bias.clone()
    out, _, routing_out, (h1, h3), dispatch_state, rng_states = moe_fwd(
        xf, w['router_w'], bias, 'sigmoid', SCALING_FACTOR, None, w['s1'], w['s2'], w['s3'],
        layout.n_experts, dispatcher, topk, w['w1'], w['w2'], w['w3'], 'sequential', backend, 0.0, False, False,
    )
    original_scores, routing_indices, org_topk_scores, tokens_per_expert = routing_out
    xf_grad, ew1_grad, ew2_grad, ew3_grad, s1_grad, s2_grad, s3_grad, rw_grad = moe_bwd(
        out_grad[..., cols].contiguous(), None, xf, xf.view(-1, dim),
        original_scores, routing_indices, org_topk_scores, tokens_per_expert,
        w['router_w'], bias, None if bias is None else 1e-3, 'sigmoid', SCALING_FACTOR, None,
        h1, h3, w['s1'], w['s2'], w['s3'], layout.n_experts, dispatcher, topk, w['w1'], w['w2'], w['w3'],
        'sequential', backend, 0.0, dispatch_state, *rng_states, False,
    )
    got = dict(out=out, x=xf_grad, router_w=rw_grad, w1=ew1_grad, w2=ew2_grad, w3=ew3_grad,
               s1=s1_grad, s2=s2_grad, s3=s3_grad)
    return got, routing_indices


def run_unfused(x, out_grad, weights, router_bias, layout, topk, backend):
    rank, tp = get_model_parallel_rank(), get_model_parallel_world_size()
    model_dim = x.size(-1)
    dim = model_dim // tp
    cols = slice(rank * dim, (rank + 1) * dim)
    inter_dim = weights['w1'].size(0) // layout.n_experts
    shared = weights['s1'] is not None
    w = local_weights(weights, layout, cols)
    moe = NormalizedMoE(
        0, model_dim, inter_dim, layout.n_experts, topk, weights['s1'].size(0) // inter_dim if shared else 0,
        expert_backend='sequential', permutation_backend=backend, router_score_func='sigmoid',
        router_bias=router_bias is not None, router_bias_update_rate=1e-3, router_scaling_factor=SCALING_FACTOR,
        apply_rmsnorm=True,
    ).cuda().to(x.dtype)
    assert moe.layout == layout
    # the dispatch is under test: feed the experts the layer input as is
    moe.norm = torch.nn.Identity()
    moe.norm.gather_input = False
    with torch.no_grad():
        moe.router.weight.copy_(w['router_w'])
        for i, k in enumerate(EXPERT_KEYS, 1):
            getattr(moe.experts, f'weight{i}').copy_(w[k])
        if shared:
            moe.fc1.weight.copy_(w['s1'])
            moe.fc2.weight.copy_(w['s2'])
            moe.fc3.weight.copy_(w['s3'])
        if router_bias is not None:
            moe.router.bias.copy_(router_bias)
    captured = {}
    moe.router.register_forward_hook(lambda _, __, out: captured.update(indices=out[1]))

    xf = x[..., cols].contiguous().requires_grad_()
    out, _ = moe(xf)
    og = out_grad[..., cols].contiguous()
    out.backward(og)
    grad = lambda p: p.grad if p.grad is not None else torch.zeros_like(p)
    got = {'router_w': moe.router.weight.grad}
    got.update({k: grad(getattr(moe.experts, f'weight{i}')) for i, k in enumerate(EXPERT_KEYS, 1)})
    got.update({k: (getattr(moe, f'fc{i}').weight.grad if shared else None) for i, k in enumerate(SHARED_KEYS, 1)})
    # the layer adds its input back: compare the experts' part only
    got.update(out=out.detach() - xf.detach(), x=xf.grad - og)
    return got, captured['indices']


def all_gather(t: Tensor) -> List[Tensor]:
    parts = [torch.empty_like(t) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(parts, t.contiguous())
    return parts


def assemble(got: Dict[str, Optional[Tensor]], routing_indices: Tensor, tp: int, e_out: int, n_experts: int):
    """Every rank's results, put together: per replica (full width) and per expert group (all experts)."""
    world = torch.distributed.get_world_size()
    n_replicas = world // tp
    per_replica = {}
    # (key, dim of the TP split)
    split = [('out', -1), ('x', -1), ('router_w', -1)]
    if got['s1'] is not None:
        split += [('s1', -1), ('s2', 0), ('s3', -1)]
    for key, dim in split:
        parts = all_gather(got[key])
        per_replica[key] = [torch.cat(parts[r * tp:(r + 1) * tp], dim=dim) for r in range(n_replicas)]
    indices = all_gather(routing_indices)
    per_replica['indices'] = [indices[r * tp] for r in range(n_replicas)]
    for r in range(n_replicas):
        for k in range(1, tp):
            assert torch.equal(indices[r * tp + k], indices[r * tp]), "TP ranks disagree on the routing"

    per_group = {}
    for key in EXPERT_KEYS:
        parts = all_gather(got[key])
        groups = []
        for g in range(n_replicas // e_out):
            full = [None] * n_experts
            for o in range(e_out):
                for k in range(tp):
                    layout = ExpertLayout(n_experts, tp, e_out, k, o)
                    rows = parts[(g * e_out + o) * tp + k].view(layout.n_local, -1, parts[0].size(-1))
                    for i in range(layout.n_local):
                        full[layout.start + i] = rows[i]
            groups.append(torch.cat(full, dim=0))
        per_group[key] = groups
    return per_replica, per_group


def summed_over_groups(per_group_tp: Dict[str, List[Tensor]], e_out: int) -> Dict[str, List[Tensor]]:
    """EP = TP expert gradients (one 'group' per replica) summed over the replicas of each EP > TP group."""
    return {k: [sum(v[g * e_out:(g + 1) * e_out]) for g in range(len(v) // e_out)] for k, v in per_group_tp.items()}


def main(
    tp: int = 2,
    ep: int = 4,
    dtype: str = 'bf16',
    backend: str = 'triton',
    paths: str = 'fused,unfused',
    bsz: int = 2,
    seq_len: int = 128,
    model_dim: int = 256,
    inter_dim: int = 64,
    shared_dim: int = 64,
    n_experts: int = 32,
    topk: int = 8,
    skew: bool = False,
    router_bias: bool = True,
    seed: int = 0,
    tol: Optional[float] = None,
):
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.distributed.init_process_group(backend='nccl')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    world = torch.distributed.get_world_size()
    rank0 = torch.distributed.get_rank() == 0
    assert ep % tp == 0 and world % ep == 0, f"world {world}, tp {tp}, ep {ep}"
    e_out = ep // tp
    n_replicas = world // tp
    torch_dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32}[dtype]
    tol = tol if tol is not None else (2e-2 if dtype == 'bf16' else 1e-5)
    if skew:
        assert topk <= n_experts // ep, "--skew needs topk <= experts per GPU"

    torch.manual_seed(seed)

    def randn(*shape, fan_in):
        return (torch.randn(*shape, device='cuda') / fan_in ** 0.5).to(torch_dtype)

    # every replica's tokens, on every rank
    xs = [randn(bsz, seq_len, model_dim, fan_in=1) for _ in range(n_replicas)]
    out_grads = [randn(bsz, seq_len, model_dim, fan_in=1) for _ in range(n_replicas)]
    weights = {
        'router_w': randn(n_experts, model_dim, fan_in=model_dim),
        'w1': randn(n_experts * inter_dim, model_dim, fan_in=model_dim),
        'w2': randn(n_experts * model_dim, inter_dim, fan_in=inter_dim),
        'w3': randn(n_experts * inter_dim, model_dim, fan_in=model_dim),
        's1': randn(shared_dim, model_dim, fan_in=model_dim) if shared_dim else None,
        's2': randn(model_dim, shared_dim, fan_in=shared_dim) if shared_dim else None,
        's3': randn(shared_dim, model_dim, fan_in=model_dim) if shared_dim else None,
    }
    bias = None
    if skew:
        # the experts of TP rank 0, replica 0 of the expert group
        bias = torch.zeros(n_experts, device='cuda')
        bias[:n_experts // ep] = 10.0
    elif router_bias:
        bias = 0.1 * torch.randn(n_experts, device='cuda')

    replica = torch.distributed.get_rank() // tp
    x, out_grad = xs[replica], out_grads[replica]
    runs = {'fused': run_fused, 'unfused': run_unfused}
    failed = False
    for name in paths.split(','):
        run = runs[name]
        results = {}
        for size in (tp, ep):
            initialize_model_parallel(tp, 1, timeout=600, expert_parallel_size=size)
            layout = ExpertLayout.from_process_groups(n_experts)
            assert (layout.tp, layout.e_out) == (tp, size // tp)
            got, indices = run(x, out_grad, weights, bias, layout, topk, backend)
            results[size] = assemble(got, indices, tp, size // tp, n_experts)
            if size == ep:
                again, _ = run(x, out_grad, weights, bias, layout, topk, backend)
                again = assemble(again, indices, tp, e_out, n_experts)
            destroy_model_parallel()
            del got

        if rank0:
            per_replica, per_group = results[ep]
            tp_replica, tp_group = results[tp]
            tp_group = summed_over_groups(tp_group, e_out)
            refs = [reference(xs[r], weights, per_replica['indices'][r], out_grads[r]) for r in range(n_replicas)]
            checks = []
            for key in per_replica:
                if key == 'indices':
                    continue
                for r in range(n_replicas):
                    checks.append((key, f'replica {r}', per_replica[key][r], refs[r][key],
                                   tp_replica[key][r], again[0][key][r]))
            for key in per_group:
                for g in range(len(per_group[key])):
                    ref = sum(refs[r][key] for r in range(g * e_out, (g + 1) * e_out))
                    checks.append((key, f'group {g}', per_group[key][g], ref, tp_group[key][g], again[1][key][g]))
            worst = {}
            for key, where, value, ref, at_tp, rerun in checks:
                err_ref, err_tp = rel_err(value, ref), rel_err(value, at_tp)
                finite = bool(torch.isfinite(value).all() and torch.isfinite(at_tp).all())
                same = torch.equal(value, rerun)
                ok = err_ref < tol and err_tp < tol and finite and same
                if not ok:
                    print(f"FAIL {name} {key} {where}: rel_err ref={err_ref:.2e} ep=tp={err_tp:.2e} "
                          f"finite={finite} run_to_run_equal={same}")
                failed |= not ok
                w = worst.setdefault(key, [0.0, 0.0, True, True])
                w[0], w[1] = max(w[0], err_ref), max(w[1], err_tp)
                w[2], w[3] = w[2] and finite, w[3] and same
            for r in range(n_replicas):
                if not torch.equal(per_replica['indices'][r], tp_replica['indices'][r]):
                    print(f"FAIL {name}: replica {r} routes differently at EP {ep} and EP {tp}")
                    failed = True
            print(f"{name} tp={tp} ep={ep} dtype={dtype} backend={backend} skew={skew} topk={topk} "
                  f"shared={shared_dim > 0} bias={bias is not None}, tol {tol:.0e}:")
            for key, (e_ref, e_tp, finite, same) in worst.items():
                print(f"  {key:9s} max rel_err ref={e_ref:.2e} ep=tp={e_tp:.2e} finite={finite} "
                      f"run_to_run_equal={same} {'ok' if e_ref < tol and e_tp < tol and finite and same else 'FAIL'}")
        flag = torch.tensor([float(failed)], device='cuda')
        torch.distributed.broadcast(flag, 0)
        failed = bool(flag.item())

    torch.distributed.destroy_process_group()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    fire.Fire(main)
