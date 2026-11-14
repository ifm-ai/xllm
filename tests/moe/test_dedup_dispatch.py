"""Check the MoE dispatch and combine across TP ranks against an fp32 reference.

Covers the fused path (moe_fwd / moe_bwd) and the unfused NormalizedMoE, on one
TP group spanning all processes. Launch one process per rank, e.g.:

    torchrun --nproc_per_node 4 tests/moe/test_dedup_dispatch.py --backend triton
    torchrun --nproc_per_node 8 tests/moe/test_dedup_dispatch.py --backend torch --skew --topk 4
    torchrun --nproc_per_node 2 tests/moe/test_dedup_dispatch.py --shared_dim 0 --router_bias False

--skew routes every token to rank 0's experts, so the other ranks receive no rows.
--shared_dim sets the shared-expert width (0 disables it); --router_bias adds a random bias.
--gather_before_norm and --recompute_shared exercise the fused path's options.
"""
import os
from typing import Dict, Optional

import fire
import torch
import torch.nn.functional as F
from torch import Tensor

from xllm.distributed import (
    initialize_model_parallel,
    get_model_parallel_group,
    get_model_parallel_rank,
    get_model_parallel_world_size,
)
from xllm.models.fused_blocks.moe import moe_fwd, moe_bwd
from xllm.modules.moe.dispatch import build_dispatcher
from xllm.modules.moe.layout import ExpertLayout
from xllm.modules.moe.moe import NormalizedMoE

SCALING_FACTOR = 2.5
EXPERT_KEYS = ('w1', 'w2', 'w3')


def reference(x, weights: Dict[str, Optional[Tensor]], routing_indices, out_grad) -> Dict[str, Tensor]:
    """fp32 MoE output (routed + shared experts) and gradients for fixed routing indices."""
    leaves = {k: w.detach().float().requires_grad_() for k, w in weights.items() if w is not None}
    x = x.detach().float().requires_grad_()
    router_w, w1, w2, w3 = leaves['router_w'], leaves['w1'], leaves['w2'], leaves['w3']
    n_experts, model_dim = router_w.shape
    inter_dim = w1.size(0) // n_experts
    flat_x = x.view(-1, model_dim)
    flat_idx = routing_indices.view(flat_x.size(0), -1)
    scores = torch.gather(torch.sigmoid(flat_x @ router_w.t()), -1, flat_idx)
    scores = scores / scores.sum(dim=-1, keepdim=True) * SCALING_FACTOR

    y = torch.zeros_like(flat_x)
    for e in range(n_experts):
        tok, slot = (flat_idx == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        rows = slice(e * inter_dim, (e + 1) * inter_dim)
        xe = flat_x[tok]
        h = F.silu(xe @ w1[rows].t()) * (xe @ w3[rows].t())
        ye = h @ w2[e * model_dim:(e + 1) * model_dim].t()
        y = y.index_add(0, tok, scores[tok, slot].unsqueeze(1) * ye)
    if 's1' in leaves:
        y = y + (F.silu(flat_x @ leaves['s1'].t()) * (flat_x @ leaves['s3'].t())) @ leaves['s2'].t()
    y = y.view_as(x)

    (y * out_grad.float()).sum().backward()
    return {'out': y.detach(), 'x': x.grad, **{k: t.grad for k, t in leaves.items()}}


def local_experts(w: Tensor, n_experts: int, rank: int, world_size: int) -> Tensor:
    n_local = n_experts // world_size
    return w.view(n_experts, -1, w.size(-1))[rank * n_local:(rank + 1) * n_local].reshape(-1, w.size(-1))


def rel_err(a: Tensor, b: Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-12)).item()


def gather(t: Tensor, dim: int) -> Tensor:
    parts = [torch.empty_like(t) for _ in range(get_model_parallel_world_size())]
    torch.distributed.all_gather(parts, t.contiguous(), group=get_model_parallel_group())
    return torch.cat(parts, dim=dim)


def _grad(param: Tensor) -> Tensor:
    return param.grad if param.grad is not None else torch.zeros_like(param)


def shard(weights, n_experts, rank, world_size, cols):
    """This rank's part of every weight, laid out as the fused and unfused paths expect."""
    local = {k: local_experts(weights[k], n_experts, rank, world_size) for k in EXPERT_KEYS}
    local['router_w'] = weights['router_w'][:, cols]
    if weights['s1'] is not None:
        # shared fc1 / fc3 split their input features across TP, fc2 its output features
        local.update(s1=weights['s1'][:, cols], s2=weights['s2'][cols], s3=weights['s3'][:, cols])
    else:
        local.update(s1=None, s2=None, s3=None)
    return {k: (v.contiguous() if v is not None else None) for k, v in local.items()}


def collect(grads: Dict[str, Optional[Tensor]], out: Tensor, x_grad: Tensor, x_grad_full: bool):
    got = {'out': gather(out, -1), 'x': x_grad if x_grad_full else gather(x_grad, -1),
           'router_w': gather(grads['router_w'], -1)}
    got.update({k: gather(grads[k], 0) for k in EXPERT_KEYS})
    if grads.get('s1') is not None:
        got.update(s1=gather(grads['s1'], -1), s2=gather(grads['s2'], 0), s3=gather(grads['s3'], -1))
    return got


def run_fused(x, weights, router_bias, out_grad, n_experts, topk, backend, gather_before_norm, recompute_shared):
    rank, world_size = get_model_parallel_rank(), get_model_parallel_world_size()
    dim = x.size(-1) // world_size
    cols = slice(rank * dim, (rank + 1) * dim)
    dispatcher = build_dispatcher(ExpertLayout.from_process_groups(n_experts), topk, backend)
    w = shard(weights, n_experts, rank, world_size, cols)
    xf = x[..., cols].contiguous()
    bias = None if router_bias is None else router_bias.clone()

    out, _, routing_out, (h1, h3), dispatch_state, rng_states, _ = moe_fwd(
        xf, w['router_w'], bias, 'sigmoid', SCALING_FACTOR, None, w['s1'], w['s2'], w['s3'],
        n_experts, dispatcher, topk,
        w['w1'], w['w2'], w['w3'], 'sequential', backend, 0.0, recompute_shared, recompute_shared,
    )
    if recompute_shared:
        # the fused block drops these and lets the backward recompute them
        h1, h3 = None, None
    original_scores, routing_indices, org_topk_scores, tokens_per_expert = routing_out
    xf_grad, ew1_grad, ew2_grad, ew3_grad, s1_grad, s2_grad, s3_grad, rw_grad = moe_bwd(
        out_grad[..., cols].contiguous(), None, xf, xf.view(-1, dim),
        original_scores, routing_indices, org_topk_scores, tokens_per_expert,
        w['router_w'], bias, None if bias is None else 1e-3, 'sigmoid', SCALING_FACTOR, None,
        h1, h3, w['s1'], w['s2'], w['s3'], n_experts, dispatcher, topk, w['w1'], w['w2'], w['w3'],
        'sequential', backend, 0.0, dispatch_state, *rng_states, gather_before_norm,
    )
    grads = dict(router_w=rw_grad, w1=ew1_grad, w2=ew2_grad, w3=ew3_grad, s1=s1_grad, s2=s2_grad, s3=s3_grad)
    return collect(grads, out, xf_grad, gather_before_norm), routing_indices


def run_unfused(x, weights, router_bias, out_grad, n_experts, topk, backend, *_):
    rank, world_size = get_model_parallel_rank(), get_model_parallel_world_size()
    model_dim = x.size(-1)
    dim = model_dim // world_size
    cols = slice(rank * dim, (rank + 1) * dim)
    inter_dim = weights['w1'].size(0) // n_experts
    shared = weights['s1'] is not None
    w = shard(weights, n_experts, rank, world_size, cols)
    moe = NormalizedMoE(
        0, model_dim, inter_dim, n_experts, topk, weights['s1'].size(0) // inter_dim if shared else 0,
        expert_backend='sequential', permutation_backend=backend, router_score_func='sigmoid',
        router_bias=router_bias is not None, router_bias_update_rate=1e-3, router_scaling_factor=SCALING_FACTOR,
        apply_rmsnorm=True,
    ).cuda().bfloat16()
    # the dispatch is under test: feed the experts the layer input as is
    moe.norm = torch.nn.Identity()
    moe.norm.gather_input = False
    with torch.no_grad():
        moe.router.weight.copy_(w['router_w'])
        moe.experts.weight1.copy_(w['w1'])
        moe.experts.weight2.copy_(w['w2'])
        moe.experts.weight3.copy_(w['w3'])
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
    grads = {'router_w': moe.router.weight.grad}
    grads.update({k: _grad(getattr(moe.experts, f'weight{i}')) for i, k in enumerate(EXPERT_KEYS, 1)})
    if shared:
        grads.update(s1=moe.fc1.weight.grad, s2=moe.fc2.weight.grad, s3=moe.fc3.weight.grad)
    # the layer adds its input back: compare the experts' part only
    return collect(grads, out.detach() - xf.detach(), xf.grad - og, False), captured['indices']


def main(
    backend: str = 'triton',
    bsz: int = 2,
    seq_len: int = 128,
    model_dim: int = 256,
    inter_dim: int = 64,
    shared_dim: int = 64,
    n_experts: int = 32,
    topk: int = 8,
    skew: bool = False,
    router_bias: bool = True,
    gather_before_norm: bool = False,
    recompute_shared: bool = False,
    seed: int = 0,
    tol: float = 2e-2,
):
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')
    world_size = torch.distributed.get_world_size()
    initialize_model_parallel(world_size, 1, timeout=600)
    if skew:
        assert topk <= n_experts // world_size, "--skew needs topk <= experts per rank"

    torch.manual_seed(seed)

    def randn(*shape, fan_in):
        return (torch.randn(*shape, device='cuda') / fan_in ** 0.5).bfloat16()

    x = randn(bsz, seq_len, model_dim, fan_in=1)
    out_grad = randn(bsz, seq_len, model_dim, fan_in=1)
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
        bias = torch.zeros(n_experts, device='cuda')
        bias[:n_experts // world_size] = 10.0
    elif router_bias:
        bias = 0.1 * torch.randn(n_experts, device='cuda')

    failed = False
    options = (backend, gather_before_norm, recompute_shared)
    for name, run in (('fused', run_fused), ('unfused', run_unfused)):
        got, routing_indices = run(x, weights, bias, out_grad, n_experts, topk, *options)
        again, _ = run(x, weights, bias, out_grad, n_experts, topk, *options)
        ref = reference(x, weights, routing_indices, out_grad)
        for key in ref:
            err = rel_err(got[key], ref[key])
            same = torch.equal(got[key], again[key])
            ok = err < tol and same
            failed |= not ok
            if torch.distributed.get_rank() == 0:
                print(f"{name:8s} {key:9s} rel_err={err:.2e} run_to_run_equal={same} {'ok' if ok else 'FAIL'}")
        del got, again, ref

    torch.distributed.destroy_process_group()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    fire.Fire(main)
