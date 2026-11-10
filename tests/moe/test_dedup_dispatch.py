"""Check the MoE dispatch and combine across TP ranks against an fp32 reference.

Covers the fused path (moe_fwd / moe_bwd) and the unfused NormalizedMoE, on one
TP group spanning all processes. Launch one process per rank, e.g.:

    torchrun --nproc_per_node 4 tests/moe/test_dedup_dispatch.py --backend triton
    torchrun --nproc_per_node 8 tests/moe/test_dedup_dispatch.py --backend torch --skew

--skew routes every token to rank 0's experts, so the other ranks receive no rows.
"""
import os
from typing import Dict

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
from xllm.modules.moe.moe import NormalizedMoE

SCALING_FACTOR = 2.5


def reference(x, router_w, w1, w2, w3, routing_indices, out_grad) -> Dict[str, Tensor]:
    """fp32 MoE output (routed experts only) and gradients for fixed routing indices."""
    n_experts, model_dim = router_w.shape
    inter_dim = w1.size(0) // n_experts
    x, router_w, w1, w2, w3 = [t.detach().float().requires_grad_() for t in (x, router_w, w1, w2, w3)]
    scores = torch.gather(torch.sigmoid(x @ router_w.t()), -1, routing_indices)
    scores = scores / scores.sum(dim=-1, keepdim=True) * SCALING_FACTOR
    w1e = w1.view(n_experts, inter_dim, model_dim)
    w2e = w2.view(n_experts, model_dim, inter_dim)
    w3e = w3.view(n_experts, inter_dim, model_dim)
    y = torch.zeros_like(x)
    for k in range(routing_indices.size(-1)):
        e = routing_indices[..., k]
        h = F.silu(torch.einsum('bld,blhd->blh', x, w1e[e])) * torch.einsum('bld,blhd->blh', x, w3e[e])
        y = y + scores[..., k:k + 1] * torch.einsum('blh,bldh->bld', h, w2e[e])
    (y * out_grad.float()).sum().backward()
    return {'out': y.detach(), 'x': x.grad, 'router_w': router_w.grad, 'w1': w1.grad, 'w2': w2.grad, 'w3': w3.grad}


def local_experts(w: Tensor, n_experts: int, rank: int, world_size: int) -> Tensor:
    n_local = n_experts // world_size
    return w.view(n_experts, -1, w.size(-1))[rank * n_local:(rank + 1) * n_local].reshape(-1, w.size(-1))


def rel_err(a: Tensor, b: Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-12)).item()


def gather_cols(t: Tensor) -> Tensor:
    parts = [torch.empty_like(t) for _ in range(get_model_parallel_world_size())]
    torch.distributed.all_gather(parts, t.contiguous(), group=get_model_parallel_group())
    return torch.cat(parts, dim=-1)


def gather_rows(t: Tensor) -> Tensor:
    parts = [torch.empty_like(t) for _ in range(get_model_parallel_world_size())]
    torch.distributed.all_gather(parts, t.contiguous(), group=get_model_parallel_group())
    return torch.cat(parts, dim=0)


def run_fused(x, router_w, w1, w2, w3, router_bias, out_grad, n_experts, topk, backend):
    rank, world_size = get_model_parallel_rank(), get_model_parallel_world_size()
    dim = x.size(-1) // world_size
    cols = slice(rank * dim, (rank + 1) * dim)
    n_local = n_experts // world_size
    xf = x[..., cols].contiguous().bfloat16()
    rw = router_w[:, cols].contiguous().bfloat16()
    ew1, ew2, ew3 = [local_experts(w, n_experts, rank, world_size).bfloat16() for w in (w1, w2, w3)]
    bias = None if router_bias is None else router_bias.clone()

    out, _, routing_out, _, routing_state, rng_states = moe_fwd(
        xf, rw, bias, 'sigmoid', SCALING_FACTOR, None, None, None, None,
        n_experts, n_local, rank * n_local, (rank + 1) * n_local, topk,
        ew1, ew2, ew3, 'sequential', backend, 0.0, False, False,
    )
    original_scores, routing_indices, org_topk_scores, tokens_per_expert = routing_out
    xf_grad, ew1_grad, ew2_grad, ew3_grad, _, _, _, rw_grad = moe_bwd(
        out_grad[..., cols].contiguous().bfloat16(), None, xf, xf.view(-1, dim),
        original_scores, routing_indices, org_topk_scores, tokens_per_expert,
        rw, bias, None if bias is None else 1e-3, 'sigmoid', SCALING_FACTOR, None,
        None, None, None, None, None, n_experts, n_local, topk, ew1, ew2, ew3,
        'sequential', backend, 0.0, *routing_state, *rng_states, False,
    )
    got = {
        'out': gather_cols(out), 'x': gather_cols(xf_grad), 'router_w': gather_cols(rw_grad),
        'w1': gather_rows(ew1_grad), 'w2': gather_rows(ew2_grad), 'w3': gather_rows(ew3_grad),
    }
    return got, routing_indices


def _grad(param: Tensor) -> Tensor:
    return param.grad if param.grad is not None else torch.zeros_like(param)


def run_unfused(x, router_w, w1, w2, w3, router_bias, out_grad, n_experts, topk, backend):
    rank, world_size = get_model_parallel_rank(), get_model_parallel_world_size()
    model_dim = x.size(-1)
    dim = model_dim // world_size
    cols = slice(rank * dim, (rank + 1) * dim)
    moe = NormalizedMoE(
        0, model_dim, w1.size(0) // n_experts, n_experts, topk, 0,
        expert_backend='sequential', permutation_backend=backend, router_score_func='sigmoid',
        router_bias=router_bias is not None, router_bias_update_rate=1e-3, router_scaling_factor=SCALING_FACTOR,
        apply_rmsnorm=True,
    ).cuda().bfloat16()
    # the dispatch is under test: feed the experts the layer input as is
    moe.norm = torch.nn.Identity()
    moe.norm.gather_input = False
    with torch.no_grad():
        moe.router.weight.copy_(router_w[:, cols])
        moe.experts.weight1.copy_(local_experts(w1, n_experts, rank, world_size))
        moe.experts.weight2.copy_(local_experts(w2, n_experts, rank, world_size))
        moe.experts.weight3.copy_(local_experts(w3, n_experts, rank, world_size))
        if router_bias is not None:
            moe.router.bias.copy_(router_bias)
    captured = {}
    moe.router.register_forward_hook(lambda _, __, out: captured.update(indices=out[1]))

    xf = x[..., cols].contiguous().bfloat16().requires_grad_()
    out, _ = moe(xf)
    out.backward(out_grad[..., cols].contiguous().bfloat16())
    got = {
        # the layer adds its input back: compare the experts' part only
        'out': gather_cols(out.detach() - xf.detach()), 'x': gather_cols(xf.grad - out_grad[..., cols].bfloat16()),
        'router_w': gather_cols(moe.router.weight.grad),
        # a rank that receives no tokens never runs its experts, so their grads stay None
        **{f'w{i}': gather_rows(_grad(getattr(moe.experts, f'weight{i}'))) for i in (1, 2, 3)},
    }
    return got, captured['indices']


def main(
    backend: str = 'triton',
    bsz: int = 2,
    seq_len: int = 128,
    model_dim: int = 256,
    inter_dim: int = 64,
    n_experts: int = 32,
    topk: int = 8,
    skew: bool = False,
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
    x = torch.randn(bsz, seq_len, model_dim, device='cuda')
    router_w = torch.randn(n_experts, model_dim, device='cuda') / model_dim ** 0.5
    w1 = torch.randn(n_experts * inter_dim, model_dim, device='cuda') / model_dim ** 0.5
    w2 = torch.randn(n_experts * model_dim, inter_dim, device='cuda') / inter_dim ** 0.5
    w3 = torch.randn(n_experts * inter_dim, model_dim, device='cuda') / model_dim ** 0.5
    out_grad = torch.randn(bsz, seq_len, model_dim, device='cuda')
    router_bias = None
    if skew:
        router_bias = torch.zeros(n_experts, device='cuda')
        router_bias[:n_experts // world_size] = 10.0

    failed = False
    for name, run in (('fused', run_fused), ('unfused', run_unfused)):
        got, routing_indices = run(x, router_w, w1, w2, w3, router_bias, out_grad, n_experts, topk, backend)
        again, _ = run(x, router_w, w1, w2, w3, router_bias, out_grad, n_experts, topk, backend)
        ref = reference(
            x.bfloat16(), router_w.bfloat16(), w1.bfloat16(), w2.bfloat16(), w3.bfloat16(),
            routing_indices, out_grad.bfloat16(),
        )
        for key in ref:
            err = rel_err(got[key], ref[key])
            # the router's score gradient uses atomic adds in the triton backend
            same = torch.equal(got[key], again[key])
            ok = err < tol and (same or (backend == 'triton' and key in ('x', 'router_w')))
            failed |= not ok
            if torch.distributed.get_rank() == 0:
                print(f"{name:8s} {key:9s} rel_err={err:.2e} run_to_run_equal={same} {'ok' if ok else 'FAIL'}")

    torch.distributed.destroy_process_group()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    fire.Fire(main)
