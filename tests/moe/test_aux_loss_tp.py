"""Check that the load-balancing aux loss reaches the router weights once, whatever the TP size.

Runs the fused and the unfused MoE with each aux loss type and compares the router-weight gradient with an fp32
calculation of the same layer. The aux gradient is large so that counting it TP times cannot pass. Launch one
process per TP rank:

    torchrun --nproc_per_node 8 tests/moe/test_aux_loss_tp.py
"""
import os

import fire
import torch
import torch.nn.functional as F

from xllm.distributed import initialize_model_parallel, get_model_parallel_group, get_model_parallel_rank
from xllm.models.fused_blocks.moe import moe_fwd, moe_bwd
from xllm.modules.moe.moe import NormalizedMoE
from xllm.modules.moe.router import calc_probs

SCALING_FACTOR = 2.5


def reference_router_grad(x, out_grad, w, idx, aux_type, aux_grad, n_experts, topk):
    """fp32 router-weight gradient of <experts' output, out_grad> + aux_grad * aux, routing fixed to idx."""
    tokens = x.size(0)
    inter_dim = w['w1'].size(0) // n_experts
    x, out_grad = x.float(), out_grad.float()
    router_w = w['router_w'].float().requires_grad_()
    scores = torch.sigmoid(x @ router_w.t())
    topk_scores = torch.gather(scores, -1, idx)
    topk_scores = topk_scores / topk_scores.sum(-1, keepdim=True) * SCALING_FACTOR
    y = torch.zeros_like(x)
    for e in range(n_experts):
        t, k = (idx == e).nonzero(as_tuple=True)
        if t.numel() == 0:
            continue
        rows = slice(e * inter_dim, (e + 1) * inter_dim)
        h = F.silu(x[t] @ w['w1'][rows].float().t()) * (x[t] @ w['w3'][rows].float().t())
        y = y.index_add(0, t, topk_scores[t, k, None] * (h @ w['w2'][e * x.size(1):(e + 1) * x.size(1)].float().t()))
    probs, _ = calc_probs(scores.unsqueeze(0), 'sigmoid')
    prob_per_expert = probs.mean(dim=(0, 1))
    freq_per_expert = torch.bincount(idx.flatten(), minlength=n_experts).float() / (tokens * topk)
    if aux_type == 'dot':
        aux = torch.dot(freq_per_expert, prob_per_expert) * n_experts
    else:
        aux = torch.dot(freq_per_expert - 1.0 / n_experts, torch.log(prob_per_expert))
    ((y * out_grad).sum() + aux_grad * aux).backward()
    return router_w.grad


def gather_cols(t):
    parts = [torch.empty_like(t) for _ in range(torch.distributed.get_world_size(get_model_parallel_group()))]
    torch.distributed.all_gather(parts, t.contiguous(), group=get_model_parallel_group())
    return torch.cat(parts, -1)


def run_fused(xf, og, w, aux_type, aux_grad, n_experts, n_local, topk, backend):
    rank = get_model_parallel_rank()
    out, aux, routing_out, _, routing_state, rng_states = moe_fwd(
        xf, w['router_w'], None, 'sigmoid', SCALING_FACTOR, aux_type, None, None, None,
        n_experts, n_local, rank * n_local, (rank + 1) * n_local, topk, w['w1'], w['w2'], w['w3'],
        'sequential', backend, 0.0, False, False,
    )
    grads = moe_bwd(
        og, torch.tensor(aux_grad, device=xf.device), xf, xf.view(-1, xf.size(-1)), *routing_out,
        w['router_w'], None, None, 'sigmoid', SCALING_FACTOR, aux_type, None, None, None, None, None,
        n_experts, n_local, topk, w['w1'], w['w2'], w['w3'], 'sequential', backend, 0.0,
        *routing_state, *rng_states, False,
    )
    return grads[7], routing_out[1]


def run_unfused(xf, og, w, aux_type, aux_grad, n_experts, n_local, topk, backend):
    model_dim = w['w1'].size(1)
    inter_dim = w['w1'].size(0) // n_local
    moe = NormalizedMoE(
        0, model_dim, inter_dim, n_experts, topk, 0, expert_backend='sequential', permutation_backend=backend,
        router_score_func='sigmoid', router_scaling_factor=SCALING_FACTOR, apply_rmsnorm=True,
    ).cuda().bfloat16()
    # the router is under test: feed the experts the layer input as is
    moe.norm = torch.nn.Identity()
    moe.norm.gather_input = False
    with torch.no_grad():
        moe.router.weight.copy_(w['router_w'])
        moe.experts.weight1.copy_(w['w1'])
        moe.experts.weight2.copy_(w['w2'])
        moe.experts.weight3.copy_(w['w3'])
    captured = {}
    moe.router.register_forward_hook(lambda _, __, out: captured.update(indices=out[1]))
    out, aux = moe(xf, aux_type)
    ((out.float() * og.float()).sum() + aux_grad * aux).backward()
    return moe.router.weight.grad, captured['indices']


def main(
    model_dim: int = 512,
    n_experts: int = 32,
    inter_dim: int = 64,
    topk: int = 8,
    tokens: int = 512,
    aux_grad: float = 1e4,
    backend: str = 'torch',
    tol: float = 2e-2,
):
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.distributed.init_process_group('nccl')
    world_size = torch.distributed.get_world_size()
    initialize_model_parallel(world_size, 1, timeout=600)
    rank = get_model_parallel_rank()
    dim, n_local = model_dim // world_size, n_experts // world_size
    cols = slice(rank * dim, (rank + 1) * dim)

    torch.manual_seed(0)

    def randn(*shape, fan_in):
        return (torch.randn(*shape, device='cuda') / fan_in ** 0.5).bfloat16()

    x = randn(1, tokens, model_dim, fan_in=1)
    out_grad = randn(1, tokens, model_dim, fan_in=1)
    full = {
        'router_w': randn(n_experts, model_dim, fan_in=model_dim),
        'w1': randn(n_experts * inter_dim, model_dim, fan_in=model_dim),
        'w2': randn(n_experts * model_dim, inter_dim, fan_in=inter_dim),
        'w3': randn(n_experts * inter_dim, model_dim, fan_in=model_dim),
    }
    local = {k: v.view(n_experts, -1, v.size(-1))[rank * n_local:(rank + 1) * n_local].reshape(-1, v.size(-1))
             for k, v in full.items() if k != 'router_w'}
    local['router_w'] = full['router_w'][:, cols]
    local = {k: v.contiguous() for k, v in local.items()}
    xf, og = x[..., cols].contiguous(), out_grad[..., cols].contiguous()

    failed = False
    for name, run in (('fused', run_fused), ('unfused', run_unfused)):
        for aux_type in ('dot', 'entropy'):
            router_grad, indices = run(xf, og, local, aux_type, aux_grad, n_experts, n_local, topk, backend)
            router_grad = gather_cols(router_grad)
            if torch.distributed.get_rank() == 0:
                ref = reference_router_grad(
                    x.view(tokens, -1), out_grad.view(tokens, -1), full, indices.view(tokens, topk),
                    aux_type, aux_grad, n_experts, topk,
                )
                err = ((router_grad.float() - ref).abs().max() / ref.abs().max()).item()
                ok = err < tol
                failed |= not ok
                print(f"tp={world_size} {name:8s} {aux_type:8s} router_w rel_err={err:.2e} {'ok' if ok else 'FAIL'}")
    torch.distributed.destroy_process_group()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    fire.Fire(main)
