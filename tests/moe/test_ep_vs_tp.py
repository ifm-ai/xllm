"""Compare the MoE layer at EP > TP with the same layer at EP = TP, rank by rank, on the paths that
tests/moe/test_ep_dispatch.py does not take.

Every (DP, CP) shard has its own tokens, of a different length per shard unless --vary False, and keeps them at
both EP. The output, the input gradient, the router gradient, the aux loss ('dot') and the updated router bias of
every rank, and the routed experts' gradients summed over all GPUs by global expert index, must agree to
round-off at EP > TP and EP = TP, be finite, and be bitwise equal in two runs. Variants:

  plain      NormalizedMoE forward and backward, with the aux loss in the loss
  ckpt       the same under a non-reentrant activation checkpoint (layerwise_ckpt): backward recomputes the
             layer; bitwise equal to plain
  no_grad    NormalizedMoE forward under torch.no_grad (the output); bitwise equal to plain
  inference  the same under torch.inference_mode
  fused      moe_fwd / moe_bwd, with the aux loss

The dispatch checks its plan (NodeDispatcher.check_plan): the counts and the routing scores sent with the
routing against those read from it and sent apart. Last, one plan, without these checks, must block the host
once at both EP.

Launch one process per rank, e.g.:

    torchrun --nproc_per_node 8 tests/moe/test_ep_vs_tp.py --tp 2 --cp 2 --ep 4
    torchrun --nproc_per_node 8 tests/moe/test_ep_vs_tp.py --tp 1 --ep 8   # 4 experts per replica, topk 8
    torchrun --nproc_per_node 8 tests/moe/test_ep_vs_tp.py --tp 2 --cp 2 --ep 4 --skew

--skew routes every token to the experts of the GPU at TP rank 0 of the first replica of each expert group
(with topk <= N / EP), so the other GPUs, and the other replicas, receive no rows.
"""
import os
import warnings
from typing import Dict

import fire
import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.checkpoint import checkpoint

import xllm.distributed.initialize as parallel_state
from xllm.distributed import (
    initialize_model_parallel,
    destroy_model_parallel,
    get_model_parallel_rank,
    get_model_parallel_world_size,
)
from xllm.models.fused_blocks.moe import moe_fwd, moe_bwd
from xllm.modules.moe.dispatch import build_dispatcher, NodeDispatcher
from xllm.modules.moe.layout import ExpertLayout
from xllm.modules.moe.moe import NormalizedMoE
from tests.moe.test_dedup_dispatch import SCALING_FACTOR, EXPERT_KEYS, rel_err
from tests.moe.test_ep_dispatch import local_weights

AUX_COEFF = 0.01
BIAS_UPDATE_RATE = 1e-3
VARIANTS = ('plain', 'ckpt', 'no_grad', 'inference', 'fused')


def tp_cols(x: Tensor) -> slice:
    dim = x.size(-1) // get_model_parallel_world_size()
    rank = get_model_parallel_rank()
    return slice(rank * dim, (rank + 1) * dim)


def run_unfused(variant: str, x: Tensor, out_grad: Tensor, weights, bias, layout: ExpertLayout, topk, backend):
    cols = tp_cols(x)
    model_dim = x.size(-1)
    inter_dim = weights['w1'].size(0) // layout.n_experts
    w = local_weights(weights, layout, cols)
    moe = NormalizedMoE(
        0, model_dim, inter_dim, layout.n_experts, topk, weights['s1'].size(0) // inter_dim,
        expert_backend='sequential', permutation_backend=backend, router_score_func='sigmoid',
        router_bias=bias is not None, router_bias_update_rate=BIAS_UPDATE_RATE, router_scaling_factor=SCALING_FACTOR,
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
        for i in (1, 2, 3):
            getattr(moe, f'fc{i}').weight.copy_(w[f's{i}'])
        if bias is not None:
            moe.router.bias.copy_(bias)
    xf = x[..., cols].contiguous()
    og = out_grad[..., cols].contiguous()
    if variant in ('no_grad', 'inference'):
        with torch.no_grad() if variant == 'no_grad' else torch.inference_mode():
            out, _ = moe(xf, 'dot')
        # the layer adds its input back: compare the experts' part only
        return {'out': out.float() - xf.float()}
    xf.requires_grad_()
    if variant == 'ckpt':
        out, aux = checkpoint(moe, xf, 'dot', use_reentrant=False, preserve_rng_state=True)
    else:
        out, aux = moe(xf, 'dot')
    ((out * og).float().sum() + AUX_COEFF * aux).backward()
    got = {'out': out.detach().float() - xf.detach().float(), 'x': (xf.grad - og).float(),
           'router_w': moe.router.weight.grad.float(), 'aux': aux.detach().float().view(1)}
    for i, k in enumerate(EXPERT_KEYS, 1):
        p = getattr(moe.experts, f'weight{i}')
        got[k] = p.grad if p.grad is not None else torch.zeros_like(p)
    if bias is not None:
        got['bias'] = moe.router.bias.detach().float().clone()
    return got


def run_fused(x: Tensor, out_grad: Tensor, weights, bias, layout: ExpertLayout, topk, backend):
    cols = tp_cols(x)
    dim = cols.stop - cols.start
    w = local_weights(weights, layout, cols)
    dispatcher = build_dispatcher(layout, topk, backend)
    xf = x[..., cols].contiguous()
    b = None if bias is None else bias.clone()
    out, aux, routing_out, (h1, h3), state, rng_states = moe_fwd(
        xf, w['router_w'], b, 'sigmoid', SCALING_FACTOR, 'dot', w['s1'], w['s2'], w['s3'],
        layout.n_experts, dispatcher, topk, w['w1'], w['w2'], w['w3'], 'sequential', backend, 0.0, False, False,
    )
    original_scores, routing_indices, org_topk_scores, tokens_per_expert = routing_out
    aux_grad = torch.tensor(AUX_COEFF, device=xf.device, dtype=aux.dtype)
    xf_grad, g1, g2, g3, _, _, _, rw_grad = moe_bwd(
        out_grad[..., cols].contiguous(), aux_grad, xf, xf.view(-1, dim),
        original_scores, routing_indices, org_topk_scores, tokens_per_expert,
        w['router_w'], b, None if b is None else BIAS_UPDATE_RATE, 'sigmoid', SCALING_FACTOR, 'dot',
        h1, h3, w['s1'], w['s2'], w['s3'], layout.n_experts, dispatcher, topk, w['w1'], w['w2'], w['w3'],
        'sequential', backend, 0.0, state, *rng_states, False,
    )
    got = {'out': out.float(), 'x': xf_grad.float(), 'router_w': rw_grad.float(), 'w1': g1, 'w2': g2, 'w3': g3,
           'aux': aux.detach().float().view(1)}
    if b is not None:
        got['bias'] = b.float()
    return got


def global_expert_grads(got: Dict[str, Tensor], layout: ExpertLayout) -> Dict[str, Tensor]:
    """The routed experts' gradients by global expert index, summed over all GPUs: the same at every EP."""
    out = {}
    for k in EXPERT_KEYS:
        g = got[k]
        full = torch.zeros(layout.n_experts, g.size(0) // layout.n_local, g.size(-1), device=g.device,
                           dtype=torch.float32)
        full[layout.start:layout.end] = g.view(layout.n_local, -1, g.size(-1)).float()
        dist.all_reduce(full)
        out[k] = full
    return out


def count_syncs(fn) -> int:
    """The host syncs of fn() that torch.cuda.set_sync_debug_mode reports."""
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        torch.cuda.set_sync_debug_mode('warn')
        try:
            fn()
        finally:
            torch.cuda.set_sync_debug_mode('default')
    # not the warning that set_sync_debug_mode is a prototype
    return sum('called a synchronizing' in str(w.message) for w in caught)


def plan_syncs(x: Tensor, weights, layout: ExpertLayout, topk: int, backend: str) -> int:
    """The host syncs of one dispatcher.plan of the shard's tokens, after a first plan."""
    dispatcher = build_dispatcher(layout, topk, backend)
    logits = x.flatten(end_dim=-2).float() @ weights['router_w'].float().t()
    scores, indices = torch.sigmoid(logits).topk(topk, dim=-1)
    tokens_per_expert = torch.bincount(indices.flatten(), minlength=layout.n_experts)
    check, NodeDispatcher.check_plan = NodeDispatcher.check_plan, False
    try:
        dispatcher.plan(indices.flatten(), tokens_per_expert, scores.to(x.dtype))
        return count_syncs(lambda: dispatcher.plan(indices.flatten(), tokens_per_expert, scores.to(x.dtype)))
    finally:
        NodeDispatcher.check_plan = check


def destroy_groups():
    """Free the communicators before the next layout: there are too many to keep them all."""
    groups = {id(group): group for name in dir(parallel_state) if name.endswith('_GROUP')
              for group in [getattr(parallel_state, name)] if group is not None}
    for group in groups.values():
        dist.destroy_process_group(group)
    destroy_model_parallel()


def main(
    tp: int = 2,
    cp: int = 1,
    ep: int = 4,
    dtype: str = 'fp32',
    backend: str = 'triton',
    variants: str = ','.join(VARIANTS),
    bsz: int = 2,
    seq_len: int = 64,
    vary: bool = True,
    model_dim: int = 128,
    inter_dim: int = 32,
    shared_dim: int = 32,
    n_experts: int = 32,
    topk: int = 8,
    skew: bool = False,
    router_bias: bool = True,
    seed: int = 0,
    tol: float = None,
):
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group(backend='nccl')
    NodeDispatcher.check_plan = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    world, rank = dist.get_world_size(), dist.get_rank()
    assert ep % tp == 0 and world % (ep * cp) == 0, f"world {world}, tp {tp}, cp {cp}, ep {ep}"
    torch_dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32}[dtype]
    tol = tol if tol is not None else (3e-2 if dtype == 'bf16' else 1e-5)
    variants = variants.split(',') if isinstance(variants, str) else list(variants)
    assert set(variants) <= set(VARIANTS), variants
    assert shared_dim > 0
    if skew:
        assert topk <= n_experts // ep, "--skew needs topk <= experts per GPU"
    torch.manual_seed(seed)

    def randn(*shape, fan_in):
        return (torch.randn(*shape, device='cuda') / fan_in ** 0.5).to(torch_dtype)

    # every (DP, CP) shard's tokens, on every rank
    n_shards = world // tp
    lens = [seq_len + (8 * s if vary else 0) for s in range(n_shards)]
    xs = [randn(bsz, n, model_dim, fan_in=1) for n in lens]
    out_grads = [randn(bsz, n, model_dim, fan_in=1) for n in lens]
    weights = {
        'router_w': randn(n_experts, model_dim, fan_in=model_dim),
        'w1': randn(n_experts * inter_dim, model_dim, fan_in=model_dim),
        'w2': randn(n_experts * model_dim, inter_dim, fan_in=inter_dim),
        'w3': randn(n_experts * inter_dim, model_dim, fan_in=model_dim),
        's1': randn(shared_dim, model_dim, fan_in=model_dim),
        's2': randn(model_dim, shared_dim, fan_in=shared_dim),
        's3': randn(shared_dim, model_dim, fan_in=model_dim),
    }
    bias = None
    if skew:
        # the experts of TP rank 0, replica 0 of each expert group
        bias = torch.zeros(n_experts, device='cuda')
        bias[:n_experts // ep] = 10.0
    elif router_bias:
        bias = 0.1 * torch.randn(n_experts, device='cuda')
    x, out_grad = xs[rank // tp], out_grads[rank // tp]

    results = {}
    errors = []
    for variant in variants:
        n_errors = len(errors)
        for size in (tp, ep):
            initialize_model_parallel(tp, cp, timeout=600, expert_parallel_size=size)
            layout = ExpertLayout.from_process_groups(n_experts)
            assert (layout.tp, layout.e_out) == (tp, size // tp)
            if variant == 'fused':
                run = lambda: run_fused(x, out_grad, weights, bias, layout, topk, backend)
            else:
                run = lambda: run_unfused(variant, x, out_grad, weights, bias, layout, topk, backend)
            got = run()
            if variant in ('plain', 'ckpt', 'fused'):
                again = run()
                errors += [f"{variant} ep={size} {k}: two runs differ" for k in got
                           if not torch.equal(got[k], again[k])]
            if 'w1' in got:
                got.update(global_expert_grads(got, layout))
            errors += [f"{variant} ep={size} {k}: not finite" for k in got if not torch.isfinite(got[k]).all()]
            results[variant, size] = got
            destroy_groups()
        at_tp, at_ep = results[variant, tp], results[variant, ep]
        worst = {}
        for k in at_tp:
            worst[k] = rel_err(at_ep[k], at_tp[k]) if at_tp[k].numel() else 0.0
            if not worst[k] < tol:
                errors.append(f"{variant} {k}: rel_err(ep={ep} vs ep={tp}) {worst[k]:.2e} >= {tol:.0e}")
        # the checkpoint recomputes the same layer, and the forward does not depend on autograd
        same_as = {'ckpt': 'plain', 'no_grad': 'plain', 'inference': 'plain'}.get(variant)
        if same_as in variants:
            for size in (tp, ep):
                errors += [f"{variant} ep={size} {k}: differs from {same_as}" for k in results[variant, size]
                           if not torch.equal(results[variant, size][k], results[same_as, size][k])]
        # every rank's worst error and verdict
        flags = torch.tensor([max(worst.values()), float(len(errors) - n_errors)], device='cuda', dtype=torch.float64)
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f"{variant:9s} tp={tp} cp={cp} ep={ep} {dtype} vary={vary} skew={skew} topk={topk} N={n_experts}: "
                  f"max rel_err(ep vs tp) {flags[0].item():.2e} {'ok' if flags[1].item() == 0 else 'FAIL'}"
                  f"  rank 0: {', '.join(f'{k} {v:.1e}' for k, v in worst.items())}", flush=True)

    # one host sync per plan, at both EP
    for size in (tp, ep):
        initialize_model_parallel(tp, cp, timeout=600, expert_parallel_size=size)
        syncs = plan_syncs(x, weights, ExpertLayout.from_process_groups(n_experts), topk, backend)
        if syncs != 1:
            errors.append(f"plan ep={size}: {syncs} host syncs, not 1")
        all_syncs = torch.tensor([syncs], device='cuda')
        dist.all_reduce(all_syncs, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f"plan      ep={size}: at most {all_syncs.item()} host syncs per rank", flush=True)
        destroy_groups()

    for line in errors:
        print(f"FAIL rank {rank}: {line}", flush=True)
    failed = torch.tensor([float(len(errors))], device='cuda')
    dist.all_reduce(failed)
    dist.destroy_process_group()
    if failed.item():
        raise SystemExit(1)


if __name__ == '__main__':
    fire.Fire(main)
