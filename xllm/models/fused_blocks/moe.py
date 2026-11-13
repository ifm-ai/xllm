from functools import partial
import torch
from torch.nn import functional as F
from einops import rearrange

from xllm.distributed import (
    get_model_parallel_rank,
    get_model_parallel_world_size
)
from xllm.distributed.utils import reshape_gathered_tensor_along_specific_dim
from xllm.modules.fused_ops import (
    memory_efficient_dropout_fwd,
    memory_efficient_dropout_bwd,
)
from xllm.modules.moe.router import calc_probs
from xllm.modules.moe.dedup import masked_gather_sum
from xllm.modules.moe.dispatch import moe_overlap
from .utils import (
    grouped_experts_fwd,
    grouped_experts_bwd,
)
from .distributed import (
    all_gather,
    all_reduce,
)


def moe_fwd(
    xf, router_w, router_bias, routing_score_func, routing_scaling_factor, router_load_balancing_type,
    fc1_w, fc2_w, fc3_w, n_experts, dispatcher, topk,
    expert_w1, expert_w2, expert_w3, expert_backend, moe_permutation_backend,
    hidden_dropout, recompute_fc1_out, recompute_fc3_out,
):
    bsz, seq_len, _ = xf.size()
    # Router logits
    # B x L x N
    routing_logits = F.linear(xf, router_w).to(torch.float32)
    routing_logits, handle_router = all_reduce(routing_logits, parallel_region='model', async_op=True)

    # Router indices
    if routing_score_func == 'softmax':
        routing_score_fn = partial(F.softmax, dim=-1, dtype=torch.float32)
    elif routing_score_func == 'sigmoid':
        routing_score_fn = torch.sigmoid
    else:
        raise ValueError(f"Unknown score function: {routing_score_func}.")

    if handle_router is not None:
        handle_router.wait()
    # B x L x N
    routing_scores = routing_score_fn(routing_logits)
    original_scores = routing_scores
    if router_bias is not None:
        routing_scores = routing_scores + router_bias.to(routing_scores)
    # B x L x K
    routing_indices = torch.topk(routing_scores, topk, dim=-1)[1]
    org_topk_scores = torch.gather(original_scores, dim=-1, index=routing_indices)
    topk_scores = org_topk_scores / org_topk_scores.sum(dim=-1, keepdim=True) if topk > 1 else org_topk_scores
    topk_scores = (topk_scores * routing_scaling_factor if routing_scaling_factor else topk_scores).to(xf)
    # B*L*K
    routing_indices_flat = routing_indices.flatten()
    # N
    tokens_per_expert = torch.bincount(routing_indices_flat, minlength=n_experts)

    if router_load_balancing_type is None:
        aux_loss = None
    else:
        # B x L x N
        probs, _ = calc_probs(original_scores, routing_score_func)
        prob_per_expert = probs.mean(dim=(0, 1))
        freq_per_expert = tokens_per_expert / (bsz * seq_len * topk)
        if router_load_balancing_type == 'dot':
            aux_loss = torch.dot(freq_per_expert, prob_per_expert) * n_experts
        elif router_load_balancing_type == 'entropy':
            aux_loss = torch.dot(freq_per_expert - (1.0 / n_experts), torch.log(prob_per_expert))
        else:
            raise ValueError(f"Unknown load balancing type: {router_load_balancing_type}.")

    rank = get_model_parallel_rank()
    world_size = get_model_parallel_world_size()
    n_local_experts = dispatcher.n_local
    plan = dispatcher.plan(routing_indices_flat, tokens_per_expert, topk_scores)
    group_sizes = plan.group_sizes

    # each token's slice goes once to every rank holding one of its experts
    overlap = moe_overlap()
    xu = dispatcher.dispatch(xf.view(bsz * seq_len, -1), plan, overlap)

    # the shared expert's input GEMMs run under the dispatch
    if fc1_w is not None:
        h1, handle_h1 = all_reduce(F.linear(xf, fc1_w), parallel_region='model', async_op=True)
    else:
        h1, handle_h1 = None, None

    if fc3_w is not None:
        h3, handle_h3 = all_reduce(F.linear(xf, fc3_w), parallel_region='model', async_op=True)
    else:
        h3, handle_h3 = None, None

    # B' x D: local tokens, rebuilt
    xu = xu.wait()
    # the routing scores of the tokens this TP group computes
    scores = dispatcher.local_scores(topk_scores, plan)
    if plan.n_pairs > 0:
        ew1 = rearrange(expert_w1, '(n s) d -> n s d', n=n_local_experts)
        ew2 = rearrange(expert_w2, '(n s) d -> n s d', n=n_local_experts)
        ew3 = rearrange(expert_w3, '(n s) d -> n s d', n=n_local_experts)
        # P x D: one row per local (token, expert) pair, in expert order
        xp = torch.index_select(xu, 0, plan.tp.pair_rows)
        del xu
        y2, _, _, _, _, expert_hidden_rng_state = grouped_experts_fwd(
            xp, ew1, ew2, ew3, group_sizes, expert_backend,
            memory_efficient=True, dropout=hidden_dropout,
        )
        # B' x D: experts' outputs scaled by their routing scores and summed per token
        pair_scores = scores.reshape(-1)[plan.tp.pair_slots]
        y2 = masked_gather_sum(y2, plan.tp.pair_table, pair_scores, moe_permutation_backend)
    else:
        y2 = xu
        expert_hidden_rng_state = None

    y2 = dispatcher.combine(y2, plan, overlap)

    if h1 is not None:
        if handle_h1 is not None:
            handle_h1.wait()
        h1s = F.silu(h1)
    else:
        h1s = None

    if h3 is not None:
        assert h1s is not None
        if handle_h3 is not None:
            handle_h3.wait()
        hidden, shared_hidden_rng_state = memory_efficient_dropout_fwd(h1s * h3, hidden_dropout, True)
        y1 = F.linear(hidden, fc2_w)
        if not recompute_fc1_out:
            h1 = torch.chunk(h1, world_size, dim=-1)[rank].contiguous()
        if not recompute_fc3_out:
            h3 = torch.chunk(h3, world_size, dim=-1)[rank].contiguous()
    else:
        assert h1 is None
        y1, shared_hidden_rng_state = None, None

    # B x L x D/TP: add up the parts returned by each rank
    y2 = y2.wait().view(bsz, seq_len, -1)

    # combine shared expert
    out = torch.add(y1, y2, out=y2) if y1 is not None else y2

    return (
        out, aux_loss, (original_scores, routing_indices, org_topk_scores, tokens_per_expert), (h1, h3),
        plan.state,
        (expert_hidden_rng_state, shared_hidden_rng_state)
    )


def moe_bwd(
    out_grad, aux_loss_grad, xf, xf_flat, original_scores, routing_indices, org_topk_scores, tokens_per_expert,
    router_w, router_bias, router_bias_update_rate, routing_score_func, routing_scaling_factor, router_load_balancing_type,
    h1, h3, fc1_w, fc2_w, fc3_w, n_experts, dispatcher, topk, expert_w1, expert_w2, expert_w3,
    expert_backend, moe_permutation_backend, hidden_dropout, dispatch_state,
    expert_hidden_rng_state, shared_hidden_rng_state, gather_before_norm,
):
    bsz, seq_len, _ = xf.size()
    n_local_experts = dispatcher.n_local
    # B*L*K
    routing_indices_flat = routing_indices.flatten()
    x_rows = xf.view(bsz * seq_len, -1)
    grad_rows = out_grad.reshape(bsz * seq_len, -1)
    # the layer input and the output gradient go to the experts again. At EP > TP their first hop, to the expert
    # group's replicas, needs only its own part of the plan: it starts first, and runs under the recompute below
    outer = dispatcher.replan_outer(routing_indices_flat, dispatch_state)
    xu = dispatcher.start_dispatch(x_rows, outer)
    yu_grad = dispatcher.start_dispatch(grad_rows, outer)

    # recompute router
    if original_scores is None:
        routing_logits = F.linear(xf, router_w).to(torch.float32)
        routing_logits, handle_router = all_reduce(routing_logits, parallel_region='model', async_op=True)
    else:
        routing_logits, handle_router = None, None

    # recompute h1
    if fc1_w is not None:
        if h1 is None:
            h1, handle_h1 = all_reduce(F.linear(xf, fc1_w), parallel_region='model', async_op=True)
        else:
            h1, handle_h1 = all_gather(h1, parallel_region='model', async_op=True)
    else:
        handle_h1 = None

    # recompute h3
    if fc3_w is not None:
        if h3 is None:
            h3, handle_h3 = all_reduce(F.linear(xf, fc3_w), parallel_region='model', async_op=True)
        else:
            h3, handle_h3 = all_gather(h3, parallel_region='model', async_op=True)
    else:
        handle_h3 = None

    if topk > 1:
        topk_scores_denom = org_topk_scores.sum(dim=-1, keepdim=True)
        topk_scores = org_topk_scores / topk_scores_denom
    else:
        topk_scores_denom = None
        topk_scores = org_topk_scores
    scaled_topk_scores = (topk_scores * routing_scaling_factor if routing_scaling_factor else topk_scores).to(xf)

    # rebuilt from the routing rather than kept from forward: cheap, and less state to save
    plan = dispatcher.replan(routing_indices_flat, dispatch_state, outer)
    group_sizes = plan.group_sizes

    # recompute xp: each token's slice goes once to every rank holding one of its experts
    overlap = moe_overlap()
    xu = dispatcher.dispatch(x_rows, plan, overlap, started=xu)

    # the output gradient goes the same way, unscaled; the owner applies the routing scores
    yu_grad = dispatcher.dispatch(grad_rows, plan, overlap, started=yu_grad)

    if original_scores is None:
        # Router indices
        if routing_score_func == 'softmax':
            routing_score_fn = partial(F.softmax, dim=-1, dtype=torch.float32)
        elif routing_score_func == 'sigmoid':
            routing_score_fn = torch.sigmoid
        else:
            raise ValueError(f"Unknown score function: {routing_score_func}.")

        if handle_router is not None:
            handle_router.wait()
        # B x L x N
        original_scores = routing_score_fn(routing_logits)

    # B x L x D/TP
    if fc1_w is not None:
        if handle_h1 is not None:
            handle_h1.wait()
            if h1.dim() == 4:
                h1 = reshape_gathered_tensor_along_specific_dim(h1, gather_dim=2)
        h1s = F.silu(h1)

        if handle_h3 is not None:
            handle_h3.wait()
            if h3.dim() == 4:
                h3 = reshape_gathered_tensor_along_specific_dim(h3, gather_dim=2)

        y1_grad = out_grad
        h, h_noise = memory_efficient_dropout_fwd(h1s * h3, hidden_dropout, True, shared_hidden_rng_state)
        h_grad = y1_grad.matmul(fc2_w)
        h_grad, handle_h = all_reduce(h_grad, parallel_region='model', async_op=True)
        h = h.flatten(end_dim=-2)
        fc2_w_grad = y1_grad.flatten(end_dim=-2).t().matmul(h)
    else:
        assert h1 is None and h3 is None
        h_grad, fc2_w_grad, h1s = None, None, None
        handle_h, h_noise = None, None

    # B' x D: local tokens, rebuilt
    xu = xu.wait()
    # the routing scores of the tokens this TP group computes; EP > TP: those received in forward, kept in
    # the plan state, which equal the recomputed ones bitwise
    scores = dispatcher.local_scores(scaled_topk_scores, plan)
    if plan.n_pairs > 0:
        ew1 = rearrange(expert_w1, '(n s) d -> n s d', n=n_local_experts)
        ew2 = rearrange(expert_w2, '(n s) d -> n s d', n=n_local_experts)
        ew3 = rearrange(expert_w3, '(n s) d -> n s d', n=n_local_experts)
        # P x D: one row per local (token, expert) pair, in expert order
        xp = torch.index_select(xu, 0, plan.tp.pair_rows)
        del xu
        # the expert outputs are not needed: their routing scores' gradient comes from the backward below
        _, eh1, eh1s, eh3, eh, expert_hidden_noise = grouped_experts_fwd(
            xp, ew1, ew2, ew3, group_sizes, expert_backend, False, hidden_dropout, expert_hidden_rng_state,
            with_output=False,
        )
    else:
        xp = xu
        eh1, eh1s, eh3, eh = None, None, None, None
        ew1, ew2, ew3 = None, None, None
        expert_hidden_noise = None

    # B' x D: output gradient of the local tokens, rebuilt
    yu_grad = yu_grad.wait()
    if plan.n_pairs > 0:
        # P x D: scaled by each pair's routing score
        pair_scores = scores.reshape(-1)[plan.tp.pairs]
        y2_grad = torch.index_select(yu_grad, 0, plan.tp.pair_rows).mul_(pair_scores.unsqueeze(1))
        del yu_grad
        # the expert weight gradients are deferred: they run under the combine below
        xp_grad, expert_wgrad, output_dots = grouped_experts_bwd(
            y2_grad, xp, ew1, ew2, ew3, eh1, eh1s, eh3, eh, group_sizes, expert_backend,
            True, hidden_dropout, expert_hidden_rng_state, expert_hidden_noise, with_output_dots=True,
            defer_wgrad=True,
        )
        # routing scores' gradient, <out_grad, expert output>, for the local pairs; other ranks add theirs
        # through the logits-gradient all-reduce below. A score that underflowed to 0 would give 0 / 0; its
        # gradient is later multiplied by that score's probability, which is 0, so the clamp only avoids NaN.
        topk_scores_grad = torch.zeros(plan.n_tokens * topk, device=xf.device, dtype=torch.float32)
        topk_scores_grad[plan.tp.pairs] = output_dots / pair_scores.float().clamp_min(torch.finfo(torch.float32).tiny)
        # B' x D: summed per local token
        xu_grad = masked_gather_sum(xp_grad, plan.tp.pair_table, None, moe_permutation_backend)
        del xp_grad
    else:
        xu_grad = yu_grad
        topk_scores_grad = torch.zeros(plan.n_tokens * topk, device=xf.device, dtype=torch.float32)
        expert_wgrad = None

    # the routing scores' gradient, back to the tokens' replica, ahead of the combine on the expert outer
    # group; waited for where it is used
    topk_scores_grad = dispatcher.scores_grad(topk_scores_grad.view(plan.n_tokens, topk), plan)
    # the input gradient from the experts, back to the tokens; the expert weight gradients and the shared
    # expert's backward run under it
    xf_grad = dispatcher.combine(xu_grad, plan, overlap)
    del xu_grad

    if expert_wgrad is not None:
        ew1_grad, ew2_grad, ew3_grad = expert_wgrad()
        # the expert recompute is no longer needed
        del expert_wgrad, y2_grad, xp, eh1, eh1s, eh3, eh, expert_hidden_noise
        ew1_grad = rearrange(ew1_grad, 'n s d -> (n s) d')
        ew2_grad = rearrange(ew2_grad, 'n s d -> (n s) d')
        ew3_grad = rearrange(ew3_grad, 'n s d -> (n s) d')
    else:
        # FSDP must reduce the same parameters even when this rank has no expert tokens.
        ew1_grad = torch.zeros_like(expert_w1)
        ew2_grad = torch.zeros_like(expert_w2)
        ew3_grad = torch.zeros_like(expert_w3)

    if fc1_w is not None:
        if handle_h is not None:
            handle_h.wait()
        h_grad = memory_efficient_dropout_bwd(h_grad, hidden_dropout, shared_hidden_rng_state, h_noise)
        h1s_grad = h_grad * h3
        h1_grad = torch.ops.aten.silu_backward(h1s_grad, h1)
        h3_grad = h_grad * h1s
        # xf_grad from shared experts
        xf_grad_shared = h1_grad.matmul(fc1_w) + h3_grad.matmul(fc3_w)
        fc1_w_grad = h1_grad.flatten(end_dim=-2).t().matmul(xf_flat)
        fc3_w_grad = h3_grad.flatten(end_dim=-2).t().matmul(xf_flat)
    else:
        xf_grad_shared = None
        fc1_w_grad, fc3_w_grad = None, None

    # router grad
    if router_load_balancing_type is None:
        assert aux_loss_grad is None
        original_scores_grad_aux = None
    else:
        # B x L x N
        probs, org_scores_denom = calc_probs(original_scores, routing_score_func)
        freq_per_expert = tokens_per_expert / (bsz * seq_len * topk)
        if router_load_balancing_type == 'dot':
            coeff = n_experts / (bsz * seq_len)
            probs_grad = (aux_loss_grad * freq_per_expert * coeff).view(1, 1, n_experts)
        elif router_load_balancing_type == 'entropy':
            prob_per_expert = probs.sum(dim=(0, 1))
            freq_per_expert = freq_per_expert - (1.0 / n_experts)
            probs_grad = (aux_loss_grad * freq_per_expert).div(prob_per_expert).view(1, 1, n_experts)
        else:
            raise ValueError(f"Unknown load balancing type: {router_load_balancing_type}.")

        if routing_score_func == 'softmax':
            original_scores_grad_aux = probs_grad
        elif routing_score_func == 'sigmoid':
            original_scores_grad_aux = (probs_grad - (probs_grad * probs).sum(dim=-1, keepdim=True)) / org_scores_denom
        else:
            raise ValueError(f"Unknown score function: {routing_score_func}.")

    # reduce tokens/expert for router bias updating
    if router_bias is not None:
        tokens_per_expert, handle_bias = all_reduce(tokens_per_expert, parallel_region='hybrid_shard', async_op=True)
    else:
        handle_bias = None

    # B x L x K
    topk_scores_grad = topk_scores_grad.wait().view(bsz, seq_len, topk).float()
    if routing_scaling_factor is not None:
        topk_scores_grad = topk_scores_grad * routing_scaling_factor

    if topk > 1:
        topk_scores_grad = (topk_scores_grad - (topk_scores_grad * topk_scores).sum(dim=-1, keepdim=True)) / topk_scores_denom

    original_scores_grad = torch.zeros_like(original_scores).scatter(-1, routing_indices, topk_scores_grad)
    if original_scores_grad_aux is not None:
        original_scores_grad = original_scores_grad + original_scores_grad_aux

    if routing_score_func == 'softmax':
        routing_logits_grad = torch.ops.aten._softmax_backward_data(
            original_scores_grad, original_scores, -1, torch.float32
        )
    elif routing_score_func == 'sigmoid':
        routing_logits_grad = torch.ops.aten.sigmoid_backward(original_scores_grad, original_scores)
    else:
        raise ValueError(f"Unknown score function: {routing_score_func}.")

    routing_logits_grad, handle_logits = all_reduce(routing_logits_grad, parallel_region='model', async_op=True)

    # xf grad from expert
    # B x L x D/TP: add up the parts returned by each rank
    xf_grad = xf_grad.wait().view(bsz, seq_len, -1)

    if xf_grad_shared is not None:
        xf_grad = torch.add(xf_grad, xf_grad_shared, out=xf_grad)

    # update router bias
    if router_bias is not None:
        if handle_bias is not None:
            handle_bias.wait()
        average_tokens = tokens_per_expert.sum() / n_experts
        bias_update = torch.sign(average_tokens - tokens_per_expert)
        router_bias.add_(bias_update, alpha=router_bias_update_rate)

    if handle_logits is not None:
        handle_logits.wait()

    routing_logits_grad = routing_logits_grad.to(xf)
    # (B x L x N) x (N x D/TP) -> B x L x D/TP
    xf_grad_router = routing_logits_grad.matmul(router_w)
    xf_grad = torch.add(xf_grad, xf_grad_router, out=xf_grad)

    if gather_before_norm:
        xf_grad, handle_xf = all_gather(xf_grad, parallel_region='model', async_op=True)
    else:
        handle_xf = None

    # B*L x N
    routing_logits_grad = routing_logits_grad.view(bsz * seq_len, -1)
    # (N x B*L) x (B*L x D/TP) -> N x D/TP
    router_w_grad = routing_logits_grad.t().matmul(xf_flat)

    if handle_xf is not None:
        handle_xf.wait()
        xf_grad = reshape_gathered_tensor_along_specific_dim(xf_grad, gather_dim=2)

    return xf_grad, ew1_grad, ew2_grad, ew3_grad, fc1_w_grad, fc2_w_grad, fc3_w_grad, router_w_grad
