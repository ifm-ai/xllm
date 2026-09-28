from functools import partial
import math
import torch
from torch.nn import functional as F
from einops import rearrange

from xllm.distributed import (
    get_context_parallel_rank,
    get_context_parallel_world_size,
)
from xllm.distributed.utils import reshape_gathered_tensor_along_specific_dim
from xllm.modules.fused_ops import (
    memory_efficient_dropout_fwd,
    multi_group_matmul_fwd,
    multi_group_matmul_bwd,
)
from xllm.modules.moe.router import calc_probs
from .utils import (
    layer_or_rmsnorm_bwd,
    rmsnorm_fwd,
    rmsnorm_bwd,
    fused_permute_y_fwd,
    fused_permute_y_bwd_y_grad,
    fused_permute_y_bwd_scores_grad,
    flash_attention_fwd,
    flash_attention_bwd,
    recompute_flash_attention_lse,
    apply_rope,
)
from .distributed import (
    reduce_scatter,
    all_gather,
    scatter,
    all_reduce,
)


def mova_fwd(
    mx, freqs_cis, segments, wq, wk, wv, wr, wo, q_norm_w, k_norm_w,
    head_dim, rope_head_dim, local_heads, local_kv_heads, eps, gather_before_norm,
    router_w, router_bias, routing_score_func, routing_scaling_factor, router_load_balancing_type,
    n_values, topk, value_backend, moe_permutation_backend, attn_gate_func,
    dropout, attention_dropout, hidden_dropout,
):
    bsz, seq_len, _ = mx.shape
    attn_gate_fn = {"silu": F.silu, "softplus": partial(F.softplus, beta=math.log(2))}[attn_gate_func]
    assert attn_gate_fn is not None

    if gather_before_norm:
        sx = scatter(mx, parallel_region='model')
        handle_mx = None
    else:
        sx = mx
        mx, handle_mx = all_gather(mx, parallel_region='model', async_op=True)

    # router
    # B x L x N
    routing_logits = F.linear(sx, router_w).to(torch.float32)
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
    topk_scores = (topk_scores * routing_scaling_factor if routing_scaling_factor else topk_scores).to(sx)
    # B*L*K
    routing_indices_flat = routing_indices.flatten()
    # N
    tokens_per_expert = torch.bincount(routing_indices_flat, minlength=n_values)

    if router_load_balancing_type is None:
        aux_loss = None
    else:
        # B x L x N
        probs, _ = calc_probs(original_scores, routing_score_func)
        prob_per_expert = probs.mean(dim=(0, 1))
        freq_per_expert = tokens_per_expert / (bsz * seq_len * topk)
        if router_load_balancing_type == 'dot':
            aux_loss = torch.dot(freq_per_expert, prob_per_expert) * n_values
        elif router_load_balancing_type == 'entropy':
            aux_loss = torch.dot(freq_per_expert - (1.0 / n_values), torch.log(prob_per_expert))
        else:
            raise ValueError(f"Unknown load balancing type: {router_load_balancing_type}.")

    # B*L*K
    sorted_indices = torch.argsort(routing_indices_flat, stable=True)
    permute_indices = sorted_indices // topk
    # N
    group_sizes = tokens_per_expert.tolist()
    # B*L*K x D/TP
    permuted_sx = torch.index_select(sx.view(bsz * seq_len, -1), 0, permute_indices)
    # B*L*K x V
    wv = rearrange(wv, '(n v) d -> n v d', n=n_values)
    xv = multi_group_matmul_fwd(permuted_sx, wv, group_sizes, True, value_backend)
    # B*L*K x V/TP
    xv, handle_xv = reduce_scatter(xv, parallel_region='model', async_op=True)

    if handle_mx is not None:
        handle_mx.wait()
        mx = reshape_gathered_tensor_along_specific_dim(mx, gather_dim=2)

    attn_scale = 1.0 / math.sqrt(head_dim)
    end_seq = (get_context_parallel_rank() + 1) * seq_len
    # compute k
    xk = F.linear(mx, wk)
    if k_norm_w is not None:
        xk, _ = rmsnorm_fwd(xk, k_norm_w, local_kv_heads, eps)
    xk = xk.view(bsz, seq_len, local_kv_heads, head_dim)
    xk = apply_rope(xk, freqs_cis, head_dim, rope_head_dim, False)
    xk_, handle_xk = all_gather(xk, parallel_region='context', async_op=True)

    # compute r
    r = attn_gate_fn(F.linear(mx, wr, None)) if wr is not None else None

    # compute v
    if handle_xv is not None:
        handle_xv.wait()

    # permute xv
    xv_, _ = fused_permute_y_fwd(
        sorted_indices, F.silu(xv), topk_scores, bsz, seq_len, topk, moe_permutation_backend,
    )
    # B x L x H x D
    xv_ = xv_.view(bsz, seq_len, local_kv_heads, head_dim)
    xv_, handle_xv = all_gather(xv_, parallel_region='context', async_op=True)

    # compute q
    xq = F.linear(mx, wq)
    if q_norm_w is not None:
        xq, _ = rmsnorm_fwd(xq, q_norm_w, local_heads, eps)
    xq = xq.view(bsz, seq_len, local_heads, head_dim)
    xq = apply_rope(xq, freqs_cis, head_dim, rope_head_dim, False)

    if handle_xk is not None:
        handle_xk.wait()
        xk_ = reshape_gathered_tensor_along_specific_dim(xk_, gather_dim=1)[:, :end_seq]
    if handle_xv is not None:
        handle_xv.wait()
        xv_ = reshape_gathered_tensor_along_specific_dim(xv_, gather_dim=1)[:, :end_seq]

    if segments is not None:
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, total_seqlen_k = segments
        xk_ = xk_[:, -total_seqlen_k:]
        xv_ = xv_[:, -total_seqlen_k:]
    else:
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, total_seqlen_k = None, None, None, None, None

    attn_out, attn_lse, flash_rng_state, attn_rng_state = flash_attention_fwd(
        xq, xk_, xv_, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, attn_scale, attention_dropout
    )

    # B x L x H x D/H -> B x L x D
    attn = attn_out.view(bsz, seq_len, -1)
    if r is not None:
        attn = torch.mul(attn, r, out=r)
    attn, attn_out_rng_state = memory_efficient_dropout_fwd(attn, hidden_dropout, True)

    xh, _ = reduce_scatter(F.linear(attn, wo, None), parallel_region='model', async_op=False)
    xh, xh_rng_state = memory_efficient_dropout_fwd(xh, dropout, True)

    return (
        (xq, xk, xv), (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, total_seqlen_k, end_seq),
        (attn_out, attn_lse), xh, aux_loss, (flash_rng_state, attn_rng_state, attn_out_rng_state, xh_rng_state),
        (original_scores, routing_indices, org_topk_scores, tokens_per_expert, group_sizes)
    )


def mova_bwd(
    attn_grad, r_grad, aux_loss_grad, attn_out, attn_lse, xq_, xk_, xv_,
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, total_seqlen_k, end_seq, freqs_cis,
    local_kv_heads, rope_head_dim, xq, xk, xv, xvv_, xq_invvar, xk_invvar, rmx, mx,
    q_norm_w, k_norm_w, wq, wk, wv, wr, attn_scale, attn_gate_func, attention_dropout, flash_rng_state, deterministic,
    sx, permuted_sx, original_scores, routing_indices, org_topk_scores, tokens_per_expert,
    sorted_indices, inverse_indices, group_sizes, router_w, router_bias, router_bias_update_rate,
    routing_score_func, routing_scaling_factor, router_load_balancing_type, n_values, topk,
    value_backend, moe_permutation_backend, x_, x_mean, x_invvar,
    attn_norm_w, attn_norm_b, norm_groups, apply_rmsnorm, gather_before_norm,
):
    # B x L x D -> B x L x H x D/H
    bsz, seq_len, local_heads, head_dim = attn_grad.shape
    attn_gate_fn_bwd = {
        "silu": torch.ops.aten.silu_backward,
        "softplus": partial(torch.ops.aten.softplus_backward, beta=math.log(2), threshold=20)
    }[attn_gate_func]

    xq_grad, xk_grad_, xv_grad_ = flash_attention_bwd(
        attn_grad, attn_out, xq_, xk_, xv_, attn_lse, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        attn_scale, attention_dropout, flash_rng_state, deterministic
    )

    tail_seq = get_context_parallel_world_size() * seq_len - end_seq
    pre_seq = 0 if total_seqlen_k is None else end_seq - total_seqlen_k
    padded_grad = torch.zeros(bsz, max(pre_seq, tail_seq), local_kv_heads, head_dim, device=xk_grad_.device, dtype=xk_grad_.dtype)
    xv_grad_ = torch.cat([padded_grad[:, :pre_seq], xv_grad_, padded_grad[:, :tail_seq]], dim=1)
    xv_grad_, handle_xv = reduce_scatter(xv_grad_, parallel_region='context', dim=1, async_op=True)
    xk_grad_ = torch.cat([padded_grad[:, :pre_seq], xk_grad_, padded_grad[:, :tail_seq]], dim=1)
    xk_grad, handle_xk = reduce_scatter(xk_grad_, parallel_region='context', dim=1, async_op=True)

    if topk > 1:
        topk_scores_denom = org_topk_scores.sum(dim=-1, keepdim=True)
        topk_scores = org_topk_scores / topk_scores_denom
    else:
        topk_scores_denom = None
        topk_scores = org_topk_scores
    scaled_topk_scores = (topk_scores * routing_scaling_factor if routing_scaling_factor else topk_scores).to(mx)

    # compute xq_grad
    xq_grad = apply_rope(xq_grad, freqs_cis, head_dim, rope_head_dim, True)
    if q_norm_w is not None:
        xq_grad = xq_grad.view(bsz, seq_len, local_heads * head_dim)
        xq_grad, q_norm_w_grad = rmsnorm_bwd(xq_grad, xq, xq_invvar, q_norm_w, local_heads)
    else:
        q_norm_w_grad = None
    # B*L x H*D/H
    xq_grad = xq_grad.view(bsz * seq_len, local_heads * head_dim)
    mx_grad = xq_grad.matmul(wq)

    # compute xv_grad
    if handle_xv is not None:
        handle_xv.wait()
    # B x L x D/TP
    xv_grad_ = xv_grad_.view(bsz, seq_len, -1)
    xvv_grad_, _ = fused_permute_y_bwd_y_grad(
        xv_grad_, sorted_indices, scaled_topk_scores, bsz, seq_len, topk, moe_permutation_backend
    )
    xv_grad = torch.ops.aten.silu_backward(xvv_grad_, xv)
    # B*L*K x D
    xv_grad, handle_xv = all_gather(xv_grad, parallel_region='model', async_op=True)

    # compute rmx_grad
    if r_grad is not None:
        # B*L x D
        rmx_grad = attn_gate_fn_bwd(r_grad, rmx).view(bsz * seq_len, local_heads * head_dim)
        mx_grad = torch.addmm(mx_grad, rmx_grad, wr, out=mx_grad)
    else:
        rmx_grad = None

    # compute xk_grad
    if handle_xk is not None:
        handle_xk.wait()
    xk_grad = apply_rope(xk_grad, freqs_cis, head_dim, rope_head_dim, True)
    if k_norm_w is not None:
        xk_grad = xk_grad.view(bsz, seq_len, local_kv_heads * head_dim)
        xk_grad, k_norm_w_grad = rmsnorm_bwd(xk_grad, xk, xk_invvar, k_norm_w, local_kv_heads)
    else:
        k_norm_w_grad = None
    # B*L x H*D/H
    xk_grad = xk_grad.view(bsz * seq_len, local_kv_heads * head_dim)
    mx_grad = torch.addmm(mx_grad, xk_grad, wk, out=mx_grad)

    # router grad
    topk_scores_grad = fused_permute_y_bwd_scores_grad(
        xv_grad_, inverse_indices, xvv_, bsz, seq_len, topk, moe_permutation_backend
    )

    if router_load_balancing_type is None:
        assert aux_loss_grad is None
        original_scores_grad_aux = None
    else:
        # B x L x N
        probs, org_scores_denom = calc_probs(original_scores, routing_score_func)
        freq_per_expert = tokens_per_expert / (bsz * seq_len * topk)
        if router_load_balancing_type == 'dot':
            coeff = n_values / (bsz * seq_len)
            probs_grad = (aux_loss_grad * freq_per_expert * coeff).view(1, 1, n_values)
        elif router_load_balancing_type == 'entropy':
            prob_per_expert = probs.sum(dim=(0, 1))
            freq_per_expert = freq_per_expert - (1.0 / n_values)
            probs_grad = (aux_loss_grad * freq_per_expert).div(prob_per_expert).view(1, 1, n_values)
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

    topk_scores_grad = topk_scores_grad.float()
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

    if handle_xv is not None:
        handle_xv.wait()
        xv_grad = reshape_gathered_tensor_along_specific_dim(xv_grad, gather_dim=1)
    # B*L*K x D/TP
    wv = rearrange(wv, '(n v) d -> n v d', n=n_values)
    psx_grad, wv_grad = multi_group_matmul_bwd(xv_grad, wv, permuted_sx, group_sizes, True, value_backend)
    wv_grad = rearrange(wv_grad, 'n v d -> (n v) d')
    sx_grad = torch.index_select(psx_grad, 0, inverse_indices)
    # B x L x K x D/TP -> B x L x D/TP
    sx_grad = sx_grad.view(bsz, seq_len, topk, -1).sum(2)

    # update router bias
    if router_bias is not None:
        if handle_bias is not None:
            handle_bias.wait()
        average_tokens = tokens_per_expert.sum() / n_values
        bias_update = torch.sign(average_tokens - tokens_per_expert)
        router_bias.add_(bias_update, alpha=router_bias_update_rate)

    if handle_logits is not None:
        handle_logits.wait()
    routing_logits_grad = routing_logits_grad.to(sx)
    # (B x L x N) x (N x D/TP) -> B x L x D/TP
    sx_grad_router = routing_logits_grad.matmul(router_w)
    sx_grad = torch.add(sx_grad, sx_grad_router, out=sx_grad)

    # B x L x D
    mx_grad = mx_grad.view(bsz, seq_len, -1)
    handle_norm_w = None
    handle_norm_b = None
    if gather_before_norm:
        sx_grad, handle_smx = all_gather(sx_grad, parallel_region='model', async_op=True)
    else:
        mx_grad, handle_smx = reduce_scatter(mx_grad, parallel_region='model', async_op=True)

    mx_flat = mx.flatten(end_dim=-2)
    wq_grad = xq_grad.t().matmul(mx_flat)
    wr_grad = None if rmx_grad is None else rmx_grad.t().matmul(mx_flat)

    if handle_smx is not None:
        handle_smx.wait()
        if gather_before_norm:
            sx_grad = reshape_gathered_tensor_along_specific_dim(sx_grad, gather_dim=2)

    mx_grad = torch.add(mx_grad, sx_grad, out=mx_grad)
    x_grad, attn_norm_w_grad, attn_norm_b_grad = layer_or_rmsnorm_bwd(
        mx_grad, x_, x_mean, x_invvar, attn_norm_w, attn_norm_b, norm_groups, apply_rmsnorm
    )

    if gather_before_norm:
        x_grad, handle_x = reduce_scatter(x_grad, parallel_region='model', async_op=True)
        if attn_norm_w_grad is not None:
            attn_norm_w_grad, handle_norm_w = reduce_scatter(attn_norm_w_grad, parallel_region='model', async_op=True)
        if attn_norm_b_grad is not None:
            attn_norm_b_grad, handle_norm_b = reduce_scatter(attn_norm_b_grad, parallel_region='model', async_op=True)
    else:
        handle_x, handle_norm_w, handle_norm_b = None, None, None

    wk_grad = xk_grad.t().matmul(mx_flat)
    # B*L x N
    routing_logits_grad = routing_logits_grad.view(bsz * seq_len, -1)
    # (N x B*L) x (B*L x D/TP) -> N x D/TP
    router_w_grad = routing_logits_grad.t().matmul(sx.flatten(end_dim=-2))

    if handle_x is not None:
        handle_x.wait()
    if handle_norm_w is not None:
        handle_norm_w.wait()
    if handle_norm_b is not None:
        handle_norm_b.wait()

    return x_grad, wq_grad, wk_grad, wv_grad, wr_grad, q_norm_w_grad, k_norm_w_grad, attn_norm_w_grad, attn_norm_b_grad, router_w_grad


def mova_recompute(
    attn_out, attn_lse, xq, xk, xv, total_seqlen_k, end_seq,
    sx, original_scores, routing_indices, org_topk_scores, group_sizes,
    router_w, routing_score_func, routing_scaling_factor, n_values, topk,
    value_backend, moe_permutation_backend, mx, freqs_cis,
    wq, wk, wv, wr, q_norm_w, k_norm_w, head_dim, rope_head_dim,
    local_heads, local_kv_heads, eps,
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    attn_scale, attn_gate_func, attention_dropout, attn_rng_state,
):
    def recompute_qk(cq, wc, norm_w, n_heads):
        if cq is None:
            cq = F.linear(mx, wc)
            if norm_w is not None:
                cq_, cq_invvar = rmsnorm_fwd(cq, norm_w, n_heads, eps)
            else:
                cq_ = cq
                cq_invvar = None
            cq_ = cq_.view(bsz, seq_len, n_heads, head_dim)
            cq_ = apply_rope(cq_, freqs_cis, head_dim, rope_head_dim, False)
        else:
            cq_ = cq
            cq_invvar = None

        return cq, cq_, cq_invvar

    bsz, seq_len, _ = mx.shape
    attn_gate_fn = {"silu": F.silu, "softplus": partial(F.softplus, beta=math.log(2))}[attn_gate_func]

    # recompute router
    if original_scores is None:
        # B x L x N
        routing_logits = F.linear(sx, router_w).to(torch.float32)
        routing_logits, handle_router = all_reduce(routing_logits, parallel_region='model', async_op=True)
    else:
        routing_logits, handle_router = None, None

    topk_scores = org_topk_scores / org_topk_scores.sum(dim=-1, keepdim=True) if topk > 1 else org_topk_scores
    topk_scores = (topk_scores * routing_scaling_factor if routing_scaling_factor else topk_scores).to(sx)

    # B*L*K
    sorted_indices = torch.argsort(routing_indices.flatten(), stable=True)
    # recompute psx
    # B*L*K x D/TP
    permute_indices = sorted_indices // topk
    permuted_sx = torch.index_select(sx.view(bsz * seq_len, -1), 0, permute_indices)

    # recompute kv
    if xv is None:
        # compute v
        wv = rearrange(wv, '(n v) d -> n v d', n=n_values)
        xv = multi_group_matmul_fwd(permuted_sx, wv, group_sizes, True, value_backend)
        # B*L*K x V/TP
        xv, handle_xv = reduce_scatter(xv, parallel_region='model', async_op=True)
    else:
        handle_xv = None

    # compute k
    xk, xk_, xk_invvar = recompute_qk(xk, wk, k_norm_w, local_kv_heads)
    xk_, handle_xk = all_gather(xk_, parallel_region='context', async_op=True)

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

    # compute v, cont
    if handle_xv is not None:
        handle_xv.wait()

    xvv_ = F.silu(xv)
    # permute xv
    xv_, inverse_indices = fused_permute_y_fwd(
        sorted_indices, xvv_, topk_scores, bsz, seq_len, topk, moe_permutation_backend,
    )
    # B x L x H x D
    xv_ = xv_.view(bsz, seq_len, local_kv_heads, head_dim)
    xv_, handle_xv = all_gather(xv_, parallel_region='context', async_op=True)

    # compute q
    xq, xq_, xq_invvar = recompute_qk(xq, wq, q_norm_w, local_heads)
    if wr is not None:
        # B x L x D
        rmx = F.linear(mx, wr, None)
        r = attn_gate_fn(rmx)
    else:
        rmx, r = None, None

    # recompute attention output
    if attn_out is None:
        assert attn_lse is None

        if handle_xk is not None:
            handle_xk.wait()
            handle_xk = None
            xk_ = reshape_gathered_tensor_along_specific_dim(xk_, gather_dim=1)[:, :end_seq]

        if handle_xv is not None:
            handle_xv.wait()
            handle_xv = None
            xv_ = reshape_gathered_tensor_along_specific_dim(xv_, gather_dim=1)[:, :end_seq]

        if total_seqlen_k is not None:
            xk_ = xk_[:, -total_seqlen_k:]
            xv_ = xv_[:, -total_seqlen_k:]

        attn_out, attn_lse, _ = recompute_flash_attention_lse(
            xq_, xk_, xv_, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, attn_scale, attention_dropout, attn_rng_state
        )

    return (
        (attn_out, attn_lse), (xq, xq_, xq_invvar), (xk, xk_, xk_invvar), (xv, xvv_, xv_), (handle_xk, handle_xv),
        (rmx, r), (permuted_sx, original_scores, sorted_indices, inverse_indices)
    )
