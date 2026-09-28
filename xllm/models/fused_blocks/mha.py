import math
from functools import partial
import torch
from torch.nn import functional as F

from xllm.distributed import (
    get_context_parallel_rank,
    get_context_parallel_world_size,
)
from xllm.distributed.utils import reshape_gathered_tensor_along_specific_dim
from xllm.modules.fused_ops import (
    memory_efficient_dropout_fwd,
)
from .utils import (
    layer_or_rmsnorm_bwd,
    rmsnorm_fwd,
    rmsnorm_bwd,
    flash_attention_fwd,
    flash_attention_bwd,
    recompute_flash_attention_lse,
    apply_rope,
)
from .distributed import (
    reduce_scatter,
    all_gather,
)


def multihead_attention_fwd(
    mx, freqs_cis, segments, wq, wk, wv, wr, wo, q_norm_w, k_norm_w,
    head_dim, rope_head_dim, local_heads, local_kv_heads, eps,
    attn_gate_func, dropout, attention_dropout, hidden_dropout,
):
    bsz, seq_len, _ = mx.shape
    attn_gate_fn = {"silu": F.silu, "softplus": partial(F.softplus, beta=math.log(2))}[attn_gate_func]
    assert attn_gate_fn is not None

    attn_scale = 1.0 / math.sqrt(head_dim)
    end_seq = (get_context_parallel_rank() + 1) * seq_len
    # compute v
    xv = F.linear(mx, wv).view(bsz, seq_len, local_kv_heads, head_dim)
    xv_, handle_xv = all_gather(xv, parallel_region='context', async_op=True)

    # compute k
    xk = F.linear(mx, wk)
    if k_norm_w is not None:
        xk, _ = rmsnorm_fwd(xk, k_norm_w, local_kv_heads, eps)
    xk = xk.view(bsz, seq_len, local_kv_heads, head_dim)
    xk = apply_rope(xk, freqs_cis, head_dim, rope_head_dim, False)
    xk_, handle_xk = all_gather(xk, parallel_region='context', async_op=True)

    # compute q
    xq = F.linear(mx, wq)
    if q_norm_w is not None:
        xq, _ = rmsnorm_fwd(xq, q_norm_w, local_heads, eps)
    xq = xq.view(bsz, seq_len, local_heads, head_dim)
    xq = apply_rope(xq, freqs_cis, head_dim, rope_head_dim, False)

    if handle_xv is not None:
        handle_xv.wait()
        xv_ = reshape_gathered_tensor_along_specific_dim(xv_, gather_dim=1)[:, :end_seq]
    if handle_xk is not None:
        handle_xk.wait()
        xk_ = reshape_gathered_tensor_along_specific_dim(xk_, gather_dim=1)[:, :end_seq]

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
    if wr is not None:
        r = attn_gate_fn(F.linear(mx, wr, None))
        attn = torch.mul(attn, r, out=r)
    attn, attn_out_rng_state = memory_efficient_dropout_fwd(attn, hidden_dropout, True)

    xh, _ = reduce_scatter(F.linear(attn, wo, None), parallel_region='model', async_op=False)
    xh, xh_rng_state = memory_efficient_dropout_fwd(xh, dropout, True)

    return (
        (xq, xk, xv), (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, total_seqlen_k, end_seq),
        (attn_out, attn_lse), xh, (flash_rng_state, attn_rng_state, attn_out_rng_state, xh_rng_state)
    )


def multihead_attention_bwd(
    attn_grad, r_grad, attn_out, attn_lse, xq_, xk_, xv_, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    total_seqlen_k, end_seq, freqs_cis, local_kv_heads, rope_head_dim,
    xq, xk, xq_invvar, xk_invvar, rmx, mx, q_norm_w, k_norm_w, wq, wk, wv, wr,
    attn_scale, attn_gate_func, attention_dropout, flash_rng_state, deterministic,
    x_, x_mean, x_invvar, attn_norm_w, attn_norm_b, norm_groups, apply_rmsnorm, gather_before_norm,
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
    xk_grad_ = torch.cat([padded_grad[:, :pre_seq], xk_grad_, padded_grad[:, :tail_seq]], dim=1)
    xk_grad, handle_xk = reduce_scatter(xk_grad_, parallel_region='context', dim=1, async_op=True)
    xv_grad_ = torch.cat([padded_grad[:, :pre_seq], xv_grad_, padded_grad[:, :tail_seq]], dim=1)
    xv_grad, handle_xv = reduce_scatter(xv_grad_, parallel_region='context', dim=1, async_op=True)

    # compute rmx_grad
    if r_grad is not None:
        # B*L x D
        rmx_grad = attn_gate_fn_bwd(r_grad, rmx).view(bsz * seq_len, local_heads * head_dim)
        mx_grad_r = rmx_grad.matmul(wr)
    else:
        rmx_grad = None
        mx_grad_r = None

    # compute xq_grad
    xq_grad = apply_rope(xq_grad, freqs_cis, head_dim, rope_head_dim, True)
    if q_norm_w is not None:
        xq_grad = xq_grad.view(bsz, seq_len, local_heads * head_dim)
        xq_grad, q_norm_w_grad = rmsnorm_bwd(xq_grad, xq, xq_invvar, q_norm_w, local_heads)
    else:
        q_norm_w_grad = None
    # B*L x H*D/H
    xq_grad = xq_grad.view(bsz * seq_len, local_heads * head_dim)
    mx_grad = xq_grad.matmul(wq) if mx_grad_r is None else torch.addmm(mx_grad_r, xq_grad, wq, out=mx_grad_r)

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

    # compute xv_grad
    if handle_xv is not None:
        handle_xv.wait()

    xv_grad = xv_grad.view(bsz * seq_len, local_kv_heads * head_dim)
    mx_grad = torch.addmm(mx_grad, xv_grad, wv, out=mx_grad)

    # B x L x D
    mx_grad = mx_grad.view(bsz, seq_len, -1)
    handle_norm_w = None
    handle_norm_b = None
    if gather_before_norm:
        x_grad, attn_norm_w_grad, attn_norm_b_grad = layer_or_rmsnorm_bwd(
            mx_grad, x_, x_mean, x_invvar, attn_norm_w, attn_norm_b, norm_groups, apply_rmsnorm
        )
        x_grad, handle_x = reduce_scatter(x_grad, parallel_region='model', async_op=True)
        if attn_norm_w_grad is not None:
            attn_norm_w_grad, handle_norm_w = reduce_scatter(attn_norm_w_grad, parallel_region='model', async_op=True)
        if attn_norm_b_grad is not None:
            attn_norm_b_grad, handle_norm_b = reduce_scatter(attn_norm_b_grad, parallel_region='model', async_op=True)
    else:
        mx_grad, handle_x = reduce_scatter(mx_grad, parallel_region='model', async_op=True)
        x_grad, attn_norm_w_grad, attn_norm_b_grad = None, None, None

    mx_flat = mx.flatten(end_dim=-2)
    wq_grad = xq_grad.t().matmul(mx_flat)
    wk_grad = xk_grad.t().matmul(mx_flat)
    wv_grad = xv_grad.t().matmul(mx_flat)
    wr_grad = None if rmx_grad is None else rmx_grad.t().matmul(mx_flat)

    if handle_x is not None:
        handle_x.wait()
    if handle_norm_w is not None:
        handle_norm_w.wait()
    if handle_norm_b is not None:
        handle_norm_b.wait()

    if not gather_before_norm:
        x_grad, attn_norm_w_grad, attn_norm_b_grad = layer_or_rmsnorm_bwd(
            mx_grad, x_, x_mean, x_invvar, attn_norm_w, attn_norm_b, norm_groups, apply_rmsnorm
        )

    return x_grad, wq_grad, wk_grad, wv_grad, wr_grad, q_norm_w_grad, k_norm_w_grad, attn_norm_w_grad, attn_norm_b_grad


def multihead_attention_recompute(
    attn_out, attn_lse, xq, xk, xv, total_seqlen_k, end_seq,
    mx, freqs_cis, wq, wk, wv, wr, q_norm_w, k_norm_w,
    head_dim, rope_head_dim, local_heads, local_kv_heads, eps,
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    attn_scale, attn_gate_func, attention_dropout, attn_rng_state,
):
    def recompute_xq(cq):
        if cq is None:
            cq = F.linear(mx, wq)
            if q_norm_w is not None:
                cq_, cq_invvar = rmsnorm_fwd(cq, q_norm_w, local_heads, eps)
            else:
                cq_ = cq
                cq_invvar = None
            cq_ = cq_.view(bsz, seq_len, local_heads, head_dim)
            cq_ = apply_rope(cq_, freqs_cis, head_dim, rope_head_dim, False)
        else:
            cq_ = cq
            cq_invvar = None

        return cq, cq_, cq_invvar

    bsz, seq_len, _ = mx.shape
    attn_gate_fn = {"silu": F.silu, "softplus": partial(F.softplus, beta=math.log(2))}[attn_gate_func]

    # recompute kv
    if xv is None:
        # compute v
        xv = F.linear(mx, wv).view(bsz, seq_len, local_kv_heads, head_dim)
        xv_, handle_xv = all_gather(xv, parallel_region='context', async_op=True)

        # compute k
        xk = F.linear(mx, wk)
        if k_norm_w is not None:
            xk_, xk_invvar = rmsnorm_fwd(xk, k_norm_w, local_kv_heads, eps)
        else:
            xk_ = xk
            xk_invvar = None
        xk_ = xk_.view(bsz, seq_len, local_kv_heads, head_dim)
        xk_ = apply_rope(xk_, freqs_cis, head_dim, rope_head_dim, False)
        xk_, handle_xk = all_gather(xk_, parallel_region='context', async_op=True)
        # compute q
        xq, xq_, xq_invvar = recompute_xq(xq)
    else:
        xv_, handle_xv = all_gather(xv, parallel_region='context', async_op=True)
        xk_, handle_xk = all_gather(xk, parallel_region='context', async_op=True)
        xk_invvar = None
        # recompute q
        xq, xq_, xq_invvar = recompute_xq(xq)

    if wr is not None:
        # B x L x D
        rmx = F.linear(mx, wr, None)
        r = attn_gate_fn(rmx)
    else:
        rmx, r = None, None

    # recompute attention output
    if attn_out is None:
        assert attn_lse is None

        if handle_xv is not None:
            handle_xv.wait()
            handle_xv = None
            xv_ = reshape_gathered_tensor_along_specific_dim(xv_, gather_dim=1)[:, :end_seq]
        if handle_xk is not None:
            handle_xk.wait()
            handle_xk = None
            xk_ = reshape_gathered_tensor_along_specific_dim(xk_, gather_dim=1)[:, :end_seq]

        if total_seqlen_k is not None:
            xk_ = xk_[:, -total_seqlen_k:]
            xv_ = xv_[:, -total_seqlen_k:]

        attn_out, attn_lse, _ = recompute_flash_attention_lse(
            xq_, xk_, xv_, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, attn_scale, attention_dropout, attn_rng_state
        )

    return (
        (attn_out, attn_lse), (xq, xq_, xq_invvar), (xk, xk_, xk_invvar), (xv, xv_), (handle_xk, handle_xv), (rmx, r)
    )
