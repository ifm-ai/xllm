import math
from functools import partial
import torch
from torch.nn import functional as F
from einops import rearrange

from xllm.distributed import (
    get_context_parallel_rank,
    get_context_parallel_world_size,
)
from xllm.distributed.utils import reshape_gathered_tensor_along_specific_dim
from xllm.health_metrics import get_active_health_collector
from xllm.modules.fused_ops import (
    memory_efficient_dropout_fwd,
)
from .utils import (
    layer_or_rmsnorm_bwd,
    rmsnorm_fwd,
    rmsnorm_bwd,
    flash_attention_fwd,
    flash_attention_bwd,
    recompute_flash_attention,
    softdelta_attn_fwd,
    softdelta_attn_bwd,
    recompute_softdelta_attn,
    apply_rope,
    apply_ropes
)
from .distributed import (
    reduce_scatter,
    all_gather,
)


def multihead_attention_fwd(
    layer_id, mx, freqs_cis, bos_mask, segments,
    wq, wk, wv, wr, wo, q_norm_w, k_norm_w,
    head_dim, rope_head_dim, local_heads, local_kv_heads,
    attn_act_func, apply_attn_gate, attn_gate_func, eps,
    dropout, attention_dropout, hidden_dropout,
    causal_attn_backend, stability_control, recompute_attn,
):
    bsz, seq_len, _ = mx.shape
    nk = {"softmax": 1, "softdelta": 2}[attn_act_func]
    attn_gate_fn = {"silu": F.silu, "softplus": partial(F.softplus, beta=math.log(2))}[attn_gate_func]
    assert nk is not None and attn_gate_fn is not None

    attn_scale = 1.0 / math.sqrt(head_dim)
    end_seq = (get_context_parallel_rank() + 1) * seq_len
    # compute v
    xv = F.linear(mx, wv).view(bsz, seq_len, local_kv_heads, head_dim)
    xv_, handle_xv = all_gather(xv, parallel_region='context', async_op=True)

    # compute k
    xk = F.linear(mx, wk)
    xk_ = rmsnorm_fwd(xk, k_norm_w, local_kv_heads * nk, eps)[0] if k_norm_w is not None else xk
    xk_ = rearrange(xk_, 'b l (k s) -> b l k s', k=local_kv_heads)
    if attn_act_func == 'softmax':
        xk_ = apply_rope(xk_, freqs_cis, head_dim, rope_head_dim, False)
        kq_ = None
    else:
        assert attn_act_func == 'softdelta'
        kq_ = xk_[:, :, :, :head_dim]
        xk_ = xk_[:, :, :, head_dim:]
        kq_, xk_ = apply_ropes(kq_, xk_, freqs_cis, head_dim, rope_head_dim, False)
    local_xk = xk_  # the all-gather below replaces xk_ with every context rank's keys
    xk_, handle_xk = all_gather(xk_, parallel_region='context', async_op=True)

    # compute q
    xq = F.linear(mx, wq)
    xq_ = rmsnorm_fwd(xq, q_norm_w, local_heads, eps)[0] if q_norm_w is not None else xq
    xq_ = rearrange(xq_, 'b l (h s) -> b l h s', h=local_heads)
    xq_ = apply_rope(xq_, freqs_cis, head_dim, rope_head_dim, False)

    if wr is not None:
        rg = rearrange(F.linear(mx, wr, None), 'b l (h s) -> b l h s', h=local_heads)
        # B x L x H*S
        if apply_attn_gate:
            r = rearrange(rg[:, :, :, :head_dim], 'b l h s -> b l (h s)')
            r = attn_gate_fn(r)
        else:
            r = None
        # B x L x H
        g = rg[:, :, :, -1] if attn_act_func == 'softdelta' else None
    else:
        r = g = None

    health_collector = get_active_health_collector()
    if health_collector is not None and layer_id is not None:
        # Local post-qk-norm, post-RoPE heads, before K is gathered across context ranks.
        health_collector.record_query_key_stats(layer_id, xq_, local_xk, gate=g)

    if handle_xv is not None:
        handle_xv.wait()
        xv_ = reshape_gathered_tensor_along_specific_dim(xv_, gather_dim=1)[:, :end_seq]
    if handle_xk is not None:
        handle_xk.wait()
        xk_ = reshape_gathered_tensor_along_specific_dim(xk_, gather_dim=1)[:, :end_seq]

    if attn_act_func == 'softmax':
        attn_readers = None
        attn_out, attn_lse, flash_rng_state, attn_rng_state = flash_attention_fwd(
            xq_, xk_, xv_, bos_mask, segments, attn_scale, attention_dropout, stability_control,
            backend=causal_attn_backend, save_aux=not recompute_attn
        )
    elif attn_act_func == 'softdelta':
        attn_out, attn_readers, attn_lse, flash_rng_state, attn_rng_state = softdelta_attn_fwd(
            xq_, kq_, xk_, xv_, g, bos_mask, segments, attn_scale, attention_dropout, stability_control,
            backend=causal_attn_backend, save_aux=not recompute_attn
        )
    else:
        raise NotImplementedError

    attn = rearrange(attn_out, 'b l h v -> b l (h v)')
    if r is not None:
        attn = torch.mul(attn, r, out=r)
    attn, attn_out_rng_state = memory_efficient_dropout_fwd(attn, hidden_dropout, True)

    xh, _ = reduce_scatter(F.linear(attn, wo, None), parallel_region='model', async_op=False)
    xh, xh_rng_state = memory_efficient_dropout_fwd(xh, dropout, True)

    return (
        (xq, xk, xv), (attn_out, attn_readers, attn_lse), xh,
        (flash_rng_state, attn_rng_state, attn_out_rng_state, xh_rng_state), end_seq
    )


def multihead_attention_bwd(
    attn_grad, r_grad, attn_for_bwd, attn_readers, attn_lse, xq_, kq_, xk_, xv_,
    total_seqlen_k, end_seq, freqs_cis, bos_mask, segments, local_kv_heads, rope_head_dim,
    xq, xk, xq_rstd, xk_rstd, rmx, g, mx, q_norm_w, k_norm_w, wq, wk, wv, wr,
    attn_scale, attn_act_func, apply_attn_gate, attn_gate_func, attention_dropout, flash_rng_state,
    causal_attn_backend, stability_control, deterministic,
    x_, x_mean, x_invvar, attn_norm_w, attn_norm_b, norm_groups,
    apply_rmsnorm, gather_before_norm,
):

    # B x L x D -> B x L x H x D/H
    bsz, seq_len, local_heads, head_dim = attn_grad.shape
    nk = {"softmax": 1, "softdelta": 2}[attn_act_func]
    attn_gate_fn_bwd = {
        "silu": torch.ops.aten.silu_backward,
        "softplus": partial(torch.ops.aten.softplus_backward, beta=math.log(2), threshold=20)
    }[attn_gate_func]

    if attn_act_func == 'softmax':
        xq_grad, xk_grad_, xv_grad_ = flash_attention_bwd(
            attn_grad, attn_for_bwd, xq_, xk_, xv_, attn_lse, bos_mask, segments,
            attn_scale, attention_dropout, flash_rng_state,
            stability_control, deterministic, causal_attn_backend
        )
        kq_grad = g_grad = None
    elif attn_act_func == 'softdelta':
        xq_grad, kq_grad, xk_grad_, xv_grad_, g_grad = softdelta_attn_bwd(
            attn_grad, attn_for_bwd, xq_, kq_, xk_, xv_, g, attn_readers, attn_lse,
            bos_mask, segments, attn_scale, attention_dropout, flash_rng_state,
            stability_control, deterministic, causal_attn_backend
        )
    else:
        raise NotImplementedError

    tail_seq = get_context_parallel_world_size() * seq_len - end_seq
    pre_seq = 0 if total_seqlen_k is None else end_seq - total_seqlen_k
    padded_grad = torch.zeros(bsz, max(pre_seq, tail_seq), local_kv_heads, head_dim, device=xk_grad_.device, dtype=xk_grad_.dtype)
    xk_grad_ = torch.cat([padded_grad[:, :pre_seq], xk_grad_, padded_grad[:, :tail_seq]], dim=1)
    xk_grad, handle_xk = reduce_scatter(xk_grad_, parallel_region='context', dim=1, async_op=True)
    xv_grad_ = torch.cat([padded_grad[:, :pre_seq], xv_grad_, padded_grad[:, :tail_seq]], dim=1)
    xv_grad, handle_xv = reduce_scatter(xv_grad_, parallel_region='context', dim=1, async_op=True)

    # compute rmx_grad
    # B x L x H
    if apply_attn_gate:
        # B x L x H x S
        r_grad = rearrange(r_grad, 'b l (h s) -> b l h s', h=local_heads)
        r_grad = attn_gate_fn_bwd(r_grad, rmx)
        rmx_grad = r_grad if g_grad is None else torch.cat([r_grad, g_grad[..., None]], dim=-1)
    else:
        rmx_grad = g_grad

    if rmx_grad is not None:
        rmx_grad = rmx_grad.view(bsz * seq_len, -1)
        mx_grad_r = rmx_grad.matmul(wr)
    else:
        mx_grad_r = None

    # compute xq_grad
    xq_grad = apply_rope(xq_grad, freqs_cis, head_dim, rope_head_dim, True)
    if q_norm_w is not None:
        xq_grad = rearrange(xq_grad, 'b l h s -> b l (h s)')
        xq_grad, q_norm_w_grad = rmsnorm_bwd(xq_grad, xq, xq_rstd, q_norm_w, local_heads)
    else:
        q_norm_w_grad = None
    # B*L x H*D/H
    xq_grad = xq_grad.view(bsz * seq_len, local_heads * head_dim)
    mx_grad = xq_grad.matmul(wq) if mx_grad_r is None else torch.addmm(mx_grad_r, xq_grad, wq, out=mx_grad_r)

    # compute xk_grad
    if handle_xk is not None:
        handle_xk.wait()

    xk_grad = apply_rope(xk_grad, freqs_cis, head_dim, rope_head_dim, True)
    if kq_ is not None:
        kq_grad = apply_rope(kq_grad, freqs_cis, head_dim, rope_head_dim, True)
        xk_grad = torch.cat([kq_grad, xk_grad], dim=-1)

    if k_norm_w is not None:
        xk_grad = rearrange(xk_grad, 'b l k s -> b l (k s)')
        xk_grad, k_norm_w_grad = rmsnorm_bwd(xk_grad, xk, xk_rstd, k_norm_w, local_kv_heads * nk)
    else:
        k_norm_w_grad = None
    # B*L x H*D/H
    xk_grad = xk_grad.view(bsz * seq_len, local_kv_heads * head_dim * nk)
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
    attn_out, attn_readers, attn_lse, xq, xk, xv, end_seq, mx, freqs_cis, bos_mask, segments,
    wq, wk, wv, wr, q_norm_w, k_norm_w, head_dim, rope_head_dim, local_heads, local_kv_heads,
    attn_scale, attn_act_func, apply_attn_gate, attn_gate_func, eps,
    attention_dropout, attn_rng_state, causal_attn_backend, stability_control
):
    bsz, seq_len, _ = mx.shape
    nk = {"softmax": 1, "softdelta": 2}[attn_act_func]
    attn_gate_fn = {"silu": F.silu, "softplus": partial(F.softplus, beta=math.log(2))}[attn_gate_func]

    # recompute kv
    if xv is None:
        assert xk is None
        # compute v
        xv = rearrange(F.linear(mx, wv), 'b l (k v) -> b l k v', k=local_kv_heads)
        xv_, handle_xv = all_gather(xv, parallel_region='context', async_op=True)

        # compute k
        xk = F.linear(mx, wk)
    else:
        xv_, handle_xv = all_gather(xv, parallel_region='context', async_op=True)

    xk_, xk_rstd = rmsnorm_fwd(xk, k_norm_w, local_kv_heads * nk, eps) if k_norm_w is not None else (xk, None)
    xk_ = rearrange(xk_, 'b l (k s) -> b l k s', k=local_kv_heads)
    if attn_act_func == 'softmax':
        xk_ = apply_rope(xk_, freqs_cis, head_dim, rope_head_dim, False)
        kq_ = None
    else:
        assert attn_act_func == 'softdelta'
        kq_ = xk_[:, :, :, :head_dim]
        xk_ = xk_[:, :, :, head_dim:]
        kq_, xk_ = apply_ropes(kq_, xk_, freqs_cis, head_dim, rope_head_dim, False)
    xk_, handle_xk = all_gather(xk_, parallel_region='context', async_op=True)

    # compute q
    if xq is None:
        xq = F.linear(mx, wq)
    xq_, xq_rstd = rmsnorm_fwd(xq, q_norm_w, local_heads, eps) if q_norm_w is not None else (xq, None)
    xq_ = rearrange(xq_, 'b l (h s) -> b l h s', h=local_heads)
    xq_ = apply_rope(xq_, freqs_cis, head_dim, rope_head_dim, False)

    if wr is not None:
        rg = rearrange(F.linear(mx, wr, None), 'b l (h s) -> b l h s', h=local_heads)
        if apply_attn_gate:
            # B x L x H x S
            rmx = rg[:, :, :, :head_dim]
            r = rearrange(attn_gate_fn(rmx), 'b l h s -> b l (h s)')
        else:
            r, rmx = None, None
        # B x L x H
        g = rg[:, :, :, -1] if attn_act_func == 'softdelta' else None
    else:
        r = rmx = g = None

    # recompute attention output
    if attn_out is None:
        assert attn_readers is None and attn_lse is None

        if handle_xv is not None:
            handle_xv.wait()
            handle_xv = None
            xv_ = reshape_gathered_tensor_along_specific_dim(xv_, gather_dim=1)[:, :end_seq]
        if handle_xk is not None:
            handle_xk.wait()
            handle_xk = None
            xk_ = reshape_gathered_tensor_along_specific_dim(xk_, gather_dim=1)[:, :end_seq]

        if attn_act_func == 'softmax':
            attn_readers = None
            attn_out, attn_lse, _ = recompute_flash_attention(
                xq_, xk_, xv_, bos_mask, segments, attn_scale, attention_dropout,
                attn_rng_state, stability_control, backend=causal_attn_backend
            )
        elif attn_act_func == 'softdelta':
            attn_out, attn_readers, attn_lse, _ = recompute_softdelta_attn(
                xq_, kq_, xk_, xv_, g, bos_mask, segments, attn_scale, attention_dropout,
                attn_rng_state, stability_control, backend=causal_attn_backend
            )
        else:
            raise NotImplementedError

    return (
        (attn_out, attn_readers, attn_lse), (xq, xq_, xq_rstd), (xk, kq_, xk_, xk_rstd), (xv, xv_), (handle_xk, handle_xv), (rmx, r, g)
    )
