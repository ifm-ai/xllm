"""Fused training block of the dense Transformer layer, and the fused output layer.

`TransformerBlockFunction` runs one layer (norm, attention, residual, norm, FFN, residual) as one autograd Function,
saving the two sublayer inputs and a few attention tensors and recomputing the rest in the backward.

With hyper-connections (`residual_func` ihc / ohc) the layer's state is N = 4 streams X [B, L, N, D] and the two
residual adds are the stream path of xllm/models/fused_blocks/hyper_connection.py (`FusedStreams`), which also
documents how the backward gets the streams back (inverting the writes, or recomputing them from kept inputs every
`residual_ckpt_interval` layers) and the one-backward-per-forward contract of the stream hand-off.
"""

from typing import Optional, Tuple, Any
import math
import torch
from torch.nn import functional as F
from einops import rearrange

from xllm.distributed.utils import reshape_gathered_tensor_along_specific_dim
from xllm.health_metrics import get_active_health_collector
from xllm.modules.fused_ops import (
    memory_efficient_dropout_fwd,
    memory_efficient_dropout_bwd,
)
from xllm.modules.residual.hyper_connection import HYPER_CONNECTIONS, StreamSlot
from xllm.models.fused_blocks.hyper_connection import FusedStreams
from xllm.models.fused_blocks.utils import (
    layer_or_rmsnorm_fwd,
    layer_or_rmsnorm_bwd,
    swiglu_forward,
    swiglu_backward,
)
from xllm.models.fused_blocks.mha import (
    multihead_attention_fwd,
    multihead_attention_bwd,
    multihead_attention_recompute
)
from xllm.models.fused_blocks.cross_entropy import (
    fused_linear_cross_entropy_fwd,
    fused_linear_cross_entropy_bwd
)
from xllm.models.fused_blocks.distributed import (
    reduce_scatter,
    all_gather,
)


class TransformerBlockFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        layer_id: int,
        x: torch.Tensor,  # (bsz, slen, d/MP)
        freqs_cis: Optional[torch.Tensor],  # (slen, d)
        bos_mask: Optional[torch.Tensor],
        segments: Optional[Any],
        attn_norm_w: torch.Tensor,  # (d/MP)
        attn_norm_b: Optional[torch.Tensor],  # (d/MP)
        wq: torch.Tensor,  # (d/MP, d)
        wk: torch.Tensor,  # (d/MP, d)
        wv: torch.Tensor,  # (d/MP, d)
        wr: Optional[torch.Tensor],  # (d/MP, d)
        wg: Optional[torch.Tensor],  # (d/MP, d)
        wo: torch.Tensor,  # (d, v/MP)
        q_norm_w: Optional[torch.Tensor],  # (d/MP)
        local_heads: int,
        local_kv_heads: int,
        head_dim: int,
        rope_head_dim: int,
        attn_gate_func: str,
        causal_attn_backend: str,
        hc1_w: Optional[torch.Tensor],  # hyper-connections: the attention sublayer's controller, (N * d, n_logits)
        hc1_b: Optional[torch.Tensor],  # (n_logits)
        hc1_s: Optional[torch.Tensor],  # (2)
        ffn_norm_w: torch.Tensor,  # (d)
        ffn_norm_b: Optional[torch.Tensor],  # (d)
        norm_local_groups: int,
        fc1_w: torch.Tensor,
        fc2_w: torch.Tensor,
        fc3_w: Optional[torch.Tensor],
        hc2_w: Optional[torch.Tensor],  # hyper-connections: the FFN sublayer's controller
        hc2_b: Optional[torch.Tensor],  # (n_logits)
        hc2_s: Optional[torch.Tensor],  # (2)
        dropout: float,
        attention_dropout: float,
        hidden_dropout: float,
        swiglu: bool,
        layernorm_eps: float,
        rmsnorm_eps: float,
        apply_rmsnorm: bool,
        gather_before_norm: bool,
        residual_func: str,
        residual_heads: Optional[int],
        hc_eps: Optional[float],  # hyper-connections: the controllers' normalization eps
        slot: Optional[StreamSlot],
        last_layer: bool,
        ckpt_interval: int,  # hyper-connections: 0 inverts the writes in the backward, k keeps every k-th layer's input
        attn_stability_control: int,
        deterministic: bool,
        recompute_q: bool,
        recompute_kv: bool,
        recompute_attention: bool,
        recompute_fc1_out: bool,
        recompute_fc3_out: bool,
    ) -> Tuple[torch.Tensor, None, None]:

        hc_cls = HYPER_CONNECTIONS.get(residual_func)  # the HyperConnection class; None: plain residual
        streams = FusedStreams(hc_cls, slot, layer_id, last_layer, ckpt_interval, hc_eps) if hc_cls is not None else None
        hc = streams is not None
        if hc:
            x = streams.read1(x, hc1_w, hc1_b, hc1_s, hc2_w, hc2_b, hc2_s)  # x is the streams; the attention branch reads them
        else:
            residual = x
        bsz, seq_len, _ = x.size()

        x_ = x
        handle_ffn_norm_w = None
        handle_ffn_norm_b = None
        if gather_before_norm:
            x_, _ = all_gather(x, parallel_region='model', async_op=False)
            if attn_norm_w is not None:
                attn_norm_w, _ = all_gather(attn_norm_w, parallel_region='model', async_op=False)
            if attn_norm_b is not None:
                attn_norm_b, _ = all_gather(attn_norm_b, parallel_region='model', async_op=False)
            if ffn_norm_w is not None:
                ffn_norm_w, handle_ffn_norm_w = all_gather(ffn_norm_w, parallel_region='model', async_op=True)
            if ffn_norm_b is not None:
                ffn_norm_b, handle_ffn_norm_b = all_gather(ffn_norm_b, parallel_region='model', async_op=True)

        mx, _, _ = layer_or_rmsnorm_fwd(x_, attn_norm_w, attn_norm_b, norm_local_groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm)
        if not gather_before_norm:
            mx, _ = all_gather(mx, parallel_region='model', async_op=False)

        # MHA forward
        xqkv, attn_out, xh, rng_states, end_seq = multihead_attention_fwd(
            mx, freqs_cis, bos_mask, segments, wq, wk, wv, wr, wo, q_norm_w,
            head_dim, rope_head_dim, local_heads, local_kv_heads, rmsnorm_eps,
            attn_gate_func, dropout, attention_dropout, hidden_dropout,
            causal_attn_backend, attn_stability_control, recompute_attention, layer_id,
        )
        xq, xk, xv, xk_rstd = xqkv
        attn_for_save, attn_lse = attn_out
        flash_rng_state, attn_rng_state, attn_out_rng_state, xh_rng_state = rng_states

        if segments is not None:
            if causal_attn_backend == 'flash':
                seg1, seg2, max_seqlen_q, max_seqlen_k, total_seqlen_k = segments
            elif causal_attn_backend == 'xattn':
                seg1, total_seqlen_k = segments
                seg2, max_seqlen_q, max_seqlen_k = None, None, None
            else:
                raise ValueError(f"Unknown causal attention backend: {causal_attn_backend}")
        else:
            seg1, seg2, max_seqlen_q, max_seqlen_k, total_seqlen_k = None, None, None, None, None

        # residual
        if hc:
            xo = streams.write1(xh)  # the attention output into the streams; the FFN branch reads the result
        else:
            xo = torch.add(xh, residual, out=xh)
            residual = xo

        xo_ = xo
        if gather_before_norm:
            xo_, _ = all_gather(xo, parallel_region='model', async_op=False)
            if handle_ffn_norm_w is not None:
                handle_ffn_norm_w.wait()
                ffn_norm_w = reshape_gathered_tensor_along_specific_dim(ffn_norm_w, gather_dim=0)
            if handle_ffn_norm_b is not None:
                handle_ffn_norm_b.wait()
                ffn_norm_b = reshape_gathered_tensor_along_specific_dim(ffn_norm_b, gather_dim=0)

        xf, _, _ = layer_or_rmsnorm_fwd(xo_, ffn_norm_w, ffn_norm_b, norm_local_groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm)
        if not gather_before_norm:
            xf, _ = all_gather(xf, parallel_region='model', async_op=False)

        # FFN & SwiGLU
        xf = rearrange(xf, 'b l d -> (b l) d')
        h1, h3, out, hidden_rng_state = swiglu_forward(xf, fc1_w, fc2_w, fc3_w, hidden_dropout, swiglu)
        out = rearrange(out, '(b l) d -> b l d', b=bsz)
        out, _ = reduce_scatter(out, parallel_region='model', async_op=False)
        out, out_rng_state = memory_efficient_dropout_fwd(out, dropout, True)
        # residual
        if hc:
            out = streams.write2(out)  # the FFN output into the streams, in place over X1
        else:
            out = torch.add(out, residual)
        rc_saved = streams.saved() if hc else (x, xo)  # the residual's tensors

        ctx.save_for_backward(
            freqs_cis,  # (slen, s)
            bos_mask,  # (bsz, slen)
            seg1,
            seg2,
            attn_norm_w,  # (d/MP)
            attn_norm_b,  # (d/MP)
            wq,  # (d/MP, d)
            wk,  # (d/MP, d)
            wv,  # (d/MP, d)
            wr,  # (d/MP, d)
            wg,  # (h/MP, d)
            wo,  # (d, v/MP)
            q_norm_w,  # (d/MP)
            ffn_norm_w,  # (d)
            ffn_norm_b,  # (d)
            fc1_w,
            fc2_w,
            fc3_w,
            xq if not recompute_q else None,  # (bsz, slen, n_heads, head_dim)
            xk if not recompute_kv else None,  # (bsz, slen, n_kv_heads, head_dim)
            xk_rstd if not recompute_kv else None,  # (bsz, slen, kv_heads)
            xv if not recompute_kv else None,  # (bsz, slen, n_kv_heads, head_dim)
            attn_for_save,  # (bsz, slen, n_heads, head_dim)
            attn_lse,  # (bsz, h/MP, slen, slen)
            flash_rng_state,
            h1 if not recompute_fc1_out else None,  # (bsz*slen, v/MP)
            h3 if not recompute_fc3_out else None,  # (bsz*slen, v/MP)
            *rc_saved,
        )
        ctx.local_heads = local_heads
        ctx.local_kv_heads = local_kv_heads
        ctx.head_dim = head_dim
        ctx.rope_head_dim = rope_head_dim
        ctx.norm_local_groups = norm_local_groups
        ctx.layernorm_eps = layernorm_eps
        ctx.rmsnorm_eps = rmsnorm_eps
        ctx.attn_gate_func = attn_gate_func
        ctx.dropout = dropout
        ctx.attention_dropout = attention_dropout
        ctx.hidden_dropout = hidden_dropout
        ctx.swiglu = swiglu
        ctx.apply_rmsnorm = apply_rmsnorm
        ctx.gather_before_norm = gather_before_norm
        ctx.causal_attn_backend = causal_attn_backend
        ctx.residual_func = residual_func
        ctx.residual_heads = residual_heads
        ctx.streams = streams
        ctx.deterministic = deterministic
        ctx.attn_stability_control = attn_stability_control
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.total_seqlen_k = total_seqlen_k
        ctx.end_seq = end_seq

        ctx.attn_rng_state = attn_rng_state
        ctx.attn_out_rng_state = attn_out_rng_state
        ctx.xh_rng_state = xh_rng_state
        ctx.hidden_rng_state = hidden_rng_state
        ctx.out_rng_state = out_rng_state

        return out, None, None

    @staticmethod
    def backward(ctx, out_grad, aux_loss_grad, cache):
        assert cache is None and aux_loss_grad is None

        (
            freqs_cis,  # (slen, s)
            bos_mask,  # (bsz, slen)
            seg1,
            seg2,
            attn_norm_w,  # (d/MP)
            attn_norm_b,  # (d/MP)
            wq,  # (d/MP, d)
            wk,  # (d/MP, d)
            wv,  # (d/MP, d)
            wr,  # (d/MP, d)
            wg,  # (h/MP, d)
            wo,  # (d, v/MP)
            q_norm_w,  # (d/MP)
            ffn_norm_w,  # (d)
            ffn_norm_b,  # (d)
            fc1_w,
            fc2_w,
            fc3_w,
            xq,  # (bsz, slen, n_heads, head_dim)
            xk,  # (bsz, slen, n_kv_heads, head_dim)
            xk_rstd,  # (bsz, slen, kv_heads)
            xv,  # (bsz, slen, n_kv_heads, head_dim)
            attn_for_save,  # (bsz, slen, n_heads, head_dim)
            attn_lse,  # (bsz, h/MP, slen, slen)
            flash_rng_state,
            h1,  # (bsz*slen, v/MP)
            h3,  # (bsz*slen, v/MP)
            *rc,  # the residual's tensors
        ) = ctx.saved_tensors

        local_heads = ctx.local_heads
        local_kv_heads = ctx.local_kv_heads
        head_dim = ctx.head_dim
        rope_head_dim = ctx.rope_head_dim
        norm_local_groups = ctx.norm_local_groups
        layernorm_eps = ctx.layernorm_eps
        rmsnorm_eps = ctx.rmsnorm_eps
        dropout = ctx.dropout
        attention_dropout = ctx.attention_dropout
        hidden_dropout = ctx.hidden_dropout
        swiglu = ctx.swiglu
        apply_rmsnorm = ctx.apply_rmsnorm
        gather_before_norm = ctx.gather_before_norm
        causal_attn_backend = ctx.causal_attn_backend
        attn_gate_func = ctx.attn_gate_func
        residual_func = ctx.residual_func
        residual_heads = ctx.residual_heads
        attn_stability_control = ctx.attn_stability_control
        deterministic = ctx.deterministic
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        total_seqlen_k = ctx.total_seqlen_k
        end_seq = ctx.end_seq

        attn_rng_state = ctx.attn_rng_state
        attn_out_rng_state = ctx.attn_out_rng_state
        xh_rng_state = ctx.xh_rng_state
        hidden_rng_state = ctx.hidden_rng_state
        out_rng_state = ctx.out_rng_state

        attn_scale = 1.0 / math.sqrt(head_dim)
        streams = ctx.streams
        hc = streams is not None
        if hc:
            # hyper-connections: X1 and xo = read(X1, a2) back, with the second write's backward; dy2 is the FFN output's
            # gradient. x is read from X0 after the FFN backward
            xo, out_grad = streams.restore(rc, out_grad)
            x = None
        else:
            x, xo = rc
            residual_grad = out_grad
        bsz, seq_len, _ = xo.size()

        if gather_before_norm:
            xo_, _ = all_gather(xo, parallel_region='model', async_op=False)
            x_, handle_x = all_gather(x, parallel_region='model', async_op=True) if not hc else (None, None)
            out_grad = memory_efficient_dropout_bwd(out_grad, dropout, out_rng_state)
            out_grad, handle_out = all_gather(out_grad, parallel_region='model', async_op=True)
            # recompute xf
            xf, xo_mean, xo_invvar = layer_or_rmsnorm_fwd(
                xo_, ffn_norm_w, ffn_norm_b, norm_local_groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm
            )
            handle_xf = None
        else:
            xo_ = xo
            x_ = x
            handle_x = None
            # recompute xf
            xf, xo_mean, xo_invvar = layer_or_rmsnorm_fwd(
                xo_, ffn_norm_w, ffn_norm_b, norm_local_groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm
            )
            xf, handle_xf = all_gather(xf, parallel_region='model', async_op=True)

            out_grad = memory_efficient_dropout_bwd(out_grad, dropout, out_rng_state)
            out_grad, handle_out = all_gather(out_grad, parallel_region='model', async_op=True)

        if handle_x is not None:
            handle_x.wait()
            x_ = reshape_gathered_tensor_along_specific_dim(x_, gather_dim=2)

        # recompute mx (hyper-connections: after the FFN backward)
        if not hc:
            mx, x_mean, x_invvar = layer_or_rmsnorm_fwd(
                x_, attn_norm_w, attn_norm_b, norm_local_groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm
            )

        if handle_xf is not None:
            handle_xf.wait()
            xf = reshape_gathered_tensor_along_specific_dim(xf, gather_dim=2)

        xf = rearrange(xf, 'b l d -> (b l) d')
        # recompute h1 & h3 for FFN or SwiGLU
        if h1 is None:
            h1 = F.linear(xf, fc1_w)

        if swiglu and h3 is None:
            h3 = F.linear(xf, fc3_w)

        if handle_out is not None:
            handle_out.wait()
            out_grad = reshape_gathered_tensor_along_specific_dim(out_grad, gather_dim=2)

        if not hc and not gather_before_norm:
            mx, handle_mx = all_gather(mx, parallel_region='model', async_op=True)
        else:
            handle_mx = None

        out_grad = rearrange(out_grad, 'b l d -> (b l) d')
        xo_grad, fc1_w_grad, fc2_w_grad, fc3_w_grad, ffn_norm_w_grad, ffn_norm_b_grad = swiglu_backward(
            out_grad, xf, fc1_w, fc2_w, fc3_w, h1, h3, hidden_dropout, hidden_rng_state,
            xo_, xo_mean, xo_invvar, ffn_norm_w, ffn_norm_b, norm_local_groups, apply_rmsnorm, gather_before_norm
        )

        if hc:
            del xf, h1, h3, xo, xo_, xo_mean, xo_invvar, out_grad
            # the FFN controller's backward and dX1, then X0 and x = read(X0, a1); dy1 is the attention output's gradient
            x, xh_grad = streams.after_ffn(xo_grad)
            del xo_grad
            x_ = x
            mx, x_mean, x_invvar = layer_or_rmsnorm_fwd(
                x_, attn_norm_w, attn_norm_b, norm_local_groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm
            )
        else:
            residual_grad = torch.add(xo_grad, residual_grad, out=xo_grad)
            xh_grad = residual_grad
        xh_grad = memory_efficient_dropout_bwd(xh_grad, dropout, xh_rng_state)
        xh_grad, handle_xh = all_gather(xh_grad, parallel_region='model', async_op=True)

        if bos_mask is not None:
            if causal_attn_backend == 'flash':
                segments = seg1, seg2, max_seqlen_q, max_seqlen_k, total_seqlen_k
            elif causal_attn_backend == 'xattn':
                segments = seg1, total_seqlen_k
            else:
                raise ValueError(f"Unknown causal attention backend: {causal_attn_backend}")
        else:
            segments = None

        if handle_mx is not None:
            handle_mx.wait()
            mx = reshape_gathered_tensor_along_specific_dim(mx, gather_dim=2)

        attn_outs, q_outs, k_outs, v_outs, handle_kv, attn_gate_outs = multihead_attention_recompute(
            attn_for_save, attn_lse, xq, xk, xv, xk_rstd, end_seq,
            mx, freqs_cis, bos_mask, segments, wq, wk, wv, wr, q_norm_w,
            head_dim, rope_head_dim, local_heads, local_kv_heads, rmsnorm_eps,
            attn_scale, attn_gate_func, attention_dropout, attn_rng_state,
            causal_attn_backend, attn_stability_control
        )
        attn_out, attn_for_save, attn_lse = attn_outs
        xq, xq_, xq_rstd = q_outs
        xk, xk_, xk_rstd = k_outs
        xv, xv_ = v_outs
        handle_xk, handle_xv = handle_kv
        rmx, r = attn_gate_outs

        attn = attn_out * r if r is not None else attn_out
        attn, attn_noise = memory_efficient_dropout_fwd(attn, hidden_dropout, True, attn_out_rng_state)

        if handle_xh is not None:
            handle_xh.wait()
            xh_grad = reshape_gathered_tensor_along_specific_dim(xh_grad, gather_dim=2)
        # B x L x D
        attn_grad = xh_grad.matmul(wo)
        attn_grad = memory_efficient_dropout_bwd(attn_grad, hidden_dropout, attn_out_rng_state, attn_noise)
        attn_r_grad = torch.mul(attn_grad, r, out=r) if r is not None else attn_grad
        r_grad = torch.mul(attn_grad, attn_out, out=attn_grad) if r is not None else None

        # grads for wh2
        attn_flat = attn.flatten(end_dim=-2)
        wo_grad = xh_grad.flatten(end_dim=-2).t().matmul(attn_flat)

        if handle_xv is not None:
            assert handle_xk is not None
            handle_xv.wait()
            xv_ = reshape_gathered_tensor_along_specific_dim(xv_, gather_dim=1)[:, :end_seq]
            handle_xk.wait()
            xk_ = reshape_gathered_tensor_along_specific_dim(xk_, gather_dim=1)[:, :end_seq]

        # MHA backward
        # B x L x D -> B x L x H x D/H
        attn_r_grad = attn_r_grad.view(bsz, seq_len, local_heads, head_dim)
        x_grad, wq_grad, wk_grad, wv_grad, wr_grad, q_norm_w_grad, attn_norm_w_grad, attn_norm_b_grad = multihead_attention_bwd(
            attn_r_grad, r_grad, attn_for_save, attn_lse, xq_, xk_, xv_, total_seqlen_k, end_seq,
            freqs_cis, bos_mask, segments, local_kv_heads, rope_head_dim,
            xq, xk, xq_rstd, xk_rstd, rmx, mx, q_norm_w, wq, wk, wv, wr,
            attn_scale, attn_gate_func, attention_dropout, flash_rng_state,
            causal_attn_backend, attn_stability_control, deterministic,
            x_, x_mean, x_invvar, attn_norm_w, attn_norm_b, norm_local_groups,
            apply_rmsnorm, gather_before_norm
        )

        # residual connection
        if hc:
            del mx, x, x_, x_mean, x_invvar
            # the attention controller's backward, then the whole dX0 in one pass
            x_grad, (hc1_w_grad, hc1_b_grad, hc1_s_grad, hc2_w_grad, hc2_b_grad, hc2_s_grad) = streams.after_branch(x_grad)
        else:
            x_grad = torch.add(x_grad, residual_grad, out=x_grad)
            hc1_w_grad = hc1_b_grad = hc1_s_grad = hc2_w_grad = hc2_b_grad = hc2_s_grad = None

        return (
            None,  # layer_id
            x_grad,
            None,  # freqs_cis
            None,  # bos mask
            None,  # segments
            attn_norm_w_grad,
            attn_norm_b_grad,
            wq_grad,
            wk_grad,
            wv_grad,
            wr_grad,
            None,
            wo_grad,
            q_norm_w_grad,
            None,  # local_heads
            None,  # local_kv_heads
            None,  # head_dim
            None,  # rope_head_dim
            None,  # attn_gate_func
            None,  # causal_attn_backend
            hc1_w_grad,
            hc1_b_grad,
            hc1_s_grad,
            ffn_norm_w_grad,
            ffn_norm_b_grad,
            None,  # norm_local_groups
            fc1_w_grad,
            fc2_w_grad,
            fc3_w_grad,
            hc2_w_grad,
            hc2_b_grad,
            hc2_s_grad,
            None,  # dropout
            None,  # attention_dropout
            None,  # hidden_dropout
            None,  # swiglu
            None,  # layernorm_eps
            None,  # rmsnorm_eps
            None,  # apply_rmsnorm
            None,  # gather_before_norm
            None,  # residual_func
            None,  # residual_heads
            None,  # hc_eps
            None,  # slot
            None,  # last_layer
            None,  # ckpt_interval
            None,  # attn_stability_control
            None,  # deterministic
            None,  # recompute_q
            None,  # recompute_kv
            None,  # recompute_attention
            None,  # recompute_fc1_out
            None,  # recompute_fc3_out
        )


class TransformerOutputLayerFunction(torch.autograd.Function):
    """

    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,  # (bsz, slen, d/MP)
        y: torch.Tensor,  # (bsz, slen)
        mask: Optional[torch.Tensor],  # (bsz, slen)
        final_norm_w: torch.Tensor,  # (d/MP)
        final_norm_b: torch.Tensor,  # (d/MP)
        norm_local_groups: int,
        wo: torch.Tensor,  # (voc/MP, d)
        layernorm_eps: float,
        rmsnorm_eps: float,
        apply_rmsnorm: bool,
        gather_before_norm: bool,
        recompute_logits: bool,
    ):
        bsz, seq_len, _ = x.size()

        if gather_before_norm:
            x, _ = all_gather(x, parallel_region='model', async_op=False)
            if final_norm_w is not None:
                final_norm_w, _ = all_gather(final_norm_w, parallel_region='model', async_op=False)
            if final_norm_b is not None:
                final_norm_b, _ = all_gather(final_norm_b, parallel_region='model', async_op=False)

        mx, _, _ = layer_or_rmsnorm_fwd(x, final_norm_w, final_norm_b, norm_local_groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm)
        collector = get_active_health_collector()
        output = fused_linear_cross_entropy_fwd(
            mx, wo, y, mask, compute_grad=(not recompute_logits),
            gather_input=(not gather_before_norm),
            collect_logits_stats=(
                collector is not None and collector.enable_logits_stats
            ),
        )
        if output.logits_stats is not None:
            assert collector is not None
            collector.record_logits_stats(
                output.logits_stats.sum,
                output.logits_stats.count,
                output.logits_stats.minimum,
                output.logits_stats.maximum,
            )
        loss, grad_mx, grad_wo = output.loss, output.grad_input, output.grad_weight

        ctx.save_for_backward(
            x,  # (bsz, slen, d/MP)
            y if recompute_logits else None,  # (bsz, slen)
            mask if recompute_logits else None,  # (bsz, slen)
            final_norm_w,  # (d/MP)
            final_norm_b,  # (d/MP)
            wo,  # (voc/MP, d)
            grad_mx,  # (bsz, slen, d/MP)
            grad_wo,  # (voc/MP, d)
        )

        ctx.norm_local_groups = norm_local_groups
        ctx.layernorm_eps = layernorm_eps
        ctx.rmsnorm_eps = rmsnorm_eps
        ctx.apply_rmsnorm = apply_rmsnorm
        ctx.gather_before_norm = gather_before_norm

        return loss

    def backward(ctx, loss_grad):

        (
            x,  # (bsz, slen, d/MP)
            y,  # (bsz, slen)
            mask,  # (bsz, slen)
            final_norm_w,  # (d/MP)
            final_norm_b,  # (d/MP)
            wo,  # (voc/MP, d)
            grad_mx,  # (bsz, slen, d/MP)
            grad_wo,  # (voc/MP, d)
        ) = ctx.saved_tensors

        norm_local_groups = ctx.norm_local_groups
        layernorm_eps = ctx.layernorm_eps
        rmsnorm_eps = ctx.rmsnorm_eps
        apply_rmsnorm = ctx.apply_rmsnorm
        gather_before_norm = ctx.gather_before_norm

        # recompute mx
        mx, x_mean, x_invvar = layer_or_rmsnorm_fwd(
            x, final_norm_w, final_norm_b, norm_local_groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm
        )

        mx_grad, wo_grad = fused_linear_cross_entropy_bwd(
            loss_grad, grad_mx, grad_wo, mx, wo, y, mask, gather_input=(not gather_before_norm)
        )

        x_grad, final_norm_w_grad, final_norm_b_grad = layer_or_rmsnorm_bwd(
            mx_grad, x, x_mean, x_invvar, final_norm_w, final_norm_b, norm_local_groups, apply_rmsnorm
        )

        if gather_before_norm:
            x_grad, _ = reduce_scatter(x_grad, parallel_region='model', async_op=False)
            if final_norm_w_grad is not None:
                final_norm_w_grad, _ = reduce_scatter(final_norm_w_grad, parallel_region='model', async_op=False)
            if final_norm_b_grad is not None:
                final_norm_b_grad, _ = reduce_scatter(final_norm_b_grad, parallel_region='model', async_op=False)

        return (
            x_grad,
            None,  # y
            None,  # mask
            final_norm_w_grad,
            final_norm_b_grad,
            None,  # norm_local_groups
            wo_grad,
            None,  # layernorm_eps
            None,  # rmsnorm_eps
            None,  # apply rmsnorm
            None,  # gather before norm
            None,  # recompute logits
        )
