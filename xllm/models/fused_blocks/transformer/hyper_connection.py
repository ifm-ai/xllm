"""Fused training block of a dense Transformer layer with hyper-connections (ihc / ohc).

TransformerBlockFunction with the two residual adds replaced by the stream ops of
xllm/modules/fused_ops/hyper_connection.py (xllm/modules/residual/hyper_connection.py is the module path). A separate
Function rather than a residual hook inside TransformerBlockFunction: the streams change the block's input shape, add a
controller before each branch and a cross-layer stream recovery in the backward, none of which the plain block has. The
controller's coefficient kernels come from the sublayer's HyperConnection class (`coef_fwd` / `coef_bwd`), the same ones
the module path runs on CUDA, so the two paths compute the same model.

    X  [B, L, N, D] streams in
    (a1, R1, b1) = controller1(X);   h1 = read(X, a1);   y1 = attention(norm(h1));  X1 = write(X, R1, b1, y1)
    (a2, R2, b2) = controller2(X1);  h2 = read(X1, a2);  y2 = ffn(norm(h2));        X2 = write(X1, R2, b2, y2)

Saved per layer: the branch outputs y1 and y2 (D per token each: the write backward needs them for db and dR,
where the plain residual needs nothing), the attention activations the base block saves, and the controllers'
logits, rstd and coefficients (a few dozen FP32 numbers per token). The streams are NOT saved: the transport is
orthogonal (ohc) or the identity (ihc), so a write is invertible, X = R^T (X' - b y), and the backward recovers
the streams from the layer above:

    X2  from the top layer's output (kept on its context), or from the layer above's backward through a StreamSlot
    X1 = unwrite(X2, R2, b2, y2);  X0 = unwrite(X1, R1, b1, y1);  X0 goes into the slot for the layer below

Streams as hi + lo. A write computes X' in FP32 and stores it as two tensors in the model dtype: hi = round(X'),
the streams every other kernel reads (the controllers, the reads, the write backward: the model sees BF16 streams
as in the module path), and lo = round(X' - hi), so that hi + lo carries X' to about 2^-17. The next write reads
hi + lo, i.e. the streams accumulate across layers at that precision instead of being rounded to BF16 at every
layer (the usual FP32-residual practice), and the inversion in the backward, from hi + lo, reproduces the forward's
streams to the same 2^-17 instead of drifting by the 2^-9 rounding of a singly stored X' at every step. The lo of a
layer's output reaches the next layer's forward through the same StreamSlot; the first layer's input (N copies of
the embedding, `expand_streams`) has no lo and is saved as one copy, D per token, the only streams saved besides
the top layer's (hi, lo).

Recomputed in the backward: the branch inputs h1 = read(X0, a1), h2 = read(X1, a2) (one read pass each) and
their norms; q, k / v and fc1 / fc3 follow the base block's recompute switches.

Only the identity (ihc) and SO(4) (ohc) transports: a Sinkhorn transport (mhc) is not invertible (a doubly
stochastic matrix can be singular), it would need stored streams. Model parallelism: the hyper-connection module
asserts a model-parallel world size of one, so the base block's model-parallel gathers and scatters are left out;
context parallelism stays inside the attention helpers as in the base block.
"""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from einops import rearrange

from xllm.distributed.utils import reshape_gathered_tensor_along_specific_dim
from xllm.modules.fused_ops import (
    memory_efficient_dropout_fwd,
    memory_efficient_dropout_bwd,
)
from xllm.modules.fused_ops import hyper_connection as fused
from xllm.models.fused_blocks.utils import (
    layer_or_rmsnorm_fwd,
    swiglu_forward,
    swiglu_backward,
)
from xllm.models.fused_blocks.mha import (
    multihead_attention_fwd,
    multihead_attention_bwd,
    multihead_attention_recompute
)



class StreamSlot:
    """Hands one layer's stream data to the adjacent layer outside the autograd graph: in the forward the lo part of a
    layer's output streams to the next layer, in the backward the (hi, lo) streams one layer recovered (its input X0)
    to the layer below, whose output they are. The layers run strictly in order both ways, so one slot per model
    suffices; it is empty between passes. The model resets it at the start of every forward, so a pass that was cut
    short (an exception, a backward over part of the graph) cannot leave streams behind for the next step."""

    def __init__(self):
        self.streams = None

    def reset(self):
        self.streams = None

    def put(self, streams):
        assert self.streams is None, "StreamSlot: the adjacent layer did not take the previous streams"
        self.streams = streams

    def take(self):
        streams, self.streams = self.streams, None
        assert streams is not None, "StreamSlot: no streams from the adjacent layer"
        return streams


def controller_fwd(hc_cls, X: Tensor, weight: Tensor, bias: Tensor, scale: Tensor, eps: float):
    """One sublayer's controller on contiguous streams X [T, N, D] -> (logits, rstd, a, R, b); R None for the identity.
    `hc_cls` is the sublayer's HyperConnection class: its `coef_fwd` owns the variant's coefficient kernels."""
    T, N, D = X.shape
    logits, rstd = fused.logits_fwd(X.view(T, N * D), weight.to(X.dtype).contiguous(), eps)
    a, R, b = hc_cls.coef_fwd(logits, scale.float().contiguous(), bias.float().contiguous())
    return logits, rstd, a, R, b


def controller_bwd(hc_cls, X: Tensor, weight: Tensor, bias: Tensor, scale: Tensor, eps: float,
                   logits: Tensor, rstd: Tensor, da: Tensor, dR: Optional[Tensor], db: Tensor):
    """-> dX [T, N, D] (the controller's share), dweight, dbias, dscale in the parameters' dtypes."""
    T, N, D = X.shape
    dlogits, dscale, dbias = hc_cls.coef_bwd(logits, scale.float().contiguous(), bias.float().contiguous(), da, dR, db)
    dX, dW = fused.logits_bwd(X.view(T, N * D), weight.to(X.dtype).contiguous(), rstd, logits, dlogits)
    return dX.view(T, N, D), dW.to(weight.dtype), dbias.to(bias.dtype), dscale.to(scale.dtype)


class HyperConnectionBlockFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        layer_id: int,
        streams: Tensor,
        freqs_cis: Optional[Tensor],
        bos_mask: Optional[Tensor],
        segments,
        attn_norm_w: Optional[Tensor],
        attn_norm_b: Optional[Tensor],
        wq: Tensor,
        wk: Tensor,
        wv: Tensor,
        wr: Optional[Tensor],
        wo: Tensor,
        q_norm_w: Optional[Tensor],
        local_heads: int,
        local_kv_heads: int,
        head_dim: int,
        rope_head_dim: int,
        attn_gate_func: str,
        causal_attn_backend: str,
        ffn_norm_w: Optional[Tensor],
        ffn_norm_b: Optional[Tensor],
        norm_local_groups: int,
        fc1_w: Tensor,
        fc2_w: Tensor,
        fc3_w: Optional[Tensor],
        dropout: float,
        attention_dropout: float,
        hidden_dropout: float,
        swiglu: bool,
        layernorm_eps: float,
        rmsnorm_eps: float,
        apply_rmsnorm: bool,
        gather_before_norm: bool,
        hc_cls: type,
        hc_eps: float,
        hc1_w: Tensor,
        hc1_b: Tensor,
        hc1_s: Tensor,
        hc2_w: Tensor,
        hc2_b: Tensor,
        hc2_s: Tensor,
        attn_stability_control: int,
        deterministic: bool,
        recompute_q: bool,
        recompute_kv: bool,
        recompute_attention: bool,
        recompute_fc1_out: bool,
        recompute_fc3_out: bool,
        store_input: bool,
        store_output: bool,
        slot: Optional[StreamSlot],
    ):
        assert (store_output and layer_id == 0) or slot is not None, "the streams' lo parts pass through the StreamSlot"
        assert not store_input or layer_id == 0, "only the first layer saves its input (the embedding, no lo part)"
        bsz, seq_len, N, D = streams.shape
        T = bsz * seq_len
        X = streams.contiguous().view(T, N, D)
        X_lo = slot.take() if layer_id > 0 else None  # the lo part of the layer below's output streams

        # sublayer 1: controller, read, attention, write
        l1, rstd1, a1, R1, b1 = controller_fwd(hc_cls, X, hc1_w, hc1_b, hc1_s, hc_eps)
        h1 = fused.read_fwd(X, a1).view(bsz, seq_len, D)
        mx, _, _ = layer_or_rmsnorm_fwd(h1, attn_norm_w, attn_norm_b, norm_local_groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm)
        del h1

        xqkv, attn_out, y1, rng_states, end_seq = multihead_attention_fwd(
            mx, freqs_cis, bos_mask, segments, wq, wk, wv, wr, wo, q_norm_w,
            head_dim, rope_head_dim, local_heads, local_kv_heads, rmsnorm_eps,
            attn_gate_func, dropout, attention_dropout, hidden_dropout,
            causal_attn_backend, attn_stability_control, recompute_attention, layer_id,
        )
        del mx
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

        y1 = y1.contiguous().view(T, D)  # the attention branch output after its dropout
        X1, X1_lo = fused.write_fwd_split(X, X_lo, R1, b1, y1)
        del X_lo

        # sublayer 2: controller, read, FFN, write
        l2, rstd2, a2, R2, b2 = controller_fwd(hc_cls, X1, hc2_w, hc2_b, hc2_s, hc_eps)
        h2 = fused.read_fwd(X1, a2).view(bsz, seq_len, D)
        xf, _, _ = layer_or_rmsnorm_fwd(h2, ffn_norm_w, ffn_norm_b, norm_local_groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm)
        del h2
        xf = rearrange(xf, 'b l d -> (b l) d')
        h1_, h3_, out, hidden_rng_state = swiglu_forward(xf, fc1_w, fc2_w, fc3_w, hidden_dropout, swiglu)
        del xf
        y2, out_rng_state = memory_efficient_dropout_fwd(out, dropout, True)
        y2 = y2.contiguous()
        X2, X2_lo = fused.write_fwd_split(X1, X1_lo, R2, b2, y2)
        del X1, X1_lo
        if not store_output:
            slot.put(X2_lo)  # for the next layer's forward

        X_saved = X[:, 0].contiguous() if store_input else None  # the first layer's streams: N copies of the embedding
        ctx.save_for_backward(
            X_saved,  # the first layer only: (T, D)
            freqs_cis,
            bos_mask,
            seg1,
            seg2,
            attn_norm_w,
            attn_norm_b,
            wq,
            wk,
            wv,
            wr,
            wo,
            q_norm_w,
            ffn_norm_w,
            ffn_norm_b,
            fc1_w,
            fc2_w,
            fc3_w,
            hc1_w,
            hc1_b,
            hc1_s,
            hc2_w,
            hc2_b,
            hc2_s,
            l1,  # (T, n_logits) FP32
            rstd1,  # (T,) FP32
            a1,  # (T, N) FP32
            R1,  # (T, N, N) FP32 or None
            b1,  # (T, N) FP32
            l2,
            rstd2,
            a2,
            R2,
            b2,
            y1,  # (T, D)
            y2,  # (T, D)
            xq if not recompute_q else None,  # (bsz, slen, n_heads, head_dim)
            xk if not recompute_kv else None,  # (bsz, slen, n_kv_heads, head_dim)
            xk_rstd if not recompute_kv else None,  # (bsz, slen, kv_heads)
            xv if not recompute_kv else None,  # (bsz, slen, n_kv_heads, head_dim)
            attn_for_save,  # (bsz, slen, n_heads, head_dim)
            attn_lse,
            flash_rng_state,
            h1_ if not recompute_fc1_out else None,  # (T, ffn)
            h3_ if not recompute_fc3_out else None,  # (T, ffn)
        )
        # The top layer's output streams (hi, lo): on the context, not in the saved tensors, so the backward can drop
        # them as soon as it has inverted them (a saved tensor lives until the backward returns; at the top layer,
        # where every layer's saved activations are still alive, this is the step's peak). X2 detached: the output
        # itself would close a cycle (ctx -> X2 -> grad_fn = ctx) that kept the streams alive until a GC run whenever
        # the forward is not followed by a backward.
        ctx.X_out = (X2.detach(), X2_lo) if store_output else None
        del X2_lo
        ctx.shape = (bsz, seq_len, N, D)
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
        ctx.hc_cls = hc_cls
        ctx.hc_eps = hc_eps
        ctx.layer_id = layer_id
        ctx.slot = slot
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

        return X2.view(bsz, seq_len, N, D), None, None

    @staticmethod
    def backward(ctx, out_grad, aux_loss_grad, cache):
        assert cache is None and aux_loss_grad is None

        (
            X_in, freqs_cis, bos_mask, seg1, seg2,
            attn_norm_w, attn_norm_b, wq, wk, wv, wr, wo, q_norm_w,
            ffn_norm_w, ffn_norm_b, fc1_w, fc2_w, fc3_w,
            hc1_w, hc1_b, hc1_s, hc2_w, hc2_b, hc2_s,
            l1, rstd1, a1, R1, b1, l2, rstd2, a2, R2, b2, y1, y2,
            xq, xk, xk_rstd, xv, attn_for_save, attn_lse, flash_rng_state, h1_, h3_,
        ) = ctx.saved_tensors

        bsz, seq_len, N, D = ctx.shape
        T = bsz * seq_len
        if X_in is not None:
            X_in = X_in.unsqueeze(1).expand(T, N, D).contiguous()  # the first layer: re-expand the embedding
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
        hc_cls = ctx.hc_cls
        hc_eps = ctx.hc_eps
        attn_stability_control = ctx.attn_stability_control
        deterministic = ctx.deterministic
        end_seq = ctx.end_seq
        attn_scale = 1.0 / math.sqrt(head_dim)

        G2 = out_grad.contiguous().view(T, N, D)

        # --- recover the mid streams X1 = (hi, lo): from X2 (the layer above's backward, or the top layer's output)
        # by inverting the second write, or forward from the first layer's exact input. X0 follows after the FFN
        # backward, so that the FFN's transients do not overlap the stream chain. ---
        X_out, ctx.X_out = ctx.X_out, None
        at_top = X_out is not None
        if at_top:
            X2, X2_lo = X_out[0].view(T, N, D), X_out[1]  # the model's output: not ours to overwrite
            del X_out
        else:
            X2, X2_lo = ctx.slot.take()  # always take: the slot must be emptied
        if X_in is not None:  # the first layer
            X1, X1_lo = fused.write_fwd_split(X_in, None, R1, b1, y1)
        else:
            X1, X1_lo = fused.unwrite(X2, X2_lo, R2, b2, y2, inplace=not at_top)  # the chain's tensors are reused
        del X2, X2_lo

        # --- sublayer 2 (FFN) ---
        h2 = fused.read_fwd(X1, a2).view(bsz, seq_len, D)
        xf, h2_mean, h2_invvar = layer_or_rmsnorm_fwd(
            h2, ffn_norm_w, ffn_norm_b, norm_local_groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm
        )
        xf = rearrange(xf, 'b l d -> (b l) d')
        if h1_ is None:
            h1_ = F.linear(xf, fc1_w)
        if swiglu and h3_ is None:
            h3_ = F.linear(xf, fc3_w)

        # X2 = write(X1, R2, b2, y2): dX1 (transport share), dR2, db2, dy2
        dX1_w, dR2, db2, dy2 = fused.write_bwd(X1, R2, b2, y2, G2)
        if dX1_w is None:
            dX1_w = G2  # identity transport: dX1 = dX2; G2 must not be modified in place
        del G2
        dy2 = memory_efficient_dropout_bwd(dy2, dropout, ctx.out_rng_state)
        h2_grad, fc1_w_grad, fc2_w_grad, fc3_w_grad, ffn_norm_w_grad, ffn_norm_b_grad = swiglu_backward(
            dy2, xf, fc1_w, fc2_w, fc3_w, h1_, h3_, hidden_dropout, ctx.hidden_rng_state,
            h2, h2_mean, h2_invvar, ffn_norm_w, ffn_norm_b, norm_local_groups, apply_rmsnorm, gather_before_norm
        )
        del dy2, xf, h1_, h3_, h2, h2_mean, h2_invvar
        # h2 = read(X1, a2) and the controller on X1
        dX1, da2 = fused.read_bwd(X1, a2, h2_grad.reshape(T, D))
        del h2_grad
        dX1_c, hc2_w_grad, hc2_b_grad, hc2_s_grad = controller_bwd(
            hc_cls, X1, hc2_w, hc2_b, hc2_s, hc_eps, l2, rstd2, da2, dR2, db2
        )
        del da2, db2, dR2
        dX1 = dX1.add_(dX1_w).add_(dX1_c)  # dX1_w may alias out_grad: accumulate into the fresh read gradient
        del dX1_w, dX1_c

        # --- recover the input streams X0 = (hi, lo) and hand them to the layer below ---
        if X_in is not None:
            X = X_in
        else:
            X, X_lo = fused.unwrite(X1, X1_lo, R1, b1, y1, inplace=True)  # X1's tensors are reused
            ctx.slot.put((X, X_lo))
            del X_lo
        del X_in, X1, X1_lo

        # --- sublayer 1 (attention) ---
        h1 = fused.read_fwd(X, a1).view(bsz, seq_len, D)
        mx, h1_mean, h1_invvar = layer_or_rmsnorm_fwd(
            h1, attn_norm_w, attn_norm_b, norm_local_groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm
        )
        # X1 = write(X, R1, b1, y1)
        dX0_w, dR1, db1, dy1 = fused.write_bwd(X, R1, b1, y1, dX1)
        if dX0_w is None:
            dX0_w = dX1  # identity transport: dX0 = dX1
        del dX1
        xh_grad = memory_efficient_dropout_bwd(dy1.view(bsz, seq_len, D), dropout, ctx.xh_rng_state)
        del dy1

        if bos_mask is not None:
            if causal_attn_backend == 'flash':
                segments = seg1, seg2, ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.total_seqlen_k
            elif causal_attn_backend == 'xattn':
                segments = seg1, ctx.total_seqlen_k
            else:
                raise ValueError(f"Unknown causal attention backend: {causal_attn_backend}")
        else:
            segments = None

        attn_outs, q_outs, k_outs, v_outs, handle_kv, attn_gate_outs = multihead_attention_recompute(
            attn_for_save, attn_lse, xq, xk, xv, xk_rstd, end_seq,
            mx, freqs_cis, bos_mask, segments, wq, wk, wv, wr, q_norm_w,
            head_dim, rope_head_dim, local_heads, local_kv_heads, rmsnorm_eps,
            attn_scale, attn_gate_func, attention_dropout, ctx.attn_rng_state,
            causal_attn_backend, attn_stability_control
        )
        attn_out, attn_for_save, attn_lse = attn_outs
        xq, xq_, xq_rstd = q_outs
        xk, xk_, xk_rstd = k_outs
        xv, xv_ = v_outs
        handle_xk, handle_xv = handle_kv
        rmx, r = attn_gate_outs

        attn = attn_out * r if r is not None else attn_out
        attn, attn_noise = memory_efficient_dropout_fwd(attn, hidden_dropout, True, ctx.attn_out_rng_state)

        # B x L x D
        attn_grad = xh_grad.matmul(wo)
        attn_grad = memory_efficient_dropout_bwd(attn_grad, hidden_dropout, ctx.attn_out_rng_state, attn_noise)
        attn_r_grad = torch.mul(attn_grad, r, out=r) if r is not None else attn_grad
        r_grad = torch.mul(attn_grad, attn_out, out=attn_grad) if r is not None else None

        wo_grad = xh_grad.flatten(end_dim=-2).t().matmul(attn.flatten(end_dim=-2))
        del xh_grad, attn

        if handle_xv is not None:
            assert handle_xk is not None
            handle_xv.wait()
            xv_ = reshape_gathered_tensor_along_specific_dim(xv_, gather_dim=1)[:, :end_seq]
            handle_xk.wait()
            xk_ = reshape_gathered_tensor_along_specific_dim(xk_, gather_dim=1)[:, :end_seq]

        attn_r_grad = attn_r_grad.view(bsz, seq_len, local_heads, head_dim)
        h1_grad, wq_grad, wk_grad, wv_grad, wr_grad, q_norm_w_grad, attn_norm_w_grad, attn_norm_b_grad = multihead_attention_bwd(
            attn_r_grad, r_grad, attn_for_save, attn_lse, xq_, xk_, xv_, ctx.total_seqlen_k, end_seq,
            freqs_cis, bos_mask, segments, local_kv_heads, rope_head_dim,
            xq, xk, xq_rstd, xk_rstd, rmx, mx, q_norm_w, wq, wk, wv, wr,
            attn_scale, attn_gate_func, attention_dropout, flash_rng_state,
            causal_attn_backend, attn_stability_control, deterministic,
            h1, h1_mean, h1_invvar, attn_norm_w, attn_norm_b, norm_local_groups,
            apply_rmsnorm, gather_before_norm
        )
        del mx, h1, h1_mean, h1_invvar

        # h1 = read(X, a1) and the controller on X
        dX0, da1 = fused.read_bwd(X, a1, h1_grad.reshape(T, D))
        del h1_grad
        dX0_c, hc1_w_grad, hc1_b_grad, hc1_s_grad = controller_bwd(
            hc_cls, X, hc1_w, hc1_b, hc1_s, hc_eps, l1, rstd1, da1, dR1, db1
        )
        dX0 = dX0.add_(dX0_w).add_(dX0_c)
        del dX0_w, dX0_c

        return (
            None,  # layer_id
            dX0.view(bsz, seq_len, N, D),
            None,  # freqs_cis
            None,  # bos_mask
            None,  # segments
            attn_norm_w_grad,
            attn_norm_b_grad,
            wq_grad,
            wk_grad,
            wv_grad,
            wr_grad,
            wo_grad,
            q_norm_w_grad,
            None,  # local_heads
            None,  # local_kv_heads
            None,  # head_dim
            None,  # rope_head_dim
            None,  # attn_gate_func
            None,  # causal_attn_backend
            ffn_norm_w_grad,
            ffn_norm_b_grad,
            None,  # norm_local_groups
            fc1_w_grad,
            fc2_w_grad,
            fc3_w_grad,
            None,  # dropout
            None,  # attention_dropout
            None,  # hidden_dropout
            None,  # swiglu
            None,  # layernorm_eps
            None,  # rmsnorm_eps
            None,  # apply_rmsnorm
            None,  # gather_before_norm
            None,  # hc_cls
            None,  # hc_eps
            hc1_w_grad,
            hc1_b_grad,
            hc1_s_grad,
            hc2_w_grad,
            hc2_b_grad,
            hc2_s_grad,
            None,  # attn_stability_control
            None,  # deterministic
            None,  # recompute_q
            None,  # recompute_kv
            None,  # recompute_attention
            None,  # recompute_fc1_out
            None,  # recompute_fc3_out
            None,  # store_input
            None,  # store_output
            None,  # slot
        )
