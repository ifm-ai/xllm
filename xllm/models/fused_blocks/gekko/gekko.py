from typing import Optional, Tuple, Any
import math
from functools import partial
import torch
from torch.nn import functional as F
from einops import rearrange

from xllm.distributed.utils import reshape_gathered_tensor_along_specific_dim
from xllm.modules.context_parallel import (
    should_send_to_next,
    should_recv_from_prev,
)
from xllm.modules.fused_ops import (
    memory_efficient_dropout_fwd,
    memory_efficient_dropout_bwd,
    adaptive_working_memory_fwd,
    adaptive_working_memory_accum_fwd,
    adaptive_working_memory_bwd,
)
from xllm.models.fused_blocks.utils import (
    layer_or_rmsnorm_fwd,
    rmsnorm_fwd,
    rmsnorm_bwd,
    mem_effn_rmsnorm_bwd,
    timenorm_fwd,
    timenorm_bwd,
    causal_conv1d_fwd,
    causal_conv1d_bwd,
    sliding_chunk_attention_fwd,
    sliding_chunk_attention_bwd,
    recompute_sliding_chunk_attention,
    swiglu_forward,
    swiglu_backward,
    apply_rope,
    apply_ropes
)
from xllm.models.fused_blocks.cross_entropy import (
    fused_linear_cross_entropy_fwd,
    fused_linear_cross_entropy_bwd,
)
from xllm.models.fused_blocks.distributed import (
    reduce_scatter,
    all_gather,
    recv_prev_count,
    recv_prev_mean_var,
    recv_prev_conv_state,
    recv_prev_key_or_value,
    send_count_to_next,
    send_prev_conv_state_to_next,
    send_prev_key_or_value_to_next,
    send_mean_var_to_next,
    recv_mean_var_grad_from_next,
    recv_prev_conv_state_grad_from_next,
    recv_prev_key_or_value_grad_from_next,
    send_mean_var_grad_to_prev,
    send_prev_conv_state_grad_to_prev,
    send_prev_key_or_value_grad_to_prev,
    recv_prev_memory,
    recv_prev_log_norm_term,
    recv_memory_grad_from_next,
    recv_log_norm_term_grad_from_next,
    send_memory_to_next,
    send_log_norm_term_to_next,
    send_memory_grad_to_prev,
    send_log_norm_term_grad_to_prev,
)


class GekkoBlockFunction(torch.autograd.Function):
    """

    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,  # (bsz, slen, d/MP)
        freqs_cis: Optional[torch.Tensor],  # (slen, d)
        bos_mask: Optional[torch.Tensor],  # (bsz, slen)
        segment_idx: Optional[torch.Tensor],  # (bsz, slen)
        prev_segment_count: Optional[torch.Tensor],  # (bsz),
        timenorm_w: torch.Tensor,  # (d/MP)
        timenorm_b: torch.Tensor,  # (d/MP)
        timenorm_prior_count: torch.Tensor,
        timenorm_prior_mean: torch.Tensor,
        timenorm_prior_logv_or_var: torch.Tensor,
        timenorm_local_groups: int,
        timenorm_beta1: Optional[float],
        timenorm_beta2: Optional[float],
        timenorm_backend: str,
        wq: torch.Tensor,  # (s/MP, d)
        wk: torch.Tensor,  # (s/MP)
        wv: torch.Tensor,  # (v/MP, d)
        bv: Optional[torch.Tensor],  # (v/MP)
        wr: torch.Tensor,  # (v/MP, d)
        br: Optional[torch.Tensor],  # (v/MP)
        wg: Optional[torch.Tensor],  # (d/MP, d)
        wo: torch.Tensor,  # (d, v/MP)
        q_conv_w: torch.Tensor,  # (w, d/MP)
        k_conv_w: torch.Tensor,  # (w, d/MP)
        v_conv_w: torch.Tensor,  # (w, v/MP)
        causal_conv_backend: str,
        causal_conv_weight_normalization: bool,
        qnorm_w: torch.Tensor,  # (s/MP)
        local_heads: int,
        local_kv_heads: int,
        attention_chunk_size: int,
        head_dim: int,
        v_head_dim: int,
        rope_head_dim: int,
        attn_gate_func: str,
        sca_backend: str,
        awm_orthogonal_update: bool,
        attn_res_w: Optional[torch.Tensor],  # (d/MP, c)
        ffn_norm_w: torch.Tensor,  # (d)
        ffn_norm_b: torch.Tensor,  # (d)
        ffn_norm_local_groups: int,
        fc1_w: torch.Tensor,
        fc2_w: torch.Tensor,
        fc3_w: Optional[torch.Tensor],
        ffn_res_w: Optional[torch.Tensor],  # (d/MP, c)
        dropout: float,
        attention_dropout: float,
        hidden_dropout: float,
        swiglu: bool,
        timenorm_eps: float,
        layernorm_eps: float,
        rmsnorm_eps: float,
        apply_rmsnorm: bool,
        residual_func: str,
        residual_heads: Optional[int],
        fp32_attn_output: bool,
        deterministic: bool,
        recompute_q: bool,
        recompute_kv: bool,
        recompute_sca: bool,
        recompute_awk: bool,
        recompute_fc1_out: bool,
        recompute_fc3_out: bool,
    ) -> Tuple[torch.Tensor, None, None]:

        assert timenorm_local_groups is not None
        attn_gate_fn = {"silu": F.silu, "softplus": partial(F.softplus, beta=math.log(2))}[attn_gate_func]
        assert attn_gate_fn is not None

        bsz, seq_len, _ = x.size()
        width = q_conv_w.shape[1]
        residual = x
        recv_from_prev = should_recv_from_prev()
        send_to_next = should_send_to_next()
        sca_scale = 1.0 / math.sqrt(head_dim)

        # recv prev mean & var & count for tsn
        if recv_from_prev:
            prev_mean_var, handle_tsn = recv_prev_mean_var(x, timenorm_local_groups)
            prev_count, handle_count = recv_prev_count(x, bos_mask)
            bos_mask_curr = bos_mask[:, attention_chunk_size:] if bos_mask is not None else None
            handle_tsn.wait()
            prev_mean, prev_var = torch.unbind(prev_mean_var, dim=2)
            if handle_count is not None:
                handle_count.wait()
        else:
            bos_mask_curr = bos_mask
            prev_count = timenorm_prior_count.expand(bsz).contiguous()
            prev_mean = timenorm_prior_mean.type_as(x).expand(bsz, -1).contiguous()
            prev_var = timenorm_prior_logv_or_var.type_as(x).expand(bsz, -1).contiguous()

        # Timestep Normalization
        # start to recv tensors for conv_kv & sca
        if recv_from_prev:
            conv_state_v, handle_conv_v = recv_prev_conv_state(x, width, local_kv_heads * v_head_dim)
            conv_state_k, handle_conv_k = recv_prev_conv_state(x, width, local_kv_heads * head_dim)
            prev_v, handle_prev_v = recv_prev_key_or_value(x, chunk_size=attention_chunk_size, n_heads=local_kv_heads, head_dim=v_head_dim)
            prev_k, handle_prev_k = recv_prev_key_or_value(x, chunk_size=attention_chunk_size, n_heads=local_kv_heads, head_dim=head_dim * 3)
        else:
            prev_k, prev_v = None, None
            handle_prev_v = None
            handle_prev_k = None
            conv_state_k, conv_state_v = None, None
            handle_conv_k, handle_conv_v = None, None

        # B x L x D
        mx, tsn_count, tsn_mean, tsn_var, _, _ = timenorm_fwd(
            x, bos_mask_curr, prev_count, prev_mean, prev_var, timenorm_w, timenorm_b,
            timenorm_local_groups, timenorm_beta1, timenorm_beta2, timenorm_eps, timenorm_backend
        )

        if send_to_next:
            _, handle1 = send_mean_var_to_next(tsn_mean, tsn_var)
            _, handle2 = send_count_to_next(tsn_count, bos_mask)
        else:
            handle1 = None
            handle2 = None

        mx, _ = all_gather(mx, parallel_region='model', async_op=False)

        # start to recv prev tensors for conv_q & awk
        if recv_from_prev:
            conv_state_q, handle_conv_q = recv_prev_conv_state(x, width, local_heads * head_dim)
            memory, handle_memory = recv_prev_memory(x, n_heads=local_kv_heads, qk_head_dim=head_dim, v_head_dim=v_head_dim)
            log_norm_term, handle_lnt = recv_prev_log_norm_term(x, n_heads=local_kv_heads, qk_head_dim=head_dim)
        else:
            conv_state_q = None
            memory = None
            log_norm_term = None
            handle_conv_q = None
            handle_memory = None
            handle_lnt = None

        # B x L x E
        xv = F.linear(mx, wv, bv)
        if handle_conv_v is not None:
            handle_conv_v.wait()
        xv_, final_conv_state_v, _ = causal_conv1d_fwd(
            xv, v_conv_w, None, conv_state_v, bos_mask_curr, send_to_next, 'silu',
            causal_conv_weight_normalization, causal_conv_backend, deterministic
        )
        # B x L x K x V
        xv_ = rearrange(xv_, 'b l (k v) -> b l k v', k=local_kv_heads)

        if send_to_next:
            _, handle3 = send_prev_conv_state_to_next(final_conv_state_v.contiguous())
        else:
            handle3 = None

        # B x L x S
        xk = F.linear(mx, wk, None)
        if handle_conv_k is not None:
            handle_conv_k.wait()
        xk_, final_conv_state_k, _ = causal_conv1d_fwd(
            xk, k_conv_w, None, conv_state_k, bos_mask_curr, send_to_next, None,
            causal_conv_weight_normalization, causal_conv_backend, deterministic
        )

        if send_to_next:
            _, handle4 = send_prev_conv_state_to_next(final_conv_state_k.contiguous())
            _, handle5 = send_prev_key_or_value_to_next(xv_[:, (seq_len - attention_chunk_size):])
        else:
            handle4 = None
            handle5 = None

        sk, _ = rmsnorm_fwd(xk_, None, local_kv_heads, rmsnorm_eps)
        sk = rearrange(sk, 'b l (k s) -> b l k s', k=local_kv_heads)
        # apply rotary embeddings
        sk = apply_rope(sk, freqs_cis, head_dim, rope_head_dim, False)

        # B x L x K x S
        akk = rearrange(xk_, 'b l (k s) -> b l k s', k=local_kv_heads)
        # apply softmax to aqk
        aqk = F.softmax(akk, dim=-1, dtype=torch.float32).to(akk)

        if send_to_next:
            psk = sk[:, (seq_len - attention_chunk_size):]
            paqk = aqk[:, (seq_len - attention_chunk_size):]
            pakk = akk[:, (seq_len - attention_chunk_size):]
            _, handle6 = send_prev_key_or_value_to_next(torch.cat([psk, paqk, pakk], dim=3))
        else:
            handle6 = None

        # B x L x S
        xq = F.linear(mx, wq, None)
        if handle_conv_q is not None:
            handle_conv_q.wait()
        xq_, final_conv_state_q, _ = causal_conv1d_fwd(
            xq, q_conv_w, None, conv_state_q, bos_mask_curr, send_to_next, None,
            causal_conv_weight_normalization, causal_conv_backend, deterministic
        )

        if send_to_next:
            _, handle7 = send_prev_conv_state_to_next(final_conv_state_q.contiguous())
        else:
            handle7 = None

        sq, _ = rmsnorm_fwd(xq_, qnorm_w, local_heads, rmsnorm_eps)
        sq = rearrange(sq, 'b l (h s) -> b l h s', h=local_heads)
        # apply rotary embeddings
        sq = apply_rope(sq, freqs_cis, head_dim, rope_head_dim, False)

        # B x L x H x S
        aq = rearrange(xq_, 'b l (h s) -> b l h s', h=local_heads)
        # apply softmax to aq
        aq = F.softmax(aq, dim=-1, dtype=torch.float32).to(aq)

        if handle_prev_v is not None:
            handle_prev_v.wait()
            handle_prev_k.wait()
            prev_sk, prev_aqk, prev_akk = torch.split(prev_k, [head_dim, head_dim, head_dim], dim=-1)
            prev_sv = prev_v
            # B x C x H x S -> B x H x C x S
            prev_aqk = prev_aqk.transpose(1, 2)
            prev_akk = prev_akk.transpose(1, 2)
            prev_av = prev_v.transpose(1, 2)
        else:
            prev_sk, prev_aqk, prev_akk = None, None, None
            prev_sv, prev_av = None, None

        if send_to_next:
            if handle_memory is not None:
                handle_memory.wait()
                handle_lnt.wait()

            # B x L x H x S -> B x H x L x S
            aq = aq.transpose(1, 2)
            aqk = aqk.transpose(1, 2)
            akk = akk.transpose(1, 2)
            av = xv_.transpose(1, 2)

            awk_out, awk_mask, new_memory, new_lnt = adaptive_working_memory_fwd(
                aq, aqk, akk, av, attention_chunk_size, memory, log_norm_term,
                prev_aqk, prev_akk, prev_av, segment_idx, prev_segment_count,
                awm_orthogonal_update, rmsnorm_eps
            )

            _, handle8 = send_memory_to_next(new_memory)
            _, handle9 = send_log_norm_term_to_next(new_lnt)

            sca_out, sca_for_save, sca_aux, attn_w_rng_state = sliding_chunk_attention_fwd(
                sq, sk, xv_, attention_chunk_size, sca_scale, prev_sk, prev_sv,
                bos_mask, segment_idx, attention_dropout, fp32_attn_output, sca_backend, not recompute_sca
            )
            # B x L x E
            r = attn_gate_fn(F.linear(mx, wr, br))
        else:
            sca_out, sca_for_save, sca_aux, attn_w_rng_state = sliding_chunk_attention_fwd(
                sq, sk, xv_, attention_chunk_size, sca_scale, prev_sk, prev_sv,
                bos_mask, segment_idx, attention_dropout, fp32_attn_output, sca_backend, not recompute_sca
            )
            # B x L x E
            r = attn_gate_fn(F.linear(mx, wr, br))

            if handle_memory is not None:
                handle_memory.wait()
                handle_lnt.wait()

            # B x L x H x S -> B x H x L x S
            aq = aq.transpose(1, 2)
            aqk = aqk.transpose(1, 2)
            akk = akk.transpose(1, 2)
            av = xv_.transpose(1, 2)

            awk_out, awk_mask, new_memory, new_lnt = adaptive_working_memory_fwd(
                aq, aqk, akk, av, attention_chunk_size, memory, log_norm_term,
                prev_aqk, prev_akk, prev_av, segment_idx, prev_segment_count,
                awm_orthogonal_update, rmsnorm_eps
            )

            handle8 = None
            handle9 = None

        if not recompute_sca and sca_for_save is None:
            assert sca_backend == 'swift'
            sca_for_save = sca_out
        # B x L x E
        sca_out = rearrange(sca_out, 'b l h v -> b l (h v)')
        awk_out = rearrange(awk_out, 'b h l v -> b l (h v)')
        attn, _ = rmsnorm_fwd(sca_out + awk_out, None, local_heads, rmsnorm_eps)
        attn = torch.mul(attn, r, out=attn)
        attn, attn_out_rng_state = memory_efficient_dropout_fwd(attn, hidden_dropout, True)

        # B x L x E -> B x L x D
        xh, _ = reduce_scatter(F.linear(attn, wo, None), parallel_region='model', async_op=False)
        xh, xh_rng_state = memory_efficient_dropout_fwd(xh, dropout, True)

        # residual
        xo = torch.add(xh, residual, out=xh)
        residual = xo

        if send_to_next:
            handle1.wait()
            if handle2 is not None:
                handle2.wait()
            handle3.wait()
            handle4.wait()
            handle5.wait()
            handle6.wait()
            handle7.wait()
            handle8.wait()
            handle9.wait()

        xf, _, _ = layer_or_rmsnorm_fwd(xo, ffn_norm_w, ffn_norm_b, ffn_norm_local_groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm)

        # FFN & SwiGLU
        xf, _ = all_gather(xf, parallel_region='model', async_op=False)

        xf = rearrange(xf, 'b l d -> (b l) d')
        h1, h3, out, hidden_rng_state = swiglu_forward(xf, fc1_w, fc2_w, fc3_w, hidden_dropout, swiglu)
        out = rearrange(out, '(b l) d -> b l d', b=bsz)
        out, _ = reduce_scatter(out, parallel_region='model', async_op=False)
        out, out_rng_state = memory_efficient_dropout_fwd(out, dropout, True)
        # residual
        out = torch.add(out, residual)

        ctx.save_for_backward(
            x,  # (bsz, slen, d/MP)
            freqs_cis,  # (slen, s)
            bos_mask,  # (bsz, slen)
            segment_idx,  # (bsz, slen)
            prev_segment_count,  # (bsz,)
            timenorm_w,  # (d/MP)
            timenorm_b,  # (d/MP)
            wq,  # (z/MP, d)
            wk,  # (z/MP)
            wv,  # (v/MP, d)
            bv,  # (v/MP)
            wr,  # (v/MP, d)
            br,  # (v/MP)
            wg,  # (h/MP, d)
            wo,  # (v/MP, d)
            q_conv_w,  # (w, d/MP)
            k_conv_w,  # (w, d/MP)
            v_conv_w,  # (w, v/MP)
            qnorm_w,  # (s/MP)
            attn_res_w,  # (d/MP, c)
            ffn_norm_w,  # (d)
            ffn_norm_b,  # (d)
            fc1_w,
            fc2_w,
            fc3_w,
            ffn_res_w,  # (d/MP, c)
            prev_count,  # (bsz)
            prev_mean,  # (bsz, n_groups/MP)
            prev_var,  # (bsz, n_groups/MP)
            xq if not recompute_q else None,  # (bsz, slen, s/MP)
            xk if not recompute_kv else None,  # (bsz, slen, s/MP)
            xv if not recompute_kv else None,  # (bsz, slen, v/MP)
            conv_state_q,  # (bsz, width-1, s/MP)
            conv_state_k,  # (bsz, width-1, s/MP)
            conv_state_v,  # (bsz, width-1, v/MP)
            prev_k,  # (bsz, chunk, s/MP * 2)
            prev_v,  # (bsz, chunk, v/MP)
            sca_for_save,  # (bsz, slen, h/MP, v/MP)
            sca_aux,  # (bsz, h/MP, slen, chunksize)
            awk_out if not recompute_awk else None,  # (bsz, slen, v/MP)
            memory,  # (bsz, h/MP, s/(h*MP), v/(h*MP))
            log_norm_term,  # (bsz, h/MP, s/MP)
            xo,  # (bsz, slen, d/MP)
            h1 if not recompute_fc1_out else None,  # (bsz*slen, v/MP)
            h3 if not recompute_fc3_out else None,  # (bsz*slen, v/MP)
        )
        ctx.local_heads = local_heads
        ctx.local_kv_heads = local_kv_heads
        ctx.attention_chunk_size = attention_chunk_size
        ctx.head_dim = head_dim
        ctx.v_head_dim = v_head_dim
        ctx.rope_head_dim = rope_head_dim
        ctx.attn_gate_func = attn_gate_func
        ctx.awm_orthogonal_update = awm_orthogonal_update
        ctx.causal_conv_backend = causal_conv_backend
        ctx.causal_conv_weight_normalization = causal_conv_weight_normalization
        ctx.dropout = dropout
        ctx.attention_dropout = attention_dropout
        ctx.hidden_dropout = hidden_dropout
        ctx.swiglu = swiglu
        ctx.timenorm_eps = timenorm_eps
        ctx.layernorm_eps = layernorm_eps
        ctx.rmsnorm_eps = rmsnorm_eps
        ctx.apply_rmsnorm = apply_rmsnorm
        ctx.residual_func = residual_func
        ctx.residual_heads = residual_heads
        ctx.sca_backend = sca_backend
        ctx.deterministic = deterministic
        ctx.fp32_attn_output = fp32_attn_output
        ctx.recv_from_prev = recv_from_prev
        ctx.send_to_next = send_to_next
        ctx.timenorm_local_groups = timenorm_local_groups
        ctx.timenorm_beta1 = timenorm_beta1
        ctx.timenorm_beta2 = timenorm_beta2
        ctx.timenorm_backend = timenorm_backend
        ctx.ffn_norm_local_groups = ffn_norm_local_groups

        ctx.attn_out_rng_state = attn_out_rng_state
        ctx.attn_w_rng_state = attn_w_rng_state
        ctx.xh_rng_state = xh_rng_state
        ctx.hidden_rng_state = hidden_rng_state
        ctx.out_rng_state = out_rng_state

        return out, None, None

    @staticmethod
    def backward(ctx, out_grad, aux_loss_grad, cache):
        assert cache is None and aux_loss_grad is None

        (
            x,  # (bsz, slen, d/MP)
            freqs_cis,  # (slen, s)
            bos_mask,  # (bsz, slen)
            segment_idx,  # (bsz, slen)
            prev_segment_count,  # (bsz, )
            timenorm_w,  # (d/MP)
            timenorm_b,  # (d/MP)
            wq,  # (z/MP, d)
            wk,  # (z/MP)
            wv,  # (v/MP, d)
            bv,  # (v/MP)
            wr,  # (v/MP, d)
            br,  # (v/MP)
            wg,  # (h/MP, d)
            wo,  # (v/MP, d)
            q_conv_w,  # (w, d/MP)
            k_conv_w,  # (w, d/MP)
            v_conv_w,  # (w, v/MP)
            qnorm_w,  # (s/MP)
            attn_res_w,  # (d/MP, c)
            ffn_norm_w,  # (d)
            ffn_norm_b,  # (d)
            fc1_w,
            fc2_w,
            fc3_w,
            ffn_res_w,  # (d/MP, c)
            prev_count,  # (bsz)
            prev_mean,  # (bsz, n_groups/MP)
            prev_var,  # (bsz, n_groups/MP)
            xq,  # (bsz, slen, s/MP)
            xk,  # (bsz, slen, s/MP)
            xv,  # (bsz, slen, v/MP)
            conv_state_q,  # (bsz, width-1, s/MP)
            conv_state_k,  # (bsz, width-1, s/MP)
            conv_state_v,  # (bsz, width-1, v/MP)
            prev_k,  # (bsz, chunk, s/MP)
            prev_v,  # (bsz, chunk, v/MP)
            sca_for_save,  # (bsz, slen, h/MP, v/MP)
            sca_aux,  # (bsz, h/MP, slen, chunksize)
            awk_out,  # (bsz, slen, v/MP)
            memory,  # (bsz, h/MP, s/(h*MP), v/(h*MP))
            log_norm_term,  # (bsz, h/MP, s/MP)
            xo,  # (bsz, slen, d/MP)
            h1,  # (bsz*slen, v/MP)
            h3,  # (bsz*slen, v/MP)
        ) = ctx.saved_tensors

        bsz, seq_len, _ = x.size()
        width = q_conv_w.shape[1]
        residual_grad = out_grad

        recv_from_prev = ctx.recv_from_prev
        send_to_next = ctx.send_to_next
        timenorm_local_groups = ctx.timenorm_local_groups
        timenorm_beta1 = ctx.timenorm_beta1
        timenorm_beta2 = ctx.timenorm_beta2
        timenorm_backend = ctx.timenorm_backend
        ffn_norm_local_groups = ctx.ffn_norm_local_groups

        local_heads = ctx.local_heads
        local_kv_heads = ctx.local_kv_heads
        attention_chunk_size = ctx.attention_chunk_size
        head_dim = ctx.head_dim
        v_head_dim = ctx.v_head_dim
        rope_head_dim = ctx.rope_head_dim
        attn_gate_func = ctx.attn_gate_func
        attn_gate_fn = {"silu": F.silu, "softplus": partial(F.softplus, beta=math.log(2))}[attn_gate_func]
        attn_gate_fn_bwd = {
            "silu": torch.ops.aten.silu_backward,
            "softplus": partial(torch.ops.aten.softplus_backward, beta=math.log(2), threshold=20)
        }[attn_gate_func]
        awm_orthogonal_update = ctx.awm_orthogonal_update
        causal_conv_backend = ctx.causal_conv_backend
        causal_conv_weight_normalization = ctx.causal_conv_weight_normalization

        dropout = ctx.dropout
        attention_dropout = ctx.attention_dropout
        hidden_dropout = ctx.hidden_dropout
        swiglu = ctx.swiglu
        timenorm_eps = ctx.timenorm_eps
        layernorm_eps = ctx.layernorm_eps
        rmsnorm_eps = ctx.rmsnorm_eps
        apply_rmsnorm = ctx.apply_rmsnorm
        residual_func = ctx.residual_func
        residual_heads = ctx.residual_heads
        fp32_attn_output = ctx.fp32_attn_output
        deterministic = ctx.deterministic
        sca_backend = ctx.sca_backend

        sca_scale = 1.0 / math.sqrt(head_dim)

        attn_out_rng_state = ctx.attn_out_rng_state
        attn_w_rng_state = ctx.attn_w_rng_state
        xh_rng_state = ctx.xh_rng_state
        hidden_rng_state = ctx.hidden_rng_state
        out_rng_state = ctx.out_rng_state

        if recv_from_prev and bos_mask is not None:
            bos_mask_curr = bos_mask[:, attention_chunk_size:]
        else:
            bos_mask_curr = bos_mask

        # recompute xf
        xf, xo_mean, xo_invvar = layer_or_rmsnorm_fwd(
            xo, ffn_norm_w, ffn_norm_b, ffn_norm_local_groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm
        )
        xf, handle_xf = all_gather(xf, parallel_region='model', async_op=True)

        out_grad = memory_efficient_dropout_bwd(out_grad, dropout, out_rng_state)
        out_grad, handle_out = all_gather(out_grad, parallel_region='model', async_op=True)

        # recompute mx
        # B x L x D
        mx, tsn_count, tsn_mean, tsn_var, cummean, cumrstd = timenorm_fwd(
            x, bos_mask_curr, prev_count, prev_mean, prev_var, timenorm_w, timenorm_b,
            timenorm_local_groups, timenorm_beta1, timenorm_beta2, timenorm_eps, timenorm_backend
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

        # gather mx
        mx, handle_mx = all_gather(mx, parallel_region='model', async_op=True)

        out_grad = rearrange(out_grad, 'b l d -> (b l) d')
        xo_grad, fc1_w_grad, fc2_w_grad, fc3_w_grad, ffn_norm_w_grad, ffn_norm_b_grad = swiglu_backward(
            out_grad, xf, fc1_w, fc2_w, fc3_w, h1, h3, hidden_dropout, hidden_rng_state,
            xo, xo_mean, xo_invvar, ffn_norm_w, ffn_norm_b, ffn_norm_local_groups, apply_rmsnorm, False,
        )

        residual_grad = torch.add(xo_grad, residual_grad, out=xo_grad)
        xh_grad = residual_grad
        xh_grad = memory_efficient_dropout_bwd(xh_grad, dropout, xh_rng_state)
        xh_grad, handle_xh = all_gather(xh_grad, parallel_region='model', async_op=True)

        if prev_v is not None:
            prev_sk, prev_aqk, prev_akk = torch.split(prev_k, [head_dim, head_dim, head_dim], dim=-1)
            prev_sv = prev_v
            # B x C x H x S/H -> B x H x C x S/H
            prev_aqk = prev_aqk.transpose(1, 2)
            prev_akk = prev_akk.transpose(1, 2)
            prev_av = prev_v.transpose(1, 2)
        else:
            prev_sk, prev_aqk, prev_akk = None, None, None
            prev_sv, prev_av = None, None

        # B x L x D
        if handle_mx is not None:
            handle_mx.wait()
            mx = reshape_gathered_tensor_along_specific_dim(mx, gather_dim=2)

        # recompute kv
        if xv is None:
            assert xk is None
            xv = F.linear(mx, wv, bv)
            xk = F.linear(mx, wk, None)
        xv_, _, _ = causal_conv1d_fwd(
            xv, v_conv_w, None, conv_state_v, bos_mask_curr, False, 'silu',
            causal_conv_weight_normalization, causal_conv_backend, deterministic
        )
        xk_, _, _ = causal_conv1d_fwd(
            xk, k_conv_w, None, conv_state_k, bos_mask_curr, False, None,
            causal_conv_weight_normalization, causal_conv_backend, deterministic
        )
        # B x L x K x V
        xv_ = rearrange(xv_, 'b l (k v) -> b l k v', k=local_kv_heads)

        # recompute q
        if xq is None:
            xq = F.linear(mx, wq, None)
        xq_, _, _ = causal_conv1d_fwd(
            xq, q_conv_w, None, conv_state_q, bos_mask_curr, False, None,
            causal_conv_weight_normalization, causal_conv_backend, deterministic
        )

        sk, sk_rstd = rmsnorm_fwd(xk_, None, local_kv_heads, rmsnorm_eps)
        sq, sq_rstd = rmsnorm_fwd(xq_, qnorm_w, local_heads, rmsnorm_eps)
        # B x L x H x S
        sk = rearrange(sk, 'b l (k s) -> b l k s', k=local_kv_heads)
        sq = rearrange(sq, 'b l (h s) -> b l h s', h=local_heads)
        # apply rotary embeddings
        sq, sk = apply_ropes(sq, sk, freqs_cis, head_dim, rope_head_dim, False)

        # B x L x K|H x S
        akk = rearrange(xk_, 'b l (k s) -> b l k s', k=local_kv_heads)
        aq = rearrange(xq_, 'b l (h s) -> b l h s', h=local_heads)
        # softmax for aq and aqk
        aqk_fp32 = F.softmax(akk, dim=-1, dtype=torch.float32)
        aqk = aqk_fp32.to(akk)
        aq_fp32 = F.softmax(aq, dim=-1, dtype=torch.float32)
        aq = aq_fp32.to(aq)

        if send_to_next:
            memory_grad, handle_memory = recv_memory_grad_from_next(x, local_kv_heads, head_dim, v_head_dim)
            log_norm_term_grad, handle_lnt = recv_log_norm_term_grad_from_next(x, local_kv_heads, head_dim)
            k_grad_from_next, handle_prev_k = recv_prev_key_or_value_grad_from_next(x, attention_chunk_size, local_kv_heads, head_dim * 3)
            v_grad_from_next, handle_prev_v = recv_prev_key_or_value_grad_from_next(x, attention_chunk_size, local_kv_heads, v_head_dim)
        else:
            memory_grad = None
            log_norm_term_grad = None
            k_grad_from_next = None
            v_grad_from_next = None
            handle_memory = None
            handle_lnt = None
            handle_prev_k = None
            handle_prev_v = None

        # recompute attention output
        if sca_for_save is None:
            assert sca_aux is None
            sca_out, sca_for_save, sca_aux = recompute_sliding_chunk_attention(
                sq, sk, xv_, attention_chunk_size, sca_scale, prev_sk, prev_sv, bos_mask, segment_idx,
                attention_dropout, fp32_attn_output, sca_backend, attn_w_rng_state
            )
        else:
            sca_out = sca_for_save.to(x.dtype)

        # B x L x E
        sca_out = rearrange(sca_out, 'b l h v -> b l (h v)')
        # B x L x H x S -> B x H x L x S
        aq = aq.transpose(1, 2)
        aqk = aqk.transpose(1, 2)
        akk = akk.transpose(1, 2)
        av = xv_.transpose(1, 2)
        memory_outs, lnt_outs, kv_outs, kk_outs, prev_outs = adaptive_working_memory_accum_fwd(
            aq, aqk, akk, av, attention_chunk_size, memory, log_norm_term, prev_aqk, prev_akk, prev_av,
            segment_idx, prev_segment_count, awm_orthogonal_update, rmsnorm_eps, awk_out is None
        )
        accum_memory, memory_residual, memory_mask = memory_outs
        log_norm_term, curr_log_norm_term, accum_log_norm_term, ratio = lnt_outs
        awk_kkey, awk_rvalue, awk_rvalue_rstd, awk_vvalue, awk_avalue, awk_out_rec, awk_out_mask = kv_outs
        ak_fp32, akk_fp32, awk_key_mask = kk_outs
        prev_kkey, prev_k_fp32, prev_kk_fp32, prev_k_mask = prev_outs

        if awk_out is None:
            awk_out = rearrange(awk_out_rec, 'b h l v -> b l (h v)')
        else:
            assert awk_out_rec is None

        # recompute attn out
        # B x L x E
        rmx = F.linear(mx, wr, br)
        r = attn_gate_fn(rmx)
        attn_norm, attn_rstd = rmsnorm_fwd(sca_out + awk_out, None, local_heads, rmsnorm_eps)
        attn = torch.mul(attn_norm, r, out=awk_out)
        attn, attn_noise = memory_efficient_dropout_fwd(attn, hidden_dropout, True, attn_out_rng_state)

        if handle_xh is not None:
            handle_xh.wait()
            xh_grad = reshape_gathered_tensor_along_specific_dim(xh_grad, gather_dim=2)

        # B x L x E
        attn_grad = xh_grad.matmul(wo)
        attn_grad = memory_efficient_dropout_bwd(attn_grad, hidden_dropout, attn_out_rng_state, attn_noise)
        attn_norm_grad = torch.mul(attn_grad, r, out=r)
        sca_out_grad = mem_effn_rmsnorm_bwd(attn_norm_grad, attn_norm, attn_rstd, local_heads)
        # B x L x E -> B x L x H x E
        sca_out_grad = rearrange(sca_out_grad, 'b l (h v) -> b l h v', h=local_heads)
        # B x L x H x E -> B x H x L x E
        awk_out_grad = sca_out_grad.transpose(1, 2)

        if send_to_next:
            conv_state_q_grad, handle_conv_q = recv_prev_conv_state_grad_from_next(x, width, local_heads * head_dim)
        else:
            conv_state_q_grad = None
            handle_conv_q = None

        if recv_from_prev:
            if handle_memory is not None:
                handle_memory.wait()
                handle_lnt.wait()

            (
                aq_grad, aqk_grad, akk_grad, av_grad,
                memory_grad, log_norm_term_grad,
                prev_aqk_grad, prev_akk_grad, prev_av_grad
            ) = adaptive_working_memory_bwd(
                awk_out_grad, memory_grad, log_norm_term_grad,
                aq, aqk, awk_kkey, av, attention_chunk_size,
                awk_rvalue, awk_rvalue_rstd, awk_vvalue, awk_avalue,
                ak_fp32, akk_fp32, awk_key_mask,
                accum_memory, memory_residual, memory_mask,
                log_norm_term, curr_log_norm_term, accum_log_norm_term, ratio,
                awk_out_mask, prev_aqk, prev_kkey, prev_av,
                prev_k_fp32, prev_kk_fp32, prev_k_mask, awm_orthogonal_update
            )

            assert memory_grad is not None and log_norm_term_grad is not None
            _, handle8 = send_memory_grad_to_prev(memory_grad)
            _, handle7 = send_log_norm_term_grad_to_prev(log_norm_term_grad)

            sq_grad, sk_grad, xv_grad, prev_sk_grad, prev_sv_grad = sliding_chunk_attention_bwd(
                sca_out_grad, sq, sk, xv_, sca_for_save, sca_aux, attention_chunk_size, sca_scale,
                prev_sk, prev_sv, bos_mask, segment_idx, deterministic, sca_backend
            )
            assert prev_aqk_grad is not None and prev_akk_grad is not None and prev_av_grad is not None
            assert prev_sk_grad is not None and prev_sv_grad is not None
            prev_aqk_grad = prev_aqk_grad.transpose(1, 2)
            prev_akk_grad = prev_akk_grad.transpose(1, 2)
            _, handle6 = send_prev_key_or_value_grad_to_prev(torch.cat([prev_sk_grad, prev_aqk_grad, prev_akk_grad], dim=3))
            prev_v_grad = prev_sv_grad + prev_av_grad.transpose(1, 2)
            _, handle5 = send_prev_key_or_value_grad_to_prev(prev_v_grad)
        else:
            sq_grad, sk_grad, xv_grad, prev_sk_grad, prev_sv_grad = sliding_chunk_attention_bwd(
                sca_out_grad, sq, sk, xv_, sca_for_save, sca_aux, attention_chunk_size, sca_scale,
                prev_sk, prev_sv, bos_mask, segment_idx, deterministic, sca_backend
            )

            if handle_memory is not None:
                handle_memory.wait()
                handle_lnt.wait()

            (
                aq_grad, aqk_grad, akk_grad, av_grad,
                memory_grad, log_norm_term_grad,
                prev_aqk_grad, prev_akk_grad, prev_av_grad
            ) = adaptive_working_memory_bwd(
                awk_out_grad, memory_grad, log_norm_term_grad,
                aq, aqk, awk_kkey, av, attention_chunk_size,
                awk_rvalue, awk_rvalue_rstd, awk_vvalue, awk_avalue,
                ak_fp32, akk_fp32, awk_key_mask,
                accum_memory, memory_residual, memory_mask,
                log_norm_term, curr_log_norm_term, accum_log_norm_term, ratio,
                awk_out_mask, prev_aqk, prev_kkey, prev_av,
                prev_k_fp32, prev_kk_fp32, prev_k_mask, awm_orthogonal_update
            )

            handle5 = None
            handle6 = None
            handle7 = None
            handle8 = None

        if send_to_next:
            conv_state_k_grad, handle_conv_k = recv_prev_conv_state_grad_from_next(x, width, local_kv_heads * head_dim)
            conv_state_v_grad, handle_conv_v = recv_prev_conv_state_grad_from_next(x, width, local_kv_heads * v_head_dim)
        else:
            conv_state_k_grad = None
            conv_state_v_grad = None
            handle_conv_k = None
            handle_conv_v = None

        # B x L x E
        r_grad = torch.mul(attn_grad, attn_norm, out=attn_grad)
        # B*L x D
        rmx_grad = rearrange(attn_gate_fn_bwd(r_grad, rmx), 'b l d -> (b l) d')
        mx_grad = torch.mm(rmx_grad, wr)

        # apply rotary embeddings
        sq_grad = apply_rope(sq_grad, freqs_cis, head_dim, rope_head_dim, True)
        # q norm grad
        xq_grad, qnorm_w_grad = rmsnorm_bwd(rearrange(sq_grad, 'b l h s -> b l (h s)'), xq_, sq_rstd, qnorm_w, local_heads)

        # B x H x L x S -> B x L x H x S
        aq_grad = aq_grad.transpose(1, 2)
        aq_grad = torch.ops.aten._softmax_backward_data(
            aq_grad.float(), aq_fp32, -1, torch.float32
        ).to(aq)

        aq_grad = rearrange(aq_grad, 'b l h s -> b l (h s)')
        xq_grad = torch.add(xq_grad, aq_grad, out=xq_grad)

        # B x K x L x S -> B x L x K x S
        aqk_grad = aqk_grad.transpose(1, 2)
        akk_grad = akk_grad.transpose(1, 2)
        if handle_prev_k is not None:
            handle_prev_k.wait()
            (
                sk_grad_from_next, aqk_grad_from_next, akk_grad_from_next
            ) = torch.split(k_grad_from_next, [head_dim, head_dim, head_dim], dim=-1)
            sk_grad[:, (seq_len - attention_chunk_size):] += sk_grad_from_next
            aqk_grad[:, (seq_len - attention_chunk_size):] += aqk_grad_from_next
            akk_grad[:, (seq_len - attention_chunk_size):] += akk_grad_from_next

        # apply rotary embeddings
        sk_grad = apply_rope(sk_grad, freqs_cis, head_dim, rope_head_dim, True)
        # k norm grad
        xk_grad = mem_effn_rmsnorm_bwd(rearrange(sk_grad, 'b l k s -> b l (k s)'), sk, sk_rstd, local_kv_heads)

        aqk_grad = torch.ops.aten._softmax_backward_data(
            aqk_grad.float(), aqk_fp32, -1, torch.float32
        ).to(aqk)
        ak_grad = rearrange(torch.add(akk_grad, aqk_grad, out=aqk_grad), 'b l k s -> b l (k s)')

        # grads of xk & xv
        xk_grad = torch.add(xk_grad, ak_grad, out=xk_grad)
        xv_grad = torch.add(xv_grad, av_grad.transpose(1, 2), out=xv_grad)
        if handle_prev_v is not None:
            handle_prev_v.wait()
            xv_grad[:, (seq_len - attention_chunk_size):] += v_grad_from_next
        xv_grad = rearrange(xv_grad, 'b l k v -> b l (k v)')

        # xq grad from causal conv
        if handle_conv_q is not None:
            handle_conv_q.wait()
        xq_grad, conv_state_q_grad, q_conv_w_grad, _ = causal_conv1d_bwd(
            xq_grad, conv_state_q_grad, xq, q_conv_w, None, conv_state_q, bos_mask_curr, None,
            causal_conv_weight_normalization, causal_conv_backend, deterministic
        )

        if recv_from_prev:
            _, handle4 = send_prev_conv_state_grad_to_prev(conv_state_q_grad)
        else:
            handle4 = None

        # xk grad from causal conv
        if handle_conv_k is not None:
            handle_conv_k.wait()
        xk_grad, conv_state_k_grad, k_conv_w_grad, _ = causal_conv1d_bwd(
            xk_grad, conv_state_k_grad, xk, k_conv_w, None, conv_state_k, bos_mask_curr, None,
            causal_conv_weight_normalization, causal_conv_backend, deterministic
        )

        if recv_from_prev:
            _, handle3 = send_prev_conv_state_grad_to_prev(conv_state_k_grad)
        else:
            handle3 = None

        # xv grad from causal conv
        if handle_conv_v is not None:
            handle_conv_v.wait()
        xv_grad, conv_state_v_grad, v_conv_w_grad, _ = causal_conv1d_bwd(
            xv_grad, conv_state_v_grad, xv, v_conv_w, None, conv_state_v, bos_mask_curr, 'silu',
            causal_conv_weight_normalization, causal_conv_backend, deterministic
        )

        if recv_from_prev:
            _, handle2 = send_prev_conv_state_grad_to_prev(conv_state_v_grad)
        else:
            handle2 = None

        if send_to_next:
            mean_var_grad, handle_tsn = recv_mean_var_grad_from_next(x, timenorm_local_groups)
            tsn_mean_grad = tsn_var_grad = None
        else:
            tsn_mean_grad = tsn_var_grad = torch.zeros_like(prev_mean)
            mean_var_grad = None
            handle_tsn = None

        # mx grad from qk
        xq_grad = rearrange(xq_grad, 'b l d -> (b l) d')
        xk_grad = rearrange(xk_grad, 'b l d -> (b l) d')
        xv_grad = rearrange(xv_grad, 'b l d -> (b l) d')
        mx_grad = torch.addmm(mx_grad, xq_grad, wq, out=mx_grad)
        mx_grad = torch.addmm(mx_grad, xk_grad, wk, out=mx_grad)
        mx_grad = torch.addmm(mx_grad, xv_grad, wv, out=mx_grad)

        # reduce mx grad
        # B x L x D
        mx_grad = rearrange(mx_grad, '(b l) d -> b l d', b=bsz)
        mx_grad, handle_mx = reduce_scatter(mx_grad, parallel_region='model', async_op=True)

        # grads for wo
        attn_flat = attn.flatten(end_dim=-2)
        wo_grad = xh_grad.flatten(end_dim=-2).t().matmul(attn_flat)

        # grads for wk, wv & bv
        mx_flat = mx.flatten(end_dim=-2)
        wk_grad = torch.mm(xk_grad.t(), mx_flat)
        wv_grad = torch.mm(xv_grad.t(), mx_flat)
        bv_grad = None if bv is None else xv_grad.sum(dim=0)

        if handle_mx is not None:
            handle_mx.wait()

        if handle_tsn is not None:
            handle_tsn.wait()
            tsn_mean_grad, tsn_var_grad = torch.unbind(mean_var_grad, dim=2)

        x_grad, prev_mean_grad, prev_var_grad, timenorm_w_grad, timenorm_b_grad = timenorm_bwd(
            mx_grad, tsn_mean_grad, tsn_var_grad, x, bos_mask_curr, prev_count, prev_mean, cummean, cumrstd,
            timenorm_w, timenorm_b, timenorm_local_groups, timenorm_beta1, timenorm_beta2, timenorm_eps, False, timenorm_backend
        )

        if recv_from_prev:
            _, handle1 = send_mean_var_grad_to_prev(prev_mean_grad, prev_var_grad)
        else:
            handle1 = None

        # residual connection
        x_grad = torch.add(x_grad, residual_grad, out=x_grad)
        wr_grad = torch.mm(rmx_grad.t(), mx_flat)
        br_grad = None if br is None else rmx_grad.sum(dim=0)
        wq_grad = torch.mm(xq_grad.t(), mx_flat)

        if recv_from_prev:
            handle8.wait()
            handle7.wait()
            handle6.wait()
            handle5.wait()
            handle4.wait()
            handle3.wait()
            handle2.wait()
            handle1.wait()

        return (
            x_grad,
            None,  # freqs_cis
            None,  # bos mask
            None,  # segment idx
            None,  # prev segment count
            timenorm_w_grad,
            timenorm_b_grad,
            None,  # tsn prior count
            None,  # tsn prior mean
            None,  # tsn prior logv
            None,  # tsn local groups
            None,  # tsn beta1
            None,  # tsn beta2
            None,  # tsn backend
            wq_grad,
            wk_grad,
            wv_grad,
            bv_grad,
            wr_grad,
            br_grad,
            None,
            wo_grad,
            q_conv_w_grad,
            k_conv_w_grad,
            v_conv_w_grad,
            None,  # causal_conv_backend
            None,  # causal_conv_weight_normalization
            qnorm_w_grad,
            None,  # local heads
            None,  # local kv heads
            None,  # chunk size
            None,  # head dim
            None,  # v head dim
            None,  # rope head dim
            None,  # attn gate func
            None,  # sca backend
            None,  # awm_orthogonal_update
            None,  # attn_res_w
            ffn_norm_w_grad,
            ffn_norm_b_grad,
            None,  # ffn_norm local groups
            fc1_w_grad,
            fc2_w_grad,
            fc3_w_grad,
            None,  # ffn_res_w
            None,  # dropout
            None,  # attention dropout
            None,  # hidden dropout
            None,  # swiglu
            None,  # timenorm_eps
            None,  # layernorm_eps
            None,  # rmsnorm_eps
            None,  # apply_rmsnorm
            None,  # residual_func
            None,  # residual_heads
            None,  # fp32_attn_output
            None,  # deterministic
            None,  # recompute_q
            None,  # recompute_kv
            None,  # recompute_sca
            None,  # recompute_awk
            None,  # recompute_fc1_out
            None,  # recompute_fc3_out
        )


class GekkoOutputLayerFunction(torch.autograd.Function):
    """

    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,  # (bsz, slen, d/MP)
        y: torch.Tensor,  # (bsz, slen)
        mask: Optional[torch.Tensor],  # (bsz, slen)
        bos_mask: Optional[torch.Tensor],  # (bsz, slen)
        timenorm_w: torch.Tensor,  # (d/MP)
        timenorm_b: torch.Tensor,  # (d/MP)
        timenorm_prior_count: torch.Tensor,
        timenorm_prior_mean: torch.Tensor,
        timenorm_prior_logv_or_var: torch.Tensor,
        timenorm_local_groups: int,
        timenorm_beta1: Optional[float],
        timenorm_beta2: Optional[float],
        timenorm_backend: str,
        wo: torch.Tensor,  # (voc/MP, d)
        eps: float,
        recompute_logits: bool,
    ):
        assert timenorm_local_groups is not None
        bsz, seq_len, _ = x.size()
        recv_from_prev = should_recv_from_prev()
        send_to_next = should_send_to_next()

        # recv prev tensors
        if recv_from_prev:
            prev_mean_var, handle1 = recv_prev_mean_var(x, timenorm_local_groups)
            prev_count, handle2 = recv_prev_count(x, bos_mask)
            handle1.wait()
            prev_mean, prev_var = torch.unbind(prev_mean_var, dim=2)
            if handle2 is not None:
                handle2.wait()
        else:
            prev_count = timenorm_prior_count.expand(bsz).contiguous()
            prev_mean = timenorm_prior_mean.type_as(x).expand(bsz, -1).contiguous()
            prev_var = timenorm_prior_logv_or_var.type_as(x).expand(bsz, -1).contiguous()

        # Final Timestep Normalization
        # B x L x D
        out_tsn, tsn_count, tsn_mean, tsn_var, _, _ = timenorm_fwd(
            x, bos_mask, prev_count, prev_mean, prev_var, timenorm_w, timenorm_b,
            timenorm_local_groups, timenorm_beta1, timenorm_beta2, eps, timenorm_backend
        )

        if send_to_next:
            _, handle1 = send_mean_var_to_next(tsn_mean, tsn_var)
            _, handle2 = send_count_to_next(tsn_count, bos_mask)
        else:
            handle1 = None
            handle2 = None

        loss, grad_tsn, grad_wo = fused_linear_cross_entropy_fwd(out_tsn, wo, y, mask, compute_grad=(not recompute_logits))

        if send_to_next:
            handle1.wait()
            if handle2 is not None:
                handle2.wait()

        ctx.save_for_backward(
            x,  # (bsz, slen, d/MP)
            y if recompute_logits else None,  # (bsz, slen)
            mask if recompute_logits else None,  # (bsz, slen)
            bos_mask,  # (bsz, slen)
            timenorm_w,  # (d/MP)
            timenorm_b,  # (d/MP)
            wo,  # (voc/MP, d)
            prev_count,  # (bsz)
            prev_mean,  # (bsz, n_groups/MP)
            prev_var,  # (bsz, n_groups/MP)
            grad_tsn,  # (bsz, slen, d/MP)
            grad_wo,  # (voc/MP, d)
        )
        ctx.recv_from_prev = recv_from_prev
        ctx.send_to_next = send_to_next
        ctx.timenorm_local_groups = timenorm_local_groups
        ctx.timenorm_beta1 = timenorm_beta1
        ctx.timenorm_beta2 = timenorm_beta2
        ctx.timenorm_backend = timenorm_backend
        ctx.eps = eps

        return loss, None

    @staticmethod
    def backward(ctx, loss_grad, cache):
        assert cache is None

        (
            x,  # (bsz, slen, d/MP)
            y,  # (bsz, slen)
            mask,  # (bsz, slen)
            bos_mask,  # (bsz, slen)
            timenorm_w,  # (d/MP)
            timenorm_b,  # (d/MP)
            wo,  # (voc/MP, d)
            prev_count,  # (bsz)
            prev_mean,  # (bsz, n_groups/MP)
            prev_var,  # (bsz, n_groups/MP)
            grad_tsn,  # (bsz, slen, d/MP)
            grad_wo,  # (voc/MP, d)
        ) = ctx.saved_tensors

        recv_from_prev = ctx.recv_from_prev
        send_to_next = ctx.send_to_next
        timenorm_local_groups = ctx.timenorm_local_groups
        timenorm_beta1 = ctx.timenorm_beta1
        timenorm_beta2 = ctx.timenorm_beta2
        timenorm_backend = ctx.timenorm_backend
        eps = ctx.eps

        # recompute tsn
        # B x L x D
        out_tsn, tsn_count, tsn_mean, tsn_var, cummean, cumrstd = timenorm_fwd(
            x, bos_mask, prev_count, prev_mean, prev_var, timenorm_w, timenorm_b,
            timenorm_local_groups, timenorm_beta1, timenorm_beta2, eps, timenorm_backend
        )

        if send_to_next:
            mean_var_grad, handle1 = recv_mean_var_grad_from_next(x, timenorm_local_groups)
        else:
            mean_var_grad = None
            handle1 = None

        tsn_grad, wo_grad = fused_linear_cross_entropy_bwd(loss_grad, grad_tsn, grad_wo, out_tsn, wo, y, mask)

        if handle1 is not None:
            handle1.wait()
            tsn_mean_grad, tsn_var_grad = torch.unbind(mean_var_grad, dim=2)
        else:
            tsn_mean_grad = tsn_var_grad = torch.zeros_like(prev_mean)

        x_grad, prev_mean_grad, prev_var_grad, timenorm_w_grad, timenorm_b_grad = timenorm_bwd(
            tsn_grad, tsn_mean_grad, tsn_var_grad, x, bos_mask, prev_count, prev_mean, cummean, cumrstd,
            timenorm_w, timenorm_b, timenorm_local_groups, timenorm_beta1, timenorm_beta2, eps, False, timenorm_backend
        )

        if recv_from_prev:
            _, handle1 = send_mean_var_grad_to_prev(prev_mean_grad, prev_var_grad)
        else:
            handle1 = None

        if recv_from_prev:
            handle1.wait()

        return (
            x_grad,
            None,  # y
            None,  # mask
            None,  # bos mask
            timenorm_w_grad,
            timenorm_b_grad,
            None,  # tsn prior count
            None,  # tsn prior mean
            None,  # tsn prior logv
            None,  # tsn local groups
            None,  # tsn beta1
            None,  # tsn beta2
            None,  # tsn backend
            wo_grad,
            None,  # eps
            None,  # recompute logits
        )
