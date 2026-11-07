from typing import Optional, Tuple, Any
import math
import os
from functools import partial
import torch
from torch.nn import functional as F
from einops import rearrange

from xllm.distributed import get_model_parallel_world_size
from xllm.distributed.utils import reshape_gathered_tensor_along_specific_dim
from xllm.distributed.initialize import get_model_parallel_world_size
from xllm.health_metrics import get_active_health_collector
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
    awm_cute_bwd_supported,
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
    swiglu_recompute,
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


def _cute_qk(x: torch.Tensor, groups: int, rope_head_dim: int, freqs_cis: Optional[torch.Tensor]) -> bool:
    """Whether the fused CuTe DSL q / k preparation (rmsnorm + rope + softmax and its backward; Blackwell, bf16;
    env XLLM_CUTE_QK=0 disables it) handles this input."""
    try:
        from xllm.modules.fused_ops.cute.qk_prep import qk_prep_supported
    except ImportError:
        return False
    r = rope_head_dim if freqs_cis is not None else 0
    if r > 0 and (freqs_cis.dtype != torch.complex64 or tuple(freqs_cis.shape) != (x.shape[1], r // 2)):
        return False
    return qk_prep_supported(x, groups, r)


def _fused_conv_qk(x: torch.Tensor, conv_w: torch.Tensor, groups: int, rope_head_dim: int, freqs_cis: Optional[torch.Tensor],
                   backend: str) -> bool:
    """Whether the causal conv and the q / k preparation run as one CuTe DSL kernel (forward and backward; triton conv
    backend semantics, no conv state; env XLLM_CONV_QK=0 disables it). The callers also require no conv state."""
    if backend != 'triton' or not _cute_qk(x, groups, rope_head_dim, freqs_cis):
        return False
    try:
        from xllm.modules.fused_ops.cute.conv_qk_prep import conv_qk_prep_supported
    except ImportError:
        return False
    return conv_qk_prep_supported(x, conv_w, groups, rope_head_dim if freqs_cis is not None else 0)


def _cute_gate(x: torch.Tensor, heads: int, gate: str, dropout: float) -> bool:
    """Whether the fused CuTe DSL attention output gate (rmsnorm(sca + awk) * gate(rmx) and its backward; Blackwell,
    bf16, no hidden dropout; env XLLM_CUTE_GATE=0 disables it) handles this input."""
    try:
        from xllm.modules.fused_ops.cute.attn_gate import attn_gate_supported
    except ImportError:
        return False
    return attn_gate_supported(x, heads, gate, dropout)


_SAVE_QK_PREP = os.environ.get("XLLM_GEKKO_SAVE_QK_PREP", "0") == "1"
# opt-in (XLLM_GEKKO_SAVE_RMX=1, implied by XLLM_GEKKO_SAVE_QK_PREP): keep only the gate projection rmx for the backward
# (one [B, L, E] bf16 tensor per layer instead of a GEMM recompute)
_SAVE_RMX = _SAVE_QK_PREP or os.environ.get("XLLM_GEKKO_SAVE_RMX", "0") == "1"
_NAN_PROBE = os.environ.get("XLLM_NAN_PROBE", "0") == "1"
# opt-in (XLLM_FP8_KEEP_X=1, fp8 GEMMs only): keep the fp8 operands of mx and xf (and the norm statistics) for the
# backward also with recompute_conv_qk=True, instead of recomputing the time norm / FFN norm and quantising again
# (+2 bytes per element of mx and xf per layer)
_FP8_KEEP_X = os.environ.get("XLLM_FP8_KEEP_X", "0") == "1"  # debugging: report non-finite backward intermediates


def _nan_probe(tag: str, **tensors):
    if not _NAN_PROBE:
        return
    for name, t in tensors.items():
        if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all():
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(f"NANPROBE rank {rank} {tag}:{name} {tuple(t.shape)} nonfinite {(~torch.isfinite(t)).sum().item()}", flush=True)


def _fp8_ok(x: torch.Tensor, context_parallel: bool, tokens: int, no_dropout_single_mp: bool, swiglu: bool, *weights) -> bool:
    """Whether the MXFP8 GEMMs (``ModelConf.fp8_gemm``) handle this call: Blackwell, bf16, no context parallelism (the
    received / sent chunks are not quantised), dropout 0 and no model parallelism (the residual is added after the
    GEMM), SwiGLU, and the token count and every weight dimension multiples of 128."""
    if context_parallel or not no_dropout_single_mp or not swiglu or x.dtype != torch.bfloat16:
        return False
    try:
        from xllm.modules.fused_ops.mxfp8 import mx_supported
    except ImportError:
        return False
    return mx_supported(x.device, tokens, *[d for w in weights if w is not None for d in w.shape])


class GekkoBlockFunction(torch.autograd.Function):
    """

    """

    @staticmethod
    def forward(
        ctx: Any,
        layer_id: int,
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
        awm_backend: str,
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
        attn_stability_control: int,
        deterministic: bool,
        recompute_q: bool,
        recompute_kv: bool,
        recompute_sca: bool,
        recompute_awk: bool,
        recompute_fc1_out: bool,
        recompute_fc3_out: bool,
        recompute_conv_qk: bool = True,
        fp8_gemm: bool = False,
        fp8_attn_dp: bool = False,
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
        # a sampled health-logging step (HealthMetricCollector): its statistics need the pre-RoPE q / k, the gate
        # activation and the memory update ratio, which the fused conv + q / k prep, q / k prep, gate and AWM kernels
        # never materialise, so such a step runs the reference path (the backward follows through ctx.health)
        health_collector = get_active_health_collector()
        health = health_collector is not None
        if health:
            awm_backend = 'torch'

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
        mx, tsn_count, tsn_mean, tsn_var, cummean, cumrstd, tsn_step_count = timenorm_fwd(
            x, bos_mask_curr, prev_count, prev_mean, prev_var, timenorm_w, timenorm_b,
            timenorm_local_groups, timenorm_beta1, timenorm_beta2, timenorm_eps, timenorm_backend, want_step_count=True
        )

        if send_to_next:
            _, handle1 = send_mean_var_to_next(tsn_mean, tsn_var)
            _, handle2 = send_count_to_next(tsn_count, bos_mask)
        else:
            handle1 = None
            handle2 = None

        mx, _ = all_gather(mx, parallel_region='model', async_op=False)

        # MXFP8 GEMMs (ModelConf.fp8_gemm, xllm/modules/fused_ops/mxfp8.py): the q/k/v/r weights and mx quantised once
        fp8 = fp8_gemm and _fp8_ok(x, recv_from_prev or send_to_next, bsz * seq_len,
                                   dropout == 0.0 and hidden_dropout == 0.0 and get_model_parallel_world_size() == 1,
                                   swiglu, wq, wk, wv, wr, wo, fc1_w, fc2_w, fc3_w)
        if fp8:
            from xllm.modules.fused_ops.mxfp8 import mx_quantize, MxLinearSet
            w_qkvr = MxLinearSet([wq, wk, wv, wr])
            # recompute_conv_qk=False (or XLLM_FP8_KEEP_X): both operands are kept for the backward instead of the
            # bf16 mx (same bytes)
            keep_x = not recompute_conv_qk or _FP8_KEEP_X
            mx_row, mx_col = mx_quantize(mx.reshape(-1, mx.shape[-1]), colwise=keep_x)
            _lin = lambda i, b: w_qkvr.forward(i, mx_row, b).view(bsz, seq_len, -1)

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
        xv = _lin(2, bv) if fp8 else F.linear(mx, wv, bv)
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
        xk = _lin(1, None) if fp8 else F.linear(mx, wk, None)
        if handle_conv_k is not None:
            handle_conv_k.wait()
        fc = freqs_cis if rope_head_dim > 0 else None
        fused_k = not health and conv_state_k is None and not send_to_next and _fused_conv_qk(
            xk, k_conv_w, local_kv_heads, rope_head_dim, freqs_cis, causal_conv_backend)
        if fused_k:
            # conv, rmsnorm, rope and softmax in one kernel (the conv output is kept: the memory's key input)
            from xllm.modules.fused_ops.cute.conv_qk_prep import conv_qk_prep_fwd
            xk_, sk, aqk = conv_qk_prep_fwd(xk, k_conv_w, causal_conv_weight_normalization, bos_mask_curr, local_kv_heads, None,
                                            fc, rope_head_dim, rmsnorm_eps, want_c=True)
        else:
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

        # B x L x K x S
        akk = rearrange(xk_, 'b l (k s) -> b l k s', k=local_kv_heads)
        health_sca_key_l2 = None
        if fused_k:
            pass
        elif not health and _cute_qk(xk_, local_kv_heads, rope_head_dim, freqs_cis):
            # rmsnorm, rope and softmax from one read of xk_
            from xllm.modules.fused_ops.cute.qk_prep import qk_prep_fwd
            sk, _, aqk, _ = qk_prep_fwd(xk_, local_kv_heads, None, freqs_cis if rope_head_dim > 0 else None, rope_head_dim, rmsnorm_eps)
        else:
            sk, _ = rmsnorm_fwd(xk_, None, local_kv_heads, rmsnorm_eps)
            sk = rearrange(sk, 'b l (k s) -> b l k s', k=local_kv_heads)
            if health and health_collector.enable_gekko_stats:
                # partial RoPE mutates sk: reduce its pre-RoPE vectors first
                health_sca_key_l2 = torch.linalg.vector_norm(sk, dim=-1, dtype=torch.float32)
            # apply rotary embeddings
            sk = apply_rope(sk, freqs_cis, head_dim, rope_head_dim, False)
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
        xq = _lin(0, None) if fp8 else F.linear(mx, wq, None)
        if handle_conv_q is not None:
            handle_conv_q.wait()
        fused_q = not health and conv_state_q is None and not send_to_next and _fused_conv_qk(
            xq, q_conv_w, local_heads, rope_head_dim, freqs_cis, causal_conv_backend)
        if fused_q:
            # conv, q norm, rope and softmax in one kernel; the conv output is not needed (recomputed in the backward)
            from xllm.modules.fused_ops.cute.conv_qk_prep import conv_qk_prep_fwd
            xq_, sq, aq = conv_qk_prep_fwd(xq, q_conv_w, causal_conv_weight_normalization, bos_mask_curr, local_heads, qnorm_w,
                                           fc, rope_head_dim, rmsnorm_eps)
        else:
            xq_, final_conv_state_q, _ = causal_conv1d_fwd(
                xq, q_conv_w, None, conv_state_q, bos_mask_curr, send_to_next, None,
                causal_conv_weight_normalization, causal_conv_backend, deterministic
            )

        if send_to_next:
            _, handle7 = send_prev_conv_state_to_next(final_conv_state_q.contiguous())
        else:
            handle7 = None

        if fused_q:
            pass
        elif not health and _cute_qk(xq_, local_heads, rope_head_dim, freqs_cis):
            from xllm.modules.fused_ops.cute.qk_prep import qk_prep_fwd
            sq, _, aq, _ = qk_prep_fwd(xq_, local_heads, qnorm_w, freqs_cis if rope_head_dim > 0 else None, rope_head_dim, rmsnorm_eps)
        else:
            sq, _ = rmsnorm_fwd(xq_, qnorm_w, local_heads, rmsnorm_eps)
            sq = rearrange(sq, 'b l (h s) -> b l h s', h=local_heads)
            if health:
                # both SCA norms are measured before RoPE
                health_collector.record_gekko_query_key_stats(
                    layer_id, xq_.reshape(bsz, seq_len, local_heads, head_dim),
                    xk_.reshape(bsz, seq_len, local_kv_heads, head_dim),
                    sq, sca_key_l2=health_sca_key_l2,
                )
            # apply rotary embeddings
            sq = apply_rope(sq, freqs_cis, head_dim, rope_head_dim, False)
            # B x L x H x S
            aq = rearrange(xq_, 'b l (h s) -> b l h s', h=local_heads)
            # apply softmax to aq
            aq = F.softmax(aq, dim=-1, dtype=torch.float32).to(aq)

        # the q/k prep outputs can only be kept for the backward when both came from the fused kernel
        # (XLLM_GEKKO_SAVE_QK_PREP=1 is recompute_conv_qk=False; the fused q path has no conv output, so test xq)
        recompute_conv_qk = recompute_conv_qk and not _SAVE_QK_PREP
        qk_saved = not health and not recompute_conv_qk and _cute_qk(xk_, local_kv_heads, rope_head_dim, freqs_cis) \
            and _cute_qk(xq, local_heads, rope_head_dim, freqs_cis)
        qk_keep = (sq, sk, aq, aqk) if qk_saved else (None,) * 4  # [B, L, H, S] (aq / aqk are transposed below)
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
                awm_orthogonal_update, rmsnorm_eps, health_layer_id=layer_id, backend=awm_backend
            )
            awm_workspace = ()

            _, handle8 = send_memory_to_next(new_memory)
            _, handle9 = send_log_norm_term_to_next(new_lnt)

            sca_out, sca_for_save, sca_aux, attn_w_rng_state = sliding_chunk_attention_fwd(
                sq, sk, xv_, attention_chunk_size, sca_scale, prev_sk, prev_sv,
                bos_mask, segment_idx, attention_dropout, attn_stability_control, sca_backend, not recompute_sca
            )
            # B x L x E
            rmx = _lin(3, br) if fp8 else F.linear(mx, wr, br)
        else:
            sca_out, sca_for_save, sca_aux, attn_w_rng_state = sliding_chunk_attention_fwd(
                sq, sk, xv_, attention_chunk_size, sca_scale, prev_sk, prev_sv,
                bos_mask, segment_idx, attention_dropout, attn_stability_control, sca_backend, not recompute_sca
            )
            # B x L x E
            rmx = _lin(3, br) if fp8 else F.linear(mx, wr, br)

            if handle_memory is not None:
                handle_memory.wait()
                handle_lnt.wait()

            # B x L x H x S -> B x H x L x S
            aq = aq.transpose(1, 2)
            aqk = aqk.transpose(1, 2)
            akk = akk.transpose(1, 2)
            av = xv_.transpose(1, 2)

            if awm_backend == 'cute' and awm_cute_bwd_supported(aq, av, attention_chunk_size, awm_orthogonal_update, memory, prev_akk):
                # CuTe DSL kernels: keep the workspaces of the forward for the backward
                from xllm.modules.fused_ops.cute.adaptive_working_memory import adaptive_working_memory_fwd as cute_awm_fwd
                awk_out, awk_mask, new_memory, new_lnt, awm_workspace = cute_awm_fwd(
                    aq, aqk, akk, av, attention_chunk_size, memory, log_norm_term,
                    prev_aqk, prev_akk, prev_av, segment_idx, prev_segment_count, return_workspace=True
                )
            else:
                awk_out, awk_mask, new_memory, new_lnt = adaptive_working_memory_fwd(
                    aq, aqk, akk, av, attention_chunk_size, memory, log_norm_term,
                    prev_aqk, prev_akk, prev_av, segment_idx, prev_segment_count,
                    awm_orthogonal_update, rmsnorm_eps, health_layer_id=layer_id, backend=awm_backend
                )
                awm_workspace = ()

            handle8 = None
            handle9 = None

        if not recompute_sca and sca_for_save is None:
            assert sca_backend in ('swift', 'cute')  # cute falls back to swift for unsupported shapes
            sca_for_save = sca_out
        # B x L x E
        sca_out = rearrange(sca_out, 'b l h v -> b l (h v)')
        awk_out = rearrange(awk_out, 'b h l v -> b l (h v)')
        attn_q = None
        if fp8 and not health and _cute_gate(sca_out, local_heads, attn_gate_func, hidden_dropout):
            # the gate straight into the fp8 operands of attn (Hopper, value heads of 128 / 256)
            from xllm.modules.fused_ops.mxfp8 import mx_gate_quantize
            attn_q = mx_gate_quantize(sca_out, awk_out, rmx, local_heads, attn_gate_func, rmsnorm_eps)
        if attn_q is not None:
            attn, attn_out_rng_state = None, None
        elif not health and _cute_gate(sca_out, local_heads, attn_gate_func, hidden_dropout):
            # add, per-head rmsnorm, gate activation and multiply in one pass
            from xllm.modules.fused_ops.cute.attn_gate import attn_gate_fwd
            attn = attn_gate_fwd(sca_out, awk_out, rmx, local_heads, attn_gate_func, rmsnorm_eps)
            attn_out_rng_state = None
        else:
            r = attn_gate_fn(rmx)
            if health:
                health_collector.record_gekko_branches(layer_id, sca_out, awk_out, r, awk_mask)
            attn, _ = rmsnorm_fwd(sca_out + awk_out, None, local_heads, rmsnorm_eps)
            attn = torch.mul(attn, r, out=attn)
            attn, attn_out_rng_state = memory_efficient_dropout_fwd(attn, hidden_dropout, True)

        # B x L x E -> B x L x D (+ residual: in the GEMM epilogue when nothing sits in between)
        fuse_residual = dropout == 0.0 and get_model_parallel_world_size() == 1
        if fp8:
            attn_row, attn_col = attn_q if attn_q is not None else mx_quantize(attn.reshape(-1, attn.shape[-1]))
            w_o = MxLinearSet([wo])
            xo = w_o.forward(0, attn_row, residual=residual.reshape(-1, residual.shape[-1])).view(residual.shape)
            xh_rng_state = None
        elif fuse_residual:
            xo = torch.addmm(residual.reshape(-1, residual.shape[-1]), attn.reshape(-1, attn.shape[-1]), wo.t()).view(residual.shape)
            xh_rng_state = None
        else:
            xh, _ = reduce_scatter(F.linear(attn, wo, None), parallel_region='model', async_op=False)
            xh, xh_rng_state = memory_efficient_dropout_fwd(xh, dropout, True)
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

        xf, xo_mean, xo_invvar = layer_or_rmsnorm_fwd(xo, ffn_norm_w, ffn_norm_b, ffn_norm_local_groups, layernorm_eps, rmsnorm_eps,
                                                      apply_rmsnorm)

        # FFN & SwiGLU
        xf, _ = all_gather(xf, parallel_region='model', async_op=False)

        xf = rearrange(xf, 'b l d -> (b l) d')
        fp8_ffn = {'keep_xf': keep_x} if fp8 else None
        h1, h3, out, hidden_rng_state = swiglu_forward(xf, fc1_w, fc2_w, fc3_w, hidden_dropout, swiglu,
                                                       residual=rearrange(residual, 'b l d -> (b l) d') if fuse_residual else None,
                                                       fp8_state=fp8_ffn)
        out = rearrange(out, '(b l) d -> b l d', b=bsz)
        if fuse_residual:
            out_rng_state = None
        else:
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
            # recompute_conv_qk=False: the activations the backward recomputes otherwise (q/k prep outputs only when
            # both came from the fused kernel, see qk_saved); XLLM_GEKKO_SAVE_RMX alone keeps the gate projection;
            # fp8: mx / xf are kept as fp8 operands (ctx.fp8)
            *((None if fp8 else mx, xq_, xk_, xv_, *qk_keep, rmx, None if fp8 else xf, cummean, cumrstd, tsn_step_count, xo_mean, xo_invvar)
              if not recompute_conv_qk else (None,) * 8 + (rmx if _SAVE_RMX else None,) +
              ((None, cummean, cumrstd, tsn_step_count, xo_mean, xo_invvar) if fp8 and keep_x else (None,) * 6)),
            *awm_workspace,  # CuTe AWM forward workspaces (training case), else nothing
        )
        ctx.local_heads = local_heads
        ctx.local_kv_heads = local_kv_heads
        ctx.attention_chunk_size = attention_chunk_size
        ctx.head_dim = head_dim
        ctx.v_head_dim = v_head_dim
        ctx.rope_head_dim = rope_head_dim
        ctx.attn_gate_func = attn_gate_func
        ctx.awm_orthogonal_update = awm_orthogonal_update
        ctx.awm_backend = awm_backend
        ctx.causal_conv_backend = causal_conv_backend
        ctx.fused_q, ctx.fused_k = fused_q, fused_k
        ctx.health = health
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
        ctx.fp8_attn_dp = fp8_attn_dp
        ctx.deterministic = deterministic
        ctx.attn_stability_control = attn_stability_control
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
        # fp8 operands kept for the backward: the quantised weights, attn^T (wo's gradient), the FFN state
        ctx.fp8 = {'w_qkvr': w_qkvr, 'w_o': w_o, 'attn_col': attn_col, 'ffn': fp8_ffn, 'mx_row': mx_row, 'mx_col': mx_col} if fp8 else None

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
            mx_s, xq_s, xk_s, xv_s, sq_s, sk_s, aq_s, aqk_s, rmx_s, xf_s, cummean_s, cumrstd_s, tsn_step_count_s, xo_mean_s, xo_invvar_s,
            *awm_workspace,
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
        awm_backend = ctx.awm_backend
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
        attn_stability_control = ctx.attn_stability_control
        deterministic = ctx.deterministic
        sca_backend = ctx.sca_backend
        fp8 = ctx.fp8
        if fp8:
            from xllm.modules.fused_ops.mxfp8 import mx_quantize, mx_mm

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

        # recompute xf (unless kept from the forward: recompute_conv_qk=False; fp8: as the two fp8 operands)
        keep_fp8 = fp8 is not None and fp8['mx_col'] is not None
        if keep_fp8:
            xf, xo_mean, xo_invvar, handle_xf = None, xo_mean_s, xo_invvar_s, None
        elif xf_s is None:
            xf, xo_mean, xo_invvar = layer_or_rmsnorm_fwd(
                xo, ffn_norm_w, ffn_norm_b, ffn_norm_local_groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm
            )
            xf, handle_xf = all_gather(xf, parallel_region='model', async_op=True)
        else:
            xf, xo_mean, xo_invvar, handle_xf = xf_s, xo_mean_s, xo_invvar_s, None

        out_grad = memory_efficient_dropout_bwd(out_grad, dropout, out_rng_state)
        out_grad, handle_out = all_gather(out_grad, parallel_region='model', async_op=True)

        # recompute mx
        # B x L x D
        if keep_fp8:
            mx, cummean, cumrstd, tsn_step_count = None, cummean_s, cumrstd_s, tsn_step_count_s
        elif mx_s is None:
            mx, tsn_count, tsn_mean, tsn_var, cummean, cumrstd, tsn_step_count = timenorm_fwd(
                x, bos_mask_curr, prev_count, prev_mean, prev_var, timenorm_w, timenorm_b,
                timenorm_local_groups, timenorm_beta1, timenorm_beta2, timenorm_eps, timenorm_backend, want_step_count=True
            )
        else:
            mx, cummean, cumrstd, tsn_step_count = mx_s, cummean_s, cumrstd_s, tsn_step_count_s

        if handle_xf is not None:
            handle_xf.wait()
            xf = reshape_gathered_tensor_along_specific_dim(xf, gather_dim=2)

        if xf_s is None and xf is not None:
            xf = rearrange(xf, 'b l d -> (b l) d')
        if fp8 and not keep_fp8:
            fp8['ffn']['xf_row'], fp8['ffn']['xf_col'] = mx_quantize(xf, rowwise=h1 is None or h3 is None)
        # recompute h1 & h3 for FFN or SwiGLU
        if fp8:
            if h1 is None or h3 is None:
                h1, h3 = fp8['ffn']['w13'].forward_all(fp8['ffn']['xf_row'])  # one GEMM, column halves
        else:
            h1, h3 = swiglu_recompute(xf, fc1_w, fc3_w, h1, h3, hidden_dropout, swiglu)

        if handle_out is not None:
            handle_out.wait()
            out_grad = reshape_gathered_tensor_along_specific_dim(out_grad, gather_dim=2)

        # gather mx (the kept one is the gathered one)
        if mx_s is None and not keep_fp8:
            mx, handle_mx = all_gather(mx, parallel_region='model', async_op=True)
        else:
            handle_mx = None

        out_grad = rearrange(out_grad, 'b l d -> (b l) d')
        # (+ residual_grad, fused into the norm backward's store)
        xo_grad, fc1_w_grad, fc2_w_grad, fc3_w_grad, ffn_norm_w_grad, ffn_norm_b_grad = swiglu_backward(
            out_grad, xf, fc1_w, fc2_w, fc3_w, h1, h3, hidden_dropout, hidden_rng_state,
            xo, xo_mean, xo_invvar, ffn_norm_w, ffn_norm_b, ffn_norm_local_groups, apply_rmsnorm, False, addend=residual_grad,
            fp8_state=fp8['ffn'] if fp8 else None,
        )

        residual_grad = xo_grad
        _nan_probe("ffn", out_grad=out_grad, residual_grad=residual_grad)
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

        fused_q, fused_k = ctx.fused_q, ctx.fused_k
        fc = freqs_cis if rope_head_dim > 0 else None
        if fused_q or fused_k:
            from xllm.modules.fused_ops.cute.conv_qk_prep import conv_qk_prep_fwd, conv_qk_prep_bwd
        if keep_fp8:
            mx_row, mx_col = fp8['mx_row'], fp8['mx_col']
        elif fp8:
            mx_row, mx_col = mx_quantize(mx.reshape(-1, mx.shape[-1]), rowwise=xv is None or xq is None or rmx_s is None)
        # recompute the projections (their weight gradients need them) and, unless kept by the forward, the convs and
        # the q/k prep (fused: conv + q/k prep in one kernel)
        if xv is None:
            assert xk is None
            xv = fp8['w_qkvr'].forward(2, mx_row, bv).view(bsz, seq_len, -1) if fp8 else F.linear(mx, wv, bv)
            xk = fp8['w_qkvr'].forward(1, mx_row, None).view(bsz, seq_len, -1) if fp8 else F.linear(mx, wk, None)
        if xq is None:
            xq = fp8['w_qkvr'].forward(0, mx_row, None).view(bsz, seq_len, -1) if fp8 else F.linear(mx, wq, None)
        if xv_s is None:
            xv_, _, _ = causal_conv1d_fwd(
                xv, v_conv_w, None, conv_state_v, bos_mask_curr, False, 'silu',
                causal_conv_weight_normalization, causal_conv_backend, deterministic
            )
            # B x L x K x V
            xv_ = rearrange(xv_, 'b l (k v) -> b l k v', k=local_kv_heads)
        else:
            xv_ = xv_s
        sk_f = sq_f = None
        if xk_s is not None:
            xk_ = xk_s
        elif fused_k and sq_s is None:
            xk_, sk_f, aqk_f = conv_qk_prep_fwd(xk, k_conv_w, causal_conv_weight_normalization, bos_mask_curr, local_kv_heads,
                                                None, fc, rope_head_dim, rmsnorm_eps, want_c=True)
        else:
            xk_, _, _ = causal_conv1d_fwd(
                xk, k_conv_w, None, conv_state_k, bos_mask_curr, False, None,
                causal_conv_weight_normalization, causal_conv_backend, deterministic
            )
        if xq_s is not None or (fused_q and sq_s is not None):
            xq_ = xq_s  # the fused q backward reads xq, not the conv output
        elif fused_q:
            xq_, sq_f, aq_f = conv_qk_prep_fwd(xq, q_conv_w, causal_conv_weight_normalization, bos_mask_curr, local_heads,
                                               qnorm_w, fc, rope_head_dim, rmsnorm_eps)
        else:
            xq_, _, _ = causal_conv1d_fwd(
                xq, q_conv_w, None, conv_state_q, bos_mask_curr, False, None,
                causal_conv_weight_normalization, causal_conv_backend, deterministic
            )

        # B x L x K|H x S
        akk = rearrange(xk_, 'b l (k s) -> b l k s', k=local_kv_heads)
        cute_qk = not ctx.health and _cute_qk(xq, local_heads, rope_head_dim, freqs_cis) and _cute_qk(xk, local_kv_heads, rope_head_dim, freqs_cis)
        if sq_s is not None:
            # kept from the forward (fused kernel outputs); the gradients below are fused the same way
            assert cute_qk
            from xllm.modules.fused_ops.cute.qk_prep import qk_prep_bwd
            sq, sk, aq, aqk = sq_s, sk_s, aq_s, aqk_s
            sk_ = sk_rstd = sq_rstd = aqk_fp32 = aq_fp32 = None
        elif cute_qk:
            # rmsnorm, rope and softmax from one read each of xk_ and xq_; the gradients below are fused the same way
            from xllm.modules.fused_ops.cute.qk_prep import qk_prep_fwd, qk_prep_bwd
            if sk_f is not None:
                sk, aqk = sk_f, aqk_f
            else:
                sk, _, aqk, _ = qk_prep_fwd(xk_, local_kv_heads, None, fc, rope_head_dim, rmsnorm_eps)
            if sq_f is not None:
                sq, aq = sq_f, aq_f
            else:
                sq, _, aq, _ = qk_prep_fwd(xq_, local_heads, qnorm_w, fc, rope_head_dim, rmsnorm_eps)
            sk_ = sk_rstd = sq_rstd = aqk_fp32 = aq_fp32 = None
        else:
            sk_, sk_rstd = rmsnorm_fwd(xk_, None, local_kv_heads, rmsnorm_eps)
            sq, sq_rstd = rmsnorm_fwd(xq_, qnorm_w, local_heads, rmsnorm_eps)
            # B x L x H x S
            sk = rearrange(sk_, 'b l (k s) -> b l k s', k=local_kv_heads)
            sq = rearrange(sq, 'b l (h s) -> b l h s', h=local_heads)
            # apply rotary embeddings
            sq, sk = apply_ropes(sq, sk, freqs_cis, head_dim, rope_head_dim, False)
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
                attention_dropout, attn_stability_control, sca_backend, attn_w_rng_state
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
        use_cute_awm = awm_backend == 'cute' and awm_cute_bwd_supported(
            aq, av, attention_chunk_size, awm_orthogonal_update, memory, prev_akk
        )
        if use_cute_awm:
            # CuTe DSL kernels: the forward's phases are recomputed here, the gradients after the output gradient
            from xllm.modules.fused_ops.cute.adaptive_working_memory_bwd import awm_bwd_prepare, awm_bwd_finish
            awm_cute_state, awk_out_rec = awm_bwd_prepare(
                aq, aqk, akk, av, attention_chunk_size, segment_idx, awk_out is None,
                tuple(awm_workspace) if awm_workspace else None
            )
        else:
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
        if rmx_s is not None:
            rmx = rmx_s
        elif fp8:
            # in fp8 like the forward (a bf16 recompute: one 8192 x 8192 x 4096 GEMM per 7.1B layer, and a gate input
            # that differs from the forward's)
            rmx = fp8['w_qkvr'].forward(3, mx_row, br).view(bsz, seq_len, -1)
        else:
            rmx = F.linear(mx, wr, br)
        cute_gate = not ctx.health and _cute_gate(sca_out, local_heads, attn_gate_func, hidden_dropout)
        if not cute_gate:
            r = attn_gate_fn(rmx)
            attn_norm, attn_rstd = rmsnorm_fwd(sca_out + awk_out, None, local_heads, rmsnorm_eps)
            attn = torch.mul(attn_norm, r, out=awk_out)
            attn, attn_noise = memory_efficient_dropout_fwd(attn, hidden_dropout, True, attn_out_rng_state)

        if handle_xh is not None:
            handle_xh.wait()
            xh_grad = reshape_gathered_tensor_along_specific_dim(xh_grad, gather_dim=2)

        # B x L x E
        if fp8:
            xh_row, xh_col = mx_quantize(xh_grad.reshape(-1, xh_grad.shape[-1]), kind="grad")
            attn_grad = mx_mm(xh_row, fp8['w_o'].col).view(*xh_grad.shape[:-1], -1)
        else:
            attn_grad = xh_grad.matmul(wo)
        if cute_gate:
            # the recompute of the gate, its backward and the memory-efficient rmsnorm backward in one pass
            from xllm.modules.fused_ops.cute.attn_gate import attn_gate_bwd
            sca_out_grad, rmx_grad, attn, sca_delta = attn_gate_bwd(attn_grad, sca_out, awk_out, rmx, local_heads, attn_gate_func, rmsnorm_eps,
                                                                   want_attn=not fp8, want_delta=(sca_backend in ('cute', 'swift', 'xattn', 'flash') and prev_v is None))
            _nan_probe("gate", attn_grad=attn_grad, sca_out=sca_out, awk_out=awk_out, sca_out_grad=sca_out_grad, rmx_grad=rmx_grad)
        else:
            attn_grad = memory_efficient_dropout_bwd(attn_grad, hidden_dropout, attn_out_rng_state, attn_noise)
            attn_norm_grad = torch.mul(attn_grad, r, out=r)
            sca_out_grad = mem_effn_rmsnorm_bwd(attn_norm_grad, attn_norm, attn_rstd, local_heads)
            sca_delta = None
        # B x L x E -> B x L x H x E
        sca_out_grad = rearrange(sca_out_grad, 'b l (h v) -> b l h v', h=local_heads)
        # B x L x H x E -> B x H x L x E
        awk_out_grad = sca_out_grad.transpose(1, 2)

        if send_to_next:
            conv_state_q_grad, handle_conv_q = recv_prev_conv_state_grad_from_next(x, width, local_heads * head_dim)
        else:
            conv_state_q_grad = None
            handle_conv_q = None

        dv_accumulated = False
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
                prev_sk, prev_sv, bos_mask, segment_idx, attn_stability_control, deterministic, sca_backend, delta=sca_delta,
                fp8_dp=ctx.fp8_attn_dp
            )
            assert prev_aqk_grad is not None and prev_akk_grad is not None and prev_av_grad is not None
            assert prev_sk_grad is not None and prev_sv_grad is not None
            prev_aqk_grad = prev_aqk_grad.transpose(1, 2)
            prev_akk_grad = prev_akk_grad.transpose(1, 2)
            _, handle6 = send_prev_key_or_value_grad_to_prev(torch.cat([prev_sk_grad, prev_aqk_grad, prev_akk_grad], dim=3))
            prev_v_grad = prev_sv_grad + prev_av_grad.transpose(1, 2)
            _, handle5 = send_prev_key_or_value_grad_to_prev(prev_v_grad)
        else:
            _nan_probe("sca_in", sq=sq, sk=sk, xv=xv_, y=sca_for_save, aux=sca_aux)
            sq_grad, sk_grad, xv_grad, prev_sk_grad, prev_sv_grad = sliding_chunk_attention_bwd(
                sca_out_grad, sq, sk, xv_, sca_for_save, sca_aux, attention_chunk_size, sca_scale,
                prev_sk, prev_sv, bos_mask, segment_idx, attn_stability_control, deterministic, sca_backend, delta=sca_delta,
                fp8_dp=ctx.fp8_attn_dp
            )
            _nan_probe("sca", sq_grad=sq_grad, sk_grad=sk_grad, xv_grad=xv_grad)

            if handle_memory is not None:
                handle_memory.wait()
                handle_lnt.wait()

            if use_cute_awm:
                # the AWM value gradient is added into the attention's xv_grad by the kernel's store
                dv_accumulated = xv_grad.dtype == torch.bfloat16 and xv_grad.dim() == 4 and xv_grad.is_contiguous()
                (
                    aq_grad, aqk_grad, akk_grad, av_grad, memory_grad, log_norm_term_grad
                ) = awm_bwd_finish(awm_cute_state, awk_out_grad, memory_grad, log_norm_term_grad,
                                   dv_accum=xv_grad if dv_accumulated else None)
                prev_aqk_grad = prev_akk_grad = prev_av_grad = None
                _nan_probe("awm", aq_grad=aq_grad, aqk_grad=aqk_grad, akk_grad=akk_grad, av_grad=av_grad, xv_grad=xv_grad,
                           memory_grad=memory_grad)
            else:
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
        if cute_gate:
            rmx_grad = rearrange(rmx_grad, 'b l d -> (b l) d')
        else:
            r_grad = torch.mul(attn_grad, attn_norm, out=attn_grad)
            # B*L x D
            rmx_grad = rearrange(attn_gate_fn_bwd(r_grad, rmx), 'b l d -> (b l) d')
        mx_grad = None if fp8 else torch.mm(rmx_grad, wr)

        # B x H x L x S -> B x L x H x S
        aq_grad = aq_grad.transpose(1, 2)
        if fused_q:
            # rope backward, q norm backward, softmax backward, their sum and the conv backward in one pass
            xq_grad, q_conv_w_grad, qnorm_w_grad = conv_qk_prep_bwd(
                sq_grad, aq_grad, None, xq, q_conv_w, causal_conv_weight_normalization, bos_mask_curr, local_heads, qnorm_w, fc,
                rope_head_dim, rmsnorm_eps)
            _nan_probe("convq", xq=xq, xq_grad=xq_grad, q_conv_w_grad=q_conv_w_grad, qnorm_w_grad=qnorm_w_grad)
        elif cute_qk:
            # rope backward, q norm backward, softmax backward and their sum in one pass
            xq_grad, qnorm_w_grad = qk_prep_bwd(sq_grad, aq_grad, None, xq_, local_heads, qnorm_w, fc, rope_head_dim, rmsnorm_eps)
        else:
            # apply rotary embeddings
            sq_grad = apply_rope(sq_grad, freqs_cis, head_dim, rope_head_dim, True)
            # q norm grad
            xq_grad, qnorm_w_grad = rmsnorm_bwd(rearrange(sq_grad, 'b l h s -> b l (h s)'), xq_, sq_rstd, qnorm_w, local_heads)
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

        if fused_k:
            xk_grad, k_conv_w_grad, _ = conv_qk_prep_bwd(
                sk_grad, aqk_grad, akk_grad, xk, k_conv_w, causal_conv_weight_normalization, bos_mask_curr, local_kv_heads, None,
                fc, rope_head_dim, rmsnorm_eps)
            _nan_probe("convk", xk=xk, xk_grad=xk_grad, k_conv_w_grad=k_conv_w_grad)
        elif cute_qk:
            xk_grad, _ = qk_prep_bwd(sk_grad, aqk_grad, akk_grad, xk_, local_kv_heads, None, fc, rope_head_dim, rmsnorm_eps)
        else:
            # apply rotary embeddings
            sk_grad = apply_rope(sk_grad, freqs_cis, head_dim, rope_head_dim, True)
            # k norm grad
            xk_grad = mem_effn_rmsnorm_bwd(rearrange(sk_grad, 'b l k s -> b l (k s)'), sk_, sk_rstd, local_kv_heads)
            aqk_grad = torch.ops.aten._softmax_backward_data(
                aqk_grad.float(), aqk_fp32, -1, torch.float32
            ).to(aqk)
            ak_grad = rearrange(torch.add(akk_grad, aqk_grad, out=aqk_grad), 'b l k s -> b l (k s)')
            # grads of xk & xv
            xk_grad = torch.add(xk_grad, ak_grad, out=xk_grad)
        if not dv_accumulated:
            xv_grad = torch.add(xv_grad, av_grad.transpose(1, 2), out=xv_grad)
        if handle_prev_v is not None:
            handle_prev_v.wait()
            xv_grad[:, (seq_len - attention_chunk_size):] += v_grad_from_next
        xv_grad = rearrange(xv_grad, 'b l k v -> b l (k v)')

        # xq grad from causal conv
        if handle_conv_q is not None:
            handle_conv_q.wait()
        if not fused_q:
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
        if not fused_k:
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
        if fp8:
            # one GEMM: [xq_grad | xk_grad | xv_grad | rmx_grad] [wq; wk; wv; wr]
            mx_grad, dqkvr_col = fp8['w_qkvr'].dgrad([xq_grad, xk_grad, xv_grad, rmx_grad])
        else:
            mx_grad = torch.addmm(mx_grad, xq_grad, wq, out=mx_grad)
            mx_grad = torch.addmm(mx_grad, xk_grad, wk, out=mx_grad)
            mx_grad = torch.addmm(mx_grad, xv_grad, wv, out=mx_grad)

        # reduce mx grad
        # B x L x D
        mx_grad = rearrange(mx_grad, '(b l) d -> b l d', b=bsz)
        mx_grad, handle_mx = reduce_scatter(mx_grad, parallel_region='model', async_op=True)

        # grads for wo
        if fp8:
            wo_grad, = fp8['w_o'].wgrad(xh_col, fp8['attn_col'])
        else:
            attn_flat = attn.flatten(end_dim=-2)
            wo_grad = xh_grad.flatten(end_dim=-2).t().matmul(attn_flat)

        # grads for wk, wv & bv
        mx_flat = None if mx is None else mx.flatten(end_dim=-2)
        if fp8:
            # one GEMM: [xq_grad | xk_grad | xv_grad | rmx_grad]^T mx -> the four weight gradients (row slices)
            wq_grad, wk_grad, wv_grad, wr_grad = fp8['w_qkvr'].wgrad(dqkvr_col, mx_col)
        else:
            wk_grad = torch.mm(xk_grad.t(), mx_flat)
            wv_grad = torch.mm(xv_grad.t(), mx_flat)
        # column sum as a cuBLAS matrix-vector product: ~2x the bandwidth of sum(dim=0) for wide rows
        bv_grad = None if bv is None else torch.mv(xv_grad.t(), xv_grad.new_ones(xv_grad.shape[0]))

        if handle_mx is not None:
            handle_mx.wait()

        if handle_tsn is not None:
            handle_tsn.wait()
            tsn_mean_grad, tsn_var_grad = torch.unbind(mean_var_grad, dim=2)

        # the residual connection's gradient is accumulated by the norm backward (residual_grad is dead afterwards)
        x_grad, prev_mean_grad, prev_var_grad, timenorm_w_grad, timenorm_b_grad = timenorm_bwd(
            mx_grad, tsn_mean_grad, tsn_var_grad, x, bos_mask_curr, prev_count, prev_mean, cummean, cumrstd,
            timenorm_w, timenorm_b, timenorm_local_groups, timenorm_beta1, timenorm_beta2, timenorm_eps, False, timenorm_backend,
            residual=residual_grad,
            step_count=tsn_step_count,
        )

        if recv_from_prev:
            _, handle1 = send_mean_var_grad_to_prev(prev_mean_grad, prev_var_grad)
        else:
            handle1 = None

        if not fp8:
            wr_grad = torch.mm(rmx_grad.t(), mx_flat)
        br_grad = None if br is None else rmx_grad.sum(dim=0)
        if not fp8:
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
            None,  # layer_id
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
            None,  # awm_backend
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
            None,  # attn_stability_control
            None,  # deterministic
            None,  # recompute_q
            None,  # recompute_kv
            None,  # recompute_sca
            None,  # recompute_awk
            None,  # recompute_fc1_out
            None,  # recompute_fc3_out
            None,  # recompute_conv_qk
            None,  # fp8_gemm
            None,  # fp8_attn_dp
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

        collector = get_active_health_collector()
        output = fused_linear_cross_entropy_fwd(
            out_tsn, wo, y, mask, compute_grad=(not recompute_logits),
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
        loss, grad_tsn, grad_wo = output.loss, output.grad_input, output.grad_weight

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
        out_tsn, tsn_count, tsn_mean, tsn_var, cummean, cumrstd, tsn_step_count = timenorm_fwd(
            x, bos_mask, prev_count, prev_mean, prev_var, timenorm_w, timenorm_b,
            timenorm_local_groups, timenorm_beta1, timenorm_beta2, eps, timenorm_backend, want_step_count=True
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
            timenorm_w, timenorm_b, timenorm_local_groups, timenorm_beta1, timenorm_beta2, eps, False, timenorm_backend,
            step_count=tsn_step_count,
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
