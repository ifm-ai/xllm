from typing import Optional, Tuple, Any
import math
import torch

from xllm.distributed.utils import reshape_gathered_tensor_along_specific_dim
from xllm.modules.fused_ops import (
    memory_efficient_dropout_fwd,
    memory_efficient_dropout_bwd,
)
from xllm.models.fused_blocks.utils import (
    layer_or_rmsnorm_fwd,
    layer_or_rmsnorm_bwd,
)
from xllm.models.fused_blocks.mha import (
    multihead_attention_fwd,
    multihead_attention_bwd,
    multihead_attention_recompute
)
from xllm.models.fused_blocks.moe import (
    moe_fwd,
    moe_bwd,
)
from xllm.models.fused_blocks.distributed import (
    scatter,
    all_gather,
)


class TransformerMoEBlockFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,  # (bsz, slen, d/MP)
        freqs_cis: Optional[torch.Tensor],  # (slen, d)
        segments: Optional[Tuple[torch.Tensor, torch.Tensor, int, int, int]],
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
        attn_res_w: Optional[torch.Tensor],  # (d/MP, c)
        moe_norm_w: torch.Tensor,  # (d)
        moe_norm_b: Optional[torch.Tensor],  # (d)
        norm_local_groups: int,
        fc1_w: Optional[torch.Tensor],
        fc2_w: Optional[torch.Tensor],
        fc3_w: Optional[torch.Tensor],
        router_w: torch.Tensor,
        n_experts: int,
        n_local_experts: int,
        expert_start_idx: int,
        expert_end_idx: int,
        topk: int,
        moe_permutation_backend: str,
        routing_score_func: str,
        router_bias: Optional[torch.Tensor],
        router_bias_update_rate: Optional[float],
        routing_scaling_factor: Optional[float],
        router_load_balancing_type: Optional[str],
        expert_w1: torch.Tensor,
        expert_w2: torch.Tensor,
        expert_w3: torch.Tensor,
        expert_backend: str,
        moe_res_w: Optional[torch.Tensor],  # (d/MP, c)
        dropout: float,
        attention_dropout: float,
        hidden_dropout: float,
        layernorm_eps: float,
        rmsnorm_eps: float,
        apply_rmsnorm: bool,
        gather_before_norm: bool,
        residual_func: str,
        residual_heads: Optional[int],
        fp32_attn_output: bool,
        deterministic: bool,
        recompute_q: bool,
        recompute_kv: bool,
        recompute_attention: bool,
        recompute_fc1_out: bool,
        recompute_fc3_out: bool,
        recompute_router: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:

        bsz, seq_len, _ = x.size()
        residual = x

        x_ = x
        handle_moe_norm_w = None
        handle_moe_norm_b = None
        if gather_before_norm:
            x_, _ = all_gather(x, parallel_region='model', async_op=False)
            if attn_norm_w is not None:
                attn_norm_w, _ = all_gather(attn_norm_w, parallel_region='model', async_op=False)
            if attn_norm_b is not None:
                attn_norm_b, _ = all_gather(attn_norm_b, parallel_region='model', async_op=False)
            if moe_norm_w is not None:
                moe_norm_w, handle_moe_norm_w = all_gather(moe_norm_w, parallel_region='model', async_op=True)
            if moe_norm_b is not None:
                moe_norm_b, handle_moe_norm_b = all_gather(moe_norm_b, parallel_region='model', async_op=True)

        mx, _, _ = layer_or_rmsnorm_fwd(x_, attn_norm_w, attn_norm_b, norm_local_groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm)
        if not gather_before_norm:
            mx, _ = all_gather(mx, parallel_region='model', async_op=False)

        # MHA forward
        xqkv, cu_seqlens, attn_out, xh, rng_states = multihead_attention_fwd(
            mx, freqs_cis, segments, wq, wk, wv, wr, wo, q_norm_w,
            head_dim, rope_head_dim, local_heads, local_kv_heads, rmsnorm_eps,
            attn_gate_func, dropout, attention_dropout, hidden_dropout,
        )
        xq, xk, xv, xk_rstd = xqkv
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, total_seqlen_k, end_seq = cu_seqlens
        attn_out, attn_lse = attn_out
        flash_rng_state, attn_rng_state, attn_out_rng_state, xh_rng_state = rng_states

        # residual
        xo = torch.add(xh, residual, out=xh)
        residual = xo

        xo_ = xo
        if gather_before_norm:
            xo_, _ = all_gather(xo, parallel_region='model', async_op=False)
            if handle_moe_norm_w is not None:
                handle_moe_norm_w.wait()
                moe_norm_w = reshape_gathered_tensor_along_specific_dim(moe_norm_w, gather_dim=0)
            if handle_moe_norm_b is not None:
                handle_moe_norm_b.wait()
                moe_norm_b = reshape_gathered_tensor_along_specific_dim(moe_norm_b, gather_dim=0)

        xf, _, _ = layer_or_rmsnorm_fwd(xo_, moe_norm_w, moe_norm_b, norm_local_groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm)
        if gather_before_norm:
            xf = scatter(xf, parallel_region='model')

        # MoE forward
        out, aux_loss, routing_out, shared_hiddens, routing_state, rng_states = moe_fwd(
            xf, router_w,  router_bias, routing_score_func, routing_scaling_factor, router_load_balancing_type,
            fc1_w, fc2_w, fc3_w, n_experts, n_local_experts, expert_start_idx, expert_end_idx, topk,
            expert_w1, expert_w2, expert_w3, expert_backend, moe_permutation_backend,
            hidden_dropout, recompute_fc1_out, recompute_fc3_out
        )
        original_scores, routing_indices, org_topk_scores, tokens_per_expert = routing_out
        h1, h3 = shared_hiddens
        group_sizes, input_splits, output_splits, curr_bsz = routing_state
        expert_hidden_rng_state, shared_hidden_rng_state = rng_states

        out, out_rng_state = memory_efficient_dropout_fwd(out, dropout, True)
        # residual
        out = torch.add(out, residual, out=out)

        ctx.save_for_backward(
            x,  # (bsz, slen, d/MP)
            freqs_cis,  # (slen, s)
            cu_seqlens_q,
            cu_seqlens_k,
            attn_norm_w,  # (d/MP)
            attn_norm_b,  # (d/MP)
            wq,  # (d/MP, d)
            wk,  # (d/MP, d)
            wv,  # (d/MP, d)
            wr,  # (d/MP, d)
            wg,  # (h/MP, d)
            wo,  # (d, v/MP)
            q_norm_w,  # (d/MP)
            attn_res_w,  # (d/MP, c)
            moe_norm_w,  # (d)
            moe_norm_b,  # (d)
            fc1_w,
            fc2_w,
            fc3_w,
            router_w,
            router_bias,
            expert_w1,
            expert_w2,
            expert_w3,
            moe_res_w,  # (d/MP, c)
            xq if not recompute_q else None,  # (bsz, slen, n_heads, head_dim)
            xk if not recompute_kv else None,  # (bsz, slen, n_kv_heads, head_dim)
            xk_rstd if not recompute_kv else None,  # (bsz, slen, kv_heads)
            xv if not recompute_kv else None,  # (bsz, slen, n_kv_heads, head_dim)
            attn_out if not recompute_attention else None,  # (bsz, slen, n_heads, head_dim)
            attn_lse if not recompute_attention else None,  # (bsz, h/MP, slen, slen)
            flash_rng_state,
            xo,  # (bsz, slen, d/MP)
            original_scores if not recompute_router else None,  # (bsz, slen, n_experts)
            routing_indices,  # (bsz, slen, topk)
            org_topk_scores,  # (bsz, slen, topk)
            tokens_per_expert,  # (n_experts)
            h1 if not recompute_fc1_out else None,  # (bsz, slen, v/MP)
            h3 if not recompute_fc3_out else None,  # (bsz, slen, v/MP)
        )
        ctx.local_heads = local_heads
        ctx.local_kv_heads = local_kv_heads
        ctx.head_dim = head_dim
        ctx.rope_head_dim = rope_head_dim
        ctx.norm_local_groups = norm_local_groups
        ctx.n_experts = n_experts
        ctx.n_local_experts = n_local_experts
        ctx.expert_start_idx = expert_start_idx
        ctx.expert_end_idx = expert_end_idx
        ctx.expert_backend = expert_backend
        ctx.topk = topk
        ctx.moe_permutation_backend = moe_permutation_backend
        ctx.routing_score_func = routing_score_func
        ctx.router_bias_update_rate = router_bias_update_rate
        ctx.routing_scaling_factor = routing_scaling_factor
        ctx.router_load_balancing_type = router_load_balancing_type
        ctx.group_sizes = group_sizes
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.curr_bsz = curr_bsz
        ctx.layernorm_eps = layernorm_eps
        ctx.rmsnorm_eps = rmsnorm_eps
        ctx.attn_gate_func = attn_gate_func
        ctx.dropout = dropout
        ctx.attention_dropout = attention_dropout
        ctx.hidden_dropout = hidden_dropout
        ctx.apply_rmsnorm = apply_rmsnorm
        ctx.gather_before_norm = gather_before_norm
        ctx.causal_attn_backend = causal_attn_backend
        ctx.residual_func = residual_func
        ctx.residual_heads = residual_heads
        ctx.fp32_attn_output = fp32_attn_output
        ctx.deterministic = deterministic
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.total_seqlen_k = total_seqlen_k
        ctx.end_seq = end_seq

        ctx.attn_rng_state = attn_rng_state
        ctx.attn_out_rng_state = attn_out_rng_state
        ctx.xh_rng_state = xh_rng_state
        ctx.expert_hidden_rng_state = expert_hidden_rng_state
        ctx.shared_hidden_rng_state = shared_hidden_rng_state
        ctx.out_rng_state = out_rng_state

        return out, aux_loss, None

    @staticmethod
    def backward(ctx, out_grad, aux_loss_grad, cache):
        assert cache is None

        (
            x,  # (bsz, slen, d/MP)
            freqs_cis,  # (slen, s)
            cu_seqlens_q,
            cu_seqlens_k,
            attn_norm_w,  # (d/MP)
            attn_norm_b,  # (d/MP)
            wq,  # (d/MP, d)
            wk,  # (d/MP, d)
            wv,  # (d/MP, d)
            wr,  # (d/MP, d)
            wg,  # (h/MP, d)
            wo,  # (d, v/MP)
            q_norm_w,  # (d/MP)
            attn_res_w,  # (d/MP, c)
            moe_norm_w,  # (d)
            moe_norm_b,  # (d)
            fc1_w,
            fc2_w,
            fc3_w,
            router_w,
            router_bias,
            expert_w1,
            expert_w2,
            expert_w3,
            moe_res_w,  # (d/MP, c)
            xq,  # (bsz, slen, n_heads, head_dim)
            xk,  # (bsz, slen, n_kv_heads, head_dim)
            xk_rstd,  # (bsz, slen, kv_heads)
            xv,  # (bsz, slen, n_kv_heads, head_dim)
            attn_out,  # (bsz, slen, n_heads, head_dim)
            attn_lse,  # (bsz, h/MP, slen, slen)
            flash_rng_state,
            xo,  # (bsz, slen, d/MP)
            original_scores,  # (bsz, slen, n_experts)
            routing_indices,  # (bsz, slen, topk)
            org_topk_scores,  # (bsz, slen, topk)
            tokens_per_expert,  # (n_experts)
            h1,  # (bsz, slen, v/MP)
            h3,  # (bsz, slen, v/MP)
        ) = ctx.saved_tensors

        local_heads = ctx.local_heads
        local_kv_heads = ctx.local_kv_heads
        head_dim = ctx.head_dim
        rope_head_dim = ctx.rope_head_dim
        norm_local_groups = ctx.norm_local_groups

        n_experts = ctx.n_experts
        n_local_experts = ctx.n_local_experts
        topk = ctx.topk
        expert_start_idx = ctx.expert_start_idx
        expert_end_idx = ctx.expert_end_idx
        expert_backend = ctx.expert_backend
        moe_permutation_backend = ctx.moe_permutation_backend
        routing_score_func = ctx.routing_score_func
        router_bias_update_rate = ctx.router_bias_update_rate
        routing_scaling_factor = ctx.routing_scaling_factor
        router_load_balancing_type = ctx.router_load_balancing_type
        group_sizes = ctx.group_sizes
        input_splits = ctx.input_splits
        output_splits = ctx.output_splits
        curr_bsz = ctx.curr_bsz

        dropout = ctx.dropout
        attention_dropout = ctx.attention_dropout
        hidden_dropout = ctx.hidden_dropout
        layernorm_eps = ctx.layernorm_eps
        rmsnorm_eps = ctx.rmsnorm_eps
        apply_rmsnorm = ctx.apply_rmsnorm
        gather_before_norm = ctx.gather_before_norm
        causal_attn_backend = ctx.causal_attn_backend
        attn_gate_func = ctx.attn_gate_func
        residual_func = ctx.residual_func
        residual_heads = ctx.residual_heads
        fp32_attn_output = ctx.fp32_attn_output
        deterministic = ctx.deterministic
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        total_seqlen_k = ctx.total_seqlen_k
        end_seq = ctx.end_seq

        attn_rng_state = ctx.attn_rng_state
        attn_out_rng_state = ctx.attn_out_rng_state
        xh_rng_state = ctx.xh_rng_state
        expert_hidden_rng_state = ctx.expert_hidden_rng_state
        shared_hidden_rng_state = ctx.shared_hidden_rng_state
        out_rng_state = ctx.out_rng_state

        attn_scale = 1.0 / math.sqrt(head_dim)
        bsz, seq_len, _ = xo.size()
        residual_grad = out_grad

        # recompute xf
        xo_ = xo
        x_ = x
        handle_x = None
        if gather_before_norm:
            xo_, _ = all_gather(xo, parallel_region='model', async_op=False)
            x_, handle_x = all_gather(x, parallel_region='model', async_op=True)

        xf, xo_mean, xo_invvar = layer_or_rmsnorm_fwd(
            xo_, moe_norm_w, moe_norm_b, norm_local_groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm
        )
        if gather_before_norm:
            xf = scatter(xf, parallel_region='model')

        xf_flat = xf.flatten(end_dim=-2)
        # output dropout
        out_grad = memory_efficient_dropout_bwd(out_grad, dropout, out_rng_state)

        if handle_x is not None:
            handle_x.wait()
            x_ = reshape_gathered_tensor_along_specific_dim(x_, gather_dim=2)

        # recompute mx
        mx, x_mean, x_invvar = layer_or_rmsnorm_fwd(
            x_, attn_norm_w, attn_norm_b, norm_local_groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm
        )
        if not gather_before_norm:
            mx, handle_mx = all_gather(mx, parallel_region='model', async_op=True)
        else:
            handle_mx = None

        # MoE backward
        xf_grad, ew1_grad, ew2_grad, ew3_grad, fc1_w_grad, fc2_w_grad, fc3_w_grad, router_w_grad = moe_bwd(
            out_grad, aux_loss_grad, xf, xf_flat, original_scores, routing_indices, org_topk_scores, tokens_per_expert,
            router_w, router_bias, router_bias_update_rate, routing_score_func, routing_scaling_factor, router_load_balancing_type,
            h1, h3, fc1_w, fc2_w, fc3_w, n_experts, n_local_experts, topk, expert_w1, expert_w2, expert_w3,
            expert_backend, moe_permutation_backend, hidden_dropout, group_sizes, input_splits, output_splits, curr_bsz,
            expert_hidden_rng_state, shared_hidden_rng_state, gather_before_norm,
        )

        xo_grad, moe_norm_w_grad, moe_norm_b_grad = layer_or_rmsnorm_bwd(
            xf_grad, xo_, xo_mean, xo_invvar, moe_norm_w, moe_norm_b, norm_local_groups, apply_rmsnorm
        )

        if gather_before_norm:
            xo_grad = scatter(xo_grad, parallel_region='model').contiguous()
            if moe_norm_w_grad is not None:
                moe_norm_w_grad = scatter(moe_norm_w_grad, parallel_region='model')
            if moe_norm_b_grad is not None:
                moe_norm_b_grad = scatter(moe_norm_b_grad, parallel_region='model')

        residual_grad = torch.add(xo_grad, residual_grad, out=xo_grad)
        xh_grad = residual_grad
        xh_grad = memory_efficient_dropout_bwd(xh_grad, dropout, xh_rng_state)
        xh_grad, handle_xh = all_gather(xh_grad, parallel_region='model', async_op=True)

        if handle_mx is not None:
            handle_mx.wait()
            mx = reshape_gathered_tensor_along_specific_dim(mx, gather_dim=2)

        attn_outs, q_outs, k_outs, v_outs, handle_kv, attn_gate_outs = multihead_attention_recompute(
            attn_out, attn_lse, xq, xk, xv, xk_rstd, total_seqlen_k, end_seq,
            mx, freqs_cis, wq, wk, wv, wr, q_norm_w,
            head_dim, rope_head_dim, local_heads, local_kv_heads, rmsnorm_eps,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            attn_scale, attn_gate_func, attention_dropout, attn_rng_state,
        )
        attn_out, attn_lse = attn_outs
        xq, xq_, xq_rstd = q_outs
        xk, xk_, xk_rstd = k_outs
        xv, xv_ = v_outs
        handle_xk, handle_xv = handle_kv
        rmx, r = attn_gate_outs

        # B x L x H x D/H -> B x L x D
        attn_r = attn_out.view(bsz, seq_len, -1)
        attn = attn_r * r if r is not None else attn_r
        attn, attn_noise = memory_efficient_dropout_fwd(attn, hidden_dropout, True, attn_out_rng_state)

        if handle_xh is not None:
            handle_xh.wait()
            xh_grad = reshape_gathered_tensor_along_specific_dim(xh_grad, gather_dim=2)
        # B x L x D
        attn_grad = xh_grad.matmul(wo)
        attn_grad = memory_efficient_dropout_bwd(attn_grad, hidden_dropout, attn_out_rng_state, attn_noise)
        attn_r_grad = torch.mul(attn_grad, r, out=r) if r is not None else attn_grad
        r_grad = torch.mul(attn_grad, attn_r, out=attn_grad) if r is not None else None

        # grads for wh2
        attn_flat = attn.flatten(end_dim=-2)
        wo_grad = xh_grad.flatten(end_dim=-2).t().matmul(attn_flat)

        if handle_xv is not None:
            assert handle_xk is not None
            handle_xv.wait()
            xv_ = reshape_gathered_tensor_along_specific_dim(xv_, gather_dim=1)[:, :end_seq]
            handle_xk.wait()
            xk_ = reshape_gathered_tensor_along_specific_dim(xk_, gather_dim=1)[:, :end_seq]
            if total_seqlen_k is not None:
                xk_ = xk_[:, -total_seqlen_k:]
                xv_ = xv_[:, -total_seqlen_k:]

        # MHA backward
        # B x L x D -> B x L x H x D/H
        attn_r_grad = attn_r_grad.view(bsz, seq_len, local_heads, head_dim)
        x_grad, wq_grad, wk_grad, wv_grad, wr_grad, q_norm_w_grad, attn_norm_w_grad, attn_norm_b_grad = multihead_attention_bwd(
            attn_r_grad, r_grad, attn_out, attn_lse, xq_, xk_, xv_, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            total_seqlen_k, end_seq, freqs_cis, local_kv_heads, rope_head_dim,
            xq, xk, xq_rstd, xk_rstd, rmx, mx, q_norm_w, wq, wk, wv, wr,
            attn_scale, attn_gate_func, attention_dropout, flash_rng_state, deterministic,
            x_, x_mean, x_invvar, attn_norm_w, attn_norm_b, norm_local_groups, apply_rmsnorm, gather_before_norm
        )

        # residual connection
        x_grad = torch.add(x_grad, residual_grad, out=x_grad)

        return (
            x_grad,
            None,  # freqs_cis
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
            None,  # attn_res_w
            moe_norm_w_grad,
            moe_norm_b_grad,
            None,  # norm_local_groups
            fc1_w_grad,
            fc2_w_grad,
            fc3_w_grad,
            router_w_grad,
            None,  # n_experts
            None,  # n_local_experts
            None,  # expert_start_idx
            None,  # expert_end_idx
            None,  # topk
            None,  # moe_permutation_backend
            None,  # routing_score_func
            None,  # router_bias
            None,  # router_bias_update_rate
            None,  # routing_scaling_factor
            None,  # router_load_balancing_type
            ew1_grad,
            ew2_grad,
            ew3_grad,
            None,  # moe expert backend
            None,  # moe_res_w
            None,  # dropout
            None,  # attention_dropout
            None,  # hidden_dropout
            None,  # layernorm_eps
            None,  # rmsnorm_eps
            None,  # apply_rmsnorm
            None,  # gather_before_norm
            None,  # residual_func
            None,  # residual_heads
            None,  # fp32_attn_output
            None,  # deterministic
            None,  # recompute_q
            None,  # recompute_kv
            None,  # recompute_attention
            None,  # recompute_fc1_out
            None,  # recompute_fc3_out
            None,  # recompute_router
        )
