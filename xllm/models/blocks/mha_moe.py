from typing import Optional, Tuple, Any
from torch import Tensor
from torch import nn

from xllm.modules import (
    MultiheadAttention,
    NormalizedMoE,
)
from xllm.modules.residual import num_params_in_residual
from xllm.config import ModelConf
from xllm.models.fused_blocks import TransformerMoEBlockFunction


class TransformerMoEBlock(nn.Module):

    def __init__(self, cfg: ModelConf, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.residual_func = cfg.residual_func
        self.residual_heads = cfg.residual_heads
        self.fused_block = cfg.fused_block
        self.recompute_q = cfg.recompute_q
        self.recompute_kv = cfg.recompute_v
        self.recompute_attention = cfg.recompute_attention
        self.recompute_fc1_out = cfg.recompute_fc1_out
        self.recompute_fc3_out = cfg.recompute_fc3_out
        self.recompute_router = cfg.recompute_router
        self.dropout = cfg.dropout
        self.apply_nope = cfg.apply_nope_in_causal_attn
        self.apply_causal_conv = cfg.apply_conv_in_causal_attn
        self.apply_rmsnorm = cfg.apply_rmsnorm
        self.rmsnorm_eps = cfg.rmsnorm_eps

        if self.fused_block:
            assert cfg.causal_attn_backend in ['flash', 'xattn'], f"flash or xattn attention backend is required for fused block."

        self.attention = MultiheadAttention(
            layer_id=layer_id,
            mdim=cfg.model_dim,
            n_heads=cfg.num_heads,
            n_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            rope_head_dim=cfg.rope_head_dim,
            apply_nope=cfg.apply_nope_in_causal_attn,
            attn_act_func=cfg.attn_act_func,
            qknorm=cfg.qknorm,
            apply_causal_conv=cfg.apply_conv_in_causal_attn,
            causal_conv_width=cfg.causal_conv_width,
            causal_conv_backend=cfg.causal_conv_backend,
            causal_conv_weight_normalization=cfg.causal_conv_weight_normalization,
            norm_num_groups=cfg.layernorm_num_groups,
            norm_affine=cfg.norm_affine,
            layernorm_eps=cfg.layernorm_eps,
            rmsnorm_eps=cfg.rmsnorm_eps,
            memory_efficient_norm=cfg.memory_efficient_norm,
            apply_rmsnorm=cfg.apply_rmsnorm,
            apply_attn_gate=cfg.apply_attn_gate,
            attn_gate_func=cfg.attn_gate_func,
            causal_attn_backend=cfg.causal_attn_backend,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            hidden_dropout=cfg.hidden_dropout,
            residual_func=cfg.residual_func,
            residual_heads=cfg.residual_heads,
            init_mode=cfg.init_mode,
            init_std=cfg.init_std,
            stream_eps_scale=cfg.stream_eps_scale,
        )

        self.moe = NormalizedMoE(
            layer_id=layer_id,
            model_dim=cfg.model_dim,
            expert_inter_dim=cfg.expert_inter_dim,
            num_experts=cfg.num_experts,
            num_activated_experts=cfg.num_activated_experts,
            num_shared_experts=cfg.num_shared_experts,
            expert_backend=cfg.moe_expert_backend,
            permutation_backend=cfg.moe_permutation_backend,
            router_score_func=cfg.moe_router_score_func,
            router_bias=cfg.moe_router_bias,
            router_bias_update_rate=cfg.moe_router_bias_update_rate,
            router_scaling_factor=cfg.moe_router_scaling_factor,
            dropout=cfg.dropout,
            hidden_dropout=cfg.hidden_dropout,
            norm_num_groups=cfg.layernorm_num_groups,
            norm_affine=cfg.norm_affine,
            layernorm_eps=cfg.layernorm_eps,
            rmsnorm_eps=cfg.rmsnorm_eps,
            memory_efficient_norm=cfg.memory_efficient_norm,
            apply_rmsnorm=cfg.apply_rmsnorm,
            residual_func=cfg.residual_func,
            residual_heads=cfg.residual_heads,
            init_mode=cfg.init_mode,
            init_std=cfg.init_std,
            stream_eps_scale=cfg.stream_eps_scale,
        )

    def forward(
        self,
        x: Tensor,
        freqs_cis: Optional[Tensor],
        bos_mask: Optional[Tensor] = None,
        segments: Optional[Any] = None,
        moe_router_load_balancing_type: Optional[str] = None,
        attn_stability_control: int = 0,
        deterministic: bool = True,
        cache: Optional[Tuple[Tuple[Tensor, Tensor, int],
                              Tuple[Tensor, Tensor, Tensor]]] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Any]]:
        if self.fused_block and self.training:
            fn = TransformerMoEBlockFunction
            return fn.apply(
                self.layer_id,
                x,
                None if self.apply_nope else freqs_cis,
                bos_mask,
                segments,
                self.attention.norm.weight,
                self.attention.norm.bias,
                self.attention.wq.weight,
                self.attention.wk.weight,
                self.attention.wv.weight,
                self.attention.wr.weight if self.attention.wr is not None else None,
                self.attention.wo.weight,
                self.attention.query_norm.weight if self.attention.qknorm else None,
                self.attention.key_norm.weight if self.attention.qknorm else None,
                self.attention.local_heads,
                self.attention.local_kv_heads,
                self.attention.head_dim,
                self.attention.rope_head_dim,
                self.apply_nope,
                self.attention.attn_act_func,
                self.attention.apply_attn_gate,
                self.attention.attn_gate_fn,
                self.attention.causal_attn_backend,
                self.moe.norm.weight,
                self.moe.norm.bias,
                self.moe.norm.groups_per_partition,
                self.moe.fc1.weight if self.moe.fc1 is not None else None,
                self.moe.fc2.weight if self.moe.fc2 is not None else None,
                self.moe.fc3.weight if self.moe.fc3 is not None else None,
                self.moe.router.weight,
                self.moe.n_experts,
                self.moe.n_local_experts,
                self.moe.expert_start_idx,
                self.moe.expert_end_idx,
                self.moe.topk,
                self.moe.permutation_backend,
                self.moe.router.score_func,
                self.moe.router.bias,
                self.moe.router.bias_update_rate,
                self.moe.router.scaling_factor,
                moe_router_load_balancing_type,
                self.moe.experts.weight1,
                self.moe.experts.weight2,
                self.moe.experts.weight3,
                self.moe.experts.backend,
                self.dropout,
                self.attention.attention_dropout,
                self.attention.hidden_dropout,
                self.moe.norm_eps,  # both pre-norms' eps
                self.rmsnorm_eps,
                self.apply_rmsnorm,
                self.attention.norm.gather_input,
                self.residual_func,
                self.residual_heads,
                attn_stability_control,
                deterministic,
                self.recompute_q,
                self.recompute_kv,
                self.recompute_attention,
                self.recompute_fc1_out,
                self.recompute_fc3_out,
                self.recompute_router,
            )

        y, cache = self.attention(
            x, freqs_cis, bos_mask, segments, attn_stability_control, deterministic, cache
        )
        # MoE
        out, aux_loss = self.moe(y, moe_router_load_balancing_type)

        return out, aux_loss, cache

    def num_parameters(self):
        model_dim = self.attention.mdim
        attn_act_func = self.attention.attn_act_func
        nk = {"softmax": 1, "softdelta": 2}[attn_act_func]
        head_dim = self.attention.head_dim
        n_heads = self.attention.n_heads
        n_kv_heads = self.attention.n_kv_heads
        apply_attn_gate = self.attention.apply_attn_gate
        qknorm = self.attention.qknorm
        conv_width = self.attention.q_conv.kernel_size if self.apply_causal_conv else 0
        expert_inter_dim = self.moe.inter_dim
        n_shared_experts = self.moe.n_shared_experts
        n_experts = self.moe.n_experts
        n_activated_experts = self.moe.topk

        # MHA
        attn_params = model_dim * (n_heads * 2 + n_kv_heads * (nk + 1)) * head_dim
        if attn_act_func == 'softdelta':
            attn_params += model_dim * n_heads
        if apply_attn_gate:
            attn_params += model_dim * n_heads * head_dim
        attn_params += num_params_in_residual(self.residual_func, model_dim, self.residual_heads, n_heads * head_dim)
        # Norm
        norm_params = model_dim * (2 if self.apply_rmsnorm else 4)
        if qknorm:
            norm_params += (n_heads + n_kv_heads * nk) * head_dim
        # conv params
        conv_params = conv_width * (n_heads * head_dim + 2 * n_kv_heads * head_dim)

        activated_params = attn_params + norm_params + conv_params
        total_params = activated_params
        # MoE
        shared_expert_params = model_dim * expert_inter_dim * n_shared_experts * 3
        routed_expert_params = model_dim * expert_inter_dim * n_experts * 3
        activated_expert_params = model_dim * expert_inter_dim * n_activated_experts * 3
        router_params = model_dim * n_experts
        residual_params = num_params_in_residual(self.residual_func, model_dim, self.residual_heads, model_dim)
        activated_params += shared_expert_params + activated_expert_params + router_params + residual_params
        total_params += shared_expert_params + routed_expert_params + router_params

        return total_params, activated_params

    def tflops_per_token(self, seq_len: int):
        model_dim = self.attention.mdim
        attn_act_func = self.attention.attn_act_func
        head_dim = self.attention.head_dim
        n_heads = self.attention.n_heads
        n_kv_heads = self.attention.n_kv_heads
        qknorm = self.attention.qknorm
        conv_width = self.attention.q_conv.kernel_size if self.apply_causal_conv else 0

        qdim = n_heads * head_dim
        kdim = vdim = n_kv_heads * head_dim

        _, tflops_per_token = self.num_parameters()
        if attn_act_func == 'softmax':
            tflops_per_token += n_heads * head_dim * seq_len
        elif attn_act_func == 'softdelta':
            tflops_per_token += model_dim * n_heads
            tflops_per_token += (n_heads + n_kv_heads) * head_dim * seq_len  # qkv - kkv
        else:
            raise ValueError(f"Unknown attention activation function: {attn_act_func}")

        # norm flops
        norm_flops = model_dim * 2 + ((qdim + kdim) if qknorm else 0)
        # conv flops
        conv_flops = conv_width * (qdim + kdim + vdim) * 2
        tflops_per_token += norm_flops + conv_flops
        # residual
        # TODO
        res_flops = 0.0

        return tflops_per_token
