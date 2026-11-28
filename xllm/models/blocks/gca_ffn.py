from typing import Optional, Tuple, Any
from torch import Tensor
from torch import nn

from xllm.modules import (
    GatedChunkAttention,
    NormalizedFeedForwardNetwork,
)
from xllm.modules.residual import num_params_in_residual
from xllm.config import ModelConf
from xllm.models.fused_blocks import GekkoGCABlockFunction


class GekkoGCABlock(nn.Module):
    def __init__(self, cfg: ModelConf, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.residual_func = cfg.residual_func
        self.residual_heads = cfg.residual_heads
        self.fused_block = cfg.fused_block
        self.recompute_q = cfg.recompute_q
        self.recompute_kv = cfg.recompute_v
        self.recompute_sca = cfg.recompute_attention
        self.recompute_awk = cfg.recompute_awk
        self.recompute_fc1_out = cfg.recompute_fc1_out
        self.recompute_fc3_out = cfg.recompute_fc3_out
        self.apply_rmsnorm = cfg.apply_rmsnorm
        self.rmsnorm_eps = cfg.rmsnorm_eps

        self.gca = GatedChunkAttention(
            layer_id=layer_id,
            mdim=cfg.model_dim,
            n_heads=cfg.num_heads,
            n_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            v_head_dim=cfg.v_head_dim,
            rope_head_dim=cfg.rope_head_dim,
            chunk_size=cfg.chunk_size,
            attn_act_func=cfg.attn_act_func,
            attn_gate_func=cfg.attn_gate_func,
            awm_forget_gate=cfg.apply_awm_gate,
            awm_orthogonal_update=cfg.awm_orthogonal_update,
            causal_conv_width=cfg.causal_conv_width,
            causal_conv_backend=cfg.causal_conv_backend,
            causal_conv_weight_normalization=cfg.causal_conv_weight_normalization,
            sca_backend=cfg.sca_backend,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            hidden_dropout=cfg.hidden_dropout,
            timenorm_num_groups=cfg.timenorm_num_groups,
            timenorm_beta1=cfg.timenorm_beta1,
            timenorm_beta2=cfg.timenorm_beta2,
            timenorm_backend=cfg.timenorm_backend,
            timenorm_eps=cfg.timenorm_eps,
            rmsnorm_eps=cfg.rmsnorm_eps,
            stream_eps_scale=cfg.stream_eps_scale,
            apply_bias_term=cfg.apply_bias_term,
            memory_efficient_norm=cfg.memory_efficient_norm,
            residual_func=cfg.residual_func,
            residual_heads=cfg.residual_heads,
            init_mode=cfg.init_mode,
            init_std=cfg.init_std,
        )

        self.nffn = NormalizedFeedForwardNetwork(
            layer_id=layer_id,
            model_dim=cfg.model_dim,
            ffn_hidden_dim=cfg.ffn_hidden_dim,
            swiglu=cfg.swiglu,
            dropout=cfg.dropout,
            hidden_dropout=cfg.hidden_dropout,
            norm_num_groups=cfg.layernorm_num_groups,
            norm_affine=cfg.norm_affine,
            layernorm_eps=cfg.layernorm_eps,
            rmsnorm_eps=cfg.rmsnorm_eps,
            stream_eps_scale=cfg.stream_eps_scale,
            apply_rmsnorm=cfg.apply_rmsnorm,
            memory_efficient_norm=cfg.memory_efficient_norm,
            residual_func=cfg.residual_func,
            residual_heads=cfg.residual_heads,
            init_mode=cfg.init_mode,
            init_std=cfg.init_std,
        )

    def forward(
        self,
        x: Tensor,
        freqs_cis: Optional[Tensor],
        bos_mask: Optional[Tensor] = None,
        segment_idx: Optional[Tensor] = None,
        prev_segment_count: Optional[Tensor] = None,
        moe_router_load_balancing_type: Optional[str] = None,
        attn_stability_control: int = 0,
        deterministic: bool = True,
        cache: Optional[Tuple[Tuple[Tensor, Tensor, int],
                              Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
                              Tuple[Tensor, Tensor, Tensor],
                              Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Any]]:
        if self.fused_block and self.training:
            fn = GekkoGCABlockFunction
            return fn.apply(
                self.layer_id,
                x,
                freqs_cis,
                bos_mask,
                segment_idx,
                prev_segment_count,
                self.gca.timenorm.weight,
                self.gca.timenorm.bias,
                self.gca.timenorm.prior_count,
                self.gca.timenorm.prior_mean,
                self.gca.timenorm.prior_var,
                self.gca.timenorm.groups_per_partition,
                self.gca.timenorm.beta1,
                self.gca.timenorm.beta2,
                self.gca.timenorm_backend,
                self.gca.wq.weight,
                self.gca.wk.weight,
                self.gca.wv.weight,
                self.gca.wv.bias,
                self.gca.wr.weight,
                self.gca.wr.bias,
                self.gca.wo.weight,
                self.gca.q_conv.weight,
                self.gca.k_conv.weight,
                self.gca.v_conv.weight,
                self.gca.causal_conv_backend,
                self.gca.q_conv.normalize_weight,
                self.gca.query_norm.weight,
                self.gca.key_norm.weight,
                self.gca.local_heads,
                self.gca.local_kv_heads,
                self.gca.chunk_size,
                self.gca.head_dim,
                self.gca.v_head_dim,
                self.gca.rope_head_dim,
                self.gca.attn_act_func,
                self.gca.attn_gate_fn,
                self.gca.sca_backend,
                self.gca.awm_orthogonal_update,
                self.nffn.norm.weight,
                self.nffn.norm.bias,
                self.nffn.norm.groups_per_partition,
                self.nffn.fc1.weight,
                self.nffn.fc2.weight,
                self.nffn.fc3.weight if self.nffn.swiglu else None,
                self.gca.dropout,
                self.gca.attention_dropout,
                self.gca.hidden_dropout,
                self.nffn.swiglu,
                self.gca.timenorm.eps,
                self.nffn.norm.eps,
                self.rmsnorm_eps,
                self.apply_rmsnorm,
                self.residual_func,
                self.residual_heads,
                attn_stability_control,
                deterministic,
                self.recompute_q,
                self.recompute_kv,
                self.recompute_sca,
                self.recompute_awk,
                self.recompute_fc1_out,
                self.recompute_fc3_out,
            )

        y, cache = self.gca(
            x, freqs_cis, bos_mask, segment_idx, prev_segment_count, attn_stability_control, deterministic, cache
        )
        out = self.nffn(y)
        return out, None, cache

    def num_parameters(self):
        model_dim = self.gca.mdim
        head_dim = self.gca.head_dim
        v_head_dim = self.gca.v_head_dim
        n_heads = self.gca.num_heads
        n_kv_heads = self.gca.num_kv_heads
        apply_bias_term = self.gca.apply_bias_term
        conv_width = self.gca.q_conv.kernel_size
        attn_act_func = self.gca.attn_act_func
        ffn_hidden_dim = self.nffn.hidden_dim
        swiglu = self.nffn.swiglu
        # GDA
        gda_params = model_dim * (n_heads + n_kv_heads) * (head_dim + v_head_dim)
        # AWM forget gate
        if self.gca.forget_gate:
            gda_params += (model_dim + 2) * n_kv_heads * head_dim
        # out proj
        gda_params += model_dim * n_heads * v_head_dim
        # bias term
        gda_params += (n_kv_heads + n_heads) * v_head_dim if apply_bias_term else n_kv_heads * v_head_dim
        # residual
        gda_params += num_params_in_residual(self.residual_func, model_dim, self.residual_heads, n_heads * v_head_dim)
        # conv params
        conv_params = conv_width * (n_heads * head_dim + n_kv_heads * (head_dim + v_head_dim))
        # norm params
        norm_params = model_dim * (3 if self.apply_rmsnorm else 4) + head_dim * (n_heads + n_kv_heads)
        if attn_act_func == 'softdelta':
            gda_params += model_dim * (n_kv_heads * head_dim + n_heads)
            conv_params += conv_width * n_kv_heads * head_dim
            norm_params += n_kv_heads * head_dim
        # FFN
        ffn_params = model_dim * ffn_hidden_dim * (3 if swiglu else 2)
        ffn_params += num_params_in_residual(self.residual_func, model_dim, self.residual_heads, ffn_hidden_dim)

        activated_params = gda_params + conv_params + norm_params + ffn_params
        total_params = activated_params
        return total_params, activated_params

    def tflops_per_token(self, seq_len: int):
        chunk_size = self.gca.chunk_size
        model_dim = self.gca.mdim
        head_dim = self.gca.head_dim
        v_head_dim = self.gca.v_head_dim
        n_heads = self.gca.num_heads
        n_kv_heads = self.gca.num_kv_heads
        conv_width = self.gca.q_conv.kernel_size
        attn_act_func = self.gca.attn_act_func

        qdim = n_heads * head_dim
        kdim = n_kv_heads * head_dim
        vdim = n_kv_heads * v_head_dim
        hdim = n_heads * v_head_dim
        # TFLoPs
        _, tflops_per_token = self.num_parameters()
        # norm flops
        norm_flops = model_dim * 8 + hdim * 2 + qdim + kdim
        # conv flops
        conv_flops = conv_width * (qdim + kdim + vdim) * 2
        tflops_per_token += norm_flops + conv_flops
        # SCA flops
        if attn_act_func == 'softmax':
            tflops_per_token += (qdim + hdim) * chunk_size * 1.5
        elif attn_act_func == 'softdelta':
            tflops_per_token += (qdim + hdim + kdim) * chunk_size * 1.5
        else:
            raise ValueError(f"Unknown attention activation function: {attn_act_func}")
        # AWM flops
        tflops_per_token += (kdim * 2 + qdim) * v_head_dim
        # residual
        # TODO
        res_flops = 0.0

        return tflops_per_token
