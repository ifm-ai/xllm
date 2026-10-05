from typing import Optional, Tuple, List, Any
import torch
from torch import Tensor
from torch import nn
from torch.utils.checkpoint import checkpoint

from xllm.configuration import ConfStore
from xllm.data.dataset_streamer.tokenizer import Tokenizer
from xllm.modules import (
    GatedDeltaAttention,
    NormalizedFeedForwardNetwork,
    RotaryEmbedding,
    TimestepDecayNorm,
)
from xllm.modules.residual import num_params_in_residual
from xllm.modules.fused_ops import memory_efficient_dropout
from xllm.modules.model_parallel import (
    ParallelEmbedding,
    ColumnParallelLinear,
    vocab_parallel_cross_entropy,
)
from xllm.config import ModelConf
from xllm.distributed import (
    get_context_parallel_world_size,
    get_context_parallel_rank,
)
from xllm.modules.context_parallel import (
    should_send_to_next,
    should_recv_from_prev,
    send_to_next_context_parallel_region,
    recv_from_prev_context_parallel_region,
    gather_from_context_parallel_region,
)
from xllm.models.xllm import XLLModel
from xllm.models.fused_blocks import (
    GekkoBlockFunction,
    GekkoOutputLayerFunction
)
from xllm.utils import get_init_fn


class GekkoBlock(nn.Module):
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
        self.timenorm_eps = cfg.timenorm_eps
        self.layernorm_eps = cfg.layernorm_eps
        self.rmsnorm_eps = cfg.rmsnorm_eps

        self.gda = GatedDeltaAttention(
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
        fp32_attn_output: bool = False,
        deterministic: bool = True,
        cache: Optional[Tuple[Tuple[Tensor, Tensor, int],
                              Tuple[Tensor, Tensor, Tensor],
                              Tuple[Tensor, Tensor, Tensor],
                              Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Any]]:
        if self.fused_block and self.training:
            fn = GekkoBlockFunction
            return fn.apply(
                x,
                freqs_cis,
                bos_mask,
                segment_idx,
                prev_segment_count,
                self.gda.timenorm.weight,
                self.gda.timenorm.bias,
                self.gda.timenorm.prior_count,
                self.gda.timenorm.prior_mean,
                self.gda.timenorm.prior_var,
                self.gda.timenorm.groups_per_partition,
                self.gda.timenorm.beta1,
                self.gda.timenorm.beta2,
                self.gda.timenorm_backend,
                self.gda.wq.weight,
                self.gda.wk.weight,
                self.gda.wv.weight,
                self.gda.wv.bias,
                self.gda.wr.weight,
                self.gda.wr.bias,
                self.gda.wg.weight if self.gda.attn_act_func == 'softdelta' else None,
                self.gda.wo.weight,
                self.gda.q_conv.weight,
                self.gda.k_conv.weight,
                self.gda.v_conv.weight,
                self.gda.causal_conv_backend,
                self.gda.q_conv.normalize_weight,
                self.gda.query_norm.weight,
                self.gda.key_norm.weight,
                self.gda.local_heads,
                self.gda.local_kv_heads,
                self.gda.chunk_size,
                self.gda.head_dim,
                self.gda.v_head_dim,
                self.gda.rope_head_dim,
                self.gda.attn_gate_fn,
                self.gda.sca_backend,
                self.gda.awm_orthogonal_update,
                None,
                self.nffn.norm.weight,
                self.nffn.norm.bias,
                self.nffn.norm.groups_per_partition,
                self.nffn.fc1.weight,
                self.nffn.fc2.weight,
                self.nffn.fc3.weight if self.nffn.swiglu else None,
                None,
                self.gda.dropout,
                self.gda.attention_dropout,
                self.gda.hidden_dropout,
                self.nffn.swiglu,
                self.timenorm_eps,
                self.layernorm_eps,
                self.rmsnorm_eps,
                self.apply_rmsnorm,
                self.residual_func,
                self.residual_heads,
                fp32_attn_output,
                deterministic,
                self.recompute_q,
                self.recompute_kv,
                self.recompute_sca,
                self.recompute_awk,
                self.recompute_fc1_out,
                self.recompute_fc3_out
            )

        y, cache = self.gda(
            x, freqs_cis, bos_mask, segment_idx, prev_segment_count, fp32_attn_output, deterministic, cache
        )
        out = self.nffn(y)
        return out, None, cache


class GekkoOutputLayer(nn.Module):
    def __init__(self, cfg: ModelConf):
        super().__init__()

        self.model_dim = cfg.model_dim
        self.output_size = cfg.vocab_size
        self.recompute_logits = cfg.recompute_logits

        self.final_norm = TimestepDecayNorm(
            self.model_dim, cfg.timenorm_num_groups, cfg.timenorm_beta1, cfg.timenorm_beta2,
            eps=cfg.timenorm_eps, backend=cfg.timenorm_backend,
            memory_efficient=cfg.memory_efficient_norm
        )

        init_fn = get_init_fn('gaussian', dim=self.model_dim, std=cfg.init_logits_std)
        self.output = ColumnParallelLinear(self.model_dim, self.output_size, bias=False,
                                           input_is_parallel=True, gather_output=False, init_method=init_fn)

        self.context_parallel_rank = get_context_parallel_rank()

    def _receive_prev_count(self, x, bos_mask):
        bsz, seq_len, _ = x.size()
        if bos_mask is None:
            prev_count = torch.full((bsz,), seq_len * self.context_parallel_rank, dtype=torch.int64, device=x.device)
        else:
            prev_count = torch.empty(bsz, dtype=torch.int64, device=x.device)
            prev_count = recv_from_prev_context_parallel_region(prev_count)
        return prev_count

    def _receive_prev_tensors(self, x):
        bsz, seq_len, _ = x.size()
        n_groups = self.final_norm.groups_per_partition
        prev_tensor = torch.empty((bsz, n_groups * 2), dtype=x.dtype, device=x.device, requires_grad=self.training)
        prev_tensor = recv_from_prev_context_parallel_region(prev_tensor)
        return prev_tensor

    def _pack_prev_tensors(self, prev_mean, prev_var):
        # B x (G*2)
        prev_tensor = torch.cat([prev_mean, prev_var], dim=-1)
        return prev_tensor

    def _unpack_prev_tensors(self, x, prev_tensor):
        n_groups = self.final_norm.groups_per_partition
        prev_mean, prev_var = torch.split(prev_tensor, [n_groups, n_groups], dim=-1)
        prev_mean = prev_mean
        prev_var = prev_var
        return prev_mean, prev_var

    def forward(
        self,
        x: Tensor,
        y: Optional[Tensor],
        mask: Optional[Tensor] = None,
        bos_mask: Optional[Tensor] = None,
        cache: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    ):
        bsz, seq_len, _ = x.size()
        if bos_mask is not None:
            bos_mask = bos_mask[:, -seq_len:]

        if self.training:
            assert y is not None
            fn = GekkoOutputLayerFunction
            return fn.apply(
                x,
                y,
                mask,
                bos_mask,
                self.final_norm.weight,
                self.final_norm.bias,
                self.final_norm.prior_count,
                self.final_norm.prior_mean,
                self.final_norm.prior_var,
                self.final_norm.groups_per_partition,
                self.final_norm.beta1,
                self.final_norm.beta2,
                self.final_norm.backend,
                self.output.weight,
                self.final_norm.eps,
                self.recompute_logits,
            )

        if cache is not None:
            prev_count, prev_mean, prev_var = cache
        elif should_recv_from_prev():
            prev_count = self._receive_prev_count(x, bos_mask)
            prev_tensor = self._receive_prev_tensors(x)
            prev_mean, prev_var = self._unpack_prev_tensors(x, prev_tensor)
        else:
            prev_count, prev_mean, prev_var = None, None, None

        x, prev_count, prev_mean, prev_var = self.final_norm(x, bos_mask, prev_count, prev_mean, prev_var)

        if cache is not None:
            cache = (prev_count.detach(), prev_mean.detach(), prev_var.detach())
        elif should_send_to_next():
            if bos_mask is not None:
                send_to_next_context_parallel_region(prev_count)
            prev_tensor = self._pack_prev_tensors(prev_mean, prev_var)
            prev_tensor = send_to_next_context_parallel_region(prev_tensor)
            x = x + prev_tensor.to(x).mean() * 0

        logits = self.output(x).float()
        if y is None:
            return logits, cache
        else:
            loss = vocab_parallel_cross_entropy(logits, y)
            if mask is not None:
                loss = loss * mask.to(loss)
            return loss, cache



class Gekko(XLLModel):
    """
    Gekko Architecture in
    """
    def __init__(self, cfg: ModelConf, tokenizer: Tokenizer):
        super().__init__(cfg, tokenizer)

        self.chunk_size = cfg.chunk_size
        self.context_parallel_size = get_context_parallel_world_size()
        self.context_parallel_rank = get_context_parallel_rank()

        if cfg.ddp_backend == 'fsdp1':
            from torch.distributed.fsdp.wrap import wrap
        else:
            assert cfg.ddp_backend == 'fsdp2'
            from xllm.distributed.wrap import wrap

        init_fn = get_init_fn('gaussian', dim=self.model_dim, std=cfg.init_embed_std)
        self.embed = wrap(ParallelEmbedding(
            self.vocab_size, self.model_dim, scale_emb=cfg.scale_emb, gather_output=False, init_method=init_fn
        ))
        self.causal_conv_width = cfg.causal_conv_width
        self.apply_bias_term = cfg.apply_bias_term

        if self.rope_head_dim > 0:
            self.rope = RotaryEmbedding(self.rope_head_dim, cfg.chunk_size * 16, base=cfg.rope_base)
        else:
            self.rope = None

        self.layers = nn.ModuleList()
        for layer_id in range(self.num_layers):
            layer = GekkoBlock(cfg, layer_id)
            self.layers.append(wrap(layer))

        self.output = wrap(GekkoOutputLayer(cfg))

    def forward(
        self,
        tokens: Tensor,
        multi_segments: bool,
        targets: Optional[Tensor] = None,
        token_mask: Optional[Tensor] = None,
        moe_router_load_balancing_type: Optional[str] = None,
        fp32_attn_output: bool = False,
        deterministic: bool = True,
        cache: Optional[Tuple[List[Tuple[Tuple[Tensor, Tensor, int],
                                         Tuple[Tensor, Tensor, Tensor, Tensor],
                                         Tuple[Tensor, Tensor, Tensor],
                                         Tuple[Tensor, Tensor, Tensor]]],
                              Tuple[Tensor, Tensor, Tensor],
                              Tuple[Tensor, Tensor, Tensor]]] = None,
    ):

        bsz, seq_len = tokens.shape

        if targets is None:
            assert token_mask is None

        if multi_segments:
            bos_mask = torch.eq(tokens, self.bos_id)
            segment_idx = torch.cumsum(bos_mask, dim=-1)
            prev_segment_count = None
        else:
            bos_mask = None
            segment_idx = None
            prev_segment_count = None

        if self.training:
            assert cache is None, "training model does not support kv cache."
            assert seq_len % self.context_parallel_size == 0
            seq_len = seq_len // self.context_parallel_size
            assert seq_len % self.chunk_size == 0
            # split tokens into chunks
            cache_layers, cache_output, cache_segment = None, None, None
            start = self.context_parallel_rank * seq_len
            prev = max(0, start - self.chunk_size)
            end = (self.context_parallel_rank + 1) * seq_len
            tokens = tokens[:, start:end]
            if bos_mask is not None:
                if self.context_parallel_rank > 0:
                    prev_segment_count = segment_idx[:, max(prev - 1, 0)]
                segment_idx = segment_idx[:, prev:end]
                bos_mask = bos_mask[:, prev:end]
            if targets is not None:
                targets = targets[:, start:end]
                token_mask = token_mask[:, start:end] if token_mask is not None else None
        else:
            assert self.context_parallel_size == 1, "inference mode does not support context parallel."
            cache_layers, cache_output, cache_segment = cache
            start = 0 if cache_layers[0][0] is None else cache_layers[0][0][-1]
            end = start + seq_len
            if seq_len >= self.chunk_size:
                assert start % self.chunk_size == 0 and end % self.chunk_size == 0
            elif seq_len > 1:
                assert start % self.chunk_size == 0

            if bos_mask is not None:
                prev_bos_mask, prev_segment_idx, prev_segment_count = cache_segment
                if prev_bos_mask is not None:
                    bos_mask = torch.cat([prev_bos_mask, bos_mask], dim=1)
                    segment_idx = torch.cat([prev_segment_idx, segment_idx + prev_segment_idx[:, -1:]], dim=1)

        # embeddings
        emb = self.embed(tokens)
        x = memory_efficient_dropout(emb, self.dropout, self.training)
        # rope frequencies
        freq_cis = None if self.rope is None else self.rope.get_freqs_cis(start, end, x.device)

        aux_loss_sum = None
        for i, layer in enumerate(self.layers):
            layer_cache = cache_layers[i] if cache is not None else None
            if self.layerwise_ckpt:
                x, aux_loss, layer_cache = checkpoint(
                    layer, x, freq_cis, bos_mask, segment_idx, prev_segment_count,
                    moe_router_load_balancing_type, fp32_attn_output, deterministic,
                    layer_cache, use_reentrant=False, preserve_rng_state=True
                )
            else:
                x, aux_loss, layer_cache = layer(
                    x, freq_cis, bos_mask, segment_idx, prev_segment_count,
                    moe_router_load_balancing_type, fp32_attn_output, deterministic, layer_cache
                )

            if aux_loss is not None:
                aux_loss_sum = aux_loss if aux_loss_sum is None else aux_loss_sum + aux_loss

            if cache is not None:
                cache_layers[i] = layer_cache

        logits_or_loss, cache_output = self.output(x, targets, token_mask, bos_mask, cache_output)
        if targets is not None and self.context_parallel_size > 1:
            logits_or_loss = gather_from_context_parallel_region(logits_or_loss)

        if cache is not None:
            if bos_mask is not None:
                prev_bos_mask = bos_mask
                prev_segment_idx = segment_idx
                if prev_segment_count is None:
                    prev_segment_count = segment_idx[:, 0]

                if end % self.chunk_size == 0:
                    length = prev_segment_idx.shape[1]
                    prev_segment_count = prev_segment_idx[:, max(length - self.chunk_size - 1, 0)]
                    prev_segment_idx = prev_segment_idx[:, length - self.chunk_size:]
                    prev_bos_mask = prev_bos_mask[:, length - self.chunk_size:]

                cache_segment = (prev_bos_mask, prev_segment_idx, prev_segment_count)
            cache = (cache_layers, cache_output, cache_segment)

        return logits_or_loss, aux_loss_sum, cache

    def support_multi_segments_with_cache(self) -> bool:
        return True

    def num_parameters(self):
        embed_params = self.model_dim * self.vocab_size * 2
        gda_params_per_block = self.model_dim * (self.num_heads + self.num_kv_heads) * (self.head_dim + self.v_head_dim)
        # out proj
        gda_params_per_block += self.model_dim * self.num_heads * self.v_head_dim
        if self.apply_bias_term:
            gda_params_per_block += (self.num_kv_heads + self.num_heads) * self.v_head_dim
        else:
            gda_params_per_block += self.num_kv_heads * self.v_head_dim
        gda_params_per_block += num_params_in_residual(self.residual_func, self.model_dim, self.residual_heads, self.num_heads * self.v_head_dim)
        # conv params
        conv_params_per_block = self.causal_conv_width * (self.num_heads * self.head_dim + self.num_kv_heads * (self.head_dim + self.v_head_dim))
        # norm params
        norm_params_per_block = self.model_dim * (3 if self.apply_rmsnorm else 4) + self.head_dim * (self.num_heads + self.num_kv_heads)
        if self.attn_act_func == 'softdelta':
            gda_params_per_block += self.model_dim * self.num_heads * (self.head_dim + 1)
            conv_params_per_block += self.causal_conv_width * self.num_heads * self.head_dim
            norm_params_per_block += self.head_dim * self.num_heads
        # FFN
        ffn_params_per_block = self.model_dim * self.ffn_hidden_dim * (3 if self.swiglu else 2)
        ffn_params_per_block += num_params_in_residual(self.residual_func, self.model_dim, self.residual_heads, self.ffn_hidden_dim)

        activated_params = (gda_params_per_block + conv_params_per_block + norm_params_per_block) * self.num_layers
        activated_params += ffn_params_per_block * self.num_dense_layers + embed_params
        total_params = activated_params
        if self.num_experts > 0:
            shared_expert_params = self.model_dim * self.expert_inter_dim * self.num_shared_experts * 3
            routed_expert_params = self.model_dim * self.expert_inter_dim * self.num_experts * 3
            activated_expert_params = self.model_dim * self.expert_inter_dim * self.num_activated_experts * 3
            router_params = self.model_dim * self.num_experts
            activated_params += (shared_expert_params + activated_expert_params + router_params) * (self.num_layers - self.num_dense_layers)
            total_params += (shared_expert_params + routed_expert_params + router_params) * (self.num_layers - self.num_dense_layers)
        # final norm
        activated_params += self.model_dim * 2
        total_params += self.model_dim * 2
        return total_params, activated_params, embed_params

    def tflops_per_token(self, seq_len: int):
        expansion_factor = 6
        q_dim = self.num_heads * self.head_dim
        k_dim = self.num_kv_heads * self.head_dim
        v_dim = self.num_kv_heads * self.v_head_dim
        h_dim = self.num_heads * self.v_head_dim
        embed_flops = self.model_dim + self.vocab_size
        logits_flops = self.model_dim * self.vocab_size
        # norm flops
        norm_flops = self.model_dim * (2 if self.apply_rmsnorm else 4) + h_dim * 2
        norm_flops += (q_dim + k_dim) * 2
        timestep_norm_flops = self.model_dim * 8
        # residual flops
        # TODO
        res_flops = 0.0
        conv_flops = self.causal_conv_width * (q_dim + k_dim + v_dim) * 2
        sca_flops = self.model_dim * (q_dim + k_dim + v_dim + h_dim)
        sca_flops += (q_dim + h_dim) * self.chunk_size
        awk_flops = (k_dim + q_dim) * self.v_head_dim
        attn_flops = sca_flops + awk_flops + self.model_dim * h_dim
        ffn_flops = self.model_dim * self.ffn_hidden_dim * (3 if self.swiglu else 2)
        moe_flops = self.model_dim * self.expert_inter_dim * (self.num_activated_experts + self.num_shared_experts) * 3
        moe_flops += self.model_dim * self.num_experts

        total_tflops = embed_flops + logits_flops + timestep_norm_flops
        total_tflops += self.num_layers * (norm_flops + timestep_norm_flops + conv_flops + attn_flops + res_flops)
        total_tflops += self.num_dense_layers * ffn_flops
        total_tflops += (self.num_layers - self.num_dense_layers) * moe_flops
        total_tflops = total_tflops * expansion_factor / 10 ** 12
        return total_tflops


# register some models configurations
# fmt: off

ConfStore["gekko-0.9B"] = ModelConf(arch='gekko', num_layers=28, model_dim=1536, num_heads=8, num_kv_heads=4,
                                    head_dim=256, v_head_dim=256, rope_head_dim=128, rope_base=500000, chunk_size=4096,
                                    causal_conv_width=4, causal_conv_weight_normalization=True, ffn_hidden_dim=4096, swiglu=True,
                                    layernorm_num_groups=2, timenorm_num_groups=24, timenorm_eps=1e-5, layernorm_eps=1e-5, rmsnorm_eps=1e-6)

ConfStore["gekko-4B"] = ModelConf(arch='gekko', num_layers=36, model_dim=2560, num_heads=32, num_kv_heads=8,
                                  head_dim=128, v_head_dim=128, rope_head_dim=128, rope_base=500000, chunk_size=4096,
                                  causal_conv_width=4, causal_conv_weight_normalization=True, ffn_hidden_dim=9216, swiglu=True,
                                  layernorm_num_groups=2, timenorm_num_groups=40, timenorm_eps=1e-5, layernorm_eps=1e-5, rmsnorm_eps=1e-6)

ConfStore["gekko-7.1B"] = ModelConf(arch='gekko', num_layers=32, model_dim=4096, num_heads=4, num_kv_heads=4, head_dim=256, v_head_dim=2048,
                                    causal_conv_width=4, chunk_size=4096, rope_base=500000, ffn_hidden_dim=8192, swiglu=True,
                                    layernorm_num_groups=4, timenorm_num_groups=64, timenorm_eps=1e-5, layernorm_eps=1e-5, rmsnorm_eps=1e-6)

# fmt: on
