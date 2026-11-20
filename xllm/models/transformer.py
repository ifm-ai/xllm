from typing import Optional, Tuple, List, Any
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from xllm.configuration import ConfStore
from xllm.data.dataset_streamer.tokenizer import Tokenizer
from xllm.modules.fused_ops import memory_efficient_dropout
from xllm.modules.rotary_positional_embedding import RotaryEmbedding
from xllm.modules.model_parallel import ParallelEmbedding
from xllm.modules.context_parallel import gather_from_context_parallel_region
from xllm.modules.utils import (
    get_bos_mask,
    get_segment_idx_from_bos_mask,
)
from xllm.config import ModelConf
from xllm.health_metrics import get_active_health_collector
from xllm.distributed import (
    get_context_parallel_world_size,
    get_context_parallel_rank,
)
from xllm.models.xllm import XLLModel
from xllm.models.blocks import (
    TransformerBlock,
    TransformerMoEBlock,
    TransformerMoVABlock,
    TransformerOutputLayer
)
from xllm.utils import get_init_fn


class Transformer(XLLModel):
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

        assert self.v_head_dim == self.head_dim
        if self.rope_head_dim > 0:
            self.rope = RotaryEmbedding(self.rope_head_dim, cfg.chunk_size * 16, base=cfg.rope_base)
        else:
            self.rope = None
        self.qknorm = cfg.qknorm
        self.apply_attn_gate = cfg.apply_attn_gate

        self.layers = nn.ModuleList()
        for layer_id in range(self.num_layers):
            block_cls = TransformerBlock if layer_id < self.num_dense_layers else \
                (TransformerMoVABlock if self.num_values > 0 else TransformerMoEBlock)
            layer = block_cls(cfg, layer_id)
            self.layers.append(wrap(layer))

        self.output = wrap(TransformerOutputLayer(cfg))

    def forward(
        self,
        tokens: Tensor,
        multi_segments: bool,
        targets: Optional[Tensor] = None,
        token_mask: Optional[Tensor] = None,
        moe_router_load_balancing_type: Optional[str] = None,
        attn_stability_control: int = 0,
        deterministic: bool = True,
        cache: Optional[List[Tuple[Tuple[Tensor, Tensor, int],
                                   Tuple[Tensor, Tensor, Tensor]]]] = None,
    ):
        bsz, seq_len = tokens.shape

        if targets is None:
            assert token_mask is None

        if multi_segments:
            assert cache is None
            bos_mask = get_bos_mask(tokens, self.eos_id, use_eos=True)
            segments = self.get_segment_arg(
                bos_mask, self.causal_attn_backend, self.context_parallel_rank, self.context_parallel_size
            )
        else:
            bos_mask = None
            segments = None

        if self.training:
            assert cache is None, "training model does not support kv cache."
            assert seq_len % self.context_parallel_size == 0
            seq_len = seq_len // self.context_parallel_size
            assert seq_len % self.chunk_size == 0
            cache_len = seq_len * self.context_parallel_rank
            start = self.context_parallel_rank * seq_len
            end = (self.context_parallel_rank + 1) * seq_len
            tokens = tokens[:, start:end]
            if bos_mask is not None:
                bos_mask = bos_mask[:, :end]
            if targets is not None:
                targets = targets[:, start:end]
                token_mask = token_mask[:, start:end] if token_mask is not None else None
        else:
            assert self.context_parallel_size == 1, "inference mode does not support context parallel."
            cache_len = 0 if cache is None else cache[0][0][-1]

        # embeddings
        emb = self.embed(tokens)
        x = memory_efficient_dropout(emb, self.dropout, self.training)
        # rope frequencies
        freq_cis = None if self.rope is None else self.rope.get_freqs_cis(cache_len, cache_len + seq_len, x.device)

        aux_loss_sum = None
        health_collector = get_active_health_collector()
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            if self.layerwise_ckpt:
                x, aux_loss, layer_cache = checkpoint(
                    layer, x, freq_cis, bos_mask, segments, moe_router_load_balancing_type,
                    attn_stability_control, deterministic, layer_cache,
                    use_reentrant=False, preserve_rng_state=True
                )
            else:
                x, aux_loss, layer_cache = layer(
                    x, freq_cis, bos_mask, segments, moe_router_load_balancing_type,
                    attn_stability_control, deterministic, layer_cache
                )

            if health_collector is not None:
                health_collector.record_activation(i, x)

            if aux_loss is not None:
                aux_loss_sum = aux_loss if aux_loss_sum is None else aux_loss_sum + aux_loss

            if cache is not None:
                cache[i] = layer_cache

        logits_or_loss = self.output(x, targets, token_mask)
        if targets is not None and self.context_parallel_size > 1:
            logits_or_loss = gather_from_context_parallel_region(logits_or_loss)

        return logits_or_loss, aux_loss_sum, cache

    def support_multi_segments_with_cache(self) -> bool:
        # TODO: causal attn w. xattn backend supports it.
        return False

    @staticmethod
    def get_segment_arg(bos_mask, causal_attn_backend, cp_rank, cp_size):
        bsz, seq_len = bos_mask.shape
        seq_len = seq_len // cp_size
        start = cp_rank * seq_len
        end = (cp_rank + 1) * seq_len
        if causal_attn_backend == 'flash':
            if cp_size > 1:
                assert bsz == 1, f"Flash attention does not support context parallel w. bsz {bsz} > 1"
            # query seqlens
            bos_idx_q = torch.nonzero(F.pad(bos_mask[:, (start + 1):end], (1, 0), value=1))
            cu_seqlens_q = bos_idx_q[:, 0] * seq_len + bos_idx_q[:, 1]
            cu_seqlens_q = F.pad(cu_seqlens_q, (0, 1), value=bsz * seq_len)
            max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
            # key seqlens
            bos_idx_k = torch.nonzero(F.pad(bos_mask[:, 1:end], (1, 0), value=1))
            cu_seqlens_k = bos_idx_k[:, 0] * end + bos_idx_k[:, 1]
            cu_seqlens_k = F.pad(cu_seqlens_k, (0, 1), value=bsz * end)
            num_q_seqs = cu_seqlens_q.shape[0]
            cu_seqlens_k = cu_seqlens_k[-num_q_seqs:]
            offset = cu_seqlens_k[0]
            cu_seqlens_k = cu_seqlens_k - offset
            max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
            total_seqlen_k = cu_seqlens_k[-1].item()
            return cu_seqlens_q.int(), cu_seqlens_k.int(), max_seqlen_q, max_seqlen_k, total_seqlen_k
        elif causal_attn_backend == 'swift':
            segment_idx = get_segment_idx_from_bos_mask(bos_mask)
            # B x L1
            q_segment_idx = segment_idx[:, start:end]
            # B x L2
            k_segment_idx = segment_idx[:, :end]
            return q_segment_idx, k_segment_idx
        elif causal_attn_backend == 'xattn':
            segment_idx = get_segment_idx_from_bos_mask(bos_mask)
            # B x L1
            q_segment_idx = segment_idx[:, start:end]
            # B x 1
            last_qidx = q_segment_idx[:, 0:1]
            # B x L2
            segment_idx = segment_idx[:, :end]
            total_seqlen_k = torch.sum(segment_idx >= last_qidx, dim=1).max().item()
            return segment_idx, total_seqlen_k
        else:
            assert causal_attn_backend is None
            segment_idx = get_segment_idx_from_bos_mask(bos_mask)
            # B x L1
            q_segment_idx = segment_idx[:, start:end]
            # B x L2
            k_segment_idx = segment_idx[:, :end]
            # B x L1 x L2
            seg_mask = torch.ne(q_segment_idx.unsqueeze(2), k_segment_idx.unsqueeze(1))
            return seg_mask


# register some models configurations
# fmt: off

# llama models
ConfStore["llama3-7B"] = ModelConf(arch='transformer', num_layers=32, model_dim=4096, num_heads=32, num_kv_heads=8,
                                   rope_base=500000, ffn_hidden_dim=14336, swiglu=True, apply_rmsnorm=True, layernorm_num_groups=1)

ConfStore["llama3-70B"] = ModelConf(arch='transformer', num_layers=80, model_dim=8192, num_heads=64, num_kv_heads=8,
                                    rope_base=500000, ffn_hidden_dim=28672, swiglu=True, apply_rmsnorm=True, layernorm_num_groups=1)

ConfStore["llama3-400B"] = ModelConf(arch='transformer', num_layers=126, model_dim=16384, num_heads=128, num_kv_heads=8,
                                     rope_base=500000, ffn_hidden_dim=53248, swiglu=True, apply_rmsnorm=True, layernorm_num_groups=8)

# K2 models
ConfStore["k2v2-70B"] = ConfStore["llama3-70B"]

ConfStore["k2-horizon-0.9B"] = ModelConf(arch='transformer', num_layers=28, model_dim=1536,
                                         num_heads=32, num_kv_heads=8, head_dim=64,
                                         rope_base=500000, ffn_hidden_dim=5120, swiglu=True,
                                         apply_rmsnorm=True, layernorm_num_groups=1, rmsnorm_eps=1e-6)

ConfStore["k2-horizon-3.7B"] = ModelConf(arch='transformer', num_layers=36, model_dim=2560,
                                         num_heads=32, num_kv_heads=8, head_dim=128,
                                         rope_base=500000, ffn_hidden_dim=10240, swiglu=True,
                                         apply_rmsnorm=True, layernorm_num_groups=2, rmsnorm_eps=1e-6)

ConfStore["k2-horizon-7B"] = ModelConf(arch='transformer', num_layers=36, model_dim=4096,
                                       num_heads=32, num_kv_heads=8, head_dim=128,
                                       rope_base=500000, ffn_hidden_dim=12288, swiglu=True,
                                       apply_rmsnorm=True, layernorm_num_groups=4, rmsnorm_eps=1e-6)

ConfStore["k2-horizon-32B"] = ModelConf(arch='transformer', num_layers=64, model_dim=5120,
                                        num_heads=64, num_kv_heads=8, head_dim=128,
                                        rope_base=500000, ffn_hidden_dim=26624, swiglu=True,
                                        apply_rmsnorm=True, layernorm_num_groups=4, rmsnorm_eps=1e-6)

# MoE
ConfStore["mixtral8x7B"] = ModelConf(arch='transformer', num_layers=32, model_dim=4096, num_heads=32, num_kv_heads=8,
                                     ffn_hidden_dim=14336, swiglu=True, rope_base=500000, apply_rmsnorm=True, layernorm_num_groups=1,
                                     num_experts=8, num_activated_experts=2, expert_inter_dim=14336, moe_router_score_func='softmax')

ConfStore["k2-horizon-moe-36B-a4B"] = ModelConf(arch='transformer', num_layers=48, num_dense_layers=3, model_dim=2560,
                                                num_heads=32, num_kv_heads=8, head_dim=128, rope_head_dim=128, rope_base=500000,
                                                apply_attn_gate=True, attn_gate_func='softplus', ffn_hidden_dim=6144, swiglu=True,
                                                num_experts=128, num_activated_experts=8, num_shared_experts=1, expert_inter_dim=768,
                                                moe_router_score_func='sigmoid', moe_router_bias=True, moe_router_bias_update_rate=1e-3,
                                                moe_router_scaling_factor=2.5, apply_rmsnorm=True, layernorm_num_groups=2, rmsnorm_eps=1e-6)

ConfStore["k2-horizon-mova-36B-a4B"] = ModelConf(arch='transformer', num_layers=48, num_dense_layers=3, model_dim=2560,
                                                 num_heads=32, num_kv_heads=8, head_dim=128, rope_head_dim=128, rope_base=500000,
                                                 apply_attn_gate=True, attn_gate_func='softplus', ffn_hidden_dim=6144, swiglu=True,
                                                 num_values=64, num_activated_values=4,
                                                 num_experts=100, num_activated_experts=8, num_shared_experts=1, expert_inter_dim=768,
                                                 moe_router_score_func='sigmoid', moe_router_bias=True, moe_router_bias_update_rate=1e-3,
                                                 moe_router_scaling_factor=2.5, apply_rmsnorm=True, layernorm_num_groups=2, rmsnorm_eps=1e-6)


ConfStore["k2-horizon-375B-a23B"] = ModelConf(arch='transformer', num_layers=61, num_dense_layers=3, model_dim=6144,
                                              num_heads=48, num_kv_heads=8, head_dim=128, rope_head_dim=64, rope_base=500000,
                                              apply_attn_gate=False, ffn_hidden_dim=16384, swiglu=True,
                                              num_experts=192, num_activated_experts=8, num_shared_experts=1, expert_inter_dim=1792,
                                              moe_router_score_func='sigmoid', moe_router_bias=True, moe_router_bias_update_rate=1e-3,
                                              moe_router_scaling_factor=2.5, apply_rmsnorm=True, layernorm_num_groups=1, rmsnorm_eps=1e-6)
