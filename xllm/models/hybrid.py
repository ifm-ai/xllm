from typing import Optional, Tuple, List, Any
import torch
from torch import Tensor
from torch import nn
from torch.utils.checkpoint import checkpoint

from xllm.configuration import ConfStore
from xllm.data.dataset_streamer.tokenizer import Tokenizer
from xllm.modules.fused_ops import memory_efficient_dropout
from xllm.modules.rotary_positional_embedding import RotaryEmbedding
from xllm.modules.model_parallel import ParallelEmbedding
from xllm.modules.context_parallel import gather_from_context_parallel_region
from xllm.config import ModelConf
from xllm.health_metrics import get_active_health_collector
from xllm.distributed import (
    get_context_parallel_world_size,
    get_context_parallel_rank,
)
from xllm.modules.utils import (
    get_bos_mask,
    get_segment_idx_from_bos_mask
)
from xllm.models.xllm import XLLModel
from xllm.models.blocks import (
    TransformerBlock,
    TransformerMoEBlock,
    TransformerMoVABlock,
    GekkoBlock,
    GekkoOutputLayer
)
from xllm.models.transformer import Transformer
from xllm.utils import get_init_fn


class HybridGekko(XLLModel):
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
            self.vocab_size, self.model_dim, embed_multiplier=cfg.embed_multiplier, gather_output=False, init_method=init_fn
        ))
        self.causal_conv_width = cfg.causal_conv_width
        self.apply_bias_term = cfg.apply_bias_term

        if self.rope_head_dim > 0:
            self.rope = RotaryEmbedding(self.rope_head_dim, cfg.chunk_size * 16, base=cfg.rope_base)
        else:
            self.rope = None

        self.layers = nn.ModuleList()
        for layer_id in range(self.num_layers):
            block_cls = self._get_block_cls(layer_id)
            layer = block_cls(cfg, layer_id)
            self.layers.append(wrap(layer))

        self.output = wrap(GekkoOutputLayer(cfg))

    def _get_block_cls(self, layer_id):
        if layer_id in self.full_attention_layers:
            block_cls = TransformerBlock if layer_id < self.num_dense_layers else \
                (TransformerMoVABlock if self.num_values > 0 else TransformerMoEBlock)
        else:
            block_cls = GekkoBlock
        return block_cls

    def forward(
        self,
        tokens: Tensor,
        multi_segments: bool,
        targets: Optional[Tensor] = None,
        token_mask: Optional[Tensor] = None,
        moe_router_load_balancing_type: Optional[str] = None,
        attn_stability_control: int = 0,
        deterministic: bool = True,
        cache: Optional[Tuple[List[Any], Tuple[Tensor, Tensor, Tensor],
                              Tuple[Tensor, Tensor, Tensor]]] = None,
    ):
        bsz, seq_len = tokens.shape

        if targets is None:
            assert token_mask is None

        if multi_segments:
            assert cache is None or self.support_multi_segments_with_cache()
            bos_mask = get_bos_mask(tokens, self.eos_id, use_eos=True)
            segments = Transformer.get_segment_arg(
                bos_mask, self.causal_attn_backend, self.context_parallel_rank, self.context_parallel_size
            ) if self.causal_attn_backend == 'flash' else None  # flash attention
            segment_idx = get_segment_idx_from_bos_mask(bos_mask)
            prev_segment_count = None
        else:
            segments = None
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
                segment_idx = segment_idx[:, :end]
                bos_mask = bos_mask[:, :end]
            if targets is not None:
                targets = targets[:, start:end]
                token_mask = token_mask[:, start:end] if token_mask is not None else None
        else:
            assert self.context_parallel_size == 1, "inference mode does not support context parallel."
            if cache is not None:
                cache_layers, cache_output, cache_segment = cache
            else:
                cache_layers, cache_output, cache_segment = None, None, None

            start = 0 if cache is None else cache_layers[0][0][-1]
            end = start + seq_len
            if seq_len >= self.chunk_size:
                assert start % self.chunk_size == 0 and end % self.chunk_size == 0
            elif seq_len > 1:
                assert start % self.chunk_size == 0

            if cache is not None and bos_mask is not None:
                prev_bos_mask, prev_segment_idx, prev_segment_count = cache_segment
                if prev_bos_mask is not None:
                    bos_mask = torch.cat([prev_bos_mask, bos_mask], dim=1)
                    segment_idx = torch.cat([prev_segment_idx, segment_idx + prev_segment_idx[:, -1:]], dim=1)

        if segment_idx is not None and segments is None:
            segments = self.get_segment_arg(segment_idx, seq_len, self.causal_attn_backend)

        # embeddings
        emb = self.embed(tokens)
        x = memory_efficient_dropout(emb, self.dropout, self.training)
        # rope frequencies
        freq_cis = None if self.rope is None else self.rope.get_freqs_cis(start, end, x.device)

        aux_loss_sum = None
        health_collector = get_active_health_collector()
        for i, layer in enumerate(self.layers):
            layer_cache = cache_layers[i] if cache is not None else None
            inputs = \
                (
                    x, freq_cis, bos_mask, segments, moe_router_load_balancing_type,
                    attn_stability_control, deterministic, layer_cache
                ) \
                if i in self.full_attention_layers else \
                (
                    x, freq_cis, bos_mask, segment_idx, prev_segment_count,
                    moe_router_load_balancing_type, attn_stability_control, deterministic, layer_cache
                )

            if self.layerwise_ckpt:
                x, aux_loss, layer_cache = checkpoint(
                    layer, *inputs, use_reentrant=False, preserve_rng_state=True
                )
            else:
                x, aux_loss, layer_cache = layer(*inputs)

            if health_collector is not None:
                health_collector.record_activation(i, x)

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

                cache_segment = (prev_bos_mask, prev_segment_idx, prev_segment_count)
            cache = (cache_layers, cache_output, cache_segment)

        return logits_or_loss, aux_loss_sum, cache

    def support_multi_segments_with_cache(self) -> bool:
        return self.causal_attn_backend != 'flash'

    @staticmethod
    def get_segment_arg(segment_idx, query_length, causal_attn_backend):
        bsz, seq_len = segment_idx.shape
        qstart = seq_len - query_length

        if causal_attn_backend == 'swift':
            q_segment_idx = segment_idx[:, qstart:]
            k_segment_idx = segment_idx
            return q_segment_idx, k_segment_idx
        elif causal_attn_backend == 'xattn':
            # B x 1
            last_qidx = segment_idx[:, qstart:(qstart + 1)]
            total_seqlen_k = torch.sum(segment_idx >= last_qidx, dim=1).max().item()
            return segment_idx, total_seqlen_k
        else:
            assert causal_attn_backend is None
            # B x L1
            q_segment_idx = segment_idx[:, qstart:]
            # B x L2
            k_segment_idx = segment_idx
            # B x L1 x L2
            seg_mask = torch.ne(q_segment_idx.unsqueeze(2), k_segment_idx.unsqueeze(1))
            return seg_mask


# register some models configurations
# fmt: off

ConfStore["hybrid-gekko-0.9B"] = ModelConf(
    arch='hybrid', num_layers=28, full_attn_layer_list='7,17,27', chunk_size=4096, model_dim=1536,
    num_heads=8, num_kv_heads=4, head_dim=256, v_head_dim=256, qknorm=True,
    apply_attn_gate=True, attn_gate_func='softplus', apply_conv_in_causal_attn=True,
    rope_head_dim=128, rope_base=500000, apply_nope_in_causal_attn=True,
    causal_conv_width=4, causal_conv_weight_normalization=True, ffn_hidden_dim=4096, swiglu=True,
    layernorm_num_groups=2, timenorm_num_groups=24, apply_rmsnorm=True,
    timenorm_eps=1e-5, layernorm_eps=1e-5, rmsnorm_eps=1e-6
)

ConfStore["hybrid-gekko-4B"] = ModelConf(
    arch='hybrid', num_layers=36, full_attn_layer_list='8,17,26,35', chunk_size=4096, model_dim=2560,
    num_heads=32, num_kv_heads=8, head_dim=128, v_head_dim=128, qknorm=True,
    apply_attn_gate=True, attn_gate_func='softplus', apply_conv_in_causal_attn=True,
    rope_head_dim=128, rope_base=500000, apply_nope_in_causal_attn=True,
    causal_conv_width=4, causal_conv_weight_normalization=True, ffn_hidden_dim=9216, swiglu=True,
    layernorm_num_groups=2, timenorm_num_groups=40, apply_rmsnorm=True,
    timenorm_eps=1e-5, layernorm_eps=1e-5, rmsnorm_eps=1e-6
)

ConfStore["hybrid-gekko-7B"] = ModelConf(
    arch='hybrid', num_layers=46, chunk_size=4096, full_attn_layer_list='5,15,25,35,45', model_dim=3584,
    num_heads=40, num_kv_heads=8, head_dim=128, v_head_dim=128, qknorm=True,
    apply_attn_gate=True, attn_gate_func='softplus', apply_conv_in_causal_attn=True,
    rope_head_dim=128, rope_base=500000, apply_nope_in_causal_attn=True,
    causal_conv_width=4, causal_conv_weight_normalization=True, ffn_hidden_dim=8192, swiglu=True,
    layernorm_num_groups=4, timenorm_num_groups=48, apply_rmsnorm=True,
    timenorm_eps=1e-5, layernorm_eps=1e-5, rmsnorm_eps=1e-6
)
