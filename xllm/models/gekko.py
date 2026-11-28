from typing import Optional, Tuple, List
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
    GCABlock,
    GekkoOutputLayer,
)
from xllm.utils import get_init_fn
from xllm.optim.roles import register_role


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
        embed = ParallelEmbedding(
            self.vocab_size, self.model_dim, embed_multiplier=cfg.embed_multiplier, gather_output=False, init_method=init_fn
        )
        register_role(embed, "weight", "embed")  # xllm/optim/roles.py
        self.embed = wrap(embed)
        self.causal_conv_width = cfg.causal_conv_width
        self.apply_bias_term = cfg.apply_bias_term

        if self.rope_head_dim > 0:
            self.rope = RotaryEmbedding(self.rope_head_dim, cfg.chunk_size * 16, base=cfg.rope_base)
        else:
            self.rope = None

        self.layers = nn.ModuleList()
        for layer_id in range(self.num_layers):
            layer = GCABlock(cfg, layer_id)
            self.layers.append(wrap(layer))

        self.output = wrap(GekkoOutputLayer(cfg))

    def forward(
        self,
        tokens: Tensor,
        multi_segments: bool,
        targets: Optional[Tensor] = None,
        token_mask: Optional[Tensor] = None,
        moe_router_load_balancing_type: Optional[str] = None,
        attn_stability_control: int = 0,
        deterministic: bool = True,
        cache: Optional[Tuple[List[Tuple[Tuple[Tensor, Tensor, int],
                                         Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
                                         Tuple[Tensor, Tensor, Tensor],
                                         Tuple[Tensor, Tensor, Tensor]]],
                              Tuple[Tensor, Tensor, Tensor],
                              Tuple[Tensor, Tensor, Tensor]]] = None,
    ):

        bsz, seq_len = tokens.shape

        if targets is None:
            assert token_mask is None

        if multi_segments:
            bos_mask = get_bos_mask(tokens, self.eos_id, use_eos=True)
            segment_idx = get_segment_idx_from_bos_mask(bos_mask)
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
        health_collector = get_active_health_collector()
        for i, layer in enumerate(self.layers):
            layer_cache = cache_layers[i] if cache is not None else None
            if self.layerwise_ckpt:
                x, aux_loss, layer_cache = checkpoint(
                    layer, x, freq_cis, bos_mask, segment_idx, prev_segment_count,
                    moe_router_load_balancing_type, attn_stability_control, deterministic,
                    layer_cache, use_reentrant=False, preserve_rng_state=True
                )
            else:
                x, aux_loss, layer_cache = layer(
                    x, freq_cis, bos_mask, segment_idx, prev_segment_count,
                    moe_router_load_balancing_type, attn_stability_control, deterministic, layer_cache
                )

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
                    prev_segment_idx = prev_segment_idx[:, length - self.chunk_size:]
                    prev_bos_mask = prev_bos_mask[:, length - self.chunk_size:]

                cache_segment = (prev_bos_mask, prev_segment_idx, prev_segment_count)
            cache = (cache_layers, cache_output, cache_segment)

        return logits_or_loss, aux_loss_sum, cache

    def support_multi_segments_with_cache(self) -> bool:
        return True


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

# fmt: on
