import math
from typing import Optional, Any, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange, repeat

from xllm.distributed import (
    get_context_parallel_rank,
)
from xllm.modules.context_parallel import (
    gather_copy_context_parallel_region
)
from xllm.modules.fused_ops import (
    flash_attention,
    swift_efficient_attention,
    memory_efficient_dropout,
)
from xllm.modules.rotary_positional_embedding import (
    apply_rotary_embedding,
    apply_rotary_embeddings
)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    return repeat(x, 'b l h d -> b l (h n) d', n=n_rep)


class CausalSoftmaxAttention(nn.Module):
    """
    Causal softmax attention in multi-head attention
    """

    def __init__(
        self,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rope_head_dim: int,
        scale: Optional[float],
        dropout: float,
        backend: Optional[str],
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.dropout = dropout
        self.backend = backend
        self.scale = 1.0 / math.sqrt(self.head_dim) if scale is None else scale

        if backend == 'xattn':
            raise NotImplementedError

    def forward(
        self,
        xq: Tensor,
        xk: Tensor,
        xv: Tensor,
        g: Optional[Tensor],
        freqs_cis: Optional[Tensor],
        segments: Optional[Any],
        stability_control: int = 0,
        deterministic: bool = True,
        cache: Optional[Tuple[Tensor, Tensor, int]] = None,
    ):
        bs, slen, _ = xq.shape
        # B x L x H x D/H
        xq = rearrange(xq, 'b l (h d) -> b l h d', h=self.n_heads)
        xk = rearrange(xk, 'b l (h d) -> b l h d', h=self.n_kv_heads)
        xv = rearrange(xv, 'b l (h d) -> b l h d', h=self.n_kv_heads)

        if self.rope_head_dim == self.head_dim:
            xq, xk = apply_rotary_embeddings(xq, xk, freqs_cis=freqs_cis, backward=False)
        elif self.rope_head_dim > 0:
            xq_rope, xq_nope = torch.split(xq, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
            xk_rope, xk_nope = torch.split(xk, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
            xq_rope, xk_rope = apply_rotary_embeddings(xq_rope, xk_rope, freqs_cis=freqs_cis, backward=False)
            xq = torch.cat([xq_rope, xq_nope], dim=-1)
            xk = torch.cat([xk_rope, xk_nope], dim=-1)

        if cache is not None:
            assert segments is None
            cache_k, cache_v, cache_len = cache
            xk = torch.cat([cache_k, xk], dim=1) if cache_k is not None else xk
            xv = torch.cat([cache_v, xv], dim=1) if cache_v is not None else xv

            cache_k = xk.detach()
            cache_v = xv.detach()
            new_cache = (cache_k, cache_v, cache_len + slen)
        else:
            end = (get_context_parallel_rank() + 1) * slen
            xk = gather_copy_context_parallel_region(xk, dim=1)[:, :end]
            xv = gather_copy_context_parallel_region(xv, dim=1)[:, :end]
            cache_len = end - slen
            new_cache = None

        if self.backend is None:
            n_rep = self.n_heads // self.n_kv_heads
            xk = repeat_kv(xk, n_rep)
            xv = repeat_kv(xv, n_rep)

            xq = rearrange(xq, 'b l h d -> b h l d')
            xk = rearrange(xk, 'b l h d -> b h d l')
            xv = rearrange(xv, 'b l h d -> b h l d')

            attn_mask = torch.full((bs, slen, slen + cache_len), float("-inf"), device=xq.device)
            attn_mask = torch.triu(attn_mask, diagonal=cache_len + 1).type_as(xq)
            if segments is not None:
                attn_mask = attn_mask.masked_fill(segments, value=float("-inf"))
            # B x H x L x L
            scores = torch.matmul(xq, xk) * self.scale
            scores = scores + attn_mask.unsqueeze(1)
            scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(xq)
            scores = memory_efficient_dropout(scores, self.dropout, self.training)
            # B x H x L x D -> B x L x H x D
            output = rearrange(torch.matmul(scores, xv), 'b h l d -> b l h d')
        elif self.backend == 'swift':
            n_rep = self.n_heads // self.n_kv_heads
            xk = repeat_kv(xk, n_rep)
            xv = repeat_kv(xv, n_rep)

            if segments is not None:
                q_segment_idx, k_segment_idx = segments
            else:
                q_segment_idx, k_segment_idx = None, None

            if slen == 1 and cache is not None:
                use_causal_mask = False
            else:
                use_causal_mask = True
            output = swift_efficient_attention(
                xq, xk, xv, q_segment_idx, k_segment_idx, self.scale, self.dropout, use_causal_mask, self.training
            )
        elif self.backend == 'flash':
            if segments is not None:
                cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, total_seqlen_k = segments
                xk = xk[:, -total_seqlen_k:]
                xv = xv[:, -total_seqlen_k:]
            else:
                cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = None, None, None, None

            output = flash_attention(
                xq, xk, xv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                self.scale, self.dropout, True, deterministic, self.training
            )
        else:
            raise ValueError(f"Unknown causal attention backend: {self.backend}")

        output = rearrange(output, 'b l h d -> b l (h d)')
        return output, new_cache

    def extra_repr(self) -> str:
        return 'heads={}, kv_heads={}, head_dim={}, rope_head_dim={}, backend={}'.format(
            self.n_heads, self.n_kv_heads, self.head_dim, self.rope_head_dim, self.backend
        )


class CausalSoftdeltaAttention(nn.Module):
    """
    Causal softdelta attention
    """

    def __init__(
        self,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rope_head_dim: int,
        scale: Optional[float],
        dropout: float,
        backend: Optional[str],
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.dropout = dropout
        self.backend = backend
        self.scale = 1.0 / math.sqrt(self.head_dim)  if scale is None else scale

    def forward(
        self,
        xq: Tensor,
        xk: Tensor,
        xv: Tensor,
        g: Optional[Tensor],
        freqs_cis: Optional[Tensor],
        segments: Optional[Any],
        stability_control: int = 0,
        deterministic: bool = True,
        cache: Optional[Tuple[Tensor, Tensor, int]] = None,
    ):
        bs, slen, _ = xq.shape
        xq = rearrange(xq, 'b l (h k d) -> b l h k d', h=self.n_heads, k=2)
        xq, kq = torch.unbind(xq, dim=3)
        xk = rearrange(xk, 'b l (h d) -> b l h d', h=self.n_kv_heads)
        xv = rearrange(xv, 'b l (h d) -> b l h d', h=self.n_kv_heads)

        if self.rope_head_dim == self.head_dim:
            xq, xk = apply_rotary_embeddings(xq, xk, freqs_cis=freqs_cis, backward=False)
            kq = apply_rotary_embedding(kq, freqs_cis=freqs_cis, backward=False)
        elif self.rope_head_dim > 0:
            xq_rope, xq_nope = torch.split(xq, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
            kq_rope, kq_nope = torch.split(kq, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
            xk_rope, xk_nope = torch.split(xk, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
            xq_rope, xk_rope = apply_rotary_embeddings(xq_rope, xk_rope, freqs_cis=freqs_cis, backward=False)
            kq_rope = apply_rotary_embedding(kq_rope, freqs_cis=freqs_cis, backward=False)
            xq = torch.cat([xq_rope, xq_nope], dim=-1)
            kq = torch.cat([kq_rope, kq_nope], dim=-1)
            xk = torch.cat([xk_rope, xk_nope], dim=-1)

        if cache is not None:
            assert segments is None
            cache_k, cache_v, cache_len = cache
            xk = torch.cat([cache_k, xk], dim=1) if cache_k is not None else xk
            xv = torch.cat([cache_v, xv], dim=1) if cache_v is not None else xv

            cache_k = xk.detach()
            cache_v = xv.detach()
            new_cache = (cache_k, cache_v, cache_len + slen)
        else:
            end = (get_context_parallel_rank() + 1) * slen
            xk = gather_copy_context_parallel_region(xk, dim=1)[:, :end]
            xv = gather_copy_context_parallel_region(xv, dim=1)[:, :end]
            cache_len = end - slen
            new_cache = None

        if self.backend is None:
            n_rep = self.n_heads // self.n_kv_heads
            xk = repeat_kv(xk, n_rep)
            xv = repeat_kv(xv, n_rep)

            xq = rearrange(xq, 'b l h d -> b h l d')
            kq = rearrange(kq, 'b l h d -> b h l d')
            xk = rearrange(xk, 'b l h d -> b h d l')
            xv = rearrange(xv, 'b l h d -> b h l d')
            # B x L1 x L2
            full_inf = torch.full((bs, slen, slen + cache_len), float("-inf"), device=xq.device)
            attn_mask1 = torch.triu(full_inf, diagonal=cache_len + 1).type_as(xq)
            attn_mask2 = torch.triu(full_inf, diagonal=cache_len).type_as(xq)
            if segments is not None:
                attn_mask1 = attn_mask1.masked_fill(segments, value=float("-inf"))
                attn_mask2 = attn_mask2.masked_fill(segments, value=float("-inf"))
            # B x L1 x 1
            all_inf = torch.all(attn_mask2.isinf(), dim=2, keepdim=True)
            attn_mask2 = attn_mask2.masked_fill(all_inf, value=0.0)
            # B x H x L1 x L2
            scores1 = torch.matmul(xq, xk) * self.scale
            scores2 = torch.matmul(kq, xk) * self.scale
            scores1 = scores1 + attn_mask1.unsqueeze(1)
            scores2 = scores2 + attn_mask2.unsqueeze(1)
            scores1 = F.softmax(scores1, dim=-1, dtype=torch.float32).to(xq)
            scores2 = F.softmax(scores2, dim=-1, dtype=torch.float32).to(kq)
            scores2 = scores2 * torch.logical_not(all_inf).to(kq).unsqueeze(1)
            scores1 = memory_efficient_dropout(scores1, self.dropout, self.training)
            scores2 = memory_efficient_dropout(scores2, self.dropout, self.training)
            # B x H x L x D -> B x L x H x D
            out1 = rearrange(torch.matmul(scores1, xv), 'b h l d -> b l h d')
            out2 = rearrange(torch.matmul(scores2, xv), 'b h l d -> b l h d')
            out2 = out2 * rearrange(g, 'b l h -> b l h 1')
            output = out1 - out2
        else:
            raise ValueError(f"Unknown causal attention backend: {self.backend}")

        output = rearrange(output, 'b l h d -> b l (h d)')
        return output, new_cache

    def extra_repr(self) -> str:
        return 'heads={}, kv_heads={}, head_dim={}, rope_head_dim={}, backend={}'.format(
            self.n_heads, self.n_kv_heads, self.head_dim, self.rope_head_dim, self.backend
        )
