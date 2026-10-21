import math
from typing import Optional
import torch
from torch import Tensor, nn
from einops import rearrange

from .rotary_positional_embedding import apply_rotary_embeddings
from .fused_ops import (
    sliding_chunk_attention,
    swift_efficient_attention,
)
from .causal_attention import repeat_kv


class SlidingChunkAttention(nn.Module):
    """
    Sliding Chunk attention
    """

    def __init__(
        self,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        v_head_dim: int,
        rope_head_dim: int,
        chunk_size: int,
        scale: Optional[float],
        dropout: float,
        backend: Optional[str],
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.rope_head_dim = rope_head_dim
        self.chunk_size = chunk_size
        self.scale = 1.0 / math.sqrt(self.head_dim) if scale is None else scale
        self.dropout = dropout
        self.backend = backend

    def forward(
        self,
        xq: Tensor,
        xk: Tensor,
        xv: Tensor,
        freqs_cis: Optional[Tensor],
        prev_k: Optional[Tensor] = None,
        prev_v: Optional[Tensor] = None,
        bos_mask: Optional[Tensor] = None,
        segment_idx: Optional[Tensor] = None,
        stability_control: int = 0,
        deterministic: bool = True,
    ):
        bs, slen, _ = xq.shape
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

        if slen >= self.chunk_size:
            assert slen % self.chunk_size == 0
            output = sliding_chunk_attention(
                xq, xk, xv, self.chunk_size, self.scale, prev_k, prev_v, bos_mask,
                segment_idx, self.dropout, stability_control, deterministic, self.backend, self.training
            )
            prev_k = xk[:, (slen - self.chunk_size):]
            prev_v = xv[:, (slen - self.chunk_size):]
        else:
            # tail or step-by-step generation mode
            xk = torch.cat([prev_k, xk], dim=1) if prev_k is not None else xk
            xv = torch.cat([prev_v, xv], dim=1) if prev_v is not None else xv
            assert xk.shape[1] <= self.chunk_size * 2
            n_rep = self.n_heads // self.n_kv_heads
            xk_ = repeat_kv(xk, n_rep) if n_rep > 1 else xk
            xv_ = repeat_kv(xv, n_rep) if n_rep > 1 else xv

            if segment_idx is not None:
                q_segment_idx = segment_idx if prev_k is None else segment_idx[:, self.chunk_size:]
                k_segment_idx = segment_idx
            else:
                q_segment_idx = None
                k_segment_idx = None

            use_causal_mask = slen > 1
            output = swift_efficient_attention(
                xq, xk_, xv_, q_segment_idx, k_segment_idx, self.scale, self.dropout, use_causal_mask, self.training
            )
            prev_k = xk
            prev_v = xv

        output = rearrange(output, 'b l h d -> b l (h d)')
        return output, prev_k, prev_v

    def extra_repr(self) -> str:
        return 'heads={} ({}), head_dim={} ({}), v_head_dim={}, chunk={}, scale={}, backend={}'.format(
            self.n_heads, self.n_kv_heads, self.head_dim, self.rope_head_dim, self.v_head_dim,
            self.chunk_size, self.scale, self.backend
        )
