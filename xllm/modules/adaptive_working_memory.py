from typing import Optional
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .fused_ops import adaptive_working_memory, rejection
from .rms_norm import group_rms_norm


class AdaptiveWorkingMemory(nn.Module):
    """
    Adaptive Working Memory
    """
    def __init__(
        self,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        v_head_dim: int,
        chunk_size: int,
        orthogonal_update: bool,
        rmsnorm_eps: float,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.chunk_size = chunk_size
        self.ortho = orthogonal_update
        self.eps = rmsnorm_eps

    def forward(
        self,
        xq: Tensor,  # B x L x (H*S)
        xqk: Tensor,  # B x L x (K*S)
        xkk: Tensor,  # B x L x (K*S)
        xv: Tensor,  # B x L x (K*V)
        memory: Optional[Tensor],  # B x K x S x V
        log_norm_term: Optional[Tensor],  # B x K x S
        prev_qk: Optional[Tensor],  # B x K x C x S
        prev_kk: Optional[Tensor],  # B x K x C x S
        prev_v: Optional[Tensor],  # B x K x C x V
        segment_idx: Optional[Tensor] = None,  # B x L(+C)
        prev_segment_count: Optional[Tensor] = None,  # B
    ):
        bsz, seq_len, _ = xq.shape

        # apply softmax to xq & xqk
        # B x L x H*S -> B x H x L x S
        xq = rearrange(xq, 'b l (h s) -> b h l s', h=self.n_heads)
        xq = F.softmax(xq, dim=-1, dtype=torch.float32).to(xq)
        # B x L x K*S|V -> B x K x L x S|V
        xqk = rearrange(xqk, 'b l (k s) -> b k l s', k=self.n_kv_heads)
        xqk = F.softmax(xqk, dim=-1, dtype=torch.float32).to(xqk)
        xkk = rearrange(xkk, 'b l (k s) -> b k l s', k=self.n_kv_heads)
        xv = rearrange(xv, 'b l (k v) -> b k l v', k=self.n_kv_heads)

        if seq_len >= self.chunk_size:
            out, mask, memory, log_norm_term = adaptive_working_memory(
                xq, xqk, xkk, xv, self.chunk_size, memory, log_norm_term, prev_qk, prev_kk, prev_v,
                segment_idx, prev_segment_count, self.ortho, self.eps
            )
            prev_qk = xqk[:, :, seq_len - self.chunk_size:]
            prev_kk = xkk[:, :, seq_len - self.chunk_size:]
            prev_v = xv[:, :, seq_len - self.chunk_size:]
        else:
            # B x L
            mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=xq.device)
            # tail or step-by-step generation mode
            xqk = torch.cat([prev_qk, xqk], dim=2) if prev_qk is not None else xqk
            xkk = torch.cat([prev_kk, xkk], dim=2) if prev_kk is not None else xkk
            xv = torch.cat([prev_v, xv], dim=2) if prev_v is not None else xv
            assert xkk.shape[2] == xqk.shape[2]

            if segment_idx is not None and prev_v is not None:
                assert segment_idx.shape[1] == seq_len + self.chunk_size
                prev_segment_idx = segment_idx[:, :self.chunk_size]
                segment_idx = segment_idx[:, self.chunk_size:]
            else:
                prev_segment_idx = None

            if memory is None:
                assert xqk.shape[2] <= self.chunk_size
                out = torch.zeros(bsz, self.n_heads, seq_len, self.v_head_dim, dtype=xq.dtype, device=xq.device)
                mask[:] = True
            else:
                assert self.chunk_size < xqk.shape[2] <= self.chunk_size * 2
                n_rep = self.n_heads // self.n_kv_heads
                # B x K x S x V -> B x H x S x V
                memory_ = repeat(memory, 'b h s v -> b (h n) s v', n=n_rep)
                # (B x H x L x S) x (B x H x S x V) -> B x H x L x V
                out = torch.matmul(xq, memory_)
                if segment_idx is not None:
                    mask = torch.ne(segment_idx, prev_segment_count.view(bsz, 1), out=mask)
                else:
                    mask[:] = torch.isinf(log_norm_term[:, 0:1, 0])
                out = torch.masked_fill(out, mask.view(bsz, 1, seq_len, 1), value=0.)
            # update memory
            if xqk.shape[2] % self.chunk_size == 0:
                memory, log_norm_term = self.update_memory(
                    xqk, xkk, xv, memory, log_norm_term, segment_idx, prev_segment_idx, prev_segment_count
                )
            prev_qk = xqk
            prev_kk = xkk
            prev_v = xv

        out = rearrange(out, 'b h l v -> b l (h v)')
        return out, mask, memory, log_norm_term, prev_qk, prev_kk, prev_v

    def update_memory(
        self, qkey, kkey, value, memory, log_norm_term,
        segment_idx, prev_segment_idx, prev_segment_count
    ):
        bsz = qkey.shape[0]
        if memory is None:
            assert qkey.shape[2] == self.chunk_size
            memory = torch.zeros(bsz, self.n_kv_heads, self.head_dim, self.v_head_dim, dtype=qkey.dtype, device=qkey.device)
            log_norm_term = torch.full((bsz, self.n_kv_heads, self.head_dim), float("-inf"), device=qkey.device)
        else:
            assert qkey.shape[2] == self.chunk_size * 2
            prev_v = value[:, :, :self.chunk_size]
            prev_qk = qkey[:, :, :self.chunk_size]
            prev_kk = kkey[:, :, :self.chunk_size]
            prev_kk_fp32 = prev_kk.float()
            # B x H x S x C
            prev_kk_fp32 = prev_kk_fp32.transpose(2, 3).contiguous()
            if segment_idx is not None:
                prev_segment_idx = torch.cat([prev_segment_idx, segment_idx], dim=1)[:, :self.chunk_size]
                prev_k_mask = torch.ne(prev_segment_idx, prev_segment_idx[:, -1:]).view(bsz, 1, 1, self.chunk_size)
                prev_kk_fp32.masked_fill_(prev_k_mask, value=float("-inf"))
                memory_mask = torch.ne(prev_segment_idx[:, -1], prev_segment_count)
                memory = memory.masked_fill(memory_mask.view(bsz, 1, 1, 1), value=0.)
                log_norm_term = log_norm_term.masked_fill(memory_mask.view(bsz, 1, 1), value=float("-inf"))
            prev_kk = F.softmax(prev_kk_fp32, dim=-1).to(kkey)
            curr_log_norm_term = torch.logsumexp(prev_kk_fp32, dim=-1)
            log_norm_term = torch.logsumexp(torch.stack([curr_log_norm_term, log_norm_term], dim=0), dim=0)
            ratio = torch.exp(curr_log_norm_term - log_norm_term).to(memory).unsqueeze(3)
            if self.ortho:
                # B x H x C x V -> B*H x C x V
                rv = rearrange(torch.matmul(prev_qk, memory), 'b h c v -> (b h) c v')
                # B*H x C x V -> B x H x C x V
                rv = rearrange(group_rms_norm(rv, None, 1, self.eps), '(b h) c v -> b h c v', b=bsz)
                vv = rejection(prev_v, rv)
            else:
                # B x H x C x V
                vv = prev_v - torch.matmul(prev_qk, memory)
            memory_update = torch.matmul(prev_kk, vv)
            memory = torch.addcmul(memory, memory_update - memory, ratio)

        return memory, log_norm_term

    def extra_repr(self) -> str:
        return 'heads={} ({}), qk_head_dim={}, v_head_dim={}, chunk={}, ortho={}, eps={}'.format(
            self.n_heads, self.n_kv_heads, self.head_dim, self.v_head_dim, self.chunk_size, self.ortho, self.eps
        )
