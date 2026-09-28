from typing import Tuple, Optional
import math

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx
import torch.nn.functional as F
from einops import rearrange, repeat

from xllm_extension.ops import (
    group_rms_norm_fwd,
    group_rms_norm_bwd,
)
from xllm.modules.fused_ops import (
    rejection_fwd,
    rejection_bwd
)
from .utils import logsumexp_backward


class AdaptiveWorkingMemoryFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Tensor,
        qk: Tensor,
        k: Tensor,
        v: Tensor,
        chunk_size: int,
        memory: Optional[Tensor],
        log_norm_term: Optional[Tensor],
        prev_qk: Optional[Tensor],
        prev_k: Optional[Tensor],
        prev_v: Optional[Tensor],
        segment_idx: Optional[Tensor] = None,
        prev_segment_count: Optional[Tensor] = None,
        ortho: bool = False,
        eps: float = 1e-6
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        y, mask, mem, lnt = _adaptive_working_memory_fwd(
            q, qk, k, v, chunk_size, memory, log_norm_term, prev_qk, prev_k, prev_v,
            segment_idx, prev_segment_count, ortho, eps
        )
        ctx.save_for_backward(
            q, qk, k, v, memory, log_norm_term, prev_qk, prev_k, prev_v, segment_idx, prev_segment_count
        )
        ctx.chunk_size = chunk_size
        ctx.ortho = ortho
        ctx.eps = eps
        return y, mask, mem, lnt

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: Tensor,
        _,
        mem_grad: Tensor,
        lnt_grad: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, None, Tensor, Tensor,
               Optional[Tensor], Optional[Tensor], Optional[Tensor],
               None, None, None, None]:
        (
            q, qk, k, v, memory, log_norm_term, prev_qk, prev_k, prev_v, segment_idx, prev_segment_count
         ) = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        ortho = ctx.ortho
        eps = ctx.eps
        # accumulative memory
        (
            (accum_memory, memory_residual, memory_mask),
            (log_norm_term, curr_log_norm_term, accum_log_norm_term, ratio),
            (kk, rvalue, rvalue_rstd, vvalue, avalue, out, out_mask),
            (k_fp32, kk_fp32, key_mask),
            (prev_kk, prev_k_fp32, prev_kk_fp32, prev_k_mask)
        ) = _adaptive_working_memory_accum_fwd(
            q, qk, k, v, chunk_size, memory, log_norm_term, prev_qk, prev_k, prev_v,
            segment_idx, prev_segment_count, ortho, eps, False
        )
        # backward
        (
            q_grad, qk_grad, k_grad, v_grad, mem_grad, lnt_grad,
            prev_qk_grad, prev_k_grad, prev_v_grad
        ) = _adaptive_working_memory_bwd(
            y_grad, mem_grad, lnt_grad, q, qk, kk, v, chunk_size,
            rvalue, rvalue_rstd, vvalue, avalue, k_fp32, kk_fp32, key_mask,
            accum_memory, memory_residual, memory_mask,
            log_norm_term, curr_log_norm_term, accum_log_norm_term, ratio, out_mask,
            prev_qk, prev_kk, prev_v, prev_k_fp32, prev_kk_fp32, prev_k_mask, ortho
        )

        return (q_grad, qk_grad, k_grad, v_grad, None, mem_grad, lnt_grad,
                prev_qk_grad, prev_k_grad, prev_v_grad, None, None, None, None)


adaptive_working_memory = AdaptiveWorkingMemoryFunc.apply


def _adaptive_working_memory_fwd(
    query: Tensor,  # B x H x L x S
    qkey: Tensor,  # B x K x L x S
    key: Tensor,  # B x K x L x S
    value: Tensor,  # B x K x L x V
    chunk_size: int,
    memory: Optional[Tensor],  # B x K x S x V
    log_norm_term: Optional[Tensor],  # B x K x S
    prev_qkey: Optional[Tensor],  # B x K x C x S
    prev_key: Optional[Tensor],  # B x K x C x S
    prev_value: Optional[Tensor],  # B x K x C x V
    segment_idx: Optional[Tensor] = None,  # B x L(+C)
    prev_segment_count: Optional[Tensor] = None,  # B
    ortho: bool = False,
    eps: float = 1e-6
):
    bsz, n_heads, seq_len, qk_head_dim = query.shape
    _, n_kv_heads, _, v_head_dim = value.shape
    assert seq_len % chunk_size == 0
    nc = seq_len // chunk_size
    n_rep = n_heads // n_kv_heads
    if memory is None:
        assert log_norm_term is None
        assert prev_key is None and prev_value is None
        assert segment_idx is None or segment_idx.shape[1] == seq_len
        assert prev_segment_count is None

    if segment_idx is not None and prev_key is not None:
        assert segment_idx.shape[1] == seq_len + chunk_size
        prev_segment_idx = segment_idx[:, :chunk_size]
        segment_idx = segment_idx[:, chunk_size:]
    else:
        prev_segment_idx = None

    # B x N x C
    out_mask = torch.zeros(bsz, nc, chunk_size, dtype=torch.bool, device=query.device)
    if segment_idx is not None:
        segment_idx = rearrange(segment_idx, 'b (n c) -> b n c', n=nc)
        # B x 1 x 1 x L
        key_mask = rearrange(torch.ne(segment_idx, segment_idx[:, :, -1:]), 'b n c -> b 1 1 (n c)')
        prev_k_mask = None
        # B x N
        memory_mask = torch.zeros(bsz, nc, dtype=torch.bool, device=query.device)
        torch.ne(segment_idx[:, :-2, -1], segment_idx[:, 1:-1, -1], out=memory_mask[:, 2:])
        torch.ne(segment_idx[:, 2:], segment_idx[:, :-2, -1:], out=out_mask[:, 2:])
        if prev_segment_idx is not None:
            # B x 1 x 1 x C
            prev_k_mask = rearrange(torch.ne(prev_segment_idx, prev_segment_idx[:, -1:]), 'b c -> b 1 1 c')
            torch.ne(
                torch.stack([prev_segment_count, prev_segment_idx[:, -1]], dim=1),
                torch.stack([prev_segment_idx[:, -1], segment_idx[:, 0, -1]], dim=1),
                out=memory_mask[:, :2]
            )
            torch.ne(
                segment_idx[:, :2],
                torch.stack([prev_segment_count.view(bsz, 1), prev_segment_idx[:, -1:]], dim=1),
                out=out_mask[:, :2]
            )
    else:
        key_mask = None
        prev_k_mask = None
        memory_mask = None

    # B x K x L x S -> B x K x S x L
    k_fp32 = rearrange(key, 'b k l d -> b k d l').float().contiguous()
    if key_mask is not None:
        k_fp32.masked_fill_(key_mask, value=float("-inf"))
    # B x K x S x N x C
    k_fp32 = rearrange(k_fp32, 'b k d (n c) -> b k d n c', n=nc)
    # B x K x S x L
    kkey = rearrange(F.softmax(k_fp32, dim=-1).to(key), 'b k d n c -> b k d (n c)')
    # B x K x S x N
    curr_log_norm_term = torch.empty(bsz, n_kv_heads, qk_head_dim, nc, dtype=torch.float32, device=key.device)
    torch.logsumexp(k_fp32[:, :, :, :-1], dim=-1, out=curr_log_norm_term[:, :, :, 1:])

    if prev_key is not None:
        assert prev_qkey is not None
        # B x K x S x C
        prev_k_fp32 = rearrange(prev_key, 'b k c d -> b k d c').float().contiguous()
        if prev_k_mask is not None:
            prev_k_fp32.masked_fill_(prev_k_mask, value=float("-inf"))
        prev_kkey = F.softmax(prev_k_fp32, dim=-1).to(key)
        torch.logsumexp(prev_k_fp32, dim=-1, out=curr_log_norm_term[:, :, :, 0])
    else:
        assert prev_qkey is None
        prev_kkey = None
        curr_log_norm_term[:, :, :, 0] = float("-inf")

    # B x H x L x V
    out = torch.empty(bsz, n_heads, seq_len, v_head_dim, dtype=value.dtype, device=value.device)
    prev_qk = prev_qkey
    prev_kk = prev_kkey
    prev_v = prev_value
    for c in range(nc):
        bos = c * chunk_size
        eos = (c + 1) * chunk_size
        # B x H x C x S
        q = query[:, :, bos:eos]
        if memory is None:
            out[:, :, bos:eos] = 0
            out_mask[:, c] = True
            memory = torch.zeros(bsz, n_kv_heads, qk_head_dim, v_head_dim, dtype=q.dtype, device=q.device)
            log_norm_term = torch.full((bsz, n_kv_heads, qk_head_dim), float("-inf"), device=q.device)
        else:
            # (B x H x C x S) x (B x H x S, V) -> B x H x C x V
            torch.matmul(q, repeat(memory, 'b k s v -> b (k n) s v', n=n_rep), out=out[:, :, bos:eos])
            out_mask[:, c].masked_fill_(torch.isinf(log_norm_term[:, 0:1, 0]), value=True)
            if memory_mask is not None:
                mmask = rearrange(memory_mask[:, c], 'b -> b 1 1')
                log_norm_term = log_norm_term.masked_fill(mmask, value=float("-inf"))
                memory = memory.masked_fill(mmask.unsqueeze(1), value=0.)

            # B x K x S
            log_norm_term_curr = curr_log_norm_term[:, :, :, c]
            log_norm_term = torch.logaddexp(log_norm_term_curr, log_norm_term)
            # B x K x S x 1
            ratio = rearrange(torch.exp(log_norm_term_curr - log_norm_term).to(memory), 'b k d -> b k d 1')
            # (B*K) x C x V
            prev_v = rearrange(prev_v, 'b k c v -> (b k) c v')
            prev_qk = rearrange(prev_qk, 'b k c s -> (b k) c s')
            prev_kk = rearrange(prev_kk, 'b k s c -> (b k) s c')

            # (B*K x C x S) x (B*K x S x V) -> B*K x C x V
            if ortho:
                rv = torch.bmm(prev_qk, rearrange(memory, 'b k s v -> (b k) s v'))
                rv, _ = group_rms_norm_fwd(rv, v_head_dim, 1, eps)
                vv, _ = rejection_fwd(prev_v, rv)
            else:
                vv = torch.baddbmm(prev_v, prev_qk, rearrange(memory, 'b k s v -> (b k) s v'), alpha=-1.0)

            # (B*K x S x C) x (B*K x C x V) -> B*K x S x V
            memory_update = rearrange(torch.bmm(prev_kk, vv), '(b k) s v -> b k s v', b=bsz)
            memory = torch.addcmul(memory, memory_update - memory, ratio)

        prev_v = value[:, :, bos:eos]
        prev_qk = qkey[:, :, bos:eos]
        prev_kk = kkey[:, :, :, bos:eos]

    # B x H x L x V
    out_mask = rearrange(out_mask, 'b n c -> b (n c)')
    out.masked_fill_(rearrange(out_mask, 'b l -> b 1 l 1'), 0.)

    return out, out_mask, memory, log_norm_term


def _adaptive_working_memory_accum_fwd(
    query: Tensor,  # B x H x L x S
    qkey: Tensor,  # B x K x L x S
    key: Tensor,  # B x K x L x S
    value: Tensor,  # B x K x L x V
    chunk_size: int,
    memory: Optional[Tensor],  # B x K x S x V
    log_norm_term: Optional[Tensor],  # B x K x S
    prev_qkey: Optional[Tensor],  # B x K x C x S
    prev_key: Optional[Tensor],  # B x K x C x S
    prev_value: Optional[Tensor],  # B x K x C x V
    segment_idx: Optional[Tensor] = None,  # B x L(+C)
    prev_segment_count: Optional[Tensor] = None,  # B
    ortho: bool = False,
    eps: float = 1e-6,
    recompute_out: bool = False
):
    bsz, n_heads, seq_len, qk_head_dim = query.shape
    _, n_kv_heads, _, v_head_dim = value.shape
    assert seq_len % chunk_size == 0
    nc = seq_len // chunk_size
    n_rep = n_heads // n_kv_heads

    if segment_idx is not None and prev_key is not None:
        prev_segment_idx = segment_idx[:, :chunk_size]
        segment_idx = segment_idx[:, chunk_size:]
    else:
        prev_segment_idx = None

    # B x N x C
    out_mask = torch.zeros(bsz, nc, chunk_size, dtype=torch.bool, device=query.device)
    if segment_idx is not None:
        segment_idx = rearrange(segment_idx, 'b (n c) -> b n c', n=nc)
        # B x 1 x 1 x L
        key_mask = rearrange(torch.ne(segment_idx, segment_idx[:, :, -1:]), 'b n c -> b 1 1 (n c)')
        prev_k_mask = None
        # B x N
        memory_mask = torch.zeros(bsz, nc, dtype=torch.bool, device=query.device)
        torch.ne(segment_idx[:, :-2, -1], segment_idx[:, 1:-1, -1], out=memory_mask[:, 2:])
        torch.ne(segment_idx[:, 2:], segment_idx[:, :-2, -1:], out=out_mask[:, 2:])
        if prev_segment_idx is not None:
            # B x 1 x 1 x C
            prev_k_mask = rearrange(torch.ne(prev_segment_idx, prev_segment_idx[:, -1:]), 'b c -> b 1 1 c')
            torch.ne(
                torch.stack([prev_segment_count, prev_segment_idx[:, -1]], dim=1),
                torch.stack([prev_segment_idx[:, -1], segment_idx[:, 0, -1]], dim=1),
                out=memory_mask[:, :2]
            )
            torch.ne(
                segment_idx[:, :2],
                torch.stack([prev_segment_count.view(bsz, 1), prev_segment_idx[:, -1:]], dim=1),
                out=out_mask[:, :2]
            )
    else:
        key_mask = None
        prev_k_mask = None
        memory_mask = None

    # B x K x L x S -> B x K x S x L
    k_fp32 = rearrange(key, 'b k l d -> b k d l').float().contiguous()
    if key_mask is not None:
        k_fp32.masked_fill_(key_mask, value=float("-inf"))
        key_mask = rearrange(key_mask, 'b 1 1 l -> b 1 l 1')
    # B x K x S x N x C
    k_fp32 = rearrange(k_fp32, 'b k d (n c) -> b k d n c', n=nc)
    kk_fp32 = F.softmax(k_fp32, dim=-1)
    # B x K x S x L
    kkey = rearrange(F.softmax(k_fp32, dim=-1).to(key), 'b k d n c -> b k d (n c)')
    # B x K x S x N
    curr_log_norm_term = torch.empty(bsz, n_kv_heads, qk_head_dim, nc, dtype=torch.float32, device=key.device)
    torch.logsumexp(k_fp32[:, :, :, :-1], dim=4, out=curr_log_norm_term[:, :, :, 1:])
    accum_log_norm_term = torch.empty(bsz, n_kv_heads, qk_head_dim, nc, dtype=torch.float32, device=key.device)
    ratio = torch.empty(nc, bsz, n_kv_heads, qk_head_dim, 1, dtype=key.dtype, device=key.device)

    if prev_key is not None:
        assert prev_qkey is not None
        # B x K x S x C
        prev_k_fp32 = rearrange(prev_key, 'b k c d -> b k d c').float().contiguous()
        if prev_k_mask is not None:
            prev_k_fp32.masked_fill_(prev_k_mask, value=float("-inf"))
            prev_k_mask = rearrange(prev_k_mask, 'b 1 1 c -> b 1 c 1')
        prev_kk_fp32 = F.softmax(prev_k_fp32, dim=-1)
        prev_kkey = prev_kk_fp32.to(key)
        torch.logsumexp(prev_k_fp32, dim=3, out=curr_log_norm_term[:, :, :, 0])
    else:
        assert prev_qkey is None
        prev_kkey = None
        prev_k_fp32 = None
        prev_kk_fp32 = None
        curr_log_norm_term[:, :, :, 0] = float("-inf")

    accum_memory = torch.zeros(nc, bsz, n_kv_heads, qk_head_dim, v_head_dim, dtype=key.dtype, device=key.device)
    memory_residual = torch.zeros_like(accum_memory)
    # B x H x L x V
    out = torch.empty(bsz, n_heads, seq_len, v_head_dim, dtype=value.dtype, device=value.device) if recompute_out else None
    # B*K x L x V
    vvalue = torch.empty(bsz * n_kv_heads, seq_len, v_head_dim, dtype=value.dtype, device=value.device)
    rvalue = torch.empty(bsz * n_kv_heads, seq_len, v_head_dim, dtype=value.dtype, device=value.device) if ortho else None
    rvalue_rstd = torch.empty(nc, bsz * n_kv_heads * chunk_size, 1, dtype=torch.float32, device=value.device) if ortho else None
    # B*K x L x 1
    avalue = torch.empty(bsz * n_kv_heads, seq_len, 1, dtype=value.dtype, device=value.device) if ortho else None

    # B x K x L x S -> B*K x L x S
    qkey = rearrange(qkey, 'b k l s -> (b k) l s')
    kkey = rearrange(kkey, 'b k s l -> (b k) s l')
    value = rearrange(value, 'b k l v -> (b k) l v')
    if prev_qkey is not None:
        prev_qkey = rearrange(prev_qkey, 'b k c s -> (b k) c s')
        prev_kkey = rearrange(prev_kkey, 'b k s c -> (b k) s c')
        prev_value = rearrange(prev_value, 'b k c v -> (b k) c v')
    prev_qk = prev_qkey
    prev_kk = prev_kkey
    prev_v = prev_value

    for c in range(nc):
        bos = c * chunk_size
        eos = (c + 1) * chunk_size
        # B x H x C x S
        q = query[:, :, bos:eos]
        if memory is None:
            if recompute_out:
                out[:, :, bos:eos] = 0
            out_mask[:, c] = True
            memory = torch.zeros(bsz, n_kv_heads, qk_head_dim, v_head_dim, dtype=q.dtype, device=q.device)
            log_norm_term = torch.full((bsz, n_kv_heads, qk_head_dim), float("-inf"), device=q.device)
            accum_log_norm_term[:, :, :, c] = log_norm_term
        else:
            if recompute_out:
                torch.matmul(q, repeat(memory, 'b k s v -> b (k n) s v', n=n_rep), out=out[:, :, bos:eos])
            out_mask[:, c].masked_fill_(torch.isinf(log_norm_term[:, 0:1, 0]), value=True)
            accum_memory[c] = memory
            accum_log_norm_term[:, :, :, c] = log_norm_term
            if memory_mask is not None:
                mmask = rearrange(memory_mask[:, c], 'b -> b 1 1')
                log_norm_term = log_norm_term.masked_fill(mmask, value=float("-inf"))
                memory = memory.masked_fill(mmask.unsqueeze(1), value=0.)

            # B x K x S
            log_norm_term_curr = curr_log_norm_term[:, :, :, c]
            log_norm_term = torch.logaddexp(log_norm_term_curr, log_norm_term)
            # B x K x S x 1
            ratio[c] = rearrange(torch.exp(log_norm_term_curr - log_norm_term).to(memory), 'b k d -> b k d 1')

            if ortho:
                # (B*K x C x S) x (B*K x S x V) -> B*K x C x V
                rv = torch.bmm(prev_qk, rearrange(memory, 'b k s v -> (b k) s v'))
                rv, rv_rstd = group_rms_norm_fwd(rv, v_head_dim, 1, eps)
                vv, avv = rejection_fwd(prev_v, rv)
                rvalue[:, bos:eos] = rv
                # B*K*C x 1
                rvalue_rstd[c] = rv_rstd
                # B*K x C x 1
                avalue[:, bos:eos] = avv
            else:
                vv = torch.baddbmm(prev_v, prev_qk, rearrange(memory, 'b k s v -> (b k) s v'), alpha=-1.0)

            # B*K x C x V
            vvalue[:, bos:eos] = vv
            # (B*K x S x C) x (B*K x C x V) -> B*K x S x V
            memory_update = rearrange(torch.bmm(prev_kk, vv), '(b k) s v -> b k s v', b=bsz)
            memory_residual[c] = memory_update - memory
            memory = torch.addcmul(memory, memory_residual[c], ratio[c])

        prev_v = value[:, bos:eos]
        prev_qk = qkey[:, bos:eos]
        prev_kk = kkey[:, :, bos:eos]

    # B x H x L x V
    out_mask = rearrange(out_mask, 'b n c -> b 1 (n c) 1')
    if recompute_out:
        out.masked_fill_(out_mask, 0.)

    return ((accum_memory, memory_residual, memory_mask),
            (log_norm_term, curr_log_norm_term, accum_log_norm_term, ratio),
            (kkey, rvalue, rvalue_rstd, vvalue, avalue, out, out_mask), (k_fp32, kk_fp32, key_mask),
            (prev_kkey, prev_k_fp32, prev_kk_fp32, prev_k_mask))


def _adaptive_working_memory_bwd(
    out_grad: Tensor,  # B x H x L x V
    memory_grad: Tensor,  # B x K x S x V
    log_norm_term_grad: Tensor,  # B x K x S
    query: Tensor,  # B x H x L x S
    qkey: Tensor,  # B x K x L x S
    kkey: Tensor,  # B*K x S x L
    value: Tensor,  # B x K x L x V
    chunk_size: int,
    rvalue: Tensor,  # B*K x L x V
    rvalue_rstd: Tensor,  # N x B*K*C x 1
    vvalue: Tensor,  # B*K x L x V
    avalue: Tensor,  # B*K x L x 1
    k_fp32: Tensor,
    kk_fp32: Tensor,
    key_mask: Tensor,
    accum_memory: Tensor,
    memory_residual: Tensor,
    memory_mask: Tensor,
    log_norm_term: Optional[Tensor],
    curr_log_norm_term: Tensor,
    accum_log_norm_term: Tensor,
    ratio: Tensor,
    out_mask: Tensor,
    prev_qkey: Optional[Tensor],  # B x K x C x S
    prev_kkey: Optional[Tensor],  # B*K x S x C
    prev_value: Optional[Tensor],  # B x K x C x V
    prev_k_fp32: Optional[Tensor],
    prev_kk_fp32: Optional[Tensor],
    prev_k_mask: Optional[Tensor],
    ortho: bool
):
    bsz, n_heads, seq_len, qk_head_dim = query.shape
    n_kv_heads = qkey.shape[1]
    v_head_dim = vvalue.shape[2]
    assert seq_len % chunk_size == 0
    nc = seq_len // chunk_size
    n_rep = n_heads // n_kv_heads

    query_grad = torch.empty(bsz, n_heads, seq_len, qk_head_dim, dtype=query.dtype, device=query.device)
    qkey_grad = torch.zeros_like(qkey)
    key_grad = torch.zeros_like(qkey)
    value_grad = torch.zeros(bsz, n_kv_heads, seq_len, v_head_dim, dtype=out_grad.dtype, device=out_grad.device)
    prev_qkey_grad = None
    prev_key_grad = None
    prev_value_grad = None
    curr_log_norm_term_grad = torch.empty_like(curr_log_norm_term)

    out_grad = rearrange(out_grad.masked_fill(out_mask, 0.), 'b h l v -> (b h) l v')
    query = rearrange(query, 'b h l d -> (b h) d l')
    qkey = rearrange(qkey, 'b k l d -> (b k) d l')
    value = rearrange(value, 'b k l v -> (b k) l v')
    kkey = rearrange(kkey, 'bk d l -> bk l d')
    vvalue = rearrange(vvalue, 'bk l d -> bk d l')
    if prev_qkey is not None:
        prev_qkey = rearrange(prev_qkey, 'b h c d -> (b h) d c')
        prev_kkey = rearrange(prev_kkey, 'bh d c -> bh c d')
        prev_value = rearrange(prev_value, 'b k c v -> (b k) c v')

    tmp_lnt_grad = torch.empty(2, bsz, n_kv_heads, qk_head_dim, dtype=torch.float32, device=query.device)
    for c in reversed(range(nc)):
        pbos = (c - 1) * chunk_size
        bos = c * chunk_size
        eos = (c + 1) * chunk_size
        # B*H x S x C
        q = query[:, :, bos:eos]
        # B*H x C x V
        og = out_grad[:, bos:eos]
        # B*K x S x C
        prev_qk = qkey[:, :, pbos:bos] if c > 0 else prev_qkey
        # B*K x C x S
        prev_kk = kkey[:, pbos:bos] if c > 0 else prev_kkey
        # B*K x C x V
        prev_v = value[:, pbos:bos] if c > 0 else prev_value

        # B*K x V x C
        vv = vvalue[:, :, bos:eos]
        # B*K x C x V
        rv = rvalue[:, bos:eos] if ortho else None
        # B*K*C x 1
        rv_rstd = rvalue_rstd[c] if ortho else None
        # B*K x C x 1
        avv = avalue[:, bos:eos] if ortho else None

        if prev_qk is None:
            query_grad[:, :, bos:eos] = 0
            memory_grad = None
            log_norm_term_grad = None
        elif memory_grad is None:
            assert log_norm_term_grad is None
            log_norm_term = accum_log_norm_term[:, :, :, c]
            curr_log_norm_term_grad[:, :, :, c] = 0.
            log_norm_term_grad = torch.zeros_like(log_norm_term)
            # B x K x V x S
            memory = rearrange(accum_memory[c], 'b k s v -> b k v s')
            memory = repeat(memory, 'b k v s -> b (k n) v s', n=n_rep)
            memory = rearrange(memory, 'b h v s -> (b h) v s')
            query_grad[:, :, bos:eos] = rearrange(torch.bmm(og, memory), '(b h) c s -> b h c s', b=bsz)
            # B x K x S x V
            memory_grad = rearrange(torch.bmm(q, og), '(b k n) s v -> b k n s v', b=bsz, k=n_kv_heads).sum(dim=2)
        else:
            log_norm_term_prev = accum_log_norm_term[:, :, :, c]
            curr_memory = rearrange(accum_memory[c], 'b k s v -> b k v s')
            memory = repeat(curr_memory, 'b k v s -> b (k n) v s', n=n_rep)
            memory = rearrange(memory, 'b h v s -> (b h) v s')
            query_grad[:, :, bos:eos] = rearrange(torch.bmm(og, memory), '(b h) c s -> b h c s', b=bsz)

            if memory_mask is not None:
                mmask = rearrange(memory_mask[:, c], 'b -> b 1 1')
                log_norm_term_prev = log_norm_term_prev.masked_fill(mmask, value=float("-inf"))
                curr_memory.masked_fill_(mmask.unsqueeze(1), value=0.)
            else:
                mmask = None

            # B x K x S
            ratio_grad = (memory_grad * memory_residual[c]).sum(dim=3).float()
            log_norm_term_curr = curr_log_norm_term[:, :, :, c]
            lnt_curr_grad = ratio_grad * ratio[c].squeeze(3)
            log_norm_term_grad = log_norm_term_grad - ratio_grad * ratio[c].squeeze(3)
            tmp_lnt_grad = logsumexp_backward(
                log_norm_term_grad, torch.stack([log_norm_term_curr, log_norm_term_prev], dim=0),
                log_norm_term, dim=0, grad_out=tmp_lnt_grad
            )
            curr_log_norm_term_grad[:, :, :, c] = tmp_lnt_grad[0] + lnt_curr_grad
            log_norm_term_grad = tmp_lnt_grad[1]

            # B x K x S x V -> B*K x S x V
            memory_update_grad = rearrange(ratio[c] * memory_grad, 'b k s v -> (b k) s v')
            memory_grad = rearrange((1.0 - ratio[c]) * memory_grad, 'b k s v -> (b k) s v')
            curr_memory = rearrange(curr_memory, 'b k v s -> (b k) v s')
            # B*K x C x V
            vv_grad = torch.bmm(prev_kk, memory_update_grad)
            if ortho:
                v_grad, rv_grad = rejection_bwd(vv_grad, prev_v, rv, avv)
                rv_grad = group_rms_norm_bwd(rv_grad, rv, v_head_dim, 1, rv_rstd, True)
            else:
                v_grad = vv_grad
                rv_grad = -vv_grad

            memory_grad = torch.baddbmm(memory_grad, prev_qk, rv_grad, out=memory_grad)
            memory_grad = rearrange(memory_grad, '(b k) s v -> b k s v', b=bsz)
            if mmask is not None:
                log_norm_term_grad.masked_fill_(mmask, value=0.)
                memory_grad.masked_fill_(mmask.unsqueeze(1), value=0.)

            log_norm_term = accum_log_norm_term[:, :, :, c]
            # B x K x S x V
            mem_grad_from_out = rearrange(torch.bmm(q, og), '(b k n) s v -> b k n s v', b=bsz, k=n_kv_heads).sum(dim=2)
            memory_grad = torch.add(memory_grad, mem_grad_from_out, out=memory_grad)

            prev_kk_grad = rearrange(torch.bmm(memory_update_grad, vv), '(b k) s c -> b k s c', b=bsz).float()
            prev_qk_grad = rearrange(torch.bmm(rv_grad, curr_memory), '(b k) c s -> b k c s', b=bsz)
            v_grad = rearrange(v_grad, '(b h) c v -> b h c v', b=bsz)

            if c == 0:
                prev_value_grad = v_grad
                prev_key_grad = rearrange(torch.ops.aten._softmax_backward_data(
                    prev_kk_grad, prev_kk_fp32, -1, torch.float32
                ).to(qkey), 'b h s c -> b h c s')
                prev_qkey_grad = prev_qk_grad
            else:
                value_grad[:, :, pbos:bos] = v_grad
                key_grad[:, :, pbos:bos] = rearrange(torch.ops.aten._softmax_backward_data(
                    prev_kk_grad, kk_fp32[:, :, :, c - 1], -1, torch.float32
                ).to(qkey), 'b h s c -> b h c s')
                qkey_grad[:, :, pbos:bos] = prev_qk_grad

    # B x H x S/H x (N-1) x C -> B x H x S/H x (N-1)*C -> B x H x (N-1)*C x S/H
    key_grad_log_term = logsumexp_backward(
        curr_log_norm_term_grad[:, :, :, 1:], k_fp32[:, :, :, :-1], curr_log_norm_term[:, :, :, 1:], dim=4
    ).to(qkey)
    key_grad_log_term = rearrange(key_grad_log_term, 'b h s n c -> b h (n c) s')

    key_grad[:, :, :seq_len - chunk_size] += key_grad_log_term
    if key_mask is not None:
        key_grad.masked_fill_(key_mask, value=0.)

    if prev_qkey is not None:
        # B x H x S/H x C -> B x H x C x S/H
        prev_key_grad += rearrange(logsumexp_backward(
            curr_log_norm_term_grad[:, :, :, 0], prev_k_fp32, curr_log_norm_term[:, :, :, 0], dim=3
        ).to(qkey), 'b h s c -> b h c s')

        if prev_k_mask is not None:
            prev_key_grad.masked_fill_(prev_k_mask, value=0.)

    return query_grad, qkey_grad, key_grad, value_grad, memory_grad, log_norm_term_grad, prev_qkey_grad, prev_key_grad, prev_value_grad


def adaptive_working_memory_fwd(
    query: Tensor,
    qkey: Tensor,
    key: Tensor,
    value: Tensor,
    chunk_size: int,
    memory: Optional[Tensor],
    log_norm_term: Optional[Tensor],
    prev_qkey: Optional[Tensor],
    prev_key: Optional[Tensor],
    prev_value: Optional[Tensor],
    segment_idx: Optional[Tensor] = None,
    prev_segment_count: Optional[Tensor] = None,
    ortho: bool = False,
    eps: float = 1e-6
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return _adaptive_working_memory_fwd(
        query, qkey, key, value, chunk_size, memory, log_norm_term,
        prev_qkey, prev_key, prev_value,
        segment_idx, prev_segment_count, ortho, eps
    )


def adaptive_working_memory_accum_fwd(
    query: Tensor,
    qkey: Tensor,
    key: Tensor,
    value: Tensor,
    chunk_size: int,
    memory: Optional[Tensor],
    log_norm_term: Optional[Tensor],
    prev_qkey: Optional[Tensor],
    prev_key: Optional[Tensor],
    prev_value: Optional[Tensor],
    segment_idx: Optional[Tensor] = None,
    prev_segment_count: Optional[Tensor] = None,
    ortho: bool = False,
    eps: float = 1e-6,
    recompute_out: bool = False,
) -> Tuple[Tuple[Tensor, Tensor, Tensor],
           Tuple[Tensor, Tensor, Tensor, Tensor],
           Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
           Tuple[Tensor, Tensor, Tensor],
           Tuple[Tensor, Tensor, Tensor, Tensor]]:
    return _adaptive_working_memory_accum_fwd(
        query, qkey, key, value, chunk_size, memory, log_norm_term,
        prev_qkey, prev_key, prev_value,
        segment_idx, prev_segment_count, ortho, eps, recompute_out
    )


def adaptive_working_memory_bwd(
    out_grad: Tensor,
    memory_grad: Tensor,
    log_norm_term_grad: Tensor,
    query: Tensor,
    qkey: Tensor,
    kkey: Tensor,
    value: Tensor,
    chunk_size: int,
    rvalue: Tensor,
    rvalue_rstd: Tensor,
    vvalue: Tensor,
    avalue: Tensor,
    k_fp32: Tensor,
    kk_fp32: Tensor,
    key_mask: Tensor,
    accum_memory: Tensor,
    memory_residual: Tensor,
    memory_mask: Tensor,
    log_norm_term: Optional[Tensor],
    curr_log_norm_term: Tensor,
    accum_log_norm_term: Tensor,
    ratio: Tensor,
    out_mask: Tensor,
    prev_qkey: Optional[Tensor],
    prev_kkey: Optional[Tensor],
    prev_value: Optional[Tensor],
    prev_k_fp32: Optional[Tensor],
    prev_kk_fp32: Optional[Tensor],
    prev_k_mask: Optional[Tensor],
    ortho: bool
) -> Tuple[Tensor, Tensor, Tensor, Tensor,
           Optional[Tensor], Optional[Tensor], Optional[Tensor],
           Optional[Tensor], Optional[Tensor]]:
    return _adaptive_working_memory_bwd(
        out_grad, memory_grad, log_norm_term_grad, query, qkey, kkey, value, chunk_size,
        rvalue, rvalue_rstd, vvalue, avalue, k_fp32, kk_fp32, key_mask,
        accum_memory, memory_residual, memory_mask,
        log_norm_term, curr_log_norm_term, accum_log_norm_term, ratio, out_mask,
        prev_qkey, prev_kkey, prev_value, prev_k_fp32, prev_kk_fp32, prev_k_mask, ortho
    )
