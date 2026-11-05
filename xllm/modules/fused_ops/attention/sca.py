from typing import Tuple, Optional
import functools
import os
import weakref

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx
from einops import rearrange, repeat
from xllm_extension.ops import (
    attention_fwd,
    attention_bwd,
    multiseg_attention_fwd
)
import xllm_extension.ops as _xllm_ops
_MULTISEG_FWD_OUT = getattr(_xllm_ops, "multiseg_attention_fwd_out", None)  # older builds: copy instead
_ATTN_BWD_OUT = None if os.environ.get("XLLM_SWIFT_BWD_COPY") == "1" else getattr(_xllm_ops, "attention_bwd_out", None)
_SWIFT_QSPLIT = int(os.environ.get("XLLM_SWIFT_QSPLIT", "2"))
# with segment ids the split is chosen per step among these (an explicit XLLM_SWIFT_QSPLIT fixes it)
_SWIFT_QSPLITS = (_SWIFT_QSPLIT,) if "XLLM_SWIFT_QSPLIT" in os.environ else (2, 4)
# cost of one more attention call in attention-matrix elements per (batch, head), fitted on gekko-7.1B shapes
# (S 256, V 2048: ~8.3 us per MiB of bf16 matrix over 4 heads, ~84 us per call)
_SWIFT_CALL_COST = 1.3e6
_SWIFT_DS = None  # (quack dS GEMM, attention_bwd_ds_out) once resolved, False when unavailable


def _swift_ds_ops():
    """dS = P (dP - delta) as one quack GEMM (dP = dY V^T, P and delta read in the epilogue) for the swift backward,
    which then skips its dP GEMM and softmax backward (attention_bwd_ds_out). Optional (quack-kernels, sm_90+);
    env XLLM_SWIFT_DS=0 disables it."""
    global _SWIFT_DS
    if _SWIFT_DS is None:
        _SWIFT_DS = False
        if os.environ.get("XLLM_SWIFT_DS", "1") != "0" and hasattr(_xllm_ops, "attention_bwd_ds_out") \
                and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
            try:
                from quack.epilogue.frontend import gemm_epilogue
                from quack.epilogue.ops import TileLoad, ColVecLoad

                @gemm_epilogue(ops={"p": TileLoad("p"), "delta": ColVecLoad("delta")})
                def ds_epi(acc, p, delta):
                    return {"D": p * (acc - delta)}
                _SWIFT_DS = (ds_epi, _xllm_ops.attention_bwd_ds_out)
            except Exception:
                pass
    return _SWIFT_DS


def _swift_blocks(seq_len: int, chunk_size: int, split: int):
    """(query start, query end, key start) of the swift attention calls: every chunk's queries in `split` sub-chunks
    attending to the keys from the previous chunk's start up to the sub-chunk's end, so most of the causal triangle's
    masked half is never computed (split 2: 17% less GEMM / softmax work at 2 chunks). split 1 = one call per chunk.
    With segment ids, _swift_seg_blocks also moves the key starts past earlier segments."""
    cq = chunk_size // split
    return [(c * chunk_size + s * cq, c * chunk_size + (s + 1) * cq, max(c - 1, 0) * chunk_size)
            for c in range(seq_len // chunk_size) for s in range(split)]


def _swift_w_numel(blocks, bsz: int, n_heads: int) -> int:
    return sum(bsz * n_heads * (qe - qb) * (qe - kb) for qb, qe, kb in blocks)


_SWIFT_SEG_SKIP = os.environ.get("XLLM_SWIFT_SEG_SKIP", "1") != "0"
_swift_seg_cache = []  # [(weakref to segment_idx, version, chunk_size, split, blocks)], newest last


def _swift_seg_blocks(segment_idx: Tensor, chunk_size: int):
    """_swift_blocks with each call's key start moved up to the segment start of its first query (over the batch,
    rounded down to 64): keys before it are masked for every query of the call. The split among _SWIFT_QSPLITS with
    the least matrix area + per-call cost is used (deterministic in the segment ids, so the backward finds the same
    calls). The segment ids are the same for all layers of a step: the host copy (one sync) and the choice are cached
    on the tensor object for the other layers and the backward."""
    for ref, ver, cs, blocks in reversed(_swift_seg_cache):
        if ref() is segment_idx and ver == segment_idx._version and cs == chunk_size:
            return blocks
    seg = segment_idx.detach().cpu()
    bsz, seq_len = seg.shape
    # first position of every position's segment (segment ids are non-decreasing along the sequence)
    pos = torch.arange(seq_len).expand(bsz, seq_len)
    is_start = torch.ones(bsz, seq_len, dtype=torch.bool)
    is_start[:, 1:] = seg[:, 1:] != seg[:, :-1]
    start = torch.where(is_start, pos, torch.zeros_like(pos)).cummax(dim=1).values.min(dim=0).values.tolist()
    best = None
    for split in _SWIFT_QSPLITS:
        if chunk_size % (8 * split):
            continue
        blocks = [(qb, qe, max(kb, start[qb] // 64 * 64)) for qb, qe, kb in _swift_blocks(seq_len, chunk_size, split)]
        cost = _swift_w_numel(blocks, 1, 1) + _SWIFT_CALL_COST * len(blocks)
        if best is None or cost < best[0]:
            best = (cost, blocks)
    blocks = best[1]
    _swift_seg_cache.append((weakref.ref(segment_idx), segment_idx._version, chunk_size, blocks))
    del _swift_seg_cache[:-4]
    return blocks
try:
    from xattn.ops import (
        flash_sca_fwd,
        flash_sca_bwd
    )
    XATTN_SCA_ENABLED = True
except ImportError:
    XATTN_SCA_ENABLED = False
try:
    from ..cute.sliding_chunk_attention import (
        sca_fwd as cute_sca_fwd,
        sca_bwd as cute_sca_bwd,
        segment_starts as cute_segment_starts,
        query_limits as cute_query_limits,
    )
    CUTE_SCA_ENABLED = True
except ImportError:
    CUTE_SCA_ENABLED = False

_CUTE_FALLBACK_WARNED = set()


@functools.lru_cache(maxsize=None)
def _xattn_takes_dpsum() -> bool:
    """Whether the installed xattn's flash_sca_bwd takes a precomputed rowsum(dY * Y) (a local xattn patch, see
    handoff h100_scripts/patches); XLLM_XATTN_DPSUM=0 turns it off."""
    if os.environ.get("XLLM_XATTN_DPSUM", "1") == "0":
        return False
    import inspect
    return "dpsum" in inspect.signature(flash_sca_bwd).parameters


def cute_sca_supported(q: Tensor, v: Tensor, chunk_size: int, prev_k: Optional[Tensor], dropout: float) -> bool:
    """The CuTe kernels take qk head dims up to 256 (multiples of 64), v head dims that are multiples of 64 up to
    256 or of 256 beyond, chunks of multiples of 128, no previous chunk (context parallel ranks > 0) and no attention
    dropout."""
    S, V = q.shape[3], v.shape[3]
    return (CUTE_SCA_ENABLED and prev_k is None and dropout == 0.0 and S % 64 == 0 and S <= 256 and V % 64 == 0
            and (V <= 256 or V % 256 == 0) and chunk_size % 128 == 0 and q.is_cuda
            and torch.cuda.get_device_capability(q.device)[0] >= 10)


def _resolve_backend(backend: str, q: Tensor, v: Tensor, chunk_size: int, prev_k: Optional[Tensor], dropout: float) -> str:
    if backend == 'cute' and not cute_sca_supported(q, v, chunk_size, prev_k, dropout):
        key = (tuple(q.shape[2:]), v.shape[3], chunk_size, prev_k is not None, dropout)
        if key not in _CUTE_FALLBACK_WARNED:
            _CUTE_FALLBACK_WARNED.add(key)
            import warnings
            warnings.warn(f"cute SCA unsupported for head dims {q.shape[3]}/{v.shape[3]}, chunk {chunk_size}, "
                          f"prev_k={prev_k is not None}, dropout={dropout}: falling back to swift", stacklevel=3)
        return 'swift'
    return backend


class SlidingChunkAttentionFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        chunk_size: int,
        scale: float,
        prev_k: Optional[Tensor] = None,
        prev_v: Optional[Tensor] = None,
        bos_mask: Optional[Tensor] = None,
        segment_idx: Optional[Tensor] = None,
        dropout: float = 0.0,
        high_precision_level: int = 0,
        deterministic: bool = True,
        backend: str = 'swift',
        training: bool = True,
    ) -> Tensor:
        p = dropout if training else 0.0
        y, y_for_save, aux = sliding_chunk_attention_fwd(
            q, k, v, chunk_size, scale, prev_k, prev_v, bos_mask,
            segment_idx, p, high_precision_level, backend, training
        )
        ctx.save_for_backward(
            q, k, v, y_for_save, aux,
            prev_k, prev_v, bos_mask, segment_idx,
        )

        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.dropout = dropout
        ctx.high_precision_level = high_precision_level
        ctx.deterministic = deterministic
        ctx.backend = backend
        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, None, None,
               Optional[Tensor], Optional[Tensor],
               None, None, None, None, None, None, None]:
        q, k, v, y, aux, prev_k, prev_v, bos_mask, segment_idx = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        scale = ctx.scale
        dropout = ctx.dropout
        high_precision_level = ctx.high_precision_level
        deterministic = ctx.deterministic
        backend = ctx.backend
        q_grad, k_grad, v_grad, prev_k_grad, prev_v_grad = sliding_chunk_attention_bwd(
            y_grad, q, k, v, y, aux, chunk_size, scale, prev_k, prev_v,
            bos_mask, segment_idx, high_precision_level, deterministic, backend
        )

        return q_grad, k_grad, v_grad, None, None, prev_k_grad, prev_v_grad, \
            None, None, None, None, None, None, None


sliding_chunk_attention = SlidingChunkAttentionFunc.apply


def _sequential_sliding_chunk_attention_fwd(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    chunk_size: int,
    scale: float,
    prev_key_chunk: Optional[Tensor] = None,
    prev_value_chunk: Optional[Tensor] = None,
    segment_idx: Optional[Tensor] = None,
    prev_segment_idx: Optional[Tensor] = None,
    dropout: float = 0.0,
    save_attention_matrix: bool = False
) -> Tuple[Tensor, Optional[Tensor]]:
    bsz, seq_len, n_heads, _ = query.shape
    _, _, n_kv_heads, v_head_dim = value.shape
    assert seq_len == key.shape[1] == value.shape[1]
    assert key.shape[2] == n_kv_heads
    assert segment_idx is None or segment_idx.shape[1] == seq_len
    assert prev_segment_idx is None or prev_segment_idx.shape[1] == chunk_size
    assert seq_len % chunk_size == 0

    nc = seq_len // chunk_size
    n_rep = n_heads // n_kv_heads
    # B x L x H x V
    out = torch.empty(bsz, seq_len, n_heads, v_head_dim, dtype=value.dtype, device=value.device)
    if save_attention_matrix:
        # chunk-major: the [B, H, C, keys of chunk c] matrix of every chunk is contiguous in one flat buffer, so the
        # backward reads it without a copy (a [B, H, C, all keys] layout made both passes strided copies)
        offset = 0 if prev_key_chunk is not None else 1
        attn_w = torch.empty(bsz * n_heads * chunk_size * (2 * nc - offset) * chunk_size, dtype=value.dtype,
                             device=value.device)
        w_off = 0
    else:
        offset = None
        attn_w = None
    split = _SWIFT_QSPLIT
    if (save_attention_matrix and segment_idx is not None and _MULTISEG_FWD_OUT is not None and
            _ATTN_BWD_OUT is not None and prev_key_chunk is None and n_rep == 1 and split > 1 and
            chunk_size % (8 * split) == 0):
        blocks = (_swift_seg_blocks(segment_idx, chunk_size) if _SWIFT_SEG_SKIP else
                  _swift_blocks(seq_len, chunk_size, split))
        attn_w = torch.empty(_swift_w_numel(blocks, bsz, n_heads), dtype=value.dtype, device=value.device)
        w_off = 0
        for qb, qe, kb in blocks:
            w_shape = (bsz, n_heads, qe - qb, qe - kb)
            n_w = w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3]
            _MULTISEG_FWD_OUT(query[:, qb:qe], key[:, kb:qe], value[:, kb:qe], segment_idx[:, qb:qe],
                              segment_idx[:, kb:qe], scale, dropout, True, attn_w[w_off:w_off + n_w].view(w_shape),
                              out[:, qb:qe])
            w_off += n_w
        return out, attn_w
    for c in range(nc):
        qbos = c * chunk_size
        kvbos = c * chunk_size if c == 0 else (c - 1) * chunk_size
        eos = (c + 1) * chunk_size
        q = query[:, qbos:eos]
        k = key[:, kvbos:eos]
        v = value[:, kvbos:eos]
        q_idx = None if segment_idx is None else segment_idx[:, qbos:eos]
        k_idx = None if segment_idx is None else segment_idx[:, kvbos:eos]
        if c == 0 and prev_key_chunk is not None:
            k = torch.cat([prev_key_chunk, k], dim=1)
            v = torch.cat([prev_value_chunk, v], dim=1)
            if k_idx is not None:
                k_idx = torch.cat([prev_segment_idx, k_idx], dim=1)

        # repeat KV
        if n_rep > 1:
            k = repeat(k, 'b l h d -> b l (h n) d', n=n_rep)
            v = repeat(v, 'b l h d -> b l (h n) d', n=n_rep)
        # y: [bsz, chunk_size, nheads, vdim]
        # w: [bsz, nheads, chunk_size, 2*chunk_size]
        if q_idx is not None and save_attention_matrix and _MULTISEG_FWD_OUT is not None:
            # the probabilities go straight into their chunk-major slot of attn_w (no copy)
            w_shape = (bsz, n_heads, q.shape[1], k.shape[1])
            n_w = w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3]
            _MULTISEG_FWD_OUT(q, k, v, q_idx, k_idx, scale, dropout, True, attn_w[w_off:w_off + n_w].view(w_shape),
                              out[:, qbos:eos])
            w_off += n_w
            continue
        if q_idx is None:
            y, w = attention_fwd(q, k, v, scale, dropout, True)
        else:
            y, w = multiseg_attention_fwd(q, k, v, q_idx, k_idx, scale, dropout, True)

        out[:, qbos:eos] = y
        if save_attention_matrix:
            attn_w[w_off:w_off + w.numel()].view(w.shape).copy_(w)
            w_off += w.numel()

    return out, attn_w


def _sequential_sliding_chunk_attention_bwd(
    grad: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    w: Tensor,
    chunk_size: int,
    scale: float,
    prev_key_chunk: Optional[Tensor] = None,
    prev_value_chunk: Optional[Tensor] = None,
    segment_idx: Optional[Tensor] = None,
    delta: Optional[Tensor] = None,  # rowsum(grad * y) [B, H, L] fp32, enables the fused dS GEMM
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    bsz, seq_len, n_heads, v_head_dim = grad.shape
    _, _, n_kv_heads, qk_head_dim = key.shape
    assert seq_len == query.shape[1] and seq_len == key.shape[1] == value.shape[1]
    assert n_heads == query.shape[2] and n_kv_heads == key.shape[2] == value.shape[2]
    assert seq_len % chunk_size == 0
    nc = seq_len // chunk_size
    n_rep = n_heads // n_kv_heads
    # create grad tensors
    query_grad = torch.empty(bsz, seq_len, n_heads, qk_head_dim, dtype=query.dtype, device=query.device)
    key_grad = torch.empty(bsz, seq_len, n_kv_heads, qk_head_dim, dtype=key.dtype, device=key.device)
    value_grad = torch.empty(bsz, seq_len, n_kv_heads, v_head_dim, dtype=value.dtype, device=value.device)
    if prev_key_chunk is not None:
        splits = [2 * chunk_size] * nc
        prev_key_grad = torch.empty(bsz, chunk_size, n_kv_heads, qk_head_dim, dtype=key.dtype, device=key.device)
        prev_value_grad = torch.empty(bsz, chunk_size, n_kv_heads, v_head_dim, dtype=key.dtype, device=key.device)
    else:
        splits = [chunk_size, ] + [2 * chunk_size] * (nc - 1)
        prev_key_grad = None
        prev_value_grad = None

    if _ATTN_BWD_OUT is not None and prev_key_chunk is None and n_rep == 1 and w.dim() == 1:
        # gradients written in place; key rows already written by earlier calls are accumulated in the GEMMs (beta = 1).
        # The forward's call layout (sub-chunks or one call per chunk) follows from the attention buffer's size.
        candidates = []
        if _SWIFT_QSPLIT > 1 and _SWIFT_SEG_SKIP and segment_idx is not None:
            candidates.append(lambda: _swift_seg_blocks(segment_idx, chunk_size))
        if _SWIFT_QSPLIT > 1:
            candidates.append(lambda: _swift_blocks(seq_len, chunk_size, _SWIFT_QSPLIT))
        candidates.append(lambda: _swift_blocks(seq_len, chunk_size, 1))
        for make in candidates:
            blocks = make()
            if _swift_w_numel(blocks, bsz, n_heads) == w.numel():
                break
        ds_ops = _swift_ds_ops() if delta is not None and bsz == 1 and grad.dtype == torch.bfloat16 else False
        ds_buf = None
        w_off = 0
        for qb, qe, kb in blocks:
            w_shape = (bsz, n_heads, qe - qb, qe - kb)
            n_w = w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3]
            if ds_ops:
                if ds_buf is None:
                    ds_buf = torch.empty(max(_swift_w_numel([b], bsz, n_heads) for b in blocks), dtype=w.dtype, device=w.device)
                p = w[w_off:w_off + n_w].view(w_shape)
                ds = ds_buf[:n_w].view(w_shape)
                ds_ops[0].gemm(grad[0, qb:qe].transpose(0, 1), value[0, kb:qe].transpose(0, 1), ds[0],
                               epi_args=dict(p=p[0], delta=delta[0, :, qb:qe]), tile_M=128, tile_N=256, cluster_M=2, cluster_N=1)
                ds_ops[1](grad[:, qb:qe], query[:, qb:qe], key[:, kb:qe], value[:, kb:qe], p, ds, scale,
                          query_grad[:, qb:qe], key_grad[:, kb:qe], value_grad[:, kb:qe], qb - kb)
                w_off += n_w
                continue
            _ATTN_BWD_OUT(grad[:, qb:qe], query[:, qb:qe], key[:, kb:qe], value[:, kb:qe],
                          w[w_off:w_off + n_w].view(w_shape), scale, True, query_grad[:, qb:qe], key_grad[:, kb:qe],
                          value_grad[:, kb:qe], qb - kb)
            w_off += n_w
        return query_grad, key_grad, value_grad, None, None
    if w.dim() == 1:  # chunk-major flat buffer of _sequential_sliding_chunk_attention_fwd
        sizes = [bsz * n_heads * chunk_size * kc for kc in splits]
        ws = [t.view(bsz, n_heads, chunk_size, kc) for t, kc in zip(torch.split(w, sizes), splits)]
    else:
        ws = torch.split(w, splits, dim=-1)
    for c in range(nc):
        qbos = c * chunk_size
        kvbos = c * chunk_size if c == 0 else (c - 1) * chunk_size
        eos = (c + 1) * chunk_size
        y_grad = grad[:, qbos:eos]
        q = query[:, qbos:eos]
        k = key[:, kvbos:eos]
        v = value[:, kvbos:eos]
        if c == 0 and prev_key_chunk is not None:
            k = torch.cat([prev_key_chunk, k], dim=1)
            v = torch.cat([prev_value_chunk, v], dim=1)

        # repeat KV
        if n_rep > 1:
            k = repeat(k, 'b l h d -> b l (h n) d', n=n_rep)
            v = repeat(v, 'b l h d -> b l (h n) d', n=n_rep)

        q_grad, k_grad, v_grad = attention_bwd(y_grad, q, k, v, ws[c], scale, True)
        # B x L x H x D
        if n_rep > 1:
            k_grad = rearrange(k_grad, 'b l (h n) d -> b l h n d', n=n_rep).sum(dim=3)
            v_grad = rearrange(v_grad, 'b l (h n) d -> b l h n d', n=n_rep).sum(dim=3)
        # update grad tensors
        query_grad[:, qbos:eos] = q_grad
        curr_k_grad = k_grad[:, -chunk_size:]
        curr_v_grad = v_grad[:, -chunk_size:]
        key_grad[:, qbos:eos] = curr_k_grad
        value_grad[:, qbos:eos] = curr_v_grad

        prev_k_grad = None if k_grad.shape[1] == chunk_size else k_grad[:, :chunk_size]
        prev_v_grad = None if v_grad.shape[1] == chunk_size else v_grad[:, :chunk_size]
        if c == 0:
            prev_key_grad = prev_k_grad
            prev_value_grad = prev_v_grad
        else:
            key_grad[:, kvbos:qbos] += prev_k_grad
            value_grad[:, kvbos:qbos] += prev_v_grad

    return query_grad, key_grad, value_grad, prev_key_grad, prev_value_grad


def sliding_chunk_attention_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int,
    scale: float,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    segment_idx: Optional[Tensor] = None,
    dropout: float = 0.0,
    high_precision_level: int = 0,
    backend: str = 'xattn',
    requires_grad: bool = False
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    backend = _resolve_backend(backend, q, v, chunk_size, prev_k, dropout)
    if backend in ['xattn', 'flash']:
        assert XATTN_SCA_ENABLED, "xattn was not installed."
        assert dropout == 0.0, f"xattn SCA does not support attention dropout: {dropout}"
        if bos_mask is not None:
            segment_idx = None

        y, y_fp32, lse = flash_sca_fwd(
            q, k, v, chunk_size, scale, prev_k, prev_v,
            bos_mask=bos_mask, segment_idx=segment_idx,
            high_precision_level=high_precision_level
        )
        lse = lse if requires_grad else None
        y_for_bwd = (y if high_precision_level == 0 else y_fp32) if requires_grad else None
        return y, y_for_bwd, lse
    elif backend == 'cute':
        assert CUTE_SCA_ENABLED, "the CuTe DSL (nvidia-cutlass-dsl) is not installed."
        assert prev_k is None and prev_v is None, "cute SCA: the previous chunk (context parallel ranks > 0) is not supported."
        assert dropout == 0.0, f"cute SCA does not support attention dropout: {dropout}"
        if segment_idx is None and bos_mask is not None:
            segment_idx = torch.cumsum(bos_mask.long(), dim=1) - 1
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        kmin = cute_segment_starts(segment_idx, q.shape[0], q.shape[1], q.device)
        y, lse2 = cute_sca_fwd(q, k, v, chunk_size, scale, kmin)
        return y, (y if requires_grad else None), (lse2 if requires_grad else None)
    elif backend == 'swift':
        if segment_idx is not None and prev_k is not None:
            assert segment_idx.shape[1] == q.shape[1] + chunk_size
            prev_segment_idx = segment_idx[:, :chunk_size]
            segment_idx = segment_idx[:, chunk_size:]
        else:
            prev_segment_idx = None

        y, w = _sequential_sliding_chunk_attention_fwd(
            q, k, v, chunk_size, scale, prev_k, prev_v, segment_idx, prev_segment_idx, dropout, requires_grad
        )
        return y, None, w
    else:
        raise ValueError(f"Unknown backend: {backend}.")


def sliding_chunk_attention_bwd(
    y_grad: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    y: Optional[Tensor],
    aux: Tensor,  # attn_w for swift & lse for flash
    chunk_size: int,
    scale: float,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    segment_idx: Optional[Tensor] = None,
    high_precision_level: int = 0,
    deterministic: bool = False,
    backend: str = 'xattn',
    delta: Optional[Tensor] = None,  # cute: rowsum(y_grad * y) [B, H, L] fp32 if already available
    fp8_dp: bool = False,  # cute: the dP = dO V^T GEMMs of the V > 256 backward in MXFP8 (ModelConf.fp8_attn_dp)
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    backend = _resolve_backend(backend, q, v, chunk_size, prev_k, 0.0)
    if backend in ['xattn', 'flash']:
        assert XATTN_SCA_ENABLED, "xattn was not installed."
        if bos_mask is not None:
            segment_idx = None

        extra = dict(dpsum=delta) if delta is not None and _xattn_takes_dpsum() else {}
        return flash_sca_bwd(
            y_grad, q, k, v, y, aux, chunk_size, scale, prev_k, prev_v,
            bos_mask=bos_mask, segment_idx=segment_idx,
            high_precision_level=high_precision_level,
            deterministic=deterministic, **extra,
        )
    elif backend == 'cute':
        assert prev_k is None and prev_v is None
        if segment_idx is None and bos_mask is not None:
            segment_idx = torch.cumsum(bos_mask.long(), dim=1) - 1
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        kmin = cute_segment_starts(segment_idx, q.shape[0], q.shape[1], q.device)
        qmax = cute_query_limits(kmin)
        dq, dk, dv = cute_sca_bwd(q, k, v, y, aux, y_grad, chunk_size, scale, kmin, qmax, delta=delta, fp8_dp=fp8_dp)
        return dq, dk, dv, None, None
    elif backend == 'swift':
        return _sequential_sliding_chunk_attention_bwd(
            y_grad, q, k, v, aux, chunk_size, scale, prev_k, prev_v,
            segment_idx if prev_k is None else None, delta
        )
    else:
        raise ValueError(f"Unknown backend: {backend}.")
