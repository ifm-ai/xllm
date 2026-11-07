"""Fused SwiGLU elementwise ops (CuTe DSL, Blackwell), bf16 in / out with fp32 math and the torch path's roundings:

    forward:  hidden = bf16(bf16(silu(h1)) * h3)                                   (torch: 2 kernels, 5 passes -> 3)
    backward: h = hidden (for the w2 weight gradient), h1_grad = silu'(h1) bf16(h_grad h3), h3_grad = bf16(h_grad silu(h1))
                                                                                  (torch: 5 kernels, 14 passes -> 6)"""
import os
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

try:
    from .adaptive_working_memory import _compiled, _fake
except ImportError:  # loaded from its file (tests)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("cute_adaptive_working_memory_fwd",
                                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaptive_working_memory.py"))
    _awm = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_awm)
    _compiled, _fake = _awm._compiled, _awm._fake

THREADS = 256
VEC = 8


@cute.jit
def _sigmoid(x: Float32) -> Float32:
    return 1.0 / (1.0 + cute.math.exp(-x, fastmath=True))


class SwigluFwd:
    @cute.jit
    def __call__(self, tH1: cute.Tensor, tH3: cute.Tensor, tOut: cute.Tensor, n: Int32, stream: cuda.CUstream):
        n_thr = n // VEC
        self.kernel(tH1.iterator, tH3.iterator, tOut.iterator, n_thr).launch(
            grid=[(n_thr + THREADS - 1) // THREADS, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pH1: cute.Pointer, pH3: cute.Pointer, pOut: cute.Pointer, n_thr: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gid = bidx * THREADS + tidx
        if gid < n_thr:
            off = Int64(gid) * VEC
            h1 = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            h1.store(cute.make_tensor((pH1 + off).align(16), cute.make_layout(VEC)).load().to(Float32))
            h3 = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            h3.store(cute.make_tensor((pH3 + off).align(16), cute.make_layout(VEC)).load().to(Float32))
            o = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            for k in cutlass.range_constexpr(VEC):
                s = BFloat16(h1[k] * _sigmoid(h1[k])).to(Float32)
                o[k] = s * h3[k]
            cute.make_tensor((pOut + off).align(16), cute.make_layout(VEC)).store(o.load().to(BFloat16))


class SwigluBwd:
    def __init__(self, want_h: bool = True):
        self.want_h = want_h  # False: the recomputed hidden is not written (tH unused)

    @cute.jit
    def __call__(self, tH1: cute.Tensor, tH3: cute.Tensor, tG: cute.Tensor, tH: cute.Tensor, tD1: cute.Tensor, tD3: cute.Tensor,
                 n: Int32, stream: cuda.CUstream):
        n_thr = n // VEC
        self.kernel(tH1.iterator, tH3.iterator, tG.iterator, tH.iterator, tD1.iterator, tD3.iterator, n_thr).launch(
            grid=[(n_thr + THREADS - 1) // THREADS, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pH1: cute.Pointer, pH3: cute.Pointer, pG: cute.Pointer, pH: cute.Pointer, pD1: cute.Pointer, pD3: cute.Pointer,
               n_thr: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gid = bidx * THREADS + tidx
        if gid < n_thr:
            off = Int64(gid) * VEC
            h1 = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            h1.store(cute.make_tensor((pH1 + off).align(16), cute.make_layout(VEC)).load().to(Float32))
            h3 = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            h3.store(cute.make_tensor((pH3 + off).align(16), cute.make_layout(VEC)).load().to(Float32))
            g = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            g.store(cute.make_tensor((pG + off).align(16), cute.make_layout(VEC)).load().to(Float32))
            h = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            d1 = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            d3 = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            for k in cutlass.range_constexpr(VEC):
                x = h1[k]
                sg = _sigmoid(x)
                s = BFloat16(x * sg).to(Float32)  # bf16 silu(h1) as in the torch path
                if cutlass.const_expr(self.want_h):
                    h[k] = s * h3[k]
                gs = BFloat16(g[k] * h3[k]).to(Float32)  # bf16 h1s_grad
                d1[k] = gs * (sg * (1.0 + x * (1.0 - sg)))
                d3[k] = g[k] * s
            if cutlass.const_expr(self.want_h):
                cute.make_tensor((pH + off).align(16), cute.make_layout(VEC)).store(h.load().to(BFloat16))
            cute.make_tensor((pD1 + off).align(16), cute.make_layout(VEC)).store(d1.load().to(BFloat16))
            cute.make_tensor((pD3 + off).align(16), cute.make_layout(VEC)).store(d3.load().to(BFloat16))


def swiglu_fwd(h1: torch.Tensor, h3: torch.Tensor) -> torch.Tensor:
    """bf16(silu(h1)) * h3 (bf16)."""
    h1 = h1.contiguous()
    h3 = h3.contiguous()
    out = torch.empty_like(h1)
    fn = _compiled(("swiglu_fwd",), SwigluFwd, lambda: (_fake(BFloat16), _fake(BFloat16), _fake(BFloat16), Int32(0)))
    fn(h1.reshape(-1), h3.reshape(-1), out.reshape(-1), h1.numel())
    return out


def swiglu_bwd(h1: torch.Tensor, h3: torch.Tensor, h_grad: torch.Tensor, want_h: bool = True) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """(hidden, h1_grad, h3_grad) of hidden = silu(h1) * h3 for the gradient h_grad (all bf16); ``want_h=False`` skips
    the hidden recompute (None)."""
    h1 = h1.contiguous()
    h3 = h3.contiguous()
    h_grad = h_grad.contiguous()
    d1 = torch.empty_like(h1)
    d3 = torch.empty_like(h1)
    h = torch.empty_like(h1) if want_h else d1
    fn = _compiled(("swiglu_bwd", want_h), lambda: SwigluBwd(want_h), lambda: (_fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16),
                                                        _fake(BFloat16), Int32(0)))
    fn(h1.reshape(-1), h3.reshape(-1), h_grad.reshape(-1), h.reshape(-1), d1.reshape(-1), d3.reshape(-1), h1.numel())
    return (h if want_h else None), d1, d3


def swiglu_supported(h1: torch.Tensor, dropout: float) -> bool:
    if os.environ.get("XLLM_CUTE_SWIGLU", "1") == "0" or h1.dtype != torch.bfloat16 or not h1.is_cuda or dropout != 0:
        return False
    if h1.numel() % VEC or h1.numel() >= 2 ** 31:
        return False
    return torch.cuda.get_device_capability(h1.device)[0] >= 9


# ----------------------------------------------------------------------------------------------------------------------
# SwiGLU straight into MXFP8 operands (xllm/modules/fused_ops/mxfp8.py): no bf16 hidden / h1_grad / h3_grad.
#
# A CTA of 256 threads handles a 32 x 64 tile: thread (row, c8) holds 8 channels of one row (16-byte loads, 8 threads
# per row = 128 contiguous bytes), so the rowwise groups of 32 are 4 consecutive lanes (two xor shuffles for the amax),
# and the colwise groups (32 rows of one column) go through a 32 x 65 fp32 smem transpose: thread t < 64 owns column
# t's amax / scale and broadcasts its inverse scale through smem. Both operands are written in the (M, C) layout
# (8-byte stores) with the swizzled e8m0 scales of mxfp8.py. Traffic per element: bwd 6 bytes read + 4 written (cute
# SwigluBwd + mxfp8 quantisation: 6 + 4 + 4 + 4), fwd 4 + 2 (was 4 + 2 + 2 + 2).

from cutlass import Uint8, Float8E4M3FN
from cutlass._mlir.dialects import arith as _mlir_arith
from cutlass._mlir import ir as _ir

MX_ROWS = 32  # rows per CTA tile
MX_COLS = 64  # columns per CTA tile (8 threads x VEC)
_TILE_LD = MX_COLS + 1  # padded smem row: conflict-free column reads


@cute.jit
def _f2i(x: Float32) -> Int32:
    return Int32(_mlir_arith.bitcast(_ir.IntegerType.get_signless(32), Float32(x).ir_value()))


@cute.jit
def _i2f(w: Int32) -> Float32:
    return Float32(_mlir_arith.bitcast(_ir.F32Type.get(), Int32(w).ir_value()))


@cute.jit
def _e8m0(amax: Float32):
    """(biased e8m0 exponent, inverse scale 2^(127 - e8)) of a group's amax: the amax lands in [256, 512)."""
    e8 = ((_f2i(amax) >> 23) & 0xFF) - 8
    e8 = e8 - (e8 & (e8 >> 31))  # max(e8, 0)
    return e8, _i2f((254 - e8) << 23)


@cute.jit
def _scale_off(r: Int32, scol: Int32, ncb: Int32) -> Int64:
    """Byte offset of scale (row r, scale column scol) in the swizzled layout (mxfp8.py: tiles of 128 rows x 4 scale
    columns, 512 bytes each, element (r % 32) * 16 + ((r // 32) % 4) * 4 + scol % 4)."""
    return Int64(((r // 128) * ncb + (scol >> 2)) * 512 + (r % 32) * 16 + ((r >> 5) & 3) * 4 + (scol & 3))


@cute.jit
def _mx_tile(v: cute.Tensor, tile: cute.Tensor, sinv: cute.Tensor, tidx: Int32, row: Int32, c8: Int32, r: Int32, rb: Int32,
             col0: Int32, pRow: cute.Pointer, pSr: cute.Pointer, pCol: cute.Pointer, pSc: cute.Pointer, ld_out: Int32,
             ncb_r: Int32, ncb_c: Int32):
    """v: this thread's VEC fp32 values = row r, columns col0 + c8 * VEC .. of the CTA tile (rows rb * 32 ..) -> both
    operands. Two CTA barriers (the smem tile, the inverse scales)."""
    cbase = col0 + c8 * VEC
    # rowwise: the group of 32 = 4 consecutive lanes (same row)
    amax = Float32(0.0)
    for k in cutlass.range_constexpr(VEC):
        amax = cute.arch.fmax(amax, cute.arch.fmax(v[k], -v[k]))
    amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
    amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=2))
    e8, inv = _e8m0(amax)
    q = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
    for k in cutlass.range_constexpr(VEC):
        q[k] = v[k] * inv
    q8 = cute.make_rmem_tensor(cute.make_layout(VEC), Float8E4M3FN)
    q8.store(q.load().to(Float8E4M3FN))
    obase = Int64(r) * ld_out + cbase
    cute.make_tensor((pRow + obase).align(8), cute.make_layout(VEC)).store(cute.recast_tensor(q8, Uint8).load())
    if c8 % 4 == 0:
        sr = cute.make_tensor(pSr + _scale_off(r, cbase >> 5, ncb_r), cute.make_layout(1))
        sr[0] = Uint8(e8)
    # colwise: scales along the 32 rows of the tile; thread t < 64 owns column t after the smem transpose
    for k in cutlass.range_constexpr(VEC):
        tile[row * _TILE_LD + c8 * VEC + k] = v[k]
    cute.arch.sync_threads()
    if tidx < MX_COLS:
        camax = Float32(0.0)
        for i in cutlass.range_constexpr(MX_ROWS):
            x = tile[i * _TILE_LD + tidx]
            camax = cute.arch.fmax(camax, cute.arch.fmax(x, -x))
        ce8, cinv = _e8m0(camax)
        sinv[tidx] = cinv
        sc = cute.make_tensor(pSc + _scale_off(col0 + tidx, rb, ncb_c), cute.make_layout(1))
        sc[0] = Uint8(ce8)
    cute.arch.sync_threads()
    for k in cutlass.range_constexpr(VEC):
        q[k] = v[k] * sinv[c8 * VEC + k]
    q8.store(q.load().to(Float8E4M3FN))
    cute.make_tensor((pCol + obase).align(8), cute.make_layout(VEC)).store(cute.recast_tensor(q8, Uint8).load())


@cute.jit
def _mx_tile2(v1: cute.Tensor, v3: cute.Tensor, tile: cute.Tensor, sinv: cute.Tensor, tidx: Int32, row: Int32, c8: Int32, r: Int32,
              rb: Int32, col0: Int32, F: Int32, pRow: cute.Pointer, pSr: cute.Pointer, pCol: cute.Pointer, pSc: cute.Pointer,
              ld_out: Int32, ncb_r: Int32, ncb_c: Int32):
    """``_mx_tile`` for two outputs at once (v1 at columns col0 .., v3 at F + col0 ..): both smem tiles are written
    before one barrier, threads 0-63 / 64-127 reduce the columns of the two tiles in parallel, one more barrier."""
    cbase = col0 + c8 * VEC
    q = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
    q8 = cute.make_rmem_tensor(cute.make_layout(VEC), Float8E4M3FN)
    a1 = Float32(0.0)
    a3 = Float32(0.0)
    for k in cutlass.range_constexpr(VEC):
        a1 = cute.arch.fmax(a1, cute.arch.fmax(v1[k], -v1[k]))
        a3 = cute.arch.fmax(a3, cute.arch.fmax(v3[k], -v3[k]))
    for o in cutlass.range_constexpr(2):
        a1 = cute.arch.fmax(a1, cute.arch.shuffle_sync_bfly(a1, offset=1 << o))
        a3 = cute.arch.fmax(a3, cute.arch.shuffle_sync_bfly(a3, offset=1 << o))
    e1, i1 = _e8m0(a1)
    e3, i3 = _e8m0(a3)
    obase = Int64(r) * ld_out + cbase
    for k in cutlass.range_constexpr(VEC):
        q[k] = v1[k] * i1
    q8.store(q.load().to(Float8E4M3FN))
    cute.make_tensor((pRow + obase).align(8), cute.make_layout(VEC)).store(cute.recast_tensor(q8, Uint8).load())
    for k in cutlass.range_constexpr(VEC):
        q[k] = v3[k] * i3
    q8.store(q.load().to(Float8E4M3FN))
    cute.make_tensor((pRow + obase + F).align(8), cute.make_layout(VEC)).store(cute.recast_tensor(q8, Uint8).load())
    if c8 % 4 == 0:
        sr = cute.make_tensor(pSr + _scale_off(r, cbase >> 5, ncb_r), cute.make_layout(1))
        sr[0] = Uint8(e1)
        sr3 = cute.make_tensor(pSr + _scale_off(r, (F + cbase) >> 5, ncb_r), cute.make_layout(1))
        sr3[0] = Uint8(e3)
    for k in cutlass.range_constexpr(VEC):
        tile[row * _TILE_LD + c8 * VEC + k] = v1[k]
        tile[MX_ROWS * _TILE_LD + row * _TILE_LD + c8 * VEC + k] = v3[k]
    cute.arch.sync_threads()
    if tidx < 2 * MX_COLS:
        tb = (tidx // MX_COLS) * (MX_ROWS * _TILE_LD)
        c = tidx % MX_COLS
        camax = Float32(0.0)
        for i in cutlass.range_constexpr(MX_ROWS):
            x = tile[tb + i * _TILE_LD + c]
            camax = cute.arch.fmax(camax, cute.arch.fmax(x, -x))
        ce8, cinv = _e8m0(camax)
        sinv[tidx] = cinv
        sc = cute.make_tensor(pSc + _scale_off((tidx // MX_COLS) * F + col0 + c, rb, ncb_c), cute.make_layout(1))
        sc[0] = Uint8(ce8)
    cute.arch.sync_threads()
    for k in cutlass.range_constexpr(VEC):
        q[k] = v1[k] * sinv[c8 * VEC + k]
    q8.store(q.load().to(Float8E4M3FN))
    cute.make_tensor((pCol + obase).align(8), cute.make_layout(VEC)).store(cute.recast_tensor(q8, Uint8).load())
    for k in cutlass.range_constexpr(VEC):
        q[k] = v3[k] * sinv[MX_COLS + c8 * VEC + k]
    q8.store(q.load().to(Float8E4M3FN))
    cute.make_tensor((pCol + obase + F).align(8), cute.make_layout(VEC)).store(cute.recast_tensor(q8, Uint8).load())


class SwigluMx:
    """hidden = silu(h1) * h3 (bwd=False) or (h1_grad, h3_grad) for the gradient g (bwd=True, h3_grad at column offset
    F of the (M, 2F) operands), written as MX operands (uint8 data buffers + swizzled e8m0 scales)."""

    def __init__(self, bwd: bool):
        self.bwd = bwd

    @cute.jit
    def __call__(self, tH1: cute.Tensor, tH3: cute.Tensor, tG: cute.Tensor, tRow: cute.Tensor, tSr: cute.Tensor,
                 tCol: cute.Tensor, tSc: cute.Tensor, M: Int32, F: Int32, ld_in: Int32, ld_g: Int32, ld_out: Int32, stream: cuda.CUstream):
        self.kernel(tH1.iterator, tH3.iterator, tG.iterator, tRow.iterator, tSr.iterator, tCol.iterator, tSc.iterator,
                    M, F, ld_in, ld_g, ld_out).launch(grid=[(M // MX_ROWS) * (F // MX_COLS), 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pH1: cute.Pointer, pH3: cute.Pointer, pG: cute.Pointer, pRow: cute.Pointer, pSr: cute.Pointer,
               pCol: cute.Pointer, pSc: cute.Pointer, M: Int32, F: Int32, ld_in: Int32, ld_g: Int32, ld_out: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        tile = smem.allocate_tensor(Float32, cute.make_layout(2 * MX_ROWS * _TILE_LD), 16)
        sinv = smem.allocate_tensor(Float32, cute.make_layout(2 * MX_COLS), 16)
        nct = F // MX_COLS
        rb = bidx // nct
        cb = bidx % nct
        row = tidx // 8
        c8 = tidx % 8
        r = rb * MX_ROWS + row
        col0 = cb * MX_COLS
        ibase = Int64(r) * ld_in + col0 + c8 * VEC
        h1 = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
        h1.store(cute.make_tensor((pH1 + ibase).align(16), cute.make_layout(VEC)).load().to(Float32))
        h3 = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
        h3.store(cute.make_tensor((pH3 + ibase).align(16), cute.make_layout(VEC)).load().to(Float32))
        v1 = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
        ncb_r = ld_out // 128
        ncb_c = M // 128
        if cutlass.const_expr(self.bwd):
            g = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            g.store(cute.make_tensor((pG + (Int64(r) * ld_g + col0 + c8 * VEC)).align(16), cute.make_layout(VEC)).load().to(Float32))
            v3 = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            for k in cutlass.range_constexpr(VEC):
                x = h1[k]
                sg = _sigmoid(x)
                s = BFloat16(x * sg).to(Float32)  # bf16 silu(h1) as in the torch path
                gs = BFloat16(g[k] * h3[k]).to(Float32)  # bf16 h1s_grad
                v1[k] = gs * (sg * (1.0 + x * (1.0 - sg)))
                v3[k] = g[k] * s
            _mx_tile2(v1, v3, tile, sinv, tidx, row, c8, r, rb, col0, F, pRow, pSr, pCol, pSc, ld_out, ncb_r, ncb_c)
        else:
            for k in cutlass.range_constexpr(VEC):
                x = h1[k]
                s = BFloat16(x * _sigmoid(x)).to(Float32)
                v1[k] = s * h3[k]
            _mx_tile(v1, tile, sinv, tidx, row, c8, r, rb, col0, pRow, pSr, pCol, pSc, ld_out, ncb_r, ncb_c)


def swiglu_mx(h1: torch.Tensor, h3: torch.Tensor, h_grad: Optional[torch.Tensor], row: torch.Tensor, sr: torch.Tensor,
              col: torch.Tensor, sc: torch.Tensor):
    """h1, h3[, h_grad] (M, F) bf16 (h1 / h3 with the same row stride, e.g. the halves of one (M, 2F) GEMM output;
    h_grad with its own) -> the MX operands: ``row`` / ``col`` (M, ld_out) uint8 data
    buffers (ld_out = F for the forward's hidden, 2F for the backward's [h1_grad | h3_grad]), ``sr`` / ``sc`` their
    swizzled e8m0 scales (uint8). Dimensions: M % 128 == 0, F % 128 == 0."""
    M, F = h1.shape
    bwd = h_grad is not None
    assert h1.dtype == h3.dtype == torch.bfloat16 and h1.stride(1) == 1 and h3.stride(0) == h1.stride(0) and h3.stride(1) == 1
    assert M % 128 == 0 and F % 128 == 0 and h1.stride(0) % 64 == 0, (h1.shape, h1.stride())
    assert row.dtype == torch.uint8 and row.shape == (M, (2 * F if bwd else F)) and row.is_contiguous() and col.shape == row.shape
    if bwd:
        assert h_grad.shape == h1.shape and h_grad.stride(1) == 1 and h_grad.stride(0) % 64 == 0 and h_grad.dtype == torch.bfloat16
    fn = _compiled(("swiglu_mx", bwd), lambda: SwigluMx(bwd), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Uint8), _fake(Uint8, 1), _fake(Uint8), _fake(Uint8, 1),
        Int32(0), Int32(0), Int32(0), Int32(0), Int32(0)))
    fn(_flat(h1), _flat(h3), _flat(h_grad if bwd else h1),
       row.reshape(-1), sr.reshape(-1), col.reshape(-1), sc.reshape(-1), M, F, h1.stride(0), (h_grad if bwd else h1).stride(0), row.shape[1])


def _flat(t: torch.Tensor) -> torch.Tensor:
    """A 1-D view over the storage of a row-strided (M, F) matrix (rows may be wider than F: a column slice)."""
    if t.is_contiguous():
        return t.reshape(-1)
    return t.as_strided((t.stride(0) * (t.shape[0] - 1) + t.shape[1],), (1,), t.storage_offset())
