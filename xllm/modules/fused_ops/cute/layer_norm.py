"""Group layer norm forward / backward (CuTe DSL, Blackwell) for the Gekko FFN norm: rows (b, l, g) of
C = D / G channels, bf16 in / out, fp32 statistics (mean, rstd [B * L, G]) and affine weight (w + 1) / bias.

    forward:  y = (x - mean) rstd w + b
    backward: g = dy w, dx = rstd (g - mean(g) - xhat mean(g xhat)), dw = sum dy xhat, db = sum dy

A row takes max(1, C / 256) warps (8 channels per lane, 16-byte loads); in the backward a row slot only sees rows
of one group so the weight / bias gradient partials stay in registers (partial[slot, C], summed on the host).
The extension's kernels run at ~1.7 TB/s on these shapes; these run at ~5-6 TB/s."""
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

MAX_WARPS = 16


def _tiling(C: int) -> Tuple[int, int, int]:
    """(warps per row, elements per lane, rows per CTA) for C channels per row: rows that need a cross-warp reduction
    take a CTA of their own (the barriers of a small CTA are cheaper and more CTAs fit per SM: 7.1B backward 148 -> 128 us
    and forward 69 -> 63 for 1 instead of 2 rows per CTA; 0.9B / 4B forward unchanged), single-warp rows share CTAs of
    8 warps."""
    wpr = max(1, C // 256)
    e = C // (32 * wpr)
    rows_per_cta = 8 if wpr == 1 else 1
    return wpr, e, rows_per_cta


@cute.jit
def _warp_sum(v: Float32) -> Float32:
    for o in cutlass.range_constexpr(5):
        v = v + cute.arch.shuffle_sync_bfly(v, offset=1 << o)
    return v


@cute.jit
def _row_sum(v: Float32, red: cute.Tensor, wpr: cutlass.Constexpr, widx: Int32, lane: Int32, slot: cutlass.Constexpr) -> Float32:
    """Sum over the wpr warps of a row. ``slot`` selects a group of MAX_WARPS floats of ``red``: consecutive reductions
    use different slots so that no barrier is needed after the reads (a reduction's slot is written again only after
    a later reduction's barrier, which every warp reaches after its reads)."""
    v = _warp_sum(v)
    if cutlass.const_expr(wpr > 1):
        if lane == 0:
            red[slot * MAX_WARPS + widx] = v
        cute.arch.sync_threads()
        base = slot * MAX_WARPS + (widx // wpr) * wpr
        v = Float32(0.0)
        for i in cutlass.range_constexpr(wpr):
            v = v + red[base + i]
    return v


@cute.jit
def _row_sum2(v1: Float32, v2: Float32, red: cute.Tensor, wpr: cutlass.Constexpr, widx: Int32, lane: Int32, slot: Int32) -> Tuple[Float32, Float32]:
    """Two independent row sums with one barrier (slots slot and slot + 1)."""
    v1 = _warp_sum(v1)
    v2 = _warp_sum(v2)
    if cutlass.const_expr(wpr > 1):
        if lane == 0:
            red[slot * MAX_WARPS + widx] = v1
            red[(slot + 1) * MAX_WARPS + widx] = v2
        cute.arch.sync_threads()
        base = slot * MAX_WARPS + (widx // wpr) * wpr
        v1 = Float32(0.0)
        v2 = Float32(0.0)
        for i in cutlass.range_constexpr(wpr):
            v1 = v1 + red[base + i]
            v2 = v2 + red[base + MAX_WARPS + i]
    return v1, v2


class LnFwd:
    def __init__(self, G: int, C: int):
        self.G, self.C = G, C
        self.WPR, self.E, self.RPC = _tiling(C)
        assert C % 32 == 0 and self.E in (4, 8) and self.WPR * self.RPC <= MAX_WARPS

    @cute.jit
    def __call__(self, tX: cute.Tensor, tW: cute.Tensor, tB: cute.Tensor, tY: cute.Tensor, tMean: cute.Tensor, tRstd: cute.Tensor,
                 rows: Int32, eps: Float32, stream: cuda.CUstream):
        self.kernel(tX.iterator, tW.iterator, tB.iterator, tY.iterator, tMean.iterator, tRstd.iterator, rows, eps).launch(
            grid=[(rows + self.RPC - 1) // self.RPC, 1, 1], block=[32 * self.WPR * self.RPC, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pX: cute.Pointer, pW: cute.Pointer, pB: cute.Pointer, pY: cute.Pointer, pMean: cute.Pointer, pRstd: cute.Pointer,
               rows: Int32, eps: Float32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        G, C, E, WPR, RPC = self.G, self.C, self.E, self.WPR, self.RPC
        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_tensor(Float32, cute.make_layout(2 * MAX_WARPS), 16)
        lane = tidx % 32
        widx = tidx // 32
        row = bidx * RPC + widx // WPR
        col = (widx % WPR) * (32 * E) + lane * E
        valid = row < rows
        if not valid:
            row = rows - 1
        g = row % G
        off = Int64(row) * C + col
        xv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
        xv.store(cute.make_tensor((pX + off).align(2 * E), cute.make_layout(E)).load().to(Float32))
        wv = cute.make_rmem_tensor(cute.make_layout(E), Float32)  # the affine loads are issued before the first barrier
        bv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
        wv.store(cute.make_tensor((pW + (g * C + col)).align(2 * E), cute.make_layout(E)).load().to(Float32))
        bv.store(cute.make_tensor((pB + (g * C + col)).align(2 * E), cute.make_layout(E)).load().to(Float32))
        s = Float32(0.0)
        for k in cutlass.range_constexpr(E):
            s = s + xv[k]
        mean = _row_sum(s, red, WPR, widx, lane, 0) * (1.0 / C)
        v = Float32(0.0)
        for k in cutlass.range_constexpr(E):
            d = xv[k] - mean
            v = v + d * d
        var = _row_sum(v, red, WPR, widx, lane, 1) * (1.0 / C)
        rstd = cute.math.rsqrt(var + eps, fastmath=True)
        yv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
        yv.store((xv.load() - mean) * rstd * wv.load() + bv.load())
        if valid:
            cute.make_tensor((pY + off).align(2 * E), cute.make_layout(E)).store(yv.load().to(BFloat16))
            if tidx % (32 * WPR) == 0:
                cute.make_tensor(pMean + row, cute.make_layout(1))[0] = mean
                cute.make_tensor(pRstd + row, cute.make_layout(1))[0] = rstd


class LnBwd:
    """``has_add``: dx += addend (bf16, dy's layout), the residual-branch gradient fused into the store."""

    def __init__(self, G: int, C: int, n_slots: int, has_add: bool = False):
        self.G, self.C, self.n_slots, self.has_add = G, C, n_slots, has_add
        self.WPR, self.E, self.RPC = _tiling(C)
        assert C % 32 == 0 and self.E in (4, 8) and self.WPR * self.RPC <= MAX_WARPS and n_slots % (G * self.RPC) == 0

    @cute.jit
    def __call__(self, tDy: cute.Tensor, tX: cute.Tensor, tMean: cute.Tensor, tRstd: cute.Tensor, tW: cute.Tensor, tDx: cute.Tensor,
                 tPart: cute.Tensor, tAdd: cute.Tensor, rows: Int32, stream: cuda.CUstream):
        self.kernel(tDy.iterator, tX.iterator, tMean.iterator, tRstd.iterator, tW.iterator, tDx.iterator, tPart.iterator, tAdd.iterator,
                    rows).launch(grid=[self.n_slots // self.RPC, 1, 1], block=[32 * self.WPR * self.RPC, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pDy: cute.Pointer, pX: cute.Pointer, pMean: cute.Pointer, pRstd: cute.Pointer, pW: cute.Pointer, pDx: cute.Pointer,
               pPart: cute.Pointer, pAdd: cute.Pointer, rows: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        G, C, E, WPR, RPC = self.G, self.C, self.E, self.WPR, self.RPC
        n_slots = self.n_slots
        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_tensor(Float32, cute.make_layout(4 * MAX_WARPS), 16)
        lane = tidx % 32
        widx = tidx // 32
        slot = bidx * RPC + widx // WPR  # rows slot, slot + n_slots, ... (all of group slot % G)
        col = (widx % WPR) * (32 * E) + lane * E
        g = slot % G
        wv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
        wv.store(cute.make_tensor((pW + (g * C + col)).align(2 * E), cute.make_layout(E)).load().to(Float32))
        acc_w = cute.make_rmem_tensor(cute.make_layout(E), Float32)
        acc_b = cute.make_rmem_tensor(cute.make_layout(E), Float32)
        for k in cutlass.range_constexpr(E):
            acc_w[k] = 0.0
            acc_b[k] = 0.0
        n_it = (rows - slot + n_slots - 1) // n_slots
        for it in cutlass.range(n_it):
            row = slot + it * n_slots
            off = Int64(row) * C + col
            mean = cute.make_tensor(pMean + row, cute.make_layout(1))[0]
            rstd = cute.make_tensor(pRstd + row, cute.make_layout(1))[0]
            xh = cute.make_rmem_tensor(cute.make_layout(E), Float32)
            xh.store((cute.make_tensor((pX + off).align(2 * E), cute.make_layout(E)).load().to(Float32) - mean) * rstd)
            dy = cute.make_rmem_tensor(cute.make_layout(E), Float32)
            dy.store(cute.make_tensor((pDy + off).align(2 * E), cute.make_layout(E)).load().to(Float32))
            s1 = Float32(0.0)
            s2 = Float32(0.0)
            for k in cutlass.range_constexpr(E):
                acc_w[k] = acc_w[k] + dy[k] * xh[k]
                acc_b[k] = acc_b[k] + dy[k]
                dy[k] = dy[k] * wv[k]
                s1 = s1 + dy[k]
                s2 = s2 + dy[k] * xh[k]
            av = cute.make_rmem_tensor(cute.make_layout(E), Float32)
            if cutlass.const_expr(self.has_add):  # loaded before the barrier
                av.store(cute.make_tensor((pAdd + off).align(2 * E), cute.make_layout(E)).load().to(Float32))
            s1, s2 = _row_sum2(s1, s2, red, WPR, widx, lane, (it % 2) * 2)
            s1 = s1 * (1.0 / C)
            s2 = s2 * (1.0 / C)
            dx = cute.make_rmem_tensor(cute.make_layout(E), Float32)
            for k in cutlass.range_constexpr(E):
                dx[k] = rstd * (dy[k] - s1 - xh[k] * s2)
            if cutlass.const_expr(self.has_add):
                for k in cutlass.range_constexpr(E):
                    dx[k] = dx[k] + av[k]
            cute.make_tensor((pDx + off).align(2 * E), cute.make_layout(E)).store(dx.load().to(BFloat16))
        pw = pPart + (Int64(slot) * C + col)
        cute.make_tensor(pw.align(4 * E), cute.make_layout(E)).store(acc_w.load())
        cute.make_tensor((pw + Int64(n_slots) * C).align(4 * E), cute.make_layout(E)).store(acc_b.load())


def ln_fwd(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, G: int, eps: float):
    """(y bf16 [B, L, D], mean fp32 [B * L, G], rstd fp32 [B * L, G]) of the group layer norm with the affine
    (weight + 1) and bias, like xllm_ops.group_layer_norm_fwd_affine."""
    B, L, D = x.shape
    C = D // G
    rows = B * L * G
    x = x.contiguous()
    dev = x.device
    w = (weight + 1.0).to(torch.bfloat16).reshape(-1)
    b = bias.to(torch.bfloat16).reshape(-1)
    y = torch.empty_like(x)
    mean = torch.empty(B * L, G, dtype=torch.float32, device=dev)
    rstd = torch.empty(B * L, G, dtype=torch.float32, device=dev)
    fn = _compiled(("ln_fwd", G, C), lambda: LnFwd(G, C), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Float32, 4), Int32(0), Float32(0.0)))
    fn(x.reshape(-1), w, b, y.reshape(-1), mean.reshape(-1), rstd.reshape(-1), rows, float(eps))
    return y, mean, rstd


def ln_bwd(dy: torch.Tensor, x: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, weight: torch.Tensor, G: int,
           addend: Optional[torch.Tensor] = None):
    """(dx bf16 [B, L, D], dweight bf16 [D], dbias bf16 [D]) of ln_fwd (weight raw, as given to ln_fwd); ``addend`` (bf16
    [B, L, D], e.g. the residual-branch gradient) is added to dx in the kernel."""
    B, L, D = x.shape
    C = D // G
    rows = B * L * G
    x = x.contiguous()
    dev = x.device
    _, _, rpc = _tiling(C)
    unit = G * rpc
    n_slots = -(-int(os.environ.get("XLLM_LN_SLOTS", "4096")) // unit) * unit
    w = (weight + 1.0).to(torch.bfloat16).reshape(-1)
    dx = torch.empty_like(x)
    part = torch.empty(2 * n_slots, C, dtype=torch.float32, device=dev)
    has_add = addend is not None
    fn = _compiled(("ln_bwd", G, C, n_slots, has_add), lambda: LnBwd(G, C, n_slots, has_add), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Float32, 4), _fake(BFloat16), _fake(BFloat16), _fake(Float32),
        _fake(BFloat16), Int32(0)))
    fn(dy.contiguous().reshape(-1), x.reshape(-1), mean.reshape(-1), rstd.reshape(-1), w, dx.reshape(-1), part.reshape(-1),
       (addend.contiguous() if has_add else dx).reshape(-1), rows)
    sums = part.view(2, n_slots // G, G, C).sum(1).reshape(2, D)
    return dx, sums[0].to(torch.bfloat16), sums[1].to(torch.bfloat16)


def ln_supported(x: torch.Tensor, G: int) -> bool:
    if os.environ.get("XLLM_CUTE_LN", "1") == "0" or x.dtype != torch.bfloat16 or not x.is_cuda or x.dim() != 3 or x.shape[-1] % G:
        return False
    C = x.shape[-1] // G
    wpr, e, rpc = _tiling(C)
    if C % 32 or e not in (4, 8) or wpr * rpc > MAX_WARPS or x.numel() >= 2 ** 31:
        return False
    return torch.cuda.get_device_capability(x.device)[0] >= 9
