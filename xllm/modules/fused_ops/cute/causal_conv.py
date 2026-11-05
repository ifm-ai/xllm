"""Depthwise causal conv1d of the Gekko block (CuTe DSL, Blackwell): width-4 taps per channel, segment resets from
the bos mask (a position only sees earlier positions of its own segment), optional silu; no bias / state.

    out[t, d] = act(sum_lag w[d, W - 1 - lag] x[t - lag, d] [no bos in (t - lag, t]])

A thread owns 8 channels and walks a tile of T timesteps, 4 rows per iteration, with the last 3 rows in registers
(one read of x, one write of out; packed f32x2 math, a mask-free path when no boundary is near). Forward only by
default: 0.9B q [2, 16384, 2048] 128 -> 75 us vs the Triton kernel, v (silu) 126 -> 59, 7.1B v 207 -> 164. The
backward kernel (ConvBwd, 8 channels per thread, softmax in-kernel, ConvDwFinish for the weight gradient) beats the Triton
backward on the smaller shapes only (4B q/k [1, 8192, 1024]: 46 vs 97 us; it is issue bound at 255 registers, 2 CTAs/SM):
conv_bwd_preferred picks it for x.numel() <= 2^24 (XLLM_CUTE_CONV_BWD=0 disables it)."""
import os
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8

try:
    from .adaptive_working_memory import _compiled, _fake
except ImportError:  # loaded from its file (tests)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("cute_adaptive_working_memory_fwd",
                                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaptive_working_memory.py"))
    _awm = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_awm)
    _compiled, _fake = _awm._compiled, _awm._fake

THREADS = 128
VEC = 8
T = 32
U = 4  # timesteps per loop iteration (independent loads in flight)
WIDTH = 4
HALO = WIDTH - 1
assert T % U == 0 and U >= WIDTH - 1


@cute.jit
def _silu(x: Float32) -> Float32:
    return x * cute.arch.rcp_approx(1.0 + cute.math.exp2(-x * 1.4426950408889634, fastmath=True))


@cute.jit
def _dsilu(x: Float32) -> Float32:
    s = cute.arch.rcp_approx(1.0 + cute.math.exp2(-x * 1.4426950408889634, fastmath=True))
    return s * (1.0 + x * (1.0 - s))


class ConvFwd:
    """``softmax``: the kernel reads the raw weight [D, W] (``wdt``) and normalizes the taps of each channel itself
    (softmax over W, fp32), saving the host-side softmax / transpose / cast launches; else weight is [W, D] fp32."""

    def __init__(self, act: bool, has_bos: bool, softmax: bool = False, wdt=Float32):
        self.act, self.has_bos, self.softmax, self.wdt = act, has_bos, softmax, wdt

    @cute.jit
    def __call__(self, tX: cute.Tensor, tW: cute.Tensor, tBos: cute.Tensor, tOut: cute.Tensor, B: Int32, L: Int32, D: Int32,
                 stream: cuda.CUstream):
        self.kernel(tX.iterator, tW.iterator, tBos.iterator, tOut.iterator, L, D).launch(
            grid=[(D + THREADS * VEC - 1) // (THREADS * VEC), (L + T - 1) // T, B], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pX: cute.Pointer, pW: cute.Pointer, pBos: cute.Pointer, pOut: cute.Pointer, L: Int32, D: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, b = cute.arch.block_idx()
        c = (bx * THREADS + tidx) * VEC
        if c < D:
            t0 = by * T
            base = pX + Int64(b) * L * D + c
            obase = pOut + Int64(b) * L * D + c
            bbase = pBos + Int64(b) * L
            w = cute.make_rmem_tensor(cute.make_layout((WIDTH, VEC)), Float32)  # w[lag, k] = weight[W - 1 - lag, c + k]
            if cutlass.const_expr(self.softmax):
                # raw weight [D, W]: the W taps of the VEC channels are contiguous; softmax over the taps
                wr = cute.make_tensor((pW + c * WIDTH).align(16), cute.make_layout(VEC * WIDTH)).load().to(Float32)
                for k in cutlass.range_constexpr(VEC):
                    m = wr[k * WIDTH]
                    for j in cutlass.range_constexpr(1, WIDTH):
                        m = cute.arch.fmax(m, wr[k * WIDTH + j])
                    ssum = Float32(0.0)
                    for j in cutlass.range_constexpr(WIDTH):
                        e = cute.math.exp2((wr[k * WIDTH + j] - m) * 1.4426950408889634, fastmath=True)
                        w[WIDTH - 1 - j, k] = e
                        ssum = ssum + e
                    inv = Float32(1.0) / ssum
                    for j in cutlass.range_constexpr(WIDTH):
                        w[j, k] = w[j, k] * inv
            else:
                for lag in cutlass.range_constexpr(WIDTH):
                    wl = cute.make_tensor((pW + ((WIDTH - 1 - lag) * D + c)).align(32), cute.make_layout(VEC)).load()
                    for k in cutlass.range_constexpr(VEC):
                        w[lag, k] = wl[k]
            xw = cute.make_rmem_tensor(cute.make_layout((WIDTH - 1, VEC)), Float32)  # xw[j] = x[t - 1 - j]
            flags = cute.make_rmem_tensor(cute.make_layout(WIDTH - 1), Float32)  # flags[j] = bos[t - 1 - j]
            for j in cutlass.range_constexpr(WIDTH - 1):
                flags[j] = 0.0
                tp = t0 - 1 - j
                if tp >= 0:
                    xr = cute.make_tensor((base + Int64(tp) * D).align(16), cute.make_layout(VEC)).load().to(Float32)
                    for k in cutlass.range_constexpr(VEC):
                        xw[j, k] = xr[k]
                    if cutlass.const_expr(self.has_bos):
                        flags[j] = cute.make_tensor(bbase + tp, cute.make_layout(1))[0].to(Float32)
                else:
                    for k in cutlass.range_constexpr(VEC):
                        xw[j, k] = 0.0
            for i4 in cutlass.range(T // U):
                tb = t0 + i4 * U
                xs = cute.make_rmem_tensor(cute.make_layout((U, VEC)), Float32)
                fb = cute.make_rmem_tensor(cute.make_layout(U), Float32)
                for u in cutlass.range_constexpr(U):
                    fb[u] = 0.0
                    if tb + u < L:
                        xr = cute.make_tensor((base + Int64(tb + u) * D).align(16), cute.make_layout(VEC)).load().to(Float32)
                        for k in cutlass.range_constexpr(VEC):
                            xs[u, k] = xr[k]
                        if cutlass.const_expr(self.has_bos):
                            fb[u] = cute.make_tensor(bbase + (tb + u), cute.make_layout(1))[0].to(Float32)
                    else:
                        for k in cutlass.range_constexpr(VEC):
                            xs[u, k] = 0.0
                # no segment boundary in rows tb - 2 .. tb + 3 (the common case): no masks (CTA-uniform branch)
                any_bos = fb[0] + fb[1] + fb[2] + fb[3] + flags[0] + flags[1]
                if any_bos == 0.0:
                    for u in cutlass.range_constexpr(U):
                        o = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
                        for k2 in cutlass.range_constexpr(VEC // 2):
                            k = 2 * k2
                            a0, a1 = cute.arch.mul_packed_f32x2((xs[u, k], xs[u, k + 1]), (w[0, k], w[0, k + 1]))
                            x1a = xs[u - 1, k] if u >= 1 else xw[0, k]
                            x1b = xs[u - 1, k + 1] if u >= 1 else xw[0, k + 1]
                            a0, a1 = cute.arch.fma_packed_f32x2((x1a, x1b), (w[1, k], w[1, k + 1]), (a0, a1))
                            x2a = xs[u - 2, k] if u >= 2 else xw[1 - u, k]
                            x2b = xs[u - 2, k + 1] if u >= 2 else xw[1 - u, k + 1]
                            a0, a1 = cute.arch.fma_packed_f32x2((x2a, x2b), (w[2, k], w[2, k + 1]), (a0, a1))
                            x3a = xs[u - 3, k] if u >= 3 else xw[2 - u, k]
                            x3b = xs[u - 3, k + 1] if u >= 3 else xw[2 - u, k + 1]
                            a0, a1 = cute.arch.fma_packed_f32x2((x3a, x3b), (w[3, k], w[3, k + 1]), (a0, a1))
                            o[k] = _silu(a0) if self.act else a0
                            o[k + 1] = _silu(a1) if self.act else a1
                        if tb + u < L:
                            cute.make_tensor((obase + Int64(tb + u) * D).align(16), cute.make_layout(VEC)).store(o.load().to(BFloat16))
                else:
                    for u in cutlass.range_constexpr(U):
                        f1 = fb[u - 1] if u >= 1 else flags[0]
                        f2 = fb[u - 2] if u >= 2 else flags[1 - u]
                        v1 = 1.0 - fb[u]
                        v2 = v1 * (1.0 - f1)
                        v3 = v2 * (1.0 - f2)
                        o = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
                        for k2 in cutlass.range_constexpr(VEC // 2):
                            k = 2 * k2
                            a0, a1 = cute.arch.mul_packed_f32x2((xs[u, k], xs[u, k + 1]), (w[0, k], w[0, k + 1]))
                            x1a = xs[u - 1, k] if u >= 1 else xw[0, k]
                            x1b = xs[u - 1, k + 1] if u >= 1 else xw[0, k + 1]
                            m0, m1 = cute.arch.mul_packed_f32x2((x1a, x1b), (v1, v1))
                            a0, a1 = cute.arch.fma_packed_f32x2((m0, m1), (w[1, k], w[1, k + 1]), (a0, a1))
                            x2a = xs[u - 2, k] if u >= 2 else xw[1 - u, k]
                            x2b = xs[u - 2, k + 1] if u >= 2 else xw[1 - u, k + 1]
                            m0, m1 = cute.arch.mul_packed_f32x2((x2a, x2b), (v2, v2))
                            a0, a1 = cute.arch.fma_packed_f32x2((m0, m1), (w[2, k], w[2, k + 1]), (a0, a1))
                            x3a = xs[u - 3, k] if u >= 3 else xw[2 - u, k]
                            x3b = xs[u - 3, k + 1] if u >= 3 else xw[2 - u, k + 1]
                            m0, m1 = cute.arch.mul_packed_f32x2((x3a, x3b), (v3, v3))
                            a0, a1 = cute.arch.fma_packed_f32x2((m0, m1), (w[3, k], w[3, k + 1]), (a0, a1))
                            o[k] = _silu(a0) if self.act else a0
                            o[k + 1] = _silu(a1) if self.act else a1
                        if tb + u < L:
                            cute.make_tensor((obase + Int64(tb + u) * D).align(16), cute.make_layout(VEC)).store(o.load().to(BFloat16))
                for j in cutlass.range_constexpr(WIDTH - 1):
                    for k in cutlass.range_constexpr(VEC):
                        xw[j, k] = xs[U - 1 - j, k]
                    flags[j] = fb[U - 1 - j]


class ConvBwd:
    """Backward: g = out_grad * silu'(pre) (pre recomputed), dx[t] = sum_lag w[W - 1 - lag] g[t + lag] [valid], and the
    per-tile weight-gradient partials part[tile, W - 1 - lag, d] (fp32). A thread owns VEC = 8 channels (16-byte loads
    of x and out_grad) and walks UB = 4 rows per iteration; the fp32 math runs over two halves of 4 channels, the last
    3 x / g rows are carried as bf16. With ``softmax`` the raw weight [D, W] is normalized per channel in the kernel."""

    def __init__(self, act: bool, has_bos: bool, tile: int = T, softmax: bool = False, wdt=Float32):
        self.act, self.has_bos, self.tile, self.softmax, self.wdt = act, has_bos, tile, softmax, wdt
        assert tile % U == 0

    @cute.jit
    def __call__(self, tG: cute.Tensor, tX: cute.Tensor, tW: cute.Tensor, tBos: cute.Tensor, tDx: cute.Tensor, tPart: cute.Tensor,
                 B: Int32, L: Int32, D: Int32, stream: cuda.CUstream):
        n_tiles = (L + self.tile - 1) // self.tile
        self.kernel(tG.iterator, tX.iterator, tW.iterator, tBos.iterator, tDx.iterator, tPart.iterator, L, D, n_tiles).launch(
            grid=[(D + THREADS * VEC - 1) // (THREADS * VEC), n_tiles, B], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pG: cute.Pointer, pX: cute.Pointer, pW: cute.Pointer, pBos: cute.Pointer, pDx: cute.Pointer, pPart: cute.Pointer,
               L: Int32, D: Int32, n_tiles: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, b = cute.arch.block_idx()
        T = self.tile
        UB = U
        HB = VEC // 2
        c = (bx * THREADS + tidx) * VEC
        if c < D:
            t0 = by * T
            base = pX + Int64(b) * L * D + c
            gbase = pG + Int64(b) * L * D + c
            dbase = pDx + Int64(b) * L * D + c
            bbase = pBos + Int64(b) * L
            w = cute.make_rmem_tensor(cute.make_layout((WIDTH, VEC)), Float32)  # w[lag, k] = weight[W - 1 - lag, c + k]
            if cutlass.const_expr(self.softmax):
                wr = cute.make_tensor((pW + c * WIDTH).align(16), cute.make_layout(VEC * WIDTH)).load().to(Float32)
                for k in cutlass.range_constexpr(VEC):
                    m = wr[k * WIDTH]
                    for j in cutlass.range_constexpr(1, WIDTH):
                        m = cute.arch.fmax(m, wr[k * WIDTH + j])
                    ssum = Float32(0.0)
                    for j in cutlass.range_constexpr(WIDTH):
                        e = cute.math.exp2((wr[k * WIDTH + j] - m) * 1.4426950408889634, fastmath=True)
                        w[WIDTH - 1 - j, k] = e
                        ssum = ssum + e
                    inv = Float32(1.0) / ssum
                    for j in cutlass.range_constexpr(WIDTH):
                        w[j, k] = w[j, k] * inv
            else:
                for lag in cutlass.range_constexpr(WIDTH):
                    wl = cute.make_tensor((pW + ((WIDTH - 1 - lag) * D + c)).align(32), cute.make_layout(VEC)).load()
                    for k in cutlass.range_constexpr(VEC):
                        w[lag, k] = wl[k]
            dw = cute.make_rmem_tensor(cute.make_layout((WIDTH, VEC)), Float32)
            for lag in cutlass.range_constexpr(WIDTH):
                for k in cutlass.range_constexpr(VEC):
                    dw[lag, k] = 0.0
            # windows (bf16): xw[j] = x[t - 1 - j], gw[j] = g[t - 1 - j]; flags[j] = bos[t - 1 - j]
            xw = cute.make_rmem_tensor(cute.make_layout((HALO, VEC)), BFloat16)
            gw = cute.make_rmem_tensor(cute.make_layout((HALO, VEC)), BFloat16)
            flags = cute.make_rmem_tensor(cute.make_layout(HALO), Float32)
            for j in cutlass.range_constexpr(HALO):
                flags[j] = 0.0
                for k in cutlass.range_constexpr(VEC):
                    xw[j, k] = BFloat16(0.0)
                    gw[j, k] = BFloat16(0.0)
                tp = t0 - 1 - j
                if tp >= 0:
                    xr = cute.make_tensor((base + Int64(tp) * D).align(16), cute.make_layout(VEC)).load()
                    for k in cutlass.range_constexpr(VEC):
                        xw[j, k] = xr[k]
                    if cutlass.const_expr(self.has_bos):
                        flags[j] = cute.make_tensor(bbase + tp, cute.make_layout(1))[0].to(Float32)
            # rows t0 .. t0 + T + 2 in groups of UB (the last 3 are the halo: g only)
            for i4 in cutlass.range((T + WIDTH - 1 + UB - 1) // UB):
                tb = t0 + i4 * UB
                xn = cute.make_rmem_tensor(cute.make_layout((UB, VEC)), BFloat16)  # raw rows tb .. tb + 3
                gn = cute.make_rmem_tensor(cute.make_layout((UB, VEC)), BFloat16)
                fl = cute.make_rmem_tensor(cute.make_layout(WIDTH - 1 + UB + HALO), Float32)  # bos of rows tb - 3 .. tb + 6
                for j in cutlass.range_constexpr(HALO):
                    fl[j] = flags[HALO - 1 - j]
                for u in cutlass.range_constexpr(UB):
                    for k in cutlass.range_constexpr(VEC):
                        xn[u, k] = BFloat16(0.0)
                        gn[u, k] = BFloat16(0.0)
                    if tb + u < L:
                        xr = cute.make_tensor((base + Int64(tb + u) * D).align(16), cute.make_layout(VEC)).load()
                        gr = cute.make_tensor((gbase + Int64(tb + u) * D).align(16), cute.make_layout(VEC)).load()
                        for k in cutlass.range_constexpr(VEC):
                            xn[u, k] = xr[k]
                            gn[u, k] = gr[k]
                for u in cutlass.range_constexpr(UB + HALO):
                    tt = tb + u
                    f = Float32(0.0)
                    if cutlass.const_expr(self.has_bos):
                        tc = cutlass.select_(tt < L, tt, L - 1)
                        f = cutlass.select_(tt < L, cute.make_tensor(bbase + tc, cute.make_layout(1))[0].to(Float32), Float32(0.0))
                    fl[HALO + u] = f
                any_bos = Float32(0.0)
                for j in cutlass.range_constexpr(1, WIDTH - 1 + UB + HALO):
                    any_bos = any_bos + fl[j]
                dxo = cute.make_rmem_tensor(cute.make_layout((UB, VEC)), BFloat16)  # dx of rows tb - 3 .. tb
                for h in cutlass.range_constexpr(2):
                    xs = cute.make_rmem_tensor(cute.make_layout((WIDTH - 1 + UB, HB)), Float32)  # rows tb - 3 .. tb + 3
                    gs = cute.make_rmem_tensor(cute.make_layout((WIDTH - 1 + UB, HB)), Float32)
                    for j in cutlass.range_constexpr(HALO):
                        for k in cutlass.range_constexpr(HB):
                            xs[j, k] = xw[HALO - 1 - j, h * HB + k].to(Float32)
                            gs[j, k] = gw[HALO - 1 - j, h * HB + k].to(Float32)
                    for u in cutlass.range_constexpr(UB):
                        for k in cutlass.range_constexpr(HB):
                            xs[HALO + u, k] = xn[u, h * HB + k].to(Float32)
                            gs[HALO + u, k] = gn[u, h * HB + k].to(Float32)
                    if any_bos == 0.0:
                        # ---- no boundary near: g, dw, dx without masks
                        for u in cutlass.range_constexpr(UB):
                            j = HALO + u
                            in_tile = (i4 * UB + u) < T
                            for k2 in cutlass.range_constexpr(HB // 2):
                                k = 2 * k2
                                kk = h * HB + k
                                a0, a1 = cute.arch.mul_packed_f32x2((xs[j, k], xs[j, k + 1]), (w[0, kk], w[0, kk + 1]))
                                a0, a1 = cute.arch.fma_packed_f32x2((xs[j - 1, k], xs[j - 1, k + 1]), (w[1, kk], w[1, kk + 1]), (a0, a1))
                                a0, a1 = cute.arch.fma_packed_f32x2((xs[j - 2, k], xs[j - 2, k + 1]), (w[2, kk], w[2, kk + 1]), (a0, a1))
                                a0, a1 = cute.arch.fma_packed_f32x2((xs[j - 3, k], xs[j - 3, k + 1]), (w[3, kk], w[3, kk + 1]), (a0, a1))
                                if cutlass.const_expr(self.act):
                                    g0, g1 = cute.arch.mul_packed_f32x2((gs[j, k], gs[j, k + 1]), (_dsilu(a0), _dsilu(a1)))
                                    gs[j, k] = g0
                                    gs[j, k + 1] = g1
                            if in_tile:
                                for k2 in cutlass.range_constexpr(HB // 2):
                                    k = 2 * k2
                                    kk = h * HB + k
                                    for lag in cutlass.range_constexpr(WIDTH):
                                        dw[lag, kk], dw[lag, kk + 1] = cute.arch.fma_packed_f32x2(
                                            (gs[j, k], gs[j, k + 1]), (xs[j - lag, k], xs[j - lag, k + 1]), (dw[lag, kk], dw[lag, kk + 1]))
                        for u in cutlass.range_constexpr(UB):
                            for k2 in cutlass.range_constexpr(HB // 2):
                                k = 2 * k2
                                kk = h * HB + k
                                d0, d1 = cute.arch.mul_packed_f32x2((gs[u, k], gs[u, k + 1]), (w[0, kk], w[0, kk + 1]))
                                d0, d1 = cute.arch.fma_packed_f32x2((gs[u + 1, k], gs[u + 1, k + 1]), (w[1, kk], w[1, kk + 1]), (d0, d1))
                                d0, d1 = cute.arch.fma_packed_f32x2((gs[u + 2, k], gs[u + 2, k + 1]), (w[2, kk], w[2, kk + 1]), (d0, d1))
                                d0, d1 = cute.arch.fma_packed_f32x2((gs[u + 3, k], gs[u + 3, k + 1]), (w[3, kk], w[3, kk + 1]), (d0, d1))
                                dxo[u, kk] = BFloat16(d0)
                                dxo[u, kk + 1] = BFloat16(d1)
                    else:
                        # ---- masked path
                        for u in cutlass.range_constexpr(UB):
                            j = HALO + u
                            v1 = 1.0 - fl[j]
                            v2 = v1 * (1.0 - fl[j - 1])
                            v3 = v2 * (1.0 - fl[j - 2])
                            in_tile = (i4 * UB + u) < T
                            for k in cutlass.range_constexpr(HB):
                                kk = h * HB + k
                                a = xs[j, k] * w[0, kk] + xs[j - 1, k] * v1 * w[1, kk] + xs[j - 2, k] * v2 * w[2, kk] + xs[j - 3, k] * v3 * w[3, kk]
                                if cutlass.const_expr(self.act):
                                    gs[j, k] = gs[j, k] * _dsilu(a)
                            if in_tile:
                                for k in cutlass.range_constexpr(HB):
                                    kk = h * HB + k
                                    dw[0, kk] = dw[0, kk] + gs[j, k] * xs[j, k]
                                    dw[1, kk] = dw[1, kk] + gs[j, k] * xs[j - 1, k] * v1
                                    dw[2, kk] = dw[2, kk] + gs[j, k] * xs[j - 2, k] * v2
                                    dw[3, kk] = dw[3, kk] + gs[j, k] * xs[j - 3, k] * v3
                        for u in cutlass.range_constexpr(UB):
                            u1 = 1.0 - fl[u + 1]
                            u2 = u1 * (1.0 - fl[u + 2])
                            u3 = u2 * (1.0 - fl[u + 3])
                            for k in cutlass.range_constexpr(HB):
                                kk = h * HB + k
                                dxo[u, kk] = BFloat16(gs[u, k] * w[0, kk] + gs[u + 1, k] * w[1, kk] * u1 + gs[u + 2, k] * w[2, kk] * u2
                                                      + gs[u + 3, k] * w[3, kk] * u3)
                    for j in cutlass.range_constexpr(HALO):
                        for k in cutlass.range_constexpr(HB):
                            gw[j, h * HB + k] = BFloat16(gs[WIDTH - 1 + UB - 1 - j, k])
                for u in cutlass.range_constexpr(UB):
                    ts = tb + u - HALO
                    if ts >= t0 and ts < L:
                        o = cute.make_rmem_tensor(cute.make_layout(VEC), BFloat16)
                        for k in cutlass.range_constexpr(VEC):
                            o[k] = dxo[u, k]
                        cute.make_tensor((dbase + Int64(ts) * D).align(16), cute.make_layout(VEC)).store(o.load())
                for j in cutlass.range_constexpr(HALO):
                    flags[j] = fl[WIDTH - 1 + UB - 1 - j]
                    for k in cutlass.range_constexpr(VEC):
                        xw[j, k] = xn[UB - 1 - j, k]
            pp = pPart + (Int64(b * n_tiles + by) * WIDTH) * D + c
            for lag in cutlass.range_constexpr(WIDTH):
                o = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
                for k in cutlass.range_constexpr(VEC):
                    o[k] = dw[lag, k]
                cute.make_tensor((pp + (WIDTH - 1 - lag) * D).align(32), cute.make_layout(VEC)).store(o.load())


class ConvDwFinish:
    """dW [D, W] (``wdt``) from the summed partials s [W, D] fp32: with ``softmax`` the gradient through the per-channel
    softmax of the raw weight (p * (s - sum_j p_j s_j)), else a transpose + cast."""

    def __init__(self, softmax: bool, wdt=Float32):
        self.softmax, self.wdt = softmax, wdt

    @cute.jit
    def __call__(self, tS: cute.Tensor, tW: cute.Tensor, tDw: cute.Tensor, D: Int32, stream: cuda.CUstream):
        self.kernel(tS.iterator, tW.iterator, tDw.iterator, D).launch(grid=[(D + THREADS - 1) // THREADS, 1, 1], block=[THREADS, 1, 1],
                                                                       stream=stream)

    @cute.kernel
    def kernel(self, pS: cute.Pointer, pW: cute.Pointer, pDw: cute.Pointer, D: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        d = bx * THREADS + tidx
        if d < D:
            sv = cute.make_rmem_tensor(cute.make_layout(WIDTH), Float32)
            for j in cutlass.range_constexpr(WIDTH):
                sv[j] = cute.make_tensor(pS + (j * D + d), cute.make_layout(1))[0]
            o = cute.make_rmem_tensor(cute.make_layout(WIDTH), self.wdt)
            if cutlass.const_expr(self.softmax):
                wr = cute.make_tensor((pW + d * WIDTH).align(8), cute.make_layout(WIDTH)).load().to(Float32)
                m = wr[0]
                for j in cutlass.range_constexpr(1, WIDTH):
                    m = cute.arch.fmax(m, wr[j])
                pv = cute.make_rmem_tensor(cute.make_layout(WIDTH), Float32)
                ssum = Float32(0.0)
                for j in cutlass.range_constexpr(WIDTH):
                    e = cute.math.exp2((wr[j] - m) * 1.4426950408889634, fastmath=True)
                    pv[j] = e
                    ssum = ssum + e
                inv = Float32(1.0) / ssum
                dot = Float32(0.0)
                for j in cutlass.range_constexpr(WIDTH):
                    pv[j] = pv[j] * inv
                    dot = dot + pv[j] * sv[j]
                for j in cutlass.range_constexpr(WIDTH):
                    o[j] = (pv[j] * (sv[j] - dot)).to(self.wdt)
            else:
                for j in cutlass.range_constexpr(WIDTH):
                    o[j] = sv[j].to(self.wdt)
            cute.make_tensor((pDw + d * WIDTH).align(8), cute.make_layout(WIDTH)).store(o.load())


_BOS_CACHE = {}
_CUTE_DT = {torch.float32: Float32, torch.bfloat16: BFloat16}


def _bos_u8(bos_mask: Optional[torch.Tensor], x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """The bos mask as a flat uint8 tensor, cached per mask tensor (the three convs of every layer share it)."""
    if bos_mask is None:
        return torch.zeros(16, dtype=torch.uint8, device=x.device), False
    if torch.cuda.is_current_stream_capturing():  # a graph owns its copy (the cache entry could be evicted later)
        return bos_mask.to(torch.uint8).contiguous().reshape(-1), True
    key = (id(bos_mask), bos_mask._version, bos_mask.shape)
    hit = _BOS_CACHE.get("bos")
    if hit is not None and hit[0] == key:
        return hit[2], True
    u8 = bos_mask.to(torch.uint8).contiguous().reshape(-1)
    _BOS_CACHE["bos"] = (key, bos_mask, u8)
    return u8, True


def conv_fwd(x: torch.Tensor, weight: torch.Tensor, bos_mask: Optional[torch.Tensor], activation: Optional[str],
             softmax: bool = False) -> torch.Tensor:
    """out [B, L, D] bf16 of the causal conv of x [B, L, D] (bf16) with weight [D, W] (any float dtype, W = 4);
    ``softmax``: the taps are softmax-normalized per channel inside the kernel (weight normalization)."""
    B, L, D = x.shape
    x = x.contiguous()
    bos, has_bos = _bos_u8(bos_mask, x)
    act = activation in ("silu", "swish")
    out = torch.empty_like(x)
    if softmax:
        w = weight.contiguous()  # [D, W] raw
        wdt = _CUTE_DT[w.dtype]
        fn = _compiled(("conv_fwd_sm", act, has_bos, w.dtype), lambda: ConvFwd(act, has_bos, True, wdt), lambda: (
            _fake(BFloat16), _fake(wdt, 16), _fake(Uint8, 1), _fake(BFloat16), Int32(0), Int32(0), Int32(0)))
    else:
        w = weight.t().contiguous().to(torch.float32)  # [W, D]
        fn = _compiled(("conv_fwd", act, has_bos), lambda: ConvFwd(act, has_bos), lambda: (
            _fake(BFloat16), _fake(Float32, 32), _fake(Uint8, 1), _fake(BFloat16), Int32(0), Int32(0), Int32(0)))
    fn(x.reshape(-1), w.reshape(-1), bos, out.reshape(-1), B, L, D)
    return out


def _bwd_tile(B: int, L: int, D: int, device: torch.device) -> int:
    """Longer tiles (fewer weight-gradient partials) when the grid still fills the machine twice."""
    return 64 if B * ((L + 63) // 64) * ((D + THREADS * VEC - 1) // (THREADS * VEC)) >= 2 * _sm_count(device) else 32


_SM = {}


def _sm_count(device: torch.device) -> int:
    d = device.index if device.index is not None else torch.cuda.current_device()
    n = _SM.get(d)
    if n is None:
        n = _SM[d] = torch.cuda.get_device_properties(d).multi_processor_count
    return n


def conv_bwd(out_grad: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bos_mask: Optional[torch.Tensor], activation: Optional[str],
             softmax: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """(dx [B, L, D] bf16, dweight [D, W] in weight's dtype) of conv_fwd; ``softmax``: weight is the raw [D, W] weight
    whose per-channel softmax the forward used (the returned gradient is w.r.t. the raw weight)."""
    B, L, D = x.shape
    x = x.contiguous()
    bos, has_bos = _bos_u8(bos_mask, x)
    act = activation in ("silu", "swish")
    tile = _bwd_tile(B, L, D, x.device)
    n_tiles = (L + tile - 1) // tile
    dx = torch.empty_like(x)
    part = torch.empty(B * n_tiles, WIDTH, D, dtype=torch.float32, device=x.device)
    if softmax:
        w = weight.contiguous()  # raw [D, W]
        wdt = _CUTE_DT[w.dtype]
        fn = _compiled(("conv_bwd_sm", act, has_bos, tile, w.dtype), lambda: ConvBwd(act, has_bos, tile, True, wdt), lambda: (
            _fake(BFloat16), _fake(BFloat16), _fake(wdt, 16), _fake(Uint8, 1), _fake(BFloat16), _fake(Float32, 32), Int32(0), Int32(0), Int32(0)))
    else:
        w = weight.t().contiguous().to(torch.float32)  # [W, D]
        fn = _compiled(("conv_bwd", act, has_bos, tile), lambda: ConvBwd(act, has_bos, tile), lambda: (
            _fake(BFloat16), _fake(BFloat16), _fake(Float32, 32), _fake(Uint8, 1), _fake(BFloat16), _fake(Float32, 32), Int32(0), Int32(0), Int32(0)))
    fn(out_grad.contiguous().reshape(-1), x.reshape(-1), w.reshape(-1), bos, dx.reshape(-1), part.reshape(-1), B, L, D)
    ssum = part.sum(0)  # [W, D] fp32
    wdt = _CUTE_DT[weight.dtype]
    dweight = torch.empty(D, WIDTH, dtype=weight.dtype, device=x.device)
    fin = _compiled(("conv_dw_finish", softmax, weight.dtype), lambda: ConvDwFinish(softmax, wdt), lambda: (
        _fake(Float32, 4), _fake(wdt, 8), _fake(wdt, 8), Int32(0)))
    fin(ssum.reshape(-1), (w if softmax else dweight).reshape(-1), dweight.reshape(-1), D)  # tW unused without softmax
    return dx, dweight


def conv_bwd_preferred(x: torch.Tensor, weight: torch.Tensor, bias, initial_state, final_state_grad=None, out_grad=None) -> bool:
    """Whether conv_bwd should replace the Triton backward: supported (incl. a bf16 out_grad of x's shape) and small enough
    that the Triton grid is underfilled."""
    if os.environ.get("XLLM_CUTE_CONV_BWD", "1") == "0" or x.numel() > 2 ** 24:
        return False
    if out_grad is not None and (out_grad.dtype != torch.bfloat16 or out_grad.shape != x.shape):
        return False
    return conv_supported(x, weight, bias, initial_state, False, final_state_grad)


def conv_supported(x: torch.Tensor, weight: torch.Tensor, bias, initial_state, output_final_state: bool, final_state_grad=None) -> bool:
    if os.environ.get("XLLM_CUTE_CONV", "1") == "0" or x.dtype != torch.bfloat16 or not x.is_cuda or x.dim() != 3:
        return False
    if bias is not None or initial_state is not None or output_final_state or final_state_grad is not None:
        return False
    if weight.shape != (x.shape[-1], WIDTH) or x.shape[-1] % VEC or x.shape[-1] % 4 or x.numel() >= 2 ** 31:
        return False
    if weight.dtype not in _CUTE_DT:
        return False
    return torch.cuda.get_device_capability(x.device)[0] >= 10
