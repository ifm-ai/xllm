"""Causal conv1d fused with the q / k preparation of the Gekko block (CuTe DSL, sm_90+): per head row of S channels

    forward:  c = bf16(conv(x)), y = rope(rmsnorm(c) * w), p = softmax(c)                         (+ c, optional)
    backward: g = bf16(qk_prep_bwd(dy, dp, extra, c)), dx = conv_bwd(g), the conv weight gradient and the norm
              weight gradient; c is recomputed from x in registers

replacing causal_conv1d_fwd + qk_prep_fwd (conv output written and read back) and qk_prep_bwd + causal_conv1d_bwd
(conv output gradient written and read back, plus a conv recompute for the input of qk_prep_bwd). The conv is the
depthwise width-4 causal conv of causal_conv.py (no bias / state / activation; segment resets from the bos mask):

    c[t, d] = sum_lag w[d, W - 1 - lag] x[t - lag, d] [no bos in (t - lag, t]]

A warp owns one head (S / 32 channels per lane, as in qk_prep.py) of a tile of T timesteps and walks it U rows per
iteration, the last 3 x rows in registers (forward ascending; backward descending, recomputing c and the q / k
gradient of the 3 rows after the tile for the conv's look-ahead). Each iteration's phases run over its U rows at
once, so the warp reductions of the rows overlap, and its inputs arrive by cp.async through a per-warp shared
memory ring NST - 1 iterations ahead (each lane reads back only what it copied: no barriers). The warps of a CTA
take consecutive tiles of one head, so the weight-gradient partials are summed per CTA in shared memory.
Roundings match the unfused kernels: c and g are rounded to bf16 where those kernels store them."""
import os
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8
from cutlass.cute.nvgpu import cpasync

try:
    from .adaptive_working_memory import _compiled, _fake
    from .qk_prep import LOG2E, _dummy
except ImportError:  # loaded from its file (tests)
    import importlib.util as _ilu
    _d = os.path.dirname(os.path.abspath(__file__))
    _spec = _ilu.spec_from_file_location("cute_adaptive_working_memory_fwd", os.path.join(_d, "adaptive_working_memory.py"))
    _awm = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_awm)
    _compiled, _fake = _awm._compiled, _awm._fake
    _spec = _ilu.spec_from_file_location("cute_qk_prep", os.path.join(_d, "qk_prep.py"))
    _qk = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_qk)
    LOG2E, _dummy = _qk.LOG2E, _qk._dummy

WARPS = 8
WIDTH = 4
HALO = WIDTH - 1


def _rows_per_warp(B: int, L: int, G: int) -> int:
    """Tile length T: ~2048 warps per launch, 16..128 rows (XLLM_CONV_QK_T overrides)."""
    t = int(os.environ.get("XLLM_CONV_QK_T", "0"))
    if t <= 0:
        t = B * L * G // 2048
        t = max(16, min(128, 1 << max(0, (t - 1).bit_length())))
    return t


BOS_PAD, BOS_PAD_R = 8, 136  # zero bos entries before / after each sequence (>= 3 / T + 3: no bounds checks)


def _heads_per_warp(S: int, G: int) -> int:
    """Heads per warp: up to 256 channels, a power of two dividing G."""
    hp = 256 // S if S < 256 and 256 % S == 0 else 1
    while G % hp:
        hp //= 2
    return hp


@cute.jit
def _warp_reduce_n(v: cute.Tensor, n_sum: cutlass.Constexpr, n_max: cutlass.Constexpr, rounds: cutlass.Constexpr = 5):
    """Butterfly all-reduce of v[0 .. n_sum) (sums) and v[n_sum .. n_sum + n_max) (maxima) across groups of
    2^rounds lanes, the independent chains interleaved (one shuffle latency per round instead of one per value)."""
    for o in cutlass.range_constexpr(rounds):
        for i in cutlass.range_constexpr(n_sum):
            v[i] = v[i] + cute.arch.shuffle_sync_bfly(v[i], offset=1 << o)
        for i in cutlass.range_constexpr(n_sum, n_sum + n_max):
            v[i] = cute.arch.fmax(v[i], cute.arch.shuffle_sync_bfly(v[i], offset=1 << o))


class _ConvQkBase:
    def __init__(self, G: int, S: int, R: int, T: int, U: int, NST: int, affine: bool, has_bos: bool, normalize: bool):
        # heads narrower than 256 channels are packed HP to a warp (G / HP groups of S * HP channels, S / E lanes per
        # head) so each lane still owns 8 channels and the per-head reductions run over half / quarter warps
        HP = _heads_per_warp(S, G)
        self.SH, self.LH = S, 32 // HP
        self.G, self.S, self.R, self.T, self.U, self.NST, self.affine, self.has_bos = G // HP, S * HP, R, T, U, NST, affine, has_bos
        self.normalize = normalize
        self.E = self.S // 32
        self.RED = self.LH.bit_length() - 1  # butterfly rounds of a head's lanes
        assert self.S % 32 == 0 and self.E in (2, 4, 8, 16) and R % self.E == 0 and 0 <= R <= S and NST >= 2 and T <= BOS_PAD_R - HALO

    @cute.jit
    def _weights(self, pWc: cute.Pointer, pW: cute.Pointer, c0: Int32, wc: cute.Tensor, wv: cute.Tensor):
        """wc[lag, k] = conv tap W - 1 - lag of channel c0 + k (softmax over the taps if normalize) from the fp32
        [D, W] weight; wv[k] = bf16(norm weight + 1) (affine) or 1."""
        E = self.E
        wr = cute.make_tensor((pWc + c0 * WIDTH).align(16), cute.make_layout(WIDTH * E)).load()
        for k in cutlass.range_constexpr(E):
            if cutlass.const_expr(self.normalize):
                m = cute.arch.fmax(cute.arch.fmax(wr[k * WIDTH], wr[k * WIDTH + 1]), cute.arch.fmax(wr[k * WIDTH + 2], wr[k * WIDTH + 3]))
                e = cute.make_rmem_tensor(cute.make_layout(WIDTH), Float32)
                for i in cutlass.range_constexpr(WIDTH):
                    e[i] = cute.math.exp(wr[k * WIDTH + i] - m)
                inv = 1.0 / (e[0] + e[1] + e[2] + e[3])
                for lag in cutlass.range_constexpr(WIDTH):
                    wc[lag, k] = e[WIDTH - 1 - lag] * inv
            else:
                for lag in cutlass.range_constexpr(WIDTH):
                    wc[lag, k] = wr[k * WIDTH + WIDTH - 1 - lag]
            wv[k] = 1.0
        if cutlass.const_expr(self.affine):
            wn = cute.make_tensor((pW + c0).align(min(16, 4 * E)), cute.make_layout(E)).load()
            wb = cute.make_rmem_tensor(cute.make_layout(E), Float32)
            for k in cutlass.range_constexpr(E):
                wb[k] = wn[k] + 1.0
            wv.store(wb.load().to(BFloat16).to(Float32))

    @cute.jit
    def _cp(self, src: cute.Pointer, dst: cute.Pointer, dtype: cutlass.Constexpr):
        """cp.async of this lane's E elements (bf16 or fp32) from global to shared memory."""
        nbytes = self.E * dtype.width // 8
        per = min(16, nbytes)
        ne = per * 8 // dtype.width
        atom = cute.make_copy_atom(cpasync.CopyG2SOp(), dtype, num_bits_per_copy=per * 8)
        for i in cutlass.range_constexpr(nbytes // per):
            cute.copy(atom, cute.make_tensor((src + i * ne).align(per), cute.make_layout(ne)),
                      cute.make_tensor((dst + i * ne).align(per), cute.make_layout(ne)))

    @cute.jit
    def _zero(self, p: cute.Pointer, dtype: cutlass.Constexpr = BFloat16):
        """Zeros into this lane's E elements (bf16 or fp32) of a ring slot (rows outside the sequence)."""
        z = cute.make_rmem_tensor(cute.make_layout(self.E), dtype)
        z.fill(0.0)
        cute.make_tensor(p.align(min(16, self.E * dtype.width // 8)), cute.make_layout(self.E)).store(z.load())

    @cute.jit
    def _ld(self, p: cute.Pointer, dtype: cutlass.Constexpr):
        return cute.make_tensor(p.align(min(16, self.E * dtype.width // 8)), cute.make_layout(self.E)).load().to(Float32)


class ConvQkFwd(_ConvQkBase):
    """c, y, p of one tile: x [B, L, G * S] bf16, conv weight [G * S, W] fp32, norm weight [G * S] fp32 (+ 1 in bf16)."""

    def __init__(self, G: int, S: int, R: int, T: int, affine: bool, has_bos: bool, normalize: bool, want_c: bool, U: int = 4,
                 NST: int = 2):
        super().__init__(G, S, R, T, U, NST, affine, has_bos, normalize)
        self.want_c = want_c
        assert T % U == 0

    @cute.jit
    def __call__(self, tX: cute.Tensor, tWc: cute.Tensor, tBos: cute.Tensor, tF: cute.Tensor, tW: cute.Tensor, tC: cute.Tensor,
                 tY: cute.Tensor, tSm: cute.Tensor, B: Int32, L: Int32, eps: Float32, stream: cuda.CUstream):
        n_tiles = (L + self.T - 1) // self.T
        n_tg = (n_tiles + WARPS - 1) // WARPS
        self.kernel(tX.iterator, tWc.iterator, tBos.iterator, tF.iterator, tW.iterator, tC.iterator, tY.iterator, tSm.iterator,
                    n_tiles, L, eps).launch(grid=[n_tg * self.G, B, 1], block=[32 * WARPS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pX: cute.Pointer, pWc: cute.Pointer, pBos: cute.Pointer, pF: cute.Pointer, pW: cute.Pointer, pC: cute.Pointer,
               pY: cute.Pointer, pSm: cute.Pointer, n_tiles: Int32, L: Int32, eps: Float32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, b, _ = cute.arch.block_idx()
        G, S, R, E, T, U, NST = self.G, self.S, self.R, self.E, self.T, self.U, self.NST
        D = G * S
        lane = tidx % 32
        warp = tidx // 32
        g = bidx % G
        tile = (bidx // G) * WARPS + warp
        smem = cutlass.utils.SmemAllocator()
        n_slot = WARPS * NST * U * 32 * E
        sX = smem.allocate_tensor(BFloat16, cute.make_layout(n_slot), 16)
        sF = smem.allocate_tensor(Float32, cute.make_layout(n_slot if R > 0 else 4), 16)
        if tile < n_tiles:
            t0 = tile * T
            c0 = g * S + lane * E
            base = pX + Int64(b) * L * D + c0
            obase = Int64(b) * L * D + c0
            bbase = pBos + (Int64(b) * (L + BOS_PAD + BOS_PAD_R) + BOS_PAD)
            rope = (lane % self.LH) * E < R
            n_groups = T // U
            wslot = warp * NST

            for j in cutlass.range_constexpr(NST - 1):
                self.issue(j, n_groups, t0, L, D, base, pF, sX, sF, wslot, lane, rope)
            wc = cute.make_rmem_tensor(cute.make_layout((WIDTH, E)), Float32)  # wc[lag] = weight[:, W - 1 - lag]
            wv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
            self._weights(pWc, pW, c0, wc, wv)
            # xa[m] = x[tb - 3 + m] (m = 0..2: the previous rows, 3.. : rows tb .. tb + U - 1); fa: bos of the same rows
            xa = cute.make_rmem_tensor(cute.make_layout((HALO + U, E)), Float32)
            fa = cute.make_rmem_tensor(cute.make_layout(HALO + U), Float32)
            for j in cutlass.range_constexpr(HALO):
                tp = t0 - HALO + j
                fa[j] = 0.0
                for k in cutlass.range_constexpr(E):
                    xa[j, k] = 0.0
                if tp >= 0:
                    xr = cute.make_tensor((base + Int64(tp) * D).align(2 * E), cute.make_layout(E)).load().to(Float32)
                    for k in cutlass.range_constexpr(E):
                        xa[j, k] = xr[k]
                    if cutlass.const_expr(self.has_bos):
                        fa[j] = cute.make_tensor(bbase + tp, cute.make_layout(1))[0].to(Float32)
            cv = cute.make_rmem_tensor(cute.make_layout((U, E)), Float32)
            red = cute.make_rmem_tensor(cute.make_layout(2 * U), Float32)
            for it in cutlass.range(n_groups):
                tb = t0 + it * U
                so = ((wslot + it % NST) * U * 32 + lane) * E
                cute.arch.cp_async_wait_group(NST - 2)
                self.issue(it + NST - 1, n_groups, t0, L, D, base, pF, sX, sF, wslot, lane, rope)
                for u in cutlass.range_constexpr(U):
                    xr = self._ld(sX.iterator + (so + u * 32 * E), BFloat16)
                    for k in cutlass.range_constexpr(E):
                        xa[HALO + u, k] = xr[k]
                    fa[HALO + u] = 0.0
                    if cutlass.const_expr(self.has_bos):
                        fa[HALO + u] = cute.make_tensor(bbase + (tb + u), cute.make_layout(1))[0].to(Float32)
                # conv, rounded to bf16 as the unfused conv stores it; sum of squares and max per row
                for u in cutlass.range_constexpr(U):
                    j = HALO + u
                    v1 = 1.0 - fa[j]
                    v2 = v1 * (1.0 - fa[j - 1])
                    v3 = v2 * (1.0 - fa[j - 2])
                    cf = cute.make_rmem_tensor(cute.make_layout(E), Float32)
                    for k in cutlass.range_constexpr(E):
                        cf[k] = xa[j, k] * wc[0, k] + (xa[j - 1, k] * v1) * wc[1, k] + (xa[j - 2, k] * v2) * wc[2, k] + (xa[j - 3, k] * v3) * wc[3, k]
                    cb = cf.load().to(BFloat16)
                    if cutlass.const_expr(self.want_c):
                        if tb + u < L:
                            cute.make_tensor((pC + (obase + Int64(tb + u) * D)).align(2 * E), cute.make_layout(E)).store(cb)
                    cr = cb.to(Float32)
                    ss = Float32(0.0)
                    mx = Float32(-3.0e38)
                    for k in cutlass.range_constexpr(E):
                        cv[u, k] = cr[k]
                        ss = ss + cr[k] * cr[k]
                        mx = cute.arch.fmax(mx, cr[k])
                    red[u] = ss
                    red[U + u] = mx
                _warp_reduce_n(red, U, U, self.RED)
                # y = rope(bf16(rmsnorm(c) w)); exp for the softmax (in place of c)
                for u in cutlass.range_constexpr(U):
                    rstd = cute.math.rsqrt(red[u] * (1.0 / self.SH) + eps, fastmath=True)
                    yv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
                    for k in cutlass.range_constexpr(E):
                        yv[k] = cv[u, k] * rstd * wv[k]
                    yb = yv.load().to(BFloat16)  # the torch path rounds before the rotation
                    if tb + u < L:
                        yo = cute.make_tensor((pY + (obase + Int64(tb + u) * D)).align(2 * E), cute.make_layout(E))
                        if cutlass.const_expr(R > 0):
                            if rope:
                                yr = yb.to(Float32)
                                fr = self._ld(sF.iterator + (so + u * 32 * E), Float32)
                                ov = cute.make_rmem_tensor(cute.make_layout(E), Float32)
                                for i in cutlass.range_constexpr(E // 2):
                                    x0, x1, c, s = yr[2 * i], yr[2 * i + 1], fr[2 * i], fr[2 * i + 1]
                                    ov[2 * i] = x0 * c - x1 * s
                                    ov[2 * i + 1] = x0 * s + x1 * c
                                yo.store(ov.load().to(BFloat16))
                            else:
                                yo.store(yb)
                        else:
                            yo.store(yb)
                    neg = -red[U + u] * LOG2E
                    tsum = Float32(0.0)
                    for k in cutlass.range_constexpr(E):
                        cv[u, k] = cute.math.exp2(cv[u, k] * LOG2E + neg, fastmath=True)
                        tsum = tsum + cv[u, k]
                    red[u] = tsum
                _warp_reduce_n(red, U, 0, self.RED)
                for u in cutlass.range_constexpr(U):
                    inv = cute.arch.rcp_approx(red[u])
                    pv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
                    for k in cutlass.range_constexpr(E):
                        pv[k] = cv[u, k] * inv
                    if tb + u < L:
                        cute.make_tensor((pSm + (obase + Int64(tb + u) * D)).align(2 * E), cute.make_layout(E)).store(pv.load().to(BFloat16))
                for j in cutlass.range_constexpr(HALO):
                    fa[j] = fa[U + j]
                    for k in cutlass.range_constexpr(E):
                        xa[j, k] = xa[U + j, k]
            cute.arch.cp_async_wait_group(0)

    @cute.jit
    def issue(self, j: Int32, n_groups: Int32, t0: Int32, L: Int32, D: cutlass.Constexpr, base: cute.Pointer, pF: cute.Pointer,
              sX: cute.Tensor, sF: cute.Tensor, wslot: Int32, lane: Int32, rope):
        """cp.async of group j's x rows (and rope table rows) into ring slot j % NST; always one commit group."""
        E, U, R = self.E, self.U, self.R
        if j < n_groups:
            so = ((wslot + j % self.NST) * U * 32 + lane) * E
            for u in cutlass.range_constexpr(U):
                t = t0 + j * U + u
                tt = cutlass.select_(t < L, t, L - 1)
                self._cp(base + Int64(tt) * D, sX.iterator + (so + u * 32 * E), BFloat16)
                if cutlass.const_expr(R > 0):
                    if rope:
                        self._cp(pF + (Int64(tt) * R + (lane % self.LH) * E), sF.iterator + (so + u * 32 * E), Float32)
        cute.arch.cp_async_commit_group()


class ConvQkBwd(_ConvQkBase):
    """dx of one tile and the weight-gradient sums: rows t0 + T + 3 .. t0 in groups of U, descending (the first 3 rows
    after the tile only give the look-ahead gradient). part[cta, W + 1, S]: the conv weight sums (tap-major) and the
    norm weight sums (affine) of the CTA's tiles."""

    def __init__(self, G: int, S: int, R: int, T: int, affine: bool, has_bos: bool, normalize: bool, has_extra: bool, U: int = 2,
                 NST: int = 2):
        super().__init__(G, S, R, T, U, NST, affine, has_bos, normalize)
        self.has_extra = has_extra
        assert T % U == 0 and (T + HALO + 1) % U == 0

    @cute.jit
    def __call__(self, tDy: cute.Tensor, tDp: cute.Tensor, tEx: cute.Tensor, tX: cute.Tensor, tWc: cute.Tensor, tBos: cute.Tensor,
                 tF: cute.Tensor, tW: cute.Tensor, tDx: cute.Tensor, tPart: cute.Tensor, B: Int32, L: Int32, eps: Float32,
                 stream: cuda.CUstream):
        n_tiles = (L + self.T - 1) // self.T
        n_tg = (n_tiles + WARPS - 1) // WARPS
        self.kernel(tDy.iterator, tDp.iterator, tEx.iterator, tX.iterator, tWc.iterator, tBos.iterator, tF.iterator, tW.iterator,
                    tDx.iterator, tPart.iterator, n_tiles, L, eps).launch(
            grid=[n_tg * self.G, B, 1], block=[32 * WARPS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pDy: cute.Pointer, pDp: cute.Pointer, pEx: cute.Pointer, pX: cute.Pointer, pWc: cute.Pointer, pBos: cute.Pointer,
               pF: cute.Pointer, pW: cute.Pointer, pDx: cute.Pointer, pPart: cute.Pointer, n_tiles: Int32, L: Int32, eps: Float32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, b, _ = cute.arch.block_idx()
        nb, _, _ = cute.arch.grid_dim()
        G, S, R, E, T, U, NST = self.G, self.S, self.R, self.E, self.T, self.U, self.NST
        D = G * S
        lane = tidx % 32
        warp = tidx // 32
        g = bidx % G
        tile = (bidx // G) * WARPS + warp
        smem = cutlass.utils.SmemAllocator()
        n_slot = WARPS * NST * U * 32 * E
        sX = smem.allocate_tensor(BFloat16, cute.make_layout(n_slot), 16)
        sDy = smem.allocate_tensor(BFloat16, cute.make_layout(n_slot), 16)
        sDp = smem.allocate_tensor(BFloat16, cute.make_layout(n_slot), 16)
        sEx = smem.allocate_tensor(BFloat16, cute.make_layout(n_slot if self.has_extra else 8), 16)
        sF = smem.allocate_tensor(Float32, cute.make_layout(n_slot if R > 0 else 4), 16)
        sRed = smem.allocate_tensor(Float32, cute.make_layout(WARPS * (WIDTH + 1) * S), 128)
        c0 = g * S + lane * E
        accn = cute.make_rmem_tensor(cute.make_layout(E), Float32)
        accc = cute.make_rmem_tensor(cute.make_layout((WIDTH, E)), Float32)
        for k in cutlass.range_constexpr(E):
            accn[k] = 0.0
            for lag in cutlass.range_constexpr(WIDTH):
                accc[lag, k] = 0.0
        if tile < n_tiles:
            t0 = tile * T
            base = pX + Int64(b) * L * D + c0
            obase = Int64(b) * L * D + c0
            bbase = pBos + (Int64(b) * (L + BOS_PAD + BOS_PAD_R) + BOS_PAD)
            rope = (lane % self.LH) * E < R
            tb0 = t0 + T + HALO
            n_groups = (T + HALO + 1) // U
            wslot = warp * NST
            bufs = (sX, sDy, sDp, sEx, sF)

            for j in cutlass.range_constexpr(NST - 1):
                self.issue(j, n_groups, tb0, L, D, base, obase, pDy, pDp, pEx, pF, bufs, wslot, lane, rope)
            wc = cute.make_rmem_tensor(cute.make_layout((WIDTH, E)), Float32)  # wc[lag] = weight[:, W - 1 - lag]
            wv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
            self._weights(pWc, pW, c0, wc, wv)
            # descending groups: rows r_u = tb - u. xa[m] = x[tb - m] (m = 0..2 kept from the previous group, the rest new);
            # ga[m] = g[tb + 3 - m] (m = 0..2 kept, then rows tb, tb - 1, ..); fa[m] = bos[tb + 3 - m]
            xa = cute.make_rmem_tensor(cute.make_layout((HALO + U, E)), Float32)
            ga = cute.make_rmem_tensor(cute.make_layout((HALO + U, E)), Float32)
            fa = cute.make_rmem_tensor(cute.make_layout(U + 5), Float32)
            for j in cutlass.range_constexpr(HALO):
                for k in cutlass.range_constexpr(E):
                    xa[j, k] = 0.0
                    ga[j, k] = 0.0
                tp = tb0 - j
                if tp < L:
                    xr = cute.make_tensor((base + Int64(tp) * D).align(2 * E), cute.make_layout(E)).load().to(Float32)
                    for k in cutlass.range_constexpr(E):
                        xa[j, k] = xr[k]
            cv = cute.make_rmem_tensor(cute.make_layout((U, E)), Float32)
            gy = cute.make_rmem_tensor(cute.make_layout((U, E)), Float32)
            pv = cute.make_rmem_tensor(cute.make_layout((U, E)), Float32)
            red = cute.make_rmem_tensor(cute.make_layout(2 * U), Float32)
            rs = cute.make_rmem_tensor(cute.make_layout(U), Float32)
            for it in cutlass.range(n_groups):
                tb = tb0 - it * U
                so = ((wslot + it % NST) * U * 32 + lane) * E
                cute.arch.cp_async_wait_group(NST - 2)
                self.issue(it + NST - 1, n_groups, tb0, L, D, base, obase, pDy, pDp, pEx, pF, bufs, wslot, lane, rope)
                nbos = Float32(0.0)
                for m in cutlass.range_constexpr(U + 5):
                    fa[m] = 0.0
                    if cutlass.const_expr(self.has_bos):
                        fa[m] = cute.make_tensor(bbase + (tb + HALO - m), cute.make_layout(1))[0].to(Float32)
                        nbos = nbos + fa[m]
                # no segment start within the group's conv windows (the common case): the code without masks
                if nbos == 0.0:
                    self.group(False, tb, t0, L, so, fa, xa, ga, cv, gy, pv, red, rs, wc, wv, accn, accc, bufs, pDx, obase, eps, rope)
                else:
                    self.group(True, tb, t0, L, so, fa, xa, ga, cv, gy, pv, red, rs, wc, wv, accn, accc, bufs, pDx, obase, eps, rope)
                for j in cutlass.range_constexpr(HALO):
                    for k in cutlass.range_constexpr(E):
                        xa[j, k] = xa[U + j, k]
                        ga[j, k] = ga[U + j, k]
            cute.arch.cp_async_wait_group(0)
        # the CTA's weight-gradient sums: per-warp rows in shared memory, summed by all threads
        rb = warp * (WIDTH + 1) * S + lane * E
        for lag in cutlass.range_constexpr(WIDTH):
            o = cute.make_rmem_tensor(cute.make_layout(E), Float32)
            for k in cutlass.range_constexpr(E):
                o[k] = accc[lag, k]
            cute.make_tensor((sRed.iterator + (rb + (WIDTH - 1 - lag) * S)).align(min(16, 4 * E)), cute.make_layout(E)).store(o.load())
        cute.make_tensor((sRed.iterator + (rb + WIDTH * S)).align(min(16, 4 * E)), cute.make_layout(E)).store(accn.load())
        cute.arch.barrier()
        n_red = (WIDTH + 1) * S if self.affine else WIDTH * S
        pbase = pPart + (Int64(b) * nb + bidx) * ((WIDTH + 1) * S)
        for e in cutlass.range(tidx, n_red, 32 * WARPS):
            s = Float32(0.0)
            for w in cutlass.range_constexpr(WARPS):
                s = s + sRed[w * (WIDTH + 1) * S + e]
            cute.make_tensor(pbase + e, cute.make_layout(1))[0] = s

    @cute.jit
    def group(self, masked: cutlass.Constexpr, tb: Int32, t0: Int32, L: Int32, so: Int32, fa: cute.Tensor, xa: cute.Tensor,
              ga: cute.Tensor, cv: cute.Tensor, gy: cute.Tensor, pv: cute.Tensor, red: cute.Tensor, rs: cute.Tensor, wc: cute.Tensor,
              wv: cute.Tensor, accn: cute.Tensor, accc: cute.Tensor, bufs, pDx: cute.Pointer, obase: Int64, eps: Float32, rope):
        """One group of U rows (r_u = tb - u) from the landed ring slot so: g, the weight sums and dx. masked: a segment
        start is within the group's conv windows (fa holds bos[tb + 3 - m])."""
        S, R, E, T, U = self.S, self.R, self.E, self.T, self.U
        D = self.G * S
        sX, sDy, sDp, sEx, sF = bufs
        # conv recompute (bf16-rounded), sum of squares and max per row; rope backward of dy
        for u in cutlass.range_constexpr(U):
            xr = self._ld(sX.iterator + (so + u * 32 * E), BFloat16)
            for k in cutlass.range_constexpr(E):
                xa[HALO + u, k] = xr[k]
            gr = self._ld(sDy.iterator + (so + u * 32 * E), BFloat16)
            for k in cutlass.range_constexpr(E):
                gy[u, k] = gr[k]
            cf = cute.make_rmem_tensor(cute.make_layout(E), Float32)
            if cutlass.const_expr(masked):
                v1 = 1.0 - fa[HALO + u]
                v2 = v1 * (1.0 - fa[HALO + u + 1])
                v3 = v2 * (1.0 - fa[HALO + u + 2])
                for k in cutlass.range_constexpr(E):
                    cf[k] = xa[u, k] * wc[0, k] + (xa[u + 1, k] * v1) * wc[1, k] + (xa[u + 2, k] * v2) * wc[2, k] + (xa[u + 3, k] * v3) * wc[3, k]
            else:
                for k in cutlass.range_constexpr(E):
                    cf[k] = xa[u, k] * wc[0, k] + xa[u + 1, k] * wc[1, k] + xa[u + 2, k] * wc[2, k] + xa[u + 3, k] * wc[3, k]
            cr = cf.load().to(BFloat16).to(Float32)
            ss = Float32(0.0)
            mx = Float32(-3.0e38)
            for k in cutlass.range_constexpr(E):
                cv[u, k] = cr[k]
                ss = ss + cr[k] * cr[k]
                mx = cute.arch.fmax(mx, cr[k])
            red[u] = ss
            red[U + u] = mx
            if cutlass.const_expr(R > 0):
                if rope:
                    fr = self._ld(sF.iterator + (so + u * 32 * E), Float32)
                    for i in cutlass.range_constexpr(E // 2):
                        x0, x1, c, s = gy[u, 2 * i], gy[u, 2 * i + 1], fr[2 * i], fr[2 * i + 1]
                        gy[u, 2 * i] = x0 * c + x1 * s
                        gy[u, 2 * i + 1] = x1 * c - x0 * s
        _warp_reduce_n(red, U, U, self.RED)
        # rmsnorm backward dot, softmax exponentials; c becomes xhat = c rstd
        for u in cutlass.range_constexpr(U):
            rstd = cute.math.rsqrt(red[u] * (1.0 / self.SH) + eps, fastmath=True)
            rs[u] = rstd
            in_tile = tb - u < t0 + T
            neg = -red[U + u] * LOG2E
            dot = Float32(0.0)
            tsum = Float32(0.0)
            for k in cutlass.range_constexpr(E):
                pv[u, k] = cute.math.exp2(cv[u, k] * LOG2E + neg, fastmath=True)
                tsum = tsum + pv[u, k]
                xh = cv[u, k] * rstd
                cv[u, k] = xh
                if cutlass.const_expr(self.affine):
                    if in_tile:
                        accn[k] = accn[k] + gy[u, k] * xh
                gy[u, k] = gy[u, k] * wv[k]
                dot = dot + gy[u, k] * xh
            red[u] = dot
            red[U + u] = tsum
        _warp_reduce_n(red, 2 * U, 0, self.RED)
        # norm gradient rstd (dxhat - xhat mean(dxhat xhat)); softmax probabilities and their dot with dp
        for u in cutlass.range_constexpr(U):
            rstd = rs[u]
            dot = red[u] * (1.0 / self.SH)
            inv = cute.arch.rcp_approx(red[U + u])
            gp = self._ld(sDp.iterator + (so + u * 32 * E), BFloat16)
            sdot = Float32(0.0)
            for k in cutlass.range_constexpr(E):
                gy[u, k] = rstd * (gy[u, k] - cv[u, k] * dot)
                pv[u, k] = pv[u, k] * inv
                sdot = sdot + pv[u, k] * gp[k]
            red[u] = sdot
        _warp_reduce_n(red, U, 0, self.RED)
        # g = bf16(norm gradient + softmax gradient + extra) (the unfused path stores it in bf16); conv weight sums
        for u in cutlass.range_constexpr(U):
            r = tb - u
            gp = self._ld(sDp.iterator + (so + u * 32 * E), BFloat16)
            gf = cute.make_rmem_tensor(cute.make_layout(E), Float32)
            for k in cutlass.range_constexpr(E):
                gf[k] = gy[u, k] + pv[u, k] * (gp[k] - red[u])
            if cutlass.const_expr(self.has_extra):
                er = self._ld(sEx.iterator + (so + u * 32 * E), BFloat16)
                for k in cutlass.range_constexpr(E):
                    gf[k] = gf[k] + er[k]
            gb = gf.load().to(BFloat16).to(Float32)
            for k in cutlass.range_constexpr(E):
                ga[HALO + u, k] = gb[k]
            if r < t0 + T:
                if r < L:
                    if cutlass.const_expr(masked):
                        v1 = 1.0 - fa[HALO + u]
                        v2 = v1 * (1.0 - fa[HALO + u + 1])
                        v3 = v2 * (1.0 - fa[HALO + u + 2])
                        for k in cutlass.range_constexpr(E):
                            gk = ga[HALO + u, k]
                            accc[0, k] = accc[0, k] + gk * xa[u, k]
                            accc[1, k] = accc[1, k] + gk * xa[u + 1, k] * v1
                            accc[2, k] = accc[2, k] + gk * xa[u + 2, k] * v2
                            accc[3, k] = accc[3, k] + gk * xa[u + 3, k] * v3
                    else:
                        for k in cutlass.range_constexpr(E):
                            gk = ga[HALO + u, k]
                            for lag in cutlass.range_constexpr(WIDTH):
                                accc[lag, k] = accc[lag, k] + gk * xa[u + lag, k]
        # dx[r_u] = sum_lag w[lag] g[r_u + lag] [no bos in (r_u, r_u + lag]]
        for u in cutlass.range_constexpr(U):
            r = tb - u
            if r < t0 + T:
                if r < L:
                    dx = cute.make_rmem_tensor(cute.make_layout(E), Float32)
                    if cutlass.const_expr(masked):
                        u1 = 1.0 - fa[HALO + u - 1]
                        u2 = u1 * (1.0 - fa[HALO + u - 2])
                        u3 = u2 * (1.0 - fa[HALO + u - 3])
                        for k in cutlass.range_constexpr(E):
                            dx[k] = (ga[HALO + u, k] * wc[0, k] + (ga[HALO + u - 1, k] * u1) * wc[1, k]
                                     + (ga[HALO + u - 2, k] * u2) * wc[2, k] + (ga[HALO + u - 3, k] * u3) * wc[3, k])
                    else:
                        for k in cutlass.range_constexpr(E):
                            dx[k] = (ga[HALO + u, k] * wc[0, k] + ga[HALO + u - 1, k] * wc[1, k]
                                     + ga[HALO + u - 2, k] * wc[2, k] + ga[HALO + u - 3, k] * wc[3, k])
                    cute.make_tensor((pDx + (obase + Int64(r) * D)).align(2 * E), cute.make_layout(E)).store(dx.load().to(BFloat16))

    @cute.jit
    def issue(self, j: Int32, n_groups: Int32, tb0: Int32, L: Int32, D: cutlass.Constexpr, base: cute.Pointer, obase: Int64,
              pDy: cute.Pointer, pDp: cute.Pointer, pEx: cute.Pointer, pF: cute.Pointer, bufs, wslot: Int32, lane: Int32, rope):
        """cp.async of group j (tb = tb0 - j U): x of rows tb - 3 - u, dy / dp / extra / rope table of rows tb - u, into
        ring slot j % NST (the slots of rows outside [0, L) are zero-filled); always one commit group."""
        E, U, R = self.E, self.U, self.R
        sX, sDy, sDp, sEx, sF = bufs
        if j < n_groups:
            so = ((wslot + j % self.NST) * U * 32 + lane) * E
            tb = tb0 - j * U
            for u in cutlass.range_constexpr(U):
                o = so + u * 32 * E
                tp = tb - HALO - u
                if (tp >= 0) & (tp < L):
                    self._cp(base + Int64(tp) * D, sX.iterator + o, BFloat16)
                else:
                    self._zero(sX.iterator + o)
                r = tb - u
                if r < L:
                    off = obase + Int64(r) * D
                    self._cp(pDy + off, sDy.iterator + o, BFloat16)
                    self._cp(pDp + off, sDp.iterator + o, BFloat16)
                    if cutlass.const_expr(self.has_extra):
                        self._cp(pEx + off, sEx.iterator + o, BFloat16)
                    if cutlass.const_expr(R > 0):
                        if rope:
                            self._cp(pF + (Int64(r) * R + (lane % self.LH) * E), sF.iterator + o, Float32)
                else:
                    self._zero(sDy.iterator + o)
                    self._zero(sDp.iterator + o)
                    if cutlass.const_expr(self.has_extra):
                        self._zero(sEx.iterator + o)
                    # the rope backward reads the table of these rows too: stale shared memory times the zero dy
                    # is NaN when the stale bits are (rows L - 3 .. L - 1 of dx)
                    if cutlass.const_expr(R > 0):
                        if rope:
                            self._zero(sF.iterator + o, Float32)
        cute.arch.cp_async_commit_group()


def _bos_u8(bos_mask: Optional[torch.Tensor], x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """The bos mask as uint8 [B, BOS_PAD + L + BOS_PAD_R] (zero padding), and whether there is one. Cached on the mask
    (built once per step, used by the q and k calls of every layer, forward and backward)."""
    if bos_mask is None:
        return _dummy(x.device)[0].view(torch.uint8), False
    hit = getattr(bos_mask, "_xllm_bos_pad", None)
    if hit is not None and hit[0] == bos_mask._version:
        return hit[1], True
    B, L = bos_mask.shape
    bos = torch.zeros(B, L + BOS_PAD + BOS_PAD_R, dtype=torch.uint8, device=x.device)
    bos[:, BOS_PAD:BOS_PAD + L] = bos_mask
    bos = bos.reshape(-1)
    bos_mask._xllm_bos_pad = (bos_mask._version, bos)
    return bos, True


def _env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def conv_qk_prep_fwd(x: torch.Tensor, conv_weight: torch.Tensor, normalize: bool, bos_mask: Optional[torch.Tensor], G: int,
                     weight: Optional[torch.Tensor], freqs_cis: Optional[torch.Tensor], rope_dim: int, eps: float, want_c: bool = False):
    """x [B, L, G * S] bf16 (before the conv), conv_weight [G * S, 4] (softmax-normalized over the taps if normalize) ->
    (c [B, L, G * S] the conv output or None, y [B, L, G, S], p [B, L, G, S]) as causal_conv1d_fwd + qk_prep_fwd."""
    B, L, D = x.shape
    S = D // G
    R = rope_dim if freqs_cis is not None else 0
    T = _rows_per_warp(B, L, G // _heads_per_warp(D // G, G))
    x = x.contiguous()
    dev = x.device
    wc = conv_weight.float().contiguous()
    bos, has_bos = _bos_u8(bos_mask, x)
    y = torch.empty(B, L, G, S, dtype=torch.bfloat16, device=dev)
    sm = torch.empty_like(y)
    c = torch.empty(B, L, D, dtype=torch.bfloat16, device=dev) if want_c else None
    db, df = _dummy(dev)
    f = torch.view_as_real(freqs_cis).contiguous().reshape(-1) if R > 0 else df
    w = weight.float().contiguous().reshape(-1) if weight is not None else df
    u, nst = _env("XLLM_CONV_QK_UF", 4), _env("XLLM_CONV_QK_NF", 2)
    fn = _compiled(("conv_qk_fwd", G, S, R, T, weight is not None, has_bos, normalize, want_c, u, nst),
                   lambda: ConvQkFwd(G, S, R, T, weight is not None, has_bos, normalize, want_c, u, nst),
                   lambda: (_fake(BFloat16), _fake(Float32), _fake(Uint8, 1), _fake(Float32), _fake(Float32), _fake(BFloat16),
                            _fake(BFloat16), _fake(BFloat16), Int32(0), Int32(0), Float32(0.0)))
    fn(x.reshape(-1), wc.reshape(-1), bos, f, w, (c if want_c else y).reshape(-1), y.reshape(-1), sm.reshape(-1), B, L, float(eps))
    return c, y, sm


def conv_qk_prep_bwd(dy: torch.Tensor, dp: torch.Tensor, extra: Optional[torch.Tensor], x: torch.Tensor, conv_weight: torch.Tensor,
                     normalize: bool, bos_mask: Optional[torch.Tensor], G: int, weight: Optional[torch.Tensor],
                     freqs_cis: Optional[torch.Tensor], rope_dim: int, eps: float):
    """(dx [B, L, G * S] bf16, conv weight gradient [G * S, 4] in conv_weight's dtype, norm weight gradient [G * S] bf16
    or None) of conv_qk_prep_fwd: dy the gradient of y, dp of p, extra an additional gradient of c ([B, L, G, S] or
    None)."""
    B, L, D = x.shape
    S = D // G
    R = rope_dim if freqs_cis is not None else 0
    T = _rows_per_warp(B, L, G // _heads_per_warp(D // G, G))
    x = x.contiguous()
    dev = x.device
    wc = conv_weight.float().contiguous()
    bos, has_bos = _bos_u8(bos_mask, x)
    HP = _heads_per_warp(S, G)
    Gp, Sp = G // HP, S * HP
    n_cta = -(-(-(-L // T)) // WARPS) * Gp
    affine = weight is not None
    dx = torch.empty(B, L, D, dtype=torch.bfloat16, device=dev)
    part = torch.empty(B * n_cta, WIDTH + 1, Sp, dtype=torch.float32, device=dev)
    db, df = _dummy(dev)
    f = torch.view_as_real(freqs_cis).contiguous().reshape(-1) if R > 0 else df
    w = weight.float().contiguous().reshape(-1) if affine else df
    u, nst = _env("XLLM_CONV_QK_UB", 2), _env("XLLM_CONV_QK_NB", 2)
    fn = _compiled(("conv_qk_bwd", G, S, R, T, affine, has_bos, normalize, extra is not None, u, nst),
                   lambda: ConvQkBwd(G, S, R, T, affine, has_bos, normalize, extra is not None, u, nst),
                   lambda: (_fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32), _fake(Uint8, 1),
                            _fake(Float32), _fake(Float32), _fake(BFloat16), _fake(Float32), Int32(0), Int32(0), Float32(0.0)))
    fn(dy.contiguous().reshape(-1), dp.contiguous().reshape(-1), (extra.contiguous() if extra is not None else dy).reshape(-1),
       x.reshape(-1), wc.reshape(-1), bos, f, w, dx.reshape(-1), part.reshape(-1), B, L, float(eps))
    # CTA rows are (b, tile group, head): head = row % G
    sums = part.view(-1, Gp, WIDTH + 1, Sp).sum(0)
    dwc = sums[:, :WIDTH].permute(0, 2, 1).reshape(D, WIDTH)
    if normalize:
        dwc = torch.ops.aten._softmax_backward_data(dwc, torch.softmax(wc, dim=-1), -1, torch.float32)
    dwn = sums[:, WIDTH].reshape(-1).to(torch.bfloat16) if affine else None
    return dx, dwc.to(conv_weight.dtype), dwn


def conv_qk_prep_supported(x: torch.Tensor, conv_weight: torch.Tensor, G: int, rope_dim: int) -> bool:
    """Whether conv_qk_prep_{fwd,bwd} handle this input (env XLLM_CONV_QK=0 disables them)."""
    if os.environ.get("XLLM_CONV_QK", "1") == "0" or x.dtype != torch.bfloat16 or not x.is_cuda or x.dim() != 3:
        return False
    D = x.shape[-1]
    if D % G or conv_weight.shape != (D, WIDTH):
        return False
    S = D // G
    E = S // 32
    if S % 32 or E not in (2, 4, 8, 16) or rope_dim < 0 or rope_dim > S or rope_dim % E or x.numel() >= 2 ** 31:
        return False
    return torch.cuda.get_device_capability(x.device)[0] >= 9
