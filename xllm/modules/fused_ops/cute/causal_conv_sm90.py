"""Depthwise causal conv1d (width 4, optional silu; no bias / state) for Hopper and newer (CuTe DSL), with the warp
structure of conv_qk_prep.py: a warp owns 256 channels (8 per lane) of a T-row tile, U rows per iteration with the
inputs prefetched by cp.async into a per-warp shared memory ring and a mask-free path when no segment start is near.

    out[t, d] = act(sum_lag w[d, W - 1 - lag] x[t - lag, d] [no bos in (t - lag, t]])

The backward walks its tile descending, recomputes the pre-activation from x, and writes dx (with the look-ahead of
the 3 rows after the tile) and per-CTA conv weight sums (summed on the host). Used for the gekko block's v conv on
sm_90 (the Triton kernels pay a large fixed cost per call there); semantics of causal_conv.py / the Triton kernels."""
import os
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8

try:
    from .adaptive_working_memory import _compiled, _fake
    from .conv_qk_prep import (_ConvQkBase, _bos_u8, _rows_per_warp, _env, WARPS, WIDTH, HALO, BOS_PAD, BOS_PAD_R, LOG2E)
except ImportError:  # loaded from its file (tests)
    import importlib.util as _ilu
    _d = os.path.dirname(os.path.abspath(__file__))
    _spec = _ilu.spec_from_file_location("cute_adaptive_working_memory_fwd", os.path.join(_d, "adaptive_working_memory.py"))
    _awm = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_awm)
    _compiled, _fake = _awm._compiled, _awm._fake
    _spec = _ilu.spec_from_file_location("cute_conv_qk_prep", os.path.join(_d, "conv_qk_prep.py"))
    _cq = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_cq)
    _ConvQkBase, _bos_u8, _rows_per_warp, _env = _cq._ConvQkBase, _cq._bos_u8, _cq._rows_per_warp, _cq._env
    WARPS, WIDTH, HALO, BOS_PAD, BOS_PAD_R, LOG2E = _cq.WARPS, _cq.WIDTH, _cq.HALO, _cq.BOS_PAD, _cq.BOS_PAD_R, _cq.LOG2E

CH = 256  # channels per warp


@cute.jit
def _sigmoid(a: Float32) -> Float32:
    return cute.arch.rcp_approx(1.0 + cute.math.exp2(-a * LOG2E, fastmath=True))


class ConvActFwd(_ConvQkBase):
    def __init__(self, G: int, T: int, has_bos: bool, act: bool, U: int = 4, NST: int = 2):
        super().__init__(G, CH, 0, T, U, NST, False, has_bos, False)
        self.act = act
        assert T % U == 0

    @cute.jit
    def __call__(self, tX: cute.Tensor, tWc: cute.Tensor, tBos: cute.Tensor, tOut: cute.Tensor, B: Int32, L: Int32, stream: cuda.CUstream):
        n_tiles = (L + self.T - 1) // self.T
        n_tg = (n_tiles + WARPS - 1) // WARPS
        self.kernel(tX.iterator, tWc.iterator, tBos.iterator, tOut.iterator, n_tiles, L).launch(
            grid=[n_tg * self.G, B, 1], block=[32 * WARPS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pX: cute.Pointer, pWc: cute.Pointer, pBos: cute.Pointer, pOut: cute.Pointer, n_tiles: Int32, L: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, b, _ = cute.arch.block_idx()
        G, S, E, T, U, NST = self.G, self.S, self.E, self.T, self.U, self.NST
        D = G * S
        lane = tidx % 32
        warp = tidx // 32
        g = bidx % G
        tile = (bidx // G) * WARPS + warp
        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(BFloat16, cute.make_layout(WARPS * NST * U * 32 * E), 16)
        if tile < n_tiles:
            t0 = tile * T
            c0 = g * S + lane * E
            base = pX + Int64(b) * L * D + c0
            obase = pOut + Int64(b) * L * D + c0
            bbase = pBos + (Int64(b) * (L + BOS_PAD + BOS_PAD_R) + BOS_PAD)
            n_groups = T // U
            wslot = warp * NST
            for j in cutlass.range_constexpr(NST - 1):
                self.issue(j, n_groups, t0, L, D, base, sX, wslot, lane)
            wc = cute.make_rmem_tensor(cute.make_layout((WIDTH, E)), Float32)  # wc[lag] = weight[:, W - 1 - lag]
            wv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
            self._weights(pWc, pWc, c0, wc, wv)
            # xa[m] = x[tb - 3 + m]; fa: bos of the same rows
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
            for it in cutlass.range(n_groups):
                tb = t0 + it * U
                so = ((wslot + it % NST) * U * 32 + lane) * E
                cute.arch.cp_async_wait_group(NST - 2)
                self.issue(it + NST - 1, n_groups, t0, L, D, base, sX, wslot, lane)
                nbos = Float32(0.0)
                for u in cutlass.range_constexpr(U):
                    xr = self._ld(sX.iterator + (so + u * 32 * E), BFloat16)
                    for k in cutlass.range_constexpr(E):
                        xa[HALO + u, k] = xr[k]
                    fa[HALO + u] = 0.0
                    if cutlass.const_expr(self.has_bos):
                        fa[HALO + u] = cute.make_tensor(bbase + (tb + u), cute.make_layout(1))[0].to(Float32)
                for m in cutlass.range_constexpr(1, HALO + U):
                    nbos = nbos + fa[m]
                if nbos == 0.0:
                    self.rows(False, tb, L, D, xa, fa, wc, obase)
                else:
                    self.rows(True, tb, L, D, xa, fa, wc, obase)
                for j in cutlass.range_constexpr(HALO):
                    fa[j] = fa[U + j]
                    for k in cutlass.range_constexpr(E):
                        xa[j, k] = xa[U + j, k]
            cute.arch.cp_async_wait_group(0)

    @cute.jit
    def rows(self, masked: cutlass.Constexpr, tb: Int32, L: Int32, D: cutlass.Constexpr, xa: cute.Tensor, fa: cute.Tensor,
             wc: cute.Tensor, obase: cute.Pointer):
        E, U = self.E, self.U
        for u in cutlass.range_constexpr(U):
            j = HALO + u
            cf = cute.make_rmem_tensor(cute.make_layout(E), Float32)
            if cutlass.const_expr(masked):
                v1 = 1.0 - fa[j]
                v2 = v1 * (1.0 - fa[j - 1])
                v3 = v2 * (1.0 - fa[j - 2])
                for k in cutlass.range_constexpr(E):
                    cf[k] = xa[j, k] * wc[0, k] + (xa[j - 1, k] * v1) * wc[1, k] + (xa[j - 2, k] * v2) * wc[2, k] + (xa[j - 3, k] * v3) * wc[3, k]
            else:
                for k in cutlass.range_constexpr(E):
                    cf[k] = xa[j, k] * wc[0, k] + xa[j - 1, k] * wc[1, k] + xa[j - 2, k] * wc[2, k] + xa[j - 3, k] * wc[3, k]
            if cutlass.const_expr(self.act):
                for k in cutlass.range_constexpr(E):
                    cf[k] = cf[k] * _sigmoid(cf[k])
            if tb + u < L:
                cute.make_tensor((obase + Int64(tb + u) * D).align(2 * E), cute.make_layout(E)).store(cf.load().to(BFloat16))

    @cute.jit
    def issue(self, j: Int32, n_groups: Int32, t0: Int32, L: Int32, D: cutlass.Constexpr, base: cute.Pointer, sX: cute.Tensor,
              wslot: Int32, lane: Int32):
        E, U = self.E, self.U
        if j < n_groups:
            so = ((wslot + j % self.NST) * U * 32 + lane) * E
            for u in cutlass.range_constexpr(U):
                t = t0 + j * U + u
                tt = cutlass.select_(t < L, t, L - 1)
                self._cp(base + Int64(tt) * D, sX.iterator + (so + u * 32 * E), BFloat16)
        cute.arch.cp_async_commit_group()


class ConvActBwd(_ConvQkBase):
    """dx and the per-CTA conv weight sums part[cta, W, 256] (tap-major) of one tile: rows t0 + T + 3 .. t0 descending in
    groups of U; g = dy * silu'(pre) (pre recomputed from x) in fp32."""

    def __init__(self, G: int, T: int, has_bos: bool, act: bool, U: int = 4, NST: int = 2):
        super().__init__(G, CH, 0, T, U, NST, False, has_bos, False)
        self.act = act
        assert T % U == 0 and (T + HALO + 1) % U == 0

    @cute.jit
    def __call__(self, tDy: cute.Tensor, tX: cute.Tensor, tWc: cute.Tensor, tBos: cute.Tensor, tDx: cute.Tensor, tPart: cute.Tensor,
                 B: Int32, L: Int32, stream: cuda.CUstream):
        n_tiles = (L + self.T - 1) // self.T
        n_tg = (n_tiles + WARPS - 1) // WARPS
        self.kernel(tDy.iterator, tX.iterator, tWc.iterator, tBos.iterator, tDx.iterator, tPart.iterator, n_tiles, L).launch(
            grid=[n_tg * self.G, B, 1], block=[32 * WARPS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pDy: cute.Pointer, pX: cute.Pointer, pWc: cute.Pointer, pBos: cute.Pointer, pDx: cute.Pointer, pPart: cute.Pointer,
               n_tiles: Int32, L: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, b, _ = cute.arch.block_idx()
        nb, _, _ = cute.arch.grid_dim()
        G, S, E, T, U, NST = self.G, self.S, self.E, self.T, self.U, self.NST
        D = G * S
        lane = tidx % 32
        warp = tidx // 32
        g = bidx % G
        tile = (bidx // G) * WARPS + warp
        smem = cutlass.utils.SmemAllocator()
        n_slot = WARPS * NST * U * 32 * E
        sX = smem.allocate_tensor(BFloat16, cute.make_layout(n_slot), 16)
        sDy = smem.allocate_tensor(BFloat16, cute.make_layout(n_slot), 16)
        sRed = smem.allocate_tensor(Float32, cute.make_layout(WARPS * WIDTH * S), 128)
        c0 = g * S + lane * E
        accc = cute.make_rmem_tensor(cute.make_layout((WIDTH, E)), Float32)
        for k in cutlass.range_constexpr(E):
            for lag in cutlass.range_constexpr(WIDTH):
                accc[lag, k] = 0.0
        if tile < n_tiles:
            t0 = tile * T
            base = pX + Int64(b) * L * D + c0
            obase = Int64(b) * L * D + c0
            bbase = pBos + (Int64(b) * (L + BOS_PAD + BOS_PAD_R) + BOS_PAD)
            tb0 = t0 + T + HALO
            n_groups = (T + HALO + 1) // U
            wslot = warp * NST
            for j in cutlass.range_constexpr(NST - 1):
                self.issue(j, n_groups, tb0, L, D, base, obase, pDy, sX, sDy, wslot, lane)
            wc = cute.make_rmem_tensor(cute.make_layout((WIDTH, E)), Float32)
            wv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
            self._weights(pWc, pWc, c0, wc, wv)
            # xa[m] = x[tb - m]; ga[m] = g[tb + 3 - m]; fa[m] = bos[tb + 3 - m]
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
            for it in cutlass.range(n_groups):
                tb = tb0 - it * U
                so = ((wslot + it % NST) * U * 32 + lane) * E
                cute.arch.cp_async_wait_group(NST - 2)
                self.issue(it + NST - 1, n_groups, tb0, L, D, base, obase, pDy, sX, sDy, wslot, lane)
                nbos = Float32(0.0)
                for m in cutlass.range_constexpr(U + 5):
                    fa[m] = 0.0
                    if cutlass.const_expr(self.has_bos):
                        fa[m] = cute.make_tensor(bbase + (tb + HALO - m), cute.make_layout(1))[0].to(Float32)
                        nbos = nbos + fa[m]
                for u in cutlass.range_constexpr(U):
                    xr = self._ld(sX.iterator + (so + u * 32 * E), BFloat16)
                    for k in cutlass.range_constexpr(E):
                        xa[HALO + u, k] = xr[k]
                if nbos == 0.0:
                    self.rows(False, tb, t0, L, so, fa, xa, ga, wc, accc, sDy, pDx, obase)
                else:
                    self.rows(True, tb, t0, L, so, fa, xa, ga, wc, accc, sDy, pDx, obase)
                for j in cutlass.range_constexpr(HALO):
                    for k in cutlass.range_constexpr(E):
                        xa[j, k] = xa[U + j, k]
                        ga[j, k] = ga[U + j, k]
            cute.arch.cp_async_wait_group(0)
        rb = warp * WIDTH * S + lane * E
        for lag in cutlass.range_constexpr(WIDTH):
            o = cute.make_rmem_tensor(cute.make_layout(E), Float32)
            for k in cutlass.range_constexpr(E):
                o[k] = accc[lag, k]
            cute.make_tensor((sRed.iterator + (rb + (WIDTH - 1 - lag) * S)).align(16), cute.make_layout(E)).store(o.load())
        cute.arch.barrier()
        pbase = pPart + (Int64(b) * nb + bidx) * (WIDTH * S)
        for e in cutlass.range(tidx, WIDTH * S, 32 * WARPS):
            s = Float32(0.0)
            for w in cutlass.range_constexpr(WARPS):
                s = s + sRed[w * WIDTH * S + e]
            cute.make_tensor(pbase + e, cute.make_layout(1))[0] = s

    @cute.jit
    def rows(self, masked: cutlass.Constexpr, tb: Int32, t0: Int32, L: Int32, so: Int32, fa: cute.Tensor, xa: cute.Tensor,
             ga: cute.Tensor, wc: cute.Tensor, accc: cute.Tensor, sDy: cute.Tensor, pDx: cute.Pointer, obase: Int64):
        E, T, U = self.E, self.T, self.U
        D = self.G * self.S
        for u in cutlass.range_constexpr(U):
            r = tb - u
            gy = self._ld(sDy.iterator + (so + u * 32 * E), BFloat16)
            v1 = Float32(1.0)
            v2 = Float32(1.0)
            v3 = Float32(1.0)
            if cutlass.const_expr(masked):
                v1 = 1.0 - fa[HALO + u]
                v2 = v1 * (1.0 - fa[HALO + u + 1])
                v3 = v2 * (1.0 - fa[HALO + u + 2])
            for k in cutlass.range_constexpr(E):
                gk = gy[k]
                if cutlass.const_expr(self.act):
                    if cutlass.const_expr(masked):
                        a = xa[u, k] * wc[0, k] + (xa[u + 1, k] * v1) * wc[1, k] + (xa[u + 2, k] * v2) * wc[2, k] + (xa[u + 3, k] * v3) * wc[3, k]
                    else:
                        a = xa[u, k] * wc[0, k] + xa[u + 1, k] * wc[1, k] + xa[u + 2, k] * wc[2, k] + xa[u + 3, k] * wc[3, k]
                    sg = _sigmoid(a)
                    gk = gk * (sg * (1.0 + a * (1.0 - sg)))
                ga[HALO + u, k] = gk
            if r < t0 + T:
                if r < L:
                    for k in cutlass.range_constexpr(E):
                        gk = ga[HALO + u, k]
                        accc[0, k] = accc[0, k] + gk * xa[u, k]
                        if cutlass.const_expr(masked):
                            accc[1, k] = accc[1, k] + gk * xa[u + 1, k] * v1
                            accc[2, k] = accc[2, k] + gk * xa[u + 2, k] * v2
                            accc[3, k] = accc[3, k] + gk * xa[u + 3, k] * v3
                        else:
                            accc[1, k] = accc[1, k] + gk * xa[u + 1, k]
                            accc[2, k] = accc[2, k] + gk * xa[u + 2, k]
                            accc[3, k] = accc[3, k] + gk * xa[u + 3, k]
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
              pDy: cute.Pointer, sX: cute.Tensor, sDy: cute.Tensor, wslot: Int32, lane: Int32):
        """x of rows tb - 3 - u and dy of rows tb - u into ring slot j % NST (zero-filled outside [0, L))."""
        E, U = self.E, self.U
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
                    self._cp(pDy + (obase + Int64(r) * D), sDy.iterator + o, BFloat16)
                else:
                    self._zero(sDy.iterator + o)
        cute.arch.cp_async_commit_group()


def conv_act_fwd(x: torch.Tensor, weight: torch.Tensor, bos_mask: Optional[torch.Tensor], activation: Optional[str]) -> torch.Tensor:
    """out [B, L, D] bf16 of the causal conv of x [B, L, D] (bf16) with weight [D, 4] (any float dtype)."""
    B, L, D = x.shape
    G = D // CH
    T = _rows_per_warp(B, L, G)
    x = x.contiguous()
    w = weight.float().contiguous()
    bos, has_bos = _bos_u8(bos_mask, x)
    act = activation in ("silu", "swish")
    out = torch.empty_like(x)
    u, nst = _env("XLLM_CONV_SM90_UF", 4), _env("XLLM_CONV_SM90_NF", 2)
    fn = _compiled(("conv_act_fwd", G, T, has_bos, act, u, nst), lambda: ConvActFwd(G, T, has_bos, act, u, nst), lambda: (
        _fake(BFloat16), _fake(Float32), _fake(Uint8, 1), _fake(BFloat16), Int32(0), Int32(0)))
    fn(x.reshape(-1), w.reshape(-1), bos, out.reshape(-1), B, L)
    return out


def conv_act_bwd(out_grad: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bos_mask: Optional[torch.Tensor],
                 activation: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """(dx [B, L, D] bf16, dweight [D, 4] in weight's dtype) of conv_act_fwd."""
    B, L, D = x.shape
    G = D // CH
    T = _rows_per_warp(B, L, G)
    x = x.contiguous()
    w = weight.float().contiguous()
    bos, has_bos = _bos_u8(bos_mask, x)
    act = activation in ("silu", "swish")
    n_cta = -(-(-(-L // T)) // WARPS) * G
    dx = torch.empty_like(x)
    part = torch.empty(B * n_cta, WIDTH, CH, dtype=torch.float32, device=x.device)
    u, nst = _env("XLLM_CONV_SM90_UB", 4), _env("XLLM_CONV_SM90_NB", 2)
    if (T + HALO + 1) % u:
        u = 2
    fn = _compiled(("conv_act_bwd", G, T, has_bos, act, u, nst), lambda: ConvActBwd(G, T, has_bos, act, u, nst), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(Float32), _fake(Uint8, 1), _fake(BFloat16), _fake(Float32), Int32(0), Int32(0)))
    fn(out_grad.contiguous().reshape(-1), x.reshape(-1), w.reshape(-1), bos, dx.reshape(-1), part.reshape(-1), B, L)
    dweight = part.view(-1, G, WIDTH, CH).sum(0).permute(0, 2, 1).reshape(D, WIDTH)
    return dx, dweight.to(weight.dtype)


def conv_act_supported(x: torch.Tensor, weight: torch.Tensor, bias, initial_state, output_final_state: bool,
                       final_state_grad=None) -> bool:
    """Whether conv_act_{fwd,bwd} handle this call (sm_90 only; env XLLM_CONV_SM90=0 disables them)."""
    if os.environ.get("XLLM_CONV_SM90", "1") == "0" or x.dtype != torch.bfloat16 or not x.is_cuda or x.dim() != 3:
        return False
    if bias is not None or initial_state is not None or output_final_state or final_state_grad is not None:
        return False
    if weight.shape != (x.shape[-1], WIDTH) or x.shape[-1] % CH or x.numel() >= 2 ** 31:
        return False
    return torch.cuda.get_device_capability(x.device)[0] == 9
