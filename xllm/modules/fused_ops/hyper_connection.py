"""Triton kernels for hyper-connection streams (xllm/modules/residual/hyper_connection.py).

Streams X [T, N, D] stay in BF16 (or FP16); FP32 lives only in registers. Coefficients carry a
group axis (G = 1 without channel groups): a, b [T, G, N], R [T, G, N, N] with
destination rows; channel d belongs to group d // (D / G). The kernels address the
coefficients through their strides; the oHC coefficients are stored component-major
(physically [T, N, G] / [T, N * N, G]) so that a group block of one component is contiguous.

    hc_logits(Xf, W, eps)     rstd(Xf) * (Xf @ W)        Xf = X.flatten(-2) [T, N * D], FP32 out
    hc_read(X, a)             h  = sum_n a_n X_n          [T, D]
    hc_write(X, R, b, y)      X' = R X + b y              [T, N, D], R None = identity
    ohc_coefficients(...)     oHC gates and SO(4) transport of every group from shared logits
    sinkhorn(z, iterations)   mHC transport from logits [..., 4, 4]

The projection is a Triton rstd kernel and a cuBLAS BF16 matmul with FP32 accumulate and
output; its backward forms dX = dP W^T - c X in one kernel. Read, write, the oHC
coefficients and Sinkhorn are one kernel each way. Every op saves only its inputs, except
the projection, which also keeps its FP32 rstd [T] and unscaled logits [T, n_logits].
"""

import torch
import triton
import triton.language as tl


OHC_EPS = 1e-12


def supported(x: torch.Tensor) -> bool:
    """Fused path: CUDA BF16 / FP16 streams [..., N, D] with N <= 4."""
    return x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) and x.dim() >= 2 and x.shape[-2] <= 4


def _tiles(D: int, G: int, max_tile: int = 1024):
    """(group size, channel tile, channel tiles per group, groups per tile) for D channels in G groups; a tile
    holds at most max_tile group x channel elements per stream. A group larger than the tile, or one whose
    largest power-of-two divisor is at most 8 (groups of 24: 3 x 8, a vector per thread), runs in exact
    channel tiles; any other group is one masked power-of-two tile."""
    gs = D // G
    pow2 = gs & -gs  # largest power of two dividing gs
    if gs > max_tile or (pow2 <= 8 and pow2 < gs):
        ch = min(pow2, max_tile)
        nch = gs // ch
    else:
        ch, nch = triton.next_power_of_2(gs), 1
    gb = max(1, min(triton.next_power_of_2(G), max_tile // ch))
    return gs, ch, nch, gb


def _write_bwd_config(D: int, G: int):
    """(max tile, warps) of the transport write backward (H200 sweep, D = 1536): it keeps 20 accumulator tiles
    live, so long groups take small tiles and exact small groups a mid-size tile."""
    gs = D // G
    pow2 = gs & -gs
    if gs > 512:
        return 256, 2
    if pow2 <= 8 and pow2 < gs:
        return 256, 4
    return 1024, 2


# ---------------------------------------------------------------------------------------------
# controller projection: logits = rstd * (Xf @ W)


@triton.jit
def _rstd_kernel(x_ptr, rstd_ptr, K, eps, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        o = k0 + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + t * K + o, mask=o < K, other=0.0).to(tl.float32)
        acc += x * x
    tl.store(rstd_ptr + t, tl.rsqrt(tl.sum(acc, axis=0) / K + eps))


@triton.jit
def _proj_dx_kernel(x_ptr, dp_ptr, wt_ptr, c_ptr, dx_ptr, T, K, NL,
                    BT: tl.constexpr, BK: tl.constexpr, LP: tl.constexpr):
    # dx[t, k] = sum_l dp[t, l] w[k, l] - c[t] x[t, k]
    ot = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    ok = tl.program_id(1).to(tl.int64) * BK + tl.arange(0, BK)
    ol = tl.arange(0, LP)
    tm = ot < T
    km = ok < K
    lm = ol < NL
    dp = tl.load(dp_ptr + ot[:, None] * NL + ol[None, :], mask=tm[:, None] & lm[None, :], other=0.0)
    wt = tl.load(wt_ptr + ol[:, None] * K + ok[None, :], mask=lm[:, None] & km[None, :], other=0.0)  # W^T [NL, K]
    acc = tl.dot(dp, wt)
    c = tl.load(c_ptr + ot, mask=tm, other=0.0)
    m = tm[:, None] & km[None, :]
    x = tl.load(x_ptr + ot[:, None] * K + ok[None, :], mask=m, other=0.0).to(tl.float32)
    tl.store(dx_ptr + ot[:, None] * K + ok[None, :], (acc - c[:, None] * x).to(dx_ptr.dtype.element_ty), mask=m)


class _Logits(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, eps):
        K, NL = weight.shape
        xf = x.reshape(-1, K)
        T = xf.shape[0]
        rstd = torch.empty(T, device=x.device, dtype=torch.float32)
        _rstd_kernel[(T,)](xf, rstd, K, eps, BLOCK=min(4096, triton.next_power_of_2(K)), num_warps=8)
        w = weight.to(xf.dtype).contiguous()
        p = torch.mm(xf, w, out_dtype=torch.float32)
        ctx.save_for_backward(xf, w, rstd, p)
        ctx.x_shape, ctx.w_dtype = x.shape, weight.dtype
        return (p * rstd[:, None]).view(*x.shape[:-1], NL)

    @staticmethod
    def backward(ctx, grad):
        xf, w, rstd, p = ctx.saved_tensors
        T, K = xf.shape
        NL = w.shape[1]
        g = grad.reshape(T, NL).float()
        c = (g * p).sum(-1) * rstd.pow(3) / K
        dp = (g * rstd[:, None]).to(xf.dtype).contiguous()
        dw = torch.mm(xf.t(), dp, out_dtype=torch.float32).to(ctx.w_dtype) if ctx.needs_input_grad[1] else None
        dx = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty_like(xf)
            BT, BK = 32, 128
            grid = (triton.cdiv(T, BT), triton.cdiv(K, BK))
            _proj_dx_kernel[grid](xf, dp, w.t().contiguous(), c, dx, T, K, NL, BT=BT, BK=BK,
                                  LP=max(16, triton.next_power_of_2(NL)), num_warps=4)
            dx = dx.view(ctx.x_shape)
        return dx, dw, None


def hc_logits(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """[..., K] streams (flattened N * D, contiguous), weight [K, NL] -> FP32 logits [..., NL]."""
    return _Logits.apply(x.contiguous(), weight, eps)


# ---------------------------------------------------------------------------------------------
# stream read and write. One program per token; the N <= 4 streams are explicit [GB, CH] tiles
# (streams n >= N are masked off), per-group reductions accumulate elementwise over the channel
# tiles of a group and reduce once.


@triton.jit
def _ld(ptr, t, n, N, D, d, m2):
    return tl.load(ptr + (t * N + n) * D + d, mask=m2 & (n < N), other=0.0).to(tl.float32)


@triton.jit
def _st(ptr, t, n, N, D, d, m2, v):
    tl.store(ptr + (t * N + n) * D + d, v.to(ptr.dtype.element_ty), mask=m2 & (n < N))


@triton.jit
def _ldc(ptr, t, g, n, N, gm, st, sg, sn):
    return tl.load(ptr + t * st + g * sg + n * sn, mask=gm & (n < N), other=0.0)


@triton.jit
def _stc(ptr, t, g, n, N, gm, st, sg, sn, acc):
    tl.store(ptr + t * st + g * sg + n * sn, tl.sum(acc, axis=1), mask=gm & (n < N))


@triton.jit
def _ldr(ptr, t, g, n, m, N, gm, st, sg, sn, sm):
    return tl.load(ptr + t * st + g * sg + n * sn + m * sm, mask=gm & (n < N) & (m < N), other=0.0)


@triton.jit
def _str(ptr, t, g, n, m, N, gm, st, sg, sn, sm, acc):
    tl.store(ptr + t * st + g * sg + n * sn + m * sm, tl.sum(acc, axis=1), mask=gm & (n < N) & (m < N))


@triton.jit
def _read_fwd_kernel(x_ptr, a_ptr, h_ptr, N, G, GS, NCH, sat, sag, san, CH: tl.constexpr, GB: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    D = G * GS
    gi = tl.arange(0, GB)
    ci = tl.arange(0, CH)
    for g0 in range(0, G, GB):
        g = g0 + gi
        gm = g < G
        a0 = _ldc(a_ptr, t, g, 0, N, gm, sat, sag, san)[:, None]
        a1 = _ldc(a_ptr, t, g, 1, N, gm, sat, sag, san)[:, None]
        a2 = _ldc(a_ptr, t, g, 2, N, gm, sat, sag, san)[:, None]
        a3 = _ldc(a_ptr, t, g, 3, N, gm, sat, sag, san)[:, None]
        for k in range(NCH):
            c = k * CH + ci
            m2 = gm[:, None] & (c < GS)[None, :]
            d = g[:, None] * GS + c[None, :]
            h = (a0 * _ld(x_ptr, t, 0, N, D, d, m2) + a1 * _ld(x_ptr, t, 1, N, D, d, m2)
                 + a2 * _ld(x_ptr, t, 2, N, D, d, m2) + a3 * _ld(x_ptr, t, 3, N, D, d, m2))
            tl.store(h_ptr + t * D + d, h.to(h_ptr.dtype.element_ty), mask=m2)


@triton.jit
def _read_bwd_kernel(x_ptr, a_ptr, dh_ptr, dx_ptr, da_ptr, N, G, GS, NCH, sat, sag, san,
                     CH: tl.constexpr, GB: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    D = G * GS
    gi = tl.arange(0, GB)
    ci = tl.arange(0, CH)
    for g0 in range(0, G, GB):
        g = g0 + gi
        gm = g < G
        a0 = _ldc(a_ptr, t, g, 0, N, gm, sat, sag, san)[:, None]
        a1 = _ldc(a_ptr, t, g, 1, N, gm, sat, sag, san)[:, None]
        a2 = _ldc(a_ptr, t, g, 2, N, gm, sat, sag, san)[:, None]
        a3 = _ldc(a_ptr, t, g, 3, N, gm, sat, sag, san)[:, None]
        p0 = tl.zeros([GB, CH], dtype=tl.float32)
        p1 = tl.zeros([GB, CH], dtype=tl.float32)
        p2 = tl.zeros([GB, CH], dtype=tl.float32)
        p3 = tl.zeros([GB, CH], dtype=tl.float32)
        for k in range(NCH):
            c = k * CH + ci
            m2 = gm[:, None] & (c < GS)[None, :]
            d = g[:, None] * GS + c[None, :]
            dh = tl.load(dh_ptr + t * D + d, mask=m2, other=0.0).to(tl.float32)
            _st(dx_ptr, t, 0, N, D, d, m2, a0 * dh)
            _st(dx_ptr, t, 1, N, D, d, m2, a1 * dh)
            _st(dx_ptr, t, 2, N, D, d, m2, a2 * dh)
            _st(dx_ptr, t, 3, N, D, d, m2, a3 * dh)
            p0 += _ld(x_ptr, t, 0, N, D, d, m2) * dh
            p1 += _ld(x_ptr, t, 1, N, D, d, m2) * dh
            p2 += _ld(x_ptr, t, 2, N, D, d, m2) * dh
            p3 += _ld(x_ptr, t, 3, N, D, d, m2) * dh
        _stc(da_ptr, t, g, 0, N, gm, sat, sag, san, p0)
        _stc(da_ptr, t, g, 1, N, gm, sat, sag, san, p1)
        _stc(da_ptr, t, g, 2, N, gm, sat, sag, san, p2)
        _stc(da_ptr, t, g, 3, N, gm, sat, sag, san, p3)


class _Read(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, a):
        T, N, D = x.shape
        G = a.shape[1]
        gs, ch, nch, gb = _tiles(D, G)
        h = torch.empty(T, D, device=x.device, dtype=x.dtype)
        _read_fwd_kernel[(T,)](x, a, h, N, G, gs, nch, *a.stride(), CH=ch, GB=gb, num_warps=4)
        ctx.save_for_backward(x, a)
        return h

    @staticmethod
    def backward(ctx, dh):
        x, a = ctx.saved_tensors
        T, N, D = x.shape
        G = a.shape[1]
        gs, ch, nch, gb = _tiles(D, G)
        dx = torch.empty_like(x)
        da = torch.empty_like(a)  # same strides as a
        _read_bwd_kernel[(T,)](x, a, dh.contiguous(), dx, da, N, G, gs, nch, *a.stride(), CH=ch, GB=gb,
                               num_warps=2)
        return dx, da


def _coef(c: torch.Tensor, T: int, G: int, *tail) -> torch.Tensor:
    """FP32 [T, G, *tail] view of coefficients [B, L, (G,) *tail]. A dense layout (contiguous or the oHC
    component-major one) is kept without a copy; the backward allocates gradients with torch.empty_like and
    writes them through the input strides, so anything else (slices, broadcasts) is made contiguous."""
    c = c.float().reshape(T, G, *tail)
    return c if c.stride() == torch.empty_like(c).stride() else c.contiguous()


def hc_read(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """x [B, L, N, D], a [B, L, N] or [B, L, G, N] (FP32) -> h [B, L, D] in x's dtype."""
    B, L, N, D = x.shape
    G = 1 if a.dim() == 3 else a.shape[-2]
    h = _Read.apply(x.contiguous().view(B * L, N, D), _coef(a, B * L, G, N))
    return h.view(B, L, D)


@triton.jit
def _mix(r0, r1, r2, r3, x0, x1, x2, x3):
    # sum_m R[n, m] x_m for one destination row n (r_m = R[n, m] per group)
    return r0[:, None] * x0 + r1[:, None] * x1 + r2[:, None] * x2 + r3[:, None] * x3


@triton.jit
def _write_fwd_kernel(x_ptr, r_ptr, b_ptr, y_ptr, out_ptr, N, G, GS, NCH, sbt, sbg, sbn, srt, srg, srn, srm,
                      CH: tl.constexpr, GB: tl.constexpr, HAS_R: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    D = G * GS
    gi = tl.arange(0, GB)
    ci = tl.arange(0, CH)
    for g0 in range(0, G, GB):
        g = g0 + gi
        gm = g < G
        b0 = _ldc(b_ptr, t, g, 0, N, gm, sbt, sbg, sbn)[:, None]
        b1 = _ldc(b_ptr, t, g, 1, N, gm, sbt, sbg, sbn)[:, None]
        b2 = _ldc(b_ptr, t, g, 2, N, gm, sbt, sbg, sbn)[:, None]
        b3 = _ldc(b_ptr, t, g, 3, N, gm, sbt, sbg, sbn)[:, None]
        # transport entries R[n, m] of the groups in this block (masked off without a transport)
        rm = gm & HAS_R
        r00 = _ldr(r_ptr, t, g, 0, 0, N, rm, srt, srg, srn, srm)
        r01 = _ldr(r_ptr, t, g, 0, 1, N, rm, srt, srg, srn, srm)
        r02 = _ldr(r_ptr, t, g, 0, 2, N, rm, srt, srg, srn, srm)
        r03 = _ldr(r_ptr, t, g, 0, 3, N, rm, srt, srg, srn, srm)
        r10 = _ldr(r_ptr, t, g, 1, 0, N, rm, srt, srg, srn, srm)
        r11 = _ldr(r_ptr, t, g, 1, 1, N, rm, srt, srg, srn, srm)
        r12 = _ldr(r_ptr, t, g, 1, 2, N, rm, srt, srg, srn, srm)
        r13 = _ldr(r_ptr, t, g, 1, 3, N, rm, srt, srg, srn, srm)
        r20 = _ldr(r_ptr, t, g, 2, 0, N, rm, srt, srg, srn, srm)
        r21 = _ldr(r_ptr, t, g, 2, 1, N, rm, srt, srg, srn, srm)
        r22 = _ldr(r_ptr, t, g, 2, 2, N, rm, srt, srg, srn, srm)
        r23 = _ldr(r_ptr, t, g, 2, 3, N, rm, srt, srg, srn, srm)
        r30 = _ldr(r_ptr, t, g, 3, 0, N, rm, srt, srg, srn, srm)
        r31 = _ldr(r_ptr, t, g, 3, 1, N, rm, srt, srg, srn, srm)
        r32 = _ldr(r_ptr, t, g, 3, 2, N, rm, srt, srg, srn, srm)
        r33 = _ldr(r_ptr, t, g, 3, 3, N, rm, srt, srg, srn, srm)
        for k in range(NCH):
            c = k * CH + ci
            m2 = gm[:, None] & (c < GS)[None, :]
            d = g[:, None] * GS + c[None, :]
            y = tl.load(y_ptr + t * D + d, mask=m2, other=0.0).to(tl.float32)
            x0 = _ld(x_ptr, t, 0, N, D, d, m2)
            x1 = _ld(x_ptr, t, 1, N, D, d, m2)
            x2 = _ld(x_ptr, t, 2, N, D, d, m2)
            x3 = _ld(x_ptr, t, 3, N, D, d, m2)
            if HAS_R:
                _st(out_ptr, t, 0, N, D, d, m2, _mix(r00, r01, r02, r03, x0, x1, x2, x3) + b0 * y)
                _st(out_ptr, t, 1, N, D, d, m2, _mix(r10, r11, r12, r13, x0, x1, x2, x3) + b1 * y)
                _st(out_ptr, t, 2, N, D, d, m2, _mix(r20, r21, r22, r23, x0, x1, x2, x3) + b2 * y)
                _st(out_ptr, t, 3, N, D, d, m2, _mix(r30, r31, r32, r33, x0, x1, x2, x3) + b3 * y)
            else:
                _st(out_ptr, t, 0, N, D, d, m2, x0 + b0 * y)
                _st(out_ptr, t, 1, N, D, d, m2, x1 + b1 * y)
                _st(out_ptr, t, 2, N, D, d, m2, x2 + b2 * y)
                _st(out_ptr, t, 3, N, D, d, m2, x3 + b3 * y)


@triton.jit
def _write_bwd_kernel(x_ptr, r_ptr, b_ptr, y_ptr, go_ptr, dx_ptr, dr_ptr, db_ptr, dy_ptr, N, G, GS, NCH,
                      sbt, sbg, sbn, srt, srg, srn, srm, CH: tl.constexpr, GB: tl.constexpr, HAS_R: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    D = G * GS
    gi = tl.arange(0, GB)
    ci = tl.arange(0, CH)
    for g0 in range(0, G, GB):
        g = g0 + gi
        gm = g < G
        b0 = _ldc(b_ptr, t, g, 0, N, gm, sbt, sbg, sbn)[:, None]
        b1 = _ldc(b_ptr, t, g, 1, N, gm, sbt, sbg, sbn)[:, None]
        b2 = _ldc(b_ptr, t, g, 2, N, gm, sbt, sbg, sbn)[:, None]
        b3 = _ldc(b_ptr, t, g, 3, N, gm, sbt, sbg, sbn)[:, None]
        # transport entries R[n, m] of the groups in this block (masked off without a transport)
        rm = gm & HAS_R
        r00 = _ldr(r_ptr, t, g, 0, 0, N, rm, srt, srg, srn, srm)
        r01 = _ldr(r_ptr, t, g, 0, 1, N, rm, srt, srg, srn, srm)
        r02 = _ldr(r_ptr, t, g, 0, 2, N, rm, srt, srg, srn, srm)
        r03 = _ldr(r_ptr, t, g, 0, 3, N, rm, srt, srg, srn, srm)
        r10 = _ldr(r_ptr, t, g, 1, 0, N, rm, srt, srg, srn, srm)
        r11 = _ldr(r_ptr, t, g, 1, 1, N, rm, srt, srg, srn, srm)
        r12 = _ldr(r_ptr, t, g, 1, 2, N, rm, srt, srg, srn, srm)
        r13 = _ldr(r_ptr, t, g, 1, 3, N, rm, srt, srg, srn, srm)
        r20 = _ldr(r_ptr, t, g, 2, 0, N, rm, srt, srg, srn, srm)
        r21 = _ldr(r_ptr, t, g, 2, 1, N, rm, srt, srg, srn, srm)
        r22 = _ldr(r_ptr, t, g, 2, 2, N, rm, srt, srg, srn, srm)
        r23 = _ldr(r_ptr, t, g, 2, 3, N, rm, srt, srg, srn, srm)
        r30 = _ldr(r_ptr, t, g, 3, 0, N, rm, srt, srg, srn, srm)
        r31 = _ldr(r_ptr, t, g, 3, 1, N, rm, srt, srg, srn, srm)
        r32 = _ldr(r_ptr, t, g, 3, 2, N, rm, srt, srg, srn, srm)
        r33 = _ldr(r_ptr, t, g, 3, 3, N, rm, srt, srg, srn, srm)
        q0 = tl.zeros([GB, CH], dtype=tl.float32)  # db_n = sum_c g_n y
        q1 = tl.zeros([GB, CH], dtype=tl.float32)
        q2 = tl.zeros([GB, CH], dtype=tl.float32)
        q3 = tl.zeros([GB, CH], dtype=tl.float32)
        # dR[n, m] = sum_c g_n x_m (only with a transport)
        p00 = tl.zeros([GB, CH], dtype=tl.float32)
        p01 = tl.zeros([GB, CH], dtype=tl.float32)
        p02 = tl.zeros([GB, CH], dtype=tl.float32)
        p03 = tl.zeros([GB, CH], dtype=tl.float32)
        p10 = tl.zeros([GB, CH], dtype=tl.float32)
        p11 = tl.zeros([GB, CH], dtype=tl.float32)
        p12 = tl.zeros([GB, CH], dtype=tl.float32)
        p13 = tl.zeros([GB, CH], dtype=tl.float32)
        p20 = tl.zeros([GB, CH], dtype=tl.float32)
        p21 = tl.zeros([GB, CH], dtype=tl.float32)
        p22 = tl.zeros([GB, CH], dtype=tl.float32)
        p23 = tl.zeros([GB, CH], dtype=tl.float32)
        p30 = tl.zeros([GB, CH], dtype=tl.float32)
        p31 = tl.zeros([GB, CH], dtype=tl.float32)
        p32 = tl.zeros([GB, CH], dtype=tl.float32)
        p33 = tl.zeros([GB, CH], dtype=tl.float32)
        for k in range(NCH):
            c = k * CH + ci
            m2 = gm[:, None] & (c < GS)[None, :]
            d = g[:, None] * GS + c[None, :]
            y = tl.load(y_ptr + t * D + d, mask=m2, other=0.0).to(tl.float32)
            g0_ = _ld(go_ptr, t, 0, N, D, d, m2)
            g1_ = _ld(go_ptr, t, 1, N, D, d, m2)
            g2_ = _ld(go_ptr, t, 2, N, D, d, m2)
            g3_ = _ld(go_ptr, t, 3, N, D, d, m2)
            tl.store(dy_ptr + t * D + d, (b0 * g0_ + b1 * g1_ + b2 * g2_ + b3 * g3_).to(dy_ptr.dtype.element_ty),
                     mask=m2)
            q0 += g0_ * y
            q1 += g1_ * y
            q2 += g2_ * y
            q3 += g3_ * y
            if HAS_R:
                x0 = _ld(x_ptr, t, 0, N, D, d, m2)
                x1 = _ld(x_ptr, t, 1, N, D, d, m2)
                x2 = _ld(x_ptr, t, 2, N, D, d, m2)
                x3 = _ld(x_ptr, t, 3, N, D, d, m2)
                # dX_m = sum_n R[n, m] g_n
                _st(dx_ptr, t, 0, N, D, d, m2, _mix(r00, r10, r20, r30, g0_, g1_, g2_, g3_))
                _st(dx_ptr, t, 1, N, D, d, m2, _mix(r01, r11, r21, r31, g0_, g1_, g2_, g3_))
                _st(dx_ptr, t, 2, N, D, d, m2, _mix(r02, r12, r22, r32, g0_, g1_, g2_, g3_))
                _st(dx_ptr, t, 3, N, D, d, m2, _mix(r03, r13, r23, r33, g0_, g1_, g2_, g3_))
                p00 += g0_ * x0
                p01 += g0_ * x1
                p02 += g0_ * x2
                p03 += g0_ * x3
                p10 += g1_ * x0
                p11 += g1_ * x1
                p12 += g1_ * x2
                p13 += g1_ * x3
                p20 += g2_ * x0
                p21 += g2_ * x1
                p22 += g2_ * x2
                p23 += g2_ * x3
                p30 += g3_ * x0
                p31 += g3_ * x1
                p32 += g3_ * x2
                p33 += g3_ * x3
        _stc(db_ptr, t, g, 0, N, gm, sbt, sbg, sbn, q0)
        _stc(db_ptr, t, g, 1, N, gm, sbt, sbg, sbn, q1)
        _stc(db_ptr, t, g, 2, N, gm, sbt, sbg, sbn, q2)
        _stc(db_ptr, t, g, 3, N, gm, sbt, sbg, sbn, q3)
        if HAS_R:
            _str(dr_ptr, t, g, 0, 0, N, gm, srt, srg, srn, srm, p00)
            _str(dr_ptr, t, g, 0, 1, N, gm, srt, srg, srn, srm, p01)
            _str(dr_ptr, t, g, 0, 2, N, gm, srt, srg, srn, srm, p02)
            _str(dr_ptr, t, g, 0, 3, N, gm, srt, srg, srn, srm, p03)
            _str(dr_ptr, t, g, 1, 0, N, gm, srt, srg, srn, srm, p10)
            _str(dr_ptr, t, g, 1, 1, N, gm, srt, srg, srn, srm, p11)
            _str(dr_ptr, t, g, 1, 2, N, gm, srt, srg, srn, srm, p12)
            _str(dr_ptr, t, g, 1, 3, N, gm, srt, srg, srn, srm, p13)
            _str(dr_ptr, t, g, 2, 0, N, gm, srt, srg, srn, srm, p20)
            _str(dr_ptr, t, g, 2, 1, N, gm, srt, srg, srn, srm, p21)
            _str(dr_ptr, t, g, 2, 2, N, gm, srt, srg, srn, srm, p22)
            _str(dr_ptr, t, g, 2, 3, N, gm, srt, srg, srn, srm, p23)
            _str(dr_ptr, t, g, 3, 0, N, gm, srt, srg, srn, srm, p30)
            _str(dr_ptr, t, g, 3, 1, N, gm, srt, srg, srn, srm, p31)
            _str(dr_ptr, t, g, 3, 2, N, gm, srt, srg, srn, srm, p32)
            _str(dr_ptr, t, g, 3, 3, N, gm, srt, srg, srn, srm, p33)


class _Write(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, R, b, y):
        T, N, D = x.shape
        G = b.shape[1]
        gs, ch, nch, gb = _tiles(D, G)
        out = torch.empty_like(x)
        has_r = R is not None
        rs = R.stride() if has_r else (0, 0, 0, 0)
        _write_fwd_kernel[(T,)](x, R if has_r else b, b, y, out, N, G, gs, nch, *b.stride(), *rs,
                                CH=ch, GB=gb, HAS_R=has_r, num_warps=4)
        ctx.has_r = has_r
        ctx.save_for_backward(x, R if has_r else None, b, y)
        return out

    @staticmethod
    def backward(ctx, go):
        x, R, b, y = ctx.saved_tensors
        T, N, D = x.shape
        G = b.shape[1]
        tile, warps = _write_bwd_config(D, G) if ctx.has_r else (1024, 4)
        gs, ch, nch, gb = _tiles(D, G, tile)
        go = go.contiguous()
        db = torch.empty_like(b)  # same strides as b
        dy = torch.empty_like(y)
        if ctx.has_r:
            dx = torch.empty_like(x)
            dR = torch.empty_like(R)  # same strides as R
            rs = R.stride()
        else:
            dx, dR, rs = go, None, (0, 0, 0, 0)  # identity transport: dX = dX'
        _write_bwd_kernel[(T,)](x, R if ctx.has_r else b, b, y, go, dx, dR if ctx.has_r else b, db, dy,
                                N, G, gs, nch, *b.stride(), *rs, CH=ch, GB=gb, HAS_R=ctx.has_r, num_warps=warps)
        return dx, dR, db, dy


def hc_write(x: torch.Tensor, R, b: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x [B, L, N, D], R [B, L, (G,) N, N] FP32 or None, b [B, L, (G,) N] FP32, y [B, L, D] -> X'."""
    B, L, N, D = x.shape
    T = B * L
    G = 1 if b.dim() == 3 else b.shape[-2]
    R = None if R is None else _coef(R, T, G, N, N)
    out = _Write.apply(x.contiguous().view(T, N, D), R, _coef(b, T, G, N), y.to(x.dtype).contiguous().view(T, D))
    return out.view(B, L, N, D)


# ---------------------------------------------------------------------------------------------
# oHC coefficients from the shared logits l [T, 16] = (a: 4, b: 4, rotation: 8):
#   a_g = sigmoid(s_a l_a + bias_g),  b_g = 2 sigmoid(s_b l_b + bias_g)
#   R_g = SO(4)(alpha rate_g l_r + bias_g):  q = (1 + z0, z1, z2, z3) / |.|,  r = (1 + z4, ..) / |.|,  R = L(q) Rc(r)
# Outputs are component-major: a, b physically [T, 4, G], R [T, 16, G]; u = dL / dlogits sums [16, T];
# per-program bias partials [P, 16, G].


@triton.jit
def _ohc_rotation(lp, bp, G, rate, tm, gm, eps):
    z0 = rate * tl.load(lp + 8, mask=tm, other=0.0)[:, None] + tl.load(bp + 8 * G, mask=gm, other=0.0)[None, :]
    z1 = rate * tl.load(lp + 9, mask=tm, other=0.0)[:, None] + tl.load(bp + 9 * G, mask=gm, other=0.0)[None, :]
    z2 = rate * tl.load(lp + 10, mask=tm, other=0.0)[:, None] + tl.load(bp + 10 * G, mask=gm, other=0.0)[None, :]
    z3 = rate * tl.load(lp + 11, mask=tm, other=0.0)[:, None] + tl.load(bp + 11 * G, mask=gm, other=0.0)[None, :]
    z4 = rate * tl.load(lp + 12, mask=tm, other=0.0)[:, None] + tl.load(bp + 12 * G, mask=gm, other=0.0)[None, :]
    z5 = rate * tl.load(lp + 13, mask=tm, other=0.0)[:, None] + tl.load(bp + 13 * G, mask=gm, other=0.0)[None, :]
    z6 = rate * tl.load(lp + 14, mask=tm, other=0.0)[:, None] + tl.load(bp + 14 * G, mask=gm, other=0.0)[None, :]
    z7 = rate * tl.load(lp + 15, mask=tm, other=0.0)[:, None] + tl.load(bp + 15 * G, mask=gm, other=0.0)[None, :]
    vw = 1.0 + z0
    va = 1.0 + z4
    sq = tl.rsqrt(vw * vw + z1 * z1 + z2 * z2 + z3 * z3 + eps)
    sr = tl.rsqrt(va * va + z5 * z5 + z6 * z6 + z7 * z7 + eps)
    return vw, z1, z2, z3, sq, va, z5, z6, z7, sr


@triton.jit
def _ohc_coef_fwd_kernel(l_ptr, scale_ptr, bias_ptr, rate_ptr, alpha_ptr, a_ptr, b_ptr, r_ptr, T, G, eps,
                         BT: tl.constexpr, GP: tl.constexpr):
    ot = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    og = tl.arange(0, GP)
    tm = ot < T
    gm = og < G
    m = tm[:, None] & gm[None, :]
    lp = l_ptr + ot * 16
    bp = bias_ptr + og  # bias is component-major [16, G]
    ap = a_ptr + ot[:, None] * (4 * G) + og[None, :]
    bq = b_ptr + ot[:, None] * (4 * G) + og[None, :]
    rp = r_ptr + ot[:, None] * (16 * G) + og[None, :]
    sa = tl.load(scale_ptr)
    sb = tl.load(scale_ptr + 1)
    rate = tl.load(rate_ptr + og, mask=gm, other=0.0)[None, :] * tl.load(alpha_ptr)
    for n in tl.static_range(4):
        za = sa * tl.load(lp + n, mask=tm, other=0.0)[:, None] + tl.load(bp + n * G, mask=gm, other=0.0)[None, :]
        tl.store(ap + n * G, tl.sigmoid(za), mask=m)
        zb = sb * tl.load(lp + 4 + n, mask=tm, other=0.0)[:, None] + tl.load(bp + (4 + n) * G, mask=gm, other=0.0)[None, :]
        tl.store(bq + n * G, 2.0 * tl.sigmoid(zb), mask=m)
    vw, vx, vy, vz, sq, va, vb, vc, vd, sr = _ohc_rotation(lp, bp, G, rate, tm, gm, eps)
    qw = vw * sq
    qx = vx * sq
    qy = vy * sq
    qz = vz * sq
    ra = va * sr
    rb = vb * sr
    rc = vc * sr
    rd = vd * sr
    tl.store(rp + 0 * G, qw * ra + qx * rb + qy * rc + qz * rd, mask=m)
    tl.store(rp + 1 * G, qw * rb - qx * ra - qy * rd + qz * rc, mask=m)
    tl.store(rp + 2 * G, qw * rc + qx * rd - qy * ra - qz * rb, mask=m)
    tl.store(rp + 3 * G, qw * rd - qx * rc + qy * rb - qz * ra, mask=m)
    tl.store(rp + 4 * G, qx * ra - qw * rb + qz * rc - qy * rd, mask=m)
    tl.store(rp + 5 * G, qx * rb + qw * ra - qz * rd - qy * rc, mask=m)
    tl.store(rp + 6 * G, qx * rc - qw * rd - qz * ra + qy * rb, mask=m)
    tl.store(rp + 7 * G, qx * rd + qw * rc + qz * rb + qy * ra, mask=m)
    tl.store(rp + 8 * G, qy * ra - qz * rb - qw * rc + qx * rd, mask=m)
    tl.store(rp + 9 * G, qy * rb + qz * ra + qw * rd + qx * rc, mask=m)
    tl.store(rp + 10 * G, qy * rc - qz * rd + qw * ra - qx * rb, mask=m)
    tl.store(rp + 11 * G, qy * rd + qz * rc - qw * rb - qx * ra, mask=m)
    tl.store(rp + 12 * G, qz * ra + qy * rb - qx * rc - qw * rd, mask=m)
    tl.store(rp + 13 * G, qz * rb - qy * ra + qx * rd - qw * rc, mask=m)
    tl.store(rp + 14 * G, qz * rc + qy * rd + qx * ra + qw * rb, mask=m)
    tl.store(rp + 15 * G, qz * rd - qy * rc - qx * rb + qw * ra, mask=m)


@triton.jit
def _ohc_reduce_store(dz, j, w, u_ptr, part_ptr, T, ot, tm, og, gm, G):
    # u[j, t] = sum_g w_g dz[t, g];  part[program, j, g] = sum_t dz[t, g]
    tl.store(u_ptr + j * T + ot, tl.sum(dz * w, axis=1), mask=tm)
    tl.store(part_ptr + (tl.program_id(0).to(tl.int64) * 16 + j) * G + og, tl.sum(dz, axis=0), mask=gm)


@triton.jit
def _ohc_coef_bwd_kernel(l_ptr, scale_ptr, bias_ptr, rate_ptr, alpha_ptr, da_ptr, db_ptr, dr_ptr, u_ptr, part_ptr,
                         T, G, eps, sat, sag, san, sbt, sbg, sbn, srt, srg, srn,
                         BT: tl.constexpr, GP: tl.constexpr):
    ot = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    og = tl.arange(0, GP)
    tm = ot < T
    gm = og < G
    m = tm[:, None] & gm[None, :]
    lp = l_ptr + ot * 16
    bp = bias_ptr + og
    dap = da_ptr + ot[:, None] * sat + og[None, :] * sag
    dbp = db_ptr + ot[:, None] * sbt + og[None, :] * sbg
    drp = dr_ptr + ot[:, None] * srt + og[None, :] * srg
    sa = tl.load(scale_ptr)
    sb = tl.load(scale_ptr + 1)
    rate = tl.load(rate_ptr + og, mask=gm, other=0.0)[None, :]
    one = tl.where(m, 1.0, 0.0)
    for n in tl.static_range(4):
        za = sa * tl.load(lp + n, mask=tm, other=0.0)[:, None] + tl.load(bp + n * G, mask=gm, other=0.0)[None, :]
        a = tl.sigmoid(za)
        dza = tl.where(m, tl.load(dap + n * san, mask=m, other=0.0) * a * (1.0 - a), 0.0)
        _ohc_reduce_store(dza, n, one, u_ptr, part_ptr, T, ot, tm, og, gm, G)
        zb = sb * tl.load(lp + 4 + n, mask=tm, other=0.0)[:, None] + tl.load(bp + (4 + n) * G, mask=gm, other=0.0)[None, :]
        s = tl.sigmoid(zb)
        dzb = tl.where(m, tl.load(dbp + n * sbn, mask=m, other=0.0) * 2.0 * s * (1.0 - s), 0.0)
        _ohc_reduce_store(dzb, 4 + n, one, u_ptr, part_ptr, T, ot, tm, og, gm, G)

    vw, vx, vy, vz, sq, va, vb, vc, vd, sr = _ohc_rotation(lp, bp, G, rate * tl.load(alpha_ptr), tm, gm, eps)
    qw = vw * sq
    qx = vx * sq
    qy = vy * sq
    qz = vz * sq
    ra = va * sr
    rb = vb * sr
    rc = vc * sr
    rd = vd * sr
    dr00 = tl.load(drp + 0 * srn, mask=m, other=0.0)
    dr01 = tl.load(drp + 1 * srn, mask=m, other=0.0)
    dr02 = tl.load(drp + 2 * srn, mask=m, other=0.0)
    dr03 = tl.load(drp + 3 * srn, mask=m, other=0.0)
    dr10 = tl.load(drp + 4 * srn, mask=m, other=0.0)
    dr11 = tl.load(drp + 5 * srn, mask=m, other=0.0)
    dr12 = tl.load(drp + 6 * srn, mask=m, other=0.0)
    dr13 = tl.load(drp + 7 * srn, mask=m, other=0.0)
    dr20 = tl.load(drp + 8 * srn, mask=m, other=0.0)
    dr21 = tl.load(drp + 9 * srn, mask=m, other=0.0)
    dr22 = tl.load(drp + 10 * srn, mask=m, other=0.0)
    dr23 = tl.load(drp + 11 * srn, mask=m, other=0.0)
    dr30 = tl.load(drp + 12 * srn, mask=m, other=0.0)
    dr31 = tl.load(drp + 13 * srn, mask=m, other=0.0)
    dr32 = tl.load(drp + 14 * srn, mask=m, other=0.0)
    dr33 = tl.load(drp + 15 * srn, mask=m, other=0.0)
    # dL = dR Rc^T, dRc = L^T dR
    dl00 = dr00 * ra + dr01 * rb + dr02 * rc + dr03 * rd
    dl01 = -dr00 * rb + dr01 * ra - dr02 * rd + dr03 * rc
    dl02 = -dr00 * rc + dr01 * rd + dr02 * ra - dr03 * rb
    dl03 = -dr00 * rd - dr01 * rc + dr02 * rb + dr03 * ra
    dl10 = dr10 * ra + dr11 * rb + dr12 * rc + dr13 * rd
    dl11 = -dr10 * rb + dr11 * ra - dr12 * rd + dr13 * rc
    dl12 = -dr10 * rc + dr11 * rd + dr12 * ra - dr13 * rb
    dl13 = -dr10 * rd - dr11 * rc + dr12 * rb + dr13 * ra
    dl20 = dr20 * ra + dr21 * rb + dr22 * rc + dr23 * rd
    dl21 = -dr20 * rb + dr21 * ra - dr22 * rd + dr23 * rc
    dl22 = -dr20 * rc + dr21 * rd + dr22 * ra - dr23 * rb
    dl23 = -dr20 * rd - dr21 * rc + dr22 * rb + dr23 * ra
    dl30 = dr30 * ra + dr31 * rb + dr32 * rc + dr33 * rd
    dl31 = -dr30 * rb + dr31 * ra - dr32 * rd + dr33 * rc
    dl32 = -dr30 * rc + dr31 * rd + dr32 * ra - dr33 * rb
    dl33 = -dr30 * rd - dr31 * rc + dr32 * rb + dr33 * ra
    dc00 = qw * dr00 + qx * dr10 + qy * dr20 + qz * dr30
    dc01 = qw * dr01 + qx * dr11 + qy * dr21 + qz * dr31
    dc02 = qw * dr02 + qx * dr12 + qy * dr22 + qz * dr32
    dc03 = qw * dr03 + qx * dr13 + qy * dr23 + qz * dr33
    dc10 = -qx * dr00 + qw * dr10 + qz * dr20 - qy * dr30
    dc11 = -qx * dr01 + qw * dr11 + qz * dr21 - qy * dr31
    dc12 = -qx * dr02 + qw * dr12 + qz * dr22 - qy * dr32
    dc13 = -qx * dr03 + qw * dr13 + qz * dr23 - qy * dr33
    dc20 = -qy * dr00 - qz * dr10 + qw * dr20 + qx * dr30
    dc21 = -qy * dr01 - qz * dr11 + qw * dr21 + qx * dr31
    dc22 = -qy * dr02 - qz * dr12 + qw * dr22 + qx * dr32
    dc23 = -qy * dr03 - qz * dr13 + qw * dr23 + qx * dr33
    dc30 = -qz * dr00 + qy * dr10 - qx * dr20 + qw * dr30
    dc31 = -qz * dr01 + qy * dr11 - qx * dr21 + qw * dr31
    dc32 = -qz * dr02 + qy * dr12 - qx * dr22 + qw * dr32
    dc33 = -qz * dr03 + qy * dr13 - qx * dr23 + qw * dr33
    gqw = dl00 + dl11 + dl22 + dl33
    gqx = -dl01 + dl10 - dl23 + dl32
    gqy = -dl02 + dl13 + dl20 - dl31
    gqz = -dl03 - dl12 + dl21 + dl30
    gra = dc00 + dc11 + dc22 + dc33
    grb = dc01 - dc10 - dc23 + dc32
    grc = dc02 + dc13 - dc20 - dc31
    grd = dc03 - dc12 + dc21 - dc30
    # through the normalization q = v s, s = (|v|^2 + eps)^-1/2: dv = s dq - s^3 (v . dq) v
    tq = (vw * gqw + vx * gqx + vy * gqy + vz * gqz) * sq * sq * sq
    tr = (va * gra + vb * grb + vc * grc + vd * grd) * sr * sr * sr
    _ohc_reduce_store(tl.where(m, sq * gqw - tq * vw, 0.0), 8, rate, u_ptr, part_ptr, T, ot, tm, og, gm, G)
    _ohc_reduce_store(tl.where(m, sq * gqx - tq * vx, 0.0), 9, rate, u_ptr, part_ptr, T, ot, tm, og, gm, G)
    _ohc_reduce_store(tl.where(m, sq * gqy - tq * vy, 0.0), 10, rate, u_ptr, part_ptr, T, ot, tm, og, gm, G)
    _ohc_reduce_store(tl.where(m, sq * gqz - tq * vz, 0.0), 11, rate, u_ptr, part_ptr, T, ot, tm, og, gm, G)
    _ohc_reduce_store(tl.where(m, sr * gra - tr * va, 0.0), 12, rate, u_ptr, part_ptr, T, ot, tm, og, gm, G)
    _ohc_reduce_store(tl.where(m, sr * grb - tr * vb, 0.0), 13, rate, u_ptr, part_ptr, T, ot, tm, og, gm, G)
    _ohc_reduce_store(tl.where(m, sr * grc - tr * vc, 0.0), 14, rate, u_ptr, part_ptr, T, ot, tm, og, gm, G)
    _ohc_reduce_store(tl.where(m, sr * grd - tr * vd, 0.0), 15, rate, u_ptr, part_ptr, T, ot, tm, og, gm, G)


def _ohc_blocks(G):
    gp = triton.next_power_of_2(G)
    return max(1, 1024 // gp), gp


class _OhcCoefficients(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, scale, bias, rates, alpha):
        T = logits.shape[0]
        G = bias.shape[0]
        bt, gp = _ohc_blocks(G)
        bias_t = bias.t().contiguous()  # [16, G]
        dev = logits.device
        a = torch.empty(T, 4, G, device=dev, dtype=torch.float32)
        b = torch.empty(T, 4, G, device=dev, dtype=torch.float32)
        R = torch.empty(T, 16, G, device=dev, dtype=torch.float32)
        _ohc_coef_fwd_kernel[(triton.cdiv(T, bt),)](logits, scale, bias_t, rates, alpha, a, b, R, T, G, OHC_EPS,
                                                    BT=bt, GP=gp, num_warps=4)
        ctx.save_for_backward(logits, scale, bias_t, rates, alpha)
        # logical [T, G, 4] / [T, G, 4, 4] views of the component-major storage
        return a.transpose(1, 2), b.transpose(1, 2), R.view(T, 4, 4, G).permute(0, 3, 1, 2)

    @staticmethod
    def backward(ctx, da, db, dR):
        logits, scale, bias_t, rates, alpha = ctx.saved_tensors
        T = logits.shape[0]
        G = bias_t.shape[1]
        bt, gp = _ohc_blocks(G)
        nprog = triton.cdiv(T, bt)
        dev = logits.device
        if da is None:
            da = torch.zeros(T, 4, G, device=dev).transpose(1, 2)
        if db is None:
            db = torch.zeros(T, 4, G, device=dev).transpose(1, 2)
        if dR is None:
            dR = torch.zeros(T, 4, 4, G, device=dev).permute(0, 3, 1, 2)
        # R entries (n, m) must sit at a fixed stride n * 4 + m: flatten the matrix axes of dR
        if dR.stride(2) != 4 * dR.stride(3):
            dR = dR.contiguous()
        u = torch.empty(16, T, device=dev, dtype=torch.float32)
        part = torch.empty(nprog, 16, G, device=dev, dtype=torch.float32)
        _ohc_coef_bwd_kernel[(nprog,)](logits, scale, bias_t, rates, alpha, da, db, dR, u, part, T, G, OHC_EPS,
                                       *da.stride(), *db.stride(), dR.stride(0), dR.stride(1), dR.stride(3),
                                       BT=bt, GP=gp, num_warps=4)
        u = u.t()
        mult = torch.cat([scale[0].expand(4), scale[1].expand(4), alpha.reshape(1).expand(8)])
        dlogits = u * mult
        dscale = torch.stack([(logits[:, :4] * u[:, :4]).sum(), (logits[:, 4:8] * u[:, 4:8]).sum()])
        return dlogits, dscale, part.sum(0).t(), None, None


def ohc_coefficients(logits, scale, bias, rates, alpha):
    """logits [B, L, 16], scale [2], bias [G, 16], rates [G], alpha [] (all FP32) ->
    a, R, b with a, b [B, L, G, 4], R [B, L, G, 4, 4] (component-major storage)."""
    B, L, _ = logits.shape
    G = bias.shape[0]
    a, b, R = _OhcCoefficients.apply(logits.reshape(B * L, 16).contiguous(), scale.contiguous(),
                                     bias.contiguous(), rates.contiguous(), alpha.reshape(()).contiguous())
    return a.view(B, L, G, 4), R.view(B, L, G, 4, 4), b.view(B, L, G, 4)


# ---------------------------------------------------------------------------------------------
# mHC transport: Sinkhorn-Knopp on 4 x 4 logits (column then row log-normalization, `iterations` times).
# Forward keeps the iterates in registers and saves only the logits; backward recomputes them into a
# transient per-program scratch (persistent programs, ~86 MB at 20 iterations) and runs the normalizations
# in reverse:
#   y = x - lse(x) along an axis  ->  dx = dy - exp(y) * sum(dy) along that axis.


@triton.jit
def _lse(x, axis: tl.constexpr):
    mx = tl.max(x, axis=axis, keep_dims=True)
    return mx + tl.log(tl.sum(tl.exp(x - mx), axis=axis, keep_dims=True))


@triton.jit
def _sinkhorn_fwd_kernel(z_ptr, out_ptr, T, ITERS: tl.constexpr, BT: tl.constexpr):
    ot = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    ii = tl.arange(0, 4)
    off = ot[:, None, None] * 16 + ii[None, :, None] * 4 + ii[None, None, :]
    m = (ot < T)[:, None, None]
    x = tl.load(z_ptr + off, mask=m, other=0.0)
    for _ in range(ITERS):
        x = x - _lse(x, 1)
        x = x - _lse(x, 2)
    tl.store(out_ptr + off, tl.exp(x), mask=m)


@triton.jit
def _sinkhorn_bwd_kernel(z_ptr, dout_ptr, dz_ptr, hist_ptr, T, NBLK, ITERS: tl.constexpr, BT: tl.constexpr):
    pid = tl.program_id(0)
    ii = tl.arange(0, 4)
    loc = tl.arange(0, BT)[:, None, None] * 16 + ii[None, :, None] * 4 + ii[None, None, :]
    hp = hist_ptr + pid.to(tl.int64) * (2 * ITERS * BT * 16) + loc
    for blk in range(pid, NBLK, tl.num_programs(0)):
        ot = blk.to(tl.int64) * BT + tl.arange(0, BT)
        off = ot[:, None, None] * 16 + ii[None, :, None] * 4 + ii[None, None, :]
        m = (ot < T)[:, None, None]
        x = tl.load(z_ptr + off, mask=m, other=0.0)
        for k in range(ITERS):
            x = x - _lse(x, 1)
            tl.store(hp + (2 * k) * (BT * 16), x)
            x = x - _lse(x, 2)
            tl.store(hp + (2 * k + 1) * (BT * 16), x)
        g = tl.load(dout_ptr + off, mask=m, other=0.0) * tl.exp(x)
        for kk in range(ITERS):
            k = ITERS - 1 - kk
            y = tl.load(hp + (2 * k + 1) * (BT * 16))
            g = g - tl.exp(y) * tl.sum(g, axis=2, keep_dims=True)
            y = tl.load(hp + (2 * k) * (BT * 16))
            g = g - tl.exp(y) * tl.sum(g, axis=1, keep_dims=True)
        tl.store(dz_ptr + off, g, mask=m)


SINKHORN_BT = 32


class _Sinkhorn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z, iterations):
        T = z.shape[0]
        out = torch.empty_like(z)
        _sinkhorn_fwd_kernel[(triton.cdiv(T, SINKHORN_BT),)](z, out, T, ITERS=iterations, BT=SINKHORN_BT, num_warps=2)
        ctx.iterations = iterations
        ctx.save_for_backward(z)
        return out

    @staticmethod
    def backward(ctx, dout):
        (z,) = ctx.saved_tensors
        T = z.shape[0]
        it = ctx.iterations
        nblk = triton.cdiv(T, SINKHORN_BT)
        nprog = min(nblk, 8 * torch.cuda.get_device_properties(z.device).multi_processor_count)
        hist = torch.empty(nprog * 2 * it * SINKHORN_BT * 16, device=z.device, dtype=torch.float32)
        dz = torch.empty_like(z)
        _sinkhorn_bwd_kernel[(nprog,)](z, dout.contiguous(), dz, hist, T, nblk, ITERS=it, BT=SINKHORN_BT,
                                       num_warps=2)
        return dz, None


def sinkhorn(logits: torch.Tensor, iterations: int) -> torch.Tensor:
    """FP32 logits [..., 4, 4] -> approximately doubly stochastic [..., 4, 4]."""
    shape = logits.shape
    return _Sinkhorn.apply(logits.float().reshape(-1, 16).contiguous(), iterations).view(shape)
