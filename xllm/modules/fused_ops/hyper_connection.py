"""Triton kernels for hyper-connection streams (xllm/modules/residual/hyper_connection.py).

Streams X [T, N, D] stay in BF16 (or FP16); FP32 lives only in registers. Coefficients are FP32 and
contiguous: a, b [T, N], R [T, N, N] with destination rows.

    hc_logits(Xf, W, eps)     rstd(Xf) * (Xf @ W)        Xf = X.flatten(-2) [T, N * D], FP32 out
    hc_read(X, a)             h  = sum_n a_n X_n          [T, D]
    hc_write(X, R, b, y)      X' = R X + b y              [T, N, D], R None = identity
    ihc_coef_fwd / bwd        iHC gates from the logits                  raw launchers, see below
    ohc_coef_fwd / bwd        oHC gates and SO(4) transport from the logits
    sinkhorn(z, iterations)   mHC transport from logits [..., 4, 4]

The projection is a Triton rstd kernel and a cuBLAS BF16 matmul with FP32 accumulate and
output; its backward forms dX = dP W^T - c X in one kernel. Read, write, the oHC
coefficients and Sinkhorn are one kernel each way. Every op saves only its inputs, except
the projection, which also keeps its FP32 rstd [T] and logits [T, n_logits].

The kernel launches are plain functions on contiguous [T, ...] tensors (`logits_fwd` / `logits_bwd`,
`read_fwd` / `read_bwd`, `write_fwd` / `write_fwd_split` / `write_bwd` / `unwrite`, `ihc_coef_fwd` / `ihc_coef_bwd`,
`ohc_coef_fwd` / `ohc_coef_bwd`), so a fused transformer block (xllm/models/fused_blocks/transformer/hyper_connection.py)
can call them from its own forward and backward; the autograd Functions here wrap them for the module path.
"""

import torch
import triton
import triton.language as tl


MAX_STREAMS = 4  # the stream kernels tile four streams per program; fewer are masked, more are not supported


def supported(x: torch.Tensor) -> bool:
    """Fused path: CUDA BF16 / FP16 streams [..., N, D]. The kernels have no path for N > MAX_STREAMS, so that is an
    error rather than a silent fall-back to the (several times larger) torch reference path."""
    if not (x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)):
        return False
    assert x.dim() >= 2 and x.shape[-2] <= MAX_STREAMS, f"the stream kernels support at most {MAX_STREAMS} streams, got {tuple(x.shape)}"
    return True


def _check_streams(x: torch.Tensor):
    """The raw launchers' contract: contiguous [T, N, D] with N <= MAX_STREAMS (the kernels mask n >= N)."""
    assert x.dim() == 3 and x.is_contiguous() and x.shape[1] <= MAX_STREAMS, \
        f"streams must be contiguous [T, N <= {MAX_STREAMS}, D], got {tuple(x.shape)}"


def _check_coef(logits: torch.Tensor, n_logits: int, scale: torch.Tensor, bias: torch.Tensor):
    """The coefficient kernels' contract: FP32 contiguous logits [T, n_logits], scale [2], bias [n_logits]."""
    assert logits.dim() == 2 and logits.shape[1] == n_logits and logits.dtype == torch.float32 and logits.is_contiguous(), \
        f"expected FP32 contiguous logits [T, {n_logits}], got {tuple(logits.shape)} {logits.dtype}"
    assert scale.shape == (2,) and bias.shape == (n_logits,) and scale.is_contiguous() and bias.is_contiguous()


def _tile(D: int, max_tile: int = 1024) -> int:
    """Channel tile of a stream op: D padded to a power of two when it fits, else a power of two that
    divides D (1536 = 3 x 512), else a masked max_tile."""
    if D <= max_tile:
        return triton.next_power_of_2(D)
    pow2 = D & -D  # largest power of two dividing D
    return min(pow2, max_tile) if pow2 >= 256 else max_tile


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


def logits_fwd(xf, w, eps):
    """xf [T, K] contiguous streams, w [K, NL] contiguous in xf's dtype -> FP32 logits [T, NL], rstd [T]."""
    T, K = xf.shape
    rstd = torch.empty(T, device=xf.device, dtype=torch.float32)
    _rstd_kernel[(T,)](xf, rstd, K, eps, BLOCK=min(4096, triton.next_power_of_2(K)), num_warps=8)
    logits = torch.mm(xf, w, out_dtype=torch.float32)
    return logits.mul_(rstd[:, None]), rstd


def logits_bwd(xf, w, rstd, logits, grad, need_dx=True, need_dw=True):
    """-> dx [T, K] in xf's dtype, dw [K, NL] FP32 (None when not needed)."""
    T, K = xf.shape
    NL = w.shape[1]
    g = grad.reshape(T, NL).float()
    c = (g * logits).sum(-1) * rstd.square() / K  # (g . p) rstd^3 / K for the unscaled p = logits / rstd
    dp = (g * rstd[:, None]).to(xf.dtype).contiguous()
    dw = torch.mm(xf.t(), dp, out_dtype=torch.float32) if need_dw else None
    dx = None
    if need_dx:
        dx = torch.empty_like(xf)
        BT, BK = 32, 128
        grid = (triton.cdiv(T, BT), triton.cdiv(K, BK))
        _proj_dx_kernel[grid](xf, dp, w.t().contiguous(), c, dx, T, K, NL, BT=BT, BK=BK,
                              LP=max(16, triton.next_power_of_2(NL)), num_warps=4)
    return dx, dw


class _Logits(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, eps):
        K, NL = weight.shape
        xf = x.reshape(-1, K)
        w = weight.to(xf.dtype).contiguous()
        logits, rstd = logits_fwd(xf, w, eps)
        ctx.save_for_backward(xf, w, rstd, logits)
        ctx.x_shape, ctx.w_dtype = x.shape, weight.dtype
        return logits.view(*x.shape[:-1], NL)

    @staticmethod
    def backward(ctx, grad):
        xf, w, rstd, logits = ctx.saved_tensors
        dx, dw = logits_bwd(xf, w, rstd, logits, grad, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        if dx is not None:
            dx = dx.view(ctx.x_shape)
        if dw is not None:
            dw = dw.to(ctx.w_dtype)
        return dx, dw, None


def hc_logits(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """[..., K] streams (flattened N * D, contiguous), weight [K, NL] -> FP32 logits [..., NL]."""
    return _Logits.apply(x.contiguous(), weight, eps)


# ---------------------------------------------------------------------------------------------
# stream read and write. One program per token over channel tiles [CH]; the N <= 4 streams are
# explicit (streams n >= N are masked off), the coefficient reductions accumulate elementwise
# over the tiles and reduce once.


@triton.jit
def _ld(ptr, t, n, N, D, d, m):
    return tl.load(ptr + (t * N + n) * D + d, mask=m & (n < N), other=0.0).to(tl.float32)


@triton.jit
def _st(ptr, t, n, N, D, d, m, v):
    tl.store(ptr + (t * N + n) * D + d, v.to(ptr.dtype.element_ty), mask=m & (n < N))


@triton.jit
def _ld2(hi_ptr, lo_ptr, t, n, N, D, d, m, HAS_LO: tl.constexpr):
    # a stream kept as hi + lo (two low-precision tensors, hi = round(x), lo = round(x - hi)); lo optional
    x = _ld(hi_ptr, t, n, N, D, d, m)
    if HAS_LO:
        x += _ld(lo_ptr, t, n, N, D, d, m)
    return x


@triton.jit
def _st2(hi_ptr, lo_ptr, t, n, N, D, d, m, v):
    # split an FP32 value into hi = round(v) and lo = round(v - hi) in the tensors' dtype
    h = v.to(hi_ptr.dtype.element_ty)
    tl.store(hi_ptr + (t * N + n) * D + d, h, mask=m & (n < N))
    tl.store(lo_ptr + (t * N + n) * D + d, (v - h.to(tl.float32)).to(lo_ptr.dtype.element_ty), mask=m & (n < N))


@triton.jit
def _ldc(ptr, t, n, N):
    return tl.load(ptr + t * N + n, mask=n < N, other=0.0)


@triton.jit
def _stc(ptr, t, n, N, acc):
    tl.store(ptr + t * N + n, tl.sum(acc, axis=0), mask=n < N)


@triton.jit
def _ldr(ptr, t, n, m, N, HAS_R: tl.constexpr):
    return tl.load(ptr + (t * N + n) * N + m, mask=(n < N) & (m < N) & HAS_R, other=0.0)


@triton.jit
def _str(ptr, t, n, m, N, acc):
    tl.store(ptr + (t * N + n) * N + m, tl.sum(acc, axis=0), mask=(n < N) & (m < N))


@triton.jit
def _coefs(r_ptr, b_ptr, t, N, HAS_R: tl.constexpr):
    # one token's write gates b_n and transport entries R[n, m] (masked off without a transport)
    b0 = _ldc(b_ptr, t, 0, N)
    b1 = _ldc(b_ptr, t, 1, N)
    b2 = _ldc(b_ptr, t, 2, N)
    b3 = _ldc(b_ptr, t, 3, N)
    r00 = _ldr(r_ptr, t, 0, 0, N, HAS_R)
    r01 = _ldr(r_ptr, t, 0, 1, N, HAS_R)
    r02 = _ldr(r_ptr, t, 0, 2, N, HAS_R)
    r03 = _ldr(r_ptr, t, 0, 3, N, HAS_R)
    r10 = _ldr(r_ptr, t, 1, 0, N, HAS_R)
    r11 = _ldr(r_ptr, t, 1, 1, N, HAS_R)
    r12 = _ldr(r_ptr, t, 1, 2, N, HAS_R)
    r13 = _ldr(r_ptr, t, 1, 3, N, HAS_R)
    r20 = _ldr(r_ptr, t, 2, 0, N, HAS_R)
    r21 = _ldr(r_ptr, t, 2, 1, N, HAS_R)
    r22 = _ldr(r_ptr, t, 2, 2, N, HAS_R)
    r23 = _ldr(r_ptr, t, 2, 3, N, HAS_R)
    r30 = _ldr(r_ptr, t, 3, 0, N, HAS_R)
    r31 = _ldr(r_ptr, t, 3, 1, N, HAS_R)
    r32 = _ldr(r_ptr, t, 3, 2, N, HAS_R)
    r33 = _ldr(r_ptr, t, 3, 3, N, HAS_R)
    return b0, b1, b2, b3, r00, r01, r02, r03, r10, r11, r12, r13, r20, r21, r22, r23, r30, r31, r32, r33


@triton.jit
def _read_fwd_kernel(x_ptr, a_ptr, h_ptr, N, D, CH: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    a0 = _ldc(a_ptr, t, 0, N)
    a1 = _ldc(a_ptr, t, 1, N)
    a2 = _ldc(a_ptr, t, 2, N)
    a3 = _ldc(a_ptr, t, 3, N)
    ci = tl.arange(0, CH)
    for d0 in range(0, D, CH):
        d = d0 + ci
        m = d < D
        h = (a0 * _ld(x_ptr, t, 0, N, D, d, m) + a1 * _ld(x_ptr, t, 1, N, D, d, m)
             + a2 * _ld(x_ptr, t, 2, N, D, d, m) + a3 * _ld(x_ptr, t, 3, N, D, d, m))
        tl.store(h_ptr + t * D + d, h.to(h_ptr.dtype.element_ty), mask=m)


@triton.jit
def _read_bwd_kernel(x_ptr, a_ptr, dh_ptr, dx_ptr, da_ptr, N, D, CH: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    a0 = _ldc(a_ptr, t, 0, N)
    a1 = _ldc(a_ptr, t, 1, N)
    a2 = _ldc(a_ptr, t, 2, N)
    a3 = _ldc(a_ptr, t, 3, N)
    p0 = tl.zeros([CH], dtype=tl.float32)  # da_n = sum_d x_n dh
    p1 = tl.zeros([CH], dtype=tl.float32)
    p2 = tl.zeros([CH], dtype=tl.float32)
    p3 = tl.zeros([CH], dtype=tl.float32)
    ci = tl.arange(0, CH)
    for d0 in range(0, D, CH):
        d = d0 + ci
        m = d < D
        dh = tl.load(dh_ptr + t * D + d, mask=m, other=0.0).to(tl.float32)
        _st(dx_ptr, t, 0, N, D, d, m, a0 * dh)
        _st(dx_ptr, t, 1, N, D, d, m, a1 * dh)
        _st(dx_ptr, t, 2, N, D, d, m, a2 * dh)
        _st(dx_ptr, t, 3, N, D, d, m, a3 * dh)
        p0 += _ld(x_ptr, t, 0, N, D, d, m) * dh
        p1 += _ld(x_ptr, t, 1, N, D, d, m) * dh
        p2 += _ld(x_ptr, t, 2, N, D, d, m) * dh
        p3 += _ld(x_ptr, t, 3, N, D, d, m) * dh
    _stc(da_ptr, t, 0, N, p0)
    _stc(da_ptr, t, 1, N, p1)
    _stc(da_ptr, t, 2, N, p2)
    _stc(da_ptr, t, 3, N, p3)


def read_fwd(x, a):
    """x [T, N, D], a [T, N] FP32, both contiguous -> h [T, D] in x's dtype."""
    _check_streams(x)
    T, N, D = x.shape
    h = torch.empty(T, D, device=x.device, dtype=x.dtype)
    _read_fwd_kernel[(T,)](x, a, h, N, D, CH=_tile(D), num_warps=4)
    return h


def read_bwd(x, a, dh):
    """-> dx [T, N, D], da [T, N]."""
    _check_streams(x)
    T, N, D = x.shape
    dx = torch.empty_like(x)
    da = torch.empty_like(a)
    _read_bwd_kernel[(T,)](x, a, dh.contiguous(), dx, da, N, D, CH=_tile(D), num_warps=2)
    return dx, da


class _Read(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, a):
        ctx.save_for_backward(x, a)
        return read_fwd(x, a)

    @staticmethod
    def backward(ctx, dh):
        x, a = ctx.saved_tensors
        return read_bwd(x, a, dh)


def hc_read(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """x [B, L, N, D], a [B, L, N] (FP32) -> h [B, L, D] in x's dtype."""
    B, L, N, D = x.shape
    h = _Read.apply(x.contiguous().view(B * L, N, D), a.float().reshape(B * L, N).contiguous())
    return h.view(B, L, D)


@triton.jit
def _mix(r0, r1, r2, r3, x0, x1, x2, x3):
    # sum_m R[n, m] x_m for one destination row n
    return r0 * x0 + r1 * x1 + r2 * x2 + r3 * x3


@triton.jit
def _write_fwd_kernel(x_ptr, xlo_ptr, r_ptr, b_ptr, y_ptr, out_ptr, outlo_ptr, N, D, CH: tl.constexpr, HAS_R: tl.constexpr,
                      HAS_LO: tl.constexpr, SPLIT: tl.constexpr):
    # X' = R X + b y. HAS_LO: X is kept as hi + lo; SPLIT: X' is stored as hi + lo (else rounded once into out_ptr).
    t = tl.program_id(0).to(tl.int64)
    (b0, b1, b2, b3, r00, r01, r02, r03, r10, r11, r12, r13, r20, r21, r22, r23,
     r30, r31, r32, r33) = _coefs(r_ptr, b_ptr, t, N, HAS_R)
    ci = tl.arange(0, CH)
    for d0 in range(0, D, CH):
        d = d0 + ci
        m = d < D
        y = tl.load(y_ptr + t * D + d, mask=m, other=0.0).to(tl.float32)
        x0 = _ld2(x_ptr, xlo_ptr, t, 0, N, D, d, m, HAS_LO)
        x1 = _ld2(x_ptr, xlo_ptr, t, 1, N, D, d, m, HAS_LO)
        x2 = _ld2(x_ptr, xlo_ptr, t, 2, N, D, d, m, HAS_LO)
        x3 = _ld2(x_ptr, xlo_ptr, t, 3, N, D, d, m, HAS_LO)
        if HAS_R:
            o0 = _mix(r00, r01, r02, r03, x0, x1, x2, x3) + b0 * y
            o1 = _mix(r10, r11, r12, r13, x0, x1, x2, x3) + b1 * y
            o2 = _mix(r20, r21, r22, r23, x0, x1, x2, x3) + b2 * y
            o3 = _mix(r30, r31, r32, r33, x0, x1, x2, x3) + b3 * y
        else:
            o0 = x0 + b0 * y
            o1 = x1 + b1 * y
            o2 = x2 + b2 * y
            o3 = x3 + b3 * y
        if SPLIT:
            _st2(out_ptr, outlo_ptr, t, 0, N, D, d, m, o0)
            _st2(out_ptr, outlo_ptr, t, 1, N, D, d, m, o1)
            _st2(out_ptr, outlo_ptr, t, 2, N, D, d, m, o2)
            _st2(out_ptr, outlo_ptr, t, 3, N, D, d, m, o3)
        else:
            _st(out_ptr, t, 0, N, D, d, m, o0)
            _st(out_ptr, t, 1, N, D, d, m, o1)
            _st(out_ptr, t, 2, N, D, d, m, o2)
            _st(out_ptr, t, 3, N, D, d, m, o3)


@triton.jit
def _write_bwd_kernel(x_ptr, r_ptr, b_ptr, y_ptr, go_ptr, dx_ptr, dr_ptr, db_ptr, dy_ptr, N, D,
                      CH: tl.constexpr, HAS_R: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    (b0, b1, b2, b3, r00, r01, r02, r03, r10, r11, r12, r13, r20, r21, r22, r23,
     r30, r31, r32, r33) = _coefs(r_ptr, b_ptr, t, N, HAS_R)
    q0 = tl.zeros([CH], dtype=tl.float32)  # db_n = sum_d g_n y
    q1 = tl.zeros([CH], dtype=tl.float32)
    q2 = tl.zeros([CH], dtype=tl.float32)
    q3 = tl.zeros([CH], dtype=tl.float32)
    # dR[n, m] = sum_d g_n x_m (only with a transport)
    p00 = tl.zeros([CH], dtype=tl.float32)
    p01 = tl.zeros([CH], dtype=tl.float32)
    p02 = tl.zeros([CH], dtype=tl.float32)
    p03 = tl.zeros([CH], dtype=tl.float32)
    p10 = tl.zeros([CH], dtype=tl.float32)
    p11 = tl.zeros([CH], dtype=tl.float32)
    p12 = tl.zeros([CH], dtype=tl.float32)
    p13 = tl.zeros([CH], dtype=tl.float32)
    p20 = tl.zeros([CH], dtype=tl.float32)
    p21 = tl.zeros([CH], dtype=tl.float32)
    p22 = tl.zeros([CH], dtype=tl.float32)
    p23 = tl.zeros([CH], dtype=tl.float32)
    p30 = tl.zeros([CH], dtype=tl.float32)
    p31 = tl.zeros([CH], dtype=tl.float32)
    p32 = tl.zeros([CH], dtype=tl.float32)
    p33 = tl.zeros([CH], dtype=tl.float32)
    ci = tl.arange(0, CH)
    for d0 in range(0, D, CH):
        d = d0 + ci
        m = d < D
        y = tl.load(y_ptr + t * D + d, mask=m, other=0.0).to(tl.float32)
        g0 = _ld(go_ptr, t, 0, N, D, d, m)
        g1 = _ld(go_ptr, t, 1, N, D, d, m)
        g2 = _ld(go_ptr, t, 2, N, D, d, m)
        g3 = _ld(go_ptr, t, 3, N, D, d, m)
        tl.store(dy_ptr + t * D + d, (b0 * g0 + b1 * g1 + b2 * g2 + b3 * g3).to(dy_ptr.dtype.element_ty), mask=m)
        q0 += g0 * y
        q1 += g1 * y
        q2 += g2 * y
        q3 += g3 * y
        if HAS_R:
            x0 = _ld(x_ptr, t, 0, N, D, d, m)
            x1 = _ld(x_ptr, t, 1, N, D, d, m)
            x2 = _ld(x_ptr, t, 2, N, D, d, m)
            x3 = _ld(x_ptr, t, 3, N, D, d, m)
            # dX_m = sum_n R[n, m] g_n
            _st(dx_ptr, t, 0, N, D, d, m, _mix(r00, r10, r20, r30, g0, g1, g2, g3))
            _st(dx_ptr, t, 1, N, D, d, m, _mix(r01, r11, r21, r31, g0, g1, g2, g3))
            _st(dx_ptr, t, 2, N, D, d, m, _mix(r02, r12, r22, r32, g0, g1, g2, g3))
            _st(dx_ptr, t, 3, N, D, d, m, _mix(r03, r13, r23, r33, g0, g1, g2, g3))
            p00 += g0 * x0
            p01 += g0 * x1
            p02 += g0 * x2
            p03 += g0 * x3
            p10 += g1 * x0
            p11 += g1 * x1
            p12 += g1 * x2
            p13 += g1 * x3
            p20 += g2 * x0
            p21 += g2 * x1
            p22 += g2 * x2
            p23 += g2 * x3
            p30 += g3 * x0
            p31 += g3 * x1
            p32 += g3 * x2
            p33 += g3 * x3
    _stc(db_ptr, t, 0, N, q0)
    _stc(db_ptr, t, 1, N, q1)
    _stc(db_ptr, t, 2, N, q2)
    _stc(db_ptr, t, 3, N, q3)
    if HAS_R:
        _str(dr_ptr, t, 0, 0, N, p00)
        _str(dr_ptr, t, 0, 1, N, p01)
        _str(dr_ptr, t, 0, 2, N, p02)
        _str(dr_ptr, t, 0, 3, N, p03)
        _str(dr_ptr, t, 1, 0, N, p10)
        _str(dr_ptr, t, 1, 1, N, p11)
        _str(dr_ptr, t, 1, 2, N, p12)
        _str(dr_ptr, t, 1, 3, N, p13)
        _str(dr_ptr, t, 2, 0, N, p20)
        _str(dr_ptr, t, 2, 1, N, p21)
        _str(dr_ptr, t, 2, 2, N, p22)
        _str(dr_ptr, t, 2, 3, N, p23)
        _str(dr_ptr, t, 3, 0, N, p30)
        _str(dr_ptr, t, 3, 1, N, p31)
        _str(dr_ptr, t, 3, 2, N, p32)
        _str(dr_ptr, t, 3, 3, N, p33)


def write_fwd(x, R, b, y):
    """x [T, N, D], R [T, N, N] FP32 or None (identity), b [T, N] FP32, y [T, D] in x's dtype, all contiguous
    -> X' [T, N, D], rounded once to x's dtype."""
    return write_fwd_split(x, None, R, b, y, split=False)[0]


def write_fwd_split(hi, lo, R, b, y, split=True):
    """`write_fwd` on streams kept as hi + lo (lo None: hi alone), -> (hi', lo') of X': hi' = round(X'), the streams the
    next kernels read, and lo' = round(X' - hi'), so that hi' + lo' holds X' to ~2^-17 and the write can be inverted
    without the single-rounding drift. `split=False`: X' rounded once, (hi', None)."""
    _check_streams(hi)
    T, N, D = hi.shape
    out = torch.empty_like(hi)
    outlo = torch.empty_like(hi) if split else None
    has_r = R is not None
    _write_fwd_kernel[(T,)](hi, lo if lo is not None else hi, R if has_r else b, b, y, out, outlo if split else out, N, D,
                            CH=_tile(D), HAS_R=has_r, HAS_LO=lo is not None, SPLIT=split, num_warps=4)
    return out, outlo


def write_bwd(x, R, b, y, go):
    """-> dx, dR, db [T, N], dy [T, D]. For the identity transport dX = dX' and dR does not exist: dx and dR are None."""
    _check_streams(x)
    T, N, D = x.shape
    has_r = R is not None
    # the transport backward keeps 20 accumulator tiles live (H200 sweep, D = 1536): small tiles, 2 warps
    tile, warps = ((256, 2) if D > 512 else (1024, 2)) if has_r else (1024, 4)
    go = go.contiguous()
    db = torch.empty_like(b)
    dy = torch.empty_like(y)
    if has_r:
        dx = torch.empty_like(x)
        dR = torch.empty_like(R)
    else:
        dx, dR = None, None  # identity transport: dX = dX', nothing to write
    _write_bwd_kernel[(T,)](x, R if has_r else b, b, y, go, dx if has_r else go, dR if has_r else b, db, dy, N, D,
                            CH=_tile(D, tile), HAS_R=has_r, num_warps=warps)
    return dx, dR, db, dy


@triton.jit
def _unwrite_kernel(xn_ptr, xnlo_ptr, r_ptr, b_ptr, y_ptr, out_ptr, outlo_ptr, N, D, CH: tl.constexpr, HAS_R: tl.constexpr,
                    HAS_LO: tl.constexpr):
    # X = R^T (X' - b y) for an orthogonal (or identity) transport: the streams before a write, from the streams after it,
    # X' read as hi + lo (HAS_LO) and X stored as hi + lo. In place when out_ptr is xn_ptr: one program per token, each
    # tile loaded before it is stored.
    t = tl.program_id(0).to(tl.int64)
    (b0, b1, b2, b3, r00, r01, r02, r03, r10, r11, r12, r13, r20, r21, r22, r23,
     r30, r31, r32, r33) = _coefs(r_ptr, b_ptr, t, N, HAS_R)
    ci = tl.arange(0, CH)
    for d0 in range(0, D, CH):
        d = d0 + ci
        m = d < D
        y = tl.load(y_ptr + t * D + d, mask=m, other=0.0).to(tl.float32)
        z0 = _ld2(xn_ptr, xnlo_ptr, t, 0, N, D, d, m, HAS_LO) - b0 * y
        z1 = _ld2(xn_ptr, xnlo_ptr, t, 1, N, D, d, m, HAS_LO) - b1 * y
        z2 = _ld2(xn_ptr, xnlo_ptr, t, 2, N, D, d, m, HAS_LO) - b2 * y
        z3 = _ld2(xn_ptr, xnlo_ptr, t, 3, N, D, d, m, HAS_LO) - b3 * y
        if HAS_R:
            # X_m = sum_n R[n, m] z_n
            x0 = _mix(r00, r10, r20, r30, z0, z1, z2, z3)
            x1 = _mix(r01, r11, r21, r31, z0, z1, z2, z3)
            x2 = _mix(r02, r12, r22, r32, z0, z1, z2, z3)
            x3 = _mix(r03, r13, r23, r33, z0, z1, z2, z3)
        else:
            x0 = z0
            x1 = z1
            x2 = z2
            x3 = z3
        _st2(out_ptr, outlo_ptr, t, 0, N, D, d, m, x0)
        _st2(out_ptr, outlo_ptr, t, 1, N, D, d, m, x1)
        _st2(out_ptr, outlo_ptr, t, 2, N, D, d, m, x2)
        _st2(out_ptr, outlo_ptr, t, 3, N, D, d, m, x3)


def unwrite(hi_next, lo_next, R, b, y, inplace=False):
    """Inverse of `write_fwd_split` for an orthogonal or identity transport: X' as (hi, lo) [T, N, D] (lo None: a
    singly rounded X'), R [T, N, N] FP32 or None, b [T, N] FP32, y [T, D] -> X as (hi, lo): hi for the kernels that
    consume the streams, hi + lo to chain further inversions. `inplace` overwrites hi_next and lo_next with the result."""
    _check_streams(hi_next)
    T, N, D = hi_next.shape
    has_r = R is not None
    if inplace:
        assert lo_next is not None
        hi, lo = hi_next, lo_next
    else:
        hi, lo = torch.empty_like(hi_next), torch.empty_like(hi_next)
    _unwrite_kernel[(T,)](hi_next, lo_next if lo_next is not None else hi_next, R if has_r else b, b, y, hi, lo, N, D,
                          CH=_tile(D, 512), HAS_R=has_r, HAS_LO=lo_next is not None, num_warps=4)
    return hi, lo


class _Write(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, R, b, y):
        ctx.save_for_backward(x, R, b, y)
        return write_fwd(x, R, b, y)

    @staticmethod
    def backward(ctx, go):
        x, R, b, y = ctx.saved_tensors
        dx, dR, db, dy = write_bwd(x, R, b, y, go)
        return go if dx is None else dx, dR, db, dy


def hc_write(x: torch.Tensor, R, b: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x [B, L, N, D], R [B, L, N, N] FP32 or None, b [B, L, N] FP32, y [B, L, D] -> X' [B, L, N, D]."""
    B, L, N, D = x.shape
    T = B * L
    R = None if R is None else R.float().reshape(T, N, N).contiguous()
    out = _Write.apply(x.contiguous().view(T, N, D), R, b.float().reshape(T, N).contiguous(),
                       y.to(x.dtype).contiguous().view(T, D))
    return out.view(B, L, N, D)


# ---------------------------------------------------------------------------------------------
# oHC coefficients from the logits l [T, 16] = (a: 4, b: 4, rotation: 8):
#   a = sigmoid(s_a l_a + bias),  b = 2 sigmoid(s_b l_b + bias)
#   R = SO(4)(alpha l_r + bias):  q = (1 + z0, z1, z2, z3) / |.|,  r = (1 + z4, ..) / |.|,  R = L(q) Rc(r)
# Outputs a, b [T, 4], R [T, 4, 4]; the backward writes u = dL / dz [T, 16], the gradient of the pre-activations.


@triton.jit
def _ohc_rotation(lp, bias_ptr, alpha, tm, eps):
    z0 = alpha * tl.load(lp + 8, mask=tm, other=0.0) + tl.load(bias_ptr + 8)
    z1 = alpha * tl.load(lp + 9, mask=tm, other=0.0) + tl.load(bias_ptr + 9)
    z2 = alpha * tl.load(lp + 10, mask=tm, other=0.0) + tl.load(bias_ptr + 10)
    z3 = alpha * tl.load(lp + 11, mask=tm, other=0.0) + tl.load(bias_ptr + 11)
    z4 = alpha * tl.load(lp + 12, mask=tm, other=0.0) + tl.load(bias_ptr + 12)
    z5 = alpha * tl.load(lp + 13, mask=tm, other=0.0) + tl.load(bias_ptr + 13)
    z6 = alpha * tl.load(lp + 14, mask=tm, other=0.0) + tl.load(bias_ptr + 14)
    z7 = alpha * tl.load(lp + 15, mask=tm, other=0.0) + tl.load(bias_ptr + 15)
    vw = 1.0 + z0
    va = 1.0 + z4
    sq = tl.rsqrt(vw * vw + z1 * z1 + z2 * z2 + z3 * z3 + eps)
    sr = tl.rsqrt(va * va + z5 * z5 + z6 * z6 + z7 * z7 + eps)
    # the raw quaternions v, their inverse norms s and the unit quaternions q = v s, r = v s (the backward needs all)
    return vw, z1, z2, z3, sq, va, z5, z6, z7, sr, vw * sq, z1 * sq, z2 * sq, z3 * sq, va * sr, z5 * sr, z6 * sr, z7 * sr


@triton.jit
def _gates_fwd(lp, scale_ptr, bias_ptr, a_ptr, b_ptr, ot, tm):
    # a_n = sigmoid(s_a l_n + bias_n), b_n = 2 sigmoid(s_b l_{4+n} + bias_{4+n}) from the first 8 logits of a token
    sa = tl.load(scale_ptr)
    sb = tl.load(scale_ptr + 1)
    for n in tl.static_range(4):
        za = sa * tl.load(lp + n, mask=tm, other=0.0) + tl.load(bias_ptr + n)
        tl.store(a_ptr + ot * 4 + n, tl.sigmoid(za), mask=tm)
        zb = sb * tl.load(lp + 4 + n, mask=tm, other=0.0) + tl.load(bias_ptr + 4 + n)
        tl.store(b_ptr + ot * 4 + n, 2.0 * tl.sigmoid(zb), mask=tm)


@triton.jit
def _gates_bwd(lp, scale_ptr, bias_ptr, da_ptr, db_ptr, dl_ptr, ps_ptr, pb_ptr, ot, tm):
    # dlogits of the 8 gate logits, and this program's partial sums of dscale (2) and dbias (8)
    sa = tl.load(scale_ptr)
    sb = tl.load(scale_ptr + 1)
    psa = 0.0
    psb = 0.0
    for n in tl.static_range(4):
        la = tl.load(lp + n, mask=tm, other=0.0)
        a = tl.sigmoid(sa * la + tl.load(bias_ptr + n))
        ua = tl.load(da_ptr + ot * 4 + n, mask=tm, other=0.0) * a * (1.0 - a)  # dL / dz_a
        tl.store(dl_ptr + n, ua * sa, mask=tm)
        tl.store(pb_ptr + n, tl.sum(ua, axis=0))
        psa += tl.sum(ua * la, axis=0)
        lb = tl.load(lp + 4 + n, mask=tm, other=0.0)
        s = tl.sigmoid(sb * lb + tl.load(bias_ptr + 4 + n))
        ub = tl.load(db_ptr + ot * 4 + n, mask=tm, other=0.0) * 2.0 * s * (1.0 - s)  # dL / dz_b
        tl.store(dl_ptr + 4 + n, ub * sb, mask=tm)
        tl.store(pb_ptr + 4 + n, tl.sum(ub, axis=0))
        psb += tl.sum(ub * lb, axis=0)
    tl.store(ps_ptr, psa)
    tl.store(ps_ptr + 1, psb)


@triton.jit
def _ihc_coef_fwd_kernel(l_ptr, scale_ptr, bias_ptr, a_ptr, b_ptr, T, BT: tl.constexpr):
    ot = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    _gates_fwd(l_ptr + ot * 8, scale_ptr, bias_ptr, a_ptr, b_ptr, ot, ot < T)


@triton.jit
def _ihc_coef_bwd_kernel(l_ptr, scale_ptr, bias_ptr, da_ptr, db_ptr, dl_ptr, ps_ptr, pb_ptr, T, BT: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    ot = pid * BT + tl.arange(0, BT)
    _gates_bwd(l_ptr + ot * 8, scale_ptr, bias_ptr, da_ptr, db_ptr, dl_ptr + ot * 8, ps_ptr + pid * 2, pb_ptr + pid * 8,
               ot, ot < T)


@triton.jit
def _ohc_coef_fwd_kernel(l_ptr, scale_ptr, bias_ptr, a_ptr, b_ptr, r_ptr, T, alpha, eps, BT: tl.constexpr):
    ot = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    tm = ot < T
    lp = l_ptr + ot * 16
    _gates_fwd(lp, scale_ptr, bias_ptr, a_ptr, b_ptr, ot, tm)
    _, _, _, _, _, _, _, _, _, _, qw, qx, qy, qz, ra, rb, rc, rd = _ohc_rotation(lp, bias_ptr, alpha, tm, eps)
    rp = r_ptr + ot * 16
    tl.store(rp + 0, qw * ra + qx * rb + qy * rc + qz * rd, mask=tm)
    tl.store(rp + 1, qw * rb - qx * ra - qy * rd + qz * rc, mask=tm)
    tl.store(rp + 2, qw * rc + qx * rd - qy * ra - qz * rb, mask=tm)
    tl.store(rp + 3, qw * rd - qx * rc + qy * rb - qz * ra, mask=tm)
    tl.store(rp + 4, qx * ra - qw * rb + qz * rc - qy * rd, mask=tm)
    tl.store(rp + 5, qx * rb + qw * ra - qz * rd - qy * rc, mask=tm)
    tl.store(rp + 6, qx * rc - qw * rd - qz * ra + qy * rb, mask=tm)
    tl.store(rp + 7, qx * rd + qw * rc + qz * rb + qy * ra, mask=tm)
    tl.store(rp + 8, qy * ra - qz * rb - qw * rc + qx * rd, mask=tm)
    tl.store(rp + 9, qy * rb + qz * ra + qw * rd + qx * rc, mask=tm)
    tl.store(rp + 10, qy * rc - qz * rd + qw * ra - qx * rb, mask=tm)
    tl.store(rp + 11, qy * rd + qz * rc - qw * rb - qx * ra, mask=tm)
    tl.store(rp + 12, qz * ra + qy * rb - qx * rc - qw * rd, mask=tm)
    tl.store(rp + 13, qz * rb - qy * ra + qx * rd - qw * rc, mask=tm)
    tl.store(rp + 14, qz * rc + qy * rd + qx * ra + qw * rb, mask=tm)
    tl.store(rp + 15, qz * rd - qy * rc - qx * rb + qw * ra, mask=tm)


@triton.jit
def _ohc_coef_bwd_kernel(l_ptr, scale_ptr, bias_ptr, da_ptr, db_ptr, dr_ptr, dl_ptr, ps_ptr, pb_ptr, T, alpha, eps,
                         BT: tl.constexpr):
    # -> dlogits [T, 16]; per program the partial sums of dscale [2] and dbias [16] (the host adds the programs)
    pid = tl.program_id(0).to(tl.int64)
    ot = pid * BT + tl.arange(0, BT)
    tm = ot < T
    lp = l_ptr + ot * 16
    dlp = dl_ptr + ot * 16
    pbp = pb_ptr + pid * 16
    _gates_bwd(lp, scale_ptr, bias_ptr, da_ptr, db_ptr, dlp, ps_ptr + pid * 2, pbp, ot, tm)

    vw, vx, vy, vz, sq, va, vb, vc, vd, sr, qw, qx, qy, qz, ra, rb, rc, rd = _ohc_rotation(lp, bias_ptr, alpha, tm, eps)
    drp = dr_ptr + ot * 16
    dr00 = tl.load(drp + 0, mask=tm, other=0.0)
    dr01 = tl.load(drp + 1, mask=tm, other=0.0)
    dr02 = tl.load(drp + 2, mask=tm, other=0.0)
    dr03 = tl.load(drp + 3, mask=tm, other=0.0)
    dr10 = tl.load(drp + 4, mask=tm, other=0.0)
    dr11 = tl.load(drp + 5, mask=tm, other=0.0)
    dr12 = tl.load(drp + 6, mask=tm, other=0.0)
    dr13 = tl.load(drp + 7, mask=tm, other=0.0)
    dr20 = tl.load(drp + 8, mask=tm, other=0.0)
    dr21 = tl.load(drp + 9, mask=tm, other=0.0)
    dr22 = tl.load(drp + 10, mask=tm, other=0.0)
    dr23 = tl.load(drp + 11, mask=tm, other=0.0)
    dr30 = tl.load(drp + 12, mask=tm, other=0.0)
    dr31 = tl.load(drp + 13, mask=tm, other=0.0)
    dr32 = tl.load(drp + 14, mask=tm, other=0.0)
    dr33 = tl.load(drp + 15, mask=tm, other=0.0)
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
    # dL / dz of the 8 rotation logits: dlogits = alpha * dz, dbias = sum over the tokens of dz
    u0 = sq * gqw - tq * vw
    u1 = sq * gqx - tq * vx
    u2 = sq * gqy - tq * vy
    u3 = sq * gqz - tq * vz
    u4 = sr * gra - tr * va
    u5 = sr * grb - tr * vb
    u6 = sr * grc - tr * vc
    u7 = sr * grd - tr * vd
    tl.store(dlp + 8, alpha * u0, mask=tm)
    tl.store(dlp + 9, alpha * u1, mask=tm)
    tl.store(dlp + 10, alpha * u2, mask=tm)
    tl.store(dlp + 11, alpha * u3, mask=tm)
    tl.store(dlp + 12, alpha * u4, mask=tm)
    tl.store(dlp + 13, alpha * u5, mask=tm)
    tl.store(dlp + 14, alpha * u6, mask=tm)
    tl.store(dlp + 15, alpha * u7, mask=tm)
    tl.store(pbp + 8, tl.sum(tl.where(tm, u0, 0.0), axis=0))
    tl.store(pbp + 9, tl.sum(tl.where(tm, u1, 0.0), axis=0))
    tl.store(pbp + 10, tl.sum(tl.where(tm, u2, 0.0), axis=0))
    tl.store(pbp + 11, tl.sum(tl.where(tm, u3, 0.0), axis=0))
    tl.store(pbp + 12, tl.sum(tl.where(tm, u4, 0.0), axis=0))
    tl.store(pbp + 13, tl.sum(tl.where(tm, u5, 0.0), axis=0))
    tl.store(pbp + 14, tl.sum(tl.where(tm, u6, 0.0), axis=0))
    tl.store(pbp + 15, tl.sum(tl.where(tm, u7, 0.0), axis=0))


COEF_BT = 1024  # tokens per program of the coefficient kernels


def _coef_bwd_buffers(T, NL, device, da, db):
    """The coefficient backward's outputs and zero stand-ins for None gradients: dlogits [T, NL], the per-program
    partial sums ps [P, 2] (dscale) and pb [P, NL] (dbias), da and db [T, 4]."""
    P = triton.cdiv(T, COEF_BT)
    da = torch.zeros(T, 4, device=device) if da is None else da.contiguous()
    db = torch.zeros(T, 4, device=device) if db is None else db.contiguous()
    return (torch.empty(T, NL, device=device, dtype=torch.float32), torch.empty(P, 2, device=device, dtype=torch.float32),
            torch.empty(P, NL, device=device, dtype=torch.float32), da, db)


def ihc_coef_fwd(logits, scale, bias):
    """logits [T, 8], scale [2], bias [8], all FP32 contiguous -> a [T, 4], b [T, 4]."""
    _check_coef(logits, 8, scale, bias)
    T = logits.shape[0]
    a = torch.empty(T, 4, device=logits.device, dtype=torch.float32)
    b = torch.empty(T, 4, device=logits.device, dtype=torch.float32)
    _ihc_coef_fwd_kernel[(triton.cdiv(T, COEF_BT),)](logits, scale, bias, a, b, T, BT=COEF_BT, num_warps=4)
    return a, b


def ihc_coef_bwd(logits, scale, bias, da, db):
    """-> dlogits [T, 8], dscale [2], dbias [8] (FP32); a None gradient counts as zero."""
    _check_coef(logits, 8, scale, bias)
    T = logits.shape[0]
    dl, ps, pb, da, db = _coef_bwd_buffers(T, 8, logits.device, da, db)
    _ihc_coef_bwd_kernel[(ps.shape[0],)](logits, scale, bias, da, db, dl, ps, pb, T, BT=COEF_BT, num_warps=4)
    return dl, ps.sum(0), pb.sum(0)


def ohc_coef_fwd(logits, scale, bias, alpha, eps):
    """logits [T, 16], scale [2], bias [16], all FP32 contiguous; alpha the rotation logits' scale, eps of the quaternion
    normalization -> a [T, 4], R [T, 4, 4], b [T, 4]."""
    _check_coef(logits, 16, scale, bias)
    T = logits.shape[0]
    dev = logits.device
    a = torch.empty(T, 4, device=dev, dtype=torch.float32)
    b = torch.empty(T, 4, device=dev, dtype=torch.float32)
    R = torch.empty(T, 4, 4, device=dev, dtype=torch.float32)
    _ohc_coef_fwd_kernel[(triton.cdiv(T, COEF_BT),)](logits, scale, bias, a, b, R, T, alpha, eps,
                                                     BT=COEF_BT, num_warps=4)
    return a, R, b


def ohc_coef_bwd(logits, scale, bias, da, dR, db, alpha, eps):
    """-> dlogits [T, 16], dscale [2], dbias [16] (FP32); a None gradient counts as zero."""
    _check_coef(logits, 16, scale, bias)
    T = logits.shape[0]
    dl, ps, pb, da, db = _coef_bwd_buffers(T, 16, logits.device, da, db)
    dR = torch.zeros(T, 4, 4, device=logits.device) if dR is None else dR.contiguous()
    _ohc_coef_bwd_kernel[(ps.shape[0],)](logits, scale, bias, da, db, dR, dl, ps, pb, T, alpha, eps,
                                         BT=COEF_BT, num_warps=4)
    return dl, ps.sum(0), pb.sum(0)


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
