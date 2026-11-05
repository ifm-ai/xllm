"""The stream read h = sum_n a_n X_n and write X' = R X + b y, one kernel each way: the raw launchers (`read_fwd`,
`read_bwd`, `read_da`, `write_fwd`, `write_bwd`) and the module path's autograd Functions (`hc_read`, `hc_write`).
`write_fwd(split=True)` stores X' as hi + lo (hi = round(X'), lo = round(X' - hi): the FP32 write to ~2^-17), the form
the fused block keeps the streams in between layers.
"""

import torch
import triton
import triton.language as tl

from .common import _check_streams, _ptr, _tile, _ld, _st, _ld2, _st2, _ldc, _stc, _ldr, _str, _coefs, _mix


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
def _read_bwd_kernel(x_ptr, a_ptr, dh_ptr, dx_ptr, da_ptr, N, D, CH: tl.constexpr, HAS_DX: tl.constexpr):
    # da_n = sum_d x_n dh and, with HAS_DX, dx_n = a_n dh (the fused block adds a dh to dX in `_read_proj_bwd_kernel`)
    t = tl.program_id(0).to(tl.int64)
    if HAS_DX:
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
        if HAS_DX:
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
    _read_bwd_kernel[(T,)](x, a, dh.contiguous(), dx, da, N, D, CH=_tile(D), HAS_DX=True, num_warps=2)
    return dx, da


def read_da(x, dh):
    """The read backward's coefficient gradient alone: x [T, N, D], dh [T, D] -> da [T, N] FP32. The fused block needs
    da for the controller backward before it can form dX (`read_proj_bwd`)."""
    _check_streams(x)
    T, N, D = x.shape
    da = torch.empty(T, N, device=x.device, dtype=torch.float32)
    _read_bwd_kernel[(T,)](x, x, dh.contiguous(), x, da, N, D, CH=_tile(D), HAS_DX=False, num_warps=4)  # a, dx unused
    return da


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
def _write_fwd_kernel(x_ptr, xlo_ptr, r_ptr, b_ptr, y_ptr, out_ptr, outlo_ptr, rstd_ptr, N, D, eps, CH: tl.constexpr,
                      HAS_R: tl.constexpr, HAS_LO: tl.constexpr, SPLIT: tl.constexpr, RSTD: tl.constexpr):
    # X' = R X + b y. HAS_LO: X is kept as hi + lo; SPLIT: X' is stored as hi + lo (else rounded once into out_ptr), in
    # place when out_ptr is x_ptr (each tile is loaded before it is stored).
    # RSTD: rstd[t] = rsqrt(mean(hi'^2) + eps) over the token's N * D channels, what `_rstd_kernel` computes for the
    # next controller from the stored hi', here from the rounded values in registers.
    t = tl.program_id(0).to(tl.int64)
    (b0, b1, b2, b3, r00, r01, r02, r03, r10, r11, r12, r13, r20, r21, r22, r23,
     r30, r31, r32, r33) = _coefs(r_ptr, b_ptr, t, N, HAS_R)
    if RSTD:
        ss = tl.zeros([CH], dtype=tl.float32)
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
            h0 = _st2(out_ptr, outlo_ptr, t, 0, N, D, d, m, o0)
            h1 = _st2(out_ptr, outlo_ptr, t, 1, N, D, d, m, o1)
            h2 = _st2(out_ptr, outlo_ptr, t, 2, N, D, d, m, o2)
            h3 = _st2(out_ptr, outlo_ptr, t, 3, N, D, d, m, o3)
        else:
            h0 = _st(out_ptr, t, 0, N, D, d, m, o0)
            h1 = _st(out_ptr, t, 1, N, D, d, m, o1)
            h2 = _st(out_ptr, t, 2, N, D, d, m, o2)
            h3 = _st(out_ptr, t, 3, N, D, d, m, o3)
        if RSTD:
            ss += h0 * h0 + h1 * h1 + h2 * h2 + h3 * h3  # masked streams and channels are zero
    if RSTD:
        tl.store(rstd_ptr + t, tl.rsqrt(tl.sum(ss, axis=0) / (N * D) + eps))


@triton.jit
def _write_bwd_kernel(x_ptr, r_ptr, b_ptr, y_ptr, go_ptr, dx_ptr, dr_ptr, db_ptr, dy_ptr, N, D,
                      CH: tl.constexpr, HAS_R: tl.constexpr, HAS_DX: tl.constexpr):
    # dy = sum_n b_n g_n, db_n = sum_d g_n y and, with a transport, dR[n, m] = sum_d g_n x_m and, HAS_DX,
    # dX_m = sum_n R[n, m] g_n
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
            if HAS_DX:
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


def write_fwd(hi, R, b, y, lo=None, split=False, rstd_eps=None, inplace=False):
    """X' = R X + b y: hi [T, N, D] (X, or its hi part when `lo` is given: X = hi + lo), R [T, N, N] FP32 or None
    (identity), b [T, N] FP32, y [T, D] in hi's dtype, all contiguous -> (hi', lo', rstd).
    `split=False`: X' rounded once into hi', lo' None. `split=True`: hi' = round(X'), the streams the next kernels read,
    and lo' = round(X' - hi'), so that hi' + lo' holds X' to ~2^-17 and the write can be inverted without the
    single-rounding drift. `rstd_eps`: also rstd [T] FP32 of hi', rsqrt(mean(hi'^2) + eps) over the N * D channels,
    the normalization of the controller that reads X' next (`logits_fwd(..., rstd=rstd)`), without a pass over hi';
    else None. `inplace` overwrites hi (and lo, for a split write) with X'."""
    _check_streams(hi)
    T, N, D = hi.shape
    if inplace:
        assert split == (lo is not None), "write_fwd in place stores X' over X: (hi, lo) for a split write, hi alone otherwise"
        out, outlo = hi, lo
    else:
        out = torch.empty_like(hi)
        outlo = torch.empty_like(hi) if split else None
    rstd = torch.empty(T, device=hi.device, dtype=torch.float32) if rstd_eps is not None else None
    _write_fwd_kernel[(T,)](hi, _ptr(lo, hi), _ptr(R, b), b, y, out, _ptr(outlo, out), _ptr(rstd, out), N, D,
                            0.0 if rstd_eps is None else rstd_eps, CH=_tile(D), HAS_R=R is not None,
                            HAS_LO=lo is not None, SPLIT=split, RSTD=rstd is not None, num_warps=4)
    return out, outlo, rstd


def write_bwd(x, R, b, y, go, need_dx=True):
    """-> dx, dR, db [T, N], dy [T, D]. For the identity transport dX = dX' and dR does not exist: dx and dR are None.
    `need_dx=False` leaves dx out for a transport too (the fused block forms its R^T G in `read_proj_bwd`)."""
    _check_streams(x)
    T, N, D = x.shape
    has_r = R is not None
    # the transport backward keeps 20 accumulator tiles live: small tiles, 2 warps (the best of an H200 sweep at D = 1536)
    tile, warps = ((256, 2) if D > 512 else (1024, 2)) if has_r else (1024, 4)
    go = go.contiguous()
    db = torch.empty_like(b)
    dy = torch.empty_like(y)
    dx = torch.empty_like(x) if has_r and need_dx else None
    dR = torch.empty_like(R) if has_r else None  # identity transport: dX = dX', nothing to write
    _write_bwd_kernel[(T,)](x, _ptr(R, b), b, y, go, _ptr(dx, go), _ptr(dR, b), db, dy, N, D,
                            CH=_tile(D, tile), HAS_R=has_r, HAS_DX=dx is not None, num_warps=warps)
    return dx, dR, db, dy


class _Write(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, R, b, y):
        ctx.save_for_backward(x, R, b, y)
        return write_fwd(x, R, b, y)[0]

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
