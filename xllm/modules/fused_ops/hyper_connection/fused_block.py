"""The fused block's backward through one sublayer, X' = R X + b y with h = read(X, a) feeding the branch, in two
passes over the streams instead of the inverse write + read + write_bwd + read_bwd + projection dx + two accumulations:

    unwrite_bwd    X' (hi + lo), G = dL/dX'  ->  X (hi + lo, in place), h, dy, db, dR      (then the branch backward)
    rewrite_bwd    the input below, G        ->  X (the write below again), h, dy, db, dR   (when that input is at hand)
    read_write_bwd X, G                      ->  h, dy, db, dR
    read_da        X, dh = dL/dh             ->  da                                          (then the controller backward)
    read_proj_bwd  X, G, dh, dP, W^T, c      ->  dX = R^T G + a dh + dP W^T - c X              (in place over G when allowed)

G is read twice (once per pass) so that no transient R^T G has to be written and read back. Both passes are one program
per tile (a token x CH channels; BT tokens x CH channels) rather than per token: no accumulator lives across a loop, so
the tiles can be as large as the memory system wants (the per-token loop with its 20 accumulators ran ~10% slower).
"""

import torch
import triton
import triton.language as tl

from .common import _check_streams, _ptr, _tile, _ld, _ld2, _st, _st2, _ldc, _coefs, _mix


INVERT, REWRITE, PLAIN = 0, 1, 2  # where `_stream_bwd_kernel` gets the streams X of the write from


@triton.jit
def _stream_bwd_kernel(xs_ptr, xslo_ptr, r0_ptr, b0_ptr, y0_ptr, r_ptr, b_ptr, y_ptr, a_ptr, go_ptr, x_ptr, xlo_ptr, h_ptr,
                       db_ptr, dr_ptr, dy_ptr, N, D, NT, CH: tl.constexpr, HAS_R: tl.constexpr, SRC: tl.constexpr,
                       HAS_R0: tl.constexpr):
    # One token x one channel tile: the streams X of the write X' = R X + b y, then in the same pass
    #   h = sum_m a_m X_m                     the branch input                                    (`_read_fwd_kernel`)
    #   dy = sum_n b_n G_n,  db_n = sum_d G_n y,  dR[n, m] = sum_d G_n X_m                    (`_write_bwd_kernel` without dX)
    # db and dR are the tile's partial sums, db[t, j, 0..3] and dR[t, j, 0..15], added over the tiles j by the host.
    # X comes from (SRC), every tile loaded before it is stored:
    #   INVERT   X' = hi + lo at xs_ptr: X_m = sum_n R[n, m] (X'_n - b_n y), stored as hi + lo (in place when x_ptr is xs_ptr)
    #   REWRITE  the input of the write below at xs_ptr: X = R0 X_s + b0 y0, the forward write again, stored as hi
    #   PLAIN    X itself at xs_ptr, nothing stored
    t = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    (b0, b1, b2, b3, r00, r01, r02, r03, r10, r11, r12, r13, r20, r21, r22, r23,
     r30, r31, r32, r33) = _coefs(r_ptr, b_ptr, t, N, HAS_R)
    a0 = _ldc(a_ptr, t, 0, N)
    a1 = _ldc(a_ptr, t, 1, N)
    a2 = _ldc(a_ptr, t, 2, N)
    a3 = _ldc(a_ptr, t, 3, N)
    d = j * CH + tl.arange(0, CH)
    m = d < D
    y = tl.load(y_ptr + t * D + d, mask=m, other=0.0).to(tl.float32)
    if SRC == 0:
        z0 = _ld2(xs_ptr, xslo_ptr, t, 0, N, D, d, m, True) - b0 * y
        z1 = _ld2(xs_ptr, xslo_ptr, t, 1, N, D, d, m, True) - b1 * y
        z2 = _ld2(xs_ptr, xslo_ptr, t, 2, N, D, d, m, True) - b2 * y
        z3 = _ld2(xs_ptr, xslo_ptr, t, 3, N, D, d, m, True) - b3 * y
        if HAS_R:
            x0 = _st2(x_ptr, xlo_ptr, t, 0, N, D, d, m, _mix(r00, r10, r20, r30, z0, z1, z2, z3))
            x1 = _st2(x_ptr, xlo_ptr, t, 1, N, D, d, m, _mix(r01, r11, r21, r31, z0, z1, z2, z3))
            x2 = _st2(x_ptr, xlo_ptr, t, 2, N, D, d, m, _mix(r02, r12, r22, r32, z0, z1, z2, z3))
            x3 = _st2(x_ptr, xlo_ptr, t, 3, N, D, d, m, _mix(r03, r13, r23, r33, z0, z1, z2, z3))
        else:
            x0 = _st2(x_ptr, xlo_ptr, t, 0, N, D, d, m, z0)
            x1 = _st2(x_ptr, xlo_ptr, t, 1, N, D, d, m, z1)
            x2 = _st2(x_ptr, xlo_ptr, t, 2, N, D, d, m, z2)
            x3 = _st2(x_ptr, xlo_ptr, t, 3, N, D, d, m, z3)
    elif SRC == 1:
        (c0, c1, c2, c3, q00, q01, q02, q03, q10, q11, q12, q13, q20, q21, q22, q23,
         q30, q31, q32, q33) = _coefs(r0_ptr, b0_ptr, t, N, HAS_R0)
        y0 = tl.load(y0_ptr + t * D + d, mask=m, other=0.0).to(tl.float32)
        s0 = _ld(xs_ptr, t, 0, N, D, d, m)
        s1 = _ld(xs_ptr, t, 1, N, D, d, m)
        s2 = _ld(xs_ptr, t, 2, N, D, d, m)
        s3 = _ld(xs_ptr, t, 3, N, D, d, m)
        if HAS_R0:
            x0 = _st(x_ptr, t, 0, N, D, d, m, _mix(q00, q01, q02, q03, s0, s1, s2, s3) + c0 * y0)
            x1 = _st(x_ptr, t, 1, N, D, d, m, _mix(q10, q11, q12, q13, s0, s1, s2, s3) + c1 * y0)
            x2 = _st(x_ptr, t, 2, N, D, d, m, _mix(q20, q21, q22, q23, s0, s1, s2, s3) + c2 * y0)
            x3 = _st(x_ptr, t, 3, N, D, d, m, _mix(q30, q31, q32, q33, s0, s1, s2, s3) + c3 * y0)
        else:
            x0 = _st(x_ptr, t, 0, N, D, d, m, s0 + c0 * y0)
            x1 = _st(x_ptr, t, 1, N, D, d, m, s1 + c1 * y0)
            x2 = _st(x_ptr, t, 2, N, D, d, m, s2 + c2 * y0)
            x3 = _st(x_ptr, t, 3, N, D, d, m, s3 + c3 * y0)
    else:
        x0 = _ld(xs_ptr, t, 0, N, D, d, m)
        x1 = _ld(xs_ptr, t, 1, N, D, d, m)
        x2 = _ld(xs_ptr, t, 2, N, D, d, m)
        x3 = _ld(xs_ptr, t, 3, N, D, d, m)
    tl.store(h_ptr + t * D + d, (a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3).to(h_ptr.dtype.element_ty), mask=m)
    g0 = _ld(go_ptr, t, 0, N, D, d, m)
    g1 = _ld(go_ptr, t, 1, N, D, d, m)
    g2 = _ld(go_ptr, t, 2, N, D, d, m)
    g3 = _ld(go_ptr, t, 3, N, D, d, m)
    tl.store(dy_ptr + t * D + d, (b0 * g0 + b1 * g1 + b2 * g2 + b3 * g3).to(dy_ptr.dtype.element_ty), mask=m)
    pp = db_ptr + (t * NT + j) * 4
    tl.store(pp + 0, tl.sum(g0 * y, axis=0))
    tl.store(pp + 1, tl.sum(g1 * y, axis=0))
    tl.store(pp + 2, tl.sum(g2 * y, axis=0))
    tl.store(pp + 3, tl.sum(g3 * y, axis=0))
    if HAS_R:
        pp = dr_ptr + (t * NT + j) * 16
        tl.store(pp + 0, tl.sum(g0 * x0, axis=0))
        tl.store(pp + 1, tl.sum(g0 * x1, axis=0))
        tl.store(pp + 2, tl.sum(g0 * x2, axis=0))
        tl.store(pp + 3, tl.sum(g0 * x3, axis=0))
        tl.store(pp + 4, tl.sum(g1 * x0, axis=0))
        tl.store(pp + 5, tl.sum(g1 * x1, axis=0))
        tl.store(pp + 6, tl.sum(g1 * x2, axis=0))
        tl.store(pp + 7, tl.sum(g1 * x3, axis=0))
        tl.store(pp + 8, tl.sum(g2 * x0, axis=0))
        tl.store(pp + 9, tl.sum(g2 * x1, axis=0))
        tl.store(pp + 10, tl.sum(g2 * x2, axis=0))
        tl.store(pp + 11, tl.sum(g2 * x3, axis=0))
        tl.store(pp + 12, tl.sum(g3 * x0, axis=0))
        tl.store(pp + 13, tl.sum(g3 * x1, axis=0))
        tl.store(pp + 14, tl.sum(g3 * x2, axis=0))
        tl.store(pp + 15, tl.sum(g3 * x3, axis=0))


def _stream_bwd(src, xs, xslo, R0, b0, y0, R, b, y, a, go, x, xlo):
    """`_stream_bwd_kernel` on contiguous streams xs [T, N, D] -> h [T, D], dR [T, N, N] or None, db [T, N], dy [T, D];
    x / xlo receive X when the source stores it."""
    _check_streams(xs)
    T, N, D = xs.shape
    has_r = R is not None
    h = torch.empty(T, D, device=xs.device, dtype=xs.dtype)
    dy = torch.empty_like(y)
    CH = _tile(D, 512)
    NT = triton.cdiv(D, CH)
    part_db = torch.empty(T, NT, 4, device=xs.device, dtype=torch.float32)  # the tiles' partial sums
    part_dR = torch.empty(T, NT, 16, device=xs.device, dtype=torch.float32) if has_r else None
    _stream_bwd_kernel[(T, NT)](xs, _ptr(xslo, xs), _ptr(R0, b), _ptr(b0, b), _ptr(y0, y), _ptr(R, b), b, y, a,
                                go.contiguous(), _ptr(x, xs), _ptr(xlo, xs), h, part_db, _ptr(part_dR, part_db), dy,
                                N, D, NT, CH=CH, HAS_R=has_r, SRC=src, HAS_R0=R0 is not None, num_warps=1)
    db = part_db.sum(1)[:, :N].contiguous()  # over the channel tiles, in a fixed order
    dR = part_dR.sum(1).view(T, 4, 4)[:, :N, :N].contiguous() if has_r else None
    return h, dR, db, dy


def unwrite_bwd(hi_next, lo_next, R, b, y, a, go, inplace=False):
    """`unwrite` of X' = (hi, lo) [T, N, D], the recomputed read h = read(X, a) of the recovered streams and `write_bwd`
    without its dX, for G = go [T, N, D] -> X as (hi, lo), h [T, D], dR [T, N, N] or None, db [T, N], dy [T, D].
    `inplace` overwrites hi_next and lo_next with X. The transport share R^T G of dX is formed by `read_proj_bwd`."""
    assert lo_next is not None, "unwrite_bwd takes X' as (hi, lo)"
    hi, lo = (hi_next, lo_next) if inplace else (torch.empty_like(hi_next), torch.empty_like(hi_next))
    h, dR, db, dy = _stream_bwd(INVERT, hi_next, lo_next, None, None, None, R, b, y, a, go, hi, lo)
    return hi, lo, h, dR, db, dy


def rewrite_bwd(x, R0, b0, y0, a, R, b, y, go):
    """The write below again, X = R0 x + b0 y0 (`write_fwd` on hi alone), then the recomputed read h = read(X, a) and the
    `write_bwd` of X' = R X + b y without its dX, for G = go -> X, h, dR, db, dy: the backward's first pass over the
    streams when the layer's input x is at hand (kept, recomputed, or the first layer's embedding)."""
    x_next = torch.empty_like(x)
    h, dR, db, dy = _stream_bwd(REWRITE, x, None, R0, b0, y0, R, b, y, a, go, x_next, None)
    return x_next, h, dR, db, dy


def read_write_bwd(x, a, R, b, y, go):
    """The recomputed read h = read(x, a) and the `write_bwd` of X' = R x + b y without its dX, for G = go -> h, dR, db, dy,
    in one pass over x."""
    return _stream_bwd(PLAIN, x, None, None, None, None, R, b, y, a, go, None, None)


@triton.jit
def _ldt(ptr, ot, n, N, tm):
    # one coefficient of BT tokens: [BT, 1], streams n >= N masked off
    return tl.load(ptr + ot * N + n, mask=tm & (n < N), other=0.0)[:, None]


@triton.jit
def _ldrt(ptr, ot, n, m, N, tm):
    return tl.load(ptr + (ot * N + n) * N + m, mask=tm & (n < N) & (m < N), other=0.0)[:, None]


@triton.jit
def _ldg(ptr, ot, n, N, D, d, m2):
    # a [BT, CH] tile of stream n
    return tl.load(ptr + (ot * N + n)[:, None] * D + d[None, :], mask=m2 & (n < N), other=0.0).to(tl.float32)


@triton.jit
def _stg(ptr, ot, m, N, D, d, m2, v):
    tl.store(ptr + (ot * N + m)[:, None] * D + d[None, :], v.to(ptr.dtype.element_ty), mask=m2 & (m < N))


@triton.jit
def _proj_acc(x_ptr, wt_ptr, dp, ol, lm, ot, m, N, D, K, d, dm, m2, am, dh, c):
    # a_m dh + dP W^T[m D + d] - c X_m for one stream m of a [BT, CH] tile (dP W^T on the tensor cores)
    k = m * D + d
    wt = tl.load(wt_ptr + ol[:, None] * K + k[None, :], mask=lm[:, None] & (dm & (m < N))[None, :], other=0.0)  # [LP, CH]
    x = _ldg(x_ptr, ot, m, N, D, d, m2)
    return am * dh + tl.dot(dp, wt) - c * x


@triton.jit
def _read_proj_bwd_kernel(x_ptr, r_ptr, a_ptr, go_ptr, dh_ptr, dp_ptr, wt_ptr, c_ptr, dx_ptr, T, N, D, NL,
                          BT: tl.constexpr, CH: tl.constexpr, LP: tl.constexpr, HAS_R: tl.constexpr):
    # dX = R^T G + a dh + dP W^T - c X on a tile of BT tokens x CH channels, all N streams: the write's transport share
    # (G = dL/dX'), the read's share (dh = dL/dh) and the controller's share (dP = dlogits rstd [T, NL], W^T [NL, N * D],
    # c as in `logits_bwd`). With a transport every G_n feeds every dX_m: the four accumulators are formed first and the
    # G tiles then streamed one at a time (one live instead of four). With the identity, one stream at a time, nothing
    # held across streams. In place when dx_ptr is go_ptr: a dX row is stored only after the G rows it depends on were
    # loaded (all four with a transport, its own without).
    ot = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    d = tl.program_id(1) * CH + tl.arange(0, CH)
    tm = ot < T
    dm = d < D
    m2 = tm[:, None] & dm[None, :]
    ol = tl.arange(0, LP)
    lm = ol < NL
    K = N * D
    dp = tl.load(dp_ptr + ot[:, None] * NL + ol[None, :], mask=tm[:, None] & lm[None, :], other=0.0)  # [BT, LP]
    c = tl.load(c_ptr + ot, mask=tm, other=0.0)[:, None]
    dh = tl.load(dh_ptr + ot[:, None] * D + d[None, :], mask=m2, other=0.0).to(tl.float32)
    if HAS_R:
        o0 = _proj_acc(x_ptr, wt_ptr, dp, ol, lm, ot, 0, N, D, K, d, dm, m2, _ldt(a_ptr, ot, 0, N, tm), dh, c)
        o1 = _proj_acc(x_ptr, wt_ptr, dp, ol, lm, ot, 1, N, D, K, d, dm, m2, _ldt(a_ptr, ot, 1, N, tm), dh, c)
        o2 = _proj_acc(x_ptr, wt_ptr, dp, ol, lm, ot, 2, N, D, K, d, dm, m2, _ldt(a_ptr, ot, 2, N, tm), dh, c)
        o3 = _proj_acc(x_ptr, wt_ptr, dp, ol, lm, ot, 3, N, D, K, d, dm, m2, _ldt(a_ptr, ot, 3, N, tm), dh, c)
        g = _ldg(go_ptr, ot, 0, N, D, d, m2)
        o0 += _ldrt(r_ptr, ot, 0, 0, N, tm) * g
        o1 += _ldrt(r_ptr, ot, 0, 1, N, tm) * g
        o2 += _ldrt(r_ptr, ot, 0, 2, N, tm) * g
        o3 += _ldrt(r_ptr, ot, 0, 3, N, tm) * g
        g = _ldg(go_ptr, ot, 1, N, D, d, m2)
        o0 += _ldrt(r_ptr, ot, 1, 0, N, tm) * g
        o1 += _ldrt(r_ptr, ot, 1, 1, N, tm) * g
        o2 += _ldrt(r_ptr, ot, 1, 2, N, tm) * g
        o3 += _ldrt(r_ptr, ot, 1, 3, N, tm) * g
        g = _ldg(go_ptr, ot, 2, N, D, d, m2)
        o0 += _ldrt(r_ptr, ot, 2, 0, N, tm) * g
        o1 += _ldrt(r_ptr, ot, 2, 1, N, tm) * g
        o2 += _ldrt(r_ptr, ot, 2, 2, N, tm) * g
        o3 += _ldrt(r_ptr, ot, 2, 3, N, tm) * g
        g = _ldg(go_ptr, ot, 3, N, D, d, m2)
        o0 += _ldrt(r_ptr, ot, 3, 0, N, tm) * g
        o1 += _ldrt(r_ptr, ot, 3, 1, N, tm) * g
        o2 += _ldrt(r_ptr, ot, 3, 2, N, tm) * g
        o3 += _ldrt(r_ptr, ot, 3, 3, N, tm) * g
        _stg(dx_ptr, ot, 0, N, D, d, m2, o0)
        _stg(dx_ptr, ot, 1, N, D, d, m2, o1)
        _stg(dx_ptr, ot, 2, N, D, d, m2, o2)
        _stg(dx_ptr, ot, 3, N, D, d, m2, o3)
    else:
        for m in tl.static_range(4):
            o = _proj_acc(x_ptr, wt_ptr, dp, ol, lm, ot, m, N, D, K, d, dm, m2, _ldt(a_ptr, ot, m, N, tm), dh, c)
            _stg(dx_ptr, ot, m, N, D, d, m2, o + _ldg(go_ptr, ot, m, N, D, d, m2))


def read_proj_bwd(x, R, a, go, dh, dp, wt, c, out=None):
    """One sublayer's stream gradient dX = R^T G + a dh + dP W^T - c X [T, N, D] in x's dtype: x the sublayer's input
    streams (hi), R [T, N, N] FP32 or None (identity), a [T, N] FP32, G = go [T, N, D] the gradient of its output streams,
    dh [T, D] the gradient of its read, (dP, c) from `proj_bwd_terms` and W^T [NL, N * D] contiguous in x's dtype.
    `out` receives dX; it may be `go` itself."""
    _check_streams(x)
    T, N, D = x.shape
    NL = wt.shape[0]
    dx = torch.empty_like(x) if out is None else out
    has_r = R is not None
    BT, CH, warps = 16, min(256, triton.next_power_of_2(D)), 4  # the best of an H200 sweep at D = 1536
    _read_proj_bwd_kernel[(triton.cdiv(T, BT), triton.cdiv(D, CH))](
        x, _ptr(R, a), a, go, dh.contiguous(), dp, wt, c, dx, T, N, D, NL,
        BT=BT, CH=CH, LP=max(16, triton.next_power_of_2(NL)), HAS_R=has_r, num_warps=warps)
    return dx
