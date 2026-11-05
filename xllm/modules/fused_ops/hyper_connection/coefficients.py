"""The iHC and oHC coefficients from a token's controller logits, forward and backward, FP32 throughout.

oHC coefficients from the logits l [T, 16] = (a: 4, b: 4, rotation: 8):
  a = sigmoid(s_a l_a + bias),  b = 2 sigmoid(s_b l_b + bias)
  R = SO(4)(alpha l_r + bias):  q = (1 + z0, z1, z2, z3) / |.|,  r = (1 + z4, ..) / |.|,  R = L(q) Rc(r)
Outputs a, b [T, 4], R [T, 4, 4]; the backward writes u = dL / dz [T, 16], the gradient of the pre-activations.
"""

import torch
import triton
import triton.language as tl

from .common import _check_coef


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
def _ihc_coef_bwd_kernel(l_ptr, scale_ptr, bias_ptr, da_ptr, db_ptr, dl_ptr, pp_ptr, T, BT: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    ot = pid * BT + tl.arange(0, BT)
    psp = pp_ptr + pid * 10  # this program's partial sums: dscale [2], dbias [8]
    _gates_bwd(l_ptr + ot * 8, scale_ptr, bias_ptr, da_ptr, db_ptr, dl_ptr + ot * 8, psp, psp + 2, ot, ot < T)


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
def _ohc_coef_bwd_kernel(l_ptr, scale_ptr, bias_ptr, da_ptr, db_ptr, dr_ptr, dl_ptr, pp_ptr, T, alpha, eps,
                         BT: tl.constexpr):
    # -> dlogits [T, 16]; per program the partial sums of dscale [2] and dbias [16] (the host adds the programs)
    pid = tl.program_id(0).to(tl.int64)
    ot = pid * BT + tl.arange(0, BT)
    tm = ot < T
    lp = l_ptr + ot * 16
    dlp = dl_ptr + ot * 16
    psp = pp_ptr + pid * 18
    pbp = psp + 2
    _gates_bwd(lp, scale_ptr, bias_ptr, da_ptr, db_ptr, dlp, psp, pbp, ot, tm)

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
    partial sums pp [P, 2 + NL] = (dscale, dbias), da and db [T, 4]."""
    P = triton.cdiv(T, COEF_BT)
    da = torch.zeros(T, 4, device=device) if da is None else da.contiguous()
    db = torch.zeros(T, 4, device=device) if db is None else db.contiguous()
    return (torch.empty(T, NL, device=device, dtype=torch.float32),
            torch.empty(P, 2 + NL, device=device, dtype=torch.float32), da, db)


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
    dl, pp, da, db = _coef_bwd_buffers(T, 8, logits.device, da, db)
    _ihc_coef_bwd_kernel[(pp.shape[0],)](logits, scale, bias, da, db, dl, pp, T, BT=COEF_BT, num_warps=4)
    sums = pp.sum(0)
    return dl, sums[:2], sums[2:]


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
    dl, pp, da, db = _coef_bwd_buffers(T, 16, logits.device, da, db)
    dR = torch.zeros(T, 4, 4, device=logits.device) if dR is None else dR.contiguous()
    _ohc_coef_bwd_kernel[(pp.shape[0],)](logits, scale, bias, da, db, dR, dl, pp, T, alpha, eps,
                                         BT=COEF_BT, num_warps=4)
    sums = pp.sum(0)
    return dl, sums[:2], sums[2:]
