"""Group RMS / layer norm forward (CuTe DSL, any arch): one warp per (row, group), vector loads, shuffle
reductions, several warps per block. Drop-in for xllm_extension group_rms_norm_fwd(_affine) /
group_layer_norm_fwd_affine, whose one-warp blocks with strided scalar loads run ~1.8x over the bandwidth floor.

  rms:   rstd = rsqrt(mean(x^2) + eps),  y = x rstd (w)
  layer: mean, rstd = rsqrt(var + eps),  y = (x - mean) rstd w + b
x [rows, G * CPG] contiguous (bf16 / fp16), CPG a multiple of 128 up to 2048; y like x, mean / rstd [rows, G] fp32.
"""
import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float16, Float32, Int32, Int64

TORCH2CUTE = {torch.bfloat16: BFloat16, torch.float16: Float16}


@cute.jit
def _rotate_pairs(y: cute.Tensor, pF: cute.Pointer, pos, col, R: cutlass.Constexpr, VW: cutlass.Constexpr,
                  conj: cutlass.Constexpr):
    """Rotate the VW / 2 channel pairs (col + 2p, col + 2p + 1) of y in place by freqs[pos] (fp32 (cos, sin) pairs,
    R per position), conjugated for the backward."""
    f = cute.make_tensor((pF + (Int64(pos) * R + col)).align(4 * VW), cute.make_layout(VW)).load()
    for p in cutlass.range_constexpr(VW // 2):
        a, c = y[2 * p], y[2 * p + 1]
        cs, sn = f[2 * p], f[2 * p + 1]
        if cutlass.const_expr(conj):
            sn = -sn
        y[2 * p] = a * cs - c * sn
        y[2 * p + 1] = a * sn + c * cs


class GroupNormFwd:
    def __init__(self, dtype, CPG: int, G: int, layer: bool, affine: bool, has_add: bool = False, rope_R: int = 0,
                 warps: int = 8):
        assert CPG % 128 == 0 and CPG <= 2048 and (affine or not layer)
        self.dtype, self.CPG, self.G, self.layer, self.affine, self.warps = dtype, CPG, G, layer, affine, warps
        self.has_add = has_add  # normalize x + x2 (the sum is not written)
        self.rope_R = rope_R  # > 0: rotate the first rope_R channels of every group (a head) by freqs[position]
        assert rope_R % 8 == 0 and rope_R <= CPG
        self.E = CPG // 32  # elements per lane
        self.VW = min(8, self.E)  # vector width
        self.NV = self.E // self.VW  # vectors per lane

    @cute.jit
    def __call__(self, tX: cute.Tensor, tX2: cute.Tensor, tW: cute.Tensor, tB: cute.Tensor, tY: cute.Tensor,
                 tMean: cute.Tensor, tRstd: cute.Tensor, tF: cute.Tensor, rows: Int32, L: Int32, eps: Float32,
                 stream: cuda.CUstream):
        n = rows * self.G
        grid = (n + self.warps - 1) // self.warps
        self.kernel(tX.iterator, tX2.iterator, tW.iterator, tB.iterator, tY.iterator, tMean.iterator, tRstd.iterator,
                    tF.iterator, n, L, eps).launch(grid=[grid, 1, 1], block=[32 * self.warps, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pX: cute.Pointer, pX2: cute.Pointer, pW: cute.Pointer, pB: cute.Pointer, pY: cute.Pointer,
               pMean: cute.Pointer, pRstd: cute.Pointer, pF: cute.Pointer, n: Int32, L: Int32, eps: Float32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        item = bidx * self.warps + tidx // 32  # (row, group) index: row * G + group
        if item < n:
            g = item % self.G
            base = Int64(item) * self.CPG
            al = 2 * self.VW
            x = cute.make_rmem_tensor(cute.make_layout(self.E), Float32)
            for j in cutlass.range_constexpr(self.NV):
                off = base + j * 32 * self.VW + lane * self.VW
                v = cute.make_tensor((pX + off).align(al), cute.make_layout(self.VW)).load().to(Float32)
                if cutlass.const_expr(self.has_add):
                    v = v + cute.make_tensor((pX2 + off).align(al), cute.make_layout(self.VW)).load().to(Float32)
                for e in cutlass.range_constexpr(self.VW):
                    x[j * self.VW + e] = v[e]
            mean = Float32(0.0)
            if cutlass.const_expr(self.layer):
                s = Float32(0.0)
                for i in cutlass.range_constexpr(self.E):
                    s = s + x[i]
                for o in cutlass.range_constexpr(5):
                    s = s + cute.arch.shuffle_sync_bfly(s, offset=1 << o)
                mean = s * (1.0 / self.CPG)
            q = Float32(0.0)
            for i in cutlass.range_constexpr(self.E):
                d = x[i] - mean
                q = q + d * d
            for o in cutlass.range_constexpr(5):
                q = q + cute.arch.shuffle_sync_bfly(q, offset=1 << o)
            rstd = cute.math.rsqrt(q * (1.0 / self.CPG) + eps)
            if lane == 0:
                cute.make_tensor(pRstd + item, cute.make_layout(1))[0] = rstd
                if cutlass.const_expr(self.layer):
                    cute.make_tensor(pMean + item, cute.make_layout(1))[0] = mean
            for j in cutlass.range_constexpr(self.NV):
                col = j * 32 * self.VW + lane * self.VW
                y = cute.make_rmem_tensor(cute.make_layout(self.VW), Float32)
                for e in cutlass.range_constexpr(self.VW):
                    y[e] = (x[j * self.VW + e] - mean) * rstd
                if cutlass.const_expr(self.affine):
                    w = cute.make_tensor((pW + (g * self.CPG + col)).align(al), cute.make_layout(self.VW)).load().to(Float32)
                    for e in cutlass.range_constexpr(self.VW):
                        y[e] = y[e] * w[e]
                    if cutlass.const_expr(self.layer):
                        bb = cute.make_tensor((pB + (g * self.CPG + col)).align(al), cute.make_layout(self.VW)).load().to(Float32)
                        for e in cutlass.range_constexpr(self.VW):
                            y[e] = y[e] + bb[e]
                if cutlass.const_expr(self.rope_R > 0):
                    if col < self.rope_R:
                        _rotate_pairs(y, pF, (item // self.G) % L, col, self.rope_R, self.VW, False)
                cute.make_tensor((pY + (base + col)).align(al), cute.make_layout(self.VW)).store(y.load().to(self.dtype))


_COMPILED = {}


def _fake(dtype, align):
    return cute.runtime.make_fake_compact_tensor(dtype, (cute.sym_int(),), assumed_align=align)


def supported(x: torch.Tensor, num_groups: int) -> bool:
    D = x.shape[-1]
    return (x.is_cuda and x.dtype in TORCH2CUTE and x.is_contiguous() and D % num_groups == 0
            and (D // num_groups) % 128 == 0 and D // num_groups <= 2048 and x.data_ptr() % 16 == 0)


def group_norm_fwd(x: torch.Tensor, num_groups: int, eps: float, weight=None, bias=None, layer: bool = False, x2=None,
                   freqs_cis=None, rope_head_dim: int = 0):
    """Returns (y, rstd) for rms, (y, mean, rstd) for layer norm; mean / rstd [rows, G] fp32. With x2 (same shape),
    normalizes x + x2 (fp32 sum, not written out). With freqs_cis ([L, rope_head_dim / 2] complex64, x [B, L, D] and
    a group per head), y gets the rotary embedding of its first rope_head_dim channels of every group."""
    D = x.shape[-1]
    G = num_groups
    CPG = D // G
    rows = x.numel() // D
    affine = weight is not None
    y = torch.empty_like(x)
    stat_shape = (rows, G)  # as the xllm_extension ops return them
    rstd = torch.empty(stat_shape, dtype=torch.float32, device=x.device)
    mean = torch.empty(stat_shape, dtype=torch.float32, device=x.device) if layer else rstd
    has_add = x2 is not None
    R = rope_head_dim if freqs_cis is not None else 0
    key = (x.dtype, CPG, G, layer, affine, has_add, R)
    if key not in _COMPILED:
        dt = TORCH2CUTE[x.dtype]
        al = 2 * min(8, CPG // 32)
        _COMPILED[key] = cute.compile(
            GroupNormFwd(dt, CPG, G, layer, affine, has_add, R), _fake(dt, al), _fake(dt, al), _fake(dt, al),
            _fake(dt, al), _fake(dt, al), _fake(Float32, 4), _fake(Float32, 4), _fake(Float32, 32), Int32(0), Int32(0),
            Float32(0.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
    w = weight.contiguous() if affine else x
    b = bias.contiguous() if (layer and bias is not None) else x
    x2v = x2.contiguous() if has_add else x
    f = torch.view_as_real(freqs_cis.contiguous()).reshape(-1) if R else rstd.reshape(-1)
    L = x.shape[-2] if R else 1
    _COMPILED[key](x.reshape(-1), x2v.reshape(-1), w.reshape(-1), b.reshape(-1), y.reshape(-1), mean.reshape(-1),
                   rstd.reshape(-1), f, rows, L, eps)
    return (y, mean, rstd) if layer else (y, rstd)


class GroupNormBwd:
    """Group RMS / layer norm backward with the gamma / beta reductions fused (CuTe DSL, any arch).

      xhat = (x - mean) rstd   (rms: mean = 0; mem_eff: xhat = the saved output y, rms without affine only)
      g = dy w (affine) or dy
      dx = rstd (g - (sum g [layer] + xhat sum(g xhat)) / CPG),  dgamma += dy xhat,  dbeta += dy
    One CTA of 8 warps per (group, split of the rows); a warp takes every 8th row of the split (one row-group of
    CPG elements, vector loads), keeps its dgamma / dbeta partials in registers, the CTA adds them up in shared
    memory and writes them per split ([splits, D] fp32, summed by the caller). Replaces GroupRMSNormBwdKernel / GroupLayerNormBwdKernel plus the
    separate Gamma(Beta)BwdLargeKernel passes (2.4-3.8x over the bandwidth floor at the gekko shapes)."""

    def __init__(self, dtype, CPG: int, G: int, layer: bool, affine: bool, mem_eff: bool, rope_R: int = 0,
                 warps: int = 8):
        assert CPG % 128 == 0 and CPG <= 2048 and not (mem_eff and (affine or layer)) and (affine or not layer)
        self.dtype, self.CPG, self.G, self.layer, self.affine, self.mem_eff, self.warps = \
            dtype, CPG, G, layer, affine, mem_eff, warps
        self.rope_R = rope_R  # > 0: dy is the gradient of the rotated output; un-rotate its first rope_R channels
        self.E = CPG // 32
        self.VW = min(8, self.E)
        self.NV = self.E // self.VW

    @cute.jit
    def __call__(self, tDy: cute.Tensor, tX: cute.Tensor, tW: cute.Tensor, tMean: cute.Tensor, tRstd: cute.Tensor,
                 tDx: cute.Tensor, tDw: cute.Tensor, tDb: cute.Tensor, tF: cute.Tensor, rows: Int32, splits: Int32,
                 L: Int32, stream: cuda.CUstream):
        self.kernel(tDy.iterator, tX.iterator, tW.iterator, tMean.iterator, tRstd.iterator, tDx.iterator, tDw.iterator,
                    tDb.iterator, tF.iterator, rows, splits, L).launch(
            grid=[self.G * splits, 1, 1], block=[32 * self.warps, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pDy: cute.Pointer, pX: cute.Pointer, pW: cute.Pointer, pMean: cute.Pointer, pRstd: cute.Pointer,
               pDx: cute.Pointer, pDw: cute.Pointer, pDb: cute.Pointer, pF: cute.Pointer, rows: Int32, splits: Int32,
               L: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = tidx // 32
        g = bidx % self.G
        sp = bidx // self.G
        per = (rows + splits - 1) // splits
        r0 = sp * per
        r1 = cutlass.min(rows, r0 + per)
        al = 2 * self.VW
        D = self.G * self.CPG
        smem = cutlass.utils.SmemAllocator()
        sW = smem.allocate_tensor(Float32, cute.make_layout(self.CPG), byte_alignment=16)
        sB = smem.allocate_tensor(Float32, cute.make_layout(self.CPG), byte_alignment=16)
        for i in cutlass.range(tidx, self.CPG, 32 * self.warps):
            sW[i] = Float32(0.0)
            sB[i] = Float32(0.0)
        accw = cute.make_rmem_tensor(cute.make_layout(self.E), Float32)
        accb = cute.make_rmem_tensor(cute.make_layout(self.E), Float32)
        for i in cutlass.range_constexpr(self.E):
            accw[i] = Float32(0.0)
            accb[i] = Float32(0.0)
        cute.arch.sync_threads()
        for r in cutlass.range(r0 + warp, r1, self.warps):
            base = Int64(r) * D + g * self.CPG
            v = cute.make_tensor(pRstd + (r * self.G + g), cute.make_layout(1))[0]
            mu = Float32(0.0)
            if cutlass.const_expr(self.layer):
                mu = cute.make_tensor(pMean + (r * self.G + g), cute.make_layout(1))[0]
            # x-hat and dy stay in registers; w is re-read per row (L1) and g = dy w recomputed, so that the
            # register footprint is 4 x CPG / 32 floats per lane (with the dgamma / dbeta partials)
            xh = cute.make_rmem_tensor(cute.make_layout(self.E), Float32)
            dyr = cute.make_rmem_tensor(cute.make_layout(self.E), Float32)
            s1 = Float32(0.0)
            s2 = Float32(0.0)
            for j in cutlass.range_constexpr(self.NV):
                off = base + j * 32 * self.VW + lane * self.VW
                dyv = cute.make_tensor((pDy + off).align(al), cute.make_layout(self.VW)).load().to(Float32)
                if cutlass.const_expr(self.rope_R > 0):
                    col = j * 32 * self.VW + lane * self.VW
                    dyt = cute.make_rmem_tensor(cute.make_layout(self.VW), Float32)
                    dyt.store(dyv)
                    if col < self.rope_R:
                        _rotate_pairs(dyt, pF, r % L, col, self.rope_R, self.VW, True)
                    dyv = dyt.load()
                xv = cute.make_tensor((pX + off).align(al), cute.make_layout(self.VW)).load().to(Float32)
                wv = dyv
                if cutlass.const_expr(self.affine):
                    wv = cute.make_tensor((pW + (g * self.CPG + j * 32 * self.VW + lane * self.VW)).align(al),
                                          cute.make_layout(self.VW)).load().to(Float32)
                for e in cutlass.range_constexpr(self.VW):
                    i = j * self.VW + e
                    dyr[i] = dyv[e]
                    xh[i] = xv[e] if cutlass.const_expr(self.mem_eff) else (xv[e] - mu) * v
                    ge = dyv[e] * wv[e] if cutlass.const_expr(self.affine) else dyv[e]
                    s2 = s2 + ge * xh[i]
                    if cutlass.const_expr(self.layer):
                        s1 = s1 + ge
            for o in cutlass.range_constexpr(5):
                s2 = s2 + cute.arch.shuffle_sync_bfly(s2, offset=1 << o)
                if cutlass.const_expr(self.layer):
                    s1 = s1 + cute.arch.shuffle_sync_bfly(s1, offset=1 << o)
            inv = 1.0 / self.CPG
            for j in cutlass.range_constexpr(self.NV):
                off = base + j * 32 * self.VW + lane * self.VW
                o = cute.make_rmem_tensor(cute.make_layout(self.VW), Float32)
                wv2 = o.load()
                if cutlass.const_expr(self.affine):
                    wv2 = cute.make_tensor((pW + (g * self.CPG + j * 32 * self.VW + lane * self.VW)).align(al),
                                           cute.make_layout(self.VW)).load().to(Float32)
                for e in cutlass.range_constexpr(self.VW):
                    i = j * self.VW + e
                    ge = dyr[i] * wv2[e] if cutlass.const_expr(self.affine) else dyr[i]
                    o[e] = v * (ge - (s1 + xh[i] * s2) * inv)
                    if cutlass.const_expr(self.affine):
                        accw[i] = accw[i] + dyr[i] * xh[i]
                        if cutlass.const_expr(self.layer):
                            accb[i] = accb[i] + dyr[i]
                cute.make_tensor((pDx + off).align(al), cute.make_layout(self.VW)).store(o.load().to(self.dtype))
        if cutlass.const_expr(self.affine):
            for j in cutlass.range_constexpr(self.NV):
                for e in cutlass.range_constexpr(self.VW):
                    c = j * 32 * self.VW + lane * self.VW + e
                    cute.arch.atomic_add((sW.iterator + c).llvm_ptr, accw[j * self.VW + e], sem="relaxed", scope="cta")
                    if cutlass.const_expr(self.layer):
                        cute.arch.atomic_add((sB.iterator + c).llvm_ptr, accb[j * self.VW + e], sem="relaxed", scope="cta")
            cute.arch.sync_threads()
            # this split's partial sums (no global atomics: with few groups hundreds of CTAs would add into the
            # same addresses); the caller sums the splits, which also keeps dgamma / dbeta deterministic
            out_off = Int64(sp) * D + g * self.CPG
            gW = cute.make_tensor(pDw + out_off, cute.make_layout(self.CPG))
            gB = cute.make_tensor(pDb + out_off, cute.make_layout(self.CPG))
            for i in cutlass.range(tidx, self.CPG, 32 * self.warps):
                gW[i] = sW[i]
                if cutlass.const_expr(self.layer):
                    gB[i] = sB[i]


_COMPILED_BWD = {}


def group_norm_bwd(dy: torch.Tensor, x_or_y: torch.Tensor, num_groups: int, rstd: torch.Tensor, mean=None, weight=None,
                   layer: bool = False, mem_eff: bool = False, freqs_cis=None, rope_head_dim: int = 0):
    """Returns dx (like x), and dgamma / dbeta (dtype of weight) when affine: (dx,), (dx, dw) or (dx, dw, db).
    With freqs_cis, dy is the gradient of the rotated output of group_norm_fwd(..., freqs_cis, rope_head_dim)."""
    D = x_or_y.shape[-1]
    G = num_groups
    CPG = D // G
    rows = x_or_y.numel() // D
    affine = weight is not None
    dx = torch.empty_like(x_or_y)
    sms = torch.cuda.get_device_properties(dy.device).multi_processor_count
    splits = max(1, min(-(-4 * sms // G), rows // (8 * 32)))  # >= 32 rows per warp: amortize the per-CTA reduction
    accs = torch.empty(2 if layer else 1, splits, D, dtype=torch.float32, device=dy.device) if affine else rstd
    R = rope_head_dim if freqs_cis is not None else 0
    key = (x_or_y.dtype, CPG, G, layer, affine, mem_eff, R)
    if key not in _COMPILED_BWD:
        dt = TORCH2CUTE[x_or_y.dtype]
        al = 2 * min(8, CPG // 32)
        _COMPILED_BWD[key] = cute.compile(
            GroupNormBwd(dt, CPG, G, layer, affine, mem_eff, R), _fake(dt, al), _fake(dt, al), _fake(dt, al),
            _fake(Float32, 4), _fake(Float32, 4), _fake(dt, al), _fake(Float32, 4), _fake(Float32, 4), _fake(Float32, 32),
            Int32(0), Int32(0), Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
    w = weight.contiguous() if affine else dy
    m = mean.contiguous() if layer else rstd
    accw = accs[0] if affine else rstd
    accb = accs[1] if (affine and layer) else accw
    f = torch.view_as_real(freqs_cis.contiguous()).reshape(-1) if R else rstd.reshape(-1)
    L = x_or_y.shape[-2] if R else 1
    _COMPILED_BWD[key](dy.contiguous().reshape(-1), x_or_y.reshape(-1), w.reshape(-1), m.reshape(-1), rstd.reshape(-1),
                       dx.reshape(-1), accw.reshape(-1), accb.reshape(-1), f, rows, splits, L)
    if not affine:
        return (dx,)
    sums = accs.sum(1).to(weight.dtype)
    return (dx, sums[0], sums[1]) if layer else (dx, sums[0])


def supported_bwd(dy: torch.Tensor, x: torch.Tensor, num_groups: int) -> bool:
    return supported(x, num_groups) and dy.shape == x.shape and dy.dtype == x.dtype
