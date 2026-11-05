"""Hopper (sm90) backend of the fp8 GEMMs of xllm/modules/fused_ops/mxfp8.py (``ModelConf.fp8_gemm``).

sm90 has no MX (e8m0, 32-element) scaling; cuBLASLt (>= 12.9) takes blockwise fp32 scales on it instead: e4m3 with one
scale per 1 x 128 group along K or per 128 x 128 tile, fp32 accumulation (the DeepSeek-V3 recipe). Measured on H100
(one GPU, gekko-4B / 7.1B shapes): 1 x 128 times 128 x 128 runs at 1.5 - 1.7x bf16, 1 x 128 times 1 x 128 only at
1.1 - 1.3x. So the weights use 128 x 128 tiles (W and W^T share them), activations and gradients 1 x 128 groups, and
for the weight gradient dW = dY^T X one of the two transposed operands is tiled too (``WGRAD_TILE``). The scales are
powers of two chosen like the MX ones (the group's amax lands in [256, 512), the few values above 448 saturate).

The fp8 wgmma reads both operands K-major only (no MN-major fp8 operands on Hopper), so unlike the Blackwell path the
colwise operand X^T (C, R) is stored transposed: the quantisation kernel loads a 128 x 128 tile of X once and writes
both the rowwise (R, C) and the transposed colwise (C, R) piece (~2.6 TB/s). The contraction dimension of a blockwise GEMM must be
a multiple of 512 (K / 128 scales a multiple of 4: other K are accepted by the shape check but give wrong columns, and
a padded scale stride is rejected), and every dimension of a linear is K of one of its three GEMMs: see ``supported``.

Opt-in ``XLLM_FP8_DEEPGEMM=1``: GEMMs with a 1 x 128 A and a 128 x 128 B operand (and no residual) run through
DeepGEMM's sm90 kernels (deepseek-ai/DeepGEMM, same recipe; results bitwise equal to cuBLASLt on the same operands,
1.1 - 1.35x faster at the gekko shapes on H200). DeepGEMM only takes that operand pairing, so the weight gradient
then tiles X^T instead of dY^T (``WGRAD_TILE`` defaults to "x"). Its kernels are JIT-compiled per shape on first use;
``XLLM_FP8_DEEPGEMM_SMS`` caps the SMs they use (default all).
"""
import os
from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

E4M3 = torch.float8_e4m3fn
BLOCK = 128


# which transposed operand of the weight-gradient GEMM dW = dY^T X takes the 128 x 128 blocks (cuBLASLt on sm90 runs
# the blockwise GEMM at ~1.6x bf16 when one operand has 128 x 128 blocks, ~1.2x when both are 1 x 128): "x" (X^T,
# the layer input) or "dy" (dY^T, the output gradient)
DEEPGEMM = os.environ.get("XLLM_FP8_DEEPGEMM", "0") == "1"
# SMs for DeepGEMM's persistent kernels, 0 = all (H200: 116 / 124 made no difference; H100 measured 120 best next to the
# overlapped reduce-scatter)
DEEPGEMM_SMS = int(os.environ.get("XLLM_FP8_DEEPGEMM_SMS", "0"))
WGRAD_TILE = os.environ.get("XLLM_FP8_SM90_WGRAD_TILE", "x" if DEEPGEMM else "dy")


class Fp8Mat:
    """One operand: ``data`` (rows, cols) e4m3, K-major (contiguous rows, possibly a row slice of a bigger buffer) and
    its fp32 scales: ``tile`` False - one per 1 x 128 group, ``scale`` (rows, cols / 128) M-major (stride (1, ld));
    ``tile`` True - one per 128 x 128 tile, ``scale`` (rows / 128, cols / 128) contiguous (cuBLASLt rejects a padded row stride)."""

    __slots__ = ("data", "scale", "tile")

    def __init__(self, data: torch.Tensor, scale: torch.Tensor, tile: bool):
        self.data = data
        self.scale = scale
        self.tile = tile

    @property
    def rows(self) -> int:
        return self.data.shape[0]

    @property
    def cols(self) -> int:
        return self.data.shape[1]

    def row_slice(self, r0: int, r1: int) -> "Fp8Mat":
        assert r0 % BLOCK == 0 and r1 % BLOCK == 0, (r0, r1)
        sc = self.scale[r0 // BLOCK:r1 // BLOCK] if self.tile else self.scale[r0:r1]
        return Fp8Mat(self.data[r0:r1], sc, self.tile)

    def a_scale(self) -> torch.Tensor:
        return self.scale

    def b_scale(self) -> torch.Tensor:
        """The scales as cuBLASLt takes them for the B operand (K / 128, N[/128])."""
        return self.scale.t() if self.tile else self.scale.t().contiguous()


def supported(*dims: int) -> bool:
    """Every dimension (tokens, in / out features) a multiple of 512."""
    return all(d % (4 * BLOCK) == 0 for d in dims)


class _LazyTransposed:
    """The colwise operand W^T of a 128 x 128-tiled weight: exactly the transpose of the rowwise operand (data and tile
    scales), built from it on first use (the backward's dX GEMM) by an fp8 transpose instead of being quantised in
    the forward and kept until the backward (7.1B: 6.4 GB of the fp8 weights of all layers)."""

    tile = True

    def __init__(self, src: Fp8Mat):
        self.src = src
        self._m = None

    def _get(self) -> Fp8Mat:
        if self._m is None:
            self._m = transpose(self.src)
        return self._m

    @property
    def data(self) -> torch.Tensor:
        return self._get().data

    @property
    def scale(self) -> torch.Tensor:
        return self._get().scale

    @property
    def rows(self) -> int:
        return self.src.cols

    @property
    def cols(self) -> int:
        return self.src.rows

    def row_slice(self, r0: int, r1: int) -> Fp8Mat:
        return self._get().row_slice(r0, r1)

    def a_scale(self) -> torch.Tensor:
        return self._get().a_scale()

    def b_scale(self) -> torch.Tensor:
        return self._get().b_scale()


@triton.jit
def _transpose_kernel(src, ld_s, dst, ld_d, BM: tl.constexpr):
    """One 128 x 128 tile of an (R, C) byte matrix -> its transpose in the (C, R) matrix (stored through transposed
    addresses, see _fp8_kernel)."""
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    rows = rb * BM + tl.arange(0, BM)
    cols = cb * BM + tl.arange(0, BM)
    v = tl.load(src + rows[:, None] * ld_s + cols[None, :])
    tl.store(dst + cols[None, :] * ld_d + rows[:, None], v)


def transpose(m: Fp8Mat) -> Fp8Mat:
    """The transposed operand of a 128 x 128-tiled one (same values: m^T with the transposed tile scales)."""
    assert m.tile and m.data.stride(1) == 1, (m.tile, m.data.stride())
    R, C = m.data.shape
    out = Fp8Mat(torch.empty(C, R, dtype=E4M3, device=m.data.device), m.scale.t().contiguous(), True)
    _transpose_kernel[(R // BLOCK, C // BLOCK)](m.data.view(torch.uint8), m.data.stride(0), out.data.view(torch.uint8),
                                                R, BM=BLOCK, num_warps=8)
    return out


class _LazyTransposedCat(_LazyTransposed):
    """The colwise operand [W1; W2; ..]^T (K, sum N) of separately stored 128 x 128-tiled weights (fp8 all-gather:
    fp8_fsdp.Fp8Weight), built on first use by transposing each into its column block."""

    def __init__(self, srcs):
        self.srcs = list(srcs)
        self._m = None

    def _get(self) -> Fp8Mat:
        if self._m is None:
            K = self.srcs[0].cols
            N = sum(m.rows for m in self.srcs)
            dev = self.srcs[0].data.device
            data = torch.empty(K, N, dtype=E4M3, device=dev)
            off = 0
            for m in self.srcs:
                assert m.tile and m.cols == K and m.data.stride(1) == 1
                _transpose_kernel[(m.rows // BLOCK, K // BLOCK)](m.data.view(torch.uint8), m.data.stride(0),
                                                                  data.view(torch.uint8)[:, off:], N, BM=BLOCK, num_warps=8)
                off += m.rows
            scale = torch.cat([m.scale.t() for m in self.srcs], dim=1).contiguous()
            self._m = Fp8Mat(data, scale, True)
        return self._m

    @property
    def rows(self) -> int:
        return self.srcs[0].cols

    @property
    def cols(self) -> int:
        return sum(m.rows for m in self.srcs)


def lazy_transposed_cat(srcs) -> _LazyTransposedCat:
    return _LazyTransposedCat(srcs)


# the weights' colwise operands are built in the backward from the rowwise ones (_LazyTransposed); 0: quantised with
# the rowwise ones in the forward and kept
LAZY_WEIGHT_T = os.environ.get("XLLM_FP8_SM90_LAZY_WT", "1") == "1"


def _empty(rows: int, cols: int, device, tile: bool) -> Fp8Mat:
    assert rows % BLOCK == 0 and cols % BLOCK == 0, (rows, cols)
    data = torch.empty(rows, cols, dtype=E4M3, device=device)
    if tile:
        scale = torch.empty(rows // BLOCK, cols // BLOCK, dtype=torch.float32, device=device)
    else:
        scale = torch.empty(cols // BLOCK, rows, dtype=torch.float32, device=device).t()
    return Fp8Mat(data, scale, tile)


@triton.jit
def _pow2_scale(amax):
    """amax (fp32) -> (scale 2^(floor(log2(amax)) - 8), its inverse), clamped to the normal range."""
    ebits = (amax.to(tl.int32, bitcast=True) >> 23) & 0xFF
    e = tl.minimum(tl.maximum(ebits - 8, 1), 254)
    scale = (e << 23).to(tl.float32, bitcast=True)
    inv = ((254 - e) << 23).to(tl.float32, bitcast=True)
    return scale, inv


@triton.jit
def _store_tile(x, orb, ocb, qr_ptr, ld_r, sr_ptr, lds_r, qc_ptr, ld_c, sc_ptr, lds_c,
                ROWWISE: tl.constexpr, COLWISE: tl.constexpr, ROW_TILE: tl.constexpr, COL_TILE: tl.constexpr,
                BM: tl.constexpr):
    """x (BM, BM) fp32 = tile (orb, ocb) of an (R, C) matrix -> its rowwise operand (R, C) and colwise operand (C, R).
    Scales of a 1 x 128 operand: column stride lds (rowwise: sr[ocb * lds_r + row]); of a 128 x 128 operand: row
    stride lds (rowwise: sr[orb * lds_r + ocb])."""
    orows = orb * BM + tl.arange(0, BM)
    ocols = ocb * BM + tl.arange(0, BM)
    ax = tl.abs(x)
    if ROW_TILE or COL_TILE:
        ts, tinv = _pow2_scale(tl.max(tl.max(ax, axis=1), axis=0))
    if ROWWISE:
        if ROW_TILE:
            q = (x * tinv).to(tl.float8e4nv)
            tl.store(sr_ptr + orb * lds_r + ocb, ts)
        else:
            s, inv = _pow2_scale(tl.max(ax, axis=1))
            q = (x * inv[:, None]).to(tl.float8e4nv)
            tl.store(sr_ptr + ocb * lds_r + orows, s)
        tl.store(qr_ptr + orows[:, None] * ld_r + ocols[None, :], q)
    if COLWISE:
        # stored through transposed addresses (Triton lays the store out): tl.trans of an fp8 tile stores wrong
        # elements with Triton 3.8 (e.g. when the rowwise output is the same expression)
        if COL_TILE:
            q = (x * tinv).to(tl.float8e4nv)
            tl.store(sc_ptr + ocb * lds_c + orb, ts)
        else:
            s, inv = _pow2_scale(tl.max(ax, axis=0))
            q = (x * inv[None, :]).to(tl.float8e4nv)
            tl.store(sc_ptr + orb * lds_c + ocols, s)
        tl.store(qc_ptr + ocols[None, :] * ld_c + orows[:, None], q)


@triton.jit
def _fp8_kernel(x_ptr, stride, r0, c0,
                qr_ptr, ld_r, sr_ptr, lds_r,
                qc_ptr, ld_c, sc_ptr, lds_c,
                ROWWISE: tl.constexpr, COLWISE: tl.constexpr, ROW_TILE: tl.constexpr, COL_TILE: tl.constexpr,
                BM: tl.constexpr, ROUND_BF16: tl.constexpr = False):
    """Tile (rb, cb) of X (bf16, or fp32 rounded to bf16) -> rowwise operand at (r0 + r, c0 + c) and colwise operand X^T at (c0 + c, r0 + r)."""
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    rows = rb * BM + tl.arange(0, BM)
    cols = cb * BM + tl.arange(0, BM)
    x = tl.load(x_ptr + rows[:, None] * stride + cols[None, :])
    if ROUND_BF16:  # an fp32 input quantised as its bf16 value (FSDP2 master weights: fp8 all-gather)
        x = x.to(tl.bfloat16)
    x = x.to(tl.float32)
    _store_tile(x, r0 // BM + rb, c0 // BM + cb, qr_ptr, ld_r, sr_ptr, lds_r, qc_ptr, ld_c, sc_ptr, lds_c,
                ROWWISE, COLWISE, ROW_TILE, COL_TILE, BM)


@triton.jit
def _swiglu_fwd_kernel(h1_ptr, h3_ptr, stride,
                       qr_ptr, ld_r, sr_ptr, lds_r, qc_ptr, ld_c, sc_ptr, lds_c,
                       COL_TILE: tl.constexpr, BM: tl.constexpr):
    """hidden = silu(h1) * h3 (rounded to bf16 like the unfused kernel) -> its rowwise and colwise operands."""
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    off = (rb * BM + tl.arange(0, BM))[:, None] * stride + (cb * BM + tl.arange(0, BM))[None, :]
    h1 = tl.load(h1_ptr + off).to(tl.float32)
    h3 = tl.load(h3_ptr + off).to(tl.float32)
    h = (h1 * tl.sigmoid(h1) * h3).to(tl.bfloat16).to(tl.float32)
    _store_tile(h, rb, cb, qr_ptr, ld_r, sr_ptr, lds_r, qc_ptr, ld_c, sc_ptr, lds_c, True, True, False, COL_TILE, BM)


@triton.jit
def _swiglu_bwd_kernel(h1_ptr, h3_ptr, g_ptr, stride, g_stride, E,
                       qr_ptr, ld_r, sr_ptr, lds_r, qc_ptr, ld_c, sc_ptr, lds_c,
                       COL_TILE: tl.constexpr, BM: tl.constexpr):
    """d hidden = g -> dh1 = g h3 silu'(h1), dh3 = g silu(h1) (rounded to bf16 like the unfused kernel) -> the
    rowwise and colwise operands of [dh1 | dh3] (dh3 at column E + c)."""
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    rows = (rb * BM + tl.arange(0, BM))[:, None]
    cols = (cb * BM + tl.arange(0, BM))[None, :]
    h1 = tl.load(h1_ptr + rows * stride + cols).to(tl.float32)
    h3 = tl.load(h3_ptr + rows * stride + cols).to(tl.float32)
    g = tl.load(g_ptr + rows * g_stride + cols).to(tl.float32)
    sg = tl.sigmoid(h1)
    d3 = (g * h1 * sg).to(tl.bfloat16).to(tl.float32)
    d1 = (g * h3 * sg * (1.0 + h1 * (1.0 - sg))).to(tl.bfloat16).to(tl.float32)
    _store_tile(d1, rb, cb, qr_ptr, ld_r, sr_ptr, lds_r, qc_ptr, ld_c, sc_ptr, lds_c, True, True, False, COL_TILE, BM)
    _store_tile(d3, rb, E // BM + cb, qr_ptr, ld_r, sr_ptr, lds_r, qc_ptr, ld_c, sc_ptr, lds_c, True, True, False, COL_TILE, BM)


@triton.jit
def _gate_kernel(s_ptr, a_ptr, r_ptr, stride, eps,
                 qr_ptr, ld_r, sr_ptr, lds_r, qc_ptr, ld_c, sc_ptr, lds_c,
                 V: tl.constexpr, SILU: tl.constexpr, COL_TILE: tl.constexpr, BM: tl.constexpr):
    """attn = bf16(bf16(rmsnorm_head(bf16(s + a))) * bf16(gate(r))) for 128 rows x one head of V (128 or 256)
    columns (cute AttnGateFwd's roundings) -> the rowwise and colwise operands of attn (attn is not written)."""
    rb = tl.program_id(0)
    h = tl.program_id(1)
    rows = (rb * BM + tl.arange(0, BM))[:, None]
    ss = tl.zeros((BM,), dtype=tl.float32)
    for c in tl.static_range(V // BM):
        cols = (h * V + c * BM + tl.arange(0, BM))[None, :]
        x = (tl.load(s_ptr + rows * stride + cols).to(tl.float32) + tl.load(a_ptr + rows * stride + cols).to(tl.float32))
        x = x.to(tl.bfloat16).to(tl.float32)
        ss += tl.sum(x * x, axis=1)
    rstd = tl.math.rsqrt(ss * (1.0 / V) + eps)
    for c in tl.static_range(V // BM):
        cols = (h * V + c * BM + tl.arange(0, BM))[None, :]
        x = (tl.load(s_ptr + rows * stride + cols).to(tl.float32) + tl.load(a_ptr + rows * stride + cols).to(tl.float32))
        x = x.to(tl.bfloat16).to(tl.float32)
        n = (x * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
        r = tl.load(r_ptr + rows * stride + cols).to(tl.float32)
        if SILU:
            g = r * tl.sigmoid(r)
        else:  # softplus with beta = ln 2 (cute attn_gate): log2(1 + 2^r), r beyond the threshold 20 / ln 2
            g = tl.where(r > 20.0 / 0.6931471805599453, r, tl.log2(1.0 + tl.exp2(r)))
        y = (n * g.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        _store_tile(y, rb, (h * V) // BM + c, qr_ptr, ld_r, sr_ptr, lds_r, qc_ptr, ld_c, sc_ptr, lds_c,
                    True, True, False, COL_TILE, BM)


def gate_quantize(sca: torch.Tensor, awk: torch.Tensor, rmx: torch.Tensor, H: int, gate: str, eps: float):
    """The operands of attn = rmsnorm_per_head(sca + awk) * gate(rmx) (cute attn_gate_fwd; all (M, H * V) bf16 with the
    same contiguous layout) without writing attn, or None when the shape is not handled (V not 128 / 256)."""
    M = sca.numel() // sca.shape[-1]
    D = sca.shape[-1]
    V = D // H
    if V not in (128, 256) or M % BLOCK or not (sca.is_contiguous() and awk.is_contiguous() and rmx.is_contiguous()):
        return None
    if not (sca.dtype == awk.dtype == rmx.dtype == torch.bfloat16 and sca.shape == awk.shape == rmx.shape):
        return None
    rt, ct = _tiles("act")
    row, col = _empty(M, D, sca.device, rt), _empty(D, M, sca.device, ct)
    _gate_kernel[(M // BLOCK, H)](sca, awk, rmx, D, float(eps), *_out_args(row, col), V=V, SILU=gate == "silu",
                                  COL_TILE=ct, BM=BLOCK, num_warps=16)
    return row, col


def _out_args(row: Fp8Mat, col: Fp8Mat):
    return (row.data, row.data.stride(0), row.scale, _sstride(row), col.data, col.data.stride(0), col.scale, _sstride(col))


def swiglu_quantize(h1: torch.Tensor, h3: torch.Tensor) -> Tuple[Fp8Mat, Fp8Mat]:
    """The operands of hidden = silu(h1) * h3 (M, E) ("act": the A operand of the fc2 GEMM, the B operand hidden^T of
    its weight gradient) without writing hidden."""
    M, E = h1.shape
    # h1 / h3 may be the column halves of one (M, 2 E) GEMM output (row stride 2 E)
    assert h1.shape == h3.shape and h1.stride() == h3.stride() and h1.stride(1) == 1 and h1.dtype == torch.bfloat16
    rt, ct = _tiles("act")
    row, col = _empty(M, E, h1.device, rt), _empty(E, M, h1.device, ct)
    _swiglu_fwd_kernel[(M // BLOCK, E // BLOCK)](h1, h3, h1.stride(0), *_out_args(row, col), COL_TILE=ct, BM=BLOCK, num_warps=16)
    return row, col


def swiglu_bwd_quantize(h1: torch.Tensor, h3: torch.Tensor, g: torch.Tensor) -> Tuple[Fp8Mat, Fp8Mat]:
    """g = d hidden (M, E) -> the operands of [dh1 | dh3] (M, 2 E) ("grad": the A operand of the dX GEMM against
    [w1; w3], the A operand of the [w1; w3] weight gradient) without writing dh1 / dh3."""
    M, E = h1.shape
    # h1 / h3 may be the column halves of one (M, 2 E) GEMM output (row stride 2 E)
    assert h1.shape == h3.shape == g.shape and h1.stride() == h3.stride() and h1.stride(1) == 1 and g.stride(1) == 1
    rt, ct = _tiles("grad")
    row, col = _empty(M, 2 * E, h1.device, rt), _empty(2 * E, M, h1.device, ct)
    _swiglu_bwd_kernel[(M // BLOCK, E // BLOCK)](h1, h3, g, h1.stride(0), g.stride(0), E, *_out_args(row, col), COL_TILE=ct, BM=BLOCK, num_warps=16)
    return row, col


def _sstride(m: Fp8Mat) -> int:
    return m.scale.stride(0) if m.tile else m.scale.stride(1)


def _launch(x: torch.Tensor, row: Optional[Fp8Mat], col: Optional[Fp8Mat], off: Tuple[int, int]):
    R, C = x.shape
    assert x.dtype in (torch.bfloat16, torch.float32) and x.stride(1) == 1 and R % BLOCK == 0 and C % BLOCK == 0, (x.dtype, x.stride(), x.shape)
    assert off[0] % BLOCK == 0 and off[1] % BLOCK == 0, off
    d = x
    _fp8_kernel[(R // BLOCK, C // BLOCK)](
        x, x.stride(0), off[0], off[1],
        row.data if row else d, row.data.stride(0) if row else 0, row.scale if row else d, _sstride(row) if row else 0,
        col.data if col else d, col.data.stride(0) if col else 0, col.scale if col else d, _sstride(col) if col else 0,
        ROWWISE=row is not None, COLWISE=col is not None, ROW_TILE=bool(row and row.tile), COL_TILE=bool(col and col.tile),
        BM=BLOCK, ROUND_BF16=x.dtype == torch.float32, num_warps=8,
    )


def _tiles(kind: str) -> Tuple[bool, bool]:
    """(rowwise tiled, colwise tiled) for "act" (a layer input: rowwise is the A operand of Y = X W^T, colwise the B
    operand X^T of dW), "grad" (an output gradient: rowwise the A operand of dX = dY W, colwise the A operand dY^T of
    dW) and "weight" (both operands of W in 128 x 128 tiles: the B operands of Y = X W^T and dX = dY W)."""
    if kind == "weight":
        return True, True
    return False, (kind == "act") == (WGRAD_TILE == "x")


def quantize(x: torch.Tensor, rowwise: bool = True, colwise: bool = True, kind: str = "act") -> Tuple[Optional[Fp8Mat], Optional[Fp8Mat]]:
    if kind == "weight":
        return quantize_cat([x], 0, rowwise, colwise, kind)
    R, C = x.shape
    rt, ct = _tiles(kind)
    row = _empty(R, C, x.device, rt) if rowwise else None
    col = _empty(C, R, x.device, ct) if colwise else None
    _launch(x, row, col, (0, 0))
    return row, col


def quantize_cat(xs: Sequence[torch.Tensor], dim: int, rowwise: bool = True, colwise: bool = True, kind: str = "act") -> Tuple[Optional[Fp8Mat], Optional[Fp8Mat]]:
    assert dim in (0, 1)
    R = sum(x.shape[0] for x in xs) if dim == 0 else xs[0].shape[0]
    C = sum(x.shape[1] for x in xs) if dim == 1 else xs[0].shape[1]
    rt, ct = _tiles(kind)
    lazy = kind == "weight" and rowwise and colwise and LAZY_WEIGHT_T
    row = _empty(R, C, xs[0].device, rt) if rowwise else None
    col = _empty(C, R, xs[0].device, ct) if colwise and not lazy else None
    off = 0
    for x in xs:
        assert x.shape[1 - dim] == (C, R)[dim], (x.shape, R, C)
        _launch(x, row, col, (off, 0) if dim == 0 else (0, off))
        off += x.shape[dim]
    return row, (_LazyTransposed(row) if lazy else col)


_FUSED_C = []  # xllm_extension.ops.fp8_blockwise_gemm (cuBLASLt with a C matrix), or None


def _fused_c_op():
    if not _FUSED_C and os.environ.get("XLLM_FP8_FUSED_C", "1") == "0":
        _FUSED_C.append(None)
    if not _FUSED_C:
        try:
            import xllm_extension
            _FUSED_C.append(getattr(xllm_extension.ops, "fp8_blockwise_gemm", None))
        except ImportError:
            _FUSED_C.append(None)
    return _FUSED_C[0]


_DG = []  # the deep_gemm module (XLLM_FP8_DEEPGEMM=1 and installed), or None


def _deep_gemm():
    if not _DG:
        mod = None
        if DEEPGEMM:
            try:
                import deep_gemm as mod
                if DEEPGEMM_SMS:
                    mod.set_num_sms(DEEPGEMM_SMS)
            except ImportError:
                pass
        _DG.append(mod)
    return _DG[0]


def mm(a: Fp8Mat, b: Fp8Mat, bias: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None,
       residual: Optional[torch.Tensor] = None) -> torch.Tensor:
    """a (M, K), b (N, K) -> a @ b^T (M, N) bf16 (+ ``residual`` (M, N) bf16: accumulated in the GEMM epilogue by
    xllm_extension's cuBLASLt op, which torch._scaled_mm cannot do; a separate add without the extension)."""
    assert a.cols == b.cols and a.cols % (4 * BLOCK) == 0, (a.data.shape, b.data.shape)
    if residual is not None:
        op = _fused_c_op()
        if op is not None and bias is None and out is None and not a.tile and residual.dtype == torch.bfloat16:
            d = torch.empty(a.rows, b.rows, dtype=torch.bfloat16, device=a.data.device)
            op(a.data, a.a_scale(), b.data, b.b_scale(), d, residual.contiguous(), b.tile)
            return d
        d = mm(a, b, bias, out)
        return torch.add(d, residual, out=d)
    if bias is None and not a.tile and b.tile and (out is None or out.is_contiguous()):
        dg = _deep_gemm()
        if dg is not None:
            d = out if out is not None else torch.empty(a.rows, b.rows, dtype=torch.bfloat16, device=a.data.device)
            dg.fp8_gemm_nt((a.data, a.scale), (b.data, b.scale), d)
            return d
    kw = {} if out is None else {"out": out}
    return torch._scaled_mm(a.data, b.data.t(), a.a_scale(), b.b_scale(), bias=bias, out_dtype=torch.bfloat16, **kw)


def dequantize(m: Fp8Mat) -> torch.Tensor:
    R, C = m.rows, m.cols
    if m.tile:
        s = m.scale.float().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
        return m.data.float() * s
    return m.data.float().view(R, C // BLOCK, BLOCK).mul(m.scale.float()[:, :, None]).view(R, C)


def quantize_reference(x: torch.Tensor, tile: bool = False) -> torch.Tensor:
    R, C = x.shape
    xb = x.float().view(R // BLOCK, BLOCK, C // BLOCK, BLOCK) if tile else x.float().view(R, C // BLOCK, BLOCK)
    amax = xb.abs().amax((1, 3), keepdim=True) if tile else xb.abs().amax(-1, keepdim=True)
    scale = torch.exp2((torch.floor(torch.log2(amax.clamp(min=2.0 ** -118))) - 8).clamp(min=-126))
    return (xb / scale).clamp(-448, 448).to(E4M3).float().mul(scale).view(R, C)
