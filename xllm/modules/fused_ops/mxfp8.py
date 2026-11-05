"""MXFP8 GEMMs for the fused Gekko block on Blackwell (``ModelConf.fp8_gemm``).

Operands are OCP MX tensors: e4m3 elements with one e8m0 (power of two) scale per 32 consecutive elements along the
contraction dimension; the GEMMs run on cuBLASLt through ``torch._scaled_mm`` (fp32 accumulation, bf16 output).
Compared with per-tensor fp8 scaling, the block scales keep the dynamic range of every 32-element group (no amax
history / delayed scaling, an outlier does not flush its neighbours to zero) and the quantisation is one pass.

Every bf16 matrix X (R, C) is quantised by one kernel launch into the two operands the three GEMMs of a linear need:
  rowwise : X   as (R, C), scales along C -> the A operand of Y = X W^T (or the B operand W of the same GEMM);
  colwise : X^T as (C, R), scales along R -> the A operand dY^T and the B operand X^T of dW = dY^T X, or the B
                                             operand W^T of dX = dY W.
Both operands are stored in X's own (R, C) layout: cuBLASLt on Blackwell takes MX operands in either K-major or
MN-major layout at the same speed (measured), so the colwise operand is the transposed *view* of a (R, C) buffer
quantised in groups of 32 along R, and the kernel never writes transposed (both passes store coalesced rows).
``MxLinearSet`` quantises the stacked weights [W1; W2; ...] of one layer at once and keeps both operands for the
backward: the forward GEMMs read row slices of the rowwise operand, dX = [dY1 | dY2 | ...] [W1; W2; ...] is one GEMM
on the colwise operand, and [dW1; dW2; ...] = [dY1 | dY2 | ...]^T X is one GEMM whose row slices are the weight
gradients (``mx_quantize_cat`` builds the operands of the concatenated gradients without materialising it). Up to 4
matrices share one quantisation launch; after the first launch per constexpr set the kernel is issued through the
compiled kernel's C launcher (``_Launcher``). The cute SwiGLU kernels write their outputs as MX operands directly
(``swiglu_fwd_mx`` / ``swiglu_bwd_mx``).

Scale layout: cuBLASLt's 32x4x4 swizzle (``SwizzleType.SWIZZLE_32_4_4``): the (R, C/32) scale matrix is split into
tiles of 128 rows x 4 scale columns (= 128 elements); tile (rb, cb) is 512 contiguous bytes at (rb * ncb + cb) * 512 and
its element (r, c) sits at (r % 32) * 16 + ((r // 32) % 4) * 4 + c (``_scale_off``). All dimensions must be multiples
of 128. Rounding: scale
= 2^(floor(log2(amax)) - 8) (the group's amax lands in [256, 512); the few values above 448 saturate), round to
nearest even e4m3.

Measured (B200, cuBLASLt 13.1; per layer, fwd + dX + dW of all linears, operands already quantised): gekko-4B
M=8192 bf16 3457 -> 1973 us, gekko-7.1B M=16384 12888 -> 7009, gekko-0.9B M=32768 4042 -> 2334. The quantisation
kernel: 16384 x 8192 both operands in ~100 us (one read, 32 x 128 tiles, 2 warps); numbers in handoff/HANDOFF.md.
"""
import functools
from typing import List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

E4M3 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu
BLOCK = 128


@functools.lru_cache(maxsize=None)
def _sm90(device_index: int) -> bool:
    """Hopper: the operands use 128-element fp32 block scales instead (xllm/modules/fused_ops/fp8_sm90.py)."""
    return torch.cuda.get_device_capability(device_index)[0] == 9


def _hopper(t: torch.Tensor) -> bool:
    return _sm90(t.device.index if t.device.index is not None else torch.cuda.current_device())


class MxMat:
    """One MX operand: ``data`` (rows, cols) e4m3 - K-major (contiguous rows, possibly a row slice of a bigger buffer)
    or MN-major (the transposed view of a contiguous (cols, rows) buffer: the colwise operands); ``scale`` the
    swizzled e8m0 scales (uint8 storage) of exactly these rows, groups of 32 along cols."""

    __slots__ = ("data", "scale", "_scale8")

    def __init__(self, data: torch.Tensor, scale: torch.Tensor):
        self.data = data
        self.scale = scale
        self._scale8 = None

    @property
    def scale8(self) -> torch.Tensor:
        """The scales as a float8_e8m0fnu tensor (the view is made once: ``_scaled_mm`` is called per GEMM)."""
        if self._scale8 is None:
            self._scale8 = self.scale.view(E8M0)
        return self._scale8

    @property
    def rows(self) -> int:
        return self.data.shape[0]

    @property
    def cols(self) -> int:
        return self.data.shape[1]

    def row_slice(self, r0: int, r1: int) -> "MxMat":
        assert r0 % BLOCK == 0 and r1 % BLOCK == 0 and self.data.stride(1) == 1, (r0, r1, self.data.stride())
        per_row = self.cols // 32
        return MxMat(self.data[r0:r1], self.scale[r0 * per_row:r1 * per_row])


def _empty_mx(rows: int, cols: int, device, transposed: bool = False) -> MxMat:
    """An operand (rows, cols); ``transposed``: as the transposed view of a (cols, rows) buffer (MN-major)."""
    assert rows % BLOCK == 0 and cols % BLOCK == 0, (rows, cols)
    data = torch.empty(cols, rows, dtype=E4M3, device=device).t() if transposed else torch.empty(rows, cols, dtype=E4M3, device=device)
    return MxMat(data, torch.empty(rows * (cols // 32), dtype=torch.uint8, device=device))


@triton.jit
def _e8m0(amax):
    """amax (fp32) -> (biased e8m0 exponent as uint8, the inverse scale 2^(127 - e8) as fp32)."""
    ebits = (amax.to(tl.int32, bitcast=True) >> 23) & 0xFF  # floor(log2(amax)) + 127 for a normal amax
    e8 = tl.minimum(tl.maximum(ebits - 8, 0), 254)
    inv = ((254 - e8) << 23).to(tl.float32, bitcast=True)
    return e8.to(tl.uint8), inv


@triton.jit
def _scale_off(r, scol, ncb):
    """Offset of scale (row r, scale column scol) in the swizzled layout of a scale matrix with ncb column tiles
    (tile (r // 128, scol // 4), element (r % 32) * 16 + ((r // 32) % 4) * 4 + scol % 4)."""
    return ((r // 128) * ncb + scol // 4) * 512 + (r % 32) * 16 + ((r // 32) % 4) * 4 + (scol % 4)


_MAXT = 4  # tensors per grouped launch
_TR, _TC = 32, 128  # tile: 32 rows x 128 columns, one global read for both groupings
# Triton specialises integer arguments on divisibility by 16 (and == 1) and compiles one kernel per class; the tile
# counts differ per call site and are only index arithmetic, so they are left out of the specialisation (one compiled
# kernel per constexpr set); strides and offsets are multiples of 128 everywhere (one class) and must stay specialised:
# the address computations need their divisibility for 16-byte vectorised loads / stores.
_NO_SPEC = ["t1", "t2", "t3", "n0", "n1", "n2", "n3", "ncb_r", "ncb_c"]


@triton.jit(do_not_specialize=_NO_SPEC)
def _mx_kernel(p0, p1, p2, p3, s0, s1, s2, s3, t1, t2, t3, n0, n1, n2, n3, o0, o1, o2, o3,
               qr_ptr, ld_r, ncb_r, sr_ptr, qc_ptr, ld_c, ncb_c, sc_ptr,
               DIM: tl.constexpr, ROWWISE: tl.constexpr, COLWISE: tl.constexpr, TR: tl.constexpr, TC: tl.constexpr):
    """One launch for up to 4 bf16 matrices X_i (pointers p_i - tensors, an int64 address cast to a pointer loses the
    alignment information and the vectorised loads -, row strides s_i, n_i column tiles,
    first tile t_i (t0 = 0)) written side by side into the output buffers: X_i's element (r, c) lands at rows /
    columns offset o_i along DIM (0: [X1; X2; ..] stacked, 1: [X1 | X2 | ..] side by side). A program handles one
    TR x TC tile from one read: the rowwise operand (scales along the columns: TC // 32 groups per row) and the colwise
    operand (scales along the rows: one group of TR = 32 rows per column), both in the (R, C) layout of X (the
    colwise operand's transposed view is the MN-major (C, R) operand; ncb_r = C_total / 128, ncb_c = R_total / 128
    scale column tiles). 2 warps: the tile is 8 KB of bf16, more warps only add layout shuffles."""
    pid = tl.program_id(0)
    i = (pid >= t1).to(tl.int32) + (pid >= t2).to(tl.int32) + (pid >= t3).to(tl.int32)
    p = tl.where(i == 0, p0, tl.where(i == 1, p1, tl.where(i == 2, p2, p3)))
    stride = tl.where(i == 0, s0, tl.where(i == 1, s1, tl.where(i == 2, s2, s3)))
    nct = tl.where(i == 0, n0, tl.where(i == 1, n1, tl.where(i == 2, n2, n3)))
    off = tl.where(i == 0, o0, tl.where(i == 1, o1, tl.where(i == 2, o2, o3)))
    local = pid - tl.where(i == 0, 0, tl.where(i == 1, t1, tl.where(i == 2, t2, t3)))
    rb = local // nct
    cb = local % nct
    r0 = tl.where(DIM == 0, off, 0)
    c0 = tl.where(DIM == 1, off, 0)
    x_ptr = p
    rows = rb * TR + tl.arange(0, TR)
    cols = cb * TC + tl.arange(0, TC)
    x = tl.load(x_ptr + rows[:, None] * stride + cols[None, :]).to(tl.float32)
    orows = r0 + rows
    ocols = c0 + cols
    if ROWWISE:
        xg = tl.reshape(x, (TR, TC // 32, 32))
        e8, inv = _e8m0(tl.max(tl.abs(xg), axis=2))  # (TR, TC // 32)
        q = tl.reshape(xg * inv[:, :, None], (TR, TC)).to(tl.float8e4nv)
        tl.store(qr_ptr + orows[:, None] * ld_r + ocols[None, :], q)
        g = tl.arange(0, TC // 32)
        tl.store(sr_ptr + _scale_off(orows[:, None], (c0 // 32) + cb * (TC // 32) + g[None, :], ncb_r), e8)
    if COLWISE:
        e8, inv = _e8m0(tl.max(tl.abs(x), axis=0))  # (TC,)
        q = (x * inv[None, :]).to(tl.float8e4nv)
        tl.store(qc_ptr + orows[:, None] * ld_c + ocols[None, :], q)
        tl.store(sc_ptr + _scale_off(ocols, (r0 // 32) + rb, ncb_c), e8)


class _Launcher:
    """Launches ``_mx_kernel`` through Triton once per constexpr combination and then directly through the compiled
    kernel's C launcher (Triton's Python launch path costs ~19 us of host time per call; the int arguments are
    ``do_not_specialize`` so one compiled kernel serves every shape). Falls back to the normal path if Triton's
    internals do not match."""

    def __init__(self):
        self.cache = {}
        self.direct = True

    @staticmethod
    def _key(args, DIM, ROWWISE, COLWISE):
        # the specialised integer arguments (strides, offsets: multiples of 128, or 0) select the compiled kernel too
        return (DIM, ROWWISE, COLWISE, torch.cuda.current_device(), tuple(int(v) % 16 == 0 for v in args[4:8] + args[15:19]))

    def __call__(self, grid: int, args: tuple, DIM: bool, ROWWISE: bool, COLWISE: bool):
        key = self._key(args, DIM, ROWWISE, COLWISE)
        k = self.cache.get(key)
        if k is None or not self.direct:
            k = _mx_kernel.run(*args, grid=(grid,), warmup=False, DIM=DIM, ROWWISE=ROWWISE, COLWISE=COLWISE, TR=_TR, TC=_TC, num_warps=2)
            if k is not None and self.direct:
                try:
                    k.run, k.function, k.packed_metadata  # the launcher attributes used below
                    self.cache[key] = k
                except AttributeError:
                    self.direct = False
            return
        stream = _raw_stream(key[3])  # ~1 us against 5 for torch.cuda.current_stream().cuda_stream
        k.run(grid, 1, 1, stream, k.function, k.packed_metadata, None, None, None, *args, DIM, ROWWISE, COLWISE, _TR, _TC)


_launcher = _Launcher()
_raw_stream = torch._C._cuda_getCurrentRawStream  # what Triton's own driver uses


def _launch(xs: Sequence[torch.Tensor], dim: int, row: Optional[MxMat], col: Optional[MxMat], off: int = 0):
    """Quantise the matrices xs, laid out along ``dim`` (0: stacked, 1: side by side) from offset ``off`` in the output
    buffers of the rowwise operand ``row`` and / or the colwise operand ``col`` (a transposed view)."""
    assert 1 <= len(xs) <= _MAXT, len(xs)
    ptrs, strides, starts, ncts, offs = [], [], [], [], []
    tiles = 0
    for x in xs:
        R, C = x.shape
        assert x.dtype == torch.bfloat16 and x.stride(1) == 1 and R % BLOCK == 0 and C % BLOCK == 0, (x.dtype, x.stride(), x.shape)
        ptrs.append(x)
        strides.append(x.stride(0))
        starts.append(tiles)
        ncts.append(C // _TC)
        offs.append(off)
        tiles += (R // _TR) * (C // _TC)
        off += x.shape[dim]
    pad = _MAXT - len(xs)
    ptrs += [ptrs[0]] * pad
    strides += [0] * pad
    starts += [tiles] * pad  # unused slots start after the last tile
    ncts += [1] * pad
    offs += [0] * pad
    d = xs[0]  # placeholder for an unused output
    cbuf = col.data.t() if col is not None else None  # the (R_total, C_total) buffer
    args = (*ptrs, *strides, starts[1], starts[2], starts[3], *ncts, *offs,
            row.data if row else d, row.data.stride(0) if row else 0, (row.cols // BLOCK) if row else 0, row.scale if row else d,
            cbuf if col else d, cbuf.stride(0) if col else 0, (col.cols // BLOCK) if col else 0, col.scale if col else d)
    _launcher(tiles, args, dim == 1, row is not None, col is not None)


def mx_quantize(x: torch.Tensor, rowwise: bool = True, colwise: bool = True, kind: str = "act") -> Tuple[Optional[MxMat], Optional[MxMat]]:
    """x (R, C) bf16 -> (rowwise operand X (R, C), colwise operand X^T (C, R)); either may be skipped (None).
    ``kind`` ("act" / "grad" / "weight"): picks the block shapes on Hopper (fp8_sm90._tiles), ignored on Blackwell."""
    if _hopper(x):
        from xllm.modules.fused_ops import fp8_sm90
        return fp8_sm90.quantize(x, rowwise, colwise, kind)
    R, C = x.shape
    row = _empty_mx(R, C, x.device) if rowwise else None
    col = _empty_mx(C, R, x.device, transposed=True) if colwise else None
    _launch([x], 0, row, col)
    return row, col


def mx_quantize_cat(xs: Sequence[torch.Tensor], dim: int, rowwise: bool = True, colwise: bool = True, kind: str = "act") -> Tuple[Optional[MxMat], Optional[MxMat]]:
    """The operands of the concatenation of xs along ``dim`` (1: [X1 | X2 | ...] sharing the rows; 0: [X1; X2; ...]
    sharing the columns) without materialising it: (rowwise operand of the concatenation, colwise operand of it)."""
    assert dim in (0, 1)
    if _hopper(xs[0]):
        from xllm.modules.fused_ops import fp8_sm90
        return fp8_sm90.quantize_cat(xs, dim, rowwise, colwise, kind)
    R = sum(x.shape[0] for x in xs) if dim == 0 else xs[0].shape[0]
    C = sum(x.shape[1] for x in xs) if dim == 1 else xs[0].shape[1]
    row = _empty_mx(R, C, xs[0].device) if rowwise else None
    col = _empty_mx(C, R, xs[0].device, transposed=True) if colwise else None
    for x in xs:
        assert x.shape[1 - dim] == (C, R)[dim], (x.shape, R, C)
    for k in range(0, len(xs), _MAXT):  # one launch per 4 matrices
        chunk = xs[k:k + _MAXT]
        _launch(chunk, dim, row, col, sum(x.shape[dim] for x in xs[:k]))
    return row, col


def _cute_swiglu():
    """The cute SwiGLU module (through the package, or from its file when this module is loaded standalone)."""
    try:
        from xllm.modules.fused_ops.cute import swiglu
        return swiglu
    except ImportError:
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location("cute_swiglu_mx", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cute", "swiglu.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def swiglu_fwd_mx(h1: torch.Tensor, h3: torch.Tensor) -> Tuple[MxMat, MxMat]:
    """(rowwise, colwise) operands of hidden = silu(h1) * h3 straight from the cute SwiGLU kernel (no bf16 hidden)."""
    if _hopper(h1):
        from xllm.modules.fused_ops import fp8_sm90
        return fp8_sm90.swiglu_quantize(h1, h3)
    M, F = h1.shape
    row, col = _empty_mx(M, F, h1.device), _empty_mx(F, M, h1.device, transposed=True)
    _cute_swiglu().swiglu_mx(h1, h3, None, row.data.view(torch.uint8), row.scale, col.data.t().view(torch.uint8), col.scale)
    return row, col


def swiglu_bwd_mx(h1: torch.Tensor, h3: torch.Tensor, h_grad: torch.Tensor) -> Tuple[MxMat, MxMat]:
    """(rowwise operand of [h1_grad | h3_grad] (M, 2F), colwise operand (2F, M)) straight from the cute SwiGLU
    backward kernel (no bf16 gradients)."""
    if _hopper(h1):
        from xllm.modules.fused_ops import fp8_sm90
        return fp8_sm90.swiglu_bwd_quantize(h1, h3, h_grad)
    M, F = h1.shape
    row, col = _empty_mx(M, 2 * F, h1.device), _empty_mx(2 * F, M, h1.device, transposed=True)
    _cute_swiglu().swiglu_mx(h1, h3, h_grad, row.data.view(torch.uint8), row.scale, col.data.t().view(torch.uint8), col.scale)
    return row, col


def mx_gate_quantize(sca: torch.Tensor, awk: torch.Tensor, rmx: torch.Tensor, H: int, gate: str, eps: float):
    """(rowwise, colwise) operands of the attention gate's output straight from one kernel (Hopper: fp8_sm90.
    gate_quantize, value heads of 128 / 256), or None: then attn_gate_fwd + mx_quantize."""
    if _hopper(sca):
        from xllm.modules.fused_ops import fp8_sm90
        return fp8_sm90.gate_quantize(sca, awk, rmx, H, gate, eps)
    return None


def mx_mm(a: MxMat, b: MxMat, bias: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None,
          residual: Optional[torch.Tensor] = None) -> torch.Tensor:
    """a (M, K), b (N, K) -> a @ b^T (M, N) bf16 (fp32 accumulation; ``bias`` (N,) bf16 added in the epilogue)."""
    assert a.cols == b.cols, (a.data.shape, b.data.shape)
    if _hopper(a.data):
        from xllm.modules.fused_ops import fp8_sm90
        return fp8_sm90.mm(a, b, bias, out, residual)  # residual: in the GEMM epilogue
    if residual is not None:
        d = mx_mm(a, b, bias, out)
        return torch.add(d, residual, out=d)
    if out is None:
        return torch._scaled_mm(a.data, b.data.t(), a.scale8, b.scale8, bias=bias, out_dtype=torch.bfloat16)
    return torch._scaled_mm(a.data, b.data.t(), a.scale8, b.scale8, bias=bias, out_dtype=torch.bfloat16, out=out)


def mx_supported(device, *dims: int) -> bool:
    """Blackwell or Hopper, and every dimension a multiple of 128 (Hopper: 512)."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(device)[0] not in (9, 10):
        return False
    if torch.cuda.get_device_capability(device)[0] == 9:
        from xllm.modules.fused_ops import fp8_sm90
        return fp8_sm90.supported(*dims)
    return all(d % BLOCK == 0 for d in dims)


class MxLinearSet:
    """The fp8 operands of the weights W_i (N_i, K) of one layer that consume the same input (quantised once per
    forward, kept for the backward): rowwise operand of [W1; W2; ...] (per-weight row slices for the forward GEMMs)
    and colwise operand (K, sum N) for the one fused dX GEMM."""

    def __init__(self, weights: Sequence[torch.Tensor]):
        self.sizes = [w.shape[0] for w in weights]
        self.offsets = [sum(self.sizes[:i]) for i in range(len(weights) + 1)]
        if _hopper(weights[0]) and all(hasattr(w, "fp8_row") for w in weights):
            # fp8 all-gathered weights (fp8_fsdp.Fp8Weight, XLLM_FP8_AG): already the rowwise operands
            from xllm.modules.fused_ops import fp8_sm90
            self.rows = [w.fp8_row() for w in weights]
            self.row, self.col = None, fp8_sm90.lazy_transposed_cat(self.rows)
            return
        self.row, self.col = mx_quantize_cat([w.detach() for w in weights], dim=0, kind="weight")
        self.rows = [self.row.row_slice(self.offsets[i], self.offsets[i + 1]) for i in range(len(weights))]

    def forward(self, i: int, x_row: MxMat, bias: Optional[torch.Tensor] = None,
                residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x_row (M, K) rowwise -> x @ W_i^T (M, N_i) (+ residual (M, N_i))."""
        return mx_mm(x_row, self.rows[i], bias, residual=residual)

    def forward_all(self, x_row: MxMat) -> List[torch.Tensor]:
        """One GEMM for every weight: [x @ W_i^T] as column slices (views) of the (M, sum N) output."""
        if self.row is None:  # fp8 all-gathered weights (separate operands): one GEMM per weight
            return [self.forward(i, x_row) for i in range(len(self.sizes))]
        out = mx_mm(x_row, self.row)
        return [out[:, self.offsets[i]:self.offsets[i + 1]] for i in range(len(self.sizes))]

    def dgrad(self, dys: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, MxMat]:
        """dys: dY_i (M, N_i) bf16 -> (dX = sum_i dY_i W_i (M, K), colwise operand of [dY1 | dY2 | ...] for wgrad)."""
        assert len(dys) == len(self.sizes), (len(dys), self.sizes)
        dy_row, dy_col = mx_quantize_cat(dys, dim=1, kind="grad")
        return mx_mm(dy_row, self.col), dy_col

    def wgrad(self, dy_col: MxMat, x_col: MxMat) -> List[torch.Tensor]:
        """dy_col from ``dgrad``, x_col (K, M) the colwise operand of the input -> [dW_i = dY_i^T X (N_i, K)]: one GEMM
        (sum N, K), the gradients are its row slices."""
        stacked = mx_mm(dy_col, x_col)
        return [stacked[self.offsets[i]:self.offsets[i + 1]] for i in range(len(self.sizes))]


# ----------------------------------------------------------------------------------------------------------------------
# reference (tests)

def mx_dequantize(m: MxMat) -> torch.Tensor:
    """The values an MX operand represents, as (rows, cols) fp32."""
    R, C = m.rows, m.cols
    s = m.scale.view(R // BLOCK, C // BLOCK, 32, 4, 4)  # tile (rb, cb): [r_lo, r_hi, c]
    s = s.permute(0, 3, 2, 1, 4).reshape(R, C // 32)  # (rb, r_hi, r_lo) -> row; (cb, c) -> scale column
    scale = torch.exp2(s.float() - 127)
    return m.data.float().reshape(R, C // 32, 32).mul(scale[:, :, None]).reshape(R, C)


def mx_quantize_reference(x: torch.Tensor) -> torch.Tensor:
    """fp32 values of the OCP MX quantisation of x (scales along the last dimension)."""
    R, C = x.shape
    xb = x.float().view(R, C // 32, 32)
    amax = xb.abs().amax(-1, keepdim=True)
    e = torch.floor(torch.log2(amax.clamp(min=2.0 ** -126))) - 8
    scale = torch.exp2(e.clamp(min=-127))
    return (xb / scale).clamp(-448, 448).to(E4M3).float().mul(scale).view(R, C)
