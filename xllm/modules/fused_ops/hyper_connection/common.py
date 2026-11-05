"""The stream kernels' contracts and shared pieces: the stream layout [T, N <= 4, D], the launchers' checks, the channel
tile, and the Triton helpers that load and store one stream's tile, a token's coefficients and the 4 x 4 mixes.
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


def _ptr(t, stand_in):
    """A kernel pointer argument that may be absent: `stand_in` is passed in its place (a tensor the kernel does have),
    never dereferenced since the matching constexpr flag is off."""
    return stand_in if t is None else t


def _tile(D: int, max_tile: int = 1024) -> int:
    """Channel tile of a stream op: D padded to a power of two when it fits, else a power of two that
    divides D (1536 = 3 x 512), else a masked max_tile."""
    if D <= max_tile:
        return triton.next_power_of_2(D)
    pow2 = D & -D  # largest power of two dividing D
    return min(pow2, max_tile) if pow2 >= 256 else max_tile


# ---------------------------------------------------------------------------------------------
# Triton helpers of the per-token stream kernels: one program per token over channel tiles [CH]; the four streams are
# explicit (n >= N masked off), the coefficient reductions accumulate elementwise over the tiles and reduce once.


@triton.jit
def _ld(ptr, t, n, N, D, d, m):
    return tl.load(ptr + (t * N + n) * D + d, mask=m & (n < N), other=0.0).to(tl.float32)


@triton.jit
def _st(ptr, t, n, N, D, d, m, v):
    # store v rounded to the tensor's dtype; -> the rounded value as FP32
    h = v.to(ptr.dtype.element_ty)
    tl.store(ptr + (t * N + n) * D + d, h, mask=m & (n < N))
    return h.to(tl.float32)


@triton.jit
def _ld2(hi_ptr, lo_ptr, t, n, N, D, d, m, HAS_LO: tl.constexpr):
    # a stream kept as hi + lo (two low-precision tensors, hi = round(x), lo = round(x - hi)); lo optional
    x = _ld(hi_ptr, t, n, N, D, d, m)
    if HAS_LO:
        x += _ld(lo_ptr, t, n, N, D, d, m)
    return x


@triton.jit
def _st2(hi_ptr, lo_ptr, t, n, N, D, d, m, v):
    # split an FP32 value into hi = round(v) and lo = round(v - hi) in the tensors' dtype; -> hi as FP32
    h = v.to(hi_ptr.dtype.element_ty)
    tl.store(hi_ptr + (t * N + n) * D + d, h, mask=m & (n < N))
    h = h.to(tl.float32)
    tl.store(lo_ptr + (t * N + n) * D + d, (v - h).to(lo_ptr.dtype.element_ty), mask=m & (n < N))
    return h


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
def _mix(r0, r1, r2, r3, x0, x1, x2, x3):
    # sum_m R[n, m] x_m for one destination row n
    return r0 * x0 + r1 * x1 + r2 * x2 + r3 * x3
