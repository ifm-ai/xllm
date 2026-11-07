"""Rotary position embedding (CuTe DSL): one bf16 pass over [B, L, N, D] with fp32 math.

The torch version (x.float() -> complex multiply -> bf16 -> torch.cat with the non-rotated tail) runs 4-5 passes
over the tensor; this kernel reads x once and writes the output once (gekko-0.9B B2 L16K: 68 -> ~5 ms per step).
``backward`` applies the conjugate (the gradient of the forward)."""
import os
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

try:
    from .adaptive_working_memory import _compiled, _fake
except ImportError:  # loaded from its file (tests)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("cute_adaptive_working_memory_fwd",
                                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaptive_working_memory.py"))
    _awm = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_awm)
    _compiled, _fake = _awm._compiled, _awm._fake


class Rope:
    """out[b, l, n, :R] = rotate(x[b, l, n, :R], freqs[l]), out[b, l, n, R:] = x[b, l, n, R:] (bf16 in / out).
    freqs: the fp32 (cos, sin) pairs of the complex64 [L, R / 2] table; a thread handles 8 elements (4 pairs)."""

    THREADS = 256

    def __init__(self, N: int, D: int, R: int, backward: bool):
        self.N, self.D, self.R, self.backward = N, D, R, backward
        assert D % 8 == 0 and R % 8 == 0 and 0 < R <= D

    @cute.jit
    def __call__(self, tX: cute.Tensor, tF: cute.Tensor, tOut: cute.Tensor, B: Int32, L: Int32, stream: cuda.CUstream):
        n_thr = B * L * self.N * (self.D // 8)
        self.kernel(tX.iterator, tF.iterator, tOut.iterator, L, n_thr).launch(
            grid=[(n_thr + self.THREADS - 1) // self.THREADS, 1, 1], block=[self.THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pX: cute.Pointer, pF: cute.Pointer, pOut: cute.Pointer, L: Int32, n_thr: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        N, D, R = self.N, self.D, self.R
        gid = bidx * self.THREADS + tidx
        if gid < n_thr:
            e = (gid % (D // 8)) * 8
            row = gid // (D // 8)  # (b, l, n)
            l = (row // N) % L
            off = Int64(row) * D + e
            x8 = cute.make_tensor((pX + off).align(16), cute.make_layout(8)).load()
            if e < R:
                f8 = cute.make_tensor((pF + (Int64(l) * R + e)).align(16), cute.make_layout(8)).load()
                xf = cute.make_rmem_tensor(cute.make_layout(8), Float32)
                xf.store(x8.to(Float32))
                ff = cute.make_rmem_tensor(cute.make_layout(8), Float32)
                ff.store(f8)
                of = cute.make_rmem_tensor(cute.make_layout(8), Float32)
                for i in cutlass.range_constexpr(4):
                    x0, x1, c = xf[2 * i], xf[2 * i + 1], ff[2 * i]
                    s = -ff[2 * i + 1] if self.backward else ff[2 * i + 1]
                    of[2 * i] = x0 * c - x1 * s
                    of[2 * i + 1] = x0 * s + x1 * c
                cute.make_tensor((pOut + off).align(16), cute.make_layout(8)).store(of.load().to(BFloat16))
            else:
                cute.make_tensor((pOut + off).align(16), cute.make_layout(8)).store(x8)


def rope(x: torch.Tensor, freqs_cis: torch.Tensor, rope_dim: Optional[int] = None, backward: bool = False) -> torch.Tensor:
    """Rotary embedding of x [B, L, N, D] (bf16) over its first rope_dim channels (default D); freqs_cis complex64
    [L, rope_dim / 2]. backward=True applies the conjugate rotation (the gradient of the forward)."""
    B, L, N, D = x.shape
    R = D if rope_dim is None else rope_dim
    assert x.dtype == torch.bfloat16 and freqs_cis.dtype == torch.complex64 and freqs_cis.shape == (L, R // 2)
    assert x.numel() < 2 ** 31
    x = x.contiguous()
    f = torch.view_as_real(freqs_cis).contiguous()
    out = torch.empty_like(x)
    fn = _compiled(("rope", N, D, R, bool(backward)), lambda: Rope(N, D, R, bool(backward)), lambda: (
        _fake(BFloat16), _fake(Float32), _fake(BFloat16), Int32(0), Int32(0)))
    fn(x.reshape(-1), f.reshape(-1), out.reshape(-1), B, L)
    return out


def rope_supported(x: torch.Tensor, rope_dim: int) -> bool:
    if os.environ.get("XLLM_CUTE_ROPE", "1") == "0" or x.dtype != torch.bfloat16 or not x.is_cuda:
        return False
    D = x.shape[-1]
    if x.dim() != 4 or D % 8 or rope_dim % 8 or rope_dim <= 0 or rope_dim > D or x.numel() >= 2 ** 31:
        return False
    return torch.cuda.get_device_capability(x.device)[0] >= 9
