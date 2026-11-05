"""Fused softmax backward over the last dim (CuTe DSL, any arch), accumulated into an existing gradient:

    out += (add if given) + y * (g - sum(g * y)),   y = softmax(x)  (fp32, recomputed from the logits x)

for rows of S = 64 .. 256 elements, one warp per row (S / 32 elements per lane, one vector load each). Replaces
softmax(x, fp32) kept for the backward + g.float() + _softmax_backward_data + .to(bf16) + the adds of the gekko
fused block, which together move ~9x the bytes of this kernel.
"""
import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float16, Float32, Int32, Int64

TORCH2CUTE = {torch.bfloat16: BFloat16, torch.float16: Float16}
LOG2E = 1.4426950408889634


class SoftmaxBwdAcc:
    def __init__(self, dtype, S: int, has_add: bool, warps: int = 8):
        assert S % 32 == 0 and 64 <= S <= 256
        self.dtype, self.S, self.has_add, self.warps = dtype, S, has_add, warps
        self.E = S // 32  # elements per lane

    @cute.jit
    def __call__(self, tX: cute.Tensor, tG: cute.Tensor, tAdd: cute.Tensor, tOut: cute.Tensor, rows: Int32,
                 stream: cuda.CUstream):
        grid = (rows + self.warps - 1) // self.warps
        self.kernel(tX.iterator, tG.iterator, tAdd.iterator, tOut.iterator, rows).launch(
            grid=[grid, 1, 1], block=[32 * self.warps, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pX: cute.Pointer, pG: cute.Pointer, pAdd: cute.Pointer, pOut: cute.Pointer, rows: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        row = bidx * self.warps + tidx // 32
        if row < rows:
            off = Int64(row) * self.S + lane * self.E
            al = 2 * self.E  # bytes of a lane's slice
            x = cute.make_tensor((pX + off).align(al), cute.make_layout(self.E)).load().to(Float32)
            g = cute.make_tensor((pG + off).align(al), cute.make_layout(self.E)).load().to(Float32)
            m = x[0]
            for i in cutlass.range_constexpr(1, self.E):
                m = cute.arch.fmax(m, x[i])
            for o in cutlass.range_constexpr(5):
                m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=1 << o))
            e = cute.make_rmem_tensor(cute.make_layout(self.E), Float32)
            s = Float32(0.0)
            for i in cutlass.range_constexpr(self.E):
                e[i] = cute.math.exp2((x[i] - m) * LOG2E, fastmath=True)
                s = s + e[i]
            for o in cutlass.range_constexpr(5):
                s = s + cute.arch.shuffle_sync_bfly(s, offset=1 << o)
            inv = cute.arch.rcp_approx(s)
            d = Float32(0.0)
            for i in cutlass.range_constexpr(self.E):
                e[i] = e[i] * inv
                d = d + e[i] * g[i]
            for o in cutlass.range_constexpr(5):
                d = d + cute.arch.shuffle_sync_bfly(d, offset=1 << o)
            outp = (pOut + off).align(al)
            acc = cute.make_tensor(outp, cute.make_layout(self.E)).load().to(Float32)
            if cutlass.const_expr(self.has_add):
                acc = acc + cute.make_tensor((pAdd + off).align(al), cute.make_layout(self.E)).load().to(Float32)
            r = cute.make_rmem_tensor(cute.make_layout(self.E), Float32)
            for i in cutlass.range_constexpr(self.E):
                r[i] = acc[i] + e[i] * (g[i] - d)
            cute.make_tensor(outp, cute.make_layout(self.E)).store(r.load().to(self.dtype))


_COMPILED = {}


def _fake(dtype, align):
    return cute.runtime.make_fake_compact_tensor(dtype, (cute.sym_int(),), assumed_align=align)


def supported(x: torch.Tensor) -> bool:
    return x.is_cuda and x.dtype in TORCH2CUTE and x.shape[-1] % 32 == 0 and 64 <= x.shape[-1] <= 256


def softmax_bwd_acc_(out: torch.Tensor, x: torch.Tensor, g: torch.Tensor, add: torch.Tensor = None) -> torch.Tensor:
    """out (in place) += (add) + softmax_backward(g, softmax(x)) over the last dim; all [..., S] contiguous, same
    dtype. Returns out."""
    S = x.shape[-1]
    rows = x.numel() // S
    for t in (out, x, g) + ((add,) if add is not None else ()):
        assert t.is_contiguous() and t.numel() == x.numel() and t.dtype == x.dtype, \
            [(tuple(u.shape), u.dtype, u.is_contiguous()) for u in (out, x, g) + ((add,) if add is not None else ())]
    key = (x.dtype, S, add is not None)
    if key not in _COMPILED:
        dt = TORCH2CUTE[x.dtype]
        al = 2 * (S // 32)
        _COMPILED[key] = cute.compile(
            SoftmaxBwdAcc(dt, S, add is not None), _fake(dt, al), _fake(dt, al), _fake(dt, al), _fake(dt, al), Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
    _COMPILED[key](x.reshape(-1), g.reshape(-1), (add if add is not None else out).reshape(-1), out.reshape(-1), rows)
    return out
