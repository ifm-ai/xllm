"""mHC transport: Sinkhorn-Knopp on 4 x 4 logits (column then row log-normalization, `iterations` times).
Forward keeps the iterates in registers and saves only the logits; backward recomputes them into a
transient per-program scratch (persistent programs, ~86 MB at 20 iterations) and runs the normalizations
in reverse:
  y = x - lse(x) along an axis  ->  dx = dy - exp(y) * sum(dy) along that axis.
"""

import torch
import triton
import triton.language as tl


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
