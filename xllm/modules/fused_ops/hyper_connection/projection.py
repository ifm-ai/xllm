"""The controller projection logits = rstd(Xf) * (Xf @ W) on the flattened streams Xf [T, N * D]: an rstd kernel and a
cuBLAS BF16 matmul with FP32 accumulate and output; its backward forms dX = dP W^T - c X in one kernel and dW on cuBLAS.
`hc_logits` is the module path's autograd Function; the fused block calls `logits_fwd` / `logits_bwd`.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rstd_kernel(x_ptr, rstd_ptr, K, eps, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        o = k0 + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + t * K + o, mask=o < K, other=0.0).to(tl.float32)
        acc += x * x
    tl.store(rstd_ptr + t, tl.rsqrt(tl.sum(acc, axis=0) / K + eps))


@triton.jit
def _proj_dx_kernel(x_ptr, dp_ptr, wt_ptr, c_ptr, dx_ptr, T, K, NL,
                    BT: tl.constexpr, BK: tl.constexpr, LP: tl.constexpr):
    # dx[t, k] = sum_l dp[t, l] w[k, l] - c[t] x[t, k]
    ot = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    ok = tl.program_id(1).to(tl.int64) * BK + tl.arange(0, BK)
    ol = tl.arange(0, LP)
    tm = ot < T
    km = ok < K
    lm = ol < NL
    dp = tl.load(dp_ptr + ot[:, None] * NL + ol[None, :], mask=tm[:, None] & lm[None, :], other=0.0)
    wt = tl.load(wt_ptr + ol[:, None] * K + ok[None, :], mask=lm[:, None] & km[None, :], other=0.0)  # W^T [NL, K]
    acc = tl.dot(dp, wt)
    c = tl.load(c_ptr + ot, mask=tm, other=0.0)
    m = tm[:, None] & km[None, :]
    x = tl.load(x_ptr + ot[:, None] * K + ok[None, :], mask=m, other=0.0).to(tl.float32)
    tl.store(dx_ptr + ot[:, None] * K + ok[None, :], (acc - c[:, None] * x).to(dx_ptr.dtype.element_ty), mask=m)


def logits_fwd(xf, w, eps, rstd=None):
    """xf [T, K] contiguous streams, w [K, NL] contiguous in xf's dtype -> FP32 logits [T, NL], rstd [T]. `rstd` given
    (from the write that produced xf, `write_fwd(rstd_eps=...)`) skips its kernel."""
    T, K = xf.shape
    if rstd is None:
        rstd = torch.empty(T, device=xf.device, dtype=torch.float32)
        _rstd_kernel[(T,)](xf, rstd, K, eps, BLOCK=min(4096, triton.next_power_of_2(K)), num_warps=8)
    logits = torch.mm(xf, w, out_dtype=torch.float32)
    return logits.mul_(rstd[:, None]), rstd


def proj_bwd_terms(rstd, logits, grad, K, dtype):
    """The per-token terms of the projection backward, dx = dP W^T - c x and dw = xf^T dP: dP [T, NL] in the streams'
    dtype (dlogits rstd, the tensor-core operand of both), c [T] FP32."""
    T, NL = logits.shape
    g = grad.reshape(T, NL).float()
    c = (g * logits).sum(-1) * rstd.square() / K  # (g . p) rstd^3 / K for the unscaled p = logits / rstd
    dp = (g * rstd[:, None]).to(dtype).contiguous()
    return dp, c


def logits_bwd(xf, w, rstd, logits, grad, need_dx=True, need_dw=True):
    """-> dx [T, K] in xf's dtype, dw [K, NL] FP32 (None when not needed), and the terms (dP, c) of `proj_bwd_terms`
    (the fused block adds its dP W^T - c X share of dX in `read_proj_bwd` instead of taking dx)."""
    T, K = xf.shape
    NL = w.shape[1]
    dp, c = proj_bwd_terms(rstd, logits, grad, K, xf.dtype)
    dw = torch.mm(xf.t(), dp, out_dtype=torch.float32) if need_dw else None
    dx = None
    if need_dx:
        dx = torch.empty_like(xf)
        BT, BK = 32, 128
        grid = (triton.cdiv(T, BT), triton.cdiv(K, BK))
        _proj_dx_kernel[grid](xf, dp, w.t().contiguous(), c, dx, T, K, NL, BT=BT, BK=BK,
                              LP=max(16, triton.next_power_of_2(NL)), num_warps=4)
    return dx, dw, dp, c


class _Logits(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, eps):
        K, NL = weight.shape
        xf = x.reshape(-1, K)
        w = weight.to(xf.dtype).contiguous()
        logits, rstd = logits_fwd(xf, w, eps)
        ctx.save_for_backward(xf, w, rstd, logits)
        ctx.x_shape, ctx.w_dtype = x.shape, weight.dtype
        return logits.view(*x.shape[:-1], NL)

    @staticmethod
    def backward(ctx, grad):
        xf, w, rstd, logits = ctx.saved_tensors
        dx, dw, _, _ = logits_bwd(xf, w, rstd, logits, grad, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        if dx is not None:
            dx = dx.view(ctx.x_shape)
        if dw is not None:
            dw = dw.to(ctx.w_dtype)
        return dx, dw, None


def hc_logits(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """[..., K] streams (flattened N * D, contiguous), weight [K, NL] -> FP32 logits [..., NL]."""
    return _Logits.apply(x.contiguous(), weight, eps)
