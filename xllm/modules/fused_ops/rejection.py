from typing import Any, Tuple, Optional

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx


class RejectionFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        h: Tensor
    ) -> Tensor:
        y, a = rejection_fwd(x, h)
        ctx.save_for_backward(x, h, a)
        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        x, h, a = ctx.saved_tensors
        x_grad, h_grad = rejection_bwd(y_grad, x, h, a)
        return x_grad, h_grad


rejection = RejectionFunc.apply


def rejection_fwd(
    x: Tensor,
    h: Tensor
) -> Tuple[Tensor, Tensor]:
    dim = h.shape[-1]
    inv_scale = 1.0 / float(dim)
    # [B, *, S] x [B, *,  S] -> [B, *, 1]
    alpha = torch.sum(x * h, dim=-1, keepdim=True, dtype=torch.float32) * inv_scale  # fp32
    alpha = alpha.to(h.dtype)  # fp32 -> bf16
    y = torch.addcmul(x, alpha, h, value=-1.0)
    return y, alpha


def rejection_bwd(
    y_grad: Tensor,
    x: Tensor,
    h: Tensor,
    alpha: Tensor,
) -> Tuple[Tensor, Tensor]:
    dim = h.shape[-1]
    inv_scale = 1.0 / float(dim)
    # [B, *, S]
    p_grad = -y_grad
    # [B, *, 1]
    a_grad = torch.sum(p_grad * h, dim=-1, keepdim=True, dtype=torch.float32) * inv_scale  # fp32
    a_grad = a_grad.to(h.dtype)  # fp32 -> bf16
    # grad of h
    h_grad = torch.mul(p_grad, alpha, out=p_grad)
    h_grad = torch.addcmul(h_grad, a_grad, x, out=h_grad)
    # grad of x
    x_grad = torch.addcmul(y_grad, a_grad, h)

    return x_grad, h_grad
