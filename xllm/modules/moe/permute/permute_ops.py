"""Fused inverse-perm + weighted reduction Triton kernel for MoE output.

Fuses inverse-perm lookup + weighted reduction into one kernel,
replacing argsort + index_select + mul + sum in the MoE forward pass.
Includes a fused backward pass for autograd support.
"""

from typing import Tuple

import torch
from torch import Tensor
import triton

from .permute_kernels import (
    _fused_y_perm_fwd_kernel,
    _y_grad_bwd_kernel,
    _scores_grad_bwd_kernel,
    _fused_y_perm_bwd_kernel
)


def _compute_inverse_indices(sorted_indices):
    inverse_indices = torch.empty_like(sorted_indices)
    inverse_indices[sorted_indices] = torch.arange(
        sorted_indices.shape[0], device=sorted_indices.device
    )
    return inverse_indices


def fused_y_perm_fwd(
    sorted_indices: Tensor,
    y: Tensor,
    routing_scores: Tensor,
    bsz: int,
    slen: int,
    topk: int
) -> Tuple[Tensor, Tensor]:
    """
    Forward pass of fused inverse-perm + weighted reduction for MoE output.
    Args:
        sorted_indices (LongTensor):
            sorted indexing tensor (shape [bsz*slen*topk])
        y (Tensor):
            y tensor to permute (shape [bsz*slen*topk, dim])
        routing_scores (Tensor):
            tensor for routing scores (shape [bsz, slen, topk])
        bsz (int): batch size
        slen (int): sequence length
        topk (int): routing topk

    Returns:
        output (Tensor):
            permuted and aggregated y (shape [bsz, slen, dim])
        inverse_indices (LongTensor):
            inverse indexing tensor (shape [bsz*slen*topk])
    """
    D = y.shape[1]
    output = torch.empty(bsz * slen, D, device=y.device, dtype=y.dtype)

    inverse_indices = _compute_inverse_indices(sorted_indices)
    rs_flat = routing_scores.reshape(-1)

    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    grid = (bsz * slen, triton.cdiv(D, BLOCK_D))

    _fused_y_perm_fwd_kernel[grid](
        output, y, inverse_indices, rs_flat,
        K=topk, D=D, BLOCK_D=BLOCK_D,
    )

    return output.view(bsz, slen, D), inverse_indices


def fused_y_perm_bwd_y_grad(
    grad_out: Tensor,
    inverse_indices: Tensor,
    routing_scores: Tensor,
    bsz: int,
    slen: int,
    topk: int
) -> Tensor:
    """
    Backward pass of fused inverse-perm + weighted reduction for MoE output.
    Args:
        grad_out (Tensor):
            grad tensor of output (shape [bsz, slen, dim])
        inverse_indices (LongTensor):
            inverse indexing tensor (shape [bsz*slen*topk])
        routing_scores (Tensor):
            tensor for routing scores (shape [bsz, slen, topk])
        bsz (int): batch size
        slen (int): sequence length
        topk (int): routing topk

    Returns:
        grad_y (Tensor):
            grad tensor of y (shape [bsz*slen*topk, dim])
    """
    D = grad_out.shape[2]

    grad_output = grad_out.reshape(bsz * slen, D).contiguous()
    rs_flat = routing_scores.reshape(-1)

    grad_y = torch.empty(bsz * slen * topk, D, device=grad_out.device, dtype=grad_out.dtype)

    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    grid = (bsz * slen, triton.cdiv(D, BLOCK_D))

    _y_grad_bwd_kernel[grid](
        grad_y, grad_output,
        inverse_indices, rs_flat,
        K=topk, D=D, BLOCK_D=BLOCK_D,
    )

    return grad_y


def fused_y_perm_bwd_scores_grad(
    grad_out: Tensor,
    inverse_indices: Tensor,
    y: Tensor,
    bsz: int,
    slen: int,
    topk: int
) -> Tensor:
    """
    Backward pass of fused inverse-perm + weighted reduction for MoE output.
    Args:
        grad_out (Tensor):
            grad tensor of output (shape [bsz, slen, dim])
        inverse_indices (LongTensor):
            inverse indexing tensor (shape [bsz*slen*topk])
        y (Tensor):
            y tensor to permute (shape [bsz*slen*topk, dim])
        bsz (int): batch size
        slen (int): sequence length
        topk (int): routing topk

    Returns:
        grad_y (Tensor):
            grad tensor of y (shape [bsz*slen*topk, dim])
        grad_scores (LongTensor):
            grad tensor of routing scores (shape [bsz, slen, topk])

    """
    D = y.shape[1]

    grad_output = grad_out.reshape(bsz * slen, D).contiguous()

    grad_scores = torch.zeros(
        bsz * slen * topk, device=y.device, dtype=torch.float32
    )

    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    grid = (bsz * slen, triton.cdiv(D, BLOCK_D))

    _scores_grad_bwd_kernel[grid](
        grad_scores, grad_output,
        y, inverse_indices,
        K=topk, D=D, BLOCK_D=BLOCK_D,
    )

    grad_scores = grad_scores.to(grad_output.dtype).view(bsz, slen, topk)

    return grad_scores


def fused_y_perm_bwd(
    grad_out: Tensor,
    inverse_indices: Tensor,
    y: Tensor,
    routing_scores: Tensor,
    bsz: int,
    slen: int,
    topk: int
) -> Tuple[Tensor, Tensor]:
    """
    Backward pass of fused inverse-perm + weighted reduction for MoE output.
    Args:
        grad_out (Tensor):
            grad tensor of output (shape [bsz, slen, dim])
        inverse_indices (LongTensor):
            inverse indexing tensor (shape [bsz*slen*topk])
        y (Tensor):
            y tensor to permute (shape [bsz*slen*topk, dim])
        routing_scores (Tensor):
            tensor for routing scores (shape [bsz, slen, topk])
        bsz (int): batch size
        slen (int): sequence length
        topk (int): routing topk

    Returns:
        grad_y (Tensor):
            grad tensor of y (shape [bsz*slen*topk, dim])
        grad_scores (LongTensor):
            grad tensor of routing scores (shape [bsz, slen, topk])

    """
    D = y.shape[1]

    grad_output = grad_out.reshape(bsz * slen, D).contiguous()
    rs_flat = routing_scores.reshape(-1)

    grad_y = torch.empty_like(y)
    grad_scores = torch.zeros(
        bsz * slen * topk, device=y.device, dtype=torch.float32
    )

    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    grid = (bsz * slen, triton.cdiv(D, BLOCK_D))

    _fused_y_perm_bwd_kernel[grid](
        grad_y, grad_scores, grad_output,
        y, inverse_indices, rs_flat,
        K=topk, D=D, BLOCK_D=BLOCK_D,
    )

    grad_scores = grad_scores.to(grad_output.dtype).view(bsz, slen, topk)

    return grad_y, grad_scores


class FusedYPerm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sorted_indices, y, routing_scores, bsz, slen, topk):
        output, inverse_indices = fused_y_perm_fwd(sorted_indices, y, routing_scores, bsz, slen, topk)

        ctx.save_for_backward(inverse_indices, y, routing_scores)
        ctx.bsz = bsz
        ctx.slen = slen
        ctx.topk = topk

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inverse_indices, y, routing_scores = ctx.saved_tensors
        bsz, slen, topk = ctx.bsz, ctx.slen, ctx.topk
        grad_y, grad_scores = fused_y_perm_bwd(grad_output, inverse_indices, y, routing_scores, bsz, slen, topk)

        # sorted_indices, bsz, slen, topk are not differentiable
        return None, grad_y, grad_scores, None, None, None


def fused_permute_y(
    sorted_indices: Tensor,
    y: Tensor,
    routing_scores: Tensor,
    bsz: int,
    slen: int,
    topk: int,
    backend: str
) -> Tensor:
    """
    Forward pass of fused inverse-perm + weighted reduction for MoE output.
    Args:
        sorted_indices (LongTensor):
            sorted indexing tensor (shape [bsz*slen*topk])
        y (Tensor):
            y tensor to permute (shape [bsz*slen*topk, dim])
        routing_scores (Tensor):
            tensor for routing scores (shape [bsz, slen, topk])
        bsz (int): batch size
        slen (int): sequence length
        topk (int): routing topk
        backend (str): the backend implementation

    Returns:
        output (Tensor):
            permuted and aggregated y (shape [bsz, slen, dim])
    """

    if backend == 'torch':
        # re-order
        inverse_indices = torch.argsort(sorted_indices, stable=True)
        # B*L*K x D/TP
        y = torch.index_select(y, 0, inverse_indices)
        # B x L x D/TP
        return (y.view(bsz, slen, topk, -1) * routing_scores.unsqueeze(3)).sum(dim=2)
    elif backend == 'triton':
        # fused inverse-perm + weighted reduction
        return FusedYPerm.apply(sorted_indices, y, routing_scores, bsz, slen, topk)
    else:
        raise ValueError(f"Unknown backend: {backend}.")
