import torch

from .triton.triton_causal_conv1d_fwd import kernel_fn as kernel_fwd_fn
from .triton.triton_causal_conv1d_bwd import kernel_fn as kernel_bwd_fn


def _triton_causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    bos_mask: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, None]:
    """xllm API wrapper for the Triton forward kernel.

    The Triton kernel uses transposed weight and state layouts:
    weight is (width, dim) and state is (batch, dim, width - 1).
    xllm exposes weight as (dim, width) and state as (batch, width - 1, dim).
    """
    if activation == "swish":
        activation = "silu"

    triton_weight = weight.transpose(0, 1).contiguous()
    triton_initial_state = (
        initial_state.transpose(1, 2).contiguous()
        if initial_state is not None
        else None
    )

    out, final_state = kernel_fwd_fn(
        x,
        triton_weight,
        bias,
        triton_initial_state,
        bos_mask,
        activation,
        deterministic=deterministic,
    )

    if output_final_state:
        return out, final_state.transpose(1, 2).contiguous(), None
    return out, None, None


_ZEROS = {}


def _zeros(shape, dtype, device):
    """Cached zero tensors for the kernels' optional read-only inputs (the training path has no bias / state)."""
    key = (shape, dtype, device)
    z = _ZEROS.get(key)
    if z is None:
        z = _ZEROS[key] = torch.zeros(shape, dtype=dtype, device=device)
    return z


def _triton_causal_conv1d_bwd(
    out_grad: torch.Tensor,
    final_state_grad: torch.Tensor | None,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    bos_mask: torch.Tensor | None = None,
    activation: str | None = None,
    deterministic: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Adapt xllm's public tensor layouts to the optimized kernels."""
    if activation == "swish":
        activation = "silu"

    triton_initial_state = (
        initial_state.transpose(1, 2).contiguous()
        if initial_state is not None
        else None
    )
    if final_state_grad is not None:
        triton_final_state_grad = final_state_grad.transpose(1, 2).contiguous()
    else:  # read-only zeros in the kernel layout [B, D, W - 1], reused across calls
        triton_final_state_grad = _zeros((x.shape[0], x.shape[2], weight.shape[1] - 1), x.dtype, x.device)
    effective_bias = bias.contiguous() if bias is not None else _zeros((x.shape[2],), x.dtype, x.device)

    x_grad, weight_grad, bias_grad, triton_initial_state_grad = kernel_bwd_fn(
        x.contiguous(),
        weight.contiguous(),
        effective_bias,
        triton_initial_state,
        bos_mask.contiguous() if bos_mask is not None else None,
        activation,
        out_grad.contiguous(),
        triton_final_state_grad,
        deterministic=deterministic,
    )
    initial_state_grad = (
        triton_initial_state_grad.transpose(1, 2).contiguous()
        if triton_initial_state_grad is not None
        else None
    )
    return (
        x_grad,
        initial_state_grad,
        weight_grad,
        bias_grad if bias is not None else None,
    )
