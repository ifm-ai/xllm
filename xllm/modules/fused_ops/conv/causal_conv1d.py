from typing import Any, Tuple, Optional

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

from .fla_causal_conv1d import (
    _fla_causal_conv1d_fwd,
    _fla_causal_conv1d_bwd
)
from .triton_causal_conv1d import (
    _triton_causal_conv1d_fwd,
    _triton_causal_conv1d_bwd
)


class CausalConv1dFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        initial_state: Optional[Tensor] = None,
        bos_mask: Optional[Tensor] = None,
        output_final_state: bool = False,
        activation: Optional[str] = None,
        backend: str = 'triton',
        deterministic: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        y, final_state, _ = _causal_conv1d_fwd(
            x, weight, bias, initial_state, bos_mask, output_final_state,
            activation, backend, deterministic
        )
        ctx.save_for_backward(x, weight, bias, initial_state, bos_mask)
        ctx.activation = activation
        ctx.backend = backend
        ctx.deterministic = deterministic

        return y, final_state

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        out_grad: Tensor,
        final_state_grad: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor],
               None, None, None, None, None]:
        x, weight, bias, initial_state, bos_mask = ctx.saved_tensors
        activation = ctx.activation
        backend = ctx.backend
        deterministic = ctx.deterministic
        x_grad, initial_state_grad, w_grad, b_grad = _causal_conv1d_bwd(
            out_grad, final_state_grad, x, weight, bias, initial_state, bos_mask,
            activation, backend, deterministic
        )

        return x_grad, w_grad, b_grad, initial_state_grad, None, None, None, None, None


def causal_conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    initial_state: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    output_final_state: bool = False,
    activation: Optional[str] = None,
    backend: str = 'triton',
    deterministic: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Causal convolution with optional deterministic Triton gradients.

    ``deterministic=False`` permits atomic parameter-gradient accumulation on
    Triton unless PyTorch's global deterministic setting is enabled. The flag
    does not configure the external FLA backend. True also selects deterministic
    convolution for the forward and backward reference fallbacks.
    """
    return CausalConv1dFunc.apply(
        x, weight, bias, initial_state, bos_mask, output_final_state,
        activation, backend, deterministic
    )


def _causal_conv1d_fwd(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    initial_state: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    output_final_state: bool = False,
    activation: Optional[str] = None,
    backend: str = 'triton',
    deterministic: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Any]:
    if backend == 'fla':
        return _fla_causal_conv1d_fwd(
            x, weight, bias, initial_state, bos_mask, output_final_state, activation
        )
    elif backend == 'triton':
        return _triton_causal_conv1d_fwd(
            x, weight, bias, initial_state, bos_mask, output_final_state, activation, deterministic
        )
    else:
        raise ValueError(f"Unknown backend: {backend}.")


def _causal_conv1d_bwd(
    out_grad: Tensor,
    final_state_grad: Optional[Tensor],
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    initial_state: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    activation: Optional[str] = None,
    backend: str = 'triton',
    deterministic: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
    if backend == 'fla':
        return _fla_causal_conv1d_bwd(
            out_grad, final_state_grad, x, weight, bias, initial_state, bos_mask, activation
        )
    elif backend == 'triton':
        return _triton_causal_conv1d_bwd(
            out_grad, final_state_grad, x, weight, bias, initial_state, bos_mask, activation, deterministic
        )
    else:
        raise ValueError(f"Unknown backend: {backend}.")


def causal_conv1d_fwd(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    initial_state: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    output_final_state: bool = False,
    activation: Optional[str] = None,
    backend: str = 'triton',
    deterministic: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Any]:
    """
    Args:
        x (Tensor): (batch, seqlen, dim)
        weight (Tensor): (dim, width)
        bias (Optional[Tensor]): (dim)
        initial_state (Optional[Tensor]): (batch, width - 1, dim)
        bos_mask (Optional[Tensor]): (batch, seqlen)
        output_final_state (bool):
            whether to output the final state of shape [batch, width - 1, dim]. Default: `False`.
        activation (Optional[str]):
            Activations applied to output, only `swish`/`silu` or `None` (i.e., no activation) are supported.
            Default: `None`.
        backend (str): backend implementation
        deterministic (bool): Select deterministic convolution in the Triton
            reference fallback. Defaults to False. The optimized forward kernels
            are deterministic in either mode. Does not configure FLA.

    Return:
        out (Tensor): (batch, seqlen, dim)
        final_states (Optional[Tensor]): (batch, width - 1, dim)
        *
    """
    return _causal_conv1d_fwd(
        x, weight, bias, initial_state, bos_mask, output_final_state,
        activation, backend, deterministic
    )


def causal_conv1d_bwd(
    out_grad: Tensor,
    final_state_grad: Optional[Tensor],
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    initial_state: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    activation: Optional[str] = None,
    backend: str = 'triton',
    deterministic: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
    """
    Args:
        out_grad (Tensor): (batch, seqlen, dim)
        final_state_grad (Optional[Tensor]): (batch, width - 1, dim)
        x (Tensor): (batch, seqlen, dim)
        weight (Tensor): (dim, width)
        bias (Optional[Tensor]): (dim)
        initial_state (Optional[Tensor]): (batch, width - 1, dim)
        bos_mask (Optional[Tensor]): (batch, seqlen)
        activation (Optional[str]):
            Activations applied to output, only `swish`/`silu` or `None` (i.e., no activation) are supported.
            Default: `None`.
        backend (str): backend implementation
        deterministic (bool): Use fixed-order Triton parameter-gradient reductions.
            Defaults to False. False permits atomics unless PyTorch's global
            deterministic setting is enabled. Does not configure FLA.

    Return:
        x_grad (Tensor): (batch, seqlen, dim)
        initial_state_grad (Optional[Tensor]): (batch, width - 1, dim)
        weight_grad (Tensor): (dim, width)
        bias_grad (Optional[Tensor]): (dim)
    """
    return _causal_conv1d_bwd(
        out_grad, final_state_grad, x, weight, bias, initial_state, bos_mask,
        activation, backend, deterministic
    )
