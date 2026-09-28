# coding=utf-8

from typing import Any, Optional

import torch

from xllm.distributed import (
    get_context_parallel_group,
    get_context_parallel_world_size,
    get_context_parallel_rank,
    get_context_parallel_prev_rank,
    get_context_parallel_next_rank,
)
from xllm.distributed.utils import (
    split_tensor_along_specific_dim,
    reshape_gathered_tensor_along_specific_dim,
    reshape_scattered_tensor_along_specific_dim,
)


def should_send_to_next():
    return get_context_parallel_world_size() > 1 and get_context_parallel_next_rank() is not None


def should_recv_from_prev():
    return get_context_parallel_world_size() > 1 and get_context_parallel_prev_rank() is not None


def _send_tensor_to_next(tensor: torch.Tensor, async_op: bool = False):
    """send the input tensor to the next gpu in the context parallel group."""
    group = get_context_parallel_group()
    dst = get_context_parallel_next_rank()

    # Send to destination.
    if async_op:
        handle = torch.distributed.isend(tensor, dst, group=group)
    else:
        torch.distributed.send(tensor, dst, group=group)
        handle = None

    return tensor, handle


def _send_grad_to_prev(grad: torch.Tensor, async_op: bool = False):
    """send the input tensor to the next gpu in the context parallel group."""
    group = get_context_parallel_group()
    dst = get_context_parallel_prev_rank()

    # Send to destination.
    if async_op:
        handle = torch.distributed.isend(grad, dst, group=group)
    else:
        torch.distributed.send(grad, dst, group=group)
        handle = None

    return grad, handle


def _receive_grad_from_next(ctx: Any, grad: torch.Tensor, async_op: bool = False):
    """receive the input tensor from the previous gpu in the context parallel group."""
    group = get_context_parallel_group()
    src = get_context_parallel_next_rank()

    if ctx:
        ctx.mark_dirty(grad)

    # Recevie from source.
    # TODO: add original grad to recv grad
    if async_op:
        handle = torch.distributed.irecv(grad, src, group=group)
    else:
        torch.distributed.recv(grad, src, group=group)
        handle = None

    return grad, handle


def _receive_tensor_from_prev(ctx: Any, tensor: torch.Tensor, async_op: bool = False):
    """receive the input tensor from the previous gpu in the context parallel group."""
    group = get_context_parallel_group()
    src = get_context_parallel_prev_rank()

    if ctx:
        ctx.mark_dirty(tensor)

    # Recevie from source.
    if async_op:
        handle = torch.distributed.irecv(tensor, src, group=group)
    else:
        torch.distributed.recv(tensor, src, group=group)
        handle = None

    return tensor, handle


def _split(input_: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """Split the tensor along the specific dimension and keep the corresponding slice."""
    world_size = get_context_parallel_world_size()

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along specific dimension.
    split_dim = input_.dim() - 1 if dim is None else dim
    input_list = split_tensor_along_specific_dim(input_, world_size, split_dim=split_dim)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_context_parallel_rank()
    output = input_list[rank].contiguous()

    return output


def _gather(input_: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """Gather tensors and concatenate along the specific dimension."""
    group = get_context_parallel_group()
    world_size = get_context_parallel_world_size()

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # shape and dimension.
    gather_dim = input_.dim() - 1 if dim is None else dim
    output = torch.empty(world_size, *input_.shape, dtype=input_.dtype, device=input_.device)
    torch.distributed.all_gather_into_tensor(output, input_, group=group)
    output = reshape_gathered_tensor_along_specific_dim(output, gather_dim)

    return output


def _reduce_scatter(input_: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """Reduce-scatter the input tensor across model parallel group."""
    group = get_context_parallel_group()
    world_size = get_context_parallel_world_size()

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # reduce-scatter.
    scatter_dim = input_.dim() - 1 if dim is None else dim
    input_ = reshape_scattered_tensor_along_specific_dim(input_, world_size, scatter_dim)
    output = torch.empty(*input_.shape[1:], dtype=input_.dtype, device=input_.device)
    torch.distributed.reduce_scatter_tensor(output, input_, group=group)

    return output


class _SendToNextContextParallelRegion(torch.autograd.Function):
    """Send the input to the next context parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        out, handle = _send_tensor_to_next(input_)
        assert handle is None
        return out

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore
        grad_inp, handle = _receive_grad_from_next(None, grad_out)
        assert handle is None
        return grad_inp


class _ReceiveFromPreviousContextParallelRegion(torch.autograd.Function):
    """Receive the input from the previous context parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        out, handle = _receive_tensor_from_prev(None, input_)
        assert handle is None
        return out

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore
        grad_inp, handle = _send_grad_to_prev(grad_out)
        assert handle is None
        return grad_inp


class _GatherFromContextParallelRegion(torch.autograd.Function):
    """Gather the input from context parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_, dim):  # type: ignore
        ctx.dim = dim
        return _gather(input_, dim)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        dim = ctx.dim
        return _split(grad_output, dim), None


class _GatherCopyContextParallelRegion(torch.autograd.Function):
    """Gather the input from context parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_, dim):  # type: ignore
        ctx.dim = dim
        return _gather(input_, dim)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        dim = ctx.dim
        return _reduce_scatter(grad_output, dim), None

# -----------------
# Helper functions.
# -----------------


def send_to_next_context_parallel_region(input_: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return _SendToNextContextParallelRegion.apply(input_)


def recv_from_prev_context_parallel_region(input_: torch.Tensor) -> Optional[torch.Tensor]:
    return _ReceiveFromPreviousContextParallelRegion.apply(input_)


def gather_from_context_parallel_region(input_: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    return _GatherFromContextParallelRegion.apply(input_, dim)


def gather_copy_context_parallel_region(input_: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    return _GatherCopyContextParallelRegion.apply(input_, dim)
