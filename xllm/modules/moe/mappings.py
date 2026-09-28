# coding=utf-8

# Most parts of the code here are adapted from NVIDIA/Megatron-LLM
# repo: https://github.com/NVIDIA/Megatron-LM

from typing import Any, Optional, List

import torch

from xllm.distributed import (
    get_model_parallel_group,
    get_model_parallel_world_size,
)


def _all_to_all(
    input_: torch.Tensor,
    input_split_sizes: Optional[List[int]] = None,
    output_split_sizes: Optional[List[int]] = None,
) -> torch.Tensor:
    """Split input tensor and then scatter the split list to all processes."""
    group = get_model_parallel_group()
    world_size = get_model_parallel_world_size()

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    assert (input_split_sizes is None) == (output_split_sizes is None)
    out_shape = input_.shape if output_split_sizes is None else [sum(output_split_sizes)] + list(input_.shape[1:])
    output = torch.empty(out_shape, device=input_.device, dtype=input_.dtype)
    torch.distributed.all_to_all_single(output, input_, output_split_sizes, input_split_sizes, group=group)

    return output


class _AlltoAllExpertParallelRegion(torch.autograd.Function):
    """ALl to ALL in the expert parallel region."""

    @staticmethod
    def forward(ctx, input_, input_split_sizes, output_split_sizes):  # type: ignore
        ctx.input_split_sizes = input_split_sizes
        ctx.output_split_sizes = output_split_sizes
        return _all_to_all(input_, input_split_sizes, output_split_sizes)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        output_split_sizes = ctx.input_split_sizes
        input_split_sizes = ctx.output_split_sizes
        return _all_to_all(grad_output, input_split_sizes, output_split_sizes), None, None

# -----------------
# Helper functions.
# -----------------


def all_to_all_expert_parallel_region(
    input_: torch.Tensor, input_split_sizes: Optional[List[int]] = None, output_split_sizes: Optional[List[int]] = None
) -> torch.Tensor:
    return _AlltoAllExpertParallelRegion.apply(input_, input_split_sizes, output_split_sizes)
