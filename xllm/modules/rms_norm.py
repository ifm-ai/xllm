from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd.function import FunctionCtx

from xllm_extension.ops import (
    group_rms_norm_fwd,
    group_rms_norm_bwd,
    group_rms_norm_fwd_affine,
    group_rms_norm_bwd_affine,
)
from xllm.modules.model_parallel import (
    gather_copy_model_parallel_region,
    gather_from_model_parallel_region
)
from xllm.distributed import get_model_parallel_group, get_model_parallel_world_size
from xllm.distributed.utils import divide_and_check_no_remainder


def partitioned_rms_norm_fwd(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    num_features: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm over num_features features split across the model-parallel group.

    x holds this rank's slice of the features (B x L x D/TP) and weight the matching slice. Each rank sums the
    squares of its features, one all-reduce gives every rank the per-token total, and each rank normalises only
    its own features: the result equals the gathered norm's slice, without gathering. Returns y and the fp32
    per-token rstd.
    """
    xf = x.float()
    sumsq = xf.square().sum(dim=-1)
    torch.distributed.all_reduce(sumsq, group=get_model_parallel_group())
    rstd = torch.rsqrt(sumsq / num_features + eps)
    y = xf * rstd.unsqueeze(-1)
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype), rstd


def partitioned_rms_norm_bwd(
    y_grad: torch.Tensor,
    x: torch.Tensor,
    rstd: torch.Tensor,
    weight: Optional[torch.Tensor],
    num_features: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Backward of partitioned_rms_norm_fwd: one all-reduce of a per-token dot product over the group."""
    rstd = rstd.unsqueeze(-1)
    x_hat = x.float() * rstd
    g = y_grad.float()
    weight_grad = None
    if weight is not None:
        weight_grad = (g * x_hat).flatten(end_dim=-2).sum(dim=0).to(weight.dtype)
        g = g * weight.float()
    dot = (g * x_hat).sum(dim=-1)
    torch.distributed.all_reduce(dot, group=get_model_parallel_group())
    x_grad = (g - x_hat * (dot / num_features).unsqueeze(-1)) * rstd
    return x_grad.to(x.dtype), weight_grad


class PartitionedRMSNormFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        num_features: int,
        eps: float,
    ):
        y, rstd = partitioned_rms_norm_fwd(x, weight, num_features, eps)
        ctx.save_for_backward(x, weight, rstd)
        ctx.num_features = num_features
        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: torch.Tensor,
    ):
        x, weight, rstd = ctx.saved_tensors
        x_grad, weight_grad = partitioned_rms_norm_bwd(y_grad, x, rstd, weight, ctx.num_features)
        return x_grad, weight_grad, None, None


partitioned_rms_norm = PartitionedRMSNormFunc.apply


class GroupRMSNormFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        num_groups: int,
        eps: float = 1e-6,
        memory_efficient: bool = False,
    ):
        num_features = x.shape[-1]
        if weight is not None:
            y, rstd = group_rms_norm_fwd_affine(x, num_features, num_groups, weight, eps)
        else:
            y, rstd = group_rms_norm_fwd(x, num_features, num_groups, eps)

        ctx.save_for_backward(y if memory_efficient else x, weight, rstd)
        ctx.num_features = num_features
        ctx.num_groups = num_groups
        ctx.memory_efficient = memory_efficient

        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: torch.Tensor,
    ):
        x_or_y, weight, rstd = ctx.saved_tensors
        num_features = ctx.num_features
        num_groups = ctx.num_groups
        memory_efficient = ctx.memory_efficient

        if weight is not None:
            x_grad, weight_grad = group_rms_norm_bwd_affine(
                y_grad, x_or_y, num_features, num_groups, rstd, weight, memory_efficient
            )
        else:
            x_grad = group_rms_norm_bwd(
                y_grad, x_or_y, num_features, num_groups, rstd, memory_efficient
            )
            weight_grad = None

        return x_grad, weight_grad, None, None, None


group_rms_norm = GroupRMSNormFunc.apply


class GroupRMSNorm(nn.Module):
    def __init__(
        self,
        num_features,
        num_groups=1,
        eps=1e-6,
        elementwise_affine=True,
        memory_efficient=False,
        disable_input_reduce: bool = False,
        partitioned_output: bool = False,
    ):
        super().__init__()

        self.num_features = num_features
        self.num_groups = num_groups
        self.features_per_group = divide_and_check_no_remainder(num_features, num_groups)
        self.disable_input_reduce = disable_input_reduce

        world_size = get_model_parallel_world_size()
        self.features_per_partition = divide_and_check_no_remainder(num_features, world_size)
        # partitioned_output: where one group spans the model-parallel ranks, normalise each rank's slice with a
        # per-token all-reduce instead of gathering, and return the slice. The caller must then pass each rank
        # the full gradient of its own slice (as with disable_input_reduce); memory_efficient does not apply.
        self.partitioned_output = partitioned_output and self.num_groups == 1 and world_size > 1
        assert not self.partitioned_output or disable_input_reduce, "partitioned_output needs disable_input_reduce"
        self.gather_input = self.num_groups == 1 and world_size > 1 and not self.partitioned_output
        if self.gather_input or self.partitioned_output:
            self.groups_per_partition = num_groups
        else:
            self.groups_per_partition = divide_and_check_no_remainder(num_groups, world_size)

        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.memory_efficient = memory_efficient
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.zeros(self.features_per_partition))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def forward(self, x):
        if self.partitioned_output:
            weight = self.weight + 1.0 if self.elementwise_affine else None
            return partitioned_rms_norm(x, weight, self.num_features, self.eps)
        if self.gather_input:
            gather_fn = gather_from_model_parallel_region if self.disable_input_reduce else gather_copy_model_parallel_region
            x = gather_fn(x)
            weight = gather_fn(self.weight + 1.0) if self.elementwise_affine else None
        else:
            weight = self.weight + 1.0 if self.elementwise_affine else None
        return group_rms_norm(x, weight, self.groups_per_partition, self.eps, self.memory_efficient)

    def extra_repr(self):
        return "num_features={num_features} ({features_per_partition}, {features_per_group}), " \
               "num_groups={num_groups} ({groups_per_partition}), " \
               "eps={eps}, affine={elementwise_affine}, " \
               "memory_efficient={memory_efficient}, " \
               "partitioned_output={partitioned_output}".format(**self.__dict__)
