from typing import Optional

import torch
import torch.nn as nn
from torch.autograd.function import FunctionCtx

from xllm_extension.ops import (
    group_layer_norm_fwd,
    group_layer_norm_bwd,
    group_layer_norm_fwd_affine,
    group_layer_norm_bwd_affine,
)
from xllm.modules.model_parallel import (
    gather_copy_model_parallel_region,
    gather_from_model_parallel_region
)
from xllm.distributed import get_model_parallel_world_size
from xllm.distributed.utils import divide_and_check_no_remainder


class GroupLayerNormFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        num_groups: int,
        eps: float = 1e-5,
        memory_efficient: bool = False,
    ):
        num_features = x.shape[-1]
        if weight is not None:
            y, mean, rstd = group_layer_norm_fwd_affine(x, num_features, num_groups, weight, bias, eps)
        else:
            y, mean, rstd = group_layer_norm_fwd(x, num_features, num_groups, eps)

        ctx.save_for_backward(y if memory_efficient else x, weight, bias, mean, rstd)
        ctx.num_features = num_features
        ctx.num_groups = num_groups
        ctx.memory_efficient = memory_efficient

        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: torch.Tensor,
    ):
        x_or_y, weight, bias, mean, rstd = ctx.saved_tensors
        num_features = ctx.num_features
        num_groups = ctx.num_groups
        memory_efficient = ctx.memory_efficient

        if weight is not None:
            x_grad, weight_grad, bias_grad = group_layer_norm_bwd_affine(
                y_grad, x_or_y, num_features, num_groups, mean, rstd, weight, bias, memory_efficient
            )
        else:
            x_grad = group_layer_norm_bwd(
                y_grad, x_or_y, num_features, num_groups, mean, rstd, memory_efficient
            )
            weight_grad, bias_grad = None, None

        return x_grad, weight_grad, bias_grad, None, None, None


group_layer_norm = GroupLayerNormFunc.apply


class GroupLayerNorm(nn.Module):
    def __init__(
        self,
        num_features,
        num_groups=1,
        eps=1e-5,
        elementwise_affine=True,
        memory_efficient=False,
        disable_input_reduce: bool = False,
    ):
        super().__init__()

        self.num_features = num_features
        self.num_groups = num_groups
        self.features_per_group = divide_and_check_no_remainder(num_features, num_groups)
        self.disable_input_reduce = disable_input_reduce

        world_size = get_model_parallel_world_size()
        self.features_per_partition = divide_and_check_no_remainder(num_features, world_size)
        self.gather_input = self.num_groups == 1 and world_size > 1
        if self.gather_input:
            self.groups_per_partition = num_groups
        else:
            self.groups_per_partition = divide_and_check_no_remainder(num_groups, world_size)

        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.memory_efficient = memory_efficient
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.zeros(self.features_per_partition))
            self.bias = nn.Parameter(torch.zeros(self.features_per_partition))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if self.gather_input:
            gather_fn = gather_from_model_parallel_region if self.disable_input_reduce else gather_copy_model_parallel_region
            x = gather_fn(x)
            weight = gather_fn(self.weight + 1.0) if self.elementwise_affine else None
            bias = gather_fn(self.bias) if self.elementwise_affine else None
        else:
            weight = self.weight + 1.0 if self.elementwise_affine else None
            bias = self.bias
        return group_layer_norm(x, weight, bias, self.groups_per_partition, self.eps, self.memory_efficient)

    def extra_repr(self):
        return "num_features={num_features} ({features_per_partition}, {features_per_group}), " \
               "num_groups={num_groups} ({groups_per_partition}), " \
               "eps={eps}, affine={elementwise_affine}, " \
               "memory_efficient={memory_efficient}".format(**self.__dict__)
