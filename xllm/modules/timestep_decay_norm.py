from typing import Optional, Tuple

import torch
import torch.nn as nn

from torch.autograd.function import FunctionCtx
from torch.nn.parameter import Parameter

from xllm_extension.ops import (
    group_timestep_decay_norm_fwd,
    group_timestep_decay_norm_cub_fwd,
    group_timestep_decay_norm_bwd,
    group_timestep_decay_norm_cub_bwd,
)

from xllm.distributed import get_model_parallel_world_size
from xllm.distributed.utils import divide_and_check_no_remainder


class TimestepDecayNormFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        bos_mask: Optional[torch.Tensor],
        prev_count: torch.Tensor,
        prev_mean: torch.Tensor,
        prev_var: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        num_groups: int,
        padding_mask: Optional[torch.Tensor] = None,
        beta1: float = 0.999,
        beta2: float = 0.9999,
        eps: float = 1e-5,
        memory_efficient: bool = False,
        backend: str = 'cub',
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if backend == 'cub':
            y, count, mean, var, cummean, cumrstd = group_timestep_decay_norm_cub_fwd(
                x, bos_mask, prev_count, prev_mean, prev_var, gamma, beta, padding_mask, num_groups, beta1, beta2, eps,
            )
            ctx.save_for_backward(x, bos_mask, prev_count, cummean, cumrstd, gamma, beta, padding_mask)
        else:
            y, count, mean, var, cummean, cumrstd = group_timestep_decay_norm_fwd(
                x, bos_mask, prev_count, prev_mean, prev_var, gamma, beta, padding_mask, num_groups, beta1, beta2, eps,
            )
            ctx.save_for_backward(x if not memory_efficient else y, bos_mask, prev_count, cummean, cumrstd, gamma, beta, padding_mask)
        ctx.num_groups = num_groups  # num_groups is not a torch.Tensor
        ctx.beta1 = beta1  # beta1 is not a torch.Tensor
        ctx.beta2 = beta2  # beta2 is not a torch.Tensor
        ctx.eps = eps  # eps is not a torch.Tensor
        ctx.memory_efficient = memory_efficient
        ctx.backend = backend
        return y, count, mean, var

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: torch.Tensor,
        _,
        mean_grad: torch.Tensor,
        var_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, None, None, None, None, None, None, None]:
        x_or_y, bos_mask, prev_count, cummean, cumrstd, gamma, beta, padding_mask = ctx.saved_tensors
        num_groups = ctx.num_groups
        beta1 = ctx.beta1
        beta2 = ctx.beta2
        eps = ctx.eps
        memory_efficient = ctx.memory_efficient
        backend = ctx.backend
        if backend == 'cub':
            x_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad = group_timestep_decay_norm_cub_bwd(
                y_grad, mean_grad, var_grad, x_or_y, prev_count, bos_mask, cummean, cumrstd, gamma,
                padding_mask, num_groups, beta1, beta2
            )
        else:
            x_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad = group_timestep_decay_norm_bwd(
                y_grad, mean_grad, var_grad, x_or_y, prev_count, bos_mask, cummean, cumrstd, gamma, beta,
                padding_mask, num_groups, beta1, beta2, eps, memory_efficient
            )
        return x_grad, None, None, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad, None, None, None, None, None, None, None


timestep_decay_norm = TimestepDecayNormFunc.apply


class TimestepDecayNorm(nn.Module):

    def __init__(
        self,
        num_features: int,
        num_groups: int,
        beta1: float = 0.999,
        beta2: float = 0.9999,
        eps: float = 1e-5,
        memory_efficient: bool = False,
        backend: str = 'cub'
    ) -> None:

        super().__init__()

        self.num_features = num_features
        self.num_groups = num_groups

        assert num_groups < num_features and num_features % num_groups == 0
        if backend == 'cub':
            assert not memory_efficient, 'timestep decay norm cub backend does not support memory efficient backward'

        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.features_per_partition = divide_and_check_no_remainder(num_features, world_size)
        self.groups_per_partition = divide_and_check_no_remainder(num_groups, world_size)

        self.register_buffer("prior_count", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("prior_mean", torch.zeros(self.groups_per_partition))
        self.register_buffer("prior_var", torch.zeros(self.groups_per_partition))

        self.register_parameter("weight", Parameter(torch.zeros(self.features_per_partition)))
        self.register_parameter("bias", Parameter(torch.zeros(self.features_per_partition)))

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.memory_efficient = memory_efficient
        self.backend = backend

    def forward(
        self,
        x: torch.Tensor,
        bos_mask: Optional[torch.Tensor] = None,
        prev_count: Optional[torch.Tensor] = None,
        prev_mean: Optional[torch.Tensor] = None,
        prev_var: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = x.size(0)
        if prev_count is None:
            prev_count = self.prior_count.expand(batch_size).contiguous()
        if prev_mean is None:
            prev_mean = self.prior_mean.type_as(x).expand(batch_size, -1).contiguous()
        if prev_var is None:
            prev_var = self.prior_var.type_as(x).expand(batch_size, -1).contiguous()

        output = timestep_decay_norm(
            x, bos_mask, prev_count, prev_mean, prev_var, self.weight + 1.0, self.bias,
            self.groups_per_partition, padding_mask, self.beta1, self.beta2,
            self.eps, self.memory_efficient, self.backend
        )
        return output

    def extra_repr(self) -> str:
        return 'num_features={num_features} ({features_per_partition}), ' \
               'num_groups={num_groups} ({groups_per_partition}), ' \
               'betas=({beta1}, {beta2}), eps={eps}, backend={backend}, ' \
               'memory_efficient={memory_efficient}'.format(**self.__dict__)
