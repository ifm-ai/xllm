from typing import Optional, Tuple

import torch
import torch.nn as nn

from torch.autograd.function import FunctionCtx
from torch.nn.parameter import Parameter

from xllm_extension.ops import (
    sequence_norm_fwd,
    group_sequence_norm_fwd,
    sequence_norm_bwd,
    group_sequence_norm_bwd
)


class SequenceNormFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        num_groups: Optional[int] = None,
        eps: float = 1e-5,
        length_last: bool = False
    ) -> torch.Tensor:
        if num_groups is None:
            y, count, mean, rstd = sequence_norm_fwd(
                x, gamma, beta, padding_mask, eps, length_last
            )
        else:
            y, count, mean, rstd = group_sequence_norm_fwd(
                x, gamma, beta, padding_mask, num_groups, eps, length_last
            )
        ctx.save_for_backward(x, count, mean, rstd, gamma, padding_mask)
        ctx.num_groups = num_groups  # num_groups is not a torch.Tensor
        ctx.length_last = length_last  # length_last is not a torch.Tensor
        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        x, count, mean, rstd, gamma, padding_mask = ctx.saved_tensors
        num_groups = ctx.num_groups
        length_last = ctx.length_last
        if num_groups is None:
            x_grad, gamma_grad, beta_grad = sequence_norm_bwd(
                y_grad, x, count, mean, rstd, gamma, padding_mask, length_last
            )
        else:
            x_grad, gamma_grad, beta_grad = group_sequence_norm_bwd(
                y_grad, x, count, mean, rstd, gamma, padding_mask, num_groups, length_last
            )
        return x_grad, gamma_grad, beta_grad, None, None, None, None


sequence_norm = SequenceNormFunc.apply


class SequenceNorm(nn.Module):

    def __init__(
        self,
        num_features: int,
        num_groups: Optional[int] = None,
        eps: float = 1e-5
    ) -> None:
        super().__init__()

        assert num_groups is None or num_features % num_groups == 0

        self.num_features = num_features
        self.num_groups = num_groups
        self.register_parameter("weight", Parameter(torch.zeros(num_features), requires_grad=True))
        self.register_parameter("bias", Parameter(torch.zeros(num_features), requires_grad=True))
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        length_last: bool = False
    ) -> torch.Tensor:
        return sequence_norm(x, self.weight + 1.0, self.bias, padding_mask, self.num_groups, self.eps, length_last)
