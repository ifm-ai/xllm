from typing import Callable, Optional

import torch
from torch.nn.parameter import Parameter

from xllm.distributed import (
    get_model_parallel_rank,
    get_model_parallel_world_size,
)
from xllm.distributed.utils import divide_and_check_no_remainder


def all_reduce(x: torch.Tensor, world_size: int, group: torch.distributed.ProcessGroup, async_op: bool = False):
    if world_size == 1:
        return x, None
    # All-reduce.
    handle = torch.distributed.all_reduce(x, group=group, async_op=async_op)
    return x, handle


def calc_probs(scores: torch.Tensor, score_func: str):
    # B x L x N
    if score_func == 'softmax':
        denom = None
        probs = scores
    elif score_func == 'sigmoid':
        denom = scores.sum(dim=-1, keepdim=True)
        probs = scores / denom
    else:
        raise ValueError(f"Unknown score function: {score_func}.")

    return probs, denom


class BaseRouter(torch.nn.Module):
    """Base router.
    Similar to RowParallelLinear, with specific communication control.

    Arguments:
        in_features: first dimension of matrix A.
        num_experts: second dimension of matrix A.
        topk: (average) number of activated experts per token.
    """

    routing_type: str

    def __init__(
        self,
        in_features: int,
        num_experts: int,
        topk: int,
        score_func: str = 'sigmoid',
        scaling_factor: Optional[float] = None,
    ) -> None:
        super(BaseRouter, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.num_experts = num_experts
        self.topk = topk
        self.score_func = score_func
        self.scaling_factor = scaling_factor

        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide_and_check_no_remainder(in_features, world_size)
        self.weight = Parameter(torch.empty(self.num_experts, self.input_size_per_partition))

    def init_weight(self, init_method: Callable[[torch.Tensor], torch.Tensor]) -> float:
        # Returns the std of the full weight.
        rank = get_model_parallel_rank()
        master_weight = torch.empty(self.num_experts, self.in_features, dtype=self.weight.dtype, requires_grad=False)
        init_method(master_weight)
        weight_std = master_weight.float().std(unbiased=False).item()
        weight_list = torch.split(master_weight, self.input_size_per_partition, dim=1)
        my_weight = weight_list[rank]
        with torch.no_grad():
            self.weight.copy_(my_weight)
        # clear master weight
        del master_weight
        del weight_list
        return weight_std
