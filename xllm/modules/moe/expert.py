from typing import Optional, Callable, List

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import rearrange

from xllm.distributed import (
    get_model_parallel_rank,
    get_model_parallel_world_size,
)
from xllm.distributed.utils import divide_and_check_no_remainder
from xllm.utils import get_init_fn
from xllm.modules.fused_ops import memory_efficient_dropout
from xllm.modules.fused_ops import mgmm


def _init_affine_weight(
    weight,
    n_experts: int,
    out_features: int,
    in_features: int,
    init_method: Callable[[torch.Tensor], torch.Tensor],
):
    assert weight.dim() == 2, f"weight dim is not 2, {weight.dim()}"
    rank = get_model_parallel_rank()
    world_size = get_model_parallel_world_size()
    n_local_experts = divide_and_check_no_remainder(n_experts, world_size)
    expert_start_idx = rank * n_local_experts
    expert_end_idx = (rank + 1) * n_local_experts

    my_weight_list = []
    for i in range(n_experts):
        w = torch.empty(out_features, in_features, dtype=weight.dtype, requires_grad=False)
        init_method(w)
        if expert_start_idx <= i < expert_end_idx:
            my_weight_list.append(w)
        else:
            del w

    with torch.no_grad():
        assert len(my_weight_list) == n_local_experts
        torch.cat(my_weight_list, dim=0, out=weight)
    # clear master weights
    del my_weight_list


class Expert(nn.Module):
    def __init__(
        self,
        model_dim: int,
        expert_inter_dim: int,
        num_experts: int,
        backend: str = 'sequential',
        hidden_dropout: float = 0.0,
        init_mode: str = 'gaussian',
        init_std: Optional[float] = None
    ):
        super().__init__()

        self.model_dim = model_dim
        self.inter_dim = expert_inter_dim
        self.n_experts = num_experts
        self.backend = backend
        self.hidden_dropout = hidden_dropout
        self.init_mode = init_mode
        self.init_std = init_std
        world_size = get_model_parallel_world_size()
        self.n_local_experts = divide_and_check_no_remainder(num_experts, world_size)

        self.weight1 = Parameter(torch.empty(self.n_local_experts * self.inter_dim, self.model_dim))
        self.weight2 = Parameter(torch.empty(self.n_local_experts * self.model_dim, self.inter_dim))
        self.weight3 = Parameter(torch.empty(self.n_local_experts * self.inter_dim, self.model_dim))

        # init w1 & w3
        init_fn = get_init_fn(init_mode, dim=model_dim, std=init_std)
        _init_affine_weight(self.weight1, num_experts, self.inter_dim, self.model_dim, init_fn)
        _init_affine_weight(self.weight3, num_experts, self.inter_dim, self.model_dim, init_fn)
        # init w2
        init_fn = get_init_fn(init_mode, dim=expert_inter_dim, std=init_std)
        _init_affine_weight(self.weight2, num_experts, self.model_dim, self.inter_dim, init_fn)

    def forward(self, x: torch.Tensor, group_sizes: List[int]) -> torch.Tensor:
        w1 = rearrange(self.weight1, '(n s) d -> n s d', n=self.n_local_experts)
        w2 = rearrange(self.weight2, '(n s) d -> n s d', n=self.n_local_experts)
        w3 = rearrange(self.weight3, '(n s) d -> n s d', n=self.n_local_experts)
        h1 = F.silu(mgmm(x, w1, group_sizes, True, self.backend))
        h3 = mgmm(x, w3, group_sizes, True, self.backend)
        hidden = memory_efficient_dropout(h1 * h3, self.hidden_dropout, self.training)
        y = mgmm(hidden, w2, group_sizes, True, self.backend)
        return y

    def extra_repr(self) -> str:
        return 'mdim={}, edim={}, expert={} ({}), backend={}, init={} ({})'.format(
            self.model_dim, self.inter_dim, self.n_experts, self.n_local_experts, self.backend, self.init_mode, self.init_std
        )
