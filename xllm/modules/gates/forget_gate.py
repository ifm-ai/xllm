import math
from typing import Callable, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange

from xllm.distributed import (
    get_model_parallel_rank,
    get_model_parallel_world_size,
)
from xllm.distributed.utils import divide_and_check_no_remainder
from xllm.utils import get_init_fn
from xllm.modules.model_parallel import reduce_scatter_model_parallel_region


def _init_affine_weight(
    weight,
    in_features: int,
    out_features: int,
    init_method: Callable[[Tensor], Tensor],
):
    assert weight.dim() == 2, f"weight dim is not 2, {weight.dim()}"
    rank = get_model_parallel_rank()
    world_size = get_model_parallel_world_size()
    input_size_per_partition = divide_and_check_no_remainder(in_features, world_size)

    # Initialize master weight
    master_weight = torch.empty(in_features, out_features, dtype=weight.dtype, requires_grad=False)
    init_method(master_weight)

    weight_list = torch.split(master_weight, input_size_per_partition, dim=0)
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=0, out=weight)
    # clear master weights
    del master_weight
    del my_weight_list


def _init_alphabet(
    alphabet,
    out_features: int
):
    world_size = get_model_parallel_world_size()
    rank = get_model_parallel_rank()
    output_size_per_partition = divide_and_check_no_remainder(out_features, world_size)
    alphabet = alphabet.view(output_size_per_partition, 2)

    a = torch.log(torch.rand(out_features))

    dt_min = 0.001
    dt_max = 0.1
    dt_init_floor = 1e-4
    dt = torch.exp(torch.rand(out_features) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
    dt = torch.clamp(dt, min=dt_init_floor)
    inv_dt = dt + torch.log(-torch.expm1(-dt))

    a = torch.split(a, output_size_per_partition)[rank]
    b = torch.split(inv_dt, output_size_per_partition)[rank]
    with torch.no_grad():
        alphabet[:, 0] = a
        alphabet[:, 1] = b


class ForgetGate(nn.Module):
    """
    Forget gate in gated AWM
    """

    def __init__(
        self,
        layer_id: int,
        in_features: int,
        out_features: int,
        init_std: Optional[float] = None
    ):
        super().__init__()
        self.layer_id = layer_id
        self.in_features = in_features
        self.out_features = out_features
        self.init_std = init_std

        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide_and_check_no_remainder(in_features, world_size)
        self.output_size_per_partition = divide_and_check_no_remainder(out_features, world_size)

        init_fn = get_init_fn('gaussian', dim=in_features, std=init_std)
        self.weight = nn.Parameter(torch.empty(self.input_size_per_partition, self.out_features))
        self.alphabet = nn.Parameter(torch.empty(self.output_size_per_partition * 2))
        _init_affine_weight(self.weight, in_features, out_features, init_fn)
        _init_alphabet(self.alphabet, out_features)

