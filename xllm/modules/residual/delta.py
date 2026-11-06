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
from xllm.modules.rms_norm import GroupRMSNorm


def _init_affine_weight(
    weight,
    n_heads: int,
    num_features: int,
    init_method: Callable[[Tensor], Tensor],
):
    assert weight.dim() == 2, f"weight dim is not 2, {weight.dim()}"
    rank = get_model_parallel_rank()
    world_size = get_model_parallel_world_size()
    features_per_partition = divide_and_check_no_remainder(num_features, world_size)

    # Initialize master weight
    master_weight = torch.empty(num_features, n_heads, dtype=weight.dtype, requires_grad=False)
    init_method(master_weight)

    weight_list = torch.split(master_weight, features_per_partition, dim=0)
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=0, out=weight)
    # clear master weights
    del master_weight
    del my_weight_list


def _init_alphabet(
    alphabet,
    n_heads: int
):
    world_size = get_model_parallel_world_size()
    rank = get_model_parallel_rank()
    n_local_heads = divide_and_check_no_remainder(n_heads, world_size)
    alphabet = alphabet.view(n_local_heads, 2)

    a = torch.log(torch.rand(n_heads))

    dt_min = 0.001
    dt_max = 0.1
    dt_init_floor = 1e-4
    dt = torch.exp(torch.rand(n_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
    dt = torch.clamp(dt, min=dt_init_floor)
    inv_dt = dt + torch.log(-torch.expm1(-dt))

    a = torch.split(a, n_local_heads)[rank]
    b = torch.split(inv_dt, n_local_heads)[rank]
    with torch.no_grad():
        alphabet[:, 0] = a
        alphabet[:, 1] = b


class DeltaResidual(nn.Module):
    """
    Delta Residual in:
    """
    def __init__(
        self,
        rc_id: int,
        model_dim: int,
        num_heads: int,
        num_features: int,
        eps: float,
        init_std: Optional[float] = None
    ):
        super().__init__()
        self.rc_id = rc_id

        model_parallel_world_size = get_model_parallel_world_size()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = divide_and_check_no_remainder(model_dim, num_heads)
        self.n_local_heads = divide_and_check_no_remainder(num_heads, model_parallel_world_size)
        self.num_features = num_features
        self.init_std = init_std

        features_per_partition = divide_and_check_no_remainder(num_features, model_parallel_world_size)
        init_fn = get_init_fn('gaussian', dim=num_features, std=init_std)
        self.weight = nn.Parameter(torch.empty(features_per_partition, self.num_heads))
        self.alphabet = nn.Parameter(torch.empty(self.n_local_heads * 2))
        _init_affine_weight(self.weight, self.num_heads, num_features, init_fn)
        _init_alphabet(self.alphabet, self.num_heads)

        self.norm = GroupRMSNorm(
            model_dim,
            num_groups=num_heads,
            elementwise_affine=False,
            eps=eps,
            memory_efficient=True,
        )

    def forward(
        self, h, residual, c, **kwargs
    ) -> Tensor:
        bsz, slen, _ = h.shape
        # N
        a, b = torch.unbind(self.alphabet.view(self.n_local_heads, 2).float(), dim=1)  # fp32
        # B*L x D/TP
        c = rearrange(c, 'b l d -> (b l) d')
        # (B*L x D/TP) x (D/TP x N) -> B*L x N
        beta = reduce_scatter_model_parallel_region(torch.mm(c, self.weight))
        # B*L x N
        beta = -torch.exp(a) * F.softplus(beta + b)
        beta = -torch.expm1(beta)

        # B*L x N x S
        k = rearrange(self.norm(h), 'b l (n s) -> (b l) n s', n=self.n_local_heads)
        h = rearrange(h, 'b l (n s) -> (b l) n s', n=self.n_local_heads)
        r = rearrange(residual, 'b l (n s) -> (b l) n s', n=self.n_local_heads)

        # compute dot(qk, r)
        inv_scale = 1.0 / float(self.head_dim)
        # (B*L x N x S) x (B*L x N x S) -> B*L x N
        # alpha = torch.linalg.vecdot(k, r, dim=-1) * inv_scale
        alpha = torch.sum(k * r, dim=-1, dtype=torch.float32) * inv_scale
        # B*L x N x 1
        alpha = (alpha * beta).to(h.dtype)[:, :, None]  # fp32 -> bf16

        # B*L x N x S
        out = torch.addcmul(h + r, alpha, k, value=-1.0)
        # B x L x D
        out = rearrange(out, '(b l) n s -> b l (n s)', b=bsz)
        return out

    def extra_repr(self) -> str:
        return 'dim={}, heads={} ({}), head_dim={}, features={}, init=gaussian ({})'.format(
            self.model_dim, self.num_heads, self.n_local_heads, self.head_dim, self.num_features, self.init_std
        )
