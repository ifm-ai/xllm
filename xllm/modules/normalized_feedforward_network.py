from typing import Optional

from torch import nn, Tensor
import torch.nn.functional as F

from xllm.modules.model_parallel import (
    RowParallelLinear,
    ColumnParallelLinear,
    gather_copy_model_parallel_region,
)
from xllm.modules.fused_ops import memory_efficient_dropout
from xllm.modules.layer_norm import GroupLayerNorm
from xllm.modules.rms_norm import GroupRMSNorm
from xllm.modules.residual import build_residual
from xllm.utils import get_init_fn


class NormalizedFeedForwardNetwork(nn.Module):
    def __init__(
        self,
        layer_id: int,
        model_dim: int,
        ffn_hidden_dim: int,
        dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        swiglu: bool = False,
        norm_num_groups: int = 1,
        norm_affine: bool = True,
        layernorm_eps: float = 1e-5,
        rmsnorm_eps: float = 1e-6,
        memory_efficient_norm: bool = False,
        apply_rmsnorm: bool = False,
        residual_func: str = 'base',
        residual_heads: Optional[int] = None,
        init_mode: str = 'gaussian',
        init_std: Optional[float] = None
    ):
        super().__init__()
        self.layer_id = layer_id

        self.model_dim = model_dim
        self.hidden_dim = ffn_hidden_dim
        self.dropout = dropout
        self.hidden_dropout = hidden_dropout
        self.swiglu = swiglu
        self.init_mode = init_mode
        self.init_std = init_std

        norm_cls = GroupRMSNorm if apply_rmsnorm else GroupLayerNorm
        norm_eps = rmsnorm_eps if apply_rmsnorm else layernorm_eps
        self.norm = norm_cls(
            model_dim,
            num_groups=norm_num_groups,
            elementwise_affine=norm_affine,
            eps=norm_eps,
            memory_efficient=memory_efficient_norm
        )

        # layers
        self.fc1 = ColumnParallelLinear(
            model_dim,
            ffn_hidden_dim,
            bias=False,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
            init_method=get_init_fn(init_mode, dim=model_dim, std=init_std),
        )
        self.fc2 = RowParallelLinear(
            ffn_hidden_dim,
            model_dim,
            bias=False,
            input_is_parallel=True,
            parallel_output=True,
            init_method=get_init_fn(init_mode, dim=ffn_hidden_dim, std=init_std),
        )
        self.fc3 = ColumnParallelLinear(
            model_dim,
            ffn_hidden_dim,
            bias=False,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
            init_method=get_init_fn(init_mode, dim=model_dim, std=init_std),
        ) if self.swiglu else None

        self.residual = build_residual(
            residual_func, 2 * layer_id + 2, model_dim, num_heads=residual_heads,
            num_features=ffn_hidden_dim, eps=rmsnorm_eps, init_std=init_std
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        # B x L x D
        residual = x
        x = self.norm(x)
        if not self.norm.gather_input:
            x = gather_copy_model_parallel_region(x)

        # fc1 & fc3
        if self.swiglu:
            hidden = F.silu(self.fc1(x)) * self.fc3(x)
            hidden = memory_efficient_dropout(hidden, self.hidden_dropout, self.training)
        else:
            hidden = F.silu(self.fc1(x))
            hidden = memory_efficient_dropout(hidden, self.hidden_dropout, self.training)

        # fc2
        y = self.fc2(hidden)
        y = memory_efficient_dropout(y, self.dropout, self.training)
        # residual
        out = self.residual(y, residual, c=hidden)
        return out

    def extra_repr(self) -> str:
        return 'dim={}, hdim={}, swiglu={}, init={} ({})'.format(
            self.model_dim, self.hidden_dim, self.swiglu, self.init_mode, self.init_std,
        )
