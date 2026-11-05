from typing import Optional

from torch import nn

from .base import BaseResidual
from .delta import DeltaResidual
from .ortho import OrthoResidual
from .hyper_connection import (
    HYPER_CONNECTIONS,
    NUM_STREAMS,
    num_params_in_hyper_connection,
    flops_in_hyper_connection,
)


def build_residual(
    residual_func: str,
    rc_id: int,
    model_dim: int,
    num_heads: Optional[int],
    num_features: Optional[int] = None,
    eps: Optional[float] = None,
    init_std: Optional[float] = None
) -> nn.Module:
    if residual_func in ['base', 'add']:
        return BaseResidual(rc_id, model_dim)
    elif residual_func == 'ortho':
        assert num_heads is not None
        return OrthoResidual(rc_id, model_dim, num_heads, eps)
    elif residual_func in HYPER_CONNECTIONS:
        return HYPER_CONNECTIONS[residual_func](rc_id, model_dim, eps=eps)
    else:
        raise ValueError(f"Unknown residual function: {residual_func}")


def num_params_in_residual(
    residual_func,
    model_dim: int,
    num_heads: Optional[int],
    num_features: Optional[int],
) -> int:
    if residual_func == 'delta':
        return num_features * num_heads + num_heads * 2
    elif residual_func == 'peri':
        return model_dim
    elif residual_func in HYPER_CONNECTIONS:
        return num_params_in_hyper_connection(residual_func, model_dim)
    else:
        return 0


def flops_in_residual(residual_func: str, model_dim: int) -> int:
    """Multiply-adds per token of one sublayer's residual (0 but for hyper-connections)."""
    return flops_in_hyper_connection(residual_func, model_dim) if residual_func in HYPER_CONNECTIONS else 0


def num_residual_streams(residual_func: str) -> int:
    """Streams of the residual state: N for hyper-connections ([B, L, N, D]), 1 otherwise ([B, L, D])."""
    return NUM_STREAMS if residual_func in HYPER_CONNECTIONS else 1
