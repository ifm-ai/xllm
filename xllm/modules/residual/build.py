from typing import Optional

from torch import nn

from .base import BaseResidual
from .delta import DeltaResidual
from .ortho import OrthoResidual


def build_residual(
    residual_func: str,
    rc_id: int,
    model_dim: int,
    num_heads: Optional[int],
    num_features: Optional[int] = None,
    eps: Optional[float] = None,
    init_std: Optional[float] = None,
    multiplier: float = 1.0,
) -> nn.Module:
    if residual_func in ['base', 'add']:
        return BaseResidual(rc_id, model_dim, multiplier)
    elif residual_func == 'ortho':
        assert num_heads is not None
        assert multiplier == 1.0, 'ortho residual does not support a branch multiplier'
        return OrthoResidual(rc_id, model_dim, num_heads, eps)
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
    else:
        return 0
