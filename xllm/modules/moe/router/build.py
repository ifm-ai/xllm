from typing import Callable, Optional

import torch
import torch.nn.init as init

from .base import BaseRouter
from .topk import TopKRouter
from .dynamic_qb import DynamicQBRouter


def build_router(
    router_type: str,
    in_features: int,
    num_experts: int,
    topk: int,
    bias: bool = False,
    bias_update_rate: Optional[float] = None,
    dynamic_qb_update_method: str = 'average',
    dynamic_qb_decay: float = 0.3,
    dynamic_qb_globalnorm: bool = True,
    score_func: str = 'sigmoid',
    scaling_factor: Optional[float] = None,
    init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
) -> BaseRouter:
    if router_type == 'topk':
        return TopKRouter(
            in_features, num_experts, topk, bias=bias, bias_update_rate=bias_update_rate,
            score_func=score_func, scaling_factor=scaling_factor, init_method=init_method
        )
    elif router_type == 'dynamic_qb':
        assert not bias, "Dynamic QB keeps its thresholds in the router bias."
        return DynamicQBRouter(
            in_features, num_experts, topk, update_method=dynamic_qb_update_method, decay=dynamic_qb_decay,
            globalnorm=dynamic_qb_globalnorm, score_func=score_func, scaling_factor=scaling_factor,
            init_method=init_method
        )
    else:
        raise ValueError(f"Unknown router type: {router_type}")
