import math
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn import functional as F

from xllm.distributed.utils import reshape_gathered_tensor_along_specific_dim
from .utils import (
    rmsnorm_fwd,
    rmsnorm_bwd,
)
from .distributed import (
    reduce_scatter,
    all_gather,
)


def residual_connection_fwd(
    h: Tensor,
    residual: Tensor,
    out: Optional[Tensor],
    rc_func: str,
    rc_id: int,
    local_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    eps: Optional[float] = None
) -> Tensor:
    if rc_func in ['base', 'add']:
        return base_rc_fwd(h, residual, out)
    elif rc_func == 'delta':
        return delta_rc_fwd(h, residual, out, rc_id, local_heads, head_dim, eps)
    else:
        raise ValueError(f"Unknown residual function: {rc_func}")


def base_rc_fwd(
    h: Tensor,
    residual: Tensor,
    out: Optional[Tensor],
) -> Tensor:
    return torch.add(h, residual, out=out)


def delta_rc_fwd(
    h: Tensor,
    residual: Tensor,
    out: Optional[Tensor],
    rc_id: int,
    local_heads: int,
    head_dim: int,
    eps: float
) -> Tensor:
    bsz, slen, mdim = h.shape
    # qk = L2-Norm(h)
    qk = rmsnorm_fwd(h, None, local_heads, eps) / math.sqrt(head_dim)
    return None


def residual_connection_bwd(
    out_grad: Tensor,
    h: Tensor,
    residual: Tensor,
    rc_func: str,
    rc_id: int,
    local_heads: int,
    head_dim: int,
    eps: float
) -> Tuple[Tensor, Tensor]:
    if rc_func in ['base', 'add']:
        return base_rc_bwd(out_grad)
    elif rc_func == 'delta':
        return delta_rc_bwd(out_grad, h, residual, rc_id, local_heads, head_dim, eps)
    else:
        raise ValueError(f"Unknown residual function: {rc_func}")


def base_rc_bwd(
    out_grad: Tensor,
) -> Tuple[Tensor, Tensor]:
    return out_grad, out_grad


def delta_rc_bwd(
    out_grad: Tensor,
    h: Tensor,
    residual: Tensor,
    rc_id: int,
    local_heads: int,
    head_dim: int,
    eps: float
) -> Tuple[Tensor, Tensor]:
    pass
