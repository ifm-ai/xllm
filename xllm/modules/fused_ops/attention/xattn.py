from typing import Tuple, Optional

import torch
from torch import Tensor
try:
    from xattn.ops import (
        causal_flash_attn,
        causal_flash_attn_fwd,
        causal_flash_attn_bwd
    )
    XATTN_FLASH_ATTN_ENABLED = True
except ImportError:
    XATTN_FLASH_ATTN_ENABLED = False


def xattn_causal_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    bos_mask: Optional[Tensor] = None,
    segment_idx: Optional[Tensor] = None,
    high_precision_level: int = 0,
    deterministic: bool = False
) -> Tensor:
    assert XATTN_FLASH_ATTN_ENABLED, "xattn was not installed."
    if bos_mask is not None:
        segment_idx = None

    return causal_flash_attn(
        q, k, v, scale,
        bos_mask=bos_mask,
        segment_idx=segment_idx,
        high_precision_level=high_precision_level,
        deterministic=deterministic
    )


def xattn_causal_flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    bos_mask: Optional[Tensor] = None,
    segment_idx: Optional[Tensor] = None,
    high_precision_level: int = 0,
    requires_grad: bool = False
) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    assert XATTN_FLASH_ATTN_ENABLED, "xattn was not installed."
    if bos_mask is not None:
        segment_idx = None

    high_precision_level = high_precision_level if requires_grad else 0
    y, y_fp32, lse = causal_flash_attn_fwd(
        q, k, v, scale,
        bos_mask=bos_mask,
        segment_idx=segment_idx,
        high_precision_level=high_precision_level
    )
    lse = lse if requires_grad else None
    y_for_bwd = y if requires_grad and high_precision_level == 0 else y_fp32
    return y, y_for_bwd, lse


def xattn_causal_flash_attn_bwd(
    y_grad: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    y: Tensor,
    lse: Tensor,
    scale: Optional[float] = None,
    bos_mask: Optional[Tensor] = None,
    segment_idx: Optional[Tensor] = None,
    high_precision_level: int = 0,
    deterministic: bool = False
) -> Tuple[Tensor, Tensor, Tensor]:
    assert XATTN_FLASH_ATTN_ENABLED, "xattn was not installed."
    if bos_mask is not None:
        segment_idx = None

    return causal_flash_attn_bwd(
        y_grad, q, k, v, y, lse, scale,
        bos_mask=bos_mask, segment_idx=segment_idx,
        high_precision_level=high_precision_level,
        deterministic=deterministic
    )
