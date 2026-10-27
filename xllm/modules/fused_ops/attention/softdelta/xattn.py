from typing import Tuple, Optional

import torch
from torch import Tensor
try:
    from xattn.ops import (
        group_softdelta_attention,
        group_softdelta_attention_fwd,
        group_softdelta_attention_bwd
    )
    XATTN_SOFTDELTA_ATTN_ENABLED = True
except ImportError:
    XATTN_SOFTDELTA_ATTN_ENABLED = False


def xattn_softdelta_attn(
    q: torch.Tensor,
    qk: torch.Tensor,
    kk: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float] = None,
    bos_mask: Optional[Tensor] = None,
    segment_idx: Optional[Tensor] = None,
    high_precision_level: int = 0,
    deterministic: bool = False
) -> Tensor:
    assert XATTN_SOFTDELTA_ATTN_ENABLED, "xattn was not installed."
    if bos_mask is not None:
        segment_idx = None

    return group_softdelta_attention(
        q, kk, v, g, qk, scale,
        bos_mask=bos_mask,
        segment_idx=segment_idx,
        high_precision_level=high_precision_level,
        deterministic=deterministic
    )


def xattn_softdelta_attn_fwd(
    q: torch.Tensor,
    qk: torch.Tensor,
    kk: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float] = None,
    bos_mask: Optional[Tensor] = None,
    segment_idx: Optional[Tensor] = None,
    high_precision_level: int = 0,
    requires_grad: bool = False
) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    assert XATTN_SOFTDELTA_ATTN_ENABLED, "xattn was not installed."
    if bos_mask is not None:
        segment_idx = None

    y, readers, lse = group_softdelta_attention_fwd(
        q, kk, v, g, qk, scale,
        bos_mask=bos_mask,
        segment_idx=segment_idx,
        high_precision_level=high_precision_level
    )
    lse = lse if requires_grad else None
    readers = readers if requires_grad else None
    return y, readers, lse


def xattn_causal_flash_attn_bwd(
    y_grad: Tensor,
    q: Tensor,
    qk: torch.Tensor,
    k: Tensor,
    v: Tensor,
    g: torch.Tensor,
    readers: Tensor,
    lse: Tensor,
    scale: Optional[float] = None,
    bos_mask: Optional[Tensor] = None,
    segment_idx: Optional[Tensor] = None,
    high_precision_level: int = 0,
    deterministic: bool = False
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert XATTN_SOFTDELTA_ATTN_ENABLED, "xattn was not installed."
    if bos_mask is not None:
        segment_idx = None

    return group_softdelta_attention_bwd(
        y_grad, q, k, v, y, lse, scale,
        bos_mask=bos_mask, segment_idx=segment_idx,
        high_precision_level=high_precision_level,
        deterministic=deterministic
    )
