from typing import Tuple, Optional

import torch
from torch import Tensor
try:
    from xattn.ops import (
        softdelta_attention,
        softdelta_attention_fwd,
        softdelta_attention_bwd
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

    return softdelta_attention(
        q, kk, qk, v, g, scale,
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

    y, readers, lse = softdelta_attention_fwd(
        q, kk, qk, v, g, scale,
        bos_mask=bos_mask,
        segment_idx=segment_idx,
        high_precision_level=high_precision_level
    )
    lse = lse if requires_grad else None
    readers = readers if requires_grad else None
    return y, readers, lse


def xattn_softdelta_attn_bwd(
    y_grad: Tensor,
    q: Tensor,
    qk: torch.Tensor,
    kk: Tensor,
    v: Tensor,
    g: torch.Tensor,
    y: torch.Tensor,
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

    dq, dkk, dv, dg, dqk = softdelta_attention_bwd(
        y_grad, q, kk, qk, v, g, y, readers, lse, scale,
        bos_mask=bos_mask, segment_idx=segment_idx,
        high_precision_level=high_precision_level,
        deterministic=deterministic
    )
    return dq, dqk, dkk, dv, dg
