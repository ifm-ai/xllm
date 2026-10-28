from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx
from einops import rearrange, repeat
try:
    from xattn.ops import (
        sliding_chunk_group_softdelta_attention
    )
    XATTN_SCA_SOFTDELTA_ENABLED = True
except ImportError:
    XATTN_SCA_SOFTDELTA_ENABLED = False


def sliding_chunk_softdelta_attention(
    q: torch.Tensor,
    qk: torch.Tensor,
    kk: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
    scale: float,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    segment_idx: Optional[Tensor] = None,
    dropout: float = 0.0,
    high_precision_level: int = 0,
    deterministic: bool = True,
    backend: str = 'xattn',
    training: bool = True,
) -> Tensor:
    p = dropout if training else 0.0
    if backend in ['xattn', 'flash']:
        assert XATTN_SCA_SOFTDELTA_ENABLED, "xattn was not installed."
        if bos_mask is not None:
            segment_idx = None

        return sliding_chunk_group_softdelta_attention(
            q, kk, v, g, chunk_size, qk, scale, prev_k, prev_v,
            bos_mask=bos_mask, segment_idx=segment_idx,
            high_precision_level=high_precision_level,
            deterministic=deterministic,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}.")
