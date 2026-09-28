from typing import Tuple, Optional

import torch
from torch.autograd.function import FunctionCtx

from xllm_extension.ops import (
    attention_fwd,
    attention_bwd,
    multiseg_attention_fwd,
)


class SwiftAttentionFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_segment_idx: Optional[torch.Tensor],
        k_segment_idx: Optional[torch.Tensor],
        scale: float = 1.0,
        dropout: float = 0.0,
        use_causal_mask: bool = True,
        training: bool = True
    ) -> torch.Tensor:
        p = dropout if training else 0.0
        y, w = _swift_efficient_attention_fwd(
            q, k, v, q_segment_idx, k_segment_idx, scale, p, use_causal_mask
        )
        ctx.save_for_backward(q, k, v, w)
        # scale is not a torch.Tensor
        ctx.scale = scale
        # use_causal_mask is not a torch.Tensor
        ctx.use_causal_mask = use_causal_mask
        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               None, None, None, None, None, None]:
        q, k, v, w = ctx.saved_tensors
        scale = ctx.scale
        use_causal_mask = ctx.use_causal_mask
        q_grad, k_grad, v_grad = _swift_efficient_attention_bwd(y_grad, q, k, v, w, scale, use_causal_mask)
        return q_grad, k_grad, v_grad, None, None, None, None, None, None


swift_efficient_attention = SwiftAttentionFunc.apply


def _swift_efficient_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_segment_idx: Optional[torch.Tensor],
    k_segment_idx: Optional[torch.Tensor],
    scale: float = 1.0,
    dropout: float = 0.0,
    use_causal_mask: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if q_segment_idx is None:
        y, w = attention_fwd(q, k, v, scale, dropout, use_causal_mask)
    else:
        y, w = multiseg_attention_fwd(q, k, v, q_segment_idx, k_segment_idx, scale, dropout, use_causal_mask)
    return y, w


def _swift_efficient_attention_bwd(
    grad: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    scale: float = 1.0,
    use_causal_mask: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_grad, k_grad, v_grad = attention_bwd(grad, q, k, v, w, scale, use_causal_mask)
    return q_grad, k_grad, v_grad


def swift_efficient_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_segment_idx: Optional[torch.Tensor],
    k_segment_idx: Optional[torch.Tensor],
    scale: float = 1.0,
    dropout: float = 0.0,
    use_causal_mask: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    return _swift_efficient_attention_fwd(
        q, k, v, q_segment_idx, k_segment_idx, scale, dropout, use_causal_mask
    )


def swift_efficient_attention_bwd(
    grad: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    scale: float = 1.0,
    use_causal_mask: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _swift_efficient_attention_bwd(grad, q, k, v, w, scale, use_causal_mask)
