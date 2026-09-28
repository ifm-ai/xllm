from typing import Tuple, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from fla.modules.conv.triton import (
    causal_conv1d_fwd,
    causal_conv1d_bwd
)
from fla.ops.utils import prepare_chunk_indices
from einops import rearrange
from xllm.modules.utils import get_cu_seqlens_from_bos_mask


def _fla_causal_conv1d_fwd(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    initial_state: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    output_final_state: bool = False,
    activation: Optional[str] = None,
) -> Tuple[Tensor, Optional[Tensor], Tuple[Optional[Tensor], Optional[Tensor]]]:
    """
    Args:
        x (Tensor): (batch, seqlen, dim)
        weight (Tensor): (dim, width)
        bias Optional[Tensor]: (dim)
        initial_state Optional[Tensor]: (batch, width - 1, dim)
        bos_mask Optional[Tensor]: (batch, seqlen)
        output_final_state (bool):
            whether to output the final state of shape [batch, width - 1, dim]. Default: `False`.
        activation (Optional[str]):
            Activations applied to output, only `swish`/`silu` or `None` (i.e., no activation) are supported.
            Default: `None`.

    Return:
        out (Tensor): (batch, dim, seqlen)
        final_states (Optional[Tensor]): (batch, width - 1, dim)
        (cu_seqlens, chunk_indices)
    """
    BT = 64
    bsz, seq_len, dim = x.shape
    width = weight.shape[1]
    if initial_state is not None:
        assert width == initial_state.shape[1] + 1
        # B x (L+W-1) x D
        x = torch.cat([initial_state, x], dim=1)
        if bos_mask is not None:
            bos_mask = F.pad(bos_mask, (width - 1, 0), value=0)

    if bos_mask is not None:
        cu_seqlens, end_seqidx = get_cu_seqlens_from_bos_mask(bos_mask)
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        x = rearrange(x, 'b l d -> 1 (b l) d')
    else:
        cu_seqlens = None
        end_seqidx = None
        chunk_indices = None

    out, final_state = causal_conv1d_fwd(
        x=x,
        weight=weight,
        bias=bias,
        residual=None,
        initial_state=None,
        output_final_state=output_final_state,
        activation=activation,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        BT=BT,
    )

    if bos_mask is not None:
        out = rearrange(out, '1 (b l) d -> b l d', b=bsz)

    if initial_state is not None:
        out = out[:, (width - 1):]

    if final_state is not None:
        if end_seqidx is not None:
            final_state = final_state[end_seqidx]
        final_state = rearrange(final_state, 'b d w -> b w d')
        # B x W-1 x D
        final_state = final_state[:, 1:]

    return out, final_state, (cu_seqlens, chunk_indices)


def _fla_causal_conv1d_bwd(
    out_grad: Tensor,
    final_state_grad: Optional[Tensor],
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    initial_state: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    activation: Optional[str] = None,
) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
    """
    Args:
        out_grad (Tensor): (batch, seqlen, dim)
        final_state_grad (Optional[Tensor]): (batch, width - 1, dim)
        x (Tensor): (batch, seqlen, dim)
        weight (Tensor): (dim, width)
        bias (Optional[Tensor]): (dim)
        initial_state (Optional[Tensor]): (batch, width - 1, dim)
        bos_mask (Optional[Tensor]): (batch, seqlen)
        activation (Optional[str]):
            Activations applied to output, only `swish`/`silu` or `None` (i.e., no activation) are supported.
            Default: `None`.
        backend (str): backend implementation

    Return:
        x_grad (Tensor): (batch, seqlen, dim)
        initial_state_grad (Optional[Tensor]): (batch, width - 1, dim)
        weight_grad (Tensor): (dim, width)
        bias_grad (Optional[Tensor]): (dim)
    """
    BT = 64
    bsz, seq_len, dim = x.shape
    width = weight.shape[1]
    if initial_state is not None:
        assert width == initial_state.shape[1] + 1
        # B x (L+W-1) x D
        x = torch.cat([initial_state, x], dim=1)
        out_grad = F.pad(out_grad, (0, 0, width - 1, 0), value=0)
        if bos_mask is not None:
            bos_mask = F.pad(bos_mask, (width - 1, 0), value=0)

    if bos_mask is not None:
        cu_seqlens, _ = get_cu_seqlens_from_bos_mask(bos_mask)
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        x = rearrange(x, 'b l d -> 1 (b l) d')
    else:
        cu_seqlens = None
        chunk_indices = None

    if cu_seqlens is not None:
        x = rearrange(x, 'b l d -> 1 (b l) d')
        out_grad = rearrange(out_grad, 'b l d -> 1 (b l) d')

    x_grad, w_grad, b_grad, _, _ = causal_conv1d_bwd(
        x=x,
        dy=out_grad,
        dht=None,
        weight=weight,
        bias=bias,
        residual=None,
        initial_state=None,
        activation=activation,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        BT=BT
    )

    if cu_seqlens is not None:
        x_grad = rearrange(x_grad, '1 (b l) d -> b l d', b=bsz)

    if final_state_grad is not None:
        assert width == final_state_grad.shape[1] + 1
        w = min(x_grad.shape[1], final_state_grad.shape[1])
        # B x (W-1) x D
        x_grad_from_final_state = final_state_grad[:, -w:]
        if bos_mask is not None:
            segidx = torch.cumsum(bos_mask[:, -w:], dim=-1)
            # B x W x 1
            grad_mask = torch.ne(segidx, segidx[:, -1:]).unsqueeze(2)
            x_grad_from_final_state = x_grad_from_final_state.masked_fill(grad_mask, value=0)
        x_grad[:, -w:] += x_grad_from_final_state

    if initial_state is not None:
        initial_state_grad = x_grad[:, :(width - 1)]
        x_grad = x_grad[:, (width - 1):]
    else:
        initial_state_grad = None

    return x_grad, initial_state_grad, w_grad, b_grad
