from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


def get_bos_mask(tokens: Tensor, special_id: int, use_eos: bool) -> Tensor:
    """
    Get bos mask from BOS id
    Args:
        tokens: LongTensor with token ids (shape [B, T])
        special_id: int: bos id if use_eos is False else eos_id
        use_eos: bool: True if use eos_id

    Return:
        out: BoolTensor for bos masks (shape [B, T])
    """
    if use_eos:
        return get_bos_mask_from_eos(tokens, eos_id=special_id)

    return torch.eq(tokens, special_id)


def get_bos_mask_from_eos(tokens: Tensor, eos_id: int) -> Tensor:
    """
    Get bos mask from EOS id
    Args:
        tokens: LongTensor with token ids (shape [B, T])
        eos_id: int

    Return:
        out: BoolTensor for bos masks (shape [B, T])
    """
    eos_mask = torch.eq(tokens, eos_id)
    return F.pad(eos_mask[:, :-1], (1, 0), mode='constant', value=1)


def get_segment_idx_from_bos_mask(bos_mask: Tensor) -> Tensor:
    """
    Get segment indexes from the bos mask of one long sequence
    Args:
        bos_mask: BoolTensor (shape [B, T])

    Return:
        out: LongTensor for segment idx (shape [B, T])
    """
    return torch.cumsum(bos_mask, dim=-1)


def get_cu_seqlens_from_bos_mask(bos_mask: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Get cumulative seqlens from bos mask of one log sequence
    Args:
        bos_mask: BoolTensor (shape [B, T])

    Returns:
        cu_seqlens: cumulative sequence lengths
        end_idx: the indices of the segments at the end of each sequence.
    """
    bsz, seq_len = bos_mask.shape
    bos_mask = F.pad(bos_mask[:, 1:], (1, 0), value=1)
    # B
    end_idx = torch.cumsum(torch.sum(bos_mask, dim=1), dim=0) - 1
    bos_idx = torch.nonzero(bos_mask)
    cu_seqlens = bos_idx[:, 0] * seq_len + bos_idx[:, 1]
    cu_seqlens = F.pad(cu_seqlens, (0, 1), value=bsz * seq_len)
    return cu_seqlens, end_idx
