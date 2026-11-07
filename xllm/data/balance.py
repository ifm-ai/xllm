from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from xllm.modules.utils import get_bos_mask_from_eos


def attention_row_cost(tokens: Tensor, eos_id: int, chunk_size: Optional[int]) -> Tensor:
    """Visible (query, key) pairs per row of a packed batch [B, L]: causal within each EOS-delimited segment and, with
    chunk_size, within the sliding chunk window (the query's chunk and the one before it)."""
    B, L = tokens.shape
    pos = torch.arange(L, device=tokens.device).expand(B, L)
    bos = get_bos_mask_from_eos(tokens, eos_id=eos_id)
    lo = torch.cummax(torch.where(bos, pos, torch.zeros_like(pos)), dim=1).values
    if chunk_size:
        lo = torch.maximum(lo, (pos // chunk_size - 1).clamp_min(0) * chunk_size)
    return (pos - lo + 1).sum(dim=1)


def _assign(cost: List[int], world: int, rows: int) -> List[List[int]]:
    """Largest cost first onto the rank with the least load and a free slot (ties by index, so every rank computes
    the same assignment)."""
    loads, slots = [0] * world, [[] for _ in range(world)]
    for i in sorted(range(len(cost)), key=lambda i: (-cost[i], i)):
        r = min((r for r in range(world) if len(slots[r]) < rows), key=lambda r: (loads[r], r))
        slots[r].append(i)
        loads[r] += cost[i]
    return [sorted(s) for s in slots]


def balance_rows_across_ranks(
    x: Tensor, y: Tensor, mask: Optional[Tensor], eos_id: int, chunk_size: Optional[int],
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Redistribute the rows of the data-parallel ranks' batches so that every rank gets about the same attention
    work (EOS-packed rows differ a lot: the step waits for the rank whose rows have the longest segments). Every rank
    keeps its row count, so a loss normalized by the global token count and its gradient are unchanged; only which
    rank computes which row changes. The rows are all-gathered (B * L int64 per rank) and assigned greedily."""
    world = dist.get_world_size(group)
    B, L = x.shape
    if world == 1 or B == 1:
        return x, y, mask
    parts = [x, y] + ([mask] if mask is not None else [])
    packed = torch.stack([t.to(torch.int64) for t in parts])
    gathered = torch.empty((world,) + tuple(packed.shape), dtype=packed.dtype, device=packed.device)
    dist.all_gather_into_tensor(gathered, packed, group=group)
    rows = gathered.transpose(0, 1).reshape(len(parts), world * B, L)
    slots = _assign(attention_row_cost(rows[0], eos_id, chunk_size).tolist(), world, B)
    idx = torch.tensor(slots[dist.get_rank(group)], device=x.device)
    out = [rows[i].index_select(0, idx).to(t.dtype) for i, t in enumerate(parts)]
    return out[0], out[1], (out[2] if mask is not None else None)
