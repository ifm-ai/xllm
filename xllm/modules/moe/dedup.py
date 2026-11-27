"""Send each token once per destination rank in the MoE all-to-all.

Each TP rank holds a 1/TP slice of every token. The MoE dispatch used to send a
token's slice once per (token, expert) pair, so a rank owning several of a
token's experts received the same slice several times. A DedupPlan lists, for
every destination rank, the tokens that need it: each slice is sent once per
rank, the owner expands the rebuilt token locally for each of its experts, and
sums its experts' weighted outputs per token before sending them back.

Routing is identical on every TP rank (the router logits are all-reduced), so
every rank builds the same plan and no metadata travels.
"""
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
import triton
import triton.language as tl


@dataclass
class DedupPlan:
    # S: token ids to send, grouped by destination rank, ascending within a group
    send_tokens: Tensor
    # S: destination rank of each sent row
    send_ranks: Tensor
    # rows sent to (and received back from) each rank
    send_splits: List[int]
    # rows received from (and sent back to) each rank: the local tokens, from every rank
    recv_splits: List[int]
    # T x W: row of the send buffer holding (token, rank), -1 if the token skips that rank
    home_table: Tensor
    # P: flat (token * K + slot) index of each local (token, expert) pair, in expert order
    pairs: Tensor
    # P: row of the rebuilt local tokens for each local pair
    pair_rows: Tensor
    # P: top-k slot of each local pair
    pair_cols: Tensor
    # n x K: local pair index for (local token, slot), -1 if that expert is not local
    pair_table: Tensor
    # n x K: flat (token * K + slot) index of each entry of pair_table into the routing scores
    pair_slots: Tensor


def token_rank_hits(routing_indices: Tensor, n_local_experts: int, topk: int, world_size: int) -> Tensor:
    """T x W bool: whether a token picked at least one expert owned by each rank."""
    dest = routing_indices.view(-1, topk) // n_local_experts
    hits = torch.zeros(dest.size(0), world_size, dtype=torch.bool, device=dest.device)
    return hits.scatter_(1, dest, True)


def build_dedup_plan(
    hits: Tensor,
    sorted_indices: Tensor,
    topk: int,
    rank: int,
    dest_counts: List[int],
    pair_start: int,
    n_pairs: int,
) -> DedupPlan:
    n_tokens, world_size = hits.shape
    device = hits.device
    tokens = torch.arange(n_tokens, device=device)
    # W x T: each rank's tokens first, ascending
    by_rank = torch.where(hits.t(), tokens, n_tokens).sort(dim=1).values
    send_tokens = torch.cat([by_rank[r, :dest_counts[r]] for r in range(world_size)])
    # output_size avoids a device sync
    send_ranks = torch.repeat_interleave(
        torch.arange(world_size, device=device), hits.sum(dim=0), output_size=send_tokens.numel()
    )
    home_table = torch.full((n_tokens, world_size), -1, dtype=torch.long, device=device)
    home_table[send_tokens, send_ranks] = torch.arange(send_tokens.numel(), device=device)

    n_local = dest_counts[rank]
    local_tokens = by_rank[rank, :n_local]
    local_pos = torch.full((n_tokens,), -1, dtype=torch.long, device=device)
    local_pos[local_tokens] = torch.arange(n_local, device=device)
    pairs = sorted_indices[pair_start:pair_start + n_pairs]
    pair_rows = local_pos[pairs // topk]
    pair_cols = pairs % topk
    pair_table = torch.full((n_local * topk,), -1, dtype=torch.long, device=device)
    pair_table[pair_rows * topk + pair_cols] = torch.arange(n_pairs, device=device)
    pair_slots = local_tokens.unsqueeze(1) * topk + torch.arange(topk, device=device)

    return DedupPlan(
        send_tokens=send_tokens,
        send_ranks=send_ranks,
        send_splits=list(dest_counts),
        recv_splits=[n_local] * world_size,
        home_table=home_table,
        pairs=pairs,
        pair_rows=pair_rows,
        pair_cols=pair_cols,
        pair_table=pair_table.view(n_local, topk),
        pair_slots=pair_slots,
    )


@triton.jit
def _masked_gather_sum_kernel(
    output_ptr,
    src_ptr,
    table_ptr,
    weights_ptr,
    J: tl.constexpr,
    D: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_row = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for j in tl.static_range(0, J):
        src_idx = tl.load(table_ptr + pid_row * J + j)
        vals = tl.load(src_ptr + src_idx * D + d_offsets, mask=d_mask & (src_idx >= 0), other=0.0).to(tl.float32)
        if HAS_WEIGHTS:
            vals = vals * tl.load(weights_ptr + pid_row * J + j).to(tl.float32)
        acc += vals

    tl.store(output_ptr + pid_row * D + d_offsets, acc.to(output_ptr.dtype.element_ty), mask=d_mask)


def masked_gather_sum(src: Tensor, table: Tensor, weights: Optional[Tensor], backend: str) -> Tensor:
    """out[m] = sum over j with table[m, j] >= 0 of weights[m, j] * src[table[m, j]].

    Accumulates in fp32 and never adds twice into one output row, so the result is deterministic.
    """
    n_rows, n_cols = table.shape
    dim = src.size(1)
    if n_rows == 0 or src.size(0) == 0:
        return torch.zeros(n_rows, dim, device=src.device, dtype=src.dtype)

    if backend == 'torch':
        acc_dtype = torch.promote_types(src.dtype, torch.float32)
        acc = torch.zeros(n_rows, dim, device=src.device, dtype=acc_dtype)
        for j in range(n_cols):
            idx = table[:, j]
            vals = torch.index_select(src, 0, idx.clamp(min=0)).to(acc_dtype)
            if weights is not None:
                vals = vals * weights[:, j:j + 1].to(acc_dtype)
            acc += torch.where((idx >= 0).unsqueeze(1), vals, 0.0)
        return acc.to(src.dtype)
    elif backend == 'triton':
        has_weights = weights is not None
        src = src.contiguous()
        table = table.contiguous()
        # the kernel ignores the weights pointer when has_weights is False
        weights = weights.contiguous() if has_weights else table
        output = torch.empty(n_rows, dim, device=src.device, dtype=src.dtype)
        BLOCK_D = min(triton.next_power_of_2(dim), 1024)
        grid = (n_rows, triton.cdiv(dim, BLOCK_D))
        _masked_gather_sum_kernel[grid](
            output, src, table, weights,
            J=n_cols, D=dim, HAS_WEIGHTS=has_weights, BLOCK_D=BLOCK_D,
        )
        return output
    else:
        raise ValueError(f"Unknown backend: {backend}.")


class _GatherRows(torch.autograd.Function):
    """out = src[index]; each src row's gradient is summed over its uses, listed in `table`."""

    @staticmethod
    def forward(ctx, src: Tensor, index: Tensor, table: Tensor, backend: str) -> Tensor:
        ctx.save_for_backward(table)
        ctx.backend = backend
        return torch.index_select(src, 0, index)

    @staticmethod
    def backward(ctx, out_grad: Tensor):
        table, = ctx.saved_tensors
        return masked_gather_sum(out_grad, table, None, ctx.backend), None, None, None


class _WeightedSum(torch.autograd.Function):
    """out = masked_gather_sum(src, table, weights); src row p is table entry (rows[p], cols[p])."""

    @staticmethod
    def forward(ctx, src: Tensor, table: Tensor, weights: Optional[Tensor], rows: Tensor, cols: Tensor, backend: str):
        ctx.save_for_backward(src if weights is not None else None, table, weights, rows, cols)
        return masked_gather_sum(src, table, weights, backend)

    @staticmethod
    def backward(ctx, out_grad: Tensor):
        src, table, weights, rows, cols = ctx.saved_tensors
        n_cols = table.size(1)
        out_grad_rows = torch.index_select(out_grad, 0, rows)
        if weights is None:
            return out_grad_rows, None, None, None, None, None
        row_weights = weights.reshape(-1).index_select(0, rows * n_cols + cols)
        src_grad = (out_grad_rows.float() * row_weights.float().unsqueeze(1)).to(out_grad.dtype)
        weights_grad = None
        if ctx.needs_input_grad[2]:
            row_dots = (out_grad_rows.float() * src.float()).sum(dim=1)
            weights_grad = torch.zeros(table.numel(), device=out_grad.device, dtype=torch.float32)
            weights_grad[rows * n_cols + cols] = row_dots
            weights_grad = weights_grad.view_as(table).to(weights.dtype)
        return src_grad, None, weights_grad, None, None, None


def gather_rows(src: Tensor, index: Tensor, table: Tensor, backend: str) -> Tensor:
    return _GatherRows.apply(src, index, table, backend)


def weighted_sum(
    src: Tensor, table: Tensor, weights: Optional[Tensor], rows: Tensor, cols: Tensor, backend: str
) -> Tensor:
    return _WeightedSum.apply(src, table, weights, rows, cols, backend)
