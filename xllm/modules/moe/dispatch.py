"""Token dispatch of the routed experts.

A dispatcher moves each token's 1/TP slice to the GPUs that own its experts and brings their outputs
back, summed per token. The MoE layers only see this interface, so each expert parallel layout is one
implementation of it:

    TPDispatcher    EP = TP: the TP ranks of a replica split the experts, tokens move inside the TP group.

Token sets: T, the tokens of this replica (every TP rank holds a 1/TP slice of each); M, the tokens this
replica's TP group computes experts for (M = T when EP = TP); n, the tokens this rank computes. Steps:

    plan(routing_indices, tokens_per_expert)    who sends what where, and the rows of each local expert
    dispatch(x, plan).wait()                    T x D/TP -> n x D, the full rows of this rank's tokens
    combine(y, plan).wait()                     n x D -> T x D/TP, summed over the ranks that computed them
    local_scores(scores, plan)                  T x K routing scores -> M x K, the scores of the M tokens
    scores_grad(grad, plan)                     M x K gradient of local_scores -> T x K

dispatch and combine return a Pending: the communication is started and wait() finishes it, so the
fused blocks run other work in between. combine is the adjoint of dispatch, which dispatch_rows and
combine_rows use for autograd. The fused blocks keep plan.state between forward and backward and
rebuild the plan with replan.
"""
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import Tensor
from einops import rearrange

from xllm.distributed import (
    get_model_parallel_group,
    get_model_parallel_rank,
    get_model_parallel_world_size,
)
from .dedup import DedupPlan, token_rank_hits, build_dedup_plan, masked_gather_sum
from .layout import ExpertLayout


def all_to_all(x: Tensor, input_splits: List[int], output_splits: List[int], group, world_size: int):
    """Start an all_to_all_single of the rows of x; returns the output and the handle to wait on."""
    if world_size == 1:
        return x, None
    output = torch.empty([sum(output_splits)] + list(x.shape[1:]), device=x.device, dtype=x.dtype)
    handle = torch.distributed.all_to_all_single(
        output, x, output_splits, input_splits, group=group, async_op=True
    )
    return output, handle


class Pending:
    """A dispatch or a combine in flight. wait() returns its result."""

    def __init__(self, output: Tensor, handle, finish: Callable[[Tensor], Tensor], keep: Optional[Tensor] = None):
        self.output = output
        self.handle = handle
        self.finish = finish
        # the send buffer, alive until the communication ends
        self.keep = keep

    def wait(self) -> Tensor:
        if self.handle is not None:
            self.handle.wait()
        output, finish = self.output, self.finish
        self.output, self.handle, self.finish, self.keep = None, None, None, None
        return finish(output)


@dataclass
class TPState:
    """What TPDispatcher.replan needs besides the routing indices."""
    group_sizes: List[int]
    pair_start: int
    dest_counts: List[int]


@dataclass
class DispatchPlan:
    # the M tokens to the TP ranks, and the (token, expert) pairs of this rank
    tp: DedupPlan
    # rows of each local expert
    group_sizes: List[int]
    # M
    n_tokens: int
    # kept by the fused blocks for replan
    state: Any

    @property
    def n_pairs(self) -> int:
        return sum(self.group_sizes)


class TPDispatcher:
    """EP = TP: each TP rank owns n_local consecutive experts; a token's slice goes once to every TP rank
    that owns one of its experts (see dedup.py).

    The expert indices it is given number the n_experts experts of the TP group. With masked=True they
    may also hold n_experts, for a slot whose expert is elsewhere: such a slot goes nowhere.
    """

    def __init__(self, n_experts: int, n_local: int, topk: int, backend: str, masked: bool = False):
        self.n_experts = n_experts
        self.n_local = n_local
        self.topk = topk
        self.backend = backend
        self.masked = masked
        self.rank = get_model_parallel_rank()
        self.world_size = get_model_parallel_world_size()
        self.start = self.rank * n_local
        assert n_experts == n_local * self.world_size, f"{n_experts} != {n_local} x {self.world_size}"

    @property
    def group(self):
        # looked up when used, not kept: a module holding the dispatcher can then be deep-copied
        return get_model_parallel_group()

    def _hits(self, routing_indices: Tensor) -> Tensor:
        if not self.masked:
            return token_rank_hits(routing_indices, self.n_local, self.topk, self.world_size)
        # the masked slots fall in an extra column, dropped
        return token_rank_hits(routing_indices, self.n_local, self.topk, self.world_size + 1)[:, :self.world_size]

    def _build(self, routing_indices: Tensor, hits: Tensor, state: TPState) -> DispatchPlan:
        sorted_indices = torch.argsort(routing_indices, stable=True)
        plan = build_dedup_plan(
            hits, sorted_indices, self.topk, self.rank, state.dest_counts, state.pair_start, sum(state.group_sizes)
        )
        return DispatchPlan(plan, state.group_sizes, hits.size(0), state)

    def plan(self, routing_indices: Tensor, tokens_per_expert: Tensor) -> DispatchPlan:
        """routing_indices: M*K flat expert indices; tokens_per_expert: the n_experts counts of the indices."""
        # M x TP
        hits = self._hits(routing_indices)
        # N + TP
        counts = torch.cat([tokens_per_expert, hits.sum(dim=0)]).tolist()
        expert_counts, dest_counts = counts[:self.n_experts], counts[self.n_experts:]
        state = TPState(
            group_sizes=expert_counts[self.start:self.start + self.n_local],
            pair_start=sum(expert_counts[:self.start]),
            dest_counts=dest_counts,
        )
        return self._build(routing_indices, hits, state)

    def replan(self, routing_indices: Tensor, state: TPState) -> DispatchPlan:
        """The plan for the same routing, rebuilt without communication or sync from its state."""
        return self._build(routing_indices, self._hits(routing_indices), state)

    def dispatch(self, x: Tensor, plan: DispatchPlan) -> Pending:
        # S x D/TP: each token's slice once per rank holding one of its experts
        send = torch.index_select(x, 0, plan.tp.send_tokens)
        output, handle = all_to_all(send, plan.tp.send_splits, plan.tp.recv_splits, self.group, self.world_size)
        # n x D: this rank's tokens, rebuilt
        finish = lambda output: rearrange(output, '(w b) d -> b (w d)', w=self.world_size)
        return Pending(output, handle, finish, keep=send)

    def combine(self, y: Tensor, plan: DispatchPlan) -> Pending:
        y = rearrange(y, 'b (w d) -> (w b) d', w=self.world_size)
        output, handle = all_to_all(y, plan.tp.recv_splits, plan.tp.send_splits, self.group, self.world_size)
        # T x D/TP: add up the parts returned by each rank
        finish = lambda output: masked_gather_sum(output, plan.tp.home_table, None, self.backend)
        return Pending(output, handle, finish)

    def local_scores(self, scores: Tensor, plan: DispatchPlan) -> Tensor:
        return scores

    def scores_grad(self, grad: Tensor, plan: DispatchPlan) -> Tensor:
        return grad


@dataclass
class _Saved:
    """Stands for a tensor of a plan saved for backward."""
    index: int


def _strip_tensors(obj, tensors: List[Tensor]):
    """obj with each tensor in it (dataclasses, nested) moved to tensors and replaced by its _Saved index."""
    if isinstance(obj, Tensor):
        tensors.append(obj)
        return _Saved(len(tensors) - 1)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = {f.name: _strip_tensors(getattr(obj, f.name), tensors) for f in dataclasses.fields(obj)}
        return dataclasses.replace(obj, **fields)
    return obj


def _put_tensors(obj, tensors: Tuple[Tensor, ...]):
    """The inverse of _strip_tensors."""
    if isinstance(obj, _Saved):
        return tensors[obj.index]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = {f.name: _put_tensors(getattr(obj, f.name), tensors) for f in dataclasses.fields(obj)}
        return dataclasses.replace(obj, **fields)
    return obj


def _save_plan(ctx, dispatcher, plan: DispatchPlan):
    """Keep the plan for backward with its tensors saved by save_for_backward, not as attributes of ctx: an
    activation checkpoint frees them after forward and takes them from its recompute, which rebuilds the plan."""
    tensors = []
    ctx.dispatcher, ctx.plan = dispatcher, _strip_tensors(plan, tensors)
    ctx.save_for_backward(*tensors)


def _saved_plan(ctx) -> DispatchPlan:
    return _put_tensors(ctx.plan, ctx.saved_tensors)


def build_dispatcher(layout: ExpertLayout, topk: int, backend: str):
    """The dispatcher of the calling rank's expert layout."""
    if layout.e_out == 1:
        return TPDispatcher(layout.n_experts, layout.n_local, topk, backend)
    raise NotImplementedError("EP > TP is not supported yet")


class _DispatchRows(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, dispatcher, plan: DispatchPlan) -> Tensor:
        _save_plan(ctx, dispatcher, plan)
        return dispatcher.dispatch(x, plan).wait()

    @staticmethod
    def backward(ctx, grad: Tensor):
        return ctx.dispatcher.combine(grad, _saved_plan(ctx)).wait(), None, None


class _CombineRows(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y: Tensor, dispatcher, plan: DispatchPlan) -> Tensor:
        _save_plan(ctx, dispatcher, plan)
        return dispatcher.combine(y, plan).wait()

    @staticmethod
    def backward(ctx, grad: Tensor):
        return ctx.dispatcher.dispatch(grad, _saved_plan(ctx)).wait(), None, None


def dispatch_rows(x: Tensor, dispatcher, plan: DispatchPlan) -> Tensor:
    """dispatcher.dispatch with autograd: T x D/TP -> n x D."""
    return _DispatchRows.apply(x, dispatcher, plan)


def combine_rows(y: Tensor, dispatcher, plan: DispatchPlan) -> Tensor:
    """dispatcher.combine with autograd: n x D -> T x D/TP."""
    return _CombineRows.apply(y, dispatcher, plan)
