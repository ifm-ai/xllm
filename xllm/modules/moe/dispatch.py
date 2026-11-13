"""Token dispatch of the routed experts.

A dispatcher moves each token's 1/TP slice to the GPUs that own its experts and brings their outputs
back, summed per token. The MoE layers only see this interface, so each expert parallel layout is one
implementation of it:

    TPDispatcher    EP = TP: the TP ranks of a replica split the experts, tokens move inside the TP group.
    NodeDispatcher  EP = TP x E_out: E_out replicas (nodes at TP 8) split the experts; a token goes once to
                    every replica holding one of its experts, then inside that replica's TP group.

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
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import Tensor
from einops import rearrange

from xllm.distributed import (
    get_model_parallel_group,
    get_model_parallel_rank,
    get_model_parallel_world_size,
    get_expert_outer_group,
)
from .dedup import DedupPlan, token_rank_hits, build_send_lists, build_dedup_plan, masked_gather_sum
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
class NodeState:
    """What NodeDispatcher.replan needs besides the routing indices."""
    # rows sent to and received from each replica of the expert group
    send_splits: List[int]
    recv_splits: List[int]
    # M*K int32: the experts of the received tokens, in the TP group's numbering (see NodeDispatcher)
    node_indices: Tensor
    tp: TPState


@dataclass
class OuterPlan:
    """The T tokens of a replica to the replicas of its expert group (NodeDispatcher)."""
    # S: token ids to send, grouped by destination replica, ascending within a group
    send_tokens: Tensor
    # T x E_out: row of the send buffer holding (token, replica), -1 if the token skips that replica
    home_table: Tensor
    send_splits: List[int]
    recv_splits: List[int]


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
    # EP > TP: the T tokens to the replicas that compute them
    outer: Optional[OuterPlan] = None

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

    def _state(self, expert_counts: List[int], dest_counts: List[int]) -> TPState:
        return TPState(
            group_sizes=expert_counts[self.start:self.start + self.n_local],
            pair_start=sum(expert_counts[:self.start]),
            dest_counts=dest_counts,
        )

    def counts(self, routing_indices: Tensor, tokens_per_expert: Tensor) -> Tuple[List[int], List[int]]:
        """The counts plan builds from: the n_experts tokens_per_expert, and the tokens of each TP rank (a
        token counted once per rank holding one of its experts). One host sync."""
        counts = torch.cat([tokens_per_expert, self._hits(routing_indices).sum(dim=0)]).tolist()
        return counts[:self.n_experts], counts[self.n_experts:]

    def plan(self, routing_indices: Tensor, tokens_per_expert: Tensor) -> DispatchPlan:
        """routing_indices: M*K flat expert indices; tokens_per_expert: the n_experts counts of the indices."""
        # M x TP
        hits = self._hits(routing_indices)
        # N + TP
        counts = torch.cat([tokens_per_expert, hits.sum(dim=0)]).tolist()
        expert_counts, dest_counts = counts[:self.n_experts], counts[self.n_experts:]
        return self._build(routing_indices, hits, self._state(expert_counts, dest_counts))

    def plan_from_counts(
        self, routing_indices: Tensor, expert_counts: List[int], dest_counts: List[int]
    ) -> DispatchPlan:
        """plan, given the counts it would read (see counts) instead of reading them: no host sync."""
        return self._build(routing_indices, self._hits(routing_indices), self._state(expert_counts, dest_counts))

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


class NodeDispatcher:
    """EP = TP x E_out: once per replica, then inside the replica's TP group.

    The GPU at TP rank k of replica o owns the experts k * N / TP + o * n_local + [0, n_local) (ExpertLayout),
    so the replica o owns TP x n_local experts. A token goes to each replica holding one of its experts once,
    each TP rank sending its 1/TP slice to the same TP rank there (the expert outer group); that replica runs
    TPDispatcher on the M tokens it received, numbering its experts k * n_local + i and every other expert
    TP * n_local (masked), and sums its experts' outputs per token before they go back. The routing indices
    and scores of the sent tokens go along with them. Every TP rank of a replica sends and receives the same
    tokens in the same order, so the replica's TP ranks agree on the M tokens and their routing.

    Per layer and rank, besides TPDispatcher's traffic on the M tokens: S x D/TP rows over the expert outer
    group in each dispatch and combine, S the (token, other replica) pairs plus the token's own replica,
    S x K int32 indices once per plan and S x K scores once per local_scores and scores_grad, and one
    all-to-all of E_out x (1 + TP x n_local + TP) counts per plan, its only host sync.

    Building one does not communicate. check_outer_group does: it is a collective over the expert outer
    group, which build_dispatcher runs once per dispatcher it builds.
    """

    # check the counts sent with the tokens against those TPDispatcher.plan reads from their routing (tests)
    check_plan = os.environ.get('XLLM_CHECK_DISPATCH', '0') == '1'

    def __init__(self, layout: ExpertLayout, topk: int, backend: str):
        assert layout.e_out > 1
        self.layout = layout
        self.topk = topk
        self.backend = backend
        self.n_local = layout.n_local
        # experts of one TP rank across the expert group
        self.share = layout.n_experts // layout.tp
        self.outer_rank = layout.outer_rank
        self.e_out = layout.e_out
        self.tp_size = layout.tp
        self.n_node = layout.tp * layout.n_local
        # E_out x n_node: the global index of expert k * n_local + i of replica o (see _node_indices)
        o, k, i = torch.meshgrid(
            torch.arange(self.e_out), torch.arange(layout.tp), torch.arange(self.n_local), indexing='ij'
        )
        self.node_experts = (k * self.share + o * self.n_local + i).flatten().to(torch.cuda.current_device())
        # N: replica o and TP rank k owning each expert, as o * TP + k (one lookup in plan, not five kernels)
        slots = torch.arange(self.node_experts.numel(), device=self.node_experts.device)
        self.expert_owner = torch.empty_like(self.node_experts)
        self.expert_owner[self.node_experts] = slots // self.n_local
        self.tp = TPDispatcher(self.n_node, self.n_local, topk, backend, masked=True)

    def check_outer_group(self):
        """The tokens of replica o's experts go to rank o of the expert outer group: check that this rank
        owns them, and raise RuntimeError if not.

        A collective: one all-gather of a long over the expert outer group, which every rank of the group
        must run for the same dispatchers in the same order.
        """
        layout = self.layout
        world_size = torch.distributed.get_world_size(group=self.group)
        start = torch.tensor([layout.start], dtype=torch.long, device=torch.cuda.current_device())
        starts = torch.empty(world_size, dtype=torch.long, device=start.device)
        torch.distributed.all_gather_into_tensor(starts, start, group=self.group)
        starts = starts.tolist()
        expected = [layout.tp_rank * self.share + o * self.n_local for o in range(self.e_out)]
        if starts != expected:
            raise RuntimeError(
                f"the expert outer group (ranks {torch.distributed.get_process_group_ranks(self.group)}) does not "
                f"follow the expert layout: its ranks own the experts starting at {starts}, the dispatch sends "
                f"them those starting at {expected} ({layout})"
            )

    @property
    def group(self):
        # looked up when used, not kept, as TPDispatcher's
        return get_expert_outer_group()

    def _send_lists(self, routing_indices: Tensor, send_splits: List[int]) -> Tuple[Tensor, Tensor]:
        hits = self._hits(routing_indices)
        _, send_tokens, _, home_table = build_send_lists(hits, send_splits)
        return send_tokens, home_table

    def _hits(self, routing_indices: Tensor) -> Tensor:
        # T x E_out: whether a token picked an expert of each replica of the expert group
        dest = routing_indices.view(-1, self.topk) % self.share // self.n_local
        hits = torch.zeros(dest.size(0), self.e_out, dtype=torch.bool, device=dest.device)
        return hits.scatter_(1, dest, True)

    def _node_indices(self, indices: Tensor) -> Tensor:
        """The experts of this replica numbered k * n_local + i, the others n_node."""
        rest = indices % self.share
        mine = rest // self.n_local == self.outer_rank
        return torch.where(mine, indices // self.share * self.n_local + rest % self.n_local, self.n_node)

    def _build(self, routing_indices: Tensor, state: NodeState, tp_plan: DispatchPlan) -> DispatchPlan:
        send_tokens, home_table = self._send_lists(routing_indices, state.send_splits)
        outer = OuterPlan(send_tokens, home_table, state.send_splits, state.recv_splits)
        return DispatchPlan(tp_plan.tp, tp_plan.group_sizes, tp_plan.n_tokens, state, outer)

    def _counts(self, routing_indices: Tensor, tokens_per_expert: Tensor) -> Tuple[Tensor, Tensor]:
        """T x E_out hits (see _hits), and E_out x (1 + n_node + TP) counts: for the tokens sent to each replica,
        their number, then what that replica's TPDispatcher.counts reads from them: the tokens of each of its
        n_node experts, numbered as in _node_indices, and the tokens of each of its TP ranks."""
        indices = routing_indices.view(-1, self.topk)
        # T x K: replica o, TP rank k owning each slot's expert, as o * TP + k
        owner = self.expert_owner[indices]
        # T x (E_out x TP): whether a token picked an expert of each GPU of the expert group
        rank_hits = torch.zeros(indices.size(0), self.e_out * self.tp_size, dtype=torch.bool, device=indices.device)
        rank_hits.scatter_(1, owner, True)
        hits = rank_hits.view(-1, self.e_out, self.tp_size).any(dim=2)
        counts = torch.cat([
            hits.sum(dim=0).unsqueeze(1),
            # every slot of an expert goes to its replica: its tokens there are its tokens here
            tokens_per_expert[self.node_experts].view(self.e_out, self.n_node),
            rank_hits.sum(dim=0).view(self.e_out, self.tp_size),
        ], dim=1)
        return hits, counts

    def plan(self, routing_indices: Tensor, tokens_per_expert: Tensor) -> DispatchPlan:
        """routing_indices: T*K flat expert indices; tokens_per_expert: the n_experts counts of the indices.

        One host sync: each replica sends the others, with the number of tokens it sends them, the counts their
        TP plan needs, so it is built without reading them back from the received routing."""
        hits, send_counts = self._counts(routing_indices, tokens_per_expert)
        ones = [1] * self.e_out
        # E_out x (1 + n_node + TP): row o from replica o
        recv_counts, handle = all_to_all(send_counts, ones, ones, self.group, self.e_out)
        handle.wait()
        counts = torch.cat([send_counts[:, 0], recv_counts[:, 0], recv_counts[:, 1:].sum(dim=0)]).tolist()
        send_splits, recv_splits = counts[:self.e_out], counts[self.e_out:2 * self.e_out]
        expert_counts, dest_counts = counts[2 * self.e_out:-self.tp_size], counts[-self.tp_size:]
        _, send_tokens, _, home_table = build_send_lists(hits, send_splits)

        # the routing of the tokens each replica receives
        send_indices = routing_indices.view(-1, self.topk).index_select(0, send_tokens).int()
        recv_indices, handle = all_to_all(send_indices, send_splits, recv_splits, self.group, self.e_out)
        handle.wait()
        # M*K
        node_indices = self._node_indices(recv_indices.long()).flatten()
        if self.check_plan:
            tokens_per_expert = torch.bincount(node_indices, minlength=self.n_node + 1)[:self.n_node]
            expected = self.tp.counts(node_indices, tokens_per_expert)
            if (expert_counts, dest_counts) != expected:
                raise RuntimeError(
                    f"the counts sent with the tokens {(expert_counts, dest_counts)} are not those of their routing "
                    f"{expected}"
                )
        tp_plan = self.tp.plan_from_counts(node_indices, expert_counts, dest_counts)

        # int32: half the memory, kept until backward
        state = NodeState(send_splits, recv_splits, node_indices.int(), tp_plan.state)
        outer = OuterPlan(send_tokens, home_table, send_splits, recv_splits)
        return DispatchPlan(tp_plan.tp, tp_plan.group_sizes, tp_plan.n_tokens, state, outer)

    def replan(self, routing_indices: Tensor, state: NodeState) -> DispatchPlan:
        """The plan for the same routing, rebuilt without communication or sync from its state."""
        return self._build(routing_indices, state, self.tp.replan(state.node_indices.long(), state.tp))

    def _to_replicas(self, x: Tensor, plan: DispatchPlan) -> Tuple[Tensor, Any, Tensor]:
        send = torch.index_select(x, 0, plan.outer.send_tokens)
        output, handle = all_to_all(send, plan.outer.send_splits, plan.outer.recv_splits, self.group, self.e_out)
        return output, handle, send

    def _from_replicas(self, y: Tensor, plan: DispatchPlan) -> Pending:
        output, handle = all_to_all(y, plan.outer.recv_splits, plan.outer.send_splits, self.group, self.e_out)
        # T rows: add up the parts returned by each replica
        finish = lambda output: masked_gather_sum(output, plan.outer.home_table, None, self.backend)
        return Pending(output, handle, finish)

    def dispatch(self, x: Tensor, plan: DispatchPlan) -> Pending:
        output, handle, send = self._to_replicas(x, plan)
        # M x D/TP received, then n x D inside the TP group
        finish = lambda output: self.tp.dispatch(output, plan).wait()
        return Pending(output, handle, finish, keep=send)

    def combine(self, y: Tensor, plan: DispatchPlan) -> Pending:
        # M x D/TP: summed over this replica's experts, then sent back
        return self._from_replicas(self.tp.combine(y, plan).wait(), plan)

    def local_scores(self, scores: Tensor, plan: DispatchPlan) -> Tensor:
        return _LocalScores.apply(scores, self, plan)

    def send_scores(self, scores: Tensor, plan: DispatchPlan) -> Tensor:
        """T x K -> M x K, the scores of the received tokens. Use local_scores for autograd."""
        output, handle, _ = self._to_replicas(scores.reshape(-1, self.topk), plan)
        handle.wait()
        return output

    def scores_grad(self, grad: Tensor, plan: DispatchPlan) -> Tensor:
        """M x K -> T x K: each slot's gradient comes from the replica that computed it, the others add 0."""
        return self._from_replicas(grad.reshape(-1, self.topk).contiguous(), plan).wait()


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


class _LocalScores(torch.autograd.Function):

    @staticmethod
    def forward(ctx, scores: Tensor, dispatcher: NodeDispatcher, plan: DispatchPlan) -> Tensor:
        _save_plan(ctx, dispatcher, plan)
        ctx.shape = scores.shape
        return dispatcher.send_scores(scores, plan)

    @staticmethod
    def backward(ctx, grad: Tensor):
        return ctx.dispatcher.scores_grad(grad, _saved_plan(ctx)).view(ctx.shape), None, None


def build_dispatcher(layout: ExpertLayout, topk: int, backend: str):
    """The dispatcher of the calling rank's expert layout.

    With E_out > 1 this is a collective over the expert outer group (NodeDispatcher.check_outer_group): every
    rank of the group must build the same MoE layers in the same order, as building the same model does.
    """
    if layout.e_out == 1:
        return TPDispatcher(layout.n_experts, layout.n_local, topk, backend)
    dispatcher = NodeDispatcher(layout, topk, backend)
    dispatcher.check_outer_group()
    return dispatcher


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
