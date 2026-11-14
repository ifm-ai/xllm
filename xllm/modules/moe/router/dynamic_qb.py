from typing import Callable, Optional, Tuple
from functools import partial
import math

import torch
import torch.nn.functional as F
import torch.nn.init as init

from xllm.distributed import (
    get_model_parallel_group,
    get_model_parallel_world_size,
    get_hybrid_shard_data_parallel_group,
    get_hybrid_shard_data_parallel_world_size
)
from .base import all_reduce, calc_probs, BaseRouter


def dynamic_qb_initial_bias(
    weight_std: float,
    in_features: int,
    num_experts: int,
    topk: int,
    score_func: str,
    globalnorm: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    q = 1.0 - topk / num_experts
    if q <= 0.0:
        return torch.zeros(num_experts, device=device, dtype=dtype)

    sigma = weight_std * math.sqrt(in_features)
    logit_q = sigma * torch.special.ndtri(torch.tensor(q, dtype=torch.float64))
    if score_func == 'sigmoid':
        value = torch.sigmoid(logit_q)
        if globalnorm:
            value = value / (0.5 * num_experts)
    elif score_func == 'softmax':
        qs = torch.arange(1, num_experts + 1, dtype=torch.float64) / (num_experts + 1)
        value = torch.exp(logit_q) / torch.exp(sigma * torch.special.ndtri(qs)).sum()
    else:
        raise ValueError(f"Unknown score function: {score_func}.")
    return torch.full((num_experts,), value.item(), device=device, dtype=dtype)


def _select_none_or_all_threshold(
    target: int, min_scores: torch.Tensor, max_scores: torch.Tensor
) -> torch.Tensor:
    # thresholds of the strict "score > bias" gate when no token or every token is selected
    if target <= 0:
        return torch.nextafter(max_scores, torch.full_like(max_scores, math.inf))
    return torch.nextafter(min_scores, torch.full_like(min_scores, -math.inf))


def dynamic_qb_quantile(scores: torch.Tensor, topk: int) -> torch.Tensor:
    flat_scores = scores.flatten(end_dim=-2)
    num_tokens, num_experts = flat_scores.shape
    target = math.floor(num_tokens * topk / num_experts)
    if not 0 < target < num_tokens:
        return _select_none_or_all_threshold(
            target, flat_scores.min(dim=0).values, flat_scores.max(dim=0).values
        )

    # The (target + 1)-th largest score of each expert; top-k over contiguous rows is much faster than kthvalue.
    return torch.topk(flat_scores.t().contiguous(), target + 1, dim=1, sorted=False).values.min(dim=1).values


def global_score_bounds(
    flat_scores: torch.Tensor,
    world_size: int,
    group: torch.distributed.ProcessGroup,
) -> Tuple[torch.Tensor, torch.Tensor]:
    min_scores, max_scores = torch.aminmax(flat_scores, dim=0)
    # one MAX all-reduce for both bounds
    bounds = torch.stack([-min_scores, max_scores])
    if world_size > 1:
        torch.distributed.all_reduce(bounds, op=torch.distributed.ReduceOp.MAX, group=group)
    return -bounds[0], bounds[1]


def dynamic_qb_binary_search(
    scores: torch.Tensor,
    topk: int,
    world_size: int,
    group: torch.distributed.ProcessGroup,
    num_iters: int = 16,
) -> torch.Tensor:
    flat_scores = scores.flatten(end_dim=-2)
    num_tokens, num_experts = flat_scores.shape
    # every rank has the same number of tokens
    total_tokens = num_tokens * world_size
    target = math.floor(total_tokens * topk / num_experts)
    min_scores, max_scores = global_score_bounds(flat_scores, world_size, group)
    if not 0 < target < total_tokens:
        return _select_none_or_all_threshold(target, min_scores, max_scores)

    lo = min_scores
    hi = max_scores
    for _ in range(num_iters):
        mid = (lo + hi) * 0.5
        counts = (flat_scores > mid.view(1, num_experts)).sum(dim=0)
        counts, _ = all_reduce(counts, world_size, group)
        too_many = counts > target
        lo = torch.where(too_many, mid, lo)
        hi = torch.where(too_many, hi, mid)
    return hi


def dynamic_qb_histogram(
    scores: torch.Tensor,
    topk: int,
    world_size: int,
    group: torch.distributed.ProcessGroup,
    num_bins: int = 256,
) -> torch.Tensor:
    flat_scores = scores.flatten(end_dim=-2)
    num_tokens, num_experts = flat_scores.shape
    # every rank has the same number of tokens
    total_tokens = num_tokens * world_size
    target = math.floor(total_tokens * topk / num_experts)
    min_scores, max_scores = global_score_bounds(flat_scores, world_size, group)
    if not 0 < target < total_tokens:
        return _select_none_or_all_threshold(target, min_scores, max_scores)

    score_range = max_scores - min_scores
    denom = score_range.clamp_min(torch.finfo(flat_scores.dtype).tiny)
    # bin of every score, offset by its expert; in place, since the T x N temporaries dominate the cost
    bins = (flat_scores - min_scores.view(1, num_experts)).div_(denom.view(1, num_experts)).mul_(num_bins)
    bins = bins.to(torch.int32).clamp_(0, num_bins - 1)
    bins.add_(torch.arange(0, num_experts * num_bins, num_bins, device=scores.device, dtype=torch.int32))
    # N x bins
    hist = torch.bincount(bins.flatten(), minlength=num_experts * num_bins).view(num_experts, num_bins)
    hist, _ = all_reduce(hist, world_size, group)

    # interpolate the (target + 1)-th largest score within its bin
    rank = target + 1
    desc_hist = hist.flip(dims=[1])
    desc_cumsum = desc_hist.cumsum(dim=1)
    boundary_desc = (desc_cumsum >= rank).long().argmax(dim=1)
    expert_indices = torch.arange(num_experts, device=scores.device)
    prev_cumsum = torch.where(
        boundary_desc > 0,
        desc_cumsum[expert_indices, boundary_desc - 1],
        torch.zeros(num_experts, device=scores.device, dtype=desc_cumsum.dtype),
    )
    boundary_count = desc_hist[expert_indices, boundary_desc].clamp_min(1)
    rank_in_bin = (rank - prev_cumsum).to(scores)
    fraction_from_right = rank_in_bin / boundary_count.to(scores)
    boundary_asc = (num_bins - 1 - boundary_desc).to(scores)
    bin_width = score_range / num_bins
    bin_right = min_scores + (boundary_asc + 1.0) * bin_width
    return bin_right - fraction_from_right * bin_width


def update_dynamic_qb_bias(
    bias: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
    update_method: str,
    decay: float,
) -> None:
    if update_method == 'average':
        next_bias = dynamic_qb_quantile(scores, topk).to(bias)
        # Averaged over all ranks: TP ranks hold the same scores, and a single
        # communicator keeps the sum bit-identical on them.
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        next_bias, _ = all_reduce(next_bias, world_size, torch.distributed.group.WORLD)
        next_bias = next_bias / world_size
    elif update_method in ['histogram', 'binary_search']:
        # counts and min/max reductions are exact, so every TP rank gets the same result
        world_size = get_hybrid_shard_data_parallel_world_size()
        group = get_hybrid_shard_data_parallel_group()
        if update_method == 'histogram':
            next_bias = dynamic_qb_histogram(scores, topk, world_size, group)
        else:
            next_bias = dynamic_qb_binary_search(scores, topk, world_size, group)
        next_bias = next_bias.to(bias)
    else:
        raise ValueError(f"Unknown dynamic QB update method: {update_method}.")
    bias.lerp_(next_bias, 1.0 - decay)


def dynamic_qb_routing_scores(
    original_scores: torch.Tensor, score_func: str, globalnorm: bool
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # scores compared against the thresholds, and their denominator under globalnorm
    if globalnorm:
        return calc_probs(original_scores, score_func)
    return original_scores, None


def dynamic_qb_select(
    routing_scores: torch.Tensor, bias: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Routes with scores above the expert thresholds, grouped by expert, which is the dispatch order.
    num_experts = routing_scores.shape[-1]
    # B*L x N
    flat_scores = routing_scores.view(-1, num_experts)
    active_mask = flat_scores.t() > bias.to(flat_scores).view(num_experts, 1)
    expert_indices, token_indices = torch.nonzero(active_mask, as_tuple=True)
    return expert_indices, token_indices, flat_scores[token_indices, expert_indices]


def dynamic_qb_normalize(
    scores: torch.Tensor, token_indices: torch.Tensor, num_tokens: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Scores of tokens with multiple routes sum to one; single routes keep their score.
    routes_per_token = torch.bincount(token_indices, minlength=num_tokens)
    multi_route = routes_per_token[token_indices] > 1
    score_sums = torch.zeros(num_tokens, device=scores.device, dtype=scores.dtype)
    score_sums.index_add_(0, token_indices, scores)
    denom = torch.where(multi_route, score_sums[token_indices], 1.0)
    return scores / denom, denom, multi_route


def dynamic_qb_weights_fwd(
    selected_scores: torch.Tensor,
    token_indices: torch.Tensor,
    num_tokens: int,
    normalize: bool,
    scaling_factor: Optional[float],
) -> torch.Tensor:
    # fp32 combine weights of the routes
    weights = dynamic_qb_normalize(selected_scores, token_indices, num_tokens)[0] if normalize else selected_scores
    return weights * scaling_factor if scaling_factor is not None else weights


def dynamic_qb_weights_bwd(
    weights_grad: torch.Tensor,
    routing_scores: torch.Tensor,
    routing_scores_denom: Optional[torch.Tensor],
    selected_scores: torch.Tensor,
    expert_indices: torch.Tensor,
    token_indices: torch.Tensor,
    normalize: bool,
    scaling_factor: Optional[float],
) -> torch.Tensor:
    # backward of dynamic_qb_weights_fwd and dynamic_qb_routing_scores
    num_experts = routing_scores.shape[-1]
    num_tokens = routing_scores.numel() // num_experts
    selected_grad = weights_grad.float()
    if scaling_factor is not None:
        selected_grad = selected_grad * scaling_factor
    if normalize:
        normalized_scores, denom, multi_route = dynamic_qb_normalize(selected_scores, token_indices, num_tokens)
        correction = torch.zeros(num_tokens, device=selected_grad.device, dtype=selected_grad.dtype)
        correction.index_add_(0, token_indices, selected_grad * normalized_scores * multi_route)
        selected_grad = (selected_grad - correction[token_indices]) / denom

    routing_scores_grad = torch.zeros_like(routing_scores)
    routing_scores_grad.view(num_tokens, num_experts)[token_indices, expert_indices] = selected_grad
    if routing_scores_denom is None:
        return routing_scores_grad
    # calc_probs backward
    return (
        routing_scores_grad - (routing_scores_grad * routing_scores).sum(dim=-1, keepdim=True)
    ) / routing_scores_denom


# elements per chunk in dynamic_qb_combine: 1 GiB of fp32 routes, plus 0.5 GiB for their bf16 gather
_COMBINE_CHUNK_NUMEL = 268_435_456


def _token_chunks(lengths: torch.Tensor, num_routes: int, max_routes: int):
    # (first token, end token, first route, end route) of consecutive token ranges, each holding
    # fewer than max_routes routes plus the routes of one token
    num_tokens = lengths.numel()
    if num_routes <= max_routes:
        return [(0, num_tokens, 0, num_routes)]
    offsets = torch.cumsum(lengths, 0)
    targets = torch.arange(max_routes, num_routes, max_routes, device=lengths.device)
    token_ends = torch.searchsorted(offsets, targets, right=True).clamp_(1, num_tokens)
    # one host sync for both ends
    token_ends, route_ends = torch.stack([token_ends, offsets[token_ends - 1]]).tolist()
    chunks, t0, r0 = [], 0, 0
    for t1, r1 in zip(token_ends, route_ends):
        if t1 > t0:
            chunks.append((t0, t1, r0, r1))
            t0, r0 = t1, r1
    if t0 < num_tokens:
        chunks.append((t0, num_tokens, r0, num_routes))
    return chunks


def dynamic_qb_combine(
    route_out: torch.Tensor,
    token_indices: torch.Tensor,
    num_tokens: int,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Sums the (weighted) routes of each token in fp32, since summing into bf16 rounds after every route.
    # The routes come grouped by expert; segment_reduce sums them in token order, so the result is
    # deterministic (index_add_ accumulates with atomics, in a different order every run).
    num_routes, dim = route_out.shape
    order = torch.argsort(token_indices, stable=True)
    lengths = torch.bincount(token_indices, minlength=num_tokens)
    combined = torch.empty(num_tokens, dim, device=route_out.device, dtype=route_out.dtype)
    max_routes = max(1, _COMBINE_CHUNK_NUMEL // max(dim, 1))
    for t0, t1, r0, r1 in _token_chunks(lengths, num_routes, max_routes):
        idx = order[r0:r1]
        if weights is None:
            routes = route_out[idx].float()
        else:
            routes = route_out[idx] * weights[idx].float().unsqueeze(1)
        combined[t0:t1] = torch.segment_reduce(routes, 'sum', lengths=lengths[t0:t1], axis=0, unsafe=True)
        # free each fp32 chunk before allocating the next
        del routes
    return combined


class DynamicQBRouting(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        n_experts: int,
        topk: int,
        score_func: str,
        bias: torch.Tensor,
        update_method: str,
        decay: float,
        globalnorm: bool,
        scaling_factor: Optional[float],
    ):  # type: ignore

        world_size = get_model_parallel_world_size()
        group = get_model_parallel_group()

        bsz, seq_len, _ = x.shape

        # B x L x N
        logits = F.linear(x, weight).to(torch.float32)
        # All-reduce across all the partitions.
        logits, _ = all_reduce(logits, world_size, group, async_op=False)

        if score_func == 'softmax':
            score_fn = partial(F.softmax, dim=-1, dtype=torch.float32)
        elif score_func == 'sigmoid':
            score_fn = torch.sigmoid
        else:
            raise ValueError(f"Unknown score function: {score_func}.")

        original_scores = score_fn(logits)
        routing_scores, _ = dynamic_qb_routing_scores(original_scores, score_func, globalnorm)
        # routes, grouped by expert
        expert_indices, token_indices, selected_scores = dynamic_qb_select(routing_scores, bias)
        scores = dynamic_qb_weights_fwd(
            selected_scores, token_indices, bsz * seq_len, not globalnorm and topk > 1, scaling_factor
        )
        # N
        tokens_per_expert = torch.bincount(expert_indices, minlength=n_experts)

        ctx.save_for_backward(x, weight, original_scores, selected_scores, expert_indices, token_indices)
        # Backward updates the thresholds in place, so they are kept out of the saved tensors.
        ctx.bias = bias
        ctx.topk = topk
        ctx.score_func = score_func
        ctx.update_method = update_method
        ctx.decay = decay
        ctx.globalnorm = globalnorm
        ctx.scaling_factor = scaling_factor

        return scores, token_indices, tokens_per_expert

    @staticmethod
    def backward(
        ctx,
        scores_grad,
        token_indices_grad,
        tokens_per_expert_grad,
    ):  # type: ignore
        x, weight, original_scores, selected_scores, expert_indices, token_indices = ctx.saved_tensors
        bias = ctx.bias
        topk = ctx.topk
        score_func = ctx.score_func
        update_method = ctx.update_method
        decay = ctx.decay
        globalnorm = ctx.globalnorm
        scaling_factor = ctx.scaling_factor

        bsz, seq_len, _ = x.shape

        routing_scores, routing_scores_denom = dynamic_qb_routing_scores(original_scores, score_func, globalnorm)
        original_scores_grad = dynamic_qb_weights_bwd(
            scores_grad, routing_scores, routing_scores_denom, selected_scores, expert_indices, token_indices,
            not globalnorm and topk > 1, scaling_factor,
        )

        if score_func == 'softmax':
            logits_grad = torch.ops.aten._softmax_backward_data(
                original_scores_grad, original_scores, -1, torch.float32
            )
        elif score_func == 'sigmoid':
            logits_grad = torch.ops.aten.sigmoid_backward(original_scores_grad, original_scores)
        else:
            raise ValueError(f"Unknown score function: {score_func}.")

        # update router thresholds
        with torch.no_grad():
            update_dynamic_qb_bias(bias, routing_scores, topk, update_method, decay)

        world_size = get_model_parallel_world_size()
        group = get_model_parallel_group()
        logits_grad, handle_logits = all_reduce(logits_grad, world_size, group, async_op=True)

        if handle_logits is not None:
            handle_logits.wait()
        # B*L x N
        logits_grad = logits_grad.to(x).view(bsz * seq_len, -1)
        # B*L x D/TP
        x = x.view(bsz * seq_len, -1)
        # (N x B*L) x (B*L x D/TP) -> N x D/TP
        w_grad = logits_grad.t().matmul(x)
        # (B*L x N) x (N x D/TP) -> B*L x D/TP
        x_grad = logits_grad.matmul(weight)
        x_grad = x_grad.view(bsz, seq_len, -1)

        return x_grad, w_grad, None, None, None, None, None, None, None, None


dynamic_qb_routing = DynamicQBRouting.apply


class DynamicQBRouter(BaseRouter):
    """Dynamic quantile-balancing router.
    Similar to RowParallelLinear, with specific communication control.

    Arguments:
        in_features: first dimension of matrix A.
        topk: target average number of activated experts per token.
        update_method: estimator of the score quantiles (average, histogram, or binary_search).
        decay: EMA decay of the thresholds.
        globalnorm: If true, compare globally normalized scores against the thresholds.
        init_method: method to initialize weights.
    """

    routing_type = 'dynamic_qb'

    def __init__(
        self,
        in_features: int,
        num_experts: int,
        topk: int,
        update_method: str = 'average',
        decay: float = 0.3,
        globalnorm: bool = True,
        score_func: str = 'sigmoid',
        scaling_factor: Optional[float] = None,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
    ) -> None:
        super(DynamicQBRouter, self).__init__(in_features, num_experts, topk, score_func, scaling_factor)

        assert update_method in ['average', 'histogram', 'binary_search'], f"Unknown dynamic QB update method: {update_method}"
        assert 0.0 <= decay < 1.0, f"Invalid dynamic QB decay: {decay}"
        self.update_method = update_method
        self.decay = decay
        self.globalnorm = globalnorm
        # thresholds, kept in `bias` like the top-k selection bias
        self.register_buffer('bias', torch.empty(self.num_experts, dtype=torch.float32))

        # Initialize weight and thresholds.
        weight_std = self.init_weight(init_method)
        with torch.no_grad():
            self.bias.copy_(
                dynamic_qb_initial_bias(
                    weight_std, in_features, num_experts, topk, score_func, globalnorm,
                    device=self.bias.device, dtype=self.bias.dtype
                )
            )

    def forward(
        self, x: torch.Tensor, load_balancing_type: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert load_balancing_type is None, "Dynamic QB does not use auxiliary load-balancing loss."
        # routes, grouped by expert
        scores, token_indices, tokens_per_expert = dynamic_qb_routing(
            x, self.weight, self.num_experts, self.topk, self.score_func,
            self.bias, self.update_method, self.decay, self.globalnorm, self.scaling_factor
        )
        return scores, token_indices, tokens_per_expert, None

    def extra_repr(self) -> str:
        return 'in_features={} ({}), n_experts={}, target_k={}, score_func={}, update={}, decay={}, globalnorm={}, scaling_factor={}'.format(
            self.in_features, self.input_size_per_partition, self.num_experts, self.topk,
            self.score_func, self.update_method, self.decay, self.globalnorm, self.scaling_factor,
        )
