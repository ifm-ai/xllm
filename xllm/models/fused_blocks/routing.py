from typing import Optional, Tuple
import torch
from torch import Tensor

from xllm.modules.moe.router import (
    dynamic_qb_combine,
    dynamic_qb_routing_scores,
    dynamic_qb_select,
    dynamic_qb_weights_fwd,
    dynamic_qb_weights_bwd,
    update_dynamic_qb_bias,
)
from .utils import (
    fused_permute_y_fwd,
    fused_permute_y_bwd_y_grad,
    fused_permute_y_bwd_scores_grad,
)
from .distributed import all_reduce


def routing_fwd(
    original_scores: Tensor,
    router_bias: Optional[Tensor],
    routing_type: str,
    routing_score_func: str,
    routing_scaling_factor: Optional[float],
    num_tokens: int,
    n_experts: int,
    topk: int,
    qb_globalnorm: Optional[bool],
    dtype: torch.dtype,
) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:
    if routing_type == 'topk':
        routing_scores = original_scores
        if router_bias is not None:
            routing_scores = routing_scores + router_bias.to(routing_scores)
        # B x L x K
        routing_indices = torch.topk(routing_scores, topk, dim=-1)[1]
        org_topk_scores = torch.gather(original_scores, dim=-1, index=routing_indices)
        token_indices = None
        # N
        tokens_per_expert = torch.bincount(routing_indices.flatten(), minlength=n_experts)
    elif routing_type == 'dynamic_qb':
        routing_scores, _ = dynamic_qb_routing_scores(original_scores, routing_score_func, qb_globalnorm)
        # routes, grouped by expert
        routing_indices, token_indices, org_topk_scores = dynamic_qb_select(routing_scores, router_bias)
        # N
        tokens_per_expert = torch.bincount(routing_indices, minlength=n_experts)
    else:
        raise ValueError(f"Unknown routing type: {routing_type}.")

    topk_scores = routing_weights_fwd(
        org_topk_scores, token_indices, routing_type, routing_scaling_factor, num_tokens, topk, qb_globalnorm, dtype
    )
    return routing_indices, token_indices, org_topk_scores, topk_scores, tokens_per_expert


def routing_weights_fwd(
    org_topk_scores: Tensor,
    token_indices: Optional[Tensor],
    routing_type: str,
    routing_scaling_factor: Optional[float],
    num_tokens: int,
    topk: int,
    qb_globalnorm: Optional[bool],
    dtype: torch.dtype,
) -> Tensor:
    if routing_type == 'topk':
        topk_scores = org_topk_scores / org_topk_scores.sum(dim=-1, keepdim=True) if topk > 1 else org_topk_scores
        return (topk_scores * routing_scaling_factor if routing_scaling_factor else topk_scores).to(dtype)
    elif routing_type == 'dynamic_qb':
        # kept in fp32
        return dynamic_qb_weights_fwd(
            org_topk_scores, token_indices, num_tokens, not qb_globalnorm and topk > 1, routing_scaling_factor
        )
    else:
        raise ValueError(f"Unknown routing type: {routing_type}.")


def routing_bwd(
    topk_scores_grad: Tensor,
    original_scores: Tensor,
    routing_indices: Tensor,
    token_indices: Optional[Tensor],
    org_topk_scores: Tensor,
    routing_type: str,
    routing_score_func: str,
    routing_scaling_factor: Optional[float],
    topk: int,
    qb_globalnorm: Optional[bool],
) -> Tensor:
    if routing_type == 'topk':
        topk_scores_grad = topk_scores_grad.float()
        if routing_scaling_factor is not None:
            topk_scores_grad = topk_scores_grad * routing_scaling_factor
        if topk > 1:
            topk_scores_denom = org_topk_scores.sum(dim=-1, keepdim=True)
            topk_scores = org_topk_scores / topk_scores_denom
            topk_scores_grad = (topk_scores_grad - (topk_scores_grad * topk_scores).sum(dim=-1, keepdim=True)) / topk_scores_denom
        return torch.zeros_like(original_scores).scatter(-1, routing_indices, topk_scores_grad)
    elif routing_type == 'dynamic_qb':
        routing_scores, routing_scores_denom = dynamic_qb_routing_scores(
            original_scores, routing_score_func, qb_globalnorm
        )
        return dynamic_qb_weights_bwd(
            topk_scores_grad, routing_scores, routing_scores_denom, org_topk_scores,
            routing_indices, token_indices, not qb_globalnorm and topk > 1, routing_scaling_factor,
        )
    else:
        raise ValueError(f"Unknown routing type: {routing_type}.")


def routing_permutation(
    routing_indices: Tensor,
    token_indices: Optional[Tensor],
    routing_type: str,
    topk: int,
) -> Tuple[Optional[Tensor], Tensor]:
    if routing_type == 'topk':
        # B*L*K
        sorted_indices = torch.argsort(routing_indices.flatten(), stable=True)
        return sorted_indices, sorted_indices // topk
    elif routing_type == 'dynamic_qb':
        # routes are already grouped by expert
        return None, token_indices
    else:
        raise ValueError(f"Unknown routing type: {routing_type}.")


def permute_y_fwd(
    sorted_indices: Optional[Tensor],
    token_indices: Optional[Tensor],
    y: Tensor,
    routing_scores: Tensor,
    routing_type: str,
    bsz: int,
    slen: int,
    topk: int,
    backend: str
) -> Tuple[Tensor, Optional[Tensor]]:
    if routing_type == 'topk':
        return fused_permute_y_fwd(sorted_indices, y, routing_scores, bsz, slen, topk, backend)
    elif routing_type == 'dynamic_qb':
        # B x L x D/TP
        return dynamic_qb_combine(y, token_indices, bsz * slen, routing_scores).view(bsz, slen, -1), None
    else:
        raise ValueError(f"Unknown routing type: {routing_type}.")


def permute_y_bwd_y_grad(
    out_grad: Tensor,
    sorted_indices: Optional[Tensor],
    token_indices: Optional[Tensor],
    routing_scores: Tensor,
    routing_type: str,
    bsz: int,
    slen: int,
    topk: int,
    backend: str
) -> Tuple[Tensor, Optional[Tensor]]:
    if routing_type == 'topk':
        return fused_permute_y_bwd_y_grad(out_grad, sorted_indices, routing_scores, bsz, slen, topk, backend)
    elif routing_type == 'dynamic_qb':
        # routes x D/TP
        y_grad = torch.index_select(out_grad.view(bsz * slen, -1), 0, token_indices)
        return y_grad * routing_scores.to(y_grad).unsqueeze(1), None
    else:
        raise ValueError(f"Unknown routing type: {routing_type}.")


def permute_y_bwd_scores_grad(
    out_grad: Tensor,
    inverse_indices: Optional[Tensor],
    token_indices: Optional[Tensor],
    y: Tensor,
    routing_type: str,
    bsz: int,
    slen: int,
    topk: int,
    backend: str
) -> Tensor:
    if routing_type == 'topk':
        return fused_permute_y_bwd_scores_grad(out_grad, inverse_indices, y, bsz, slen, topk, backend)
    elif routing_type == 'dynamic_qb':
        # routes x D/TP
        selected_out_grad = torch.index_select(out_grad.view(bsz * slen, -1), 0, token_indices)
        return (y * selected_out_grad).sum(dim=1, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown routing type: {routing_type}.")


def permute_x_bwd(
    xp_grad: Tensor,
    inverse_indices: Optional[Tensor],
    token_indices: Optional[Tensor],
    routing_type: str,
    bsz: int,
    slen: int,
    topk: int,
) -> Tensor:
    if routing_type == 'topk':
        # re-order
        xp_grad = torch.index_select(xp_grad, 0, inverse_indices)
        # B x L x K x D/TP -> B x L x D/TP
        return xp_grad.view(bsz, slen, topk, -1).sum(dim=2)
    elif routing_type == 'dynamic_qb':
        # B x L x D/TP
        return dynamic_qb_combine(xp_grad, token_indices, bsz * slen).view(bsz, slen, -1)
    else:
        raise ValueError(f"Unknown routing type: {routing_type}.")


def router_bias_reduce(
    tokens_per_expert: Tensor,
    router_bias: Optional[Tensor],
    routing_type: str,
):
    # reduce tokens/expert for top-k router bias updating
    if routing_type == 'topk' and router_bias is not None:
        return all_reduce(tokens_per_expert, parallel_region='hybrid_shard', async_op=True)
    return tokens_per_expert, None


def router_bias_update(
    router_bias: Optional[Tensor],
    tokens_per_expert: Tensor,
    handle_bias,
    original_scores: Tensor,
    routing_type: str,
    router_bias_update_rate: Optional[float],
    routing_score_func: str,
    n_experts: int,
    topk: int,
    qb_update_method: Optional[str],
    qb_decay: Optional[float],
    qb_globalnorm: Optional[bool],
) -> None:
    if router_bias is None:
        return
    if routing_type == 'topk':
        if handle_bias is not None:
            handle_bias.wait()
        average_tokens = tokens_per_expert.sum() / n_experts
        bias_update = torch.sign(average_tokens - tokens_per_expert)
        router_bias.add_(bias_update, alpha=router_bias_update_rate)
    elif routing_type == 'dynamic_qb':
        routing_scores, _ = dynamic_qb_routing_scores(original_scores, routing_score_func, qb_globalnorm)
        with torch.no_grad():
            update_dynamic_qb_bias(router_bias, routing_scores, topk, qb_update_method, qb_decay)
    else:
        raise ValueError(f"Unknown routing type: {routing_type}.")
