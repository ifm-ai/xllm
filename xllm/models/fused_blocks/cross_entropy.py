from typing import NamedTuple, Optional, Tuple
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from xllm.distributed import (
    get_model_parallel_group,
    get_model_parallel_rank,
    get_model_parallel_world_size,
)
from xllm.distributed.utils import reshape_gathered_tensor_along_specific_dim
from .distributed import (
    all_gather,
    all_reduce,
    reduce_scatter,
)

NUM_BLOCKS_PER_CHUNK = 1


class LogitsStats(NamedTuple):
    sum: Tensor
    count: Tensor
    minimum: Tensor
    maximum: Tensor


class FusedCrossEntropyOutput(NamedTuple):
    loss: Tensor
    grad_input: Optional[Tensor]
    grad_weight: Optional[Tensor]
    logits_stats: Optional[LogitsStats]


def next_power_of_2(n):
    if n <= 0:
        return 1  # Handle non-positive numbers
    exponent = math.ceil(math.log2(n))
    return 2 ** exponent


def _fused_linear_cross_entropy(
    input_: Tensor,  # [batch_size, seq_len, feature_dim]
    weight: Tensor,  # [vocab_size, feature_dim]
    target: Tensor,  # [batch_size, seq_len]
    mask: Optional[Tensor],  # [batch_size, seq_len]
    class_weight: Optional[Tensor],
    ignore_index: int,
    label_smoothing: float,
    compute_loss: bool,
    compute_grad: bool,
    gather_input: bool,
    collect_logits_stats: bool = False,
) -> Tuple[
    Optional[Tensor], Optional[Tensor], Optional[Tensor],
    Optional[LogitsStats],
]:
    """
        Fusing the last linear layer with cross-entropy loss
            Reference1: https://github.com/mgmalek/efficient_cross_entropy
            Reference2: https://github.com/linkedin/Liger-Kernel
    """
    if class_weight is not None:
        raise NotImplementedError("Weighted cross entropy has not been implemented.")
    if label_smoothing > 0.0:
        raise NotImplementedError("Label smoothing has not been implemented.")
    assert ignore_index < 0, 'ignore index should be negative'
    assert compute_loss or compute_grad

    model_parallel_size = get_model_parallel_world_size()
    rank = get_model_parallel_rank()

    bsz, seq_len, mdim = input_.shape
    partial_voc_size = weight.shape[0]
    if gather_input:
        mdim = mdim * model_parallel_size
        x, handle_x = all_gather(input_, parallel_region='model', async_op=True)
    else:
        x = input_
        handle_x = None

    inc_factor = math.ceil(partial_voc_size / mdim)
    chunk_size = next_power_of_2(math.ceil(bsz * seq_len / inc_factor) * NUM_BLOCKS_PER_CHUNK)
    num_chunks = math.ceil(bsz * seq_len / chunk_size)

    target_1d = target.clone()
    if mask is not None:
        target_1d[torch.logical_not(mask)] = ignore_index
    target_1d = target_1d.view(-1)

    ignore_index_mask = target_1d == ignore_index
    if model_parallel_size > 1:
        vocab_start_idx = partial_voc_size * rank
        vocab_end_idx = partial_voc_size * (rank + 1)
        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target_1d < vocab_start_idx) | (target_1d >= vocab_end_idx)
        target_1d = target_1d - vocab_start_idx
        target_1d[target_mask] = 0
    else:
        target_mask = ignore_index_mask
        target_1d[target_mask] = 0

    # TODO: enable when debugging
    # assert torch.all((target_1d >= 0) & (target_1d < partial_voc_size)), 'invalid target index'
    if compute_loss:
        loss = torch.empty(bsz, seq_len, dtype=torch.float32, device=input_.device)
        loss_flat = loss.view(bsz * seq_len)
    else:
        loss = loss_flat = None
    if compute_grad:
        grad_input = torch.empty(bsz * seq_len, input_.shape[-1], dtype=input_.dtype, device=input_.device)
        grad_weight = torch.zeros_like(weight)
        weight_fp32 = weight.float()
    else:
        grad_input = None
        grad_weight = None
        weight_fp32 = None

    if handle_x is not None:
        handle_x.wait()
        x = reshape_gathered_tensor_along_specific_dim(x, gather_dim=2)

    x_2d = x.view(-1, mdim)
    prev_handle = None
    prev_grad = None
    prev_start_index = None
    prev_end_index = None
    logits_sum = (
        torch.zeros((), dtype=torch.float32, device=input_.device)
        if collect_logits_stats else None
    )
    logits_count = (
        torch.logical_not(ignore_index_mask).sum() * partial_voc_size
        if collect_logits_stats else None
    )
    logits_minimum = (
        torch.full((), float("inf"), dtype=torch.float32, device=input_.device)
        if collect_logits_stats else None
    )
    logits_maximum = (
        torch.full((), -float("inf"), dtype=torch.float32, device=input_.device)
        if collect_logits_stats else None
    )
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, bsz * seq_len)
        # C x D
        inp_chunk = x_2d[start_idx:end_idx]
        # C
        target_1d_chunk = target_1d[start_idx:end_idx]
        target_mask_chunk = target_mask[start_idx:end_idx]
        ignore_index_mask_chunk = ignore_index_mask[start_idx:end_idx]
        # C x V
        partial_logits = F.linear(inp_chunk, weight).float()
        if logits_sum is not None:
            assert logits_minimum is not None and logits_maximum is not None
            token_sums = partial_logits.sum(dim=-1)
            token_sums.masked_fill_(ignore_index_mask_chunk, 0.0)
            logits_sum = logits_sum + token_sums.sum()
            token_logit_minimums, logits_max = torch.aminmax(partial_logits, dim=-1)
            token_logit_minimums.masked_fill_(ignore_index_mask_chunk, float("inf"))
            token_logit_maximums = logits_max.masked_fill(
                ignore_index_mask_chunk, -float("inf")
            )
            logits_minimum = torch.minimum(logits_minimum, token_logit_minimums.min())
            logits_maximum = torch.maximum(logits_maximum, token_logit_maximums.max())
        else:
            # C
            logits_max = torch.max(partial_logits, dim=-1)[0]
        if model_parallel_size > 1:
            torch.distributed.all_reduce(
                logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=get_model_parallel_group(),
            )
        partial_logits = partial_logits - logits_max.unsqueeze(1)
        # C
        arange_1d = torch.arange(partial_logits.shape[0], device=partial_logits.device)
        if compute_loss:
            predicted_logits = partial_logits[arange_1d, target_1d_chunk]
            predicted_logits = predicted_logits.clone().contiguous()
            predicted_logits[target_mask_chunk] = 0.0
            predicted_logits, handle_pred = all_reduce(predicted_logits, parallel_region='model', async_op=True)
        else:
            predicted_logits, handle_pred = None, None

        # C x V
        exp_logits = torch.exp(partial_logits, out=partial_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)

        if handle_pred is not None:
            handle_pred.wait()

        sum_exp_logits, _ = all_reduce(sum_exp_logits, parallel_region='model', async_op=False)

        if compute_loss:
            loss_flat[start_idx:end_idx] = torch.log(sum_exp_logits) - predicted_logits

        if compute_grad:
            # softmax scores C x V
            scores = torch.div(exp_logits, sum_exp_logits.unsqueeze(dim=-1), out=exp_logits)
            score_update = 1.0 - target_mask_chunk.float()
            scores[arange_1d, target_1d_chunk] -= score_update
            # zero out ignored index
            scores.masked_fill_(ignore_index_mask_chunk.unsqueeze(1), value=0.0)
            # (C x V) x (V x D) -> C x D
            curr_grad = torch.mm(scores, weight_fp32)

            if prev_grad is not None:
                if prev_handle is not None:
                    prev_handle.wait()
                grad_input[prev_start_index:prev_end_index] = prev_grad

            if gather_input:
                prev_grad, prev_handle = reduce_scatter(curr_grad, parallel_region='model', async_op=True)
            else:
                prev_grad = curr_grad
                prev_handle = None
            prev_start_index = start_idx
            prev_end_index = end_idx

            grad_weight = torch.addmm(
                input=grad_weight,
                mat1=scores.t().to(dtype=x.dtype),
                mat2=inp_chunk,
                out=grad_weight,
            )

    if prev_grad is not None:
        if prev_handle is not None:
            prev_handle.wait()
        grad_input[prev_start_index:prev_end_index] = prev_grad
        grad_input = grad_input.view(bsz, seq_len, -1)

    if compute_loss:
        loss_flat[ignore_index_mask] = 0.0

    logits_stats = None
    if logits_sum is not None:
        assert logits_count is not None
        assert logits_minimum is not None and logits_maximum is not None
        logits_stats = LogitsStats(
            logits_sum, logits_count, logits_minimum, logits_maximum,
        )
    return loss, grad_input, grad_weight, logits_stats


def _fused_linear_cross_entropy_bwd(
    loss_grad: Tensor,
    grad_input: Optional[Tensor],
    grad_weight: Optional[Tensor],
    input_: Optional[Tensor],  # [batch_size, seq_len, feature_dim]
    weight: Optional[Tensor],  # [vocab_size, feature_dim]
    target: Optional[Tensor],  # [batch_size, seq_len]
    mask: Optional[Tensor],  # [batch_size, seq_len]
    class_weight: Optional[Tensor],
    ignore_index: int,
    label_smoothing: float,
    gather_input: bool,
) -> Tuple[Tensor, Tensor]:
    # TODO: enable when debugging
    # assert torch.all(loss_grad == loss_grad[0, 0])
    # assert loss_grad.dim() == 2
    if grad_input is None:
        _, grad_input, grad_weight, _ = _fused_linear_cross_entropy(
            input_, weight, target, mask, class_weight, ignore_index, label_smoothing, False, True, gather_input
        )

    grad_input = torch.mul(grad_input, loss_grad[0, 0], out=grad_input)
    grad_weight = torch.mul(grad_weight, loss_grad[0, 0], out=grad_weight)

    return grad_input, grad_weight


def fused_linear_cross_entropy_fwd(
    input_: Tensor,
    weight: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    class_weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    compute_grad: bool = False,
    gather_input: bool = True,
    collect_logits_stats: bool = False,
) -> FusedCrossEntropyOutput:
    loss, grad_input, grad_weight, logits_stats = _fused_linear_cross_entropy(
        input_=input_,
        weight=weight,
        target=target,
        mask=mask,
        class_weight=class_weight,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        compute_loss=True,
        compute_grad=compute_grad,
        gather_input=gather_input,
        collect_logits_stats=collect_logits_stats,
    )
    assert loss is not None
    return FusedCrossEntropyOutput(loss, grad_input, grad_weight, logits_stats)


def fused_linear_cross_entropy_bwd(
    loss_grad: Tensor,
    grad_input: Optional[Tensor],
    grad_weight: Optional[Tensor],
    input_: Optional[Tensor],
    weight: Optional[Tensor],
    target: Optional[Tensor],
    mask: Optional[Tensor] = None,
    class_weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    gather_input: bool = True,
) -> Tuple[Tensor, Tensor]:
    return _fused_linear_cross_entropy_bwd(
        loss_grad, grad_input, grad_weight, input_, weight, target, mask, class_weight, ignore_index, label_smoothing, gather_input
    )
