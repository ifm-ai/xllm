from typing import Callable, Optional, Tuple
from functools import partial
import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from xllm.distributed import (
    get_model_parallel_group,
    get_model_parallel_rank,
    get_model_parallel_world_size,
    get_hybrid_shard_data_parallel_group,
    get_hybrid_shard_data_parallel_world_size
)
from xllm.distributed.utils import divide_and_check_no_remainder


def all_reduce(x: torch.Tensor, world_size: int, group: torch.distributed.ProcessGroup, async_op: bool = False):
    if world_size == 1:
        return x, None
    # All-reduce.
    handle = torch.distributed.all_reduce(x, group=group, async_op=async_op)
    return x, handle


def calc_probs(scores: torch.Tensor, score_func: str):
    # B x L x N
    if score_func == 'softmax':
        denom = None
        probs = scores
    elif score_func == 'sigmoid':
        denom = scores.sum(dim=-1, keepdim=True)
        probs = scores / denom
    else:
        raise ValueError(f"Unknown score function: {score_func}.")

    return probs, denom


class TopKRouting(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        n_experts: int,
        topk: int,
        score_func: str,
        router_bias: Optional[torch.Tensor],
        bias_update_rate: Optional[float],
        scaling_factor: Optional[float],
        load_balancing_type: Optional[str] = None
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

        scores = score_fn(logits)
        original_scores = scores
        if router_bias is not None:
            scores = scores + router_bias.to(scores)
        # B x L x K
        indices = torch.topk(scores, topk, dim=-1)[1]
        scores = torch.gather(original_scores, dim=-1, index=indices)
        scores = scores / (scores.sum(dim=-1, keepdim=True)) if topk > 1 else scores
        if scaling_factor:
            scores = scores * scaling_factor
        scores = scores.to(x)
        # N
        tokens_per_expert = torch.bincount(indices.flatten(), minlength=n_experts)

        if load_balancing_type is None:
            aux_loss = None
        else:
            # B x L x N
            probs, _ = calc_probs(original_scores, score_func)
            prob_per_expert = probs.mean(dim=(0, 1))
            freq_per_expert = tokens_per_expert / (bsz * seq_len * topk)
            if load_balancing_type == 'dot':
                aux_loss = torch.dot(freq_per_expert, prob_per_expert) * n_experts
            elif load_balancing_type == 'entropy':
                aux_loss = torch.dot(freq_per_expert - (1.0 / n_experts), torch.log(prob_per_expert))
            else:
                raise ValueError(f"Unknown load balancing type: {load_balancing_type}.")

        ctx.save_for_backward(x, weight, original_scores, indices, router_bias, tokens_per_expert)
        ctx.n_experts = n_experts
        ctx.topk = topk
        ctx.score_func = score_func
        ctx.bias_update_rate = bias_update_rate
        ctx.scaling_factor = scaling_factor
        ctx.load_balancing_type = load_balancing_type

        return scores, indices, tokens_per_expert, aux_loss

    @staticmethod
    def backward(
        ctx,
        scores_grad,
        indices_grad,
        tokens_per_expert_grad,
        aux_loss_grad,
    ):  # type: ignore
        x, weight, original_scores, indices, router_bias, tokens_per_expert = ctx.saved_tensors
        n_experts = ctx.n_experts
        topk = ctx.topk
        score_func = ctx.score_func
        bias_update_rate = ctx.bias_update_rate
        scaling_factor = ctx.scaling_factor
        load_balancing_type = ctx.load_balancing_type

        bsz, seq_len, _ = x.shape

        if load_balancing_type is None:
            assert aux_loss_grad is None
            original_scores_grad_aux = None
        else:
            # B x L x N
            probs, org_scores_denom = calc_probs(original_scores, score_func)
            freq_per_expert = tokens_per_expert / (bsz * seq_len * topk)
            if load_balancing_type == 'dot':
                coeff = n_experts / (bsz * seq_len)
                probs_grad = (aux_loss_grad * freq_per_expert * coeff).view(1, 1, n_experts)
            elif load_balancing_type == 'entropy':
                prob_per_expert = probs.sum(dim=(0, 1))
                freq_per_expert = freq_per_expert - (1.0 / n_experts)
                probs_grad = (aux_loss_grad * freq_per_expert).div(prob_per_expert).view(1, 1, n_experts)
            else:
                raise ValueError(f"Unknown load balancing type: {load_balancing_type}.")

            if score_func == 'softmax':
                original_scores_grad_aux = probs_grad
            elif score_func == 'sigmoid':
                original_scores_grad_aux = (probs_grad - (probs_grad * probs).sum(dim=-1, keepdim=True)) / org_scores_denom
            else:
                raise ValueError(f"Unknown score function: {score_func}.")

        # reduce tokens/expert for router bias updating
        if router_bias is not None:
            group = get_hybrid_shard_data_parallel_group()
            world_size = get_hybrid_shard_data_parallel_world_size()
            tokens_per_expert, handle_bias = all_reduce(tokens_per_expert, world_size, group, async_op=True)
        else:
            handle_bias = None

        scores_grad = scores_grad.float()
        if scaling_factor is not None:
            scores_grad = scores_grad * scaling_factor

        scores = torch.gather(original_scores, dim=-1, index=indices)
        if topk > 1:
            scores_denom = scores.sum(dim=-1, keepdim=True)
            scores = scores / scores_denom
            scores_grad = (scores_grad - (scores_grad * scores).sum(dim=-1, keepdim=True)) / scores_denom

        original_scores_grad = torch.zeros_like(original_scores).scatter(-1, indices, scores_grad)
        if original_scores_grad_aux is not None:
            original_scores_grad = original_scores_grad + original_scores_grad_aux

        if score_func == 'softmax':
            logits_grad = torch.ops.aten._softmax_backward_data(
                original_scores_grad, original_scores, -1, torch.float32
            )
        elif score_func == 'sigmoid':
            logits_grad = torch.ops.aten.sigmoid_backward(original_scores_grad, original_scores)
        else:
            raise ValueError(f"Unknown score function: {score_func}.")

        world_size = get_model_parallel_world_size()
        group = get_model_parallel_group()
        logits_grad, handle_logits = all_reduce(logits_grad, world_size, group, async_op=True)
        # update router bias
        if router_bias is not None:
            if handle_bias is not None:
                handle_bias.wait()
            average_tokens = tokens_per_expert.sum() / n_experts
            bias_update = torch.sign(average_tokens - tokens_per_expert)
            router_bias.add_(bias_update, alpha=bias_update_rate)

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

        return x_grad, w_grad, None, None, None, None, None, None, None


topk_routing = TopKRouting.apply


class TopKRouter(torch.nn.Module):
    """TopK router.
    Similar to RowParallelLinear, with specific communication control.

    Arguments:
        in_features: first dimension of matrix A.
        topk: second dimension of matrix A.
        bias: If true, add bias
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
    """

    def __init__(
        self,
        in_features: int,
        num_experts: int,
        topk: int,
        bias: bool = True,
        bias_update_rate: Optional[float] = None,
        score_func: str = 'sigmoid',
        scaling_factor: Optional[float] = None,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
    ) -> None:
        super(TopKRouter, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.num_experts = num_experts
        self.topk = topk
        self.score_func = score_func
        self.scaling_factor = scaling_factor

        # Divide the weight matrix along the last dimension.
        rank = get_model_parallel_rank()
        world_size = get_model_parallel_world_size()

        self.input_size_per_partition = divide_and_check_no_remainder(in_features, world_size)
        self.weight = Parameter(torch.empty(self.num_experts, self.input_size_per_partition))
        if bias:
            assert bias_update_rate is not None and bias_update_rate > 0.0, f"Invalid bias update rate: {bias_update_rate}"
            self.register_buffer('bias', torch.zeros(self.num_experts))
        else:
            self.register_buffer("bias", None)
        self.bias_update_rate = bias_update_rate

        # Initialize weight.
        master_weight = torch.empty(num_experts, in_features, dtype=self.weight.dtype, requires_grad=False)
        init_method(master_weight)
        weight_list = torch.split(master_weight, self.input_size_per_partition, dim=1)
        my_weight = weight_list[rank]
        with torch.no_grad():
            self.weight.copy_(my_weight)
        # clear master weight
        del master_weight
        del weight_list

    def forward(
        self, x: torch.Tensor, load_balancing_type: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        scores, indices, tokens_per_expert, aux_loss = topk_routing(
            x, self.weight, self.num_experts, self.topk, self.score_func,
            self.bias, self.bias_update_rate, self.scaling_factor, load_balancing_type
        )
        return scores, indices, tokens_per_expert, aux_loss

    def extra_repr(self) -> str:
        return 'in_features={} ({}), n_experts={}, topk={}, score_func={}, bias={}, scaling_factor={}'.format(
            self.in_features, self.input_size_per_partition, self.num_experts, self.topk,
            self.score_func, self.bias is not None, self.scaling_factor,
        )
