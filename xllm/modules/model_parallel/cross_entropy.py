from typing import Optional
import torch
from torch import Tensor
import torch.nn.functional as F

from xllm.distributed import (
    get_model_parallel_group,
    get_model_parallel_rank,
    get_model_parallel_world_size
)


class VocabParallelCrossEntropy(torch.autograd.Function):
    """
    https://github.com/NVIDIA/Megatron-LM/blob/8ce8256ff09373a275245aff3db68a3688218c44/megatron/core/tensor_parallel/cross_entropy.py#L14
    """

    @staticmethod
    def forward(
        ctx,
        partial_logits: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0
    ):

        if weight is not None:
            raise NotImplementedError("Weighted cross entropy has not been implemented.")
        if label_smoothing > 0.0:
            raise NotImplementedError("Label smoothing has not been implemented.")
        assert ignore_index < 0, 'ignore index should be negative'

        # Get the partition's vocab indecies
        rank = get_model_parallel_rank()
        partial_voc_size = partial_logits.shape[-1]
        vocab_start_idx = partial_voc_size * rank
        vocab_end_idx = partial_voc_size * (rank + 1)

        # B*L
        target_1d = target.clone().view(-1)
        ignore_index_mask = target_1d == ignore_index
        target_mask = (target_1d < vocab_start_idx) | (target_1d >= vocab_end_idx)
        target_1d = target_1d - vocab_start_idx
        target_1d[target_mask] = 0
        assert torch.all((target_1d >= 0) & (target_1d < partial_voc_size)), 'invalid target index'

        # B x L x 1
        logits_max = torch.max(partial_logits, dim=-1)[0]
        torch.distributed.all_reduce(
            logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=get_model_parallel_group(),
        )
        partial_logits = partial_logits - logits_max.unsqueeze(dim=-1)
        # B*L x V
        logits_2d = partial_logits.view(-1, partial_voc_size)
        # B*L
        arange_1d = torch.arange(logits_2d.shape[0], device=logits_2d.device)
        predicted_logits = logits_2d[arange_1d, target_1d]
        predicted_logits = predicted_logits.clone().contiguous()
        predicted_logits[target_mask] = 0.0
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_model_parallel_group(),
        )

        # B*L x V
        exp_logits = torch.exp(logits_2d, out=logits_2d)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_model_parallel_group(),
        )
        # B*L
        loss = torch.log(sum_exp_logits) - predicted_logits
        loss[ignore_index_mask] = 0.0
        # B x L
        loss = loss.reshape(target.shape)

        # save for backward
        scores = torch.div(exp_logits, sum_exp_logits.unsqueeze(dim=-1), out=exp_logits)
        ctx.save_for_backward(scores, target_mask, target_1d, ignore_index_mask)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        scores, target_mask, masked_target_1d, ignore_index_mask = ctx.saved_tensors
        partial_voc_size = scores.shape[-1]
        # B*L
        grad_output_1d = grad_output.view(-1)
        grad_output_1d[ignore_index_mask] = 0.0
        # B*L x V
        grad_input = scores
        # B*L
        arange_1d = torch.arange(grad_input.shape[0], device=scores.device)
        softmax_update = 1.0 - target_mask.float()

        grad_input[arange_1d, masked_target_1d] -= softmax_update

        # Finally elementwise multiplication with the output gradients.
        grad_input = torch.mul(grad_input, grad_output_1d.unsqueeze(dim=-1), out=grad_input)
        # B x L x V
        grad_input = grad_input.view(*grad_output.shape, partial_voc_size)

        return grad_input, None, None, None, None


def vocab_parallel_cross_entropy(
    logits: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Tensor:
    r"""This criterion computes the cross entropy loss between input logits and target.

        See :class:`~torch.nn.CrossEntropyLoss` for details.

        Args:
            logits (Tensor) : Predicted unnormalized logits;
                Shape: :math:`(B, L1, C)`.
            target (Tensor) : Ground truth class indices or class probabilities;
                Shape: :math:`(B, L2)`.
            weight (Tensor, optional): a manual rescaling weight given to each
                class. If given, has to be a Tensor of size `C`
            ignore_index (int, optional): Specifies a target value that is ignored
                and does not contribute to the input gradient. When :attr:`size_average` is
                ``True``, the loss is averaged over non-ignored targets. Note that
                :attr:`ignore_index` is only applicable when the target contains class indices.
                Default: -100
            label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount
                of smoothing when computing the loss, where 0.0 means no smoothing. The targets
                become a mixture of the original ground truth and a uniform distribution as described in
                `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.

            .. math::
                \begin{aligned}
                    C ={} & \text{number of classes} \\
                    B ={} & \text{batch size} \\
                    L1 ={} & \text{sequence length on each GPU} \\
                    L2 ={} & \text{total sequence length} \\
                \end{aligned}

    """
    bsz, seq_len, _ = logits.shape
    if get_model_parallel_world_size() == 1:
        losses = F.cross_entropy(logits.flatten(0, 1), target.flatten(0, 1), reduction="none",
                                 weight=weight, ignore_index=ignore_index, label_smoothing=label_smoothing)
        losses = losses.view(bsz, seq_len)
    else:
        losses = VocabParallelCrossEntropy.apply(logits, target)
    return losses
