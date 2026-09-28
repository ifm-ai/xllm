import math
import numpy as np
from functools import partial
import torch
import torch.nn.functional as F
from torch import Tensor
import fire


def manual_router(
    logits: Tensor,
    n_experts: int,
    topk: int,
    score_func: str,
    router_bias: Tensor,
    scaling_factor: float,
    load_balancing_type: str
):
    if score_func == 'softmax':
        score_fn = partial(F.softmax, dim=-1, dtype=torch.float32)
    elif score_func == 'sigmoid':
        score_fn = torch.sigmoid
    else:
        raise ValueError(f"Unknown score function: {score_func}.")

    bsz, seq_len, _ = logits.shape

    scores = score_fn(logits)
    original_scores = scores
    scores = scores + router_bias.to(scores)

    # B x L x K
    indices = torch.topk(scores, topk, dim=-1)[1]
    scores = torch.gather(original_scores, dim=-1, index=indices)
    scores = scores / (scores.sum(dim=-1, keepdim=True)) if topk > 1 else scores
    scores = scores * scaling_factor
    # N
    tokens_per_expert = torch.bincount(indices.flatten(), minlength=n_experts)

    if load_balancing_type is None:
        aux_loss = None
    elif load_balancing_type == 'dot':
        # B x L x N
        if score_func == 'softmax':
            probs = original_scores
        elif score_func == 'sigmoid':
            probs = original_scores / original_scores.sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown score function: {score_func}.")
        # N
        prob_per_expert = probs.mean(dim=(0, 1))
        freq_per_expert = tokens_per_expert * (n_experts / (bsz * seq_len * topk))
        aux_loss = torch.dot(freq_per_expert, prob_per_expert)
    elif load_balancing_type == 'entropy':
        # B x L x N
        if score_func == 'softmax':
            probs = original_scores
        elif score_func == 'sigmoid':
            probs = original_scores / original_scores.sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown score function: {score_func}.")
        # N
        prob_per_expert = probs.mean(dim=(0, 1))
        freq_per_expert = tokens_per_expert / (bsz * seq_len * topk)
        aux_loss = torch.dot(freq_per_expert - (1.0 / n_experts), torch.log(prob_per_expert)) + np.log(n_experts)
    else:
        raise ValueError(f"Unknown load balancing type: {load_balancing_type}.")

    return scores, indices, tokens_per_expert, aux_loss


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


def router_fwd(
    logits: Tensor,
    n_experts: int,
    topk: int,
    score_func: str,
    router_bias: Tensor,
    scaling_factor: float,
    load_balancing_type: str
):
    if score_func == 'softmax':
        score_fn = partial(F.softmax, dim=-1, dtype=torch.float32)
    elif score_func == 'sigmoid':
        score_fn = torch.sigmoid
    else:
        raise ValueError(f"Unknown score function: {score_func}.")

    bsz, seq_len, _ = logits.shape
    org_dtype = logits.dtype

    logits = logits.to(torch.float32)
    scores = score_fn(logits)
    original_scores = scores
    scores = scores + router_bias.to(scores)

    # B x L x K
    indices = torch.topk(scores, topk, dim=-1)[1]
    scores = torch.gather(original_scores, dim=-1, index=indices)
    scores = scores / (scores.sum(dim=-1, keepdim=True)) if topk > 1 else scores
    scores = scores * scaling_factor
    scores = scores.to(dtype=org_dtype)
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
            aux_loss = torch.dot(freq_per_expert - (1.0 / n_experts), torch.log(prob_per_expert)) + math.log(n_experts)
        else:
            raise ValueError(f"Unknown load balancing type: {load_balancing_type}.")

    return scores, indices, tokens_per_expert, aux_loss, original_scores


def routing_bwd(
    scores_grad,
    aux_loss_grad,
    original_scores,
    indices,
    tokens_per_expert,
    n_experts: int,
    topk: int,
    score_func: str,
    scaling_factor: float,
    load_balancing_type: str,
    org_dtype: torch.dtype,
):
    bsz, seq_len, _ = original_scores.shape

    if load_balancing_type is None:
        assert aux_loss_grad is None
        original_scores_grad_aux = None
    else:
        # B x L x N
        probs, org_scores_denom = calc_probs(original_scores, score_func)
        freq_per_expert = tokens_per_expert / (bsz * seq_len * topk)
        if load_balancing_type == 'dot':
            # 1 x 1 x N
            coeff = n_experts / (bsz * seq_len)
            probs_grad = (aux_loss_grad * freq_per_expert * coeff).view(1, 1, n_experts)
        elif load_balancing_type == 'entropy':
            # 1 x 1 x N
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

    # B x L x N
    logits_grad = logits_grad.to(dtype=org_dtype)
    return logits_grad


def test(bsz: int, seq_len:int, n_experts: int, topk: int, score_func: str, aux_type: str, dtype: str):
    pt_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    with torch.no_grad():
        logits = torch.randn(bsz, seq_len, n_experts, requires_grad=False, dtype=pt_dtype, device="cuda")
        router_bias = torch.randn(n_experts, requires_grad=False, dtype=torch.float32, device="cuda")

    logits = logits.clone().detach().requires_grad_(True)

    scores_manual, idx_manual, tpe_manual, aux_loss_manual = manual_router(
        logits.float(), n_experts, topk, score_func, router_bias, 2.5, aux_type
    )
    s_manual_flat = scores_manual.flatten()
    num_elem = s_manual_flat.shape[0]
    weight = torch.randn(num_elem, 1, requires_grad=False, dtype=torch.float32, device="cuda")
    loss = s_manual_flat @ weight + aux_loss_manual
    scores_manual.retain_grad()
    aux_loss_manual.retain_grad()
    loss.backward()
    score_grad = scores_manual.grad.to(pt_dtype)
    aux_loss_grad = aux_loss_manual.grad
    logits_grad = logits.grad

    # sequential mgmm
    with torch.no_grad():
        atol = {"fp32": 1e-6, "bf16": 1e-2, "fp16": 1e-3}[dtype]
        rtol = {"fp32": 1e-6, "bf16": 1e-2, "fp16": 1e-3}[dtype]
        scores, indices, tokens_per_expert, aux_loss, original_scores = router_fwd(
            logits, n_experts, topk, score_func, router_bias, 2.5, aux_type
        )
        logits_grad_auto = routing_bwd(
            score_grad, aux_loss_grad, original_scores, indices, tokens_per_expert,
            n_experts, topk, score_func, 2.5, aux_type, pt_dtype,
        )
        torch.testing.assert_close(scores, scores_manual.to(pt_dtype), rtol=rtol, atol=atol)
        torch.testing.assert_close(aux_loss, aux_loss_manual, rtol=1e-6, atol=1e-6)
        print(f"B={bsz}, L={seq_len}, N={n_experts}, K={topk}, fn={score_func}, aux={aux_type}, dtype={dtype}: pass fwd test")
        torch.testing.assert_close(logits_grad_auto, logits_grad, rtol=rtol, atol=atol)
        print(f"B={bsz}, L={seq_len}, N={n_experts}, K={topk}, fn={score_func}, aux={aux_type}, dtype={dtype}: pass bwd test")


def main(seed: int, dtype: str):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    for B, L, N, K in [
        [2, 8192, 192, 8]
    ]:
        test(B, L, N, K, "softmax", "dot", dtype)
        test(B, L, N, K, "sigmoid", "dot", dtype)

        test(B, L, N, K, "softmax", "entropy", dtype)
        test(B, L, N, K, "sigmoid", "entropy", dtype)


if __name__ == "__main__":
    fire.Fire(main)
