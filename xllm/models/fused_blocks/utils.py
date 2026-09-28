from typing import Optional, Tuple, List, Any
import math
import torch
from torch import Tensor
from torch.nn import functional as F
from einops import rearrange

import xllm_extension.ops as xllm_ops
from xllm.modules.rotary_positional_embedding import (
    apply_rotary_embedding,
    apply_rotary_embeddings,
)
from xllm.modules.fused_ops import (
    memory_efficient_dropout_fwd,
    memory_efficient_dropout_bwd,
    flash_attention_fwd as flash_fwd,
    flash_attention_bwd as flash_bwd,
    causal_conv1d_fwd as conv1d_fwd,
    causal_conv1d_bwd as conv1d_bwd,
    sliding_chunk_attention_fwd as sca_fwd,
    sliding_chunk_attention_bwd as sca_bwd,
    multi_group_matmul_fwd,
    multi_group_matmul_bwd,
)
from xllm.modules.moe.permute import (
    fused_y_perm_fwd,
    fused_y_perm_bwd_y_grad,
    fused_y_perm_bwd_scores_grad
)

from .distributed import _c2r, _r2c, reduce_scatter


def timenorm_fwd(
    x: Tensor,
    bos_mask: Tensor,
    prev_count: Tensor,
    prev_mean: Tensor,
    prev_var: Tensor,
    gamma: Tensor,
    beta: Tensor,
    num_groups: int,
    beta1: float,
    beta2: float,
    eps: float,
    backend: str,
):
    gamma = gamma + 1.0
    timestep_norm_fwd = xllm_ops.group_timestep_decay_norm_cub_fwd if backend == 'cub' else xllm_ops.group_timestep_decay_norm_fwd
    y, count, mean, var, cummean, cumrstd = timestep_norm_fwd(
        x, bos_mask, prev_count, prev_mean, prev_var, gamma, beta, None, num_groups, beta1, beta2, eps,
    )
    return y, count, mean, var, cummean, cumrstd


def timenorm_bwd(
    y_grad: Tensor,
    mean_grad: Tensor,
    var_grad: Tensor,
    x_or_y: Tensor,
    bos_mask: Tensor,
    prev_count: Tensor,
    prev_mean: Tensor,
    cummean: Tensor,
    cumrstd: Tensor,
    gamma: Tensor,
    beta: Tensor,
    num_groups: int,
    beta1: float,
    beta2: float,
    eps: float,
    memory_efficient: bool,
    backend: str,
):
    gamma = gamma + 1.0
    if backend == 'cub':
        assert not memory_efficient
        x_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad = xllm_ops.group_timestep_decay_norm_cub_bwd(
            y_grad, mean_grad, var_grad, x_or_y, prev_count, bos_mask, cummean, cumrstd,
            gamma, None, num_groups, beta1, beta2
        )
    else:
        x_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad = xllm_ops.group_timestep_decay_norm_bwd(
            y_grad, mean_grad, var_grad, x_or_y, prev_count, bos_mask, cummean, cumrstd,
            gamma, beta, None, num_groups, beta1, beta2, eps, memory_efficient
        )
    return x_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad


def _cema_coeffs_fwd(
    alpha: Tensor,
    delta: Tensor,
    theta: Tensor,
    gamma: Tensor,
    ndim: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    # D
    theta = torch.sigmoid(theta.float()) * (2 * math.pi / ndim)
    # N
    wavelets = torch.arange(1, ndim + 1, dtype=theta.dtype, device=theta.device)
    # D x N
    theta = wavelets * theta.unsqueeze(1)

    # D x N
    alpha = torch.sigmoid(alpha.float())
    delta = torch.sigmoid(delta.float())
    # coeffs
    p = alpha
    q = torch.polar(1.0 - alpha * delta, theta)
    # D x N
    gamma = _r2c(gamma.float())
    return p, q, gamma


def _cema_coeffs_bwd(
    grad_p: Tensor,
    grad_q: Tensor,
    grad_gamma: Tensor,
    p: Tensor,
    q: Tensor,
    theta: Tensor,
    delta: Tensor,
    ndim: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # re-compute delta & theta
    dtype = delta.dtype
    theta = torch.sigmoid(theta.float())
    delta = torch.sigmoid(delta.float())
    alpha = p

    # grad of abs & angle
    grad_abs, grad_angle = polar_backward(grad_q, q)
    grad_delta = grad_abs * (-p)
    grad_alpha = grad_p - grad_abs * delta
    # N
    wavelets = torch.arange(1, ndim + 1, dtype=p.dtype, device=p.device)
    # D x N -> D
    grad_theta = (grad_angle * wavelets).sum(dim=1) * (2 * math.pi / ndim)

    # grad of sigmoid
    grad_alpha = torch.ops.aten.sigmoid_backward(grad_alpha, alpha).to(dtype)
    grad_delta = torch.ops.aten.sigmoid_backward(grad_delta, delta).to(dtype)
    grad_theta = torch.ops.aten.sigmoid_backward(grad_theta, theta).to(dtype)
    # grad of gamma
    grad_gamma = _c2r(grad_gamma).to(dtype)
    return grad_alpha, grad_delta, grad_theta, grad_gamma


def polar_backward(grad: Tensor, output: Tensor) -> Tuple[Tensor, Tensor]:
    grad_conj = torch.conj(grad)
    grad_abs = (grad_conj * torch.sgn(output)).real
    out_mul_1_j = output * 1j
    grad_angle = (grad_conj * out_mul_1_j).real
    return grad_abs, grad_angle


def cema_fwd(
    x: Tensor,
    hx: Optional[Tensor],
    alpha: Tensor,
    delta: Tensor,
    theta: Tensor,
    gamma: Tensor,
    bos_mask: Optional[Tensor],
    ndim: int,
    backend: str,
) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
    # calc coeffs
    p, q, gamma = _cema_coeffs_fwd(alpha, delta, theta, gamma, ndim)

    cema_scan_fwd = xllm_ops.cema_cub_scan_fwd if backend == 'cub' else xllm_ops.cema_blelloch_scan_fwd
    # B x D x L
    residual = x
    output, h, chunk_decay, chunk_gain = cema_scan_fwd(x, p, q, gamma, bos_mask, hx)
    # residual
    output = output + residual
    return output, h, chunk_decay, chunk_gain


def cema_bwd(
    y_grad: Tensor,
    h_grad: Optional[Tensor],
    x: Tensor,
    chunk_decay: Tensor,
    chunk_gain: Tensor,
    bos_mask: Optional[Tensor],
    alpha: Tensor,
    delta: Tensor,
    theta: Tensor,
    gamma: Tensor,
    ndim: int,
    backend: str,
) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor, Tensor, Tensor]:
    # re-calc coeffs
    p, q, gamma = _cema_coeffs_fwd(alpha, delta, theta, gamma, ndim)

    cema_scan_bwd = xllm_ops.cema_cub_scan_bwd if backend == 'cub' else xllm_ops.cema_blelloch_scan_bwd
    # B x D x L
    x_grad_residual = y_grad
    x_grad, p_grad, q_grad, gamma_grad, hx_grad = cema_scan_bwd(
        y_grad, h_grad, chunk_decay, chunk_gain, x, p, q, gamma, bos_mask
    )
    x_grad = x_grad + x_grad_residual

    alpha_grad, delta_grad, theta_grad, gamma_grad = _cema_coeffs_bwd(p_grad, q_grad, gamma_grad, p, q, theta, delta, ndim)

    return x_grad, hx_grad, alpha_grad, delta_grad, theta_grad, gamma_grad


def causal_conv1d_fwd(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    initial_state: Optional[Tensor],
    bos_mask: Optional[Tensor],
    output_final_state: bool,
    activation: Optional[str],
    apply_weight_normalization: bool,
    backend: str,
    deterministic: bool
) -> Tuple[Tensor, Optional[Tensor], Any]:
    w = F.softmax(weight, dim=-1, dtype=torch.float32).to(x) if apply_weight_normalization else weight
    return conv1d_fwd(
        x, w, bias, initial_state, bos_mask, output_final_state, activation, backend, deterministic
    )


def causal_conv1d_bwd(
    out_grad: Tensor,
    final_state_grad: Optional[Tensor],
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    initial_state: Optional[Tensor],
    bos_mask: Optional[Tensor],
    activation: Optional[str],
    apply_weight_normalization: bool,
    backend: str,
    deterministic: bool
) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
    if apply_weight_normalization:
        w_fp32 = F.softmax(weight, dim=-1, dtype=torch.float32)
        w = w_fp32.to(x)
    else:
        w = weight
        w_fp32 = None
    x_grad, initial_state_grad, w_grad, b_grad = conv1d_bwd(
        out_grad, final_state_grad, x, w, bias, initial_state, bos_mask,
        activation, backend, deterministic
    )
    if apply_weight_normalization:
        w_grad = torch.ops.aten._softmax_backward_data(w_grad.float(), w_fp32, -1, torch.float32).to(w)

    return x_grad, initial_state_grad, w_grad, b_grad


def layernorm_fwd(
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    num_groups: int,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    xdim = x.shape[-1]
    if weight is not None:
        return xllm_ops.group_layer_norm_fwd_affine(x, xdim, num_groups, weight + 1.0, bias, eps)
    else:
        return xllm_ops.group_layer_norm_fwd(x, xdim, num_groups, eps)


def layernorm_bwd(
    y_grad: Tensor,
    x: Tensor,
    mean: Optional[Tensor],
    rstd: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    num_groups: int,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    xdim = x.shape[-1]
    if weight is not None:
        return xllm_ops.group_layer_norm_bwd_affine(y_grad, x, xdim, num_groups, mean, rstd, weight + 1.0, bias, False)
    else:
        return xllm_ops.group_layer_norm_bwd(y_grad, x, xdim, num_groups, mean, rstd, False), None, None


def rmsnorm_fwd(
    x: Tensor,
    weight: Optional[Tensor],
    num_groups: int,
    eps: float,
) -> Tuple[Tensor, Tensor]:
    xdim = x.shape[-1]
    if weight is not None:
        return xllm_ops.group_rms_norm_fwd_affine(x, xdim, num_groups, weight + 1.0, eps)
    else:
        return xllm_ops.group_rms_norm_fwd(x, xdim, num_groups, eps)


def rmsnorm_bwd(
    y_grad: Tensor,
    x: Tensor,
    rstd: Tensor,
    weight: Optional[Tensor],
    num_groups: int,
) -> Tuple[Tensor, Optional[Tensor]]:
    xdim = x.shape[-1]
    if weight is not None:
        return xllm_ops.group_rms_norm_bwd_affine(y_grad, x, xdim, num_groups, rstd, weight + 1.0, False)
    else:
        return xllm_ops.group_rms_norm_bwd(y_grad, x, xdim, num_groups, rstd, False), None


def mem_effn_rmsnorm_bwd(
    y_grad: Tensor,
    y: Tensor,
    rstd: Tensor,
    num_groups: int,
) -> Tensor:
    ydim = y.shape[-1]
    return xllm_ops.group_rms_norm_bwd(y_grad, y, ydim, num_groups, rstd, True)


def layer_or_rmsnorm_fwd(
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    num_groups: int,
    layernorm_eps: float,
    rmsnorm_eps: float,
    apply_rmsnorm: bool,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    if apply_rmsnorm:
        x_mean = None
        y, x_invvar = rmsnorm_fwd(x, weight, num_groups, rmsnorm_eps)
    else:
        y, x_mean, x_invvar = layernorm_fwd(x, weight, bias, num_groups, layernorm_eps)

    return y, x_mean, x_invvar


def layer_or_rmsnorm_bwd(
    y_grad: Tensor,
    x: Tensor,
    mean: Optional[Tensor],
    rstd: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    num_groups: int,
    apply_rmsnorm: bool,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    if apply_rmsnorm:
        x_grad, w_grad = rmsnorm_bwd(y_grad, x, rstd, weight, num_groups)
        b_grad = None
    else:
        x_grad, w_grad, b_grad = layernorm_bwd(y_grad, x, mean, rstd, weight, bias, num_groups)

    return x_grad, w_grad, b_grad


def normalize_fwd(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    num_heads: int,
    eps: float
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    xdim = x.shape[-1]
    hdim = xdim // num_heads
    z, rstd = xllm_ops.group_rms_norm_fwd(x, xdim, num_heads, eps)
    weight = weight.view(4, -1)
    bias = bias.view(4, -1)
    weight = (weight + 1.0) / math.sqrt(hdim)
    # B x L x 4 x S
    y = z.unsqueeze(2) * weight + bias
    # B x L x 4 x S -> B x L x S
    sq, sk, aq, ak = torch.unbind(y, dim=2)

    return sq, sk, aq, ak, z, rstd


def normalize_bwd(
    sq_grad: Tensor,
    sk_grad: Tensor,
    aq_grad: Tensor,
    ak_grad: Tensor,
    z: Tensor,
    rstd: Optional[Tensor],
    weight: Tensor,
    num_heads: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    xdim = z.shape[-1]
    hdim = xdim // num_heads
    # B x L x S -> B x L x 4 x S
    y_grad = torch.stack([sq_grad, sk_grad, aq_grad, ak_grad], dim=2)
    # 4 x S
    weight = weight.view(4, -1)
    weight = (weight + 1.0) / math.sqrt(hdim)
    grad_bias = y_grad.sum(dim=(0, 1)).flatten()
    grad_weight = (y_grad * z.unsqueeze(2)).sum(dim=(0, 1)).flatten() / math.sqrt(hdim)
    # B x L x S
    z_grad = (y_grad * weight).sum(dim=2)
    x_grad = xllm_ops.group_rms_norm_bwd(z_grad, z, xdim, num_heads, rstd, True)

    return x_grad, grad_weight, grad_bias


def recompute_flash_attention_lse(
    xq: Tensor,
    xk: Tensor,
    xv: Tensor,
    cu_seqlens_q: Optional[Tensor],
    cu_seqlens_k: Optional[Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    scale: float,
    dropout: float,
    rng_state: Optional[Tensor]
) -> Tuple[Tensor, Tensor, Tensor]:
    if rng_state is not None:
        assert dropout > 0, dropout
        org_rng_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng_state)
    else:
        org_rng_state = None

    y, lse, flash_state = flash_fwd(
        xq, xk, xv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale, dropout, True,
    )
    if rng_state is not None:
        # reset the original rng state
        assert org_rng_state is not None
        torch.cuda.set_rng_state(org_rng_state)

    return y, lse, flash_state


def flash_attention_fwd(
    xq: Tensor,
    xk: Tensor,
    xv: Tensor,
    cu_seqlens_q: Optional[Tensor],
    cu_seqlens_k: Optional[Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    scale: float,
    dropout: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    rng_state = None if dropout == 0 else torch.cuda.get_rng_state()
    y, lse, flash_rng_state = flash_fwd(
        xq, xk, xv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale, dropout, True,
    )
    return y, lse, flash_rng_state, rng_state


def flash_attention_bwd(
    y_grad: Tensor,
    y: Tensor,
    xq: Tensor,
    xk: Tensor,
    xv: Tensor,
    softmax_lse: Tensor,
    cu_seqlens_q: Optional[Tensor],
    cu_seqlens_k: Optional[Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    scale: float,
    dropout: float = 0.0,
    rng_state: Optional[Tensor] = None,
    deterministic: bool = True
) -> Tuple[Tensor, Tensor, Tensor]:
    return flash_bwd(
        y_grad, y, xq, xk, xv, softmax_lse, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        scale, dropout, rng_state, use_causal_mask=True, deterministic=deterministic
    )


def recompute_sliding_chunk_attention(
    xq: Tensor,
    xk: Tensor,
    xv: Tensor,
    chunk_size: int,
    scale: float,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    bos_mask: Optional[Tensor],
    segment_idx: Optional[Tensor],
    dropout: float,
    fp32_attn_output: bool,
    backend: str,
    rng_state: Optional[Tensor]
) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    # set rng state to the state in fwd
    if rng_state is not None:
        assert dropout > 0, dropout
        org_rng_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng_state)
    else:
        org_rng_state = None

    y, y_for_bwd, aux = sca_fwd(
        xq, xk, xv, chunk_size, scale, prev_k, prev_v, bos_mask, segment_idx,
        dropout, fp32_attn_output, backend, requires_grad=True
    )
    if rng_state is not None:
        # reset the original rng state
        assert org_rng_state is not None
        torch.cuda.set_rng_state(org_rng_state)

    return y, y_for_bwd, aux


def sliding_chunk_attention_fwd(
    xq: Tensor,
    xk: Tensor,
    xv: Tensor,
    chunk_size: int,
    scale: float,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    bos_mask: Optional[Tensor],
    segment_idx: Optional[Tensor],
    dropout: float,
    fp32_attn_output: bool,
    backend: str,
    save_aux: bool
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    rng_state = None if dropout == 0 else torch.cuda.get_rng_state()
    y, y_for_bwd, aux = sca_fwd(
        xq, xk, xv, chunk_size, scale, prev_k, prev_v, bos_mask, segment_idx,
        dropout, fp32_attn_output, backend, requires_grad=save_aux
    )
    return y, y_for_bwd, aux, rng_state


def sliding_chunk_attention_bwd(
    y_grad: Tensor,
    xq: Tensor,
    xk: Tensor,
    xv: Tensor,
    y: Optional[Tensor],
    aux: Tensor,
    chunk_size: int,
    scale: float,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    bos_mask: Optional[Tensor],
    segment_idx: Optional[Tensor],
    deterministic: bool,
    backend: str,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    return sca_bwd(
        y_grad, xq, xk, xv, y, aux, chunk_size, scale, prev_k, prev_v,
        bos_mask, segment_idx, deterministic, backend
    )


def swiglu_forward(
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    w3: Optional[Tensor],
    dropout: float,
    swiglu: bool,
):
    # fc1 & fc3
    if swiglu:
        h1 = F.linear(x, w1)
        h3 = F.linear(x, w3)
        hidden = F.silu(h1)
        hidden = torch.mul(hidden, h3, out=hidden)
    else:
        h1 = F.linear(x, w1)
        h3 = None
        hidden = F.silu(h1)

    hidden, hidden_rng_state = memory_efficient_dropout_fwd(hidden, dropout, True)
    out = F.linear(hidden, w2)
    return h1, h3, out, hidden_rng_state


def swiglu_backward(
    y_grad: Tensor,
    xf: Tensor,
    w1: Tensor,
    w2: Tensor,
    w3: Optional[Tensor],
    h1: Tensor,
    h3: Optional[Tensor],
    dropout: float,
    rng_state: Optional[Tensor],
    x: Tensor,
    x_mean: Optional[Tensor],
    x_invvar: Tensor,
    norm_w: Optional[Tensor],
    norm_b: Optional[Tensor],
    norm_groups: int,
    apply_rmsnorm: bool,
    gather_before_norm: bool,
):
    assert xf.shape == y_grad.shape, (xf.shape, y_grad.shape)
    bsz, seq_len, _ = x.shape

    # B*L x E
    h1s = F.silu(h1)
    h, h_noise = memory_efficient_dropout_fwd(h1s if h3 is None else h1s * h3, dropout, True, rng_state)
    # B*L x D, D x E -> B*L x E
    h_grad = torch.mm(y_grad, w2)
    h_grad = memory_efficient_dropout_bwd(h_grad, dropout, rng_state, h_noise)

    if h3 is not None:
        h1s_grad = h_grad * h3
        h3_grad = h_grad * h1s
    else:
        h1s_grad = h_grad
        h3_grad = None

    h1_grad = torch.ops.aten.silu_backward(h1s_grad, h1)
    xf_grad = h1_grad.matmul(w1)
    if h3_grad is not None:
        xf_grad = torch.addmm(xf_grad, h3_grad, w3, out=xf_grad)

    handle_norm_w = None
    handle_norm_b = None
    xf_grad = rearrange(xf_grad, '(b l) d -> b l d', b=bsz)
    if gather_before_norm:
        x_grad, norm_w_grad, norm_b_grad = layer_or_rmsnorm_bwd(
            xf_grad, x, x_mean, x_invvar, norm_w, norm_b, norm_groups, apply_rmsnorm
        )
        x_grad, handle = reduce_scatter(x_grad, parallel_region='model', async_op=True)
        if norm_w_grad is not None:
            norm_w_grad, handle_norm_w = reduce_scatter(norm_w_grad, parallel_region='model', async_op=True)
        if norm_b_grad is not None:
            norm_b_grad, handle_norm_b = reduce_scatter(norm_b_grad, parallel_region='model', async_op=True)
    else:
        xf_grad, handle = reduce_scatter(xf_grad, parallel_region='model', async_op=True)

    # E x B*L, B*L x D -> E x D
    w1_grad = torch.mm(h1_grad.t(), xf)
    w3_grad = None if h3_grad is None else torch.mm(h3_grad.t(), xf)
    w2_grad = torch.mm(y_grad.t(), h)

    # wait for x_grad
    if handle is not None:
        handle.wait()
    if handle_norm_w is not None:
        handle_norm_w.wait()
    if handle_norm_b is not None:
        handle_norm_b.wait()

    if not gather_before_norm:
        x_grad, norm_w_grad, norm_b_grad = layer_or_rmsnorm_bwd(
            xf_grad, x, x_mean, x_invvar, norm_w, norm_b, norm_groups, apply_rmsnorm
        )

    return x_grad, w1_grad, w2_grad, w3_grad, norm_w_grad, norm_b_grad


def fused_permute_y_fwd(
    sorted_indices: Tensor,
    y: Tensor,
    routing_scores: Tensor,
    bsz: int,
    slen: int,
    topk: int,
    backend: str
) -> Tuple[Tensor, Tensor]:
    if backend == 'torch':
        # re-order
        inverse_indices = torch.empty_like(sorted_indices)
        inverse_indices[sorted_indices] = torch.arange(
            sorted_indices.shape[0], device=sorted_indices.device
        )
        # B*L*K x D/TP
        y = torch.index_select(y, 0, inverse_indices)
        # B x L x D/TP
        out = (y.view(bsz, slen, topk, -1) * routing_scores.unsqueeze(3)).sum(dim=2)
    elif backend == 'triton':
        # fused inverse-perm + weighted reduction
        out, inverse_indices = fused_y_perm_fwd(
            sorted_indices, y, routing_scores, bsz, slen, topk
        )
    else:
        raise ValueError(f"Unknown backend: {backend}.")

    return out, inverse_indices


def fused_permute_y_bwd_y_grad(
    out_grad: Tensor,
    sorted_indices: Tensor,
    routing_scores: Tensor,
    bsz: int,
    slen: int,
    topk: int,
    backend: str
) -> Tuple[Tensor, Tensor]:
    inverse_indices = torch.empty_like(sorted_indices)
    inverse_indices[sorted_indices] = torch.arange(
        sorted_indices.shape[0], device=sorted_indices.device
    )
    if backend == 'torch':
        # (B x L x 1 x D/TP) * (B x L x K x 1) -> B x L x K x D/TP
        y_grad = out_grad.unsqueeze(2) * routing_scores.unsqueeze(3)
        # B x L x K x D/TP -> B*L*K x D/TP
        y_grad = torch.index_select(y_grad.view(bsz * slen * topk, -1), 0, sorted_indices)
    elif backend == 'triton':
        y_grad = fused_y_perm_bwd_y_grad(
            out_grad, inverse_indices, routing_scores, bsz, slen, topk
        )
    else:
        raise ValueError(f"Unknown backend: {backend}.")

    return y_grad, inverse_indices


def fused_permute_y_bwd_scores_grad(
    out_grad: Tensor,
    inverse_indices: Tensor,
    y: Tensor,
    bsz: int,
    slen: int,
    topk: int,
    backend: str
) -> Tensor:
    if backend == 'torch':
        # B*L*K x D/TP -> B x L x K x D/TP
        y = torch.index_select(y, 0, inverse_indices).view(bsz, slen, topk, -1)
        scores_grad = torch.einsum('blkd,bld->blk', y, out_grad)
    elif backend == 'triton':
        scores_grad = fused_y_perm_bwd_scores_grad(
            out_grad, inverse_indices, y, bsz, slen, topk
        )
    else:
        raise ValueError(f"Unknown backend: {backend}.")

    return scores_grad


def grouped_experts_fwd(
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    w3: Tensor,
    group_sizes: List[int],
    backend: str,
    memory_efficient: bool,
    dropout: float,
    hidden_rng_state: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    h1 = multi_group_matmul_fwd(x, w1, group_sizes, True, backend)
    h3 = multi_group_matmul_fwd(x, w3, group_sizes, True, backend)
    h1s = F.silu(h1)
    if memory_efficient:
        hidden = torch.mul(h1s, h3, out=h1s)
        h1s = None
    else:
        hidden = h1s * h3
    hidden, hidden_rng_state_or_noise = memory_efficient_dropout_fwd(hidden, dropout, True, hidden_rng_state)
    out = multi_group_matmul_fwd(hidden, w2, group_sizes, True, backend)
    return out, h1, h1s, h3, hidden, hidden_rng_state_or_noise


def grouped_experts_bwd(
    y_grad: Tensor,
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    w3: Tensor,
    h1: Tensor,
    h1s: Tensor,
    h3: Tensor,
    hidden: Tensor,
    group_sizes: List[int],
    backend: str,
    memory_efficient: bool,
    dropout: float,
    hidden_rng_state: Optional[Tensor],
    hidden_noise: Optional[Tensor]
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    h_grad, w2_grad = multi_group_matmul_bwd(y_grad, w2, hidden, group_sizes, True, backend)
    h_grad = memory_efficient_dropout_bwd(h_grad, dropout, hidden_rng_state, hidden_noise)
    # x_grad from h1
    h1s_grad = torch.mul(h_grad, h3, out=h3) if memory_efficient else h_grad * h3
    h1_grad = torch.ops.aten.silu_backward(h1s_grad, h1)
    x_grad, w1_grad = multi_group_matmul_bwd(h1_grad, w1, x, group_sizes, True, backend)
    # x_grad from h3
    h3_grad = torch.mul(h_grad, h1s, out=h_grad)
    x_grad_h3, w3_grad = multi_group_matmul_bwd(h3_grad, w3, x, group_sizes, True, backend)

    x_grad = torch.add(x_grad, x_grad_h3, out=x_grad)
    return x_grad, w1_grad, w2_grad, w3_grad


def apply_rope(x: Tensor, freqs_cis: Tensor, head_dim: int, rope_head_dim: int, backward: bool):
    if rope_head_dim == head_dim:
        x = apply_rotary_embedding(x, freqs_cis=freqs_cis, backward=backward)
    elif rope_head_dim > 0:
        x_rope = apply_rotary_embedding(x[:, :, :, :rope_head_dim], freqs_cis=freqs_cis, backward=backward)
        x[:, :, :, :rope_head_dim] = x_rope
    return x


def apply_ropes(xq: Tensor, xk: Tensor, freqs_cis: Tensor, head_dim: int, rope_head_dim: int, backward: bool):
    if rope_head_dim == head_dim:
        xq, xk = apply_rotary_embeddings(xq, xk, freqs_cis=freqs_cis, backward=backward)
    elif rope_head_dim > 0:
        xq_rope, xk_rope = apply_rotary_embeddings(xq[:, :, :, :rope_head_dim], xk[:, :, :, :rope_head_dim], freqs_cis=freqs_cis, backward=backward)
        xq[:, :, :, :rope_head_dim] = xq_rope
        xk[:, :, :, :rope_head_dim] = xk_rope
    return xq, xk
