from typing import Optional, Tuple, List, Any
import math
import os
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
    xattn_causal_flash_attn_fwd as xattn_fwd,
    xattn_causal_flash_attn_bwd as xattn_bwd,
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
    want_step_count: bool = False,
):
    """With ``want_step_count`` the per-step counts [B, L] int32 are returned as a 7th value (None unless the cute
    backend computed them); pass them to ``timenorm_bwd`` so the backward does not recompute them."""
    gamma = gamma + 1.0
    if backend == 'cute':
        from xllm.modules.fused_ops.cute.timestep_decay_norm import timestep_decay_norm_fwd
        y, count, mean, var, cummean, cumrstd, step_count = timestep_decay_norm_fwd(
            x, bos_mask, prev_count, prev_mean, prev_var, gamma, beta, None, num_groups, beta1, beta2, eps,
        )
        if want_step_count:
            return y, count, mean, var, cummean, cumrstd, step_count
        return y, count, mean, var, cummean, cumrstd
    timestep_norm_fwd = xllm_ops.group_timestep_decay_norm_cub_fwd if backend == 'cub' else xllm_ops.group_timestep_decay_norm_fwd
    y, count, mean, var, cummean, cumrstd = timestep_norm_fwd(
        x, bos_mask, prev_count, prev_mean, prev_var, gamma, beta, None, num_groups, beta1, beta2, eps,
    )
    if want_step_count:
        return y, count, mean, var, cummean, cumrstd, None
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
    residual: Optional[Tensor] = None,
    step_count: Optional[Tensor] = None,
    addend: Optional[Tensor] = None,
):
    """``residual``: a gradient added to x_grad. The cute kernel reduce-adds its dx into it (``residual`` is
    overwritten and returned as x_grad); the other backends add afterwards. ``addend`` (B200 name): the same
    without overwriting the given tensor."""
    if addend is not None:
        assert residual is None
        residual = addend.clone()
    gamma = gamma + 1.0
    if backend == 'cute':
        assert not memory_efficient
        from xllm.modules.fused_ops.cute.timestep_decay_norm import timestep_decay_norm_bwd
        # per-step counts come from the forward (want_step_count) or are recomputed from prev_count and the bos mask
        x_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad = timestep_decay_norm_bwd(
            y_grad, mean_grad, var_grad, x_or_y, bos_mask, cummean, cumrstd, step_count,
            gamma, None, num_groups, beta1, beta2, prev_count=prev_count,
            residual=residual if residual is not None and residual.is_contiguous() and residual.dtype == x_or_y.dtype else None,
        )
        if residual is not None and x_grad is not residual:
            x_grad = torch.add(x_grad, residual, out=x_grad)
        return x_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad
    elif backend == 'cub':
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
    if residual is not None:
        x_grad = torch.add(x_grad, residual, out=x_grad)
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
    w = weight
    if apply_weight_normalization:
        from xllm.modules.fused_ops.conv.causal_conv1d import _cute_conv_supported
        if backend == 'triton' and _cute_conv_supported(x, weight, bias, initial_state, output_final_state):
            # the CuTe conv normalizes the taps itself (saves the softmax / transpose / cast launches per call)
            from xllm.modules.fused_ops.cute.causal_conv import conv_fwd as cute_conv_fwd
            return cute_conv_fwd(x, weight, bos_mask, activation, softmax=True), None, None
        w = F.softmax(weight, dim=-1, dtype=torch.float32)  # fp32
        if backend == 'fla':
            w = w.to(x)

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
        from xllm.modules.fused_ops.conv.causal_conv1d import _cute_conv_bwd_preferred
        if backend == 'triton' and _cute_conv_bwd_preferred(x, weight, bias, initial_state, final_state_grad, out_grad):
            # the CuTe kernel normalizes the taps itself and returns the gradient w.r.t. the raw weight
            from xllm.modules.fused_ops.cute.causal_conv import conv_bwd as cute_conv_bwd
            x_grad, w_grad = cute_conv_bwd(out_grad, x, weight, bos_mask, activation, softmax=True)
            return x_grad, None, w_grad, None
        w_fp32 = F.softmax(weight, dim=-1, dtype=torch.float32)  # fp32
        w = w_fp32.to(x) if backend == 'fla' else w_fp32
    else:
        w = weight
        w_fp32 = None

    x_grad, initial_state_grad, w_grad, b_grad = conv1d_bwd(
        out_grad, final_state_grad, x, w, bias, initial_state, bos_mask,
        activation, backend, deterministic
    )
    if apply_weight_normalization:
        w_grad = torch.ops.aten._softmax_backward_data(w_grad.float(), w_fp32, -1, torch.float32).to(weight)

    return x_grad, initial_state_grad, w_grad, b_grad


def _cute_ln(x: Tensor, weight: Optional[Tensor], bias: Optional[Tensor], num_groups: int) -> bool:
    """Whether the CuTe DSL group layer norm (Blackwell, bf16, affine, >= 2048 tokens; env XLLM_CUTE_LN=0 disables it)
    handles this input."""
    if weight is None or bias is None or x.dim() != 3 or x.shape[0] * x.shape[1] < 2048:
        return False
    try:
        from xllm.modules.fused_ops.cute.layer_norm import ln_supported
    except ImportError:
        return False
    return ln_supported(x, num_groups)


def layernorm_fwd(
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    num_groups: int,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    xdim = x.shape[-1]
    if _cute_ln(x, weight, bias, num_groups):
        from xllm.modules.fused_ops.cute.layer_norm import ln_fwd
        return ln_fwd(x, weight, bias, num_groups, eps)
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
    addend: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """``addend`` (x's shape) is added to x_grad (fused into the CuTe kernel's store)."""
    xdim = x.shape[-1]
    if _cute_ln(x, weight, bias, num_groups):
        from xllm.modules.fused_ops.cute.layer_norm import ln_bwd
        return ln_bwd(y_grad, x, mean, rstd, weight, num_groups, addend=addend)
    if weight is not None:
        x_grad, w_grad, b_grad = xllm_ops.group_layer_norm_bwd_affine(y_grad, x, xdim, num_groups, mean, rstd, weight + 1.0, bias, False)
    else:
        x_grad, w_grad, b_grad = xllm_ops.group_layer_norm_bwd(y_grad, x, xdim, num_groups, mean, rstd, False), None, None
    if addend is not None:
        x_grad = torch.add(x_grad, addend, out=x_grad)
    return x_grad, w_grad, b_grad


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
    addend: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    if apply_rmsnorm:
        x_grad, w_grad = rmsnorm_bwd(y_grad, x, rstd, weight, num_groups)
        b_grad = None
        if addend is not None:
            x_grad = torch.add(x_grad, addend, out=x_grad)
    else:
        x_grad, w_grad, b_grad = layernorm_bwd(y_grad, x, mean, rstd, weight, bias, num_groups, addend=addend)

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
    bos_mask: Optional[Tensor],
    segments: Optional[Any],
    scale: float,
    dropout: float,
    rng_state: Optional[Tensor],
    stability_control: int,
    backend: str,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    if rng_state is not None:
        assert dropout > 0, dropout
        org_rng_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng_state)
    else:
        org_rng_state = None

    y, y_for_bwd, lse, flash_rng_state, _ = flash_attention_fwd(
        xq, xk, xv, bos_mask, segments, scale, dropout, stability_control, backend, save_aux=True
    )
    if rng_state is not None:
        # reset the original rng state
        assert org_rng_state is not None
        torch.cuda.set_rng_state(org_rng_state)

    return y, y_for_bwd, lse, flash_rng_state


def flash_attention_fwd(
    xq: Tensor,
    xk: Tensor,
    xv: Tensor,
    bos_mask: Optional[Tensor],
    segments: Optional[Any],
    scale: float,
    dropout: float,
    stability_control: int,
    backend: str,
    save_aux: bool
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor],
           Optional[Tensor], Optional[Tensor]]:
    rng_state = None if dropout == 0 else torch.cuda.get_rng_state()
    if backend == 'flash':
        if segments is not None:
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, total_seqlen_k = segments
            xk = xk[:, -total_seqlen_k:]
            xv = xv[:, -total_seqlen_k:]
        else:
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, total_seqlen_k = None, None, None, None, None

        y, lse, flash_rng_state = flash_fwd(
            xq, xk, xv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale, dropout, True,
        )
        y_for_bwd = y if save_aux else None
        lse = lse if save_aux else None
    elif backend == 'xattn':
        if segments is not None:
            segidx, total_seqlen_k = segments
            xk = xk[:, -total_seqlen_k:]
            xv = xv[:, -total_seqlen_k:]
            bos_mask = bos_mask[:, -total_seqlen_k:] if bos_mask is not None else None
            segidx = segidx[:, -total_seqlen_k:]
        else:
            segidx = None

        y, y_for_bwd, lse = xattn_fwd(xq, xk, xv, scale, bos_mask, segidx, stability_control, requires_grad=save_aux)
        flash_rng_state = None
    else:
        raise ValueError(f"Unknown causal attention backend: {backend}")

    return y, y_for_bwd, lse, flash_rng_state, rng_state


def flash_attention_bwd(
    y_grad: Tensor,
    y: Tensor,
    xq: Tensor,
    xk: Tensor,
    xv: Tensor,
    softmax_lse: Tensor,
    bos_mask: Optional[Tensor],
    segments: Optional[Any],
    scale: float,
    dropout: float,
    rng_state: Optional[Tensor],
    stability_control: int,
    deterministic: bool,
    backend: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    if backend == 'flash':
        if segments is not None:
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, total_seqlen_k = segments
            xk = xk[:, -total_seqlen_k:]
            xv = xv[:, -total_seqlen_k:]
        else:
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, total_seqlen_k = None, None, None, None, None

        return flash_bwd(
            y_grad, y, xq, xk, xv, softmax_lse, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            scale, dropout, rng_state, use_causal_mask=True, deterministic=deterministic
        )
    elif backend == 'xattn':
        if segments is not None:
            segidx, total_seqlen_k = segments
            xk = xk[:, -total_seqlen_k:]
            xv = xv[:, -total_seqlen_k:]
            bos_mask = bos_mask[:, -total_seqlen_k:] if bos_mask is not None else None
            segidx = segidx[:, -total_seqlen_k:]
        else:
            segidx = None

        return xattn_bwd(
            y_grad, xq, xk, xv, y, softmax_lse, scale, bos_mask, segidx, stability_control, deterministic
        )
    else:
        raise ValueError(f"Unknown causal attention backend: {backend}")


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
    stability_control: int,
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
        dropout, stability_control, backend, requires_grad=True
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
    stability_control: int,
    backend: str,
    save_aux: bool
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    rng_state = None if dropout == 0 else torch.cuda.get_rng_state()
    y, y_for_bwd, aux = sca_fwd(
        xq, xk, xv, chunk_size, scale, prev_k, prev_v, bos_mask, segment_idx,
        dropout, stability_control, backend, requires_grad=save_aux
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
    stability_control: int,
    deterministic: bool,
    backend: str,
    delta: Optional[Tensor] = None,
    fp8_dp: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    return sca_bwd(
        y_grad, xq, xk, xv, y, aux, chunk_size, scale, prev_k, prev_v,
        bos_mask, segment_idx, stability_control, deterministic, backend, delta=delta, fp8_dp=fp8_dp
    )


def _cute_swiglu(h1: Tensor, dropout: float) -> bool:
    """Whether the fused CuTe DSL SwiGLU elementwise kernels (Blackwell, bf16, no hidden dropout; env
    XLLM_CUTE_SWIGLU=0 disables them) handle this input."""
    try:
        from xllm.modules.fused_ops.cute.swiglu import swiglu_supported
    except ImportError:
        return False
    return swiglu_supported(h1, dropout)


_QUACK_GEMM = None


def _quack_swiglu(x: Tensor, w1: Tensor, dropout: float) -> bool:
    """Whether quack's fused SwiGLU GEMMs (``quack-kernels``, CuTe DSL, Hopper / Blackwell; optional, used when
    installed; env XLLM_QUACK_FFN=0 disables them) handle this FFN: fc1 / fc3 as one GEMM over [w1; w3] with
    silu(h1) * h3 in its epilogue, and the fc2 input-gradient GEMM with the SwiGLU backward in its epilogue."""
    global _QUACK_GEMM
    if os.environ.get("XLLM_QUACK_FFN", "1") == "0" or dropout > 0 or x.dtype != torch.bfloat16 or not x.is_cuda:
        return False
    if w1.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[-1] % 8 or w1.shape[0] % 8:
        return False
    if _QUACK_GEMM is None:
        try:
            from quack.gemm_interface import gemm, gemm_dgated, gemm_gated
            _QUACK_GEMM = (gemm_gated, gemm_dgated, gemm)
        except Exception:
            _QUACK_GEMM = False
    return bool(_QUACK_GEMM) and torch.cuda.get_device_capability(x.device)[0] >= 9


def _is_packed_swiglu(h3: Optional[Tensor]) -> bool:
    """swiglu_forward's quack path returns the interleaved pre-activation [h1 | h3] as h1 and an empty h3."""
    return h3 is not None and h3.numel() == 0


def swiglu_recompute(x: Tensor, w1: Tensor, w3: Optional[Tensor], h1: Optional[Tensor], h3: Optional[Tensor],
                     dropout: float, swiglu: bool) -> Tuple[Tensor, Optional[Tensor]]:
    """The backward's recompute of the FFN pre-activations not saved by the forward: with both missing on the quack
    path, the packed [h1 | h3] from one GEMM over [w1; w3] (and the empty h3 marker), else F.linear for each."""
    if swiglu and h1 is None and h3 is None and _quack_swiglu(x, w1, dropout):
        return _QUACK_GEMM[2](x, torch.cat([w1, w3]).t(), concat_layout=("B",)), x.new_empty(0)
    if h1 is None:
        h1 = F.linear(x, w1)
    if swiglu and h3 is None:
        h3 = F.linear(x, w3)
    return h1, h3


def swiglu_forward(
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    w3: Optional[Tensor],
    dropout: float,
    swiglu: bool,
    residual: Optional[Tensor] = None,
    fp8_state: Optional[dict] = None,
):
    """With ``residual`` ([tokens, D]), returns residual + fc2(hidden) from one GEMM (the add in its epilogue).
    On the quack path h1 is the interleaved pre-activation [h1 | h3] ([tokens, 2 E]) and h3 an empty marker
    (swiglu_backward handles both forms; a recomputed h1 / h3 pair takes the unfused backward).
    ``fp8_state`` (a dict): MXFP8 GEMMs (xllm/modules/fused_ops/mxfp8.py; SwiGLU, no dropout); the dict receives the
    operands the backward needs (the quantised weights and hidden^T)."""
    if fp8_state is not None:
        from xllm.modules.fused_ops.mxfp8 import mx_quantize, swiglu_fwd_mx, MxLinearSet
        assert swiglu and dropout == 0.0
        w13 = MxLinearSet([w1, w3])
        w2m = MxLinearSet([w2])
        x_row, x_col = mx_quantize(x, colwise=fp8_state.get('keep_xf', False))  # colwise kept for the backward's w1 / w3 gradients
        h1, h3 = w13.forward_all(x_row)  # one GEMM: x [w1; w3]^T, h1 / h3 its column halves (the SwiGLU kernel reads strided rows)
        hid_row, hid_col = swiglu_fwd_mx(h1, h3)  # silu(h1) * h3 straight into both fp8 operands (no bf16 hidden)
        out = w2m.forward(0, hid_row, residual=residual)
        fp8_state.update(w13=w13, w2=w2m, hid_col=hid_col)
        if x_col is not None:
            fp8_state.update(xf_row=x_row, xf_col=x_col)
        return h1, h3, out, None
    # fc1 & fc3
    if swiglu and _quack_swiglu(x, w1, dropout):
        gemm_gated = _QUACK_GEMM[0]
        h1, hidden = gemm_gated(x, torch.cat([w1, w3]).t(), activation="swiglu", concat_layout=("B",))
        h3 = h1.new_empty(0)
    elif swiglu:
        h1 = F.linear(x, w1)
        h3 = F.linear(x, w3)
        if _cute_swiglu(h1, dropout):
            from xllm.modules.fused_ops.cute.swiglu import swiglu_fwd
            hidden = swiglu_fwd(h1, h3)
        else:
            hidden = F.silu(h1)
            hidden = torch.mul(hidden, h3, out=hidden)
    else:
        h1 = F.linear(x, w1)
        h3 = None
        hidden = F.silu(h1)

    hidden, hidden_rng_state = memory_efficient_dropout_fwd(hidden, dropout, True)
    out = F.linear(hidden, w2) if residual is None else torch.addmm(residual, hidden, w2.t())
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
    addend: Optional[Tensor] = None,
    fp8_state: Optional[dict] = None,
):
    """``addend`` (x's shape, e.g. the residual-branch gradient) is added to the returned x_grad. ``fp8_state``: the
    dict filled by ``swiglu_forward`` plus ``xf_col`` (the colwise operand of xf) -> MXFP8 GEMMs."""
    assert xf is None or xf.shape == y_grad.shape, (xf.shape, y_grad.shape)  # None: fp8 operands kept instead
    bsz, seq_len, _ = x.shape

    packed = False
    if fp8_state is not None:
        from xllm.modules.fused_ops.mxfp8 import mx_quantize, mx_mm, swiglu_bwd_mx
        dy_row, dy_col = mx_quantize(y_grad, kind="grad")
        h_grad = mx_mm(dy_row, fp8_state['w2'].col)
        # the SwiGLU backward straight into the operands of the gradient GEMMs (no bf16 h1_grad / h3_grad; hidden^T
        # is kept from the forward), then one GEMM: [h1_grad | h3_grad] [w1; w3]
        d13_row, d13_col = swiglu_bwd_mx(h1, h3, h_grad)
        xf_grad = mx_mm(d13_row, fp8_state['w13'].col)
    else:
        # quack forward: h1 is the packed [h1 | h3] (2 E wide); a caller that recomputed only one of them hands over
        # an unpacked h1 with the empty marker (repacked here) or the packed h1 with a recomputed h3 (ignored)
        packed = _is_packed_swiglu(h3) or (w3 is not None and h1.shape[-1] == 2 * w1.shape[0])
        if packed and h1.shape[-1] != 2 * w1.shape[0]:
            h1 = _QUACK_GEMM[2](xf, torch.cat([w1, w3]).t(), concat_layout=("B",))
    if packed:
        # fc2 input gradient with the SwiGLU backward in its epilogue: interleaved d[h1 | h3] and the recomputed hidden
        _, gemm_dgated, gemm = _QUACK_GEMM
        w13 = torch.cat([w1, w3])
        h13_grad, h = gemm_dgated(y_grad.contiguous(), w2, h1, activation="swiglu", dynamic_scheduler=True)
        xf_grad = gemm(h13_grad, w13, dynamic_scheduler=True, concat_layout=("B",))
    elif fp8_state is None:
        # B*L x D, D x E -> B*L x E
        h_grad = torch.mm(y_grad, w2)
        if h3 is not None and _cute_swiglu(h1, dropout):
            # the recompute of hidden and both gradients in one pass
            from xllm.modules.fused_ops.cute.swiglu import swiglu_bwd
            h, h1_grad, h3_grad = swiglu_bwd(h1, h3, h_grad)
        else:
            # B*L x E
            h1s = F.silu(h1)
            h, h_noise = memory_efficient_dropout_fwd(h1s if h3 is None else h1s * h3, dropout, True, rng_state)
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
        if addend is not None and handle is None:
            x_grad = torch.add(x_grad, addend, out=x_grad)
        if norm_w_grad is not None:
            norm_w_grad, handle_norm_w = reduce_scatter(norm_w_grad, parallel_region='model', async_op=True)
        if norm_b_grad is not None:
            norm_b_grad, handle_norm_b = reduce_scatter(norm_b_grad, parallel_region='model', async_op=True)
    else:
        xf_grad, handle = reduce_scatter(xf_grad, parallel_region='model', async_op=True)

    # E x B*L, B*L x D -> E x D
    if fp8_state is not None:
        w1_grad, w3_grad = fp8_state['w13'].wgrad(d13_col, fp8_state['xf_col'])  # one GEMM for both
        w2_grad, = fp8_state['w2'].wgrad(dy_col, fp8_state['hid_col'])
    elif packed:
        # [w1; w3] gradient in one GEMM from the interleaved d[h1 | h3] (concat output: two contiguous halves)
        w1_grad, w3_grad = _QUACK_GEMM[2](h13_grad.t(), xf, dynamic_scheduler=True, concat_layout=("out",)).chunk(2)
        w2_grad = torch.mm(y_grad.t(), h)
    else:
        w1_grad = torch.mm(h1_grad.t(), xf)
        w3_grad = None if h3_grad is None else torch.mm(h3_grad.t(), xf)
        w2_grad = torch.mm(y_grad.t(), h)

    # wait for x_grad
    if handle is not None:
        handle.wait()
        if gather_before_norm and addend is not None:
            x_grad = torch.add(x_grad, addend, out=x_grad)
    if handle_norm_w is not None:
        handle_norm_w.wait()
    if handle_norm_b is not None:
        handle_norm_b.wait()

    if not gather_before_norm:
        x_grad, norm_w_grad, norm_b_grad = layer_or_rmsnorm_bwd(
            xf_grad, x, x_mean, x_invvar, norm_w, norm_b, norm_groups, apply_rmsnorm, addend=addend
        )
    elif handle is None and addend is not None:
        x_grad = torch.add(x_grad, addend, out=x_grad)

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


def _cute_rope(x: Tensor, freqs_cis: Tensor, rope_head_dim: int, backward: bool) -> Optional[Tensor]:
    """The single-pass CuTe DSL rotary embedding (Blackwell, bf16; env XLLM_CUTE_ROPE=0 disables), else None."""
    try:
        from xllm.modules.fused_ops.cute.rope import rope, rope_supported
    except ImportError:
        return None
    if not rope_supported(x, rope_head_dim) or freqs_cis.dtype != torch.complex64 or freqs_cis.shape != (x.shape[1], rope_head_dim // 2):
        return None
    return rope(x, freqs_cis, rope_head_dim, backward)


def apply_rope(x: Tensor, freqs_cis: Tensor, head_dim: int, rope_head_dim: int, backward: bool):
    if rope_head_dim > 0 and (y := _cute_rope(x, freqs_cis, rope_head_dim, backward)) is not None:
        return y
    if rope_head_dim == head_dim:
        x = apply_rotary_embedding(x, freqs_cis=freqs_cis, backward=backward)
    elif rope_head_dim > 0:
        x_rope = apply_rotary_embedding(x[:, :, :, :rope_head_dim], freqs_cis=freqs_cis, backward=backward)
        # The input may still be needed by normalization backward.
        x = torch.cat((x_rope, x[:, :, :, rope_head_dim:]), dim=-1)
    return x


def apply_ropes(xq: Tensor, xk: Tensor, freqs_cis: Tensor, head_dim: int, rope_head_dim: int, backward: bool):
    if rope_head_dim > 0 and (yq := _cute_rope(xq, freqs_cis, rope_head_dim, backward)) is not None:
        return yq, _cute_rope(xk, freqs_cis, rope_head_dim, backward)
    if rope_head_dim == head_dim:
        xq, xk = apply_rotary_embeddings(xq, xk, freqs_cis=freqs_cis, backward=backward)
    elif rope_head_dim > 0:
        xq_rope, xk_rope = apply_rotary_embeddings(xq[:, :, :, :rope_head_dim], xk[:, :, :, :rope_head_dim], freqs_cis=freqs_cis, backward=backward)
        xq = torch.cat((xq_rope, xq[:, :, :, rope_head_dim:]), dim=-1)
        xk = torch.cat((xk_rope, xk[:, :, :, rope_head_dim:]), dim=-1)
    return xq, xk
