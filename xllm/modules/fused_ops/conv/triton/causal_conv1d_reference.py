"""Readable PyTorch reference for causal depthwise conv1d."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize_activation(activation: str | None) -> str | None:
    if activation == "swish":
        return "silu"
    if activation not in (None, "silu"):
        raise NotImplementedError("activation must be None, silu, or swish")
    return activation


def _dense_bos_mask(
    bos_mask: torch.Tensor | None,
    batch: int,
    seqlen: int,
) -> torch.Tensor | None:
    if bos_mask is not None and bos_mask.shape != (batch, seqlen):
        raise ValueError("BOS mask shape does not match x")
    return bos_mask


def _accumulation_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _bos_prefix(bos_mask: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(bos_mask, dim=1, dtype=torch.int32)


def _lag_valid(prefix: torch.Tensor, lag: int) -> torch.Tensor:
    batch, seqlen = prefix.shape
    valid = torch.zeros(batch, seqlen, device=prefix.device, dtype=torch.bool)
    if lag < seqlen:
        valid[:, lag:] = (prefix[:, lag:] - prefix[:, :-lag]) == 0
    return valid


def _preactivation(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    bos_prefix: torch.Tensor | None,
    deterministic: bool = False,
) -> torch.Tensor:
    batch, seqlen, dim = x.shape
    weight_dim, width = weight.shape
    if weight_dim != dim or width < 1:
        raise ValueError("weight must have shape (dim, width) with width >= 1")
    if bias is not None and bias.shape != (dim,):
        raise ValueError("bias must have shape (dim,)")
    if initial_state is not None and initial_state.shape != (batch, width - 1, dim):
        raise ValueError("initial_state must have shape (batch, width - 1, dim)")

    accumulation_dtype = _accumulation_dtype(x)
    x_acc = x.to(accumulation_dtype)
    weight_acc = weight.to(accumulation_dtype)
    bias_acc = bias.to(accumulation_dtype) if bias is not None else None
    initial_acc = (
        initial_state.to(accumulation_dtype) if initial_state is not None else None
    )

    if bos_prefix is None:
        history = (
            x_acc.new_zeros(batch, width - 1, dim)
            if initial_acc is None
            else initial_acc
        )
        padded_x = torch.cat((history, x_acc), dim=1).transpose(1, 2)
        if x.is_cuda and deterministic:
            # Pass determinism per operation without mutating process-wide
            # cuDNN flags. Keep the fast convolution fallback for this layout.
            return torch.ops.aten._convolution(
                padded_x, weight_acc.unsqueeze(1), bias_acc,
                [1], [0], [1], False, [0], dim,
                benchmark=False, deterministic=True,
                cudnn_enabled=torch.backends.cudnn.enabled,
                allow_tf32=torch.backends.cudnn.allow_tf32,
            ).transpose(1, 2)
        return F.conv1d(
            padded_x, weight_acc.unsqueeze(1), bias_acc, groups=dim,
        ).transpose(1, 2)

    out = x_acc * weight_acc[:, -1].view(1, 1, dim)
    for lag in range(1, width):
        shifted = torch.zeros_like(x_acc)
        if lag < seqlen:
            shifted[:, lag:] = x_acc[:, :-lag]
            shifted = shifted * _lag_valid(bos_prefix, lag).unsqueeze(2)
        out = out + shifted * weight_acc[:, -1 - lag].view(1, 1, dim)

    if initial_acc is not None:
        contributions = []
        for t in range(min(seqlen, width - 1)):
            contribution = (
                initial_acc[:, t:, :]
                * weight_acc[:, : width - 1 - t].T.unsqueeze(0)
            ).sum(dim=1)
            contribution = contribution * (bos_prefix[:, t] == 0).unsqueeze(1)
            contributions.append(contribution)
        if contributions:
            initial_out = torch.stack(contributions, dim=1)
            initial_out = F.pad(initial_out, (0, 0, 0, seqlen - initial_out.shape[1]))
            out = out + initial_out

    if bias_acc is not None:
        out = out + bias_acc.view(1, 1, dim)
    return out


def _final_states(
    x: torch.Tensor,
    initial_state: torch.Tensor | None,
    bos_prefix: torch.Tensor | None,
    width: int,
) -> torch.Tensor:
    batch, seqlen, dim = x.shape
    state_length = width - 1
    if state_length == 0:
        return x.new_empty(batch, 0, dim)

    if bos_prefix is None:
        if seqlen >= state_length:
            return x[:, -state_length:]
        history = (
            x.new_zeros(batch, state_length, dim)
            if initial_state is None
            else initial_state
        )
        return torch.cat((history[:, seqlen:], x), dim=1)

    states = []
    for state_index in range(state_length):
        source_t = seqlen - (state_length - state_index)
        if source_t >= 0:
            value = x[:, source_t]
            if source_t + 1 < seqlen:
                valid = bos_prefix[:, -1] - bos_prefix[:, source_t] == 0
                value = value * valid.unsqueeze(1)
        elif initial_state is not None:
            value = initial_state[:, state_index + seqlen]
            value = value * (bos_prefix[:, -1] == 0).unsqueeze(1)
        else:
            value = x.new_zeros(batch, dim)
        states.append(value)
    return torch.stack(states, dim=1)


def causal_conv1d_reference_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    bos_mask: torch.Tensor | None = None,
    activation: str | None = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(out, final_state)`` using xllm's public tensor layouts."""
    activation = _normalize_activation(activation)
    dense_bos_mask = _dense_bos_mask(bos_mask, x.shape[0], x.shape[1])
    bos_prefix = _bos_prefix(dense_bos_mask) if dense_bos_mask is not None else None
    preactivation = _preactivation(x, weight, bias, initial_state, bos_prefix, deterministic)
    out = F.silu(preactivation) if activation == "silu" else preactivation
    final_state = _final_states(x, initial_state, bos_prefix, weight.shape[1])
    return out.to(x.dtype), final_state


def _silu_grad(x: torch.Tensor) -> torch.Tensor:
    sigmoid = torch.sigmoid(x)
    return sigmoid * (1.0 + x * (1.0 - sigmoid))


def causal_conv1d_reference_bwd(
    out_grad: torch.Tensor,
    final_state_grad: torch.Tensor | None,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    bos_mask: torch.Tensor | None = None,
    activation: str | None = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    """Return ``(dx, dinitial_state, dweight, dbias)`` explicitly."""
    activation = _normalize_activation(activation)
    batch, seqlen, dim = x.shape
    width = weight.shape[1]
    dense_bos_mask = _dense_bos_mask(bos_mask, batch, seqlen)
    prefix = _bos_prefix(dense_bos_mask) if dense_bos_mask is not None else None
    accumulation_dtype = _accumulation_dtype(x)

    x_acc = x.to(accumulation_dtype)
    weight_acc = weight.to(accumulation_dtype)
    initial_acc = (
        initial_state.to(accumulation_dtype) if initial_state is not None else None
    )
    grad = out_grad.to(accumulation_dtype)
    if activation == "silu":
        preactivation = _preactivation(x, weight, bias, initial_state, prefix, deterministic)
        grad = grad * _silu_grad(preactivation)

    x_grad = torch.zeros_like(x_acc)
    weight_grad = torch.zeros_like(weight_acc)
    bias_grad = grad.sum(dim=(0, 1)) if bias is not None else None
    initial_state_grad = (
        torch.zeros_like(initial_acc) if initial_acc is not None else None
    )

    for lag in range(width):
        weight_index = width - 1 - lag
        if lag == 0:
            lag_grad = grad
            lag_x = x_acc
        elif lag < seqlen:
            valid = (
                torch.ones(
                    batch,
                    seqlen - lag,
                    device=x.device,
                    dtype=torch.bool,
                )
                if prefix is None
                else _lag_valid(prefix, lag)[:, lag:]
            )
            lag_grad = grad[:, lag:] * valid.unsqueeze(2)
            lag_x = x_acc[:, :-lag]
        else:
            continue

        weight_grad[:, weight_index] += (lag_grad * lag_x).sum(dim=(0, 1))
        x_grad[:, : seqlen - lag] += (
            lag_grad * weight_acc[:, weight_index].view(1, 1, dim)
        )

    if initial_acc is not None and initial_state_grad is not None:
        for t in range(min(seqlen, width - 1)):
            valid = (
                torch.ones(batch, device=x.device, dtype=torch.bool)
                if prefix is None
                else prefix[:, t] == 0
            )
            grad_t = grad[:, t] * valid.unsqueeze(1)
            for state_index in range(t, width - 1):
                weight_index = state_index - t
                weight_grad[:, weight_index] += (
                    grad_t * initial_acc[:, state_index]
                ).sum(dim=0)
                initial_state_grad[:, state_index] += (
                    grad_t * weight_acc[:, weight_index].view(1, dim)
                )

    if final_state_grad is not None:
        final_grad = final_state_grad.to(accumulation_dtype)
        for state_index in range(width - 1):
            source_t = seqlen - (width - 1 - state_index)
            grad_state = final_grad[:, state_index]
            if source_t >= 0:
                if prefix is not None and source_t + 1 < seqlen:
                    valid = prefix[:, -1] - prefix[:, source_t] == 0
                    grad_state = grad_state * valid.unsqueeze(1)
                x_grad[:, source_t] += grad_state
            elif initial_state_grad is not None:
                if prefix is not None:
                    grad_state = grad_state * (prefix[:, -1] == 0).unsqueeze(1)
                initial_state_grad[:, state_index + seqlen] += grad_state

    return (
        x_grad.to(x.dtype),
        initial_state_grad.to(initial_state.dtype)
        if initial_state_grad is not None and initial_state is not None
        else None,
        weight_grad.to(weight.dtype),
        bias_grad.to(bias.dtype)
        if bias_grad is not None and bias is not None
        else None,
    )
