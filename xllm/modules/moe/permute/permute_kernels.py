"""Fused inverse-perm + weighted reduction Triton kernel for MoE output.

Fuses inverse-perm lookup + weighted reduction into one kernel,
replacing argsort + index_select + mul + sum in the MoE forward pass.
Includes a fused backward pass for autograd support.
"""

import triton
import triton.language as tl


@triton.jit
def _fused_y_perm_fwd_kernel(
    output_ptr,
    y_ptr,
    inverse_indices_ptr,
    routing_scores_ptr,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for k in tl.static_range(0, K):
        flat_idx = pid_token * K + k
        inv_idx = tl.load(inverse_indices_ptr + flat_idx)
        y_vals = tl.load(y_ptr + inv_idx * D + d_offsets, mask=d_mask, other=0.0)
        score = tl.load(routing_scores_ptr + flat_idx)
        acc += y_vals.to(tl.float32) * score.to(tl.float32)

    out_offsets = pid_token * D + d_offsets
    tl.store(output_ptr + out_offsets, acc.to(output_ptr.dtype.element_ty), mask=d_mask)


@triton.jit
def _y_grad_bwd_kernel(
    grad_y_ptr,
    grad_output_ptr,
    inverse_indices_ptr,
    routing_scores_ptr,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    # Load grad_output[token] once, reuse for all K experts
    go_vals = tl.load(
        grad_output_ptr + pid_token * D + d_offsets, mask=d_mask, other=0.0
    ).to(tl.float32)

    for k in tl.static_range(0, K):
        flat_idx = pid_token * K + k
        inv_idx = tl.load(inverse_indices_ptr + flat_idx)
        score = tl.load(routing_scores_ptr + flat_idx).to(tl.float32)

        # grad_y2[inv_idx] = grad_output[token] * score
        # (no race: inverse permutation is a bijection)
        grad_y2_vals = go_vals * score
        tl.store(
            grad_y_ptr + inv_idx * D + d_offsets,
            grad_y2_vals.to(grad_y_ptr.dtype.element_ty),
            mask=d_mask,
        )


@triton.jit
def _scores_grad_bwd_kernel(
    grad_scores_ptr,
    grad_output_ptr,
    y_ptr,
    inverse_indices_ptr,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    # Load grad_output[token] once, reuse for all K experts
    go_vals = tl.load(
        grad_output_ptr + pid_token * D + d_offsets, mask=d_mask, other=0.0
    ).to(tl.float32)

    for k in tl.static_range(0, K):
        flat_idx = pid_token * K + k
        inv_idx = tl.load(inverse_indices_ptr + flat_idx)

        # grad_score[flat_idx] += dot(grad_output[token], y2[inv_idx])
        # partial dot over this D-block; needs atomic add across D-blocks
        y_vals = tl.load(y_ptr + inv_idx * D + d_offsets, mask=d_mask, other=0.0).to(
            tl.float32
        )
        partial_dot = tl.sum(go_vals * y_vals, axis=0)
        tl.atomic_add(grad_scores_ptr + flat_idx, partial_dot)


@triton.jit
def _fused_y_perm_bwd_kernel(
    grad_y_ptr,
    grad_scores_ptr,
    grad_output_ptr,
    y_ptr,
    inverse_indices_ptr,
    routing_scores_ptr,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    # Load grad_output[token] once, reuse for all K experts
    go_vals = tl.load(
        grad_output_ptr + pid_token * D + d_offsets, mask=d_mask, other=0.0
    ).to(tl.float32)

    for k in tl.static_range(0, K):
        flat_idx = pid_token * K + k
        inv_idx = tl.load(inverse_indices_ptr + flat_idx)
        score = tl.load(routing_scores_ptr + flat_idx).to(tl.float32)

        # grad_y2[inv_idx] = grad_output[token] * score
        # (no race: inverse permutation is a bijection)
        grad_y2_vals = go_vals * score
        tl.store(
            grad_y_ptr + inv_idx * D + d_offsets,
            grad_y2_vals.to(grad_y_ptr.dtype.element_ty),
            mask=d_mask,
        )

        # grad_score[flat_idx] += dot(grad_output[token], y2[inv_idx])
        # partial dot over this D-block; needs atomic add across D-blocks
        y_vals = tl.load(y_ptr + inv_idx * D + d_offsets, mask=d_mask, other=0.0).to(
            tl.float32
        )
        partial_dot = tl.sum(go_vals * y_vals, axis=0)
        tl.atomic_add(grad_scores_ptr + flat_idx, partial_dot)
