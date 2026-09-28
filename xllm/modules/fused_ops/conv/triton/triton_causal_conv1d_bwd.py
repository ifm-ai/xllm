"""Direct Triton implementation of causal depthwise-conv1d backward."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from xllm.modules.fused_ops.conv.triton.causal_conv1d_reference import causal_conv1d_reference_bwd


@triton.jit
def _ffma_ftz(a, b, c):
    return tl.inline_asm_elementwise(
        asm="""
        {
            fma.rn.ftz.f32 $0, $1, $2, $3;
        }
        """,
        constraints="=f,f,f,f",
        args=[a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _prefetch_l1_if(ptr, do_prefetch):
    return tl.inline_asm_elementwise(
        asm="""
        {
            @$2 prefetch.global.L1 [$1];
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,b",
        args=[ptr, do_prefetch],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _conv2_ffma_z(x0, src1, w0, w1, bias):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 acc0;
            .reg .f32 acc1;
            mov.f32 acc0, $10;
            mov.f32 acc1, $11;
            fma.rn.ftz.f32 acc0, $2, $8, acc0;
            fma.rn.ftz.f32 acc1, $3, $9, acc1;
            fma.rn.ftz.f32 $0, $4, $6, acc0;
            fma.rn.ftz.f32 $1, $5, $7, acc1;
        }
        """,
        constraints="=f,=f,f,f,f,f,f,f,f,f,f,f",
        args=[x0, src1, w0, w1, bias],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _conv3_ffma_z(x0, src1, src2, w0, w1, w2, bias):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 acc0;
            .reg .f32 acc1;
            mov.f32 acc0, $14;
            mov.f32 acc1, $15;
            fma.rn.ftz.f32 acc0, $2, $12, acc0;
            fma.rn.ftz.f32 acc1, $3, $13, acc1;
            fma.rn.ftz.f32 acc0, $4, $10, acc0;
            fma.rn.ftz.f32 acc1, $5, $11, acc1;
            fma.rn.ftz.f32 $0, $6, $8, acc0;
            fma.rn.ftz.f32 $1, $7, $9, acc1;
        }
        """,
        constraints="=f,=f,f,f,f,f,f,f,f,f,f,f,f,f,f,f",
        args=[x0, src1, src2, w0, w1, w2, bias],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _masked_ffma_ftz(acc, a, b, valid):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred p;
            setp.ne.u32 p, $4, 0;
            mov.f32 $0, $1;
            @p fma.rn.ftz.f32 $0, $2, $3, $1;
        }
        """,
        constraints="=f,f,f,f,r",
        args=[acc, a, b, valid.to(tl.int32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _silu_backward(z, dout):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 y0;
            .reg .f32 y1;
            .reg .f32 t0;
            .reg .f32 t1;
            .reg .f32 one_minus0;
            .reg .f32 one_minus1;
            .reg .f32 inner0;
            .reg .f32 inner1;
            fma.rn.ftz.f32 y0, $2, 0f3F000000, 0f00000000;
            fma.rn.ftz.f32 y1, $3, 0f3F000000, 0f00000000;
            tanh.approx.f32 t0, y0;
            tanh.approx.f32 t1, y1;
            fma.rn.ftz.f32 $0, t0, 0f3F000000, 0f3F000000;
            fma.rn.ftz.f32 $1, t1, 0f3F000000, 0f3F000000;
            fma.rn.ftz.f32 one_minus0, t0, 0fBF000000, 0f3F000000;
            fma.rn.ftz.f32 inner0, $2, one_minus0, 0f3F800000;
            mul.rn.ftz.f32 $0, $0, inner0;
            fma.rn.ftz.f32 one_minus1, t1, 0fBF000000, 0f3F000000;
            fma.rn.ftz.f32 inner1, $3, one_minus1, 0f3F800000;
            mul.rn.ftz.f32 $1, $1, inner1;
            mul.rn.ftz.f32 $1, $5, $1;
            mul.rn.ftz.f32 $0, $4, $0;
        }
        """,
        constraints="=f,=f,f,f,f,f",
        args=[z, dout],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _fused_streaming_bwd_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    initial_ptr,
    bos_ptr,
    dout_ptr,
    dx_ptr,
    accum_ptr,
    seqlen,
    dim,
    chunks_per_batch,
    WIDTH: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    HAS_BOS: tl.constexpr,
    ACTIVATION_SILU: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOOP_UNROLL: tl.constexpr,
    DETERMINISTIC: tl.constexpr,
):
    pid_d = tl.program_id(0)
    chunk_id = tl.program_id(1)
    b = chunk_id // chunks_per_batch
    chunk = chunk_id - b * chunks_per_batch
    start_t = chunk * BLOCK_N
    is_first = chunk == 0

    d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d < dim
    weight_base = d * WIDTH
    w0 = tl.load(weight_ptr + weight_base, mask=d_mask, other=0.0).to(
        tl.float32
    )
    w1 = tl.load(weight_ptr + weight_base + 1, mask=d_mask, other=0.0).to(
        tl.float32
    )
    if WIDTH >= 3:
        w2 = tl.load(weight_ptr + weight_base + 2, mask=d_mask, other=0.0).to(
            tl.float32
        )
    if WIDTH >= 4:
        w3 = tl.load(weight_ptr + weight_base + 3, mask=d_mask, other=0.0).to(
            tl.float32
        )
    bias = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    row_base = (b * seqlen + start_t) * dim + d
    bos_base = b * seqlen
    init_base = b * dim * (WIDTH - 1) + d * (WIDTH - 1)

    if HAS_INITIAL:
        prev1_init = tl.load(
            initial_ptr + init_base + WIDTH - 2,
            mask=d_mask & is_first,
            other=0.0,
        ).to(tl.float32)
        if WIDTH >= 3:
            prev2_init = tl.load(
                initial_ptr + init_base + WIDTH - 3,
                mask=d_mask & is_first,
                other=0.0,
            ).to(tl.float32)
        if WIDTH >= 4:
            prev3_init = tl.load(
                initial_ptr + init_base,
                mask=d_mask & is_first,
                other=0.0,
            ).to(tl.float32)
    else:
        prev1_init = tl.zeros((BLOCK_D,), tl.float32)
        if WIDTH >= 3:
            prev2_init = tl.zeros((BLOCK_D,), tl.float32)
        if WIDTH >= 4:
            prev3_init = tl.zeros((BLOCK_D,), tl.float32)

    prev1_x = tl.load(
        x_ptr + row_base - dim,
        mask=d_mask & (~is_first),
        other=0.0,
    ).to(tl.float32)
    prev1 = tl.where(is_first, prev1_init, prev1_x)
    if WIDTH >= 3:
        prev2_x = tl.load(
            x_ptr + row_base - 2 * dim,
            mask=d_mask & (~is_first),
            other=0.0,
        ).to(tl.float32)
        if HAS_BOS:
            bos_tm1 = tl.load(
                bos_ptr + bos_base + start_t - 1,
                mask=~is_first,
                other=False,
            )
            prev2_x = tl.where(~bos_tm1, prev2_x, 0.0)
        prev2 = tl.where(is_first, prev2_init, prev2_x)
    if WIDTH >= 4:
        prev3_x = tl.load(
            x_ptr + row_base - 3 * dim,
            mask=d_mask & (~is_first),
            other=0.0,
        ).to(tl.float32)
        if HAS_BOS:
            bos_tm2 = tl.load(
                bos_ptr + bos_base + start_t - 2,
                mask=~is_first,
                other=False,
            )
            prev3_x = tl.where((~bos_tm1) & (~bos_tm2), prev3_x, 0.0)
        prev3 = tl.where(is_first, prev3_init, prev3_x)

    g_m1 = tl.zeros((BLOCK_D,), tl.float32)
    if WIDTH >= 3:
        g_m2 = tl.zeros((BLOCK_D,), tl.float32)
    if WIDTH >= 4:
        g_m3 = tl.zeros((BLOCK_D,), tl.float32)
    keep_m1 = False
    if WIDTH >= 3:
        keep_m2 = False

    sum_bias = tl.zeros((BLOCK_D,), tl.float32)
    sum_w0 = tl.zeros((BLOCK_D,), tl.float32)
    sum_w1 = tl.zeros((BLOCK_D,), tl.float32)
    if WIDTH >= 3:
        sum_w2 = tl.zeros((BLOCK_D,), tl.float32)
    if WIDTH >= 4:
        sum_w3 = tl.zeros((BLOCK_D,), tl.float32)

    for i in tl.range(0, BLOCK_N + WIDTH - 1, loop_unroll_factor=LOOP_UNROLL):
        t = start_t + i
        row_valid = t < seqlen
        base = row_base + i * dim
        if LOOP_UNROLL == 2:
            # The factor-two SASS consumes each row's values shortly after its
            # LDG and does not hoist the next unrolled row.  Warm the row two
            # recurrence steps ahead without extending the value live ranges.
            prefetch_t = t + 2
            prefetch_base = (
                (b * seqlen + prefetch_t) * dim + pid_d * BLOCK_D
            )
            do_prefetch = prefetch_t < seqlen
            _prefetch_l1_if(x_ptr + prefetch_base, do_prefetch)
            _prefetch_l1_if(dout_ptr + prefetch_base, do_prefetch)
        x0 = tl.load(
            x_ptr + base, mask=d_mask & row_valid, other=0.0
        ).to(tl.float32)
        dout = tl.load(
            dout_ptr + base, mask=d_mask & row_valid, other=0.0
        ).to(tl.float32)

        if HAS_BOS:
            bos_t = tl.load(
                bos_ptr + bos_base + t,
                mask=row_valid,
                other=True,
            )
            keep_prev = row_valid & (~bos_t)
        else:
            keep_prev = row_valid

        src1 = tl.where(keep_prev, prev1, 0.0)
        if WIDTH >= 3:
            src2 = tl.where(keep_prev, prev2, 0.0)
        if WIDTH >= 4:
            src3 = tl.where(keep_prev, prev3, 0.0)

        if ACTIVATION_SILU:
            if WIDTH == 2:
                z = _conv2_ffma_z(x0, src1, w0, w1, bias)
            elif WIDTH == 3:
                z = _conv3_ffma_z(x0, src1, src2, w0, w1, w2, bias)
            else:
                z = _ffma_ftz(src1, w2, x0 * w3)
                z = _ffma_ftz(src2, w1, z)
                z = _ffma_ftz(src3, w0, z)
                z += bias
            # The masked dout load is already zero outside row_valid, and the
            # helper's final operation multiplies by dout.
            g = _silu_backward(z, dout)
        else:
            g = tl.where(row_valid, dout, 0.0)

        reduce_g = tl.where(i < BLOCK_N, g, 0.0)
        sum_bias += reduce_g
        sum_w0 = _ffma_ftz(
            reduce_g,
            src1 if WIDTH == 2 else (src2 if WIDTH == 3 else src3),
            sum_w0,
        )
        if WIDTH == 2:
            sum_w1 = _ffma_ftz(reduce_g, x0, sum_w1)
        elif WIDTH == 3:
            sum_w1 = _ffma_ftz(reduce_g, src1, sum_w1)
            sum_w2 = _ffma_ftz(reduce_g, x0, sum_w2)
        else:
            sum_w1 = _ffma_ftz(reduce_g, src2, sum_w1)
            sum_w2 = _ffma_ftz(reduce_g, src1, sum_w2)
            sum_w3 = _ffma_ftz(reduce_g, x0, sum_w3)

        out_t = t - (WIDTH - 1)
        out_valid = (i >= (WIDTH - 1)) & (out_t < seqlen)
        if WIDTH == 2:
            dx = _masked_ffma_ftz(g_m1 * w1, g, w0, keep_prev)
        elif WIDTH == 3:
            valid1 = keep_m1
            valid2 = valid1 & keep_prev
            dx = g_m2 * w2
            dx = _masked_ffma_ftz(dx, g_m1, w1, valid1)
            dx = _masked_ffma_ftz(dx, g, w0, valid2)
        else:
            valid1 = keep_m2
            valid2 = valid1 & keep_m1
            valid3 = valid2 & keep_prev
            dx = g_m3 * w3
            dx = _masked_ffma_ftz(dx, g_m2, w2, valid1)
            dx = _masked_ffma_ftz(dx, g_m1, w1, valid2)
            dx = _masked_ffma_ftz(dx, g, w0, valid3)

        tl.store(
            dx_ptr + (b * seqlen + out_t) * dim + d,
            dx,
            mask=d_mask & out_valid,
        )

        if WIDTH == 2:
            g_m1 = g
        elif WIDTH == 3:
            g_m2 = g_m1
            g_m1 = g
            keep_m1 = keep_prev
        else:
            g_m3 = g_m2
            g_m2 = g_m1
            g_m1 = g
            keep_m2 = keep_m1
            keep_m1 = keep_prev

        if WIDTH >= 4:
            prev3 = src2
        if WIDTH >= 3:
            prev2 = src1
        prev1 = x0

    active_chunk = start_t < seqlen
    if DETERMINISTIC:
        # Each chunk owns its partials; block scheduling cannot change a sum.
        accum_ptr += chunk_id * (WIDTH + 1) * dim
        tl.store(accum_ptr + d, sum_bias, mask=d_mask & active_chunk)
        tl.store(accum_ptr + dim + d, sum_w0, mask=d_mask & active_chunk)
        tl.store(accum_ptr + 2 * dim + d, sum_w1, mask=d_mask & active_chunk)
        if WIDTH >= 3:
            tl.store(accum_ptr + 3 * dim + d, sum_w2, mask=d_mask & active_chunk)
        if WIDTH >= 4:
            tl.store(accum_ptr + 4 * dim + d, sum_w3, mask=d_mask & active_chunk)
    else:
        tl.atomic_add(accum_ptr + d, sum_bias, mask=d_mask & active_chunk, sem="relaxed")
        tl.atomic_add(
            accum_ptr + dim + d,
            sum_w0,
            mask=d_mask & active_chunk,
            sem="relaxed",
        )
        tl.atomic_add(
            accum_ptr + 2 * dim + d,
            sum_w1,
            mask=d_mask & active_chunk,
            sem="relaxed",
        )
        if WIDTH >= 3:
            tl.atomic_add(
                accum_ptr + 3 * dim + d,
                sum_w2,
                mask=d_mask & active_chunk,
                sem="relaxed",
            )
        if WIDTH >= 4:
            tl.atomic_add(
                accum_ptr + 4 * dim + d,
                sum_w3,
                mask=d_mask & active_chunk,
                sem="relaxed",
            )


@triton.jit
def _reduce_parameter_grads_kernel(
    partial_ptr,
    dweight_ptr,
    dbias_ptr,
    dim,
    num_chunks,
    WIDTH: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    d = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    parameter = tl.program_id(1)
    r = tl.arange(0, BLOCK_R)
    sums = tl.zeros((BLOCK_R, BLOCK_D), tl.float32)
    # Bound register use independently of sequence length. Every lane visits
    # chunks in a fixed order, followed by a fixed reduction tree.
    for start in range(0, num_chunks, BLOCK_R):
        chunk = start + r
        offsets = (chunk[:, None] * (WIDTH + 1) + parameter) * dim + d[None, :]
        sums += tl.load(
            partial_ptr + offsets,
            mask=(chunk[:, None] < num_chunks) & (d[None, :] < dim),
            other=0.0,
        )
    total = tl.sum(sums, axis=0)
    if parameter == 0:
        tl.store(dbias_ptr + d, total, mask=d < dim)
    else:
        tl.store(dweight_ptr + d * WIDTH + parameter - 1, total, mask=d < dim)



@triton.jit
def _store_dinitial_recompute(
    x_ptr,
    weight_ptr,
    bias_ptr,
    initial_ptr,
    bos_ptr,
    dout_ptr,
    dinitial_ptr,
    b,
    d,
    d_mask,
    seqlen,
    dim,
    WIDTH: tl.constexpr,
    HAS_BOS: tl.constexpr,
    ACTIVATION_SILU: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Recompute the at-most-three boundary values that affect initial state."""
    weight_base = d * WIDTH
    w0 = tl.load(weight_ptr + weight_base, mask=d_mask, other=0.0).to(tl.float32)
    if WIDTH >= 3:
        w1 = tl.load(weight_ptr + weight_base + 1, mask=d_mask, other=0.0).to(
            tl.float32
        )
    if WIDTH >= 4:
        w2 = tl.load(weight_ptr + weight_base + 2, mask=d_mask, other=0.0).to(
            tl.float32
        )

    init_base = (b * dim + d) * (WIDTH - 1)
    if ACTIVATION_SILU:
        w_last = tl.load(
            weight_ptr + weight_base + WIDTH - 1,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        bias = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
        prev1 = tl.load(
            initial_ptr + init_base + WIDTH - 2,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        if WIDTH >= 3:
            prev2 = tl.load(
                initial_ptr + init_base + WIDTH - 3,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
        if WIDTH >= 4:
            prev3 = tl.load(
                initial_ptr + init_base,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)

    di0 = tl.zeros((BLOCK_D,), tl.float32)
    if WIDTH >= 3:
        di1 = tl.zeros((BLOCK_D,), tl.float32)
    if WIDTH >= 4:
        di2 = tl.zeros((BLOCK_D,), tl.float32)
    initial_alive = True
    bos_base = b * seqlen
    row_base = b * seqlen * dim + d

    for i in tl.static_range(0, WIDTH - 1):
        if HAS_BOS:
            bos_t = tl.load(bos_ptr + bos_base + i)
            keep_prev = ~bos_t
            initial_alive = initial_alive & keep_prev
        else:
            keep_prev = True

        token_base = row_base + i * dim
        dout = tl.load(
            dout_ptr + token_base, mask=d_mask, other=0.0
        ).to(tl.float32)
        if ACTIVATION_SILU:
            x0 = tl.load(
                x_ptr + token_base,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            src1 = tl.where(keep_prev, prev1, 0.0)
            if WIDTH == 2:
                z = x0 * w_last + src1 * w0 + bias
            elif WIDTH == 3:
                src2 = tl.where(keep_prev, prev2, 0.0)
                z = x0 * w_last + src1 * w1 + src2 * w0 + bias
            else:
                src2 = tl.where(keep_prev, prev2, 0.0)
                src3 = tl.where(keep_prev, prev3, 0.0)
                z = x0 * w_last + src1 * w2 + src2 * w1 + src3 * w0 + bias
            g = _silu_backward(z, dout)
        else:
            g = dout

        if i == 0:
            di0 = tl.where(initial_alive, g * w0, 0.0)
            if WIDTH >= 3:
                di1 = tl.where(initial_alive, g * w1, 0.0)
            if WIDTH >= 4:
                di2 = tl.where(initial_alive, g * w2, 0.0)
        elif i == 1:
            if WIDTH >= 3:
                di1 += tl.where(initial_alive, g * w0, 0.0)
            if WIDTH >= 4:
                di2 += tl.where(initial_alive, g * w1, 0.0)
        else:
            if WIDTH >= 4:
                di2 += tl.where(initial_alive, g * w0, 0.0)

        if ACTIVATION_SILU:
            if WIDTH >= 4:
                prev3 = src2
            if WIDTH >= 3:
                prev2 = src1
            prev1 = x0

    tl.store(dinitial_ptr + init_base, di0, mask=d_mask)
    if WIDTH >= 3:
        tl.store(dinitial_ptr + init_base + 1, di1, mask=d_mask)
    if WIDTH >= 4:
        tl.store(dinitial_ptr + init_base + 2, di2, mask=d_mask)


@triton.jit
def _tail_g(
    x_ptr,
    weight_ptr,
    bias_ptr,
    initial_ptr,
    bos_ptr,
    dout_ptr,
    b,
    q,
    d,
    d_mask,
    seqlen,
    dim,
    WIDTH: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    HAS_BOS: tl.constexpr,
    ACTIVATION_SILU: tl.constexpr,
):
    base = (b * seqlen + q) * dim + d
    dout = tl.load(dout_ptr + base, mask=d_mask, other=0.0).to(tl.float32)
    if not ACTIVATION_SILU:
        return dout

    z = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    weight_base = d * WIDTH
    init_base = (b * dim + d) * (WIDTH - 1)
    for lag in tl.static_range(0, WIDTH):
        source_t = q - lag
        visible = True
        if HAS_BOS:
            for reset_lag in tl.static_range(0, lag):
                reset_t = q - reset_lag
                reset = tl.load(
                    bos_ptr + b * seqlen + reset_t,
                    mask=reset_t >= 0,
                    other=False,
                )
                visible = visible & ((reset_t < 0) | (~reset))
        source_base = (b * seqlen + source_t) * dim + d
        source = tl.load(
            x_ptr + source_base,
            mask=d_mask & (source_t >= 0) & visible,
            other=0.0,
        ).to(tl.float32)
        if HAS_INITIAL:
            source += tl.load(
                initial_ptr + init_base + source_t + WIDTH - 1,
                mask=d_mask & (source_t < 0) & visible,
                other=0.0,
            ).to(tl.float32)
        weight = tl.load(
            weight_ptr + weight_base + WIDTH - 1 - lag,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        z += source * weight
    return _silu_backward(z, dout)


@triton.jit
def _dfinal_tail_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    initial_ptr,
    bos_ptr,
    dout_ptr,
    dfinal_ptr,
    dinitial_ptr,
    dx_ptr,
    accum_ptr,
    dweight_ptr,
    dbias_ptr,
    seqlen,
    dim,
    WIDTH: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    HAS_BOS: tl.constexpr,
    ACTIVATION_SILU: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DETERMINISTIC: tl.constexpr,
    REDUCE_CHUNKS: tl.constexpr,
):
    pid_d = tl.program_id(0)
    b = tl.program_id(1)
    d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d < dim
    q0 = seqlen - (WIDTH - 1)
    g0 = _tail_g(
        x_ptr,
        weight_ptr,
        bias_ptr,
        initial_ptr,
        bos_ptr,
        dout_ptr,
        b,
        q0,
        d,
        d_mask,
        seqlen,
        dim,
        WIDTH,
        HAS_INITIAL,
        HAS_BOS,
        ACTIVATION_SILU,
    )
    weight_base = d * WIDTH
    w_last = tl.load(
        weight_ptr + weight_base + WIDTH - 1,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    df_base = (b * dim + d) * (WIDTH - 1)
    df0 = tl.load(dfinal_ptr + df_base, mask=d_mask, other=0.0).to(tl.float32)

    if WIDTH == 2:
        dx0 = g0 * w_last + df0
        tl.store(dx_ptr + (b * seqlen + q0) * dim + d, dx0, mask=d_mask)
    elif WIDTH == 3:
        g1 = _tail_g(
            x_ptr,
            weight_ptr,
            bias_ptr,
            initial_ptr,
            bos_ptr,
            dout_ptr,
            b,
            q0 + 1,
            d,
            d_mask,
            seqlen,
            dim,
            WIDTH,
            HAS_INITIAL,
            HAS_BOS,
            ACTIVATION_SILU,
        )
        w_lag1 = tl.load(
            weight_ptr + weight_base + WIDTH - 2,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_BOS:
            clear1 = ~tl.load(bos_ptr + b * seqlen + q0 + 1)
        else:
            clear1 = True
        df1 = tl.load(dfinal_ptr + df_base + 1, mask=d_mask, other=0.0).to(
            tl.float32
        )
        dx0 = g0 * w_last + tl.where(clear1, g1 * w_lag1, 0.0)
        dx0 += tl.where(clear1, df0, 0.0)
        dx1 = g1 * w_last + df1
        tl.store(dx_ptr + (b * seqlen + q0) * dim + d, dx0, mask=d_mask)
        tl.store(dx_ptr + (b * seqlen + q0 + 1) * dim + d, dx1, mask=d_mask)
    else:
        g1 = _tail_g(
            x_ptr,
            weight_ptr,
            bias_ptr,
            initial_ptr,
            bos_ptr,
            dout_ptr,
            b,
            q0 + 1,
            d,
            d_mask,
            seqlen,
            dim,
            WIDTH,
            HAS_INITIAL,
            HAS_BOS,
            ACTIVATION_SILU,
        )
        g2 = _tail_g(
            x_ptr,
            weight_ptr,
            bias_ptr,
            initial_ptr,
            bos_ptr,
            dout_ptr,
            b,
            q0 + 2,
            d,
            d_mask,
            seqlen,
            dim,
            WIDTH,
            HAS_INITIAL,
            HAS_BOS,
            ACTIVATION_SILU,
        )
        w_lag1 = tl.load(
            weight_ptr + weight_base + WIDTH - 2,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        w_lag2 = tl.load(
            weight_ptr + weight_base + WIDTH - 3,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_BOS:
            clear1 = ~tl.load(bos_ptr + b * seqlen + q0 + 1)
            clear2_tail = ~tl.load(bos_ptr + b * seqlen + q0 + 2)
            clear2 = clear1 & clear2_tail
        else:
            clear1 = True
            clear2_tail = True
            clear2 = True
        df1 = tl.load(dfinal_ptr + df_base + 1, mask=d_mask, other=0.0).to(
            tl.float32
        )
        df2 = tl.load(dfinal_ptr + df_base + 2, mask=d_mask, other=0.0).to(
            tl.float32
        )
        dx0 = g0 * w_last
        dx0 += tl.where(clear1, g1 * w_lag1, 0.0)
        dx0 += tl.where(clear2, g2 * w_lag2, 0.0)
        dx0 += tl.where(clear2, df0, 0.0)
        dx1 = g1 * w_last + tl.where(clear2_tail, g2 * w_lag1, 0.0)
        dx1 += tl.where(clear2_tail, df1, 0.0)
        dx2 = g2 * w_last + df2
        tl.store(dx_ptr + (b * seqlen + q0) * dim + d, dx0, mask=d_mask)
        tl.store(dx_ptr + (b * seqlen + q0 + 1) * dim + d, dx1, mask=d_mask)
        tl.store(dx_ptr + (b * seqlen + q0 + 2) * dim + d, dx2, mask=d_mask)

    if HAS_INITIAL:
        _store_dinitial_recompute(
            x_ptr,
            weight_ptr,
            bias_ptr,
            initial_ptr,
            bos_ptr,
            dout_ptr,
            dinitial_ptr,
            b,
            d,
            d_mask,
            seqlen,
            dim,
            WIDTH,
            HAS_BOS,
            ACTIVATION_SILU,
            BLOCK_D,
        )

    if not DETERMINISTIC:
        final_mask = d_mask & (b == 0)
        db = tl.load(accum_ptr + d, mask=final_mask, other=0.0)
        tl.store(dbias_ptr + d, db, mask=final_mask)
        dw0 = tl.load(accum_ptr + dim + d, mask=final_mask, other=0.0)
        dw1 = tl.load(accum_ptr + 2 * dim + d, mask=final_mask, other=0.0)
        tl.store(dweight_ptr + d * WIDTH, dw0, mask=final_mask)
        tl.store(dweight_ptr + d * WIDTH + 1, dw1, mask=final_mask)
        if WIDTH >= 3:
            dw2 = tl.load(accum_ptr + 3 * dim + d, mask=final_mask, other=0.0)
            tl.store(dweight_ptr + d * WIDTH + 2, dw2, mask=final_mask)
        if WIDTH >= 4:
            dw3 = tl.load(accum_ptr + 4 * dim + d, mask=final_mask, other=0.0)
            tl.store(dweight_ptr + d * WIDTH + 3, dw3, mask=final_mask)
    elif REDUCE_CHUNKS > 0:
        # The main launch has completed; only b=0 writes parameter gradients.
        final_mask = d_mask & (b == 0)
        for parameter in tl.static_range(WIDTH + 1):
            total = tl.load(accum_ptr + parameter * dim + d, mask=final_mask, other=0.0)
            for chunk in tl.static_range(1, REDUCE_CHUNKS):
                total += tl.load(
                    accum_ptr + (chunk * (WIDTH + 1) + parameter) * dim + d,
                    mask=final_mask, other=0.0,
                )
            if parameter == 0:
                tl.store(dbias_ptr + d, total, mask=final_mask)
            else:
                tl.store(dweight_ptr + d * WIDTH + parameter - 1, total, mask=final_mask)


def _direct_path_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    bos_mask: torch.Tensor | None,
    activation: str | None,
    dout: torch.Tensor | None,
    dfinal_states: torch.Tensor | None,
) -> bool:
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.ndim != 3
        or weight.ndim != 2
        or bias is None
        or dout is None
        or dfinal_states is None
        or activation not in (None, "silu")
    ):
        return False

    batch, seqlen, dim = x.shape
    width = weight.shape[1]
    if (
        width not in (2, 3, 4)
        or seqlen < width - 1
        or weight.shape != (dim, width)
        or bias.shape != (dim,)
        or dout.shape != x.shape
        or dfinal_states.shape != (batch, dim, width - 1)
    ):
        return False
    if initial_states is not None and initial_states.shape != (
        batch,
        dim,
        width - 1,
    ):
        return False
    if bos_mask is not None and (
        bos_mask.shape != (batch, seqlen)
        or bos_mask.dtype != torch.bool
        or bos_mask.device != x.device
        or not bos_mask.is_contiguous()
    ):
        return False

    tensors = (x, weight, bias, dout, dfinal_states)
    if initial_states is not None:
        tensors = (*tensors, initial_states)
    return all(
        tensor.device == x.device
        and tensor.dtype == torch.bfloat16
        and tensor.is_contiguous()
        for tensor in tensors
    )


def _direct_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    initial_states: torch.Tensor | None,
    bos_mask: torch.Tensor | None,
    activation: str | None,
    dout: torch.Tensor,
    dfinal_states: torch.Tensor,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    batch, seqlen, dim = x.shape
    width = weight.shape[1]
    block_n = 256
    block_d = 64
    chunks_per_batch = triton.cdiv(seqlen, block_n)
    num_chunks = batch * chunks_per_batch
    # One or two partials fit in the existing tail kernel. This also avoids a
    # zeroing launch for small calls that permit nondeterministic algorithms.
    tail_reduce_chunks = num_chunks if num_chunks <= 2 else 0
    deterministic = deterministic or tail_reduce_chunks > 0

    dx = torch.empty(
        (batch, seqlen, dim), device=x.device, dtype=torch.bfloat16
    )
    dweight = torch.empty_like(weight)
    dbias = torch.empty_like(bias)
    dinitial = torch.empty_like(initial_states) if initial_states is not None else None
    # Per-call scratch also keeps independent CUDA graph captures isolated.
    accum = (
        torch.empty(
            (num_chunks, width + 1, dim),
            device=x.device,
            dtype=torch.float32,
        )
        if deterministic
        else torch.zeros((width + 1, dim), device=x.device, dtype=torch.float32)
    )
    dummy = x

    _fused_streaming_bwd_kernel[
        (triton.cdiv(dim, block_d), batch * chunks_per_batch)
    ](
        x,
        weight,
        bias,
        initial_states if initial_states is not None else dummy,
        bos_mask if bos_mask is not None else dummy,
        dout,
        dx,
        accum,
        seqlen,
        dim,
        chunks_per_batch,
        width,
        initial_states is not None,
        bos_mask is not None,
        activation == "silu",
        block_n,
        block_d,
        2 if bos_mask is not None or width == 3 else 1,
        deterministic,
        num_warps=1,
        num_stages=2,
    )
    tail_block_d = 256
    _dfinal_tail_kernel[(triton.cdiv(dim, tail_block_d), batch)](
        x,
        weight,
        bias,
        initial_states if initial_states is not None else dummy,
        bos_mask if bos_mask is not None else dummy,
        dout,
        dfinal_states,
        dinitial if dinitial is not None else dummy,
        dx,
        accum,
        dweight,
        dbias,
        seqlen,
        dim,
        width,
        initial_states is not None,
        bos_mask is not None,
        activation == "silu",
        tail_block_d,
        deterministic,
        tail_reduce_chunks,
        num_warps=4,
        num_stages=1,
    )
    if deterministic and not tail_reduce_chunks:
        # Narrower channel tiles expose enough blocks to hide memory latency
        # when reducing hundreds of chunks, while retaining a fixed sum order.
        reduce_block_d = 16
        _reduce_parameter_grads_kernel[(triton.cdiv(dim, reduce_block_d), width + 1)](
            accum,
            dweight,
            dbias,
            dim,
            num_chunks,
            width,
            32,
            reduce_block_d,
            num_warps=4,
        )
    return dx, dweight, dbias, dinitial


def _reference_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    bos_mask: torch.Tensor | None,
    activation: str | None,
    dout: torch.Tensor | None,
    dfinal_states: torch.Tensor | None,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if dout is None:
        raise TypeError("dout is required for causal_conv1d backward")

    public_initial_state = (
        initial_states.transpose(1, 2).contiguous()
        if initial_states is not None
        else None
    )
    public_final_state_grad = (
        dfinal_states.transpose(1, 2).contiguous()
        if dfinal_states is not None
        else None
    )
    x_grad, initial_state_grad, weight_grad, bias_grad = (
        causal_conv1d_reference_bwd(
            dout,
            public_final_state_grad,
            x,
            weight,
            bias,
            public_initial_state,
            bos_mask,
            activation,
            deterministic=deterministic,
        )
    )
    return (
        x_grad,
        weight_grad,
        bias_grad,
        initial_state_grad.transpose(1, 2).contiguous()
        if initial_state_grad is not None
        else None,
    )


def kernel_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    bos_mask: torch.Tensor | None = None,
    activation: str | None = None,
    dout: torch.Tensor | None = None,
    dfinal_states: torch.Tensor | None = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Enable fixed-order parameter reductions with ``deterministic=True``.

    PyTorch's global deterministic setting takes precedence over the opt-out.
    """
    deterministic = deterministic or torch.are_deterministic_algorithms_enabled()
    if _direct_path_supported(
        x,
        weight,
        bias,
        initial_states,
        bos_mask,
        activation,
        dout,
        dfinal_states,
    ):
        assert bias is not None and dout is not None and dfinal_states is not None
        return _direct_backward(
            x,
            weight,
            bias,
            initial_states,
            bos_mask,
            activation,
            dout,
            dfinal_states,
            deterministic,
        )

    return _reference_backward(
        x,
        weight,
        bias,
        initial_states,
        bos_mask,
        activation,
        dout,
        dfinal_states,
        deterministic,
    )
