"""Triton implementation of causal depthwise conv1d forward."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from xllm.modules.fused_ops.conv.triton.causal_conv1d_reference import causal_conv1d_reference_fwd


def reference_kernel_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    bos_mask: torch.Tensor | None = None,
    activation: str | None = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt the kernel's internal tensor layouts to the shared reference."""
    public_weight = weight.transpose(0, 1).contiguous()
    public_initial_state = (
        initial_states.transpose(1, 2).contiguous()
        if initial_states is not None
        else None
    )
    out, final_state = causal_conv1d_reference_fwd(
        x,
        public_weight,
        bias,
        public_initial_state,
        bos_mask,
        activation,
        deterministic=deterministic or torch.are_deterministic_algorithms_enabled(),
    )
    return out, final_state.transpose(1, 2).contiguous()


@triton.jit
def _fast_silu(x):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 y;
            .reg .f32 e;
            .reg .f32 denom;
            .reg .f32 inv;
            fma.rn.ftz.f32 y, $1, 0fBFB8AA3B, 0f00000000;
            ex2.approx.ftz.f32 e, y;
            add.rn.ftz.f32 denom, e, 0f3F800000;
            rcp.approx.ftz.f32 inv, denom;
            fma.rn.ftz.f32 $0, $1, inv, 0f00000000;
        }
        """,
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _prefetch_l1(ptr):
    return tl.inline_asm_elementwise(
        asm="""
        {
            prefetch.global.L1 [$1];
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,l",
        args=[ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


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
def _conv1d_out_kernel(
    x_ptr,
    weight_t_ptr,
    bias_ptr,
    init_ptr,
    bos_ptr,
    out_ptr,
    rows_per_batch: tl.constexpr,
    start_t: tl.constexpr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    width: tl.constexpr,
    has_bias: tl.constexpr,
    has_init: tl.constexpr,
    has_bos: tl.constexpr,
    activation_silu: tl.constexpr,
    num_d_blocks: tl.constexpr,
    block_d: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_d = pid % num_d_blocks
    row = pid // num_d_blocks
    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim

    b = row // rows_per_batch
    t = row - b * rows_per_batch + start_t
    token = b * seqlen + t
    base = token * dim + d

    acc = tl.load(x_ptr + base, mask=d_mask, other=0.0).to(tl.float32)
    w = tl.load(weight_t_ptr + (width - 1) * dim + d, mask=d_mask, other=0.0).to(
        tl.float32
    )
    acc *= w

    if has_bos:
        bos_base = b * seqlen
        bos_t = tl.load(bos_ptr + bos_base + t)
    else:
        bos_base = 0
        bos_t = False

    if width >= 2:
        lag = 1
        w1 = tl.load(weight_t_ptr + (width - 1 - lag) * dim + d, mask=d_mask, other=0.0).to(
            tl.float32
        )
        src_valid = t >= lag
        if has_bos:
            src_valid = src_valid & (~bos_t)
        x1 = tl.load(x_ptr + base - lag * dim, mask=d_mask & src_valid, other=0.0).to(
            tl.float32
        )
        if has_init:
            init_idx = (width - 1) + t - lag
            init_valid = t < lag
            if has_bos:
                bos0 = tl.load(bos_ptr + bos_base, mask=seqlen >= 1, other=0)
                init_valid = init_valid & (~bos0)
            init_val = tl.load(
                init_ptr + b * dim * (width - 1) + d * (width - 1) + init_idx,
                mask=d_mask & init_valid,
                other=0.0,
            ).to(tl.float32)
            x1 = tl.where(src_valid, x1, init_val)
        acc += x1 * w1

    if width >= 3:
        lag = 2
        w2 = tl.load(weight_t_ptr + (width - 1 - lag) * dim + d, mask=d_mask, other=0.0).to(
            tl.float32
        )
        src_valid = t >= lag
        if has_bos:
            bos_tm1 = tl.load(
                bos_ptr + bos_base + t - 1, mask=t >= 1, other=0
            )
            src_valid = src_valid & (~bos_t) & (~bos_tm1)
        x2 = tl.load(x_ptr + base - lag * dim, mask=d_mask & src_valid, other=0.0).to(
            tl.float32
        )
        if has_init:
            init_idx = (width - 1) + t - lag
            init_valid = t < lag
            if has_bos:
                bos0 = tl.load(bos_ptr + bos_base, mask=seqlen >= 1, other=0)
                bos1 = tl.load(bos_ptr + bos_base + 1, mask=(t >= 1) & (seqlen >= 2), other=0)
                init_valid = init_valid & (~bos0) & ((t < 1) | (~bos1))
            init_val = tl.load(
                init_ptr + b * dim * (width - 1) + d * (width - 1) + init_idx,
                mask=d_mask & init_valid,
                other=0.0,
            ).to(tl.float32)
            x2 = tl.where(src_valid, x2, init_val)
        acc += x2 * w2

    if width >= 4:
        lag = 3
        w3 = tl.load(weight_t_ptr + (width - 1 - lag) * dim + d, mask=d_mask, other=0.0).to(
            tl.float32
        )
        src_valid = t >= lag
        if has_bos:
            bos_tm1 = tl.load(
                bos_ptr + bos_base + t - 1, mask=t >= 1, other=0
            )
            bos_tm2 = tl.load(
                bos_ptr + bos_base + t - 2, mask=t >= 2, other=0
            )
            src_valid = src_valid & (~bos_t) & (~bos_tm1) & (~bos_tm2)
        x3 = tl.load(x_ptr + base - lag * dim, mask=d_mask & src_valid, other=0.0).to(
            tl.float32
        )
        if has_init:
            init_idx = (width - 1) + t - lag
            init_valid = t < lag
            if has_bos:
                bos0 = tl.load(bos_ptr + bos_base, mask=seqlen >= 1, other=0)
                bos1 = tl.load(bos_ptr + bos_base + 1, mask=(t >= 1) & (seqlen >= 2), other=0)
                bos2 = tl.load(bos_ptr + bos_base + 2, mask=(t >= 2) & (seqlen >= 3), other=0)
                init_valid = init_valid & (~bos0) & ((t < 1) | (~bos1)) & ((t < 2) | (~bos2))
            init_val = tl.load(
                init_ptr + b * dim * (width - 1) + d * (width - 1) + init_idx,
                mask=d_mask & init_valid,
                other=0.0,
            ).to(tl.float32)
            x3 = tl.where(src_valid, x3, init_val)
        acc += x3 * w3

    if has_bias:
        acc += tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    if activation_silu:
        acc = _fast_silu(acc)

    tl.store(out_ptr + base, acc, mask=d_mask)


@triton.jit
def _conv1d_prefix_t3_kernel(
    x_ptr,
    weight_t_ptr,
    bias_ptr,
    init_ptr,
    bos_ptr,
    out_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    num_d_blocks: tl.constexpr,
    block_d: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_d = pid % num_d_blocks
    b = pid // num_d_blocks

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim

    bos_base = b * seqlen
    bos_t0 = tl.load(bos_ptr + bos_base)
    bos_t1 = tl.load(bos_ptr + bos_base + 1)
    bos_t2 = tl.load(bos_ptr + bos_base + 2)

    token0 = b * seqlen
    base0 = token0 * dim + d
    base1 = base0 + dim
    base2 = base1 + dim

    x_t0 = tl.load(x_ptr + base0, mask=d_mask, other=0.0).to(tl.float32)
    x_t1 = tl.load(x_ptr + base1, mask=d_mask, other=0.0).to(tl.float32)
    x_t2 = tl.load(x_ptr + base2, mask=d_mask, other=0.0).to(tl.float32)

    init_base = b * dim * 3 + d * 3
    init0 = tl.load(init_ptr + init_base, mask=d_mask, other=0.0).to(tl.float32)
    init1 = tl.load(init_ptr + init_base + 1, mask=d_mask, other=0.0).to(tl.float32)
    init2 = tl.load(init_ptr + init_base + 2, mask=d_mask, other=0.0).to(tl.float32)

    w0 = tl.load(weight_t_ptr + 3 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w1 = tl.load(weight_t_ptr + 2 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w2 = tl.load(weight_t_ptr + dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w3 = tl.load(weight_t_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    init_clear0 = ~bos_t0
    acc0 = x_t0 * w0
    acc0 += tl.where(init_clear0, init2, 0.0) * w1
    acc0 += tl.where(init_clear0, init1, 0.0) * w2
    acc0 += tl.where(init_clear0, init0, 0.0) * w3
    acc0 += bias
    acc0 = _fast_silu(acc0)

    valid1_1 = ~bos_t1
    init_clear1 = (~bos_t0) & (~bos_t1)
    acc1 = x_t1 * w0
    acc1 += tl.where(valid1_1, x_t0, 0.0) * w1
    acc1 += tl.where(init_clear1, init2, 0.0) * w2
    acc1 += tl.where(init_clear1, init1, 0.0) * w3
    acc1 += bias
    acc1 = _fast_silu(acc1)

    valid2_1 = ~bos_t2
    valid2_2 = valid2_1 & (~bos_t1)
    init_clear2 = (~bos_t0) & (~bos_t1) & (~bos_t2)
    acc2 = x_t2 * w0
    acc2 += tl.where(valid2_1, x_t1, 0.0) * w1
    acc2 += tl.where(valid2_2, x_t0, 0.0) * w2
    acc2 += tl.where(init_clear2, init2, 0.0) * w3
    acc2 += bias
    acc2 = _fast_silu(acc2)

    tl.store(out_ptr + base0, acc0, mask=d_mask)
    tl.store(out_ptr + base1, acc1, mask=d_mask)
    tl.store(out_ptr + base2, acc2, mask=d_mask)


@triton.jit
def _conv1d_main_kernel(
    x_ptr,
    weight_t_ptr,
    bias_ptr,
    bos_ptr,
    out_ptr,
    rows_per_batch: tl.constexpr,
    start_t: tl.constexpr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    width: tl.constexpr,
    has_bias: tl.constexpr,
    has_bos: tl.constexpr,
    activation_silu: tl.constexpr,
    num_d_blocks: tl.constexpr,
    block_d: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_d = pid % num_d_blocks
    row = pid // num_d_blocks
    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim

    b = row // rows_per_batch
    t = row - b * rows_per_batch + start_t
    token = b * seqlen + t
    base = token * dim + d

    acc = tl.load(x_ptr + base, mask=d_mask, other=0.0).to(tl.float32)
    w = tl.load(weight_t_ptr + (width - 1) * dim + d, mask=d_mask, other=0.0).to(
        tl.float32
    )
    acc *= w

    if has_bos:
        bos_base = b * seqlen
        bos_t = tl.load(bos_ptr + bos_base + t)
    else:
        bos_base = 0
        bos_t = False

    if width >= 2:
        lag = 1
        w1 = tl.load(weight_t_ptr + (width - 1 - lag) * dim + d, mask=d_mask, other=0.0).to(
            tl.float32
        )
        valid1 = True
        if has_bos:
            valid1 = ~bos_t
        x1 = tl.load(x_ptr + base - lag * dim, mask=d_mask & valid1, other=0.0).to(
            tl.float32
        )
        acc += x1 * w1

    if width >= 3:
        lag = 2
        w2 = tl.load(weight_t_ptr + (width - 1 - lag) * dim + d, mask=d_mask, other=0.0).to(
            tl.float32
        )
        valid2 = True
        if has_bos:
            bos_tm1 = tl.load(bos_ptr + bos_base + t - 1)
            valid2 = (~bos_t) & (~bos_tm1)
        x2 = tl.load(x_ptr + base - lag * dim, mask=d_mask & valid2, other=0.0).to(
            tl.float32
        )
        acc += x2 * w2

    if width >= 4:
        lag = 3
        w3 = tl.load(weight_t_ptr + (width - 1 - lag) * dim + d, mask=d_mask, other=0.0).to(
            tl.float32
        )
        valid3 = True
        if has_bos:
            bos_tm1 = tl.load(bos_ptr + bos_base + t - 1)
            bos_tm2 = tl.load(bos_ptr + bos_base + t - 2)
            valid3 = (~bos_t) & (~bos_tm1) & (~bos_tm2)
        x3 = tl.load(x_ptr + base - lag * dim, mask=d_mask & valid3, other=0.0).to(
            tl.float32
        )
        acc += x3 * w3

    if has_bias:
        acc += tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    if activation_silu:
        acc = _fast_silu(acc)

    tl.store(out_ptr + base, acc, mask=d_mask)


@triton.jit
def _conv1d_main_t8_kernel(
    x_ptr,
    weight_t_ptr,
    bias_ptr,
    bos_ptr,
    out_ptr,
    rows_per_batch: tl.constexpr,
    tile_rows_per_batch: tl.constexpr,
    start_t: tl.constexpr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    num_d_blocks: tl.constexpr,
    block_d: tl.constexpr,
    full_tile: tl.constexpr,
    store_bf16: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_d = pid % num_d_blocks
    tile_row = pid // num_d_blocks
    b = tile_row // tile_rows_per_batch
    tt = tile_row - b * tile_rows_per_batch
    main_t0 = tt * 8
    t0 = main_t0 + start_t
    t1 = t0 + 1
    t2 = t0 + 2
    t3 = t0 + 3
    t4 = t0 + 4
    t5 = t0 + 5
    t6 = t0 + 6
    t7 = t0 + 7
    if full_tile:
        t1_valid = True
        t2_valid = True
        t3_valid = True
        t4_valid = True
        t5_valid = True
        t6_valid = True
        t7_valid = True
    else:
        t1_valid = (main_t0 + 1) < rows_per_batch
        t2_valid = (main_t0 + 2) < rows_per_batch
        t3_valid = (main_t0 + 3) < rows_per_batch
        t4_valid = (main_t0 + 4) < rows_per_batch
        t5_valid = (main_t0 + 5) < rows_per_batch
        t6_valid = (main_t0 + 6) < rows_per_batch
        t7_valid = (main_t0 + 7) < rows_per_batch

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim

    bos_base = b * seqlen
    bos_tm2 = tl.load(bos_ptr + bos_base + t0 - 2)
    bos_tm1 = tl.load(bos_ptr + bos_base + t0 - 1)
    bos_t0 = tl.load(bos_ptr + bos_base + t0)
    if full_tile:
        bos_t1 = tl.load(bos_ptr + bos_base + t1)
        bos_t2 = tl.load(bos_ptr + bos_base + t2)
        bos_t3 = tl.load(bos_ptr + bos_base + t3)
        bos_t4 = tl.load(bos_ptr + bos_base + t4)
        bos_t5 = tl.load(bos_ptr + bos_base + t5)
        bos_t6 = tl.load(bos_ptr + bos_base + t6)
        bos_t7 = tl.load(bos_ptr + bos_base + t7)
    else:
        bos_t1 = tl.load(bos_ptr + bos_base + t1, mask=t1_valid, other=0)
        bos_t2 = tl.load(bos_ptr + bos_base + t2, mask=t2_valid, other=0)
        bos_t3 = tl.load(bos_ptr + bos_base + t3, mask=t3_valid, other=0)
        bos_t4 = tl.load(bos_ptr + bos_base + t4, mask=t4_valid, other=0)
        bos_t5 = tl.load(bos_ptr + bos_base + t5, mask=t5_valid, other=0)
        bos_t6 = tl.load(bos_ptr + bos_base + t6, mask=t6_valid, other=0)
        bos_t7 = tl.load(bos_ptr + bos_base + t7, mask=t7_valid, other=0)

    token0 = b * seqlen + t0
    base0 = token0 * dim + d
    base1 = base0 + dim
    base2 = base1 + dim
    base3 = base2 + dim
    base4 = base3 + dim
    base5 = base4 + dim
    base6 = base5 + dim
    base7 = base6 + dim

    if full_tile:
        x_tm3 = tl.load(
            x_ptr + base0 - 3 * dim,
            mask=d_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        x_tm2 = tl.load(
            x_ptr + base0 - 2 * dim,
            mask=d_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        x_tm1 = tl.load(
            x_ptr + base0 - dim,
            mask=d_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        x_t0 = tl.load(
            x_ptr + base0,
            mask=d_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        x_t1 = tl.load(
            x_ptr + base1,
            mask=d_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
    else:
        x_tm3 = tl.load(x_ptr + base0 - 3 * dim, mask=d_mask, other=0.0).to(
            tl.float32
        )
        x_tm2 = tl.load(x_ptr + base0 - 2 * dim, mask=d_mask, other=0.0).to(
            tl.float32
        )
        x_tm1 = tl.load(x_ptr + base0 - dim, mask=d_mask, other=0.0).to(tl.float32)
        x_t0 = tl.load(x_ptr + base0, mask=d_mask, other=0.0).to(tl.float32)
        x_t1 = tl.load(
            x_ptr + base1,
            mask=d_mask & t1_valid,
            other=0.0,
        ).to(tl.float32)

    w0 = tl.load(weight_t_ptr + 3 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w1 = tl.load(weight_t_ptr + 2 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w2 = tl.load(weight_t_ptr + dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w3 = tl.load(weight_t_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    if full_tile:
        no_bos = ~(
            bos_tm2
            | bos_tm1
            | bos_t0
            | bos_t1
            | bos_t2
            | bos_t3
            | bos_t4
            | bos_t5
            | bos_t6
            | bos_t7
        )
        if no_bos:
            x_t2 = tl.load(
                x_ptr + base2,
                mask=d_mask,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            x_t3 = tl.load(
                x_ptr + base3,
                mask=d_mask,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            _prefetch_l1(x_ptr + base4)
            x_m3 = x_tm3
            x_m2 = x_tm2
            x_m1 = x_tm1
            x_cur = x_t0
            x_next1 = x_t1
            x_next2 = x_t2
            x_next3 = x_t3
            for i in tl.static_range(0, 8):
                acc = x_cur * w0 + bias
                acc += x_m1 * w1
                acc += x_m2 * w2
                acc += x_m3 * w3
                acc = _fast_silu(acc)
                if store_bf16:
                    store_acc = acc.to(tl.bfloat16)
                else:
                    store_acc = acc
                tl.store(
                    out_ptr + base0 + i * dim,
                    store_acc,
                    mask=d_mask,
                    cache_modifier=".cg",
                )

                if i < 4:
                    x_new = tl.load(
                        x_ptr + base0 + (i + 4) * dim,
                        mask=d_mask,
                        other=0.0,
                        cache_modifier=".cg",
                    ).to(tl.float32)
                else:
                    x_new = x_next3
                x_m3 = x_m2
                x_m2 = x_m1
                x_m1 = x_cur
                x_cur = x_next1
                x_next1 = x_next2
                x_next2 = x_next3
                x_next3 = x_new
            return

    x_m3 = x_tm3
    x_m2 = x_tm2
    x_m1 = x_tm1
    x_cur = x_t0
    x_next1 = x_t1
    bos_m2 = bos_tm2
    bos_m1 = bos_tm1
    bos_cur = bos_t0
    bos_next1 = bos_t1
    row_valid_cur = True
    row_valid_next1 = t1_valid

    for i in tl.static_range(0, 8):
        valid1 = ~bos_cur
        valid2 = valid1 & (~bos_m1)
        valid3 = valid2 & (~bos_m2)
        acc = x_cur * w0 + bias
        acc += tl.where(valid1, x_m1, 0.0) * w1
        acc += tl.where(valid2, x_m2, 0.0) * w2
        acc += tl.where(valid3, x_m3, 0.0) * w3
        acc = _fast_silu(acc)
        if store_bf16:
            store_acc = acc.to(tl.bfloat16)
        else:
            store_acc = acc
        tl.store(
            out_ptr + base0 + i * dim,
            store_acc,
            mask=d_mask & row_valid_cur,
            cache_modifier=".cg",
        )

        if i < 6:
            if i == 0:
                bos_new = bos_t2
                row_valid_new = t2_valid
            elif i == 1:
                bos_new = bos_t3
                row_valid_new = t3_valid
            elif i == 2:
                bos_new = bos_t4
                row_valid_new = t4_valid
            elif i == 3:
                bos_new = bos_t5
                row_valid_new = t5_valid
            elif i == 4:
                bos_new = bos_t6
                row_valid_new = t6_valid
            else:
                bos_new = bos_t7
                row_valid_new = t7_valid

            if full_tile:
                x_new = tl.load(
                    x_ptr + base0 + (i + 2) * dim,
                    mask=d_mask,
                    other=0.0,
                    cache_modifier=".cg",
                ).to(tl.float32)
            else:
                x_new = tl.load(
                    x_ptr + base0 + (i + 2) * dim,
                    mask=d_mask & row_valid_new,
                    other=0.0,
                ).to(tl.float32)
        else:
            x_new = x_next1
            bos_new = bos_next1
            row_valid_new = row_valid_next1

        x_m3 = x_m2
        x_m2 = x_m1
        x_m1 = x_cur
        x_cur = x_next1
        x_next1 = x_new
        bos_m2 = bos_m1
        bos_m1 = bos_cur
        bos_cur = bos_next1
        bos_next1 = bos_new
        row_valid_cur = row_valid_next1
        row_valid_next1 = row_valid_new


@triton.jit
def _conv1d_main_t5_tail_kernel(
    x_ptr,
    weight_t_ptr,
    bias_ptr,
    bos_ptr,
    out_ptr,
    final_ptr,
    start_t: tl.constexpr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    num_d_blocks: tl.constexpr,
    block_d: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_d = pid % num_d_blocks
    b = pid // num_d_blocks

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim

    t0 = start_t
    t1 = t0 + 1
    t2 = t0 + 2
    t3 = t0 + 3
    t4 = t0 + 4

    bos_base = b * seqlen
    bos_tm2 = tl.load(bos_ptr + bos_base + t0 - 2)
    bos_tm1 = tl.load(bos_ptr + bos_base + t0 - 1)
    bos_t0 = tl.load(bos_ptr + bos_base + t0)
    bos_t1 = tl.load(bos_ptr + bos_base + t1)
    bos_t2 = tl.load(bos_ptr + bos_base + t2)
    bos_t3 = tl.load(bos_ptr + bos_base + t3)
    bos_t4 = tl.load(bos_ptr + bos_base + t4)

    token0 = b * seqlen + t0
    base0 = token0 * dim + d
    base1 = base0 + dim
    base2 = base1 + dim
    base3 = base2 + dim
    base4 = base3 + dim

    x_tm3 = tl.load(x_ptr + base0 - 3 * dim, mask=d_mask, other=0.0).to(tl.float32)
    x_tm2 = tl.load(x_ptr + base0 - 2 * dim, mask=d_mask, other=0.0).to(tl.float32)
    x_tm1 = tl.load(x_ptr + base0 - dim, mask=d_mask, other=0.0).to(tl.float32)
    x_t0 = tl.load(x_ptr + base0, mask=d_mask, other=0.0).to(tl.float32)
    x_t1 = tl.load(x_ptr + base1, mask=d_mask, other=0.0).to(tl.float32)

    w0 = tl.load(weight_t_ptr + 3 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w1 = tl.load(weight_t_ptr + 2 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w2 = tl.load(weight_t_ptr + dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w3 = tl.load(weight_t_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    valid0_1 = ~bos_t0
    valid0_2 = valid0_1 & (~bos_tm1)
    valid0_3 = valid0_2 & (~bos_tm2)
    acc = x_t0 * w0 + bias
    acc += tl.where(valid0_1, x_tm1, 0.0) * w1
    acc += tl.where(valid0_2, x_tm2, 0.0) * w2
    acc += tl.where(valid0_3, x_tm3, 0.0) * w3
    acc = _fast_silu(acc)
    tl.store(out_ptr + base0, acc, mask=d_mask)

    x_t2 = tl.load(x_ptr + base2, mask=d_mask, other=0.0).to(tl.float32)
    valid1_1 = ~bos_t1
    valid1_2 = valid1_1 & (~bos_t0)
    valid1_3 = valid1_2 & (~bos_tm1)
    acc = x_t1 * w0 + bias
    acc += tl.where(valid1_1, x_t0, 0.0) * w1
    acc += tl.where(valid1_2, x_tm1, 0.0) * w2
    acc += tl.where(valid1_3, x_tm2, 0.0) * w3
    acc = _fast_silu(acc)
    tl.store(out_ptr + base1, acc, mask=d_mask)

    x_t3 = tl.load(x_ptr + base3, mask=d_mask, other=0.0).to(tl.float32)
    valid2_1 = ~bos_t2
    valid2_2 = valid2_1 & (~bos_t1)
    valid2_3 = valid2_2 & (~bos_t0)
    acc = x_t2 * w0 + bias
    acc += tl.where(valid2_1, x_t1, 0.0) * w1
    acc += tl.where(valid2_2, x_t0, 0.0) * w2
    acc += tl.where(valid2_3, x_tm1, 0.0) * w3
    acc = _fast_silu(acc)
    tl.store(out_ptr + base2, acc, mask=d_mask)

    x_t4 = tl.load(x_ptr + base4, mask=d_mask, other=0.0).to(tl.float32)
    valid3_1 = ~bos_t3
    valid3_2 = valid3_1 & (~bos_t2)
    valid3_3 = valid3_2 & (~bos_t1)
    acc = x_t3 * w0 + bias
    acc += tl.where(valid3_1, x_t2, 0.0) * w1
    acc += tl.where(valid3_2, x_t1, 0.0) * w2
    acc += tl.where(valid3_3, x_t0, 0.0) * w3
    acc = _fast_silu(acc)
    tl.store(out_ptr + base3, acc, mask=d_mask)

    valid4_1 = ~bos_t4
    valid4_2 = valid4_1 & (~bos_t3)
    valid4_3 = valid4_2 & (~bos_t2)
    acc = x_t4 * w0 + bias
    acc += tl.where(valid4_1, x_t3, 0.0) * w1
    acc += tl.where(valid4_2, x_t2, 0.0) * w2
    acc += tl.where(valid4_3, x_t1, 0.0) * w3
    acc = _fast_silu(acc)
    tl.store(out_ptr + base4, acc, mask=d_mask)

    final_base = b * dim * 3 + d
    state0 = tl.where((~bos_t3) & (~bos_t4), x_t2, 0.0)
    state1 = tl.where(~bos_t4, x_t3, 0.0)
    tl.store(final_ptr + final_base, state0, mask=d_mask)
    tl.store(final_ptr + final_base + dim, state1, mask=d_mask)
    tl.store(final_ptr + final_base + 2 * dim, x_t4, mask=d_mask)


@triton.jit
def _conv1d_prefix_t3_tail_t5_kernel(
    x_ptr,
    weight_t_ptr,
    bias_ptr,
    init_ptr,
    bos_ptr,
    out_ptr,
    final_ptr,
    tail_start: tl.constexpr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    num_d_blocks: tl.constexpr,
    block_d: tl.constexpr,
    prefix_programs: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < prefix_programs:
        pid_d = pid % num_d_blocks
        b = pid // num_d_blocks

        d = pid_d * block_d + tl.arange(0, block_d)
        d_mask = d < dim

        bos_base = b * seqlen
        bos_t0 = tl.load(bos_ptr + bos_base)
        bos_t1 = tl.load(bos_ptr + bos_base + 1)
        bos_t2 = tl.load(bos_ptr + bos_base + 2)

        token0 = b * seqlen
        base0 = token0 * dim + d
        base1 = base0 + dim
        base2 = base1 + dim

        x_t0 = tl.load(x_ptr + base0, mask=d_mask, other=0.0).to(tl.float32)
        x_t1 = tl.load(x_ptr + base1, mask=d_mask, other=0.0).to(tl.float32)
        x_t2 = tl.load(x_ptr + base2, mask=d_mask, other=0.0).to(tl.float32)

        init_base = b * dim * 3 + d * 3
        init0 = tl.load(init_ptr + init_base, mask=d_mask, other=0.0).to(tl.float32)
        init1 = tl.load(init_ptr + init_base + 1, mask=d_mask, other=0.0).to(tl.float32)
        init2 = tl.load(init_ptr + init_base + 2, mask=d_mask, other=0.0).to(tl.float32)

        w0 = tl.load(weight_t_ptr + 3 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
        w1 = tl.load(weight_t_ptr + 2 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
        w2 = tl.load(weight_t_ptr + dim + d, mask=d_mask, other=0.0).to(tl.float32)
        w3 = tl.load(weight_t_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

        init_clear0 = ~bos_t0
        acc0 = x_t0 * w0
        acc0 += tl.where(init_clear0, init2, 0.0) * w1
        acc0 += tl.where(init_clear0, init1, 0.0) * w2
        acc0 += tl.where(init_clear0, init0, 0.0) * w3
        acc0 += bias
        acc0 = _fast_silu(acc0)

        valid1_1 = ~bos_t1
        init_clear1 = (~bos_t0) & (~bos_t1)
        acc1 = x_t1 * w0
        acc1 += tl.where(valid1_1, x_t0, 0.0) * w1
        acc1 += tl.where(init_clear1, init2, 0.0) * w2
        acc1 += tl.where(init_clear1, init1, 0.0) * w3
        acc1 += bias
        acc1 = _fast_silu(acc1)

        valid2_1 = ~bos_t2
        valid2_2 = valid2_1 & (~bos_t1)
        init_clear2 = (~bos_t0) & (~bos_t1) & (~bos_t2)
        acc2 = x_t2 * w0
        acc2 += tl.where(valid2_1, x_t1, 0.0) * w1
        acc2 += tl.where(valid2_2, x_t0, 0.0) * w2
        acc2 += tl.where(init_clear2, init2, 0.0) * w3
        acc2 += bias
        acc2 = _fast_silu(acc2)

        tl.store(out_ptr + base0, acc0, mask=d_mask)
        tl.store(out_ptr + base1, acc1, mask=d_mask)
        tl.store(out_ptr + base2, acc2, mask=d_mask)
    else:
        tail_pid = pid - prefix_programs
        pid_d = tail_pid % num_d_blocks
        b = tail_pid // num_d_blocks

        d = pid_d * block_d + tl.arange(0, block_d)
        d_mask = d < dim

        t0 = tail_start
        t1 = t0 + 1
        t2 = t0 + 2
        t3 = t0 + 3
        t4 = t0 + 4

        bos_base = b * seqlen
        bos_tm2 = tl.load(bos_ptr + bos_base + t0 - 2)
        bos_tm1 = tl.load(bos_ptr + bos_base + t0 - 1)
        bos_t0 = tl.load(bos_ptr + bos_base + t0)
        bos_t1 = tl.load(bos_ptr + bos_base + t1)
        bos_t2 = tl.load(bos_ptr + bos_base + t2)
        bos_t3 = tl.load(bos_ptr + bos_base + t3)
        bos_t4 = tl.load(bos_ptr + bos_base + t4)

        token0 = b * seqlen + t0
        base0 = token0 * dim + d
        base1 = base0 + dim
        base2 = base1 + dim
        base3 = base2 + dim
        base4 = base3 + dim

        x_tm3 = tl.load(x_ptr + base0 - 3 * dim, mask=d_mask, other=0.0).to(tl.float32)
        x_tm2 = tl.load(x_ptr + base0 - 2 * dim, mask=d_mask, other=0.0).to(tl.float32)
        x_tm1 = tl.load(x_ptr + base0 - dim, mask=d_mask, other=0.0).to(tl.float32)
        x_t0 = tl.load(x_ptr + base0, mask=d_mask, other=0.0).to(tl.float32)
        x_t1 = tl.load(x_ptr + base1, mask=d_mask, other=0.0).to(tl.float32)
        x_t2 = tl.load(x_ptr + base2, mask=d_mask, other=0.0).to(tl.float32)
        x_t3 = tl.load(x_ptr + base3, mask=d_mask, other=0.0).to(tl.float32)
        x_t4 = tl.load(x_ptr + base4, mask=d_mask, other=0.0).to(tl.float32)

        w0 = tl.load(weight_t_ptr + 3 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
        w1 = tl.load(weight_t_ptr + 2 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
        w2 = tl.load(weight_t_ptr + dim + d, mask=d_mask, other=0.0).to(tl.float32)
        w3 = tl.load(weight_t_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

        valid0_1 = ~bos_t0
        valid0_2 = valid0_1 & (~bos_tm1)
        valid0_3 = valid0_2 & (~bos_tm2)
        acc = x_t0 * w0 + bias
        acc += tl.where(valid0_1, x_tm1, 0.0) * w1
        acc += tl.where(valid0_2, x_tm2, 0.0) * w2
        acc += tl.where(valid0_3, x_tm3, 0.0) * w3
        acc = _fast_silu(acc)
        tl.store(out_ptr + base0, acc, mask=d_mask)

        valid1_1 = ~bos_t1
        valid1_2 = valid1_1 & (~bos_t0)
        valid1_3 = valid1_2 & (~bos_tm1)
        acc = x_t1 * w0 + bias
        acc += tl.where(valid1_1, x_t0, 0.0) * w1
        acc += tl.where(valid1_2, x_tm1, 0.0) * w2
        acc += tl.where(valid1_3, x_tm2, 0.0) * w3
        acc = _fast_silu(acc)
        tl.store(out_ptr + base1, acc, mask=d_mask)

        final_base = b * dim * 3 + d
        valid2_1 = ~bos_t2
        valid2_2 = valid2_1 & (~bos_t1)
        valid2_3 = valid2_2 & (~bos_t0)
        acc = x_t2 * w0 + bias
        acc += tl.where(valid2_1, x_t1, 0.0) * w1
        acc += tl.where(valid2_2, x_t0, 0.0) * w2
        acc += tl.where(valid2_3, x_tm1, 0.0) * w3
        acc = _fast_silu(acc)
        tl.store(out_ptr + base2, acc, mask=d_mask)

        state0 = tl.where((~bos_t3) & (~bos_t4), x_t2, 0.0)
        state1 = tl.where(~bos_t4, x_t3, 0.0)
        tl.store(final_ptr + final_base, state0, mask=d_mask, cache_modifier=".cg")
        tl.store(
            final_ptr + final_base + dim,
            state1,
            mask=d_mask,
            cache_modifier=".cg",
        )
        tl.store(
            final_ptr + final_base + 2 * dim,
            x_t4,
            mask=d_mask,
            cache_modifier=".cg",
        )

        valid3_1 = ~bos_t3
        valid3_2 = valid3_1 & (~bos_t2)
        valid3_3 = valid3_2 & (~bos_t1)
        acc = x_t3 * w0 + bias
        acc += tl.where(valid3_1, x_t2, 0.0) * w1
        acc += tl.where(valid3_2, x_t1, 0.0) * w2
        acc += tl.where(valid3_3, x_t0, 0.0) * w3
        acc = _fast_silu(acc)
        tl.store(out_ptr + base3, acc, mask=d_mask)

        valid4_1 = ~bos_t4
        valid4_2 = valid4_1 & (~bos_t3)
        valid4_3 = valid4_2 & (~bos_t2)
        acc = x_t4 * w0 + bias
        acc += tl.where(valid4_1, x_t3, 0.0) * w1
        acc += tl.where(valid4_2, x_t2, 0.0) * w2
        acc += tl.where(valid4_3, x_t1, 0.0) * w3
        acc = _fast_silu(acc)
        tl.store(out_ptr + base4, acc, mask=d_mask)


@triton.jit
def _final_states_kernel(
    x_ptr,
    init_ptr,
    bos_ptr,
    final_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    width: tl.constexpr,
    has_init: tl.constexpr,
    has_bos: tl.constexpr,
    block_d: tl.constexpr,
):
    pid_d = tl.program_id(0)
    b = tl.program_id(1)
    s = tl.program_id(2)
    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim

    source_t = seqlen - ((width - 1) - s)
    x_valid = source_t >= 0
    value = tl.load(
        x_ptr + (b * seqlen + source_t) * dim + d,
        mask=d_mask & x_valid,
        other=0.0,
    )

    if has_bos:
        bos_base = b * seqlen
        tail_clear = True
        if width >= 2:
            pos = source_t + 1
            check = x_valid & (pos < seqlen)
            bos = tl.load(bos_ptr + bos_base + pos, mask=check, other=0)
            tail_clear = tail_clear & ((~check) | (~bos))
        if width >= 3:
            pos = source_t + 2
            check = x_valid & (pos < seqlen)
            bos = tl.load(bos_ptr + bos_base + pos, mask=check, other=0)
            tail_clear = tail_clear & ((~check) | (~bos))
        if width >= 4:
            pos = source_t + 3
            check = x_valid & (pos < seqlen)
            bos = tl.load(bos_ptr + bos_base + pos, mask=check, other=0)
            tail_clear = tail_clear & ((~check) | (~bos))
        value = tl.where(tail_clear, value, 0.0)

    if has_init:
        init_idx = s + seqlen
        init_valid = source_t < 0
        init_valid = init_valid & (init_idx < (width - 1))
        if has_bos:
            prefix_clear = True
            if width >= 2:
                check = seqlen >= 1
                bos = tl.load(bos_ptr + b * seqlen, mask=check, other=0)
                prefix_clear = prefix_clear & ((~check) | (~bos))
            if width >= 3:
                check = seqlen >= 2
                bos = tl.load(bos_ptr + b * seqlen + 1, mask=check, other=0)
                prefix_clear = prefix_clear & ((~check) | (~bos))
            if width >= 4:
                check = seqlen >= 3
                bos = tl.load(bos_ptr + b * seqlen + 2, mask=check, other=0)
                prefix_clear = prefix_clear & ((~check) | (~bos))
            init_valid = init_valid & prefix_clear
        init_val = tl.load(
            init_ptr + b * dim * (width - 1) + d * (width - 1) + init_idx,
            mask=d_mask & init_valid,
            other=0.0,
        )
        value = tl.where(x_valid, value, init_val)

    tl.store(final_ptr + b * dim * (width - 1) + s * dim + d, value, mask=d_mask)


def kernel_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    bos_mask: torch.Tensor | None = None,
    activation: str | None = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if activation not in (None, "silu"):
        return reference_kernel_fn(x, weight, bias, initial_states, bos_mask, activation, deterministic)
    if not x.is_cuda:
        return reference_kernel_fn(x, weight, bias, initial_states, bos_mask, activation, deterministic)
    if x.ndim != 3 or weight.ndim != 2:
        return reference_kernel_fn(x, weight, bias, initial_states, bos_mask, activation, deterministic)
    if x.dtype not in (torch.bfloat16, torch.float32):
        return reference_kernel_fn(x, weight, bias, initial_states, bos_mask, activation, deterministic)

    batch, seqlen, dim = x.shape
    width, weight_dim = weight.shape
    if weight_dim != dim or width < 2 or width > 4:
        return reference_kernel_fn(x, weight, bias, initial_states, bos_mask, activation, deterministic)
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != dim):
        return reference_kernel_fn(x, weight, bias, initial_states, bos_mask, activation, deterministic)
    if initial_states is not None and initial_states.shape != (batch, dim, width - 1):
        return reference_kernel_fn(x, weight, bias, initial_states, bos_mask, activation, deterministic)
    if bos_mask is not None and bos_mask.shape != (batch, seqlen):
        return reference_kernel_fn(x, weight, bias, initial_states, bos_mask, activation, deterministic)

    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()
    if initial_states is not None and not initial_states.is_contiguous():
        initial_states = initial_states.contiguous()
    if bos_mask is not None and not bos_mask.is_contiguous():
        bos_mask = bos_mask.contiguous()
    weight_t = weight

    out = torch.empty_like(x)
    final_states = torch.empty_strided(
        (batch, dim, width - 1),
        (dim * (width - 1), 1, dim),
        device=x.device,
        dtype=x.dtype,
    )

    block_d = 2048
    num_d_blocks = triton.cdiv(dim, block_d)
    dummy = x

    prefix_rows = min(width - 1, seqlen)
    main_rows = seqlen - prefix_rows
    use_prefix_t3 = False
    combine_prefix_t5_tail = False
    if prefix_rows > 0:
        use_prefix_t3 = (
            prefix_rows == 3
            and width == 4
            and bias is not None
            and initial_states is not None
            and bos_mask is not None
            and activation == "silu"
        )
        if use_prefix_t3:
            prefix_block_d = 128
            prefix_num_d_blocks = triton.cdiv(dim, prefix_block_d)
            main_full_tile_rows_for_prefix = main_rows // 8
            main_tail_rows_for_prefix = main_rows - main_full_tile_rows_for_prefix * 8
            combine_prefix_t5_tail = main_rows > 0 and main_tail_rows_for_prefix == 5
            if not combine_prefix_t5_tail:
                prefix_grid = (prefix_num_d_blocks * batch,)
                _conv1d_prefix_t3_kernel[prefix_grid](
                    x,
                    weight_t,
                    bias,
                    initial_states,
                    bos_mask,
                    out,
                    seqlen,
                    dim,
                    prefix_num_d_blocks,
                    prefix_block_d,
                    num_warps=2,
                    num_stages=1,
                )
        else:
            prefix_grid = (num_d_blocks * batch * prefix_rows,)
            _conv1d_out_kernel[prefix_grid](
                x,
                weight_t,
                bias if bias is not None else dummy,
                initial_states if initial_states is not None else dummy,
                bos_mask if bos_mask is not None else dummy,
                out,
                prefix_rows,
                0,
                seqlen,
                dim,
                width,
                bias is not None,
                initial_states is not None,
                bos_mask is not None,
                activation == "silu",
                num_d_blocks,
                block_d,
                num_warps=8,
            )

    final_states_done = False
    if main_rows > 0:
        if width == 4 and bias is not None and bos_mask is not None and activation == "silu":
            main_block_d = 256
            main_num_d_blocks = triton.cdiv(dim, main_block_d)
            main_full_tile_rows = main_rows // 8
            main_tail_rows = main_rows - main_full_tile_rows * 8
            if main_full_tile_rows > 0:
                main_grid = (main_num_d_blocks * batch * main_full_tile_rows,)
                _conv1d_main_t8_kernel[main_grid](
                    x,
                    weight_t,
                    bias,
                    bos_mask,
                    out,
                    main_full_tile_rows * 8,
                    main_full_tile_rows,
                    prefix_rows,
                    seqlen,
                    dim,
                    main_num_d_blocks,
                    main_block_d,
                    True,
                    x.dtype == torch.bfloat16,
                    num_warps=2,
                    num_stages=1,
                    maxnreg=56,
                )
            if main_tail_rows == 5:
                tail_block_d = 128
                tail_num_d_blocks = triton.cdiv(dim, tail_block_d)
                tail_grid = (tail_num_d_blocks * batch,)
                if combine_prefix_t5_tail:
                    boundary_block_d = 64
                    boundary_num_d_blocks = triton.cdiv(dim, boundary_block_d)
                    boundary_programs = boundary_num_d_blocks * batch
                    boundary_grid = (boundary_programs * 2,)
                    _conv1d_prefix_t3_tail_t5_kernel[boundary_grid](
                        x,
                        weight_t,
                        bias,
                        initial_states,
                        bos_mask,
                        out,
                        final_states,
                        prefix_rows + main_full_tile_rows * 8,
                        seqlen,
                        dim,
                        boundary_num_d_blocks,
                        boundary_block_d,
                        boundary_programs,
                        num_warps=1,
                        num_stages=1,
                    )
                else:
                    _conv1d_main_t5_tail_kernel[tail_grid](
                        x,
                        weight_t,
                        bias,
                        bos_mask,
                        out,
                        final_states,
                        prefix_rows + main_full_tile_rows * 8,
                        seqlen,
                        dim,
                        tail_num_d_blocks,
                        tail_block_d,
                        num_warps=1,
                        num_stages=1,
                    )
                final_states_done = True
            elif main_tail_rows > 0:
                tail_grid = (main_num_d_blocks * batch,)
                _conv1d_main_t8_kernel[tail_grid](
                    x,
                    weight_t,
                    bias,
                    bos_mask,
                    out,
                    main_tail_rows,
                    1,
                    prefix_rows + main_full_tile_rows * 8,
                    seqlen,
                    dim,
                    main_num_d_blocks,
                    main_block_d,
                    False,
                    x.dtype == torch.bfloat16,
                    num_warps=2,
                    num_stages=1,
                    maxnreg=56,
                )
        else:
            main_grid = (num_d_blocks * batch * main_rows,)
            _conv1d_main_kernel[main_grid](
                x,
                weight_t,
                bias if bias is not None else dummy,
                bos_mask if bos_mask is not None else dummy,
                out,
                main_rows,
                prefix_rows,
                seqlen,
                dim,
                width,
                bias is not None,
                bos_mask is not None,
                activation == "silu",
                num_d_blocks,
                block_d,
                num_warps=2,
            )

    if not final_states_done:
        final_grid = (num_d_blocks, batch, width - 1)
        _final_states_kernel[final_grid](
            x,
            initial_states if initial_states is not None else dummy,
            bos_mask if bos_mask is not None else dummy,
            final_states,
            seqlen,
            dim,
            width,
            initial_states is not None,
            bos_mask is not None,
            block_d,
            num_warps=8,
        )

    return out, final_states
