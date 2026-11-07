"""Performance patches for FSDP2 internals.

``fast_reduce_scatter_copy_in`` replaces ``torch._chunk_cat`` as the gradient copy-in of FSDP2's reduce-scatter.
``_chunk_cat`` with a dtype change (bf16 grads -> fp32 reduce dtype) runs at 0.3-2 TB/s on B200 (0.25-0.65 ms per
layer on the Gekko presets); one Triton kernel over a (gradient, shard) work table runs at the copy bandwidth
(5-6 TB/s). Semantics are those of ``torch._chunk_cat(grads, 0, world_size, out=out.view(world_size, -1))``: each
gradient is zero-padded along dim 0 to a multiple of ``world_size``, split into ``world_size`` row blocks, and block
``c`` of every gradient is concatenated into row ``c`` of the output. The constant part of the work table is cached
per gradient-shape signature; the gradients themselves are kernel arguments, so nothing is staged through the host.
Disable with ``XLLM_FSDP_FAST_COPY_IN=0``.

``XLLM_FSDP_DIRECT_AG`` (opt-in) moves the all-gather side's copies off the compute stream: 1 runs FSDP2's split
copy-out on the all-gather stream right after the collective, 2 all-gathers each parameter directly into its output.
"""
import os
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl

_BLOCK = 4096
_LOG2_ITEMS = 9  # work table padded to 512 (gradient, shard) items
_MAX_GRADS = 32
_SENTINEL = 1 << 62
_tables: Dict[Tuple, Tuple[torch.Tensor, int]] = {}


@triton.jit
def _copy_in_kernel(meta_ptr, out_ptr, scale, NT: tl.constexpr, LOG2: tl.constexpr, BLOCK: tl.constexpr, SCALE: tl.constexpr, g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15, g16, g17, g18, g19, g20, g21, g22, g23, g24, g25, g26, g27, g28, g29, g30, g31):
    """meta rows: [gradient index, source element offset, destination element offset, valid elements, total
    elements, first block]; rows sorted by first block and padded with sentinels. Each program copies BLOCK elements
    of one row (int32 in-row offsets: the vectorized bf16 path needs them)."""
    pid = tl.program_id(0).to(tl.int64)
    lo = tl.zeros((), dtype=tl.int64)
    for s in tl.static_range(LOG2):
        mid = lo + (1 << (LOG2 - 1 - s))
        first = tl.load(meta_ptr + mid * 6 + 5)
        lo = tl.where(pid >= first, mid, lo)
    gi = tl.load(meta_ptr + lo * 6)
    src_off = tl.load(meta_ptr + lo * 6 + 1)
    dst_off = tl.load(meta_ptr + lo * 6 + 2)
    n_valid = tl.load(meta_ptr + lo * 6 + 3).to(tl.int32)
    n_total = tl.load(meta_ptr + lo * 6 + 4).to(tl.int32)
    first = tl.load(meta_ptr + lo * 6 + 5)
    src = g0
    if NT > 1 and gi == 1:
        src = g1
    if NT > 2 and gi == 2:
        src = g2
    if NT > 3 and gi == 3:
        src = g3
    if NT > 4 and gi == 4:
        src = g4
    if NT > 5 and gi == 5:
        src = g5
    if NT > 6 and gi == 6:
        src = g6
    if NT > 7 and gi == 7:
        src = g7
    if NT > 8 and gi == 8:
        src = g8
    if NT > 9 and gi == 9:
        src = g9
    if NT > 10 and gi == 10:
        src = g10
    if NT > 11 and gi == 11:
        src = g11
    if NT > 12 and gi == 12:
        src = g12
    if NT > 13 and gi == 13:
        src = g13
    if NT > 14 and gi == 14:
        src = g14
    if NT > 15 and gi == 15:
        src = g15
    if NT > 16 and gi == 16:
        src = g16
    if NT > 17 and gi == 17:
        src = g17
    if NT > 18 and gi == 18:
        src = g18
    if NT > 19 and gi == 19:
        src = g19
    if NT > 20 and gi == 20:
        src = g20
    if NT > 21 and gi == 21:
        src = g21
    if NT > 22 and gi == 22:
        src = g22
    if NT > 23 and gi == 23:
        src = g23
    if NT > 24 and gi == 24:
        src = g24
    if NT > 25 and gi == 25:
        src = g25
    if NT > 26 and gi == 26:
        src = g26
    if NT > 27 and gi == 27:
        src = g27
    if NT > 28 and gi == 28:
        src = g28
    if NT > 29 and gi == 29:
        src = g29
    if NT > 30 and gi == 30:
        src = g30
    if NT > 31 and gi == 31:
        src = g31
    offs = ((pid - first) * BLOCK).to(tl.int32) + tl.arange(0, BLOCK)
    v = tl.load(src + src_off + offs, mask=offs < n_valid, other=0.0)
    if SCALE:  # the gradient divide factor (a power of 2: exact), folded into the copy
        v = v.to(tl.float32) * scale
    tl.store(out_ptr + dst_off + offs, v.to(out_ptr.dtype.element_ty), mask=offs < n_total)


def _table(grads, world_size, out_cols, device):
    """Constant part of the work table for one gradient-shape signature: one (meta, blocks) entry per launch of
    _MAX_GRADS gradients (the kernel takes the gradients as pointer arguments; a parameter group of several layers
    has more than _MAX_GRADS of them)."""
    items = []
    col = 0
    for gi, g in enumerate(grads):
        rows = g.shape[0] if g.dim() else 1
        inner = g.numel() // rows if rows else 0
        rpc = -(-rows // world_size)
        sz = rpc * inner
        if sz == 0:
            continue
        assert sz < 2**31, sz
        for c in range(world_size):
            n_valid = max(0, min(rows, (c + 1) * rpc) - c * rpc) * inner
            items.append([gi, c * rpc * inner, c * out_cols + col, n_valid, sz, 0])
        col += sz
    assert col == out_cols, (col, out_cols)
    launches = []
    for lo in range(0, len(grads), _MAX_GRADS):
        chunk = [[it[0] - lo] + it[1:] for it in items if lo <= it[0] < lo + _MAX_GRADS]
        assert len(chunk) <= (1 << _LOG2_ITEMS), len(chunk)
        blk = 0
        for it in chunk:
            it[5] = blk
            blk += -(-it[4] // _BLOCK)
        meta = torch.full((1 << _LOG2_ITEMS, 6), _SENTINEL, dtype=torch.int64)
        if chunk:
            meta[: len(chunk)] = torch.tensor(chunk, dtype=torch.int64)
        launches.append((meta.to(device), blk))
    return launches


# gradient scale for the next copy-in (set by folded_gradient_divide_factors in the same foreach_reduce call)
_PENDING_SCALE = [None]


def folded_gradient_divide_factors(*args, **kwargs):
    """FSDP2's _get_gradient_divide_factors, except that a post-reduction division by a power of 2 (the sum-only
    reductions of XLLM_FSDP_SYMM: a separate div_ over the fp32 reduce-scatter output, 7.1B ~3 ms per step) is folded
    into the gradient copy-in (fast_reduce_scatter_copy_in): scaling by 2^-k before the sum is exact, so the
    gradients are unchanged. Disable with XLLM_FSDP_FOLD_DIV=0."""
    pre, post, rs_op, ar_op = _orig_divide_factors(*args, **kwargs)
    _PENDING_SCALE[0] = None
    if pre is None and post is not None and post > 1 and float(post).is_integer() and (int(post) & (int(post) - 1)) == 0:
        _PENDING_SCALE[0] = 1.0 / post
        return None, None, rs_op, ar_op
    return pre, post, rs_op, ar_op


def fast_reduce_scatter_copy_in(unsharded_grads: list, reduce_scatter_input: torch.Tensor, world_size: int) -> None:
    out = reduce_scatter_input.view(world_size, -1)
    scale, _PENDING_SCALE[0] = _PENDING_SCALE[0], None
    grads = [g.contiguous() for g in unsharded_grads]
    if not grads or any(g.dtype != grads[0].dtype for g in grads):
        torch._chunk_cat(grads, 0, world_size, out=out)
        if scale is not None:
            out.mul_(scale)
        return
    key = (world_size, out.shape[1], out.dtype, grads[0].dtype, tuple(g.shape for g in grads))
    tab = _tables.get(key)
    if tab is None:
        tab = _tables[key] = _table(grads, world_size, out.shape[1], out.device)
    for k, (meta, n_blocks) in enumerate(tab):
        if n_blocks == 0:
            continue
        chunk = grads[k * _MAX_GRADS:(k + 1) * _MAX_GRADS]
        args = chunk + [chunk[0]] * (_MAX_GRADS - len(chunk))
        _copy_in_kernel[(n_blocks,)](meta, out, 1.0 if scale is None else scale, len(chunk), _LOG2_ITEMS, _BLOCK,
                                     scale is not None, *args, num_warps=4)


_orig_divide_factors = None

_DIRECT = ["direct all-gather"]  # marker in AllGatherResult.all_gather_input_split_sizes
_orig_all_gather = None
_orig_copy_out = None


def _direct_ok(fsdp_params, all_gather_comm) -> bool:
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import DefaultAllGather
    from torch.distributed.fsdp._fully_shard._fsdp_param import ShardedState
    if type(all_gather_comm) is not DefaultAllGather:
        return False
    for p in fsdp_params:
        if (p.fsdp_placement.dim != 0 or p.offload_to_cpu or p.sharded_state != ShardedState.SHARDED
                or hasattr(p._sharded_local_tensor, "fsdp_pre_all_gather")):
            return False
    return True


def direct_foreach_all_gather(fsdp_params, group, async_op, all_gather_copy_in_stream, all_gather_stream, device,
                              all_gather_comm):
    """FSDP2's all-gather without the flat buffer: one all_gather_into_tensor per parameter (one NCCL group) straight
    into the parameter's all-gather output, which the unsharded parameter views. With dim-0 sharding a parameter's
    gathered padded shards are exactly that output's layout, so neither the copy of the shards into a flat input nor
    the split copy-out (``split_with_sizes_copy``, on the compute stream) runs; only the cast of the local shards to
    the parameter dtype does. The outputs are allocated on the calling (compute) stream, which the copy-in and
    all-gather streams then wait for."""
    if not _direct_ok(fsdp_params, all_gather_comm):
        return _orig_all_gather(fsdp_params, group, async_op, all_gather_copy_in_stream, all_gather_stream, device,
                                all_gather_comm)
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import AllGatherResult
    world_size = group.size()
    current = torch.cuda.current_stream(device)
    shards = [p._sharded_param_data for p in fsdp_params]
    dtypes = [p.param_dtype or s.dtype for p, s in zip(fsdp_params, shards)]
    for p, s, dt in zip(fsdp_params, shards, dtypes):
        p.init_all_gather_outputs([s.numel()], [dt], world_size, device)
        p.alloc_all_gather_outputs()
    all_gather_copy_in_stream.wait_stream(current)
    with torch.cuda.stream(all_gather_copy_in_stream):
        inputs = list(shards)
        cast = [i for i, (s, dt) in enumerate(zip(shards, dtypes)) if s.dtype != dt]
        flat = None
        if cast:
            numels = [shards[i].numel() for i in cast]
            flat = torch.empty(sum(numels), dtype=dtypes[cast[0]], device=device)
            outs = torch.split(flat, numels)
            if all(dtypes[i] == dtypes[cast[0]] for i in cast):
                with torch.no_grad():
                    torch._foreach_copy_(list(outs), [shards[i] for i in cast])
                for i, o in zip(cast, outs):
                    inputs[i] = o
            else:
                flat = None
                for i in cast:
                    inputs[i] = shards[i].to(dtypes[i])
    all_gather_stream.wait_stream(all_gather_copy_in_stream)
    with torch.cuda.stream(all_gather_stream):
        with dist._coalescing_manager(group, device, async_ops=async_op) as cm:
            for p, inp in zip(fsdp_params, inputs):
                dist.all_gather_into_tensor(p.all_gather_outputs[0], inp, group=group)
        event = all_gather_stream.record_event()
    # all_gather_output keeps the cast inputs alive until FSDP drops the result
    return AllGatherResult((flat, inputs), event, cm if async_op else None, [[d] for d in dtypes],
                           [[s.numel()] for s in shards], _DIRECT)


def early_foreach_all_gather(fsdp_params, group, async_op, all_gather_copy_in_stream, all_gather_stream, device,
                             all_gather_comm):
    """FSDP2's flat all-gather with its split copy-out moved from the compute stream (where FSDP2 runs it when the
    parameters are needed) onto the all-gather stream right after the collective, so it overlaps the compute of the
    previous module instead of delaying this one. The parameters' all-gather outputs are allocated on the calling
    (compute) stream, which the all-gather stream then waits for."""
    if not _direct_ok(fsdp_params, all_gather_comm):
        return _orig_all_gather(fsdp_params, group, async_op, all_gather_copy_in_stream, all_gather_stream, device,
                                all_gather_comm)
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import AllGatherResult
    current = torch.cuda.current_stream(device)
    res = _orig_all_gather(fsdp_params, group, async_op, all_gather_copy_in_stream, all_gather_stream, device,
                           all_gather_comm)
    world_size = group.size()
    for p, numels, dts in zip(fsdp_params, res.param_all_gather_input_numels, res.param_all_gather_input_dtypes):
        p.init_all_gather_outputs(numels, dts, world_size, device)
        p.alloc_all_gather_outputs()
    all_gather_stream.wait_stream(current)
    with torch.cuda.stream(all_gather_stream):
        if res.all_gather_work is not None:
            res.all_gather_work.wait()
        out = [t.view(world_size, -1) for p in fsdp_params for t in p.all_gather_outputs]
        with torch.autograd._unsafe_preserve_version_counter(tuple(out)):
            torch.split_with_sizes_copy(res.all_gather_output.view(world_size, -1), res.all_gather_input_split_sizes,
                                        dim=1, out=out)
        event = all_gather_stream.record_event()
    return AllGatherResult(res.all_gather_output, event, None, res.param_all_gather_input_dtypes,
                           res.param_all_gather_input_numels, _DIRECT)


@torch.no_grad()
def direct_foreach_all_gather_copy_out(all_gather_result, fsdp_params, group) -> None:
    if all_gather_result.all_gather_input_split_sizes is not _DIRECT:
        return _orig_copy_out(all_gather_result, fsdp_params, group)
    if all_gather_result.all_gather_event is not None:
        torch.cuda.current_stream().wait_event(all_gather_result.all_gather_event)
    if all_gather_result.all_gather_work is not None:
        all_gather_result.all_gather_work.wait()


def _flat_grad_buffers(local_grads):
    """FSDP2 hands out the sharded gradients of a parameter group as views of one flat reduce-scatter output (plus
    zero padding between uneven shards). For a norm or a scaling the whole buffers are as good as the hundreds of
    views and cost a few launches instead of a foreach op over every tensor. Returns the flat buffers when every
    gradient is a contiguous, non-overlapping view of such a buffer (and the views cover it except for the padding,
    which the reduce-scatter zero-fills), else None."""
    by_storage = {}
    for g in local_grads:
        if not g.is_contiguous() or g.numel() == 0:
            return None
        st = g.untyped_storage()
        by_storage.setdefault((st.data_ptr(), g.dtype), (st, []))[1].append((g.storage_offset(), g.numel()))
    flats = []
    for st, views in by_storage.values():
        views.sort()
        end = 0
        for off, n in views:
            if off < end:
                return None
            end = off + n
        numel = st.nbytes() // local_grads[0].element_size()
        if end > numel or numel - end > numel // 8:  # more than a shard's worth of trailing bytes: not a grad buffer
            return None
        flats.append(torch.empty(0, dtype=local_grads[0].dtype, device=local_grads[0].device).set_(st, 0, (numel,)))
    return flats



def install_fsdp2_patches() -> None:
    import torch.distributed.fsdp._fully_shard._fsdp_collectives as col
    if os.environ.get("XLLM_FSDP_FAST_COPY_IN", "1") != "0" and getattr(col, "foreach_reduce_scatter_copy_in", None) is not None:
        col.foreach_reduce_scatter_copy_in = fast_reduce_scatter_copy_in
        global _orig_divide_factors
        if (os.environ.get("XLLM_FSDP_FOLD_DIV", "1") != "0" and _orig_divide_factors is None
                and getattr(col, "_get_gradient_divide_factors", None) is not None):
            _orig_divide_factors = col._get_gradient_divide_factors
            col._get_gradient_divide_factors = folded_gradient_divide_factors
    # opt-in: XLLM_FSDP_DIRECT_AG=1 moves the all-gather copy-out onto the all-gather stream (H100 8 GPUs: 7.1B
    # 797 -> 793 ms, 4B unchanged); =2 gathers per parameter without the flat buffer (4B 506 -> 514 ms: the grouped
    # NCCL all-gathers run longer and slow the overlapped compute)
    if os.environ.get("XLLM_FSDP_DIRECT_AG", "0") != "0":
        import torch.distributed.fsdp._fully_shard._fsdp_param_group as pg
        global _orig_all_gather, _orig_copy_out
        if _orig_all_gather is None and hasattr(pg, "foreach_all_gather") and hasattr(pg, "foreach_all_gather_copy_out"):
            _orig_all_gather, _orig_copy_out = pg.foreach_all_gather, pg.foreach_all_gather_copy_out
            pg.foreach_all_gather = (direct_foreach_all_gather if os.environ.get("XLLM_FSDP_DIRECT_AG") == "2"
                                     else early_foreach_all_gather)
            pg.foreach_all_gather_copy_out = direct_foreach_all_gather_copy_out


def apply_fsdp2_module_options(model: torch.nn.Module) -> None:
    """Per-module FSDP2 settings from the environment, applied after wrapping.

    ``XLLM_FSDP_SYMM=1`` (opt-in) stages the all-gathers and reduce-scatters in NCCL symmetric memory
    (``set_symm_mem_for_comm``, with sum-only reductions, which NCCL's symmetric reduce-scatter requires; FSDP2 then
    divides separately). On one NVLink node NCCL runs its symmetric reduce-scatter kernel and its symmetric all-gather
    kernel, or the copy-engine all-gather (no SMs) with ``NCCL_CTA_POLICY=2`` in the launch environment; both take
    fewer SMs from the overlapped compute. H100 8 GPUs, per-rank batches, NCCL_CTA_POLICY=2, 2 reps: 4B 528.5 ->
    501.5 ms, 0.9B 591 -> 578.5, 7.1B 813 -> 771, losses equal to 4 decimals. Not combined with
    ``XLLM_FSDP_DIRECT_AG=2`` (its per-parameter all-gathers do not use the FSDP2 staging buffers)."""
    from torch.distributed.fsdp import FSDPModule
    if os.environ.get("XLLM_FSDP_SYMM", "0") != "0" and os.environ.get("XLLM_FSDP_DIRECT_AG") != "2":
        for m in model.modules():
            if isinstance(m, FSDPModule) and hasattr(m, "set_symm_mem_for_comm"):
                m.set_force_sum_reduction_for_comms(True)
                m.set_symm_mem_for_comm()

