"""Adaptive working memory (AWM) forward with CuTe DSL kernels for Blackwell (SM100).

The memory update of a chunk is linear in the memory state M (S x V):

    U = kk (v - qk M) = kk v - (kk qk) M,      M' = M + r * (U - M)

with kk = exp(k - lse)^T the softmax of the chunk's keys over time. So the forward runs in three phases:

  1. per update chunk j (parallel):  lse_j = logsumexp_t k_t,  A_j = kk_j qk_j (S x S),  B_j = kk_j v_j (S x V)
  2. per (batch, kv head) (sequential over chunks, tiny):  M_{c+1} = (1 - r) M_c + r (B_c - A_c M_c)
  3. per chunk c (parallel):  out_c = q_c M_c, rows outside the memory's segment zeroed

Phases 1 and 3 read every input once and write the output once; the large reductions over the chunk do not
depend on the recurrence. This file holds the kernels; see adaptive_working_memory_fwd for the host side.

Kernels (one launch each, in stream order):
  AwmKeyLsePart      partial log-sum-exp of the keys over time (R row splits per chunk)
  AwmChunkGemm       [A | B] by tcgen05; transform warps turn the raw key tile into P = exp(k - lse), split-K
                     partials added with TMA reduce-add
  AwmAToBf16 +       recurrence on warp MMAs over a bf16 copy of A (V < 1024), or
  AwmRecurrence /    on tcgen05 tf32 with the state as M^T in shared memory (V >= 1024)
  AwmRecurrenceUmma
  AwmOutGemm         persistent warp-specialized tcgen05 GEMM, memory tile resident in shared memory
Masks (key rows, memory resets, output rows) are derived from segment_idx inside the kernels.

bf16, L = 16384, chunk 4096, with segment_idx, B200 (GPU time per call vs the PyTorch reference):
  gekko-0.9B (B2 H8 K4 S256 V256)    ~125us  (1376us)
  gekko-4B   (B1 H32 K8 S128 V128)    ~91us   (729us)
  gekko-7.1B (B1 H4 K4 S256 V2048)   ~207us  (1259us)
"""

import math
import os
import weakref
from functools import lru_cache
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32, Int32, Int64, TFloat32, Uint8, Uint32, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass.cute.nvgpu import cpasync, tcgen05


class AwmOutGemm:
    """Phase 3: out[b, l, h, :] = q[b, l, h, :] @ M_c(b, h // n_rep) for the chunk c of row l, zero on masked rows.

    q is read as [B, L, H, S] and out written as [B, L, H, V] (the layouts of the fused block, no transposes);
    the memory states come transposed and stacked as MT[(b * K + k) * nc + c, v, s].

    Persistent warp-specialized kernel: warps 0-3 epilogue (TMEM -> registers -> mask -> bf16 -> smem -> TMA
    store), warp 4 issues the tcgen05 MMAs, warp 5 the TMA loads. A tile is 128 rows x VT columns with the
    whole S as K. Tiles are ordered (vt, b, k, c, head of the group, row block) and every CTA takes a contiguous
    range, so the memory tile (VT x S, resident in shared memory) is reloaded only when the group changes.
    """

    def __init__(self, H: int, K: int, S: int, V: int, C: int, has_prev: bool, has_seg: bool, VT: int = 256,
                 a_stages: int = 4, b_resident: bool = True):
        """``b_resident``: the memory tile (VT x S) stays in shared memory for a group of tiles; otherwise it is
        streamed in K blocks beside the q tiles (for S too large to keep it resident)."""
        self.H, self.K, self.S, self.V, self.C = H, K, S, V, C
        self.has_prev, self.has_seg, self.b_resident = has_prev, has_seg, b_resident
        self.n_rep = H // K
        self.TM = 128
        self.VT = min(V, VT)
        self.KB = 64  # K block of one A stage (128 B of bf16: one swizzle atom)
        self.a_stages = a_stages
        self.b_stages = 1 if b_resident else a_stages
        self.acc_stages = 2
        self.epi_stages = 2  # per epilogue group
        self.epi_groups = 2  # two groups of 4 warps take alternate 32-column sub-tiles of a tile
        self.mma_tiler = (self.TM, self.VT, self.KB)
        self.mma_tiler_b = (self.TM, self.VT, self.S)
        self.num_epi_warps = 4 * self.epi_groups
        self.mma_warp = self.num_epi_warps
        self.tma_warp = self.num_epi_warps + 1
        self.threads = 32 * (self.num_epi_warps + 2)
        assert H % K == 0 and V % self.VT == 0 and C % self.TM == 0 and S % self.KB == 0
        assert self.VT % 16 == 0 and self.VT <= 256
        self.n_vt = V // self.VT
        self.row_tiles = C // self.TM
        self.k_tiles = S // self.KB

    @cute.jit
    def __call__(self, tQ: cute.Tensor, tMT: cute.Tensor, tOut: cute.Tensor, tSeg: cute.Tensor, tPrevCount: cute.Tensor,
                 tDead: cute.Tensor, tOutMask: cute.Tensor, B: Int32, L: Int32, seg_off: Int32, seg_stride: Int32,
                 num_ctas: Int32, stream: cuda.CUstream):
        H, K, S, V = self.H, self.K, self.S, self.V
        nc = L // self.C
        pQ, pMT, pOut = tQ.iterator, tMT.iterator, tOut.iterator
        # q [B, L, H, S] as (M = L, K = S, H, B)
        mQ = cute.make_tensor(pQ, cute.make_layout((L, S, H, B), stride=(H * S, 1, S, L * H * S)))
        # MT [(b, k, c), V, S] as (N = V, K = S, (c, k, b))
        mMT = cute.make_tensor(pMT, cute.make_layout((V, S, B * K * nc), stride=(S, 1, V * S)))
        # out [B, L, H, V] as (M = L, N = V, H, B)
        mOut = cute.make_tensor(pOut, cute.make_layout((L, V, H, B), stride=(H * V, 1, V, L * H * V)))
        mOutMask = cute.make_tensor(tOutMask.iterator, cute.make_layout((L, B), stride=(1, L)))
        mDead = cute.make_tensor(tDead.iterator, cute.make_layout((nc, B), stride=(1, nc)))

        op = tcgen05.MmaF16BF16Op(BFloat16, Float32, (self.TM, self.VT, 16), tcgen05.CtaGroup.ONE,
                                  tcgen05.OperandSource.SMEM, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
        tiled_mma = cute.make_tiled_mma(op)
        a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, BFloat16, self.a_stages)
        b_tiler = self.mma_tiler_b if const_expr(self.b_resident) else self.mma_tiler
        b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, b_tiler, BFloat16, self.b_stages)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_a, mQ_tma = cute.nvgpu.make_tiled_tma_atom_A(
            tma_op, mQ, cute.slice_(a_smem_layout, (None, None, None, 0)), self.mma_tiler, tiled_mma)
        tma_b, mMT_tma = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op, mMT, cute.slice_(b_smem_layout, (None, None, None, 0)), b_tiler, tiled_mma)

        c_layout_kind = utils.LayoutEnum.ROW_MAJOR
        epi_tile = sm100_utils.compute_epilogue_tile_shape((self.TM, self.VT, self.KB), False, c_layout_kind, BFloat16)
        epi_smem_layout = sm100_utils.make_smem_layout_epi(BFloat16, c_layout_kind, epi_tile, self.epi_stages * self.epi_groups)
        tma_c, mOut_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mOut, cute.slice_(epi_smem_layout, (None, None, 0)), epi_tile)

        num_tiles = self.n_vt * B * K * nc * self.n_rep * self.row_tiles
        self.kernel(tiled_mma, tma_a, mQ_tma, tma_b, mMT_tma, tma_c, mOut_tma, tSeg.iterator, tPrevCount.iterator,
                    mDead, mOutMask, a_smem_layout, b_smem_layout, epi_smem_layout, epi_tile, c_layout_kind,
                    B, nc, num_tiles, seg_off, seg_stride).launch(grid=[num_ctas, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, tma_a: cute.CopyAtom, mQ: cute.Tensor, tma_b: cute.CopyAtom,
               mMT: cute.Tensor, tma_c: cute.CopyAtom, mOut: cute.Tensor, pSeg: cute.Pointer, pPrevCount: cute.Pointer,
               mDead: cute.Tensor, mOutMask: cute.Tensor, a_smem_layout: cute.ComposedLayout,
               b_smem_layout: cute.ComposedLayout, epi_smem_layout: cute.ComposedLayout, epi_tile: cute.Tile,
               c_layout_kind: cutlass.Constexpr, B: Int32, nc: Int32, num_tiles: Int32, seg_off: Int32,
               seg_stride: Int32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nctas, _, _ = cute.arch.grid_dim()

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b)
            cpasync.prefetch_descriptor(tma_c)

        @cute.struct
        class SharedStorage:
            a_mbar: cute.struct.MemRange[cutlass.Int64, self.a_stages * 2]
            b_mbar: cute.struct.MemRange[cutlass.Int64, self.b_stages * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, self.acc_stages * 2]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        tmem_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * (self.num_epi_warps + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)

        a_bytes = cute.size_in_bytes(BFloat16, cute.select(a_smem_layout, mode=[0, 1, 2]))
        b_bytes = cute.size_in_bytes(BFloat16, cute.select(b_smem_layout, mode=[0, 1, 2]))
        thread_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        a_prod, a_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_mbar.data_ptr(), num_stages=self.a_stages,
            producer_group=thread_group, consumer_group=thread_group, tx_count=a_bytes,
        ).make_participants()
        b_prod, b_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_mbar.data_ptr(), num_stages=self.b_stages,
            producer_group=thread_group, consumer_group=thread_group, tx_count=b_bytes,
        ).make_participants()
        acc_prod, acc_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(), num_stages=self.acc_stages, producer_group=thread_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epi_warps),
        ).make_participants()

        sA = smem.allocate_tensor(BFloat16, a_smem_layout.outer, byte_alignment=128, swizzle=a_smem_layout.inner)
        sB = smem.allocate_tensor(BFloat16, b_smem_layout.outer, byte_alignment=128, swizzle=b_smem_layout.inner)
        sC = smem.allocate_tensor(BFloat16, epi_smem_layout.outer, byte_alignment=128, swizzle=epi_smem_layout.inner)

        # (bM, bK, RestM, RestK, H, B) / (bN, bK, RestN, RestK, NL) / (bM, bN, RestM, RestN, H, B)
        gA = cute.local_tile(mQ, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None, None))
        # (VT, S or KB, RestN, RestK, NL): the whole K of a memory tile when resident, K blocks when streamed
        gB = cute.local_tile(mMT, cute.slice_(self.mma_tiler_b if const_expr(self.b_resident) else self.mma_tiler, (0, None, None)),
                             (None, None, None))
        gC = cute.local_tile(mOut, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None, None))
        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stages))

        tAsA, tAgA = cpasync.tma_partition(tma_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3),
                                           cute.group_modes(tCgA, 0, 3))
        tBsB, tBgB = cpasync.tma_partition(tma_b, 0, cute.make_layout(1), cute.group_modes(sB, 0, 3),
                                           cute.group_modes(tCgB, 0, 3))
        gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None, None)], epi_tile)
        tCsC, tCgC_tma = cpasync.tma_partition(tma_c, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2),
                                               cute.group_modes(gC_epi, 0, 2))

        # contiguous tile range of this CTA
        start = bidx * num_tiles // nctas
        end = (bidx + 1) * num_tiles // nctas
        group_tiles = self.n_rep * self.row_tiles

        if warp == self.tma_warp:
            prev_group = Int32(-1)
            t = start
            while t < end:
                rt, hr, c, k, b, vt = self._decode(t, B, nc)
                group = t // group_tiles
                mem_idx = (b * self.K + k) * nc + c
                if const_expr(self.b_resident):
                    if group != prev_group:
                        hb = b_prod.acquire_and_advance()
                        cute.copy(tma_b, tBgB[(None, vt, 0, mem_idx)], tBsB[(None, hb.index)], tma_bar_ptr=hb.barrier)
                        prev_group = group
                row_block = c * self.row_tiles + rt
                h = k * self.n_rep + hr
                for kt in cutlass.range(self.k_tiles):
                    ha = a_prod.acquire_and_advance()
                    cute.copy(tma_a, tAgA[(None, row_block, kt, h, b)], tAsA[(None, ha.index)], tma_bar_ptr=ha.barrier)
                    if const_expr(not self.b_resident):
                        hb = b_prod.acquire_and_advance()
                        cute.copy(tma_b, tBgB[(None, vt, kt, mem_idx)], tBsB[(None, hb.index)], tma_bar_ptr=hb.barrier)
                t += 1
            a_prod.tail()
            b_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            t = start
            while t < end:
                group = t // group_tiles
                if const_expr(self.b_resident):
                    if t == start or t % group_tiles == 0:
                        b_cons.wait()
                acc_empty = acc_prod.acquire_and_advance()
                tCtAcc = tCtAcc_base[(None, None, None, acc_empty.index)]
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kt in cutlass.range(self.k_tiles):
                    ha = a_cons.wait_and_advance()
                    if const_expr(self.b_resident):
                        for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                            cute.gemm(tiled_mma, tCtAcc, tCrA[(None, None, kb, ha.index)],
                                      tCrB[(None, None, kt * cute.size(tCrA, mode=[2]) + kb, 0)], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    else:
                        hb = b_cons.wait_and_advance()
                        for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                            cute.gemm(tiled_mma, tCtAcc, tCrA[(None, None, kb, ha.index)], tCrB[(None, None, kb, hb.index)], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        hb.release()
                    ha.release()
                acc_empty.commit()
                # release the memory tile after the group's last tile (the MMAs above still read it)
                if const_expr(self.b_resident):
                    if t + 1 == end or (t + 1) // group_tiles != group:
                        b_cons.release()
                        b_cons.advance()
                t += 1
            acc_prod.tail()
        elif warp < self.mma_warp:
            tmem.allocate(512)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            grp = warp // 4
            etid = tidx % 128  # TMEM lane / row of the tile (warp % 4 selects the lanes)
            epi_bar = pipeline.NamedBarrier(barrier_id=3 + grp, num_threads=128)
            epi_pipe = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stages, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128))
            copy_atom_t2r = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE), Float32)
            t = start
            while t < end:
                rt, hr, c, k, b, vt = self._decode(t, B, nc)
                row_block = c * self.row_tiles + rt
                h = k * self.n_rep + hr
                # this thread's row of the tile (TMEM lane = row for M = 128) and its mask: the chunk's memory is
                # empty, or the row is not in the segment that the memory ends in (that of chunk c - 2)
                row = row_block * self.TM + etid
                # mask inputs loaded before waiting for the accumulator (their latency hides behind the wait)
                dead = mDead[c, b]
                seg_row = Int64(0)
                anchor = Int64(0)
                if const_expr(self.has_seg):
                    seg_row = cute.make_tensor(pSeg + (b * seg_stride + seg_off + row), cute.make_layout(1))[0]
                    if c >= 2 or const_expr(self.has_prev):
                        anchor = _seg_anchor(pSeg, pPrevCount, b, c - 2, seg_off, seg_stride, self.C)
                    else:
                        anchor = seg_row
                acc_full = acc_cons.wait_and_advance()
                masked = (dead != 0) | (seg_row != anchor)
                keep = cutlass.select_(masked, Float32(0.0), Float32(1.0))
                if hr == 0 and vt == 0 and k == 0 and grp == 0:
                    mOutMask[row, b] = cutlass.select_(masked, Uint8(1), Uint8(0))
                tCtAcc = tCtAcc_base[(None, None, None, acc_full.index)]
                tCtAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
                tCgC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, row_block, vt, h, b)], epi_tile)
                tCgC_tma_cur = tCgC_tma[(None, None, None, row_block, vt, h, b)]
                tiled_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[(None, None, 0, 0)])
                thr_t2r = tiled_t2r.get_slice(etid)
                tTR_tAcc = thr_t2r.partition_S(tCtAcc_epi)
                tTR_gC = thr_t2r.partition_D(tCgC_epi)
                tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0)].shape, Float32)
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                copy_atom_r2s = sm100_utils.get_smem_store_op(c_layout_kind, BFloat16, Float32, tiled_t2r)
                tiled_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_t2r)
                thr_r2s = tiled_r2s.get_slice(etid)
                tRS_sC = thr_r2s.partition_D(sC)
                tRS_rAcc = tiled_r2s.retile(tTR_rAcc)
                tRS_rC = cute.make_rmem_tensor(tRS_rAcc.shape, BFloat16)
                tCgC_grouped = cute.group_modes(tCgC_tma_cur, 1, cute.rank(tCgC_tma_cur))
                for sub2 in cutlass.range(cute.size(tTR_tAcc.shape, mode=[3]) // self.epi_groups):
                    sub = sub2 * self.epi_groups + grp
                    cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, sub)], tTR_rAcc)
                    buf = grp * self.epi_stages + sub2 % self.epi_stages
                    tRS_rC.store((tRS_rAcc.load() * keep).to(BFloat16))
                    cute.copy(tiled_r2s, tRS_rC, tRS_sC[(None, None, None, buf)])
                    cute.arch.fence_view_async_shared()
                    epi_bar.arrive_and_wait()
                    if warp % 4 == 0:
                        cute.copy(tma_c, tCsC[(None, buf)], tCgC_grouped[(None, sub)])
                        epi_pipe.producer_commit()
                        epi_pipe.producer_acquire()
                    epi_bar.arrive_and_wait()
                with cute.arch.elect_one():
                    acc_full.release()
                t += 1
            epi_pipe.producer_tail()
            tmem.relinquish_alloc_permit()
            cute.arch.barrier(barrier_id=5, number_of_threads=32 * self.num_epi_warps)
            tmem.free(tmem_ptr)

    @cute.jit
    def _decode(self, t: Int32, B: Int32, nc: Int32):
        """Tile index -> (row block in chunk, head in group, chunk, kv head, batch, v tile). Groups (one memory
        tile) are ordered (b, k, c, vt) with the v tile fastest: the CTAs of the v tiles of the same rows run
        side by side and read each q tile from L2."""
        rt = t % self.row_tiles
        t1 = t // self.row_tiles
        hr = t1 % self.n_rep
        t2 = t1 // self.n_rep
        vt = t2 % self.n_vt
        t3 = t2 // self.n_vt
        c = t3 % nc
        t4 = t3 // nc
        k = t4 % self.K
        b = t4 // self.K
        return rt, hr, c, k, b, vt


LOG2E = math.log2(math.e)
NEG_BIG = -1.0e30  # finite stand-in for -inf in running maxima (exp2 of differences stays 0, never NaN)


class AwmKeyLsePart:
    """Phase 1a: partial base-2 log-sum-exp of the keys over time. For update chunk j (output index
    j + j_off) and row split rs (C / R rows), part[b, k, j + j_off, rs, s] = (m, l) with
    log2 sum_t 2^(k[t, s] log2 e) = m + log2 l over the split's rows; rows outside the segment of their chunk's
    last row are excluded (segment ids read from seg[b * seg_stride + seg_off + t] when has_seg).
    The consumers combine the R partials. k is read as [B, L, K, S].

    One CTA per (b, k, j, 64 columns of s, row split), 256 threads = 8 column groups (8 columns, one 16-byte
    load each) x 32 row groups; online in fp32, rescaled once per 4 rows; row groups combined in shared memory."""

    def __init__(self, K: int, S: int, C: int, R: int, has_seg: bool):
        self.K, self.S, self.C, self.R, self.has_seg = K, S, C, R, has_seg
        self.SB = 64
        self.RG = 32
        self.UNROLL = 4
        assert S % self.SB == 0 and C % (R * self.RG * self.UNROLL) == 0

    @cute.jit
    def __call__(self, tK: cute.Tensor, tSeg: cute.Tensor, tPart: cute.Tensor, B: Int32, L: Int32, nu: Int32,
                 j_off: Int32, nu_tot: Int32, seg_off: Int32, seg_stride: Int32, stream: cuda.CUstream):
        K, S, R = self.K, self.S, self.R
        mK = cute.make_tensor(tK.iterator, cute.make_layout((S, L, K, B), stride=(1, K * S, S, L * K * S)))
        mPart = cute.make_tensor(tPart.iterator, cute.make_layout((2, S, R, nu_tot, K, B), stride=(
            1, 2, 2 * S, 2 * S * R, 2 * S * R * nu_tot, 2 * S * R * nu_tot * K)))
        grid = B * K * nu * (S // self.SB) * R
        self.kernel(mK, tSeg.iterator, mPart, nu, j_off, seg_off, seg_stride).launch(
            grid=[grid, 1, 1], block=[8 * self.RG, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mK: cute.Tensor, pSeg: cute.Pointer, mPart: cute.Tensor, nu: Int32, j_off: Int32,
               seg_off: Int32, seg_stride: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nsb = self.S // self.SB
        rs = bidx % self.R
        rest = bidx // self.R
        sb = rest % nsb
        rest = rest // nsb
        j = rest % nu
        rest = rest // nu
        k = rest % self.K
        b = rest // self.K
        sg = tidx % 8
        rg = tidx // 8
        s0 = sb * self.SB + sg * 8
        rows = self.C // self.R
        smem = utils.SmemAllocator()
        sKeep = smem.allocate_tensor(Uint8, cute.make_layout(rows), byte_alignment=16)
        if const_expr(self.has_seg):
            # keep flags of the CTA's rows (inside the segment of the chunk's last row), once
            seg0 = b * seg_stride + seg_off + j * self.C
            seg_last = cute.make_tensor(pSeg + (seg0 + self.C - 1), cute.make_layout(1))[0]
            for i in cutlass.range(tidx, rows, 8 * self.RG):
                seg_r = cute.make_tensor(pSeg + (seg0 + rs * rows + i), cute.make_layout(1))[0]
                sKeep[i] = cutlass.select_(seg_r != seg_last, Uint8(0), Uint8(1))
            cute.arch.sync_threads()
        m = [Float32(NEG_BIG) for _ in range(8)]
        acc = [Float32(0.0) for _ in range(8)]
        r_base = j * self.C + rs * rows + rg
        for it in cutlass.range(rows // (self.RG * self.UNROLL)):
            xs = []
            for u in cutlass.range_constexpr(self.UNROLL):
                r = r_base + (it * self.UNROLL + u) * self.RG
                v = cute.make_tensor((mK.iterator + mK.layout((s0, r, k, b))).align(16), cute.make_layout(8)).load().to(Float32)
                if const_expr(self.has_seg):
                    masked = sKeep[r - j * self.C - rs * rows] == 0
                    xs.append([cutlass.select_(masked, Float32(NEG_BIG), v[e] * LOG2E) for e in range(8)])
                else:
                    xs.append([v[e] * LOG2E for e in range(8)])
            for e in cutlass.range_constexpr(8):
                bmax = xs[0][e]
                for u in cutlass.range_constexpr(1, self.UNROLL):
                    bmax = cute.arch.fmax(bmax, xs[u][e])
                m_new = cute.arch.fmax(m[e], bmax)
                a = acc[e] * cute.math.exp2(m[e] - m_new, fastmath=True)
                for u in cutlass.range_constexpr(self.UNROLL):
                    a = a + cute.math.exp2(xs[u][e] - m_new, fastmath=True)
                acc[e] = a
                m[e] = m_new
        sM = smem.allocate_tensor(Float32, cute.make_layout((self.SB, self.RG), stride=(1, self.SB)), byte_alignment=16)
        sL = smem.allocate_tensor(Float32, cute.make_layout((self.SB, self.RG), stride=(1, self.SB)), byte_alignment=16)
        for e in cutlass.range_constexpr(8):
            sM[sg * 8 + e, rg] = m[e]
            sL[sg * 8 + e, rg] = acc[e]
        cute.arch.sync_threads()
        if tidx < self.SB:
            mm = sM[tidx, 0]
            for g in cutlass.range_constexpr(1, self.RG):
                mm = cute.arch.fmax(mm, sM[tidx, g])
            tot = Float32(0.0)
            for g in cutlass.range_constexpr(self.RG):
                tot = tot + sL[tidx, g] * cute.math.exp2(sM[tidx, g] - mm, fastmath=True)
            mPart[0, sb * self.SB + tidx, rs, j + j_off, k, b] = mm
            mPart[1, sb * self.SB + tidx, rs, j + j_off, k, b] = tot


def _u32_as_f32(u):
    return Float32(mlir_arith.bitcast(ir.F32Type.get(), Uint32(u).ir_value()))


@cute.jit
def _seg_anchor(pSeg: cute.Pointer, pPrevCount: cute.Pointer, b, x, seg_off, seg_stride, C: cutlass.Constexpr):
    """Segment id at the end of chunk x: the last row of chunk x for x >= 0, of the previous call's last chunk for
    x = -1 (stored in front of the current rows), the previous segment count for x = -2."""
    val = Int64(0)
    if x == -2:
        val = cute.make_tensor(pPrevCount + b, cute.make_layout(1))[0]
    else:
        val = cute.make_tensor(pSeg + (b * seg_stride + seg_off + x * C + C - 1), cute.make_layout(1))[0]
    return val


@cute.jit
def _combine_lse2(mPart: cute.Tensor, s, j, k, b, R: cutlass.Constexpr):
    """log2 sum over the R partials (m, l) of row s: max m + log2 sum l 2^(m - max m)."""
    mm = mPart[0, s, 0, j, k, b]
    for r in cutlass.range_constexpr(1, R):
        mm = cute.arch.fmax(mm, mPart[0, s, r, j, k, b])
    tot = Float32(0.0)
    for r in cutlass.range_constexpr(R):
        tot = tot + mPart[1, s, r, j, k, b] * cute.math.exp2(mPart[0, s, r, j, k, b] - mm, fastmath=True)
    return mm + cute.math.log2(tot, fastmath=True)


def chunk_gemm_tiling(S: int, V: int) -> Tuple[int, int, int]:
    """(NS, nsub, n_blocks): the N = S + V columns of [qk | v] in sub-tiles of NS columns (one MMA each),
    nsub sub-tiles per CTA (nsub * NS <= 512 TMEM columns, nsub dividing the number of sub-tiles so that every
    CTA loads the same number of bytes per stage). Fewest CTAs per output row block, then the widest MMA."""
    best = None
    for ns in (256, 128, 64):
        if S % ns or V % ns:
            continue
        total = (S + V) // ns
        nsub = max(d for d in range(1, 512 // ns + 1) if total % d == 0)
        cand = (total // nsub, -ns, nsub)
        if best is None or cand < best:
            best = cand
    nb, neg_ns, nsub = best
    return -neg_ns, nsub, nb


class AwmChunkGemm:
    """Phase 1b: AB[b, k, j, s, :] += sum_t P[s, t] [qk | v][t, :] over the rows t of a split of update chunk j,
    P[s, t] = exp(k[b, t, k, s] - lse[b, k, j, s]) (0 on masked rows). AB is fp32, zeroed by the caller.

    Both MMA operands are MN-major (k, qk and v are read as [B, L, K, *], the s / n columns contiguous).
    One CTA per (b, k, j, 128 rows of s, block of nsub sub-tiles of NS columns, split of the chunk):
      warp 5:    TMA loads of the raw k tile (128 s x KB rows) and the nsub [qk | v] sub-tiles of a stage
      warps 0-3: P = exp2(k * log2e - lse * log2e) of the stage into a second buffer (16-byte vectors at the
                 same swizzled addresses as the raw tile, the 8 lse values of a thread held in registers);
                 at the end, the epilogue (TMEM -> registers -> red.global.add.v4.f32 into AB)
      warp 4:    tcgen05 MMAs, one fp32 accumulator of NS TMEM columns per sub-tile
    """

    def __init__(self, K: int, S: int, V: int, C: int, splits: int, R: int, has_seg: bool, stages: int = 4):
        self.K, self.S, self.V, self.C, self.R, self.has_seg = K, S, V, C, R, has_seg
        self.NS, self.nsub, self.n_blocks = chunk_gemm_tiling(S, V)
        self.TM = 128
        self.KB = 32
        # rings: raw key tiles (cheap, deep: their TMA latency is on the transform warps' path), P tiles, and
        # [qk | v] tiles with the shared memory left
        stage_ab = 128 * self.KB * 2
        stage_b = self.nsub * self.NS * self.KB * 2
        budget = 220 * 1024 - C
        self.p_stages = 4
        self.b_stages = max(2, min(6, (budget - (4 + self.p_stages) * stage_ab) // stage_b))
        self.b_stages = max(min(self.b_stages, 4), min(6, (budget - (8 + self.p_stages) * stage_ab) // stage_b))
        self.k_stages = max(4, min(8, (budget - self.p_stages * stage_ab - self.b_stages * stage_b) // stage_ab))
        self.epi_stages = 2
        self.splits = splits
        self.m_blocks = S // self.TM
        self.chunk_k_tiles = C // self.KB  # split as evenly as possible among the splits
        self.qk_subs = S // self.NS
        self.NAB = S + V
        self.num_tp_warps = 4
        self.mma_warp = 4
        self.tma_warp = 5
        self.threads = 32 * 6
        assert S % self.TM == 0 and C % self.KB == 0 and splits <= self.chunk_k_tiles
        assert self.nsub * self.NS <= 512

    @cute.jit
    def __call__(self, tK: cute.Tensor, tQK: cute.Tensor, tV: cute.Tensor, tSeg: cute.Tensor, tPart: cute.Tensor,
                 tAB: cute.Tensor, B: Int32, L: Int32, nu: Int32, j_off: Int32, nu_tot: Int32, seg_off: Int32,
                 seg_stride: Int32, stream: cuda.CUstream):
        K, S, V = self.K, self.S, self.V
        # operands as (MN, K = time, kv head, batch), MN contiguous
        mK = cute.make_tensor(tK.iterator, cute.make_layout((S, L, K, B), stride=(1, K * S, S, L * K * S)))
        mQK = cute.make_tensor(tQK.iterator, cute.make_layout((S, L, K, B), stride=(1, K * S, S, L * K * S)))
        mV = cute.make_tensor(tV.iterator, cute.make_layout((V, L, K, B), stride=(1, K * V, V, L * K * V)))
        R = self.R
        mPart = cute.make_tensor(tPart.iterator, cute.make_layout((2, S, R, nu_tot, K, B), stride=(
            1, 2, 2 * S, 2 * S * R, 2 * S * R * nu_tot, 2 * S * R * nu_tot * K)))

        op = tcgen05.MmaF16BF16Op(BFloat16, Float32, (self.TM, self.NS, 16), tcgen05.CtaGroup.ONE,
                                  tcgen05.OperandSource.SMEM, tcgen05.OperandMajorMode.MN, tcgen05.OperandMajorMode.MN)
        tiled_mma = cute.make_tiled_mma(op)
        tiler = (self.TM, self.NS, self.KB)
        raw_layout = sm100_utils.make_smem_layout_a(tiled_mma, tiler, BFloat16, self.k_stages)
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, tiler, BFloat16, self.p_stages)
        b_layout = sm100_utils.make_smem_layout_b(tiled_mma, tiler, BFloat16, self.b_stages * self.nsub)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_k, mK_t = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mK, cute.slice_(raw_layout, (None, None, None, 0)), tiler, tiled_mma)
        b_one = cute.slice_(b_layout, (None, None, None, 0))
        tma_qk, mQK_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mQK, b_one, tiler, tiled_mma)
        tma_v, mV_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mV, b_one, tiler, tiled_mma)
        # epilogue: TMA reduce-add of 128 x 32 fp32 tiles into AB, staged in the (then idle) raw key buffer
        mAB_mn = cute.make_tensor(tAB.iterator, cute.make_layout((S, self.NAB, nu_tot, K, B), stride=(
            self.NAB, 1, S * self.NAB, nu_tot * S * self.NAB, K * nu_tot * S * self.NAB)))
        epi_tile = (self.TM, 32)
        epi_layout = sm100_utils.make_smem_layout_epi(Float32, utils.LayoutEnum.ROW_MAJOR, epi_tile, self.epi_stages)
        tma_red, mAB_t = cpasync.make_tiled_tma_atom(cpasync.CopyReduceBulkTensorTileS2GOp(), mAB_mn,
                                                     cute.slice_(epi_layout, (None, None, 0)), epi_tile)
        grid = B * K * nu * self.m_blocks * self.n_blocks * self.splits
        self.kernel(tiled_mma, tma_k, mK_t, tma_qk, mQK_t, tma_v, mV_t, tSeg.iterator, mPart, tma_red, mAB_t,
                    raw_layout, a_layout, b_layout, epi_layout, B, nu, j_off, seg_off, seg_stride).launch(grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream)

    def _p_offset(self, g, kk):
        """Element offset of P[8 g .. 8 g + 7, kk] in a stage of the MN-major A operand (128-byte swizzle)."""
        o = ((8 * g) % 64 + ((8 * g) // 64) * (64 * self.KB) + (kk % 16) * 64 + (kk // 16) * 1024) * 2
        return (o ^ (((o >> 7) & 7) << 4)) // 2

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, tma_k: cute.CopyAtom, mK: cute.Tensor, tma_qk: cute.CopyAtom,
               mQK: cute.Tensor, tma_v: cute.CopyAtom, mV: cute.Tensor, pSeg: cute.Pointer, mPart: cute.Tensor,
               tma_red: cute.CopyAtom, mAB: cute.Tensor, raw_layout: cute.ComposedLayout, a_layout: cute.ComposedLayout,
               b_layout: cute.ComposedLayout,
               epi_layout: cute.ComposedLayout, B: Int32, nu: Int32, j_off: Int32, seg_off: Int32, seg_stride: Int32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        # unit decode: (split, n block, m block, j, k, b)
        ks = bidx % self.splits
        rest = bidx // self.splits
        nb = rest % self.n_blocks
        rest = rest // self.n_blocks
        mb = rest % self.m_blocks
        rest = rest // self.m_blocks
        j = rest % nu
        rest = rest // nu
        k = rest % self.K
        b = rest // self.K
        kt_lo = ks * self.chunk_k_tiles // self.splits
        n_kt = (ks + 1) * self.chunk_k_tiles // self.splits - kt_lo
        row0 = j * self.C + kt_lo * self.KB  # first time row of this split
        kt0 = row0 // self.KB

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_qk)
            cpasync.prefetch_descriptor(tma_v)

        @cute.struct
        class SharedStorage:
            k_mbar: cute.struct.MemRange[cutlass.Int64, self.k_stages * 2]
            p_mbar: cute.struct.MemRange[cutlass.Int64, self.p_stages * 2]
            b_mbar: cute.struct.MemRange[cutlass.Int64, self.b_stages * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_tp_warps + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)

        a_bytes = cute.size_in_bytes(BFloat16, cute.select(a_layout, mode=[0, 1, 2]))
        b_bytes = cute.size_in_bytes(BFloat16, cute.select(b_layout, mode=[0, 1, 2])) * self.nsub
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        tp_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * self.num_tp_warps)
        # the transform warps release a raw stage with one arrive per warp (lane 0, after a warp sync)
        k_prod, k_cons = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.k_mbar.data_ptr(), num_stages=self.k_stages, producer_group=one,
            consumer_group=tp_group, tx_count=a_bytes, tidx=tidx % 32,
            enable_multicast_signaling=True).make_participants()
        p_prod, p_cons = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.p_mbar.data_ptr(), num_stages=self.p_stages, producer_group=tp_group,
            consumer_group=one).make_participants()
        b_prod, b_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_mbar.data_ptr(), num_stages=self.b_stages, producer_group=one,
            consumer_group=one, tx_count=b_bytes).make_participants()
        acc_prod, acc_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(), num_stages=1, producer_group=one,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_tp_warps)).make_participants()

        pRaw = smem.allocate_array(BFloat16, cute.cosize(raw_layout.outer), byte_alignment=1024)
        pP = smem.allocate_array(BFloat16, cute.cosize(a_layout.outer), byte_alignment=1024)
        sRaw = cute.make_tensor(cute.recast_ptr(pRaw, swizzle_=raw_layout.inner), raw_layout.outer)
        sP = cute.make_tensor(cute.recast_ptr(pP, swizzle_=a_layout.inner), a_layout.outer)
        sB = smem.allocate_tensor(BFloat16, b_layout.outer, byte_alignment=1024, swizzle=b_layout.inner)
        sKeep = smem.allocate_tensor(Uint8, cute.make_layout(self.C), byte_alignment=16)

        # (bM, bK, RestM, RestK, K, B) / (bN, bK, RestN, RestK, K, B)
        gK = cute.local_tile(mK, (self.TM, self.KB), (None, None, None, None))
        gQK = cute.local_tile(mQK, (self.NS, self.KB), (None, None, None, None))
        gV = cute.local_tile(mV, (self.NS, self.KB), (None, None, None, None))
        thr_mma = tiled_mma.get_slice(0)
        tCgK = thr_mma.partition_A(gK)
        tCgQK = thr_mma.partition_B(gQK)
        tCgV = thr_mma.partition_B(gV)
        tCrA = tiled_mma.make_fragment_A(sP)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C((self.TM, self.NS))
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.nsub))
        tKs, tKg = cpasync.tma_partition(tma_k, 0, cute.make_layout(1), cute.group_modes(sRaw, 0, 3), cute.group_modes(tCgK, 0, 3))
        tQKs, tQKg = cpasync.tma_partition(tma_qk, 0, cute.make_layout(1), cute.group_modes(sB, 0, 3), cute.group_modes(tCgQK, 0, 3))
        tVs, tVg = cpasync.tma_partition(tma_v, 0, cute.make_layout(1), cute.group_modes(sB, 0, 3), cute.group_modes(tCgV, 0, 3))

        if warp == self.tma_warp:
            for t in cutlass.range(n_kt):
                hk = k_prod.acquire_and_advance()
                cute.copy(tma_k, tKg[(None, mb, kt0 + t, k, b)], tKs[(None, hk.index)], tma_bar_ptr=hk.barrier)
                hb = b_prod.acquire_and_advance()
                for sub in cutlass.range_constexpr(self.nsub):
                    gsub = nb * self.nsub + sub
                    if gsub < self.qk_subs:
                        cute.copy(tma_qk, tQKg[(None, gsub, kt0 + t, k, b)], tQKs[(None, hb.index * self.nsub + sub)],
                                  tma_bar_ptr=hb.barrier)
                    else:
                        cute.copy(tma_v, tVg[(None, gsub - self.qk_subs, kt0 + t, k, b)], tVs[(None, hb.index * self.nsub + sub)],
                                  tma_bar_ptr=hb.barrier)
            k_prod.tail()
            b_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_empty = acc_prod.acquire_and_advance()
            for t in cutlass.range(n_kt):
                hp = p_cons.wait_and_advance()
                hb = b_cons.wait_and_advance()
                for sub in cutlass.range_constexpr(self.nsub):
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, t > 0)
                    for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        cute.gemm(tiled_mma, tCtAcc[(None, None, None, sub)], tCrA[(None, None, kb, hp.index)],
                                  tCrB[(None, None, kb, hb.index * self.nsub + sub)], tCtAcc[(None, None, None, sub)])
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                hp.release()
                hb.release()
            acc_empty.commit()
            acc_prod.tail()
        elif warp < self.mma_warp:
            tmem.allocate(512)
            # every TMEM user passes the allocation barrier before its main loop (the MMA warp waits there)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            # ---- transform: P = exp2(k * log2e - lse * log2e), 0 on masked rows ----
            g = tidx % 16  # 8 consecutive s rows: 8g .. 8g + 7 of the CTA's 128
            kq = tidx // 16  # time rows kq + 8 r of a stage
            s_base = mb * self.TM + 8 * g
            lse2 = [_combine_lse2(mPart, s_base + e, j + j_off, k, b, self.R) for e in range(8)]
            nlse2 = [-v for v in lse2]
            if const_expr(self.has_seg):
                # keep flags of the split's rows (inside the segment of the chunk's last row), once per CTA
                seg_row0 = b * seg_stride + seg_off + row0
                seg_last = cute.make_tensor(pSeg + (b * seg_stride + seg_off + j * self.C + self.C - 1), cute.make_layout(1))[0]
                for i in cutlass.range(tidx, n_kt * self.KB, 32 * self.num_tp_warps):
                    seg_r = cute.make_tensor(pSeg + (seg_row0 + i), cute.make_layout(1))[0]
                    sKeep[i] = cutlass.select_(seg_r != seg_last, Uint8(0), Uint8(1))
                cute.arch.barrier(barrier_id=3, number_of_threads=32 * self.num_tp_warps)
            stage_elems = self.TM * self.KB
            nr = self.KB // 8
            offs = [self._p_offset(g, kq + 8 * r) for r in range(nr)]
            for t in cutlass.range(n_kt):
                hk = k_cons.wait_and_advance()
                hp = p_prod.acquire_and_advance()
                # all loads of the stage first (the compiler cannot reorder them across the stores to pP)
                words = [cute.make_tensor(cute.recast_ptr((pRaw + (hk.index * stage_elems + offs[r])).align(16), dtype=Uint32),
                                          cute.make_layout(4)).load() for r in range(nr)]
                for r in cutlass.range_constexpr(nr):
                    pv = cute.make_rmem_tensor((8,), Float32)
                    for e2 in cutlass.range_constexpr(4):
                        x0 = _u32_as_f32(words[r][e2] << 16)
                        x1 = _u32_as_f32(words[r][e2] & Uint32(0xFFFF0000))
                        y0, y1 = cute.arch.fma_packed_f32x2((x0, x1), (LOG2E, LOG2E), (nlse2[2 * e2], nlse2[2 * e2 + 1]))
                        pv[2 * e2] = cute.math.exp2(y0, fastmath=True)
                        pv[2 * e2 + 1] = cute.math.exp2(y1, fastmath=True)
                    if const_expr(self.has_seg):
                        if sKeep[t * self.KB + kq + 8 * r] == 0:
                            pv.fill(0.0)
                    cute.make_tensor((pP + (hp.index * stage_elems + offs[r])).align(16), cute.make_layout(8)).store(pv.load().to(BFloat16))
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                hk.release()
                hp.commit()
            # ---- epilogue: TMEM -> registers -> shared memory -> TMA reduce-add into AB ----
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_full = acc_cons.wait_and_advance()
            # the raw key stages are idle now: they stage the epilogue tiles
            sC = cute.make_tensor(cute.recast_ptr(cute.recast_ptr(pRaw, dtype=Float32), swizzle_=epi_layout.inner), epi_layout.outer)
            epi_bar = pipeline.NamedBarrier(barrier_id=3, num_threads=32 * self.num_tp_warps)
            epi_pipe = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stages, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * self.num_tp_warps))
            copy_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE), Float32)
            # (128, 32, RestM, RestN, j, K, B)
            gAB = cute.local_tile(mAB, (self.TM, 32), (None, None, None, None, None))
            tSs, tSg = cpasync.tma_partition(tma_red, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), cute.group_modes(gAB, 0, 2))
            n_epi = 0
            for sub in cutlass.range_constexpr(self.nsub):
                acc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0, sub)], (self.TM, 32))
                tiled_t2r = tcgen05.make_tmem_copy(copy_atom, acc_epi[(None, None, 0, 0)])
                thr_t2r = tiled_t2r.get_slice(tidx)
                tTR = thr_t2r.partition_S(acc_epi)
                tTR_rAcc = cute.make_rmem_tensor(thr_t2r.partition_D(cute.make_identity_tensor((self.TM, 32))).shape, Float32)
                r2s = cute.make_tiled_copy_D(sm100_utils.get_smem_store_op(utils.LayoutEnum.ROW_MAJOR, Float32, Float32, tiled_t2r), tiled_t2r)
                thr_r2s = r2s.get_slice(tidx)
                tRS_sC = thr_r2s.partition_D(sC)
                tRS_r = r2s.retile(tTR_rAcc)
                col_tile0 = (nb * self.nsub + sub) * (self.NS // 32)
                for cchunk in cutlass.range_constexpr(self.NS // 32):
                    buf = n_epi % self.epi_stages
                    n_epi += 1
                    cute.copy(tiled_t2r, tTR[(None, None, None, 0, cchunk)], tTR_rAcc)
                    cute.copy(r2s, tRS_r, tRS_sC[(None, None, None, buf)])
                    cute.arch.fence_view_async_shared()
                    epi_bar.arrive_and_wait()
                    if warp == 0:
                        cute.copy(tma_red, tSs[(None, buf)], tSg[(None, mb, col_tile0 + cchunk, j + j_off, k, b)])
                        epi_pipe.producer_commit()
                        epi_pipe.producer_acquire()
                    epi_bar.arrive_and_wait()
            epi_pipe.producer_tail()
            with cute.arch.elect_one():
                acc_full.release()
            tmem.relinquish_alloc_permit()
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_tp_warps)
            tmem.free(tmem_ptr)


class AwmRecurrence:
    """Phase 2: per (b, kv head) the memory over the chunks:

        M_c is the state used by chunk c; the update at chunk c (c >= first) uses AB / lse of update index c - first:
        M, lnt <- 0, -inf where memory_mask[b, c];   lnt' = logaddexp(lse, lnt);   r = exp(lse - lnt')
        M <- (1 - r) M + r (B - A M)

    Writes MT[b, k, c] = M_c^T (bf16, the B operand of phase 3), the final memory / log normalizer, and flags
    chunks whose memory is still empty (lnt[b, 0, 0] = -inf).

    One CTA (8 warps) per (b, k, VT columns of V). M stays in fp32 registers in the warp-MMA accumulator layout;
    A M runs as bf16 warp MMAs (fp32 accumulation) on a bf16 copy of A (half the L2 traffic of the n_vt CTAs
    that each read all of A). The A blocks (S x KB) of all updates do not depend on M, so they stream through
    a cp.async ring, STAGES - 1 blocks in flight."""

    def __init__(self, K: int, S: int, V: int, C: int, R: int, has_memory: bool, has_prev: bool, has_seg: bool,
                 VT: int = 64, save: bool = False):
        self.K, self.S, self.V, self.C, self.R = K, S, V, C, R
        self.has_memory, self.has_prev, self.has_seg, self.save = has_memory, has_prev, has_seg, save
        self.VT = min(VT, V)
        self.KB = 64
        self.STAGES = 4  # A blocks in flight: STAGES - 1
        self.NAB = S + V
        self.threads = 256
        self.PAD = 8  # bf16 elements of row padding in shared memory (bank conflicts of the fragment loads)
        assert V % self.VT == 0 and S % self.KB == 0 and S % 128 == 0 and self.VT % 16 == 0

    @cute.jit
    def __call__(self, tPart: cute.Tensor, tAB: cute.Tensor, tA16: cute.Tensor, tMemIn: cute.Tensor, tLntIn: cute.Tensor,
                 tSeg: cute.Tensor, tPrevCount: cute.Tensor, tDead: cute.Tensor, tMT: cute.Tensor, tMemOut: cute.Tensor,
                 tLntOut: cute.Tensor, tMseq: cute.Tensor, tD: cute.Tensor, tR: cute.Tensor, tMneg: cute.Tensor, B: Int32,
                 nc: Int32, nu: Int32, seg_off: Int32, seg_stride: Int32, stream: cuda.CUstream):
        K, S, V, R = self.K, self.S, self.V, self.R
        mMseq, mD, mR, mMneg = _saved_views(tMseq, tD, tR, tMneg, S, V, nc, nu, K, B)
        mPart = cute.make_tensor(tPart.iterator, cute.make_layout((2, S, R, nu, K, B), stride=(
            1, 2, 2 * S, 2 * S * R, 2 * S * R * nu, 2 * S * R * nu * K)))
        mAB = cute.make_tensor(tAB.iterator, cute.make_layout((self.NAB, S, nu, K, B), stride=(
            1, self.NAB, S * self.NAB, nu * S * self.NAB, K * nu * S * self.NAB)))
        # bf16 copy of A: [B, K, nu, S, S] (t contiguous)
        mA16 = cute.make_tensor(tA16.iterator, cute.make_layout((S, S, nu, K, B), stride=(1, S, S * S, nu * S * S, K * nu * S * S)))
        mMemIn = cute.make_tensor(tMemIn.iterator, cute.make_layout((V, S, K, B), stride=(1, V, S * V, K * S * V)))
        mLntIn = cute.make_tensor(tLntIn.iterator, cute.make_layout((S, K, B), stride=(1, S, K * S)))
        mDead = cute.make_tensor(tDead.iterator, cute.make_layout((nc, B), stride=(1, nc)))
        mMT = cute.make_tensor(tMT.iterator, cute.make_layout((S, V, nc, K, B), stride=(1, S, V * S, nc * V * S, K * nc * V * S)))
        mMemOut = cute.make_tensor(tMemOut.iterator, cute.make_layout((V, S, K, B), stride=(1, V, S * V, K * S * V)))
        mLntOut = cute.make_tensor(tLntOut.iterator, cute.make_layout((S, K, B), stride=(1, S, K * S)))
        op = cute.nvgpu.warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(op, cute.make_layout((8, 1, 1)), permutation_mnk=(128, 16, 16))
        grid = B * K * (V // self.VT)
        self.kernel(tiled_mma, mPart, mAB, mA16, mMemIn, mLntIn, tSeg.iterator, tPrevCount.iterator, mDead, mMT,
                    mMemOut, mLntOut, mMseq, mD, mR, mMneg, nc, nu, seg_off, seg_stride).launch(grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.jit
    def _load_a_block(self, mA16: cute.Tensor, sA: cute.Tensor, atom, q, n_blocks, nkb, k, b, tidx):
        """cp.async of A block q (update q // nkb, columns (q % nkb) * KB ...) into ring buffer q % STAGES; one
        commit group per call, empty past the last block (keeps the group count of the wait uniform)."""
        if q < n_blocks:
            u = q // nkb
            col0 = (q - u * nkb) * self.KB
            rq = tidx // (self.KB // 8)
            cq = (tidx % (self.KB // 8)) * 8
            src0 = mA16.iterator + mA16.layout((col0 + cq, rq, u, k, b))
            dst0 = sA.iterator + sA.layout((rq, cq, q % self.STAGES))
            rows_per_it = self.threads * 8 // self.KB
            for it in cutlass.range_constexpr(self.S // rows_per_it):
                src = cute.make_tensor((src0 + it * rows_per_it * self.S).align(16), cute.make_layout(8))
                dst = cute.make_tensor((dst0 + it * rows_per_it * (self.KB + self.PAD)).align(16), cute.make_layout(8))
                cute.copy(atom, src, dst)
        cute.arch.cp_async_commit_group()

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, mPart: cute.Tensor, mAB: cute.Tensor, mA16: cute.Tensor, mMemIn: cute.Tensor,
               mLntIn: cute.Tensor, pSeg: cute.Pointer, pPrevCount: cute.Pointer, mDead: cute.Tensor, mMT: cute.Tensor,
               mMemOut: cute.Tensor, mLntOut: cute.Tensor, mMseq: cute.Tensor, mD: cute.Tensor, mR: cute.Tensor,
               mMneg: cute.Tensor, nc: Int32, nu: Int32, seg_off: Int32, seg_stride: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        S, VT, KB = self.S, self.VT, self.KB
        n_vt = self.V // VT
        vt = bidx % n_vt
        k = (bidx // n_vt) % self.K
        b = bidx // (n_vt * self.K)
        v0 = vt * VT
        nkb = S // KB
        smem = utils.SmemAllocator()
        sA = smem.allocate_tensor(BFloat16, cute.make_layout((S, KB, self.STAGES), stride=(KB + self.PAD, 1, S * (KB + self.PAD))),
                                  byte_alignment=16)
        sMT = smem.allocate_tensor(BFloat16, cute.make_layout((VT, S), stride=(S + self.PAD, 1)), byte_alignment=16)
        sR = smem.allocate_tensor(Float32, cute.make_layout(S), byte_alignment=16)
        atom = cute.make_copy_atom(cpasync.CopyG2SOp(), BFloat16, num_bits_per_copy=128)

        first = 0 if const_expr(self.has_prev) else 1  # first chunk with an update
        n_blocks = nu * nkb  # A blocks of all updates, streamed in order
        for p0 in cutlass.range_constexpr(self.STAGES - 1):
            self._load_a_block(mA16, sA, atom, Int32(p0), n_blocks, nkb, k, b, tidx)

        thr_mma = tiled_mma.get_slice(tidx)
        coords = thr_mma.partition_C(cute.make_identity_tensor((S, VT)))
        m = cute.make_rmem_tensor(coords.shape, Float32)
        acc = cute.make_rmem_tensor(coords.shape, Float32)
        bval = cute.make_rmem_tensor(coords.shape, Float32)
        n_frag = cute.size(coords)
        if const_expr(self.has_memory):
            for i in cutlass.range_constexpr(n_frag):
                s1, v1 = coords[i]
                m[i] = mMemIn[v0 + v1, s1, k, b].to(Float32)
        else:
            m.fill(0.0)
        lnt = Float32(-math.inf)
        if tidx < S:
            if const_expr(self.has_memory):
                lnt = mLntIn[tidx, k, b]

        q = Int32(0)  # next A block to consume
        for c in cutlass.range(nc):
            for i in cutlass.range_constexpr(n_frag):
                s2, v2 = coords[i]
                mMT[s2, v0 + v2, c, k, b] = m[i].to(BFloat16)
                if const_expr(self.save):
                    mMseq[v0 + v2, s2, c, k, b] = m[i].to(BFloat16)
            if k == 0 and vt == 0 and tidx == 0:
                mDead[c, b] = cutlass.select_(lnt == Float32(-math.inf), Uint8(1), Uint8(0))
            if c >= first:
                u = c - first
                for i in cutlass.range_constexpr(n_frag):
                    s6, v6 = coords[i]
                    bval[i] = mAB[S + v0 + v6, s6, u, k, b]
                reset = False
                if const_expr(self.has_seg):
                    # the memory restarts when the chunks before the update's chunk end in different segments
                    if c >= 2 or const_expr(self.has_prev):
                        reset = _seg_anchor(pSeg, pPrevCount, b, c - 2, seg_off, seg_stride, self.C) != \
                            _seg_anchor(pSeg, pPrevCount, b, c - 1, seg_off, seg_stride, self.C)
                if reset:
                    m.fill(0.0)
                if tidx < S:
                    if reset:
                        lnt = Float32(-math.inf)
                    l_cur = _combine_lse2(mPart, tidx, u, k, b, self.R) * (1.0 / LOG2E)
                    hi = cute.arch.fmax(l_cur, lnt)
                    new_lnt = hi + cute.math.log(cute.math.exp(l_cur - hi) + cute.math.exp(lnt - hi))
                    r_row = cute.math.exp(l_cur - new_lnt)
                    sR[tidx] = r_row
                    if const_expr(self.save):
                        mR[tidx, u, k, b] = r_row
                    lnt = new_lnt
                # M^T (bf16) into shared memory: the B operand
                for i in cutlass.range_constexpr(n_frag):
                    s3, v3 = coords[i]
                    sMT[v3, s3] = m[i].to(BFloat16)
                    if const_expr(self.save):
                        mMneg[v0 + v3, s3, u, k, b] = (-m[i]).to(BFloat16)
                acc.fill(0.0)
                for kb in cutlass.range(nkb):
                    buf = q % self.STAGES
                    cute.arch.cp_async_wait_group(self.STAGES - 2)
                    cute.arch.barrier()  # block q landed for everyone; block q - 1's buffer is free; sMT / sR written
                    self._load_a_block(mA16, sA, atom, q + self.STAGES - 1, n_blocks, nkb, k, b, tidx)
                    tCsA = thr_mma.partition_A(sA[None, None, buf])
                    tCsB = thr_mma.partition_B(cute.local_tile(sMT, (VT, KB), (0, kb)))
                    tCrA = tiled_mma.make_fragment_A(tCsA)
                    tCrB = tiled_mma.make_fragment_B(tCsB)
                    cute.autovec_copy(tCsA, tCrA)
                    cute.autovec_copy(tCsB, tCrB)
                    for kk in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        cute.gemm(tiled_mma, acc, tCrA[None, None, kk], tCrB[None, None, kk], acc)
                    q += 1
                for i in cutlass.range_constexpr(n_frag):
                    s4, v4 = coords[i]
                    r = sR[s4]
                    d = bval[i] - acc[i] - m[i]  # M' = M + r D
                    if const_expr(self.save):
                        mD[v0 + v4, s4, u, k, b] = d
                    m[i] = m[i] + r * d
                cute.arch.barrier()  # sMT / sR are rewritten in the next chunk
        for i in cutlass.range_constexpr(n_frag):
            s5, v5 = coords[i]
            mMemOut[v0 + v5, s5, k, b] = m[i].to(BFloat16)
        if vt == 0 and tidx < S:
            mLntOut[tidx, k, b] = lnt


class AwmRecurrenceUmma:
    """Phase 2 on tcgen05 (same contract as AwmRecurrence). One CTA per (b, k, VT columns of V):

      * the memory state lives in shared memory as M^T (fp32, K-major tf32 layout): it is the B operand of the
        MMAs and the state updated in place
      * TMA warp:   streams the A blocks (S x 32, fp32) of all updates through a ring (they do not depend on M)
      * MMA warp:   per update, acc[h] = A[h-th 128 rows] M (tf32, fp32 accumulation in TMEM)
      * 4 * S/128 update warps (one row s per thread): MT / dead flags / resets / r, then
        M[s, :] = (1 - r) M[s, :] + r (B[s, :] - acc[s, :]) from TMEM, written back to shared memory
    """

    def __init__(self, K: int, S: int, V: int, C: int, R: int, has_memory: bool, has_prev: bool, has_seg: bool,
                 VT: int = 64, save: bool = False):
        self.K, self.S, self.V, self.C, self.R = K, S, V, C, R
        self.has_memory, self.has_prev, self.has_seg, self.save = has_memory, has_prev, has_seg, save
        self.VT = min(VT, V)
        self.KB = 32  # 128 bytes of fp32: one swizzle atom
        self.stages = 4
        self.halves = S // 128
        self.NAB = S + V
        self.num_upd_warps = 4 * self.halves
        self.mma_warp = self.num_upd_warps
        self.tma_warp = self.num_upd_warps + 1
        self.threads = 32 * (self.num_upd_warps + 2)
        self.tmem_cols = max(32, 1 << (self.halves * self.VT - 1).bit_length())
        assert S % 128 == 0 and V % self.VT == 0 and self.VT % 32 == 0 and self.halves * self.VT <= 512

    @cute.jit
    def __call__(self, tPart: cute.Tensor, tAB: cute.Tensor, tMemIn: cute.Tensor, tLntIn: cute.Tensor,
                 tSeg: cute.Tensor, tPrevCount: cute.Tensor, tDead: cute.Tensor, tMT: cute.Tensor, tMemOut: cute.Tensor,
                 tLntOut: cute.Tensor, tMseq: cute.Tensor, tD: cute.Tensor, tR: cute.Tensor, tMneg: cute.Tensor, B: Int32,
                 nc: Int32, nu: Int32, seg_off: Int32, seg_stride: Int32, stream: cuda.CUstream):
        K, S, V, R = self.K, self.S, self.V, self.R
        mMseq, mD, mR, mMneg = _saved_views(tMseq, tD, tR, tMneg, S, V, nc, nu, K, B)
        mPart = cute.make_tensor(tPart.iterator, cute.make_layout((2, S, R, nu, K, B), stride=(
            1, 2, 2 * S, 2 * S * R, 2 * S * R * nu, 2 * S * R * nu * K)))
        mAB = cute.make_tensor(tAB.iterator, cute.make_layout((self.NAB, S, nu, K, B), stride=(
            1, self.NAB, S * self.NAB, nu * S * self.NAB, K * nu * S * self.NAB)))
        # A blocks: AB[:, :, :S] as (M = s, K = t, update, K, B), K-major
        mA = cute.make_tensor(cute.recast_ptr(tAB.iterator, dtype=TFloat32), cute.make_layout((S, S, nu, K, B), stride=(
            self.NAB, 1, S * self.NAB, nu * S * self.NAB, K * nu * S * self.NAB)))
        mMemIn = cute.make_tensor(tMemIn.iterator, cute.make_layout((V, S, K, B), stride=(1, V, S * V, K * S * V)))
        mLntIn = cute.make_tensor(tLntIn.iterator, cute.make_layout((S, K, B), stride=(1, S, K * S)))
        mDead = cute.make_tensor(tDead.iterator, cute.make_layout((nc, B), stride=(1, nc)))
        mMT = cute.make_tensor(tMT.iterator, cute.make_layout((S, V, nc, K, B), stride=(1, S, V * S, nc * V * S, K * nc * V * S)))
        mMemOut = cute.make_tensor(tMemOut.iterator, cute.make_layout((V, S, K, B), stride=(1, V, S * V, K * S * V)))
        mLntOut = cute.make_tensor(tLntOut.iterator, cute.make_layout((S, K, B), stride=(1, S, K * S)))

        op = tcgen05.MmaTF32Op((128, self.VT, 8), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                               tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
        tiled_mma = cute.make_tiled_mma(op)
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, (128, self.VT, self.KB), TFloat32, self.stages * self.halves)
        b_layout = sm100_utils.make_smem_layout_b(tiled_mma, (128, self.VT, S), TFloat32, 1)
        tma_a, mA_t = cute.nvgpu.make_tiled_tma_atom_A(cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE), mA,
                                                       cute.slice_(a_layout, (None, None, None, 0)),
                                                       (128, self.VT, self.KB), tiled_mma)
        grid = B * K * (V // self.VT)
        self.kernel(tiled_mma, tma_a, mA_t, mPart, mAB, mMemIn, mLntIn, tSeg.iterator, tPrevCount.iterator, mDead,
                    mMT, mMemOut, mLntOut, mMseq, mD, mR, mMneg, a_layout, b_layout, nc, nu, seg_off, seg_stride).launch(
            grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, tma_a: cute.CopyAtom, mA: cute.Tensor, mPart: cute.Tensor,
               mAB: cute.Tensor, mMemIn: cute.Tensor, mLntIn: cute.Tensor, pSeg: cute.Pointer, pPrevCount: cute.Pointer,
               mDead: cute.Tensor, mMT: cute.Tensor, mMemOut: cute.Tensor, mLntOut: cute.Tensor, mMseq: cute.Tensor,
               mD: cute.Tensor, mR: cute.Tensor, mMneg: cute.Tensor, a_layout: cute.ComposedLayout,
               b_layout: cute.ComposedLayout, nc: Int32, nu: Int32, seg_off: Int32, seg_stride: Int32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        bidx, _, _ = cute.arch.block_idx()
        S, VT, KB = self.S, self.VT, self.KB
        n_vt = self.V // VT
        vt = bidx % n_vt
        k = (bidx // n_vt) % self.K
        b = bidx // (n_vt * self.K)
        v0 = vt * VT
        nkb = S // KB
        first = 0 if const_expr(self.has_prev) else 1  # first chunk with an update

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_a)

        @cute.struct
        class SharedStorage:
            a_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            m_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_upd_warps + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        upd = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * self.num_upd_warps)
        a_bytes = cute.size_in_bytes(TFloat32, cute.select(a_layout, mode=[0, 1, 2])) * self.halves
        a_prod, a_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_mbar.data_ptr(), num_stages=self.stages, producer_group=one, consumer_group=one,
            tx_count=a_bytes).make_participants()
        acc_prod, acc_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(), num_stages=1, producer_group=one,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_upd_warps)).make_participants()
        m_prod, m_cons = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.m_mbar.data_ptr(), num_stages=1, producer_group=upd,
            consumer_group=one).make_participants()

        sA = smem.allocate_tensor(TFloat32, a_layout.outer, byte_alignment=1024, swizzle=a_layout.inner)
        pM = smem.allocate_array(Float32, cute.cosize(b_layout.outer), byte_alignment=1024)
        sMT = cute.make_tensor(cute.recast_ptr(cute.recast_ptr(pM, dtype=TFloat32), swizzle_=b_layout.inner), b_layout.outer)

        gA = cute.local_tile(mA, (128, KB), (None, None, None, None, None))
        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sMT)
        acc_shape = tiled_mma.partition_shape_C((128, VT))
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.halves))
        tAs, tAg = cpasync.tma_partition(tma_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
        # updates actually present: nu is max(nc - 1 + has_prev, 1) for the tensor strides; with a single chunk and no
        # previous chunk there is none (the MMA warp would wait forever for an update the compute warps never produce)
        nu_act = nc - (0 if const_expr(self.has_prev) else 1)
        n_blocks = nu_act * nkb

        if warp == self.tma_warp:
            for q in cutlass.range(n_blocks):
                u = q // nkb
                kb = q - u * nkb
                ha = a_prod.acquire_and_advance()
                for h in cutlass.range_constexpr(self.halves):
                    cute.copy(tma_a, tAg[(None, h, kb, u, k, b)], tAs[(None, ha.index * self.halves + h)], tma_bar_ptr=ha.barrier)
            a_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tCtAcc = cute.make_tensor(tmem.retrieve_ptr(Float32), tCtAcc_fake.layout)
            kpb = cute.size(tCrA, mode=[2])  # MMA k steps per A block
            for u in cutlass.range(nu_act):
                hm = m_cons.wait_and_advance()
                acc_empty = acc_prod.acquire_and_advance()
                for kb in cutlass.range(nkb):
                    ha = a_cons.wait_and_advance()
                    for h in cutlass.range_constexpr(self.halves):
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, kb > 0)
                        for kk in cutlass.range_constexpr(kpb):
                            cute.gemm(tiled_mma, tCtAcc[(None, None, None, h)], tCrA[(None, None, kk, ha.index * self.halves + h)],
                                      tCrB[(None, None, kb * kpb + kk, 0)], tCtAcc[(None, None, None, h)])
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    ha.release()
                acc_empty.commit()
                hm.release()
            acc_prod.tail()
        else:
            tmem.allocate(self.tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            half = warp // 4
            srow = half * 128 + 32 * (warp % 4) + lane  # this thread's row of M (= TMEM lane of its half)
            # M^T[v, srow] at base + 32 v + 4 ((c16 ^ (v & 7)) ...): the 128-byte swizzle in 4-element chunks
            mt_base = (srow // 32) * (32 * VT) + srow % 4
            c16 = (srow % 32) // 4
            mt_x = [mt_base + 4 * (c16 ^ e) for e in range(8)]
            # this row of the state lives in shared memory only (M^T, the MMA's B operand), updated in place
            if const_expr(self.has_memory):
                mrow = cute.make_tensor(mMemIn.iterator + mMemIn.layout((v0, srow, k, b)), cute.make_layout(VT)).load()
                for v in cutlass.range_constexpr(VT):
                    pM[mt_x[v % 8] + 32 * v] = mrow[v].to(Float32)
            else:
                for v in cutlass.range_constexpr(VT):
                    pM[mt_x[v % 8] + 32 * v] = Float32(0.0)
            lnt = Float32(-math.inf)
            if const_expr(self.has_memory):
                lnt = mLntIn[srow, k, b]
            copy_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE), Float32)
            acc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0, half)], (128, 32))
            tiled_t2r = tcgen05.make_tmem_copy(copy_atom, acc_epi[(None, None, 0, 0)])
            thr_t2r = tiled_t2r.get_slice(tidx % 128)
            tTR = thr_t2r.partition_S(acc_epi)
            regs = cute.make_rmem_tensor(thr_t2r.partition_D(cute.make_identity_tensor((128, 32))).shape, Float32)
            dvec = cute.make_rmem_tensor((32,), Float32)
            mo = cute.make_rmem_tensor((VT,), BFloat16)
            for c in cutlass.range(nc):
                for v in cutlass.range_constexpr(VT):
                    mMT[srow, v0 + v, c, k, b] = pM[mt_x[v % 8] + 32 * v].to(BFloat16)
                if const_expr(self.save):
                    for v in cutlass.range_constexpr(VT):
                        mo[v] = pM[mt_x[v % 8] + 32 * v].to(BFloat16)
                    cute.make_tensor(mMseq.iterator + mMseq.layout((v0, srow, c, k, b)), cute.make_layout(VT)).store(mo.load())
                if k == 0 and vt == 0 and tidx == 0:
                    mDead[c, b] = cutlass.select_(lnt == Float32(-math.inf), Uint8(1), Uint8(0))
                if c >= first:
                    u = c - first
                    pB = mAB.iterator + mAB.layout((S + v0, srow, u, k, b))
                    reset = False
                    if const_expr(self.has_seg):
                        if c >= 2 or const_expr(self.has_prev):
                            reset = _seg_anchor(pSeg, pPrevCount, b, c - 2, seg_off, seg_stride, self.C) != \
                                _seg_anchor(pSeg, pPrevCount, b, c - 1, seg_off, seg_stride, self.C)
                    hmp = m_prod.acquire_and_advance()
                    if reset:
                        for v in cutlass.range_constexpr(VT):
                            pM[mt_x[v % 8] + 32 * v] = Float32(0.0)
                        lnt = Float32(-math.inf)
                    l_cur = _combine_lse2(mPart, srow, u, k, b, self.R) * (1.0 / LOG2E)
                    hi = cute.arch.fmax(l_cur, lnt)
                    new_lnt = hi + cute.math.log(cute.math.exp(l_cur - hi) + cute.math.exp(lnt - hi))
                    r = cute.math.exp(l_cur - new_lnt)
                    if const_expr(self.save):
                        mR[srow, u, k, b] = r
                        for v in cutlass.range_constexpr(VT):
                            mo[v] = (-pM[mt_x[v % 8] + 32 * v]).to(BFloat16)
                        cute.make_tensor(mMneg.iterator + mMneg.layout((v0, srow, u, k, b)), cute.make_layout(VT)).store(mo.load())
                    lnt = new_lnt
                    # hand the state to the MMA warp
                    cute.arch.fence_view_async_shared()
                    hmp.commit()
                    acc_full = acc_cons.wait_and_advance()
                    for ch in cutlass.range_constexpr(VT // 32):
                        bval = cute.make_tensor((pB + ch * 32).align(16), cute.make_layout(32)).load()
                        cute.copy(tiled_t2r, tTR[(None, None, None, 0, ch)], regs)
                        for e in cutlass.range_constexpr(32):
                            off = mt_x[e % 8] + 32 * (ch * 32 + e)
                            d = bval[e] - regs[e] - pM[off]  # M' = M + r D
                            dvec[e] = d
                            pM[off] = pM[off] + r * d
                        if const_expr(self.save):
                            cute.make_tensor((mD.iterator + mD.layout((v0 + ch * 32, srow, u, k, b))).align(16),
                                             cute.make_layout(32)).store(dvec.load())
                    with cute.arch.elect_one():
                        acc_full.release()
            for v in cutlass.range_constexpr(VT):
                mo[v] = pM[mt_x[v % 8] + 32 * v].to(BFloat16)
            cute.make_tensor(mMemOut.iterator + mMemOut.layout((v0, srow, k, b)), cute.make_layout(VT)).store(mo.load())
            if vt == 0:
                mLntOut[srow, k, b] = lnt
            tmem.relinquish_alloc_permit()
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_upd_warps)
            tmem.free(tmem_ptr)


@cute.jit
def _saved_views(tMseq: cute.Tensor, tD: cute.Tensor, tR: cute.Tensor, tMneg: cute.Tensor, S, V, nc, nu, K, B):
    """Views of the tensors a saving recurrence writes for the backward: Mseq[b, k, c, s, v] (bf16, the state of
    chunk c before any reset), D[b, k, u, s, v] = B - A M - M of update u (fp32, so that M' = M + r D),
    R[b, k, u, s] = r and Mneg[b, k, u, s, v] = -M (bf16, the state update u starts from, 0 after a reset).
    All as (v, s, ...) / (s, ...) coordinates."""
    mMseq = cute.make_tensor(tMseq.iterator, cute.make_layout((V, S, nc, K, B), stride=(1, V, S * V, nc * S * V, K * nc * S * V)))
    mD = cute.make_tensor(tD.iterator, cute.make_layout((V, S, nu, K, B), stride=(1, V, S * V, nu * S * V, K * nu * S * V)))
    mR = cute.make_tensor(tR.iterator, cute.make_layout((S, nu, K, B), stride=(1, S, nu * S, K * nu * S)))
    mMneg = cute.make_tensor(tMneg.iterator, cute.make_layout((V, S, nu, K, B), stride=(1, V, S * V, nu * S * V, K * nu * S * V)))
    return mMseq, mD, mR, mMneg


class AwmAToBf16:
    """A16[b, k, u, s, :] = bf16(AB[b, k, u, s, :S]): the bf16 copy of A read by the warp-MMA recurrence.
    One thread per 8 consecutive elements (32-byte load, 16-byte store)."""

    def __init__(self, S: int, V: int):
        self.S, self.NAB = S, S + V
        self.threads = 256

    @cute.jit
    def __call__(self, tAB: cute.Tensor, tA16: cute.Tensor, rows: Int32, stream: cuda.CUstream):
        per_row = self.S // 8
        total = rows * per_row
        grid = (total + self.threads - 1) // self.threads
        self.kernel(tAB.iterator, tA16.iterator, total).launch(grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pAB: cute.Pointer, pA16: cute.Pointer, total: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * self.threads + tidx
        if i < total:
            row = i // (self.S // 8)
            col = (i - row * (self.S // 8)) * 8
            x = cute.make_tensor((pAB + (row * self.NAB + col)).align(32), cute.make_layout(8)).load()
            cute.make_tensor((pA16 + (row * self.S + col)).align(16), cute.make_layout(8)).store(x.to(BFloat16))


class AwmLse2:
    """lse2[b, k, u, s] = mm + log2(sum_r l_r exp2(m_r - mm)), mm = max_r m_r: the combined base-2 log normalizer of
    the R partials (m, l) of part [B, K, nu_all, R, S, 2] for updates j_off .. j_off + nu - 1 (one thread per output;
    the 7 torch ops of _lse2_of in one launch)."""

    def __init__(self, R: int, S: int):
        self.R, self.S = R, S
        self.threads = 256

    @cute.jit
    def __call__(self, tPart: cute.Tensor, tOut: cute.Tensor, total: Int32, nu: Int32, nu_all: Int32, j_off: Int32,
                 stream: cuda.CUstream):
        grid = (total + self.threads - 1) // self.threads
        self.kernel(tPart.iterator, tOut.iterator, total, nu, nu_all, j_off).launch(
            grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pPart: cute.Pointer, pOut: cute.Pointer, total: Int32, nu: Int32, nu_all: Int32, j_off: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * self.threads + tidx
        if i < total:
            S, R = self.S, self.R
            si = i % S
            t = i // S
            u = t % nu
            bk = t // nu
            base = (Int64(bk) * nu_all + j_off + u) * (R * S * 2) + si * 2
            m = cute.make_rmem_tensor(cute.make_layout(R), Float32)
            mm = Float32(-math.inf)
            for r in cutlass.range_constexpr(R):
                m[r] = cute.make_tensor(pPart + (base + r * S * 2), cute.make_layout(1))[0]
                mm = cute.arch.fmax(mm, m[r])
            acc = Float32(0.0)
            for r in cutlass.range_constexpr(R):
                acc = acc + cute.make_tensor(pPart + (base + r * S * 2 + 1), cute.make_layout(1))[0] * cute.math.exp2(m[r] - mm)
            cute.make_tensor(pOut + i, cute.make_layout(1))[0] = mm + cute.math.log2(acc)


# ======================================================================================================
# Host side
# ======================================================================================================

import torch  # noqa: E402

_COMPILED = {}


def _fake(dtype, align=16):
    return cute.runtime.make_fake_compact_tensor(dtype, (cute.sym_int(),), assumed_align=align)


def _fake_stream():
    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)


def _compiled(key, make_op, make_args):
    """The compiled kernel for ``key`` (fake arguments are only built on the first call)."""
    fn = _COMPILED.get(key)
    if fn is None:
        fn = _COMPILED[key] = cute.compile(make_op(), *make_args(), _fake_stream(), options="--enable-tvm-ffi")
    return fn


_U8 = cutlass.Uint8


def _sm_count(device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return _sm_count_of(idx)


@lru_cache(maxsize=None)
def _sm_count_of(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


_AWM_SM90_CUBLAS = os.environ.get("AWM_SM90_CUBLAS", "0") == "1"  # Hopper: cuBLAS reference paths instead of the wgmma kernels


@lru_cache(maxsize=None)
def _sm90():
    """The Hopper kernel module (adaptive_working_memory_sm90.py), also when this file is loaded by path (tests)."""
    try:
        from . import adaptive_working_memory_sm90 as mod
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "cute_adaptive_working_memory_sm90", os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaptive_working_memory_sm90.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def _is_hopper(device) -> bool:
    """No tcgen05 below sm_100: the GEMM phases run on cuBLAS instead (bf16 operands, fp32 accumulation)."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return _major_of(idx) < 10


@lru_cache(maxsize=None)
def _major_of(device_index: int) -> int:
    return torch.cuda.get_device_capability(device_index)[0]


def _chunk_ab_cublas(k, qk, v, seg, part, AB, K, S, V, C, nu, j_off, seg_off, has_seg):
    """AwmChunkGemm on cuBLAS: P[s, t] = exp2(k[t, s] log2e - lse2[s]) (bf16, 0 on rows outside the segment of
    the chunk's last row), AB[:, :, j_off + j] = P_j [qk_j | v_j] with fp32 accumulation."""
    B = k.shape[0]
    p = part[:, :, j_off:j_off + nu]  # [B, K, nu, R, S, (m, l)]
    m, l = p[..., 0], p[..., 1]
    mm = m.amax(3, keepdim=True)
    lse2 = mm.squeeze(3) + torch.log2((l * torch.exp2(m - mm)).sum(3))  # [B, K, nu, S]
    kc = k[:, :nu * C].view(B, nu, C, K, S)
    e = torch.exp2(kc.float() * LOG2E - lse2.permute(0, 2, 1, 3).unsqueeze(2))
    if has_seg:
        sg = seg[:, seg_off:seg_off + nu * C].view(B, nu, C)
        e = torch.where((sg == sg[:, :, -1:]).view(B, nu, C, 1, 1), e, 0.0)
    pt = e.to(torch.bfloat16).permute(0, 3, 1, 4, 2).reshape(B * K * nu, S, C)
    rhs = torch.cat([qk[:, :nu * C].view(B, nu, C, K, S), v[:, :nu * C].view(B, nu, C, K, V)], -1)
    rhs = rhs.permute(0, 3, 1, 2, 4).reshape(B * K * nu, C, S + V)
    AB[:, :, j_off:j_off + nu] = torch.bmm(pt, rhs, out_dtype=torch.float32).view(B, K, nu, S, S + V)


def _lse2_of(part, j_off, nu):
    """Combined base-2 lse [B, K, nu, S] of the R partials (m, l) of update chunks j_off .. j_off + nu - 1."""
    B, K, nu_all, R, S, _ = part.shape
    if not part.is_contiguous() or os.environ.get("XLLM_AWM_LSE2_TORCH") == "1":
        p = part[:, :, j_off:j_off + nu]
        m, l = p[..., 0], p[..., 1]
        mm = m.amax(3, keepdim=True)
        return (mm.squeeze(3) + torch.log2((l * torch.exp2(m - mm)).sum(3))).contiguous()
    out = torch.empty(B, K, nu, S, dtype=torch.float32, device=part.device)
    f = _compiled(("lse2", R, S), lambda: AwmLse2(R, S), lambda: (_fake(Float32, 8), _fake(Float32, 4), *([Int32(0)] * 4)))
    f(part.reshape(-1), out.reshape(-1), out.numel(), nu, nu_all, j_off)
    return out


_SEG_BOUNDS = []  # [(weakref to seg, version, seg_off, nu, C, lo)]: the per-update row bounds, cached per step
_HI = {}


def _chunk_row_bounds(seg, seg_off, B, nu, C, has_seg, dev):
    """(lo, hi) int32 [B, nu]: the rows [lo, C) of each update chunk in the segment of its last row (all rows
    without segments). They depend on the segment ids only, so they are cached across the layers (and the
    backward) of a step on the segment tensor object."""
    key = (B, nu, C, dev)
    if key not in _HI:
        _HI[key] = torch.full((B, nu), C, dtype=torch.int32, device=dev)
    hi = _HI[key]
    if not has_seg:
        return torch.zeros(B, nu, dtype=torch.int32, device=dev), hi
    for ref, ver, off, n, c, lo in reversed(_SEG_BOUNDS):
        if ref() is seg and ver == seg._version and off == seg_off and n == nu and c == C:
            return lo, hi
    sg = seg[:, seg_off:seg_off + nu * C].view(B, nu, C)
    lo = (C - (sg == sg[:, :, -1:]).sum(-1)).to(torch.int32)
    _SEG_BOUNDS.append((weakref.ref(seg), seg._version, seg_off, nu, C, lo))
    del _SEG_BOUNDS[:-8]
    return lo, hi


def _splitk_splits(base_ctas, n_t, device):
    """Splits of the time range so that about two waves of CTAs run."""
    return max(1, min(n_t, -(-2 * _sm_count(device) // base_ctas)))


def _chunk_ab_sm90(k, qk, v, seg, part, AB, K, S, V, C, nu, j_off, seg_off, has_seg):
    """AwmChunkGemm on Hopper (AwmChunkGemmSm90 'ab'): rows of each update chunk in the segment of its last row."""
    B, L = k.shape[0], k.shape[1]
    dev = k.device
    lse2 = _lse2_of(part, j_off, nu)
    lo, hi = _chunk_row_bounds(seg, seg_off, B, nu, C, has_seg, dev)
    sm90 = _sm90()
    m_blocks, n_blocks = S // 128, (S + V) // min(256, S)
    splits = _splitk_splits(B * K * nu * m_blocks * n_blocks, C // 64, dev)
    f = _compiled(("ab90", K, S, V, C, splits), lambda: sm90.AwmChunkGemmSm90("ab", K, S, V, C, 1, splits), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Int32, 4), _fake(Int32, 4), _fake(Float32),
        _fake(Float32), *([Int32(0)] * 6)))
    f(k.reshape(-1), qk.reshape(-1), v.reshape(-1), lo.reshape(-1), hi.reshape(-1), lse2.reshape(-1), AB.reshape(-1),
      B, L, nu, 0, j_off, AB.shape[2])


def _out_gemm_cublas(q, MT, seg, prev_cnt, dead, seg_off, H, K, S, V, C, has_prev, has_seg, out_mask):
    """AwmOutGemm on cuBLAS: out[b, l, h] = q[b, l, h] M_c (c = l // C, kv head h // n_rep), rows zeroed when the
    chunk's memory is empty or the row is not in the segment the memory ends in (that of chunk c - 2)."""
    B, L = q.shape[0], q.shape[1]
    nc, n_rep = L // C, H // K
    qv = q.view(B, nc, C, K, n_rep, S).permute(0, 3, 1, 4, 2, 5).reshape(B * K * nc, n_rep * C, S)
    out = torch.bmm(qv, MT.view(B * K * nc, V, S).transpose(1, 2))
    out = out.view(B, K, nc, n_rep, C, V).permute(0, 2, 4, 1, 3, 5).contiguous().view(B, L, H, V)
    masked = (dead != 0).repeat_interleave(C, 1)  # [B, L]
    if has_seg:
        rows = seg[:, seg_off:seg_off + L]
        idx = torch.arange(nc, device=q.device) - 2  # anchor chunk c - 2: its last row, -1 = the previous call's chunk
        anchor = seg[:, (seg_off + idx * C + C - 1).clamp_min(0)]
        if has_prev:  # -2: the previous segment count
            anchor = torch.where(idx == -2, prev_cnt.view(B, 1), anchor)
        anchor = anchor.repeat_interleave(C, 1)
        if not has_prev:  # chunks 0 and 1 have no anchor chunk: never masked by segment
            anchor = torch.where(torch.arange(L, device=q.device) < 2 * C, rows, anchor)
        masked = masked | (rows != anchor)
    out_mask.copy_(masked)
    return out.masked_fill_(masked.view(B, L, 1, 1), 0.0), out_mask


def _blhs(t: torch.Tensor) -> torch.Tensor:
    """[B, H, L, D] view -> [B, L, H, D] contiguous storage (free for the transposed views of the fused block)."""
    t = t.transpose(1, 2)
    return t if t.is_contiguous() else t.contiguous()


_DUMMY = {}


def _out_tiling(S, V):
    """(v tile, A stages, resident) of the output GEMM: the resident memory tile (VT x S bf16) and the q ring
    share the shared memory; with a single 256-wide v tile the ring holds just one tile of K. When the tile does
    not leave room for the ring, it is streamed in K blocks instead (stages of q and memory blocks)."""
    vt = min(V, 256)
    stages = (192 * 1024 - vt * S * 2) // (128 * 64 * 2)
    if stages >= 2:
        return vt, min(16, stages), True
    return vt, 4, False


def _dummy(dtype, device):
    """A 1-element tensor standing in for an absent optional input (never read)."""
    key = (dtype, device)
    if key not in _DUMMY:
        _DUMMY[key] = torch.zeros(1, dtype=dtype, device=device)
    return _DUMMY[key]


def _chunk_ab(k, qk, v, seg, part, AB, K, S, V, C, R, nu, j_off, seg_off, has_seg):
    """Phase 1 on [B, L, K, *] tensors for update chunks 0..nu-1 (written at chunk index j_off + j of the
    workspaces): lse partials [B, K, nu_tot, R, S, 2] and AB [B, K, nu_tot, S, S + V] (fp32, zeroed)."""
    B, L = k.shape[0], k.shape[1]
    nu_tot = part.shape[2]
    seg_stride = seg.shape[1] if has_seg else 0
    f_lse = _compiled(("lse", K, S, C, R, has_seg), lambda: AwmKeyLsePart(K, S, C, R, has_seg), lambda: (
        _fake(BFloat16), _fake(Int64, 8), _fake(Float32), *([Int32(0)] * 7)))
    f_lse(k.reshape(-1), seg.reshape(-1), part.reshape(-1), B, L, nu, j_off, nu_tot, seg_off, seg_stride)
    if _is_hopper(k.device):
        if _AWM_SM90_CUBLAS:
            _chunk_ab_cublas(k, qk, v, seg, part, AB, K, S, V, C, nu, j_off, seg_off, has_seg)
        else:
            _chunk_ab_sm90(k, qk, v, seg, part, AB, K, S, V, C, nu, j_off, seg_off, has_seg)
        return
    _, _, n_blocks = chunk_gemm_tiling(S, V)
    base = B * K * nu * (S // 128) * n_blocks
    splits = max(1, min(C // 32, round(_sm_count(k.device) / base)))
    f_ab = _compiled(("ab", K, S, V, C, splits, R, has_seg), lambda: AwmChunkGemm(K, S, V, C, splits, R, has_seg),
                     lambda: (_fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Int64, 8), _fake(Float32),
                              _fake(Float32), *([Int32(0)] * 7)))
    f_ab(k.reshape(-1), qk.reshape(-1), v.reshape(-1), seg.reshape(-1), part.reshape(-1), AB.reshape(-1),
         B, L, nu, j_off, nu_tot, seg_off, seg_stride)


def _phases12(qkey, key, value, segment_idx, prev_segment_count, memory, log_norm_term, prev_qkey, prev_key,
              prev_value, C, save=False):
    """Phases 1 and 2 on [B, H, L, *] views: the per-chunk lse partials and [A | B], then the recurrence. Returns
    a dict with the workspaces (part, AB, A16 (bf16 copy of A, when made), MT (the stacked M_c^T, bf16), mem_out,
    lnt_out, dead) and, with ``save``, the tensors of the backward (Mseq, D, R; see _saved_views)."""
    B, K, L, S = qkey.shape
    V = value.shape[3]
    nc = L // C
    dev = qkey.device
    has_prev = prev_key is not None
    has_mem = memory is not None
    has_seg = segment_idx is not None
    qk, k, v = _blhs(qkey), _blhs(key), _blhs(value)
    seg = segment_idx.contiguous() if has_seg else _dummy(torch.int64, dev)
    prev_cnt = prev_segment_count.contiguous() if (has_seg and has_prev) else _dummy(torch.int64, dev)
    seg_off = C if (has_seg and has_prev) else 0  # segment ids of the current rows start after the previous chunk
    seg_stride = seg.shape[1] if has_seg else 0
    R = min(8, C // 128)

    # phase 1: update chunks (the previous call's last chunk at index 0 when given, then chunks 0 .. nc-2)
    nu_tot = nc - 1 + int(has_prev)
    nu = max(nu_tot, 1)
    part = torch.empty(B, K, nu, R, S, 2, dtype=torch.float32, device=dev)
    AB = torch.zeros(B, K, nu, S, S + V, dtype=torch.float32, device=dev)
    if nc > 1:
        _chunk_ab(k, qk, v, seg, part, AB, K, S, V, C, R, nc - 1, int(has_prev), seg_off, has_seg)
    if has_prev:
        _chunk_ab(_blhs(prev_key), _blhs(prev_qkey), _blhs(prev_value), seg, part, AB, K, S, V, C, R, 1, 0, 0, has_seg)

    # phase 2: recurrence (fp32), memory resets, chunks whose memory is still empty
    MT = torch.empty(B, K, nc, V, S, dtype=torch.bfloat16, device=dev)
    mem_out = torch.empty(B, K, S, V, dtype=torch.bfloat16, device=dev)
    lnt_out = torch.empty(B, K, S, dtype=torch.float32, device=dev)
    dead = torch.empty(B, nc, dtype=torch.uint8, device=dev)
    # wide V: tcgen05 (few, wide CTAs); otherwise warp MMAs over more, narrower CTAs (measured on B200)
    if V >= 1024 and not _is_hopper(dev):
        rec_vt, rec_cls = 64, AwmRecurrenceUmma
    else:
        rec_vt = next((vt for vt in (64, 32, 16) if V % vt == 0 and B * K * (V // vt) >= 128), 16)
        rec_cls = AwmRecurrence
    A16 = None
    if rec_cls is AwmRecurrence or save:
        # bf16 copy of A: the warp-MMA recurrence reads it, and the backward uses it as an MN-major operand
        A16 = torch.empty(B, K, nu, S, S, dtype=torch.bfloat16, device=dev)
        f_cvt = _compiled(("a16", S, V), lambda: AwmAToBf16(S, V), lambda: (_fake(Float32, 32), _fake(BFloat16), Int32(0)))
        f_cvt(AB.reshape(-1), A16.reshape(-1), B * K * nu * S)
    if save:
        Mseq = torch.empty(B, K, nc, S, V, dtype=torch.bfloat16, device=dev)
        D = torch.empty(B, K, nu, S, V, dtype=torch.float32, device=dev)
        Rr = torch.empty(B, K, nu, S, dtype=torch.float32, device=dev)
        Mneg = torch.empty(B, K, nu, S, V, dtype=torch.bfloat16, device=dev)
    else:
        Mseq, D, Rr, Mneg = _dummy(torch.bfloat16, dev), _dummy(torch.float32, dev), _dummy(torch.float32, dev), _dummy(torch.bfloat16, dev)
    rec_key = ("rec", rec_cls.__name__, K, S, V, C, R, has_mem, has_prev, has_seg, rec_vt, save)
    f_rec = _compiled(rec_key, lambda: rec_cls(K, S, V, C, R, has_mem, has_prev, has_seg, rec_vt, save), lambda: (
        _fake(Float32), _fake(Float32), *([_fake(BFloat16)] if rec_cls is AwmRecurrence else []),
        _fake(BFloat16), _fake(Float32), _fake(Int64, 8), _fake(Int64, 8),
        _fake(_U8, 1), _fake(BFloat16), _fake(BFloat16), _fake(Float32), _fake(BFloat16), _fake(Float32),
        _fake(Float32), _fake(BFloat16), *([Int32(0)] * 5)))
    rec_args = [part.reshape(-1), AB.reshape(-1)]
    if rec_cls is AwmRecurrence:
        rec_args.append(A16.reshape(-1))
    f_rec(*rec_args, (memory if has_mem else mem_out).reshape(-1),
          (log_norm_term.float() if has_mem else lnt_out).reshape(-1), seg.reshape(-1), prev_cnt.reshape(-1),
          dead.reshape(-1), MT.reshape(-1), mem_out.reshape(-1), lnt_out.reshape(-1), Mseq.reshape(-1),
          D.reshape(-1), Rr.reshape(-1), Mneg.reshape(-1), B, nc, nu, seg_off, seg_stride)
    return dict(part=part, AB=AB, A16=A16, MT=MT, mem_out=mem_out, lnt_out=lnt_out, dead=dead, Mseq=Mseq, D=D,
                R=Rr, Mneg=Mneg, R_splits=R, nu=nu_tot, seg=seg, prev_cnt=prev_cnt, seg_off=seg_off, seg_stride=seg_stride)


def _out_gemm(q, MT, seg, prev_cnt, dead, seg_off, seg_stride, H, K, S, V, C, has_prev, has_seg, out_mask=None):
    """Phase 3 on q [B, L, H, S] (contiguous) and the stacked MT[(b, k, c), V, S] (bf16): out [B, L, H, V] =
    q_c M_c with masked rows zeroed, and the row mask [B, L] (written into ``out_mask`` when given)."""
    B, L = q.shape[0], q.shape[1]
    dev = q.device
    nc = L // C
    if out_mask is None:
        out_mask = torch.empty(B, L, dtype=torch.bool, device=dev)
    if _is_hopper(dev) and _AWM_SM90_CUBLAS:
        return _out_gemm_cublas(q, MT, seg, prev_cnt, dead, seg_off, H, K, S, V, C, has_prev, has_seg, out_mask)
    out = torch.empty(B, L, H, V, dtype=torch.bfloat16, device=dev)
    if _is_hopper(dev):
        sm90 = _sm90()
        f = _compiled(("out90", H, K, S, V, C, has_prev, has_seg), lambda: sm90.AwmOutGemmSm90(H, K, S, V, C, has_prev, has_seg),
                      lambda: (_fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Int64, 8), _fake(Int64, 8),
                               _fake(_U8, 1), _fake(_U8, 1), *([Int32(0)] * 4)))
        f(q.reshape(-1), MT.reshape(-1), out.reshape(-1), seg.reshape(-1), prev_cnt.reshape(-1), dead.reshape(-1),
          out_mask.view(torch.uint8).reshape(-1), B, L, seg_off, seg_stride)
        return out, out_mask
    out_vt, out_stages, resident = _out_tiling(S, V)
    out_key = ("out", H, K, S, V, C, has_prev, has_seg, out_vt, out_stages, resident)
    f_out = _compiled(out_key, lambda: AwmOutGemm(H, K, S, V, C, has_prev, has_seg, out_vt, out_stages, resident), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Int64, 8), _fake(Int64, 8), _fake(_U8, 1),
        _fake(_U8, 1), *([Int32(0)] * 5)))
    groups = (V // out_vt) * B * K * nc  # tiles sharing one resident memory tile
    group_tiles = (H // K) * (C // 128)
    num_tiles = groups * group_tiles
    # whole groups per CTA range (a memory tile reload stalls the MMAs): CTAs = groups x splits of a group
    splits = max(1, min(group_tiles, _sm_count(dev) // groups)) if groups <= _sm_count(dev) else 0
    out_ctas = groups * splits if splits else _sm_count(dev)
    f_out(q.reshape(-1), MT.reshape(-1), out.reshape(-1), seg.reshape(-1), prev_cnt.reshape(-1), dead.reshape(-1),
          out_mask.view(torch.uint8).reshape(-1), B, L, seg_off, seg_stride, min(out_ctas, num_tiles))
    return out, out_mask


WORKSPACE_KEYS = ("part", "AB", "A16", "Mseq", "D", "R", "Mneg", "dead", "MT")  # saved for the backward (awm_bwd_prepare)


def adaptive_working_memory_fwd(
    query, qkey, key, value, chunk_size, memory, log_norm_term, prev_qkey, prev_key, prev_value,
    segment_idx=None, prev_segment_count=None, ortho=False, eps=1e-6, return_workspace=False,
):
    """Drop-in for xllm.modules.fused_ops.adaptive_working_memory._adaptive_working_memory_fwd (ortho=False):
    query [B, H, L, S], qkey / key [B, K, L, S], value [B, K, L, V] (views of [B, L, *, D] storage are read
    without copies). Returns out [B, H, L, V] (a view of [B, L, H, V] storage), out_mask [B, L],
    memory [B, K, S, V] (dtype of query) and log_norm_term [B, K, S] (fp32). All masks are derived from
    segment_idx inside the kernels. With ``return_workspace`` (training case only), also the tuple of the
    WORKSPACE_KEYS tensors that let the backward skip the recomputation of phases 1 and 2."""
    assert not ortho, "orthogonal update is not supported by the CuTe kernels"
    B, H, L, S = query.shape
    K, V = value.shape[1], value.shape[3]
    C = chunk_size
    assert L % C == 0 and C % 128 == 0 and query.dtype == torch.bfloat16
    has_prev = prev_key is not None
    has_mem = memory is not None
    has_seg = segment_idx is not None
    if not has_mem:
        assert log_norm_term is None and not has_prev
    w = _phases12(qkey, key, value, segment_idx, prev_segment_count, memory, log_norm_term, prev_qkey, prev_key,
                  prev_value, C, save=return_workspace)
    out, out_mask = _out_gemm(_blhs(query), w["MT"], w["seg"], w["prev_cnt"], w["dead"], w["seg_off"], w["seg_stride"],
                              H, K, S, V, C, has_prev, has_seg)
    if return_workspace:
        assert not has_mem
        return out.transpose(1, 2), out_mask, w["mem_out"], w["lnt_out"], tuple(w[k] for k in WORKSPACE_KEYS)
    return out.transpose(1, 2), out_mask, w["mem_out"], w["lnt_out"]
