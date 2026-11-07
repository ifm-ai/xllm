"""Adaptive working memory (AWM) kernels for Hopper (SM90): wgmma versions of the tcgen05 GEMM phases of
adaptive_working_memory.py / adaptive_working_memory_bwd.py (same contracts, same masks).

  AwmOutGemmSm90   phase 3: out = q_c M_c (masked rows 0) on [B, L, H, *] layouts; also dq = dOut M_c^T in the
                   backward (roles of S and V swapped)

TMA loads by warp 0, two consumer warpgroups of 64 rows each (wgmma, fp32 accumulators in registers), masked
bf16 epilogue through shared memory and TMA stores.
"""

import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

try:
    from .adaptive_working_memory import _seg_anchor
except ImportError:  # loaded from its file (tests)
    import importlib.util as _ilu
    import os as _os
    _spec = _ilu.spec_from_file_location("cute_adaptive_working_memory_fwd_sm90",
                                         _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "adaptive_working_memory.py"))
    _fwd = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_fwd)
    _seg_anchor = _fwd._seg_anchor

ROW_MAJOR = utils.LayoutEnum.ROW_MAJOR
COL_MAJOR = utils.LayoutEnum.COL_MAJOR
LOG2E = 1.4426950408889634


class AwmChunkGemmSm90:
    """Split-K GEMMs over the time rows of a chunk with both operands MN-major, fp32 results reduce-added:

      mode 'ab'  (AwmChunkGemm):  AB[b, k, j_off + j, s, :] += sum_t P[s, t] [qk | v][t, :] over the rows t of update
                 chunk j, P[s, t] = exp2(k[b, t, k, s] log2e - lse2[b, k, j, s]);  A source k [B, L, K, S]
      mode 'dmo' (AwmDmoGemm):    dMo[b, k, c, s, :] += sum_{h in group of k} sum_t q[b, t, h, s] dOut[b, t, h, :];
                 A source q [B, L, H, S], B source dOut [B, L, H, V]

    Only the rows t in [lo[b, j], hi[b, j]) of a chunk count (the rows of the segment of the chunk's last row for
    'ab', the rows kept by the output mask for 'dmo': one contiguous range in both cases); time tiles outside are
    skipped and the consumer warps rewrite the A stage in shared memory before the wgmmas (P for 'ab', masked rows
    zeroed). One CTA per (b, k, chunk, 128 rows of s, NT columns, split of the time range); warp 0 issues the TMA
    loads, two warpgroups of 64 rows run the wgmmas, the epilogue adds the fp32 accumulators into the output."""

    def __init__(self, mode: str, K: int, S: int, V: int, C: int, n_rep: int, splits: int):
        assert mode in ("ab", "dmo")
        self.mode, self.K, self.S, self.V, self.C, self.n_rep, self.splits = mode, K, S, V, C, n_rep, splits
        self.TM, self.KB = 128, 64
        self.N = S + V if mode == "ab" else V
        self.NT = min(256, S if mode == "ab" else V)
        assert S % self.TM == 0 and C % self.KB == 0 and self.N % self.NT == 0 and V % self.NT == 0
        self.m_blocks = S // self.TM
        self.n_blocks = self.N // self.NT
        self.chunk_tiles = C // self.KB
        stage_bytes = (self.TM + self.NT) * self.KB * 2
        self.stages = max(2, min(6, (200 * 1024) // stage_bytes))
        self.threads = 256
        self.tile_mnk = (self.TM, self.NT, self.KB)

    @cute.jit
    def __call__(self, tA: cute.Tensor, tB1: cute.Tensor, tB2: cute.Tensor, tLo: cute.Tensor, tHi: cute.Tensor,
                 tLse2: cute.Tensor, tOut: cute.Tensor, B: Int32, L: Int32, nch: Int32, ch_off: Int32, out_off: Int32,
                 nch_out: Int32, stream: cuda.CUstream):
        """nch chunks of rows starting at chunk ch_off of the sources, written at chunk out_off + j of tOut
        [B, K, nch_out, S, N]; tLo / tHi [B, nch] (int32) and tLse2 [B, K, nch, S] ('ab') are indexed by j."""
        K, S, V, NT = self.K, self.S, self.V, self.NT
        Ha = K * self.n_rep
        # A: (M = S, K = time, head, B), s contiguous; B sources: (N, time, head, B), n contiguous
        mA = cute.make_tensor(tA.iterator, cute.make_layout((S, L, Ha, B), stride=(1, Ha * S, S, L * Ha * S)))
        nb1 = S if const_expr(self.mode == "ab") else V
        mB1 = cute.make_tensor(tB1.iterator, cute.make_layout((nb1, L, Ha, B), stride=(1, Ha * nb1, nb1, L * Ha * nb1)))
        mB2 = cute.make_tensor(tB2.iterator, cute.make_layout((V, L, Ha, B), stride=(1, Ha * V, V, L * Ha * V)))
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            BFloat16, BFloat16, warpgroup.OperandMajorMode.MN, warpgroup.OperandMajorMode.MN, Float32, (2, 1, 1),
            tiler_mn=(64, NT))
        a_smem_layout = sm90_utils.make_smem_layout_a(COL_MAJOR, self.tile_mnk, BFloat16, self.stages)
        b_smem_layout = sm90_utils.make_smem_layout_b(COL_MAJOR, self.tile_mnk, BFloat16, self.stages)
        g2s = cpasync.CopyBulkTensorTileG2SOp()
        tma_a, mA_tma = cpasync.make_tiled_tma_atom(g2s, mA, cute.slice_(a_smem_layout, (None, None, 0)), (self.TM, self.KB))
        b_one = cute.slice_(b_smem_layout, (None, None, 0))
        tma_b1, mB1_tma = cpasync.make_tiled_tma_atom(g2s, mB1, b_one, (NT, self.KB))
        tma_b2, mB2_tma = cpasync.make_tiled_tma_atom(g2s, mB2, b_one, (NT, self.KB))

        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            sA: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(a_smem_layout)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(b_smem_layout)], 1024]

        self.shared_storage = SharedStorage
        mLo = cute.make_tensor(tLo.iterator, cute.make_layout((nch, B), stride=(1, nch)))
        mHi = cute.make_tensor(tHi.iterator, cute.make_layout((nch, B), stride=(1, nch)))
        mLse2 = cute.make_tensor(tLse2.iterator, cute.make_layout((S, nch, K, B), stride=(1, S, nch * S, K * nch * S)))
        grid = B * K * nch * self.m_blocks * self.n_blocks * self.splits
        self.kernel(tiled_mma, tma_a, mA_tma, tma_b1, mB1_tma, tma_b2, mB2_tma, mLo, mHi, mLse2, tOut.iterator,
                    a_smem_layout, b_smem_layout, B, nch, ch_off, out_off, nch_out).launch(
            grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, tma_a: cute.CopyAtom, mA: cute.Tensor, tma_b1: cute.CopyAtom,
               mB1: cute.Tensor, tma_b2: cute.CopyAtom, mB2: cute.Tensor, mLo: cute.Tensor, mHi: cute.Tensor,
               mLse2: cute.Tensor, pOut: cute.Pointer, a_smem_layout: cute.ComposedLayout,
               b_smem_layout: cute.ComposedLayout, B: Int32, nch: Int32, ch_off: Int32, out_off: Int32, nch_out: Int32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if warp == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b1)
            cpasync.prefetch_descriptor(tma_b2)
        # block -> (split, n block, m block, chunk, kv head, batch)
        sp = bidx % self.splits
        t1 = bidx // self.splits
        nb = t1 % self.n_blocks
        t2 = t1 // self.n_blocks
        mb = t2 % self.m_blocks
        t3 = t2 // self.m_blocks
        j = t3 % nch
        t4 = t3 // nch
        k = t4 % self.K
        b = t4 // self.K
        lo = mLo[j, b]
        hi = mHi[j, b]
        t_first = lo // self.KB
        n_t = cutlass.select_(hi > lo, (hi + self.KB - 1) // self.KB - t_first, Int32(0))
        kt_total = self.n_rep * n_t
        kt0 = sp * kt_total // self.splits
        n_kt = (sp + 1) * kt_total // self.splits - kt0

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        ab_bytes = (cute.size_in_bytes(BFloat16, cute.slice_(a_smem_layout, (None, None, 0)))
                    + cute.size_in_bytes(BFloat16, cute.slice_(b_smem_layout, (None, None, 0))))
        ab_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.ab_mbar.data_ptr(), num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads // 32),
            tx_count=ab_bytes, cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)), defer_sync=True)
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)

        wg = cute.arch.make_warp_uniform(tidx // 128)
        wg_mma = tiled_mma.get_slice(cute.make_layout(2, stride=128)(wg))
        thr_mma = tiled_mma.get_slice(tidx)
        tCrA = tiled_mma.make_fragment_A(wg_mma.partition_A(sA))
        tCrB = tiled_mma.make_fragment_B(wg_mma.partition_B(sB))
        acc = cute.make_rmem_tensor(wg_mma.partition_shape_C((self.TM, self.NT)), Float32)
        pipeline_init_wait(cluster_shape_mn=(1, 1))

        if n_kt > 0:  # an empty range (every row masked) adds nothing
            # B source and its column block: [qk | v] for 'ab' (a block never straddles the two), dOut for 'dmo'
            use_b1 = Int32(1)
            ncol = nb
            if const_expr(self.mode == "ab"):
                use_b1 = cutlass.select_(nb * self.NT < self.S, Int32(1), Int32(0))
                ncol = cutlass.select_(use_b1 != 0, nb, nb - self.S // self.NT)
            prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
            if warp == 0:
                for _ in cutlass.range(min(self.stages, n_kt), unroll=1):
                    self._load(ab_pipe, prod, tma_a, mA, tma_b1, mB1, tma_b2, mB2, sA, sB, use_b1, ncol, mb, k, b, j,
                               ch_off, t_first, n_t, kt0)
                    prod.advance()
            # this thread's row s of the A tile (fixed through the rewrite passes) and its lse
            srow = tidx % self.TM
            lse2 = Float32(0.0)
            if const_expr(self.mode == "ab"):
                lse2 = mLse2[mb * self.TM + srow, j, k, b]
            cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            rel = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
            for i in cutlass.range(n_kt, unroll=1):
                ab_pipe.consumer_wait(cons)
                trow0 = (t_first + (kt0 + i) % n_t) * self.KB  # first row (in the chunk) of this time tile
                if const_expr(self.mode == "ab") or trow0 < lo or trow0 + self.KB > hi:
                    self._rewrite_a(sA, cons.index, srow, lse2, trow0, lo, hi, tidx)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    pipeline.sync(barrier_id=2)
                warpgroup.fence()
                for kb in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                    cute.gemm(tiled_mma, acc, tCrA[(None, None, kb, cons.index)], tCrB[(None, None, kb, cons.index)], acc)
                    tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                warpgroup.commit_group()
                cons.advance()
                if i > 0:
                    warpgroup.wait_group(1)
                    ab_pipe.consumer_release(rel)
                    rel.advance()
                    if warp == 0 and prod.count < n_kt:
                        self._load(ab_pipe, prod, tma_a, mA, tma_b1, mB1, tma_b2, mB2, sA, sB, use_b1, ncol, mb, k, b,
                                   j, ch_off, t_first, n_t, kt0)
                        prod.advance()
            warpgroup.wait_group(0)
            # epilogue: add the fp32 accumulators into the output
            N = self.N
            base = (((Int64(b) * self.K + k) * nch_out + out_off + j) * self.S + mb * self.TM) * N + nb * self.NT
            tCcC = thr_mma.partition_C(cute.make_identity_tensor((self.TM, self.NT)))
            for e in cutlass.range_constexpr(cute.size(acc)):
                r, cc = tCcC[e]
                cute.arch.atomic_add(pOut + (base + r * N + cc), acc[e], sem="relaxed", scope="gpu")

    @cute.jit
    def _load(self, ab_pipe, prod, tma_a: cute.CopyAtom, mA: cute.Tensor, tma_b1: cute.CopyAtom, mB1: cute.Tensor,
              tma_b2: cute.CopyAtom, mB2: cute.Tensor, sA: cute.Tensor, sB: cute.Tensor, use_b1: Int32, ncol: Int32,
              mb: Int32, k: Int32, b: Int32, j: Int32, ch_off: Int32, t_first: Int32, n_t: Int32, kt0: Int32):
        """Warp 0: TMA loads of K step kt0 + prod.count (head of the group, time tile) into stage prod.index."""
        kt = kt0 + prod.count
        h = k * self.n_rep + kt // n_t
        ttile = (ch_off + j) * self.chunk_tiles + t_first + kt % n_t  # time tile index in the source rows
        ab_pipe.producer_acquire(prod)
        bar = ab_pipe.producer_get_barrier(prod)
        gA = cute.local_tile(mA[None, None, h, b], (self.TM, self.KB), (mb, ttile))
        tAsA, tAgA = cpasync.tma_partition(tma_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
        cute.copy(tma_a, tAgA, tAsA[(None, prod.index)], tma_bar_ptr=bar)
        if use_b1 != 0:
            gB = cute.local_tile(mB1[None, None, h, b], (self.NT, self.KB), (ncol, ttile))
            tBsB, tBgB = cpasync.tma_partition(tma_b1, 0, cute.make_layout(1), cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))
            cute.copy(tma_b1, tBgB, tBsB[(None, prod.index)], tma_bar_ptr=bar)
        else:
            gB = cute.local_tile(mB2[None, None, h, b], (self.NT, self.KB), (ncol, ttile))
            tBsB, tBgB = cpasync.tma_partition(tma_b2, 0, cute.make_layout(1), cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))
            cute.copy(tma_b2, tBgB, tBsB[(None, prod.index)], tma_bar_ptr=bar)
        ab_pipe.producer_commit(prod)

    @cute.jit
    def _rewrite_a(self, sA: cute.Tensor, stage: Int32, srow: Int32, lse2: Float32, trow0: Int32, lo: Int32, hi: Int32,
                   tidx: Int32):
        """Consumer threads: rewrite the A tile of a stage in place, row srow, half of the time columns per thread:
        P = exp2(a log2e - lse2) ('ab') or a ('dmo'), 0 outside the kept rows [lo, hi) of the chunk."""
        half = tidx // self.TM
        for i in cutlass.range_constexpr(self.KB // 2):
            tt = half * (self.KB // 2) + i
            trow = trow0 + tt
            keep = (trow >= lo) & (trow < hi)
            a = BFloat16(sA[(srow, tt, stage)]).to(Float32)
            if const_expr(self.mode == "ab"):
                a = cute.math.exp2(a * LOG2E - lse2, fastmath=True)
            sA[(srow, tt, stage)] = Float32(cutlass.select_(keep, a, Float32(0.0))).to(BFloat16)


class AwmOutGemmSm90:
    """Phase 3: out[b, l, h, :] = q[b, l, h, :] @ M_c(b, h // n_rep) for the chunk c of row l, zero on masked rows.

    q is read as [B, L, H, S] and out written as [B, L, H, V] (no transposes); the memory states come transposed
    and stacked as MT[(b * K + k) * nc + c, v, s] (K-major B operand). One CTA per tile of 128 rows x VT columns
    with the whole S as K, tiles numbered as in AwmOutGemm._decode (v tile inside a memory group, so the CTAs of
    the v tiles of the same rows run side by side and share each q tile through L2)."""

    def __init__(self, H: int, K: int, S: int, V: int, C: int, has_prev: bool, has_seg: bool, VT: int = 256):
        self.H, self.K, self.S, self.V, self.C = H, K, S, V, C
        self.has_prev, self.has_seg = has_prev, has_seg
        self.n_rep = H // K
        self.TM = 128
        self.VT = min(V, VT)
        self.KB = 64  # one 128-byte swizzle atom of bf16
        assert H % K == 0 and V % self.VT == 0 and C % self.TM == 0 and S % self.KB == 0
        self.n_vt = V // self.VT
        self.row_tiles = C // self.TM
        self.k_tiles = S // self.KB
        stage_bytes = (self.TM + self.VT) * self.KB * 2
        self.stages = max(2, min(self.k_tiles, (200 * 1024) // stage_bytes))  # ring of K blocks
        if os.environ.get("AWM_OUT90_STAGES"):  # tuning override
            self.stages = int(os.environ["AWM_OUT90_STAGES"])
        self.epi_stages = 2
        self.threads = 256  # two consumer warpgroups; warp 0 also issues the TMA loads
        self.tile_mnk = (self.TM, self.VT, self.KB)

    @cute.jit
    def __call__(self, tQ: cute.Tensor, tMT: cute.Tensor, tOut: cute.Tensor, tSeg: cute.Tensor, tPrevCount: cute.Tensor,
                 tDead: cute.Tensor, tOutMask: cute.Tensor, B: Int32, L: Int32, seg_off: Int32, seg_stride: Int32,
                 stream: cuda.CUstream):
        H, K, S, V = self.H, self.K, self.S, self.V
        nc = L // self.C
        # q [B, L, H, S] as (M = L, K = S, H, B); MT [(b, k, c), V, S] as (N = V, K = S, (c, k, b)); out as (L, V, H, B)
        mQ = cute.make_tensor(tQ.iterator, cute.make_layout((L, S, H, B), stride=(H * S, 1, S, L * H * S)))
        mMT = cute.make_tensor(tMT.iterator, cute.make_layout((V, S, B * K * nc), stride=(S, 1, V * S)))
        mOut = cute.make_tensor(tOut.iterator, cute.make_layout((L, V, H, B), stride=(H * V, 1, V, L * H * V)))
        mOutMask = cute.make_tensor(tOutMask.iterator, cute.make_layout((L, B), stride=(1, L)))
        mDead = cute.make_tensor(tDead.iterator, cute.make_layout((nc, B), stride=(1, nc)))

        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            BFloat16, BFloat16, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K, Float32, (2, 1, 1),
            tiler_mn=(64, self.VT))
        a_smem_layout = sm90_utils.make_smem_layout_a(ROW_MAJOR, self.tile_mnk, BFloat16, self.stages)
        b_smem_layout = sm90_utils.make_smem_layout_b(ROW_MAJOR, self.tile_mnk, BFloat16, self.stages)
        epi_tile = sm90_utils.compute_tile_shape_or_override(self.tile_mnk, BFloat16, is_cooperative=True)
        epi_smem_layout = sm90_utils.make_smem_layout_epi(BFloat16, ROW_MAJOR, epi_tile, self.epi_stages)
        tma_a, mQ_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mQ, cute.slice_(a_smem_layout, (None, None, 0)), (self.TM, self.KB))
        tma_b, mMT_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mMT, cute.slice_(b_smem_layout, (None, None, 0)), (self.VT, self.KB))
        tma_c, mOut_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mOut, cute.slice_(epi_smem_layout, (None, None, 0)), epi_tile)

        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            keep: cute.struct.MemRange[Float32, self.TM]
            sA: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(a_smem_layout)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(b_smem_layout)], 1024]
            sC: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(epi_smem_layout)], 1024]

        self.shared_storage = SharedStorage
        num_tiles = self.n_vt * B * K * nc * self.n_rep * self.row_tiles
        self.kernel(tiled_mma, tma_a, mQ_tma, tma_b, mMT_tma, tma_c, mOut_tma, tSeg.iterator, tPrevCount.iterator,
                    mDead, mOutMask, a_smem_layout, b_smem_layout, epi_smem_layout, epi_tile, B, nc, seg_off,
                    seg_stride).launch(grid=[num_tiles, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, tma_a: cute.CopyAtom, mQ: cute.Tensor, tma_b: cute.CopyAtom,
               mMT: cute.Tensor, tma_c: cute.CopyAtom, mOut: cute.Tensor, pSeg: cute.Pointer, pPrevCount: cute.Pointer,
               mDead: cute.Tensor, mOutMask: cute.Tensor, a_smem_layout: cute.ComposedLayout,
               b_smem_layout: cute.ComposedLayout, epi_smem_layout: cute.ComposedLayout, epi_tile: cute.Tile,
               B: Int32, nc: Int32, seg_off: Int32, seg_stride: Int32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        t, _, _ = cute.arch.block_idx()
        if warp == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b)
            cpasync.prefetch_descriptor(tma_c)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        ab_bytes = (cute.size_in_bytes(BFloat16, cute.slice_(a_smem_layout, (None, None, 0)))
                    + cute.size_in_bytes(BFloat16, cute.slice_(b_smem_layout, (None, None, 0))))
        ab_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.ab_mbar.data_ptr(), num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads // 32),
            tx_count=ab_bytes, cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)), defer_sync=True)
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sC = storage.sC.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
        sKeep = storage.keep.get_tensor(cute.make_layout(self.TM))

        rt, hr, c, k, b, vt = self._decode(t, B, nc)
        row_block = c * self.row_tiles + rt
        h = k * self.n_rep + hr
        mem_idx = (b * self.K + k) * nc + c
        # (TM, KB, k_tiles) / (VT, KB, k_tiles) / (TM, VT)
        gA = cute.local_tile(mQ[None, None, h, b], (self.TM, self.KB), (row_block, None))
        gB = cute.local_tile(mMT[None, None, mem_idx], (self.VT, self.KB), (vt, None))
        gC = cute.local_tile(mOut[None, None, h, b], (self.TM, self.VT), (row_block, vt))
        tAsA, tAgA = cpasync.tma_partition(tma_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
        tBsB, tBgB = cpasync.tma_partition(tma_b, 0, cute.make_layout(1), cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))

        wg = cute.arch.make_warp_uniform(tidx // 128)
        wg_mma = tiled_mma.get_slice(cute.make_layout(2, stride=128)(wg))
        thr_mma = tiled_mma.get_slice(tidx)
        tCrA = tiled_mma.make_fragment_A(wg_mma.partition_A(sA))
        tCrB = tiled_mma.make_fragment_B(wg_mma.partition_B(sB))
        acc = cute.make_rmem_tensor(wg_mma.partition_shape_C((self.TM, self.VT)), Float32)
        pipeline_init_wait(cluster_shape_mn=(1, 1))

        prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
        if warp == 0:
            for _ in cutlass.range(min(self.stages, self.k_tiles), unroll=1):
                self._load_ab(ab_pipe, prod, tma_a, tAgA, tAsA, tma_b, tBgB, tBsB)
                prod.advance()

        # row mask of the tile (the loads are in flight): the chunk's memory is empty, or the row is not in the
        # segment that the memory ends in (that of chunk c - 2)
        if tidx < self.TM:
            row = row_block * self.TM + tidx
            dead = mDead[c, b]
            seg_row = Int64(0)
            anchor = Int64(0)
            if const_expr(self.has_seg):
                seg_row = cute.make_tensor(pSeg + (b * seg_stride + seg_off + row), cute.make_layout(1))[0]
                if c >= 2 or const_expr(self.has_prev):
                    anchor = _seg_anchor(pSeg, pPrevCount, b, c - 2, seg_off, seg_stride, self.C)
                else:
                    anchor = seg_row
            masked = (dead != 0) | (seg_row != anchor)
            sKeep[tidx] = cutlass.select_(masked, Float32(0.0), Float32(1.0))
            if hr == 0 and vt == 0 and k == 0:
                mOutMask[row, b] = cutlass.select_(masked, Uint8(1), Uint8(0))

        # mainloop: the wgmmas of K block kt overlap the wait for block kt + 1; a stage is released once the
        # wgmmas that read it have completed, and warp 0 refills it with the next K block
        cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
        rel = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
        tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
        for kt in cutlass.range(self.k_tiles, unroll=1):
            ab_pipe.consumer_wait(cons)
            warpgroup.fence()
            for kb in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                cute.gemm(tiled_mma, acc, tCrA[(None, None, kb, cons.index)], tCrB[(None, None, kb, cons.index)], acc)
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            cons.advance()
            if kt > 0:
                warpgroup.wait_group(1)  # the wgmmas of block kt - 1 are done: free its stage
                ab_pipe.consumer_release(rel)
                rel.advance()
                if warp == 0 and prod.count < self.k_tiles:
                    self._load_ab(ab_pipe, prod, tma_a, tAgA, tAsA, tma_b, tBgB, tBsB)
                    prod.advance()
        warpgroup.wait_group(0)
        cute.arch.sync_threads()  # sKeep written; every stage consumed

        # epilogue: mask rows, bf16, registers -> smem (stmatrix) -> TMA store per epilogue sub-tile
        tCcC = thr_mma.partition_C(cute.make_identity_tensor((self.TM, self.VT)))
        for i in cutlass.range_constexpr(cute.size(acc)):
            acc[i] = acc[i] * sKeep[tCcC[i][0]]
        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(ROW_MAJOR, elem_ty_d=BFloat16, elem_ty_acc=Float32)
        copy_atom_c = cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(False, 4), BFloat16)
        tiled_r2s = cute.make_tiled_copy_S(copy_atom_r2s, cute.make_tiled_copy_C_atom(copy_atom_c, tiled_mma))
        thr_r2s = tiled_r2s.get_slice(tidx)
        tRS_sD = thr_r2s.partition_D(sC)
        tRS_rAcc = tiled_r2s.retile(acc)
        rD_shape = cute.shape(thr_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, Float32)
        n_rd = cute.size(tRS_rD)
        gC_epi = cute.zipped_divide(gC, epi_tile)
        bSG_sD, bSG_gD = cpasync.tma_partition(tma_c, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), gC_epi)
        n_epi = cute.size(gC_epi, mode=[1])
        epi_shape = gC_epi.shape[1]
        epi_layout = cute.make_layout(epi_shape, stride=(epi_shape[1], 1))
        c_pipe = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stages, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads))
        for e in cutlass.range_constexpr(n_epi):
            for j in cutlass.range_constexpr(n_rd):
                tRS_rD[j] = tRS_rAcc[e * n_rd + j]
            tRS_rD_out = cute.make_rmem_tensor_like(tRS_rD_layout, BFloat16)
            tRS_rD_out.store(tRS_rD.load().to(BFloat16))
            buf = e % self.epi_stages
            cute.copy(tiled_r2s, tRS_rD_out, tRS_sD[(None, None, None, buf)])
            cute.arch.fence_proxy("async.shared", space="cta")
            pipeline.sync(barrier_id=1)
            if warp == 0:
                cute.copy(tma_c, bSG_sD[(None, buf)], bSG_gD[(None, epi_layout.get_hier_coord(e))])
                c_pipe.producer_commit()
                c_pipe.producer_acquire()
            pipeline.sync(barrier_id=1)
        if warp == 0:
            c_pipe.producer_tail()

    @cute.jit
    def _load_ab(self, ab_pipe, prod, tma_a: cute.CopyAtom, tAgA: cute.Tensor, tAsA: cute.Tensor, tma_b: cute.CopyAtom,
                 tBgB: cute.Tensor, tBsB: cute.Tensor):
        """Warp 0: TMA loads of the A and B tiles of K block prod.count into stage prod.index."""
        ab_pipe.producer_acquire(prod)
        cute.copy(tma_a, tAgA[(None, prod.count)], tAsA[(None, prod.index)], tma_bar_ptr=ab_pipe.producer_get_barrier(prod))
        cute.copy(tma_b, tBgB[(None, prod.count)], tBsB[(None, prod.index)], tma_bar_ptr=ab_pipe.producer_get_barrier(prod))
        ab_pipe.producer_commit(prod)

    @cute.jit
    def _decode(self, t: Int32, B: Int32, nc: Int32):
        """Tile index -> (row block in chunk, head in group, chunk, kv head, batch, v tile), as AwmOutGemm._decode."""
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




class AwmChunkBwdSm90:
    """Phase 1' (AwmChunkBwd) per update chunk j and 128 time rows of it, the rows before the segment of the chunk's
    last row ([lo, C) kept) contributing nothing:

      dP^T = qk dA^T + v dB^T              (128 x S; wgmma, K = S + V streamed in 64-wide feature blocks)
      P^T  = exp2(k log2e - lse2)          (k tile by TMA into shared memory, ldmatrix into the accumulator layout)
      dk   = P^T (dP^T + dlse - delta)
      dqk  = P^T dA,  dv = P^T dB          (P^T bf16 written over the k tile with stmatrix: the A operand; dA / dB
                                            streamed MN-major through the ring memory of the first GEMM)

    k, qk, v, dk, dqk, dv are [B, L, K, *]; dA16 / dB16 [B, K, nu, S, S or V] (bf16), lse2 and c2 = dlse - delta
    [B, K, nu, S] (fp32). One CTA per (b, k, j, time block), 256 threads (two warpgroups of 64 rows), warp 0 issues
    the TMA loads and stores; results leave through shared memory and TMA stores. The rows of the last chunk (no
    update) are zeroed by the caller."""

    def __init__(self, K: int, S: int, V: int, C: int, dv_accumulate: bool = False):
        self.K, self.S, self.V, self.C = K, S, V, C
        self.dv_accumulate = dv_accumulate  # dv tiles reduce-added (TMA) into dv, which holds another gradient
        self.TM, self.KB = 128, 64
        self.N = min(S, 256)  # N of every tile (dP^T, dqk, dv)
        assert S % self.KB == 0 and S <= 256 and V % self.N == 0 and C % self.TM == 0 and min(V, 256) == self.N
        self.f_blocks = (S + V) // self.KB
        self.s_blocks = S // self.KB
        self.n_vt = V // self.N
        self.row_blocks = C // self.TM
        self.stages = 2
        self.epi_stages = 2
        self.threads = 256

    @cute.jit
    def __call__(self, tK: cute.Tensor, tQK: cute.Tensor, tV: cute.Tensor, tDA: cute.Tensor, tDB: cute.Tensor,
                 tLse2: cute.Tensor, tC2: cute.Tensor, tLo: cute.Tensor, tDK: cute.Tensor, tDQK: cute.Tensor,
                 tDV: cute.Tensor, B: Int32, L: Int32, nu: Int32, stream: cuda.CUstream):
        K, S, V, N = self.K, self.S, self.V, self.N
        rows = lambda t, F: cute.make_tensor(t.iterator, cute.make_layout((L, F, K, B), stride=(K * F, 1, F, L * K * F)))
        mK, mQK, mV, mDK, mDQK, mDV = rows(tK, S), rows(tQK, S), rows(tV, V), rows(tDK, S), rows(tDQK, S), rows(tDV, V)
        # dA / dB as the K-major B operand of dP^T (N = s, K = feature) and as the MN-major B operand of dqk / dv
        # (N = feature, K = s): the same [S, F] storage per (b, k, j)
        nb = nu * K * B
        mDA_k = cute.make_tensor(tDA.iterator, cute.make_layout((S, S, nb), stride=(S, 1, S * S)))
        mDB_k = cute.make_tensor(tDB.iterator, cute.make_layout((S, V, nb), stride=(V, 1, S * V)))
        mDA_mn = cute.make_tensor(tDA.iterator, cute.make_layout((S, S, nb), stride=(1, S, S * S)))
        mDB_mn = cute.make_tensor(tDB.iterator, cute.make_layout((V, S, nb), stride=(1, V, S * V)))

        mma1 = sm90_utils.make_trivial_tiled_mma(BFloat16, BFloat16, warpgroup.OperandMajorMode.K,
                                                 warpgroup.OperandMajorMode.K, Float32, (2, 1, 1), tiler_mn=(64, N))
        mma2 = sm90_utils.make_trivial_tiled_mma(BFloat16, BFloat16, warpgroup.OperandMajorMode.K,
                                                 warpgroup.OperandMajorMode.MN, Float32, (2, 1, 1), tiler_mn=(64, N))
        t1 = (self.TM, N, self.KB)
        a1_layout = sm90_utils.make_smem_layout_a(ROW_MAJOR, t1, BFloat16, self.stages)
        b1_layout = sm90_utils.make_smem_layout_b(ROW_MAJOR, t1, BFloat16, self.stages)
        b2_layout = sm90_utils.make_smem_layout_b(COL_MAJOR, t1, BFloat16, self.stages)
        p_layout = sm90_utils.make_smem_layout_a(ROW_MAJOR, (self.TM, N, S), BFloat16, 1)
        epi_tile = sm90_utils.compute_tile_shape_or_override((self.TM, N, self.KB), BFloat16, is_cooperative=True)
        epi_layout = sm90_utils.make_smem_layout_epi(BFloat16, ROW_MAJOR, epi_tile, self.epi_stages)
        g2s, s2g = cpasync.CopyBulkTensorTileG2SOp(), cpasync.CopyBulkTensorTileS2GOp()
        a1_one, b1_one = cute.slice_(a1_layout, (None, None, 0)), cute.slice_(b1_layout, (None, None, 0))
        b2_one, epi_one = cute.slice_(b2_layout, (None, None, 0)), cute.slice_(epi_layout, (None, None, 0))
        tma_qk, mQK_t = cpasync.make_tiled_tma_atom(g2s, mQK, a1_one, (self.TM, self.KB))
        tma_v, mV_t = cpasync.make_tiled_tma_atom(g2s, mV, a1_one, (self.TM, self.KB))
        tma_da, mDA_t = cpasync.make_tiled_tma_atom(g2s, mDA_k, b1_one, (N, self.KB))
        tma_db, mDB_t = cpasync.make_tiled_tma_atom(g2s, mDB_k, b1_one, (N, self.KB))
        tma_das, mDAs_t = cpasync.make_tiled_tma_atom(g2s, mDA_mn, b2_one, (N, self.KB))
        tma_dbv, mDBv_t = cpasync.make_tiled_tma_atom(g2s, mDB_mn, b2_one, (N, self.KB))
        tma_k, mK_t = cpasync.make_tiled_tma_atom(g2s, mK, cute.slice_(p_layout, (None, None, 0)), (self.TM, S))
        tma_dk, mDK_t = cpasync.make_tiled_tma_atom(s2g, mDK, epi_one, epi_tile)
        tma_dqk, mDQK_t = cpasync.make_tiled_tma_atom(s2g, mDQK, epi_one, epi_tile)
        dv_op = cpasync.CopyReduceBulkTensorTileS2GOp() if cutlass.const_expr(self.dv_accumulate) else s2g
        tma_dv, mDV_t = cpasync.make_tiled_tma_atom(dv_op, mDV, epi_one, epi_tile)
        assert cute.cosize(b2_layout) == cute.cosize(b1_layout) and cute.cosize(epi_layout) <= cute.cosize(a1_layout)

        @cute.struct
        class SharedStorage:
            ring_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            ring2_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            k_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            vec: cute.struct.MemRange[Float32, 2 * S]
            sA1: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(a1_layout)], 1024]
            sB1: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(b1_layout)], 1024]
            sP: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(p_layout)], 1024]

        self.shared_storage = SharedStorage
        vec = cute.make_layout((S, nu, K, B), stride=(1, S, nu * S, K * nu * S))
        mLse2 = cute.make_tensor(tLse2.iterator, vec)
        mC2 = cute.make_tensor(tC2.iterator, vec)
        mLo = cute.make_tensor(tLo.iterator, cute.make_layout((nu, B), stride=(1, nu)))
        grid = B * K * nu * self.row_blocks
        self.kernel(mma1, mma2, tma_qk, mQK_t, tma_v, mV_t, tma_da, mDA_t, tma_db, mDB_t, tma_das, mDAs_t, tma_dbv,
                    mDBv_t, tma_k, mK_t, tma_dk, mDK_t, tma_dqk, mDQK_t, tma_dv, mDV_t, mLse2, mC2, mLo, a1_layout,
                    b1_layout, b2_layout, p_layout, epi_layout, epi_tile, nu).launch(
            grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mma1: cute.TiledMma, mma2: cute.TiledMma, tma_qk: cute.CopyAtom, mQK: cute.Tensor,
               tma_v: cute.CopyAtom, mV: cute.Tensor, tma_da: cute.CopyAtom, mDA: cute.Tensor, tma_db: cute.CopyAtom,
               mDB: cute.Tensor, tma_das: cute.CopyAtom, mDAs: cute.Tensor, tma_dbv: cute.CopyAtom, mDBv: cute.Tensor,
               tma_k: cute.CopyAtom, mK: cute.Tensor, tma_dk: cute.CopyAtom, mDK: cute.Tensor, tma_dqk: cute.CopyAtom,
               mDQK: cute.Tensor, tma_dv: cute.CopyAtom, mDV: cute.Tensor, mLse2: cute.Tensor, mC2: cute.Tensor,
               mLo: cute.Tensor, a1_layout: cute.ComposedLayout, b1_layout: cute.ComposedLayout,
               b2_layout: cute.ComposedLayout, p_layout: cute.ComposedLayout, epi_layout: cute.ComposedLayout,
               epi_tile: cute.Tile, nu: Int32):
        S, N = self.S, self.N
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if warp == 0:
            for atom in (tma_qk, tma_v, tma_da, tma_db, tma_das, tma_dbv, tma_k, tma_dk, tma_dqk, tma_dv):
                cpasync.prefetch_descriptor(atom)
        rb = bidx % self.row_blocks
        t1 = bidx // self.row_blocks
        j = t1 % nu
        t2 = t1 // nu
        k = t2 % self.K
        b = t2 // self.K
        mem = (b * self.K + k) * nu + j  # (b, k, j) index of dA / dB
        lo = mLo[j, b]
        row0 = rb * self.TM  # first row of the block in the chunk
        rblk = j * self.row_blocks + rb  # ... as a 128-row block index of the sequence

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        one = cute.make_layout((1, 1, 1, 1))
        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads // 32)
        nbytes = lambda lay: cute.size_in_bytes(BFloat16, cute.slice_(lay, (None, None, 0)))
        ring = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.ring_mbar.data_ptr(), num_stages=self.stages, producer_group=prod_group,
            consumer_group=cons_group, tx_count=nbytes(a1_layout) + nbytes(b1_layout), cta_layout_vmnk=one, defer_sync=True)
        ring2 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.ring2_mbar.data_ptr(), num_stages=self.stages, producer_group=prod_group,
            consumer_group=cons_group, tx_count=nbytes(b2_layout), cta_layout_vmnk=one, defer_sync=True)
        kpipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.k_mbar.data_ptr(), num_stages=1, producer_group=prod_group,
            consumer_group=cons_group, tx_count=nbytes(p_layout), cta_layout_vmnk=one, defer_sync=True)
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
        sA1 = storage.sA1.get_tensor(a1_layout.outer, swizzle=a1_layout.inner)
        sB1 = storage.sB1.get_tensor(b1_layout.outer, swizzle=b1_layout.inner)
        sP = storage.sP.get_tensor(p_layout.outer, swizzle=p_layout.inner)
        # the second ring reuses the memory of the first one's B stages, the epilogue tiles that of its A stages
        sB2 = cute.make_tensor(cute.recast_ptr(sB1.iterator, b2_layout.inner, dtype=BFloat16), b2_layout.outer)
        sC = cute.make_tensor(cute.recast_ptr(sA1.iterator, epi_layout.inner, dtype=BFloat16), epi_layout.outer)
        sVec = storage.vec.get_tensor(cute.make_layout((S, 2), stride=(1, S)))

        wg = cute.arch.make_warp_uniform(tidx // 128)
        wg_sel = cute.make_layout(2, stride=128)(wg)
        wg1, wg2 = mma1.get_slice(wg_sel), mma2.get_slice(wg_sel)
        tCrA1 = mma1.make_fragment_A(wg1.partition_A(sA1))
        tCrB1 = mma1.make_fragment_B(wg1.partition_B(sB1))
        tCrP = mma2.make_fragment_A(wg2.partition_A(sP))
        tCrB2 = mma2.make_fragment_B(wg2.partition_B(sB2))
        acc = cute.make_rmem_tensor(wg1.partition_shape_C((self.TM, N)), Float32)
        tCc = mma1.get_slice(tidx).partition_C(cute.make_identity_tensor((self.TM, N)))
        pipeline_init_wait(cluster_shape_mn=(1, 1))

        skip = row0 + self.TM <= lo  # every row dropped: zero gradients
        if tidx < S:
            sVec[tidx, 0] = mLse2[tidx, j, k, b]
            sVec[tidx, 1] = mC2[tidx, j, k, b]
        prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
        prod2 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
        if not skip:
            if warp == 0:
                kprod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                kpipe.producer_acquire(kprod)
                gk = cute.group_modes(cute.local_tile(mK[None, None, k, b], (self.TM, S), (rblk, 0)), 0, 2)
                tks, tkg = cpasync.tma_partition(tma_k, 0, cute.make_layout(1), cute.group_modes(sP, 0, 2), gk)
                cute.copy(tma_k, tkg, tks[(None, 0)], tma_bar_ptr=kpipe.producer_get_barrier(kprod))
                kpipe.producer_commit(kprod)
                for _ in cutlass.range(min(self.stages, self.f_blocks), unroll=1):
                    self._load1(ring, prod, tma_qk, mQK, tma_v, mV, tma_da, mDA, tma_db, mDB, sA1, sB1, rblk, k, b, mem)
                    prod.advance()
            # ---- dP^T = [qk | v] [dA | dB]^T over the feature blocks ----
            cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            rel = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            mma1.set(warpgroup.Field.ACCUMULATE, False)
            for f in cutlass.range(self.f_blocks, unroll=1):
                ring.consumer_wait(cons)
                warpgroup.fence()
                for kb in cutlass.range(cute.size(tCrA1, mode=[2]), unroll_full=True):
                    cute.gemm(mma1, acc, tCrA1[(None, None, kb, cons.index)], tCrB1[(None, None, kb, cons.index)], acc)
                    mma1.set(warpgroup.Field.ACCUMULATE, True)
                warpgroup.commit_group()
                cons.advance()
                if f > 0:
                    warpgroup.wait_group(1)
                    ring.consumer_release(rel)
                    rel.advance()
                    if warp == 0 and prod.count < self.f_blocks:
                        self._load1(ring, prod, tma_qk, mQK, tma_v, mV, tma_da, mDA, tma_db, mDB, sA1, sB1, rblk, k, b, mem)
                        prod.advance()
            warpgroup.wait_group(0)
            pipeline.sync(barrier_id=2)  # every warpgroup is done with the first ring: its memory is reused
            if warp == 0:
                for _ in cutlass.range(min(self.stages, self.s_blocks), unroll=1):
                    self._load2(ring2, prod2, tma_das, mDAs, sB2, Int32(0), prod2.count, mem)
                    prod2.advance()

            # ---- P^T from the k tile (ldmatrix in the accumulator layout), dk, then P^T (bf16) over the k tile ----
            kcons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            kpipe.consumer_wait(kcons)
            ld_p = cute.make_tiled_copy_C(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), BFloat16), mma1)
            st_p = cute.make_tiled_copy_C(cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(False, 4), BFloat16), mma1)
            thr_ld, thr_st = ld_p.get_slice(tidx), st_p.get_slice(tidx)
            sP0 = cute.slice_(sP, (None, None, 0))
            rK = cute.make_rmem_tensor_like(acc, BFloat16)
            cute.copy(ld_p, thr_ld.partition_S(sP0), thr_ld.retile(rK))
            pipeline.sync(barrier_id=2)  # every thread has its k values (and sVec is written): P^T may overwrite them
            rP = cute.make_rmem_tensor_like(acc, BFloat16)
            for e in cutlass.range_constexpr(cute.size(acc)):
                r, cc = tCc[e]
                p = cute.math.exp2(rK[e].to(Float32) * LOG2E - sVec[cc, 0], fastmath=True)
                p = Float32(cutlass.select_(row0 + r >= lo, p, Float32(0.0)))
                rP[e] = p.to(BFloat16)
                acc[e] = p * (acc[e] + sVec[cc, 1])
            cute.copy(st_p, thr_st.retile(rP), thr_st.partition_D(sP0))
            cute.arch.fence_proxy("async.shared", space="cta")
            self._store(acc, sC, tma_dk, mDK, epi_tile, mma1, rblk, Int32(0), k, b, tidx, warp)  # its barriers order sP too

            # ---- dqk = P^T dA, then dv = P^T dB one N-wide tile at a time: n_tiles x s_blocks steps of the ring ----
            cons2 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            rel2 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            kpb = cute.size(tCrB2, mode=[2])  # mma K steps per 64-row s block
            n_steps = (1 + self.n_vt) * self.s_blocks
            for nt in cutlass.range(1 + self.n_vt, unroll=1):
                mma2.set(warpgroup.Field.ACCUMULATE, False)
                for sbk in cutlass.range(self.s_blocks, unroll=1):
                    ring2.consumer_wait(cons2)
                    warpgroup.fence()
                    for kb in cutlass.range(kpb, unroll_full=True):
                        cute.gemm(mma2, acc, tCrP[(None, None, sbk * kpb + kb, 0)], tCrB2[(None, None, kb, cons2.index)], acc)
                        mma2.set(warpgroup.Field.ACCUMULATE, True)
                    warpgroup.commit_group()
                    cons2.advance()
                    warpgroup.wait_group(0)
                    ring2.consumer_release(rel2)
                    rel2.advance()
                    if warp == 0 and prod2.count < n_steps:
                        self._load2(ring2, prod2, tma_das, mDAs, sB2, Int32(0), prod2.count, mem, tma_dbv, mDBv)
                        prod2.advance()
                if nt == 0:
                    self._store(acc, sC, tma_dqk, mDQK, epi_tile, mma1, rblk, Int32(0), k, b, tidx, warp)
                else:
                    self._store(acc, sC, tma_dv, mDV, epi_tile, mma1, rblk, nt - 1, k, b, tidx, warp)
        else:
            for e in cutlass.range_constexpr(cute.size(acc)):
                acc[e] = Float32(0.0)
            self._store(acc, sC, tma_dk, mDK, epi_tile, mma1, rblk, Int32(0), k, b, tidx, warp)
            self._store(acc, sC, tma_dqk, mDQK, epi_tile, mma1, rblk, Int32(0), k, b, tidx, warp)
            for vt in cutlass.range(self.n_vt, unroll=1):
                self._store(acc, sC, tma_dv, mDV, epi_tile, mma1, rblk, vt, k, b, tidx, warp)

    @cute.jit
    def _store(self, acc: cute.Tensor, sC: cute.Tensor, tma_c: cute.CopyAtom, mC: cute.Tensor, epi_tile: cute.Tile,
               tiled_mma: cute.TiledMma, rblk: Int32, nt: Int32, k: Int32, b: Int32, tidx: Int32, warp: Int32):
        """The 128 x N accumulator tile as bf16 into rows rblk, columns tile nt of mC: registers -> shared memory
        (stmatrix) -> TMA store, one epilogue sub-tile at a time (all threads; leaves every store completed)."""
        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(ROW_MAJOR, elem_ty_d=BFloat16, elem_ty_acc=Float32)
        copy_atom_c = cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(False, 4), BFloat16)
        tiled_r2s = cute.make_tiled_copy_S(copy_atom_r2s, cute.make_tiled_copy_C_atom(copy_atom_c, tiled_mma))
        thr_r2s = tiled_r2s.get_slice(tidx)
        tRS_sD = thr_r2s.partition_D(sC)
        tRS_rAcc = tiled_r2s.retile(acc)
        rD_shape = cute.shape(thr_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, Float32)
        n_rd = cute.size(tRS_rD)
        gC = cute.local_tile(mC[None, None, k, b], (self.TM, self.N), (rblk, nt))
        gC_epi = cute.zipped_divide(gC, epi_tile)
        bSG_sD, bSG_gD = cpasync.tma_partition(tma_c, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), gC_epi)
        n_epi = cute.size(gC_epi, mode=[1])
        epi_shape = gC_epi.shape[1]
        epi_order = cute.make_layout(epi_shape, stride=(epi_shape[1], 1))
        c_pipe = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stages, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads))
        for e in cutlass.range_constexpr(n_epi):
            for i in cutlass.range_constexpr(n_rd):
                tRS_rD[i] = tRS_rAcc[e * n_rd + i]
            out = cute.make_rmem_tensor_like(tRS_rD_layout, BFloat16)
            out.store(tRS_rD.load().to(BFloat16))
            buf = e % self.epi_stages
            cute.copy(tiled_r2s, out, tRS_sD[(None, None, None, buf)])
            cute.arch.fence_proxy("async.shared", space="cta")
            pipeline.sync(barrier_id=1)
            if warp == 0:
                cute.copy(tma_c, bSG_sD[(None, buf)], bSG_gD[(None, epi_order.get_hier_coord(e))])
                c_pipe.producer_commit()
                c_pipe.producer_acquire()
            pipeline.sync(barrier_id=1)
        if warp == 0:
            c_pipe.producer_tail()
        pipeline.sync(barrier_id=1)

    @cute.jit
    def _load1(self, ring, prod, tma_qk: cute.CopyAtom, mQK: cute.Tensor, tma_v: cute.CopyAtom, mV: cute.Tensor,
               tma_da: cute.CopyAtom, mDA: cute.Tensor, tma_db: cute.CopyAtom, mDB: cute.Tensor, sA1: cute.Tensor,
               sB1: cute.Tensor, rblk: Int32, k: Int32, b: Int32, mem: Int32):
        """Feature block prod.count of [qk | v] (rows of the block) and of [dA | dB] into stage prod.index."""
        f = prod.count
        ring.producer_acquire(prod)
        bar = ring.producer_get_barrier(prod)
        sa, sb = cute.group_modes(sA1, 0, 2), cute.group_modes(sB1, 0, 2)
        if f < self.s_blocks:
            ga = cute.group_modes(cute.local_tile(mQK[None, None, k, b], (self.TM, self.KB), (rblk, f)), 0, 2)
            tsa, tga = cpasync.tma_partition(tma_qk, 0, cute.make_layout(1), sa, ga)
            cute.copy(tma_qk, tga, tsa[(None, prod.index)], tma_bar_ptr=bar)
            gb = cute.group_modes(cute.local_tile(mDA[None, None, mem], (self.N, self.KB), (0, f)), 0, 2)
            tsb, tgb = cpasync.tma_partition(tma_da, 0, cute.make_layout(1), sb, gb)
            cute.copy(tma_da, tgb, tsb[(None, prod.index)], tma_bar_ptr=bar)
        else:
            fv = f - self.s_blocks
            ga = cute.group_modes(cute.local_tile(mV[None, None, k, b], (self.TM, self.KB), (rblk, fv)), 0, 2)
            tsa, tga = cpasync.tma_partition(tma_v, 0, cute.make_layout(1), sa, ga)
            cute.copy(tma_v, tga, tsa[(None, prod.index)], tma_bar_ptr=bar)
            gb = cute.group_modes(cute.local_tile(mDB[None, None, mem], (self.N, self.KB), (0, fv)), 0, 2)
            tsb, tgb = cpasync.tma_partition(tma_db, 0, cute.make_layout(1), sb, gb)
            cute.copy(tma_db, tgb, tsb[(None, prod.index)], tma_bar_ptr=bar)
        ring.producer_commit(prod)

    @cute.jit
    def _load2(self, ring2, prod, tma_das: cute.CopyAtom, mDAs: cute.Tensor, sB2: cute.Tensor, nt0: Int32, step: Int32,
               mem: Int32, tma_dbv: cute.CopyAtom = None, mDBv: cute.Tensor = None):
        """Step ``step`` of the second ring: s block step % s_blocks of dA (step < s_blocks) or of the v tile
        step // s_blocks - 1 of dB, MN-major, into stage prod.index."""
        ring2.producer_acquire(prod)
        bar = ring2.producer_get_barrier(prod)
        sbk = step % self.s_blocks
        ntile = step // self.s_blocks
        sb = cute.group_modes(sB2, 0, 2)
        if ntile == 0:
            gb = cute.group_modes(cute.local_tile(mDAs[None, None, mem], (self.N, self.KB), (nt0, sbk)), 0, 2)
            tsb, tgb = cpasync.tma_partition(tma_das, 0, cute.make_layout(1), sb, gb)
            cute.copy(tma_das, tgb, tsb[(None, prod.index)], tma_bar_ptr=bar)
        else:
            if const_expr(tma_dbv is not None):
                gb = cute.group_modes(cute.local_tile(mDBv[None, None, mem], (self.N, self.KB), (ntile - 1, sbk)), 0, 2)
                tsb, tgb = cpasync.tma_partition(tma_dbv, 0, cute.make_layout(1), sb, gb)
                cute.copy(tma_dbv, tgb, tsb[(None, prod.index)], tma_bar_ptr=bar)
        ring2.producer_commit(prod)


class AwmRecurrenceBwdWarp:
    """Phase 2' (reverse recurrence) on warp MMAs, the training case (no previous chunk; any arch with mma.sync):
    per (b, kv head), backwards over the updates u (chunk c = u + 1), with G = dL/dM' (starts at dmem or 0):

        drp[.., u, vt, s] = sum_{v in tile vt} G[s, v] D_u[s, v]        (dr partials, summed by AwmLseChain)
        dB_u = bf16(r G);   G <- (1 - r) G - A_u^T (r G);   G <- 0 on a memory reset;   G <- G + dMo_c
    and the final G (the input memory's gradient). Same structure as the forward AwmRecurrence: one CTA (8 warps)
    per (b, k, VT columns of V), G in fp32 registers in the warp-MMA accumulator layout, A_u^T (a transposed bf16
    copy of A) streamed through a cp.async ring, r G (bf16) as the B operand in shared memory."""

    def __init__(self, K: int, S: int, V: int, C: int, has_seg: bool, has_dmem: bool, VT: int = 64):
        self.K, self.S, self.V, self.C, self.has_seg, self.has_dmem = K, S, V, C, has_seg, has_dmem
        self.VT = min(VT, V)
        self.KB = 64
        self.STAGES = 4
        self.threads = 256
        self.PAD = 8
        assert V % self.VT == 0 and S % self.KB == 0 and S % 128 == 0 and self.VT % 16 == 0
        self.n_vt = V // self.VT

    @cute.jit
    def __call__(self, tA16T: cute.Tensor, tDMo: cute.Tensor, tR: cute.Tensor, tD: cute.Tensor, tSeg: cute.Tensor,
                 tDMem: cute.Tensor, tDB: cute.Tensor, tDrp: cute.Tensor, tGout: cute.Tensor, B: Int32, nc: Int32,
                 nu: Int32, seg_stride: Int32, stream: cuda.CUstream):
        K, S, V = self.K, self.S, self.V
        # A_u^T [B, K, nu, S(out), S(t)] with t contiguous: the layout the forward's block loader expects
        mA16 = cute.make_tensor(tA16T.iterator, cute.make_layout((S, S, nu, K, B), stride=(1, S, S * S, nu * S * S, K * nu * S * S)))
        mDMo = cute.make_tensor(tDMo.iterator, cute.make_layout((V, S, nc, K, B), stride=(1, V, S * V, nc * S * V, K * nc * S * V)))
        mD = cute.make_tensor(tD.iterator, cute.make_layout((V, S, nu, K, B), stride=(1, V, S * V, nu * S * V, K * nu * S * V)))
        mR = cute.make_tensor(tR.iterator, cute.make_layout((S, nu, K, B), stride=(1, S, nu * S, K * nu * S)))
        mDB = cute.make_tensor(tDB.iterator, cute.make_layout((V, S, nu, K, B), stride=(1, V, S * V, nu * S * V, K * nu * S * V)))
        mDrp = cute.make_tensor(tDrp.iterator, cute.make_layout((S, self.n_vt, nu, K, B), stride=(
            1, S, self.n_vt * S, nu * self.n_vt * S, K * nu * self.n_vt * S)))
        mG = cute.make_tensor(tGout.iterator, cute.make_layout((V, S, K, B), stride=(1, V, S * V, K * S * V)))
        mDMem = cute.make_tensor(tDMem.iterator, cute.make_layout((V, S, K, B), stride=(1, V, S * V, K * S * V)))
        op = cute.nvgpu.warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(op, cute.make_layout((8, 1, 1)), permutation_mnk=(128, 16, 16))
        grid = B * K * self.n_vt
        self.kernel(tiled_mma, mA16, mDMo, mR, mD, tSeg.iterator, mDMem, mDB, mDrp, mG, nc, nu, seg_stride).launch(
            grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.jit
    def _load_a_block(self, mA16: cute.Tensor, sA: cute.Tensor, atom, q, n_blocks, nkb, nu, k, b, tidx):
        """cp.async of A^T block q (update nu - 1 - q // nkb, the updates run backwards) into buffer q % STAGES."""
        if q < n_blocks:
            qu = q // nkb
            u = nu - 1 - qu
            col0 = (q - qu * nkb) * self.KB
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
    def kernel(self, tiled_mma: cute.TiledMma, mA16: cute.Tensor, mDMo: cute.Tensor, mR: cute.Tensor, mD: cute.Tensor,
               pSeg: cute.Pointer, mDMem: cute.Tensor, mDB: cute.Tensor, mDrp: cute.Tensor, mG: cute.Tensor, nc: Int32,
               nu: Int32, seg_stride: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        S, VT, KB = self.S, self.VT, self.KB
        vt = bidx % self.n_vt
        k = (bidx // self.n_vt) % self.K
        b = bidx // (self.n_vt * self.K)
        v0 = vt * VT
        nkb = S // KB
        smem = utils.SmemAllocator()
        sA = smem.allocate_tensor(BFloat16, cute.make_layout((S, KB, self.STAGES), stride=(KB + self.PAD, 1, S * (KB + self.PAD))),
                                  byte_alignment=16)
        sRG = smem.allocate_tensor(BFloat16, cute.make_layout((VT, S), stride=(S + self.PAD, 1)), byte_alignment=16)
        sR = smem.allocate_tensor(Float32, cute.make_layout(S), byte_alignment=16)
        sDr = smem.allocate_tensor(Float32, cute.make_layout(S), byte_alignment=16)
        atom = cute.make_copy_atom(cpasync.CopyG2SOp(), BFloat16, num_bits_per_copy=128)
        n_blocks = nu * nkb
        for p0 in cutlass.range_constexpr(self.STAGES - 1):
            self._load_a_block(mA16, sA, atom, Int32(p0), n_blocks, nkb, nu, k, b, tidx)

        thr_mma = tiled_mma.get_slice(tidx)
        coords = thr_mma.partition_C(cute.make_identity_tensor((S, VT)))
        g = cute.make_rmem_tensor(coords.shape, Float32)
        acc = cute.make_rmem_tensor(coords.shape, Float32)
        n_frag = cute.size(coords)
        if const_expr(self.has_dmem):
            for i in cutlass.range_constexpr(n_frag):
                s1, v1 = coords[i]
                g[i] = mDMem[v0 + v1, s1, k, b]
        else:
            g.fill(0.0)

        q = Int32(0)
        for qu in cutlass.range(nu):
            u = nu - 1 - qu
            c = u + 1
            if tidx < S:
                sR[tidx] = mR[tidx, u, k, b]
                sDr[tidx] = Float32(0.0)
            cute.arch.barrier()
            # dr partials of this v tile, and r G (bf16): dB and the B operand
            for i in cutlass.range_constexpr(n_frag):
                s2, v2 = coords[i]
                cute.arch.atomic_add((sDr.iterator + s2).llvm_ptr, g[i] * mD[v0 + v2, s2, u, k, b], sem="relaxed", scope="cta")
                rg = (sR[s2] * g[i]).to(BFloat16)
                mDB[v0 + v2, s2, u, k, b] = rg
                sRG[v2, s2] = rg
            acc.fill(0.0)
            for kb in cutlass.range(nkb):
                buf = q % self.STAGES
                cute.arch.cp_async_wait_group(self.STAGES - 2)
                cute.arch.barrier()  # block q landed; block q - 1's buffer is free; sRG / sDr written
                self._load_a_block(mA16, sA, atom, q + self.STAGES - 1, n_blocks, nkb, nu, k, b, tidx)
                tCsA = thr_mma.partition_A(sA[None, None, buf])
                tCsB = thr_mma.partition_B(cute.local_tile(sRG, (VT, KB), (0, kb)))
                tCrA = tiled_mma.make_fragment_A(tCsA)
                tCrB = tiled_mma.make_fragment_B(tCsB)
                cute.autovec_copy(tCsA, tCrA)
                cute.autovec_copy(tCsB, tCrB)
                for kk in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                    cute.gemm(tiled_mma, acc, tCrA[None, None, kk], tCrB[None, None, kk], acc)
                q += 1
            if tidx < S:
                mDrp[tidx, vt, u, k, b] = sDr[tidx]
            reset = cutlass.Boolean(False)
            if const_expr(self.has_seg):
                if c >= 2:  # a memory reset before update u: no gradient to the state before it
                    reset = _seg_anchor(pSeg, pSeg, b, c - 2, 0, seg_stride, self.C) != \
                        _seg_anchor(pSeg, pSeg, b, c - 1, 0, seg_stride, self.C)
            keep = cutlass.select_(reset, Float32(0.0), Float32(1.0))
            for i in cutlass.range_constexpr(n_frag):
                s4, v4 = coords[i]
                g[i] = ((1.0 - sR[s4]) * g[i] - acc[i]) * keep + mDMo[v0 + v4, s4, c, k, b]
            cute.arch.barrier()  # sR / sRG / sDr are rewritten in the next update
        for i in cutlass.range_constexpr(n_frag):
            s5, v5 = coords[i]
            mG[v0 + v5, s5, k, b] = g[i]
