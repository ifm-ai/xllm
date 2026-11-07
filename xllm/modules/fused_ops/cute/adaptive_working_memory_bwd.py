"""Adaptive working memory (AWM) backward with CuTe DSL kernels for Blackwell (SM100).

Mirror of the forward phases (adaptive_working_memory.py; the spec is awm_bwd_phased in
adaptive_working_memory_reference.py). With G = dL/dM' the gradient of the state after an update:

  3'. per chunk c (parallel):  dq_c = dOut_c M_c^T (masked rows 0),  dMo_c = sum_{h in group} q_c^T dOut_c
  2'. reverse recurrence over the updates (sequential per (b, k)):  rG = r G,  dB = rG,  dA = -(rG) M^T,
      dr = sum_v G D,  G <- (1 - r) G - A^T (rG) + dMo_c   (G <- dMo_c on a memory reset)
      then the lse chain:  g <- g - r dr;  dlse = r dr + r g;  g <- (1 - r) g   (g <- 0 on a reset)
  1'. per update chunk (parallel over 128-row blocks of time):  P^T = exp(k - lse) (masked rows 0),
      dP^T = qk dA^T + v dB^T,  dk = P^T (dP^T - delta + dlse) with delta = rowdot(dA, A) + rowdot(dB, B),
      dqk = P^T dA,  dv = P^T dB

Kernels (in stream order; the forward's workspaces come from adaptive_working_memory_fwd(return_workspace=True)
or are recomputed with its save option):
  AwmZeroTail       zero rows of the last chunk of dqk / dk / dv
  AwmOutGemm (forward kernel, roles of S and V swapped)   dq and the output row mask
  AwmDmoGemm        split-K tcgen05 GEMM with MN-major operands, masked time tiles skipped, TMA reduce-add
  AwmRecurrenceBwd  tcgen05 reverse recurrence (one GEMM per update), G in registers, rG as bf16 smem operand
  torch.matmul      dA = dB (-M)^T from the saved negated state
  AwmBwdDelta       delta = rowdot(dA, A) + rowdot(dB, B);  AwmLseChain: the lse chain -> dlse
  AwmChunkBwd       persistent, per 128 time rows: dP^T, P^T transform, dk / dqk / dv (TMA-store epilogues)

Training case only (no input memory / previous chunk, V a multiple of S); the torch backward is used otherwise.

bf16, L = 16384, chunk 4096, with segment_idx, B200 (GPU time per call, saved workspaces; the torch
recompute + backward is about 3.5 ms per call):
  gekko-0.9B (B2 H8 K4 S256 V256)   ~236us  (chunk 88, dq 49, dMo 48, recurrence 30)
  gekko-4B   (B1 H32 K8 S128 V128)  ~142us
  gekko-7.1B (B1 H4 K4 S256 V2048)  ~400us  (chunk 181, dq 72, dMo 72, recurrence 30)
Training step on 1 GPU: AWM backward 99 -> 6 ms per step (0.9B, B2), 98 -> 11 ms (7.1B, B1).
"""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05

try:
    from .adaptive_working_memory import (  # noqa: F401
        LOG2E, _U8, _blhs, _combine_lse2, _compiled, _dummy, _fake, _is_hopper, _out_gemm, _phases12, _seg_anchor,
        _sm_count, _u32_as_f32, chunk_gemm_tiling, WORKSPACE_KEYS,
    )
    from . import adaptive_working_memory as _FWD
except ImportError:  # loaded from its file (tests): load the forward module from its file as well
    import importlib.util as _ilu
    import os as _os
    _spec = _ilu.spec_from_file_location("cute_adaptive_working_memory_fwd",
                                         _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "adaptive_working_memory.py"))
    _fwd = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_fwd)
    LOG2E, _U8, _blhs, _combine_lse2, _compiled, _dummy, _fake, _out_gemm, _phases12, _seg_anchor, _sm_count = (
        _fwd.LOG2E, _fwd._U8, _fwd._blhs, _fwd._combine_lse2, _fwd._compiled, _fwd._dummy, _fwd._fake, _fwd._out_gemm,
        _fwd._phases12, _fwd._seg_anchor, _fwd._sm_count)
    _u32_as_f32, chunk_gemm_tiling, WORKSPACE_KEYS = _fwd._u32_as_f32, _fwd.chunk_gemm_tiling, _fwd.WORKSPACE_KEYS
    _is_hopper = _fwd._is_hopper
    _FWD = _fwd

import torch  # noqa: E402


# ======================================================================================================
# Phase 3': dMo[b, k, c] = sum_{h in group} sum_{kept rows t of chunk c} q[b, t, h, :]^T dOut[b, t, h, :]
# ======================================================================================================

DMO_KB = 64  # time rows per stage of AwmDmoGemm


class AwmDmoGemm:
    """dMo[b, k, c, s, :] += sum_t q[b, t, h, s] dOut[b, t, h, :] over the rows t of a split of the K range
    (h in the group of k) x (rows of chunk c); rows flagged in the output mask are dropped. dMo is fp32, zeroed
    by the caller.

    Same structure as AwmChunkGemm (both operands MN-major, q and dOut read as [B, L, H, *]):
      warp 5:    TMA loads of the raw q tile (128 s x KB rows) and the nsub dOut sub-tiles of a stage
      warps 0-3: copy of the raw tile into the A buffer with masked rows zeroed; at the end the epilogue
      warp 4:    tcgen05 MMAs
    """

    def __init__(self, H: int, K: int, S: int, V: int, C: int, splits: int):
        self.H, self.K, self.S, self.V, self.C = H, K, S, V, C
        self.n_rep = H // K
        self.NS, self.nsub, self.n_blocks = chunk_gemm_tiling(0, V)
        self.TM = 128
        self.KB = DMO_KB
        stage_ab = 128 * self.KB * 2
        stage_b = self.nsub * self.NS * self.KB * 2
        budget = 220 * 1024 - C
        self.p_stages = 3
        self.b_stages = max(2, min(4, (budget - (3 + self.p_stages) * stage_ab) // stage_b))
        self.k_stages = max(2, min(6, (budget - self.p_stages * stage_ab - self.b_stages * stage_b) // stage_ab))
        assert (self.p_stages + self.k_stages) * stage_ab + self.b_stages * stage_b <= budget
        self.epi_stages = 2
        self.splits = splits
        self.m_blocks = S // self.TM
        self.chunk_k_tiles = C // self.KB
        self.k_tiles = self.n_rep * self.chunk_k_tiles  # K range of one (b, k, c)
        self.num_tp_warps = 4
        self.mma_warp = 4
        self.tma_warp = 5
        self.threads = 32 * 6
        assert S % self.TM == 0 and C % self.KB == 0 and splits <= self.k_tiles
        assert self.nsub * self.NS <= 512

    @cute.jit
    def __call__(self, tQ: cute.Tensor, tDO: cute.Tensor, tMask: cute.Tensor, tDMo: cute.Tensor, B: Int32,
                 L: Int32, stream: cuda.CUstream):
        H, K, S, V = self.H, self.K, self.S, self.V
        nc = L // self.C
        mQ = cute.make_tensor(tQ.iterator, cute.make_layout((S, L, H, B), stride=(1, H * S, S, L * H * S)))
        mDO = cute.make_tensor(tDO.iterator, cute.make_layout((V, L, H, B), stride=(1, H * V, V, L * H * V)))
        op = tcgen05.MmaF16BF16Op(BFloat16, Float32, (self.TM, self.NS, 16), tcgen05.CtaGroup.ONE,
                                  tcgen05.OperandSource.SMEM, tcgen05.OperandMajorMode.MN, tcgen05.OperandMajorMode.MN)
        tiled_mma = cute.make_tiled_mma(op)
        tiler = (self.TM, self.NS, self.KB)
        raw_layout = sm100_utils.make_smem_layout_a(tiled_mma, tiler, BFloat16, self.k_stages)
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, tiler, BFloat16, self.p_stages)
        b_layout = sm100_utils.make_smem_layout_b(tiled_mma, tiler, BFloat16, self.b_stages * self.nsub)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_q, mQ_t = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mQ, cute.slice_(raw_layout, (None, None, None, 0)), tiler, tiled_mma)
        tma_do, mDO_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mDO, cute.slice_(b_layout, (None, None, None, 0)), tiler, tiled_mma)
        mDMo = cute.make_tensor(tDMo.iterator, cute.make_layout((S, V, nc, K, B), stride=(V, 1, S * V, nc * S * V, K * nc * S * V)))
        epi_tile = (self.TM, 32)
        epi_layout = sm100_utils.make_smem_layout_epi(Float32, utils.LayoutEnum.ROW_MAJOR, epi_tile, self.epi_stages)
        tma_red, mDMo_t = cpasync.make_tiled_tma_atom(cpasync.CopyReduceBulkTensorTileS2GOp(), mDMo,
                                                      cute.slice_(epi_layout, (None, None, 0)), epi_tile)
        grid = B * K * nc * self.m_blocks * self.n_blocks * self.splits
        self.kernel(tiled_mma, tma_q, mQ_t, tma_do, mDO_t, tMask.iterator, tma_red, mDMo_t, raw_layout, a_layout,
                    b_layout, epi_layout, L, nc).launch(grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream)

    def _p_offset(self, g, kk):
        """Element offset of A[8 g .. 8 g + 7, kk] in a stage of the MN-major A operand (128-byte swizzle)."""
        o = ((8 * g) % 64 + ((8 * g) // 64) * (64 * self.KB) + (kk % 16) * 64 + (kk // 16) * 1024) * 2
        return (o ^ (((o >> 7) & 7) << 4)) // 2

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, tma_q: cute.CopyAtom, mQ: cute.Tensor, tma_do: cute.CopyAtom,
               mDO: cute.Tensor, pMask: cute.Pointer, tma_red: cute.CopyAtom, mDMo: cute.Tensor,
               raw_layout: cute.ComposedLayout, a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout,
               epi_layout: cute.ComposedLayout, L: Int32, nc: Int32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        # unit decode: (split, n block, m block, c, k, b)
        ks = bidx % self.splits
        rest = bidx // self.splits
        nb = rest % self.n_blocks
        rest = rest // self.n_blocks
        mb = rest % self.m_blocks
        rest = rest // self.m_blocks
        c = rest % nc
        rest = rest // nc
        k = rest % self.K
        b = rest // self.K
        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_q)
            cpasync.prefetch_descriptor(tma_do)

        @cute.struct
        class SharedStorage:
            k_mbar: cute.struct.MemRange[cutlass.Int64, self.k_stages * 2]
            p_mbar: cute.struct.MemRange[cutlass.Int64, self.p_stages * 2]
            b_mbar: cute.struct.MemRange[cutlass.Int64, self.b_stages * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sCnt = smem.allocate_tensor(Int32, cute.make_layout(8), byte_alignment=16)
        # the masked rows of a chunk are a suffix: only the time tiles with kept rows, for every head of the group
        cnt = Int32(0)
        for i in cutlass.range(tidx, self.C, self.threads):
            mrow = cute.make_tensor(pMask + (b * L + c * self.C + i), cute.make_layout(1))[0]
            cnt = cnt + cutlass.select_(mrow != Uint8(0), Int32(0), Int32(1))
        for o in cutlass.range_constexpr(5):
            cnt = cnt + cute.arch.shuffle_sync_bfly(cnt, offset=1 << o)
        if tidx % 32 == 0:
            sCnt[warp] = cnt
        cute.arch.sync_threads()
        keep_rows = Int32(0)
        for wi in cutlass.range_constexpr(self.threads // 32):
            keep_rows = keep_rows + sCnt[wi]
        keep_tiles = (keep_rows + self.KB - 1) // self.KB
        k_tiles = self.n_rep * keep_tiles
        kt_lo = ks * k_tiles // self.splits
        n_kt = (ks + 1) * k_tiles // self.splits - kt_lo
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_tp_warps + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)

        a_bytes = cute.size_in_bytes(BFloat16, cute.select(a_layout, mode=[0, 1, 2]))
        b_bytes = cute.size_in_bytes(BFloat16, cute.select(b_layout, mode=[0, 1, 2])) * self.nsub
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        tp_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * self.num_tp_warps)
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

        gQ = cute.local_tile(mQ, (self.TM, self.KB), (None, None, None, None))
        gDO = cute.local_tile(mDO, (self.NS, self.KB), (None, None, None, None))
        thr_mma = tiled_mma.get_slice(0)
        tCgQ = thr_mma.partition_A(gQ)
        tCgDO = thr_mma.partition_B(gDO)
        tCrA = tiled_mma.make_fragment_A(sP)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C((self.TM, self.NS))
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.nsub))
        tQs, tQg = cpasync.tma_partition(tma_q, 0, cute.make_layout(1), cute.group_modes(sRaw, 0, 3), cute.group_modes(tCgQ, 0, 3))
        tDOs, tDOg = cpasync.tma_partition(tma_do, 0, cute.make_layout(1), cute.group_modes(sB, 0, 3), cute.group_modes(tCgDO, 0, 3))
        row_tiles0 = c * self.chunk_k_tiles  # first time tile of the chunk
        h0 = k * self.n_rep

        if warp == self.tma_warp:
            for t in cutlass.range(n_kt):
                kt = kt_lo + t
                h = h0 + kt // keep_tiles
                tt = row_tiles0 + kt % keep_tiles
                hk = k_prod.acquire_and_advance()
                cute.copy(tma_q, tQg[(None, mb, tt, h, b)], tQs[(None, hk.index)], tma_bar_ptr=hk.barrier)
                hb = b_prod.acquire_and_advance()
                for sub in cutlass.range_constexpr(self.nsub):
                    cute.copy(tma_do, tDOg[(None, nb * self.nsub + sub, tt, h, b)], tDOs[(None, hb.index * self.nsub + sub)],
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
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            # keep flags of the chunk's rows (output mask 0), once per CTA
            for i in cutlass.range(tidx, self.C, 32 * self.num_tp_warps):
                mrow = cute.make_tensor(pMask + (b * L + c * self.C + i), cute.make_layout(1))[0]
                sKeep[i] = cutlass.select_(mrow != Uint8(0), Uint8(0), Uint8(1))
            cute.arch.barrier(barrier_id=3, number_of_threads=32 * self.num_tp_warps)
            # ---- copy with masked rows zeroed ----
            g = tidx % 16  # 8 consecutive s rows: 8g .. 8g + 7 of the CTA's 128
            kq = tidx // 16  # time rows kq + 8 r of a stage
            stage_elems = self.TM * self.KB
            nr = self.KB // 8
            offs = [self._p_offset(g, kq + 8 * r) for r in range(nr)]
            zero4 = cute.make_rmem_tensor((4,), Uint32)
            zero4.fill(Uint32(0))
            for t in cutlass.range(n_kt):
                kt = kt_lo + t
                trow0 = (kt % keep_tiles) * self.KB
                hk = k_cons.wait_and_advance()
                hp = p_prod.acquire_and_advance()
                words = [cute.make_tensor(cute.recast_ptr((pRaw + (hk.index * stage_elems + offs[r])).align(16), dtype=Uint32),
                                          cute.make_layout(4)).load() for r in range(nr)]
                for r in cutlass.range_constexpr(nr):
                    dst = cute.make_tensor(cute.recast_ptr((pP + (hp.index * stage_elems + offs[r])).align(16), dtype=Uint32),
                                           cute.make_layout(4))
                    if sKeep[trow0 + kq + 8 * r] == 0:
                        dst.store(zero4.load())
                    else:
                        dst.store(words[r])
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                hk.release()
                hp.commit()
            # ---- epilogue: TMEM -> registers -> shared memory -> TMA reduce-add into dMo ----
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_full = acc_cons.wait_and_advance()
            if n_kt > 0:  # an empty K range (all rows masked) leaves the accumulator uninitialized
                sC = cute.make_tensor(cute.recast_ptr(cute.recast_ptr(pRaw, dtype=Float32), swizzle_=epi_layout.inner), epi_layout.outer)
                epi_bar = pipeline.NamedBarrier(barrier_id=3, num_threads=32 * self.num_tp_warps)
                epi_pipe = pipeline.PipelineTmaStore.create(
                    num_stages=self.epi_stages, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * self.num_tp_warps))
                copy_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE), Float32)
                gD = cute.local_tile(mDMo, (self.TM, 32), (None, None, None, None, None))
                tSs, tSg = cpasync.tma_partition(tma_red, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), cute.group_modes(gD, 0, 2))
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
                            cute.copy(tma_red, tSs[(None, buf)], tSg[(None, mb, col_tile0 + cchunk, c, k, b)])
                            epi_pipe.producer_commit()
                            epi_pipe.producer_acquire()
                        epi_bar.arrive_and_wait()
                epi_pipe.producer_tail()
            with cute.arch.elect_one():
                acc_full.release()
            tmem.relinquish_alloc_permit()
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_tp_warps)
            tmem.free(tmem_ptr)


# ======================================================================================================
# Phase 2': reverse recurrence
# ======================================================================================================

def _kmajor_off(row, col, rows):
    """Element offset of (row, col) in a 128-byte-swizzled K-major bf16 tile of ``rows`` rows (K blocks of
    64 elements, one after the other). Python ints or DSL ints."""
    return (col // 64) * (64 * rows) + 64 * row + 8 * (((col % 64) // 8) ^ (row % 8)) + col % 8


class AwmRecurrenceBwd:
    """Phase 2': per (b, k, VT columns of V), G = dL/dM' backwards over the updates (c = u + first):

        rG = r G;  dB_u[:, vt] = rG;  dr_part = sum_{v in vt} G D
        G <- (1 - r) G - A_u^T (rG) + dMo_c   (G <- dMo_c on a reset)

    The 4 * S/128 update warps (one row s per thread) keep G[s, vt] in registers and write rG^T as the bf16
    K-major B operand into shared memory; the MMA warp runs X = A^T rG on tcgen05 (A^T streamed by the TMA
    warp as MN-major blocks of the bf16 copy of A). dA = -(rG) M^T does not feed the chain and is computed
    afterwards from dB and the saved memory sequence (a batched matmul)."""

    def __init__(self, K: int, S: int, V: int, C: int, has_prev: bool, has_seg: bool, has_dmem: bool, VT: int = 64):
        self.K, self.S, self.V, self.C = K, S, V, C
        self.has_prev, self.has_seg, self.has_dmem = has_prev, has_seg, has_dmem
        self.VT = VT
        self.KB = 32
        self.stages = 8
        self.halves = S // 128
        self.n_vt = V // VT
        self.num_upd_warps = 4 * self.halves
        self.mma_warp = self.num_upd_warps
        self.tma_warp = self.num_upd_warps + 1
        self.threads = 32 * (self.num_upd_warps + 2)
        self.tmem_cols = max(32, 1 << (self.halves * VT - 1).bit_length())
        assert S % 128 == 0 and S <= 256 and V % VT == 0 and VT == 64

    @cute.jit
    def __call__(self, tA16: cute.Tensor, tDMo: cute.Tensor, tR: cute.Tensor, tD: cute.Tensor, tSeg: cute.Tensor,
                 tPrevCount: cute.Tensor, tDMem: cute.Tensor, tDB: cute.Tensor, tDrp: cute.Tensor, tGout: cute.Tensor,
                 B: Int32, nc: Int32, nu: Int32, seg_off: Int32, seg_stride: Int32, stream: cuda.CUstream):
        K, S, V, VT, n_vt = self.K, self.S, self.V, self.VT, self.n_vt
        # A^T blocks: A16[b, k, u, s, s'] as (M = s', K = s, u, k, b), MN-major
        mA = cute.make_tensor(tA16.iterator, cute.make_layout((S, S, nu, K, B), stride=(1, S, S * S, nu * S * S, K * nu * S * S)))
        mDMo = cute.make_tensor(tDMo.iterator, cute.make_layout((V, S, nc, K, B), stride=(1, V, S * V, nc * S * V, K * nc * S * V)))
        mR = cute.make_tensor(tR.iterator, cute.make_layout((S, nu, K, B), stride=(1, S, nu * S, K * nu * S)))
        mD = cute.make_tensor(tD.iterator, cute.make_layout((V, S, nu, K, B), stride=(1, V, S * V, nu * S * V, K * nu * S * V)))
        mDMem = cute.make_tensor(tDMem.iterator, cute.make_layout((V, S, K, B), stride=(1, V, S * V, K * S * V)))
        mDB = cute.make_tensor(tDB.iterator, cute.make_layout((V, S, nu, K, B), stride=(1, V, S * V, nu * S * V, K * nu * S * V)))
        mDrp = cute.make_tensor(tDrp.iterator, cute.make_layout((S, n_vt, nu, K, B), stride=(1, S, n_vt * S, nu * n_vt * S, K * nu * n_vt * S)))
        mGout = cute.make_tensor(tGout.iterator, cute.make_layout((V, S, K, B), stride=(1, V, S * V, K * S * V)))

        op1 = tcgen05.MmaF16BF16Op(BFloat16, Float32, (128, VT, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                                   tcgen05.OperandMajorMode.MN, tcgen05.OperandMajorMode.K)
        tm1 = cute.make_tiled_mma(op1)
        a_layout = sm100_utils.make_smem_layout_a(tm1, (128, VT, self.KB), BFloat16, self.stages * self.halves)
        rgt_layout = sm100_utils.make_smem_layout_b(tm1, (128, VT, S), BFloat16, 1)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_a, mA_t = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mA, cute.slice_(a_layout, (None, None, None, 0)), (128, VT, self.KB), tm1)
        grid = B * K * n_vt
        self.kernel(tm1, tma_a, mA_t, mDMo, mR, mD, tSeg.iterator, tPrevCount.iterator, mDMem, mDB, mDrp, mGout, a_layout,
                    rgt_layout, nc, nu, seg_off, seg_stride).launch(grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.jit
    def _keep(self, pSeg: cute.Pointer, pPrevCount: cute.Pointer, b, c, seg_off, seg_stride):
        """0 when the memory resets before the update at chunk c (the chunks before it end in different segments),
        else 1."""
        keep = Float32(1.0)
        if const_expr(self.has_seg):
            if c >= 2 or const_expr(self.has_prev):
                if _seg_anchor(pSeg, pPrevCount, b, c - 2, seg_off, seg_stride, self.C) != \
                        _seg_anchor(pSeg, pPrevCount, b, c - 1, seg_off, seg_stride, self.C):
                    keep = Float32(0.0)
        return keep

    @cute.kernel
    def kernel(self, tm1: cute.TiledMma, tma_a: cute.CopyAtom, mA: cute.Tensor, mDMo: cute.Tensor, mR: cute.Tensor,
               mD: cute.Tensor, pSeg: cute.Pointer, pPrevCount: cute.Pointer, mDMem: cute.Tensor, mDB: cute.Tensor,
               mDrp: cute.Tensor, mGout: cute.Tensor, a_layout: cute.ComposedLayout, rgt_layout: cute.ComposedLayout,
               nc: Int32, nu: Int32, seg_off: Int32, seg_stride: Int32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        bidx, _, _ = cute.arch.block_idx()
        S, VT, KB = self.S, self.VT, self.KB
        vt = bidx % self.n_vt
        k = (bidx // self.n_vt) % self.K
        b = bidx // (self.n_vt * self.K)
        v0 = vt * VT
        nkb = S // KB
        first = 0 if const_expr(self.has_prev) else 1

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_a)

        @cute.struct
        class SharedStorage:
            a_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            m_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_upd_warps + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        upd = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * self.num_upd_warps)
        a_bytes = cute.size_in_bytes(BFloat16, cute.select(a_layout, mode=[0, 1, 2])) * self.halves
        a_prod, a_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_mbar.data_ptr(), num_stages=self.stages, producer_group=one, consumer_group=one,
            tx_count=a_bytes).make_participants()
        m_prod, m_cons = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.m_mbar.data_ptr(), num_stages=1, producer_group=upd, consumer_group=one).make_participants()
        acc_prod, acc_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(), num_stages=1, producer_group=one,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_upd_warps)).make_participants()

        sA = smem.allocate_tensor(BFloat16, a_layout.outer, byte_alignment=1024, swizzle=a_layout.inner)
        pRGT = smem.allocate_array(BFloat16, cute.cosize(rgt_layout.outer), byte_alignment=1024)
        sRGT = cute.make_tensor(cute.recast_ptr(pRGT, swizzle_=rgt_layout.inner), rgt_layout.outer)

        gA = cute.local_tile(mA, (128, KB), (None, None, None, None, None))
        thr1 = tm1.get_slice(0)
        tCgA = thr1.partition_A(gA)
        tCrA1 = tm1.make_fragment_A(sA)
        tCrB1 = tm1.make_fragment_B(sRGT)
        acc1_fake = tm1.make_fragment_C(cute.append(tm1.partition_shape_C((128, VT)), self.halves))
        tAs, tAg = cpasync.tma_partition(tma_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))

        if warp == self.tma_warp:
            for ur in cutlass.range(nu):
                u = nu - 1 - ur
                for kb in cutlass.range(nkb):
                    ha = a_prod.acquire_and_advance()
                    for h in cutlass.range_constexpr(self.halves):
                        cute.copy(tma_a, tAg[(None, h, kb, u, k, b)], tAs[(None, ha.index * self.halves + h)], tma_bar_ptr=ha.barrier)
            a_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            acc1 = cute.make_tensor(tmem.retrieve_ptr(Float32), acc1_fake.layout)
            kpb = cute.size(tCrA1, mode=[2])
            for ur in cutlass.range(nu):
                hm = m_cons.wait_and_advance()
                e1 = acc_prod.acquire_and_advance()
                for kb in cutlass.range(nkb):
                    ha = a_cons.wait_and_advance()
                    for h in cutlass.range_constexpr(self.halves):
                        tm1.set(tcgen05.Field.ACCUMULATE, kb > 0)
                        for kk in cutlass.range_constexpr(kpb):
                            cute.gemm(tm1, acc1[(None, None, None, h)], tCrA1[(None, None, kk, ha.index * self.halves + h)],
                                      tCrB1[(None, None, kb * kpb + kk, 0)], acc1[(None, None, None, h)])
                            tm1.set(tcgen05.Field.ACCUMULATE, True)
                    ha.release()
                e1.commit()
                hm.release()
            acc_prod.tail()
        else:
            tmem.allocate(self.tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            acc1 = cute.make_tensor(tmem_ptr, acc1_fake.layout)
            half = warp // 4
            m_row = 32 * (warp % 4) + lane  # row within the half (= TMEM lane)
            srow = half * 128 + m_row
            # rG^T (VT x S, K = s): column srow
            rgt_base = (srow // 64) * (64 * VT) + srow % 8
            cx = (srow % 64) // 8
            rgt_x = [rgt_base + 8 * (cx ^ e) for e in range(8)]
            G = cute.make_rmem_tensor((VT,), Float32)
            if const_expr(self.has_dmem):
                G.store(cute.make_tensor((mDMem.iterator + mDMem.layout((v0, srow, k, b))).align(16), cute.make_layout(VT)).load())
            else:
                G.fill(0.0)
            copy_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE), Float32)
            acc1_epi = cute.flat_divide(acc1[((None, None), 0, 0, half)], (128, 32))
            t2r1 = tcgen05.make_tmem_copy(copy_atom, acc1_epi[(None, None, 0, 0)])
            tTR1 = t2r1.get_slice(tidx % 128).partition_S(acc1_epi)
            regs = cute.make_rmem_tensor(t2r1.get_slice(tidx % 128).partition_D(cute.make_identity_tensor((128, 32))).shape, Float32)
            rg8 = cute.make_rmem_tensor((8,), BFloat16)
            # per-update scalars, loaded one update ahead (the next update's are fetched while this one runs)
            r_next = mR[srow, nu - 1, k, b]
            keep_next = self._keep(pSeg, pPrevCount, b, nu - 1 + first, seg_off, seg_stride)
            for ur in cutlass.range(nu):
                u = nu - 1 - ur
                c = u + first
                r = r_next
                keep = keep_next  # 0 on a reset: no gradient flows to the state before it
                if ur + 1 < nu:
                    r_next = mR[srow, u - 1, k, b]
                    keep_next = self._keep(pSeg, pPrevCount, b, c - 1, seg_off, seg_stride)
                hmp = m_prod.acquire_and_advance()
                # rG into the operand buffer and dB; dr partial (D streamed in 8-element pieces)
                dot = Float32(0.0)
                for j in cutlass.range_constexpr(VT // 8):
                    d8 = cute.make_tensor((mD.iterator + mD.layout((v0 + 8 * j, srow, u, k, b))).align(16), cute.make_layout(8)).load()
                    for e in cutlass.range_constexpr(8):
                        v = 8 * j + e
                        rg8[e] = (r * G[v]).to(BFloat16)
                        dot = dot + G[v] * d8[e]
                        pRGT[rgt_x[v % 8] + 64 * v] = rg8[e]
                    cute.make_tensor((mDB.iterator + mDB.layout((v0 + 8 * j, srow, u, k, b))).align(16), cute.make_layout(8)).store(rg8.load())
                mDrp[srow, vt, u, k, b] = dot
                cute.arch.fence_view_async_shared()
                hmp.commit()
                # G <- (1 - r) G - X + dMo_c, or dMo_c on a reset (dMo streamed in 32-element pieces)
                f1 = acc_cons.wait_and_advance()
                for ch in cutlass.range_constexpr(VT // 32):
                    dmo = cute.make_tensor((mDMo.iterator + mDMo.layout((v0 + ch * 32, srow, c, k, b))).align(16), cute.make_layout(32)).load()
                    cute.copy(t2r1, tTR1[(None, None, None, 0, ch)], regs)
                    for e in cutlass.range_constexpr(32):
                        v = ch * 32 + e
                        G[v] = keep * ((1.0 - r) * G[v] - regs[e]) + dmo[e]
                with cute.arch.elect_one():
                    f1.release()
            cute.make_tensor((mGout.iterator + mGout.layout((v0, srow, k, b))).align(16), cute.make_layout(VT)).store(G.load())
            tmem.relinquish_alloc_permit()
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_upd_warps)
            tmem.free(tmem_ptr)


class AwmBwdDelta:
    """delta[b, k, u, s] = sum_s' dA16[s, s'] A[s, s'] + sum_v dB16[s, v] B[s, v]  (= sum_t P[s, t] dP[s, t]).
    One warp per row (b, k, u, s), lanes over the columns."""

    def __init__(self, S: int, V: int):
        self.S, self.V, self.NAB = S, V, S + V
        self.threads = 256

    @cute.jit
    def __call__(self, tDA16: cute.Tensor, tAB: cute.Tensor, tDB: cute.Tensor, tDelta: cute.Tensor, rows: Int32,
                 stream: cuda.CUstream):
        grid = (rows * 32 + self.threads - 1) // self.threads
        self.kernel(tDA16.iterator, tAB.iterator, tDB.iterator, tDelta.iterator, rows).launch(
            grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pDA16: cute.Pointer, pAB: cute.Pointer, pDB: cute.Pointer, pDelta: cute.Pointer, rows: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        row = (bidx * self.threads + tidx) // 32  # (b, k, u, s) flattened
        S, V = self.S, self.V
        if row < rows:
            a_row = pAB + row * self.NAB
            delta = Float32(0.0)
            for j in cutlass.range(lane * 8, S, 256):
                da8 = cute.make_tensor((pDA16 + (row * S + j)).align(16), cute.make_layout(8)).load()
                a8 = cute.make_tensor((a_row + j).align(16), cute.make_layout(8)).load()
                for e in cutlass.range_constexpr(8):
                    delta = delta + da8[e].to(Float32) * a8[e]
            for j in cutlass.range(lane * 8, V, 256):
                db8 = cute.make_tensor((pDB + (row * V + j)).align(16), cute.make_layout(8)).load()
                b8 = cute.make_tensor((a_row + (S + j)).align(16), cute.make_layout(8)).load()
                for e in cutlass.range_constexpr(8):
                    delta = delta + db8[e].to(Float32) * b8[e]
            for o in cutlass.range_constexpr(5):
                delta = delta + cute.arch.shuffle_sync_bfly(delta, offset=1 << o)
            if lane == 0:
                cute.make_tensor(pDelta + row, cute.make_layout(1))[0] = delta


class AwmZeroTail:
    """Zero the rows of the last chunk of dqk, dk [B, L, K, S] and dv [B, L, K, V] (no update uses them); dv is left
    alone with ``with_dv=False`` (it accumulates into an existing gradient)."""

    def __init__(self, K: int, S: int, V: int, C: int, with_dv: bool = True):
        self.K, self.S, self.V, self.C, self.with_dv = K, S, V, C, with_dv
        self.threads = 256

    @cute.jit
    def __call__(self, tDQK: cute.Tensor, tDK: cute.Tensor, tDV: cute.Tensor, B: Int32, L: Int32, stream: cuda.CUstream):
        n1 = B * self.C * self.K * self.S // 8
        n2 = (B * self.C * self.K * self.V // 8) if self.with_dv else 0
        grid = (n1 + n2 + self.threads - 1) // self.threads
        self.kernel(tDQK.iterator, tDK.iterator, tDV.iterator, L, n1, n2).launch(grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pDQK: cute.Pointer, pDK: cute.Pointer, pDV: cute.Pointer, L: Int32, n1: Int32, n2: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * self.threads + tidx
        z8 = cute.make_rmem_tensor((8,), BFloat16)
        z8.fill(BFloat16(0.0))
        if i < n1:
            per_b = self.C * self.K * self.S // 8
            b = i // per_b
            off = b * (L * self.K * self.S) + (L - self.C) * self.K * self.S + (i - b * per_b) * 8
            cute.make_tensor((pDQK + off).align(16), cute.make_layout(8)).store(z8.load())
            cute.make_tensor((pDK + off).align(16), cute.make_layout(8)).store(z8.load())
        elif i < n1 + n2:
            i2 = i - n1
            per_b = self.C * self.K * self.V // 8
            b = i2 // per_b
            off = b * (L * self.K * self.V) + (L - self.C) * self.K * self.V + (i2 - b * per_b) * 8
            cute.make_tensor((pDV + off).align(16), cute.make_layout(8)).store(z8.load())


class AwmLseChain:
    """dlse[b, k, u, s] from the dr partials and r, backwards over the updates (one thread per (b, k, s)):
    g <- g - r dr;  dlse = r dr + r g;  g <- (1 - r) g  (g <- 0 on a memory reset). g starts at dL/dlnt_out
    (when given) and ends as dL/dlnt_in."""

    def __init__(self, K: int, S: int, C: int, n_vt: int, has_prev: bool, has_seg: bool, has_dlnt: bool):
        self.K, self.S, self.C, self.n_vt = K, S, C, n_vt
        self.has_prev, self.has_seg, self.has_dlnt = has_prev, has_seg, has_dlnt
        self.threads = 256

    @cute.jit
    def __call__(self, tDrp: cute.Tensor, tR: cute.Tensor, tSeg: cute.Tensor, tPrevCount: cute.Tensor, tDLntIn: cute.Tensor,
                 tDlse: cute.Tensor, tGout: cute.Tensor, B: Int32, nu: Int32, seg_off: Int32, seg_stride: Int32,
                 stream: cuda.CUstream):
        K, S, n_vt = self.K, self.S, self.n_vt
        mDrp = cute.make_tensor(tDrp.iterator, cute.make_layout((S, n_vt, nu, K, B), stride=(1, S, n_vt * S, nu * n_vt * S, K * nu * n_vt * S)))
        mR = cute.make_tensor(tR.iterator, cute.make_layout((S, nu, K, B), stride=(1, S, nu * S, K * nu * S)))
        mDLntIn = cute.make_tensor(tDLntIn.iterator, cute.make_layout((S, K, B), stride=(1, S, K * S)))
        mDlse = cute.make_tensor(tDlse.iterator, cute.make_layout((S, nu, K, B), stride=(1, S, nu * S, K * nu * S)))
        mGout = cute.make_tensor(tGout.iterator, cute.make_layout((S, K, B), stride=(1, S, K * S)))
        total = B * K * S
        grid = (total + self.threads - 1) // self.threads
        self.kernel(mDrp, mR, tSeg.iterator, tPrevCount.iterator, mDLntIn, mDlse, mGout, total, nu, seg_off, seg_stride).launch(
            grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mDrp: cute.Tensor, mR: cute.Tensor, pSeg: cute.Pointer, pPrevCount: cute.Pointer, mDLntIn: cute.Tensor,
               mDlse: cute.Tensor, mGout: cute.Tensor, total: Int32, nu: Int32, seg_off: Int32, seg_stride: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * self.threads + tidx
        first = 0 if const_expr(self.has_prev) else 1
        if i < total:
            s = i % self.S
            k = (i // self.S) % self.K
            b = i // (self.S * self.K)
            g = Float32(0.0)
            if const_expr(self.has_dlnt):
                g = mDLntIn[s, k, b]
            for ur in cutlass.range(nu):
                u = nu - 1 - ur
                c = u + first
                dr = Float32(0.0)
                for t in cutlass.range_constexpr(self.n_vt):
                    dr = dr + mDrp[s, t, u, k, b]
                r = mR[s, u, k, b]
                g_new = g - dr * r
                mDlse[s, u, k, b] = dr * r + g_new * r
                g = g_new * (1.0 - r)
                if const_expr(self.has_seg):
                    if c >= 2 or const_expr(self.has_prev):
                        if _seg_anchor(pSeg, pPrevCount, b, c - 2, seg_off, seg_stride, self.C) != \
                                _seg_anchor(pSeg, pPrevCount, b, c - 1, seg_off, seg_stride, self.C):
                            g = Float32(0.0)
            mGout[s, k, b] = g


# ======================================================================================================
# Phase 1': per update chunk, 128 time rows per CTA
# ======================================================================================================

class AwmChunkBwd:
    """For the rows t of a 128-row block (tile) of update chunk j (= u, no previous chunk) of (b, k):

        P^T[t, s] = exp(k[t, s] - lse[s])  (0 on masked rows)
        dP^T = qk dA^T + v dB^T  (128 x S, K = S + V)
        dk[t, :] = P^T[t, :] (dP^T[t, :] - delta + dlse),   dqk = P^T dA (K = S),   dv = P^T dB (K = S, V/S tiles)

    Persistent: every CTA takes the tiles bidx, bidx + grid, ... of the update chunks. Warps 0-7 in two groups
    (thread = one row t of the tile; the groups take alternate rows of the transform and alternate 32-column
    pieces of the epilogues): the P^T transform of the raw key tile (TMA ring, 128 x 64 blocks) into the resident
    bf16 A operand, then the epilogues (TMEM -> registers -> bf16 -> shared memory -> TMA store); warp 8 the
    tcgen05 MMAs; warp 9 the TMA loads. GEMM operand tiles are 128 x 32 (A, K-major) and S x 32 (B; K-major
    for dP^T, MN-major for dqk / dv) in 4-stage rings: the loads come from L2 and are latency bound, so several
    blocks are kept in flight. The two
    128 x S TMEM accumulators alternate between dP^T, dqk and the dv tiles, so the next tile's dP^T GEMM runs
    while the epilogues of this one drain."""

    def __init__(self, K: int, S: int, V: int, C: int, R: int, has_seg: bool, accum_dv: bool = False):
        # accum_dv: dv is added into the given buffer (TMA reduce-add store) instead of overwriting it
        self.K, self.S, self.V, self.C, self.R, self.has_seg, self.accum_dv = K, S, V, C, R, has_seg, accum_dv
        self.KB = 64  # raw key blocks (one 128-byte swizzle row)
        self.KBG = 32  # K blocks of the GEMM operand rings
        self.a_stages = 4
        self.b_stages = 4
        self.epi_groups = 2
        self.epi_stages = 2  # staging buffers per group
        self.raw_stages = 2
        self.num_epi_warps = 4 * self.epi_groups
        self.mma_warp = self.num_epi_warps
        self.tma_warp = self.num_epi_warps + 1
        self.threads = 32 * (self.num_epi_warps + 2)
        self.n_vtiles = V // S
        self.kb_s = S // self.KB
        self.kbg_s = S // self.KBG
        self.kbg_v = V // self.KBG
        self.tmem_cols = max(32, 1 << (2 * S - 1).bit_length())
        assert S % 128 == 0 and S <= 256 and V % S == 0 and C % 128 == 0

    @cute.jit
    def __call__(self, tQK: cute.Tensor, tK: cute.Tensor, tV: cute.Tensor, tDA: cute.Tensor, tDB: cute.Tensor, tPart: cute.Tensor,
                 tDelta: cute.Tensor, tDlse: cute.Tensor, tSeg: cute.Tensor, tDQK: cute.Tensor, tDK: cute.Tensor, tDV: cute.Tensor,
                 B: Int32, L: Int32, nu: Int32, seg_off: Int32, seg_stride: Int32, num_ctas: Int32, stream: cuda.CUstream):
        K, S, V, R, KB, KBG = self.K, self.S, self.V, self.R, self.KB, self.KBG
        # inputs [B, L, K, *] as (M = t, K = col, k, b), K-major
        mQK = cute.make_tensor(tQK.iterator, cute.make_layout((L, S, K, B), stride=(K * S, 1, S, L * K * S)))
        mK = cute.make_tensor(tK.iterator, cute.make_layout((L, S, K, B), stride=(K * S, 1, S, L * K * S)))
        mV = cute.make_tensor(tV.iterator, cute.make_layout((L, V, K, B), stride=(K * V, 1, V, L * K * V)))
        # dA16[b, k, u, s, s'] / dB16[b, k, u, s, v]: K-major B (N = s) for dP^T, MN-major B (N = s' / v) for dqk / dv
        mDAk = cute.make_tensor(tDA.iterator, cute.make_layout((S, S, nu, K, B), stride=(S, 1, S * S, nu * S * S, K * nu * S * S)))
        mDBk = cute.make_tensor(tDB.iterator, cute.make_layout((S, V, nu, K, B), stride=(V, 1, S * V, nu * S * V, K * nu * S * V)))
        mDAmn = cute.make_tensor(tDA.iterator, cute.make_layout((S, S, nu, K, B), stride=(1, S, S * S, nu * S * S, K * nu * S * S)))
        mDBmn = cute.make_tensor(tDB.iterator, cute.make_layout((V, S, nu, K, B), stride=(1, V, S * V, nu * S * V, K * nu * S * V)))
        mPart = cute.make_tensor(tPart.iterator, cute.make_layout((2, S, R, nu, K, B), stride=(
            1, 2, 2 * S, 2 * S * R, 2 * S * R * nu, 2 * S * R * nu * K)))
        mDelta = cute.make_tensor(tDelta.iterator, cute.make_layout((S, nu, K, B), stride=(1, S, nu * S, K * nu * S)))
        mDlse = cute.make_tensor(tDlse.iterator, cute.make_layout((S, nu, K, B), stride=(1, S, nu * S, K * nu * S)))
        # outputs [B, L, K, *] as (M = t, N = col, k, b)
        mDQK = cute.make_tensor(tDQK.iterator, cute.make_layout((L, S, K, B), stride=(K * S, 1, S, L * K * S)))
        mDK = cute.make_tensor(tDK.iterator, cute.make_layout((L, S, K, B), stride=(K * S, 1, S, L * K * S)))
        mDV = cute.make_tensor(tDV.iterator, cute.make_layout((L, V, K, B), stride=(K * V, 1, V, L * K * V)))

        op_kk = tcgen05.MmaF16BF16Op(BFloat16, Float32, (128, S, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                                     tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
        op_kmn = tcgen05.MmaF16BF16Op(BFloat16, Float32, (128, S, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                                      tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN)
        tm_kk = cute.make_tiled_mma(op_kk)
        tm_kmn = cute.make_tiled_mma(op_kmn)
        a_layout = sm100_utils.make_smem_layout_a(tm_kk, (128, S, KBG), BFloat16, self.a_stages)
        raw_layout = sm100_utils.make_smem_layout_a(tm_kk, (128, S, KB), BFloat16, self.raw_stages)
        p_layout = sm100_utils.make_smem_layout_a(tm_kmn, (128, S, S), BFloat16, 1)
        bk_layout = sm100_utils.make_smem_layout_b(tm_kk, (128, S, KBG), BFloat16, self.b_stages)
        bmn_layout = sm100_utils.make_smem_layout_b(tm_kmn, (128, S, KBG), BFloat16, self.b_stages)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        a_one = cute.slice_(a_layout, (None, None, None, 0))
        tma_qk, mQK_t = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mQK, a_one, (128, S, KBG), tm_kk)
        tma_v, mV_t = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mV, a_one, (128, S, KBG), tm_kk)
        tma_k, mK_t = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mK, cute.slice_(raw_layout, (None, None, None, 0)), (128, S, KB), tm_kk)
        bk_one = cute.slice_(bk_layout, (None, None, None, 0))
        bmn_one = cute.slice_(bmn_layout, (None, None, None, 0))
        tma_dak, mDAk_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mDAk, bk_one, (128, S, KBG), tm_kk)
        tma_dbk, mDBk_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mDBk, bk_one, (128, S, KBG), tm_kk)
        tma_damn, mDAmn_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mDAmn, bmn_one, (128, S, KBG), tm_kmn)
        tma_dbmn, mDBmn_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mDBmn, bmn_one, (128, S, KBG), tm_kmn)
        # epilogue: 128 x 32 bf16 pieces staged in shared memory, TMA stores
        epi_tile = (128, 32)
        epi_layout = sm100_utils.make_smem_layout_epi(BFloat16, utils.LayoutEnum.ROW_MAJOR, epi_tile, self.epi_stages * self.epi_groups)
        epi_one = cute.slice_(epi_layout, (None, None, 0))
        tma_oqk, mDQK_t = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mDQK, epi_one, epi_tile)
        tma_ok, mDK_t = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mDK, epi_one, epi_tile)
        ov_op = cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionKind.ADD) if self.accum_dv else cpasync.CopyBulkTensorTileS2GOp()
        tma_ov, mDV_t = cpasync.make_tiled_tma_atom(ov_op, mDV, epi_one, epi_tile)
        num_tiles = B * K * nu * (self.C // 128)
        self.kernel(tm_kk, tm_kmn, tma_qk, mQK_t, tma_v, mV_t, tma_k, mK_t, tma_dak, mDAk_t, tma_dbk, mDBk_t, tma_damn, mDAmn_t,
                    tma_dbmn, mDBmn_t, mPart, mDelta, mDlse, tSeg.iterator, tma_oqk, mDQK_t, tma_ok, mDK_t, tma_ov, mDV_t,
                    a_layout, raw_layout, p_layout, bk_layout, bmn_layout, epi_layout, nu, num_tiles, seg_off, seg_stride).launch(
            grid=[num_ctas, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.jit
    def _decode(self, t: Int32, nu: Int32):
        tiles_per_chunk = self.C // 128
        tb = t % tiles_per_chunk
        rest = t // tiles_per_chunk
        u = rest % nu
        rest = rest // nu
        k = rest % self.K
        b = rest // self.K
        return tb, u, k, b

    @cute.kernel
    def kernel(self, tm_kk: cute.TiledMma, tm_kmn: cute.TiledMma, tma_qk: cute.CopyAtom, mQK: cute.Tensor, tma_v: cute.CopyAtom,
               mV: cute.Tensor, tma_k: cute.CopyAtom, mK: cute.Tensor, tma_dak: cute.CopyAtom, mDAk: cute.Tensor,
               tma_dbk: cute.CopyAtom, mDBk: cute.Tensor, tma_damn: cute.CopyAtom, mDAmn: cute.Tensor, tma_dbmn: cute.CopyAtom,
               mDBmn: cute.Tensor, mPart: cute.Tensor, mDelta: cute.Tensor, mDlse: cute.Tensor, pSeg: cute.Pointer,
               tma_oqk: cute.CopyAtom, mDQK: cute.Tensor, tma_ok: cute.CopyAtom, mDK: cute.Tensor, tma_ov: cute.CopyAtom,
               mDV: cute.Tensor, a_layout: cute.ComposedLayout, raw_layout: cute.ComposedLayout,
               p_layout: cute.ComposedLayout, bk_layout: cute.ComposedLayout, bmn_layout: cute.ComposedLayout,
               epi_layout: cute.ComposedLayout, nu: Int32, num_tiles: Int32, seg_off: Int32, seg_stride: Int32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nctas, _, _ = cute.arch.grid_dim()
        S, V, KB, KBG = self.S, self.V, self.KB, self.KBG
        tiles_per_chunk = self.C // 128
        kb_total = self.kbg_s + self.kbg_v  # K blocks of dP^T

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_qk)
            cpasync.prefetch_descriptor(tma_v)
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_dak)
            cpasync.prefetch_descriptor(tma_dbk)
            cpasync.prefetch_descriptor(tma_damn)
            cpasync.prefetch_descriptor(tma_dbmn)
        if warp % 4 == 0:
            if warp < self.mma_warp:
                cpasync.prefetch_descriptor(tma_oqk)
                cpasync.prefetch_descriptor(tma_ok)
                cpasync.prefetch_descriptor(tma_ov)

        @cute.struct
        class SharedStorage:
            a_mbar: cute.struct.MemRange[cutlass.Int64, self.a_stages * 2]
            b_mbar: cute.struct.MemRange[cutlass.Int64, self.b_stages * 2]
            raw_mbar: cute.struct.MemRange[cutlass.Int64, self.raw_stages * 2]
            p_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_epi_warps + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        epi_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * self.num_epi_warps)
        a_bytes = cute.size_in_bytes(BFloat16, cute.select(a_layout, mode=[0, 1, 2]))
        b_bytes = cute.size_in_bytes(BFloat16, cute.select(bk_layout, mode=[0, 1, 2]))
        raw_bytes = cute.size_in_bytes(BFloat16, cute.select(raw_layout, mode=[0, 1, 2]))
        a_prod, a_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_mbar.data_ptr(), num_stages=self.a_stages, producer_group=one, consumer_group=one,
            tx_count=a_bytes).make_participants()
        b_prod, b_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_mbar.data_ptr(), num_stages=self.b_stages, producer_group=one, consumer_group=one,
            tx_count=b_bytes).make_participants()
        raw_prod, raw_cons = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.raw_mbar.data_ptr(), num_stages=self.raw_stages, producer_group=one,
            consumer_group=epi_group, tx_count=raw_bytes, tidx=tidx % 32, enable_multicast_signaling=True).make_participants()
        p_prod, p_cons = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.p_mbar.data_ptr(), num_stages=1, producer_group=epi_group, consumer_group=one).make_participants()
        acc_prod, acc_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(), num_stages=2, producer_group=one,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epi_warps)).make_participants()

        # the small per-tile arrays first: they fill the padding before the 1024-byte aligned buffers
        sLse2 = smem.allocate_tensor(Float32, cute.make_layout(S), byte_alignment=16)
        sCorr = smem.allocate_tensor(Float32, cute.make_layout(S), byte_alignment=16)  # dlse - delta
        sKeep = smem.allocate_tensor(Uint8, cute.make_layout(128), byte_alignment=16)
        sA = smem.allocate_tensor(BFloat16, a_layout.outer, byte_alignment=1024, swizzle=a_layout.inner)
        pRaw = smem.allocate_array(BFloat16, cute.cosize(raw_layout.outer), byte_alignment=1024)
        sRaw = cute.make_tensor(cute.recast_ptr(pRaw, swizzle_=raw_layout.inner), raw_layout.outer)
        pB = smem.allocate_array(BFloat16, cute.cosize(bk_layout.outer), byte_alignment=1024)
        sBk = cute.make_tensor(cute.recast_ptr(pB, swizzle_=bk_layout.inner), bk_layout.outer)
        sBmn = cute.make_tensor(cute.recast_ptr(pB, swizzle_=bmn_layout.inner), bmn_layout.outer)
        pP = smem.allocate_array(BFloat16, cute.cosize(p_layout.outer), byte_alignment=1024)
        sP = cute.make_tensor(cute.recast_ptr(pP, swizzle_=p_layout.inner), p_layout.outer)
        sC = smem.allocate_tensor(BFloat16, epi_layout.outer, byte_alignment=1024, swizzle=epi_layout.inner)

        gQK = cute.local_tile(mQK, (128, KBG), (None, None, None, None))
        gV = cute.local_tile(mV, (128, KBG), (None, None, None, None))
        gK = cute.local_tile(mK, (128, KB), (None, None, None, None))
        gDAk = cute.local_tile(mDAk, (S, KBG), (None, None, None, None, None))
        gDBk = cute.local_tile(mDBk, (S, KBG), (None, None, None, None, None))
        gDAmn = cute.local_tile(mDAmn, (S, KBG), (None, None, None, None, None))
        gDBmn = cute.local_tile(mDBmn, (S, KBG), (None, None, None, None, None))
        thr_kk = tm_kk.get_slice(0)
        thr_kmn = tm_kmn.get_slice(0)
        tQKs, tQKg = cpasync.tma_partition(tma_qk, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(thr_kk.partition_A(gQK), 0, 3))
        tVs, tVg = cpasync.tma_partition(tma_v, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(thr_kk.partition_A(gV), 0, 3))
        tKs, tKg = cpasync.tma_partition(tma_k, 0, cute.make_layout(1), cute.group_modes(sRaw, 0, 3), cute.group_modes(thr_kk.partition_A(gK), 0, 3))
        tDAks, tDAkg = cpasync.tma_partition(tma_dak, 0, cute.make_layout(1), cute.group_modes(sBk, 0, 3), cute.group_modes(thr_kk.partition_B(gDAk), 0, 3))
        tDBks, tDBkg = cpasync.tma_partition(tma_dbk, 0, cute.make_layout(1), cute.group_modes(sBk, 0, 3), cute.group_modes(thr_kk.partition_B(gDBk), 0, 3))
        tDAmns, tDAmng = cpasync.tma_partition(tma_damn, 0, cute.make_layout(1), cute.group_modes(sBmn, 0, 3), cute.group_modes(thr_kmn.partition_B(gDAmn), 0, 3))
        tDBmns, tDBmng = cpasync.tma_partition(tma_dbmn, 0, cute.make_layout(1), cute.group_modes(sBmn, 0, 3), cute.group_modes(thr_kmn.partition_B(gDBmn), 0, 3))
        tCrA = tm_kk.make_fragment_A(sA)
        tCrBk = tm_kk.make_fragment_B(sBk)
        tCrP = tm_kmn.make_fragment_A(sP)
        tCrBmn = tm_kmn.make_fragment_B(sBmn)
        acc_fake = tm_kk.make_fragment_C(cute.append(tm_kk.partition_shape_C((128, S)), 2))
        # output pieces (128, 32, L/128, cols/32, K, B)
        gOQK = cute.local_tile(mDQK, (128, 32), (None, None, None, None))
        gOK = cute.local_tile(mDK, (128, 32), (None, None, None, None))
        gOV = cute.local_tile(mDV, (128, 32), (None, None, None, None))
        tOQKs, tOQKg = cpasync.tma_partition(tma_oqk, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), cute.group_modes(gOQK, 0, 2))
        tOKs, tOKg = cpasync.tma_partition(tma_ok, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), cute.group_modes(gOK, 0, 2))
        tOVs, tOVg = cpasync.tma_partition(tma_ov, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), cute.group_modes(gOV, 0, 2))

        if warp == self.tma_warp:
            for t in cutlass.range(bidx, num_tiles, nctas):
                tb, u, k, b = self._decode(t, nu)
                if True:
                    row_block = u * tiles_per_chunk + tb
                    # the first raw key blocks (the transform runs beside the dP^T GEMM), then the dP^T operands,
                    # then the remaining raw blocks (the ring is shallow: do not stall the GEMM loads behind them)
                    raw_first = min(self.raw_stages, self.kb_s)
                    for kb in cutlass.range_constexpr(raw_first):
                        hr = raw_prod.acquire_and_advance()
                        cute.copy(tma_k, tKg[(None, row_block, kb, k, b)], tKs[(None, hr.index)], tma_bar_ptr=hr.barrier)
                    for kb in cutlass.range(kb_total):
                        ha = a_prod.acquire_and_advance()
                        hb = b_prod.acquire_and_advance()
                        if kb < self.kbg_s:
                            cute.copy(tma_qk, tQKg[(None, row_block, kb, k, b)], tQKs[(None, ha.index)], tma_bar_ptr=ha.barrier)
                            cute.copy(tma_dak, tDAkg[(None, 0, kb, u, k, b)], tDAks[(None, hb.index)], tma_bar_ptr=hb.barrier)
                        else:
                            cute.copy(tma_v, tVg[(None, row_block, kb - self.kbg_s, k, b)], tVs[(None, ha.index)], tma_bar_ptr=ha.barrier)
                            cute.copy(tma_dbk, tDBkg[(None, 0, kb - self.kbg_s, u, k, b)], tDBks[(None, hb.index)], tma_bar_ptr=hb.barrier)
                    for kb in cutlass.range_constexpr(raw_first, self.kb_s):
                        hr = raw_prod.acquire_and_advance()
                        cute.copy(tma_k, tKg[(None, row_block, kb, k, b)], tKs[(None, hr.index)], tma_bar_ptr=hr.barrier)
                    for kb in cutlass.range(self.kbg_s):
                        hb = b_prod.acquire_and_advance()
                        cute.copy(tma_damn, tDAmng[(None, 0, kb, u, k, b)], tDAmns[(None, hb.index)], tma_bar_ptr=hb.barrier)
                    for nt in cutlass.range(self.n_vtiles):
                        for kb in cutlass.range(self.kbg_s):
                            hb = b_prod.acquire_and_advance()
                            cute.copy(tma_dbmn, tDBmng[(None, nt, kb, u, k, b)], tDBmns[(None, hb.index)], tma_bar_ptr=hb.barrier)
            raw_prod.tail()
            a_prod.tail()
            b_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
            kpb = cute.size(tCrA, mode=[2])  # k steps per operand block
            for t in cutlass.range(bidx, num_tiles, nctas):
                tb, u, k, b = self._decode(t, nu)
                if True:
                    # dP^T
                    e_dp = acc_prod.acquire_and_advance()
                    acc_dp = acc[(None, None, None, e_dp.index)]
                    for kb in cutlass.range(kb_total):
                        ha = a_cons.wait_and_advance()
                        hb = b_cons.wait_and_advance()
                        tm_kk.set(tcgen05.Field.ACCUMULATE, kb > 0)
                        for kk in cutlass.range_constexpr(kpb):
                            cute.gemm(tm_kk, acc_dp, tCrA[(None, None, kk, ha.index)], tCrBk[(None, None, kk, hb.index)], acc_dp)
                            tm_kk.set(tcgen05.Field.ACCUMULATE, True)
                        ha.release()
                        hb.release()
                    e_dp.commit()
                    # dqk = P^T dA, then the dv tiles = P^T dB
                    hp = p_cons.wait_and_advance()
                    for g in cutlass.range(1 + self.n_vtiles):
                        e_o = acc_prod.acquire_and_advance()
                        acc_g = acc[(None, None, None, e_o.index)]
                        for kb in cutlass.range(self.kbg_s):
                            hb = b_cons.wait_and_advance()
                            tm_kmn.set(tcgen05.Field.ACCUMULATE, kb > 0)
                            for kk in cutlass.range_constexpr(kpb):
                                cute.gemm(tm_kmn, acc_g, tCrP[(None, None, kb * kpb + kk, 0)], tCrBmn[(None, None, kk, hb.index)], acc_g)
                                tm_kmn.set(tcgen05.Field.ACCUMULATE, True)
                            hb.release()
                        e_o.commit()
                    hp.release()
            acc_prod.tail()
        elif warp < self.mma_warp:
            tmem.allocate(self.tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
            grp = warp // 4  # epilogue group
            trow = tidx % 128  # this thread's row of the tile (= TMEM lane)
            cs = tidx % 8  # transform: 8 columns 8 cs .. 8 cs + 7 of each K block, rows rq + 16 r, r in the group's half
            rq = (tidx // 8) % 16
            copy_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE), Float32)
            acc_r = [cute.flat_divide(acc[((None, None), 0, 0, i)], (128, 32)) for i in range(2)]
            t2r = [tcgen05.make_tmem_copy(copy_atom, acc_r[i][(None, None, 0, 0)]) for i in range(2)]
            tTR = [t2r[i].get_slice(trow).partition_S(acc_r[i]) for i in range(2)]
            tTR = [cute.group_modes(tTR[i], 3, cute.rank(tTR[i])) for i in range(2)]  # (.., piece)
            regs = cute.make_rmem_tensor(t2r[0].get_slice(trow).partition_D(cute.make_identity_tensor((128, 32))).shape, Float32)
            rbf = cute.make_rmem_tensor(regs.shape, BFloat16)
            r2s = cute.make_tiled_copy_D(sm100_utils.get_smem_store_op(utils.LayoutEnum.ROW_MAJOR, BFloat16, Float32, t2r[0]), t2r[0])
            tRS_sC = r2s.get_slice(trow).partition_D(sC)
            tRS_r = r2s.retile(rbf)
            epi_bar = pipeline.NamedBarrier(barrier_id=4 + grp, num_threads=128)
            epi_pipe = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stages, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128))
            n_epi = Int32(0)
            pv = cute.make_rmem_tensor((8,), Float32)
            for t in cutlass.range(bidx, num_tiles, nctas):
                tb, u, k, b = self._decode(t, nu)
                row_block = u * tiles_per_chunk + tb
                grow = row_block * 128 + trow  # sequence row
                if True:
                    # per-column constants and the row keep flags of this tile
                    for s in cutlass.range(tidx, S, 32 * self.num_epi_warps):
                        sLse2[s] = _combine_lse2(mPart, s, u, k, b, self.R)
                        sCorr[s] = mDlse[s, u, k, b] - mDelta[s, u, k, b]
                    keep = Uint8(1)
                    if const_expr(self.has_seg):
                        seg_row = cute.make_tensor(pSeg + (b * seg_stride + seg_off + grow), cute.make_layout(1))[0]
                        seg_last = cute.make_tensor(pSeg + (b * seg_stride + seg_off + u * self.C + self.C - 1), cute.make_layout(1))[0]
                        keep = cutlass.select_(seg_row != seg_last, Uint8(0), Uint8(1))
                    sKeep[trow] = keep
                    cute.arch.barrier(barrier_id=3, number_of_threads=32 * self.num_epi_warps)
                    # ---- transform: P^T[t, s] = exp2(k log2e - lse2[s]) into the resident A operand ----
                    hp = p_prod.acquire_and_advance()
                    for kb in cutlass.range(self.kb_s):
                        hr = raw_cons.wait_and_advance()
                        lse8 = [sLse2[kb * KB + 8 * cs + e] for e in range(8)]
                        words = [cute.make_tensor(cute.recast_ptr(
                            (pRaw + (hr.index * (128 * KB) + 64 * (rq + 16 * (4 * grp + r)) + 8 * (cs ^ (rq % 8)))).align(16), dtype=Uint32),
                            cute.make_layout(4)).load() for r in range(4)]
                        for r in cutlass.range_constexpr(4):
                            row = rq + 16 * (4 * grp + r)
                            for e2 in cutlass.range_constexpr(4):
                                x0 = _u32_as_f32(words[r][e2] << 16)
                                x1 = _u32_as_f32(words[r][e2] & Uint32(0xFFFF0000))
                                y0, y1 = cute.arch.fma_packed_f32x2((x0, x1), (LOG2E, LOG2E), (-lse8[2 * e2], -lse8[2 * e2 + 1]))
                                pv[2 * e2] = cute.math.exp2(y0, fastmath=True)
                                pv[2 * e2 + 1] = cute.math.exp2(y1, fastmath=True)
                            if sKeep[row] == 0:
                                pv.fill(0.0)
                            cute.make_tensor((pP + (kb * (128 * KB) + 64 * row + 8 * (cs ^ (rq % 8)))).align(16),
                                             cute.make_layout(8)).store(pv.load().to(BFloat16))
                        cute.arch.fence_view_async_shared()
                        cute.arch.sync_warp()
                        hr.release()
                    cute.arch.fence_view_async_shared()
                    hp.commit()
                    # the dk epilogue reads rows of P^T written by other warps
                    cute.arch.barrier(barrier_id=3, number_of_threads=32 * self.num_epi_warps)
                    # ---- epilogues: dk from the dP^T region, then dqk and the dv tiles ----
                    for g in cutlass.range(2 + self.n_vtiles):
                        f = acc_cons.wait_and_advance()
                        for ch2 in cutlass.range_constexpr(S // 32 // self.epi_groups):
                            ch = ch2 * self.epi_groups + grp  # this group's pieces
                            if f.index == 0:
                                cute.copy(t2r[0], tTR[0][(None, None, None, ch)], regs)
                            else:
                                cute.copy(t2r[1], tTR[1][(None, None, None, ch)], regs)
                            if g == 0:  # dk = P^T (dP^T + dlse - delta)
                                for q8 in cutlass.range_constexpr(4):
                                    s0 = ch * 32 + q8 * 8
                                    p8 = cute.make_tensor((pP + _kmajor_off(trow, s0, 128)).align(16), cute.make_layout(8)).load()
                                    for e in cutlass.range_constexpr(8):
                                        regs[q8 * 8 + e] = p8[e].to(Float32) * (regs[q8 * 8 + e] + sCorr[s0 + e])
                            rbf.store(regs.load().to(BFloat16))
                            buf = grp * self.epi_stages + n_epi % self.epi_stages
                            cute.copy(r2s, tRS_r, tRS_sC[(None, None, None, buf)])
                            cute.arch.fence_view_async_shared()
                            epi_bar.arrive_and_wait()
                            if warp % 4 == 0:
                                if g == 0:
                                    cute.copy(tma_ok, tOKs[(None, buf)], tOKg[(None, row_block, ch, k, b)])
                                elif g == 1:
                                    cute.copy(tma_oqk, tOQKs[(None, buf)], tOQKg[(None, row_block, ch, k, b)])
                                else:
                                    cute.copy(tma_ov, tOVs[(None, buf)], tOVg[(None, row_block, (g - 2) * (S // 32) + ch, k, b)])
                                epi_pipe.producer_commit()
                                epi_pipe.producer_acquire()
                            epi_bar.arrive_and_wait()
                            n_epi += 1
                        with cute.arch.elect_one():
                            f.release()
                    # sLse2 / sCorr / sKeep are rewritten for the next tile after everyone is done with them
                    cute.arch.barrier(barrier_id=3, number_of_threads=32 * self.num_epi_warps)
            epi_pipe.producer_tail()
            tmem.relinquish_alloc_permit()
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_epi_warps)
            tmem.free(tmem_ptr)


# ======================================================================================================
# Host side
# ======================================================================================================

# ======================================================================================================
# Hopper (no tcgen05): AwmDmoGemm, AwmRecurrenceBwd and AwmChunkBwd on cuBLAS / torch, same numerics
# (bf16 operands, fp32 accumulation); the delta, lse chain and zero-tail kernels are shared
# ======================================================================================================


def _group_rows(t, K, nc, C):
    """[B, L, H, D] -> [(b, k, c), (head of the group, row of the chunk), D] (contiguous)."""
    B, L, H, D = t.shape
    return t.view(B, nc, C, K, H // K, D).permute(0, 3, 1, 4, 2, 5).reshape(B * K * nc, (H // K) * C, D)


def _dmo_cublas(q, dout, out_mask, K, S, V, C):
    """dMo[b, k, c] = sum over the heads of the group and the kept rows of chunk c of q^T dOut (fp32)."""
    B, L = q.shape[0], q.shape[1]
    nc = L // C
    dm = dout.masked_fill(out_mask.view(B, L, 1, 1), 0.0)
    return torch.bmm(_group_rows(q, K, nc, C).transpose(1, 2), _group_rows(dm, K, nc, C),
                     out_dtype=torch.float32).view(B, K, nc, S, V)


def _fwd_mod():
    """The forward module (package import, or the copy loaded by path for the tests)."""
    return _FWD


def _dmo_sm90(q, dout, out_mask, H, K, S, V, C):
    """AwmDmoGemm on Hopper (AwmChunkGemmSm90 'dmo'): the rows kept by the output mask are a prefix of each chunk."""
    B, L = q.shape[0], q.shape[1]
    nc = L // C
    dev = q.device
    fwd = _fwd_mod()
    hi = (~out_mask).view(B, nc, C).sum(-1).to(torch.int32)
    lo = torch.zeros_like(hi)
    dMo = torch.zeros(B, K, nc, S, V, dtype=torch.float32, device=dev)
    n_rep = H // K
    splits = fwd._splitk_splits(B * K * nc * (S // 128) * (V // min(256, V)), n_rep * (C // 64), dev)
    sm90 = fwd._sm90()
    f = _compiled(("dmo90", K, S, V, C, n_rep, splits), lambda: sm90.AwmChunkGemmSm90("dmo", K, S, V, C, n_rep, splits),
                  lambda: (_fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Int32, 4), _fake(Int32, 4),
                           _fake(Float32), _fake(Float32), *([Int32(0)] * 6)))
    f(q.reshape(-1), dout.reshape(-1), dout.reshape(-1), lo.reshape(-1), hi.reshape(-1), dMo.reshape(-1)[:1],
      dMo.reshape(-1), B, L, nc, 0, 0, nc)
    return dMo


def _recurrence_bwd_torch(w, dMo, dmem, K, S, V, C, has_seg):
    """AwmRecurrenceBwd (training case: no previous chunk): dB16 = r G per update, dr (full sums over V, as one
    v tile [B, K, nu, 1, S]) and the final G (the input memory's gradient), backwards over the updates."""
    B, nc = dMo.shape[0], dMo.shape[2]
    nu = max(w["nu"], 1)
    dev = dMo.device
    A16, R, D = w["A16"], w["R"], w["D"]
    G = dmem.float().clone() if dmem is not None else torch.zeros(B, K, S, V, dtype=torch.float32, device=dev)
    dB16 = torch.empty(B, K, nu, S, V, dtype=torch.bfloat16, device=dev)
    drp = torch.empty(B, K, nu, 1, S, dtype=torch.float32, device=dev)
    ends = w["seg"][:, C - 1::C][:, :nc] if has_seg else None  # segment id at the end of each chunk
    for u in reversed(range(w["nu"])):
        c = u + 1
        r = R[:, :, u].unsqueeze(-1)
        drp[:, :, u, 0] = (G * D[:, :, u]).sum(-1)
        rG = (r * G).to(torch.bfloat16)
        dB16[:, :, u] = rG
        X = torch.bmm(A16[:, :, u].reshape(B * K, S, S).transpose(1, 2), rG.view(B * K, S, V), out_dtype=torch.float32)
        G = (1 - r) * G - X.view(B, K, S, V)
        if has_seg and c >= 2:  # memory reset before update u: no gradient to the state before it
            G = G.masked_fill((ends[:, c - 2] != ends[:, c - 1]).view(B, 1, 1, 1), 0.0)
        G = G + dMo[:, :, c]
    return dB16, drp, G


def _recurrence_bwd_warp(w, dMo, dmem, K, S, V, C, has_seg):
    """AwmRecurrenceBwdWarp (warp MMAs; training case): dB16, the dr partials per 64-column v tile, the final G."""
    B, nc = dMo.shape[0], dMo.shape[2]
    nu = max(w["nu"], 1)
    dev = dMo.device
    # v tile as in the forward recurrence: the widest that still gives >= 128 CTAs (the updates are sequential)
    VT = next((vt for vt in (64, 32, 16) if V % vt == 0 and B * K * (V // vt) >= 128), 16)
    n_vt = V // VT
    dB16 = torch.empty(B, K, nu, S, V, dtype=torch.bfloat16, device=dev)
    drp = torch.empty(B, K, nu, n_vt, S, dtype=torch.float32, device=dev)
    gmem = torch.empty(B, K, S, V, dtype=torch.float32, device=dev)
    if w["nu"] == 0:
        drp.zero_()
        gmem.copy_(dmem.float() if dmem is not None else dMo[:, :, 0] * 0)
        return dB16, drp, gmem
    a16t = w["A16"].transpose(-1, -2).contiguous()
    has_dmem = dmem is not None
    sm90 = _fwd_mod()._sm90()
    f = _compiled(("recbw", K, S, V, C, has_seg, has_dmem, VT), lambda: sm90.AwmRecurrenceBwdWarp(K, S, V, C, has_seg, has_dmem, VT),
                  lambda: (_fake(BFloat16), _fake(Float32), _fake(Float32), _fake(Float32), _fake(Int64, 8),
                           _fake(Float32), _fake(BFloat16), _fake(Float32), _fake(Float32), *([Int32(0)] * 4)))
    f(a16t.reshape(-1), dMo.reshape(-1), w["R"].reshape(-1), w["D"].reshape(-1), w["seg"].reshape(-1),
      (dmem.float().contiguous() if has_dmem else gmem).reshape(-1), dB16.reshape(-1), drp.reshape(-1),
      gmem.reshape(-1), B, nc, w["nu"], w["seg_stride"])
    return dB16, drp, gmem


def _chunk_bwd_cublas(w, dA16, dB16, delta, dlse, qk, k, v, dqk, dk, dv, K, S, V, C, has_seg):
    """AwmChunkBwd: with P[s, t] = exp2(k[t, s] log2e - lse2[s]) (rows outside the segment of the chunk's last
    row 0), dP = dA qk^T + dB v^T, dk = P (dP - delta + dlse), dqk = P^T dA, dv = P^T dB for the update chunks."""
    B = k.shape[0]
    nu = w["nu"]
    BKn = B * K * nu
    p = w["part"][:, :, :nu]
    m, l = p[..., 0], p[..., 1]
    mm = m.amax(3, keepdim=True)
    lse2 = mm.squeeze(3) + torch.log2((l * torch.exp2(m - mm)).sum(3))  # [B, K, nu, S]
    kc = k[:, :nu * C].view(B, nu, C, K, S).permute(0, 3, 1, 4, 2)  # [B, K, nu, S, C]
    e = torch.exp2(kc.float() * LOG2E - lse2.unsqueeze(-1))
    if has_seg:
        sg = w["seg"][:, :nu * C].view(B, nu, C)
        e = torch.where((sg == sg[:, :, -1:]).view(B, 1, nu, 1, C), e, 0.0)
    e = e.reshape(BKn, S, C)
    pt = e.to(torch.bfloat16)
    rows = lambda t, D: t[:, :nu * C].view(B, nu, C, K, D).permute(0, 3, 1, 2, 4).reshape(BKn, C, D)
    dA, dB = dA16.reshape(BKn, S, S), dB16.reshape(BKn, S, V)
    dP = torch.bmm(dA, rows(qk, S).transpose(1, 2), out_dtype=torch.float32)
    dP.add_(torch.bmm(dB, rows(v, V).transpose(1, 2), out_dtype=torch.float32))
    dkT = e * (dP - (delta - dlse).reshape(BKn, S, 1))
    back = lambda t, D: t.view(B, K, nu, C, D).permute(0, 2, 3, 1, 4).reshape(B, nu * C, K, D)
    dk[:, :nu * C] = back(dkT.transpose(1, 2), S)
    dqk[:, :nu * C] = back(torch.bmm(pt.transpose(1, 2), dA), S)
    dv[:, :nu * C] = back(torch.bmm(pt.transpose(1, 2), dB), V)


def _chunk_bwd_sm90(w, dA16, dB16, delta, dlse, qk, k, v, dqk, dk, dv, K, S, V, C, has_seg, dv_accumulate=False):
    """AwmChunkBwd on Hopper (AwmChunkBwdSm90): rows of each update chunk outside the segment of its last row drop."""
    B, L = k.shape[0], k.shape[1]
    nu = w["nu"]
    dev = k.device
    fwd = _fwd_mod()
    lse2 = fwd._lse2_of(w["part"], 0, nu)
    lo, _ = fwd._chunk_row_bounds(w["seg"], 0, B, nu, C, has_seg, dev)
    sm90 = fwd._sm90()
    c2 = (dlse - delta).contiguous()
    f = _compiled(("chunkb90", K, S, V, C, dv_accumulate), lambda: sm90.AwmChunkBwdSm90(K, S, V, C, dv_accumulate), lambda: (
        *[_fake(BFloat16) for _ in range(5)], _fake(Float32), _fake(Float32), _fake(Int32, 4),
        *[_fake(BFloat16) for _ in range(3)], *([Int32(0)] * 3)))
    f(k.reshape(-1), qk.reshape(-1), v.reshape(-1), dA16.reshape(-1), dB16.reshape(-1), lse2.reshape(-1),
      c2.reshape(-1), lo.reshape(-1), dk.reshape(-1), dqk.reshape(-1), dv.reshape(-1), B, L, nu)


def _phase3_bwd(dout, q, w, H, K, S, V, C, has_prev, has_seg):
    """dq [B, L, H, S] (bf16), the output row mask [B, L] and dMo [B, K, nc, S, V] (fp32) from dOut and q
    ([B, L, H, *], contiguous) and the forward workspaces ``w`` (_phases12 with save)."""
    B, L = q.shape[0], q.shape[1]
    nc = L // C
    dev = q.device
    # dq = dOut M_c^T: the output GEMM with the roles of S and V swapped (M_c as the K-major [S, V] operand)
    dq, out_mask = _out_gemm(dout, w["Mseq"], w["seg"], w["prev_cnt"], w["dead"], w["seg_off"], w["seg_stride"],
                             H, K, V, S, C, has_prev, has_seg)
    if _is_hopper(dev):
        if _fwd_mod()._AWM_SM90_CUBLAS:
            return dq, out_mask, _dmo_cublas(q, dout, out_mask, K, S, V, C)
        return dq, out_mask, _dmo_sm90(q, dout, out_mask, H, K, S, V, C)
    dMo = torch.zeros(B, K, nc, S, V, dtype=torch.float32, device=dev)
    base = B * K * nc * (S // 128) * chunk_gemm_tiling(0, V)[2]
    k_tiles = (H // K) * (C // DMO_KB)
    splits = max(1, min(k_tiles, round(_sm_count(dev) / base)))
    f = _compiled(("dmo", H, K, S, V, C, splits), lambda: AwmDmoGemm(H, K, S, V, C, splits), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(_U8, 1), _fake(Float32), Int32(0), Int32(0)))
    f(q.reshape(-1), dout.reshape(-1), out_mask.view(torch.uint8).reshape(-1), dMo.reshape(-1), B, L)
    return dq, out_mask, dMo


def _phase2_bwd(w, dMo, dmem, dlnt, K, S, V, C, has_prev, has_seg):
    """Reverse recurrence and the lse chain from the forward workspaces ``w`` (saved) and dMo [B, K, nc, S, V]:
    dB16 / dA16 [B, K, nu, S, *] (bf16), delta and dlse [B, K, nu, S] (fp32), and the gradients of the
    input memory [B, K, S, V] (fp32) and log normalizer [B, K, S]. ``dmem`` / ``dlnt`` may be None."""
    B, nc = dMo.shape[0], dMo.shape[2]
    nu = max(w["nu"], 1)
    dev = dMo.device
    VT = 64
    n_vt = V // VT
    has_dmem, has_dlnt = dmem is not None, dlnt is not None
    if w["nu"] == 0:
        # no update chunk (a single chunk and no previous chunk): nothing flows through the recurrence. The
        # recurrence kernel would still walk one update (reading dMo past the last chunk). The per-update
        # outputs are unused (the chunk backward is skipped too); the memory used by chunk 0 is the input memory.
        dB16 = torch.zeros(B, K, nu, S, V, dtype=torch.bfloat16, device=dev)
        dA16 = torch.zeros(B, K, nu, S, S, dtype=torch.bfloat16, device=dev)
        delta = torch.zeros(B, K, nu, S, dtype=torch.float32, device=dev)
        dlse = torch.zeros(B, K, nu, S, dtype=torch.float32, device=dev)
        gmem = dMo[:, :, 0].float() + (dmem.float() if has_dmem else 0)
        glnt = dlnt.float().clone() if has_dlnt else torch.zeros(B, K, S, dtype=torch.float32, device=dev)
        return dB16, dA16, delta, dlse, gmem, glnt
    if _is_hopper(dev) and (_fwd_mod()._AWM_SM90_CUBLAS or has_prev):
        n_vt = 1  # dr written as full sums over V
        dB16, drp, gmem = _recurrence_bwd_torch(w, dMo, dmem, K, S, V, C, has_seg)
    elif _is_hopper(dev):
        dB16, drp, gmem = _recurrence_bwd_warp(w, dMo, dmem, K, S, V, C, has_seg)
        n_vt = drp.shape[3]
    else:
        dB16, drp, gmem = _recurrence_bwd_sm100(w, dMo, dmem, K, S, V, C, has_prev, has_seg, n_vt)
    # dA_u = -(rG) M^T with M the state update u starts from (Mneg = -M, 0 after a reset): off the sequential chain
    dA16 = torch.matmul(dB16, w["Mneg"].transpose(-1, -2))
    delta = torch.empty(B, K, nu, S, dtype=torch.float32, device=dev)
    f_delta = _compiled(("delta", S, V), lambda: AwmBwdDelta(S, V), lambda: (
        _fake(BFloat16), _fake(Float32), _fake(BFloat16), _fake(Float32), Int32(0)))
    f_delta(dA16.reshape(-1), w["AB"].reshape(-1), dB16.reshape(-1), delta.reshape(-1), B * K * nu * S)
    dlse = torch.empty(B, K, nu, S, dtype=torch.float32, device=dev)
    glnt = torch.empty(B, K, S, dtype=torch.float32, device=dev)
    f_chain = _compiled(("chain", K, S, C, n_vt, has_prev, has_seg, has_dlnt), lambda: AwmLseChain(K, S, C, n_vt, has_prev, has_seg, has_dlnt),
                        lambda: (_fake(Float32), _fake(Float32), _fake(Int64, 8), _fake(Int64, 8), _fake(Float32), _fake(Float32),
                                 _fake(Float32), *([Int32(0)] * 4)))
    f_chain(drp.reshape(-1), w["R"].reshape(-1), w["seg"].reshape(-1), w["prev_cnt"].reshape(-1),
            (dlnt.float().contiguous() if has_dlnt else _dummy(torch.float32, dev)).reshape(-1), dlse.reshape(-1), glnt.reshape(-1),
            B, nu, w["seg_off"], w["seg_stride"])
    return dB16, dA16, delta, dlse, gmem, glnt


def _recurrence_bwd_sm100(w, dMo, dmem, K, S, V, C, has_prev, has_seg, n_vt):
    """AwmRecurrenceBwd (tcgen05): dB16 [B, K, nu, S, V], the dr partials [B, K, nu, n_vt, S] and the final G."""
    B, nc = dMo.shape[0], dMo.shape[2]
    nu = max(w["nu"], 1)
    dev = dMo.device
    has_dmem = dmem is not None
    dB16 = torch.empty(B, K, nu, S, V, dtype=torch.bfloat16, device=dev)
    drp = torch.empty(B, K, nu, n_vt, S, dtype=torch.float32, device=dev)
    gmem = torch.empty(B, K, S, V, dtype=torch.float32, device=dev)
    f_rec = _compiled(("recb", K, S, V, C, has_prev, has_seg, has_dmem), lambda: AwmRecurrenceBwd(K, S, V, C, has_prev, has_seg, has_dmem),
                      lambda: (_fake(BFloat16), _fake(Float32), _fake(Float32), _fake(Float32), _fake(Int64, 8),
                               _fake(Int64, 8), _fake(Float32), _fake(BFloat16), _fake(Float32), _fake(Float32),
                               *([Int32(0)] * 5)))
    f_rec(w["A16"].reshape(-1), dMo.reshape(-1), w["R"].reshape(-1), w["D"].reshape(-1),
          w["seg"].reshape(-1), w["prev_cnt"].reshape(-1), (dmem.float().contiguous() if has_dmem else _dummy(torch.float32, dev)).reshape(-1),
          dB16.reshape(-1), drp.reshape(-1), gmem.reshape(-1), B, nc, nu, w["seg_off"], w["seg_stride"])
    return dB16, drp, gmem


def _phase1_bwd(w, dA16, dB16, delta, dlse, qk, k, v, K, S, V, C, has_seg, dv_accum=None):
    """dqk, dk [B, L, K, S] and dv [B, L, K, V] (bf16) for the update chunks (rows of the last chunk are 0);
    with ``dv_accum`` (a contiguous bf16 [B, L, K, V] gradient, e.g. the attention's) dv is added into it instead."""
    B, L = k.shape[0], k.shape[1]
    nu = max(w["nu"], 1)
    dev = k.device
    accum = dv_accum is not None
    dqk = torch.empty(B, L, K, S, dtype=torch.bfloat16, device=dev)
    dk = torch.empty(B, L, K, S, dtype=torch.bfloat16, device=dev)
    hopper = _is_hopper(dev)
    if hopper:  # wgmma chunk kernel with TMA reduce-add dv stores
        accum = (dv_accum is not None and w["nu"] > 0 and not _fwd_mod()._AWM_SM90_CUBLAS
                 and min(S, 256) == min(V, 256))
    else:  # Blackwell chunk kernel (reduce-add dv stores)
        accum = dv_accum is not None
    if accum:
        assert dv_accum.shape == (B, L, K, V) and dv_accum.dtype == torch.bfloat16 and dv_accum.is_contiguous()
        dv = dv_accum
    else:
        dv = torch.empty(B, L, K, V, dtype=torch.bfloat16, device=dev)
    f_zero = _compiled(("zerotail", K, S, V, C, not accum), lambda: AwmZeroTail(K, S, V, C, not accum), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), Int32(0), Int32(0)))
    f_zero(dqk.reshape(-1), dk.reshape(-1), dv.reshape(-1), B, L)  # the last chunk has no update
    if w["nu"] > 0 and hopper:
        if _fwd_mod()._AWM_SM90_CUBLAS or min(S, 256) != min(V, 256):
            _chunk_bwd_cublas(w, dA16, dB16, delta, dlse, qk, k, v, dqk, dk, dv, K, S, V, C, has_seg)
        else:
            _chunk_bwd_sm90(w, dA16, dB16, delta, dlse, qk, k, v, dqk, dk, dv, K, S, V, C, has_seg, accum)
    elif w["nu"] > 0:
        f = _compiled(("chunkb", K, S, V, C, w["R_splits"], has_seg, accum),
                      lambda: AwmChunkBwd(K, S, V, C, w["R_splits"], has_seg, accum),
                      lambda: (_fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32),
                               _fake(Float32), _fake(Float32), _fake(Int64, 8), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16),
                               *([Int32(0)] * 6)))
        num_tiles = B * K * nu * (C // 128)
        f(qk.reshape(-1), k.reshape(-1), v.reshape(-1), dA16.reshape(-1), dB16.reshape(-1), w["part"].reshape(-1), delta.reshape(-1),
          dlse.reshape(-1), w["seg"].reshape(-1), dqk.reshape(-1), dk.reshape(-1), dv.reshape(-1), B, L, nu, w["seg_off"], w["seg_stride"],
          min(num_tiles, _sm_count(dev)))
    if dv_accum is not None and not accum:  # Hopper cuBLAS / mixed-size paths: a separate add
        dv = torch.add(dv_accum, dv, out=dv_accum)
    return dqk, dk, dv


def awm_bwd_prepare(query, qkey, key, value, chunk_size, segment_idx=None, recompute_out=False, workspace=None):
    """First half of the backward (training case: no input memory / previous chunk): the forward's phases 1
    and 2 with the save option, or the ``workspace`` the forward returned (return_workspace). Returns the state
    for awm_bwd_finish and, with ``recompute_out``, the forward output [B, H, L, V] (a view of [B, L, H, V]
    storage), else None."""
    B, H, L, S = query.shape
    K, V = value.shape[1], value.shape[3]
    has_seg = segment_idx is not None
    if workspace is None:
        w = _phases12(qkey, key, value, segment_idx, None, None, None, None, None, None, chunk_size, save=True)
    else:
        dev = query.device
        w = dict(zip(WORKSPACE_KEYS, workspace))
        seg = segment_idx.contiguous() if has_seg else _dummy(torch.int64, dev)
        w.update(R_splits=min(8, chunk_size // 128), nu=L // chunk_size - 1, seg=seg, prev_cnt=_dummy(torch.int64, dev),
                 seg_off=0, seg_stride=seg.shape[1] if has_seg else 0)
    state = dict(w=w, q=_blhs(query), qk=_blhs(qkey), k=_blhs(key), v=_blhs(value), H=H, K=K, S=S, V=V, C=chunk_size,
                 has_seg=has_seg)
    out = None
    if recompute_out:
        out, _ = _out_gemm(state["q"], w["MT"], w["seg"], w["prev_cnt"], w["dead"], w["seg_off"], w["seg_stride"], H, K, S, V,
                           chunk_size, False, has_seg)
        out = out.transpose(1, 2)
    return state, out


def awm_bwd_finish(state, out_grad, memory_grad=None, log_norm_term_grad=None, dv_accum=None):
    """Second half: phases 3', 2' and 1'. Returns dq [B, H, L, S], dqk / dk [B, K, L, S], dv [B, K, L, V] (bf16,
    views of [B, L, *, D] storage) and None, None for the gradients of the absent input memory / normalizer.
    ``dv_accum`` (contiguous bf16 [B, L, K, V]): dv is added into it in the kernel's store (the returned dv is its
    view) instead of being written to a new tensor."""
    w, H, K, S, V, C, has_seg = (state[n] for n in ("w", "H", "K", "S", "V", "C", "has_seg"))
    dout = _blhs(out_grad.to(torch.bfloat16))
    dq, out_mask, dMo = _phase3_bwd(dout, state["q"], w, H, K, S, V, C, False, has_seg)
    dB16, dA16, delta, dlse, _, _ = _phase2_bwd(w, dMo, memory_grad, log_norm_term_grad, K, S, V, C, False, has_seg)
    dqk, dk, dv = _phase1_bwd(w, dA16, dB16, delta, dlse, state["qk"], state["k"], state["v"], K, S, V, C, has_seg, dv_accum)
    return dq.transpose(1, 2), dqk.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2), None, None


def adaptive_working_memory_bwd(out_grad, memory_grad, log_norm_term_grad, query, qkey, key, value, chunk_size,
                                segment_idx=None):
    """Backward of adaptive_working_memory_fwd for the training case without an input memory / previous chunk:
    out_grad [B, H, L, V], memory_grad [B, K, S, V] / log_norm_term_grad [B, K, S] (optional), the inputs as in
    the forward. Returns dq [B, H, L, S], dqk / dk [B, K, L, S], dv [B, K, L, V] (bf16, views of [B, L, *, D]
    storage), and the gradients of the (absent) input memory and log normalizer as None.

    The forward's phases 1 and 2 are recomputed with the save option (cheaper than keeping the workspaces)."""
    state, _ = awm_bwd_prepare(query, qkey, key, value, chunk_size, segment_idx)
    return awm_bwd_finish(state, out_grad, memory_grad, log_norm_term_grad)
