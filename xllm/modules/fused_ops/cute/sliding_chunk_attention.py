"""Sliding chunk attention (SCA) with CuTe DSL kernels for Blackwell (SM100).

Queries of chunk c (rows [cC, (c+1)C)) attend causally to the keys of chunks c - 1 and c (chunk 0 only to itself),
restricted to the query's own segment (segment ids are nondecreasing, so a row's valid keys are the range
[segment start, row]). Nothing is materialized: the forward is a flash-attention style kernel that keeps the
128-row query tile in shared memory, streams 64-key blocks of K / V, runs the online softmax in 4 warps (one row
per thread) and feeds P to the PV MMA from tensor memory. It saves only the output and the row log-sum-exp.

Layouts are those of the fused block: q [B, L, H, S], k [B, L, K, S], v [B, L, K, V] (bf16), y [B, L, H, V].
lse2 [B, H, L] (fp32) is the row log2-sum-exp of the scaled scores (scale * log2 e applied), the form the
backward recomputes P from.

Forward: ScaFwdT keeps the q tile in tensor memory as the A operand of S = Q K^T (full tensor-core rate); for
gekko-4B its 256 TMEM columns let two CTAs share an SM (32-key blocks), the S = 256 shapes use 64-key blocks
with P over the S columns. ScaFwd (q in shared memory) stays as the SCA_FWD_TMEM=0 fallback.
Backward: for S + V <= 256 (gekko-4B) the fused ScaBwdDkDv (dK and dV of a key tile in one pass) and ScaBwdDqT
kernels keep the resident tiles (K, V / Q, dO) in tensor memory as the A operands of every GEMM, so the small-N
GEMMs run at the tensor-core rate instead of the shared-memory operand bandwidth; two groups of compute warps
alternate blocks. The other shapes use ScaBwdDv, ScaBwdDk and ScaBwdDq (both operands in shared memory; for
V > 256 dP comes from cuBLAS fp32 GEMMs). SCA_DBG=1/3 compile measurement variants (no softmax math / MMAs only).
"""

import math
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import os
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05

try:
    from .adaptive_working_memory import _compiled, _fake, _sm_count, NEG_BIG, LOG2E  # noqa: F401
except ImportError:  # loaded from its file (tests)
    import importlib.util as _ilu
    import os as _os
    _spec = _ilu.spec_from_file_location("cute_adaptive_working_memory_fwd",
                                         _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "adaptive_working_memory.py"))
    _fwd = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_fwd)
    _compiled, _fake, _sm_count, NEG_BIG, LOG2E = _fwd._compiled, _fwd._fake, _fwd._sm_count, _fwd.NEG_BIG, _fwd.LOG2E

def _fake2(dtype, align=16):
    """A 2-D fake tensor: element counts beyond the int32 range of the 1-D fake tensors (only its pointer is used)."""
    return cute.runtime.make_fake_compact_tensor(dtype, (cute.sym_int(), cute.sym_int()), assumed_align=align)


def _col2d(t: torch.Tensor) -> torch.Tensor:
    """A contiguous tensor as the column-major 2-D view the 2-D fake tensors expect (pointer only)."""
    return t.reshape(1, -1).t()


RESCALE_THRESHOLD = 8.0  # log2 units: the running max is only moved (and O rescaled) when it grows by more


class ScaFwd:
    """y = softmax(scale q k^T, masked) v for one (batch, head, chunk, 128-row tile, 256-wide v slab) per CTA.

    Warps 0-3: softmax (thread = row = TMEM lane), O rescale and the epilogue; warp 4: tcgen05 MMAs; warp 5: TMA.
    TMEM: S [128 x KB] fp32 x 2 stages, P [128 x KB] bf16 (A operand of the PV MMA), O [128 x VT] fp32.
    Shared memory: the q tile (128 x S), 2 stages of K (KB x S) and V (KB x VT) blocks. For V > 256 the CTAs of
    the V // 256 slabs of a tile each recompute S (its cost is S / V of the PV MMA)."""

    def __init__(self, H: int, K: int, S: int, V: int, C: int, KB: int = 64, kv_stages: int = 2):
        self.H, self.K, self.S, self.V, self.C, self.KB = H, K, S, V, C, KB
        self.n_rep = H // K
        self.TM = 128
        self.VT = min(V, 256)
        self.n_vt = V // self.VT
        self.kv_stages = kv_stages
        self.num_softmax_warps = 4
        self.mma_warp = 4
        self.tma_warp = 5
        self.threads = 32 * 6
        self.row_tiles = C // self.TM
        assert H % K == 0 and S % 64 == 0 and S <= 256 and V % 32 == 0 and V % self.VT == 0 and C % KB == 0 and C % 128 == 0
        self.tmem_s = 0
        self.tmem_p = 2 * KB
        self.tmem_o = 2 * KB + KB // 2
        total = self.tmem_o + self.VT
        self.tmem_cols = max(32, 1 << (total - 1).bit_length())

    @cute.jit
    def __call__(self, tQ: cute.Tensor, tK: cute.Tensor, tV: cute.Tensor, tKmin: cute.Tensor, tY: cute.Tensor,
                 tLse: cute.Tensor, B: Int32, L: Int32, scale2: Float32, stream: cuda.CUstream):
        H, K, S, V, KB, VT = self.H, self.K, self.S, self.V, self.KB, self.VT
        nc = L // self.C
        # q as (M = L, K = S, H, B); k as (N = L, K = S, Kh, B); v as (N = V, K = L, Kh, B) (MN-major B of PV)
        mQ = cute.make_tensor(tQ.iterator, cute.make_layout((L, S, H, B), stride=(H * S, 1, S, L * H * S)))
        mK = cute.make_tensor(tK.iterator, cute.make_layout((L, S, K, B), stride=(K * S, 1, S, L * K * S)))
        mV = cute.make_tensor(tV.iterator, cute.make_layout((V, L, K, B), stride=(1, K * V, V, L * K * V)))

        tm_qk = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, KB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        tm_pv = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, VT, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))
        q_layout = sm100_utils.make_smem_layout_a(tm_qk, (self.TM, KB, S), BFloat16, 1)
        k_layout = sm100_utils.make_smem_layout_b(tm_qk, (self.TM, KB, S), BFloat16, self.kv_stages)
        v_layout = sm100_utils.make_smem_layout_b(tm_pv, (self.TM, VT, KB), BFloat16, self.kv_stages)
        p_layout = sm100_utils.make_smem_layout_a(tm_pv, (self.TM, VT, KB), BFloat16, 1)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_q, mQ_t = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mQ, cute.slice_(q_layout, (None, None, None, 0)), (self.TM, KB, S), tm_qk)
        tma_k, mK_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mK, cute.slice_(k_layout, (None, None, None, 0)), (self.TM, KB, S), tm_qk)
        tma_v, mV_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mV, cute.slice_(v_layout, (None, None, None, 0)), (self.TM, VT, KB), tm_pv)
        num_tiles = B * nc * H * self.row_tiles * self.n_vt
        self.kernel(tm_qk, tm_pv, tma_q, mQ_t, tma_k, mK_t, tma_v, mV_t, tKmin.iterator, tY.iterator, tLse.iterator,
                    q_layout, k_layout, v_layout, p_layout, L, nc, scale2).launch(
            grid=[num_tiles, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tm_qk: cute.TiledMma, tm_pv: cute.TiledMma, tma_q: cute.CopyAtom, mQ: cute.Tensor,
               tma_k: cute.CopyAtom, mK: cute.Tensor, tma_v: cute.CopyAtom, mV: cute.Tensor, pKmin: cute.Pointer,
               pY: cute.Pointer, pLse: cute.Pointer, q_layout: cute.ComposedLayout, k_layout: cute.ComposedLayout,
               v_layout: cute.ComposedLayout, p_layout: cute.ComposedLayout, L: Int32, nc: Int32, scale2: Float32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H, K, S, V, KB, C, VT = self.H, self.K, self.S, self.V, self.KB, self.C, self.VT

        # tile -> (v slab, row tile, head, chunk, batch); the slabs of a tile and the heads of a kv group are
        # adjacent (they share the q tile / the K and V blocks)
        vt = bidx % self.n_vt
        t0 = bidx // self.n_vt
        rt = t0 % self.row_tiles
        t1 = t0 // self.row_tiles
        h = t1 % H
        t2 = t1 // H
        c = t2 % nc
        b = t2 // nc
        kh = h // self.n_rep
        q0 = c * C + rt * self.TM  # first query row of the tile
        kstart = Int32(0)
        if c > 0:
            kstart = (c - 1) * C
        kmin0 = cute.make_tensor(pKmin + (b * L + q0), cute.make_layout(1))[0]
        kb_lo = (kmin0 - kstart) // KB
        if kb_lo < 0:
            kb_lo = Int32(0)
        kb_hi = (q0 + self.TM - 1 - kstart) // KB
        n_blocks = kb_hi - kb_lo + 1
        kblk0 = kstart // KB + kb_lo  # first key block (units of KB rows of the sequence)

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_q)
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_v)

        @cute.struct
        class SharedStorage:
            q_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            k_mbar: cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
            v_mbar: cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
            s_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            p_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_softmax_warps + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        sm_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * self.num_softmax_warps)
        q_bytes = cute.size_in_bytes(BFloat16, cute.select(q_layout, mode=[0, 1, 2]))
        k_bytes = cute.size_in_bytes(BFloat16, cute.select(k_layout, mode=[0, 1, 2]))
        v_bytes = cute.size_in_bytes(BFloat16, cute.select(v_layout, mode=[0, 1, 2]))
        q_prod, q_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.q_mbar.data_ptr(), num_stages=1, producer_group=one, consumer_group=one,
            tx_count=q_bytes).make_participants()
        k_prod, k_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.k_mbar.data_ptr(), num_stages=self.kv_stages, producer_group=one, consumer_group=one,
            tx_count=k_bytes).make_participants()
        v_prod, v_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.v_mbar.data_ptr(), num_stages=self.kv_stages, producer_group=one, consumer_group=one,
            tx_count=v_bytes).make_participants()
        s_prod, s_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.s_mbar.data_ptr(), num_stages=2, producer_group=one,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_softmax_warps)).make_participants()
        p_prod, p_cons = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.p_mbar.data_ptr(), num_stages=1, producer_group=sm_group, consumer_group=one).make_participants()

        sQ = smem.allocate_tensor(BFloat16, q_layout.outer, byte_alignment=1024, swizzle=q_layout.inner)
        sK = smem.allocate_tensor(BFloat16, k_layout.outer, byte_alignment=1024, swizzle=k_layout.inner)
        sV = smem.allocate_tensor(BFloat16, v_layout.outer, byte_alignment=1024, swizzle=v_layout.inner)

        gQ = cute.local_tile(mQ, (self.TM, S), (None, None, None, None))  # (128, S, L/128, 1, H, B)
        gK = cute.local_tile(mK, (KB, S), (None, None, None, None))  # (KB, S, L/KB, 1, Kh, B)
        gV = cute.local_tile(mV, (VT, KB), (None, None, None, None))  # (VT, KB, V/VT, L/KB, Kh, B)
        thr_qk = tm_qk.get_slice(0)
        thr_pv = tm_pv.get_slice(0)
        tQs, tQg = cpasync.tma_partition(tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(thr_qk.partition_A(gQ), 0, 3))
        tKs, tKg = cpasync.tma_partition(tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(thr_qk.partition_B(gK), 0, 3))
        tVs, tVg = cpasync.tma_partition(tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(thr_pv.partition_B(gV), 0, 3))
        tCrQ = tm_qk.make_fragment_A(sQ)
        tCrK = tm_qk.make_fragment_B(sK)
        tCrV = tm_pv.make_fragment_B(sV)
        s_fake = tm_qk.make_fragment_C(cute.append(tm_qk.partition_shape_C((self.TM, KB)), 2))
        o_fake = tm_pv.make_fragment_C(tm_pv.partition_shape_C((self.TM, VT)))
        tP_fake = cute.make_tensor(cute.make_ptr(BFloat16, 0, cute.AddressSpace.tmem), p_layout.outer)
        tPr_fake = tm_pv.make_fragment_A(tP_fake)[(None, None, None, 0)]

        if warp == self.tma_warp:
            hq = q_prod.acquire_and_advance()
            cute.copy(tma_q, tQg[(None, q0 // self.TM, 0, h, b)], tQs[(None, hq.index)], tma_bar_ptr=hq.barrier)
            for j in cutlass.range(n_blocks):
                hk = k_prod.acquire_and_advance()
                cute.copy(tma_k, tKg[(None, kblk0 + j, 0, kh, b)], tKs[(None, hk.index)], tma_bar_ptr=hk.barrier)
                hv = v_prod.acquire_and_advance()
                cute.copy(tma_v, tVg[(None, vt, kblk0 + j, kh, b)], tVs[(None, hv.index)], tma_bar_ptr=hv.barrier)
            q_prod.tail()
            k_prod.tail()
            v_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tO = cute.make_tensor(tmem_ptr + self.tmem_o, o_fake.layout)
            tPr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_p, dtype=BFloat16), tPr_fake.layout)
            q_cons.wait()
            # S_0
            es = s_prod.acquire_and_advance()
            hk = k_cons.wait_and_advance()
            tS0 = tS[(None, None, None, es.index)]
            for kk in cutlass.range_constexpr(cute.size(tCrQ, mode=[2])):
                tm_qk.set(tcgen05.Field.ACCUMULATE, kk > 0)
                cute.gemm(tm_qk, tS0, tCrQ[(None, None, kk, 0)], tCrK[(None, None, kk, hk.index)], tS0)
            hk.release()
            es.commit()
            for j in cutlass.range(n_blocks):
                if j + 1 < n_blocks:
                    es = s_prod.acquire_and_advance()
                    hk = k_cons.wait_and_advance()
                    tSj = tS[(None, None, None, es.index)]
                    for kk in cutlass.range_constexpr(cute.size(tCrQ, mode=[2])):
                        tm_qk.set(tcgen05.Field.ACCUMULATE, kk > 0)
                        cute.gemm(tm_qk, tSj, tCrQ[(None, None, kk, 0)], tCrK[(None, None, kk, hk.index)], tSj)
                    hk.release()
                    es.commit()
                hp = p_cons.wait_and_advance()
                hv = v_cons.wait_and_advance()
                for kk in cutlass.range_constexpr(cute.size(tPr, mode=[2])):
                    tm_pv.set(tcgen05.Field.ACCUMULATE, (j > 0) | (kk > 0))
                    cute.gemm(tm_pv, tO, tPr[(None, None, kk)], tCrV[(None, None, kk, hv.index)], tO)
                hv.release()
                hp.release()
            q_cons.release()
            s_prod.tail()
        else:
            tmem.allocate(self.tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tO = cute.make_tensor(tmem_ptr + self.tmem_o, o_fake.layout)
            tPr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_p, dtype=BFloat16), tPr_fake.layout)
            trow = tidx % 128
            qpos = q0 + trow
            kmin_row = cute.make_tensor(pKmin + (b * L + qpos), cute.make_layout(1))[0]
            kmin_last = cute.make_tensor(pKmin + (b * L + q0 + self.TM - 1), cute.make_layout(1))[0]
            # S loads: this thread's row of a stage (KB columns)
            ld_s = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(KB), tcgen05.Pack.NONE), Float32)
            tS_r = [cute.flat_divide(tS[((None, None), 0, 0, i)], (self.TM, KB)) for i in range(2)]
            t2r_s = [tcgen05.make_tmem_copy(ld_s, tS_r[i][(None, None, 0, 0)]) for i in range(2)]
            tTR_s = [t2r_s[i].get_slice(trow).partition_S(tS_r[i]) for i in range(2)]
            tTR_s = [cute.group_modes(tTR_s[i], 3, cute.rank(tTR_s[i])) for i in range(2)]
            xs = cute.make_rmem_tensor(t2r_s[0].get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, KB))).shape, Float32)
            # P store: bf16 pairs into the TMEM A operand of the PV MMA
            st_p = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(KB // 2), tcgen05.Unpack.NONE), BFloat16)
            r2t_p = tcgen05.make_tmem_copy(st_p, tPr)
            thr_p = r2t_p.get_slice(trow)
            tRT_tP = thr_p.partition_D(tPr)
            tRT_c = thr_p.partition_S(thr_pv.partition_A(cute.make_identity_tensor((self.TM, KB))))
            pbf = cute.make_rmem_tensor(xs.shape, BFloat16)
            pbf_v = cute.make_tensor(pbf.iterator, cute.make_layout(tRT_c.shape))
            # O pieces of 32 columns: loads (epilogue, rescale) and stores (rescale)
            ld_o = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32), tcgen05.Pack.NONE), Float32)
            st_o = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(32), tcgen05.Unpack.NONE), Float32)
            tO_r = cute.flat_divide(tO[((None, None), 0, 0)], (self.TM, 32))
            t2r_o = tcgen05.make_tmem_copy(ld_o, tO_r[(None, None, 0, 0)])
            r2t_o = tcgen05.make_tmem_copy(st_o, tO_r[(None, None, 0, 0)])
            tTR_o = t2r_o.get_slice(trow).partition_S(tO_r)
            tTR_o = cute.group_modes(tTR_o, 3, cute.rank(tTR_o))
            tRT_o = r2t_o.get_slice(trow).partition_D(tO_r)
            tRT_o = cute.group_modes(tRT_o, 3, cute.rank(tRT_o))
            ov = cute.make_rmem_tensor(t2r_o.get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, 32))).shape, Float32)
            obf = cute.make_rmem_tensor(ov.shape, BFloat16)

            m_run = Float32(NEG_BIG)
            l_run = Float32(0.0)
            for j in cutlass.range(n_blocks):
                bstart = kstart + (kb_lo + j) * KB
                fs = s_cons.wait_and_advance()
                if fs.index == 0:
                    cute.copy(t2r_s[0], tTR_s[0][(None, None, None, 0)], xs)
                else:
                    cute.copy(t2r_s[1], tTR_s[1][(None, None, None, 0)], xs)
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    fs.release()
                # mask: the valid keys of a row are [kmin_row, qpos]
                if (bstart < kmin_last) | (bstart + KB - 1 > q0):
                    lo = kmin_row - bstart
                    hi = qpos - bstart
                    for e in cutlass.range_constexpr(KB):
                        keep = (e >= lo) & (e <= hi)
                        xs[e] = cutlass.select_(keep, xs[e], Float32(float('-inf')))  # exp2 -> 0, never moves the max
                rmax = xs[0]
                for e in cutlass.range_constexpr(1, KB):
                    rmax = cute.arch.fmax(rmax, xs[e])
                m_old = m_run
                m_new = cute.arch.fmax(m_old, rmax * scale2)
                moved = m_new > m_old + RESCALE_THRESHOLD
                if moved:
                    m_run = m_new
                m_use = m_run
                if m_run < -1.0e29:  # no valid key yet: exp2 of the stand-in stays 0
                    m_use = Float32(0.0)
                neg_m = -m_use
                rsum = Float32(0.0)
                for e2 in cutlass.range_constexpr(KB // 2):
                    y0, y1 = cute.arch.fma_packed_f32x2((xs[2 * e2], xs[2 * e2 + 1]), (scale2, scale2), (neg_m, neg_m))
                    xs[2 * e2] = cute.math.exp2(y0, fastmath=True)
                    xs[2 * e2 + 1] = cute.math.exp2(y1, fastmath=True)
                    rsum = rsum + xs[2 * e2] + xs[2 * e2 + 1]
                pbf.store(xs.load().to(BFloat16))
                # the P buffer is free (and O complete) once the previous PV MMA finished
                hp = p_prod.acquire_and_advance()
                # O rescale: TMEM loads / stores are warp-collective, so the whole warp takes the branch
                corr = Float32(1.0)
                if moved & (j > 0):
                    corr = cute.math.exp2(m_old - m_new, fastmath=True)
                    l_run = l_run * corr
                if cute.arch.vote_any_sync(moved & (j > 0)):
                    for piece in cutlass.range_constexpr(VT // 32):
                        cute.copy(t2r_o, tTR_o[(None, None, None, piece)], ov)
                        cute.arch.fence_view_async_tmem_load()
                        ov.store(ov.load() * corr)
                        cute.copy(r2t_o, ov, tRT_o[(None, None, None, piece)])
                    cute.arch.fence_view_async_tmem_store()
                l_run = l_run + rsum
                cute.copy(r2t_p, pbf_v, tRT_tP)
                cute.arch.fence_view_async_tmem_store()
                hp.commit()
            # all PV MMAs done
            hp = p_prod.acquire_and_advance()
            inv_l = Float32(1.0) / l_run
            row_ptr = pY + ((Int64(b * L + qpos) * H + h) * V + vt * VT)
            for piece in cutlass.range_constexpr(VT // 32):
                cute.copy(t2r_o, tTR_o[(None, None, None, piece)], ov)
                cute.arch.fence_view_async_tmem_load()
                obf.store((ov.load() * inv_l).to(BFloat16))
                for e8 in cutlass.range_constexpr(4):
                    cute.make_tensor((row_ptr + (piece * 32 + e8 * 8)).align(16), cute.make_layout(8)).store(
                        cute.make_tensor(obf.iterator + e8 * 8, cute.make_layout(8)).load())
            if vt == 0:
                cute.make_tensor(pLse + ((b * H + h) * L + qpos), cute.make_layout(1))[0] = m_run + cute.math.log2(l_run, fastmath=True)
            tmem.relinquish_alloc_permit()
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_softmax_warps)
            tmem.free(tmem_ptr)


class ScaFwdT:
    """y = softmax(scale q k^T, masked) v for one (batch, head, chunk, 128-row tile, 256-wide v slab) per CTA, with
    the q tile in tensor memory as the A operand of S = Q K^T (full tensor-core rate for the 32-key blocks; with
    both operands in shared memory the N = 64 GEMM of ScaFwd runs at ~55%) and P as the A operand of PV.

    Warps 0-3: softmax (thread = row = TMEM lane), O rescale and the epilogue; warp 4: tcgen05 MMAs; warp 5: TMA.
    TMEM: Q (S/2 columns), S x 2 stages (KB each), P x 2 (over the S columns when `alias_p`, else KB/2 each), O (VT).
    gekko-4B (S = V = 128): 64 + 64 + 128 = 256 columns -> two CTAs per SM hide each other's latencies.
    S = V = 256: 128 + 64 + 32 + 256 = 480 -> one CTA per SM. Shared memory: 4-stage K and V rings only."""

    def __init__(self, H: int, K: int, S: int, V: int, C: int, KB: int = 32, kv_stages: int = 0):
        self.H, self.K, self.S, self.V, self.C, self.KB = H, K, S, V, C, KB
        self.n_rep = H // K
        self.TM = 128
        self.VT = min(V, 256)
        self.n_vt = V // self.VT
        # K + V ring stages within ~200 KB of shared memory
        self.kv_stages = kv_stages if kv_stages else max(2, min(4, (200 * 1024) // (KB * (S + self.VT) * 2)))
        self.num_softmax_warps = 4
        self.mma_warp = 4
        self.tma_warp = 5
        self.threads = 32 * 6
        self.row_tiles = C // self.TM
        assert H % K == 0 and S % 64 == 0 and S <= 256 and V % 32 == 0 and V % self.VT == 0 and C % KB == 0 and C % 128 == 0 and KB in (32, 64)
        self.tmem_q = 0
        self.tmem_s = S // 2
        # P over the S columns of its stage unless separate P stages still fit the budget (256: two CTAs per SM)
        budget = 256 if S // 2 + 2 * KB + self.VT <= 256 else 512
        self.alias_p = S // 2 + 2 * KB + KB + self.VT > budget
        self.tmem_p = self.tmem_s if self.alias_p else self.tmem_s + 2 * KB
        self.tmem_o = self.tmem_s + 2 * KB + (0 if self.alias_p else KB)
        total = self.tmem_o + self.VT
        assert total <= 512
        self.tmem_cols = max(32, 1 << (total - 1).bit_length())
        self.dbg = int(os.environ.get("SCA_DBG", "0"))  # 1: no softmax math, 3: MMAs only

    @cute.jit
    def __call__(self, tQ: cute.Tensor, tK: cute.Tensor, tV: cute.Tensor, tKmin: cute.Tensor, tY: cute.Tensor,
                 tLse: cute.Tensor, B: Int32, L: Int32, scale2: Float32, stream: cuda.CUstream):
        H, K, S, V, KB, VT = self.H, self.K, self.S, self.V, self.KB, self.VT
        nc = L // self.C
        mK = cute.make_tensor(tK.iterator, cute.make_layout((L, S, K, B), stride=(K * S, 1, S, L * K * S)))
        mV = cute.make_tensor(tV.iterator, cute.make_layout((V, L, K, B), stride=(1, K * V, V, L * K * V)))
        tm_qk = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, KB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        tm_pv = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, VT, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))
        q_layout = sm100_utils.make_smem_layout_a(tm_qk, (self.TM, KB, S), BFloat16, 1)  # TMEM A operand
        k_layout = sm100_utils.make_smem_layout_b(tm_qk, (self.TM, KB, S), BFloat16, self.kv_stages)
        v_layout = sm100_utils.make_smem_layout_b(tm_pv, (self.TM, VT, KB), BFloat16, self.kv_stages)
        p_layout = sm100_utils.make_smem_layout_a(tm_pv, (self.TM, VT, KB), BFloat16, 1)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_k, mK_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mK, cute.slice_(k_layout, (None, None, None, 0)), (self.TM, KB, S), tm_qk)
        tma_v, mV_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mV, cute.slice_(v_layout, (None, None, None, 0)), (self.TM, VT, KB), tm_pv)
        num_tiles = B * nc * H * self.row_tiles * self.n_vt
        self.kernel(tm_qk, tm_pv, tma_k, mK_t, tma_v, mV_t, tQ.iterator, tKmin.iterator, tY.iterator, tLse.iterator,
                    q_layout, k_layout, v_layout, p_layout, L, nc, scale2).launch(
            grid=[num_tiles, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tm_qk: cute.TiledMma, tm_pv: cute.TiledMma, tma_k: cute.CopyAtom, mK: cute.Tensor,
               tma_v: cute.CopyAtom, mV: cute.Tensor, pQ: cute.Pointer, pKmin: cute.Pointer, pY: cute.Pointer,
               pLse: cute.Pointer, q_layout: cute.ComposedLayout, k_layout: cute.ComposedLayout,
               v_layout: cute.ComposedLayout, p_layout: cute.ComposedLayout, L: Int32, nc: Int32, scale2: Float32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H, K, S, V, KB, C, VT = self.H, self.K, self.S, self.V, self.KB, self.C, self.VT
        vt = bidx % self.n_vt
        t0 = bidx // self.n_vt
        rt = t0 % self.row_tiles
        t1 = t0 // self.row_tiles
        h = t1 % H
        t2 = t1 // H
        c = t2 % nc
        b = t2 // nc
        kh = h // self.n_rep
        q0 = c * C + rt * self.TM
        kstart = Int32(0)
        if c > 0:
            kstart = (c - 1) * C
        kmin0 = cute.make_tensor(pKmin + (b * L + q0), cute.make_layout(1))[0]
        kb_lo = (kmin0 - kstart) // KB
        if kb_lo < 0:
            kb_lo = Int32(0)
        kb_hi = (q0 + self.TM - 1 - kstart) // KB
        n_blocks = kb_hi - kb_lo + 1
        kblk0 = kstart // KB + kb_lo

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_v)

        @cute.struct
        class SharedStorage:
            k_mbar: cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
            v_mbar: cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
            raw_mbar: cute.struct.MemRange[cutlass.Int64, 8]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # raw barriers: s_full x2, s_empty x2, p_full x2, p_empty x2 (non-alias), pv_done
        raw = storage.raw_mbar.data_ptr()
        S_FULL, S_EMPTY, P_FULL, P_EMPTY, PV_DONE = 0, 2, 4, 6, 7
        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(raw + (S_FULL + i), 1)
                    cute.arch.mbarrier_init(raw + (S_EMPTY + i), 1 if self.alias_p else self.num_softmax_warps)
                    cute.arch.mbarrier_init(raw + (P_FULL + i), 32 * self.num_softmax_warps)
                if const_expr(not self.alias_p):
                    cute.arch.mbarrier_init(raw + P_EMPTY, 1)
                cute.arch.mbarrier_init(raw + PV_DONE, 1)
            cute.arch.mbarrier_init_fence()
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_softmax_warps + 1))
        q_bar = pipeline.NamedBarrier(barrier_id=4, num_threads=32 * (self.num_softmax_warps + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        k_bytes = cute.size_in_bytes(BFloat16, cute.select(k_layout, mode=[0, 1, 2]))
        v_bytes = cute.size_in_bytes(BFloat16, cute.select(v_layout, mode=[0, 1, 2]))
        k_prod, k_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.k_mbar.data_ptr(), num_stages=self.kv_stages, producer_group=one, consumer_group=one,
            tx_count=k_bytes).make_participants()
        v_prod, v_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.v_mbar.data_ptr(), num_stages=self.kv_stages, producer_group=one, consumer_group=one,
            tx_count=v_bytes).make_participants()
        cute.arch.barrier()
        sK = smem.allocate_tensor(BFloat16, k_layout.outer, byte_alignment=1024, swizzle=k_layout.inner)
        sV = smem.allocate_tensor(BFloat16, v_layout.outer, byte_alignment=1024, swizzle=v_layout.inner)

        gK = cute.local_tile(mK, (KB, S), (None, None, None, None))
        gV = cute.local_tile(mV, (VT, KB), (None, None, None, None))
        thr_qk = tm_qk.get_slice(0)
        thr_pv = tm_pv.get_slice(0)
        tKs, tKg = cpasync.tma_partition(tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(thr_qk.partition_B(gK), 0, 3))
        tVs, tVg = cpasync.tma_partition(tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(thr_pv.partition_B(gV), 0, 3))
        tCrK = tm_qk.make_fragment_B(sK)
        tCrV = tm_pv.make_fragment_B(sV)
        s_fake = tm_qk.make_fragment_C(cute.append(tm_qk.partition_shape_C((self.TM, KB)), 2))
        o_fake = tm_pv.make_fragment_C(tm_pv.partition_shape_C((self.TM, VT)))
        tQ_fake = cute.make_tensor(cute.make_ptr(BFloat16, 0, cute.AddressSpace.tmem), q_layout.outer)
        tQr_fake = tm_qk.make_fragment_A(tQ_fake)[(None, None, None, 0)]
        tP_fake = cute.make_tensor(cute.make_ptr(BFloat16, 0, cute.AddressSpace.tmem), p_layout.outer)
        tPr_fake = tm_pv.make_fragment_A(tP_fake)[(None, None, None, 0)]
        p_stride = KB if self.alias_p else KB // 2  # columns between the two P stages

        if warp == self.tma_warp:
            for j in cutlass.range(n_blocks):
                hk = k_prod.acquire_and_advance()
                cute.copy(tma_k, tKg[(None, kblk0 + j, 0, kh, b)], tKs[(None, hk.index)], tma_bar_ptr=hk.barrier)
                hv = v_prod.acquire_and_advance()
                cute.copy(tma_v, tVg[(None, vt, kblk0 + j, kh, b)], tVs[(None, hv.index)], tma_bar_ptr=hv.barrier)
            k_prod.tail()
            v_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tO = cute.make_tensor(tmem_ptr + self.tmem_o, o_fake.layout)
            tQr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_q, dtype=BFloat16), tQr_fake.layout)
            tPr = [cute.make_tensor(cute.recast_ptr(tmem_ptr + (self.tmem_p + i * p_stride), dtype=BFloat16), tPr_fake.layout) for i in range(2)]
            q_bar.arrive_and_wait()  # q tile stored in TMEM by the softmax warps
            op_qk = tcgen05.MmaF16BF16Op(BFloat16, Float32, (self.TM, KB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
                                         tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
            op_pv = tcgen05.MmaF16BF16Op(BFloat16, Float32, (self.TM, VT, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
                                         tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN)
            if const_expr(self.dbg == 3):
                for j in cutlass.range(n_blocks):
                    hk = k_cons.wait_and_advance()
                    hv = v_cons.wait_and_advance()
                    _umma(op_qk, tS[(None, None, None, 0)], tQr, tCrK, hk.index, False, raw + S_FULL)
                    _umma(op_pv, tO, tPr[0], tCrV, hv.index, True, raw + PV_DONE)
                    hk.release()
                    hv.release()
            else:
                ph_se0, ph_se1 = Int32(0), Int32(0)
                ph_pf0, ph_pf1 = Int32(0), Int32(0)
                for jj in cutlass.range_constexpr(2):
                    if jj < n_blocks:
                        hk = k_cons.wait_and_advance()
                        _umma(op_qk, tS[(None, None, None, jj)], tQr, tCrK, hk.index, False, raw + (S_FULL + jj))
                        hk.release()
                n_pairs = (n_blocks + 1) // 2
                for jp in cutlass.range(n_pairs):
                    j = 2 * jp
                    # ---- block j (stage 0): PV_j then S_{j+2} (alias: the stage is freed by PV_j's completion) ----
                    cute.arch.mbarrier_wait(raw + (P_FULL + 0), ph_pf0)
                    ph_pf0 ^= 1
                    hv = v_cons.wait_and_advance()
                    _umma(op_pv, tO, tPr[0], tCrV, hv.index, True, raw + (S_EMPTY + 0) if self.alias_p else raw + P_EMPTY)
                    hv.release()
                    with cute.arch.elect_one():
                        tcgen05.commit(raw + PV_DONE)
                    if j + 2 < n_blocks:
                        cute.arch.mbarrier_wait(raw + (S_EMPTY + 0), ph_se0)
                        ph_se0 ^= 1
                        hk = k_cons.wait_and_advance()
                        _umma(op_qk, tS[(None, None, None, 0)], tQr, tCrK, hk.index, False, raw + S_FULL)
                        hk.release()
                    # ---- block j + 1 (stage 1) ----
                    if j + 1 < n_blocks:
                        cute.arch.mbarrier_wait(raw + (P_FULL + 1), ph_pf1)
                        ph_pf1 ^= 1
                        hv = v_cons.wait_and_advance()
                        _umma(op_pv, tO, tPr[1], tCrV, hv.index, True, raw + (S_EMPTY + 1) if self.alias_p else raw + P_EMPTY)
                        hv.release()
                        with cute.arch.elect_one():
                            tcgen05.commit(raw + PV_DONE)
                        if j + 3 < n_blocks:
                            cute.arch.mbarrier_wait(raw + (S_EMPTY + 1), ph_se1)
                            ph_se1 ^= 1
                            hk = k_cons.wait_and_advance()
                            _umma(op_qk, tS[(None, None, None, 1)], tQr, tCrK, hk.index, False, raw + (S_FULL + 1))
                            hk.release()
        else:
            tmem.allocate(self.tmem_cols)
            tmem.relinquish_alloc_permit()  # right away: the other CTA of the SM may allocate its columns
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tO = cute.make_tensor(tmem_ptr + self.tmem_o, o_fake.layout)
            tQr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_q, dtype=BFloat16), tQr_fake.layout)
            tPr = [cute.make_tensor(cute.recast_ptr(tmem_ptr + (self.tmem_p + i * p_stride), dtype=BFloat16), tPr_fake.layout) for i in range(2)]
            trow = tidx % 128
            qpos = q0 + trow
            kmin_row = cute.make_tensor(pKmin + (b * L + qpos), cute.make_layout(1))[0]
            kmin_last = cute.make_tensor(pKmin + (b * L + q0 + self.TM - 1), cute.make_layout(1))[0]
            # q tile -> TMEM (A operand of S): this thread's row; zero the O accumulator (the PV MMAs accumulate)
            st_q = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(S // 2), tcgen05.Unpack.NONE), BFloat16)
            r2t_q = tcgen05.make_tmem_copy(st_q, tQr)
            thr_q = r2t_q.get_slice(trow)
            tRT_tQ = thr_q.partition_D(tQr)
            tRT_cq = thr_q.partition_S(thr_qk.partition_A(cute.make_identity_tensor((self.TM, S))))
            rQ = cute.make_rmem_tensor((S,), BFloat16)
            q_ptr = pQ + (Int64(b * L + qpos) * H + h) * S
            for e8 in cutlass.range_constexpr(S // 8):
                cute.make_tensor(rQ.iterator + e8 * 8, cute.make_layout(8)).store(
                    cute.make_tensor((q_ptr + e8 * 8).align(16), cute.make_layout(8)).load())
            cute.copy(r2t_q, cute.make_tensor(rQ.iterator, cute.make_layout(tRT_cq.shape)), tRT_tQ)
            ld_o = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32), tcgen05.Pack.NONE), Float32)
            st_o = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(32), tcgen05.Unpack.NONE), Float32)
            tO_r = cute.flat_divide(tO[((None, None), 0, 0)], (self.TM, 32))
            t2r_o = tcgen05.make_tmem_copy(ld_o, tO_r[(None, None, 0, 0)])
            r2t_o = tcgen05.make_tmem_copy(st_o, tO_r[(None, None, 0, 0)])
            tTR_o = t2r_o.get_slice(trow).partition_S(tO_r)
            tTR_o = cute.group_modes(tTR_o, 3, cute.rank(tTR_o))
            tRT_o = r2t_o.get_slice(trow).partition_D(tO_r)
            tRT_o = cute.group_modes(tRT_o, 3, cute.rank(tRT_o))
            ov = cute.make_rmem_tensor(t2r_o.get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, 32))).shape, Float32)
            obf = cute.make_rmem_tensor(ov.shape, BFloat16)
            for e in cutlass.range_constexpr(32):
                ov[e] = Float32(0.0)
            for piece in cutlass.range_constexpr(VT // 32):
                cute.copy(r2t_o, ov, tRT_o[(None, None, None, piece)])
            cute.arch.fence_view_async_tmem_store()
            q_bar.arrive_and_wait()
            # S loads (this thread's row of a stage) and P stores (bf16 pairs, the A operand of PV)
            ld_s = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(KB), tcgen05.Pack.NONE), Float32)
            tS_r = [cute.flat_divide(tS[((None, None), 0, 0, i)], (self.TM, KB)) for i in range(2)]
            t2r_s = [tcgen05.make_tmem_copy(ld_s, tS_r[i][(None, None, 0, 0)]) for i in range(2)]
            tTR_s = [t2r_s[i].get_slice(trow).partition_S(tS_r[i]) for i in range(2)]
            tTR_s = [cute.group_modes(tTR_s[i], 3, cute.rank(tTR_s[i])) for i in range(2)]
            xs = cute.make_rmem_tensor(t2r_s[0].get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, KB))).shape, Float32)
            st_p = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(KB // 2), tcgen05.Unpack.NONE), BFloat16)
            r2t_p = [tcgen05.make_tmem_copy(st_p, tPr[i]) for i in range(2)]
            tRT_tP = [r2t_p[i].get_slice(trow).partition_D(tPr[i]) for i in range(2)]
            tRT_c = r2t_p[0].get_slice(trow).partition_S(thr_pv.partition_A(cute.make_identity_tensor((self.TM, KB))))
            pbf = cute.make_rmem_tensor(xs.shape, BFloat16)
            pbf_v = cute.make_tensor(pbf.iterator, cute.make_layout(tRT_c.shape))

            m_run = Float32(NEG_BIG)
            l_run = Float32(0.0)
            ph_sf0, ph_sf1 = Int32(0), Int32(0)
            ph_pe = Int32(1)
            ph_pv = Int32(0)
            n_loop = n_blocks
            if const_expr(self.dbg == 3):
                n_loop = Int32(0)
            for j in cutlass.range(n_loop):
                bstart = kstart + (kb_lo + j) * KB
                stg = j % 2
                if stg == 0:
                    cute.arch.mbarrier_wait(raw + (S_FULL + 0), ph_sf0)
                    ph_sf0 ^= 1
                    cute.copy(t2r_s[0], tTR_s[0][(None, None, None, 0)], xs)
                else:
                    cute.arch.mbarrier_wait(raw + (S_FULL + 1), ph_sf1)
                    ph_sf1 ^= 1
                    cute.copy(t2r_s[1], tTR_s[1][(None, None, None, 0)], xs)
                cute.arch.fence_view_async_tmem_load()
                if const_expr(not self.alias_p):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(raw + (S_EMPTY + stg))
                moved = False
                rsum = Float32(0.0)
                m_old = m_run
                m_new = m_run
                if const_expr(self.dbg == 0):
                    # mask: the valid keys of a row are [kmin_row, qpos] (warp-uniform branch, see _dkdv_block)
                    masked = cute.arch.make_warp_uniform(Int32((bstart < kmin_last) | (bstart + KB - 1 > q0)))
                    if masked != 0:
                        lo = kmin_row - bstart
                        hi = qpos - bstart
                        for e in cutlass.range_constexpr(KB):
                            keep = (e >= lo) & (e <= hi)
                            xs[e] = cutlass.select_(keep, xs[e], Float32(float('-inf')))
                    rmax = xs[0]
                    for e in cutlass.range_constexpr(1, KB):
                        rmax = cute.arch.fmax(rmax, xs[e])
                    m_new = cute.arch.fmax(m_old, rmax * scale2)
                    moved = m_new > m_old + RESCALE_THRESHOLD
                    if moved:
                        m_run = m_new
                    m_use = m_run
                    if m_run < -1.0e29:  # no valid key yet: exp2 of the stand-in stays 0
                        m_use = Float32(0.0)
                    neg_m = -m_use
                    for e2 in cutlass.range_constexpr(KB // 2):
                        y0, y1 = cute.arch.fma_packed_f32x2((xs[2 * e2], xs[2 * e2 + 1]), (scale2, scale2), (neg_m, neg_m))
                        xs[2 * e2] = cute.math.exp2(y0, fastmath=True)
                        xs[2 * e2 + 1] = cute.math.exp2(y1, fastmath=True)
                        rsum = rsum + xs[2 * e2] + xs[2 * e2 + 1]
                    pbf.store(xs.load().to(BFloat16))
                # O is complete and stable once PV_{j-1} finished
                if j > 0:
                    cute.arch.mbarrier_wait(raw + PV_DONE, ph_pv)
                    ph_pv ^= 1
                corr = Float32(1.0)
                if moved & (j > 0):
                    corr = cute.math.exp2(m_old - m_new, fastmath=True)
                    l_run = l_run * corr
                if cute.arch.vote_any_sync(moved & (j > 0)):  # TMEM loads / stores are warp-collective
                    for piece in cutlass.range_constexpr(VT // 32):
                        cute.copy(t2r_o, tTR_o[(None, None, None, piece)], ov)
                        cute.arch.fence_view_async_tmem_load()
                        ov.store(ov.load() * corr)
                        cute.copy(r2t_o, ov, tRT_o[(None, None, None, piece)])
                    cute.arch.fence_view_async_tmem_store()
                l_run = l_run + rsum
                if const_expr(not self.alias_p):
                    cute.arch.mbarrier_wait(raw + P_EMPTY, ph_pe)  # P stage free: PV_{j-2} done (single empty barrier, in order)
                    ph_pe ^= 1
                if stg == 0:
                    cute.copy(r2t_p[0], pbf_v, tRT_tP[0])
                else:
                    cute.copy(r2t_p[1], pbf_v, tRT_tP[1])
                cute.arch.fence_view_async_tmem_store()
                cute.arch.mbarrier_arrive(raw + (P_FULL + stg))
            # all PV MMAs done
            cute.arch.mbarrier_wait(raw + PV_DONE, ph_pv)
            inv_l = Float32(1.0) / l_run
            row_ptr = pY + ((Int64(b * L + qpos) * H + h) * V + vt * VT)
            for piece in cutlass.range_constexpr(VT // 32):
                cute.copy(t2r_o, tTR_o[(None, None, None, piece)], ov)
                cute.arch.fence_view_async_tmem_load()
                obf.store((ov.load() * inv_l).to(BFloat16))
                for e8 in cutlass.range_constexpr(4):
                    cute.make_tensor((row_ptr + (piece * 32 + e8 * 8)).align(16), cute.make_layout(8)).store(
                        cute.make_tensor(obf.iterator + e8 * 8, cute.make_layout(8)).load())
            if vt == 0:
                cute.make_tensor(pLse + ((b * H + h) * L + qpos), cute.make_layout(1))[0] = m_run + cute.math.log2(l_run, fastmath=True)
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_softmax_warps)
            tmem.free(tmem_ptr)


class ScaPmat:
    """P = exp2(S scale2 - lse2) (bf16) of the forward, materialized chunk-major for the large-V backward:
    Pm[(c, b, h)][q - cC][k - (c - 1) C] for the queries q of chunk c and the keys k of chunks c - 1 and c
    (columns [0, C) of chunk 0 are unused). Masked / skipped entries are zero (the buffer is zeroed by the host),
    so dV of the keys of chunk c is the cuBLAS GEMM Pm[c][:, C:]^T dO_c + Pm[c + 1][:, :C]^T dO_{c+1}.

    One (batch, head, chunk, 128-row q tile) per CTA; the q tile in TMEM as the A operand of S = Q K^T (64-key
    blocks), 4 softmax warps (thread = row) write their 64 bf16 values per block straight to global memory.
    TMEM: Q (S/2 columns) + S x 2: 256 columns for S = 256, two CTAs per SM."""

    def __init__(self, H: int, K: int, S: int, C: int, KB: int = 64, k_stages: int = 3):
        self.H, self.K, self.S, self.C, self.KB = H, K, S, C, KB
        self.n_rep = H // K
        self.TM = 128
        self.k_stages = k_stages
        self.num_softmax_warps = 4
        self.mma_warp = 4
        self.tma_warp = 5
        self.threads = 32 * 6
        self.row_tiles = C // self.TM
        assert H % K == 0 and S % 64 == 0 and S <= 256 and C % KB == 0 and C % 128 == 0 and KB == 64
        self.tmem_q = 0
        self.tmem_s = S // 2
        total = self.tmem_s + 2 * KB
        self.tmem_cols = max(32, 1 << (total - 1).bit_length())

    @cute.jit
    def __call__(self, tQ: cute.Tensor, tK: cute.Tensor, tLse: cute.Tensor, tKmin: cute.Tensor, tPm: cute.Tensor,
                 B: Int32, L: Int32, scale2: Float32, stream: cuda.CUstream):
        H, K, S, KB = self.H, self.K, self.S, self.KB
        nc = L // self.C
        mK = cute.make_tensor(tK.iterator, cute.make_layout((L, S, K, B), stride=(K * S, 1, S, L * K * S)))
        tm_qk = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, KB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        q_layout = sm100_utils.make_smem_layout_a(tm_qk, (self.TM, KB, S), BFloat16, 1)  # TMEM A operand
        k_layout = sm100_utils.make_smem_layout_b(tm_qk, (self.TM, KB, S), BFloat16, self.k_stages)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_k, mK_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mK, cute.slice_(k_layout, (None, None, None, 0)), (self.TM, KB, S), tm_qk)
        num_tiles = B * nc * H * self.row_tiles
        self.kernel(tm_qk, tma_k, mK_t, tQ.iterator, tLse.iterator, tKmin.iterator, tPm.iterator, q_layout, k_layout,
                    B, L, nc, scale2).launch(grid=[num_tiles, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tm_qk: cute.TiledMma, tma_k: cute.CopyAtom, mK: cute.Tensor, pQ: cute.Pointer, pLse: cute.Pointer,
               pKmin: cute.Pointer, pPm: cute.Pointer, q_layout: cute.ComposedLayout, k_layout: cute.ComposedLayout,
               B: Int32, L: Int32, nc: Int32, scale2: Float32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H, K, S, KB, C = self.H, self.K, self.S, self.KB, self.C
        rt = bidx % self.row_tiles
        t1 = bidx // self.row_tiles
        h = t1 % H
        t2 = t1 // H
        c = t2 % nc
        b = t2 // nc
        kh = h // self.n_rep
        q0 = c * C + rt * self.TM
        kstart = Int32(0)
        if c > 0:
            kstart = (c - 1) * C
        kmin0 = cute.make_tensor(pKmin + (b * L + q0), cute.make_layout(1))[0]
        kb_lo = (kmin0 - kstart) // KB
        if kb_lo < 0:
            kb_lo = Int32(0)
        kb_hi = (q0 + self.TM - 1 - kstart) // KB
        n_blocks = kb_hi - kb_lo + 1
        kblk0 = kstart // KB + kb_lo

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_k)

        @cute.struct
        class SharedStorage:
            k_mbar: cute.struct.MemRange[cutlass.Int64, self.k_stages * 2]
            raw_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        raw = storage.raw_mbar.data_ptr()
        S_FULL, S_EMPTY = 0, 2
        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(raw + (S_FULL + i), 1)
                    cute.arch.mbarrier_init(raw + (S_EMPTY + i), self.num_softmax_warps)
            cute.arch.mbarrier_init_fence()
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_softmax_warps + 1))
        q_bar = pipeline.NamedBarrier(barrier_id=4, num_threads=32 * (self.num_softmax_warps + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        k_bytes = cute.size_in_bytes(BFloat16, cute.select(k_layout, mode=[0, 1, 2]))
        k_prod, k_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.k_mbar.data_ptr(), num_stages=self.k_stages, producer_group=one, consumer_group=one,
            tx_count=k_bytes).make_participants()
        cute.arch.barrier()
        sK = smem.allocate_tensor(BFloat16, k_layout.outer, byte_alignment=1024, swizzle=k_layout.inner)
        gK = cute.local_tile(mK, (KB, S), (None, None, None, None))
        thr_qk = tm_qk.get_slice(0)
        tKs, tKg = cpasync.tma_partition(tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(thr_qk.partition_B(gK), 0, 3))
        tCrK = tm_qk.make_fragment_B(sK)
        s_fake = tm_qk.make_fragment_C(cute.append(tm_qk.partition_shape_C((self.TM, KB)), 2))
        tQ_fake = cute.make_tensor(cute.make_ptr(BFloat16, 0, cute.AddressSpace.tmem), q_layout.outer)
        tQr_fake = tm_qk.make_fragment_A(tQ_fake)[(None, None, None, 0)]

        if warp == self.tma_warp:
            for j in cutlass.range(n_blocks):
                hk = k_prod.acquire_and_advance()
                cute.copy(tma_k, tKg[(None, kblk0 + j, 0, kh, b)], tKs[(None, hk.index)], tma_bar_ptr=hk.barrier)
            k_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tQr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_q, dtype=BFloat16), tQr_fake.layout)
            q_bar.arrive_and_wait()
            op_qk = tcgen05.MmaF16BF16Op(BFloat16, Float32, (self.TM, KB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
                                         tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
            ph_se0, ph_se1 = Int32(0), Int32(0)
            for jj in cutlass.range_constexpr(2):
                if jj < n_blocks:
                    hk = k_cons.wait_and_advance()
                    _umma(op_qk, tS[(None, None, None, jj)], tQr, tCrK, hk.index, False, raw + (S_FULL + jj))
                    hk.release()
            n_pairs = (n_blocks + 1) // 2
            for jp in cutlass.range(n_pairs):
                j = 2 * jp
                if j + 2 < n_blocks:
                    cute.arch.mbarrier_wait(raw + (S_EMPTY + 0), ph_se0)
                    ph_se0 ^= 1
                    hk = k_cons.wait_and_advance()
                    _umma(op_qk, tS[(None, None, None, 0)], tQr, tCrK, hk.index, False, raw + S_FULL)
                    hk.release()
                if j + 3 < n_blocks:
                    cute.arch.mbarrier_wait(raw + (S_EMPTY + 1), ph_se1)
                    ph_se1 ^= 1
                    hk = k_cons.wait_and_advance()
                    _umma(op_qk, tS[(None, None, None, 1)], tQr, tCrK, hk.index, False, raw + (S_FULL + 1))
                    hk.release()
        else:
            tmem.allocate(self.tmem_cols)
            tmem.relinquish_alloc_permit()
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tQr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_q, dtype=BFloat16), tQr_fake.layout)
            trow = tidx % 128
            qpos = q0 + trow
            kmin_row = cute.make_tensor(pKmin + (b * L + qpos), cute.make_layout(1))[0]
            kmin_last = cute.make_tensor(pKmin + (b * L + q0 + self.TM - 1), cute.make_layout(1))[0]
            neg_lse = -cute.make_tensor(pLse + ((b * H + h) * L + qpos), cute.make_layout(1))[0]
            # q tile -> TMEM (A operand of S): this thread's row
            st_q = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(S // 2), tcgen05.Unpack.NONE), BFloat16)
            r2t_q = tcgen05.make_tmem_copy(st_q, tQr)
            thr_q = r2t_q.get_slice(trow)
            tRT_tQ = thr_q.partition_D(tQr)
            tRT_cq = thr_q.partition_S(thr_qk.partition_A(cute.make_identity_tensor((self.TM, S))))
            rQ = cute.make_rmem_tensor((S,), BFloat16)
            q_ptr = pQ + (Int64(b * L + qpos) * H + h) * S
            for e8 in cutlass.range_constexpr(S // 8):
                cute.make_tensor(rQ.iterator + e8 * 8, cute.make_layout(8)).store(
                    cute.make_tensor((q_ptr + e8 * 8).align(16), cute.make_layout(8)).load())
            cute.copy(r2t_q, cute.make_tensor(rQ.iterator, cute.make_layout(tRT_cq.shape)), tRT_tQ)
            cute.arch.fence_view_async_tmem_store()
            q_bar.arrive_and_wait()
            ld_s = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(KB), tcgen05.Pack.NONE), Float32)
            tS_r = [cute.flat_divide(tS[((None, None), 0, 0, i)], (self.TM, KB)) for i in range(2)]
            t2r_s = [tcgen05.make_tmem_copy(ld_s, tS_r[i][(None, None, 0, 0)]) for i in range(2)]
            tTR_s = [t2r_s[i].get_slice(trow).partition_S(tS_r[i]) for i in range(2)]
            tTR_s = [cute.group_modes(tTR_s[i], 3, cute.rank(tTR_s[i])) for i in range(2)]
            xs = cute.make_rmem_tensor(t2r_s[0].get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, KB))).shape, Float32)
            pbf = cute.make_rmem_tensor(xs.shape, BFloat16)
            # this row of Pm: [(c, b, h)][qpos - cC][key - (c - 1) C]
            row_ptr = pPm + (Int64((c * B + b) * H + h) * C + (qpos - c * C)) * (2 * C)
            ph_sf0, ph_sf1 = Int32(0), Int32(0)
            for j in cutlass.range(n_blocks):
                bstart = kstart + (kb_lo + j) * KB
                stg = j % 2
                if stg == 0:
                    cute.arch.mbarrier_wait(raw + (S_FULL + 0), ph_sf0)
                    ph_sf0 ^= 1
                    cute.copy(t2r_s[0], tTR_s[0][(None, None, None, 0)], xs)
                else:
                    cute.arch.mbarrier_wait(raw + (S_FULL + 1), ph_sf1)
                    ph_sf1 ^= 1
                    cute.copy(t2r_s[1], tTR_s[1][(None, None, None, 0)], xs)
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(raw + (S_EMPTY + stg))
                masked = cute.arch.make_warp_uniform(Int32((bstart < kmin_last) | (bstart + KB - 1 > q0)))
                if masked != 0:
                    lo = kmin_row - bstart
                    hi = qpos - bstart
                    for e in cutlass.range_constexpr(KB):
                        keep = (e >= lo) & (e <= hi)
                        xs[e] = cutlass.select_(keep, xs[e], Float32(float('-inf')))
                for e2 in cutlass.range_constexpr(KB // 2):
                    y0, y1 = cute.arch.fma_packed_f32x2((xs[2 * e2], xs[2 * e2 + 1]), (scale2, scale2), (neg_lse, neg_lse))
                    xs[2 * e2] = cute.math.exp2(y0, fastmath=True)
                    xs[2 * e2 + 1] = cute.math.exp2(y1, fastmath=True)
                pbf.store(xs.load().to(BFloat16))
                col0 = bstart - kstart
                if c == 0:
                    col0 = col0 + C
                blk_ptr = row_ptr + col0
                for e8 in cutlass.range_constexpr(KB // 8):
                    cute.make_tensor((blk_ptr + e8 * 8).align(16), cute.make_layout(8)).store(
                        cute.make_tensor(pbf.iterator + e8 * 8, cute.make_layout(8)).load())
            # zeros for the columns of this row the GEMM reads but no block wrote: before the first block (segment
            # start of the tile) and after the last one (causal); chunk 0 starts at column C
            zb = cute.make_rmem_tensor((8,), BFloat16)
            for e in cutlass.range_constexpr(8):
                zb[e] = BFloat16(0.0)
            cstart = Int32(0)
            if c == 0:
                cstart = C
            cfirst = cstart + kb_lo * KB
            clast = cstart + (kb_hi + 1) * KB
            for col in cutlass.range(cstart, cfirst, 8):
                cute.make_tensor((row_ptr + col).align(16), cute.make_layout(8)).store(
                    cute.make_tensor(zb.iterator, cute.make_layout(8)).load())
            for col in cutlass.range(clast, 2 * C, 8):
                cute.make_tensor((row_ptr + col).align(16), cute.make_layout(8)).store(
                    cute.make_tensor(zb.iterator, cute.make_layout(8)).load())
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_softmax_warps)
            tmem.free(tmem_ptr)


class ScaBwdDv:
    """dv[keys of a 128-row tile] = sum over the attending queries P^T dO, for one (batch, kv head, key tile) per CTA
    (shapes that do not fit ScaBwdDkDv: S + V > 256).

    The CTA loops over the heads of the kv group and the 64-query blocks that can attend the tile (from the tile's
    own rows to the last query whose segment reaches the tile, within the next chunk). Per block: S^T = K Q^T
    (tcgen05, K tile resident), P^T = exp2(S^T scale2 - lse2[q]) with the mask (warps 0-3, thread = key row), P^T
    into TMEM, dV += P^T dO (dO block as MN-major B). TMEM: S^T x 2, P^T, dV."""

    def __init__(self, H: int, K: int, S: int, V: int, C: int, QB: int = 64, stages: int = 2):
        self.H, self.K, self.S, self.V, self.C, self.QB = H, K, S, V, C, QB
        self.n_rep = H // K
        self.TM = 128
        self.VT = min(V, 256)
        self.n_vt = V // self.VT
        self.stages = stages
        self.num_cw = 4  # compute warps
        self.mma_warp = 4
        self.tma_warp = 5
        self.threads = 32 * 6
        assert H % K == 0 and S % 64 == 0 and S <= 256 and V % 32 == 0 and V % self.VT == 0 and C % 128 == 0 and C % QB == 0
        self.tmem_s = 0
        self.tmem_p = 2 * QB
        self.tmem_o = 2 * QB + QB // 2
        total = self.tmem_o + self.VT
        self.tmem_cols = max(32, 1 << (total - 1).bit_length())

    @cute.jit
    def __call__(self, tQ: cute.Tensor, tK: cute.Tensor, tDO: cute.Tensor, tLse: cute.Tensor, tKmin: cute.Tensor,
                 tQmax: cute.Tensor, tDV: cute.Tensor, B: Int32, L: Int32, scale2: Float32, stream: cuda.CUstream):
        H, K, S, V, QB, VT = self.H, self.K, self.S, self.V, self.QB, self.VT
        nc = L // self.C
        # k as (M = L, K = S, Kh, B); q as (N = L, K = S, H, B); dO as (N = V, K = L, H, B) (MN-major B)
        mK = cute.make_tensor(tK.iterator, cute.make_layout((L, S, K, B), stride=(K * S, 1, S, L * K * S)))
        mQ = cute.make_tensor(tQ.iterator, cute.make_layout((L, S, H, B), stride=(H * S, 1, S, L * H * S)))
        mDO = cute.make_tensor(tDO.iterator, cute.make_layout((V, L, H, B), stride=(1, H * V, V, L * H * V)))
        tm_s = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, QB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        tm_o = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, VT, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))
        a_layout = sm100_utils.make_smem_layout_a(tm_s, (self.TM, QB, S), BFloat16, 1)
        q_layout = sm100_utils.make_smem_layout_b(tm_s, (self.TM, QB, S), BFloat16, self.stages)
        do_layout = sm100_utils.make_smem_layout_b(tm_o, (self.TM, VT, QB), BFloat16, self.stages)
        p_layout = sm100_utils.make_smem_layout_a(tm_o, (self.TM, VT, QB), BFloat16, 1)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_a, mK_t = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mK, cute.slice_(a_layout, (None, None, None, 0)), (self.TM, QB, S), tm_s)
        tma_q, mQ_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mQ, cute.slice_(q_layout, (None, None, None, 0)), (self.TM, QB, S), tm_s)
        tma_do, mDO_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mDO, cute.slice_(do_layout, (None, None, None, 0)), (self.TM, VT, QB), tm_o)
        num_tiles = B * K * (L // self.TM) * self.n_vt
        self.kernel(tm_s, tm_o, tma_a, mK_t, tma_q, mQ_t, tma_do, mDO_t, tLse.iterator, tKmin.iterator, tQmax.iterator,
                    tDV.iterator, a_layout, q_layout, do_layout, p_layout, L, nc, scale2).launch(
            grid=[num_tiles, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tm_s: cute.TiledMma, tm_o: cute.TiledMma, tma_a: cute.CopyAtom, mK: cute.Tensor,
               tma_q: cute.CopyAtom, mQ: cute.Tensor, tma_do: cute.CopyAtom, mDO: cute.Tensor, pLse: cute.Pointer,
               pKmin: cute.Pointer, pQmax: cute.Pointer, pDV: cute.Pointer, a_layout: cute.ComposedLayout,
               q_layout: cute.ComposedLayout, do_layout: cute.ComposedLayout, p_layout: cute.ComposedLayout,
               L: Int32, nc: Int32, scale2: Float32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H, K, S, V, QB, C, VT = self.H, self.K, self.S, self.V, self.QB, self.C, self.VT
        key_tiles = L // self.TM
        vt = bidx % self.n_vt
        t0 = bidx // self.n_vt
        kt = t0 % key_tiles
        t1 = t0 // key_tiles
        kh = t1 % K
        b = t1 // K
        k0 = kt * self.TM
        ck = k0 // C
        # query blocks that can attend the tile: from its own rows (causal) to the last query whose segment
        # starts at or before the tile's last key, within the next chunk
        qmax = cute.make_tensor(pQmax + (b * L + k0 + self.TM - 1), cute.make_layout(1))[0]
        q_end = (ck + 2) * C - 1
        if q_end > L - 1:
            q_end = L - 1
        if qmax < q_end:
            q_end = qmax
        qb_lo = k0 // QB
        qb_hi = q_end // QB
        n_qb = qb_hi - qb_lo + 1
        n_blocks = n_qb * self.n_rep

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_q)
            cpasync.prefetch_descriptor(tma_do)

        @cute.struct
        class SharedStorage:
            a_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            do_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            s_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            p_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_cw + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cw_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * self.num_cw)
        a_bytes = cute.size_in_bytes(BFloat16, cute.select(a_layout, mode=[0, 1, 2]))
        q_bytes = cute.size_in_bytes(BFloat16, cute.select(q_layout, mode=[0, 1, 2]))
        do_bytes = cute.size_in_bytes(BFloat16, cute.select(do_layout, mode=[0, 1, 2]))
        a_prod, a_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_mbar.data_ptr(), num_stages=1, producer_group=one, consumer_group=one,
            tx_count=a_bytes).make_participants()
        q_prod, q_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.q_mbar.data_ptr(), num_stages=self.stages, producer_group=one, consumer_group=one,
            tx_count=q_bytes).make_participants()
        do_prod, do_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.do_mbar.data_ptr(), num_stages=self.stages, producer_group=one, consumer_group=one,
            tx_count=do_bytes).make_participants()
        s_prod, s_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.s_mbar.data_ptr(), num_stages=2, producer_group=one,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_cw)).make_participants()
        p_prod, p_cons = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.p_mbar.data_ptr(), num_stages=1, producer_group=cw_group, consumer_group=one).make_participants()
        # per-block column data of the compute warps (2 stages)
        sLse = smem.allocate_tensor(Float32, cute.make_layout(2 * QB), byte_alignment=16)
        sKmin = smem.allocate_tensor(Int32, cute.make_layout(2 * QB), byte_alignment=16)
        sA = smem.allocate_tensor(BFloat16, a_layout.outer, byte_alignment=1024, swizzle=a_layout.inner)
        sQ = smem.allocate_tensor(BFloat16, q_layout.outer, byte_alignment=1024, swizzle=q_layout.inner)
        sDO = smem.allocate_tensor(BFloat16, do_layout.outer, byte_alignment=1024, swizzle=do_layout.inner)

        gA = cute.local_tile(mK, (self.TM, S), (None, None, None, None))  # (128, S, L/128, 1, Kh, B)
        gQ = cute.local_tile(mQ, (QB, S), (None, None, None, None))  # (QB, S, L/QB, 1, H, B)
        gDO = cute.local_tile(mDO, (VT, QB), (None, None, None, None))  # (VT, QB, V/VT, L/QB, H, B)
        thr_s = tm_s.get_slice(0)
        thr_o = tm_o.get_slice(0)
        tAs, tAg = cpasync.tma_partition(tma_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(thr_s.partition_A(gA), 0, 3))
        tQs, tQg = cpasync.tma_partition(tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(thr_s.partition_B(gQ), 0, 3))
        tDOs, tDOg = cpasync.tma_partition(tma_do, 0, cute.make_layout(1), cute.group_modes(sDO, 0, 3), cute.group_modes(thr_o.partition_B(gDO), 0, 3))
        tCrA = tm_s.make_fragment_A(sA)
        tCrQ = tm_s.make_fragment_B(sQ)
        tCrDO = tm_o.make_fragment_B(sDO)
        s_fake = tm_s.make_fragment_C(cute.append(tm_s.partition_shape_C((self.TM, QB)), 2))
        o_fake = tm_o.make_fragment_C(tm_o.partition_shape_C((self.TM, VT)))
        tP_fake = cute.make_tensor(cute.make_ptr(BFloat16, 0, cute.AddressSpace.tmem), p_layout.outer)
        tPr_fake = tm_o.make_fragment_A(tP_fake)[(None, None, None, 0)]

        if warp == self.tma_warp:
            ha = a_prod.acquire_and_advance()
            cute.copy(tma_a, tAg[(None, kt, 0, kh, b)], tAs[(None, ha.index)], tma_bar_ptr=ha.barrier)
            for j in cutlass.range(n_blocks):
                r = j // n_qb
                qb = qb_lo + j - r * n_qb
                h = kh * self.n_rep + r
                hq = q_prod.acquire_and_advance()
                cute.copy(tma_q, tQg[(None, qb, 0, h, b)], tQs[(None, hq.index)], tma_bar_ptr=hq.barrier)
                hd = do_prod.acquire_and_advance()
                cute.copy(tma_do, tDOg[(None, vt, qb, h, b)], tDOs[(None, hd.index)], tma_bar_ptr=hd.barrier)
            a_prod.tail()
            q_prod.tail()
            do_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tO = cute.make_tensor(tmem_ptr + self.tmem_o, o_fake.layout)
            tPr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_p, dtype=BFloat16), tPr_fake.layout)
            a_cons.wait()
            es = s_prod.acquire_and_advance()
            hq = q_cons.wait_and_advance()
            tS0 = tS[(None, None, None, es.index)]
            for kk in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                tm_s.set(tcgen05.Field.ACCUMULATE, kk > 0)
                cute.gemm(tm_s, tS0, tCrA[(None, None, kk, 0)], tCrQ[(None, None, kk, hq.index)], tS0)
            hq.release()
            es.commit()
            for j in cutlass.range(n_blocks):
                if j + 1 < n_blocks:
                    es = s_prod.acquire_and_advance()
                    hq = q_cons.wait_and_advance()
                    tSj = tS[(None, None, None, es.index)]
                    for kk in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        tm_s.set(tcgen05.Field.ACCUMULATE, kk > 0)
                        cute.gemm(tm_s, tSj, tCrA[(None, None, kk, 0)], tCrQ[(None, None, kk, hq.index)], tSj)
                    hq.release()
                    es.commit()
                hp = p_cons.wait_and_advance()
                hd = do_cons.wait_and_advance()
                for kk in cutlass.range_constexpr(cute.size(tPr, mode=[2])):
                    tm_o.set(tcgen05.Field.ACCUMULATE, (j > 0) | (kk > 0))
                    cute.gemm(tm_o, tO, tPr[(None, None, kk)], tCrDO[(None, None, kk, hd.index)], tO)
                hd.release()
                hp.release()
            a_cons.release()
            s_prod.tail()
        else:
            tmem.allocate(self.tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tO = cute.make_tensor(tmem_ptr + self.tmem_o, o_fake.layout)
            tPr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_p, dtype=BFloat16), tPr_fake.layout)
            trow = tidx % 128
            kpos = k0 + trow
            ld_s = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(QB), tcgen05.Pack.NONE), Float32)
            tS_r = [cute.flat_divide(tS[((None, None), 0, 0, i)], (self.TM, QB)) for i in range(2)]
            t2r_s = [tcgen05.make_tmem_copy(ld_s, tS_r[i][(None, None, 0, 0)]) for i in range(2)]
            tTR_s = [t2r_s[i].get_slice(trow).partition_S(tS_r[i]) for i in range(2)]
            tTR_s = [cute.group_modes(tTR_s[i], 3, cute.rank(tTR_s[i])) for i in range(2)]
            xs = cute.make_rmem_tensor(t2r_s[0].get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, QB))).shape, Float32)
            st_p = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(QB // 2), tcgen05.Unpack.NONE), BFloat16)
            r2t_p = tcgen05.make_tmem_copy(st_p, tPr)
            thr_p = r2t_p.get_slice(trow)
            tRT_tP = thr_p.partition_D(tPr)
            tRT_c = thr_p.partition_S(thr_o.partition_A(cute.make_identity_tensor((self.TM, QB))))
            pbf = cute.make_rmem_tensor(xs.shape, BFloat16)
            pbf_v = cute.make_tensor(pbf.iterator, cute.make_layout(tRT_c.shape))
            ld_o = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32), tcgen05.Pack.NONE), Float32)
            tO_r = cute.flat_divide(tO[((None, None), 0, 0)], (self.TM, 32))
            t2r_o = tcgen05.make_tmem_copy(ld_o, tO_r[(None, None, 0, 0)])
            tTR_o = t2r_o.get_slice(trow).partition_S(tO_r)
            tTR_o = cute.group_modes(tTR_o, 3, cute.rank(tTR_o))
            ov = cute.make_rmem_tensor(t2r_o.get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, 32))).shape, Float32)
            obf = cute.make_rmem_tensor(ov.shape, BFloat16)
            cw_bar = pipeline.NamedBarrier(barrier_id=3, num_threads=32 * self.num_cw)

            for j in cutlass.range(n_blocks):
                r = j // n_qb
                qb = qb_lo + j - r * n_qb
                h = kh * self.n_rep + r
                q0 = qb * QB
                stg = j % 2
                # column data of this block: lse2[q], kmin[q] (stage j % 2; the barrier of the previous block
                # ordered its readers before these writes)
                if tidx < QB:
                    sLse[stg * QB + tidx] = cute.make_tensor(pLse + ((b * H + h) * L + q0 + tidx), cute.make_layout(1))[0]
                    sKmin[stg * QB + tidx] = cute.make_tensor(pKmin + (b * L + q0 + tidx), cute.make_layout(1))[0]
                cw_bar.arrive_and_wait()
                fs = s_cons.wait_and_advance()
                if fs.index == 0:
                    cute.copy(t2r_s[0], tTR_s[0][(None, None, None, 0)], xs)
                else:
                    cute.copy(t2r_s[1], tTR_s[1][(None, None, None, 0)], xs)
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    fs.release()
                # mask: query q attends key k iff kmin[q] <= k <= q
                kmin_last = sKmin[stg * QB + QB - 1]
                if (q0 < k0 + self.TM - 1) | (kmin_last > k0):
                    for e in cutlass.range_constexpr(QB):
                        keep = (kpos >= sKmin[stg * QB + e]) & (kpos <= q0 + e)
                        xs[e] = cutlass.select_(keep, xs[e], Float32(float('-inf')))
                for e2 in cutlass.range_constexpr(QB // 2):
                    y0, y1 = cute.arch.fma_packed_f32x2((xs[2 * e2], xs[2 * e2 + 1]), (scale2, scale2),
                                                        (-sLse[stg * QB + 2 * e2], -sLse[stg * QB + 2 * e2 + 1]))
                    xs[2 * e2] = cute.math.exp2(y0, fastmath=True)
                    xs[2 * e2 + 1] = cute.math.exp2(y1, fastmath=True)
                pbf.store(xs.load().to(BFloat16))
                hp = p_prod.acquire_and_advance()
                cute.copy(r2t_p, pbf_v, tRT_tP)
                cute.arch.fence_view_async_tmem_store()
                hp.commit()
            hp = p_prod.acquire_and_advance()
            row_ptr = pDV + ((Int64(b * L + kpos) * K + kh) * V + vt * VT)
            for piece in cutlass.range_constexpr(VT // 32):
                cute.copy(t2r_o, tTR_o[(None, None, None, piece)], ov)
                cute.arch.fence_view_async_tmem_load()
                obf.store(ov.load().to(BFloat16))
                for e8 in cutlass.range_constexpr(4):
                    cute.make_tensor((row_ptr + (piece * 32 + e8 * 8)).align(16), cute.make_layout(8)).store(
                        cute.make_tensor(obf.iterator + e8 * 8, cute.make_layout(8)).load())
            tmem.relinquish_alloc_permit()
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_cw)
            tmem.free(tmem_ptr)


class ScaBwdDk:
    """dk[keys of a 128-row tile] = scale sum_q dS^T Q for one (batch, kv head, key tile) per CTA.

    Per 64-query block (heads of the kv group x attending blocks, as ScaBwdDv): S^T = K Q^T and dP^T = V dO^T
    (tcgen05; K tile resident in shared memory, V tile resident in TMEM as the A operand), the compute warps
    (thread = key row) form P^T = exp2(S^T scale2 - lse2[q]) and dS^T = P^T (dP^T - delta[q]) scale into the
    shared-memory A operand of dK += dS^T Q (the Q block is reused as MN-major B). TMEM: S^T, dP^T, dK, V.

    For V > 256 (V and dO tiles too large for shared memory) dP^T is read instead from the fp32 GEMMs
    dPcur[(c, b, h)] = V_c dO_c^T and dPprev[(c - 1, b, h)] = V_{c-1} dO_c^T computed on the host."""

    def __init__(self, H: int, K: int, S: int, V: int, C: int, QB: int = 64, stages: int = 2):
        self.H, self.K, self.S, self.V, self.C, self.QB = H, K, S, V, C, QB
        self.n_rep = H // K
        self.TM = 128
        self.load_dp = V > 256
        self.Vd = min(V, 256)  # width of the dO / V operand tiles (unused layouts in load_dp mode)
        self.stages = stages
        self.num_cw = 4
        self.mma_warp = 4
        self.tma_warp = 5
        self.threads = 32 * 6
        assert H % K == 0 and S % 64 == 0 and S <= 256 and V % 64 == 0 and C % 128 == 0 and C % QB == 0
        self.tmem_s = 0
        self.tmem_dp = QB
        self.tmem_dk = 2 * QB
        self.tmem_v = 2 * QB + S
        total = self.tmem_v + (0 if self.load_dp else V // 2)
        self.tmem_cols = max(32, 1 << (total - 1).bit_length())
        assert self.tmem_cols <= 512

    @cute.jit
    def __call__(self, tQ: cute.Tensor, tK: cute.Tensor, tV: cute.Tensor, tDO: cute.Tensor, tLse: cute.Tensor,
                 tDelta: cute.Tensor, tKmin: cute.Tensor, tQmax: cute.Tensor, tDK: cute.Tensor, tDPprev: cute.Tensor,
                 tDPcur: cute.Tensor, B: Int32, L: Int32, scale2: Float32, scale: Float32, stream: cuda.CUstream):
        H, K, S, V, QB = self.H, self.K, self.S, self.Vd, self.QB
        nc = L // self.C
        mK = cute.make_tensor(tK.iterator, cute.make_layout((L, S, K, B), stride=(K * S, 1, S, L * K * S)))
        mQ = cute.make_tensor(tQ.iterator, cute.make_layout((L, S, H, B), stride=(H * S, 1, S, L * H * S)))
        mDO = cute.make_tensor(tDO.iterator, cute.make_layout((L, V, H, B), stride=(H * self.V, 1, self.V, L * H * self.V)))
        tm_s = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, QB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        tm_dp = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, QB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        tm_dk = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, S, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))
        a_layout = sm100_utils.make_smem_layout_a(tm_s, (self.TM, QB, S), BFloat16, 1)
        q_layout = sm100_utils.make_smem_layout_b(tm_s, (self.TM, QB, S), BFloat16, self.stages)
        qmn_layout = sm100_utils.make_smem_layout_b(tm_dk, (self.TM, S, QB), BFloat16, self.stages)  # same bytes
        do_layout = sm100_utils.make_smem_layout_b(tm_dp, (self.TM, QB, V), BFloat16, self.stages)
        ds_layout = sm100_utils.make_smem_layout_a(tm_dk, (self.TM, S, QB), BFloat16, 1)
        v_layout = sm100_utils.make_smem_layout_a(tm_dp, (self.TM, QB, V), BFloat16, 1)  # TMEM A operand
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_a, mK_t = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mK, cute.slice_(a_layout, (None, None, None, 0)), (self.TM, QB, S), tm_s)
        tma_q, mQ_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mQ, cute.slice_(q_layout, (None, None, None, 0)), (self.TM, QB, S), tm_s)
        tma_do, mDO_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mDO, cute.slice_(do_layout, (None, None, None, 0)), (self.TM, QB, V), tm_dp)
        num_tiles = B * K * (L // self.TM)
        self.kernel(tm_s, tm_dp, tm_dk, tma_a, mK_t, tma_q, mQ_t, tma_do, mDO_t, tV.iterator, tLse.iterator, tDelta.iterator,
                    tKmin.iterator, tQmax.iterator, tDK.iterator, tDPprev.iterator, tDPcur.iterator, a_layout, q_layout,
                    qmn_layout, do_layout, ds_layout, v_layout, B, L, nc, scale2, scale).launch(
            grid=[num_tiles, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tm_s: cute.TiledMma, tm_dp: cute.TiledMma, tm_dk: cute.TiledMma, tma_a: cute.CopyAtom, mK: cute.Tensor,
               tma_q: cute.CopyAtom, mQ: cute.Tensor, tma_do: cute.CopyAtom, mDO: cute.Tensor, pV: cute.Pointer,
               pLse: cute.Pointer, pDelta: cute.Pointer, pKmin: cute.Pointer, pQmax: cute.Pointer, pDK: cute.Pointer,
               pDPprev: cute.Pointer, pDPcur: cute.Pointer, a_layout: cute.ComposedLayout, q_layout: cute.ComposedLayout,
               qmn_layout: cute.ComposedLayout, do_layout: cute.ComposedLayout, ds_layout: cute.ComposedLayout,
               v_layout: cute.ComposedLayout, B: Int32, L: Int32, nc: Int32, scale2: Float32, scale: Float32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H, K, S, V, QB, C = self.H, self.K, self.S, self.V, self.QB, self.C
        key_tiles = L // self.TM
        kt = bidx % key_tiles
        t1 = bidx // key_tiles
        kh = t1 % K
        b = t1 // K
        k0 = kt * self.TM
        ck = k0 // C
        qmax = cute.make_tensor(pQmax + (b * L + k0 + self.TM - 1), cute.make_layout(1))[0]
        q_end = (ck + 2) * C - 1
        if q_end > L - 1:
            q_end = L - 1
        if qmax < q_end:
            q_end = qmax
        qb_lo = k0 // QB
        qb_hi = q_end // QB
        n_qb = qb_hi - qb_lo + 1
        n_blocks = n_qb * self.n_rep

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_q)
            if const_expr(not self.load_dp):
                cpasync.prefetch_descriptor(tma_do)

        @cute.struct
        class SharedStorage:
            a_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            do_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            s_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            dp_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            ds_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_cw + 1))
        v_bar = pipeline.NamedBarrier(barrier_id=4, num_threads=32 * (self.num_cw + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cw_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * self.num_cw)
        cw_warps = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_cw)
        a_bytes = cute.size_in_bytes(BFloat16, cute.select(a_layout, mode=[0, 1, 2]))
        q_bytes = cute.size_in_bytes(BFloat16, cute.select(q_layout, mode=[0, 1, 2]))
        do_bytes = cute.size_in_bytes(BFloat16, cute.select(do_layout, mode=[0, 1, 2]))
        a_prod, a_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_mbar.data_ptr(), num_stages=1, producer_group=one, consumer_group=one,
            tx_count=a_bytes).make_participants()
        q_prod, q_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.q_mbar.data_ptr(), num_stages=self.stages, producer_group=one, consumer_group=one,
            tx_count=q_bytes).make_participants()
        do_prod, do_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.do_mbar.data_ptr(), num_stages=self.stages, producer_group=one, consumer_group=one,
            tx_count=do_bytes).make_participants()
        s_prod, s_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.s_mbar.data_ptr(), num_stages=1, producer_group=one, consumer_group=cw_warps).make_participants()
        dp_prod, dp_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.dp_mbar.data_ptr(), num_stages=1, producer_group=one, consumer_group=cw_warps).make_participants()
        ds_prod, ds_cons = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.ds_mbar.data_ptr(), num_stages=1, producer_group=cw_group, consumer_group=one).make_participants()
        sLse = smem.allocate_tensor(Float32, cute.make_layout(2 * QB), byte_alignment=16)
        sDelta = smem.allocate_tensor(Float32, cute.make_layout(2 * QB), byte_alignment=16)
        sKmin = smem.allocate_tensor(Int32, cute.make_layout(2 * QB), byte_alignment=16)
        sA = smem.allocate_tensor(BFloat16, a_layout.outer, byte_alignment=1024, swizzle=a_layout.inner)
        pQ = smem.allocate_array(BFloat16, cute.cosize(q_layout.outer), byte_alignment=1024)
        sQ = cute.make_tensor(cute.recast_ptr(pQ, swizzle_=q_layout.inner), q_layout.outer)
        sQmn = cute.make_tensor(cute.recast_ptr(pQ, swizzle_=qmn_layout.inner), qmn_layout.outer)
        if const_expr(not self.load_dp):
            sDO = smem.allocate_tensor(BFloat16, do_layout.outer, byte_alignment=1024, swizzle=do_layout.inner)
        pDSs = smem.allocate_array(BFloat16, cute.cosize(ds_layout.outer), byte_alignment=1024)
        sDS = cute.make_tensor(cute.recast_ptr(pDSs, swizzle_=ds_layout.inner), ds_layout.outer)

        gA = cute.local_tile(mK, (self.TM, S), (None, None, None, None))
        gQ = cute.local_tile(mQ, (QB, S), (None, None, None, None))
        thr_s = tm_s.get_slice(0)
        thr_dp = tm_dp.get_slice(0)
        tAs, tAg = cpasync.tma_partition(tma_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(thr_s.partition_A(gA), 0, 3))
        tQs, tQg = cpasync.tma_partition(tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(thr_s.partition_B(gQ), 0, 3))
        if const_expr(not self.load_dp):
            gDO = cute.local_tile(mDO, (QB, self.Vd), (None, None, None, None))
            tDOs, tDOg = cpasync.tma_partition(tma_do, 0, cute.make_layout(1), cute.group_modes(sDO, 0, 3), cute.group_modes(thr_dp.partition_B(gDO), 0, 3))
            tCrDO = tm_dp.make_fragment_B(sDO)
        tCrA = tm_s.make_fragment_A(sA)
        tCrQ = tm_s.make_fragment_B(sQ)
        tCrQmn = tm_dk.make_fragment_B(sQmn)
        tCrDS = tm_dk.make_fragment_A(sDS)
        s_fake = tm_s.make_fragment_C(tm_s.partition_shape_C((self.TM, QB)))
        dk_fake = tm_dk.make_fragment_C(tm_dk.partition_shape_C((self.TM, S)))
        tV_fake = cute.make_tensor(cute.make_ptr(BFloat16, 0, cute.AddressSpace.tmem), v_layout.outer)
        tVr_fake = tm_dp.make_fragment_A(tV_fake)[(None, None, None, 0)]

        if warp == self.tma_warp:
            ha = a_prod.acquire_and_advance()
            cute.copy(tma_a, tAg[(None, kt, 0, kh, b)], tAs[(None, ha.index)], tma_bar_ptr=ha.barrier)
            for j in cutlass.range(n_blocks):
                r = j // n_qb
                qb = qb_lo + j - r * n_qb
                h = kh * self.n_rep + r
                hq = q_prod.acquire_and_advance()
                cute.copy(tma_q, tQg[(None, qb, 0, h, b)], tQs[(None, hq.index)], tma_bar_ptr=hq.barrier)
                if const_expr(not self.load_dp):
                    hd = do_prod.acquire_and_advance()
                    cute.copy(tma_do, tDOg[(None, qb, 0, h, b)], tDOs[(None, hd.index)], tma_bar_ptr=hd.barrier)
            a_prod.tail()
            q_prod.tail()
            if const_expr(not self.load_dp):
                do_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tDP = cute.make_tensor(tmem_ptr + self.tmem_dp, s_fake.layout)
            tDK = cute.make_tensor(tmem_ptr + self.tmem_dk, dk_fake.layout)
            tVr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_v, dtype=BFloat16), tVr_fake.layout)
            if const_expr(not self.load_dp):
                v_bar.arrive_and_wait()  # V tile stored in TMEM by the compute warps
            a_cons.wait()
            # the Q stage of a block is held until its dK MMA: q_cons waits (one block ahead), q_rel releases
            q_rel = q_cons.clone()
            # S^T_0 and dP^T_0
            es = s_prod.acquire_and_advance()
            hq = q_cons.wait_and_advance()
            for kk in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                tm_s.set(tcgen05.Field.ACCUMULATE, kk > 0)
                cute.gemm(tm_s, tS, tCrA[(None, None, kk, 0)], tCrQ[(None, None, kk, hq.index)], tS)
            es.commit()
            if const_expr(not self.load_dp):
                ed = dp_prod.acquire_and_advance()
                hd = do_cons.wait_and_advance()
                for kk in cutlass.range_constexpr(cute.size(tVr, mode=[2])):
                    tm_dp.set(tcgen05.Field.ACCUMULATE, kk > 0)
                    cute.gemm(tm_dp, tDP, tVr[(None, None, kk)], tCrDO[(None, None, kk, hd.index)], tDP)
                hd.release()
                ed.commit()
            for j in cutlass.range(n_blocks):
                if j + 1 < n_blocks:
                    es = s_prod.acquire_and_advance()
                    hq = q_cons.wait_and_advance()
                    for kk in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        tm_s.set(tcgen05.Field.ACCUMULATE, kk > 0)
                        cute.gemm(tm_s, tS, tCrA[(None, None, kk, 0)], tCrQ[(None, None, kk, hq.index)], tS)
                    es.commit()
                    if const_expr(not self.load_dp):
                        ed = dp_prod.acquire_and_advance()
                        hd = do_cons.wait_and_advance()
                        for kk in cutlass.range_constexpr(cute.size(tVr, mode=[2])):
                            tm_dp.set(tcgen05.Field.ACCUMULATE, kk > 0)
                            cute.gemm(tm_dp, tDP, tVr[(None, None, kk)], tCrDO[(None, None, kk, hd.index)], tDP)
                        hd.release()
                        ed.commit()
                # dK += dS^T_j Q_j
                hs = ds_cons.wait_and_advance()
                qj = q_rel.current_handle().index
                for kk in cutlass.range_constexpr(cute.size(tCrDS, mode=[2])):
                    tm_dk.set(tcgen05.Field.ACCUMULATE, (j > 0) | (kk > 0))
                    cute.gemm(tm_dk, tDK, tCrDS[(None, None, kk, 0)], tCrQmn[(None, None, kk, qj)], tDK)
                q_rel.release()
                q_rel.advance()
                hs.release()
            a_cons.release()
            s_prod.tail()
            if const_expr(not self.load_dp):
                dp_prod.tail()
        else:
            tmem.allocate(self.tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tDP = cute.make_tensor(tmem_ptr + self.tmem_dp, s_fake.layout)
            tDK = cute.make_tensor(tmem_ptr + self.tmem_dk, dk_fake.layout)
            tVr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_v, dtype=BFloat16), tVr_fake.layout)
            trow = tidx % 128
            kpos = k0 + trow
            if const_expr(not self.load_dp):
                # V tile -> TMEM (A operand of dP^T): this thread's key row
                st_v = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(V // 2), tcgen05.Unpack.NONE), BFloat16)
                r2t_v = tcgen05.make_tmem_copy(st_v, tVr)
                thr_v = r2t_v.get_slice(trow)
                tRT_tV = thr_v.partition_D(tVr)
                tRT_cv = thr_v.partition_S(thr_dp.partition_A(cute.make_identity_tensor((self.TM, V))))
                rV = cute.make_rmem_tensor((V,), BFloat16)
                v_ptr = pV + (Int64(b * L + kpos) * K + kh) * V
                for e8 in cutlass.range_constexpr(V // 8):
                    cute.make_tensor(rV.iterator + e8 * 8, cute.make_layout(8)).store(
                        cute.make_tensor((v_ptr + e8 * 8).align(16), cute.make_layout(8)).load())
                cute.copy(r2t_v, cute.make_tensor(rV.iterator, cute.make_layout(tRT_cv.shape)), tRT_tV)
                cute.arch.fence_view_async_tmem_store()
                v_bar.arrive_and_wait()
            # dP^T matrices of this key tile's chunk (load_dp mode): rows = keys of chunk ck, columns = queries
            m_cur = Int64((b * nc + ck) * H) * C * C  # + h * C * C per head: dPcur [B, nc, H, C, C]
            m_prev = Int64((b * (nc - 1) + ck) * H) * C * C  # dPprev [B, nc - 1, H, C, C], index ck = c - 1
            dp_row = Int64(kpos - ck * C) * C
            # S^T / dP^T loads (QB columns of this thread's row)
            ld_s = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(QB), tcgen05.Pack.NONE), Float32)
            tS_r = cute.flat_divide(tS[((None, None), 0, 0)], (self.TM, QB))
            tDP_r = cute.flat_divide(tDP[((None, None), 0, 0)], (self.TM, QB))
            t2r_s = tcgen05.make_tmem_copy(ld_s, tS_r[(None, None, 0, 0)])
            t2r_dp = tcgen05.make_tmem_copy(ld_s, tDP_r[(None, None, 0, 0)])
            tTR_s = t2r_s.get_slice(trow).partition_S(tS_r)
            tTR_s = cute.group_modes(tTR_s, 3, cute.rank(tTR_s))
            tTR_dp = t2r_dp.get_slice(trow).partition_S(tDP_r)
            tTR_dp = cute.group_modes(tTR_dp, 3, cute.rank(tTR_dp))
            xs = cute.make_rmem_tensor(t2r_s.get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, QB))).shape, Float32)
            xd = cute.make_rmem_tensor(xs.shape, Float32)
            dsb = cute.make_rmem_tensor(xs.shape, BFloat16)
            # dK pieces (epilogue)
            ld_o = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32), tcgen05.Pack.NONE), Float32)
            tDK_r = cute.flat_divide(tDK[((None, None), 0, 0)], (self.TM, 32))
            t2r_o = tcgen05.make_tmem_copy(ld_o, tDK_r[(None, None, 0, 0)])
            tTR_o = t2r_o.get_slice(trow).partition_S(tDK_r)
            tTR_o = cute.group_modes(tTR_o, 3, cute.rank(tTR_o))
            ov = cute.make_rmem_tensor(t2r_o.get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, 32))).shape, Float32)
            obf = cute.make_rmem_tensor(ov.shape, BFloat16)
            cw_bar = pipeline.NamedBarrier(barrier_id=3, num_threads=32 * self.num_cw)
            # column data of block 0
            nxt_lse = Float32(0.0)
            nxt_delta = Float32(0.0)
            nxt_kmin = Int32(0)
            if tidx < QB:
                h0 = kh * self.n_rep
                nxt_lse = cute.make_tensor(pLse + ((b * H + h0) * L + qb_lo * QB + tidx), cute.make_layout(1))[0]
                nxt_delta = cute.make_tensor(pDelta + ((b * H + h0) * L + qb_lo * QB + tidx), cute.make_layout(1))[0]
                nxt_kmin = cute.make_tensor(pKmin + (b * L + qb_lo * QB + tidx), cute.make_layout(1))[0]
            for j in cutlass.range(n_blocks):
                r = j // n_qb
                qb = qb_lo + j - r * n_qb
                h = kh * self.n_rep + r
                q0 = qb * QB
                stg = j % 2
                if tidx < QB:
                    sLse[stg * QB + tidx] = nxt_lse
                    sDelta[stg * QB + tidx] = nxt_delta
                    sKmin[stg * QB + tidx] = nxt_kmin
                cw_bar.arrive_and_wait()
                # prefetch the next block's column data (latency hidden behind the S^T wait)
                if j + 1 < n_blocks:
                    if tidx < QB:
                        r1 = (j + 1) // n_qb
                        q1 = (qb_lo + (j + 1) - r1 * n_qb) * QB
                        h1 = kh * self.n_rep + r1
                        nxt_lse = cute.make_tensor(pLse + ((b * H + h1) * L + q1 + tidx), cute.make_layout(1))[0]
                        nxt_delta = cute.make_tensor(pDelta + ((b * H + h1) * L + q1 + tidx), cute.make_layout(1))[0]
                        nxt_kmin = cute.make_tensor(pKmin + (b * L + q1 + tidx), cute.make_layout(1))[0]
                # P^T
                fs = s_cons.wait_and_advance()
                cute.copy(t2r_s, tTR_s[(None, None, None, 0)], xs)
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    fs.release()
                kmin_last = sKmin[stg * QB + QB - 1]
                if (q0 < k0 + self.TM - 1) | (kmin_last > k0):
                    for e in cutlass.range_constexpr(QB):
                        keep = (kpos >= sKmin[stg * QB + e]) & (kpos <= q0 + e)
                        xs[e] = cutlass.select_(keep, xs[e], Float32(float('-inf')))
                for e2 in cutlass.range_constexpr(QB // 2):
                    y0, y1 = cute.arch.fma_packed_f32x2((xs[2 * e2], xs[2 * e2 + 1]), (scale2, scale2),
                                                        (-sLse[stg * QB + 2 * e2], -sLse[stg * QB + 2 * e2 + 1]))
                    xs[2 * e2] = cute.math.exp2(y0, fastmath=True)
                    xs[2 * e2 + 1] = cute.math.exp2(y1, fastmath=True)
                # dS^T = P^T (dP^T - delta) scale
                if const_expr(self.load_dp):
                    cq = q0 // C
                    dp_base = pDPcur + (m_cur + Int64(h) * C * C + dp_row + Int64(q0 - cq * C))
                    if cq != ck:
                        dp_base = pDPprev + (m_prev + Int64(h) * C * C + dp_row + Int64(q0 - cq * C))
                    for e4 in cutlass.range_constexpr(QB // 4):
                        cute.make_tensor(xd.iterator + e4 * 4, cute.make_layout(4)).store(
                            cute.make_tensor((dp_base + e4 * 4).align(16), cute.make_layout(4)).load())
                else:
                    fd = dp_cons.wait_and_advance()
                    cute.copy(t2r_dp, tTR_dp[(None, None, None, 0)], xd)
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        fd.release()
                for e in cutlass.range_constexpr(QB):
                    xd[e] = xs[e] * (xd[e] - sDelta[stg * QB + e]) * scale
                dsb.store(xd.load().to(BFloat16))
                # shared-memory A operand of dK (K-major, 128 x QB, 128-byte swizzle), free once dK_{j-1} is done
                hs = ds_prod.acquire_and_advance()
                for e8 in cutlass.range_constexpr(QB // 8):
                    cute.make_tensor((pDSs + (64 * trow + 8 * (e8 ^ (trow % 8)))).align(16), cute.make_layout(8)).store(
                        cute.make_tensor(dsb.iterator + e8 * 8, cute.make_layout(8)).load())
                cute.arch.fence_view_async_shared()
                hs.commit()
            # dK epilogue (the last dK MMA completed when the dS buffer is released again)
            hs = ds_prod.acquire_and_advance()
            row_ptr = pDK + (Int64(b * L + kpos) * K + kh) * S
            for piece in cutlass.range_constexpr(S // 32):
                cute.copy(t2r_o, tTR_o[(None, None, None, piece)], ov)
                cute.arch.fence_view_async_tmem_load()
                obf.store(ov.load().to(BFloat16))
                for e8 in cutlass.range_constexpr(4):
                    cute.make_tensor((row_ptr + (piece * 32 + e8 * 8)).align(16), cute.make_layout(8)).store(
                        cute.make_tensor(obf.iterator + e8 * 8, cute.make_layout(8)).load())
            tmem.relinquish_alloc_permit()
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_cw)
            tmem.free(tmem_ptr)


# barrier slots of ScaBwdDkDv (Int64 array in shared memory)
_S_FULL, _S_EMPTY, _DP_FULL, _DP_EMPTY, _P_FULL, _DS_FULL, _DS_EMPTY, _DONE = 0, 2, 4, 6, 8, 10, 12, 14


@cute.jit
def _dkdv_block(masked, g, slot, trow, kpos, q0, scale2, raw, sLse, sDel, sKm, pDSs, ds_elems,
                t2r_s0, t2r_s1, tTR_s0, tTR_s1, t2r_dp0, t2r_dp1, tTR_dp0, tTR_dp1, r2t_p0, r2t_p1, tRT_tP0, tRT_tP1,
                xs, xd, pbf, pbf_v, dsb, ph_sf, ph_dpf, ph_dse, QB, dbg):
    """One block of ScaBwdDkDv for compute group g (the thread's key row is kpos): P^T from S^T (masked: keys
    above the diagonal and outside the query's segment -> -inf), P^T into TMEM, dS^T = P^T (dP^T - delta) into
    shared memory. The two variants (masked / unmasked) are separate code regions on purpose: an if around the
    selects alone gets if-converted into every block. Returns the advanced barrier phases."""
    # S^T_j (its stage is freed by the dV MMA)
    cute.arch.mbarrier_wait(raw + (_S_FULL + g), ph_sf)
    ph_sf ^= 1
    if g == 0:
        cute.copy(t2r_s0, tTR_s0[(None, None, None, 0)], xs)
    else:
        cute.copy(t2r_s1, tTR_s1[(None, None, None, 0)], xs)
    cute.arch.fence_view_async_tmem_load()
    if const_expr(dbg == 0):
        if const_expr(masked):
            lo = kpos - q0
            for e4 in cutlass.range_constexpr(QB // 4):
                km4 = cute.make_tensor((sKm.iterator + (slot + e4 * 4)).align(16), cute.make_layout(4)).load()
                for i in cutlass.range_constexpr(4):
                    e = e4 * 4 + i
                    keep = (kpos >= km4[i]) & (e >= lo)
                    xs[e] = cutlass.select_(keep, xs[e], Float32(float('-inf')))
        for e4 in cutlass.range_constexpr(QB // 4):
            l4 = cute.make_tensor((sLse.iterator + (slot + e4 * 4)).align(16), cute.make_layout(4)).load()
            y0, y1 = cute.arch.fma_packed_f32x2((xs[4 * e4], xs[4 * e4 + 1]), (scale2, scale2), (-l4[0], -l4[1]))
            y2, y3 = cute.arch.fma_packed_f32x2((xs[4 * e4 + 2], xs[4 * e4 + 3]), (scale2, scale2), (-l4[2], -l4[3]))
            xs[4 * e4] = cute.math.exp2(y0, fastmath=True)
            xs[4 * e4 + 1] = cute.math.exp2(y1, fastmath=True)
            xs[4 * e4 + 2] = cute.math.exp2(y2, fastmath=True)
            xs[4 * e4 + 3] = cute.math.exp2(y3, fastmath=True)
        pbf.store(xs.load().to(BFloat16))
    if g == 0:
        cute.copy(r2t_p0, pbf_v, tRT_tP0)
    else:
        cute.copy(r2t_p1, pbf_v, tRT_tP1)
    cute.arch.fence_view_async_tmem_store()
    cute.arch.mbarrier_arrive(raw + (_P_FULL + g))
    # dP^T_j (its stage is freed here); dS^T = P^T (dP^T - delta), the scale is applied to dK in the epilogue
    cute.arch.mbarrier_wait(raw + (_DP_FULL + g), ph_dpf)
    ph_dpf ^= 1
    if g == 0:
        cute.copy(t2r_dp0, tTR_dp0[(None, None, None, 0)], xd)
    else:
        cute.copy(t2r_dp1, tTR_dp1[(None, None, None, 0)], xd)
    cute.arch.fence_view_async_tmem_load()
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(raw + (_DP_EMPTY + g))
    if const_expr(dbg == 0):
        for e4 in cutlass.range_constexpr(QB // 4):
            d4 = cute.make_tensor((sDel.iterator + (slot + e4 * 4)).align(16), cute.make_layout(4)).load()
            d0, d1 = cute.arch.sub_packed_f32x2((xd[4 * e4], xd[4 * e4 + 1]), (d4[0], d4[1]))
            d2, d3 = cute.arch.sub_packed_f32x2((xd[4 * e4 + 2], xd[4 * e4 + 3]), (d4[2], d4[3]))
            xd[4 * e4], xd[4 * e4 + 1] = cute.arch.mul_packed_f32x2((xs[4 * e4], xs[4 * e4 + 1]), (d0, d1))
            xd[4 * e4 + 2], xd[4 * e4 + 3] = cute.arch.mul_packed_f32x2((xs[4 * e4 + 2], xs[4 * e4 + 3]), (d2, d3))
        dsb.store(xd.load().to(BFloat16))
    cute.arch.mbarrier_wait(raw + (_DS_EMPTY + g), ph_dse)
    ph_dse ^= 1
    # K-major 128 x QB bf16 with the 64-byte swizzle: 16-byte chunk c of row m at m*QB + 8*(c ^ ((m >> 1) & 3))
    ds_base = pDSs + (g * ds_elems + QB * trow)
    for e8 in cutlass.range_constexpr(QB // 8):
        cute.make_tensor((ds_base + 8 * (e8 ^ ((trow >> 1) & 3))).align(16), cute.make_layout(8)).store(
            cute.make_tensor(dsb.iterator + e8 * 8, cute.make_layout(8)).load())
    cute.arch.fence_view_async_shared()
    cute.arch.mbarrier_arrive(raw + (_DS_FULL + g))
    return ph_sf, ph_dpf, ph_dse


@cute.jit
def _umma(op, acc, frA, frB, b_idx, acc0, bar):
    """acc (+)= A B over the K blocks of the fragments (A: 3 modes, B: 4 modes indexed by b_idx), then a tcgen05
    commit on bar; acc0 (static): accumulate into acc at the first K block. The tiled MMA is rebuilt from the
    op here: the ACCUMULATE field is staged state that must not leak out of the region of its creation."""
    tm = cute.make_tiled_mma(op)
    for kk in cutlass.range_constexpr(cute.size(frA, mode=[2])):
        tm.set(tcgen05.Field.ACCUMULATE, acc0 or kk > 0)
        cute.gemm(tm, acc, frA[(None, None, kk)], frB[(None, None, kk, b_idx)], acc)
    with cute.arch.elect_one():
        tcgen05.commit(bar)


class ScaBwdDkDv:
    """dk and dv of a 128-key tile in one pass, for one (batch, kv head, key tile) per CTA; needs
    1.5 (S + V) + 4 QB <= 512 TMEM columns (gekko-4B: S = V = 128, QB = 32).

    The N = 64 GEMMs of ScaBwdDv / ScaBwdDk with both operands in shared memory run at ~55% of the tensor-core
    rate (operand bandwidth), so here every GEMM has its A operand in tensor memory: the K and V tiles are
    stored there once (bf16, by the compute warps), P^T is written there over the S^T columns of its block,
    and only dK = dS^T Q reads its A operand (dS^T, bf16) from shared memory. With A in TMEM the 32-query
    blocks run at the full rate.

    Per block (heads of the kv group x attending blocks, as ScaBwdDv): S^T = K Q^T and dP^T = V dO^T, then
    two groups of 4 compute warps (thread = key row), alternating blocks, form P^T = exp2(S^T scale2 - lse2[q])
    and dS^T = P^T (dP^T - delta[q]); dV += P^T dO and dK += dS^T Q follow (dO / Q blocks reused as MN-major
    B; the scale of dS is applied to dK in the epilogue). The MMA warp issues, per block j, dP^T_{j+1}, dV_j,
    dK_{j-1} and S^T_{j+2}; an S^T / P^T stage is freed by the completion of its dV MMA, a dP^T stage by the
    compute warps after loading it, a dS^T buffer by its dK MMA.
    TMEM: K, V, S^T|P^T x 2, dP^T x 2, dV, dK."""

    def __init__(self, H: int, K: int, S: int, V: int, C: int, QB: int = 32, q_stages: int = 4, do_stages: int = 3):
        self.H, self.K, self.S, self.V, self.C, self.QB = H, K, S, V, C, QB
        self.n_rep = H // K
        self.TM = 128
        self.q_stages = q_stages
        self.do_stages = do_stages
        self.num_cw = 8  # two groups of 4
        self.mma_warp = 8
        self.tma_warp = 9
        self.threads = 32 * 10
        assert H % K == 0 and S % 64 == 0 and V % 64 == 0 and C % 128 == 0 and C % QB == 0 and QB == 32
        self.tmem_k = 0
        self.tmem_v = S // 2
        self.tmem_s = (S + V) // 2
        self.tmem_dp = self.tmem_s + 2 * QB
        self.tmem_dv = self.tmem_dp + 2 * QB
        self.tmem_dk = self.tmem_dv + V
        total = self.tmem_dk + S
        assert total <= 512
        self.tmem_cols = max(32, 1 << (total - 1).bit_length())
        self.dbg = int(os.environ.get("SCA_DBG", "0"))  # 1: skip the softmax / dS math, 3: MMAs only

    @cute.jit
    def __call__(self, tQ: cute.Tensor, tK: cute.Tensor, tV: cute.Tensor, tDO: cute.Tensor, tLse: cute.Tensor,
                 tDelta: cute.Tensor, tKmin: cute.Tensor, tQmax: cute.Tensor, tDK: cute.Tensor, tDV: cute.Tensor,
                 B: Int32, L: Int32, scale2: Float32, scale: Float32, stream: cuda.CUstream):
        H, K, S, V, QB = self.H, self.K, self.S, self.V, self.QB
        nc = L // self.C
        mQ = cute.make_tensor(tQ.iterator, cute.make_layout((L, S, H, B), stride=(H * S, 1, S, L * H * S)))
        mDO = cute.make_tensor(tDO.iterator, cute.make_layout((L, V, H, B), stride=(H * V, 1, V, L * H * V)))
        # S^T = K Q^T and dP^T = V dO^T: TMEM A (K-major), smem B (K-major)
        tm_s = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, QB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        # dV += P^T dO: TMEM A (K-major), smem B (MN-major)
        tm_dv = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, V, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))
        # dK += dS^T Q: smem A (K-major), smem B (MN-major)
        tm_dk = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, S, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))
        k_layout = sm100_utils.make_smem_layout_a(tm_s, (self.TM, QB, S), BFloat16, 1)  # TMEM A operands
        v_layout = sm100_utils.make_smem_layout_a(tm_s, (self.TM, QB, V), BFloat16, 1)
        p_layout = sm100_utils.make_smem_layout_a(tm_dv, (self.TM, V, QB), BFloat16, 1)
        q_layout = sm100_utils.make_smem_layout_b(tm_s, (self.TM, QB, S), BFloat16, self.q_stages)
        qmn_layout = sm100_utils.make_smem_layout_b(tm_dk, (self.TM, S, QB), BFloat16, self.q_stages)  # same bytes
        do_layout = sm100_utils.make_smem_layout_b(tm_s, (self.TM, QB, V), BFloat16, self.do_stages)
        domn_layout = sm100_utils.make_smem_layout_b(tm_dv, (self.TM, V, QB), BFloat16, self.do_stages)  # same bytes
        ds_layout = sm100_utils.make_smem_layout_a(tm_dk, (self.TM, S, QB), BFloat16, 2)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_q, mQ_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mQ, cute.slice_(q_layout, (None, None, None, 0)), (self.TM, QB, S), tm_s)
        tma_do, mDO_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mDO, cute.slice_(do_layout, (None, None, None, 0)), (self.TM, QB, V), tm_s)
        num_tiles = B * K * (L // self.TM)
        self.kernel(tm_s, tm_dv, tm_dk, tma_q, mQ_t, tma_do, mDO_t, tK.iterator, tV.iterator, tLse.iterator,
                    tDelta.iterator, tKmin.iterator, tQmax.iterator, tDK.iterator, tDV.iterator, k_layout, v_layout,
                    p_layout, q_layout, qmn_layout, do_layout, domn_layout, ds_layout, B, L, nc, scale2, scale).launch(
            grid=[num_tiles, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tm_s: cute.TiledMma, tm_dv: cute.TiledMma, tm_dk: cute.TiledMma, tma_q: cute.CopyAtom,
               mQ: cute.Tensor, tma_do: cute.CopyAtom, mDO: cute.Tensor, pK: cute.Pointer, pV: cute.Pointer,
               pLse: cute.Pointer, pDelta: cute.Pointer, pKmin: cute.Pointer, pQmax: cute.Pointer, pDK: cute.Pointer,
               pDV: cute.Pointer, k_layout: cute.ComposedLayout, v_layout: cute.ComposedLayout,
               p_layout: cute.ComposedLayout, q_layout: cute.ComposedLayout, qmn_layout: cute.ComposedLayout,
               do_layout: cute.ComposedLayout, domn_layout: cute.ComposedLayout, ds_layout: cute.ComposedLayout,
               B: Int32, L: Int32, nc: Int32, scale2: Float32, scale: Float32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H, K, S, V, QB, C = self.H, self.K, self.S, self.V, self.QB, self.C
        key_tiles = L // self.TM
        kt = bidx % key_tiles
        t1 = bidx // key_tiles
        kh = t1 % K
        b = t1 // K
        k0 = kt * self.TM
        ck = k0 // C
        qmax = cute.make_tensor(pQmax + (b * L + k0 + self.TM - 1), cute.make_layout(1))[0]
        q_end = (ck + 2) * C - 1
        if q_end > L - 1:
            q_end = L - 1
        if qmax < q_end:
            q_end = qmax
        qb_lo = k0 // QB
        qb_hi = q_end // QB
        n_qb = qb_hi - qb_lo + 1
        n_blocks = n_qb * self.n_rep

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_q)
            cpasync.prefetch_descriptor(tma_do)

        @cute.struct
        class SharedStorage:
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.q_stages * 2]
            do_mbar: cute.struct.MemRange[cutlass.Int64, self.do_stages * 2]
            raw_mbar: cute.struct.MemRange[cutlass.Int64, 16]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # raw barriers: [s_full x2, s_empty x2, dp_full x2, dp_empty x2, p_full x2, ds_full x2, ds_empty x2, done]
        raw = storage.raw_mbar.data_ptr()
        S_FULL, S_EMPTY, DP_FULL, DP_EMPTY, P_FULL, DS_FULL, DS_EMPTY, DONE = 0, 2, 4, 6, 8, 10, 12, 14
        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(raw + (S_FULL + i), 1)
                    cute.arch.mbarrier_init(raw + (S_EMPTY + i), 1)
                    cute.arch.mbarrier_init(raw + (DP_FULL + i), 1)
                    cute.arch.mbarrier_init(raw + (DP_EMPTY + i), 4)
                    cute.arch.mbarrier_init(raw + (P_FULL + i), 128)
                    cute.arch.mbarrier_init(raw + (DS_FULL + i), 128)
                    cute.arch.mbarrier_init(raw + (DS_EMPTY + i), 1)
                cute.arch.mbarrier_init(raw + DONE, 1)
            cute.arch.mbarrier_init_fence()
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_cw + 1))
        kv_bar = pipeline.NamedBarrier(barrier_id=4, num_threads=32 * (self.num_cw + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        q_bytes = cute.size_in_bytes(BFloat16, cute.select(q_layout, mode=[0, 1, 2]))
        do_bytes = cute.size_in_bytes(BFloat16, cute.select(do_layout, mode=[0, 1, 2]))
        q_prod, q_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.q_mbar.data_ptr(), num_stages=self.q_stages, producer_group=one, consumer_group=one,
            tx_count=q_bytes).make_participants()
        do_prod, do_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.do_mbar.data_ptr(), num_stages=self.do_stages, producer_group=one, consumer_group=one,
            tx_count=do_bytes).make_participants()
        cute.arch.barrier()
        # per-warp copies of a block's column data (lse2, delta, kmin of its QB queries)
        sLse = smem.allocate_tensor(Float32, cute.make_layout(self.num_cw * QB), byte_alignment=16)
        sDel = smem.allocate_tensor(Float32, cute.make_layout(self.num_cw * QB), byte_alignment=16)
        sKm = smem.allocate_tensor(Int32, cute.make_layout(self.num_cw * QB), byte_alignment=16)
        pQ = smem.allocate_array(BFloat16, cute.cosize(q_layout.outer), byte_alignment=1024)
        sQ = cute.make_tensor(cute.recast_ptr(pQ, swizzle_=q_layout.inner), q_layout.outer)
        sQmn = cute.make_tensor(cute.recast_ptr(pQ, swizzle_=qmn_layout.inner), qmn_layout.outer)
        pDOs = smem.allocate_array(BFloat16, cute.cosize(do_layout.outer), byte_alignment=1024)
        sDO = cute.make_tensor(cute.recast_ptr(pDOs, swizzle_=do_layout.inner), do_layout.outer)
        sDOmn = cute.make_tensor(cute.recast_ptr(pDOs, swizzle_=domn_layout.inner), domn_layout.outer)
        pDSs = smem.allocate_array(BFloat16, cute.cosize(ds_layout.outer), byte_alignment=1024)
        sDS = cute.make_tensor(cute.recast_ptr(pDSs, swizzle_=ds_layout.inner), ds_layout.outer)
        ds_stage_elems = cute.cosize(cute.select(ds_layout.outer, mode=[0, 1, 2]))

        gQ = cute.local_tile(mQ, (QB, S), (None, None, None, None))
        gDO = cute.local_tile(mDO, (QB, V), (None, None, None, None))
        thr_s = tm_s.get_slice(0)
        thr_dv = tm_dv.get_slice(0)
        tQs, tQg = cpasync.tma_partition(tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(thr_s.partition_B(gQ), 0, 3))
        tDOs, tDOg = cpasync.tma_partition(tma_do, 0, cute.make_layout(1), cute.group_modes(sDO, 0, 3), cute.group_modes(thr_s.partition_B(gDO), 0, 3))
        tCrQ = tm_s.make_fragment_B(sQ)
        tCrDO = tm_s.make_fragment_B(sDO)
        tCrQmn = tm_dk.make_fragment_B(sQmn)
        tCrDOmn = tm_dv.make_fragment_B(sDOmn)
        tCrDS = tm_dk.make_fragment_A(sDS)
        s_fake = tm_s.make_fragment_C(cute.append(tm_s.partition_shape_C((self.TM, QB)), 2))
        dv_fake = tm_dv.make_fragment_C(tm_dv.partition_shape_C((self.TM, V)))
        dk_fake = tm_dk.make_fragment_C(tm_dk.partition_shape_C((self.TM, S)))
        tK_fake = cute.make_tensor(cute.make_ptr(BFloat16, 0, cute.AddressSpace.tmem), k_layout.outer)
        tKr_fake = tm_s.make_fragment_A(tK_fake)[(None, None, None, 0)]
        tV_fake = cute.make_tensor(cute.make_ptr(BFloat16, 0, cute.AddressSpace.tmem), v_layout.outer)
        tVr_fake = tm_s.make_fragment_A(tV_fake)[(None, None, None, 0)]
        tP_fake = cute.make_tensor(cute.make_ptr(BFloat16, 0, cute.AddressSpace.tmem), p_layout.outer)
        tPr_fake = tm_dv.make_fragment_A(tP_fake)[(None, None, None, 0)]

        if warp == self.tma_warp:
            for j in cutlass.range(n_blocks):
                r = j // n_qb
                qb = qb_lo + j - r * n_qb
                h = kh * self.n_rep + r
                hq = q_prod.acquire_and_advance()
                cute.copy(tma_q, tQg[(None, qb, 0, h, b)], tQs[(None, hq.index)], tma_bar_ptr=hq.barrier)
                hd = do_prod.acquire_and_advance()
                cute.copy(tma_do, tDOg[(None, qb, 0, h, b)], tDOs[(None, hd.index)], tma_bar_ptr=hd.barrier)
            q_prod.tail()
            do_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tDP = cute.make_tensor(tmem_ptr + self.tmem_dp, s_fake.layout)
            tDV = cute.make_tensor(tmem_ptr + self.tmem_dv, dv_fake.layout)
            tDK = cute.make_tensor(tmem_ptr + self.tmem_dk, dk_fake.layout)
            tKr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_k, dtype=BFloat16), tKr_fake.layout)
            tVr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_v, dtype=BFloat16), tVr_fake.layout)
            tPr = [cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_s + i * QB, dtype=BFloat16), tPr_fake.layout) for i in range(2)]
            kv_bar.arrive_and_wait()  # K / V tiles stored in TMEM by the compute warps
            op_s = tcgen05.MmaF16BF16Op(BFloat16, Float32, (self.TM, QB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
                                        tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
            op_dv = tcgen05.MmaF16BF16Op(BFloat16, Float32, (self.TM, V, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
                                         tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN)
            op_dk = tcgen05.MmaF16BF16Op(BFloat16, Float32, (self.TM, S, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                                         tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN)
            q_rel = q_cons.clone()
            do_rel = do_cons.clone()

            if const_expr(self.dbg == 3):
                # MMAs only (no compute handshakes, garbage results): the tensor-pipe rate
                for j in cutlass.range(n_blocks):
                    hq = q_cons.wait_and_advance()
                    hd = do_cons.wait_and_advance()
                    _umma(op_s, tS[(None, None, None, 0)], tKr, tCrQ, hq.index, False, raw + S_FULL)
                    _umma(op_s, tDP[(None, None, None, 0)], tVr, tCrDO, hd.index, False, raw + DP_FULL)
                    _umma(op_dv, tDV, tPr[0], tCrDOmn, hd.index, True, raw + S_EMPTY)
                    _umma(op_dk, tDK, tCrDS[(None, None, None, 0)], tCrQmn, hq.index, True, raw + DS_EMPTY)
                    hq.release()
                    hd.release()
            else:
                # phases of the barriers this warp waits on
                # the first dP^T into stage 1 (block 1) has no load to wait for: start that barrier at parity 1
                ph_dpe0, ph_dpe1 = Int32(0), Int32(1)
                ph_pf0, ph_pf1 = Int32(0), Int32(0)
                ph_se0, ph_se1 = Int32(0), Int32(0)
                ph_dsf0, ph_dsf1 = Int32(0), Int32(0)
                # prologue: S^T_0, S^T_1, dP^T_0
                hq = q_cons.wait_and_advance()
                _umma(op_s, tS[(None, None, None, 0)], tKr, tCrQ, hq.index, False, raw + S_FULL)
                if n_blocks > 1:
                    hq = q_cons.wait_and_advance()
                    _umma(op_s, tS[(None, None, None, 1)], tKr, tCrQ, hq.index, False, raw + (S_FULL + 1))
                hd = do_cons.wait_and_advance()
                _umma(op_s, tDP[(None, None, None, 0)], tVr, tCrDO, hd.index, False, raw + DP_FULL)
                n_pairs = (n_blocks + 1) // 2
                for jp in cutlass.range(n_pairs):
                    # ---- block j = 2 jp (stage 0): dV_j, dK_{j-1}, dP^T_{j+1}, S^T_{j+2} ----
                    j = 2 * jp
                    cute.arch.mbarrier_wait(raw + (P_FULL + 0), ph_pf0)
                    ph_pf0 ^= 1
                    _umma(op_dv, tDV, tPr[0], tCrDOmn, do_rel.current_handle().index, True, raw + S_EMPTY)
                    do_rel.release()
                    do_rel.advance()
                    if j >= 1:  # dK_{j-1} (dS^T in buffer 1)
                        cute.arch.mbarrier_wait(raw + (DS_FULL + 1), ph_dsf1)
                        ph_dsf1 ^= 1
                        _umma(op_dk, tDK, tCrDS[(None, None, None, 1)], tCrQmn, q_rel.current_handle().index, True, raw + (DS_EMPTY + 1))
                        q_rel.release()
                        q_rel.advance()
                    if j + 1 < n_blocks:  # dP^T_{j+1} into stage 1 (freed by group 1 after loading dP^T_{j-1})
                        cute.arch.mbarrier_wait(raw + (DP_EMPTY + 1), ph_dpe1)
                        ph_dpe1 ^= 1
                        hd = do_cons.wait_and_advance()
                        _umma(op_s, tDP[(None, None, None, 1)], tVr, tCrDO, hd.index, False, raw + (DP_FULL + 1))
                    if j + 2 < n_blocks:  # S^T_{j+2} into stage 0 (freed by the completion of dV_j)
                        cute.arch.mbarrier_wait(raw + (S_EMPTY + 0), ph_se0)
                        ph_se0 ^= 1
                        hq = q_cons.wait_and_advance()
                        _umma(op_s, tS[(None, None, None, 0)], tKr, tCrQ, hq.index, False, raw + S_FULL)
                    # ---- block j + 1 (stage 1) ----
                    if j + 1 < n_blocks:
                        cute.arch.mbarrier_wait(raw + (P_FULL + 1), ph_pf1)
                        ph_pf1 ^= 1
                        _umma(op_dv, tDV, tPr[1], tCrDOmn, do_rel.current_handle().index, True, raw + (S_EMPTY + 1))
                        do_rel.release()
                        do_rel.advance()
                        cute.arch.mbarrier_wait(raw + (DS_FULL + 0), ph_dsf0)
                        ph_dsf0 ^= 1
                        _umma(op_dk, tDK, tCrDS[(None, None, None, 0)], tCrQmn, q_rel.current_handle().index, True, raw + DS_EMPTY)
                        q_rel.release()
                        q_rel.advance()
                        if j + 2 < n_blocks:
                            cute.arch.mbarrier_wait(raw + (DP_EMPTY + 0), ph_dpe0)
                            ph_dpe0 ^= 1
                            hd = do_cons.wait_and_advance()
                            _umma(op_s, tDP[(None, None, None, 0)], tVr, tCrDO, hd.index, False, raw + DP_FULL)
                        if j + 3 < n_blocks:
                            cute.arch.mbarrier_wait(raw + (S_EMPTY + 1), ph_se1)
                            ph_se1 ^= 1
                            hq = q_cons.wait_and_advance()
                            _umma(op_s, tS[(None, None, None, 1)], tKr, tCrQ, hq.index, False, raw + (S_FULL + 1))
                # dK_{n-1}
                if n_blocks % 2 == 1:
                    cute.arch.mbarrier_wait(raw + (DS_FULL + 0), ph_dsf0)
                    _umma(op_dk, tDK, tCrDS[(None, None, None, 0)], tCrQmn, q_rel.current_handle().index, True, raw + DS_EMPTY)
                else:
                    cute.arch.mbarrier_wait(raw + (DS_FULL + 1), ph_dsf1)
                    _umma(op_dk, tDK, tCrDS[(None, None, None, 1)], tCrQmn, q_rel.current_handle().index, True, raw + (DS_EMPTY + 1))
                q_rel.release()
                q_rel.advance()
            with cute.arch.elect_one():
                tcgen05.commit(raw + DONE)
        else:
            tmem.allocate(self.tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tDP = cute.make_tensor(tmem_ptr + self.tmem_dp, s_fake.layout)
            tDV = cute.make_tensor(tmem_ptr + self.tmem_dv, dv_fake.layout)
            tDK = cute.make_tensor(tmem_ptr + self.tmem_dk, dk_fake.layout)
            tKr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_k, dtype=BFloat16), tKr_fake.layout)
            tVr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_v, dtype=BFloat16), tVr_fake.layout)
            tPr = [cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_s + i * QB, dtype=BFloat16), tPr_fake.layout) for i in range(2)]
            grp = warp // 4  # compute group: blocks j with j % 2 == grp
            trow = tidx % 128
            kpos = k0 + trow
            # K / V tiles -> TMEM (A operands): group 0 stores K rows, group 1 V rows
            if grp == 0:
                st_k = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(S // 2), tcgen05.Unpack.NONE), BFloat16)
                r2t_k = tcgen05.make_tmem_copy(st_k, tKr)
                thr_k = r2t_k.get_slice(trow)
                tRT_tK = thr_k.partition_D(tKr)
                tRT_ck = thr_k.partition_S(thr_s.partition_A(cute.make_identity_tensor((self.TM, S))))
                rK = cute.make_rmem_tensor((S,), BFloat16)
                k_ptr = pK + (Int64(b * L + kpos) * K + kh) * S
                for e8 in cutlass.range_constexpr(S // 8):
                    cute.make_tensor(rK.iterator + e8 * 8, cute.make_layout(8)).store(
                        cute.make_tensor((k_ptr + e8 * 8).align(16), cute.make_layout(8)).load())
                cute.copy(r2t_k, cute.make_tensor(rK.iterator, cute.make_layout(tRT_ck.shape)), tRT_tK)
            else:
                st_v = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(V // 2), tcgen05.Unpack.NONE), BFloat16)
                r2t_v = tcgen05.make_tmem_copy(st_v, tVr)
                thr_v = r2t_v.get_slice(trow)
                tRT_tV = thr_v.partition_D(tVr)
                tRT_cv = thr_v.partition_S(thr_s.partition_A(cute.make_identity_tensor((self.TM, V))))
                rV = cute.make_rmem_tensor((V,), BFloat16)
                v_ptr = pV + (Int64(b * L + kpos) * K + kh) * V
                for e8 in cutlass.range_constexpr(V // 8):
                    cute.make_tensor(rV.iterator + e8 * 8, cute.make_layout(8)).store(
                        cute.make_tensor((v_ptr + e8 * 8).align(16), cute.make_layout(8)).load())
                cute.copy(r2t_v, cute.make_tensor(rV.iterator, cute.make_layout(tRT_cv.shape)), tRT_tV)
            # zero the dV (group 0) / dK (group 1) accumulators: the dV / dK MMAs always accumulate
            st_z = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(32), tcgen05.Unpack.NONE), Float32)
            zr = cute.make_rmem_tensor((32,), Float32)
            for e in cutlass.range_constexpr(32):
                zr[e] = Float32(0.0)
            tDV_z = cute.flat_divide(tDV[((None, None), 0, 0)], (self.TM, 32))
            tDK_z = cute.flat_divide(tDK[((None, None), 0, 0)], (self.TM, 32))
            r2t_zv = tcgen05.make_tmem_copy(st_z, tDV_z[(None, None, 0, 0)])
            r2t_zk = tcgen05.make_tmem_copy(st_z, tDK_z[(None, None, 0, 0)])
            tRT_zv = r2t_zv.get_slice(trow).partition_D(tDV_z)
            tRT_zv = cute.group_modes(tRT_zv, 3, cute.rank(tRT_zv))
            tRT_zk = r2t_zk.get_slice(trow).partition_D(tDK_z)
            tRT_zk = cute.group_modes(tRT_zk, 3, cute.rank(tRT_zk))
            zr_v = cute.make_tensor(zr.iterator, cute.make_layout(r2t_zv.get_slice(trow).partition_S(cute.make_identity_tensor((self.TM, 32))).shape))
            if grp == 0:
                for piece in cutlass.range_constexpr(V // 32):
                    cute.copy(r2t_zv, zr_v, tRT_zv[(None, None, None, piece)])
            else:
                for piece in cutlass.range_constexpr(S // 32):
                    cute.copy(r2t_zk, zr_v, tRT_zk[(None, None, None, piece)])
            cute.arch.fence_view_async_tmem_store()
            kv_bar.arrive_and_wait()
            # S^T / dP^T loads (QB columns of this thread's row, per stage)
            ld_s = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(QB), tcgen05.Pack.NONE), Float32)
            tS_r = [cute.flat_divide(tS[((None, None), 0, 0, i)], (self.TM, QB)) for i in range(2)]
            tDP_r = [cute.flat_divide(tDP[((None, None), 0, 0, i)], (self.TM, QB)) for i in range(2)]
            t2r_s = [tcgen05.make_tmem_copy(ld_s, tS_r[i][(None, None, 0, 0)]) for i in range(2)]
            t2r_dp = [tcgen05.make_tmem_copy(ld_s, tDP_r[i][(None, None, 0, 0)]) for i in range(2)]
            tTR_s = [t2r_s[i].get_slice(trow).partition_S(tS_r[i]) for i in range(2)]
            tTR_s = [cute.group_modes(tTR_s[i], 3, cute.rank(tTR_s[i])) for i in range(2)]
            tTR_dp = [t2r_dp[i].get_slice(trow).partition_S(tDP_r[i]) for i in range(2)]
            tTR_dp = [cute.group_modes(tTR_dp[i], 3, cute.rank(tTR_dp[i])) for i in range(2)]
            xs = cute.make_rmem_tensor(t2r_s[0].get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, QB))).shape, Float32)
            xd = cute.make_rmem_tensor(xs.shape, Float32)
            dsb = cute.make_rmem_tensor(xs.shape, BFloat16)
            # P^T stores (bf16 A operand over the S^T columns of the stage)
            st_p = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(QB // 2), tcgen05.Unpack.NONE), BFloat16)
            r2t_p = [tcgen05.make_tmem_copy(st_p, tPr[i]) for i in range(2)]
            tRT_tP = [r2t_p[i].get_slice(trow).partition_D(tPr[i]) for i in range(2)]
            tRT_c = r2t_p[0].get_slice(trow).partition_S(thr_dv.partition_A(cute.make_identity_tensor((self.TM, QB))))
            pbf = cute.make_rmem_tensor(xs.shape, BFloat16)
            pbf_v = cute.make_tensor(pbf.iterator, cute.make_layout(tRT_c.shape))
            # dV / dK pieces (epilogue)
            ld_o = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32), tcgen05.Pack.NONE), Float32)
            tDV_r = cute.flat_divide(tDV[((None, None), 0, 0)], (self.TM, 32))
            tDK_r = cute.flat_divide(tDK[((None, None), 0, 0)], (self.TM, 32))
            t2r_dv = tcgen05.make_tmem_copy(ld_o, tDV_r[(None, None, 0, 0)])
            t2r_dk = tcgen05.make_tmem_copy(ld_o, tDK_r[(None, None, 0, 0)])
            tTR_dv = t2r_dv.get_slice(trow).partition_S(tDV_r)
            tTR_dv = cute.group_modes(tTR_dv, 3, cute.rank(tTR_dv))
            tTR_dk = t2r_dk.get_slice(trow).partition_S(tDK_r)
            tTR_dk = cute.group_modes(tTR_dk, 3, cute.rank(tTR_dk))
            ov = cute.make_rmem_tensor(t2r_dv.get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, 32))).shape, Float32)
            obf = cute.make_rmem_tensor(ov.shape, BFloat16)
            g = grp  # this group's blocks: g, g + 2, ...
            n_own = (n_blocks - g + 1) // 2
            if const_expr(self.dbg == 3):
                n_own = Int32(0)
            ph_sf = Int32(0)
            ph_dpf = Int32(0)
            ph_dse = Int32(1)
            # (head r, query block qb) of the first own block; advanced by 2 blocks per iteration. The column
            # data of a block (lse2, delta, kmin of its QB queries; QB = 32 = one value per lane) is prefetched
            # one own block ahead into registers, then staged through a per-warp shared-memory slot.
            lane = tidx % 32
            slot = warp * QB
            r = g // n_qb
            qb = qb_lo + g - r * n_qb
            nxt_lse = Float32(0.0)
            nxt_del = Float32(0.0)
            nxt_km = Int32(0)
            nxt_kml = Int32(0)
            if n_own > 0:
                h = kh * self.n_rep + r
                nxt_lse = cute.make_tensor(pLse + ((b * H + h) * L + qb * QB + lane), cute.make_layout(1))[0]
                nxt_del = cute.make_tensor(pDelta + ((b * H + h) * L + qb * QB + lane), cute.make_layout(1))[0]
                nxt_km = cute.make_tensor(pKmin + (b * L + qb * QB + lane), cute.make_layout(1))[0]
                nxt_kml = cute.make_tensor(pKmin + (b * L + qb * QB + QB - 1), cute.make_layout(1))[0]
            for t in cutlass.range(n_own):
                q0 = qb * QB
                kmin_last = nxt_kml
                sLse[slot + lane] = nxt_lse
                sDel[slot + lane] = nxt_del
                sKm[slot + lane] = nxt_km
                cute.arch.sync_warp()
                # prefetch of the next own block (two blocks ahead; n_qb may be 1)
                qb_n = qb + 2
                r_n = r
                if qb_n > qb_hi:
                    qb_n = qb_n - n_qb
                    r_n = r_n + 1
                if qb_n > qb_hi:
                    qb_n = qb_n - n_qb
                    r_n = r_n + 1
                if t + 1 < n_own:
                    h_n = kh * self.n_rep + r_n
                    nxt_lse = cute.make_tensor(pLse + ((b * H + h_n) * L + qb_n * QB + lane), cute.make_layout(1))[0]
                    nxt_del = cute.make_tensor(pDelta + ((b * H + h_n) * L + qb_n * QB + lane), cute.make_layout(1))[0]
                    nxt_km = cute.make_tensor(pKmin + (b * L + qb_n * QB + lane), cute.make_layout(1))[0]
                    nxt_kml = cute.make_tensor(pKmin + (b * L + qb_n * QB + QB - 1), cute.make_layout(1))[0]
                masked = cute.arch.make_warp_uniform(Int32((q0 < k0 + self.TM - 1) | (kmin_last > k0)))
                if const_expr(self.dbg == 5):
                    masked = Int32(0)
                if masked != 0:
                    ph_sf, ph_dpf, ph_dse = _dkdv_block(
                        True, g, slot, trow, kpos, q0, scale2, raw, sLse, sDel, sKm, pDSs, ds_stage_elems,
                        t2r_s[0], t2r_s[1], tTR_s[0], tTR_s[1], t2r_dp[0], t2r_dp[1], tTR_dp[0], tTR_dp[1],
                        r2t_p[0], r2t_p[1], tRT_tP[0], tRT_tP[1], xs, xd, pbf, pbf_v, dsb, ph_sf, ph_dpf, ph_dse, QB, self.dbg)
                else:
                    ph_sf, ph_dpf, ph_dse = _dkdv_block(
                        False, g, slot, trow, kpos, q0, scale2, raw, sLse, sDel, sKm, pDSs, ds_stage_elems,
                        t2r_s[0], t2r_s[1], tTR_s[0], tTR_s[1], t2r_dp[0], t2r_dp[1], tTR_dp[0], tTR_dp[1],
                        r2t_p[0], r2t_p[1], tRT_tP[0], tRT_tP[1], xs, xd, pbf, pbf_v, dsb, ph_sf, ph_dpf, ph_dse, QB, self.dbg)
                qb = qb_n
                r = r_n
            # epilogue: group 0 writes dV, group 1 writes dK
            cute.arch.mbarrier_wait(raw + DONE, Int32(0))
            if grp == 0:
                row_ptr = pDV + (Int64(b * L + kpos) * K + kh) * V
                for piece in cutlass.range_constexpr(V // 32):
                    cute.copy(t2r_dv, tTR_dv[(None, None, None, piece)], ov)
                    cute.arch.fence_view_async_tmem_load()
                    obf.store(ov.load().to(BFloat16))
                    for e8 in cutlass.range_constexpr(4):
                        cute.make_tensor((row_ptr + (piece * 32 + e8 * 8)).align(16), cute.make_layout(8)).store(
                            cute.make_tensor(obf.iterator + e8 * 8, cute.make_layout(8)).load())
            else:
                row_ptr = pDK + (Int64(b * L + kpos) * K + kh) * S
                for piece in cutlass.range_constexpr(S // 32):
                    cute.copy(t2r_dk, tTR_dk[(None, None, None, piece)], ov)
                    cute.arch.fence_view_async_tmem_load()
                    obf.store((ov.load() * scale).to(BFloat16))
                    for e8 in cutlass.range_constexpr(4):
                        cute.make_tensor((row_ptr + (piece * 32 + e8 * 8)).align(16), cute.make_layout(8)).store(
                            cute.make_tensor(obf.iterator + e8 * 8, cute.make_layout(8)).load())
            tmem.relinquish_alloc_permit()
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_cw)
            tmem.free(tmem_ptr)


@cute.jit
def _dv_block(masked, g, slot, trow, kpos, q0, scale2, raw, sLse, sKm, t2r_s0, t2r_s1, tTR_s0, tTR_s1, r2t_p0, r2t_p1,
              tRT_tP0, tRT_tP1, xs, pbf, pbf_v, ph_sf, QB, dbg):
    """One block of ScaBwdDvT for compute group g: P^T = exp2(S^T scale2 - lse2) (masked: keys above the diagonal
    and outside the query's segment -> -inf) into TMEM over the S^T columns of its stage. Returns the phase."""
    cute.arch.mbarrier_wait(raw + (_S_FULL + g), ph_sf)
    ph_sf ^= 1
    if g == 0:
        cute.copy(t2r_s0, tTR_s0[(None, None, None, 0)], xs)
    else:
        cute.copy(t2r_s1, tTR_s1[(None, None, None, 0)], xs)
    cute.arch.fence_view_async_tmem_load()
    if const_expr(dbg == 0):
        if const_expr(masked):
            lo = kpos - q0
            for e4 in cutlass.range_constexpr(QB // 4):
                km4 = cute.make_tensor((sKm.iterator + (slot + e4 * 4)).align(16), cute.make_layout(4)).load()
                for i in cutlass.range_constexpr(4):
                    e = e4 * 4 + i
                    keep = (kpos >= km4[i]) & (e >= lo)
                    xs[e] = cutlass.select_(keep, xs[e], Float32(float('-inf')))
        for e4 in cutlass.range_constexpr(QB // 4):
            l4 = cute.make_tensor((sLse.iterator + (slot + e4 * 4)).align(16), cute.make_layout(4)).load()
            y0, y1 = cute.arch.fma_packed_f32x2((xs[4 * e4], xs[4 * e4 + 1]), (scale2, scale2), (-l4[0], -l4[1]))
            y2, y3 = cute.arch.fma_packed_f32x2((xs[4 * e4 + 2], xs[4 * e4 + 3]), (scale2, scale2), (-l4[2], -l4[3]))
            xs[4 * e4] = cute.math.exp2(y0, fastmath=True)
            xs[4 * e4 + 1] = cute.math.exp2(y1, fastmath=True)
            xs[4 * e4 + 2] = cute.math.exp2(y2, fastmath=True)
            xs[4 * e4 + 3] = cute.math.exp2(y3, fastmath=True)
        pbf.store(xs.load().to(BFloat16))
    if g == 0:
        cute.copy(r2t_p0, pbf_v, tRT_tP0)
    else:
        cute.copy(r2t_p1, pbf_v, tRT_tP1)
    cute.arch.fence_view_async_tmem_store()
    cute.arch.mbarrier_arrive(raw + (_P_FULL + g))
    return ph_sf


class ScaBwdDvT:
    """dv of a 128-key tile (one 256-wide v slab) for one (batch, kv head, key tile, v slab) per CTA, with the K
    tile as the TMEM A operand of S^T = K Q^T and P^T (bf16, over the S^T columns of its stage) as the A operand of
    dV += P^T dO: the shapes ScaBwdDkDv cannot hold (S + V > 256) but where S / 2 + 2 QB + min(V, 256) <= 512.
    ScaBwdDv keeps both operands of S^T in shared memory (~55% of the tensor-core rate on N = 64 blocks).

    Structure as ScaBwdDkDv without dK: 32-query blocks (heads of the kv group x attending blocks), S^T double
    buffered, two groups of 4 compute warps alternating blocks; the MMA warp issues dV_j then S^T_{j+2} (its stage
    is freed by the completion of dV_j). For V > 256 the CTAs of the V / 256 slabs each recompute S^T.
    TMEM: K, S^T|P^T x 2, dV."""

    def __init__(self, H: int, K: int, S: int, V: int, C: int, QB: int = 0, q_stages: int = 4, do_stages: int = 3):
        self.H, self.K, self.S, self.V, self.C = H, K, S, V, C
        self.n_rep = H // K
        self.TM = 128
        self.VT = min(V, 256)
        # 64-query blocks (half the MMA instructions) when they fit the 512 TMEM columns, else 32
        self.QB = QB if QB else (64 if S // 2 + 128 + self.VT <= 512 else 32)
        QB = self.QB
        self.n_vt = V // self.VT
        # shared memory: QB x S q stages + QB x VT dO stages (<= ~200 KB): 4 + 3 at QB = 32, 3 + 2 at QB = 64
        self.q_stages = q_stages if QB == 32 else 3
        self.do_stages = do_stages if QB == 32 else 2
        self.num_cw = 8  # two groups of 4
        self.mma_warp = 8
        self.tma_warp = 9
        self.threads = 32 * 10
        assert H % K == 0 and S % 64 == 0 and V % 64 == 0 and V % self.VT == 0 and C % 128 == 0 and C % QB == 0 and QB in (32, 64)
        self.tmem_k = 0
        self.tmem_s = S // 2
        self.tmem_dv = self.tmem_s + 2 * QB
        total = self.tmem_dv + self.VT
        assert total <= 512
        self.tmem_cols = max(32, 1 << (total - 1).bit_length())
        self.dbg = int(os.environ.get("SCA_DBG", "0"))

    @cute.jit
    def __call__(self, tQ: cute.Tensor, tK: cute.Tensor, tDO: cute.Tensor, tLse: cute.Tensor, tKmin: cute.Tensor,
                 tQmax: cute.Tensor, tDV: cute.Tensor, B: Int32, L: Int32, scale2: Float32, stream: cuda.CUstream):
        H, K, S, V, QB, VT = self.H, self.K, self.S, self.V, self.QB, self.VT
        nc = L // self.C
        mQ = cute.make_tensor(tQ.iterator, cute.make_layout((L, S, H, B), stride=(H * S, 1, S, L * H * S)))
        mDO = cute.make_tensor(tDO.iterator, cute.make_layout((L, V, H, B), stride=(H * V, 1, V, L * H * V)))
        tm_s = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, QB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        tm_dv = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, VT, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))
        k_layout = sm100_utils.make_smem_layout_a(tm_s, (self.TM, QB, S), BFloat16, 1)
        p_layout = sm100_utils.make_smem_layout_a(tm_dv, (self.TM, VT, QB), BFloat16, 1)
        q_layout = sm100_utils.make_smem_layout_b(tm_s, (self.TM, QB, S), BFloat16, self.q_stages)
        do_layout = sm100_utils.make_smem_layout_b(tm_s, (self.TM, QB, VT), BFloat16, self.do_stages)
        domn_layout = sm100_utils.make_smem_layout_b(tm_dv, (self.TM, VT, QB), BFloat16, self.do_stages)  # same bytes
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_q, mQ_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mQ, cute.slice_(q_layout, (None, None, None, 0)), (self.TM, QB, S), tm_s)
        tma_do, mDO_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mDO, cute.slice_(do_layout, (None, None, None, 0)), (self.TM, QB, VT), tm_s)
        num_tiles = B * K * (L // self.TM) * self.n_vt
        self.kernel(tm_s, tm_dv, tma_q, mQ_t, tma_do, mDO_t, tK.iterator, tLse.iterator, tKmin.iterator, tQmax.iterator,
                    tDV.iterator, k_layout, p_layout, q_layout, do_layout, domn_layout, B, L, nc, scale2).launch(
            grid=[num_tiles, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tm_s: cute.TiledMma, tm_dv: cute.TiledMma, tma_q: cute.CopyAtom, mQ: cute.Tensor, tma_do: cute.CopyAtom,
               mDO: cute.Tensor, pK: cute.Pointer, pLse: cute.Pointer, pKmin: cute.Pointer, pQmax: cute.Pointer,
               pDV: cute.Pointer, k_layout: cute.ComposedLayout, p_layout: cute.ComposedLayout, q_layout: cute.ComposedLayout,
               do_layout: cute.ComposedLayout, domn_layout: cute.ComposedLayout, B: Int32, L: Int32, nc: Int32, scale2: Float32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H, K, S, V, QB, C, VT = self.H, self.K, self.S, self.V, self.QB, self.C, self.VT
        key_tiles = L // self.TM
        vt = bidx % self.n_vt
        t0 = bidx // self.n_vt
        kt = t0 % key_tiles
        t1 = t0 // key_tiles
        kh = t1 % K
        b = t1 // K
        k0 = kt * self.TM
        ck = k0 // C
        qmax = cute.make_tensor(pQmax + (b * L + k0 + self.TM - 1), cute.make_layout(1))[0]
        q_end = (ck + 2) * C - 1
        if q_end > L - 1:
            q_end = L - 1
        if qmax < q_end:
            q_end = qmax
        qb_lo = k0 // QB
        qb_hi = q_end // QB
        n_qb = qb_hi - qb_lo + 1
        n_blocks = n_qb * self.n_rep

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_q)
            cpasync.prefetch_descriptor(tma_do)

        @cute.struct
        class SharedStorage:
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.q_stages * 2]
            do_mbar: cute.struct.MemRange[cutlass.Int64, self.do_stages * 2]
            raw_mbar: cute.struct.MemRange[cutlass.Int64, 16]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        raw = storage.raw_mbar.data_ptr()
        S_FULL, S_EMPTY, P_FULL, DONE = _S_FULL, _S_EMPTY, _P_FULL, _DONE
        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(raw + (S_FULL + i), 1)
                    cute.arch.mbarrier_init(raw + (S_EMPTY + i), 1)
                    cute.arch.mbarrier_init(raw + (P_FULL + i), 128)
                cute.arch.mbarrier_init(raw + DONE, 1)
            cute.arch.mbarrier_init_fence()
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_cw + 1))
        kv_bar = pipeline.NamedBarrier(barrier_id=4, num_threads=32 * (self.num_cw + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        q_bytes = cute.size_in_bytes(BFloat16, cute.select(q_layout, mode=[0, 1, 2]))
        do_bytes = cute.size_in_bytes(BFloat16, cute.select(do_layout, mode=[0, 1, 2]))
        q_prod, q_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.q_mbar.data_ptr(), num_stages=self.q_stages, producer_group=one, consumer_group=one,
            tx_count=q_bytes).make_participants()
        do_prod, do_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.do_mbar.data_ptr(), num_stages=self.do_stages, producer_group=one, consumer_group=one,
            tx_count=do_bytes).make_participants()
        cute.arch.barrier()
        sLse = smem.allocate_tensor(Float32, cute.make_layout(self.num_cw * QB), byte_alignment=16)
        sKm = smem.allocate_tensor(Int32, cute.make_layout(self.num_cw * QB), byte_alignment=16)
        pQ = smem.allocate_array(BFloat16, cute.cosize(q_layout.outer), byte_alignment=1024)
        sQ = cute.make_tensor(cute.recast_ptr(pQ, swizzle_=q_layout.inner), q_layout.outer)
        pDOs = smem.allocate_array(BFloat16, cute.cosize(do_layout.outer), byte_alignment=1024)
        sDO = cute.make_tensor(cute.recast_ptr(pDOs, swizzle_=do_layout.inner), do_layout.outer)
        sDOmn = cute.make_tensor(cute.recast_ptr(pDOs, swizzle_=domn_layout.inner), domn_layout.outer)

        gQ = cute.local_tile(mQ, (QB, S), (None, None, None, None))
        gDO = cute.local_tile(mDO, (QB, VT), (None, None, None, None))
        thr_s = tm_s.get_slice(0)
        thr_dv = tm_dv.get_slice(0)
        tQs, tQg = cpasync.tma_partition(tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(thr_s.partition_B(gQ), 0, 3))
        tDOs, tDOg = cpasync.tma_partition(tma_do, 0, cute.make_layout(1), cute.group_modes(sDO, 0, 3), cute.group_modes(thr_s.partition_B(gDO), 0, 3))
        tCrQ = tm_s.make_fragment_B(sQ)
        tCrDOmn = tm_dv.make_fragment_B(sDOmn)
        s_fake = tm_s.make_fragment_C(cute.append(tm_s.partition_shape_C((self.TM, QB)), 2))
        dv_fake = tm_dv.make_fragment_C(tm_dv.partition_shape_C((self.TM, VT)))
        tK_fake = cute.make_tensor(cute.make_ptr(BFloat16, 0, cute.AddressSpace.tmem), k_layout.outer)
        tKr_fake = tm_s.make_fragment_A(tK_fake)[(None, None, None, 0)]
        tP_fake = cute.make_tensor(cute.make_ptr(BFloat16, 0, cute.AddressSpace.tmem), p_layout.outer)
        tPr_fake = tm_dv.make_fragment_A(tP_fake)[(None, None, None, 0)]

        if warp == self.tma_warp:
            for j in cutlass.range(n_blocks):
                r = j // n_qb
                qb = qb_lo + j - r * n_qb
                h = kh * self.n_rep + r
                hq = q_prod.acquire_and_advance()
                cute.copy(tma_q, tQg[(None, qb, 0, h, b)], tQs[(None, hq.index)], tma_bar_ptr=hq.barrier)
                hd = do_prod.acquire_and_advance()
                cute.copy(tma_do, tDOg[(None, qb, vt, h, b)], tDOs[(None, hd.index)], tma_bar_ptr=hd.barrier)
            q_prod.tail()
            do_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tDV = cute.make_tensor(tmem_ptr + self.tmem_dv, dv_fake.layout)
            tKr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_k, dtype=BFloat16), tKr_fake.layout)
            tPr = [cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_s + i * QB, dtype=BFloat16), tPr_fake.layout) for i in range(2)]
            kv_bar.arrive_and_wait()  # K tile stored in TMEM by the compute warps
            op_s = tcgen05.MmaF16BF16Op(BFloat16, Float32, (self.TM, QB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
                                        tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
            op_dv = tcgen05.MmaF16BF16Op(BFloat16, Float32, (self.TM, VT, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
                                         tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN)
            if const_expr(self.dbg == 3):
                for j in cutlass.range(n_blocks):
                    hq = q_cons.wait_and_advance()
                    hd = do_cons.wait_and_advance()
                    _umma(op_s, tS[(None, None, None, 0)], tKr, tCrQ, hq.index, False, raw + S_FULL)
                    _umma(op_dv, tDV, tPr[0], tCrDOmn, hd.index, True, raw + S_EMPTY)
                    hq.release()
                    hd.release()
            else:
                ph_pf0, ph_pf1 = Int32(0), Int32(0)
                ph_se0, ph_se1 = Int32(0), Int32(0)
                # prologue: S^T_0, S^T_1 (the Q stage of a block is released by the completion of its S^T MMA)
                hq = q_cons.wait_and_advance()
                _umma(op_s, tS[(None, None, None, 0)], tKr, tCrQ, hq.index, False, raw + S_FULL)
                hq.release()
                if n_blocks > 1:
                    hq = q_cons.wait_and_advance()
                    _umma(op_s, tS[(None, None, None, 1)], tKr, tCrQ, hq.index, False, raw + (S_FULL + 1))
                    hq.release()
                n_pairs = (n_blocks + 1) // 2
                for jp in cutlass.range(n_pairs):
                    j = 2 * jp
                    cute.arch.mbarrier_wait(raw + (P_FULL + 0), ph_pf0)
                    ph_pf0 ^= 1
                    hd = do_cons.wait_and_advance()
                    _umma(op_dv, tDV, tPr[0], tCrDOmn, hd.index, True, raw + S_EMPTY)
                    hd.release()
                    if j + 2 < n_blocks:
                        cute.arch.mbarrier_wait(raw + (S_EMPTY + 0), ph_se0)
                        ph_se0 ^= 1
                        hq = q_cons.wait_and_advance()
                        _umma(op_s, tS[(None, None, None, 0)], tKr, tCrQ, hq.index, False, raw + S_FULL)
                        hq.release()
                    if j + 1 < n_blocks:
                        cute.arch.mbarrier_wait(raw + (P_FULL + 1), ph_pf1)
                        ph_pf1 ^= 1
                        hd = do_cons.wait_and_advance()
                        _umma(op_dv, tDV, tPr[1], tCrDOmn, hd.index, True, raw + (S_EMPTY + 1))
                        hd.release()
                        if j + 3 < n_blocks:
                            cute.arch.mbarrier_wait(raw + (S_EMPTY + 1), ph_se1)
                            ph_se1 ^= 1
                            hq = q_cons.wait_and_advance()
                            _umma(op_s, tS[(None, None, None, 1)], tKr, tCrQ, hq.index, False, raw + (S_FULL + 1))
                            hq.release()
            with cute.arch.elect_one():
                tcgen05.commit(raw + DONE)
        else:
            tmem.allocate(self.tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tDV = cute.make_tensor(tmem_ptr + self.tmem_dv, dv_fake.layout)
            tKr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_k, dtype=BFloat16), tKr_fake.layout)
            tPr = [cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_s + i * QB, dtype=BFloat16), tPr_fake.layout) for i in range(2)]
            grp = warp // 4
            trow = tidx % 128
            kpos = k0 + trow
            if grp == 0:
                st_k = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(S // 2), tcgen05.Unpack.NONE), BFloat16)
                r2t_k = tcgen05.make_tmem_copy(st_k, tKr)
                thr_k = r2t_k.get_slice(trow)
                tRT_tK = thr_k.partition_D(tKr)
                tRT_ck = thr_k.partition_S(thr_s.partition_A(cute.make_identity_tensor((self.TM, S))))
                rK = cute.make_rmem_tensor((S,), BFloat16)
                k_ptr = pK + (Int64(b * L + kpos) * K + kh) * S
                for e8 in cutlass.range_constexpr(S // 8):
                    cute.make_tensor(rK.iterator + e8 * 8, cute.make_layout(8)).store(
                        cute.make_tensor((k_ptr + e8 * 8).align(16), cute.make_layout(8)).load())
                cute.copy(r2t_k, cute.make_tensor(rK.iterator, cute.make_layout(tRT_ck.shape)), tRT_tK)
            else:
                # zero the dV accumulator (the dV MMAs always accumulate)
                st_z = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(32), tcgen05.Unpack.NONE), Float32)
                zr = cute.make_rmem_tensor((32,), Float32)
                for e in cutlass.range_constexpr(32):
                    zr[e] = Float32(0.0)
                tDV_z = cute.flat_divide(tDV[((None, None), 0, 0)], (self.TM, 32))
                r2t_zv = tcgen05.make_tmem_copy(st_z, tDV_z[(None, None, 0, 0)])
                tRT_zv = r2t_zv.get_slice(trow).partition_D(tDV_z)
                tRT_zv = cute.group_modes(tRT_zv, 3, cute.rank(tRT_zv))
                zr_v = cute.make_tensor(zr.iterator, cute.make_layout(r2t_zv.get_slice(trow).partition_S(cute.make_identity_tensor((self.TM, 32))).shape))
                for piece in cutlass.range_constexpr(VT // 32):
                    cute.copy(r2t_zv, zr_v, tRT_zv[(None, None, None, piece)])
            cute.arch.fence_view_async_tmem_store()
            kv_bar.arrive_and_wait()
            ld_s = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(QB), tcgen05.Pack.NONE), Float32)
            tS_r = [cute.flat_divide(tS[((None, None), 0, 0, i)], (self.TM, QB)) for i in range(2)]
            t2r_s = [tcgen05.make_tmem_copy(ld_s, tS_r[i][(None, None, 0, 0)]) for i in range(2)]
            tTR_s = [t2r_s[i].get_slice(trow).partition_S(tS_r[i]) for i in range(2)]
            tTR_s = [cute.group_modes(tTR_s[i], 3, cute.rank(tTR_s[i])) for i in range(2)]
            xs = cute.make_rmem_tensor(t2r_s[0].get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, QB))).shape, Float32)
            st_p = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(QB // 2), tcgen05.Unpack.NONE), BFloat16)
            r2t_p = [tcgen05.make_tmem_copy(st_p, tPr[i]) for i in range(2)]
            tRT_tP = [r2t_p[i].get_slice(trow).partition_D(tPr[i]) for i in range(2)]
            tRT_c = r2t_p[0].get_slice(trow).partition_S(thr_dv.partition_A(cute.make_identity_tensor((self.TM, QB))))
            pbf = cute.make_rmem_tensor(xs.shape, BFloat16)
            pbf_v = cute.make_tensor(pbf.iterator, cute.make_layout(tRT_c.shape))
            ld_o = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32), tcgen05.Pack.NONE), Float32)
            tDV_r = cute.flat_divide(tDV[((None, None), 0, 0)], (self.TM, 32))
            t2r_dv = tcgen05.make_tmem_copy(ld_o, tDV_r[(None, None, 0, 0)])
            tTR_dv = t2r_dv.get_slice(trow).partition_S(tDV_r)
            tTR_dv = cute.group_modes(tTR_dv, 3, cute.rank(tTR_dv))
            ov = cute.make_rmem_tensor(t2r_dv.get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, 32))).shape, Float32)
            obf = cute.make_rmem_tensor(ov.shape, BFloat16)
            g = grp
            n_own = (n_blocks - g + 1) // 2
            if const_expr(self.dbg == 3):
                n_own = Int32(0)
            ph_sf = Int32(0)
            lane = tidx % 32
            slot = warp * QB
            r = g // n_qb
            qb = qb_lo + g - r * n_qb
            NL = QB // 32  # column values per lane
            nxt_lse = cute.make_rmem_tensor(cute.make_layout(NL), Float32)
            nxt_km = cute.make_rmem_tensor(cute.make_layout(NL), Int32)
            nxt_kml = Int32(0)
            if n_own > 0:
                h = kh * self.n_rep + r
                for u in cutlass.range_constexpr(NL):
                    nxt_lse[u] = cute.make_tensor(pLse + ((b * H + h) * L + qb * QB + u * 32 + lane), cute.make_layout(1))[0]
                    nxt_km[u] = cute.make_tensor(pKmin + (b * L + qb * QB + u * 32 + lane), cute.make_layout(1))[0]
                nxt_kml = cute.make_tensor(pKmin + (b * L + qb * QB + QB - 1), cute.make_layout(1))[0]
            for t in cutlass.range(n_own):
                q0 = qb * QB
                kmin_last = nxt_kml
                for u in cutlass.range_constexpr(NL):
                    sLse[slot + u * 32 + lane] = nxt_lse[u]
                    sKm[slot + u * 32 + lane] = nxt_km[u]
                cute.arch.sync_warp()
                qb_n = qb + 2
                r_n = r
                if qb_n > qb_hi:
                    qb_n = qb_n - n_qb
                    r_n = r_n + 1
                if qb_n > qb_hi:
                    qb_n = qb_n - n_qb
                    r_n = r_n + 1
                if t + 1 < n_own:
                    h_n = kh * self.n_rep + r_n
                    for u in cutlass.range_constexpr(NL):
                        nxt_lse[u] = cute.make_tensor(pLse + ((b * H + h_n) * L + qb_n * QB + u * 32 + lane), cute.make_layout(1))[0]
                        nxt_km[u] = cute.make_tensor(pKmin + (b * L + qb_n * QB + u * 32 + lane), cute.make_layout(1))[0]
                    nxt_kml = cute.make_tensor(pKmin + (b * L + qb_n * QB + QB - 1), cute.make_layout(1))[0]
                masked = cute.arch.make_warp_uniform(Int32((q0 < k0 + self.TM - 1) | (kmin_last > k0)))
                if masked != 0:
                    ph_sf = _dv_block(True, g, slot, trow, kpos, q0, scale2, raw, sLse, sKm, t2r_s[0], t2r_s[1], tTR_s[0], tTR_s[1],
                                      r2t_p[0], r2t_p[1], tRT_tP[0], tRT_tP[1], xs, pbf, pbf_v, ph_sf, QB, self.dbg)
                else:
                    ph_sf = _dv_block(False, g, slot, trow, kpos, q0, scale2, raw, sLse, sKm, t2r_s[0], t2r_s[1], tTR_s[0], tTR_s[1],
                                      r2t_p[0], r2t_p[1], tRT_tP[0], tRT_tP[1], xs, pbf, pbf_v, ph_sf, QB, self.dbg)
                qb = qb_n
                r = r_n
            cute.arch.mbarrier_wait(raw + DONE, Int32(0))
            if grp == 0:
                row_ptr = pDV + (Int64(b * L + kpos) * K + kh) * V + vt * VT
                for piece in cutlass.range_constexpr(VT // 32):
                    cute.copy(t2r_dv, tTR_dv[(None, None, None, piece)], ov)
                    cute.arch.fence_view_async_tmem_load()
                    obf.store(ov.load().to(BFloat16))
                    for e8 in cutlass.range_constexpr(4):
                        cute.make_tensor((row_ptr + (piece * 32 + e8 * 8)).align(16), cute.make_layout(8)).store(
                            cute.make_tensor(obf.iterator + e8 * 8, cute.make_layout(8)).load())
            tmem.relinquish_alloc_permit()
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_cw)
            tmem.free(tmem_ptr)


@cute.jit
def _dk_block(masked, load_dp, g, slot, trow, kpos, q0, scale2, raw, sLse, sDel, sKm, pDSs, ds_elems,
              t2r_s0, t2r_s1, tTR_s0, tTR_s1, t2r_dp0, t2r_dp1, tTR_dp0, tTR_dp1, xs, xd, dsb, ph_sf, ph_dpf, ph_dse, QB, dbg,
              pDPc, pDPp, dp_off, dp_off_p, ck, h, C):
    """One block of ScaBwdDkT for compute group g: P^T = exp2(S^T scale2 - lse2) (masked variant: -inf outside the
    key's queries), dS^T = P^T (dP^T - delta) into the shared-memory A operand of dK; the S^T and dP^T stages are
    released by the warps after their loads. load_dp: dP^T is read from the host-side fp32 GEMMs (V > 256) instead
    of a TMEM stage: dPcur[(ck, b, h)][key, query] for the key tile's own chunk, dPprev[(ck, b, h)] for the next.
    Returns the advanced barrier phases."""
    cute.arch.mbarrier_wait(raw + (_S_FULL + g), ph_sf)
    ph_sf ^= 1
    if g == 0:
        cute.copy(t2r_s0, tTR_s0[(None, None, None, 0)], xs)
    else:
        cute.copy(t2r_s1, tTR_s1[(None, None, None, 0)], xs)
    cute.arch.fence_view_async_tmem_load()
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(raw + (_S_EMPTY + g))
    if const_expr(load_dp):
        cq = q0 // C
        dp_base = pDPc + (dp_off + Int64(h) * C * C + Int64(q0 - cq * C))
        if cq != ck:
            dp_base = pDPp + (dp_off_p + Int64(h) * C * C + Int64(q0 - cq * C))
        for e4 in cutlass.range_constexpr(QB // 4):
            cute.make_tensor(xd.iterator + e4 * 4, cute.make_layout(4)).store(
                cute.make_tensor((dp_base + e4 * 4).align(16), cute.make_layout(4)).load())
    if const_expr(dbg == 0):
        if const_expr(masked):
            lo = kpos - q0
            for e4 in cutlass.range_constexpr(QB // 4):
                km4 = cute.make_tensor((sKm.iterator + (slot + e4 * 4)).align(16), cute.make_layout(4)).load()
                for i in cutlass.range_constexpr(4):
                    e = e4 * 4 + i
                    keep = (kpos >= km4[i]) & (e >= lo)
                    xs[e] = cutlass.select_(keep, xs[e], Float32(float('-inf')))
        for e4 in cutlass.range_constexpr(QB // 4):
            l4 = cute.make_tensor((sLse.iterator + (slot + e4 * 4)).align(16), cute.make_layout(4)).load()
            y0, y1 = cute.arch.fma_packed_f32x2((xs[4 * e4], xs[4 * e4 + 1]), (scale2, scale2), (-l4[0], -l4[1]))
            y2, y3 = cute.arch.fma_packed_f32x2((xs[4 * e4 + 2], xs[4 * e4 + 3]), (scale2, scale2), (-l4[2], -l4[3]))
            xs[4 * e4] = cute.math.exp2(y0, fastmath=True)
            xs[4 * e4 + 1] = cute.math.exp2(y1, fastmath=True)
            xs[4 * e4 + 2] = cute.math.exp2(y2, fastmath=True)
            xs[4 * e4 + 3] = cute.math.exp2(y3, fastmath=True)
    if const_expr(not load_dp):
        cute.arch.mbarrier_wait(raw + (_DP_FULL + g), ph_dpf)
        ph_dpf ^= 1
        if g == 0:
            cute.copy(t2r_dp0, tTR_dp0[(None, None, None, 0)], xd)
        else:
            cute.copy(t2r_dp1, tTR_dp1[(None, None, None, 0)], xd)
        cute.arch.fence_view_async_tmem_load()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(raw + (_DP_EMPTY + g))
    if const_expr(dbg == 0):
        for e4 in cutlass.range_constexpr(QB // 4):
            d4 = cute.make_tensor((sDel.iterator + (slot + e4 * 4)).align(16), cute.make_layout(4)).load()
            d0, d1 = cute.arch.sub_packed_f32x2((xd[4 * e4], xd[4 * e4 + 1]), (d4[0], d4[1]))
            d2, d3 = cute.arch.sub_packed_f32x2((xd[4 * e4 + 2], xd[4 * e4 + 3]), (d4[2], d4[3]))
            xd[4 * e4], xd[4 * e4 + 1] = cute.arch.mul_packed_f32x2((xs[4 * e4], xs[4 * e4 + 1]), (d0, d1))
            xd[4 * e4 + 2], xd[4 * e4 + 3] = cute.arch.mul_packed_f32x2((xs[4 * e4 + 2], xs[4 * e4 + 3]), (d2, d3))
        dsb.store(xd.load().to(BFloat16))
    cute.arch.mbarrier_wait(raw + (_DS_EMPTY + g), ph_dse)
    ph_dse ^= 1
    # K-major 128 x QB bf16: QB = 32 -> 64-byte swizzle (c ^ ((m >> 1) & 3)), QB = 64 -> 128-byte swizzle (c ^ (m & 7))
    ds_base = pDSs + (g * ds_elems + QB * trow)
    for e8 in cutlass.range_constexpr(QB // 8):
        if const_expr(QB == 32):
            cute.make_tensor((ds_base + 8 * (e8 ^ ((trow >> 1) & 3))).align(16), cute.make_layout(8)).store(
                cute.make_tensor(dsb.iterator + e8 * 8, cute.make_layout(8)).load())
        else:
            cute.make_tensor((ds_base + 8 * (e8 ^ (trow % 8))).align(16), cute.make_layout(8)).store(
                cute.make_tensor(dsb.iterator + e8 * 8, cute.make_layout(8)).load())
    cute.arch.fence_view_async_shared()
    cute.arch.mbarrier_arrive(raw + (_DS_FULL + g))
    return ph_sf, ph_dpf, ph_dse


class ScaBwdDkT:
    """dk of a 128-key tile for one (batch, kv head, key tile) per CTA with the V tile as the TMEM A operand of
    dP^T = V dO^T: the shapes ScaBwdDkDv cannot hold (S + V > 256) but where V / 2 + 4 QB + S <= 512 (gekko-0.9B:
    S = V = 256 -> exactly 512 columns). The K tile stays in shared memory (TMA, A operand of S^T = K Q^T) and
    dS^T (bf16) is the shared-memory A operand of dK += dS^T Q; ScaBwdDk also keeps V in TMEM but runs 64-query
    blocks with a single stage of S^T / dP^T.

    Structure as ScaBwdDkDv without dV: 32-query blocks (heads of the kv group x attending blocks), S^T and dP^T
    double buffered (their stages are released by the compute warps after the loads), two groups of 4 compute warps
    alternating blocks; the MMA warp issues, per block j, dK_{j-1}, dP^T_{j+1} and S^T_{j+2}.
    TMEM: V, S^T x 2, dP^T x 2, dK."""

    def __init__(self, H: int, K: int, S: int, V: int, C: int, QB: int = 0, q_stages: int = 4, do_stages: int = 3, load_dp: bool = False):
        self.H, self.K, self.S, self.V, self.C = H, K, S, V, C
        self.n_rep = H // K
        self.TM = 128
        # load_dp (V > 256): dP^T comes from the host-side fp32 GEMMs, no V tile / dO blocks / dP^T MMAs; then 64-query
        # blocks fit (2 QB + S <= 512), else 32-query blocks with V in TMEM (V / 2 + 4 QB + S <= 512)
        self.load_dp = load_dp or V > 256
        self.Vd = 64 if self.load_dp else V  # width of the (unused) dO tiles in load_dp mode
        self.QB = QB if QB else (64 if self.load_dp else 32)
        QB = self.QB
        self.q_stages = q_stages if QB == 32 else 3
        self.do_stages = 1 if self.load_dp else do_stages
        self.num_cw = 8
        self.mma_warp = 8
        self.tma_warp = 9
        self.threads = 32 * 10
        assert H % K == 0 and S % 64 == 0 and V % 64 == 0 and (self.load_dp or V <= 256) and C % 128 == 0 and C % QB == 0 and QB in (32, 64)
        self.tmem_v = 0
        self.tmem_s = 0 if self.load_dp else V // 2
        self.tmem_dp = self.tmem_s + 2 * QB
        self.tmem_dk = self.tmem_dp + (0 if self.load_dp else 2 * QB)
        total = self.tmem_dk + S
        assert total <= 512
        self.tmem_cols = max(32, 1 << (total - 1).bit_length())
        self.dbg = int(os.environ.get("SCA_DBG", "0"))

    @cute.jit
    def __call__(self, tQ: cute.Tensor, tK: cute.Tensor, tV: cute.Tensor, tDO: cute.Tensor, tLse: cute.Tensor,
                 tDelta: cute.Tensor, tKmin: cute.Tensor, tQmax: cute.Tensor, tDK: cute.Tensor, tDPprev: cute.Tensor,
                 tDPcur: cute.Tensor, B: Int32, L: Int32, scale2: Float32, scale: Float32, stream: cuda.CUstream):
        H, K, S, QB = self.H, self.K, self.S, self.QB
        V = self.Vd
        nc = L // self.C
        mK = cute.make_tensor(tK.iterator, cute.make_layout((L, S, K, B), stride=(K * S, 1, S, L * K * S)))
        mQ = cute.make_tensor(tQ.iterator, cute.make_layout((L, S, H, B), stride=(H * S, 1, S, L * H * S)))
        mDO = cute.make_tensor(tDO.iterator, cute.make_layout((L, V, H, B), stride=(H * self.V, 1, self.V, L * H * self.V)))
        # S^T = K Q^T: smem A (K-major), smem B (K-major)
        tm_s = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, QB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        # dP^T = V dO^T: TMEM A, smem B (K-major)
        tm_dp = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, QB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        # dK += dS^T Q: smem A (K-major), smem B (MN-major)
        tm_dk = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, S, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))
        a_layout = sm100_utils.make_smem_layout_a(tm_s, (self.TM, QB, S), BFloat16, 1)
        v_layout = sm100_utils.make_smem_layout_a(tm_dp, (self.TM, QB, V), BFloat16, 1)  # TMEM A operand
        q_layout = sm100_utils.make_smem_layout_b(tm_s, (self.TM, QB, S), BFloat16, self.q_stages)
        qmn_layout = sm100_utils.make_smem_layout_b(tm_dk, (self.TM, S, QB), BFloat16, self.q_stages)  # same bytes
        do_layout = sm100_utils.make_smem_layout_b(tm_dp, (self.TM, QB, V), BFloat16, self.do_stages)
        ds_layout = sm100_utils.make_smem_layout_a(tm_dk, (self.TM, S, QB), BFloat16, 2)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_a, mK_t = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mK, cute.slice_(a_layout, (None, None, None, 0)), (self.TM, QB, S), tm_s)
        tma_q, mQ_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mQ, cute.slice_(q_layout, (None, None, None, 0)), (self.TM, QB, S), tm_s)
        tma_do, mDO_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mDO, cute.slice_(do_layout, (None, None, None, 0)), (self.TM, QB, V), tm_dp)
        num_tiles = B * K * (L // self.TM)
        self.kernel(tm_s, tm_dp, tm_dk, tma_a, mK_t, tma_q, mQ_t, tma_do, mDO_t, tV.iterator, tLse.iterator,
                    tDelta.iterator, tKmin.iterator, tQmax.iterator, tDK.iterator, tDPprev.iterator, tDPcur.iterator, a_layout, v_layout,
                    q_layout, qmn_layout, do_layout, ds_layout, B, L, nc, scale2, scale).launch(
            grid=[num_tiles, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tm_s: cute.TiledMma, tm_dp: cute.TiledMma, tm_dk: cute.TiledMma, tma_a: cute.CopyAtom,
               mK: cute.Tensor, tma_q: cute.CopyAtom, mQ: cute.Tensor, tma_do: cute.CopyAtom, mDO: cute.Tensor,
               pV: cute.Pointer, pLse: cute.Pointer, pDelta: cute.Pointer, pKmin: cute.Pointer, pQmax: cute.Pointer,
               pDK: cute.Pointer, pDPprev: cute.Pointer, pDPcur: cute.Pointer, a_layout: cute.ComposedLayout, v_layout: cute.ComposedLayout,
               q_layout: cute.ComposedLayout, qmn_layout: cute.ComposedLayout, do_layout: cute.ComposedLayout,
               ds_layout: cute.ComposedLayout, B: Int32, L: Int32, nc: Int32, scale2: Float32, scale: Float32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H, K, S, QB, C = self.H, self.K, self.S, self.QB, self.C
        V = self.Vd
        key_tiles = L // self.TM
        kt = bidx % key_tiles
        t1 = bidx // key_tiles
        kh = t1 % K
        b = t1 // K
        k0 = kt * self.TM
        ck = k0 // C
        qmax = cute.make_tensor(pQmax + (b * L + k0 + self.TM - 1), cute.make_layout(1))[0]
        q_end = (ck + 2) * C - 1
        if q_end > L - 1:
            q_end = L - 1
        if qmax < q_end:
            q_end = qmax
        qb_lo = k0 // QB
        qb_hi = q_end // QB
        n_qb = qb_hi - qb_lo + 1
        n_blocks = n_qb * self.n_rep

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_q)
            cpasync.prefetch_descriptor(tma_do)

        @cute.struct
        class SharedStorage:
            a_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.q_stages * 2]
            do_mbar: cute.struct.MemRange[cutlass.Int64, self.do_stages * 2]
            raw_mbar: cute.struct.MemRange[cutlass.Int64, 16]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        raw = storage.raw_mbar.data_ptr()
        S_FULL, S_EMPTY, DP_FULL, DP_EMPTY, DS_FULL, DS_EMPTY, DONE = _S_FULL, _S_EMPTY, _DP_FULL, _DP_EMPTY, _DS_FULL, _DS_EMPTY, _DONE
        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(raw + (S_FULL + i), 1)
                    cute.arch.mbarrier_init(raw + (S_EMPTY + i), 4)
                    cute.arch.mbarrier_init(raw + (DP_FULL + i), 1)
                    cute.arch.mbarrier_init(raw + (DP_EMPTY + i), 4)
                    cute.arch.mbarrier_init(raw + (DS_FULL + i), 128)
                    cute.arch.mbarrier_init(raw + (DS_EMPTY + i), 1)
                cute.arch.mbarrier_init(raw + DONE, 1)
            cute.arch.mbarrier_init_fence()
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_cw + 1))
        kv_bar = pipeline.NamedBarrier(barrier_id=4, num_threads=32 * (self.num_cw + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        a_bytes = cute.size_in_bytes(BFloat16, cute.select(a_layout, mode=[0, 1, 2]))
        q_bytes = cute.size_in_bytes(BFloat16, cute.select(q_layout, mode=[0, 1, 2]))
        do_bytes = cute.size_in_bytes(BFloat16, cute.select(do_layout, mode=[0, 1, 2]))
        a_prod, a_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_mbar.data_ptr(), num_stages=1, producer_group=one, consumer_group=one,
            tx_count=a_bytes).make_participants()
        q_prod, q_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.q_mbar.data_ptr(), num_stages=self.q_stages, producer_group=one, consumer_group=one,
            tx_count=q_bytes).make_participants()
        do_prod, do_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.do_mbar.data_ptr(), num_stages=self.do_stages, producer_group=one, consumer_group=one,
            tx_count=do_bytes).make_participants()
        cute.arch.barrier()
        sLse = smem.allocate_tensor(Float32, cute.make_layout(self.num_cw * QB), byte_alignment=16)
        sDel = smem.allocate_tensor(Float32, cute.make_layout(self.num_cw * QB), byte_alignment=16)
        sKm = smem.allocate_tensor(Int32, cute.make_layout(self.num_cw * QB), byte_alignment=16)
        pA = smem.allocate_array(BFloat16, cute.cosize(a_layout.outer), byte_alignment=1024)
        sA = cute.make_tensor(cute.recast_ptr(pA, swizzle_=a_layout.inner), a_layout.outer)
        pQ = smem.allocate_array(BFloat16, cute.cosize(q_layout.outer), byte_alignment=1024)
        sQ = cute.make_tensor(cute.recast_ptr(pQ, swizzle_=q_layout.inner), q_layout.outer)
        sQmn = cute.make_tensor(cute.recast_ptr(pQ, swizzle_=qmn_layout.inner), qmn_layout.outer)
        pDOs = smem.allocate_array(BFloat16, cute.cosize(do_layout.outer), byte_alignment=1024)
        sDO = cute.make_tensor(cute.recast_ptr(pDOs, swizzle_=do_layout.inner), do_layout.outer)
        pDSs = smem.allocate_array(BFloat16, cute.cosize(ds_layout.outer), byte_alignment=1024)
        sDS = cute.make_tensor(cute.recast_ptr(pDSs, swizzle_=ds_layout.inner), ds_layout.outer)
        ds_stage_elems = cute.cosize(cute.select(ds_layout.outer, mode=[0, 1, 2]))

        gA = cute.local_tile(mK, (self.TM, S), (None, None, None, None))
        gQ = cute.local_tile(mQ, (QB, S), (None, None, None, None))
        gDO = cute.local_tile(mDO, (QB, V), (None, None, None, None))
        thr_s = tm_s.get_slice(0)
        thr_dp = tm_dp.get_slice(0)
        tAs, tAg = cpasync.tma_partition(tma_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(thr_s.partition_A(gA), 0, 3))
        tQs, tQg = cpasync.tma_partition(tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(thr_s.partition_B(gQ), 0, 3))
        tDOs, tDOg = cpasync.tma_partition(tma_do, 0, cute.make_layout(1), cute.group_modes(sDO, 0, 3), cute.group_modes(thr_dp.partition_B(gDO), 0, 3))
        tCrA = tm_s.make_fragment_A(sA)
        tCrQ = tm_s.make_fragment_B(sQ)
        tCrDO = tm_dp.make_fragment_B(sDO)
        tCrQmn = tm_dk.make_fragment_B(sQmn)
        tCrDS = tm_dk.make_fragment_A(sDS)
        s_fake = tm_s.make_fragment_C(cute.append(tm_s.partition_shape_C((self.TM, QB)), 2))
        dk_fake = tm_dk.make_fragment_C(tm_dk.partition_shape_C((self.TM, S)))
        tV_fake = cute.make_tensor(cute.make_ptr(BFloat16, 0, cute.AddressSpace.tmem), v_layout.outer)
        tVr_fake = tm_dp.make_fragment_A(tV_fake)[(None, None, None, 0)]

        if warp == self.tma_warp:
            ha = a_prod.acquire_and_advance()
            cute.copy(tma_a, tAg[(None, kt, 0, kh, b)], tAs[(None, ha.index)], tma_bar_ptr=ha.barrier)
            for j in cutlass.range(n_blocks):
                r = j // n_qb
                qb = qb_lo + j - r * n_qb
                h = kh * self.n_rep + r
                hq = q_prod.acquire_and_advance()
                cute.copy(tma_q, tQg[(None, qb, 0, h, b)], tQs[(None, hq.index)], tma_bar_ptr=hq.barrier)
                if const_expr(not self.load_dp):
                    hd = do_prod.acquire_and_advance()
                    cute.copy(tma_do, tDOg[(None, qb, 0, h, b)], tDOs[(None, hd.index)], tma_bar_ptr=hd.barrier)
            a_prod.tail()
            q_prod.tail()
            if const_expr(not self.load_dp):
                do_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tDP = cute.make_tensor(tmem_ptr + self.tmem_dp, s_fake.layout)
            tDK = cute.make_tensor(tmem_ptr + self.tmem_dk, dk_fake.layout)
            tVr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_v, dtype=BFloat16), tVr_fake.layout)
            kv_bar.arrive_and_wait()  # V tile stored in TMEM by the compute warps
            a_cons.wait()
            tCrA0 = tCrA[(None, None, None, 0)]
            op_s = tcgen05.MmaF16BF16Op(BFloat16, Float32, (self.TM, QB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                                        tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
            op_dp = tcgen05.MmaF16BF16Op(BFloat16, Float32, (self.TM, QB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
                                         tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
            op_dk = tcgen05.MmaF16BF16Op(BFloat16, Float32, (self.TM, S, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                                         tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN)
            q_rel = q_cons.clone()
            if const_expr(self.dbg == 3):
                for j in cutlass.range(n_blocks):
                    hq = q_cons.wait_and_advance()
                    if const_expr(not self.load_dp):
                        hd = do_cons.wait_and_advance()
                    _umma(op_s, tS[(None, None, None, 0)], tCrA0, tCrQ, hq.index, False, raw + S_FULL)
                    if const_expr(not self.load_dp):
                        _umma(op_dp, tDP[(None, None, None, 0)], tVr, tCrDO, hd.index, False, raw + DP_FULL)
                        hd.release()
                    _umma(op_dk, tDK, tCrDS[(None, None, None, 0)], tCrQmn, hq.index, True, raw + DS_EMPTY)
                    hq.release()
            else:
                # both stages are filled in the prologue: the first wait on an empty barrier is for its first release
                ph_se0, ph_se1 = Int32(0), Int32(0)
                ph_dpe0, ph_dpe1 = Int32(0), Int32(0)
                ph_dsf0, ph_dsf1 = Int32(0), Int32(0)
                # prologue: S^T_0, dP^T_0, S^T_1, dP^T_1
                hq = q_cons.wait_and_advance()
                _umma(op_s, tS[(None, None, None, 0)], tCrA0, tCrQ, hq.index, False, raw + S_FULL)
                if const_expr(not self.load_dp):
                    hd = do_cons.wait_and_advance()
                    _umma(op_dp, tDP[(None, None, None, 0)], tVr, tCrDO, hd.index, False, raw + DP_FULL)
                    hd.release()
                if n_blocks > 1:
                    hq = q_cons.wait_and_advance()
                    _umma(op_s, tS[(None, None, None, 1)], tCrA0, tCrQ, hq.index, False, raw + (S_FULL + 1))
                    if const_expr(not self.load_dp):
                        hd = do_cons.wait_and_advance()
                        _umma(op_dp, tDP[(None, None, None, 1)], tVr, tCrDO, hd.index, False, raw + (DP_FULL + 1))
                        hd.release()
                n_pairs = (n_blocks + 1) // 2
                for jp in cutlass.range(n_pairs):
                    j = 2 * jp
                    # ---- block j (stage 0): dK_{j-1}, then S^T_{j+2} / dP^T_{j+2} into stage 0 once the warps freed it
                    if j >= 1:
                        cute.arch.mbarrier_wait(raw + (DS_FULL + 1), ph_dsf1)
                        ph_dsf1 ^= 1
                        _umma(op_dk, tDK, tCrDS[(None, None, None, 1)], tCrQmn, q_rel.current_handle().index, True, raw + (DS_EMPTY + 1))
                        q_rel.release()
                        q_rel.advance()
                    if j + 2 < n_blocks:
                        cute.arch.mbarrier_wait(raw + (S_EMPTY + 0), ph_se0)
                        ph_se0 ^= 1
                        hq = q_cons.wait_and_advance()
                        _umma(op_s, tS[(None, None, None, 0)], tCrA0, tCrQ, hq.index, False, raw + S_FULL)
                        if const_expr(not self.load_dp):
                            cute.arch.mbarrier_wait(raw + (DP_EMPTY + 0), ph_dpe0)
                            ph_dpe0 ^= 1
                            hd = do_cons.wait_and_advance()
                            _umma(op_dp, tDP[(None, None, None, 0)], tVr, tCrDO, hd.index, False, raw + DP_FULL)
                            hd.release()
                    # ---- block j + 1 (stage 1): dK_j, then S^T_{j+3} / dP^T_{j+3} into stage 1
                    if j + 1 < n_blocks:
                        cute.arch.mbarrier_wait(raw + (DS_FULL + 0), ph_dsf0)
                        ph_dsf0 ^= 1
                        _umma(op_dk, tDK, tCrDS[(None, None, None, 0)], tCrQmn, q_rel.current_handle().index, True, raw + DS_EMPTY)
                        q_rel.release()
                        q_rel.advance()
                        if j + 3 < n_blocks:
                            cute.arch.mbarrier_wait(raw + (S_EMPTY + 1), ph_se1)
                            ph_se1 ^= 1
                            hq = q_cons.wait_and_advance()
                            _umma(op_s, tS[(None, None, None, 1)], tCrA0, tCrQ, hq.index, False, raw + (S_FULL + 1))
                            if const_expr(not self.load_dp):
                                cute.arch.mbarrier_wait(raw + (DP_EMPTY + 1), ph_dpe1)
                                ph_dpe1 ^= 1
                                hd = do_cons.wait_and_advance()
                                _umma(op_dp, tDP[(None, None, None, 1)], tVr, tCrDO, hd.index, False, raw + (DP_FULL + 1))
                                hd.release()
                # dK_{n-1}
                if n_blocks % 2 == 1:
                    cute.arch.mbarrier_wait(raw + (DS_FULL + 0), ph_dsf0)
                    _umma(op_dk, tDK, tCrDS[(None, None, None, 0)], tCrQmn, q_rel.current_handle().index, True, raw + DS_EMPTY)
                else:
                    cute.arch.mbarrier_wait(raw + (DS_FULL + 1), ph_dsf1)
                    _umma(op_dk, tDK, tCrDS[(None, None, None, 1)], tCrQmn, q_rel.current_handle().index, True, raw + (DS_EMPTY + 1))
                q_rel.release()
                q_rel.advance()
            a_cons.release()
            with cute.arch.elect_one():
                tcgen05.commit(raw + DONE)
        else:
            tmem.allocate(self.tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tDP = cute.make_tensor(tmem_ptr + self.tmem_dp, s_fake.layout)
            tDK = cute.make_tensor(tmem_ptr + self.tmem_dk, dk_fake.layout)
            tVr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_v, dtype=BFloat16), tVr_fake.layout)
            grp = warp // 4
            trow = tidx % 128
            kpos = k0 + trow
            if grp == 1:
              if const_expr(not self.load_dp):
                st_v = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(V // 2), tcgen05.Unpack.NONE), BFloat16)
                r2t_v = tcgen05.make_tmem_copy(st_v, tVr)
                thr_v = r2t_v.get_slice(trow)
                tRT_tV = thr_v.partition_D(tVr)
                tRT_cv = thr_v.partition_S(thr_dp.partition_A(cute.make_identity_tensor((self.TM, V))))
                rV = cute.make_rmem_tensor((V,), BFloat16)
                v_ptr = pV + (Int64(b * L + kpos) * K + kh) * V
                for e8 in cutlass.range_constexpr(V // 8):
                    cute.make_tensor(rV.iterator + e8 * 8, cute.make_layout(8)).store(
                        cute.make_tensor((v_ptr + e8 * 8).align(16), cute.make_layout(8)).load())
                cute.copy(r2t_v, cute.make_tensor(rV.iterator, cute.make_layout(tRT_cv.shape)), tRT_tV)
            else:
                st_z = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(32), tcgen05.Unpack.NONE), Float32)
                zr = cute.make_rmem_tensor((32,), Float32)
                for e in cutlass.range_constexpr(32):
                    zr[e] = Float32(0.0)
                tDK_z = cute.flat_divide(tDK[((None, None), 0, 0)], (self.TM, 32))
                r2t_zk = tcgen05.make_tmem_copy(st_z, tDK_z[(None, None, 0, 0)])
                tRT_zk = r2t_zk.get_slice(trow).partition_D(tDK_z)
                tRT_zk = cute.group_modes(tRT_zk, 3, cute.rank(tRT_zk))
                zr_v = cute.make_tensor(zr.iterator, cute.make_layout(r2t_zk.get_slice(trow).partition_S(cute.make_identity_tensor((self.TM, 32))).shape))
                for piece in cutlass.range_constexpr(S // 32):
                    cute.copy(r2t_zk, zr_v, tRT_zk[(None, None, None, piece)])
            cute.arch.fence_view_async_tmem_store()
            kv_bar.arrive_and_wait()
            ld_s = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(QB), tcgen05.Pack.NONE), Float32)
            tS_r = [cute.flat_divide(tS[((None, None), 0, 0, i)], (self.TM, QB)) for i in range(2)]
            tDP_r = [cute.flat_divide(tDP[((None, None), 0, 0, i)], (self.TM, QB)) for i in range(2)]
            t2r_s = [tcgen05.make_tmem_copy(ld_s, tS_r[i][(None, None, 0, 0)]) for i in range(2)]
            t2r_dp = [tcgen05.make_tmem_copy(ld_s, tDP_r[i][(None, None, 0, 0)]) for i in range(2)]
            tTR_s = [t2r_s[i].get_slice(trow).partition_S(tS_r[i]) for i in range(2)]
            tTR_s = [cute.group_modes(tTR_s[i], 3, cute.rank(tTR_s[i])) for i in range(2)]
            tTR_dp = [t2r_dp[i].get_slice(trow).partition_S(tDP_r[i]) for i in range(2)]
            tTR_dp = [cute.group_modes(tTR_dp[i], 3, cute.rank(tTR_dp[i])) for i in range(2)]
            xs = cute.make_rmem_tensor(t2r_s[0].get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, QB))).shape, Float32)
            xd = cute.make_rmem_tensor(xs.shape, Float32)
            dsb = cute.make_rmem_tensor(xs.shape, BFloat16)
            ld_o = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32), tcgen05.Pack.NONE), Float32)
            tDK_r = cute.flat_divide(tDK[((None, None), 0, 0)], (self.TM, 32))
            t2r_dk = tcgen05.make_tmem_copy(ld_o, tDK_r[(None, None, 0, 0)])
            tTR_dk = t2r_dk.get_slice(trow).partition_S(tDK_r)
            tTR_dk = cute.group_modes(tTR_dk, 3, cute.rank(tTR_dk))
            ov = cute.make_rmem_tensor(t2r_dk.get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, 32))).shape, Float32)
            obf = cute.make_rmem_tensor(ov.shape, BFloat16)
            g = grp
            n_own = (n_blocks - g + 1) // 2
            if const_expr(self.dbg == 3):
                n_own = Int32(0)
            ph_sf = Int32(0)
            ph_dpf = Int32(0)
            ph_dse = Int32(1)
            lane = tidx % 32
            slot = warp * QB
            r = g // n_qb
            qb = qb_lo + g - r * n_qb
            NL = QB // 32  # column values per lane
            nxt_lse = cute.make_rmem_tensor(cute.make_layout(NL), Float32)
            nxt_del = cute.make_rmem_tensor(cute.make_layout(NL), Float32)
            nxt_km = cute.make_rmem_tensor(cute.make_layout(NL), Int32)
            nxt_kml = Int32(0)
            dp_off = Int64((b * nc + ck) * H) * C * C + Int64(kpos - ck * C) * C  # + h C C: this key row of dPcur [B, nc, H, C, C]
            dp_off_p = Int64((b * (nc - 1) + ck) * H) * C * C + Int64(kpos - ck * C) * C  # of dPprev [B, nc - 1, H, C, C]
            if n_own > 0:
                h = kh * self.n_rep + r
                for u in cutlass.range_constexpr(NL):
                    nxt_lse[u] = cute.make_tensor(pLse + ((b * H + h) * L + qb * QB + u * 32 + lane), cute.make_layout(1))[0]
                    nxt_del[u] = cute.make_tensor(pDelta + ((b * H + h) * L + qb * QB + u * 32 + lane), cute.make_layout(1))[0]
                    nxt_km[u] = cute.make_tensor(pKmin + (b * L + qb * QB + u * 32 + lane), cute.make_layout(1))[0]
                nxt_kml = cute.make_tensor(pKmin + (b * L + qb * QB + QB - 1), cute.make_layout(1))[0]
            for t in cutlass.range(n_own):
                q0 = qb * QB
                h_cur = kh * self.n_rep + r
                kmin_last = nxt_kml
                for u in cutlass.range_constexpr(NL):
                    sLse[slot + u * 32 + lane] = nxt_lse[u]
                    sDel[slot + u * 32 + lane] = nxt_del[u]
                    sKm[slot + u * 32 + lane] = nxt_km[u]
                cute.arch.sync_warp()
                qb_n = qb + 2
                r_n = r
                if qb_n > qb_hi:
                    qb_n = qb_n - n_qb
                    r_n = r_n + 1
                if qb_n > qb_hi:
                    qb_n = qb_n - n_qb
                    r_n = r_n + 1
                if t + 1 < n_own:
                    h_n = kh * self.n_rep + r_n
                    for u in cutlass.range_constexpr(NL):
                        nxt_lse[u] = cute.make_tensor(pLse + ((b * H + h_n) * L + qb_n * QB + u * 32 + lane), cute.make_layout(1))[0]
                        nxt_del[u] = cute.make_tensor(pDelta + ((b * H + h_n) * L + qb_n * QB + u * 32 + lane), cute.make_layout(1))[0]
                        nxt_km[u] = cute.make_tensor(pKmin + (b * L + qb_n * QB + u * 32 + lane), cute.make_layout(1))[0]
                    nxt_kml = cute.make_tensor(pKmin + (b * L + qb_n * QB + QB - 1), cute.make_layout(1))[0]
                masked = cute.arch.make_warp_uniform(Int32((q0 < k0 + self.TM - 1) | (kmin_last > k0)))
                if masked != 0:
                    ph_sf, ph_dpf, ph_dse = _dk_block(
                        True, self.load_dp, g, slot, trow, kpos, q0, scale2, raw, sLse, sDel, sKm, pDSs, ds_stage_elems,
                        t2r_s[0], t2r_s[1], tTR_s[0], tTR_s[1], t2r_dp[0], t2r_dp[1], tTR_dp[0], tTR_dp[1],
                        xs, xd, dsb, ph_sf, ph_dpf, ph_dse, QB, self.dbg, pDPcur, pDPprev, dp_off, dp_off_p, ck, h_cur, C)
                else:
                    ph_sf, ph_dpf, ph_dse = _dk_block(
                        False, self.load_dp, g, slot, trow, kpos, q0, scale2, raw, sLse, sDel, sKm, pDSs, ds_stage_elems,
                        t2r_s[0], t2r_s[1], tTR_s[0], tTR_s[1], t2r_dp[0], t2r_dp[1], tTR_dp[0], tTR_dp[1],
                        xs, xd, dsb, ph_sf, ph_dpf, ph_dse, QB, self.dbg, pDPcur, pDPprev, dp_off, dp_off_p, ck, h_cur, C)
                qb = qb_n
                r = r_n
            cute.arch.mbarrier_wait(raw + DONE, Int32(0))
            if grp == 1:
                row_ptr = pDK + (Int64(b * L + kpos) * K + kh) * S
                for piece in cutlass.range_constexpr(S // 32):
                    cute.copy(t2r_dk, tTR_dk[(None, None, None, piece)], ov)
                    cute.arch.fence_view_async_tmem_load()
                    obf.store((ov.load() * scale).to(BFloat16))
                    for e8 in cutlass.range_constexpr(4):
                        cute.make_tensor((row_ptr + (piece * 32 + e8 * 8)).align(16), cute.make_layout(8)).store(
                            cute.make_tensor(obf.iterator + e8 * 8, cute.make_layout(8)).load())
            tmem.relinquish_alloc_permit()
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_cw)
            tmem.free(tmem_ptr)


def sca_bwd(q, k, v, y, lse2, dout, chunk_size, scale, kmin, qmax, delta=None, fp8_dp=False):
    """Gradients of sca_fwd: dq [B, L, H, S], dk [B, L, K, S], dv [B, L, K, V] (bf16). delta [B, H, L] fp32 =
    rowsum(dout * y) may be passed in (the fused attention gate backward produces it), else it is computed here.
    fp8_dp: the V > 256 paths compute dP = dO V^T with MXFP8 GEMMs (dO and V quantised to e4m3 with e8m0 scales per
    32 elements along V, fp32 accumulation and output; see _mx_dp) instead of bf16 cuBLAS GEMMs."""
    B, L, H, S = q.shape
    K, V = v.shape[2], v.shape[3]
    C = chunk_size
    nc = L // C
    n_rep = H // K
    dev = q.device
    dout = dout.contiguous()
    fp8_dp = bool(fp8_dp) and V > 256 and _mx_dp_supported(dev, C, V)
    # delta[b, h, q] = sum_v dO O
    if delta is None:
        delta = torch.empty(B, H, L, dtype=torch.float32, device=dev)
        f_delta = _compiled(("sca_rowdot", H, V), lambda: RowDot(H, V), lambda: (
            _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), Int32(0), Int32(0)))
        f_delta(dout.reshape(-1), y.reshape(-1), delta.reshape(-1), B, L)
    else:
        assert delta.shape == (B, H, L) and delta.dtype == torch.float32
        delta = delta.contiguous()
    dv = torch.empty(B, L, K, V, dtype=torch.bfloat16, device=dev)
    if 3 * (S + V) // 2 + 4 * 32 <= 512 and os.environ.get("SCA_FUSED_DKDV", "1") == "1":
        # dK and dV in one pass (S^T / P^T computed once)
        dk = torch.empty(B, L, K, S, dtype=torch.bfloat16, device=dev)
        f_dkdv = _compiled(("sca_bwd_dkdv", H, K, S, V, C), lambda: ScaBwdDkDv(H, K, S, V, C), lambda: (
            _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Float32, 4),
            _fake(Int32, 4), _fake(Int32, 4), _fake(BFloat16), _fake(BFloat16), Int32(0), Int32(0), Float32(0.0), Float32(0.0)))
        f_dkdv(q.reshape(-1), k.reshape(-1), v.reshape(-1), dout.reshape(-1), lse2.reshape(-1), delta.reshape(-1),
               kmin.reshape(-1), qmax.reshape(-1), dk.reshape(-1), dv.reshape(-1), B, L, float(scale * LOG2E), float(scale))
        dp_cur = dp_prev = torch.zeros(1, dtype=torch.float32, device=dev)
        return _sca_bwd_dq(q, k, v, dout, lse2, delta, kmin, dp_prev, dp_cur, B, L, H, K, S, V, C, scale, dev), dk, dv
    if V > 256:
        # both backward paths are compiled at the first call: the choice below is per batch, and a rank compiling the
        # other path mid-run (seconds) would stall every rank of a data-parallel step
        _compile_bwd_paths(H, K, S, V, C)
    if V > 256 and _use_pmat_dv(qmax, B, L, C):
        # dV, dK and dQ from the materialized bf16 P and dS (chunk-major [C queries x 2C keys]) with cuBLAS: the v-slab dV
        # kernel recomputes S^T V / 256 times (as many flops as its own MMAs) and the dK / dQ kernels recompute it again.
        # The cost is that of dense [C x C] GEMMs, so this path only wins when most of the window is attended (few
        # segment boundaries): 7.1B, B200, per layer without boundaries dV 2.9 -> 1.5 ms, dK + dQ 1.8 -> 1.0 ms;
        # with 7 segments per 16K tokens the kernels take 1.1 / 0.9 ms.
        pm = torch.empty(nc, B, H, C, 2 * C, dtype=torch.bfloat16, device=dev)  # the kernel also writes the zeros
        f_pm = _compiled(("sca_pmat", H, K, S, C), lambda: ScaPmat(H, K, S, C), lambda: (
            _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Int32, 4), _fake2(BFloat16), Int32(0), Int32(0), Float32(0.0)))
        f_pm(q.reshape(-1), k.reshape(-1), lse2.reshape(-1), kmin.reshape(-1), _col2d(pm), B, L, float(scale * LOG2E))
        doc = _chunk_major(dout, H, H, V, C, B, L, nc, dev)
        vc = _chunk_major(v, H, K, V, C, B, L, nc, dev)  # heads repeated
        qc = _chunk_major(q, H, H, S, C, B, L, nc, dev)
        kc = _chunk_major(k, H, K, S, C, B, L, nc, dev)
        f_un = _compiled(("sca_cmuncopy", K, V, C), lambda: ChunkMajorUncopy(K, V, C), lambda: (
            _fake(BFloat16), _fake(BFloat16), Int32(0), Int32(0)))
        f_unq = _compiled(("sca_cmuncopy", H, S, C), lambda: ChunkMajorUncopy(H, S, C), lambda: (
            _fake(BFloat16), _fake(BFloat16), Int32(0), Int32(0)))
        f_unk = _compiled(("sca_cmuncopy", K, S, C), lambda: ChunkMajorUncopy(K, S, C), lambda: (
            _fake(BFloat16), _fake(BFloat16), Int32(0), Int32(0)))
        cur = (slice(None),) * 4 + (slice(C, None),)   # columns [C, 2C): the keys of the row's own chunk
        prv = (slice(None),) * 4 + (slice(None, C),)   # columns [0, C): the keys of the previous chunk
        # dV_c = P_c[:, C:]^T dO_c + P_{c+1}[:, :C]^T dO_{c+1}; the causal [C x C] block of P_c is lower triangular:
        # its sub-blocks above the diagonal are skipped (n_sub x n_sub sub-blocks: 10 / 16 of the flops at 4)
        n_sub = _tril_sub(C)
        outc = torch.empty(nc, B, H, C, V, dtype=torch.bfloat16, device=dev)
        _bmm_tril_tn(pm, doc, outc, C, n_sub)
        outc = outc.view(-1, C, V)
        if nc > 1:
            outc.view(nc, B, H, C, V)[:-1].view(-1, C, V).baddbmm_(pm[1:][prv].reshape(-1, C, C).transpose(1, 2), doc[1:].view(-1, C, V))
        outc = outc.view(nc, B, K, n_rep, C, V)
        dvc = outc[:, :, :, 0] if n_rep == 1 else outc.float().sum(3).to(torch.bfloat16)
        f_un(dvc.contiguous().reshape(-1), dv.reshape(-1), B, L)
        del outc, dvc
        # dS = P (dP - delta) scale with dP = dO V^T (fp32 GEMMs, [queries x keys])
        # (dPq_cur only below the diagonal sub-blocks: ScaDsMat never reads dP past the row's 8-aligned diagonal)
        dpq_cur = torch.empty(nc, B, H, C, C, dtype=torch.float32, device=dev)
        if fp8_dp:
            dpq_prev = torch.empty(nc - 1, B, H, C, C, dtype=torch.float32, device=dev) if nc > 1 else dpq_cur
            _mx_dp(vc.view(-1, V), doc.view(-1, V), dpq_cur, dpq_prev, nc, B, H, C, chunk_major=True)
        else:
            _bmm_tril_nt(doc, vc, dpq_cur, C, n_sub)
            if nc > 1:
                dpq_prev = torch.bmm(doc[1:].reshape(-1, C, V), vc[:-1].reshape(-1, C, V).transpose(1, 2), out_dtype=torch.float32)
            else:
                dpq_prev = dpq_cur
        ds = torch.empty_like(pm)
        f_ds = _compiled(("sca_dsmat", H, C), lambda: ScaDsMat(H, C), lambda: (
            _fake2(BFloat16), _fake2(Float32), _fake2(Float32), _fake(Float32, 4), _fake2(BFloat16), Int32(0), Int32(0), Float32(0.0)))
        f_ds(_col2d(pm), _col2d(dpq_cur), _col2d(dpq_prev), delta.reshape(-1), _col2d(ds), B, L, float(scale))
        del pm, dpq_cur, dpq_prev
        # dQ_c = dS_c[:, C:] K_c + dS_c[:, :C] K_{c-1}
        dqc = torch.empty(nc, B, H, C, S, dtype=torch.bfloat16, device=dev)
        _bmm_tril_nn(ds, kc, dqc, C, n_sub)
        dqc = dqc.view(-1, C, S)
        if nc > 1:
            dqc.view(nc, B, H, C, S)[1:].view(-1, C, S).baddbmm_(ds[1:][prv].reshape(-1, C, C), kc[:-1].view(-1, C, S))
        dq = torch.empty(B, L, H, S, dtype=torch.bfloat16, device=dev)
        f_unq(dqc.reshape(-1), dq.reshape(-1), B, L)
        # dK_c = dS_c[:, C:]^T Q_c + dS_{c+1}[:, :C]^T Q_{c+1}
        dkc = torch.empty(nc, B, H, C, S, dtype=torch.bfloat16, device=dev)
        _bmm_tril_tn(ds, qc, dkc, C, n_sub)
        dkc = dkc.view(-1, C, S)
        if nc > 1:
            dkc.view(nc, B, H, C, S)[:-1].view(-1, C, S).baddbmm_(ds[1:][prv].reshape(-1, C, C).transpose(1, 2), qc[1:].view(-1, C, S))
        dkc = dkc.view(nc, B, K, n_rep, C, S)
        dkc = dkc[:, :, :, 0] if n_rep == 1 else dkc.float().sum(3).to(torch.bfloat16)
        dk = torch.empty(B, L, K, S, dtype=torch.bfloat16, device=dev)
        f_unk(dkc.contiguous().reshape(-1), dk.reshape(-1), B, L)
        return dq, dk, dv
    if S in (64, 128, 256) and S // 2 + 64 + min(V, 256) <= 512 and os.environ.get("SCA_DV_TMEM", "1") == "1":  # K tile -> TMEM store
        f_dv = _compiled(("sca_bwd_dvt", H, K, S, V, C), lambda: ScaBwdDvT(H, K, S, V, C), lambda: (
            _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Int32, 4), _fake(Int32, 4),
            _fake(BFloat16), Int32(0), Int32(0), Float32(0.0)))
    else:
        f_dv = _compiled(("sca_bwd_dv", H, K, S, V, C), lambda: ScaBwdDv(H, K, S, V, C), lambda: (
            _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Int32, 4), _fake(Int32, 4),
            _fake(BFloat16), Int32(0), Int32(0), Float32(0.0)))
    f_dv(q.reshape(-1), k.reshape(-1), dout.reshape(-1), lse2.reshape(-1), kmin.reshape(-1), qmax.reshape(-1), dv.reshape(-1),
         B, L, float(scale * LOG2E))
    if V > 256:
        # dP for the dK / dQ kernels (fp32 GEMMs): dPcur[b, c, h] = V_c dO_c^T, dPprev[b, c - 1, h] = V_{c-1} dO_c^T,
        # per head on strided views of v and dout (the chunk and batch dims of a head collapse into one batch
        # stride, so no chunk-major copies). dPcur is only read where the query attends the key (query >= key):
        # the sub-blocks below the diagonal are zeroed instead of computed (the kernels multiply them by P^T = 0)
        # (only the sub-blocks below the diagonal are zeroed, by one kernel: the strided per-block fills cost
        # 3 x 25 us per head, a full torch.zeros 175 us per layer on 7.1B)
        dp_cur = torch.empty(B, nc, H, C, C, dtype=torch.float32, device=dev)
        dp_prev = torch.empty(B, nc - 1, H, C, C, dtype=torch.float32, device=dev) if nc > 1 else dp_cur
        n_sub = _tril_sub(C)
        if fp8_dp:
            # full [C x C] MXFP8 GEMMs (the sub-blocks below the diagonal are computed rather than zeroed: one 2-D
            # _scaled_mm per block, and sub-block GEMMs of these shapes run far below the full ones' rate)
            _mx_dp(v.view(B * L, K, V), dout.view(B * L, H, V), dp_cur, dp_prev, nc, B, H, C, chunk_major=False)
        else:
            if n_sub > 1:
                f_zt = _compiled(("sca_zerotril", C, n_sub), lambda: ScaZeroTril(C, n_sub), lambda: (_fake2(Float32), Int32(0)))
                f_zt(_col2d(dp_cur), B * nc * H)
            for h in range(H):
                vv = v[:, :, h // n_rep, :].view(B, nc, C, V)
                dd = dout[:, :, h, :].view(B, nc, C, V)
                _bmm_triu_nt(vv, dd, dp_cur[:, :, h], C, n_sub, zero_below=False)
                if nc > 1:
                    for bi in range(B):
                        torch.bmm(vv[bi, :-1], dd[bi, 1:].transpose(1, 2), out_dtype=torch.float32, out=dp_prev[bi, :, h])
    else:
        dp_cur = dp_prev = torch.zeros(1, dtype=torch.float32, device=dev)
    dk = torch.empty(B, L, K, S, dtype=torch.bfloat16, device=dev)
    if _dkt_fits(S, V) and os.environ.get("SCA_DK_TMEM", "1") == "1":
        f_dkt = _dkt_compiled(H, K, S, V, C)
        f_dkt(q.reshape(-1), k.reshape(-1), v.reshape(-1), dout.reshape(-1), lse2.reshape(-1), delta.reshape(-1), kmin.reshape(-1),
              qmax.reshape(-1), dk.reshape(-1), _col2d(dp_prev), _col2d(dp_cur), B, L, float(scale * LOG2E), float(scale))
        dq = _sca_bwd_dq(q, k, v, dout, lse2, delta, kmin, dp_prev, dp_cur, B, L, H, K, S, V, C, scale, dev)
        return dq, dk, dv
    f_dk = _compiled(("sca_bwd_dk", H, K, S, V, C), lambda: ScaBwdDk(H, K, S, V, C), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Float32, 4),
        _fake(Int32, 4), _fake(Int32, 4), _fake(BFloat16), _fake2(Float32), _fake2(Float32), Int32(0), Int32(0),
        Float32(0.0), Float32(0.0)))
    f_dk(q.reshape(-1), k.reshape(-1), v.reshape(-1), dout.reshape(-1), lse2.reshape(-1), delta.reshape(-1), kmin.reshape(-1),
         qmax.reshape(-1), dk.reshape(-1), _col2d(dp_prev), _col2d(dp_cur), B, L, float(scale * LOG2E), float(scale))
    dq = _sca_bwd_dq(q, k, v, dout, lse2, delta, kmin, dp_prev, dp_cur, B, L, H, K, S, V, C, scale, dev)
    return dq, dk, dv


def _mxfp8():
    """The MXFP8 module (through the package, or from its file when this module is loaded standalone)."""
    try:
        from xllm.modules.fused_ops import mxfp8
        return mxfp8
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("mxfp8_sca", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mxfp8.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def _mx_dp_supported(dev, C, V):
    """Whether the fp8 dP GEMMs handle this shape: Blackwell, C and V multiples of 128 (env SCA_DP_FP8=0 disables)."""
    if os.environ.get("SCA_DP_FP8", "1") == "0":
        return False
    try:
        return _mxfp8().mx_supported(dev, C, V)
    except ImportError:
        return False


def _mx_dp(vm, dm, dp_cur, dp_prev, nc, B, H, C, chunk_major):
    """dP = dO V^T per (batch, chunk, head) with MXFP8 GEMMs (fp32 out): dO and V quantised rowwise (e4m3, one e8m0
    scale per 32 elements along V) by one kernel each, one 2-D cuBLASLt GEMM per [C x C] block.
    chunk_major=True (the materialized-P path): vm / dm are the chunk-major copies [(nc, B, H, C), V] (V's heads
    repeated) and dp_cur [nc, B, H, C, C] = dO_c V_c^T [queries x keys], dp_prev [nc - 1, B, H, C, C] = dO_c V_{c-1}^T.
    chunk_major=False (the kernel path): vm [B L, K, V] and dm [B L, H, V] are the token-major tensors and
    dp_cur [B, nc, H, C, C] = V_c dO_c^T [keys x queries], dp_prev [B, nc - 1, H, C, C] = V_{c-1} dO_c^T.
    7.1B (C 4096, V 2048, 4 heads, 16K tokens), B200: per layer 1113 (bf16 cuBLAS, 10 / 16 sub-blocks) -> 972 us
    (133 us of quantisation + 28 full GEMMs of 29.5 us; sub-block GEMMs of these shapes are slower than full ones)."""
    mx = _mxfp8()
    if chunk_major:
        vr, _ = mx.mx_quantize(vm, colwise=False)
        dr, _ = mx.mx_quantize(dm, colwise=False)
        vrow = drow = lambda c, b, h: ((c * B + b) * H + h) * C  # the chunk-major row of (c, b, h); V's heads repeated
    else:
        K = vm.shape[1]
        L = vm.shape[0] // B
        vr, _ = mx.mx_quantize_cat([vm[:, kk, :] for kk in range(K)], 0, colwise=False)  # [V_k0; V_k1; ...] (K B L, V)
        dr, _ = mx.mx_quantize_cat([dm[:, h, :] for h in range(H)], 0, colwise=False)  # [dO_h0; dO_h1; ...] (H B L, V)
        drow = lambda c, b, h: (h * B + b) * L + c * C  # the row of (c, b, head h) in the head-stacked operand
        vrow = lambda c, b, h: (h // (H // K) * B + b) * L + c * C
    for b in range(B):
        for h in range(H):
            for c in range(nc):
                d = dr.row_slice(drow(c, b, h), drow(c, b, h) + C)
                blocks = [(c, dp_cur[c, b, h] if chunk_major else dp_cur[b, c, h])]
                if c > 0:
                    blocks.append((c - 1, dp_prev[c - 1, b, h] if chunk_major else dp_prev[b, c - 1, h]))
                for cv, out in blocks:
                    vv = vr.row_slice(vrow(cv, b, h), vrow(cv, b, h) + C)
                    if chunk_major:  # [queries x keys]
                        torch._scaled_mm(d.data, vv.data.t(), d.scale8, vv.scale8, out_dtype=torch.float32, out=out)
                    else:  # [keys x queries]
                        torch._scaled_mm(vv.data, d.data.t(), vv.scale8, d.scale8, out_dtype=torch.float32, out=out)


def _tril_sub(C):
    """Sub-blocks per side for the causal [C x C] GEMMs (env SCA_TRIL_N; 4 skips 6 of 16 sub-blocks, 8 skips 28 of 64
    but the smaller GEMMs run less efficiently: 7.1B C=4096 per layer cur-chunk dV 616 -> 481 / 544 us, dP 736 -> 501 / 507)."""
    n = int(os.environ.get("SCA_TRIL_N", "4"))
    while n > 1 and C % (n * 64):
        n //= 2
    return n


def _bmm_tril_tn(pm, x, out, C, n):
    """out[(c, b, h)] = P[:, C:]^T x per (c, b, h) for the chunk-major P / dS [nc, B, H, C, 2C] (bf16, lower triangular
    in its [C, 2C) columns), x [nc, B, H, C, X] -> out [nc, B, H, C, X]: row block i of P only touches out[:(i + 1) c]."""
    c = C // n
    for i in range(n - 1, -1, -1):
        a = pm[:, :, :, i * c:(i + 1) * c, C:C + (i + 1) * c].reshape(-1, c, (i + 1) * c).transpose(1, 2)
        b = x[:, :, :, i * c:(i + 1) * c].reshape(-1, c, x.shape[-1])
        o = out[:, :, :, :(i + 1) * c].reshape(-1, (i + 1) * c, out.shape[-1])
        if i == n - 1:
            torch.bmm(a, b, out=o)
        else:
            o.baddbmm_(a, b)


def _bmm_tril_nn(ds, x, out, C, n):
    """out[(c, b, h)] = dS[:, C:] x: row block i of dS only reads x[:(i + 1) c]."""
    c = C // n
    for i in range(n):
        a = ds[:, :, :, i * c:(i + 1) * c, C:C + (i + 1) * c].reshape(-1, c, (i + 1) * c)
        b = x[:, :, :, :(i + 1) * c].reshape(-1, (i + 1) * c, x.shape[-1])
        torch.bmm(a, b, out=out[:, :, :, i * c:(i + 1) * c].reshape(-1, c, out.shape[-1]))


def _bmm_tril_nt(x, y, out, C, n):
    """out[(c, b, h)][i-th row block, :(i + 1) c] = x[i-th row block] y[:(i + 1) c]^T (fp32 out [nc, B, H, C, C]):
    the sub-blocks above the diagonal are left unwritten."""
    c = C // n
    for i in range(n):
        a = x[:, :, :, i * c:(i + 1) * c].reshape(-1, c, x.shape[-1])
        b = y[:, :, :, :(i + 1) * c].reshape(-1, (i + 1) * c, y.shape[-1]).transpose(1, 2)
        torch.bmm(a, b, out_dtype=torch.float32, out=out[:, :, :, i * c:(i + 1) * c, :(i + 1) * c].reshape(-1, c, (i + 1) * c))


def _bmm_triu_nt(x, y, out, C, n, zero_below=True):
    """out[..., i-th row block, i c:] = x[..., i-th row block, :] y[..., i c:, :]^T (fp32 out [..., C, C]), the sub-blocks
    below the diagonal set to zero (unless the caller zeroed out already); the batch dims of x, y and out must
    collapse into one stride."""
    c = C // n
    for i in range(n):
        a = x[..., i * c:(i + 1) * c, :].reshape(-1, c, x.shape[-1])
        b = y[..., i * c:, :].reshape(-1, C - i * c, y.shape[-1]).transpose(1, 2)
        torch.bmm(a, b, out_dtype=torch.float32, out=out[..., i * c:(i + 1) * c, i * c:].reshape(-1, c, C - i * c))
        if i > 0 and zero_below:
            out[..., i * c:(i + 1) * c, :i * c].zero_()


def _dkt_fits(S, V):
    """ScaBwdDkT: V <= 256 with the V tile in TMEM (V / 2 + 128 + S <= 512), or V > 256 reading dP from the GEMMs
    (128 + S <= 512)."""
    if V > 256:
        return 128 + S <= 512
    return V in (64, 128, 256) and V // 2 + 128 + S <= 512


def _dkt_compiled(H, K, S, V, C):
    return _compiled(("sca_bwd_dkt", H, K, S, V, C), lambda: ScaBwdDkT(H, K, S, V, C), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Float32, 4),
        _fake(Int32, 4), _fake(Int32, 4), _fake(BFloat16), _fake2(Float32), _fake2(Float32), Int32(0), Int32(0), Float32(0.0), Float32(0.0)))


_BWD_PATHS_COMPILED = set()


def _compile_bwd_paths(H, K, S, V, C):
    """Compile the kernels of both V > 256 backward paths (GEMM path: ScaPmat, ScaDsMat, chunk-major copies; kernel
    path: ScaBwdDv / DvT, ScaBwdDk, dQ) once per shape."""
    key = (H, K, S, V, C)
    if key in _BWD_PATHS_COMPILED:
        return
    _BWD_PATHS_COMPILED.add(key)
    _compiled(("sca_pmat", H, K, S, C), lambda: ScaPmat(H, K, S, C), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Int32, 4), _fake2(BFloat16), Int32(0), Int32(0), Float32(0.0)))
    _compiled(("sca_dsmat", H, C), lambda: ScaDsMat(H, C), lambda: (
        _fake2(BFloat16), _fake2(Float32), _fake2(Float32), _fake(Float32, 4), _fake2(BFloat16), Int32(0), Int32(0), Float32(0.0)))
    for (h, k, d) in ((H, K, V), (H, H, V), (H, H, S), (H, K, S)):
        _compiled(("sca_cmcopy", h, k, d, C), lambda: ChunkMajorCopy(h, k, d, C), lambda: (
            _fake(BFloat16), _fake(BFloat16), Int32(0), Int32(0)))
    for (k, d) in ((K, V), (H, S), (K, S)):
        _compiled(("sca_cmuncopy", k, d, C), lambda: ChunkMajorUncopy(k, d, C), lambda: (
            _fake(BFloat16), _fake(BFloat16), Int32(0), Int32(0)))
    if S in (64, 128, 256) and S // 2 + 64 + min(V, 256) <= 512 and os.environ.get("SCA_DV_TMEM", "1") == "1":
        _compiled(("sca_bwd_dvt", H, K, S, V, C), lambda: ScaBwdDvT(H, K, S, V, C), lambda: (
            _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Int32, 4), _fake(Int32, 4),
            _fake(BFloat16), Int32(0), Int32(0), Float32(0.0)))
    else:
        _compiled(("sca_bwd_dv", H, K, S, V, C), lambda: ScaBwdDv(H, K, S, V, C), lambda: (
            _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Int32, 4), _fake(Int32, 4),
            _fake(BFloat16), Int32(0), Int32(0), Float32(0.0)))
    if _dkt_fits(S, V) and os.environ.get("SCA_DK_TMEM", "1") == "1":
        _dkt_compiled(H, K, S, V, C)
    else:
        _compiled(("sca_bwd_dk", H, K, S, V, C), lambda: ScaBwdDk(H, K, S, V, C), lambda: (
            _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Float32, 4),
            _fake(Int32, 4), _fake(Int32, 4), _fake(BFloat16), _fake2(Float32), _fake2(Float32), Int32(0), Int32(0),
            Float32(0.0), Float32(0.0)))
    if _dqt_mode(S, V) is not None and os.environ.get("SCA_DQ_TMEM", "1") == "1":
        _dqt_compiled(H, K, S, V, C)
    else:
        _compiled(("sca_bwd_dq", H, K, S, V, C), lambda: ScaBwdDq(H, K, S, V, C), lambda: (
            _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Float32, 4),
            _fake(Int32, 4), _fake(BFloat16), _fake2(Float32), _fake2(Float32), Int32(0), Int32(0), Float32(0.0), Float32(0.0)))


def _use_pmat_dv(qmax, B, L, C):
    """Whether the V > 256 backward should materialize P / dS and use cuBLAS (fixed cost) rather than the kernels
    (cost ~ attended fraction of the [C x 2C] windows): GEMMs when the fraction exceeds a threshold, 0.75 on one
    GPU (7.1B per layer: kernels 2.7 ms at 0.47 / 4.5 at 1.0, GEMMs 3.6-3.8) and 0.5 under data parallelism, where
    the slowest rank's batch sets the step time and the fixed-cost path limits the straggling (8 x B200 7.1B: both
    ~875 ms/step). Env SCA_DV_PMAT=0/1 forces the kernel / GEMM path, SCA_PMAT_FRAC overrides the threshold."""
    env = os.environ.get("SCA_DV_PMAT", "auto")
    if env in ("0", "1"):
        return env == "1"
    frac = os.environ.get("SCA_PMAT_FRAC")
    if frac is None:
        dp = torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        frac = 0.5 if dp else 0.75

    def make():
        # one host sync per distinct qmax (one per step: every layer's backward sees the same segment tensors),
        # not one per layer: each .item() drains the launch queue and leaves the GPU idle until the host catches up
        k0 = torch.arange(0, L, 128, device=qmax.device)
        cap = torch.minimum((k0 // C + 2) * C - 1, torch.full_like(k0, L - 1))
        attended = (torch.minimum(qmax[:, k0 + 127].long(), cap) - k0 + 1).clamp_(min=0).sum()
        full = sum(min((k // C + 2) * C - 1, L - 1) - k + 1 for k in range(0, L, 128)) * B
        return bool(attended.item() > float(frac) * full)

    return _cached(("pmat", B, L, C, frac), qmax, make)


def _chunk_major(x, H, K, V, C, B, L, nc, dev):
    """[nc, B, H, C, V] copy of x [B, L, K, V] with the kv heads repeated to H (ChunkMajorCopy)."""
    f_cp = _compiled(("sca_cmcopy", H, K, V, C), lambda: ChunkMajorCopy(H, K, V, C), lambda: (
        _fake(BFloat16), _fake(BFloat16), Int32(0), Int32(0)))
    out = torch.empty(nc, B, H, C, V, dtype=torch.bfloat16, device=dev)
    f_cp(x.reshape(-1), out.reshape(-1), B, L)
    return out


def _dqt_mode(S, V):
    """(KB, q_smem) of the ScaBwdDqT variant for the shape, or None: V > 256 reads dP from the GEMMs (Q in TMEM,
    64-key blocks: S / 2 + 128 + S <= 512); else Q and dO in TMEM with 64-key blocks ((S + V) / 2 + 256 + S <= 512),
    else Q in shared memory with 32-key blocks (V / 2 + 128 + S <= 512, V a power of two)."""
    if V > 256:
        return (64, False) if S // 2 + 128 + S <= 512 else None
    if (S + V) // 2 + 256 + S <= 512:
        return (64, False)
    if V in (64, 128, 256) and V // 2 + 128 + S <= 512:
        return (32, True)
    return None


def _dqt_compiled(H, K, S, V, C):
    KB, q_smem = _dqt_mode(S, V)
    return _compiled(("sca_bwd_dqt", H, K, S, V, C, KB, q_smem), lambda: ScaBwdDqT(H, K, S, V, C, KB=KB, q_smem=q_smem), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Float32, 4),
        _fake(Int32, 4), _fake(BFloat16), _fake2(Float32), _fake2(Float32), Int32(0), Int32(0), Float32(0.0), Float32(0.0)))


def _sca_bwd_dq(q, k, v, dout, lse2, delta, kmin, dp_prev, dp_cur, B, L, H, K, S, V, C, scale, dev):
    dq = torch.empty(B, L, H, S, dtype=torch.bfloat16, device=dev)
    if _dqt_mode(S, V) is not None and os.environ.get("SCA_DQ_TMEM", "1") == "1":
        f_dq = _dqt_compiled(H, K, S, V, C)
        f_dq(q.reshape(-1), k.reshape(-1), v.reshape(-1), dout.reshape(-1), lse2.reshape(-1), delta.reshape(-1),
             kmin.reshape(-1), dq.reshape(-1), _col2d(dp_prev), _col2d(dp_cur), B, L, float(scale * LOG2E), float(scale))
        return dq
    f_dq = _compiled(("sca_bwd_dq", H, K, S, V, C), lambda: ScaBwdDq(H, K, S, V, C), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Float32, 4),
        _fake(Int32, 4), _fake(BFloat16), _fake2(Float32), _fake2(Float32), Int32(0), Int32(0), Float32(0.0), Float32(0.0)))
    f_dq(q.reshape(-1), k.reshape(-1), v.reshape(-1), dout.reshape(-1), lse2.reshape(-1), delta.reshape(-1), kmin.reshape(-1),
         dq.reshape(-1), _col2d(dp_prev), _col2d(dp_cur), B, L, float(scale * LOG2E), float(scale))
    return dq


class ScaBwdDq:
    """dq[queries of a 128-row tile] = scale sum_k dS K for one (batch, head, chunk, query tile) per CTA (shapes
    that do not fit ScaBwdDqT: (S + V) / 2 + S > 256).

    Mirror of ScaBwdDk with queries as rows: the q tile is resident in shared memory (A of S = Q K^T), the dO tile
    in TMEM (A of dP = dO V^T), 64-key blocks of K (K-major B of S, MN-major B of dQ += dS K over the same bytes)
    and V (K-major B of dP) stream through 2-stage rings; the compute warps (thread = query row) form
    P = exp2(S scale2 - lse2) and dS = P (dP - delta) scale into the shared-memory A operand of the dQ MMA.
    TMEM: S, dP, dQ, dO. For V > 256 dP is read from the host-side fp32 GEMMs (see ScaBwdDk)."""

    def __init__(self, H: int, K: int, S: int, V: int, C: int, KB: int = 64, stages: int = 2):
        self.H, self.K, self.S, self.V, self.C, self.KB = H, K, S, V, C, KB
        self.n_rep = H // K
        self.TM = 128
        self.load_dp = V > 256
        self.Vd = min(V, 256)
        self.stages = stages
        self.num_cw = 4
        self.mma_warp = 4
        self.tma_warp = 5
        self.threads = 32 * 6
        self.row_tiles = C // self.TM
        assert H % K == 0 and S % 64 == 0 and S <= 256 and V % 64 == 0 and C % 128 == 0 and C % KB == 0
        self.tmem_s = 0
        self.tmem_dp = KB
        self.tmem_dq = 2 * KB
        self.tmem_do = 2 * KB + S
        total = self.tmem_do + (0 if self.load_dp else V // 2)
        self.tmem_cols = max(32, 1 << (total - 1).bit_length())
        assert self.tmem_cols <= 512

    @cute.jit
    def __call__(self, tQ: cute.Tensor, tK: cute.Tensor, tV: cute.Tensor, tDO: cute.Tensor, tLse: cute.Tensor,
                 tDelta: cute.Tensor, tKmin: cute.Tensor, tDQ: cute.Tensor, tDPprev: cute.Tensor, tDPcur: cute.Tensor,
                 B: Int32, L: Int32, scale2: Float32, scale: Float32, stream: cuda.CUstream):
        H, K, S, V, KB = self.H, self.K, self.S, self.Vd, self.KB
        nc = L // self.C
        mQ = cute.make_tensor(tQ.iterator, cute.make_layout((L, S, H, B), stride=(H * S, 1, S, L * H * S)))
        mK = cute.make_tensor(tK.iterator, cute.make_layout((L, S, K, B), stride=(K * S, 1, S, L * K * S)))
        mV = cute.make_tensor(tV.iterator, cute.make_layout((L, V, K, B), stride=(K * self.V, 1, self.V, L * K * self.V)))
        tm_s = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, KB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        tm_dp = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, KB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        tm_dq = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, S, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))
        a_layout = sm100_utils.make_smem_layout_a(tm_s, (self.TM, KB, S), BFloat16, 1)
        k_layout = sm100_utils.make_smem_layout_b(tm_s, (self.TM, KB, S), BFloat16, self.stages)
        kmn_layout = sm100_utils.make_smem_layout_b(tm_dq, (self.TM, S, KB), BFloat16, self.stages)
        v_layout = sm100_utils.make_smem_layout_b(tm_dp, (self.TM, KB, V), BFloat16, self.stages)
        ds_layout = sm100_utils.make_smem_layout_a(tm_dq, (self.TM, S, KB), BFloat16, 1)
        do_layout = sm100_utils.make_smem_layout_a(tm_dp, (self.TM, KB, V), BFloat16, 1)  # TMEM A operand
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_a, mQ_t = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mQ, cute.slice_(a_layout, (None, None, None, 0)), (self.TM, KB, S), tm_s)
        tma_k, mK_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mK, cute.slice_(k_layout, (None, None, None, 0)), (self.TM, KB, S), tm_s)
        tma_v, mV_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mV, cute.slice_(v_layout, (None, None, None, 0)), (self.TM, KB, V), tm_dp)
        num_tiles = B * nc * H * self.row_tiles
        self.kernel(tm_s, tm_dp, tm_dq, tma_a, mQ_t, tma_k, mK_t, tma_v, mV_t, tDO.iterator, tLse.iterator, tDelta.iterator,
                    tKmin.iterator, tDQ.iterator, tDPprev.iterator, tDPcur.iterator, a_layout, k_layout, kmn_layout, v_layout,
                    ds_layout, do_layout, B, L, nc, scale2, scale).launch(grid=[num_tiles, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tm_s: cute.TiledMma, tm_dp: cute.TiledMma, tm_dq: cute.TiledMma, tma_a: cute.CopyAtom, mQ: cute.Tensor,
               tma_k: cute.CopyAtom, mK: cute.Tensor, tma_v: cute.CopyAtom, mV: cute.Tensor, pDO: cute.Pointer,
               pLse: cute.Pointer, pDelta: cute.Pointer, pKmin: cute.Pointer, pDQ: cute.Pointer, pDPprev: cute.Pointer,
               pDPcur: cute.Pointer, a_layout: cute.ComposedLayout, k_layout: cute.ComposedLayout, kmn_layout: cute.ComposedLayout,
               v_layout: cute.ComposedLayout, ds_layout: cute.ComposedLayout, do_layout: cute.ComposedLayout,
               B: Int32, L: Int32, nc: Int32, scale2: Float32, scale: Float32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H, K, S, V, KB, C = self.H, self.K, self.S, self.V, self.KB, self.C
        rt = bidx % self.row_tiles
        t1 = bidx // self.row_tiles
        h = t1 % H
        t2 = t1 // H
        c = t2 % nc
        b = t2 // nc
        kh = h // self.n_rep
        q0 = c * C + rt * self.TM
        kstart = Int32(0)
        if c > 0:
            kstart = (c - 1) * C
        kmin0 = cute.make_tensor(pKmin + (b * L + q0), cute.make_layout(1))[0]
        kb_lo = (kmin0 - kstart) // KB
        if kb_lo < 0:
            kb_lo = Int32(0)
        kb_hi = (q0 + self.TM - 1 - kstart) // KB
        n_blocks = kb_hi - kb_lo + 1
        kblk0 = kstart // KB + kb_lo

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_k)
            if const_expr(not self.load_dp):
                cpasync.prefetch_descriptor(tma_v)

        @cute.struct
        class SharedStorage:
            a_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            k_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            v_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            s_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            dp_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            ds_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_cw + 1))
        do_bar = pipeline.NamedBarrier(barrier_id=4, num_threads=32 * (self.num_cw + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cw_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * self.num_cw)
        cw_warps = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_cw)
        a_bytes = cute.size_in_bytes(BFloat16, cute.select(a_layout, mode=[0, 1, 2]))
        k_bytes = cute.size_in_bytes(BFloat16, cute.select(k_layout, mode=[0, 1, 2]))
        v_bytes = cute.size_in_bytes(BFloat16, cute.select(v_layout, mode=[0, 1, 2]))
        a_prod, a_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_mbar.data_ptr(), num_stages=1, producer_group=one, consumer_group=one,
            tx_count=a_bytes).make_participants()
        k_prod, k_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.k_mbar.data_ptr(), num_stages=self.stages, producer_group=one, consumer_group=one,
            tx_count=k_bytes).make_participants()
        v_prod, v_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.v_mbar.data_ptr(), num_stages=self.stages, producer_group=one, consumer_group=one,
            tx_count=v_bytes).make_participants()
        s_prod, s_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.s_mbar.data_ptr(), num_stages=1, producer_group=one, consumer_group=cw_warps).make_participants()
        dp_prod, dp_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.dp_mbar.data_ptr(), num_stages=1, producer_group=one, consumer_group=cw_warps).make_participants()
        ds_prod, ds_cons = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.ds_mbar.data_ptr(), num_stages=1, producer_group=cw_group, consumer_group=one).make_participants()
        sA = smem.allocate_tensor(BFloat16, a_layout.outer, byte_alignment=1024, swizzle=a_layout.inner)
        pK = smem.allocate_array(BFloat16, cute.cosize(k_layout.outer), byte_alignment=1024)
        sK = cute.make_tensor(cute.recast_ptr(pK, swizzle_=k_layout.inner), k_layout.outer)
        sKmn = cute.make_tensor(cute.recast_ptr(pK, swizzle_=kmn_layout.inner), kmn_layout.outer)
        if const_expr(not self.load_dp):
            sV = smem.allocate_tensor(BFloat16, v_layout.outer, byte_alignment=1024, swizzle=v_layout.inner)
        pDSs = smem.allocate_array(BFloat16, cute.cosize(ds_layout.outer), byte_alignment=1024)
        sDS = cute.make_tensor(cute.recast_ptr(pDSs, swizzle_=ds_layout.inner), ds_layout.outer)

        gA = cute.local_tile(mQ, (self.TM, S), (None, None, None, None))
        gK = cute.local_tile(mK, (KB, S), (None, None, None, None))
        thr_s = tm_s.get_slice(0)
        thr_dp = tm_dp.get_slice(0)
        tAs, tAg = cpasync.tma_partition(tma_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(thr_s.partition_A(gA), 0, 3))
        tKs, tKg = cpasync.tma_partition(tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(thr_s.partition_B(gK), 0, 3))
        if const_expr(not self.load_dp):
            gV = cute.local_tile(mV, (KB, self.Vd), (None, None, None, None))
            tVs, tVg = cpasync.tma_partition(tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(thr_dp.partition_B(gV), 0, 3))
            tCrV = tm_dp.make_fragment_B(sV)
        tCrA = tm_s.make_fragment_A(sA)
        tCrK = tm_s.make_fragment_B(sK)
        tCrKmn = tm_dq.make_fragment_B(sKmn)
        tCrDS = tm_dq.make_fragment_A(sDS)
        s_fake = tm_s.make_fragment_C(tm_s.partition_shape_C((self.TM, KB)))
        dq_fake = tm_dq.make_fragment_C(tm_dq.partition_shape_C((self.TM, S)))
        tDO_fake = cute.make_tensor(cute.make_ptr(BFloat16, 0, cute.AddressSpace.tmem), do_layout.outer)
        tDOr_fake = tm_dp.make_fragment_A(tDO_fake)[(None, None, None, 0)]

        if warp == self.tma_warp:
            ha = a_prod.acquire_and_advance()
            cute.copy(tma_a, tAg[(None, q0 // self.TM, 0, h, b)], tAs[(None, ha.index)], tma_bar_ptr=ha.barrier)
            for j in cutlass.range(n_blocks):
                hk = k_prod.acquire_and_advance()
                cute.copy(tma_k, tKg[(None, kblk0 + j, 0, kh, b)], tKs[(None, hk.index)], tma_bar_ptr=hk.barrier)
                if const_expr(not self.load_dp):
                    hv = v_prod.acquire_and_advance()
                    cute.copy(tma_v, tVg[(None, kblk0 + j, 0, kh, b)], tVs[(None, hv.index)], tma_bar_ptr=hv.barrier)
            a_prod.tail()
            k_prod.tail()
            if const_expr(not self.load_dp):
                v_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tDP = cute.make_tensor(tmem_ptr + self.tmem_dp, s_fake.layout)
            tDQ = cute.make_tensor(tmem_ptr + self.tmem_dq, dq_fake.layout)
            tDOr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_do, dtype=BFloat16), tDOr_fake.layout)
            if const_expr(not self.load_dp):
                do_bar.arrive_and_wait()  # dO tile stored in TMEM by the compute warps
            a_cons.wait()
            k_rel = k_cons.clone()
            es = s_prod.acquire_and_advance()
            hk = k_cons.wait_and_advance()
            for kk in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                tm_s.set(tcgen05.Field.ACCUMULATE, kk > 0)
                cute.gemm(tm_s, tS, tCrA[(None, None, kk, 0)], tCrK[(None, None, kk, hk.index)], tS)
            es.commit()
            if const_expr(not self.load_dp):
                ed = dp_prod.acquire_and_advance()
                hv = v_cons.wait_and_advance()
                for kk in cutlass.range_constexpr(cute.size(tDOr, mode=[2])):
                    tm_dp.set(tcgen05.Field.ACCUMULATE, kk > 0)
                    cute.gemm(tm_dp, tDP, tDOr[(None, None, kk)], tCrV[(None, None, kk, hv.index)], tDP)
                hv.release()
                ed.commit()
            for j in cutlass.range(n_blocks):
                if j + 1 < n_blocks:
                    es = s_prod.acquire_and_advance()
                    hk = k_cons.wait_and_advance()
                    for kk in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        tm_s.set(tcgen05.Field.ACCUMULATE, kk > 0)
                        cute.gemm(tm_s, tS, tCrA[(None, None, kk, 0)], tCrK[(None, None, kk, hk.index)], tS)
                    es.commit()
                    if const_expr(not self.load_dp):
                        ed = dp_prod.acquire_and_advance()
                        hv = v_cons.wait_and_advance()
                        for kk in cutlass.range_constexpr(cute.size(tDOr, mode=[2])):
                            tm_dp.set(tcgen05.Field.ACCUMULATE, kk > 0)
                            cute.gemm(tm_dp, tDP, tDOr[(None, None, kk)], tCrV[(None, None, kk, hv.index)], tDP)
                        hv.release()
                        ed.commit()
                hs = ds_cons.wait_and_advance()
                kj = k_rel.current_handle().index
                for kk in cutlass.range_constexpr(cute.size(tCrDS, mode=[2])):
                    tm_dq.set(tcgen05.Field.ACCUMULATE, (j > 0) | (kk > 0))
                    cute.gemm(tm_dq, tDQ, tCrDS[(None, None, kk, 0)], tCrKmn[(None, None, kk, kj)], tDQ)
                k_rel.release()
                k_rel.advance()
                hs.release()
            a_cons.release()
            s_prod.tail()
            if const_expr(not self.load_dp):
                dp_prod.tail()
        else:
            tmem.allocate(self.tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tDP = cute.make_tensor(tmem_ptr + self.tmem_dp, s_fake.layout)
            tDQ = cute.make_tensor(tmem_ptr + self.tmem_dq, dq_fake.layout)
            tDOr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_do, dtype=BFloat16), tDOr_fake.layout)
            trow = tidx % 128
            qpos = q0 + trow
            kmin_row = cute.make_tensor(pKmin + (b * L + qpos), cute.make_layout(1))[0]
            kmin_last = cute.make_tensor(pKmin + (b * L + q0 + self.TM - 1), cute.make_layout(1))[0]
            lse_row = cute.make_tensor(pLse + ((b * H + h) * L + qpos), cute.make_layout(1))[0]
            delta_row = cute.make_tensor(pDelta + ((b * H + h) * L + qpos), cute.make_layout(1))[0]
            if const_expr(not self.load_dp):
                # dO tile -> TMEM (A operand of dP): this thread's query row
                st_do = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(V // 2), tcgen05.Unpack.NONE), BFloat16)
                r2t_do = tcgen05.make_tmem_copy(st_do, tDOr)
                thr_do = r2t_do.get_slice(trow)
                tRT_tDO = thr_do.partition_D(tDOr)
                tRT_cdo = thr_do.partition_S(thr_dp.partition_A(cute.make_identity_tensor((self.TM, V))))
                rDO = cute.make_rmem_tensor((V,), BFloat16)
                do_ptr = pDO + (Int64(b * L + qpos) * H + h) * V
                for e8 in cutlass.range_constexpr(V // 8):
                    cute.make_tensor(rDO.iterator + e8 * 8, cute.make_layout(8)).store(
                        cute.make_tensor((do_ptr + e8 * 8).align(16), cute.make_layout(8)).load())
                cute.copy(r2t_do, cute.make_tensor(rDO.iterator, cute.make_layout(tRT_cdo.shape)), tRT_tDO)
                cute.arch.fence_view_async_tmem_store()
                do_bar.arrive_and_wait()
            # dP matrices (load_dp mode): dPcur[(c, b, h)][key - cC][q - cC], dPprev[(c - 1, b, h)][key - (c-1)C][q - cC]
            q_in = Int64(qpos - c * C)
            cur_base = pDPcur + (Int64((b * nc + c) * H + h) * C * C + q_in)  # dPcur [B, nc, H, C, C]
            prev_base = pDPprev + (Int64((b * (nc - 1) + (c - 1)) * H + h) * C * C + q_in)  # dPprev [B, nc - 1, H, C, C]
            ld_s = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(KB), tcgen05.Pack.NONE), Float32)
            tS_r = cute.flat_divide(tS[((None, None), 0, 0)], (self.TM, KB))
            tDP_r = cute.flat_divide(tDP[((None, None), 0, 0)], (self.TM, KB))
            t2r_s = tcgen05.make_tmem_copy(ld_s, tS_r[(None, None, 0, 0)])
            t2r_dp = tcgen05.make_tmem_copy(ld_s, tDP_r[(None, None, 0, 0)])
            tTR_s = t2r_s.get_slice(trow).partition_S(tS_r)
            tTR_s = cute.group_modes(tTR_s, 3, cute.rank(tTR_s))
            tTR_dp = t2r_dp.get_slice(trow).partition_S(tDP_r)
            tTR_dp = cute.group_modes(tTR_dp, 3, cute.rank(tTR_dp))
            xs = cute.make_rmem_tensor(t2r_s.get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, KB))).shape, Float32)
            xd = cute.make_rmem_tensor(xs.shape, Float32)
            dsb = cute.make_rmem_tensor(xs.shape, BFloat16)
            ld_o = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32), tcgen05.Pack.NONE), Float32)
            tDQ_r = cute.flat_divide(tDQ[((None, None), 0, 0)], (self.TM, 32))
            t2r_o = tcgen05.make_tmem_copy(ld_o, tDQ_r[(None, None, 0, 0)])
            tTR_o = t2r_o.get_slice(trow).partition_S(tDQ_r)
            tTR_o = cute.group_modes(tTR_o, 3, cute.rank(tTR_o))
            ov = cute.make_rmem_tensor(t2r_o.get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, 32))).shape, Float32)
            obf = cute.make_rmem_tensor(ov.shape, BFloat16)
            neg_lse = -lse_row
            for j in cutlass.range(n_blocks):
                bstart = kstart + (kb_lo + j) * KB
                fs = s_cons.wait_and_advance()
                cute.copy(t2r_s, tTR_s[(None, None, None, 0)], xs)
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    fs.release()
                if (bstart < kmin_last) | (bstart + KB - 1 > q0):
                    lo = kmin_row - bstart
                    hi = qpos - bstart
                    for e in cutlass.range_constexpr(KB):
                        keep = (e >= lo) & (e <= hi)
                        xs[e] = cutlass.select_(keep, xs[e], Float32(float('-inf')))
                for e2 in cutlass.range_constexpr(KB // 2):
                    y0, y1 = cute.arch.fma_packed_f32x2((xs[2 * e2], xs[2 * e2 + 1]), (scale2, scale2), (neg_lse, neg_lse))
                    xs[2 * e2] = cute.math.exp2(y0, fastmath=True)
                    xs[2 * e2 + 1] = cute.math.exp2(y1, fastmath=True)
                if const_expr(self.load_dp):
                    ckb = bstart // C
                    dp_base = cur_base + Int64(bstart - ckb * C) * C
                    if ckb != c:
                        dp_base = prev_base + Int64(bstart - ckb * C) * C
                    for e in cutlass.range_constexpr(KB):
                        xd[e] = cute.make_tensor(dp_base + e * C, cute.make_layout(1))[0]
                else:
                    fd = dp_cons.wait_and_advance()
                    cute.copy(t2r_dp, tTR_dp[(None, None, None, 0)], xd)
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        fd.release()
                for e in cutlass.range_constexpr(KB):
                    xd[e] = xs[e] * (xd[e] - delta_row) * scale
                dsb.store(xd.load().to(BFloat16))
                hs = ds_prod.acquire_and_advance()
                for e8 in cutlass.range_constexpr(KB // 8):
                    cute.make_tensor((pDSs + (64 * trow + 8 * (e8 ^ (trow % 8)))).align(16), cute.make_layout(8)).store(
                        cute.make_tensor(dsb.iterator + e8 * 8, cute.make_layout(8)).load())
                cute.arch.fence_view_async_shared()
                hs.commit()
            hs = ds_prod.acquire_and_advance()
            row_ptr = pDQ + (Int64(b * L + qpos) * H + h) * S
            for piece in cutlass.range_constexpr(S // 32):
                cute.copy(t2r_o, tTR_o[(None, None, None, piece)], ov)
                cute.arch.fence_view_async_tmem_load()
                obf.store(ov.load().to(BFloat16))
                for e8 in cutlass.range_constexpr(4):
                    cute.make_tensor((row_ptr + (piece * 32 + e8 * 8)).align(16), cute.make_layout(8)).store(
                        cute.make_tensor(obf.iterator + e8 * 8, cute.make_layout(8)).load())
            tmem.relinquish_alloc_permit()
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_cw)
            tmem.free(tmem_ptr)


@cute.jit
def _dq_block(masked, load_dp, g, trow, qpos, bstart, kmin_row, neg_lse, delta_row, scale2, raw, pDSs, ds_elems,
              t2r_s0, t2r_s1, tTR_s0, tTR_s1, t2r_dp0, t2r_dp1, tTR_dp0, tTR_dp1, xs, xd, dsb,
              ph_sf, ph_dpf, ph_dse, KB, dbg, cur_base, prev_base, c, C):
    """One KB-key block of ScaBwdDqT for compute group g (the thread's query row is qpos): P = exp2(S scale2 -
    lse2) (masked: keys outside [kmin_row, qpos] -> -inf), dS = P (dP - delta) into shared memory. load_dp: dP
    comes from the host-side fp32 GEMMs (V > 256; this query's column of dPcur / dPprev, one value per key).
    The masked / unmasked variants are separate code regions on purpose (see _dkdv_block). Returns the phases."""
    cute.arch.mbarrier_wait(raw + (_S_FULL + g), ph_sf)
    ph_sf ^= 1
    if g == 0:
        cute.copy(t2r_s0, tTR_s0[(None, None, None, 0)], xs)
    else:
        cute.copy(t2r_s1, tTR_s1[(None, None, None, 0)], xs)
    if const_expr(not load_dp):
        cute.arch.mbarrier_wait(raw + (_DP_FULL + g), ph_dpf)
        ph_dpf ^= 1
        if g == 0:
            cute.copy(t2r_dp0, tTR_dp0[(None, None, None, 0)], xd)
        else:
            cute.copy(t2r_dp1, tTR_dp1[(None, None, None, 0)], xd)
    cute.arch.fence_view_async_tmem_load()
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(raw + (_S_EMPTY + g))
        if const_expr(not load_dp):
            cute.arch.mbarrier_arrive(raw + (_DP_EMPTY + g))
    if const_expr(load_dp):
        ckb = bstart // C
        dp_base = cur_base + Int64(bstart - ckb * C) * C
        if ckb != c:
            dp_base = prev_base + Int64(bstart - ckb * C) * C
        for e in cutlass.range_constexpr(KB):
            xd[e] = cute.make_tensor(dp_base + e * C, cute.make_layout(1))[0]
    if const_expr(dbg == 0):
        if const_expr(masked):
            lo = kmin_row - bstart
            hi = qpos - bstart
            for e in cutlass.range_constexpr(KB):
                keep = (e >= lo) & (e <= hi)
                xs[e] = cutlass.select_(keep, xs[e], Float32(float('-inf')))
        for e2 in cutlass.range_constexpr(KB // 2):
            y0, y1 = cute.arch.fma_packed_f32x2((xs[2 * e2], xs[2 * e2 + 1]), (scale2, scale2), (neg_lse, neg_lse))
            xs[2 * e2] = cute.math.exp2(y0, fastmath=True)
            xs[2 * e2 + 1] = cute.math.exp2(y1, fastmath=True)
        for e2 in cutlass.range_constexpr(KB // 2):
            d0, d1 = cute.arch.sub_packed_f32x2((xd[2 * e2], xd[2 * e2 + 1]), (delta_row, delta_row))
            xd[2 * e2], xd[2 * e2 + 1] = cute.arch.mul_packed_f32x2((xs[2 * e2], xs[2 * e2 + 1]), (d0, d1))
        dsb.store(xd.load().to(BFloat16))
    cute.arch.mbarrier_wait(raw + (_DS_EMPTY + g), ph_dse)
    ph_dse ^= 1
    # K-major 128 x KB bf16: KB = 64 -> 128-byte swizzle (16-byte chunk c of row m at m*KB + 8*(c ^ (m & 7))),
    # KB = 32 -> 64-byte swizzle (c ^ ((m >> 1) & 3))
    ds_base = pDSs + (g * ds_elems + KB * trow)
    for e8 in cutlass.range_constexpr(KB // 8):
        if const_expr(KB == 64):
            cute.make_tensor((ds_base + 8 * (e8 ^ (trow % 8))).align(16), cute.make_layout(8)).store(
                cute.make_tensor(dsb.iterator + e8 * 8, cute.make_layout(8)).load())
        else:
            cute.make_tensor((ds_base + 8 * (e8 ^ ((trow >> 1) & 3))).align(16), cute.make_layout(8)).store(
                cute.make_tensor(dsb.iterator + e8 * 8, cute.make_layout(8)).load())
    cute.arch.fence_view_async_shared()
    cute.arch.mbarrier_arrive(raw + (_DS_FULL + g))
    return ph_sf, ph_dpf, ph_dse


class ScaBwdDqT:
    """dq[queries of a 128-row tile] = scale sum_k dS K for one (batch, head, chunk, query tile) per CTA, for
    S + V <= 256 (the Q and dO tiles fit tensor memory as the A operands of S = Q K^T and dP = dO V^T, which
    then run at the full tensor-core rate; ScaBwdDq keeps Q in shared memory for the other shapes).

    64-key blocks of K (K-major B of S, MN-major B of dQ += dS K over the same bytes) and V (K-major B of dP)
    stream through TMA rings; S and dP are double-buffered in TMEM and consumed by two groups of 4 compute
    warps (thread = query row) alternating blocks, which form P = exp2(S scale2 - lse2) and
    dS = P (dP - delta) into the shared-memory A operand (2 buffers) of the dQ MMA; the scale is applied in the
    epilogue. The MMA warp issues, per block j, S_{j+2}, dP_{j+2} (stages freed by the compute warps after
    their loads) and dQ_j. TMEM: Q, dO, S x 2, dP x 2, dQ."""

    def __init__(self, H: int, K: int, S: int, V: int, C: int, KB: int = 64, k_stages: int = 4, v_stages: int = 3, q_smem: bool = False,
                 load_dp: bool = False):
        self.H, self.K, self.S, self.V, self.C, self.KB, self.q_smem = H, K, S, V, C, KB, q_smem
        # load_dp (V > 256): dP comes from the host-side fp32 GEMMs; no dO tile / V blocks / dP MMAs
        self.load_dp = load_dp or V > 256
        self.Vd = 64 if self.load_dp else V  # width of the (unused) V tiles in load_dp mode
        self.n_rep = H // K
        self.TM = 128
        self.k_stages = k_stages
        self.v_stages = 1 if self.load_dp else v_stages
        self.num_cw = 8
        self.mma_warp = 8
        self.tma_warp = 9
        self.threads = 32 * 10
        self.row_tiles = C // self.TM
        assert H % K == 0 and S % 64 == 0 and V % 64 == 0 and C % 128 == 0 and C % KB == 0 and KB in (32, 64)
        # q_smem: the q tile stays in shared memory (TMA, A operand of S = Q K^T) for the shapes whose Q + dO tiles do
        # not fit tensor memory next to the stages and dQ (gekko-0.9B: S = V = 256 with KB = 32 -> exactly 512 columns)
        self.tmem_q = 0
        self.tmem_do = 0 if q_smem else S // 2
        self.tmem_s = self.tmem_do + (0 if self.load_dp else V // 2)
        self.tmem_dp = self.tmem_s + 2 * KB
        self.tmem_dq = self.tmem_dp + (0 if self.load_dp else 2 * KB)
        total = self.tmem_dq + S
        assert total <= 512
        self.tmem_cols = max(32, 1 << (total - 1).bit_length())
        self.dbg = int(os.environ.get("SCA_DBG", "0"))

    @cute.jit
    def __call__(self, tQ: cute.Tensor, tK: cute.Tensor, tV: cute.Tensor, tDO: cute.Tensor, tLse: cute.Tensor,
                 tDelta: cute.Tensor, tKmin: cute.Tensor, tDQ: cute.Tensor, tDPprev: cute.Tensor, tDPcur: cute.Tensor,
                 B: Int32, L: Int32, scale2: Float32, scale: Float32, stream: cuda.CUstream):
        H, K, S, KB = self.H, self.K, self.S, self.KB
        V = self.Vd
        nc = L // self.C
        mK = cute.make_tensor(tK.iterator, cute.make_layout((L, S, K, B), stride=(K * S, 1, S, L * K * S)))
        mV = cute.make_tensor(tV.iterator, cute.make_layout((L, V, K, B), stride=(K * self.V, 1, self.V, L * K * self.V)))
        mQ = cute.make_tensor(tQ.iterator, cute.make_layout((L, S, H, B), stride=(H * S, 1, S, L * H * S)))
        tm_s = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, KB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        tm_sq = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(  # S = Q K^T with the q tile in shared memory (q_smem)
            BFloat16, Float32, (self.TM, KB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        tm_dq = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, S, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))
        q_layout = sm100_utils.make_smem_layout_a(tm_s, (self.TM, KB, S), BFloat16, 1)  # TMEM A operands
        a_layout = sm100_utils.make_smem_layout_a(tm_sq, (self.TM, KB, S), BFloat16, 1)  # q tile in smem (q_smem)
        do_layout = sm100_utils.make_smem_layout_a(tm_s, (self.TM, KB, V), BFloat16, 1)
        k_layout = sm100_utils.make_smem_layout_b(tm_s, (self.TM, KB, S), BFloat16, self.k_stages)
        kmn_layout = sm100_utils.make_smem_layout_b(tm_dq, (self.TM, S, KB), BFloat16, self.k_stages)  # same bytes
        v_layout = sm100_utils.make_smem_layout_b(tm_s, (self.TM, KB, V), BFloat16, self.v_stages)
        ds_layout = sm100_utils.make_smem_layout_a(tm_dq, (self.TM, S, KB), BFloat16, 2)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_k, mK_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mK, cute.slice_(k_layout, (None, None, None, 0)), (self.TM, KB, S), tm_s)
        tma_v, mV_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mV, cute.slice_(v_layout, (None, None, None, 0)), (self.TM, KB, V), tm_s)
        tma_a, mQ_t = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mQ, cute.slice_(a_layout, (None, None, None, 0)), (self.TM, KB, S), tm_sq)
        num_tiles = B * nc * H * self.row_tiles
        self.kernel(tm_s, tm_sq, tm_dq, tma_k, mK_t, tma_v, mV_t, tma_a, mQ_t, tQ.iterator, tDO.iterator, tLse.iterator, tDelta.iterator,
                    tKmin.iterator, tDQ.iterator, tDPprev.iterator, tDPcur.iterator, q_layout, a_layout, do_layout, k_layout, kmn_layout, v_layout, ds_layout,
                    B, L, nc, scale2, scale).launch(grid=[num_tiles, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tm_s: cute.TiledMma, tm_sq: cute.TiledMma, tm_dq: cute.TiledMma, tma_k: cute.CopyAtom, mK: cute.Tensor,
               tma_v: cute.CopyAtom, mV: cute.Tensor, tma_a: cute.CopyAtom, mQ: cute.Tensor, pQ: cute.Pointer, pDO: cute.Pointer, pLse: cute.Pointer,
               pDelta: cute.Pointer, pKmin: cute.Pointer, pDQ: cute.Pointer, pDPprev: cute.Pointer, pDPcur: cute.Pointer,
               q_layout: cute.ComposedLayout, a_layout: cute.ComposedLayout,
               do_layout: cute.ComposedLayout, k_layout: cute.ComposedLayout, kmn_layout: cute.ComposedLayout,
               v_layout: cute.ComposedLayout, ds_layout: cute.ComposedLayout, B: Int32, L: Int32, nc: Int32,
               scale2: Float32, scale: Float32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H, K, S, KB, C = self.H, self.K, self.S, self.KB, self.C
        V = self.Vd
        rt = bidx % self.row_tiles
        t1 = bidx // self.row_tiles
        h = t1 % H
        t2 = t1 // H
        c = t2 % nc
        b = t2 // nc
        kh = h // self.n_rep
        q0 = c * C + rt * self.TM
        kstart = Int32(0)
        if c > 0:
            kstart = (c - 1) * C
        kmin0 = cute.make_tensor(pKmin + (b * L + q0), cute.make_layout(1))[0]
        kb_lo = (kmin0 - kstart) // KB
        if kb_lo < 0:
            kb_lo = Int32(0)
        kb_hi = (q0 + self.TM - 1 - kstart) // KB
        n_blocks = kb_hi - kb_lo + 1
        kblk0 = kstart // KB + kb_lo

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_v)
            if const_expr(self.q_smem):
                cpasync.prefetch_descriptor(tma_a)

        @cute.struct
        class SharedStorage:
            a_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            k_mbar: cute.struct.MemRange[cutlass.Int64, self.k_stages * 2]
            v_mbar: cute.struct.MemRange[cutlass.Int64, self.v_stages * 2]
            raw_mbar: cute.struct.MemRange[cutlass.Int64, 16]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        raw = storage.raw_mbar.data_ptr()
        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(raw + (_S_FULL + i), 1)
                    cute.arch.mbarrier_init(raw + (_S_EMPTY + i), 4)
                    cute.arch.mbarrier_init(raw + (_DP_FULL + i), 1)
                    cute.arch.mbarrier_init(raw + (_DP_EMPTY + i), 4)
                    cute.arch.mbarrier_init(raw + (_DS_FULL + i), 128)
                    cute.arch.mbarrier_init(raw + (_DS_EMPTY + i), 1)
                cute.arch.mbarrier_init(raw + _DONE, 1)
            cute.arch.mbarrier_init_fence()
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_cw + 1))
        a_bar = pipeline.NamedBarrier(barrier_id=4, num_threads=32 * (self.num_cw + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        k_bytes = cute.size_in_bytes(BFloat16, cute.select(k_layout, mode=[0, 1, 2]))
        v_bytes = cute.size_in_bytes(BFloat16, cute.select(v_layout, mode=[0, 1, 2]))
        a_bytes = cute.size_in_bytes(BFloat16, cute.select(a_layout, mode=[0, 1, 2]))
        a_prod, a_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_mbar.data_ptr(), num_stages=1, producer_group=one, consumer_group=one,
            tx_count=a_bytes).make_participants()
        k_prod, k_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.k_mbar.data_ptr(), num_stages=self.k_stages, producer_group=one, consumer_group=one,
            tx_count=k_bytes).make_participants()
        v_prod, v_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.v_mbar.data_ptr(), num_stages=self.v_stages, producer_group=one, consumer_group=one,
            tx_count=v_bytes).make_participants()
        cute.arch.barrier()
        pK = smem.allocate_array(BFloat16, cute.cosize(k_layout.outer), byte_alignment=1024)
        sK = cute.make_tensor(cute.recast_ptr(pK, swizzle_=k_layout.inner), k_layout.outer)
        sKmn = cute.make_tensor(cute.recast_ptr(pK, swizzle_=kmn_layout.inner), kmn_layout.outer)
        sV = smem.allocate_tensor(BFloat16, v_layout.outer, byte_alignment=1024, swizzle=v_layout.inner)
        pDSs = smem.allocate_array(BFloat16, cute.cosize(ds_layout.outer), byte_alignment=1024)
        sDS = cute.make_tensor(cute.recast_ptr(pDSs, swizzle_=ds_layout.inner), ds_layout.outer)
        ds_stage_elems = cute.cosize(cute.select(ds_layout.outer, mode=[0, 1, 2]))
        # the q tile in shared memory (q_smem; a 16-byte dummy otherwise so the partitions below type-check)
        pA = smem.allocate_array(BFloat16, cute.cosize(a_layout.outer) if self.q_smem else 8, byte_alignment=1024)
        sA = cute.make_tensor(cute.recast_ptr(pA, swizzle_=a_layout.inner), a_layout.outer)

        gK = cute.local_tile(mK, (KB, S), (None, None, None, None))
        gV = cute.local_tile(mV, (KB, V), (None, None, None, None))
        gA = cute.local_tile(mQ, (self.TM, S), (None, None, None, None))
        thr_s = tm_s.get_slice(0)
        thr_sq = tm_sq.get_slice(0)
        tKs, tKg = cpasync.tma_partition(tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(thr_s.partition_B(gK), 0, 3))
        tVs, tVg = cpasync.tma_partition(tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(thr_s.partition_B(gV), 0, 3))
        tAs, tAg = cpasync.tma_partition(tma_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(thr_sq.partition_A(gA), 0, 3))
        tCrA = tm_sq.make_fragment_A(sA)
        tCrK = tm_s.make_fragment_B(sK)
        tCrV = tm_s.make_fragment_B(sV)
        tCrKmn = tm_dq.make_fragment_B(sKmn)
        tCrDS = tm_dq.make_fragment_A(sDS)
        s_fake = tm_s.make_fragment_C(cute.append(tm_s.partition_shape_C((self.TM, KB)), 2))
        dq_fake = tm_dq.make_fragment_C(tm_dq.partition_shape_C((self.TM, S)))
        tQ_fake = cute.make_tensor(cute.make_ptr(BFloat16, 0, cute.AddressSpace.tmem), q_layout.outer)
        tQr_fake = tm_s.make_fragment_A(tQ_fake)[(None, None, None, 0)]
        tDO_fake = cute.make_tensor(cute.make_ptr(BFloat16, 0, cute.AddressSpace.tmem), do_layout.outer)
        tDOr_fake = tm_s.make_fragment_A(tDO_fake)[(None, None, None, 0)]

        if warp == self.tma_warp:
            if const_expr(self.q_smem):
                ha = a_prod.acquire_and_advance()
                cute.copy(tma_a, tAg[(None, q0 // self.TM, 0, h, b)], tAs[(None, ha.index)], tma_bar_ptr=ha.barrier)
            for j in cutlass.range(n_blocks):
                hk = k_prod.acquire_and_advance()
                cute.copy(tma_k, tKg[(None, kblk0 + j, 0, kh, b)], tKs[(None, hk.index)], tma_bar_ptr=hk.barrier)
                if const_expr(not self.load_dp):
                    hv = v_prod.acquire_and_advance()
                    cute.copy(tma_v, tVg[(None, kblk0 + j, 0, kh, b)], tVs[(None, hv.index)], tma_bar_ptr=hv.barrier)
            if const_expr(self.q_smem):
                a_prod.tail()
            k_prod.tail()
            if const_expr(not self.load_dp):
                v_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tDP = cute.make_tensor(tmem_ptr + self.tmem_dp, s_fake.layout)
            tDQ = cute.make_tensor(tmem_ptr + self.tmem_dq, dq_fake.layout)
            tQr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_q, dtype=BFloat16), tQr_fake.layout)
            tDOr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_do, dtype=BFloat16), tDOr_fake.layout)
            a_bar.arrive_and_wait()  # Q / dO tiles stored in TMEM by the compute warps
            if const_expr(self.q_smem):
                a_cons.wait()
            tCrA0 = tCrA[(None, None, None, 0)]
            op_s = tcgen05.MmaF16BF16Op(BFloat16, Float32, (self.TM, KB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
                                        tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
            op_sq = tcgen05.MmaF16BF16Op(BFloat16, Float32, (self.TM, KB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                                         tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
            op_dq = tcgen05.MmaF16BF16Op(BFloat16, Float32, (self.TM, S, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                                         tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN)
            k_rel = k_cons.clone()
            if const_expr(self.dbg == 3):
                for j in cutlass.range(n_blocks):
                    hk = k_cons.wait_and_advance()
                    if const_expr(self.q_smem):
                        _umma(op_sq, tS[(None, None, None, 0)], tCrA0, tCrK, hk.index, False, raw + _S_FULL)
                    else:
                        _umma(op_s, tS[(None, None, None, 0)], tQr, tCrK, hk.index, False, raw + _S_FULL)
                    if const_expr(not self.load_dp):
                        hv = v_cons.wait_and_advance()
                        _umma(op_s, tDP[(None, None, None, 0)], tDOr, tCrV, hv.index, False, raw + _DP_FULL)
                        hv.release()
                    _umma(op_dq, tDQ, tCrDS[(None, None, None, 0)], tCrKmn, hk.index, True, raw + _DS_EMPTY)
                    hk.release()
            else:
                ph_se0, ph_se1 = Int32(0), Int32(0)
                ph_dpe0, ph_dpe1 = Int32(0), Int32(0)
                ph_dsf0, ph_dsf1 = Int32(0), Int32(0)
                # prologue: S_0, dP_0, S_1, dP_1
                for jj in cutlass.range_constexpr(2):
                    if jj < n_blocks:
                        hk = k_cons.wait_and_advance()
                        if const_expr(self.q_smem):
                            _umma(op_sq, tS[(None, None, None, jj)], tCrA0, tCrK, hk.index, False, raw + (_S_FULL + jj))
                        else:
                            _umma(op_s, tS[(None, None, None, jj)], tQr, tCrK, hk.index, False, raw + (_S_FULL + jj))
                        if const_expr(not self.load_dp):
                            hv = v_cons.wait_and_advance()
                            _umma(op_s, tDP[(None, None, None, jj)], tDOr, tCrV, hv.index, False, raw + (_DP_FULL + jj))
                            hv.release()
                n_pairs = (n_blocks + 1) // 2
                for jp in cutlass.range(n_pairs):
                    j = 2 * jp
                    # ---- block j (stage 0): S_{j+2}, dP_{j+2}, dQ_j ----
                    if j + 2 < n_blocks:
                        cute.arch.mbarrier_wait(raw + (_S_EMPTY + 0), ph_se0)
                        ph_se0 ^= 1
                        hk = k_cons.wait_and_advance()
                        if const_expr(self.q_smem):
                            _umma(op_sq, tS[(None, None, None, 0)], tCrA0, tCrK, hk.index, False, raw + _S_FULL)
                        else:
                            _umma(op_s, tS[(None, None, None, 0)], tQr, tCrK, hk.index, False, raw + _S_FULL)
                        if const_expr(not self.load_dp):
                            cute.arch.mbarrier_wait(raw + (_DP_EMPTY + 0), ph_dpe0)
                            ph_dpe0 ^= 1
                            hv = v_cons.wait_and_advance()
                            _umma(op_s, tDP[(None, None, None, 0)], tDOr, tCrV, hv.index, False, raw + _DP_FULL)
                            hv.release()
                    cute.arch.mbarrier_wait(raw + (_DS_FULL + 0), ph_dsf0)
                    ph_dsf0 ^= 1
                    _umma(op_dq, tDQ, tCrDS[(None, None, None, 0)], tCrKmn, k_rel.current_handle().index, True, raw + _DS_EMPTY)
                    k_rel.release()
                    k_rel.advance()
                    # ---- block j + 1 (stage 1) ----
                    if j + 1 < n_blocks:
                        if j + 3 < n_blocks:
                            cute.arch.mbarrier_wait(raw + (_S_EMPTY + 1), ph_se1)
                            ph_se1 ^= 1
                            hk = k_cons.wait_and_advance()
                            if const_expr(self.q_smem):
                                _umma(op_sq, tS[(None, None, None, 1)], tCrA0, tCrK, hk.index, False, raw + (_S_FULL + 1))
                            else:
                                _umma(op_s, tS[(None, None, None, 1)], tQr, tCrK, hk.index, False, raw + (_S_FULL + 1))
                            if const_expr(not self.load_dp):
                                cute.arch.mbarrier_wait(raw + (_DP_EMPTY + 1), ph_dpe1)
                                ph_dpe1 ^= 1
                                hv = v_cons.wait_and_advance()
                                _umma(op_s, tDP[(None, None, None, 1)], tDOr, tCrV, hv.index, False, raw + (_DP_FULL + 1))
                                hv.release()
                        cute.arch.mbarrier_wait(raw + (_DS_FULL + 1), ph_dsf1)
                        ph_dsf1 ^= 1
                        _umma(op_dq, tDQ, tCrDS[(None, None, None, 1)], tCrKmn, k_rel.current_handle().index, True, raw + (_DS_EMPTY + 1))
                        k_rel.release()
                        k_rel.advance()
            if const_expr(self.q_smem):
                a_cons.release()
            with cute.arch.elect_one():
                tcgen05.commit(raw + _DONE)
        else:
            tmem.allocate(self.tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tDP = cute.make_tensor(tmem_ptr + self.tmem_dp, s_fake.layout)
            tDQ = cute.make_tensor(tmem_ptr + self.tmem_dq, dq_fake.layout)
            tQr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_q, dtype=BFloat16), tQr_fake.layout)
            tDOr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_do, dtype=BFloat16), tDOr_fake.layout)
            grp = warp // 4
            trow = tidx % 128
            qpos = q0 + trow
            kmin_row = cute.make_tensor(pKmin + (b * L + qpos), cute.make_layout(1))[0]
            kmin_last = cute.make_tensor(pKmin + (b * L + q0 + self.TM - 1), cute.make_layout(1))[0]
            neg_lse = -cute.make_tensor(pLse + ((b * H + h) * L + qpos), cute.make_layout(1))[0]
            delta_row = cute.make_tensor(pDelta + ((b * H + h) * L + qpos), cute.make_layout(1))[0]
            # Q (group 0) / dO (group 1) tiles -> TMEM (A operands): this thread's query row; zero the dQ accumulator
            if grp == 0:
              if const_expr(not self.q_smem):
                st_q = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(S // 2), tcgen05.Unpack.NONE), BFloat16)
                r2t_q = tcgen05.make_tmem_copy(st_q, tQr)
                thr_q = r2t_q.get_slice(trow)
                tRT_tQ = thr_q.partition_D(tQr)
                tRT_cq = thr_q.partition_S(thr_s.partition_A(cute.make_identity_tensor((self.TM, S))))
                rQ = cute.make_rmem_tensor((S,), BFloat16)
                q_ptr = pQ + (Int64(b * L + qpos) * H + h) * S
                for e8 in cutlass.range_constexpr(S // 8):
                    cute.make_tensor(rQ.iterator + e8 * 8, cute.make_layout(8)).store(
                        cute.make_tensor((q_ptr + e8 * 8).align(16), cute.make_layout(8)).load())
                cute.copy(r2t_q, cute.make_tensor(rQ.iterator, cute.make_layout(tRT_cq.shape)), tRT_tQ)
            else:
              if const_expr(not self.load_dp):
                st_do = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(V // 2), tcgen05.Unpack.NONE), BFloat16)
                r2t_do = tcgen05.make_tmem_copy(st_do, tDOr)
                thr_do = r2t_do.get_slice(trow)
                tRT_tDO = thr_do.partition_D(tDOr)
                tRT_cdo = thr_do.partition_S(thr_s.partition_A(cute.make_identity_tensor((self.TM, V))))
                rDO = cute.make_rmem_tensor((V,), BFloat16)
                do_ptr = pDO + (Int64(b * L + qpos) * H + h) * V
                for e8 in cutlass.range_constexpr(V // 8):
                    cute.make_tensor(rDO.iterator + e8 * 8, cute.make_layout(8)).store(
                        cute.make_tensor((do_ptr + e8 * 8).align(16), cute.make_layout(8)).load())
                cute.copy(r2t_do, cute.make_tensor(rDO.iterator, cute.make_layout(tRT_cdo.shape)), tRT_tDO)
            st_z = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(32), tcgen05.Unpack.NONE), Float32)
            zr = cute.make_rmem_tensor((32,), Float32)
            for e in cutlass.range_constexpr(32):
                zr[e] = Float32(0.0)
            tDQ_z = cute.flat_divide(tDQ[((None, None), 0, 0)], (self.TM, 32))
            r2t_z = tcgen05.make_tmem_copy(st_z, tDQ_z[(None, None, 0, 0)])
            tRT_z = r2t_z.get_slice(trow).partition_D(tDQ_z)
            tRT_z = cute.group_modes(tRT_z, 3, cute.rank(tRT_z))
            zr_v = cute.make_tensor(zr.iterator, cute.make_layout(r2t_z.get_slice(trow).partition_S(cute.make_identity_tensor((self.TM, 32))).shape))
            for piece in cutlass.range_constexpr(S // 64):
                if grp == 0:
                    cute.copy(r2t_z, zr_v, tRT_z[(None, None, None, 2 * piece)])
                else:
                    cute.copy(r2t_z, zr_v, tRT_z[(None, None, None, 2 * piece + 1)])
            cute.arch.fence_view_async_tmem_store()
            a_bar.arrive_and_wait()
            ld_s = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(KB), tcgen05.Pack.NONE), Float32)
            tS_r = [cute.flat_divide(tS[((None, None), 0, 0, i)], (self.TM, KB)) for i in range(2)]
            tDP_r = [cute.flat_divide(tDP[((None, None), 0, 0, i)], (self.TM, KB)) for i in range(2)]
            t2r_s = [tcgen05.make_tmem_copy(ld_s, tS_r[i][(None, None, 0, 0)]) for i in range(2)]
            t2r_dp = [tcgen05.make_tmem_copy(ld_s, tDP_r[i][(None, None, 0, 0)]) for i in range(2)]
            tTR_s = [t2r_s[i].get_slice(trow).partition_S(tS_r[i]) for i in range(2)]
            tTR_s = [cute.group_modes(tTR_s[i], 3, cute.rank(tTR_s[i])) for i in range(2)]
            tTR_dp = [t2r_dp[i].get_slice(trow).partition_S(tDP_r[i]) for i in range(2)]
            tTR_dp = [cute.group_modes(tTR_dp[i], 3, cute.rank(tTR_dp[i])) for i in range(2)]
            xs = cute.make_rmem_tensor(t2r_s[0].get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, KB))).shape, Float32)
            xd = cute.make_rmem_tensor(xs.shape, Float32)
            dsb = cute.make_rmem_tensor(xs.shape, BFloat16)
            ld_o = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32), tcgen05.Pack.NONE), Float32)
            t2r_o = tcgen05.make_tmem_copy(ld_o, tDQ_z[(None, None, 0, 0)])
            tTR_o = t2r_o.get_slice(trow).partition_S(tDQ_z)
            tTR_o = cute.group_modes(tTR_o, 3, cute.rank(tTR_o))
            ov = cute.make_rmem_tensor(t2r_o.get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, 32))).shape, Float32)
            obf = cute.make_rmem_tensor(ov.shape, BFloat16)
            g = grp
            n_own = (n_blocks - g + 1) // 2
            if const_expr(self.dbg == 3):
                n_own = Int32(0)
            ph_sf = Int32(0)
            ph_dpf = Int32(0)
            ph_dse = Int32(1)
            # dP matrices (load_dp): dPcur[(c, b, h)][key - cC][q - cC], dPprev[(c - 1, b, h)][key - (c-1)C][q - cC]
            q_in = Int64(qpos - c * C)
            cur_base = pDPcur + (Int64((b * nc + c) * H + h) * C * C + q_in)  # dPcur [B, nc, H, C, C]
            prev_base = pDPprev + (Int64((b * (nc - 1) + (c - 1)) * H + h) * C * C + q_in)  # dPprev [B, nc - 1, H, C, C]
            for t in cutlass.range(n_own):
                j = g + 2 * t
                bstart = kstart + (kb_lo + j) * KB
                masked = cute.arch.make_warp_uniform(Int32((bstart < kmin_last) | (bstart + KB - 1 > q0)))
                if masked != 0:
                    ph_sf, ph_dpf, ph_dse = _dq_block(
                        True, self.load_dp, g, trow, qpos, bstart, kmin_row, neg_lse, delta_row, scale2, raw, pDSs, ds_stage_elems,
                        t2r_s[0], t2r_s[1], tTR_s[0], tTR_s[1], t2r_dp[0], t2r_dp[1], tTR_dp[0], tTR_dp[1], xs, xd, dsb,
                        ph_sf, ph_dpf, ph_dse, KB, self.dbg, cur_base, prev_base, c, C)
                else:
                    ph_sf, ph_dpf, ph_dse = _dq_block(
                        False, self.load_dp, g, trow, qpos, bstart, kmin_row, neg_lse, delta_row, scale2, raw, pDSs, ds_stage_elems,
                        t2r_s[0], t2r_s[1], tTR_s[0], tTR_s[1], t2r_dp[0], t2r_dp[1], tTR_dp[0], tTR_dp[1], xs, xd, dsb,
                        ph_sf, ph_dpf, ph_dse, KB, self.dbg, cur_base, prev_base, c, C)
            # epilogue: group g writes the dQ pieces g, g + 2, ...
            cute.arch.mbarrier_wait(raw + _DONE, Int32(0))
            row_ptr = pDQ + (Int64(b * L + qpos) * H + h) * S
            for piece in cutlass.range_constexpr(S // 64):
                if grp == 0:
                    cute.copy(t2r_o, tTR_o[(None, None, None, 2 * piece)], ov)
                else:
                    cute.copy(t2r_o, tTR_o[(None, None, None, 2 * piece + 1)], ov)
                cute.arch.fence_view_async_tmem_load()
                obf.store((ov.load() * scale).to(BFloat16))
                col0 = (2 * piece + grp) * 32
                for e8 in cutlass.range_constexpr(4):
                    cute.make_tensor((row_ptr + (col0 + e8 * 8)).align(16), cute.make_layout(8)).store(
                        cute.make_tensor(obf.iterator + e8 * 8, cute.make_layout(8)).load())
            tmem.relinquish_alloc_permit()
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_cw)
            tmem.free(tmem_ptr)


class ScaDsMat:
    """dS = P (dP - delta) scale (bf16) in the chunk-major layout of ScaPmat ([C queries x 2C keys] per (c, b, h)),
    from P (bf16, same layout), the fp32 dP GEMMs dPq_cur[(c, b, h)] = dO_c V_c^T (keys of chunk c = columns
    [C, 2C)) and dPq_prev[(c - 1, b, h)] = dO_c V_{c-1}^T (columns [0, C); unused for chunk 0) and delta [b, h, l].
    One CTA of 128 threads per row, 8 elements per thread and iteration (fusing this into ScaPmat was slower:
    the strided per-thread dP reads are latency bound there)."""

    def __init__(self, H: int, C: int):
        self.H, self.C = H, C
        assert C % 8 == 0

    @cute.jit
    def __call__(self, tP: cute.Tensor, tDPc: cute.Tensor, tDPp: cute.Tensor, tDelta: cute.Tensor, tDS: cute.Tensor,
                 B: Int32, L: Int32, scale: Float32, stream: cuda.CUstream):
        rows = B * L * self.H
        self.kernel(tP.iterator, tDPc.iterator, tDPp.iterator, tDelta.iterator, tDS.iterator, B, L, scale).launch(
            grid=[rows, 1, 1], block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pP: cute.Pointer, pDPc: cute.Pointer, pDPp: cute.Pointer, pDelta: cute.Pointer, pDS: cute.Pointer,
               B: Int32, L: Int32, scale: Float32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H, C = self.H, self.C
        # row (c, b, h, t) in order
        t = bidx % C
        r1 = bidx // C
        h = r1 % H
        r2 = r1 // H
        b = r2 % B
        c = r2 // B
        delta = cute.make_tensor(pDelta + ((b * H + h) * L + c * C + t), cute.make_layout(1))[0]
        d_s = -delta * scale
        prow = pP + Int64(bidx) * (2 * C)
        dsrow = pDS + Int64(bidx) * (2 * C)
        cur = pDPc + Int64(bidx) * C
        prv = pDPp + Int64(((c - 1) * B + b) * H + h) * C * C + Int64(t) * C
        n_it = (C + 8 * 128 - 1) // (8 * 128)
        lim = (t // 8 + 1) * 8  # P is zero past the diagonal and dP is not written there (sub-blocked GEMM)
        for i in cutlass.range_constexpr(n_it):
            e = (i * 128 + tidx) * 8
            if e < C:
                # columns [C, 2C): the keys of the row's own chunk
                if e < lim:
                    p8 = cute.make_tensor((prow + (C + e)).align(16), cute.make_layout(8)).load().to(Float32)
                    d8 = cute.make_tensor((cur + e).align(16), cute.make_layout(8)).load()
                    cute.make_tensor((dsrow + (C + e)).align(16), cute.make_layout(8)).store((p8 * (d8 * scale + d_s)).to(BFloat16))
                else:
                    z8 = cute.make_tensor((prow + (C + e)).align(16), cute.make_layout(8)).load().to(Float32)
                    cute.make_tensor((dsrow + (C + e)).align(16), cute.make_layout(8)).store((z8 * 0.0).to(BFloat16))
        if c > 0:
            for i in cutlass.range_constexpr(n_it):
                e = (i * 128 + tidx) * 8
                if e < C:
                    p8 = cute.make_tensor((prow + e).align(16), cute.make_layout(8)).load().to(Float32)
                    d8 = cute.make_tensor((prv + e).align(16), cute.make_layout(8)).load()
                    cute.make_tensor((dsrow + e).align(16), cute.make_layout(8)).store((p8 * (d8 * scale + d_s)).to(BFloat16))


class ScaZeroTril:
    """Zero the sub-blocks below the diagonal of N [C x C] fp32 matrices (n x n sub-blocks of c = C / n): row r of
    block i = r // c keeps its columns [0, i c) - the parts the causal dP GEMMs skip and the dK / dQ kernels multiply by
    P^T = 0 (they must not hold NaNs). One warp per row, 16-byte stores."""

    WARPS = 8

    def __init__(self, C: int, n: int):
        self.C, self.n = C, n
        self.c = C // n
        assert C % n == 0 and self.c % 4 == 0

    @cute.jit
    def __call__(self, tM: cute.Tensor, N: Int32, stream: cuda.CUstream):
        rows = N * (self.C - self.c)  # rows of blocks 1 .. n - 1
        self.kernel(tM.iterator, N, rows).launch(grid=[(rows + self.WARPS - 1) // self.WARPS, 1, 1], block=[32 * self.WARPS, 1, 1],
                                                 stream=stream)

    @cute.kernel
    def kernel(self, pM: cute.Pointer, N: Int32, rows: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        ridx = bidx * self.WARPS + tidx // 32
        if ridx < rows:
            per = self.C - self.c
            m = ridx // per
            r = self.c + ridx % per  # row within the matrix (blocks 1 .. n - 1)
            ncols = (r // self.c) * self.c
            base = pM + Int64(m) * self.C * self.C + Int64(r) * self.C
            z = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            for k in cutlass.range_constexpr(4):
                z[k] = 0.0
            for col in cutlass.range(lane * 4, ncols, 128):
                cute.make_tensor((base + col).align(16), cute.make_layout(4)).store(z.load())


class ChunkMajorUncopy:
    """out[b, c C + t, k, :] = x[c, b, k, t, :] (bf16 rows of V): the chunk-major GEMM output back into the
    sequence layout (torch's permute copy of the same bytes runs at ~0.75 TB/s)."""

    def __init__(self, K: int, V: int, C: int):
        self.K, self.V, self.C = K, V, C
        assert V % 8 == 0

    @cute.jit
    def __call__(self, tX: cute.Tensor, tOut: cute.Tensor, B: Int32, L: Int32, stream: cuda.CUstream):
        rows = B * L * self.K
        self.kernel(tX.iterator, tOut.iterator, B, L).launch(grid=[rows, 1, 1], block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pX: cute.Pointer, pOut: cute.Pointer, B: Int32, L: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        K, V, C = self.K, self.V, self.C
        # output row (b, l, k) in order
        k = bidx % K
        r1 = bidx // K
        l = r1 % L
        b = r1 // L
        c = l // C
        t = l - c * C
        src = pX + (((Int64(c * B + b) * K + k) * C + t) * V)
        dst = pOut + (Int64(bidx) * V)
        for i in cutlass.range_constexpr(V // 8 // 128 if V // 8 >= 128 else 1):
            e = (i * 128 + tidx) * 8
            if e < V:
                cute.make_tensor((dst + e).align(16), cute.make_layout(8)).store(
                    cute.make_tensor((src + e).align(16), cute.make_layout(8)).load())


class ChunkMajorCopy:
    """out[c, b, h, t, :] = x[b, c C + t, h // n_rep, :] (bf16 rows of V): the chunk-major, head-repeated copies that
    give the dP GEMMs of the large-V backward uniform batch strides (torch's permute copies run at ~1.7 TB/s)."""

    def __init__(self, H: int, K: int, V: int, C: int):
        self.H, self.K, self.V, self.C = H, K, V, C
        self.n_rep = H // K
        assert V % 8 == 0 and H % K == 0

    @cute.jit
    def __call__(self, tX: cute.Tensor, tOut: cute.Tensor, B: Int32, L: Int32, stream: cuda.CUstream):
        rows = B * L * self.H
        self.kernel(tX.iterator, tOut.iterator, B, L).launch(grid=[rows, 1, 1], block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pX: cute.Pointer, pOut: cute.Pointer, B: Int32, L: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H, K, V, C = self.H, self.K, self.V, self.C
        nc = L // C
        # output row (c, b, h, t) in order, so consecutive CTAs write consecutive rows
        t = bidx % C
        r1 = bidx // C
        h = r1 % H
        r2 = r1 // H
        b = r2 % B
        c = r2 // B
        src = pX + ((Int64(b * L + c * C + t) * K + h // self.n_rep) * V)
        dst = pOut + (Int64(bidx) * V)
        for i in cutlass.range_constexpr(V // 8 // 128 if V // 8 >= 128 else 1):
            e = (i * 128 + tidx) * 8
            if e < V:
                cute.make_tensor((dst + e).align(16), cute.make_layout(8)).store(
                    cute.make_tensor((src + e).align(16), cute.make_layout(8)).load())


class RowDot:
    """delta[b, h, l] = sum_v dout[b, l, h, v] y[b, l, h, v] in fp32 from the bf16 rows (one warp per row)."""

    def __init__(self, H: int, V: int):
        self.H, self.V = H, V
        assert V % 8 == 0

    @cute.jit
    def __call__(self, tDO: cute.Tensor, tY: cute.Tensor, tDelta: cute.Tensor, B: Int32, L: Int32, stream: cuda.CUstream):
        rows = B * L * self.H
        self.kernel(tDO.iterator, tY.iterator, tDelta.iterator, L, rows).launch(
            grid=[(rows + 3) // 4, 1, 1], block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pDO: cute.Pointer, pY: cute.Pointer, pDelta: cute.Pointer, L: Int32, rows: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H, V = self.H, self.V
        lane = tidx % 32
        row = bidx * 4 + tidx // 32  # (b, l, h) in order
        if row < rows:
            acc = Float32(0.0)
            base = Int64(row) * V
            for i in cutlass.range_constexpr((V // 8 + 31) // 32):
                e = (i * 32 + lane) * 8
                if e < V:
                    a = cute.make_tensor((pDO + (base + e)).align(16), cute.make_layout(8)).load()
                    c = cute.make_tensor((pY + (base + e)).align(16), cute.make_layout(8)).load()
                    for k in cutlass.range_constexpr(8):
                        acc = acc + a[k].to(Float32) * c[k].to(Float32)
            for off in cutlass.range_constexpr(5):
                acc = acc + cute.arch.shuffle_sync_bfly(acc, offset=1 << off)
            if lane == 0:
                h = row % H
                bl = row // H
                b = bl // L
                l = bl % L
                cute.make_tensor(pDelta + ((b * H + h) * L + l), cute.make_layout(1))[0] = acc


# ======================================================================================================
# Host side
# ======================================================================================================

# Every layer of a training step derives the same kmin / qmax from the same segment_idx tensor (and the backward
# derives them again): a one-entry cache per helper, keyed on the identity and version of its input, saves the
# ~0.3 ms of host time per call that made the forward of the smaller models launch-bound. Holding a reference to
# the key tensor keeps its id from being reused while the entry is alive.
_SEG_CACHE: dict = {}


def _cached(name, src, make):
    if torch.cuda.is_current_stream_capturing():
        # inside a CUDA graph capture the result must belong to the graph's pool and must not be evicted by a later
        # caller: compute it for this graph and do not cache it
        return make()
    key = (id(src), src._version, src.shape)
    hit = _SEG_CACHE.get(name)
    if hit is not None and hit[0] == key:
        return hit[2]
    out = make()
    _SEG_CACHE[name] = (key, src, out)
    return out


def segment_starts(segment_idx: Optional[torch.Tensor], B: int, L: int, device) -> torch.Tensor:
    """kmin[b, t]: the first position of the segment of position t (0 without segments), int32 [B, L]."""
    if segment_idx is None:
        return torch.zeros(B, L, dtype=torch.int32, device=device)

    def make():
        seg = segment_idx[:, :L]
        idx = torch.arange(L, device=device, dtype=torch.int32).expand(B, L)
        new = torch.ones(B, L, dtype=torch.bool, device=device)
        new[:, 1:] = seg[:, 1:] != seg[:, :-1]
        return torch.cummax(torch.where(new, idx, torch.zeros_like(idx)), dim=1).values.to(torch.int32)

    return _cached(("kmin", L), segment_idx, make)


def query_limits(kmin: torch.Tensor) -> torch.Tensor:
    """qmax[b, k]: the last query whose segment starts at or before key k (int32 [B, L]); kmin is nondecreasing."""
    B, L = kmin.shape

    def make():
        pos = torch.arange(L, device=kmin.device, dtype=kmin.dtype).expand(B, L).contiguous()
        return (torch.searchsorted(kmin.contiguous(), pos, right=True) - 1).to(torch.int32)

    return _cached("qmax", kmin, make)


def sca_bwd_dv(q, k, dout, lse2, kmin, qmax, chunk_size, scale):
    B, L, H, S = q.shape
    K, V = dout.shape[2] // (H // k.shape[2]) if False else k.shape[2], dout.shape[3]
    dv = torch.empty(B, L, K, V, dtype=torch.bfloat16, device=q.device)
    f = _compiled(("sca_bwd_dv", H, K, S, V, chunk_size), lambda: ScaBwdDv(H, K, S, V, chunk_size), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Int32, 4), _fake(Int32, 4),
        _fake(BFloat16), Int32(0), Int32(0), Float32(0.0)))
    f(q.reshape(-1), k.reshape(-1), dout.reshape(-1), lse2.reshape(-1), kmin.reshape(-1), qmax.reshape(-1), dv.reshape(-1),
      B, L, float(scale * LOG2E))
    return dv


def sca_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk_size: int, scale: float,
            kmin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """q [B, L, H, S], k [B, L, K, S], v [B, L, K, V] (bf16, contiguous) -> y [B, L, H, V] bf16, lse2 [B, H, L] fp32."""
    B, L, H, S = q.shape
    K, V = v.shape[2], v.shape[3]
    C = chunk_size
    assert L % C == 0 and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    dev = q.device
    y = torch.empty(B, L, H, V, dtype=torch.bfloat16, device=dev)
    lse2 = torch.empty(B, H, L, dtype=torch.float32, device=dev)
    if os.environ.get("SCA_FWD_TMEM", "1") == "1" and S in (64, 128, 256):  # the q tile -> TMEM store needs a power of two
        # 32-key blocks when the TMEM budget then allows two CTAs per SM, else 64-key blocks (fewer MMA instructions)
        KB = 32 if S // 2 + 64 + min(V, 256) <= 256 else 64
        KB = int(os.environ.get("SCA_FWD_KB", KB))
        f = _compiled(("sca_fwd_t", H, K, S, V, C, KB), lambda: ScaFwdT(H, K, S, V, C, KB=KB), lambda: (
            _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Int32, 4), _fake(BFloat16), _fake(Float32, 4),
            Int32(0), Int32(0), Float32(0.0)))
    else:
        f = _compiled(("sca_fwd", H, K, S, V, C), lambda: ScaFwd(H, K, S, V, C), lambda: (
            _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Int32, 4), _fake(BFloat16), _fake(Float32, 4),
            Int32(0), Int32(0), Float32(0.0)))
    f(q.reshape(-1), k.reshape(-1), v.reshape(-1), kmin.reshape(-1), y.reshape(-1), lse2.reshape(-1), B, L,
      float(scale * LOG2E))
    return y, lse2


def sca_reference(q, k, v, chunk_size, scale, segment_idx=None, dtype=torch.float32):
    """Dense PyTorch reference (per chunk): y [B, L, H, V] and lse2 [B, H, L] (log2 domain, scaled)."""
    B, L, H, S = q.shape
    K, V = v.shape[2], v.shape[3]
    C = chunk_size
    nc = L // C
    n_rep = H // K
    qf, kf, vf = q.to(dtype), k.to(dtype), v.to(dtype)
    kf = kf.repeat_interleave(n_rep, dim=2)
    vf = vf.repeat_interleave(n_rep, dim=2)
    y = torch.zeros(B, L, H, V, dtype=dtype, device=q.device)
    lse2 = torch.zeros(B, H, L, dtype=dtype, device=q.device)
    pos = torch.arange(L, device=q.device)
    for c in range(nc):
        qs, qe = c * C, (c + 1) * C
        ks = 0 if c == 0 else (c - 1) * C
        s = torch.einsum("bqhd,bkhd->bhqk", qf[:, qs:qe], kf[:, ks:qe]) * scale
        mask = pos[ks:qe][None, :] <= pos[qs:qe][:, None]
        if segment_idx is not None:
            mask = mask[None] & (segment_idx[:, ks:qe, None] == segment_idx[:, qs:qe]).transpose(1, 2)[:, None] if False else \
                mask[None] & (segment_idx[:, qs:qe, None] == segment_idx[:, None, ks:qe])
            s = s.masked_fill(~mask[:, None], float("-inf"))
        else:
            s = s.masked_fill(~mask[None, None], float("-inf"))
        m = s.amax(-1, keepdim=True)
        p = torch.exp(s - m)
        l = p.sum(-1, keepdim=True)
        y[:, qs:qe] = torch.einsum("bhqk,bkhv->bqhv", p / l, vf[:, ks:qe])
        lse2[:, :, qs:qe] = (m + torch.log(l)).squeeze(-1) * LOG2E
    return y, lse2
