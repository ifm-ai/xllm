"""Group timestep decay norm implemented with the CuTe Python DSL (forward).

The running decayed statistics of every (batch, group) chain form an affine recurrence along time,
``M_t = a_t * M_{t-1} + (1 - beta) * m_t`` with ``a_t`` in {beta, 0 (bos), 1 (padding)}, so the forward
pass is a single sweep over x with a decoupled look-back scan:

* A tile is (batch b, ``groups_per_tile`` groups, ``tile_len`` consecutive timesteps). Persistent CTAs
  take tiles through an atomic ticket in segment-major order, so a tile's predecessors are always held
  by running CTAs (deadlock free) and are usually already done.
* Warp roles inside a CTA, connected by named barriers and an mbarrier per x stage:
    - producer warp: tickets, row masks, TMA loads of x into an S-stage ring, TMA stores of y;
    - compute warps: per (row, group) moments, then the normalization of the tile D steps earlier,
      written in place into its stage (y), plus the per-row statistics for the backward pass;
    - aggregate warp: the local scan of a tile; publishes the tile aggregate as soon as the moments
      exist (never waiting on other tiles) and replaces each row's moments with its local prefix;
    - resolve warp: the look-back over the preceding tiles of the chain, publishes the inclusive
      prefix, and turns every row's local prefix into its mean / rstd (rows are independent).
* Look-back state is published as 64-bit words that pack a status header with the payload, written
  with relaxed stores: no memory fences, which would also wait for in-flight loads.
* The look-back composes whichever predecessor states are published first, so results are not bitwise
  reproducible run to run (differences are at fp32 rounding level).

A scan element is (k, r, p1, p2, aM, aV) and maps a state (M, V, count) to
``(p1 * M + aM, p2 * V + aV, k + (1 - r) * count)``; ``r`` marks a reset (bos) inside the element.
"""
import math
from functools import lru_cache
from typing import Optional, Tuple

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float16, Float32, Int32, Int64, Uint8, Uint32, const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith as mlir_arith

TORCH2CUTE = {torch.float32: Float32, torch.bfloat16: BFloat16, torch.float16: Float16}

FLAG_AGGREGATE = 1
FLAG_INCLUSIVE = 2
# named barriers from BAR_BASE: NB moment buffers (compute -> aggregate warp), NA hand-off buffers
# (aggregate -> resolve warp), NB statistics buffers (resolve warp -> compute), S y-ready stages
# (compute -> producer)
BAR_BASE = 1


def _compose(x, y):
    """Compose scan elements: apply ``x`` (older) first, then ``y`` (newer)."""
    xk, xr, xp1, xp2, xm, xv = x
    yk, yr, yp1, yp2, ym, yv = y
    return (
        yk + (1 - yr) * xk,
        xr | yr,
        xp1 * yp1,
        xp2 * yp2,
        yp1 * xm + ym,
        yp2 * xv + yv,
    )


def _shuffle_elem(e, offset, up: bool):
    fn = cute.arch.shuffle_sync_up if up else cute.arch.shuffle_sync_down
    return tuple(fn(v, offset=offset) for v in e)


def _select_elem(pred, a, b):
    return tuple(cutlass.select_(pred, u, v) for u, v in zip(a, b))


@lru_cache(maxsize=None)
def _has_f32x2() -> bool:
    """Packed f32x2 add/mul/fma are sm_100+; on Hopper the pair helpers below fall back to scalar ops."""
    return torch.cuda.get_device_capability()[0] >= 10


def _add2(a, b):
    if _has_f32x2():
        return cute.arch.add_packed_f32x2(a, b)
    return a[0] + b[0], a[1] + b[1]


def _mul2(a, b):
    if _has_f32x2():
        return cute.arch.mul_packed_f32x2(a, b)
    return a[0] * b[0], a[1] * b[1]


def _fma2(a, b, c):
    if _has_f32x2():
        return cute.arch.fma_packed_f32x2(a, b, c)
    return a[0] * b[0] + c[0], a[1] * b[1] + c[1]


class TimestepDecayNormFwd:

    def __init__(
        self,
        dtype,
        w_dtype,
        s_dtype,
        hidden: int,
        num_groups: int,
        has_bos: bool,
        has_pad: bool,
        groups_per_tile: int,
        tile_len: int,
        num_threads: int,
        prefetch: int,
        lag: int,
        min_ctas_per_sm: int = 0,
        vec_mult: int = 1,
    ):
        self.dtype = dtype
        self.w_dtype = w_dtype
        self.s_dtype = s_dtype
        self.H = hidden
        self.G = num_groups
        self.Dg = hidden // num_groups
        self.has_bos = has_bos
        self.has_pad = has_pad
        self.Gt = groups_per_tile
        self.T = tile_len
        self.NT = num_threads
        self.P = prefetch  # tiles in flight ahead of the one being reduced
        self.D = lag  # tiles between the moments and the normalization of a tile
        self.S = self.D + 1 + self.P  # x ring stages
        # moment / statistics hand-off buffers. Compute warps are not synchronized with each other, so
        # one may already write the moments of tile i+1 while another still reads the statistics of
        # tile i-D (stored in place of its moments): NB must not divide D+1.
        self.NB = self.D + 2
        # two resolve warps take alternate tiles (the look-back latency of one overlaps the other); the
        # aggregate -> resolve hand-off buffers come in pairs so that each resolve warp owns its barrier ids
        self.NRW = 2
        self.NA = (self.D + 2 + self.NRW - 1) // self.NRW * self.NRW
        self.min_ctas_per_sm = min_ctas_per_sm

        width = max(dtype.width, w_dtype.width)
        self.VEC = math.gcd(self.Dg, 128 * vec_mult // width)  # elements per thread and row
        self.W = self.Gt * self.Dg  # columns per tile
        self.TPG = self.Dg // self.VEC  # threads per (row, group)
        self.TPR = self.W // self.VEC  # threads per row
        # warp roles: NCW compute warps (moments, normalization), an aggregate warp (local scan,
        # aggregate publish), NRW resolve warps (look-back, per-row statistics) and a producer warp
        # (tickets, TMA loads of x, TMA stores of y)
        self.NC = num_threads
        self.NCW = self.NC // 32
        self.NT = self.NC + 32 * (2 + self.NRW)
        self.R = self.NC // self.TPR  # rows per pass of the compute warps
        self.RPT = self.T // self.R  # rows per compute thread
        self.LPR = self.T // 32  # rows per lane of the scan warp
        self.num_group_tiles = self.G // self.Gt

        assert hidden % num_groups == 0
        assert num_groups % groups_per_tile == 0
        assert self.TPG & (self.TPG - 1) == 0 and self.TPG <= 32, "threads per group must be a power of two <= 32"
        assert self.NC % 32 == 0 and self.NC % self.TPR == 0, "a row must not be split across passes"
        assert self.T % self.R == 0 and self.T % 32 == 0
        assert self.Gt <= 32 and self.D >= 1 and self.P >= 0 and self.S <= 32
        assert self.VEC % 2 == 0, "packed f32x2 arithmetic needs an even vector width"
        assert BAR_BASE + 2 * self.NB + self.S <= 16, "named barriers"

    @cute.jit
    def __call__(
        self,
        tX: cute.Tensor,
        tBos: cute.Tensor,
        tPad: cute.Tensor,
        tPrevCount: cute.Tensor,
        tPrevMean: cute.Tensor,
        tPrevVar: cute.Tensor,
        tGamma: cute.Tensor,
        tBeta: cute.Tensor,
        tY: cute.Tensor,
        tCount: cute.Tensor,
        tMean: cute.Tensor,
        tVar: cute.Tensor,
        tCumMean: cute.Tensor,
        tCumRstd: cute.Tensor,
        tStepCount: cute.Tensor,
        tWsInt: cute.Tensor,
        tWs64: cute.Tensor,
        B: Int32,
        L: Int32,
        num_segs: Int32,
        beta1: Float32,
        beta2: Float32,
        eps: Float32,
        num_ctas: Int32,
        stream: cuda.CUstream,
    ):
        # every argument is a flat tensor (TVM-FFI calling convention); the kernel works on pointers
        pX = tX.iterator
        pBos = tBos.iterator
        pPad = tPad.iterator
        pPrevCount = tPrevCount.iterator
        pPrevMean = tPrevMean.iterator
        pPrevVar = tPrevVar.iterator
        pGamma = tGamma.iterator
        pBeta = tBeta.iterator
        pY = tY.iterator
        pCount = tCount.iterator
        pMean = tMean.iterator
        pVar = tVar.iterator
        pCumMean = tCumMean.iterator
        pCumRstd = tCumRstd.iterator
        pStepCount = tStepCount.iterator
        pWsInt = tWsInt.iterator
        pWs64 = tWs64.iterator
        # TMA descriptors for x and y viewed as [B, L, H], one (1, T, W) box per tile
        xy_layout = cute.make_layout((B, L, self.H), stride=(L * self.H, self.H, 1))
        tile_smem = cute.make_ordered_layout((1, self.T, self.W), order=(2, 1, 0))
        tma_load, mXt = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), cute.make_tensor(pX, xy_layout), tile_smem, (1, self.T, self.W))
        tma_store, mYt = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), cute.make_tensor(pY, xy_layout), tile_smem, (1, self.T, self.W))
        self.kernel(
            tma_load, mXt, tma_store, mYt,
            pBos, pPad, pPrevCount, pPrevMean, pPrevVar, pGamma, pBeta,
            pCount, pMean, pVar, pCumMean, pCumRstd, pStepCount,
            pWsInt, pWs64, B, L, num_segs, beta1, beta2, eps,
        ).launch(grid=[num_ctas, 1, 1], block=[self.NT, 1, 1], stream=stream, min_blocks_per_mp=self.min_ctas_per_sm)

    @cute.kernel
    def kernel(
        self,
        tma_load: cute.CopyAtom,
        mXt: cute.Tensor,
        tma_store: cute.CopyAtom,
        mYt: cute.Tensor,
        pBos: cute.Pointer,
        pPad: cute.Pointer,
        pPrevCount: cute.Pointer,
        pPrevMean: cute.Pointer,
        pPrevVar: cute.Pointer,
        pGamma: cute.Pointer,
        pBeta: cute.Pointer,
        pCount: cute.Pointer,
        pMean: cute.Pointer,
        pVar: cute.Pointer,
        pCumMean: cute.Pointer,
        pCumRstd: cute.Pointer,
        pStepCount: cute.Pointer,
        pWsInt: cute.Pointer,
        pWs64: cute.Pointer,
        B: Int32,
        L: Int32,
        num_segs: Int32,
        beta1: Float32,
        beta2: Float32,
        eps: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()

        num_chains = B * self.num_group_tiles
        num_tiles = num_chains * num_segs

        # Global views are built here: inside nested dynamic regions the DSL supports tensor
        # indexing, but not pointer arithmetic that involves compile-time attributes.
        mBos = cute.make_tensor(pBos, cute.make_layout((B, L), stride=(L, 1)))
        mPad = cute.make_tensor(pPad, cute.make_layout((B, L), stride=(L, 1)))
        mPrevCount = cute.make_tensor(pPrevCount, cute.make_layout(B))
        mCount = cute.make_tensor(pCount, cute.make_layout(B))
        mPrevMean = cute.make_tensor(pPrevMean, cute.make_layout((B, self.G), stride=(self.G, 1)))
        mPrevVar = cute.make_tensor(pPrevVar, cute.make_layout((B, self.G), stride=(self.G, 1)))
        mMean = cute.make_tensor(pMean, cute.make_layout((B, self.G), stride=(self.G, 1)))
        mVar = cute.make_tensor(pVar, cute.make_layout((B, self.G), stride=(self.G, 1)))
        mCumMean = cute.make_tensor(pCumMean, cute.make_layout((B, self.G, L), stride=(self.G * L, L, 1)))
        mCumRstd = cute.make_tensor(pCumRstd, cute.make_layout((B, self.G, L), stride=(self.G * L, L, 1)))
        mStepCount = cute.make_tensor(pStepCount, cute.make_layout((B, L), stride=(L, 1)))
        # workspace (zeroed by the host): [ticket counter | aggregate words (tile, group, 2) | inclusive words]
        pTicket = pWsInt
        mAggW = cute.make_tensor(pWs64 + 2, cute.make_layout(num_tiles * self.Gt * 2))
        mIncW = cute.make_tensor(pWs64 + (2 + num_tiles * self.Gt * 2), cute.make_layout(num_tiles * self.Gt * 2))
        w_dyn = Int32(self.W)
        vec_dyn = Int32(self.VEC)

        smem = cutlass.memory.SmemAllocator()
        # S-stage ring of x tiles: TMA loads x into a stage, the compute warps overwrite it with y,
        # and a TMA store writes it back
        sX = smem.allocate_tensor(
            self.dtype, cute.make_layout((1, self.T, self.W, self.S), stride=(self.T * self.W, self.W, 1, self.T * self.W)),
            byte_alignment=128)
        full_bar = smem.allocate_array(Int64, num_elems=self.S)
        # aggregate -> resolve hand-off, one per buffer (slot a is reused by tile j + NA, whose aggregate
        # starts only after tile j was resolved: NA >= D + 1)
        agg_bar = smem.allocate_array(Int64, num_elems=self.NA)
        # per stage: the tile's ticket and its row codes (bit 0: live, bit 1: bos), written by the producer
        sTicket = smem.allocate_tensor(Int32, cute.make_layout(self.S), byte_alignment=16)
        # per stage: (b, seg, gtile) of its tile, decoded once by the producer
        sTile = smem.allocate_tensor(Int32, cute.make_layout((self.S, 4), stride=(4, 1)), byte_alignment=16)
        sCode = smem.allocate_tensor(Uint8, cute.make_layout((self.S, self.T), stride=(self.T, 1)), byte_alignment=16)
        # compute -> aggregate / resolve (moments) and resolve -> compute (statistics): NB buffers.
        # The resolve warp overwrites a row's moments with its mean / rstd in place.
        stat_layout = cute.make_layout((self.NB, self.Gt, self.T), stride=(self.Gt * self.T, self.T, 1))
        sCm = smem.allocate_tensor(Float32, stat_layout, byte_alignment=16)
        sRs = smem.allocate_tensor(Float32, stat_layout, byte_alignment=16)
        # per row: (step count << 1) | live
        sInfo = smem.allocate_tensor(Int32, cute.make_layout((self.NB, self.T), stride=(self.T, 1)), byte_alignment=16)
        # aggregate -> resolve: NA buffers
        sAgg = smem.allocate_tensor(Float32, cute.make_layout((self.NA, self.Gt, 2), stride=(2 * self.Gt, 2, 1)), byte_alignment=16)
        sAggMaskI = smem.allocate_tensor(Int32, cute.make_layout((self.NA, 2), stride=(2, 1)), byte_alignment=8)
        # per row of the local prefix: live | reset << 1 | k << 2
        sRow = smem.allocate_tensor(Int32, cute.make_layout((self.NA, self.T), stride=(self.T, 1)), byte_alignment=16)
        sTk = smem.allocate_tensor(Int32, cute.make_layout(self.NA), byte_alignment=8)
        sTkInfo = smem.allocate_tensor(Int32, cute.make_layout((self.NA, 4), stride=(4, 1)), byte_alignment=16)
        sIn = smem.allocate_tensor(Float32, cute.make_layout((self.NRW, self.Gt, 2), stride=(2 * self.Gt, 2, 1)), byte_alignment=16)
        sInC = smem.allocate_tensor(Int32, cute.make_layout((self.NRW, 1), stride=(1, 1)))

        one_minus_beta1 = 1.0 - beta1
        one_minus_beta2 = 1.0 - beta2
        log2_beta1 = cute.math.log2(beta1)
        log2_beta2 = cute.math.log2(beta2)
        rsqrt_eps = cute.math.rsqrt(eps)
        inv_dg = Float32(1.0 / self.Dg)
        stages = Int32(self.S)
        nb = Int32(self.NB)
        na = Int32(self.NA)
        handoff = self.NC + 32  # compute warps + one of the other warps

        if tidx < self.S:
            cute.arch.mbarrier_init(full_bar + tidx, 1)
        if tidx < self.NA:
            cute.arch.mbarrier_init(agg_bar + tidx, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if warp < self.NCW:
            # ================================ compute warps ================================================
            # iteration i: moments of tile i, normalization of tile i-D (y in place, then YREADY)
            rW = cute.make_rmem_tensor((self.VEC,), self.w_dtype)
            rB = cute.make_rmem_tensor((self.VEC,), self.w_dtype)
            cidx = tidx % self.TPR
            rsub = tidx // self.TPR
            gl = (cidx * self.VEC) // self.Dg  # group of this thread inside the tile
            tile_elems = Int32(self.T * self.W)
            rOld = cute.make_rmem_tensor((self.D,), Int32)  # tickets of tiles i-D .. i-1
            rOld.fill(-1)
            it = Int32(0)
            running = Int32(1)
            while running != 0:
                stage = it % stages
                cute.arch.mbarrier_wait(full_bar + stage, (it // stages) % 2)
                cur = sTicket[stage]
                if cur >= num_tiles:
                    running = Int32(0)
                else:
                    buf = it % nb
                    old = rOld[0]
                    if old >= 0:
                        self._load_affine(sTile[(it - self.D) % stages, 2], rW, rB, pGamma, pBeta, cidx, w_dyn, vec_dyn)
                    self._moments(sX, stage * tile_elems + cidx * vec_dyn, buf, cidx, rsub, gl, w_dyn, inv_dg, sCm, sRs)
                    cute.arch.barrier_arrive(barrier_id=BAR_BASE + buf, number_of_threads=handoff)

                    if old >= 0:
                        self._finish_tile(sTile, it - self.D, rW, rB, pGamma, pBeta, sX, sCm, sRs, sInfo,
                                          mCumMean, mCumRstd, mStepCount, num_chains, L, tidx, cidx, rsub, gl,
                                          stages, nb, tile_elems, w_dyn, vec_dyn)
                    for d in cutlass.range_constexpr(self.D - 1):
                        rOld[d] = rOld[d + 1]
                    rOld[self.D - 1] = cur
                    it = it + 1

            # termination: let the aggregate warp read the invalid ticket, then finish the last D tiles
            cute.arch.barrier_arrive(barrier_id=BAR_BASE + it % nb, number_of_threads=handoff)
            for d in cutlass.range_constexpr(self.D):
                old = rOld[d]
                if old >= 0:
                    self._load_affine(sTile[(it - self.D + d) % stages, 2], rW, rB, pGamma, pBeta, cidx, w_dyn, vec_dyn)
                    self._finish_tile(sTile, it - self.D + d, rW, rB, pGamma, pBeta, sX, sCm, sRs, sInfo,
                                      mCumMean, mCumRstd, mStepCount, num_chains, L, tidx, cidx, rsub, gl,
                                      stages, nb, tile_elems, w_dyn, vec_dyn)
        elif warp == self.NCW:
            # ================================ aggregate warp ===============================================
            # Scans each tile locally and publishes its aggregate right away: it never waits on global
            # state, so the aggregates that other tiles' look-back depends on appear as soon as the
            # data does.
            jt = Int32(0)
            running = Int32(1)
            while running != 0:
                cute.arch.barrier(barrier_id=BAR_BASE + jt % nb, number_of_threads=handoff)
                abuf = jt % na
                tk = sTicket[jt % stages]
                if lane == 0:
                    sTk[abuf] = tk
                if lane < 3:
                    sTkInfo[abuf, lane] = sTile[jt % stages, lane]
                if tk >= num_tiles:
                    running = Int32(0)
                    # also stop the other resolve warp, which waits for the following tile
                    nbuf = (jt + 1) % na
                    if lane == 0:
                        sTk[nbuf] = tk
                    cute.arch.sync_warp()
                    if lane == 0:
                        cute.arch.mbarrier_arrive(agg_bar + nbuf)
                else:
                    self._aggregate_tile(tk, jt % nb, abuf, jt % stages, lane, mAggW, sCode, sCm, sRs, sRow,
                                         sAgg, sAggMaskI, beta1, beta2, one_minus_beta1, one_minus_beta2)
                    jt = jt + 1
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(agg_bar + abuf)
        elif warp <= self.NCW + self.NRW:
            # ================================ resolve warp =================================================
            rw = warp - self.NCW - 1
            jt = rw
            running = Int32(1)
            while running != 0:
                abuf = jt % na
                cute.arch.mbarrier_wait(agg_bar + abuf, (jt // na) % 2)
                tk = sTk[abuf]
                if tk >= num_tiles:
                    running = Int32(0)
                else:
                    self._resolve_tile(tk, sTkInfo, jt % nb, abuf, num_chains, num_segs, L, lane,
                                       mAggW, mIncW, mPrevCount, mPrevMean, mPrevVar, mCount, mMean, mVar,
                                       sRow, sAgg, sAggMaskI, sIn[rw, None, None], sInC[rw, None], sCm, sRs, sInfo,
                                       log2_beta1, log2_beta2, eps, rsqrt_eps)
                    cute.arch.barrier_arrive(barrier_id=BAR_BASE + self.NB + jt % nb, number_of_threads=handoff)
                    jt = jt + self.NRW
        else:
            # ================================ producer warp ================================================
            # tickets, row codes, TMA loads of x and TMA stores of y (once the compute warps release a stage)
            gX = cute.local_tile(mXt, (1, self.T, self.W), (None, None, None))
            gY = cute.local_tile(mYt, (1, self.T, self.W), (None, None, None))
            tsX, tgX = cpasync.tma_partition(tma_load, 0, cute.make_layout(1), cute.group_modes(sX, 0, 3), cute.group_modes(gX, 0, 3))
            tsY, tgY = cpasync.tma_partition(tma_store, 0, cute.make_layout(1), cute.group_modes(sX, 0, 3), cute.group_modes(gY, 0, 3))
            # Tickets: lanes 0 and 1 hold the next two tickets. A CTA consumes its tickets in increasing order
            # (needed for deadlock freedom), so the first two come as one contiguous block and each consumed
            # slot is refilled with a fresh ticket used two tiles later. Claiming tiles only shortly before
            # they are processed keeps the look-back waits short.
            base = Int32(0)
            if lane == 0:
                base = cute.arch.atomic_add(pTicket, Int32(2), sem="relaxed", scope="gpu")
            my_ticket = cute.arch.shuffle_sync(base, 0) + lane
            rBos = [cute.make_rmem_tensor((self.LPR,), Uint8) for _ in range(2)]
            rPad = [cute.make_rmem_tensor((self.LPR,), Uint8) for _ in range(2)]
            self._load_raw_masks(cute.arch.shuffle_sync(my_ticket, 0), rBos[0], rPad[0], mBos, mPad, num_chains, num_tiles, L, lane)
            jt = Int32(0)
            running = Int32(1)
            while running != 0:
                for hh in cutlass.range_constexpr(2):
                    if running != 0:
                        stage = jt % stages
                        tk = cute.arch.shuffle_sync(my_ticket, hh)
                        tk_next = cute.arch.shuffle_sync(my_ticket, 1 - hh)
                        if tk < num_tiles:
                            if lane == hh:
                                my_ticket = cute.arch.atomic_add(pTicket, Int32(1), sem="relaxed", scope="gpu")
                        # raw masks of the next tile, decoded one iteration later
                        self._load_raw_masks(tk_next, rBos[1 - hh], rPad[1 - hh], mBos, mPad, num_chains, num_tiles, L, lane)
                        # reuse the stage: store the y of tile jt-S once the compute warps released it
                        if jt >= stages:
                            cute.arch.barrier(barrier_id=BAR_BASE + 2 * self.NB + stage, number_of_threads=handoff)
                            self._store_y(sTile, stage, tma_store, tsY, tgY)
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                        seg = tk // num_chains
                        chain = tk - seg * num_chains
                        b = chain // self.num_group_tiles
                        gtile = chain - b * self.num_group_tiles
                        if lane == 0:
                            sTicket[stage] = tk
                            sTile[stage, 0] = b
                            sTile[stage, 1] = seg
                            sTile[stage, 2] = gtile
                        self._store_codes(tk, rBos[hh], rPad[hh], sCode, stage, num_chains, num_tiles, L, lane)
                        cute.arch.sync_warp()
                        if tk < num_tiles:
                            if lane == 0:
                                cute.arch.mbarrier_arrive_and_expect_tx(full_bar + stage, self.T * self.W * self.dtype.width // 8)
                            cute.copy(tma_load, tgX[None, b, seg, gtile], tsX[None, stage], tma_bar_ptr=full_bar + stage)
                        else:
                            if lane == 0:
                                cute.arch.mbarrier_arrive(full_bar + stage)
                            running = Int32(0)
                        jt = jt + 1
            # drain: y of the last S-1 loaded tiles (the last iteration was the invalid ticket)
            for d in cutlass.range_constexpr(self.S - 1):
                m = jt - self.S + d
                if m >= 0:
                    mst = m % stages
                    cute.arch.barrier(barrier_id=BAR_BASE + 2 * self.NB + mst, number_of_threads=handoff)
                    self._store_y(sTile, mst, tma_store, tsY, tgY)
            cute.arch.cp_async_bulk_wait_group(0)


    def _load_row_f32(self, ptr):
        """VEC consecutive inputs from (16-byte aligned) shared memory as a list of fp32 values. bf16 is the
        upper half of an fp32, so a bf16 pair in a 32-bit word w widens to (w << 16, w & 0xffff0000): one
        integer op per element."""
        if const_expr(self.dtype == BFloat16):
            words = cute.make_tensor(cute.recast_ptr(ptr, dtype=Uint32), cute.make_layout(self.VEC // 2)).load()
            vals = []
            for k in range(self.VEC // 2):
                vals.append(_u32_as_f32(words[k] << 16))
                vals.append(_u32_as_f32(words[k] & Uint32(0xFFFF0000)))
            return vals
        v = cute.make_tensor(ptr, cute.make_layout(self.VEC)).load().to(Float32)
        return [v[k] for k in range(self.VEC)]

    def _scatter_rows(self, per_row, li):
        """Sum per-row partials over the TPG lanes of a (row, group) by a reduce-scatter: each shuffle level
        (largest offset first) exchanges half of the remaining rows, so lane ``li`` ends up with the totals of
        ``k`` consecutive row slots (block * k + q) after len(row) * RPT * (1 - 1 / 2^s) shuffles instead of
        len(row) * RPT * log2(TPG). Levels left over when RPT has fewer factors of two than TPG are butterflies
        (those lanes duplicate the result). Returns (k rows of totals, block, whether this lane writes)."""
        levels = int(math.log2(self.TPG))
        n = len(per_row[0])
        sl = 0
        while sl < levels and (self.RPT >> sl) % 2 == 0:
            sl += 1
        vals = [v for row in per_row for v in row]
        for o in range(sl):
            off = self.TPG >> (o + 1)
            upper = (li & off) != 0
            half = len(vals) // 2
            vals = [cutlass.select_(upper, vals[i + half], vals[i])
                    + cute.arch.shuffle_sync_bfly(cutlass.select_(upper, vals[i], vals[i + half]), offset=off)
                    for i in range(half)]
        for o in range(levels - sl):
            vals = [v + cute.arch.shuffle_sync_bfly(v, offset=1 << o) for v in vals]
        k = self.RPT >> sl
        dup = levels - sl
        return [vals[q * n:(q + 1) * n] for q in range(k)], li >> dup, (li & ((1 << dup) - 1)) == 0

    @cute.jit
    def _moments(self, sX, xs_base, buf, cidx, rsub, gl, w_dyn, inv_dg, sMomM, sMomV):
        """Compute warps: mean and (biased) variance of each (row, group) of this thread's rows.

        Stage-major over the RPT rows (independent chains interleave), packed f32x2 arithmetic on the
        raw inputs, and a single predicated store block at the end."""
        npair = self.VEC // 2
        levels = int(math.log2(self.TPG))
        rows = [rsub + i * self.R for i in range(self.RPT)]
        xs = [self._load_row_f32((sX.iterator + (xs_base + row * w_dyn)).align(16)) for row in rows]
        sums = []
        for i in cutlass.range_constexpr(self.RPT):
            pairs = [(xs[i][2 * k], xs[i][2 * k + 1]) for k in range(npair)]
            for lv in cutlass.range_constexpr(int(math.log2(npair))):
                pairs = [_add2(pairs[2 * k], pairs[2 * k + 1]) for k in range(npair >> (lv + 1))]
            sums.append(pairs[0][0] + pairs[0][1])
        for o in cutlass.range_constexpr(levels):
            sums = [v + cute.arch.shuffle_sync_bfly(v, offset=1 << o) for v in sums]
        means = [v * inv_dg for v in sums]
        sqs = []
        for i in cutlass.range_constexpr(self.RPT):
            m = (-means[i], -means[i])
            d = [_add2((xs[i][2 * k], xs[i][2 * k + 1]), m) for k in range(npair)]
            acc = _mul2(d[0], d[0])
            for k in cutlass.range_constexpr(1, npair):
                acc = _fma2(d[k], d[k], acc)
            sqs.append(acc[0] + acc[1])
        # the means are needed by every lane (butterflies); the squared deviations only by the lane that
        # stores the row, so they are reduce-scattered
        mine, block, writer = self._scatter_rows([(v,) for v in sqs], cidx % self.TPG)
        for q in cutlass.range_constexpr(len(mine)):
            slot = block * len(mine) + q
            mean = means[q]
            for i in cutlass.range_constexpr(q + len(mine), self.RPT, len(mine)):
                mean = cutlass.select_(slot == i, means[i], mean)
            if writer:
                sMomM[buf, gl, rsub + slot * self.R] = mean
                sMomV[buf, gl, rsub + slot * self.R] = mine[q][0] * inv_dg

    @cute.jit
    def _load_affine(self, gtile, rW, rB, pGamma, pBeta, cidx, w_dyn, vec_dyn):
        """This thread's gamma / beta slice for the group block ``gtile``."""
        col = gtile * w_dyn + cidx * vec_dyn
        rW.store(cute.make_tensor((pGamma + col).align(16), cute.make_layout(self.VEC)).load())
        rB.store(cute.make_tensor((pBeta + col).align(16), cute.make_layout(self.VEC)).load())

    @cute.jit
    def _store_y(self, sTile, stage, tma_store, tsY, tgY):
        """Producer warp: TMA store of the y tile held in ring stage ``stage``."""
        cute.copy(tma_store, tsY[None, stage], tgY[None, sTile[stage, 0], sTile[stage, 1], sTile[stage, 2]])
        cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def _load_raw_masks(self, tk, rBos, rPad, mBos, mPad, num_chains, num_tiles, L, lane):
        """Producer warp: raw mask bytes of this lane's LPR rows of tile ``tk`` (kept in flight)."""
        if tk < num_tiles:
            seg = tk // num_chains
            chain = tk - seg * num_chains
            b = chain // self.num_group_tiles
            t0 = seg * self.T + lane * self.LPR
            for jj in cutlass.range_constexpr(self.LPR):
                if t0 + jj < L:
                    if const_expr(self.has_bos):
                        rBos[jj] = mBos[b, t0 + jj]
                    if const_expr(self.has_pad):
                        rPad[jj] = mPad[b, t0 + jj]

    @cute.jit
    def _store_codes(self, tk, rBos, rPad, sCode, stage, num_chains, num_tiles, L, lane):
        """Producer warp: row codes (bit 0: live, bit 1: bos) of tile ``tk`` into its stage."""
        seg = tk // num_chains
        t0 = seg * self.T + lane * self.LPR
        codes = cute.make_rmem_tensor((self.LPR,), Uint8)
        for jj in cutlass.range_constexpr(self.LPR):
            live = cutlass.select_(t0 + jj < L, Int32(1), Int32(0))
            if const_expr(self.has_pad):
                live = live & (1 - Int32(rPad[jj]))
            code = live
            if const_expr(self.has_bos):
                code = live | ((live & Int32(rBos[jj])) << 1)
            codes[jj] = Uint8(code)
        cute.local_tile(sCode[stage, None], (self.LPR,), (lane,)).store(codes.load())

    @cute.jit
    def _aggregate_tile(self, tk, h, a, stage, lane, mAggW, sCode, sMomM, sMomV, sRow, sAgg, sAggMaskI,
                        beta1, beta2, one_minus_beta1, one_minus_beta2):
        """Aggregate warp: local scan of tile ``tk``. Publishes the tile aggregate, and replaces every row's
        moments with its local prefix (the row's state if the tile started from a zero state), so that
        the resolve warp only needs one independent composition per row. Each lane owns LPR consecutive
        rows, accessed with vector shared-memory loads / stores (no bank conflicts)."""
        identity = (Int32(0), Int32(0), Float32(1.0), Float32(1.0), Float32(0.0), Float32(0.0))
        raw = cute.local_tile(sCode[stage, None], (self.LPR,), (lane,)).load()
        codes = cute.make_rmem_tensor((self.LPR,), Int32)
        for jj in cutlass.range_constexpr(self.LPR):
            codes[jj] = Int32(raw[jj])
        rm = cute.make_rmem_tensor((self.LPR,), Float32)
        rv = cute.make_rmem_tensor((self.LPR,), Float32)
        rrow = cute.make_rmem_tensor((self.LPR,), Int32)
        for g in cutlass.range_constexpr(self.Gt):
            mv = cute.local_tile(sMomM[h, g, None], (self.LPR,), (lane,)).load()
            vv = cute.local_tile(sMomV[h, g, None], (self.LPR,), (lane,)).load()
            elems = [_row_elem(codes[jj], mv[jj], vv[jj], beta1, beta2, one_minus_beta1, one_minus_beta2)
                     for jj in range(self.LPR)]
            acc = identity
            for jj in cutlass.range_constexpr(self.LPR):
                acc = _compose(acc, elems[jj])
            for o in cutlass.range_constexpr(5):
                other = _shuffle_elem(acc, 1 << o, up=True)
                acc = _select_elem(lane >= (1 << o), _compose(other, acc), acc)
            state = _select_elem(lane >= 1, _shuffle_elem(acc, 1, up=True), identity)
            for jj in cutlass.range_constexpr(self.LPR):
                state = _compose(state, elems[jj])
                rm[jj] = state[4]
                rv[jj] = state[5]
                # live | reset << 1 | k << 2 of the local prefix
                rrow[jj] = (codes[jj] & 1) | (state[1] << 1) | (state[0] << 2)
            cute.local_tile(sMomM[h, g, None], (self.LPR,), (lane,)).store(rm.load())
            cute.local_tile(sMomV[h, g, None], (self.LPR,), (lane,)).store(rv.load())
            if g == 0:
                cute.local_tile(sRow[a, None], (self.LPR,), (lane,)).store(rrow.load())
            if lane == 31:
                hdr = _agg_header(acc[0], acc[1])
                cute.arch.store((mAggW.iterator + (tk * self.Gt + g) * 2).llvm_ptr, _pack(hdr, acc[4]), sem="relaxed", scope="gpu")
                cute.arch.store((mAggW.iterator + (tk * self.Gt + g) * 2 + 1).llvm_ptr, _pack(hdr, acc[5]), sem="relaxed", scope="gpu")
                sAgg[a, g, 0] = acc[4]
                sAgg[a, g, 1] = acc[5]
                if g == 0:
                    sAggMaskI[a, 0] = acc[0]
                    sAggMaskI[a, 1] = acc[1]

    @cute.jit
    def _resolve_tile(self, tk, sTkInfo, h, a, num_chains, num_segs, L, lane,
                      mAggW, mIncW, mPrevCount, mPrevMean, mPrevVar, mCount, mMean, mVar,
                      sRow, sAgg, sAggMaskI, sIn, sInC, sCm, sRs, sInfo,
                      log2_beta1, log2_beta2, eps, rsqrt_eps):
        """Resolve warp: look-back for tile ``tk``, inclusive publish, then per row (independently) the
        absolute state from the row's local prefix, and its mean / rstd (written in place)."""
        b = sTkInfo[a, 0]
        seg = sTkInfo[a, 1]
        gtile = sTkInfo[a, 2]
        chain = b * self.num_group_tiles + gtile
        g0 = gtile * self.Gt
        self._lookback(tk, seg, chain, b, g0, gtile, num_chains, num_segs, lane,
                       mAggW, mIncW, mPrevCount, mPrevMean, mPrevVar, mCount, mMean, mVar,
                       sAgg[a, None, None], sAggMaskI[a, None], sIn, sInC, log2_beta1, log2_beta2)
        cute.arch.sync_warp()
        c0 = sInC[0]
        # beta^count = beta^k * beta^c0 unless the row's prefix contains a reset (then beta^k)
        c0f = Int32(c0).to(Float32)
        b1c0 = cute.math.exp2(c0f * log2_beta1, fastmath=True)
        b2c0 = cute.math.exp2(c0f * log2_beta2, fastmath=True)
        info = cute.local_tile(sRow[a, None], (self.LPR,), (lane,)).load()
        rcm = cute.make_rmem_tensor((self.LPR,), Float32)
        rrs = cute.make_rmem_tensor((self.LPR,), Float32)
        rinfo = cute.make_rmem_tensor((self.LPR,), Int32)
        counts = []
        dec1 = []
        dec2 = []
        corr1 = []
        corr2 = []
        for jj in cutlass.range_constexpr(self.LPR):
            k = info[jj] >> 2
            reset = (info[jj] >> 1) & 1
            cnt = k + (1 - reset) * c0
            kf = Int32(k).to(Float32)
            p1 = cute.math.exp2(kf * log2_beta1, fastmath=True)
            p2 = cute.math.exp2(kf * log2_beta2, fastmath=True)
            dec1.append(cutlass.select_(reset != 0, Float32(0.0), p1))
            dec2.append(cutlass.select_(reset != 0, Float32(0.0), p2))
            corr1.append(cute.arch.rcp_approx(1.0 - cutlass.select_(reset != 0, p1, p1 * b1c0)))
            corr2.append(cute.arch.rcp_approx(1.0 - cutlass.select_(reset != 0, p2, p2 * b2c0)))
            counts.append(cnt)
            rinfo[jj] = (cnt << 1) | (info[jj] & 1)
        for g in cutlass.range_constexpr(self.Gt):
            m0 = sIn[g, 0]
            v0 = sIn[g, 1]
            am = cute.local_tile(sCm[h, g, None], (self.LPR,), (lane,)).load()
            av = cute.local_tile(sRs[h, g, None], (self.LPR,), (lane,)).load()
            for jj in cutlass.range_constexpr(self.LPR):
                has_cnt = counts[jj] > 0
                M = dec1[jj] * m0 + am[jj]
                V = dec2[jj] * v0 + av[jj]
                rcm[jj] = cutlass.select_(has_cnt, M * corr1[jj], Float32(0.0))
                rrs[jj] = cutlass.select_(has_cnt, cute.math.rsqrt(V * corr2[jj] + eps), rsqrt_eps)
            cute.local_tile(sCm[h, g, None], (self.LPR,), (lane,)).store(rcm.load())
            cute.local_tile(sRs[h, g, None], (self.LPR,), (lane,)).store(rrs.load())
        cute.local_tile(sInfo[h, None], (self.LPR,), (lane,)).store(rinfo.load())

    @cute.jit
    def _finish_tile(self, sTile, j, rW, rB, pGamma, pBeta, sX, sCm, sRs, sInfo,
                     mCumMean, mCumRstd, mStepCount, num_chains, L, tidx, cidx, rsub, gl,
                     stages, nb, tile_elems, w_dyn, vec_dyn):
        """Compute warps: normalize the tile of local index ``j`` in place in its stage, release the
        stage to the producer for the TMA store of y, and store the per-row statistics."""
        b = sTile[j % stages, 0]
        seg = sTile[j % stages, 1]
        gtile = sTile[j % stages, 2]
        t0 = seg * self.T
        h = j % nb
        stage = j % stages
        cute.arch.barrier(barrier_id=BAR_BASE + self.NB + h, number_of_threads=self.NC + 32)
        w = rW.load().to(Float32)
        bias = rB.load().to(Float32)
        xs_base = stage * tile_elems + cidx * vec_dyn
        npair = self.VEC // 2
        rows = [rsub + i * self.R for i in range(self.RPT)]
        tiles = [cute.make_tensor((sX.iterator + (xs_base + row * w_dyn)).align(16), cute.make_layout(self.VEC)) for row in rows]
        xs = [self._load_row_f32(tl.iterator) for tl in tiles]
        for i in cutlass.range_constexpr(self.RPT):
            row = rows[i]
            rs = sRs[h, gl, row]
            shift = -sCm[h, gl, row] * rs
            x = xs[i]
            y = cute.make_rmem_tensor((self.VEC,), Float32)
            if const_expr(self.has_pad):
                # padded rows are zero (rows past L are clipped by the TMA store anyway)
                live = Int32(sInfo[h, row] & 1).to(Float32)
                for k in cutlass.range_constexpr(npair):
                    t = _fma2((x[2 * k], x[2 * k + 1]), (rs, rs), (shift, shift))
                    bl = _mul2((bias[2 * k], bias[2 * k + 1]), (live, live))
                    t = _mul2(t, (live, live))
                    y[2 * k], y[2 * k + 1] = _fma2(t, (w[2 * k], w[2 * k + 1]), bl)
            else:
                for k in cutlass.range_constexpr(npair):
                    # y = ((x - mean) * rstd) * gamma + beta
                    t = _fma2((x[2 * k], x[2 * k + 1]), (rs, rs), (shift, shift))
                    y[2 * k], y[2 * k + 1] = _fma2(t, (w[2 * k], w[2 * k + 1]), (bias[2 * k], bias[2 * k + 1]))
            tiles[i].store(y.load().to(self.dtype))
        # make y visible to the TMA unit, then hand the stage to the producer
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier_arrive(barrier_id=BAR_BASE + 2 * self.NB + stage, number_of_threads=self.NC + 32)
        # per-row statistics for the backward pass, coalesced across the compute threads
        g0 = gtile * self.Gt
        for k in cutlass.range_constexpr((self.Gt * self.T + self.NC - 1) // self.NC):
            e = tidx + k * self.NC
            g = e // self.T
            row = e - g * self.T
            t = t0 + row
            if e < self.Gt * self.T and t < L:
                mCumMean[b, g0 + g, t] = sCm[h, g, row]
                mCumRstd[b, g0 + g, t] = sRs[h, g, row]
                if g == 0 and gtile == 0:
                    mStepCount[b, t] = sInfo[h, row] >> 1


    @cute.jit
    def _lookback(self, ticket, seg, chain, b, g0, gtile, num_chains, num_segs, lane,
                  mAggW, mIncW, mPrevCount, mPrevMean, mPrevVar, mCount, mMean, mVar,
                  sAgg, sAggMaskI, sIn, sInC, log2_beta1, log2_beta2):
        """Resolve the exclusive prefix of a tile and publish its inclusive prefix.

        Every published word packs a status header with its payload, so publishing is a single
        relaxed 64-bit store and needs no memory fence (a fence would also wait for the in-flight
        prefetch loads of the next tile).
        """
        agg_k = sAggMaskI[0]
        agg_r = sAggMaskI[1]

        # exclusive prefix, accumulated right-to-left over windows of 32 predecessors
        ex_k = Int32(0)
        ex_r = Int32(0)
        ex_p1 = Float32(1.0)
        ex_p2 = Float32(1.0)
        exM = cute.make_rmem_tensor((self.Gt,), Float32)
        exV = cute.make_rmem_tensor((self.Gt,), Float32)
        exM.fill(0.0)
        exV.fill(0.0)
        wM = cute.make_rmem_tensor((self.Gt,), Float32)
        wV = cute.make_rmem_tensor((self.Gt,), Float32)
        aM = cute.make_rmem_tensor((self.Gt,), Float32)
        aV = cute.make_rmem_tensor((self.Gt,), Float32)
        cur = seg - 1
        done = Int32(0)
        while done == 0:
            pseg = cur - lane
            pslot = pseg * num_chains + chain
            # poll until every lane up to the nearest inclusive prefix holds a published state
            flag = Int32(0)
            ek = Int32(0)
            er = Int32(0)
            first = Int32(32)
            polling = Int32(1)
            while polling != 0:
                if flag == 0:
                    if pseg < 0:
                        flag = Int32(FLAG_INCLUSIVE)
                        ek = Int32(mPrevCount[b])
                        er = Int32(1)
                        for gg in cutlass.range_constexpr(self.Gt):
                            wM[gg] = mPrevMean[b, g0 + gg].to(Float32)
                            wV[gg] = mPrevVar[b, g0 + gg].to(Float32)
                    else:
                        # one round trip: fetch the inclusive and the aggregate words together
                        inc_ok = Int32(1)
                        agg_ok = Int32(1)
                        inc_hdr = Int32(0)
                        agg_hdr = Int32(0)
                        for gg in cutlass.range_constexpr(self.Gt):
                            i0 = cute.arch.load((mIncW.iterator + (pslot * self.Gt + gg) * 2).llvm_ptr, Int64, sem="relaxed", scope="gpu")
                            i1 = cute.arch.load((mIncW.iterator + (pslot * self.Gt + gg) * 2 + 1).llvm_ptr, Int64, sem="relaxed", scope="gpu")
                            a0 = cute.arch.load((mAggW.iterator + (pslot * self.Gt + gg) * 2).llvm_ptr, Int64, sem="relaxed", scope="gpu")
                            a1 = cute.arch.load((mAggW.iterator + (pslot * self.Gt + gg) * 2 + 1).llvm_ptr, Int64, sem="relaxed", scope="gpu")
                            inc_hdr = _hi(i0)
                            agg_hdr = _hi(a0)
                            inc_ok = inc_ok & _status_is(inc_hdr, FLAG_INCLUSIVE) & _status_is(_hi(i1), FLAG_INCLUSIVE)
                            agg_ok = agg_ok & _status_is(agg_hdr, FLAG_AGGREGATE) & _status_is(_hi(a1), FLAG_AGGREGATE)
                            wM[gg] = _lo_f32(i0)
                            wV[gg] = _lo_f32(i1)
                            aM[gg] = _lo_f32(a0)
                            aV[gg] = _lo_f32(a1)
                        for gg in cutlass.range_constexpr(self.Gt):
                            wM[gg] = cutlass.select_(inc_ok != 0, wM[gg], aM[gg])
                            wV[gg] = cutlass.select_(inc_ok != 0, wV[gg], aV[gg])
                        if inc_ok != 0:
                            flag = Int32(FLAG_INCLUSIVE)
                            ek = inc_hdr & COUNT_MASK
                            er = Int32(1)
                        elif agg_ok != 0:
                            flag = Int32(FLAG_AGGREGATE)
                            ek = agg_hdr & AGG_K_MASK
                            er = (agg_hdr >> 29) & 1
                incl_mask = cute.arch.vote_ballot_sync(flag == FLAG_INCLUSIVE)
                first = cutlass.select_(incl_mask == 0, Int32(32), cute.arch.popc(incl_mask ^ (incl_mask - 1)) - 1)
                polling = cutlass.select_(cute.arch.vote_all_sync((lane > first) | (flag != 0)), Int32(0), Int32(1))

            if first == 0:
                # common case: the nearest predecessor is already inclusive (or the initial state)
                ek = cute.arch.shuffle_sync(ek, 0)
                for gg in cutlass.range_constexpr(self.Gt):
                    exM[gg] = ex_p1 * cute.arch.shuffle_sync(wM[gg], 0) + exM[gg]
                    exV[gg] = ex_p2 * cute.arch.shuffle_sync(wV[gg], 0) + exV[gg]
                ex_k = ex_k + (1 - ex_r) * ek
                ex_r = Int32(1)
                ex_p1 = Float32(0.0)
                ex_p2 = Float32(0.0)
            else:
                # this lane's element; identity beyond the first inclusive lane
                use = lane <= first
                is_agg = flag == FLAG_AGGREGATE
                ek = cutlass.select_(use, ek, Int32(0))
                er = cutlass.select_(use, er, Int32(0))
                kf = Int32(ek).to(Float32)
                ep1 = cutlass.select_(er != 0, Float32(0.0), cute.math.exp2(kf * log2_beta1, fastmath=True))
                ep2 = cutlass.select_(er != 0, Float32(0.0), cute.math.exp2(kf * log2_beta2, fastmath=True))
                ep1 = cutlass.select_(use & is_agg, ep1, cutlass.select_(use, Float32(0.0), Float32(1.0)))
                ep2 = cutlass.select_(use & is_agg, ep2, cutlass.select_(use, Float32(0.0), Float32(1.0)))
                for gg in cutlass.range_constexpr(self.Gt):
                    wM[gg] = cutlass.select_(use, wM[gg], Float32(0.0))
                    wV[gg] = cutlass.select_(use, wV[gg], Float32(0.0))

                # ordered tree reduction: higher lanes are older
                for o in cutlass.range_constexpr(5):
                    sh = 1 << o
                    has_src = lane + sh < 32
                    sk = cute.arch.shuffle_sync_down(ek, offset=sh)
                    sr = cute.arch.shuffle_sync_down(er, offset=sh)
                    sp1 = cute.arch.shuffle_sync_down(ep1, offset=sh)
                    sp2 = cute.arch.shuffle_sync_down(ep2, offset=sh)
                    for gg in cutlass.range_constexpr(self.Gt):
                        om = cute.arch.shuffle_sync_down(wM[gg], offset=sh)
                        ov = cute.arch.shuffle_sync_down(wV[gg], offset=sh)
                        wM[gg] = cutlass.select_(has_src, ep1 * om + wM[gg], wM[gg])
                        wV[gg] = cutlass.select_(has_src, ep2 * ov + wV[gg], wV[gg])
                    ek = cutlass.select_(has_src, ek + (1 - er) * sk, ek)
                    er = cutlass.select_(has_src, er | sr, er)
                    ep1 = cutlass.select_(has_src, sp1 * ep1, ep1)
                    ep2 = cutlass.select_(has_src, sp2 * ep2, ep2)
                # broadcast the window composite from lane 0 and fold it in front of the running prefix
                ek = cute.arch.shuffle_sync(ek, 0)
                er = cute.arch.shuffle_sync(er, 0)
                ep1 = cute.arch.shuffle_sync(ep1, 0)
                ep2 = cute.arch.shuffle_sync(ep2, 0)
                for gg in cutlass.range_constexpr(self.Gt):
                    wm = cute.arch.shuffle_sync(wM[gg], 0)
                    wv = cute.arch.shuffle_sync(wV[gg], 0)
                    exM[gg] = ex_p1 * wm + exM[gg]
                    exV[gg] = ex_p2 * wv + exV[gg]
                ex_k = ex_k + (1 - ex_r) * ek
                ex_r = ex_r | er
                ex_p1 = ep1 * ex_p1
                ex_p2 = ep2 * ex_p2
            done = cutlass.select_(first < 32, Int32(1), Int32(0))
            cur = cur - 32

        # inclusive prefix of this tile = exclusive prefix followed by the tile aggregate
        inc_k = agg_k + (1 - agg_r) * ex_k
        agg_p1 = cutlass.select_(agg_r != 0, Float32(0.0), cute.math.exp2(Float32(agg_k) * log2_beta1, fastmath=True))
        agg_p2 = cutlass.select_(agg_r != 0, Float32(0.0), cute.math.exp2(Float32(agg_k) * log2_beta2, fastmath=True))
        for gg in cutlass.range_constexpr(self.Gt):
            if lane == gg:
                inM = agg_p1 * exM[gg] + sAgg[gg, 0]
                inV = agg_p2 * exV[gg] + sAgg[gg, 1]
                hdr = _inc_header(inc_k)
                cute.arch.store((mIncW.iterator + (ticket * self.Gt + gg) * 2).llvm_ptr, _pack(hdr, inM), sem="relaxed", scope="gpu")
                cute.arch.store((mIncW.iterator + (ticket * self.Gt + gg) * 2 + 1).llvm_ptr, _pack(hdr, inV), sem="relaxed", scope="gpu")
                sIn[gg, 0] = exM[gg]
                sIn[gg, 1] = exV[gg]
                if seg == num_segs - 1:
                    mMean[b, g0 + gg] = inM.to(self.s_dtype)
                    mVar[b, g0 + gg] = inV.to(self.s_dtype)
        if lane == 0:
            sInC[0] = ex_k
            if seg == num_segs - 1 and gtile == 0:
                mCount[b] = Int64(inc_k)


# Published words: [status:2 | reset:1 | k:29] (aggregate) or [status:2 | count:30] (inclusive) in
# the high 32 bits, the fp32 payload in the low 32 bits.
COUNT_MASK = (1 << 30) - 1
AGG_K_MASK = (1 << 29) - 1


def _f32_bits(x):
    return Int32(mlir_arith.bitcast(ir.IntegerType.get_signless(32), Float32(x).ir_value()))


def _lo_f32(w):
    return Float32(mlir_arith.bitcast(ir.F32Type.get(), Int32(w & Int64(0xFFFFFFFF)).ir_value()))


def _u32_as_f32(u):
    return Float32(mlir_arith.bitcast(ir.F32Type.get(), Uint32(u).ir_value()))


def _hi(w):
    return Int32(w >> 32)


def _pack(hdr, value):
    return (Int64(hdr) << 32) | (Int64(_f32_bits(value)) & Int64(0xFFFFFFFF))


def _status_is(hdr, status):
    return cutlass.select_(((hdr >> 30) & 3) == status, Int32(1), Int32(0))


def _agg_header(k, r):
    return Int32(FLAG_AGGREGATE << 30) | (r << 29) | (k & AGG_K_MASK)


def _inc_header(count):
    # counts saturate at 2^30 - 1; beta^count has long underflowed to zero by then
    return Int32(-(1 << 31)) | cutlass.select_(count > COUNT_MASK, Int32(COUNT_MASK), count)


def _row_elem(code, m, v, beta1, beta2, one_minus_beta1, one_minus_beta2):
    """Scan element for one timestep of one group; ``code`` bit 0 = live, bit 1 = reset (bos)."""
    live = code & 1
    reset = (code >> 1) & 1
    is_live = live != 0
    decays = reset == 0
    return (
        live,
        reset,
        cutlass.select_(is_live, cutlass.select_(decays, beta1, Float32(0.0)), Float32(1.0)),
        cutlass.select_(is_live, cutlass.select_(decays, beta2, Float32(0.0)), Float32(1.0)),
        cutlass.select_(is_live, one_minus_beta1 * m, Float32(0.0)),
        cutlass.select_(is_live, one_minus_beta2 * v, Float32(0.0)),
    )


_COMPILED = {}


def _fake(dtype, align):
    """A flat fake tensor of unknown length for compilation."""
    return cute.runtime.make_fake_compact_tensor(dtype, (cute.sym_int(),), assumed_align=align)


@lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


SMEM_PER_SM = 227 * 1024  # usable shared memory per SM on sm_90 / sm_100 (1 KiB reserved per CTA)


def _fwd_smem_bytes(esize: int, Dg: int, gt: int, tile_len: int, prefetch: int, lag: int) -> int:
    """Shared memory of one CTA (mirrors the allocations in the kernel, plus alignment slack)."""
    T, W = tile_len, gt * Dg
    S, NB, NA = lag + 1 + prefetch, lag + 2, (lag + 3) // 2 * 2
    x_ring = S * T * W * esize
    moments = 2 * NB * gt * T * 4 + NB * T * 4
    handoff = NA * (gt * 2 * 4 + 2 * 4 + T * 4 + 4)
    return x_ring + moments + handoff + S * (T + 4 + 8) + 2 * (gt * 8 + 4) + NA * 8 + 1024


@lru_cache(maxsize=None)
def default_fwd_config(dtype: torch.dtype, H: int, G: int) -> Tuple[int, int, int, int, int, int]:
    """(groups_per_tile, tile_len, compute_threads, ctas_per_sm, prefetch, lag).

    Tiles of about 32 KiB of x, one tile of prefetch and one of lag; as many CTAs per SM as the
    shared memory allows (at most 3).
    """
    esize = dtype.itemsize
    Dg = H // G
    gt = 1
    while gt * Dg * esize < 64 and G % (gt * 2) == 0 and gt < 4:
        gt *= 2
    row_bytes = gt * Dg * esize
    tile_len = 256
    while tile_len > 64 and tile_len * row_bytes > 32 * 1024:
        tile_len //= 2
    prefetch, lag = 1, 1
    ctas = max(1, min(3, SMEM_PER_SM // (_fwd_smem_bytes(esize, Dg, gt, tile_len, prefetch, lag) + 1024)))
    return gt, tile_len, 256, ctas, prefetch, lag


def timestep_decay_norm_fwd(
    x: torch.Tensor,
    bos_mask: Optional[torch.Tensor],
    prev_count: torch.Tensor,
    prev_mean: torch.Tensor,
    prev_var: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
    num_groups: int,
    beta1: float,
    beta2: float,
    eps: float,
    config: Optional[Tuple[int, int, int, int, int, int]] = None,
):
    """Returns (y, count, mean, var, cummean, cumrstd, step_count)."""
    assert x.is_cuda and x.dim() == 3
    B, L, H = x.shape
    G = num_groups
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    prev_count = prev_count.contiguous()
    prev_mean = prev_mean.contiguous()
    prev_var = prev_var.contiguous()
    gt, tile_len, nt, ctas_per_sm, prefetch, lag = config if config is not None else default_fwd_config(x.dtype, H, G)
    num_segs = (L + tile_len - 1) // tile_len
    num_tiles = B * (G // gt) * num_segs
    num_ctas = min(num_tiles, _sm_count(x.device.index) * ctas_per_sm)

    y = torch.empty_like(x)
    count = torch.empty_like(prev_count)
    mean = torch.empty_like(prev_mean)
    var = torch.empty_like(prev_var)
    cummean = torch.empty(B, G, L, dtype=torch.float32, device=x.device)
    cumrstd = torch.empty(B, G, L, dtype=torch.float32, device=x.device)
    step_count = torch.empty(B, L, dtype=torch.int32, device=x.device)
    # [ticket counter | aggregate words | inclusive words], all zero (= unpublished)
    ws = torch.zeros(2 + num_tiles * gt * 4, dtype=torch.int64, device=x.device)

    has_bos = bos_mask is not None
    has_pad = padding_mask is not None
    key = (x.dtype, gamma.dtype, prev_mean.dtype, H, G, has_bos, has_pad, gt, tile_len, nt, prefetch, lag, ctas_per_sm)
    dummy = ws.view(torch.uint8)  # any device tensor stands in for an absent mask
    tensors = [
        x, bos_mask.view(torch.uint8) if has_bos else dummy, padding_mask.view(torch.uint8) if has_pad else dummy,
        prev_count, prev_mean, prev_var, gamma, beta, y, count, mean, var, cummean, cumrstd, step_count,
        ws.view(torch.int32), ws,
    ]
    if key not in _COMPILED:
        aligns = [16, 1, 1, 8, 2, 2, 16, 16, 16, 8, 2, 2, 4, 4, 4, 16, 16]
        fakes = [_fake(TORCH2CUTE.get(t.dtype, {torch.uint8: Uint8, torch.int64: Int64, torch.int32: Int32}.get(t.dtype)), al)
                 for t, al in zip(tensors, aligns)]
        op = TimestepDecayNormFwd(
            TORCH2CUTE[x.dtype], TORCH2CUTE[gamma.dtype], TORCH2CUTE[prev_mean.dtype],
            H, G, has_bos, has_pad, gt, tile_len, nt, prefetch, lag, ctas_per_sm,
        )
        _COMPILED[key] = cute.compile(
            op, *fakes, Int32(0), Int32(0), Int32(0), Float32(0.0), Float32(0.0), Float32(0.0), Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
    _COMPILED[key](*[t.reshape(-1) for t in tensors], B, L, num_segs, beta1, beta2, eps, num_ctas)
    return y, count, mean, var, cummean, cumrstd, step_count


# ======================================================================================================
# Backward
# ======================================================================================================
#
# With r_t the rstd, c_t the step count, db_t = sum(dy * gamma) and ds_t = sum(dy * gamma * (x - mean_t))
# over a (row, group), the gradient flowing into the decayed mean / variance follows the time-reversed
# recurrence u_{t-1} = a_t * (u_t + lu_t) with a_t in {beta, 0 (bos), 1 (padding)} and
#   lu_t = -r_t * db_t / (1 - beta1^c_t),   lv_t = -0.5 * r_t^3 * ds_t / (1 - beta2^c_t),
#   du_t = (1 - beta1) * (u_t + lu_t),      dx = dy * r * gamma + (du + 2 * dv * (x - m_t)) / Dg.
# It is the forward scan run on reversed chains (same scan elements, same look-back), so the kernel
# reuses the forward pipeline: tickets walk the segments of every chain from the last one.

class TimestepDecayNormBwd(TimestepDecayNormFwd):

    def __init__(self, dtype, w_dtype, s_dtype, hidden, num_groups, has_bos, has_pad,
                 groups_per_tile, tile_len, num_threads, prefetch, lag, min_ctas_per_sm=0, vec_mult=1, accumulate=False):
        super().__init__(dtype, w_dtype, s_dtype, hidden, num_groups, has_bos, has_pad,
                         groups_per_tile, tile_len, num_threads, prefetch, lag, min_ctas_per_sm, vec_mult)
        # accumulate: dx tiles leave through TMA reduce-add stores into dx (which holds a residual gradient)
        self.accumulate = accumulate
        # two resolve warps take alternate tiles (the look-back latency of one overlaps the other); the
        # aggregate -> resolve hand-off then needs an even number of buffers so that each resolve warp
        # owns its own barrier ids
        self.NRW = 2
        self.NA = 4
        self.NT = self.NC + 32 * (2 + self.NRW)
        assert BAR_BASE + 2 * self.NB + self.NA + self.S <= 16, "named barriers"
        assert self.TPR <= 32 and 32 % self.TPR == 0, "a row of a tile must fit in a warp"
        assert 2 * self.VEC % (32 // self.TPR) == 0, "reduce-scatter of the dgamma / dbeta partials"
        assert 32 // self.TPR >= 2, "the reduced dgamma / dbeta runs of a lane must not mix gamma and beta"
        self.GPL = max(1, self.W // 32)  # gamma elements per producer lane

    @cute.jit
    def __call__(
        self,
        tX: cute.Tensor,
        tDy: cute.Tensor,
        tBos: cute.Tensor,
        tPad: cute.Tensor,
        tMeanGrad: cute.Tensor,
        tVarGrad: cute.Tensor,
        tCumMean: cute.Tensor,
        tCumRstd: cute.Tensor,
        tStepCount: cute.Tensor,
        tGamma: cute.Tensor,
        tDx: cute.Tensor,
        tPrevMeanGrad: cute.Tensor,
        tPrevVarGrad: cute.Tensor,
        tGammaPart: cute.Tensor,
        tBetaPart: cute.Tensor,
        tZeroCount: cute.Tensor,
        tScratchCount: cute.Tensor,
        tWsInt: cute.Tensor,
        tWs64: cute.Tensor,
        B: Int32,
        L: Int32,
        num_segs: Int32,
        beta1: Float32,
        beta2: Float32,
        num_ctas: Int32,
        stream: cuda.CUstream,
    ):
        xy_layout = cute.make_layout((B, L, self.H), stride=(L * self.H, self.H, 1))
        tile_smem = cute.make_ordered_layout((1, self.T, self.W), order=(2, 1, 0))
        tile = (1, self.T, self.W)
        tma_x, mXt = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), cute.make_tensor(tX.iterator, xy_layout), tile_smem, tile)
        tma_dy, mDyt = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), cute.make_tensor(tDy.iterator, xy_layout), tile_smem, tile)
        dx_op = cpasync.CopyReduceBulkTensorTileS2GOp() if cutlass.const_expr(self.accumulate) else cpasync.CopyBulkTensorTileS2GOp()
        tma_dx, mDxt = cpasync.make_tiled_tma_atom(dx_op, cute.make_tensor(tDx.iterator, xy_layout), tile_smem, tile)
        self.kernel(
            tma_x, mXt, tma_dy, mDyt, tma_dx, mDxt,
            tBos.iterator, tPad.iterator, tMeanGrad.iterator, tVarGrad.iterator, tCumMean.iterator, tCumRstd.iterator,
            tStepCount.iterator, tGamma.iterator, tPrevMeanGrad.iterator, tPrevVarGrad.iterator,
            tGammaPart.iterator, tBetaPart.iterator, tZeroCount.iterator, tScratchCount.iterator,
            tWsInt.iterator, tWs64.iterator, B, L, num_segs, beta1, beta2,
        ).launch(grid=[num_ctas, 1, 1], block=[self.NT, 1, 1], stream=stream, min_blocks_per_mp=self.min_ctas_per_sm)

    @cute.kernel
    def kernel(
        self,
        tma_x: cute.CopyAtom,
        mXt: cute.Tensor,
        tma_dy: cute.CopyAtom,
        mDyt: cute.Tensor,
        tma_dx: cute.CopyAtom,
        mDxt: cute.Tensor,
        pBos: cute.Pointer,
        pPad: cute.Pointer,
        pMeanGrad: cute.Pointer,
        pVarGrad: cute.Pointer,
        pCumMean: cute.Pointer,
        pCumRstd: cute.Pointer,
        pStepCount: cute.Pointer,
        pGamma: cute.Pointer,
        pPrevMeanGrad: cute.Pointer,
        pPrevVarGrad: cute.Pointer,
        pGammaPart: cute.Pointer,
        pBetaPart: cute.Pointer,
        pZeroCount: cute.Pointer,
        pScratchCount: cute.Pointer,
        pWsInt: cute.Pointer,
        pWs64: cute.Pointer,
        B: Int32,
        L: Int32,
        num_segs: Int32,
        beta1: Float32,
        beta2: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()

        num_chains = B * self.num_group_tiles
        num_tiles = num_chains * num_segs

        mBos = cute.make_tensor(pBos, cute.make_layout((B, L), stride=(L, 1)))
        mPad = cute.make_tensor(pPad, cute.make_layout((B, L), stride=(L, 1)))
        # the look-back's initial state is (mean_grad, var_grad); its final inclusive state the grads of
        # (prev_mean, prev_var). Counts play no role backwards (zero in, scratch out).
        mZeroCount = cute.make_tensor(pZeroCount, cute.make_layout(B))
        mScratchCount = cute.make_tensor(pScratchCount, cute.make_layout(B))
        mMeanGrad = cute.make_tensor(pMeanGrad, cute.make_layout((B, self.G), stride=(self.G, 1)))
        mVarGrad = cute.make_tensor(pVarGrad, cute.make_layout((B, self.G), stride=(self.G, 1)))
        mPrevMeanGrad = cute.make_tensor(pPrevMeanGrad, cute.make_layout((B, self.G), stride=(self.G, 1)))
        mPrevVarGrad = cute.make_tensor(pPrevVarGrad, cute.make_layout((B, self.G), stride=(self.G, 1)))
        mCumMean = cute.make_tensor(pCumMean, cute.make_layout((B, self.G, L), stride=(self.G * L, L, 1)))
        mCumRstd = cute.make_tensor(pCumRstd, cute.make_layout((B, self.G, L), stride=(self.G * L, L, 1)))
        mStepCount = cute.make_tensor(pStepCount, cute.make_layout((B, L), stride=(L, 1)))
        # dgamma / dbeta accumulators [2, H] (fp32, zeroed with the workspace), added to tile by tile
        mGammaPart = cute.make_tensor(pGammaPart, cute.make_layout(self.H))
        mBetaPart = cute.make_tensor(pBetaPart + self.H, cute.make_layout(self.H))
        pTicket = pWsInt
        mAggW = cute.make_tensor(pWs64 + 2, cute.make_layout(num_tiles * self.Gt * 2))
        mIncW = cute.make_tensor(pWs64 + (2 + num_tiles * self.Gt * 2), cute.make_layout(num_tiles * self.Gt * 2))
        w_dyn = Int32(self.W)
        vec_dyn = Int32(self.VEC)

        smem = cutlass.memory.SmemAllocator()
        # S-stage ring of (x, dy) tiles; dx is written in place of dy and TMA-stored
        ring_layout = cute.make_layout((1, self.T, self.W, self.S), stride=(self.T * self.W, self.W, 1, self.T * self.W))
        sX = smem.allocate_tensor(self.dtype, ring_layout, byte_alignment=128)
        sDy = smem.allocate_tensor(self.dtype, ring_layout, byte_alignment=128)
        full_bar = smem.allocate_array(Int64, num_elems=self.S)
        sTicket = smem.allocate_tensor(Int32, cute.make_layout(self.S), byte_alignment=16)
        # per stage: (b, seg, gtile) of its tile, decoded once by the producer
        sTile = smem.allocate_tensor(Int32, cute.make_layout((self.S, 4), stride=(4, 1)), byte_alignment=16)
        # per stage, written by the producer with the tile: (step count << 2 | row code), the forward's
        # mean / rstd of each (row, group), and the gamma slice
        sMeta = smem.allocate_tensor(Int32, cute.make_layout((self.S, self.T), stride=(self.T, 1)), byte_alignment=16)
        st_layout = cute.make_layout((self.S, self.Gt, self.T), stride=(self.Gt * self.T, self.T, 1))
        sCm = smem.allocate_tensor(Float32, st_layout, byte_alignment=16)
        sR = smem.allocate_tensor(Float32, st_layout, byte_alignment=16)
        sGam = smem.allocate_tensor(self.w_dtype, cute.make_layout((self.S, self.W), stride=(self.W, 1)), byte_alignment=16)
        # per (row, group), NB buffers: the group mean m, and (lu, lv) -> (qu, qv) -> (du, dv)
        stat_layout = cute.make_layout((self.NB, self.Gt, self.T), stride=(self.Gt * self.T, self.T, 1))
        sM = smem.allocate_tensor(Float32, stat_layout, byte_alignment=16)
        sU = smem.allocate_tensor(Float32, stat_layout, byte_alignment=16)
        sV = smem.allocate_tensor(Float32, stat_layout, byte_alignment=16)
        sRow = smem.allocate_tensor(Int32, cute.make_layout((self.NA, self.T), stride=(self.T, 1)), byte_alignment=16)
        sAgg = smem.allocate_tensor(Float32, cute.make_layout((self.NA, self.Gt, 2), stride=(2 * self.Gt, 2, 1)), byte_alignment=16)
        sAggMaskI = smem.allocate_tensor(Int32, cute.make_layout((self.NA, 2), stride=(2, 1)), byte_alignment=8)
        sTk = smem.allocate_tensor(Int32, cute.make_layout(self.NA), byte_alignment=8)
        sTkInfo = smem.allocate_tensor(Int32, cute.make_layout((self.NA, 4), stride=(4, 1)), byte_alignment=16)
        sIn = smem.allocate_tensor(Float32, cute.make_layout((self.NRW, self.Gt, 2), stride=(2 * self.Gt, 2, 1)), byte_alignment=16)
        sInC = smem.allocate_tensor(Int32, cute.make_layout((self.NRW, 1), stride=(1, 1)), byte_alignment=8)

        log2_beta1 = cute.math.log2(beta1)
        log2_beta2 = cute.math.log2(beta2)
        inv_dg = Float32(1.0 / self.Dg)
        stages = Int32(self.S)
        nb = Int32(self.NB)
        na = Int32(self.NA)
        handoff = self.NC + 32

        if tidx < self.S:
            cute.arch.mbarrier_init(full_bar + tidx, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if warp < self.NCW:
            # ================================ compute warps ================================================
            cidx = tidx % self.TPR
            rsub = tidx // self.TPR
            gl = (cidx * self.VEC) // self.Dg
            tile_elems = Int32(self.T * self.W)
            rOld = cute.make_rmem_tensor((self.D,), Int32)
            rOld.fill(-1)
            it = Int32(0)
            running = Int32(1)
            while running != 0:
                stage = it % stages
                cute.arch.mbarrier_wait(full_bar + stage, (it // stages) % 2)
                cur = sTicket[stage]
                if cur >= num_tiles:
                    running = Int32(0)
                else:
                    buf = it % nb
                    self._row_sums(stage, sX, sDy, stage * tile_elems + cidx * vec_dyn, buf, sGam, cidx, rsub, gl, w_dyn,
                                   inv_dg, log2_beta1, log2_beta2, sMeta, sCm, sR, sM, sU, sV)
                    cute.arch.barrier_arrive(barrier_id=BAR_BASE + buf, number_of_threads=handoff)
                    old = rOld[0]
                    if old >= 0:
                        self._finish_bwd(sTile, it - self.D, sGam, sX, sDy, sM, sCm, sR, sU, sV, sRow, sIn,
                                         mGammaPart, mBetaPart, num_chains, num_segs, tidx, warp, lane,
                                         cidx, rsub, gl, stages, nb, na, tile_elems, w_dyn, vec_dyn, inv_dg,
                                         beta1, beta2, log2_beta1, log2_beta2)
                    for d in cutlass.range_constexpr(self.D - 1):
                        rOld[d] = rOld[d + 1]
                    rOld[self.D - 1] = cur
                    it = it + 1
            cute.arch.barrier_arrive(barrier_id=BAR_BASE + it % nb, number_of_threads=handoff)
            for d in cutlass.range_constexpr(self.D):
                old = rOld[d]
                if old >= 0:
                    self._finish_bwd(sTile, it - self.D + d, sGam, sX, sDy, sM, sCm, sR, sU, sV, sRow, sIn,
                                     mGammaPart, mBetaPart, num_chains, num_segs, tidx, warp, lane,
                                     cidx, rsub, gl, stages, nb, na, tile_elems, w_dyn, vec_dyn, inv_dg,
                                     beta1, beta2, log2_beta1, log2_beta2)
        elif warp == self.NCW:
            # ================================ aggregate warp ===============================================
            jt = Int32(0)
            running = Int32(1)
            while running != 0:
                cute.arch.barrier(barrier_id=BAR_BASE + jt % nb, number_of_threads=handoff)
                abuf = jt % na
                tk = sTicket[jt % stages]
                if lane == 0:
                    sTk[abuf] = tk
                if lane < 3:
                    sTkInfo[abuf, lane] = sTile[jt % stages, lane]
                if tk >= num_tiles:
                    running = Int32(0)
                    # also stop the other resolve warp, which waits for the following tile
                    nbuf = (jt + 1) % na
                    if lane == 0:
                        sTk[nbuf] = tk
                    cute.arch.barrier_arrive(barrier_id=BAR_BASE + self.NB + abuf, number_of_threads=64)
                    cute.arch.barrier_arrive(barrier_id=BAR_BASE + self.NB + nbuf, number_of_threads=64)
                else:
                    self._aggregate_bwd(tk, jt % nb, abuf, jt % stages, lane, mAggW, sMeta, sU, sV, sRow,
                                        sAgg, sAggMaskI, beta1, beta2)
                    cute.arch.barrier_arrive(barrier_id=BAR_BASE + self.NB + self.NA + jt % nb, number_of_threads=handoff + 32)
                    jt = jt + 1
        elif warp <= self.NCW + self.NRW:
            # ================================ resolve warps ================================================
            rw = warp - self.NCW - 1
            jt = rw
            running = Int32(1)
            while running != 0:
                abuf = jt % na
                cute.arch.barrier(barrier_id=BAR_BASE + self.NB + abuf, number_of_threads=64)
                tk = sTk[abuf]
                if tk >= num_tiles:
                    running = Int32(0)
                else:
                    self._resolve_bwd(tk, sTkInfo, jt % nb, abuf, num_chains, num_segs, lane,
                                      mAggW, mIncW, mZeroCount, mMeanGrad, mVarGrad, mScratchCount, mPrevMeanGrad, mPrevVarGrad,
                                      sAgg, sAggMaskI, sIn[rw, None, None], sInC[rw, None], log2_beta1, log2_beta2)
                    cute.arch.barrier_arrive(barrier_id=BAR_BASE + self.NB + self.NA + jt % nb, number_of_threads=handoff + 32)
                    jt = jt + self.NRW
        else:
            # ================================ producer warp ================================================
            gX = cute.local_tile(mXt, (1, self.T, self.W), (None, None, None))
            gDy = cute.local_tile(mDyt, (1, self.T, self.W), (None, None, None))
            gDx = cute.local_tile(mDxt, (1, self.T, self.W), (None, None, None))
            tsX, tgX = cpasync.tma_partition(tma_x, 0, cute.make_layout(1), cute.group_modes(sX, 0, 3), cute.group_modes(gX, 0, 3))
            tsDy, tgDy = cpasync.tma_partition(tma_dy, 0, cute.make_layout(1), cute.group_modes(sDy, 0, 3), cute.group_modes(gDy, 0, 3))
            tsDx, tgDx = cpasync.tma_partition(tma_dx, 0, cute.make_layout(1), cute.group_modes(sDy, 0, 3), cute.group_modes(gDx, 0, 3))
            base = Int32(0)
            if lane == 0:
                base = cute.arch.atomic_add(pTicket, Int32(2), sem="relaxed", scope="gpu")
            my_ticket = cute.arch.shuffle_sync(base, 0) + lane
            regs = [self._meta_regs() for _ in range(2)]
            self._load_meta(self._rev(cute.arch.shuffle_sync(my_ticket, 0), num_chains, num_segs), regs[0],
                            mBos, mPad, mCumMean, mCumRstd, mStepCount, pGamma, num_chains, num_tiles, L, lane, w_dyn)
            jt = Int32(0)
            running = Int32(1)
            while running != 0:
                for hh in cutlass.range_constexpr(2):
                    if running != 0:
                        stage = jt % stages
                        tk = cute.arch.shuffle_sync(my_ticket, hh)
                        tk_next = cute.arch.shuffle_sync(my_ticket, 1 - hh)
                        if tk < num_tiles:
                            if lane == hh:
                                my_ticket = cute.arch.atomic_add(pTicket, Int32(1), sem="relaxed", scope="gpu")
                        self._load_meta(self._rev(tk_next, num_chains, num_segs), regs[1 - hh],
                                        mBos, mPad, mCumMean, mCumRstd, mStepCount, pGamma, num_chains, num_tiles, L, lane, w_dyn)
                        if jt >= stages:
                            cute.arch.barrier(barrier_id=BAR_BASE + 2 * self.NB + self.NA + stage, number_of_threads=handoff)
                            self._store_y(sTile, stage, tma_dx, tsDx, tgDx)
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                        ft = self._rev(tk, num_chains, num_segs)
                        seg = ft // num_chains
                        chain = ft - seg * num_chains
                        b = chain // self.num_group_tiles
                        gtile = chain - b * self.num_group_tiles
                        if lane == 0:
                            sTicket[stage] = tk
                            sTile[stage, 0] = b
                            sTile[stage, 1] = seg
                            sTile[stage, 2] = gtile
                        self._store_meta(ft, regs[hh], sMeta, sCm, sR, sGam, stage, num_chains, L, lane)
                        cute.arch.sync_warp()
                        if tk < num_tiles:
                            if lane == 0:
                                cute.arch.mbarrier_arrive_and_expect_tx(full_bar + stage, 2 * self.T * self.W * self.dtype.width // 8)
                            cute.copy(tma_x, tgX[None, b, seg, gtile], tsX[None, stage], tma_bar_ptr=full_bar + stage)
                            cute.copy(tma_dy, tgDy[None, b, seg, gtile], tsDy[None, stage], tma_bar_ptr=full_bar + stage)
                        else:
                            if lane == 0:
                                cute.arch.mbarrier_arrive(full_bar + stage)
                            running = Int32(0)
                        jt = jt + 1
            for d in cutlass.range_constexpr(self.S - 1):
                m = jt - self.S + d
                if m >= 0:
                    mst = m % stages
                    cute.arch.barrier(barrier_id=BAR_BASE + 2 * self.NB + self.NA + mst, number_of_threads=handoff)
                    self._store_y(sTile, mst, tma_dx, tsDx, tgDx)
            cute.arch.cp_async_bulk_wait_group(0)

    def _meta_regs(self):
        """Producer registers for one tile's metadata (raw masks, statistics, counts, gamma)."""
        return (cute.make_rmem_tensor((self.LPR,), Uint8), cute.make_rmem_tensor((self.LPR,), Uint8),
                cute.make_rmem_tensor((self.Gt, self.LPR), Float32), cute.make_rmem_tensor((self.Gt, self.LPR), Float32),
                cute.make_rmem_tensor((self.LPR,), Int32), cute.make_rmem_tensor((self.GPL,), self.w_dtype))

    @cute.jit
    def _load_meta(self, ft, regs, mBos, mPad, mCumMean, mCumRstd, mStepCount, pGamma, num_chains, num_tiles, L, lane, w_dyn):
        """Producer warp: issue the loads of tile ``ft``'s metadata into registers (consumed one tile later)."""
        rBos, rPad, rCm, rR, rCnt, rGam = regs
        if ft < num_tiles:
            seg = ft // num_chains
            chain = ft - seg * num_chains
            b = chain // self.num_group_tiles
            gtile = chain - b * self.num_group_tiles
            t0 = seg * self.T + lane * self.LPR
            for jj in cutlass.range_constexpr(self.LPR):
                t = t0 + jj
                tt = cutlass.select_(t < L, t, Int32(0))
                if const_expr(self.has_bos):
                    rBos[jj] = mBos[b, tt]
                if const_expr(self.has_pad):
                    rPad[jj] = mPad[b, tt]
                rCnt[jj] = mStepCount[b, tt]
                for g in cutlass.range_constexpr(self.Gt):
                    rCm[g, jj] = mCumMean[b, gtile * self.Gt + g, tt]
                    rR[g, jj] = mCumRstd[b, gtile * self.Gt + g, tt]
            if lane * self.GPL < self.W:
                rGam.store(cute.make_tensor(pGamma + (gtile * w_dyn + lane * self.GPL), cute.make_layout(self.GPL)).load())

    @cute.jit
    def _store_meta(self, ft, regs, sMeta, sCm, sR, sGam, stage, num_chains, L, lane):
        """Producer warp: write tile ``ft``'s metadata into its stage (rows past L: count and code 0)."""
        rBos, rPad, rCm, rR, rCnt, rGam = regs
        seg = ft // num_chains
        t0 = seg * self.T + lane * self.LPR
        meta = cute.make_rmem_tensor((self.LPR,), Int32)
        for jj in cutlass.range_constexpr(self.LPR):
            valid = t0 + jj < L
            live = cutlass.select_(valid, Int32(1), Int32(0))
            if const_expr(self.has_pad):
                live = live & (1 - Int32(rPad[jj]))
            code = live
            if const_expr(self.has_bos):
                code = live | ((live & Int32(rBos[jj])) << 1)
            meta[jj] = (cutlass.select_(valid, rCnt[jj], Int32(0)) << 2) | code
        cute.local_tile(sMeta[stage, None], (self.LPR,), (lane,)).store(meta.load())
        for g in cutlass.range_constexpr(self.Gt):
            cute.local_tile(sCm[stage, g, None], (self.LPR,), (lane,)).store(rCm[g, None].load())
            cute.local_tile(sR[stage, g, None], (self.LPR,), (lane,)).store(rR[g, None].load())
        if lane * self.GPL < self.W:
            cute.local_tile(sGam[stage, None], (self.GPL,), (lane,)).store(rGam.load())

    def _rev(self, tk, num_chains, num_segs):
        """Ticket -> forward tile index: tickets walk the segments of every chain from the last one.
        (Invalid tickets map to invalid tiles.)"""
        rseg = tk // num_chains
        return (num_segs - 1 - rseg) * num_chains + (tk - rseg * num_chains) + cutlass.select_(
            rseg >= num_segs, rseg * num_chains * 2, Int32(0))

    @cute.jit
    def _row_sums(self, stage, sX, sDy, xs_base, buf, sGam, cidx, rsub, gl, w_dyn, inv_dg,
                  log2_beta1, log2_beta2, sMeta, sCm, sR, sM, sU, sV):
        """Compute warps: per (row, group) the mean of x, db = sum(dy * gamma), ds = sum(dy * gamma * (x - cm)),
        and the local gradients (lu, lv) of the decayed mean / variance."""
        npair = self.VEC // 2
        rows = [rsub + i * self.R for i in range(self.RPT)]
        w = cute.local_tile(sGam[stage, None], (self.VEC,), (cidx,)).load().to(Float32)
        xs = [self._load_row_f32((sX.iterator + (xs_base + row * w_dyn)).align(16)) for row in rows]
        dys = [self._load_row_f32((sDy.iterator + (xs_base + row * w_dyn)).align(16)) for row in rows]
        sx = []
        sg = []
        sgx = []
        for i in cutlass.range_constexpr(self.RPT):
            ax = (Float32(0.0), Float32(0.0))
            ag = (Float32(0.0), Float32(0.0))
            agx = (Float32(0.0), Float32(0.0))
            for k in cutlass.range_constexpr(npair):
                xp = (xs[i][2 * k], xs[i][2 * k + 1])
                gp = _mul2((dys[i][2 * k], dys[i][2 * k + 1]), (w[2 * k], w[2 * k + 1]))
                ax = _add2(ax, xp)
                ag = _add2(ag, gp)
                agx = _fma2(gp, xp, agx)
            sx.append(ax[0] + ax[1])
            sg.append(ag[0] + ag[1])
            sgx.append(agx[0] + agx[1])
        # reduce-scatter: lane li of the group ends up with the totals of its own rows, so the scalar work
        # below runs once per row instead of once per lane
        mine, block, writer = self._scatter_rows([(sx[i], sg[i], sgx[i]) for i in range(self.RPT)], cidx % self.TPG)
        for q in cutlass.range_constexpr(len(mine)):
            mine_x, mine_g, mine_gx = mine[q]
            if writer:
                row = rsub + (block * len(mine) + q) * self.R
                cm = sCm[stage, gl, row]
                rs = sR[stage, gl, row]
                cnt = sMeta[stage, row] >> 2
                ds = mine_gx - cm * mine_g
                cf = Int32(cnt).to(Float32)
                inv1 = cute.arch.rcp_approx(1.0 - cute.math.exp2(cf * log2_beta1, fastmath=True))
                inv2 = cute.arch.rcp_approx(1.0 - cute.math.exp2(cf * log2_beta2, fastmath=True))
                sM[buf, gl, row] = mine_x * inv_dg
                sU[buf, gl, row] = cutlass.select_(cnt > 0, -rs * mine_g * inv1, Float32(0.0))
                sV[buf, gl, row] = cutlass.select_(cnt > 0, -0.5 * rs * rs * rs * ds * inv2, Float32(0.0))

    @cute.jit
    def _aggregate_bwd(self, tk, h, a, stage, lane, mAggW, sMeta, sU, sV, sRow, sAgg, sAggMaskI, beta1, beta2):
        """Aggregate warp: time-reversed local scan of a tile. Publishes the tile aggregate and hands the tile to
        its resolve warp first (both are on the look-back's critical path); then per row stores
        q = (exclusive local prefix) + (local gradient) in place of the local gradient, and the exclusive
        prefix's (k, reset) and the row's live bit."""
        identity = (Int32(0), Int32(0), Float32(1.0), Float32(1.0), Float32(0.0), Float32(0.0))
        meta = cute.local_tile(sMeta[stage, None], (self.LPR,), (lane,)).load()
        codes = [meta[jj] & 3 for jj in range(self.LPR)]
        lus, lvs, elemss, accs = [], [], [], []
        for g in cutlass.range_constexpr(self.Gt):
            lu = cute.local_tile(sU[h, g, None], (self.LPR,), (lane,)).load()
            lv = cute.local_tile(sV[h, g, None], (self.LPR,), (lane,)).load()
            elems = []
            for jj in cutlass.range_constexpr(self.LPR):
                live = codes[jj] & 1
                reset = (codes[jj] >> 1) & 1
                is_live = live != 0
                p1 = cutlass.select_(is_live, cutlass.select_(reset != 0, Float32(0.0), beta1), Float32(1.0))
                p2 = cutlass.select_(is_live, cutlass.select_(reset != 0, Float32(0.0), beta2), Float32(1.0))
                elems.append((live, reset & live, p1, p2,
                              cutlass.select_(is_live, p1 * lu[jj], Float32(0.0)),
                              cutlass.select_(is_live, p2 * lv[jj], Float32(0.0))))
            # application order runs backwards in time: last row of the lane first, higher lanes first
            acc = identity
            for jj in cutlass.range_constexpr(self.LPR):
                acc = _compose(acc, elems[self.LPR - 1 - jj])
            for o in cutlass.range_constexpr(5):
                other = _shuffle_elem(acc, 1 << o, up=False)
                acc = _select_elem(lane + (1 << o) < 32, _compose(other, acc), acc)
            if lane == 0:
                hdr = _agg_header(acc[0], acc[1])
                cute.arch.store((mAggW.iterator + (tk * self.Gt + g) * 2).llvm_ptr, _pack(hdr, acc[4]), sem="relaxed", scope="gpu")
                cute.arch.store((mAggW.iterator + (tk * self.Gt + g) * 2 + 1).llvm_ptr, _pack(hdr, acc[5]), sem="relaxed", scope="gpu")
                sAgg[a, g, 0] = acc[4]
                sAgg[a, g, 1] = acc[5]
                if g == 0:
                    sAggMaskI[a, 0] = acc[0]
                    sAggMaskI[a, 1] = acc[1]
            lus.append(lu)
            lvs.append(lv)
            elemss.append(elems)
            accs.append(acc)
        cute.arch.barrier_arrive(barrier_id=BAR_BASE + self.NB + a, number_of_threads=64)
        rq = cute.make_rmem_tensor((self.LPR,), Float32)
        rqv = cute.make_rmem_tensor((self.LPR,), Float32)
        rrow = cute.make_rmem_tensor((self.LPR,), Int32)
        for g in cutlass.range_constexpr(self.Gt):
            lu, lv, elems = lus[g], lvs[g], elemss[g]
            state = _select_elem(lane < 31, _shuffle_elem(accs[g], 1, up=False), identity)
            for jj in cutlass.range_constexpr(self.LPR):
                r = self.LPR - 1 - jj
                rq[r] = state[4] + lu[r]
                rqv[r] = state[5] + lv[r]
                rrow[r] = (codes[r] & 1) | (state[1] << 1) | (state[0] << 2)
                state = _compose(state, elems[r])
            cute.local_tile(sU[h, g, None], (self.LPR,), (lane,)).store(rq.load())
            cute.local_tile(sV[h, g, None], (self.LPR,), (lane,)).store(rqv.load())
            if g == 0:
                cute.local_tile(sRow[a, None], (self.LPR,), (lane,)).store(rrow.load())

    @cute.jit
    def _resolve_bwd(self, tk, sTkInfo, h, a, num_chains, num_segs, lane,
                     mAggW, mIncW, mZeroCount, mMeanGrad, mVarGrad, mScratchCount, mPrevMeanGrad, mPrevVarGrad,
                     sAgg, sAggMaskI, sIn, sInC, log2_beta1, log2_beta2):
        """Resolve warp: look-back along the reversed chain; leaves the tile's entry state (U, V) in sIn.
        (The compute warps turn it into du = (1 - beta1) * (p * U + qu) per row.)"""
        b = sTkInfo[a, 0]
        rseg = num_segs - 1 - sTkInfo[a, 1]
        gtile = sTkInfo[a, 2]
        chain = b * self.num_group_tiles + gtile
        g0 = gtile * self.Gt
        self._lookback(tk, rseg, chain, b, g0, gtile, num_chains, num_segs, lane,
                       mAggW, mIncW, mZeroCount, mMeanGrad, mVarGrad, mScratchCount, mPrevMeanGrad, mPrevVarGrad,
                       sAgg[a, None, None], sAggMaskI[a, None], sIn, sInC, log2_beta1, log2_beta2)
        cute.arch.sync_warp()

    @cute.jit
    def _finish_bwd(self, sTile, j, sGam, sX, sDy, sM, sCm, sR, sU, sV, sRow, sIn,
                    mGammaPart, mBetaPart, num_chains, num_segs, tidx, warp, lane, cidx, rsub, gl,
                    stages, nb, na, tile_elems, w_dyn, vec_dyn, inv_dg, beta1, beta2, log2_beta1, log2_beta2):
        """Compute warps: dx of tile ``tk`` (local index ``j``) in place of dy, released to the producer for
        the TMA store, and the tile's dgamma / dbeta partials."""
        stage = j % stages
        gtile = sTile[stage, 2]
        h = j % nb
        cute.arch.barrier(barrier_id=BAR_BASE + self.NB + self.NA + h, number_of_threads=self.NC + 64)
        w = cute.local_tile(sGam[stage, None], (self.VEC,), (cidx,)).load().to(Float32)
        npair = self.VEC // 2
        xs_base = stage * tile_elems + cidx * vec_dyn
        rows = [rsub + i * self.R for i in range(self.RPT)]
        dyt = [cute.make_tensor((sDy.iterator + (xs_base + row * w_dyn)).align(16), cute.make_layout(self.VEC)) for row in rows]
        xs = [self._load_row_f32((sX.iterator + (xs_base + row * w_dyn)).align(16)) for row in rows]
        dys = [self._load_row_f32(t.iterator) for t in dyt]
        accg = [(Float32(0.0), Float32(0.0)) for _ in range(npair)]
        accb = [(Float32(0.0), Float32(0.0)) for _ in range(npair)]
        # per-row coefficients: lane li of each group computes those of row p * TPG + li, the others fetch them
        abuf = j % na
        li = cidx % self.TPG
        group_lane0 = lane - li
        coefs = []
        for p in cutlass.range_constexpr((self.RPT + self.TPG - 1) // self.TPG):
            lo, hi = p * self.TPG, min((p + 1) * self.TPG, self.RPT)
            row_l = rsub + (lo + cutlass.min(li, hi - lo - 1)) * self.R
            # the row's gradient of the decayed statistics from the tile's entry state and the row's local
            # prefix: du = (1 - beta1) * (beta1^k * U + qu), with k / reset of the prefix (live rows only)
            info = sRow[abuf, row_l]
            live_l = Int32(info & 1).to(Float32)
            kf = Int32(info >> 2).to(Float32)
            keep = ((info >> 1) & 1) == 0
            dec1 = cutlass.select_(keep, cute.math.exp2(kf * log2_beta1, fastmath=True), Float32(0.0))
            dec2 = cutlass.select_(keep, cute.math.exp2(kf * log2_beta2, fastmath=True), Float32(0.0))
            du = (1.0 - beta1) * (dec1 * sIn[j % self.NRW, gl, 0] + sU[h, gl, row_l]) * live_l
            dv = (1.0 - beta2) * (dec2 * sIn[j % self.NRW, gl, 1] + sV[h, gl, row_l]) * live_l
            rs_l = sR[stage, gl, row_l] * live_l
            # dx = dy * rstd * gamma + (du + 2 dv (x - m)) / Dg = dy * (gamma * rstd) + (x * a + c)
            # (zero on padded rows: du, dv and rs are 0 there); dgamma += dy * (x - cm) * rstd, dbeta += dy
            a_l = 2.0 * dv * inv_dg
            c_l = du * inv_dg - sM[h, gl, row_l] * a_l
            ncr_l = -sCm[stage, gl, row_l] * rs_l
            for i in cutlass.range_constexpr(lo, hi):
                src = group_lane0 + (i - lo)
                coefs.append(tuple(cute.arch.shuffle_sync(v, src) for v in (live_l, rs_l, a_l, c_l, ncr_l)))
        for i in cutlass.range_constexpr(self.RPT):
            live, rs, a, c, ncr = coefs[i]
            dx = cute.make_rmem_tensor((self.VEC,), Float32)
            for k in cutlass.range_constexpr(npair):
                xp = (xs[i][2 * k], xs[i][2 * k + 1])
                dyp = (dys[i][2 * k], dys[i][2 * k + 1])
                wrs = _mul2((w[2 * k], w[2 * k + 1]), (rs, rs))
                t = _fma2(xp, (a, a), (c, c))
                dx[2 * k], dx[2 * k + 1] = _fma2(dyp, wrs, t)
                xn = _fma2(xp, (rs, rs), (ncr, ncr))
                accg[k] = _fma2(dyp, xn, accg[k])
                accb[k] = _fma2(dyp, (live, live), accb[k])
            dyt[i].store(dx.load().to(self.dtype))
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier_arrive(barrier_id=BAR_BASE + 2 * self.NB + self.NA + stage, number_of_threads=self.NC + 32)
        # dgamma / dbeta of the tile: the row slots of a warp by a reduce-scatter (each shuffle level keeps
        # half of the columns), after which every lane holds a run of consecutive columns that it adds to
        # the global fp32 accumulators with vector reductions (no shared memory, no CTA-wide sync)
        vals = [v for pr in accg for v in pr] + [v for pr in accb for v in pr]  # gamma columns, then beta
        base = Int32(0)
        for o in cutlass.range_constexpr(int(math.log2(32 // self.TPR))):
            off = self.TPR << o
            upper = (lane & off) != 0
            half = len(vals) // 2
            vals = [cutlass.select_(upper, vals[i + half], vals[i])
                    + cute.arch.shuffle_sync_bfly(cutlass.select_(upper, vals[i], vals[i + half]), offset=off)
                    for i in range(half)]
            base = base + cutlass.select_(upper, Int32(half), Int32(0))
        chunk = math.gcd(len(vals), 4)
        racc = cute.make_rmem_tensor((chunk,), Float32)
        col0 = gtile * self.W + cidx * self.VEC
        for c0 in cutlass.range_constexpr(0, len(vals), chunk):
            for jj in cutlass.range_constexpr(chunk):
                racc[jj] = vals[c0 + jj]
            e = base + c0  # a chunk never straddles the gamma / beta boundary
            dst = cutlass.select_(e < self.VEC, mGammaPart.iterator + (col0 + e), mBetaPart.iterator + (col0 + e - self.VEC))
            cute.arch.red(dst.llvm_ptr, racc.load(), op="add", dtype=Float32, sem="relaxed", scope="gpu")

_COMPILED_BWD = {}
_ZEROS = {}


def _zeros_int64(device: torch.device, n: int) -> torch.Tensor:
    """A cached read-only zero vector."""
    z = _ZEROS.get(device)
    if z is None or z.numel() < n:
        z = _ZEROS[device] = torch.zeros(max(n, 64), dtype=torch.int64, device=device)
    return z[:n]


def step_counts(bos_mask: Optional[torch.Tensor], padding_mask: Optional[torch.Tensor], prev_count: torch.Tensor,
                L: int) -> torch.Tensor:
    """The forward pass's per-step count [B, L] (int32): resets to 1 on bos, unchanged on padding."""
    B = prev_count.shape[0]
    device = prev_count.device
    live = torch.ones(B, L, dtype=torch.bool, device=device) if padding_mask is None else ~padding_mask
    n = torch.cumsum(live, dim=1, dtype=torch.int64)  # live rows up to and including t
    if bos_mask is None:
        return (prev_count.view(B, 1) + n).to(torch.int32)
    positions = torch.arange(L, device=device).expand(B, L)
    last_reset = torch.cummax(torch.where(bos_mask & live, positions, -1), dim=1).values
    n_before = torch.gather(n, 1, last_reset.clamp_min(0)) - 1  # live rows before the last reset
    count = torch.where(last_reset >= 0, n - n_before, prev_count.view(B, 1) + n)
    return count.to(torch.int32)


@lru_cache(maxsize=None)
def default_bwd_config(dtype: torch.dtype, H: int, G: int) -> Tuple[int, int, int, int, int, int, int]:
    """(groups_per_tile, tile_len, compute_threads, ctas_per_sm, prefetch, lag, vec_mult).

    One CTA per SM with the longest tile that fits in shared memory: the per-tile look-back latency is
    what limits the backward, so fewer, larger tiles beat more resident CTAs."""
    esize = dtype.itemsize
    Dg = H // G
    gt = 1
    while gt * Dg * esize < 64 and G % (gt * 2) == 0 and gt < 4:
        gt *= 2
    prefetch, lag, nc = 1, 1, 256
    vec = math.gcd(Dg, 128 // max(esize * 8, 16))
    rows_per_pass = nc // (gt * Dg // vec)
    step = math.lcm(32, rows_per_pass)
    tile_len = step
    for t in range(256 // step * step, step - 1, -step):
        if _bwd_smem_bytes(esize, Dg, gt, t, prefetch, lag, nc) + 1024 <= SMEM_PER_SM:
            tile_len = t
            break
    return gt, tile_len, nc, 1, prefetch, lag, 1


def _bwd_smem_bytes(esize: int, Dg: int, gt: int, tile_len: int, prefetch: int, lag: int, nc: int) -> int:
    """Shared memory of one backward CTA (mirrors the allocations in the kernel, plus alignment slack)."""
    T, W = tile_len, gt * Dg
    S, NB, NA = lag + 1 + prefetch, lag + 2, 4
    rings = 2 * S * T * W * esize + S * (T * 4 + 2 * gt * T * 4 + W * esize + 16 + 8 + 16)
    stats = 3 * NB * gt * T * 4
    handoff = NA * (T * 4 + gt * 8 + 8 + 4)
    reduce = 0
    return rings + stats + handoff + reduce + gt * 8 + 4 + 1024


def timestep_decay_norm_bwd(
    y_grad: torch.Tensor,
    mean_grad: Optional[torch.Tensor],
    var_grad: Optional[torch.Tensor],
    x: torch.Tensor,
    bos_mask: Optional[torch.Tensor],
    cummean: torch.Tensor,
    cumrstd: torch.Tensor,
    step_count: Optional[torch.Tensor],
    gamma: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
    num_groups: int,
    beta1: float,
    beta2: float,
    config: Optional[Tuple[int, int, int, int, int, int, int]] = None,
    prev_count: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    addend: Optional[torch.Tensor] = None,
):
    """Returns (x_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad).

    ``step_count`` is the per-step count returned by the forward pass; when it is not at hand it is
    recomputed from ``prev_count`` and the masks. With ``residual`` (contiguous, x's shape and dtype) the dx tiles
    are reduce-added into it by the TMA stores and x_grad is ``residual`` itself (overwritten with residual + dx).
    ``addend`` (B200 API): the same without modifying the given tensor (x_grad = addend + dx in a new tensor)."""
    if addend is not None:
        assert residual is None and addend.shape == x.shape and addend.dtype == x.dtype
        residual = addend.contiguous().clone()
    B, L, H = x.shape
    if step_count is None:
        step_count = step_counts(bos_mask, padding_mask, prev_count, L)
    G = num_groups
    x = x.contiguous()
    y_grad = y_grad.contiguous()
    gamma = gamma.contiguous()
    if mean_grad is None:
        mean_grad = torch.zeros(B, G, dtype=x.dtype, device=x.device)
    if var_grad is None:
        var_grad = torch.zeros(B, G, dtype=x.dtype, device=x.device)
    mean_grad = mean_grad.contiguous()
    var_grad = var_grad.contiguous()
    gt, tile_len, nt, ctas_per_sm, prefetch, lag, vec_mult = config if config is not None else default_bwd_config(x.dtype, H, G)
    num_segs = (L + tile_len - 1) // tile_len
    num_tiles = B * (G // gt) * num_segs
    num_ctas = min(num_tiles, _sm_count(x.device.index) * ctas_per_sm)

    accumulate = residual is not None
    if accumulate:
        assert residual.shape == x.shape and residual.dtype == x.dtype and residual.is_contiguous()
    x_grad = residual if accumulate else torch.empty_like(x)
    prev_mean_grad = torch.empty_like(mean_grad)
    prev_var_grad = torch.empty_like(var_grad)
    zero_count = _zeros_int64(x.device, B)
    scratch_count = torch.empty(B, dtype=torch.int64, device=x.device)
    # [ticket counter | aggregate words | inclusive words | dgamma, dbeta accumulators (2H fp32)], all zero
    ws_words = 2 + num_tiles * gt * 4
    ws = torch.zeros(ws_words + H, dtype=torch.int64, device=x.device)
    affine_grad = ws[ws_words:].view(torch.float32)

    has_bos = bos_mask is not None
    has_pad = padding_mask is not None
    key = (x.dtype, gamma.dtype, mean_grad.dtype, H, G, has_bos, has_pad, gt, tile_len, nt, prefetch, lag, ctas_per_sm, vec_mult,
           accumulate)
    dummy = ws.view(torch.uint8)
    tensors = [
        x, y_grad, bos_mask.view(torch.uint8) if has_bos else dummy, padding_mask.view(torch.uint8) if has_pad else dummy,
        mean_grad, var_grad, cummean, cumrstd, step_count, gamma, x_grad, prev_mean_grad, prev_var_grad,
        affine_grad, affine_grad, zero_count, scratch_count, ws.view(torch.int32), ws,
    ]
    if key not in _COMPILED_BWD:
        aligns = [16, 16, 1, 1, 2, 2, 4, 4, 4, 16, 16, 2, 2, 16, 16, 8, 8, 16, 16]
        dtypes = {torch.uint8: Uint8, torch.int64: Int64, torch.int32: Int32}
        fakes = [_fake(TORCH2CUTE.get(t.dtype, dtypes.get(t.dtype)), al) for t, al in zip(tensors, aligns)]
        op = TimestepDecayNormBwd(
            TORCH2CUTE[x.dtype], TORCH2CUTE[gamma.dtype], TORCH2CUTE[mean_grad.dtype],
            H, G, has_bos, has_pad, gt, tile_len, nt, prefetch, lag, ctas_per_sm, vec_mult, accumulate,
        )
        _COMPILED_BWD[key] = cute.compile(
            op, *fakes, Int32(0), Int32(0), Int32(0), Float32(0.0), Float32(0.0), Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
    _COMPILED_BWD[key](*[t.reshape(-1) for t in tensors], B, L, num_segs, beta1, beta2, num_ctas)
    gamma_grad, beta_grad = affine_grad.view(2, H).to(gamma.dtype).unbind(0)
    return x_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad
