// TMA backward of the group RMS norm (from x or from y) and the group layer norm (from x) for bf16 affine calls of at
// most 8192 channels: a persistent, warp-specialized pipeline. dispatch.cu's backward() reaches it through plan() and
// backward().
//
// One persistent CTA an SM (grid = min(SMs, tiles)); a tile is TR whole rows, and CTA b takes tiles b, b + grid, ...
// Warp 0 is the producer: one lane streams each tile into a ring of S shared-memory stages with 1D bulk copies (x_or_y
// and dy are one contiguous byte range each, then rstd and, for the layer norm, mean), completion counted on the
// stage's `full` mbarrier; consumers release a stage on its `empty` mbarrier. Warps 1..NW are consumers, and each owns
// fixed columns for the whole kernel, so gamma and its dgamma/dbeta partials stay in registers:
//   narrow (V = cpg / 8 <= 32 vectors of 16 bytes a group): a warp-chunk is GW = 32 / V whole groups, one vector a
//     lane; warp (c, wr) owns warp-chunks c + q * NWc (q < Q) and rows wr, wr + NWr, ... of each tile; a group's sums
//     are segmented shuffles (any V);
//   wide (V > 32): warp c owns group c, Q = ceil(V / 32) vectors a lane, full-warp sums;
//   split (gn_bwd_tma_split_kernel): a group over NWg warps, its partial sums exchanged through shared memory.
// dx goes out with streaming stores. dgamma/dbeta: each warp-row writes its partials to its own shared slot, the slots
// are added in warp-row order and written to a [grid, nred * C] fp32 workspace; after a grid barrier (cooperative
// launch) every CTA sums a slice of the columns over all CTA partials in CTA order. The tile map and every sum have a
// fixed order, so the result is the same bits run to run.
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <torch/extension.h>

#include "bwd_tma.h"
#include "tma.cuh"

namespace xllm::ops::blackwell::tma {
namespace {

namespace cg = cooperative_groups;

struct BwdArgs {
  const uint8_t* x;    // x or y [R, C]
  const uint8_t* dy;   // [R, C]
  const float* mean;   // [R, G] (layer norm)
  const float* rstd;   // [R, G]
  const uint4* gamma;  // [C]
  uint4* dx;           // [R, C]
  float* part;         // [grid, nred * C] CTA partials
  void* dgamma;
  void* dbeta;
  long long R;
  // V vectors a group; GW groups a warp-chunk and U warp-chunks a row (narrow); Q vectors or warp-chunks a lane; NWc
  // consumer warps across the columns, NWr warp-rows, NWg warps a group (split).
  int C, G, V, GW, U, TR, S, NWc, NWr, NWg, num_tiles;
  int stage_bytes, off_dy, off_rs, off_mu;
  float inv_cpg;
};

// The producer ends the ring with tile -1 in the next stage: a plain arrive with no bytes completes the phase the
// consumers wait on.
__device__ __forceinline__ void end_of_tiles(int* tidx, uint64_t* full, int s) {
  tidx[s] = -1;
  mbar_arrive(&full[s]);
}

__device__ __forceinline__ int stage_tile(const int* tidx, int s) {
  return reinterpret_cast<const volatile int*>(tidx)[s];
}

// Shared memory: a control block at fixed offsets (full[8], empty[8], the tile of each stage[8]), then the ring of
// S <= kMaxStages stages, then the split kernel's row partials.
constexpr int kMaxStages = 8;
constexpr int kCtlBytes = 256;
static_assert(kMaxStages * (2 * 8 + 4) <= kCtlBytes, "control block");

template <typename T>
__device__ __forceinline__ void store_param(void* p, int i, float v);
template <>
__device__ __forceinline__ void store_param<__nv_bfloat16>(void* p, int i, float v) {
  reinterpret_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(v);
}

// Sum over the group's lanes: a compile-time power-of-two width VK (xor butterfly), or VK = 0 for a run-time width.
template <int VK>
__device__ __forceinline__ float gsum(float v, int V, int gl, int head, bool pow2) {
  if constexpr (VK > 0) {
#pragma unroll
    for (int o = 1; o < VK; o <<= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
  } else {
    return seg_sum(v, V, gl, head, pow2);
  }
}
template <int VK>
__device__ __forceinline__ float2 gsum2(float2 v, int V, int gl, int head, bool pow2) {
  if constexpr (VK > 0) {
#pragma unroll
    for (int o = 1; o < VK; o <<= 1) {
      v.x += __shfl_xor_sync(0xffffffffu, v.x, o);
      v.y += __shfl_xor_sync(0xffffffffu, v.y, o);
    }
    return v;
  } else {
    return seg_sum2(v, V, gl, head, pow2);
  }
}

// gamma of one owned vector in fp32, and its reciprocal for the reconstructed input.
template <bool MEMEFF>
struct ChunkConst {
  float g[8];
  float rg[MEMEFF ? 8 : 1];
};

// One vector of one row: x_hat, gamma * dy and dy in fp32. x_hat is x * rs (RMS), x * rs - mu * rs (layer norm) or
// y * (1 / gamma) (RMS from y).
template <typename T, bool LAYER, bool MEMEFF>
__device__ __forceinline__ void chunk_vals(const uint4& xv, const uint4& dv, const ChunkConst<MEMEFF>& k, float rs,
                                           float mu, float (&xh)[8], float (&gd)[8], float (&d)[8]) {
  static_assert(!(LAYER && MEMEFF), "the layer norm's reconstructed input is not served here");
  float x[8];
  unpack8<T>(xv, x);
  unpack8<T>(dv, d);
  if constexpr (MEMEFF) {
#pragma unroll
    for (int e = 0; e < 8; ++e) xh[e] = x[e] * k.rg[e];
  } else {
    const float b = LAYER ? -mu * rs : 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) xh[e] = LAYER ? fmaf(x[e], rs, b) : x[e] * rs;
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) gd[e] = k.g[e] * d[e];
}

// dgamma/dbeta, called by every thread after the main loop: each warp-row stores its columns in its own slot (one
// owner lane a column), the slots are added in warp-row order into this CTA's partial, and after the grid barrier
// every CTA sums a slice of the columns over the CTA partials in CTA order. The slots and the slice scratch reuse the
// ring (the host checks they fit).
template <typename T, bool LAYER, int Q>
__device__ __forceinline__ void reduce_params(const BwdArgs& a, uint8_t* smem, int warp, int NW, const bool (&ok)[Q],
                                              const int (&ch)[Q], const float (&accg)[Q][8],
                                              const float (&accb)[LAYER ? Q : 1][8]) {
  const int nred = LAYER ? 2 : 1;
  const int NC = nred * a.C;
  float* red = reinterpret_cast<float*>(smem);  // [NWr][NC]
  __syncthreads();                              // every stage consumed: the ring is free
  if (warp >= 1 && warp <= NW) {
    float* mine = red + (size_t)((warp - 1) / a.NWc) * NC;
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      if (ok[q]) {
        float4* g4 = reinterpret_cast<float4*>(mine + ch[q] * 8);
        g4[0] = make_float4(accg[q][0], accg[q][1], accg[q][2], accg[q][3]);
        g4[1] = make_float4(accg[q][4], accg[q][5], accg[q][6], accg[q][7]);
        if constexpr (LAYER) {
          float4* b4 = reinterpret_cast<float4*>(mine + a.C + ch[q] * 8);
          b4[0] = make_float4(accb[q][0], accb[q][1], accb[q][2], accb[q][3]);
          b4[1] = make_float4(accb[q][4], accb[q][5], accb[q][6], accb[q][7]);
        }
      }
    }
  }
  __syncthreads();
  float* mypart = a.part + (size_t)blockIdx.x * NC;
  for (int i = threadIdx.x; i < NC; i += blockDim.x) {
    float v = red[i];
    for (int w = 1; w < a.NWr; ++w) v += red[(size_t)w * NC + i];
    mypart[i] = v;
  }
  cg::this_grid().sync();

  const int per = (NC + gridDim.x - 1) / gridDim.x;
  const int j0 = blockIdx.x * per;
  const int j1 = min(NC, j0 + per);
  const int ncols = j1 - j0;
  if (ncols <= 0) return;
  const int nthr = blockDim.x;
  if (ncols * 2 > nthr) {
    for (int j = j0 + threadIdx.x; j < j1; j += nthr) {
      float sum = 0.f;
      for (int i = 0; i < (int)gridDim.x; ++i) sum += a.part[(size_t)i * NC + j];
      if (j < a.C) store_param<T>(a.dgamma, j, sum);
      else store_param<T>(a.dbeta, j - a.C, sum);
    }
    return;
  }
  // Fewer columns than half the threads: `slices` threads a column sum strided CTA subsets, then one thread adds them.
  const int slices = nthr / ncols;
  const int tcol = threadIdx.x % ncols, tsl = threadIdx.x / ncols;
  __syncthreads();
  if (tsl < slices) {
    float sum = 0.f;
    for (int i = tsl; i < (int)gridDim.x; i += slices) sum += a.part[(size_t)i * NC + j0 + tcol];
    red[tsl * ncols + tcol] = sum;
  }
  __syncthreads();
  if (threadIdx.x < ncols) {
    float tot = 0.f;
    for (int i = 0; i < slices; ++i) tot += red[i * ncols + threadIdx.x];
    const int j = j0 + threadIdx.x;
    if (j < a.C) store_param<T>(a.dgamma, j, tot);
    else store_param<T>(a.dbeta, j - a.C, tot);
  }
}

// The narrow and wide mappings. RR rows a consumer warp at once (narrow): their dependent chains (shared load, sums,
// shuffles, store) overlap.
template <typename T, bool LAYER, bool MEMEFF, bool WIDE, int Q, int VK, int RR>
__global__ void __launch_bounds__(WIDE ? 288 : 544, 1) gn_bwd_tma_kernel(const BwdArgs a) {
  extern __shared__ __align__(1024) uint8_t smem_all[];
  uint8_t* smem = smem_all + kCtlBytes;  // the ring (and, after the main loop, the dgamma/dbeta scratch)
  constexpr bool NEED_MEAN = LAYER && !MEMEFF;
  const int S = a.S;
  uint64_t* full = reinterpret_cast<uint64_t*>(smem_all);
  uint64_t* empty = full + kMaxStages;
  int* tidx = reinterpret_cast<int*>(empty + kMaxStages);  // the tile in each stage
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int NW = a.NWc * a.NWr;
  const int C8 = a.C >> 3;

  if (threadIdx.x == 0) {
    for (int s = 0; s < S; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], NW);
    }
    fence_mbar_init();
  }
  __syncthreads();

  float accg[Q][8], accb[LAYER ? Q : 1][8];
  int ch[Q];
  bool ok[Q];
#pragma unroll
  for (int q = 0; q < Q; ++q) {
    ch[q] = 0;
    ok[q] = false;
  }
#pragma unroll
  for (int q = 0; q < Q; ++q)
#pragma unroll
    for (int e = 0; e < 8; ++e) accg[q][e] = 0.f;
#pragma unroll
  for (int q = 0; q < (LAYER ? Q : 1); ++q)
#pragma unroll
    for (int e = 0; e < 8; ++e) accb[q][e] = 0.f;

  if (warp == 0) {
    if (lane == 0) {
      int s = 0;
      uint32_t ph = 0;
      int t = blockIdx.x;
      for (int k = 0;; ++k) {
        if (k >= S) mbar_wait(&empty[s], ph ^ 1u);
        if (t >= a.num_tiles) {
          end_of_tiles(tidx, full, s);
          break;
        }
        tidx[s] = t;
        const long long r0 = (long long)t * a.TR;
        const int rows = (int)min((long long)a.TR, a.R - r0);
        const uint32_t xb = (uint32_t)rows * a.C * 2u;
        const uint32_t sb = (uint32_t)rows * a.G * 4u;
        uint8_t* st = smem + (size_t)s * a.stage_bytes;
        mbar_arrive_expect_tx(&full[s], 2u * xb + sb * (NEED_MEAN ? 2u : 1u));
        bulk_g2s(st, a.x + r0 * a.C * 2, xb, &full[s]);
        bulk_g2s(st + a.off_dy, a.dy + r0 * a.C * 2, xb, &full[s]);
        bulk_g2s(st + a.off_rs, a.rstd + r0 * a.G, sb, &full[s]);
        if (NEED_MEAN) bulk_g2s(st + a.off_mu, a.mean + r0 * a.G, sb, &full[s]);
        t += (int)gridDim.x;
        if (++s == S) { s = 0; ph ^= 1u; }
      }
    }
  } else if (warp <= NW) {
    const int cw = warp - 1;
    const int cidx = cw % a.NWc, wr = cw / a.NWc;
    const int V = a.V;
    int grp[Q];
    ChunkConst<MEMEFF> kc[Q];
    const int gl = WIDE ? 0 : lane % V;
    const int head = lane - gl;
    const bool pow2 = (V & (V - 1)) == 0;
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      int gg;
      bool okq;
      int c;
      if (!WIDE) {
        const int wc = cidx + q * a.NWc;
        gg = wc * a.GW + lane / V;
        okq = wc < a.U && lane < a.GW * V && gg < a.G;
        c = wc * a.GW * V + lane;
      } else {
        gg = cidx;
        const int j = q * 32 + lane;
        okq = j < V;
        c = gg * V + j;
      }
      ok[q] = okq;
      ch[q] = okq ? c : 0;
      grp[q] = okq ? gg : 0;
      uint4 graw = make_uint4(Pack<T>::kOnes, Pack<T>::kOnes, Pack<T>::kOnes, Pack<T>::kOnes);
      if (okq) graw = a.gamma[ch[q]];
      unpack8<T>(graw, kc[q].g);
      if constexpr (MEMEFF) {
#pragma unroll
        for (int e = 0; e < 8; ++e) kc[q].rg[e] = __frcp_rn(kc[q].g[e]);
      }
    }

    int s = 0;
    uint32_t ph = 0;
    for (;;) {
      mbar_wait(&full[s], ph);
      const int t = stage_tile(tidx, s);
      if (t < 0) break;
      const uint8_t* st = smem + (size_t)s * a.stage_bytes;
      const long long r0 = (long long)t * a.TR;
      const int rows = (int)min((long long)a.TR, a.R - r0);
      // TR is a multiple of RR * NWr, so every warp-row gets the same rows of every tile but the short last one.
      int r = wr;
      const uint4* xs = reinterpret_cast<const uint4*>(st) + r * C8;
      const uint4* ds = reinterpret_cast<const uint4*>(st + a.off_dy) + r * C8;
      const float* rss = reinterpret_cast<const float*>(st + a.off_rs) + r * a.G;
      const float* mus = reinterpret_cast<const float*>(st + a.off_mu) + r * a.G;
      uint4* dxr = a.dx + (r0 + r) * C8;
      const int xstep = a.NWr * C8, sstep = a.NWr * a.G;
      constexpr int RRk = WIDE ? 1 : RR;
      for (; r < rows; r += RRk * a.NWr, xs += RRk * xstep, ds += RRk * xstep, rss += RRk * sstep,
                       mus += RRk * sstep, dxr += RRk * xstep) {
        if constexpr (!WIDE) {
#pragma unroll
          for (int q = 0; q < Q; ++q) {
            float xh[RR][8], gd[RR][8], d[RR][8], rs[RR];
            float2 sm[RR];
#pragma unroll
            for (int j = 0; j < RR; ++j) {
              const bool rv = r + j * a.NWr < rows;
              const int jo = rv ? j : 0;  // a missing last row recomputes row 0 and stores nothing
              rs[j] = rss[jo * sstep + grp[q]];
              const float mu = NEED_MEAN ? mus[jo * sstep + grp[q]] : 0.f;
              chunk_vals<T, LAYER, MEMEFF>(xs[jo * xstep + ch[q]], ds[jo * xstep + ch[q]], kc[q], rs[j], mu, xh[j],
                                           gd[j], d[j]);
              float2 s0 = make_float2(0.f, 0.f), s1 = s0;
#pragma unroll
              for (int e = 0; e < 8; e += 2) {
                s0.x = fmaf(gd[j][e], xh[j][e], s0.x);
                s1.x = fmaf(gd[j][e + 1], xh[j][e + 1], s1.x);
                if (LAYER) { s0.y += gd[j][e]; s1.y += gd[j][e + 1]; }
              }
              sm[j] = make_float2(s0.x + s1.x, s0.y + s1.y);
            }
#pragma unroll
            for (int j = 0; j < RR; ++j) {
              if constexpr (LAYER) sm[j] = gsum2<VK>(sm[j], V, gl, head, pow2);
              else sm[j].x = gsum<VK>(sm[j].x, V, gl, head, pow2);
            }
#pragma unroll
            for (int j = 0; j < RR; ++j) {
              if (ok[q] && r + j * a.NWr < rows) {
                const float coef1 = sm[j].x * a.inv_cpg;
                const float coef2 = LAYER ? sm[j].y * a.inv_cpg : 0.f;
                float o[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) o[e] = rs[j] * fmaf(-xh[j][e], coef1, LAYER ? gd[j][e] - coef2 : gd[j][e]);
                st_cs_v4(dxr + j * xstep + ch[q], pack8<T>(o));
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                  accg[q][e] = fmaf(d[j][e], xh[j][e], accg[q][e]);
                  if (LAYER) accb[q][e] += d[j][e];
                }
              }
            }
          }
        } else {
          const float rs = rss[grp[0]];
          const float mu = NEED_MEAN ? mus[grp[0]] : 0.f;
          float2 sm = make_float2(0.f, 0.f);
#pragma unroll
          for (int q = 0; q < Q; ++q) {
            if (ok[q]) {
              float xh[8], gd[8], d[8];
              chunk_vals<T, LAYER, MEMEFF>(xs[ch[q]], ds[ch[q]], kc[q], rs, mu, xh, gd, d);
#pragma unroll
              for (int e = 0; e < 8; ++e) {
                sm.x = fmaf(gd[e], xh[e], sm.x);
                if (LAYER) sm.y += gd[e];
              }
            }
          }
          sm.x = warp_sum(sm.x);
          if (LAYER) sm.y = warp_sum(sm.y);
          const float coef1 = sm.x * a.inv_cpg;
          const float coef2 = LAYER ? sm.y * a.inv_cpg : 0.f;
#pragma unroll
          for (int q = 0; q < Q; ++q) {
            if (ok[q]) {
              float xh[8], gd[8], d[8], o[8];
              chunk_vals<T, LAYER, MEMEFF>(xs[ch[q]], ds[ch[q]], kc[q], rs, mu, xh, gd, d);
#pragma unroll
              for (int e = 0; e < 8; ++e) o[e] = rs * fmaf(-xh[e], coef1, LAYER ? gd[e] - coef2 : gd[e]);
              st_cs_v4(dxr + ch[q], pack8<T>(o));
#pragma unroll
              for (int e = 0; e < 8; ++e) {
                accg[q][e] = fmaf(d[e], xh[e], accg[q][e]);
                if (LAYER) accb[q][e] += d[e];
              }
            }
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[s]);
      if (++s == S) { s = 0; ph ^= 1u; }
    }
  }

  reduce_params<T, LAYER, Q>(a, smem, warp, NW, ok, ch, accg, accb);
}

__device__ __forceinline__ void named_bar(int id, int nthreads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory");
}

// Split groups (RMS from x): a group of V > 32 vectors over NWg warps of QW vectors a lane. Pass 1 over this warp's
// rows of the tile writes one partial (row, warp) to shared memory; a named barrier per warp-row; pass 2 adds the
// group's NWg partials in warp order and writes dx. The partials are double-buffered by tile parity: a warp writes a
// buffer again only after the next tile's barrier, which every warp of its warp-row reaches after its pass 2 over it.
template <typename T, bool LAYER, int QW>
__global__ void __launch_bounds__(544, 1) gn_bwd_tma_split_kernel(const BwdArgs a) {
  extern __shared__ __align__(1024) uint8_t smem_all[];
  uint8_t* smem = smem_all + kCtlBytes;  // the ring
  constexpr int NRED = LAYER ? 2 : 1;
  const int S = a.S;
  uint64_t* full = reinterpret_cast<uint64_t*>(smem_all);
  uint64_t* empty = full + kMaxStages;
  int* tidx = reinterpret_cast<int*>(empty + kMaxStages);                         // the tile in each stage
  float* part_rows = reinterpret_cast<float*>(smem + (size_t)S * a.stage_bytes);  // [2][TR][NWc][NRED]
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int NW = a.NWc * a.NWr;
  const int C8 = a.C >> 3;

  if (threadIdx.x == 0) {
    for (int s = 0; s < S; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], NW);
    }
    fence_mbar_init();
  }
  __syncthreads();

  float accg[QW][8], accb[LAYER ? QW : 1][8];
  int ch[QW];
  bool ok[QW];
#pragma unroll
  for (int q = 0; q < QW; ++q) {
    ch[q] = 0;
    ok[q] = false;
#pragma unroll
    for (int e = 0; e < 8; ++e) accg[q][e] = 0.f;
  }
#pragma unroll
  for (int q = 0; q < (LAYER ? QW : 1); ++q)
#pragma unroll
    for (int e = 0; e < 8; ++e) accb[q][e] = 0.f;

  if (warp == 0) {
    if (lane == 0) {
      int s = 0;
      uint32_t ph = 0;
      int t = blockIdx.x;
      for (int k = 0;; ++k) {
        if (k >= S) mbar_wait(&empty[s], ph ^ 1u);
        if (t >= a.num_tiles) {
          end_of_tiles(tidx, full, s);
          break;
        }
        tidx[s] = t;
        const long long r0 = (long long)t * a.TR;
        const int rows = (int)min((long long)a.TR, a.R - r0);
        const uint32_t xb = (uint32_t)rows * a.C * 2u;
        const uint32_t sb = (uint32_t)rows * a.G * 4u;
        uint8_t* st = smem + (size_t)s * a.stage_bytes;
        mbar_arrive_expect_tx(&full[s], 2u * xb + sb * (LAYER ? 2u : 1u));
        bulk_g2s(st, a.x + r0 * a.C * 2, xb, &full[s]);
        bulk_g2s(st + a.off_dy, a.dy + r0 * a.C * 2, xb, &full[s]);
        bulk_g2s(st + a.off_rs, a.rstd + r0 * a.G, sb, &full[s]);
        if (LAYER) bulk_g2s(st + a.off_mu, a.mean + r0 * a.G, sb, &full[s]);
        t += (int)gridDim.x;
        if (++s == S) { s = 0; ph ^= 1u; }
      }
    }
  } else if (warp <= NW) {
    const int cw = warp - 1;
    const int cc = cw % a.NWc, wr = cw / a.NWc;
    const int g = cc / a.NWg, sub = cc % a.NWg;
    ChunkConst<false> kc[QW];
#pragma unroll
    for (int q = 0; q < QW; ++q) {
      const int j = sub * 32 + lane + q * 32 * a.NWg;
      ok[q] = j < a.V;
      ch[q] = ok[q] ? g * a.V + j : 0;
      uint4 graw = make_uint4(Pack<T>::kOnes, Pack<T>::kOnes, Pack<T>::kOnes, Pack<T>::kOnes);
      if (ok[q]) graw = a.gamma[ch[q]];
      unpack8<T>(graw, kc[q].g);
    }
    const int bufsz = a.TR * a.NWc * NRED;
    int s = 0;
    uint32_t ph = 0;
    for (int k = 0;; ++k) {
      mbar_wait(&full[s], ph);
      const int t = stage_tile(tidx, s);
      if (t < 0) break;
      const uint8_t* st = smem + (size_t)s * a.stage_bytes;
      const long long r0 = (long long)t * a.TR;
      const int rows = (int)min((long long)a.TR, a.R - r0);
      const uint4* xs = reinterpret_cast<const uint4*>(st);
      const uint4* ds = reinterpret_cast<const uint4*>(st + a.off_dy);
      const float* rss = reinterpret_cast<const float*>(st + a.off_rs);
      const float* mus = reinterpret_cast<const float*>(st + a.off_mu);
      float* buf = part_rows + (size_t)(k & 1) * bufsz;
      // pass 1: this warp's partial sums of each of its rows
      for (int r = wr; r < rows; r += a.NWr) {
        const float rs = rss[r * a.G + g];
        const float mu = LAYER ? mus[r * a.G + g] : 0.f;
        float2 sm = make_float2(0.f, 0.f);
#pragma unroll
        for (int q = 0; q < QW; ++q) {
          if (ok[q]) {
            float xh[8], gd[8], d[8];
            chunk_vals<T, LAYER, false>(xs[r * C8 + ch[q]], ds[r * C8 + ch[q]], kc[q], rs, mu, xh, gd, d);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              sm.x = fmaf(gd[e], xh[e], sm.x);
              if (LAYER) sm.y += gd[e];
            }
          }
        }
        sm.x = warp_sum(sm.x);
        if (LAYER) sm.y = warp_sum(sm.y);
        if (lane == 0) {
          float* p = buf + (r * a.NWc + cc) * NRED;
          p[0] = sm.x;
          if (LAYER) p[1] = sm.y;
        }
      }
      named_bar(1 + wr, 32 * a.NWc);
      // pass 2: the group's sums, then dx and the dgamma/dbeta partials
      for (int r = wr; r < rows; r += a.NWr) {
        const float rs = rss[r * a.G + g];
        const float mu = LAYER ? mus[r * a.G + g] : 0.f;
        const float* p = buf + (r * a.NWc + g * a.NWg) * NRED;
        float s1 = 0.f, s2 = 0.f;
        for (int w = 0; w < a.NWg; ++w) {
          s1 += p[w * NRED];
          if (LAYER) s2 += p[w * NRED + 1];
        }
        const float coef1 = s1 * a.inv_cpg;
        const float coef2 = LAYER ? s2 * a.inv_cpg : 0.f;
        uint4* dxr = a.dx + (r0 + r) * C8;
#pragma unroll
        for (int q = 0; q < QW; ++q) {
          if (ok[q]) {
            float xh[8], gd[8], d[8], o[8];
            chunk_vals<T, LAYER, false>(xs[r * C8 + ch[q]], ds[r * C8 + ch[q]], kc[q], rs, mu, xh, gd, d);
#pragma unroll
            for (int e = 0; e < 8; ++e) o[e] = rs * fmaf(-xh[e], coef1, LAYER ? gd[e] - coef2 : gd[e]);
            st_cs_v4(dxr + ch[q], pack8<T>(o));
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              accg[q][e] = fmaf(d[e], xh[e], accg[q][e]);
              if (LAYER) accb[q][e] += d[e];
            }
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[s]);
      if (++s == S) { s = 0; ph ^= 1u; }
    }
  }
  reduce_params<T, LAYER, QW>(a, smem, warp, NW, ok, ch, accg, accb);
}

// ---------------------------------------------------------------- launch

constexpr int kSmemBudget = 220 * 1024;  // ring and control block, at most (B300: 227 KB a block)

bool aligned16(const at::Tensor& t) { return (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15u) == 0; }

struct Launch {
  int dev, grid, threads, optin;
  size_t smem;
  cudaStream_t stream;
};

// Sets the kernel's dynamic shared memory opt-in once a device, then launches it cooperatively (the dgamma/dbeta grid
// barrier needs every CTA resident). Returns false, with nothing launched and the error cleared, if the device refuses
// either; a refused opt-in is asked again on the next call.
template <class Kernel>
bool launch_cooperative(Kernel kern, unsigned long long& opted, const BwdArgs& a, const Launch& L) {
  const unsigned long long bit = L.dev < 64 ? (1ull << L.dev) : 0ull;
  if (!(opted & bit)) {
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, L.optin) != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }
    opted |= bit;
  }
  BwdArgs copy = a;
  void* args[] = {&copy};
  if (cudaLaunchCooperativeKernel((void*)kern, dim3(L.grid), dim3(L.threads), args, L.smem, L.stream) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return true;
}

template <bool LAYER, bool MEMEFF, bool WIDE, int Q, int VK, int RR>
bool launch(const BwdArgs& a, const Launch& L) {
  static unsigned long long opted = 0;  // devices this kernel has the opt-in on
  return launch_cooperative(gn_bwd_tma_kernel<__nv_bfloat16, LAYER, MEMEFF, WIDE, Q, VK, RR>, opted, a, L);
}

// Narrow groups: VK is the group width when it is 8 or 16 vectors, else 0 (run-time width); Q = 1 or 2 warp-chunks a
// lane. With one warp-chunk a warp, two rows at once overlap their chains; with two, two rows would spill at the
// 96-register cap of 17 warps, so one. dispatch.cu's backward() sends calls from y only at 8192 channels, where a row
// is 32 warp-chunks (Q = 2), so only that form is built for them.
template <bool LAYER, bool MEMEFF>
bool launch_narrow(const BwdArgs& a, int Q, const Launch& L) {
  const int VK = (a.V == 8 || a.V == 16) ? a.V : 0;
  if constexpr (!MEMEFF) {
    if (Q == 1) {
      if (VK == 8) return launch<LAYER, MEMEFF, false, 1, 8, 2>(a, L);
      if (VK == 16) return launch<LAYER, MEMEFF, false, 1, 16, 2>(a, L);
      return launch<LAYER, MEMEFF, false, 1, 0, 2>(a, L);
    }
  }
  if (Q == 2) {
    if (VK == 8) return launch<LAYER, MEMEFF, false, 2, 8, 1>(a, L);
    if (VK == 16) return launch<LAYER, MEMEFF, false, 2, 16, 1>(a, L);
    return launch<LAYER, MEMEFF, false, 2, 0, 1>(a, L);
  }
  return false;
}

// Wide groups, one warp a group (RMS from x): Q = 4..6 vectors a lane.
bool launch_wide(const BwdArgs& a, int Q, const Launch& L) {
  switch (Q) {
    case 4: return launch<false, false, true, 4, 0, 1>(a, L);
    case 5: return launch<false, false, true, 5, 0, 1>(a, L);
    case 6: return launch<false, false, true, 6, 0, 1>(a, L);
    default: return false;
  }
}

// Split groups (RMS from x): QW = 2..4 vectors a lane.
template <int QW>
bool launch_split_q(const BwdArgs& a, const Launch& L) {
  static unsigned long long opted = 0;
  return launch_cooperative(gn_bwd_tma_split_kernel<__nv_bfloat16, false, QW>, opted, a, L);
}

bool launch_split(const BwdArgs& a, int QW, const Launch& L) {
  if (QW == 2) return launch_split_q<2>(a, L);
  if (QW == 3) return launch_split_q<3>(a, L);
  if (QW == 4) return launch_split_q<4>(a, L);
  return false;
}

}  // namespace

Plan plan(bool layer, int64_t G, bool memeff, const at::Tensor& x_or_y, const at::Tensor& dy,
          const at::Tensor& mean, const at::Tensor& rstd, const at::Tensor& gamma, const at::Tensor& beta,
          const at::Tensor& dx, const at::Tensor& dgamma, const at::Tensor& dbeta) {
  Plan p;
  const auto dt = x_or_y.scalar_type();
  if (dt != at::kBFloat16) return p;
  if (dy.scalar_type() != dt || dx.scalar_type() != dt) return p;
  if (x_or_y.dim() < 2 || !x_or_y.is_contiguous() || !dy.is_contiguous() || !dx.is_contiguous()) return p;
  const int64_t C = x_or_y.size(-1);
  if (C <= 0 || G <= 0 || C % G) return p;
  const int64_t R = x_or_y.numel() / C;
  if (R <= 0 || dy.numel() != R * C || dx.numel() != R * C) return p;
  const int64_t cpg = C / G;
  if (C % 8 || cpg % 8 || C > 8192) return p;
  if ((R * G) % 4) return p;
  // Affine only: RMS from x or from y, the layer norm from x.
  if (gamma.numel() == 0 || (layer && memeff)) return p;
  const bool need_mean = layer && !memeff;
  if (rstd.scalar_type() != at::kFloat || !rstd.is_contiguous() || rstd.numel() != R * G || !aligned16(rstd))
    return p;
  if (need_mean && (mean.scalar_type() != at::kFloat || !mean.is_contiguous() || mean.numel() != R * G ||
                    !aligned16(mean)))
    return p;
  if (gamma.scalar_type() != dt || !gamma.is_contiguous() || gamma.numel() != C || !aligned16(gamma)) return p;
  if (dgamma.numel() != C || dgamma.scalar_type() != dt) return p;
  if (layer && (beta.scalar_type() != dt || !beta.is_contiguous() || beta.numel() != C || !aligned16(beta) ||
                dbeta.numel() != C || dbeta.scalar_type() != dt))
    return p;
  if (!aligned16(x_or_y) || !aligned16(dy) || !aligned16(dx)) return p;

  const int V = (int)(cpg / 8);
  const bool wide = V > 32;
  int GW = 1, U = 1, Q = 1, NWc = 1, NWr = 1, NWg = 1;
  bool split = false;
  if (!wide) {
    GW = 32 / V;
    U = (int)((G + GW - 1) / GW);
    if (U > 32) return p;
    Q = U <= 16 ? 1 : 2;
    NWc = (U + Q - 1) / Q;
    NWr = std::max(1, 16 / NWc);
  } else {
    if (layer || memeff) return p;
    Q = (V + 31) / 32;
    // One warp a group (Q = 4..6 vectors a lane, at most 8 groups) unless the group is a whole number of 64-vector
    // units: then the split mapping fills 16 warps with no idle lane and is faster. Wider groups can only be split.
    const bool one_warp = Q >= 4 && Q <= 6 && G <= 8;
    if (one_warp && !(V >= 128 && V % 64 == 0)) {
      NWc = (int)G;
      NWr = std::max(1, 8 / NWc);
    } else {
      if (V < 128) return p;
      // Split: the QW in {2, 3, 4} that fills the most of the 16 consumer warps, the smallest on a tie.
      int best_w = 0;
      for (int qw : {2, 3, 4}) {
        const int ng = (V + 32 * qw - 1) / (32 * qw);
        if ((int64_t)ng * G > 16) continue;
        const int nc = (int)G * ng;
        int nr = 1;  // warp-rows: a power of two (TR is a multiple of it), at most 8 named barriers
        while (nr * 2 * nc <= 16 && nr < 8) nr *= 2;
        if (nc * nr > best_w) {
          best_w = nc * nr;
          Q = qw;
          NWg = ng;
          NWc = nc;
          NWr = nr;
        }
      }
      if (best_w == 0) return p;
      split = true;
    }
  }

  // Launch geometry on x_or_y's device.
  const int dev = x_or_y.get_device();
  const DevInfo info = dev_info(dev);
  if (info.sms <= 0) return p;
  // 1 KB for the control block; 4 KB more for the split kernel's row partials ([2][TR][NWc][nred] floats).
  const int budget = std::min(kSmemBudget, info.smem_optin) - 1024 - (split ? 4096 : 0);
  const int RR = wide ? 1 : (Q >= 2 ? 1 : 2);
  const int NW = NWc * NWr;
  // Rows a tile: about 48 KB of each of x_or_y and dy for rows of 16 KB, 40 KB otherwise, rounded up to a multiple of
  // RR * NWr (every warp-row gets the same rows), with the statistics' bytes (rows * G * 4) a multiple of 16 and at
  // least two stages.
  const int unit = RR * NWr;
  const int tb = C >= 8192 ? 49152 : 40960;
  const int nst = need_mean ? 2 : 1;
  auto stage_of = [&](int tr) {
    const int s = 2 * tr * (int)C * 2 + tr * (int)G * 4 * nst;
    return (s + 127) / 128 * 128;
  };
  int TR = std::max(unit, (tb / (2 * (int)C) + unit - 1) / unit * unit);
  while ((TR * G) % 4) TR += unit;
  while (budget / stage_of(TR) < 2) {
    int t = TR - unit;
    while (t > 0 && (t * G) % 4) t -= unit;
    if (t <= 0) return p;
    TR = t;
  }
  const int stage = stage_of(TR);
  const int S = std::min(8, budget / stage);
  const int xbytes = TR * (int)C * 2;
  const int sbytes = TR * (int)G * 4;
  const long long num_tiles = (R + TR - 1) / TR;
  const int grid = (int)std::min<long long>(info.sms, num_tiles);
  const int nred = layer ? 2 : 1;
  // The dgamma/dbeta combine reuses the ring: one slot of nred * C floats a warp-row, then one float a thread.
  const size_t scratch = std::max<size_t>((size_t)NWr * nred * C, (size_t)32 * (1 + NWc * NWr)) * sizeof(float);
  if (scratch > (size_t)S * stage) return p;
  const size_t prow = split ? (size_t)2 * TR * NWc * nred * sizeof(float) : 0;
  if (prow > 4096) return p;
  const size_t smem = (size_t)kCtlBytes + (size_t)S * stage + prow;

  // RMS narrow mappings decline from 65536 rows, where they measured slower than the generic kernels (the layer
  // norm with the same plan measured faster on this kernel, so it stays).
  if (!layer && !wide && !split && R >= 65536) return p;

  p.ok = true;
  p.split = split;
  p.wide = wide;
  p.layer = layer;
  p.need_mean = need_mean;
  p.memeff = memeff;
  p.V = V;
  p.GW = GW;
  p.U = U;
  p.Q = Q;
  p.NWc = NWc;
  p.NWr = NWr;
  p.NWg = NWg;
  p.TR = TR;
  p.S = S;
  p.stage_bytes = stage;
  p.num_tiles = (int)num_tiles;
  // The last wave of tiles leaves CTAs idle, and a mapping may fill fewer than its consumer warps (8 for one warp a
  // group, else 16).
  p.occupancy = (float)num_tiles / (float)(((num_tiles + grid - 1) / grid) * grid) *
                std::min(1.f, (float)(NWc * NWr) / (wide && !split ? 8.f : 16.f));
  p.grid = grid;
  p.threads = 32 * (1 + NW);
  p.dev = dev;
  p.R = R;
  p.C = C;
  p.G = G;
  p.smem = (long long)smem;
  p.smem_optin = (long long)info.smem_optin;
  p.off_dy = xbytes;
  p.off_rs = 2 * xbytes;
  p.off_mu = 2 * xbytes + sbytes;
  p.inv_cpg = 1.0f / (float)cpg;
  p.nred = nred;
  return p;
}

bool backward(const Plan& p, const at::Tensor& x_or_y, const at::Tensor& dy, const at::Tensor& mean,
              const at::Tensor& rstd, const at::Tensor& gamma, const at::Tensor& beta, at::Tensor& dx,
              at::Tensor& dgamma, at::Tensor& dbeta) {
  if (!p.ok) return false;
  BwdArgs a{};
  a.x = reinterpret_cast<const uint8_t*>(x_or_y.data_ptr());
  a.dy = reinterpret_cast<const uint8_t*>(dy.data_ptr());
  a.mean = p.need_mean ? mean.data_ptr<float>() : nullptr;
  a.rstd = rstd.data_ptr<float>();
  a.gamma = reinterpret_cast<const uint4*>(gamma.data_ptr());
  a.dx = reinterpret_cast<uint4*>(dx.data_ptr());
  a.R = p.R;
  a.C = (int)p.C;
  a.G = (int)p.G;
  a.V = p.V;
  a.GW = p.GW;
  a.U = p.U;
  a.TR = p.TR;
  a.NWg = p.NWg;
  a.S = p.S;
  a.NWc = p.NWc;
  a.NWr = p.NWr;
  a.num_tiles = p.num_tiles;
  a.stage_bytes = p.stage_bytes;
  a.off_dy = p.off_dy;
  a.off_rs = p.off_rs;
  a.off_mu = p.off_mu;
  a.inv_cpg = p.inv_cpg;
  // The CTA partials come from the caching allocator on the current stream, so the call is safe on any stream and
  // under graph capture.
  const at::Tensor part = at::empty({(int64_t)p.grid * p.nred * (int64_t)p.C}, x_or_y.options().dtype(at::kFloat));
  a.part = part.data_ptr<float>();
  a.dgamma = dgamma.data_ptr();
  a.dbeta = p.layer ? dbeta.data_ptr() : nullptr;
  const Launch L{p.dev, p.grid, p.threads, (int)p.smem_optin, (size_t)p.smem, at::cuda::getCurrentCUDAStream()};
  if (p.split) return launch_split(a, p.Q, L);
  if (p.wide) return launch_wide(a, p.Q, L);
  if (p.layer) return launch_narrow<true, false>(a, p.Q, L);
  return p.memeff ? launch_narrow<false, true>(a, p.Q, L) : launch_narrow<false, false>(a, p.Q, L);
}

}  // namespace xllm::ops::blackwell::tma
