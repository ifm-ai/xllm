// Flat forward of the group RMS norm and group layer norm (forward_flat, called by dispatch.cu's forward() in
// dispatch.cu): every call whose groups are at most 1024 vectors of 16 bytes (8192 channels in 16 bits, 4096 in fp32),
// and every call whose group width is not a whole number of vectors. x[rows, groups, cpg] is one flat list of rows *
// groups contiguous groups, and the statistics of group g are mean[g] and rstd[g].
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <type_traits>

#include "dispatch.h"
#include "pdl.h"

namespace xllm::ops::blackwell {
namespace {

constexpr int kBlock = 256;

__device__ __forceinline__ float cvt(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float cvt(__half v) { return __half2float(v); }
__device__ __forceinline__ float cvt(float v) { return v; }
__device__ __forceinline__ void put(__nv_bfloat16& d, float s) { d = __float2bfloat16(s); }
__device__ __forceinline__ void put(__half& d, float s) { d = __float2half(s); }
__device__ __forceinline__ void put(float& d, float s) { d = s; }

// Two neighbouring 16-bit values widened to a float pair, and a float pair rounded once to 16 bits.
__device__ __forceinline__ float2 widen2(const __nv_bfloat16* p) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
}
__device__ __forceinline__ float2 widen2(const __half* p) {
  return __half22float2(*reinterpret_cast<const __half2*>(p));
}
__device__ __forceinline__ void narrow2(__nv_bfloat16* d, float2 v) {
  *reinterpret_cast<__nv_bfloat162*>(d) = __float22bfloat162_rn(v);
}
__device__ __forceinline__ void narrow2(__half* d, float2 v) { *reinterpret_cast<__half2*>(d) = __float22half2_rn(v); }

// One 16-byte chunk of a group.
template <typename T, int VEC>
struct alignas(16) Chunk {
  T e[VEC];
};

// The sum over the TPG lanes of a group (TPG <= 32, a power of two), returned to all of them: an xor butterfly over
// the distances TPG/2 .. 1. A group's lanes start at a multiple of TPG, so every partner is inside the group.
template <int TPG>
__device__ __forceinline__ float warp_sum_bcast(float v) {
  if constexpr (TPG > 1) {
#pragma unroll
    for (int d = TPG / 2; d > 0; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
  }
  return v;
}

// The sum over the TPG > 32 lanes of a group (whole warps), returned to all of them: each warp's sum goes to
// shm[slot * NBANK + warp], and after one barrier every thread adds its group's TPG / 32 warp sums in warp order.
// No barrier follows the read. The next write of the same slot comes after the grid-stride loop's own barrier, which
// every thread reaches on every iteration, and the code between the two touches no shared memory. The layer norm's two
// reductions use slots 0 and 1, so the variance's write cannot disturb a thread still reading the mean.
template <int TPG, int NBANK>
__device__ __forceinline__ float shm_sum_bcast(float v, float* shm, int slot) {
  constexpr int WPG = TPG / 32;
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
  if ((threadIdx.x & 31) == 0) shm[slot * NBANK + (threadIdx.x >> 5)] = v;
  __syncthreads();
  const int base = (int)(threadIdx.x / TPG) * WPG;
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < WPG; ++i) t += shm[slot * NBANK + base + i];
  return t;
}

// The vectorized flat kernel. A group is TPG consecutive lanes, Q 16-byte chunks a lane (lane p holds chunks p,
// p + TPG, ...); a CTA of BLOCK threads holds BLOCK / TPG groups and walks the groups with a grid-stride loop. x is
// read once: its values stay in registers from the load to the store. A lane sums its chunks in ascending order, then
// the group's lanes are summed by warp_sum_bcast (TPG <= 32) or shm_sum_bcast (TPG > 32). The layer norm takes the
// mean, then the sum of squared deviations from it, and the epilogue scales the deviations (y is exact where x is the
// mean).
//   HOIST (affine layer norm only): gamma and beta are loaded beside x, before the reductions, so their latency is
//   hidden behind them; it costs registers, and launch_arm takes it only on short grids.
//   BLOCK: 256, or 64 (narrow CTAs) and 512 for the arms some underfilled grids take.
template <typename T, int TPG, int Q, bool LAYER, bool AFFINE, bool HOIST = false, int BLOCK = kBlock>
__global__ __launch_bounds__(BLOCK) void fwd_kernel(const T* __restrict__ x, const T* __restrict__ gamma,
                                                     const T* __restrict__ beta, T* __restrict__ yout,
                                                     float* __restrict__ mean_out, float* __restrict__ rstd,
                                                     unsigned n_groups, unsigned groups, int chunks, int cpg,
                                                     float eps, float inv_c) {
  cudaGridDependencySynchronize();
  constexpr int VEC = 16 / (int)sizeof(T);
  constexpr int GPB = BLOCK / TPG;  // groups per CTA iteration
  // bf16 RMS without affine on the (256, 4) and (32, 2) arms: the squares are summed in two chains, even and odd
  // channels, on float pairs, and folded once before the reduction.
  constexpr bool kPairSq = std::is_same<T, __nv_bfloat16>::value && !LAYER && !AFFINE &&
                            ((TPG == 256 && Q == 4) || (TPG == 32 && Q == 2));
  // The 16-bit layer norm's sum and centred sum of squares run the same way, on float pairs.
  constexpr bool kPairLayer = LAYER && sizeof(T) == 2;
  static_assert(BLOCK % 32 == 0, "the CTA width must be a whole number of warps");
  static_assert(TPG <= 32 || (BLOCK / TPG) * (TPG / 32) == BLOCK / 32,
                "for TPG > 32 the bank stride must hold exactly one word per warp of this CTA");
  __shared__ float shm[2 * (BLOCK / 32)];
  const int tid = (int)threadIdx.x;
  const int p = tid & (TPG - 1);   // chunk slot inside the group
  const int gsub = tid / TPG;      // group slot inside the CTA
  const unsigned step = (unsigned)gridDim.x * GPB;

  for (unsigned g0 = (unsigned)blockIdx.x * GPB; g0 < n_groups; g0 += step) {
    if constexpr (TPG > 32) __syncthreads();
    const unsigned g = g0 + (unsigned)gsub;
    const bool active = g < n_groups;
    const size_t goff = (size_t)g * (size_t)cpg;
    const T* xg = x + goff;

    Chunk<T, VEC> v[Q];
    // The layer norm keeps the converted values (later the deviations): it reads them three times.
    float fv[Q][VEC];
    constexpr bool kHoistAff = HOIST && AFFINE && LAYER;
    const size_t coffH = kHoistAff ? (size_t)(g % groups) * (size_t)cpg : (size_t)0;
    Chunk<T, VEC> gv_[kHoistAff ? Q : 1];
    Chunk<T, VEC> bv_[kHoistAff ? Q : 1];
    float s = 0.f;   // RMS: sum of squares
    float sx = 0.f;  // layer norm: sum
    float2 s2 = make_float2(0.f, 0.f);
    float2 sx2 = make_float2(0.f, 0.f);
#pragma unroll
    for (int i = 0; i < Q; ++i) {
      const int chunk_idx = p + i * TPG;
      if (active && chunk_idx < chunks) {
        if constexpr (kHoistAff) {
          gv_[i] = *reinterpret_cast<const Chunk<T, VEC>*>(gamma + coffH + (size_t)chunk_idx * VEC);
          bv_[i] = *reinterpret_cast<const Chunk<T, VEC>*>(beta + coffH + (size_t)chunk_idx * VEC);
        }
        v[i] = *reinterpret_cast<const Chunk<T, VEC>*>(xg + (size_t)chunk_idx * VEC);
        if constexpr (kPairSq) {
#pragma unroll
          for (int j = 0; j < VEC; j += 2) {
            const float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(v[i].e + j));
            s2 = __ffma2_rn(f2, f2, s2);
          }
        } else if constexpr (kPairLayer) {
#pragma unroll
          for (int j = 0; j < VEC; j += 2) {
            const float2 f2 = widen2(&v[i].e[j]);
            fv[i][j] = f2.x;
            fv[i][j + 1] = f2.y;
            sx2 = __fadd2_rn(sx2, f2);
          }
        } else {
#pragma unroll
          for (int j = 0; j < VEC; ++j) {
            const float f = cvt(v[i].e[j]);
            if constexpr (LAYER) {
              fv[i][j] = f;
              sx += f;
            } else {
              s = fmaf(f, f, s);
            }
          }
        }
      }
    }

    if constexpr (kPairSq) s = s2.x + s2.y;
    if constexpr (kPairLayer) sx = sx2.x + sx2.y;

    float mu = 0.f;
    float var;
    if constexpr (LAYER) {
      mu = ((TPG > 32) ? shm_sum_bcast<TPG, BLOCK / 32>(sx, shm, 0) : warp_sum_bcast<TPG>(sx)) * inv_c;
      float c2 = 0.f;
      float2 c22 = make_float2(0.f, 0.f);
      const float2 m2 = make_float2(-mu, -mu);
#pragma unroll
      for (int i = 0; i < Q; ++i) {
        const int chunk_idx = p + i * TPG;
        if (active && chunk_idx < chunks) {
          // The deviations replace the values they come from.
          if constexpr (kPairLayer) {
#pragma unroll
            for (int j = 0; j < VEC; j += 2) {
              const float2 d = __fadd2_rn(make_float2(fv[i][j], fv[i][j + 1]), m2);
              fv[i][j] = d.x;
              fv[i][j + 1] = d.y;
              c22 = __ffma2_rn(d, d, c22);
            }
          } else {
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
              const float d = fv[i][j] - mu;
              fv[i][j] = d;
              c2 = fmaf(d, d, c2);
            }
          }
        }
      }
      if constexpr (kPairLayer) c2 = c22.x + c22.y;
      var = ((TPG > 32) ? shm_sum_bcast<TPG, BLOCK / 32>(c2, shm, 1) : warp_sum_bcast<TPG>(c2)) * inv_c;
    } else {
      var = ((TPG > 32) ? shm_sum_bcast<TPG, BLOCK / 32>(s, shm, 0) : warp_sum_bcast<TPG>(s)) * inv_c;
    }

    const float r = rsqrtf(var + eps);
    if (active && p == 0) {
      rstd[g] = r;
      if constexpr (LAYER) mean_out[g] = mu;
    }

    const float a = r;
    const float b = -mu * r;
    const size_t coff = AFFINE ? (size_t)(g % groups) * (size_t)cpg : (size_t)0;
    T* yg = yout + goff;
#pragma unroll
    for (int i = 0; i < Q; ++i) {
      const int chunk_idx = p + i * TPG;
      if (active && chunk_idx < chunks) {
        const size_t off = (size_t)chunk_idx * VEC;
        Chunk<T, VEC> o;
        if constexpr (AFFINE) {
          Chunk<T, VEC> gv;
          if constexpr (kHoistAff) gv = gv_[i];
          else gv = *reinterpret_cast<const Chunk<T, VEC>*>(gamma + coff + off);
          if constexpr (LAYER) {
            // The affine layer norm reads beta: a layer norm with gamma always comes with beta.
            Chunk<T, VEC> bv;
            if constexpr (kHoistAff) bv = bv_[i];
            else bv = *reinterpret_cast<const Chunk<T, VEC>*>(beta + coff + off);
            if constexpr (sizeof(T) == 2 && !(TPG == 4 && Q == 4)) {
              // Two channels an instruction in float, y rounded once at the store.
              const float2 r2 = make_float2(r, r);
#pragma unroll
              for (int j = 0; j < VEC; j += 2) {
                const float2 t = __fmul2_rn(make_float2(fv[i][j], fv[i][j + 1]), r2);
                narrow2(&o.e[j], __ffma2_rn(t, widen2(&gv.e[j]), widen2(&bv.e[j])));
              }
            } else {
              // fp32, and the 16-bit (4, 4) arm: one channel an instruction.
#pragma unroll
              for (int j = 0; j < VEC; ++j) put(o.e[j], (fv[i][j] * r) * cvt(gv.e[j]) + cvt(bv.e[j]));
            }
          } else {
            if constexpr (sizeof(T) == 2 && !(TPG == 32 && Q == 6)) {
              // x * rstd * gamma, two channels an instruction in float, y rounded once at the store.
              const float2 r2 = make_float2(r, r);
#pragma unroll
              for (int j = 0; j < VEC; j += 2) {
                narrow2(&o.e[j], __fmul2_rn(__fmul2_rn(widen2(&v[i].e[j]), r2), widen2(&gv.e[j])));
              }
            } else {
              // fp32, and the 16-bit (32, 6) arm: one channel an instruction.
#pragma unroll
              for (int j = 0; j < VEC; ++j) put(o.e[j], (cvt(v[i].e[j]) * a + b) * cvt(gv.e[j]));
            }
          }
        } else {
          if constexpr (LAYER) {
#pragma unroll
            for (int j = 0; j < VEC; ++j) put(o.e[j], fv[i][j] * r);
          } else {
#pragma unroll
            for (int j = 0; j < VEC; ++j) put(o.e[j], cvt(v[i].e[j]) * a + b);
          }
        }
        *reinterpret_cast<Chunk<T, VEC>*>(yg + off) = o;
      }
    }
  }
}

// One step of compensated (Kahan) summation: `comp` carries the rounding error of every earlier step back into `sum`.
__device__ __forceinline__ void kahan_add(float& sum, float& comp, float v) {
  const float y = v - comp;
  const float t = sum + y;
  comp = (t - sum) - y;
  sum = t;
}

// One lane's sum of term(i) over i = lane, lane + 32, ... < cpg. Sixteen terms are loaded and added at a time, and the
// sixteen-term partial sums are added with compensation, each while the next sixteen loads are in flight.
template <typename F>
__device__ __forceinline__ float lane_sum(int lane, int cpg, F term) {
  float sum = 0.f, comp = 0.f, part = 0.f;
  int i = lane;
  for (; i + 15 * 32 < cpg; i += 16 * 32) {
    float v[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) v[k] = term(i + k * 32);
    kahan_add(sum, comp, part);
    part = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) part += v[k];
  }
  for (; i < cpg; i += 32) part += term(i);
  kahan_add(sum, comp, part);
  return sum;
}

// Groups from this many channels (192 terms a lane) take the compensated sums: a plain running sum loses bits in
// proportion to the terms a lane adds. Narrower groups keep the plain loop, which measured up to 5% faster.
constexpr int kCompensatedFrom = 6144;

// The scalar flat kernel, for any group width: one warp a group, kBlock / 32 groups a CTA, lane l reading elements l,
// l + 32, ... A lane's sums run in that order (in sixteen-term blocks when COMPENSATED), then over the warp with an
// xor butterfly. x is read once for the statistics (twice for the layer norm: the sum, then the centred squares) and
// once more for the output.
template <typename T, bool LAYER, bool AFFINE, bool COMPENSATED>
__global__ __launch_bounds__(kBlock) void fwd_scalar_kernel(const T* __restrict__ x, const T* __restrict__ gamma,
                                                            const T* __restrict__ beta, T* __restrict__ yout,
                                                            float* __restrict__ mean_out, float* __restrict__ rstd,
                                                            unsigned long long n_groups, unsigned long long groups,
                                                            int cpg, float eps, float inv_c) {
  cudaGridDependencySynchronize();
  const unsigned long long g = (unsigned long long)blockIdx.x * (kBlock / 32) + (threadIdx.x >> 5);
  if (g >= n_groups) return;
  const int lane = (int)(threadIdx.x & 31);
  const size_t goff = (size_t)g * (size_t)cpg;
  const T* xg = x + goff;

  float s = 0.f;
  float sx = 0.f;
  if constexpr (COMPENSATED) {
    if constexpr (LAYER) {
      sx = lane_sum(lane, cpg, [&](int i) { return cvt(xg[i]); });
    } else {
      s = lane_sum(lane, cpg, [&](int i) {
        const float f = cvt(xg[i]);
        return f * f;
      });
    }
  } else {
    for (int i = lane; i < cpg; i += 32) {
      const float f = cvt(xg[i]);
      s = fmaf(f, f, s);
      if constexpr (LAYER) sx += f;
    }
  }
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) s += __shfl_xor_sync(0xffffffffu, s, d);
  if constexpr (LAYER) {
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) sx += __shfl_xor_sync(0xffffffffu, sx, d);
  }

  float mu = 0.f;
  float var;
  if constexpr (LAYER) {
    mu = sx * inv_c;
    float c2 = 0.f;
    if constexpr (COMPENSATED) {
      c2 = lane_sum(lane, cpg, [&](int i) {
        const float d = cvt(xg[i]) - mu;
        return d * d;
      });
    } else {
      for (int i = lane; i < cpg; i += 32) {
        const float d = cvt(xg[i]) - mu;
        c2 = fmaf(d, d, c2);
      }
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) c2 += __shfl_xor_sync(0xffffffffu, c2, d);
    var = c2 * inv_c;
  } else {
    var = s * inv_c;
  }

  const float r = rsqrtf(var + eps);
  if (lane == 0) {
    rstd[g] = r;
    if constexpr (LAYER) mean_out[g] = mu;
  }

  const size_t coff = AFFINE ? (size_t)(g % groups) * (size_t)cpg : (size_t)0;
  T* yg = yout + goff;
  for (int i = lane; i < cpg; i += 32) {
    float v = (cvt(xg[i]) - mu) * r;
    if constexpr (AFFINE) {
      v *= cvt(gamma[coff + i]);
      if constexpr (LAYER) v += cvt(beta[coff + i]);
    }
    put(yg[i], v);
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Launch

// The SM count of the first device the process launches on.
int sm_count() {
  static const int count = [] {
    int dev = 0, n = 0;
    cudaGetDevice(&dev);
    return cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && n > 0 ? n : 132;
  }();
  return count;
}

// CTAs for n groups at groups_per_cta a CTA, at most bpsm CTAs an SM (the kernel grid-strides over the rest).
int grid_for(unsigned n, int groups_per_cta, int bpsm) {
  const long long tiles = ((long long)n + groups_per_cta - 1) / groups_per_cta;
  return (int)std::min(tiles, (long long)sm_count() * bpsm);
}

// One call, as every arm launches it.
struct Params {
  const void* x;
  const void* gamma;  // null without affine
  const void* beta;   // null unless the affine layer norm
  void* y;
  float* mean;        // null unless the layer norm
  float* rstd;
  unsigned n;         // groups in all, rows * groups
  unsigned groups;    // groups a row
  int cpg;            // channels a group
  int chunks;         // 16-byte chunks a group
  float eps;
  bool layer;
  int bpsm;           // the grid cap, in CTAs an SM
  cudaStream_t stream;
};

template <typename T, int TPG, int Q, bool LAYER, bool AFFINE, bool HOIST = false, int BLOCK = kBlock>
void launch(const Params& p) {
  LaunchKernelPdl(fwd_kernel<T, TPG, Q, LAYER, AFFINE, HOIST, BLOCK>,
                  dim3(grid_for(p.n, BLOCK / TPG, p.bpsm)), dim3(BLOCK), 0, p.stream,
                  static_cast<const T*>(p.x), static_cast<const T*>(p.gamma),
                  static_cast<const T*>(p.beta), static_cast<T*>(p.y), p.mean, p.rstd, p.n,
                  p.groups, p.chunks, p.cpg, p.eps, 1.f / (float)p.cpg);
}

// The arms whose affine layer norm may prefetch its factors (HOIST).
template <typename T, int TPG, int Q>
constexpr bool kHoistArm =
    TPG <= 32 ||
    (sizeof(T) == 4 ? (TPG == 128 && Q == 4) : ((TPG == 128 && (Q == 2 || Q == 4)) || (TPG == 256 && Q == 4)));

// An affine arm on 64-thread CTAs, four times the CTAs of a 256-thread launch, for grids below one CTA an SM. The layer
// norm prefetches its factors: every grid this short is within the prefetch's wave bound (launch_arm).
template <typename T, int TPG, int Q>
void launch_narrow_cta(const Params& p) {
  if (p.layer) launch<T, TPG, Q, true, true, true, 64>(p);
  else launch<T, TPG, Q, false, true, false, 64>(p);
}

// One (TPG, Q) arm with affine: on 256-thread CTAs, but on the narrow CTAs for fp32 warp-sized arms on grids below one
// CTA an SM. The prefetching body of the layer norm costs registers and so resident CTAs; it is taken where the grid
// is at most 8 CTAs an SM (one chunk a lane) or 2 (more chunks a lane), where that residency is not needed.
template <typename T, int TPG, int Q>
void launch_affine_arm(const Params& p) {
  if constexpr (sizeof(T) == 4 && TPG <= 32) {
    if (grid_for(p.n, kBlock / TPG, p.bpsm) < sm_count()) return launch_narrow_cta<T, TPG, Q>(p);
  }
  if (!p.layer) return launch<T, TPG, Q, false, true>(p);
  if constexpr (kHoistArm<T, TPG, Q>) {
    if (grid_for(p.n, kBlock / TPG, p.bpsm) <= (Q == 1 ? 8 : 2) * sm_count())
      return launch<T, TPG, Q, true, true, true>(p);
  }
  launch<T, TPG, Q, true, true, false>(p);
}

// One (TPG, Q) arm, on 256-thread CTAs without affine.
template <typename T, int TPG, int Q>
void launch_arm(const Params& p) {
  if (p.gamma != nullptr) return launch_affine_arm<T, TPG, Q>(p);
  if (p.layer) launch<T, TPG, Q, true, false>(p);
  else launch<T, TPG, Q, false, false>(p);
}

// Packed lanes, groups of 1..511 chunks: two to eight chunks a lane (one for a one-chunk group), so a warp holds more
// whole groups.
template <typename T>
void launch_packed(const Params& p) {
  constexpr bool k16 = sizeof(T) == 2;
  const bool affine = p.gamma != nullptr;
  const int c = p.chunks;
  // 16 chunks with affine: eight lanes of two chunks rather than four of four (fewer registers, more CTAs an SM); in
  // 16 bits only on grids of at most 48 CTAs an SM, where the extra CTAs are work the machine can take.
  if (c == 16 && affine && (!k16 || grid_for(p.n, kBlock / 8, p.bpsm) <= 48 * sm_count()))
    return launch_affine_arm<T, 8, 2>(p);
  if constexpr (k16) {
    // 3..4 chunks with affine on a grid below one CTA an SM: 64-thread CTAs.
    if (c > 2 && c <= 4 && affine && grid_for(p.n, kBlock / 2, p.bpsm) < sm_count())
      return launch_narrow_cta<T, 2, 2>(p);
    // 256 chunks with affine: 64 lanes of four chunks.
    if (c == 256 && affine) return launch_affine_arm<T, 64, 4>(p);
  }
  if (c <= 1) return launch_arm<T, 1, 1>(p);
  if (c <= 2) return launch_arm<T, 2, 1>(p);
  if (c <= 4) return launch_arm<T, 2, 2>(p);
  if (c <= 8) return launch_arm<T, 4, 2>(p);
  if (c <= 16) return launch_arm<T, 4, 4>(p);
  if (c <= 32) return launch_arm<T, 8, 4>(p);
  if (c <= 64) return launch_arm<T, 32, 2>(p);
  if (c <= 128) return launch_arm<T, 32, 4>(p);
  if (c <= 160) return launch_arm<T, 32, 5>(p);
  if (c <= 192) return launch_arm<T, 32, 6>(p);
  if (c <= 256) return launch_arm<T, 32, 8>(p);
  launch_arm<T, 64, 8>(p);
}

// Spread lanes: one chunk a lane for the power-of-two groups of 4..32 chunks choose_mapping sends here, and whole warps
// a group, four chunks a lane, from 512 chunks (a lane's data registers, not the reduction, bound these widths).
template <typename T>
void launch_spread(const Params& p) {
  const int c = p.chunks;
  if (c <= 4) return launch_arm<T, 4, 1>(p);
  if (c <= 8) return launch_arm<T, 8, 1>(p);
  if (c <= 16) return launch_arm<T, 16, 1>(p);
  if (c <= 32) return launch_arm<T, 32, 1>(p);
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    // bf16 with affine, 513..1024 chunks, on a grid below one CTA an SM: 512 lanes of two chunks on a 512-thread CTA.
    // Without affine the wider CTA only lengthens the reduction.
    if (p.gamma != nullptr && c > 512 && grid_for(p.n, 1, p.bpsm) < sm_count()) {
      if (p.layer) return launch<T, 512, 2, true, true, true, 512>(p);
      return launch<T, 512, 2, false, true, false, 512>(p);
    }
  }
  if (c <= 512) return launch_arm<T, 128, 4>(p);
  launch_arm<T, 256, 4>(p);
}

// The scalar kernel covers every group with its own warp: no grid-stride loop, so no cap on the grid.
template <typename T, bool LAYER, bool AFFINE>
void launch_scalar_norm(const Params& p) {
  const long long tiles = ((long long)p.n + (kBlock / 32) - 1) / (kBlock / 32);
  const int grid = (int)std::min<long long>(tiles, 0x7fffffffLL);
  auto issue = [&](auto kernel) {
    LaunchKernelPdl(kernel, dim3(grid), dim3(kBlock), 0, p.stream, static_cast<const T*>(p.x),
                    static_cast<const T*>(p.gamma), static_cast<const T*>(p.beta),
                    static_cast<T*>(p.y), p.mean, p.rstd, (unsigned long long)p.n, p.groups, p.cpg,
                    p.eps, 1.f / (float)p.cpg);
  };
  if (p.cpg >= kCompensatedFrom) issue(fwd_scalar_kernel<T, LAYER, AFFINE, true>);
  else issue(fwd_scalar_kernel<T, LAYER, AFFINE, false>);
}

template <typename T>
void launch_scalar(const Params& p) {
  if (p.layer) {
    if (p.gamma != nullptr) launch_scalar_norm<T, true, true>(p);
    else launch_scalar_norm<T, true, false>(p);
  } else {
    if (p.gamma != nullptr) launch_scalar_norm<T, false, true>(p);
    else launch_scalar_norm<T, false, false>(p);
  }
}

// How a call is mapped onto lanes. Groups that are not a whole number of chunks take the scalar kernel. Whole chunks
// take packed lanes, except: groups of 512 chunks and wider, and power-of-two groups of 4..32 chunks whose packed grid
// would have fewer CTAs than the machine has SMs, take spread lanes (more CTAs); 256-chunk groups in the same case
// take 128 lanes of two chunks (team) instead of 32 lanes of eight.
enum class Mapping { kScalar, kPacked, kSpread, kTeam };

Mapping choose_mapping(const Params& p) {
  const int c = p.chunks;
  if (c >= 512) return Mapping::kSpread;
  const int sms = sm_count();
  if (c >= 4 && c <= 32 && (c & (c - 1)) == 0) {
    const int packed = grid_for(p.n, kBlock / (c <= 4 ? 2 : c <= 16 ? 4 : 8), p.bpsm);
    if (packed < sms && grid_for(p.n, kBlock / c, p.bpsm) > packed) return Mapping::kSpread;
  }
  if (c == 256) {
    const int packed = grid_for(p.n, kBlock / 32, p.bpsm);
    if (packed < sms && grid_for(p.n, kBlock / 128, p.bpsm) > packed) return Mapping::kTeam;
  }
  return Mapping::kPacked;
}

template <typename T>
void launch_flat(const Params& p, Mapping m) {
  switch (m) {
    case Mapping::kScalar:
      return launch_scalar<T>(p);
    case Mapping::kPacked:
      return launch_packed<T>(p);
    case Mapping::kSpread:
      return launch_spread<T>(p);
    case Mapping::kTeam:
      return launch_arm<T, 128, 2>(p);
  }
}

}  // namespace

// Operands are contiguous (the Call contract), inputs 16-byte aligned (dispatch.cu sends any other call to the generic
// kernels) and outputs freshly allocated, so every 16-byte chunk access is aligned.
bool forward_flat(const Shape& s, const Call& c) {
  const int vec = s.dtype == at::kFloat ? 4 : 8;  // elements in 16 bytes
  const bool whole_chunks = s.cpg % vec == 0;
  const int64_t chunks = s.cpg / vec;
  // Wider groups are the wide kernels'. The flat index of a group is an unsigned.
  if (s.rows < 1 || (whole_chunks && chunks > 1024) ||
      (unsigned long long)s.rows * (unsigned long long)s.groups >= 0xffffffffull)
    return false;
  const bool layer = s.norm == kLayer;
  Params p;
  p.x = c.x_or_y->const_data_ptr();
  p.gamma = s.affine ? c.gamma->const_data_ptr() : nullptr;
  p.beta = s.has_beta ? c.beta->const_data_ptr() : nullptr;
  p.y = c.y->data_ptr();
  p.mean = layer ? c.mean_out->data_ptr<float>() : nullptr;
  p.rstd = c.rstd_out->data_ptr<float>();
  p.n = (unsigned)(s.rows * s.groups);
  p.groups = (unsigned)s.groups;
  p.cpg = (int)s.cpg;
  p.chunks = (int)chunks;
  p.eps = (float)c.eps;
  p.layer = layer;
  // Grids of at most 64 CTAs an SM for wide groups, the layer norm and groups of 5, 8 or 10 chunks; 128 otherwise.
  p.bpsm = (chunks >= 512 || layer || chunks == 5 || chunks == 8 || chunks == 10) ? 64 : 128;
  p.stream = c10::cuda::getCurrentCUDAStream(c.x_or_y->device().index()).stream();  // x's device is the current one
  const Mapping m = whole_chunks ? choose_mapping(p) : Mapping::kScalar;
  if (s.dtype == at::kBFloat16) launch_flat<__nv_bfloat16>(p, m);
  else if (s.dtype == at::kHalf) launch_flat<__half>(p, m);
  else launch_flat<float>(p, m);
  return true;
}

}  // namespace xllm::ops::blackwell
