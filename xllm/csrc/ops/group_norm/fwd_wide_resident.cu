// Resident wide forward of the 16-bit group RMS norm and group layer norm (forward_wide_resident, called by
// dispatch.cu's forward() in dispatch.cu): groups of 8193..65536 channels, one CTA a (row, group) segment holding the
// whole segment, either in registers (wide_resident_fwd_kernel) or, for bf16 groups of 7169..8192 vectors, staged in
// shared memory (the tile kernels). A segment is read from global memory once and written once, with no cluster
// barrier.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <atomic>
#include <cstdint>
#include <type_traits>

#include "dispatch.h"

namespace xllm::ops::blackwell {
namespace {

constexpr int kMaxThreads = 1024;
constexpr int kVecBytes = 16;
using bf16 = at::BFloat16;
// The tile kernels' segments: 16-byte vectors of 8 bf16, 8 vectors a thread.
constexpr int kTileVec = 8;
constexpr int kTileVpt = 8;

template <typename T, int kVec>
union alignas(kVecBytes) VecT {
  float4 u;
  T v[kVec];
};

// One output vector of a tile kernel: (x - mean) * rstd, times gamma and plus beta (layer norm) with affine, in float
// and rounded once to bf16; with affine two channels an instruction.
template <bool AFFINE, bool LAYER>
__device__ __forceinline__ void tile_epilogue(VecT<bf16, kTileVec>& out, const bf16* p, const VecT<bf16, kTileVec>& w,
                                              const VecT<bf16, kTileVec>& b, float mean, float rstd) {
  constexpr int kVec = kTileVec;
  if constexpr (AFFINE) {
    auto* o2 = reinterpret_cast<__nv_bfloat162*>(out.v);
    const auto* p2 = reinterpret_cast<const __nv_bfloat162*>(p);
    const auto* w2 = reinterpret_cast<const __nv_bfloat162*>(w.v);
    const auto* b2 = reinterpret_cast<const __nv_bfloat162*>(b.v);
    const float2 m = make_float2(-mean, -mean);
    const float2 r = make_float2(rstd, rstd);
#pragma unroll
    for (int k = 0; k < kVec / 2; ++k) {
      float2 t = __bfloat1622float2(p2[k]);
      if constexpr (LAYER) t = __fadd2_rn(t, m);
      t = __fmul2_rn(t, r);
      const float2 g = __bfloat1622float2(w2[k]);
      if constexpr (LAYER) o2[k] = __float22bfloat162_rn(__ffma2_rn(t, g, __bfloat1622float2(b2[k])));
      else o2[k] = __float22bfloat162_rn(__fmul2_rn(t, g));
    }
  } else {
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      float o = (static_cast<float>(p[k]) - mean) * rstd;
      out.v[k] = static_cast<bf16>(o);
    }
  }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// The sum across the CTA, returned to every thread: warp sums into slot[warp], a barrier, warp 0 reduces them into
// slot[0], a barrier, every thread reads slot[0]. The second barrier lets the caller write `slot` again at once.
__device__ __forceinline__ float block_sum(float v, float* slot, int nwarp) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) slot[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float t = (lane < nwarp) ? slot[lane] : 0.f;
    t = warp_sum(t);
    if (lane == 0) slot[0] = t;
  }
  __syncthreads();
  return slot[0];
}

// The same sum with one barrier: every warp reduces the nwarp published partials with warp 0's tree, so every thread
// ends with block_sum's value. `slot` must not be written again until a later barrier.
__device__ __forceinline__ float block_sum_once(float v, float* slot, int nwarp) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) slot[warp] = v;
  __syncthreads();
  const float t = (lane < nwarp) ? slot[lane] : 0.f;
  return warp_sum(t);
}

// Two sums through one barrier, each reduced as block_sum_once reduces it: a in slot[0, nwarp), b in
// slot[nwarp, 2 * nwarp). The same rule holds for the next write of `slot`.
__device__ __forceinline__ void block_sum_pair(float& a, float& b, float* slot, int nwarp) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  a = warp_sum(a);
  b = warp_sum(b);
  if (lane == 0) {
    slot[warp] = a;
    slot[nwarp + warp] = b;
  }
  __syncthreads();
  const float ta = (lane < nwarp) ? slot[lane] : 0.f;
  const float tb = (lane < nwarp) ? slot[nwarp + lane] : 0.f;
  a = warp_sum(ta);
  b = warp_sum(tb);
}

// One CTA a segment, the segment in registers: VPT vectors of 8 elements a thread (vector j = tid + i * blockDim.x),
// read as streaming loads (x is read once; L2 is left to gamma and beta, which every row reads). The output overwrites
// the data in place, which keeps the bf16 VPT 8 form's register spill smallest. A thread sums its vectors in order,
// then the CTA reduces (sh holds 2 * nwarp floats). The layer norm reduces both raw moments together and keeps
// var = E[x^2] - mean^2 where var >= E[x^2] / 16 and E[x^2] is finite (at most four bits lost to cancellation);
// otherwise it recomputes the variance from the centred squares of the values in registers.
template <typename T, int VPT, bool AFFINE, bool LAYER>
__global__ __launch_bounds__(kMaxThreads) void wide_resident_fwd_kernel(
    const T* __restrict__ x, const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ y,
    float* __restrict__ mean_out, float* __restrict__ rstd_out, int64_t num_groups, int64_t cpg, float eps) {
  constexpr int kVec = kVecBytes / sizeof(T);
  extern __shared__ float sh[];
  const int tid = threadIdx.x;
  const int nwarp = blockDim.x >> 5;
  const int64_t s = blockIdx.x;
  const int num_vec = static_cast<int>(cpg / kVec);
  const T* xs = x + s * cpg;

  VecT<T, kVec> data[VPT];
  bool on[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int j = tid + i * static_cast<int>(blockDim.x);
    on[i] = j < num_vec;
    if (on[i]) {
      data[i].u = __ldcs(reinterpret_cast<const float4*>(xs + static_cast<int64_t>(j) * kVec));
    } else {
#pragma unroll
      for (int k = 0; k < kVec; ++k) data[i].v[k] = T(0);
    }
  }
  float sum = 0.f;
  float sq = 0.f;  // layer norm: the raw sum of squares
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (!on[i]) continue;
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      const float f = static_cast<float>(data[i].v[k]);
      sum += f;
      if constexpr (LAYER) sq = fmaf(f, f, sq);
    }
  }
  const float inv_n = 1.f / static_cast<float>(cpg);
  constexpr float kRawGuard = 0.0625f;
  float mean = 0.f;
  float total = 0.f;  // the variance (layer norm) or the mean square (RMS), before eps
  if constexpr (LAYER) {
    block_sum_pair(sum, sq, sh, nwarp);
    mean = sum * inv_n;
    const float m2raw = sq * inv_n;
    float var = fmaf(-mean, mean, m2raw);
    const bool bad = !(var >= kRawGuard * m2raw) || !isfinite(m2raw);
    // A CTA-wide vote, so every thread reaches the fallback's barriers. Its barrier also orders the reads of the
    // pair's partials before the fallback writes slot 0 again.
    const bool any_bad = __syncthreads_or(bad);
    if (any_bad) {
      float c2 = 0.f;
#pragma unroll
      for (int i = 0; i < VPT; ++i) {
        if (!on[i]) continue;
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
          const float d = static_cast<float>(data[i].v[k]) - mean;
          c2 = fmaf(d, d, c2);
        }
      }
      const float vc = block_sum(c2, sh, nwarp) * inv_n;
      if (bad) var = vc;
    }
    total = var;
  } else {
    float m2 = 0.f;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      if (!on[i]) continue;
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        const float d = static_cast<float>(data[i].v[k]) - mean;
        m2 += d * d;
      }
    }
    total = block_sum(m2, sh + nwarp, nwarp) * inv_n;
  }
  const float rstd = rsqrtf(total + eps);
  if (tid == 0) {
    rstd_out[s] = rstd;
    if (LAYER) mean_out[s] = mean;
  }

  T* ys = y + s * cpg;
  const int64_t c0 = (s % num_groups) * cpg;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (!on[i]) continue;
    const int j = tid + i * static_cast<int>(blockDim.x);
    const int64_t ch = c0 + static_cast<int64_t>(j) * kVec;
    VecT<T, kVec> w, b;
    if (AFFINE) {
      w.u = *reinterpret_cast<const float4*>(gamma + ch);
      if (LAYER) b.u = *reinterpret_cast<const float4*>(beta + ch);
    }
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      float o = (static_cast<float>(data[i].v[k]) - mean) * rstd;
      if (AFFINE) o *= static_cast<float>(w.v[k]);
      if (AFFINE && LAYER) o += static_cast<float>(b.v[k]);
      data[i].v[k] = static_cast<T>(o);
    }
    __stcs(reinterpret_cast<float4*>(ys + static_cast<int64_t>(j) * kVec), data[i].u);
  }
}

// The tile kernels hold a bf16 segment of 7169..8192 vectors in dynamic shared memory: the segment (cpg elements),
// then the reduction partials. A thread copies its own vectors (vector j = tid + i * blockDim.x, i < 8) with
// asynchronous copies and later reads only those, so the tile needs no barrier of its own.
//
// Waits for commit group GROUP of the affine layer norm's staging (two vectors a group, four groups) and adds its
// elements into sum and q2, in vector and element order: the sum of the first vectors runs while the rest are in
// flight.
#define WIDE_TILE_STAGED_SUM(GROUP)                                         \
  __pipeline_wait_prior(VPT / 2 - 1 - (GROUP));                             \
  _Pragma("unroll") for (int i = 2 * (GROUP); i < 2 * (GROUP) + 2; ++i) {   \
    if (!on[i]) continue;                                                   \
    const T* p = tile + base[i];                                            \
    _Pragma("unroll") for (int k = 0; k < kVec; ++k) {                      \
      const float f = static_cast<float>(p[k]);                             \
      sum += f;                                                             \
      q2 = fmaf(f, f, q2);                                                  \
    }                                                                       \
  }

// One CTA a segment, staged in shared memory and read back for the statistics (in two halves, the first summed while
// the second is in flight; four groups of two vectors on the affine layer norm), then once more for the output. The
// layer norm keeps the one-pass variance under the same guard as wide_resident_fwd_kernel, its fallback taking the
// centred squares from the tile.
template <bool AFFINE, bool LAYER>
__global__ __launch_bounds__(kMaxThreads) void wide_resident_tile_fwd_kernel(
    const bf16* __restrict__ x, const bf16* __restrict__ gamma, const bf16* __restrict__ beta, bf16* __restrict__ y,
    float* __restrict__ mean_out, float* __restrict__ rstd_out, int64_t num_groups, int64_t cpg, float eps) {
  using T = bf16;
  constexpr int kVec = kTileVec;
  constexpr int VPT = kTileVpt;
  extern __shared__ __align__(16) unsigned char smem[];
  T* const tile = reinterpret_cast<T*>(smem);
  float* const sh = reinterpret_cast<float*>(smem + static_cast<size_t>(cpg) * sizeof(T));
  const int tid = threadIdx.x;
  const int nwarp = blockDim.x >> 5;
  const int64_t s = blockIdx.x;
  const int num_vec = static_cast<int>(cpg / kVec);
  const int64_t bd = static_cast<int64_t>(blockDim.x);
  const T* xs = x + s * cpg;

  bool on[VPT];
  int64_t base[VPT];
  const float inv_n = 1.f / static_cast<float>(cpg);
  float sum = 0.f;
  float m2 = 0.f;  // RMS: the sum of squares
  constexpr float kRawGuard = 0.0625f;
  float q2 = 0.f;  // layer norm: the raw sum of squares
  constexpr int kHalf = VPT / 2;
  if constexpr (LAYER && AFFINE) {
    constexpr int kChunk = 2;
    static_assert(VPT % kChunk == 0, "the layer staging groups must divide the vectors a thread holds");
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      const int64_t j2 = static_cast<int64_t>(tid) + i * bd;
      on[i] = j2 < num_vec;
      base[i] = j2 * kVec;
      if (on[i]) {
        __pipeline_memcpy_async(tile + base[i], xs + base[i], kVec * sizeof(T));
      }
      if ((i % kChunk) == kChunk - 1) __pipeline_commit();
    }
    WIDE_TILE_STAGED_SUM(0)
    WIDE_TILE_STAGED_SUM(1)
    WIDE_TILE_STAGED_SUM(2)
    WIDE_TILE_STAGED_SUM(3)
  } else {
#pragma unroll
    for (int i = 0; i < kHalf; ++i) {
      const int64_t j2 = static_cast<int64_t>(tid) + i * bd;
      on[i] = j2 < num_vec;
      base[i] = j2 * kVec;
      if (on[i]) {
        __pipeline_memcpy_async(tile + base[i], xs + base[i], kVec * sizeof(T));
      }
    }
    __pipeline_commit();
#pragma unroll
    for (int i = kHalf; i < VPT; ++i) {
      const int64_t j2 = static_cast<int64_t>(tid) + i * bd;
      on[i] = j2 < num_vec;
      base[i] = j2 * kVec;
      if (on[i]) {
        __pipeline_memcpy_async(tile + base[i], xs + base[i], kVec * sizeof(T));
      }
    }
    __pipeline_commit();
    __pipeline_wait_prior(1);
#pragma unroll
    for (int i = 0; i < kHalf; ++i) {
      if (!on[i]) continue;
      const T* p = tile + base[i];
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        const float f = static_cast<float>(p[k]);
        if constexpr (LAYER) {
          sum += f;
          q2 = fmaf(f, f, q2);
        } else {
          m2 += f * f;
        }
      }
    }
    __pipeline_wait_prior(0);
#pragma unroll
    for (int i = kHalf; i < VPT; ++i) {
      if (!on[i]) continue;
      const T* p = tile + base[i];
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        const float f = static_cast<float>(p[k]);
        if constexpr (LAYER) {
          sum += f;
          q2 = fmaf(f, f, q2);
        } else {
          m2 += f * f;
        }
      }
    }
  }

  float mean = 0.f;
  float total = 0.f;  // the variance (layer norm) or the mean square (RMS), before eps
  if constexpr (LAYER) {
    block_sum_pair(sum, q2, sh, nwarp);
    mean = sum * inv_n;
    const float m2raw = q2 * inv_n;
    float var = fmaf(-mean, mean, m2raw);
    const bool bad = !(var >= kRawGuard * m2raw) || !isfinite(m2raw);
    // A CTA-wide vote; its barrier also orders the reads of the pair's partials before the fallback writes slot 0.
    const bool any_bad = __syncthreads_or(bad);
    if (any_bad) {
      float c2 = 0.f;
#pragma unroll
      for (int i = 0; i < VPT; ++i) {
        if (!on[i]) continue;
        const T* p = tile + base[i];
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
          const float d = static_cast<float>(p[k]) - mean;
          c2 = fmaf(d, d, c2);
        }
      }
      const float vc = block_sum(c2, sh, nwarp) * inv_n;
      if (bad) var = vc;
    }
    total = fmaxf(var, 0.f);
  } else {
    total = block_sum_once(m2, sh, nwarp) * inv_n;
  }

  const float rstd = rsqrtf(total + eps);
  if (tid == 0) {
    rstd_out[s] = rstd;
    if (LAYER) mean_out[s] = mean;
  }

  T* ys = y + s * cpg;
  const int64_t c0 = (s % num_groups) * cpg;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (!on[i]) continue;
    const T* p = tile + base[i];
    const int64_t ch = c0 + base[i];
    VecT<T, kVec> w, b;
    if (AFFINE) {
      w.u = *reinterpret_cast<const float4*>(gamma + ch);
      if (LAYER) b.u = *reinterpret_cast<const float4*>(beta + ch);
    }
    VecT<T, kVec> out;
    tile_epilogue<AFFINE, LAYER>(out, p, w, b, mean, rstd);
    __stcs(reinterpret_cast<float4*>(ys + base[i]), out.u);
  }
}

// The tile kernel's affine RMS norm for calls of one group on more than two segments an SM: a fixed grid of resident
// CTAs walks the segments with a stride of gridDim.x, and a thread keeps its gamma vectors in registers, loading them
// again only when the group changes. block_sum_once reuses the same partials every segment, so each segment ends with
// a barrier: without it a fast warp could publish the next segment's partial before a slow warp read this one's.
__global__ __launch_bounds__(kMaxThreads) void wide_resident_tile_rms_persist_fwd_kernel(
    const bf16* __restrict__ x, const bf16* __restrict__ gamma, bf16* __restrict__ y, float* __restrict__ rstd_out,
    int64_t segments, int64_t num_groups, int64_t cpg, float eps) {
  using T = bf16;
  constexpr int kVec = kTileVec;
  constexpr int VPT = kTileVpt;
  extern __shared__ __align__(16) unsigned char smem[];
  T* const tile = reinterpret_cast<T*>(smem);
  float* const sh = reinterpret_cast<float*>(smem + static_cast<size_t>(cpg) * sizeof(T));
  const int tid = threadIdx.x;
  const int nwarp = blockDim.x >> 5;
  const int num_vec = static_cast<int>(cpg / kVec);
  const int64_t bd = static_cast<int64_t>(blockDim.x);
  const float inv_n = 1.f / static_cast<float>(cpg);
  constexpr int kHalf = VPT / 2;

  bool on[VPT];
  int64_t base[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int64_t j2 = static_cast<int64_t>(tid) + i * bd;
    on[i] = j2 < num_vec;
    base[i] = j2 * kVec;
  }
  // The thread's gamma vectors, valid while cur_group is the group of the segment in hand.
  VecT<T, kVec> gw[VPT];
  int64_t cur_group = -1;

  for (int64_t s = blockIdx.x; s < segments; s += gridDim.x) {
    const T* xs = x + s * cpg;
    const float mean = 0.f;
    float m2 = 0.f;
#pragma unroll
    for (int i = 0; i < kHalf; ++i) {
      if (on[i]) {
        __pipeline_memcpy_async(tile + base[i], xs + base[i], kVec * sizeof(T));
      }
    }
    __pipeline_commit();
#pragma unroll
    for (int i = kHalf; i < VPT; ++i) {
      if (on[i]) {
        __pipeline_memcpy_async(tile + base[i], xs + base[i], kVec * sizeof(T));
      }
    }
    __pipeline_commit();
    __pipeline_wait_prior(1);
#pragma unroll
    for (int i = 0; i < kHalf; ++i) {
      if (!on[i]) continue;
      const T* p = tile + base[i];
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        const float f = static_cast<float>(p[k]);
        m2 += f * f;
      }
    }
    __pipeline_wait_prior(0);
#pragma unroll
    for (int i = kHalf; i < VPT; ++i) {
      if (!on[i]) continue;
      const T* p = tile + base[i];
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        const float f = static_cast<float>(p[k]);
        m2 += f * f;
      }
    }
    const float total = block_sum_once(m2, sh, nwarp) * inv_n;
    const float rstd = rsqrtf(total + eps);
    if (tid == 0) rstd_out[s] = rstd;

    const int64_t g = s % num_groups;
    const int64_t c0 = g * cpg;
    if (g != cur_group) {
#pragma unroll
      for (int i = 0; i < VPT; ++i) {
        if (on[i]) gw[i].u = *reinterpret_cast<const float4*>(gamma + c0 + base[i]);
      }
      cur_group = g;
    }

    T* ys = y + s * cpg;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      if (!on[i]) continue;
      const T* p = tile + base[i];
      VecT<T, kVec> out;
      tile_epilogue<true, false>(out, p, gw[i], gw[0], mean, rstd);  // gw[0]: the unused beta
      __stcs(reinterpret_cast<float4*>(ys + base[i]), out.u);
    }
    __syncthreads();
  }
}

// The tile kernel's layer norm for calls of more than two segments an SM: one resident CTA an SM walks the segments
// with a stride of gridDim.x. The partials alternate between two banks of 2 * nwarp floats from one segment to the
// next, so the loop needs no barrier of its own: a warp writes a bank again only two segments later, after the
// barrier of the segment between, which every warp passes only after reading that bank.
template <bool AFFINE>
__global__ __launch_bounds__(kMaxThreads) void wide_resident_tile_layer_persist_fwd_kernel(
    const bf16* __restrict__ x, const bf16* __restrict__ gamma, const bf16* __restrict__ beta, bf16* __restrict__ y,
    float* __restrict__ mean_out, float* __restrict__ rstd_out, int64_t segments, int64_t num_groups, int64_t cpg,
    float eps) {
  using T = bf16;
  constexpr int kVec = kTileVec;
  constexpr int VPT = kTileVpt;
  extern __shared__ __align__(16) unsigned char smem[];
  T* const tile = reinterpret_cast<T*>(smem);
  float* const sh = reinterpret_cast<float*>(smem + static_cast<size_t>(cpg) * sizeof(T));
  const int tid = threadIdx.x;
  const int nwarp = blockDim.x >> 5;
  const int num_vec = static_cast<int>(cpg / kVec);
  const int64_t bd = static_cast<int64_t>(blockDim.x);
  const float inv_n = 1.f / static_cast<float>(cpg);
  constexpr int kHalf = VPT / 2;
  constexpr float kRawGuard = 0.0625f;

  bool on[VPT];
  int64_t base[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int64_t j2 = static_cast<int64_t>(tid) + i * bd;
    on[i] = j2 < num_vec;
    base[i] = j2 * kVec;
  }

  int64_t shb = 0;  // offset of this segment's bank in sh

  for (int64_t s = blockIdx.x; s < segments; s += gridDim.x) {
    const T* xs = x + s * cpg;
    float sum = 0.f;
    float q2 = 0.f;

    if constexpr (AFFINE) {
      constexpr int kChunk = 2;
      static_assert(VPT % kChunk == 0, "the layer staging groups must divide the vectors a thread holds");
#pragma unroll
      for (int i = 0; i < VPT; ++i) {
        if (on[i]) {
          __pipeline_memcpy_async(tile + base[i], xs + base[i], kVec * sizeof(T));
        }
        if ((i % kChunk) == kChunk - 1) __pipeline_commit();
      }
      WIDE_TILE_STAGED_SUM(0)
      WIDE_TILE_STAGED_SUM(1)
      WIDE_TILE_STAGED_SUM(2)
      WIDE_TILE_STAGED_SUM(3)
    } else {
#pragma unroll
      for (int i = 0; i < kHalf; ++i) {
        if (on[i]) {
          __pipeline_memcpy_async(tile + base[i], xs + base[i], kVec * sizeof(T));
        }
      }
      __pipeline_commit();
#pragma unroll
      for (int i = kHalf; i < VPT; ++i) {
        if (on[i]) {
          __pipeline_memcpy_async(tile + base[i], xs + base[i], kVec * sizeof(T));
        }
      }
      __pipeline_commit();
      __pipeline_wait_prior(1);
#pragma unroll
      for (int i = 0; i < kHalf; ++i) {
        if (!on[i]) continue;
        const T* p = tile + base[i];
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
          const float f = static_cast<float>(p[k]);
          sum += f;
          q2 = fmaf(f, f, q2);
        }
      }
      __pipeline_wait_prior(0);
#pragma unroll
      for (int i = kHalf; i < VPT; ++i) {
        if (!on[i]) continue;
        const T* p = tile + base[i];
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
          const float f = static_cast<float>(p[k]);
          sum += f;
          q2 = fmaf(f, f, q2);
        }
      }
    }

    // The one-pass variance under the guard of wide_resident_fwd_kernel. Every thread holds the same reduced values,
    // so the test is uniform over the CTA; the warp vote only lets the compiler see that. The fallback recomputes its
    // vector offsets: keeping `on` and `base` live across it slowed the common path.
    block_sum_pair(sum, q2, sh + shb, nwarp);
    float mean = sum * inv_n;
    const float m2raw = q2 * inv_n;
    float var = fmaf(-mean, mean, m2raw);
    if (__any_sync(0xffffffffu, !(var >= kRawGuard * m2raw) || !isfinite(m2raw))) {
      float c2 = 0.f;
#pragma unroll
      for (int i = 0; i < VPT; ++i) {
        const int64_t j2 = static_cast<int64_t>(tid) + i * bd;
        if (j2 >= num_vec) continue;
        const T* p = tile + j2 * kVec;
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
          const float d = static_cast<float>(p[k]) - mean;
          c2 = fmaf(d, d, c2);
        }
      }
      __syncthreads();  // every warp has read the pair's partials before this bank is written again
      var = block_sum(c2, sh + shb, nwarp) * inv_n;
    }
    const float total = fmaxf(var, 0.f);

    const float rstd = rsqrtf(total + eps);
    if (tid == 0) {
      rstd_out[s] = rstd;
      mean_out[s] = mean;
    }

    T* ys = y + s * cpg;
    const int64_t c0 = (s % num_groups) * cpg;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      if (!on[i]) continue;
      const T* p = tile + base[i];
      const int64_t ch = c0 + base[i];
      VecT<T, kVec> w, b;
      if (AFFINE) {
        w.u = *reinterpret_cast<const float4*>(gamma + ch);
        b.u = *reinterpret_cast<const float4*>(beta + ch);
      }
      VecT<T, kVec> out;
      tile_epilogue<AFFINE, true>(out, p, w, b, mean, rstd);
      __stcs(reinterpret_cast<float4*>(ys + base[i]), out.u);
    }

    shb ^= 2 * nwarp;
  }
}

#undef WIDE_TILE_STAGED_SUM

// Raises kKernel's dynamic shared memory limit to the device's opt-in maximum, once per device, and says whether
// `bytes` fit under it. `device` is the current device, the one cudaFuncSetAttribute acts on. A refused raise is
// remembered (-1) for that device, and the call then declines.
template <auto kKernel>
bool ensure_smem(size_t bytes, int device) {
  constexpr int kMaxDevices = 64;
  static std::atomic<int> ceiling[kMaxDevices];  // 0: not raised yet
  if (device < 0 || device >= kMaxDevices) return false;
  int have = ceiling[device].load(std::memory_order_relaxed);
  if (have == 0) {
    int optin = 0;
    const bool ok = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) == cudaSuccess &&
                    optin > 0 &&
                    cudaFuncSetAttribute(reinterpret_cast<const void*>(kKernel),
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, optin) == cudaSuccess;
    have = ok ? optin : -1;
    if (!ok) (void)cudaGetLastError();  // a refusal declines the call; it is not an error of the call
    ceiling[device].store(have, std::memory_order_relaxed);
  }
  return have > 0 && bytes <= static_cast<size_t>(have);
}

// One call's operands, as the launches pass them.
template <typename T>
struct ResidentCall {
  const T* x;
  const T* gamma;
  const T* beta;
  T* y;
  float* mean;
  float* rstd;
  int64_t groups;
  int64_t cpg;
  float eps;
  int64_t segments;
  int threads;
  cudaStream_t stream;
};

template <typename T, int VPT, bool AFFINE, bool LAYER>
void launch_register_kernel(const ResidentCall<T>& r) {
  const size_t shared = sizeof(float) * 2 * (r.threads / 32);
  wide_resident_fwd_kernel<T, VPT, AFFINE, LAYER><<<r.segments, r.threads, shared, r.stream>>>(
      r.x, r.gamma, r.beta, r.y, r.mean, r.rstd, r.groups, r.cpg, r.eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The register kernel at the vectors a thread holds: vpt 2, 3..4 and 5..8 take VPT 2, 4 and 8.
template <typename T, bool AFFINE, bool LAYER>
void launch_registers(const ResidentCall<T>& r, int vpt) {
  if (vpt <= 2) launch_register_kernel<T, 2, AFFINE, LAYER>(r);
  else if (vpt <= 4) launch_register_kernel<T, 4, AFFINE, LAYER>(r);
  else launch_register_kernel<T, 8, AFFINE, LAYER>(r);
}

template <bool AFFINE, bool LAYER>
bool launch_tile_kernel(const ResidentCall<bf16>& r, size_t smem, int device) {
  if (!ensure_smem<&wide_resident_tile_fwd_kernel<AFFINE, LAYER>>(smem, device)) return false;
  wide_resident_tile_fwd_kernel<AFFINE, LAYER><<<r.segments, r.threads, smem, r.stream>>>(
      r.x, r.gamma, r.beta, r.y, r.mean, r.rstd, r.groups, r.cpg, r.eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

template <bool AFFINE>
bool launch_layer_persist(const ResidentCall<bf16>& r, int64_t workers, int device) {
  // Two banks of partials, each 2 * nwarp floats.
  const size_t smem =
      static_cast<size_t>(r.cpg) * sizeof(bf16) + sizeof(float) * 4 * static_cast<size_t>(r.threads / 32);
  if (!ensure_smem<&wide_resident_tile_layer_persist_fwd_kernel<AFFINE>>(smem, device)) return false;
  wide_resident_tile_layer_persist_fwd_kernel<AFFINE><<<static_cast<unsigned>(workers), r.threads, smem, r.stream>>>(
      r.x, r.gamma, r.beta, r.y, r.mean, r.rstd, r.segments, r.groups, r.cpg, r.eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

// The bf16 tile kernels: one CTA a segment, or one resident CTA an SM on grids above two segments an SM (layer norm,
// and the affine RMS norm of one group, whose gamma a CTA then keeps). A persistent kernel the device refuses falls
// back to the one-CTA-a-segment kernel; when that is refused too, the call declines.
bool launch_tile(const ResidentCall<bf16>& r, bool affine, bool layer, int64_t sms, int device) {
  // The segment and one bank of partials (2 * nwarp floats).
  const size_t smem =
      static_cast<size_t>(r.cpg) * sizeof(bf16) + sizeof(float) * 2 * static_cast<size_t>(r.threads / 32);
  const bool persist = r.segments > 2 * sms;  // every resident CTA walks at least two segments
  if (persist && affine && !layer && r.groups == 1 &&
      ensure_smem<&wide_resident_tile_rms_persist_fwd_kernel>(smem, device)) {
    wide_resident_tile_rms_persist_fwd_kernel<<<static_cast<unsigned>(sms), r.threads, smem, r.stream>>>(
        r.x, r.gamma, r.y, r.rstd, r.segments, r.groups, r.cpg, r.eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
  }
  if (persist && layer) {
    if (affine ? launch_layer_persist<true>(r, sms, device) : launch_layer_persist<false>(r, sms, device)) return true;
  }
  if (affine) {
    if (layer) return launch_tile_kernel<true, true>(r, smem, device);
    return launch_tile_kernel<true, false>(r, smem, device);
  }
  if (layer) return launch_tile_kernel<false, true>(r, smem, device);
  return launch_tile_kernel<false, false>(r, smem, device);
}

// Operands are contiguous (the Call contract), inputs 16-byte aligned (dispatch.cu sends any other call to the generic
// kernels) and outputs freshly allocated.
template <typename T>
bool launch(const Shape& s, const Call& c) {
  constexpr int kVec = kVecBytes / sizeof(T);
  if (s.cpg % kVec != 0) return false;
  const int64_t num_vec = s.cpg / kVec;  // 1025..8192
  const int vpt = static_cast<int>((num_vec + kMaxThreads - 1) / kMaxThreads);
  const int threads = static_cast<int>(((num_vec + vpt - 1) / vpt + 31) / 32 * 32);
  const int64_t segments = s.rows * s.groups;
  // x's device is the current device: the entry points hold a device guard. It is also the device the tile kernels'
  // shared memory opt-in (cudaFuncSetAttribute) acts on.
  const c10::DeviceIndex device = c.x_or_y->device().index();
  const int64_t sms = at::cuda::getDeviceProperties(device)->multiProcessorCount;
  // Groups of 7169..8192 vectors in bf16 may take the tile kernels.
  const bool tile_width = std::is_same_v<T, bf16> && vpt == 8;
  // One CTA a segment leaves the device idle below 2/3 of a segment an SM, where the generic kernels spread a segment
  // over a cluster of CTAs. Above 2 segments an SM the register kernel measured slower than the generic kernels; the
  // tile kernels take those grids.
  if (segments * 3 < sms * 2 || (segments > 2 * sms && !tile_width)) return false;
  const auto stream = at::cuda::getCurrentCUDAStream(device);
  const ResidentCall<T> r{c.x_or_y->const_data_ptr<T>(),
                          s.affine ? c.gamma->const_data_ptr<T>() : nullptr,
                          s.has_beta ? c.beta->const_data_ptr<T>() : nullptr,
                          c.y->data_ptr<T>(),
                          s.norm == kLayer ? c.mean_out->data_ptr<float>() : nullptr,
                          c.rstd_out->data_ptr<float>(),
                          s.groups,
                          s.cpg,
                          static_cast<float>(c.eps),
                          segments,
                          threads,
                          stream};
  const bool layer = s.norm == kLayer;
  // The tile kernels on grids of at most one segment an SM, and above two; the register kernel in between.
  if constexpr (std::is_same_v<T, bf16>) {
    if (tile_width && (segments <= sms || segments > 2 * sms))
      return launch_tile(r, s.affine, layer, sms, device);
  }
  if (s.affine) {
    if (layer) launch_registers<T, true, true>(r, vpt);
    else launch_registers<T, true, false>(r, vpt);
  } else {
    if (layer) launch_registers<T, false, true>(r, vpt);
    else launch_registers<T, false, false>(r, vpt);
  }
  return true;
}

}  // namespace

// More than 8192 channels a group in 16 bits. fp32 groups that wide take the wide kernels or the generic kernels.
bool forward_wide_resident(const Shape& s, const Call& c) {
  if (s.cpg <= 8192 || s.rows < 1) return false;
  if (s.dtype == at::kBFloat16) return launch<bf16>(s, c);
  if (s.dtype == at::kHalf) return launch<at::Half>(s, c);
  return false;
}

}  // namespace xllm::ops::blackwell
