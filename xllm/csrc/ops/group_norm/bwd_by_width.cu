// By-width backward of the group RMS norm and group layer norm for bf16 calls without an affine transform and with
// wide groups: dx in one pass, x_or_y and dy read once and dx written once. dispatch.cu's backward() reaches it through
// GroupNormBwdByWidth, which also answers for the affine calls of the widths it claims (they go to the generic
// kernels).
//
// gn_bwd_wide serves 2048 < cpg <= 8192 with at least 2048 rows (dispatch.cu's backward() sends single-group
// 8192-channel calls): a CTA keeps one group and walks the rows two at a time, two 16-byte vectors of 8 elements a
// thread, staged in shared memory. Each row's two sums are reduced in a fixed order (shuffles, then the warp partials
// in warp order), and nothing is summed across CTAs, so the result does not depend on the grid.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

#include <algorithm>

#include "bwd_by_width.h"
#include "width_common.cuh"

namespace xllm::ops::blackwell {
namespace {

using namespace width;

constexpr int kVec = 8;  // 2-byte elements in a 16-byte vector

// 16-byte global -> shared copy that holds no register (cp.async.cg caches in L2 only).
__device__ __forceinline__ void cp16_async(void* dst_shared, const void* src_global) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst_shared));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src_global) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
// Wait until at most kN of this thread's commit groups are in flight.
template <int kN>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kN) : "memory");
}

// A launch of fewer than 2048 rows may carry programmatic dependent launch, and the kernel then waits for its
// predecessor before its first global read. This file launches only longer calls, without the attribute.
__device__ __forceinline__ bool UsePdl(int64_t rows) { return rows < 2048; }

// One group a CTA: CTA b keeps group b % G and takes rows b / G, b / G + nslice, ... two at a time (a lone last row
// is paired with itself at weight 0 and stored once). Thread t holds vectors t, t + blockDim.x, ... (Q of them) of the
// group; a pass copies them for both rows, x and dy, into shared memory with cp.async, and every thread reads back only
// its own slots, so the stage needs no barrier. A row's sums are warp shuffles, then every warp adds the warp partials
// in warp order; the partials are double-buffered by pass parity, so one barrier a pass suffices. The kernel takes the
// operands of the affine backward too; gamma, beta, dgamma and dbeta are never read.
template <typename T, int Q, bool LAYER, bool MEMEFF>
__global__ void gn_bwd_wide(const T* __restrict__ x_or_y, const T* __restrict__ dy, const float* __restrict__ mean,
                            const float* __restrict__ rstd, const T* __restrict__ /*gamma*/,
                            const T* __restrict__ /*beta*/, T* __restrict__ dx, float* __restrict__ /*dgamma*/,
                            float* __restrict__ /*dbeta*/, int64_t rows, int G, int64_t cpg) {
  if (UsePdl(rows)) cudaGridDependencySynchronize();
  const int gi = blockIdx.x % G;
  const int slice = blockIdx.x / G;
  const int nslice = gridDim.x / G;
  const int64_t gi_cpg = static_cast<int64_t>(gi) * cpg;
  const int64_t col = static_cast<int64_t>(threadIdx.x) * kVec;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * kVec;
  const float inv_cpg = 1.f / static_cast<float>(cpg);
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int nwarp = (blockDim.x + 31) >> 5;
  __shared__ float sh1[2][33], sh2[2][33], sh3[2][33], sh4[2][33];
  extern __shared__ __align__(16) char wide_stage[];  // x and dy of both rows, Q vectors a thread each
  const int nt = static_cast<int>(blockDim.x);
  const int tid = static_cast<int>(threadIdx.x);
  Vec<T, kVec>* sxa = nullptr;
  Vec<T, kVec>* sda = nullptr;
  Vec<T, kVec>* sxb = nullptr;
  Vec<T, kVec>* sdb = nullptr;

  const int64_t C = static_cast<int64_t>(G) * cpg;
  int64_t r = slice;
  if (r >= rows) return;
  for (int64_t it = 0; r < rows; r += 2 * nslice, ++it) {
    const int buf = static_cast<int>(it & 1);
    const int64_t r0 = r;
    const bool two = (r + nslice < rows);
    const int64_t r1 = two ? (r + nslice) : r;
    const float w1 = two ? 1.f : 0.f;

    Vec<T, kVec>* const stg = reinterpret_cast<Vec<T, kVec>*>(wide_stage);
    sxa = stg;
    sda = sxa + Q * nt;
    sxb = sda + Q * nt;
    sdb = sxb + Q * nt;
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      const int64_t off = col + q * stride;
      if (off < cpg) {
        const int64_t at0 = r0 * C + gi_cpg + off;
        cp16_async(&sxa[q * nt + tid], x_or_y + at0);
        cp16_async(&sda[q * nt + tid], dy + at0);
      }
    }
    cp_commit();
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      const int64_t off = col + q * stride;
      if (off < cpg) {
        const int64_t at1 = r1 * C + gi_cpg + off;
        cp16_async(&sxb[q * nt + tid], x_or_y + at1);
        cp16_async(&sdb[q * nt + tid], dy + at1);
      }
    }
    cp_commit();
    cp_wait<0>();  // both rows are consumed by the same loop

    const float rsa = rstd[r0 * G + gi], rsb = rstd[r1 * G + gi];
    float mua = 0.f, mub = 0.f;
    if constexpr (LAYER) {
      mua = mean[r0 * G + gi];
      mub = mean[r1 * G + gi];
    }
    // The layer norm's xhat is one fma: (y - mu) * rs == y * rs + kma, with kma = -mu * rs.
    const float kma = -mua * rsa, kmb = -mub * rsb;
    float s1a = 0.f, s2a = 0.f, s1b = 0.f, s2b = 0.f;
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      const int64_t off = col + q * stride;
      if (off < cpg) {
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          const float ya = to_f(sxa[q * nt + tid].v[i]), yb = to_f(sxb[q * nt + tid].v[i]);
          const float dva = to_f(sda[q * nt + tid].v[i]), dvb = to_f(sdb[q * nt + tid].v[i]);
          float ha, hb;
          if constexpr (MEMEFF) {
            ha = ya;
            hb = yb;
          } else if constexpr (LAYER) {
            ha = __fmaf_rn(ya, rsa, kma);
            hb = __fmaf_rn(yb, rsb, kmb);
          } else {
            ha = ya * rsa;
            hb = yb * rsb;
          }
          const float dha = dva;
          const float dhb = dvb * w1;
          s1a += dha * ha;
          s1b += dhb * hb;
          if constexpr (LAYER) {
            s2a += dha;
            s2b += dhb;
          }
        }
      }
    }
    s1a = warp_sum(s1a);
    s2a = warp_sum(s2a);
    s1b = warp_sum(s1b);
    s2b = warp_sum(s2b);
    if (lane == 0) {
      sh1[buf][wid] = s1a;
      sh2[buf][wid] = s2a;
      sh3[buf][wid] = s1b;
      sh4[buf][wid] = s2b;
    }
    __syncthreads();
    float c1a = 0.f, c2a = 0.f, c1b = 0.f, c2b = 0.f;
    for (int w = 0; w < nwarp; ++w) {
      c1a += sh1[buf][w];
      c2a += sh2[buf][w];
      c1b += sh3[buf][w];
      c2b += sh4[buf][w];
    }
    c1a *= inv_cpg;
    c2a *= inv_cpg;
    c1b *= inv_cpg;
    c2b *= inv_cpg;
    // The layer norm's dx is two fmas: (dh - h * c1 - c2) * rs == h * nq1 + (dh * rs + nq3).
    const float nq1a = -c1a * rsa, nq1b = -c1b * rsb;
    const float nq3a = -c2a * rsa, nq3b = -c2b * rsb;

    const int64_t base0 = r0 * C + gi_cpg, base1 = r1 * C + gi_cpg;
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      const int64_t off = col + q * stride;
      if (off < cpg) {
        Vec<T, kVec> oa, ob;
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          const float ya = to_f(sxa[q * nt + tid].v[i]), yb = to_f(sxb[q * nt + tid].v[i]);
          const float dva = to_f(sda[q * nt + tid].v[i]), dvb = to_f(sdb[q * nt + tid].v[i]);
          float ha, hb;
          if constexpr (MEMEFF) {
            ha = ya;
            hb = yb;
          } else if constexpr (LAYER) {
            ha = __fmaf_rn(ya, rsa, kma);
            hb = __fmaf_rn(yb, rsb, kmb);
          } else {
            ha = ya * rsa;
            hb = yb * rsb;
          }
          const float dha = dva;
          const float dhb = dvb;
          if constexpr (LAYER) {
            from_f(oa.v[i], __fmaf_rn(ha, nq1a, __fmaf_rn(dha, rsa, nq3a)));
            from_f(ob.v[i], __fmaf_rn(hb, nq1b, __fmaf_rn(dhb, rsb, nq3b)));
          } else {
            from_f(oa.v[i], (dha - ha * c1a) * rsa);
            from_f(ob.v[i], (dhb - hb * c1b) * rsb);
          }
        }
        store_stream<T, kVec>(dx + base0 + off, oa);
        if (two) store_stream<T, kVec>(dx + base1 + off, ob);
      }
    }
  }
}

// ---------------------------------------------------------------- launch

int sm_count() {
  static int cached = [] {
    int n = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    return n > 0 ? n : 148;
  }();
  return cached;
}

template <typename T>
struct Operands {
  const T* x_or_y;
  const T* dy;
  const float* mean;  // null for RMS
  const float* rstd;
  T* dx;
  int64_t rows, G, cpg;
  cudaStream_t stream;
  int device;
};

// The kernel's dynamic shared memory ceiling, set once a device: 64 KiB, the stage of the widest group (8192 channels:
// 1024 vectors, four slabs).
template <typename T, int Q, bool LAYER, bool MEMEFF>
void opt_in_stage(int dev) {
  static unsigned long long opted = 0;  // bit d: set on device d
  const unsigned long long bit = dev < 64 ? (1ull << dev) : 0ull;
  if (opted & bit) return;
  const int bytes = 4 * 1024 * static_cast<int>(sizeof(Vec<T, kVec>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(gn_bwd_wide<T, Q, LAYER, MEMEFF>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      bytes));
  opted |= bit;
}

// The grid is G groups x nslice row slices. nslice starts from the thread slots (2048 a SM) and is capped at one wave
// of resident blocks when it would run two or more; then it becomes one slice a row pair while that stays within four
// waves, so that no CTA carries a weight-0 partner row; with several groups the slices never exceed the row pairs.
template <typename T, int Q, bool LAYER, bool MEMEFF>
void launch_wide(const Operands<T>& o) {
  const int64_t per = (o.cpg / kVec + Q - 1) / Q;  // vectors a thread
  const unsigned threads = static_cast<unsigned>(std::min<int64_t>(1024, ((per + 31) / 32) * 32));
  const int per_sm = std::max(1, static_cast<int>(2048 / threads));
  const size_t stage = static_cast<size_t>(4) * Q * threads * sizeof(Vec<T, kVec>);
  opt_in_stage<T, Q, LAYER, MEMEFF>(o.device);

  int64_t nslice = std::max<int64_t>(1, static_cast<int64_t>(sm_count()) * per_sm / o.G);
  static int occupancy[33];  // resident blocks a SM, by threads / 32; 0 until queried
  int& resident = occupancy[std::min<unsigned>(threads / 32, 32u)];
  if (resident <= 0) {
    int r = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r, gn_bwd_wide<T, Q, LAYER, MEMEFF>, threads, stage);
    resident = (r > 0) ? r : 1;
  }
  const int64_t one_wave = static_cast<int64_t>(sm_count()) * resident / o.G;
  if (one_wave > 0 && nslice >= 2 * one_wave) nslice = one_wave;
  const int64_t pairs = (o.rows + 1) / 2;
  if (pairs > nslice && one_wave > 0 && pairs <= 4 * one_wave) nslice = pairs;
  if (o.G > 1 && o.rows < nslice) nslice = std::max<int64_t>(1, (o.rows + 1) / 2);
  gn_bwd_wide<T, Q, LAYER, MEMEFF><<<static_cast<unsigned>(nslice * o.G), threads, stage, o.stream>>>(
      o.x_or_y, o.dy, o.mean, o.rstd, nullptr, nullptr, o.dx, nullptr, nullptr, o.rows, static_cast<int>(o.G), o.cpg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// More than 256 vectors a group (two a thread) and at least 2048 rows; false for anything else.
template <typename T, bool LAYER, bool MEMEFF>
bool launch_mode(const Operands<T>& o) {
  if (o.cpg <= 2048 || o.rows < 2048) return false;
  launch_wide<T, 2, LAYER, MEMEFF>(o);
  return true;
}

}  // namespace

ByWidthRoute GroupNormBwdByWidth(int64_t norm, const at::Tensor& x_or_y, const at::Tensor& dy, const at::Tensor& mean,
                                 const at::Tensor& rstd, const at::Tensor& gamma, const at::Tensor& beta,
                                 bool memory_efficient, at::Tensor& dx, at::Tensor& dgamma, at::Tensor& dbeta) {
  const int64_t channels = x_or_y.size(-1);
  const int64_t rows = x_or_y.numel() / channels;
  const int64_t G = rstd.size(-1);
  const int64_t cpg = channels / G;
  const bool affine = gamma.numel() > 0;
  const bool has_beta = affine && norm != 0;

  // The widths this file claims (40 and 80 channels only with an affine transform), and the dtype and layout it needs.
  const bool claimed = cpg > 256 || (affine && (cpg == 40 || cpg == 80));
  const bool covered = claimed && cpg % 8 == 0 && cpg <= 8192 && x_or_y.scalar_type() == at::kBFloat16 &&
                       x_or_y.is_contiguous() && dy.is_contiguous() && rstd.is_contiguous() && dx.is_contiguous() &&
                       rstd.numel() == rows * G && (!affine || gamma.is_contiguous()) &&
                       (!has_beta || beta.is_contiguous()) && (norm == 0 || memory_efficient || mean.is_contiguous()) &&
                       (!affine || (dgamma.is_contiguous() && (!has_beta || dbeta.is_contiguous())));
  if (!covered) return ByWidthRoute::NotCovered;
  if (affine) return ByWidthRoute::NeedsFixedReduction;

  using T = __nv_bfloat16;
  const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  const Operands<T> o{static_cast<const T*>(x_or_y.data_ptr()), static_cast<const T*>(dy.data_ptr()),
                      norm != 0 ? mean.data_ptr<float>() : nullptr, rstd.data_ptr<float>(),
                      static_cast<T*>(dx.data_ptr()), rows, G, cpg, stream.stream(),
                      static_cast<int>(stream.device_index())};
  bool launched;
  if (norm != 0) {
    launched = memory_efficient ? launch_mode<T, true, true>(o) : launch_mode<T, true, false>(o);
  } else {
    launched = memory_efficient ? launch_mode<T, false, true>(o) : launch_mode<T, false, false>(o);
  }
  return launched ? ByWidthRoute::Launched : ByWidthRoute::NotCovered;
}

}  // namespace xllm::ops::blackwell
