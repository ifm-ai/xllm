// Resident wide-row backward of the group RMS norm and group layer norm for bf16 calls without an affine transform and
// 65536-channel groups: each (row, group) segment is held by one 4-CTA cluster, read once and written once, in one
// launch with no workspace. dispatch.cu's backward() reaches it through backward_wide_rows.
//
// dx_j = rstd * (d_j - xhat_j * A / cpg - B / cpg) needs the segment's sums A = sum d_j * xhat_j and, for the layer
// norm, B = sum d_j. Cluster rank r owns vectors [2048 r, 2048 r + 2048) of the segment (16 bytes, 8 channels each),
// thread t vectors 2048 r + t + 512 i (i < 4). Each CTA reduces its slab's sums (warp shuffles, then warp 0 adds the
// warp partials in order) and writes them into slot `rank` of every rank's shared memory; after one cluster barrier
// each CTA adds the four slots in ascending rank order, so every CTA has the same totals and no CTA reads remote memory
// after the barrier.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cstdint>

#include "dispatch.h"

namespace xllm::ops::blackwell {
namespace {

constexpr int kThreads = 512;
constexpr int kVecBytes = 16;
constexpr int kVecsPerThread = 4;
constexpr int kMaxBlocks = 4;  // CTAs a cluster, and a segment

namespace cg = cooperative_groups;

template <typename T, int kVec>
union alignas(kVecBytes) VecT {
  float4 u;
  float2 u2[kVec / 4 > 1 ? kVec / 4 : 1];
  T v[kVec];
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// The block's total, to every thread; slot[0] holds it until the caller is done.
__device__ __forceinline__ float block_sum(float v, float* slot) {
  constexpr int kNwarp = kThreads / 32;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) slot[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float t = (lane < kNwarp) ? slot[lane] : 0.f;
    t = warp_sum(t);
    if (lane == 0) slot[0] = t;
  }
  __syncthreads();
  return slot[0];
}

// Both layer-norm sums in one reduction (the same shuffle trees as block_sum, one round trip): slots [0, kNwarp) and
// [kNwarp] for A, then the same for B.
__device__ __forceinline__ void block_sum2(float& a, float& b, float* slot) {
  constexpr int kNwarp = kThreads / 32;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  a = warp_sum(a);
  b = warp_sum(b);
  if (lane == 0) {
    slot[warp] = a;
    slot[kNwarp + 1 + warp] = b;
  }
  __syncthreads();
  if (warp == 0) {
    float ta = (lane < kNwarp) ? slot[lane] : 0.f;
    float tb = (lane < kNwarp) ? slot[kNwarp + 1 + lane] : 0.f;
    ta = warp_sum(ta);
    tb = warp_sum(tb);
    if (lane == 0) {
      slot[kNwarp] = ta;
      slot[2 * kNwarp + 1] = tb;
    }
  }
  __syncthreads();
  a = slot[kNwarp];
  b = slot[2 * kNwarp + 1];
}

// 16-byte global -> shared copy that holds no register (cp.async.cg caches in L2 only).
__device__ __forceinline__ void cp16_async(void* dst_shared, const void* src_global) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst_shared));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src_global));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
// Wait until at most kN of this thread's commit groups are in flight.
template <int kN>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kN));
}

// STAGED copies the slab into shared memory with cp.async (vectors 0-1 of both tensors, then 2-3, the first group
// consumed while the second is in flight) instead of holding it in registers; a thread reads back only its own slots,
// so the stage needs no barrier. x_hat (and dy) are stored back in T and the dx pass reads them from there, so x_hat
// is rounded to T once, as the generic kernels round their x.
template <typename T, bool LAYER, bool ME, bool STAGED>
__global__ __launch_bounds__(kThreads, 2) void wide_row_bwd_kernel(
    const T* __restrict__ dy, const T* __restrict__ x_or_y, const float* __restrict__ mean,
    const float* __restrict__ rstd, T* __restrict__ dx, int64_t cpg) {
  cudaGridDependencySynchronize();  // before the first global read
  constexpr int kVec = kVecBytes / sizeof(T);
  constexpr int kPerBlockVec = kThreads * kVecsPerThread;
  constexpr int kNwarp = kThreads / 32;
  extern __shared__ __align__(16) float sh[];
  // sh: the block reduction's slots (A, then B), the cluster's (A, B) slots, the folded pair, then (STAGED) the stage
  // from a 16-byte boundary.
  float* warp_a = sh;
  float* result = sh + 2 * (kNwarp + 1) + 2;
  constexpr int kScratchFloats = 2 * (kNwarp + 1) + 4 + 2 * kMaxBlocks;
  VecT<T, kVec>* const stage = reinterpret_cast<VecT<T, kVec>*>(sh + ((kScratchFloats + 3) & ~3));
  VecT<T, kVec>* const sdy = stage;  // [kVecsPerThread][kThreads]
  VecT<T, kVec>* const sx = stage + kVecsPerThread * kThreads;

  const cg::cluster_group cluster = cg::this_cluster();
  const unsigned rank = cluster.block_rank();
  const unsigned nblocks = cluster.num_blocks();
  const int64_t segment = static_cast<int64_t>(blockIdx.x) / nblocks;
  // The host launches only whole clusters of full CTAs, so every vector is in range; the mask keeps the kernel safe
  // on its own.
  const int num_vec = static_cast<int>(cpg / kVec);
  const int base = static_cast<int>(rank) * kPerBlockVec;
  const T* dys = dy + segment * cpg;
  const T* xs = x_or_y + segment * cpg;
  const float rs = rstd[segment];
  float mu = 0.f;
  if constexpr (LAYER) mu = mean[segment];

  static_assert(kVecsPerThread == 4, "the stage is two commit groups of two vectors");
  bool on[kVecsPerThread];
  VecT<T, kVec> d[kVecsPerThread];   // the unstaged slab
  VecT<T, kVec> xv[kVecsPerThread];
  float sum_a = 0.f, sum_b = 0.f;
  if constexpr (STAGED) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int j = base + threadIdx.x + i * kThreads;
      on[i] = j < num_vec;
      if (on[i]) {
        cp16_async(&sdy[i * kThreads + threadIdx.x], dys + static_cast<int64_t>(j) * kVec);
        cp16_async(&sx[i * kThreads + threadIdx.x], xs + static_cast<int64_t>(j) * kVec);
      }
    }
    cp_commit();
#pragma unroll
    for (int i = 2; i < kVecsPerThread; ++i) {
      const int j = base + threadIdx.x + i * kThreads;
      on[i] = j < num_vec;
      if (on[i]) {
        cp16_async(&sdy[i * kThreads + threadIdx.x], dys + static_cast<int64_t>(j) * kVec);
        cp16_async(&sx[i * kThreads + threadIdx.x], xs + static_cast<int64_t>(j) * kVec);
      }
    }
    cp_commit();

#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      if (i == 0) cp_wait<1>();  // group 0 has landed; group 1 stays in flight
      if (i == 2) cp_wait<0>();  // group 1 has landed
      if (!on[i]) continue;
      const int si = i * kThreads + threadIdx.x;
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        const float dh = static_cast<float>(sdy[si].v[k]);
        float xh;
        if constexpr (ME) {
          xh = static_cast<float>(sx[si].v[k]);
        } else {
          xh = (static_cast<float>(sx[si].v[k]) - mu) * rs;
        }
        sdy[si].v[k] = static_cast<T>(dh);
        sx[si].v[k] = static_cast<T>(xh);
        sum_a += dh * xh;
        if constexpr (LAYER) sum_b += dh;
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      const int j = base + threadIdx.x + i * kThreads;
      on[i] = j < num_vec;
      if (on[i]) {
        d[i].u = __ldcs(reinterpret_cast<const float4*>(dys + static_cast<int64_t>(j) * kVec));
        xv[i].u = __ldcs(reinterpret_cast<const float4*>(xs + static_cast<int64_t>(j) * kVec));
      }
    }

#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      if (!on[i]) continue;
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        const float dh = static_cast<float>(d[i].v[k]);
        float xh;
        if constexpr (ME) {
          xh = static_cast<float>(xv[i].v[k]);
        } else {
          xh = (static_cast<float>(xv[i].v[k]) - mu) * rs;
        }
        d[i].v[k] = static_cast<T>(dh);
        xv[i].v[k] = static_cast<T>(xh);
        sum_a += dh * xh;
        if constexpr (LAYER) sum_b += dh;
      }
    }
  }
  // Every global read of this kernel is done: an armed successor may launch now. Its cudaGridDependencySynchronize()
  // still waits for this whole grid to complete and its writes to be visible, so the dx stores below are covered.
  cudaTriggerProgrammaticLaunchCompletion();
  float tot_a = sum_a, tot_b = sum_b;
  if constexpr (LAYER) {
    block_sum2(tot_a, tot_b, warp_a);
  } else {
    tot_a = block_sum(sum_a, warp_a);
  }
  // Thread 0 writes this CTA's (A, B) into slot `rank` of every rank's shared memory; after cluster.sync each CTA reads
  // only its own slots.
  float* slots = sh + 2 * (kNwarp + 1) + 4;
  if (threadIdx.x == 0) {
    for (unsigned r = 0; r < nblocks; ++r) {
      float* p = cluster.map_shared_rank(slots, r);
      if constexpr (LAYER) {
        reinterpret_cast<float2*>(p)[rank] = make_float2(tot_a, tot_b);
      } else {
        p[2 * rank + 0] = tot_a;
        p[2 * rank + 1] = tot_b;
      }
    }
  }
  cluster.sync();
  if (threadIdx.x == 0) {
    float a_sum = 0.f, b_sum = 0.f;
    for (unsigned r = 0; r < nblocks; ++r) {
      if constexpr (LAYER) {
        const float2 t = reinterpret_cast<const float2*>(slots)[r];
        a_sum += t.x;
        b_sum += t.y;
      } else {
        a_sum += slots[2 * r];
        b_sum += slots[2 * r + 1];
      }
    }
    result[0] = a_sum;
    result[1] = b_sum;
  }
  __syncthreads();

  const float inv_n = 1.f / static_cast<float>(cpg);
  const float mean_a = result[0] * inv_n;
  const float mean_b = (LAYER ? result[1] : 0.f) * inv_n;
  T* dxs = dx + segment * cpg;
  if constexpr (STAGED) {
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      if (!on[i]) continue;
      const int j = base + threadIdx.x + i * kThreads;
      const int si = i * kThreads + threadIdx.x;
      VecT<T, kVec> out;
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        const float dh = static_cast<float>(sdy[si].v[k]);
        const float xh = static_cast<float>(sx[si].v[k]);
        out.v[k] = static_cast<T>(rs * (dh - xh * mean_a - mean_b));
      }
      *reinterpret_cast<float4*>(dxs + static_cast<int64_t>(j) * kVec) = out.u;
    }
  } else {
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      if (!on[i]) continue;
      const int j = base + threadIdx.x + i * kThreads;
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        const float dh = static_cast<float>(d[i].v[k]);
        const float xh = static_cast<float>(xv[i].v[k]);
        const float o = rs * (dh - xh * mean_a - mean_b);
        d[i].v[k] = static_cast<T>(o);
      }
      *reinterpret_cast<float4*>(dxs + static_cast<int64_t>(j) * kVec) = d[i].u;
    }
  }
}

// Dynamic shared memory: the float scratch, plus the stage (two tensors, kVecsPerThread vectors a thread) when staged.
constexpr size_t kScratchBytes = sizeof(float) * ((2 * (kThreads / 32 + 1) + 4 + 2 * kMaxBlocks + 3) & ~size_t{3});
constexpr size_t kStagedBytes = kScratchBytes + size_t{2} * kVecsPerThread * kThreads * kVecBytes;

// Launches one form; the staged form opts in to its shared memory once a device (a refusal throws: the launch could
// not run either). If the driver refuses the launch, it is tried again without the programmatic-dependent-launch
// attribute (the cluster dimension stays).
template <typename T, bool LAYER, bool ME, bool STAGED>
void launch_form(cudaLaunchConfig_t& config, int dev, const T* dy, const T* x_or_y, const float* mean,
                 const float* rstd, T* dx, int64_t cpg) {
  auto kernel = wide_row_bwd_kernel<T, LAYER, ME, STAGED>;
  config.dynamicSmemBytes = STAGED ? kStagedBytes : kScratchBytes;
  if constexpr (STAGED) {
    static unsigned long long opted = 0;  // bit d: set on device d
    const unsigned long long bit = dev < 64 ? (1ull << dev) : 0ull;
    if (!(opted & bit)) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          static_cast<int>(kStagedBytes)));
      opted |= bit;
    }
  }
  if (cudaLaunchKernelEx(&config, kernel, dy, x_or_y, mean, rstd, dx, cpg) != cudaSuccess) {
    (void)cudaGetLastError();
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, dy, x_or_y, mean, rstd, dx, cpg));
  }
}

template <typename T, bool LAYER, bool ME>
void launch_mode(bool staged, cudaLaunchConfig_t& config, int dev, const T* dy, const T* x_or_y, const float* mean,
                 const float* rstd, T* dx, int64_t cpg) {
  if (staged) launch_form<T, LAYER, ME, true>(config, dev, dy, x_or_y, mean, rstd, dx, cpg);
  else launch_form<T, LAYER, ME, false>(config, dev, dy, x_or_y, mean, rstd, dx, cpg);
}

// Grid: segments x kMaxBlocks CTAs in clusters of kMaxBlocks, with programmatic dependent launch. The stage pays for
// itself from 512 rows.
template <typename T>
void launch(const Shape& s, const Call& c) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(static_cast<unsigned>(s.rows * s.groups * kMaxBlocks));
  config.blockDim = dim3(kThreads);
  config.stream = stream;
  cudaLaunchAttribute attribute[2] = {};
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = kMaxBlocks;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;
  attribute[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[1].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attribute;
  config.numAttrs = 2;

  const int dev = static_cast<int>(stream.device_index());
  const T* dy = c.dy->data_ptr<T>();
  const T* x_or_y = c.x_or_y->data_ptr<T>();
  const float* mean = c.mean->numel() > 0 ? c.mean->data_ptr<float>() : nullptr;
  const float* rstd = c.rstd->data_ptr<float>();
  T* dx = c.dx->data_ptr<T>();
  const bool layer = s.norm == kLayer;
  const bool me = s.memory_efficient;
  const bool staged = s.rows >= 512;
  if (layer) {
    if (me) launch_mode<T, true, true>(staged, config, dev, dy, x_or_y, mean, rstd, dx, s.cpg);
    else launch_mode<T, true, false>(staged, config, dev, dy, x_or_y, mean, rstd, dx, s.cpg);
  } else {
    if (me) launch_mode<T, false, true>(staged, config, dev, dy, x_or_y, mean, rstd, dx, s.cpg);
    else launch_mode<T, false, false>(staged, config, dev, dy, x_or_y, mean, rstd, dx, s.cpg);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

bool backward_wide_rows(const Shape& s, const Call& c) {
  using T = at::BFloat16;
  // Exactly kMaxBlocks full CTAs a segment: 4 x 512 threads x 4 vectors x 8 channels.
  constexpr int64_t kCpg = int64_t{kMaxBlocks} * kThreads * kVecsPerThread * (kVecBytes / (int)sizeof(T));
  if (s.affine || s.dtype != at::kBFloat16) return false;
  if (s.cpg != kCpg) return false;
  if (s.rows < 1 || s.groups < 1) return false;
  auto aligned = [](const at::Tensor& t) {
    return t.is_contiguous() && (reinterpret_cast<uintptr_t>(t.data_ptr()) % kVecBytes) == 0;
  };
  if (!aligned(*c.dy) || !aligned(*c.x_or_y) || !aligned(*c.dx) || !c.rstd->is_contiguous()) return false;
  if (s.norm == kLayer && !c.mean->is_contiguous()) return false;
  launch<T>(s, c);
  return true;
}

}  // namespace xllm::ops::blackwell
