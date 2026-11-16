// One-cluster backward of bf16 affine RMS calls from x with 2048-channel groups and 128 rows: dx and dgamma in one
// launch with no workspace and no atomics. dispatch.cu's backward() reaches it through GroupNormBwdOneClusterGrad.
//
// Grid: groups x 16 CTAs, one 16-CTA cluster a group. Rank r walks rows [8 r, 8 r + 8) in ascending order, a whole
// row a CTA: 256 threads x 8 channels (one 16-byte vector a thread); a row's sum is warp shuffles, then the eight warp
// sums in warp order. A thread keeps its 8 channels' dgamma sums in registers over the rank's rows, then writes them
// into row r of the [16][128] slab of the rank that owns those channels (rank t / 16) through distributed shared
// memory. Every slab entry has exactly one writer, so nothing is zeroed (a peer may write before the owner reaches the
// barrier). After one cluster barrier each rank adds its own slab's 16 rows in ascending rank order and stores bf16
// dgamma for its 128 channels; no rank touches remote memory after the barrier.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cooperative_groups.h>
#include <cuda_bf16.h>

#include "bwd_one_cluster_grad.h"

namespace cg = cooperative_groups;

namespace xllm::ops::blackwell {
namespace {

constexpr int kThreads = 256;    // a rank's threads
constexpr int kCh = 8;           // channels a thread; kThreads * kCh == 2048 channels a group
constexpr int kRanks = 16;       // CTAs a cluster
constexpr int kRowsRank = 8;     // 128 rows / 16 ranks
constexpr int kSlab = 128;       // channels a rank finishes (kRanks * kSlab == 2048)
constexpr int kChunks = kRanks;  // rows of a slab: one a source rank
constexpr int kWarps = kThreads / 32;

// Eight bf16 elements; the host requires 16-byte aligned bases and a 2048-channel group.
struct __align__(16) V8bf {
  __nv_bfloat16 e[8];
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

// Launched with programmatic dependent launch: the wait for the previous kernel comes before the first global read.
__global__ __launch_bounds__(kThreads) void one_cluster_grad_k(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ dy,
    const float* __restrict__ rstd, const __nv_bfloat16* __restrict__ gamma,
    __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dgamma,
    int G, int cpg, int64_t C) {
  cudaGridDependencySynchronize();
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = (int)cluster.block_rank();
  const int g = (int)(blockIdx.x / kRanks);

  // The slab is written and read as float4, so its alignment is declared (a row is 512 bytes).
  __shared__ __align__(16) float slab[kChunks][kSlab];
  __shared__ float sh[kWarps];

  const int lane_off = (int)threadIdx.x * kCh;
  const int64_t grp_base = (int64_t)g * cpg;
  const float inv_cpg = 1.f / (float)cpg;

  const V8bf vg = *reinterpret_cast<const V8bf*>(gamma + grp_base + lane_off);
  float gf[kCh];
#pragma unroll
  for (int i = 0; i < kCh; ++i) gf[i] = __bfloat162float(vg.e[i]);

  float dga[kCh];
#pragma unroll
  for (int i = 0; i < kCh; ++i) dga[i] = 0.f;

  for (int r = 0; r < kRowsRank; ++r) {
    const int64_t row = (int64_t)rank * kRowsRank + r;
    const int64_t seg = row * C + grp_base + lane_off;
    const V8bf vx = *reinterpret_cast<const V8bf*>(x + seg);
    const V8bf vd = *reinterpret_cast<const V8bf*>(dy + seg);
    const float inv = rstd[row * G + g];

    float xh[kCh], dh[kCh];
    float sa = 0.f;
#pragma unroll
    for (int i = 0; i < kCh; ++i) {
      xh[i] = __bfloat162float(vx.e[i]) * inv;
      dh[i] = __bfloat162float(vd.e[i]) * gf[i];
      sa += xh[i] * dh[i];
    }
    sa = warp_sum(sa);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = sa;
    __syncthreads();
    float A = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) A += sh[w];
    const float proj = A * inv_cpg;

    V8bf vo;
#pragma unroll
    for (int i = 0; i < kCh; ++i) {
      vo.e[i] = __float2bfloat16((dh[i] - xh[i] * proj) * inv);
      dga[i] += __bfloat162float(vd.e[i]) * xh[i];
    }
    *reinterpret_cast<V8bf*>(dx + seg) = vo;
    __syncthreads();  // sh is reused by the next row
  }

  const int dest = lane_off / kSlab;  // == threadIdx.x / 16
  float* dst = (float*)cluster.map_shared_rank((void*)slab, dest);
  float* slot = dst + rank * kSlab + (lane_off - dest * kSlab);
  *reinterpret_cast<float4*>(slot) = make_float4(dga[0], dga[1], dga[2], dga[3]);
  *reinterpret_cast<float4*>(slot + 4) = make_float4(dga[4], dga[5], dga[6], dga[7]);

  cluster.sync();

  if ((int)threadIdx.x < kSlab) {
    float acc = 0.f;
#pragma unroll
    for (int c = 0; c < kChunks; ++c) acc += slab[c][threadIdx.x];
    dgamma[grp_base + rank * kSlab + threadIdx.x] = __float2bfloat16(acc);
  }
}

bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; }

// A 16-CTA cluster is beyond the portable maximum of 8. cudaFuncSetAttribute applies to the current device, so the
// opt-in is kept per device; a refusal is remembered and declines every later call on that device, and devices from
// ordinal 64 decline.
template <typename K>
inline bool cluster_ok_on(K kernel, int dev) {
  static unsigned long long opted = 0;
  static unsigned long long failed = 0;
  if (dev < 0 || dev >= 64) return false;
  const unsigned long long bit = 1ull << dev;
  if (failed & bit) return false;
  if (opted & bit) return true;
  const cudaError_t e =
      cudaFuncSetAttribute((const void*)kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  if (e != cudaSuccess) {
    cudaGetLastError();
    failed |= bit;
    return false;
  }
  opted |= bit;
  return true;
}

// Grid groups x 16 in clusters of 16, with programmatic dependent launch; if the driver refuses that attribute, the
// cluster launch alone. False, with the error cleared, if the launch is refused.
bool launch(const at::Tensor& dy, const at::Tensor& x, const at::Tensor& rstd, const at::Tensor& gamma,
            int64_t groups, int64_t cpg, const at::Tensor& dx, const at::Tensor& dgamma) {
  const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  if (!cluster_ok_on(one_cluster_grad_k, (int)stream.device_index())) return false;

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3((unsigned)(groups * kRanks));
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream.stream();
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = kRanks;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;

  auto launch_once = [&]() {
    return cudaLaunchKernelEx(&cfg, one_cluster_grad_k, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                              reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr()), rstd.data_ptr<float>(),
                              reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
                              reinterpret_cast<__nv_bfloat16*>(dx.data_ptr()),
                              reinterpret_cast<__nv_bfloat16*>(dgamma.data_ptr()), (int)groups, (int)cpg,
                              groups * cpg);
  };
  cudaError_t e = launch_once();
  if (e != cudaSuccess) {
    (void)cudaGetLastError();
    cfg.numAttrs = 1;
    e = launch_once();
  }
  if (e != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();  // an asynchronous launch failure still returns cudaSuccess above
  return true;
}

}  // namespace

bool GroupNormBwdOneClusterGrad(const at::Tensor& dy, const at::Tensor& x_or_y, const at::Tensor& rstd,
                                const at::Tensor& gamma, int64_t channels, int64_t groups, bool layer,
                                bool memory_efficient, const at::Tensor& dx, const at::Tensor& dgamma) {
  if (layer || memory_efficient) return false;  // RMS from x only
  if (dy.scalar_type() != at::kBFloat16 || x_or_y.scalar_type() != at::kBFloat16) return false;
  if (gamma.scalar_type() != at::kBFloat16 || dx.scalar_type() != at::kBFloat16) return false;
  if (dgamma.scalar_type() != at::kBFloat16) return false;
  if (gamma.numel() != channels) return false;
  if (groups <= 0 || channels <= 0 || channels % groups != 0) return false;
  const int64_t cpg = channels / groups;
  if (cpg != (int64_t)kThreads * kCh) return false;
  const int64_t rows = dy.numel() / channels;
  if (rows != (int64_t)kRanks * kRowsRank) return false;
  if (x_or_y.numel() != dy.numel() || dx.numel() != dy.numel()) return false;
  if (rstd.scalar_type() != at::kFloat || !rstd.is_contiguous() || rstd.numel() != rows * groups) return false;
  if (!dy.is_contiguous() || !x_or_y.is_contiguous() || !dx.is_contiguous()) return false;
  if (!gamma.is_contiguous() || !dgamma.is_contiguous() || dgamma.numel() != channels) return false;
  if (!aligned16(dy.data_ptr()) || !aligned16(x_or_y.data_ptr()) || !aligned16(dx.data_ptr()) ||
      !aligned16(gamma.data_ptr()) || !aligned16(dgamma.data_ptr()))
    return false;
  return launch(dy, x_or_y, rstd, gamma, groups, cpg, dx, dgamma);
}

}  // namespace xllm::ops::blackwell
