// One-cluster backward of bf16 affine calls with 512-channel groups and 128 rows, from x or from y: dx, dgamma and
// dbeta in one launch with no workspace and no atomics. dispatch.cu's backward() reaches it through
// GroupNormBwdOneClusterGrad512.
//
// Grid: groups x 16 CTAs, one 16-CTA cluster a group. Rank r owns rows [8 r, 8 r + 8), one warp a row: 32 lanes x 16
// channels (two 16-byte vectors a lane), so a row's sums are one warp butterfly. The rank adds its eight rows'
// dgamma/dbeta contributions in ascending row order (through shared memory); thread t then owns channels
// [2 t, 2 t + 2), which lie in the 32-channel slab of rank t / 16, and writes them into row r of that rank's [16][32]
// slab through distributed shared memory. Every slab entry has exactly one writer, so nothing is zeroed (a peer may
// write before the owner reaches the barrier). After one cluster barrier each rank adds its own slab's 16 rows in
// ascending rank order and stores bf16 dgamma (and dbeta); no rank touches remote memory after the barrier.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cooperative_groups.h>
#include <cuda_bf16.h>

#include "bwd_one_cluster_grad512.h"

namespace cg = cooperative_groups;

namespace xllm::ops::blackwell {
namespace {

constexpr int kThreads = 256;    // 8 warps
constexpr int kWarps = 8;        // warps a rank == rows a rank
constexpr int kRanks = 16;       // CTAs a cluster
constexpr int kRowsRank = 8;     // 128 rows / 16 ranks
constexpr int kCpg = 512;        // channels a group
constexpr int kChLane = 16;      // channels a lane == kCpg / 32
constexpr int kSlab = 32;        // channels a rank finishes == kCpg / kRanks
constexpr int kChunks = kRanks;  // rows of a slab: one a source rank

// Shared memory: [kWarps][kCpg] row contributions to dgamma (and dbeta), then the [kChunks][kSlab] slab(s).
constexpr int kSmemRms = (kWarps * kCpg + kChunks * kSlab) * (int)sizeof(float);
constexpr int kSmemLn = (2 * kWarps * kCpg + 2 * kChunks * kSlab) * (int)sizeof(float);

// Eight bf16 elements; the host requires 16-byte aligned bases, and lane offsets are multiples of 16 elements.
struct __align__(16) V8bf {
  __nv_bfloat16 e[8];
};

__device__ __forceinline__ float warp_all_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// From x. Launched with programmatic dependent launch: the wait for the previous kernel comes before the first
// global read.
template <bool LN>
__global__ __launch_bounds__(kThreads) void one_cluster_grad512_k(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ dy,
    const float* __restrict__ mean, const float* __restrict__ rstd, const __nv_bfloat16* __restrict__ gamma,
    __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dgamma, __nv_bfloat16* __restrict__ dbeta,
    int G, int64_t C) {
  cudaGridDependencySynchronize();
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = (int)cluster.block_rank();
  const int g = (int)(blockIdx.x / kRanks);
  const int warp = (int)(threadIdx.x >> 5);
  const int lane = (int)(threadIdx.x & 31);

  extern __shared__ float smem[];
  float* sg = smem;                            // [kWarps][kCpg] dgamma contributions, a row each
  float* sb = sg + kWarps * kCpg;              // [kWarps][kCpg] dbeta contributions (LN)
  float* slg = sb + (LN ? kWarps * kCpg : 0);  // [kChunks][kSlab] this rank's dgamma slab
  float* slb = slg + kChunks * kSlab;          // [kChunks][kSlab] this rank's dbeta slab (LN)

  const int64_t grp_base = (int64_t)g * kCpg;
  const int c0 = lane * kChLane;
  const int64_t row = (int64_t)rank * kRowsRank + warp;
  const int64_t seg = row * C + grp_base + c0;
  const float inv = rstd[row * G + g];
  const float mu = LN ? mean[row * G + g] : 0.f;
  const float inv_cpg = 1.f / (float)kCpg;

  const V8bf vg0 = *reinterpret_cast<const V8bf*>(gamma + grp_base + c0);
  const V8bf vg1 = *reinterpret_cast<const V8bf*>(gamma + grp_base + c0 + 8);
  const V8bf vx0 = *reinterpret_cast<const V8bf*>(x + seg);
  const V8bf vx1 = *reinterpret_cast<const V8bf*>(x + seg + 8);
  const V8bf vd0 = *reinterpret_cast<const V8bf*>(dy + seg);
  const V8bf vd1 = *reinterpret_cast<const V8bf*>(dy + seg + 8);

  float xh[kChLane], dh[kChLane], dgv[kChLane], dbv[kChLane];
  float sa = 0.f, sm = 0.f;
#pragma unroll
  for (int i = 0; i < kChLane; ++i) {
    const V8bf& vx = (i < 8) ? vx0 : vx1;
    const V8bf& vd = (i < 8) ? vd0 : vd1;
    const V8bf& vg = (i < 8) ? vg0 : vg1;
    const int j = i & 7;
    const float xv = __bfloat162float(vx.e[j]);
    const float dv = __bfloat162float(vd.e[j]);
    const float gv = __bfloat162float(vg.e[j]);
    xh[i] = LN ? (xv - mu) * inv : xv * inv;
    dh[i] = dv * gv;
    sa += xh[i] * dh[i];
    sm += dh[i];
    dgv[i] = dv * xh[i];
    dbv[i] = dv;
  }
  sa = warp_all_sum(sa);
  sm = warp_all_sum(sm);
  const float proj = sa * inv_cpg;
  const float projb = LN ? sm * inv_cpg : 0.f;

  V8bf vo0, vo1;
#pragma unroll
  for (int i = 0; i < kChLane; ++i) {
    float v = dh[i] - xh[i] * proj;
    if (LN) v -= projb;
    V8bf& vo = (i < 8) ? vo0 : vo1;
    vo.e[i & 7] = __float2bfloat16(v * inv);
  }
  *reinterpret_cast<V8bf*>(dx + seg) = vo0;
  *reinterpret_cast<V8bf*>(dx + seg + 8) = vo1;

  float* sgw = sg + warp * kCpg + c0;
  float* sbw = sb + warp * kCpg + c0;
#pragma unroll
  for (int i = 0; i < kChLane; ++i) {
    sgw[i] = dgv[i];
    if (LN) sbw[i] = dbv[i];
  }
  __syncthreads();

  const int ca = 2 * (int)threadIdx.x;
  float a0 = 0.f, a1 = 0.f, b0 = 0.f, b1 = 0.f;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    a0 += sg[w * kCpg + ca];
    a1 += sg[w * kCpg + ca + 1];
    if (LN) {
      b0 += sb[w * kCpg + ca];
      b1 += sb[w * kCpg + ca + 1];
    }
  }
  const int dest = ca / kSlab;
  const int col = ca - dest * kSlab;
  float* dstg = (float*)cluster.map_shared_rank((void*)slg, dest);
  *reinterpret_cast<float2*>(dstg + rank * kSlab + col) = make_float2(a0, a1);
  if (LN) {
    float* dstb = (float*)cluster.map_shared_rank((void*)slb, dest);
    *reinterpret_cast<float2*>(dstb + rank * kSlab + col) = make_float2(b0, b1);
  }

  cluster.sync();

  if ((int)threadIdx.x < kSlab) {
    float a = 0.f, b = 0.f;
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      a += slg[c * kSlab + threadIdx.x];
      if (LN) b += slb[c * kSlab + threadIdx.x];
    }
    const int64_t o = grp_base + rank * kSlab + threadIdx.x;
    dgamma[o] = __float2bfloat16(a);
    if (LN) dbeta[o] = __float2bfloat16(b);
  }
}

// From y (memory_efficient): x_hat = (y - beta) * (1 / gamma), with y = gamma * x_hat for RMS; mean is not read. Same
// mapping and fold as one_cluster_grad512_k; launched without programmatic dependent launch.
template <bool LN>
__global__ __launch_bounds__(kThreads) void one_cluster_grad512_me_k(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ dy,
    const float* __restrict__ /*mean*/, const float* __restrict__ rstd, const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dgamma, __nv_bfloat16* __restrict__ dbeta,
    int G, int64_t C) {
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = (int)cluster.block_rank();
  const int g = (int)(blockIdx.x / kRanks);
  const int warp = (int)(threadIdx.x >> 5);
  const int lane = (int)(threadIdx.x & 31);

  extern __shared__ float smem[];
  float* sg = smem;
  float* sb = sg + kWarps * kCpg;
  float* slg = sb + (LN ? kWarps * kCpg : 0);
  float* slb = slg + kChunks * kSlab;

  const int64_t grp_base = (int64_t)g * kCpg;
  const int c0 = lane * kChLane;
  const int64_t row = (int64_t)rank * kRowsRank + warp;
  const int64_t seg = row * C + grp_base + c0;
  const float inv = rstd[row * G + g];
  const float inv_cpg = 1.f / (float)kCpg;

  const V8bf vg0 = *reinterpret_cast<const V8bf*>(gamma + grp_base + c0);
  const V8bf vg1 = *reinterpret_cast<const V8bf*>(gamma + grp_base + c0 + 8);
  V8bf vb0{}, vb1{};
  float rg[kChLane];
  if (LN) {
    vb0 = *reinterpret_cast<const V8bf*>(beta + grp_base + c0);
    vb1 = *reinterpret_cast<const V8bf*>(beta + grp_base + c0 + 8);
  }
#pragma unroll
  for (int i = 0; i < kChLane; ++i) {
    const float gv = __bfloat162float(((i < 8) ? vg0 : vg1).e[i & 7]);
    rg[i] = 1.f / gv;
  }
  const V8bf vx0 = *reinterpret_cast<const V8bf*>(x + seg);
  const V8bf vx1 = *reinterpret_cast<const V8bf*>(x + seg + 8);
  const V8bf vd0 = *reinterpret_cast<const V8bf*>(dy + seg);
  const V8bf vd1 = *reinterpret_cast<const V8bf*>(dy + seg + 8);

  float xh[kChLane], dh[kChLane], dgv[kChLane], dbv[kChLane];
  float sa = 0.f, sm = 0.f;
#pragma unroll
  for (int i = 0; i < kChLane; ++i) {
    const V8bf& vx = (i < 8) ? vx0 : vx1;
    const V8bf& vd = (i < 8) ? vd0 : vd1;
    const V8bf& vg = (i < 8) ? vg0 : vg1;
    const int j = i & 7;
    const float xv = __bfloat162float(vx.e[j]);
    const float dv = __bfloat162float(vd.e[j]);
    const float gv = __bfloat162float(vg.e[j]);
    const float bv = LN ? __bfloat162float(((i < 8) ? vb0 : vb1).e[i & 7]) : 0.f;
    xh[i] = (xv - bv) * rg[i];
    dh[i] = dv * gv;
    sa += xh[i] * dh[i];
    sm += dh[i];
    dgv[i] = dv * xh[i];
    dbv[i] = dv;
  }
  sa = warp_all_sum(sa);
  sm = warp_all_sum(sm);
  const float proj = sa * inv_cpg;
  const float projb = LN ? sm * inv_cpg : 0.f;

  V8bf vo0, vo1;
#pragma unroll
  for (int i = 0; i < kChLane; ++i) {
    float v = dh[i] - xh[i] * proj;
    if (LN) v -= projb;
    V8bf& vo = (i < 8) ? vo0 : vo1;
    vo.e[i & 7] = __float2bfloat16(v * inv);
  }
  *reinterpret_cast<V8bf*>(dx + seg) = vo0;
  *reinterpret_cast<V8bf*>(dx + seg + 8) = vo1;

  float* sgw = sg + warp * kCpg + c0;
  float* sbw = sb + warp * kCpg + c0;
#pragma unroll
  for (int i = 0; i < kChLane; ++i) {
    sgw[i] = dgv[i];
    if (LN) sbw[i] = dbv[i];
  }
  __syncthreads();

  const int ca = 2 * (int)threadIdx.x;
  float a0 = 0.f, a1 = 0.f, b0 = 0.f, b1 = 0.f;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    a0 += sg[w * kCpg + ca];
    a1 += sg[w * kCpg + ca + 1];
    if (LN) {
      b0 += sb[w * kCpg + ca];
      b1 += sb[w * kCpg + ca + 1];
    }
  }
  const int dest = ca / kSlab;
  const int col = ca - dest * kSlab;
  float* dstg = (float*)cluster.map_shared_rank((void*)slg, dest);
  *reinterpret_cast<float2*>(dstg + rank * kSlab + col) = make_float2(a0, a1);
  if (LN) {
    float* dstb = (float*)cluster.map_shared_rank((void*)slb, dest);
    *reinterpret_cast<float2*>(dstb + rank * kSlab + col) = make_float2(b0, b1);
  }

  cluster.sync();

  if ((int)threadIdx.x < kSlab) {
    float a = 0.f, b = 0.f;
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      a += slg[c * kSlab + threadIdx.x];
      if (LN) b += slb[c * kSlab + threadIdx.x];
    }
    const int64_t o = grp_base + rank * kSlab + threadIdx.x;
    dgamma[o] = __float2bfloat16(a);
    if (LN) dbeta[o] = __float2bfloat16(b);
  }
}

bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; }

// A 16-CTA cluster is beyond the portable maximum of 8, and cudaFuncSetAttribute applies to the current device, so each
// kernel opts in once a device. A refusal is remembered and declines every later call on that device; devices from
// ordinal 64 decline.
template <auto Kernel>
bool cluster16_allowed(int dev) {
  static unsigned long long opted = 0;
  static unsigned long long failed = 0;
  if (dev < 0 || dev >= 64) return false;
  const unsigned long long bit = 1ull << dev;
  if (failed & bit) return false;
  if (opted & bit) return true;
  if (cudaFuncSetAttribute((const void*)Kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) != cudaSuccess) {
    (void)cudaGetLastError();
    failed |= bit;
    return false;
  }
  opted |= bit;
  return true;
}

// Grid groups x 16 in clusters of 16. With `pdl` the launch carries programmatic dependent launch and, if the driver
// refuses it, is tried again without. False, with the error cleared, if the launch is refused.
template <auto Kernel, typename... Args>
bool launch(int64_t groups, int smem, bool pdl, Args... args) {
  const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  if (!cluster16_allowed<Kernel>((int)stream.device_index())) return false;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3((unsigned)(groups * kRanks));
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = (size_t)smem;
  cfg.stream = stream.stream();
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = kRanks;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = pdl ? 2 : 1;
  cudaError_t e = cudaLaunchKernelEx(&cfg, Kernel, args...);
  if (e != cudaSuccess && pdl) {
    (void)cudaGetLastError();
    cfg.numAttrs = 1;
    e = cudaLaunchKernelEx(&cfg, Kernel, args...);
  }
  if (e != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();  // an asynchronous launch failure still returns cudaSuccess above
  return true;
}

}  // namespace

bool GroupNormBwdOneClusterGrad512(const at::Tensor& dy, const at::Tensor& x_or_y, const at::Tensor& mean,
                                   const at::Tensor& rstd, const at::Tensor& gamma, const at::Tensor& beta,
                                   int64_t channels, int64_t groups, bool layer, bool memory_efficient,
                                   const at::Tensor& dx, const at::Tensor& dgamma, const at::Tensor& dbeta) {
  const auto bf16 = at::kBFloat16;
  if (dy.scalar_type() != bf16 || x_or_y.scalar_type() != bf16 || gamma.scalar_type() != bf16 ||
      dx.scalar_type() != bf16 || dgamma.scalar_type() != bf16 || (layer && dbeta.scalar_type() != bf16))
    return false;
  if (gamma.numel() != channels) return false;
  if (groups <= 0 || channels <= 0 || channels % groups != 0) return false;
  if (channels / groups != (int64_t)kCpg) return false;
  const int64_t rows = dy.numel() / channels;
  if (rows != (int64_t)kRanks * kRowsRank) return false;
  if (x_or_y.numel() != dy.numel() || dx.numel() != dy.numel()) return false;
  if (rstd.scalar_type() != at::kFloat || !rstd.is_contiguous() || rstd.numel() != rows * groups) return false;
  if (!dy.is_contiguous() || !x_or_y.is_contiguous() || !dx.is_contiguous()) return false;
  if (!gamma.is_contiguous() || !dgamma.is_contiguous() || dgamma.numel() != channels) return false;
  if (layer && (!dbeta.is_contiguous() || dbeta.numel() != channels)) return false;
  if (!aligned16(dy.data_ptr()) || !aligned16(x_or_y.data_ptr()) || !aligned16(dx.data_ptr()) ||
      !aligned16(gamma.data_ptr()) || !aligned16(dgamma.data_ptr()))
    return false;

  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x_or_y.data_ptr());
  const auto* dyp = reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr());
  const auto* gp = reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr());
  auto* dxp = reinterpret_cast<__nv_bfloat16*>(dx.data_ptr());
  auto* dgp = reinterpret_cast<__nv_bfloat16*>(dgamma.data_ptr());
  auto* dbp = layer ? reinterpret_cast<__nv_bfloat16*>(dbeta.data_ptr()) : nullptr;
  const float* rp = rstd.data_ptr<float>();
  const int64_t C = groups * kCpg;
  if (memory_efficient) {
    // beta is read as 16-byte vectors: an offset view declines here instead of faulting.
    if (layer && (beta.scalar_type() != bf16 || !beta.is_contiguous() || beta.numel() != channels ||
                  !aligned16(beta.data_ptr())))
      return false;
    if (!layer)
      return launch<one_cluster_grad512_me_k<false>>(groups, kSmemRms, false, xp, dyp, (const float*)nullptr, rp,
                                                     gp, (const __nv_bfloat16*)nullptr, dxp, dgp, dbp, (int)groups, C);
    return launch<one_cluster_grad512_me_k<true>>(groups, kSmemLn, false, xp, dyp, (const float*)nullptr, rp, gp,
                                                  reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr()), dxp, dgp,
                                                  dbp, (int)groups, C);
  }
  if (layer && (mean.scalar_type() != at::kFloat || !mean.is_contiguous() || mean.numel() != rows * groups))
    return false;
  if (!layer)
    return launch<one_cluster_grad512_k<false>>(groups, kSmemRms, true, xp, dyp, (const float*)nullptr, rp, gp, dxp,
                                                dgp, dbp, (int)groups, C);
  return launch<one_cluster_grad512_k<true>>(groups, kSmemLn, true, xp, dyp, mean.data_ptr<float>(), rp, gp, dxp, dgp,
                                             dbp, (int)groups, C);
}

}  // namespace xllm::ops::blackwell
