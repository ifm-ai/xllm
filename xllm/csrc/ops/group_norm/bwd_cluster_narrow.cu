// Cluster backward of narrow-group affine calls, bf16 and fp32: dx, dgamma and dbeta in one launch with no atomics, no
// workspace and no second kernel. dispatch.cu's backward() reaches it through GroupNormBwdClusterNarrow.
//
// Grid: (C / ct) clusters of K = 8 CTAs. Cluster j owns channels [j * ct, j * ct + ct) (ct = 128 or 256, whole
// groups). Rank k owns rows [k * rpr, k * rpr + rpr), split again among nsub = 4 warp-row groups; within one, a group
// is L lanes of a warp holding VE elements each, and a warp keeps its channels' dgamma/dbeta in registers over its
// rows and writes their dx. Fold: each CTA adds its nsub partials in order, writes the result into slot `rank` of
// rank 0's shared memory, one cluster barrier, then rank 0 sums the K slots in ascending rank order and stores the
// output dtype. Fixed order throughout, so the result is the same bits run to run.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <type_traits>

#include "bwd_cluster_narrow.h"

namespace cg = cooperative_groups;

namespace xllm::ops::blackwell {
namespace {

constexpr int kCT = 256;  // channels a cluster owns, unless the tile is halved
constexpr int kK = 8;     // CTAs a cluster
constexpr int kNSub = 4;  // warp-row groups a rank's rows are split into

// VE elements of T in one vector, aligned to its own size (8 or 16 bytes).
template <typename T, int VE>
struct alignas(sizeof(T) * VE) VecN {
  T e[VE];
};

template <typename T, int VE>
__device__ __forceinline__ VecN<T, VE> loadv(const T* p) {
  return *reinterpret_cast<const VecN<T, VE>*>(p);
}
template <typename T, int VE>
__device__ __forceinline__ void storev(T* p, const VecN<T, VE>& v) {
  *reinterpret_cast<VecN<T, VE>*>(p) = v;
}
template <typename T>
__device__ __forceinline__ float tof(T v);
template <>
__device__ __forceinline__ float tof<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <>
__device__ __forceinline__ float tof<float>(float v) { return v; }
template <typename T>
__device__ __forceinline__ T fromf(float v);
template <>
__device__ __forceinline__ __nv_bfloat16 fromf<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <>
__device__ __forceinline__ float fromf<float>(float v) { return v; }

// Sum over the L lanes of this lane's segment (j = lane % L), returned to every lane of the segment.
__device__ __forceinline__ float seg_sum(float v, int j, int L, int lane) {
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    float o = __shfl_down_sync(0xffffffffu, v, off);
    if (off < L && j + off < L) v += o;
  }
  return __shfl_sync(0xffffffffu, v, lane - j);
}

// Warp `warp` of a CTA is column slot cs = warp % ncs of warp-row group sub = warp / ncs; lane j of segment sg holds
// channels [VE j, VE j + VE) of group cs * W + sg of the tile. Rows go U = 4 at a time, their loads issued first.
// Launched with programmatic dependent launch: the wait for the previous kernel comes before the first global read.
template <typename T, int VE, int LB, bool LN, bool ME>
__global__ __launch_bounds__(LB) void cluster_narrow_k(
    const T* __restrict__ xo, const T* __restrict__ dy, const float* __restrict__ mean, const float* __restrict__ rstd,
    const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ dx, T* __restrict__ dgamma,
    T* __restrict__ dbeta, int64_t rows, int G, int cpg, int L, int ct, int ngt, int K, int nsub, float inv_cpg) {
  cudaGridDependencySynchronize();
  constexpr int U = 4;
  const int nred = LN ? 2 : 1;
  // [nsub][nred][ct] partials, then [K][nred][ct] slots that the cluster's ranks write into rank 0's copy.
  extern __shared__ float sm[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int W = 32 / L;
  const int ncs = ngt / W;
  const int cs = warp % ncs;
  const int sub = warp / ncs;
  const int sg = lane / L;
  const int j = lane - sg * L;
  const int64_t C = (int64_t)G * cpg;
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = (int)cluster.block_rank();
  const int tile = (int)(blockIdx.x / K);
  const int g_loc = cs * W + sg;
  const int g = tile * ngt + g_loc;
  const bool active = (sg < W) && (sub < nsub) && (g < G);
  const int gs = active ? g : 0;
  const int64_t off = active ? (int64_t)g * cpg + (int64_t)j * VE : 0;
  float ga[VE], ba[VE], rga[VE];
#pragma unroll
  for (int i = 0; i < VE; ++i) {
    ga[i] = 1.f;
    ba[i] = 0.f;
  }
  {
    const VecN<T, VE> vg = loadv<T, VE>(gamma + off);
#pragma unroll
    for (int i = 0; i < VE; ++i) ga[i] = tof<T>(vg.e[i]);
    if (LN && ME) {
      const VecN<T, VE> vb = loadv<T, VE>(beta + off);
#pragma unroll
      for (int i = 0; i < VE; ++i) ba[i] = tof<T>(vb.e[i]);
    }
  }
  // bf16 from y multiplies by 1 / gamma, computed once; fp32 keeps the divide, as the extra registers would cost it a
  // CTA an SM.
  if constexpr (std::is_same_v<T, __nv_bfloat16> && ME) {
#pragma unroll
    for (int i = 0; i < VE; ++i) rga[i] = 1.f / ga[i];
  }
  float dga[VE], dba[VE];
#pragma unroll
  for (int i = 0; i < VE; ++i) {
    dga[i] = 0.f;
    dba[i] = 0.f;
  }
  const int64_t rpr = (rows + K - 1) / K;
  const int64_t rps = (rpr + nsub - 1) / nsub;
  int64_t r0 = (int64_t)rank * rpr + (int64_t)sub * rps;
  int64_t r1 = r0 + rps;
  const int64_t r_end = (int64_t)(rank + 1) * rpr;
  if (r1 > r_end) r1 = r_end;
  if (r1 > rows) r1 = rows;
  if (r0 > rows) r0 = rows;
  auto one = [&](VecN<T, VE> vx, VecN<T, VE> vd, float rr, float mu, int64_t p) {
    float xf[VE], df[VE];
#pragma unroll
    for (int i = 0; i < VE; ++i) xf[i] = tof<T>(vx.e[i]);
#pragma unroll
    for (int i = 0; i < VE; ++i) df[i] = tof<T>(vd.e[i]);
    float xh[VE], dh[VE];
    float a = 0.f, b = 0.f;
#pragma unroll
    for (int i = 0; i < VE; ++i) {
      float x;
      if (ME) {
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
          x = LN ? (xf[i] - ba[i]) * rga[i] : xf[i] * rga[i];
        } else {
          x = LN ? (xf[i] - ba[i]) / ga[i] : xf[i] / ga[i];
        }
      } else {
        x = LN ? (xf[i] - mu) * rr : xf[i] * rr;
      }
      const float d = df[i] * ga[i];
      xh[i] = x;
      dh[i] = d;
      a += x * d;
      if (LN) b += d;
    }
    const float proj = seg_sum(a, j, L, lane) * inv_cpg;
    const float mb = LN ? seg_sum(b, j, L, lane) * inv_cpg : 0.f;
    if (active) {
      VecN<T, VE> ov;
#pragma unroll
      for (int i = 0; i < VE; ++i) {
        float t = dh[i] - xh[i] * proj;
        if (LN) t -= mb;
        ov.e[i] = fromf<T>(t * rr);
      }
      storev<T, VE>(dx + p, ov);
    }
#pragma unroll
    for (int i = 0; i < VE; ++i) {
      dga[i] += df[i] * xh[i];
      if (LN) dba[i] += df[i];
    }
  };
  int64_t r = r0;
  for (; r + U <= r1; r += U) {
    VecN<T, VE> vx[U], vd[U];
    float rr[U], mu[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int64_t p = (r + u) * C + off;
      vx[u] = loadv<T, VE>(xo + p);
      vd[u] = loadv<T, VE>(dy + p);
      rr[u] = rstd[(r + u) * G + gs];
      mu[u] = (LN && !ME) ? mean[(r + u) * G + gs] : 0.f;
    }
#pragma unroll
    for (int u = 0; u < U; ++u) one(vx[u], vd[u], rr[u], mu[u], (r + u) * C + off);
  }
  for (; r < r1; ++r) {
    const int64_t p = r * C + off;
    one(loadv<T, VE>(xo + p), loadv<T, VE>(dy + p), rstd[r * G + gs], (LN && !ME) ? mean[r * G + gs] : 0.f, p);
  }
  if (active) {
    const int64_t lc = (int64_t)g_loc * cpg + (int64_t)j * VE;
#pragma unroll
    for (int i = 0; i < VE; ++i) {
      sm[((int64_t)sub * nred + 0) * ct + lc + i] = dga[i];
      if (LN) sm[((int64_t)sub * nred + 1) * ct + lc + i] = dba[i];
    }
  }
  __syncthreads();
  if (nsub > 1) {
    for (int c = threadIdx.x; c < ct; c += blockDim.x) {
      float s0 = sm[0 * nred * ct + c];
      float s1 = LN ? sm[0 * nred * ct + ct + c] : 0.f;
      for (int k = 1; k < nsub; ++k) {
        s0 += sm[((int64_t)k * nred + 0) * ct + c];
        if (LN) s1 += sm[((int64_t)k * nred + 1) * ct + c];
      }
      sm[0 * nred * ct + c] = s0;
      if (LN) sm[0 * nred * ct + ct + c] = s1;
    }
  }
  __syncthreads();
  // Each rank writes its folded partial into its own slot of rank 0's region (remote shared stores); rank 0 copies
  // into slot 0.
  float* dest = sm + (int64_t)nsub * nred * ct;
  {
    float* dst0 = (float*)cluster.map_shared_rank((void*)dest, 0);
    float* slot = dst0 + (int64_t)rank * nred * ct;
    for (int c = threadIdx.x; c < nred * ct; c += blockDim.x) slot[c] = sm[c];
  }
  cluster.sync();  // every slot is written; nothing remote is touched after this, so the other ranks may exit
  if (rank != 0) return;
  {
    const int64_t base = (int64_t)tile * ct;
    for (int c = threadIdx.x; c < ct; c += blockDim.x) {
      float s0 = 0.f, s1 = 0.f;
      for (int rk = 0; rk < K; ++rk) {
        const float* w = dest + (int64_t)rk * nred * ct;
        s0 += w[c];
        if (LN) s1 += w[ct + c];
      }
      dgamma[base + c] = fromf<T>(s0);
      if (LN) dbeta[base + c] = fromf<T>(s1);
    }
  }
}

// Launches the form <T, VE, LB> for the call's norm and input mode. If the driver refuses the launch, it is tried again
// without the programmatic-dependent-launch attribute (the cluster dimension stays); if that is refused too, the error
// is cleared and the call declines.
template <typename T, int VE, int LB>
bool launch(cudaLaunchConfig_t& cfg, bool layer, bool memory_efficient, const torch::Tensor& dy,
            const torch::Tensor& x_or_y, const float* mp, const float* rp, const torch::Tensor& gamma,
            const torch::Tensor& beta, torch::Tensor& dx, torch::Tensor& dgamma, torch::Tensor& dbeta, int64_t rows,
            int64_t num_groups, int64_t cpg, int L, int ct, int ngt, int nsub, float inv_cpg) {
  auto kern = layer ? (memory_efficient ? cluster_narrow_k<T, VE, LB, true, true>
                                        : cluster_narrow_k<T, VE, LB, true, false>)
                    : (memory_efficient ? cluster_narrow_k<T, VE, LB, false, true>
                                        : cluster_narrow_k<T, VE, LB, false, false>);
  const T* xp = reinterpret_cast<const T*>(x_or_y.data_ptr());
  const T* dyp = reinterpret_cast<const T*>(dy.data_ptr());
  const T* gp = reinterpret_cast<const T*>(gamma.data_ptr());
  const T* bp = (beta.numel() > 0) ? reinterpret_cast<const T*>(beta.data_ptr()) : nullptr;
  T* dxp = reinterpret_cast<T*>(dx.data_ptr());
  T* dgp = reinterpret_cast<T*>(dgamma.data_ptr());
  T* dbp = (layer && dbeta.numel() > 0) ? reinterpret_cast<T*>(dbeta.data_ptr()) : nullptr;
  auto go = [&] {
    return cudaLaunchKernelEx(&cfg, kern, xp, dyp, mp, rp, gp, bp, dxp, dgp, dbp, rows, (int)num_groups, (int)cpg, L,
                              ct, ngt, kK, nsub, inv_cpg);
  };
  if (go() == cudaSuccess) return true;
  (void)cudaGetLastError();
  cfg.numAttrs = 1;
  if (go() == cudaSuccess) return true;
  (void)cudaGetLastError();
  return false;
}

}  // namespace

bool GroupNormBwdClusterNarrow(const torch::Tensor& dy, const torch::Tensor& x_or_y, const torch::Tensor& mean,
                               const torch::Tensor& rstd, const torch::Tensor& gamma, const torch::Tensor& beta,
                               int64_t channels, int64_t num_groups, bool layer, bool memory_efficient,
                               torch::Tensor dx, torch::Tensor dgamma, torch::Tensor dbeta) {
  if (num_groups < 1 || channels < 1 || num_groups > channels) return false;
  if (channels % num_groups != 0) return false;
  const int64_t cpg = channels / num_groups;
  const int64_t rows = x_or_y.numel() / channels;
  if (rows < 1) return false;
  if (gamma.numel() == 0) return false;  // affine only: the fold exists for dgamma/dbeta
  const auto dt = x_or_y.scalar_type();
  if (dt != at::kBFloat16 && dt != at::kFloat) return false;
  const int es = (int)x_or_y.element_size();

  // The forms <T, VE, block>:
  //   bf16, 32 <= cpg <= 128, rows <= 256: 8-byte vectors (VE = 4), twice the lanes a group with half the
  //     accumulators of a 16-byte vector. 128-channel tiles (twice the clusters) where the 256-channel grid leaves
  //     SMs idle (channels <= 4608) and cpg >= 64: <bf16, 4, 128>; otherwise <bf16, 4, 256>;
  //   other bf16 calls: 16-byte vectors, <bf16, 8, 128>;
  //   fp32: 16-byte vectors (VE = 4); 128-channel tiles below 8192 channels, <float, 4, 128>, else <float, 4, 256>.
  const bool ve4 = (dt == at::kBFloat16) && (cpg >= 32) && (cpg <= 128) && (rows <= 256);
  const bool tile128 = ve4 && (channels <= 4608) && (channels % 256 == 0) && (cpg >= 64);
  const bool narrow_fp32 = (es == 4) && (channels < 8192);
  const int VE = ve4 ? 4 : (16 / es);
  const int L = (int)(cpg / VE);  // lanes a group
  // A group is a whole warp segment, a tile is a whole number of groups, and its groups fill whole warps.
  if (cpg < VE || (cpg * es) % 16 != 0) return false;
  if (L < 1 || L > 32) return false;
  if (32 % L != 0) return false;
  const int ct = (narrow_fp32 || tile128) ? 128 : kCT;
  const int nsub = kNSub;
  if (channels % ct != 0) return false;
  const int ngt = ct / (int)cpg;
  if (ngt * (int)cpg != ct) return false;
  if (ngt % (32 / L) != 0) return false;
  if (!x_or_y.is_contiguous() || !dy.is_contiguous() || !dx.is_contiguous() || !rstd.is_contiguous()) return false;
  if (!gamma.is_contiguous() || (layer && !beta.is_contiguous())) return false;
  if (layer && memory_efficient && beta.numel() == 0) return false;  // (y - beta) / gamma needs both
  if (layer && !memory_efficient && (mean.numel() == 0 || !mean.is_contiguous())) return false;
  const intptr_t al = reinterpret_cast<intptr_t>(x_or_y.data_ptr()) | reinterpret_cast<intptr_t>(dy.data_ptr()) |
                      reinterpret_cast<intptr_t>(dx.data_ptr()) | reinterpret_cast<intptr_t>(rstd.data_ptr()) |
                      reinterpret_cast<intptr_t>(gamma.data_ptr()) |
                      (layer ? reinterpret_cast<intptr_t>(beta.data_ptr()) : 0);
  if (al % 16 != 0) return false;

  // The block is 32 * ncs * nsub threads; it must equal the launch bound of the form chosen below.
  const int ncs = ngt / (32 / L);
  const int nthreads = 32 * ncs * nsub;
  const int lb = tile128 ? 128 : (ve4 ? 256 : ((es == 2 || narrow_fp32) ? 128 : 256));
  if (nthreads != lb) return false;
  const size_t smem = (size_t)(nsub + kK) * (layer ? 2 : 1) * ct * sizeof(float);
  if (smem > 48 * 1024) return false;
  const int grid = (int)(channels / ct) * kK;
  const float inv_cpg = 1.f / (float)cpg;

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(nthreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = kK;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;

  const float* mp = (layer && !memory_efficient) ? mean.data_ptr<float>() : nullptr;
  const float* rp = rstd.data_ptr<float>();
  if (dt == at::kBFloat16) {
    if (tile128)
      return launch<__nv_bfloat16, 4, 128>(cfg, layer, memory_efficient, dy, x_or_y, mp, rp, gamma, beta, dx, dgamma,
                                           dbeta, rows, num_groups, cpg, L, ct, ngt, nsub, inv_cpg);
    if (ve4)
      return launch<__nv_bfloat16, 4, 256>(cfg, layer, memory_efficient, dy, x_or_y, mp, rp, gamma, beta, dx, dgamma,
                                           dbeta, rows, num_groups, cpg, L, ct, ngt, nsub, inv_cpg);
    return launch<__nv_bfloat16, 8, 128>(cfg, layer, memory_efficient, dy, x_or_y, mp, rp, gamma, beta, dx, dgamma,
                                         dbeta, rows, num_groups, cpg, L, ct, ngt, nsub, inv_cpg);
  }
  if (narrow_fp32)
    return launch<float, 4, 128>(cfg, layer, memory_efficient, dy, x_or_y, mp, rp, gamma, beta, dx, dgamma, dbeta,
                                 rows, num_groups, cpg, L, ct, ngt, nsub, inv_cpg);
  return launch<float, 4, 256>(cfg, layer, memory_efficient, dy, x_or_y, mp, rp, gamma, beta, dx, dgamma, dbeta, rows,
                               num_groups, cpg, L, ct, ngt, nsub, inv_cpg);
}

}  // namespace xllm::ops::blackwell
