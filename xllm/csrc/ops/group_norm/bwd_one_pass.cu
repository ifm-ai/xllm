// One-pass backward of the group RMS norm and group layer norm for bf16 affine calls with groups of 8 to 256 channels
// (1 to 32 vectors of 16 bytes): x_or_y and dy are read once for dx, dgamma and dbeta. dispatch.cu's backward() reaches
// it through GroupNormBwdOnePass.
//
// A group is VL lanes of a warp, one 16-byte vector each (VL need not be a power of two), so a group's sums are one
// segmented shuffle reduction. Warps are column-stationary: warp (cs, rc) owns groups [cs * W, cs * W + W) of every
// row of row chunk rc and keeps its channels' dgamma/dbeta partials in registers over the chunk. Each warp writes its
// partials once to an fp32 workspace [nrc][C], and bwd_finalize_k sums the row chunks of each channel in a fixed order
// and converts them to the output dtype, so the result is the same bits run to run.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

#include "bwd_one_pass.h"

namespace xllm::ops::blackwell {
namespace {

constexpr int kVecElems = 8;  // bf16 elements in one 16-byte vector
constexpr int kWarpsPerBlock = 8;

template <typename T>
struct __align__(16) Vec8 {
  T e[kVecElems];
};

template <typename T>
struct Cvt;
template <>
struct Cvt<__nv_bfloat16> {
  static __device__ __forceinline__ float f(__nv_bfloat16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ __nv_bfloat16 t(float v) { return __float2bfloat16(v); }
};

template <typename T>
__device__ __forceinline__ Vec8<T> load8(const T* p) {
  return *reinterpret_cast<const Vec8<T>*>(p);
}
template <typename T>
__device__ __forceinline__ void store8(T* p, const Vec8<T>& v) {
  *reinterpret_cast<Vec8<T>*>(p) = v;
}
template <typename T>
__device__ __forceinline__ void to_f8(const Vec8<T>& v, float* f) {
#pragma unroll
  for (int i = 0; i < kVecElems; ++i) f[i] = Cvt<T>::f(v.e[i]);
}
template <typename T>
__device__ __forceinline__ void from_f8(const float* f, Vec8<T>& v) {
#pragma unroll
  for (int i = 0; i < kVecElems; ++i) v.e[i] = Cvt<T>::t(f[i]);
}

// Sum over the L lanes of this lane's segment (j = lane % L), returned to every lane of the segment. A lane adds its
// partner at offset `off` only when the partner is in its own segment, which sums a segment of any length <= 32. All
// 32 lanes must make the call.
__device__ __forceinline__ float seg_sum(float v, int j, int L, int lane) {
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    float o = __shfl_down_sync(0xffffffffu, v, off);
    if (off < L && j + off < L) v += o;
  }
  return __shfl_sync(0xffffffffu, v, lane - j);
}

// Lanes of warp (cs, rc): lane j of segment sg holds channels [8 j, 8 j + 8) of group cs * W + sg; lanes past W * L,
// and groups past G, are idle (they read group 0, in bounds, and store nothing). The warp takes rows [rc * rpc,
// rc * rpc + rpc) of its groups, U at a time: the U rows' loads are issued before the first row's shuffles, which
// would otherwise hold them back. ME (memory_efficient) is a template parameter so that the calls from x carry none of
// the reconstruction's registers.
template <typename T, bool LN, bool ME, int U>
__global__ __launch_bounds__(256) void bwd_small_k(
    const T* __restrict__ xo, const T* __restrict__ dy, const float* __restrict__ mean,
    const float* __restrict__ rstd, const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ dx,
    float* __restrict__ dg, float* __restrict__ db, int64_t rows, int G, int cpg, int L, int ncs, int nrc,
    int64_t rpc, float inv_cpg) {
  const int lane = threadIdx.x & 31;
  const int warp = blockIdx.x * kWarpsPerBlock + (int)(threadIdx.x >> 5);
  const int cs = warp % ncs;
  const int rc = warp / ncs;
  if (rc >= nrc) return;
  const int W = 32 / L;  // groups a warp covers
  const int sg = lane / L;
  const int j = lane - sg * L;
  const int64_t g0 = (int64_t)cs * W + sg;
  const bool active = (sg < W) && (g0 < G);
  const int gs = active ? (int)g0 : 0;
  const int64_t C = (int64_t)G * cpg;
  const int64_t off = active ? g0 * cpg + (int64_t)j * kVecElems : 0;
  float ga[kVecElems], ba[kVecElems];
  to_f8(load8(gamma + off), ga);
  if (LN && ME) to_f8(load8(beta + off), ba);
  float dga[kVecElems], dba[kVecElems];  // this lane's 8 channels, summed over the row chunk
#pragma unroll
  for (int i = 0; i < kVecElems; ++i) {
    dga[i] = 0.f;
    dba[i] = 0.f;
  }
  const int64_t r0 = (int64_t)rc * rpc;
  int64_t r1 = r0 + rpc;
  if (r1 > rows) r1 = rows;
  auto one = [&](Vec8<T> vx, Vec8<T> vd, float rr, float mu, int64_t p) {
    float xf[kVecElems], df[kVecElems];
    to_f8(vx, xf);
    to_f8(vd, df);
    float xh[kVecElems], dh[kVecElems];
    float a = 0.f, b = 0.f;
#pragma unroll
    for (int i = 0; i < kVecElems; ++i) {
      float x, d;
      if (ME) {
        x = (xf[i] - (LN ? ba[i] : 0.f)) / ga[i];
      } else {
        x = LN ? (xf[i] - mu) * rr : xf[i] * rr;
      }
      d = df[i] * ga[i];
      xh[i] = x;
      dh[i] = d;
      a += x * d;
      if (LN) b += d;
    }
    const float proj = seg_sum(a, j, L, lane) * inv_cpg;
    const float mb = LN ? seg_sum(b, j, L, lane) * inv_cpg : 0.f;
    if (active) {
      float o[kVecElems];
#pragma unroll
      for (int i = 0; i < kVecElems; ++i) {
        float t = dh[i] - xh[i] * proj;
        if (LN) t -= mb;
        o[i] = t * rr;
      }
      Vec8<T> ov;
      from_f8(o, ov);
      store8(dx + p, ov);
    }
#pragma unroll
    for (int i = 0; i < kVecElems; ++i) {
      dga[i] += df[i] * xh[i];
      if (LN) dba[i] += df[i];
    }
  };
  int64_t r = r0;
  for (; r + U <= r1; r += U) {
    Vec8<T> vx[U], vd[U];
    float rr[U], mu[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int64_t p = (r + u) * C + off;
      vx[u] = load8(xo + p);
      vd[u] = load8(dy + p);
      rr[u] = rstd[(r + u) * G + gs];
      mu[u] = (LN && !ME) ? mean[(r + u) * G + gs] : 0.f;
    }
#pragma unroll
    for (int u = 0; u < U; ++u) one(vx[u], vd[u], rr[u], mu[u], (r + u) * C + off);
  }
  for (; r < r1; ++r) {
    const int64_t p = r * C + off;
    one(load8(xo + p), load8(dy + p), rstd[r * G + gs], (LN && !ME) ? mean[r * G + gs] : 0.f, p);
  }
  if (active) {
    const int64_t ch = (int64_t)rc * C + g0 * cpg + (int64_t)j * kVecElems;
#pragma unroll
    for (int i = 0; i < kVecElems; ++i) {
      dg[ch + i] = dga[i];
      if (LN) db[ch + i] = dba[i];
    }
  }
}

// dgamma (and dbeta) of channel c: the sum of the nrc row-chunk partials of c, in the output dtype. A block covers 32
// channels; warp w sums its contiguous slice of the row chunks in ascending order, then warp 0 adds the eight slice
// sums in warp order.
template <typename T>
__global__ __launch_bounds__(256) void bwd_finalize_k(const float* __restrict__ dg, const float* __restrict__ db,
                                                     T* __restrict__ dgamma, T* __restrict__ dbeta, int C, int nrc,
                                                     int ln) {
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int c = blockIdx.x * 32 + lane;
  const bool live = c < C;  // C is a multiple of 8, not always of 32: skip the work, never the barrier
  __shared__ float sh[2][kWarpsPerBlock][32];
  const int chunk = (nrc + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int lo = w * chunk;
  int hi = lo + chunk;
  if (hi > nrc) hi = nrc;
  float s = 0.f, sb = 0.f;
  if (live) {
    for (int i = lo; i < hi; ++i) {
      s += dg[(int64_t)i * C + c];
      if (ln) sb += db[(int64_t)i * C + c];
    }
  }
  sh[0][w][lane] = s;
  sh[1][w][lane] = sb;
  __syncthreads();
  if (w == 0 && live) {
    float t = sh[0][0][lane], tb = sh[1][0][lane];
#pragma unroll
    for (int k = 1; k < kWarpsPerBlock; ++k) {
      t += sh[0][k][lane];
      tb += sh[1][k][lane];
    }
    dgamma[c] = Cvt<T>::t(t);
    if (ln) dbeta[c] = Cvt<T>::t(tb);
  }
}

// ---------------------------------------------------------------- plan and launch

// 32 warps an SM in flight (4736 on 148 SMs), each holding U rows of two tensors.
constexpr int64_t kTargetWarps = 4736;

// Row chunks: enough warps for the target, at least four rows a chunk, partials under 32 MB, at most 65535.
int64_t chunks_for(int64_t rows, int64_t units, int64_t C) {
  int64_t nrc = (kTargetWarps + units - 1) / units;
  const int64_t cap = rows > 4 ? (rows + 3) / 4 : 1;
  if (nrc > cap) nrc = cap;
  const int64_t wcap = C > 0 ? (4 << 20) / C : 1;
  if (nrc > wcap) nrc = wcap;
  if (nrc < 1) nrc = 1;
  if (nrc > 65535) nrc = 65535;
  return nrc;
}

struct Plan {
  int VL;       // 16-byte vectors a group == lanes a group
  int units;    // warps along the group axis: ceil(G / (32 / VL))
  int nrc;      // row chunks
  int64_t rpc;  // rows a chunk
};

Plan make_plan(int64_t rows, int64_t G, int64_t cpg, int es) {
  Plan p{};
  p.VL = (int)(cpg * es / 16);
  const int W = 32 / p.VL;
  p.units = (int)((G + W - 1) / W);
  p.nrc = (int)chunks_for(rows, p.units, G * cpg);
  p.rpc = (rows + p.nrc - 1) / p.nrc;
  return p;
}

template <typename T>
struct Args {
  const T* xo;
  const T* dy;
  const float* mean;  // layer norm only
  const float* rstd;
  const T* gamma;
  const T* beta;      // layer norm only
  T* dx;
  T* dgamma;
  T* dbeta;           // layer norm only
  float* ws;          // the dgamma partials [nrc][C], then the dbeta partials
  int64_t rows, G, cpg, C;
  float inv_cpg;
};

// Rows a warp holds at once from x: 5 at 8 vectors, and at 16 vectors when there are at least 8 warps along the group
// axis; 2 otherwise (at other widths the idle tail lanes make the deeper batch cost more than it saves, and 6 rows
// spill). The memory-efficient form keeps gamma and beta live and divides, so it holds 2 rows to stay at two CTAs an
// SM.
constexpr int RowsFromX(int VL, int units) { return ((VL == 16 && units >= 8) || VL == 8) ? 5 : 2; }
constexpr int kRowsMemoryEfficient = 2;

template <typename T, bool LN, bool ME, int U>
void launch_small(const Plan& p, const Args<T>& a, cudaStream_t st) {
  const int nctas = (int)((p.units * (int64_t)p.nrc + kWarpsPerBlock - 1) / kWarpsPerBlock);
  bwd_small_k<T, LN, ME, U><<<nctas, 256, 0, st>>>(a.xo, a.dy, a.mean, a.rstd, a.gamma, a.beta, a.dx, a.ws,
                                                   a.ws + (int64_t)p.nrc * a.C, a.rows, (int)a.G, (int)a.cpg, p.VL,
                                                   p.units, p.nrc, p.rpc, a.inv_cpg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, bool LN>
void launch(const Plan& p, const Args<T>& a, bool memory_efficient, cudaStream_t st) {
  if (memory_efficient) {
    launch_small<T, LN, true, kRowsMemoryEfficient>(p, a, st);
  } else if (RowsFromX(p.VL, p.units) == 5) {
    launch_small<T, LN, false, 5>(p, a, st);
  } else {
    launch_small<T, LN, false, 2>(p, a, st);
  }
  const int C = (int)a.C;
  bwd_finalize_k<T><<<(C + 31) / 32, 256, 0, st>>>(a.ws, a.ws + (int64_t)p.nrc * a.C, a.dgamma, a.dbeta, C, p.nrc,
                                                  LN ? 1 : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

bool GroupNormBwdOnePass(const torch::Tensor& dy, const torch::Tensor& x_or_y, const torch::Tensor& mean,
                         const torch::Tensor& rstd, const torch::Tensor& gamma, const torch::Tensor& beta,
                         int64_t channels, int64_t num_groups, bool layer, bool memory_efficient, torch::Tensor dx,
                         torch::Tensor dgamma, torch::Tensor dbeta) {
  if (num_groups < 1 || channels < 1 || num_groups > channels) return false;  // before dividing
  if (channels % num_groups != 0) return false;
  const int64_t cpg = channels / num_groups;
  const int64_t rows = x_or_y.numel() / channels;
  const int es = (int)x_or_y.element_size();
  // bf16 affine calls, groups of 1 to 32 whole 16-byte vectors, contiguous and 16-byte aligned tensors.
  if (x_or_y.scalar_type() != at::kBFloat16 || gamma.numel() == 0) return false;
  if ((cpg * es) % 16 != 0 || cpg < kVecElems || cpg * es > 32 * 16) return false;
  if (!x_or_y.is_contiguous() || !dy.is_contiguous() || !dx.is_contiguous()) return false;
  if (!rstd.is_contiguous()) return false;
  if (!gamma.is_contiguous() || (layer && !beta.is_contiguous())) return false;
  const intptr_t al = reinterpret_cast<intptr_t>(x_or_y.data_ptr()) | reinterpret_cast<intptr_t>(dy.data_ptr()) |
                      reinterpret_cast<intptr_t>(dx.data_ptr()) | reinterpret_cast<intptr_t>(rstd.data_ptr());
  if (al % 16 != 0) return false;
  const intptr_t ag = reinterpret_cast<intptr_t>(gamma.data_ptr()) |
                      (layer ? reinterpret_cast<intptr_t>(beta.data_ptr()) : 0);
  if (ag % 16 != 0) return false;
  if (layer && !memory_efficient && mean.numel() == 0) return false;  // nothing to recover x_hat from
  if (layer && memory_efficient && beta.numel() == 0) return false;   // (y - beta) / gamma needs both
  if (layer && !memory_efficient && !mean.is_contiguous()) return false;

  using T = __nv_bfloat16;
  const Plan p = make_plan(rows, num_groups, cpg, es);
  const torch::Tensor ws = at::empty({2 * p.nrc * channels}, x_or_y.options().dtype(at::kFloat));
  const Args<T> a{reinterpret_cast<const T*>(x_or_y.data_ptr()),
                  reinterpret_cast<const T*>(dy.data_ptr()),
                  layer ? mean.data_ptr<float>() : nullptr,
                  rstd.data_ptr<float>(),
                  reinterpret_cast<const T*>(gamma.data_ptr()),
                  layer ? reinterpret_cast<const T*>(beta.data_ptr()) : nullptr,
                  reinterpret_cast<T*>(dx.data_ptr()),
                  reinterpret_cast<T*>(dgamma.data_ptr()),
                  layer ? reinterpret_cast<T*>(dbeta.data_ptr()) : nullptr,
                  ws.data_ptr<float>(),
                  rows,
                  num_groups,
                  cpg,
                  channels,
                  1.f / (float)cpg};
  const cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (layer) launch<T, true>(p, a, memory_efficient, st);
  else launch<T, false>(p, a, memory_efficient, st);
  return true;
}

}  // namespace xllm::ops::blackwell
