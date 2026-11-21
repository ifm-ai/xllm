// Dense-lane kernels of the group RMS norm and group layer norm, for groups of 1..32 vectors of 16 bytes (8..256
// channels in 16 bits, 4..128 in fp32): the forward (Fwd) and the backward (Bwd), called by dispatch.cu's forward() and
// dispatch.cu's backward() in dispatch.cu. A group's statistics are reduced inside the warp that holds the group: a
// segmented shuffle tree, no shared memory, no __syncthreads.
#include "dense_lanes.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "pdl.h"
#include "width_common.cuh"

namespace xllm::ops::blackwell::dense_lanes {
namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kBlock = 32 * kWarpsPerBlock;

// ---------------------------------------------------------------------------------------------------------------------
// Data path

__device__ __forceinline__ uint4 ldg16(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ void stg16(void* p, const uint4& v) {
  asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};"
               :
               : "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

// A 16-byte vector of T unpacked to N floats, and N floats rounded to T and packed.
template <typename T>
struct Cvt;

template <>
struct Cvt<float> {
  static constexpr int N = 4;
  __device__ __forceinline__ static void unpack(const uint4& v, float* f) {
    f[0] = __uint_as_float(v.x);
    f[1] = __uint_as_float(v.y);
    f[2] = __uint_as_float(v.z);
    f[3] = __uint_as_float(v.w);
  }
  __device__ __forceinline__ static uint4 pack(const float* f) {
    uint4 v;
    v.x = __float_as_uint(f[0]);
    v.y = __float_as_uint(f[1]);
    v.z = __float_as_uint(f[2]);
    v.w = __float_as_uint(f[3]);
    return v;
  }
  __device__ __forceinline__ static float from_float(float x) { return x; }
};

template <>
struct Cvt<__nv_bfloat16> {
  static constexpr int N = 8;
  __device__ __forceinline__ static void unpack(const uint4& v, float* f) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 t = __bfloat1622float2(p[i]);
      f[2 * i] = t.x;
      f[2 * i + 1] = t.y;
    }
  }
  __device__ __forceinline__ static uint4 pack(const float* f) {
    uint4 v;
    __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    return v;
  }
  __device__ __forceinline__ static __nv_bfloat16 from_float(float x) { return __float2bfloat16(x); }
};

template <>
struct Cvt<__half> {
  static constexpr int N = 8;
  __device__ __forceinline__ static void unpack(const uint4& v, float* f) {
    const __half2* p = reinterpret_cast<const __half2*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 t = __half22float2(p[i]);
      f[2 * i] = t.x;
      f[2 * i + 1] = t.y;
    }
  }
  __device__ __forceinline__ static uint4 pack(const float* f) {
    uint4 v;
    __half2* p = reinterpret_cast<__half2*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
    return v;
  }
  __device__ __forceinline__ static __half from_float(float x) { return __float2half(x); }
};

// A pair of 16-bit values widened to float, and a float pair rounded once to 16 bits.
template <typename T>
struct PackPair;
template <>
struct PackPair<__half> {
  using type = __half2;
  static __device__ __forceinline__ float2 widen(__half2 a) { return __half22float2(a); }
  static __device__ __forceinline__ __half2 narrow(float2 a) { return __float22half2_rn(a); }
};
template <>
struct PackPair<__nv_bfloat16> {
  using type = __nv_bfloat162;
  static __device__ __forceinline__ float2 widen(__nv_bfloat162 a) { return __bfloat1622float2(a); }
  static __device__ __forceinline__ __nv_bfloat162 narrow(float2 a) { return __float22bfloat162_rn(a); }
};

// The sum over the W lanes of this lane's segment (W in [1, 32], segments at multiples of W), returned to every lane
// of it: each lane adds the value d lanes up while that lane is still in the segment (d = 1, 2, 4, ...), and the
// segment's first lane broadcasts the total. ceil(log2 W) steps for any W.
__device__ __forceinline__ float seg_sum(float v, int lane, int W) {
  const int lid = lane % W;
#pragma unroll 1
  for (int d = 1; d < W; d <<= 1) {
    const float u = __shfl_down_sync(0xffffffffu, v, d);
    if (lid + d < W) v += u;
  }
  return __shfl_sync(0xffffffffu, v, lane - lid);
}

// seg_sum, or for a power-of-two W an xor butterfly over the distances W/2 .. 1: every partner is inside the segment
// and every lane ends with the total. The butterfly adds the same terms in another association than seg_sum.
template <bool POW2>
__device__ __forceinline__ float seg_sum_sel(float v, int lane, int W) {
  if constexpr (!POW2) {
    return seg_sum(v, lane, W);
  } else {
#pragma unroll 1
    for (int d = W >> 1; d > 0; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
    return v;
  }
}

// Two sums through seg_sum_sel's tree in one pass; each is reduced exactly as it would be alone.
template <bool POW2>
__device__ __forceinline__ float2 seg_sum2_sel(float2 v, int lane, int W) {
  if constexpr (!POW2) {
    return make_float2(seg_sum(v.x, lane, W), seg_sum(v.y, lane, W));
  } else {
#pragma unroll 1
    for (int d = W >> 1; d > 0; d >>= 1) {
      v.x += __shfl_xor_sync(0xffffffffu, v.x, d);
      v.y += __shfl_xor_sync(0xffffffffu, v.y, d);
    }
    return v;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Forward

// A group of V vectors takes W = V / Q lanes, Q vectors (chunks) a lane: Q = 2 for an even V without affine (half the
// lanes, twice the groups a warp), Q = 1 otherwise. gw = 32 / W groups share a warp (lanes past gw * W idle); a warp
// takes ILP slots of gw groups at a time and grid-strides over the rows * groups slots. Chunk j of a lane sits at lane
// lid + j * W, so each load reads a contiguous run of the group. A lane sums its elements in order, then seg_sum_sel
// reduces over the group's W lanes (POW2: W is a power of two). Compile-time choices of an instantiation:
//   kPairAcc / kPairRMS: the sums of the bf16 (and fp32 layer norm) Q = 2 arms without affine run as two chains, even
//     and odd elements, on float pairs, folded once.
//   kOnePass: the layer norm on power-of-two segments reduces the sum and the sum of squares together and keeps
//     var = E[x^2] - mean^2 where var >= E[x^2] / 16 and E[x^2] is finite; if any group of the warp fails that, the
//     warp computes the centred squares and that group takes them. Not on the bf16 ILP 2 Q 2 arm (measured slower).
//   kStoreNow: arms without affine store each slot as soon as it is normalized (shorter live ranges, more resident
//     warps), but the bf16 RMS ILP 2 Q 2 POW2 arm (measured slower); the others store all ILP slots at the end.
template <typename T, bool LAYER, bool AFFINE, int ILP, int Q, bool POW2>
__global__ void __launch_bounds__(kBlock) gn_fwd_kernel(const T* __restrict__ x, T* __restrict__ y,
                                                       float* __restrict__ rstd_out, float* __restrict__ mean_out,
                                                       const T* __restrict__ gamma, const T* __restrict__ beta,
                                                       int G, int cpg, int W, float eps, int64_t total_slots) {
  cudaGridDependencySynchronize();
  constexpr int N = Cvt<T>::N;
  constexpr int M = Q * N;
  const int lane = threadIdx.x & 31;
  const int gw = 32 / W;
  const int live = gw * W;
  const int soff = lane / W;
  const int lid = lane - soff * W;
  const float inv_n = 1.0f / (float)cpg;
  const int warps = blockDim.x >> 5;
  const int64_t step = (int64_t)ILP * gw;
  const int64_t wstep = (int64_t)gridDim.x * warps * step;
  int64_t s0 = ((int64_t)blockIdx.x * warps + (threadIdx.x >> 5)) * step;

  constexpr bool kStoreNow =
      !AFFINE && !(std::is_same<T, __nv_bfloat16>::value && !LAYER && ILP == 2 && Q == 2 && POW2);
  for (; s0 < total_slots; s0 += wstep) {
    uint4 raw[ILP][Q];
    int64_t off[ILP];
    bool ok[ILP];
    const int64_t jstep = (int64_t)W * N;  // chunk j sits j*W*N elements on: a constant, so one 64-bit address a slot
#pragma unroll
    for (int u = 0; u < ILP; ++u) {
      const int64_t slot = s0 + u * gw + soff;
      ok[u] = (lane < live) && (slot < total_slots);
      off[u] = slot * (int64_t)cpg + (int64_t)lid * N;
#pragma unroll
      for (int j = 0; j < Q; ++j) raw[u][j] = ok[u] ? ldg16(x + off[u] + j * jstep) : make_uint4(0, 0, 0, 0);
    }
    float gv[ILP][M], bv[ILP][M];
    if (AFFINE) {
#pragma unroll
      for (int u = 0; u < ILP; ++u) {
        const int64_t slot = s0 + u * gw + soff;
        const int64_t base = (slot % G) * (int64_t)cpg;
#pragma unroll
        for (int j = 0; j < Q; ++j) {
          const int64_t col = base + (int64_t)(lid + j * W) * N;
          Cvt<T>::unpack(ok[u] ? ldg16(gamma + col) : make_uint4(0, 0, 0, 0), gv[u] + j * N);
          if (LAYER) Cvt<T>::unpack(ok[u] ? ldg16(beta + col) : make_uint4(0, 0, 0, 0), bv[u] + j * N);
        }
      }
    }
    float out[ILP][M];
    float rst[ILP], muv[ILP];
#pragma unroll
    for (int u = 0; u < ILP; ++u) {
      float v[M];
#pragma unroll
      for (int j = 0; j < Q; ++j) Cvt<T>::unpack(raw[u][j], v + j * N);
      if (LAYER) {
        constexpr bool kPairBf16 = std::is_same<T, __nv_bfloat16>::value;
        constexpr bool kPairF32 = std::is_same<T, float>::value;
        constexpr bool kPairAcc = (kPairBf16 || kPairF32) && !AFFINE && Q == 2 && (M % 2) == 0;
        constexpr bool kOnePass = POW2 && !(kPairBf16 && ILP == 2 && Q == 2);
        float s = 0.f;
        float q = 0.f;  // kOnePass: the raw sum of squares
        if constexpr (kPairAcc) {
          float2 sp = make_float2(0.f, 0.f);
          float2 qp = make_float2(0.f, 0.f);
#pragma unroll
          for (int k = 0; k < M; k += 2) {
            const float2 vp = make_float2(v[k], v[k + 1]);
            sp = __fadd2_rn(sp, vp);
            if constexpr (kOnePass) qp = __ffma2_rn(vp, vp, qp);
          }
          s = sp.x + sp.y;
          if constexpr (kOnePass) q = qp.x + qp.y;
        } else {
#pragma unroll
          for (int k = 0; k < M; ++k) {
            s += v[k];
            if constexpr (kOnePass) q = fmaf(v[k], v[k], q);
          }
        }
        float var;
        float mu;
        if constexpr (kOnePass) {
          constexpr float kRawGuard = 0.0625f;
          const float2 rq = seg_sum2_sel<POW2>(make_float2(s, q), lane, W);
          mu = rq.x * inv_n;
          const float m2 = rq.y * inv_n;
          var = fmaf(-mu, mu, m2);
          // An overflowed sum of squares reads inf >= inf, hence the finiteness test.
          const bool bad = !(var >= kRawGuard * m2) || !isfinite(m2);
          // A warp-wide vote, so every lane reaches the fallback's shuffles; a group that passed keeps its value.
          const bool any_bad = __any_sync(0xffffffffu, bad);
          if (any_bad) {
            q = 0.f;
            if constexpr (kPairAcc) {
              float2 cp = make_float2(0.f, 0.f);
#pragma unroll
              for (int k = 0; k < M; k += 2) {
                const float2 d = make_float2(v[k] - mu, v[k + 1] - mu);
                cp = __ffma2_rn(d, d, cp);
              }
              q = cp.x + cp.y;
            } else {
#pragma unroll
              for (int k = 0; k < M; ++k) {
                const float d = v[k] - mu;
                q = fmaf(d, d, q);
              }
            }
            const float vc = seg_sum_sel<POW2>(q, lane, W) * inv_n;
            if (bad) var = vc;
          }
        } else {
          mu = seg_sum_sel<POW2>(s, lane, W) * inv_n;
          float c2 = 0.f;
          if constexpr (kPairAcc) {
            float2 cp = make_float2(0.f, 0.f);
#pragma unroll
            for (int k = 0; k < M; k += 2) {
              const float2 d = make_float2(v[k] - mu, v[k + 1] - mu);
              cp = __ffma2_rn(d, d, cp);
            }
            c2 = cp.x + cp.y;
          } else {
#pragma unroll
            for (int k = 0; k < M; ++k) {
              const float d = v[k] - mu;
              c2 += d * d;
            }
          }
          var = seg_sum_sel<POW2>(c2, lane, W) * inv_n;
        }
        const float r = rsqrtf(var + eps);
        muv[u] = mu;
        rst[u] = r;
#pragma unroll
        for (int k = 0; k < M; ++k) {
          const float xh = (v[k] - mu) * r;
          out[u][k] = AFFINE ? xh * gv[u][k] + bv[u][k] : xh;
        }
      } else {
        constexpr bool kPairRMS =
            std::is_same<T, __nv_bfloat16>::value && !AFFINE && Q == 2 && (M % 2) == 0;
        float sq = 0.f;
        if constexpr (kPairRMS) {
          float2 sp = make_float2(0.f, 0.f);
#pragma unroll
          for (int k = 0; k < M; k += 2) {
            const float2 xp = make_float2(v[k], v[k + 1]);
            sp = __ffma2_rn(xp, xp, sp);
          }
          sq = sp.x + sp.y;
        } else {
#pragma unroll
          for (int k = 0; k < M; ++k) sq += v[k] * v[k];
        }
        const float r = rsqrtf(seg_sum_sel<POW2>(sq, lane, W) * inv_n + eps);
        rst[u] = r;
#pragma unroll
        for (int k = 0; k < M; ++k) {
          const float xh = v[k] * r;
          out[u][k] = AFFINE ? xh * gv[u][k] : xh;
        }
      }
      if constexpr (kStoreNow) {
        if (ok[u]) {
#pragma unroll
          for (int j = 0; j < Q; ++j) stg16(y + off[u] + j * jstep, Cvt<T>::pack(out[u] + j * N));
          if (lid == 0) {
            const int64_t slot = s0 + u * gw + soff;
            rstd_out[slot] = rst[u];
            if (LAYER) mean_out[slot] = muv[u];
          }
        }
      }
    }
    if constexpr (!kStoreNow) {
#pragma unroll
      for (int u = 0; u < ILP; ++u) {
        if (ok[u]) {
#pragma unroll
          for (int j = 0; j < Q; ++j) stg16(y + off[u] + j * jstep, Cvt<T>::pack(out[u] + j * N));
          if (lid == 0) {
            const int64_t slot = s0 + u * gw + soff;
            rstd_out[slot] = rst[u];
            if (LAYER) mean_out[slot] = muv[u];
          }
        }
      }
    }
  }
}

// The affine 16-bit forward at 40 channels a group (5 vectors): a warp keeps one block of W = 6 groups (30 lanes,
// one vector a lane) and walks the rows with a stride of nj, so gamma and beta are read once and stay in registers.
// U = 4 rows are loaded before the first is processed. Sums: a lane's 8 values in order, then group_sum<5> over the
// group's lanes. The layer norm keeps its factors in 16 bits and applies them two channels an instruction in float;
// the RMS norm holds gamma in float. y is rounded once.
template <typename T, bool LAYER>
__global__ __launch_bounds__(32 * kWarpsPerBlock) void gn_fwd_affine_40(
    const T* __restrict__ x, const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ y,
    float* __restrict__ mean_out, float* __restrict__ rstd_out, int64_t rows, int G, int ngb, float eps) {
  static_assert(sizeof(T) == 2, "the width-40 kernels are 16-bit");
  cudaGridDependencySynchronize();
  constexpr int V = 8, W = 6, cpg = 40;
  const float inv_cpg = 1.f / static_cast<float>(cpg);
  const int lane = threadIdx.x & 31;
  const int gwid = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
  const int nwarps = gridDim.x * kWarpsPerBlock;
  const int k = gwid % ngb;        // the group block this warp keeps
  const int j = gwid / ngb;        // this warp's place among the warps holding that block
  const int nj = (nwarps - 1 - k) / ngb + 1;
  if (j >= nj) return;

  const int nlanes = cpg / V;                       // lanes of a group
  const int segi = lane / nlanes;                   // which group of the block
  const int sub = lane - segi * nlanes;             // this lane's vector inside the group
  const int g = k * W + segi;                       // group index inside the row
  // Lanes past W groups, or past the row's last group, read group 0 (in bounds) and write nothing.
  const bool act = (segi < W) && (g < G);
  const int gs = act ? g : 0;
  const int64_t C = static_cast<int64_t>(G) * cpg;
  const int64_t chan = static_cast<int64_t>(gs) * cpg + sub * V;

  width::Vec<T, V> gp, bp;
  float gv[V];
  gp = width::load_cached<T, V>(gamma + chan);
  if constexpr (LAYER) bp = width::load_cached<T, V>(beta + chan);
  if constexpr (!LAYER) {
#pragma unroll
    for (int i = 0; i < V; ++i) gv[i] = width::to_f(gp.v[i]);
  }

  const int64_t col = sub * V;
  // A lane's walk is a fixed stride of nj rows: the addresses of the rows in flight advance by constants.
  const int64_t step = (int64_t)nj * C;          // elements between two consecutive rows of this lane's walk
  const int64_t stat_step = (int64_t)nj * G;     // the same stride in statistics units
  auto process = [&](int64_t srow, T* yrow, const width::Vec<T, V>& in) {
    float f[V];
#pragma unroll
    for (int i = 0; i < V; ++i) f[i] = width::to_f(in.v[i]);

    float mu = 0.f;
    float rstd;
    if constexpr (LAYER) {
      float s1 = 0.f;
#pragma unroll
      for (int i = 0; i < V; ++i) s1 += f[i];
      mu = width::group_sum<5>(s1, lane) * inv_cpg;
      float s2 = 0.f;
#pragma unroll
      for (int i = 0; i < V; ++i) {
        const float d = f[i] - mu;
        s2 += d * d;
      }
      rstd = rsqrtf(width::group_sum<5>(s2, lane) * inv_cpg + eps);
    } else {
      float s2 = 0.f;
#pragma unroll
      for (int i = 0; i < V; ++i) s2 += f[i] * f[i];
      rstd = rsqrtf(width::group_sum<5>(s2, lane) * inv_cpg + eps);
    }

    width::Vec<T, V> out;
    if constexpr (LAYER) {
      using P = typename PackPair<T>::type;
      P* po = reinterpret_cast<P*>(&out);
      const P* pg = reinterpret_cast<const P*>(&gp);
      const P* pb = reinterpret_cast<const P*>(&bp);
      const float2 r2 = make_float2(rstd, rstd);
      const float2 m2 = make_float2(-mu, -mu);
#pragma unroll
      for (int i = 0; i < V / 2; ++i) {
        const float2 t = __fmul2_rn(__fadd2_rn(make_float2(f[2 * i], f[2 * i + 1]), m2), r2);
        po[i] = PackPair<T>::narrow(__ffma2_rn(t, PackPair<T>::widen(pg[i]), PackPair<T>::widen(pb[i])));
      }
    } else {
      // The RMS norm: the layer norm's (x * rstd) * gamma + beta with beta = 0 (a -0 product becomes +0).
      constexpr float kNoBeta = 0.f;
#pragma unroll
      for (int i = 0; i < V; ++i) {
        float val = f[i] * rstd;
        val = val * gv[i] + kNoBeta;
        width::from_f(out.v[i], val);
      }
    }
    if (act) {
      width::store_stream<T, V>(yrow, out);
      if (col == 0) {
        rstd_out[srow] = rstd;
        if constexpr (LAYER) mean_out[srow] = mu;
      }
    }
  };

  constexpr int U = 4;
  int64_t r = j;                                  // row index of the next tile of this lane: the bound only
  int64_t srow = (int64_t)j * G + g;               // statistics index of that tile
  const T* xrow = x + (int64_t)j * C + chan;       // its input and output addresses, advanced in place
  T* yrow = y + (int64_t)j * C + chan;
  for (; r + (U - 1) * nj < rows;
       r += U * nj, srow += U * stat_step, xrow += U * step, yrow += U * step) {
    width::Vec<T, V> in[U];
#pragma unroll
    for (int u = 0; u < U; ++u) in[u] = width::load_stream<T, V>(xrow + u * step);
#pragma unroll
    for (int u = 0; u < U; ++u) process(srow + u * stat_step, yrow + u * step, in[u]);
  }
  for (; r < rows; r += nj, srow += stat_step, xrow += step, yrow += step)
    process(srow, yrow, width::load_stream<T, V>(xrow));
}

// The two elements of a 16-bit pair, so a lane's eight results are assembled as values: writing them through a
// pointer into the local vector kept the vector on the stack.
template <typename T>
__device__ __forceinline__ T packed_lo(typename PackPair<T>::type a);
template <typename T>
__device__ __forceinline__ T packed_hi(typename PackPair<T>::type a);
template <>
__device__ __forceinline__ __nv_bfloat16 packed_lo<__nv_bfloat16>(__nv_bfloat162 a) {
  return __low2bfloat16(a);
}
template <>
__device__ __forceinline__ __nv_bfloat16 packed_hi<__nv_bfloat16>(__nv_bfloat162 a) {
  return __high2bfloat16(a);
}
template <>
__device__ __forceinline__ __half packed_lo<__half>(__half2 a) {
  return __low2half(a);
}
template <>
__device__ __forceinline__ __half packed_hi<__half>(__half2 a) {
  return __high2half(a);
}

// The affine 16-bit layer norm at 40 channels a group, on grids of at least four CTAs an SM: gn_fwd_affine_40's
// mapping, compiled for four CTAs an SM (__launch_bounds__(256, 4)). Without STAGED, gamma and beta are held in float
// registers. With STAGED, the CTA first copies the factors of its warps' group blocks into shared memory, in their
// 16-bit type, and every row reads them from there (a pair a time, widened to float); a slot past the row's last
// group stages group 0, and the lane that would use it stores nothing. A slot is 40 halves (80 B), so a warp's reads
// spread over eight banks.
template <typename T, bool STAGED>
__global__ __launch_bounds__(32 * kWarpsPerBlock, 4) void gn_fwd_affine_40_layer(
    const T* __restrict__ x, const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ y,
    float* __restrict__ mean_out, float* __restrict__ rstd_out, int64_t rows, int G, int ngb, float eps) {
  static_assert(sizeof(T) == 2, "the width-40 kernels are 16-bit");
  cudaGridDependencySynchronize();
  constexpr int V = 8, W = 6, cpg = 40;
  const float inv_cpg = 1.f / static_cast<float>(cpg);
  const int lane = threadIdx.x & 31;
  const int gwid = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
  const int nwarps = gridDim.x * kWarpsPerBlock;
  const int k = gwid % ngb;        // the group block this warp keeps
  const int j = gwid / ngb;        // this warp's place among the warps holding that block
  const int nj = (nwarps - 1 - k) / ngb + 1;
  constexpr int SPW = W;                                  // group slots one warp keeps
  constexpr int NF = kWarpsPerBlock * SPW * cpg;          // factor elements this CTA stages per table
  constexpr int NVEC = kWarpsPerBlock * SPW * (cpg / V);  // 16-byte vectors copied: the real channels only
  extern __shared__ __align__(16) unsigned char sr_smem[];
  T* const sres_t = reinterpret_cast<T*>(sr_smem);
  T* const sres_tb = sres_t + (STAGED ? NF : 0);
  if constexpr (STAGED) {
    // Before the early return below: the barrier belongs to the whole CTA.
    for (int i = (int)threadIdx.x; i < NVEC; i += (int)blockDim.x) {
      const int vi = i % (cpg / V);                // this vector's place inside the group slot
      const int wi = (i / (cpg / V)) / SPW;        // the warp slot of this CTA the vector belongs to
      const int gi = (i / (cpg / V)) % SPW;        // the group slot inside that warp's group block
      const int fi = (wi * SPW + gi) * cpg + vi * V;
      const int off = vi * V;
      const int kk = (blockIdx.x * kWarpsPerBlock + wi) % ngb;
      const int gg = kk * W + gi;
      const bool inb = (gi < W) && (gg < G);
      const int64_t src = static_cast<int64_t>(inb ? gg : 0) * cpg + off;
      const width::Vec<T, V> t = width::load_cached<T, V>(gamma + src);
      *reinterpret_cast<width::Vec<T, V>*>(sres_t + fi) = t;
      const width::Vec<T, V> u = width::load_cached<T, V>(beta + src);
      *reinterpret_cast<width::Vec<T, V>*>(sres_tb + fi) = u;
    }
    __syncthreads();
  }
  if (j >= nj) return;

  const int nlanes = cpg / V;                       // lanes of a group
  const int segi = lane / nlanes;                   // which group of the block
  const int sub = lane - segi * nlanes;             // this lane's vector inside the group
  const int g = k * W + segi;                       // group index inside the row
  // Lanes past W groups, or past the row's last group, read group 0 (in bounds) and write nothing.
  const bool act = (segi < W) && (g < G);
  const int gs = act ? g : 0;
  const int64_t C = static_cast<int64_t>(G) * cpg;
  const int64_t chan = static_cast<int64_t>(gs) * cpg + sub * V;

  float gv[V];
  float bv[V];
  if constexpr (!STAGED) {
    const width::Vec<T, V> t = width::load_cached<T, V>(gamma + chan);
#pragma unroll
    for (int i = 0; i < V; ++i) gv[i] = width::to_f(t.v[i]);
    const width::Vec<T, V> u = width::load_cached<T, V>(beta + chan);
#pragma unroll
    for (int i = 0; i < V; ++i) bv[i] = width::to_f(u.v[i]);
  }

  // The shared slot this lane reads: its warp's slots, its group's slot among them (slot 0 for an inactive lane).
  const int sslot = (int)(threadIdx.x >> 5) * SPW + (act ? segi : 0);

  const int64_t col = sub * V;
  auto process = [&](int64_t r, const width::Vec<T, V>& in) {
    const int64_t off = r * C + chan;
    float f[V];
#pragma unroll
    for (int i = 0; i < V; ++i) f[i] = width::to_f(in.v[i]);

    float mu = 0.f;
    float rstd;
    float s1 = 0.f;
#pragma unroll
    for (int i = 0; i < V; ++i) s1 += f[i];
    mu = width::group_sum<5>(s1, lane) * inv_cpg;
    float s2 = 0.f;
#pragma unroll
    for (int i = 0; i < V; ++i) {
      const float d = f[i] - mu;
      s2 += d * d;
    }
    rstd = rsqrtf(width::group_sum<5>(s2, lane) * inv_cpg + eps);

    width::Vec<T, V> out;
    if constexpr (STAGED) {
      using T2 = typename PackPair<T>::type;
      const int fo = sslot * cpg + col;
      const float2 r2 = make_float2(rstd, rstd);
      const float2 m2 = make_float2(-mu, -mu);
#pragma unroll
      for (int q = 0; q < V / 2; ++q) {
        const float2 t = __fmul2_rn(__fadd2_rn(make_float2(f[2 * q], f[2 * q + 1]), m2), r2);
        const float2 gp = PackPair<T>::widen(*reinterpret_cast<const T2*>(sres_t + fo + 2 * q));
        const float2 bp = PackPair<T>::widen(*reinterpret_cast<const T2*>(sres_tb + fo + 2 * q));
        const T2 y2 = PackPair<T>::narrow(__ffma2_rn(t, gp, bp));
        out.v[2 * q] = packed_lo<T>(y2);
        out.v[2 * q + 1] = packed_hi<T>(y2);
      }
    } else {
#pragma unroll
      for (int i = 0; i < V; ++i) {
        float val = (f[i] - mu) * rstd;
        val = val * gv[i] + bv[i];
        width::from_f(out.v[i], val);
      }
    }
    if (act) {
      width::store_stream<T, V>(y + off, out);
      if (col == 0) {
        rstd_out[r * G + g] = rstd;
        mean_out[r * G + g] = mu;
      }
    }
  };

  constexpr int U = 4;
  int64_t r = j;
  for (; r + (U - 1) * nj < rows; r += U * nj) {
    width::Vec<T, V> in[U];
#pragma unroll
    for (int u = 0; u < U; ++u) in[u] = width::load_stream<T, V>(x + (r + u * nj) * C + chan);
#pragma unroll
    for (int u = 0; u < U; ++u) process(r + u * nj, in[u]);
  }
  for (; r < rows; r += nj) process(r, width::load_stream<T, V>(x + r * C + chan));
}

// ---------------------------------------------------------------------------------------------------------------------
// Backward

// Without affine: the forward's lane map (Q = 2 for an even V). dx = rstd * (dy - xhat * mean(dy * xhat)) for the RMS
// norm and rstd * (dy - mean(dy) - xhat * mean(dy * xhat)) for the layer norm, with xhat from x and the statistics,
// or y itself (memory efficient, ME). The means are seg_sum over the group's lanes of a lane's in-order sums.
template <typename T, bool LAYER, bool ME, int ILP, int Q>
__global__ void __launch_bounds__(kBlock) gn_bwd_flat_kernel(const T* __restrict__ xr, const T* __restrict__ dy,
                                                             T* __restrict__ dx, const float* __restrict__ rstd,
                                                             const float* __restrict__ mean, int cpg, int W,
                                                             int64_t total_slots) {
  constexpr int N = Cvt<T>::N;
  constexpr int M = Q * N;
  const int lane = threadIdx.x & 31;
  const int gw = 32 / W;
  const int live = gw * W;
  const int soff = lane / W;
  const int lid = lane - soff * W;
  const float inv_n = 1.0f / (float)cpg;
  const int warps = blockDim.x >> 5;
  const int64_t step = (int64_t)ILP * gw;
  const int64_t wstep = (int64_t)gridDim.x * warps * step;
  int64_t s0 = ((int64_t)blockIdx.x * warps + (threadIdx.x >> 5)) * step;

  // Launched with PDL: wait for the previous kernel before the first global read.
  cudaGridDependencySynchronize();
  for (; s0 < total_slots; s0 += wstep) {
    uint4 rx[ILP][Q], rd[ILP][Q];
    int64_t off[ILP];
    bool ok[ILP];
    float r[ILP], mu[ILP];
    const int64_t jstep = (int64_t)W * N;  // chunk j sits j*W*N elements on: a constant, so one 64-bit address a slot
#pragma unroll
    for (int u = 0; u < ILP; ++u) {
      const int64_t slot = s0 + u * gw + soff;
      ok[u] = (lane < live) && (slot < total_slots);
      off[u] = slot * (int64_t)cpg + (int64_t)lid * N;
      r[u] = ok[u] ? rstd[slot] : 0.f;
      mu[u] = (LAYER && !ME && ok[u]) ? mean[slot] : 0.f;
#pragma unroll
      for (int j = 0; j < Q; ++j) {
        rx[u][j] = ok[u] ? ldg16(xr + off[u] + j * jstep) : make_uint4(0, 0, 0, 0);
        rd[u][j] = ok[u] ? ldg16(dy + off[u] + j * jstep) : make_uint4(0, 0, 0, 0);
      }
    }
    float out[ILP][M];
#pragma unroll
    for (int u = 0; u < ILP; ++u) {
      float xv[M], dv[M], xh[M];
#pragma unroll
      for (int j = 0; j < Q; ++j) {
        Cvt<T>::unpack(rx[u][j], xv + j * N);
        Cvt<T>::unpack(rd[u][j], dv + j * N);
      }
      float s1 = 0.f, s2 = 0.f;
      if (ME) {
#pragma unroll
        for (int k = 0; k < M; ++k) {
          xh[k] = xv[k];
          s2 += dv[k] * xh[k];
          if (LAYER) s1 += dv[k];
        }
      } else if (LAYER) {
#pragma unroll
        for (int k = 0; k < M; ++k) {
          xh[k] = (xv[k] - mu[u]) * r[u];
          s2 += dv[k] * xh[k];
          s1 += dv[k];
        }
      } else {
#pragma unroll
        for (int k = 0; k < M; ++k) {
          xh[k] = xv[k] * r[u];
          s2 += dv[k] * xh[k];
        }
      }
      s2 = seg_sum(s2, lane, W) * inv_n;
      if (LAYER) s1 = seg_sum(s1, lane, W) * inv_n;
#pragma unroll
      for (int k = 0; k < M; ++k) out[u][k] = LAYER ? (dv[k] - s1 - xh[k] * s2) * r[u] : (dv[k] - xh[k] * s2) * r[u];
    }
#pragma unroll
    for (int u = 0; u < ILP; ++u) {
      if (ok[u]) {
#pragma unroll
        for (int j = 0; j < Q; ++j) stg16(dx + off[u] + j * jstep, Cvt<T>::pack(out[u] + j * N));
      }
    }
  }
}

// With affine: a warp owns the columns of one tile of gw groups (one vector a lane) and walks a chunk of rows, RB rows
// in flight, so dgamma and dbeta accumulate in registers. Its partial sums go to ws[chunk][C] (dbeta's after
// nchunks * C), one writer an element and no atomics, and gn_dg_reduce_kernel folds the chunks. x and dy are read once.
template <typename T, bool LAYER, bool ME, int RB>
__global__ void __launch_bounds__(kBlock) gn_bwd_aff_kernel(const T* __restrict__ xr, const T* __restrict__ dy,
                                                            T* __restrict__ dx, const float* __restrict__ rstd,
                                                            const float* __restrict__ mean, const T* __restrict__ gamma,
                                                            const T* __restrict__ beta, int G, int cpg, int W, int rows,
                                                            int tiles_per_row, int nchunks, int rows_per_chunk, int C,
                                                            float* __restrict__ ws) {
  constexpr int N = Cvt<T>::N;
  const int lane = threadIdx.x & 31;
  const int gw = 32 / W;
  const int soff = lane / W;
  const int lid = lane - soff * W;
  const float inv_n = 1.0f / (float)cpg;
  const int warps = blockDim.x >> 5;
  const int ntasks = tiles_per_row * nchunks;
  const int ntask_warps = gridDim.x * warps;
  for (int t = blockIdx.x * warps + (threadIdx.x >> 5); t < ntasks; t += ntask_warps) {
    const int tile = t % tiles_per_row;
    const int chunk = t / tiles_per_row;
    const int g = tile * gw + soff;
    const int64_t r0 = (int64_t)chunk * rows_per_chunk;
    const int64_t r1 = (r0 + rows_per_chunk < rows) ? r0 + rows_per_chunk : rows;
    // No `continue` per lane: every lane of the warp must reach the shuffles below.
    const bool active = (lane < gw * W) && (g < G);
    const int64_t col = active ? (int64_t)g * cpg + (int64_t)lid * N : 0;
    float gvv[N], bvv[N];
    Cvt<T>::unpack(active ? ldg16(gamma + col) : make_uint4(0, 0, 0, 0), gvv);
    if (LAYER) Cvt<T>::unpack(active ? ldg16(beta + col) : make_uint4(0, 0, 0, 0), bvv);
    float dga[N], dbe[N];
#pragma unroll
    for (int k = 0; k < N; ++k) {
      dga[k] = 0.f;
      dbe[k] = 0.f;
    }
    for (int64_t row = r0; row < r1; row += RB) {
      uint4 rx[RB], rd[RB];
      int64_t off[RB];
      bool ok[RB];
#pragma unroll
      for (int i = 0; i < RB; ++i) {
        ok[i] = active && (row + i < r1);
        const int64_t rw = (row + i < r1) ? row + i : r1 - 1;
        off[i] = rw * (int64_t)C + col;
        rx[i] = ok[i] ? ldg16(xr + off[i]) : make_uint4(0, 0, 0, 0);
        rd[i] = ok[i] ? ldg16(dy + off[i]) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int i = 0; i < RB; ++i) {
        const int64_t rw = (row + i < r1) ? row + i : r1 - 1;
        const float r = ok[i] ? rstd[rw * G + g] : 0.f;
        const float mu = (LAYER && !ME && ok[i]) ? mean[rw * G + g] : 0.f;
        float xv[N], dv[N], xh[N], dh[N];
        Cvt<T>::unpack(rx[i], xv);
        Cvt<T>::unpack(rd[i], dv);
        float s1 = 0.f, s2 = 0.f;
#pragma unroll
        for (int k = 0; k < N; ++k) {
          if (ME)
            xh[k] = LAYER ? (xv[k] - bvv[k]) / gvv[k] : xv[k] / gvv[k];
          else
            xh[k] = LAYER ? (xv[k] - mu) * r : xv[k] * r;
          dh[k] = dv[k] * gvv[k];
          s2 += dh[k] * xh[k];
          if (LAYER) s1 += dh[k];
        }
        s2 = seg_sum(s2, lane, W) * inv_n;
        if (LAYER) s1 = seg_sum(s1, lane, W) * inv_n;
        const float wgt = ok[i] ? 1.f : 0.f;
        float out[N];
#pragma unroll
        for (int k = 0; k < N; ++k) {
          out[k] = LAYER ? (dh[k] - s1 - xh[k] * s2) * r : (dh[k] - xh[k] * s2) * r;
          dga[k] += wgt * (dv[k] * xh[k]);
          dbe[k] += wgt * dv[k];
        }
        if (ok[i]) stg16(dx + off[i], Cvt<T>::pack(out));
      }
    }
    if (active) {
      float* w = ws + (int64_t)chunk * C + col;
#pragma unroll
      for (int k = 0; k < N; ++k) w[k] = dga[k];
      if (LAYER) {
        float* w2 = ws + (int64_t)nchunks * C + (int64_t)chunk * C + col;
#pragma unroll
        for (int k = 0; k < N; ++k) w2[k] = dbe[k];
      }
    }
  }
}

// dgamma (and dbeta) from the row-chunk partials, in a fixed order: a CTA of 256 threads is 8 parts of 32 columns;
// part p sums its share of the chunks with four interleaved accumulators folded as (a0 + a1) + (a2 + a3), and part 0
// adds the 8 parts in order.
template <typename T>
__global__ void gn_dg_reduce_kernel(const float* __restrict__ ws, T* __restrict__ dgamma, T* __restrict__ dbeta,
                                    int C, int nchunks, bool has_beta) {
  const int part = threadIdx.x >> 5;  // 0..7
  const int lcol = threadIdx.x & 31;  // 0..31
  const int col = blockIdx.x * 32 + lcol;
  __shared__ float sdg[8][32];
  __shared__ float sdb[8][32];
  const int per = (nchunks + 7) >> 3;
  const int lo = part * per;
  const int hi = (lo + per < nchunks) ? lo + per : nchunks;
  float s = 0.f, s2 = 0.f;
  if (col < C) {
    const float* w0 = ws + col;
    const float* w1 = ws + (int64_t)nchunks * C + col;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
    int i = lo;
    if (has_beta) {
      for (; i + 3 < hi; i += 4) {
        const int64_t o0 = (int64_t)(i + 0) * C, o1 = (int64_t)(i + 1) * C;
        const int64_t o2 = (int64_t)(i + 2) * C, o3 = (int64_t)(i + 3) * C;
        const float x0 = w0[o0], x1 = w0[o1], x2 = w0[o2], x3 = w0[o3];
        const float y0 = w1[o0], y1 = w1[o1], y2 = w1[o2], y3 = w1[o3];
        a0 += x0; a1 += x1; a2 += x2; a3 += x3;
        b0 += y0; b1 += y1; b2 += y2; b3 += y3;
      }
      for (; i < hi; ++i) {
        a0 += w0[(int64_t)i * C];
        b0 += w1[(int64_t)i * C];
      }
      s = (a0 + a1) + (a2 + a3);
      s2 = (b0 + b1) + (b2 + b3);
    } else {
      for (; i + 3 < hi; i += 4) {
        const int64_t o0 = (int64_t)(i + 0) * C, o1 = (int64_t)(i + 1) * C;
        const int64_t o2 = (int64_t)(i + 2) * C, o3 = (int64_t)(i + 3) * C;
        const float x0 = w0[o0], x1 = w0[o1], x2 = w0[o2], x3 = w0[o3];
        a0 += x0; a1 += x1; a2 += x2; a3 += x3;
      }
      for (; i < hi; ++i) {
        a0 += w0[(int64_t)i * C];
      }
      s = (a0 + a1) + (a2 + a3);
      s2 = 0.f;
    }
  }
  sdg[part][lcol] = s;
  sdb[part][lcol] = s2;
  __syncthreads();
  if (part == 0 && col < C) {
    float t = 0.f, t2 = 0.f;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      t += sdg[p][lcol];
      t2 += sdb[p][lcol];
    }
    dgamma[col] = Cvt<T>::from_float(t);
    if (has_beta) dbeta[col] = Cvt<T>::from_float(t2);
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Host side

// CTAs of kWarpsPerBlock warps for work_units at units_per_warp a warp, at most blocks_per_sm CTAs an SM.
inline int grid_blocks(int64_t work_units, int64_t units_per_warp, int sm, int blocks_per_sm) {
  const int64_t warps = (work_units + units_per_warp - 1) / units_per_warp;
  int64_t blocks = (warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int64_t cap = (int64_t)sm * blocks_per_sm;
  if (blocks > cap) blocks = cap;
  if (blocks < 1) blocks = 1;
  return (int)blocks;
}

// The CTAs of one gn_bwd_flat_kernel instantiation the device holds at once: sm times its occupancy, which is queried
// once per instantiation and device.
template <typename T, bool LAYER, bool ME, int ILP, int Q>
inline int64_t flat_one_wave(int sm, int dev) {
  static int cache[64] = {0};  // [device ordinal]
  if (dev < 0 || dev > 63) dev = 0;
  if (cache[dev] <= 0) {
    int r = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r, gn_bwd_flat_kernel<T, LAYER, ME, ILP, Q>, kBlock, 0) !=
            cudaSuccess ||
        r <= 0) {
      r = 1;
    }
    cache[dev] = r;
  }
  return (int64_t)sm * cache[dev];
}

// The operands of one call: x (or y, for a memory-efficient backward) and its outputs, y or dx.
template <typename T>
struct Operands {
  const T* x;
  const T* dy;
  const T* gamma;
  const T* beta;
  const float* mean;
  const float* rstd;
  T* out;
  float* mean_out;
  float* rstd_out;
  int64_t rows;
  int G;
  int cpg;
  int V;  // 16-byte vectors a group
  int sm;
  size_t smem_per_sm;
  int device;
  cudaStream_t stream;
};

template <typename T>
Operands<T> operands(const torch::Tensor& x, const torch::Tensor& gamma, const torch::Tensor& beta,
                     const torch::Tensor& out, int64_t num_groups, bool layer, bool affine) {
  const int64_t C = x.size(-1);
  // x's device is the current device: the entry points hold a device guard.
  const c10::DeviceIndex device = x.device().index();
  const cudaDeviceProp* props = at::cuda::getDeviceProperties(device);
  const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(device);
  Operands<T> o{};
  o.x = reinterpret_cast<const T*>(x.const_data_ptr());
  o.gamma = affine ? reinterpret_cast<const T*>(gamma.const_data_ptr()) : nullptr;
  o.beta = (affine && layer) ? reinterpret_cast<const T*>(beta.const_data_ptr()) : nullptr;
  o.out = reinterpret_cast<T*>(out.data_ptr());
  o.rows = x.numel() / C;
  o.G = (int)num_groups;
  o.cpg = (int)(C / num_groups);
  o.V = (int)(o.cpg * (int64_t)sizeof(T) / 16);
  o.sm = props->multiProcessorCount;
  o.smem_per_sm = props->sharedMemPerMultiprocessor;
  o.device = device;
  o.stream = stream;
  return o;
}

template <typename T, bool LAYER, bool AFFINE, int ILP, int Q>
void launch_fwd_kernel(const Operands<T>& o, int W, bool pow2, float eps) {
  const int64_t total_slots = o.rows * o.G;
  const int grid = grid_blocks(total_slots, (int64_t)ILP * (32 / W), o.sm, 64);
  auto issue = [&](auto kernel) {
    LaunchKernelPdl(kernel, dim3(grid), dim3(kBlock), 0, o.stream, o.x, o.out, o.rstd_out,
                    o.mean_out, o.gamma, o.beta, o.G, o.cpg, W, eps, total_slots);
  };
  if (pow2) issue(gn_fwd_kernel<T, LAYER, AFFINE, ILP, Q, true>);
  else issue(gn_fwd_kernel<T, LAYER, AFFINE, ILP, Q, false>);
}

// gn_fwd_kernel: Q = 2 (two chunks a lane) for an even V without affine. A lane carries ILP * Q chunks in flight: 2
// with affine (which also holds gamma and beta), 4 without, and 2 on the Q = 2 arm where its two-slot grid is short of
// the machine's resident capacity (4 CTAs an SM in 16 bits, 5 in fp32) and the one-slot grid has more CTAs. Grids of
// at most 64 CTAs an SM.
template <typename T, bool LAYER>
void launch_fwd_lanes(const Operands<T>& o, bool affine, float eps) {
  const int Q = (!affine && o.V >= 2 && (o.V % 2) == 0) ? 2 : 1;
  const int W = o.V / Q;
  const bool pow2 = (W & (W - 1)) == 0;
  const int64_t total_slots = o.rows * o.G;
  constexpr int kResidentCtas2 = (sizeof(T) == 2) ? 4 : 5;
  if (affine) return launch_fwd_kernel<T, LAYER, true, 2, 1>(o, W, pow2, eps);
  if (Q == 1) return launch_fwd_kernel<T, LAYER, false, 4, 1>(o, W, pow2, eps);
  const int two_slot = grid_blocks(total_slots, (int64_t)2 * (32 / W), o.sm, 64);
  if (two_slot < static_cast<int64_t>(o.sm) * kResidentCtas2 &&
      grid_blocks(total_slots, (int64_t)1 * (32 / W), o.sm, 64) > two_slot)
    return launch_fwd_kernel<T, LAYER, false, 1, 2>(o, W, pow2, eps);
  launch_fwd_kernel<T, LAYER, false, 2, 2>(o, W, pow2, eps);
}

// 16-bit affine calls of 40 channels a group take the width-40 kernels: warps keep a block of six groups (group_blocks
// a row) and walk the rows, at most 8 CTAs an SM. The layer norm takes gn_fwd_affine_40_layer (four CTAs an SM) on
// grids of more than three CTAs an SM, where gn_fwd_affine_40 (three an SM at its register count) would need a second
// wave, with its factors staged in shared memory where a warp walks at least 8 rows (the copy is amortised) and four
// CTAs' fp32 copies of them would fit in an SM's shared memory.
template <typename T>
void launch_fwd_40(const Operands<T>& o, bool layer, float eps) {
  const int group_blocks = (o.G + 5) / 6;
  const int64_t blocks = std::max<int64_t>(
      1, std::min<int64_t>((o.rows * group_blocks + kWarpsPerBlock - 1) / kWarpsPerBlock,
                           static_cast<int64_t>(o.sm) * 8));
  auto issue = [&](auto kernel, size_t smem) {
    LaunchKernelPdl(kernel, dim3(blocks), dim3(kBlock), smem, o.stream, o.x, o.gamma, o.beta, o.out,
                    o.mean_out, o.rstd_out, o.rows, o.G, group_blocks, eps);
  };
  if (!layer) {
    issue(gn_fwd_affine_40<T, false>, 0);
  } else if (blocks <= static_cast<int64_t>(o.sm) * 3) {
    issue(gn_fwd_affine_40<T, true>, 0);
  } else {
    const int64_t warp_rows = (o.rows * group_blocks) / (blocks * kWarpsPerBlock);
    const int64_t fp32_copy_bytes = (int64_t)kWarpsPerBlock * 6 * (40 + 4) * (int64_t)sizeof(float) * 2;
    if (warp_rows >= 8 && fp32_copy_bytes * 4 <= (int64_t)o.smem_per_sm) {
      // gamma and beta of the CTA's warps' groups, in their own 16-bit type.
      issue(gn_fwd_affine_40_layer<T, true>, (size_t)kWarpsPerBlock * 6 * 40 * sizeof(T) * 2);
    } else {
      issue(gn_fwd_affine_40_layer<T, false>, 0);
    }
  }
}

template <typename T>
void launch_fwd(const Operands<T>& o, bool layer, bool affine, float eps) {
  if constexpr (sizeof(T) == 2) {
    if (affine && o.cpg == 40) return launch_fwd_40<T>(o, layer, eps);
  }
  if (layer) launch_fwd_lanes<T, true>(o, affine, eps);
  else launch_fwd_lanes<T, false>(o, affine, eps);
}

// One gn_bwd_flat_kernel launch, with PDL; where the device refuses that launch, the same launch without it.
template <typename T, bool LAYER, bool ME, int ILP, int Q>
void launch_bwd_flat_kernel(const Operands<T>& o, int W) {
  const int64_t total_slots = o.rows * o.G;
  int nb = grid_blocks(total_slots, (int64_t)ILP * (32 / W), o.sm, 64);
  // A grid of four to five waves of resident CTAs is clamped to one wave: the kernel grid-strides, and there the extra
  // passes cost more than they buy. Outside that band the clamp measured slower (above five waves on the device, below
  // four on the public call).
  const int64_t ow = flat_one_wave<T, LAYER, ME, ILP, Q>(o.sm, o.device);
  if (ow > 0 && (int64_t)nb >= 4 * ow && (int64_t)nb <= 5 * ow) nb = (int)ow;
  cudaLaunchAttribute pdl_attr[1] = {};
  pdl_attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdl_attr[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t pdl_cfg{};
  pdl_cfg.gridDim = dim3(nb);
  pdl_cfg.blockDim = dim3(kBlock);
  pdl_cfg.dynamicSmemBytes = 0;
  pdl_cfg.stream = o.stream;
  pdl_cfg.attrs = pdl_attr;
  pdl_cfg.numAttrs = 1;
  const cudaError_t pdl_err = cudaLaunchKernelEx(&pdl_cfg, gn_bwd_flat_kernel<T, LAYER, ME, ILP, Q>, o.x, o.dy, o.out,
                                                 o.rstd, o.mean, o.cpg, W, total_slots);
  if (pdl_err != cudaSuccess) {
    (void)cudaGetLastError();
    gn_bwd_flat_kernel<T, LAYER, ME, ILP, Q><<<nb, kBlock, 0, o.stream>>>(o.x, o.dy, o.out, o.rstd, o.mean, o.cpg, W,
                                                                         total_slots);
    C10_CUDA_CHECK(cudaGetLastError());
  }
}

// Without affine: Q = 2 for an even V, as the forward. One slot a lane (ILP 1) at up to 128 rows of up to 128
// channels, and where two slots would leave more than half the SMs without a CTA, but on the memory-efficient fp32
// backward, which measured slower so; two slots otherwise.
template <typename T, bool LAYER, bool ME, int Q>
void launch_bwd_flat_ilp(const Operands<T>& o, int W) {
  if constexpr (!(sizeof(T) == 4 && ME)) {
    if ((o.rows <= 128 && o.cpg <= 128) || 2 * grid_blocks(o.rows * o.G, 2 * (32 / W), o.sm, 64) < o.sm)
      return launch_bwd_flat_kernel<T, LAYER, ME, 1, Q>(o, W);
  }
  launch_bwd_flat_kernel<T, LAYER, ME, 2, Q>(o, W);
}

template <typename T, bool LAYER, bool ME>
void launch_bwd_flat(const Operands<T>& o) {
  if (o.V >= 2 && (o.V % 2) == 0) launch_bwd_flat_ilp<T, LAYER, ME, 2>(o, o.V / 2);
  else launch_bwd_flat_ilp<T, LAYER, ME, 1>(o, o.V);
}

template <bool LAYER, bool ME>
void launch_bwd_affine_kernel(const Operands<float>& o, int blocks, int tiles_per_row, int nchunks, int rows_per_chunk,
                              int C, float* ws) {
  gn_bwd_aff_kernel<float, LAYER, ME, 4><<<blocks, kBlock, 0, o.stream>>>(o.x, o.dy, o.out, o.rstd, o.mean, o.gamma,
                                                                          o.beta, o.G, o.cpg, o.V, (int)o.rows,
                                                                          tiles_per_row, nchunks, rows_per_chunk, C,
                                                                          ws);
}

// With affine (fp32 only, see Bwd): one warp per (column tile, row chunk), with 64 warps' worth of chunks an SM so
// that the tasks alone fill the machine; at most 256 chunks, which bounds the partials at 8 MB at C = 8192. At most 8
// CTAs an SM: the kernel is register-bound, so a longer grid only adds a tail.
void launch_bwd_affine(const Operands<float>& o, bool layer, bool me, int C, float* dgamma, float* dbeta,
                       const at::TensorOptions& options) {
  const int gw = 32 / o.V;
  const int tiles_per_row = (o.G + gw - 1) / gw;
  int64_t nchunks = (int64_t)o.sm * 64 / tiles_per_row;
  if (nchunks > 256) nchunks = 256;
  if (nchunks < 1) nchunks = 1;
  if (nchunks > o.rows) nchunks = o.rows;
  int64_t rows_per_chunk = (o.rows + nchunks - 1) / nchunks;
  nchunks = (o.rows + rows_per_chunk - 1) / rows_per_chunk;
  const int64_t tasks = (int64_t)tiles_per_row * nchunks;
  int blocks = (int)((tasks + kWarpsPerBlock - 1) / kWarpsPerBlock);
  const int cap = o.sm * 8;
  if (blocks > cap) blocks = cap;
  if (blocks < 1) blocks = 1;

  auto ws = at::empty({(int64_t)nchunks * C * (layer ? 2 : 1)}, options.dtype(at::kFloat));
  float* wsp = ws.data_ptr<float>();
  const int nc = (int)nchunks, rpc = (int)rows_per_chunk;
  if (layer) {
    if (me) launch_bwd_affine_kernel<true, true>(o, blocks, tiles_per_row, nc, rpc, C, wsp);
    else launch_bwd_affine_kernel<true, false>(o, blocks, tiles_per_row, nc, rpc, C, wsp);
  } else {
    if (me) launch_bwd_affine_kernel<false, true>(o, blocks, tiles_per_row, nc, rpc, C, wsp);
    else launch_bwd_affine_kernel<false, false>(o, blocks, tiles_per_row, nc, rpc, C, wsp);
  }
  gn_dg_reduce_kernel<float><<<(C + 31) / 32, 256, 0, o.stream>>>(wsp, dgamma, dbeta, C, nc, layer);
}

}  // namespace

// Is the plain forward's grid already at the machine's resident capacity? The same slots a warp (ILP * (32 / W), the
// two-slot arm for an even V, the four-slot one otherwise), the same 64 CTAs an SM and the same 4 (16-bit) / 5 (fp32)
// resident CTAs as launch_fwd_lanes, so the two cannot disagree.
bool GridSaturated(int64_t rows, int64_t groups, int64_t cpg, int elemsize) {
  const int64_t V = cpg * elemsize / 16;
  if (V < 1 || V > 32) return false;
  const int64_t Q = (V >= 2 && (V % 2) == 0) ? 2 : 1;
  const int64_t W = V / Q;
  const int64_t per_warp = (Q == 2 ? 2 : 4) * (32 / W);
  const int sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t ctas = grid_blocks(rows * groups, per_warp, sm, 64);
  return ctas >= (int64_t)sm * (elemsize == 2 ? 4 : 5);
}

bool Eligible(const torch::Tensor& x, int64_t num_groups) {
  const int64_t channels = x.size(-1);
  const int64_t group_bytes = channels / num_groups * x.element_size();
  return group_bytes % 16 == 0 && group_bytes <= 32 * 16 && x.numel() >= channels;
}

void Fwd(const torch::Tensor& x, int64_t num_groups, bool layer, const torch::Tensor& gamma, const torch::Tensor& beta,
         double eps, const torch::Tensor& y, const torch::Tensor& mean_out, const torch::Tensor& rstd_out) {
  const bool affine = gamma.numel() > 0;
  auto run = [&](auto type_tag) {
    using T = decltype(type_tag);
    Operands<T> o = operands<T>(x, gamma, beta, y, num_groups, layer, affine);
    o.mean_out = layer ? mean_out.data_ptr<float>() : nullptr;
    o.rstd_out = rstd_out.data_ptr<float>();
    launch_fwd<T>(o, layer, affine, (float)eps);
  };
  switch (x.scalar_type()) {
    case at::kFloat:
      return run(float{});
    case at::kBFloat16:
      return run(__nv_bfloat16{});
    default:
      return run(__half{});
  }
}

// The dispatcher sends fp32 calls here, and bf16 calls without affine but the memory-efficient layer norm: fp16 takes
// the generic kernels up to 384 channels a group (dense lanes end at 256), the bf16 layer norm without affine from y
// takes them up to 256, and GroupNormBwdOnePass takes every bf16 call with affine of 8..256 channels.
void Bwd(const torch::Tensor& x_or_y, const torch::Tensor& dy, int64_t num_groups, bool layer, bool memory_efficient,
         const torch::Tensor& mean, const torch::Tensor& rstd, const torch::Tensor& gamma, const torch::Tensor& beta,
         const torch::Tensor& dx, const torch::Tensor& dgamma, const torch::Tensor& dbeta) {
  const bool affine = gamma.numel() > 0;
  auto run = [&](auto type_tag) {
    using T = decltype(type_tag);
    Operands<T> o = operands<T>(x_or_y, gamma, beta, dx, num_groups, layer, affine);
    o.dy = reinterpret_cast<const T*>(dy.const_data_ptr());
    o.rstd = rstd.const_data_ptr<float>();
    o.mean = (layer && !memory_efficient) ? mean.const_data_ptr<float>() : nullptr;
    if (!affine) {
      if (layer) {
        if (!memory_efficient) return launch_bwd_flat<T, true, false>(o);
        if constexpr (std::is_same_v<T, float>) launch_bwd_flat<T, true, true>(o);
        else TORCH_INTERNAL_ASSERT(false, "dense-lane backward takes no bf16 memory-efficient layer norm");
      } else {
        if (memory_efficient) launch_bwd_flat<T, false, true>(o);
        else launch_bwd_flat<T, false, false>(o);
      }
      return;
    }
    if constexpr (std::is_same_v<T, float>) {
      launch_bwd_affine(o, layer, memory_efficient, (int)x_or_y.size(-1), dgamma.data_ptr<float>(),
                        layer ? dbeta.data_ptr<float>() : nullptr, x_or_y.options());
    } else {
      TORCH_INTERNAL_ASSERT(false, "dense-lane backward with affine takes fp32 calls only");
    }
  };
  switch (x_or_y.scalar_type()) {
    case at::kFloat:
      return run(float{});
    case at::kBFloat16:
      return run(__nv_bfloat16{});
    default:
      TORCH_INTERNAL_ASSERT(false, "dense-lane backward takes fp32 and bf16 calls only");
  }
}

}  // namespace xllm::ops::blackwell::dense_lanes
