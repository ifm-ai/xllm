#include <ATen/AccumulateType.h>
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IntegerDivider.cuh>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/MaybeOwned.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_pipeline.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>

#include <ATen/native/cuda/block_reduce.cuh>
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <cub/cub.cuh>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <tuple>

#include "cuda_utils.cuh"
#include "ops/timestep_decay_norm_lookback.h"
#include "reduce.cuh"
#include "utils.h"
#include "welford.h"

namespace xllm {
namespace ops {

namespace {

constexpr int kLookbackBwdBlockSize = 512;
constexpr int kLookbackMomentBlockSize = 32;
constexpr int kParamGradBlockSize = 256;
constexpr int kLookbackTeamBlockSize = 256;
constexpr int kLookbackMaxElementwiseBlockSize = 256;
constexpr int kLookbackElementwiseTokens = 8;
// The count kernels use kLookbackCountBlockSize threads, or
// kLookbackLongCountBlockSize for sequences longer than
// kLookbackLongCountLength.
constexpr int kLookbackCountBlockSize = 128;
constexpr int kLookbackLongCountBlockSize = 1024;
constexpr int64_t kLookbackLongCountLength = 4096;
constexpr int kLookbackCountItemsPerThread = 4;
// Long-sequence count kernels use one block per kLookbackLongCountBlockSize timesteps.
constexpr int kLookbackLongCountItemsPerThread = 1;
// The scans over time run tiles of kLookbackScanBlockSize * kLookbackScanItems
// timesteps, kLookbackScanItems per thread.
constexpr int kLookbackScanBlockSize = 256;
constexpr int kLookbackScanItems = 4;

void CheckOptionalMaskShape(const c10::optional<torch::Tensor>& mask,
                            int64_t B, int64_t L, const char* mask_name) {
  if (!mask.has_value() || !mask->defined()) {
    return;
  }

  TORCH_CHECK(
      mask->dim() == 2,
      mask_name,
      " must have shape [B, L], but got dim=",
      mask->dim());
  TORCH_CHECK(
      mask->size(0) == B && mask->size(1) == L,
      mask_name,
      " must have shape [",
      B,
      ", ",
      L,
      "], but got [",
      mask->size(0),
      ", ",
      mask->size(1),
      "]");
}

void CheckTimestepDecayNormLookbackFwdInputs(
    const torch::Tensor& x, const c10::optional<torch::Tensor>& bos_mask,
    const torch::Tensor& prev_count, const torch::Tensor& prev_mean,
    const torch::Tensor& prev_var, const torch::Tensor& gamma,
    const torch::Tensor& beta, const c10::optional<torch::Tensor>& padding_mask,
    int64_t num_groups, double beta1, double beta2, double eps) {
  TORCH_CHECK(x.dim() == 3, "x must have shape [B, L, H]");
  const int64_t B = x.size(0);
  const int64_t L = x.size(1);
  const int64_t H = x.size(2);

  TORCH_CHECK(B > 0, "B must be positive, but got B=", B);
  TORCH_CHECK(L > 0, "L must be positive, but got L=", L);
  TORCH_CHECK(H > 0, "H must be positive, but got H=", H);
  TORCH_CHECK(
      num_groups > 0,
      "num_groups must be positive, but got num_groups=",
      num_groups);
  TORCH_CHECK(
      H % num_groups == 0,
      "H must be divisible by num_groups, but got H=",
      H,
      ", num_groups=",
      num_groups);
  TORCH_CHECK(
      0.0 <= beta1 && beta1 < 1.0,
      "beta1 must be in [0, 1), but got beta1=",
      beta1);
  TORCH_CHECK(
      0.0 <= beta2 && beta2 < 1.0,
      "beta2 must be in [0, 1), but got beta2=",
      beta2);
  TORCH_CHECK(eps > 0.0, "eps must be positive, but got eps=", eps);

  TORCH_CHECK(
      prev_count.dim() == 1 && prev_count.size(0) == B,
      "prev_count must have shape [B], but got [",
      prev_count.size(0),
      "]");
  TORCH_CHECK(
      prev_mean.dim() == 2 && prev_mean.size(0) == B &&
          prev_mean.size(1) == num_groups,
      "prev_mean must have shape [B, G], but got [",
      prev_mean.size(0),
      ", ",
      prev_mean.size(1),
      "]");
  TORCH_CHECK(
      prev_var.dim() == 2 && prev_var.size(0) == B &&
          prev_var.size(1) == num_groups,
      "prev_var must have shape [B, G], but got [",
      prev_var.size(0),
      ", ",
      prev_var.size(1),
      "]");
  TORCH_CHECK(
      gamma.dim() == 1 && gamma.size(0) == H,
      "gamma must have shape [H], but got [",
      gamma.size(0),
      "]");
  TORCH_CHECK(
      beta.dim() == 1 && beta.size(0) == H,
      "beta must have shape [H], but got [",
      beta.size(0),
      "]");

  CheckOptionalMaskShape(bos_mask, B, L, "bos_mask");
  CheckOptionalMaskShape(padding_mask, B, L, "padding_mask");
}

void CheckTimestepDecayNormLookbackBwdInputs(
    const torch::Tensor& y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& x,
    const torch::Tensor& prev_count, const c10::optional<torch::Tensor>& bos_mask,
    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma, const torch::Tensor& beta,
    const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups,
    double beta1, double beta2, double eps) {
  TORCH_CHECK(x.dim() == 3, "x must have shape [B, L, H]");
  const int64_t B = x.size(0);
  const int64_t L = x.size(1);
  const int64_t H = x.size(2);

  TORCH_CHECK(B > 0, "B must be positive, but got B=", B);
  TORCH_CHECK(L > 0, "L must be positive, but got L=", L);
  TORCH_CHECK(H > 0, "H must be positive, but got H=", H);
  TORCH_CHECK(
      num_groups > 0,
      "num_groups must be positive, but got num_groups=",
      num_groups);
  TORCH_CHECK(
      H % num_groups == 0,
      "H must be divisible by num_groups, but got H=",
      H,
      ", num_groups=",
      num_groups);
  TORCH_CHECK(
      0.0 <= beta1 && beta1 < 1.0,
      "beta1 must be in [0, 1), but got beta1=",
      beta1);
  TORCH_CHECK(
      0.0 <= beta2 && beta2 < 1.0,
      "beta2 must be in [0, 1), but got beta2=",
      beta2);
  TORCH_CHECK(eps > 0.0, "eps must be positive, but got eps=", eps);

  TORCH_CHECK(
      y_grad.sizes() == x.sizes(),
      "y_grad must have the same shape as x, but got y_grad shape ",
      y_grad.sizes(),
      " and x shape ",
      x.sizes());
  TORCH_CHECK(
      prev_count.dim() == 1 && prev_count.size(0) == B,
      "prev_count must have shape [B], but got [",
      prev_count.size(0),
      "]");
  TORCH_CHECK(
      mean_grad.dim() == 2 && mean_grad.size(0) == B &&
          mean_grad.size(1) == num_groups,
      "mean_grad must have shape [B, G], but got [",
      mean_grad.size(0),
      ", ",
      mean_grad.size(1),
      "]");
  TORCH_CHECK(
      var_grad.dim() == 2 && var_grad.size(0) == B &&
          var_grad.size(1) == num_groups,
      "var_grad must have shape [B, G], but got [",
      var_grad.size(0),
      ", ",
      var_grad.size(1),
      "]");
  TORCH_CHECK(
      cummean.dim() == 3 && cummean.size(0) == B &&
          cummean.size(1) == num_groups && cummean.size(2) == L,
      "cummean must have shape [B, G, L], but got [",
      cummean.size(0),
      ", ",
      cummean.size(1),
      ", ",
      cummean.size(2),
      "]");
  TORCH_CHECK(
      cumrstd.dim() == 3 && cumrstd.size(0) == B &&
          cumrstd.size(1) == num_groups && cumrstd.size(2) == L,
      "cumrstd must have shape [B, G, L], but got [",
      cumrstd.size(0),
      ", ",
      cumrstd.size(1),
      ", ",
      cumrstd.size(2),
      "]");
  TORCH_CHECK(
      gamma.dim() == 1 && gamma.size(0) == H,
      "gamma must have shape [H], but got [",
      gamma.size(0),
      "]");
  TORCH_CHECK(
      beta.dim() == 1 && beta.size(0) == H,
      "beta must have shape [H], but got [",
      beta.size(0),
      "]");

  CheckOptionalMaskShape(bos_mask, B, L, "bos_mask");
  CheckOptionalMaskShape(padding_mask, B, L, "padding_mask");
}

__device__ __forceinline__ int64_t DeviceDivUp(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

__device__ __forceinline__ float4 MakeAffine4(float x_add, float x_mul,
                                              float y_add, float y_mul) {
  return make_float4(x_add, x_mul, y_add, y_mul);
}

__device__ __forceinline__ float4 IdentityAffine4() {
  return make_float4(0.0f, 1.0f, 0.0f, 1.0f);
}

// Compose prefix, then current; a constant current map discards the prefix.
__device__ __forceinline__ float4 ComposeAffine4(const float4& prefix,
                                                 const float4& current) {
  if (current.y == 0.0f && current.w == 0.0f) {
    return current;
  }
  return make_float4(fmaf(current.y, prefix.x, current.x),
                     current.y * prefix.y,
                     fmaf(current.w, prefix.z, current.z),
                     current.w * prefix.w);
}

// Multiply with a separate rounding step before addition.
__device__ __forceinline__ float MulRn(float a, float b) {
  return __fmul_rn(a, b);
}
__device__ __forceinline__ double MulRn(double a, double b) {
  return __dmul_rn(a, b);
}

__device__ __forceinline__ float ApplyAffineX(const float4& affine,
                                              float input) {
  return fmaf(affine.y, input, affine.x);
}

__device__ __forceinline__ float ApplyAffineY(const float4& affine,
                                              float input) {
  return fmaf(affine.w, input, affine.z);
}

struct Affine4Prod {
  __device__ __forceinline__ float4 operator()(const float4& prefix,
                                               const float4& current) const {
    return ComposeAffine4(prefix, current);
  }
};

__device__ __forceinline__ int64_t XIndex(int64_t b, int64_t t, int64_t h,
                                          int64_t L, int64_t H) {
  return (b * L + t) * H + h;
}

__device__ __forceinline__ int64_t GroupIndex(int64_t b, int64_t g, int64_t t,
                                              int64_t G, int64_t L) {
  return (b * G + g) * L + t;
}

// VEC consecutive elements, moved with a single load or store.
template <typename T, int VEC>
struct alignas(sizeof(T) * VEC) AlignedVector {
  T val[VEC];
};

// Load and store whole vectors; stores use a 16-byte global PTX instruction.
template <typename T, int VEC>
__device__ __forceinline__ AlignedVector<T, VEC> LoadAligned(
    const AlignedVector<T, VEC>* src) {
  if constexpr (sizeof(T) * VEC == 16) {
    const uint4 raw = *reinterpret_cast<const uint4*>(src);
    AlignedVector<T, VEC> v;
    memcpy(&v, &raw, sizeof(raw));
    return v;
  } else {
    return *src;
  }
}

template <typename T, int VEC>
__device__ __forceinline__ void StoreAligned(AlignedVector<T, VEC>* dst,
                                             const AlignedVector<T, VEC>& v) {
  if constexpr (sizeof(T) * VEC == 16) {
    uint32_t words[4];
    memcpy(words, &v, sizeof(words));
    asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(dst),
                 "r"(words[0]), "r"(words[1]), "r"(words[2]), "r"(words[3])
                 : "memory");
  } else {
    *dst = v;
  }
}

// Convert packed bfloat16 pairs to T_ACC by extracting each 16-bit half.
template <typename T, int VEC, typename T_ACC>
__device__ __forceinline__ void ConvertVector(const AlignedVector<T, VEC>& v,
                                              T_ACC (&dst)[VEC]) {
  if constexpr (std::is_same_v<T, c10::BFloat16> &&
                std::is_same_v<T_ACC, float> && VEC % 2 == 0) {
    uint32_t words[VEC / 2];
    memcpy(words, &v, sizeof(words));
#pragma unroll
    for (int k = 0; k < VEC / 2; ++k) {
      dst[2 * k] = __uint_as_float(words[k] << 16);
      dst[2 * k + 1] = __uint_as_float(words[k] & 0xFFFF0000U);
    }
  } else {
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      dst[k] = static_cast<T_ACC>(v.val[k]);
    }
  }
}

template <int VEC, typename T, typename T_ACC>
__device__ __forceinline__ void LoadVector(const T* __restrict__ src,
                                           T_ACC (&dst)[VEC]) {
  ConvertVector(
      LoadAligned(reinterpret_cast<const AlignedVector<T, VEC>*>(src)), dst);
}

template <int VEC, typename T, typename T_ACC>
__device__ __forceinline__ void StoreVector(T* __restrict__ dst,
                                            const T_ACC (&src)[VEC]) {
  AlignedVector<T, VEC> v;
#pragma unroll
  for (int k = 0; k < VEC; ++k) {
    v.val[k] = static_cast<T>(src[k]);
  }
  StoreAligned(reinterpret_cast<AlignedVector<T, VEC>*>(dst), v);
}

// Team kernels give each (token, group) to a team: WIDTH lanes of a warp, or the whole
// block for WIDTH == 0. Team i takes group i % G over the i / G-th token range.

constexpr int TeamBlockSize(int width) {
  return width == 0 ? cuda_utils::kCUDAMaxNumThreadsPerBlock
                    : kLookbackTeamBlockSize;
}

template <int WIDTH>
__device__ __forceinline__ unsigned int TeamMask() {
  if constexpr (WIDTH == 0 || WIDTH == cuda_utils::kWarpSize) {
    return 0xFFFFFFFFU;
  } else {
    const int base = (threadIdx.x % cuda_utils::kWarpSize) & ~(WIDTH - 1);
    return ((1U << WIDTH) - 1U) << base;
  }
}

// Tracks a team's token and group offsets in x [B * L, G * Dg] and outputs [B, G, L].
template <typename T>
struct TeamCursor {
  const T* row;
  int64_t idx;
  int t;
  const int L;
  const int H;
  // From the index of (b, g, L) to that of (b + 1, g, 0).
  const int64_t batch_skip;

  __device__ __forceinline__ TeamCursor(const T* x, int64_t tok, int64_t L,
                                        int64_t G, int64_t Dg, int64_t g)
      : row(x + tok * G * Dg + g * Dg),
        idx(GroupIndex(tok / L, g, tok % L, G, L)),
        t(static_cast<int>(tok % L)),
        L(static_cast<int>(L)),
        H(static_cast<int>(G * Dg)),
        batch_skip((G - 1) * L) {}

  __device__ __forceinline__ void Next() {
    row += H;
    ++idx;
    if (++t == L) {
      t = 0;
      idx += batch_skip;
    }
  }
};

// Sum each of `vals` over a team and broadcast the totals to its threads.
// shm holds N * kWarpSize values for a block team.
template <int WIDTH, int N, typename T_ACC>
__device__ __forceinline__ void TeamAllReduceSum(T_ACC (&vals)[N],
                                                 unsigned int mask,
                                                 T_ACC* shm) {
  if constexpr (WIDTH == 0) {
    const int lane = threadIdx.x % cuda_utils::kWarpSize;
    const int wid = threadIdx.x / cuda_utils::kWarpSize;
    const int num_warps = blockDim.x / cuda_utils::kWarpSize;
#pragma unroll
    for (int n = 0; n < N; ++n) {
#pragma unroll
      for (int offset = cuda_utils::kWarpSize / 2; offset > 0; offset >>= 1) {
        vals[n] += WARP_SHFL_XOR(vals[n], offset);
      }
    }
    __syncthreads();
    if (lane == 0) {
#pragma unroll
      for (int n = 0; n < N; ++n) {
        shm[n * cuda_utils::kWarpSize + wid] = vals[n];
      }
    }
    __syncthreads();
#pragma unroll
    for (int n = 0; n < N; ++n) {
      T_ACC sum = T_ACC(0);
      for (int w = 0; w < num_warps; ++w) {
        sum += shm[n * cuda_utils::kWarpSize + w];
      }
      vals[n] = sum;
    }
  } else {
#pragma unroll
    for (int offset = WIDTH / 2; offset > 0; offset >>= 1) {
#pragma unroll
      for (int n = 0; n < N; ++n) {
        vals[n] += WARP_SHFL_XOR(vals[n], offset, WIDTH, mask);
      }
    }
  }
}

template <typename T, typename T_ACC, int BLOCK_THREADS>
__global__ void RowwiseMomentsLookbackKernel(
    int64_t G, int64_t Dg, int64_t L, const T* __restrict__ x,
    const bool* __restrict__ padding_mask, T_ACC* __restrict__ group_mean,
    T_ACC* __restrict__ group_var) {
  const int64_t row = blockIdx.x;
  const int64_t b = row / L;
  const int64_t t = row - b * L;
  const int64_t g = blockIdx.y;
  const int64_t tid = threadIdx.x;
  const int64_t h_base = g * Dg;

  __shared__ utils::WelfordData<float> shm[cuda_utils::kWarpSize];

  const bool pad = padding_mask != nullptr && padding_mask[b * L + t];
  utils::WelfordData<float> moments;
  if (!pad) {
    for (int64_t d = tid; d < Dg; d += BLOCK_THREADS) {
      moments += static_cast<float>(x[XIndex(b, t, h_base + d, L, G * Dg)]);
    }
  }

  moments = reduce::BlockReduce(moments, shm);
  if (tid == 0) {
    const int64_t idx = GroupIndex(b, g, t, G, L);
    group_mean[idx] = static_cast<T_ACC>(pad ? 0.0f : moments.m1);
    group_var[idx] = static_cast<T_ACC>(pad ? 0.0f : moments.m2);
  }
}

// Two-pass per-token group moments from registers: each thread holds
// ITEMS loads of VEC elements for each of TOKENS tokens.
template <typename T, typename T_ACC, int VEC, int ITEMS, int WIDTH>
__global__ void __launch_bounds__(TeamBlockSize(WIDTH))
    RowwiseMomentsTeamLookbackKernel(
        int64_t N, int64_t L, int64_t G, int64_t Dg, int64_t num_ranges,
        int64_t range_size, const T* __restrict__ x,
        const bool* __restrict__ padding_mask, T_ACC* __restrict__ group_mean,
        T_ACC* __restrict__ group_var) {
  constexpr int TOKENS = WIDTH == 0 ? 1 : 4 / ITEMS;
  __shared__ T_ACC shm[WIDTH == 0 ? TOKENS * cuda_utils::kWarpSize : 1];

  const int width = WIDTH > 0 ? WIDTH : blockDim.x;
  const int64_t team =
      (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / width;
  const int lane = threadIdx.x % width;
  const int64_t g = team % G;
  const int64_t range = team / G;
  if (range >= num_ranges) {
    return;
  }
  const unsigned int mask = TeamMask<WIDTH>();
  const T_ACC group_size = static_cast<T_ACC>(Dg);
  const int64_t num_vecs = Dg / VEC;
  const int64_t begin = range * range_size;
  const int64_t end = begin + range_size < N ? begin + range_size : N;
  TeamCursor<T> cursor(x, begin, L, G, Dg, g);

  for (int64_t tok0 = begin; tok0 < end; tok0 += TOKENS) {
    bool active[TOKENS];
    int64_t idx[TOKENS];
    AlignedVector<T, VEC> vals[TOKENS][ITEMS];
#pragma unroll
    for (int u = 0; u < TOKENS; ++u) {
      const int64_t tok = tok0 + u;
      active[u] = tok < end &&
                  !(padding_mask != nullptr && padding_mask[tok]);
      idx[u] = cursor.idx;
#pragma unroll
      for (int i = 0; i < ITEMS; ++i) {
        const int64_t v = lane + int64_t(i) * width;
        if (active[u] && v < num_vecs) {
          vals[u][i] = *reinterpret_cast<const AlignedVector<T, VEC>*>(
              cursor.row + v * VEC);
        }
      }
      cursor.Next();
    }

    T_ACC sum[TOKENS];
#pragma unroll
    for (int u = 0; u < TOKENS; ++u) {
      sum[u] = T_ACC(0);
#pragma unroll
      for (int i = 0; i < ITEMS; ++i) {
        if (active[u] && lane + int64_t(i) * width < num_vecs) {
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            sum[u] += static_cast<T_ACC>(vals[u][i].val[k]);
          }
        }
      }
    }
    TeamAllReduceSum<WIDTH>(sum, mask, shm);

    T_ACC mean[TOKENS];
    T_ACC m2[TOKENS];
#pragma unroll
    for (int u = 0; u < TOKENS; ++u) {
      mean[u] = sum[u] / group_size;
      m2[u] = T_ACC(0);
#pragma unroll
      for (int i = 0; i < ITEMS; ++i) {
        if (active[u] && lane + int64_t(i) * width < num_vecs) {
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            const T_ACC d = static_cast<T_ACC>(vals[u][i].val[k]) - mean[u];
            m2[u] += d * d;
          }
        }
      }
    }
    TeamAllReduceSum<WIDTH>(m2, mask, shm);

    if (lane == 0) {
#pragma unroll
      for (int u = 0; u < TOKENS; ++u) {
        if (tok0 + u < end) {
          group_mean[idx[u]] = active[u] ? mean[u] : T_ACC(0);
          group_var[idx[u]] = active[u] ? m2[u] / group_size : T_ACC(0);
        }
      }
    }
  }
}

// Local mean/variance gradients: -rstd * db and -rstd^3 * ds / 2,
// divided by their bias corrections, computed with expm1.
__device__ __forceinline__ void LocalGradients(float rstd, float ds, float db,
                                               int64_t count, float log_beta1,
                                               float log_beta2, float* local_u,
                                               float* local_v) {
  const float c = static_cast<float>(count);
  const float correction1 = -expm1f(log_beta1 * c);
  const float correction2 = -expm1f(log_beta2 * c);
  *local_u = -rstd * db / correction1;
  *local_v = -0.5f * (rstd * rstd * rstd) * ds / correction2;
}

// Recover x - cummean from y, with |scale| clamped to at least eps.
template <typename T_ACC>
__device__ __forceinline__ T_ACC CenteredFromOutput(T_ACC y, T_ACC beta,
                                                    T_ACC scale, float eps) {
  scale = copysign(max(abs(scale), static_cast<T_ACC>(eps)), scale);
  if constexpr (std::is_same_v<T_ACC, float>) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(scale));
    return (y - beta) * r;
  } else {
    return (y - beta) / scale;
  }
}

// Decoupled look-back scan over [B, G, L], with tiles spanning batch/group boundaries.
// The first timestep of each (batch, group) resets the scan to its updated state.
// IndexT is unsigned int when the indices fit in int32.

// Every kCheckpointTiles-th tile of a chain publishes its inclusive prefix.
constexpr int kCheckpointTiles = cuda_utils::kWarpSize;

__device__ __forceinline__ void StoreRelaxed(uint4* ptr, const uint4& v) {
  asm volatile("st.relaxed.gpu.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(ptr),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

__device__ __forceinline__ uint4 LoadRelaxed(const uint4* ptr) {
  uint4 v;
  asm volatile("ld.relaxed.gpu.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr)
               : "memory");
  return v;
}

// Tile aggregates and checkpoint prefixes of a single-pass scan, zero-initialized. Each
// float shares an 8-byte word with its flag, read and written whole, so no fences are needed.
template <typename Value>
struct ScanTiles {
  static constexpr int kFloats = sizeof(Value) / sizeof(float);
  // 16-byte words of two (float, flag) pairs.
  static constexpr int kWords = (kFloats + 1) / 2;

  uint4* aggregate;
  uint4* inclusive;
  unsigned int* next_tile;

  __device__ __forceinline__ void Publish(uint4* values, int64_t tile,
                                          const Value& value) const {
    const float* f = reinterpret_cast<const float*>(&value);
#pragma unroll
    for (int w = 0; w < kWords; ++w) {
      const uint4 word = {__float_as_uint(f[2 * w]), 1U,
                          2 * w + 1 < kFloats ? __float_as_uint(f[2 * w + 1])
                                              : 0U,
                          1U};
      StoreRelaxed(values + tile * kWords + w, word);
    }
  }

  // Publish by a whole warp, lane w < kWords storing word w.
  __device__ __forceinline__ void WarpPublish(uint4* values, int64_t tile,
                                              const Value& value) const {
    const float* f = reinterpret_cast<const float*>(&value);
    const int lane = threadIdx.x % cuda_utils::kWarpSize;
    uint4 word = {0U, 1U, 0U, 1U};
#pragma unroll
    for (int w = 0; w < kWords; ++w) {
      if (lane == w) {
        word.x = __float_as_uint(f[2 * w]);
        word.z = 2 * w + 1 < kFloats ? __float_as_uint(f[2 * w + 1]) : 0U;
      }
    }
    if (lane < kWords) {
      StoreRelaxed(values + tile * kWords + lane, word);
    }
  }

  // Whether the value of a tile is there, and then the value.
  __device__ __forceinline__ bool Load(const uint4* values, int64_t tile,
                                       Value* value) const {
    float* f = reinterpret_cast<float*>(value);
    bool valid = true;
#pragma unroll
    for (int w = 0; w < kWords; ++w) {
      const uint4 word = LoadRelaxed(values + tile * kWords + w);
      valid = valid && word.y != 0U && word.w != 0U;
      f[2 * w] = __uint_as_float(word.x);
      if (2 * w + 1 < kFloats) {
        f[2 * w + 1] = __uint_as_float(word.z);
      }
    }
    return valid;
  }
};

// Whether the affine map discards its input; see ComposeAffine4.
__device__ __forceinline__ bool IsConstantMap(const float4& map) {
  return map.y == 0.0f && map.w == 0.0f;
}

// Publishes the aggregate of a tile, by lane 0 of a warp, or for the first
// tile of its chain, chain_start, its inclusive prefix from first_prefix.
template <class Op, typename Value>
__device__ __forceinline__ void PublishTile(const ScanTiles<Value>& tiles,
                                            int64_t tile, int64_t chain_start,
                                            const Value& first_prefix,
                                            const Value& aggregate) {
  if (threadIdx.x % cuda_utils::kWarpSize == 0) {
    if (tile == chain_start) {
      tiles.Publish(tiles.inclusive, tile, Op()(first_prefix, aggregate));
    } else {
      tiles.Publish(tiles.aggregate, tile, aggregate);
    }
  }
}

// Loads for the look-back of tile, rel > 0 tiles into its chain, by a warp: lane p gets the
// aggregate of the tile p before; lane tail, the farthest, a checkpoint prefix or a
// constant aggregate.
template <typename Value>
__device__ __forceinline__ void WarpLookBackLanes(const ScanTiles<Value>& tiles,
                                                  int64_t tile, int64_t rel,
                                                  const Value& identity,
                                                  Value* value, int* tail) {
  const int lane = threadIdx.x % cuda_utils::kWarpSize;
  // The nearest checkpoint is lane c.
  const int c = static_cast<int>((rel - 1) % kCheckpointTiles);
  const int64_t pred = tile - 1 - lane;
  for (unsigned int delay = 32;; delay = min(2 * delay, 1024U)) {
    Value agg = identity;
    Value incl = identity;
    const bool agg_valid = lane <= c && tiles.Load(tiles.aggregate, pred, &agg);
    const bool constant = agg_valid && IsConstantMap(agg);
    const bool incl_valid =
        lane == c && !constant && tiles.Load(tiles.inclusive, pred, &incl);
    const unsigned int valid_lanes = __ballot_sync(0xFFFFFFFFU, agg_valid);
    const unsigned int ends = __ballot_sync(0xFFFFFFFFU, constant || incl_valid);
    if (ends != 0) {
      *tail = __ffs(ends) - 1;
      const unsigned int below = (1U << *tail) - 1U;
      if ((valid_lanes & below) == below) {
        *value = (lane == *tail && incl_valid) ? incl : agg;
        return;
      }
    }
    __nanosleep(delay);
  }
}

// Composes the values of lanes 0 to tail, farthest first, into lane 0 by a tree over lanes.
template <class Op, typename Value>
__device__ __forceinline__ Value WarpComposeLanes(const Value& value,
                                                  int tail) {
  const Op op;
  const int lane = threadIdx.x % cuda_utils::kWarpSize;
  Value exclusive = value;
  for (int offset = 1; offset <= tail; offset *= 2) {
    const Value farther = cub::ShuffleDown<cuda_utils::kWarpSize>(
        exclusive, offset, cuda_utils::kWarpSize - 1, 0xFFFFFFFFU);
    if (lane + offset <= tail) {
      exclusive = op(farther, exclusive);
    }
  }
  return exclusive;
}

// Exclusive prefix of a tile, computed by a warp and returned in lane 0.
// The first tile takes first_prefix. Other tiles compose preceding aggregates
// back to the nearest checkpoint or constant map, waiting for those tiles.
template <class Op, typename Value>
__device__ __forceinline__ Value WarpLookBackPrefix(
    const ScanTiles<Value>& tiles, int64_t tile, int64_t chain_start,
    const Value& first_prefix, const Value& identity) {
  const int64_t rel = tile - chain_start;
  if (rel == 0) {
    return first_prefix;
  }
  Value value = identity;
  int tail = 0;
  WarpLookBackLanes(tiles, tile, rel, identity, &value, &tail);
  return WarpComposeLanes<Op>(value, tail);
}

// Exclusive prefixes of tile and of the next tile of its chain from one look-back, given
// tile's aggregate, by a warp; valid in lane 0.
template <class Op, typename Value>
__device__ __forceinline__ void WarpLookBackPrefixPair(
    const ScanTiles<Value>& tiles, int64_t tile, int64_t chain_start,
    const Value& first_prefix, const Value& aggregate, const Value& identity,
    Value* exclusive, Value* next_exclusive) {
  const Op op;
  const int64_t rel = tile - chain_start;
  if (rel == 0) {
    *exclusive = first_prefix;
    *next_exclusive = op(first_prefix, aggregate);
    return;
  }
  Value value = identity;
  int tail = 0;
  WarpLookBackLanes(tiles, tile, rel, identity, &value, &tail);
  *exclusive = WarpComposeLanes<Op>(value, tail);
  if (rel % kCheckpointTiles == 0 || IsConstantMap(aggregate)) {
    *next_exclusive = op(*exclusive, aggregate);
  } else {
    Value shifted = cub::ShuffleUp<cuda_utils::kWarpSize>(value, 1, 0,
                                                          0xFFFFFFFFU);
    if (threadIdx.x % cuda_utils::kWarpSize == 0) {
      shifted = aggregate;
    }
    *next_exclusive = WarpComposeLanes<Op>(shifted, tail + 1);
  }
}

// Publishes the inclusive prefix of a checkpoint other than the first tile of
// its chain from its exclusive prefix, by lane 0 of a warp.
template <class Op, typename Value>
__device__ __forceinline__ void PublishCheckpoint(
    const ScanTiles<Value>& tiles, int64_t tile, int64_t chain_start,
    const Value& exclusive, const Value& aggregate) {
  const int64_t rel = tile - chain_start;
  if (threadIdx.x % cuda_utils::kWarpSize == 0 && rel != 0 &&
      rel % kCheckpointTiles == 0) {
    tiles.Publish(tiles.inclusive, tile, Op()(exclusive, aggregate));
  }
}

// The exclusive prefix of a tile published by PublishTile, see
// WarpLookBackPrefix, and its inclusive prefix if a checkpoint.
template <class Op, typename Value>
__device__ __forceinline__ Value WarpLookBackPublished(
    const ScanTiles<Value>& tiles, int64_t tile, int64_t chain_start,
    const Value& first_prefix, const Value& aggregate, const Value& identity) {
  const Value exclusive = WarpLookBackPrefix<Op>(tiles, tile, chain_start,
                                                 first_prefix, identity);
  PublishCheckpoint<Op>(tiles, tile, chain_start, exclusive, aggregate);
  return exclusive;
}

// Publishes a tile and looks back from it, see WarpLookBackPublished.
template <class Op, typename Value>
__device__ __forceinline__ Value WarpLookBack(
    const ScanTiles<Value>& tiles, int64_t tile, int64_t chain_start,
    const Value& first_prefix, const Value& aggregate, const Value& identity) {
  PublishTile<Op>(tiles, tile, chain_start, first_prefix, aggregate);
  return WarpLookBackPublished<Op>(tiles, tile, chain_start, first_prefix,
                                   aggregate, identity);
}

// The prefix of a tile of a single chain of tiles for cub::BlockScan, which
// calls it with its first warp.
template <typename Value, class Op>
struct LookBackPrefix {
  ScanTiles<Value> tiles;
  unsigned int tile;
  Value identity;

  __device__ __forceinline__ Value operator()(const Value& block_aggregate) {
    return WarpLookBack<Op>(tiles, tile, 0, identity, block_aggregate,
                            identity);
  }
};

// One tile of BLOCK_THREADS * ITEMS consecutive scan elements per block. Problem::Load gives
// a thread's elements and Problem::Store writes their inclusive scans, with a Context between.
template <class Problem, int BLOCK_THREADS, int ITEMS>
__global__ void __launch_bounds__(BLOCK_THREADS)
    DecoupledScanKernel(Problem problem,
                        ScanTiles<typename Problem::Value> tiles) {
  using Value = typename Problem::Value;
  using Op = typename Problem::Op;
  using IndexT = typename Problem::Index;
  using BlockScan =
      cub::BlockScan<Value, BLOCK_THREADS, cub::BLOCK_SCAN_WARP_SCANS>;
  using Prefix = LookBackPrefix<Value, Op>;

  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ unsigned int tile_shared;

  // Tiles are numbered in block start order, so every tile waited on has started.
  if (threadIdx.x == 0) {
    tile_shared = atomicAdd(tiles.next_tile, 1U);
  }
  __syncthreads();
  const unsigned int tile = tile_shared;
  const IndexT first =
      static_cast<IndexT>(tile) * static_cast<IndexT>(BLOCK_THREADS * ITEMS) +
      static_cast<IndexT>(threadIdx.x) * ITEMS;

  typename Problem::template Context<ITEMS> context;
  Value items[ITEMS];
  problem.Load(first, context, items);

  const Op op;
  Value running = items[0];
#pragma unroll
  for (int k = 1; k < ITEMS; ++k) {
    running = op(running, items[k]);
  }
  Prefix prefix{tiles, tile, Problem::Identity()};
  BlockScan(scan_storage).ExclusiveScan(running, running, op, prefix);
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    running = op(running, items[k]);
    items[k] = running;
  }
  problem.Store(first, context, items);
}

// ptr[first + k] for the k < ITEMS with first + k < n, in one aligned access
// when all are; first is a multiple of ITEMS.
template <int ITEMS, typename T, typename IndexT>
__device__ __forceinline__ void LoadItems(const T* ptr, IndexT first, IndexT n,
                                          T (&vals)[ITEMS]) {
  if (first + ITEMS <= n) {
    const AlignedVector<T, ITEMS> vec =
        *reinterpret_cast<const AlignedVector<T, ITEMS>*>(ptr + first);
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      vals[k] = vec.val[k];
    }
  } else {
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      vals[k] = first + k < n ? ptr[first + k] : T(0);
    }
  }
}

template <int ITEMS, typename T, typename IndexT>
__device__ __forceinline__ void StoreItems(T* ptr, IndexT first, IndexT n,
                                           const T (&vals)[ITEMS]) {
  if (first + ITEMS <= n) {
    AlignedVector<T, ITEMS> vec;
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      vec.val[k] = vals[k];
    }
    *reinterpret_cast<AlignedVector<T, ITEMS>*>(ptr + first) = vec;
  } else {
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      if (first + k < n) {
        ptr[first + k] = vals[k];
      }
    }
  }
}

// ptr[last - k] for the k < ITEMS with k <= last, in one aligned access when
// all are and last + 1 is a multiple of ITEMS.
template <int ITEMS, typename T, typename IndexT>
__device__ __forceinline__ void LoadItemsReversed(const T* ptr, IndexT last,
                                                  T (&vals)[ITEMS]) {
  if (last + 1 >= static_cast<IndexT>(ITEMS) && (last + 1) % ITEMS == 0) {
    const AlignedVector<T, ITEMS> vec =
        *reinterpret_cast<const AlignedVector<T, ITEMS>*>(ptr + last + 1 -
                                                          ITEMS);
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      vals[k] = vec.val[ITEMS - 1 - k];
    }
  } else {
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      vals[k] = static_cast<IndexT>(k) <= last ? ptr[last - k] : T(0);
    }
  }
}

template <int ITEMS, typename T, typename IndexT>
__device__ __forceinline__ void StoreItemsReversed(T* ptr, IndexT last,
                                                   const T (&vals)[ITEMS]) {
  if (last + 1 >= static_cast<IndexT>(ITEMS) && (last + 1) % ITEMS == 0) {
    AlignedVector<T, ITEMS> vec;
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      vec.val[ITEMS - 1 - k] = vals[k];
    }
    *reinterpret_cast<AlignedVector<T, ITEMS>*>(ptr + last + 1 - ITEMS) = vec;
  } else {
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      if (static_cast<IndexT>(k) <= last) {
        ptr[last - k] = vals[k];
      }
    }
  }
}

// Affine updates x -> mul * x + add for running mean/variance and bias corrections.
// Bias corrections are scanned as d = beta * d + (1 - beta).
struct DecayState {
  float mul1;
  float mean;
  float d1;
  float mul2;
  float var;
  float d2;
};

// Compose prefix, then current; a constant current map discards the prefix.
struct DecayStateProd {
  __device__ __forceinline__ DecayState
  operator()(const DecayState& prefix, const DecayState& current) const {
    if (current.mul1 == 0.0f && current.mul2 == 0.0f) {
      return current;
    }
    return {current.mul1 * prefix.mul1,
            fmaf(current.mul1, prefix.mean, current.mean),
            fmaf(current.mul1, prefix.d1, current.d1),
            current.mul2 * prefix.mul2,
            fmaf(current.mul2, prefix.var, current.var),
            fmaf(current.mul2, prefix.d2, current.d2)};
  }
};

__device__ __forceinline__ bool IsConstantMap(const DecayState& map) {
  return map.mul1 == 0.0f && map.mul2 == 0.0f;
}

// The forward scan: the element of a timestep is its update of the state, and
// its inclusive scan gives the normalizing statistics.
template <typename T_ACC, typename IndexT>
struct FwdScanProblem {
  using Index = IndexT;
  using Value = DecayState;
  using Op = DecayStateProd;
  template <int ITEMS>
  struct Context {};

  IndexT num_items;
  at::cuda::detail::IntDivider<IndexT> L_div;
  at::cuda::detail::IntDivider<IndexT> G_div;
  const bool* bos_mask;
  const bool* padding_mask;
  const int64_t* prev_count;
  const T_ACC* prev_mean;
  const T_ACC* prev_var;
  const T_ACC* group_mean;
  const T_ACC* group_var;
  float beta1;
  float beta2;
  float eps;
  T_ACC* cummean;
  T_ACC* cumrstd;
  T_ACC* mean_out;
  T_ACC* var_out;

  static __device__ __forceinline__ DecayState Identity() {
    return {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
  }

  template <int ITEMS>
  __device__ __forceinline__ void Load(IndexT first, Context<ITEMS>&,
                                       DecayState (&items)[ITEMS]) const {
    T_ACC means[ITEMS];
    T_ACC vars[ITEMS];
    LoadItems(group_mean, first, num_items, means);
    LoadItems(group_var, first, num_items, vars);
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      items[k] = Identity();
    }
    if (first >= num_items) {
      return;
    }
    const IndexT L = L_div.divisor;
    const IndexT G = G_div.divisor;
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    // The (batch, group) seq = b * G + g and the timestep t of each item.
    const auto pos = L_div.divmod(first);
    IndexT seq = pos.div;
    IndexT t = pos.mod;
    IndexT b = G_div.div(seq);
    IndexT g = seq - b * G;
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      if (first + k < num_items) {
        const IndexT bt = b * L + t;
        const bool pad = padding_mask != nullptr && padding_mask[bt];
        const bool bos = bos_mask != nullptr && bos_mask[bt];
        if (!pad) {
          const float group_mean_t = static_cast<float>(means[k]);
          const float group_var_t = static_cast<float>(vars[k]);
          items[k] = {bos ? 0.0f : beta1, one_minus_beta1 * group_mean_t,
                      one_minus_beta1,    bos ? 0.0f : beta2,
                      one_minus_beta2 * group_var_t, one_minus_beta2};
        }
        if (t == 0) {
          const int64_t count = prev_count != nullptr ? prev_count[b] : 0;
          const DecayState initial = {
              0.0f,
              prev_mean != nullptr ? static_cast<float>(prev_mean[seq]) : 0.0f,
              static_cast<float>(-expm1(static_cast<double>(count) *
                                        log(static_cast<double>(beta1)))),
              0.0f,
              prev_var != nullptr ? static_cast<float>(prev_var[seq]) : 0.0f,
              static_cast<float>(-expm1(static_cast<double>(count) *
                                        log(static_cast<double>(beta2))))};
          items[k] = Op{}(initial, items[k]);
        }
      }
      if (++t == L) {
        t = 0;
        ++seq;
        if (++g == G) {
          g = 0;
          ++b;
        }
      }
    }
  }

  template <int ITEMS>
  __device__ __forceinline__ void Store(
      IndexT first, const Context<ITEMS>&,
      const DecayState (&states)[ITEMS]) const {
    if (first >= num_items) {
      return;
    }
    const IndexT L = L_div.divisor;
    const auto pos = L_div.divmod(first);
    IndexT seq = pos.div;
    IndexT t = pos.mod;
    T_ACC means[ITEMS];
    T_ACC rstds[ITEMS];
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      const DecayState& state = states[k];
      float curr_mean = 0.0f;
      float curr_rstd = rsqrtf(eps);
      if (state.d1 > 0.0f && state.d2 > 0.0f) {
        curr_mean = state.mean / state.d1;
        curr_rstd = rsqrtf(state.var / state.d2 + eps);
      }
      means[k] = static_cast<T_ACC>(curr_mean);
      rstds[k] = static_cast<T_ACC>(curr_rstd);
      if (first + k < num_items && t == L - 1) {
        if (mean_out != nullptr) {
          mean_out[seq] = static_cast<T_ACC>(state.mean);
        }
        if (var_out != nullptr) {
          var_out[seq] = static_cast<T_ACC>(state.var);
        }
      }
      if (++t == L) {
        t = 0;
        ++seq;
      }
    }
    StoreItems(cummean, first, num_items, means);
    StoreItems(cumrstd, first, num_items, rstds);
  }
};

// The initial state of a (batch, group) as the constant map to it.
template <typename T_ACC>
__device__ __forceinline__ DecayState InitialDecayState(
    int64_t b, int64_t seq, const int64_t* prev_count, const T_ACC* prev_mean,
    const T_ACC* prev_var, float beta1, float beta2) {
  const int64_t count = prev_count != nullptr ? prev_count[b] : 0;
  return {0.0f,
          prev_mean != nullptr ? static_cast<float>(prev_mean[seq]) : 0.0f,
          static_cast<float>(-expm1(static_cast<double>(count) *
                                    log(static_cast<double>(beta1)))),
          0.0f,
          prev_var != nullptr ? static_cast<float>(prev_var[seq]) : 0.0f,
          static_cast<float>(-expm1(static_cast<double>(count) *
                                    log(static_cast<double>(beta2))))};
}

// Single-pass forward for groups of W 16-byte loads. A tile is kWarpSize timesteps of one
// batch for GT groups: teams of W threads compute the moments, warp w < GT scans group w
// after WarpLookBack, and the block normalizes x; M warps share each group's loads.
// Resident blocks claim tiles in timestep-major order and copy the next tile while
// normalizing the current one.
template <typename T, typename T_ACC, int VEC, int W, int GT, int M>
__global__ void __launch_bounds__(GT * cuda_utils::kWarpSize * M)
    GroupTimestepDecayNormLookbackFusedFwdKernel(
        int64_t B, int64_t L, int64_t G, int64_t num_chunks, int64_t num_tiles,
        const T* __restrict__ x, const bool* __restrict__ bos_mask,
        const bool* __restrict__ padding_mask,
        const int64_t* __restrict__ prev_count,
        const T_ACC* __restrict__ prev_mean, const T_ACC* __restrict__ prev_var,
        const T* __restrict__ gamma, const T* __restrict__ beta, float beta1,
        float beta2, float eps, ScanTiles<DecayState> tiles,
        T* __restrict__ y, T_ACC* __restrict__ mean_out,
        T_ACC* __restrict__ var_out, T_ACC* __restrict__ cummean,
        T_ACC* __restrict__ cumrstd) {
  constexpr int kThreads = GT * cuda_utils::kWarpSize * M;
  constexpr int kTokens = cuda_utils::kWarpSize;
  constexpr int kDg = W * VEC;
  // Threads per timestep, timesteps loaded at a time, and loads per thread.
  constexpr int kRowThreads = GT * W;
  constexpr int kRows = kThreads / kRowThreads;
  constexpr int kSteps = kTokens / kRows;
  static_assert(kRows * kSteps == kTokens, "a tile is kWarpSize timesteps");
  using Vec = AlignedVector<T, VEC>;
  using WarpScan = cub::WarpScan<DecayState>;

  // [2][kSteps][kThreads] loads: the current tile and the next one.
  extern __shared__ unsigned char fused_shared[];
  Vec* const buffers = reinterpret_cast<Vec*>(fused_shared);
  __shared__ typename WarpScan::TempStorage scan_storage[GT];
  // The moments of the (timestep, group), then their normalizing statistics.
  __shared__ T_ACC stats1[GT][kTokens];
  __shared__ T_ACC stats2[GT][kTokens];
  // A tile: its number, batch, first group and chunk of timesteps.
  struct TileInfo {
    int64_t tile;
    int64_t b;
    int64_t g0;
    int64_t chunk;
  };
  __shared__ TileInfo next_info;

  const int64_t num_sets = B * (G / GT);
  const int64_t H = G * kDg;
  const int vec = threadIdx.x % kRowThreads;
  const int row = threadIdx.x / kRowThreads;
  const int lg = vec / W;
  const int lane = vec % W;
  const unsigned int mask = TeamMask<W>();
  const int warp = threadIdx.x / cuda_utils::kWarpSize;
  const int l = threadIdx.x % cuda_utils::kWarpSize;
  const float one_minus_beta1 = 1.0f - beta1;
  const float one_minus_beta2 = 1.0f - beta2;
  const DecayState identity = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};

  // Rows kRows apart in x and y, and the first one of a thread in a tile.
  const int64_t row_stride = kRows * H;
  const auto row_offset = [&](const TileInfo& info) {
    return (info.b * L + info.chunk * kTokens + row) * H + info.g0 * kDg +
           int64_t(vec) * VEC;
  };
  const auto take = [&]() {
    if (threadIdx.x == 0) {
      const int64_t tile = atomicAdd(tiles.next_tile, 1U);
      const int64_t chunk = tile / num_sets;
      const int64_t set = tile - chunk * num_sets;
      const int64_t b = set / (G / GT);
      next_info = {tile, b, (set - b * (G / GT)) * GT, chunk};
    }
    __syncthreads();
    return next_info;
  };
  const auto prefetch = [&](const TileInfo& info, int buffer) {
    if (info.tile < num_tiles) {
      const T* src = x + row_offset(info);
      const int64_t t = info.chunk * kTokens + row;
#pragma unroll
      for (int s = 0; s < kSteps; ++s) {
        if (t + s * kRows < L) {
          __pipeline_memcpy_async(
              buffers + (buffer * kSteps + s) * kThreads + threadIdx.x,
              src + s * row_stride, sizeof(Vec));
        }
      }
    }
    __pipeline_commit();
  };

  TileInfo info = take();
  prefetch(info, 0);
  for (int buffer = 0; info.tile < num_tiles; buffer ^= 1) {
    __pipeline_wait_prior(0);
    // The statistics of the tile before are read.
    __syncthreads();

    const int64_t b = info.b;
    const int64_t g0 = info.g0;
    const int64_t chunk = info.chunk;
    const int64_t t0 = chunk * kTokens;
    const Vec* const vals = buffers + buffer * kSteps * kThreads + threadIdx.x;
    bool active[kSteps];
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
      const int64_t t = t0 + row + s * kRows;
      active[s] = t < L && !(padding_mask != nullptr &&
                             padding_mask[b * L + t0 + row + s * kRows]);
    }

    // The moments.
    {
      T_ACC sum[kSteps];
#pragma unroll
      for (int s = 0; s < kSteps; ++s) {
        sum[s] = T_ACC(0);
        if (active[s]) {
          const Vec v = vals[s * kThreads];
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            sum[s] += static_cast<T_ACC>(v.val[k]);
          }
        }
      }
      TeamAllReduceSum<W>(sum, mask, static_cast<T_ACC*>(nullptr));
      T_ACC mean[kSteps];
      T_ACC m2[kSteps];
#pragma unroll
      for (int s = 0; s < kSteps; ++s) {
        mean[s] = sum[s] / static_cast<T_ACC>(kDg);
        m2[s] = T_ACC(0);
        if (active[s]) {
          const Vec v = vals[s * kThreads];
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            const T_ACC d = static_cast<T_ACC>(v.val[k]) - mean[s];
            m2[s] += d * d;
          }
        }
      }
      TeamAllReduceSum<W>(m2, mask, static_cast<T_ACC*>(nullptr));
      if (lane == 0) {
#pragma unroll
        for (int s = 0; s < kSteps; ++s) {
          const int u = row + s * kRows;
          stats1[lg][u] = active[s] ? mean[s] : T_ACC(0);
          stats2[lg][u] =
              active[s] ? m2[s] / static_cast<T_ACC>(kDg) : T_ACC(0);
        }
      }
    }
    __syncthreads();

    // The scan over the timesteps of group g0 + warp, lane l taking t0 + l,
    // by the first GT warps.
    if (warp < GT) {
      const int64_t t = t0 + l;
      const int64_t seq = b * G + g0 + warp;
      const bool valid = t < L;
      const bool pad =
          valid && padding_mask != nullptr && padding_mask[b * L + t];
      const bool bos = valid && bos_mask != nullptr && bos_mask[b * L + t];
      DecayState item = identity;
      if (valid && !pad) {
        item = {bos ? 0.0f : beta1,
                one_minus_beta1 * static_cast<float>(stats1[warp][l]),
                one_minus_beta1,
                bos ? 0.0f : beta2,
                one_minus_beta2 * static_cast<float>(stats2[warp][l]),
                one_minus_beta2};
      }
      DecayState inclusive;
      WarpScan(scan_storage[warp]).InclusiveScan(item, inclusive,
                                                 DecayStateProd{});
      const DecayState aggregate = cub::ShuffleIndex<cuda_utils::kWarpSize>(
          inclusive, cuda_utils::kWarpSize - 1, 0xFFFFFFFFU);
      DecayState prefix = identity;
      if (chunk == 0) {
        prefix = InitialDecayState(b, seq, prev_count, prev_mean, prev_var,
                                   beta1, beta2);
      }
      prefix = WarpLookBack<DecayStateProd>(tiles, seq * num_chunks + chunk,
                                            seq * num_chunks, prefix,
                                            aggregate, identity);
      prefix =
          cub::ShuffleIndex<cuda_utils::kWarpSize>(prefix, 0, 0xFFFFFFFFU);
      const DecayState state = DecayStateProd{}(prefix, inclusive);
      float curr_mean = 0.0f;
      float curr_rstd = rsqrtf(eps);
      if (state.d1 > 0.0f && state.d2 > 0.0f) {
        curr_mean = state.mean / state.d1;
        curr_rstd = rsqrtf(state.var / state.d2 + eps);
      }
      if (valid) {
        cummean[seq * L + t] = static_cast<T_ACC>(curr_mean);
        cumrstd[seq * L + t] = static_cast<T_ACC>(curr_rstd);
        if (t == L - 1) {
          if (mean_out != nullptr) {
            mean_out[seq] = static_cast<T_ACC>(state.mean);
          }
          if (var_out != nullptr) {
            var_out[seq] = static_cast<T_ACC>(state.var);
          }
        }
      }
      stats1[warp][l] = static_cast<T_ACC>(curr_mean);
      stats2[warp][l] = static_cast<T_ACC>(curr_rstd);
    }
    __syncthreads();

    const TileInfo following = take();
    prefetch(following, buffer ^ 1);

    // The normalization, as GroupTimestepDecayNormLookbackApplyFwdKernel
    // does it.
    const int64_t col = g0 * kDg + int64_t(vec) * VEC;
    T* const dst = y + row_offset(info);
    T_ACC w[VEC];
    T_ACC bias[VEC];
    LoadVector<VEC>(gamma + col, w);
    LoadVector<VEC>(beta + col, bias);
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
      const int u = row + s * kRows;
      const int64_t t = t0 + u;
      if (t < L) {
        T_ACC out[VEC];
        const Vec v = vals[s * kThreads];
        const T_ACC mean_t = stats1[lg][u];
        const T_ACC rstd_t = stats2[lg][u];
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
          out[k] = active[s] ? (static_cast<T_ACC>(v.val[k]) - mean_t) *
                                       rstd_t * w[k] +
                                   bias[k]
                             : T_ACC(0);
        }
        StoreVector<VEC>(dst + s * row_stride, out);
      }
    }
    info = following;
  }
}

// Single-pass backward over dy and x, tiled as the fused forward, walking each (batch,
// group) from its last tile: teams of W threads compute the group sums, warp w < GT scans
// group w backwards after WarpLookBack, and the block computes dx. Tile gamma/beta gradients
// go to [B * num_chunks, H] partials reduced by GammaBetaGradReduceLookbackKernel.
// kMemoryEfficient: x holds y; sums and mean use x - cummean.
template <typename T, typename T_ACC, int VEC, int W, int GT, int M,
          bool kMemoryEfficient>
__global__ void __launch_bounds__(GT * cuda_utils::kWarpSize * M)
    GroupTimestepDecayNormLookbackFusedBwdKernel(
        int64_t B, int64_t L, int64_t G, int64_t num_chunks, int64_t num_tiles,
        const T* __restrict__ dy, const T* __restrict__ x,
        const T* __restrict__ gamma, const T* __restrict__ beta,
        const T_ACC* __restrict__ cummean, const T_ACC* __restrict__ cumrstd,
        const int64_t* __restrict__ count, const bool* __restrict__ bos_mask,
        const bool* __restrict__ padding_mask,
        const T_ACC* __restrict__ mean_grad, const T_ACC* __restrict__ var_grad,
        float beta1, float beta2, float eps, ScanTiles<float4> tiles,
        T* __restrict__ dx,
        T_ACC* __restrict__ prev_mean_grad, T_ACC* __restrict__ prev_var_grad,
        T_ACC* __restrict__ gamma_grad_partial,
        T_ACC* __restrict__ beta_grad_partial) {
  constexpr int kThreads = GT * cuda_utils::kWarpSize * M;
  constexpr int kTokens = cuda_utils::kWarpSize;
  constexpr int kDg = W * VEC;
  constexpr int kRowThreads = GT * W;
  constexpr int kRows = kThreads / kRowThreads;
  constexpr int kSteps = kTokens / kRows;
  static_assert(kRows * kSteps == kTokens, "a tile is kWarpSize timesteps");
  constexpr int kCols = kRowThreads * VEC;
  // The gamma/beta gradients of a tile are summed over its rows 16 bytes at a
  // time, by the warps that do not scan, or by all if they all do.
  constexpr int kAccVec = 16 / sizeof(T_ACC);
  constexpr int kCombineThreads =
      M > 1 ? kThreads - GT * cuda_utils::kWarpSize : kThreads;
  static_assert(kCols % kAccVec == 0, "the columns fill 16 bytes");
  using Vec = AlignedVector<T, VEC>;
  using AccVec = AlignedVector<T_ACC, kAccVec>;
  using WarpScan = cub::WarpScan<float4>;

  // [2][2][kSteps][kThreads] loads: x then dy, of the current tile and the
  // next one.
  extern __shared__ unsigned char fused_shared[];
  Vec* const buffers = reinterpret_cast<Vec*>(fused_shared);
  __shared__ typename WarpScan::TempStorage scan_storage[GT];
  // The statistics of the forward and the counts, of both tiles.
  __shared__ T_ACC cummean_shared[2][GT][kTokens];
  __shared__ T_ACC cumrstd_shared[2][GT][kTokens];
  __shared__ int64_t count_shared[2][kTokens];
  // The gamma (and beta) of the columns of both tiles.
  __shared__ Vec gamma_tile[2][kRowThreads];
  __shared__ Vec beta_tile[kMemoryEfficient ? 2 : 1][kRowThreads];
  // The group sums of the (timestep, group): sum(x), sum(g), sum(g * (x - c)),
  // then the mean of x, du and dv.
  __shared__ T_ACC stats1[GT][kTokens];
  __shared__ T_ACC stats2[GT][kTokens];
  __shared__ T_ACC stats3[GT][kTokens];
  // Per-thread gamma/beta gradients of the tile, [kRows][kCols], kept in the previous
  // tile's copy buffer (unused between its sums and the next copy) when they fit.
  constexpr bool kGradsInBuffer =
      2 * kThreads * VEC * sizeof(T_ACC) <= 2 * kSteps * kThreads * sizeof(Vec);
  __shared__ __align__(16)
      T_ACC grads_shared[kGradsInBuffer ? 1 : 2 * kThreads * VEC];
  struct TileInfo {
    int64_t tile;
    int64_t b;
    int64_t g0;
    int64_t chunk;
  };
  __shared__ TileInfo next_info;

  const int64_t num_sets = B * (G / GT);
  const int64_t H = G * kDg;
  const int vec = threadIdx.x % kRowThreads;
  const int row = threadIdx.x / kRowThreads;
  const int lg = vec / W;
  const int lane = vec % W;
  const unsigned int mask = TeamMask<W>();
  const int warp = threadIdx.x / cuda_utils::kWarpSize;
  const int l = threadIdx.x % cuda_utils::kWarpSize;
  const float log_beta1 = logf(beta1);
  const float log_beta2 = logf(beta2);
  const T_ACC group_size = static_cast<T_ACC>(kDg);
  const T_ACC coef = T_ACC(1) / group_size;

  const int64_t row_stride = kRows * H;
  const auto row_offset = [&](const TileInfo& info) {
    return (info.b * L + info.chunk * kTokens + row) * H + info.g0 * kDg +
           int64_t(vec) * VEC;
  };
  // The tiles of each (batch, group) are taken from the last chunk.
  const auto take = [&]() {
    if (threadIdx.x == 0) {
      const int64_t tile = atomicAdd(tiles.next_tile, 1U);
      const int64_t k = tile / num_sets;
      const int64_t set = tile - k * num_sets;
      const int64_t b = set / (G / GT);
      next_info = {tile, b, (set - b * (G / GT)) * GT, num_chunks - 1 - k};
    }
    __syncthreads();
    return next_info;
  };
  const auto prefetch = [&](const TileInfo& info, int buffer) {
    if (info.tile < num_tiles) {
      const int64_t offset = row_offset(info);
      const int64_t t = info.chunk * kTokens + row;
#pragma unroll
      for (int s = 0; s < kSteps; ++s) {
        if (t + s * kRows < L) {
          __pipeline_memcpy_async(
              buffers + ((buffer * 2) * kSteps + s) * kThreads + threadIdx.x,
              x + offset + s * row_stride, sizeof(Vec));
          __pipeline_memcpy_async(
              buffers + ((buffer * 2 + 1) * kSteps + s) * kThreads +
                  threadIdx.x,
              dy + offset + s * row_stride, sizeof(Vec));
        }
      }
      if (row == 0) {
        __pipeline_memcpy_async(&gamma_tile[buffer][vec],
                                gamma + info.g0 * kDg + int64_t(vec) * VEC,
                                sizeof(Vec));
        if constexpr (kMemoryEfficient) {
          __pipeline_memcpy_async(&beta_tile[buffer][vec],
                                  beta + info.g0 * kDg + int64_t(vec) * VEC,
                                  sizeof(Vec));
        }
      }
      // One statistic of each array per thread, the counts with the first
      // warp.
      const int64_t ts = info.chunk * kTokens + l;
      if (warp < GT && ts < L) {
        const int64_t idx = (info.b * G + info.g0 + warp) * L + ts;
        if constexpr (!kMemoryEfficient) {
          __pipeline_memcpy_async(&cummean_shared[buffer][warp][l],
                                  cummean + idx, sizeof(T_ACC));
        }
        __pipeline_memcpy_async(&cumrstd_shared[buffer][warp][l],
                                cumrstd + idx, sizeof(T_ACC));
        if (warp == 0) {
          __pipeline_memcpy_async(&count_shared[buffer][l],
                                  count + info.b * L + ts, sizeof(int64_t));
        }
      }
    }
    __pipeline_commit();
  };

  TileInfo info = take();
  prefetch(info, 0);
  for (int buffer = 0; info.tile < num_tiles; buffer ^= 1) {
    __pipeline_wait_prior(0);
    // The statistics of the tile before are read, and the copies of this one
    // made by all the threads have landed.
    __syncthreads();

    const int64_t b = info.b;
    const int64_t g0 = info.g0;
    const int64_t chunk = info.chunk;
    const int64_t k = num_chunks - 1 - chunk;
    const int64_t t0 = chunk * kTokens;
    const Vec* const xs =
        buffers + (buffer * 2) * kSteps * kThreads + threadIdx.x;
    const Vec* const dys =
        buffers + (buffer * 2 + 1) * kSteps * kThreads + threadIdx.x;
    T_ACC* const gamma_shared =
        kGradsInBuffer ? reinterpret_cast<T_ACC*>(
                             buffers + ((buffer ^ 1) * 2) * kSteps * kThreads)
                       : grads_shared;
    T_ACC* const beta_shared = gamma_shared + kThreads * VEC;
    T_ACC w[VEC];
    T_ACC bias[VEC];
    {
      const Vec gamma_vec = gamma_tile[buffer][vec];
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        w[i] = static_cast<T_ACC>(gamma_vec.val[i]);
      }
      if constexpr (kMemoryEfficient) {
        const Vec beta_vec = beta_tile[buffer][vec];
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          bias[i] = static_cast<T_ACC>(beta_vec.val[i]);
        }
      }
    }
    bool active[kSteps];
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
      const int64_t t = t0 + row + s * kRows;
      active[s] = t < L && !(padding_mask != nullptr &&
                             padding_mask[b * L + t0 + row + s * kRows]);
    }

    // The group sums and the gamma/beta gradients.
    {
      T_ACC sums[kSteps * 3];
      T_ACC dgamma[VEC];
      T_ACC dbeta[VEC];
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        dgamma[i] = T_ACC(0);
        dbeta[i] = T_ACC(0);
      }
#pragma unroll
      for (int s = 0; s < kSteps; ++s) {
        sums[3 * s] = T_ACC(0);
        sums[3 * s + 1] = T_ACC(0);
        sums[3 * s + 2] = T_ACC(0);
        if (active[s]) {
          const int u = row + s * kRows;
          // Memory efficient: x_val is already x - cummean.
          const T_ACC c =
              kMemoryEfficient ? T_ACC(0) : cummean_shared[buffer][lg][u];
          const T_ACC r = cumrstd_shared[buffer][lg][u];
          const Vec xv = xs[s * kThreads];
          const Vec dyv = dys[s * kThreads];
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            const T_ACC x_val =
                kMemoryEfficient
                    ? CenteredFromOutput(static_cast<T_ACC>(xv.val[i]),
                                         bias[i], r * w[i], eps)
                    : static_cast<T_ACC>(xv.val[i]);
            const T_ACC dy_val = static_cast<T_ACC>(dyv.val[i]);
            const T_ACC x_c = x_val - c;
            const T_ACC grad_hat = dy_val * w[i];
            sums[3 * s] += x_val;
            sums[3 * s + 1] += grad_hat;
            sums[3 * s + 2] += grad_hat * x_c;
            dgamma[i] += dy_val * x_c * r;
            dbeta[i] += dy_val;
          }
        }
      }
      TeamAllReduceSum<W>(sums, mask, static_cast<T_ACC*>(nullptr));
      if (lane == 0) {
#pragma unroll
        for (int s = 0; s < kSteps; ++s) {
          const int u = row + s * kRows;
          stats1[lg][u] = sums[3 * s];
          stats2[lg][u] = sums[3 * s + 1];
          stats3[lg][u] = sums[3 * s + 2];
        }
      }
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        gamma_shared[threadIdx.x * VEC + i] = dgamma[i];
        beta_shared[threadIdx.x * VEC + i] = dbeta[i];
      }
    }
    __syncthreads();

    // The gamma/beta gradients of the tile, summed over its rows in order.
    if (M == 1 || warp >= GT) {
      const int first = M > 1 ? threadIdx.x - GT * cuda_utils::kWarpSize
                              : threadIdx.x;
      const int64_t h = (b * num_chunks + chunk) * H + g0 * kDg;
#pragma unroll
      for (int q = first; q < 2 * kCols / kAccVec; q += kCombineThreads) {
        const bool is_beta = q >= kCols / kAccVec;
        const int c = (is_beta ? q - kCols / kAccVec : q) * kAccVec;
        const T_ACC* const src = is_beta ? beta_shared : gamma_shared;
        AccVec sum = *reinterpret_cast<const AccVec*>(src + c);
#pragma unroll
        for (int r = 1; r < kRows; ++r) {
          const AccVec v = *reinterpret_cast<const AccVec*>(src + r * kCols + c);
#pragma unroll
          for (int i = 0; i < kAccVec; ++i) {
            sum.val[i] += v.val[i];
          }
        }
        *reinterpret_cast<AccVec*>(
            (is_beta ? beta_grad_partial : gamma_grad_partial) + h + c) = sum;
      }
    }

    // The scan over the timesteps of group g0 + warp from the last, lane l
    // taking t0 + kTokens - 1 - l, by the first GT warps.
    if (warp < GT) {
      const int u = kTokens - 1 - l;
      const int64_t t = t0 + u;
      const int64_t seq = b * G + g0 + warp;
      const bool valid = t < L;
      const bool pad =
          valid && padding_mask != nullptr && padding_mask[b * L + t];
      const bool bos = valid && bos_mask != nullptr && bos_mask[b * L + t];
      float u_t = 0.0f;
      float v_t = 0.0f;
      if (valid && !pad && count_shared[buffer][u] > 0) {
        LocalGradients(static_cast<float>(cumrstd_shared[buffer][warp][u]),
                       static_cast<float>(stats3[warp][u]),
                       static_cast<float>(stats2[warp][u]),
                       count_shared[buffer][u], log_beta1, log_beta2, &u_t,
                       &v_t);
      }
      const float local_u = static_cast<float>(static_cast<T_ACC>(u_t));
      const float local_v = static_cast<float>(static_cast<T_ACC>(v_t));
      float4 map = IdentityAffine4();
      if (valid && !pad) {
        map = bos ? MakeAffine4(0.0f, 0.0f, 0.0f, 0.0f)
                  : MakeAffine4(beta1 * local_u, beta1, beta2 * local_v,
                                beta2);
      }
      float4 exclusive;
      float4 aggregate;
      WarpScan(scan_storage[warp]).ExclusiveScan(
          map, exclusive, IdentityAffine4(), Affine4Prod{}, aggregate);
      float4 prefix = IdentityAffine4();
      if (k == 0) {
        prefix = MakeAffine4(static_cast<float>(mean_grad[seq]), 0.0f,
                             static_cast<float>(var_grad[seq]), 0.0f);
      }
      prefix = WarpLookBack<Affine4Prod>(tiles, seq * num_chunks + k,
                                         seq * num_chunks, prefix, aggregate,
                                         IdentityAffine4());
      prefix =
          cub::ShuffleIndex<cuda_utils::kWarpSize>(prefix, 0, 0xFFFFFFFFU);
      // The gradients flowing into timestep t.
      const float4 state = Affine4Prod{}(prefix, exclusive);
      const float u_in = state.x;
      const float v_in = state.z;
      T_ACC du = T_ACC(0);
      T_ACC dv = T_ACC(0);
      if (valid && !pad) {
        du = static_cast<T_ACC>((1.0f - beta1) * (u_in + local_u));
        dv = static_cast<T_ACC>((1.0f - beta2) * (v_in + local_v));
      }
      if (t == 0) {
        prev_mean_grad[seq] = static_cast<T_ACC>(ApplyAffineX(map, u_in));
        prev_var_grad[seq] = static_cast<T_ACC>(ApplyAffineY(map, v_in));
      }
      stats1[warp][u] = (valid && !pad) ? stats1[warp][u] / group_size
                                        : T_ACC(0);
      stats2[warp][u] = du;
      stats3[warp][u] = dv;
    }
    __syncthreads();

    const TileInfo following = take();
    prefetch(following, buffer ^ 1);

    // dx.
    T* const dst = dx + row_offset(info);
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
      const int u = row + s * kRows;
      const int64_t t = t0 + u;
      if (t < L) {
        T_ACC out[VEC];
        const Vec xv = xs[s * kThreads];
        const Vec dyv = dys[s * kThreads];
        const T_ACC mean = stats1[lg][u];
        const T_ACC rstd = cumrstd_shared[buffer][lg][u];
        // coef * (du + 2 * dv * (x - mean)), exactly so for a power of 2
        // group size.
        const T_ACC du_coef = MulRn(coef, stats2[lg][u]);
        const T_ACC dv_coef = T_ACC(2) * coef * stats3[lg][u];
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          const T_ACC scale = rstd * w[i];
          const T_ACC x_val =
              kMemoryEfficient
                  ? CenteredFromOutput(static_cast<T_ACC>(xv.val[i]), bias[i],
                                       scale, eps)
                  : static_cast<T_ACC>(xv.val[i]);
          out[i] = active[s] ? static_cast<T_ACC>(dyv.val[i]) * scale +
                                   (dv_coef * (x_val - mean) + du_coef)
                             : T_ACC(0);
        }
        StoreVector<VEC>(dst + s * row_stride, out);
      }
    }
    info = following;
  }
}

// TMA kernels (sm_90+): warps split by role and pass tiles through shared-memory
// mbarriers; a producer warp issues the bulk copies.

__device__ __forceinline__ uint32_t SharedAddress(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void MbarrierInit(uint64_t* bar, int count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(
                   SharedAddress(bar)),
               "r"(count)
               : "memory");
#endif
}

// Makes the initialized mbarriers visible to the bulk copies.
__device__ __forceinline__ void MbarrierInitFence() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
}

// An arrival, releasing the writes of the thread and of those it synced with.
__device__ __forceinline__ void MbarrierArrive(uint64_t* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(
                   SharedAddress(bar))
               : "memory");
#endif
}

// An arrival that also expects bytes from bulk copies in this phase.
__device__ __forceinline__ void MbarrierArriveExpectTx(uint64_t* bar,
                                                       uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                   SharedAddress(bar)),
               "r"(bytes)
               : "memory");
#endif
}

// Waits for the phase of the given parity to complete, suspended meanwhile
// (up to a hint of 10 ms at a time) rather than spinning.
__device__ __forceinline__ void MbarrierWait(uint64_t* bar, uint32_t parity) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "{\n"
      "  .reg .pred done;\n"
      "WAIT:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n"
      "  @!done bra WAIT;\n"
      "}\n" ::"r"(SharedAddress(bar)),
      "r"(parity), "r"(10000000U)
      : "memory");
#endif
}

// Copies bytes, a multiple of 16 at 16-byte aligned addresses, from global to
// shared memory, completing them on bar.
__device__ __forceinline__ void BulkCopy(void* dst, const void* src,
                                         uint32_t bytes, uint64_t* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::"r"(SharedAddress(dst)),
      "l"(src), "r"(bytes), "r"(SharedAddress(bar))
      : "memory");
#endif
}

// The producers of the TMA kernels claim this many tiles at a time.
constexpr unsigned int kTmaClaimTiles = 2;

// The layout of a stage of the TMA forward, in bytes: a tile of rows
// timesteps of row_bytes, the bos flags of the timesteps, then the row_bytes
// of gamma and of beta for its columns, the boxes 128-byte aligned.
constexpr int TmaRoundUp128(int bytes) { return (bytes + 127) / 128 * 128; }
constexpr int TmaFwdBosOffset(int rows, int row_bytes) {
  return TmaRoundUp128(rows * row_bytes);
}
constexpr int TmaFwdParamsOffset(int rows, int row_bytes) {
  return TmaFwdBosOffset(rows, row_bytes) + TmaRoundUp128(rows);
}
constexpr int TmaFwdStageBytes(int rows, int row_bytes) {
  return TmaRoundUp128(TmaFwdParamsOffset(rows, row_bytes) + 2 * row_bytes);
}

// Copies the box of a 3d tensor map at (c0, c1, c2), innermost first, to
// shared memory, completing it on bar; the box elements out of the tensor are
// zero.
__device__ __forceinline__ void TensorCopy3d(void* dst, const CUtensorMap* map,
                                             int c0, int c1, int c2,
                                             uint64_t* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_"
      "tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(SharedAddress(dst)),
      "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(SharedAddress(bar))
      : "memory");
#endif
}

// Copies the box of a 2d tensor map at (c0, c1), innermost first, as
// TensorCopy3d.
__device__ __forceinline__ void TensorCopy2d(void* dst, const CUtensorMap* map,
                                             int c0, int c1, uint64_t* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_"
      "tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(SharedAddress(dst)),
      "l"(map), "r"(c0), "r"(c1), "r"(SharedAddress(bar))
      : "memory");
#endif
}

// A tile of the TMA kernels: its number, batch, first group, chunk of
// timesteps and the number of those in the sequence.
struct TmaTileInfo {
  int64_t tile;
  int64_t b;
  int64_t g0;
  int64_t chunk;
  // The timesteps of the tile in the sequence.
  int rows;
};

// TMA forward without padding. A tile is KC chunks of
// kWarpSize timesteps for GT groups (one of KC, GT is 1, the other at most 2); each
// (group, chunk) is a scan part. Warp roles, each going through the block's tiles in order:
// - producer (last warp): claims tiles and copies x, the bos flags (bos_map if bos_tma),
//   gamma and beta into one of S stages;
// - NM moment warps: moments of each (timestep, group);
// - publisher warp p < GT * KC: scans part p and publishes its aggregate unless it starts
//   a chain;
// - look-back warp GT * (KC + j) + w: prefixes of group g0 + w for tiles k % NL == j, and
//   the inclusive prefix of the first chunk of each chain;
// - final warp: normalizing statistics from the scans and prefixes;
// - NN normalizing warps: normalize the tile.
// Stage mbarriers: full, moments, published, prefixed, scanned, empty.
template <typename T, typename T_ACC, int VEC, int W, int GT, int KC, int NL,
          int NM, int NN, int S>
__global__ void __launch_bounds__(((KC + NL) * GT + NM + NN + 2) *
                                      cuda_utils::kWarpSize,
                                  2)
    GroupTimestepDecayNormLookbackTmaFwdKernel(
        int64_t B, int64_t L, int64_t G, int64_t num_chunks, int64_t num_tiles,
        const __grid_constant__ CUtensorMap x_map,
        const __grid_constant__ CUtensorMap bos_map, bool bos_tma,
        const bool* __restrict__ bos_mask,
        const int64_t* __restrict__ prev_count,
        const T_ACC* __restrict__ prev_mean, const T_ACC* __restrict__ prev_var,
        const T* __restrict__ gamma, const T* __restrict__ beta, float beta1,
        float beta2, float eps, ScanTiles<DecayState> tiles,
        T* __restrict__ y, T_ACC* __restrict__ mean_out,
        T_ACC* __restrict__ var_out, T_ACC* __restrict__ cummean,
        T_ACC* __restrict__ cumrstd) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int kTokens = cuda_utils::kWarpSize;
  // The timesteps of a tile, and its parts.
  constexpr int kTileRows = KC * kTokens;
  constexpr int kParts = KC * GT;
  static_assert((KC == 1 || GT == 1) && kParts <= 2, "tiles of 1 or 2 parts");
  constexpr int kDg = W * VEC;
  // Threads per timestep of the moment and normalizing warps.
  constexpr int kRowThreads = GT * W;
  static_assert(NM * cuda_utils::kWarpSize % kRowThreads == 0 &&
                    NN * cuda_utils::kWarpSize % kRowThreads == 0 &&
                    NM * cuda_utils::kWarpSize <= kTileRows * kRowThreads &&
                    NN * cuda_utils::kWarpSize <= kTileRows * kRowThreads,
                "the warps load a tile evenly");
  static_assert(NL <= S, "the look-back warps are on tiles in the stages");
  constexpr int kFinalWarp = (KC + NL) * GT;
  constexpr int kMomentWarp = kFinalWarp + 1;
  constexpr int kNormWarp = kMomentWarp + NM;
  constexpr int kProducer = kNormWarp + NN;
  constexpr int kStateFloats = sizeof(DecayState) / sizeof(float);
  using Vec = AlignedVector<T, VEC>;
  constexpr uint32_t kRowBytes = kRowThreads * sizeof(Vec);
  constexpr int kStageVecs =
      TmaFwdStageBytes(kTileRows, kRowBytes) / sizeof(Vec);
  constexpr int kBosOffset = TmaFwdBosOffset(kTileRows, kRowBytes);
  constexpr int kParamsVecs =
      TmaFwdParamsOffset(kTileRows, kRowBytes) / sizeof(Vec);
  using WarpScan = cub::WarpScan<DecayState>;

  // The S stages of kStageVecs loads: [kTileRows][kRowThreads], a timestep
  // per row, the bos flags at kBosOffset bytes, then the kRowThreads loads of
  // gamma and of beta at kParamsVecs.
  extern __shared__ __align__(128) unsigned char tma_shared[];
  Vec* const stages = reinterpret_cast<Vec*>(tma_shared);
  __shared__ typename WarpScan::TempStorage scan_storage[kParts];
  // The moments of the (timestep, group), then their normalizing statistics.
  __shared__ T_ACC stats1[S][GT][kTileRows];
  __shared__ T_ACC stats2[S][GT][kTileRows];
  // The inclusive scans of the parts, a float at a time, and their exclusive
  // prefixes.
  __shared__ float inclusive_shared[S][kParts][kStateFloats][kTokens];
  __shared__ float prefix_shared[S][kParts][kStateFloats];
  __shared__ TmaTileInfo infos[S];
  __shared__ uint64_t full[S];
  __shared__ uint64_t moments[S];
  __shared__ uint64_t published[S];
  __shared__ uint64_t prefixed[S];
  __shared__ uint64_t scanned[S];
  __shared__ uint64_t empty[S];

  const int64_t num_sets = B * (G / GT);
  const int64_t H = G * kDg;
  const int warp = threadIdx.x / cuda_utils::kWarpSize;
  const int l = threadIdx.x % cuda_utils::kWarpSize;
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < S; ++s) {
      MbarrierInit(&full[s], 1);
      MbarrierInit(&moments[s], NM);
      MbarrierInit(&published[s], kParts);
      MbarrierInit(&prefixed[s], GT);
      MbarrierInit(&scanned[s], 1);
      MbarrierInit(&empty[s], NN);
    }
    MbarrierInitFence();
  }
  __syncthreads();

  const DecayState identity = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
  if (warp == kProducer) {
    // Claim kTmaClaimTiles consecutive tiles at a time. Tile numbers fit in 32 bits.
    const unsigned int sets = static_cast<unsigned int>(num_sets);
    const unsigned int sets_per_batch = static_cast<unsigned int>(G / GT);
    unsigned int claimed = 0U;
    if (l == 0) {
      claimed = atomicAdd(tiles.next_tile, kTmaClaimTiles);
    }
    unsigned int first = 0U;
    int taken = kTmaClaimTiles;
    for (int64_t k = 0;; ++k) {
      const int s = static_cast<int>(k % S);
      if (k >= S) {
        MbarrierWait(&empty[s], static_cast<uint32_t>(k / S - 1) & 1U);
      }
      if (taken == kTmaClaimTiles) {
        first = __shfl_sync(0xFFFFFFFFU, claimed, 0);
        taken = 0;
        if (l == 0 && first < num_tiles) {
          claimed = atomicAdd(tiles.next_tile, kTmaClaimTiles);
        }
      }
      const unsigned int tile = first + taken++;
      const unsigned int tile_chunk = tile / sets;
      const unsigned int set = tile - tile_chunk * sets;
      const unsigned int b = set / sets_per_batch;
      const unsigned int g0 = (set - b * sets_per_batch) * GT;
      if (tile >= num_tiles) {
        // The end, past the last tile of each set of look-back warps.
        for (int64_t e = k; e < k + NL; ++e) {
          const int se = static_cast<int>(e % S);
          if (e > k && e >= S) {
            MbarrierWait(&empty[se], static_cast<uint32_t>(e / S - 1) & 1U);
          }
          if (l == 0) {
            infos[se].tile = num_tiles;
            MbarrierArrive(&full[se]);
          }
        }
        break;
      }
      const int64_t chunk = int64_t(tile_chunk) * KC;
      const int64_t t0 = chunk * kTokens;
      const int rows =
          static_cast<int>(L - t0 < kTileRows ? L - t0 : kTileRows);
      if (l == 0) {
        infos[s] = TmaTileInfo{tile, b, g0, chunk, rows};
      }
      // The tile as one box of x, [B, L, H], its timesteps past L zero.
      if (l == 0) {
        Vec* const stage = stages + s * kStageVecs;
        MbarrierArriveExpectTx(&full[s], (kTileRows + 2) * kRowBytes +
                                             (bos_tma ? kTileRows : 0));
        TensorCopy3d(stage, &x_map, static_cast<int>(g0 * kDg),
                     static_cast<int>(t0), static_cast<int>(b), &full[s]);
        if (bos_tma) {
          TensorCopy2d(reinterpret_cast<unsigned char*>(stage) + kBosOffset,
                       &bos_map, static_cast<int>(t0), static_cast<int>(b),
                       &full[s]);
        }
        Vec* const params = stage + kParamsVecs;
        BulkCopy(params, gamma + g0 * kDg, kRowBytes, &full[s]);
        BulkCopy(params + kRowThreads, beta + g0 * kDg, kRowBytes, &full[s]);
      }
    }
  } else if (warp >= kNormWarp) {
    // The normalization, as GroupTimestepDecayNormLookbackApplyFwdKernel
    // does it.
    constexpr int kRows = NN * cuda_utils::kWarpSize / kRowThreads;
    constexpr int kSteps = kTileRows / kRows;
    const int vec = threadIdx.x % kRowThreads;
    const int row = (threadIdx.x - kNormWarp * cuda_utils::kWarpSize) /
                    kRowThreads;
    const int lg = vec / W;
    const int64_t row_stride = kRows * H;
    // kChecked for a tile past the end of the sequence.
    const auto normalize = [&]<bool kChecked>(const TmaTileInfo& info, int s,
                                              uint32_t parity) {
      const int64_t t0 = info.chunk * kTokens;
      const Vec* const stage = stages + s * kStageVecs;
      const Vec* const vals = stage + row * kRowThreads + vec;
      const T* const params =
          reinterpret_cast<const T*>(stage + kParamsVecs + vec);
      const int64_t col = info.g0 * kDg + int64_t(vec) * VEC;
      T* const dst = y + (info.b * L + t0 + row) * H + col;
      MbarrierWait(&scanned[s], parity);
      T_ACC w[VEC];
      T_ACC bias[VEC];
      LoadVector<VEC>(params, w);
      LoadVector<VEC>(params + kRowThreads * VEC, bias);
#pragma unroll
      for (int step = 0; step < kSteps; ++step) {
        const int u = row + step * kRows;
        if (!kChecked || u < info.rows) {
          T_ACC out[VEC];
          T_ACC x_t[VEC];
          ConvertVector(LoadAligned(vals + step * kRows * kRowThreads), x_t);
          const T_ACC mean_t = stats1[s][lg][u];
          const T_ACC rstd_t = stats2[s][lg][u];
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            out[i] = (x_t[i] - mean_t) * rstd_t * w[i] + bias[i];
          }
          StoreVector<VEC>(dst + step * row_stride, out);
        }
      }
    };
    for (int64_t k = 0;; ++k) {
      const int s = static_cast<int>(k % S);
      const uint32_t parity = static_cast<uint32_t>(k / S) & 1U;
      MbarrierWait(&full[s], parity);
      const TmaTileInfo info = infos[s];
      if (info.tile >= num_tiles) {
        break;
      }
      if (info.rows < kTileRows) {
        normalize.template operator()<true>(info, s, parity);
      } else {
        normalize.template operator()<false>(info, s, parity);
      }
      __syncwarp();
      if (l == 0) {
        MbarrierArrive(&empty[s]);
      }
    }
  } else if (warp >= kMomentWarp) {
    // The moments of each (timestep, group), by teams of W threads, from x
    // converted once if it fits in the registers.
    constexpr int kRows = NM * cuda_utils::kWarpSize / kRowThreads;
    constexpr int kSteps = kTileRows / kRows;
    constexpr bool kKeep = kSteps * VEC <= 32;
    const int vec = threadIdx.x % kRowThreads;
    const int row = (threadIdx.x - kMomentWarp * cuda_utils::kWarpSize) /
                    kRowThreads;
    const int lg = vec / W;
    const int lane = vec % W;
    const unsigned int mask = TeamMask<W>();
    // kChecked for a tile past the end of the sequence.
    const auto moments_of = [&]<bool kChecked>(const TmaTileInfo& info,
                                               int s) {
      const Vec* const vals = stages + s * kStageVecs + row * kRowThreads + vec;
      bool active[kSteps];
#pragma unroll
      for (int step = 0; step < kSteps; ++step) {
        active[step] = !kChecked || row + step * kRows < info.rows;
      }
      T_ACC xs[kKeep ? kSteps : 1][VEC];
      T_ACC sum[kSteps];
#pragma unroll
      for (int step = 0; step < kSteps; ++step) {
        sum[step] = T_ACC(0);
        if (active[step]) {
          T_ACC x_t[VEC];
          ConvertVector(LoadAligned(vals + step * kRows * kRowThreads), x_t);
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            const T_ACC x_i = x_t[i];
            if constexpr (kKeep) {
              xs[step][i] = x_i;
            }
            sum[step] += x_i;
          }
        }
      }
      TeamAllReduceSum<W>(sum, mask, static_cast<T_ACC*>(nullptr));
      T_ACC mean[kSteps];
      T_ACC m2[kSteps];
#pragma unroll
      for (int step = 0; step < kSteps; ++step) {
        mean[step] = sum[step] / static_cast<T_ACC>(kDg);
        m2[step] = T_ACC(0);
        if (active[step]) {
          T_ACC x_t[VEC];
          if constexpr (!kKeep) {
            ConvertVector(LoadAligned(vals + step * kRows * kRowThreads), x_t);
          }
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            const T_ACC x_i = kKeep ? xs[kKeep ? step : 0][i] : x_t[i];
            const T_ACC d = x_i - mean[step];
            m2[step] += d * d;
          }
        }
      }
      TeamAllReduceSum<W>(m2, mask, static_cast<T_ACC*>(nullptr));
      if (lane == 0) {
#pragma unroll
        for (int step = 0; step < kSteps; ++step) {
          const int u = row + step * kRows;
          stats1[s][lg][u] = active[step] ? mean[step] : T_ACC(0);
          stats2[s][lg][u] =
              active[step] ? m2[step] / static_cast<T_ACC>(kDg) : T_ACC(0);
        }
      }
    };
    for (int64_t k = 0;; ++k) {
      const int s = static_cast<int>(k % S);
      MbarrierWait(&full[s], static_cast<uint32_t>(k / S) & 1U);
      const TmaTileInfo info = infos[s];
      if (info.tile >= num_tiles) {
        break;
      }
      if (info.rows < kTileRows) {
        moments_of.template operator()<true>(info, s);
      } else {
        moments_of.template operator()<false>(info, s);
      }
      __syncwarp();
      if (l == 0) {
        MbarrierArrive(&moments[s]);
      }
    }
  } else if (warp == kFinalWarp) {
    // The normalizing statistics of the parts of each tile from their scans
    // and prefixes, lane l taking timestep l of each.
    for (int64_t k = 0;; ++k) {
      const int s = static_cast<int>(k % S);
      const uint32_t parity = static_cast<uint32_t>(k / S) & 1U;
      MbarrierWait(&full[s], parity);
      const TmaTileInfo info = infos[s];
      if (info.tile >= num_tiles) {
        break;
      }
      MbarrierWait(&prefixed[s], parity);
#pragma unroll
      for (int part = 0; part < kParts; ++part) {
        const int n = part % GT;
        const int u = part / GT * kTokens + l;
        const int64_t t = info.chunk * kTokens + u;
        const int64_t seq = info.b * G + info.g0 + n;
        DecayState inclusive;
        DecayState prefix;
        float* const f = reinterpret_cast<float*>(&inclusive);
        float* const p = reinterpret_cast<float*>(&prefix);
#pragma unroll
        for (int i = 0; i < kStateFloats; ++i) {
          f[i] = inclusive_shared[s][part][i][l];
          p[i] = prefix_shared[s][part][i];
        }
        const DecayState state = DecayStateProd{}(prefix, inclusive);
        float curr_mean = 0.0f;
        float curr_rstd = rsqrtf(eps);
        if (state.d1 > 0.0f && state.d2 > 0.0f) {
          curr_mean = state.mean / state.d1;
          curr_rstd = rsqrtf(state.var / state.d2 + eps);
        }
        if (t < L) {
          cummean[seq * L + t] = static_cast<T_ACC>(curr_mean);
          cumrstd[seq * L + t] = static_cast<T_ACC>(curr_rstd);
          if (t == L - 1) {
            if (mean_out != nullptr) {
              mean_out[seq] = static_cast<T_ACC>(state.mean);
            }
            if (var_out != nullptr) {
              var_out[seq] = static_cast<T_ACC>(state.var);
            }
          }
        }
        stats1[s][n][u] = static_cast<T_ACC>(curr_mean);
        stats2[s][n][u] = static_cast<T_ACC>(curr_rstd);
      }
      __syncwarp();
      if (l == 0) {
        MbarrierArrive(&scanned[s]);
      }
    }
  } else if (warp >= kParts) {
    // The prefixes of group g0 + w, for the tiles k % NL == j.
    const int w = (warp - kParts) % GT;
    const int j = (warp - kParts) / GT;
    for (int64_t k = j;; k += NL) {
      const int s = static_cast<int>(k % S);
      const uint32_t parity = static_cast<uint32_t>(k / S) & 1U;
      MbarrierWait(&full[s], parity);
      const TmaTileInfo info = infos[s];
      if (info.tile >= num_tiles) {
        break;
      }
      const int64_t b = info.b;
      const int64_t chunk = info.chunk;
      const int64_t seq = b * G + info.g0 + w;
      DecayState first_prefix = identity;
      if (chunk == 0) {
        first_prefix = InitialDecayState(b, seq, prev_count, prev_mean,
                                         prev_var, beta1, beta2);
      }
      MbarrierWait(&published[s], parity);
      // The aggregates of the parts of the group, chunk c being part
      // c * GT + w.
      DecayState aggregate[KC];
#pragma unroll
      for (int c = 0; c < KC; ++c) {
        float* const f = reinterpret_cast<float*>(&aggregate[c]);
#pragma unroll
        for (int i = 0; i < kStateFloats; ++i) {
          f[i] = inclusive_shared[s][c * GT + w][i][kTokens - 1];
        }
      }
      const int64_t first_tile = seq * num_chunks + chunk;
      // The first chunk of a chain publishes its inclusive prefix, not its
      // aggregate.
      if (chunk == 0) {
        tiles.WarpPublish(tiles.inclusive, first_tile,
                          DecayStateProd{}(first_prefix, aggregate[0]));
      }
      DecayState prefix[KC];
      if constexpr (KC == 1) {
        prefix[0] = WarpLookBackPublished<DecayStateProd>(
            tiles, first_tile, seq * num_chunks, first_prefix, aggregate[0],
            identity);
      } else {
        WarpLookBackPrefixPair<DecayStateProd>(
            tiles, first_tile, seq * num_chunks, first_prefix, aggregate[0],
            identity, &prefix[0], &prefix[1]);
        PublishCheckpoint<DecayStateProd>(tiles, first_tile, seq * num_chunks,
                                          prefix[0], aggregate[0]);
        if (chunk + 1 < num_chunks) {
          PublishCheckpoint<DecayStateProd>(tiles, first_tile + 1,
                                            seq * num_chunks, prefix[1],
                                            aggregate[1]);
        }
      }
      if (l == 0) {
#pragma unroll
        for (int c = 0; c < KC; ++c) {
          const float* const p = reinterpret_cast<const float*>(&prefix[c]);
#pragma unroll
          for (int i = 0; i < kStateFloats; ++i) {
            prefix_shared[s][c * GT + w][i] = p[i];
          }
        }
      }
      __syncwarp();
      if (l == 0) {
        MbarrierArrive(&prefixed[s]);
      }
    }
  } else {
    // The scan of part p, group g0 + n over chunk c of each tile, lane l
    // taking timestep l of the chunk, and its aggregate published if the
    // chunk is in the sequence and not the first of its chain.
    const int p = warp;
    const int n = p % GT;
    const int c = p / GT;
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    for (int64_t k = 0;; ++k) {
      const int s = static_cast<int>(k % S);
      const uint32_t parity = static_cast<uint32_t>(k / S) & 1U;
      MbarrierWait(&full[s], parity);
      const TmaTileInfo info = infos[s];
      if (info.tile >= num_tiles) {
        break;
      }
      const int64_t b = info.b;
      const int64_t chunk = info.chunk + c;
      const int64_t t = chunk * kTokens + l;
      const int64_t seq = b * G + info.g0 + n;
      const bool valid = t < L;
      const unsigned char* const bos_stage =
          reinterpret_cast<const unsigned char*>(stages + s * kStageVecs) +
          kBosOffset;
      const bool bos =
          valid && (bos_tma ? bos_stage[c * kTokens + l] != 0
                            : bos_mask != nullptr && bos_mask[b * L + t]);
      MbarrierWait(&moments[s], parity);
      DecayState item = identity;
      if (valid) {
        item = {bos ? 0.0f : beta1,
                one_minus_beta1 *
                    static_cast<float>(stats1[s][n][c * kTokens + l]),
                one_minus_beta1,
                bos ? 0.0f : beta2,
                one_minus_beta2 *
                    static_cast<float>(stats2[s][n][c * kTokens + l]),
                one_minus_beta2};
      }
      DecayState inclusive;
      WarpScan(scan_storage[warp]).InclusiveScan(item, inclusive,
                                                 DecayStateProd{});
      const DecayState aggregate = cub::ShuffleIndex<cuda_utils::kWarpSize>(
          inclusive, cuda_utils::kWarpSize - 1, 0xFFFFFFFFU);
      if (chunk != 0 && chunk < num_chunks) {
        tiles.WarpPublish(tiles.aggregate, seq * num_chunks + chunk,
                          aggregate);
      }
      const float* const f = reinterpret_cast<const float*>(&inclusive);
#pragma unroll
      for (int j = 0; j < kStateFloats; ++j) {
        inclusive_shared[s][p][j][l] = f[j];
      }
      __syncwarp();
      if (l == 0) {
        MbarrierArrive(&published[s]);
      }
    }
  }
#endif
}

// The TMA backward runs this many blocks per SM.
constexpr int kTmaBwdBlocks = 2;

// Waits at named barrier id (not 0, which __syncthreads takes) for threads, a
// multiple of kWarpSize.
__device__ __forceinline__ void NamedBarrierSync(int id, int threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

// TMA backward without padding; warp roles as in the
// TMA forward, each (batch, group) walked from its last tile:
// - producer: copies x, dy, cummean, cumrstd, the counts and gamma of a tile into a stage;
// - sum set j of NS, NM warps each: group sums and gamma/beta gradients of tiles k % NS == j,
//   summed over the tile's rows into a row of partials;
// - publisher warp w < GT: scans group g0 + w backwards and publishes its aggregate;
// - look-back warp GT * (1 + j) + w: prefixes for tiles k % NL == j;
// - final warp: gradients flowing into the timesteps;
// - NN dx warps: dx.
// Stage mbarriers: full, sums, published, prefixed, scanned, empty.
// kMemoryEfficient: x_map covers y and a stage also holds beta.
template <typename T, typename T_ACC, int VEC, int W, int GT, int NL, int NS,
          int NM, int NN, int S, bool kMemoryEfficient>
__global__ void __launch_bounds__(((NL + 1) * GT + NS * NM + NN + 2) *
                                      cuda_utils::kWarpSize,
                                  kTmaBwdBlocks)
    GroupTimestepDecayNormLookbackTmaBwdKernel(
        int64_t B, int64_t L, int64_t G, int64_t num_chunks, int64_t num_tiles,
        const __grid_constant__ CUtensorMap x_map,
        const __grid_constant__ CUtensorMap dy_map,
        const __grid_constant__ CUtensorMap cummean_map,
        const __grid_constant__ CUtensorMap cumrstd_map,
        const __grid_constant__ CUtensorMap count_map,
        const T* __restrict__ gamma, const T* __restrict__ beta,
        const bool* __restrict__ bos_mask, const T_ACC* __restrict__ mean_grad,
        const T_ACC* __restrict__ var_grad, float beta1, float beta2, float eps,
        ScanTiles<float4> tiles, T* __restrict__ dx,
        T_ACC* __restrict__ prev_mean_grad, T_ACC* __restrict__ prev_var_grad,
        T_ACC* __restrict__ gamma_grad_partial,
        T_ACC* __restrict__ beta_grad_partial) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int kTokens = cuda_utils::kWarpSize;
  constexpr int kDg = W * VEC;
  constexpr int kRowThreads = GT * W;
  constexpr int kCols = kRowThreads * VEC;
  static_assert(NM * cuda_utils::kWarpSize % kRowThreads == 0 &&
                    NN * cuda_utils::kWarpSize % kRowThreads == 0 &&
                    NM * cuda_utils::kWarpSize <= kTokens * kRowThreads &&
                    NN * cuda_utils::kWarpSize <= kTokens * kRowThreads,
                "the warps load a tile of kWarpSize timesteps evenly");
  static_assert(NL <= S && NS <= S, "the warps are on tiles in the stages");
  constexpr int kFinalWarp = (NL + 1) * GT;
  constexpr int kSumWarp = kFinalWarp + 1;
  constexpr int kDxWarp = kSumWarp + NS * NM;
  constexpr int kProducer = kDxWarp + NN;
  using Vec = AlignedVector<T, VEC>;
  using WarpScan = cub::WarpScan<float4>;
  // A stage: x, dy, cummean, cumrstd, the counts, gamma and (memory efficient)
  // beta of a tile, each at a multiple of 128 bytes but gamma and beta, 16.
  constexpr uint32_t kTileBytes = kTokens * kRowThreads * sizeof(Vec);
  constexpr uint32_t kStatBytes = GT * kTokens * sizeof(T_ACC);
  constexpr uint32_t kCountBytes = kTokens * sizeof(int64_t);
  constexpr uint32_t kGammaBytes = kRowThreads * sizeof(Vec);
  constexpr uint32_t kDyOffset = kTileBytes;
  constexpr uint32_t kMeanOffset = 2 * kTileBytes;
  constexpr uint32_t kRstdOffset = kMeanOffset + kStatBytes;
  constexpr uint32_t kCountOffset = kRstdOffset + kStatBytes;
  constexpr uint32_t kGammaOffset = kCountOffset + kCountBytes;
  constexpr uint32_t kBetaOffset = kGammaOffset + kGammaBytes;
  constexpr uint32_t kCopyBytes =
      kBetaOffset + (kMemoryEfficient ? kGammaBytes : 0);
  constexpr uint32_t kStageBytes = (kCopyBytes + 127) / 128 * 128;
  static_assert(kStatBytes % 128 == 0, "the boxes are at multiples of 128");
  // The gamma/beta gradients of a tile are summed over its rows 16 bytes at a
  // time.
  constexpr int kAccVec = 16 / sizeof(T_ACC);
  static_assert(kCols % kAccVec == 0, "the columns fill 16 bytes");
  using AccVec = AlignedVector<T_ACC, kAccVec>;

  extern __shared__ __align__(128) unsigned char tma_shared[];
  __shared__ typename WarpScan::TempStorage scan_storage[GT];
  // The group sums of the (timestep, group): sum(x), sum(g), sum(g * (x - c)),
  // then the mean of x, du and dv.
  __shared__ T_ACC stats1[S][GT][kTokens];
  __shared__ T_ACC stats2[S][GT][kTokens];
  __shared__ T_ACC stats3[S][GT][kTokens];
  // The exclusive scans of the groups over the tile, their aggregates and
  // their prefixes.
  __shared__ float4 exclusive_shared[S][GT][kTokens];
  __shared__ float4 aggregate_shared[S][GT];
  __shared__ float4 prefix_shared[S][GT];
  // The local gradients and maps of the timesteps.
  __shared__ float2 local_shared[S][GT][kTokens];
  __shared__ float4 map_shared[S][GT][kTokens];
  // The gamma/beta gradients of a tile, per thread of a set of sum warps,
  // thus [kRows][kCols].
  __shared__ __align__(16) T_ACC grads_shared[NS][2 * NM *
                                                  cuda_utils::kWarpSize * VEC];
  __shared__ TmaTileInfo infos[S];
  __shared__ uint64_t full[S];
  __shared__ uint64_t sums_done[S];
  __shared__ uint64_t published[S];
  __shared__ uint64_t prefixed[S];
  __shared__ uint64_t scanned[S];
  __shared__ uint64_t empty[S];

  const int64_t num_sets = B * (G / GT);
  const int64_t H = G * kDg;
  const int warp = threadIdx.x / cuda_utils::kWarpSize;
  const int l = threadIdx.x % cuda_utils::kWarpSize;
  const float log_beta1 = logf(beta1);
  const float log_beta2 = logf(beta2);
  const T_ACC group_size = static_cast<T_ACC>(kDg);
  const T_ACC coef = T_ACC(1) / group_size;
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < S; ++s) {
      MbarrierInit(&full[s], 1);
      MbarrierInit(&sums_done[s], NM);
      MbarrierInit(&published[s], GT);
      MbarrierInit(&prefixed[s], GT);
      MbarrierInit(&scanned[s], 1);
      MbarrierInit(&empty[s], NN);
    }
    MbarrierInitFence();
  }
  __syncthreads();

  const auto stage_bytes = [&](int s) {
    return tma_shared + s * kStageBytes;
  };
  // Whether the timestep t0 + u starts a sequence.
  const auto is_bos = [&](const TmaTileInfo& info, int u) {
    const int64_t t = info.chunk * kTokens + u;
    return t < L && bos_mask != nullptr && bos_mask[info.b * L + t];
  };
  // The local gradients and the map of the timestep t = t0 + u of group n.
  const auto local_map = [&](const TmaTileInfo& info, int s, int n, int u,
                             bool bos, float* local_u, float* local_v) {
    const int64_t t = info.chunk * kTokens + u;
    const bool valid = t < L;
    const T_ACC* const rstd =
        reinterpret_cast<const T_ACC*>(stage_bytes(s) + kRstdOffset);
    const int64_t* const count =
        reinterpret_cast<const int64_t*>(stage_bytes(s) + kCountOffset);
    float u_t = 0.0f;
    float v_t = 0.0f;
    if (valid && count[u] > 0) {
      LocalGradients(static_cast<float>(rstd[n * kTokens + u]),
                     static_cast<float>(stats3[s][n][u]),
                     static_cast<float>(stats2[s][n][u]), count[u], log_beta1,
                     log_beta2, &u_t, &v_t);
    }
    *local_u = static_cast<float>(static_cast<T_ACC>(u_t));
    *local_v = static_cast<float>(static_cast<T_ACC>(v_t));
    float4 map = IdentityAffine4();
    if (valid) {
      map = bos ? MakeAffine4(0.0f, 0.0f, 0.0f, 0.0f)
                : MakeAffine4(beta1 * *local_u, beta1, beta2 * *local_v,
                              beta2);
    }
    return map;
  };
  // The prefix of the first tile of a chain, the last of its (batch, group).
  const auto first_prefix = [&](int64_t k, int64_t seq) {
    return k == 0 ? MakeAffine4(static_cast<float>(mean_grad[seq]), 0.0f,
                                static_cast<float>(var_grad[seq]), 0.0f)
                  : IdentityAffine4();
  };

  if (warp == kProducer) {
    // As in the TMA forward, but the tiles of each (batch, group) are taken
    // from the last chunk.
    const unsigned int sets = static_cast<unsigned int>(num_sets);
    const unsigned int sets_per_batch = static_cast<unsigned int>(G / GT);
    unsigned int claimed = 0U;
    if (l == 0) {
      claimed = atomicAdd(tiles.next_tile, kTmaClaimTiles);
    }
    unsigned int first = 0U;
    int taken = kTmaClaimTiles;
    for (int64_t k = 0;; ++k) {
      const int s = static_cast<int>(k % S);
      if (k >= S) {
        MbarrierWait(&empty[s], static_cast<uint32_t>(k / S - 1) & 1U);
      }
      if (taken == kTmaClaimTiles) {
        first = __shfl_sync(0xFFFFFFFFU, claimed, 0);
        taken = 0;
        if (l == 0 && first < num_tiles) {
          claimed = atomicAdd(tiles.next_tile, kTmaClaimTiles);
        }
      }
      const unsigned int tile = first + taken++;
      const unsigned int kk = tile / sets;
      const unsigned int set = tile - kk * sets;
      const unsigned int b = set / sets_per_batch;
      const unsigned int g0 = (set - b * sets_per_batch) * GT;
      if (tile >= num_tiles) {
        // The end, past the last tile of each set of look-back or sum warps.
        for (int64_t e = k; e < k + (NL > NS ? NL : NS); ++e) {
          const int se = static_cast<int>(e % S);
          if (e > k && e >= S) {
            MbarrierWait(&empty[se], static_cast<uint32_t>(e / S - 1) & 1U);
          }
          if (l == 0) {
            infos[se].tile = num_tiles;
            MbarrierArrive(&full[se]);
          }
        }
        break;
      }
      const int64_t chunk = num_chunks - 1 - kk;
      const int64_t t0 = chunk * kTokens;
      const int rows = static_cast<int>(L - t0 < kTokens ? L - t0 : kTokens);
      if (l == 0) {
        infos[s] = TmaTileInfo{tile, b, g0, chunk, rows};
        unsigned char* const dst = stage_bytes(s);
        MbarrierArriveExpectTx(&full[s], kCopyBytes);
        TensorCopy3d(dst, &x_map, static_cast<int>(g0 * kDg),
                     static_cast<int>(t0), static_cast<int>(b), &full[s]);
        TensorCopy3d(dst + kDyOffset, &dy_map, static_cast<int>(g0 * kDg),
                     static_cast<int>(t0), static_cast<int>(b), &full[s]);
        TensorCopy2d(dst + kMeanOffset, &cummean_map, static_cast<int>(t0),
                     static_cast<int>(b * G + g0), &full[s]);
        TensorCopy2d(dst + kRstdOffset, &cumrstd_map, static_cast<int>(t0),
                     static_cast<int>(b * G + g0), &full[s]);
        TensorCopy2d(dst + kCountOffset, &count_map, static_cast<int>(t0),
                     static_cast<int>(b), &full[s]);
        BulkCopy(dst + kGammaOffset, gamma + int64_t(g0) * kDg, kGammaBytes,
                 &full[s]);
        if constexpr (kMemoryEfficient) {
          BulkCopy(dst + kBetaOffset, beta + int64_t(g0) * kDg, kGammaBytes,
                   &full[s]);
        }
      }
    }
  } else if (warp >= kDxWarp) {
    // dx, as GroupTimestepDecayNormLookbackBwdKernel works it out.
    constexpr int kRows = NN * cuda_utils::kWarpSize / kRowThreads;
    constexpr int kSteps = kTokens / kRows;
    const int vec = threadIdx.x % kRowThreads;
    const int row = (threadIdx.x - kDxWarp * cuda_utils::kWarpSize) /
                    kRowThreads;
    const int lg = vec / W;
    const int64_t row_stride = kRows * H;
    for (int64_t k = 0;; ++k) {
      const int s = static_cast<int>(k % S);
      const uint32_t parity = static_cast<uint32_t>(k / S) & 1U;
      MbarrierWait(&full[s], parity);
      const TmaTileInfo info = infos[s];
      if (info.tile >= num_tiles) {
        break;
      }
      const unsigned char* const bytes = stage_bytes(s);
      const Vec* const xs = reinterpret_cast<const Vec*>(bytes) +
                            row * kRowThreads + vec;
      const Vec* const dys = reinterpret_cast<const Vec*>(bytes + kDyOffset) +
                             row * kRowThreads + vec;
      const T_ACC* const rstds =
          reinterpret_cast<const T_ACC*>(bytes + kRstdOffset) + lg * kTokens;
      T_ACC w[VEC];
      T_ACC bias[VEC];
      {
        const Vec gamma_vec = LoadAligned(
            reinterpret_cast<const Vec*>(bytes + kGammaOffset) + vec);
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          w[i] = static_cast<T_ACC>(gamma_vec.val[i]);
        }
        if constexpr (kMemoryEfficient) {
          const Vec beta_vec = LoadAligned(
              reinterpret_cast<const Vec*>(bytes + kBetaOffset) + vec);
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            bias[i] = static_cast<T_ACC>(beta_vec.val[i]);
          }
        }
      }
      T* const dst =
          dx + (info.b * L + info.chunk * kTokens + row) * H + info.g0 * kDg +
          int64_t(vec) * VEC;
      MbarrierWait(&scanned[s], parity);
#pragma unroll
      for (int step = 0; step < kSteps; ++step) {
        const int u = row + step * kRows;
        if (u < info.rows) {
          T_ACC out[VEC];
          const Vec xv = LoadAligned(xs + step * kRows * kRowThreads);
          const Vec dyv = LoadAligned(dys + step * kRows * kRowThreads);
          const T_ACC mean = stats1[s][lg][u];
          const T_ACC rstd = rstds[u];
          // coef * (du + 2 * dv * (x - mean)), exactly so for a power of 2
          // group size.
          const T_ACC du_coef = MulRn(coef, stats2[s][lg][u]);
          const T_ACC dv_coef = T_ACC(2) * coef * stats3[s][lg][u];
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            const T_ACC scale = rstd * w[i];
            const T_ACC x_val =
                kMemoryEfficient
                    ? CenteredFromOutput(static_cast<T_ACC>(xv.val[i]),
                                         bias[i], scale, eps)
                    : static_cast<T_ACC>(xv.val[i]);
            out[i] = static_cast<T_ACC>(dyv.val[i]) * scale +
                     (dv_coef * (x_val - mean) + du_coef);
          }
          StoreVector<VEC>(dst + step * row_stride, out);
        }
      }
      __syncwarp();
      if (l == 0) {
        MbarrierArrive(&empty[s]);
      }
    }
  } else if (warp >= kSumWarp) {
    // The group sums and the gamma/beta gradients, by teams of W threads.
    constexpr int kThreads = NM * cuda_utils::kWarpSize;
    constexpr int kRows = kThreads / kRowThreads;
    constexpr int kSteps = kTokens / kRows;
    const int j = (warp - kSumWarp) / NM;
    const int ct = threadIdx.x - (kSumWarp + j * NM) * cuda_utils::kWarpSize;
    const int vec = threadIdx.x % kRowThreads;
    const int row = ct / kRowThreads;
    const int lg = vec / W;
    const int lane = vec % W;
    const unsigned int mask = TeamMask<W>();
    T_ACC* const gamma_shared = grads_shared[j];
    T_ACC* const beta_shared = grads_shared[j] + kThreads * VEC;
    for (int64_t k = j;; k += NS) {
      const int s = static_cast<int>(k % S);
      MbarrierWait(&full[s], static_cast<uint32_t>(k / S) & 1U);
      const TmaTileInfo info = infos[s];
      if (info.tile >= num_tiles) {
        break;
      }
      const unsigned char* const bytes = stage_bytes(s);
      const Vec* const xs = reinterpret_cast<const Vec*>(bytes) +
                            row * kRowThreads + vec;
      const Vec* const dys = reinterpret_cast<const Vec*>(bytes + kDyOffset) +
                             row * kRowThreads + vec;
      const T_ACC* const means =
          reinterpret_cast<const T_ACC*>(bytes + kMeanOffset) + lg * kTokens;
      const T_ACC* const rstds =
          reinterpret_cast<const T_ACC*>(bytes + kRstdOffset) + lg * kTokens;
      T_ACC w[VEC];
      T_ACC bias[VEC];
      {
        const Vec gamma_vec = LoadAligned(
            reinterpret_cast<const Vec*>(bytes + kGammaOffset) + vec);
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          w[i] = static_cast<T_ACC>(gamma_vec.val[i]);
        }
        if constexpr (kMemoryEfficient) {
          const Vec beta_vec = LoadAligned(
              reinterpret_cast<const Vec*>(bytes + kBetaOffset) + vec);
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            bias[i] = static_cast<T_ACC>(beta_vec.val[i]);
          }
        }
      }
      T_ACC sums[kSteps * 3];
      T_ACC dgamma[VEC];
      T_ACC dbeta[VEC];
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        dgamma[i] = T_ACC(0);
        dbeta[i] = T_ACC(0);
      }
#pragma unroll
      for (int step = 0; step < kSteps; ++step) {
        const int u = row + step * kRows;
        sums[3 * step] = T_ACC(0);
        sums[3 * step + 1] = T_ACC(0);
        sums[3 * step + 2] = T_ACC(0);
        if (u < info.rows) {
          // Memory efficient: x_val is already x - cummean.
          const T_ACC c = kMemoryEfficient ? T_ACC(0) : means[u];
          const T_ACC r = rstds[u];
          const Vec xv = LoadAligned(xs + step * kRows * kRowThreads);
          const Vec dyv = LoadAligned(dys + step * kRows * kRowThreads);
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            const T_ACC x_val =
                kMemoryEfficient
                    ? CenteredFromOutput(static_cast<T_ACC>(xv.val[i]),
                                         bias[i], r * w[i], eps)
                    : static_cast<T_ACC>(xv.val[i]);
            const T_ACC dy_val = static_cast<T_ACC>(dyv.val[i]);
            const T_ACC x_c = x_val - c;
            const T_ACC grad_hat = dy_val * w[i];
            sums[3 * step] += x_val;
            sums[3 * step + 1] += grad_hat;
            sums[3 * step + 2] += grad_hat * x_c;
            dgamma[i] += dy_val * x_c * r;
            dbeta[i] += dy_val;
          }
        }
      }
      TeamAllReduceSum<W>(sums, mask, static_cast<T_ACC*>(nullptr));
      if (lane == 0) {
#pragma unroll
        for (int step = 0; step < kSteps; ++step) {
          const int u = row + step * kRows;
          stats1[s][lg][u] = sums[3 * step];
          stats2[s][lg][u] = sums[3 * step + 1];
          stats3[s][lg][u] = sums[3 * step + 2];
        }
      }
      __syncwarp();
      if (l == 0) {
        MbarrierArrive(&sums_done[s]);
      }
      // The gamma/beta gradients of the tile, summed over its rows in order,
      // once those of the tile before are read.
      NamedBarrierSync(1 + j, kThreads);
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        gamma_shared[ct * VEC + i] = dgamma[i];
        beta_shared[ct * VEC + i] = dbeta[i];
      }
      NamedBarrierSync(1 + j, kThreads);
      const int64_t h =
          (info.b * num_chunks + info.chunk) * H + info.g0 * kDg;
#pragma unroll
      for (int q = ct; q < 2 * kCols / kAccVec; q += kThreads) {
        const bool is_beta = q >= kCols / kAccVec;
        const int c = (is_beta ? q - kCols / kAccVec : q) * kAccVec;
        const T_ACC* const src = is_beta ? beta_shared : gamma_shared;
        AccVec sum = *reinterpret_cast<const AccVec*>(src + c);
#pragma unroll
        for (int r = 1; r < kRows; ++r) {
          const AccVec v =
              *reinterpret_cast<const AccVec*>(src + r * kCols + c);
#pragma unroll
          for (int i = 0; i < kAccVec; ++i) {
            sum.val[i] += v.val[i];
          }
        }
        *reinterpret_cast<AccVec*>(
            (is_beta ? beta_grad_partial : gamma_grad_partial) + h + c) = sum;
      }
    }
  } else if (warp == kFinalWarp) {
    // The gradients flowing into the timesteps, lane l taking the l-th from
    // the end of the tile.
    const int u = kTokens - 1 - l;
    for (int64_t k = 0;; ++k) {
      const int s = static_cast<int>(k % S);
      const uint32_t parity = static_cast<uint32_t>(k / S) & 1U;
      MbarrierWait(&full[s], parity);
      const TmaTileInfo info = infos[s];
      if (info.tile >= num_tiles) {
        break;
      }
      const int64_t t = info.chunk * kTokens + u;
      const bool valid = t < L;
      MbarrierWait(&prefixed[s], parity);
#pragma unroll
      for (int n = 0; n < GT; ++n) {
        const int64_t seq = info.b * G + info.g0 + n;
        const float local_u = local_shared[s][n][l].x;
        const float local_v = local_shared[s][n][l].y;
        const float4 map = map_shared[s][n][l];
        const float4 state =
            Affine4Prod{}(prefix_shared[s][n], exclusive_shared[s][n][l]);
        const float u_in = state.x;
        const float v_in = state.z;
        T_ACC du = T_ACC(0);
        T_ACC dv = T_ACC(0);
        if (valid) {
          du = static_cast<T_ACC>((1.0f - beta1) * (u_in + local_u));
          dv = static_cast<T_ACC>((1.0f - beta2) * (v_in + local_v));
        }
        if (t == 0) {
          prev_mean_grad[seq] = static_cast<T_ACC>(ApplyAffineX(map, u_in));
          prev_var_grad[seq] = static_cast<T_ACC>(ApplyAffineY(map, v_in));
        }
        stats1[s][n][u] = valid ? stats1[s][n][u] / group_size : T_ACC(0);
        stats2[s][n][u] = du;
        stats3[s][n][u] = dv;
      }
      __syncwarp();
      if (l == 0) {
        MbarrierArrive(&scanned[s]);
      }
    }
  } else if (warp >= GT) {
    // The prefix of group g0 + w, for the tiles k % NL == j.
    const int w = warp % GT;
    const int j = warp / GT - 1;
    for (int64_t k = j;; k += NL) {
      const int s = static_cast<int>(k % S);
      const uint32_t parity = static_cast<uint32_t>(k / S) & 1U;
      MbarrierWait(&full[s], parity);
      const TmaTileInfo info = infos[s];
      if (info.tile >= num_tiles) {
        break;
      }
      const int64_t seq = info.b * G + info.g0 + w;
      const int64_t kk = num_chunks - 1 - info.chunk;
      const float4 prefix0 = first_prefix(kk, seq);
      MbarrierWait(&published[s], parity);
      const float4 prefix = WarpLookBackPublished<Affine4Prod>(
          tiles, seq * num_chunks + kk, seq * num_chunks, prefix0,
          aggregate_shared[s][w], IdentityAffine4());
      if (l == 0) {
        prefix_shared[s][w] = prefix;
      }
      __syncwarp();
      if (l == 0) {
        MbarrierArrive(&prefixed[s]);
      }
    }
  } else {
    // The scan of group g0 + warp backwards over the timesteps of the tile,
    // lane l taking the l-th from the end, and the aggregate published.
    const int u = kTokens - 1 - l;
    for (int64_t k = 0;; ++k) {
      const int s = static_cast<int>(k % S);
      const uint32_t parity = static_cast<uint32_t>(k / S) & 1U;
      MbarrierWait(&full[s], parity);
      const TmaTileInfo info = infos[s];
      if (info.tile >= num_tiles) {
        break;
      }
      const int64_t seq = info.b * G + info.g0 + warp;
      const int64_t kk = num_chunks - 1 - info.chunk;
      const float4 prefix0 = first_prefix(kk, seq);
      const bool bos = is_bos(info, u);
      MbarrierWait(&sums_done[s], parity);
      float local_u;
      float local_v;
      const float4 map = local_map(info, s, warp, u, bos, &local_u, &local_v);
      local_shared[s][warp][l] = make_float2(local_u, local_v);
      map_shared[s][warp][l] = map;
      float4 exclusive;
      float4 aggregate;
      WarpScan(scan_storage[warp]).ExclusiveScan(
          map, exclusive, IdentityAffine4(), Affine4Prod{}, aggregate);
      PublishTile<Affine4Prod>(tiles, seq * num_chunks + kk,
                               seq * num_chunks, prefix0, aggregate);
      exclusive_shared[s][warp][l] = exclusive;
      if (l == 0) {
        aggregate_shared[s][warp] = aggregate;
      }
      __syncwarp();
      if (l == 0) {
        MbarrierArrive(&published[s]);
      }
    }
  }
#endif
}

// Each block normalizes TT consecutive timesteps of one batch, VEC elements per load.
template <typename T, typename T_ACC, int VEC, int TT>
__global__ void GroupTimestepDecayNormLookbackApplyFwdKernel(
    int64_t L, int64_t G, int64_t Dg, const T* __restrict__ x,
    const T_ACC* __restrict__ cummean, const T_ACC* __restrict__ cumrstd,
    const T* __restrict__ gamma, const T* __restrict__ beta,
    const bool* __restrict__ padding_mask, T* __restrict__ y) {
  extern __shared__ unsigned char raw_shared[];
  T_ACC* mean_shared = reinterpret_cast<T_ACC*>(raw_shared);
  T_ACC* rstd_shared = mean_shared + G * TT;

  const int64_t b = blockIdx.y;
  const int64_t t0 = int64_t(blockIdx.x) * TT;
  const int64_t H = G * Dg;

  // [G, TT], so that consecutive threads read consecutive timesteps.
  for (int64_t k = threadIdx.x; k < G * TT; k += blockDim.x) {
    const int64_t t = t0 + k % TT;
    if (t < L) {
      const int64_t idx = GroupIndex(b, k / TT, t, G, L);
      mean_shared[k] = cummean[idx];
      rstd_shared[k] = cumrstd[idx];
    }
  }
  __syncthreads();

  const int64_t num_group_vecs = Dg / VEC;
  for (int64_t j = threadIdx.x; j < H / VEC; j += blockDim.x) {
    const int64_t g = j / num_group_vecs;
    T_ACC w[VEC];
    T_ACC bias[VEC];
    LoadVector<VEC>(gamma + j * VEC, w);
    LoadVector<VEC>(beta + j * VEC, bias);
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) {
      const int64_t t = t0 + tt;
      if (t < L) {
        const int64_t offset = (b * L + t) * H + j * VEC;
        T_ACC out[VEC];
        if (padding_mask != nullptr && padding_mask[b * L + t]) {
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            out[k] = T_ACC(0);
          }
        } else {
          T_ACC x_val[VEC];
          LoadVector<VEC>(x + offset, x_val);
          const T_ACC mean = mean_shared[g * TT + tt];
          const T_ACC rstd = rstd_shared[g * TT + tt];
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            out[k] = (x_val[k] - mean) * rstd * w[k] + bias[k];
          }
        }
        StoreVector<VEC>(y + offset, out);
      }
    }
  }
}

// The count of a timestep as (whether a bos was seen, tokens since then), so
// that a scan restarts at each bos. Padding contributes (0, 0).
struct CountCarry {
  int reset;
  int64_t count;
};

struct CountCarryOp {
  __device__ __forceinline__ CountCarry
  operator()(const CountCarry& prefix, const CountCarry& current) const {
    return {prefix.reset | current.reset,
            current.reset ? current.count : prefix.count + current.count};
  }
};

// The carry of timesteps [begin, end) of batch b, each thread taking a
// contiguous share in order.
template <int BLOCK_THREADS>
__device__ __forceinline__ CountCarry
BlockCountCarry(int64_t b, int64_t L, int64_t begin, int64_t end,
                const bool* __restrict__ bos_mask,
                const bool* __restrict__ padding_mask) {
  using CountReduce = cub::BlockScan<CountCarry, BLOCK_THREADS>;
  __shared__ typename CountReduce::TempStorage reduce_storage;
  const int64_t share = DeviceDivUp(end - begin, int64_t(BLOCK_THREADS));
  const int64_t t_begin = begin + int64_t(threadIdx.x) * share;
  const int64_t t_end = t_begin + share < end ? t_begin + share : end;
  CountCarry carry = {0, 0};
  // Read both masks unconditionally.
#pragma unroll 8
  for (int64_t t = t_begin; t < t_end; ++t) {
    const bool pad = padding_mask != nullptr && padding_mask[b * L + t];
    const bool bos = bos_mask != nullptr && bos_mask[b * L + t];
    if (!pad) {
      carry = CountCarryOp{}(carry, {bos ? 1 : 0, 1});
    }
  }
  CountCarry aggregate;
  CountReduce(reduce_storage).InclusiveScan(carry, carry, CountCarryOp{},
                                            aggregate);
  return aggregate;
}

// The final count of each batch, one block per batch.
template <int BLOCK_THREADS>
__global__ void __launch_bounds__(BLOCK_THREADS)
    FinalCountLookbackKernel(int64_t L, const bool* __restrict__ bos_mask,
                             const bool* __restrict__ padding_mask,
                             const int64_t* __restrict__ prev_count,
                             int64_t* __restrict__ count_out) {
  const int64_t b = blockIdx.x;
  const CountCarry total =
      BlockCountCarry<BLOCK_THREADS>(b, L, 0, L, bos_mask, padding_mask);
  if (threadIdx.x == 0) {
    const int64_t count = prev_count != nullptr ? prev_count[b] : 0;
    count_out[b] = total.reset ? total.count : count + total.count;
  }
}

// The count of every timestep, count_out [B, L], with a block per chunk of
// BLOCK_THREADS * ITEMS timesteps of a batch: it first reduces the timesteps
// before its chunk, then scans the chunk, ITEMS timesteps per thread.
template <int BLOCK_THREADS, int ITEMS>
__global__ void __launch_bounds__(BLOCK_THREADS)
    CountLookbackKernel(int64_t L, const bool* __restrict__ bos_mask,
                        const bool* __restrict__ padding_mask,
                        const int64_t* __restrict__ prev_count,
                        int64_t* __restrict__ count_out) {
  using CountScan = cub::BlockScan<CountCarry, BLOCK_THREADS>;
  __shared__ typename CountScan::TempStorage scan_storage;

  const int64_t b = blockIdx.y;
  const int64_t base = int64_t(blockIdx.x) * BLOCK_THREADS * ITEMS;
  const CountCarry before =
      BlockCountCarry<BLOCK_THREADS>(b, L, 0, base, bos_mask, padding_mask);
  const int64_t prev = prev_count != nullptr ? prev_count[b] : 0;
  const int64_t running = before.reset ? before.count : prev + before.count;

  const int64_t t_begin = base + int64_t(threadIdx.x) * ITEMS;
  CountCarry items[ITEMS];
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const int64_t t = t_begin + i;
    const bool valid =
        t < L && !(padding_mask != nullptr && padding_mask[b * L + t]);
    const bool bos = valid && bos_mask != nullptr && bos_mask[b * L + t];
    items[i] = {bos ? 1 : 0, valid ? 1 : 0};
  }
  CountCarry prefixes[ITEMS];
  CountScan(scan_storage).InclusiveScan(items, prefixes, CountCarryOp{});
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const int64_t t = t_begin + i;
    if (t < L) {
      count_out[b * L + t] = prefixes[i].reset ? prefixes[i].count
                                               : running + prefixes[i].count;
    }
  }
}

// kMemoryEfficient: x holds y.
template <typename T, typename T_ACC, int BLOCK_THREADS, bool kMemoryEfficient>
__global__ void RowwiseInternalGradientsLookbackKernel(
    int64_t B, int64_t G, int64_t Dg, int64_t L, float beta1, float beta2,
    float eps, const T* __restrict__ dy, const T* __restrict__ x,
    const T* __restrict__ gamma, const T* __restrict__ beta,
    const T_ACC* __restrict__ cummean,
    const T_ACC* __restrict__ cumrstd, const int64_t* __restrict__ count,
    const bool* __restrict__ padding_mask, T_ACC* __restrict__ group_mean,
    T_ACC* __restrict__ local_u, T_ACC* __restrict__ local_v) {
  const int64_t b = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int64_t tid = threadIdx.x;
  const int64_t h_base = g * Dg;
  const int64_t H = G * Dg;
  const int64_t num_chunks = DeviceDivUp(L, int64_t(BLOCK_THREADS));
  const float log_beta1 = logf(beta1);
  const float log_beta2 = logf(beta2);

  for (int64_t c = 0; c < num_chunks; ++c) {
    const int64_t t = c * int64_t(BLOCK_THREADS) + tid;
    if (t >= L) {
      continue;
    }

    const bool pad = padding_mask != nullptr && padding_mask[b * L + t];
    const int64_t idx = GroupIndex(b, g, t, G, L);
    const float cummean_t =
        kMemoryEfficient ? 0.0f : static_cast<float>(cummean[idx]);

    float mean = 0.0f;
    float ds = 0.0f;
    float db = 0.0f;
    if (!pad) {
      float sum_gx = 0.0f;
      const int64_t row_offset = XIndex(b, t, h_base, L, H);
      const T* x_row = x + row_offset;
      const T* dy_row = dy + row_offset;
      const T* gamma_row = gamma + h_base;
      const T* beta_row = beta + h_base;
      const T_ACC rstd_t = cumrstd[idx];

      for (int64_t d = 0; d < Dg; ++d) {
        const float x_val =
            kMemoryEfficient
                ? static_cast<float>(CenteredFromOutput(
                      static_cast<T_ACC>(x_row[d]),
                      static_cast<T_ACC>(beta_row[d]),
                      rstd_t * static_cast<T_ACC>(gamma_row[d]), eps))
                : static_cast<float>(x_row[d]);
        const float grad_hat =
            static_cast<float>(dy_row[d]) * static_cast<float>(gamma_row[d]);
        mean += (x_val - mean) /
                static_cast<float>(d + 1);
        db += grad_hat;
        sum_gx += grad_hat * x_val;
      }
      ds = kMemoryEfficient ? sum_gx : sum_gx - cummean_t * db;
    }

    float u_t = 0.0f;
    float v_t = 0.0f;
    const int64_t c_t = count[b * L + t];
    if (!pad && c_t > 0) {
      LocalGradients(static_cast<float>(cumrstd[idx]), ds, db, c_t, log_beta1,
                     log_beta2, &u_t, &v_t);
    }
    group_mean[idx] = static_cast<T_ACC>(mean);
    local_u[idx] = static_cast<T_ACC>(u_t);
    local_v[idx] = static_cast<T_ACC>(v_t);
  }
}

// Adds one load of a token to its sums [sum(x), sum(g), sum(g * (x - c))],
// g = dy * gamma, and to the gamma/beta gradients of its columns. With
// kMemoryEfficient, x holds y and c is 0.
template <int VEC, bool kMemoryEfficient, typename T, typename T_ACC>
__device__ __forceinline__ void AccumulateInternalGradients(
    const AlignedVector<T, VEC>& x, const AlignedVector<T, VEC>& dy,
    const AlignedVector<T, VEC>& gamma, const AlignedVector<T, VEC>& beta,
    T_ACC c, T_ACC r, float eps, T_ACC (&sums)[3], T_ACC (&dgamma)[VEC],
    T_ACC (&dbeta)[VEC]) {
#pragma unroll
  for (int k = 0; k < VEC; ++k) {
    const T_ACC x_val =
        kMemoryEfficient
            ? CenteredFromOutput(static_cast<T_ACC>(x.val[k]),
                                 static_cast<T_ACC>(beta.val[k]),
                                 r * static_cast<T_ACC>(gamma.val[k]), eps)
            : static_cast<T_ACC>(x.val[k]);
    const T_ACC dy_val = static_cast<T_ACC>(dy.val[k]);
    const T_ACC x_c = x_val - c;
    const T_ACC grad_hat = dy_val * static_cast<T_ACC>(gamma.val[k]);
    sums[0] += x_val;
    sums[1] += grad_hat;
    sums[2] += grad_hat * x_c;
    dgamma[k] += dy_val * x_c * r;
    dbeta[k] += dy_val;
  }
}

// Per-token group sums for the backward and [num_ranges, H] gamma/beta gradient partials,
// reduced by GammaBetaGradReduceLookbackKernel; the teams of a group take ranges
// num_team_ranges apart. Writes the mean of x and the local gradients from
// ds = sum(g * (x - cummean)) and db = sum(g), g = dy * gamma, forming x - cummean per
// element. One token at a time; lane k writes the k-th of WIDTH token outputs.
// kMemoryEfficient: x holds y.
template <typename T, typename T_ACC, int VEC, int ITEMS, int WIDTH,
          bool kMemoryEfficient>
__global__ void __launch_bounds__(TeamBlockSize(WIDTH))
    RowwiseInternalGradientsTeamLookbackKernel(
        int64_t N, int64_t L, int64_t G, int64_t Dg, int64_t num_ranges,
        int64_t range_size, int64_t num_team_ranges, float beta1, float beta2,
        float eps, const T* __restrict__ dy, const T* __restrict__ x,
        const T* __restrict__ gamma, const T* __restrict__ beta,
        const T_ACC* __restrict__ cummean,
        const T_ACC* __restrict__ cumrstd, const int64_t* __restrict__ count,
        const bool* __restrict__ padding_mask, T_ACC* __restrict__ group_mean,
        T_ACC* __restrict__ local_u, T_ACC* __restrict__ local_v,
        T_ACC* __restrict__ gamma_grad_partial,
        T_ACC* __restrict__ beta_grad_partial) {
  constexpr int kOutputTokens = WIDTH == 0 ? 1 : WIDTH;
  __shared__ T_ACC shm[WIDTH == 0 ? 3 * cuda_utils::kWarpSize : 1];

  const int width = WIDTH > 0 ? WIDTH : blockDim.x;
  const int64_t team =
      (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / width;
  const int lane = threadIdx.x % width;
  const int64_t g = team % G;
  if (team >= G * num_team_ranges) {
    return;
  }
  const unsigned int mask = TeamMask<WIDTH>();
  const T_ACC group_size = static_cast<T_ACC>(Dg);
  const int64_t H = G * Dg;
  const int64_t num_vecs = Dg / VEC;
  const float log_beta1 = logf(beta1);
  const float log_beta2 = logf(beta2);

  AlignedVector<T, VEC> w[ITEMS];
  AlignedVector<T, VEC> bias[ITEMS];
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const int64_t v = lane + int64_t(i) * width;
    if (v < num_vecs) {
      w[i] = *reinterpret_cast<const AlignedVector<T, VEC>*>(
          gamma + g * Dg + v * VEC);
      if constexpr (kMemoryEfficient) {
        bias[i] = *reinterpret_cast<const AlignedVector<T, VEC>*>(
            beta + g * Dg + v * VEC);
      }
    }
  }

  for (int64_t range = team / G; range < num_ranges;
       range += num_team_ranges) {
    const int64_t begin = range * range_size;
    const int64_t end = begin + range_size < N ? begin + range_size : N;
    T_ACC dgamma[ITEMS][VEC];
    T_ACC dbeta[ITEMS][VEC];
  #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
  #pragma unroll
      for (int k = 0; k < VEC; ++k) {
        dgamma[i][k] = T_ACC(0);
        dbeta[i][k] = T_ACC(0);
      }
    }

    TeamCursor<T> cursor(x, begin, L, G, Dg, g);
    for (int64_t tok0 = begin; tok0 < end; tok0 += kOutputTokens) {
      const int num_tokens =
          static_cast<int>(end - tok0 < kOutputTokens ? end - tok0
                                                      : kOutputTokens);
      // The sums of token tok0 + lane.
      T_ACC own_sums[3] = {T_ACC(0), T_ACC(0), T_ACC(0)};
      T_ACC own_rstd = T_ACC(0);
      int64_t own_idx = 0;
      bool own_pad = true;
      for (int u = 0; u < num_tokens; ++u) {
        const bool pad =
            padding_mask != nullptr && padding_mask[tok0 + u];
        const T_ACC r = cumrstd[cursor.idx];
        T_ACC sums[3] = {T_ACC(0), T_ACC(0), T_ACC(0)};
        if (!pad) {
          const T_ACC c = kMemoryEfficient ? T_ACC(0) : cummean[cursor.idx];
  #pragma unroll
          for (int i = 0; i < ITEMS; ++i) {
            const int64_t v = lane + int64_t(i) * width;
            if (v < num_vecs) {
              const int64_t offset = (cursor.row - x) + v * VEC;
              AccumulateInternalGradients<VEC, kMemoryEfficient>(
                  *reinterpret_cast<const AlignedVector<T, VEC>*>(x + offset),
                  *reinterpret_cast<const AlignedVector<T, VEC>*>(dy + offset),
                  w[i], bias[i], c, r, eps, sums, dgamma[i], dbeta[i]);
            }
          }
        }
        TeamAllReduceSum<WIDTH>(sums, mask, shm);
        if (lane == u) {
          own_sums[0] = sums[0];
          own_sums[1] = sums[1];
          own_sums[2] = sums[2];
          own_rstd = r;
          own_idx = cursor.idx;
          own_pad = pad;
        }
        cursor.Next();
      }

      if (lane < num_tokens) {
        float u_t = 0.0f;
        float v_t = 0.0f;
        const int64_t c_t = count[tok0 + lane];
        if (!own_pad && c_t > 0) {
          LocalGradients(static_cast<float>(own_rstd),
                         static_cast<float>(own_sums[2]),
                         static_cast<float>(own_sums[1]), c_t, log_beta1,
                         log_beta2, &u_t, &v_t);
        }
        group_mean[own_idx] = own_sums[0] / group_size;
        local_u[own_idx] = static_cast<T_ACC>(u_t);
        local_v[own_idx] = static_cast<T_ACC>(v_t);
      }
    }

  #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
      const int64_t v = lane + int64_t(i) * width;
      if (v < num_vecs) {
        const int64_t h = range * H + g * Dg + v * VEC;
  #pragma unroll
        for (int k = 0; k < VEC; ++k) {
          gamma_grad_partial[h + k] = dgamma[i][k];
          beta_grad_partial[h + k] = dbeta[i][k];
        }
      }
    }
  }
}

// Backward scan from the last timestep: element j is [B, G, L] index num_items - 1 - j and
// holds the map from the gradients flowing into the next timestep to those into its own, so
// its inclusive scan gives the latter; at the last timestep of a (batch, group) the map is
// constant, giving the gradients of the final state.
template <typename T_ACC, typename IndexT>
struct BwdScanProblem {
  using Index = IndexT;
  using Value = float4;
  using Op = Affine4Prod;
  // The local gradients and masks of the timesteps of a thread: [k + 1] those
  // of its item k, [0] those of the timestep after item 0.
  template <int ITEMS>
  struct Context {
    float u[ITEMS + 1];
    float v[ITEMS + 1];
    bool pad[ITEMS + 1];
    bool bos[ITEMS + 1];
  };

  IndexT num_items;
  at::cuda::detail::IntDivider<IndexT> L_div;
  at::cuda::detail::IntDivider<IndexT> G_div;
  const bool* bos_mask;
  const bool* padding_mask;
  const T_ACC* mean_grad;
  const T_ACC* var_grad;
  const T_ACC* local_u;
  const T_ACC* local_v;
  float beta1;
  float beta2;
  T_ACC* du;
  T_ACC* dv;
  T_ACC* prev_mean_grad;
  T_ACC* prev_var_grad;

  static __device__ __forceinline__ float4 Identity() {
    return IdentityAffine4();
  }

  __device__ __forceinline__ float4 Map(float u, float v, bool pad,
                                        bool bos) const {
    if (pad) {
      return IdentityAffine4();
    }
    return bos ? MakeAffine4(0.0f, 0.0f, 0.0f, 0.0f)
               : MakeAffine4(beta1 * u, beta1, beta2 * v, beta2);
  }

  template <int ITEMS>
  __device__ __forceinline__ void Load(IndexT first, Context<ITEMS>& context,
                                       float4 (&items)[ITEMS]) const {
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      items[k] = Identity();
    }
#pragma unroll
    for (int k = 0; k <= ITEMS; ++k) {
      context.u[k] = 0.0f;
      context.v[k] = 0.0f;
      context.pad[k] = false;
      context.bos[k] = false;
    }
    if (first >= num_items) {
      return;
    }
    // The timestep of item k is hi - k.
    const IndexT hi = num_items - 1 - first;
    T_ACC us[ITEMS];
    T_ACC vs[ITEMS];
    LoadItemsReversed(local_u, hi, us);
    LoadItemsReversed(local_v, hi, vs);
    const IndexT L = L_div.divisor;
    const IndexT G = G_div.divisor;
    // The (batch, group) seq = b * G + g and the timestep t of each item.
    const auto pos = L_div.divmod(hi);
    IndexT seq = pos.div;
    IndexT t = pos.mod;
    IndexT b = G_div.div(seq);
    IndexT g = seq - b * G;
    if (t + 1 < L) {
      const IndexT bt = b * L + t + 1;
      context.u[0] = static_cast<float>(local_u[hi + 1]);
      context.v[0] = static_cast<float>(local_v[hi + 1]);
      context.pad[0] = padding_mask != nullptr && padding_mask[bt];
      context.bos[0] = bos_mask != nullptr && bos_mask[bt];
    }
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      if (static_cast<IndexT>(k) <= hi) {
        const IndexT bt = b * L + t;
        context.u[k + 1] = static_cast<float>(us[k]);
        context.v[k + 1] = static_cast<float>(vs[k]);
        context.pad[k + 1] = padding_mask != nullptr && padding_mask[bt];
        context.bos[k + 1] = bos_mask != nullptr && bos_mask[bt];
        items[k] = t == L - 1
                       ? MakeAffine4(static_cast<float>(mean_grad[seq]), 0.0f,
                                     static_cast<float>(var_grad[seq]), 0.0f)
                       : Map(context.u[k], context.v[k], context.pad[k],
                             context.bos[k]);
        if (t == 0) {
          t = L - 1;
          --seq;
          if (g == 0) {
            g = G - 1;
            --b;
          } else {
            --g;
          }
        } else {
          --t;
        }
      }
    }
  }

  template <int ITEMS>
  __device__ __forceinline__ void Store(IndexT first,
                                        const Context<ITEMS>& context,
                                        const float4 (&states)[ITEMS]) const {
    if (first >= num_items) {
      return;
    }
    const IndexT hi = num_items - 1 - first;
    const IndexT L = L_div.divisor;
    const auto pos = L_div.divmod(hi);
    IndexT seq = pos.div;
    IndexT t = pos.mod;
    T_ACC dus[ITEMS];
    T_ACC dvs[ITEMS];
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      // The gradients flowing into the timestep of item k.
      const float u_in = states[k].x;
      const float v_in = states[k].z;
      if (context.pad[k + 1]) {
        dus[k] = T_ACC(0);
        dvs[k] = T_ACC(0);
      } else {
        dus[k] = static_cast<T_ACC>((1.0f - beta1) * (u_in + context.u[k + 1]));
        dvs[k] = static_cast<T_ACC>((1.0f - beta2) * (v_in + context.v[k + 1]));
      }
      if (static_cast<IndexT>(k) <= hi) {
        if (t == 0) {
          const float4 map = Map(context.u[k + 1], context.v[k + 1],
                                 context.pad[k + 1], context.bos[k + 1]);
          prev_mean_grad[seq] = static_cast<T_ACC>(ApplyAffineX(map, u_in));
          prev_var_grad[seq] = static_cast<T_ACC>(ApplyAffineY(map, v_in));
          t = L - 1;
          --seq;
        } else {
          --t;
        }
      }
    }
    StoreItemsReversed(du, hi, dus);
    StoreItemsReversed(dv, hi, dvs);
  }
};

// Same tiling as GroupTimestepDecayNormLookbackApplyFwdKernel. With
// kMemoryEfficient, x holds y and group_mean is the mean of x - cummean.
template <typename T, typename T_ACC, int VEC, int TT, bool kMemoryEfficient>
__global__ void GroupTimestepDecayNormLookbackBwdKernel(
    int64_t L, int64_t G, int64_t Dg, const T* __restrict__ dy,
    const T* __restrict__ x, const T_ACC* __restrict__ group_mean,
    const T_ACC* __restrict__ cumrstd, const T* __restrict__ gamma,
    const T* __restrict__ beta, float eps,
    const bool* __restrict__ padding_mask, const T_ACC* __restrict__ du,
    const T_ACC* __restrict__ dv, T* __restrict__ dx) {
  extern __shared__ unsigned char raw_shared[];
  T_ACC* mean_shared = reinterpret_cast<T_ACC*>(raw_shared);
  T_ACC* rstd_shared = mean_shared + G * TT;
  T_ACC* du_shared = rstd_shared + G * TT;
  T_ACC* dv_shared = du_shared + G * TT;

  const int64_t b = blockIdx.y;
  const int64_t t0 = int64_t(blockIdx.x) * TT;
  const int64_t H = G * Dg;
  const T_ACC coef = T_ACC(1) / static_cast<T_ACC>(Dg);

  for (int64_t k = threadIdx.x; k < G * TT; k += blockDim.x) {
    const int64_t t = t0 + k % TT;
    if (t < L) {
      const int64_t idx = GroupIndex(b, k / TT, t, G, L);
      mean_shared[k] = group_mean[idx];
      rstd_shared[k] = cumrstd[idx];
      du_shared[k] = du[idx];
      dv_shared[k] = dv[idx];
    }
  }
  __syncthreads();

  const int64_t num_group_vecs = Dg / VEC;
  for (int64_t j = threadIdx.x; j < H / VEC; j += blockDim.x) {
    const int64_t g = j / num_group_vecs;
    T_ACC w[VEC];
    T_ACC bias[VEC];
    LoadVector<VEC>(gamma + j * VEC, w);
    if constexpr (kMemoryEfficient) {
      LoadVector<VEC>(beta + j * VEC, bias);
    }
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) {
      const int64_t t = t0 + tt;
      if (t < L) {
        const int64_t offset = (b * L + t) * H + j * VEC;
        T_ACC out[VEC];
        if (padding_mask != nullptr && padding_mask[b * L + t]) {
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            out[k] = T_ACC(0);
          }
        } else {
          T_ACC dy_val[VEC];
          T_ACC x_val[VEC];
          LoadVector<VEC>(dy + offset, dy_val);
          LoadVector<VEC>(x + offset, x_val);
          const int64_t s = g * TT + tt;
          const T_ACC mean = mean_shared[s];
          const T_ACC rstd = rstd_shared[s];
          const T_ACC du_acc = du_shared[s];
          const T_ACC dv_acc = dv_shared[s];
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            const T_ACC x_k = kMemoryEfficient
                                  ? CenteredFromOutput(x_val[k], bias[k],
                                                       rstd * w[k], eps)
                                  : x_val[k];
            out[k] = dy_val[k] * rstd * w[k] +
                     coef * (du_acc + T_ACC(2) * dv_acc * (x_k - mean));
          }
        }
        StoreVector<VEC>(dx + offset, out);
      }
    }
  }
}

// kMemoryEfficient: x holds y.
template <typename T, typename T_ACC, bool kMemoryEfficient>
__global__ void GroupGammaBetaLookbackBwdSmallKernel(
    int64_t B, int64_t L, int64_t H, int64_t G, int64_t Dg,
    const T* __restrict__ dy, const T* __restrict__ x,
    const T_ACC* __restrict__ cummean, const T_ACC* __restrict__ cumrstd,
    const T* __restrict__ gamma, const T* __restrict__ beta, float eps,
    const bool* __restrict__ padding_mask, T* __restrict__ dgamma,
    T* __restrict__ dbeta) {
  const int64_t h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H) {
    return;
  }
  const int64_t g = h / Dg;
  const T_ACC w = kMemoryEfficient ? static_cast<T_ACC>(gamma[h]) : T_ACC(0);
  const T_ACC bias =
      kMemoryEfficient ? static_cast<T_ACC>(beta[h]) : T_ACC(0);

  T_ACC dg = T_ACC(0);
  T_ACC db = T_ACC(0);
  const int64_t outer = B * L;
  for (int64_t idx = 0; idx < outer; ++idx) {
    const int64_t b = idx / L;
    const int64_t t = idx % L;
    const bool pad = padding_mask != nullptr && padding_mask[b * L + t];
    if (!pad) {
      const T_ACC dy_acc = static_cast<T_ACC>(dy[XIndex(b, t, h, L, H)]);
      const T_ACC x_acc = static_cast<T_ACC>(x[XIndex(b, t, h, L, H)]);
      const T_ACC rstd = cumrstd[GroupIndex(b, g, t, G, L)];
      const T_ACC x_c =
          kMemoryEfficient
              ? CenteredFromOutput(x_acc, bias, rstd * w, eps)
              : x_acc - cummean[GroupIndex(b, g, t, G, L)];
      dg += dy_acc * x_c * rstd;
      db += dy_acc;
    }
  }
  dgamma[h] = static_cast<T>(dg);
  dbeta[h] = static_cast<T>(db);
}

// kMemoryEfficient: x holds y.
template <typename T, typename T_ACC, bool kMemoryEfficient>
__global__ void GroupGammaBetaLookbackBwdLargeKernel(
    int64_t B, int64_t L, int64_t H, int64_t G, int64_t Dg,
    const T* __restrict__ dy, const T* __restrict__ x,
    const T_ACC* __restrict__ cummean, const T_ACC* __restrict__ cumrstd,
    const T* __restrict__ gamma, const T* __restrict__ beta, float eps,
    const bool* __restrict__ padding_mask, T* __restrict__ dgamma,
    T* __restrict__ dbeta) {
  __shared__ T_ACC dgamma_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC dbeta_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];

  const int64_t h = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t g = h / Dg;
  const int64_t outer = B * L;
  const T_ACC w = kMemoryEfficient && h < H ? static_cast<T_ACC>(gamma[h])
                                            : T_ACC(0);
  const T_ACC bias =
      kMemoryEfficient && h < H ? static_cast<T_ACC>(beta[h]) : T_ACC(0);

  T_ACC dg = T_ACC(0);
  T_ACC db = T_ACC(0);
  for (int64_t idx = threadIdx.y; idx < outer; idx += blockDim.y) {
    const int64_t b = idx / L;
    const int64_t t = idx - b * L;
    const bool pad = padding_mask != nullptr && padding_mask[b * L + t];
    if (!pad && h < H) {
      const int64_t row_offset = idx * H;
      const T_ACC dy_acc = static_cast<T_ACC>(dy[row_offset + h]);
      const T_ACC x_acc = static_cast<T_ACC>(x[row_offset + h]);
      const int64_t group_idx = GroupIndex(b, g, t, G, L);
      const T_ACC rstd = cumrstd[group_idx];
      const T_ACC x_c = kMemoryEfficient
                            ? CenteredFromOutput(x_acc, bias, rstd * w, eps)
                            : x_acc - cummean[group_idx];
      dg += dy_acc * x_c * rstd;
      db += dy_acc;
    }
  }

  dgamma_shared[threadIdx.x][threadIdx.y] = dg;
  dbeta_shared[threadIdx.x][threadIdx.y] = db;
  __syncthreads();

  T_ACC dg_reduce = dgamma_shared[threadIdx.y][threadIdx.x];
  T_ACC db_reduce = dbeta_shared[threadIdx.y][threadIdx.x];
  dg_reduce = reduce::WarpReduce(dg_reduce);
  db_reduce = reduce::WarpReduce(db_reduce);

  if (threadIdx.x == 0) {
    const int64_t out_h = blockIdx.x * blockDim.x + threadIdx.y;
    if (out_h < H) {
      dgamma[out_h] = static_cast<T>(dg_reduce);
      dbeta[out_h] = static_cast<T>(db_reduce);
    }
  }
}

// Sums the [num_rows, H] partials of RowwiseInternalGradientsTeamLookbackKernel
// over the rows, with a (kWarpSize, kWarpSize) block per kWarpSize columns.
template <typename T, typename T_ACC>
__global__ void GammaBetaGradReduceLookbackKernel(
    int64_t num_rows, int64_t H, const T_ACC* __restrict__ gamma_grad_partial,
    const T_ACC* __restrict__ beta_grad_partial, T* __restrict__ dgamma,
    T* __restrict__ dbeta) {
  __shared__ T_ACC dgamma_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC dbeta_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];

  const int64_t h = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  T_ACC dg = T_ACC(0);
  T_ACC db = T_ACC(0);
  if (h < H) {
    // Load kRows rows, then sum them in order.
    constexpr int kRows = 8;
    const int64_t stride = blockDim.y;
    int64_t row = threadIdx.y;
    for (; row + (kRows - 1) * stride < num_rows; row += kRows * stride) {
      T_ACC g[kRows];
      T_ACC b[kRows];
#pragma unroll
      for (int i = 0; i < kRows; ++i) {
        g[i] = gamma_grad_partial[(row + i * stride) * H + h];
        b[i] = beta_grad_partial[(row + i * stride) * H + h];
      }
#pragma unroll
      for (int i = 0; i < kRows; ++i) {
        dg += g[i];
        db += b[i];
      }
    }
    for (; row < num_rows; row += stride) {
      dg += gamma_grad_partial[row * H + h];
      db += beta_grad_partial[row * H + h];
    }
  }

  dgamma_shared[threadIdx.x][threadIdx.y] = dg;
  dbeta_shared[threadIdx.x][threadIdx.y] = db;
  __syncthreads();

  T_ACC dg_reduce = dgamma_shared[threadIdx.y][threadIdx.x];
  T_ACC db_reduce = dbeta_shared[threadIdx.y][threadIdx.x];
  dg_reduce = reduce::WarpReduce(dg_reduce);
  db_reduce = reduce::WarpReduce(db_reduce);

  if (threadIdx.x == 0) {
    const int64_t out_h = int64_t(blockIdx.x) * blockDim.x + threadIdx.y;
    if (out_h < H) {
      dgamma[out_h] = static_cast<T>(dg_reduce);
      dbeta[out_h] = static_cast<T>(db_reduce);
    }
  }
}

int64_t NextPowerOfTwo(int64_t n) {
  int64_t p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

// Elements per load: as many as fill 16 bytes if the group size and all the
// pointers allow it, else one.
template <typename T>
int VectorSize(int64_t Dg, std::initializer_list<const void*> ptrs) {
  constexpr int kMaxVec = 16 / sizeof(T);
  if (Dg % kMaxVec != 0) {
    return 1;
  }
  for (const void* ptr : ptrs) {
    if (reinterpret_cast<std::uintptr_t>(ptr) % 16 != 0) {
      return 1;
    }
  }
  return kMaxVec;
}

struct TeamConfig {
  int64_t width;  // threads per (token, group)
  int items;      // loads per thread
  bool block_team;
};

// The narrowest team that holds a group of num_vecs loads with at most
// max_items loads per thread. False if that would take more than a block.
bool MakeTeamConfig(int64_t num_vecs, int max_items, TeamConfig* config) {
  if (num_vecs <= cuda_utils::kWarpSize) {
    *config = {NextPowerOfTwo(num_vecs), 1, false};
    return true;
  }
  if (num_vecs <= cuda_utils::kWarpSize * max_items) {
    const int items = static_cast<int>(
        NextPowerOfTwo(utils::DivUp(num_vecs, cuda_utils::kWarpSize)));
    *config = {cuda_utils::kWarpSize, items, false};
    return true;
  }
  *config = {NextPowerOfTwo(utils::DivUp(num_vecs, int64_t(max_items))),
             max_items, true};
  return config->width <= cuda_utils::kCUDAMaxNumThreadsPerBlock;
}

// The number of (range, group) teams of kernel the GPU holds at once, over
// the groups.
template <class KernelFunc>
int64_t ResidentRanges(KernelFunc kernel, int64_t block_size,
                       int64_t threads_per_range) {
  int blocks_per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, static_cast<int>(block_size), 0));
  const int64_t max_threads =
      int64_t(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) *
      std::max(blocks_per_sm, 1) * block_size;
  return std::max<int64_t>(max_threads / threads_per_range, 1);
}

// Splits the N tokens into num_ranges contiguous ranges, at most, and returns
// how many it takes.
int64_t SplitTokens(int64_t N, int64_t num_ranges, int64_t* range_size) {
  *range_size = utils::DivUp(N, std::clamp<int64_t>(num_ranges, 1, N));
  return utils::DivUp(N, *range_size);
}

// Token ranges for one team per (range, group), sized by resident team capacity.
template <class KernelFunc>
int64_t TokenRanges(KernelFunc kernel, int64_t block_size, int64_t N,
                    int64_t threads_per_range, int64_t* range_size) {
  return SplitTokens(
      N, ResidentRanges(kernel, block_size, threads_per_range), range_size);
}

// Calls launch.template operator()<ITEMS, WIDTH>() for the config, WIDTH being
// 0 for a block team.
template <int kMaxItems, class Launch>
void DispatchTeamConfig(const TeamConfig& config, Launch&& launch) {
  if (config.block_team) {
    launch.template operator()<kMaxItems, 0>();
  } else if (config.items > 1) {
    if constexpr (kMaxItems >= 4) {
      if (config.items == 4) {
        launch.template operator()<4, cuda_utils::kWarpSize>();
        return;
      }
    }
    launch.template operator()<2, cuda_utils::kWarpSize>();
  } else {
    switch (config.width) {
      case 1:
        launch.template operator()<1, 1>();
        break;
      case 2:
        launch.template operator()<1, 2>();
        break;
      case 4:
        launch.template operator()<1, 4>();
        break;
      case 8:
        launch.template operator()<1, 8>();
        break;
      case 16:
        launch.template operator()<1, 16>();
        break;
      default:
        launch.template operator()<1, cuda_utils::kWarpSize>();
        break;
    }
  }
}

template <int WIDTH>
int64_t TeamBlockThreads(int64_t width) {
  return WIDTH == 0 ? width : kLookbackTeamBlockSize;
}

template <int WIDTH, class KernelFunc, class... Args>
void LaunchTeams(KernelFunc kernel, int64_t num_teams, int64_t width,
                 cudaStream_t cuda_stream, Args... args) {
  const int64_t block_size = TeamBlockThreads<WIDTH>(width);
  const int64_t num_blocks = utils::DivUp(num_teams * width, block_size);
  cuda_utils::LaunchKernel(kernel, dim3(num_blocks), dim3(block_size), 0,
                           cuda_stream, args...);
}

template <typename T, typename T_ACC, int VEC>
bool LaunchRowwiseMomentsTeams(int64_t N, int64_t L, int64_t G, int64_t Dg,
                               const T* x, const bool* padding_mask,
                               T_ACC* group_mean, T_ACC* group_var,
                               cudaStream_t cuda_stream) {
  TeamConfig config;
  if (!MakeTeamConfig(Dg / VEC, 4, &config)) {
    return false;
  }
  DispatchTeamConfig<4>(config, [&]<int ITEMS, int WIDTH>() {
    const auto kernel =
        RowwiseMomentsTeamLookbackKernel<T, T_ACC, VEC, ITEMS, WIDTH>;
    int64_t range_size = 0;
    const int64_t num_ranges =
        TokenRanges(kernel, TeamBlockThreads<WIDTH>(config.width), N,
                    G * config.width, &range_size);
    LaunchTeams<WIDTH>(kernel, G * num_ranges, config.width, cuda_stream, N,
                       L, G, Dg, num_ranges, range_size, x, padding_mask,
                       group_mean, group_var);
  });
  return true;
}

// False if a group is too large for the team kernel.
template <typename T, typename T_ACC>
bool LaunchRowwiseMoments(int64_t N, int64_t L, int64_t G, int64_t Dg,
                          const T* x, const bool* padding_mask,
                          T_ACC* group_mean, T_ACC* group_var,
                          cudaStream_t cuda_stream) {
  constexpr int kMaxVec = 16 / sizeof(T);
  if (VectorSize<T>(Dg, {x}) == kMaxVec) {
    return LaunchRowwiseMomentsTeams<T, T_ACC, kMaxVec>(
        N, L, G, Dg, x, padding_mask, group_mean, group_var, cuda_stream);
  }
  return LaunchRowwiseMomentsTeams<T, T_ACC, 1>(
      N, L, G, Dg, x, padding_mask, group_mean, group_var, cuda_stream);
}

template <typename T, typename T_ACC, int VEC, bool kMemoryEfficient>
bool LaunchRowwiseInternalGradientsTeams(
    int64_t N, int64_t L, int64_t G, int64_t Dg, float beta1, float beta2,
    float eps, const T* dy, const T* x, const T* gamma, const T* beta,
    const T_ACC* cummean, const T_ACC* cumrstd, const int64_t* count,
    const bool* padding_mask,
    T_ACC* group_mean, T_ACC* local_u, T_ACC* local_v, T* gamma_grad,
    T* beta_grad, const at::TensorOptions& acc_options,
    cudaStream_t cuda_stream) {
  TeamConfig config;
  if (!MakeTeamConfig(Dg / VEC, 2, &config)) {
    return false;
  }
  const int64_t H = G * Dg;
  DispatchTeamConfig<2>(config, [&]<int ITEMS, int WIDTH>() {
    const auto kernel =
        RowwiseInternalGradientsTeamLookbackKernel<T, T_ACC, VEC, ITEMS, WIDTH,
                                                   kMemoryEfficient>;
    // Resident teams iterate over the assigned (range, group) tasks.
    const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    int64_t range_size = 0;
    const int64_t num_ranges = SplitTokens(
        N,
        int64_t(prop->multiProcessorCount) * prop->maxThreadsPerMultiProcessor /
            (G * config.width),
        &range_size);
    const int64_t num_team_ranges = std::min(
        num_ranges, ResidentRanges(kernel, TeamBlockThreads<WIDTH>(config.width),
                                   G * config.width));
    const torch::Tensor gamma_grad_partial =
        torch::empty({num_ranges, H}, acc_options);
    const torch::Tensor beta_grad_partial =
        torch::empty({num_ranges, H}, acc_options);
    LaunchTeams<WIDTH>(kernel, G * num_team_ranges, config.width, cuda_stream,
                       N, L, G, Dg, num_ranges, range_size, num_team_ranges,
                       beta1, beta2, eps, dy, x,
                       gamma, beta, cummean, cumrstd, count, padding_mask,
                       group_mean, local_u, local_v,
                       gamma_grad_partial.data_ptr<T_ACC>(),
                       beta_grad_partial.data_ptr<T_ACC>());
    cuda_utils::LaunchKernel(
        GammaBetaGradReduceLookbackKernel<T, T_ACC>,
        dim3(utils::DivUp(H, cuda_utils::kWarpSize)),
        dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize), 0, cuda_stream,
        num_ranges, H, gamma_grad_partial.data_ptr<T_ACC>(),
        beta_grad_partial.data_ptr<T_ACC>(), gamma_grad, beta_grad);
  });
  return true;
}

// Also computes the gamma/beta gradients. False if a group is too large for
// the team kernel.
template <typename T, typename T_ACC, bool kMemoryEfficient>
bool LaunchRowwiseInternalGradients(
    int64_t N, int64_t L, int64_t G, int64_t Dg, float beta1, float beta2,
    float eps, const T* dy, const T* x, const T* gamma, const T* beta,
    const T_ACC* cummean, const T_ACC* cumrstd, const int64_t* count,
    const bool* padding_mask, T_ACC* group_mean, T_ACC* local_u,
    T_ACC* local_v, T* gamma_grad, T* beta_grad,
    const at::TensorOptions& acc_options, cudaStream_t cuda_stream) {
  constexpr int kMaxVec = 16 / sizeof(T);
  const int vec = kMemoryEfficient ? VectorSize<T>(Dg, {dy, x, gamma, beta})
                                   : VectorSize<T>(Dg, {dy, x, gamma});
  if (vec == kMaxVec) {
    return LaunchRowwiseInternalGradientsTeams<T, T_ACC, kMaxVec,
                                               kMemoryEfficient>(
        N, L, G, Dg, beta1, beta2, eps, dy, x, gamma, beta, cummean, cumrstd,
        count, padding_mask, group_mean, local_u, local_v, gamma_grad,
        beta_grad, acc_options, cuda_stream);
  }
  return LaunchRowwiseInternalGradientsTeams<T, T_ACC, 1, kMemoryEfficient>(
      N, L, G, Dg, beta1, beta2, eps, dy, x, gamma, beta, cummean, cumrstd,
      count, padding_mask, group_mean, local_u, local_v, gamma_grad,
      beta_grad, acc_options, cuda_stream);
}

// Grid, block and token tile of the elementwise kernels, which keep
// num_stats [G, TT] statistics in shared memory.
struct ElementwiseConfig {
  dim3 grid;
  dim3 block;
  int tokens;
  int64_t shm_size;
};

template <typename T_ACC>
ElementwiseConfig MakeElementwiseConfig(int64_t B, int64_t L, int64_t G,
                                        int64_t num_row_vecs, int num_stats) {
  const bool tile =
      L >= kLookbackElementwiseTokens &&
      num_stats * G * kLookbackElementwiseTokens * int64_t(sizeof(T_ACC)) <=
          cuda_utils::kMaxStaticSharedMemorySize;
  const int tokens = tile ? kLookbackElementwiseTokens : 1;
  // As few threads as cover the row in the same number of passes.
  const int64_t passes =
      utils::DivUp(num_row_vecs, int64_t(kLookbackMaxElementwiseBlockSize));
  const int64_t block_size =
      utils::DivUp(utils::DivUp(num_row_vecs, passes), cuda_utils::kWarpSize) *
      cuda_utils::kWarpSize;
  return {dim3(utils::DivUp(L, int64_t(tokens)), B), dim3(block_size), tokens,
          num_stats * G * tokens * int64_t(sizeof(T_ACC))};
}

template <typename T, typename T_ACC>
void LaunchApplyFwd(int64_t B, int64_t L, int64_t G, int64_t Dg, const T* x,
                    const T_ACC* cummean, const T_ACC* cumrstd,
                    const T* gamma, const T* beta, const bool* padding_mask,
                    T* y, cudaStream_t cuda_stream) {
  constexpr int kMaxVec = 16 / sizeof(T);
  const int vec = VectorSize<T>(Dg, {x, gamma, beta, y});
  const ElementwiseConfig config =
      MakeElementwiseConfig<T_ACC>(B, L, G, G * Dg / vec, 2);
#define LAUNCH_APPLY(VEC, TT)                                                \
  cuda_utils::LaunchKernel(                                                  \
      GroupTimestepDecayNormLookbackApplyFwdKernel<T, T_ACC, VEC, TT>,       \
      config.grid, config.block, config.shm_size, cuda_stream, L, G, Dg, x,  \
      cummean, cumrstd, gamma, beta, padding_mask, y)
  if (vec == kMaxVec) {
    if (config.tokens == kLookbackElementwiseTokens) {
      LAUNCH_APPLY(kMaxVec, kLookbackElementwiseTokens);
    } else {
      LAUNCH_APPLY(kMaxVec, 1);
    }
  } else {
    if (config.tokens == kLookbackElementwiseTokens) {
      LAUNCH_APPLY(1, kLookbackElementwiseTokens);
    } else {
      LAUNCH_APPLY(1, 1);
    }
  }
#undef LAUNCH_APPLY
}

template <typename T, typename T_ACC, bool kMemoryEfficient>
void LaunchBwd(int64_t B, int64_t L, int64_t G, int64_t Dg, const T* dy,
               const T* x, const T_ACC* group_mean, const T_ACC* cumrstd,
               const T* gamma, const T* beta, float eps,
               const bool* padding_mask, const T_ACC* du, const T_ACC* dv,
               T* dx, cudaStream_t cuda_stream) {
  constexpr int kMaxVec = 16 / sizeof(T);
  const int vec = kMemoryEfficient
                      ? VectorSize<T>(Dg, {dy, x, gamma, beta, dx})
                      : VectorSize<T>(Dg, {dy, x, gamma, dx});
  const ElementwiseConfig config =
      MakeElementwiseConfig<T_ACC>(B, L, G, G * Dg / vec, 4);
#define LAUNCH_BWD(VEC, TT)                                                  \
  cuda_utils::LaunchKernel(                                                  \
      GroupTimestepDecayNormLookbackBwdKernel<T, T_ACC, VEC, TT,             \
                                              kMemoryEfficient>,             \
      config.grid, config.block, config.shm_size, cuda_stream, L, G, Dg, dy, \
      x, group_mean, cumrstd, gamma, beta, eps, padding_mask, du, dv, dx)
  if (vec == kMaxVec) {
    if (config.tokens == kLookbackElementwiseTokens) {
      LAUNCH_BWD(kMaxVec, kLookbackElementwiseTokens);
    } else {
      LAUNCH_BWD(kMaxVec, 1);
    }
  } else {
    if (config.tokens == kLookbackElementwiseTokens) {
      LAUNCH_BWD(1, kLookbackElementwiseTokens);
    } else {
      LAUNCH_BWD(1, 1);
    }
  }
#undef LAUNCH_BWD
}

// The states of num_tiles scan tiles in temp, from PyTorch's allocator.
template <typename Value>
ScanTiles<Value> MakeScanTiles(int64_t num_tiles,
                               const at::TensorOptions& options,
                               cudaStream_t cuda_stream, torch::Tensor* temp) {
  // The aggregates, the inclusive prefixes and the tile counter, zeroed.
  const int64_t values_words = num_tiles * ScanTiles<Value>::kWords;
  const int64_t bytes = 2 * values_words * int64_t(sizeof(uint4)) + 16;
  *temp = torch::empty({bytes}, options.dtype(torch::kByte));
  uint4* base = static_cast<uint4*>(temp->data_ptr());
  C10_CUDA_CHECK(cudaMemsetAsync(base, 0, bytes, cuda_stream));
  return {base, base + values_words,
          reinterpret_cast<unsigned int*>(base + 2 * values_words)};
}

// Runs DecoupledScanKernel over num_items scan elements.
template <int BLOCK_THREADS, int ITEMS, class Problem>
void LaunchDecoupledScan(const Problem& problem, int64_t num_items,
                         const at::TensorOptions& options,
                         cudaStream_t cuda_stream) {
  using Value = typename Problem::Value;
  const int64_t num_tiles =
      utils::DivUp(num_items, int64_t(BLOCK_THREADS) * ITEMS);
  torch::Tensor temp;
  const ScanTiles<Value> tiles =
      MakeScanTiles<Value>(num_tiles, options, cuda_stream, &temp);
  cuda_utils::LaunchKernel(DecoupledScanKernel<Problem, BLOCK_THREADS, ITEMS>,
                           dim3(num_tiles), dim3(BLOCK_THREADS), 0,
                           cuda_stream, problem, tiles);
}

// Calls launch.template operator()<IndexT>() with the narrowest index type
// for n items.
template <class Launch>
void DispatchIndexType(int64_t n, Launch&& launch) {
  if (n <= std::numeric_limits<int32_t>::max()) {
    launch.template operator()<unsigned int>();
  } else {
    launch.template operator()<int64_t>();
  }
}


// Warps per group: at most 8 loads of x per thread in forward,
// and 4 loads each of x and dy in backward.
constexpr int kFusedFwdThreads(int w) { return w > 8 ? w / 8 : 1; }
constexpr int kFusedBwdThreads(int w) { return w > 8 ? w / 4 : (w > 1 ? 2 : 1); }

// The fused kernels take groups of 1 to 32 loads of 16 bytes.
constexpr int kMaxFusedLoads = cuda_utils::kWarpSize;

// Calls launch.template operator()<W>() for W loads per group.
template <class Launch>
void DispatchFusedLoads(int64_t w, Launch&& launch) {
  switch (w) {
    case 1:
      launch.template operator()<1>();
      break;
    case 2:
      launch.template operator()<2>();
      break;
    case 4:
      launch.template operator()<4>();
      break;
    case 8:
      launch.template operator()<8>();
      break;
    case 16:
      launch.template operator()<16>();
      break;
    default:
      launch.template operator()<32>();
      break;
  }
}

// False if the groups do not suit the fused forward.
template <typename T, typename T_ACC>
bool LaunchFusedFwd(int64_t B, int64_t L, int64_t G, int64_t Dg, const T* x,
                    const bool* bos_mask, const bool* padding_mask,
                    const int64_t* prev_count, const T_ACC* prev_mean,
                    const T_ACC* prev_var, const T* gamma, const T* beta,
                    float beta1, float beta2, float eps, T* y,
                    T_ACC* mean_out, T_ACC* var_out, T_ACC* cummean,
                    T_ACC* cumrstd, const at::TensorOptions& options,
                    cudaStream_t cuda_stream) {
  constexpr int kMaxVec = 16 / sizeof(T);
  if (VectorSize<T>(Dg, {x, gamma, beta, y}) != kMaxVec ||
      at::cuda::getCurrentDeviceProperties()->major < 8) {
    return false;
  }
  const int64_t W = Dg / kMaxVec;
  if (W > kMaxFusedLoads || (W & (W - 1)) != 0) {
    return false;
  }
  const int64_t num_chunks = utils::DivUp(L, int64_t(cuda_utils::kWarpSize));
  torch::Tensor temp;
  const ScanTiles<DecayState> tiles =
      MakeScanTiles<DecayState>(B * G * num_chunks, options, cuda_stream, &temp);
  const auto launch = [&]<int kW, int GT>() {
    const int64_t num_tiles = num_chunks * B * (G / GT);
    const auto kernel =
        GroupTimestepDecayNormLookbackFusedFwdKernel<T, T_ACC, kMaxVec, kW, GT,
                                                     kFusedFwdThreads(kW)>;
    constexpr int kThreads =
        GT * cuda_utils::kWarpSize * kFusedFwdThreads(kW);
    // Two tiles of kWarpSize timesteps for GT groups.
    constexpr int64_t kShmSize = 2 * cuda_utils::kWarpSize * GT * kW * 16;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize));
    int blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kernel, kThreads, kShmSize));
    const int64_t num_blocks = std::min<int64_t>(
        num_tiles,
        int64_t(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) *
            std::max(blocks_per_sm, 1));
    cuda_utils::LaunchKernel(
        kernel, dim3(num_blocks), dim3(kThreads), kShmSize, cuda_stream, B, L,
        G, num_chunks, num_tiles, x, bos_mask, padding_mask, prev_count,
        prev_mean, prev_var, gamma, beta, beta1, beta2, eps, tiles, y,
        mean_out, var_out, cummean, cumrstd);
  };
  const auto dispatch = [&]<int GT>() {
    DispatchFusedLoads(
        W, [&]<int kW>() { launch.template operator()<kW, GT>(); });
  };
  if (G % 2 == 0) {
    dispatch.template operator()<2>();
  } else {
    dispatch.template operator()<1>();
  }
  return true;
}

// TMA forward uses kTmaFwdWarps moment and normalizing warps (fewer for small groups),
// kTmaFwdLookBacks look-back sets, and kTmaStageBytes total staging memory.
// Larger tiles use 8 stages.
constexpr int kTmaFwdWarps = 4;
constexpr int kTmaFwdLookBacks = 2;
constexpr int kTmaStageBytes = 64 * 1024;
constexpr int TmaStages(int tile_bytes) {
  const int stages = kTmaStageBytes / tile_bytes;
  return stages < 8 ? 8 : (stages > 16 ? 16 : stages);
}
constexpr int TmaWarps(int warps, int row_threads) {
  return row_threads < warps ? row_threads : warps;
}

// cuTensorMapEncodeTiled of the driver, through the runtime.
PFN_cuTensorMapEncodeTiled_v12000 TensorMapEncodeTiled() {
  static const PFN_cuTensorMapEncodeTiled_v12000 encode = [] {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult result;
#if CUDART_VERSION >= 12050
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &result));
#else
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn,
                                           cudaEnableDefault, &result));
#endif
    TORCH_CHECK(result == cudaDriverEntryPointSuccess && fn != nullptr,
                "cuTensorMapEncodeTiled is not available");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
  }();
  return encode;
}

// The tensor map of x [B, L, H] in boxes of box_rows timesteps of `box`
// channels, which take a multiple of 16 bytes and are at most 256.
template <typename T>
CUtensorMap MakeTensorMap(const T* x, int64_t B, int64_t L, int64_t H,
                          int64_t box,
                          int64_t box_rows = cuda_utils::kWarpSize) {
  static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
  const CUtensorMapDataType type =
      sizeof(T) == 2   ? CU_TENSOR_MAP_DATA_TYPE_UINT16
      : sizeof(T) == 4 ? CU_TENSOR_MAP_DATA_TYPE_UINT32
                       : CU_TENSOR_MAP_DATA_TYPE_INT64;
  const cuuint64_t dims[3] = {cuuint64_t(H), cuuint64_t(L), cuuint64_t(B)};
  const cuuint64_t strides[2] = {cuuint64_t(H * sizeof(T)),
                                 cuuint64_t(L * H * sizeof(T))};
  const cuuint32_t box_dims[3] = {cuuint32_t(box), cuuint32_t(box_rows), 1};
  const cuuint32_t elem_strides[3] = {1, 1, 1};
  CUtensorMap map;
  const CUresult result = TensorMapEncodeTiled()(
      &map, type, 3, const_cast<T*>(x), dims, strides, box_dims, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ",
              static_cast<int>(result));
  return map;
}

// The tensor map of a [outer, inner] array in boxes of [box_outer, box_inner],
// box_inner taking a multiple of 16 bytes.
template <typename T>
CUtensorMap MakeTensorMap2d(const T* data, int64_t outer, int64_t inner,
                            int64_t box_outer, int64_t box_inner) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 4 || sizeof(T) == 8);
  const CUtensorMapDataType type =
      sizeof(T) == 1   ? CU_TENSOR_MAP_DATA_TYPE_UINT8
      : sizeof(T) == 4 ? CU_TENSOR_MAP_DATA_TYPE_UINT32
                       : CU_TENSOR_MAP_DATA_TYPE_INT64;
  const cuuint64_t dims[2] = {cuuint64_t(inner), cuuint64_t(outer)};
  const cuuint64_t strides[1] = {cuuint64_t(inner * sizeof(T))};
  const cuuint32_t box_dims[2] = {cuuint32_t(box_inner), cuuint32_t(box_outer)};
  const cuuint32_t elem_strides[2] = {1, 1};
  CUtensorMap map;
  const CUresult result = TensorMapEncodeTiled()(
      &map, type, 2, const_cast<T*>(data), dims, strides, box_dims,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ",
              static_cast<int>(result));
  return map;
}

// False if the GPU, the groups or padding do not suit the TMA forward.
template <typename T, typename T_ACC>
bool LaunchTmaFwd(int64_t B, int64_t L, int64_t G, int64_t Dg, const T* x,
                  const bool* bos_mask, const bool* padding_mask,
                  const int64_t* prev_count, const T_ACC* prev_mean,
                  const T_ACC* prev_var, const T* gamma, const T* beta,
                  float beta1, float beta2, float eps, T* y, T_ACC* mean_out,
                  T_ACC* var_out, T_ACC* cummean, T_ACC* cumrstd,
                  const at::TensorOptions& options, cudaStream_t cuda_stream) {
  constexpr int kMaxVec = 16 / sizeof(T);
  if (at::cuda::getCurrentDeviceProperties()->major < 9 ||
      padding_mask != nullptr ||
      VectorSize<T>(Dg, {x, gamma, beta, y}) != kMaxVec) {
    return false;
  }
  const int64_t W = Dg / kMaxVec;
  // One group per tile, spanning two chunks for groups of at most 8 loads, else one.
  const int64_t GT = 1;
  const int64_t KC = W <= 8 ? 2 : 1;
  const int64_t num_chunks = utils::DivUp(L, int64_t(cuda_utils::kWarpSize));
  // Boxes of at most 256 channels, 32-bit tile numbers and int32 box coordinates; short
  // sequences use the fused kernel.
  if (W > kMaxFusedLoads || (W & (W - 1)) != 0 || GT * Dg > 256 ||
      num_chunks < 2 || num_chunks * B * (G / GT) >= (int64_t(1) << 31) ||
      L > INT32_MAX || B > INT32_MAX || G * Dg > INT32_MAX) {
    return false;
  }
  torch::Tensor temp;
  const ScanTiles<DecayState> tiles =
      MakeScanTiles<DecayState>(B * G * num_chunks, options, cuda_stream, &temp);
  const CUtensorMap x_map = MakeTensorMap(x, B, L, G * Dg, GT * Dg,
                                         KC * cuda_utils::kWarpSize);
  // The bos flags of the tiles by TMA too, if their rows of [B, L] are
  // 16-byte aligned; x_map stands in for bos_map otherwise.
  const bool bos_tma = bos_mask != nullptr && L % 16 == 0 &&
                       reinterpret_cast<uintptr_t>(bos_mask) % 16 == 0;
  const CUtensorMap bos_map =
      bos_tma ? MakeTensorMap2d(bos_mask, B, L, 1, KC * cuda_utils::kWarpSize)
              : x_map;
  const auto launch = [&]<int kW, int GT, int KC>() {
    constexpr int NW = TmaWarps(kTmaFwdWarps, GT * kW);
    constexpr int NL = kTmaFwdLookBacks;
    constexpr int kStageBytes =
        TmaFwdStageBytes(KC * cuda_utils::kWarpSize, GT * kW * 16);
    constexpr int S = TmaStages(kStageBytes);
    const int64_t num_tiles =
        utils::DivUp(num_chunks, int64_t(KC)) * B * (G / GT);
    const auto kernel =
        GroupTimestepDecayNormLookbackTmaFwdKernel<T, T_ACC, kMaxVec, kW, GT,
                                                   KC, NL, NW, NW, S>;
    constexpr int kThreads =
        ((KC + NL) * GT + 2 * NW + 2) * cuda_utils::kWarpSize;
    constexpr int64_t kShmSize = S * kStageBytes;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize));
    int blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kernel, kThreads, kShmSize));
    const int64_t num_blocks = std::min<int64_t>(
        num_tiles,
        int64_t(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) *
            std::max(blocks_per_sm, 1));
    cuda_utils::LaunchKernel(
        kernel, dim3(num_blocks), dim3(kThreads), kShmSize, cuda_stream, B, L,
        G, num_chunks, num_tiles, x_map, bos_map, bos_tma, bos_mask,
        prev_count, prev_mean, prev_var, gamma, beta, beta1, beta2, eps,
        tiles, y, mean_out, var_out, cummean, cumrstd);
  };
  DispatchFusedLoads(W, [&]<int kW>() {
    launch.template operator()<kW, 1, (kW <= 8 ? 2 : 1)>();
  });
  return true;
}

// TMA backward uses kTmaBwdSumSets sum-warp sets, kTmaBwdWarps dx warps (fewer for small
// groups), kTmaBwdLookBacks look-back sets and kTmaBwdStageBytes of stages, for at least
// kTmaBwdMinChains (batch, group) chains.
constexpr int kTmaBwdWarps = 4;
constexpr int kTmaBwdSumSets = 1;
constexpr int kTmaBwdLookBacks = 2;
constexpr int64_t kTmaBwdMinChains = 48;
constexpr int kTmaBwdStageBytes = 80 * 1024;
constexpr int TmaBwdStages(int stage_bytes) {
  const int stages = kTmaBwdStageBytes / stage_bytes;
  return stages < 2 ? 2 : (stages > 8 ? 8 : stages);
}

// False if the GPU, the groups, padding or the sequence length do not suit the
// TMA backward. Also computes the gamma/beta gradients.
template <typename T, typename T_ACC, bool kMemoryEfficient>
bool LaunchTmaBwd(int64_t B, int64_t L, int64_t G, int64_t Dg, const T* dy,
                  const T* x, const T* gamma, const T* beta,
                  const T_ACC* cummean, const T_ACC* cumrstd,
                  const int64_t* count, const bool* bos_mask,
                  const bool* padding_mask, const T_ACC* mean_grad,
                  const T_ACC* var_grad, float beta1, float beta2, float eps,
                  T* dx, T_ACC* prev_mean_grad, T_ACC* prev_var_grad,
                  T* gamma_grad, T* beta_grad,
                  const at::TensorOptions& options, cudaStream_t cuda_stream) {
  constexpr int kMaxVec = 16 / sizeof(T);
  const auto aligned = [](const void* ptr) {
    return reinterpret_cast<std::uintptr_t>(ptr) % 16 == 0;
  };
  const int vec = kMemoryEfficient
                      ? VectorSize<T>(Dg, {dy, x, gamma, beta, dx})
                      : VectorSize<T>(Dg, {dy, x, gamma, dx});
  if (at::cuda::getCurrentDeviceProperties()->major < 9 ||
      padding_mask != nullptr || vec != kMaxVec || !aligned(cummean) ||
      !aligned(cumrstd) || !aligned(count)) {
    return false;
  }
  const int64_t W = Dg / kMaxVec;
  const int64_t GT = G % 2 == 0 && W <= 8 ? 2 : 1;
  const int64_t H = G * Dg;
  const int64_t num_chunks = utils::DivUp(L, int64_t(cuda_utils::kWarpSize));
  // The statistics and counts are copied by rows of L, a multiple of 16 bytes.
  // Inputs with fewer than kTmaBwdMinChains chains use the fused kernel.
  if (W > kMaxFusedLoads || (W & (W - 1)) != 0 || GT * Dg > 256 ||
      L % 4 != 0 || num_chunks < 2 || B * (G / GT) < kTmaBwdMinChains ||
      num_chunks * B * (G / GT) >= (int64_t(1) << 31) || L > INT32_MAX ||
      B * G > INT32_MAX || G * Dg > INT32_MAX) {
    return false;
  }
  torch::Tensor temp;
  const ScanTiles<float4> tiles =
      MakeScanTiles<float4>(B * G * num_chunks, options, cuda_stream, &temp);
  const torch::Tensor gamma_grad_partial =
      torch::empty({B * num_chunks, H}, options);
  const torch::Tensor beta_grad_partial =
      torch::empty({B * num_chunks, H}, options);
  const CUtensorMap x_map = MakeTensorMap(x, B, L, H, GT * Dg);
  const CUtensorMap dy_map = MakeTensorMap(dy, B, L, H, GT * Dg);
  const CUtensorMap cummean_map =
      MakeTensorMap2d(cummean, B * G, L, GT, cuda_utils::kWarpSize);
  const CUtensorMap cumrstd_map =
      MakeTensorMap2d(cumrstd, B * G, L, GT, cuda_utils::kWarpSize);
  const CUtensorMap count_map =
      MakeTensorMap2d(count, B, L, 1, cuda_utils::kWarpSize);
  const auto launch = [&]<int kW, int kGT>() {
    constexpr int NM = kGT * kFusedBwdThreads(kW);
    constexpr int NN = TmaWarps(kTmaBwdWarps, kGT * kW);
    constexpr int NL = kTmaBwdLookBacks;
    constexpr int NS = kTmaBwdSumSets;
    // Matches the kernel's stage layout.
    constexpr int kStageBytes =
        (2 * cuda_utils::kWarpSize * kGT * kW * 16 +
         2 * kGT * cuda_utils::kWarpSize * int(sizeof(T_ACC)) +
         cuda_utils::kWarpSize * int(sizeof(int64_t)) +
         (kMemoryEfficient ? 2 : 1) * kGT * kW * 16 + 127) /
        128 * 128;
    constexpr int S = TmaBwdStages(kStageBytes);
    const int64_t num_tiles = num_chunks * B * (G / kGT);
    const auto kernel =
        GroupTimestepDecayNormLookbackTmaBwdKernel<T, T_ACC, kMaxVec, kW, kGT,
                                                   NL, NS, NM, NN, S,
                                                   kMemoryEfficient>;
    constexpr int kThreads =
        ((NL + 1) * kGT + NS * NM + NN + 2) * cuda_utils::kWarpSize;
    constexpr int64_t kShmSize = int64_t(S) * kStageBytes;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize));
    int blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kernel, kThreads, kShmSize));
    const int64_t num_blocks = std::min<int64_t>(
        num_tiles,
        int64_t(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) *
            std::max(blocks_per_sm, 1));
    cuda_utils::LaunchKernel(
        kernel, dim3(num_blocks), dim3(kThreads), kShmSize, cuda_stream, B, L,
        G, num_chunks, num_tiles, x_map, dy_map, cummean_map, cumrstd_map,
        count_map, gamma, beta, bos_mask, mean_grad, var_grad, beta1, beta2,
        eps, tiles, dx, prev_mean_grad, prev_var_grad,
        gamma_grad_partial.data_ptr<T_ACC>(),
        beta_grad_partial.data_ptr<T_ACC>());
  };
  const auto dispatch = [&]<int kGT>() {
    DispatchFusedLoads(W, [&]<int kW>() {
      if constexpr (kGT == 1 || kW <= 8) {
        launch.template operator()<kW, kGT>();
      }
    });
  };
  if (GT == 2) {
    dispatch.template operator()<2>();
  } else {
    dispatch.template operator()<1>();
  }
  cuda_utils::LaunchKernel(
      GammaBetaGradReduceLookbackKernel<T, T_ACC>,
      dim3(utils::DivUp(H, cuda_utils::kWarpSize)),
      dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize), 0, cuda_stream,
      B * num_chunks, H, gamma_grad_partial.data_ptr<T_ACC>(),
      beta_grad_partial.data_ptr<T_ACC>(), gamma_grad, beta_grad);
  return true;
}

// False if the groups do not suit the fused backward. Also computes the
// gamma/beta gradients.
template <typename T, typename T_ACC, bool kMemoryEfficient>
bool LaunchFusedBwd(int64_t B, int64_t L, int64_t G, int64_t Dg, const T* dy,
                    const T* x, const T* gamma, const T* beta,
                    const T_ACC* cummean, const T_ACC* cumrstd,
                    const int64_t* count, const bool* bos_mask,
                    const bool* padding_mask, const T_ACC* mean_grad,
                    const T_ACC* var_grad, float beta1, float beta2, float eps,
                    T* dx, T_ACC* prev_mean_grad, T_ACC* prev_var_grad,
                    T* gamma_grad, T* beta_grad,
                    const at::TensorOptions& options,
                    cudaStream_t cuda_stream) {
  constexpr int kMaxVec = 16 / sizeof(T);
  const int vec = kMemoryEfficient
                      ? VectorSize<T>(Dg, {dy, x, gamma, beta, dx})
                      : VectorSize<T>(Dg, {dy, x, gamma, dx});
  if (vec != kMaxVec || at::cuda::getCurrentDeviceProperties()->major < 8) {
    return false;
  }
  const int64_t W = Dg / kMaxVec;
  if (W > kMaxFusedLoads || (W & (W - 1)) != 0) {
    return false;
  }
  const int64_t H = G * Dg;
  const int64_t num_chunks = utils::DivUp(L, int64_t(cuda_utils::kWarpSize));
  torch::Tensor temp;
  const ScanTiles<float4> tiles =
      MakeScanTiles<float4>(B * G * num_chunks, options, cuda_stream, &temp);
  const torch::Tensor gamma_grad_partial =
      torch::empty({B * num_chunks, H}, options);
  const torch::Tensor beta_grad_partial =
      torch::empty({B * num_chunks, H}, options);
  const auto launch = [&]<int kW, int GT>() {
    const int64_t num_tiles = num_chunks * B * (G / GT);
    const auto kernel =
        GroupTimestepDecayNormLookbackFusedBwdKernel<T, T_ACC, kMaxVec, kW, GT,
                                                     kFusedBwdThreads(kW),
                                                     kMemoryEfficient>;
    constexpr int kThreads =
        GT * cuda_utils::kWarpSize * kFusedBwdThreads(kW);
    // x and dy of two tiles of kWarpSize timesteps for GT groups.
    constexpr int64_t kShmSize = 4 * cuda_utils::kWarpSize * GT * kW * 16;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize));
    int blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kernel, kThreads, kShmSize));
    const int64_t num_blocks = std::min<int64_t>(
        num_tiles,
        int64_t(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) *
            std::max(blocks_per_sm, 1));
    cuda_utils::LaunchKernel(
        kernel, dim3(num_blocks), dim3(kThreads), kShmSize, cuda_stream, B, L,
        G, num_chunks, num_tiles, dy, x, gamma, beta, cummean, cumrstd, count,
        bos_mask, padding_mask, mean_grad, var_grad, beta1, beta2, eps, tiles,
        dx, prev_mean_grad, prev_var_grad, gamma_grad_partial.data_ptr<T_ACC>(),
        beta_grad_partial.data_ptr<T_ACC>());
  };
  const auto dispatch = [&]<int GT>() {
    DispatchFusedLoads(
        W, [&]<int kW>() { launch.template operator()<kW, GT>(); });
  };
  if (G % 2 == 0) {
    dispatch.template operator()<2>();
  } else {
    dispatch.template operator()<1>();
  }
  cuda_utils::LaunchKernel(
      GammaBetaGradReduceLookbackKernel<T, T_ACC>,
      dim3(utils::DivUp(H, cuda_utils::kWarpSize)),
      dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize), 0, cuda_stream,
      B * num_chunks, H, gamma_grad_partial.data_ptr<T_ACC>(),
      beta_grad_partial.data_ptr<T_ACC>(), gamma_grad, beta_grad);
  return true;
}

template <typename T_ACC>
void LaunchFwdScan(int64_t B, int64_t L, int64_t G, const bool* bos_mask,
                   const bool* padding_mask, const int64_t* prev_count,
                   const T_ACC* prev_mean, const T_ACC* prev_var,
                   const T_ACC* group_mean, const T_ACC* group_var,
                   float beta1, float beta2, float eps, T_ACC* mean_out,
                   T_ACC* var_out, T_ACC* cummean, T_ACC* cumrstd,
                   const at::TensorOptions& options,
                   cudaStream_t cuda_stream) {
  const int64_t num_items = B * G * L;
  if (num_items == 0) {
    return;
  }
  DispatchIndexType(num_items, [&]<typename IndexT>() {
    using Divider = at::cuda::detail::IntDivider<IndexT>;
    const FwdScanProblem<T_ACC, IndexT> problem{
        static_cast<IndexT>(num_items),
        Divider(static_cast<IndexT>(L)),
        Divider(static_cast<IndexT>(G)),
        bos_mask,
        padding_mask,
        prev_count,
        prev_mean,
        prev_var,
        group_mean,
        group_var,
        beta1,
        beta2,
        eps,
        cummean,
        cumrstd,
        mean_out,
        var_out};
    LaunchDecoupledScan<kLookbackScanBlockSize, kLookbackScanItems>(
        problem, num_items, options, cuda_stream);
  });
}

template <typename T_ACC>
void LaunchBwdScan(int64_t B, int64_t L, int64_t G, const bool* bos_mask,
                   const bool* padding_mask, const T_ACC* mean_grad,
                   const T_ACC* var_grad, const T_ACC* local_u,
                   const T_ACC* local_v, float beta1, float beta2,
                   T_ACC* prev_mean_grad, T_ACC* prev_var_grad, T_ACC* du,
                   T_ACC* dv, const at::TensorOptions& options,
                   cudaStream_t cuda_stream) {
  const int64_t num_items = B * G * L;
  if (num_items == 0) {
    return;
  }
  DispatchIndexType(num_items, [&]<typename IndexT>() {
    using Divider = at::cuda::detail::IntDivider<IndexT>;
    const BwdScanProblem<T_ACC, IndexT> problem{
        static_cast<IndexT>(num_items),
        Divider(static_cast<IndexT>(L)),
        Divider(static_cast<IndexT>(G)),
        bos_mask,
        padding_mask,
        mean_grad,
        var_grad,
        local_u,
        local_v,
        beta1,
        beta2,
        du,
        dv,
        prev_mean_grad,
        prev_var_grad};
    LaunchDecoupledScan<kLookbackScanBlockSize, kLookbackScanItems>(
        problem, num_items, options, cuda_stream);
  });
}

// The final count [B], or for kWriteAll the count of every timestep [B, L].
template <bool kWriteAll>
void LaunchCount(int64_t B, int64_t L, const bool* bos_mask,
                 const bool* padding_mask, const int64_t* prev_count,
                 int64_t* count_out, cudaStream_t cuda_stream) {
  const auto launch = [&]<int BLOCK_THREADS>() {
    if constexpr (kWriteAll) {
      constexpr int kItems = BLOCK_THREADS == kLookbackLongCountBlockSize
                                 ? kLookbackLongCountItemsPerThread
                                 : kLookbackCountItemsPerThread;
      const int64_t num_chunks =
          utils::DivUp(L, int64_t(BLOCK_THREADS) * kItems);
      CountLookbackKernel<BLOCK_THREADS, kItems>
          <<<dim3(num_chunks, B), BLOCK_THREADS, 0, cuda_stream>>>(
              L, bos_mask, padding_mask, prev_count, count_out);
    } else {
      FinalCountLookbackKernel<BLOCK_THREADS>
          <<<B, BLOCK_THREADS, 0, cuda_stream>>>(L, bos_mask, padding_mask,
                                                 prev_count, count_out);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  if (L > kLookbackLongCountLength) {
    launch.template operator()<kLookbackLongCountBlockSize>();
  } else {
    launch.template operator()<kLookbackCountBlockSize>();
  }
}

template <typename T>
void GroupTimestepDecayNormLookbackCUDAFwdImpl(
    const torch::Tensor& x, const torch::Tensor& bos_mask,
    const torch::Tensor& prev_count, const torch::Tensor& prev_mean,
    const torch::Tensor& prev_var, const torch::Tensor& gamma,
    const torch::Tensor& beta, const torch::Tensor& padding_mask,
    int64_t num_groups, double beta1, double beta2, double eps,
    torch::Tensor& y,
    torch::Tensor& count, torch::Tensor& mean, torch::Tensor& var,
    torch::Tensor& cummean, torch::Tensor& cumrstd) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = x.size(0);
  const int64_t L = x.size(1);
  const int64_t H = x.size(2);
  const int64_t G = num_groups;
  const int64_t Dg = H / G;

  const T* x_data = x.data_ptr<T>();
  const bool* bos_mask_data = bos_mask.defined() ? bos_mask.data_ptr<bool>() : nullptr;
  const int64_t* prev_count_data = prev_count.data_ptr<int64_t>();
  const T_ACC* prev_mean_data = prev_mean.data_ptr<T_ACC>();
  const T_ACC* prev_var_data = prev_var.data_ptr<T_ACC>();
  const T* gamma_data = gamma.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* y_data = y.data_ptr<T>();
  int64_t* count_data = count.data_ptr<int64_t>();
  T_ACC* mean_data = mean.data_ptr<T_ACC>();
  T_ACC* var_data = var.data_ptr<T_ACC>();
  T_ACC* cummean_data = cummean.data_ptr<T_ACC>();
  T_ACC* cumrstd_data = cumrstd.data_ptr<T_ACC>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  LaunchCount<false>(B, L, bos_mask_data, padding_mask_data, prev_count_data,
                     count_data, cuda_stream);
  if (LaunchTmaFwd<T, T_ACC>(
          B, L, G, Dg, x_data, bos_mask_data, padding_mask_data,
          prev_count_data, prev_mean_data, prev_var_data, gamma_data,
          beta_data, static_cast<float>(beta1), static_cast<float>(beta2),
          static_cast<float>(eps), y_data, mean_data, var_data, cummean_data,
          cumrstd_data, cummean.options(), cuda_stream) ||
      LaunchFusedFwd<T, T_ACC>(
          B, L, G, Dg, x_data, bos_mask_data, padding_mask_data,
          prev_count_data, prev_mean_data, prev_var_data, gamma_data,
          beta_data, static_cast<float>(beta1), static_cast<float>(beta2),
          static_cast<float>(eps), y_data, mean_data, var_data, cummean_data,
          cumrstd_data, cummean.options(), cuda_stream)) {
    return;
  }
  // The moments of each (timestep, group), [B, G, L].
  const torch::Tensor group_mean = torch::empty_like(cummean);
  const torch::Tensor group_var = torch::empty_like(cummean);
  T_ACC* group_mean_data = group_mean.data_ptr<T_ACC>();
  T_ACC* group_var_data = group_var.data_ptr<T_ACC>();
  if (!LaunchRowwiseMoments<T, T_ACC>(B * L, L, G, Dg, x_data,
                                      padding_mask_data, group_mean_data,
                                      group_var_data, cuda_stream)) {
    cuda_utils::LaunchKernel(
        RowwiseMomentsLookbackKernel<T, T_ACC, kLookbackMomentBlockSize>,
        dim3(B * L, G), dim3(kLookbackMomentBlockSize), 0, cuda_stream, G, Dg,
        L, x_data, padding_mask_data, group_mean_data, group_var_data);
  }
  LaunchFwdScan<T_ACC>(B, L, G, bos_mask_data, padding_mask_data,
                       prev_count_data, prev_mean_data, prev_var_data,
                       group_mean_data, group_var_data,
                       static_cast<float>(beta1), static_cast<float>(beta2),
                       static_cast<float>(eps), mean_data, var_data,
                       cummean_data, cumrstd_data, cummean.options(),
                       cuda_stream);
  LaunchApplyFwd<T, T_ACC>(B, L, G, Dg, x_data, cummean_data, cumrstd_data,
                           gamma_data, beta_data, padding_mask_data, y_data,
                           cuda_stream);
}

// kMemoryEfficient: x holds y.
template <typename T, bool kMemoryEfficient>
void GroupTimestepDecayNormLookbackCUDABwdImpl(
    const torch::Tensor& y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& x,
    const torch::Tensor& prev_count, const torch::Tensor& bos_mask,
    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma, const torch::Tensor& beta,
    const torch::Tensor& padding_mask, torch::Tensor& count_array,
    int64_t num_groups, double beta1, double beta2, double eps,
    torch::Tensor& x_grad,
    torch::Tensor& prev_mean_grad, torch::Tensor& prev_var_grad,
    torch::Tensor& gamma_grad, torch::Tensor& beta_grad) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = x.size(0);
  const int64_t L = x.size(1);
  const int64_t H = x.size(2);
  const int64_t G = num_groups;
  const int64_t Dg = H / G;

  const T* y_grad_data = y_grad.data_ptr<T>();
  const T_ACC* mean_grad_data = mean_grad.data_ptr<T_ACC>();
  const T_ACC* var_grad_data = var_grad.data_ptr<T_ACC>();
  const T* x_data = x.data_ptr<T>();
  const int64_t* prev_count_data = prev_count.data_ptr<int64_t>();
  const bool* bos_mask_data = bos_mask.defined() ? bos_mask.data_ptr<bool>() : nullptr;
  const T_ACC* cummean_data = cummean.data_ptr<T_ACC>();
  const T_ACC* cumrstd_data = cumrstd.data_ptr<T_ACC>();
  const T* gamma_data = gamma.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* x_grad_data = x_grad.data_ptr<T>();
  T_ACC* prev_mean_grad_data = prev_mean_grad.data_ptr<T_ACC>();
  T_ACC* prev_var_grad_data = prev_var_grad.data_ptr<T_ACC>();
  T* gamma_grad_data = gamma_grad.data_ptr<T>();
  T* beta_grad_data = beta_grad.data_ptr<T>();

  int64_t* count_array_data = count_array.data_ptr<int64_t>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  const float beta1_f = static_cast<float>(beta1);
  const float beta2_f = static_cast<float>(beta2);
  const float eps_f = static_cast<float>(eps);
  LaunchCount<true>(B, L, bos_mask_data, padding_mask_data, prev_count_data,
                    count_array_data, cuda_stream);
  if (LaunchTmaBwd<T, T_ACC, kMemoryEfficient>(
          B, L, G, Dg, y_grad_data, x_data, gamma_data, beta_data,
          cummean_data, cumrstd_data, count_array_data, bos_mask_data,
          padding_mask_data, mean_grad_data, var_grad_data, beta1_f, beta2_f,
          eps_f, x_grad_data, prev_mean_grad_data, prev_var_grad_data,
          gamma_grad_data, beta_grad_data, cumrstd.options(), cuda_stream) ||
      LaunchFusedBwd<T, T_ACC, kMemoryEfficient>(
          B, L, G, Dg, y_grad_data, x_data, gamma_data, beta_data,
          cummean_data, cumrstd_data, count_array_data, bos_mask_data,
          padding_mask_data, mean_grad_data, var_grad_data, beta1_f, beta2_f,
          eps_f, x_grad_data, prev_mean_grad_data, prev_var_grad_data,
          gamma_grad_data, beta_grad_data, cumrstd.options(), cuda_stream)) {
    return;
  }
  // The mean of x, the local gradients and their scan over time, [B, G, L].
  const torch::Tensor group_mean = torch::empty_like(cumrstd);
  const torch::Tensor local_u = torch::empty_like(cumrstd);
  const torch::Tensor local_v = torch::empty_like(cumrstd);
  const torch::Tensor du = torch::empty_like(cumrstd);
  const torch::Tensor dv = torch::empty_like(cumrstd);
  T_ACC* group_mean_data = group_mean.data_ptr<T_ACC>();
  T_ACC* local_u_data = local_u.data_ptr<T_ACC>();
  T_ACC* local_v_data = local_v.data_ptr<T_ACC>();
  T_ACC* du_data = du.data_ptr<T_ACC>();
  T_ACC* dv_data = dv.data_ptr<T_ACC>();
  const bool param_grads_done =
      LaunchRowwiseInternalGradients<T, T_ACC, kMemoryEfficient>(
          B * L, L, G, Dg, beta1_f, beta2_f, eps_f, y_grad_data, x_data,
          gamma_data, beta_data, cummean_data, cumrstd_data, count_array_data,
          padding_mask_data, group_mean_data, local_u_data, local_v_data,
          gamma_grad_data, beta_grad_data, group_mean.options(), cuda_stream);
  if (!param_grads_done) {
    cuda_utils::LaunchKernel(
        RowwiseInternalGradientsLookbackKernel<T, T_ACC, kLookbackBwdBlockSize,
                                               kMemoryEfficient>,
        dim3(B, G), dim3(kLookbackBwdBlockSize), 0, cuda_stream, B, G, Dg, L,
        beta1_f, beta2_f, eps_f, y_grad_data, x_data, gamma_data, beta_data,
        cummean_data, cumrstd_data, count_array_data, padding_mask_data,
        group_mean_data, local_u_data, local_v_data);
  }
  LaunchBwdScan<T_ACC>(B, L, G, bos_mask_data, padding_mask_data,
                       mean_grad_data, var_grad_data, local_u_data,
                       local_v_data, beta1_f, beta2_f, prev_mean_grad_data,
                       prev_var_grad_data, du_data, dv_data,
                       group_mean.options(), cuda_stream);
  LaunchBwd<T, T_ACC, kMemoryEfficient>(
      B, L, G, Dg, y_grad_data, x_data, group_mean_data, cumrstd_data,
      gamma_data, beta_data, eps_f, padding_mask_data, du_data, dv_data,
      x_grad_data, cuda_stream);
  if (!param_grads_done && L < cuda_utils::kColwiseThreshold) {
    const int64_t param_blocks =
        utils::DivUp(H, int64_t(kParamGradBlockSize));
    GroupGammaBetaLookbackBwdSmallKernel<T, T_ACC, kMemoryEfficient>
        <<<param_blocks, kParamGradBlockSize, 0, cuda_stream>>>(
            B, L, H, G, Dg, y_grad_data, x_data, cummean_data, cumrstd_data,
            gamma_data, beta_data, eps_f, padding_mask_data, gamma_grad_data,
            beta_grad_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (!param_grads_done) {
    const int64_t param_blocks = utils::DivUp(H, int64_t(cuda_utils::kWarpSize));
    GroupGammaBetaLookbackBwdLargeKernel<T, T_ACC, kMemoryEfficient>
        <<<param_blocks, dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize), 0,
           cuda_stream>>>(B, L, H, G, Dg, y_grad_data, x_data, cummean_data,
                          cumrstd_data, gamma_data, beta_data, eps_f,
                          padding_mask_data, gamma_grad_data, beta_grad_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}


}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
GroupTimestepDecayNormLookbackCUDAFwd(
    const torch::Tensor& x, const c10::optional<torch::Tensor>& bos_mask,
    const torch::Tensor& prev_count, const torch::Tensor& prev_mean,
    const torch::Tensor& prev_var, const torch::Tensor& gamma,
    const torch::Tensor& beta, const c10::optional<torch::Tensor>& padding_mask,
    int64_t num_groups, double beta1, double beta2, double eps) {
  CheckTimestepDecayNormLookbackFwdInputs(
      x, bos_mask, prev_count, prev_mean, prev_var, gamma, beta, padding_mask,
      num_groups, beta1, beta2, eps);

  const int64_t B = x.size(0);
  const int64_t L = x.size(1);

  c10::MaybeOwned<torch::Tensor> bos_mask_maybe_owned =
      at::borrow_from_optional_tensor(bos_mask);
  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor y =
      torch::empty_like(x, x.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor count = torch::empty_like(
      prev_count, prev_count.options().memory_format(at::MemoryFormat::Contiguous));
  // Running mean/variance use the input accumulation dtype across calls.
  const auto acc_type = at::toAccumulateType(x.scalar_type(), true);
  const torch::Tensor prev_mean_acc = prev_mean.to(acc_type);
  const torch::Tensor prev_var_acc = prev_var.to(acc_type);
  torch::Tensor mean = torch::empty_like(
      prev_mean_acc,
      prev_mean_acc.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor var = torch::empty_like(
      prev_var_acc,
      prev_var_acc.options().memory_format(at::MemoryFormat::Contiguous));

  torch::Tensor cummean =
      torch::empty({B, num_groups, L},
                   x.options().dtype(acc_type).memory_format(
                       at::MemoryFormat::Contiguous));
  torch::Tensor cumrstd =
      torch::empty({B, num_groups, L},
                   x.options().dtype(acc_type).memory_format(
                       at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(),
      "GroupTimestepDecayNormLookbackCUDAFwd", [&]() {
        const torch::Tensor x_contig = *(x.expect_contiguous());
        const torch::Tensor bos_mask_contig =
            *(bos_mask_maybe_owned->expect_contiguous());
        const torch::Tensor prev_count_contig = *(prev_count.expect_contiguous());
        const torch::Tensor prev_mean_contig = *(prev_mean_acc.expect_contiguous());
        const torch::Tensor prev_var_contig = *(prev_var_acc.expect_contiguous());
        const torch::Tensor gamma_contig = *(gamma.expect_contiguous());
        const torch::Tensor beta_contig = *(beta.expect_contiguous());
        const torch::Tensor padding_mask_contig =
            *(padding_mask_maybe_owned->expect_contiguous());

        GroupTimestepDecayNormLookbackCUDAFwdImpl<scalar_t>(
            x_contig, bos_mask_contig, prev_count_contig, prev_mean_contig,
            prev_var_contig, gamma_contig, beta_contig, padding_mask_contig,
            num_groups, beta1, beta2, eps, y, count,
            mean, var, cummean, cumrstd);
      });

  return std::make_tuple(y, count, mean, var, cummean, cumrstd);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
GroupTimestepDecayNormLookbackCUDABwd(
    const torch::Tensor& y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& x_or_y,
    const torch::Tensor& prev_count, const c10::optional<torch::Tensor>& bos_mask,
    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma, const torch::Tensor& beta,
    const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups,
    double beta1, double beta2, double eps, bool memory_efficient) {
  CheckTimestepDecayNormLookbackBwdInputs(
      y_grad, mean_grad, var_grad, x_or_y, prev_count, bos_mask, cummean,
      cumrstd, gamma, beta, padding_mask, num_groups, beta1, beta2, eps);

  const int64_t B = x_or_y.size(0);
  const int64_t L = x_or_y.size(1);

  c10::MaybeOwned<torch::Tensor> bos_mask_maybe_owned =
      at::borrow_from_optional_tensor(bos_mask);
  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor x_grad = torch::empty_like(
      x_or_y, x_or_y.options().memory_format(at::MemoryFormat::Contiguous));
  // The state and its gradients are in the accumulate type, see
  // GroupTimestepDecayNormLookbackCUDAFwd.
  const auto acc_type = at::toAccumulateType(x_or_y.scalar_type(), true);
  const torch::Tensor mean_grad_acc = mean_grad.to(acc_type);
  const torch::Tensor var_grad_acc = var_grad.to(acc_type);
  torch::Tensor prev_mean_grad = torch::empty_like(
      mean_grad_acc,
      mean_grad_acc.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor prev_var_grad = torch::empty_like(
      var_grad_acc,
      var_grad_acc.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor gamma_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor beta_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));

  torch::Tensor count_array =
      torch::empty({B, L}, prev_count.options().memory_format(
                               at::MemoryFormat::Contiguous));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x_or_y.scalar_type(),
      "GroupTimestepDecayNormLookbackCUDABwd", [&]() {
        const auto impl =
            memory_efficient
                ? GroupTimestepDecayNormLookbackCUDABwdImpl<scalar_t, true>
                : GroupTimestepDecayNormLookbackCUDABwdImpl<scalar_t, false>;
        impl(*(y_grad.expect_contiguous()),
             *(mean_grad_acc.expect_contiguous()),
             *(var_grad_acc.expect_contiguous()), *(x_or_y.expect_contiguous()),
             *(prev_count.expect_contiguous()),
             *(bos_mask_maybe_owned->expect_contiguous()),
             *(cummean.expect_contiguous()), *(cumrstd.expect_contiguous()),
             *(gamma.expect_contiguous()), *(beta.expect_contiguous()),
             *(padding_mask_maybe_owned->expect_contiguous()), count_array,
             num_groups, beta1, beta2, eps, x_grad, prev_mean_grad,
             prev_var_grad, gamma_grad, beta_grad);
      });

  return std::make_tuple(x_grad, prev_mean_grad, prev_var_grad, gamma_grad,
                         beta_grad);
}

}  // namespace ops
}  // namespace xllm
