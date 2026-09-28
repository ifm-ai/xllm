#include <ATen/AccumulateType.h>
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/MaybeOwned.h>
#include <thrust/tuple.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>

#include <ATen/native/cuda/block_reduce.cuh>
#include <cstring>
#include <tuple>
#include <vector>

#include "cuda_utils.cuh"
#include "kahan.h"
#include "ops/timestep_norm.h"
#include "reduce.cuh"
#include "register_utils.cuh"
#include "welford_decay.h"

namespace xllm {
namespace ops {

namespace {

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      __VA_ARGS__;                              \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      __VA_ARGS__;                              \
    }                                           \
  }()

template <typename T>
__device__ T clamp_by_magnitude(T curr_gamma, double eps) {
  const T kMinGamma = T(eps);
  if (curr_gamma >= 0) {
    if (curr_gamma < kMinGamma) {
      return kMinGamma;
    } else {
      return curr_gamma;
    }
  } else {
    if (curr_gamma > -kMinGamma) {
      return -kMinGamma;
    } else {
      return curr_gamma;
    }
  }
}

template <typename T, typename T_ACC, int64_t kBlockSize, int64_t kNumThreads>
__global__ void RowwiseMomentsKernel(int64_t size, const T* __restrict__ x,
                                     T_ACC* __restrict__ mean, T_ACC* __restrict__ var) {
  constexpr int64_t kElementsPerThread = kBlockSize / kNumThreads;
  const int64_t r = blockIdx.x;
  __shared__ T_ACC shm[cuda_utils::kWarpSize];

  T_ACC x_acc[kElementsPerThread];
  register_utils::Load<T, T_ACC, kElementsPerThread>(x + r * size, size, size,
                                                     std::numeric_limits<T_ACC>::infinity(), x_acc);

  const T_ACC coef = T(1) / static_cast<T_ACC>(size);
  T_ACC m1 = T_ACC(0);
  T_ACC m2 = T_ACC(0);

#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    m1 += isinf(x_acc[i]) ? T_ACC(0) : x_acc[i];
  }
  m1 = reduce::BlockAllReduce(m1, shm);
  m1 *= coef;

#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    m2 += isinf(x_acc[i]) ? T_ACC(0) : cuda_utils::Square<T_ACC>(x_acc[i] - m1);
  }
  m2 = reduce::BlockReduce(m2, shm);

  if (threadIdx.x == 0) {
    mean[r] = m1;
    var[r] = m2 * coef;
  }
}

template <typename T, typename T_ACC>
__global__ void ColwiseCumMomentsSmallKernel(
    int64_t L, int64_t num_groups,
    const bool* __restrict__ bos_mask, const int64_t* __restrict__ prev_count,
    const T* __restrict__ prev_mean, const T* __restrict__ prev_var,
    const T_ACC* __restrict__ group_mean, const T_ACC* __restrict__ group_var,
    const bool* __restrict__ padding_mask, T_ACC beta1, T_ACC beta2, T_ACC eps,
    int64_t* __restrict__ count, T* __restrict__ mean, T* __restrict__ var,
    T_ACC* __restrict__ cummean, T_ACC* __restrict__ cumrstd) {
  const int64_t b = blockIdx.y;
  const int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= num_groups) {
    return;
  }

  const T_ACC* gu_ptr = group_mean + b * L * num_groups;
  const T_ACC* gv_ptr = group_var + b * L * num_groups;
  const bool* mask_ptr = padding_mask == nullptr ? nullptr : padding_mask + b * L;
  T_ACC* cummean_ptr = cummean + b * L * num_groups;
  T_ACC* cumrstd_ptr = cumrstd + b * L * num_groups;

  const int64_t m0 = prev_count[b];
  const T_ACC m1 = static_cast<T_ACC>(prev_mean[b * num_groups + g]);
  const T_ACC m2 = static_cast<T_ACC>(prev_var[b * num_groups + g]);
  utils::KahanWrapper<utils::WelfordDecayData<T_ACC>> m(utils::WelfordDecayData<T_ACC>{
      m0, pow(beta1, static_cast<T_ACC>(m0)), pow(beta2, static_cast<T_ACC>(m0)), m1, m2, false});

  for (int64_t i = 0; i < L; ++i) {
    const T_ACC gu = gu_ptr[i * num_groups + g];
    const T_ACC gv = gv_ptr[i * num_groups + g];
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    const utils::WelfordDecayData<T_ACC> cur = {
        1, beta1, beta2,
        gu, gv, bos_mask != nullptr && bos_mask[b * L + i]};
    const utils::KahanWrapper<utils::WelfordDecayData<T_ACC>> nxt = m + cur;
    m = mask ? m : nxt;
    const T_ACC rstd = c10::cuda::compat::rsqrt(m->var / (1 - m->beta2_pow) + eps);
    cummean_ptr[i * num_groups + g] = m->mean / (1 - m->beta1_pow);
    cumrstd_ptr[i * num_groups + g] = rstd;
  }

  if (g == 0) {
    count[b] = m->count;
  }
  mean[b * num_groups + g] = static_cast<T>(m->mean);
  var[b * num_groups + g] = static_cast<T>(m->var);
}

template <typename T, typename T_ACC>
__global__ void ColwiseCumMomentsLargeKernel(
    int64_t L, int64_t num_groups, int64_t chunk_size,
    const bool* __restrict__ bos_mask, const int64_t* __restrict__ prev_count,
    const T* __restrict__ prev_mean, const T* __restrict__ prev_var,
    const T_ACC* __restrict__ group_mean, const T_ACC* __restrict__ group_var,
    const bool* __restrict__ padding_mask, T_ACC beta1, T_ACC beta2, T_ACC eps,
    int64_t* __restrict__ count, T* __restrict__ mean, T* __restrict__ var,
    T_ACC* __restrict__ cummean, T_ACC* __restrict__ cumrstd) {
  using AlignedWelfordData =
      typename std::aligned_storage<sizeof(utils::WelfordDecayData<T_ACC>),
                                    alignof(utils::WelfordDecayData<T_ACC>)>::type;
  __shared__ AlignedWelfordData shm[cuda_utils::kWarpSize * cuda_utils::kWarpSize];
  utils::WelfordDecayData<T_ACC>* shm_ptr = reinterpret_cast<utils::WelfordDecayData<T_ACC>*>(shm);

  const int64_t b = blockIdx.y;
  const int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t l = threadIdx.y * chunk_size;
  const int64_t r = min(l + chunk_size, L);
  if (g >= num_groups) {
    return;
  }

  const T_ACC* gu_ptr = group_mean + b * L * num_groups;
  const T_ACC* gv_ptr = group_var + b * L * num_groups;
  const bool* mask_ptr = padding_mask == nullptr ? nullptr : padding_mask + b * L;
  T_ACC* cummean_ptr = cummean + b * L * num_groups;
  T_ACC* cumrstd_ptr = cumrstd + b * L * num_groups;

  utils::KahanWrapper<utils::WelfordDecayData<T_ACC>> m(
      utils::WelfordDecayData<T_ACC>{0, 1, 1, 0, 0, false});
  for (int64_t i = l; i < r; ++i) {
    const T_ACC gu = gu_ptr[i * num_groups + g];
    const T_ACC gv = gv_ptr[i * num_groups + g];
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    const utils::WelfordDecayData<T_ACC> cur = {
        1, beta1, beta2,
        gu, gv, bos_mask != nullptr && bos_mask[b * L + i]};
    const utils::KahanWrapper<utils::WelfordDecayData<T_ACC>> nxt = m + cur;
    m = mask ? m : nxt;
  }
  shm_ptr[threadIdx.y * cuda_utils::kWarpSize + threadIdx.x] = *m;
  __syncthreads();

  int64_t offset = 1;
#pragma unroll
  for (int64_t d = cuda_utils::kWarpSize >> 1; d > 0; d >>= 1) {
    if (threadIdx.y < d) {
      const int64_t ai = offset * (2 * threadIdx.y + 1) - 1;
      const int64_t bi = offset * (2 * threadIdx.y + 2) - 1;
      shm_ptr[bi * cuda_utils::kWarpSize + threadIdx.x] =
          shm_ptr[ai * cuda_utils::kWarpSize + threadIdx.x] +
          shm_ptr[bi * cuda_utils::kWarpSize + threadIdx.x];
    }
    offset <<= 1;
    __syncthreads();
  }
  if (threadIdx.y == 0) {
    shm_ptr[(cuda_utils::kWarpSize - 1) * cuda_utils::kWarpSize + threadIdx.x] =
        utils::WelfordDecayData<T_ACC>{prev_count[b], pow(beta1, static_cast<T_ACC>(prev_count[b])),
                                       pow(beta2, static_cast<T_ACC>(prev_count[b])),
                                       static_cast<T_ACC>(prev_mean[b * num_groups + g]),
                                       static_cast<T_ACC>(prev_var[b * num_groups + g]),
                                       false};
  }
  __syncthreads();
#pragma unroll
  for (int64_t d = 1; d < cuda_utils::kWarpSize; d <<= 1) {
    offset >>= 1;
    if (threadIdx.y < d) {
      const int64_t ai = offset * (2 * threadIdx.y + 1) - 1;
      const int64_t bi = offset * (2 * threadIdx.y + 2) - 1;
      const utils::WelfordDecayData<T_ACC> c = shm_ptr[ai * cuda_utils::kWarpSize + threadIdx.x];
      shm_ptr[ai * cuda_utils::kWarpSize + threadIdx.x] =
          shm_ptr[bi * cuda_utils::kWarpSize + threadIdx.x];
      shm_ptr[bi * cuda_utils::kWarpSize + threadIdx.x] += c;
    }
    __syncthreads();
  }

  m = utils::KahanWrapper<utils::WelfordDecayData<T_ACC>>(
      shm_ptr[threadIdx.y * cuda_utils::kWarpSize + threadIdx.x]);
  for (int64_t i = l; i < r; ++i) {
    const T_ACC gu = gu_ptr[i * num_groups + g];
    const T_ACC gv = gv_ptr[i * num_groups + g];
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    const utils::WelfordDecayData<T_ACC> cur = {
        1, beta1, beta2,
        gu, gv, bos_mask != nullptr && bos_mask[b * L + i]};
    const utils::KahanWrapper<utils::WelfordDecayData<T_ACC>> nxt = m + cur;
    m = mask ? m : nxt;
    // m = mask ? m : m + cur;
    const T_ACC rstd = c10::cuda::compat::rsqrt(m->var / (1 - m->beta2_pow) + eps);
    cummean_ptr[i * num_groups + g] = m->mean / (1 - m->beta1_pow);
    cumrstd_ptr[i * num_groups + g] = rstd;
  }

  if (threadIdx.y == cuda_utils::kWarpSize - 1) {
    if (g == 0) {
      count[b] = m->count;
    }
    mean[b * num_groups + g] = static_cast<T>(m->mean);
    var[b * num_groups + g] = static_cast<T>(m->var);
  }
}

template <typename T, typename T_ACC>
__global__ void GroupTimestepDecayNormCUDAFwdKernel(
    int64_t L, int64_t H, int64_t num_groups, const T* __restrict__ x,
    const T_ACC* __restrict__ cummean, const T_ACC* __restrict__ cumrstd,
    const T* __restrict__ gamma, const T* __restrict__ beta, const bool* __restrict__ padding_mask,
    T* __restrict__ y) {
  extern __shared__ float shm[];

  const int64_t D = H / num_groups;
  const int64_t b = blockIdx.y;
  const int64_t l = blockIdx.x;

  T_ACC* u_shared = reinterpret_cast<T_ACC*>(shm);
  T_ACC* r_shared = u_shared + num_groups;

  const T* x_ptr = x + (b * L + l) * H;
  const T_ACC* cummean_ptr = cummean + (b * L + l) * num_groups;
  const T_ACC* cumrstd_ptr = cumrstd + (b * L + l) * num_groups;
  const bool mask = padding_mask != nullptr && padding_mask[b * L + l];
  T* y_ptr = y + (b * L + l) * H;

  if (mask) {
    for (int64_t i = threadIdx.x; i < H; i += blockDim.x) {
      y_ptr[i] = T(0);
    }
    return;
  }

  for (int64_t i = threadIdx.x; i < num_groups; i += blockDim.x) {
    u_shared[i] = cummean_ptr[i];
    r_shared[i] = cumrstd_ptr[i];
  }
  __syncthreads();

  for (int64_t i = threadIdx.x; i < H; i += blockDim.x) {
    const int64_t g = i / D;
    const T_ACC x_acc = static_cast<T_ACC>(x_ptr[i]);
    const T_ACC u = u_shared[g];
    const T_ACC r = r_shared[g];
    const T_ACC w_acc = static_cast<T_ACC>(gamma[i]);
    const T_ACC b_acc = static_cast<T_ACC>(beta[i]);
    y_ptr[i] = static_cast<T>((x_acc - u) * r * w_acc + b_acc);
  }
}

template <typename T, typename T_ACC, int64_t kBlockSize, int64_t kNumThreads, bool memory_efficient>
__global__ void RowwiseInternalGradientsKernel(
    int64_t num_groups, int64_t group_size, const T* __restrict__ y_grad,
    const T* __restrict__ x_or_y, const T_ACC* __restrict__ mean, const T_ACC* __restrict__ rstd,
    const T* __restrict__ gamma, const T* __restrict__ beta, T_ACC* __restrict__ group_mean,
    double eps, T_ACC* __restrict__ ds, T_ACC* __restrict__ db) {
  constexpr int64_t kElementsPerThread = kBlockSize / kNumThreads;
  const int64_t r = blockIdx.x;
  const int64_t g = blockIdx.y;
  extern __shared__ float shm[];

  T_ACC* dy_shared = reinterpret_cast<T_ACC*>(shm);
  T_ACC* w_shared = dy_shared + kBlockSize;
  T_ACC* b_shared = w_shared + kBlockSize;
  T_ACC* m1_shared = b_shared + kBlockSize;
  T_ACC* ds_shared = m1_shared + cuda_utils::kWarpSize;
  T_ACC* db_shared = ds_shared + cuda_utils::kWarpSize;
  T_ACC x_or_y_acc[kElementsPerThread];

  register_utils::Load<T, T_ACC, kElementsPerThread>(y_grad + (r * num_groups + g) * group_size, group_size, group_size, T_ACC(0), x_or_y_acc);

#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * kNumThreads + threadIdx.x;
    dy_shared[idx] = x_or_y_acc[i];
  }
  register_utils::Load<T, T_ACC, kElementsPerThread>(gamma + g * group_size, group_size, group_size, T_ACC(0), x_or_y_acc);

#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * kNumThreads + threadIdx.x;
    w_shared[idx] = x_or_y_acc[i];
  }
  if (memory_efficient) {
    register_utils::Load<T, T_ACC, kElementsPerThread>(beta + g * group_size, group_size, group_size, T_ACC(0), x_or_y_acc);

#pragma unroll
    for (int64_t i = 0; i < kElementsPerThread; ++i) {
      const int64_t idx = i * kNumThreads + threadIdx.x;
      b_shared[idx] = x_or_y_acc[i];
    }
  }
  __syncthreads();

  register_utils::Load<T, T_ACC, kElementsPerThread>(x_or_y + (r * num_groups + g) * group_size, group_size, group_size, T_ACC(0), x_or_y_acc);

  const T_ACC coef = T_ACC(1) / static_cast<T_ACC>(group_size);
  const T_ACC u = mean[r * num_groups + g];
  const T_ACC v = memory_efficient ? rstd[r * num_groups + g] : T_ACC(0);
  T_ACC sum1 = T_ACC(0);
  T_ACC sum2 = T_ACC(0);
  T_ACC sum3 = T_ACC(0);

#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * kNumThreads + threadIdx.x;
    const T_ACC dy_acc = dy_shared[idx];
    const T_ACC w_acc = w_shared[idx];
    const T_ACC b_acc = memory_efficient ? b_shared[idx] : T_ACC(0);
    const T_ACC x_minus_mean = !memory_efficient ? x_or_y_acc[i] - u : (x_or_y_acc[i] - b_acc) / clamp_by_magnitude(v * w_acc, eps);
    sum1 += !memory_efficient ? x_or_y_acc[i] : (idx < group_size ? x_minus_mean + u : T_ACC(0));
    sum2 += dy_acc * x_minus_mean * w_acc;
    sum3 += dy_acc * w_acc;
  }
  sum1 = reduce::BlockReduce(sum1, m1_shared);
  sum2 = reduce::BlockReduce(sum2, ds_shared);
  sum3 = reduce::BlockReduce(sum3, db_shared);
  if (threadIdx.x == 0) {
    group_mean[r * num_groups + g] = sum1 * coef;
    ds[r * num_groups + g] = sum2;
    db[r * num_groups + g] = sum3;
  }
}

template <typename T, typename T_ACC>
__global__ void ColwiseInternalGradientsSmallKernel(
    int64_t L, int64_t num_groups, double beta1, double beta2, const T* __restrict__ mean_grad,
    const T* __restrict__ var_grad, const int64_t* __restrict__ prev_count,
    const bool* __restrict__ bos_mask, const T_ACC* __restrict__ cumrstd,
    const bool* __restrict__ padding_mask, const T_ACC* ds, const T_ACC* db,
    T* __restrict__ prev_mean_grad, T* __restrict__ prev_var_grad, T_ACC* du, T_ACC* dv) {
  const int64_t b = blockIdx.y;
  const int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= num_groups) {
    return;
  }

  const T* mean_grad_ptr = mean_grad + b * num_groups;
  const T* var_grad_ptr = var_grad + b * num_groups;
  const T_ACC* rstd_ptr = cumrstd + b * L * num_groups;
  const bool* mask_ptr = padding_mask == nullptr ? nullptr : padding_mask + b * L;
  const T_ACC* ds_ptr = ds + b * L * num_groups;
  const T_ACC* db_ptr = db + b * L * num_groups;

  T* m1_grad_ptr = prev_mean_grad + b * num_groups;
  T* m2_grad_ptr = prev_var_grad + b * num_groups;
  T_ACC* du_ptr = du + b * L * num_groups;
  T_ACC* dv_ptr = dv + b * L * num_groups;

  __shared__ int32_t count[cuda_utils::kColwiseThreshold];
  if (threadIdx.x == 0) {
    int32_t c = static_cast<int32_t>(prev_count[b]);
    for (int64_t i = 0; i < L; ++i) {
      if (mask_ptr == nullptr || !mask_ptr[i]) {
        c = (bos_mask != nullptr && bos_mask[b * L + i]) ? 1 : c + 1;
      }
      count[i] = c;
    }
  }
  __syncthreads();

  T_ACC u_grad = static_cast<T_ACC>(mean_grad_ptr[g]);
  T_ACC v_grad = static_cast<T_ACC>(var_grad_ptr[g]);

  // TODO: Improve this.
  double log_beta1 = log(beta1);
  double log_beta2 = log(beta2);
  for (int64_t i = L - 1; i >= 0; --i) {
    const T_ACC r = rstd_ptr[i * num_groups + g];
    const bool mask = mask_ptr != nullptr && mask_ptr[i];

    const int64_t c = count[i];
    const double beta1_pow = exp(c * log_beta1);
    const double beta2_pow = exp(c * log_beta2);
    const T_ACC du = u_grad - r * db_ptr[i * num_groups + g] / (1 - beta1_pow);
    const T_ACC dv = v_grad - T_ACC(0.5) * cuda_utils::Cube<T_ACC>(r) * ds_ptr[i * num_groups + g] / (1 - beta2_pow);
    du_ptr[i * num_groups + g] = du * (1 - beta1);
    dv_ptr[i * num_groups + g] = dv * (1 - beta2);
    u_grad = mask ? u_grad : du * beta1;
    v_grad = mask ? v_grad : dv * beta2;
    if (!mask && bos_mask != nullptr && bos_mask[b * L + i]) {
      u_grad = T_ACC(0);
      v_grad = T_ACC(0);
    }
  }

  m1_grad_ptr[g] = static_cast<T>(u_grad);
  m2_grad_ptr[g] = static_cast<T>(v_grad);
}

template <typename T, typename T_ACC>
__global__ void ColwiseInternalGradientsLargeKernel(
    int64_t L, int64_t num_groups, double beta1, double beta2, int64_t chunk_size,
    const T* __restrict__ mean_grad, const T* __restrict__ var_grad,
    const int64_t* __restrict__ prev_count, const bool* __restrict__ bos_mask,
    const T_ACC* __restrict__ cumrstd, const bool* __restrict__ padding_mask, const T_ACC* ds,
    const T_ACC* db, T* __restrict__ prev_mean_grad, T* __restrict__ prev_var_grad, T_ACC* du,
    T_ACC* dv) {
  const int64_t kNumMicroChunk = 256;
  __shared__ int32_t count_shared[cuda_utils::kWarpSize / 2][kNumMicroChunk];
  __shared__ bool bos_exists_shared[cuda_utils::kWarpSize / 2];
  __shared__ double m1_shared[cuda_utils::kWarpSize / 2][cuda_utils::kWarpSize + 1];
  __shared__ double m2_shared[cuda_utils::kWarpSize / 2][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC du_shared[cuda_utils::kWarpSize / 2][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC dv_shared[cuda_utils::kWarpSize / 2][cuda_utils::kWarpSize + 1];

  const int64_t b = blockIdx.y;
  const int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t l = threadIdx.y * chunk_size;
  const int64_t r = min(l + chunk_size, L);
  if (g >= num_groups) {
    return;
  }

  if (threadIdx.x == 0) {
    int32_t c = 0;
    bool bos_exists = false;
    for (int64_t i = l; i < r; ++i) {
      const bool mask = bos_mask != nullptr && bos_mask[b * L + i];
      c = !mask ? c + 1 : 1;
      bos_exists |= mask;
    }
    count_shared[threadIdx.y][0] = c;
    bos_exists_shared[threadIdx.y] = bos_exists;
  }
  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    int32_t c = static_cast<int32_t>(prev_count[b]);
    bool bos_exists = false;
    for (int32_t i = 0; i < cuda_utils::kWarpSize / 2; ++i) {
      const int32_t count = count_shared[i][0];
      const bool mask = bos_exists_shared[i];
      count_shared[i][0] = c;
      c = !mask ? c + count : count;
      bos_exists |= mask;
    }
  }
  __syncthreads();

  int64_t micro_chunk_size = (r - l + kNumMicroChunk - 1) / kNumMicroChunk;
  if (threadIdx.x == 0) {
    int32_t c = static_cast<int32_t>(count_shared[threadIdx.y][0]);
    for (int64_t i = l; i < r; ++i) {
      if ((i - l) % micro_chunk_size == 0 && i != l) {
        int64_t chunk_id = (i - l) / micro_chunk_size;
        count_shared[threadIdx.y][chunk_id] = c;
      }
      const bool mask = bos_mask != nullptr && bos_mask[b * L + i];
      c = !mask ? c + 1 : 1;
    }
  }
  __syncthreads();

  const T* mean_grad_ptr = mean_grad + b * num_groups;
  const T* var_grad_ptr = var_grad + b * num_groups;
  const T_ACC* rstd_ptr = cumrstd + b * L * num_groups;
  const bool* mask_ptr = padding_mask == nullptr ? nullptr : padding_mask + b * L;
  const T_ACC* ds_ptr = ds + b * L * num_groups;
  const T_ACC* db_ptr = db + b * L * num_groups;

  T* m1_grad_ptr = prev_mean_grad + b * num_groups;
  T* m2_grad_ptr = prev_var_grad + b * num_groups;
  T_ACC* du_ptr = du + b * L * num_groups;
  T_ACC* dv_ptr = dv + b * L * num_groups;

  int64_t cnt = 0;
  T_ACC u_grad = T_ACC(0);
  T_ACC v_grad = T_ACC(0);
  double log_beta1 = log(beta1);
  double log_beta2 = log(beta2);
  for (int64_t i = r - 1; i >= l; --i) {
    int64_t chunk_id = (i - l) / micro_chunk_size;
    int64_t c = count_shared[threadIdx.y][chunk_id];
    for (int64_t j = chunk_id * micro_chunk_size + l; j <= i; ++j) {
      const bool mask = bos_mask != nullptr && bos_mask[b * L + j];
      c = !mask ? c + 1 : 1;
    }

    const T_ACC r = rstd_ptr[i * num_groups + g];
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    cnt += mask ? 0 : 1;

    const double beta1_pow = exp(c * log_beta1);
    const double beta2_pow = exp(c * log_beta2);
    const T_ACC du = u_grad - r * db_ptr[i * num_groups + g] / (1 - beta1_pow);
    const T_ACC dv = v_grad - T_ACC(0.5) * cuda_utils::Cube<T_ACC>(r) * ds_ptr[i * num_groups + g] / (1 - beta2_pow);
    u_grad = mask ? u_grad : du * beta1;
    v_grad = mask ? v_grad : dv * beta2;
    if (!mask && bos_mask != nullptr && bos_mask[b * L + i]) {
      u_grad = T_ACC(0);
      v_grad = T_ACC(0);
      cnt = 0;
    }
  }
  du_shared[threadIdx.y][threadIdx.x] = u_grad;
  dv_shared[threadIdx.y][threadIdx.x] = v_grad;
  m1_shared[threadIdx.y][threadIdx.x] = pow(beta1, cnt);
  m2_shared[threadIdx.y][threadIdx.x] = pow(beta2, cnt);
  __syncthreads();

  if (threadIdx.y == 0) {
    u_grad = static_cast<T_ACC>(mean_grad_ptr[g]);
    v_grad = static_cast<T_ACC>(var_grad_ptr[g]);
#pragma unroll
    for (int64_t i = cuda_utils::kWarpSize / 2 - 1; i >= 0; --i) {
      const T_ACC dux = du_shared[i][threadIdx.x];
      const T_ACC dvx = dv_shared[i][threadIdx.x];
      du_shared[i][threadIdx.x] = u_grad;
      dv_shared[i][threadIdx.x] = v_grad;
      if (!bos_exists_shared[i]) {
        u_grad = u_grad * m1_shared[i][threadIdx.x] + dux;
        v_grad = v_grad * m2_shared[i][threadIdx.x] + dvx;
      } else {
        u_grad = dux;
        v_grad = dvx;
      }
    }
  }
  __syncthreads();

  u_grad = du_shared[threadIdx.y][threadIdx.x];
  v_grad = dv_shared[threadIdx.y][threadIdx.x];
  for (int64_t i = r - 1; i >= l; --i) {
    int64_t chunk_id = (i - l) / micro_chunk_size;
    int64_t c = count_shared[threadIdx.y][chunk_id];
    for (int64_t j = chunk_id * micro_chunk_size + l; j <= i; ++j) {
      const bool mask = bos_mask != nullptr && bos_mask[b * L + j];
      c = !mask ? c + 1 : 1;
    }

    const T_ACC r = rstd_ptr[i * num_groups + g];
    const bool mask = mask_ptr != nullptr && mask_ptr[i];

    const double beta1_pow = exp(c * log_beta1);
    const double beta2_pow = exp(c * log_beta2);
    const T_ACC du = u_grad - r * db_ptr[i * num_groups + g] / (1 - beta1_pow);
    const T_ACC dv = v_grad - T_ACC(0.5) * cuda_utils::Cube<T_ACC>(r) * ds_ptr[i * num_groups + g] / (1 - beta2_pow);
    du_ptr[i * num_groups + g] = du * (1 - beta1);
    dv_ptr[i * num_groups + g] = dv * (1 - beta2);
    u_grad = mask ? u_grad : du * beta1;
    v_grad = mask ? v_grad : dv * beta2;
    if (!mask && bos_mask != nullptr && bos_mask[b * L + i]) {
      u_grad = T_ACC(0);
      v_grad = T_ACC(0);
    }
  }
  if (threadIdx.y == 0) {
    m1_grad_ptr[g] = static_cast<T>(u_grad);
    m2_grad_ptr[g] = static_cast<T>(v_grad);
  }
}

template <typename T, typename T_ACC, bool memory_efficient>
__global__ void GroupTimestepNormCUDABwdKernel(
    int64_t L, int64_t H, int64_t num_groups, const T* __restrict__ y_grad,
    const T* __restrict__ x_or_y, const T_ACC* __restrict__ group_mean,
    const T_ACC* __restrict__ cummean, const T_ACC* __restrict__ cumrstd,
    const T* __restrict__ gamma, const T* __restrict__ beta, const bool* __restrict__ padding_mask,
    const T_ACC* __restrict__ du, const T_ACC* __restrict__ dv, double eps,
    T* __restrict__ x_grad) {
  constexpr int64_t kMaxNumGroups = 256;
  assert(num_groups <= kMaxNumGroups);

  const int64_t D = H / num_groups;
  const int64_t b = blockIdx.y;
  const int64_t l = blockIdx.x;
  const T_ACC coef = T_ACC(1) / T_ACC(D);

  __shared__ T_ACC ux_shared[kMaxNumGroups];
  __shared__ T_ACC u_shared[kMaxNumGroups];
  __shared__ T_ACC r_shared[kMaxNumGroups];
  __shared__ T_ACC du_shared[kMaxNumGroups];
  __shared__ T_ACC dv_shared[kMaxNumGroups];

  const T* dy_ptr = y_grad + (b * L + l) * H;
  const T* x_or_y_ptr = x_or_y + (b * L + l) * H;
  const T_ACC* group_mean_ptr = group_mean + (b * L + l) * num_groups;
  const T_ACC* cummean_ptr = cummean + (b * L + l) * num_groups;
  const T_ACC* cumrstd_ptr = cumrstd + (b * L + l) * num_groups;
  const bool mask = padding_mask != nullptr && padding_mask[b * L + l];
  const T_ACC* du_ptr = du + (b * L + l) * num_groups;
  const T_ACC* dv_ptr = dv + (b * L + l) * num_groups;
  T* dx_ptr = x_grad + (b * L + l) * H;

  if (mask) {
    for (int64_t i = threadIdx.x; i < H; i += blockDim.x) {
      dx_ptr[i] = T(0);
    }
    return;
  }

  for (int64_t i = threadIdx.x; i < num_groups; i += blockDim.x) {
    ux_shared[i] = group_mean_ptr[i];
    if (memory_efficient) {
      u_shared[i] = cummean_ptr[i];
    }
    r_shared[i] = cumrstd_ptr[i];
    du_shared[i] = du_ptr[i];
    dv_shared[i] = dv_ptr[i];
  }
  __syncthreads();

  for (int64_t i = threadIdx.x; i < H; i += blockDim.x) {
    const int64_t g = i / D;
    const T_ACC dy_acc = static_cast<T_ACC>(dy_ptr[i]);
    const T_ACC x_or_y_acc = static_cast<T_ACC>(x_or_y_ptr[i]);
    const T_ACC ux = ux_shared[g];
    const T_ACC u = memory_efficient ? u_shared[g] : T_ACC(0);
    const T_ACC r = r_shared[g];
    const T_ACC w_acc = static_cast<T_ACC>(gamma[i]);
    const T_ACC b_acc = memory_efficient ? static_cast<T_ACC>(beta[i]) : T_ACC(0);
    const T_ACC dux = du_shared[g];
    const T_ACC dvx = dv_shared[g];
    const T_ACC x_acc = !memory_efficient ? x_or_y_acc : (x_or_y_acc - b_acc) / clamp_by_magnitude(r * w_acc, eps) + u;
    dx_ptr[i] = static_cast<T>(dy_acc * r * w_acc + coef * (dux + T_ACC(2) * dvx * (x_acc - ux)));
  }
}

template <typename T, typename T_ACC, bool memory_efficient>
__global__ void GroupGammaBetaCUDABwdSmallKernel(
    int64_t outer_size, int64_t inner_size, int64_t num_groups, const T* __restrict__ y_grad,
    const T* __restrict__ x_or_y, const T* __restrict__ gamma, const T* __restrict__ beta,
    const T_ACC* __restrict__ cummean, const T_ACC* __restrict__ cumrstd,
    const bool* __restrict__ padding_mask, double eps, T* __restrict__ w_grad,
    T* __restrict__ b_grad) {
  const int64_t D = inner_size / num_groups;
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t g = j / D;
  if (j >= inner_size) {
    return;
  }
  utils::KahanWrapper<T_ACC> sum1(T_ACC(0));
  utils::KahanWrapper<T_ACC> sum2(T_ACC(0));
  for (int64_t i = 0; i < outer_size; ++i) {
    const bool mask = padding_mask != nullptr && padding_mask[i];
    const T_ACC dy_acc = static_cast<T_ACC>(y_grad[i * inner_size + j]);
    const T_ACC x_or_y_acc = static_cast<T_ACC>(x_or_y[i * inner_size + j]);
    const T_ACC w_acc = memory_efficient ? static_cast<T_ACC>(gamma[j]) : T_ACC(0);
    const T_ACC b_acc = memory_efficient ? static_cast<T_ACC>(beta[j]) : T_ACC(0);
    const T_ACC u = cummean[i * num_groups + g];
    const T_ACC r = cumrstd[i * num_groups + g];
    const T_ACC x_minus_mean = !memory_efficient ? x_or_y_acc - u : (x_or_y_acc - b_acc) / clamp_by_magnitude(r * w_acc, eps);
    const utils::KahanWrapper<T_ACC> t1 = sum1 + dy_acc * x_minus_mean * r;
    const utils::KahanWrapper<T_ACC> t2 = sum2 + dy_acc;
    sum1 = mask ? sum1 : t1;
    sum2 = mask ? sum2 : t2;
  }
  w_grad[j] = static_cast<T>(*sum1);
  b_grad[j] = static_cast<T>(*sum2);
}

template <typename T, typename T_ACC, bool memory_efficient>
__global__ void GroupGammaBetaCUDABwdLargeKernel(
    int64_t outer_size, int64_t inner_size, int64_t num_groups, const T* __restrict__ y_grad,
    const T* __restrict__ x_or_y, const T* __restrict__ gamma, const T* __restrict__ beta,
    const T_ACC* __restrict__ cummean, const T_ACC* __restrict__ cumrstd,
    const bool* __restrict__ padding_mask, double eps, T* __restrict__ w_grad,
    T* __restrict__ b_grad) {
  __shared__ T_ACC ds_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC db_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];

  const int64_t D = inner_size / num_groups;
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t g = j / D;

  utils::KahanWrapper<T_ACC> sum1(T_ACC(0));
  utils::KahanWrapper<T_ACC> sum2(T_ACC(0));
  for (int64_t i = threadIdx.y; i < outer_size; i += blockDim.y) {
    const bool mask = padding_mask != nullptr && padding_mask[i];
    const T_ACC dy_acc = j < inner_size ? static_cast<T_ACC>(y_grad[i * inner_size + j]) : T_ACC(0);
    const T_ACC x_or_y_acc = j < inner_size ? static_cast<T_ACC>(x_or_y[i * inner_size + j]) : T_ACC(0);
    const T_ACC w_acc = memory_efficient && j < inner_size ? static_cast<T_ACC>(gamma[j]) : T_ACC(0);
    const T_ACC b_acc = memory_efficient && j < inner_size ? static_cast<T_ACC>(beta[j]) : T_ACC(0);
    const T_ACC u = g < num_groups ? cummean[i * num_groups + g] : T_ACC(0);
    const T_ACC r = g < num_groups ? cumrstd[i * num_groups + g] : T_ACC(0);
    const T_ACC x_minus_mean = !memory_efficient ? x_or_y_acc - u : (x_or_y_acc - b_acc) / clamp_by_magnitude(r * w_acc, eps);
    const utils::KahanWrapper<T_ACC> t1 = sum1 + dy_acc * x_minus_mean * r;
    const utils::KahanWrapper<T_ACC> t2 = sum2 + dy_acc;
    sum1 = mask ? sum1 : t1;
    sum2 = mask ? sum2 : t2;
  }
  ds_shared[threadIdx.x][threadIdx.y] = *sum1;
  db_shared[threadIdx.x][threadIdx.y] = *sum2;

  __syncthreads();

  T_ACC s1 = ds_shared[threadIdx.y][threadIdx.x];
  T_ACC s2 = db_shared[threadIdx.y][threadIdx.x];
  s1 = reduce::WarpReduce(s1);
  s2 = reduce::WarpReduce(s2);

  if (threadIdx.x == 0) {
    const int64_t h = blockIdx.x * blockDim.x + threadIdx.y;
    if (h < inner_size) {
      w_grad[h] = static_cast<T>(s1);
      b_grad[h] = static_cast<T>(s2);
    }
  }
}

#define DISPATCH_ROWWISE_REDUCE_CUDA_FWD_KERNEL(KernelFunc, T, T_ACC, outer_size, inner_size, shm_size, cuda_stream, ...) \
  do {                                                                                                                    \
    if (inner_size <= 32) {                                                                                               \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 32, 32>, outer_size, 32, shm_size, cuda_stream, __VA_ARGS__);         \
    } else if (inner_size <= 64) {                                                                                        \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 64, 32>, outer_size, 32, shm_size, cuda_stream, __VA_ARGS__);         \
    } else if (inner_size <= 128) {                                                                                       \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 128, 32>, outer_size, 32, shm_size, cuda_stream, __VA_ARGS__);        \
    } else if (inner_size <= 256) {                                                                                       \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 256, 32>, outer_size, 32, shm_size, cuda_stream, __VA_ARGS__);        \
    } else if (inner_size <= 512) {                                                                                       \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 512, 64>, outer_size, 64, shm_size, cuda_stream, __VA_ARGS__);        \
    } else if (inner_size <= 1024) {                                                                                      \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1024, 128>, outer_size, 128, shm_size, cuda_stream, __VA_ARGS__);     \
    } else if (inner_size <= 2048) {                                                                                      \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2048, 256>, outer_size, 256, shm_size, cuda_stream, __VA_ARGS__);     \
    } else if (inner_size <= 4096) {                                                                                      \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4096, 512>, outer_size, 512, shm_size, cuda_stream, __VA_ARGS__);     \
    } else if (inner_size <= 8192) {                                                                                      \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8192, 1024>, outer_size, 1024, shm_size, cuda_stream, __VA_ARGS__);   \
    } else {                                                                                                              \
      TORCH_CHECK(false);                                                                                                 \
    }                                                                                                                     \
  } while (false)

#define DISPATCH_ROWWISE_REDUCE_CUDA_BWD_KERNEL(KernelFunc, T, T_ACC, outer_size, inner_size, memory_efficient, shm_size, cuda_stream, ...) \
  do {                                                                                                                                      \
    if (inner_size <= 32) {                                                                                                                 \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 32, 32, memory_efficient>, outer_size, 32, shm_size, cuda_stream, __VA_ARGS__);         \
    } else if (inner_size <= 64) {                                                                                                          \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 64, 32, memory_efficient>, outer_size, 32, shm_size, cuda_stream, __VA_ARGS__);         \
    } else if (inner_size <= 128) {                                                                                                         \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 128, 32, memory_efficient>, outer_size, 32, shm_size, cuda_stream, __VA_ARGS__);        \
    } else if (inner_size <= 256) {                                                                                                         \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 256, 32, memory_efficient>, outer_size, 32, shm_size, cuda_stream, __VA_ARGS__);        \
    } else if (inner_size <= 512) {                                                                                                         \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 512, 64, memory_efficient>, outer_size, 64, shm_size, cuda_stream, __VA_ARGS__);        \
    } else if (inner_size <= 1024) {                                                                                                        \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1024, 128, memory_efficient>, outer_size, 128, shm_size, cuda_stream, __VA_ARGS__);     \
    } else if (inner_size <= 2048) {                                                                                                        \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2048, 256, memory_efficient>, outer_size, 256, shm_size, cuda_stream, __VA_ARGS__);     \
    } else if (inner_size <= 4096) {                                                                                                        \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4096, 512, memory_efficient>, outer_size, 512, shm_size, cuda_stream, __VA_ARGS__);     \
    } else if (inner_size <= 8192) {                                                                                                        \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8192, 1024, memory_efficient>, outer_size, 1024, shm_size, cuda_stream, __VA_ARGS__);   \
    } else {                                                                                                                                \
      TORCH_CHECK(false);                                                                                                                   \
    }                                                                                                                                       \
  } while (false)

template <typename T>
void GroupTimestepDecayNormCUDAFwdImpl(const torch::Tensor& x,
                                       const torch::Tensor& bos_mask,
                                       const torch::Tensor& prev_count,
                                       const torch::Tensor& prev_mean,
                                       const torch::Tensor& prev_var, const torch::Tensor& gamma,
                                       const torch::Tensor& beta, const torch::Tensor& padding_mask,
                                       int64_t num_groups, double beta1, double beta2, double eps,
                                       torch::Tensor& y, torch::Tensor& count, torch::Tensor& mean,
                                       torch::Tensor& var, torch::Tensor& group_mean,
                                       torch::Tensor& group_var, torch::Tensor& cummean,
                                       torch::Tensor& cumrstd) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = x.size(0);
  const int64_t L = x.size(1);
  const int64_t H = x.size(2);
  const int64_t D = H / num_groups;

  const T* x_data = x.data_ptr<T>();
  const bool* bos_mask_data = bos_mask.defined() ? bos_mask.data_ptr<bool>() : nullptr;
  const int64_t* prev_count_data = prev_count.data_ptr<int64_t>();
  const T* prev_mean_data = prev_mean.data_ptr<T>();
  const T* prev_var_data = prev_var.data_ptr<T>();
  const T* gamma_data = gamma.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  const bool* padding_mask_data = padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* y_data = y.data_ptr<T>();
  int64_t* count_data = count.data_ptr<int64_t>();
  T* mean_data = mean.data_ptr<T>();
  T* var_data = var.data_ptr<T>();
  T_ACC* group_mean_data = group_mean.data_ptr<T_ACC>();
  T_ACC* group_var_data = group_var.data_ptr<T_ACC>();
  T_ACC* cummean_data = cummean.data_ptr<T_ACC>();
  T_ACC* cumrstd_data = cumrstd.data_ptr<T_ACC>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  {
    constexpr int64_t kShmSize = 0;
    const int64_t num_threads = cuda_utils::RowwiseNumThreads(D);
    DISPATCH_ROWWISE_REDUCE_CUDA_FWD_KERNEL(RowwiseMomentsKernel, T, T_ACC, B * L * num_groups, D,
                                            kShmSize, cuda_stream, D, x_data, group_mean_data,
                                            group_var_data);
  }
  if (L < cuda_utils::kColwiseThreshold) {
    const int64_t num_threads =
        (num_groups < cuda_utils::kCUDANumThreads ? cuda_utils::kWarpSize
                                                  : cuda_utils::kCUDANumThreads);
    const int64_t M = utils::DivUp(num_groups, num_threads);
    ColwiseCumMomentsSmallKernel<T, T_ACC><<<dim3(M, B), num_threads, 0, cuda_stream>>>(
        L, num_groups, bos_mask_data, prev_count_data, prev_mean_data, prev_var_data,
        group_mean_data, group_var_data, padding_mask_data,
        static_cast<T_ACC>(beta1), static_cast<T_ACC>(beta2), static_cast<T_ACC>(eps),
        count_data, mean_data, var_data, cummean_data, cumrstd_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    const int64_t M = utils::DivUp(num_groups, cuda_utils::kWarpSize);
    const int64_t chunk_size = utils::DivUp(L, cuda_utils::kWarpSize);
    ColwiseCumMomentsLargeKernel<T, T_ACC>
        <<<dim3(M, B), dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize), 0, cuda_stream>>>(
            L, num_groups, chunk_size,
            bos_mask_data, prev_count_data, prev_mean_data, prev_var_data,
            group_mean_data, group_var_data, padding_mask_data, static_cast<T_ACC>(beta1),
            static_cast<T_ACC>(beta2), static_cast<T_ACC>(eps), count_data, mean_data, var_data,
            cummean_data, cumrstd_data);
  }
  {
    const int64_t shm_size = sizeof(T_ACC) * num_groups * 2;
    cuda_utils::LaunchKernel(GroupTimestepDecayNormCUDAFwdKernel<T, T_ACC>, dim3(L, B),
                             cuda_utils::kCUDANumThreads, shm_size, cuda_stream, L, H, num_groups,
                             x_data, cummean_data, cumrstd_data, gamma_data, beta_data,
                             padding_mask_data, y_data);
  }
}

template <typename T>
void GroupTimestepDecayNormCUDABwdImpl(const torch::Tensor& y_grad, const torch::Tensor& mean_grad,
                                       const torch::Tensor& var_grad, const torch::Tensor& x_or_y,
                                       const torch::Tensor& prev_count,
                                       const torch::Tensor& bos_mask, const torch::Tensor& cummean,
                                       const torch::Tensor& cumrstd, const torch::Tensor& gamma,
                                       const torch::Tensor& beta, const torch::Tensor& padding_mask,
                                       int64_t num_groups, double beta1, double beta2, double eps,
                                       torch::Tensor& x_grad, torch::Tensor& prev_mean_grad,
                                       torch::Tensor& prev_var_grad, torch::Tensor& gamma_grad,
                                       torch::Tensor& beta_grad, bool memory_efficient) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = x_or_y.size(0);
  const int64_t L = x_or_y.size(1);
  const int64_t H = x_or_y.size(2);
  const int64_t D = H / num_groups;

  torch::Tensor group_mean = torch::empty({B, L, num_groups}, x_or_y.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  torch::Tensor ds = torch::empty({B, L, num_groups}, x_or_y.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  torch::Tensor db = torch::empty({B, L, num_groups}, x_or_y.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));

  const T* y_grad_data = y_grad.data_ptr<T>();
  const T* mean_grad_data = mean_grad.data_ptr<T>();
  const T* var_grad_data = var_grad.data_ptr<T>();
  const T* x_or_y_data = x_or_y.data_ptr<T>();
  const int64_t* prev_count_data = prev_count.data_ptr<int64_t>();
  const bool* bos_mask_data = bos_mask.defined() ? bos_mask.data_ptr<bool>() : nullptr;
  const T_ACC* cummean_data = cummean.data_ptr<T_ACC>();
  const T_ACC* cumrstd_data = cumrstd.data_ptr<T_ACC>();
  const T* gamma_data = gamma.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  const bool* padding_mask_data = padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* x_grad_data = x_grad.data_ptr<T>();
  T* prev_mean_grad_data = prev_mean_grad.data_ptr<T>();
  T* prev_var_grad_data = prev_var_grad.data_ptr<T>();
  T* gamma_grad_data = gamma_grad.data_ptr<T>();
  T* beta_grad_data = beta_grad.data_ptr<T>();
  T_ACC* group_mean_data = group_mean.data_ptr<T_ACC>();
  T_ACC* ds_data = ds.data_ptr<T_ACC>();
  T_ACC* db_data = db.data_ptr<T_ACC>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(x_or_y));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  {
    const int64_t block_size = std::max(int64_t(1) << utils::CeilLog2(D), cuda_utils::kWarpSize);
    const int64_t shm_size = sizeof(T_ACC) * cuda_utils::kWarpSize * 3 + sizeof(T_ACC) * block_size * 2;
    const int64_t num_threads = cuda_utils::RowwiseNumThreads(D);
    BOOL_SWITCH(
        memory_efficient, MemoryEfficient,
        DISPATCH_ROWWISE_REDUCE_CUDA_BWD_KERNEL(
            RowwiseInternalGradientsKernel, T, T_ACC, dim3(B * L, num_groups), D, MemoryEfficient,
            shm_size, cuda_stream, num_groups, D, y_grad_data, x_or_y_data, cummean_data,
            cumrstd_data, gamma_data, beta_data, group_mean_data, eps, ds_data, db_data));
  }
  if (L < cuda_utils::kColwiseThreshold) {
    const int64_t num_threads = cuda_utils::RowwiseNumThreads(num_groups);
    const int64_t M = utils::DivUp<int64_t>(num_groups, num_threads);
    ColwiseInternalGradientsSmallKernel<T, T_ACC><<<dim3(M, B), num_threads, 0, cuda_stream>>>(
        L, num_groups, beta1, beta2, mean_grad_data, var_grad_data, prev_count_data, bos_mask_data,
        cumrstd_data, padding_mask_data, ds_data, db_data, prev_mean_grad_data, prev_var_grad_data,
        ds_data, db_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    const int64_t M = utils::DivUp<int64_t>(num_groups, cuda_utils::kWarpSize);
    const int64_t chunk_size = utils::DivUp<int64_t>(L, cuda_utils::kWarpSize / 2);
    ColwiseInternalGradientsLargeKernel<T, T_ACC>
        <<<dim3(M, B), dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize / 2), 0, cuda_stream>>>(
            L, num_groups, beta1, beta2, chunk_size, mean_grad_data, var_grad_data, prev_count_data,
            bos_mask_data, cumrstd_data, padding_mask_data, ds_data, db_data, prev_mean_grad_data,
            prev_var_grad_data, ds_data, db_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  {
    const int64_t shm_size = sizeof(T_ACC) * num_groups * 4;
    BOOL_SWITCH(memory_efficient, MemoryEfficient,
                cuda_utils::LaunchKernel(
                    GroupTimestepNormCUDABwdKernel<T, T_ACC, MemoryEfficient>, dim3(L, B),
                    cuda_utils::kCUDANumThreads, shm_size, cuda_stream, L, H, num_groups,
                    y_grad_data, x_or_y_data, group_mean_data, cummean_data, cumrstd_data,
                    gamma_data, beta_data, padding_mask_data, ds_data, db_data, eps, x_grad_data));
  }

  if (L < cuda_utils::kColwiseThreshold) {
    const int64_t num_threads = cuda_utils::RowwiseNumThreads(num_groups);
    const int64_t M = utils::DivUp<int64_t>(H, num_threads);
    BOOL_SWITCH(memory_efficient, MemoryEfficient,
                GroupGammaBetaCUDABwdSmallKernel<T, T_ACC, MemoryEfficient>
                <<<M, num_threads, 0, cuda_stream>>>(B * L, H, num_groups, y_grad_data, x_or_y_data,
                                                     gamma_data, beta_data, cummean_data,
                                                     cumrstd_data, padding_mask_data, eps,
                                                     gamma_grad_data, beta_grad_data));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    const int64_t M = utils::DivUp<int64_t>(H, cuda_utils::kWarpSize);
    BOOL_SWITCH(
        memory_efficient, MemoryEfficient,
        GroupGammaBetaCUDABwdLargeKernel<T, T_ACC, MemoryEfficient>
        <<<M, dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize), 0, cuda_stream>>>(
            B * L, H, num_groups, y_grad_data, x_or_y_data, gamma_data, beta_data, cummean_data,
            cumrstd_data, padding_mask_data, eps, gamma_grad_data, beta_grad_data));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

#undef DISPATCH_ROWWISE_REDUCE_CUDA_FWD_KERNEL
#undef DISPATCH_ROWWISE_REDUCE_CUDA_BWD_KERNEL

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GroupTimestepDecayNormCUDAFwd(const torch::Tensor& x, const c10::optional<torch::Tensor>& bos_mask,
                              const torch::Tensor& prev_count, const torch::Tensor& prev_mean,
                              const torch::Tensor& prev_var, const torch::Tensor& gamma,
                              const torch::Tensor& beta,
                              const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups,
                              double beta1, double beta2, double eps) {
  const int64_t B = x.size(0);
  const int64_t L = x.size(1);

  c10::MaybeOwned<torch::Tensor> bos_mask_maybe_owned = at::borrow_from_optional_tensor(bos_mask);
  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned = at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor y = torch::empty_like(x, x.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor count = torch::empty_like(prev_count, prev_count.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor mean = torch::empty_like(prev_mean, prev_mean.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor var = torch::empty_like(prev_var, prev_var.options().memory_format(at::MemoryFormat::Contiguous));

  const auto acc_type = at::toAccumulateType(x.scalar_type(), true);
  torch::Tensor group_mean = torch::empty({B, L, num_groups}, x.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor group_var = torch::empty({B, L, num_groups}, x.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor cummean = torch::empty({B, L, num_groups}, x.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor cumrstd = torch::empty({B, L, num_groups}, x.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "GroupTimestepDecayNormCUDAFwd", [&]() {
        GroupTimestepDecayNormCUDAFwdImpl<scalar_t>(
            *(x.expect_contiguous()), *(bos_mask_maybe_owned->expect_contiguous()),
            *(prev_count.expect_contiguous()),
            *(prev_mean.expect_contiguous()), *(prev_var.expect_contiguous()),
            *(gamma.expect_contiguous()), *(beta.expect_contiguous()),
            *(padding_mask_maybe_owned->expect_contiguous()), num_groups, beta1, beta2, eps, y,
            count, mean, var, group_mean, group_var, cummean, cumrstd);
      });

  return std::make_tuple(y, count, mean, var, cummean, cumrstd);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GroupTimestepDecayNormCUDABwd(const torch::Tensor& y_grad, const torch::Tensor& mean_grad,
                              const torch::Tensor& var_grad, const torch::Tensor& x_or_y,
                              const torch::Tensor& prev_count, const c10::optional<torch::Tensor>& bos_mask,
                              const torch::Tensor& cummean, const torch::Tensor& cumrstd,
                              const torch::Tensor& gamma, const torch::Tensor& beta,
                              const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups,
                              double beta1, double beta2, double eps, bool memory_efficient) {
  c10::MaybeOwned<torch::Tensor> bos_mask_maybe_owned = at::borrow_from_optional_tensor(bos_mask);
  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned = at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor x_grad = torch::empty_like(x_or_y, x_or_y.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor prev_mean_grad = torch::empty_like(mean_grad, mean_grad.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor prev_var_grad = torch::empty_like(var_grad, var_grad.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor gamma_grad = torch::empty_like(gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor beta_grad = torch::empty_like(gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x_or_y.scalar_type(), "GroupTimestepDecayNormCUDABwd", [&]() {
        GroupTimestepDecayNormCUDABwdImpl<scalar_t>(
            *(y_grad.expect_contiguous()), *(mean_grad.expect_contiguous()),
            *(var_grad.expect_contiguous()), *(x_or_y.expect_contiguous()),
            *(prev_count.expect_contiguous()), *(bos_mask_maybe_owned->expect_contiguous()),
            *(cummean.expect_contiguous()), *(cumrstd.expect_contiguous()),
            *(gamma.expect_contiguous()), *(beta.expect_contiguous()),
            *(padding_mask_maybe_owned->expect_contiguous()),
            num_groups, beta1, beta2, eps, x_grad, prev_mean_grad, prev_var_grad, gamma_grad,
            beta_grad, memory_efficient);
      });
  return std::make_tuple(x_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad);
}

}  // namespace ops
}  // namespace xllm
