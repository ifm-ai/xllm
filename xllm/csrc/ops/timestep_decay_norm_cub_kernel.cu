#include <ATen/AccumulateType.h>
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/MaybeOwned.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>

#include <ATen/native/cuda/block_reduce.cuh>
#include <cub/cub.cuh>
#include <cstring>
#include <tuple>

#include "cuda_utils.cuh"
#include "ops/timestep_decay_norm_cub.h"
#include "reduce.cuh"
#include "utils.h"
#include "welford.h"

namespace xllm {
namespace ops {

namespace {

constexpr int kCUBFwdBlockSize = 256;
constexpr int kCUBBwdBlockSize = 512;
constexpr int kCUBMomentBlockSize = 32;
constexpr int kParamGradBlockSize = 256;

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

void CheckTimestepDecayNormCubFwdInputs(
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

void CheckTimestepDecayNormCubBwdInputs(
    const torch::Tensor& y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& x,
    const torch::Tensor& prev_count, const c10::optional<torch::Tensor>& bos_mask,
    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups,
    double beta1, double beta2) {
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

__device__ __forceinline__ float4 ComposeAffine4(const float4& prefix,
                                                 const float4& current) {
  return make_float4(fmaf(current.y, prefix.x, current.x),
                     current.y * prefix.y,
                     fmaf(current.w, prefix.z, current.z),
                     current.w * prefix.w);
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

template <int STATIC_G>
__device__ __forceinline__ int64_t StaticOrRuntimeG(int64_t G) {
  if constexpr (STATIC_G > 0) {
    return STATIC_G;
  } else {
    return G;
  }
}

template <int STATIC_DG>
__device__ __forceinline__ int64_t StaticOrRuntimeDg(int64_t Dg) {
  if constexpr (STATIC_DG > 0) {
    return STATIC_DG;
  } else {
    return Dg;
  }
}

template <typename T, typename T_ACC, int BLOCK_THREADS, int STATIC_G,
          int STATIC_DG>
__global__ void RowwiseMomentsCUBKernel(
    int64_t G, int64_t Dg, int64_t L, const T* __restrict__ x,
    const bool* __restrict__ padding_mask, T_ACC* __restrict__ group_mean,
    T_ACC* __restrict__ group_var) {
  const int64_t Gv = StaticOrRuntimeG<STATIC_G>(G);
  const int64_t Dgv = StaticOrRuntimeDg<STATIC_DG>(Dg);
  const int64_t row = blockIdx.x;
  const int64_t b = row / L;
  const int64_t t = row - b * L;
  const int64_t g = blockIdx.y;
  const int64_t tid = threadIdx.x;
  const int64_t h_base = g * Dgv;

  __shared__ utils::WelfordData<float> shm[cuda_utils::kWarpSize];

  const bool pad = padding_mask != nullptr && padding_mask[b * L + t];
  utils::WelfordData<float> moments;
  if (!pad) {
    if constexpr (STATIC_DG > 0) {
      for (int64_t d = tid; d < STATIC_DG; d += BLOCK_THREADS) {
        moments +=
            static_cast<float>(x[XIndex(b, t, h_base + d, L, Gv * Dgv)]);
      }
    } else {
      for (int64_t d = tid; d < Dgv; d += BLOCK_THREADS) {
        moments +=
            static_cast<float>(x[XIndex(b, t, h_base + d, L, Gv * Dgv)]);
      }
    }
  }

  moments = reduce::BlockReduce(moments, shm);
  if (tid == 0) {
    const int64_t idx = GroupIndex(b, g, t, Gv, L);
    group_mean[idx] = static_cast<T_ACC>(pad ? 0.0f : moments.m1);
    group_var[idx] = static_cast<T_ACC>(pad ? 0.0f : moments.m2);
  }
}

template <typename T, typename T_ACC, int BLOCK_THREADS, int STATIC_G>
__global__ void GroupTimestepDecayNormCUBFwdKernel(
    int64_t G, int64_t L, const bool* __restrict__ bos_mask,
    const bool* __restrict__ padding_mask, const int64_t* __restrict__ prev_count,
    const T* __restrict__ prev_mean, const T* __restrict__ prev_var,
    const T_ACC* __restrict__ group_mean,
    const T_ACC* __restrict__ group_var, T_ACC beta1, T_ACC beta2, T_ACC eps,
    T* __restrict__ mean_out, T* __restrict__ var_out,
    T_ACC* __restrict__ cummean_out, T_ACC* __restrict__ cumrstd_out) {
  using PairScan =
      cub::BlockScan<float4, BLOCK_THREADS, cub::BLOCK_SCAN_WARP_SCANS>;

  __shared__ typename PairScan::TempStorage pair_scan_storage;
  __shared__ float running_mean;
  __shared__ float running_var;
  __shared__ float running_b1;
  __shared__ float running_b2;

  const int64_t Gv = StaticOrRuntimeG<STATIC_G>(G);
  const int64_t b = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int64_t tid = threadIdx.x;

  const float beta1_f = static_cast<float>(beta1);
  const float beta2_f = static_cast<float>(beta2);
  const float one_minus_beta1 = 1.0f - beta1_f;
  const float one_minus_beta2 = 1.0f - beta2_f;
  const float eps_f = static_cast<float>(eps);

  if (tid == 0) {
    const int64_t running_count = prev_count != nullptr ? prev_count[b] : 0;
    running_mean =
        prev_mean != nullptr ? static_cast<float>(prev_mean[b * Gv + g])
                             : 0.0f;
    running_var =
        prev_var != nullptr ? static_cast<float>(prev_var[b * Gv + g]) : 0.0f;
    running_b1 = powf(beta1_f, static_cast<float>(running_count));
    running_b2 = powf(beta2_f, static_cast<float>(running_count));
  }
  __syncthreads();

  const int64_t num_chunks = DeviceDivUp(L, int64_t(BLOCK_THREADS));
  for (int64_t c = 0; c < num_chunks; ++c) {
    const int64_t t = c * int64_t(BLOCK_THREADS) + tid;
    const bool valid = t < L;
    const bool pad = valid && padding_mask != nullptr && padding_mask[b * L + t];
    const bool bos = valid && bos_mask != nullptr && bos_mask[b * L + t];
    const bool identity = !valid || pad;

    const float run_mean = running_mean;
    const float run_var = running_var;
    const float run_b1 = running_b1;
    const float run_b2 = running_b2;

    const int64_t group_idx = valid ? GroupIndex(b, g, t, Gv, L) : 0;
    const float group_mean_t =
        valid ? static_cast<float>(group_mean[group_idx]) : 0.0f;
    const float group_var_t =
        valid ? static_cast<float>(group_var[group_idx]) : 0.0f;

    const float4 mv_item =
        identity ? IdentityAffine4()
                 : MakeAffine4(one_minus_beta1 * group_mean_t,
                               bos ? 0.0f : beta1_f,
                               one_minus_beta2 * group_var_t,
                               bos ? 0.0f : beta2_f);
    float4 mv_scan = IdentityAffine4();
    PairScan(pair_scan_storage).InclusiveScan(mv_item, mv_scan, Affine4Prod{});
    __syncthreads();

    const float4 pow_item =
        identity ? IdentityAffine4()
                 : MakeAffine4(bos ? beta1_f : 0.0f,
                               bos ? 0.0f : beta1_f,
                               bos ? beta2_f : 0.0f,
                               bos ? 0.0f : beta2_f);
    float4 pow_scan = IdentityAffine4();
    PairScan(pair_scan_storage).InclusiveScan(pow_item, pow_scan,
                                              Affine4Prod{});
    __syncthreads();

    const float mean_total = ApplyAffineX(mv_scan, run_mean);
    const float var_total = ApplyAffineY(mv_scan, run_var);
    const float b1_total = ApplyAffineX(pow_scan, run_b1);
    const float b2_total = ApplyAffineY(pow_scan, run_b2);

    if (valid) {
      float curr_mean = 0.0f;
      float curr_rstd = rsqrtf(eps_f);
      const float den1 = 1.0f - b1_total;
      const float den2 = 1.0f - b2_total;
      if (den1 > 0.0f && den2 > 0.0f) {
        curr_mean = mean_total / den1;
        curr_rstd = rsqrtf(var_total / den2 + eps_f);
      }

      cummean_out[group_idx] = static_cast<T_ACC>(curr_mean);
      cumrstd_out[group_idx] = static_cast<T_ACC>(curr_rstd);
    }

    if (tid == BLOCK_THREADS - 1) {
      running_mean = mean_total;
      running_var = var_total;
      running_b1 = b1_total;
      running_b2 = b2_total;
    }
    __syncthreads();
  }

  if (tid == BLOCK_THREADS - 1) {
    if (mean_out != nullptr) {
      mean_out[b * Gv + g] = static_cast<T>(running_mean);
    }
    if (var_out != nullptr) {
      var_out[b * Gv + g] = static_cast<T>(running_var);
    }
  }
}

template <typename T, typename T_ACC, int STATIC_G, int STATIC_DG>
__global__ void GroupTimestepDecayNormCUBApplyFwdKernel(
    int64_t L, int64_t H, int64_t G, int64_t Dg,
    const T* __restrict__ x, const T_ACC* __restrict__ cummean,
    const T_ACC* __restrict__ cumrstd, const T* __restrict__ gamma,
    const T* __restrict__ beta, const bool* __restrict__ padding_mask,
    T* __restrict__ y) {
  extern __shared__ float shm[];

  const int64_t Gv = StaticOrRuntimeG<STATIC_G>(G);
  const int64_t Dgv = StaticOrRuntimeDg<STATIC_DG>(Dg);
  const int64_t Hv =
      (STATIC_G > 0 && STATIC_DG > 0) ? STATIC_G * STATIC_DG : H;
  const int64_t t = blockIdx.x;
  const int64_t b = blockIdx.y;
  const bool pad = padding_mask != nullptr && padding_mask[b * L + t];
  const int64_t row_offset = (b * L + t) * Hv;

  if (pad) {
    for (int64_t h = threadIdx.x; h < Hv; h += blockDim.x) {
      y[row_offset + h] = T(0);
    }
    return;
  }

  float* mean_shared = shm;
  float* rstd_shared = mean_shared + Gv;
  for (int64_t g = threadIdx.x; g < Gv; g += blockDim.x) {
    const int64_t idx = GroupIndex(b, g, t, Gv, L);
    mean_shared[g] = static_cast<float>(cummean[idx]);
    rstd_shared[g] = static_cast<float>(cumrstd[idx]);
  }
  __syncthreads();

  for (int64_t h = threadIdx.x; h < Hv; h += blockDim.x) {
    const int64_t g = h / Dgv;
    const float x_val = static_cast<float>(x[row_offset + h]);
    const float mean = mean_shared[g];
    const float rstd = rstd_shared[g];
    const float w = static_cast<float>(gamma[h]);
    const float b_val = static_cast<float>(beta[h]);
    y[row_offset + h] = static_cast<T>((x_val - mean) * rstd * w + b_val);
  }
}

__global__ void FinalCountCUBKernel(
    int64_t B, int64_t L, const bool* __restrict__ bos_mask,
    const bool* __restrict__ padding_mask, const int64_t* __restrict__ prev_count,
    int64_t* __restrict__ count_out) {
  const int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B || count_out == nullptr) {
    return;
  }

  int64_t count = prev_count != nullptr ? prev_count[b] : 0;
  for (int64_t t = 0; t < L; ++t) {
    const bool pad = padding_mask != nullptr && padding_mask[b * L + t];
    const bool bos = bos_mask != nullptr && bos_mask[b * L + t];
    if (!pad) {
      count = bos ? 1 : count + 1;
    }
  }
  count_out[b] = count;
}

__global__ void ComputeCountCUBKernel(
    int64_t B, int64_t L, const bool* __restrict__ bos_mask,
    const bool* __restrict__ padding_mask, const int64_t* __restrict__ prev_count,
    int64_t* __restrict__ count_array) {
  const int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) {
    return;
  }

  int64_t c = prev_count != nullptr ? prev_count[b] : 0;
  for (int64_t t = 0; t < L; ++t) {
    const bool pad = padding_mask != nullptr && padding_mask[b * L + t];
    const bool bos = bos_mask != nullptr && bos_mask[b * L + t];
    if (!pad) {
      c = bos ? 1 : c + 1;
    }
    count_array[b * L + t] = c;
  }
}

template <typename T, typename T_ACC, int BLOCK_THREADS>
__global__ void RowwiseInternalGradientsCUBKernel(
    int64_t B, int64_t G, int64_t Dg, int64_t L, const T* __restrict__ dy,
    const T* __restrict__ x, const T* __restrict__ gamma,
    const T_ACC* __restrict__ cummean, const bool* __restrict__ padding_mask,
    T_ACC* __restrict__ group_mean, T_ACC* __restrict__ ds_out,
    T_ACC* __restrict__ db_out) {
  const int64_t b = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int64_t tid = threadIdx.x;
  const int64_t h_base = g * Dg;
  const int64_t H = G * Dg;
  const int64_t num_chunks = DeviceDivUp(L, int64_t(BLOCK_THREADS));

  for (int64_t c = 0; c < num_chunks; ++c) {
    const int64_t t = c * int64_t(BLOCK_THREADS) + tid;
    if (t >= L) {
      continue;
    }

    const bool pad = padding_mask != nullptr && padding_mask[b * L + t];
    const int64_t idx = GroupIndex(b, g, t, G, L);
    const float cummean_t = static_cast<float>(cummean[idx]);

    float mean = 0.0f;
    float ds = 0.0f;
    float db = 0.0f;
    if (!pad) {
      float sum_gx = 0.0f;
      const int64_t row_offset = XIndex(b, t, h_base, L, H);
      const T* x_row = x + row_offset;
      const T* dy_row = dy + row_offset;
      const T* gamma_row = gamma + h_base;

      for (int64_t d = 0; d < Dg; ++d) {
        const float x_val = static_cast<float>(x_row[d]);
        const float grad_hat =
            static_cast<float>(dy_row[d]) * static_cast<float>(gamma_row[d]);
        mean += (x_val - mean) /
                static_cast<float>(d + 1);
        db += grad_hat;
        sum_gx += grad_hat * x_val;
      }
      ds = sum_gx - cummean_t * db;
    }

    group_mean[idx] = static_cast<T_ACC>(mean);
    ds_out[idx] = static_cast<T_ACC>(ds);
    db_out[idx] = static_cast<T_ACC>(db);
  }
}

template <typename T, typename T_ACC, int BLOCK_THREADS, int STATIC_G>
__global__ void ColwiseInternalGradientsCUBKernel(
    int64_t B, int64_t G, int64_t L, double beta1, double beta2,
    const T* __restrict__ mean_grad, const T* __restrict__ var_grad,
    const int64_t* __restrict__ count_array, const bool* __restrict__ bos_mask,
    const T_ACC* __restrict__ cumrstd, const bool* __restrict__ padding_mask,
    const T_ACC* __restrict__ ds_in, const T_ACC* __restrict__ db_in,
    T* __restrict__ prev_mean_grad, T* __restrict__ prev_var_grad,
    T_ACC* __restrict__ du_out, T_ACC* __restrict__ dv_out) {
  using PairScan =
      cub::BlockScan<float4, BLOCK_THREADS, cub::BLOCK_SCAN_WARP_SCANS>;

  __shared__ typename PairScan::TempStorage pair_scan_storage;
  __shared__ float running_u;
  __shared__ float running_v;

  const int64_t Gv = StaticOrRuntimeG<STATIC_G>(G);
  const int64_t b = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int64_t tid = threadIdx.x;

  const float beta1_f = static_cast<float>(beta1);
  const float beta2_f = static_cast<float>(beta2);
  const float log_beta1_f = logf(beta1_f);
  const float log_beta2_f = logf(beta2_f);
  const float one_minus_beta1 = 1.0f - beta1_f;
  const float one_minus_beta2 = 1.0f - beta2_f;

  if (tid == 0) {
    running_u = static_cast<float>(mean_grad[b * Gv + g]);
    running_v = static_cast<float>(var_grad[b * Gv + g]);
  }
  __syncthreads();

  const int64_t num_chunks = DeviceDivUp(L, int64_t(BLOCK_THREADS));
  for (int64_t c = num_chunks - 1; c >= 0; --c) {
    const int64_t t = c * int64_t(BLOCK_THREADS) + (BLOCK_THREADS - 1 - tid);
    const bool valid = t < L;
    const bool pad = valid && padding_mask != nullptr && padding_mask[b * L + t];
    const bool bos = valid && bos_mask != nullptr && bos_mask[b * L + t];

    const float run_u = running_u;
    const float run_v = running_v;

    const int64_t idx = valid ? GroupIndex(b, g, t, Gv, L) : 0;
    const int64_t c_t = valid ? count_array[b * L + t] : 0;
    const float r_t = valid ? static_cast<float>(cumrstd[idx]) : 0.0f;
    const float ds_t = valid ? static_cast<float>(ds_in[idx]) : 0.0f;
    const float db_t = valid ? static_cast<float>(db_in[idx]) : 0.0f;

    float local_u = 0.0f;
    float local_v = 0.0f;
    if (valid && !pad && c_t > 0) {
      const float c_t_f = static_cast<float>(c_t);
      const float beta1_pow = expf(log_beta1_f * c_t_f);
      const float beta2_pow = expf(log_beta2_f * c_t_f);
      local_u = -r_t * db_t / (1.0f - beta1_pow);
      local_v =
          -0.5f * (r_t * r_t * r_t) * ds_t / (1.0f - beta2_pow);
    }

    const float4 transform =
        (!valid || pad)
            ? IdentityAffine4()
            : (bos ? MakeAffine4(0.0f, 0.0f, 0.0f, 0.0f)
                   : MakeAffine4(beta1_f * local_u, beta1_f,
                                 beta2_f * local_v, beta2_f));

    float4 exclusive = IdentityAffine4();
    PairScan(pair_scan_storage)
        .ExclusiveScan(transform, exclusive, IdentityAffine4(), Affine4Prod{});
    __syncthreads();

    const float u_in = ApplyAffineX(exclusive, run_u);
    const float v_in = ApplyAffineY(exclusive, run_v);

    if (valid) {
      if (!pad) {
        du_out[idx] = static_cast<T_ACC>(one_minus_beta1 * (u_in + local_u));
        dv_out[idx] = static_cast<T_ACC>(one_minus_beta2 * (v_in + local_v));
      } else {
        du_out[idx] = T_ACC(0);
        dv_out[idx] = T_ACC(0);
      }
    }

    const float4 inclusive = ComposeAffine4(exclusive, transform);
    if (tid == BLOCK_THREADS - 1) {
      running_u = ApplyAffineX(inclusive, run_u);
      running_v = ApplyAffineY(inclusive, run_v);
    }
    __syncthreads();
  }

  if (tid == 0) {
    prev_mean_grad[b * Gv + g] = static_cast<T>(running_u);
    prev_var_grad[b * Gv + g] = static_cast<T>(running_v);
  }
}

template <typename T, typename T_ACC, int STATIC_G, int STATIC_DG>
__global__ void GroupTimestepDecayNormCUBBwdKernel(
    int64_t L, int64_t H, int64_t G, const T* __restrict__ dy,
    const T* __restrict__ x, const T_ACC* __restrict__ group_mean,
    const T_ACC* __restrict__ cumrstd, const T* __restrict__ gamma,
    const bool* __restrict__ padding_mask, const T_ACC* __restrict__ du,
    const T_ACC* __restrict__ dv, T* __restrict__ dx) {
  const int64_t Gv = StaticOrRuntimeG<STATIC_G>(G);
  const int64_t Dgv = StaticOrRuntimeDg<STATIC_DG>(H / G);
  const int64_t Hv =
      (STATIC_G > 0 && STATIC_DG > 0) ? STATIC_G * STATIC_DG : H;
  const int64_t b = blockIdx.y;
  const int64_t t = blockIdx.x;
  const T_ACC coef = T_ACC(1) / static_cast<T_ACC>(Dgv);
  const int64_t row_offset = (b * L + t) * Hv;
  const T* dy_row = dy + row_offset;
  const T* x_row = x + row_offset;
  T* dx_row = dx + row_offset;

  extern __shared__ unsigned char raw_shared[];
  T_ACC* mean_shared = reinterpret_cast<T_ACC*>(raw_shared);
  T_ACC* rstd_shared = mean_shared + Gv;
  T_ACC* du_shared = rstd_shared + Gv;
  T_ACC* dv_shared = du_shared + Gv;

  const bool pad = padding_mask != nullptr && padding_mask[b * L + t];
  if (pad) {
    for (int64_t h = threadIdx.x; h < Hv; h += blockDim.x) {
      dx_row[h] = T(0);
    }
    return;
  }

  for (int64_t g = threadIdx.x; g < Gv; g += blockDim.x) {
    const int64_t idx = GroupIndex(b, g, t, Gv, L);
    mean_shared[g] = group_mean[idx];
    rstd_shared[g] = cumrstd[idx];
    du_shared[g] = du[idx];
    dv_shared[g] = dv[idx];
  }
  __syncthreads();

  for (int64_t h = threadIdx.x; h < Hv; h += blockDim.x) {
    const int64_t g = h / Dgv;
    const T_ACC dy_acc = static_cast<T_ACC>(dy_row[h]);
    const T_ACC x_acc = static_cast<T_ACC>(x_row[h]);
    const T_ACC mean = mean_shared[g];
    const T_ACC rstd = rstd_shared[g];
    const T_ACC du_acc = du_shared[g];
    const T_ACC dv_acc = dv_shared[g];
    const T_ACC w = static_cast<T_ACC>(gamma[h]);
    dx_row[h] = static_cast<T>(
        dy_acc * rstd * w + coef * (du_acc + T_ACC(2) * dv_acc * (x_acc - mean)));
  }
}

template <typename T, typename T_ACC, int STATIC_G, int STATIC_DG>
__global__ void GroupGammaBetaCUBBwdSmallKernel(
    int64_t B, int64_t L, int64_t H, int64_t G, int64_t Dg,
    const T* __restrict__ dy, const T* __restrict__ x,
    const T_ACC* __restrict__ cummean, const T_ACC* __restrict__ cumrstd,
    const bool* __restrict__ padding_mask, T* __restrict__ dgamma,
    T* __restrict__ dbeta) {
  const int64_t Gv = StaticOrRuntimeG<STATIC_G>(G);
  const int64_t Dgv = StaticOrRuntimeDg<STATIC_DG>(Dg);
  const int64_t Hv =
      (STATIC_G > 0 && STATIC_DG > 0) ? STATIC_G * STATIC_DG : H;
  const int64_t h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= Hv) {
    return;
  }
  const int64_t g = h / Dgv;

  T_ACC dg = T_ACC(0);
  T_ACC db = T_ACC(0);
  const int64_t outer = B * L;
  for (int64_t idx = 0; idx < outer; ++idx) {
    const int64_t b = idx / L;
    const int64_t t = idx % L;
    const bool pad = padding_mask != nullptr && padding_mask[b * L + t];
    if (!pad) {
      const T_ACC dy_acc = static_cast<T_ACC>(dy[XIndex(b, t, h, L, Hv)]);
      const T_ACC x_acc = static_cast<T_ACC>(x[XIndex(b, t, h, L, Hv)]);
      const T_ACC mean = cummean[GroupIndex(b, g, t, Gv, L)];
      const T_ACC rstd = cumrstd[GroupIndex(b, g, t, Gv, L)];
      dg += dy_acc * (x_acc - mean) * rstd;
      db += dy_acc;
    }
  }
  dgamma[h] = static_cast<T>(dg);
  dbeta[h] = static_cast<T>(db);
}

template <typename T, typename T_ACC, int STATIC_G, int STATIC_DG>
__global__ void GroupGammaBetaCUBBwdLargeKernel(
    int64_t B, int64_t L, int64_t H, int64_t G, int64_t Dg,
    const T* __restrict__ dy, const T* __restrict__ x,
    const T_ACC* __restrict__ cummean, const T_ACC* __restrict__ cumrstd,
    const bool* __restrict__ padding_mask, T* __restrict__ dgamma,
    T* __restrict__ dbeta) {
  __shared__ T_ACC dgamma_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC dbeta_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];

  const int64_t Gv = StaticOrRuntimeG<STATIC_G>(G);
  const int64_t Dgv = StaticOrRuntimeDg<STATIC_DG>(Dg);
  const int64_t Hv =
      (STATIC_G > 0 && STATIC_DG > 0) ? STATIC_G * STATIC_DG : H;
  const int64_t h = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t g = h / Dgv;
  const int64_t outer = B * L;

  T_ACC dg = T_ACC(0);
  T_ACC db = T_ACC(0);
  for (int64_t idx = threadIdx.y; idx < outer; idx += blockDim.y) {
    const int64_t b = idx / L;
    const int64_t t = idx - b * L;
    const bool pad = padding_mask != nullptr && padding_mask[b * L + t];
    if (!pad && h < Hv) {
      const int64_t row_offset = idx * Hv;
      const T_ACC dy_acc = static_cast<T_ACC>(dy[row_offset + h]);
      const T_ACC x_acc = static_cast<T_ACC>(x[row_offset + h]);
      const int64_t group_idx = GroupIndex(b, g, t, Gv, L);
      const T_ACC mean = cummean[group_idx];
      const T_ACC rstd = cumrstd[group_idx];
      dg += dy_acc * (x_acc - mean) * rstd;
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
    if (out_h < Hv) {
      dgamma[out_h] = static_cast<T>(dg_reduce);
      dbeta[out_h] = static_cast<T>(db_reduce);
    }
  }
}

#define TSDN_CUB_FWD_CASE(                                                  \
    T, T_ACC, STATIC_G, STATIC_DG, cuda_stream, B, L, H, G, Dg,            \
    x_data, bos_mask_data, padding_mask_data, prev_count_data,              \
    prev_mean_data, prev_var_data, gamma_data, beta_data,                   \
    group_mean_data, group_var_data, y_data, count_data, mean_data,         \
    var_data, cummean_data, cumrstd_data, beta1, beta2, eps)               \
  do {                                                                      \
    cuda_utils::LaunchKernel(                                               \
        RowwiseMomentsCUBKernel<                                            \
            T, T_ACC, kCUBMomentBlockSize, STATIC_G, STATIC_DG>,            \
        dim3((B) * (L), (G)), dim3(kCUBMomentBlockSize), 0,                 \
        cuda_stream, G, Dg, L, x_data, padding_mask_data, group_mean_data,  \
        group_var_data);                                                    \
    constexpr int64_t kCountThreads = 256;                                  \
    const int64_t count_blocks = utils::DivUp((B), kCountThreads);          \
    FinalCountCUBKernel<<<count_blocks, kCountThreads, 0, cuda_stream>>>(   \
        B, L, bos_mask_data, padding_mask_data, prev_count_data,            \
        count_data);                                                        \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                         \
    cuda_utils::LaunchKernel(                                               \
        GroupTimestepDecayNormCUBFwdKernel<                                 \
            T, T_ACC, kCUBFwdBlockSize, STATIC_G>,                          \
        dim3((B), (G)), dim3(kCUBFwdBlockSize), 0, cuda_stream, G, L,       \
        bos_mask_data, padding_mask_data, prev_count_data, prev_mean_data,  \
        prev_var_data, group_mean_data, group_var_data,                     \
        static_cast<T_ACC>(beta1), static_cast<T_ACC>(beta2),               \
        static_cast<T_ACC>(eps), mean_data, var_data, cummean_data,         \
        cumrstd_data);                                                      \
    const int64_t apply_shm_size = sizeof(float) * (G) * 2;                 \
    cuda_utils::LaunchKernel(                                               \
        GroupTimestepDecayNormCUBApplyFwdKernel<                            \
            T, T_ACC, STATIC_G, STATIC_DG>,                                 \
        dim3((L), (B)), cuda_utils::kCUDANumThreads, apply_shm_size,        \
        cuda_stream, L, H, G, Dg, x_data, cummean_data, cumrstd_data,       \
        gamma_data, beta_data, padding_mask_data, y_data);                  \
  } while (false)

#define DISPATCH_TSDN_CUB_FWD(                                              \
    T, T_ACC, cuda_stream, B, L, H, G, Dg, ...)                            \
  do {                                                                      \
    if (G == 2 && Dg == 3) {                                                \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 2, 3, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);       \
    } else if (G == 4 && Dg == 32) {                                        \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 4, 32, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);      \
    } else if (G == 32 && Dg == 8) {                                        \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 32, 8, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);      \
    } else if (G == 32 && Dg == 32) {                                       \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 32, 32, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);     \
    } else if (G == 64 && Dg == 4) {                                        \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 64, 4, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);      \
    } else if (G == 64 && Dg == 16) {                                       \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 64, 16, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);     \
    } else if (G == 128 && Dg == 8) {                                       \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 128, 8, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);     \
    } else if (G == 256 && Dg == 4) {                                       \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 256, 4, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);     \
    } else if (G == 32 && Dg == 64) {                                       \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 32, 64, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);     \
    } else if (G == 64 && Dg == 32) {                                       \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 64, 32, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);     \
    } else if (G == 128 && Dg == 16) {                                      \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 128, 16, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);    \
    } else if (G == 256 && Dg == 8) {                                       \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 256, 8, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);     \
    } else if (G == 32 && Dg == 128) {                                      \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 32, 128, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);    \
    } else if (G == 64 && Dg == 64) {                                       \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 64, 64, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);     \
    } else if (G == 128 && Dg == 32) {                                      \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 128, 32, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);    \
    } else if (G == 256 && Dg == 16) {                                      \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 256, 16, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);    \
    } else if (Dg == 128) {                                                 \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 0, 128, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);     \
    } else if (Dg == 64) {                                                  \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 0, 64, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);      \
    } else if (Dg == 32) {                                                  \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 0, 32, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);      \
    } else if (Dg == 16) {                                                  \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 0, 16, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);      \
    } else if (Dg == 8) {                                                   \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 0, 8, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);       \
    } else if (Dg == 4) {                                                   \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 0, 4, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);       \
    } else {                                                                \
      TSDN_CUB_FWD_CASE(                                                    \
          T, T_ACC, 0, 0, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);       \
    }                                                                       \
  } while (false)

#define TSDN_CUB_BWD_CASE(                                                  \
    T, T_ACC, STATIC_G, STATIC_DG, cuda_stream, B, L, H, G, Dg,            \
    y_grad_data, mean_grad_data, var_grad_data, x_data, prev_count_data,   \
    bos_mask_data, cummean_data, cumrstd_data, gamma_data,                 \
    padding_mask_data, x_grad_data, prev_mean_grad_data,                   \
    prev_var_grad_data, gamma_grad_data, beta_grad_data, count_array_data, \
    group_mean_data, ds_data, db_data, du_data, dv_data, beta1, beta2)    \
  do {                                                                      \
    constexpr int64_t kCountThreads = 256;                                  \
    const int64_t count_blocks = utils::DivUp((B), kCountThreads);          \
    ComputeCountCUBKernel<<<count_blocks, kCountThreads, 0, cuda_stream>>>( \
        B, L, bos_mask_data, padding_mask_data, prev_count_data,            \
        count_array_data);                                                  \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                         \
    cuda_utils::LaunchKernel(                                               \
        RowwiseInternalGradientsCUBKernel<T, T_ACC, kCUBBwdBlockSize>,      \
        dim3((B), (G)), dim3(kCUBBwdBlockSize), 0, cuda_stream,             \
        B, G, Dg, L, y_grad_data, x_data, gamma_data, cummean_data,         \
        padding_mask_data, group_mean_data, ds_data, db_data);              \
    cuda_utils::LaunchKernel(                                               \
        ColwiseInternalGradientsCUBKernel<                                  \
            T, T_ACC, kCUBBwdBlockSize, STATIC_G>,                          \
        dim3((B), (G)), dim3(kCUBBwdBlockSize), 0, cuda_stream,             \
        B, G, L, beta1, beta2, mean_grad_data, var_grad_data,               \
        count_array_data, bos_mask_data, cumrstd_data, padding_mask_data,   \
        ds_data, db_data, prev_mean_grad_data, prev_var_grad_data,          \
        du_data, dv_data);                                                  \
    const int64_t bwd_shm_size = sizeof(T_ACC) * (G) * 4;                   \
    cuda_utils::LaunchKernel(                                               \
        GroupTimestepDecayNormCUBBwdKernel<                                 \
            T, T_ACC, STATIC_G, STATIC_DG>,                                 \
        dim3((L), (B)), cuda_utils::kCUDANumThreads, bwd_shm_size,          \
        cuda_stream, L, H, G, y_grad_data, x_data, group_mean_data,         \
        cumrstd_data, gamma_data, padding_mask_data, du_data, dv_data,      \
        x_grad_data);                                                       \
    if ((L) < cuda_utils::kColwiseThreshold) {                              \
      const int64_t param_blocks =                                          \
          utils::DivUp((H), int64_t(kParamGradBlockSize));                  \
      GroupGammaBetaCUBBwdSmallKernel<T, T_ACC, STATIC_G, STATIC_DG>        \
          <<<param_blocks, kParamGradBlockSize, 0, cuda_stream>>>(          \
              B, L, H, G, Dg, y_grad_data, x_data, cummean_data,            \
              cumrstd_data, padding_mask_data, gamma_grad_data,             \
              beta_grad_data);                                              \
    } else {                                                                \
      const int64_t param_blocks =                                          \
          utils::DivUp((H), int64_t(cuda_utils::kWarpSize));                \
      GroupGammaBetaCUBBwdLargeKernel<T, T_ACC, STATIC_G, STATIC_DG>        \
          <<<param_blocks,                                                  \
             dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize),            \
             0,                                                             \
             cuda_stream>>>(                                                \
              B, L, H, G, Dg, y_grad_data, x_data, cummean_data,            \
              cumrstd_data, padding_mask_data, gamma_grad_data,             \
              beta_grad_data);                                              \
    }                                                                       \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                         \
  } while (false)

#define DISPATCH_TSDN_CUB_BWD(                                              \
    T, T_ACC, cuda_stream, B, L, H, G, Dg, ...)                            \
  do {                                                                      \
    if (G == 2 && Dg == 3) {                                                \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 2, 3, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);       \
    } else if (G == 4 && Dg == 32) {                                        \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 4, 32, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);      \
    } else if (G == 32 && Dg == 8) {                                        \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 32, 8, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);      \
    } else if (G == 32 && Dg == 32) {                                       \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 32, 32, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);     \
    } else if (G == 64 && Dg == 4) {                                        \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 64, 4, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);      \
    } else if (G == 64 && Dg == 16) {                                       \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 64, 16, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);     \
    } else if (G == 128 && Dg == 8) {                                       \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 128, 8, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);     \
    } else if (G == 256 && Dg == 4) {                                       \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 256, 4, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);     \
    } else if (G == 32 && Dg == 64) {                                       \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 32, 64, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);     \
    } else if (G == 64 && Dg == 32) {                                       \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 64, 32, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);     \
    } else if (G == 128 && Dg == 16) {                                      \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 128, 16, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);    \
    } else if (G == 256 && Dg == 8) {                                       \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 256, 8, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);     \
    } else if (G == 32 && Dg == 128) {                                      \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 32, 128, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);    \
    } else if (G == 64 && Dg == 64) {                                       \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 64, 64, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);     \
    } else if (G == 128 && Dg == 32) {                                      \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 128, 32, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);    \
    } else if (G == 256 && Dg == 16) {                                      \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 256, 16, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);    \
    } else if (Dg == 128) {                                                 \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 0, 128, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);     \
    } else if (Dg == 64) {                                                  \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 0, 64, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);      \
    } else if (Dg == 32) {                                                  \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 0, 32, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);      \
    } else if (Dg == 16) {                                                  \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 0, 16, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);      \
    } else if (Dg == 8) {                                                   \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 0, 8, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);       \
    } else if (Dg == 4) {                                                   \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 0, 4, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);       \
    } else {                                                                \
      TSDN_CUB_BWD_CASE(                                                    \
          T, T_ACC, 0, 0, cuda_stream, B, L, H, G, Dg, __VA_ARGS__);       \
    }                                                                       \
  } while (false)

template <typename T>
void GroupTimestepDecayNormCubCUDAFwdImpl(
    const torch::Tensor& x, const torch::Tensor& bos_mask,
    const torch::Tensor& prev_count, const torch::Tensor& prev_mean,
    const torch::Tensor& prev_var, const torch::Tensor& gamma,
    const torch::Tensor& beta, const torch::Tensor& padding_mask,
    torch::Tensor& group_mean, torch::Tensor& group_var, int64_t num_groups,
    double beta1, double beta2, double eps, torch::Tensor& y,
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
  const T* prev_mean_data = prev_mean.data_ptr<T>();
  const T* prev_var_data = prev_var.data_ptr<T>();
  const T* gamma_data = gamma.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* y_data = y.data_ptr<T>();
  int64_t* count_data = count.data_ptr<int64_t>();
  T* mean_data = mean.data_ptr<T>();
  T* var_data = var.data_ptr<T>();
  T_ACC* cummean_data = cummean.data_ptr<T_ACC>();
  T_ACC* cumrstd_data = cumrstd.data_ptr<T_ACC>();
  T_ACC* group_mean_data = group_mean.data_ptr<T_ACC>();
  T_ACC* group_var_data = group_var.data_ptr<T_ACC>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_TSDN_CUB_FWD(
      T, T_ACC, cuda_stream, B, L, H, G, Dg, x_data, bos_mask_data,
      padding_mask_data, prev_count_data, prev_mean_data, prev_var_data,
      gamma_data, beta_data, group_mean_data, group_var_data, y_data,
      count_data, mean_data, var_data, cummean_data, cumrstd_data, beta1,
      beta2, eps);
}

template <typename T>
void GroupTimestepDecayNormCubCUDABwdImpl(
    const torch::Tensor& y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& x,
    const torch::Tensor& prev_count, const torch::Tensor& bos_mask,
    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma, const torch::Tensor& padding_mask,
    torch::Tensor& count_array, torch::Tensor& group_mean, torch::Tensor& ds,
    torch::Tensor& db, torch::Tensor& du, torch::Tensor& dv,
    int64_t num_groups, double beta1, double beta2, torch::Tensor& x_grad,
    torch::Tensor& prev_mean_grad, torch::Tensor& prev_var_grad,
    torch::Tensor& gamma_grad, torch::Tensor& beta_grad) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = x.size(0);
  const int64_t L = x.size(1);
  const int64_t H = x.size(2);
  const int64_t G = num_groups;
  const int64_t Dg = H / G;

  const T* y_grad_data = y_grad.data_ptr<T>();
  const T* mean_grad_data = mean_grad.data_ptr<T>();
  const T* var_grad_data = var_grad.data_ptr<T>();
  const T* x_data = x.data_ptr<T>();
  const int64_t* prev_count_data = prev_count.data_ptr<int64_t>();
  const bool* bos_mask_data = bos_mask.defined() ? bos_mask.data_ptr<bool>() : nullptr;
  const T_ACC* cummean_data = cummean.data_ptr<T_ACC>();
  const T_ACC* cumrstd_data = cumrstd.data_ptr<T_ACC>();
  const T* gamma_data = gamma.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* x_grad_data = x_grad.data_ptr<T>();
  T* prev_mean_grad_data = prev_mean_grad.data_ptr<T>();
  T* prev_var_grad_data = prev_var_grad.data_ptr<T>();
  T* gamma_grad_data = gamma_grad.data_ptr<T>();
  T* beta_grad_data = beta_grad.data_ptr<T>();

  int64_t* count_array_data = count_array.data_ptr<int64_t>();
  T_ACC* group_mean_data = group_mean.data_ptr<T_ACC>();
  T_ACC* ds_data = ds.data_ptr<T_ACC>();
  T_ACC* db_data = db.data_ptr<T_ACC>();
  T_ACC* du_data = du.data_ptr<T_ACC>();
  T_ACC* dv_data = dv.data_ptr<T_ACC>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_TSDN_CUB_BWD(
      T, T_ACC, cuda_stream, B, L, H, G, Dg, y_grad_data, mean_grad_data,
      var_grad_data, x_data, prev_count_data, bos_mask_data, cummean_data,
      cumrstd_data, gamma_data, padding_mask_data, x_grad_data,
      prev_mean_grad_data, prev_var_grad_data, gamma_grad_data, beta_grad_data,
      count_array_data, group_mean_data, ds_data, db_data, du_data, dv_data,
      beta1, beta2);
}

#undef TSDN_CUB_FWD_CASE
#undef DISPATCH_TSDN_CUB_FWD
#undef TSDN_CUB_BWD_CASE
#undef DISPATCH_TSDN_CUB_BWD

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
GroupTimestepDecayNormCubCUDAFwd(
    const torch::Tensor& x, const c10::optional<torch::Tensor>& bos_mask,
    const torch::Tensor& prev_count, const torch::Tensor& prev_mean,
    const torch::Tensor& prev_var, const torch::Tensor& gamma,
    const torch::Tensor& beta, const c10::optional<torch::Tensor>& padding_mask,
    int64_t num_groups, double beta1, double beta2, double eps) {
  CheckTimestepDecayNormCubFwdInputs(
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
  torch::Tensor mean = torch::empty_like(
      prev_mean, prev_mean.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor var = torch::empty_like(
      prev_var, prev_var.options().memory_format(at::MemoryFormat::Contiguous));

  const auto acc_type = at::toAccumulateType(x.scalar_type(), true);
  torch::Tensor cummean =
      torch::empty({B, num_groups, L},
                   x.options().dtype(acc_type).memory_format(
                       at::MemoryFormat::Contiguous));
  torch::Tensor cumrstd =
      torch::empty({B, num_groups, L},
                   x.options().dtype(acc_type).memory_format(
                       at::MemoryFormat::Contiguous));
  torch::Tensor group_mean =
      torch::empty({B, num_groups, L},
                   x.options().dtype(acc_type).memory_format(
                       at::MemoryFormat::Contiguous));
  torch::Tensor group_var =
      torch::empty({B, num_groups, L},
                   x.options().dtype(acc_type).memory_format(
                       at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(),
      "GroupTimestepDecayNormCubCUDAFwd", [&]() {
        const torch::Tensor x_contig = *(x.expect_contiguous());
        const torch::Tensor bos_mask_contig =
            *(bos_mask_maybe_owned->expect_contiguous());
        const torch::Tensor prev_count_contig = *(prev_count.expect_contiguous());
        const torch::Tensor prev_mean_contig = *(prev_mean.expect_contiguous());
        const torch::Tensor prev_var_contig = *(prev_var.expect_contiguous());
        const torch::Tensor gamma_contig = *(gamma.expect_contiguous());
        const torch::Tensor beta_contig = *(beta.expect_contiguous());
        const torch::Tensor padding_mask_contig =
            *(padding_mask_maybe_owned->expect_contiguous());

        GroupTimestepDecayNormCubCUDAFwdImpl<scalar_t>(
            x_contig, bos_mask_contig, prev_count_contig, prev_mean_contig,
            prev_var_contig, gamma_contig, beta_contig, padding_mask_contig,
            group_mean, group_var, num_groups, beta1, beta2, eps, y, count,
            mean, var, cummean, cumrstd);
      });

  return std::make_tuple(y, count, mean, var, cummean, cumrstd);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
GroupTimestepDecayNormCubCUDABwd(
    const torch::Tensor& y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& x,
    const torch::Tensor& prev_count, const c10::optional<torch::Tensor>& bos_mask,
    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups,
    double beta1, double beta2) {
  CheckTimestepDecayNormCubBwdInputs(
      y_grad, mean_grad, var_grad, x, prev_count, bos_mask, cummean, cumrstd,
      gamma, padding_mask, num_groups, beta1, beta2);

  const int64_t B = x.size(0);
  const int64_t L = x.size(1);

  c10::MaybeOwned<torch::Tensor> bos_mask_maybe_owned =
      at::borrow_from_optional_tensor(bos_mask);
  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor x_grad =
      torch::empty_like(x, x.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor prev_mean_grad = torch::empty_like(
      mean_grad, mean_grad.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor prev_var_grad = torch::empty_like(
      var_grad, var_grad.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor gamma_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor beta_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));

  const auto acc_type = at::toAccumulateType(x.scalar_type(), true);
  torch::Tensor count_array =
      torch::empty({B, L}, prev_count.options().memory_format(
                               at::MemoryFormat::Contiguous));
  torch::Tensor group_mean =
      torch::empty({B, num_groups, L},
                   x.options().dtype(acc_type).memory_format(
                       at::MemoryFormat::Contiguous));
  torch::Tensor ds =
      torch::empty({B, num_groups, L},
                   x.options().dtype(acc_type).memory_format(
                       at::MemoryFormat::Contiguous));
  torch::Tensor db =
      torch::empty({B, num_groups, L},
                   x.options().dtype(acc_type).memory_format(
                       at::MemoryFormat::Contiguous));
  torch::Tensor du =
      torch::empty({B, num_groups, L},
                   x.options().dtype(acc_type).memory_format(
                       at::MemoryFormat::Contiguous));
  torch::Tensor dv =
      torch::empty({B, num_groups, L},
                   x.options().dtype(acc_type).memory_format(
                       at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(),
      "GroupTimestepDecayNormCubCUDABwd", [&]() {
        GroupTimestepDecayNormCubCUDABwdImpl<scalar_t>(
            *(y_grad.expect_contiguous()), *(mean_grad.expect_contiguous()),
            *(var_grad.expect_contiguous()), *(x.expect_contiguous()),
            *(prev_count.expect_contiguous()),
            *(bos_mask_maybe_owned->expect_contiguous()),
            *(cummean.expect_contiguous()), *(cumrstd.expect_contiguous()),
            *(gamma.expect_contiguous()),
            *(padding_mask_maybe_owned->expect_contiguous()), count_array,
            group_mean, ds, db, du, dv, num_groups, beta1, beta2, x_grad,
            prev_mean_grad, prev_var_grad, gamma_grad, beta_grad);
      });

  return std::make_tuple(x_grad, prev_mean_grad, prev_var_grad, gamma_grad,
                         beta_grad);
}

}  // namespace ops
}  // namespace xllm
