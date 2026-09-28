#include <ATen/AccumulateType.h>
#include <ATen/DeviceGuard.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/accumulate.h>
#include <thrust/pair.h>

#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/block_reduce.cuh>
#include <limits>

#include "../cuda_utils.cuh"
#include "group_rms_norm.h"
#include "../reduce.cuh"
#include "../register_utils.cuh"
#include "../utils.h"

namespace xllm {
namespace ops {

namespace {

// Shared arrays use the kernel's capacity, even when the actual group is smaller.
constexpr int64_t kMinGroupCapacity = 128;

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

template <typename T, typename T_ACC, int64_t kNumChannelsPerGroup, int64_t kNumThreadsX, bool kElementwiseAffine>
__global__ void GroupRMSNormFwdKernel(
    int64_t inner_size, int64_t channels_per_group, const T* __restrict__ x,
    const T* __restrict__ gamma, T_ACC eps, T* __restrict__ y, T_ACC* __restrict__ rstd) {

  int64_t kNumGroups = gridDim.y;
  constexpr int64_t kElementsPerThread = kNumChannelsPerGroup / kNumThreadsX;
  const int64_t r = blockIdx.x;
  const int64_t c = blockIdx.y;
  extern __shared__ float shm[];

  // size is 32 * num_groups * sizeof(T_ACC) (warp size)
  T_ACC* m_shared = reinterpret_cast<T_ACC*>(shm);
  // size is a blocksize * sizeof(T_ACC)
  T_ACC* w_shared = m_shared + cuda_utils::kWarpSize;
  // each thread will get kNumChannelsPerGroup / kNumThreadsX elements
  T_ACC x_acc[kElementsPerThread];

  if constexpr (kElementwiseAffine) {
    register_utils::Load<T, T_ACC, kElementsPerThread>(gamma + c * channels_per_group, channels_per_group, channels_per_group, T_ACC(0), x_acc);

#pragma unroll
    for (int64_t i = 0; i < kElementsPerThread; ++i) {
      const int64_t idx = i * kNumThreadsX + threadIdx.x;
      w_shared[idx] = x_acc[i];
    }
    __syncthreads();
  }

  register_utils::Load<T, T_ACC, kElementsPerThread>(
      x + r * inner_size + c * channels_per_group, channels_per_group, channels_per_group, std::numeric_limits<T_ACC>::infinity(), x_acc
  );

  // For RMS normalization, we only need the second moment
  T_ACC m2 = T_ACC(0);

#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    m2 += isinf(x_acc[i]) ? T_ACC(0) : cuda_utils::Square<T_ACC>(x_acc[i]);
  }
  m2 = reduce::AllReduce(m2, m_shared, reduce::SumOp<T_ACC>());
  m2 *= T_ACC(1) / static_cast<T_ACC>(channels_per_group);
  
  // Calculate reciprocal standard deviation
  m2 = c10::cuda::compat::rsqrt(m2 + eps);

  if (threadIdx.x == 0) {
    rstd[r * kNumGroups + c] = m2;
  }

#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * kNumThreadsX + threadIdx.x;
    if constexpr (kElementwiseAffine) {
      x_acc[i] = x_acc[i] * m2 * w_shared[idx];
    } else {
      x_acc[i] = x_acc[i] * m2;
    }
  }
  register_utils::Save<T_ACC, T, kElementsPerThread>(
      x_acc, channels_per_group, y + r * inner_size + c * channels_per_group
  );
}

template <typename T, typename T_ACC, int64_t kNumChannelsPerGroup, int64_t kNumThreadsX,
          bool kElementwiseAffine, bool kMemoryEfficient>
__global__ void GroupRMSNormBwdKernel(
    int64_t inner_size, int64_t channels_per_group,  // Actual size that may be smaller than kNumChannelsPerGroup
    const T* __restrict__ y_grad, const T* __restrict__ x_or_y,
    const T_ACC* __restrict__ rstd, const T* __restrict__ gamma,
    T* __restrict__ x_grad) {

  constexpr int64_t kElementsPerThread = kNumChannelsPerGroup / kNumThreadsX;
  const int64_t r = blockIdx.x;    // batch index
  const int64_t c = blockIdx.y;    // group index (< 50)
  int64_t kNumGroups = gridDim.y;

  extern __shared__ float shm[];

  T_ACC* ds_shared = reinterpret_cast<T_ACC*>(shm);
  T_ACC* dy_shared = ds_shared + cuda_utils::kWarpSize;
  T_ACC* x_shared = dy_shared + kNumChannelsPerGroup;
  T_ACC w_acc[kElementsPerThread];

  register_utils::Load<T, T_ACC, kElementsPerThread>(
      y_grad + r * inner_size + c * channels_per_group, channels_per_group, channels_per_group, T_ACC(0), w_acc
  );
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * kNumThreadsX + threadIdx.x;
    dy_shared[idx] = w_acc[i];
  }
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      x_or_y + r * inner_size + c * channels_per_group, channels_per_group, channels_per_group, T_ACC(0), w_acc
  );
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * kNumThreadsX + threadIdx.x;
    x_shared[idx] = w_acc[i];
  }

  if (kElementwiseAffine) {
    register_utils::Load<T, T_ACC, kElementsPerThread>(
        gamma + c * channels_per_group, channels_per_group, channels_per_group, T_ACC(0), w_acc
    );
  }

  __syncthreads();

  const T_ACC v = rstd[r * kNumGroups + c];
  T_ACC ds = T_ACC(0);
  const T_ACC inv_v = T_ACC(1) / v;

  if (kMemoryEfficient) {
    constexpr T_ACC eps = std::is_same<T_ACC, float>::value ? T_ACC(1e-12) : T_ACC(1e-6);
    // Compute gradients without reconstructing x
#pragma unroll
    for (int64_t i = 0; i < kElementsPerThread; ++i) {
      const int64_t idx = i * kNumThreadsX + threadIdx.x;
      const T_ACC dy_acc = dy_shared[idx];
      const T_ACC y_acc = x_shared[idx];  // Note: x_or_y contains y values in memory efficient mode
      
      if (kElementwiseAffine) {
        const T_ACC w = w_acc[i];  // gamma
        const T_ACC x = y_acc / (w * v + eps);
        ds += dy_acc * w * x;
      } else {
        const T_ACC x = y_acc * inv_v;
        ds += dy_acc * x;
      }
    }
  } else {
#pragma unroll
    for (int64_t i = 0; i < kElementsPerThread; ++i) {
      const int64_t idx = i * kNumThreadsX + threadIdx.x;
      const T_ACC dy_acc = dy_shared[idx];
      const T_ACC x_acc = x_shared[idx];
      if (kElementwiseAffine) {
        const T_ACC w = w_acc[i];
        ds += dy_acc * w * x_acc;
      } else {
        ds += dy_acc * x_acc;
      }
    }
  }

  ds = reduce::AllReduce(ds, ds_shared, reduce::SumOp<T_ACC>());

  const T_ACC coef = T_ACC(1) / static_cast<T_ACC>(channels_per_group);
  ds *= coef;
  const T_ACC dv = -cuda_utils::Cube(v) * ds;

#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * kNumThreadsX + threadIdx.x;
    if (kMemoryEfficient) {
      if (kElementwiseAffine) {
        const T_ACC gamma_val = w_acc[i];
        const T_ACC x = x_shared[idx] / (gamma_val * v);
        w_acc[i] = v * gamma_val * dy_shared[idx] + x * dv;
      } else {
        const T_ACC x = x_shared[idx] * inv_v;
        w_acc[i] = v * dy_shared[idx] + x * dv;
      }
    } else {
      if (kElementwiseAffine) {
        w_acc[i] = v * w_acc[i] * dy_shared[idx] + x_shared[idx] * dv;
      } else {
        w_acc[i] = v * dy_shared[idx] + x_shared[idx] * dv;
      }
    }
  }

  register_utils::Save<T_ACC, T, kElementsPerThread>(w_acc, channels_per_group, x_grad + r * inner_size + c * channels_per_group);
}

// version using 2d thread blocks
template <typename T, typename T_ACC, int64_t kBlockSize, int64_t kNumChannelsPerGroup,
          int64_t kNumThreadsX, int64_t kNumThreadsY, bool kMemoryEfficient>
__global__ void GammaBwdSmallKernel(
    int64_t outer_size, int64_t inner_size, const T* y_grad, const T* x_or_y,
    const T_ACC* rstd, T* gamma_grad, const T* __restrict__ gamma) {
  constexpr int64_t kElementsPerThread = kNumChannelsPerGroup / kNumThreadsX;
  T_ACC dy_acc[kElementsPerThread];
  T_ACC x_acc[kElementsPerThread];
  T_ACC sum1[kElementsPerThread];
  const int64_t c = threadIdx.y;  // Group ID
  const int64_t kNumGroups = threadIdx.y;
  const int64_t group_offset = c * kNumChannelsPerGroup;

#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    sum1[i] = T_ACC(0);
  }

  // Calculate starting position for this thread block
  const int64_t j = blockIdx.x * kBlockSize;
  const int64_t cur_size = std::min(kBlockSize, kNumChannelsPerGroup - j);

  // Accumulate gradients across batch dimension
  for (int64_t i = 0; i < outer_size; ++i) {
    const T_ACC v = rstd[i * kNumGroups + c];

    // Load gradient and input/output data for this batch
    register_utils::Load<T, T_ACC, kElementsPerThread>(y_grad + i * inner_size + group_offset + j, cur_size, cur_size, T_ACC(0), dy_acc);

    if (kMemoryEfficient) {
      // Load y values and reconstruct x
      register_utils::Load<T, T_ACC, kElementsPerThread>(x_or_y + i * inner_size + group_offset + j, cur_size, cur_size, T_ACC(0), x_acc);
          
#pragma unroll
      for (int64_t k = 0; k < kElementsPerThread; ++k) {
        // Reconstruct x from y: x = y/(gamma * v)
        const T_ACC gamma_val = gamma[group_offset + j + k];
        const T_ACC x_val = x_acc[k]/(gamma_val * v);
        
        sum1[k] += dy_acc[k] * x_val * v;
      }
    } else {
      // Direct calculation using stored x values
      register_utils::Load<T, T_ACC, kElementsPerThread>(x_or_y + i * inner_size + group_offset + j, cur_size, cur_size, T_ACC(0), x_acc);

#pragma unroll
      for (int64_t k = 0; k < kElementsPerThread; ++k) {
        sum1[k] += dy_acc[k] * x_acc[k] * v;
      }
    }
  }

  // Save accumulated gradients
  register_utils::Save<T_ACC, T, kElementsPerThread>(sum1, cur_size, gamma_grad + group_offset + j);
}

template <typename T, typename T_ACC, bool kMemoryEfficient>
__global__ void GammaBwdLargeKernel(
    int64_t outer_size, int64_t inner_size, const T* y_grad, const T* x_or_y,
    const T_ACC* rstd, T* gamma_grad, const T* __restrict__ gamma) {

  __shared__ T_ACC sum1_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];

  const int64_t kNumGroups = gridDim.y;
  const int64_t num_channels_per_group = inner_size / kNumGroups;
  const int64_t group_id = blockIdx.y;
  const int64_t group_offset = group_id * num_channels_per_group;
  
  // Calculate channel index with bounds checking
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  const bool valid_channel = j < num_channels_per_group;
  
  T_ACC sum1 = T_ACC(0);

  // Use inverse of gamma for multiplication instead of division
  T_ACC inv_w = T_ACC(1);

  if (kMemoryEfficient) {
    // Store the inverse for better performance
    inv_w = T_ACC(1) / gamma[j + group_offset];
  }

  // Iterate over batch dimension with bounds checking
  for (int64_t i = threadIdx.y; i < outer_size; i += blockDim.y) {
    if (valid_channel) {
      const T_ACC dy_acc = static_cast<T_ACC>(y_grad[i * inner_size + j + group_offset]);
      const T_ACC v = rstd[i * kNumGroups + group_id];

      if (kMemoryEfficient) {
        const T_ACC y = static_cast<T_ACC>(x_or_y[i * inner_size + j + group_offset]);
        // Replace division with multiplication by inverse
        sum1 += dy_acc * (y * inv_w);
      } else {
        const T_ACC x_acc = static_cast<T_ACC>(x_or_y[i * inner_size + j + group_offset]);
        sum1 += dy_acc * x_acc * v;
      }
    }
  }

  // Store intermediate results in shared memory
  sum1_shared[threadIdx.x][threadIdx.y] = sum1;
  __syncthreads();

  // Transpose and reduce
  sum1 = sum1_shared[threadIdx.y][threadIdx.x];
  sum1 = reduce::WarpReduce(sum1, reduce::SumOp<T_ACC>());

  // Write results with bounds checking
  if (threadIdx.x == 0) {
    const int64_t k = blockIdx.x * blockDim.x + threadIdx.y;
    if (k < num_channels_per_group) {
      gamma_grad[k + group_offset] = sum1;
    }
  }
}

// channels_per_group is the inner_size used in layernorm since grouplayernorm is a
// special case of layernorm. What we do is to have a thread for each group. So
// we launch a 2d block with (kNumThreadsX, kNumThreadsY) threads per block.
#define COMMA ,

#define DISPATCH_GROUP_RMS_NORM_CUDA_KERNEL_IMPL(                                                        \
    KernelFunc, ExtraTemplateParams, T, T_ACC, outer_size, inner_size, num_groups,                       \
    channels_per_group, shm_size, elementwise_affine, cuda_stream, ...)                                  \
  do {                                                                                                   \
    if (channels_per_group <= kMinGroupCapacity) {                                                       \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, kMinGroupCapacity, 32, elementwise_affine ExtraTemplateParams>, \
                               dim3(outer_size, num_groups), dim3(32), shm_size, cuda_stream,            \
                               __VA_ARGS__);                                                             \
    } else if (channels_per_group <= 256) {                                                              \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 256, 32, elementwise_affine ExtraTemplateParams>,    \
                               dim3(outer_size, num_groups), dim3(32), shm_size, cuda_stream,            \
                               __VA_ARGS__);                                                             \
    } else if (channels_per_group <= 512) {                                                              \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 512, 64, elementwise_affine ExtraTemplateParams>,    \
                               dim3(outer_size, num_groups), dim3(64), shm_size, cuda_stream,            \
                               __VA_ARGS__);                                                             \
    } else if (channels_per_group <= 1024) {                                                             \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1024, 128, elementwise_affine ExtraTemplateParams>,  \
                               dim3(outer_size, num_groups), dim3(128), shm_size, cuda_stream,           \
                               __VA_ARGS__);                                                             \
    } else if (channels_per_group <= 2048) {                                                             \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2048, 256, elementwise_affine ExtraTemplateParams>,  \
                               dim3(outer_size, num_groups), dim3(256), shm_size, cuda_stream,           \
                               __VA_ARGS__);                                                             \
    } else if (channels_per_group <= 4096) {                                                             \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4096, 512, elementwise_affine ExtraTemplateParams>,  \
                               dim3(outer_size, num_groups), dim3(512), shm_size, cuda_stream,           \
                               __VA_ARGS__);                                                             \
    } else if (channels_per_group <= 8192) {                                                             \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8192, 1024, elementwise_affine ExtraTemplateParams>, \
                               dim3(outer_size, num_groups), dim3(1024), shm_size, cuda_stream,          \
                               __VA_ARGS__);                                                             \
    } else {                                                                                             \
      TORCH_CHECK(false, "channels_per_group too large: ", channels_per_group);                          \
    }                                                                                                    \
  } while (false)

#define DISPATCH_GROUP_RMS_NORM_CUDA_KERNEL_FWD(                                     \
    KernelFunc, T, T_ACC, outer_size, inner_size, num_groups, channels_per_group,    \
    shm_size, elementwise_affine, cuda_stream, ...)                                  \
    DISPATCH_GROUP_RMS_NORM_CUDA_KERNEL_IMPL(                                        \
        KernelFunc, /* empty */, T, T_ACC, outer_size, inner_size, num_groups,       \
        channels_per_group, shm_size, elementwise_affine, cuda_stream, __VA_ARGS__)

#define DISPATCH_GROUP_RMS_NORM_CUDA_KERNEL_BWD(                                     \
    KernelFunc, T, T_ACC, outer_size, inner_size, num_groups, channels_per_group,    \
    memory_efficient, elementwise_affine, shm_size, cuda_stream, ...)                \
    DISPATCH_GROUP_RMS_NORM_CUDA_KERNEL_IMPL(                                        \
        KernelFunc, COMMA memory_efficient,  /* Extra template param with comma */   \
        T, T_ACC, outer_size, inner_size, num_groups, channels_per_group, shm_size,  \
        elementwise_affine, cuda_stream, __VA_ARGS__)

// for now we use 2d thread blocks. Each y will handle a different group
#define DISPATCH_GAMMA_BWD_CUDA_KERNEL(                                              \
    SmallKernelFunc, LargeKernelFunc, T, T_ACC, outer_size, inner_size, num_groups,  \
    channels_per_group, memory_efficient, cuda_stream, ...)                          \
  do {                                                                               \
    if (true) {                                                                      \
      const int64_t num_blocks_x = utils::DivUp<int64_t>(channels_per_group, 32);    \
      const int64_t num_blocks_y = num_groups;                                       \
      dim3 grid_size(num_blocks_x, num_blocks_y);                                    \
      cuda_utils::LaunchKernel(LargeKernelFunc<T, T_ACC, memory_efficient>,          \
      grid_size, dim3(32, 32), 0, cuda_stream, __VA_ARGS__);                         \
    }                                                                                \
  } while (false)

template <typename T>
void GroupRMSNormCUDAFwdAffineImpl(
    const torch::Tensor& x, const torch::Tensor& gamma, int64_t outer_size,
    int64_t inner_size, int64_t num_groups, int64_t channels_per_group,
    double eps, torch::Tensor& y, torch::Tensor& rstd) {
  using T_ACC = at::acc_type<T, /*is_cuda=*/true>;

  const T* x_data = x.data_ptr<T>();
  const T* gamma_data = gamma.data_ptr<T>();
  T* y_data = y.data_ptr<T>();
  T_ACC* rstd_data = rstd.data_ptr<T_ACC>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  const int64_t block_size = std::max(int64_t(1) << utils::CeilLog2(channels_per_group), kMinGroupCapacity);
  const int64_t shm_size = sizeof(T_ACC) * cuda_utils::kWarpSize + sizeof(T_ACC) * block_size;

  DISPATCH_GROUP_RMS_NORM_CUDA_KERNEL_FWD(
      GroupRMSNormFwdKernel, T, T_ACC, outer_size, inner_size, num_groups,
      channels_per_group, shm_size, true, cuda_stream,
      inner_size, channels_per_group, x_data, gamma_data,
      static_cast<T_ACC>(eps), y_data, rstd_data);
}

template <typename T>
void GroupRMSNormCUDAFwdImpl(
    const torch::Tensor& x, int64_t outer_size, int64_t inner_size, int64_t num_groups,
    int64_t channels_per_group, double eps, torch::Tensor& y, torch::Tensor& rstd) {
  using T_ACC = at::acc_type<T, /*is_cuda=*/true>;

  const T* x_data = x.data_ptr<T>();
  T* y_data = y.data_ptr<T>();
  T_ACC* rstd_data = rstd.data_ptr<T_ACC>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  const int64_t block_size = std::max(int64_t(1) << utils::CeilLog2(channels_per_group), cuda_utils::kWarpSize);
  const int64_t shm_size = sizeof(T_ACC) * cuda_utils::kWarpSize;

  DISPATCH_GROUP_RMS_NORM_CUDA_KERNEL_FWD(
      GroupRMSNormFwdKernel, T, T_ACC, outer_size, inner_size, num_groups,
      channels_per_group, shm_size, false, cuda_stream,
      inner_size, channels_per_group, x_data, nullptr,
      static_cast<T_ACC>(eps), y_data, rstd_data);
}

template <typename T>
void GroupRMSNormCUDABwdAffineImpl(
    const torch::Tensor& y_grad, const torch::Tensor& x_or_y,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    int64_t outer_size, int64_t inner_size, int64_t num_groups, int64_t channels_per_group,
    torch::Tensor& x_grad, torch::Tensor& gamma_grad, bool memory_efficient) {
  using T_ACC = at::acc_type<T, /*is_cuda=*/true>;

  const T* y_grad_data = y_grad.data_ptr<T>();
  const T* x_or_y_data = x_or_y.data_ptr<T>();
  const T_ACC* rstd_data = rstd.data_ptr<T_ACC>();
  const T* gamma_data = gamma.data_ptr<T>();
  T* x_grad_data = x_grad.data_ptr<T>();
  T* gamma_grad_data = gamma_grad.data_ptr<T>();

  const int64_t block_size = std::max(int64_t(1) << utils::CeilLog2(channels_per_group), kMinGroupCapacity);
  const int64_t shm_size = sizeof(T_ACC) * cuda_utils::kWarpSize + sizeof(T_ACC) * block_size * 2;
  at::cuda::OptionalCUDAGuard guard(at::device_of(x_or_y));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  BOOL_SWITCH(
      memory_efficient, MemoryEfficient,
      DISPATCH_GROUP_RMS_NORM_CUDA_KERNEL_BWD(
          GroupRMSNormBwdKernel, T, T_ACC, outer_size, inner_size, num_groups, channels_per_group,
          MemoryEfficient, true, shm_size, cuda_stream, inner_size, channels_per_group,
          y_grad_data, x_or_y_data, rstd_data, gamma_data, x_grad_data);

      DISPATCH_GAMMA_BWD_CUDA_KERNEL(
          GammaBwdSmallKernel, GammaBwdLargeKernel, T, T_ACC, outer_size,
          inner_size, num_groups, channels_per_group, MemoryEfficient, cuda_stream,
          outer_size, inner_size, y_grad_data, x_or_y_data, rstd_data, gamma_grad_data, gamma_data);
  );
}

template <typename T>
void GroupRMSNormCUDABwdImpl(
    const torch::Tensor& y_grad, const torch::Tensor& x_or_y, const torch::Tensor& rstd,
    int64_t outer_size, int64_t inner_size, int64_t num_groups, int64_t channels_per_group,
    torch::Tensor& x_grad, bool memory_efficient) {
  using T_ACC = at::acc_type<T, /*is_cuda=*/true>;

  const T* y_grad_data = y_grad.data_ptr<T>();
  const T* x_or_y_data = x_or_y.data_ptr<T>();
  const T_ACC* rstd_data = rstd.data_ptr<T_ACC>();
  T* x_grad_data = x_grad.data_ptr<T>();

  const int64_t block_size = std::max(int64_t(1) << utils::CeilLog2(channels_per_group), kMinGroupCapacity);
  const int64_t shm_size = sizeof(T_ACC) * cuda_utils::kWarpSize + sizeof(T_ACC) * block_size * 2;
  at::cuda::OptionalCUDAGuard guard(at::device_of(x_or_y));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  BOOL_SWITCH(
      memory_efficient, MemoryEfficient,
      DISPATCH_GROUP_RMS_NORM_CUDA_KERNEL_BWD(
          GroupRMSNormBwdKernel, T, T_ACC, outer_size, inner_size, num_groups, channels_per_group,
          MemoryEfficient, false, shm_size, cuda_stream, inner_size, channels_per_group,
          y_grad_data, x_or_y_data, rstd_data, nullptr, x_grad_data);
  );
}

#undef DISPATCH_GROUP_RMS_NORM_CUDA_KERNEL
#undef DISPATCH_GAMMA_BWD_CUDA_KERNEL
#undef COMMA


}  // namespace

std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormCUDAFwdAffine(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups,
    const torch::Tensor& gamma, double eps) {

  TORCH_CHECK(num_groups >= 1 && num_groups <= 32);
  TORCH_CHECK(num_channels % num_groups == 0);
  TORCH_CHECK(x.dim() == 3);
  const int64_t channels_per_group = num_channels / num_groups;
  const auto x_sizes = x.sizes();
  const int64_t outer_size = c10::multiply_integers(x_sizes.cbegin(), x_sizes.cbegin() + 2);
  const int64_t inner_size = x_sizes.back();
  const auto acc_type = at::toAccumulateType(x.scalar_type(), true);

  torch::Tensor y = torch::empty_like(x, x.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor rstd = torch::empty({outer_size, num_groups}, x.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));
    
  // dispatch kernel
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "GroupRMSNormCUDAFwdAffine", [&]() {
      GroupRMSNormCUDAFwdAffineImpl<scalar_t>(
          *(x.expect_contiguous()), *(gamma.expect_contiguous()), outer_size,
          inner_size, num_groups, channels_per_group, eps, y, rstd);
    });

  return std::make_tuple(y, rstd);
}

std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormCUDAFwd(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups, double eps) {

  TORCH_CHECK(num_groups >= 1 && num_groups <= 32);
  TORCH_CHECK(num_channels % num_groups == 0);
  TORCH_CHECK(x.dim() == 3);
  const int64_t channels_per_group = num_channels / num_groups;
  const auto x_sizes = x.sizes();
  const int64_t outer_size = c10::multiply_integers(x_sizes.cbegin(), x_sizes.cbegin() + 2);
  const int64_t inner_size = x_sizes.back();
  const auto acc_type = at::toAccumulateType(x.scalar_type(), true);

  torch::Tensor y = torch::empty_like(x, x.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor rstd = torch::empty({outer_size, num_groups}, x.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));

  // dispatch kernel
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "GroupRMSNormCUDAFwd", [&]() {
      GroupRMSNormCUDAFwdImpl<scalar_t>(
          *(x.expect_contiguous()), outer_size, inner_size, num_groups, channels_per_group, eps, y, rstd);
    });

  return std::make_tuple(y, rstd);
}

std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormCUDABwdAffine(
    const torch::Tensor& y_grad, const torch::Tensor& x,
    int64_t num_channels, int64_t num_groups,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    bool memory_efficient) {

  TORCH_CHECK(num_groups >= 1 && num_groups <= 32);
  TORCH_CHECK(num_channels % num_groups == 0);
  TORCH_CHECK(x.dim() == 3);

  const int64_t channels_per_group = num_channels / num_groups;
  const auto x_sizes = x.sizes();
  const int64_t outer_size = c10::multiply_integers(x_sizes.cbegin(), x_sizes.cbegin() + 2);
  const int64_t inner_size = x_sizes.back();

  TORCH_CHECK(rstd.numel() == outer_size*num_groups);

  torch::Tensor x_grad = torch::empty_like(x, x.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor gamma_grad = torch::empty_like(gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));
  
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "GroupRMSNormCUDABwdAffine", [&]() {
        GroupRMSNormCUDABwdAffineImpl<scalar_t>(
            *(y_grad.expect_contiguous()), *(x.expect_contiguous()), *(rstd.expect_contiguous()),
            *(gamma.expect_contiguous()), outer_size, inner_size, num_groups,
            channels_per_group, x_grad, gamma_grad, memory_efficient);
      });

  return std::make_tuple(x_grad, gamma_grad);
}

torch::Tensor GroupRMSNormCUDABwd(
    const torch::Tensor& y_grad, const torch::Tensor& x,
    int64_t num_channels, int64_t num_groups,
    const torch::Tensor& rstd, bool memory_efficient) {

  TORCH_CHECK(num_groups >= 1 && num_groups <= 32);
  TORCH_CHECK(num_channels % num_groups == 0);
  TORCH_CHECK(x.dim() == 3);

  const int64_t channels_per_group = num_channels / num_groups;
  const auto x_sizes = x.sizes();
  const int64_t outer_size = c10::multiply_integers(x_sizes.cbegin(), x_sizes.cbegin() + 2);
  const int64_t inner_size = x_sizes.back();

  TORCH_CHECK(rstd.numel() == outer_size*num_groups);

  torch::Tensor x_grad = torch::empty_like(x, x.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "GroupRMSNormCUDABwd", [&]() {
        GroupRMSNormCUDABwdImpl<scalar_t>(
            *(y_grad.expect_contiguous()), *(x.expect_contiguous()), *(rstd.expect_contiguous()),
            outer_size, inner_size, num_groups, channels_per_group, x_grad, memory_efficient);
      });

  return x_grad;
}

}  // namespace ops
}  // namespace xllm
