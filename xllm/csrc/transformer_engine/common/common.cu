/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>

#include <bit>

#include "./common.h"
#include "./utils.cuh"
#include "common/util/cuda_runtime.h"
#include "common/util/logging.h"

namespace transformer_engine {

namespace {

__global__ void __launch_bounds__(1)
    update_tensor_scale_inv_kernel(const float *__restrict__ scale_ptr,
                                   float *__restrict__ scale_inv_ptr) {
  const float scale = scale_ptr == nullptr ? 1 : *scale_ptr;
  reciprocal<float>(scale_inv_ptr, scale);
}

}  // namespace

void update_tensor_scale_inv(Tensor *t, cudaStream_t stream) {
  if (is_fp8_dtype(t->data.dtype) && is_tensor_scaling(t->scaling_mode)) {
    NVTE_CHECK(t->scale_inv.dptr != nullptr, "Tensor should have allocated scale_inv.");
    update_tensor_scale_inv_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<const float *>(t->scale.dptr),
        reinterpret_cast<float *>(t->scale_inv.dptr));
  }
}

namespace {

constexpr size_t kThreadsPerBlock = 256;
template <typename TVectorized>
__global__ void __launch_bounds__(kThreadsPerBlock)
    memset_kernel(void *__restrict__ ptr, int value, size_t size_in_bytes) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx * sizeof(TVectorized) >= size_in_bytes) {
    return;  // Out of bounds
  }

  if ((idx + 1) * sizeof(TVectorized) > size_in_bytes) {
    // If the buffer size is not an even multiple of the vectorization, manually set the remaining bytes unvectorized.
    size_t remaining_bytes = size_in_bytes - idx * sizeof(TVectorized);
    memset(reinterpret_cast<uint8_t *>(ptr) + idx * sizeof(TVectorized), value, remaining_bytes);
    return;
  }

  union {
    TVectorized value;
    uint8_t data[sizeof(TVectorized)];
  } data;
  for (size_t i = 0; i < sizeof(TVectorized); ++i) {
    data.data[i] = static_cast<uint8_t>(value);
  }
  reinterpret_cast<TVectorized *>(ptr)[idx] = data.value;
}

}  // namespace

#define MEMSET_VECTORIZED_KERNEL_DISPATCH(ptr, size_in_bytes, value, vectorizedType, stream) \
  if (size_in_bytes >= sizeof(vectorizedType) &&                                             \
      reinterpret_cast<size_t>(ptr) % sizeof(vectorizedType) == 0) {                         \
    size_t numBlocks = DIVUP(size_in_bytes, kThreadsPerBlock * sizeof(vectorizedType));      \
    dim3 grid(numBlocks, 1, 1);                                                              \
    memset_kernel<vectorizedType>                                                            \
        <<<grid, kThreadsPerBlock, 0, stream>>>(ptr, value, size_in_bytes);                  \
    return;                                                                                  \
  }

extern "C" {
void nvte_memset(void *ptr, int value, size_t size_in_bytes, cudaStream_t stream) {
  NVTE_API_CALL(nvte_memset);
  NVTE_CHECK(ptr != nullptr, "Pointer for memset must be allocated.");

  if (size_in_bytes > 4096) {
    // Use cudaMemsetAsync for larger sizes.
    cudaMemsetAsync(ptr, value, size_in_bytes, stream);
    return;
  }

  MEMSET_VECTORIZED_KERNEL_DISPATCH(ptr, size_in_bytes, value, float4, stream);
  MEMSET_VECTORIZED_KERNEL_DISPATCH(ptr, size_in_bytes, value, float2, stream);
  MEMSET_VECTORIZED_KERNEL_DISPATCH(ptr, size_in_bytes, value, float, stream);
  MEMSET_VECTORIZED_KERNEL_DISPATCH(ptr, size_in_bytes, value, uint8_t, stream);
}
}  // extern "C"

size_t get_buffer_size_bytes(const size_t elements_num, const DType buffer_dtype) {
  return (elements_num * typeToNumBits(buffer_dtype)) / 8;
}

size_t get_buffer_size_bytes(const size_t dim_first, const size_t dim_last,
                             const DType buffer_dtype) {
  if (buffer_dtype == DType::kFloat4E2M1) {
    NVTE_CHECK(dim_last % 2 == 0,
               "Last dimension of a tensor with FP4 type of data must be an even number!");
  }
  const size_t elements_num = dim_first * dim_last;
  return get_buffer_size_bytes(elements_num, buffer_dtype);
}

}  // namespace transformer_engine
