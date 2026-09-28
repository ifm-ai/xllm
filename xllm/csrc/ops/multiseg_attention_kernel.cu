#include <ATen/AccumulateType.h>
#include <ATen/DeviceGuard.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/core/MemoryFormat.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

#include "blas.h"
#include "cuda_utils.cuh"
#include "ops/attention.h"
#include "random_utils.cuh"
#include "softmax.cuh"

namespace xllm {
namespace ops {

namespace {

template <typename T>
__global__ void MakeGemmBatchedInputsCUDAKernel(
    int64_t B, int64_t N, const T* a, int64_t outer_stride_a,
    int64_t inner_stride_a, const T* b, int64_t outer_stride_b,
    int64_t inner_stride_b, T* c, int64_t outer_stride_c,
    int64_t inner_stride_c, const T** a_ptr, const T** b_ptr, T** c_ptr) {
  for (int64_t i = threadIdx.x; i < B * N; i += blockDim.x) {
    const int64_t x = i / N;
    const int64_t y = i % N;
    a_ptr[i] = a + x * outer_stride_a + y * inner_stride_a;
    b_ptr[i] = b + x * outer_stride_b + y * inner_stride_b;
    c_ptr[i] = c + x * outer_stride_c + y * inner_stride_c;
  }
}

template <typename T>
void MultiSegAttentionCUDAFwdImpl(const torch::Tensor& q, const torch::Tensor& k,
                          const torch::Tensor& v, const torch::Tensor& q_segment_idx,
                          const torch::Tensor& k_segment_idx, double scale, double dropout,
                          bool use_causal_mask, torch::Tensor& y, torch::Tensor& w) {
  using T_ACC = at::acc_type<T, /*is_cuda=*/true>;

  const int64_t B = q.size(0);
  const int64_t L1 = q.size(1);
  const int64_t L2 = k.size(1);
  const int64_t N = q.size(2);
  const int64_t H1 = q.size(3);
  const int64_t H2 = v.size(3);

  const T* q_data = q.data_ptr<T>();
  const T* k_data = k.data_ptr<T>();
  const T* v_data = v.data_ptr<T>();
  const int64_t* q_segment_idx_data = q_segment_idx.data_ptr<int64_t>();
  const int64_t* k_segment_idx_data = k_segment_idx.data_ptr<int64_t>();
  T* y_data = y.data_ptr<T>();
  T* w_data = w.data_ptr<T>();

  const int64_t ptr_size = B * N * sizeof(T*);
  torch::Tensor a_ptr = torch::empty({ptr_size}, q.options().dtype(at::kByte));
  torch::Tensor b_ptr = torch::empty({ptr_size}, q.options().dtype(at::kByte));
  torch::Tensor c_ptr = torch::empty({ptr_size}, q.options().dtype(at::kByte));
  const T** a_ptr_data = reinterpret_cast<const T**>(a_ptr.data_ptr());
  const T** b_ptr_data = reinterpret_cast<const T**>(b_ptr.data_ptr());
  T** c_ptr_data = reinterpret_cast<T**>(c_ptr.data_ptr());

  at::cuda::OptionalCUDAGuard guard(at::device_of(q));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  if (N == 1) {
    blas::GemmStridedBatchedCUDA<T>(
        handle, blas::TransposeOp::kN, blas::TransposeOp::kT, B, L1, L2, H1,
        static_cast<T_ACC>(scale), q_data, H1, L1 * H1, k_data, H1, L2 * H1,
        /*beta=*/T_ACC(0), w_data, L2, L1 * L2);
  } else {
    MakeGemmBatchedInputsCUDAKernel<T>
        <<<1, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            B, N, q_data, L1 * N * H1, H1, k_data, L2 * N * H1, H1, w_data,
            N * L1 * L2, L1 * L2, a_ptr_data, b_ptr_data, c_ptr_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    blas::GemmBatchedCUDA<T>(
        handle, blas::TransposeOp::kN, blas::TransposeOp::kT, B * N, L1, L2, H1,
        static_cast<T_ACC>(scale), a_ptr_data, N * H1, b_ptr_data, N * H1,
        /*beta=*/T_ACC(0), c_ptr_data, L2);
  }

  constexpr int64_t kShmSize = 0;
  const int64_t batch_size = B * N;
  const int64_t outer_size = L1;
  const int64_t inner_size = L2;
  if (dropout == 0.0) {
    DISPATCH_ATTENTION_SOFTMAX_CUDA_KERNEL(
        softmax::MultiSegAttentionSoftmaxFwdKernel, T, T_ACC,
        dim3(outer_size, batch_size), inner_size, kShmSize, cuda_stream,
        outer_size, inner_size, N, use_causal_mask, w_data, w_data, q_segment_idx_data, k_segment_idx_data);
  } else {
    const int64_t random_capacity =
        std::max(int64_t(1) << utils::CeilLog2(inner_size),
                 cuda_utils::kWarpSize * random_utils::kRandomUnroll);
    auto* gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
    at::PhiloxCudaState rng_engine_inputs;
    {
      std::lock_guard<std::mutex> lock(gen->mutex_);
      rng_engine_inputs = gen->philox_cuda_state(random_capacity);
    }
    DISPATCH_ATTENTION_SOFTMAX_CUDA_KERNEL(
        softmax::MultiSegAttentionDropKeySoftmaxFwdKernel, T, T_ACC,
        dim3(outer_size, batch_size), inner_size, kShmSize, cuda_stream,
        rng_engine_inputs, outer_size, inner_size, N, static_cast<T_ACC>(dropout),
        use_causal_mask, w_data, w_data, q_segment_idx_data, k_segment_idx_data);
  }

  if (N == 1) {
    blas::GemmStridedBatchedCUDA<T>(
        handle, blas::TransposeOp::kN, blas::TransposeOp::kN, B, L1, H2, L2,
        /*alpha=*/T_ACC(1), w_data, L2, L1 * L2, v_data, H2, L2 * H2,
        /*beta=*/T_ACC(0), y_data, H2, L1 * H2);
  } else {
    MakeGemmBatchedInputsCUDAKernel<T>
        <<<1, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            B, N, w_data, N * L1 * L2, L1 * L2, v_data, L2 * N * H2, H2, y_data,
            L1 * N * H2, H2, a_ptr_data, b_ptr_data, c_ptr_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    blas::GemmBatchedCUDA<T>(handle, blas::TransposeOp::kN,
                             blas::TransposeOp::kN, B * N, L1, H2, L2,
                             /*alpha=*/T_ACC(1), a_ptr_data, L2, b_ptr_data,
                             N * H2, /*beta=*/T_ACC(0), c_ptr_data, N * H2);
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> MultiSegAttentionCUDAFwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& q_segment_idx, const torch::Tensor& k_segment_idx,
    double scale, double dropout, bool use_causal_mask) {
  const int64_t B = q.size(0);
  const int64_t L1 = q.size(1);
  const int64_t L2 = k.size(1);
  const int64_t N = q.size(2);
  const int64_t H2 = v.size(3);
  torch::Tensor y = torch::empty(
      {B, L1, N, H2}, v.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor w = torch::empty(
      {B, N, L1, L2}, v.options().memory_format(at::MemoryFormat::Contiguous));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, q.scalar_type(), "MultiSegAttentionCUDAFwd", [&]() {
        MultiSegAttentionCUDAFwdImpl<scalar_t>(
            *(q.expect_contiguous()), *(k.expect_contiguous()),
            *(v.expect_contiguous()),  *(q_segment_idx.expect_contiguous()),
            *(k_segment_idx.expect_contiguous()), scale, dropout, use_causal_mask, y, w);
      });
  return std::make_tuple<torch::Tensor, torch::Tensor>(std::move(y),
                                                       std::move(w));
}

}  // namespace ops
}  // namespace xllm
