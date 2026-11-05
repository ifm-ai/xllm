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

// C[b, n] = alpha op(A[b, n]) op(B[b, n]) + beta C[b, n] over the (batch, head) pairs of [B, L, N, H] style operands
// (outer / inner = batch / head strides). With one batch the heads are a single strided batch (no pointer arrays).
template <typename T>
void HeadBatchedGemm(cublasHandle_t handle, cudaStream_t stream, blas::TransposeOp transa, blas::TransposeOp transb,
                     int64_t B, int64_t N, int64_t m, int64_t n, int64_t k, at::opmath_type<T> alpha, const T* a,
                     int64_t outer_a, int64_t inner_a, int64_t lda, const T* b, int64_t outer_b, int64_t inner_b,
                     int64_t ldb, at::opmath_type<T> beta, T* c, int64_t outer_c, int64_t inner_c, int64_t ldc,
                     const T** a_ptr, const T** b_ptr, T** c_ptr) {
  if (B == 1) {
    blas::GemmStridedBatchedCUDA<T>(handle, transa, transb, N, m, n, k, alpha, a, lda, inner_a, b, ldb, inner_b, beta,
                                    c, ldc, inner_c);
    return;
  }
  MakeGemmBatchedInputsCUDAKernel<T><<<1, cuda_utils::kCUDANumThreads, 0, stream>>>(
      B, N, a, outer_a, inner_a, b, outer_b, inner_b, c, outer_c, inner_c, a_ptr, b_ptr, c_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  blas::GemmBatchedCUDA<T>(handle, transa, transb, B * N, m, n, k, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc);
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
    HeadBatchedGemm<T>(handle, cuda_stream, blas::TransposeOp::kN, blas::TransposeOp::kT, B, N, L1, L2, H1, static_cast<T_ACC>(scale),
                       q_data, L1 * N * H1, H1, N * H1, k_data, L2 * N * H1, H1, N * H1,
                       /*beta=*/T_ACC(0), w_data, N * L1 * L2, L1 * L2, L2, a_ptr_data, b_ptr_data, c_ptr_data);
  }

  constexpr int64_t kShmSize = 0;
  const int64_t batch_size = B * N;
  const int64_t outer_size = L1;
  const int64_t inner_size = L2;
  if (dropout == 0.0) {
    // 256 threads per row for long rows (several 16-byte vectors per thread): more rows in flight than the shared
    // dispatch's 1024-thread blocks, whose two block reductions per row made the kernel latency-bound
    const dim3 grid(outer_size, batch_size);
    if (inner_size > 2048 && inner_size <= 4096) {
      cuda_utils::LaunchKernel(softmax::MultiSegAttentionSoftmaxFwdKernelV2<T, T_ACC, 4096, 256>, grid, 256, kShmSize,
                               cuda_stream, outer_size, inner_size, N, use_causal_mask, w_data, w_data,
                               q_segment_idx_data, k_segment_idx_data);
    } else if (inner_size > 4096 && inner_size <= 8192) {
      cuda_utils::LaunchKernel(softmax::MultiSegAttentionSoftmaxFwdKernelV2<T, T_ACC, 8192, 256>, grid, 256, kShmSize,
                               cuda_stream, outer_size, inner_size, N, use_causal_mask, w_data, w_data,
                               q_segment_idx_data, k_segment_idx_data);
    } else {
      DISPATCH_ATTENTION_SOFTMAX_CUDA_KERNEL(
          softmax::MultiSegAttentionSoftmaxFwdKernelV2, T, T_ACC,
          dim3(outer_size, batch_size), inner_size, kShmSize, cuda_stream,
          outer_size, inner_size, N, use_causal_mask, w_data, w_data, q_segment_idx_data, k_segment_idx_data);
    }
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
        /*beta=*/T_ACC(0), y_data, y.stride(1), y.stride(0));
  } else {
    HeadBatchedGemm<T>(handle, cuda_stream, blas::TransposeOp::kN, blas::TransposeOp::kN, B, N, L1, H2, L2, /*alpha=*/T_ACC(1),
                       w_data, N * L1 * L2, L1 * L2, L2, v_data, L2 * N * H2, H2, N * H2,
                       /*beta=*/T_ACC(0), y_data, y.stride(0), H2, y.stride(1), a_ptr_data, b_ptr_data, c_ptr_data);
  }
}

}  // namespace

void MultiSegAttentionCUDAFwdOut(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& q_segment_idx, const torch::Tensor& k_segment_idx,
    double scale, double dropout, bool use_causal_mask, torch::Tensor w,
    torch::Tensor y) {
  const int64_t B = q.size(0);
  const int64_t L1 = q.size(1);
  const int64_t L2 = k.size(1);
  const int64_t N = q.size(2);
  const int64_t H2 = v.size(3);
  TORCH_CHECK(w.is_contiguous() && w.scalar_type() == v.scalar_type() && w.dim() == 4 && w.size(0) == B &&
              w.size(1) == N && w.size(2) == L1 && w.size(3) == L2,
              "multiseg_attention_fwd_out: w must be a contiguous [B, N, L1, L2] tensor of v's dtype");
  TORCH_CHECK(y.scalar_type() == v.scalar_type() && y.dim() == 4 && y.size(0) == B && y.size(1) == L1 &&
              y.size(2) == N && y.size(3) == H2 && y.stride(3) == 1 && y.stride(2) == H2 && y.stride(1) == N * H2,
              "multiseg_attention_fwd_out: y must be a [B, L1, N, V] view with contiguous rows");
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, q.scalar_type(), "MultiSegAttentionCUDAFwdOut", [&]() {
        MultiSegAttentionCUDAFwdImpl<scalar_t>(
            *(q.expect_contiguous()), *(k.expect_contiguous()),
            *(v.expect_contiguous()),  *(q_segment_idx.expect_contiguous()),
            *(k_segment_idx.expect_contiguous()), scale, dropout, use_causal_mask, y, w);
      });
}

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
