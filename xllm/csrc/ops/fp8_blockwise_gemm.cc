// Blockwise-scaled fp8 GEMM on cuBLASLt with an optional bf16 C accumulated in the epilogue (Hopper fp8 path,
// xllm/modules/fused_ops/fp8_sm90.py): D = A @ B^T (+ C). torch._scaled_mm takes no C matrix, so the residual adds
// after the fp8 output GEMMs were separate passes (the attention output and FFN output per layer).
//   a: (M, K) e4m3, rows K-contiguous; a_scale: 1 x 128 fp32 scales (M, K / 128), M-major
//   b: (N, K) e4m3, rows K-contiguous; b_scale: 128 x 128 tile scales (K / 128, N / 128) (b_tile) or 1 x 128 (K / 128, N)
//   d, c: (M, N) bf16 contiguous
#include "fp8_blockwise_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>

#include <map>
#include <tuple>

namespace xllm {
namespace ops {

namespace {

#define XLLM_CUBLASLT_CHECK(x)                                                                 \
  do {                                                                                         \
    cublasStatus_t s_ = (x);                                                                   \
    TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, "cuBLASLt error ", static_cast<int>(s_), " in ", #x); \
  } while (0)

struct Plan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulHeuristicResult_t algo;
};

using Key = std::tuple<int, int64_t, int64_t, int64_t, int64_t, int64_t, bool, bool>;

// thread-local: the forward and the backward (autograd thread) may run GEMMs of the same shape
Plan& GetPlan(const torch::Tensor& a, const torch::Tensor& b, bool b_tile, bool has_c, void* scale_dummy) {
  thread_local std::map<Key, Plan> plans;
  const int64_t M = a.size(0), K = a.size(1), N = b.size(0);
  Key key{a.get_device(), M, N, K, a.stride(0), b.stride(0), b_tile, has_c};
  auto it = plans.find(key);
  if (it != plans.end()) return it->second;
  Plan p;
  XLLM_CUBLASLT_CHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  // column-major view: cuBLASLt "A" = b (K x N, op T), "B" = a (K x M, op N), D = N x M (= (M, N) row-major)
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  XLLM_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  XLLM_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  cublasLtMatmulMatrixScale_t ma = b_tile ? CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F : CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
  cublasLtMatmulMatrixScale_t mb = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
  XLLM_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &ma, sizeof(ma)));
  XLLM_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mb, sizeof(mb)));
  XLLM_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_dummy, sizeof(void*)));
  XLLM_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_dummy, sizeof(void*)));
  XLLM_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_8F_E4M3, K, N, b.stride(0)));
  XLLM_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_8F_E4M3, K, M, a.stride(0)));
  XLLM_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_16BF, N, M, N));
  cublasLtMatmulPreference_t pref;
  XLLM_CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws = at::cuda::getCUDABlasLtWorkspaceSize();
  XLLM_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  int n = 0;
  XLLM_CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(at::cuda::getCurrentCUDABlasLtHandle(), p.op, p.la, p.lb, p.lc,
                                                     p.lc, pref, 1, &p.algo, &n));
  cublasLtMatmulPreferenceDestroy(pref);
  TORCH_CHECK(n > 0, "no cuBLASLt algorithm for the blockwise fp8 GEMM ", M, "x", N, "x", K);
  return plans.emplace(key, p).first->second;
}

}  // namespace

void Fp8BlockwiseGemm(const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
                      const torch::Tensor& b_scale, torch::Tensor& d, const std::optional<torch::Tensor>& c,
                      bool b_tile) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && d.is_cuda(), "CUDA tensors expected");
  TORCH_CHECK(a.scalar_type() == at::kFloat8_e4m3fn && b.scalar_type() == at::kFloat8_e4m3fn, "e4m3 operands expected");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(1) && a.stride(1) == 1 && b.stride(1) == 1);
  TORCH_CHECK(a_scale.scalar_type() == at::kFloat && b_scale.scalar_type() == at::kFloat, "fp32 scales expected");
  TORCH_CHECK(d.scalar_type() == at::kBFloat16 && d.is_contiguous() && d.size(0) == a.size(0) && d.size(1) == b.size(0));
  if (c.has_value()) {
    TORCH_CHECK(c->scalar_type() == at::kBFloat16 && c->is_contiguous() && c->sizes() == d.sizes());
  }
  Plan& p = GetPlan(a, b, b_tile, c.has_value(), b_scale.data_ptr());
  const void* sa = b_scale.data_ptr();  // cuBLASLt "A" is b
  const void* sb = a_scale.data_ptr();
  XLLM_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof(sa)));
  XLLM_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof(sb)));
  const float alpha = 1.f, beta = c.has_value() ? 1.f : 0.f;
  const void* cp = c.has_value() ? c->data_ptr() : d.data_ptr();
  XLLM_CUBLASLT_CHECK(cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), p.op, &alpha, b.data_ptr(), p.la,
                                     a.data_ptr(), p.lb, &beta, cp, p.lc, d.data_ptr(), p.lc, &p.algo.algo,
                                     at::cuda::getCUDABlasLtWorkspace(), at::cuda::getCUDABlasLtWorkspaceSize(),
                                     at::cuda::getCurrentCUDAStream()));
}

void DefineFp8BlockwiseGemmOp(py::module& m) {
  m.def("fp8_blockwise_gemm", &Fp8BlockwiseGemm, "D = A @ B^T (+ C), blockwise-scaled fp8 (cuBLASLt)",
        py::arg("a"), py::arg("a_scale"), py::arg("b"), py::arg("b_scale"), py::arg("d"), py::arg("c") = py::none(),
        py::arg("b_tile") = true);
}

}  // namespace ops
}  // namespace xllm
