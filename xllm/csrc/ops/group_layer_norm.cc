#include "group_layer_norm.h"

namespace xllm {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormFwdAffine(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups,
    const torch::Tensor& gamma, const torch::Tensor& beta, double eps) {
  TORCH_CHECK(x.device().type() == torch::kCUDA);
  TORCH_CHECK(gamma.device().type() == torch::kCUDA);
  TORCH_CHECK(beta.device().type() == torch::kCUDA);
  return GroupLayerNormCUDAFwdAffine(x, num_channels, num_groups, gamma, beta, eps);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormFwd(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups, double eps) {
  TORCH_CHECK(x.device().type() == torch::kCUDA);
  return GroupLayerNormCUDAFwd(x, num_channels, num_groups, eps);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormBwdAffine(
    const torch::Tensor& y_grad, const torch::Tensor& x, int64_t num_channels,
    int64_t num_groups, const torch::Tensor& mean, const torch::Tensor& rstd,
    const torch::Tensor& gamma, const torch::Tensor& beta, bool memory_efficient) {
  TORCH_CHECK(y_grad.device().type() == torch::kCUDA);
  TORCH_CHECK(x.device().type() == torch::kCUDA);
  TORCH_CHECK(mean.device().type() == torch::kCUDA);
  TORCH_CHECK(rstd.device().type() == torch::kCUDA);
  TORCH_CHECK(gamma.device().type() == torch::kCUDA);
  TORCH_CHECK(beta.device().type() == torch::kCUDA);
  return GroupLayerNormCUDABwdAffine(y_grad, x, num_channels, num_groups, mean, rstd, gamma, beta, memory_efficient);
}

torch::Tensor GroupLayerNormBwd(
    const torch::Tensor& y_grad, const torch::Tensor& x, int64_t num_channels,
    int64_t num_groups, const torch::Tensor& mean, const torch::Tensor& rstd,
    bool memory_efficient) {
  TORCH_CHECK(y_grad.device().type() == torch::kCUDA);
  TORCH_CHECK(x.device().type() == torch::kCUDA);
  TORCH_CHECK(mean.device().type() == torch::kCUDA);
  TORCH_CHECK(rstd.device().type() == torch::kCUDA);
  return GroupLayerNormCUDABwd(y_grad, x, num_channels, num_groups, mean, rstd, memory_efficient);
}

void DefineGroupLayerNormOp(py::module& m) {
  m.def("group_layer_norm_fwd_affine", &GroupLayerNormFwdAffine, "GroupLayerNormFwdAffine")
      .def("group_layer_norm_fwd", &GroupLayerNormFwd, "GroupLayerNormFwd")
      .def("group_layer_norm_bwd_affine", &GroupLayerNormBwdAffine, "GroupLayerNormBwdAffine")
      .def("group_layer_norm_bwd", &GroupLayerNormBwd, "GroupLayerNormBwd");
}

}  // namespace ops
}  // namespace xllm
