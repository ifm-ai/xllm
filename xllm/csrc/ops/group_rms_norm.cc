#include "group_rms_norm.h"

namespace xllm {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormFwdAffine(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups,
    const torch::Tensor& gamma, double eps) {
  TORCH_CHECK(x.device().type() == torch::kCUDA);
  TORCH_CHECK(gamma.device().type() == torch::kCUDA);
  return GroupRMSNormCUDAFwdAffine(x, num_channels, num_groups, gamma, eps);
}

std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormFwd(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups, double eps) {
  TORCH_CHECK(x.device().type() == torch::kCUDA);
  return GroupRMSNormCUDAFwd(x, num_channels, num_groups, eps);
}

std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormBwdAffine(
    const torch::Tensor& y_grad, const torch::Tensor& x,
    int64_t num_channels, int64_t num_groups,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    bool memory_efficient) {
  TORCH_CHECK(y_grad.device().type() == torch::kCUDA);
  TORCH_CHECK(x.device().type() == torch::kCUDA);
  TORCH_CHECK(rstd.device().type() == torch::kCUDA);
  TORCH_CHECK(gamma.device().type() == torch::kCUDA);
  return GroupRMSNormCUDABwdAffine(y_grad, x, num_channels, num_groups, rstd, gamma, memory_efficient);
}

torch::Tensor GroupRMSNormBwd(
    const torch::Tensor& y_grad, const torch::Tensor& x,
    int64_t num_channels, int64_t num_groups,
    const torch::Tensor& rstd, bool memory_efficient) {
  TORCH_CHECK(y_grad.device().type() == torch::kCUDA);
  TORCH_CHECK(x.device().type() == torch::kCUDA);
  TORCH_CHECK(rstd.device().type() == torch::kCUDA);
  return GroupRMSNormCUDABwd(y_grad, x, num_channels, num_groups, rstd, memory_efficient);
}

void DefineGroupRMSNormOp(py::module& m) {
  m.def("group_rms_norm_fwd_affine", &GroupRMSNormFwdAffine, "GroupRMSNormFwdAffine")
      .def("group_rms_norm_fwd", &GroupRMSNormFwd, "GroupRMSNormFwd")
      .def("group_rms_norm_bwd_affine", &GroupRMSNormBwdAffine, "GroupRMSNormBwdAffine")
      .def("group_rms_norm_bwd", &GroupRMSNormBwd, "GroupRMSNormBwd");
}

}  // namespace ops
}  // namespace xllm
