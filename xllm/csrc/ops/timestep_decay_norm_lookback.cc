#include "ops/timestep_decay_norm_lookback.h"

namespace xllm {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
GroupTimestepDecayNormLookbackFwd(
    const torch::Tensor& x, const c10::optional<torch::Tensor>& bos_mask,
    const torch::Tensor& prev_count, const torch::Tensor& prev_mean,
    const torch::Tensor& prev_var, const torch::Tensor& gamma,
    const torch::Tensor& beta, const c10::optional<torch::Tensor>& padding_mask,
    int64_t num_groups, double beta1, double beta2, double eps) {
  return GroupTimestepDecayNormLookbackCUDAFwd(
      x, bos_mask, prev_count, prev_mean, prev_var, gamma, beta, padding_mask,
      num_groups, beta1, beta2, eps);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
GroupTimestepDecayNormLookbackBwd(
    const torch::Tensor& y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& x_or_y,
    const torch::Tensor& prev_count, const c10::optional<torch::Tensor>& bos_mask,
    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma, const torch::Tensor& beta,
    const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups,
    double beta1, double beta2, double eps, bool memory_efficient) {
  return GroupTimestepDecayNormLookbackCUDABwd(
      y_grad, mean_grad, var_grad, x_or_y, prev_count, bos_mask, cummean,
      cumrstd, gamma, beta, padding_mask, num_groups, beta1, beta2, eps,
      memory_efficient);
}

void DefineTimestepDecayNormLookbackOp(py::module& m) {
  m.def("group_timestep_decay_norm_lookback_fwd", &GroupTimestepDecayNormLookbackFwd,
        "GroupTimestepDecayNormLookback forward")
      .def("group_timestep_decay_norm_lookback_bwd", &GroupTimestepDecayNormLookbackBwd,
           "GroupTimestepDecayNormLookback backward");
}

}  // namespace ops
}  // namespace xllm
