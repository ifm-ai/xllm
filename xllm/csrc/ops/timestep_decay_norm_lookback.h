#pragma once

#include <c10/util/Optional.h>
#include <torch/torch.h>

#include <tuple>

#include "utils.h"

namespace xllm {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
GroupTimestepDecayNormLookbackFwd(
    const torch::Tensor& x, const c10::optional<torch::Tensor>& bos_mask,
    const torch::Tensor& prev_count, const torch::Tensor& prev_mean,
    const torch::Tensor& prev_var, const torch::Tensor& gamma,
    const torch::Tensor& beta, const c10::optional<torch::Tensor>& padding_mask,
    int64_t num_groups, double beta1, double beta2, double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
GroupTimestepDecayNormLookbackCUDAFwd(
    const torch::Tensor& x, const c10::optional<torch::Tensor>& bos_mask,
    const torch::Tensor& prev_count, const torch::Tensor& prev_mean,
    const torch::Tensor& prev_var, const torch::Tensor& gamma,
    const torch::Tensor& beta, const c10::optional<torch::Tensor>& padding_mask,
    int64_t num_groups, double beta1, double beta2, double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
GroupTimestepDecayNormLookbackBwd(
    const torch::Tensor& y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& x_or_y,
    const torch::Tensor& prev_count, const c10::optional<torch::Tensor>& bos_mask,
    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma, const torch::Tensor& beta,
    const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups,
    double beta1, double beta2, double eps, bool memory_efficient);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
GroupTimestepDecayNormLookbackCUDABwd(
    const torch::Tensor& y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& x_or_y,
    const torch::Tensor& prev_count, const c10::optional<torch::Tensor>& bos_mask,
    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma, const torch::Tensor& beta,
    const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups,
    double beta1, double beta2, double eps, bool memory_efficient);

void DefineTimestepDecayNormLookbackOp(py::module& m);

}  // namespace ops
}  // namespace xllm
