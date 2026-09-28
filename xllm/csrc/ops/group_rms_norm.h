#pragma once

#include <torch/torch.h>

#include <tuple>

#include "../utils.h"

namespace xllm {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormCUDAFwdAffine(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups,
    const torch::Tensor& gamma, double eps);

std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormCUDAFwd(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups, double eps);

std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormCUDABwdAffine(
    const torch::Tensor& y_grad, const torch::Tensor& x,
    int64_t num_channels, int64_t num_groups,
    const torch::Tensor& rstd, const torch::Tensor& gamma, bool memory_efficient);

torch::Tensor GroupRMSNormCUDABwd(
    const torch::Tensor& y_grad, const torch::Tensor& x,
    int64_t num_channels, int64_t num_groups,
    const torch::Tensor& rstd, bool memory_efficient);

void DefineGroupRMSNormOp(py::module& m);

}  // namespace ops
}  // namespace xllm
