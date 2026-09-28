#pragma once

#include <torch/torch.h>

#include <tuple>

#include "../utils.h"

namespace xllm {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormCUDAFwdAffine(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups,
    const torch::Tensor& gamma, const torch::Tensor& beta, double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormCUDAFwd(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups, double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormCUDABwdAffine(
    const torch::Tensor& y_grad, const torch::Tensor& x,
    int64_t num_channels, int64_t num_groups,
    const torch::Tensor& mean, const torch::Tensor& rstd,
    const torch::Tensor& gamma, const torch::Tensor& beta,
    bool memory_efficient);

torch::Tensor GroupLayerNormCUDABwd(
    const torch::Tensor& y_grad, const torch::Tensor& x,
    int64_t num_channels, int64_t num_groups,
    const torch::Tensor& mean, const torch::Tensor& rstd,
    bool memory_efficient);

void DefineGroupLayerNormOp(py::module& m);

}  // namespace ops
}  // namespace xllm
