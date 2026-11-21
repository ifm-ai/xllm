// The Blackwell entry points of the public group norm ops (native.cpp). ../group_rms_norm.cc and
// ../group_layer_norm.cc call them, with the signatures of the portable entry points, when Supported(x) is true.
#pragma once

#include <torch/extension.h>

namespace xllm::ops::blackwell {

// True when a call on x goes through these entry points: x is a non-empty CUDA tensor of float16, bfloat16 or float32
// on a GPU of compute capability 10.x. Every such call is deterministic, under torch.use_deterministic_algorithms too.
bool Supported(const at::Tensor& x);

std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormFwd(const torch::Tensor& x, int64_t num_channels,
                                                         int64_t num_groups, double eps);
std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormFwdAffine(const torch::Tensor& x, int64_t num_channels,
                                                               int64_t num_groups, const torch::Tensor& gamma,
                                                               double eps);
torch::Tensor GroupRMSNormBwd(const torch::Tensor& y_grad, const torch::Tensor& x, int64_t num_channels,
                              int64_t num_groups, const torch::Tensor& rstd, bool memory_efficient);
std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormBwdAffine(const torch::Tensor& y_grad, const torch::Tensor& x,
                                                               int64_t num_channels, int64_t num_groups,
                                                               const torch::Tensor& rstd, const torch::Tensor& gamma,
                                                               bool memory_efficient);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormFwd(const torch::Tensor& x,
                                                                          int64_t num_channels, int64_t num_groups,
                                                                          double eps);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormFwdAffine(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups, const torch::Tensor& gamma,
    const torch::Tensor& beta, double eps);
torch::Tensor GroupLayerNormBwd(const torch::Tensor& y_grad, const torch::Tensor& x, int64_t num_channels,
                                int64_t num_groups, const torch::Tensor& mean, const torch::Tensor& rstd,
                                bool memory_efficient);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormBwdAffine(
    const torch::Tensor& y_grad, const torch::Tensor& x, int64_t num_channels, int64_t num_groups,
    const torch::Tensor& mean, const torch::Tensor& rstd, const torch::Tensor& gamma, const torch::Tensor& beta,
    bool memory_efficient);

}  // namespace xllm::ops::blackwell
