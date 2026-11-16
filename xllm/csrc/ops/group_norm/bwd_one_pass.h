// The one-pass backward (bwd_one_pass.cu): dx, dgamma and dbeta from one read of x_or_y and dy, for bf16 affine
// calls with groups of 8 to 256 channels. Called by dispatch.cu's backward().
#pragma once

#include <torch/torch.h>

namespace xllm::ops::blackwell {

// Returns false, having done nothing, for a call it does not handle (dtype, width, layout or alignment).
bool GroupNormBwdOnePass(const torch::Tensor& dy, const torch::Tensor& x_or_y, const torch::Tensor& mean,
                         const torch::Tensor& rstd, const torch::Tensor& gamma, const torch::Tensor& beta,
                         int64_t channels, int64_t num_groups, bool layer, bool memory_efficient, torch::Tensor dx,
                         torch::Tensor dgamma, torch::Tensor dbeta);

}  // namespace xllm::ops::blackwell
