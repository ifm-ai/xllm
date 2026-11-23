// The one-cluster backward of 2048-channel RMS groups at 128 rows (bwd_one_cluster_grad.cu), called by dispatch.cu's
// backward() for bf16 affine RMS calls from x.
#pragma once

#include <torch/torch.h>

namespace xllm::ops::blackwell {

// One 16-CTA cluster computes one group's dx and its complete dgamma, in one launch with no workspace. bf16 affine
// RMS from x, channels per group exactly 2048, exactly 128 rows; false (nothing launched) for every other call.
bool GroupNormBwdOneClusterGrad(const at::Tensor& dy, const at::Tensor& x_or_y, const at::Tensor& rstd,
                                const at::Tensor& gamma, int64_t channels, int64_t groups, bool layer,
                                bool memory_efficient, const at::Tensor& dx, const at::Tensor& dgamma);

}  // namespace xllm::ops::blackwell
