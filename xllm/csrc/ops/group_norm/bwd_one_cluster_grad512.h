// The one-cluster backward of 512-channel groups at 128 rows (bwd_one_cluster_grad512.cu), called by dispatch.cu's
// backward() for bf16 affine calls of 8 such groups, from x or from y.
#pragma once

#include <torch/torch.h>

namespace xllm::ops::blackwell {

// One 16-CTA cluster computes one group's dx and its complete dgamma (and dbeta), in one launch with no workspace.
// bf16 affine, channels per group exactly 512, exactly 128 rows; false (nothing launched) for every other call.
bool GroupNormBwdOneClusterGrad512(const at::Tensor& dy, const at::Tensor& x_or_y, const at::Tensor& mean,
                                   const at::Tensor& rstd, const at::Tensor& gamma, const at::Tensor& beta,
                                   int64_t channels, int64_t groups, bool layer, bool memory_efficient,
                                   const at::Tensor& dx, const at::Tensor& dgamma, const at::Tensor& dbeta);

}  // namespace xllm::ops::blackwell
