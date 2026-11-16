// The cluster backward of narrow-group affine calls (bwd_cluster_narrow.cu): dx, dgamma and dbeta in one launch, the
// parameter sums folded across a cluster through distributed shared memory. Called by dispatch.cu's backward() for bf16
// calls of at most 256 rows and groups of at most 256 channels, and fp32 calls of at most 256 rows, groups of at most
// 128 channels and at least 4096 channels.
#pragma once

#include <torch/torch.h>

namespace xllm::ops::blackwell {

// Returns false for a call it does not handle (dtype, geometry, layout or alignment).
bool GroupNormBwdClusterNarrow(const torch::Tensor& dy, const torch::Tensor& x_or_y, const torch::Tensor& mean,
                               const torch::Tensor& rstd, const torch::Tensor& gamma, const torch::Tensor& beta,
                               int64_t channels, int64_t num_groups, bool layer, bool memory_efficient,
                               torch::Tensor dx, torch::Tensor dgamma, torch::Tensor dbeta);

}  // namespace xllm::ops::blackwell
