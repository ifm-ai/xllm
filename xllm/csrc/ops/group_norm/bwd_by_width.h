// The by-width backward (bwd_by_width.cu): dx of a bf16 call without an affine transform in one pass over x_or_y and
// dy, for groups of 2056 to 8192 channels and at least 2048 rows. Called by dispatch.cu's backward().
#pragma once

#include <ATen/ATen.h>
#include <cstdint>

namespace xllm::ops::blackwell {
// What became of a call. NotCovered: the call is not one this kernel takes; nothing was done. Launched: dx is
// written. NeedsFixedReduction: an affine call of a width this kernel claims; nothing was done, and the caller sends
// it to the generic kernels, whose dgamma/dbeta reduction runs in a fixed order.
enum class ByWidthRoute { NotCovered, Launched, NeedsFixedReduction };

// norm: 0 RMS, 1 layer norm.
ByWidthRoute GroupNormBwdByWidth(int64_t norm, const at::Tensor& x_or_y, const at::Tensor& dy, const at::Tensor& mean,
                                 const at::Tensor& rstd, const at::Tensor& gamma, const at::Tensor& beta,
                                 bool memory_efficient, at::Tensor& dx, at::Tensor& dgamma, at::Tensor& dbeta);
}  // namespace xllm::ops::blackwell
