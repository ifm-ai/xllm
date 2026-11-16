// Dense-lane group norms (dense_lanes.cu), called by dispatch.cu, for groups of 1..32 vectors of 16 bytes (8..256
// channels in 16 bits, 4..128 in fp32): a group takes as many lanes of a warp as it has vectors (half as many, two
// vectors a lane, for an even count without affine), so groups of 3, 5 or 10 vectors keep every lane busy and several
// whole groups share a warp.
#pragma once

#include <torch/extension.h>

namespace xllm::ops::blackwell::dense_lanes {

// True when these kernels take the call: groups of 1..32 vectors of 16 bytes, and at least one row. For calls the
// dispatcher has validated (contiguous, 16-byte aligned, at most INT_MAX channels).
bool Eligible(const torch::Tensor& x_or_y, int64_t num_groups);

// True when the forward's two-slot grid for this shape (the one-slot grid for an odd vector count) already fills the
// machine's resident capacity. False for groups outside 1..32 vectors. Reads no tensor.
bool GridSaturated(int64_t rows, int64_t groups, int64_t cpg, int elemsize);

// y, mean (layer norm) and rstd of x. gamma and beta are zero-size without affine.
void Fwd(const torch::Tensor& x, int64_t num_groups, bool layer, const torch::Tensor& gamma, const torch::Tensor& beta,
         double eps, const torch::Tensor& y, const torch::Tensor& mean_out, const torch::Tensor& rstd_out);

// dx, and dgamma (and dbeta) with affine, from x or, memory efficient, from y. Takes fp32 calls, and bf16 calls without
// affine but the memory-efficient layer norm: the dispatcher sends no other backward here.
void Bwd(const torch::Tensor& x_or_y, const torch::Tensor& dy, int64_t num_groups, bool layer, bool memory_efficient,
         const torch::Tensor& mean, const torch::Tensor& rstd, const torch::Tensor& gamma, const torch::Tensor& beta,
         const torch::Tensor& dx, const torch::Tensor& dgamma, const torch::Tensor& dbeta);

}  // namespace xllm::ops::blackwell::dense_lanes
