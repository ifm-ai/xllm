// The TMA backward (bwd_tma.cu) of bf16 affine calls: RMS from x or from y, and the layer norm from x. dispatch.cu's
// backward() plans a call with plan() and, when the plan is ok, launches it with backward().
#pragma once

#include <torch/extension.h>

namespace xllm::ops::blackwell::tma {

// Every eligibility and launch-geometry decision of a call, taken without allocating, launching or reading the
// device back. `ok` false: the call is not one this kernel takes.
struct Plan {
  bool ok = false;
  bool split = false;      // wide groups spread over NWg warps each (gn_bwd_tma_split_kernel)
  bool wide = false;       // groups of more than 32 vectors of 16 bytes
  bool layer = false;      // group layer norm (beta, and mean when need_mean)
  bool need_mean = false;  // the kernel reads mean as well as rstd
  bool memeff = false;     // x_or_y is y: x_hat is reconstructed (RMS only)
  int V = 0;               // 16-byte vectors a group
  int GW = 1, U = 1, Q = 1, NWc = 1, NWr = 1, NWg = 1;  // see BwdArgs in bwd_tma.cu
  int TR = 0;              // rows a tile
  int S = 1;               // stages in the ring
  int stage_bytes = 0;     // bytes a stage
  int num_tiles = 0, grid = 0, threads = 0, dev = -1;
  int64_t R = 0, C = 0, G = 0;
  long long smem = 0, smem_optin = 0;
  int off_dy = 0, off_rs = 0, off_mu = 0;  // offsets of dy, rstd and mean in a stage
  float inv_cpg = 0.f;
  int nred = 1;            // 2 for the layer norm (dgamma and dbeta), else 1
  float occupancy = 0.f;   // the share of the grid's consumer warps that are busy (see plan())
};

Plan plan(bool layer, int64_t num_groups, bool memory_efficient, const at::Tensor& x_or_y, const at::Tensor& dy,
          const at::Tensor& mean, const at::Tensor& rstd, const at::Tensor& gamma, const at::Tensor& beta,
          const at::Tensor& dx, const at::Tensor& dgamma, const at::Tensor& dbeta);

// Launches an ok plan, with an fp32 workspace of [grid, nred * C] CTA partials from the caching allocator on the
// current stream. Returns false, with nothing launched, if the device refuses the shared-memory opt-in or the
// cooperative launch.
bool backward(const Plan& p, const at::Tensor& x_or_y, const at::Tensor& dy, const at::Tensor& mean,
              const at::Tensor& rstd, const at::Tensor& gamma, const at::Tensor& beta, at::Tensor& dx,
              at::Tensor& dgamma, at::Tensor& dbeta);

}  // namespace xllm::ops::blackwell::tma
