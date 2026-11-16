// The description of one call (Shape, Call), shared by the public entry points (native.cpp), the dispatcher
// (dispatch.cu) and the kernel launchers, and the interfaces between them.
#pragma once

#include <ATen/Dispatch.h>
#include <torch/torch.h>

#include <cstdint>

namespace xllm::ops::blackwell {

constexpr int64_t kFwd = 0;
constexpr int64_t kBwd = 1;
constexpr int64_t kRMS = 0;
constexpr int64_t kLayer = 1;

// What the dispatcher and the launchers decide on, derived from the operands by check_inputs().
struct Shape {
  int64_t mode = 0;                   // kFwd or kBwd
  int64_t norm = 0;                   // kRMS or kLayer
  int64_t groups = 0;                 // G
  int64_t channels = 0;               // C = G * cpg
  int64_t cpg = 0;                    // channels per group
  int64_t rows = 0;                   // B * S
  at::ScalarType dtype = at::kFloat;  // at::kHalf, at::kBFloat16 or at::kFloat
  bool affine = false;                // gamma given
  bool has_beta = false;              // beta given (the affine layer norm)
  bool memory_efficient = false;      // backward only: x_or_y is y
};

// The operands of one call, all contiguous and on one device. An input the call does not take is a zero-size
// placeholder, and an output it does not produce is never written.
struct Call {
  const at::Tensor* x_or_y = nullptr;
  const at::Tensor* dy = nullptr;
  const at::Tensor* mean = nullptr;
  const at::Tensor* rstd = nullptr;
  const at::Tensor* gamma = nullptr;
  const at::Tensor* beta = nullptr;
  at::Tensor* y = nullptr;
  at::Tensor* mean_out = nullptr;
  at::Tensor* rstd_out = nullptr;
  at::Tensor* dx = nullptr;
  at::Tensor* dgamma = nullptr;
  at::Tensor* dbeta = nullptr;
  double eps = 0.0;
};

// Validates the operands of a call and derives its shape (native.cpp). Absent operands are undefined tensors.
Shape check_inputs(int64_t mode, int64_t norm, int64_t channels, int64_t groups, bool memory_efficient,
                   const at::Tensor& x, const at::Tensor& dy, const at::Tensor& mean, const at::Tensor& rstd,
                   const at::Tensor& gamma, const at::Tensor& beta);

// Is the call of a class where upstream's portable kernels are the fastest (dispatch.cu)? native.cpp then runs
// upstream's own entry point (../group_{rms,layer}_norm_kernel.cu) instead of forward() or backward().
bool portable(const Shape& s);

// Runs a call on the kernel its shape is routed to (dispatch.cu).
void forward(const Shape& s, const Call& c);
void backward(const Shape& s, const Call& c);

// Forward launchers; each returns false, having launched nothing, for a call it does not take.
bool forward_flat(const Shape& s, const Call& c);           // fwd_flat.cu
bool forward_wide(const Shape& s, const Call& c);           // fwd_wide.cu
bool forward_wide_resident(const Shape& s, const Call& c);  // fwd_wide_resident.cu

// The generic kernels (generic.cuh, through fallback.cu), launched into the call's outputs. They take every call.
void fallback_forward(const Shape& s, const Call& c);
void fallback_backward(const Shape& s, const Call& c);

}  // namespace xllm::ops::blackwell

// Instantiates a launcher for each dtype the public ops accept.
#define XLLM_DISPATCH_GROUP_NORM_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, \
    AT_DISPATCH_CASE(at::kHalf, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__))
