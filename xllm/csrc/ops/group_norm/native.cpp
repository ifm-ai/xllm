// The public entry points of the Blackwell group norms, called by ../group_rms_norm.cc and ../group_layer_norm.cc for
// the calls Supported() accepts. Each validates its operands; a call of a class where upstream's portable kernels are
// the fastest (portable(), dispatch.cu) then runs upstream's own entry point, before anything is allocated here, and
// costs what it cost without this directory. Any other call gets its outputs allocated and runs the dispatcher.
#include "native.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <deque>
#include <utility>

#include "../group_layer_norm.h"
#include "../group_rms_norm.h"
#include "dispatch.h"

namespace xllm::ops::blackwell {
namespace {

void check_param(const at::Tensor& param, const at::Tensor& x, int64_t channels, const char* name) {
  TORCH_CHECK(param.device() == x.device(), name, " must be on the same device as the input");
  TORCH_CHECK(param.layout() == c10::kStrided, name, " must have strided layout");
  TORCH_CHECK(param.numel() == channels, name, " must have ", channels, " elements, got ", param.numel());
  TORCH_CHECK(param.scalar_type() == x.scalar_type(), name, " must have the same dtype as the input");
}

void check_stats(const at::Tensor& stats, const at::Tensor& x, int64_t count, const char* name) {
  TORCH_CHECK(stats.device() == x.device(), name, " must be on the same device as the input");
  TORCH_CHECK(stats.layout() == c10::kStrided, name, " must have strided layout");
  TORCH_CHECK(stats.numel() == count, name, " must have ", count, " elements, got ", stats.numel());
  TORCH_CHECK(stats.scalar_type() == at::kFloat, name, " must have float32 dtype");
}

}  // namespace

Shape check_inputs(int64_t mode, int64_t norm, int64_t channels, int64_t groups, bool memory_efficient,
                   const at::Tensor& x, const at::Tensor& dy, const at::Tensor& mean, const at::Tensor& rstd,
                   const at::Tensor& gamma, const at::Tensor& beta) {
  const at::ScalarType dtype = x.scalar_type();
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat,
              "group norm supports float16, bfloat16, and float32 inputs; got ", dtype);
  TORCH_CHECK(x.is_cuda() && x.dim() == 3, "group norm expects a 3D CUDA input");
  TORCH_CHECK(x.layout() == c10::kStrided, "input must have strided layout");
  TORCH_CHECK(channels > 0 && x.size(-1) == channels, "num_channels must match the positive input channel count");
  TORCH_CHECK(groups > 0 && channels % groups == 0, "channels must be divisible by positive num_groups");
  TORCH_CHECK(channels / groups <= 65536, "channels_per_group must be at most 65536");
  const bool affine = gamma.defined() && gamma.numel() > 0;
  const bool has_beta = beta.defined() && beta.numel() > 0;
  if (affine) check_param(gamma, x, channels, "gamma");
  if (norm == kLayer && affine) {
    TORCH_CHECK(has_beta, "affine layer norm requires beta");
    check_param(beta, x, channels, "beta");
  }
  TORCH_CHECK(!has_beta || (norm == kLayer && affine), "beta requires layer norm with gamma");
  const int64_t rows = x.numel() / channels;
  if (mode == kBwd) {
    TORCH_CHECK(dy.device() == x.device(), "y_grad must be on the same device as the input");
    TORCH_CHECK(dy.layout() == c10::kStrided, "y_grad must have strided layout");
    TORCH_CHECK(dy.sizes() == x.sizes(), "y_grad must have the same shape as the input");
    TORCH_CHECK(dy.scalar_type() == dtype, "y_grad must have the same dtype as the input");
    check_stats(rstd, x, rows * groups, "rstd");
    if (norm == kLayer) check_stats(mean, x, rows * groups, "mean");
  }
  return {mode, norm, groups, channels, channels / groups, rows, dtype, affine, has_beta, memory_efficient};
}

}  // namespace xllm::ops::blackwell

namespace xllm::ops::blackwell {
namespace {

// The operand of an absent input, and the destination of an output a call does not produce: a zero-size tensor of
// the input's device and dtype, made once per thread, device and dtype (making one costs about as much host time as
// an output allocation).
const at::Tensor& Placeholder(const at::Tensor& like) {
  thread_local std::deque<at::Tensor> cache;  // a deque: growing it keeps references to its elements valid
  for (const at::Tensor& t : cache)
    if (t.device() == like.device() && t.scalar_type() == like.scalar_type()) return t;
  return cache.emplace_back(at::empty({0}, like.options()));
}

// An input as the dispatcher takes it: contiguous (borrowed when it already is), or the placeholder when absent.
c10::MaybeOwned<at::Tensor> Operand(const at::Tensor& t, const at::Tensor& placeholder) {
  return t.defined() ? t.expect_contiguous() : c10::MaybeOwned<at::Tensor>::borrowed(placeholder);
}

// A validated call the dispatcher runs: its operands made contiguous (borrowed when they are), its outputs allocated.
std::tuple<at::Tensor, at::Tensor, at::Tensor> Forward(const Shape& s, const at::Tensor& x,
                                                       const at::Tensor& gamma_in, const at::Tensor& beta_in,
                                                       double eps) {
  const c10::cuda::CUDAGuard guard(x.device());
  const at::Tensor& empty = Placeholder(x);
  const auto xc = x.expect_contiguous();
  const auto gamma = Operand(gamma_in, empty);
  const auto beta = Operand(beta_in, empty);
  at::Tensor y = at::empty_like(*xc);
  at::Tensor rstd = at::empty({s.rows, s.groups}, x.options().dtype(at::kFloat));
  at::Tensor mean = s.norm == kLayer ? at::empty_like(rstd) : empty;
  at::Tensor unused = empty;
  const Call c{&*xc, &empty, &empty, &empty, &*gamma, &*beta, &y, &mean, &rstd, &unused, &unused, &unused, eps};
  forward(s, c);
  return {std::move(y), std::move(mean), std::move(rstd)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> Backward(const Shape& s, const at::Tensor& dy_in,
                                                        const at::Tensor& x, const at::Tensor& mean_in,
                                                        const at::Tensor& rstd_in, const at::Tensor& gamma_in,
                                                        const at::Tensor& beta_in) {
  const c10::cuda::CUDAGuard guard(x.device());
  const at::Tensor& empty = Placeholder(x);
  const auto xc = x.expect_contiguous();
  const auto dy = dy_in.expect_contiguous();
  const auto mean = Operand(mean_in, empty);
  const auto rstd = rstd_in.expect_contiguous();
  const auto gamma = Operand(gamma_in, empty);
  const auto beta = Operand(beta_in, empty);
  at::Tensor dx = at::empty_like(*xc);
  at::Tensor dgamma = s.affine ? at::empty_like(*gamma) : empty;
  at::Tensor dbeta = s.has_beta ? at::empty_like(*beta) : empty;
  at::Tensor unused = empty;
  const Call c{&*xc, &*dy, &*mean, &*rstd, &*gamma, &*beta, &unused, &unused, &unused, &dx, &dgamma, &dbeta, 0.0};
  backward(s, c);
  return {std::move(dx), std::move(dgamma), std::move(dbeta)};
}

}  // namespace

bool Supported(const at::Tensor& x) {
  // Compute capability 10.x (B200, B300) only: the kernels are tuned to its shared memory and cluster limits.
  const at::ScalarType dtype = x.scalar_type();
  return x.is_cuda() && x.numel() > 0 && (dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat) &&
         at::cuda::getDeviceProperties(x.get_device())->major == 10;
}

std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormFwd(const torch::Tensor& x, int64_t num_channels,
                                                         int64_t num_groups, double eps) {
  const Shape s = check_inputs(kFwd, kRMS, num_channels, num_groups, false, x, {}, {}, {}, {}, {});
  if (portable(s)) return GroupRMSNormCUDAFwd(x, num_channels, num_groups, eps);
  auto [y, mean, rstd] = Forward(s, x, {}, {}, eps);
  return {std::move(y), std::move(rstd)};
}

std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormFwdAffine(const torch::Tensor& x, int64_t num_channels,
                                                               int64_t num_groups, const torch::Tensor& gamma,
                                                               double eps) {
  const Shape s = check_inputs(kFwd, kRMS, num_channels, num_groups, false, x, {}, {}, {}, gamma, {});
  if (portable(s)) return GroupRMSNormCUDAFwdAffine(x, num_channels, num_groups, gamma, eps);
  auto [y, mean, rstd] = Forward(s, x, gamma, {}, eps);
  return {std::move(y), std::move(rstd)};
}

torch::Tensor GroupRMSNormBwd(const torch::Tensor& y_grad, const torch::Tensor& x, int64_t num_channels,
                              int64_t num_groups, const torch::Tensor& rstd, bool memory_efficient) {
  const Shape s = check_inputs(kBwd, kRMS, num_channels, num_groups, memory_efficient, x, y_grad, {}, rstd, {}, {});
  if (portable(s)) return GroupRMSNormCUDABwd(y_grad, x, num_channels, num_groups, rstd, memory_efficient);
  return std::get<0>(Backward(s, y_grad, x, {}, rstd, {}, {}));
}

std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormBwdAffine(const torch::Tensor& y_grad, const torch::Tensor& x,
                                                               int64_t num_channels, int64_t num_groups,
                                                               const torch::Tensor& rstd, const torch::Tensor& gamma,
                                                               bool memory_efficient) {
  const Shape s = check_inputs(kBwd, kRMS, num_channels, num_groups, memory_efficient, x, y_grad, {}, rstd, gamma, {});
  if (portable(s)) return GroupRMSNormCUDABwdAffine(y_grad, x, num_channels, num_groups, rstd, gamma, memory_efficient);
  auto [dx, dgamma, dbeta] = Backward(s, y_grad, x, {}, rstd, gamma, {});
  return {std::move(dx), std::move(dgamma)};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormFwd(const torch::Tensor& x,
                                                                          int64_t num_channels, int64_t num_groups,
                                                                          double eps) {
  const Shape s = check_inputs(kFwd, kLayer, num_channels, num_groups, false, x, {}, {}, {}, {}, {});
  if (portable(s)) return GroupLayerNormCUDAFwd(x, num_channels, num_groups, eps);
  return Forward(s, x, {}, {}, eps);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormFwdAffine(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups, const torch::Tensor& gamma,
    const torch::Tensor& beta, double eps) {
  const Shape s = check_inputs(kFwd, kLayer, num_channels, num_groups, false, x, {}, {}, {}, gamma, beta);
  if (portable(s)) return GroupLayerNormCUDAFwdAffine(x, num_channels, num_groups, gamma, beta, eps);
  return Forward(s, x, gamma, beta, eps);
}

torch::Tensor GroupLayerNormBwd(const torch::Tensor& y_grad, const torch::Tensor& x, int64_t num_channels,
                                int64_t num_groups, const torch::Tensor& mean, const torch::Tensor& rstd,
                                bool memory_efficient) {
  const Shape s = check_inputs(kBwd, kLayer, num_channels, num_groups, memory_efficient, x, y_grad, mean, rstd, {}, {});
  if (portable(s)) return GroupLayerNormCUDABwd(y_grad, x, num_channels, num_groups, mean, rstd, memory_efficient);
  return std::get<0>(Backward(s, y_grad, x, mean, rstd, {}, {}));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormBwdAffine(
    const torch::Tensor& y_grad, const torch::Tensor& x, int64_t num_channels, int64_t num_groups,
    const torch::Tensor& mean, const torch::Tensor& rstd, const torch::Tensor& gamma, const torch::Tensor& beta,
    bool memory_efficient) {
  const Shape s = check_inputs(kBwd, kLayer, num_channels, num_groups, memory_efficient, x, y_grad,
                               mean, rstd, gamma, beta);
  if (portable(s))
    return GroupLayerNormCUDABwdAffine(y_grad, x, num_channels, num_groups, mean, rstd, gamma, beta, memory_efficient);
  return Backward(s, y_grad, x, mean, rstd, gamma, beta);
}

}  // namespace xllm::ops::blackwell
