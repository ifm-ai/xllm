#include <ATen/AccumulateType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/accumulate.h>

#include "group_rms_norm.h"
#include "group_norm_kernel.cuh"

namespace xllm {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormCUDAFwdAffine(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups,
    const torch::Tensor& gamma, double eps) {
  const int64_t channels_per_group = group_norm::CheckShapes(x, num_channels, num_groups);
  group_norm::CheckParam(gamma, x, num_channels, "gamma");
  const auto x_sizes = x.sizes();
  const int64_t outer_size = c10::multiply_integers(x_sizes.cbegin(), x_sizes.cbegin() + 2);
  const auto acc_type = at::toAccumulateType(x.scalar_type(), true);
  at::cuda::OptionalCUDAGuard guard(at::device_of(x));

  const torch::Tensor x_c = x.contiguous();
  const torch::Tensor gamma_c = gamma.contiguous();
  torch::Tensor y = torch::empty_like(x_c);
  torch::Tensor rstd = torch::empty({outer_size, num_groups}, x.options().dtype(acc_type));

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "GroupRMSNormCUDAFwdAffine", [&]() {
    using T_ACC = at::acc_type<scalar_t, true>;
    group_norm::GroupNormFwd<scalar_t, true>(
        outer_size, num_groups, channels_per_group, eps, x_c.data_ptr<scalar_t>(),
        gamma_c.data_ptr<scalar_t>(), nullptr, y.data_ptr<scalar_t>(), nullptr, rstd.data_ptr<T_ACC>());
  });

  return std::make_tuple(y, rstd);
}

std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormCUDAFwd(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups, double eps) {
  const int64_t channels_per_group = group_norm::CheckShapes(x, num_channels, num_groups);
  const auto x_sizes = x.sizes();
  const int64_t outer_size = c10::multiply_integers(x_sizes.cbegin(), x_sizes.cbegin() + 2);
  const auto acc_type = at::toAccumulateType(x.scalar_type(), true);
  at::cuda::OptionalCUDAGuard guard(at::device_of(x));

  const torch::Tensor x_c = x.contiguous();
  torch::Tensor y = torch::empty_like(x_c);
  torch::Tensor rstd = torch::empty({outer_size, num_groups}, x.options().dtype(acc_type));

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "GroupRMSNormCUDAFwd", [&]() {
    using T_ACC = at::acc_type<scalar_t, true>;
    group_norm::GroupNormFwd<scalar_t, true>(
        outer_size, num_groups, channels_per_group, eps, x_c.data_ptr<scalar_t>(), nullptr, nullptr,
        y.data_ptr<scalar_t>(), nullptr, rstd.data_ptr<T_ACC>());
  });

  return std::make_tuple(y, rstd);
}

std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormCUDABwdAffine(
    const torch::Tensor& y_grad, const torch::Tensor& x,
    int64_t num_channels, int64_t num_groups,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    bool memory_efficient) {
  const int64_t channels_per_group = group_norm::CheckShapes(x, num_channels, num_groups);
  group_norm::CheckParam(gamma, x, num_channels, "gamma");
  group_norm::CheckGradient(y_grad, x);
  const auto x_sizes = x.sizes();
  const int64_t outer_size = c10::multiply_integers(x_sizes.cbegin(), x_sizes.cbegin() + 2);
  group_norm::CheckStats(rstd, x, outer_size * num_groups, "rstd");
  at::cuda::OptionalCUDAGuard guard(at::device_of(x));

  const torch::Tensor y_grad_c = y_grad.contiguous();
  const torch::Tensor x_c = x.contiguous();
  const torch::Tensor rstd_c = rstd.contiguous();
  const torch::Tensor gamma_c = gamma.contiguous();
  torch::Tensor x_grad = torch::empty_like(x_c);
  torch::Tensor gamma_grad = torch::zeros_like(gamma_c);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "GroupRMSNormCUDABwdAffine", [&]() {
        using T_ACC = at::acc_type<scalar_t, true>;
        group_norm::GroupNormBwd<scalar_t, true>(
            outer_size, num_groups, channels_per_group, memory_efficient,
            y_grad_c.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), nullptr, rstd_c.data_ptr<T_ACC>(),
            gamma_c.data_ptr<scalar_t>(), nullptr, x_grad.data_ptr<scalar_t>(),
            gamma_grad.data_ptr<scalar_t>(), nullptr, x.options());
      });

  return std::make_tuple(x_grad, gamma_grad);
}

torch::Tensor GroupRMSNormCUDABwd(
    const torch::Tensor& y_grad, const torch::Tensor& x,
    int64_t num_channels, int64_t num_groups,
    const torch::Tensor& rstd, bool memory_efficient) {
  const int64_t channels_per_group = group_norm::CheckShapes(x, num_channels, num_groups);
  group_norm::CheckGradient(y_grad, x);
  const auto x_sizes = x.sizes();
  const int64_t outer_size = c10::multiply_integers(x_sizes.cbegin(), x_sizes.cbegin() + 2);
  group_norm::CheckStats(rstd, x, outer_size * num_groups, "rstd");
  at::cuda::OptionalCUDAGuard guard(at::device_of(x));

  const torch::Tensor y_grad_c = y_grad.contiguous();
  const torch::Tensor x_c = x.contiguous();
  const torch::Tensor rstd_c = rstd.contiguous();
  torch::Tensor x_grad = torch::empty_like(x_c);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "GroupRMSNormCUDABwd", [&]() {
        using T_ACC = at::acc_type<scalar_t, true>;
        group_norm::GroupNormBwd<scalar_t, true>(
            outer_size, num_groups, channels_per_group, memory_efficient,
            y_grad_c.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), nullptr, rstd_c.data_ptr<T_ACC>(),
            nullptr, nullptr, x_grad.data_ptr<scalar_t>(), nullptr, nullptr, x.options());
      });

  return x_grad;
}

}  // namespace ops
}  // namespace xllm
