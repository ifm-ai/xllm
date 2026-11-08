#include <ATen/AccumulateType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/accumulate.h>

#include "group_layer_norm.h"
#include "group_norm_kernel.cuh"

namespace xllm {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormCUDAFwdAffine(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups,
    const torch::Tensor& gamma, const torch::Tensor& beta, double eps) {
  const int64_t channels_per_group = group_norm::CheckShapes(x, num_channels, num_groups);
  group_norm::CheckParam(gamma, x, num_channels, "gamma");
  group_norm::CheckParam(beta, x, num_channels, "beta");
  const auto x_sizes = x.sizes();
  const int64_t outer_size = c10::multiply_integers(x_sizes.cbegin(), x_sizes.cbegin() + 2);
  const auto acc_type = at::toAccumulateType(x.scalar_type(), true);
  at::cuda::OptionalCUDAGuard guard(at::device_of(x));

  const torch::Tensor x_c = x.contiguous();
  const torch::Tensor gamma_c = gamma.contiguous();
  const torch::Tensor beta_c = beta.contiguous();
  torch::Tensor y = torch::empty_like(x_c);
  torch::Tensor mean = torch::empty({outer_size, num_groups}, x.options().dtype(acc_type));
  torch::Tensor rstd = torch::empty({outer_size, num_groups}, x.options().dtype(acc_type));

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "GroupLayerNormCUDAFwdAffine", [&]() {
    using T_ACC = at::acc_type<scalar_t, true>;
    group_norm::GroupNormFwd<scalar_t, false>(
        outer_size, num_groups, channels_per_group, eps, x_c.data_ptr<scalar_t>(),
        gamma_c.data_ptr<scalar_t>(), beta_c.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
        mean.data_ptr<T_ACC>(), rstd.data_ptr<T_ACC>());
  });

  return std::make_tuple(y, mean, rstd);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormCUDAFwd(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups, double eps) {
  const int64_t channels_per_group = group_norm::CheckShapes(x, num_channels, num_groups);
  const auto x_sizes = x.sizes();
  const int64_t outer_size = c10::multiply_integers(x_sizes.cbegin(), x_sizes.cbegin() + 2);
  const auto acc_type = at::toAccumulateType(x.scalar_type(), true);
  at::cuda::OptionalCUDAGuard guard(at::device_of(x));

  const torch::Tensor x_c = x.contiguous();
  torch::Tensor y = torch::empty_like(x_c);
  torch::Tensor mean = torch::empty({outer_size, num_groups}, x.options().dtype(acc_type));
  torch::Tensor rstd = torch::empty({outer_size, num_groups}, x.options().dtype(acc_type));

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "GroupLayerNormCUDAFwd", [&]() {
    using T_ACC = at::acc_type<scalar_t, true>;
    group_norm::GroupNormFwd<scalar_t, false>(
        outer_size, num_groups, channels_per_group, eps, x_c.data_ptr<scalar_t>(), nullptr, nullptr,
        y.data_ptr<scalar_t>(), mean.data_ptr<T_ACC>(), rstd.data_ptr<T_ACC>());
  });

  return std::make_tuple(y, mean, rstd);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormCUDABwdAffine(
    const torch::Tensor& y_grad, const torch::Tensor& x,
    int64_t num_channels, int64_t num_groups,
    const torch::Tensor& mean, const torch::Tensor& rstd,
    const torch::Tensor& gamma, const torch::Tensor& beta,
    bool memory_efficient) {
  const int64_t channels_per_group = group_norm::CheckShapes(x, num_channels, num_groups);
  group_norm::CheckParam(gamma, x, num_channels, "gamma");
  group_norm::CheckParam(beta, x, num_channels, "beta");
  group_norm::CheckGradient(y_grad, x);
  const auto x_sizes = x.sizes();
  const int64_t outer_size = c10::multiply_integers(x_sizes.cbegin(), x_sizes.cbegin() + 2);
  group_norm::CheckStats(mean, x, outer_size * num_groups, "mean");
  group_norm::CheckStats(rstd, x, outer_size * num_groups, "rstd");
  at::cuda::OptionalCUDAGuard guard(at::device_of(x));

  const torch::Tensor y_grad_c = y_grad.contiguous();
  const torch::Tensor x_c = x.contiguous();
  const torch::Tensor mean_c = mean.contiguous();
  const torch::Tensor rstd_c = rstd.contiguous();
  const torch::Tensor gamma_c = gamma.contiguous();
  const torch::Tensor beta_c = beta.contiguous();
  torch::Tensor x_grad = torch::empty_like(x_c);
  torch::Tensor gamma_grad = torch::zeros_like(gamma_c);
  torch::Tensor beta_grad = torch::zeros_like(gamma_c);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "GroupLayerNormCUDABwdAffine", [&]() {
        using T_ACC = at::acc_type<scalar_t, true>;
        group_norm::GroupNormBwd<scalar_t, false>(
            outer_size, num_groups, channels_per_group, memory_efficient,
            y_grad_c.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), mean_c.data_ptr<T_ACC>(),
            rstd_c.data_ptr<T_ACC>(), gamma_c.data_ptr<scalar_t>(), beta_c.data_ptr<scalar_t>(),
            x_grad.data_ptr<scalar_t>(), gamma_grad.data_ptr<scalar_t>(), beta_grad.data_ptr<scalar_t>(),
            x.options());
      });

  return std::make_tuple(x_grad, gamma_grad, beta_grad);
}

torch::Tensor GroupLayerNormCUDABwd(
    const torch::Tensor& y_grad, const torch::Tensor& x,
    int64_t num_channels, int64_t num_groups,
    const torch::Tensor& mean, const torch::Tensor& rstd,
    bool memory_efficient) {
  const int64_t channels_per_group = group_norm::CheckShapes(x, num_channels, num_groups);
  group_norm::CheckGradient(y_grad, x);
  const auto x_sizes = x.sizes();
  const int64_t outer_size = c10::multiply_integers(x_sizes.cbegin(), x_sizes.cbegin() + 2);
  group_norm::CheckStats(mean, x, outer_size * num_groups, "mean");
  group_norm::CheckStats(rstd, x, outer_size * num_groups, "rstd");
  at::cuda::OptionalCUDAGuard guard(at::device_of(x));

  const torch::Tensor y_grad_c = y_grad.contiguous();
  const torch::Tensor x_c = x.contiguous();
  const torch::Tensor mean_c = mean.contiguous();
  const torch::Tensor rstd_c = rstd.contiguous();
  torch::Tensor x_grad = torch::empty_like(x_c);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "GroupLayerNormCUDABwd", [&]() {
        using T_ACC = at::acc_type<scalar_t, true>;
        group_norm::GroupNormBwd<scalar_t, false>(
            outer_size, num_groups, channels_per_group, memory_efficient,
            y_grad_c.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), mean_c.data_ptr<T_ACC>(),
            rstd_c.data_ptr<T_ACC>(), nullptr, nullptr, x_grad.data_ptr<scalar_t>(), nullptr, nullptr,
            x.options());
      });

  return x_grad;
}

}  // namespace ops
}  // namespace xllm
