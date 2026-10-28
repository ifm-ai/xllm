#include "ops/muon.h"

#include <ATen/Dispatch.h>

#include <cmath>

namespace xllm {
namespace ops {

namespace {

void CheckPairs(const std::vector<torch::Tensor>& contiguous, const std::vector<torch::Tensor>& strided) {
  TORCH_CHECK(contiguous.size() == strided.size(), "expected lists of one length, got ", contiguous.size(),
              " and ", strided.size());
  for (size_t i = 0; i < contiguous.size(); ++i) {
    TORCH_CHECK(contiguous[i].scalar_type() == torch::kFloat && contiguous[i].is_contiguous(),
                "expected contiguous fp32 tensors");
    TORCH_CHECK(strided[i].scalar_type() == torch::kBFloat16, "expected bf16 views");
    TORCH_CHECK(contiguous[i].numel() == strided[i].numel(), "tensor ", i, ": ", contiguous[i].numel(),
                " elements against ", strided[i].numel());
    TORCH_CHECK(contiguous[i].device() == strided[i].device(), "expected tensors on one device");
  }
}

// f(k, offset) for the k-th element of `t` in row-major order, at `offset` in its storage.
template <typename F>
void ForEachElement(const torch::Tensor& t, F f) {
  const StridedView v = AsStridedView(t);
  int64_t k = 0;
  for (int64_t b = 0; b < v.B; ++b) {
    for (int64_t r = 0; r < v.R; ++r) {
      for (int64_t c = 0; c < v.C; ++c) {
        f(k++, b * v.sB + r * v.sR + c * v.sC);
      }
    }
  }
}

}  // namespace

void MuonMomentumCPU(const std::vector<torch::Tensor>& momentum, const std::vector<torch::Tensor>& grad,
                     const std::vector<torch::Tensor>& out, double mu, bool nesterov) {
  const float mu_f = static_cast<float>(mu);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, grad[0].scalar_type(), "MuonMomentumCPU", [&] {
    for (size_t i = 0; i < momentum.size(); ++i) {
      float* m = momentum[i].data_ptr<float>();
      const scalar_t* g = grad[i].data_ptr<scalar_t>();
      at::BFloat16* o = out[i].data_ptr<at::BFloat16>();
      ForEachElement(out[i], [&](int64_t k, int64_t offset) {
        const float gk = static_cast<float>(g[k]);
        const float mk = std::fma(mu_f, m[k], gk);
        m[k] = mk;
        o[offset] = at::BFloat16(nesterov ? std::fma(mu_f, mk, gk) : mk);
      });
    }
  });
}

void MuonUpdateCPU(const std::vector<torch::Tensor>& param, const std::vector<torch::Tensor>& update,
                   const std::vector<double>& scale, double lr, double weight_decay) {
  const float lr_f = static_cast<float>(lr), wd = static_cast<float>(weight_decay);
  for (size_t i = 0; i < param.size(); ++i) {
    const float s = static_cast<float>(scale[i]);
    float* p = param[i].data_ptr<float>();
    const at::BFloat16* u = update[i].data_ptr<at::BFloat16>();
    ForEachElement(update[i], [&](int64_t k, int64_t offset) {
      p[k] = std::fma(-lr_f, std::fma(wd, p[k], s * static_cast<float>(u[offset])), p[k]);
    });
  }
}

void BatchedCopyCPU(const std::vector<torch::Tensor>& dst, const std::vector<torch::Tensor>& src) {
  AT_DISPATCH_REDUCED_FLOATING_TYPES(src[0].scalar_type(), "BatchedCopyCPU", [&] {
    using S = scalar_t;
    AT_DISPATCH_REDUCED_FLOATING_TYPES(dst[0].scalar_type(), "BatchedCopyCPU", [&] {
      for (size_t i = 0; i < dst.size(); ++i) {
        const StridedView sv = AsStridedView(src[i]);
        const S* s = src[i].data_ptr<S>();
        scalar_t* d = dst[i].data_ptr<scalar_t>();
        ForEachElement(dst[i], [&](int64_t k, int64_t offset) {
          const int64_t c = k % sv.C, r = k / sv.C % sv.R, b = k / (sv.C * sv.R);
          d[offset] = scalar_t(static_cast<float>(s[b * sv.sB + r * sv.sR + c * sv.sC]));
        });
      }
    });
  });
}

void MuonMomentum(const std::vector<torch::Tensor>& momentum, const std::vector<torch::Tensor>& grad,
                  const std::vector<torch::Tensor>& out, double mu, bool nesterov) {
  CheckPairs(momentum, out);
  TORCH_CHECK(grad.size() == momentum.size(), "expected one gradient per momentum");
  if (momentum.empty()) {
    return;
  }
  for (size_t i = 0; i < grad.size(); ++i) {
    TORCH_CHECK(grad[i].scalar_type() == grad[0].scalar_type() && grad[i].is_contiguous(),
                "expected contiguous gradients of one dtype");
    TORCH_CHECK(grad[i].numel() == momentum[i].numel() && grad[i].device() == momentum[i].device(),
                "gradient ", i, " does not match its momentum");
  }
  if (momentum[0].is_cuda()) {
    MuonMomentumCUDA(momentum, grad, out, mu, nesterov);
  } else {
    MuonMomentumCPU(momentum, grad, out, mu, nesterov);
  }
}

void MuonUpdate(const std::vector<torch::Tensor>& param, const std::vector<torch::Tensor>& update,
                const std::vector<double>& scale, double lr, double weight_decay) {
  CheckPairs(param, update);
  TORCH_CHECK(scale.size() == param.size(), "expected one scale per parameter");
  if (param.empty()) {
    return;
  }
  if (param[0].is_cuda()) {
    MuonUpdateCUDA(param, update, scale, lr, weight_decay);
  } else {
    MuonUpdateCPU(param, update, scale, lr, weight_decay);
  }
}

void BatchedCopy(const std::vector<torch::Tensor>& dst, const std::vector<torch::Tensor>& src) {
  TORCH_CHECK(dst.size() == src.size(), "expected lists of one length, got ", dst.size(), " and ", src.size());
  if (dst.empty()) {
    return;
  }
  for (size_t i = 0; i < dst.size(); ++i) {
    TORCH_CHECK(dst[i].sizes() == src[i].sizes(), "copy ", i, ": shapes ", dst[i].sizes(), " and ", src[i].sizes());
    TORCH_CHECK(dst[i].scalar_type() == dst[0].scalar_type() && src[i].scalar_type() == src[0].scalar_type(),
                "expected one dtype per list");
    TORCH_CHECK(dst[i].device() == dst[0].device() && src[i].device() == dst[0].device(),
                "expected tensors on one device");
  }
  if (dst[0].is_cuda()) {
    BatchedCopyCUDA(dst, src);
  } else {
    BatchedCopyCPU(dst, src);
  }
}

void DefineMuonOp(py::module& m) {
  m.def("muon_momentum_", &MuonMomentum, "Muon momentum: m <- mu m + g; out <- bf16(nesterov ? g + mu m : m)")
      .def("muon_update_", &MuonUpdate, "Muon update: p <- p - lr (wd p + scale update)")
      .def("batched_copy_", &BatchedCopy, "dst[i] <- src[i], views of one shape in any layouts");
}

}  // namespace ops
}  // namespace xllm
