#include "ops/muon.h"

#include <ATen/Dispatch.h>

#include <cmath>

namespace xllm {
namespace ops {

namespace {

// fp32[i] and other[i]: contiguous, of one numel and device; other's tensors of one dtype.
void CheckPairs(const std::vector<torch::Tensor>& fp32, const std::vector<torch::Tensor>& other,
                const char* name) {
  TORCH_CHECK(fp32.size() == other.size(), "expected lists of one length, got ", fp32.size(), " and ",
              other.size());
  for (size_t i = 0; i < fp32.size(); ++i) {
    TORCH_CHECK(fp32[i].scalar_type() == torch::kFloat && fp32[i].is_contiguous(), name, " ", i,
                ": expected a contiguous fp32 tensor to pair with");
    TORCH_CHECK(other[i].is_contiguous() && other[i].scalar_type() == other[0].scalar_type(), name, " ", i,
                ": expected contiguous tensors of one dtype");
    TORCH_CHECK(fp32[i].numel() == other[i].numel() && fp32[i].device() == other[i].device(), name, " ", i,
                ": ", other[i].numel(), " elements against ", fp32[i].numel(), ", or on another device");
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
      for (int64_t k = 0; k < momentum[i].numel(); ++k) {
        const float gk = static_cast<float>(g[k]);
        m[k] = std::fma(mu_f, m[k], gk);
        o[k] = at::BFloat16(nesterov ? std::fma(mu_f, m[k], gk) : m[k]);
      }
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
    for (int64_t k = 0; k < param[i].numel(); ++k) {
      p[k] = std::fma(-lr_f, std::fma(wd, p[k], s * static_cast<float>(u[k])), p[k]);
    }
  }
}

void BatchedCopyCPU(const std::vector<torch::Tensor>& dst, const std::vector<torch::Tensor>& src) {
  AT_DISPATCH_REDUCED_FLOATING_TYPES(src[0].scalar_type(), "BatchedCopyCPU", [&] {
    using S = scalar_t;
    AT_DISPATCH_REDUCED_FLOATING_TYPES(dst[0].scalar_type(), "BatchedCopyCPU", [&] {
      for (size_t i = 0; i < dst.size(); ++i) {
        const StridedView dv = AsStridedView(dst[i]), sv = AsStridedView(src[i]);
        const S* s = src[i].data_ptr<S>();
        scalar_t* d = dst[i].data_ptr<scalar_t>();
        for (int64_t b = 0; b < dv.B; ++b) {
          for (int64_t r = 0; r < dv.R; ++r) {
            for (int64_t c = 0; c < dv.C; ++c) {
              d[b * dv.sB + r * dv.sR + c * dv.sC] = scalar_t(static_cast<float>(s[b * sv.sB + r * sv.sR + c * sv.sC]));
            }
          }
        }
      }
    });
  });
}

void MuonMomentum(const std::vector<torch::Tensor>& momentum, const std::vector<torch::Tensor>& grad,
                  const std::vector<torch::Tensor>& out, double mu, bool nesterov) {
  CheckPairs(momentum, grad, "gradient");
  CheckPairs(momentum, out, "output");
  if (momentum.empty()) {
    return;
  }
  TORCH_CHECK(out[0].scalar_type() == torch::kBFloat16, "expected bf16 outputs");
  if (momentum[0].is_cuda()) {
    MuonMomentumCUDA(momentum, grad, out, mu, nesterov);
  } else {
    MuonMomentumCPU(momentum, grad, out, mu, nesterov);
  }
}

void MuonUpdate(const std::vector<torch::Tensor>& param, const std::vector<torch::Tensor>& update,
                const std::vector<double>& scale, double lr, double weight_decay) {
  CheckPairs(param, update, "update");
  TORCH_CHECK(scale.size() == param.size(), "expected one scale per parameter");
  if (param.empty()) {
    return;
  }
  TORCH_CHECK(update[0].scalar_type() == torch::kBFloat16, "expected bf16 updates");
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
  m.def("muon_momentum_", &MuonMomentum, "Muon momentum: m <- mu m + g; out <- bf16(nesterov ? mu m + g : m)")
      .def("muon_update_", &MuonUpdate, "Muon update: p <- p - lr (wd p + scale u)")
      .def("batched_copy_", &BatchedCopy, "dst[i] <- src[i], views of one shape in any layouts");
}

}  // namespace ops
}  // namespace xllm
