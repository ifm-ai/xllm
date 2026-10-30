#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "utils.h"

namespace xllm {
namespace ops {

// Muon's elementwise steps, each one launch over a whole list of tensors.

// m <- mu m + g; out <- bf16(nesterov ? mu m + g : m), each a fused multiply-add (one rounding).
// m fp32, g fp32 / bf16 / fp16, out bf16; all contiguous, out[i] of any shape with m[i]'s numel.
void MuonMomentum(const std::vector<torch::Tensor>& momentum, const std::vector<torch::Tensor>& grad,
                  const std::vector<torch::Tensor>& out, double mu, bool nesterov);

// p <- p - lr (wd p + scale[i] u), in fp32 with fused multiply-adds. lr multiplies the decay rather than
// forming 1 - lr wd, whose fp32 rounding would bias the decay of every step (by up to ~1e-3 of it).
// p fp32, u bf16; all contiguous, u[i] of any shape with p[i]'s numel.
void MuonUpdate(const std::vector<torch::Tensor>& param, const std::vector<torch::Tensor>& update,
                const std::vector<double>& scale, double lr, double weight_decay);

// dst[i] <- src[i] for 2-d / 3-d views of one shape in any layouts, bf16 or fp16 (one dtype per list): the
// gathers and scatters around Muon's all-to-alls.
void BatchedCopy(const std::vector<torch::Tensor>& dst, const std::vector<torch::Tensor>& src);

void MuonMomentumCPU(const std::vector<torch::Tensor>& momentum, const std::vector<torch::Tensor>& grad,
                     const std::vector<torch::Tensor>& out, double mu, bool nesterov);
void MuonMomentumCUDA(const std::vector<torch::Tensor>& momentum, const std::vector<torch::Tensor>& grad,
                      const std::vector<torch::Tensor>& out, double mu, bool nesterov);
void MuonUpdateCPU(const std::vector<torch::Tensor>& param, const std::vector<torch::Tensor>& update,
                   const std::vector<double>& scale, double lr, double weight_decay);
void MuonUpdateCUDA(const std::vector<torch::Tensor>& param, const std::vector<torch::Tensor>& update,
                    const std::vector<double>& scale, double lr, double weight_decay);
void BatchedCopyCPU(const std::vector<torch::Tensor>& dst, const std::vector<torch::Tensor>& src);
void BatchedCopyCUDA(const std::vector<torch::Tensor>& dst, const std::vector<torch::Tensor>& src);

// A 2-d or 3-d view as [B, R, C] with its strides.
struct StridedView {
  int64_t B, R, C;
  int64_t sB, sR, sC;
};

inline StridedView AsStridedView(const torch::Tensor& t) {
  TORCH_CHECK(t.dim() == 2 || t.dim() == 3, "expected a 2-d or 3-d view, got ", t.dim(), " dims");
  if (t.dim() == 2) {
    return {1, t.size(0), t.size(1), 0, t.stride(0), t.stride(1)};
  }
  return {t.size(0), t.size(1), t.size(2), t.stride(0), t.stride(1), t.stride(2)};
}

void DefineMuonOp(py::module& m);

}  // namespace ops
}  // namespace xllm
