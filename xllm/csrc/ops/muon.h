#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "utils.h"

namespace xllm {
namespace ops {

// Muon's momentum and update, each fused over a list of tensors. The bf16 tensors are views of 2 or 3 dims
// in any layout (blocks of a packed buffer, maybe transposed); element k of a view in row-major order pairs
// with element k of the matching contiguous fp32 tensor.

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

// m <- mu m + g; out <- bf16(nesterov ? g + mu m : m), each a fused multiply-add (one rounding).
// m fp32, g fp32 or bf16, both contiguous.
void MuonMomentum(const std::vector<torch::Tensor>& momentum, const std::vector<torch::Tensor>& grad,
                  const std::vector<torch::Tensor>& out, double mu, bool nesterov);

void MuonMomentumCPU(const std::vector<torch::Tensor>& momentum, const std::vector<torch::Tensor>& grad,
                     const std::vector<torch::Tensor>& out, double mu, bool nesterov);

void MuonMomentumCUDA(const std::vector<torch::Tensor>& momentum, const std::vector<torch::Tensor>& grad,
                      const std::vector<torch::Tensor>& out, double mu, bool nesterov);

// p <- p - lr (wd p + scale[i] update[i]), in fp32 with fused multiply-adds. lr multiplies the decay rather
// than forming 1 - lr wd, whose fp32 rounding would bias the decay of every step (by up to ~1e-3 of it).
void MuonUpdate(const std::vector<torch::Tensor>& param, const std::vector<torch::Tensor>& update,
                const std::vector<double>& scale, double lr, double weight_decay);

void MuonUpdateCPU(const std::vector<torch::Tensor>& param, const std::vector<torch::Tensor>& update,
                   const std::vector<double>& scale, double lr, double weight_decay);

void MuonUpdateCUDA(const std::vector<torch::Tensor>& param, const std::vector<torch::Tensor>& update,
                    const std::vector<double>& scale, double lr, double weight_decay);

// dst[i] <- src[i] for views of one shape in any layouts, bf16 or fp16 (one dtype per list); the gathers and
// scatters of Muon's all-to-alls.
void BatchedCopy(const std::vector<torch::Tensor>& dst, const std::vector<torch::Tensor>& src);

void BatchedCopyCPU(const std::vector<torch::Tensor>& dst, const std::vector<torch::Tensor>& src);

void BatchedCopyCUDA(const std::vector<torch::Tensor>& dst, const std::vector<torch::Tensor>& src);

void DefineMuonOp(py::module& m);

}  // namespace ops
}  // namespace xllm
