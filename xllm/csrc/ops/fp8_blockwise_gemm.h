#pragma once

#include <torch/torch.h>

#include <optional>

#include "../utils.h"

namespace xllm {
namespace ops {

void Fp8BlockwiseGemm(const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
                      const torch::Tensor& b_scale, torch::Tensor& d, const std::optional<torch::Tensor>& c,
                      bool b_tile);

void DefineFp8BlockwiseGemmOp(py::module& m);

}  // namespace ops
}  // namespace xllm
