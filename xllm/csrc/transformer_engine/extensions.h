/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_H_

#include <optional>

#include "common/common.h"

namespace transformer_engine::pytorch {

/***************************************************************************************************
 * GEMM
 **************************************************************************************************/

using MaybeTensor = std::optional<at::Tensor>;


std::optional<std::vector<at::Tensor>> te_general_grouped_gemm(
    std::vector<py::handle> A, bool transa, std::vector<py::handle> B, bool transb,
    std::optional<std::vector<at::Tensor>> D, DType D_type, std::vector<int64_t> m_splits,
    std::vector<at::Tensor> bias, DType bias_type, bool single_output,
    std::vector<at::Tensor> pre_gelu_out, bool grad, std::vector<at::Tensor> workspace,
    size_t workspaceSize, bool accumulate, bool use_split_accumulator, int math_sm_count);

}  // namespace transformer_engine::pytorch

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_H_
