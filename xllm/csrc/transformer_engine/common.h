/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_COMMON_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_COMMON_H_

#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>

#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <cassert>
#include <cstring>
#include <iostream>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <vector>

#include "c10/util/ArrayRef.h"
#include "common/util/logging.h"

namespace transformer_engine::pytorch {


std::vector<size_t> getTensorShape(at::Tensor t);

inline size_t typeToNumBits(transformer_engine::DType t) {
  switch (t) {
    case transformer_engine::DType::kInt64:
      return 64;
    case transformer_engine::DType::kInt32:
    case transformer_engine::DType::kFloat32:
      return 32;
    case transformer_engine::DType::kInt16:
    case transformer_engine::DType::kFloat16:
    case transformer_engine::DType::kBFloat16:
      return 16;
    case transformer_engine::DType::kByte:
    case transformer_engine::DType::kFloat8E4M3:
    case transformer_engine::DType::kFloat8E5M2:
      return 8;
    case transformer_engine::DType::kFloat4E2M1:
      return 4;
    default:
      NVTE_ERROR("Invalid type");
  }
}

inline at::ScalarType GetATenDType(transformer_engine::DType t) {
  switch (t) {
    case transformer_engine::DType::kInt16:
      return torch::kInt16;
    case transformer_engine::DType::kInt32:
      return torch::kInt32;
    case transformer_engine::DType::kInt64:
      return torch::kInt64;
    case transformer_engine::DType::kFloat32:
      return at::kFloat;
    case transformer_engine::DType::kFloat16:
      return at::kHalf;
    case transformer_engine::DType::kBFloat16:
      return at::kBFloat16;
    case transformer_engine::DType::kByte:
      return at::kByte;
    case transformer_engine::DType::kFloat8E4M3:
      return at::kFloat8_e4m3fn;
    case transformer_engine::DType::kFloat8E5M2:
      return at::kFloat8_e5m2;
    default:
      NVTE_ERROR("Invalid type");
  }
}

inline transformer_engine::DType GetTransformerEngineDType(at::ScalarType t) {
  switch (t) {
    case at::kFloat8_e4m3fn:
      return transformer_engine::DType::kFloat8E4M3;
    case at::kFloat8_e5m2:
      return transformer_engine::DType::kFloat8E5M2;
    case at::kHalf:
      return transformer_engine::DType::kFloat16;
    case at::kFloat:
      return transformer_engine::DType::kFloat32;
    case at::kBFloat16:
      return transformer_engine::DType::kBFloat16;
    case at::kBool:
      return transformer_engine::DType::kByte;
    case torch::kByte:
      return transformer_engine::DType::kByte;
    case torch::kInt16:
      return transformer_engine::DType::kInt16;
    case torch::kInt32:
      return transformer_engine::DType::kInt32;
    case torch::kInt64:
      return transformer_engine::DType::kInt64;
    default:
      std::cout << "Type: " << static_cast<int>(t) << std::endl;
      NVTE_ERROR("Invalid type");
  }
}

inline transformer_engine::DType GetTransformerEngineDType(int DType_value) {
  return static_cast<transformer_engine::DType>(DType_value);
}



TensorWrapper makeTransformerEngineTensor(py::handle tensor, py::handle quantizer);

transformer_engine::TensorWrapper makeTransformerEngineTensor(at::Tensor tensor);

transformer_engine::TensorWrapper makeTransformerEngineTensor(void* data_ptr,
                                                              const std::vector<size_t>& shape,
                                                              const transformer_engine::DType type);

template <typename T>
T product(const std::vector<T>& shape);

size_t product(const NVTEShape& shape, size_t begin, size_t end);

std::vector<size_t> nvte_shape_to_vector(const NVTEShape& nvte_shape);

at::Tensor allocateSpace(const std::vector<size_t>& shape, const transformer_engine::DType type,
                         bool init_to_zeros);

at::Tensor allocateSpace(const NVTEShape& shape, const transformer_engine::DType type,
                         bool init_to_zeros = false);

at::Tensor allocateTorchTensor(int M, int N, transformer_engine::DType dtype);

at::Tensor allocateTorchTensor(int M, transformer_engine::DType dtype);

void* getDataPtr(at::Tensor tensor, int offset = 0);

std::vector<size_t> convertShape(const NVTEShape& shape);

int roundup(const int value, const int multiple);

NVTEShape convertTorchShape(const c10::IntArrayRef torch_shape);
}  // namespace transformer_engine::pytorch

namespace std {
template <typename T>
string to_string(const vector<T>& vec) {
  string ret = "[";
  for (const auto& val : vec) {
    ret += to_string(val) + ",";
  }
  if (ret.size() > 1) {
    ret[ret.size() - 1] = ']';
  } else {
    ret += "]";
  }
  return ret;
}

// Torch shape -> string
template <typename T>
string to_string(const c10::ArrayRef<T>& vec) {
  string ret = "[";
  for (const auto& val : vec) {
    ret += to_string(val) + ",";
  }
  if (ret.size() > 1) {
    ret[ret.size() - 1] = ']';
  } else {
    ret += "]";
  }
  return ret;
}

inline string to_string(const NVTEShape& s) {
  string ret = "[";
  for (size_t i = 0; i < s.ndim; ++i) {
    ret += to_string(s.data[i]) + ",";
  }
  if (ret.size() > 1) {
    ret[ret.size() - 1] = ']';
  } else {
    ret += "]";
  }
  return ret;
}
}  // namespace std

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_COMMON_H_
