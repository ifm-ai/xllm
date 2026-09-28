/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common/common.h"

#include "c10/util/ArrayRef.h"
#include "pybind.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine::pytorch {

std::vector<size_t> getTensorShape(at::Tensor t) {
  std::vector<size_t> shape;
  for (auto s : t.sizes()) {
    shape.push_back(s);
  }
  return shape;
}

NVTEShape convertTorchShape(const c10::IntArrayRef torch_shape) {
  NVTEShape ret;
  ret.ndim = torch_shape.size();
  constexpr int max_dimensions = sizeof(ret.data) / sizeof(size_t);
  NVTE_CHECK(ret.ndim < max_dimensions,
             "Torch tensor has too many dimensions. Max supported: ", max_dimensions, " and got ",
             ret.ndim, ".");
  for (size_t i = 0; i < ret.ndim; ++i) {
    const auto& v = torch_shape[i];
    ret.data[i] = static_cast<size_t>(v);
  }
  return ret;
}

TensorWrapper makeTransformerEngineTensor(py::handle tensor, py::handle quantizer) {
  NVTE_CHECK(!tensor.is_none(), "Tensor is not allocated!");
  // std::unique_ptr<Quantizer> my_quantizer = convert_quantizer(quantizer);

  // check for both quantizer & tensor type:
  // mxfp8 tensor -> mxfp8 quantizer
  // float8 tensor -> delayed scaling quantizer OR current scaling quantizer
  // also during dequantize, the quantizer param is unknown -> so quantizer is NoneQuantizer

  // NVTE_CHECK(dynamic_cast<NoneQuantizer*>(my_quantizer.get()) != nullptr,
  //            "Unexpected quantization params type.");

  // Regular pyTorch tensor
  at::Tensor torch_tensor = tensor.cast<at::Tensor>();

  // #TODO (pgadzinski) - needed in attention for non-contiguous tensors.
  //if (!torch_tensor.is_contiguous()) {
  //  torch_tensor = torch_tensor.contiguous();
  //}
  auto ret = TensorWrapper(NVTE_DELAYED_TENSOR_SCALING);
  ret.set_rowwise_data(torch_tensor.data_ptr(),
                       GetTransformerEngineDType(torch_tensor.scalar_type()),
                       getTensorShape(torch_tensor));
  return ret;
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr, const NVTEShape& shape, const transformer_engine::DType type) {
  return transformer_engine::TensorWrapper(data_ptr, shape, type);
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr, const std::vector<size_t>& shape, const transformer_engine::DType type) {
  return transformer_engine::TensorWrapper(data_ptr, shape, type);
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(at::Tensor tensor) {
  transformer_engine::DType dtype = GetTransformerEngineDType(tensor.scalar_type());
  std::vector<size_t> shape;
  for (auto s : tensor.sizes()) {
    shape.push_back(s);
  }
  return makeTransformerEngineTensor(tensor.data_ptr(), shape, dtype);
}

template <typename T>
T product(const std::vector<T>& shape) {
  T ret = 1;
  for (auto s : shape) {
    ret *= s;
  }
  return ret;
}

template size_t product<size_t>(const std::vector<size_t>& shape);
template int64_t product<int64_t>(const std::vector<int64_t>& shape);

size_t product(const NVTEShape& shape, size_t begin, size_t end) {
  NVTE_CHECK(begin <= end && end <= shape.ndim, "Attempted to access entries ", begin, " to ", end,
             " in a shape with ", shape.ndim, " entries");
  size_t ret = 1;
  for (size_t i = begin; i < end; ++i) {
    ret *= shape.data[i];
  }
  return ret;
}

std::vector<size_t> nvte_shape_to_vector(const NVTEShape& nvte_shape) {
  std::vector<size_t> shape;
  for (size_t i = 0; i < nvte_shape.ndim; i++) {
    shape.push_back(nvte_shape.data[i]);
  }
  return shape;
}

at::Tensor allocateSpace(const std::vector<size_t>& shape, const transformer_engine::DType type,
                         bool init_to_zeros) {
  std::vector<int64_t> shape_int64(shape.begin(), shape.end());
  c10::IntArrayRef ar_shape(shape_int64);
  if (init_to_zeros) {
    return at::zeros(ar_shape, at::CUDA(GetATenDType(type)));
  } else {
    return at::empty(ar_shape, at::CUDA(GetATenDType(type)));
  }
}

at::Tensor allocateSpace(const NVTEShape& shape, const transformer_engine::DType type,
                         bool init_to_zeros) {
  auto size = shape.ndim;
  if (size == 2 && init_to_zeros) {
    return at::zeros({static_cast<int64_t>(shape.data[0]), static_cast<int64_t>(shape.data[1])},
                     at::CUDA(GetATenDType(type)));
  } else if (size == 2) {
    return at::empty({static_cast<int64_t>(shape.data[0]), static_cast<int64_t>(shape.data[1])},
                     at::CUDA(GetATenDType(type)));
  } else if (size == 1 && init_to_zeros) {
    return at::zeros({static_cast<int64_t>(shape.data[0])}, at::CUDA(GetATenDType(type)));
  } else if (size == 1) {
    return at::empty({static_cast<int64_t>(shape.data[0])}, at::CUDA(GetATenDType(type)));
  }
  NVTE_CHECK(false, "Should never reach here! func: allocateSpace");
}

at::Tensor allocateTorchTensor(int M, int N, transformer_engine::DType dtype) {
  return at::empty({static_cast<int64_t>(M), static_cast<int64_t>(N)},
                   at::CUDA(GetATenDType(dtype)));
}

at::Tensor allocateTorchTensor(int M, transformer_engine::DType dtype) {
  return at::empty({static_cast<int64_t>(M)}, at::CUDA(GetATenDType(dtype)));
}

void* getDataPtr(at::Tensor tensor, int offset) {
  void* dptr = nullptr;
  if (tensor.numel() > 0) {
    dptr = tensor.data_ptr();
  }
  if (dptr != nullptr && offset != 0) {
    char* char_ptr = reinterpret_cast<char*>(dptr);
    char_ptr += offset * tensor.element_size();
    dptr = reinterpret_cast<void*>(char_ptr);
  }
  return dptr;
}

std::vector<size_t> convertShape(const NVTEShape& shape) {
  return std::vector<size_t>(shape.data, shape.data + shape.ndim);
}

int roundup(const int value, const int multiple) {
  assert(multiple > 0);
  return ((value + multiple - 1) / multiple) * multiple;
}

}  // namespace transformer_engine::pytorch
