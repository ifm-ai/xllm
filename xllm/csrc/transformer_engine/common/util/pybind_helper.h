/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_PYBIND_HELPER_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_PYBIND_HELPER_H_

#include <pybind11/pybind11.h>
#include <transformer_engine/transformer_engine.h>

#include "cuda_runtime.h"

#define NVTE_DECLARE_COMMON_PYBIND11_HANDLES(m)                                                    \
  pybind11::enum_<transformer_engine::DType>(m, "DType", pybind11::module_local())                 \
      .value("kByte", transformer_engine::DType::kByte)                                            \
      .value("kInt32", transformer_engine::DType::kInt32)                                          \
      .value("kFloat32", transformer_engine::DType::kFloat32)                                      \
      .value("kFloat16", transformer_engine::DType::kFloat16)                                      \
      .value("kBFloat16", transformer_engine::DType::kBFloat16)                                    \
      .value("kFloat8E4M3", transformer_engine::DType::kFloat8E4M3)                                \
      .value("kFloat8E5M2", transformer_engine::DType::kFloat8E5M2);                               \
  m.def(                                                                                           \
      "get_stream_priority_range",                                                                 \
      [](int device_id = -1) {                                                                     \
        int low_pri, high_pri;                                                                     \
        transformer_engine::cuda::stream_priority_range(&low_pri, &high_pri, device_id);           \
        return std::make_pair(low_pri, high_pri);                                                  \
      },                                                                                           \
      py::call_guard<py::gil_scoped_release>(), py::arg("device_id") = -1);                        \

#endif
