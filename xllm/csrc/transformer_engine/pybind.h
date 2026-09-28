/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#define PYBIND11_DETAILED_ERROR_MESSAGES  // TODO remove

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_PYBIND_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_PYBIND_H_

#include <Python.h>
#include <pybind11/detail/common.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include "common.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine::pytorch {

#define NVTE_SCOPED_GIL_RELEASE(code_block)      \
  do {                                           \
    if (PyGILState_Check()) {                    \
      pybind11::gil_scoped_release _gil_release; \
      code_block                                 \
    } else {                                     \
      code_block                                 \
    }                                            \
  } while (false);

}  // namespace transformer_engine::pytorch

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_PYBIND_H_
