#include <torch/torch.h>

#include "ops/attention.h"
#include "ops/ema_hidden.h"
#include "ops/ema_parameters.h"
#include "ops/cema_blelloch_scan.h"
#include "ops/cema_cub_scan.h"
#include "ops/fftconv.h"
#include "ops/group_rms_norm.h"
#include "ops/group_layer_norm.h"
#include "ops/sequence_norm.h"
#include "ops/timestep_norm.h"
#include "ops/timestep_decay_norm_cub.h"
#include "utils.h"

#include <transformer_engine/common.h>
#include <transformer_engine/extensions.h>
#include <transformer_engine/common/common.h>
#include <transformer_engine/common/util/pybind_helper.h>

namespace xllm {

PYBIND11_MODULE(xllm_extension, m) {
  m.doc() = "XLLM Cpp Extensions.";
  
  py::module m_te = m.def_submodule("te", "Submodule for Transformer Engine.");
  NVTE_DECLARE_COMMON_PYBIND11_HANDLES(m_te)
  m_te.def("general_grouped_gemm", &transformer_engine::pytorch::te_general_grouped_gemm,
        "Grouped GEMM");
  
  py::module m_ops = m.def_submodule("ops", "Submodule for custom ops.");
  ops::DefineAttentionOp(m_ops);
  ops::DefineMultiSegAttentionOp(m_ops);
  ops::DefineEMAHiddenOp(m_ops);
  ops::DefineEMAParametersOp(m_ops);
  ops::DefineCEMABlellochScanOp(m_ops);
  ops::DefineCEMACUBScanOp(m_ops);
  ops::DefineFFTConvOp(m_ops);
  ops::DefineGroupLayerNormOp(m_ops);
  ops::DefineGroupRMSNormOp(m_ops);
  ops::DefineSequenceNormOp(m_ops);
  ops::DefineTimestepNormOp(m_ops);
  ops::DefineTimestepDecayNormCubOp(m_ops);
}

}  // namespace xllm
