#pragma once

#include <c10/util/Optional.h>
#include <torch/torch.h>

#include <tuple>

namespace xllm{
namespace ops{

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor> CEMACUBScanFwd(const torch::Tensor& x,
                                                        const torch::Tensor& p,
                                                        const torch::Tensor& q,
                                                        const torch::Tensor& gamma,
                                                        const c10::optional<torch::Tensor>& bos_mask,
                                                        const c10::optional<torch::Tensor>& h0);
std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor> CEMACUBScanCUDAFwd(const torch::Tensor& x,
                                                            const torch::Tensor& p,
                                                            const torch::Tensor& q,
                                                            const torch::Tensor& gamma,
                                                            const c10::optional<torch::Tensor>& bos_mask,
                                                            const c10::optional<torch::Tensor>& h0);

std::tuple<torch::Tensor, torch::Tensor> CEMACUBScanFwdRecalc(const torch::Tensor& x,
                                                              const torch::Tensor& p,
                                                              const torch::Tensor& q,
                                                              const torch::Tensor& gamma,
                                                              const c10::optional<torch::Tensor>& bos_mask,
                                                              const torch::Tensor& chunk_decay,
                                                              const torch::Tensor& chunk_gain);

std::tuple<torch::Tensor, torch::Tensor> CEMACUBScanCUDAFwdRecalc(const torch::Tensor& x,
                                                                  const torch::Tensor& p,
                                                                  const torch::Tensor& q,
                                                                  const torch::Tensor& gamma,
                                                                  const c10::optional<torch::Tensor>& bos_mask,
                                                                  const torch::Tensor& chunk_decay,
                                                                  const torch::Tensor& chunk_gain);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
torch::Tensor, torch::Tensor> CEMACUBScanBwd(const torch::Tensor& y_grad,
                                             const c10::optional<torch::Tensor>& h_last_grad,
                                             const torch::Tensor& chunk_q_prefix,
                                             const torch::Tensor& chunk_h_prefix,
                                             const torch::Tensor& x,
                                             const torch::Tensor& p,
                                             const torch::Tensor& q,
                                             const torch::Tensor& gamma,
                                             const c10::optional<torch::Tensor>& bos_mask);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
torch::Tensor, torch::Tensor> CEMACUBScanCUDABwd(const torch::Tensor& y_grad,
                                                 const c10::optional<torch::Tensor>& h_last_grad,
                                                 const torch::Tensor& chunk_q_prefix,
                                                 const torch::Tensor& chunk_h_prefix,
                                                 const torch::Tensor& x,
                                                 const torch::Tensor& p,
                                                 const torch::Tensor& q,
                                                 const torch::Tensor& gamma,
                                                 const c10::optional<torch::Tensor>& bos_mask);

void DefineCEMACUBScanOp(py::module& m);

}  // namespace ops
}  // namespace xllm
