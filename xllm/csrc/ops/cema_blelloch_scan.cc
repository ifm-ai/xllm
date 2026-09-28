#include "ops/cema_blelloch_scan.h"

namespace xllm{
namespace ops{

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor> CEMAScanFwd(const torch::Tensor& x,
                                                     const torch::Tensor& p,
                                                     const torch::Tensor& q,
                                                     const torch::Tensor& gamma,
                                                     const c10::optional<torch::Tensor>& bos_mask,
                                                     const c10::optional<torch::Tensor>& h0){
    return CEMAScanCUDAFwd(x, p, q, gamma, bos_mask, h0);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
torch::Tensor, torch::Tensor> CEMAScanBwd(const torch::Tensor& y_grad,
                                          const c10::optional<torch::Tensor>& h_last_grad,
                                          const torch::Tensor& chunk_decay,
                                          const torch::Tensor& chunk_gain,
                                          const torch::Tensor& x,
                                          const torch::Tensor& p,
                                          const torch::Tensor& q,
                                          const torch::Tensor& gamma,
                                          const c10::optional<torch::Tensor>& bos_mask){
    return CEMAScanCUDABwd(y_grad, h_last_grad, chunk_decay, chunk_gain, x, p, q, gamma, bos_mask);
}

void DefineCEMABlellochScanOp(py::module& m) {
    m.def("cema_blelloch_scan_fwd", &CEMAScanFwd, "CEMAScanFwd")
        .def("cema_blelloch_scan_bwd", &CEMAScanBwd, "CEMAScanBwd");
}

}  // namespace ops
}  // namespace xllm
