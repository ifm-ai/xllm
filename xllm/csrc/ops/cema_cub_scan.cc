#include "ops/cema_cub_scan.h"

namespace xllm{
namespace ops{

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor> CEMACUBScanFwd(const torch::Tensor& x,
                                                         const torch::Tensor& p,
                                                         const torch::Tensor& q,
                                                         const torch::Tensor& gamma,
                                                         const c10::optional<torch::Tensor>& bos_mask,
                                                         const c10::optional<torch::Tensor>& h0){
    return CEMACUBScanCUDAFwd(x, p, q, gamma, bos_mask, h0);
}

std::tuple<torch::Tensor, torch::Tensor> CEMACUBScanFwdRecalc(const torch::Tensor& x,
                                                            const torch::Tensor& p,
                                                            const torch::Tensor& q,
                                                            const torch::Tensor& gamma,
                                                            const c10::optional<torch::Tensor>& bos_mask,
                                                            const torch::Tensor& chunk_decay,
                                                            const torch::Tensor& chunk_gain){
    return CEMACUBScanCUDAFwdRecalc(x, p, q, gamma, bos_mask, chunk_decay, chunk_gain);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
torch::Tensor, torch::Tensor> CEMACUBScanBwd(const torch::Tensor& y_grad,
                                            const c10::optional<torch::Tensor>& h_last_grad,
                                            const torch::Tensor& chunk_q_prefix,
                                            const torch::Tensor& chunk_h_prefix,
                                            const torch::Tensor& x,
                                            const torch::Tensor& p,
                                            const torch::Tensor& q,
                                            const torch::Tensor& gamma,
                                            const c10::optional<torch::Tensor>& bos_mask){
    return CEMACUBScanCUDABwd(y_grad, h_last_grad, chunk_q_prefix, chunk_h_prefix, x, p, q, gamma, bos_mask);
}

void DefineCEMACUBScanOp(py::module& m) {
    m.def("cema_cub_scan_fwd", &CEMACUBScanFwd, "CEMACUBScanFwd")
        .def("cema_cub_scan_fwd_recalc", &CEMACUBScanFwdRecalc, "CEMACUBScanFwdRecalc")
            .def("cema_cub_scan_bwd", &CEMACUBScanBwd, "CEMACUBScanBwd");
}

}  // namespace ops
}  // namespace xllm
