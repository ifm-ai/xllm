#pragma once

#include <torch/torch.h>

#include <tuple>

#include "utils.h"

namespace xllm {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor> AttentionFwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    double scale, double dropout, bool use_causal_mask);

std::tuple<torch::Tensor, torch::Tensor> AttentionCUDAFwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    double scale, double dropout, bool use_causal_mask);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> AttentionBwd(
    const torch::Tensor& grad_y, const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& w, double scale,
    bool use_causal_mask);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> AttentionCUDABwd(
    const torch::Tensor& grad_y, const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& w, double scale,
    bool use_causal_mask);

void AttentionCUDABwdOut(const torch::Tensor& grad_y, const torch::Tensor& q,
                         const torch::Tensor& k, const torch::Tensor& v,
                         const torch::Tensor& w, double scale,
                         bool use_causal_mask, torch::Tensor q_grad,
                         torch::Tensor k_grad, torch::Tensor v_grad,
                         int64_t kv_acc_rows);

void AttentionCUDABwdDsOut(const torch::Tensor& grad_y, const torch::Tensor& q,
                           const torch::Tensor& k, const torch::Tensor& v,
                           const torch::Tensor& w, const torch::Tensor& ds,
                           double scale, torch::Tensor q_grad,
                           torch::Tensor k_grad, torch::Tensor v_grad,
                           int64_t kv_acc_rows);

void DefineAttentionOp(py::module& m);

std::tuple<torch::Tensor, torch::Tensor> MultiSegAttentionFwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& q_segment_idx, const torch::Tensor& k_segment_idx,
    double scale, double dropout, bool use_causal_mask);

std::tuple<torch::Tensor, torch::Tensor> MultiSegAttentionCUDAFwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& q_segment_idx, const torch::Tensor& k_segment_idx,
    double scale, double dropout, bool use_causal_mask);

// as MultiSegAttentionFwd, writing the attention probabilities into w (contiguous [B, N, L1, L2], dtype of v)
void MultiSegAttentionFwdOut(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& q_segment_idx, const torch::Tensor& k_segment_idx,
    double scale, double dropout, bool use_causal_mask, torch::Tensor w,
    torch::Tensor y);

void MultiSegAttentionCUDAFwdOut(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& q_segment_idx, const torch::Tensor& k_segment_idx,
    double scale, double dropout, bool use_causal_mask, torch::Tensor w,
    torch::Tensor y);

void DefineMultiSegAttentionOp(py::module& m);


}  // namespace ops
}  // namespace xllm
