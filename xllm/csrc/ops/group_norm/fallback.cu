// The generic kernels (generic.cuh) for every call the dispatcher (dispatch.cu) routes to them, launched straight into
// the call's outputs; only the affine backward allocates, for its partial sums.
#include "dispatch.h"
#include "generic.cuh"

namespace xllm::ops::blackwell {
namespace {

// The operands of a Call are contiguous (dispatch.h), so their data pointers are taken as they are.
template <typename T, bool kRMS>
void launch_forward(const Shape& s, const Call& c) {
  generic::GroupNormFwd<T, kRMS>(
      s.rows, s.groups, s.cpg, c.eps, c.x_or_y->const_data_ptr<T>(),
      s.affine ? c.gamma->const_data_ptr<T>() : nullptr,
      !kRMS && s.affine ? c.beta->const_data_ptr<T>() : nullptr,
      c.y->data_ptr<T>(), kRMS ? nullptr : c.mean_out->data_ptr<float>(), c.rstd_out->data_ptr<float>());
}

template <typename T, bool kRMS>
void launch_backward(const Shape& s, const Call& c) {
  generic::GroupNormBwd<T, kRMS>(
      s.rows, s.groups, s.cpg, s.memory_efficient, c.dy->const_data_ptr<T>(), c.x_or_y->const_data_ptr<T>(),
      kRMS ? nullptr : c.mean->const_data_ptr<float>(), c.rstd->const_data_ptr<float>(),
      s.affine ? c.gamma->const_data_ptr<T>() : nullptr,
      !kRMS && s.affine ? c.beta->const_data_ptr<T>() : nullptr,
      c.dx->data_ptr<T>(), s.affine ? c.dgamma->data_ptr<T>() : nullptr,
      !kRMS && s.affine ? c.dbeta->data_ptr<T>() : nullptr);
}

}  // namespace

void fallback_forward(const Shape& s, const Call& c) {
  XLLM_DISPATCH_GROUP_NORM_TYPES(s.dtype, "GroupNormFallbackForward", [&]() {
    if (s.norm == kRMS) launch_forward<scalar_t, true>(s, c);
    else launch_forward<scalar_t, false>(s, c);
  });
}

void fallback_backward(const Shape& s, const Call& c) {
  XLLM_DISPATCH_GROUP_NORM_TYPES(s.dtype, "GroupNormFallbackBackward", [&]() {
    if (s.norm == kRMS) launch_backward<scalar_t, true>(s, c);
    else launch_backward<scalar_t, false>(s, c);
  });
}

}  // namespace xllm::ops::blackwell
