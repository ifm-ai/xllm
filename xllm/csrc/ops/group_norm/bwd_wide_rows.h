// The resident wide-row backward (bwd_wide_rows.cu): one 65536-channel bf16 segment across a 4-CTA cluster, for calls
// without an affine transform. Called by dispatch.cu's backward().
#pragma once

#include "dispatch.h"

namespace xllm::ops::blackwell {

// Launches the kernel and returns true for a bf16 call without an affine transform whose groups are exactly 65536
// channels, with contiguous 16-byte aligned operands; false (nothing launched) for every other call.
bool backward_wide_rows(const Shape& s, const Call& c);

}  // namespace xllm::ops::blackwell
