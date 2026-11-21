// Routes every call of the public ops (native.cpp). portable() names the classes where upstream's portable kernels are
// the fastest: native.cpp runs upstream's own entry points for them. forward() and backward() send every other call to
// one of the specialized kernels or to the generic kernels (generic.cuh, through fallback.cu). Each rule names a class
// of shapes and the kernel measured fastest on it on B300 among those at most 5% slower than the portable kernels both
// on device time and as a public call. A specialized kernel may still decline a call (layout, alignment, launch
// limits); the call then goes on down the chain.
#include "dispatch.h"

#include <climits>
#include <cstdint>
#include <initializer_list>

#include "bwd_by_width.h"
#include "bwd_cluster_narrow.h"
#include "bwd_one_cluster_grad.h"
#include "bwd_one_cluster_grad512.h"
#include "bwd_one_pass.h"
#include "bwd_tma.h"
#include "bwd_wide_rows.h"
#include "dense_lanes.h"

namespace xllm::ops::blackwell {
namespace {

// Some specialized kernels read 16-byte vectors without checking alignment, and they index channels with int: a call
// with an input that is not 16-byte aligned (gamma as a view into a flat parameter buffer at an odd offset, for
// example) or with more than INT_MAX channels takes the generic kernels, which check each pointer.
bool generic_only(const Shape& s, std::initializer_list<const at::Tensor*> inputs) {
  if (s.channels > INT_MAX) return true;
  for (const at::Tensor* t : inputs)
    if (reinterpret_cast<uintptr_t>(t->const_data_ptr()) % 16 != 0) return true;
  return false;
}

// Dense lanes take groups of 1 to 32 vectors of 16 bytes (dense_lanes.h).
bool dense_lanes_fit(const Shape& s, const Call& c) { return dense_lanes::Eligible(*c.x_or_y, s.groups); }

void dense_lanes_forward(const Shape& s, const Call& c) {
  dense_lanes::Fwd(*c.x_or_y, s.groups, s.norm == kLayer, *c.gamma, *c.beta, c.eps, *c.y, *c.mean_out, *c.rstd_out);
}

void dense_lanes_backward(const Shape& s, const Call& c) {
  dense_lanes::Bwd(*c.x_or_y, *c.dy, s.groups, s.norm == kLayer, s.memory_efficient, *c.mean, *c.rstd, *c.gamma,
                   *c.beta, *c.dx, *c.dgamma, *c.dbeta);
}

// Upstream's portable kernels are the fastest forward on the widest groups when there are too few segments to fill
// the machine: groups of more than 16384 channels in at most 16 segments, and groups of more than 8192 channels that
// are not a whole number of 16-byte vectors in fewer than 4096 segments. A segment is one (row, group).
bool prefer_portable_forward(const Shape& s) {
  const int64_t segments = s.rows * s.groups;
  const bool unaligned = s.cpg % (s.dtype == at::kFloat ? 4 : 8) != 0;
  return (s.cpg > 16384 && segments <= 16) || (unaligned && s.cpg > 8192 && segments < 4096);
}

// The generic kernels are faster than the specialized ones on these forward classes.
bool prefer_generic_forward(const Shape& s) {
  const bool fp32 = s.dtype == at::kFloat;
  const int64_t segments = s.rows * s.groups;
  // Widths that are not a whole number of 16-byte vectors (the flat kernel's scalar path):
  const bool unaligned = s.cpg % (fp32 ? 4 : 8) != 0;
  // ... below 128 channels on 8192 segments or more,
  if (unaligned && s.cpg < 128 && segments >= 8192) return true;
  // ... from 256 channels on fewer than 4M elements, in fp32 below 4096 channels or on fewer than 2048 segments (the
  // flat kernel's one warp a segment needs that many to stream wide groups from memory), and in 2 bytes without affine
  // at odd widths below 2048 channels on at most 4096 segments.
  if (unaligned && s.cpg >= 256 &&
      (segments * s.cpg < (int64_t{1} << 22) || (fp32 && (s.cpg < 4096 || segments < 2048)) ||
       (!fp32 && !s.affine && s.cpg % 2 == 1 && s.cpg < 2048 && segments <= 4096)))
    return true;
  // 2-byte affine: 1024 channels (the RMS norm up to 2047) on at most 1024 segments, 4096 channels on at most 256.
  if (!fp32 && s.affine &&
      (((s.cpg == 1024 || (s.norm == kRMS && s.cpg > 1024 && s.cpg < 2048)) && segments <= 1024) ||
       (s.cpg == 4096 && segments <= 256)))
    return true;
  // fp32 groups of more than 1024 channels in an odd number of 16-byte vectors, on at most 512 segments.
  if (fp32 && s.cpg > 1024 && s.cpg % 8 == 4 && segments <= 512) return true;
  // fp32 RMS, 1024 to 2047 channels, from 2048 segments.
  if (fp32 && s.norm == kRMS && s.cpg >= 1024 && s.cpg < 2048 && segments >= 2048) return true;
  // fp32 affine layer norm: 1024 channels from 16384 segments, 1025 to 2047 channels on 4096 to 32767 segments.
  if (fp32 && s.norm == kLayer && s.affine &&
      ((s.cpg == 1024 && segments >= 16384) || (s.cpg > 1024 && s.cpg < 2048 && segments >= 4096 && segments < 32768)))
    return true;
  // fp32 without affine, 81 to 127 channels.
  if (fp32 && !s.affine && s.cpg > 80 && s.cpg < 128) return true;
  // 2-byte without affine: up to 32 channels below 65536 segments, up to 48 below 32768.
  if (!fp32 && !s.affine &&
      ((s.cpg <= 32 && segments < (int64_t{1} << 16)) || (s.cpg <= 48 && segments < (int64_t{1} << 15))))
    return true;
  // fp32 without affine, up to 32 channels below 524288 segments.
  return fp32 && !s.affine && s.cpg <= 32 && segments < (int64_t{1} << 19);
}

// Dense lanes run before the flat kernel on 2-byte affine groups of 40 channels, on groups without affine whose width
// is not a power of two, and on fp32 without affine when the dense-lane grid already fills the machine.
bool prefer_dense_lanes_forward(const Shape& s) {
  if (s.affine) return s.dtype != at::kFloat && s.cpg == 40;
  const bool pow2 = (s.cpg & (s.cpg - 1)) == 0;
  return !pow2 || (s.dtype == at::kFloat && dense_lanes::GridSaturated(s.rows, s.groups, s.cpg, 4));
}

// Upstream's portable kernels are the fastest on these backward classes.
bool prefer_portable_backward(const Shape& s) {
  const bool rms = s.norm == kRMS;
  const int64_t segments = s.rows * s.groups;
  if (s.dtype == at::kBFloat16 && s.affine && s.cpg % 8 != 0) {
    // bf16 affine, widths that are not a multiple of 8: RMS on x from 256 channels,
    return (rms && !s.memory_efficient && s.cpg >= 256 && segments >= 8192) ||
           // RMS on y below 33 channels from 1M segments, from 33 to 255 channels from 65536, and from 1024 channels in
           // groups of 8-byte vectors (4 mod 8),
           (rms && s.memory_efficient && s.cpg < 33 && segments >= (int64_t{1} << 20)) ||
           (rms && s.memory_efficient && s.cpg >= 33 && s.cpg < 256 && segments >= 65536) ||
           (rms && s.memory_efficient && s.cpg >= 1024 && s.cpg % 8 == 4 && segments >= 16384) ||
           // and the layer norm on y from 65 to 255 channels: in 32 groups or more from 262144 segments, and above 128
           // channels from 65536 segments.
           (!rms && s.memory_efficient && s.cpg >= 65 && s.cpg < 256 &&
            ((s.groups >= 32 && segments >= 262144) || (s.cpg > 128 && segments >= 65536)));
  }
  // fp32 affine layer norm on x, 385 to 768 channels, and fp32 affine RMS on y above 128 channels on at least 262144
  // segments.
  if (s.dtype != at::kFloat || !s.affine) return false;
  return (!rms && !s.memory_efficient && s.cpg > 384 && s.cpg <= 768 && segments >= 16384) ||
         (rms && s.memory_efficient && s.cpg > 128 && segments >= (int64_t{1} << 18));
}

// The TMA kernel's class (bwd_tma.cu): bf16 affine calls of at most 8192 channels in groups of whole 16-byte vectors,
// from x, or RMS from y at exactly 8192 channels.
bool tma_takes(const Shape& s) {
  if (s.dtype != at::kBFloat16 || !s.affine || s.cpg % 8 != 0 || s.channels > 8192) return false;
  const bool rms = s.norm == kRMS;
  // From y: RMS, at most 128 channels a group and 512 groups, 512 to 65535 rows.
  if (s.memory_efficient &&
      !(rms && s.channels == 8192 && s.cpg <= 128 && s.groups <= 512 && s.rows >= 512 && s.rows < 65536))
    return false;
  const int64_t V = s.cpg / 8;  // 16-byte vectors a group
  if (V > 32) {
    // RMS from x only: one warp a group at 4 to 6 vectors a lane (at most 8 groups), or groups of 128 vectors or more
    // split over at most 16 warps.
    if (!rms || s.memory_efficient) return false;
    const int64_t Q = (V + 31) / 32;
    return (Q >= 4 && Q <= 6 && s.groups <= 8) || (V >= 128 && s.groups * ((V + 127) / 128) <= 16);
  }
  // Whole groups a warp, at most 32 warps a row.
  const int64_t groups_per_warp = 32 / V;
  return (s.groups + groups_per_warp - 1) / groups_per_warp <= 32;
}

// The generic kernels are faster than every specialized kernel the chain still offers on these backward classes.
// Asked after the cluster kernels, which take their exact classes first.
bool prefer_generic_backward(const Shape& s) {
  const bool fp32 = s.dtype == at::kFloat;
  const bool bf16 = s.dtype == at::kBFloat16;
  const bool rms = s.norm == kRMS;
  const int64_t segments = s.rows * s.groups;
  // fp16 up to 384 channels a group (fp16 has none of the bf16-only kernels).
  if (s.dtype == at::kHalf && s.cpg <= 384) return true;
  // fp32 without affine, up to 128 channels below 524288 segments.
  if (fp32 && !s.affine && s.cpg <= 128 && segments < (int64_t{1} << 19)) return true;
  // fp32 affine up to 128 channels, except 257 to 511 rows.
  if (fp32 && s.affine && s.cpg <= 128 && (s.rows <= 256 || s.rows >= 512)) return true;
  // 2-byte without affine above 256 channels, except bf16 single groups of 8192 channels from 2048 rows.
  if (!fp32 && !s.affine && s.cpg > 256 && !(bf16 && s.groups == 1 && s.cpg == 8192 && s.rows >= 2048)) return true;
  // bf16 without affine up to 256 channels: on fewer than 8192 segments (too few warps for dense lanes), and the layer
  // norm from y, in at most 8 groups or on fewer than 131072 segments.
  if (bf16 && !s.affine && s.cpg <= 256 && segments < 8192) return true;
  if (bf16 && !rms && !s.affine && s.cpg <= 256 &&
      (s.memory_efficient || s.groups <= 8 || segments < (int64_t{1} << 17)))
    return true;
  // bf16 RMS without affine from y: 64 to 128 channels, at most 4096 channels in all, on 2048 to 4096 rows, and up to
  // 32 channels on more than 131072 and fewer than 262144 segments.
  if (bf16 && rms && !s.affine && s.memory_efficient &&
      ((s.cpg >= 64 && s.cpg <= 128 && s.channels <= 4096 && s.rows >= 2048 && s.rows <= 4096) ||
       (s.cpg <= 32 && segments > (int64_t{1} << 17) && segments < (int64_t{1} << 18))))
    return true;
  // bf16 affine on at most 256 rows in groups up to 256 channels that are not a power of two.
  if (bf16 && s.affine && s.rows <= 256 && s.cpg <= 256 && (s.cpg & (s.cpg - 1)) != 0) return true;
  // bf16 affine layer norm from x on 257 to 2048 rows of at least 4M elements, in whole 16-byte vectors.
  if (bf16 && !rms && s.affine && !s.memory_efficient && s.cpg % 8 == 0 && s.rows > 256 && s.rows <= 2048 &&
      segments * s.cpg >= (int64_t{1} << 22))
    return true;
  // bf16 affine layer norm from y above 4096 channels.
  if (bf16 && !rms && s.affine && s.memory_efficient && s.cpg > 4096) return true;
  // bf16 affine RMS from x: from 256 channels up to 2048 rows (4096 with more than one group), up to 32 channels, and
  // below 256 channels on fewer than 65536 segments.
  if (bf16 && rms && s.affine && !s.memory_efficient &&
      ((s.cpg >= 256 && (s.rows <= 2048 || (s.rows <= 4096 && s.groups > 1))) || s.cpg <= 32 ||
       (s.cpg < 256 && segments < 65536)))
    return true;
  // bf16 affine from y up to 256 channels from 512 rows, outside the TMA kernel's class.
  return bf16 && s.affine && s.memory_efficient && s.cpg <= 256 && s.rows >= 512 && !tma_takes(s);
}

}  // namespace

bool portable(const Shape& s) { return s.mode == kFwd ? prefer_portable_forward(s) : prefer_portable_backward(s); }

void forward(const Shape& s, const Call& c) {
  if (generic_only(s, {c.x_or_y, c.gamma, c.beta})) return fallback_forward(s, c);
  if (prefer_generic_forward(s)) return fallback_forward(s, c);
  if (prefer_dense_lanes_forward(s) && dense_lanes_fit(s, c)) return dense_lanes_forward(s, c);
  // The flat kernel takes groups of up to 1024 vectors of 16 bytes and every width that is not a whole number of
  // vectors; the wide kernels take wider groups (fp32 above 4096 channels, 2-byte above 8192).
  if (forward_flat(s, c) || forward_wide(s, c) || forward_wide_resident(s, c)) return;
  if (dense_lanes_fit(s, c)) return dense_lanes_forward(s, c);
  fallback_forward(s, c);
}

void backward(const Shape& s, const Call& c) {
  const bool layer = s.norm == kLayer;
  const bool bf16 = s.dtype == at::kBFloat16;
  if (generic_only(s, {c.x_or_y, c.dy, c.mean, c.rstd, c.gamma, c.beta})) return fallback_backward(s, c);

  // Kernels for exact classes, asked before the generic preference; a call one of them declines goes on.
  // One 4-CTA cluster a segment: bf16 without affine, groups of 65536 channels.
  if (bf16 && !s.affine && s.cpg == 65536 && backward_wide_rows(s, c)) return;
  // One launch, dgamma/dbeta folded across a cluster: affine up to 256 rows, bf16 groups of a power of two up to 256
  // channels (other widths leave lanes idle), fp32 up to 128 channels a group and from 4096 channels in all.
  if (s.affine && s.rows <= 256 &&
      ((bf16 && s.cpg <= 256 && (s.cpg & (s.cpg - 1)) == 0) ||
       (s.dtype == at::kFloat && s.cpg <= 128 && s.channels >= 4096)) &&
      GroupNormBwdClusterNarrow(*c.dy, *c.x_or_y, *c.mean, *c.rstd, *c.gamma, *c.beta, s.channels,
                                s.groups, layer, s.memory_efficient, *c.dx, *c.dgamma, *c.dbeta))
    return;
  // One 16-CTA cluster a group at 128 rows: bf16 affine RMS from x at 2048 channels a group, and bf16 affine (both
  // norms, from x or y) in eight groups of 512 channels.
  if (bf16 && s.affine && !s.memory_efficient && !layer && s.cpg == 2048 && s.rows == 128 &&
      GroupNormBwdOneClusterGrad(*c.dy, *c.x_or_y, *c.rstd, *c.gamma, s.channels, s.groups, layer,
                                 s.memory_efficient, *c.dx, *c.dgamma))
    return;
  if (bf16 && s.affine && s.cpg == 512 && s.rows == 128 && s.channels == 4096 &&
      GroupNormBwdOneClusterGrad512(*c.dy, *c.x_or_y, *c.mean, *c.rstd, *c.gamma, *c.beta, s.channels,
                                    s.groups, layer, s.memory_efficient, *c.dx, *c.dgamma, *c.dbeta))
    return;

  if (prefer_generic_backward(s)) return fallback_backward(s, c);

  if (tma_takes(s)) {
    const tma::Plan tma = tma::plan(layer, s.groups, s.memory_efficient, *c.x_or_y, *c.dy, *c.mean, *c.rstd,
                                    *c.gamma, *c.beta, *c.dx, *c.dgamma, *c.dbeta);
    // A plan that keeps less than two thirds of its consumer warps busy is slower than the generic kernels.
    if (tma.ok && tma.occupancy >= 2.f / 3 &&
        tma::backward(tma, *c.x_or_y, *c.dy, *c.mean, *c.rstd, *c.gamma, *c.beta, *c.dx, *c.dgamma, *c.dbeta))
      return;
    // That plan, a plan the device would not launch, and a call from y the kernel does not run take the generic
    // kernels.
    if (tma.ok || s.memory_efficient) return fallback_backward(s, c);
  }
  if (!s.affine && dense_lanes_fit(s, c)) return dense_lanes_backward(s, c);
  // The by-width kernel runs a call without affine of its widths and hands an affine call of its widths to the
  // generic kernels, whose dgamma/dbeta reduction runs in a fixed order.
  switch (GroupNormBwdByWidth(layer, *c.x_or_y, *c.dy, *c.mean, *c.rstd, *c.gamma, *c.beta, s.memory_efficient, *c.dx,
                              *c.dgamma, *c.dbeta)) {
    case ByWidthRoute::Launched:
      return;
    case ByWidthRoute::NeedsFixedReduction:
      return fallback_backward(s, c);
    case ByWidthRoute::NotCovered:
      break;
  }
  if (GroupNormBwdOnePass(*c.dy, *c.x_or_y, *c.mean, *c.rstd, *c.gamma, *c.beta, s.channels, s.groups,
                          layer, s.memory_efficient, *c.dx, *c.dgamma, *c.dbeta))
    return;
  if (s.affine && dense_lanes_fit(s, c)) return dense_lanes_backward(s, c);
  fallback_backward(s, c);
}

}  // namespace xllm::ops::blackwell
