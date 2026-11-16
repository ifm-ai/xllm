// Wide fp32 forward of the group RMS norm and group layer norm (forward_wide, called by dispatch.cu's forward() in
// dispatch.cu): groups of 4097..8192 channels on CTAs that each own a slice of rows of one group (wide_fwd_kernel), and
// groups of 8193..65536 channels streamed by one CTA a segment when the call has at most one segment an SM
// (wide_stream_kernel). Channels per group are a multiple of 8: a thread moves 8 floats at a time.
#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <cstdint>

#include "dispatch.h"
#include "pdl.h"

namespace xllm::ops::blackwell {
namespace {

__device__ __forceinline__ void ld8(const float* __restrict__ p, float* __restrict__ f) {
  const uint4 a = *reinterpret_cast<const uint4*>(p);
  const uint4 b = *reinterpret_cast<const uint4*>(p + 4);
  f[0] = __uint_as_float(a.x); f[1] = __uint_as_float(a.y);
  f[2] = __uint_as_float(a.z); f[3] = __uint_as_float(a.w);
  f[4] = __uint_as_float(b.x); f[5] = __uint_as_float(b.y);
  f[6] = __uint_as_float(b.z); f[7] = __uint_as_float(b.w);
}
__device__ __forceinline__ void st8(float* __restrict__ p, const float* __restrict__ f) {
  __stcs(reinterpret_cast<uint4*>(p), make_uint4(__float_as_uint(f[0]), __float_as_uint(f[1]),
                                              __float_as_uint(f[2]), __float_as_uint(f[3])));
  __stcs(reinterpret_cast<uint4*>(p + 4), make_uint4(__float_as_uint(f[4]), __float_as_uint(f[5]),
                                                  __float_as_uint(f[6]), __float_as_uint(f[7])));
}
// The same 8 floats in one 32-byte access (ld/st.global.v8.f32, sm_100 and up) when `wide`, which the host sets only
// when every pointer of the call is 32-byte aligned; two 16-byte accesses otherwise.
__device__ __forceinline__ void ld8w(const float* __restrict__ p, float* __restrict__ f, bool wide) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (wide) {
    asm volatile("ld.global.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=f"(f[0]), "=f"(f[1]), "=f"(f[2]), "=f"(f[3]),
                   "=f"(f[4]), "=f"(f[5]), "=f"(f[6]), "=f"(f[7])
                 : "l"(p)
                 : "memory");
    return;
  }
#else
  (void)wide;
#endif
  ld8(p, f);
}
__device__ __forceinline__ void st8w(float* __restrict__ p, const float* __restrict__ f, bool wide) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (wide) {
    asm volatile("st.global.cs.v8.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
                 :
                 : "l"(p), "f"(f[0]), "f"(f[1]), "f"(f[2]), "f"(f[3]),
                   "f"(f[4]), "f"(f[5]), "f"(f[6]), "f"(f[7])
                 : "memory");
    return;
  }
#else
  (void)wide;
#endif
  st8(p, f);
}
// ld8w with the evict-first policy, for the last read of a segment's bytes.
__device__ __forceinline__ void ld8cs(const float* __restrict__ p, float* __restrict__ f, bool wide) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (wide) {
    asm volatile("ld.global.cs.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=f"(f[0]), "=f"(f[1]), "=f"(f[2]), "=f"(f[3]),
                   "=f"(f[4]), "=f"(f[5]), "=f"(f[6]), "=f"(f[7])
                 : "l"(p)
                 : "memory");
    return;
  }
#else
  (void)wide;
#endif
  const uint4 a = __ldcs(reinterpret_cast<const uint4*>(p));
  const uint4 b = __ldcs(reinterpret_cast<const uint4*>(p + 4));
  f[0] = __uint_as_float(a.x); f[1] = __uint_as_float(a.y);
  f[2] = __uint_as_float(a.z); f[3] = __uint_as_float(a.w);
  f[4] = __uint_as_float(b.x); f[5] = __uint_as_float(b.y);
  f[6] = __uint_as_float(b.z); f[7] = __uint_as_float(b.w);
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// The sum across the CTA, returned to every thread, with one barrier: each warp writes its sum into bank `bank` (nwarp
// floats of sh), and after the barrier every warp reduces the nwarp partials with the same butterfly. A caller
// alternates between two banks: a warp writes a bank again only two reductions later, after the barrier of the
// reduction between them, which every warp reaches only after its read of that bank.
__device__ __forceinline__ float block_sum1(float v, float* sh, int nwarp, int bank) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float* part = sh + bank * nwarp;
  v = warp_sum(v);
  if (lane == 0) part[warp] = v;
  __syncthreads();
  return warp_sum((lane < nwarp) ? part[lane] : 0.f);
}

// Two sums through one barrier, on the pair of banks 2 * pair and 2 * pair + 1 (sh holds 4 * nwarp floats); each value
// is reduced as block_sum1 reduces it. Callers alternate the pair, as block_sum1's callers alternate the bank.
__device__ __forceinline__ void block_sum2(float& a, float& b, float* sh, int nwarp, int pair) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float* pa = sh + (2 * pair) * nwarp;
  float* pb = sh + (2 * pair + 1) * nwarp;
  a = warp_sum(a);
  b = warp_sum(b);
  if (lane == 0) { pa[warp] = a; pb[warp] = b; }
  __syncthreads();
  const bool live = lane < nwarp;
  a = warp_sum(live ? pa[lane] : 0.f);
  b = warp_sum(live ? pb[lane] : 0.f);
}

// One CTA owns a slice of the rows of one group: grid (slices, groups), and CTA (r, g) normalizes rows r, r + slices,
// ... of group g. A thread holds VECS vectors of 8 floats, channels tid * 8 + k * blockDim.x * 8, and keeps their gamma
// and beta in registers across rows; a vector past the group is masked (data 0, factors 1 and 0). TPB is the launch
// bound, which the host matches with its thread count. A row's sums run over a thread's vectors, then over the warp,
// then over the warps.
// The layer norm without affine, or with affine at TPB 512, takes both raw moments in one pass and one paired
// reduction, and keeps var = E[x^2] - mean^2 where var >= E[x^2] / 16 (at most four bits lost to cancellation) and
// E[x^2] is finite; any other row takes the centred squares of the values it holds. At TPB 1024 the affine layer norm
// takes the mean, then the centred squares: under that register cap the fallback would spill.
template <bool AFFINE, bool LAYER, int VECS, int TPB>
__global__ __launch_bounds__(TPB) void wide_fwd_kernel(const float* __restrict__ x, const float* __restrict__ gamma,
                                                          const float* __restrict__ beta, float* __restrict__ y,
                                                          float* __restrict__ mean_out, float* __restrict__ rstd_out,
                                                          long long rows, int num_groups, int cpg, float eps,
                                                          bool wide) {
  cudaGridDependencySynchronize();
  extern __shared__ float sh[];
  const int tid = threadIdx.x;
  const int nwarp = blockDim.x >> 5;
  const int g = blockIdx.y;
  const long long C = (long long)num_groups * cpg;
  constexpr int kVecs = VECS;
  const int vch0 = tid * 8;                  // this thread's first vector, inside the group
  const int step_ch = (int)blockDim.x * 8;   // channels this CTA covers in one pass
  const long long base_ch = (long long)g * cpg + vch0;
  bool on[kVecs];
  int off[kVecs];
#pragma unroll
  for (int vb = 0; vb < kVecs; ++vb) {
    off[vb] = vb * step_ch;
    on[vb] = (vch0 + off[vb]) < cpg;
  }
  const float inv_n = 1.0f / (float)cpg;

  float ga[kVecs][8], be[kVecs][8];
#pragma unroll
  for (int vb = 0; vb < kVecs; ++vb) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      ga[vb][i] = 1.f;
      be[vb][i] = 0.f;
    }
  }
  if (AFFINE) {
#pragma unroll
    for (int vb = 0; vb < kVecs; ++vb) {
      if (on[vb]) {
        ld8w(gamma + base_ch + off[vb], ga[vb], wide);
        if (LAYER) ld8w(beta + base_ch + off[vb], be[vb], wide);
      }
    }
  }

  int it = 0;  // rows this CTA has done: picks the bank (pair) of the next reduction
  for (long long r = blockIdx.x; r < rows; r += gridDim.x) {
    float f[kVecs][8];
#pragma unroll
    for (int vb = 0; vb < kVecs; ++vb) {
      if (on[vb]) ld8w(x + r * C + base_ch + off[vb], f[vb], wide);
      else {
#pragma unroll
        for (int i = 0; i < 8; ++i) f[vb][i] = 0.f;
      }
    }
    float mean = 0.f, ss = 0.f;
    if (LAYER) {
      if constexpr (!AFFINE || TPB <= 512) {
        float sm = 0.f, sq = 0.f;
#pragma unroll
        for (int vb = 0; vb < kVecs; ++vb) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const float v = f[vb][i];
            sm += v;
            sq = fmaf(v, v, sq);
          }
        }
        const int pair = it++ & 1;
        block_sum2(sm, sq, sh, nwarp, pair);
        mean = sm * inv_n;
        const float m2 = sq * inv_n;
        ss = fmaf(-mean, mean, m2);
        constexpr float kRawGuard = 0.0625f;  // keep the one-pass variance while var >= m2 / 16
        // Both operands are CTA-wide sums, so the branch is uniform over the CTA and its barriers are legal.
        if (!(ss >= kRawGuard * m2) || !isfinite(m2)) {
          // The centred squares, reduced on the pair this row did not use. A masked vector centres nothing (mk = 0),
          // or its 8 * mean^2 would land in the variance.
          sq = 0.f;
#pragma unroll
          for (int vb = 0; vb < kVecs; ++vb) {
            const float mk = on[vb] ? mean : 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              const float d = f[vb][i] - mk;
              sq = fmaf(d, d, sq);
            }
          }
          ss = block_sum1(sq, sh, nwarp, 2 * (pair ^ 1)) * inv_n;
          // The next row writes the pair just read here before its own barrier: every read must finish first.
          __syncthreads();
        }
        ss += eps;
      } else {
        float sm = 0.f;
#pragma unroll
        for (int vb = 0; vb < kVecs; ++vb) {
#pragma unroll
          for (int i = 0; i < 8; ++i) sm += f[vb][i];
        }
        mean = block_sum1(sm, sh, nwarp, 0) * inv_n;
        // The centred squares; a masked vector centres nothing (mk = 0).
        float vs = 0.f;
#pragma unroll
        for (int vb = 0; vb < kVecs; ++vb) {
          const float mk = on[vb] ? mean : 0.f;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const float d = f[vb][i] - mk;
            vs += d * d;
          }
        }
        ss = block_sum1(vs, sh, nwarp, 1) * inv_n + eps;
      }
    } else {
#pragma unroll
      for (int vb = 0; vb < kVecs; ++vb) {
#pragma unroll
        for (int i = 0; i < 8; ++i) ss += f[vb][i] * f[vb][i];
      }
      ss = block_sum1(ss, sh, nwarp, it++ & 1) * inv_n + eps;
    }
    const float rstd = rsqrtf(ss);
    if (tid == 0) {
      rstd_out[r * num_groups + g] = rstd;
      if (LAYER) mean_out[r * num_groups + g] = mean;
    }
    float o[8];
#pragma unroll
    for (int vb = 0; vb < kVecs; ++vb) {
      if (AFFINE) {
        if (LAYER) {
#pragma unroll
          for (int i = 0; i < 8; ++i) o[i] = (f[vb][i] - mean) * rstd * ga[vb][i] + be[vb][i];
        } else {
#pragma unroll
          for (int i = 0; i < 8; ++i) o[i] = f[vb][i] * rstd * ga[vb][i];
        }
      } else {
        if (LAYER) {
#pragma unroll
          for (int i = 0; i < 8; ++i) o[i] = (f[vb][i] - mean) * rstd;
        } else {
#pragma unroll
          for (int i = 0; i < 8; ++i) o[i] = f[vb][i] * rstd;
        }
      }
      if (on[vb]) st8w(y + r * C + base_ch + off[vb], o, wide);
    }
  }
}

// One 1024-thread CTA a (row, group) segment, walking it in sweeps of 8192 channels (8 floats a thread). The first
// sweep takes the statistics: the sum of squares (RMS), or both raw moments (layer norm), which are kept as above
// (var >= E[x^2] / 16 and E[x^2] finite) or else replaced by a corrected two-pass variance from one more sweep. The
// last sweep normalizes and stores, reading x evict-first. The host launches one CTA a segment only when there are at
// most as many segments as SMs, so the sweeps after the first hit L2. The layer norm reduces on banks 0 and 1, RMS
// alternates its bank per segment. __launch_bounds__(1024, 1) lets each thread take 64 registers: no spill.
template <bool AFFINE, bool LAYER>
__global__ __launch_bounds__(1024, 1) void wide_stream_kernel(const float* __restrict__ x,
                                                           const float* __restrict__ gamma,
                                                           const float* __restrict__ beta, float* __restrict__ y,
                                                           float* __restrict__ mean_out, float* __restrict__ rstd_out,
                                                           long long segments, int num_groups, int cpg, float eps,
                                                           bool wide) {
  cudaGridDependencySynchronize();
  extern __shared__ float sh[];
  const int tid = threadIdx.x;
  const int nwarp = blockDim.x >> 5;
  const int step = (int)blockDim.x * 8;
  const float inv_n = 1.0f / (float)cpg;
  int it = 0;  // segments this CTA has done: picks the RMS bank
  for (long long seg = blockIdx.x; seg < segments; seg += gridDim.x) {
    const float* xs = x + seg * (long long)cpg;
    float* ys = y + seg * (long long)cpg;
    const long long gbase = (long long)(seg % (long long)num_groups) * (long long)cpg;
    float sm = 0.f, sq = 0.f;
    for (int ch = tid * 8; ch < cpg; ch += step) {
      float f[8];
      ld8w(xs + ch, f, wide);
      if (LAYER) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          sm += f[i];
          sq = fmaf(f[i], f[i], sq);
        }
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) sm += f[i] * f[i];
      }
    }
    float mean = 0.f, ss = 0.f;
    if (LAYER) {
      mean = block_sum1(sm, sh, nwarp, 0) * inv_n;
      const float m2 = block_sum1(sq, sh, nwarp, 1) * inv_n;
      float dvar = fmaf(-mean, mean, m2);
      // An overflowed sum of squares reads inf >= inf, hence the finiteness test. Both operands are CTA-wide sums, so
      // the branch is uniform over the CTA and its barriers are legal.
      if (!(dvar >= 0.0625f * m2) || !isfinite(m2)) {
        // Corrected two-pass: the deviations' own mean is what rounding `mean` to float left over.
        float e = 0.f, c = 0.f;
        for (int ch = tid * 8; ch < cpg; ch += step) {
          float g[8];
          ld8w(xs + ch, g, wide);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const float d = g[i] - mean;
            e += d;
            c += d * d;
          }
        }
        // Bank 0 was last read before the second moment's barrier, whose barrier orders every read of bank 1.
        const float de = block_sum1(e, sh, nwarp, 0) * inv_n;
        dvar = fmaf(-de, de, block_sum1(c, sh, nwarp, 1) * inv_n);
      }
      ss = fmaxf(dvar, 0.f) + eps;
    } else {
      ss = block_sum1(sm, sh, nwarp, it & 1) * inv_n + eps;
    }
    const float rstd = rsqrtf(ss);
    if (tid == 0) {
      rstd_out[seg] = rstd;
      if (LAYER) mean_out[seg] = mean;
    }
    for (int ch = tid * 8; ch < cpg; ch += step) {
      float f[8], ga[8], be[8];
      ld8cs(xs + ch, f, wide);
      if (AFFINE) {
        ld8w(gamma + gbase + ch, ga, wide);
        if (LAYER) ld8w(beta + gbase + ch, be, wide);
      }
      float o[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (AFFINE) {
          if (LAYER) {
            o[i] = (f[i] - mean) * rstd * ga[i] + be[i];
          } else {
            o[i] = f[i] * rstd * ga[i];
          }
        } else {
          o[i] = LAYER ? (f[i] - mean) * rstd : f[i] * rstd;
        }
      }
      st8w(ys + ch, o, wide);
    }
    ++it;
  }
}

}  // namespace

// Operands are contiguous (the Call contract), inputs 16-byte aligned (dispatch.cu sends any other call to the generic
// kernels) and outputs freshly allocated.
bool forward_wide(const Shape& s, const Call& c) {
  if (s.dtype != at::kFloat || s.cpg <= 4096 || s.cpg % 8 != 0 || s.groups > 65535 || s.rows < 1) return false;
  // x's device is the current device: the entry points hold a device guard.
  const c10::DeviceIndex device = c.x_or_y->device().index();
  const int64_t sms = at::cuda::getDeviceProperties(device)->multiProcessorCount;
  const int64_t segments = s.rows * s.groups;
  // A streamed segment is read again after its statistics: that pays only while every segment is in flight at once.
  if (s.cpg > 8192 && segments > sms) return false;
  // 32-byte accesses when every pointer is 32-byte aligned: each access is a multiple of 8 floats from its base.
  auto a32 = [](const at::Tensor& t) { return (reinterpret_cast<uintptr_t>(t.const_data_ptr()) & 31u) == 0; };
  const bool wide = a32(*c.x_or_y) && a32(*c.y) && (!s.affine || a32(*c.gamma)) && (!s.has_beta || a32(*c.beta));
  const auto stream = at::cuda::getCurrentCUDAStream(device);
  const float* x = c.x_or_y->const_data_ptr<float>();
  const float* gamma = s.affine ? c.gamma->const_data_ptr<float>() : nullptr;
  const float* beta = s.has_beta ? c.beta->const_data_ptr<float>() : nullptr;
  float* y = c.y->data_ptr<float>();
  float* mean = s.norm == kLayer ? c.mean_out->data_ptr<float>() : nullptr;
  float* rstd = c.rstd_out->data_ptr<float>();

  if (s.cpg > 8192) {
    // Two banks of one float a warp, and 64 bytes of slack.
    const size_t smem = sizeof(float) * 2 * (1024 / 32) + 64;
    auto issue = [&](auto kernel) {
      LaunchKernelPdl(kernel, dim3((unsigned)segments), dim3(1024), smem, stream, x, gamma, beta, y,
                      mean, rstd, (long long)segments, s.groups, s.cpg, c.eps, wide);
    };
    if (s.affine) {
      if (s.norm == kLayer) issue(wide_stream_kernel<true, true>);
      else issue(wide_stream_kernel<true, false>);
    } else {
      if (s.norm == kLayer) issue(wide_stream_kernel<false, true>);
      else issue(wide_stream_kernel<false, false>);
    }
    return true;
  }

  // Affine: 512 threads of two vectors; RMS with affine on a grid of at least 4 such CTAs an SM takes 256 threads of
  // four (fewer threads, deeper rows). Without affine: 256 threads of four vectors. Up to 2048 threads an SM of CTAs,
  // at most one CTA a row of a group.
  const bool rms_affine_deep = s.affine && s.norm != kLayer && segments >= 4 * sms;
  const int threads = s.affine && !rms_affine_deep ? 512 : 256;
  const int64_t target = sms * (2048 / threads);
  const int64_t slices = std::max<int64_t>(1, std::min(s.rows, (target + s.groups - 1) / s.groups));
  // The affine layer norm takes the one-pass variance at TPB 512 when its grid is at most one CTA an SM, where the
  // register cap's second CTA an SM would sit idle anyway.
  const bool affine_one_pass = s.affine && s.norm == kLayer && slices * s.groups <= sms;
  // Bank space: two banks of one float a warp, four on the layer norm's paired path, and 64 bytes of slack.
  const bool paired = s.norm == kLayer && (!s.affine || affine_one_pass);
  const size_t smem = sizeof(float) * (paired ? 4 : 2) * (threads / 32) + 64;
  const dim3 grid((unsigned)slices, (unsigned)s.groups);
  auto issue = [&](auto kernel) {
    LaunchKernelPdl(kernel, grid, dim3(threads), smem, stream, x, gamma, beta, y, mean, rstd, s.rows,
                    s.groups, s.cpg, c.eps, wide);
  };
  if (s.affine) {
    if (s.norm == kLayer) {
      if (affine_one_pass) issue(wide_fwd_kernel<true, true, 2, 512>);
      else issue(wide_fwd_kernel<true, true, 2, 1024>);
    } else {
      if (rms_affine_deep) issue(wide_fwd_kernel<true, false, 4, 256>);
      else issue(wide_fwd_kernel<true, false, 2, 1024>);
    }
  } else {
    if (s.norm == kLayer) issue(wide_fwd_kernel<false, true, 4, 1024>);
    else issue(wide_fwd_kernel<false, false, 4, 1024>);
  }
  return true;
}

}  // namespace xllm::ops::blackwell
