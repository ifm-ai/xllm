// Building blocks of the TMA backward pipeline (bwd_tma.cu): mbarriers, 1D bulk copies global -> shared
// (cp.async.bulk), streaming stores, bf16 <-> fp32 packing, segmented in-warp sums for groups of 1 to 32 lanes, and
// the per-device limits the launch is sized by.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace xllm::ops::blackwell::tma {

// Per-device limits, queried once per device (zeros when the query fails: the caller then declines).
struct DevInfo {
  int sms = 0;         // multiprocessors: one persistent CTA each
  int smem_optin = 0;  // largest dynamic shared memory a block may opt in to
};
inline DevInfo dev_info(int dev) {
  static DevInfo cache[64];
  DevInfo d = (dev >= 0 && dev < 64) ? cache[dev] : DevInfo{};
  if (d.sms > 0) return d;
  int sms = 0, optin = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) {
    (void)cudaGetLastError();
    return DevInfo{};
  }
  d.sms = sms;
  d.smem_optin = optin;
  if (dev >= 0 && dev < 64) cache[dev] = d;
  return d;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}

// Spin until the phase of the given parity has completed.
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra DONE;\n"
      "bra LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}

// One bulk copy of `bytes` (a multiple of 16, both addresses 16-byte aligned) from global to this CTA's shared memory;
// its completion is counted on `bar` as transaction bytes.
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}

__device__ __forceinline__ void st_cs_v4(void* p, const uint4& v) {
  asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

template <typename T>
struct Pack;

template <>
struct Pack<__nv_bfloat16> {
  static __device__ __forceinline__ float2 f2(uint32_t v) {
    return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u));
  }
  static __device__ __forceinline__ uint32_t pk(float lo, float hi) {
    __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
  }
  static constexpr uint32_t kOnes = 0x3F803F80u;  // two bf16 1.0
};

template <typename T>
__device__ __forceinline__ void unpack8(const uint4& v, float (&f)[8]) {
  const float2 a = Pack<T>::f2(v.x), b = Pack<T>::f2(v.y), c = Pack<T>::f2(v.z), d = Pack<T>::f2(v.w);
  f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y; f[4] = c.x; f[5] = c.y; f[6] = d.x; f[7] = d.y;
}

template <typename T>
__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
  return make_uint4(Pack<T>::pk(f[0], f[1]), Pack<T>::pk(f[2], f[3]), Pack<T>::pk(f[4], f[5]),
                    Pack<T>::pk(f[6], f[7]));
}

// Sum over the V consecutive lanes of a group, lanes [head, head + V) with gl = lane - head, returned to every lane of
// the group. A power-of-two V is an xor butterfly inside aligned groups; any other V is a doubling shfl_down sum that
// never crosses the group's end, then a broadcast from the head lane.
__device__ __forceinline__ float seg_sum(float v, int V, int gl, int head, bool pow2) {
  if (pow2) {
    for (int o = 1; o < V; o <<= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
  }
  for (int o = 1; o < V; o <<= 1) {
    const float t = __shfl_down_sync(0xffffffffu, v, o);
    if (gl + o < V) v += t;
  }
  return __shfl_sync(0xffffffffu, v, head);
}

__device__ __forceinline__ float2 seg_sum2(float2 v, int V, int gl, int head, bool pow2) {
  if (pow2) {
    for (int o = 1; o < V; o <<= 1) {
      v.x += __shfl_xor_sync(0xffffffffu, v.x, o);
      v.y += __shfl_xor_sync(0xffffffffu, v.y, o);
    }
    return v;
  }
  for (int o = 1; o < V; o <<= 1) {
    const float tx = __shfl_down_sync(0xffffffffu, v.x, o);
    const float ty = __shfl_down_sync(0xffffffffu, v.y, o);
    if (gl + o < V) { v.x += tx; v.y += ty; }
  }
  v.x = __shfl_sync(0xffffffffu, v.x, head);
  v.y = __shfl_sync(0xffffffffu, v.y, head);
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

}  // namespace xllm::ops::blackwell::tma
