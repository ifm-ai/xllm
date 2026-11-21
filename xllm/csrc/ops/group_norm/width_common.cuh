// Conversion, 16-byte vector and in-warp reduction helpers for 2-byte element types, shared by the by-width backward
// (bwd_by_width.cu) and the dense-lane width-40 forward (dense_lanes.cu). Each kernel keeps its own block-wide
// reduction beside it.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace xllm::ops::blackwell::width {

__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ void from_f(__half& d, float s) { d = __float2half(s); }
__device__ __forceinline__ void from_f(__nv_bfloat16& d, float s) { d = __float2bfloat16(s); }

// V elements of T that form one 16-byte vector.
template <typename T, int V>
struct alignas(sizeof(T) * V) Vec {
  T v[V];
};

// Streaming (evict-first) load and store, for data read or written once.
template <typename T, int V>
__device__ __forceinline__ Vec<T, V> load_stream(const T* p) {
  static_assert(sizeof(T) * V == 16, "one 16-byte vector");
  union {
    uint4 r;
    Vec<T, V> v;
  } u;
  u.r = __ldcs(reinterpret_cast<const uint4*>(p));
  return u.v;
}

template <typename T, int V>
__device__ __forceinline__ void store_stream(T* p, const Vec<T, V>& val) {
  static_assert(sizeof(T) * V == 16, "one 16-byte vector");
  union {
    uint4 r;
    Vec<T, V> v;
  } u;
  u.v = val;
  __stcs(reinterpret_cast<uint4*>(p), u.r);
}

// Read-only cached load, for gamma and beta, which every row reads.
template <typename T, int V>
__device__ __forceinline__ Vec<T, V> load_cached(const T* p) {
  static_assert(sizeof(T) * V == 16, "one 16-byte vector");
  union {
    uint4 r;
    Vec<T, V> v;
  } u;
  u.r = __ldg(reinterpret_cast<const uint4*>(p));
  return u.v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// The sum over the L lanes of this lane's group, lanes [L * (lane / L), L * (lane / L) + L), returned to every lane of
// the group: a butterfly for a power-of-two L, otherwise each lane adds the group's L values in lane order.
template <int L>
__device__ __forceinline__ float group_sum(float s, int lane) {
  if constexpr ((L & (L - 1)) == 0) {
#pragma unroll
    for (int off = L / 2; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off, L);
  } else {
    const int base = (lane / L) * L;
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < L; ++i) t += __shfl_sync(0xffffffffu, s, base + i);
    s = t;
  }
  return s;
}

}  // namespace xllm::ops::blackwell::width
