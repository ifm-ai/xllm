#pragma once

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <initializer_list>

#include <cstdint>
#include <initializer_list>

#include "ops/muon.h"

// The pieces Muon's list kernels share (muon_kernel.cu, muon_copy_kernel.cu).

namespace xllm {
namespace ops {
namespace muon {

// One launch steps a list of tensors; each CUDA block takes one piece of one tensor, with kThreads threads:
// - kChunk consecutive elements of contiguous tensors, kVec per access (16 bytes of bf16) where every pointer
//   is 16-byte aligned;
// - for a copy between strided views (a slice of some columns, a transposed block), a kTile x kTile tile
//   through shared memory, read along the source's contiguous dimension and written along the destination's
//   (a warp per row or column of it, so 64 consecutive bf16 per access).
constexpr int kThreads = 256;
constexpr int kVec = 8;
constexpr int kChunk = kThreads * kVec * 2;
constexpr int kTile = 64;

__host__ __device__ inline int64_t DivUp(int64_t a, int64_t b) { return (a + b - 1) / b; }

inline bool Aligned(std::initializer_list<const void*> ps) {
  for (const void* p : ps) {
    if (reinterpret_cast<uintptr_t>(p) % 16 != 0) {
      return false;
    }
  }
  return true;
}

// The tensors of a launch and their first blocks. Lists travel as kernel arguments (__grid_constant__),
// within the 32 KB CUDA allows them (12.1+, sm_70+).
template <int N>
struct Blocks {
  static constexpr int kCapacity = N;
  int size;
  int64_t first[N + 1];  // tensor i's blocks are [first[i], first[i + 1])

  // The tensor of block blockIdx.x, and the block's index within it.
  __device__ int Locate(int64_t& local) const {
    const int64_t id = blockIdx.x;
    int lo = 0, hi = size - 1;  // the last tensor whose first block is <= id
    while (lo < hi) {
      const int mid = (lo + hi + 1) / 2;
      if (first[mid] <= id) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    local = id - first[lo];
    return lo;
  }
};

// Fills a List with the tensors i < n for which fill(list, i, j) returns a positive number of blocks, as entry j,
// and runs launch(list) on each full list and on the last.
template <typename List, typename Fill, typename Launch>
void ForEachList(size_t n, Fill fill, Launch launch) {
  List a;
  a.size = 0;
  a.first[0] = 0;
  for (size_t i = 0; i < n; ++i) {
    const int64_t blocks = fill(a, i, a.size);
    if (blocks == 0) {
      continue;
    }
    a.first[a.size + 1] = a.first[a.size] + blocks;
    if (++a.size == List::kCapacity) {
      launch(a);
      a.size = 0;
    }
  }
  if (a.size > 0) {
    launch(a);
  }
}

// f(k, n, vec) for the runs of n <= kVec elements from k in chunk `local` of a numel-element tensor; vec: a
// whole run at aligned pointers, one 16-byte access per tensor.
template <typename F>
__device__ void ForEachRun(int64_t numel, bool aligned, int64_t local, F f) {
  const int64_t end = (local + 1) * kChunk < numel ? (local + 1) * kChunk : numel;
  for (int64_t k = local * kChunk + threadIdx.x * kVec; k < end; k += kThreads * kVec) {
    const int n = end - k < kVec ? static_cast<int>(end - k) : kVec;
    f(k, n, aligned && n == kVec);
  }
}

// n elements at p as floats, and back.
template <typename T>
__device__ void Load(const T* p, int n, bool vec, float (&v)[kVec]) {
  if (vec && sizeof(T) == 4) {
    const float4 x = reinterpret_cast<const float4*>(p)[0], y = reinterpret_cast<const float4*>(p)[1];
    v[0] = x.x, v[1] = x.y, v[2] = x.z, v[3] = x.w, v[4] = y.x, v[5] = y.y, v[6] = y.z, v[7] = y.w;
  } else if (vec) {
    const uint4 raw = *reinterpret_cast<const uint4*>(p);
    const T* h = reinterpret_cast<const T*>(&raw);
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      v[j] = static_cast<float>(h[j]);
    }
  } else {
    for (int j = 0; j < n; ++j) {
      v[j] = static_cast<float>(p[j]);
    }
  }
}

template <typename T>
__device__ void Store(T* p, int n, bool vec, const float (&v)[kVec]) {
  if (vec && sizeof(T) == 4) {
    reinterpret_cast<float4*>(p)[0] = make_float4(v[0], v[1], v[2], v[3]);
    reinterpret_cast<float4*>(p)[1] = make_float4(v[4], v[5], v[6], v[7]);
  } else if (vec) {
    uint4 raw;
    T* h = reinterpret_cast<T*>(&raw);
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      h[j] = T(v[j]);
    }
    *reinterpret_cast<uint4*>(p) = raw;
  } else {
    for (int j = 0; j < n; ++j) {
      p[j] = T(v[j]);
    }
  }
}

}  // namespace muon
}  // namespace ops
}  // namespace xllm
