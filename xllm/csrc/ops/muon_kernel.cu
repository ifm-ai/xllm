#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <initializer_list>

#include "ops/muon.h"

namespace xllm {
namespace ops {

namespace {

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

__host__ __device__ int64_t DivUp(int64_t a, int64_t b) { return (a + b - 1) / b; }

bool Aligned(std::initializer_list<const void*> ps) {
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

// Contiguous fp32 tensors (momentum / param), each paired with a contiguous tensor of its numel (grad / update).
struct TensorList : Blocks<512> {
  int64_t numel[kCapacity];
  bool vec[kCapacity];
  float* fp32[kCapacity];
  const void* other[kCapacity];
  c10::BFloat16* out[kCapacity];  // momentum only
  float scale[kCapacity];         // update only
};

struct CopyList : Blocks<192> {
  StridedView dst_view[kCapacity], src_view[kCapacity];
  bool linear[kCapacity];  // both contiguous
  bool vec[kCapacity];
  void* dst[kCapacity];
  const void* src[kCapacity];
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

template <typename G, bool kNesterov>
__global__ void __launch_bounds__(kThreads) MuonMomentumKernel(const __grid_constant__ TensorList a, float mu) {
  int64_t local;
  const int t = a.Locate(local);
  float* m = a.fp32[t];
  const G* g = static_cast<const G*>(a.other[t]);
  c10::BFloat16* out = a.out[t];
  ForEachRun(a.numel[t], a.vec[t], local, [&](int64_t k, int n, bool vec) {
    float mv[kVec] = {}, gv[kVec] = {}, uv[kVec];
    Load(m + k, n, vec, mv);
    Load(g + k, n, vec, gv);
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      mv[j] = __fmaf_rn(mu, mv[j], gv[j]);
      uv[j] = kNesterov ? __fmaf_rn(mu, mv[j], gv[j]) : mv[j];
    }
    Store(m + k, n, vec, mv);
    Store(out + k, n, vec, uv);
  });
}

__global__ void __launch_bounds__(kThreads) MuonUpdateKernel(const __grid_constant__ TensorList a, float lr, float wd) {
  int64_t local;
  const int t = a.Locate(local);
  float* p = a.fp32[t];
  const c10::BFloat16* u = static_cast<const c10::BFloat16*>(a.other[t]);
  const float s = a.scale[t];
  ForEachRun(a.numel[t], a.vec[t], local, [&](int64_t k, int n, bool vec) {
    float pv[kVec] = {}, uv[kVec] = {};
    Load(p + k, n, vec, pv);
    Load(u + k, n, vec, uv);
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      pv[j] = __fmaf_rn(-lr, __fmaf_rn(wd, pv[j], __fmul_rn(s, uv[j])), pv[j]);  // p - lr (wd p + s u)
    }
    Store(p + k, n, vec, pv);
  });
}

// f(ri, ci) for the elements of a tile, each warp's lanes running along its rows (rows_fast) or its columns.
template <typename F>
__device__ void ForEachInTile(bool rows_fast, F f) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
  for (int i = warp; i < kTile; i += kThreads / 32) {
#pragma unroll
    for (int j = lane; j < kTile; j += 32) {
      rows_fast ? f(j, i) : f(i, j);
    }
  }
}

template <typename S, typename D>
__global__ void __launch_bounds__(kThreads) BatchedCopyKernel(const __grid_constant__ CopyList a) {
  __shared__ float tile[kTile][kTile + 1];
  int64_t local;
  const int t = a.Locate(local);
  const S* src = static_cast<const S*>(a.src[t]);
  D* dst = static_cast<D*>(a.dst[t]);
  const StridedView &dv = a.dst_view[t], &sv = a.src_view[t];
  if (a.linear[t]) {
    ForEachRun(dv.B * dv.R * dv.C, a.vec[t], local, [&](int64_t k, int n, bool vec) {
      float v[kVec];
      Load(src + k, n, vec, v);
      Store(dst + k, n, vec, v);
    });
    return;
  }
  const int64_t tiles_c = DivUp(dv.C, kTile), per_batch = DivUp(dv.R, kTile) * tiles_c;
  const int64_t b = local / per_batch, r0 = local % per_batch / tiles_c * kTile, c0 = local % tiles_c * kTile;
  ForEachInTile(sv.sC != 1, [&](int ri, int ci) {
    const int64_t r = r0 + ri, c = c0 + ci;
    if (r < dv.R && c < dv.C) {
      tile[ri][ci] = static_cast<float>(src[b * sv.sB + r * sv.sR + c * sv.sC]);
    }
  });
  __syncthreads();
  ForEachInTile(dv.sC != 1, [&](int ri, int ci) {
    const int64_t r = r0 + ri, c = c0 + ci;
    if (r < dv.R && c < dv.C) {
      dst[b * dv.sB + r * dv.sR + c * dv.sC] = D(tile[ri][ci]);
    }
  });
}

}  // namespace

void MuonMomentumCUDA(const std::vector<torch::Tensor>& momentum, const std::vector<torch::Tensor>& grad,
                      const std::vector<torch::Tensor>& out, double mu, bool nesterov) {
  const c10::cuda::CUDAGuard guard(momentum[0].device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, grad[0].scalar_type(), "MuonMomentumCUDA", [&] {
    ForEachList<TensorList>(
        momentum.size(),
        [&](TensorList& a, size_t i, int j) {
          a.numel[j] = momentum[i].numel();
          a.fp32[j] = momentum[i].data_ptr<float>();
          a.other[j] = grad[i].data_ptr<scalar_t>();
          a.out[j] = out[i].data_ptr<c10::BFloat16>();
          a.vec[j] = Aligned({a.fp32[j], a.other[j], a.out[j]});
          return DivUp(a.numel[j], kChunk);
        },
        [&](const TensorList& a) {
          const auto kernel = nesterov ? MuonMomentumKernel<scalar_t, true> : MuonMomentumKernel<scalar_t, false>;
          kernel<<<a.first[a.size], kThreads, 0, stream>>>(a, static_cast<float>(mu));
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  });
}

void MuonUpdateCUDA(const std::vector<torch::Tensor>& param, const std::vector<torch::Tensor>& update,
                    const std::vector<double>& scale, double lr, double weight_decay) {
  const c10::cuda::CUDAGuard guard(param[0].device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ForEachList<TensorList>(
      param.size(),
      [&](TensorList& a, size_t i, int j) {
        a.numel[j] = param[i].numel();
        a.fp32[j] = param[i].data_ptr<float>();
        a.other[j] = update[i].data_ptr<c10::BFloat16>();
        a.scale[j] = static_cast<float>(scale[i]);
        a.vec[j] = Aligned({a.fp32[j], a.other[j]});
        return DivUp(a.numel[j], kChunk);
      },
      [&](const TensorList& a) {
        MuonUpdateKernel<<<a.first[a.size], kThreads, 0, stream>>>(a, static_cast<float>(lr),
                                                                    static_cast<float>(weight_decay));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

void BatchedCopyCUDA(const std::vector<torch::Tensor>& dst, const std::vector<torch::Tensor>& src) {
  const c10::cuda::CUDAGuard guard(dst[0].device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_REDUCED_FLOATING_TYPES(src[0].scalar_type(), "BatchedCopyCUDA", [&] {
    using S = scalar_t;
    AT_DISPATCH_REDUCED_FLOATING_TYPES(dst[0].scalar_type(), "BatchedCopyCUDA", [&] {
      ForEachList<CopyList>(
          dst.size(),
          [&](CopyList& a, size_t i, int j) -> int64_t {
            const StridedView v = AsStridedView(dst[i]);
            a.dst_view[j] = v;
            a.src_view[j] = AsStridedView(src[i]);
            a.dst[j] = dst[i].data_ptr();
            a.src[j] = src[i].data_ptr();
            a.linear[j] = dst[i].is_contiguous() && src[i].is_contiguous();
            a.vec[j] = Aligned({a.dst[j], a.src[j]});
            return a.linear[j] ? DivUp(dst[i].numel(), kChunk) : v.B * DivUp(v.R, kTile) * DivUp(v.C, kTile);
          },
          [&](const CopyList& a) {
            BatchedCopyKernel<S, scalar_t><<<a.first[a.size], kThreads, 0, stream>>>(a);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
    });
  });
}

}  // namespace ops
}  // namespace xllm
