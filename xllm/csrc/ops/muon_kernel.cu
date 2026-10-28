#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include "ops/muon.h"

namespace xllm {
namespace ops {

namespace {

// Each block steps one piece of one tensor of a list, with kThreads threads, in one of two ways:
// - linear, when all of the entry's tensors are contiguous: kChunk consecutive elements, kVec per access
//   (16 bytes of bf16) where the pointers are 16-byte aligned;
// - tiled, otherwise (a transposed block, a slice of some columns): a kTile x kTile tile through shared
//   memory, read along the source's contiguous dimension and written along the destination's (a warp per
//   row or column of it, so 64 consecutive bf16 per access).
constexpr int kThreads = 256;
constexpr int kVec = 8;
constexpr int kChunk = kThreads * kVec * 2;
constexpr int kTile = 64;

struct Entry {
  StridedView view;  // the bf16 / destination side, which the tiles follow
  int64_t numel;
  bool linear;
  bool aligned;
};

// The lists travel as kernel arguments, within CUDA's 32 KB parameter limit (sm_70+).
constexpr int kMaxTensors = 224;
struct TensorList {
  int64_t size;
  int64_t first_block[kMaxTensors + 1];  // block offsets, and the total
  Entry entry[kMaxTensors];
  float* contiguous[kMaxTensors];         // momentum / param
  const void* grad[kMaxTensors];          // momentum only
  c10::BFloat16* strided[kMaxTensors];    // out / update
  float scale[kMaxTensors];               // update only
};

constexpr int kMaxCopies = 160;
struct CopyList {
  int64_t size;
  int64_t first_block[kMaxCopies + 1];
  Entry entry[kMaxCopies];  // of dst
  void* dst[kMaxCopies];
  const void* src[kMaxCopies];
  StridedView src_view[kMaxCopies];
};

__host__ __device__ int64_t DivUp(int64_t a, int64_t b) { return (a + b - 1) / b; }

int64_t NumBlocks(const Entry& e) {
  const StridedView& v = e.view;
  return e.linear ? DivUp(e.numel, kChunk) : v.B * DivUp(v.R, kTile) * DivUp(v.C, kTile);
}

// The block blockIdx.x of a list: its tensor and, within it, the block's index.
template <typename List>
__device__ void LocateBlock(const List& a, int& t, int64_t& local) {
  const int64_t id = blockIdx.x;
  int lo = 0, hi = static_cast<int>(a.size) - 1;  // the last tensor whose first block is <= id
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (a.first_block[mid] <= id) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  t = lo;
  local = id - a.first_block[lo];
}

// A tiled block's batch, first row and first column.
struct Tile {
  int64_t b, r0, c0;
};

__device__ Tile LocateTile(const StridedView& v, int64_t local) {
  const int64_t tiles_c = DivUp(v.C, kTile), per_batch = DivUp(v.R, kTile) * tiles_c, in_batch = local % per_batch;
  return {local / per_batch, in_batch / tiles_c * kTile, in_batch % tiles_c * kTile};
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

// kVec elements at p (16-byte aligned) as floats, and back.
template <typename T>
__device__ void LoadVec(const T* p, float (&v)[kVec]) {
  if constexpr (sizeof(T) == 4) {
    const float4 x = reinterpret_cast<const float4*>(p)[0], y = reinterpret_cast<const float4*>(p)[1];
    v[0] = x.x, v[1] = x.y, v[2] = x.z, v[3] = x.w, v[4] = y.x, v[5] = y.y, v[6] = y.z, v[7] = y.w;
  } else {
    const uint4 raw = *reinterpret_cast<const uint4*>(p);
    const T* h = reinterpret_cast<const T*>(&raw);
    for (int j = 0; j < kVec; ++j) {
      v[j] = static_cast<float>(h[j]);
    }
  }
}

template <typename T>
__device__ void StoreVec(T* p, const float (&v)[kVec]) {
  if constexpr (sizeof(T) == 4) {
    reinterpret_cast<float4*>(p)[0] = make_float4(v[0], v[1], v[2], v[3]);
    reinterpret_cast<float4*>(p)[1] = make_float4(v[4], v[5], v[6], v[7]);
  } else {
    uint4 raw;
    T* h = reinterpret_cast<T*>(&raw);
    for (int j = 0; j < kVec; ++j) {
      h[j] = T(v[j]);
    }
    *reinterpret_cast<uint4*>(p) = raw;
  }
}

// f(k, n) for the elements k .. k + kVec of a linear block, n = kVec where they can be accessed as one vector
// (aligned, all within the tensor), else 0.
template <typename F>
__device__ void ForEachLinear(const Entry& e, int64_t local, F f) {
  const int64_t end = (local + 1) * kChunk < e.numel ? (local + 1) * kChunk : e.numel;
  for (int64_t k = local * kChunk + threadIdx.x * kVec; k < end; k += kThreads * kVec) {
    f(k, e.aligned && k + kVec <= e.numel ? kVec : 0);
  }
}

template <typename G, bool kNesterov>
__global__ void __launch_bounds__(kThreads) MuonMomentumKernel(const __grid_constant__ TensorList a, float mu) {
  __shared__ float tile[kTile][kTile + 1];
  int t;
  int64_t local;
  LocateBlock(a, t, local);
  const Entry& e = a.entry[t];
  float* m = a.contiguous[t];
  const G* g = static_cast<const G*>(a.grad[t]);
  c10::BFloat16* out = a.strided[t];
  const auto step = [mu](float& mk, float gk) {  // m <- mu m + g, then the update direction
    mk = __fmaf_rn(mu, mk, gk);
    return kNesterov ? __fmaf_rn(mu, mk, gk) : mk;
  };
  if (e.linear) {
    ForEachLinear(e, local, [&](int64_t k, int n) {
      if (n == kVec) {
        float mv[kVec], gv[kVec], uv[kVec];
        LoadVec(m + k, mv);
        LoadVec(g + k, gv);
        for (int j = 0; j < kVec; ++j) {
          uv[j] = step(mv[j], gv[j]);
        }
        StoreVec(m + k, mv);
        StoreVec(out + k, uv);
      } else {
        for (int64_t i = k; i < k + kVec && i < e.numel; ++i) {
          float mk = m[i];
          out[i] = c10::BFloat16(step(mk, static_cast<float>(g[i])));
          m[i] = mk;
        }
      }
    });
    return;
  }
  // tiled: m and g along columns, out along its contiguous dimension
  const StridedView& v = e.view;
  const Tile x = LocateTile(v, local);
  ForEachInTile(false, [&](int ri, int ci) {
    const int64_t r = x.r0 + ri, c = x.c0 + ci;
    if (r < v.R && c < v.C) {
      const int64_t k = (x.b * v.R + r) * v.C + c;
      float mk = m[k];
      tile[ri][ci] = step(mk, static_cast<float>(g[k]));
      m[k] = mk;
    }
  });
  __syncthreads();
  ForEachInTile(v.sC != 1, [&](int ri, int ci) {
    const int64_t r = x.r0 + ri, c = x.c0 + ci;
    if (r < v.R && c < v.C) {
      out[x.b * v.sB + r * v.sR + c * v.sC] = c10::BFloat16(tile[ri][ci]);
    }
  });
}

__global__ void __launch_bounds__(kThreads) MuonUpdateKernel(const __grid_constant__ TensorList a, float lr, float wd) {
  __shared__ float tile[kTile][kTile + 1];
  int t;
  int64_t local;
  LocateBlock(a, t, local);
  const Entry& e = a.entry[t];
  float* p = a.contiguous[t];
  const c10::BFloat16* u = a.strided[t];
  const float s = a.scale[t];
  const auto step = [lr, wd, s](float pk, float uk) {  // p - lr (wd p + s u)
    return __fmaf_rn(-lr, __fmaf_rn(wd, pk, __fmul_rn(s, uk)), pk);
  };
  if (e.linear) {
    ForEachLinear(e, local, [&](int64_t k, int n) {
      if (n == kVec) {
        float pv[kVec], uv[kVec];
        LoadVec(p + k, pv);
        LoadVec(u + k, uv);
        for (int j = 0; j < kVec; ++j) {
          pv[j] = step(pv[j], uv[j]);
        }
        StoreVec(p + k, pv);
      } else {
        for (int64_t i = k; i < k + kVec && i < e.numel; ++i) {
          p[i] = step(p[i], static_cast<float>(u[i]));
        }
      }
    });
    return;
  }
  // tiled: u along its contiguous dimension, p along columns
  const StridedView& v = e.view;
  const Tile x = LocateTile(v, local);
  ForEachInTile(v.sC != 1, [&](int ri, int ci) {
    const int64_t r = x.r0 + ri, c = x.c0 + ci;
    if (r < v.R && c < v.C) {
      tile[ri][ci] = static_cast<float>(u[x.b * v.sB + r * v.sR + c * v.sC]);
    }
  });
  __syncthreads();
  ForEachInTile(false, [&](int ri, int ci) {
    const int64_t r = x.r0 + ri, c = x.c0 + ci;
    if (r < v.R && c < v.C) {
      const int64_t k = (x.b * v.R + r) * v.C + c;
      p[k] = step(p[k], tile[ri][ci]);
    }
  });
}

template <typename S, typename D>
__global__ void __launch_bounds__(kThreads) BatchedCopyKernel(const __grid_constant__ CopyList a) {
  __shared__ float tile[kTile][kTile + 1];
  int t;
  int64_t local;
  LocateBlock(a, t, local);
  const Entry& e = a.entry[t];
  const S* src = static_cast<const S*>(a.src[t]);
  D* dst = static_cast<D*>(a.dst[t]);
  if (e.linear) {
    ForEachLinear(e, local, [&](int64_t k, int n) {
      if (n == kVec) {
        float v[kVec];
        LoadVec(src + k, v);
        StoreVec(dst + k, v);
      } else {
        for (int64_t i = k; i < k + kVec && i < e.numel; ++i) {
          dst[i] = D(static_cast<float>(src[i]));
        }
      }
    });
    return;
  }
  const StridedView &dv = e.view, &sv = a.src_view[t];
  const Tile x = LocateTile(dv, local);
  ForEachInTile(sv.sC != 1, [&](int ri, int ci) {
    const int64_t r = x.r0 + ri, c = x.c0 + ci;
    if (r < dv.R && c < dv.C) {
      tile[ri][ci] = static_cast<float>(src[x.b * sv.sB + r * sv.sR + c * sv.sC]);
    }
  });
  __syncthreads();
  ForEachInTile(dv.sC != 1, [&](int ri, int ci) {
    const int64_t r = x.r0 + ri, c = x.c0 + ci;
    if (r < dv.R && c < dv.C) {
      dst[x.b * dv.sB + r * dv.sR + c * dv.sC] = D(tile[ri][ci]);
    }
  });
}

bool Aligned(const void* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

// Fills Lists with the non-empty entries i, tiled by views[i] and linear if all tensors[.][i] are contiguous
// (vector accesses if they are all aligned); fill(list, i, j) sets entry j's pointers. Runs launch(list) on
// each full list and on the last.
template <typename List, typename Fill, typename Launch>
void ForEachList(const std::vector<torch::Tensor>& views, const std::vector<std::vector<torch::Tensor>>& tensors,
                 Fill fill, Launch launch) {
  constexpr int64_t kMax = sizeof(List::entry) / sizeof(Entry);
  List a;
  a.size = 0;
  a.first_block[0] = 0;
  for (size_t i = 0; i < views.size(); ++i) {
    if (views[i].numel() == 0) {
      continue;
    }
    const int j = static_cast<int>(a.size++);
    Entry& e = a.entry[j];
    e.view = AsStridedView(views[i]);
    e.numel = views[i].numel();
    e.linear = e.aligned = true;
    for (const auto& list : tensors) {
      e.linear &= list[i].is_contiguous();
      e.aligned &= Aligned(list[i].data_ptr());
    }
    fill(a, i, j);
    a.first_block[j + 1] = a.first_block[j] + NumBlocks(e);
    if (a.size == kMax) {
      launch(a);
      a.size = 0;
    }
  }
  if (a.size > 0) {
    launch(a);
  }
}

}  // namespace

void MuonMomentumCUDA(const std::vector<torch::Tensor>& momentum, const std::vector<torch::Tensor>& grad,
                      const std::vector<torch::Tensor>& out, double mu, bool nesterov) {
  const c10::cuda::CUDAGuard guard(momentum[0].device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, grad[0].scalar_type(), "MuonMomentumCUDA", [&] {
    ForEachList<TensorList>(
        out, {momentum, grad, out},
        [&](TensorList& a, size_t i, int j) {
          a.contiguous[j] = momentum[i].data_ptr<float>();
          a.grad[j] = grad[i].data_ptr<scalar_t>();
          a.strided[j] = out[i].data_ptr<c10::BFloat16>();
        },
        [&](const TensorList& a) {
          const auto kernel = nesterov ? MuonMomentumKernel<scalar_t, true> : MuonMomentumKernel<scalar_t, false>;
          kernel<<<a.first_block[a.size], kThreads, 0, stream>>>(a, static_cast<float>(mu));
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  });
}

void MuonUpdateCUDA(const std::vector<torch::Tensor>& param, const std::vector<torch::Tensor>& update,
                    const std::vector<double>& scale, double lr, double weight_decay) {
  const c10::cuda::CUDAGuard guard(param[0].device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ForEachList<TensorList>(
      update, {param, update},
      [&](TensorList& a, size_t i, int j) {
        a.contiguous[j] = param[i].data_ptr<float>();
        a.strided[j] = update[i].data_ptr<c10::BFloat16>();
        a.scale[j] = static_cast<float>(scale[i]);
      },
      [&](const TensorList& a) {
        MuonUpdateKernel<<<a.first_block[a.size], kThreads, 0, stream>>>(a, static_cast<float>(lr),
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
          dst, {dst, src},
          [&](CopyList& a, size_t i, int j) {
            a.dst[j] = dst[i].data_ptr();
            a.src[j] = src[i].data_ptr();
            a.src_view[j] = AsStridedView(src[i]);
          },
          [&](const CopyList& a) {
            BatchedCopyKernel<S, scalar_t><<<a.first_block[a.size], kThreads, 0, stream>>>(a);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
    });
  });
}

}  // namespace ops
}  // namespace xllm
