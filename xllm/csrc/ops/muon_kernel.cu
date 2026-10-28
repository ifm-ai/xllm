#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>

#include "ops/muon.h"

namespace xllm {
namespace ops {

namespace {

// Each block steps a kTile x kTile tile of one view, kTile threads per row, kRows rows at a time. A view
// whose columns are strided (a transposed block) goes through shared memory, so that both the fp32 side
// (along columns) and the view (along rows) are accessed contiguously.
constexpr int kTile = 32;
constexpr int kRows = 8;
// Tensors per launch: the list travels as kernel arguments, within CUDA's 32 KB parameter limit (sm_70+).
constexpr int kMaxTensors = 256;

struct TensorList {
  int64_t size;
  int64_t first_tile[kMaxTensors + 1];  // tile offsets, and the total
  float* contiguous[kMaxTensors];        // momentum / param
  const void* grad[kMaxTensors];         // momentum only
  c10::BFloat16* strided[kMaxTensors];   // out / update
  StridedView view[kMaxTensors];
  float scale[kMaxTensors];              // update only
};

__host__ __device__ int64_t DivUp(int64_t a, int64_t b) { return (a + b - 1) / b; }

int64_t NumTiles(const StridedView& v) { return v.B * DivUp(v.R, kTile) * DivUp(v.C, kTile); }

// The tile blockIdx.x of a list: its tensor and the tile's batch, first row and first column.
struct Tile {
  int t;
  int64_t b, r0, c0;
};

__device__ Tile LocateTile(const TensorList& a) {
  const int64_t id = blockIdx.x;
  int lo = 0, hi = static_cast<int>(a.size) - 1;  // the last tensor whose first tile is <= id
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (a.first_tile[mid] <= id) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const StridedView& v = a.view[lo];
  const int64_t tiles_c = DivUp(v.C, kTile), per_batch = DivUp(v.R, kTile) * tiles_c;
  const int64_t local = id - a.first_tile[lo], in_batch = local % per_batch;
  return {lo, local / per_batch, in_batch / tiles_c * kTile, in_batch % tiles_c * kTile};
}

template <typename G, bool kNesterov>
__global__ void MuonMomentumKernel(const __grid_constant__ TensorList a, float mu) {
  __shared__ float tile[kTile][kTile + 1];
  const Tile x = LocateTile(a);
  const StridedView& v = a.view[x.t];
  float* m = a.contiguous[x.t] + x.b * v.R * v.C;
  const G* g = static_cast<const G*>(a.grad[x.t]) + x.b * v.R * v.C;
  c10::BFloat16* out = a.strided[x.t] + x.b * v.sB;
  const bool transposed = v.sC != 1;
  for (int i = threadIdx.y; i < kTile; i += kRows) {
    const int64_t r = x.r0 + i, c = x.c0 + threadIdx.x;
    if (r < v.R && c < v.C) {
      const int64_t k = r * v.C + c;
      const float gk = static_cast<float>(g[k]);
      const float mk = __fmaf_rn(mu, m[k], gk);
      m[k] = mk;
      const float u = kNesterov ? __fmaf_rn(mu, mk, gk) : mk;
      if (transposed) {
        tile[i][threadIdx.x] = u;
      } else {
        out[r * v.sR + c] = c10::BFloat16(u);
      }
    }
  }
  if (transposed) {
    __syncthreads();
    for (int i = threadIdx.y; i < kTile; i += kRows) {
      const int64_t r = x.r0 + threadIdx.x, c = x.c0 + i;
      if (r < v.R && c < v.C) {
        out[r * v.sR + c * v.sC] = c10::BFloat16(tile[threadIdx.x][i]);
      }
    }
  }
}

__global__ void MuonUpdateKernel(const __grid_constant__ TensorList a, float lr, float wd) {
  __shared__ float tile[kTile][kTile + 1];
  const Tile x = LocateTile(a);
  const StridedView& v = a.view[x.t];
  float* p = a.contiguous[x.t] + x.b * v.R * v.C;
  const c10::BFloat16* u = a.strided[x.t] + x.b * v.sB;
  const float s = a.scale[x.t];
  const bool transposed = v.sC != 1;
  if (transposed) {
    for (int i = threadIdx.y; i < kTile; i += kRows) {
      const int64_t r = x.r0 + threadIdx.x, c = x.c0 + i;
      if (r < v.R && c < v.C) {
        tile[threadIdx.x][i] = static_cast<float>(u[r * v.sR + c * v.sC]);
      }
    }
    __syncthreads();
  }
  for (int i = threadIdx.y; i < kTile; i += kRows) {
    const int64_t r = x.r0 + i, c = x.c0 + threadIdx.x;
    if (r < v.R && c < v.C) {
      const float uk = transposed ? tile[i][threadIdx.x] : static_cast<float>(u[r * v.sR + c]);
      const int64_t k = r * v.C + c;
      p[k] = __fmaf_rn(-lr, __fmaf_rn(wd, p[k], __fmul_rn(s, uk)), p[k]);  // p - lr (wd p + s u)
    }
  }
}

// Fills TensorLists with the non-empty pairs (contiguous[i], strided[i]), fill(list, i, j) adding the rest of
// entry j, and runs launch(list) on each full one and on the last.
template <typename Fill, typename Launch>
void ForEachList(const std::vector<torch::Tensor>& contiguous, const std::vector<torch::Tensor>& strided,
                 Fill fill, Launch launch) {
  TensorList a;
  a.size = 0;
  a.first_tile[0] = 0;
  for (size_t i = 0; i < strided.size(); ++i) {
    if (strided[i].numel() == 0) {
      continue;
    }
    const int j = static_cast<int>(a.size++);
    a.contiguous[j] = contiguous[i].data_ptr<float>();
    a.strided[j] = strided[i].data_ptr<c10::BFloat16>();
    a.view[j] = AsStridedView(strided[i]);
    a.first_tile[j + 1] = a.first_tile[j] + NumTiles(a.view[j]);
    fill(a, i, j);
    if (a.size == kMaxTensors) {
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
  const dim3 block(kTile, kRows);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, grad[0].scalar_type(), "MuonMomentumCUDA", [&] {
    ForEachList(
        momentum, out, [&](TensorList& a, size_t i, int j) { a.grad[j] = grad[i].data_ptr<scalar_t>(); },
        [&](const TensorList& a) {
          const auto kernel = nesterov ? MuonMomentumKernel<scalar_t, true> : MuonMomentumKernel<scalar_t, false>;
          kernel<<<a.first_tile[a.size], block, 0, stream>>>(a, static_cast<float>(mu));
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  });
}

void MuonUpdateCUDA(const std::vector<torch::Tensor>& param, const std::vector<torch::Tensor>& update,
                    const std::vector<double>& scale, double lr, double weight_decay) {
  const c10::cuda::CUDAGuard guard(param[0].device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ForEachList(
      param, update, [&](TensorList& a, size_t i, int j) { a.scale[j] = static_cast<float>(scale[i]); },
      [&](const TensorList& a) {
        MuonUpdateKernel<<<a.first_tile[a.size], dim3(kTile, kRows), 0, stream>>>(
            a, static_cast<float>(lr), static_cast<float>(weight_decay));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

}  // namespace ops
}  // namespace xllm
