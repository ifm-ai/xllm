// Batched copies between strided views: the gathers and scatters around Muon's all-to-alls.
#include "ops/muon_kernel.cuh"

namespace xllm {
namespace ops {

namespace {

using namespace muon;

struct CopyList : Blocks<192> {
  StridedView dst_view[kCapacity], src_view[kCapacity];
  bool linear[kCapacity];  // both contiguous
  bool vec[kCapacity];
  void* dst[kCapacity];
  const void* src[kCapacity];
};

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
