// Muon's momentum and update, each one launch over a list of contiguous tensors.
#include "ops/muon_kernel.cuh"

namespace xllm {
namespace ops {

namespace {

using namespace muon;

// Contiguous fp32 tensors (momentum / param), each paired with a contiguous tensor of its numel (grad / update).
struct TensorList : Blocks<512> {
  int64_t numel[kCapacity];
  bool vec[kCapacity];
  float* fp32[kCapacity];
  const void* other[kCapacity];
  c10::BFloat16* out[kCapacity];  // momentum only
  float scale[kCapacity];         // update only
};

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

}  // namespace ops
}  // namespace xllm
