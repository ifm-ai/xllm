// Programmatic dependent launch (PDL) for the forward kernels of fwd_flat.cu, fwd_wide.cu and dense_lanes.cu.
#pragma once

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace xllm::ops::blackwell {

// kernel<<<grid, block, smem, stream>>>(args...) with cudaLaunchAttributeProgrammaticStreamSerialization: the kernel
// may start while the kernel ahead of it in the stream is still running. Every kernel launched here therefore calls
// cudaGridDependencySynchronize() as its first statement, before it reads an input or writes an output, and none of
// them triggers its successor early: the implicit trigger at kernel exit is the only one. smem stays within the 48 KB
// of dynamic shared memory a kernel takes without an opt-in.
template <class KernelFunc, class... Args>
void LaunchKernelPdl(KernelFunc kernel, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, Args... args) {
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  config.attrs = &attr;
  config.numAttrs = 1;
  AT_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::ops::blackwell
