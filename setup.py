import os
from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_torch_cxx_standard():
    torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if torch_version >= (2, 10):
        torch_cxx_standard = "20"
    else:
        torch_cxx_standard = "17"

    print(f"PyTorch version: {torch.__version__}")
    print(f"Required C++ standard: c++{torch_cxx_standard}")
    return f"c++{torch_cxx_standard}"


PATH = os.path.dirname(os.path.abspath(__file__))

CSRCS = [
    os.path.join(PATH, "xllm/csrc/blas.cc"),
    os.path.join(PATH, "xllm/csrc/xllm_extension.cc"),
    os.path.join(PATH, "xllm/csrc/ops/attention.cc"),
    os.path.join(PATH, "xllm/csrc/ops/attention_kernel.cu"),
    os.path.join(PATH, "xllm/csrc/ops/multiseg_attention_kernel.cu"),
    os.path.join(PATH, "xllm/csrc/ops/ema_hidden.cc"),
    os.path.join(PATH, "xllm/csrc/ops/ema_hidden_kernel.cu"),
    os.path.join(PATH, "xllm/csrc/ops/ema_parameters.cc"),
    os.path.join(PATH, "xllm/csrc/ops/ema_parameters_kernel.cu"),
    os.path.join(PATH, "xllm/csrc/ops/cema_blelloch_scan.cc"),
    os.path.join(PATH, "xllm/csrc/ops/cema_blelloch_scan_kernel.cu"),
    os.path.join(PATH, "xllm/csrc/ops/cema_cub_scan.cc"),
    os.path.join(PATH, "xllm/csrc/ops/cema_cub_scan_kernel.cu"),
    os.path.join(PATH, "xllm/csrc/ops/fftconv.cc"),
    os.path.join(PATH, "xllm/csrc/ops/fftconv_kernel.cu"),
    os.path.join(PATH, "xllm/csrc/ops/group_layer_norm.cc"),
    os.path.join(PATH, "xllm/csrc/ops/group_layer_norm_kernel.cu"),
    os.path.join(PATH, "xllm/csrc/ops/group_rms_norm.cc"),
    os.path.join(PATH, "xllm/csrc/ops/group_rms_norm_kernel.cu"),
    os.path.join(PATH, "xllm/csrc/ops/sequence_norm.cc"),
    os.path.join(PATH, "xllm/csrc/ops/sequence_norm_kernel.cu"),
    os.path.join(PATH, "xllm/csrc/ops/timestep_norm.cc"),
    os.path.join(PATH, "xllm/csrc/ops/timestep_norm_kernel.cu"),
    os.path.join(PATH, "xllm/csrc/ops/timestep_decay_norm_kernel.cu"),
    os.path.join(PATH, "xllm/csrc/ops/timestep_decay_norm_cub.cc"),
    os.path.join(PATH, "xllm/csrc/ops/timestep_decay_norm_cub_kernel.cu"),
    os.path.join(PATH, "xllm/csrc/transformer_engine/extensions/gemm.cpp"),
    os.path.join(PATH, "xllm/csrc/transformer_engine/common.cpp"),
    os.path.join(PATH, "xllm/csrc/transformer_engine/util.cpp"),
    os.path.join(PATH, "xllm/csrc/transformer_engine/common/util/cuda_runtime.cpp"),
    os.path.join(PATH, "xllm/csrc/transformer_engine/common/util/cuda_driver.cpp"),
    os.path.join(PATH, "xllm/csrc/transformer_engine/common/transformer_engine.cpp"),
    os.path.join(PATH, "xllm/csrc/transformer_engine/common/gemm/cublaslt_gemm.cu"),
    os.path.join(PATH, "xllm/csrc/transformer_engine/common/util/multi_stream.cpp"),
    os.path.join(PATH, "xllm/csrc/transformer_engine/common/swizzle/swizzle.cu"),
    os.path.join(PATH, "xllm/csrc/transformer_engine/common/common.cu")
]

INCLUDE_DIRS = [
    os.path.join(PATH, "xllm/csrc"),
    os.path.join(PATH, "xllm/csrc/transformer_engine"),
]

CXX_FLAGS = [
    "-O3",
    f"-std={get_torch_cxx_standard()}",
]

NVCC_FLAGS = [
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--threads",
    "4",
]


def main():
    setup(
        name='xllm',
        version="1.0.1",
        license_files=[
            "LICENSE",
            "xllm/csrc/transformer_engine/LICENSE",
            "xllm/csrc/transformer_engine/NOTICE",
        ],
        ext_modules=[
            CUDAExtension("xllm_extension",
                          CSRCS,
                          include_dirs=INCLUDE_DIRS,
                          extra_compile_args={
                              "cxx": CXX_FLAGS,
                              "nvcc": CXX_FLAGS + NVCC_FLAGS,
                          })
        ],
        cmdclass={'build_ext': BuildExtension},
    )


if __name__ == "__main__":
    main()
