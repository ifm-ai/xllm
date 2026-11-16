import glob
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

# The group-norm kernels for data-center Blackwell (compute capability 10.x: B200, B300) in xllm/csrc/ops/group_norm/
# are built only when every CUDA target is 10.x; the group-norm ops then send the calls those kernels serve to them and
# keep their portable kernels for everything else. Without TORCH_CUDA_ARCH_LIST, torch builds for the visible GPUs.
_archs = os.environ.get("TORCH_CUDA_ARCH_LIST", "").replace(";", " ").split()
_build_blackwell = bool(_archs) and all(arch.split(".")[0] == "10" for arch in _archs)
if not _archs and torch.cuda.is_available():
    _build_blackwell = torch.cuda.device_count() > 0 and all(
        torch.cuda.get_device_capability(i)[0] == 10 for i in range(torch.cuda.device_count())
    )
if _build_blackwell:
    CSRCS += sorted(glob.glob(os.path.join(PATH, "xllm/csrc/ops/group_norm/*.cu")))
    CSRCS += sorted(glob.glob(os.path.join(PATH, "xllm/csrc/ops/group_norm/*.cpp")))

INCLUDE_DIRS = [
    os.path.join(PATH, "xllm/csrc"),
    os.path.join(PATH, "xllm/csrc/transformer_engine"),
]

CXX_FLAGS = [
    "-O3",
    f"-std={get_torch_cxx_standard()}",
]

if _build_blackwell:
    CXX_FLAGS.append("-DXLLM_BLACKWELL")

NVCC_FLAGS = [
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--threads",
    "4",
]


def main():
    setup(
        name='xllm',
        version="1.2.0",
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
