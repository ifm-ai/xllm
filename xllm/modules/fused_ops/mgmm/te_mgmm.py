from typing import List, Tuple
import torch
import functools
import os

from xllm_extension import te

# TE dtype mapping
TE_DType = {
    torch.uint8: te.DType.kByte,
    torch.float8_e4m3fn: te.DType.kFloat8E4M3,
    torch.float8_e5m2: te.DType.kFloat8E5M2,
    torch.int32: te.DType.kInt32,
    torch.float32: te.DType.kFloat32,
    torch.half: te.DType.kFloat16,
    torch.bfloat16: te.DType.kBFloat16,
}

# Global workspace cache
_workspace_cache = None


def get_num_cublas_streams():
    return 4


@functools.lru_cache
def _get_device_compute_capability(device: torch.device) -> Tuple[int, int]:
    props = torch.cuda.get_device_properties(device)
    return (props.major, props.minor)


def get_device_compute_capability() -> Tuple[int, int]:
    """CUDA compute capability of current GPU"""
    return _get_device_compute_capability(torch.cuda.current_device())


def get_cublas_workspace_size_bytes() -> int:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if get_device_compute_capability()[0] >= 9:  # Hopper
        return 33_554_432
    return 4_194_304


def get_workspace(device: torch.device) -> List[torch.Tensor]:
    """Get workspace tensors for GEMM operations."""
    global _workspace_cache
    if _workspace_cache is None:
        num_streams = get_num_cublas_streams()
        workspace_size = get_cublas_workspace_size_bytes()
        _workspace_cache = [
            torch.empty(workspace_size, dtype=torch.uint8, device=device)
            for _ in range(num_streams)
        ]
    return _workspace_cache


@functools.lru_cache(maxsize=None)
def _empty_tensors(n: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, ...]:
    return tuple(torch.empty(0, device=device, dtype=dtype) for _ in range(n))


@functools.lru_cache
def get_sm_count() -> int:
    """Returns the number of streaming multiprocessors."""
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _te_multi_group_matmul(inp1, inp2, inp3, group_sizes: List[int], transpose: bool):
    bsz, d_in = inp1.shape
    n_groups, d1, d2 = inp2.shape
    
    assert len(group_sizes) == n_groups
    activation_dtype = inp1.dtype

    # inp1 is [bsz, d]
    inp1_te = list(torch.split(inp1, group_sizes, dim=0))
    # inp2 is [n, d, d']
    inp2_te = list(torch.unbind(inp2, dim=0))
    out_features = d1 if transpose else d2

    # out1 is [bsz, d']
    out1 = torch.empty(bsz, out_features, dtype=activation_dtype, device=inp1.device)
    
    # Get workspace and empty tensors
    workspaces = get_workspace(inp1.device)
    bias_empty = _empty_tensors(n_groups, inp1.device, torch.bfloat16)
    pre_gelu_empty = _empty_tensors(n_groups, inp1.device, torch.bfloat16)

    # Call TE grouped GEMM
    sm_count = get_sm_count()
    margin_sm = int(os.getenv("NVTE_EXT_MARGIN_SM", "0"))
    
    te.general_grouped_gemm(
        inp2_te,                    # A matrices
        transpose,                  # transa
        inp1_te,                    # B matrices
        False,                      # transb
        [out1],                     # output matrices
        TE_DType[activation_dtype], # output dtype
        group_sizes,                # m_splits
        bias_empty,                 # bias (unused)
        TE_DType[torch.bfloat16],   # bias_dtype
        True,                       # single_output
        pre_gelu_empty,             # pre_gelu_out
        False,                      # grad
        workspaces,                 # workspaces
        workspaces[0].shape[0],     # workspace_size
        False,                      # accumulate
        False,                      # use_split_accumulator
        sm_count - margin_sm,       # available_sm_count
    )
    
    # Handle second output if needed using TE grouped GEMM
    out2 = None
    if inp3 is not None:
        # inp3 is [bsz, d']
        inp3_te = list(torch.split(inp3, group_sizes, dim=0))

        # out2 is same shape as inp2
        out2 = torch.empty_like(inp2)
        if transpose:
            # inp3.T @ inp1 -> [d', d]
            te.general_grouped_gemm(
                inp1_te,                    # A matrices
                False,                      # transa
                inp3_te,                    # B matrices
                True,                       # transb
                [out2],                     # output matrices
                TE_DType[activation_dtype], # output dtype
                group_sizes,                # m_splits
                bias_empty,                 # bias (unused)
                TE_DType[torch.bfloat16],   # bias_dtype
                True,                       # single_output = False
                pre_gelu_empty,             # pre_gelu_out
                False,                      # grad
                workspaces,                 # workspaces
                workspaces[0].shape[0],     # workspace_size
                False,                      # accumulate
                False,                      # use_split_accumulator
                sm_count - margin_sm,       # available_sm_count
            )
        else:
            # inp1.T @ inp3 -> [d, d']
            te.general_grouped_gemm(
                inp3_te,                    # A matrices
                False,                      # transa
                inp1_te,                    # B matrices
                True,                       # transb
                [out2],                     # output matrices
                TE_DType[activation_dtype], # output dtype
                group_sizes,                # m_splits
                bias_empty,                 # bias (unused)
                TE_DType[torch.bfloat16],   # bias_dtype
                True,                       # single_output = False
                pre_gelu_empty,             # pre_gelu_out
                False,                      # grad
                workspaces,                 # workspaces
                workspaces[0].shape[0],     # workspace_size
                False,                      # accumulate
                False,                      # use_split_accumulator
                sm_count - margin_sm,       # available_sm_count
            )
    
    return out1, out2
