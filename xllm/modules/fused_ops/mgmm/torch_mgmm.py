from typing import List
import torch
import torch.nn.functional as F


def _offsets(group_sizes: List[int], device) -> torch.Tensor:
    group_sizes = torch.tensor(group_sizes, dtype=torch.int32, device=device)
    return torch.cumsum(group_sizes, dim=0, dtype=torch.int32)


def _grouped_wgrad(inp1, inp3, offsets: torch.Tensor, transpose: bool):
    if transpose:
        return F.grouped_mm(inp3.transpose(0, 1), inp1, offs=offsets)
    return F.grouped_mm(inp1.transpose(0, 1), inp3, offs=offsets)


def _torch_multi_group_matmul(inp1, inp2, inp3, group_sizes: List[int], transpose: bool):
    n_bsz, d1, d2 = inp2.shape
    assert len(group_sizes) == n_bsz
    offsets = _offsets(group_sizes, inp1.device)

    out = F.grouped_mm(inp1, inp2.transpose(1, 2) if transpose else inp2, offs=offsets)
    out2 = None if inp3 is None else _grouped_wgrad(inp1, inp3, offsets, transpose)
    return out, out2


def _torch_multi_group_matmul_wgrad(inp1, inp2, inp3, group_sizes: List[int], transpose: bool):
    # the second output of _torch_multi_group_matmul (inp3 given)
    n_bsz = inp2.shape[0]
    assert len(group_sizes) == n_bsz
    return _grouped_wgrad(inp1, inp3, _offsets(group_sizes, inp1.device), transpose)
