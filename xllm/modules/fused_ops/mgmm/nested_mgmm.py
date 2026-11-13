from typing import List
import torch

from .sequential_mgmm import _sequential_multi_group_matmul_wgrad


def _nested_multi_group_matmul(inp1, inp2, inp3, group_sizes: List[int], transpose: bool):
    bsz, d_in = inp1.shape
    n_bsz, d1, d2 = inp2.shape
    d_out = d1 if transpose else d2
    assert len(group_sizes) == n_bsz

    out = torch.empty(bsz, d_out, dtype=inp1.dtype, device=inp1.device)
    mat1 = torch.nested.nested_tensor(torch.split(inp1, split_size_or_sections=group_sizes), layout=torch.jagged)
    mat2 = inp2.transpose(1, 2) if transpose else inp2
    torch.cat(torch.bmm(mat1, mat2).unbind(), dim=0, out=out)
    # the second output is computed per group, as the sequential backend does
    out2 = None if inp3 is None else _nested_multi_group_matmul_wgrad(inp1, inp2, inp3, group_sizes, transpose)
    return out, out2


_nested_multi_group_matmul_wgrad = _sequential_multi_group_matmul_wgrad
