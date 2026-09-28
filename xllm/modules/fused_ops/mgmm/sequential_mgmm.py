from typing import List
import torch


def _sequential_multi_group_matmul(inp1, inp2, inp3, group_sizes: List[int], transpose: bool):
    bsz, d_in = inp1.shape
    n_bsz, d1, d2 = inp2.shape
    d_out = d1 if transpose else d2
    assert len(group_sizes) == n_bsz

    out = torch.empty(bsz, d_out, dtype=inp1.dtype, device=inp1.device)
    out2 = None if inp3 is None else torch.zeros_like(inp2)
    start = 0
    for i in range(n_bsz):
        if group_sizes[i] == 0:
            continue

        end = start + group_sizes[i]
        mat1 = inp1[start:end]
        mat2 = inp2[i].t() if transpose else inp2[i]
        torch.mm(mat1, mat2, out=out[start:end])
        if out2 is not None:
            mat3 = inp3[start:end]
            torch.mm(mat3.t(), mat1, out=out2[i]) if transpose else torch.mm(mat1.t(), mat3, out=out2[i])
        start = end

    return out, out2
