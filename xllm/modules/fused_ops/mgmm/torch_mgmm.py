from typing import List
import torch
import torch.nn.functional as F


def _torch_multi_group_matmul(inp1, inp2, inp3, group_sizes: List[int], transpose: bool):
    n_bsz, d1, d2 = inp2.shape
    assert len(group_sizes) == n_bsz
    group_sizes = torch.tensor(group_sizes, dtype=torch.int32, device=inp1.device)
    offsets = torch.cumsum(group_sizes, dim=0, dtype=torch.int32)

    out = F.grouped_mm(inp1, inp2.transpose(1, 2) if transpose else inp2, offs=offsets)
    if inp3 is None:
        return out, None

    if transpose:
        out2 = F.grouped_mm(inp3.transpose(0, 1), inp1, offs=offsets)
    else:
        out2 = F.grouped_mm(inp1.transpose(0, 1), inp3, offs=offsets)

    return out, out2
