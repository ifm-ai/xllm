from typing import List, Tuple

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

from .sequential_mgmm import _sequential_multi_group_matmul
from .nested_mgmm import _nested_multi_group_matmul
from .torch_mgmm import _torch_multi_group_matmul
from .te_mgmm import _te_multi_group_matmul


class MultiGroupMatMulFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        mat1: Tensor,
        mat2: Tensor,
        group_sizes: List[int],
        transpose: bool,
        backend: str
    ) -> Tensor:
        y, _ = _multi_group_matmul(mat1, mat2, None, group_sizes, transpose, backend)
        ctx.save_for_backward(mat1, mat2)
        # save non-tensor attributes
        ctx.group_sizes = group_sizes
        ctx.transpose = transpose
        ctx.backend = backend
        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: Tensor
    ) -> Tuple[Tensor, Tensor, None, None, None]:
        mat1, mat2 = ctx.saved_tensors
        group_sizes = ctx.group_sizes
        transpose = ctx.transpose
        backend = ctx.backend

        mat1_grad, mat2_grad = _multi_group_matmul(y_grad, mat2, mat1, group_sizes, not transpose, backend)
        return mat1_grad, mat2_grad, None, None, None


def _multi_group_matmul(inp1, inp2, inp3, group_sizes: List[int], transpose: bool, backend: str):
    """
    Args:
        inp1 (FloatTensor, fp32, fp16 or bf16): the 1st input matrix, shape [bsz, d]
        inp2 (FloatTensor, fp32, fp16 or bf16): the 2nd input matrix, if transpose=False shape [n, d, d'], otherwise shape [n, d', d]
        inp3 (FloatTensor, fp32, fp16 or bf16 (optional): the 1st input matrix, shape [bsz, d']
        group_sizes (list of n integers): list of sizes of the group splits for inp1 [b1, b2, ..., bn] (sum([b1, b2, ..., bn]) == bsz)
        transpose (bool): flag of if inp2 is transposed for not.
        backend (str): backend implementation

    return:
        out1 (Tensor same type with inp1 and inp2): output with shape [bsz, d']
        out2 (Tensor same type with inp1 and inp2 (optional)): output with shape same as inp2

        [inp1_1, inp1_2, ..., inp1_n] = split(inp1, group_sizes), where the shape of inp1_i is [b_i, d]
        out1 = concat([out1_1, out1_2, ..., out1_n]), where shape of out_i is [b_i, d']
        out1_i = inp1_i x inp2_i, for i = 1, 2, ..., n

        [inp3_1, inp3_2, ..., inp3_n] = split(inp1, group_sizes), where the shape of inp1_i is [b_i, d']
        out2 = stack([out2_1, out2_2, ..., out2_n]), where shape of out_i is [d, d']
        out2_i = inp1_i.t() x inp3_i, for i = 1, 2, ..., n

    some typical values of batch size (bsz) and dimension d & d'

    bsz is typically from 4k to 64k
    d & d' are typically in the range from 1024 to 8192

    """
    if backend == 'sequential':
        return _sequential_multi_group_matmul(inp1, inp2, inp3, group_sizes, transpose)
    elif backend == 'nested':
        return _nested_multi_group_matmul(inp1, inp2, inp3, group_sizes, transpose)
    elif backend == 'torch':
        return _torch_multi_group_matmul(inp1, inp2, inp3, group_sizes, transpose)
    elif backend == 'te':
        return _te_multi_group_matmul(inp1, inp2, inp3, group_sizes, transpose)
    else:
        raise ValueError(f"Unknown backend: {backend}.")


mgmm = MultiGroupMatMulFunc.apply


def multi_group_matmul_fwd(
    inp1, inp2, group_sizes: List[int], transpose: bool, backend: str
) -> Tensor:
    """
    Args:
        inp1 (FloatTensor, fp32, fp16 or bf16): the 1st input matrix, shape [bsz, d]
        inp2 (FloatTensor, fp32, fp16 or bf16): the 2nd input matrix, if transpose=False shape [n, d, d'], otherwise shape [n, d', d]
        group_sizes (list of n integers): list of sizes of the group splits for inp1 [b1, b2, ..., bn] (sum([b1, b2, ..., bn]) == bsz)
        transpose (bool): flag of if inp2 is transposed for not.
        backend (str): backend implementation

    return:
        out (Tensor same type with inp1 and inp2): output with shape [bsz, d']

        [inp1_1, inp1_2, ..., inp1_n] = split(inp1, split_sizes), where the shape of inp1_i is [b_i, d]
        out = concat([out_1, out_2, ..., out_n]), where shape of out_i is [b_i, d']
        out_i = inp1_i x inp2_i, for i = 1, 2, ..., n

    some typical values of batch size (bsz) and dimension d & d'

    bsz is typically from 4k to 64k
    d & d' are typically in the range from 1024 to 8192

    """
    out, _ = _multi_group_matmul(inp1, inp2, None, group_sizes, transpose, backend)
    return out


def multi_group_matmul_bwd(
    out_grad, inp2, inp1, group_sizes: List[int], transpose: bool, backend: str
) -> Tuple[Tensor, Tensor]:
    """
    Args:
        out_grad (FloatTensor, fp32, fp16 or bf16): the 1st input matrix, shape [bsz, d']
        inp2 (FloatTensor, fp32, fp16 or bf16): the 2nd input matrix, if transpose=False shape [n, d, d'], otherwise shape [n, d', d]
        inp1 (FloatTensor, fp32, fp16 or bf16): the 1st input matrix, shape [bsz, d]
        group_sizes (list of n integers): list of sizes of the group splits for inp1 [b1, b2, ..., bn] (sum([b1, b2, ..., bn]) == bsz)
        transpose (bool): flag of if inp2 is transposed for not.
        backend (str): backend implementation

    return:
        grad_inp1 (Tensor same type with inp1 and inp2): output with shape [bsz, d]
        grad_inp2 (Tensor same type with inp1 and inp2): output with shape [n, d, d'] if transpose=False, otherwise [n, d', d]

        [inp1_1, inp1_2, ..., inp1_n] = split(inp1, split_sizes), where the shape of inp1_i is [b_i, d]
        out = concat([out1, out2, ..., outn]), where shape of out_i is [b_i, d']
        out_i = inp1_i x inp2_i, for i = 1, 2, ..., n

    some typical values of batch size (bsz) and dimension d & d'

    bsz is typically from 4k to 64k
    d & d' are typically in the range from 1024 to 8192

    """

    return _multi_group_matmul(out_grad, inp2, inp1, group_sizes, not transpose, backend)
