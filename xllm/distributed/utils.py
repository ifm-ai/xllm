# coding=utf-8

from typing import List, Union, Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh


Scalar = Union[int, float]
ReduceOpMap = {
    "sum": dist.ReduceOp.SUM,
    "max": dist.ReduceOp.MAX,
    "min": dist.ReduceOp.MIN,
    "mean": dist.ReduceOp.AVG,
    "avg": dist.ReduceOp.AVG,
}


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide_and_check_no_remainder(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_specific_dim(
    tensor: torch.Tensor, num_partitions: int, split_dim: int, contiguous_split_chunks: bool = False
) -> List[torch.Tensor]:
    """
    Split a tensor along a specific dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        split_dim: the dimension along which to split
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    split_dim_size = divide_and_check_no_remainder(tensor.shape[split_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, split_dim_size, dim=split_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return [chunk.contiguous() for chunk in tensor_list]

    return tensor_list


def reshape_gathered_tensor_along_specific_dim(
    tensor: torch.Tensor, gather_dim: int,
) -> torch.Tensor:
    """
    reshape a tensor along a specific dimension for gather
    Arguments:
        tensor: input tensor.
        gather_dim: the dimension along which to split
    """
    shape = tensor.shape
    gather_dim_size = shape[gather_dim + 1] * shape[0]
    permute_order = list(range(1, gather_dim + 1)) + [0,] + list(range(gather_dim + 1, tensor.dim()))
    tensor = tensor.permute(permute_order)
    new_shape = shape[1:gather_dim + 1] + (gather_dim_size,) + shape[gather_dim + 2:]
    tensor = tensor.reshape(new_shape)

    return tensor


def reshape_scattered_tensor_along_specific_dim(
    tensor: torch.Tensor, num_partitions: int, scatter_dim: int,
) -> torch.Tensor:
    """
    reshape a tensor along a specific dimension for reduce_scatter
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        scatter_dim: the dimension along which to split
    """
    shape = tensor.shape
    scatter_dim_size = divide_and_check_no_remainder(shape[scatter_dim], num_partitions)
    new_shape = shape[:scatter_dim] + (num_partitions, scatter_dim_size) + shape[scatter_dim + 1:]
    tensor = tensor.reshape(new_shape)
    permute_order = [scatter_dim,] + list(range(scatter_dim)) + list(range(scatter_dim + 1, tensor.dim()))
    tensor = tensor.permute(permute_order).contiguous()

    return tensor


def reduce_scalar(x: Scalar, op: str, group: Optional[dist.ProcessGroup] = None) -> Scalar:
    tensor = torch.tensor(x).cuda()
    torch.distributed.all_reduce(tensor, op=ReduceOpMap[op.lower()], group=group)
    out = tensor.item()
    return out


def reduce_scalars(x: List[Scalar], op: str, group: Optional[dist.ProcessGroup] = None) -> List[Scalar]:
    tensor = torch.tensor(x).cuda()
    torch.distributed.all_reduce(tensor, op=ReduceOpMap[op.lower()], group=group)
    out = tensor.tolist()
    return out


def get_device_mesh(
    fully_sharded_size: int,
    replicated_size: int,
    model_parallel_size: int,
) -> DeviceMesh:
    world_size = dist.get_world_size()
    assert world_size == fully_sharded_size * replicated_size * model_parallel_size
    if replicated_size == 1:
        if model_parallel_size == 1:
            mesh = torch.arange(world_size)
            device_mesh = DeviceMesh(
                device_type="cuda",
                mesh=mesh,
                mesh_dim_names=("dp",),
            )
        else:
            mesh = torch.arange(world_size).view(fully_sharded_size, model_parallel_size)
            device_mesh = DeviceMesh(
                device_type="cuda",
                mesh=mesh,
                mesh_dim_names=("dp", "tp"),
            )["dp"]
    else:
        if model_parallel_size == 1:
            mesh = torch.arange(world_size).view(fully_sharded_size, replicated_size)
            mesh = mesh.swapdims(0, 1)
            device_mesh = DeviceMesh(
                device_type="cuda",
                mesh=mesh,
                mesh_dim_names=("cp", "dp"),
            )
        else:
            mesh = torch.arange(world_size).view(fully_sharded_size, replicated_size, model_parallel_size)
            mesh = mesh.swapdims(0, 1)
            device_mesh = DeviceMesh(
                device_type="cuda",
                mesh=mesh,
                mesh_dim_names=("cp", "dp", "tp"),
            )["cp", "dp"]

    return device_mesh
