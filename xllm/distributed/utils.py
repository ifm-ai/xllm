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


def get_expert_shard_size(fully_sharded_size: int, n_expert_groups: int) -> int:
    """FSDP shard group size of the routed experts: the most expert groups, up to fully_sharded_size, that split
    them evenly. Each expert group holds every expert once; the shard group splits those weights across expert
    groups, and its blocks of consecutive expert groups hold the same shards."""
    return max(d for d in range(1, min(fully_sharded_size, n_expert_groups) + 1) if n_expert_groups % d == 0)


def _new_groups(ranks: List[List[int]]) -> dist.ProcessGroup:
    """Create the groups of the lists of ranks, on every rank, as DeviceMesh does; returns this rank's group."""
    if dist.distributed_c10d._get_default_group().bound_device_id is not None:
        # the default group's communicator is initialized: one split for all the groups
        return dist.split_group(split_ranks=ranks)
    mine = None
    for group_ranks in ranks:
        group = dist.new_group(group_ranks)
        if dist.get_rank() in group_ranks:
            mine = group
    return mine


def get_expert_device_mesh(
    fully_sharded_size: int,
    context_parallel_size: int,
    model_parallel_size: int,
    expert_outer_size: int,
) -> DeviceMesh:
    """
    FSDP mesh of the routed experts: the expert data parallel GPUs (same TP rank and same replica in
    their expert group, one per expert group and CP rank), as (replicate, shard) like get_device_mesh.

    Arguments:
        fully_sharded_size: FSDP shard group size of the rest of the model.
        context_parallel_size: number of CP ranks.
        model_parallel_size: number of TP ranks.
        expert_outer_size: E_out = EP / TP, the number of replicas in an expert group.

    The experts are sharded over get_expert_shard_size(fully_sharded_size, DP / E_out) expert groups
    (min(fully_sharded_size, DP / E_out) when it divides DP / E_out), which keeps the shard groups of
    get_device_mesh when E_out divides DP / fully_sharded_size, and replicated over the others and the CP
    ranks. With E_out = 1 the ranks are those of get_device_mesh.

    Only the shard and the replicate groups are created, not the groups of a mesh spanning every expert
    parallel rank: those would be the expert parallel groups, never used.
    """
    world_size = dist.get_world_size()
    data_parallel_size = divide_and_check_no_remainder(world_size, context_parallel_size * model_parallel_size)
    n_expert_groups = divide_and_check_no_remainder(data_parallel_size, expert_outer_size)
    sharded_size = get_expert_shard_size(fully_sharded_size, n_expert_groups)
    # ranks in order: expert group x replica in expert group x CP x TP
    mesh = torch.arange(world_size).view(n_expert_groups, expert_outer_size, context_parallel_size, model_parallel_size)
    # expert parallel rank (replica, TP) x replicate (expert groups in a shard block, CP) x shard
    mesh = mesh.permute(1, 3, 0, 2).reshape(expert_outer_size * model_parallel_size, sharded_size, -1)
    mesh = mesh.transpose(1, 2).contiguous()
    rank = dist.get_rank()
    shard_group = _new_groups(mesh.reshape(-1, mesh.size(2)).tolist())
    replicate_group = _new_groups(mesh.transpose(1, 2).reshape(-1, mesh.size(1)).tolist()) if mesh.size(1) > 1 else None
    # this GPU's (replicate, shard) mesh; the ranks of each group are ascending, as their group ranks
    mesh = mesh[(mesh == rank).nonzero()[0, 0]].contiguous()
    if replicate_group is None:
        return DeviceMesh.from_group(shard_group, "cuda", mesh=mesh[0], mesh_dim_names=("dp",))
    return DeviceMesh.from_group([replicate_group, shard_group], "cuda", mesh=mesh, mesh_dim_names=("cp", "dp"))
