# coding=utf-8
from typing import Optional, Tuple
import logging
import datetime
import torch

from .utils import ensure_divisibility, divide_and_check_no_remainder

logger = logging.getLogger()

# Model parallel group that the current rank belongs to.
_MODEL_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
_MODEL_PARALLEL_RANK: Optional[int] = None
_MODEL_PARALLEL_WORLD_SIZE: Optional[int] = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
_DATA_PARALLEL_RANK: Optional[int] = None
_DATA_PARALLEL_WORLD_SIZE: Optional[int] = None
# Context parallel group that the current rank belongs to.
_CONTEXT_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
_CONTEXT_PARALLEL_RANK: Optional[int] = None
_CONTEXT_PARALLEL_WORLD_SIZE: Optional[int] = None
# rank for the prev and next chunk in the context parallel group
_NEXT_CHUNK_RANK: Optional[int] = None
_PREV_CHUNK_RANK: Optional[int] = None
# HSDP group for checkpoint saving and loading.
_HYBRID_SHARD_DATA_PARALLEL_WORLD_SIZE: Optional[int] = None
_HYBRID_SHARD_DATA_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
# Expert outer group (the GPUs at the same TP rank in one expert group) that the current rank belongs to.
_EXPERT_OUTER_GROUP: Optional[torch.distributed.ProcessGroup] = None
_EXPERT_OUTER_RANK: Optional[int] = None
_EXPERT_OUTER_WORLD_SIZE: Optional[int] = None


def initialize_model_parallel(
    model_parallel_size: int,
    context_parallel_size: int,
    timeout: int,
    model_parallel_backend: Optional[str] = None,
    context_parallel_backend: Optional[str] = None,
    ddp_backend: Optional[str] = None,
    expert_parallel_size: Optional[int] = None
) -> None:
    """
    Initialize GPU parallel groups.

    Arguments:
        model_parallel_size: number of GPUs used to parallelize model.
        context_parallel_size: number of GPUs used to parallelize context.
        timeout: timeout for process group.
        model_parallel_backend: model parallel backend,
        context_parallel_backend: context parallel backend,
        ddp_backend: data parallel backend
        expert_parallel_size: number of GPUs used to split the routed experts,
            a multiple of model_parallel_size (None: same as model_parallel_size).

    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    ensure_divisibility(world_size, model_parallel_size)
    ensure_divisibility(world_size, model_parallel_size * context_parallel_size)
    rank = torch.distributed.get_rank()

    data_parallel_size = int(world_size / (model_parallel_size * context_parallel_size))
    if expert_parallel_size is None:
        expert_parallel_size = model_parallel_size
    # EP = TP x E_out: E_out consecutive data parallel replicas form an expert group
    expert_outer_size = divide_and_check_no_remainder(expert_parallel_size, model_parallel_size)
    ensure_divisibility(data_parallel_size, expert_outer_size)
    device = torch.device(torch.cuda.current_device())
    timeout = datetime.timedelta(seconds=timeout)

    if torch.distributed.get_rank() == 0:
        logger.info("> initializing data    parallel with size {}".format(data_parallel_size))
        logger.info("> initializing context parallel with size {}".format(context_parallel_size))
        logger.info("> initializing model   parallel with size {}".format(model_parallel_size))
        logger.info("> initializing expert  parallel with size {}".format(expert_parallel_size))

    groups = torch.LongTensor(range(world_size)).reshape(data_parallel_size, context_parallel_size, model_parallel_size)

    found = torch.where(groups == rank)
    assert all(len(x) == 1 for x in found)
    found = [x.item() for x in found]

    # Build the data parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    for j in range(context_parallel_size):
        for k in range(model_parallel_size):
            ranks = groups[:, j, k].tolist()
            group = torch.distributed.new_group(ranks, timeout=timeout, backend=ddp_backend, device_id=device)
            if j == found[1] and k == found[2]:
                _DATA_PARALLEL_GROUP = group

    # Build the model parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, "model parallel group is already initialized"
    for i in range(data_parallel_size):
        for j in range(context_parallel_size):
            ranks = groups[i, j, :].tolist()
            group = torch.distributed.new_group(
                ranks, timeout=timeout, backend=model_parallel_backend, device_id=device
            ) if model_parallel_size > 1 else None

            if i == found[0] and j == found[1]:
                _MODEL_PARALLEL_GROUP = group

    global _CONTEXT_PARALLEL_GROUP
    assert _CONTEXT_PARALLEL_GROUP is None, "context parallel group is already initialized"
    global _PREV_CHUNK_RANK, _NEXT_CHUNK_RANK
    assert _PREV_CHUNK_RANK is None and _NEXT_CHUNK_RANK is None, "chunk parallel group is already initialized"
    for i in range(data_parallel_size):
        for k in range(model_parallel_size):
            ranks = groups[i, :, k].tolist()
            group = torch.distributed.new_group(
                ranks, timeout=timeout, backend=context_parallel_backend, device_id=device
            ) if context_parallel_size > 1 else None

            if i == found[0] and k == found[2]:
                _CONTEXT_PARALLEL_GROUP = group
                prev = found[1] - 1
                next = found[1] + 1
                _PREV_CHUNK_RANK = None if prev < 0 else ranks[prev]
                _NEXT_CHUNK_RANK = ranks[next] if next < context_parallel_size else None

    global _HYBRID_SHARD_DATA_PARALLEL_GROUP
    assert _HYBRID_SHARD_DATA_PARALLEL_GROUP is None, "hsdp group is already initialized"
    groups = torch.LongTensor(range(world_size)).reshape(-1, model_parallel_size)
    for k in range(model_parallel_size):
        ranks = groups[:, k].tolist()
        group = torch.distributed.new_group(ranks, timeout=timeout, backend=ddp_backend, device_id=device)
        if k == found[2]:
            _HYBRID_SHARD_DATA_PARALLEL_GROUP = group

    # Build the expert groups. The GPU at TP rank k in replica o of its expert group owns the experts
    # [k * N / TP + o * N / EP, k * N / TP + (o + 1) * N / EP). With CP, each CP rank has its own expert groups.
    # The TP x E_out GPUs of an expert group need no group of their own: the dispatch goes over the outer
    # group, then the model parallel group. Neither do the GPUs owning the same experts: their FSDP mesh
    # (get_expert_device_mesh) has its own shard and replicate groups.
    global _EXPERT_OUTER_GROUP
    if expert_outer_size == 1:
        # Every replica is an expert group: the experts are split as the model is.
        _EXPERT_OUTER_GROUP = None
    else:
        # expert group x replica in expert group x CP x TP
        groups = torch.LongTensor(range(world_size)).reshape(
            -1, expert_outer_size, context_parallel_size, model_parallel_size
        )
        found = torch.where(groups == rank)
        assert all(len(x) == 1 for x in found)
        found = [x.item() for x in found]
        for g in range(groups.size(0)):
            for j in range(context_parallel_size):
                for k in range(model_parallel_size):
                    ranks = groups[g, :, j, k].tolist()
                    group = torch.distributed.new_group(
                        ranks, timeout=timeout, backend=model_parallel_backend, device_id=device
                    )
                    if g == found[0] and j == found[2] and k == found[3]:
                        _EXPERT_OUTER_GROUP = group


def model_parallel_is_initialized() -> bool:
    """Check if model and data parallel groups are initialized."""
    return _DATA_PARALLEL_GROUP is not None


def get_model_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the model parallel group the caller rank belongs to."""
    return _MODEL_PARALLEL_GROUP


def get_model_parallel_world_size() -> int:
    """Return world size for the model parallel group."""
    global _MODEL_PARALLEL_WORLD_SIZE
    if _MODEL_PARALLEL_WORLD_SIZE is None:
        group = get_model_parallel_group()
        _MODEL_PARALLEL_WORLD_SIZE = 1 if group is None else torch.distributed.get_world_size(group=group)
    return _MODEL_PARALLEL_WORLD_SIZE


def get_model_parallel_rank() -> int:
    """Return my rank for the model parallel group."""
    global _MODEL_PARALLEL_RANK
    if _MODEL_PARALLEL_RANK is None:
        group = get_model_parallel_group()
        _MODEL_PARALLEL_RANK = 0 if group is None else torch.distributed.get_rank(group=group)
    return _MODEL_PARALLEL_RANK


def get_data_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def get_data_parallel_world_size() -> int:
    """Return world size for the data parallel group."""
    global _DATA_PARALLEL_WORLD_SIZE
    if _DATA_PARALLEL_WORLD_SIZE is None:
        _DATA_PARALLEL_WORLD_SIZE = torch.distributed.get_world_size(group=get_data_parallel_group())
    return _DATA_PARALLEL_WORLD_SIZE


def get_data_parallel_rank() -> int:
    """Return my rank for the data parallel group."""
    global _DATA_PARALLEL_RANK
    if _DATA_PARALLEL_RANK is None:
        _DATA_PARALLEL_RANK = torch.distributed.get_rank(group=get_data_parallel_group())
    return _DATA_PARALLEL_RANK


def get_context_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the context parallel group the caller rank belongs to."""
    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_world_size() -> int:
    """Return world size for the context parallel group."""
    global _CONTEXT_PARALLEL_WORLD_SIZE
    if _CONTEXT_PARALLEL_WORLD_SIZE is None:
        group = get_context_parallel_group()
        _CONTEXT_PARALLEL_WORLD_SIZE = 1 if group is None else torch.distributed.get_world_size(group=group)
    return _CONTEXT_PARALLEL_WORLD_SIZE


def get_context_parallel_rank() -> int:
    """Return my rank for the context parallel group."""
    global _CONTEXT_PARALLEL_RANK
    if _CONTEXT_PARALLEL_RANK is None:
        group = get_context_parallel_group()
        _CONTEXT_PARALLEL_RANK = 0 if group is None else torch.distributed.get_rank(group=group)
    return _CONTEXT_PARALLEL_RANK


def get_context_parallel_prev_rank() -> Optional[int]:
    """Return the next rank for the context parallel group."""
    assert _CONTEXT_PARALLEL_RANK is not None, "context parallel group is not initialized"
    return _PREV_CHUNK_RANK


def get_context_parallel_next_rank() -> Optional[int]:
    """Return the next rank for the context parallel group."""
    assert _CONTEXT_PARALLEL_RANK is not None, "context parallel group is not initialized"
    return _NEXT_CHUNK_RANK


def get_hybrid_shard_data_parallel_group() -> torch.distributed.ProcessGroup:
    """Return world size for the hybrid shard data parallel group."""
    assert _HYBRID_SHARD_DATA_PARALLEL_GROUP is not None, "hsdp group is not initialized"
    return _HYBRID_SHARD_DATA_PARALLEL_GROUP


def get_hybrid_shard_data_parallel_world_size() -> int:
    """Return world size for the hybrid shard data parallel group world size."""
    global _HYBRID_SHARD_DATA_PARALLEL_WORLD_SIZE
    if _HYBRID_SHARD_DATA_PARALLEL_WORLD_SIZE is None:
        _HYBRID_SHARD_DATA_PARALLEL_WORLD_SIZE = torch.distributed.get_world_size(group=get_hybrid_shard_data_parallel_group())
    return _HYBRID_SHARD_DATA_PARALLEL_WORLD_SIZE


def get_expert_outer_group() -> torch.distributed.ProcessGroup:
    """Get the expert outer group the caller rank belongs to."""
    return _EXPERT_OUTER_GROUP


def get_expert_outer_world_size() -> int:
    """Return world size for the expert outer group (E_out = EP / TP)."""
    global _EXPERT_OUTER_WORLD_SIZE
    if _EXPERT_OUTER_WORLD_SIZE is None:
        group = get_expert_outer_group()
        _EXPERT_OUTER_WORLD_SIZE = 1 if group is None else torch.distributed.get_world_size(group=group)
    return _EXPERT_OUTER_WORLD_SIZE


def get_expert_outer_rank() -> int:
    """Return my rank for the expert outer group (the position of my replica in its expert group)."""
    global _EXPERT_OUTER_RANK
    if _EXPERT_OUTER_RANK is None:
        group = get_expert_outer_group()
        _EXPERT_OUTER_RANK = 0 if group is None else torch.distributed.get_rank(group=group)
    return _EXPERT_OUTER_RANK


def get_parallel_region(parallel_region: str) -> Tuple[int, int, torch.distributed.ProcessGroup]:
    if parallel_region == 'data':
        rank = get_data_parallel_rank()
        world_size = get_data_parallel_world_size()
        group = get_data_parallel_group()
    elif parallel_region == 'model':
        rank = get_model_parallel_rank()
        world_size = get_model_parallel_world_size()
        group = get_model_parallel_group()
    elif parallel_region == 'context':
        rank = get_context_parallel_rank()
        world_size = get_context_parallel_world_size()
        group = get_context_parallel_group()
    elif parallel_region == 'hybrid_shard':
        rank = None
        world_size = get_hybrid_shard_data_parallel_world_size()
        group = get_hybrid_shard_data_parallel_group()
    else:
        raise ValueError(f'Unknown parallel region: {parallel_region}')

    return rank, world_size, group


def destroy_model_parallel() -> None:
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _MODEL_PARALLEL_WORLD_SIZE
    _MODEL_PARALLEL_WORLD_SIZE = None
    global _MODEL_PARALLEL_RANK
    _MODEL_PARALLEL_RANK = None

    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _DATA_PARALLEL_WORLD_SIZE
    _DATA_PARALLEL_WORLD_SIZE = None
    global _DATA_PARALLEL_RANK
    _DATA_PARALLEL_RANK = None

    global _CONTEXT_PARALLEL_GROUP
    _CONTEXT_PARALLEL_GROUP = None
    global _CONTEXT_PARALLEL_WORLD_SIZE
    _CONTEXT_PARALLEL_WORLD_SIZE = None
    global _CONTEXT_PARALLEL_RANK
    _CONTEXT_PARALLEL_RANK = None
    global _PREV_CHUNK_RANK
    _PREV_CHUNK_RANK = None
    global _NEXT_CHUNK_RANK
    _NEXT_CHUNK_RANK = None

    global _HYBRID_SHARD_DATA_PARALLEL_GROUP
    _HYBRID_SHARD_DATA_PARALLEL_GROUP = None

    global _EXPERT_OUTER_GROUP
    _EXPERT_OUTER_GROUP = None
    global _EXPERT_OUTER_WORLD_SIZE
    _EXPERT_OUTER_WORLD_SIZE = None
    global _EXPERT_OUTER_RANK
    _EXPERT_OUTER_RANK = None
