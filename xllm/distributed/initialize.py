# coding=utf-8
from typing import Optional, Tuple
import logging
import datetime
import torch

from .utils import ensure_divisibility

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


def initialize_model_parallel(
    model_parallel_size: int,
    context_parallel_size: int,
    timeout: int,
    model_parallel_backend: Optional[str] = None,
    context_parallel_backend: Optional[str] = None,
    ddp_backend: Optional[str] = None
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

    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    ensure_divisibility(world_size, model_parallel_size)
    ensure_divisibility(world_size, model_parallel_size * context_parallel_size)
    rank = torch.distributed.get_rank()

    data_parallel_size = int(world_size / (model_parallel_size * context_parallel_size))
    device = torch.device(torch.cuda.current_device())
    timeout = datetime.timedelta(seconds=timeout)

    if torch.distributed.get_rank() == 0:
        logger.info("> initializing data    parallel with size {}".format(data_parallel_size))
        logger.info("> initializing context parallel with size {}".format(context_parallel_size))
        logger.info("> initializing model   parallel with size {}".format(model_parallel_size))

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
