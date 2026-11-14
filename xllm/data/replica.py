"""Read each training batch once per data-parallel replica.

The training data is split by data-parallel rank only, so the TP and CP ranks of a replica train on the same
batches. Instead of every one of them reading and tokenizing the same data, the replica's first rank (TP rank 0,
CP rank 0) does it and sends each batch to the others: over its CP group, then every CP rank over its TP group.
The batches are bitwise the same as when every rank reads them.
"""
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from xllm.distributed import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_world_size,
    get_model_parallel_group,
    get_model_parallel_rank,
    get_model_parallel_world_size,
)
from .dataloader import MultiSourceDataLoader


def is_replica_reader() -> bool:
    """Whether this rank reads the data of its replica (TP rank 0 and CP rank 0)."""
    return get_model_parallel_rank() == 0 and get_context_parallel_rank() == 0


def _broadcast_from_first(tensor_or_obj: Any, group, world_size: int, is_tensor: bool) -> Any:
    # the first rank of a TP or CP group is its rank 0 (both groups hold consecutive positions)
    if world_size == 1:
        return tensor_or_obj
    src = dist.get_process_group_ranks(group)[0]
    if is_tensor:
        dist.broadcast(tensor_or_obj, src, group=group)
        return tensor_or_obj
    objs = [tensor_or_obj]
    dist.broadcast_object_list(objs, src, group=group)
    return objs[0]


def _to_replica(tensor_or_obj: Any, is_tensor: bool) -> Any:
    """From the replica's first rank to all its ranks: over its CP group (TP rank 0 only), then the TP group."""
    if get_model_parallel_rank() == 0:
        tensor_or_obj = _broadcast_from_first(
            tensor_or_obj, get_context_parallel_group(), get_context_parallel_world_size(), is_tensor
        )
    return _broadcast_from_first(tensor_or_obj, get_model_parallel_group(), get_model_parallel_world_size(), is_tensor)


class ReplicaBatchIterator:
    """The batches of a MultiSourceDataLoader built on the replica's first rank only (loader None elsewhere).

    next() returns (x, y, mask, ratios) on every rank of the replica, on the current GPU: x and y int64 and mask
    bool, batch_size x seq_len (MultiSourceDataLoader always gives a mask), and ratios the batch's float64
    (padding_ratio, truncation_ratio). Nothing is read back to the host, so the next step's kernels can be launched
    ahead as before. One broadcast per group and batch, of 3 x batch_size x seq_len + 2 int64. get_state is the
    reader's on every rank (one object broadcast), so the saved data state does not depend on which rank reads;
    set_state restores it on the reader only.
    """

    def __init__(self, loader: Optional[MultiSourceDataLoader], batch_size: int, seq_len: int):
        assert (loader is not None) == is_replica_reader()
        self.loader = loader
        self.batch_size = batch_size
        self.seq_len = seq_len
        self._batches = None

    def __iter__(self) -> "ReplicaBatchIterator":
        if self.loader is not None:
            self._batches = iter(self.loader)
        return self

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        n = self.batch_size * self.seq_len
        # x, y, mask, then padding_ratio and truncation_ratio as float64 bits
        buf = torch.empty(3 * n + 2, dtype=torch.int64, device=torch.cuda.current_device())
        if self.loader is not None:
            batch = next(self._batches)
            shape = (self.batch_size, self.seq_len)
            assert batch.x.shape == shape and batch.y.shape == shape and batch.mask.shape == shape, \
                (batch.x.shape, batch.y.shape, batch.mask.shape, shape)
            buf[:n] = torch.from_numpy(batch.x).flatten().cuda()
            buf[n:2 * n] = torch.from_numpy(batch.y).flatten().cuda()
            buf[2 * n:3 * n] = torch.from_numpy(batch.mask).flatten().cuda()
            buf[3 * n:] = torch.tensor(
                [batch.padding_ratio, batch.truncation_ratio], dtype=torch.float64
            ).view(torch.int64).cuda()
        buf = _to_replica(buf, is_tensor=True)
        x = buf[:n].view(self.batch_size, self.seq_len)
        y = buf[n:2 * n].view(self.batch_size, self.seq_len)
        mask = buf[2 * n:3 * n].view(self.batch_size, self.seq_len).bool()
        return x, y, mask, buf[3 * n:].view(torch.float64)

    def get_state(self) -> Dict[str, Any]:
        return _to_replica(self.loader.get_state() if self.loader is not None else None, is_tensor=False)

    def set_state(self, state: Dict[str, Any]) -> None:
        if self.loader is not None:
            self.loader.set_state(state)

    def close(self) -> None:
        if self.loader is not None:
            self.loader.close()
