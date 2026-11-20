"""ReplicaBatchIterator: every rank of a data-parallel replica gets its reader's batches and data state, bitwise.

The replica's first rank (TP rank 0, CP rank 0) iterates a fake loader whose batches depend on the data-parallel rank
and the step; every rank rebuilds the batch it should get and compares x, y, mask and the two ratios bitwise, then
the data state. Layouts TP x CP: 8 x 1, 4 x 2, 2 x 2, 1 x 2 (and 1 x 1: no broadcast). Launch on 8 GPUs:

    torchrun --nproc_per_node 8 tests/data/test_replica_batches.py
"""
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

from xllm.data.data_types import Batch
from xllm.data.replica import ReplicaBatchIterator, is_replica_reader
from xllm.distributed import destroy_model_parallel, get_data_parallel_rank, initialize_model_parallel

BATCH, SEQ, STEPS = 3, 64, 4


def fake_batch(dp_rank: int, step: int) -> Batch:
    rng = np.random.default_rng(1000 * dp_rank + step)
    return Batch(
        x=rng.integers(0, 50_000, (BATCH, SEQ), dtype=np.int64),
        y=rng.integers(-100, 50_000, (BATCH, SEQ), dtype=np.int64),
        mask=rng.random((BATCH, SEQ)) < 0.7,
        padding_ratio=float(rng.random()),
        truncation_ratio=float(rng.random()) / 3,
    )


class FakeLoader:
    def __init__(self, dp_rank: int):
        self.dp_rank, self.steps, self.restored, self.closed = dp_rank, 0, None, False

    def __iter__(self):
        while True:
            self.steps += 1
            yield fake_batch(self.dp_rank, self.steps)

    def get_state(self):
        return {"steps": self.steps, "dp": self.dp_rank, "rng": np.arange(5) * self.dp_rank}

    def set_state(self, state):
        self.restored = state

    def close(self):
        self.closed = True


def check(tp: int, cp: int, errors: list):
    initialize_model_parallel(tp, cp, 60)
    dp = get_data_parallel_rank()
    reader = is_replica_reader()
    loader = FakeLoader(dp) if reader else None
    batches = iter(ReplicaBatchIterator(loader, BATCH, SEQ))
    tag = f"tp{tp} cp{cp} rank {dist.get_rank()}"
    for step in range(1, STEPS + 1):
        x, y, mask, ratios = next(batches)
        want = fake_batch(dp, step)
        got = (x.cpu().numpy(), y.cpu().numpy(), mask.cpu().numpy(), ratios.cpu().tolist())
        if not (np.array_equal(got[0], want.x) and np.array_equal(got[1], want.y)
                and np.array_equal(got[2], want.mask) and got[3] == [want.padding_ratio, want.truncation_ratio]):
            errors.append(f"{tag} step {step}: batch differs from its reader's")
        if x.dtype != torch.int64 or mask.dtype != torch.bool or ratios.dtype != torch.float64 or not x.is_cuda:
            errors.append(f"{tag}: dtypes/device {x.dtype} {mask.dtype} {ratios.dtype} {x.device}")
    state = batches.get_state()
    if state["steps"] != STEPS or state["dp"] != dp or not np.array_equal(state["rng"], np.arange(5) * dp):
        errors.append(f"{tag}: data state {state} is not its reader's")
    batches.set_state({"restored": True})
    batches.close()
    if reader and (loader.restored != {"restored": True} or not loader.closed):
        errors.append(f"{tag}: set_state / close did not reach the reader's loader")
    destroy_model_parallel()


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    errors = []
    for tp, cp in [(8, 1), (4, 2), (2, 2), (1, 2), (1, 1)]:
        check(tp, cp, errors)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, errors)
    errors = [e for es in gathered for e in es]
    if dist.get_rank() == 0:
        print("\n".join(errors) if errors else "replica batches: all layouts bitwise equal to their readers' (PASS)")
    dist.destroy_process_group()
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
