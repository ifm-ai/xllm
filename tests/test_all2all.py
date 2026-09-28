import sys
import datetime
from typing import Dict, Tuple, Any
import logging
import os
import numpy as np
import torch

logger = logging.getLogger()


def initialize_logger() -> logging.Logger:
    # log everything
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    # stdout: everything
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.NOTSET)

    # stderr: warnings / errors and above
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)

    # set stream handlers
    logger.handlers.clear()
    assert len(logger.handlers) == 0, logger.handlers
    logger.handlers.append(stdout_handler)
    logger.handlers.append(stderr_handler)

    return logger


def init_torch_distributed(timeout: int = 1800) -> Tuple[int, int]:
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - global_rank
        - world_size
    """
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    logger.info(f"Run launched with torchrun, local rank: {local_rank}")

    # set GPU device
    assert 0 <= local_rank < 8
    torch.cuda.set_device(local_rank)

    torch.distributed.init_process_group(
        init_method="env://",
        backend="nccl",
        timeout=datetime.timedelta(seconds=timeout),
    )

    assert global_rank == torch.distributed.get_rank()
    assert world_size == torch.distributed.get_world_size()

    # sanity check
    assert 0 <= local_rank <= global_rank < world_size

    return global_rank, world_size


if __name__ == "__main__":
    initialize_logger()

    global_rank, world_size = init_torch_distributed()

    seed = 42 + global_rank
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.set_default_device("cuda")
    bsz, n, topk, dim = 5, 16, 6, 3
    assert n % world_size == 0
    n_local_experts = n // world_size
    logits = torch.randn(bsz, n, requires_grad=False)
    logger.info(f"local logits: {logits}")
    torch.distributed.all_reduce(logits)
    logger.info(f"global logits: {logits})")

    indices = torch.topk(logits, topk, dim=-1)[1]
    logger.info(f"indices: {indices}")
    indices = indices.flatten()
    sorted_indices = torch.argsort(indices, stable=True)
    permute_indices = sorted_indices // topk
    logger.info(f"permute idx: {permute_indices}")

    tokens_per_expert = torch.bincount(indices, minlength=n)
    input_splits = tokens_per_expert.view(-1, n_local_experts).sum(dim=1).tolist()
    output_splits = [input_splits[global_rank]] * world_size
    logger.info(f"counts: {tokens_per_expert}")
    logger.info(f"input splits: {input_splits}")
    logger.info(f"output splits: {output_splits}")

    x = torch.arange(bsz, dtype=torch.float32, requires_grad=False) + global_rank * bsz
    x = x.view(bsz, 1).expand(bsz, dim)
    logger.info(f"x: {x}")
    x = x.index_select(0, permute_indices)
    logger.info(f"permute x: {x}")
    obsz = input_splits[global_rank]
    y = torch.empty(obsz * world_size, dim, requires_grad=False)
    torch.distributed.all_to_all_single(y, x, output_splits, input_splits)
    y = y.view(world_size, obsz, -1).transpose(0, 1).reshape(obsz, -1)
    logger.info(f"y: {y}")

    y = y.view(obsz, world_size, -1).transpose(0, 1).reshape(world_size * obsz, -1)
    z = torch.empty_like(x)
    torch.distributed.all_to_all_single(z, y, input_splits, output_splits)
    logger.info(f"z: {z}")
    inverse_indices = torch.argsort(sorted_indices, stable=True)
    out = torch.index_select(z, 0, inverse_indices)
    out = out.view(bsz, topk, -1)
    logger.info(f"out: {out}")
