"""
The ft save rule: under ft a memory restore can resume before checkpoints that already exist, so training
reaches their step again (called from xllm/checkpointing.py save_latest_checkpoint when ft is enabled).
"""

import os
from logging import getLogger
from pathlib import Path
from typing import Tuple

import torch

from xllm.checkpointing import CHECKPOINT_FOLDER_NAME, _get_sorted_checkpoints, ckpt_path_to_step

logger = getLogger()

FT_SAVE, FT_SKIP = 0, 1


def ft_save_decision(dump_dir_: Path, checkpoint_dir: Path, step: int, global_rank: int) -> Tuple[int, bool]:
    """
    Under ft a memory restore can resume before checkpoints that already exist, so training reaches their step
    again. Rank 0 decides for every rank (collective): a complete checkpoint (config.json written last) is kept
    and the save skipped, since the replay writes the same state; an incomplete one is renamed aside, out of the
    way of the checkpoint globs, and saved again. Also returns whether a newer checkpoint exists (a rewind save),
    for which the retention below must not run.
    """
    # the same device on every rank: the backend's, never a per-rank guess
    nccl = torch.distributed.is_initialized() and torch.distributed.get_backend() == "nccl"
    decision = torch.zeros(2, dtype=torch.int64, device="cuda" if nccl else "cpu")
    if global_rank == 0:
        if (checkpoint_dir / "config.json").exists():
            decision[0] = FT_SKIP
        elif checkpoint_dir.exists():
            epoch = f"{os.environ.get('SLURM_JOB_ID', 'local')}.{os.environ.get('SLURM_RESTART_COUNT', '0')}"
            n = 0  # the same step can be found incomplete again later in the run
            while (stale := checkpoint_dir.parent / f".stale-{epoch}-{n}-{checkpoint_dir.name}").exists():
                n += 1
            logger.warning(f"ft: incomplete checkpoint {checkpoint_dir} moved to {stale}")
            checkpoint_dir.rename(stale)
        existing = _get_sorted_checkpoints(ckpt_folder=dump_dir_ / CHECKPOINT_FOLDER_NAME)
        decision[1] = int(len(existing) > 0 and ckpt_path_to_step(existing[-1]) > step)
    if torch.distributed.is_initialized():
        torch.distributed.broadcast(decision, src=0)
    return int(decision[0].item()), bool(decision[1].item())
