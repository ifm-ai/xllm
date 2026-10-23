"""
Trainer progress shared with the per-node ft agent through a small file in the
RAM-backed slot dir: a step counter and a pause flag (set during eval and
checkpointing, so their length never looks like a hang). Counters, not
timestamps: node clocks can differ.
"""

import os
from pathlib import Path

import numpy as np

PROGRESS_ENV = "FT_PROGRESS_FILE"


def progress_path(slot_dir: str, rank: int) -> Path:
    job = os.environ.get("SLURM_JOB_ID", "local")
    return Path(slot_dir) / f"ft_{job}_progress_r{rank:06d}"


class Progress:
    """[step counter, paused flag], memory-mapped; written by the trainer, read by the agent."""

    def __init__(self, path: Path, create: bool):
        """The agent creates (and resets) the file before spawning the trainer, which opens the existing one."""
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.zeros(2, dtype=np.int64).tofile(path)
        self.values = np.memmap(path, dtype=np.int64, mode="r+", shape=(2,))

    @staticmethod
    def for_trainer():
        path = os.environ.get(PROGRESS_ENV)
        return Progress(Path(path), create=False) if path else None

    def step(self, step: int):
        self.values[0] = step

    def pause(self, paused: bool):
        self.values[1] = int(paused)

    def read(self):
        return int(self.values[0]), bool(self.values[1])
