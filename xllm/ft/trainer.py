"""
The trainer side of ft, kept out of train.py: train.py builds one FtTrainer when ft.enabled and calls its hooks
at fixed points of the run; with ft off none of this is created or called.
"""

import logging
import os
import shutil
import socket
from pathlib import Path
from typing import Any, Dict

from xllm.ft.node import gpus_per_node
from xllm.ft.replica import Replica
from xllm.ft.sizing import startup_gate

logger = logging.getLogger()


def prepare_log_dir(cfg):
    """
    With ft on, logs and metrics go to a per-host local dir, so a shared-FS outage cannot block a restart;
    every host creates its own dirs, as xLLM's mkdir() only creates them on global rank 0.
    """
    for sub in ("restarts", "cluster_checks/jsonl"):
        Path(cfg.log_dir, sub).mkdir(parents=True, exist_ok=True)
    logger.info(f"ft: logs and metrics of this host are in {cfg.log_dir}")


def training_state(optimizer, scheduler, state, data_iterator) -> Dict[str, Any]:
    """The rank's non-tensor training state for an ft copy (the tensors are copied byte for byte)."""
    return {
        "scheduler": scheduler.state_dict(),
        "param_groups": [{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups],
        "step": state.step,
        "scale": state.scale,
        "scale_updates": state.scale_updates,
        "clip_cumulative": state.clip_cumulative,
        "data_state": data_iterator.get_state(),
    }


class FtTrainer:
    def __init__(self, cfg, model, optimizer):
        # sizing gate: host memory now, GPU memory after the first steps
        self.gpu_peak = startup_gate(model, cfg.ft.slot_dir, cfg.dcp_for_optimizer)
        self.replica = Replica(model, optimizer, cfg.ft.slot_dir, cfg.ft.k, cfg.ft.mtbf_sys, gpus_per_node(),
                               self.gpu_peak)

    def step_start(self, step: int):
        self.gpu_peak.record(step)

    def step_end(self, optimizer, scheduler, state, data_iterator):
        """After the optimizer and scheduler steps: the in-memory copy every k steps."""
        self.replica.maybe_copy(state.step, lambda: training_state(optimizer, scheduler, state, data_iterator))

    def close(self, cfg):
        self.replica.close()
        if os.environ.get("LOCAL_RANK", "0") == "0":
            # bring this host's local logs and metrics back to the shared dump dir (one copy per host, by its local
            # rank 0); a shared-FS problem must not fail the run. One folder per run, so a requeue keeps the old ones.
            run = f"{os.environ.get('SLURM_JOB_ID', 'local')}.{os.environ.get('SLURM_RESTART_COUNT', '0')}"
            try:
                shutil.copytree(cfg.log_dir, Path(cfg.dump_dir) / "ft_logs" / socket.gethostname() / run,
                                dirs_exist_ok=True)
            except OSError as e:
                logger.warning(f"ft: could not copy the local logs back ({e}); they stay in {cfg.log_dir}")
