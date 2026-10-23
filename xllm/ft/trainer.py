"""
The trainer side of ft, kept out of train.py: train.py builds one FtTrainer when ft.enabled and calls its hooks
at fixed points of the run; with ft off none of this is created or called.
"""

import logging

from xllm.ft.sizing import startup_gate

logger = logging.getLogger()


class FtTrainer:
    def __init__(self, cfg, model, optimizer):
        # sizing gate: host memory now, GPU memory after the first steps
        self.gpu_peak = startup_gate(model, cfg.ft.slot_dir, cfg.dcp_for_optimizer)

    def step_start(self, step: int):
        self.gpu_peak.record(step)
