from typing import Optional, Dict, Any, TextIO
from datetime import datetime
from logging import getLogger
from pathlib import Path
import json
from torch.utils.tensorboard import SummaryWriter
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

logger = getLogger()


class MetricLogger:
    def __init__(
        self,
        outdir: Path,
        tag: Optional[str] = None,
        enable: bool = False,
        log_tb: bool = False,
        log_wandb: bool = False,
    ):
        self.outdir = outdir
        self.tag = tag
        self.enable = enable
        self.log_tb = log_tb
        self.log_wandb = log_wandb and has_wandb

        self.tb_writer: Optional[SummaryWriter] = None
        self.jsonl_writer: Optional[TextIO] = None

        if not enable:
            assert not log_tb
            return

        outdir.mkdir(exist_ok=True, parents=True)
        if log_tb:
            tb_dir = outdir / "tb"
            tb_dir.mkdir(exist_ok=True, parents=False)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir), max_queue=1000)

        fname = f"metrics.{tag}.jsonl" if tag else "metrics.jsonl"
        fpath = outdir / fname
        self.jsonl_writer = open(fpath, "a")

    def log(self, metrics: Dict[str, Any], step: int):
        if not self.enable:
            return

        # check / sanitize inputs
        assert all(type(k) is str for k in metrics.keys())

        # log in tensorboard
        if self.tb_writer is not None:
            for k, v in metrics.items():
                k = k if self.tag is None else f"{self.tag}/{k}"
                self.tb_writer.add_scalar(tag=k, scalar_value=v, global_step=step)

        # log in wandb
        if self.log_wandb:
            for k, v in metrics.items():
                k = k if self.tag is None else f"{self.tag}/{k}"
                wandb.log({k: v}, step=step)

        created_at = datetime.utcnow().isoformat()
        json_metrics = dict(global_step=step, created_at=created_at)
        json_metrics.update(metrics)
        print(json.dumps(json_metrics), file=self.jsonl_writer, flush=True)

    def close(self):
        if self.tb_writer is not None:
            self.tb_writer.close()
            self.tb_writer = None
        if self.jsonl_writer is not None:
            self.jsonl_writer.close()
            self.jsonl_writer = None


def build_mlogger(outdir: str, tag: str, enable: bool, enable_wandb: bool=False) -> MetricLogger:
    return MetricLogger(outdir=Path(outdir), tag=tag, enable=enable, log_tb=enable, log_wandb=enable_wandb)
