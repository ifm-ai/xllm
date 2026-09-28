import os
from typing import Dict, Any
import json
import socket
import torch
from dataclasses import asdict
from datetime import datetime
from logging import getLogger
from pathlib import Path

from .comms_bench import collective_bench
from xllm.utils import get_n_restarts, mkdir

logger = getLogger()


def add_to_metrics(all_metrics: Dict[str, Any], new_metrics: Dict[str, Any], tag: str):
    tagged: Dict[str, Any] = {f"{tag}/{k}": v for k, v in new_metrics.items()}
    all_metrics.update(tagged)


def check_cluster(
    global_rank: int,
    dump_dir: Path,
    check_level: int,
    step: int
):
    logger.info("Starting cluster checks...")

    log_dir = dump_dir / "cluster_checks"
    jsonl_log_path = log_dir / "jsonl" / f"rank_{global_rank:06d}.jsonl"
    mkdir([jsonl_log_path.parent], global_rank == 0, exist_ok=True)

    n_restarts = get_n_restarts(dump_dir=dump_dir, rank=global_rank)
    all_metrics = {
        "at": datetime.now().isoformat(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "step": step,
        "job_id": os.environ.get("SLURM_JOB_ID", 0),
        "restart": os.environ.get("SLURM_RESTART_COUNT", 0),
        "n_restart": n_restarts,
    }

    if check_level >= 1:
        comms_bench = collective_bench()
        add_to_metrics(all_metrics, asdict(comms_bench), "comms_bench")

    with jsonl_log_path.open("a") as fp:
        fp.write(f"{json.dumps(all_metrics)}\n")

    logger.info(f"Done with cluster checks (logged at: {log_dir})")
    torch.cuda.empty_cache()
