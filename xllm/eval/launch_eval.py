import os
import time
from logging import getLogger
from typing import Optional, TypeVar, Callable, Any
import json
from pathlib import Path
import contextlib
import submitit
from submitit.core.utils import FailedJobError
import torch.nn as nn

from xllm.config import TrainerConf, SlurmConf
from xllm.checkpointing import get_checkpoint_dir
from xllm.data.dataset_streamer.tokenizer import Tokenizer
from xllm.utils import mkdir
from xllm.eval.execute import execute_evals
from eval import EvalConf, main as eval_fn

logger = getLogger()


# ---------------- launch -------------------

C = TypeVar("C")


@contextlib.contextmanager
def reset_slurm_env():
    """
    Temporarily unset slurm ids to avoid mistakenly inheriting.
    """
    old_environ = {
        x: os.environ.pop(x, None)
        for x in [
            "SLURM_JOB_ID",
            "SLURM_NTASKS",
            "SLURM_JOB_NUM_NODES",
            "SLURM_NODEID",
            "SLURM_JOB_NODELIST",
            "SLURM_PROCID",
            "SLURM_LOCALID",
            "SLURM_ARRAY_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
            "SLURM_CPU_BIND",
            "SLURM_CPU_BIND_TYPE",
            "SLURM_CPU_BIND_LIST",
        ]
    }
    try:
        yield
    finally:
        for k, val in old_environ.items():
            if val is not None:
                os.environ[k] = val


@reset_slurm_env()
def launch_with_submitit(
    fn: Callable[[C], Any],
    cfg: C,
    slurm_cfg: SlurmConf,
    n_gpus: int,
    folder: Path,
    job_name: str,
):
    assert n_gpus % slurm_cfg.gpus_per_node == 0, f"{n_gpus}, {slurm_cfg.gpus_per_node}"
    assert n_gpus > 0

    extra_params = {
        "slurm_time": slurm_cfg.time,
        "slurm_timeout_min": slurm_cfg.timeout_min,
        "slurm_gpus_per_node": slurm_cfg.gpus_per_node,
        "slurm_ntasks_per_node": slurm_cfg.gpus_per_node,
        "slurm_cpus_per_task": slurm_cfg.cpus_per_task,
    }
    if slurm_cfg.partition is not None:
        extra_params.update({"slurm_partition": slurm_cfg.partition})
    if slurm_cfg.account is not None:
        extra_params.update({"slurm_account": slurm_cfg.account})
    if slurm_cfg.qos is not None:
        extra_params.update({"slurm_qos": slurm_cfg.qos})
    if slurm_cfg.mem_gb is not None:
        extra_params.update({"mem_gb": slurm_cfg.mem_gb})
    if slurm_cfg.additional_params is not None:
        extra_params.update({"slurm_additional_parameters": json.loads(slurm_cfg.additional_params)})

    executor = submitit.AutoExecutor(folder=folder, slurm_max_num_timeout=-1)
    executor.update_parameters(
        nodes=int(n_gpus // slurm_cfg.gpus_per_node),
        slurm_job_name=job_name,
        slurm_srun_args=["-vv"],
        **extra_params,
    )

    # submit job
    backoff = 60 * 5  # wait 5 minutes if the slurm scheduler times out
    max_attempts = 3
    for attempt_id in range(max_attempts):
        try:
            _ = executor.submit(fn, cfg)
            return True
        except FailedJobError as e:
            logger.warning(
                f"Error submitting submit job {job_name}"
                f"({attempt_id + 1}/{max_attempts}): {e}"
            )
            time.sleep(backoff)
    logger.warning(f"Didn't managed to submit {job_name}")
    return False


def launch_async_eval(
    cfg: TrainerConf,
    step: int,
    eval_folder: Optional[Path] = None,
    eval_name: Optional[str] = None,
):
    checkpoint_dir = get_checkpoint_dir(dump_dir_=Path(cfg.dump_dir), step=step)
    assert checkpoint_dir.exists(), checkpoint_dir
    if eval_folder is None:
        eval_folder = Path(cfg.dump_dir) / "eval" / f"{step:08d}"
    assert isinstance(eval_folder, Path)
    assert cfg.slurm.is_master
    eval_folder.mkdir(exist_ok=True, parents=True)
    if eval_name is None:
        eval_name = f"{Path(cfg.dump_dir).name}_eval{step}"
    assert isinstance(eval_name, str)

    eval_cfg = EvalConf(
        valid=cfg.valid,
        multi_segments=cfg.multi_segments,
        checkpoint_dir=str(checkpoint_dir),
        full_state=False,
        dump_dir=str(eval_folder),
        dtype=cfg.dtype,
        disable_workers_print=False,
        metrics_logger_dir=cfg.dump_dir,
        model_parallel_size=cfg.model_parallel_size,
        nccl_timeout=cfg.nccl_timeout,
    )
    with (eval_folder / "cfg.json").open("w") as fp2:
        fp2.write(eval_cfg.to_json())

    succeed = launch_with_submitit(
        eval_fn,
        eval_cfg,
        cfg.slurm,
        n_gpus=cfg.async_eval_ngpus,
        folder=eval_folder,
        job_name=eval_name,
    )
    if succeed:
        logger.info(f"Launched async eval for step: {step}.")
    else:
        logger.warning(f"Failed to launch async eval for step: {step}.")


def launch_sync_eval(
    cfg: TrainerConf,
    model: nn.Module,
    tokenizer: Tokenizer,
    step: int
):
    dump_dir = Path(cfg.dump_dir) / "eval" / str(step)
    mkdir([dump_dir], cfg.slurm.is_master, exist_ok=True)
    scores = execute_evals(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg.valid,
        multi_segments=cfg.multi_segments,
        dump_dir=str(dump_dir),
        world_rank=cfg.data_parallel_rank,
        world_size=cfg.data_parallel_size,
        is_master=cfg.slurm.is_master,
    )
    return scores
