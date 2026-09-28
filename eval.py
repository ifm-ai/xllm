import json

from dataclasses import dataclass
from logging import getLogger
from typing import Optional
from pathlib import Path
import os

import torch

from xllm.config import ValidConf
from xllm.checkpointing import ckpt_path_to_step
from xllm.eval.execute import execute_evals
from xllm.logger import initialize_logger
from xllm.metrics import MetricLogger
from xllm.configuration import Config, cfg_from_cli
from xllm.reloading import reload_model
from xllm.distributed.slurm import is_master, get_global_rank
from xllm.utils import setup_env, log_host, get_default_inference_half, mkdir

from xllm.distributed import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
)

logger = getLogger()


@dataclass
class EvalConf(Config):
    valid: ValidConf
    checkpoint_dir: str
    dump_dir: str  # main directory where evals are dumped
    full_state: bool = False
    tokenizer_path: Optional[str] = None
    dtype: str = get_default_inference_half()
    model_parallel_size: Optional[int] = None  # if None using the one in the checkpoint
    multi_segments: bool = True
    metrics_logger_dir: Optional[str] = None  # dir to log metrics, for async_eval
    seed: int = 42
    disable_workers_print: bool = True
    nccl_timeout: int = 1800

    def __post_init__(self):
        checkpoint_dir = Path(self.checkpoint_dir)
        assert checkpoint_dir.is_dir(), checkpoint_dir
        if os.path.exists(self.dump_dir):
            assert os.path.isdir(self.dump_dir)
        else:
            os.makedirs(self.dump_dir, exist_ok=True)
        assert os.path.isdir(self.dump_dir)


def main(eval_cfg: EvalConf):
    initialize_logger()
    setup_env()
    log_host()

    # only process 0 prints
    if get_global_rank() > 0 and eval_cfg.disable_workers_print:
        logger.info(f"No print for worker {get_global_rank()}")
        logger.disabled = True

    mkdir([Path(eval_cfg.dump_dir)], is_master(), exist_ok=True)
    if is_master():
        with (Path(eval_cfg.dump_dir) / "eval_config.json").open("w") as fp:
            fp.write(eval_cfg.to_json())

    model, tokenizer, ckpt_cfg = reload_model(
        checkpoint_dir=eval_cfg.checkpoint_dir,
        full_state=eval_cfg.full_state,
        model_parallel_size=eval_cfg.model_parallel_size,
        context_parallel_size=1,
        dtype=eval_cfg.dtype,
        tokenizer_path=eval_cfg.tokenizer_path,
        timeout=eval_cfg.nccl_timeout,
    )

    logger.info(f"checkpoint config: {ckpt_cfg}")
    logger.info(model)

    torch.backends.cuda.matmul.allow_tf32 = True

    scores = execute_evals(
        model=model,
        tokenizer=tokenizer,
        cfg=eval_cfg.valid,
        multi_segments=eval_cfg.multi_segments,
        dump_dir=eval_cfg.dump_dir,
        world_rank=get_data_parallel_rank(),
        world_size=get_data_parallel_world_size(),
        seed=eval_cfg.seed,
        is_master=is_master(),
    )
    log = " - ".join([f"{k}: {v:.2f}" for k, v in scores.items()])
    logger.info(f"ALL RESULTS: {log}")
    logger.info(f"num_alloc_retries: {torch.cuda.memory_stats()['num_alloc_retries']}")

    if is_master():
        assert eval_cfg.dump_dir is not None
        res_path = os.path.join(eval_cfg.dump_dir, "results.json")
        with open(res_path, "w") as fp:
            logger.info(f"Dumping results in {res_path}")
            json.dump(scores, fp, sort_keys=True, indent=4)

    if eval_cfg.metrics_logger_dir is not None:
        metric_logger = MetricLogger(outdir=Path(eval_cfg.metrics_logger_dir), tag="eval", enable=is_master(), log_tb=is_master())
        metric_logger.log(scores, step=ckpt_path_to_step(Path(eval_cfg.checkpoint_dir)))
        metric_logger.close()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # eval config
    cfg: EvalConf = cfg_from_cli(schema=EvalConf)
    # evaluate
    main(cfg)
