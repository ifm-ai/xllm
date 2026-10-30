import os
from typing import Union, Tuple, List, Callable, Optional, Dict, Any
from datetime import datetime
import time
from pathlib import Path
import json
import socket
from functools import lru_cache
from logging import getLogger
from functools import partial
import contextlib
import random
import numpy as np
import math
import subprocess
import sys
import xllm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions
)

from xllm.distributed import FullyShardedDataParallel
from xllm.distributed import (
    get_parallel_region,
    get_data_parallel_world_size,
    get_model_parallel_world_size,
    get_context_parallel_world_size,
)
from xllm.distributed.utils import (
    reduce_scalar,
    reduce_scalars
)

logger = getLogger()


def get_parallel_ranks(
    global_rank: int,
    model_parallel_size: Optional[int] = None,
    context_parallel_size: Optional[int] = None
) -> Tuple[int, int, int]:
    if model_parallel_size is None and context_parallel_size is None:
        model_parallel_size = get_model_parallel_world_size()
        context_parallel_size = get_context_parallel_world_size()
    # model parallel rank
    model_parallel_rank = global_rank % model_parallel_size
    # context parallel rank
    global_rank = global_rank // model_parallel_size
    context_parallel_rank = global_rank % context_parallel_size
    # data parallel rank
    data_parallel_rank = global_rank // context_parallel_size

    return data_parallel_rank, context_parallel_rank, model_parallel_rank


def get_master_rank_in_replicated_group(
    global_rank: int,
    fully_sharded_size: Optional[int],
) -> int:
    data_parallel_size = get_data_parallel_world_size()
    model_parallel_size = get_model_parallel_world_size()
    context_parallel_size = get_context_parallel_world_size()

    fully_sharded_size = data_parallel_size if fully_sharded_size is None else fully_sharded_size
    assert data_parallel_size % fully_sharded_size == 0
    replicated_size = data_parallel_size // fully_sharded_size * context_parallel_size

    data_rank, _, model_rank = get_parallel_ranks(global_rank, model_parallel_size, replicated_size)
    return data_rank * replicated_size * model_parallel_size + model_rank


@lru_cache()
def get_default_training_half() -> str:
    return "bf16"


@lru_cache()
def get_default_inference_half() -> str:
    return "fp16"


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_torch_dtype(dtype: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]


def get_init_fn(mode, dim=None, std=None) -> Callable[[torch.Tensor], torch.Tensor]:
    if mode == 'none':
        return lambda x: x
    elif mode == 'he':
        assert std is None, 'Do not specify std in he initialization'
        a = math.sqrt(5.0)
        init_fn = partial(nn.init.kaiming_normal_, a=a)
    elif mode == 'gaussian':
        assert (dim is not None) or (std is not None), 'dim or std need to be specified'
        if std is None:
            std = 1.0 / math.sqrt(dim)
        a = -3 * std
        b = 3 * std
        init_fn = partial(nn.init.trunc_normal_, mean=0.0, std=std, a=a, b=b)
    elif mode == 'xavier':
        assert std is None, 'Do not specify std in xavier initialization'
        init_fn = partial(nn.init.xavier_uniform_)
    else:
        raise ValueError('Unknown init mode: {}'.format(mode))

    return init_fn


def aggregate_dict(metrics: Dict[str, float], ops: List[str]) -> Dict[str, float]:
    keys = sorted(metrics.keys())
    all_results = {}
    for op in ops:
        all_results[op] = reduce_scalars([metrics[k] for k in keys], op=op)

    agg = {}
    for op, results in all_results.items():
        for i, k in enumerate(keys):
            agg[f"{k}_{op}"] = float(results[i])
    return agg


def log_restart_infos(dump_dir: str, global_rank: int, step: int):
    restart_folder = Path(dump_dir) / "restarts"
    mkdir([restart_folder], global_rank == 0, exist_ok=True)
    info = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", -1),
        "restart": os.environ.get("SLURM_RESTART_COUNT", 0),
        "step": step,
        "at": datetime.utcnow().isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
    with (restart_folder / f"rank_{global_rank:06d}").open("a") as fp:
        fp.write(f"{json.dumps(info)}\n")
    return info


def get_n_restarts(dump_dir: Path, rank: int) -> int:
    log_path = dump_dir / "restarts" / f"rank_{rank:06d}"

    if not log_path.exists():
        logger.info(f"{log_path} doesn't exist")
        return -1

    with log_path.open("r") as fp:
        lines = fp.readlines()

    return len(lines) - 1


@contextlib.contextmanager
def create_on_gpu():
    original_device = torch.get_default_device()
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_device(torch.cuda.current_device())
        torch.set_default_dtype(torch.float32)
        yield
    finally:
        torch.set_default_dtype(original_dtype)
        torch.set_default_device(original_device)


def num_parameters(model: nn.Module, cfg):
    nparams: int = sum(params.nelement() for params in model.parameters())
    nparams = int(reduce_scalar(nparams, op="sum"))
    replicated_size = cfg.context_parallel_size * cfg.data_parallel_size
    if cfg.model.ddp_backend == 'fsdp1':
        fully_sharded_size = cfg.data_parallel_size if cfg.fully_sharded_size is None else cfg.fully_sharded_size
        replicated_size = replicated_size // fully_sharded_size

    assert nparams % replicated_size == 0
    # redundant params in context parallel groups
    nparams = nparams // replicated_size
    assert nparams % cfg.model_parallel_size == 0
    logger.info(f"Total num. parameters: {nparams}.")


@torch.no_grad()
def check_same_in_process_group(x: torch.Tensor, name: str, group_name: str):
    _, world_size, group = get_parallel_region(group_name)
    if world_size == 1:
        assert group is None or group_name == 'data'
        return

    assert isinstance(x, torch.Tensor)
    assert x.dtype in {torch.int64, torch.int32, torch.bool}
    x = x.cuda()
    max_x = x.detach().clone()
    dist.all_reduce(max_x, op=dist.ReduceOp.MAX, group=group)
    if not torch.equal(max_x, x):
        msg = f"ISSUE: different tensor {name} detected in the same {group_name} parallel group !!! - "
        raise RuntimeError(msg)


def check_random_for_sync(n_elements):
    x = torch.randint(n_elements, (n_elements,))
    check_same_in_process_group(x, name='random', group_name='model')
    check_same_in_process_group(x, name='random', group_name='context')
    check_same_in_process_group(x, name='random', group_name='data')


def check_batch_for_sync(batch: Dict):
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            continue
        check_same_in_process_group(value, name=key, group_name='model')
        check_same_in_process_group(value, name=key, group_name='context')


def mkdir(paths: List[Path], creator: bool, exist_ok: bool):
    if creator:
        for path in paths:
            path.mkdir(exist_ok=exist_ok, parents=True)

    wait_for_folder_to_exists(paths, log_interval=60, sleep_interval=10, timeout=300)


def wait_for_folder_to_exists(
    tgt_dirs: List[Path],
    log_interval: float,
    sleep_interval: float,
    timeout: float
):
    start = time.time()
    last_log = time.time()
    while time.time() - start <= timeout:
        if all([tgt_dir.exists() for tgt_dir in tgt_dirs]):
            return

        if time.time() - last_log > log_interval:
            last_log = time.time()
            logger.info(
                f"Waiting for {tgt_dirs} to exist for {int(time.time() - start)}s (timeout {int(timeout)}s)"
            )
        time.sleep(sleep_interval)

    raise RuntimeError(
        f"Timeout waiting for {tgt_dirs} to exist - NFS sync issue for this host ?"
    )


def wait_for_all_files_to_be_in_dir(
    root_dir: Path,
    filename_to_rank: Dict[str, int],
    log_interval: float,
    sleep_interval: float,
    timeout: float,
):
    expected = set(filename_to_rank.keys())
    world_size = len(set(filename_to_rank.values()))
    start = time.time()
    last_log = time.time()
    present = {p.name for p in root_dir.iterdir()}
    while time.time() - start <= timeout:
        present = {p.name for p in root_dir.iterdir()}
        missing = expected - present
        if len(missing) == 0:
            return

        if time.time() - last_log > log_interval:
            last_log = time.time()
            missing_ranks = {filename_to_rank[name] for name in missing}
            msg = (
                f"- missing ranks: {sorted(missing_ranks)}"
                if len(missing_ranks) <= 16
                else ""
            )
            logger.info(
                f"Waiting for {len(missing_ranks)} ranks on {world_size} "
                f"to exists in {root_dir} for {int(time.time() - start)}s (timeout {int(timeout)}s){msg}"
            )
        time.sleep(sleep_interval)

    missing = expected - present
    missing_ranks = {filename_to_rank[name] for name in missing}

    raise RuntimeError(
        f"Timeout waiting for all files to exists in {root_dir}, missing ranks: {missing_ranks}"
    )


def setup_env():
    env_vars = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "1800",
    }
    for name, value in env_vars.items():
        if os.environ.get(name) != value:
            os.environ[name] = value
            logger.warning(f"WARNING: Setting {name} to {value}")


def log_host():
    # logging on stdout / stderr, helpful when debbuging some jobs
    logger.warning(f"Host: {socket.gethostname()}")
    logger.warning(f"Job hosts: {os.environ.get('SLURM_JOB_NODELIST', '')}")
    logger.warning(f"Slurm job id: {int(os.environ.get('SLURM_JOB_ID', -1))}")


@torch.no_grad()
def clip_grad_norm_(
    fsdp_module: nn.Module,
    max_norm: Union[float, int],
    norm_type: Union[float, int] = 2.0,
) -> float:
    """
    DUPLICATED FROM FAIRSCALE, ADDING THE REDUCTION FOR MODEL PARALLEL

    Clip all gradients at this point in time. The norm is computed over all
    gradients together, as if they were concatenated into a single vector.
    Gradients are modified in-place.

    Args:
        fsdp_module (nn.Module): FSDP model
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'``
            for infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    return fsdp_module.clip_grad_norm_(max_norm, norm_type)


def get_model_state(model, full_state: bool = False, cpu_offload: bool = True) -> Dict[str, Any]:
    state_dict_options = StateDictOptions(
        full_state_dict=full_state,
        cpu_offload=cpu_offload,
    )
    return get_model_state_dict(model, options=state_dict_options)


def set_model_state(model, model_state_dict, full_state: bool = False, cpu_offload: bool = True):
    state_dict_options = StateDictOptions(
        full_state_dict=full_state,
        cpu_offload=cpu_offload,
    )
    set_model_state_dict(model, model_state_dict, options=state_dict_options)


def get_sharded_optimizer_state(model, optimizer, dcp: bool) -> Dict[str, Any]:
    if dcp:
        state_dict_options = StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
        )
        return get_optimizer_state_dict(model, optimizer, options=state_dict_options)
    else:
        return optimizer.state_dict()


def set_sharded_optimizer_state(model, optimizer, optimizer_state_dict, dcp: bool):
    if dcp:
        state_dict_options = StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
        )
        set_optimizer_state_dict(model, optimizer, optimizer_state_dict, options=state_dict_options)
    else:
        optimizer.load_state_dict(optimizer_state_dict)


def collect_provenance(cfg) -> dict:
    def git_info(path: Path) -> dict | None:
        def git(*args):
            result = subprocess.run(
                ["git", "-C", str(path), *args],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            return result.stdout.strip() if result.returncode == 0 else None

        root = git("rev-parse", "--show-toplevel")
        if root is None:
            return None

        status = git("status", "--porcelain")
        return {
            "root": root,
            "commit": git("rev-parse", "HEAD"),
            "branch": git("symbolic-ref", "--short", "-q", "HEAD"),
            "remote_origin": git("config", "--get", "remote.origin.url"),
            "dirty": None if status is None else bool(status),
        }
    xllm_module = Path(xllm.__file__).resolve()

    return {
        "command": {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "python": sys.executable,
        },
        "xllm": {
            "module_path": str(xllm_module),
            "git": git_info(xllm_module.parent),
        },
        "paths": {
            # cfg.data include dataset paths and weights
            "training_data_spec": cfg.data,
            "tokenizer": cfg.tokenizer.path,
            "configured_base_model_dir": cfg.base_model_dir,
            "output_checkpoint_root": str(
                Path(cfg.dump_dir).resolve() / "checkpoints"
            ),
        },
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_RESTART_COUNT",
                "SLURM_SUBMIT_DIR",
            )
        },
    }