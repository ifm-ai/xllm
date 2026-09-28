import abc
import os
import re
import shutil
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path
import torch
import torch.distributed.checkpoint as dist_ckpt
from torch.distributed import ProcessGroup

from xllm.config import Config
from xllm.distributed import (
    get_model_parallel_rank,
    get_hybrid_shard_data_parallel_group,
)
from xllm.utils import (
    wait_for_folder_to_exists,
    wait_for_all_files_to_be_in_dir,
    get_master_rank_in_replicated_group,
)

PREFIX = "checkpoint_"
CHECKPOINT_FOLDER_NAME = "checkpoints"
regex = re.compile(rf"{PREFIX}(?P<step>\d+)")
LOG_INTERVAL = 60

logger = getLogger()


class Checkpointer(abc.ABC):
    @abc.abstractmethod
    def save_latest_checkpoint(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        training_state: Dict[str, Any],
        step: int,
    ):
        pass

    @abc.abstractmethod
    def wait_for_all(self):
        pass

    @abc.abstractmethod
    def check_ok(self):
        pass

    @abc.abstractmethod
    def close(self):
        pass


class SyncCheckpointer(Checkpointer):
    def __init__(
        self,
        dump_dir: str,
        cfg: Config,
        global_rank: int,
        world_size: int,
        keep_last: int = -1,
        keep_checkpoint_every_step: int = -1,
    ):
        self.dump_dir = dump_dir
        self.cfg = cfg
        self.global_rank = global_rank
        self.world_size = world_size
        self.keep_last = keep_last
        self.keep_checkpoint_every_step = keep_checkpoint_every_step
        self.hsdp_group = get_hybrid_shard_data_parallel_group()

    def save_latest_checkpoint(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        training_state: Dict[str, Any],
        step: int,
    ):
        save_latest_checkpoint(
            model_state=model_state,
            optimizer_state=optimizer_state,
            training_state=training_state,
            dump_dir=self.dump_dir,
            global_rank=self.global_rank,
            world_size=self.world_size,
            step=step,
            hsdp_group = self.hsdp_group,
            keep_last=self.keep_last,
            keep_checkpoint_every_step=self.keep_checkpoint_every_step,
            cfg=self.cfg,
        )

    def check_ok(self):
        pass

    def wait_for_all(self):
        return

    def close(self):
        pass


class AsyncCheckpointer(Checkpointer):
    def __init__(
        self,
        dump_dir: str,
        cfg: Config,
        global_rank: int,
        world_size: int,
        keep_last: int = -1,
        keep_checkpoint_every_step: int = -1,
        timeout: int = 1800,
    ):
        self.timeout = timeout
        self.dump_dir = dump_dir
        self.cfg = cfg
        self.global_rank = global_rank
        self.world_size = world_size
        self.keep_last = keep_last
        self.keep_checkpoint_every_step = keep_checkpoint_every_step
        self.hsdp_group = get_hybrid_shard_data_parallel_group()

    def save_latest_checkpoint(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        training_state: Dict[str, Any],
        step: int,
    ):
        raise NotImplementedError


def save_latest_checkpoint(
    model_state: Dict[str, Any],
    optimizer_state: Dict[str, Any],
    training_state: Dict[str, Any],
    dump_dir: str,
    global_rank: int,
    world_size: int,
    step: int,
    hsdp_group: ProcessGroup,
    cfg: Config,
    keep_last: int = -1,
    keep_checkpoint_every_step: int = -1,
    sleep_interval: float = 10,
    timeout_folder_exists: float = 300,  # 5min
    timeout_all_shard_exists: float = 300,  # 5min
):

    assert keep_last >= 1 or keep_last == -1
    assert world_size > global_rank >= 0
    tp_rank = get_model_parallel_rank()

    dump_dir_ = Path(dump_dir)
    checkpoint_dir = get_checkpoint_dir(dump_dir_, step)
    model_path = checkpoint_dir / sharded_model_dir(tp_rank)
    optim_path = checkpoint_dir / sharded_optim_dir(tp_rank)
    training_checkpoint_dir = checkpoint_dir / training_state_dir()

    logger.info("All ranks to begin checkpointing...")

    if global_rank == 0:
        if checkpoint_dir.exists():
            raise RuntimeError(
                f"Checkpoint {checkpoint_dir} already exists, this should not happen!"
            )
        checkpoint_dir.mkdir(exist_ok=False, parents=True)
        training_checkpoint_dir.mkdir(exist_ok=False, parents=False)
        logger.info(f"Created {checkpoint_dir}")

    wait_for_folder_to_exists(
        [training_checkpoint_dir],
        log_interval=LOG_INTERVAL,
        sleep_interval=sleep_interval,
        timeout=timeout_folder_exists,
    )

    if torch.distributed.get_rank(hsdp_group) == 0:
        optim_path.mkdir(exist_ok=False, parents=False)

    wait_for_folder_to_exists(
        [optim_path],
        log_interval=LOG_INTERVAL,
        sleep_interval=sleep_interval,
        timeout=timeout_folder_exists,
    )

    torch.distributed.barrier()
    logger.info("All ranks ready for checkpointing.")

    assert checkpoint_dir.exists(), f"Should not happens {checkpoint_dir}"
    logger.info(f"checkpoint dir exists: {checkpoint_dir}")
    logger.info(f"Checkpointing (step {step}) in {checkpoint_dir} ...")

    dist_ckpt.save(model_state, checkpoint_id=model_path, process_group=hsdp_group)
    if cfg.dcp_for_optimizer:
        dist_ckpt.save(optimizer_state, checkpoint_id=optim_path, process_group=hsdp_group)
    else:
        master_rank = get_master_rank_in_replicated_group(global_rank, cfg.fully_sharded_size)
        optim_path = optim_path / sharded_optim_name(master_rank)
        if global_rank == master_rank:
            torch.save(optimizer_state, optim_path)

    training_state_path = training_checkpoint_dir / training_state_name(global_rank)
    torch.save(training_state, training_state_path)
    logger.info("Checkpoint done.")

    logger.info("Waiting for all ranks to finish checkpointing...")
    torch.distributed.barrier()
    logger.info("Checkpointing done for all gpus.")

    if global_rank == 0:
        all_training_states: List[Tuple[int, Path]] = []
        for this_rank in range(world_size):
            all_training_states.append((this_rank, training_checkpoint_dir / training_state_name(rank=this_rank)))

        wait_for_all_files_to_be_in_dir(
            root_dir=training_checkpoint_dir,
            filename_to_rank={p.name: r for r, p in all_training_states},
            log_interval=LOG_INTERVAL,
            sleep_interval=sleep_interval,
            timeout=timeout_all_shard_exists,
        )

        # dumping config for async eval
        with (checkpoint_dir / "config.json").open("w") as fp:
            fp.write(cfg.to_json())

        logger.info(f"Final checkpointing dir is done: {checkpoint_dir}")

    # remove stale checkpoints (master rank only)
    if global_rank != 0:
        return None

    # keep all previous checkpoints
    if keep_last < 0:
        return None

    sorted_checkpoints = _get_sorted_checkpoints(ckpt_folder=dump_dir_ / CHECKPOINT_FOLDER_NAME)
    assert len(sorted_checkpoints) >= 1, sorted_checkpoints
    assert sorted_checkpoints[-1] == checkpoint_dir
    assert keep_last >= 1

    to_keep = set(sorted_checkpoints[-keep_last:])
    assert checkpoint_dir in to_keep, (checkpoint_dir, to_keep)

    if keep_checkpoint_every_step > -1:
        to_keep.update(
            {p for p in sorted_checkpoints if ckpt_path_to_step(p) % keep_checkpoint_every_step == 0}
        )

    to_delete = [c for c in sorted_checkpoints if c not in to_keep]
    logger.info(
        f"Going to delete {len(to_delete)} past checkpoints with steps: "
        f"{[ckpt_path_to_step(p) for p in to_delete]}"
    )
    for to_delete_checkpoint_path in to_delete:
        logger.info(f"Deleting: {to_delete_checkpoint_path} ...")
        shutil.rmtree(to_delete_checkpoint_path)

    logger.info(f"Deleted {len(to_delete)} past checkpoints.")


def get_latest_checkpoint_paths_for_sharded_states(
    dump_dir: str, global_rank: int, fully_sharded_size: Optional[int], dcp_for_optimizer: bool
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    latest_checkpoint = get_latest_checkpoint_path(dump_dir, global_rank, strict=True)
    if latest_checkpoint is None:
        return None, None, None

    latest_checkpoint_dir = Path(latest_checkpoint)
    tp_rank = get_model_parallel_rank()
    model_path = latest_checkpoint_dir / sharded_model_dir(tp_rank)
    optim_path = latest_checkpoint_dir / sharded_optim_dir(tp_rank)
    training_state_path = latest_checkpoint_dir / training_state_dir() / training_state_name(global_rank)

    assert model_path.exists(), f"Checkpoint for model shard doesn't exists: {model_path}"
    assert optim_path.exists(), f"Checkpoint for optim shard doesn't exists: {optim_path}"
    assert training_state_path.exists(), f"Checkpoint for training state doesn't exists: {training_state_path}"
    if not dcp_for_optimizer:
        master_rank = get_master_rank_in_replicated_group(global_rank, fully_sharded_size)
        optim_path = optim_path / sharded_optim_name(master_rank)

    return model_path, optim_path, training_state_path


def get_base_model_checkpoint_path(
    base_model_dir: str, global_rank: int, fully_sharded_size: Optional[int], dcp_for_optimizer: bool
) -> Tuple[Path, Optional[Path]]:
    tp_rank = get_model_parallel_rank()
    model_path = Path(base_model_dir) / sharded_model_dir(tp_rank)
    if not model_path.exists():
        logger.info("Sharded base model doesn't exists, loading from full checkpoint state...")
        model_path = Path(base_model_dir) / full_model_dir(tp_rank)
    assert model_path.exists(), f"Checkpoint for based model doesn't exists: {model_path}"

    optim_path = Path(base_model_dir) / sharded_optim_dir(tp_rank)
    if not optim_path.exists():
        optim_path = None
    elif not dcp_for_optimizer:
        master_rank = get_master_rank_in_replicated_group(global_rank, fully_sharded_size)
        optim_path = optim_path / sharded_optim_name(master_rank)

    return model_path, optim_path


def get_latest_checkpoint_path(dump_dir: str, global_rank: int, strict: bool) -> Optional[str]:
    dump_dir_ = Path(dump_dir)
    checkpoint_folder = dump_dir_ / CHECKPOINT_FOLDER_NAME
    if not checkpoint_folder.exists():
        logger.info(f"Checkpoint folder {checkpoint_folder} doesn't exists!")
        return None
    sorted_checkpoints = _get_sorted_checkpoints(checkpoint_folder)
    for checkpoint_dir in reversed(sorted_checkpoints):
        config_path = checkpoint_dir / "config.json"
        if config_path.exists():
            return str(checkpoint_dir)
        elif strict:
            raise RuntimeError(
                f"Last checkpoint {checkpoint_dir} is broken!"
            )
        elif global_rank == 0:
            logger.info(f"Deleting broken checkpoint: {checkpoint_dir} ...")
            shutil.rmtree(checkpoint_dir)
            logger.info("Deleted.")
    logger.info(f"No checkpoints in {checkpoint_folder}")
    return None


def get_checkpoint_dir(dump_dir_: Path, step: int) -> Path:
    return dump_dir_ / CHECKPOINT_FOLDER_NAME / f"{PREFIX}{step:08d}"


def ckpt_path_to_step(p: Path) -> int:
    m = regex.match(p.name)
    assert m is not None, p.name
    return int(m.group("step"))


def _get_sorted_checkpoints(ckpt_folder: Path):
    sorted_checkpoints = sorted(ckpt_folder.glob(f"{PREFIX}*"), key=ckpt_path_to_step)
    return sorted_checkpoints


def sharded_model_dir(rank: int) -> str:
    return f"sharded_model.tp{rank:02d}"


def full_model_dir(rank: int) -> str:
    return f"full_model.tp{rank:02d}"


def sharded_optim_dir(rank: int) -> str:
    return f"optimizer.tp{rank:02d}"


def sharded_optim_name(rank: int) -> str:
    return f"optim{rank:06d}.pt"


def training_state_dir() -> str:
    return f"training_state"


def training_state_name(rank: int) -> str:
    return f"training_state.{rank:06d}.pth"


def get_last_restart_step(dump_dir: str, global_rank: int) -> int:
    last_ckpt = get_latest_checkpoint_path(dump_dir, global_rank, strict=False)
    if last_ckpt is None:
        return 0
    return ckpt_path_to_step(Path(last_ckpt))


def make_checkpointer(
    dump_dir: str,
    cfg: Config,
    global_rank: int,
    world_size: int,
    keep_last: int = -1,
    keep_checkpoint_every_step: int = -1,
    async_checkpointing: bool = False,
) -> Checkpointer:
    the_class = AsyncCheckpointer if async_checkpointing else SyncCheckpointer
    return the_class(
        dump_dir=dump_dir,
        global_rank=global_rank,
        world_size=world_size,
        keep_last=keep_last,
        keep_checkpoint_every_step=keep_checkpoint_every_step,
        cfg=cfg,
    )
