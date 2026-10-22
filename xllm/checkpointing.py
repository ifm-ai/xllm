import abc
import copy
import os
import re
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path
import torch
import torch.distributed.checkpoint as dist_ckpt
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint.staging import BlockingAsyncStager
from torch.distributed.checkpoint.utils import _DistWrapper

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


def _is_cpu_state_dict(state_dict: Any) -> bool:
    """Whether every tensor reachable from ``state_dict`` already lives on the CPU."""
    if isinstance(state_dict, torch.Tensor):
        return state_dict.device.type == "cpu"
    if isinstance(state_dict, dict):
        return all(_is_cpu_state_dict(v) for v in state_dict.values())
    if isinstance(state_dict, (list, tuple)):
        return all(_is_cpu_state_dict(v) for v in state_dict)
    return True


def _stage_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return a training-safe CPU snapshot of ``state_dict``.

    This is the only part of an asynchronous save that blocks the training thread.
    ``BlockingAsyncStager.stage`` (staging.py:305-313) copies every tensor to host
    memory and returns only once the copy is complete, so later in-place updates to the
    live parameters cannot leak into the checkpoint.

    State dicts that are already on the CPU are not copied again in full: xLLM builds the
    DCP ones with ``cpu_offload=True`` (utils.py:347-352, 363-369), so their sharded
    tensors are fresh host copies, and at ~3T parameters a second full copy would be tens
    of TB of host RAM.  What such a state dict still shares with the live training state
    is copied: tensors that already lived on the CPU (Adam's per-parameter ``step``) and
    the plain Python containers (``param_groups``).  Otherwise training would keep
    updating them during the background write and the checkpoint would mix steps.
    """
    if _is_cpu_state_dict(state_dict):
        return _detach_from_live_state(state_dict)
    return BlockingAsyncStager(cache_staged_state_dict=False).stage(state_dict)


def _detach_from_live_state(obj: Any) -> Any:
    """Copy everything except DTensor shards (already private host copies)."""
    if isinstance(obj, DTensor):
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _detach_from_live_state(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_detach_from_live_state(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_from_live_state(v) for v in obj)
    return copy.deepcopy(obj)


class _AsyncDCPSave:
    """One DCP save, split at the seams of ``_save_state_dict`` (state_dict_saver.py:415-519).

    ``dcp.save`` runs three steps: a plan ``reduce_scatter`` (:487), the local
    ``write_data`` file I/O (:494-503) and a write ``all_reduce`` (:513) whose reduce
    function calls ``FileSystemWriter.finish`` to write ``.metadata`` (:506-510,
    filesystem.py:758-790).  ``dcp.async_save`` cannot be used here: it asserts a CPU
    backend on the process group (:302-305) and runs those collectives on a background
    thread.  Instead the three steps are driven explicitly -- ``plan()`` on the training
    thread, ``write()`` on the background thread, ``finalize()`` back on the training
    thread -- so no collective ever runs in the background.

    Every collective still goes through ``_DistWrapper`` on ``hsdp_group`` with
    ``use_collectives=True``, so the plan (and therefore the ``__{i}_`` shard prefixes
    from ``prepare_global_plan``, filesystem.py:671-677) and the single ``.metadata``
    are byte-identical to the synchronous path, keeping resume unchanged.
    """

    def __init__(
        self,
        state_dict: Dict[str, Any],
        checkpoint_id: Path,
        process_group: ProcessGroup,
    ):
        self.state_dict = state_dict
        self.planner = DefaultSavePlanner()
        # ``per_thread_copy_ahead=0`` selects ``_SerialCpuLoader`` instead of the
        # CUDA-stream based ``_OverlappingCpuLoader`` (filesystem.py:397-412).  The state
        # dict is already on the CPU, so the copy-ahead buffer is useless, and this keeps
        # the background thread free of any CUDA stream work.  FileSystemWriter.stage
        # makes the same choice for dcp.async_save (filesystem.py:1025-1030).
        self.writer = FileSystemWriter(checkpoint_id, per_thread_copy_ahead=0)
        self.distW = _DistWrapper(process_group, True, 0)
        self.central_plan = None
        self.global_metadata = None

    def plan(self) -> None:
        """Phase 1: the plan collective.  Metadata only, so it stays on the main thread."""

        def local_step():
            self.planner.set_up_planner(
                state_dict=self.state_dict,
                storage_meta=self.writer.storage_meta(),
                is_coordinator=self.distW.is_coordinator,
            )
            self.writer.set_up_storage_writer(
                self.distW.is_coordinator,
                rank=self.distW.rank,
                use_collectives=True,
            )
            local_plan = self.planner.create_local_plan()
            return self.writer.prepare_local_plan(local_plan)

        def global_step(all_local_plans):
            all_local_plans, self.global_metadata = self.planner.create_global_plan(all_local_plans)
            return self.writer.prepare_global_plan(all_local_plans)

        self.central_plan = self.distW.reduce_scatter("plan", local_step, global_step)

    def write(self) -> List[Any]:
        """Phase 2: the file I/O.  Runs on the background thread, no collective inside."""
        final_local_plan = self.planner.finish_plan(self.central_plan)
        all_writes = self.writer.write_data(final_local_plan, self.planner)
        all_writes.wait()
        return all_writes.value()

    def finalize(self, write_results: List[Any]) -> None:
        """Phase 3: gather the write results and write ``.metadata``.  Main thread only."""

        def finish_checkpoint(all_results):
            assert self.global_metadata is not None
            self.writer.finish(metadata=self.global_metadata, results=all_results)
            return self.global_metadata

        self.distW.all_reduce("write", lambda: write_results, finish_checkpoint)


class _PendingSave:
    """A checkpoint whose file writes are in flight on the background thread."""

    def __init__(
        self,
        future: Future,
        model_save: _AsyncDCPSave,
        optim_save: Optional[_AsyncDCPSave],
        checkpoint_dir: Path,
        training_checkpoint_dir: Path,
        dump_dir: Path,
        step: int,
        global_rank: int,
        world_size: int,
        keep_last: int,
        keep_checkpoint_every_step: int,
        cfg: Config,
    ):
        self.future = future
        self.model_save = model_save
        self.optim_save = optim_save
        self.checkpoint_dir = checkpoint_dir
        self.training_checkpoint_dir = training_checkpoint_dir
        self.dump_dir = dump_dir
        self.step = step
        self.global_rank = global_rank
        self.world_size = world_size
        self.keep_last = keep_last
        self.keep_checkpoint_every_step = keep_checkpoint_every_step
        self.cfg = cfg

    def check_ok(self) -> None:
        """Non-collective: surface a failed background write.  Safe to call every step."""
        if self.future.done():
            self.future.result()  # re-raises the writer's exception, if any

    def finalize(self) -> None:
        """Complete the checkpoint.  Collective, so only at points all ranks reach."""
        results = self.future.result()
        self.model_save.finalize(results[0])
        if self.optim_save is not None:
            self.optim_save.finalize(results[1])

        # The training state is written last by the background thread, so its presence
        # implies this rank's model and optimizer shards are already durable.  This
        # barrier is what lets rank 0 rely on that for every rank (same as the
        # synchronous path, checkpointing.py:211).
        torch.distributed.barrier()

        if self.global_rank != 0:
            return

        all_training_states: List[Tuple[int, Path]] = [
            (r, self.training_checkpoint_dir / training_state_name(rank=r))
            for r in range(self.world_size)
        ]
        wait_for_all_files_to_be_in_dir(
            root_dir=self.training_checkpoint_dir,
            filename_to_rank={p.name: r for r, p in all_training_states},
            log_interval=LOG_INTERVAL,
            sleep_interval=10,
            timeout=300,
        )

        # The completion marker, written last: get_latest_checkpoint_path keys on it
        # (checkpointing.py:317-318), so a half-written checkpoint is never picked up.
        with (self.checkpoint_dir / "config.json").open("w") as fp:
            fp.write(self.cfg.to_json())
        logger.info(f"Final checkpointing dir is done: {self.checkpoint_dir}")

        # Only now that the checkpoint is complete may old ones be removed.
        _delete_stale_checkpoints(
            dump_dir_=self.dump_dir,
            checkpoint_dir=self.checkpoint_dir,
            global_rank=self.global_rank,
            keep_last=self.keep_last,
            keep_checkpoint_every_step=self.keep_checkpoint_every_step,
        )


class AsyncCheckpointer(Checkpointer):
    """Checkpointer that blocks training only while staging the state to host memory.

    The DCP planning collectives and the device->host staging run on the training
    thread, the file I/O runs on a single background thread, and the DCP write
    collectives run on the training thread at the next aligned point (next save,
    pre-eval or ``close``).  No collective -- NCCL or otherwise -- ever runs on the
    background thread, so the process groups stay NCCL-only and training is never raced.
    """

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
        # A single worker bounds host memory to one staged checkpoint and guarantees at
        # most one save is in flight.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="async-checkpoint")
        self._pending: Optional[_PendingSave] = None

    def save_latest_checkpoint(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        training_state: Dict[str, Any],
        step: int,
    ):
        # Complete the previous save first: keeps at most one save in flight and runs its
        # pending write collectives on this (training) thread.
        self.wait_for_all()

        tp_rank = get_model_parallel_rank()
        dump_dir_ = Path(self.dump_dir)
        checkpoint_dir = get_checkpoint_dir(dump_dir_, step)
        model_path = checkpoint_dir / sharded_model_dir(tp_rank)
        optim_path = checkpoint_dir / sharded_optim_dir(tp_rank)
        training_checkpoint_dir = checkpoint_dir / training_state_dir()

        logger.info("All ranks to begin checkpointing...")

        # Directory creation and the barrier are identical to the synchronous path
        # (checkpointing.py:164-191) so every rank sees the same tree before writing.
        if self.global_rank == 0:
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
            sleep_interval=10,
            timeout=300,
        )

        if torch.distributed.get_rank(self.hsdp_group) == 0:
            optim_path.mkdir(exist_ok=False, parents=False)

        wait_for_folder_to_exists(
            [optim_path],
            log_interval=LOG_INTERVAL,
            sleep_interval=10,
            timeout=300,
        )

        torch.distributed.barrier()
        logger.info("All ranks ready for checkpointing.")

        # ---- the blocking part: snapshot the state on host memory ----
        model_state = _stage_state_dict(model_state)
        optimizer_state = _stage_state_dict(optimizer_state)

        # ---- main-thread collective: DCP planning (metadata only) ----
        model_save = _AsyncDCPSave(model_state, model_path, self.hsdp_group)
        model_save.plan()

        optim_save: Optional[_AsyncDCPSave] = None
        optim_file: Optional[Path] = None
        master_rank = -1
        if self.cfg.dcp_for_optimizer:
            optim_save = _AsyncDCPSave(optimizer_state, optim_path, self.hsdp_group)
            optim_save.plan()
        else:
            master_rank = get_master_rank_in_replicated_group(
                self.global_rank, self.cfg.fully_sharded_size
            )
            optim_file = optim_path / sharded_optim_name(master_rank)

        training_state_path = training_checkpoint_dir / training_state_name(self.global_rank)

        def write_all() -> List[List[Any]]:
            """Background thread: file I/O only, no collective."""
            results = [model_save.write()]
            if optim_save is not None:
                results.append(optim_save.write())
            elif self.global_rank == master_rank:
                torch.save(optimizer_state, optim_file)
            # Written last, so its presence implies this rank's shards are durable.
            torch.save(training_state, training_state_path)
            return results

        logger.info(f"Staged checkpoint (step {step}), writing in background...")
        self._pending = _PendingSave(
            future=self._executor.submit(write_all),
            model_save=model_save,
            optim_save=optim_save,
            checkpoint_dir=checkpoint_dir,
            training_checkpoint_dir=training_checkpoint_dir,
            dump_dir=dump_dir_,
            step=step,
            global_rank=self.global_rank,
            world_size=self.world_size,
            keep_last=self.keep_last,
            keep_checkpoint_every_step=self.keep_checkpoint_every_step,
            cfg=self.cfg,
        )

    def check_ok(self):
        if self._pending is not None:
            self._pending.check_ok()

    def wait_for_all(self):
        if self._pending is None:
            return
        pending, self._pending = self._pending, None
        pending.finalize()

    def close(self):
        self.wait_for_all()
        self._executor.shutdown(wait=True)


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

    _delete_stale_checkpoints(
        dump_dir_=dump_dir_,
        checkpoint_dir=checkpoint_dir,
        global_rank=global_rank,
        keep_last=keep_last,
        keep_checkpoint_every_step=keep_checkpoint_every_step,
    )


def _delete_stale_checkpoints(
    dump_dir_: Path,
    checkpoint_dir: Path,
    global_rank: int,
    keep_last: int,
    keep_checkpoint_every_step: int,
) -> None:
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
