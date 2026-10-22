"""Tests for asynchronous checkpointing (xllm.checkpointing.AsyncCheckpointer).

Run on a single GPU:

    pytest tests/test_async_checkpoint.py -s

The test initialises a 1-rank NCCL process group -- the same backend xLLM uses in
production (xllm/distributed/slurm.py) -- and never creates a Gloo group, so it also
checks that the object collectives used by DCP planning work on NCCL.
"""
import datetime
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn

from xllm.checkpointing import (
    AsyncCheckpointer,
    SyncCheckpointer,
    _stage_state_dict,
    get_checkpoint_dir,
    get_latest_checkpoint_path,
    sharded_model_dir,
    sharded_optim_dir,
)
from xllm.config import (
    DataLoaderConfig,
    ModelConf,
    OptimConf,
    SlurmConf,
    TokenizerConf,
    TrainerConf,
    ValidConf,
)
from xllm.distributed import (
    destroy_model_parallel,
    get_hybrid_shard_data_parallel_group,
    get_model_parallel_rank,
    initialize_model_parallel,
)
from xllm.utils import (
    get_model_state,
    get_sharded_optimizer_state,
    set_model_state,
    set_sharded_optimizer_state,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="async checkpointing needs CUDA/NCCL"
)

TIMEOUT = 1800


def _make_cfg(dump_dir: str, dcp_for_optimizer: bool = True) -> TrainerConf:
    """A real TrainerConf: the checkpointer reads cfg.dcp_for_optimizer,
    cfg.fully_sharded_size and writes cfg.to_json() into config.json."""
    return TrainerConf(
        slurm=SlurmConf(),
        model=ModelConf(),
        optim=OptimConf(),
        valid=ValidConf(),
        tokenizer=TokenizerConf(),
        dataloader=DataLoaderConfig(),
        dump_dir=dump_dir,
        dcp_for_optimizer=dcp_for_optimizer,
    )


def _build_model_and_optimizer():
    torch.manual_seed(0)
    model = nn.Linear(8, 4).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # One step so Adam materialises its exp_avg / exp_avg_sq state.
    loss = model(torch.randn(2, 8, device="cuda")).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def _snapshot(model, optimizer):
    """The (model, optimizer) states exactly as train.py builds them (train.py:618-622)."""
    return (
        get_model_state(model, full_state=False),
        get_sharded_optimizer_state(model, optimizer, dcp=True),
    )


def _assert_nested_close(actual, expected, path="root"):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), f"{path}: expected a tensor"
        assert actual.shape == expected.shape, f"{path}: {actual.shape} != {expected.shape}"
        try:
            torch.testing.assert_close(actual.cpu(), expected.cpu())
        except AssertionError as e:
            raise AssertionError(f"Mismatch at {path}: {e}") from e
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected a dict"
        assert set(actual) == set(expected), f"{path}: {set(actual)} != {set(expected)}"
        for key in expected:
            _assert_nested_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)), f"{path}: expected a sequence"
        assert len(actual) == len(expected), f"{path}: {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_nested_close(a, e, f"{path}[{i}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def _relative_files(root: Path):
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}


def _check_dcp_layout(shard_dir: Path):
    """The use_collectives=True layout: one .metadata and __{rank}_ shards.

    This is what the synchronous path produces (filesystem.py:671-677, 767-771);
    any drift here would break resume.
    """
    files = sorted(p.name for p in shard_dir.iterdir() if p.is_file())
    assert ".metadata" in files, files
    assert sum(name == ".metadata" for name in files) == 1, files
    shards = [name for name in files if name != ".metadata"]
    assert shards, files
    assert all(name.startswith("__0_") for name in shards), shards


@pytest.fixture(scope="module")
def distributed():
    """1-rank NCCL process group + xLLM parallel groups, torn down afterwards."""
    torch.cuda.set_device(0)
    with tempfile.TemporaryDirectory() as store_dir:
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{store_dir}/store",
            rank=0,
            world_size=1,
            timeout=datetime.timedelta(seconds=TIMEOUT),
        )
        initialize_model_parallel(1, 1, TIMEOUT)
        try:
            yield
        finally:
            destroy_model_parallel()
            dist.destroy_process_group()


def test_async_checkpoint_matches_sync(distributed):
    with tempfile.TemporaryDirectory() as dump_dir:
        cfg = _make_cfg(dump_dir)
        model, optimizer = _build_model_and_optimizer()
        model_state, optimizer_state = _snapshot(model, optimizer)
        training_state = {"step": 1, "tokens": 128}

        checkpointer = AsyncCheckpointer(
            dump_dir=dump_dir,
            cfg=cfg,
            global_rank=0,
            world_size=1,
        )

        # ---- asynchronous save: returns after staging, writes in the background ----
        checkpointer.save_latest_checkpoint(
            model_state=model_state,
            optimizer_state=optimizer_state,
            training_state=training_state,
            step=1,
        )

        # The completion marker is only written by wait_for_all() (checkpointing.py:282),
        # so a half-written checkpoint can never be picked up.  Note: we deliberately do
        # not call get_latest_checkpoint_path() here -- with strict=False and
        # global_rank == 0 it *deletes* the checkpoint it considers broken
        # (checkpointing.py:324-327), which would remove the in-flight save.
        async_dir = get_checkpoint_dir(Path(dump_dir), 1)
        assert not (async_dir / "config.json").exists()

        # ---- flush: runs the write collectives on this thread and writes config.json ----
        checkpointer.wait_for_all()
        checkpointer.close()

        assert (async_dir / "config.json").exists()
        assert get_latest_checkpoint_path(dump_dir, 0, strict=True) == str(async_dir)

        # ---- synchronous save of the very same state, for comparison ----
        sync_checkpointer = SyncCheckpointer(
            dump_dir=dump_dir,
            cfg=cfg,
            global_rank=0,
            world_size=1,
        )
        sync_checkpointer.save_latest_checkpoint(
            model_state=model_state,
            optimizer_state=optimizer_state,
            training_state=training_state,
            step=2,
        )
        sync_dir = get_checkpoint_dir(Path(dump_dir), 2)

        # ---- same on-disk layout (not just the same bytes) ----
        assert _relative_files(async_dir) == _relative_files(sync_dir)
        _check_dcp_layout(async_dir / sharded_model_dir(get_model_parallel_rank()))
        _check_dcp_layout(async_dir / sharded_optim_dir(get_model_parallel_rank()))
        _check_dcp_layout(sync_dir / sharded_model_dir(get_model_parallel_rank()))
        _check_dcp_layout(sync_dir / sharded_optim_dir(get_model_parallel_rank()))

        # ---- resume: both checkpoints load back into the state they were saved from ----
        hsdp_group = get_hybrid_shard_data_parallel_group()
        for checkpoint_dir in (async_dir, sync_dir):
            resumed_model, resumed_optimizer = _build_model_and_optimizer()
            # Perturb, so a no-op load cannot pass the assertions below.
            with torch.no_grad():
                for param in resumed_model.parameters():
                    param.add_(1.0)

            resumed_model_state = get_model_state(resumed_model, full_state=False)
            dcp.load(
                resumed_model_state,
                checkpoint_id=checkpoint_dir / sharded_model_dir(get_model_parallel_rank()),
                process_group=hsdp_group,
            )
            set_model_state(resumed_model, resumed_model_state)

            resumed_optimizer_state = get_sharded_optimizer_state(
                resumed_model, resumed_optimizer, dcp=True
            )
            dcp.load(
                resumed_optimizer_state,
                checkpoint_id=checkpoint_dir / sharded_optim_dir(get_model_parallel_rank()),
                process_group=hsdp_group,
            )
            set_sharded_optimizer_state(
                resumed_model, resumed_optimizer, resumed_optimizer_state, dcp=True
            )

            _assert_nested_close(
                get_model_state(resumed_model, full_state=False), model_state, "model"
            )
            _assert_nested_close(
                get_sharded_optimizer_state(resumed_model, resumed_optimizer, dcp=True),
                optimizer_state,
                "optimizer",
            )


def test_stage_state_dict_copies_only_device_tensors():
    """_stage_state_dict is the only part of an async save that blocks training."""
    # Already on the CPU: DTensor shards of a cpu_offload=True DCP state dict are kept
    # (they are private host copies), but plain CPU tensors such as Adam's `step` are
    # shared with the live optimizer, so they are copied and later updates don't leak in.
    cpu_state = {"step": torch.tensor(4.0), "nested": {"b": torch.ones(2)}, "lr": [1e-3]}
    staged_cpu = _stage_state_dict(cpu_state)
    cpu_state["step"].add_(1.0)
    cpu_state["nested"]["b"].add_(1.0)
    cpu_state["lr"][0] = 5e-4
    assert float(staged_cpu["step"]) == 4.0
    torch.testing.assert_close(staged_cpu["nested"]["b"], torch.ones(2))
    assert staged_cpu["lr"] == [1e-3]

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device to stage from")

    gpu_state = {"w": torch.arange(4.0, device="cuda")}
    staged = _stage_state_dict(gpu_state)
    assert staged is not gpu_state
    assert staged["w"].device.type == "cpu"
    _assert_nested_close(staged, gpu_state, "staged")

    # The snapshot must be independent of later in-place updates to the live tensors:
    # this is what makes it safe to keep training while the write is in flight.
    gpu_state["w"].add_(100.0)
    torch.testing.assert_close(staged["w"], torch.arange(4.0))

    # Nested state dicts (the shape of optimizer.state_dict()) must also land on the CPU.
    gpu_nested = {"state": {0: {"exp_avg": torch.ones(2, device="cuda")}}}
    staged_nested = _stage_state_dict(gpu_nested)
    assert staged_nested["state"][0]["exp_avg"].device.type == "cpu"


def test_training_during_the_write_does_not_change_the_checkpoint(distributed):
    """The checkpoint holds the state at save time even if training continues while
    the files are written: Adam's ``step`` is a live CPU tensor that the DCP state dict
    shares with the optimizer, so it must be copied when staging."""
    with tempfile.TemporaryDirectory() as dump_dir:
        cfg = _make_cfg(dump_dir)
        model, optimizer = _build_model_and_optimizer()
        model_state, optimizer_state = _snapshot(model, optimizer)
        expected = {k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in next(iter(optimizer_state["state"].values())).items()}
        checkpointer = AsyncCheckpointer(dump_dir=dump_dir, cfg=cfg, global_rank=0, world_size=1)
        checkpointer.save_latest_checkpoint(
            model_state=model_state, optimizer_state=optimizer_state,
            training_state={"step": 1}, step=1,
        )
        for _ in range(3):                      # keep training before the save is finalized
            model(torch.randn(2, 8, device="cuda")).sum().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        checkpointer.close()

        resumed_model, resumed_optimizer = _build_model_and_optimizer()
        state = get_sharded_optimizer_state(resumed_model, resumed_optimizer, dcp=True)
        dcp.load(state, checkpoint_id=get_checkpoint_dir(Path(dump_dir), 1)
                 / sharded_optim_dir(get_model_parallel_rank()),
                 process_group=get_hybrid_shard_data_parallel_group())
        saved = next(iter(state["state"].values()))
        assert float(saved["step"]) == float(expected["step"]) == 1.0
