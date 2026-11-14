import json
import pickle

import torch

from xllm.ft import dumper
from xllm.ft.replica import dump_files, run_epoch, slot_path, write_slot


def make_slots(slot_dir, ranks, step, slot=0):
    slot_dir.mkdir(parents=True, exist_ok=True)
    for r in ranks:
        meta = {"step": step, "epoch": run_epoch(), "rank": r, "nbytes": 8, "training_state": {}}
        write_slot(slot_path(str(slot_dir), "own", r, slot), torch.full((8,), r, dtype=torch.uint8), pickle.dumps(meta))


def test_dump_write_commit_and_retention(tmp_path, monkeypatch):
    monkeypatch.setenv("FT_EPOCH", "11:0")
    slots, root = tmp_path / "shm", tmp_path / "ft_checkpoints"
    make_slots(slots, [0, 1], step=20)
    assert dumper.newest_common_own_step(str(slots), [0, 1]) == 20
    for step in (20, 30, 40):
        make_slots(slots, [0, 1], step=step)
        staging = dumper.staging_dir(root, gen=0, step=step)
        assert all(dumper.write_rank(str(slots), staging, r, 0, step) for r in (0, 1))
        assert dumper.commit(root, staging, 0, step, world=2, keep=2)
    assert [s for s, _ in dumper.committed_dumps(root, world=2)] == [40, 30]  # oldest dropped
    assert dumper.committed_dumps(root, world=4) == []  # another world size never matches
    payload, _, ok = dump_files(root / "checkpoints" / "checkpoint_00000040", 1)
    assert payload.read_bytes() == bytes([1] * 8) and json.loads(ok.read_text())["step"] == 40


def test_incomplete_dump_is_never_committed(tmp_path, monkeypatch):
    monkeypatch.setenv("FT_EPOCH", "11:0")
    slots, root = tmp_path / "shm", tmp_path / "ft_checkpoints"
    make_slots(slots, [0, 1], step=20)
    staging = dumper.staging_dir(root, gen=0, step=20)
    assert dumper.write_rank(str(slots), staging, 0, 0, 20)  # rank 1 never wrote its part
    assert not dumper.commit(root, staging, 0, 20, world=2, keep=2)
    assert dumper.committed_dumps(root, 2) == []
    dumper.clean_staging(root)
    assert not staging.exists()


def test_a_slot_that_moved_on_is_not_dumped(tmp_path, monkeypatch):
    monkeypatch.setenv("FT_EPOCH", "11:0")
    slots, root = tmp_path / "shm", tmp_path / "ft_checkpoints"
    make_slots(slots, [0], step=20)
    make_slots(slots, [0], step=30)  # same slot overwritten by a newer copy
    assert not dumper.write_rank(str(slots), dumper.staging_dir(root, 0, 20), 0, 0, 20)


def test_newest_xllm_checkpoint(tmp_path):
    for step, complete in ((10, True), (20, False)):
        d = tmp_path / "checkpoints" / f"checkpoint_{step:08d}"
        d.mkdir(parents=True)
        if complete:
            (d / "config.json").write_text("{}")
    assert dumper.newest_xllm_checkpoint_step(str(tmp_path)) == 10


def test_dump_leases_the_slot_while_reading_it(tmp_path, monkeypatch):
    import os
    import time

    from xllm.ft.replica import lease_path, leased

    monkeypatch.setenv("FT_EPOCH", "11:0")
    slots, root = tmp_path / "shm", tmp_path / "ft_checkpoints"
    make_slots(slots, [0], step=20)
    path = slot_path(str(slots), "own", 0, 0)
    seen = []
    real = dumper._write_rank
    monkeypatch.setattr(dumper, "_write_rank", lambda *a: seen.append(leased(path)) or real(*a))
    assert dumper.write_rank(str(slots), dumper.staging_dir(root, 0, 20), 0, 0, 20)
    assert seen == [True] and not leased(path)  # held while reading, released after
    lease_path(path).touch()
    old = time.time() - 3600
    os.utime(lease_path(path), (old, old))
    assert not leased(path)  # left by a dumper that died: ignored


def test_retention_never_deletes_the_dump_in_use(tmp_path, monkeypatch):
    monkeypatch.setenv("FT_EPOCH", "11:0")
    slots, root = tmp_path / "shm", tmp_path / "ft_checkpoints"
    in_use = str(root / "checkpoints" / "checkpoint_00000020")
    for step in (20, 30, 40):
        make_slots(slots, [0], step=step)
        staging = dumper.staging_dir(root, 0, step)
        dumper.write_rank(str(slots), staging, 0, 0, step)
        dumper.commit(root, staging, 0, step, world=1, keep=1, protect=in_use)
    assert sorted(s for s, _ in dumper.committed_dumps(root, 1)) == [20, 40]  # 30 dropped, 20 kept while loaded


def test_bad_dumps_survive_a_requeue(tmp_path, monkeypatch):
    monkeypatch.setenv("FT_EPOCH", "11:0")
    slots, root = tmp_path / "shm", tmp_path / "ft_checkpoints"
    make_slots(slots, [0], step=20)
    staging = dumper.staging_dir(root, 0, 20)
    dumper.write_rank(str(slots), staging, 0, 0, 20)
    dumper.commit(root, staging, 0, 20, world=1, keep=2)
    dumper.mark_bad(root, root / "checkpoints" / "checkpoint_00000020")
    assert dumper.bad_dumps(root) == {("checkpoint_00000020", 20)}  # read back from the file, not the store


def test_torn_or_malformed_dump_files_are_skipped(tmp_path):
    from xllm.ft import dumper

    (tmp_path / ".bad").write_text("checkpoint_00000010 10\ncheckpoint_000000")  # the writer died mid-line
    assert dumper.bad_dumps(tmp_path) == {("checkpoint_00000010", 10)}
    for name, config in (("checkpoint_00000020", '{"step": 20, "world": 16}'), ("checkpoint_00000030", '{"st')):
        (tmp_path / "checkpoints" / name).mkdir(parents=True)
        (tmp_path / "checkpoints" / name / "config.json").write_text(config)
    assert [s for s, _ in dumper.committed_dumps(tmp_path, 16)] == [20]
    (tmp_path / "checkpoints" / "checkpoint_weird").mkdir()
    (tmp_path / "checkpoints" / "checkpoint_weird" / "config.json").write_text("{}")
    assert dumper.newest_xllm_checkpoint_step(str(tmp_path)) == 30


def test_dump_kill_aborts_the_round_and_the_next_dump_works(tmp_path, monkeypatch):
    import threading

    import pytest

    monkeypatch.setenv("FT_EPOCH", "11:0")
    slots, root = tmp_path / "shm", tmp_path / "ft_checkpoints"
    make_slots(slots, [0, 1], step=20)
    abort = threading.Event()
    abort.set()
    staging = dumper.staging_dir(root, gen=0, step=20)
    with pytest.raises(OSError):
        dumper.write_rank(str(slots), staging, 0, 0, 20, abort)
    assert not abort.is_set() and not dumper.commit(root, staging, 0, 20, world=2, keep=2)
    assert all(dumper.write_rank(str(slots), staging, r, 0, 20, abort) for r in (0, 1))
    assert dumper.commit(root, staging, 0, 20, world=2, keep=2)


def test_torn_write_and_bitflip_injections(tmp_path, monkeypatch):
    import types

    from xllm.ft.agent import Agent
    from xllm.ft.replica import valid_steps

    monkeypatch.setenv("FT_EPOCH", "11:0")
    slots, root = tmp_path / "shm", tmp_path / "ft_checkpoints"
    make_slots(slots, [0], step=20, slot=0)
    make_slots(slots, [0], step=30, slot=1)
    killed = []
    agent = Agent.__new__(Agent)
    agent.conf, agent.dump_root, agent.logical, agent.gpus = types.SimpleNamespace(slot_dir=str(slots)), root, 0, 1
    agent.procs = [types.SimpleNamespace(send_signal=killed.append)]
    agent.torn_write(0)
    assert valid_steps(lambda s: slot_path(str(slots), "own", 0, s), 0) == {20: 0} and killed  # the older copy stays
    staging = dumper.staging_dir(root, gen=0, step=20)
    assert dumper.write_rank(str(slots), staging, 0, 0, 20) and dumper.commit(root, staging, 0, 20, world=1, keep=2)
    payload = dump_files(root / "checkpoints" / "checkpoint_00000020", 0)[0]
    before = payload.read_bytes()
    agent.bitflip(0)
    after = payload.read_bytes()
    assert len(before) == len(after) and sum(a != b for a, b in zip(before, after)) == 1
