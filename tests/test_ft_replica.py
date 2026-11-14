import pickle

import torch

from xllm.ft.replica import meta_path, pack, read_meta, run_epoch, slot_path, unpack, valid_steps, write_slot


def test_pack_unpack_is_byte_exact_for_any_dtype_and_device():
    src = [torch.randn(5, 3), torch.randn(7).to(torch.bfloat16), torch.tensor(12.0), torch.arange(4, dtype=torch.int64)]
    buf = torch.empty(sum(t.numel() * t.element_size() for t in src), dtype=torch.uint8)
    pack(src, buf)
    dst = [torch.zeros_like(t) for t in src]
    unpack(dst, buf)
    for a, b in zip(src, dst):
        assert a.dtype == b.dtype and torch.equal(a.view(-1).view(torch.uint8), b.view(-1).view(torch.uint8))


def test_valid_steps_ignore_other_runs_ranks_and_torn_slots(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "9")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    data = torch.zeros(8, dtype=torch.uint8)
    good = {"step": 30, "epoch": run_epoch(), "rank": 3, "nbytes": 8, "training_state": {}}
    write_slot(slot_path(str(tmp_path), "own", 3, 0), data, pickle.dumps(good))
    stale = dict(good, step=40, epoch=("9", "0"))  # written before the requeue
    write_slot(slot_path(str(tmp_path), "own", 3, 1), data, pickle.dumps(stale))
    assert valid_steps(lambda s: slot_path(str(tmp_path), "own", 3, s), 3) == {30: 0}
    assert valid_steps(lambda s: slot_path(str(tmp_path), "own", 3, s), 4) == {}  # not this rank's state
    meta_path(slot_path(str(tmp_path), "own", 3, 0)).unlink()  # a copy cut off halfway has no metadata
    assert valid_steps(lambda s: slot_path(str(tmp_path), "own", 3, s), 3) == {}


def test_write_slot_publishes_metadata_last(tmp_path):
    path = tmp_path / "ft_x_own_r000000_s0"
    write_slot(path, torch.arange(16, dtype=torch.uint8), pickle.dumps({"step": 1}))
    assert read_meta(path) == {"step": 1}
    assert path.read_bytes() == bytes(range(16))
    assert not meta_path(path).with_suffix(".tmp").exists()


def test_unpack_into_non_contiguous_tensors():
    src = [torch.randn(4, 6)]
    buf = torch.empty(4 * 6 * 4, dtype=torch.uint8)
    pack(src, buf)
    dst = torch.zeros(6, 4).t()  # non-contiguous view
    unpack([dst], buf)
    assert torch.equal(dst, src[0])


def test_a_failed_copy_is_logged_and_does_not_stop_later_copies():
    from xllm.ft.replica import Replica

    replica = Replica.__new__(Replica)
    replica.busy = True

    def full(*args):
        raise OSError(28, "No space left on device")

    replica._write_slots = full
    replica._write(0, pickle.dumps({"step": 7}), [], None, None, 0.0)
    assert replica.busy is False


def test_a_forced_copy_stays_pending_while_the_slot_is_busy(monkeypatch):
    import types

    import xllm.ft.replica as rep

    replica = rep.Replica.__new__(rep.Replica)
    replica.gpu_peak = types.SimpleNamespace(copies_allowed=True)
    replica.k, replica.force_copy, replica.busy, replica.restored_slot = 10, True, True, 0
    replica.slot_dir, replica.rank, replica.group = "/nonexistent", 0, None
    started = []
    replica._start_copy = lambda step, state, slot: started.append((step, slot))
    monkeypatch.setattr(rep.torch, "tensor", lambda v, device=None: types.SimpleNamespace(item=lambda: v[0]))
    monkeypatch.setattr(rep.dist, "all_reduce", lambda t, op=None, group=None: None)
    replica.maybe_copy(3, dict)
    assert started == [] and replica.force_copy  # busy: deferred, not dropped
    replica.busy = False
    replica.maybe_copy(4, dict)
    assert started == [(4, 1)] and not replica.force_copy  # copied at the next step, into the other slot


def run_ranks(candidates):
    """agree_step on every rank at once, with an all-reduce MIN over threads."""
    import threading

    from xllm.ft.replica import agree_step

    n, values, results = len(candidates), [None] * len(candidates), [None] * len(candidates)
    barrier = threading.Barrier(n)

    def rank(i):
        def reduce_min(v):
            values[i] = v
            barrier.wait()
            out = min(values)
            barrier.wait()
            return out

        results[i] = agree_step(candidates[i], reduce_min)

    threads = [threading.Thread(target=rank, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    return results


def test_ranks_agree_on_the_newest_step_they_all_hold():
    assert run_ranks([{10, 20}, {10, 20}]) == [20, 20]
    assert run_ranks([{10, 30}, {10, 15, 20}]) == [10, 10]  # a failed copy write: the min offer (20) is not held by all
    assert run_ranks([{20, 30}, None]) == [30, 30]  # a replaced node takes the step its holder has
    assert run_ranks([{30}, {20}]) == [-1, -1]  # nothing in common: the agent restarts from disk


def test_a_full_slot_dir_fails_the_copy_with_an_error_and_leaves_the_slot_invalid(tmp_path, monkeypatch):
    import errno

    import pytest

    path = slot_path(str(tmp_path), "own", 3, 0)
    data = torch.arange(16, dtype=torch.uint8)
    write_slot(path, data, pickle.dumps({"step": 10, "rank": 3, "nbytes": 16}))

    def full(fd, offset, length):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("xllm.ft.replica.os.posix_fallocate", full)
    with pytest.raises(OSError):  # an error the copy thread catches, not a SIGBUS inside the copy
        write_slot(path, torch.arange(32, dtype=torch.uint8), pickle.dumps({"step": 20, "rank": 3, "nbytes": 32}))
    assert not meta_path(path).exists()  # the old copy is invalid: never read half-written
