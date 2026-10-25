import os
import subprocess
import time
from pathlib import Path

import pytest

from xllm.ft.scheduler import SlurmMailbox

HELPER = Path(__file__).parent.parent / "scripts" / "ft_slurm_helper.sh"


@pytest.fixture
def helper(tmp_path):
    """The real helper script with fake sbatch/scancel that log their calls."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls"
    (bin_dir / "sbatch").write_text(f'#!/bin/bash\necho "sbatch $*" >> {log}\necho "5000$RANDOM;cluster"\n')
    (bin_dir / "scancel").write_text(f'#!/bin/bash\necho "scancel $*" >> {log}\n[ "$1" != bad ]\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    mbox = tmp_path / "mbox"
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", FT_HELPER_POLL="0.1")
    proc = subprocess.Popen(["bash", str(HELPER), str(mbox), "/path/spare.sbatch"], env=env)
    time.sleep(0.3)
    yield SlurmMailbox(mbox, timeout=10), log, mbox, proc
    proc.terminate()
    proc.wait(timeout=10)


def test_request_and_release_round_trip(helper):
    client, log, _, _ = helper
    job = client.request_node(exclude=["dgx-1", "dgx-2"])
    assert job.startswith("5000")
    client.release_node(job)
    calls = log.read_text().splitlines()
    assert calls[0] == "sbatch --parsable --exclude=dgx-1,dgx-2 /path/spare.sbatch"
    assert calls[1] == f"scancel {job}"


def test_errors_are_reported(helper):
    client, *_ = helper
    with pytest.raises(RuntimeError, match="scancel bad failed"):
        client.release_node("bad")


def test_submitted_jobs_are_cancelled_when_the_job_ends(helper):
    client, log, mbox, proc = helper
    a = client.request_node()
    b = client.request_node()
    (mbox / "stop").touch()
    proc.wait(timeout=10)
    calls = log.read_text().splitlines()
    assert f"scancel {a}" in calls and f"scancel {b}" in calls


def test_sbatch_timeout_reported(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sbatch").write_text('#!/bin/bash\nsleep 2\necho "5000;cluster"\n')
    (bin_dir / "scancel").write_text('#!/bin/bash\nexit 0\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    mbox = tmp_path / "mbox"
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", FT_HELPER_POLL="0.05", FT_SBATCH_TIMEOUT="0.2")
    proc = subprocess.Popen(["bash", str(HELPER), str(mbox), "/path/spare.sbatch"], env=env)
    time.sleep(0.3)
    try:
        client = SlurmMailbox(mbox, timeout=5)
        t0 = time.monotonic()
        with pytest.raises(RuntimeError, match="SLURM helper: error"):
            client.request_node()
        elapsed = time.monotonic() - t0
        assert elapsed < 1.5, f"Expected timeout < 1.5s, took {elapsed:.2f}s"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_orphaned_taken_request_readopted(tmp_path):
    import socket
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls"
    (bin_dir / "sbatch").write_text(f'#!/bin/bash\necho "sbatch $*" >> {log}\necho "5001;cluster"\n')
    (bin_dir / "scancel").write_text(f'#!/bin/bash\necho "scancel $*" >> {log}\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    mbox = tmp_path / "mbox"
    mbox.mkdir()
    host = socket.gethostname()
    dead_pid = 999999
    orphan = mbox / f"req.orphan1.taken.{host}.{dead_pid}"
    orphan.write_text("request_node -\n")
    os.utime(orphan, (time.time() - 10, time.time() - 10))

    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", FT_HELPER_POLL="0.05")
    proc = subprocess.Popen(["bash", str(HELPER), str(mbox), "/path/spare.sbatch"], env=env)
    time.sleep(0.5)
    try:
        rep = mbox / "rep.orphan1"
        for _ in range(30):
            if rep.exists():
                break
            time.sleep(0.1)
        assert rep.exists(), "Orphaned request was not processed"
        assert rep.read_text().strip() == "ok 5001"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


