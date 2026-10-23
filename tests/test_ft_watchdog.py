import os
import subprocess
import threading
from pathlib import Path

import pytest

WATCHDOG = Path(__file__).parent.parent / "scripts" / "ft_watchdog.sh"


def stubs(tmp_path, nodelist="node[0-1]", batch_host="node0", state="ALLOCATED", restarts=0, job_state="RUNNING"):
    """scontrol/scancel stubs that log every call; returns (bin dir, call log)."""
    bin_dir, log = tmp_path / "bin", tmp_path / "calls"
    bin_dir.mkdir(exist_ok=True)
    hosts = {"node[0-1]": "node0\nnode1", "node1": "node1"}[nodelist]
    (bin_dir / "scontrol").write_text(f"""#!/bin/bash
case "$1 $2" in
  "show job") echo "JobId=77 JobState={job_state} Restarts={restarts} BatchHost={batch_host} NodeList={nodelist}" ;;
  "show hostnames") printf '{hosts}\\n' ;;
  "show node") echo "NodeName=$3 State={state}" ;;
  *) echo "$*" >> {log} ;;
esac
""")
    (bin_dir / "scancel").write_text(f"#!/bin/bash\necho scancel $* >> {log}\n")
    for tool in ("scontrol", "scancel"):
        (bin_dir / tool).chmod(0o755)
    return bin_dir, log


def run(tmp_path, bin_dir, restart=0, max_requeues=3, timeout=20):
    mbox = tmp_path / "ft_slurm" / "77"
    mbox.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", SLURM_JOB_ID="77", FT_WATCHDOG_POLL="0.1",
               SLURM_RESTART_COUNT=str(restart), FT_MAX_REQUEUES=str(max_requeues))
    return subprocess.Popen(["bash", str(WATCHDOG), str(mbox), str(tmp_path / "ft_finished.77.0")], env=env)


def calls(log):
    return log.read_text().splitlines() if log.exists() else []


def test_batch_host_dropped_from_the_job_requeues_once_and_cancels_refills(tmp_path):
    bin_dir, log = stubs(tmp_path, nodelist="node1")
    (tmp_path / "ft_slurm" / "77").mkdir(parents=True)
    (tmp_path / "ft_slurm" / "77" / "submitted.501").write_text("")
    assert run(tmp_path, bin_dir).wait(20) == 0
    assert calls(log) == ["scancel 501", "requeue 77"]


@pytest.mark.parametrize("state", ["DOWN", "DOWN+DRAIN", "IDLE*+FAIL", "ALLOCATED+NOT_RESPONDING"])
def test_batch_host_down_counts_as_gone(tmp_path, state):
    bin_dir, log = stubs(tmp_path, state=state)
    assert run(tmp_path, bin_dir).wait(20) == 0
    assert calls(log) == ["requeue 77"]


@pytest.mark.parametrize("kw", [dict(restarts=1), dict(job_state="COMPLETING"), dict(job_state="PENDING")])
def test_ends_by_itself_once_the_run_is_over_or_requeued(tmp_path, kw):
    bin_dir, log = stubs(tmp_path, nodelist="node1", **kw)
    assert run(tmp_path, bin_dir).wait(5) == 0 and calls(log) == []


@pytest.mark.parametrize("kw", [dict(state="ALLOCATED+DRAIN"), dict(state="MIXED")])
def test_no_action_while_the_batch_host_is_alive(tmp_path, kw):
    bin_dir, log = stubs(tmp_path, **kw)
    proc = run(tmp_path, bin_dir)
    with pytest.raises(subprocess.TimeoutExpired):
        proc.wait(1.5)
    (tmp_path / "ft_slurm" / "77" / "stop").write_text("")  # the batch script's cleanup at the end of the run
    assert proc.wait(5) == 0 and calls(log) == []


def test_ends_by_itself_once_slurm_forgot_the_job(tmp_path):
    bin_dir, log = stubs(tmp_path)
    gone_job = "#!/bin/bash\necho 'slurm_load_jobs error: Invalid job id specified' >&2\nexit 1\n"
    (bin_dir / "scontrol").write_text(gone_job)
    assert run(tmp_path, bin_dir).wait(5) == 0 and calls(log) == []


def test_finished_run_is_never_requeued(tmp_path):
    bin_dir, log = stubs(tmp_path, nodelist="node1")
    (tmp_path / "ft_finished.77.0").write_text("finished\n")
    assert run(tmp_path, bin_dir).wait(5) == 0 and calls(log) == []


def test_requeues_used_up_cancels_the_job(tmp_path):
    bin_dir, log = stubs(tmp_path, nodelist="node1", restarts=3)
    assert run(tmp_path, bin_dir, restart=3, max_requeues=3).wait(20) == 0
    assert calls(log) == ["scancel 77"]


def test_two_watchdogs_requeue_exactly_once(tmp_path):
    bin_dir, log = stubs(tmp_path, nodelist="node1")
    procs = [run(tmp_path, bin_dir) for _ in range(4)]
    assert all(p.wait(20) == 0 for p in procs)
    assert calls(log) == ["requeue 77"]
