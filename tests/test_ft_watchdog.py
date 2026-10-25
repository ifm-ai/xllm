import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

WATCHDOG = Path(__file__).parent.parent / "scripts" / "ft_watchdog.sh"


def stubs(tmp_path, nodelist="node[0-1]", batch_host="node0", state="ALLOCATED", restarts=0, job_state="RUNNING",
          end="Unknown"):
    """scontrol/scancel/srun stubs that log every call; returns (bin dir, call log)."""
    bin_dir, log = tmp_path / "bin", tmp_path / "calls"
    bin_dir.mkdir(exist_ok=True)
    hosts = {"node[0-1]": "node0\nnode1", "node1": "node1"}[nodelist]
    (bin_dir / "scontrol").write_text(f"""#!/bin/bash
case "$1 $2" in
  "show job") echo "JobId=77 JobState={job_state} Restarts={restarts} BatchHost={batch_host} NodeList={nodelist} \
EndTime={end}" ;;
  "show hostnames") printf '{hosts}\\n' ;;
  "show node") echo "NodeName=$3 State={state}" ;;
  *) echo "$*" >> {log} ;;
esac
""")
    (bin_dir / "scancel").write_text(f"#!/bin/bash\necho scancel $* >> {log}\n")
    (bin_dir / "srun").write_text(f"#!/bin/bash\necho srun >> {log}\n")  # the cleanup of the RAM copies
    for tool in ("scontrol", "scancel", "srun"):
        (bin_dir / tool).chmod(0o755)
    return bin_dir, log


def run(tmp_path, bin_dir, restart=0, max_requeues=3, lease_ttl=0.1):
    """lease_ttl=0.1: a leader that never renews counts as silent at once."""
    mbox = tmp_path / "ft_slurm" / "77"
    mbox.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", SLURM_JOB_ID="77", FT_WATCHDOG_POLL="0.1",
               SLURM_RESTART_COUNT=str(restart), FT_MAX_REQUEUES=str(max_requeues), FT_LEASE_TTL=str(lease_ttl))
    marker, lease = tmp_path / "ft_finished.77.0", tmp_path / "ft_leader" / "77.0"
    return subprocess.Popen(["bash", str(WATCHDOG), str(mbox), str(marker), str(lease)], env=env)


def calls(log):
    return log.read_text().splitlines() if log.exists() else []


def test_batch_host_dropped_and_leader_silent_requeues_once_and_cancels_refills(tmp_path):
    bin_dir, log = stubs(tmp_path, nodelist="node1")
    (tmp_path / "ft_slurm" / "77").mkdir(parents=True)
    (tmp_path / "ft_slurm" / "77" / "submitted.501").write_text("")
    assert run(tmp_path, bin_dir).wait(20) == 0
    assert calls(log) == ["scancel 501", "srun", "requeue 77"]


def test_batch_host_gone_while_a_leader_renews_is_left_alone_until_it_stops(tmp_path):
    bin_dir, log = stubs(tmp_path, nodelist="node1")
    lease = tmp_path / "ft_leader" / "77.0"
    lease.mkdir(parents=True)
    (lease / "leader").write_text('{"term": 1, "host": "node1", "port": 22001, "phys": 1, "standby": 2}')
    stopped = threading.Event()

    def renew():
        n = 0
        while not stopped.wait(0.2):
            n += 1
            (lease / "renew.1").write_text(str(n))

    writer = threading.Thread(target=renew, daemon=True)
    writer.start()
    proc = run(tmp_path, bin_dir, lease_ttl=0.5)  # silent for 3 s: requeue
    with pytest.raises(subprocess.TimeoutExpired):
        proc.wait(5)
    assert calls(log) == []
    stopped.set()
    assert proc.wait(20) == 0 and calls(log) == ["srun", "requeue 77"]


def test_batch_host_gone_near_the_time_limit_requeues(tmp_path):
    import time

    end = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + 100))  # within the 900 s margin
    bin_dir, log = stubs(tmp_path, nodelist="node1", end=end)
    assert run(tmp_path, bin_dir, lease_ttl=600).wait(20) == 0
    assert calls(log) == ["srun", "requeue 77"]


@pytest.mark.parametrize("state", ["DOWN", "DOWN+DRAIN", "IDLE*+FAIL", "ALLOCATED+NOT_RESPONDING"])
def test_batch_host_down_counts_as_gone(tmp_path, state):
    bin_dir, log = stubs(tmp_path, state=state)
    assert run(tmp_path, bin_dir).wait(20) == 0
    assert calls(log) == ["srun", "requeue 77"]


@pytest.mark.parametrize("kw", [dict(restarts=1), dict(job_state="COMPLETING"), dict(job_state="PENDING")])
def test_ends_by_itself_once_the_run_is_over_or_requeued(tmp_path, kw):
    bin_dir, log = stubs(tmp_path, nodelist="node1", **kw)
    assert run(tmp_path, bin_dir).wait(5) == 0 and calls(log) == []


# the last: the batch host is gone, but the takeover window is not over and the time limit is unknown
@pytest.mark.parametrize("kw,lease_ttl", [(dict(state="ALLOCATED+DRAIN"), 0.1), (dict(state="MIXED"), 0.1),
                                          (dict(nodelist="node1"), 600)])
def test_no_action_while_the_batch_host_is_alive_or_a_leader_could_still_take_over(tmp_path, kw, lease_ttl):
    bin_dir, log = stubs(tmp_path, **kw)
    proc = run(tmp_path, bin_dir, lease_ttl=lease_ttl)
    with pytest.raises(subprocess.TimeoutExpired):
        proc.wait(1.5)
    (tmp_path / "ft_slurm" / "77" / "stop").write_text("")  # the batch script's cleanup at the end of the run
    assert proc.wait(5) == 0 and calls(log) == []


def test_ends_by_itself_once_slurm_forgot_the_job(tmp_path):
    bin_dir, log = stubs(tmp_path)
    gone_job = "#!/bin/bash\necho 'slurm_load_jobs error: Invalid job id specified' >&2\nexit 1\n"
    (bin_dir / "scontrol").write_text(gone_job)
    assert run(tmp_path, bin_dir).wait(5) == 0 and calls(log) == []


def test_finished_run_on_a_live_batch_host_is_left_to_the_batch_script(tmp_path):
    bin_dir, log = stubs(tmp_path)
    (tmp_path / "ft_finished.77.0").write_text("finished\n")
    assert run(tmp_path, bin_dir).wait(5) == 0 and calls(log) == []


def test_finished_run_without_its_batch_host_is_cancelled_never_requeued(tmp_path):
    bin_dir, log = stubs(tmp_path, nodelist="node1")
    (tmp_path / "ft_finished.77.0").write_text("finished\n")
    assert run(tmp_path, bin_dir).wait(5) == 0 and calls(log) == ["scancel 77"]


def test_requeues_used_up_cancels_the_job(tmp_path):
    bin_dir, log = stubs(tmp_path, nodelist="node1", restarts=3)
    assert run(tmp_path, bin_dir, restart=3, max_requeues=3).wait(20) == 0
    assert calls(log) == ["srun", "scancel 77"]


def test_two_watchdogs_requeue_exactly_once(tmp_path):
    bin_dir, log = stubs(tmp_path, nodelist="node1")
    procs = [run(tmp_path, bin_dir) for _ in range(4)]
    assert all(p.wait(20) == 0 for p in procs)
    assert calls(log) == ["srun", "requeue 77"]


def test_a_failed_requeue_is_retried_on_a_later_poll(tmp_path):
    bin_dir, log = stubs(tmp_path, nodelist="node1")
    real = bin_dir / "scontrol_real"
    (bin_dir / "scontrol").rename(real)
    (bin_dir / "scontrol").write_text(f"""#!/bin/bash
if [ "$1" = requeue ] && [ ! -e {tmp_path}/failed_once ]; then touch {tmp_path}/failed_once; exit 1; fi
exec {real} "$@"
""")
    (bin_dir / "scontrol").chmod(0o755)
    assert run(tmp_path, bin_dir).wait(20) == 0
    assert calls(log) == ["srun", "srun", "requeue 77"]


def test_an_empty_node_list_is_unknown_not_gone(tmp_path):
    bin_dir, log = stubs(tmp_path)
    script = (bin_dir / "scontrol").read_text()
    assert "printf 'node0\nnode1\\n'" in script
    (bin_dir / "scontrol").write_text(script.replace("printf 'node0\nnode1\\n'", "true"))
    proc = run(tmp_path, bin_dir)
    with pytest.raises(subprocess.TimeoutExpired):
        proc.wait(1.5)
    (tmp_path / "ft_slurm" / "77" / "stop").write_text("")
    assert proc.wait(5) == 0 and calls(log) == []


def test_terms_skipped_as_never_led_do_not_reset_the_silence(tmp_path):
    bin_dir, log = stubs(tmp_path, nodelist="node1")
    lease = tmp_path / "ft_leader" / "77.0"
    lease.mkdir(parents=True)
    stopped = threading.Event()

    def skip_terms():  # a new term every 0.3 s, none of them ever renewed
        term = 0
        while not stopped.wait(0.3):
            term += 1
            (lease / "leader").write_text(f'{{"term": {term}, "host": "", "port": 0, "phys": 0, "standby": -1}}')

    writer = threading.Thread(target=skip_terms, daemon=True)
    writer.start()
    try:
        assert run(tmp_path, bin_dir, lease_ttl=0.5).wait(20) == 0  # silent for 3 s in all: requeue
    finally:
        stopped.set()
    assert calls(log) == ["srun", "requeue 77"]


def helper_setup(tmp_path):
    """A fake helper (records its start, then waits) and the saved job environment it is started with."""
    xllm = tmp_path / "xllm"
    (xllm / "scripts").mkdir(parents=True)
    started = tmp_path / "helper_started"
    (xllm / "scripts" / "ft_slurm_helper.sh").write_text(
        f'echo "$1 $2 $FT_TRAIN_ARGS" >> {started}; echo $$ > {tmp_path}/helper.pid\n'
        'n=0; while true; do n=$((n + 1)); echo "h $$ $n" > "$1/helper.tick"; sleep 0.2; done\n')
    mbox = tmp_path / "ft_slurm" / "77"
    mbox.mkdir(parents=True)
    (mbox / "helper.env.0").write_text(f'declare -x XLLM_DIR="{xllm}"\ndeclare -x FT_TRAIN_ARGS="--ft.enabled true"\n')
    return mbox, started


def test_one_watchdog_takes_over_the_silent_slurm_helper_of_a_dead_batch_host(tmp_path, monkeypatch):
    monkeypatch.setenv("FT_HELPER_SILENT", "3")  # the watchdog's clock counts whole seconds
    bin_dir, log = stubs(tmp_path, nodelist="node1")
    mbox, started = helper_setup(tmp_path)
    procs = [run(tmp_path, bin_dir, lease_ttl=1) for _ in range(2)]  # the leader is silent: requeue after 6 s
    assert all(p.wait(30) == 0 for p in procs)
    assert started.read_text().splitlines() == [f"{mbox} scripts/ft_spare.sbatch --ft.enabled true"]
    pid = int((tmp_path / "helper.pid").read_text())
    time.sleep(0.5)
    with pytest.raises(ProcessLookupError):  # the helper stops with its watchdog
        os.kill(pid, 0)


def test_no_takeover_while_a_helper_still_ticks(tmp_path, monkeypatch):
    monkeypatch.setenv("FT_HELPER_SILENT", "2")
    bin_dir, log = stubs(tmp_path, nodelist="node1")
    mbox, started = helper_setup(tmp_path)
    stopped = threading.Event()

    def tick():  # the batch host's helper is alive (its host only looks DOWN)
        n = 0
        while not stopped.wait(0.2):
            n += 1
            (mbox / "helper.tick").write_text(f"node0 1 {n}")

    threading.Thread(target=tick, daemon=True).start()
    try:
        assert run(tmp_path, bin_dir, lease_ttl=0.5).wait(20) == 0
    finally:
        stopped.set()
    assert not started.exists()
