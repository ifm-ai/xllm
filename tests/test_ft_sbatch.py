import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
TRAIN = REPO / "scripts" / "ft_train.sbatch"
SPARE = REPO / "scripts" / "ft_spare.sbatch"


def test_scripts_parse():
    for script in (TRAIN, SPARE, REPO / "scripts" / "ft_slurm_helper.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_train_script_invariants():
    text = TRAIN.read_text()
    assert "#SBATCH --no-kill" in text and "#SBATCH --requeue" in text
    assert "#SBATCH --signal=B:USR1" in text  # the batch shell only: trainers and agents ignore SIGUSR1
    # a requeue keeps the job id: the previous run's stop file is removed before the helper and watchdogs start
    assert text.index('rm -f "$MAILBOX/stop"') < text.index("ft_slurm_helper.sh") < text.index("ft_watchdog.sh")
    launch = text[text.index("for host in"):text.index("ft_watchdog.sh")]
    assert "srun" not in launch  # not a job step: over ssh
    # bounded, and `cd ...;` so that the backgrounded watchdog does not keep ssh (and the batch script) waiting
    assert "timeout 60 ssh -n" in launch and "|| exit 1;" in launch and "&& SLURM_JOB_ID" not in launch
    for knob in ("PATH", "SLURM_CONF", "FT_LEASE_TTL", "FT_WATCHDOG_MARGIN", "FT_WATCHDOG_POLL", "FT_SLOT_DIR",
                 "LAUNCH"):  # read remotely: an ssh session has no SLURM environment
        assert f"{knob}=" in launch
    block = text[text.index("for host in"):text.index("\nfi\n", text.index("for host in"))]
    assert '|| wait "${launches[@]}"' in block and "\n  wait\n" not in block  # never wait on the helper
    agents = next(line for line in text.splitlines() if line.startswith("srun --no-kill"))
    assert "$LAUNCH" in agents
    step = text[text.index(agents):]
    assert step.split("\n")[1].rstrip().endswith("&")  # in the background, so a USR1 interrupts the wait
    assert step.split("\n")[2].startswith("wait $!")
    assert "trap '' USR1" in step  # no second requeue after the run is over
    assert 'timeout 300 srun --overlap --ntasks-per-node=1 $LAUNCH bash -c "rm -rf $FT_SLOT_DIR/ft_' in text
    spare = SPARE.read_text()
    assert "set -u" in spare and "${LAUNCH:-}" in spare and "--wait=0" in spare


def run_train(tmp_path, agents_rc: int, restart: int = 0, marker: bool = False):
    """Runs ft_train.sbatch with srun/scontrol stubs; returns (exit code, scontrol calls)."""
    bin_dir, dump = tmp_path / "bin", tmp_path / "dump"
    bin_dir.mkdir()
    log = tmp_path / "calls"
    (bin_dir / "scontrol").write_text(f'#!/bin/bash\necho "$*" >> {log}\n[ "$1" = show ] && echo node0\nexit 0\n')
    # the agents' step exits with agents_rc (and the leader writes the marker); the cleanup step just succeeds
    marker_file = dump / f"ft_finished.77.{restart}"
    (bin_dir / "srun").write_text(
        "#!/bin/bash\ncase \"$*\" in *xllm.ft.agent*) sleep 2; "  # the script must wait for the agents' step
        + (f"touch {marker_file}; " if marker else "")
        + f"exit {agents_rc};; esac\nexit 0\n"
    )
    for tool in ("scontrol", "srun"):
        (bin_dir / tool).chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", SLURM_JOB_ID="77", SLURM_JOB_NODELIST="node0",
               SLURM_RESTART_COUNT=str(restart), DUMP_DIR=str(dump), TRAIN_ARGS="--steps 1", XLLM_DIR=str(REPO),
               FT_MAX_REQUEUES="3")
    rc = subprocess.run(["bash", str(TRAIN)], env=env, cwd=REPO, timeout=60).returncode
    calls = log.read_text().splitlines() if log.exists() else []
    return rc, [c for c in calls if not c.startswith("show")]


@pytest.mark.parametrize(
    "agents_rc, restart, marker, want_rc, want_requeue",
    [
        (0, 0, False, 0, False),  # every agent exited 0: finished, even if the marker could not be written
        (1, 0, True, 0, False),  # the marker says finished
        (75, 0, False, 0, True),  # not finished: requeue
        (75, 3, False, 1, False),  # not finished and the requeues are used up
        (78, 0, False, 1, False),  # the sizing gate refused ft: a configuration error, never requeued
    ],
)
def test_train_script_requeues_only_unfinished_runs(tmp_path, agents_rc, restart, marker, want_rc, want_requeue):
    rc, calls = run_train(tmp_path, agents_rc, restart, marker)
    assert rc == want_rc
    assert (calls == ["requeue 77"]) == want_requeue
