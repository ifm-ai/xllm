#!/usr/bin/env bash
# Batch-host watchdog for an xLLM ft job. scripts/ft_train.sbatch starts it over ssh on one or two nodes other than
# the batch host, outside the container and outside the job's steps: every step's srun client runs on the batch
# host, and its tasks on all nodes die with it, so a watchdog in a job step could never act. With sbatch --no-kill
# a dead batch host (node 0) is dropped from the job, and the job can stay RUNNING with no batch script until its
# time limit. When SLURM shows the batch host gone (not in the job's node list any more, or DOWN/FAILed) on two
# polls in a row and the run did not finish, exactly one watchdog (a link() claim on the shared FS) cancels the
# refill jobs and requeues the job, or cancels it when the requeues are used up. As it is not part of the job, it
# ends by itself as soon as the job is over or was requeued. Files are read by open() only: on NFS, mtime and
# directory listings can be stale.
# Usage: ft_watchdog.sh <mailbox dir> <finished marker>; env: SLURM_JOB_ID, FT_MAX_REQUEUES, FT_WATCHDOG_POLL.
set -u
MBOX=$1
MARKER=$2
JOB=$SLURM_JOB_ID
RESTART=${SLURM_RESTART_COUNT:-0}
POLL=${FT_WATCHDOG_POLL:-30}
CLAIM=$(dirname "$MBOX")/.requeue.$JOB.$RESTART

done_here() { cat "$MARKER" >/dev/null 2>&1 || cat "$MBOX/stop" >/dev/null 2>&1; }

field() { # $1 = name, from `scontrol show job` output in $info
  tr ' ' '\n' <<<"$info" | sed -n "s/^$1=//p" | head -1
}

batch_host_gone() {  # 0 = gone, 1 = not gone or unknown, 2 = this run of the job is over
  if ! info=$(scontrol show job "$JOB" 2>&1); then
    grep -qi "invalid job id" <<<"$info" && return 2  # purged after its end
    [ -n "${told:-}" ] || { told=1; say "cannot reach SLURM (not acting while so): ${info:0:200}"; }
    return 1  # SLURM unreachable: never a reason to act
  fi
  [ "$(field JobState)" = RUNNING ] || return 2
  [ "$(field Restarts)" = "$RESTART" ] || return 2  # requeued already
  local host nodes state
  host=$(field BatchHost)
  nodes=$(scontrol show hostnames "$(field NodeList)" 2>/dev/null) || return 1
  [ -n "$host" ] || return 1
  grep -qx "$host" <<<"$nodes" || return 0  # dropped from the job
  state=$(scontrol show node "$host" 2>/dev/null | tr ' ' '\n' | sed -n 's/^State=//p' | head -1)
  case $state in  # DRAIN alone is not gone: the node still runs its processes
    *DOWN* | *FAIL* | *NOT_RESPONDING* | *POWER*) return 0 ;;
  esac
  return 1
}

say() { echo "ft watchdog $(hostname) $(date +%T): $*" >&2; }
say "watching job $JOB (run $RESTART)"
gone=0
while ! done_here; do
  batch_host_gone
  case $? in
    0) gone=$((gone + 1)) ;;
    2) say "the run is over or was requeued; exiting"; exit 0 ;;
    *) gone=0 ;;
  esac
  if [ "$gone" -ge 2 ]; then
    mine=$CLAIM.$(hostname).$$
    echo "$(hostname)" > "$mine"
    if ln "$mine" "$CLAIM" 2>/dev/null; then  # the first watchdog wins; the others stand down
      say "the batch host of job $JOB is gone and the run did not finish"
      for f in "$MBOX"/submitted.*; do [ -e "$f" ] && scancel "${f##*/submitted.}"; done
      if [ "$RESTART" -lt "${FT_MAX_REQUEUES:-3}" ]; then
        scontrol requeue "$JOB"
      else
        say "requeues used up; cancelling job $JOB"
        scancel "$JOB"
      fi
    fi
    rm -f "$mine"
    exit 0
  fi
  sleep "$POLL"
done
say "the run finished or the job script stopped; exiting"
