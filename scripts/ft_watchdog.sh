#!/usr/bin/env bash
# Batch-host watchdog for an xLLM ft job. scripts/ft_train.sbatch starts it over ssh on one or two nodes other than
# the batch host, outside the container and outside the job's steps, so it outlives the batch host. With sbatch
# --no-kill a dead batch host (node 0) is dropped from the job and the run goes on (a standby agent takes the
# leader's place), but the job is left RUNNING with no batch script: nothing would end it after the run, requeue it
# before its time limit, or requeue it if the takeover failed. When SLURM shows the batch host gone (not in the
# job's node list any more, or DOWN/FAILed) on two polls in a row, exactly one watchdog (a link() claim on the
# shared FS) does what the batch script would have done:
#   the run finished (the marker)                 cancel the job
#   less than FT_WATCHDOG_MARGIN s to the limit   requeue it
#   the leader's lease silent for 6 lease_ttl     requeue it (no agent could take the leader's place)
# and it takes over the batch host's SLURM helper, so the new leader still gets refill spares: the settings it
# needs are in <mailbox>/helper.env.<restart>, written by the batch script at startup.
# A requeue first cancels the refill jobs and frees the RAM copies (best effort), and cancels the job instead when
# the requeues are used up. As it is not part of the job, it ends by itself as soon as the job is over or was
# requeued. Files are read by open() only: on NFS, mtime and directory listings can be stale.
# Usage: ft_watchdog.sh <mailbox dir> <finished marker> <lease dir>; env: SLURM_JOB_ID, SLURM_RESTART_COUNT,
# FT_MAX_REQUEUES, FT_LEASE_TTL, FT_WATCHDOG_POLL, FT_WATCHDOG_MARGIN, and for the cleanup FT_SLOT_DIR, LAUNCH.
set -u
MBOX=$1
MARKER=$2
LEASE=$3
JOB=$SLURM_JOB_ID
RESTART=${SLURM_RESTART_COUNT:-0}
POLL=${FT_WATCHDOG_POLL:-30}
MARGIN=${FT_WATCHDOG_MARGIN:-900}
SILENT_MAX=$(awk "BEGIN { print int(6 * ${FT_LEASE_TTL:-60}) }")
CLAIM=$(dirname "$MBOX")/.requeue.$JOB.$RESTART
HELPER_CLAIM=$(dirname "$MBOX")/.helper.$JOB.$RESTART
HELPER_SILENT=${FT_HELPER_SILENT:-60}  # seconds without a helper tick: no helper is answering requests

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
  [ -n "$host" ] && [ -n "$nodes" ] || return 1
  grep -qx "$host" <<<"$nodes" || return 0  # dropped from the job
  state=$(scontrol show node "$host" 2>/dev/null | tr ' ' '\n' | sed -n 's/^State=//p' | head -1)
  case $state in  # DRAIN alone is not gone: the node still runs its processes
    *DOWN* | *FAIL* | *NOT_RESPONDING* | *POWER*) return 0 ;;
  esac
  return 1
}

seconds_left() {  # to the job's time limit, from the last `scontrol show job`; empty when unknown
  local end
  end=$(field EndTime)
  case $end in "" | Unknown | None) return ;; esac
  end=$(date -d "$end" +%s 2>/dev/null) && echo $((end - $(date +%s)))
}

seen=start
since=$SECONDS
leader_silent() {  # sets silent: seconds since the leader's renewal counter last changed, on this host's clock
  local term counter
  term=$(sed -n 's/.*"term": *\([0-9]*\).*/\1/p' "$LEASE/leader" 2>/dev/null)
  counter=$(cat "$LEASE/renew.$term" 2>/dev/null)
  # only a leader that renews resets it: a term skipped as never led has no counter
  if [ -n "$term" ] && [ -n "$counter" ] && [ "$term:$counter" != "$seen" ]; then seen=$term:$counter; since=$SECONDS; fi
  silent=$((SECONDS - since))
}

cleanup() {  # the batch script's cleanup of the RAM copies, best effort: srun may not work from outside the job
  timeout 300 srun --jobid "$JOB" --overlap --ntasks-per-node=1 ${LAUNCH:-} \
    bash -c "rm -rf ${FT_SLOT_DIR:-/dev/shm/xllm_ft}/ft_${JOB}_*" >/dev/null 2>&1 || say "could not free the RAM copies"
}

say() { echo "ft watchdog $(hostname) $(date +%T): $*" >&2; }

act() {  # $1 = end | requeue, $2 = why; the first watchdog wins, the others stand down
  local mine=$CLAIM.$(hostname).$$
  echo "$(hostname)" > "$mine"
  if ! ln "$mine" "$CLAIM" 2>/dev/null; then
    rm -f "$mine"
    exit 0
  fi
  rm -f "$mine"
  batch_host_gone
  case $? in  # again, now that this watchdog holds the claim: the job may have ended or been requeued meanwhile
    2) exit 0 ;;
    0) say "the batch host of job $JOB is gone and $2"
       for f in "$MBOX"/submitted.*; do [ -e "$f" ] && scancel "${f##*/submitted.}"; done
       [ "$1" = end ] || cleanup
       if [ "$1" = requeue ] && [ "$RESTART" -lt "${FT_MAX_REQUEUES:-3}" ]; then
         timeout 60 scontrol requeue "$JOB" && exit 0
       else
         [ "$1" = end ] || say "requeues used up; cancelling job $JOB"
         timeout 60 scancel "$JOB" && exit 0
       fi
       say "could not $1 job $JOB; retrying" ;;
  esac
  rm -f "$CLAIM"  # nothing done: any watchdog may act on a later poll
  gone=0
}

tick_seen=start
tick_since=$SECONDS
helper_alive() {  # a helper (the batch host's or a watchdog's) is alive while its tick keeps changing
  local tick
  tick=$(cat "$MBOX/helper.tick" 2>/dev/null)
  if [ "$tick" != "$tick_seen" ]; then tick_seen=$tick; tick_since=$SECONDS; fi
  [ $((SECONDS - tick_since)) -lt "$HELPER_SILENT" ]
}
claim_n=0
helper_pid=
start_helper() {  # no helper answers (it died with its host): one watchdog runs one from here, again if needed
  [ -n "$helper_pid" ] && kill -0 "$helper_pid" 2>/dev/null && return 0
  helper_alive && return 0
  [ -r "$MBOX/helper.env.$RESTART" ] || return 0
  claim_n=$((claim_n + 1))  # one claim per takeover: a later takeover (this helper's host died) takes the next
  local mine=$HELPER_CLAIM.$(hostname).$$
  echo "$(hostname)" > "$mine"
  if ln "$mine" "$HELPER_CLAIM.$claim_n" 2>/dev/null; then
    say "taking over the SLURM helper (refill spares): no helper tick for $HELPER_SILENT s"
    (. "$MBOX/helper.env.$RESTART" 2>/dev/null; cd "$XLLM_DIR" &&
       exec bash scripts/ft_slurm_helper.sh "$MBOX" "${FT_SPARE_SCRIPT:-scripts/ft_spare.sbatch}") \
      >> "$MBOX/helper.$(hostname).$RESTART.log" 2>&1 < /dev/null &
    helper_pid=$!
    tick_since=$SECONDS
  fi
  rm -f "$mine"
}
trap '[ -z "$helper_pid" ] || kill "$helper_pid" 2>/dev/null' EXIT  # this watchdog's helper stops with it

say "watching job $JOB (run $RESTART)"
gone=0
while ! cat "$MBOX/stop" >/dev/null 2>&1; do  # the batch script's cleanup at the end of the run
  batch_host_gone
  case $? in
    0) gone=$((gone + 1)) ;;
    2) say "the run is over or was requeued; exiting"; exit 0 ;;
    *) gone=0 ;;
  esac
  leader_silent
  helper_alive >/dev/null
  finished=0
  cat "$MARKER" >/dev/null 2>&1 && finished=1
  if [ "$gone" -ge 2 ]; then
    start_helper
    left=$(seconds_left)
    if [ "$finished" = 1 ]; then
      act end "the run finished"
    elif [ -n "$left" ] && [ "$left" -lt "$MARGIN" ]; then
      act requeue "the time limit is ${left} s away"
    elif [ "$silent" -ge "$SILENT_MAX" ]; then
      act requeue "the ft leader has been silent for ${silent} s"
    fi
  elif [ "$gone" = 0 ] && [ "$finished" = 1 ]; then
    say "the run finished; the batch script ends the job"
    exit 0
  fi
  sleep "$POLL"
done
say "the job script stopped; exiting"
