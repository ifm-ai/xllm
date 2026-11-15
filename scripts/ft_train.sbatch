#!/usr/bin/env bash
# xLLM training with in-band fault tolerance (ft, "hot swap").
#
# One ft agent runs per node; it starts that node's trainers, restarts all of
# them inside this allocation after a failure (from in-memory copies), and
# replaces a lost node by an idle spare. This script only decides requeues:
# the job is requeued (up to FT_MAX_REQUEUES times) when the run did not
# finish, which the leader agent marks with a file written at a clean end.
#
# Fill in the #SBATCH lines, TRAIN_ARGS and the container/env part of LAUNCH.
#SBATCH --job-name=xllm-ft
#SBATCH --nodes=17                  # training nodes + FT_SPARES spare nodes
#SBATCH --ntasks-per-node=1         # one ft agent per node
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --no-kill                   # a failed node does not end the job: a spare takes its place
#SBATCH --signal=B:USR1@600         # time limit: to the batch shell only, which requeues
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err    # slurm_logs/ must exist before sbatch

set -u
FT_SPARES=${FT_SPARES:-1}
DUMP_DIR=${DUMP_DIR:?set DUMP_DIR}
TRAIN_ARGS=${TRAIN_ARGS:?set TRAIN_ARGS (the usual train.py arguments, without --dump_dir)}
export LAUNCH=${LAUNCH:-}           # e.g. "--container-image=... --container-mounts=..." for pyxis
XLLM_DIR=${XLLM_DIR:-$SLURM_SUBMIT_DIR}   # the xLLM checkout (sbatch runs a copy of this script elsewhere)
cd "$XLLM_DIR"

export FT_MAX_REQUEUES=${FT_MAX_REQUEUES:-3}
FT_LEASE_TTL=${FT_LEASE_TTL:-60}                            # the leader's lease (s): the standby takes over after 1.5x
export FT_SLOT_DIR=${FT_SLOT_DIR:-/dev/shm/xllm_ft}
export FT_LEADER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export FT_STORE_PORT=${FT_STORE_PORT:-$((20000 + SLURM_JOB_ID % 20000))}
export FT_MAIN_JOB_ID=$SLURM_JOB_ID
export FT_TRAIN_ARGS="$TRAIN_ARGS --dump_dir $DUMP_DIR --ft.enabled true --ft.spares $FT_SPARES --ft.slot_dir $FT_SLOT_DIR --ft.lease_ttl $FT_LEASE_TTL"
RESTART=${SLURM_RESTART_COUNT:-0}
export FT_MAIN_RESTART=$RESTART                             # a refill job submitted before a requeue must not join the next run
MARKER=$DUMP_DIR/ft_finished.$SLURM_JOB_ID.$RESTART
MAILBOX=$DUMP_DIR/ft_slurm/$SLURM_JOB_ID
mkdir -p "$DUMP_DIR/restarts" "$MAILBOX"
rm -f "$MAILBOX/stop"
# the run that owns the mailbox: helpers of earlier runs exit (written whole: readers never see half of it)
echo "$RESTART" > "$MAILBOX/.run.$$" && mv -f "$MAILBOX/.run.$$" "$MAILBOX/run"                                       # a requeue keeps the job id: the last run's stop is stale

cleanup() {
  touch "$MAILBOX/stop"                                     # the helper cancels the refill jobs it submitted
  # RAM copies are never reused; same container as the trainers, so the same /dev/shm; bounded, so a requeue still fits
  timeout 300 srun --overlap --ntasks-per-node=1 $LAUNCH bash -c "rm -rf $FT_SLOT_DIR/ft_${SLURM_JOB_ID}_*"
}
requeue_or_fail() {
  trap '' USR1                                              # one requeue, even if a USR1 comes during the cleanup
  cleanup
  if [ "$RESTART" -lt "$FT_MAX_REQUEUES" ]; then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  echo "ft: the run did not finish and FT_MAX_REQUEUES=$FT_MAX_REQUEUES requeues are used up" >&2
  exit 1
}
trap requeue_or_fail USR1                                   # the handler exits, so it requeues once

export XLLM_DIR
# what a watchdog (below) needs to run the SLURM helper should this host die
# (the whole exported environment, as the helper here has it; readable by the job's owner only: it may hold tokens)
(umask 077 && export -p > "$MAILBOX/.helper.env.$$" && mv -f "$MAILBOX/.helper.env.$$" "$MAILBOX/helper.env.$RESTART")
bash scripts/ft_slurm_helper.sh "$MAILBOX" "${FT_SPARE_SCRIPT:-scripts/ft_spare.sbatch}" &
# the batch-host watchdog, over ssh on up to two other nodes: not in a job step, so it outlives this host
if [ "${FT_WATCHDOG:-1}" = 1 ]; then
  launches=()
  for host in $(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n 2,3p); do
    # `cd ...;` not `cd ... &&`: a backgrounded list would keep ssh's output open, and ssh would wait for it
    # the remote shell parses the command again: every path and value is quoted for it. An ssh session has no
    # SLURM environment (no scontrol on its PATH, no SLURM_CONF on m4): the batch shell's is passed on
    timeout 60 ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$host" "cd $(printf %q "$XLLM_DIR") || exit 1; \
      PATH=$(printf %q "$PATH") ${SLURM_CONF:+SLURM_CONF=$(printf %q "$SLURM_CONF")} \
      SLURM_JOB_ID=$SLURM_JOB_ID SLURM_RESTART_COUNT=$RESTART FT_MAX_REQUEUES=$FT_MAX_REQUEUES \
      FT_LEASE_TTL=$FT_LEASE_TTL FT_WATCHDOG_MARGIN=${FT_WATCHDOG_MARGIN:-900} FT_WATCHDOG_POLL=${FT_WATCHDOG_POLL:-30} \
      FT_SLOT_DIR=$(printf %q "$FT_SLOT_DIR") LAUNCH=$(printf %q "$LAUNCH") setsid nohup bash \
      scripts/ft_watchdog.sh $(printf '%q ' "$MAILBOX" "$MARKER" "$DUMP_DIR/ft_leader/$SLURM_JOB_ID.$RESTART") \
      > $(printf %q "$MAILBOX/watchdog.$host.$RESTART.log") 2>&1 < /dev/null &" \
      || echo "ft: could not start the batch-host watchdog on $host (ssh); a dead batch host leaves the job to its limit" >&2 &
    launches+=($!)
  done
  # the launches run side by side; never a bare wait, which would wait for the helper (it runs to the end)
  [ ${#launches[@]} -eq 0 ] || wait "${launches[@]}"
fi

# cd: containers start elsewhere. In the background, so a USR1 can interrupt the wait below.
srun --no-kill --kill-on-bad-exit=0 --wait=0 --ntasks-per-node=1 $LAUNCH --export=ALL \
  bash -c "cd $XLLM_DIR && python -u -m xllm.ft.agent train.py $FT_TRAIN_ARGS" &
wait $!                                                     # a USR1 interrupts the wait and runs the trap
rc=$?
trap '' USR1                                                # the run is over: a late USR1 must not requeue a second time

# the marker is written by the leader after every agent acknowledged; every agent exiting 0 also means the run
# finished (the marker cannot be written while the shared FS is down)
if [ -e "$MARKER" ] || [ "$rc" -eq 0 ]; then
  cleanup
  exit 0
fi
if [ "$rc" -eq 78 ]; then                                   # SIZING_EXIT: a configuration error, never requeued
  echo "ft: the sizing gate refused hot swap at startup (see the trainers' log); not requeued" >&2
  cleanup
  exit 1
fi
requeue_or_fail
