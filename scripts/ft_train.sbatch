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
export FT_SLOT_DIR=${FT_SLOT_DIR:-/dev/shm/xllm_ft}
export FT_LEADER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export FT_STORE_PORT=${FT_STORE_PORT:-$((20000 + SLURM_JOB_ID % 20000))}
export FT_MAIN_JOB_ID=$SLURM_JOB_ID
export FT_TRAIN_ARGS="$TRAIN_ARGS --dump_dir $DUMP_DIR --ft.enabled true --ft.spares $FT_SPARES --ft.slot_dir $FT_SLOT_DIR"
RESTART=${SLURM_RESTART_COUNT:-0}
export FT_MAIN_RESTART=$RESTART                             # a refill job submitted before a requeue must not join the next run
MARKER=$DUMP_DIR/ft_finished.$SLURM_JOB_ID.$RESTART
MAILBOX=$DUMP_DIR/ft_slurm/$SLURM_JOB_ID
mkdir -p "$DUMP_DIR/restarts" "$MAILBOX"

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
bash scripts/ft_slurm_helper.sh "$MAILBOX" "${FT_SPARE_SCRIPT:-scripts/ft_spare.sbatch}" &

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
