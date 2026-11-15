#!/usr/bin/env bash
# Refill spare for an xLLM ft job, submitted by scripts/ft_slurm_helper.sh. Its
# agent joins the main job's control store as a spare and runs the same
# training arguments if it is mapped to a training node. It never requeues:
# when released or when the main job ends, this job just ends.
#SBATCH --job-name=xllm-ft-spare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
# FT_LEADER_ADDR, FT_STORE_PORT, FT_MAIN_JOB_ID, FT_MAIN_RESTART, FT_TRAIN_ARGS and LAUNCH come from the main job's
# environment.
set -u
export FT_REFILL=1
cd "$XLLM_DIR"
# cd: containers start elsewhere.
srun --kill-on-bad-exit=0 --wait=0 --ntasks-per-node=1 ${LAUNCH:-} --export=ALL \
  bash -c "cd $XLLM_DIR && python -u -m xllm.ft.agent train.py $FT_TRAIN_ARGS"
