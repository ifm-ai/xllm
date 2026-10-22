#!/usr/bin/env bash
#SBATCH --job-name=xllm-node-death
#SBATCH --account=moe
#SBATCH --partition=main
#SBATCH --qos=normal
#SBATCH --nodes=3
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --propagate=NONE
#SBATCH --output=/lustre/users/ahmad.chamma/xllm-completedp-jobs/node_death_%j.out
#SBATCH --error=/lustre/users/ahmad.chamma/xllm-completedp-jobs/node_death_%j.err
J=${J:-$SLURM_SUBMIT_DIR}  # folder holding these scripts (submit from it)
B=${B:-$(dirname "$J")}  # the xLLM checkout
export FT_HEAD=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
export MASTER_PORT=$((29900 + RANDOM % 50))
W=$J/node_death_work_$SLURM_JOB_ID; mkdir -p $W
srun --kill-on-bad-exit=0 --wait=0 --cpu-bind=none --container-image=/lustre/users/ahmad.chamma/models/xllm_torch12.sqsh --container-mounts=/lustre/users/ahmad.chamma:/lustre/users/ahmad.chamma --export=all bash -c "cd $B && PYTHONPATH=$B:$J/pydeps exec python $J/node_death_test.py agent $W 2 2>&1" > $J/node_death_raw_$SLURM_JOB_ID.log
rm -rf $W
