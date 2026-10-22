#!/usr/bin/env bash
#SBATCH --job-name=xllm-pg-scale
#SBATCH --account=moe
#SBATCH --partition=main
#SBATCH --qos=normal
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --propagate=NONE
#SBATCH --output=/lustre/users/ahmad.chamma/xllm-completedp-jobs/pg_scale_%j.out
#SBATCH --error=/lustre/users/ahmad.chamma/xllm-completedp-jobs/pg_scale_%j.err
J=${J:-$SLURM_SUBMIT_DIR}  # folder holding these scripts (submit from it)
B=${B:-$(dirname "$J")}  # the xLLM checkout
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1) MASTER_PORT=$((29500 + RANDOM % 300))
srun --cpu-bind=none --container-image=/lustre/users/ahmad.chamma/models/xllm_torch12.sqsh --container-mounts=/lustre/users/ahmad.chamma:/lustre/users/ahmad.chamma --export=all bash -c "cd $B && PYTHONPATH=$B:$J/pydeps python $J/pg_scale_bench.py $1 2>&1" > $J/pg_scale_raw_$SLURM_JOB_ID.log
