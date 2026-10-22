#!/usr/bin/env bash
#SBATCH --job-name=xllm-restart-all
#SBATCH --account=moe
#SBATCH --partition=main
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --propagate=NONE
#SBATCH --output=/lustre/users/ahmad.chamma/xllm-completedp-jobs/restart_all_%j.out
#SBATCH --error=/lustre/users/ahmad.chamma/xllm-completedp-jobs/restart_all_%j.err
J=${J:-$SLURM_SUBMIT_DIR}  # folder holding these scripts (submit from it)
B=${B:-$(dirname "$J")}  # the xLLM checkout
export MASTER_PORT=29681
W=$J/restart_work_$SLURM_JOB_ID; mkdir -p $W
srun --cpu-bind=none --container-image=/lustre/users/ahmad.chamma/models/xllm_torch12.sqsh --container-mounts=/lustre/users/ahmad.chamma:/lustre/users/ahmad.chamma --export=all bash -c "cd $B && PYTHONPATH=$B:$J/pydeps python $J/restart_all_test.py agent $W 4 2>&1" > $J/restart_all_raw_$SLURM_JOB_ID.log
rm -rf $W
