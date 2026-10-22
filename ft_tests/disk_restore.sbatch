#!/usr/bin/env bash
#SBATCH --job-name=xllm-disk-restore
#SBATCH --account=moe
#SBATCH --partition=main
#SBATCH --qos=normal
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --propagate=NONE
#SBATCH --output=/lustre/users/ahmad.chamma/xllm-completedp-jobs/ckpt_bench_%j.out
#SBATCH --error=/lustre/users/ahmad.chamma/xllm-completedp-jobs/ckpt_bench_%j.err
J=${J:-$SLURM_SUBMIT_DIR}  # folder holding these scripts (submit from it)
B=${B:-$(dirname "$J")}  # the xLLM checkout
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1) MASTER_PORT=29651
W=$J/bench_work_$SLURM_JOB_ID; mkdir -p $W
srun --cpu-bind=none --container-image=/lustre/users/ahmad.chamma/models/xllm_torch12.sqsh --container-mounts=/lustre/users/ahmad.chamma:/lustre/users/ahmad.chamma --export=all bash -c "cd $B && PYTHONPATH=$B:$J/pydeps python $J/disk_restore_bench.py $W 2>&1" | tee $J/disk_restore_raw_$SLURM_JOB_ID.log | grep -E "^BENCH|^RESULT|Error|Traceback|out of memory" | head -40; grep -h -B2 "Error" $J/ckpt_bench_$SLURM_JOB_ID.err | grep -E "File " | sort -u | tail -5
rm -rf $W
