#!/usr/bin/env bash
#SBATCH --job-name=xllm-proc-start
#SBATCH --account=moe
#SBATCH --partition=main
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/users/ahmad.chamma/xllm-completedp-jobs/proc_start_%j.out
J=${J:-$SLURM_SUBMIT_DIR}  # folder holding these scripts (submit from it)
B=${B:-$(dirname "$J")}  # the xLLM checkout
srun --cpu-bind=none --container-image=/lustre/users/ahmad.chamma/models/xllm_torch12.sqsh --container-mounts=/lustre/users/ahmad.chamma:/lustre/users/ahmad.chamma bash -c '
cd '"$B"'; export PYTHONPATH='"$B:$J"'/pydeps
for rep in 1 2 3; do
  t0=$(date +%s.%N)
  for l in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$l python -c "import torch, xllm.models, xllm.optim, xllm.distributed; torch.cuda.init(); torch.zeros(1, device=\"cuda\")" & done
  wait
  t1=$(date +%s.%N); echo "PROCSTART rep=$rep 8 trainers spawn+import+cuda_init: $(python3 -c "print(round($t1-$t0,2))") s"
done'
