#! /bin/bash

#SBATCH --cpus-per-task=16
#SBATCH --error=slurm_logs/%j.err
#SBATCH --gpus-per-node=8
#SBATCH --job-name=llama3-8b
#SBATCH --mem=0
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%j.out

source activate <conda env>  # activate conda env

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_TIMEOUT=22
export CUDA_DEVICE_MAX_CONNECTIONS=32
#export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export ENABLE_FLASH_ATTENTION_3="True"

LR=3e-4
WD=0.1
ROPE_DIM=128
ROPE_BASE=500000
LR_SCHD="cosine"

SAVE=</path/to/dump_dir>
mkdir -p ${SAVE}/restarts
cp $0 ${SAVE}/run.sh

DATA=</path/to/data>  # setup data path
TOKENIZER=</path/to/tokenizer>  # setup tokenizer path

VALID_DATA=</path/to/validation_data>
TASK_ROOT=</path/to/eval/root_dir>
TASKS="boolq,hellaswag,piqa,winogrande_1.1,arc_easy,arc_challenge,mmlu,tqa,gsm8k"  # list of tasks for eval

WANDB_PROJECT=<wandb-project>  # setup wandb if needed
WANDB_TEAM=<wandb-team>

DISTRIBUTED_CFGS=(
  --context_parallel_size 1
  --model_parallel_size 1
  --slurm.partition "main"
  --slurm.mem_gb 1500
)

MODEL_CFGS=(
  --model "llama3-7B"
  --model.ddp_backend "fsdp2"
  --model.causal_attn_backend "flash"
  --model.layerwise_ckpt "false"
  --model.fused_block "true"
  --model.recompute_q "false"
  --model.recompute_v "false"
  --model.recompute_attention "false"
  --model.recompute_fc1_out "false"
  --model.recompute_fc3_out "false"
  --model.recompute_logits "false"
  --model.init_mode "gaussian"
  --model.init_std 0.01
  --model.rope_head_dim ${ROPE_DIM}
  --model.rope_base ${ROPE_BASE}
)

DATA_CFGS=(
  --data $DATA
  --dataloader.buffer_size 65536
  --dataloader.packing_type "bestfit"
  --dataloader.num_workers 2
  --dataloader.skip_long_docs "false"
  --tokenizer.path $TOKENIZER
  --tokenizer.type "llama3"
)

TRAINING_CFGS=(
  --steps 100000
  --seed 42
  --dtype "bf16"
  --batch_size 4
  --seq_len 8192
  --multi_segments "true"
  --deterministic "false"
  --fp32_attn_output "false"
  --nccl_timeout 1800
  --cluster_check_level 1
)

OPTIM_CFGS=(
  --optim.lr ${LR}
  --optim.scheduler ${LR_SCHD}
  --optim.lr_init_ratio 1e-4
  --optim.lr_end_ratio 0.1
  --optim.beta1 0.9
  --optim.beta2 0.95
  --optim.clip 1.0
  --optim.warmup 3000
  --optim.weight_decay ${WD}
)

VALID_CFGS=(
  --valid.ppl_file_list ${VALID_DATA}
  --valid.task_root ${TASK_ROOT}
  --valid.task_list ${TASKS}
  --valid.batch_size 4
  --valid.seq_len 8192
  --valid.n_batches 80
  --async_eval_ngpus 8
)

LOGGING_CFGS=(
  --log_freq 10
  --dump_freq 500
  --eval_freq 5000
  --keep_eval_checkpoints "true"
  --keep_n_last_checkpoints 2
  --log_wandb "true"
  --wandb_project ${WANDB_PROJECT}
  --wandb_entity ${WANDB_TEAM}
  --disable_workers_print "true"
  --dump_dir ${SAVE}
)

srun -u --label python -u train.py \
  ${DATA_CFGS[@]} \
  ${DISTRIBUTED_CFGS[@]} \
  ${MODEL_CFGS[@]} \
  ${TRAINING_CFGS[@]} \
  ${OPTIM_CFGS[@]} \
  ${VALID_CFGS[@]} \
  ${LOGGING_CFGS[@]}
