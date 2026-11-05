#! /bin/bash

#SBATCH --account=llm360
#SBATCH --qos=normal
#SBATCH --partition=main
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=8
#SBATCH --job-name=k2-horizon-0.9B-hc
#SBATCH --mem=0
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# k2-horizon-0.9B dense transformer (28L, d1536, GQA 32/8, head_dim 64) with hyper-connections
# (xllm/modules/residual/README.md): k2-horizon 5T data mix, jais64k enx10 tokenizer, no BOS,
# 8M tokens/step (4 nodes x 8 GPU x 32 x 8k), full FSDP2, AdamW or Muon, cosine. The matched recipe of the
# residual-stream study: lr 2e-3, wd 0.02, warmup 3000; FUSED=false runs every arm with fused_block off and
# layerwise_ckpt on (the baseline the same way, so the arms differ in the residual only), FUSED=true the fused
# block with the base block's recompute switches (q, fc1, fc3), for base / ihc / ohc.
#
# Submit from the repo (or worktree) root; slurm_logs/ must exist:
#   mkdir -p slurm_logs && SAVE=<dump_dir> RESIDUAL=ohc [VAR=value ...] sbatch examples/k2-horizon-0.9B-hc.sh [extra train.py flags]
# Extra flags are appended last and override everything below.
#
# Variables (default):
#   SAVE          dump dir (required)
#   SIZE          0.9B | 0.3B (0.9B): k2-horizon-0.9B (28L, d1536), or a 24L d1024 model (GQA 16/4, head_dim 64, FFN 3456,
#                 318M non-embedding params) under the same recipe for cheap ablations (1 GPU, bsz 32 x 8k: 78 GB)
#   RESIDUAL      base | ihc | mhc | ohc (ohc): plain residual, or identity / Sinkhorn / SO(4) transport
#                 between four residual streams (hyper-connections)
#   FUSED         true | false (false): fused training block (base / ihc / ohc) instead of layerwise_ckpt
#   BSZ           per-GPU batch of SEQ_LEN 8192 sequences (32)
#   OPTIM         adamw | muon (adamw); for muon: MUON_LR (1e-3), MUON_WD (0.1), MUON_ADJUST_LR rms_norm | spectral_norm
#                 (rms_norm, the Kimi scaling), MUON_NESTEROV (false), MUON_BLOCK_SCALE block | matrix (block); the
#                 hyper-connection projections are Muon matrices, their biases and scales stay on AdamW
#   LR            AdamW LR, also for the non-matrix params under muon (2e-3); WD weight decay (0.02);
#                 SCHEDULER constant | cosine | wsd (cosine);
#                 WARMUP steps (3000); WSD_DECAY_STEPS (20% of the run) for wsd
#   TPP           run length in tokens per non-embedding parameter (200); STEPS overrides the step count
#   EVAL_TPP      eval interval in TPP; every eval checkpoint is kept (20)
#   CKPTS_PER_EVAL  rolling checkpoints per eval interval (3)
#   WANDB_PROJECT (hyper_connections)
#   CAUSAL_ATTN_BACKEND  global causal attention backend (xattn; flash = FA2)
#   PYTHON        interpreter (.venv/bin/python)

: "${SAVE:?set SAVE}"
PYTHON=${PYTHON:-.venv/bin/python}
SIZE=${SIZE:-0.9B}
RESIDUAL=${RESIDUAL:-ohc}
FUSED=${FUSED:-false}
BSZ=${BSZ:-32}
OPTIM=${OPTIM:-adamw}
MUON_LR=${MUON_LR:-1e-3}
MUON_WD=${MUON_WD:-0.1}
MUON_ADJUST_LR=${MUON_ADJUST_LR:-rms_norm}
MUON_NESTEROV=${MUON_NESTEROV:-false}
MUON_BLOCK_SCALE=${MUON_BLOCK_SCALE:-block}
LR=${LR:-2e-3}
WD=${WD:-0.02}
SCHEDULER=${SCHEDULER:-cosine}
WARMUP=${WARMUP:-3000}
TPP=${TPP:-200}
EVAL_TPP=${EVAL_TPP:-20}
CKPTS_PER_EVAL=${CKPTS_PER_EVAL:-3}
WANDB_PROJECT=${WANDB_PROJECT:-hyper_connections}
CAUSAL_ATTN_BACKEND=${CAUSAL_ATTN_BACKEND:-xattn}  # global causal attention: xattn | flash

# run this checkout's xllm, not the one the venv was installed from
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$PWD}${PYTHONPATH:+:$PYTHONPATH}"

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_TIMEOUT=22
export CUDA_DEVICE_MAX_CONNECTIONS=32
# Azure ND H200 v5 interconnect; without these, 4-node full FSDP runs ~3x slower.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_TOPO_FILE=/opt/microsoft/ndv5-topo.xml
export NCCL_IB_HCA=mlx5_ib
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_MIN_NCHANNELS=32
export NCCL_PXN_DISABLE=1
export NCCL_COLLNET_ENABLE=1
export UCX_NET_DEVICES=mlx5_ib0:1
# The stream path peaks near the H200's 140 GB at bsz 32: without expandable segments the allocator OOMs on fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

SEQ_LEN=8192
ROPE_DIM=64
ROPE_BASE=500000
LOG_FREQ=10

# Run length from tokens per non-embedding parameter (embedding and output layer excluded;
# the hyper-connection controllers add < 0.5%).
case ${SIZE} in
  0.9B) NON_EMBED_PARAMS=880891392; SIZE_CFGS=() ;;
  0.3B) NON_EMBED_PARAMS=317768704
        SIZE_CFGS=(--model.num_layers 24 --model.model_dim 1024 --model.num_heads 16 --model.num_kv_heads 4
                   --model.ffn_hidden_dim 3456) ;;
  *) echo "unknown SIZE ${SIZE} (0.9B | 0.3B)"; exit 1 ;;
esac
TOKENS_PER_STEP=$(( SLURM_NNODES * SLURM_GPUS_ON_NODE * BSZ * SEQ_LEN ))
UNIT=$(( LOG_FREQ * CKPTS_PER_EVAL ))  # eval interval is a multiple of this, so rolling saves divide it
EVAL_FREQ=$(( (EVAL_TPP * NON_EMBED_PARAMS / TOKENS_PER_STEP + UNIT / 2) / UNIT * UNIT ))
STEPS=${STEPS:-$(( EVAL_FREQ * TPP / EVAL_TPP ))}
DUMP_FREQ=$(( EVAL_FREQ / CKPTS_PER_EVAL ))
WSD_DECAY_STEPS=${WSD_DECAY_STEPS:-$(( STEPS / 5 ))}
[ "${OPTIM}" = "muon" ] && MUON_TAG=" (${MUON_ADJUST_LR} lr ${MUON_LR} wd ${MUON_WD}, adamw lr ${LR} wd ${WD})" || MUON_TAG=""
echo "k2-horizon-${SIZE}: residual ${RESIDUAL}, fused block ${FUSED}, ${OPTIM}${MUON_TAG} ${SCHEDULER}," \
     "${STEPS} steps x ${TOKENS_PER_STEP} tokens (${TPP} TPP), eval+kept ckpt every ${EVAL_FREQ}, rolling ckpt every ${DUMP_FREQ}"

mkdir -p "${SAVE}/restarts"
cp "$0" "${SAVE}/run.sh"

D=/lustrefs/shared/initial-dataset/stripped/xllm_merged
DATA_MIX=(
  shaurya_nltk-web-QS-shuffled-High-randomized-questions_merged:0.04
  nemotron3_Nemotron-CC-v2_High-Quality:0.03
  nemotron3_Nemotron-CC-v2_High-Quality-Synthetic:0.05
  web-QS-shuffled_Medium-High:0.043
  nemotron3_Nemotron-CC-v2_Medium-High-Quality:0.02
  nemotron3_Nemotron-CC-v2_Medium-High-Quality-Synthetic:0.03
  nemotron3_Nemotron-CC-v2.1-above-medium:0.03
  nemotron3_Nemotron-CC-v2_Diverse-QA:0.03
  chunked-data_TxT360-QA:0.03
  chunked-data_opencoder:0.08
  nemotron3_Nemotron-Pretraining-Code:0.03
  nemotron3_Nemotron-CC-Code-v1:0.019
  chunked-data_stack-edu-jsonl:0.005
  shaurya_code_solutions_no_reasoning_8k_clean_oss_removal:0.035
  shaurya_code_solutions_no_reasoning_8k-hard_clean_oss_removal:0.027
  shaurya_code_solutions_with_reasoning_16k_clean_oss_removal:0.0185
  shaurya_code_solutions_with_reasoning_16k-hard_clean_oss_removal:0.0115
  shaurya_synthetic_code_solutions_opencoder_with_think_cleaned_oss_removal:0.02
  shaurya_code_prompts_all_knobs_8k_dataset_cleaned_oss_removal:0.005
  nemotron3_Nemotron-CC-Math-v1:0.012
  mikhail_math_qwen_no_think_dataset_filtered:0.04
  shaurya_math_oss_dataset_filtered_clean:0.05
  mikhail_math_rewrite_dataset:0.026
  shaurya_math_dialogue_dataset_100k_chunks:0.01
  shaurya_agentic-math-dialogue_cleaned_oss_removal_v2:0.005
  runner_nemotron_merged_no_think:0.0175
  shaurya_constraints_pretrain_output_cleaned_oss_removal:0.004
  nemotron3_Nemotron-Pretraining-SFT:0.022
  mikhail_3efforts_pretrain_16k:0.005
  shaurya_reasoning:0.0525
  shaurya_general_full:0.03
  shaurya_planning:0.041
  shaurya_ai:0.0155
  mikhail_games:0.007
  shaurya_shaurya-other:0.018
  shaurya_rewrite:0.02
  codeio:0.0015
  arabic:0.022
  french:0.015
  hindi:0.004
  Translated-Diverse-QA-merged:0.014
  fineweb2:0.014
)
DATA=$(printf "${D}/%s:text:text," "${DATA_MIX[@]}")
DATA=${DATA%,}
TOKENIZER=/lustrefs/users/xuezhe.ma/projects/data/tokenizers/jais64k_enx10_codex6.5_arax5_3digits

V=/lustrefs/shared/initial-dataset/pretrain_xllm
PALOMA=(4chan_meta_sep c4_100_domains c4_en dolma-v1_5 dolma_100_programing_languages dolma_100_subreddits
  falcon-refinedweb gab m2d2_s2orc_unsplit m2d2_wikipedia_unsplit manosphere_meta_sep mc4 redpajama
  twitterAAE_HELM_fixed wikitext_103)
VALID_DATA="${V}/SlimPajama-627B/validation/Slimpajama.jsonl,${V}/AraV6/valid/AraV6.valid.jsonl"
VALID_DATA+=$(printf ",${V}/paloma/validation/val_%s.chunk0.jsonl" "${PALOMA[@]}")
TASK_ROOT=/lustrefs/users/xuezhe.ma/projects/data/eval
TASKS="boolq,hellaswag,nq,piqa,siqa,winogrande_1.1,obqa,arc_easy,arc_challenge,race_middle,race_high,copa,mmlu,arabic_mmlu,tqa,gsm8k,bbh,drop,mbpp,human_eval"

WANDB_TEAM=mbzuai-llm

DISTRIBUTED_CFGS=(
  --context_parallel_size 1
  --model_parallel_size 1
  --slurm.partition "main"
  --slurm.mem_gb 1800
)

MODEL_CFGS=(
  --model "k2-horizon-0.9B"
  --model.ddp_backend "fsdp2"
  --model.causal_attn_backend ${CAUSAL_ATTN_BACKEND}
  --model.recompute_logits "false"
  --model.init_mode "gaussian"
  --model.init_std 0.02
  --model.rope_head_dim ${ROPE_DIM}
  --model.rope_base ${ROPE_BASE}
  --model.residual_func ${RESIDUAL}
  "${SIZE_CFGS[@]}"
)
if [ "${FUSED}" = "true" ]; then
  MODEL_CFGS+=(--model.fused_block "true" --model.layerwise_ckpt "false"
               --model.recompute_q "true" --model.recompute_fc1_out "true" --model.recompute_fc3_out "true")
else
  MODEL_CFGS+=(--model.fused_block "false" --model.layerwise_ckpt "true")
fi

DATA_CFGS=(
  --data "${DATA}"
  --dataloader.buffer_size 8192
  --dataloader.packing_type "bestfit"
  --dataloader.num_workers 2
  --tokenizer.path "${TOKENIZER}"
  --tokenizer.type "huggingface"
  --tokenizer.use_bos "false"
)

TRAINING_CFGS=(
  --steps ${STEPS}
  --seed 42
  --dtype "bf16"
  --batch_size ${BSZ}
  --seq_len ${SEQ_LEN}
  --multi_segments "true"
  --attn_stability_control 0
  --deterministic "false"
  --forward_prefetch "false"
  --nccl_timeout 18000
  --cluster_check_level 1
)

OPTIM_CFGS=(
  --optim.lr ${LR}
  --optim.scheduler ${SCHEDULER}
  --optim.lr_init_ratio 1e-4
  --optim.lr_end_ratio 0.1
  --optim.beta1 0.9
  --optim.beta2 0.95
  --optim.clip 1.0
  --optim.warmup ${WARMUP}
  --optim.weight_decay ${WD}
)
if [ "${SCHEDULER}" = "wsd" ]; then
  OPTIM_CFGS+=(--optim.wsd_decay_duration_steps ${WSD_DECAY_STEPS} --optim.wsd_decay_style "linear")
fi
if [ "${OPTIM}" = "muon" ]; then
  # Muon: the registered weights (attention, FFN, hyper-connection projections); AdamW (--optim.lr) for the rest.
  OPTIM_CFGS+=(
    --optim.optimizer "muon"
    --optim.muon_lr ${MUON_LR}
    --optim.muon_weight_decay ${MUON_WD}
    --optim.muon_adjust_lr ${MUON_ADJUST_LR}
    --optim.muon_nesterov ${MUON_NESTEROV}
    --optim.muon_momentum 0.95
    --optim.muon_block_scale ${MUON_BLOCK_SCALE}
    --dcp_for_optimizer "true"
  )
fi

VALID_CFGS=(
  --valid.ppl_file_list "${VALID_DATA}"
  --valid.task_root ${TASK_ROOT}
  --valid.task_list ${TASKS}
  --valid.batch_size 4
  --valid.seq_len ${SEQ_LEN}
  --valid.n_batches 10
  --async_eval_ngpus 8
  --async_checkpointing "false"
)

LOGGING_CFGS=(
  --log_freq ${LOG_FREQ}
  --dump_freq ${DUMP_FREQ}
  --eval_freq ${EVAL_FREQ}
  --keep_eval_checkpoints "true"
  --keep_n_last_checkpoints 2
  --log_wandb "true"
  --wandb_project ${WANDB_PROJECT}
  --wandb_entity ${WANDB_TEAM}
  --disable_workers_print "true"
  --dump_dir ${SAVE}
)

srun -u --label "${PYTHON}" -u train.py \
  "${DATA_CFGS[@]}" \
  "${DISTRIBUTED_CFGS[@]}" \
  "${MODEL_CFGS[@]}" \
  "${TRAINING_CFGS[@]}" \
  "${OPTIM_CFGS[@]}" \
  "${VALID_CFGS[@]}" \
  "${LOGGING_CFGS[@]}" \
  "$@"
