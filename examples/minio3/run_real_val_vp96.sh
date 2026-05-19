#!/usr/bin/env bash
# Mini-o3 VP96 real validation | 12 turns | 32k total response budget

set -xeuo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

export HF_HOME=${HF_HOME:-/mnt/localssd/hf-cache}
export WANDB_MODE=${WANDB_MODE:-disabled}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export VLLM_USE_V1=${VLLM_USE_V1:-1}

MODEL_PATH=${MODEL_PATH:-Mini-o3/Mini-o3-7B-SFT}
TRAIN_FILE=${TRAIN_FILE:-$PROJECT_DIR/data/minio3_vp96_real_val/train.parquet}
VAL_FILE=${VAL_FILE:-$PROJECT_DIR/data/minio3_vp96_real_val/val.parquet}
RUN_DIR=${RUN_DIR:-$PROJECT_DIR/save/minio3_official_verl_real_val_vp96}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
MODEL_PATH=${MODEL_PATH} \
TRAIN_FILE=${TRAIN_FILE} \
VAL_FILE=${VAL_FILE} \
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1} \
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1} \
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0} \
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096} \
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768} \
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-49152} \
ROLLOUT_TP=${ROLLOUT_TP:-1} \
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.50} \
ROLLOUT_N=${ROLLOUT_N:-1} \
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-32} \
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-12} \
MAX_USER_TURNS=${MAX_USER_TURNS:-12} \
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1} \
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1} \
SAVE_FREQ=${SAVE_FREQ:--1} \
TEST_FREQ=${TEST_FREQ:--1} \
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8} \
PROJECT_NAME=${PROJECT_NAME:-minio3_official_verl_real_val} \
EXPERIMENT_NAME=${EXPERIMENT_NAME:-mini_o3_7b_sft_vp96_real_val} \
bash "$PROJECT_DIR/examples/minio3/run_qwen3_vl_8b_crop_lora_fsdp.sh" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8192} \
  actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS:-16} \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  data.val_batch_size=${VAL_BATCH_SIZE:-16} \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.validation_data_dir="$RUN_DIR/validation_generations" \
  trainer.default_local_dir="$RUN_DIR" \
  "$@"
