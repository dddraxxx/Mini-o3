#!/usr/bin/env bash
# Frozen launcher for qwen35_9b_official_tool_h200_sft_20260529_235552.
#
# Experiment profile:
# - finetuning: LoRA rank 8, alpha 16
# - target modules: Qwen3.5 language-model MLP gate/up/down projections
# - vision/projector freeze: disabled
# - global batch size: 32 on 8x H200
# - max length/token budget: 32768
# - lr: 1e-5, cosine, warmup 0.1, 3 epochs

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-formal}
if (($# > 0)); then
  shift
fi

export RUN_PREFIX=${RUN_PREFIX:-qwen35_9b_official_tool_h200_sft_lora}
export FINETUNING_TYPE=${FINETUNING_TYPE:-lora}
export LORA_RANK=${LORA_RANK:-8}
export LORA_ALPHA=${LORA_ALPHA:-16}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-'.*model\.language_model\.layers\..*\.mlp\.(gate_proj|up_proj|down_proj)$'}
export FREEZE_VISION_TOWER=${FREEZE_VISION_TOWER:-False}
export FREEZE_MULTI_MODAL_PROJECTOR=${FREEZE_MULTI_MODAL_PROJECTOR:-False}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
export MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
export MAX_LENGTH=${MAX_LENGTH:-32768}
export MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-32768}
export USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
export SP_SIZE=${SP_SIZE:-1}
export FSDP_STRATEGY=${FSDP_STRATEGY:-fsdp2}

export LR=${LR:-1e-5}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
export WARMUP_RATIO=${WARMUP_RATIO:-0.1}
export LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
export SAVE_FREQ=${SAVE_FREQ:-100}

export IMAGE_MIN_PIXELS=${IMAGE_MIN_PIXELS:-40000}
export IMAGE_MAX_PIXELS=${IMAGE_MAX_PIXELS:-2000000}
export ADD_VISION_ID=${ADD_VISION_ID:-True}
export WHOLE_CONVERSATION_TOKENIZE=${WHOLE_CONVERSATION_TOKENIZE:-True}
export READ_PARQUET_DTYPE_BACKEND=${READ_PARQUET_DTYPE_BACKEND:-default}

exec bash "$SCRIPT_DIR/run_qwen35_official_tool_h200_sft.sh" "$MODE" "$@"
