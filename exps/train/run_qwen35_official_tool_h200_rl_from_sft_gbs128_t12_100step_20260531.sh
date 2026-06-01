#!/usr/bin/env bash
# Frozen launcher for Qwen3.5 RL initialized from the best current SFT checkpoint.
#
# Experiment profile:
# - init model: full-freeze SFT GBS128, global_step_168_hf
# - RL tuning: LoRA rank 8 on language MLP modules
# - prompt/tool surface: same final-sentence official zoom-tool format used by SFT/eval
# - train/val turn limit: 12 assistant turns, 12 user turns
# - total training steps: 100
# - reward: DeepSeek relaxed answer judge

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-formal}
if (($# > 0)); then
  shift
fi

export RUN_PREFIX=${RUN_PREFIX:-qwen35_9b_official_tool_h200_rl_from_sft_gbs128_t12_100step}
export MODEL_PATH=${MODEL_PATH:-/mnt/localssd/Mini-o3/save/qwen35_9b_official_tool_h200_sft_full_freeze_gbs128_tok32k_20260530_143908/global_step_168_hf}

export MINIO3_TOOL_PROMPT_SUITE=${MINIO3_TOOL_PROMPT_SUITE:-qwen35_official_zoom_tool_final_sentence}
export MINIO3_OFFICIAL_TOOL_NAME=${MINIO3_OFFICIAL_TOOL_NAME:-image_zoom_in_tool}
export MINIO3_AGENT_LOOP=${MINIO3_AGENT_LOOP:-mini_o3_tool_agent}
export ROLLOUT_AGENT_LOOP=${ROLLOUT_AGENT_LOOP:-mini_o3_tool_agent}
export ROLLOUT_MULTI_TURN_FORMAT=${ROLLOUT_MULTI_TURN_FORMAT:-qwen3_coder}
export ADD_VISION_ID=${ADD_VISION_ID:-True}

export MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-12}
export MAX_USER_TURNS=${MAX_USER_TURNS:-12}
export VAL_MAX_ASSISTANT_TURNS=${VAL_MAX_ASSISTANT_TURNS:-12}
export VAL_MAX_USER_TURNS=${VAL_MAX_USER_TURNS:-12}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
export ACTOR_LR=${ACTOR_LR:-1e-6}
export MINIO3_IGNORE_CLIP=${MINIO3_IGNORE_CLIP:-False}
export MINIO3_IGNORE_EXCEED=${MINIO3_IGNORE_EXCEED:-False}
export MINIO3_IGNORE_FORMAT=${MINIO3_IGNORE_FORMAT:-False}
export MINIO3_IGNORE_INVALID=${MINIO3_IGNORE_INVALID:-False}

export SELF_JUDGE_REWARD=${SELF_JUDGE_REWARD:-True}
export SELF_JUDGE_PROVIDER=${SELF_JUDGE_PROVIDER:-deepseek}
export SELF_JUDGE_MODEL=${SELF_JUDGE_MODEL:-deepseek-v4-flash}
export SELF_JUDGE_RELAXED_ANSWER=${SELF_JUDGE_RELAXED_ANSWER:-True}

export LORA_RANK=${LORA_RANK:-8}
export LORA_ALPHA=${LORA_ALPHA:-16}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-'.*model\.language_model\.layers\..*\.mlp\.(gate_proj|up_proj|down_proj)$'}

exec bash "$SCRIPT_DIR/run_qwen35_official_tool_h200_rl.sh" "$MODE" "$@"
