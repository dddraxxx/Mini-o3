#!/usr/bin/env bash
# Reproduce the full VisualProbe Qwen3.5-9B val-only evals recorded in this directory.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR=${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}
SNAPSHOT_DIR=${SNAPSHOT_DIR:-$PROJECT_DIR/exps/eval/snapshots/20260525_qwen35_vp}
VARIANT=${1:-final_sentence}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}

LOCAL_QWEN35_SNAPSHOT=/mnt/localssd/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -d "$LOCAL_QWEN35_SNAPSHOT" ]]; then
    MODEL_PATH="$LOCAL_QWEN35_SNAPSHOT"
  else
    MODEL_PATH=Qwen/Qwen3.5-9B
  fi
fi

case "$VARIANT" in
  plain_question|plainq)
    CANONICAL_VARIANT=plain_question
    TOOL_PROMPT_SUITE=qwen35_official_zoom_tool_plain_question
    RUN_PREFIX=visualprobe_full515_qwen35_9b_official_tool_plainq_minio3agent_localpath_deepseek_relaxed
    DATA_DIR_DEFAULT="$PROJECT_DIR/data/minio3_visualprobe_val_plain_question515_minio3agent_localpath"
    ;;
  final_sentence|final|fs)
    CANONICAL_VARIANT=final_sentence
    TOOL_PROMPT_SUITE=qwen35_official_zoom_tool_final_sentence
    RUN_PREFIX=visualprobe_full515_qwen35_9b_official_tool_finalsentence_minio3agent_localpath_deepseek_relaxed
    DATA_DIR_DEFAULT="$PROJECT_DIR/data/minio3_visualprobe_val_final_sentence515_minio3agent_localpath"
    ;;
  answer_tag|answer|tag)
    CANONICAL_VARIANT=answer_tag
    TOOL_PROMPT_SUITE=qwen35_official_zoom_tool
    RUN_PREFIX=visualprobe_full515_qwen35_9b_official_tool_deepseek_relaxed
    DATA_DIR_DEFAULT="$PROJECT_DIR/data/minio3_visualprobe_val_smoke515"
    ;;
  *)
    echo "usage: $0 [plain_question|final_sentence|answer_tag]" >&2
    exit 2
    ;;
esac

RUN_ID=${RUN_ID:-${RUN_PREFIX}_${TIMESTAMP}}
RUN_DIR=${RUN_DIR:-$PROJECT_DIR/save/$RUN_ID}
DATA_DIR=${DATA_DIR:-$DATA_DIR_DEFAULT}
LOG_PATH=${LOG_PATH:-$PROJECT_DIR/logs/$RUN_ID.log}
TMUX_SESSION=${TMUX_SESSION:-minio3_vp_full_${CANONICAL_VARIANT}_${TIMESTAMP}}

if [[ ! -r "$SNAPSHOT_DIR/run_real_val_visualprobe_smoke.sh" ]]; then
  echo "missing eval snapshot at $SNAPSHOT_DIR" >&2
  exit 2
fi

if [[ "${MINIO3_EVAL_FOREGROUND:-0}" != "1" && "${MINIO3_EVAL_INNER:-0}" != "1" ]]; then
  mkdir -p "$PROJECT_DIR/logs" "$RUN_DIR"
  printf -v cmd 'cd %q && MINIO3_EVAL_INNER=1 RUN_ID=%q RUN_DIR=%q DATA_DIR=%q LOG_PATH=%q MODEL_PATH=%q SNAPSHOT_DIR=%q bash %q %q > %q 2>&1' \
    "$PROJECT_DIR" "$RUN_ID" "$RUN_DIR" "$DATA_DIR" "$LOG_PATH" "$MODEL_PATH" "$SNAPSHOT_DIR" "$SCRIPT_PATH" "$CANONICAL_VARIANT" "$LOG_PATH"
  tmux new-session -d -s "$TMUX_SESSION" "$cmd"
  echo "launched tmux session: $TMUX_SESSION"
  echo "log: $LOG_PATH"
  echo "run dir: $RUN_DIR"
  echo "data dir: $DATA_DIR"
  exit 0
fi

AGENTS_FILE=${AGENTS_FILE:-/mnt/localssd/AGENTS.md}
if [[ -z "${DEEPSEEK_API_KEY:-}" && -r "$AGENTS_FILE" ]]; then
  DEEPSEEK_API_KEY=$(grep -o 'sk-[A-Za-z0-9]*' "$AGENTS_FILE" | tail -n 1 || true)
fi
if [[ -z "${HF_TOKEN:-}" && -r "$AGENTS_FILE" ]]; then
  HF_TOKEN=$(grep -o 'hf_[A-Za-z0-9]*' "$AGENTS_FILE" | head -n 1 || true)
fi
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "missing DEEPSEEK_API_KEY; export it or provide $AGENTS_FILE" >&2
  exit 2
fi
export DEEPSEEK_API_KEY
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}
fi

if [[ "$MODEL_PATH" == /* ]]; then
  export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
  export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
fi

export PROJECT_DIR SNAPSHOT_DIR
export MODEL_PATH RUN_ID RUN_DIR DATA_DIR
export SMOKE_CASES=${SMOKE_CASES:-515}
export SMOKE_TRAIN_CASES=${SMOKE_TRAIN_CASES:-8}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-512}
export VAL_N=${VAL_N:-1}
export VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
export VAL_TOP_P=${VAL_TOP_P:-1.0}
export VAL_TOP_K=${VAL_TOP_K:--1}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-96}
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-64}
export MINIO3_TOOL_PROMPT_SUITE=${MINIO3_TOOL_PROMPT_SUITE:-$TOOL_PROMPT_SUITE}
export MINIO3_OFFICIAL_TOOL_NAME=${MINIO3_OFFICIAL_TOOL_NAME:-image_zoom_in_tool}
export MINIO3_AGENT_LOOP=${MINIO3_AGENT_LOOP:-mini_o3_tool_agent}
export ROLLOUT_AGENT_LOOP=${ROLLOUT_AGENT_LOOP:-mini_o3_tool_agent}
export ROLLOUT_MULTI_TURN_FORMAT=${ROLLOUT_MULTI_TURN_FORMAT:-qwen3_coder}
export ADD_VISION_ID=${ADD_VISION_ID:-True}
export REWARD_FN_PATH=${REWARD_FN_PATH:-$SNAPSHOT_DIR/minio3_reward.py}
export PROJECT_NAME=${PROJECT_NAME:-Mini-o3-vp-formal}
export LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
export ROLLOUT_DP=${ROLLOUT_DP:-8}
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_VLLM_EXECUTOR_BACKEND=${ROLLOUT_VLLM_EXECUTOR_BACKEND:-uni}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.9}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-65536}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-16384}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
export VAL_RESPONSE_LENGTH=${VAL_RESPONSE_LENGTH:-32768}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
export ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE=${ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE:-False}
export ROLLOUT_SKIP_VLLM_DUMMY_LORA=${ROLLOUT_SKIP_VLLM_DUMMY_LORA:-True}
export SKIP_INITIAL_UPDATE_WEIGHTS=${SKIP_INITIAL_UPDATE_WEIGHTS:-True}
export CHECK_QWEN35_ENV=${CHECK_QWEN35_ENV:-True}
export LOGGER_BACKENDS=${LOGGER_BACKENDS:-'["console","file"]'}

bash "$SNAPSHOT_DIR/run_real_val_visualprobe_smoke.sh" \
  +reward.custom_reward_function.reward_kwargs.self_judge_reward=True \
  +reward.custom_reward_function.reward_kwargs.self_judge_provider=deepseek \
  +reward.custom_reward_function.reward_kwargs.self_judge_model=deepseek-v4-flash \
  +reward.custom_reward_function.reward_kwargs.self_judge_base_url=https://api.deepseek.com \
  +reward.custom_reward_function.reward_kwargs.self_judge_reasoning_effort=none \
  +reward.custom_reward_function.reward_kwargs.self_judge_max_tokens=8 \
  +reward.custom_reward_function.reward_kwargs.self_judge_temperature=0.0 \
  +reward.custom_reward_function.reward_kwargs.self_judge_max_retries=5 \
  +reward.custom_reward_function.reward_kwargs.self_judge_timeout=60 \
  +reward.custom_reward_function.reward_kwargs.self_judge_initial_delay=1.0 \
  +reward.custom_reward_function.reward_kwargs.self_judge_relaxed_answer=True
