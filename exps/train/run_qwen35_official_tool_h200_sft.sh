#!/usr/bin/env bash
# Qwen3.5-9B official-tool Mini-o3 cold-start SFT profile for the local 8x H200 node.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR=${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}
MODE=${1:-formal}
if (($# > 0)); then
  shift
fi
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}

LOCAL_QWEN35_SNAPSHOT=/mnt/localssd/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -d "$LOCAL_QWEN35_SNAPSHOT" ]]; then
    MODEL_PATH="$LOCAL_QWEN35_SNAPSHOT"
  else
    MODEL_PATH=Qwen/Qwen3.5-9B
  fi
fi

case "$MODE" in
  formal)
    RUN_PREFIX=${RUN_PREFIX:-qwen35_9b_official_tool_h200_sft}
    ;;
  smoke)
    RUN_PREFIX=${RUN_PREFIX:-qwen35_9b_official_tool_h200_sft_smoke}
    ;;
  *)
    echo "usage: $0 [formal|smoke]" >&2
    exit 2
    ;;
esac

USER_FORWARD_VARS=(
  WANDB_MODE OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS \
  TRAIN_FILES NUM_GPUS TRAIN_BATCH_SIZE MICRO_BATCH_SIZE_PER_GPU MAX_LENGTH MAX_TOKEN_LEN_PER_GPU \
  USE_DYNAMIC_BSZ TRAIN_MAX_SAMPLES VAL_FILES VAL_MAX_SAMPLES DATALOADER_NUM_WORKERS \
  IMAGE_MIN_PIXELS IMAGE_MAX_PIXELS ADD_VISION_ID WHOLE_CONVERSATION_TOKENIZE \
  READ_PARQUET_DTYPE_BACKEND \
  FINETUNING_TYPE FREEZE_VISION_TOWER FREEZE_MULTI_MODAL_PROJECTOR \
  SP_SIZE FSDP_SIZE FSDP_STRATEGY FSDP_MODEL_DTYPE PARAM_OFFLOAD OPTIMIZER_OFFLOAD USE_TORCH_COMPILE \
  LORA_RANK LORA_ALPHA LORA_TARGET_MODULES LR WEIGHT_DECAY WARMUP_RATIO LR_SCHEDULER_TYPE \
  TOTAL_EPOCHS TOTAL_TRAINING_STEPS SAVE_FREQ TEST_FREQ MAX_CKPT_TO_KEEP LOGGER_BACKENDS \
  PROJECT_NAME EXPERIMENT_NAME RESUME_MODE MODEL_ATTN_IMPLEMENTATION
)

declare -A USER_SET
for var_name in "${USER_FORWARD_VARS[@]}"; do
  if [[ -v "$var_name" ]]; then
    USER_SET[$var_name]=1
  fi
done

set_mode_default() {
  local name=$1
  local value=$2
  if [[ -z "${USER_SET[$name]:-}" ]]; then
    export "$name=$value"
  fi
}

RUN_ID=${RUN_ID:-${RUN_PREFIX}_${TIMESTAMP}}
RUN_DIR=${RUN_DIR:-$PROJECT_DIR/save/$RUN_ID}
LOG_PATH=${LOG_PATH:-$PROJECT_DIR/logs/$RUN_ID.log}
RECORD_DIR=${RECORD_DIR:-$PROJECT_DIR/artifacts/train/$RUN_ID}
TMUX_SESSION=${TMUX_SESSION:-minio3_sft_${MODE}_${TIMESTAMP}}

if [[ "${MINIO3_PRINT_CONFIG_ONLY:-0}" == "1" ]]; then
  export MINIO3_SFT_FOREGROUND=1
fi

if [[ "${MINIO3_SFT_FOREGROUND:-0}" != "1" && "${MINIO3_SFT_INNER:-0}" != "1" ]]; then
  mkdir -p "$PROJECT_DIR/logs" "$RUN_DIR"
  LAUNCH_ENV_PATH="$RUN_DIR/launch_env.sh"
  : > "$LAUNCH_ENV_PATH"
  chmod 600 "$LAUNCH_ENV_PATH"
  for var_name in "${USER_FORWARD_VARS[@]}"; do
    if [[ -n "${USER_SET[$var_name]:-}" ]]; then
      printf 'export %s=%q\n' "$var_name" "${!var_name}" >> "$LAUNCH_ENV_PATH"
    fi
  done
  printf -v cmd 'cd %q && source %q && MINIO3_SFT_INNER=1 RUN_ID=%q RUN_DIR=%q LOG_PATH=%q RECORD_DIR=%q MODEL_PATH=%q bash %q %q > %q 2>&1' \
    "$PROJECT_DIR" "$LAUNCH_ENV_PATH" "$RUN_ID" "$RUN_DIR" "$LOG_PATH" "$RECORD_DIR" "$MODEL_PATH" "$SCRIPT_PATH" "$MODE" "$LOG_PATH"
  tmux new-session -d -s "$TMUX_SESSION" "$cmd"
  echo "launched tmux session: $TMUX_SESSION"
  echo "log: $LOG_PATH"
  echo "run dir: $RUN_DIR"
  echo "record dir: $RECORD_DIR"
  echo "launch env: $LAUNCH_ENV_PATH"
  exit 0
fi

AGENTS_FILE=${AGENTS_FILE:-/mnt/localssd/AGENTS.md}
if [[ -z "${HF_TOKEN:-}" && -r "$AGENTS_FILE" ]]; then
  HF_TOKEN=$(grep -o 'hf_[A-Za-z0-9]*' "$AGENTS_FILE" | head -n 1 || true)
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}
fi
if [[ "$MODEL_PATH" == /* ]]; then
  export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
  export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
fi

export PROJECT_DIR MODEL_PATH RUN_ID RUN_DIR RECORD_DIR
export HF_HOME=${HF_HOME:-/mnt/localssd/.cache/huggingface}
export WANDB_MODE=${WANDB_MODE:-online}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

export TRAIN_FILES=${TRAIN_FILES:-$PROJECT_DIR/data/minio3_coldstart_verl_sft_qwen35_official_tool/train_shards}
export VAL_FILES=${VAL_FILES:-null}
export NUM_GPUS=${NUM_GPUS:-8}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
export MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
export MAX_LENGTH=${MAX_LENGTH:-32768}
export MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-32768}
export USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
export IMAGE_MIN_PIXELS=${IMAGE_MIN_PIXELS:-40000}
export IMAGE_MAX_PIXELS=${IMAGE_MAX_PIXELS:-2000000}
export ADD_VISION_ID=${ADD_VISION_ID:-True}
export WHOLE_CONVERSATION_TOKENIZE=${WHOLE_CONVERSATION_TOKENIZE:-True}
export READ_PARQUET_DTYPE_BACKEND=${READ_PARQUET_DTYPE_BACKEND:-default}
export FINETUNING_TYPE=${FINETUNING_TYPE:-lora}

export SP_SIZE=${SP_SIZE:-1}
export FSDP_SIZE=${FSDP_SIZE:--1}
export FSDP_STRATEGY=${FSDP_STRATEGY:-fsdp2}
export FSDP_MODEL_DTYPE=${FSDP_MODEL_DTYPE:-bfloat16}
export PARAM_OFFLOAD=${PARAM_OFFLOAD:-False}
export OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-False}
export USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-False}

case "${FINETUNING_TYPE,,}" in
  lora)
    export FINETUNING_TYPE=lora
    export LORA_RANK=${LORA_RANK:-8}
    export LORA_ALPHA=${LORA_ALPHA:-16}
    export FREEZE_VISION_TOWER=${FREEZE_VISION_TOWER:-False}
    export FREEZE_MULTI_MODAL_PROJECTOR=${FREEZE_MULTI_MODAL_PROJECTOR:-False}
    ;;
  full)
    export FINETUNING_TYPE=full
    export LORA_RANK=${LORA_RANK:-0}
    export LORA_ALPHA=${LORA_ALPHA:-0}
    export FREEZE_VISION_TOWER=${FREEZE_VISION_TOWER:-True}
    export FREEZE_MULTI_MODAL_PROJECTOR=${FREEZE_MULTI_MODAL_PROJECTOR:-True}
    ;;
  *)
    echo "FINETUNING_TYPE must be lora or full, got: $FINETUNING_TYPE" >&2
    exit 2
    ;;
esac
if [[ -z "${LORA_TARGET_MODULES:-}" ]]; then
  case "${MODEL_PATH,,}" in
    *qwen3.5*|*qwen3_5*)
      LORA_TARGET_MODULES='.*model\.language_model\.layers\..*\.mlp\.(gate_proj|up_proj|down_proj)$'
      ;;
    *)
      LORA_TARGET_MODULES='all-linear'
      ;;
  esac
fi
export LORA_TARGET_MODULES

export LR=${LR:-1e-5}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
export WARMUP_RATIO=${WARMUP_RATIO:-0.1}
export LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
export MODEL_ATTN_IMPLEMENTATION=${MODEL_ATTN_IMPLEMENTATION:-flash_attention_2}

export TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
export SAVE_FREQ=${SAVE_FREQ:-100}
export TEST_FREQ=${TEST_FREQ:--1}
export MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-5}
export LOGGER_BACKENDS=${LOGGER_BACKENDS:-'["console","wandb","file"]'}
export PROJECT_NAME=${PROJECT_NAME:-Mini-o3-qwen35-sft}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-$RUN_ID}
export RESUME_MODE=${RESUME_MODE:-auto}
export VERL_FILE_LOGGER_PATH=${VERL_FILE_LOGGER_PATH:-$RUN_DIR/train_step_metrics.jsonl}

if [[ "$MODE" == "smoke" ]]; then
  set_mode_default WANDB_MODE disabled
  set_mode_default NUM_GPUS 1
  set_mode_default TRAIN_BATCH_SIZE 1
  set_mode_default TRAIN_MAX_SAMPLES 2
  set_mode_default MAX_LENGTH 32768
  set_mode_default MAX_TOKEN_LEN_PER_GPU 32768
  set_mode_default SP_SIZE 1
  set_mode_default TOTAL_EPOCHS 1
  set_mode_default TOTAL_TRAINING_STEPS 1
  set_mode_default SAVE_FREQ -1
  set_mode_default LOGGER_BACKENDS '["console","file"]'
fi

if [[ ! -e "$TRAIN_FILES" ]]; then
  echo "missing TRAIN_FILES: $TRAIN_FILES" >&2
  exit 2
fi

if [[ "${MINIO3_PRINT_CONFIG_ONLY:-0}" == "1" ]]; then
  for name in \
    MODE RUN_ID RUN_DIR RECORD_DIR MODEL_PATH TRAIN_FILES VAL_FILES NUM_GPUS \
    TRAIN_BATCH_SIZE MICRO_BATCH_SIZE_PER_GPU MAX_LENGTH MAX_TOKEN_LEN_PER_GPU USE_DYNAMIC_BSZ \
    TRAIN_MAX_SAMPLES DATALOADER_NUM_WORKERS IMAGE_MIN_PIXELS IMAGE_MAX_PIXELS ADD_VISION_ID \
    WHOLE_CONVERSATION_TOKENIZE READ_PARQUET_DTYPE_BACKEND FINETUNING_TYPE FREEZE_VISION_TOWER \
    FREEZE_MULTI_MODAL_PROJECTOR SP_SIZE FSDP_SIZE FSDP_STRATEGY FSDP_MODEL_DTYPE \
    PARAM_OFFLOAD OPTIMIZER_OFFLOAD USE_TORCH_COMPILE LORA_RANK LORA_ALPHA LORA_TARGET_MODULES \
    LR WEIGHT_DECAY WARMUP_RATIO LR_SCHEDULER_TYPE TOTAL_EPOCHS TOTAL_TRAINING_STEPS \
    SAVE_FREQ TEST_FREQ MAX_CKPT_TO_KEEP LOGGER_BACKENDS WANDB_MODE; do
    printf '%s=%s\n' "$name" "${!name-}"
  done
  exit 0
fi

TRAIN_CMD=(
  uv run --project "$PROJECT_DIR" --no-sync torchrun
  --standalone
  --nnodes=1
  --nproc_per_node="${NUM_GPUS}"
  -m verl.trainer.sft_trainer
  "data.train_files=${TRAIN_FILES}"
  "data.val_files=${VAL_FILES}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}"
  "data.max_length=${MAX_LENGTH}"
  "data.pad_mode=no_padding"
  "data.truncation=error"
  "data.use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
  "data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
  "data.train_max_samples=${TRAIN_MAX_SAMPLES}"
  "data.val_max_samples=${VAL_MAX_SAMPLES}"
  "data.messages_key=messages"
  "+data.image_key=images"
  "data.tools_key=tools"
  "data.enable_thinking_key=enable_thinking"
  "+data.whole_conversation_tokenize=${WHOLE_CONVERSATION_TOKENIZE}"
  "+data.read_parquet_dtype_backend=${READ_PARQUET_DTYPE_BACKEND}"
  "+data.image_min_pixels=${IMAGE_MIN_PIXELS}"
  "+data.image_max_pixels=${IMAGE_MAX_PIXELS}"
  "+data.apply_chat_template_kwargs.add_vision_id=${ADD_VISION_ID}"
  "data.num_workers=${DATALOADER_NUM_WORKERS}"
  "model.path=${MODEL_PATH}"
  "model.trust_remote_code=True"
  "+model.override_config.attn_implementation=${MODEL_ATTN_IMPLEMENTATION}"
  "model.use_remove_padding=True"
  "model.enable_gradient_checkpointing=True"
  "model.freeze_vision_tower=${FREEZE_VISION_TOWER}"
  "model.freeze_multi_modal_projector=${FREEZE_MULTI_MODAL_PROJECTOR}"
  "model.lora_rank=${LORA_RANK}"
  "model.lora_alpha=${LORA_ALPHA}"
  "model.target_modules='${LORA_TARGET_MODULES}'"
  "engine=fsdp"
  "optim=fsdp"
  "engine.strategy=${FSDP_STRATEGY}"
  "engine.ulysses_sequence_parallel_size=${SP_SIZE}"
  "engine.fsdp_size=${FSDP_SIZE}"
  "engine.model_dtype=${FSDP_MODEL_DTYPE}"
  "engine.dtype=bfloat16"
  "engine.param_offload=${PARAM_OFFLOAD}"
  "engine.optimizer_offload=${OPTIMIZER_OFFLOAD}"
  "engine.use_torch_compile=${USE_TORCH_COMPILE}"
  "optim.lr=${LR}"
  "optim.weight_decay=${WEIGHT_DECAY}"
  "optim.betas=[0.9,0.95]"
  "optim.clip_grad=1.0"
  "optim.lr_warmup_steps_ratio=${WARMUP_RATIO}"
  "optim.lr_scheduler_type=${LR_SCHEDULER_TYPE}"
  "trainer.logger=${LOGGER_BACKENDS}"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.default_local_dir=${RUN_DIR}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.max_ckpt_to_keep=${MAX_CKPT_TO_KEEP}"
  "trainer.resume_mode=${RESUME_MODE}"
  "trainer.nnodes=1"
  "trainer.n_gpus_per_node=${NUM_GPUS}"
  "checkpoint.save_contents=[model,optimizer,extra]"
  "$@"
)

mkdir -p "$RUN_DIR" "$RECORD_DIR"
{
  printf '#!/usr/bin/env bash\n'
  printf 'cd %q\n' "$PROJECT_DIR"
  printf '%q ' "${TRAIN_CMD[@]}"
  printf '\n'
} > "$RECORD_DIR/launch_command.sh"
chmod +x "$RECORD_DIR/launch_command.sh"

cat > "$RECORD_DIR/README.md" <<EOF
# $RUN_ID

Formal Qwen3.5-9B cold-start SFT for Mini-o3 official zoom-tool conversations.

- Script: \`exps/train/run_qwen35_official_tool_h200_sft.sh\`
- Log: \`$LOG_PATH\`
- Save dir: \`$RUN_DIR\`
- Train files: \`$TRAIN_FILES\`
- Model: \`$MODEL_PATH\`
- Base reference: official Mini-o3 cold-start SFT uses Qwen2.5-VL-7B-Instruct, image pixels 40000/2000000, cutoff 32768, full finetune with frozen vision/projector, lr 1e-5, 3 epochs, cosine scheduler, warmup 0.1, bf16.
- Local adaptation: Qwen3.5-9B, Qwen3.5 official tool-call surface, add_vision_id=True, whole-conversation SFT tokenization, finetuning_type=$FINETUNING_TYPE.

Key parameters:

\`\`\`
FINETUNING_TYPE=$FINETUNING_TYPE
NUM_GPUS=$NUM_GPUS
TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE
MICRO_BATCH_SIZE_PER_GPU=$MICRO_BATCH_SIZE_PER_GPU
MAX_LENGTH=$MAX_LENGTH
MAX_TOKEN_LEN_PER_GPU=$MAX_TOKEN_LEN_PER_GPU
USE_DYNAMIC_BSZ=$USE_DYNAMIC_BSZ
READ_PARQUET_DTYPE_BACKEND=$READ_PARQUET_DTYPE_BACKEND
SP_SIZE=$SP_SIZE
FSDP_STRATEGY=$FSDP_STRATEGY
LR=$LR
WEIGHT_DECAY=$WEIGHT_DECAY
WARMUP_RATIO=$WARMUP_RATIO
TOTAL_EPOCHS=$TOTAL_EPOCHS
TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS
SAVE_FREQ=$SAVE_FREQ
IMAGE_MIN_PIXELS=$IMAGE_MIN_PIXELS
IMAGE_MAX_PIXELS=$IMAGE_MAX_PIXELS
FREEZE_VISION_TOWER=$FREEZE_VISION_TOWER
FREEZE_MULTI_MODAL_PROJECTOR=$FREEZE_MULTI_MODAL_PROJECTOR
LORA_RANK=$LORA_RANK
LORA_ALPHA=$LORA_ALPHA
LORA_TARGET_MODULES=$LORA_TARGET_MODULES
\`\`\`

Exact command is frozen in \`launch_command.sh\`.
EOF

"${TRAIN_CMD[@]}"
