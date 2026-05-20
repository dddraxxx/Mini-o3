#!/usr/bin/env bash
# Mini-o3 crop GRPO/LoRA | vision | official verl AgentLoop | vLLM rollout | FSDP

set -xeuo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_CMD_DEFAULT="uv run --project $PROJECT_DIR --no-sync python"
read -r -a PYTHON_CMD <<< "${PYTHON_CMD:-$PYTHON_CMD_DEFAULT}"
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

TRAIN_FILE=${TRAIN_FILE:-$PROJECT_DIR/data/minio3/train.parquet}
VAL_FILE=${VAL_FILE:-$PROJECT_DIR/data/minio3/val.parquet}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
VAL_RESPONSE_LENGTH=${VAL_RESPONSE_LENGTH:-32768}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + VAL_RESPONSE_LENGTH))}

ACTOR_LR=${ACTOR_LR:-1e-6}
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
USE_KL_LOSS=${USE_KL_LOSS:-False}
KL_LOSS_COEF=${KL_LOSS_COEF:-0}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.3}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
MINIO3_IGNORE_EXCEED=${MINIO3_IGNORE_EXCEED:-True}
MINIO3_IGNORE_VOID=${MINIO3_IGNORE_VOID:-False}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}

LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
if [[ -z "${LORA_TARGET_MODULES:-}" ]]; then
    case "${MODEL_PATH,,}" in
        *qwen3.5*|*qwen3_5*)
            # vLLM Qwen3.5 LoRA currently handles language modules only; avoid GDN fused projections.
            LORA_TARGET_MODULES='.*model\.language_model\.layers\..*\.mlp\.(gate_proj|up_proj|down_proj)$'
            ;;
        *)
            LORA_TARGET_MODULES=all-linear
            ;;
    esac
fi
MODEL_ATTN_IMPLEMENTATION=${MODEL_ATTN_IMPLEMENTATION:-flash_attention_2}
FSDP_MODEL_DTYPE=${FSDP_MODEL_DTYPE:-bfloat16}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_DP=${ROLLOUT_DP:-1}
ROLLOUT_VLLM_EXECUTOR_BACKEND=${ROLLOUT_VLLM_EXECUTOR_BACKEND:-}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
ROLLOUT_N=${ROLLOUT_N:-4}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-32}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-True}
ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE=${ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE:-False}
ROLLOUT_SKIP_VLLM_DUMMY_LORA=${ROLLOUT_SKIP_VLLM_DUMMY_LORA:-False}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-6}
MAX_USER_TURNS=${MAX_USER_TURNS:-6}
VAL_MAX_ASSISTANT_TURNS=${VAL_MAX_ASSISTANT_TURNS:-12}
VAL_MAX_USER_TURNS=${VAL_MAX_USER_TURNS:-12}
VAL_N=${VAL_N:-1}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-False}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0}
VAL_TOP_P=${VAL_TOP_P:-1.0}
VAL_TOP_K=${VAL_TOP_K:--1}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}
SAVE_FREQ=${SAVE_FREQ:-20}
SAVE_LORA_ONLY=${SAVE_LORA_ONLY:-True}
TEST_FREQ=${TEST_FREQ:-5}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
VAL_ONLY=${VAL_ONLY:-False}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
RUN_DIR=${RUN_DIR:-}
LOGGER_BACKENDS=${LOGGER_BACKENDS:-'["console","wandb"]'}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-}
TRAIN_SAMPLES_JSONL=${TRAIN_SAMPLES_JSONL:-}
TRAIN_SAMPLES_JSONL_LIMIT=${TRAIN_SAMPLES_JSONL_LIMIT:-16}
PROMPT_ADMISSION_ENABLE=${PROMPT_ADMISSION_ENABLE:-True}
PROMPT_ADMISSION_POOL_SIZE=${PROMPT_ADMISSION_POOL_SIZE:-}
PROMPT_ADMISSION_REWARD_STD_EPSILON=${PROMPT_ADMISSION_REWARD_STD_EPSILON:-1.0e-4}
PROMPT_ADMISSION_WAIT_TIMEOUT_S=${PROMPT_ADMISSION_WAIT_TIMEOUT_S:-0.1}
PROMPT_ADMISSION_CANCEL_UNFINISHED=${PROMPT_ADMISSION_CANCEL_UNFINISHED:-True}
PROMPT_ADMISSION_STATE_PATH=${PROMPT_ADMISSION_STATE_PATH:-}
GPU_MONITOR_ENABLE=${GPU_MONITOR_ENABLE:-True}
GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL:-1.0}
GPU_MONITOR_BACKEND=${GPU_MONITOR_BACKEND:-nvml}
GPU_MONITOR_SAMPLE_TIMEOUT=${GPU_MONITOR_SAMPLE_TIMEOUT:-5.0}
GPU_MONITOR_PATH=${GPU_MONITOR_PATH:-}
PERF_DEBUG_SUMMARY_ENABLE=${PERF_DEBUG_SUMMARY_ENABLE:-True}
PERF_DEBUG_SUMMARY_PATH=${PERF_DEBUG_SUMMARY_PATH:-}
# Keep Ray's dashboard HTTP server off by default to avoid port conflicts with other local jobs.
RAY_INCLUDE_DASHBOARD=${RAY_INCLUDE_DASHBOARD:-False}

PROJECT_NAME=${PROJECT_NAME:-minio3_official_verl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_8b_crop_lora_$(date +%Y%m%d_%H%M)}
TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-$PROJECT_DIR/examples/minio3/config/tool_config/minio3_crop_tool.yaml}
REWARD_FN_PATH=${REWARD_FN_PATH:-$PROJECT_DIR/examples/minio3/minio3_reward.py}

if [[ -n "${RUN_DIR}" && -z "${PROMPT_ADMISSION_STATE_PATH}" ]]; then
    PROMPT_ADMISSION_STATE_PATH="$RUN_DIR/prompt_admission_state.jsonl"
fi

if [[ -n "${RUN_DIR}" && "${LOGGER_BACKENDS}" == *\"file\"* && -z "${VERL_FILE_LOGGER_PATH:-}" ]]; then
    export VERL_FILE_LOGGER_PATH="$RUN_DIR/train_step_metrics.jsonl"
fi
if [[ -n "${VERL_FILE_LOGGER_PATH:-}" ]]; then
    mkdir -p "$(dirname "$VERL_FILE_LOGGER_PATH")"
fi

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files=${TRAIN_FILE}
    data.val_files=${VAL_FILE}
    data.image_key=images
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.val_batch_size=${VAL_BATCH_SIZE}
    data.dataloader_num_workers=${DATALOADER_NUM_WORKERS}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.tool_config_path=null
    data.function_tool_path=null
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.lora_rank=${LORA_RANK}
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    "actor_rollout_ref.model.target_modules='${LORA_TARGET_MODULES}'"
    +actor_rollout_ref.model.override_config.attn_implementation=${MODEL_ATTN_IMPLEMENTATION}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.ignore_exceed=${MINIO3_IGNORE_EXCEED}
    actor_rollout_ref.actor.ignore_void=${MINIO3_IGNORE_VOID}
    actor_rollout_ref.actor.fsdp_config.model_dtype=${FSDP_MODEL_DTYPE}
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}
)

if [[ "$USE_DYNAMIC_BSZ" == "True" || "$USE_DYNAMIC_BSZ" == "true" || "$USE_DYNAMIC_BSZ" == "1" ]]; then
    ACTOR+=(actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU})
else
    ACTOR+=(actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU})
fi

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}
    actor_rollout_ref.rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG_PATH}
    actor_rollout_ref.rollout.multi_turn.format=minio3_grounding
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS}
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_USER_TURNS}
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable
    actor_rollout_ref.rollout.agent.default_agent_loop=mini_o3_tool_agent
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS}
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K}
    actor_rollout_ref.rollout.val_kwargs.response_length=${VAL_RESPONSE_LENGTH}
    actor_rollout_ref.rollout.val_kwargs.max_assistant_turns=${VAL_MAX_ASSISTANT_TURNS}
    actor_rollout_ref.rollout.val_kwargs.max_user_turns=${VAL_MAX_USER_TURNS}
)

if [[ -n "${ROLLOUT_VLLM_EXECUTOR_BACKEND}" ]]; then
    ROLLOUT+=(
        +actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=${ROLLOUT_VLLM_EXECUTOR_BACKEND}
    )
fi

if [[ "$ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE" == "True" || "$ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE" == "true" || "$ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE" == "1" ]]; then
    ROLLOUT+=(
        +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True
    )
fi

if [[ "$ROLLOUT_SKIP_VLLM_DUMMY_LORA" == "True" || "$ROLLOUT_SKIP_VLLM_DUMMY_LORA" == "true" || "$ROLLOUT_SKIP_VLLM_DUMMY_LORA" == "1" ]]; then
    export VERL_VLLM_SKIP_DUMMY_LORA=1
fi

REF=(
    actor_rollout_ref.ref.fsdp_config.model_dtype=${FSDP_MODEL_DTYPE}
    actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD}
)

if [[ "$USE_DYNAMIC_BSZ" == "True" || "$USE_DYNAMIC_BSZ" == "true" || "$USE_DYNAMIC_BSZ" == "1" ]]; then
    ROLLOUT+=(
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    )
    REF+=(
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    )
else
    ROLLOUT+=(
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    )
    REF+=(
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    )
fi

TRAINER=(
    trainer.balance_batch=True
    "trainer.logger=${LOGGER_BACKENDS}"
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.save_lora_only=${SAVE_LORA_ONLY}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.val_only=${VAL_ONLY}
    trainer.log_val_generations=${LOG_VAL_GENERATIONS}
    trainer.prompt_admission_enable=${PROMPT_ADMISSION_ENABLE}
    trainer.prompt_admission_reward_std_epsilon=${PROMPT_ADMISSION_REWARD_STD_EPSILON}
    trainer.prompt_admission_wait_timeout_s=${PROMPT_ADMISSION_WAIT_TIMEOUT_S}
    trainer.prompt_admission_cancel_unfinished=${PROMPT_ADMISSION_CANCEL_UNFINISHED}
    trainer.prompt_admission_abort_unfinished_requests=${PROMPT_ADMISSION_ABORT_UNFINISHED_REQUESTS:-True}
)

REWARD=(
    reward.custom_reward_function.path=${REWARD_FN_PATH}
    reward.custom_reward_function.name=compute_score
)

RAY_INIT=(
    ++ray_kwargs.ray_init.include_dashboard=${RAY_INCLUDE_DASHBOARD}
)

if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
    TRAINER+=(trainer.total_training_steps=${TOTAL_TRAINING_STEPS})
fi

if [ -n "${RUN_DIR}" ]; then
    TRAINER+=(trainer.default_local_dir=${RUN_DIR})
fi

if [ -n "${PROMPT_ADMISSION_POOL_SIZE}" ]; then
    TRAINER+=(trainer.prompt_admission_pool_size=${PROMPT_ADMISSION_POOL_SIZE})
fi

if [ -n "${PROMPT_ADMISSION_STATE_PATH}" ]; then
    TRAINER+=(trainer.prompt_admission_state_path=${PROMPT_ADMISSION_STATE_PATH})
fi

if [ -n "${ROLLOUT_DATA_DIR}" ]; then
    TRAINER+=(trainer.rollout_data_dir=${ROLLOUT_DATA_DIR})
fi

if [ -n "${VALIDATION_DATA_DIR}" ]; then
    TRAINER+=(trainer.validation_data_dir=${VALIDATION_DATA_DIR})
fi

if [ -n "${TRAIN_SAMPLES_JSONL}" ]; then
    TRAINER+=(
        +trainer.train_samples_jsonl=${TRAIN_SAMPLES_JSONL}
        +trainer.train_samples_jsonl_limit=${TRAIN_SAMPLES_JSONL_LIMIT}
    )
fi

GPU_MONITOR_PID=
stop_gpu_monitor() {
    if [[ -n "${GPU_MONITOR_PID}" ]] && kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
        kill "${GPU_MONITOR_PID}" 2>/dev/null || true
        wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    fi
}
trap stop_gpu_monitor EXIT

if [[ -n "${RUN_DIR}" && ( "${GPU_MONITOR_ENABLE}" == "True" || "${GPU_MONITOR_ENABLE}" == "true" || "${GPU_MONITOR_ENABLE}" == "1" ) ]]; then
    GPU_MONITOR_PATH=${GPU_MONITOR_PATH:-$RUN_DIR/gpu_util.jsonl}
    mkdir -p "$(dirname "$GPU_MONITOR_PATH")"
    "${PYTHON_CMD[@]}" "$PROJECT_DIR/examples/minio3/monitor_gpu_util.py" \
        --output "$GPU_MONITOR_PATH" \
        --interval "$GPU_MONITOR_INTERVAL" \
        --backend "$GPU_MONITOR_BACKEND" \
        --sample-timeout "$GPU_MONITOR_SAMPLE_TIMEOUT" \
        >"${GPU_MONITOR_PATH}.log" 2>&1 &
    GPU_MONITOR_PID=$!
fi

TRAIN_EXIT_CODE=0
"${PYTHON_CMD[@]}" -m verl.trainer.main_ppo \
    +algorithm.max_num_gen_batches=${MAX_NUM_GEN_BATCHES:-256} \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${RAY_INIT[@]}" \
    "$@" || TRAIN_EXIT_CODE=$?

stop_gpu_monitor
GPU_MONITOR_PID=

if [[ -n "${RUN_DIR}" && ( "${PERF_DEBUG_SUMMARY_ENABLE}" == "True" || "${PERF_DEBUG_SUMMARY_ENABLE}" == "true" || "${PERF_DEBUG_SUMMARY_ENABLE}" == "1" ) ]]; then
    PERF_DEBUG_SUMMARY_PATH=${PERF_DEBUG_SUMMARY_PATH:-$RUN_DIR/perf_debug_summary.json}
    "${PYTHON_CMD[@]}" "$PROJECT_DIR/examples/minio3/summarize_run_metrics.py" \
        "$RUN_DIR" \
        --output "$PERF_DEBUG_SUMMARY_PATH" || true
fi

exit "$TRAIN_EXIT_CODE"
