# Qwen3.5 Official-Tool H200 RL Runbook

This note records how to prepare and launch the formal Qwen3.5-9B Mini-o3 RL
run on the local 8x H200 node.

## Scope

The runnable entrypoint is:

```bash
exps/train/run_qwen35_official_tool_h200_rl.sh
```

Use `formal` for real training. `smoke` is only for checking the code path and
uses a tiny dataset, short response budget, no DeepSeek reward, and one update.

## Preflight

Run from the repo root:

```bash
cd /mnt/localssd/Mini-o3
git status --short
tmux list-sessions 2>/dev/null | rg 'minio3|train|formal|smoke' || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
uv run --project /mnt/localssd/Mini-o3 --no-sync python - <<'PY'
import accelerate, tensordict, tilelang, torch, transformers, vllm
print("accelerate", accelerate.__version__)
print("tensordict", tensordict.__version__)
print("tilelang", tilelang.__version__)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
PY
```

Expected important versions for the current Qwen3.5 official profile:

```text
accelerate >= 1.13.0
tensordict == 0.10.0
tilelang == 0.1.10
torch == 2.10.0+cu128
vllm == 0.18.0
```

`tilelang` is required for flash-linear-attention gated delta backward on
Hopper. Without it, actor update can fail during `loss.backward()`.

The local model snapshot is preferred when present:

```text
/mnt/localssd/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

If the local snapshot is missing, the script falls back to `Qwen/Qwen3.5-9B`.
Provide `HF_TOKEN` if download or Hub access is needed.

## Data And Keys

The current H200 wrapper prepares/uses:

```text
data/minio3_real_train_h200/train.parquet
data/minio3_real_train_h200/val.parquet
```

Formal training uses DeepSeek self-judge reward by default:

```text
SELF_JUDGE_REWARD=True
SELF_JUDGE_PROVIDER=deepseek
SELF_JUDGE_MODEL=deepseek-v4-flash
SELF_JUDGE_MAX_TOKENS=8
SELF_JUDGE_TEMPERATURE=0.0
SELF_JUDGE_MAX_RETRIES=5
SELF_JUDGE_TIMEOUT=60
```

`DEEPSEEK_API_KEY` must be available in the environment, or readable from
`/mnt/localssd/AGENTS.md`. Do not print secret values into logs.

## Formal Defaults

The formal H200 profile uses:

```text
MINIO3_TOOL_PROMPT_SUITE=qwen35_official_zoom_tool_plain_question
MINIO3_OFFICIAL_TOOL_NAME=image_zoom_in_tool
ROLLOUT_MULTI_TURN_FORMAT=qwen3_coder
MAX_PROMPT_LENGTH=16384
MAX_RESPONSE_LENGTH=16384
VAL_RESPONSE_LENGTH=32768
MAX_MODEL_LEN=65536
ROLLOUT_DP=8
ROLLOUT_TP=1
ROLLOUT_N=8
AGENT_NUM_WORKERS=64
RAY_NUM_CPUS=96
FILTER_OVERLONG_PROMPTS_WORKERS=16
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=512
PPO_MINI_BATCH_SIZE=16
PPO_MICRO_BATCH_SIZE_PER_GPU=2
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=16
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=16
ROLLOUT_GPU_MEM_UTIL=0.9
MAX_NUM_BATCHED_TOKENS=65536
MAX_NUM_SEQS=256
PROMPT_ADMISSION_ENABLE=True
PROMPT_ADMISSION_POOL_SIZE=160
TOTAL_TRAINING_STEPS=200
SAVE_LORA_ONLY=True
SAVE_FREQ=10
TEST_FREQ=10
MINIO3_RAY_STARTUP_RETRIES=2
MINIO3_RAY_STARTUP_RETRY_DELAY_S=15
```

`TRAIN_BATCH_SIZE` must be divisible by `AGENT_NUM_WORKERS` only when prompt
admission is disabled. With prompt admission enabled, prompt groups are admitted
from an oversampled pool, so the script allows worker counts above the train
batch size.
`FILTER_OVERLONG_PROMPTS_WORKERS` is kept explicit because formal VisualProbe
startup tokenizes and image-processes the full train parquet before step 0. The
verl config default is one worker, which is too slow for full formal launch.
Using 64 workers was too aggressive on this H200 node because it stacked with
Ray worker prestart and over-subscribed CPU before step 0, so the current
default is 16 plus one CPU thread per worker.

`PROMPT_ADMISSION_POOL_SIZE` is intentionally larger than `TRAIN_BATCH_SIZE`.
With `TRAIN_BATCH_SIZE=64` and pool size left at the verl default of 64, prompt
admission can degenerate near the end of a step into one in-flight candidate at
a time. `160` keeps an oversampled pool available for the last accepted prompt
group and cancels unfinished groups once the target batch is filled, while
leaving agent worker count at the stable 64-worker setting.

The current H200 profile uses `PPO_MICRO_BATCH_SIZE_PER_GPU=2` after an earlier
micro-batch-4 pilot OOMed in actor `loss.backward()` around step 8.
An 80-worker launch was also stopped before step 1 because Ray registered only
two active `AgentLoopWorker.generate_sequences` tasks while repeatedly logging
worker registration timeouts.

## 200-Step Pilot

The formal H200 script now defaults to the current 200-step pilot profile:

```bash
cd /mnt/localssd/Mini-o3
bash exps/train/run_qwen35_official_tool_h200_rl.sh formal
```

The script launches a persistent tmux session and writes:

```text
logs/<run_id>.log
save/<run_id>/train_step_metrics.jsonl
save/<run_id>/rollout_generations/
save/<run_id>/validation_generations/
save/<run_id>/train_samples.jsonl
save/<run_id>/gpu_util.jsonl
save/<run_id>/perf_debug_summary.json
```

## Active 200-Step Run Snapshot

Recorded on 2026-05-25 21:47:00 PDT while the run was active.

```text
run_id=qwen35_9b_official_tool_h200_rl_20260525_192131
tmux=minio3_formal_20260525_192131
base_commit_before_this_record=80717223c14cb08f6adee4bc397521039f001c00
log=logs/qwen35_9b_official_tool_h200_rl_20260525_192131.log
run_dir=save/qwen35_9b_official_tool_h200_rl_20260525_192131
latest_checked_step=18/200
```

The active run uses this H200 formal script profile with the following printed
configuration:

```text
MODE=formal
RUN_ID=qwen35_9b_official_tool_h200_rl_20260525_192131
RUN_DIR=/mnt/localssd/Mini-o3/save/qwen35_9b_official_tool_h200_rl_20260525_192131
MODEL_PATH=/mnt/localssd/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
MINIO3_TOOL_PROMPT_SUITE=qwen35_official_zoom_tool_plain_question
MINIO3_OFFICIAL_TOOL_NAME=image_zoom_in_tool
MINIO3_AGENT_LOOP=mini_o3_tool_agent
ROLLOUT_MULTI_TURN_FORMAT=qwen3_coder
TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=16
PPO_MICRO_BATCH_SIZE_PER_GPU=2
TRAIN_MAX_SAMPLES=-1
VAL_MAX_SAMPLES=-1
FILTER_OVERLONG_PROMPTS_WORKERS=16
ROLLOUT_N=8
ROLLOUT_DP=8
ROLLOUT_TP=1
MAX_PROMPT_LENGTH=16384
MAX_RESPONSE_LENGTH=16384
VAL_RESPONSE_LENGTH=32768
MAX_MODEL_LEN=65536
MAX_NUM_BATCHED_TOKENS=65536
MAX_NUM_SEQS=256
AGENT_NUM_WORKERS=64
RAY_NUM_CPUS=96
SELF_JUDGE_REWARD=True
SELF_JUDGE_PROVIDER=deepseek
SELF_JUDGE_MODEL=deepseek-v4-flash
SELF_JUDGE_RELAXED_ANSWER=True
PROMPT_ADMISSION_ENABLE=True
PROMPT_ADMISSION_POOL_SIZE=160
TOTAL_TRAINING_STEPS=200
SAVE_FREQ=10
TEST_FREQ=10
LOGGER_BACKENDS=["console","wandb","file"]
WANDB_MODE=online
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

Latest checked step 18 completed without OOM:

```text
timing_s/step=448.14
timing_s/gen=293.29
timing_s/old_log_prob=58.69
timing_s/update_actor=88.82
timing_s/update_weights=4.88
critic/score/mean=0.42578125
perf/throughput=594.10
prompt_admission/submitted_per_accepted=3.4375
batch/void_sample_ratio=0.970703125
```

The current script/code snapshot for this run is:

- `run_qwen35_official_tool_h200_rl.sh` defaults to 64 agent workers, a 160
  prompt-admission pool, PPO micro batch 2 per GPU, Ray 96 CPUs, 200 steps,
  early Ray startup retry, and config-only printing for these parameters.
- The train-batch divisibility guard only applies when prompt admission is
  disabled.
- The active training code keeps vLLM official Qwen3.5 tool formatting through
  `qwen3_coder`, uses the official zoom tool suite, and keeps DeepSeek reward
  enabled for formal runs.

## Monitoring

After launch, use the printed tmux session, log path, and run dir:

```bash
tmux list-sessions | rg 'minio3_formal'
tail -n 200 logs/<run_id>.log
rg -n 'Traceback|RuntimeError|Exception|AssertionError|Training Progress|step:|Final validation|Dumped generations|Appended|exit ' logs/<run_id>.log
tail -n 5 save/<run_id>/train_step_metrics.jsonl
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

Early success signs:

- `LLMServerManager` appears.
- vLLM receives LoRA with `_update_weights.add_lora.done tensors=192`.
- `Training Progress` enters the first step.
- `train_step_metrics.jsonl` gains rows with `training/global_step`.

## Expected Warnings

These are known and not launch blockers by themselves:

- vLLM may warn that Qwen3.5 visual LoRA modules are ignored. The intended
  adapter target is language MLP only; `tensors=192` confirms the 32 language
  layers x 3 MLP modules x A/B LoRA tensors are being loaded.
- vLLM may warn about `mrope_interleaved` / `mrope_section` config keys.
- Ray may warn about packaging large `.git` pack files.

## Rollback Knobs

If formal run fails during vLLM profile/capture or early memory pressure, keep
the rest of the profile and try:

```bash
MAX_NUM_BATCHED_TOKENS=49152 \
TOTAL_TRAINING_STEPS=200 \
bash exps/train/run_qwen35_official_tool_h200_rl.sh formal
```

Only consider lowering `ROLLOUT_DP` after the 49152 batched-token profile also
fails.
