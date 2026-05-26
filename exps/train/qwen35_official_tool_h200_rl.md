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
PPO_MICRO_BATCH_SIZE_PER_GPU=4
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=16
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=16
ROLLOUT_GPU_MEM_UTIL=0.9
MAX_NUM_BATCHED_TOKENS=65536
MAX_NUM_SEQS=256
PROMPT_ADMISSION_ENABLE=True
PROMPT_ADMISSION_POOL_SIZE=128
SAVE_LORA_ONLY=True
SAVE_FREQ=10
TEST_FREQ=10
```

`TRAIN_BATCH_SIZE` must be divisible by `AGENT_NUM_WORKERS`; the script checks
this before launch because async agent-loop rollout chunks the batch equally.
`FILTER_OVERLONG_PROMPTS_WORKERS` is kept explicit because formal VisualProbe
startup tokenizes and image-processes the full train parquet before step 0. The
verl config default is one worker, which is too slow for full formal launch.
Using 64 workers was too aggressive on this H200 node because it stacked with
Ray worker prestart and over-subscribed CPU before step 0, so the current
default is 16 plus one CPU thread per worker.

`PROMPT_ADMISSION_POOL_SIZE` is intentionally larger than `TRAIN_BATCH_SIZE`.
With `TRAIN_BATCH_SIZE=64` and pool size left at the verl default of 64, prompt
admission can degenerate near the end of a step into one in-flight candidate at
a time. `128` keeps an oversampled pool available for the last accepted prompt
group and cancels unfinished groups once the target batch is filled.

## 200-Step Pilot

A 200-step formal pilot keeps the current formal H200 defaults and only changes
the step count:

```bash
cd /mnt/localssd/Mini-o3
TOTAL_TRAINING_STEPS=200 \
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
