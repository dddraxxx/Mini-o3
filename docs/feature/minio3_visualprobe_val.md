# Mini-o3 VisualProbe Val Smoke

This note tracks the Qwen3.5-9B VisualProbe val-only path before full
Easy/Medium/Hard evaluation.

## Goal

Run a small VisualProbe val-only smoke with official verl AgentLoop before the
full validation sweep:

- model: `Qwen/Qwen3.5-9B`
- data: stratified VisualProbe Easy/Medium/Hard, 10 cases total
- reward: `examples/minio3/empty_reward.py`, zero score only
- rollout: Mini-o3 multi-turn crop loop with `mini_o3_tool_agent`
- compare greedy validation against sampled validation with temperature 1

## Scripts

```bash
examples/minio3/prepare_visualprobe_val_smoke.py
examples/minio3/run_real_val_visualprobe_smoke.sh
examples/minio3/empty_reward.py
examples/minio3/check_qwen35_env.py
examples/minio3/install_qwen35_official_env.sh
```

The smoke script creates:

```bash
data/minio3_visualprobe_val_smoke10/train.parquet
data/minio3_visualprobe_val_smoke10/val.parquet
save/visualprobe_val_smoke10_qwen35_9b_temp*/validation_generations/
```

The train parquet contains one row because verl still expects train files to be
configured; val-only uses the val parquet.

## Current Smoke Candidate

These are the current 10-case smoke script defaults, not validated formal
defaults. The rollout-throughput values intentionally borrow the aggressive H200
script settings so the smoke can expose batching/GPU-util problems early.

| Param | Default |
| --- | --- |
| `MODEL_PATH` | `Qwen/Qwen3.5-9B` |
| `CHECK_QWEN35_ENV` | `True` |
| `SMOKE_CASES` | `10` |
| `VAL_BATCH_SIZE` | `10` |
| `MAX_PROMPT_LENGTH` | `16384` |
| `VAL_RESPONSE_LENGTH` | `32768` |
| `MAX_MODEL_LEN` | `65536` |
| `ROLLOUT_TP` | `1` |
| `ROLLOUT_DP` | `8` |
| `ROLLOUT_VLLM_EXECUTOR_BACKEND` | `uni` |
| `ROLLOUT_GPU_MEM_UTIL` | `0.9` |
| `MAX_NUM_BATCHED_TOKENS` | `65536` |
| `MAX_NUM_SEQS` | `256` |
| `AGENT_NUM_WORKERS` | `64` |
| `VAL_MAX_ASSISTANT_TURNS` | `12` |
| `VAL_MAX_USER_TURNS` | `12` |
| `VAL_N` | `1` |
| `LORA_RANK` | `0` |
| `LOG_VAL_GENERATIONS` | `10` |

Greedy run:

```bash
RUN_ID=visualprobe_val_smoke10_qwen35_9b_temp0 \
VAL_DO_SAMPLE=False \
VAL_TEMPERATURE=0 \
bash examples/minio3/run_real_val_visualprobe_smoke.sh
```

Temperature-1 run:

```bash
RUN_ID=visualprobe_val_smoke10_qwen35_9b_temp1 \
VAL_DO_SAMPLE=True \
VAL_TEMPERATURE=1.0 \
VAL_TOP_P=1.0 \
VAL_TOP_K=-1 \
bash examples/minio3/run_real_val_visualprobe_smoke.sh
```

## Things To Watch

- Prompt length may still be too small for Qwen3.5-9B visual inputs even after
  raising the smoke default to 16384. The local `Qwen/Qwen3.5-9B` config has
  image token ids and `vision_config`, so image tokens are expected to consume a
  larger prompt budget than the Qwen3-VL 8B smoke.
- The original Mini-o3/VisualProbe evaluation used temperature 1. Greedy
  validation is useful for deterministic debug, but the full VP score should
  likely use `VAL_DO_SAMPLE=True`, `VAL_TEMPERATURE=1.0`, `VAL_TOP_P=1.0`,
  `VAL_TOP_K=-1`.
- H200 rollout throughput parameters are intentionally aggressive smoke
  candidates, not settled formal defaults:
  `ROLLOUT_DP=8`, `AGENT_NUM_WORKERS=64`,
  `MAX_NUM_BATCHED_TOKENS=65536`, `MAX_NUM_SEQS=256`.
- If the smoke fails before generation, first check image-token prompt length,
  model processor/chat template compatibility, and vLLM max model length.
- If the smoke runs but GPU util is low, check per-turn tool/image
  preprocessing time and request granularity before changing admission logic.

## Smoke Results

2026-05-19 local envcheck:

- Smoke data generation succeeded:
  `data/minio3_visualprobe_val_smoke10/train.parquet` has 1 row and
  `data/minio3_visualprobe_val_smoke10/val.parquet` has 10 rows.
- Val split is stratified across all VP levels: Easy 4, Medium 3, Hard 3.
- Attempted run id:
  `visualprobe_val_smoke10_qwen35_9b_temp0_envcheck_20260519`.
- Attempted run log:
  `logs/visualprobe_val_smoke10_qwen35_9b_temp0_envcheck_20260519.log`.
- Attempted run dir:
  `save/visualprobe_val_smoke10_qwen35_9b_temp0_envcheck_20260519`.
- Current active env is not Qwen3.5-ready:
  `transformers=4.52.0`, `vllm=0.9.2`, `torch=2.7.0+cu126`.
- `transformers.models.qwen3_5` is missing and
  `vllm.model_executor.models.qwen3_5` is missing. A direct
  `AutoConfig.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)`
  fails with unknown `model_type=qwen3_5`.
- A temp-0 launch with the old env reached local Ray startup, pushed the repo
  package, then submitted no actors/tasks and used 0/8 Ray GPUs. No generation
  was produced; temp-1 was not run because temp-0 did not enter rollout.
- GPU monitor saw only the cooperative idle keeper during this failed launch.
  The formal rollout params did not get exercised yet.

The first blocker is environment support for Qwen3.5, not prompt length or
sampling. Keep `CHECK_QWEN35_ENV=True` until the env is upgraded, so the smoke
fails before Ray starts.

## Formal H200 Val Batch

For formal validation, keep the total number of concurrent val trajectories in
the same range as train rollout:

```text
train trajectory count = TRAIN_BATCH_SIZE * ROLLOUT_N = 64 * 8 = 512
val trajectory count = VAL_BATCH_SIZE * VAL_N = 512 * 1 = 512
```

The current val path does not fan out with train `ROLLOUT_N`; it effectively
runs one trajectory per val sample when `VAL_N=1`. With
`AGENT_NUM_WORKERS=64`, `VAL_BATCH_SIZE=512` gives 8 val trajectories per
agent-loop worker and about 64 concurrent requests per vLLM DP replica when
`ROLLOUT_DP=8`. This is below the H200 script's `MAX_NUM_SEQS=256` and avoids
the padding waste of small val batches.

## Official Qwen3.5 Env Target

Follow upstream verl's official Qwen3.5 FSDP script instead of the vLLM 0.12
release line:

- official script:
  <https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_5_27b_fsdp.sh>
- upstream dependency note in that script:
  `GPU vllm==0.18.0, transformers@<cc7ab9be>`
- transformers commit:
  `cc7ab9be508ce6ed3637bba9e50367b29b742dc6`
- upstream stable vLLM Docker reference:
  <https://github.com/verl-project/verl/blob/main/docker/Dockerfile.stable.vllm>

The local verl code already contains the upstream Qwen3.5 model patches
(`verl/models/transformers/qwen3_5.py` and the `qwen3_5` branches in
`verl/models/transformers/monkey_patch.py`). The remaining blocker is therefore
environment alignment, not a large verl code migration.

Use the staged installer because a single resolver transaction cannot represent
the official combo cleanly: the `vllm==0.18.0` wheel metadata still requires
`transformers<5`, while the official Qwen3.5 path uses the 5.x Qwen3.5
transformers code.

```bash
bash examples/minio3/install_qwen35_official_env.sh
```

For that reason, `requirements.txt` leaves vLLM to the staged installer. Running
a normal resolver pass with both `vllm==0.18.0` and unconstrained `transformers`
would silently downgrade the official transformers commit back to a 4.x release.

2026-05-20 local `.venv` update:

| Package | Version |
| --- | --- |
| `torch` | `2.10.0+cu128` |
| `vllm` | `0.18.0` |
| `transformers` | `5.3.0.dev0` from `cc7ab9be508ce6ed3637bba9e50367b29b742dc6` |
| `flash-attn` | `2.8.3`, rebuilt after the torch upgrade |
| `causal-conv1d` | `1.6.2.post1` |
| `flash-linear-attention` / `fla` | `0.5.0` |
| `qwen-vl-utils` | `0.0.14` |
| `numpy` | `2.2.6` |
| `pandas` | `3.0.3`, upgraded for numpy 2 ABI compatibility |

Verified locally:

- `examples/minio3/check_qwen35_env.py --model-path Qwen/Qwen3.5-9B
  --local-files-only` exits 0.
- `transformers.models.qwen3_5` is importable.
- `vllm.model_executor.models.qwen3_5` is importable.
- `flash-attn`, `causal-conv1d`, `flash-linear-attention`, and
  `qwen-vl-utils` import cleanly.
- `AsyncLLMEngine.abort` and vLLM v1 `AsyncLLM.abort` are present.
- `uv pip check` still reports the expected official-stack metadata conflict:
  `vllm` requires `transformers>=4.56.0,<5`, while the Qwen3.5 env uses
  `transformers==5.3.0.dev0`.

Target env checks before rerunning VP:

- `transformers.models.qwen3_5` is importable.
- `vllm.model_executor.models.qwen3_5` is importable.
- `AutoConfig.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)` loads
  `model_type=qwen3_5`.
- `flash-attn`, `causal-conv1d`, and `flash-linear-attention` import cleanly.
- vLLM request cancellation APIs remain available for Mini-o3 prompt admission.
