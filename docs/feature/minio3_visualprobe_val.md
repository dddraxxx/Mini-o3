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

The train parquet contains `SMOKE_TRAIN_CASES` rows, defaulting to
`TRAIN_BATCH_SIZE`. Even in val-only mode, verl builds the train dataloader and
expects it to have enough rows for the configured train batch.

## Current Smoke Candidate

These are the current 10-case smoke script defaults, not validated formal
defaults. The rollout-throughput values intentionally borrow the aggressive H200
script settings so the smoke can expose batching/GPU-util problems early.

| Param | Default |
| --- | --- |
| `MODEL_PATH` | `Qwen/Qwen3.5-9B` |
| `CHECK_QWEN35_ENV` | `True` |
| `SMOKE_CASES` | `10` |
| `SMOKE_TRAIN_CASES` | `TRAIN_BATCH_SIZE` |
| `TRAIN_BATCH_SIZE` | `8` |
| `PPO_MINI_BATCH_SIZE` | `TRAIN_BATCH_SIZE` |
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
| `SKIP_INITIAL_UPDATE_WEIGHTS` | `True` |
| `LOG_VAL_GENERATIONS` | `10` |
| `RAY_INCLUDE_DASHBOARD` | `False` |
| `RAY_NUM_CPUS` | `96` |

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
  candidates. The formal H200 full-val profile is listed separately below:
  `ROLLOUT_DP=8`, `AGENT_NUM_WORKERS=64`,
  `MAX_NUM_BATCHED_TOKENS=65536`, `MAX_NUM_SEQS=256`.
- `RAY_NUM_CPUS=96` is enough for the current H200 smoke profile. The 8
  rollout/FSDP placement-group bundles reserve at least 24 CPUs, and the 64
  AgentLoop workers need extra scheduler room.
- If the smoke fails before generation, first check image-token prompt length,
  model processor/chat template compatibility, and vLLM max model length.
- If the smoke runs but GPU util is low, check per-turn tool/image
  preprocessing time and request granularity before changing admission logic.
- In validation, `ray_trainer.py` pads each val batch to
  `actor_rollout_ref.rollout.agent.num_workers` before async rollout and unpads
  after generation. A 1-case smoke with `AGENT_NUM_WORKERS=4` therefore starts 4
  agent-loop trajectories, but writes one unpadded validation JSONL row.
- For val-only base-model evaluation with `LORA_RANK=0`, the smoke sets
  `trainer.skip_initial_update_weights=True`. This skips the initial FSDP to
  vLLM weight broadcast and also avoids sleeping the freshly loaded vLLM
  replicas. The trainer guards this flag so it is only valid for
  `trainer.val_only=True`, `lora_rank=0`, and `global_steps=0`.
- The cooperative GPU keeper can block Ray placement when it grabs GPU slots
  while Ray is launching all rollout actors. Stop it for formal val/training
  launches if Ray shows pending GPU placement groups.

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

2026-05-20 1-case FlashInfer-cache smoke:

- Run id:
  `visualprobe_val_smoke1_qwen35_9b_temp0_flashcache_20260520_012050`.
- Log:
  `logs/minio3_val_smoke1_flashcache_0520.log`.
- Run dir:
  `save/visualprobe_val_smoke1_qwen35_9b_temp0_flashcache_20260520_012050`.
- The run reached vLLM generation with Qwen3.5, produced
  `validation_generations/0.jsonl`, and wrote one metrics row to
  `train_step_metrics.jsonl`.
- Metrics are expected zero because this smoke used `empty_reward.py`; the model
  output answered the example text question but the empty reward reports
  `score=0.0`.
- `AGENT_NUM_WORKERS=4` padded the 1-case validation batch to 4 internal
  trajectories; only one unpadded row was dumped.

2026-05-20 30-case DeepSeek temp-1 smoke:

- Run id:
  `visualprobe_val_smoke30_qwen35_9b_temp1_deepseek_skipinit_cpu96_20260520_023500`.
- Log:
  `logs/minio3_vp30_deepseek_t1_skipinit_cpu96_0520.log`.
- Run dir:
  `save/visualprobe_val_smoke30_qwen35_9b_temp1_deepseek_skipinit_cpu96_20260520_023500`.
- Overrides: `SMOKE_CASES=30`, `SMOKE_TRAIN_CASES=8`,
  `VAL_BATCH_SIZE=30`, `VAL_N=1`, `VAL_DO_SAMPLE=True`,
  `VAL_TEMPERATURE=1.0`, `LOG_VAL_GENERATIONS=30`,
  `RAY_NUM_CPUS=96`, DeepSeek self-judge reward enabled.
- Confirmed progress: official Qwen3.5 env preflight passes with
  `accelerate==1.13.0`; FSDP actor load succeeds; vLLM DP=8 starts; CUDA graph
  capture completes; `trainer.skip_initial_update_weights=True` reaches
  `test_gen_batch`; 30 AgentLoop trajectories begin generation.
- Final status: wrapper exits 0, writes
  `validation_generations/0.jsonl`, `train_step_metrics.jsonl`, and
  `perf_debug_summary.json`. The vLLM EngineCore shutdown message appears after
  metrics are written and did not make the wrapper fail.
- Split coverage: 10 Easy, 10 Medium, 10 Hard.
- DeepSeek reward score mean: overall 0.30; Easy 0.60; Medium 0.10; Hard 0.20.
- Mean tool calls: overall 0.90; Easy 0.90; Medium 0.40; Hard 1.40.
- Mean turns: 2.87; min 2; max 16.
- One generation emitted a Mini-o3 grounding call with a bare identifier that
  the old JSON/Python-literal parser could not decode. The parser now falls
  back to `yaml.safe_load` for this common `{bbox_2d: [...], source:
  original_image}` format.

2026-05-25 30-case Qwen3.5 official-tool empty-reward smoke:

- First attempted H200-wide scheduler profile:
  `RAY_NUM_CPUS=128`, `AGENT_NUM_WORKERS=128`.
- Attempted run id:
  `visualprobe_val_smoke30_qwen35_9b_official_tool_empty_h200_20260525_012232`.
- Attempted run log:
  `logs/visualprobe_val_smoke30_qwen35_9b_official_tool_empty_h200_20260525_012232.log`.
- This attempt reached local Ray startup, but never reached `TaskRunner`.
  `ray status` showed 0 CPU / 0 GPU usage and raylet logged many worker
  registration timeouts from prestarting 128 Python workers. Treat `128/128`
  as not yet validated until Ray worker prestart/concurrency is tuned.
- Successful rerun id:
  `visualprobe_val_smoke30_qwen35_9b_official_tool_empty_cpu96_20260525_012811`.
- Successful rerun log:
  `logs/visualprobe_val_smoke30_qwen35_9b_official_tool_empty_cpu96_20260525_012811.log`.
- Successful rerun dir:
  `save/visualprobe_val_smoke30_qwen35_9b_official_tool_empty_cpu96_20260525_012811`.
- Overrides: `SMOKE_CASES=30`, `SMOKE_TRAIN_CASES=8`,
  `VAL_BATCH_SIZE=30`, `VAL_N=1`, `VAL_DO_SAMPLE=True`,
  `VAL_TEMPERATURE=1.0`, `RAY_NUM_CPUS=96`, `AGENT_NUM_WORKERS=64`,
  `MINIO3_TOOL_PROMPT_SUITE=qwen35_official_zoom_tool`,
  `MINIO3_OFFICIAL_TOOL_NAME=image_zoom_in_tool`,
  `ROLLOUT_MULTI_TURN_FORMAT=qwen3_coder`, empty reward.
- Confirmed: prompt contains Qwen3.5 `# Tools` block and
  `image_zoom_in_tool` schema; generated calls use official `<tool_call>`
  syntax; dumped validation JSONL has 60 `<tool_call>` blocks, 60
  `<tool_response>` blocks, and 0 legacy `<grounding>` blocks across 30 rows.
- Final status: wrapper exits 0 and writes
  `validation_generations/0.jsonl`, `train_step_metrics.jsonl`, and
  `perf_debug_summary.json`. vLLM logged `EngineCore_DP0 died unexpectedly`
  after validation generations and metrics were already dumped.
- Split coverage: 10 Easy, 10 Medium, 10 Hard.
- Empty reward metrics are expected zero. Tool-call means from trainer metrics:
  Easy 1.9, Medium 2.2, Hard 1.9. Number of turns: min 4, max 14, mean 6.0.
- Format issue from output inspection: only 12 / 30 rows include final
  `<answer>...</answer>` tags, despite the prompt asking for them. Manual rough
  read of final answers was about 17 / 30 correct overall: Easy 9 / 10,
  Medium 4 / 10, Hard 4 / 10. This is a spot check, not a reward score.
- Prompt variant for relaxed/LLM-judge eval:
  `MINIO3_TOOL_PROMPT_SUITE=qwen35_official_zoom_tool_plain_question` keeps the
  official `image_zoom_in_tool` path but removes the `<answer>...</answer>`
  instruction. The user prompt remains just image token(s) followed by the raw
  question.

## Formal Val Profiles

For formal validation, keep the total number of val trajectories in the same
range as train rollout:

```text
train trajectory count = TRAIN_BATCH_SIZE * ROLLOUT_N = 64 * 8 = 512
val trajectory count = VAL_BATCH_SIZE * VAL_N = 512 * 1 = 512
```

The current val path does not fan out with train `ROLLOUT_N`; it repeats with
`VAL_N` and then pads the validation batch to `AGENT_NUM_WORKERS`.

H200 formal full-val profile:

```bash
RAY_NUM_CPUS=128
AGENT_NUM_WORKERS=128
VAL_BATCH_SIZE=512
VAL_N=1
```

With `AGENT_NUM_WORKERS=128`, `VAL_BATCH_SIZE=512`, and `VAL_N=1`, the batch
remains 512 trajectories after padding and gives 4 val trajectories per
agent-loop worker. On the local 8x H200 node, this keeps the total eval size
aligned with train rollout while using a wider CPU/Ray scheduler budget. The
request burst is still below the H200 script's `MAX_NUM_SEQS=256`.

A100 full-val profile:

```bash
RAY_NUM_CPUS=96
AGENT_NUM_WORKERS=64
VAL_BATCH_SIZE=512
VAL_N=1
```

This keeps the same `VAL_BATCH_SIZE * VAL_N = 512` eval size, but uses a
narrower agent/Ray profile for A100-class runs.

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
| `accelerate` | `1.13.0` |
| `flashinfer-python` | `0.6.6` |
| `flashinfer-jit-cache` | `0.6.6+cu129` |
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
- `flashinfer show-config` reports matching
  `flashinfer-python==0.6.6` and `flashinfer-jit-cache==0.6.6+cu129`; the GDN
  prefill op executes on H200 without local JIT compilation.
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

FlashInfer note:

- Plain local FlashInfer JIT on this node tried to build GDN kernels against the
  system CUDA 12.4 toolchain and failed on newer CCCL/PTX symbols.
- Do not install a mismatched JIT cache package: `flashinfer-jit-cache 0.6.11`
  fails the package-version check with `flashinfer-python 0.6.6`.
- The reproducible path is the matched cache package from the FlashInfer cu129
  wheel index:
  `uv pip install flashinfer-jit-cache==0.6.6+cu129 --index-url https://flashinfer.ai/whl/cu129`.
- If FlashInfer regresses again, the fallback to investigate is vLLM Qwen3.5
  `additional_config.gdn_prefill_backend=triton`; keep FlashInfer enabled while
  the matched cache works.
