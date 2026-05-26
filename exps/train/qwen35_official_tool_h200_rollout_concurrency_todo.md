# Qwen3.5 H200 Rollout Concurrency TODO

This note records rollout-concurrency ablations for the Qwen3.5 official-tool
Mini-o3 RL profile. Do not treat these as committed defaults until measured.

## Current default

- `TRAIN_BATCH_SIZE=64`
- `ROLLOUT_N=8`
- `ROLLOUT_DP=8`
- `MAX_NUM_SEQS=256`
- `AGENT_NUM_WORKERS=64`
- `PROMPT_ADMISSION_POOL_SIZE=160`
- `RAY_NUM_CPUS=96`
- `PPO_MICRO_BATCH_SIZE_PER_GPU=2`

With `pool_size=160` and `rollout_n=8`, prompt admission can keep a larger
oversampled queue than the original 128-pool baseline, while preserving the
64-agent-worker setting that previously reached optimizer steps.

## Script cleanup completed

The launch script previously had a conservative check:

```bash
TRAIN_BATCH_SIZE % AGENT_NUM_WORKERS == 0
```

For prompt-admission training this is not required, because each prompt group is
submitted independently and repeated by `ROLLOUT_N`. The check now only applies
when `PROMPT_ADMISSION_ENABLE=False`.

Do not change `TRAIN_BATCH_SIZE` just to satisfy this check; that would change
the RL batch/update workload and make the rollout-concurrency ablation unclear.

## Ablation tiers

### Tried: 96 workers

This was too aggressive on this node.

```bash
AGENT_NUM_WORKERS=96
PROMPT_ADMISSION_POOL_SIZE=192
RAY_NUM_CPUS=128
```

Observed result: Ray started but failed before training with
`Runtime Env Agent timed out in 30000ms`.

```bash
AGENT_NUM_WORKERS=96
PROMPT_ADMISSION_POOL_SIZE=192
RAY_NUM_CPUS=96
```

Observed result: vLLM and rollout began, but Ray spawned many `default_worker.py`
processes and repeatedly logged `Some workers ... have not registered within
the timeout`. Actual agent concurrency was far below the requested 96, GPU use
was uneven, and no train-step metrics were written before the run was stopped.

### Tried: 80 workers

This was still too aggressive for Ray worker startup.

```bash
AGENT_NUM_WORKERS=80
PROMPT_ADMISSION_POOL_SIZE=160
RAY_NUM_CPUS=96
PPO_MICRO_BATCH_SIZE_PER_GPU=2
```

Observed result:

- About 1280 active rollout sequences total.
- About 160 sequences per vLLM DP engine.
- LoRA update succeeded and prompt admission began.
- Ray still repeatedly logged worker registration timeouts.
- Only two active `AgentLoopWorker.generate_sequences` tasks were observed,
  with more than 100 `default_worker.py` processes.
- The run was stopped before step 1 because it was not a healthy formal run.

### Current compromise: 64 workers, larger admission pool

Use this first.

```bash
AGENT_NUM_WORKERS=64
PROMPT_ADMISSION_POOL_SIZE=160
RAY_NUM_CPUS=96
PPO_MICRO_BATCH_SIZE_PER_GPU=2
```

Expected effect:

- Keeps the agent worker count at the known-startable setting.
- Keeps a larger admission pool than the original 128-pool baseline.
- Keeps actor update memory lower than the micro-batch-4 pilot.

Observed in active run `qwen35_9b_official_tool_h200_rl_20260525_192131`:

- Reached step 18/200 without OOM.
- Passed the previous step-8 micro-batch-4 OOM point.
- Passed step 10 checkpoint and validation.
- Recent warm steps are roughly 445-452 seconds per step.
- `prompt_admission/submitted_per_accepted` is roughly 3.1-3.5.
- `batch/void_sample_ratio` remains high at roughly 0.97.
- vLLM EngineCore utilization can look low during phase transitions, but
  generation-phase pmon showed the EngineCore processes back at high SM
  utilization; average efficiency still needs follow-up ablation.

### Tier 2: aggressive H200 concurrency

Do not try this until the 64-worker / 160-pool compromise is measured and a
separate Ray worker-startup fix exists.

```bash
AGENT_NUM_WORKERS=128
PROMPT_ADMISSION_POOL_SIZE=256
RAY_NUM_CPUS=160
```

Expected effect:

- About 2048 active rollout sequences total.
- About 256 sequences per vLLM DP engine.
- Intentionally fills the current `MAX_NUM_SEQS=256`.
- Higher risk of Ray/CPU pressure, larger cancelled-rollout waste, and more
  concurrent DeepSeek reward traffic.

## Metrics to compare

Run a short formal-profile ablation, for example 20 steps, then compare only
warm steps after startup.

- `timing_s/gen`
- `timing_s/old_log_prob`
- `timing_s/update_actor`
- `timing_s/step`
- `perf/throughput`
- `perf/mfu/actor_infer`
- `prompt_admission/submitted_per_accepted`
- `prompt_admission/wait_timeouts`
- `prompt_admission/pending_groups`
- GPU util mean and median from `gpu_util.jsonl`, especially last 10 minutes

Decision rule:

- Keep the change if `timing_s/gen` and `timing_s/step` improve without a large
  increase in `submitted_per_accepted`, reward failures, Ray instability, or
  DeepSeek throttling.
- If the 64-worker / 160-pool compromise improves generation but still leaves
  vLLM under-filled, design a Ray-safe intermediate tier before considering
  Tier 2.
- If it does not improve warm-step time, the bottleneck is likely not agent
  worker/pool concurrency, so next ablations should target vLLM batching or
  FSDP offload instead.
