# Qwen3.5 H200 Rollout Concurrency TODO

This note records rollout-concurrency ablations for the Qwen3.5 official-tool
Mini-o3 RL profile. Do not treat these as committed defaults until measured.

## Current baseline

- `TRAIN_BATCH_SIZE=64`
- `ROLLOUT_N=8`
- `ROLLOUT_DP=8`
- `MAX_NUM_SEQS=256`
- `AGENT_NUM_WORKERS=64`
- `PROMPT_ADMISSION_POOL_SIZE=128`
- `RAY_NUM_CPUS=96`

With `pool_size=128` and `rollout_n=8`, the rollout side can have up to 1024
active sequences, or about 128 sequences per vLLM DP engine. This is only half
of the current `MAX_NUM_SEQS=256` per engine.

## Script cleanup before ablation

The current launch script has a conservative check:

```bash
TRAIN_BATCH_SIZE % AGENT_NUM_WORKERS == 0
```

For prompt-admission training this should not be required, because each prompt
group is submitted independently and repeated by `ROLLOUT_N`. Validation already
pads to the agent-worker divisor. Before testing worker counts above 64, change
the check so it only applies when `PROMPT_ADMISSION_ENABLE=False`, or replace it
with a validation-only compatibility check.

Do not change `TRAIN_BATCH_SIZE` just to satisfy this check; that would change
the RL batch/update workload and make the rollout-concurrency ablation unclear.

## Ablation tiers

### Tier 1: conservative H200 concurrency

Use this first.

```bash
AGENT_NUM_WORKERS=96
PROMPT_ADMISSION_POOL_SIZE=192
RAY_NUM_CPUS=128
```

Expected effect:

- About 1536 active rollout sequences total.
- About 192 sequences per vLLM DP engine.
- Keeps below `MAX_NUM_SEQS=256`.
- Agent workers average about two prompt groups per pool, similar to baseline.

### Tier 2: aggressive H200 concurrency

Try only if Tier 1 is stable and improves rollout utilization.

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
- If Tier 1 improves generation but still leaves vLLM under-filled, test Tier 2.
- If Tier 1 does not improve warm-step time, the bottleneck is likely not agent
  worker/pool concurrency, so next ablations should target vLLM batching or
  FSDP offload instead.
