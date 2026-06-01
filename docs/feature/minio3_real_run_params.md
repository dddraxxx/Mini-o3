# Mini-o3 Real Run Params

本文记录 Mini-o3 迁移到 official verl 后的真实训练和验证参数目标。这里的参数不同于 smoke 脚本；smoke 可以继续用更短的长度和 turn 数来快速检查代码链路。

## Token Budget 语义

`MAX_RESPONSE_LENGTH` 是一条 rollout 在 prompt 之后的总 response budget，不是一轮 assistant generation 的长度。

在 multi-turn tool rollout 中，这个总 budget 包括：

- assistant 生成 token
- crop tool observation 被插回对话后的 token
- 后续 assistant/tool turn 的 token

因此“train 16k”表示整条训练 rollout 的 response 区域最多 16k token；“val 32k”表示整条验证 rollout 的 response 区域最多 32k token。

`MAX_PROMPT_LENGTH` 对应 verl 的 `data.max_prompt_length`，是 prompt-side budget。启动 dataloader 时的 overlong prompt filter 会按这个值过滤样本；对 VL 数据，这一步会走 multimodal processor，所以不是纯文本 tokenization。

当前 Mini-o3/Qwen2.5-VL wrapper 默认：

```bash
MAX_PROMPT_LENGTH=8192
```

Qwen3.5 的 chat template / vision prompt 可能更长，Qwen3.5 real run 可以先按更大的 prompt budget 准备，例如：

```bash
MAX_PROMPT_LENGTH=16384
```

如果提高 `MAX_PROMPT_LENGTH`，必须同步检查：

- `MAX_MODEL_LEN >= MAX_PROMPT_LENGTH + max(MAX_RESPONSE_LENGTH, VAL_RESPONSE_LENGTH)`
- train 的 `PPO_MAX_TOKEN_LEN_PER_GPU` 至少覆盖 `MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH`
- val 的 token budget 至少覆盖 `MAX_PROMPT_LENGTH + VAL_RESPONSE_LENGTH`

例如 Qwen3.5 使用 `MAX_PROMPT_LENGTH=16384`、train response 16k、val response 32k 时，`MAX_MODEL_LEN` 至少要到 `49152`。

## Train

真实训练默认目标：

```bash
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=16384
MAX_ASSISTANT_TURNS=6
MAX_USER_TURNS=6
MINIO3_IGNORE_CLIP=False
MINIO3_IGNORE_EXCEED=False
MINIO3_IGNORE_FORMAT=False
MINIO3_IGNORE_INVALID=False
PROMPT_ADMISSION_ENABLE=True
PROMPT_ADMISSION_REWARD_STD_EPSILON=1.0e-4
```

Mini-o3 失败 logging 使用 `clip/exceed/format/invalid`：`clip` 表示 response 长度打满，`exceed` 表示 tool/multi-turn 还想继续但撞到硬预算，`format` 表示终止时缺少合法 `Final answer:`，`invalid` 是三者 union。`MINIO3_IGNORE_*` 开关控制是否把对应 row 的整条 `response_mask` 置 0；正式 Qwen3.5 run 默认只记录这些指标，不清 loss。

`PROMPT_ADMISSION_ENABLE=True` 表示训练前先按 prompt group 做 admission：同一 prompt 的 `ROLLOUT_N` 条 response reward 必须有组内方差，否则不进入 GRPO update。

建议同时给 PPO/logprob 动态 batch 留足空间：

```bash
PPO_MAX_TOKEN_LEN_PER_GPU=32768
```

正式 train wrapper 继承旧 Mini-o3 常用 PyVision-style 脚本的其它参数：

- A100 profile: `examples/minio3/run_real_train_pyvision_style_a100.sh`
- H200 profile: `examples/minio3/run_real_train_pyvision_style_h200.sh`

这两个 wrapper 都保持 train 的 `MAX_PROMPT_LENGTH=8192`、`MAX_ASSISTANT_TURNS=6`、`MAX_USER_TURNS=6`、`MAX_RESPONSE_LENGTH=16384`，并把其它参数对齐到旧脚本。Qwen3.5 run 可以单独覆盖 `MAX_PROMPT_LENGTH`，但要按上面的公式同步放大 `MAX_MODEL_LEN` 和 token budget：

| 参数 | A100 profile | H200 profile |
| --- | --- | --- |
| train data | `minio3_real_subset` | train `minio3_full`, val `minio3_real_subset` |
| `TRAIN_BATCH_SIZE` | `64` | `64` |
| `PPO_MINI_BATCH_SIZE` | `16` | `16` |
| `PPO_MICRO_BATCH_SIZE_PER_GPU` | `2` | `4` |
| `LOG_PROB_MICRO_BATCH_SIZE_PER_GPU` | `8` | `16` |
| `REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU` | `8` | `16` |
| `ROLLOUT_N` | `8` | `8` |
| `ROLLOUT_TP` | `1` | `1` |
| `ROLLOUT_DP` | `8` | `8` |
| `ROLLOUT_VLLM_EXECUTOR_BACKEND` | `uni` | `uni` |
| `ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE` | `True` | `False` |
| `ROLLOUT_SKIP_VLLM_DUMMY_LORA` | `True` | `True` |
| `ROLLOUT_GPU_MEM_UTIL` | `0.9` | `0.9` |
| `ROLLOUT_FREE_CACHE_ENGINE` | `True` | `True` |
| `MAX_NUM_BATCHED_TOKENS` | `49152` | `65536` |
| `MAX_NUM_SEQS` | `256` | `256` |
| `AGENT_NUM_WORKERS` | `32` | `64` |
| reward backend | rule reward | DeepSeek self-judge by default |
| `SELF_JUDGE_MODEL` | unset | `deepseek-v4-flash` |
| `LLM_JUDGE_MAX_RETRIES` | unset | `5` |
| Mini-o3 loss mask | `MINIO3_IGNORE_CLIP=False`, `MINIO3_IGNORE_EXCEED=False`, `MINIO3_IGNORE_FORMAT=False`, `MINIO3_IGNORE_INVALID=False` | same |
| prompt admission | enabled, std epsilon `1.0e-4`, state JSONL under `RUN_DIR` | same |
| `SAVE_FREQ` | `10` | `10` |
| `TEST_FREQ` | `5` | `10` |
| `SAVE_LORA_ONLY` | `True` | `True` |
| logging | `train_step_metrics.jsonl`, `rollout_generations/`, `validation_generations/`, `train_samples.jsonl`, `gpu_util.jsonl`, `perf_debug_summary.json` | same, plus `MINIO3_STAGE_LOG=1` |
| actor/ref offload | enabled | enabled |
| LoRA | rank `8`, alpha `16`, text-layer q/k/v/o/mlp regex | same |

LoRA runs save `actor/lora_adapter/adapter_model.safetensors` plus optimizer and extra state by default. Set
`SAVE_LORA_ONLY=False` only when a full sharded model checkpoint is needed.

GPU util 原始采样默认写入 `gpu_util.jsonl`；调速辅助汇总默认写入
`perf_debug_summary.json`。这两个文件只用于 throughput/debug，不作为模型质量训练结果。
汇总方式见
[minio3_gpu_monitoring.md](minio3_gpu_monitoring.md)。

H200 profile 当前按本地 8x H200 正式长训默认更激进：`ROLLOUT_DP=8`、vLLM
`uni` backend、开启 vLLM mm preprocessor cache、跳过 dummy LoRA、
`MAX_NUM_BATCHED_TOKENS=65536`、`MAX_NUM_SEQS=256`、`AGENT_NUM_WORKERS=64`、
`TOTAL_TRAINING_STEPS=100`、`TEST_FREQ=10`。H200 正式长训默认使用 DeepSeek
self-judge reward：`SELF_JUDGE_REWARD=True`、`SELF_JUDGE_PROVIDER=deepseek`、
`SELF_JUDGE_MODEL=deepseek-v4-flash`、`SELF_JUDGE_MAX_TOKENS=8`、
`SELF_JUDGE_TEMPERATURE=0.0`、`LLM_JUDGE_MAX_RETRIES=5`；需要在运行环境里提供
`DEEPSEEK_API_KEY`。默认 `RUN_DIR` 带时间戳，避免正式长训
复用旧 checkpoint 目录；同时打开 `MINIO3_STAGE_LOG=1` 和
`MINIO3_TRAJ_STATUS_INTERVAL_S=15`，让 `perf_debug_summary.json` 能汇总 admission/worker
load timeline。如果 vLLM profile/dummy stage 或长训早期出现 CUDA illegal memory
access / OOM，第一回退档是只把 `MAX_NUM_BATCHED_TOKENS=49152`，其它 H200 参数保持
不变；第二回退档才考虑 `ROLLOUT_DP=1`。

2026-05-19 A100 单步对比结论：在 `TRAIN_BATCH_SIZE=64`、`ROLLOUT_N=8`、
禁 validation/checkpoint、强制 admission accept 的设置下，`MAX_NUM_SEQS=256`
比 `512` 更快更稳；在 `MAX_NUM_SEQS=256` 基础上，`MAX_NUM_BATCHED_TOKENS=49152`
比 `32768` 略快且可以稳定完成一整个 update。因此 A100 profile 当前默认固定为：

```bash
MAX_NUM_SEQS=256
MAX_NUM_BATCHED_TOKENS=49152
```

对比数据：

| run | result | `timing_s/gen` | `timing_s/step` | `perf/throughput` | active GPU util | load-balance note |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `mns256_bs64_noval_20260519_152052` (`32768/256`) | pass | `462.18s` | `733.37s` | `357.37 tok/s/GPU` | mean `43.25`, p50 `32`, p95 `99` | no parsed stage-log timeline |
| `mns512_bs64_noval_20260519_154446` (`32768/512`) | pass | `469.01s` | `751.20s` | `344.22 tok/s/GPU` | mean `41.01`, p50 `29`, p95 `99` | no parsed stage-log timeline |
| `mbt49152_mns256_bs64_20260519_165749` (`49152/256`) | pass | `439.48s` | `713.44s` | `362.38 tok/s/GPU` | mean `41.90`, p50 `29`, p95 `99` | `prompt_load/running_groups` mean `46.62`, p50 `49`, p95 `64`, max `64`; `max_worker_inflight` mean `1.90`, max `2`; `nonzero_workers` mean `27.28`, p95 `32`; `traj_active_ratio` mean `0.63`, p95 `1.00` |
| `mbt65536_mns256_bs64_20260519_164656` (`65536/256`) | fail | n/a | n/a | n/a | no useful training window | failed before prompt admission during vLLM profile/dummy run with CUDA illegal memory access |

`PROMPT_ADMISSION_POOL_SIZE=96` was tested separately on the same A100 profile. The attempted
destructive-cancel variant filled 64 accepted prompt groups but left 95 unfinished groups, then
`abort_replicas()` killed vLLM DP EngineCore during `all_reduce` and `update_weights` failed with
`EngineDeadError`. Current A100 formal default therefore stays with no oversubmit:
`PROMPT_ADMISSION_POOL_SIZE` unset, which means `pool_size=train_batch_size=64`. Larger admission
windows are experimental until vLLM request abort is safe for DP=8 or the rollout path becomes fully
async across the actor update.

The follow-up no-oversubmit run
`admission_window_default_mbt49152_mns256_bs64_20260519_195729` completed with `exit_code=0`.
It used the current formal A100 knobs (`MAX_NUM_BATCHED_TOKENS=49152`, `MAX_NUM_SEQS=256`,
`TRAIN_BATCH_SIZE=64`, `PROMPT_ADMISSION_POOL_SIZE` unset) and real prompt admission. It filled
`64` accepted groups after `145` submitted / `81` rejected groups, with `cancelled_running_groups=0`
and `response/aborted_ratio=0`. Timings were `gen=1006.62s`, `old_log_prob=102.73s`,
`update_actor=146.10s`, `update_weights=18.65s`, `step=1275.75s`, with
`perf/throughput=185.53` and `perf/mfu/actor_infer=0.411`. This is slower than forced-admit tuning
runs because it includes rejection sampling overhead, but it verifies the safe path can fill the
accepted batch and enter backward/update cleanly.

`49152/256` 的 `perf/mfu/actor_infer=0.433` 低于 `32768/256` 的 `0.476`，但端到端
`gen` 和 `step` 更短，且 load timeline 显示 64 个 prompt group 可以填满 admission 后迅速进入
`old_log_prob/update_actor`。当前调参优先级是保留 `49152/256`，后续再围绕 tool/image 处理和采样请求粒度优化 generation throughput。

`ROLLOUT_SKIP_VLLM_DUMMY_LORA=True` is an A100 DP=8 LoRA workaround for vLLM V1 spawn workers.
The launch script adds the repo root to `PYTHONPATH`, and `sitecustomize.py` patches vLLM dummy
LoRA activation only when `VERL_VLLM_SKIP_DUMMY_LORA` is set. Without this child-process patch,
`EngineCore_*` can still execute `maybe_dummy_run_with_lora()` after prompt admission fills and
crash in `vllm/lora/layers.py:set_lora()` with CUDA illegal memory access.

## Val

真实验证默认目标：

```bash
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=32768
MAX_ASSISTANT_TURNS=12
MAX_USER_TURNS=12
```

建议验证时给 PPO/logprob token budget 留更大余量，至少覆盖 prompt + response：

```bash
PPO_MAX_TOKEN_LEN_PER_GPU=49152
```

如果 `MAX_PROMPT_LENGTH` 后续提高，`PPO_MAX_TOKEN_LEN_PER_GPU` 也要相应提高。

在同一个训练 run 里做 validation 时，official verl 迁移层使用 validation-only overrides：

```bash
VAL_RESPONSE_LENGTH=32768
VAL_MAX_ASSISTANT_TURNS=12
VAL_MAX_USER_TURNS=12
```

因此 train batch 仍然按 16k/6 turn 更新，validation batch 单独按 32k/12 turn 生成。
