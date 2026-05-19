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
MINIO3_IGNORE_EXCEED=True
MINIO3_IGNORE_VOID=False
PROMPT_ADMISSION_ENABLE=True
PROMPT_ADMISSION_REWARD_STD_EPSILON=1.0e-4
```

`MINIO3_IGNORE_EXCEED=True` 表示撞到 response length 或 turn limit 的 trajectory 整条 `response_mask` 置 0，不参与 actor loss；`MINIO3_IGNORE_VOID=False` 表示缺少最终答案或 length stop 的 void trajectory 默认只记录指标，不直接清 loss。

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
| `ROLLOUT_DP` | `8` | `1` |
| `ROLLOUT_VLLM_EXECUTOR_BACKEND` | `uni` | unset |
| `ROLLOUT_SKIP_VLLM_DUMMY_LORA` | `True` | `False` |
| `ROLLOUT_GPU_MEM_UTIL` | `0.9` | `0.9` |
| `ROLLOUT_FREE_CACHE_ENGINE` | `True` | `True` |
| `MAX_NUM_BATCHED_TOKENS` | `32768` | `32768` |
| `MAX_NUM_SEQS` | `128` | `256` |
| Mini-o3 loss mask | `MINIO3_IGNORE_EXCEED=True`, `MINIO3_IGNORE_VOID=False` | same |
| prompt admission | enabled, std epsilon `1.0e-4`, state JSONL under `RUN_DIR` | same |
| `SAVE_FREQ` | `10` | `10` |
| `SAVE_LORA_ONLY` | `True` | `True` |
| logging | `train_step_metrics.jsonl`, `rollout_generations/`, `validation_generations/`, `train_samples.jsonl` | same |
| actor/ref offload | enabled | enabled |
| LoRA | rank `8`, alpha `16`, text-layer q/k/v/o/mlp regex | same |

LoRA runs save `actor/lora_adapter/adapter_model.safetensors` plus optimizer and extra state by default. Set
`SAVE_LORA_ONLY=False` only when a full sharded model checkpoint is needed.

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
