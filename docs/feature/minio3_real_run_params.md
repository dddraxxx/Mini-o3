# Mini-o3 Real Run Params

本文记录 Mini-o3 迁移到 official verl 后的真实训练和验证参数目标。这里的参数不同于 smoke 脚本；smoke 可以继续用更短的长度和 turn 数来快速检查代码链路。

## Token Budget 语义

`MAX_RESPONSE_LENGTH` 是一条 rollout 在 prompt 之后的总 response budget，不是一轮 assistant generation 的长度。

在 multi-turn tool rollout 中，这个总 budget 包括：

- assistant 生成 token
- crop tool observation 被插回对话后的 token
- 后续 assistant/tool turn 的 token

因此“train 16k”表示整条训练 rollout 的 response 区域最多 16k token；“val 32k”表示整条验证 rollout 的 response 区域最多 32k token。

## Train

真实训练默认目标：

```bash
MAX_RESPONSE_LENGTH=16384
MAX_ASSISTANT_TURNS=6
MAX_USER_TURNS=6
```

建议同时给 PPO/logprob 动态 batch 留足空间：

```bash
PPO_MAX_TOKEN_LEN_PER_GPU=32768
```

正式 train wrapper 继承旧 Mini-o3 常用 PyVision-style 脚本的其它参数：

- A100 profile: `examples/minio3/run_real_train_pyvision_style_a100.sh`
- H200 profile: `examples/minio3/run_real_train_pyvision_style_h200.sh`

这两个 wrapper 都保持 train 的 `MAX_ASSISTANT_TURNS=6`、`MAX_USER_TURNS=6`、`MAX_RESPONSE_LENGTH=16384`，并把其它参数对齐到旧脚本：

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
| `ROLLOUT_GPU_MEM_UTIL` | `0.9` | `0.9` |
| `ROLLOUT_FREE_CACHE_ENGINE` | `True` | `True` |
| `MAX_NUM_BATCHED_TOKENS` | `32768` | `32768` |
| `MAX_NUM_SEQS` | `128` | `256` |
| `SAVE_FREQ` | `10` | `10` |
| `SAVE_LORA_ONLY` | `True` | `True` |
| actor/ref offload | enabled | enabled |
| LoRA | rank `8`, alpha `16`, text-layer q/k/v/o/mlp regex | same |

LoRA runs save `actor/lora_adapter/adapter_model.safetensors` plus optimizer and extra state by default. Set
`SAVE_LORA_ONLY=False` only when a full sharded model checkpoint is needed.

## Val

真实验证默认目标：

```bash
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
