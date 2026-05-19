# Mini-o3 to Official verl Migration

## 目标

这个迁移分支以 `verl-upstream/main` 为底座，只把 Mini-o3 独有能力作为
feature overlay 搬进来。不要再从旧 fork 整块复制 `vllm_async_engine.py`、
worker glue 或 Qwen2.5-VL 兼容补丁。

目标能力分两层：

1. 保留 Mini-o3 training 特色：
   - `<grounding>{...}</grounding>` 多轮 crop 行为。
   - crop 后把 zoom-in image 作为 observation 继续推理。
   - LoRA RL、vLLM adapter reload、rejection sampling/prompt stream。
   - JSONL 样本、step metrics、trajectory artifact。
2. 接到新版 official verl：
   - Qwen3/Qwen3.5/Qwen3-VL 模型兼容走 official verl。
   - 多轮工具调用优先走 `AgentLoop` / `BaseTool`。
   - async rollout/server 先复用官方 server mode，再逐步加 Mini-o3 的全局
     ready-turn 调度和 active trajectory cap。

## 当前迁移分支

- 分支：`port/minio3-on-verl-main`
- 底座：`verl-upstream/main` at `2eab9364`
- 旧 Mini-o3 参考分支：`code/qwen35-9b-lora`

## 第一批迁移内容

第一批先迁移最小可运行的 Mini-o3 多轮视觉工具链：

| 能力 | 新版 verl 接入点 | 迁移方式 |
| --- | --- | --- |
| `<grounding>` legacy 格式 | `verl.experimental.agent_loop.tool_parser.ToolParser` | 新增 `minio3_grounding` parser，把 grounding tag 转成 `tool_crop` 调用。 |
| crop 工具 | `verl.tools.BaseTool` | 新增 `MiniO3CropTool`，从当前 trajectory 的 image/observation 中裁剪区域并返回 image。 |
| Mini-o3 observation 语义 | `ToolAgentLoop` | 新增 `mini_o3_tool_agent`，让 tool response 以 user observation 形式进入下一轮，而不是普通 OpenAI tool role。 |
| Mini-o3 JSON 数据 | official `RLHFDataset` | 新增 preprocess，把旧 `problem/images/solution` JSON 转成 official parquet。 |
| Mini-o3 rule reward | `reward.custom_reward_function` | 新增轻量 reward，抽 `<answer>` 并兼容 A/B/C/D 和 exact match。 |
| official verl 训练脚本 | `examples/grpo_trainer` 风格 | 新增 `examples/minio3/run_qwen3_vl_8b_crop_lora_fsdp.sh`。 |

这一步不迁移旧 fork 的 ready-turn queue，也不迁移旧 vLLM internals。原因是
official verl 已经有 server-mode `AgentLoopManager`、sticky-session load balancer、
fully async policy 和 Qwen3/Qwen3.5/Qwen3-VL 例子。Mini-o3 的 queue 应该作为
官方 async server 之上的调度扩展迁入。

## 后续迁移顺序

1. 跑通 parser/tool/agent loop 的 CPU import 和小单元测试。
2. 用一条单样本 image prompt 验证 `<grounding>` 会触发 crop，并且下一轮会看到
   observation image。
3. 用 `trainer.val_only=True` 跑 VisualProbe/Geo3K 风格小 eval。
4. 打开 LoRA：`actor_rollout_ref.model.lora_rank > 0`，验证 vLLM adapter reload。
5. 迁移 Mini-o3 JSONL artifacts：`train_samples.jsonl`、
   `train_step_metrics.jsonl`、`cases.jsonl`。
6. 迁移 rejection sampling / prompt stream。
7. 在 official async server mode 上实现 Mini-o3 global ready-turn scheduler：
   - 全局 ready turn queue。
   - inflight 上限。
   - active trajectory cap。
   - timeseries trace。
   - GPU util / generation latency metrics。

## 不应该原样搬的内容

- 旧 fork 的 Qwen2.5-VL token id、chat template、image token hardcode。
- 旧 `vllm_async_engine.py` 内的大块 generation loop。
- 旧 SPMD rollout glue。
- 旧 packed-weight/LoRA 兼容 patch，除非新版 official verl 仍然复现同一问题。

## 第一批验收标准

- `MiniO3GroundingToolParser` 可以从模型输出中解析 `bbox_2d` 和 `source`。
- `MiniO3CropTool` 可以从原图或 `observation_i` 里裁剪并返回 `ToolResponse(image=[...])`。
- `MiniO3ToolAgentLoop` 会把 crop 结果作为 user observation 加回下一轮。
- `examples/minio3/preprocess_visualprobe.py` 会生成 official verl parquet。
- `examples/minio3/minio3_reward.py` 会给出 Mini-o3 answer/tag/tool-call reward。
- `examples/minio3/run_qwen3_vl_8b_crop_lora_fsdp.sh` 使用 official verl 配置入口，
  没有引用旧 fork 的 `vllm_async_engine.py`。

## 当前验证结果

2026-05-18 已完成第一批 smoke/fixbug loop：

- smoke 命令：`run_logs/minio3-official-smoke.cmd.sh`
- smoke 日志：`run_logs/minio3-official-smoke.log`
- smoke 模型：`Mini-o3/Mini-o3-7B-SFT`
- smoke 数据：`data/minio3_tiny/{train,val}.parquet`
- smoke 设置：1 GPU、FSDP2、LoRA rank 8、official async vLLM rollout、
  `mini_o3_tool_agent`、`minio3_grounding`、`tool_crop`、GRPO。

通过项：

- parser/tool/agent observation 静态和轻量单测通过。
- tiny parquet 预处理和 overlong prompt filtering 通过。
- official vLLM async server 启动通过。
- LoRA weight sync 到 vLLM 的 `update_weights` 路径通过。
- 多轮 AgentLoop 生成、reward、old log prob、ref log prob、actor update 全部跑通。
- 最终日志打印 `step:1` 和 `Final validation metrics: None`，无 traceback。

2026-05-18 追加完成 eval smoke/fixbug loop：

- eval 命令：`run_logs/minio3-official-eval-smoke.cmd.sh`
- eval 日志：`run_logs/minio3-official-eval-smoke.log`
- eval 产物：`save/minio3_official_verl_eval_smoke/validation_generations/0.jsonl`
- eval 设置：`trainer.val_before_train=True`、`trainer.val_only=True`、
  `actor_rollout_ref.rollout.val_kwargs.n=1`、greedy validation generation。

通过项：

- `_validate()` 路径完成 validation generation。
- validation generation JSONL 写出 1 条样本。
- eval 生成触发 `<grounding>`，并记录 `tool_call_count/mean@1: 1.0`。
- 日志打印 `Initial validation metrics` 和 `step:0 - val-*` 指标，无 traceback。
- 当前 tiny smoke 指标为 `val-core/visual_probe_easy/acc/mean@1: 0.0`，
  仅代表这条 tiny 样本上模型没有答对，不代表 eval 链路失败。

本轮 smoke 为迁移路径验证，不代表完整长训配置。正式 Qwen3/Qwen3.5-VL
训练还需要升级到对应模型受支持的 transformers/vLLM 组合后再跑长训 smoke。
第二批还要继续迁移 Mini-o3 的训练 artifacts、rejection sampling/prompt stream，
以及 official async server 上的 global ready-turn scheduler。

2026-05-19 追加完成 Qwen3.5 LoRA rollout `dp=8` smoke：

- 默认 vLLM `mp` executor 已确认不可靠：server 参数能正确传到
  `--data_parallel_size 8 --data_parallel_size_local 8`，但 EngineCore 启动阶段会在
  torch distributed 内部端口上报 `EADDRINUSE` / `address already in use`。
- 改用 `+actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=uni`
  后，8 个 `EngineCore_DP0..DP7` 正常启动，server 参数包含
  `--distributed_executor_backend uni`、`--enable_lora`、`--enable_prefix_caching`、
  `--max_num_seqs 128`、`--max_num_batched_tokens 32768`。
- no-val smoke 跑完 1 training step：`run_logs/qwen35_lora_dp8_uni_noval_smoke_20260519_000724.log`。
  配置为 `ROLLOUT_TP=1`、`ROLLOUT_DP=8`、`MAX_RESPONSE_LENGTH=128`、
  `MAX_ASSISTANT_TURNS=1`、`TEST_FREQ=-1`、`trainer.total_training_steps=1`。
- 保存产物：
  `save/qwen35_lora_dp8_uni_noval_smoke_20260519_000724/global_step_1/actor/lora_adapter/adapter_model.safetensors`。
  `latest_checkpointed_iteration.txt` 为 `1`，`global_step_1` 约 `165M`，LoRA adapter 约 `49M`。
- 训练日志打印 `Training Progress: 100%`、`step:1` 和
  `Final validation metrics: None`。退出尾部存在 wandb atexit closed transport 和 vLLM shutdown
  `EngineCore_DP0 died unexpectedly` 日志；这是训练结束后 Ray/vLLM 清理阶段噪声，不影响 step/checkpoint
  验收。
- 注意：`trainer.test_freq=9999` 仍会因为 `is_last_step` 触发最终 validation；smoke 如果不想跑
  96-case val，需要显式使用 `trainer.test_freq=-1`。



## logging

2026-05-19 已把 formal train logging 和旧 PyVision-style Mini-o3 对齐到可用状态。A100/H200 wrapper
默认打开：

```bash
LOGGER_BACKENDS='["console","wandb","file"]'
VERL_FILE_LOGGER_PATH="$RUN_DIR/train_step_metrics.jsonl"
ROLLOUT_DATA_DIR="$RUN_DIR/rollout_generations"
VALIDATION_DATA_DIR="$RUN_DIR/validation_generations"
TRAIN_SAMPLES_JSONL="$RUN_DIR/train_samples.jsonl"
TRAIN_SAMPLES_JSONL_LIMIT=16
```

对应产物：

| 文件 / 目录 | 用途 |
| --- | --- |
| `$RUN_DIR/train_step_metrics.jsonl` | file logger 写出的 step metrics。 |
| `$RUN_DIR/rollout_generations/{step}.jsonl` | official verl rollout dump，已补 `uid`、`data_source`、`image_paths`。 |
| `$RUN_DIR/validation_generations/{step}.jsonl` | official verl validation dump。 |
| `$RUN_DIR/train_samples.jsonl` | 旧 Mini-o3 兼容 train sample append log，每 step 默认最多 16 条。 |

`train_samples.jsonl` 不保存图片 bytes，只保存 image path 引用，格式保持旧版核心字段：

```json
{
  "step": 1,
  "uid": "...",
  "data_source": "...",
  "image_paths": ["..."],
  "ground_truth": "...",
  "input": "...",
  "output": "...",
  "score": 0.0,
  "reward": 0.0
}
```

`image_paths` 来源优先级：`extra_info.image_paths`，fallback 到 `raw_prompt` 中的 image 字段。
`examples/minio3/preprocess_visualprobe.py` 现在会把 resolved image path 写进 `extra_info.image_paths`，
所以新生成的 real train parquet 可以直接回看原图。


## some still not migrated mini-o3 features

1. Rejection sampling

Mini-o3 旧的 prompt-stream rejection sampling 没完整迁移。

这可以实现，但要把“dataloader 一次给一个固定 batch”的假设改成“rollout manager 自己循环拉 prompt group，直到 accepted
 batch 满”。关键需要三块状态：

1. `PromptGroup`
每个 group 至少要有：
- `group_id`
- prompt rows / uid list
- rollout_n
- status: `pending | running | accepted | rejected | used`
- attempts
- generated trajectories
- score / reward stats
- created_step / last_update_step

2. `PromptStream`
负责跨 step 保存 group 状态：
- 新 prompt group 从 dataloader 来
- 未决 group 留在内存队列里
- rejected group 记录状态，是否重采取决于策略
- accepted group 进入当前 train batch
- used group 不再进入训练

3. `AcceptedBatchBuilder`
每个 trainer step 调它：
- 持续 dispatch group
- 并发跑 AgentLoop
- 收集完成 group
- 跑 reward / rejection rule
- 直到 accepted trajectories 数量满足 `train_batch_size`
- 然后组装成一个 `DataProto` 返回给 trainer

主要难点不是 vLLM，而是 trainer/dataflow 语义：

- 当前 `ray_trainer.py` 通常假设一个 dataloader batch 对应一个 training step。
- 如果 prompt group 会跨 step pending，就不能只存在当前 local batch 变量里，要放在一个 stateful manager 里。
- 如果一个 group rollout 完但没有 accept/reject，必须持久保存，否则下一步会丢状态。
- 多 worker 下 group status 更新要集中管理，不能各 worker 各自改，否则会重复用 prompt。

我建议实现顺序：

1. 先做单进程/单 trainer actor 内的 `PromptGroupStateManager`
只在 trainer driver 上维护 status，先不要分布式持久化。这个足够支持单 node run。

2. 把 official AgentLoopManager 包一层
不要改 vLLM server 本身。新增一个 Mini-o3 rollout collection loop，在调用 official async rollout 外面做循环和 accept
/reject。

3. 先只支持三种状态
`running / accepted / rejected`。`pending across step` 可以加，但第一版可以把没决策的当作 pending queue 留内存，不落
盘。

4. 加最小 checkpoint
每 step 写一个 `prompt_stream_state.jsonl` 或 `prompt_stream_state.pt`，包含 group id、status、attempts、uid。这样中
断恢复有基础。

5. 再接 rejection rule
先用旧 Mini-o3 的 group-level rule，比如一个 prompt group 里 reward 全 0 / 全 1 / effective ratio 不满足，就 reject
或重采。

所以答案是：**可以做，而且不需要重写 vLLM server；应该在 official async rollout 外层加一个 stateful prompt-stream sc
heduler。**

我会把它设计成：

```text
Trainer Step
  |
  v
MiniO3PromptStreamRolloutCollector
  |
  +--> PromptGroupStateManager
  |       pending / running / accepted / rejected / used
  |
  +--> Official AgentLoopManager
  |       dispatch group -> async trajectories
  |
  +--> Reward + Rejection Policy
  |
  v
accepted DataProto batch
```

这样既保留 official verl 的 AgentLoop/vLLM DP server，又恢复旧 Mini-o3 的 prompt group rejection semantics。

2. out-of-turn-limit loss mask
  这个也基本是缺的，至少不是旧 Mini-o3 等价实现。

  旧 Mini-o3 关键点是：

  - rollout 里产出 multi_turn_response_mask
  - actor 里有 use_multi_turn_response_mask=True
  - dp_actor.py 会用 multi_turn_response_mask 替代普通 response mask
  - 还会对 exceed_mask / void_mask 把 loss mask 清零
  - 旧脚本默认开：actor_rollout_ref.actor.use_multi_turn_response_mask=True

  当前 official verl 迁移版有 response_mask，而且工具 observation token 会标成 0，这部分是对的；但我没看到旧 Mini-o3
  那种 actor 侧 use_multi_turn_response_mask + exceed/void loss 清零逻辑被移植进来。

  所以结论是：

  普通 tool observation loss mask 有了；但 out-of-turn-limit / invalid trajectory 的整条或局部 loss mask 还没按旧
  Mini-o3 对齐。