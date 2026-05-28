# Qwen3.5-9B Official Tool RL Run 20260526_214535

## Run

This directory records the stopped H200 formal RL run:

```text
run_id=qwen35_9b_official_tool_h200_rl_20260526_214535
tmux=minio3_formal_20260526_214535
run_dir=save/qwen35_9b_official_tool_h200_rl_20260526_214535
log=logs/qwen35_9b_official_tool_h200_rl_20260526_214535.log
wandb=https://wandb.ai/dddraxxx/Mini-o3-qwen35-rl/runs/t8tcx0y8
workspace_head_at_stop=e06a147333d3c2fd711449b87380a72f639ba19c
```

The run was stopped manually on 2026-05-27 after reward degradation was observed.
The last complete `train_step_metrics.jsonl` row is step 159. Step 160 rollout
was dumped, but the job was interrupted during the step-160 test/validation
phase, so step 160 is not counted as a complete metric point.

## Method

Entrypoint:

```bash
cd /mnt/localssd/Mini-o3
bash exps/train/run_qwen35_official_tool_h200_rl.sh formal
```

Main configuration observed in the log:

```text
model=/mnt/localssd/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
tool_prompt_suite=qwen35_official_zoom_tool_plain_question
tool_name=image_zoom_in_tool
agent_loop=mini_o3_tool_agent
multi_turn_format=qwen3_coder
algorithm=GRPO
actor_strategy=fsdp2
lora_rank=8
lora_alpha=16
train_batch_size=64
ppo_mini_batch_size=16
ppo_micro_batch_size_per_gpu=2
rollout_n=8
rollout_dp=8
rollout_tp=1
agent_num_workers=64
ray_num_cpus=96
max_prompt_length=16384
max_response_length=16384
max_model_len=65536
max_num_batched_tokens=65536
max_num_seqs=256
max_assistant_turns=6
max_user_turns=6
val_batch_size=512
val_n=1
val_temperature=1.0
val_response_length=32768
val_max_assistant_turns=12
val_max_user_turns=12
self_judge_reward=True
self_judge_provider=deepseek
self_judge_model=deepseek-v4-flash
self_judge_relaxed_answer=True
prompt_admission=True
prompt_admission_pool_size=160
save_freq=10
test_freq=10
total_training_steps=200
```

Artifacts in this directory:

```text
train_reward_curve.csv
val_curve.csv
answer_health_by_block.csv
```

The source of truth remains the run directory under `save/`.

## Reward Curve

Train reward started around 0.45, held roughly flat through step 80, then
dropped sharply after step 100.

```text
steps     avg_reward  min       max       last
1-20      0.4509      0.3887    0.5371    0.4512
21-40     0.4563      0.3789    0.5449    0.5039
41-60     0.4519      0.3867    0.5449    0.4336
61-80     0.4498      0.3672    0.5176    0.4199
81-100    0.4258      0.3398    0.4746    0.4160
101-120   0.3642      0.2871    0.4570    0.2871
121-140   0.3528      0.3145    0.4121    0.3477
141-159   0.3793      0.3379    0.4492    0.3672
```

Overall:

```text
complete_steps=159
first_reward=0.4941
latest_reward=0.3672
mean_reward=0.4166
min_reward=0.2871
max_reward=0.5449
last5_avg=0.3918
last10_avg=0.3850
last20_avg=0.3777
```

Validation also degraded:

```text
step  easy    medium  hard
10    0.5313  0.2813  0.1875
20    0.3750  0.2500  0.1563
30    0.4063  0.2188  0.1875
40    0.4375  0.4063  0.1250
50    0.4688  0.2500  0.1875
60    0.4688  0.1563  0.1250
70    0.3438  0.2188  0.0938
80    0.4375  0.3125  0.1250
90    0.3438  0.1563  0.0938
100   0.3438  0.0938  0.1250
110   0.2188  0.1250  0.0313
120   0.2500  0.1563  0.0313
130   0.2188  0.0938  0.0625
140   0.1563  0.0938  0.0938
150   0.1875  0.0938  0.1875
```

## Findings

The main failure mode was not answer extraction growing without bound. The
extracted `prediction` stayed short and judge errors stayed at zero. The
problem was that the policy increasingly kept calling the zoom tool and failed
to finish within the six assistant turns.

Block summary from rollout JSONL:

```text
steps     reward_avg  empty_answer_avg  turn_limit_avg  output_p90_avg  pred_p50_avg
1-20      0.4509      14.8              14.9            12931           64.3
21-40     0.4563      16.7              16.9            16619           63.1
41-60     0.4519      21.9              22.4            10457           61.9
61-80     0.4498      36.6              37.4            7207            56.9
81-100    0.4258      84.6              90.5            5269            43.2
101-120   0.3642      157.2             164.2           38715           32.2
121-140   0.3528      143.2             146.8           71162           33.8
141-159   0.3793      111.0             112.1           71787           37.1
```

Empty answers were confirmed to be turn-limit cases:

```text
tool_call_count=6
exceed_reason=assistant_turn_limit_with_tool_call
output_tail ends with </tool_call>
prediction_source=missing
```

There was also a non-fatal malformed tool-call parser error near step 157:

```text
Error in extracting tool call from response: substring not found
ValueError: substring not found
```

The job continued after this parser error. It was stopped by user request,
which produced the expected `KeyboardInterrupt` and vLLM engine shutdown noise
in the log.

## Interpretation

This run trained the model toward longer tool-use trajectories rather than
toward concise final answers. Once turn-limit misses became common, both train
reward and validation accuracy dropped. For the next run, the most direct
change is to penalize or otherwise control excessive tool continuation before
increasing the turn limit, because raising the turn limit alone may allow even
longer rollouts without fixing the stop behavior.
