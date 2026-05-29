# Qwen3.5-9B Official Tool RL Run 20260527_190822

## Run

This directory records the completed H200 formal RL run:

```text
run_id=qwen35_9b_official_tool_h200_rl_20260527_190822
tmux=minio3_formal_20260527_190822
run_dir=save/qwen35_9b_official_tool_h200_rl_20260527_190822
log=logs/qwen35_9b_official_tool_h200_rl_20260527_190822.log
wandb=https://wandb.ai/dddraxxx/Mini-o3-qwen35-rl/runs/pz45ci1a
workspace_head_at_launch=a81c6fc1
```

The job completed all 100 training steps on 2026-05-28. The tmux session exited
normally. The final vLLM `EngineCore_DP1 died unexpectedly` line happened after
final validation and wrapper shutdown; the launcher exited 0.

## Method

Reproducible entrypoint:

```bash
cd /mnt/localssd/Mini-o3
bash exps/train/run_qwen35_official_tool_h200_rl_t12_100step_20260527.sh formal
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
actor_lr=1e-6
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
max_assistant_turns=12
max_user_turns=12
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
total_training_steps=100
```

Artifacts in this directory:

```text
train_curve.csv
train_blocks_10step.csv
val_curve.csv
```

The source of truth remains the run directory under `save/`.

## Reward And Turn Limits

Train reward stayed near the prior early-run level through most of the run, but
turn-limit failures accelerated late in training.

```text
steps   reward_avg  reward_min  reward_max  turn_limit_avg/512  turn_limit_max  empty_sum
1-10    0.4512      0.3926      0.5117      1.8                 5               18
11-20   0.4496      0.4102      0.4941      1.9                 5               19
21-30   0.4672      0.4121      0.5312      1.7                 5               17
31-40   0.4465      0.4043      0.4668      2.6                 5               26
41-50   0.4529      0.4316      0.4824      3.4                 7               36
51-60   0.4570      0.4355      0.5137      5.9                 12              60
61-70   0.4529      0.4180      0.4844      5.7                 11              59
71-80   0.4348      0.3887      0.5117      8.1                 19              81
81-90   0.4488      0.4043      0.4883      32.6                52              324
91-100  0.4322      0.3613      0.4863      78.0                101             730
```

Overall:

```text
complete_steps=100
first_reward=0.4863
latest_reward=0.3613
mean_reward=0.4490
min_reward=0.3613
max_reward=0.5312
last5_avg=0.4184
last10_avg=0.4322
last20_avg=0.4405
```

The dominant exceed reason remained:

```text
assistant_turn_limit_with_tool_call
```

Other exceed reasons were rare and mostly response/tool-observation length
cases:

```text
response_length_with_tool_call
official_tool_response_would_exceed_response_length
```

## Validation

Validation did not improve by the end of the 100-step run. The final validation
fell to:

```text
step  easy    medium  hard    val_turn_mean  val_turn_max
10    0.4062  0.2500  0.1250  7.73           24
20    0.3750  0.2812  0.1875  7.27           24
30    0.3125  0.2188  0.2500  7.15           24
40    0.5625  0.2500  0.0938  7.25           24
50    0.3750  0.3438  0.2500  8.06           24
60    0.4375  0.2500  0.2188  7.81           24
70    0.3438  0.1875  0.2188  7.67           24
80    0.3750  0.2812  0.2188  7.94           24
90    0.3438  0.1875  0.2188  9.35           24
100   0.2188  0.1875  0.1250  12.48          24
```

## Findings

Raising the train turn budget from 6 to 12 helped early and mid-run stability:
turn-limit misses were around 2 per 512 rollouts through step 30 and stayed
below 10 per 512 on average until step 80.

The same failure mode still emerged late. Step 81-90 rose to 32.6 turn-limit
misses per 512, and step 91-100 rose to 78.0 per 512. The final step had 101
turn-limit misses out of 512 rollouts and train reward dropped to 0.3613.

The run also slowed as turn-limit behavior increased:

```text
steps   step_time_avg_s  gen_time_avg_s  throughput_avg
1-10    356.2            225.7           597.8
21-30   430.9            291.4           583.4
71-80   586.7            442.2           508.7
91-100  603.3            444.2           473.9
```

## Interpretation

This experiment shows that increasing the turn budget alone is not sufficient.
It delays the failure relative to the 6-turn run, but the policy still learns to
continue tool calls instead of finishing. The next training attempt should add
an explicit stop/conciseness pressure or a tool-continuation penalty rather than
only raising the turn cap again.
