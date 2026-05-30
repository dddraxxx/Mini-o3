# Qwen3.5 Official-Tool H200 RL Experiment Index

This index keeps the major Qwen3.5 official-tool H200 RL runs comparable. Each
experiment has a frozen launcher so the moving default script can keep evolving.

## Runs

```text
run_id                                             launcher                                                   result
qwen35_9b_official_tool_h200_rl_20260526_214535   run_qwen35_official_tool_h200_rl_t6_200step_20260526.sh   stopped at step 159 after reward and validation degradation
qwen35_9b_official_tool_h200_rl_20260527_190822   run_qwen35_official_tool_h200_rl_t12_100step_20260527.sh  completed 100 steps; turn-limit failure delayed but still returned late
qwen35_9b_official_tool_h200_rl_t12_no_exceed_mask_100step_20260528_215420  run_qwen35_official_tool_h200_rl_t12_no_exceed_mask_100step_20260528.sh  stopped at step 80; reward polluted by plain-answer extraction mismatch
pending                                            moving run_qwen35_official_tool_h200_rl.sh                 planned: final-sentence prompt suite with T12 no-exceed-mask defaults
```

## T6 200-Step Run

Details:

```text
record_dir=exps/train/qwen35_9b_official_tool_h200_rl_20260526_214535
run_dir=save/qwen35_9b_official_tool_h200_rl_20260526_214535
log=logs/qwen35_9b_official_tool_h200_rl_20260526_214535.log
launcher=exps/train/run_qwen35_official_tool_h200_rl_t6_200step_20260526.sh
wandb=https://wandb.ai/dddraxxx/Mini-o3-qwen35-rl/runs/t8tcx0y8
```

Core method:

```text
train_turn_limit=6
val_turn_limit=12
total_training_steps=200
actor_lr=1e-6
rollout_n=8
train_batch_size=64
agent_num_workers=64
prompt_admission_pool_size=160
deepseek_self_judge=True
```

Main conclusion:

```text
The model increasingly continued zoom-tool calls and failed to emit final
answers within six assistant turns. Reward held around 0.45 until roughly step
80, degraded after step 100, and validation also fell. Empty answers were
confirmed to be turn-limit cases, not answer-extraction growth.
```

Key metrics:

```text
complete_steps=159
mean_reward=0.4166
latest_reward=0.3672
last20_reward_avg=0.3777
turn_limit_avg_by_block:
  1-20=14.9/512
  81-100=90.5/512
  101-120=164.2/512
  141-159=112.1/512
final_checked_val_step=150
final_checked_val_easy=0.1875
final_checked_val_medium=0.0938
final_checked_val_hard=0.1875
```

## T12 100-Step Run

Details:

```text
record_dir=exps/train/qwen35_9b_official_tool_h200_rl_20260527_190822
run_dir=save/qwen35_9b_official_tool_h200_rl_20260527_190822
log=logs/qwen35_9b_official_tool_h200_rl_20260527_190822.log
launcher=exps/train/run_qwen35_official_tool_h200_rl_t12_100step_20260527.sh
wandb=https://wandb.ai/dddraxxx/Mini-o3-qwen35-rl/runs/pz45ci1a
```

Core method:

```text
train_turn_limit=12
val_turn_limit=12
total_training_steps=100
actor_lr=1e-6
rollout_n=8
train_batch_size=64
agent_num_workers=64
prompt_admission_pool_size=160
deepseek_self_judge=True
```

Main conclusion:

```text
Increasing train turn limit from 6 to 12 delayed turn-limit failures but did
not fix the behavior. Turn-limit misses were low through step 70, rose at step
81-90, and became severe at step 91-100. Final validation also degraded.
```

Key metrics:

```text
complete_steps=100
mean_reward=0.4490
latest_reward=0.3613
last20_reward_avg=0.4405
turn_limit_avg_by_block:
  1-10=1.8/512
  51-60=5.9/512
  71-80=8.1/512
  81-90=32.6/512
  91-100=78.0/512
final_val_step=100
final_val_easy=0.2188
final_val_medium=0.1875
final_val_hard=0.1250
final_val_turn_mean=12.48
```

## Shared Interpretation

Both experiments point to the same failure mode: RL improves or maintains early
reward while gradually increasing tool continuation. Once the policy keeps
calling tools instead of finalizing, turn-limit misses rise, rollouts slow down,
and validation degrades.

For the next run, do not rely on a larger turn limit as the main fix. Use one
or more controls that directly target stop behavior:

```text
- penalize assistant-turn-limit trajectories even when ignore_exceed masks actor loss
- penalize excessive tool calls or long tool-use chains
- add a positive reward component for concise final answers
- consider an early stop/rollback rule when turn-limit avg exceeds a threshold
```

The important monitoring signal is not only train reward. Track
`assistant_turn_limit_with_tool_call` counts and validation turn mean every 10
steps.

## T12 No-Exceed-Mask Ablation

Launcher:

```text
exps/train/run_qwen35_official_tool_h200_rl_t12_no_exceed_mask_100step_20260528.sh
```

Core method:

```text
train_turn_limit=12
val_turn_limit=12
total_training_steps=100
actor_lr=1e-6
rollout_n=8
train_batch_size=64
agent_num_workers=64
prompt_admission_pool_size=160
deepseek_self_judge=True
ignore_exceed=False
ignore_void=False
```

This ablation removes the prior trick that zeroed actor loss for exceeded
Mini-o3 trajectories. Exceeded tool-call trajectories should now remain in the
actor update with their low reward/advantage signal. The intended readout is
whether this directly discourages non-finalizing tool continuation, compared
with the T12 run where `ignore_exceed=True`.

Result:

```text
run_id=qwen35_9b_official_tool_h200_rl_t12_no_exceed_mask_100step_20260528_215420
stopped_step=80
step80_train_reward=0.4199
step80_val_easy=0.4063
step80_val_medium=0.2500
step80_val_hard=0.2188
```

The run was stopped before step 100 after visual inspection showed the reward
path was sometimes punishing correct-but-verbose final answers. Example:
`Final answer` content such as `"VINTAGE". The letters are ...` was reduced by
plain relaxed extraction to only the trailing explanatory sentence, and then
DeepSeek judged that extracted prediction as incorrect.

## Planned Final-Sentence Prompt Follow-Up

The moving H200 launcher now defaults to:

```text
MINIO3_TOOL_PROMPT_SUITE=qwen35_official_zoom_tool_final_sentence
train_turn_limit=12
val_turn_limit=12
total_training_steps=100
ignore_exceed=False
ignore_void=False
```

This prompt keeps the clean official zoom-tool interface but asks the model to
end the final natural-language response with one standalone sentence:

```text
Final answer: <short answer>.
```

The goal is to make the existing relaxed reward extraction and prompt agree:
the reward can keep using the final answer marker plus last complete sentence,
while the model is explicitly trained to place the concise answer there. The
`Final answer:` marker is stripped before the extracted prediction is logged or
sent to the judge.
