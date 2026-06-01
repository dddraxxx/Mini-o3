# Mini-o3 Invalid-Trajectory Logging

This note defines the Mini-o3 rollout failure logging used by the Qwen3.5
official-tool RL runs.

## Motivation

The old `void` metric mixed multiple failure modes and was hard to interpret.
We now log orthogonal causes and a union metric:

```text
invalid = clip OR exceed OR format
```

## Metrics

`batch/clip_sample_ratio`

Rows whose response window is full. In trainer metrics this is derived from
`attention_mask[:, -response_length:]`, so it matches the effective
`MAX_RESPONSE_LENGTH` budget. The agent loop also writes `clip_mask` and
`clip_reason` for generation JSONL dumps when it directly detects the cap.

`batch/exceed_sample_ratio`

Rows where the agent still has a tool/multi-turn action pending but hits a hard
budget. Typical reasons are `assistant_turn_limit_with_tool_call`,
`user_turn_limit_with_tool_call`, `response_length_with_tool_call`,
`observation_would_exceed_response_length`, and
`official_tool_response_would_exceed_response_length`.

`batch/format_sample_ratio`

Rows that terminate without a valid terminal `Final answer:` sentence when the
agent is not pending another tool call. This is the format failure category only;
length clipping is tracked separately by `clip`.

`batch/invalid_sample_ratio`

The union of `clip`, `exceed`, and `format`. This is the main health metric for
bad Mini-o3 trajectories.

## Per-Sample Fields

Generation dumps can include:

```text
clip_mask, clip_reason
exceed_mask, exceed_reason
format_mask, format_reason
invalid_mask, invalid_reasons
```

`invalid_reasons` is a list because a row can be both clipped and missing a
valid final answer.

## Loss-Mask Controls

The actor config uses the new names:

```text
actor_rollout_ref.actor.ignore_clip
actor_rollout_ref.actor.ignore_exceed
actor_rollout_ref.actor.ignore_format
actor_rollout_ref.actor.ignore_invalid
```

`ignore_invalid=True` masks the full union. Otherwise the individual switches
can be enabled independently. Current formal Qwen3.5 RL launchers default all
four switches to `False`, so invalid rows are logged but still train the actor.

The previous `void_mask`, `void_reason`, `batch/void_sample_ratio`, and
`MINIO3_IGNORE_VOID` surface is retired. Historically, `void` meant either a
length stop or a missing final answer, and it did not include turn/tool-budget
exceed cases. New runs should read `clip`, `format`, and `exceed` separately,
and use `invalid` for their union.
