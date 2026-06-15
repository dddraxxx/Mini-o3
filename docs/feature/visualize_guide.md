# Visualization Artifact Guide

This guide records the preferred format for Mini-o3 visualization artifacts,
especially rollout/eval cases with image-tool traces.

## Location

- Put generated visualization artifacts under the top-level `artifacts/`
  directory, for example `artifacts/eval/<run_or_eval_name>/` or
  `artifacts/train/<run_name>/`.
- Do not put visualization image bundles under `exps/eval/assets/` or
  `exps/eval/artifacts/`.
- Keep `exps/eval/` for lightweight reports, reproducible scripts, and summary
  markdown that should be committed.
- Visual assets may live inside the artifact directory as `assets/`, with case
  markdown under `cases/`.

## Markdown Links

- Use relative links inside artifact markdown files.
- `index.md` should link to cases as `cases/<case>.md`.
- Case markdown should link to images as `../assets/<case_id>/<image>.jpg`.
- Do not write absolute local links such as `/mnt/localssd/...` inside artifact
  markdown.
- Before reporting completion, run a link check from each markdown file's own
  directory and verify:
  - `bad_links = 0`
  - `absolute_links = 0`

## Index Format

For sampled rollout case artifacts, `index.md` should include:

- Run path and source JSONL pattern.
- Sampling policy and seed.
- A summary table with per-step counts and sampled-label counts.
- A case table with compact columns, for example:

```markdown
| step | label | score | tools | question | case | source |
| ---: | --- | ---: | ---: | --- | --- | --- |
```

Use `label` for status only:

- `valid`
- `invalid_clip`
- `invalid_exceed`
- `invalid_format`

Correctness belongs in `score`, not in the label.

## Case Format

Each case markdown should follow this section order:

1. Metadata bullets: label, data source, score, final-answer presence,
   `img_idx` sequence, tool count, invalid detail.
2. `## Question`
3. `## Answer`
4. `## Tool Trace Crops`
5. `## Original Image(s)`
6. `## Full Trace`
7. `## Judge`

The question must be shown explicitly in its own section. Do not make the user
infer it from the model trace.

`## Full Trace` must contain the whole trajectory needed to inspect behavior:
the user question, assistant reasoning/output, tool calls, tool responses, and
final answer if present. Do not show only a tail. Do not hide the full trace in a
collapsed details block.

It is fine to put raw call/response JSON inside `<details>` blocks under each
tool, but the main trajectory itself should be visible directly.

## Tool Crop Format

For each tool call, show the crop before the original full image section. Include
both the source image with the bbox overlaid and the returned crop:

```markdown
| Source with bbox | Returned crop |
|---|---|
| ![tool 1 source](../assets/<case>/tool_01_source_bbox.jpg) | ![tool 1 crop](../assets/<case>/tool_01_returned_crop.jpg) |
```

Record the key tool metadata above the images:

- `source_space`
- `source_index`
- arg `img_idx`
- arg `bbox`
- pixel bbox
- source size
- crop size

If there are no tool calls, write `No tool calls recorded.`

## Sampling

When visualizing training rollouts over time, prefer reproducible random
sampling over hand-picked class balancing unless the artifact is explicitly a
failure-case review.

For the base raw-crop 100-step rollout review, the preferred pattern was:

- steps: `10,20,...,100`
- samples: `3` per step
- seed: deterministic, `20260602 + step`
- total cases: `30`
