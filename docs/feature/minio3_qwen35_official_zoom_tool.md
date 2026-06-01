# Mini-o3 Qwen3.5 Official Zoom Tool Path

This note tracks the migration from Mini-o3 legacy `<grounding>` crop calls to
the Qwen3.5 official tool-calling format for the same zoom/crop capability.

## Goal

Use `Qwen/Qwen3.5-9B` with the tool-call format defined by that model's own
chat template while preserving Mini-o3's image zoom behavior.

Important scope boundary: Qwen3.5 provides the official function-calling
protocol and chat template. It does not provide a built-in zoom implementation.
Mini-o3 should keep using `verl.tools.minio3_crop_tool.MiniO3CropTool` as the
zoom backend and expose it through Qwen3.5's official tool schema path.

Important naming boundary: `qwen3_coder` below is the parser name used by
vLLM/verl for the Qwen XML function-call syntax. It does not mean using the
Qwen3-Coder model or the Qwen3-Coder chat template as the authority. The
authority for this note is the local `Qwen/Qwen3.5-9B` snapshot:

```text
model_type    = qwen3_5
architecture  = Qwen3_5ForConditionalGeneration
processor     = Qwen3VLProcessor
chat template = models--Qwen--Qwen3.5-9B/.../chat_template.jinja
```

That Qwen3.5 chat template itself contains the `# Tools` section and the
`<tool_call><function=...><parameter=...>` XML call shape.

## Current State

The current Mini-o3 path is legacy-format specific:

```text
dataset prompt -> asks model to emit <grounding>{...}</grounding>
rollout format -> minio3_grounding
agent loop     -> mini_o3_tool_agent
tool parser    -> MiniO3GroundingToolParser
tool backend   -> tool_crop / MiniO3CropTool
observation    -> user message with a zoom-in image
reward         -> counts <grounding> tags
```

That path works for the Qwen3-VL-style Mini-o3 format, but it bypasses
Qwen3.5's official tool template:

- `MiniO3ToolAgentLoop._handle_pending_state()` calls `apply_chat_template()`
with `tools=None`, so the model never sees the tool schema.
- `examples/minio3/preprocess_visualprobe.py` explicitly tells the model to
output `<grounding>{...}</grounding>`.
- `examples/minio3/minio3_reward.py` counts only `<grounding>` calls.

The local Qwen3.5 environment is currently Qwen3.5-capable:

```text
transformers=5.3.0.dev0
vllm=0.18.0
accelerate=1.13.0
torch=2.10.0+cu128
```

Verified local preflight:

```bash
uv run --active --no-sync python examples/minio3/check_qwen35_env.py \
  --model-path Qwen/Qwen3.5-9B \
  --local-files-only
```

## Target Shape

The Qwen3.5 official tool path should look like this:

```text
dataset prompt -> says when to use the zoom tool, but does not hand-write XML
chat template  -> injects the tool schema into the system message
rollout format -> qwen3_coder
agent loop     -> tool_agent or qwen3.5 mode in mini_o3_tool_agent
tool parser    -> Qwen3XMLToolParser
tool call      -> <tool_call><function=tool_crop or image_zoom_in_tool>...</function></tool_call>
tool backend   -> tool_crop / MiniO3CropTool
observation    -> <tool_response> with the zoom-in image
reward         -> counts either official tool calls or tool metrics
```

The Qwen3.5 model card's vLLM/SGLang serving examples use the parser name
`qwen3_coder` for tool calls:

- `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
- Reference: [https://huggingface.co/Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)

In this repo's verl fork, the equivalent parser already exists as:

```text
verl.experimental.agent_loop.tool_parser.Qwen3XMLToolParser
registered name: qwen3_coder
```

The class docstring says it was adapted from Qwen3-Coder tooling, but the
parsed surface matches `Qwen/Qwen3.5-9B`'s own chat template. Treat
`qwen3_coder` as an implementation name for the XML parser, not as model
evidence.

## Prompt Suites

Keep two Qwen3.5 prompt suites. Both target `Qwen/Qwen3.5-9B`, but they teach
different tool surfaces and should not be mixed inside the same rollout.

### Suite A: `qwen35_official_zoom_tool`

This suite follows the Qwen-Agent "Think with Images" zoom-tool pattern:
the system prompt teaches an inspect -> tool -> review loop, while the actual
tool-call syntax comes from the Qwen3.5 chat template and tool schema.

Use this suite when the rollout path passes active tool schemas into
`apply_chat_template()` and uses the `qwen3_coder` XML parser name.

Initial system prompt:

```text
You are a visual research assistant. Answer the user's image question by
examining the image carefully and using the available zoom tool when visual
details are unclear.

For each question, follow this loop:
1. First inspect the image with the user's question in mind.
2. State what is visible and what needs closer inspection.
3. If needed, call the zoom tool on a precise region.
4. Review the zoom observation before deciding whether another zoom is needed.
5. When there is enough evidence, give the final answer inside <answer> and
</answer>.
```

### Suite A2: `qwen35_official_zoom_tool_plain_question`

This suite keeps the same official Qwen3.5 tool surface, parser, agent loop,
and `image_zoom_in_tool` selection as Suite A, but does not prescribe a final
answer delimiter. Use it when evaluating with a relaxed/plain final-answer
extractor or an LLM judge.

The user prompt is only the image token(s) followed by the raw question. The
system prompt still teaches the inspect -> zoom -> review loop:

```text
You are a visual research assistant. Answer the user's image question by
examining the image carefully and using the available zoom tool when visual
details are unclear.

For each question, follow this loop:
1. First inspect the image with the user's question in mind.
2. State what is visible and what needs closer inspection.
3. If needed, call the zoom tool on a precise region.
4. Review the zoom observation before deciding whether another zoom is needed.
```

Official reference tool surface:

```text
function name: image_zoom_in_tool
required parameter: bbox_2d
required parameter: label
required parameter: img_idx
bbox coordinates: relative [0, 1000] as [x1, y1, x2, y2]
img_idx: image index starting from 0
```

Repo bridge:

```text
backend implementation: MiniO3CropTool
preferred compatibility name during migration: tool_crop
strict official alias to add later: image_zoom_in_tool
coordinate adapter needed if using the official alias: [0, 1000] -> [0, 1]
```

Rendered by the Qwen3.5 chat template, the model should see the tool schema
under `# Tools` and produce calls like:

```text
<tool_call>
<function=image_zoom_in_tool>
<parameter=bbox_2d>
[400, 250, 620, 520]
</parameter>
<parameter=label>
central tower decoration
</parameter>
<parameter=img_idx>
0
</parameter>
</function>
</tool_call>
```

If we keep the migration name `tool_crop` for the first smoke, the rendered call
shape is the same but the function name is `tool_crop`. The prompt should still
avoid hand-writing the XML syntax; the Qwen3.5 chat template owns that syntax.

Observation prompt:

```text
Use the Qwen3.5 tool response path. The crop result should be returned as a
tool response containing the zoomed image. Do not add Mini-o3 grounding
instructions to the observation text in this suite.
```

### Suite B: `qwen35_minio3_legacy_grounding`

This suite keeps the original Mini-o3 `<grounding>` interface, but makes the
wording explicit for Qwen3.5. It is useful for continuity with existing SFT/RL
data and for A/B testing against the official tool surface.

Use this suite with:

```text
rollout format: minio3_grounding
agent loop: mini_o3_tool_agent
parser: MiniO3GroundingToolParser
backend: MiniO3CropTool
```

Initial system prompt:

```text
You are a helpful visual reasoning assistant. Answer the user's question based
on the image. Write your reasoning inside <think> and </think>.

When visual details are unclear, request a zoom crop by outputting exactly one
grounding call:
<grounding>{"bbox_2d": [x0, y0, x1, y1], "source": "original_image"}</grounding>

The bbox uses relative coordinates in [0, 1]. (x0, y0) is the top-left corner
and (x1, y1) is the bottom-right corner. The source can be "original_image" or
"observation_i" for a previous zoom observation. Do not use <tool_call> in this
suite. Once the final answer is confirmed, put it inside <answer> and
</answer>.
```

Grounding call example:

```text
<grounding>{"bbox_2d": [0.40, 0.25, 0.62, 0.52], "source": "original_image"}</grounding>
```

Observation prompt after a crop:

```text
After the above Action {action_turn}, here is the zoom-in image
(Observation {observation_id}). Continue your reasoning inside <think> and
</think>. If needed, continue to zoom in on the original image or any
observation by outputting <grounding> and </grounding> as before. If the final
answer is confirmed, put it inside <answer> and </answer>.
```

### Suite Differences

| Topic | Official Qwen3.5 suite | Mini-o3 legacy suite |
| --- | --- | --- |
| Prompt style | Describes when to use the zoom tool; chat template teaches syntax | Explicitly teaches `<grounding>{...}</grounding>` |
| Tool name | `image_zoom_in_tool` in strict official mode; `tool_crop` acceptable for first repo smoke | `tool_crop` backend hidden behind `<grounding>` |
| Parser | `qwen3_coder` parser name for Qwen XML tool calls | `minio3_grounding` |
| Coordinates | `[0, 1000]` for official `image_zoom_in_tool` | `[0, 1]` floats |
| Extra args | `label`, `img_idx` | `source` |
| Observation | `role=tool` / `<tool_response>` from Qwen3.5 template | user observation text with Action/Observation ids |
| Reward counting | count `<tool_call>` or tool metrics | count `<grounding>` |
| Best use | Qwen3.5-native tool-call smoke and future aligned training | continuity with current Mini-o3 data and regression tests |

## Code Changes

The compatibility layer is implemented as switchable knobs, not as a one-way
replacement. The default remains legacy Mini-o3 so older runs keep the same
template/parser behavior unless the official Qwen3.5 path is requested.

### 1. Dataset prompt

`examples/minio3/preprocess_visualprobe.py` supports a prompt suite switch:

```text
MINIO3_TOOL_PROMPT_SUITE=qwen35_minio3_legacy_grounding
MINIO3_TOOL_PROMPT_SUITE=qwen35_official_zoom_tool
MINIO3_TOOL_PROMPT_SUITE=qwen35_official_zoom_tool_plain_question
```

Legacy mode writes `agent_name=mini_o3_tool_agent` and
`extra_info.tool_selection=["tool_crop"]`. Official mode writes
`agent_name=tool_agent` by default and switches `extra_info.tool_selection` to
the configured official tool name.

`qwen35_official_zoom_tool_plain_question` is the same official Qwen3.5 tool
surface as `qwen35_official_zoom_tool`, but removes the `<answer>...</answer>`
instruction from the system prompt. The user message remains exactly the image
placeholder(s) followed by the raw question, for example:

```text
<image>
What is shown?
```

The train and val-smoke wrappers pass these through:

```text
MINIO3_TOOL_PROMPT_SUITE
MINIO3_OFFICIAL_TOOL_NAME
MINIO3_AGENT_LOOP
```

### 2. Rollout format

For Qwen3.5 official mode, pass:

```bash
actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
```

The legacy path keeps:

```bash
actor_rollout_ref.rollout.multi_turn.format=minio3_grounding
```

### 3. Agent loop

Two implementation options are supported.

Option A, fastest direct smoke:

```text
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
```

This uses official verl tool semantics directly:

- pending prompt includes `tools=schemas`
- tool response is encoded as `role=tool`
- Qwen3.5 template renders `<tool_response>`

Tradeoff: Mini-o3-specific `format_mask` / `clip_mask` / `exceed_mask` behavior from
`MiniO3ToolAgentLoop` is not preserved.

Option B, preferred path:

Keep `mini_o3_tool_agent`, but add an official-tool mode:

- when `multi_turn.format == qwen3_coder`, pass active tool schemas to
`apply_chat_template()`;
- do not set the legacy stop sequence `</grounding>`;
- encode zoom responses as official `role=tool` messages or explicitly mirror
the Qwen3.5 `<tool_response>` template;
- preserve Mini-o3 `format_mask`, `clip_mask`, `exceed_mask`, stage logging, and response mask
behavior.

Do not keep the current `MiniO3ToolAgentLoop._handle_pending_state()` behavior
unchanged for official mode. It currently passes `tools=None` into
`apply_chat_template()`, which prevents Qwen3.5's chat template from rendering
the `# Tools` block and therefore prevents official tool calling.

Implementation note: Qwen3.5's chat template cannot be safely applied to a
standalone incremental `role=tool` message because it expects a full user query
context. The current implementation therefore encodes the Qwen3.5 tool-response
delta explicitly as:

```text
<|im_start|>user
<tool_response>
...
</tool_response><|im_end|>
<|im_start|>assistant
<think>
```

This keeps the initial prompt rendered by the official Qwen3.5 chat template
while avoiding template errors on the tool-only continuation.

### 4. Reward and metrics

Update `examples/minio3/minio3_reward.py` so it does not depend only on
`<grounding>`:

- keep `<answer>...</answer>` as the final answer contract;
- count legacy calls via `<grounding>...</grounding>`;
- count official calls via `<tool_call>...</tool_call>` or rollout tool metrics;
- keep `format_reward_weight=0.0` during early smoke if format churn could hide
rollout failures.

### 5. Tool schema wording

For compatibility, `MiniO3CropTool.get_openai_tool_schema()` keeps the function
name `tool_crop`, but the description is Qwen3.5-friendly:

```text
Zoom into a specific rectangular region of the original image or a previous
observation image and return the cropped zoom-in image.
```

For strict official-suite parity, the repo also has a tool config exposing an
alias schema named `image_zoom_in_tool` that maps internally to
`MiniO3CropTool` and accepts
Qwen-Agent-style parameters:

```text
bbox_2d: relative [0, 1000]
label: object or region label
img_idx: image index starting from 0
```

The crop backend accepts both legacy `source` values and official `img_idx`
values. It also supports `coordinate_scale=1000` for the official schema while
retaining the legacy `[0, 1]` default.

## Smoke Plan

Start with val-only smoke before touching train:

```bash
MODEL_PATH=Qwen/Qwen3.5-9B \
MINIO3_TOOL_PROMPT_SUITE=qwen35_official_zoom_tool \
MINIO3_OFFICIAL_TOOL_NAME=image_zoom_in_tool \
ROLLOUT_MULTI_TURN_FORMAT=qwen3_coder \
ROLLOUT_AGENT_LOOP=mini_o3_tool_agent \
RUN_ID=visualprobe_val_smoke10_qwen35_9b_official_tool \
bash examples/minio3/run_real_val_visualprobe_smoke.sh
```

Using `MINIO3_OFFICIAL_TOOL_NAME=image_zoom_in_tool` selects
`examples/minio3/config/tool_config/minio3_image_zoom_in_tool.yaml` unless
`TOOL_CONFIG_PATH` is overridden. For a lower-risk migration smoke, keep
`MINIO3_OFFICIAL_TOOL_NAME=tool_crop`; this uses the same Qwen3.5 XML call
shape but the existing public tool name.

Expected checks:

- `examples/minio3/check_qwen35_env.py` passes before Ray startup.
- The rendered prompt contains `# Tools` and the selected zoom tool schema.
- Generation emits `<tool_call>` rather than `<grounding>`.
- `Qwen3XMLToolParser` extracts a zoom-tool function call.
- `MiniO3CropTool` returns a zoom-in image.
- The next turn contains a tool response / zoom observation image.
- With `qwen35_official_zoom_tool`, validation generations are still asked to
  end with `<answer>...</answer>`.
- With `qwen35_official_zoom_tool_plain_question`, validation generations are
  not asked to use any final-answer delimiter.

For regression coverage, also run the same script without the official env
knobs. That should stay on `qwen35_minio3_legacy_grounding`,
`minio3_grounding`, and `mini_o3_tool_agent`.

## SFT/RL Alignment Decision

For Qwen3.5 official-tool runs, both SFT conversion and RL/eval runtime should
use `add_vision_id=True`. This makes the rendered prompt show images as
`Picture 1:`, `Picture 2:`, etc. The picture label is one-based display text;
the official `image_zoom_in_tool.img_idx` parameter remains zero-based.

Successful zoom-tool responses should follow the official Qwen-Agent behavior:
return the crop image only. Do not add local observation prose such as
`Zoom-in observation.` or `This returned image is now image index ...` by
default.

Cold-start SFT conversion should preserve the assistant reasoning text while
rewriting only the tool/final-answer protocol:

- `<grounding>{"bbox_2d": ..., "source": ...}</grounding>` becomes structured
  `assistant.tool_calls` for `image_zoom_in_tool`.
- Legacy `[0, 1]` float boxes become Qwen-Agent `[0, 1000]` integer boxes.
- `source=original_image` maps to `img_idx=0`; `source=observation_k` maps to
  `img_idx=k`.
- `label` uses a deterministic fallback, currently `selected region`.
- Legacy human observation messages become `role=tool` messages containing only
  `<image>`.
- `<answer>...</answer>` becomes the current `Final answer: ...` final response
  style.
- The old system prompt is replaced with the Qwen3.5 official zoom-tool prompt;
  it must not teach `<grounding>`, `source`, or `<answer>`.

The converter entrypoint is:

```bash
uv run --project . --no-sync python examples/minio3/preprocess_coldstart_sft.py \
  --input data/minio3_coldstart_hf \
  --output data/minio3_coldstart_verl_sft_qwen35_official_tool/train_shards \
  --rows-per-shard 512 \
  --min-pixels 40000 \
  --max-pixels 2000000
```

Recommended SFT config:

```text
data.apply_chat_template_kwargs.add_vision_id=True
data.messages_key=messages
data.image_key=images
data.tools_key=tools
data.whole_conversation_tokenize=True
data.read_parquet_dtype_backend=default
data.image_min_pixels=40000
data.image_max_pixels=2000000
```

Practical loader notes:

- Use parquet shards rather than one large byte-image parquet. A single 8GB+
  nested `images` column can hit pyarrow chunked nested-array limits.
- Keep pixel limits in SFT config instead of embedding them into the byte-image
  struct. The dataset injects `min_pixels` and `max_pixels` before qwen-vl-utils
  image preprocessing.
- Qwen3.5 official tool conversations require whole-conversation tokenization;
  per-message tokenization can fail because the Qwen3.5 chat template expects
  valid tool-call context.
