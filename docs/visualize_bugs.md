# Visualization Bugs

## Qwen3.5 Tool-Crop Bboxes: Runtime vs Raw Image Space

Date: 2026-05-31

Context: Mini-o3 Qwen3.5 VisualProbe rollout reports generated from `tool_trace`.

Symptom:

- Tool calls looked badly offset when drawing `minio3_crop/x0..y1` on the raw disk image.
- Crops looked like the model was zooming into unrelated regions.

Root cause:

- `image_zoom_in_tool` crops on the Qwen runtime image, not on the raw original image.
- The runtime image is produced by `qwen_vl_utils.fetch_image` during multimodal preprocessing.
- For the current Qwen3.5 path, `processor.image_processor.patch_size == 16`.
- Example: raw `visual_probe_train_1055.jpg` is `5200x3467`, but the runtime image is `1728x1152`.
- Therefore `bbox_2d=[200,500,400,700]` maps to runtime pixels `[345,576,691,806]`, matching `tool_trace`, but those pixels are wrong if drawn on `5200x3467`.

Fix:

- New rollouts keep a separate raw image bank inside `AgentData.raw_image_data`.
- `MiniO3CropTool` defaults to `use_raw_image_bank: true`.
- The tool now crops from the raw original image or a raw previous observation crop, while the model-facing observation image still goes through the normal Qwen/vLLM multimodal processing path.
- The tool trace includes `minio3_crop/source_space`; new fixed runs should report `raw` when the raw bank is available.

Correct visualization for new runs:

- If `tool_trace.response.metrics["minio3_crop/source_space"] == "raw"`, draw `x0..y1` on the raw source image/crop bank.
- Reconstruct the raw bank in order: original raw images first, then each returned crop as the next observation.

Correct visualization for old runs or fallback runs:

```python
from qwen_vl_utils import fetch_image

runtime_image = fetch_image(
    {
        "image": image_path,
        "min_pixels": 40000,
        "max_pixels": 2000000,
    },
    image_patch_size=16,
)
```

If `source_space` is missing or equals `runtime`, draw `tool_trace.response.metrics["minio3_crop/x0"]` through `y1` on `runtime_image`, not on the raw file.

Notes:

- For follow-up tool calls, `img_idx > 0` refers to prior observation images appended to the agent image bank.
- For fixed raw-bank runs, reconstruct the raw bank in order. For old runtime runs, reconstruct the runtime bank in order.
- If this is not done, second-stage crops from `img_idx=1`, `img_idx=2`, etc. will be visualized against the wrong source image.
