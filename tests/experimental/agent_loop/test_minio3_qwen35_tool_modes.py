import asyncio
import json

from PIL import Image

from examples.minio3.minio3_reward import compute_score
from examples.minio3.preprocess_visualprobe import LEGACY_GROUNDING_PROMPT_SUITE
from examples.minio3.preprocess_visualprobe import OFFICIAL_ZOOM_PROMPT_SUITE
from examples.minio3.preprocess_visualprobe import _convert_row
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser
from verl.tools.minio3_crop_tool import MiniO3CropTool
from verl.tools.schemas import OpenAIFunctionParametersSchema
from verl.tools.schemas import OpenAIFunctionPropertySchema
from verl.tools.schemas import OpenAIFunctionSchema
from verl.tools.schemas import OpenAIFunctionToolSchema


class DummyTokenizer:
    def __init__(self, text: str) -> None:
        self.text = text

    def decode(self, _: list[int]) -> str:
        return self.text


def _zoom_tool_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="image_zoom_in_tool",
            description="Zoom into a region.",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "bbox_2d": OpenAIFunctionPropertySchema(type="array"),
                    "label": OpenAIFunctionPropertySchema(type="string"),
                    "img_idx": OpenAIFunctionPropertySchema(type="number"),
                },
                required=["bbox_2d", "label", "img_idx"],
            ),
        ),
    )


def test_preprocess_prompt_suites_keep_legacy_and_official_separate():
    row = {"images": ["image.jpg"], "question": "What is on top?", "answer": "bird"}

    legacy = _convert_row(
        row,
        0,
        "train",
        "/tmp/images",
        min_pixels=1,
        max_pixels=100,
        tool_prompt_suite=LEGACY_GROUNDING_PROMPT_SUITE,
    )
    official = _convert_row(
        row,
        0,
        "train",
        "/tmp/images",
        min_pixels=1,
        max_pixels=100,
        tool_prompt_suite=OFFICIAL_ZOOM_PROMPT_SUITE,
        official_tool_name="image_zoom_in_tool",
    )

    assert "<grounding>" in legacy["prompt"][0]["content"]
    assert legacy["agent_name"] == "mini_o3_tool_agent"
    assert legacy["extra_info"]["tool_selection"] == ["tool_crop"]

    assert "<grounding>" not in official["prompt"][0]["content"]
    assert "<tool_call>" not in official["prompt"][0]["content"]
    assert official["agent_name"] == "tool_agent"
    assert official["extra_info"]["tool_selection"] == ["image_zoom_in_tool"]


def test_qwen3_xml_parser_accepts_official_zoom_tool_call():
    text = """<think>
Need a closer view.
</think>
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
</tool_call>"""
    parser = Qwen3XMLToolParser(DummyTokenizer(text))

    content, function_calls = asyncio.run(parser.extract_tool_calls([1, 2, 3], [_zoom_tool_schema()]))

    assert "<tool_call>" not in content
    assert len(function_calls) == 1
    assert function_calls[0].name == "image_zoom_in_tool"
    assert json.loads(function_calls[0].arguments) == {
        "bbox_2d": [400, 250, 620, 520],
        "label": "central tower decoration",
        "img_idx": 0,
    }


def test_qwen3_tool_response_delta_is_template_safe():
    text = ToolAgentLoop._build_qwen3_tool_response_text(
        [{"role": "tool", "content": [{"type": "image"}, {"type": "text", "text": "Zoom-in observation."}]}]
    )

    assert text.startswith("<|im_start|>user\n<tool_response>\n")
    assert "<|vision_start|><|image_pad|><|vision_end|>" in text
    assert text.endswith("<|im_start|>assistant\n<think>\n")

    no_thinking_text = ToolAgentLoop._build_qwen3_tool_response_text(
        [{"role": "tool", "content": "Zoom-in observation."}],
        enable_thinking=False,
    )
    assert no_thinking_text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_minio3_crop_tool_supports_qwen_agent_zoom_parameters():
    tool = MiniO3CropTool(config={"type": "native", "use_relative_coordinates": True, "coordinate_scale": 1000}, tool_schema=None)

    class AgentData:
        image_data = [Image.new("RGB", (20, 20)), Image.new("RGB", (100, 80))]

    response, _, metrics = asyncio.run(
        tool.execute(
            "instance",
            {"bbox_2d": [0, 0, 500, 500], "label": "upper left", "img_idx": 1},
            agent_data=AgentData(),
        )
    )

    assert metrics["minio3_crop/source_index"] == 1
    assert response.image[0].size == (50, 40)


def test_reward_counts_legacy_and_official_tool_calls():
    legacy = compute_score(
        "test",
        '<think>x</think><grounding>{"bbox_2d":[0,0,1,1]}</grounding><answer>A</answer>',
        "A",
    )
    official = compute_score(
        "test",
        "<think>x</think><tool_call><function=image_zoom_in_tool></function></tool_call><answer>A</answer>",
        "A",
    )

    assert legacy["tool_call_count"] == 1.0
    assert official["tool_call_count"] == 1.0
    assert legacy["format_score"] == 1.0
    assert official["format_score"] == 1.0
