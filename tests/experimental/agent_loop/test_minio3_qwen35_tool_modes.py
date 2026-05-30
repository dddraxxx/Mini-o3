import asyncio
import json

from PIL import Image

from examples.minio3.minio3_reward import compute_score
from examples.minio3.preprocess_visualprobe import OFFICIAL_ZOOM_FINAL_SENTENCE_PROMPT_SUITE
from examples.minio3.preprocess_visualprobe import LEGACY_GROUNDING_PROMPT_SUITE
from examples.minio3.preprocess_visualprobe import OFFICIAL_ZOOM_PROMPT_SUITE
from examples.minio3.preprocess_visualprobe import OFFICIAL_ZOOM_PLAIN_QUESTION_PROMPT_SUITE
from examples.minio3.preprocess_visualprobe import _convert_row
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser
from verl.tools.minio3_crop_tool import MiniO3CropTool
from verl.tools.schemas import OpenAIFunctionParametersSchema
from verl.tools.schemas import OpenAIFunctionPropertySchema
from verl.tools.schemas import OpenAIFunctionSchema
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.tools.schemas import ToolResponse


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
    plain_question = _convert_row(
        row,
        0,
        "train",
        "/tmp/images",
        min_pixels=1,
        max_pixels=100,
        tool_prompt_suite=OFFICIAL_ZOOM_PLAIN_QUESTION_PROMPT_SUITE,
        official_tool_name="image_zoom_in_tool",
    )
    final_sentence = _convert_row(
        row,
        0,
        "train",
        "/tmp/images",
        min_pixels=1,
        max_pixels=100,
        tool_prompt_suite=OFFICIAL_ZOOM_FINAL_SENTENCE_PROMPT_SUITE,
        official_tool_name="image_zoom_in_tool",
    )

    assert "<grounding>" in legacy["prompt"][0]["content"]
    assert legacy["agent_name"] == "mini_o3_tool_agent"
    assert legacy["extra_info"]["tool_selection"] == ["tool_crop"]

    assert "<grounding>" not in official["prompt"][0]["content"]
    assert "<tool_call>" not in official["prompt"][0]["content"]
    assert official["agent_name"] == "tool_agent"
    assert official["extra_info"]["tool_selection"] == ["image_zoom_in_tool"]

    assert "<answer>" not in plain_question["prompt"][0]["content"]
    assert "</answer>" not in plain_question["prompt"][0]["content"]
    assert plain_question["prompt"][1]["content"] == "<image>\nWhat is on top?"
    assert plain_question["agent_name"] == "tool_agent"
    assert plain_question["extra_info"]["tool_selection"] == ["image_zoom_in_tool"]

    assert "<answer>" not in final_sentence["prompt"][0]["content"]
    assert "</answer>" not in final_sentence["prompt"][0]["content"]
    assert "Final answer: <short answer>." in final_sentence["prompt"][0]["content"]
    assert "Do not add explanation" in final_sentence["prompt"][0]["content"]
    assert final_sentence["agent_name"] == "tool_agent"
    assert final_sentence["extra_info"]["tool_prompt_suite"] == OFFICIAL_ZOOM_FINAL_SENTENCE_PROMPT_SUITE
    assert final_sentence["extra_info"]["tool_selection"] == ["image_zoom_in_tool"]


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
        [{"role": "tool", "content": [{"type": "image"}]}],
        add_vision_id=True,
        image_start_index=2,
    )

    assert text.startswith("<|im_start|>user\n<tool_response>\n")
    assert "Picture 2: <|vision_start|><|image_pad|><|vision_end|>" in text
    assert "Zoom-in observation." not in text
    assert text.endswith("<|im_start|>assistant\n<think>\n")

    no_thinking_text = ToolAgentLoop._build_qwen3_tool_response_text(
        [{"role": "tool", "content": [{"type": "image"}]}],
        enable_thinking=False,
    )
    assert "<|vision_start|><|image_pad|><|vision_end|>" in no_thinking_text
    assert no_thinking_text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_tool_trace_records_structured_calls_without_image_bytes():
    class AgentData:
        assistant_turns = 2
        tool_calls_trace = []
        tool_responses_trace = []

    loop = ToolAgentLoop.__new__(ToolAgentLoop)
    loop._record_tool_interaction(
        AgentData,
        FunctionCall(name="image_zoom_in_tool", arguments='{"bbox_2d": [1, 2, 3, 4], "img_idx": 0}'),
        ToolResponse(image=[Image.new("RGB", (4, 4))]),
        0.0,
        {"minio3_crop/source_index": 0},
    )

    assert AgentData.tool_calls_trace == [
        {
            "turn": 2,
            "name": "image_zoom_in_tool",
            "arguments": {"bbox_2d": [1, 2, 3, 4], "img_idx": 0},
        }
    ]
    assert AgentData.tool_responses_trace == [
        {
            "turn": 2,
            "name": "image_zoom_in_tool",
            "text": "",
            "image_count": 1,
            "video_count": 0,
            "has_image": True,
            "has_video": False,
            "reward": 0.0,
            "metrics": {"minio3_crop/source_index": 0},
        }
    ]


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
