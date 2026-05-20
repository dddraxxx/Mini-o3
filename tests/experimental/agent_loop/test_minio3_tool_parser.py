import asyncio
import json

from verl.experimental.agent_loop.tool_parser import MiniO3GroundingToolParser


class DummyTokenizer:
    def __init__(self, text: str) -> None:
        self.text = text

    def decode(self, _: list[int]) -> str:
        return self.text


def test_minio3_grounding_parser_accepts_bare_yaml_style_source():
    text = "<think>zoom</think><grounding>{bbox_2d: [1, 2, 3, 4], source: original_image}</grounding>"
    parser = MiniO3GroundingToolParser(DummyTokenizer(text))

    content, function_calls = asyncio.run(parser.extract_tool_calls([1, 2, 3]))

    assert "<grounding>" not in content
    assert len(function_calls) == 1
    assert function_calls[0].name == "tool_crop"
    assert json.loads(function_calls[0].arguments) == {
        "bbox_2d": [1, 2, 3, 4],
        "source": "original_image",
    }


def test_minio3_grounding_parser_defaults_source_for_json_call():
    text = '<grounding>{"bbox_2d": [10, 20, 30, 40]}</grounding>'
    parser = MiniO3GroundingToolParser(DummyTokenizer(text))

    _, function_calls = asyncio.run(parser.extract_tool_calls([1, 2, 3]))

    assert len(function_calls) == 1
    assert json.loads(function_calls[0].arguments) == {
        "bbox_2d": [10, 20, 30, 40],
        "source": "original_image",
    }
