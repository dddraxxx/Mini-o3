import json

from examples.minio3.preprocess_coldstart_sft import OFFICIAL_ZOOM_FINAL_SENTENCE_SYSTEM_PROMPT
from examples.minio3.preprocess_coldstart_sft import convert_row


def test_convert_coldstart_grounding_to_qwen35_tool_call():
    conversation = [
        {"from": "system", "value": "legacy system"},
        {"from": "human", "value": "<image>\nWhat color is the coat?"},
        {
            "from": "gpt",
            "value": (
                "<think>Need a closer look.</think>"
                '<grounding>{"bbox_2d": [0.1, 0.2, 0.3, 0.4], "source": "original_image"}</grounding>'
            ),
        },
        {
            "from": "human",
            "value": (
                "After the above Action 0, here is the the zoom-in image (Observation 1):\n"
                "<image>.\nContinue your reasoning process inside <think> and </think>."
            ),
        },
        {
            "from": "gpt",
            "value": "<think>The crop is enough.</think><answer> Orange. </answer>",
        },
    ]
    row = {
        "conversations": json.dumps(conversation),
        "images": [{"bytes": b"root", "path": ""}, {"bytes": b"crop", "path": ""}],
        "data_source": "unit",
        "sample_index": 3,
        "rollout_index": 4,
        "image_names": ["root.png", "crop.png"],
    }

    converted = convert_row(row, system_prompt=OFFICIAL_ZOOM_FINAL_SENTENCE_SYSTEM_PROMPT)
    messages = converted["messages"]

    assert messages[0] == {"role": "system", "content": OFFICIAL_ZOOM_FINAL_SENTENCE_SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "<image>\nWhat color is the coat?"}
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "<think>Need a closer look.</think>"
    assert messages[2]["tool_calls"] == [
        {
            "type": "function",
            "function": {
                "name": "image_zoom_in_tool",
                "arguments": {
                    "bbox_2d": [100, 200, 300, 400],
                    "label": "selected region",
                    "img_idx": 0,
                },
            },
        }
    ]
    assert messages[3] == {"role": "tool", "content": "<image>"}
    assert messages[4] == {"role": "assistant", "content": "<think>The crop is enough.</think>Final answer: Orange."}
    assert converted["tools"][0]["function"]["parameters"]["required"] == ["bbox_2d", "label", "img_idx"]
    assert converted["enable_thinking"] is True
    assert "min_pixels" not in converted["images"][0]
    assert "max_pixels" not in converted["images"][0]

    converted_with_pixels = convert_row(
        row,
        system_prompt=OFFICIAL_ZOOM_FINAL_SENTENCE_SYSTEM_PROMPT,
        embed_image_pixel_limits=True,
    )
    assert converted_with_pixels["images"][0]["min_pixels"] == 40000
    assert converted_with_pixels["images"][0]["max_pixels"] == 2000000


def test_convert_observation_source_to_zero_based_img_idx():
    conversation = [
        {"from": "human", "value": "<image>\nQuestion"},
        {
            "from": "gpt",
            "value": '<think>x</think><grounding>{bbox_2d: [100, 200, 300, 400], source: observation_2}</grounding>',
        },
        {
            "from": "human",
            "value": "After the above Action 0, here is the the zoom-in image (Observation 1):\n<image>.",
        },
        {
            "from": "human",
            "value": "After the above Action 1, here is the the zoom-in image (Observation 2):\n<image>.",
        },
    ]
    row = {
        "conversations": conversation,
        "images": [{"bytes": b"0"}, {"bytes": b"1"}, {"bytes": b"2"}],
        "data_source": "unit",
        "sample_index": 1,
        "rollout_index": 2,
    }

    converted = convert_row(row, system_prompt=OFFICIAL_ZOOM_FINAL_SENTENCE_SYSTEM_PROMPT)

    assert converted["messages"][2]["tool_calls"][0]["function"]["arguments"] == {
        "bbox_2d": [100, 200, 300, 400],
        "label": "selected region",
        "img_idx": 2,
    }
