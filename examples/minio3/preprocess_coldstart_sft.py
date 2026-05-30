#!/usr/bin/env python3
"""Convert Mini-o3 cold-start SFT parquet to Qwen3.5 official tool-call SFT parquet."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from examples.minio3.preprocess_visualprobe import OFFICIAL_ZOOM_FINAL_SENTENCE_SYSTEM_PROMPT
from examples.minio3.preprocess_visualprobe import OFFICIAL_ZOOM_PLAIN_QUESTION_SYSTEM_PROMPT
from examples.minio3.preprocess_visualprobe import OFFICIAL_ZOOM_SYSTEM_PROMPT


GROUNDING_RE = re.compile(r"<grounding>\s*(\{.*?\})\s*</grounding>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
OBSERVATION_RE = re.compile(r"observation_(\d+)$")

SYSTEM_PROMPTS = {
    "qwen35_official_zoom_tool": OFFICIAL_ZOOM_SYSTEM_PROMPT,
    "qwen35_official_zoom_tool_plain_question": OFFICIAL_ZOOM_PLAIN_QUESTION_SYSTEM_PROMPT,
    "qwen35_official_zoom_tool_final_sentence": OFFICIAL_ZOOM_FINAL_SENTENCE_SYSTEM_PROMPT,
}


def official_zoom_tool_schema(tool_name: str = "image_zoom_in_tool") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": "Zoom into a specific rectangular region of an image and return the cropped zoom-in image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox_2d": {
                        "type": "array",
                        "description": "Bounding box [x1, y1, x2, y2] in relative coordinates from 0 to 1000.",
                    },
                    "label": {
                        "type": "string",
                        "description": "Name or short description of the object or region in the bounding box.",
                    },
                    "img_idx": {
                        "type": "number",
                        "description": "Index of the image to crop, starting from 0.",
                    },
                },
                "required": ["bbox_2d", "label", "img_idx"],
            },
        },
    }


def _plain(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, bytes | bytearray | str):
        return _plain(value.tolist())
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(val) for key, val in value.items()}
    return value


def _load_conversation(value: Any) -> list[dict[str, Any]]:
    value = _plain(value)
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError(f"conversations must be a list or JSON string, got {type(value).__name__}")
    return value


def _parse_grounding_payload(text: str) -> dict[str, Any]:
    for parser in (json.loads, ast.literal_eval, yaml.safe_load):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Could not parse grounding payload: {text[:200]!r}")


def _source_to_img_idx(source: Any) -> int:
    if source in (None, "", "original_image"):
        return 0
    if isinstance(source, int):
        return source
    if isinstance(source, float) and source.is_integer():
        return int(source)
    if isinstance(source, str):
        stripped = source.strip()
        if stripped.isdigit():
            return int(stripped)
        match = OBSERVATION_RE.match(stripped)
        if match:
            return int(match.group(1))
    raise ValueError(f"Unsupported grounding source: {source!r}")


def _convert_bbox(bbox: Any) -> list[int]:
    bbox = _plain(bbox)
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        raise ValueError(f"bbox_2d must be a list of four numbers, got {bbox!r}")
    values = [float(item) for item in bbox]
    if max(abs(item) for item in values) <= 1.5:
        values = [item * 1000.0 for item in values]
    return [max(0, min(1000, int(round(item)))) for item in values]


def _normalize_answer(text: str) -> str:
    return " ".join(text.strip().split())


def _convert_assistant_message(
    value: str,
    *,
    tool_name: str,
    label_fallback: str,
) -> dict[str, Any]:
    grounding_matches = list(GROUNDING_RE.finditer(value))
    answer_matches = list(ANSWER_RE.finditer(value))

    content = GROUNDING_RE.sub("", value)
    content = ANSWER_RE.sub(lambda match: f"Final answer: {_normalize_answer(match.group(1))}", content)
    content = content.strip()

    if grounding_matches:
        tool_calls = []
        for match in grounding_matches:
            payload = _parse_grounding_payload(match.group(1))
            tool_calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": {
                            "bbox_2d": _convert_bbox(payload.get("bbox_2d")),
                            "label": str(payload.get("label") or label_fallback),
                            "img_idx": _source_to_img_idx(payload.get("source", "original_image")),
                        },
                    },
                }
            )
        return {"role": "assistant", "content": content, "tool_calls": tool_calls}

    if answer_matches and "Final answer:" not in content:
        raise ValueError("answer conversion failed unexpectedly")
    return {"role": "assistant", "content": content}


def _is_observation_message(value: str) -> bool:
    return "Observation" in value and "<image>" in value and "zoom-in image" in value


def _count_image_placeholders(messages: list[dict[str, Any]]) -> int:
    count = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            count += content.count("<image>")
    return count


def convert_conversation(
    conversation: Any,
    *,
    system_prompt: str,
    tool_name: str = "image_zoom_in_tool",
    label_fallback: str = "selected region",
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for item in _load_conversation(conversation):
        role = item.get("from") or item.get("role")
        value = str(item.get("value", item.get("content", "")))
        if role == "system":
            continue
        if role in {"human", "user"}:
            if _is_observation_message(value):
                messages.append({"role": "tool", "content": "<image>"})
            else:
                messages.append({"role": "user", "content": value.strip()})
            continue
        if role in {"gpt", "assistant"}:
            messages.append(
                _convert_assistant_message(value, tool_name=tool_name, label_fallback=label_fallback)
            )
            continue
        raise ValueError(f"Unsupported conversation role: {role!r}")
    return messages


def convert_row(
    row: dict[str, Any],
    *,
    system_prompt: str,
    tool_name: str = "image_zoom_in_tool",
    label_fallback: str = "selected region",
    strict_image_count: bool = True,
    min_pixels: int = 40000,
    max_pixels: int = 2000000,
    embed_image_pixel_limits: bool = False,
) -> dict[str, Any]:
    messages = convert_conversation(
        row["conversations"],
        system_prompt=system_prompt,
        tool_name=tool_name,
        label_fallback=label_fallback,
    )
    images = _plain(row.get("images", []))
    if embed_image_pixel_limits:
        images = [
            {**image, "min_pixels": min_pixels, "max_pixels": max_pixels} if isinstance(image, dict) else image
            for image in images
        ]
    if strict_image_count and _count_image_placeholders(messages) != len(images):
        raise ValueError(
            "Image placeholder count does not match images column: "
            f"{_count_image_placeholders(messages)} placeholders vs {len(images)} images"
        )
    sample_index = _plain(row.get("sample_index"))
    rollout_index = _plain(row.get("rollout_index"))
    uid = f"{row.get('data_source', 'minio3_coldstart')}:{sample_index}:{rollout_index}"
    return {
        "uid": uid,
        "messages": messages,
        "images": images,
        "tools": [official_zoom_tool_schema(tool_name)],
        "enable_thinking": True,
        "data_source": row.get("data_source", "minio3_coldstart"),
        "sample_index": sample_index,
        "rollout_index": rollout_index,
        "image_names": _plain(row.get("image_names", [])),
    }


def _resolve_inputs(paths: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            data_dir = path / "data"
            search_dir = data_dir if data_dir.is_dir() else path
            resolved.extend(sorted(search_dir.glob("*.parquet")))
        else:
            resolved.append(path)
    if not resolved:
        raise ValueError("No input parquet files found")
    return resolved


def convert_files(args: argparse.Namespace) -> None:
    if args.system_prompt_suite not in SYSTEM_PROMPTS:
        raise ValueError(f"Unsupported system prompt suite: {args.system_prompt_suite!r}")
    input_paths = _resolve_inputs(args.input)
    frames = []
    remaining = args.limit
    system_prompt = SYSTEM_PROMPTS[args.system_prompt_suite]
    for path in input_paths:
        frame = pd.read_parquet(path)
        if remaining is not None:
            frame = frame.head(max(remaining, 0))
            remaining -= len(frame)
        frames.append(frame)
        if remaining is not None and remaining <= 0:
            break
    source = pd.concat(frames, ignore_index=True)

    converted = [
        convert_row(
            row.to_dict(),
            system_prompt=system_prompt,
            tool_name=args.tool_name,
            label_fallback=args.label_fallback,
            strict_image_count=not args.no_strict_image_count,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            embed_image_pixel_limits=args.embed_image_pixel_limits,
        )
        for _, row in source.iterrows()
    ]
    output = Path(args.output)
    dataframe = pd.DataFrame(converted)
    if args.rows_per_shard > 0:
        output.mkdir(parents=True, exist_ok=True)
        num_shards = math.ceil(len(dataframe) / args.rows_per_shard)
        for shard_idx in range(num_shards):
            start = shard_idx * args.rows_per_shard
            end = min(len(dataframe), start + args.rows_per_shard)
            shard_path = output / f"train-{shard_idx:05d}-of-{num_shards:05d}.parquet"
            dataframe.iloc[start:end].to_parquet(shard_path, index=False, compression=args.compression)
        print(f"wrote {len(dataframe)} rows to {output} across {num_shards} shards")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(output, index=False, compression=args.compression)
        print(f"wrote {len(dataframe)} rows to {output}")
    print("recommended SFT config: data.apply_chat_template_kwargs.add_vision_id=True")
    if not args.embed_image_pixel_limits:
        print(f"recommended SFT config: data.image_min_pixels={args.min_pixels} data.image_max_pixels={args.max_pixels}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Input coldstart parquet file(s) or directory.")
    parser.add_argument("--output", required=True, help="Output parquet path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    parser.add_argument(
        "--system-prompt-suite",
        default="qwen35_official_zoom_tool_final_sentence",
        choices=sorted(SYSTEM_PROMPTS),
    )
    parser.add_argument("--tool-name", default="image_zoom_in_tool")
    parser.add_argument("--label-fallback", default="selected region")
    parser.add_argument("--min-pixels", type=int, default=40000)
    parser.add_argument("--max-pixels", type=int, default=2000000)
    parser.add_argument(
        "--embed-image-pixel-limits",
        action="store_true",
        help="Embed min_pixels/max_pixels into each image payload. Disabled by default for byte-image parquet compatibility.",
    )
    parser.add_argument("--compression", default="zstd", help="Parquet compression codec.")
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=0,
        help="Write output as a directory of parquet shards with this many rows per shard. 0 writes one parquet file.",
    )
    parser.add_argument("--no-strict-image-count", action="store_true")
    return parser


def main() -> None:
    convert_files(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
