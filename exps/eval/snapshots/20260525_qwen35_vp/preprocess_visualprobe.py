#!/usr/bin/env python3
"""Convert Mini-o3 JSON data into official verl RLHF parquet format."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import datasets


LEGACY_GROUNDING_PROMPT_SUITE = "qwen35_minio3_legacy_grounding"
OFFICIAL_ZOOM_PROMPT_SUITE = "qwen35_official_zoom_tool"
OFFICIAL_ZOOM_PLAIN_QUESTION_PROMPT_SUITE = "qwen35_official_zoom_tool_plain_question"
PROMPT_SUITES = {
    LEGACY_GROUNDING_PROMPT_SUITE,
    OFFICIAL_ZOOM_PROMPT_SUITE,
    OFFICIAL_ZOOM_PLAIN_QUESTION_PROMPT_SUITE,
}

TOOL_CROP_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question based on the image provided. "
    "Output your thinking process within the <think> and </think> tags. Whenever you find "
    "anything unclear, you can zoom in a specific region in the given image to see more clearly "
    "by outputing <grounding>{\"bbox_2d\": [x0, y0, x1, y1], \"source\": \"original_image\"}</grounding>, "
    "where (x0, y0) and (x1, y1) are the top-left and bottom-right coordinates of the region "
    "that you want to zoom in, respectively (suppose the width and height of the image are 1.0), "
    "and 'source' refers to the image that you zoom in and could be either 'original_image' or "
    "'observation_i'. Once the final answer is confirmed, put it within <answer> and </answer>."
)

OFFICIAL_ZOOM_SYSTEM_PROMPT = (
    "You are a visual research assistant. Answer the user's image question by examining the image "
    "carefully and using the available zoom tool when visual details are unclear.\n\n"
    "For each question, follow this loop:\n"
    "1. First inspect the image with the user's question in mind.\n"
    "2. State what is visible and what needs closer inspection.\n"
    "3. If needed, call the zoom tool on a precise region.\n"
    "4. Review the zoom observation before deciding whether another zoom is needed.\n"
    "5. When there is enough evidence, give the final answer inside <answer> and </answer>."
)

OFFICIAL_ZOOM_PLAIN_QUESTION_SYSTEM_PROMPT = (
    "You are a visual research assistant. Answer the user's image question by examining the image "
    "carefully and using the available zoom tool when visual details are unclear.\n\n"
    "For each question, follow this loop:\n"
    "1. First inspect the image with the user's question in mind.\n"
    "2. State what is visible and what needs closer inspection.\n"
    "3. If needed, call the zoom tool on a precise region.\n"
    "4. Review the zoom observation before deciding whether another zoom is needed."
)


def _is_official_zoom_suite(tool_prompt_suite: str) -> bool:
    return tool_prompt_suite in {OFFICIAL_ZOOM_PROMPT_SUITE, OFFICIAL_ZOOM_PLAIN_QUESTION_PROMPT_SUITE}


def _system_prompt_for_suite(tool_prompt_suite: str) -> str:
    if tool_prompt_suite == LEGACY_GROUNDING_PROMPT_SUITE:
        return TOOL_CROP_SYSTEM_PROMPT
    if tool_prompt_suite == OFFICIAL_ZOOM_PROMPT_SUITE:
        return OFFICIAL_ZOOM_SYSTEM_PROMPT
    if tool_prompt_suite == OFFICIAL_ZOOM_PLAIN_QUESTION_PROMPT_SUITE:
        return OFFICIAL_ZOOM_PLAIN_QUESTION_SYSTEM_PROMPT
    raise ValueError(f"Unsupported tool prompt suite: {tool_prompt_suite!r}")


def _default_agent_name(tool_prompt_suite: str) -> str:
    if _is_official_zoom_suite(tool_prompt_suite):
        return "tool_agent"
    return "mini_o3_tool_agent"


def _load_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(rows).__name__}")
    return rows


def _resolve_image_path(image_root: str | None, image_path: str) -> str:
    if os.path.isabs(image_path) or image_root is None:
        return image_path
    return str(Path(image_root) / image_path)


def _build_prompt(problem: str, image_count: int) -> str:
    if "<image>" in problem:
        return problem
    image_prefix = "\n".join(["<image>"] * max(image_count, 1))
    return f"{image_prefix}\n{problem}"


def _convert_row(
    row: dict[str, Any],
    idx: int,
    split: str,
    image_root: str | None,
    min_pixels: int,
    max_pixels: int,
    tool_prompt_suite: str = LEGACY_GROUNDING_PROMPT_SUITE,
    official_tool_name: str = "tool_crop",
    agent_name: str | None = None,
) -> dict[str, Any]:
    if tool_prompt_suite not in PROMPT_SUITES:
        raise ValueError(f"tool_prompt_suite must be one of {sorted(PROMPT_SUITES)}, got {tool_prompt_suite!r}")

    images = row.get("images") or []
    if not isinstance(images, list):
        raise ValueError(f"row {idx}: images must be a list")

    problem = row.get("problem") or row.get("question") or row.get("prompt")
    if not isinstance(problem, str):
        raise ValueError(f"row {idx}: missing string problem/question/prompt")

    answer = row.get("solution", row.get("answer", row.get("ground_truth", "")))
    data_source = row.get("data_source") or f"minio3_{split}"
    image_paths = [_resolve_image_path(image_root, path) for path in images]
    image_payload = [{"image": path, "min_pixels": min_pixels, "max_pixels": max_pixels} for path in image_paths]
    selected_tool = official_tool_name if _is_official_zoom_suite(tool_prompt_suite) else "tool_crop"
    selected_agent = agent_name or _default_agent_name(tool_prompt_suite)

    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": _system_prompt_for_suite(tool_prompt_suite)},
            {"role": "user", "content": _build_prompt(problem, len(images))},
        ],
        "images": image_payload,
        "ability": "vision_qa",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "agent_name": selected_agent,
        "extra_info": {
            "split": split,
            "index": row.get("doc_id", idx),
            "doc_id": row.get("doc_id", idx),
            "image_paths": image_paths,
            "answer": answer,
            "question": problem,
            "tool_prompt_suite": tool_prompt_suite,
            "tool_selection": [selected_tool],
            "acc_reward_weight": 1.0,
            "format_reward_weight": 0.0,
            "tool_call_penalty": 0.0,
        },
    }


def _flatten_paths(paths: list[list[str]]) -> list[str]:
    return [path for group in paths for path in group]


def _write_split(
    input_jsons: list[str],
    output_path: str,
    split: str,
    image_root: str | None,
    limit: int | None,
    min_pixels: int,
    max_pixels: int,
    tool_prompt_suite: str,
    official_tool_name: str,
    agent_name: str | None,
) -> None:
    rows: list[dict[str, Any]] = []
    for input_json in input_jsons:
        rows.extend(_load_json(input_json))
    if limit is not None:
        rows = rows[:limit]
    converted = [
        _convert_row(
            row,
            idx,
            split,
            image_root,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            tool_prompt_suite=tool_prompt_suite,
            official_tool_name=official_tool_name,
            agent_name=agent_name,
        )
        for idx, row in enumerate(rows)
    ]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    datasets.Dataset.from_list(converted).to_parquet(output_path)
    print(f"wrote {output_path} rows={len(converted)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", nargs="+", action="append", required=True)
    parser.add_argument("--val-json", nargs="+", action="append", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--local-save-dir", default="data/minio3")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--min-pixels", type=int, default=40000)
    parser.add_argument("--max-pixels", type=int, default=1000000)
    parser.add_argument(
        "--tool-prompt-suite",
        choices=sorted(PROMPT_SUITES),
        default=os.environ.get("MINIO3_TOOL_PROMPT_SUITE", LEGACY_GROUNDING_PROMPT_SUITE),
    )
    parser.add_argument("--official-tool-name", default=os.environ.get("MINIO3_OFFICIAL_TOOL_NAME", "tool_crop"))
    parser.add_argument("--agent-name", default=os.environ.get("MINIO3_AGENT_LOOP") or None)
    args = parser.parse_args()
    train_jsons = _flatten_paths(args.train_json)
    val_jsons = _flatten_paths(args.val_json)

    _write_split(
        train_jsons,
        os.path.join(args.local_save_dir, "train.parquet"),
        "train",
        args.image_root,
        args.train_limit,
        args.min_pixels,
        args.max_pixels,
        args.tool_prompt_suite,
        args.official_tool_name,
        args.agent_name,
    )
    _write_split(
        val_jsons,
        os.path.join(args.local_save_dir, "val.parquet"),
        "val",
        args.image_root,
        args.val_limit,
        args.min_pixels,
        args.max_pixels,
        args.tool_prompt_suite,
        args.official_tool_name,
        args.agent_name,
    )


if __name__ == "__main__":
    main()
