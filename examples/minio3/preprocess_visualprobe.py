#!/usr/bin/env python3
"""Convert Mini-o3 JSON data into official verl RLHF parquet format."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import datasets


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
) -> dict[str, Any]:
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

    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": TOOL_CROP_SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(problem, len(images))},
        ],
        "images": image_payload,
        "ability": "vision_qa",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "agent_name": "mini_o3_tool_agent",
        "extra_info": {
            "split": split,
            "index": row.get("doc_id", idx),
            "doc_id": row.get("doc_id", idx),
            "image_paths": image_paths,
            "answer": answer,
            "question": problem,
            "tool_selection": ["tool_crop"],
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
) -> None:
    rows: list[dict[str, Any]] = []
    for input_json in input_jsons:
        rows.extend(_load_json(input_json))
    if limit is not None:
        rows = rows[:limit]
    converted = [
        _convert_row(row, idx, split, image_root, min_pixels=min_pixels, max_pixels=max_pixels)
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
    )
    _write_split(
        val_jsons,
        os.path.join(args.local_save_dir, "val.parquet"),
        "val",
        args.image_root,
        args.val_limit,
        args.min_pixels,
        args.max_pixels,
    )


if __name__ == "__main__":
    main()
