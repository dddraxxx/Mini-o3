#!/usr/bin/env python3
"""Build a small stratified VisualProbe validation parquet for val-only smoke."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.minio3.preprocess_visualprobe import _convert_row
from examples.minio3.preprocess_visualprobe import LEGACY_GROUNDING_PROMPT_SUITE


SPLITS = (
    ("easy", "VisualProbe_Easy/val.json"),
    ("medium", "VisualProbe_Medium/val.json"),
    ("hard", "VisualProbe_Hard/val.json"),
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON list in {path}, got {type(rows).__name__}")
    return rows


def _select_stratified(dataset_root: Path, case_count: int) -> list[dict[str, Any]]:
    split_rows = [(name, _load_rows(dataset_root / rel_path)) for name, rel_path in SPLITS]
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < case_count:
        progressed = False
        for split_name, rows in split_rows:
            if offset >= len(rows):
                continue
            row = dict(rows[offset])
            row.setdefault("data_source", f"visual_probe_{split_name}")
            row.setdefault("extra_info", {})
            selected.append(row)
            progressed = True
            if len(selected) >= case_count:
                break
        if not progressed:
            break
        offset += 1
    if len(selected) < case_count:
        raise RuntimeError(f"Requested {case_count} cases, only selected {len(selected)}")
    return selected


def _write_parquet(
    rows: list[dict[str, Any]],
    output_path: Path,
    split: str,
    image_root: str,
    min_pixels: int,
    max_pixels: int,
    tool_prompt_suite: str,
    official_tool_name: str,
    agent_name: str | None,
) -> None:
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(converted).to_parquet(str(output_path))
    print(f"wrote {output_path} rows={len(converted)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--local-save-dir", default="data/minio3_visualprobe_val_smoke10")
    parser.add_argument("--case-count", type=int, default=10)
    parser.add_argument("--train-case-count", type=int, default=1)
    parser.add_argument("--min-pixels", type=int, default=40000)
    parser.add_argument("--max-pixels", type=int, default=2000000)
    parser.add_argument("--tool-prompt-suite", default=os.environ.get("MINIO3_TOOL_PROMPT_SUITE", LEGACY_GROUNDING_PROMPT_SUITE))
    parser.add_argument("--official-tool-name", default=os.environ.get("MINIO3_OFFICIAL_TOOL_NAME", "tool_crop"))
    parser.add_argument("--agent-name", default=os.environ.get("MINIO3_AGENT_LOOP") or None)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    image_root = args.image_root or os.fspath(dataset_root)
    out_dir = Path(args.local_save_dir)
    if args.train_case_count < 1:
        raise ValueError("--train-case-count must be >= 1")
    selected_rows = _select_stratified(dataset_root, max(args.case_count, args.train_case_count))
    val_rows = selected_rows[: args.case_count]
    train_rows = selected_rows[: args.train_case_count]

    _write_parquet(
        train_rows,
        out_dir / "train.parquet",
        "train",
        image_root,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        tool_prompt_suite=args.tool_prompt_suite,
        official_tool_name=args.official_tool_name,
        agent_name=args.agent_name,
    )
    _write_parquet(
        val_rows,
        out_dir / "val.parquet",
        "val",
        image_root,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        tool_prompt_suite=args.tool_prompt_suite,
        official_tool_name=args.official_tool_name,
        agent_name=args.agent_name,
    )


if __name__ == "__main__":
    main()
