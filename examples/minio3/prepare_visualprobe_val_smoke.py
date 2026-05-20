#!/usr/bin/env python3
"""Build a small stratified VisualProbe validation parquet for val-only smoke."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import datasets

from examples.minio3.preprocess_visualprobe import _convert_row


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
) -> None:
    converted = [
        _convert_row(row, idx, split, image_root, min_pixels=min_pixels, max_pixels=max_pixels)
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
    parser.add_argument("--min-pixels", type=int, default=40000)
    parser.add_argument("--max-pixels", type=int, default=2000000)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    image_root = args.image_root or os.fspath(dataset_root)
    out_dir = Path(args.local_save_dir)
    val_rows = _select_stratified(dataset_root, args.case_count)

    _write_parquet(
        val_rows[:1],
        out_dir / "train.parquet",
        "train",
        image_root,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    _write_parquet(
        val_rows,
        out_dir / "val.parquet",
        "val",
        image_root,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )


if __name__ == "__main__":
    main()
