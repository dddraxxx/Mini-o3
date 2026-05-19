#!/usr/bin/env python3
"""Summarize Mini-o3 run metrics and GPU utilization logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _metric_data(row: dict[str, Any]) -> dict[str, Any]:
    data = row.get("data")
    return data if isinstance(data, dict) else row


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "mean": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "samples": len(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _parse_percent(value: str) -> float:
    return float(value.strip().replace("%", "").strip())


def _parse_mib(value: str) -> float:
    return float(value.strip().replace("MiB", "").strip())


def _gpu_samples_from_nvidia_smi_csv(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    samples = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {key.strip(): value for key, value in row.items() if key is not None}
            try:
                samples.append(
                    {
                        "index": float(row["index"].strip()),
                        "gpu_util_pct": _parse_percent(row["utilization.gpu [%]"]),
                        "memory_used_mib": _parse_mib(row["memory.used [MiB]"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return samples


def _gpu_samples_from_nvml_jsonl(path: Path) -> list[dict[str, float]]:
    samples = []
    for row in _jsonl(path):
        for gpu in row.get("gpus", []):
            try:
                samples.append(
                    {
                        "index": float(gpu["index"]),
                        "gpu_util_pct": float(gpu["gpu_util_pct"]),
                        "memory_used_mib": float(gpu["memory_used_mib"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return samples


def _gpu_summary(run_dir: Path, active_memory_mib: float) -> dict[str, Any]:
    jsonl_path = run_dir / "gpu_util.jsonl"
    legacy_nvml_path = run_dir / "gpu_util_nvml.jsonl"
    csv_path = run_dir / "gpu_util.csv"
    if jsonl_path.exists():
        source = str(jsonl_path)
        samples = _gpu_samples_from_nvml_jsonl(jsonl_path)
    elif legacy_nvml_path.exists():
        source = str(legacy_nvml_path)
        samples = _gpu_samples_from_nvml_jsonl(legacy_nvml_path)
    else:
        source = str(csv_path) if csv_path.exists() else None
        samples = _gpu_samples_from_nvidia_smi_csv(csv_path)

    all_utils = [s["gpu_util_pct"] for s in samples]
    active = [s for s in samples if s["memory_used_mib"] >= active_memory_mib]
    active_utils = [s["gpu_util_pct"] for s in active]
    active_mem = [s["memory_used_mib"] for s in active]
    return {
        "source": source,
        "all_gpu_util": _stats(all_utils),
        "active_gpu_util": _stats(active_utils),
        "active_memory_mib": _stats(active_mem),
        "active_memory_threshold_mib": active_memory_mib,
    }


def summarize_run(run_dir: Path, active_memory_mib: float) -> dict[str, Any]:
    metrics_rows = _jsonl(run_dir / "train_step_metrics.jsonl")
    metrics = _metric_data(metrics_rows[-1]) if metrics_rows else {}
    admission_rows = _jsonl(run_dir / "prompt_admission_state.jsonl")
    admission = admission_rows[-1].get("metrics", {}) if admission_rows else {}

    keys = [
        "timing_s/gen",
        "timing_s/old_log_prob",
        "timing_s/update_actor",
        "timing_s/update_weights",
        "timing_s/step",
        "perf/mfu/actor_infer",
        "perf/mfu/actor",
        "perf/throughput",
        "perf/total_num_tokens",
        "response_length/mean",
        "response_length/max",
        "num_turns/mean",
        "num_turns/max",
        "prompt_admission/submitted_groups",
        "prompt_admission/accepted_groups",
        "prompt_admission/rejected_groups",
    ]
    selected = {key: metrics.get(key, admission.get(key)) for key in keys}

    exit_path = run_dir / "exit_code.txt"
    return {
        "run_dir": str(run_dir),
        "exit_code": exit_path.read_text(encoding="utf-8").strip() if exit_path.exists() else None,
        "metrics_lines": len(metrics_rows),
        "prompt_admission_lines": len(admission_rows),
        "metrics": selected,
        "gpu": _gpu_summary(run_dir, active_memory_mib),
    }


def _print_text(summaries: list[dict[str, Any]]) -> None:
    for summary in summaries:
        print(f"run: {summary['run_dir']}")
        print(f"  exit_code: {summary['exit_code']} metrics_lines: {summary['metrics_lines']}")
        metrics = summary["metrics"]
        for key, value in metrics.items():
            if value is not None:
                print(f"  {key}: {value}")
        gpu = summary["gpu"]
        print(f"  gpu_source: {gpu['source']}")
        for label in ("all_gpu_util", "active_gpu_util", "active_memory_mib"):
            stats = gpu[label]
            if stats["samples"]:
                print(
                    f"  {label}: mean={stats['mean']:.2f} p50={stats['p50']:.2f} "
                    f"p95={stats['p95']:.2f} max={stats['max']:.2f} samples={stats['samples']}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--active-memory-mib",
        type=float,
        default=10_000.0,
        help="Only GPU samples above this memory are considered active training samples.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args()

    summaries = [summarize_run(path, args.active_memory_mib) for path in args.run_dirs]
    if args.json:
        print(json.dumps(summaries, indent=2, sort_keys=True))
    else:
        _print_text(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
