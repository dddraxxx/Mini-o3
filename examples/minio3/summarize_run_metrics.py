#!/usr/bin/env python3
"""Summarize Mini-o3 run metrics and GPU utilization logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[2]
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PROMPT_LOAD_RE = re.compile(
    r"prompt_admission\.load reason=(?P<reason>\S+) "
    r"running_groups=(?P<running_groups>\d+) "
    r"total_inflight=(?P<total_inflight>\d+) "
    r"max_worker_inflight=(?P<max_worker_inflight>\d+) "
    r"worker_inflight=\[(?P<worker_inflight>[^\]]*)\] "
    r"age_max_s=(?P<age_max_s>[0-9.]+)"
)
TRAJ_RUNNING_RE = re.compile(
    r"worker\.traj\.running pid=(?P<pid>\d+) "
    r"active=(?P<active>\d+)/(?P<total>\d+) "
    r"age_max_s=(?P<age_max_s>[0-9.]+)"
)


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


def _counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _clean_log_line(line: str) -> str:
    return ANSI_RE.sub("", line).replace("\r", "")


def _parse_int_list(value: str) -> list[int]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


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


def _infer_log_path(run_dir: Path) -> Path | None:
    launch_env = run_dir / "launch_env.txt"
    if launch_env.exists():
        for line in launch_env.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("LOG="):
                path = Path(line.split("=", 1)[1])
                if path.exists():
                    return path

    candidates = [
        run_dir / "run.log",
        PROJECT_DIR / "run_logs" / f"{run_dir.name}.log",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _log_timeline_summary(run_dir: Path) -> dict[str, Any]:
    log_path = _infer_log_path(run_dir)
    prompt_rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    if log_path is None:
        return {
            "source": None,
            "prompt_load": {},
            "traj_active": {},
        }

    with log_path.open(encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = _clean_log_line(raw_line)
            prompt_match = PROMPT_LOAD_RE.search(line)
            if prompt_match:
                inflight = _parse_int_list(prompt_match.group("worker_inflight"))
                prompt_rows.append(
                    {
                        "line": line_no,
                        "reason": prompt_match.group("reason"),
                        "running_groups": int(prompt_match.group("running_groups")),
                        "total_inflight": int(prompt_match.group("total_inflight")),
                        "max_worker_inflight": int(prompt_match.group("max_worker_inflight")),
                        "nonzero_workers": sum(1 for value in inflight if value > 0),
                        "worker_count": len(inflight),
                        "age_max_s": float(prompt_match.group("age_max_s")),
                    }
                )
            traj_match = TRAJ_RUNNING_RE.search(line)
            if traj_match:
                active = int(traj_match.group("active"))
                total = int(traj_match.group("total"))
                traj_rows.append(
                    {
                        "line": line_no,
                        "pid": int(traj_match.group("pid")),
                        "active": active,
                        "total": total,
                        "active_ratio": active / total if total else 0.0,
                        "age_max_s": float(traj_match.group("age_max_s")),
                    }
                )

    return {
        "source": str(log_path),
        "prompt_load": {
            "samples": len(prompt_rows),
            "reason_counts": _counts([row["reason"] for row in prompt_rows]),
            "running_groups": _stats([float(row["running_groups"]) for row in prompt_rows]),
            "total_inflight": _stats([float(row["total_inflight"]) for row in prompt_rows]),
            "max_worker_inflight": _stats([float(row["max_worker_inflight"]) for row in prompt_rows]),
            "nonzero_workers": _stats([float(row["nonzero_workers"]) for row in prompt_rows]),
            "age_max_s": _stats([float(row["age_max_s"]) for row in prompt_rows]),
        },
        "traj_active": {
            "samples": len(traj_rows),
            "active": _stats([float(row["active"]) for row in traj_rows]),
            "total": _stats([float(row["total"]) for row in traj_rows]),
            "active_ratio": _stats([float(row["active_ratio"]) for row in traj_rows]),
            "age_max_s": _stats([float(row["age_max_s"]) for row in traj_rows]),
        },
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
        "log_timeline": _log_timeline_summary(run_dir),
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
        timeline = summary["log_timeline"]
        print(f"  log_source: {timeline['source']}")
        prompt = timeline["prompt_load"]
        if prompt.get("samples"):
            print(f"  prompt_load_samples: {prompt['samples']} reason_counts={prompt['reason_counts']}")
            for label in ("running_groups", "total_inflight", "max_worker_inflight", "nonzero_workers", "age_max_s"):
                stats = prompt[label]
                print(
                    f"  prompt_load/{label}: mean={stats['mean']:.2f} p50={stats['p50']:.2f} "
                    f"p95={stats['p95']:.2f} max={stats['max']:.2f}"
                )
        traj = timeline["traj_active"]
        if traj.get("samples"):
            print(f"  traj_active_samples: {traj['samples']}")
            for label in ("active", "total", "active_ratio", "age_max_s"):
                stats = traj[label]
                print(
                    f"  traj_active/{label}: mean={stats['mean']:.2f} p50={stats['p50']:.2f} "
                    f"p95={stats['p95']:.2f} max={stats['max']:.2f}"
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
