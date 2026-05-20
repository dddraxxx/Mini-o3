#!/usr/bin/env python3
"""Log GPU utilization as JSONL.

This is intended for long Mini-o3 training runs where one-off GPU snapshots are
too noisy. The output is one JSON object per sampling tick with a list of
per-GPU records.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


_STOP = False


def _handle_stop(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True


def _decode(raw: bytes | str) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _to_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    cleaned = value.strip().replace("MiB", "").replace("%", "").strip()
    if not cleaned or cleaned.lower() in {"[not supported]", "[n/a]", "n/a"}:
        return None
    return int(float(cleaned))


def _run_nvidia_smi(args: list[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["nvidia-smi", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def _sample_nvidia_smi(timeout_s: float) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    gpu_query = [
        "--query-gpu=index,pci.bus_id,utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu_result = _run_nvidia_smi(gpu_query, timeout_s)
    except subprocess.TimeoutExpired:
        return [], [f"nvidia-smi gpu query timed out after {timeout_s}s"]

    if gpu_result.returncode != 0:
        errors.append(gpu_result.stderr.strip() or f"nvidia-smi gpu query failed: {gpu_result.returncode}")

    gpus_by_bus: dict[str, dict] = {}
    reader = csv.reader(gpu_result.stdout.splitlines())
    for row in reader:
        if len(row) < 6:
            continue
        index, bus_id, gpu_util, mem_util, mem_used, mem_total = [item.strip() for item in row[:6]]
        gpu = {
            "index": _to_int(index),
            "bus_id": bus_id,
            "gpu_util_pct": _to_int(gpu_util),
            "mem_util_pct": _to_int(mem_util),
            "memory_used_mib": _to_int(mem_used),
            "memory_total_mib": _to_int(mem_total),
            "processes": [],
        }
        gpus_by_bus[bus_id] = gpu

    proc_query = [
        "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc_result = _run_nvidia_smi(proc_query, timeout_s)
    except subprocess.TimeoutExpired:
        errors.append(f"nvidia-smi process query timed out after {timeout_s}s")
    else:
        if proc_result.returncode != 0 and proc_result.stderr.strip():
            errors.append(proc_result.stderr.strip())
        reader = csv.reader(proc_result.stdout.splitlines())
        for row in reader:
            if len(row) < 4:
                continue
            bus_id, pid, name, used_mem = [item.strip() for item in row[:4]]
            gpu = gpus_by_bus.get(bus_id)
            if gpu is None:
                continue
            gpu["processes"].append(
                {
                    "pid": _to_int(pid),
                    "name": name,
                    "used_memory_mib": _to_int(used_mem),
                }
            )

    return list(gpus_by_bus.values()), errors


def _processes(handle) -> list[dict[str, int | str | None]]:
    import pynvml

    procs = []
    try:
        running = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    except pynvml.NVMLError:
        return procs
    for proc in running:
        name = ""
        try:
            name = _decode(pynvml.nvmlSystemGetProcessName(proc.pid))
        except pynvml.NVMLError:
            pass
        procs.append(
            {
                "pid": int(proc.pid),
                "name": name,
                "used_memory_mib": int(proc.usedGpuMemory // (1024 * 1024)) if proc.usedGpuMemory else None,
            }
        )
    return procs


def _sample_nvml() -> list[dict]:
    import pynvml

    gpus = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpus.append(
            {
                "index": index,
                "gpu_util_pct": int(util.gpu),
                "mem_util_pct": int(util.memory),
                "memory_used_mib": int(mem.used // (1024 * 1024)),
                "memory_total_mib": int(mem.total // (1024 * 1024)),
                "processes": _processes(handle),
            }
        )
    return gpus


def sample_once(backend: str, sample_timeout_s: float) -> dict:
    now = time.time()
    errors: list[str] = []
    if backend == "nvml":
        gpus = _sample_nvml()
    else:
        gpus, errors = _sample_nvidia_smi(sample_timeout_s)
    return {
        "timestamp_s": now,
        "time_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "backend": backend,
        "gpus": gpus,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="JSONL path to write.")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds.")
    parser.add_argument("--flush-every", type=int, default=1, help="Flush after this many samples.")
    parser.add_argument(
        "--backend",
        choices=("nvidia-smi", "nvml"),
        default="nvml",
        help="Sampling backend. NVML is the default; use nvidia-smi as a subprocess fallback if needed.",
    )
    parser.add_argument("--sample-timeout", type=float, default=5.0, help="Per nvidia-smi query timeout.")
    args = parser.parse_args()

    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    if args.flush_every <= 0:
        raise SystemExit("--flush-every must be positive")
    if args.sample_timeout <= 0:
        raise SystemExit("--sample-timeout must be positive")

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.backend == "nvml":
        import pynvml

        pynvml.nvmlInit()
    count = 0
    try:
        with output.open("a", encoding="utf-8") as f:
            while not _STOP:
                started = time.monotonic()
                row = sample_once(args.backend, args.sample_timeout)
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
                count += 1
                if count % args.flush_every == 0:
                    f.flush()
                elapsed = time.monotonic() - started
                sleep_for = max(0.0, args.interval - elapsed)
                time.sleep(sleep_for)
    finally:
        if args.backend == "nvml":
            import pynvml

            pynvml.nvmlShutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
