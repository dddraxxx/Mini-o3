# Mini-o3 GPU Monitoring

## Goal

`nvidia-smi` single snapshots are too noisy for Mini-o3 async rollout runs. The
repo now uses a lightweight continuous monitor so each run keeps a comparable GPU
utilization trace next to the trainer metrics.

Primary speed comparisons should still use verl trainer metrics:

- `timing_s/gen`
- `timing_s/step`
- `perf/throughput`
- acceptance counts from `prompt_admission_state.jsonl`

GPU utilization is an auxiliary signal for spotting idle gaps, memory pressure,
or uneven device usage.

## Scripts

Monitor:

```bash
uv run --active --no-sync python examples/minio3/monitor_gpu_util.py \
  --output "$RUN_DIR/gpu_util.jsonl" \
  --interval 1.0 \
  --backend nvidia-smi
```

Summarize:

```bash
uv run --active --no-sync python examples/minio3/summarize_run_metrics.py "$RUN_DIR"
```

The summarizer also reads `run_logs/<run-name>.log` when present and extracts
Mini-o3 stage-log load-balance samples:

- `prompt_admission.load`: running prompt groups, total in-flight groups,
  per-worker balance, and running age.
- `worker.traj.running`: active trajectories inside each agent-loop worker.

Set these when running throughput experiments that need load-balance evidence:

```bash
MINIO3_STAGE_LOG=1
MINIO3_TRAJ_STATUS_INTERVAL_S=15
```

The monitor writes one JSON object per sampling tick. Each row includes:

- wall-clock timestamp
- backend name
- per-GPU utilization percent
- per-GPU memory used / total
- compute processes on that GPU
- sampling errors, if any

## Default Train Integration

`examples/minio3/run_qwen3_vl_8b_crop_lora_fsdp.sh` starts the monitor
automatically when `RUN_DIR` is set:

```bash
GPU_MONITOR_ENABLE=True
GPU_MONITOR_PATH="$RUN_DIR/gpu_util.jsonl"
GPU_MONITOR_INTERVAL=1.0
GPU_MONITOR_BACKEND=nvidia-smi
GPU_MONITOR_SAMPLE_TIMEOUT=5.0
```

The default backend is `nvidia-smi --query-*` with a per-sample timeout. This is
not a single manual snapshot; it is a continuous trace. A direct NVML backend is
available with `GPU_MONITOR_BACKEND=nvml`, but it is not the default because NVML
can block on this node.

## Accuracy Notes

- `perf/throughput` is exact relative to verl's recorded token count and step
  timer, but it is whole-step throughput, not pure generation throughput.
- `perf/mfu/*` is a FLOPs estimate from model config and token lengths. Treat it
  as a trend signal, not a hardware-accurate counter.
- `gpu_util.jsonl` is also a sampling signal. Use mean / p50 / p95 over the run
  window instead of inspecting a single point.

For max-num-seqs comparisons, prefer this order:

1. `timing_s/gen`
2. `timing_s/step`
3. `perf/throughput`
4. accepted / rejected prompt groups
5. GPU utilization summary
