#!/usr/bin/env python3
"""Fail fast when the active env cannot load Qwen3.5 models."""

from __future__ import annotations

import argparse
import importlib.util
import sys

import torch
import transformers
from transformers import AutoConfig
import vllm


REQUIRED_MODULES = (
    ("transformers.models.qwen3_5", "transformers Qwen3.5 model registry"),
    ("vllm.model_executor.models.qwen3_5", "vLLM Qwen3.5 executor"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    print(
        "qwen35_env_check "
        f"model={args.model_path} "
        f"transformers={transformers.__version__} "
        f"vllm={vllm.__version__} "
        f"torch={torch.__version__}"
    )

    missing = [
        f"{module} ({label})"
        for module, label in REQUIRED_MODULES
        if importlib.util.find_spec(module) is None
    ]
    errors = []
    if missing:
        errors.extend(missing)

    try:
        config = AutoConfig.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:  # noqa: BLE001 - preflight should print the concrete failure.
        errors.append(f"AutoConfig load failed: {type(exc).__name__}: {exc}")
    else:
        model_type = getattr(config, "model_type", None)
        if model_type != "qwen3_5":
            errors.append(f"AutoConfig model_type={model_type!r}, expected 'qwen3_5'")

    if not errors:
        return 0

    print("Qwen3.5 preflight failed:", file=sys.stderr)
    for item in errors:
        print(f"  - {item}", file=sys.stderr)
    print(
        "Use a Qwen3.5-capable env before launching rollout; otherwise Ray/vLLM "
        "may start and hang before generation.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
