#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)

PYTHON=${PYTHON:-"$PROJECT_DIR/.venv/bin/python"}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}
MAX_JOBS=${MAX_JOBS:-32}
TRANSFORMERS_REF=${TRANSFORMERS_REF:-cc7ab9be508ce6ed3637bba9e50367b29b742dc6}

uv pip install --python "$PYTHON" ninja packaging wheel setuptools

uv pip install --python "$PYTHON" \
  "vllm==0.18.0" \
  "qwen-vl-utils>=0.0.14" \
  "pandas>=2.3.0,<4"

uv pip install --python "$PYTHON" \
  "flashinfer-jit-cache==0.6.6+cu129" \
  --index-url https://flashinfer.ai/whl/cu129

uv pip install --python "$PYTHON" --no-deps \
  "transformers @ git+https://github.com/huggingface/transformers.git@$TRANSFORMERS_REF"

uv pip install --python "$PYTHON" \
  "huggingface-hub>=1.3.0,<2.0" \
  "typer>=0.25.1"

env MAX_JOBS="$MAX_JOBS" FLASH_ATTENTION_FORCE_BUILD=TRUE \
  uv pip install --python "$PYTHON" \
    --no-build-isolation \
    --no-cache \
    --reinstall-package flash-attn \
    "flash-attn==2.8.3"

env MAX_JOBS="$MAX_JOBS" \
  uv pip install --python "$PYTHON" \
    --no-build-isolation \
    "causal-conv1d>=1.6.2.post1" \
    "flash-linear-attention>=0.5.0"

uv run --python "$PYTHON" --no-sync python \
  "$PROJECT_DIR/examples/minio3/check_qwen35_env.py" \
  --model-path "$MODEL_PATH" \
  --local-files-only
