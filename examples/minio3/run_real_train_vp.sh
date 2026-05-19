#!/usr/bin/env bash
# Backward-compatible alias for the A100 PyVision-style Mini-o3 train profile.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
exec bash "$PROJECT_DIR/examples/minio3/run_real_train_pyvision_style_a100.sh" "$@"
