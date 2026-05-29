#!/usr/bin/env bash
# Frozen launcher for qwen35_9b_official_tool_h200_rl_20260527_190822.
#
# Experiment profile:
# - train turn limit: MAX_ASSISTANT_TURNS=12, MAX_USER_TURNS=12
# - val turn limit: VAL_MAX_ASSISTANT_TURNS=12, VAL_MAX_USER_TURNS=12
# - total training steps: 100
# - actor lr: 1e-6

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-formal}
if (($# > 0)); then
  shift
fi

export RUN_PREFIX=${RUN_PREFIX:-qwen35_9b_official_tool_h200_rl_t12_100step}
export MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-12}
export MAX_USER_TURNS=${MAX_USER_TURNS:-12}
export VAL_MAX_ASSISTANT_TURNS=${VAL_MAX_ASSISTANT_TURNS:-12}
export VAL_MAX_USER_TURNS=${VAL_MAX_USER_TURNS:-12}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
export ACTOR_LR=${ACTOR_LR:-1e-6}

exec bash "$SCRIPT_DIR/run_qwen35_official_tool_h200_rl.sh" "$MODE" "$@"
