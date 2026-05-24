#!/usr/bin/env bash
# Build official verl parquet data from the Mini-o3 JSON datasets.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
read -r -a PYTHON_CMD <<< "${PYTHON_CMD:-uv run --no-sync python}"

PROFILE=${PROFILE:-a100}
EVAL_VP_ONLY=${EVAL_VP_ONLY:-True}

case "$PROFILE" in
    a100)
        TRAIN_BASE_IMAGE_DIR=${TRAIN_BASE_IMAGE_DIR:-/mnt/localssd/Mini-o3/data/minio3_real_subset}
        VAL_BASE_IMAGE_DIR=${VAL_BASE_IMAGE_DIR:-$TRAIN_BASE_IMAGE_DIR}
        DATA_DIR=${DATA_DIR:-$PROJECT_DIR/data/minio3_real_train_a100}
        IMAGE_ROOT=${IMAGE_ROOT:-$TRAIN_BASE_IMAGE_DIR}
        ;;
    h200)
        TRAIN_BASE_IMAGE_DIR=${TRAIN_BASE_IMAGE_DIR:-/mnt/localssd/Mini-o3/data/minio3_full}
        VAL_BASE_IMAGE_DIR=${VAL_BASE_IMAGE_DIR:-/mnt/localssd/Mini-o3/data/minio3_real_subset}
        DATA_DIR=${DATA_DIR:-$PROJECT_DIR/data/minio3_real_train_h200}
        MIXED_IMAGE_ROOT=${MIXED_IMAGE_ROOT:-$PROJECT_DIR/data/minio3_mixed_image_root}
        mkdir -p "$MIXED_IMAGE_ROOT"
        ln -sfnT "$TRAIN_BASE_IMAGE_DIR/VisualProbe_train" "$MIXED_IMAGE_ROOT/VisualProbe_train"
        ln -sfnT "$TRAIN_BASE_IMAGE_DIR/DeepEyes_train_4K" "$MIXED_IMAGE_ROOT/DeepEyes_train_4K"
        ln -sfnT "$VAL_BASE_IMAGE_DIR/VisualProbe_Easy" "$MIXED_IMAGE_ROOT/VisualProbe_Easy"
        ln -sfnT "$VAL_BASE_IMAGE_DIR/VisualProbe_Medium" "$MIXED_IMAGE_ROOT/VisualProbe_Medium"
        ln -sfnT "$VAL_BASE_IMAGE_DIR/VisualProbe_Hard" "$MIXED_IMAGE_ROOT/VisualProbe_Hard"
        if [[ -d "$VAL_BASE_IMAGE_DIR/Vstar_Bench" ]]; then
            ln -sfnT "$VAL_BASE_IMAGE_DIR/Vstar_Bench" "$MIXED_IMAGE_ROOT/Vstar_Bench"
        fi
        IMAGE_ROOT=${IMAGE_ROOT:-$MIXED_IMAGE_ROOT}
        ;;
    *)
        echo "Unknown PROFILE=$PROFILE; expected a100 or h200" >&2
        exit 2
        ;;
esac

VISUALPROBE_TRAIN_DATA="$TRAIN_BASE_IMAGE_DIR/VisualProbe_train/train.json"
DEEPEYES_TRAIN_4K_DATA="$TRAIN_BASE_IMAGE_DIR/DeepEyes_train_4K/train.json"
VSTAR_BENCH_VAL_DATA="$VAL_BASE_IMAGE_DIR/Vstar_Bench/val.json"
VISUALPROBE_EASY_VAL_DATA="$VAL_BASE_IMAGE_DIR/VisualProbe_Easy/val.json"
VISUALPROBE_MEDIUM_VAL_DATA="$VAL_BASE_IMAGE_DIR/VisualProbe_Medium/val.json"
VISUALPROBE_HARD_VAL_DATA="$VAL_BASE_IMAGE_DIR/VisualProbe_Hard/val.json"

required=(
    "$VISUALPROBE_TRAIN_DATA"
    "$DEEPEYES_TRAIN_4K_DATA"
    "$VISUALPROBE_EASY_VAL_DATA"
    "$VISUALPROBE_MEDIUM_VAL_DATA"
    "$VISUALPROBE_HARD_VAL_DATA"
)
if [[ "$EVAL_VP_ONLY" != "True" && "$EVAL_VP_ONLY" != "true" && "$EVAL_VP_ONLY" != "1" ]]; then
    required+=("$VSTAR_BENCH_VAL_DATA")
fi

for path in "${required[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required data file: $path" >&2
        exit 2
    fi
done

val_jsons=("$VISUALPROBE_EASY_VAL_DATA" "$VISUALPROBE_MEDIUM_VAL_DATA" "$VISUALPROBE_HARD_VAL_DATA")
if [[ "$EVAL_VP_ONLY" != "True" && "$EVAL_VP_ONLY" != "true" && "$EVAL_VP_ONLY" != "1" ]]; then
    val_jsons=("$VSTAR_BENCH_VAL_DATA" "${val_jsons[@]}")
fi

mkdir -p "$DATA_DIR"
"${PYTHON_CMD[@]}" "$PROJECT_DIR/examples/minio3/preprocess_visualprobe.py" \
    --train-json "$VISUALPROBE_TRAIN_DATA" "$DEEPEYES_TRAIN_4K_DATA" \
    --val-json "${val_jsons[@]}" \
    --image-root "$IMAGE_ROOT" \
    --local-save-dir "$DATA_DIR" \
    --min-pixels "${MIN_PIXELS:-40000}" \
    --max-pixels "${MAX_PIXELS:-2000000}" \
    --tool-prompt-suite "${MINIO3_TOOL_PROMPT_SUITE:-qwen35_minio3_legacy_grounding}" \
    --official-tool-name "${MINIO3_OFFICIAL_TOOL_NAME:-tool_crop}" \
    --agent-name "${MINIO3_AGENT_LOOP:-${ROLLOUT_AGENT_LOOP:-}}"

echo "TRAIN_FILE=$DATA_DIR/train.parquet"
echo "VAL_FILE=$DATA_DIR/val.parquet"
