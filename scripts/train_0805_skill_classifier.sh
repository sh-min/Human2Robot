#!/usr/bin/env bash
set -euo pipefail

# Train one deterministic 08-05 fold. Run twice with TRAIN_EP/VAL_EP swapped
# for the complete two-fold calibration diagnostic.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA="${DATA:-$ROOT/data/cube_dataset/26.08.05_stereo_calibrated}"
TRAIN_ENV="${TRAIN_ENV:-vjepa2-312}"
CONFIG="${CONFIG:-$ROOT/src/skill_classifier/config/0805_vjepa_mano_mlp.yaml}"
TRAIN_EP="${TRAIN_EP:-1}"
VAL_EP="${VAL_EP:-2}"
VARIANT_NAME="${VARIANT_NAME:-calibrated}"
HAND_REPRESENTATION="${HAND_REPRESENTATION:-axis_angle}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/skill_classifier/0805_${VARIANT_NAME}}"
EXP_ID="${EXP_ID:-fold_${TRAIN_EP}_to_${VAL_EP}}"

cd "$ROOT"
for ID in "$TRAIN_EP" "$VAL_EP"; do
    FEATURE="$DATA/$ID/features.pt"
    if [ ! -s "$FEATURE" ]; then
        echo "Missing aligned feature bundle: $FEATURE" >&2
        exit 1
    fi
done

PYTHONPATH="$ROOT/src" MPLCONFIGDIR=/tmp/skill-classifier-mpl \
    conda run -n "$TRAIN_ENV" --no-capture-output \
    python -m skill_classifier.train \
        --config "$CONFIG" \
        --exp_id "$EXP_ID" \
        --set \
            train_data_root="$DATA" \
            val_data_root="$DATA" \
            train_recording_glob="$TRAIN_EP" \
            val_recording_glob="$VAL_EP" \
            hand_representation="$HAND_REPRESENTATION" \
            seed="$SEED" \
            output_dir="$OUTPUT_DIR" \
        "$@"
