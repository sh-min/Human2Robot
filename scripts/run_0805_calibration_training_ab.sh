#!/usr/bin/env bash
set -euo pipefail

# Reproducible 08-05 classifier calibration A/B experiment.
#
# HaWoR/HaCo outputs must already exist in both roots. The frozen V-JEPA 2
# encoder is used only for feature extraction; the four small MLP folds are
# the only models trained here.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CKPT="${VJEPA_CKPT:-$ROOT/weights/vjepa2/vitl.pt}"
EXPECTED_CKPT_BYTES="${VJEPA_EXPECTED_BYTES:-5127726842}"
EXPECTED_CKPT_SHA256="${VJEPA_EXPECTED_SHA256:-5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012}"
CALIBRATED_DATA="${CALIBRATED_DATA:-$ROOT/data/kitchen_dataset/26.08.05_stereo_calibrated}"
APPROX_DATA="${APPROX_DATA:-$ROOT/data/kitchen_dataset/26.08.05_stereo_approx}"
ACTION_LABELS="${VJEPA_ACTION_LABELS:-Cup,Lock,Choco,Snack,Sweep,Trans}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
ROT6D_SEEDS="${ROT6D_SEEDS:-42 43 44 45 46}"

case "$FORCE_TRAIN" in
    0|1) ;;
    *) echo "FORCE_TRAIN must be 0 or 1" >&2; exit 1 ;;
esac

if [ ! -s "$CKPT" ]; then
    echo "Missing V-JEPA checkpoint: $CKPT" >&2
    exit 1
fi
ACTUAL_CKPT_BYTES=$(stat -c '%s' "$CKPT")
if [ "$ACTUAL_CKPT_BYTES" -ne "$EXPECTED_CKPT_BYTES" ]; then
    echo "Unexpected V-JEPA checkpoint size: $ACTUAL_CKPT_BYTES" >&2
    exit 1
fi
ACTUAL_CKPT_SHA256=$(sha256sum "$CKPT" | awk '{print $1}')
if [ "$ACTUAL_CKPT_SHA256" != "$EXPECTED_CKPT_SHA256" ]; then
    echo "Unexpected V-JEPA checkpoint SHA-256: $ACTUAL_CKPT_SHA256" >&2
    exit 1
fi

for VARIANT in calibrated approx; do
    if [ "$VARIANT" = "calibrated" ]; then
        DATA_ROOT="$CALIBRATED_DATA"
    else
        DATA_ROOT="$APPROX_DATA"
    fi

    echo "[$VARIANT] extracting frozen V-JEPA features and aligned MH MANO"
    DATA="$DATA_ROOT" \
    STAGES=vjepa \
    ALL=1 \
    VJEPA_CKPT="$CKPT" \
    VJEPA_ACTION_LABELS="$ACTION_LABELS" \
        bash "$ROOT/scripts/run_0804_stereo_hawor_haco.sh"
done

conda run -n vjepa2-312 --no-capture-output \
    python "$ROOT/scripts/validate_0805_feature_ab.py" \
    --approx-root "$APPROX_DATA" \
    --calibrated-root "$CALIBRATED_DATA" \
    --atol 0

# Preserve the original raw-axis-angle experiment exactly for comparison.
for VARIANT in calibrated approx; do
    if [ "$VARIANT" = "calibrated" ]; then
        DATA_ROOT="$CALIBRATED_DATA"
    else
        DATA_ROOT="$APPROX_DATA"
    fi
    for TRAIN_EP in 1 2; do
        if [ "$TRAIN_EP" = "1" ]; then
            VAL_EP=2
        else
            VAL_EP=1
        fi
        SUMMARY="$ROOT/output/skill_classifier/0805_${VARIANT}/fold_${TRAIN_EP}_to_${VAL_EP}/evaluation_summary.json"
        if [ "$FORCE_TRAIN" = "0" ] && [ -s "$SUMMARY" ]; then
            echo "[$VARIANT $TRAIN_EP->$VAL_EP] skip (evaluation summary exists)"
            continue
        fi
        echo "[$VARIANT $TRAIN_EP->$VAL_EP] training deterministic MLP fold"
        DATA="$DATA_ROOT" \
        VARIANT_NAME="$VARIANT" \
        TRAIN_EP="$TRAIN_EP" \
        VAL_EP="$VAL_EP" \
            bash "$ROOT/scripts/train_0805_skill_classifier.sh"
    done
done

python "$ROOT/scripts/summarize_0805_calibration_training.py"

# Axis-angle has a +/-pi branch discontinuity in episode 1. Run the primary
# robustness result with continuous rot6d MANO features and paired seeds.
read -r -a SEED_VALUES <<< "$ROT6D_SEEDS"
if [ "${#SEED_VALUES[@]}" -eq 0 ]; then
    echo "ROT6D_SEEDS must contain at least one integer" >&2
    exit 1
fi
for SEED_VALUE in "${SEED_VALUES[@]}"; do
    case "$SEED_VALUE" in
        ''|*[!0-9]*) echo "ROT6D_SEEDS contains a non-integer: $SEED_VALUE" >&2; exit 1 ;;
    esac
    for VARIANT in calibrated approx; do
        if [ "$VARIANT" = "calibrated" ]; then
            DATA_ROOT="$CALIBRATED_DATA"
        else
            DATA_ROOT="$APPROX_DATA"
        fi
        ROT6D_OUTPUT="$ROOT/output/skill_classifier/0805_rot6d_${VARIANT}/seed_${SEED_VALUE}"
        for TRAIN_EP in 1 2; do
            if [ "$TRAIN_EP" = "1" ]; then
                VAL_EP=2
            else
                VAL_EP=1
            fi
            SUMMARY="$ROT6D_OUTPUT/fold_${TRAIN_EP}_to_${VAL_EP}/evaluation_summary.json"
            if [ "$FORCE_TRAIN" = "0" ] && [ -s "$SUMMARY" ]; then
                echo "[rot6d $VARIANT seed=$SEED_VALUE $TRAIN_EP->$VAL_EP] skip"
                continue
            fi
            echo "[rot6d $VARIANT seed=$SEED_VALUE $TRAIN_EP->$VAL_EP] training"
            DATA="$DATA_ROOT" \
            VARIANT_NAME="rot6d_${VARIANT}" \
            HAND_REPRESENTATION=rot6d \
            SEED="$SEED_VALUE" \
            TRAIN_EP="$TRAIN_EP" \
            VAL_EP="$VAL_EP" \
            OUTPUT_DIR="$ROT6D_OUTPUT" \
                bash "$ROOT/scripts/train_0805_skill_classifier.sh"
        done
    done
done

python "$ROOT/scripts/summarize_0805_rot6d_calibration_training.py" \
    --seeds "${SEED_VALUES[@]}"
