#!/usr/bin/env bash
set -euo pipefail

# Produce the MH final-view composite using SH visibility plus max-fused
# SH/MH HaCo.  The robot overlay itself remains MH-only.
#
# Prerequisites per episode:
#   - correct HaWoR + HaCo for camera_1 (SH) and camera_2 (MH)
#   - dual-view masks from run_0804_stereo_visibility_assets.sh
#   - MH background and modal object mask from the same asset runner
#   - MH RB5/XHand arrays under the MH processed demo's overlay_processor/
#
# Usage:
#   bash scripts/run_0804_visibility_haco_composite.sh 1
#   ALL=1 bash scripts/run_0804_visibility_haco_composite.sh
#   FORCE=1 SIDE=left bash scripts/run_0804_visibility_haco_composite.sh 1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA="${DATA:-$ROOT/data/kitchen_dataset/26.08.04_stereo}"
ALL="${ALL:-0}"
FORCE="${FORCE:-0}"
INPAINT_ENV="${INPAINT_ENV:-inpaint}"
SIDE="${SIDE:-}"
CAMERA1_SIDE="${CAMERA1_SIDE:-}"
CAMERA2_SIDE="${CAMERA2_SIDE:-}"

cd "$ROOT"

if [ "$#" -gt 0 ]; then
    EPISODES=("$@")
elif [ "$ALL" = "1" ]; then
    mapfile -t CANDIDATES < <(
        find "$DATA" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -V
    )
    EPISODES=()
    for CANDIDATE in "${CANDIDATES[@]}"; do
        [[ "$CANDIDATE" =~ ^[0-9]+$ ]] && EPISODES+=("$CANDIDATE")
    done
else
    EPISODES=("1")
fi
if [ "${#EPISODES[@]}" -eq 0 ]; then
    echo "No numeric episodes found under $DATA" >&2
    exit 1
fi
for ID in "${EPISODES[@]}"; do
    [[ "$ID" =~ ^[0-9]+$ ]] || {
        echo "Episode names must be numeric, got '$ID'" >&2
        exit 1
    }
done

conda run -n "$INPAINT_ENV" python -c 'import cv2, numpy'

validate_composite_outputs() {
    local out_dir="$1"
    local expected_camera1_offset="$2"
    conda run -n "$INPAINT_ENV" python -c '
import json, numpy as np, sys
from pathlib import Path
root = Path(sys.argv[1])
expected_offset = int(sys.argv[2])
report = json.loads((root / "report.json").read_text())
modes = set(report.get("output_modes", []))
required = {"visibility_haco", "haco_only"}
if not required <= modes:
    raise SystemExit(f"report missing output modes: {sorted(required - modes)}")
actual_offset = report.get("temporal_alignment", {}).get("camera1_frame_offset")
if actual_offset != expected_offset:
    raise SystemExit(
        f"camera-1 temporal offset mismatch: {actual_offset} != {expected_offset}"
    )
mask = np.load(root / "occluded_finger_mask_visibility_haco.npy", mmap_mode="r")
if mask.ndim != 3 or len(mask) <= 0:
    raise SystemExit(f"invalid visibility-HaCo mask shape: {mask.shape}")
with np.load(root / "stereo_evidence.npz") as evidence:
    stored_offset = int(np.asarray(evidence["camera1_frame_offset"]).item())
    source_indices = np.asarray(evidence["camera1_source_frame_index"])
if stored_offset != expected_offset:
    raise SystemExit(
        f"evidence camera-1 offset mismatch: {stored_offset} != {expected_offset}"
    )
expected_indices = np.arange(len(mask), dtype=np.int64) + expected_offset
expected_indices[(expected_indices < 0) | (expected_indices >= len(mask))] = -1
if not np.array_equal(source_indices, expected_indices):
    raise SystemExit("evidence camera-1 source-frame lookup is inconsistent")
' "$out_dir" "$expected_camera1_offset"
}

for ID in "${EPISODES[@]}"; do
    EP="$DATA/$ID"
    C1="$EP/camera_1"
    C2="$EP/camera_2"
    C1_HAWOR="$C1/rgb_hawor/retarget_input.npz"
    C2_HAWOR="$C2/rgb_hawor/retarget_input.npz"
    C1_VISIBLE="$C1/visibility/processed/view/0/segmentation_processor/masks_arm.npy"
    MH_PD="$C2/visibility/processed/view/0"
    C2_VISIBLE="$MH_PD/segmentation_processor/masks_arm.npy"
    BACKGROUND="$MH_PD/inpaint_processor/video_human_inpaint.mkv"
    OBJECT_MASK="$MH_PD/object_layer/object_mask_modal.npy"
    OVERLAY_DIR="$MH_PD/overlay_processor"
    OUT_DIR="$MH_PD/stereo_occlusion"
    MANIFEST="$EP/stereo_manifest.json"
    EXPECTED=$(python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["num_frames"])' \
        "$EP/gt_labels.json")
    CAMERA1_FRAME_OFFSET=$(python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["temporal_alignment"]["camera1_frame_offset"])' \
        "$MANIFEST")

    REQUIRED=(
        "$MANIFEST"
        "$C1_HAWOR"
        "$C2_HAWOR"
        "$C1_VISIBLE"
        "$C2_VISIBLE"
        "$C1/contact"
        "$C2/contact"
        "$BACKGROUND"
        "$OBJECT_MASK"
        "$OVERLAY_DIR/manifest.json"
        "$OVERLAY_DIR/robot_rgb.npy"
        "$OVERLAY_DIR/robot_depth.npy"
        "$OVERLAY_DIR/robot_mask.npy"
        "$OVERLAY_DIR/robot_finger_labels.npy"
    )
    for PATH_REQUIRED in "${REQUIRED[@]}"; do
        if [ ! -e "$PATH_REQUIRED" ]; then
            echo "[$ID] missing prerequisite: $PATH_REQUIRED" >&2
            exit 1
        fi
    done
    if [ "$C1_HAWOR" -nt "$C1_VISIBLE" ] || \
       [ "$C2_HAWOR" -nt "$C2_VISIBLE" ] || \
       [ "$C2_HAWOR" -nt "$OBJECT_MASK" ] || \
       [ "$EP/gt_labels.json" -nt "$OBJECT_MASK" ] || \
       [ "$C2_VISIBLE" -nt "$BACKGROUND" ] || \
       [ "$OBJECT_MASK" -nt "$BACKGROUND" ]; then
        echo "[$ID] stale visibility/object/background asset; rerun the asset stages" >&2
        exit 1
    fi
    for CONTACT_DIR in "$C1/contact" "$C2/contact"; do
        CONTACT_COUNT=$(find "$CONTACT_DIR" -maxdepth 1 -type f \
            -name 'rgb_frame*.npz' | wc -l)
        if [ "$CONTACT_COUNT" -ne "$EXPECTED" ]; then
            echo "[$ID] incomplete dual-view HaCo: $CONTACT_DIR " \
                "($CONTACT_COUNT != $EXPECTED)" >&2
            exit 1
        fi
    done

    OUTPUTS=(
        "$OUT_DIR/video_overlay_visibility_haco.mp4"
        "$OUT_DIR/occluded_finger_mask_visibility_haco.npy"
        "$OUT_DIR/video_overlay_haco_only.mp4"
        "$OUT_DIR/stereo_evidence.npz"
        "$OUT_DIR/report.json"
    )
    COMPLETE=1
    for OUTPUT in "${OUTPUTS[@]}"; do
        if [ ! -s "$OUTPUT" ]; then
            COMPLETE=0
        fi
    done
    if [ "$FORCE" != "1" ] && [ "$COMPLETE" = "1" ]; then
        if validate_composite_outputs "$OUT_DIR" "$CAMERA1_FRAME_OFFSET"; then
            echo "[$ID] composite skip (validated complete): $OUT_DIR"
            continue
        fi
        COMPLETE=0
    fi
    if [ "$FORCE" != "1" ] && [ -e "$OUT_DIR" ]; then
        echo "[$ID] incomplete composite cache; rerun with FORCE=1: $OUT_DIR" >&2
        exit 1
    fi

    ARGS=(
        --camera1_rgb_dir "$C1/rgb"
        --camera2_rgb_dir "$C2/rgb"
        --camera1_hawor "$C1_HAWOR"
        --camera2_hawor "$C2_HAWOR"
        --camera1_visible_mask "$C1_VISIBLE"
        --camera2_visible_mask "$C2_VISIBLE"
        --camera1_contact_dir "$C1/contact"
        --contact_dir "$C2/contact"
        --background "$BACKGROUND"
        --overlay_dir "$OVERLAY_DIR"
        --object_mask "$OBJECT_MASK"
        --out_dir "$OUT_DIR"
        --camera1_frame_offset "$CAMERA1_FRAME_OFFSET"
        --fps 24
        --include_visibility_haco
        --include_haco_only
    )
    if [ -n "$SIDE" ]; then
        ARGS+=(--side "$SIDE")
    fi
    if [ -n "$CAMERA1_SIDE" ]; then
        ARGS+=(--camera1_side "$CAMERA1_SIDE")
    fi
    if [ -n "$CAMERA2_SIDE" ]; then
        ARGS+=(--camera2_side "$CAMERA2_SIDE")
    fi

    echo "[$ID] SH visibility + dual-HaCo -> MH composite " \
        "(camera1 offset=$CAMERA1_FRAME_OFFSET)"
    conda run -n "$INPAINT_ENV" --no-capture-output \
        python "$ROOT/src/inpainting/composite_rb5_stereo_occlusion.py" \
        "${ARGS[@]}"

    for OUTPUT in "${OUTPUTS[@]}"; do
        test -s "$OUTPUT"
    done
    validate_composite_outputs "$OUT_DIR" "$CAMERA1_FRAME_OFFSET"
    echo "[$ID] final: $OUT_DIR/video_overlay_visibility_haco.mp4"
done

echo
echo "Completed MH visibility-HaCo composites for: ${EPISODES[*]}"
