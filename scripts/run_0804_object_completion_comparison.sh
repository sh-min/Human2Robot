#!/usr/bin/env bash
set -euo pipefail

# Object-aware human-hand removal and XHand barrier comparison for 08_04 MH.
#
# Usage:
#   bash scripts/run_0804_object_completion_comparison.sh 1
#   FORCE=1 INPAINT_BATCH_SIZE=4 bash scripts/run_0804_object_completion_comparison.sh 1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA="${DATA:-$ROOT/data/cube_dataset/26.08.04_stereo}"
ENVIRONMENT="${ENVIRONMENT:-inpaint-gpu}"
FORCE="${FORCE:-0}"
INPAINT_BATCH_SIZE="${INPAINT_BATCH_SIZE:-4}"

if [ "$#" -gt 0 ]; then
    EPISODES=("$@")
else
    EPISODES=("1")
fi

fresh_file() {
    local target="$1"
    shift
    [ -s "$target" ] || return 1
    local source
    for source in "$@"; do
        [ -e "$source" ] || return 1
        [ "$source" -nt "$target" ] && return 1
    done
    return 0
}

completion_matches() {
    local directory="$1"
    conda run -n "$ENVIRONMENT" python -c '
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
required = [
    root / "report.json",
    root / "video_hand_removed_modal_only.mp4",
    root / "video_object_completed.mp4",
    root / "object_mask_observed_clean.npy",
    root / "object_mask_amodal.npy",
    root / "object_surface_depth_completed.npy",
]
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    raise SystemExit(1)
report = json.loads(required[0].read_text())
if report.get("method") != "hand_cleaned_modal_object_constrained_e2fgvi":
    raise SystemExit(1)
invariants = report.get("invariants", {})
if not all(invariants.get(name) is True for name in (
    "trusted_modal_subset_input_modal",
    "trusted_modal_subset_amodal",
    "hand_contested_disjoint_trusted_modal",
    "hidden_disjoint_trusted_modal",
    "trusted_modal_rgb_has_priority",
    "hand_contested_input_modal_is_not_rgb_protected",
    "trajectory_arrays_unchanged",
)):
    raise SystemExit(1)
if int(invariants.get("preencode_trusted_modal_rgb_values_changed", -1)) != 0:
    raise SystemExit(1)
if int(invariants.get("preencode_values_changed_outside_hidden", -1)) != 0:
    raise SystemExit(1)
if int(report.get("counts", {}).get("hidden_pixels_without_completed_depth", -1)) != 0:
    raise SystemExit(1)
config = report.get("config", {})
if any(int(config.get(name, -1)) != 16 for name in (
    "hand_dilate_px",
    "modal_hand_exclusion_px",
    "colour_donor_hand_dilate_px",
)):
    raise SystemExit(1)
clean = np.load(root / "object_mask_observed_clean.npy", mmap_mode="r")
mask = np.load(root / "object_mask_amodal.npy", mmap_mode="r")
depth = np.load(root / "object_surface_depth_completed.npy", mmap_mode="r")
if clean.dtype != np.bool_ or mask.dtype != np.bool_ or not (
    clean.shape == mask.shape == depth.shape
):
    raise SystemExit(1)
' "$directory" >/dev/null 2>&1
}

barrier_matches() {
    local directory="$1"
    conda run -n "$ENVIRONMENT" python -c '
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
required = [root / "report.json", root / "video_overlay_hand_barrier.mp4"]
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    raise SystemExit(1)
report = json.loads(required[0].read_text())
config = report.get("config", {})
counts = report.get("counts", {})
if report.get("method") != "visual_camera_z_xhand_barrier":
    raise SystemExit(1)
if config.get("object_restore_mask_explicit") is not True:
    raise SystemExit(1)
actual = (
    float(config.get("thumb_shell_m", -1)),
    float(config.get("finger_shell_m", -1)),
    float(config.get("palm_shell_m", -1)),
    int(config.get("temporal_max_gap_frames", -1)),
)
expected = (0.01958, 0.01465, 0.015, 2)
if any(abs(a-b) > 1e-9 for a,b in zip(actual[:3], expected[:3])) or actual[3] != expected[3]:
    raise SystemExit(1)
if int(counts.get("residual_violation_pixels", -1)) != 0:
    raise SystemExit(1)
' "$directory" >/dev/null 2>&1
}

for ID in "${EPISODES[@]}"; do
    [[ "$ID" =~ ^[0-9]+$ ]] || {
        echo "Episode names must be numeric, got '$ID'" >&2
        exit 1
    }
    EP="$DATA/$ID"
    C2="$EP/camera_2"
    PD="$C2/visibility/processed/view/0"
    SOURCE="$C2/source.mov"
    LABELS="$EP/gt_labels.json"
    MODAL="$PD/object_layer/object_mask_modal.npy"
    ARM="$PD/segmentation_processor/masks_arm.npy"
    HAND="$C2/rgb_hawor.invalid_realsense_focal/tracks_0_695/model_masks.npy"
    SURFACE="$PD/object_surface_3d/object_surface_depth.npy"
    OVERLAY="$PD/xhand_object_barrier_render/overlay_processor"
    FINGER_BEST="$PD/contact_occlusion_dual_haco_object3d_contact_aligned_force_temporal_raw"
    BASELINE_MASK="$FINGER_BEST/occluded_finger_mask.npy"
    COMPLETION="$PD/object_completion_e2fgvi"
    FINAL_BARRIER="$PD/xhand_object_barrier_object_completed_raw"
    COMPARISON="$PD/contact_occlusion_compare_object_completion_raw"

    REQUIRED=(
        "$SOURCE"
        "$LABELS"
        "$MODAL"
        "$ARM"
        "$HAND"
        "$SURFACE"
        "$OVERLAY/robot_rgb.npy"
        "$OVERLAY/robot_depth.npy"
        "$OVERLAY/robot_mask.npy"
        "$OVERLAY/robot_hand_mask.npy"
        "$OVERLAY/robot_finger_labels.npy"
        "$BASELINE_MASK"
    )
    for REQUIRED_PATH in "${REQUIRED[@]}"; do
        [ -e "$REQUIRED_PATH" ] || {
            echo "[$ID] missing prerequisite: $REQUIRED_PATH" >&2
            exit 1
        }
    done

    COMPLETION_INPUTS=(
        "$ROOT/src/inpainting/inpaint_object_completion.py"
        "$ROOT/src/inpainting/inpaint_hands.py"
        "$SOURCE" "$LABELS" "$MODAL" "$ARM" "$HAND" "$SURFACE"
    )
    if [ "$FORCE" != "1" ] && \
       completion_matches "$COMPLETION" && \
       fresh_file "$COMPLETION/report.json" "${COMPLETION_INPUTS[@]}"; then
        echo "[$ID] object-aware human removal is current"
    else
        echo "[$ID] running object-aware human removal (E2FGVI)"
        PYTHONPATH="$ROOT/src/inpainting" \
        INPAINT_BATCH_SIZE="$INPAINT_BATCH_SIZE" \
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python "$ROOT/src/inpainting/inpaint_object_completion.py" \
            --source "$SOURCE" \
            --modal_mask "$MODAL" \
            --arm_mask "$ARM" \
            --hand_support "$HAND" \
            --surface_depth "$SURFACE" \
            --labels_json "$LABELS" \
            --inpaint_height 360 \
            --out_dir "$COMPLETION"
    fi

    BARRIER_INPUTS=(
        "$ROOT/src/inpainting/composite_xhand_object_barrier.py"
        "$ROOT/src/inpainting/composite_rb5_contact_occlusion.py"
        "$COMPLETION/report.json"
        "$COMPLETION/video_object_completed.mp4"
        "$COMPLETION/object_mask_observed_clean.npy"
        "$COMPLETION/object_mask_amodal.npy"
        "$COMPLETION/object_surface_depth_completed.npy"
        "$SOURCE" "$MODAL" "$BASELINE_MASK"
        "$OVERLAY/robot_rgb.npy"
        "$OVERLAY/robot_depth.npy"
        "$OVERLAY/robot_mask.npy"
        "$OVERLAY/robot_hand_mask.npy"
        "$OVERLAY/robot_finger_labels.npy"
    )
    if [ "$FORCE" != "1" ] && \
       barrier_matches "$FINAL_BARRIER" && \
       fresh_file "$FINAL_BARRIER/report.json" "${BARRIER_INPUTS[@]}"; then
        echo "[$ID] completed-object XHand barrier is current"
    else
        echo "[$ID] rendering completed-object XHand barrier"
        PYTHONPATH="$ROOT/src/inpainting" \
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python "$ROOT/src/inpainting/composite_xhand_object_barrier.py" \
            --background "$COMPLETION/video_object_completed.mp4" \
            --raw_video "$SOURCE" \
            --overlay_dir "$OVERLAY" \
            --object_mask "$COMPLETION/object_mask_amodal.npy" \
            --object_restore_mask "$COMPLETION/object_mask_observed_clean.npy" \
            --object_surface_depth "$COMPLETION/object_surface_depth_completed.npy" \
            --baseline_mask "$BASELINE_MASK" \
            --thumb_shell_m 0.01958 \
            --finger_shell_m 0.01465 \
            --palm_shell_m 0.015 \
            --temporal_max_gap_frames 2 \
            --temporal_motion_px 6 \
            --temporal_front_slack_m 0.015 \
            --out_dir "$FINAL_BARRIER"
    fi

    COMPARE_INPUTS=(
        "$ROOT/src/inpainting/compare_object_completion.py"
        "$ROOT/src/inpainting/compare_xhand_object_barriers.py"
        "$ROOT/src/inpainting/make_video_comparison_grid.py"
        "$SOURCE"
        "$COMPLETION/report.json"
        "$COMPLETION/video_hand_removed_modal_only.mp4"
        "$COMPLETION/video_object_completed.mp4"
        "$COMPLETION/object_mask_observed_clean.npy"
        "$COMPLETION/object_mask_amodal.npy"
        "$FINAL_BARRIER/report.json"
        "$FINAL_BARRIER/video_overlay_hand_barrier.mp4"
    )
    FULL_VIDEO="$COMPARISON/video_compare_object_completion_2x2.mp4"
    ROI_VIDEO="$COMPARISON/video_compare_object_completion_roi_2x2.mp4"
    REPORT="$COMPARISON/comparison_report.json"
    if [ "$FORCE" != "1" ] && \
       fresh_file "$FULL_VIDEO" "${COMPARE_INPUTS[@]}" && \
       fresh_file "$ROI_VIDEO" "${COMPARE_INPUTS[@]}" && \
       fresh_file "$REPORT" "${COMPARE_INPUTS[@]}"; then
        echo "[$ID] object-completion comparison is current"
    else
        echo "[$ID] rendering full-frame and dynamic-ROI completion comparisons"
        PYTHONPATH="$ROOT/src/inpainting" \
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python "$ROOT/src/inpainting/compare_object_completion.py" \
            --source "$SOURCE" \
            --completion_dir "$COMPLETION" \
            --barrier_dir "$FINAL_BARRIER" \
            --out_dir "$COMPARISON"
    fi
    echo "[$ID] full comparison: $FULL_VIDEO"
    echo "[$ID] ROI comparison:  $ROI_VIDEO"
done
