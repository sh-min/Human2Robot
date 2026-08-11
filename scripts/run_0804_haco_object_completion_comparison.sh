#!/usr/bin/env bash
set -euo pipefail

# Compare the existing broad HaWoR-hand object completion with a dual-view
# HaCo-selected completion, then apply the same visual camera-Z XHand barrier.
# MH owns all projected contact geometry. SH contributes same-finger HaCo
# confidence only; no uncalibrated SH pixels or vertices enter the MH image.
#
# Usage:
#   bash scripts/run_0804_haco_object_completion_comparison.sh 1
#   FORCE=1 INPAINT_BATCH_SIZE=4 bash \
#       scripts/run_0804_haco_object_completion_comparison.sh 1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA="${DATA:-$ROOT/data/kitchen_dataset/26.08.04_stereo}"
ENVIRONMENT="${ENVIRONMENT:-inpaint-gpu}"
FORCE="${FORCE:-0}"
INPAINT_BATCH_SIZE="${INPAINT_BATCH_SIZE:-4}"

if [ "$#" -gt 0 ]; then
    EPISODES=("$@")
else
    EPISODES=("1")
fi

for ID in "${EPISODES[@]}"; do
    [[ "$ID" =~ ^[0-9]+$ ]] || {
        echo "Episode names must be numeric, got '$ID'" >&2
        exit 1
    }
done

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

fresh_tree() {
    local target="$1"
    local source_tree="$2"
    [ -s "$target" ] || return 1
    [ -d "$source_tree" ] || return 1
    if find "$source_tree" -type f -newer "$target" -print -quit |
        grep -q .; then
        return 1
    fi
}

broad_completion_matches() {
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
expected_true = (
    "trusted_modal_subset_input_modal",
    "trusted_modal_subset_amodal",
    "hand_contested_disjoint_trusted_modal",
    "hidden_disjoint_trusted_modal",
    "trusted_modal_rgb_has_priority",
    "hand_contested_input_modal_is_not_rgb_protected",
    "trajectory_arrays_unchanged",
)
if not all(invariants.get(name) is True for name in expected_true):
    raise SystemExit(1)
if int(invariants.get("preencode_trusted_modal_rgb_values_changed", -1)) != 0:
    raise SystemExit(1)
if int(invariants.get("preencode_values_changed_outside_hidden", -1)) != 0:
    raise SystemExit(1)
if int(report.get("counts", {}).get(
    "hidden_pixels_without_completed_depth", -1
)) != 0:
    raise SystemExit(1)
clean = np.load(required[3], mmap_mode="r", allow_pickle=False)
amodal = np.load(required[4], mmap_mode="r", allow_pickle=False)
depth = np.load(required[5], mmap_mode="r", allow_pickle=False)
if clean.dtype != np.bool_ or amodal.dtype != np.bool_ or not (
    clean.shape == amodal.shape == depth.shape
):
    raise SystemExit(1)
' "$directory" >/dev/null 2>&1
}

haco_completion_matches() {
    local directory="$1"
    local contact_dir="$2"
    local aux_contact_dir="$3"
    local hawor_npz="$4"
    local side="$5"
    local aux_side="$6"
    conda run -n "$ENVIRONMENT" python -c '
import json, sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
contact_dir = Path(sys.argv[2]).resolve()
aux_contact_dir = Path(sys.argv[3]).resolve()
hawor_npz = Path(sys.argv[4]).resolve()
side, aux_side = sys.argv[5:7]
required = [
    root / "report.json",
    root / "video_hand_removed_modal_only.mp4",
    root / "video_object_completed.mp4",
    root / "debug_object_completion.mp4",
    root / "object_mask_observed_clean.npy",
    root / "object_mask_amodal.npy",
    root / "object_surface_depth_completed.npy",
    root / "completion_evidence.npz",
    root / "haco_contact_support.npy",
    root / "haco_evidence.npz",
]
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    raise SystemExit(1)
report = json.loads(required[0].read_text())
if report.get("method") != (
    "dual_haco_selected_hand_cleaned_object_constrained_e2fgvi"
):
    raise SystemExit(1)
if report.get("generated_texture") is not True:
    raise SystemExit(1)
if report.get("physical_geometry_guarantee") is not False:
    raise SystemExit(1)

sources = report.get("sources", {})
expected_sources = {
    "contact_dir": contact_dir,
    "aux_contact_dir": aux_contact_dir,
    "hawor_npz": hawor_npz,
}
for name, expected in expected_sources.items():
    actual = sources.get(name)
    if actual is None or Path(actual).resolve() != expected:
        raise SystemExit(1)

config = report.get("config", {})
if int(config.get("haco_modal_hand_exclusion_px", -1)) != 4:
    raise SystemExit(1)
if int(config.get("haco_temporal_grace_frames", -1)) != 2:
    raise SystemExit(1)
if config.get("side") != side or config.get("aux_side") != aux_side:
    raise SystemExit(1)

invariants = report.get("invariants", {})
expected_true = (
    "trusted_modal_subset_input_modal",
    "trusted_modal_subset_amodal",
    "hand_contested_disjoint_trusted_modal",
    "hidden_disjoint_trusted_modal",
    "trusted_modal_rgb_has_priority",
    "hand_contested_input_modal_is_not_rgb_protected",
    "trajectory_arrays_unchanged",
    "haco_selected_hidden_subset_raw_hidden",
    "haco_does_not_measure_object_rgb_or_depth",
    "primary_view_owns_haco_projection",
    "auxiliary_haco_is_confidence_only",
)
if not all(invariants.get(name) is True for name in expected_true):
    raise SystemExit(1)
if int(invariants.get("preencode_trusted_modal_rgb_values_changed", -1)) != 0:
    raise SystemExit(1)
if int(invariants.get("preencode_values_changed_outside_hidden", -1)) != 0:
    raise SystemExit(1)
if int(report.get("counts", {}).get(
    "hidden_pixels_without_completed_depth", -1
)) != 0:
    raise SystemExit(1)

clean = np.load(required[4], mmap_mode="r", allow_pickle=False)
amodal = np.load(required[5], mmap_mode="r", allow_pickle=False)
depth = np.load(required[6], mmap_mode="r", allow_pickle=False)
support = np.load(required[8], mmap_mode="r", allow_pickle=False)
if clean.dtype != np.bool_ or amodal.dtype != np.bool_ or support.dtype != np.bool_ or not (
    clean.shape == amodal.shape == depth.shape == support.shape
):
    raise SystemExit(1)
' "$directory" "$contact_dir" "$aux_contact_dir" "$hawor_npz" \
        "$side" "$aux_side" >/dev/null 2>&1
}

barrier_matches() {
    local directory="$1"
    local completion="$2"
    local raw_video="$3"
    local overlay_dir="$4"
    local baseline_mask="$5"
    conda run -n "$ENVIRONMENT" python -c '
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
completion = Path(sys.argv[2]).resolve()
required = [root / "report.json", root / "video_overlay_hand_barrier.mp4"]
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    raise SystemExit(1)
report = json.loads(required[0].read_text())
if report.get("method") != "visual_camera_z_xhand_barrier":
    raise SystemExit(1)
if report.get("pose_state_modified") is not False:
    raise SystemExit(1)
if report.get("metric_collision_guarantee") is not False:
    raise SystemExit(1)
expected_sources = {
    "background": completion / "video_object_completed.mp4",
    "raw_video": Path(sys.argv[3]).resolve(),
    "overlay_dir": Path(sys.argv[4]).resolve(),
    "object_mask": completion / "object_mask_amodal.npy",
    "object_restore_mask": completion / "object_mask_observed_clean.npy",
    "object_surface_depth": completion / "object_surface_depth_completed.npy",
    "baseline_mask": Path(sys.argv[5]).resolve(),
}
sources = report.get("sources", {})
for name, expected in expected_sources.items():
    actual = sources.get(name)
    if actual is None or Path(actual).resolve() != expected:
        raise SystemExit(1)
config = report.get("config", {})
actual = (
    float(config.get("thumb_shell_m", -1)),
    float(config.get("finger_shell_m", -1)),
    float(config.get("palm_shell_m", -1)),
    int(config.get("temporal_max_gap_frames", -1)),
)
expected = (0.01958, 0.01465, 0.015, 2)
if any(abs(a - b) > 1e-9 for a, b in zip(actual[:3], expected[:3])):
    raise SystemExit(1)
if actual[3] != expected[3]:
    raise SystemExit(1)
if config.get("object_restore_mask_explicit") is not True:
    raise SystemExit(1)
counts = report.get("counts", {})
if int(counts.get("residual_violation_pixels", -1)) != 0:
    raise SystemExit(1)
invariants = report.get("invariants", {})
for name in (
    "baseline_subset_final",
    "final_occlusion_subset_of_xhand",
    "rb5_arm_excluded",
    "barrier_support_uses_object_mask_only",
    "raw_rgb_restore_uses_object_restore_mask_only",
    "valid_surface_barrier_residual_is_zero",
    "trajectory_arrays_unchanged",
):
    if invariants.get(name) is not True:
        raise SystemExit(1)
' "$directory" "$completion" "$raw_video" "$overlay_dir" \
        "$baseline_mask" >/dev/null 2>&1
}

comparison_matches() {
    local directory="$1"
    local broad_completion="$2"
    local haco_completion="$3"
    local barrier_dir="$4"
    conda run -n "$ENVIRONMENT" python -c '
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
required = [
    root / "comparison_report.json",
    root / "video_compare_haco_object_completion_2x2.mp4",
    root / "video_compare_haco_object_completion_roi_2x2.mp4",
]
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    raise SystemExit(1)
report = json.loads(required[0].read_text())
serialized = json.dumps(report)
for expected in (Path(value).resolve() for value in sys.argv[2:5]):
    if str(expected) not in serialized:
        raise SystemExit(1)
if report.get("pose_state_modified") is not False:
    raise SystemExit(1)
if report.get("physical_collision_solver") is not False:
    raise SystemExit(1)
' "$directory" "$broad_completion" "$haco_completion" "$barrier_dir" \
        >/dev/null 2>&1
}

conda run -n "$ENVIRONMENT" python -c 'import cv2, numpy, torch'
cd "$ROOT"

for ID in "${EPISODES[@]}"; do
    EP="$DATA/$ID"
    C1="$EP/camera_1"
    C2="$EP/camera_2"
    PD="$C2/visibility/processed/view/0"
    SOURCE="$C2/source.mov"
    STEREO_MANIFEST="$EP/stereo_manifest.json"
    LABELS="$EP/gt_labels.json"
    MODAL="$PD/object_layer/object_mask_modal.npy"
    ARM="$PD/segmentation_processor/masks_arm.npy"
    HAND="$C2/rgb_hawor.invalid_realsense_focal/tracks_0_695/model_masks.npy"
    SURFACE="$PD/object_surface_3d/object_surface_depth.npy"
    HAWOR="$C2/rgb_hawor/retarget_input.npz"
    CONTACT="$C2/contact"
    AUX_CONTACT="$C1/contact"
    SIDE="left"
    AUX_SIDE="left"
    OVERLAY="$PD/xhand_object_barrier_render/overlay_processor"
    FINGER_BEST="$PD/contact_occlusion_dual_haco_object3d_contact_aligned_force_temporal_raw"
    BASELINE_MASK="$FINGER_BEST/occluded_finger_mask.npy"
    BROAD_COMPLETION="$PD/object_completion_e2fgvi"
    HACO_COMPLETION="$PD/object_completion_dual_haco_e2fgvi"
    FINAL_BARRIER="$PD/xhand_object_barrier_object_completed_dual_haco_raw"
    COMPARISON="$PD/contact_occlusion_compare_object_completion_dual_haco_raw"

    REQUIRED=(
        "$SOURCE"
        "$STEREO_MANIFEST"
        "$LABELS"
        "$MODAL"
        "$ARM"
        "$HAND"
        "$SURFACE"
        "$HAWOR"
        "$CONTACT"
        "$AUX_CONTACT"
        "$OVERLAY/robot_rgb.npy"
        "$OVERLAY/robot_depth.npy"
        "$OVERLAY/robot_mask.npy"
        "$OVERLAY/robot_hand_mask.npy"
        "$OVERLAY/robot_finger_labels.npy"
        "$BASELINE_MASK"
        "$OVERLAY/manifest.json"
    )
    for REQUIRED_PATH in "${REQUIRED[@]}"; do
        [ -e "$REQUIRED_PATH" ] || {
            echo "[$ID] missing prerequisite: $REQUIRED_PATH" >&2
            exit 1
        }
    done
    AUX_FRAME_OFFSET=$(python -c '
import json, sys
print(json.load(open(sys.argv[1]))["temporal_alignment"]["camera1_frame_offset"])
' "$STEREO_MANIFEST")
    if [ "$AUX_FRAME_OFFSET" != "0" ]; then
        echo "[$ID] dual-HaCo completion requires zero SH frame offset; " \
            "manifest has $AUX_FRAME_OFFSET" >&2
        exit 1
    fi
    OVERLAY_SIDE=$(python -c '
import json, sys
print(json.load(open(sys.argv[1]))["side"])
' "$OVERLAY/manifest.json")
    if [ "$OVERLAY_SIDE" != "$SIDE" ]; then
        echo "[$ID] HaCo side $SIDE differs from overlay side $OVERLAY_SIDE" >&2
        exit 1
    fi
    if ! broad_completion_matches "$BROAD_COMPLETION"; then
        echo "[$ID] missing or invalid broad completion: $BROAD_COMPLETION" >&2
        echo "[$ID] run scripts/run_0804_object_completion_comparison.sh first" >&2
        exit 1
    fi

    COMPLETION_INPUTS=(
        "$ROOT/src/inpainting/inpaint_object_completion.py"
        "$ROOT/src/inpainting/inpaint_hands.py"
        "$ROOT/src/inpainting/composite_rb5_contact_occlusion.py"
        "$SOURCE" "$LABELS" "$MODAL" "$ARM" "$HAND" "$SURFACE" "$HAWOR"
    )
    if [ "$FORCE" != "1" ] && \
       haco_completion_matches "$HACO_COMPLETION" "$CONTACT" \
           "$AUX_CONTACT" "$HAWOR" "$SIDE" "$AUX_SIDE" && \
       fresh_file "$HACO_COMPLETION/report.json" "${COMPLETION_INPUTS[@]}" && \
       fresh_tree "$HACO_COMPLETION/report.json" "$CONTACT" && \
       fresh_tree "$HACO_COMPLETION/report.json" "$AUX_CONTACT"; then
        echo "[$ID] dual-view HaCo object completion is current"
    else
        echo "[$ID] running MH+SH HaCo-selected object completion (E2FGVI)"
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
            --contact_dir "$CONTACT" \
            --aux_contact_dir "$AUX_CONTACT" \
            --hawor_npz "$HAWOR" \
            --side "$SIDE" \
            --aux_side "$AUX_SIDE" \
            --haco_modal_hand_exclusion_px 4 \
            --haco_temporal_grace_frames 2 \
            --inpaint_height 360 \
            --out_dir "$HACO_COMPLETION"
        haco_completion_matches "$HACO_COMPLETION" "$CONTACT" \
            "$AUX_CONTACT" "$HAWOR" "$SIDE" "$AUX_SIDE"
    fi

    BARRIER_INPUTS=(
        "$ROOT/src/inpainting/composite_xhand_object_barrier.py"
        "$ROOT/src/inpainting/composite_rb5_contact_occlusion.py"
        "$HACO_COMPLETION/report.json"
        "$HACO_COMPLETION/video_object_completed.mp4"
        "$HACO_COMPLETION/object_mask_observed_clean.npy"
        "$HACO_COMPLETION/object_mask_amodal.npy"
        "$HACO_COMPLETION/object_surface_depth_completed.npy"
        "$SOURCE" "$BASELINE_MASK"
        "$OVERLAY/robot_rgb.npy"
        "$OVERLAY/robot_depth.npy"
        "$OVERLAY/robot_mask.npy"
        "$OVERLAY/robot_hand_mask.npy"
        "$OVERLAY/robot_finger_labels.npy"
    )
    if [ "$FORCE" != "1" ] && \
       barrier_matches "$FINAL_BARRIER" "$HACO_COMPLETION" "$SOURCE" \
           "$OVERLAY" "$BASELINE_MASK" && \
       fresh_file "$FINAL_BARRIER/report.json" "${BARRIER_INPUTS[@]}"; then
        echo "[$ID] dual-HaCo completed-object XHand barrier is current"
    else
        echo "[$ID] rendering dual-HaCo completed-object XHand barrier"
        PYTHONPATH="$ROOT/src/inpainting" \
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python "$ROOT/src/inpainting/composite_xhand_object_barrier.py" \
            --background "$HACO_COMPLETION/video_object_completed.mp4" \
            --raw_video "$SOURCE" \
            --overlay_dir "$OVERLAY" \
            --object_mask "$HACO_COMPLETION/object_mask_amodal.npy" \
            --object_restore_mask \
                "$HACO_COMPLETION/object_mask_observed_clean.npy" \
            --object_surface_depth \
                "$HACO_COMPLETION/object_surface_depth_completed.npy" \
            --baseline_mask "$BASELINE_MASK" \
            --thumb_shell_m 0.01958 \
            --finger_shell_m 0.01465 \
            --palm_shell_m 0.015 \
            --temporal_max_gap_frames 2 \
            --temporal_motion_px 6 \
            --temporal_front_slack_m 0.015 \
            --out_dir "$FINAL_BARRIER"
        barrier_matches "$FINAL_BARRIER" "$HACO_COMPLETION" "$SOURCE" \
            "$OVERLAY" "$BASELINE_MASK"
    fi

    COMPARE_INPUTS=(
        "$ROOT/src/inpainting/compare_haco_object_completion.py"
        "$ROOT/src/inpainting/compare_xhand_object_barriers.py"
        "$ROOT/src/inpainting/make_video_comparison_grid.py"
        "$SOURCE"
        "$BROAD_COMPLETION/report.json"
        "$BROAD_COMPLETION/video_object_completed.mp4"
        "$BROAD_COMPLETION/object_mask_amodal.npy"
        "$HACO_COMPLETION/report.json"
        "$HACO_COMPLETION/video_object_completed.mp4"
        "$HACO_COMPLETION/object_mask_amodal.npy"
        "$FINAL_BARRIER/report.json"
        "$FINAL_BARRIER/video_overlay_hand_barrier.mp4"
    )
    FULL_VIDEO="$COMPARISON/video_compare_haco_object_completion_2x2.mp4"
    ROI_VIDEO="$COMPARISON/video_compare_haco_object_completion_roi_2x2.mp4"
    REPORT="$COMPARISON/comparison_report.json"
    if [ "$FORCE" != "1" ] && \
       comparison_matches "$COMPARISON" "$BROAD_COMPLETION" \
           "$HACO_COMPLETION" "$FINAL_BARRIER" && \
       fresh_file "$FULL_VIDEO" "${COMPARE_INPUTS[@]}" && \
       fresh_file "$ROI_VIDEO" "${COMPARE_INPUTS[@]}" && \
       fresh_file "$REPORT" "${COMPARE_INPUTS[@]}"; then
        echo "[$ID] broad-vs-dual-HaCo comparison is current"
    else
        echo "[$ID] rendering broad-vs-dual-HaCo comparison"
        PYTHONPATH="$ROOT/src/inpainting" \
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python "$ROOT/src/inpainting/compare_haco_object_completion.py" \
            --source "$SOURCE" \
            --broad_completion_dir "$BROAD_COMPLETION" \
            --haco_completion_dir "$HACO_COMPLETION" \
            --barrier_dir "$FINAL_BARRIER" \
            --out_dir "$COMPARISON"
        comparison_matches "$COMPARISON" "$BROAD_COMPLETION" \
            "$HACO_COMPLETION" "$FINAL_BARRIER"
    fi

    echo "[$ID] full comparison: $FULL_VIDEO"
    echo "[$ID] ROI comparison:  $ROI_VIDEO"
    echo "[$ID] HaCo completion: $HACO_COMPLETION/video_object_completed.mp4"
    echo "[$ID] final overlay:   $FINAL_BARRIER/video_overlay_hand_barrier.mp4"
done
