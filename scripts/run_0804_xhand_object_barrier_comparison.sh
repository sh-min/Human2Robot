#!/usr/bin/env bash
set -euo pipefail

# Whole-XHand visual camera-Z object barrier comparison for 08_04 MH videos.
#
# Usage:
#   bash scripts/run_0804_xhand_object_barrier_comparison.sh 1
#   FORCE=1 bash scripts/run_0804_xhand_object_barrier_comparison.sh 1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA="${DATA:-$ROOT/data/kitchen_dataset/26.08.04_stereo}"
ENVIRONMENT="${ENVIRONMENT:-inpaint-gpu}"
FORCE="${FORCE:-0}"

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

overlay_matches() {
    local directory="$1"
    conda run -n "$ENVIRONMENT" python -c '
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
manifest_path = root / "manifest.json"
required = [
    root / "robot_rgb.npy",
    root / "robot_depth.npy",
    root / "robot_mask.npy",
    root / "robot_finger_labels.npy",
    root / "robot_finger_mask.npy",
    root / "robot_hand_mask.npy",
]
if not manifest_path.is_file() or not all(p.is_file() and p.stat().st_size > 0 for p in required):
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text())
if manifest.get("hand_mask", {}).get("filename") != "robot_hand_mask.npy":
    raise SystemExit(1)
if manifest.get("hand_mask", {}).get("excludes_arm") is not True:
    raise SystemExit(1)
hand = np.load(root / "robot_hand_mask.npy", mmap_mode="r")
robot = np.load(root / "robot_mask.npy", mmap_mode="r")
finger = np.load(root / "robot_finger_mask.npy", mmap_mode="r")
if hand.shape != robot.shape or hand.shape != finger.shape or hand.dtype != np.bool_:
    raise SystemExit(1)
for t in range(len(hand)):
    h, r, f = np.asarray(hand[t]), np.asarray(robot[t]), np.asarray(finger[t])
    if np.any(h & ~r) or np.any(f & ~h):
        raise SystemExit(1)
' "$directory" >/dev/null 2>&1
}

barrier_matches() {
    local directory="$1"
    local thumb="$2"
    local finger="$3"
    local palm="$4"
    local temporal="$5"
    conda run -n "$ENVIRONMENT" python -c '
import json, math, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
required = [
    root / "report.json",
    root / "video_overlay_hand_barrier.mp4",
    root / "occluded_hand_mask.npy",
]
if not all(p.is_file() and p.stat().st_size > 0 for p in required):
    raise SystemExit(1)
report = json.loads(required[0].read_text())
config = report.get("config", {})
actual = (
    float(config.get("thumb_shell_m", -1)),
    float(config.get("finger_shell_m", -1)),
    float(config.get("palm_shell_m", -1)),
    int(config.get("temporal_max_gap_frames", -1)),
)
expected = tuple(map(float, sys.argv[2:5])) + (int(sys.argv[5]),)
if report.get("method") != "visual_camera_z_xhand_barrier":
    raise SystemExit(1)
if report.get("pose_state_modified") is not False:
    raise SystemExit(1)
if report.get("metric_collision_guarantee") is not False:
    raise SystemExit(1)
if any(not math.isclose(a, b, abs_tol=1e-9) for a, b in zip(actual[:3], expected[:3])):
    raise SystemExit(1)
if actual[3] != expected[3]:
    raise SystemExit(1)
if int(report.get("counts", {}).get("residual_violation_pixels", -1)) != 0:
    raise SystemExit(1)
mask = np.load(required[2], mmap_mode="r")
if mask.dtype != np.bool_ or int(mask.sum()) != int(report["counts"]["final_occluded_pixels"]):
    raise SystemExit(1)
' "$directory" "$thumb" "$finger" "$palm" "$temporal" >/dev/null 2>&1
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
    OVERLAY_INPUT="$PD/rb5_overlay_input_left.npz"
    JOINT_NAMES="$PD/rb5_overlay_input_left_jointnames.json"
    REFERENCE_OVERLAY="$PD/overlay_processor"
    BARRIER_RENDER_ROOT="$PD/xhand_object_barrier_render"
    BARRIER_OVERLAY="$BARRIER_RENDER_ROOT/overlay_processor"
    OBJECT_MASK="$PD/object_layer/object_mask_modal.npy"
    SURFACE_DEPTH="$PD/object_surface_3d/object_surface_depth.npy"
    FINGER_BEST="$PD/contact_occlusion_dual_haco_object3d_contact_aligned_force_temporal_raw"
    BASELINE_MASK="$FINGER_BEST/occluded_finger_mask.npy"
    ZERO="$PD/xhand_object_barrier_zero_raw"
    SHELL="$PD/xhand_object_barrier_shell_raw"
    SHELL_TEMPORAL="$PD/xhand_object_barrier_shell_temporal_raw"
    COMPARISON="$PD/contact_occlusion_compare_xhand_object_barrier_raw"

    REQUIRED=(
        "$SOURCE"
        "$OVERLAY_INPUT"
        "$JOINT_NAMES"
        "$REFERENCE_OVERLAY/manifest.json"
        "$REFERENCE_OVERLAY/robot_depth.npy"
        "$REFERENCE_OVERLAY/robot_mask.npy"
        "$REFERENCE_OVERLAY/robot_finger_labels.npy"
        "$REFERENCE_OVERLAY/robot_finger_mask.npy"
        "$OBJECT_MASK"
        "$SURFACE_DEPTH"
        "$FINGER_BEST/report.json"
        "$FINGER_BEST/video_overlay_contact.mp4"
        "$BASELINE_MASK"
    )
    for REQUIRED_PATH in "${REQUIRED[@]}"; do
        [ -e "$REQUIRED_PATH" ] || {
            echo "[$ID] missing prerequisite: $REQUIRED_PATH" >&2
            exit 1
        }
    done

    RENDER_INPUTS=(
        "$ROOT/src/inpainting/render_rb5_pyrender_overlay.py"
        "$OVERLAY_INPUT"
        "$JOINT_NAMES"
    )
    if [ "$FORCE" = "1" ] || \
       ! overlay_matches "$BARRIER_OVERLAY" || \
       ! fresh_file "$BARRIER_OVERLAY/manifest.json" "${RENDER_INPUTS[@]}"; then
        echo "[$ID] rendering whole-XHand semantic ownership"
        PYOPENGL_PLATFORM=egl \
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python "$ROOT/src/inpainting/render_rb5_pyrender_overlay.py" \
            --data "$OVERLAY_INPUT" \
            --jn "$JOINT_NAMES" \
            --out "$BARRIER_RENDER_ROOT" \
            --background "$SOURCE" \
            --render_scale 0.75 \
            --arm_mode full \
            --overwrite
    else
        echo "[$ID] whole-XHand semantic render is current"
    fi

    BARRIER_INPUTS=(
        "$ROOT/src/inpainting/composite_xhand_object_barrier.py"
        "$ROOT/src/inpainting/composite_rb5_contact_occlusion.py"
        "$SOURCE"
        "$BARRIER_OVERLAY/manifest.json"
        "$BARRIER_OVERLAY/robot_rgb.npy"
        "$BARRIER_OVERLAY/robot_depth.npy"
        "$BARRIER_OVERLAY/robot_mask.npy"
        "$BARRIER_OVERLAY/robot_hand_mask.npy"
        "$BARRIER_OVERLAY/robot_finger_labels.npy"
        "$OBJECT_MASK"
        "$SURFACE_DEPTH"
        "$BASELINE_MASK"
    )

    run_barrier() {
        local label="$1"
        local directory="$2"
        local thumb="$3"
        local finger="$4"
        local palm="$5"
        local temporal="$6"
        shift 6
        if [ "$FORCE" != "1" ] && \
           barrier_matches "$directory" "$thumb" "$finger" "$palm" "$temporal" && \
           fresh_file "$directory/report.json" "${BARRIER_INPUTS[@]}"; then
            echo "[$ID] $label is current"
            return
        fi
        echo "[$ID] rendering $label"
        PYTHONPATH="$ROOT/src/inpainting" \
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python "$ROOT/src/inpainting/composite_xhand_object_barrier.py" \
            --background "$SOURCE" \
            --raw_video "$SOURCE" \
            --overlay_dir "$BARRIER_OVERLAY" \
            --object_mask "$OBJECT_MASK" \
            --object_surface_depth "$SURFACE_DEPTH" \
            --baseline_mask "$BASELINE_MASK" \
            --thumb_shell_m "$thumb" \
            --finger_shell_m "$finger" \
            --palm_shell_m "$palm" \
            --temporal_max_gap_frames "$temporal" \
            "$@" \
            --out_dir "$directory"
    }

    run_barrier "whole-XHand zero-margin barrier" "$ZERO" 0 0 0 0
    run_barrier "whole-XHand thickness barrier" "$SHELL" \
        0.01958 0.01465 0.015 0
    run_barrier "whole-XHand thickness + temporal barrier" "$SHELL_TEMPORAL" \
        0.01958 0.01465 0.015 2 \
        --temporal_motion_px 6 \
        --temporal_front_slack_m 0.015

    COMPARE_INPUTS=(
        "$ROOT/src/inpainting/compare_xhand_object_barriers.py"
        "$ROOT/src/inpainting/make_video_comparison_grid.py"
        "$OBJECT_MASK"
        "$OVERLAY_INPUT"
        "$JOINT_NAMES"
        "$FINGER_BEST/report.json" "$FINGER_BEST/video_overlay_contact.mp4" "$BASELINE_MASK"
        "$ZERO/report.json" "$ZERO/video_overlay_hand_barrier.mp4" "$ZERO/occluded_hand_mask.npy"
        "$SHELL/report.json" "$SHELL/video_overlay_hand_barrier.mp4" "$SHELL/occluded_hand_mask.npy"
        "$SHELL_TEMPORAL/report.json" "$SHELL_TEMPORAL/video_overlay_hand_barrier.mp4" "$SHELL_TEMPORAL/occluded_hand_mask.npy"
    )
    FULL_VIDEO="$COMPARISON/video_compare_xhand_object_barrier_2x2.mp4"
    ROI_VIDEO="$COMPARISON/video_compare_xhand_object_barrier_roi_2x2.mp4"
    REPORT="$COMPARISON/comparison_report.json"
    if [ "$FORCE" != "1" ] && \
       fresh_file "$FULL_VIDEO" "${COMPARE_INPUTS[@]}" && \
       fresh_file "$ROI_VIDEO" "${COMPARE_INPUTS[@]}" && \
       fresh_file "$REPORT" "${COMPARE_INPUTS[@]}"; then
        echo "[$ID] XHand barrier comparison is current"
    else
        echo "[$ID] rendering full-frame and dynamic-ROI 2x2 comparisons"
        PYTHONPATH="$ROOT/src/inpainting" \
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python "$ROOT/src/inpainting/compare_xhand_object_barriers.py" \
            --finger_best_dir "$FINGER_BEST" \
            --whole_hand_zero_dir "$ZERO" \
            --whole_hand_shell_dir "$SHELL" \
            --whole_hand_shell_temporal_dir "$SHELL_TEMPORAL" \
            --object_mask "$OBJECT_MASK" \
            --reference_overlay_dir "$REFERENCE_OVERLAY" \
            --barrier_overlay_dir "$BARRIER_OVERLAY" \
            --overlay_input "$OVERLAY_INPUT" \
            --joint_names "$JOINT_NAMES" \
            --out_dir "$COMPARISON"
    fi
    echo "[$ID] full comparison: $FULL_VIDEO"
    echo "[$ID] ROI comparison:  $ROI_VIDEO"
done
