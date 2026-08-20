#!/usr/bin/env bash
set -euo pipefail

# Build a fitted nominal object volume and compare three mesh barriers against
# the existing dual-HaCo completed-object 2.5-D barrier.
#
# Usage:
#   bash scripts/run_0804_object_mesh_volume_comparison.sh 1
#   FORCE=1 bash scripts/run_0804_object_mesh_volume_comparison.sh 1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA="${DATA:-$ROOT/data/cube_dataset/26.08.04_stereo}"
ENVIRONMENT="${ENVIRONMENT:-inpaint-gpu}"
FORCE="${FORCE:-0}"

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

mesh_matches() {
    local directory="$1"
    PYTHONPATH="$ROOT/src/inpainting" \
    conda run -n "$ENVIRONMENT" python -c '
import sys
from pathlib import Path
from compare_object_mesh_volume import validate_mesh_volume_arrays

root = Path(sys.argv[1])
required = [
    root / "object_pose_cam.npy",
    root / "pose_confidence.npy",
    root / "fit_evidence.npz",
    root / "debug_object_mesh_volume.mp4",
]
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    raise SystemExit(1)
result = validate_mesh_volume_arrays(root)
if result["pose_valid_frames"] <= 0:
    raise SystemExit(1)
report = result["report"]
expected_sources = {
    key: Path(value).resolve()
    for key, value in zip(
        (
            "mapping",
            "labels_json",
            "amodal_mask",
            "completed_front_depth",
            "wrist_npz",
            "debug_video",
        ),
        sys.argv[2:8],
    )
}
sources = report.get("sources", {})
for key, expected in expected_sources.items():
    actual = sources.get(key)
    if actual is None or Path(actual).resolve() != expected:
        raise SystemExit(1)
if abs(float(report.get("camera", {}).get("render_scale", -1)) - 0.5) > 1e-12:
    raise SystemExit(1)
invariants = report.get("invariants", {})
if not all(
    invariants.get(key) is True
    for key in (
        "canonical_meshes_watertight",
        "invalid_pose_frames_have_empty_geometry",
        "valid_mesh_pixels_have_ordered_front_back",
        "mesh_mask_equals_positive_front_and_back",
        "robot_trajectory_arrays_unchanged",
    )
):
    raise SystemExit(1)
canonical = report.get("canonical_meshes", {})
if not isinstance(canonical, dict) or not canonical:
    raise SystemExit(1)
for item in canonical.values():
    mesh_path = root / item.get("mesh_file", "")
    if not mesh_path.is_file() or mesh_path.stat().st_size <= 0:
        raise SystemExit(1)
' "$directory" "$MAPPING" "$LABELS" "$OBJECT_SUPPORT" "$FRONT_DEPTH" \
        "$WRIST" "$SOURCE" >/dev/null 2>&1
}

composite_matches() {
    local directory="$1"
    local mode="$2"
    local thumb="$3"
    local finger="$4"
    local palm="$5"
    local temporal="$6"
    conda run -n "$ENVIRONMENT" python -c '
import json, math, sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
mode = sys.argv[2]
expected = tuple(map(float, sys.argv[3:6])) + (int(sys.argv[6]),)
expected_sources = {
    key: Path(value).resolve()
    for key, value in zip(
        (
            "background",
            "raw_video",
            "overlay_dir",
            "mesh_dir",
            "object_support_mask",
            "object_restore_mask",
            "baseline_mask",
        ),
        sys.argv[7:14],
    )
}
required = [
    root / "report.json",
    root / "video_overlay_mesh_volume.mp4",
    root / "video_robot_only_mesh_volume.mp4",
    root / "debug_mesh_volume.mp4",
    root / "occluded_hand_mask.npy",
    root / "mesh_volume_classification.npy",
    root / "mesh_volume_evidence.npz",
]
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    raise SystemExit(1)
report = json.loads(required[0].read_text())
if report.get("method") != "visual_xhand_mesh_volume_barrier":
    raise SystemExit(1)
config = report.get("config", {})
if report.get("mode", config.get("mode")) != mode:
    raise SystemExit(1)
actual = (
    float(config.get("thumb_shell_m", -1)),
    float(config.get("finger_shell_m", -1)),
    float(config.get("palm_shell_m", -1)),
    int(config.get("temporal_max_gap_frames", -1)),
)
if any(not math.isclose(a, b, abs_tol=1e-9) for a, b in zip(actual[:3], expected[:3])):
    raise SystemExit(1)
if actual[3] != expected[3]:
    raise SystemExit(1)
if report.get("pose_state_modified") is not False:
    raise SystemExit(1)
if report.get("metric_collision_guarantee") is not False:
    raise SystemExit(1)
sources = report.get("sources", {})
for key, expected_path in expected_sources.items():
    actual = sources.get(key)
    if actual is None or Path(actual).resolve() != expected_path:
        raise SystemExit(1)
counts = report.get("counts", {})
if int(counts.get("residual_violation_pixels", -1)) != 0:
    raise SystemExit(1)
if report.get("invariants", {}).get(
    "valid_volume_barrier_residual_is_zero"
) is not True:
    raise SystemExit(1)
mask = np.load(required[4], mmap_mode="r", allow_pickle=False)
if mask.dtype != np.bool_ or mask.ndim != 3:
    raise SystemExit(1)
if int(mask.sum()) != int(counts.get("final_occluded_pixels", -1)):
    raise SystemExit(1)
' "$directory" "$mode" "$thumb" "$finger" "$palm" "$temporal" \
        "$BACKGROUND" "$SOURCE" "$OVERLAY" "$MESH" "$OBJECT_SUPPORT" \
        "$OBJECT_RESTORE" "$BASELINE_MASK" \
        >/dev/null 2>&1
}

comparison_matches() {
    local directory="$1"
    local mesh_dir="$2"
    shift 2
    PYTHONPATH="$ROOT/src/inpainting" \
    conda run -n "$ENVIRONMENT" python -c '
import json, sys
from pathlib import Path
from make_video_comparison_grid import probe_video

root = Path(sys.argv[1])
mesh_dir = Path(sys.argv[2]).resolve()
source_dirs = [Path(value).resolve() for value in sys.argv[3:7]]
required = [
    root / "comparison_report.json",
    root / "video_compare_object_mesh_volume_2x2.mp4",
    root / "video_compare_object_mesh_volume_roi_2x2.mp4",
]
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    raise SystemExit(1)
report = json.loads(required[0].read_text())
if report.get("comparison") != (
    "HaCo 2.5-D versus fitted nominal object mesh volume"
):
    raise SystemExit(1)
if report.get("pose_state_modified") is not False:
    raise SystemExit(1)
if report.get("metric_collision_guarantee") is not False:
    raise SystemExit(1)
if Path(report.get("mesh_builder", {}).get("directory", "")).resolve() != mesh_dir:
    raise SystemExit(1)
reported_sources = report.get("sources", {})
expected_modes = (
    "haco_2p5d",
    "mesh_front",
    "mesh_volume_shell",
    "mesh_volume_shell_temporal",
)
for mode, expected in zip(expected_modes, source_dirs):
    if Path(reported_sources.get(mode, {}).get("directory", "")).resolve() != expected:
        raise SystemExit(1)
for video in required[1:]:
    metadata = probe_video(video)
    if metadata.codec_name != "h264" or metadata.pixel_format != "yuv420p":
        raise SystemExit(1)
' "$directory" "$mesh_dir" "$@" >/dev/null 2>&1
}

conda run -n "$ENVIRONMENT" python -c 'import cv2, numpy, trimesh'
cd "$ROOT"

for ID in "${EPISODES[@]}"; do
    EP="$DATA/$ID"
    C2="$EP/camera_2"
    PD="$C2/visibility/processed/view/0"
    SOURCE="$C2/source.mov"
    LABELS="$EP/gt_labels.json"
    MAPPING="$ROOT/configs/objects/0804_mesh_volume.json"
    COMPLETION="$PD/object_completion_dual_haco_e2fgvi"
    BACKGROUND="$COMPLETION/video_object_completed.mp4"
    OBJECT_SUPPORT="$COMPLETION/object_mask_amodal.npy"
    OBJECT_RESTORE="$COMPLETION/object_mask_observed_clean.npy"
    FRONT_DEPTH="$COMPLETION/object_surface_depth_completed.npy"
    WRIST="$PD/rb5_overlay_input_left.npz"
    JOINT_NAMES="$PD/rb5_overlay_input_left_jointnames.json"
    OVERLAY="$PD/xhand_object_barrier_render/overlay_processor"
    FINGER_BEST="$PD/contact_occlusion_dual_haco_object3d_contact_aligned_force_temporal_raw"
    BASELINE_MASK="$FINGER_BEST/occluded_finger_mask.npy"
    HACO_2P5D="$PD/xhand_object_barrier_object_completed_dual_haco_raw"
    MESH="$PD/object_mesh_volume_nominal"
    MESH_FRONT="$PD/xhand_object_mesh_front_raw"
    MESH_VOLUME_SHELL="$PD/xhand_object_mesh_volume_shell_raw"
    MESH_VOLUME_TEMPORAL="$PD/xhand_object_mesh_volume_shell_temporal_raw"
    COMPARISON="$PD/contact_occlusion_compare_object_mesh_volume_raw"

    REQUIRED=(
        "$SOURCE"
        "$LABELS"
        "$MAPPING"
        "$BACKGROUND"
        "$OBJECT_SUPPORT"
        "$OBJECT_RESTORE"
        "$FRONT_DEPTH"
        "$WRIST"
        "$JOINT_NAMES"
        "$OVERLAY/manifest.json"
        "$OVERLAY/robot_rgb.npy"
        "$OVERLAY/robot_depth.npy"
        "$OVERLAY/robot_mask.npy"
        "$OVERLAY/robot_hand_mask.npy"
        "$OVERLAY/robot_finger_labels.npy"
        "$BASELINE_MASK"
        "$HACO_2P5D/report.json"
        "$HACO_2P5D/video_overlay_hand_barrier.mp4"
        "$HACO_2P5D/occluded_hand_mask.npy"
    )
    for REQUIRED_PATH in "${REQUIRED[@]}"; do
        [ -e "$REQUIRED_PATH" ] || {
            echo "[$ID] missing prerequisite: $REQUIRED_PATH" >&2
            echo "[$ID] run scripts/run_0804_haco_object_completion_comparison.sh first" >&2
            exit 1
        }
    done

    MESH_INPUTS=(
        "$ROOT/src/inpainting/build_object_mesh_volume.py"
        "$MAPPING"
        "$LABELS"
        "$OBJECT_SUPPORT"
        "$FRONT_DEPTH"
        "$WRIST"
        "$SOURCE"
    )
    if [ "$FORCE" != "1" ] && \
       mesh_matches "$MESH" && \
       fresh_file "$MESH/report.json" "${MESH_INPUTS[@]}" && \
       fresh_tree "$MESH/report.json" "$ROOT/configs/objects" && \
       fresh_tree "$MESH/report.json" "$ROOT/src/sim/mujoco_sim/assets"; then
        echo "[$ID] fitted nominal mesh volume is current"
    else
        echo "[$ID] fitting nominal object meshes and rendering front/back depth"
        PYTHONPATH="$ROOT/src/inpainting" \
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python "$ROOT/src/inpainting/build_object_mesh_volume.py" \
            --mapping "$MAPPING" \
            --labels_json "$LABELS" \
            --amodal_mask "$OBJECT_SUPPORT" \
            --front_depth "$FRONT_DEPTH" \
            --wrist_npz "$WRIST" \
            --video "$SOURCE" \
            --render_scale 0.5 \
            --out_dir "$MESH"
        mesh_matches "$MESH"
    fi

    COMPOSITOR_INPUTS=(
        "$ROOT/src/inpainting/composite_xhand_mesh_volume.py"
        "$ROOT/src/inpainting/composite_xhand_object_barrier.py"
        "$BACKGROUND"
        "$SOURCE"
        "$OVERLAY/manifest.json"
        "$OVERLAY/robot_rgb.npy"
        "$OVERLAY/robot_depth.npy"
        "$OVERLAY/robot_mask.npy"
        "$OVERLAY/robot_hand_mask.npy"
        "$OVERLAY/robot_finger_labels.npy"
        "$MESH/report.json"
        "$MESH/object_mesh_front_depth.npy"
        "$MESH/object_mesh_back_depth.npy"
        "$MESH/object_mesh_mask.npy"
        "$MESH/pose_valid.npy"
        "$OBJECT_SUPPORT"
        "$OBJECT_RESTORE"
        "$BASELINE_MASK"
    )

    run_compositor() {
        local label="$1"
        local directory="$2"
        local mode="$3"
        local thumb="$4"
        local finger="$5"
        local palm="$6"
        local temporal="$7"
        shift 7
        if [ "$FORCE" != "1" ] && \
           composite_matches "$directory" "$mode" "$thumb" "$finger" \
               "$palm" "$temporal" && \
           fresh_file "$directory/report.json" "${COMPOSITOR_INPUTS[@]}"; then
            echo "[$ID] $label is current"
            return
        fi
        echo "[$ID] rendering $label"
        PYTHONPATH="$ROOT/src/inpainting" \
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python "$ROOT/src/inpainting/composite_xhand_mesh_volume.py" \
            --background "$BACKGROUND" \
            --raw_video "$SOURCE" \
            --overlay_dir "$OVERLAY" \
            --mesh_dir "$MESH" \
            --object_support_mask "$OBJECT_SUPPORT" \
            --object_restore_mask "$OBJECT_RESTORE" \
            --baseline_mask "$BASELINE_MASK" \
            --mode "$mode" \
            --thumb_shell_m "$thumb" \
            --finger_shell_m "$finger" \
            --palm_shell_m "$palm" \
            --temporal_max_gap_frames "$temporal" \
            "$@" \
            --out_dir "$directory"
        composite_matches "$directory" "$mode" "$thumb" "$finger" \
            "$palm" "$temporal"
    }

    run_compositor "nominal-mesh front-only barrier" \
        "$MESH_FRONT" front 0 0 0 0
    run_compositor "nominal-mesh volume + XHand shell barrier" \
        "$MESH_VOLUME_SHELL" volume 0.01958 0.01465 0.015 0
    run_compositor "nominal-mesh volume + shell + temporal barrier" \
        "$MESH_VOLUME_TEMPORAL" volume 0.01958 0.01465 0.015 2 \
        --temporal_motion_px 6 \
        --temporal_front_slack_m 0.015

    COMPARE_INPUTS=(
        "$ROOT/src/inpainting/compare_object_mesh_volume.py"
        "$ROOT/src/inpainting/compare_xhand_object_barriers.py"
        "$ROOT/src/inpainting/make_video_comparison_grid.py"
        "$MAPPING"
        "$OBJECT_SUPPORT"
        "$WRIST"
        "$JOINT_NAMES"
        "$MESH/report.json"
        "$MESH/object_mesh_front_depth.npy"
        "$MESH/object_mesh_back_depth.npy"
        "$MESH/object_mesh_mask.npy"
        "$MESH/pose_valid.npy"
        "$HACO_2P5D/report.json"
        "$HACO_2P5D/video_overlay_hand_barrier.mp4"
        "$HACO_2P5D/occluded_hand_mask.npy"
        "$MESH_FRONT/report.json"
        "$MESH_FRONT/video_overlay_mesh_volume.mp4"
        "$MESH_FRONT/occluded_hand_mask.npy"
        "$MESH_VOLUME_SHELL/report.json"
        "$MESH_VOLUME_SHELL/video_overlay_mesh_volume.mp4"
        "$MESH_VOLUME_SHELL/occluded_hand_mask.npy"
        "$MESH_VOLUME_TEMPORAL/report.json"
        "$MESH_VOLUME_TEMPORAL/video_overlay_mesh_volume.mp4"
        "$MESH_VOLUME_TEMPORAL/occluded_hand_mask.npy"
    )
    FULL_VIDEO="$COMPARISON/video_compare_object_mesh_volume_2x2.mp4"
    ROI_VIDEO="$COMPARISON/video_compare_object_mesh_volume_roi_2x2.mp4"
    REPORT="$COMPARISON/comparison_report.json"
    if [ "$FORCE" != "1" ] && \
       comparison_matches "$COMPARISON" "$MESH" "$HACO_2P5D" \
           "$MESH_FRONT" "$MESH_VOLUME_SHELL" "$MESH_VOLUME_TEMPORAL" && \
       fresh_file "$FULL_VIDEO" "${COMPARE_INPUTS[@]}" && \
       fresh_file "$ROI_VIDEO" "${COMPARE_INPUTS[@]}" && \
       fresh_file "$REPORT" "${COMPARE_INPUTS[@]}"; then
        echo "[$ID] object mesh-volume comparison is current"
    else
        echo "[$ID] rendering full-frame and dynamic-ROI mesh comparisons"
        PYTHONPATH="$ROOT/src/inpainting" \
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python "$ROOT/src/inpainting/compare_object_mesh_volume.py" \
            --haco_2p5d_dir "$HACO_2P5D" \
            --mesh_front_dir "$MESH_FRONT" \
            --mesh_volume_shell_dir "$MESH_VOLUME_SHELL" \
            --mesh_volume_shell_temporal_dir "$MESH_VOLUME_TEMPORAL" \
            --mesh_dir "$MESH" \
            --object_mask "$OBJECT_SUPPORT" \
            --mapping "$MAPPING" \
            --overlay_input "$WRIST" \
            --joint_names "$JOINT_NAMES" \
            --out_dir "$COMPARISON"
        comparison_matches "$COMPARISON" "$MESH" "$HACO_2P5D" \
            "$MESH_FRONT" "$MESH_VOLUME_SHELL" "$MESH_VOLUME_TEMPORAL"
    fi

    echo "[$ID] fitted mesh volume: $MESH"
    echo "[$ID] full comparison:    $FULL_VIDEO"
    echo "[$ID] ROI comparison:     $ROI_VIDEO"
done
