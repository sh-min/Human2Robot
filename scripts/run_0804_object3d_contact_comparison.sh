#!/usr/bin/env bash
set -euo pipefail

# Build and compare MH object-surface 3-D contact occlusion variants.
# SH remains a synchronized HaCo confidence auxiliary; all projected contact,
# dense object geometry, and final compositing stay in the MH camera frame.
#
# Usage:
#   bash scripts/run_0804_object3d_contact_comparison.sh 1
#   FORCE=1 bash scripts/run_0804_object3d_contact_comparison.sh 1
#   ALL=1 bash scripts/run_0804_object3d_contact_comparison.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA="${DATA:-$ROOT/data/cube_dataset/26.08.04_stereo}"
ENVIRONMENT="${ENVIRONMENT:-inpaint-gpu}"
FORCE="${FORCE:-0}"
ALL="${ALL:-0}"
CHECKPOINT="${CHECKPOINT:-$ROOT/weights/depth_anything/depth_anything_v2_metric_hypersim_vits.pth}"

if [ "$#" -gt 0 ]; then
    EPISODES=("$@")
elif [ "$ALL" = "1" ]; then
    mapfile -t EPISODES < <(
        find "$DATA" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
            awk '/^[0-9]+$/' | sort -V
    )
else
    EPISODES=("1")
fi

for ID in "${EPISODES[@]}"; do
    [[ "$ID" =~ ^[0-9]+$ ]] || {
        echo "Episode names must be numeric, got '$ID'" >&2
        exit 1
    }
done

test -s "$CHECKPOINT"
conda run -n "$ENVIRONMENT" python -c 'import cv2, numpy, torch'
cd "$ROOT"

report_matches() {
    local directory="$1"
    local mode="$2"
    local alignment="${3:-}"
    local expected_surface="${4:-}"
    local expected_scene_depth="${5:-}"
    conda run -n "$ENVIRONMENT" python -c '
import json, sys
from pathlib import Path
root, mode, alignment = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
expected_surface, expected_scene = sys.argv[4], sys.argv[5]
required = [
    root / "report.json",
    root / "video_overlay_contact.mp4",
    root / "occluded_finger_mask.npy",
]
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    raise SystemExit(1)
report = json.loads(required[0].read_text())
if report.get("occlusion_mode") != mode:
    raise SystemExit(1)
if alignment and report.get("object_surface_3d", {}).get("alignment") != alignment:
    raise SystemExit(1)
if mode == "object3d" and not report.get("invariants", {}).get(
    "object3d_haco_is_selector_only"
):
    raise SystemExit(1)
sources = report.get("sources", {})
if expected_surface and sources.get("object_surface_depth") != str(
    Path(expected_surface).resolve()
):
    raise SystemExit(1)
if expected_scene and sources.get("scene_depth") != str(
    Path(expected_scene).resolve()
):
    raise SystemExit(1)
' "$directory" "$mode" "$alignment" "$expected_surface" \
        "$expected_scene_depth"
}

fresh_file() {
    local target="$1"
    shift
    [ -s "$target" ] || return 1
    local source
    for source in "$@"; do
        [ -e "$source" ] || return 1
        if [ "$source" -nt "$target" ]; then
            return 1
        fi
    done
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

surface_matches() {
    local directory="$1"
    local scene_depth="$2"
    local object_mask="$3"
    local hawor="$4"
    conda run -n "$ENVIRONMENT" python -c '
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
required = [
    root / "report.json",
    root / "object_surface_depth.npy",
    root / "object_surface_points.npy",
    root / "surface_stats.npz",
    root / "video_object_surface_3d.mp4",
]
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    raise SystemExit(1)
report = json.loads(required[0].read_text())
expected = {
    "scene_depth": str(Path(sys.argv[2]).resolve()),
    "object_mask": str(Path(sys.argv[3]).resolve()),
    "hawor_npz": str(Path(sys.argv[4]).resolve()),
}
if any(report.get("sources", {}).get(key) != value for key, value in expected.items()):
    raise SystemExit(1)
if report.get("representation", {}).get("not_watertight_mesh") is not True:
    raise SystemExit(1)
' "$directory" "$scene_depth" "$object_mask" "$hawor"
}

depth_matches() {
    local processed="$1"
    local checkpoint="$2"
    conda run -n "$ENVIRONMENT" python -c '
import cv2, numpy as np, sys
from pathlib import Path
processed, checkpoint = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
depth_dir = processed / "depth_processor"
raw_path = depth_dir / "depth_metric_raw.npy"
aligned_path = depth_dir / "depth_aligned_metric.npy"
params_path = depth_dir / "depth_metric_params.npz"
if not all(path.is_file() and path.stat().st_size > 0 for path in (
    raw_path, aligned_path, params_path
)):
    raise SystemExit(1)
capture = cv2.VideoCapture(str(processed / "video_L.mp4"))
try:
    expected = (
        int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )
finally:
    capture.release()
raw = np.load(raw_path, mmap_mode="r")
aligned = np.load(aligned_path, mmap_mode="r")
if raw.shape != expected or aligned.shape != expected:
    raise SystemExit(1)
with np.load(params_path) as params:
    stored = Path(str(params["checkpoint"])).resolve()
    encoder = str(params["encoder"])
if stored != checkpoint or encoder != "vits":
    raise SystemExit(1)
' "$processed" "$checkpoint"
}

for ID in "${EPISODES[@]}"; do
    EP="$DATA/$ID"
    C1="$EP/camera_1"
    C2="$EP/camera_2"
    PD="$C2/visibility/processed/view/0"
    SOURCE="$C2/source.mov"
    HAWOR="$C2/rgb_hawor/retarget_input.npz"
    OVERLAY="$PD/overlay_processor"
    OBJECT_MASK="$PD/object_layer/object_mask_modal.npy"
    DEPTH_DIR="$PD/depth_processor"
    ALIGNED_DEPTH="$DEPTH_DIR/depth_aligned_metric.npy"
    SURFACE_DIR="$PD/object_surface_3d"
    SURFACE_DEPTH="$SURFACE_DIR/object_surface_depth.npy"
    BASELINE="$PD/contact_occlusion_dual_haco_raw"
    SCALAR="$PD/contact_occlusion_dual_haco_object3d_scalar_raw"
    SURFACE="$PD/contact_occlusion_dual_haco_object3d_surface_raw"
    ALIGNED="$PD/contact_occlusion_dual_haco_object3d_contact_aligned_raw"
    COMPARISON="$PD/contact_occlusion_compare_object3d_raw"

    REQUIRED=(
        "$EP/stereo_manifest.json"
        "$SOURCE"
        "$HAWOR"
        "$C1/contact"
        "$C2/contact"
        "$PD/video_L.mp4"
        "$PD/hand_processor/hand_data_left.npz"
        "$OVERLAY/robot_depth.npy"
        "$OBJECT_MASK"
    )
    for REQUIRED_PATH in "${REQUIRED[@]}"; do
        if [ ! -e "$REQUIRED_PATH" ]; then
            echo "[$ID] missing prerequisite: $REQUIRED_PATH" >&2
            exit 1
        fi
    done
    FRAME_OFFSET=$(python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["temporal_alignment"]["camera1_frame_offset"])' \
        "$EP/stereo_manifest.json")
    SIDE=$(python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["side"])' \
        "$OVERLAY/manifest.json")

    DEPTH_PARAMS="$DEPTH_DIR/depth_metric_params.npz"
    RAW_DEPTH="$DEPTH_DIR/depth_metric_raw.npy"
    DEPTH_READY=0
    if [ "$FORCE" != "1" ] && depth_matches "$PD" "$CHECKPOINT" && \
       fresh_file "$ALIGNED_DEPTH" \
           "$PD/video_L.mp4" \
           "$PD/hand_processor/hand_data_left.npz" \
           "$PD/hand_processor/hand_data_right.npz" \
           "$CHECKPOINT" \
           "$ROOT/src/inpainting/estimate_metric_depth.py" && \
       fresh_file "$DEPTH_PARAMS" \
           "$PD/video_L.mp4" "$CHECKPOINT" \
           "$ROOT/src/inpainting/estimate_metric_depth.py" && \
       [ -s "$RAW_DEPTH" ]; then
        DEPTH_READY=1
    fi
    if [ "$DEPTH_READY" != "1" ]; then
        echo "[$ID] Depth Anything V2 metric surface inference (MH)"
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python src/inpainting/estimate_metric_depth.py \
            --processed_demo "$PD" \
            --encoder vits \
            --input_size 518 \
            --checkpoint "$CHECKPOINT"
    else
        echo "[$ID] metric depth skip: $ALIGNED_DEPTH"
    fi

    SURFACE_READY=0
    if [ "$FORCE" != "1" ] && \
       surface_matches "$SURFACE_DIR" "$ALIGNED_DEPTH" \
           "$OBJECT_MASK" "$HAWOR" && \
       fresh_file "$SURFACE_DIR/report.json" \
           "$ALIGNED_DEPTH" "$OBJECT_MASK" "$HAWOR" "$SOURCE" \
           "$ROOT/src/inpainting/build_object_surface_model.py" && \
       fresh_file "$SURFACE_DEPTH" \
           "$ALIGNED_DEPTH" "$OBJECT_MASK" "$HAWOR" \
           "$ROOT/src/inpainting/build_object_surface_model.py"; then
        SURFACE_READY=1
    fi
    if [ "$SURFACE_READY" != "1" ]; then
        echo "[$ID] robust modal-object surface + point cloud (MH)"
        PYTHONPATH="$ROOT/src/inpainting" \
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python src/inpainting/build_object_surface_model.py \
            --scene_depth "$ALIGNED_DEPTH" \
            --object_mask "$OBJECT_MASK" \
            --hawor_npz "$HAWOR" \
            --video "$SOURCE" \
            --out_dir "$SURFACE_DIR"
    else
        echo "[$ID] object surface skip: $SURFACE_DEPTH"
    fi

    COMMON_ARGS=(
        --processed_demo "$PD"
        --episode_dir "$C2"
        --background "$SOURCE"
        --raw_video "$SOURCE"
        --hawor_npz "$HAWOR"
        --contact_dir "$C2/contact"
        --aux_contact_dir "$C1/contact"
        --aux_frame_offset "$FRAME_OFFSET"
        --aux_side "$SIDE"
        --overlay_dir "$OVERLAY"
        --object_mask "$OBJECT_MASK"
    )
    COMPOSITOR_INPUTS=(
        "$ROOT/src/inpainting/composite_rb5_contact_occlusion.py"
        "$SOURCE"
        "$HAWOR"
        "$C1/contact"
        "$C2/contact"
        "$EP/stereo_manifest.json"
        "$OVERLAY/manifest.json"
        "$OVERLAY/robot_rgb.npy"
        "$OVERLAY/robot_depth.npy"
        "$OVERLAY/robot_mask.npy"
        "$OVERLAY/robot_finger_mask.npy"
        "$OVERLAY/robot_finger_labels.npy"
        "$OBJECT_MASK"
    )

    BASELINE_READY=0
    if [ "$FORCE" != "1" ] && report_matches "$BASELINE" haco && \
       fresh_file "$BASELINE/report.json" "${COMPOSITOR_INPUTS[@]}" && \
       fresh_tree "$BASELINE/report.json" "$C1/contact" && \
       fresh_tree "$BASELINE/report.json" "$C2/contact"; then
        BASELINE_READY=1
    fi
    if [ "$BASELINE_READY" != "1" ]; then
        echo "[$ID] current HaCo proxy baseline"
        PYTHONPATH="$ROOT/src/inpainting" \
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python src/inpainting/composite_rb5_contact_occlusion.py \
            "${COMMON_ARGS[@]}" --occlusion_mode haco --out_dir "$BASELINE"
    fi
    SCALAR_READY=0
    if [ "$FORCE" != "1" ] && \
       report_matches "$SCALAR" ensemble "" "" "$ALIGNED_DEPTH" && \
       fresh_file "$SCALAR/report.json" \
           "${COMPOSITOR_INPUTS[@]}" "$ALIGNED_DEPTH" && \
       fresh_tree "$SCALAR/report.json" "$C1/contact" && \
       fresh_tree "$SCALAR/report.json" "$C2/contact"; then
        SCALAR_READY=1
    fi
    if [ "$SCALAR_READY" != "1" ]; then
        echo "[$ID] ablation: one robust object depth per frame"
        PYTHONPATH="$ROOT/src/inpainting" \
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python src/inpainting/composite_rb5_contact_occlusion.py \
            "${COMMON_ARGS[@]}" \
            --scene_depth "$ALIGNED_DEPTH" \
            --object_depth_mask "$OBJECT_MASK" \
            --occlusion_mode ensemble --out_dir "$SCALAR"
    fi
    DENSE_READY=0
    if [ "$FORCE" != "1" ] && \
       report_matches "$SURFACE" object3d none "$SURFACE_DEPTH" && \
       fresh_file "$SURFACE/report.json" \
           "${COMPOSITOR_INPUTS[@]}" "$SURFACE_DEPTH" && \
       fresh_tree "$SURFACE/report.json" "$C1/contact" && \
       fresh_tree "$SURFACE/report.json" "$C2/contact"; then
        DENSE_READY=1
    fi
    if [ "$DENSE_READY" != "1" ]; then
        echo "[$ID] ablation: dense object surface without contact alignment"
        PYTHONPATH="$ROOT/src/inpainting" \
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python src/inpainting/composite_rb5_contact_occlusion.py \
            "${COMMON_ARGS[@]}" \
            --object_surface_depth "$SURFACE_DEPTH" \
            --object_surface_alignment none \
            --occlusion_mode object3d --out_dir "$SURFACE"
    fi
    ALIGNED_READY=0
    if [ "$FORCE" != "1" ] && \
       report_matches "$ALIGNED" object3d contact "$SURFACE_DEPTH" && \
       fresh_file "$ALIGNED/report.json" \
           "${COMPOSITOR_INPUTS[@]}" "$SURFACE_DEPTH" && \
       fresh_tree "$ALIGNED/report.json" "$C1/contact" && \
       fresh_tree "$ALIGNED/report.json" "$C2/contact"; then
        ALIGNED_READY=1
    fi
    if [ "$ALIGNED_READY" != "1" ]; then
        echo "[$ID] proposed: dense object surface + local HaCo registration"
        PYTHONPATH="$ROOT/src/inpainting" \
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python src/inpainting/composite_rb5_contact_occlusion.py \
            "${COMMON_ARGS[@]}" \
            --object_surface_depth "$SURFACE_DEPTH" \
            --object_surface_alignment contact \
            --occlusion_mode object3d --out_dir "$ALIGNED"
    fi

    COMPARISON_VIDEO="$COMPARISON/video_compare_object3d_contact_2x2.mp4"
    COMPARISON_REPORT="$COMPARISON/comparison_report.json"
    COMPARISON_READY=0
    if [ "$FORCE" != "1" ] && \
       fresh_file "$COMPARISON_VIDEO" \
           "$ROOT/src/inpainting/compare_object3d_contact_occlusion.py" \
           "$BASELINE/report.json" "$BASELINE/video_overlay_contact.mp4" \
           "$BASELINE/occluded_finger_mask.npy" \
           "$SCALAR/report.json" "$SCALAR/video_overlay_contact.mp4" \
           "$SCALAR/occluded_finger_mask.npy" \
           "$SURFACE/report.json" "$SURFACE/video_overlay_contact.mp4" \
           "$SURFACE/occluded_finger_mask.npy" \
           "$ALIGNED/report.json" "$ALIGNED/video_overlay_contact.mp4" \
           "$ALIGNED/occluded_finger_mask.npy" && \
       fresh_file "$COMPARISON_REPORT" \
           "$ROOT/src/inpainting/compare_object3d_contact_occlusion.py" \
           "$BASELINE/report.json" "$SCALAR/report.json" \
           "$SURFACE/report.json" "$ALIGNED/report.json"; then
        COMPARISON_READY=1
    fi
    if [ "$COMPARISON_READY" != "1" ]; then
        echo "[$ID] synchronized 2x2 comparison"
        PYTHONPATH="$ROOT/src/inpainting" \
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python src/inpainting/compare_object3d_contact_occlusion.py \
            --baseline_dir "$BASELINE" \
            --scalar_dir "$SCALAR" \
            --surface_dir "$SURFACE" \
            --contact_aligned_dir "$ALIGNED" \
            --out_dir "$COMPARISON"
    fi
    echo "[$ID] result: $COMPARISON/video_compare_object3d_contact_2x2.mp4"
done
