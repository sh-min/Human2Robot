#!/usr/bin/env bash
set -euo pipefail

# Compare full-finger dense-surface forcing and short-gap penetration filtering.
#
# Usage:
#   bash scripts/run_0804_object3d_penetration_comparison.sh 1
#   FORCE=1 bash scripts/run_0804_object3d_penetration_comparison.sh 1

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

fresh_tree() {
    local target="$1"
    local tree="$2"
    [ -s "$target" ] && [ -d "$tree" ] || return 1
    ! find "$tree" -type f -newer "$target" -print -quit | grep -q .
}

variant_matches() {
    local directory="$1"
    local expected_force="$2"
    local expected_gap="$3"
    local expected_surface="$4"
    conda run -n "$ENVIRONMENT" python -c '
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
required = [
    root / "report.json",
    root / "video_overlay_contact.mp4",
    root / "occluded_finger_mask.npy",
]
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    raise SystemExit(1)
report = json.loads(required[0].read_text())
config = report.get("config", {})
if report.get("occlusion_mode") != "object3d":
    raise SystemExit(1)
if report.get("object_surface_3d", {}).get("alignment") != "contact":
    raise SystemExit(1)
if bool(config.get("object3d_force_surface", False)) != (sys.argv[2] == "1"):
    raise SystemExit(1)
if int(config.get("object3d_temporal_max_gap_frames", 0)) != int(sys.argv[3]):
    raise SystemExit(1)
if report.get("sources", {}).get("object_surface_depth") != str(
    Path(sys.argv[4]).resolve()
):
    raise SystemExit(1)
' "$directory" "$expected_force" "$expected_gap" "$expected_surface" \
        >/dev/null 2>&1
}

for ID in "${EPISODES[@]}"; do
    [[ "$ID" =~ ^[0-9]+$ ]] || {
        echo "Episode names must be numeric, got '$ID'" >&2
        exit 1
    }
    EP="$DATA/$ID"
    C1="$EP/camera_1"
    C2="$EP/camera_2"
    PD="$C2/visibility/processed/view/0"
    SOURCE="$C2/source.mov"
    HAWOR="$C2/rgb_hawor/retarget_input.npz"
    OVERLAY="$PD/overlay_processor"
    OBJECT_MASK="$PD/object_layer/object_mask_modal.npy"
    SURFACE_DEPTH="$PD/object_surface_3d/object_surface_depth.npy"
    BASELINE="$PD/contact_occlusion_dual_haco_object3d_contact_aligned_raw"
    SURFACE_FORCE="$PD/contact_occlusion_dual_haco_object3d_contact_aligned_force_behind_raw"
    TEMPORAL="$PD/contact_occlusion_dual_haco_object3d_contact_aligned_temporal_filter_raw"
    FORCE_TEMPORAL="$PD/contact_occlusion_dual_haco_object3d_contact_aligned_force_temporal_raw"
    COMPARISON="$PD/contact_occlusion_compare_object3d_penetration_raw"

    REQUIRED=(
        "$EP/stereo_manifest.json"
        "$SOURCE"
        "$HAWOR"
        "$C1/contact"
        "$C2/contact"
        "$OVERLAY/manifest.json"
        "$OVERLAY/robot_rgb.npy"
        "$OVERLAY/robot_depth.npy"
        "$OVERLAY/robot_mask.npy"
        "$OVERLAY/robot_finger_mask.npy"
        "$OVERLAY/robot_finger_labels.npy"
        "$OBJECT_MASK"
        "$SURFACE_DEPTH"
        "$BASELINE/report.json"
        "$BASELINE/video_overlay_contact.mp4"
        "$BASELINE/occluded_finger_mask.npy"
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
        --object_surface_depth "$SURFACE_DEPTH"
        --object_surface_alignment contact
        --occlusion_mode object3d
    )
    INPUTS=(
        "$ROOT/src/inpainting/composite_rb5_contact_occlusion.py"
        "$SOURCE"
        "$HAWOR"
        "$EP/stereo_manifest.json"
        "$OVERLAY/manifest.json"
        "$OVERLAY/robot_rgb.npy"
        "$OVERLAY/robot_depth.npy"
        "$OVERLAY/robot_mask.npy"
        "$OVERLAY/robot_finger_mask.npy"
        "$OVERLAY/robot_finger_labels.npy"
        "$OBJECT_MASK"
        "$SURFACE_DEPTH"
    )

    run_variant() {
        local name="$1"
        local directory="$2"
        local expected_force="$3"
        local expected_gap="$4"
        shift 4
        local ready=0
        if [ "$FORCE" != "1" ] && \
           variant_matches "$directory" "$expected_force" \
               "$expected_gap" "$SURFACE_DEPTH" && \
           fresh_file "$directory/report.json" "${INPUTS[@]}" && \
           fresh_tree "$directory/report.json" "$C1/contact" && \
           fresh_tree "$directory/report.json" "$C2/contact"; then
            ready=1
        fi
        if [ "$ready" = "1" ]; then
            echo "[$ID] $name is current"
            return
        fi
        echo "[$ID] rendering $name"
        PYTHONPATH="$ROOT/src/inpainting" \
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python src/inpainting/composite_rb5_contact_occlusion.py \
            "${COMMON_ARGS[@]}" "$@" --out_dir "$directory"
    }

    cd "$ROOT"
    run_variant "full-finger surface-force" "$SURFACE_FORCE" 1 0 \
        --object3d_force_surface --object3d_force_margin_m 0
    run_variant "two-frame temporal suppression" "$TEMPORAL" 0 2 \
        --object3d_temporal_max_gap_frames 2 \
        --object3d_temporal_motion_px 6 \
        --object3d_temporal_front_slack_m 0.015
    run_variant "surface-force + temporal suppression" "$FORCE_TEMPORAL" 1 2 \
        --object3d_force_surface --object3d_force_margin_m 0 \
        --object3d_temporal_max_gap_frames 2 \
        --object3d_temporal_motion_px 6 \
        --object3d_temporal_front_slack_m 0.015

    COMPARE_INPUTS=(
        "$ROOT/src/inpainting/compare_object3d_penetration_strategies.py"
        "$ROOT/src/inpainting/make_video_comparison_grid.py"
        "$BASELINE/report.json" "$BASELINE/video_overlay_contact.mp4"
        "$BASELINE/occluded_finger_mask.npy"
        "$SURFACE_FORCE/report.json" "$SURFACE_FORCE/video_overlay_contact.mp4"
        "$SURFACE_FORCE/occluded_finger_mask.npy"
        "$TEMPORAL/report.json" "$TEMPORAL/video_overlay_contact.mp4"
        "$TEMPORAL/occluded_finger_mask.npy"
        "$FORCE_TEMPORAL/report.json" "$FORCE_TEMPORAL/video_overlay_contact.mp4"
        "$FORCE_TEMPORAL/occluded_finger_mask.npy"
    )
    COMPARE_VIDEO="$COMPARISON/video_compare_object3d_penetration_2x2.mp4"
    COMPARE_REPORT="$COMPARISON/comparison_report.json"
    COMPARE_READY=0
    if [ "$FORCE" != "1" ] && \
       fresh_file "$COMPARE_VIDEO" "${COMPARE_INPUTS[@]}" && \
       fresh_file "$COMPARE_REPORT" "${COMPARE_INPUTS[@]}"; then
        COMPARE_READY=1
    fi
    if [ "$COMPARE_READY" != "1" ]; then
        echo "[$ID] rendering synchronized penetration-control 2x2"
        PYTHONPATH="$ROOT/src/inpainting" \
        conda run -n "$ENVIRONMENT" --no-capture-output \
            python src/inpainting/compare_object3d_penetration_strategies.py \
            --baseline_dir "$BASELINE" \
            --surface_force_dir "$SURFACE_FORCE" \
            --temporal_dir "$TEMPORAL" \
            --force_temporal_dir "$FORCE_TEMPORAL" \
            --out_dir "$COMPARISON"
    fi
    echo "[$ID] result: $COMPARE_VIDEO"
done
