#!/usr/bin/env bash
set -euo pipefail

# Render the complete synchronized pre/post Object3D comparison for 08_04.
#
# Usage:
#   bash scripts/run_0804_all_before_after_comparison.sh 1
#   FORCE=1 bash scripts/run_0804_all_before_after_comparison.sh 1
#   ALL=1 bash scripts/run_0804_all_before_after_comparison.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA="${DATA:-$ROOT/data/kitchen_dataset/26.08.04_stereo}"
FORCE="${FORCE:-0}"
ALL="${ALL:-0}"

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

    PD="$DATA/$ID/camera_2/visibility/processed/view/0"
    OUT="$PD/contact_occlusion_compare_all_before_after_raw"
    TARGET="$OUT/video_compare_all_before_after_3x4.mp4"
    REPORT="$OUT/comparison_report.json"
    SOURCES=(
        "$PD/contact_occlusion_dual_haco_raw/video_overlay_contact.mp4"
        "$PD/contact_occlusion_dual_haco_raw/report.json"
        "$PD/contact_occlusion_dual_haco_xhanddepth_s0p5_t39p16mm_f29p3mm_raw/video_overlay_contact.mp4"
        "$PD/contact_occlusion_dual_haco_xhanddepth_s0p5_t39p16mm_f29p3mm_raw/report.json"
        "$PD/contact_occlusion_dual_haco_xhanddepth_s1_t39p16mm_f29p3mm_raw/video_overlay_contact.mp4"
        "$PD/contact_occlusion_dual_haco_xhanddepth_s1_t39p16mm_f29p3mm_raw/report.json"
        "$PD/contact_occlusion_dual_haco_boundaryfill_3px_cap25_raw/video_overlay_contact.mp4"
        "$PD/contact_occlusion_dual_haco_boundaryfill_3px_cap25_raw/report.json"
        "$PD/stereo_occlusion_visibility_force_raw/video_overlay_visibility.mp4"
        "$PD/stereo_occlusion_visibility_force_raw/report.json"
        "$PD/contact_occlusion_compare_xhand_surface_strategies_raw/video_overlay_baseline_force_union.mp4"
        "$PD/contact_occlusion_compare_xhand_surface_strategies_raw/video_overlay_union_safety_shell_diagnostic.mp4"
        "$PD/contact_occlusion_compare_xhand_surface_strategies_raw/video_overlay_surface_front_side_half.mp4"
        "$PD/contact_occlusion_compare_xhand_surface_strategies_raw/video_overlay_surface_front_side_half_back_full.mp4"
        "$PD/contact_occlusion_compare_xhand_surface_strategies_raw/comparison_report.json"
        "$PD/contact_occlusion_dual_haco_object3d_scalar_raw/video_overlay_contact.mp4"
        "$PD/contact_occlusion_dual_haco_object3d_scalar_raw/report.json"
        "$PD/contact_occlusion_dual_haco_object3d_surface_raw/video_overlay_contact.mp4"
        "$PD/contact_occlusion_dual_haco_object3d_surface_raw/report.json"
        "$PD/contact_occlusion_dual_haco_object3d_contact_aligned_raw/video_overlay_contact.mp4"
        "$PD/contact_occlusion_dual_haco_object3d_contact_aligned_raw/report.json"
        "$ROOT/src/inpainting/compare_all_contact_occlusion_results.py"
        "$ROOT/src/inpainting/make_video_comparison_grid.py"
    )
    for SOURCE in "${SOURCES[@]}"; do
        if [ ! -s "$SOURCE" ]; then
            echo "[$ID] missing comparison source: $SOURCE" >&2
            exit 1
        fi
    done

    CURRENT=1
    if [ "$FORCE" = "1" ] || [ ! -s "$TARGET" ] || [ ! -s "$REPORT" ]; then
        CURRENT=0
    else
        for SOURCE in "${SOURCES[@]}"; do
            if [ "$SOURCE" -nt "$TARGET" ] || [ "$SOURCE" -nt "$REPORT" ]; then
                CURRENT=0
                break
            fi
        done
    fi

    if [ "$CURRENT" = "1" ]; then
        echo "[$ID] complete before/after comparison is current"
        continue
    fi

    echo "[$ID] rendering synchronized 3x4 before/after comparison"
    (
        cd "$ROOT"
        python src/inpainting/compare_all_contact_occlusion_results.py \
            --processed_demo "$PD" \
            --out_dir "$OUT"
    )
    echo "[$ID] result: $TARGET"
done
