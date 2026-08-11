#!/usr/bin/env bash
set -euo pipefail

# Re-render RBY1 + XHand overlays using the complete smoothed trajectory:
# finger qpos + wrist position + wrist orientation. Each episode is rendered
# to a temporary directory, validated, then atomically replaces the old output.
#
# Usage:
#   DATA=data/kitchen_dataset/26.07.27 \
#     bash scripts/rerender_smoothed_rby1_overlays.sh 07_27-1
#   DATA=data/kitchen_dataset/26.07.27 ALL=1 \
#     bash scripts/rerender_smoothed_rby1_overlays.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA="${DATA:-$ROOT/data/kitchen_dataset/26.07.27}"
ALL="${ALL:-0}"

if [ "$#" -gt 0 ]; then
    EPISODE_IDS=("$@")
elif [ "$ALL" = "1" ]; then
    EPISODE_IDS=()
    for EP in "$DATA"/*; do
        [ -d "$EP" ] || continue
        [ -d "$EP/rgb" ] || continue
        EPISODE_IDS+=("$(basename "$EP")")
    done
else
    echo "episode ID를 지정하거나 ALL=1을 설정하세요." >&2
    exit 2
fi

for ID in "${EPISODE_IDS[@]}"; do
    EP="$DATA/$ID"
    PD="$EP/inpainting_processed/$ID/0"
    HAWOR="$EP/rgb_hawor"
    NPZ="$HAWOR/retarget_input.npz"
    LEFT_PKL="$HAWOR/qpos_xhand_left_smooth.pkl"
    RIGHT_PKL="$HAWOR/qpos_xhand_right_smooth.pkl"
    BG="$PD/inpaint_processor/video_human_inpaint.mkv"
    FINAL_MKV="$PD/video_overlay_rby1_xhand.mkv"
    FINAL_MP4="$PD/video_overlay_rby1_xhand.mp4"
    AUX="$PD/overlay_processor_arm"
    GRID="$PD/pipeline_components_rby1_xhand.mp4"

    for REQUIRED in "$NPZ" "$LEFT_PKL" "$RIGHT_PKL" "$BG"; do
        test -s "$REQUIRED" || {
            echo "[$ID] 누락: $REQUIRED" >&2
            exit 1
        }
    done

    TMP="$(mktemp -d "$PD/.smoothed-rerender.XXXXXX")"
    TMP_AUX="$TMP/aux"
    TMP_MKV="$TMP/video_overlay_rby1_xhand.mkv"
    TMP_MP4="$TMP/video_overlay_rby1_xhand.mp4"
    TMP_GRID="$TMP/pipeline_components_rby1_xhand.mp4"

    echo "[$ID] 완전 스무딩 렌더링"
    PYOPENGL_PLATFORM=egl \
    MPLCONFIGDIR=/tmp/inpaint-mpl \
    conda run -n inpaint-gpu --no-capture-output \
      python -u "$ROOT/src/inpainting/render_rby1_xhand_full_arm.py" \
        --processed_demo "$PD" \
        --hawor_npz "$NPZ" \
        --right_pkl "$RIGHT_PKL" \
        --left_pkl "$LEFT_PKL" \
        --hand both \
        --require_smoothed \
        --fps 30 \
        --output "$TMP_MKV" \
        --aux_output_dir "$TMP_AUX"

    ffmpeg -nostdin -y -hide_banner -loglevel error \
      -i "$TMP_MKV" \
      -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' \
      -c:v libx264 -crf 18 -pix_fmt yuv420p \
      "$TMP_MP4"

    EXPECTED=$(ffprobe -v error -select_streams v:0 -count_frames \
      -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$BG")
    ACTUAL=$(ffprobe -v error -select_streams v:0 -count_frames \
      -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$TMP_MP4")
    test "$EXPECTED" -eq "$ACTUAL"
    test -s "$TMP_AUX/render_metadata.json"

    mkdir -p "$AUX"
    mv -f "$TMP_MKV" "$FINAL_MKV"
    mv -f "$TMP_MP4" "$FINAL_MP4"
    mv -f "$TMP_AUX/video_robot_only.mkv" "$AUX/video_robot_only.mkv"
    mv -f "$TMP_AUX/robot_mask.npz" "$AUX/robot_mask.npz"
    mv -f "$TMP_AUX/render_metadata.json" "$AUX/render_metadata.json"

    MPLCONFIGDIR=/tmp/inpaint-mpl \
    conda run -n inpaint-gpu --no-capture-output \
      python "$ROOT/src/inpainting/visualize_pipeline_grid.py" \
        --processed_demo "$PD" \
        --out "$TMP_GRID"
    test -s "$TMP_GRID"
    mv -f "$TMP_GRID" "$GRID"

    rmdir "$TMP_AUX"
    rmdir "$TMP"
    echo "[$ID] 교체 완료 — $ACTUAL frames"
done

echo "완전 스무딩 RBY1 + XHand 오버레이 재생성 완료"
