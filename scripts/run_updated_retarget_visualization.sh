#!/usr/bin/env bash
set -euo pipefail

# Rebuild retargeting and the current HaWoR/HaCo + required-stage
# visualizations using the depth-aware render helpers.
#
# New outputs are written to temporary names and frame-validated before the
# canonical outputs are replaced.  Set CLEAN_STALE=1 to delete files from the
# retired clipped/full-arm render attempts after successful validation.
#
# Usage:
#   CLEAN_STALE=1 bash scripts/run_updated_retarget_visualization.sh \
#     IMG_5020 IMG_5021

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA="${DATA:-$ROOT/data/kitchen_dataset/26.07.24}"
CLEAN_STALE="${CLEAN_STALE:-0}"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 IMG_5020 [IMG_5021 ...]" >&2
    exit 2
fi

cd "$ROOT"

conda run -n RFM_retarget python -c \
  'import dex_retargeting, numpy, scipy'
conda run -n inpaint-gpu python -c \
  'import cv2, mediapy, numpy, pyrender, trimesh'

for ID in "$@"; do
    EP="$DATA/$ID"
    HAWOR="$EP/rgb_hawor"
    NPZ="$HAWOR/retarget_input.npz"
    PD="$EP/inpainting_processed/$ID/0"

    RIGHT_RAW="$HAWOR/qpos_xhand_right.pkl"
    LEFT_RAW="$HAWOR/qpos_xhand_left.pkl"
    RIGHT_SMOOTH="$HAWOR/qpos_xhand_right_smooth.pkl"
    LEFT_SMOOTH="$HAWOR/qpos_xhand_left_smooth.pkl"

    NEW_FINAL="$PD/video_overlay_rby1_xhand.new.mp4"
    NEW_ROBOT="$PD/overlay_processor/video_robot_only.new.mp4"

    FINAL="$PD/video_overlay_rby1_xhand.mp4"
    ROBOT="$PD/overlay_processor/video_robot_only.mp4"

    test -s "$NPZ"
    test -s "$PD/video_L.mp4"
    test -s "$PD/inpaint_processor/video_human_inpaint.mkv"
    test -s "$PD/segmentation_processor/masks_arm.npy"

    EXPECTED=$(
      conda run -n RFM_retarget python -c \
        "import numpy as np; print(np.load('$NPZ')['joints_left'].shape[0])" \
      | tr -d '[:space:]'
    )

    echo
    echo "========================================"
    echo "[$ID] updated retarget/render 시작 (${EXPECTED} frames)"
    echo "========================================"

    (
      cd "$ROOT/src/retargeting"
      conda run -n RFM_retarget --no-capture-output \
        python retarget_from_npz.py \
          --npz "$NPZ" \
          --hand both \
          --right_embodiment xhand \
          --left_embodiment xhand \
          --smooth
    )
    test -s "$RIGHT_RAW"
    test -s "$LEFT_RAW"
    test -s "$RIGHT_SMOOTH"
    test -s "$LEFT_SMOOTH"

    (
      cd "$ROOT/src/inpainting"
      PYOPENGL_PLATFORM=egl \
      MPLCONFIGDIR=/tmp/inpaint-mpl \
      conda run -n inpaint-gpu --no-capture-output \
        python -u render_xhand_overlay_depth.py \
          --processed_demo "$PD" \
          --hawor_npz "$NPZ" \
          --right_pkl "$RIGHT_SMOOTH" \
          --left_pkl "$LEFT_SMOOTH" \
          --hand both \
          --relight auto
    )
    test -s "$PD/overlay_processor/robot_rgb.npy"
    test -s "$PD/overlay_processor/robot_depth.npy"
    test -s "$PD/overlay_processor/robot_mask.npy"

    conda run -n inpaint-gpu --no-capture-output \
      python "$ROOT/src/inpainting/composite_robot_unclipped.py" \
        --processed_demo "$PD" \
        --out "$NEW_FINAL" \
        --robot_only_out "$NEW_ROBOT" \
        --fps 30

    FINAL_FRAMES=$(
      ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$NEW_FINAL"
    )
    ROBOT_FRAMES=$(
      ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$NEW_ROBOT"
    )
    if [ "$FINAL_FRAMES" -ne "$EXPECTED" ] \
       || [ "$ROBOT_FRAMES" -ne "$EXPECTED" ]; then
        echo "[$ID] frame mismatch: expected=$EXPECTED final=$FINAL_FRAMES robot=$ROBOT_FRAMES" >&2
        exit 1
    fi

    if [ "$CLEAN_STALE" = "1" ]; then
        # Exact paths only: outputs from retired clipped/full-arm attempts.
        rm -f -- \
          "$PD/video_overlay_rby1_xhand.mkv" \
          "$PD/video_overlay_rby1_xhand.mp4" \
          "$PD/pipeline_components_rby1_xhand.mp4" \
          "$PD/video_overlay_xhand.mkv" \
          "$PD/video_overlay_xhand.mp4" \
          "$PD/pipeline_components.mp4" \
          "$PD/retarget_updated_helper_preview.mp4" \
          "$PD/retarget_updated_helper_qc.jpg" \
          "$PD/overlay_processor/video_overlay_raw.mkv" \
          "$PD/overlay_processor/residual_mask.npy" \
          "$PD/overlay_processor/video_robot_only.mp4" \
          "$PD/overlay_processor_arm/video_robot_only.mkv" \
          "$PD/overlay_processor_arm/robot_mask.npz" \
          "$PD/render_backup_before_updated_helper/video_overlay_rby1_xhand.mkv" \
          "$PD/render_backup_before_updated_helper/video_overlay_rby1_xhand.mp4" \
          "$PD/render_backup_before_updated_helper/pipeline_components_rby1_xhand.mp4"

        if [ -d "$PD/overlay_processor_arm" ]; then
            rmdir "$PD/overlay_processor_arm" 2>/dev/null || true
        fi
        if [ -d "$PD/render_backup_before_updated_helper" ]; then
            rmdir "$PD/render_backup_before_updated_helper" 2>/dev/null || true
        fi
    fi

    mv "$NEW_FINAL" "$FINAL"
    mv "$NEW_ROBOT" "$ROBOT"

    echo "[$ID] 완료"
    echo "  final: $FINAL"
    echo "  robot: $ROBOT"
done

RECORDING_GLOB=$(IFS=,; echo "$*")

conda run -n inpaint-gpu --no-capture-output \
  python "$ROOT/src/hand_estimation/visualize_hawor_haco.py" \
    --data_root "$DATA" \
    --recording_glob "$RECORDING_GLOB"

conda run -n inpaint-gpu --no-capture-output \
  python "$ROOT/src/inpainting/visualize_required_pipeline.py" \
    --data_root "$DATA" \
    --recording_glob "$RECORDING_GLOB"

for ID in "$@"; do
    PD="$DATA/$ID/inpainting_processed/$ID/0"
    EXPECTED=$(
      conda run -n RFM_retarget python -c \
        "import numpy as np; print(np.load('$DATA/$ID/rgb_hawor/retarget_input.npz')['joints_left'].shape[0])" \
      | tr -d '[:space:]'
    )
    GRID="$PD/pipeline_required_components.mp4"
    GRID_FRAMES=$(
      ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$GRID"
    )
    if [ "$GRID_FRAMES" -ne "$EXPECTED" ]; then
        echo "[$ID] comparison frame mismatch: expected=$EXPECTED grid=$GRID_FRAMES" >&2
        exit 1
    fi
    echo "[$ID] comparison: $GRID"
done

echo
echo "전체 updated retarget/render 완료"
