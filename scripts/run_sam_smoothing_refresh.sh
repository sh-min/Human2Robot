#!/usr/bin/env bash
set -euo pipefail

# Rebuild the SAM2-dependent stages while retaining the exact pre-smoothing
# outputs for A/B comparison. HaWoR, HaCo, retargeting, and robot rendering are
# reused because SAM2 smoothing does not change any of them.
#
# Usage:
#   bash scripts/run_sam_smoothing_refresh.sh IMG_5019
#   bash scripts/run_sam_smoothing_refresh.sh IMG_5019 IMG_5020

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA="${DATA:-$ROOT/data/cube_dataset/26.07.24}"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 IMG_5019 [IMG_5020 ...]" >&2
    exit 2
fi

cd "$ROOT"

conda run -n inpaint-gpu python -c \
  'import cv2, mediapy, numpy, torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'

for ID in "$@"; do
    EP="$DATA/$ID"
    PD="$EP/inpainting_processed/$ID/0"
    SEG="$PD/segmentation_processor"
    INP="$PD/inpaint_processor"

    MASK_ON="$SEG/masks_arm.npy"
    MASK_OFF="$SEG/masks_arm_no_smooth.npy"
    MASK_NEW="$SEG/masks_arm.smooth.new.npy"
    BG_ON="$INP/video_human_inpaint.mkv"
    BG_OFF="$INP/video_human_inpaint_no_smooth.mkv"
    BG_NEW="$INP/video_human_inpaint.smooth.new.mkv"
    FINAL_ON="$PD/video_overlay_rby1_xhand.mp4"
    FINAL_OFF="$PD/video_overlay_rby1_xhand_no_smooth.mp4"
    FINAL_NEW="$PD/video_overlay_rby1_xhand.smooth.new.mp4"
    ROBOT="$PD/overlay_processor/video_robot_only.mp4"
    GRID_ON="$PD/pipeline_required_components.mp4"
    GRID_OFF="$PD/pipeline_required_components_no_smooth.mp4"

    echo
    echo "========================================"
    echo "[$ID] SAM2 smoothing refresh"
    echo "========================================"

    test -s "$PD/video_L.mp4"
    test -s "$PD/video_rgb_imgs.mkv"
    test -s "$EP/visualization/hawor_haco_comparison.mp4"
    test -s "$PD/overlay_processor/robot_rgb.npy"
    test -s "$PD/overlay_processor/robot_mask.npy"

    # The first run starts from the exact outputs produced before smoothing was
    # merged. Move, rather than copy, the ~1 GB mask volume; resume safely from
    # MASK_OFF if a later stage is interrupted.
    if [ ! -s "$MASK_OFF" ]; then
        test -s "$MASK_ON"
        mv "$MASK_ON" "$MASK_OFF"
        echo "[$ID] saved mask baseline: $MASK_OFF"
    fi
    if [ ! -s "$BG_OFF" ]; then
        test -s "$BG_ON"
        mv "$BG_ON" "$BG_OFF"
        echo "[$ID] saved inpaint baseline: $BG_OFF"
    fi
    if [ ! -s "$FINAL_OFF" ]; then
        test -s "$FINAL_ON"
        mv "$FINAL_ON" "$FINAL_OFF"
        echo "[$ID] saved final baseline: $FINAL_OFF"
    fi
    if [ ! -s "$GRID_OFF" ] && [ -s "$GRID_ON" ]; then
        mv "$GRID_ON" "$GRID_OFF"
    fi

    rm -f -- "$MASK_NEW" "$BG_NEW" "$FINAL_NEW"

    (
      cd "$ROOT/src/inpainting"
      MPLCONFIGDIR=/tmp/inpaint-mpl \
      conda run -n inpaint-gpu --no-capture-output \
        python smooth_arm_masks.py \
          --input "$MASK_OFF" \
          --output "$MASK_NEW"
    )
    mv "$MASK_NEW" "$MASK_ON"

    (
      cd "$ROOT/src/inpainting"
      MPLCONFIGDIR=/tmp/inpaint-mpl \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      conda run -n inpaint-gpu --no-capture-output \
        python inpaint_hands.py \
          --processed_demo "$PD" \
          --mode legacy \
          --mask "$MASK_ON" \
          --output "$BG_NEW" \
          --output_resolution 540 \
          --dilate_iter 4 \
          --fps 30
    )
    mv "$BG_NEW" "$BG_ON"

    conda run -n inpaint-gpu --no-capture-output \
      python "$ROOT/src/inpainting/composite_robot_unclipped.py" \
        --processed_demo "$PD" \
        --background "$BG_ON" \
        --out "$FINAL_NEW" \
        --robot_only_out "$ROBOT" \
        --fps 30
    mv "$FINAL_NEW" "$FINAL_ON"

    conda run -n inpaint-gpu --no-capture-output \
      python "$ROOT/src/inpainting/visualize_smoothing_comparison.py" \
        --processed_demo "$PD"

    EXPECTED=$(
      ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=nb_read_frames -of default=nw=1:nk=1 \
        "$PD/video_L.mp4"
    )
    for OUTPUT in \
      "$BG_ON" \
      "$FINAL_ON" \
      "$PD/sam2_smoothing_comparison.mp4"; do
        FRAMES=$(
          ffprobe -v error -select_streams v:0 -count_frames \
            -show_entries stream=nb_read_frames -of default=nw=1:nk=1 \
            "$OUTPUT"
        )
        if [ "$FRAMES" -ne "$EXPECTED" ]; then
            echo "[$ID] frame mismatch: $OUTPUT expected=$EXPECTED got=$FRAMES" >&2
            exit 1
        fi
    done

    echo "[$ID] smoothing stages complete (${EXPECTED} frames)"
    echo "  on/off comparison: $PD/sam2_smoothing_comparison.mp4"
done

RECORDING_GLOB=$(IFS=,; echo "$*")
conda run -n inpaint-gpu --no-capture-output \
  python "$ROOT/src/inpainting/visualize_required_pipeline.py" \
    --data_root "$DATA" \
    --recording_glob "$RECORDING_GLOB"

for ID in "$@"; do
    PD="$DATA/$ID/inpainting_processed/$ID/0"
    GRID_ON="$PD/pipeline_required_components.mp4"
    EXPECTED=$(
      ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=nb_read_frames -of default=nw=1:nk=1 \
        "$PD/video_L.mp4"
    )
    GRID_FRAMES=$(
      ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=nb_read_frames -of default=nw=1:nk=1 \
        "$GRID_ON"
    )
    if [ "$GRID_FRAMES" -ne "$EXPECTED" ]; then
        echo "[$ID] grid mismatch: expected=$EXPECTED got=$GRID_FRAMES" >&2
        exit 1
    fi
    echo "[$ID] complete (${EXPECTED} frames)"
    echo "  smooth pipeline: $GRID_ON"
    echo "  on/off comparison: $PD/sam2_smoothing_comparison.mp4"
done
