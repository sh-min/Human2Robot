#!/usr/bin/env bash
set -euo pipefail

# Full replacement pipeline:
#   human hand+arm segmentation -> background inpainting
#   -> complete RBY1 arm + XHand render -> unclipped final composite
#
# Usage:
#   bash scripts/run_all_rby1_xhand_overlay.sh              # first episode only
#   FORCE=1 bash scripts/run_all_rby1_xhand_overlay.sh IMG_5019 IMG_5020
#   DATA=/path/to/data ALL=1 bash scripts/run_all_rby1_xhand_overlay.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA="${DATA:-$ROOT/data/cube_dataset/26.07.24}"
FORCE="${FORCE:-0}"
ALL="${ALL:-0}"

cd "$ROOT"

conda run -n RFM_retarget python -c \
  'import torch, dex_retargeting, pinocchio, scipy, trimesh'
conda run -n inpaint-gpu python -c \
  'import torch, cv2, mediapy, pyrender, mmcv, pinocchio'

if [ "$#" -gt 0 ]; then
    EPISODE_IDS=("$@")
elif [ "$ALL" = "1" ]; then
    EPISODE_IDS=()
    for EP in "$DATA"/*; do
        [ -d "$EP" ] || continue
        EPISODE_IDS+=("$(basename "$EP")")
    done
else
    EPISODE_IDS=()
    for EP in "$DATA"/*; do
        [ -d "$EP" ] || continue
        EPISODE_IDS+=("$(basename "$EP")")
        break
    done
fi

if [ "${#EPISODE_IDS[@]}" -eq 0 ]; then
    echo "처리할 episode가 없습니다: $DATA" >&2
    exit 1
fi

for ID in "${EPISODE_IDS[@]}"; do
    EP="$DATA/$ID"
    RGB="$EP/rgb"
    VIDEO=""
    for CANDIDATE in \
      "$EP/$ID.MOV" \
      "$EP/$ID.mov" \
      "$EP/$ID.MP4" \
      "$EP/$ID.mp4"; do
        if [ -s "$CANDIDATE" ]; then
            VIDEO="$CANDIDATE"
            break
        fi
    done
    if [ -z "$VIDEO" ]; then
        echo "[$ID] 원본 MOV/MP4를 찾지 못했습니다: $EP" >&2
        exit 1
    fi
    HAWOR="$EP/rgb_hawor"
    NPZ="$HAWOR/retarget_input.npz"

    RAW_ROOT="$EP/inpainting_raw"
    PROC_ROOT="$EP/inpainting_processed"
    PD="$PROC_ROOT/$ID/0"

    LEFT_PKL="$HAWOR/qpos_xhand_left_smooth.pkl"
    RIGHT_PKL="$HAWOR/qpos_xhand_right_smooth.pkl"
    MASK="$PD/segmentation_processor/masks_arm.npy"
    BG_MKV="$PD/inpaint_processor/video_human_inpaint.mkv"
    FINAL_MKV="$PD/video_overlay_rby1_xhand.mkv"
    FINAL_MP4="$PD/video_overlay_rby1_xhand.mp4"
    GRID_MP4="$PD/pipeline_components_rby1_xhand.mp4"

    echo
    echo "========================================"
    echo "처리: $ID"
    echo "========================================"

    mkdir -p "$RGB"
    VIDEO_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames \
      -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$VIDEO")
    RGB_FRAMES=$(find "$RGB" -maxdepth 1 -type f -name '*.jpg' | wc -l)
    if [ "$RGB_FRAMES" -ne "$VIDEO_FRAMES" ]; then
        echo "[$ID] RGB 재추출: ${RGB_FRAMES} -> ${VIDEO_FRAMES}프레임"
        ffmpeg -nostdin -y -hide_banner -loglevel error \
          -i "$VIDEO" -q:v 2 "$RGB/%06d.jpg"
        RGB_FRAMES=$(find "$RGB" -maxdepth 1 -type f -name '*.jpg' | wc -l)
    fi
    test "$RGB_FRAMES" -eq "$VIDEO_FRAMES"

    NPZ_FRAMES=0
    if [ -s "$NPZ" ]; then
        NPZ_FRAMES=$(conda run -n hawor python -c \
          "import numpy as np; print(np.load('$NPZ')['joints_left'].shape[0])" \
          | tr -d '[:space:]')
    fi
    if [ "$NPZ_FRAMES" -ne "$RGB_FRAMES" ]; then
        if [ -d "$HAWOR" ]; then
            BACKUP="$EP/rgb_hawor_incomplete_${NPZ_FRAMES}f"
            if [ ! -e "$BACKUP" ]; then
                echo "[$ID] 불완전 HaWoR 결과 백업: $BACKUP"
                mv "$HAWOR" "$BACKUP"
            else
                echo "[$ID] 기존 백업 사용: $BACKUP"
                mv "$HAWOR" "$EP/rgb_hawor_incomplete_${NPZ_FRAMES}f_$(date +%s)"
            fi
        fi
        echo "[$ID] HaWoR 재실행"
        MPLCONFIGDIR=/tmp/hawor-mpl \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        conda run -n hawor --no-capture-output \
          python "$ROOT/src/hand_estimation/extract_for_retarget.py" \
          --rgb_dir "$RGB" \
          --img_glob '*.jpg' \
          --img_focal 1220 \
          --skip_slam \
          --vts_proj
    fi
    test -s "$NPZ"

    PKL_FRAMES=0
    if [ -s "$LEFT_PKL" ] && [ -s "$RIGHT_PKL" ]; then
        PKL_FRAMES=$(conda run -n RFM_retarget python -c \
          "import pickle; print(len(pickle.load(open('$LEFT_PKL','rb'))['data']))" \
          | tr -d '[:space:]')
    fi
    if [ "$PKL_FRAMES" -ne "$RGB_FRAMES" ]; then
        echo "[$ID] XHand retargeting"
        (
          cd "$ROOT/src/retargeting"
          conda run -n RFM_retarget --no-capture-output \
            python retarget_from_npz.py \
            --npz "$NPZ" \
            --hand both \
            --smooth
        )
    fi
    test -s "$LEFT_PKL"
    test -s "$RIGHT_PKL"

    MPLCONFIGDIR=/tmp/inpaint-mpl \
    conda run -n inpaint-gpu --no-capture-output \
      python "$ROOT/src/inpainting/prepare_demo.py" \
      --input "$RGB" \
      --data_root "$RAW_ROOT" \
      --processed_root "$PROC_ROOT" \
      --demo_name "$ID" \
      --demo_num 0 \
      --fps 30 \
      --glob '*.jpg' \
      --overwrite

    MPLCONFIGDIR=/tmp/inpaint-mpl \
    conda run -n inpaint-gpu --no-capture-output \
      python "$ROOT/src/inpainting/inject_hawor_data.py" \
      --processed_demo "$PD" \
      --hawor_npz "$NPZ" \
      --overwrite

    if [ "$FORCE" = "1" ] || [ ! -s "$MASK" ]; then
        echo "[$ID] SAM2 손+팔 전체 마스크"
        MPLCONFIGDIR=/tmp/inpaint-mpl \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        conda run -n inpaint-gpu --no-capture-output \
          python "$ROOT/src/inpainting/segment_arms.py" \
          --processed_demo "$PD"
    else
        echo "[$ID] 손+팔 마스크 건너뜀"
    fi
    test -s "$MASK"

    if [ "$FORCE" = "1" ] || [ ! -s "$BG_MKV" ]; then
        echo "[$ID] 손+팔 제거 후 배경 인페인팅"
        (
          cd "$ROOT/src/inpainting"
          MPLCONFIGDIR=/tmp/inpaint-mpl \
          PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          conda run -n inpaint-gpu --no-capture-output \
            python inpaint_hands.py \
            --processed_demo "$PD" \
            --mode legacy \
            --output_resolution 540 \
            --dilate_iter 4 \
            --fps 30
        )
    else
        echo "[$ID] 배경 인페인팅 건너뜀"
    fi
    test -s "$BG_MKV"

    if [ "$FORCE" = "1" ] || [ ! -s "$FINAL_MKV" ]; then
        echo "[$ID] RBY1 팔 + XHand 렌더 및 마스크 독립 합성"
        (
          cd "$ROOT/src/inpainting"
          PYOPENGL_PLATFORM=egl \
          MPLCONFIGDIR=/tmp/inpaint-mpl \
          conda run -n inpaint-gpu --no-capture-output \
            python -u render_rby1_xhand_full_arm.py \
            --processed_demo "$PD" \
            --hawor_npz "$NPZ" \
            --right_pkl "$RIGHT_PKL" \
            --left_pkl "$LEFT_PKL" \
            --hand both \
            --require_smoothed \
            --fps 30
        )
    else
        echo "[$ID] 로봇팔 합성 건너뜀"
    fi
    test -s "$FINAL_MKV"

    ffmpeg -nostdin -y -hide_banner -loglevel error \
      -i "$FINAL_MKV" \
      -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' \
      -c:v libx264 -crf 18 -pix_fmt yuv420p \
      "$FINAL_MP4"
    test -s "$FINAL_MP4"

    MPLCONFIGDIR=/tmp/inpaint-mpl \
    conda run -n inpaint-gpu --no-capture-output \
      python "$ROOT/src/inpainting/visualize_pipeline_grid.py" \
      --processed_demo "$PD" \
      --out "$GRID_MP4"
    test -s "$GRID_MP4"

    FINAL_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames \
      -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$FINAL_MP4")
    echo "[완료] $ID — ${FINAL_FRAMES}프레임"
    echo "  최종: $FINAL_MP4"
    echo "  비교: $GRID_MP4"
done

echo
echo "========================================"
echo "전체 RBY1 + XHand 처리 완료"
echo "========================================"
