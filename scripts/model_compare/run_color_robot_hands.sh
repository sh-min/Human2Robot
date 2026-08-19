#!/usr/bin/env bash
set -euo pipefail

# Apply the GitHub layered-overlay pipeline to the newly added color*.mp4
# videos without modifying their source files.  With no arguments the shortest
# color.mp4 pilot is processed.  Pass --all for every unique color*.mp4, or
# pass explicit video paths.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/output/model_compare/color_robot_hands}"
FOCAL="${FOCAL:-600}"
FPS="${FPS:-30}"
MASK_DILATE="${MASK_DILATE:-25}"

declare -a INPUTS=()
if [[ "${1:-}" == "--all" ]]; then
    while IFS= read -r -d '' video; do
        INPUTS+=("$video")
    done < <(find "$ROOT" -maxdepth 1 -type f -iname 'color*.mp4' -print0 | sort -z)
    shift
elif (( $# > 0 )); then
    for video in "$@"; do
        INPUTS+=("$(realpath "$video")")
    done
else
    INPUTS+=("$ROOT/color.mp4")
fi

mkdir -p "$OUTPUT_ROOT"
declare -A SEEN_HASHES=()

for video in "${INPUTS[@]}"; do
    test -s "$video"
    digest="$(sha256sum "$video" | awk '{print $1}')"
    if [[ -n "${SEEN_HASHES[$digest]:-}" ]]; then
        echo "[duplicate] $video == ${SEEN_HASHES[$digest]}"
        continue
    fi
    SEEN_HASHES[$digest]="$video"

    filename="$(basename "$video")"
    stem="${filename%.*}"
    slug="$(printf '%s' "$stem" | sed -E 's/[^[:alnum:]]+/_/g; s/^_+|_+$//g' | tr '[:upper:]' '[:lower:]')"
    episode="$OUTPUT_ROOT/$slug"
    rgb="$episode/rgb"
    hawor="$episode/rgb_hawor"
    processed_root="$episode/processed"
    processed_demo="$processed_root/$slug/0"
    npz="$hawor/retarget_input.npz"
    left_pkl="$hawor/qpos_xhand_left_smooth.pkl"
    right_pkl="$hawor/qpos_xhand_right_smooth.pkl"
    final="$processed_demo/overlay_processor_layered/video_overlay.mp4"

    mkdir -p "$rgb" "$hawor" "$processed_root"
    frame_count="$(ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$video")"

    rgb_count="$(find "$rgb" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
    if [[ "$rgb_count" -ne "$frame_count" ]]; then
        find "$rgb" -maxdepth 1 -type f -name '*.jpg' -delete
        ffmpeg -nostdin -y -hide_banner -loglevel error \
            -i "$video" -q:v 2 "$rgb/%06d.jpg"
    fi

    npz_count=0
    if [[ -s "$npz" ]]; then
        npz_count="$(conda run -n hawor python -c \
            "import numpy as np; print(np.load(r'$npz')['joints_left'].shape[0])" \
            | tr -d '[:space:]')"
    fi
    if [[ "$npz_count" -ne "$frame_count" ]]; then
        echo "[$slug] HaWoR bimanual estimation"
        MPLCONFIGDIR=/tmp/color-hawor-mpl \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        conda run -n hawor --no-capture-output \
            python "$ROOT/src/hand_estimation/extract_for_retarget.py" \
            --rgb_dir "$rgb" \
            --img_glob '*.jpg' \
            --workdir "$hawor" \
            --img_focal "$FOCAL" \
            --skip_slam \
            --vts_proj
    fi
    test -s "$npz"

    if [[ ! -s "$left_pkl" || ! -s "$right_pkl" ]]; then
        echo "[$slug] XHand retargeting"
        (
            cd "$ROOT/src/retargeting"
            conda run -n RFM_retarget --no-capture-output \
                python retarget_from_npz.py \
                --npz "$npz" \
                --hand both \
                --smooth
        )
    fi
    test -s "$left_pkl"
    test -s "$right_pkl"

    if [[ ! -s "$processed_demo/video_L.mp4" ]]; then
        echo "[$slug] prepare demo"
        MPLCONFIGDIR=/tmp/color-inpaint-mpl \
        conda run -n inpaint-gpu --no-capture-output \
            python "$ROOT/src/inpainting/prepare_demo.py" \
            --input "$rgb" \
            --data_root "$episode/raw" \
            --processed_root "$processed_root" \
            --demo_name "$slug" \
            --demo_num 0 \
            --fps "$FPS" \
            --glob '*.jpg' \
            --overwrite
    fi

    if [[ ! -s "$processed_demo/bbox_processor/bbox_data.npz" ]]; then
        echo "[$slug] inject HaWoR prompts"
        conda run -n inpaint-gpu --no-capture-output \
            python "$ROOT/src/inpainting/inject_hawor_data.py" \
            --processed_demo "$processed_demo" \
            --hawor_npz "$npz" \
            --overwrite
    fi

    if [[ ! -s "$processed_demo/segmentation_processor/masks_arm.npy" ]]; then
        echo "[$slug] SAM2 hand/arm mask"
        MPLCONFIGDIR=/tmp/color-inpaint-mpl \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        conda run -n inpaint-gpu --no-capture-output \
            python "$ROOT/src/inpainting/segment_arms.py" \
            --processed_demo "$processed_demo"
    fi

    static_bg="$processed_demo/inpaint_processor/video_human_inpaint_static.mkv"
    if [[ ! -s "$static_bg" ]]; then
        echo "[$slug] fixed-camera temporal human-arm removal"
        conda run -n inpaint-gpu --no-capture-output \
            python "$ROOT/scripts/model_compare/inpaint_static_color_background.py" \
            --video "$processed_demo/video_L.mp4" \
            --mask "$processed_demo/segmentation_processor/masks_arm.npy" \
            --output "$static_bg" \
            --mask-dilate "$MASK_DILATE" \
            --feather-sigma 2.0
    fi

    if [[ ! -s "$processed_demo/overlay_processor/robot_mask.npy" ]]; then
        echo "[$slug] RBY1/XHand RGBD render"
        PYOPENGL_PLATFORM=egl \
        MPLCONFIGDIR=/tmp/color-inpaint-mpl \
        conda run -n inpaint-gpu --no-capture-output \
            python -u "$ROOT/src/inpainting/render_xhand_overlay_depth.py" \
            --processed_demo "$processed_demo" \
            --hawor_npz "$npz" \
            --right_pkl "$right_pkl" \
            --left_pkl "$left_pkl" \
            --hand both \
            --smooth \
            --relight auto
    fi

    if [[ ! -s "$final" || "$static_bg" -nt "$final" ]]; then
        echo "[$slug] GitHub layered composite"
        conda run -n inpaint-gpu --no-capture-output \
            python "$ROOT/src/inpainting/composite_layered.py" \
            --processed_demo "$processed_demo" \
            --hawor_npz "$npz" \
            --bg_video inpaint_processor/video_human_inpaint_static.mkv \
            --object_mask_npy object_layer/object_mask_amodal.npy \
            --fps "$FPS" \
            --no_object
    fi
    test -s "$final"

    ffmpeg -nostdin -y -hide_banner -loglevel error \
        -i "$final" -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart \
        "$episode/${slug}_robot_hands.mp4"
    echo "[done] $episode/${slug}_robot_hands.mp4"
done
