#!/usr/bin/env bash
set -euo pipefail

# Prepare HaWoR and HaCo inputs for the accepted contact-conditioned robot
# replacement pipeline.  The rejected plain VECTOR visualization is not
# rendered or published here.
#
# Source videos are read-only. All frames, estimates, and comparison videos go
# under output/retargeting_eval/26.08.13_choco.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-$ROOT/data/cube_dataset/26.08.13_choco}"
MANIFEST="${MANIFEST:-$ROOT/configs/retargeting/aug13_no_pilot.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/output/retargeting_eval/26.08.13_choco}"
FOCAL="${FOCAL:-600}"

if (( $# > 0 )); then
    episodes=("$@")
else
    mapfile -t episodes < <(sed -E '/^[[:space:]]*(#|$)/d' "$MANIFEST")
fi

mkdir -p "$OUTPUT_ROOT"

for episode in "${episodes[@]}"; do
    source_dir="$DATASET/$episode"
    source_video="$source_dir/source.MOV"
    work="$OUTPUT_ROOT/$episode"
    rgb="$work/rgb"
    hawor="$work/rgb_hawor"
    contact="$work/contact"
    expected="$(python3 -c \
        "import json; print(json.load(open(r'$source_dir/gt_labels.json'))['num_frames'])")"

    test -s "$source_video"
    mkdir -p "$rgb" "$hawor" "$contact"

    rgb_count="$(find "$rgb" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
    if [[ "$rgb_count" -eq 0 ]]; then
        echo "[$episode] extracting $expected RGB frames"
        ffmpeg -nostdin -y -hide_banner -loglevel error \
            -i "$source_video" -q:v 2 "$rgb/%06d.jpg"
        rgb_count="$(find "$rgb" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
    fi
    if [[ "$rgb_count" -ne "$expected" ]]; then
        echo "[$episode] RGB frame mismatch: $rgb_count != $expected" >&2
        exit 1
    fi

    if [[ ! -s "$hawor/retarget_input.npz" ]]; then
        echo "[$episode] HaWoR"
        MPLCONFIGDIR=/tmp/aug13-hawor-mpl \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        conda run -n hawor --no-capture-output \
            python "$ROOT/src/hand_estimation/extract_for_retarget.py" \
            --rgb_dir "$rgb" \
            --img_glob '*.jpg' \
            --workdir "$hawor" \
            --img_focal "$FOCAL" \
            --skip_slam
    fi

    if [[ ! -s "$hawor/qpos_xhand_right_smooth.pkl" || \
          ! -s "$hawor/qpos_xhand_left_smooth.pkl" ]]; then
        echo "[$episode] DexPilot initializer for contact retargeting"
        (
            cd "$ROOT/src/retargeting"
            conda run -n RFM_retarget --no-capture-output \
                python retarget_from_npz.py \
                --npz "$hawor/retarget_input.npz" \
                --hand both --smooth
        )
    fi

    contact_count="$(find "$contact" -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l)"
    if [[ "$contact_count" -eq 0 ]]; then
        echo "[$episode] HACO contact"
        conda run -n haco --no-capture-output \
            python "$ROOT/src/contact_estimation/extract_hand_contact.py" \
            --input_dir "$work" --img_glob '*.jpg' --no_viz
        contact_count="$(find "$contact" -maxdepth 1 -type f -name '*.npz' | wc -l)"
    fi
    if [[ "$contact_count" -ne "$expected" ]]; then
        echo "[$episode] contact frame mismatch: $contact_count != $expected" >&2
        exit 1
    fi

    if [[ ! -s "$hawor/qpos_xhand_contact_right_smooth.pkl" || \
          ! -s "$hawor/qpos_xhand_contact_left_smooth.pkl" ]]; then
        echo "[$episode] contact-aware refinement"
        (
            cd "$ROOT/src/retargeting"
            conda run -n RFM_retarget --no-capture-output \
                python retarget_from_npz.py \
                --npz "$hawor/retarget_input.npz" \
                --contact --contact_dir "$contact" \
                --hand both --smooth
        )
    fi

    echo "[done] $episode contact inputs"
done
