#!/usr/bin/env bash
set -euo pipefail

# Build the dual-view SAM visibility masks and MH-only final-view assets used
# by composite_rb5_stereo_occlusion.py.
#
# Required first: correct phone-intrinsic HaWoR outputs for both views.
#
# Usage:
#   SH_FOCAL=<iphone13_fx> MH_FOCAL=<iphone17_fx> \
#     bash scripts/run_0804_stereo_visibility_assets.sh 1
#   SH_FOCAL=<iphone13_fx> MH_FOCAL=<iphone17_fx> ALL=1 \
#     bash scripts/run_0804_stereo_visibility_assets.sh
#
# STAGES defaults to masks,object,inpaint.  `masks` runs on SH and MH;
# `object` and `inpaint` run only on MH.  FORCE=1 explicitly regenerates the
# selected outputs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA="${DATA:-$ROOT/data/cube_dataset/26.08.04_stereo}"
STAGES="${STAGES:-masks,object,inpaint}"
ALL="${ALL:-0}"
FORCE="${FORCE:-0}"
SH_FOCAL="${SH_FOCAL:-}"
MH_FOCAL="${MH_FOCAL:-}"
INPAINT_ENV="${INPAINT_ENV:-inpaint-gpu}"

cd "$ROOT"

stage_enabled() {
    case ",$STAGES," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

IFS=',' read -r -a REQUESTED_STAGES <<< "$STAGES"
if [ "${#REQUESTED_STAGES[@]}" -eq 0 ]; then
    echo "STAGES must contain at least one stage" >&2
    exit 1
fi
for STAGE in "${REQUESTED_STAGES[@]}"; do
    case "$STAGE" in
        masks|object|inpaint) ;;
        *)
            echo "Unknown stage '$STAGE'; allowed: masks,object,inpaint" >&2
            exit 1
            ;;
    esac
done
if [ -z "$SH_FOCAL" ] || [ -z "$MH_FOCAL" ]; then
    echo "Set SH_FOCAL and MH_FOCAL to the calibrated phone pixel focal lengths." >&2
    exit 1
fi
python -c '
import math, sys
if any(not math.isfinite(float(v)) or float(v) <= 0 for v in sys.argv[1:]):
    raise SystemExit("focal lengths must be finite and positive")
' "$SH_FOCAL" "$MH_FOCAL"

if [ "$#" -gt 0 ]; then
    EPISODES=("$@")
elif [ "$ALL" = "1" ]; then
    mapfile -t CANDIDATES < <(
        find "$DATA" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -V
    )
    EPISODES=()
    for CANDIDATE in "${CANDIDATES[@]}"; do
        [[ "$CANDIDATE" =~ ^[0-9]+$ ]] && EPISODES+=("$CANDIDATE")
    done
else
    EPISODES=("1")
fi
if [ "${#EPISODES[@]}" -eq 0 ]; then
    echo "No numeric episodes found under $DATA" >&2
    exit 1
fi
for ID in "${EPISODES[@]}"; do
    [[ "$ID" =~ ^[0-9]+$ ]] || {
        echo "Episode names must be numeric, got '$ID'" >&2
        exit 1
    }
done

if stage_enabled masks || stage_enabled object || stage_enabled inpaint; then
    conda run -n "$INPAINT_ENV" python -c \
        'import torch, cv2, numpy, mediapy; assert torch.cuda.is_available()'
fi

validate_hawor() {
    local npz_path="$1"
    local expected="$2"
    local focal="$3"
    conda run -n "$INPAINT_ENV" python -c '
import math, numpy as np, sys
path, expected, focal = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
with np.load(path) as data:
    shapes = {
        "joints_left": (expected, 21, 3),
        "joints_right": (expected, 21, 3),
        "verts_left": (expected, 778, 3),
        "verts_right": (expected, 778, 3),
        "valid": (2, expected),
    }
    missing = sorted(set(shapes) - set(data.files))
    if missing:
        raise SystemExit(f"HaWoR missing keys: {missing}")
    bad = {k: data[k].shape for k, shape in shapes.items() if data[k].shape != shape}
    if bad:
        raise SystemExit(f"invalid HaWoR shapes: {bad}")
    if not bool(np.asarray(data["frame_is_cam_space"]).item()):
        raise SystemExit("HaWoR must be camera-space (--skip_slam)")
    stored = float(np.asarray(data["img_focal"]).item())
    if not math.isclose(stored, focal, rel_tol=0.0, abs_tol=1e-3):
        raise SystemExit(f"HaWoR focal {stored} != requested {focal}")
' "$npz_path" "$expected" "$focal"
}

validate_mask() {
    local mask_path="$1"
    local expected="$2"
    conda run -n "$INPAINT_ENV" python -c '
import numpy as np, sys
path, expected = sys.argv[1], int(sys.argv[2])
mask = np.load(path, mmap_mode="r")
if mask.ndim != 3 or len(mask) != expected or mask.shape[1] <= 0 or mask.shape[2] <= 0:
    raise SystemExit(f"invalid mask shape: {mask.shape}")
' "$mask_path" "$expected"
}

for ID in "${EPISODES[@]}"; do
    EP="$DATA/$ID"
    GT="$EP/gt_labels.json"
    test -s "$GT"
    EXPECTED=$(python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["num_frames"])' "$GT")
    echo
    echo "[$ID] visibility assets for $EXPECTED frames"

    for CAMERA in camera_1 camera_2; do
        if [ "$CAMERA" = "camera_1" ] && ! stage_enabled masks; then
            continue
        fi
        VIEW="$EP/$CAMERA"
        RGB="$VIEW/rgb"
        NPZ="$VIEW/rgb_hawor/retarget_input.npz"
        if [ "$CAMERA" = "camera_1" ]; then
            ROLE="SH auxiliary"
            FOCAL="$SH_FOCAL"
        else
            ROLE="MH primary"
            FOCAL="$MH_FOCAL"
        fi
        test -s "$NPZ"
        validate_hawor "$NPZ" "$EXPECTED" "$FOCAL"

        RAW_ROOT="$VIEW/visibility/raw"
        PROCESSED_ROOT="$VIEW/visibility/processed"
        PD="$PROCESSED_ROOT/view/0"
        VIDEO="$PD/video_L.mp4"
        VISIBLE_MASK="$PD/segmentation_processor/masks_arm.npy"

        if [ "$FORCE" = "1" ] || [ ! -s "$VIDEO" ]; then
            PREPARE_ARGS=(
                --input "$RGB"
                --data_root "$RAW_ROOT"
                --processed_root "$PROCESSED_ROOT"
                --demo_name view
                --demo_num 0
                --fps 24
                --glob 'rgb_frame*.jpg'
            )
            if [ "$FORCE" = "1" ]; then
                PREPARE_ARGS+=(--overwrite)
            fi
            conda run -n "$INPAINT_ENV" --no-capture-output \
                python "$ROOT/src/inpainting/prepare_demo.py" "${PREPARE_ARGS[@]}"
        fi
        VIDEO_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames \
            -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$VIDEO")
        if [ "$VIDEO_FRAMES" -ne "$EXPECTED" ]; then
            echo "[$ID/$CAMERA] prepared video count $VIDEO_FRAMES != $EXPECTED" >&2
            exit 1
        fi

        if [ "$FORCE" = "1" ] || [ "$NPZ" -nt "$PD/hand_processor/hand_data_left.npz" ] || \
           [ ! -s "$PD/hand_processor/hand_data_left.npz" ] || \
           [ ! -s "$PD/hand_processor/hand_data_right.npz" ]; then
            INJECT_ARGS=(--processed_demo "$PD" --hawor_npz "$NPZ")
            if ! stage_enabled inpaint; then
                INJECT_ARGS+=(--skip_rgb_copy)
            fi
            if [ "$FORCE" = "1" ] || \
               [ "$VIDEO" -nt "$PD/video_rgb_imgs.mkv" ]; then
                INJECT_ARGS+=(--overwrite)
            fi
            conda run -n "$INPAINT_ENV" --no-capture-output \
                python "$ROOT/src/inpainting/inject_hawor_data.py" "${INJECT_ARGS[@]}"
        fi

        if stage_enabled masks; then
            if [ "$FORCE" = "1" ] || [ ! -s "$VISIBLE_MASK" ] || \
               [ "$NPZ" -nt "$VISIBLE_MASK" ] || [ "$VIDEO" -nt "$VISIBLE_MASK" ]; then
                echo "[$ID/$CAMERA] SAM2 hand+arm mask ($ROLE)"
                MPLCONFIGDIR=/tmp/inpaint-mpl \
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                conda run -n "$INPAINT_ENV" --no-capture-output \
                    python "$ROOT/src/inpainting/segment_arms.py" \
                    --processed_demo "$PD"
            fi
            validate_mask "$VISIBLE_MASK" "$EXPECTED"
        fi
    done

    MH_PD="$EP/camera_2/visibility/processed/view/0"
    MH_VISIBLE_MASK="$MH_PD/segmentation_processor/masks_arm.npy"
    OBJECT_MASK="$MH_PD/object_layer/object_mask_modal.npy"
    BACKGROUND="$MH_PD/inpaint_processor/video_human_inpaint.mkv"

    if stage_enabled object; then
        if [ "$FORCE" = "1" ] || [ ! -s "$OBJECT_MASK" ] || \
           [ "$GT" -nt "$OBJECT_MASK" ] || \
           [ "$EP/camera_2/rgb_hawor/retarget_input.npz" -nt "$OBJECT_MASK" ] || \
           [ "$MH_PD/video_L.mp4" -nt "$OBJECT_MASK" ]; then
            echo "[$ID/camera_2] SAM2 annotated-object mask (MH only)"
            MPLCONFIGDIR=/tmp/inpaint-mpl \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            conda run -n "$INPAINT_ENV" --no-capture-output \
                python "$ROOT/src/inpainting/segment_annotated_objects.py" \
                --processed_demo "$MH_PD" \
                --labels_json "$GT"
        fi
        validate_mask "$OBJECT_MASK" "$EXPECTED"
    fi

    if stage_enabled inpaint; then
        test -s "$MH_VISIBLE_MASK"
        test -s "$OBJECT_MASK"
        MH_NPZ="$EP/camera_2/rgb_hawor/retarget_input.npz"
        if [ "$MH_NPZ" -nt "$MH_VISIBLE_MASK" ] || \
           [ "$MH_NPZ" -nt "$OBJECT_MASK" ] || \
           [ "$GT" -nt "$OBJECT_MASK" ]; then
            echo "[$ID/camera_2] upstream masks are stale; run STAGES=masks,object first" >&2
            exit 1
        fi
        if [ "$FORCE" = "1" ] || [ ! -s "$BACKGROUND" ] || \
           [ "$MH_VISIBLE_MASK" -nt "$BACKGROUND" ] || \
           [ "$OBJECT_MASK" -nt "$BACKGROUND" ] || \
           [ "$MH_PD/video_rgb_imgs.mkv" -nt "$BACKGROUND" ]; then
            echo "[$ID/camera_2] MH human removal with object protection"
            MPLCONFIGDIR=/tmp/inpaint-mpl \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            conda run -n "$INPAINT_ENV" --no-capture-output \
                python "$ROOT/src/inpainting/inpaint_hands.py" \
                --processed_demo "$MH_PD" \
                --mode legacy \
                --protect_mask "$OBJECT_MASK" \
                --output_resolution 540 \
                --dilate_iter 4 \
                --fps 24
        fi
        test -s "$BACKGROUND"
    fi

    echo "[$ID] SH visible mask: $EP/camera_1/visibility/processed/view/0/segmentation_processor/masks_arm.npy"
    echo "[$ID] MH visible mask: $MH_VISIBLE_MASK"
    echo "[$ID] MH object mask:  $OBJECT_MASK"
    echo "[$ID] MH background:   $BACKGROUND"
done

echo
echo "Completed stages '$STAGES' for: ${EPISODES[*]}"
