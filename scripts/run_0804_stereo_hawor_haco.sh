#!/usr/bin/env bash
set -euo pipefail

# Resumable dual-view hand/contact extraction for the 08_04 dataset.
#
# Roles are deliberately fixed to match composite_rb5_stereo_occlusion.py:
#   camera_1 = SH auxiliary evidence
#   camera_2 = MH primary/final view
#
# Usage:
#   SH_FOCAL=<iphone13_fx> MH_FOCAL=<iphone17_fx> \
#     bash scripts/run_0804_stereo_hawor_haco.sh 1
#   FOCAL_MODE=approx_26mm VIEWS=mh \
#     bash scripts/run_0804_stereo_hawor_haco.sh 1
#   SH_FOCAL=<iphone13_fx> MH_FOCAL=<iphone17_fx> \
#     STAGES=hawor,haco ALL=1 bash scripts/run_0804_stereo_hawor_haco.sh
#   STAGES=vjepa VJEPA_CKPT=/path/to/vitl.pt ALL=1 bash ...

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA="${DATA:-$ROOT/data/cube_dataset/26.08.04_stereo}"
STAGES="${STAGES:-hawor,haco}"
ALL="${ALL:-0}"
HACO_VIZ="${HACO_VIZ:-0}"
VJEPA_CKPT="${VJEPA_CKPT:-$ROOT/weights/vjepa2/vitl.pt}"
VJEPA_OVERWRITE="${VJEPA_OVERWRITE:-0}"
VJEPA_ACTION_LABELS="${VJEPA_ACTION_LABELS:-}"
SH_FOCAL="${SH_FOCAL:-}"
MH_FOCAL="${MH_FOCAL:-}"
FOCAL_MODE="${FOCAL_MODE:-calibrated}"
VIEWS="${VIEWS:-both}"
APPROX_FOCAL_PX="${APPROX_FOCAL_PX:-924.4444444444}"
EXPECTED_SH_LENS="${EXPECTED_SH_LENS:-iPhone 13 26mm}"
EXPECTED_MH_LENS="${EXPECTED_MH_LENS:-iPhone 17 26mm}"
EXPECTED_FRAME_SIZE="${EXPECTED_FRAME_SIZE:-1280x720}"

cd "$ROOT"

case "$VIEWS" in
    both) CAMERAS=(camera_1 camera_2) ;;
    sh) CAMERAS=(camera_1) ;;
    mh) CAMERAS=(camera_2) ;;
    *) echo "VIEWS must be both, sh, or mh" >&2; exit 1 ;;
esac
case "$FOCAL_MODE" in
    calibrated) ;;
    approx_26mm)
        SH_FOCAL="${SH_FOCAL:-$APPROX_FOCAL_PX}"
        MH_FOCAL="${MH_FOCAL:-$APPROX_FOCAL_PX}"
        echo "WARNING: using approximate 26mm-equivalent focal " \
            "fx=$APPROX_FOCAL_PX px for a 1280px-wide frame."
        echo "This mode is for contact/overlay pilots, not metric geometry."
        ;;
    *) echo "FOCAL_MODE must be calibrated or approx_26mm" >&2; exit 1 ;;
esac

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
        hawor|haco|vjepa|preflight) ;;
        *)
            echo "Unknown stage '$STAGE'; allowed: hawor,haco,vjepa,preflight" >&2
            exit 1
            ;;
    esac
done

if stage_enabled hawor || stage_enabled haco; then
    if { [ "$VIEWS" = "both" ] || [ "$VIEWS" = "sh" ]; } && \
       [ -z "$SH_FOCAL" ]; then
        echo "SH processing requires SH_FOCAL, or FOCAL_MODE=approx_26mm." >&2
        exit 1
    fi
    if { [ "$VIEWS" = "both" ] || [ "$VIEWS" = "mh" ]; } && \
       [ -z "$MH_FOCAL" ]; then
        echo "MH processing requires MH_FOCAL, or FOCAL_MODE=approx_26mm." >&2
        exit 1
    fi
    FOCAL_VALUES=()
    if [ "$VIEWS" = "both" ] || [ "$VIEWS" = "sh" ]; then
        FOCAL_VALUES+=("$SH_FOCAL")
    fi
    if [ "$VIEWS" = "both" ] || [ "$VIEWS" = "mh" ]; then
        FOCAL_VALUES+=("$MH_FOCAL")
    fi
    if [ "$FOCAL_MODE" = "calibrated" ]; then
        echo "Using explicitly supplied phone focal values."
    else
        echo "Approximate focal is explicitly enabled; calibration is deferred."
    fi
    if [ "${#FOCAL_VALUES[@]}" -eq 0 ]; then
        echo "No focal value selected for VIEWS=$VIEWS" >&2
        exit 1
    fi
    python -c '
import math, sys
values = [float(value) for value in sys.argv[1:]]
if any(not math.isfinite(value) or value <= 0 for value in values):
    raise SystemExit("camera focal lengths must be finite and positive")
' "${FOCAL_VALUES[@]}"
    if [ "$FOCAL_MODE" = "calibrated" ] && \
       { [ -z "$SH_FOCAL" ] && [ -z "$MH_FOCAL" ]; }; then
        echo "Set the selected view focal value(s)." >&2
        echo "Do not use 08_04/calibration/realsense_checkerboard_not_iphone.json." >&2
        exit 1
    fi
    echo "Camera mapping: SH=camera_1 fx=${SH_FOCAL:-disabled}, " \
        "MH=camera_2 fx=${MH_FOCAL:-disabled}; selected=$VIEWS"
fi

focals_match() {
    python -c '
import math, sys
raise SystemExit(0 if math.isclose(float(sys.argv[1]), float(sys.argv[2]), rel_tol=0.0, abs_tol=1e-3) else 1)
' "$1" "$2"
}

load_hawor_info() {
    local npz_path="$1"
    local conda_env="$2"
    local expected_frames="$3"
    local output
    if ! output=$(conda run -n "$conda_env" python -c '
import math, numpy as np, sys
path, expected = sys.argv[1], int(sys.argv[2])
with np.load(path) as data:
    required = {
        "joints_left": (expected, 21, 3),
        "joints_right": (expected, 21, 3),
        "verts_left": (expected, 778, 3),
        "verts_right": (expected, 778, 3),
        "mano_trans": (2, expected, 3),
        "mano_global_orient": (2, expected, 3),
        "mano_hand_pose": (2, expected, 15, 3),
        "mano_betas": (2, expected, 10),
        "valid": (2, expected),
    }
    missing = sorted((set(required) | {"img_focal", "start_idx", "end_idx", "frame_is_cam_space"}) - set(data.files))
    if missing:
        raise SystemExit(f"HaWoR NPZ missing keys: {missing}")
    bad = {key: (data[key].shape, shape) for key, shape in required.items() if data[key].shape != shape}
    if bad:
        raise SystemExit(f"HaWoR NPZ shape mismatch: {bad}")
    start = int(np.asarray(data["start_idx"]).item())
    end = int(np.asarray(data["end_idx"]).item())
    if start != 0 or end != expected:
        raise SystemExit(f"HaWoR frame range must be 0:{expected}, got {start}:{end}")
    if not bool(np.asarray(data["frame_is_cam_space"]).item()):
        raise SystemExit("HaWoR NPZ is not camera-space; rerun with --skip_slam")
    focal = float(np.asarray(data["img_focal"]).item())
    if not math.isfinite(focal) or focal <= 0:
        raise SystemExit(f"invalid HaWoR focal: {focal}")
    print(expected)
    print(focal)
' "$npz_path" "$expected_frames"); then
        return 1
    fi
    mapfile -t HAWOR_INFO <<< "$output"
    if [ "${#HAWOR_INFO[@]}" -ne 2 ]; then
        return 1
    fi
    HAWOR_COUNT="${HAWOR_INFO[0]}"
    HAWOR_FOCAL="${HAWOR_INFO[1]}"
}

validate_contact_output() {
    local contact_dir="$1"
    local rgb_dir="$2"
    local expected_frames="$3"
    local expected_focal="$4"
    local output
    if ! output=$(conda run -n haco python -c '
import math, numpy as np, sys
from pathlib import Path
contact_dir, rgb_dir = Path(sys.argv[1]), Path(sys.argv[2])
expected, target_focal = int(sys.argv[3]), float(sys.argv[4])
contacts = sorted(contact_dir.glob("rgb_frame*.npz"))
images = sorted(rgb_dir.glob("rgb_frame*.jpg"))
if len(contacts) != expected or len(images) != expected:
    raise SystemExit(f"contact/RGB count mismatch: {len(contacts)}/{len(images)} != {expected}")
for index, (contact, image) in enumerate(zip(contacts, images)):
    if contact.stem != image.stem:
        raise SystemExit(f"contact stem mismatch at {index}: {contact.name} vs {image.name}")
    with np.load(contact) as data:
        common = {"source_filename", "hawor_frame_index", "img_focal", "contact_threshold"}
        side_keys = {
            f"{side}_{suffix}"
            for side in ("left", "right")
            for suffix in ("valid", "contact_mask", "contact_probability", "contact_indices", "contact_verts_3d")
        }
        missing = sorted((common | side_keys) - set(data.files))
        if missing:
            raise SystemExit(f"{contact}: missing keys {missing}")
        if str(np.asarray(data["source_filename"]).item()) != image.name:
            raise SystemExit(f"{contact}: source_filename mismatch")
        if int(np.asarray(data["hawor_frame_index"]).item()) != index:
            raise SystemExit(f"{contact}: hawor_frame_index mismatch")
        focal = float(np.asarray(data["img_focal"]).item())
        if not math.isclose(focal, target_focal, rel_tol=0.0, abs_tol=1e-3):
            raise SystemExit(f"{contact}: focal {focal} != {target_focal}")
        for side in ("left", "right"):
            mask = data[f"{side}_contact_mask"]
            probability = data[f"{side}_contact_probability"]
            indices = data[f"{side}_contact_indices"]
            vertices = data[f"{side}_contact_verts_3d"]
            if mask.shape != (778,) or probability.shape != (778,):
                raise SystemExit(f"{contact}: invalid {side} dense contact shape")
            if indices.ndim != 1 or vertices.shape != (len(indices), 3):
                raise SystemExit(f"{contact}: invalid {side} sparse contact shape")
print(target_focal)
' "$contact_dir" "$rgb_dir" "$expected_frames" "$expected_focal"); then
        return 1
    fi
    mapfile -t CONTACT_INFO <<< "$output"
    if [ "${#CONTACT_INFO[@]}" -ne 1 ]; then
        return 1
    fi
    CONTACT_FOCAL="${CONTACT_INFO[0]}"
}

if [ "$#" -gt 0 ]; then
    EPISODES=("$@")
elif [ "$ALL" = "1" ]; then
    mapfile -t CANDIDATE_EPISODES < <(
        find "$DATA" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -V
    )
    EPISODES=()
    for CANDIDATE in "${CANDIDATE_EPISODES[@]}"; do
        if [[ "$CANDIDATE" =~ ^[0-9]+$ ]]; then
            EPISODES+=("$CANDIDATE")
        fi
    done
else
    EPISODES=("1")
fi

if [ "${#EPISODES[@]}" -eq 0 ]; then
    echo "No episodes found under $DATA" >&2
    exit 1
fi
for ID in "${EPISODES[@]}"; do
    if ! [[ "$ID" =~ ^[0-9]+$ ]]; then
        echo "Episode names must be numeric, got '$ID'" >&2
        exit 1
    fi
done

if stage_enabled hawor; then
    conda run -n hawor python -c 'import torch, cv2, numpy; assert torch.cuda.is_available()'
fi
if stage_enabled haco; then
    conda run -n haco python -c 'import torch, cv2, numpy; assert torch.cuda.is_available()'
fi
if stage_enabled vjepa; then
    if [ ! -s "$VJEPA_CKPT" ]; then
        echo "V-JEPA checkpoint missing: $VJEPA_CKPT" >&2
        exit 1
    fi
    conda run -n vjepa2-312 python -c 'import torch, numpy; assert torch.cuda.is_available()'
fi

for ID in "${EPISODES[@]}"; do
    EP="$DATA/$ID"
    test -s "$EP/gt_labels.json"
    EXPECTED=$(
        python -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_frames"])' \
            "$EP/gt_labels.json"
    )
    echo
    echo "[$ID] $EXPECTED synchronized frames"

    for CAMERA in "${CAMERAS[@]}"; do
        VIEW="$EP/$CAMERA"
        RGB="$VIEW/rgb"
        HAWOR="$VIEW/rgb_hawor"
        NPZ="$HAWOR/retarget_input.npz"
        CONTACT="$VIEW/contact"
        if [ "$CAMERA" = "camera_1" ]; then
            ROLE="SH auxiliary"
            FOCAL="$SH_FOCAL"
            EXPECTED_LENS="$EXPECTED_SH_LENS"
        else
            ROLE="MH primary"
            FOCAL="$MH_FOCAL"
            EXPECTED_LENS="$EXPECTED_MH_LENS"
        fi
        SOURCE_MOV="$VIEW/source.mov"
        test -s "$SOURCE_MOV"
        SOURCE_SIZE=$(
            ffprobe -v error -select_streams v:0 \
                -show_entries stream=width,height -of csv=p=0:s=x "$SOURCE_MOV"
        )
        SOURCE_LENS=$(
            ffprobe -v error \
                -show_entries format_tags=com.blackmagic-design.camera.lensType \
                -of default=noprint_wrappers=1:nokey=1 "$SOURCE_MOV"
        )
        if [ "$SOURCE_SIZE" != "$EXPECTED_FRAME_SIZE" ]; then
            echo "[$ID/$CAMERA] source size $SOURCE_SIZE != $EXPECTED_FRAME_SIZE" >&2
            exit 1
        fi
        if [ "$SOURCE_LENS" != "$EXPECTED_LENS" ]; then
            echo "[$ID/$CAMERA] source lens '$SOURCE_LENS' != '$EXPECTED_LENS'" >&2
            exit 1
        fi
        RGB_COUNT=$(find "$RGB" -maxdepth 1 -type f -name 'rgb_frame*.jpg' | wc -l)
        if [ "$RGB_COUNT" -ne "$EXPECTED" ]; then
            echo "[$ID/$CAMERA] RGB mismatch: $RGB_COUNT != $EXPECTED" >&2
            exit 1
        fi

        if stage_enabled hawor; then
            if [ -s "$NPZ" ]; then
                if ! load_hawor_info "$NPZ" hawor "$EXPECTED"; then
                    echo "[$ID/$CAMERA] invalid HaWoR NPZ schema/coordinates" >&2
                    exit 1
                fi
                if [ "$HAWOR_COUNT" -ne "$EXPECTED" ]; then
                    echo "[$ID/$CAMERA] incomplete HaWoR NPZ: $HAWOR_COUNT != $EXPECTED" >&2
                    echo "Move $HAWOR aside explicitly before retrying." >&2
                    exit 1
                fi
                if ! focals_match "$HAWOR_FOCAL" "$FOCAL"; then
                    echo "[$ID/$CAMERA] cached HaWoR focal $HAWOR_FOCAL != requested $FOCAL" >&2
                    echo "Move $HAWOR aside explicitly before retrying." >&2
                    exit 1
                fi
                echo "[$ID/$CAMERA] HaWoR skip ($ROLE, complete)"
            else
                PARTIAL_HAWOR=$(find "$HAWOR" -mindepth 1 -type f -print -quit)
                if [ -n "$PARTIAL_HAWOR" ]; then
                    echo "[$ID/$CAMERA] partial HaWoR cache without a final NPZ" >&2
                    echo "Move $HAWOR aside explicitly before retrying." >&2
                    exit 1
                fi
                echo "[$ID/$CAMERA] HaWoR ($ROLE)"
                MPLCONFIGDIR=/tmp/hawor-mpl \
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                conda run -n hawor --no-capture-output \
                    python "$ROOT/src/hand_estimation/extract_for_retarget.py" \
                    --rgb_dir "$RGB" \
                    --img_glob 'rgb_frame*.jpg' \
                    --img_focal "$FOCAL" \
                    --fps 24 \
                    --checkpoint "$ROOT/third_party/HaWoR/weights/hawor/checkpoints/hawor.ckpt" \
                    --infiller_weight "$ROOT/third_party/HaWoR/weights/hawor/checkpoints/infiller.pt" \
                    --skip_slam \
                    --vts_proj
                test -s "$NPZ"
                if ! load_hawor_info "$NPZ" hawor "$EXPECTED"; then
                    echo "[$ID/$CAMERA] invalid HaWoR output schema/coordinates" >&2
                    exit 1
                fi
                if [ "$HAWOR_COUNT" -ne "$EXPECTED" ] || ! focals_match "$HAWOR_FOCAL" "$FOCAL"; then
                    echo "[$ID/$CAMERA] invalid HaWoR output count/focal: $HAWOR_COUNT, $HAWOR_FOCAL" >&2
                    exit 1
                fi
            fi
        fi

        if stage_enabled haco; then
            if [ ! -s "$NPZ" ]; then
                echo "[$ID/$CAMERA] HaCo requires $NPZ" >&2
                exit 1
            fi
            if ! load_hawor_info "$NPZ" haco "$EXPECTED"; then
                echo "[$ID/$CAMERA] invalid HaWoR input schema/coordinates" >&2
                exit 1
            fi
            if [ "$HAWOR_COUNT" -ne "$EXPECTED" ]; then
                echo "[$ID/$CAMERA] HaCo input count mismatch: $HAWOR_COUNT != $EXPECTED" >&2
                exit 1
            fi
            if ! focals_match "$HAWOR_FOCAL" "$FOCAL"; then
                echo "[$ID/$CAMERA] HaWoR focal $HAWOR_FOCAL != requested $FOCAL" >&2
                exit 1
            fi
            CONTACT_COUNT=$(find "$CONTACT" -maxdepth 1 -type f -name 'rgb_frame*.npz' | wc -l)
            if [ "$CONTACT_COUNT" -eq "$EXPECTED" ]; then
                if ! validate_contact_output "$CONTACT" "$RGB" "$EXPECTED" "$HAWOR_FOCAL"; then
                    echo "[$ID/$CAMERA] invalid cached HaCo output" >&2
                    exit 1
                fi
                if ! focals_match "$CONTACT_FOCAL" "$HAWOR_FOCAL"; then
                    echo "[$ID/$CAMERA] cached HaCo focal $CONTACT_FOCAL != HaWoR $HAWOR_FOCAL" >&2
                    echo "Move $CONTACT aside explicitly before retrying." >&2
                    exit 1
                fi
                echo "[$ID/$CAMERA] HaCo skip ($ROLE, complete)"
            elif [ "$CONTACT_COUNT" -ne 0 ]; then
                echo "[$ID/$CAMERA] incomplete HaCo output: $CONTACT_COUNT != $EXPECTED" >&2
                echo "Move $CONTACT aside explicitly before retrying." >&2
                exit 1
            else
                echo "[$ID/$CAMERA] HaCo ($ROLE)"
                HACO_ARGS=(
                    --input_dir "$VIEW"
                    --img_glob 'rgb_frame*.jpg'
                )
                if [ "$HACO_VIZ" != "1" ]; then
                    HACO_ARGS+=(--no_viz)
                fi
                PYOPENGL_PLATFORM=egl \
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                conda run -n haco --no-capture-output \
                    python "$ROOT/src/contact_estimation/extract_hand_contact.py" \
                    "${HACO_ARGS[@]}"
                CONTACT_COUNT=$(find "$CONTACT" -maxdepth 1 -type f -name 'rgb_frame*.npz' | wc -l)
                if [ "$CONTACT_COUNT" -ne "$EXPECTED" ]; then
                    echo "[$ID/$CAMERA] HaCo output mismatch: $CONTACT_COUNT != $EXPECTED" >&2
                    exit 1
                fi
                if ! validate_contact_output "$CONTACT" "$RGB" "$EXPECTED" "$HAWOR_FOCAL"; then
                    echo "[$ID/$CAMERA] invalid HaCo output" >&2
                    exit 1
                fi
                if ! focals_match "$CONTACT_FOCAL" "$HAWOR_FOCAL"; then
                    echo "[$ID/$CAMERA] HaCo output focal $CONTACT_FOCAL != HaWoR $HAWOR_FOCAL" >&2
                    exit 1
                fi
            fi
        fi
    done
    if stage_enabled vjepa; then
        MH_NPZ="$EP/camera_2/rgb_hawor/retarget_input.npz"
        if [ ! -s "$MH_NPZ" ]; then
            echo "[$ID] V-JEPA bundle requires MH HaWoR NPZ: $MH_NPZ" >&2
            exit 1
        fi
        if ! load_hawor_info "$MH_NPZ" vjepa2-312 "$EXPECTED"; then
            echo "[$ID] invalid MH HaWoR NPZ for V-JEPA/MANO bundle" >&2
            exit 1
        fi
    fi
done

if stage_enabled vjepa; then
    RECORDING_GLOB=$(IFS=,; echo "${EPISODES[*]}")
    VJEPA_ARGS=(
        --data_root "$DATA"
        --recording_glob "$RECORDING_GLOB"
        --checkpoint "$VJEPA_CKPT"
        --device cuda
        --num_frames 16
        --sampling_profile vjepa2_4fps
        --source_fps 24
        --sample_fps 4
        --spatial_profile vjepa2_eval_center_crop
        --label_boundary_policy ignore
    )
    if [ "$VJEPA_OVERWRITE" = "1" ]; then
        VJEPA_ARGS+=(--overwrite)
    fi
    if [ -n "$VJEPA_ACTION_LABELS" ]; then
        VJEPA_ARGS+=(--action_labels "$VJEPA_ACTION_LABELS")
    fi
    PYTHONPATH="$ROOT/src" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    conda run -n vjepa2-312 --no-capture-output \
        python -m data_preprocess.preprocess "${VJEPA_ARGS[@]}"
fi

echo
echo "Completed stages '$STAGES' for: ${EPISODES[*]}"
