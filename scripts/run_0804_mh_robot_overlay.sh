#!/usr/bin/env bash
set -euo pipefail

# Build the MH-only RB5 + XHand overlay arrays consumed by the stereo
# visibility-HaCo compositor.  The safe default stops after a preview montage;
# request STAGES=render only after inspecting rb5_preview.png.
#
# Usage:
#   bash scripts/run_0804_mh_robot_overlay.sh 1
#   STAGES=render bash scripts/run_0804_mh_robot_overlay.sh 1
#   BACKGROUND_MODE=source STAGES=retarget,preview bash ... 1
#   BACKGROUND_MODE=source STAGES=render,composite bash ... 1
#   BACKGROUND_MODE=source USE_SH_HACO=1 STAGES=occlude bash ... 1
#   BACKGROUND_MODE=source USE_SH_HACO=0 STAGES=occlude bash ... 1  # MH only
#   CONTACT_INTERIOR_EXPAND_PX=3 STAGES=occlude bash ... 1  # A/B variant
#   CONTACT_DEPTH_THICKNESS_SCALE=0.5 STAGES=occlude bash ... 1  # half XHand
#   CONTACT_DEPTH_THICKNESS_SCALE=1.0 STAGES=occlude bash ... 1  # full XHand
#   FORCE=1 SIDE=left STAGES=retarget,preview bash ... 1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA="${DATA:-$ROOT/data/cube_dataset/26.08.04_stereo}"
STAGES="${STAGES:-retarget,preview}"
ALL="${ALL:-0}"
FORCE="${FORCE:-0}"
SIDE="${SIDE:-auto}"
RETARGET_ENV="${RETARGET_ENV:-RFM_retarget}"
RENDER_ENV="${RENDER_ENV:-inpaint-gpu}"
RENDER_SCALE="${RENDER_SCALE:-0.75}"
ARM_MODE="${ARM_MODE:-full}"
BACKGROUND_MODE="${BACKGROUND_MODE:-inpaint}"
USE_SH_HACO="${USE_SH_HACO:-1}"
CONTACT_INTERIOR_EXPAND_PX="${CONTACT_INTERIOR_EXPAND_PX:-0}"
CONTACT_INTERIOR_EXPAND_CAP_FRACTION="${CONTACT_INTERIOR_EXPAND_CAP_FRACTION:-0.25}"
CONTACT_DEPTH_THICKNESS_SCALE="${CONTACT_DEPTH_THICKNESS_SCALE:-0}"
XHAND_THUMB_THICKNESS_M="${XHAND_THUMB_THICKNESS_M:-0.03916}"
XHAND_FINGER_THICKNESS_M="${XHAND_FINGER_THICKNESS_M:-0.02930}"

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
        retarget|preview|render|composite|occlude) ;;
        *)
            echo "Unknown stage '$STAGE'; allowed: " \
                "retarget,preview,render,composite,occlude" >&2
            exit 1
            ;;
    esac
done
case "$SIDE" in
    auto|left|right) ;;
    *) echo "SIDE must be auto, left, or right" >&2; exit 1 ;;
esac
case "$BACKGROUND_MODE" in
    inpaint|source) ;;
    *) echo "BACKGROUND_MODE must be inpaint or source" >&2; exit 1 ;;
esac
case "$USE_SH_HACO" in
    0|1) ;;
    *) echo "USE_SH_HACO must be 0 or 1" >&2; exit 1 ;;
esac
CONTACT_INTERIOR_CAP_TAG=$(python -c '
import math, sys
px, fraction = int(sys.argv[1]), float(sys.argv[2])
if px < 0:
    raise SystemExit("CONTACT_INTERIOR_EXPAND_PX must be non-negative")
if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
    raise SystemExit("CONTACT_INTERIOR_EXPAND_CAP_FRACTION must be in [0,1]")
percent = fraction * 100.0
print((f"{percent:.3f}").rstrip("0").rstrip(".").replace(".", "p"))
' "$CONTACT_INTERIOR_EXPAND_PX" \
  "$CONTACT_INTERIOR_EXPAND_CAP_FRACTION")
read -r CONTACT_DEPTH_SCALE_TAG XHAND_THUMB_MM_TAG \
  XHAND_FINGER_MM_TAG CONTACT_DEPTH_THICKNESS_ENABLED < <(python -c '
import math, sys
scale, thumb, finger = map(float, sys.argv[1:])
if not math.isfinite(scale) or scale < 0.0:
    raise SystemExit("CONTACT_DEPTH_THICKNESS_SCALE must be finite and non-negative")
if not math.isfinite(thumb) or thumb <= 0.0:
    raise SystemExit("XHAND_THUMB_THICKNESS_M must be finite and positive")
if not math.isfinite(finger) or finger <= 0.0:
    raise SystemExit("XHAND_FINGER_THICKNESS_M must be finite and positive")
if not math.isfinite(scale * thumb) or not math.isfinite(scale * finger):
    raise SystemExit("scaled XHand thickness bias must be finite")
def tag(value):
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")
print(tag(scale), tag(thumb * 1000.0), tag(finger * 1000.0), int(scale > 0.0))
' "$CONTACT_DEPTH_THICKNESS_SCALE" "$XHAND_THUMB_THICKNESS_M" \
  "$XHAND_FINGER_THICKNESS_M")

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
for ID in "${EPISODES[@]}"; do
    [[ "$ID" =~ ^[0-9]+$ ]] || {
        echo "Episode names must be numeric, got '$ID'" >&2
        exit 1
    }
done

if stage_enabled retarget || [ "$SIDE" = "auto" ]; then
    conda run -n "$RETARGET_ENV" python -c \
        'import numpy, scipy, pinocchio, dex_retargeting'
fi
if stage_enabled preview || stage_enabled render || \
   stage_enabled composite || stage_enabled occlude; then
    conda run -n "$RENDER_ENV" python -c \
        'import numpy, pyrender, trimesh, cv2'
fi

validate_overlay_dir() {
    local overlay_dir="$1"
    local expected_side="$2"
    conda run -n "$RENDER_ENV" python -c '
import json, numpy as np, sys
from pathlib import Path
root, side = Path(sys.argv[1]), sys.argv[2]
manifest_path = root / "manifest.json"
manifest = json.loads(manifest_path.read_text())
actual_side = manifest.get("side")
if actual_side != side:
    raise SystemExit(f"overlay side {actual_side!r} != {side!r}")
t = int(manifest["frame_count"])
w, h = map(int, manifest["render_size"])
expected = {
    "robot_rgb.npy": (t, h, w, 3),
    "robot_depth.npy": (t, h, w),
    "robot_mask.npy": (t, h, w),
    "robot_finger_labels.npy": (t, h, w),
    "robot_finger_surface_labels.npy": (t, h, w),
    "robot_finger_mask.npy": (t, h, w),
}
for name, shape in expected.items():
    path = root / name
    if not path.is_file():
        raise SystemExit(f"missing overlay array: {path}")
    array = np.load(path, mmap_mode="r")
    if array.shape != shape:
        raise SystemExit(f"{path}: shape {array.shape} != {shape}")
labels = np.load(root / "robot_finger_labels.npy", mmap_mode="r")
if int(labels.max(initial=0)) > 5:
    raise SystemExit("robot_finger_labels contains an unknown label")
surface_path = root / "robot_finger_surface_labels.npy"
surface_labels = np.load(surface_path, mmap_mode="r")
if surface_labels.dtype != np.uint8:
    raise SystemExit(
        f"{surface_path}: dtype {surface_labels.dtype} != uint8"
    )
if surface_labels.size and (
    int(surface_labels.min()) < 0 or int(surface_labels.max()) > 15
):
    raise SystemExit("robot_finger_surface_labels must be within [0,15]")
mismatch = 0
for frame_index in range(t):
    packed_frame = np.asarray(surface_labels[frame_index])
    decoded_frame = np.zeros(packed_frame.shape, dtype=np.uint8)
    active = packed_frame > 0
    decoded_frame[active] = (
        (packed_frame[active].astype(np.int16) - 1) // 3 + 1
    ).astype(np.uint8)
    mismatch += int(np.count_nonzero(decoded_frame != labels[frame_index]))
if mismatch:
    raise SystemExit(
        "robot_finger_surface_labels do not decode exactly to "
        f"robot_finger_labels ({mismatch} mismatched pixels)"
    )
expected_surface_contract = {
    "filename": "robot_finger_surface_labels.npy",
    "dtype": "uint8",
    "background": 0,
    "valid_range": [0, 15],
    "surface_ids": {"palmar": 1, "lateral": 2, "dorsal": 3},
    "packing": "(finger_id - 1) * 3 + surface_id",
    "decode": {
        "finger_id": "((packed_id - 1) // 3) + 1",
        "surface_id": "((packed_id - 1) % 3) + 1",
        "condition": "packed_id > 0",
    },
    "normal_frame": "xhand_link",
    "face_normal_dot_threshold": 0.5,
    "palmar_normal_axes": {
        "right": {"thumb": [0.0, 0.0, -1.0], "other": [0.0, 1.0, 0.0]},
        "left": {"thumb": [0.0, 0.0, -1.0], "other": [0.0, -1.0, 0.0]},
    },
}
actual_surface_contract = manifest.get("finger_surface_labels")
if not isinstance(actual_surface_contract, dict):
    raise SystemExit(
        "manifest finger_surface_labels contract is missing or incompatible"
    )
incompatible_surface_fields = {
    key: {
        "expected": expected_value,
        "actual": actual_surface_contract.get(key),
    }
    for key, expected_value in expected_surface_contract.items()
    if actual_surface_contract.get(key) != expected_value
}
if incompatible_surface_fields:
    raise SystemExit(
        "manifest finger_surface_labels contract has incompatible fields: "
        + json.dumps(incompatible_surface_fields, sort_keys=True)
    )
' "$overlay_dir" "$expected_side"
}

validate_contact_occlusion() {
    local output_dir="$1"
    local expected_frames="$2"
    local expect_dual_haco="$3"
    local expected_expand_px="$4"
    local expected_expand_cap="$5"
    local expected_thickness_scale="$6"
    local expected_thumb_thickness="$7"
    local expected_finger_thickness="$8"
    conda run -n "$RENDER_ENV" python -c '
import json, numpy as np, sys
from pathlib import Path
root = Path(sys.argv[1])
expected, expect_dual = int(sys.argv[2]), bool(int(sys.argv[3]))
expected_expand_px, expected_expand_cap = int(sys.argv[4]), float(sys.argv[5])
expected_scale = float(sys.argv[6])
expected_thumb, expected_finger = float(sys.argv[7]), float(sys.argv[8])
report = json.loads((root / "report.json").read_text())
if report.get("occlusion_mode") != "haco":
    raise SystemExit("contact occlusion report is not HaCo mode")
actual_frames = report.get("frames")
if int(actual_frames or -1) != expected:
    raise SystemExit(f"contact occlusion frame mismatch: {actual_frames} != {expected}")
if not report.get("invariants", {}).get("occluded_subset_of_robot_fingers"):
    raise SystemExit("contact occlusion finger-subset invariant is missing")
expansion = report.get("contact_interior_expansion", {})
if bool(expansion.get("enabled")) != (expected_expand_px > 0):
    raise SystemExit("contact interior expansion mode mismatch")
if int(expansion.get("expand_px", -1)) != expected_expand_px:
    raise SystemExit("contact interior expansion radius mismatch")
actual_cap = expansion.get("added_pixel_cap_fraction_of_verified_seed")
if actual_cap is None or not np.isclose(float(actual_cap), expected_expand_cap):
    raise SystemExit("contact interior expansion cap mismatch")
config = report.get("config", {})
if int(config.get("contact_interior_expand_px", -1)) != expected_expand_px:
    raise SystemExit("contact interior config radius mismatch")
if not np.isclose(
    float(config.get("contact_interior_expand_cap_fraction", -1.0)),
    expected_expand_cap,
):
    raise SystemExit("contact interior config cap mismatch")
actual_scale = float(config.get("contact_depth_thickness_scale", 0.0))
if not np.isclose(actual_scale, expected_scale):
    raise SystemExit("XHand contact depth thickness scale mismatch")
actual_thumb = float(config.get("xhand_thumb_thickness_m", 0.03916))
actual_finger = float(config.get("xhand_finger_thickness_m", 0.02930))
if not np.isclose(actual_thumb, expected_thumb):
    raise SystemExit("XHand thumb thickness mismatch")
if not np.isclose(actual_finger, expected_finger):
    raise SystemExit("XHand non-thumb thickness mismatch")
bias = report.get("xhand_contact_depth_bias")
if bias is None:
    if expected_scale > 0.0:
        raise SystemExit("missing XHand contact depth bias report")
else:
    if bool(bias.get("enabled")) != (expected_scale > 0.0):
        raise SystemExit("XHand contact depth bias mode mismatch")
    if not np.isclose(float(bias.get("scale", -1.0)), expected_scale):
        raise SystemExit("XHand contact depth bias report scale mismatch")
    if bias.get("metric_object_depth_gate_modified") is not False:
        raise SystemExit("XHand bias must not modify metric object-depth gate")
    applied = bias.get("applied_bias_m", {})
    expected_bias = {
        "thumb": expected_scale * expected_thumb,
        "index": expected_scale * expected_finger,
        "middle": expected_scale * expected_finger,
        "ring": expected_scale * expected_finger,
        "pinky": expected_scale * expected_finger,
    }
    if any(
        name not in applied
        or not np.isclose(float(applied[name]), value)
        for name, value in expected_bias.items()
    ):
        raise SystemExit("per-finger XHand contact depth bias mismatch")
    invariants = report.get("invariants", {})
    if not invariants.get("xhand_thickness_bias_is_contact_proxy_only"):
        raise SystemExit("XHand contact-proxy-only invariant is missing")
    if not invariants.get("sensor_object_depth_gate_is_unbiased"):
        raise SystemExit("sensor object-depth unbiased invariant is missing")
if expect_dual:
    expected_fusion = "per-finger maximum of primary/auxiliary HaCo scores"
    if report.get("contact_fusion") != expected_fusion:
        raise SystemExit("contact occlusion report is not dual-view HaCo")
    policy = report.get("contact_activation_policy", {})
    if policy.get("name") != "mh_geometry_with_sh_confidence_rescue":
        raise SystemExit("dual-view HaCo report has the wrong SH policy")
    if policy.get("auxiliary_geometry_used") is not False:
        raise SystemExit("SH geometry must not be used without calibration")
    if not report.get("sources", {}).get("aux_contact_dir"):
        raise SystemExit("dual-view HaCo report has no SH contact source")
    invariants = report.get("invariants", {})
    if not invariants.get("auxiliary_haco_is_confidence_only"):
        raise SystemExit("dual-view HaCo confidence-only invariant is missing")
    if not invariants.get("primary_view_owns_contact_projection_and_depth"):
        raise SystemExit("primary-view geometry invariant is missing")
for name in (
    "video_overlay_contact.mp4",
    "video_robot_only_contact.mp4",
    "debug_contact_occlusion.mp4",
    "occluded_finger_mask.npy",
):
    path = root / name
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing contact occlusion output: {path}")
mask = np.load(root / "occluded_finger_mask.npy", mmap_mode="r")
expected_shape = (expected, int(report["height"]), int(report["width"]))
if mask.shape != expected_shape or mask.dtype != np.bool_:
    raise SystemExit(
        f"contact mask contract mismatch: {mask.shape}/{mask.dtype} "
        f"!= {expected_shape}/bool"
    )
' "$output_dir" "$expected_frames" "$expect_dual_haco" \
  "$expected_expand_px" "$expected_expand_cap" \
  "$expected_thickness_scale" "$expected_thumb_thickness" \
  "$expected_finger_thickness"
}

for ID in "${EPISODES[@]}"; do
    EP="$DATA/$ID"
    C2="$EP/camera_2"
    NPZ="$C2/rgb_hawor/retarget_input.npz"
    CONTACT="$C2/contact"
    SH_CONTACT="$EP/camera_1/contact"
    PD="$C2/visibility/processed/view/0"
    if [ "$BACKGROUND_MODE" = "source" ]; then
        BACKGROUND="$C2/source.mov"
    else
        BACKGROUND="$PD/inpaint_processor/video_human_inpaint.mkv"
    fi
    test -s "$NPZ"
    test -d "$CONTACT"
    if stage_enabled preview || stage_enabled render || \
       stage_enabled composite || stage_enabled occlude; then
        test -s "$BACKGROUND"
    fi

    EPISODE_SIDE="$SIDE"
    if [ "$EPISODE_SIDE" = "auto" ]; then
        EPISODE_SIDE=$(conda run -n "$RETARGET_ENV" python -c '
import numpy as np, sys
with np.load(sys.argv[1]) as data:
    left, right = (int(x) for x in np.asarray(data["valid"], dtype=bool).sum(axis=1))
if max(left, right) == 0 or left == right:
    raise SystemExit(f"cannot infer a dominant hand: left={left}, right={right}")
print("left" if left > right else "right")
' "$NPZ" | tr -d '[:space:]')
    fi

    PKL="$C2/rgb_hawor/qpos_xhand_contact_${EPISODE_SIDE}_smooth.pkl"
    OVERLAY_INPUT="$PD/rb5_overlay_input_${EPISODE_SIDE}.npz"
    JOINT_NAMES="$PD/rb5_overlay_input_${EPISODE_SIDE}_jointnames.json"
    PREVIEW="$PD/rb5_preview.png"
    OVERLAY_DIR="$PD/overlay_processor"
    RAW_OVERLAY="$PD/video_overlay_robot_raw.mp4"
    ROBOT_ONLY="$OVERLAY_DIR/video_robot_only.mp4"
    OBJECT_MASK="$PD/object_layer/object_mask_modal.npy"
    if [ "$BACKGROUND_MODE" = "source" ] && [ "$USE_SH_HACO" = "1" ]; then
        CONTACT_OCCLUSION_DIR="$PD/contact_occlusion_dual_haco_raw"
    elif [ "$BACKGROUND_MODE" = "source" ]; then
        CONTACT_OCCLUSION_DIR="$PD/contact_occlusion_raw"
    elif [ "$USE_SH_HACO" = "1" ]; then
        CONTACT_OCCLUSION_DIR="$PD/contact_occlusion_dual_haco"
    else
        CONTACT_OCCLUSION_DIR="$PD/contact_occlusion"
    fi
    if [ "$CONTACT_INTERIOR_EXPAND_PX" -gt 0 ]; then
        if [ "$BACKGROUND_MODE" = "source" ]; then
            CONTACT_OCCLUSION_DIR="${CONTACT_OCCLUSION_DIR%_raw}_boundaryfill_${CONTACT_INTERIOR_EXPAND_PX}px_cap${CONTACT_INTERIOR_CAP_TAG}_raw"
        else
            CONTACT_OCCLUSION_DIR="${CONTACT_OCCLUSION_DIR}_boundaryfill_${CONTACT_INTERIOR_EXPAND_PX}px_cap${CONTACT_INTERIOR_CAP_TAG}"
        fi
    fi
    if [ "$CONTACT_DEPTH_THICKNESS_ENABLED" = "1" ]; then
        XHAND_DEPTH_TAG="xhanddepth_s${CONTACT_DEPTH_SCALE_TAG}_t${XHAND_THUMB_MM_TAG}mm_f${XHAND_FINGER_MM_TAG}mm"
        if [ "$BACKGROUND_MODE" = "source" ]; then
            CONTACT_OCCLUSION_DIR="${CONTACT_OCCLUSION_DIR%_raw}_${XHAND_DEPTH_TAG}_raw"
        else
            CONTACT_OCCLUSION_DIR="${CONTACT_OCCLUSION_DIR}_${XHAND_DEPTH_TAG}"
        fi
    fi

    if stage_enabled retarget; then
        if [ "$FORCE" = "1" ] || [ ! -s "$PKL" ]; then
            echo "[$ID] MH contact-aware XHand retarget ($EPISODE_SIDE)"
            conda run -n "$RETARGET_ENV" --no-capture-output \
                python "$ROOT/src/retargeting/retarget_from_npz.py" \
                --npz "$NPZ" \
                --hand "$EPISODE_SIDE" \
                --out_dir "$C2/rgb_hawor" \
                --contact \
                --contact_dir "$CONTACT" \
                --smooth
        fi
        test -s "$PKL"

        if [ "$FORCE" = "1" ] || [ ! -s "$OVERLAY_INPUT" ]; then
            echo "[$ID] MH RB5 IK adapter ($EPISODE_SIDE)"
            conda run -n "$RETARGET_ENV" --no-capture-output \
                python "$ROOT/src/inpainting/rb5_build_overlay_input.py" \
                --hawor_npz "$NPZ" \
                --pkl "$PKL" \
                --side "$EPISODE_SIDE" \
                --out "$OVERLAY_INPUT" \
                --img_w 1280 \
                --img_h 720
        fi
    fi
    test -s "$OVERLAY_INPUT"
    test -s "$JOINT_NAMES"

    if stage_enabled preview; then
        if [ "$FORCE" = "1" ] || [ ! -s "$PREVIEW" ]; then
            PREVIEW_ARGS=(
                --data "$OVERLAY_INPUT"
                --jn "$JOINT_NAMES"
                --out "$PD"
                --background "$BACKGROUND"
                --render_scale "$RENDER_SCALE"
                --arm_mode "$ARM_MODE"
                --preview
                --start 0
                --n 6
            )
            if [ "$FORCE" = "1" ]; then
                PREVIEW_ARGS+=(--overwrite)
            fi
            echo "[$ID] MH RB5 placement preview"
            PYOPENGL_PLATFORM=egl \
            conda run -n "$RENDER_ENV" --no-capture-output \
                python "$ROOT/src/inpainting/render_rb5_pyrender_overlay.py" \
                "${PREVIEW_ARGS[@]}"
        fi
        echo "[$ID] inspect before full render: $PREVIEW"
    fi

    if stage_enabled render; then
        if [ "$FORCE" != "1" ] && [ -s "$OVERLAY_DIR/manifest.json" ]; then
            if ! validate_overlay_dir "$OVERLAY_DIR" "$EPISODE_SIDE"; then
                echo "[$ID] incomplete/incompatible overlay cache; rerun with FORCE=1" >&2
                exit 1
            fi
            echo "[$ID] MH robot arrays skip (validated complete)"
        else
            RENDER_ARGS=(
                --data "$OVERLAY_INPUT"
                --jn "$JOINT_NAMES"
                --out "$PD"
                --background "$BACKGROUND"
                --render_scale "$RENDER_SCALE"
                --arm_mode "$ARM_MODE"
            )
            if [ "$FORCE" = "1" ]; then
                RENDER_ARGS+=(--overwrite)
            fi
            echo "[$ID] MH full RB5 + XHand arrays"
            PYOPENGL_PLATFORM=egl \
            conda run -n "$RENDER_ENV" --no-capture-output \
                python "$ROOT/src/inpainting/render_rb5_pyrender_overlay.py" \
                "${RENDER_ARGS[@]}"
        fi
        validate_overlay_dir "$OVERLAY_DIR" "$EPISODE_SIDE"
        echo "[$ID] overlay arrays: $OVERLAY_DIR"
    fi

    if stage_enabled composite; then
        validate_overlay_dir "$OVERLAY_DIR" "$EPISODE_SIDE"
        if [ "$FORCE" != "1" ] && { [ -s "$RAW_OVERLAY" ] || [ -s "$ROBOT_ONLY" ]; }; then
            if [ -s "$RAW_OVERLAY" ] && [ -s "$ROBOT_ONLY" ]; then
                echo "[$ID] raw MH overlay skip (complete): $RAW_OVERLAY"
            else
                echo "[$ID] incomplete raw overlay; rerun with FORCE=1" >&2
                exit 1
            fi
        else
            echo "[$ID] simple robot-over-MH composite ($BACKGROUND_MODE background)"
            conda run -n "$RENDER_ENV" --no-capture-output \
                python "$ROOT/src/inpainting/composite_robot_unclipped.py" \
                --processed_demo "$PD" \
                --background "$BACKGROUND" \
                --out "$RAW_OVERLAY" \
                --robot_only_out "$ROBOT_ONLY" \
                --fps 24
        fi
        test -s "$RAW_OVERLAY"
        test -s "$ROBOT_ONLY"
        echo "[$ID] raw MH overlay: $RAW_OVERLAY"
    fi

    if stage_enabled occlude; then
        validate_overlay_dir "$OVERLAY_DIR" "$EPISODE_SIDE"
        test -s "$OBJECT_MASK"
        EXPECTED_FRAMES=$(python -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["num_frames"])' \
            "$EP/gt_labels.json")
        CONTACT_COMPLETE=$(find "$CONTACT" -maxdepth 1 -type f \
            -name 'rgb_frame*.npz' | wc -l)
        if [ "$CONTACT_COMPLETE" -ne "$EXPECTED_FRAMES" ]; then
            echo "[$ID] incomplete MH HaCo: $CONTACT_COMPLETE != $EXPECTED_FRAMES" >&2
            exit 1
        fi
        AUX_FRAME_OFFSET=0
        if [ "$USE_SH_HACO" = "1" ]; then
            test -s "$EP/stereo_manifest.json"
            AUX_FRAME_OFFSET=$(python -c \
                'import json,sys; print(json.load(open(sys.argv[1]))["temporal_alignment"]["camera1_frame_offset"])' \
                "$EP/stereo_manifest.json")
            SH_CONTACT_COMPLETE=$(find "$SH_CONTACT" -maxdepth 1 -type f \
                -name 'rgb_frame*.npz' | wc -l)
            if [ "$SH_CONTACT_COMPLETE" -ne "$EXPECTED_FRAMES" ]; then
                echo "[$ID] incomplete SH HaCo: " \
                    "$SH_CONTACT_COMPLETE != $EXPECTED_FRAMES" >&2
                exit 1
            fi
        fi
        COMPLETE_CONTACT=0
        if [ -s "$CONTACT_OCCLUSION_DIR/report.json" ]; then
            if validate_contact_occlusion \
                "$CONTACT_OCCLUSION_DIR" "$EXPECTED_FRAMES" \
                "$USE_SH_HACO" "$CONTACT_INTERIOR_EXPAND_PX" \
                "$CONTACT_INTERIOR_EXPAND_CAP_FRACTION" \
                "$CONTACT_DEPTH_THICKNESS_SCALE" \
                "$XHAND_THUMB_THICKNESS_M" \
                "$XHAND_FINGER_THICKNESS_M"; then
                COMPLETE_CONTACT=1
            fi
        fi
        if [ "$FORCE" != "1" ] && [ "$COMPLETE_CONTACT" = "1" ]; then
            echo "[$ID] HaCo contact-depth occlusion skip (complete): " \
                "$CONTACT_OCCLUSION_DIR"
        elif [ "$FORCE" != "1" ] && [ -e "$CONTACT_OCCLUSION_DIR" ]; then
            echo "[$ID] incomplete contact-occlusion output; rerun with FORCE=1" >&2
            exit 1
        else
            CONTACT_ARGS=(
                --processed_demo "$PD"
                --episode_dir "$C2"
                --background "$BACKGROUND"
                --raw_video "$C2/source.mov"
                --overlay_dir "$OVERLAY_DIR"
                --object_mask "$OBJECT_MASK"
                --occlusion_mode haco
                --out_dir "$CONTACT_OCCLUSION_DIR"
                --contact_interior_expand_px "$CONTACT_INTERIOR_EXPAND_PX"
                --contact_interior_expand_cap_fraction \
                    "$CONTACT_INTERIOR_EXPAND_CAP_FRACTION"
                --contact_depth_thickness_scale \
                    "$CONTACT_DEPTH_THICKNESS_SCALE"
                --xhand_thumb_thickness_m "$XHAND_THUMB_THICKNESS_M"
                --xhand_finger_thickness_m "$XHAND_FINGER_THICKNESS_M"
            )
            if [ "$USE_SH_HACO" = "1" ]; then
                CONTACT_ARGS+=(
                    --aux_contact_dir "$SH_CONTACT"
                    --aux_frame_offset "$AUX_FRAME_OFFSET"
                    --aux_side "$EPISODE_SIDE"
                )
                echo "[$ID] SH+MH HaCo fusion; MH contact-depth -> hide finger"
            else
                echo "[$ID] MH HaCo contact-depth -> hide robot finger"
            fi
            PYTHONPATH="$ROOT/src/inpainting" \
            conda run -n "$RENDER_ENV" --no-capture-output \
                python "$ROOT/src/inpainting/composite_rb5_contact_occlusion.py" \
                "${CONTACT_ARGS[@]}"
            validate_contact_occlusion \
                "$CONTACT_OCCLUSION_DIR" "$EXPECTED_FRAMES" \
                "$USE_SH_HACO" "$CONTACT_INTERIOR_EXPAND_PX" \
                "$CONTACT_INTERIOR_EXPAND_CAP_FRACTION" \
                "$CONTACT_DEPTH_THICKNESS_SCALE" \
                "$XHAND_THUMB_THICKNESS_M" \
                "$XHAND_FINGER_THICKNESS_M"
        fi
        echo "[$ID] contact-aware MH overlay: " \
            "$CONTACT_OCCLUSION_DIR/video_overlay_contact.mp4"
    fi
done

echo
echo "Completed stages '$STAGES' for: ${EPISODES[*]}"
