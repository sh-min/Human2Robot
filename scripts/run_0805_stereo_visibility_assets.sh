#!/usr/bin/env bash
set -euo pipefail

# Build only the 08-05 SH hand/arm visibility masks needed by the stereo
# occlusion compositor.  camera_1 is SH (auxiliary evidence); camera_2 is MH
# (the final/output view).  The established MH mask under camera_2/inpainting
# is immutable here: this runner validates and reuses it, but never regenerates
# or copies it.
#
# Derived SH cache policy:
#   * missing outputs are created;
#   * complete, valid, current outputs are reused;
#   * an existing stale/partial/invalid output is never replaced unless
#     FORCE=1 was explicitly supplied.
#
# Examples:
#   bash scripts/run_0805_stereo_visibility_assets.sh 1
#   ALL=1 BRANCHES=approx,calibrated \
#     bash scripts/run_0805_stereo_visibility_assets.sh
#   BRANCHES=calibrated STAGES=prepare,inject \
#     bash scripts/run_0805_stereo_visibility_assets.sh 1 2
#   FORCE=1 BRANCHES=calibrated STAGES=segment \
#     bash scripts/run_0805_stereo_visibility_assets.sh 1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
APPROX_ROOT="${APPROX_ROOT:-$ROOT/data/cube_dataset/26.08.05_stereo_approx}"
CALIBRATED_ROOT="${CALIBRATED_ROOT:-$ROOT/data/cube_dataset/26.08.05_stereo_calibrated}"
ENVIRONMENT="${ENVIRONMENT:-inpaint-gpu}"
BRANCHES="${BRANCHES:-approx,calibrated}"
STAGES="${STAGES:-all}"
ALL="${ALL:-0}"
FORCE="${FORCE:-0}"

APPROX_SH_FOCAL="${APPROX_SH_FOCAL:-924.4444580078125}"
APPROX_MH_FOCAL="${APPROX_MH_FOCAL:-924.4444580078125}"
CALIBRATED_SH_FOCAL="${CALIBRATED_SH_FOCAL:-1070.3365288025332}"
CALIBRATED_MH_FOCAL="${CALIBRATED_MH_FOCAL:-1030.2115914516535}"
EXPECTED_SIDE="left"
EXPECTED_FPS="24"
EXPECTED_WIDTH="1280"
EXPECTED_HEIGHT="720"
SAM2_CHECKPOINT="$ROOT/third_party/sam2/checkpoints/sam2_hiera_large.pt"

die() {
    echo "$*" >&2
    exit 1
}

case "$ALL" in
    0|1) ;;
    *) die "ALL must be 0 or 1" ;;
esac
case "$FORCE" in
    0|1) ;;
    *) die "FORCE must be 0 or 1" ;;
esac

stage_enabled() {
    [ "$STAGES" = "all" ] || [[ ",$STAGES," == *",$1,"* ]]
}

IFS=',' read -r -a REQUESTED_STAGES <<< "$STAGES"
[ "${#REQUESTED_STAGES[@]}" -gt 0 ] || die "STAGES must not be empty"
for STAGE in "${REQUESTED_STAGES[@]}"; do
    case "$STAGE" in
        all|prepare|inject|segment|validate) ;;
        *) die "Unknown stage '$STAGE'; use prepare,inject,segment,validate, or all" ;;
    esac
done

IFS=',' read -r -a REQUESTED_BRANCHES <<< "$BRANCHES"
[ "${#REQUESTED_BRANCHES[@]}" -gt 0 ] || die "BRANCHES must not be empty"
for BRANCH in "${REQUESTED_BRANCHES[@]}"; do
    case "$BRANCH" in
        approx|calibrated) ;;
        *) die "Unknown branch '$BRANCH'; use approx and/or calibrated" ;;
    esac
done

python -c '
import math, sys
for value in sys.argv[1:]:
    focal = float(value)
    if not math.isfinite(focal) or focal <= 0.0:
        raise SystemExit(f"invalid focal length: {value}")
' "$APPROX_SH_FOCAL" "$APPROX_MH_FOCAL" \
  "$CALIBRATED_SH_FOCAL" "$CALIBRATED_MH_FOCAL"

if [ "$#" -gt 0 ]; then
    EPISODES=("$@")
elif [ "$ALL" = "1" ]; then
    mapfile -t EPISODES < <(
        for BRANCH in "${REQUESTED_BRANCHES[@]}"; do
            if [ "$BRANCH" = "approx" ]; then
                DATA_ROOT="$APPROX_ROOT"
            else
                DATA_ROOT="$CALIBRATED_ROOT"
            fi
            [ -d "$DATA_ROOT" ] || continue
            find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n'
        done | awk '/^[0-9]+$/' | sort -Vu
    )
else
    EPISODES=("1")
fi
[ "${#EPISODES[@]}" -gt 0 ] || die "No numeric episodes selected"
for ID in "${EPISODES[@]}"; do
    [[ "$ID" =~ ^[0-9]+$ ]] || die "Episode names must be numeric, got '$ID'"
done

command -v ffprobe >/dev/null || die "ffprobe is required"
command -v conda >/dev/null || die "conda is required"
command -v realpath >/dev/null || die "realpath is required"
conda run -n "$ENVIRONMENT" python -c 'import cv2, numpy' >/dev/null

fresh_file() {
    local target="$1"
    shift
    [ -s "$target" ] || return 1
    local source
    for source in "$@"; do
        [ -e "$source" ] || return 1
        [ "$source" -nt "$target" ] && return 1
    done
    return 0
}

rgb_tree_not_newer_than() {
    local rgb_dir="$1"
    local frames="$2"
    local target="$3"
    local frame_index frame_path
    for ((frame_index = 0; frame_index < frames; frame_index++)); do
        printf -v frame_path '%s/rgb_frame%06d.jpg' "$rgb_dir" "$frame_index"
        [ -s "$frame_path" ] || return 1
        [ "$frame_path" -nt "$target" ] && return 1
    done
    return 0
}

any_path_exists() {
    local path
    for path in "$@"; do
        [ -e "$path" ] && return 0
    done
    return 1
}

validate_manifest() {
    local manifest="$1"
    local gt="$2"
    local branch="$3"
    local sh_focal="$4"
    local mh_focal="$5"
    python -c '
import json, math, sys
from pathlib import Path

manifest_path, gt_path = map(Path, sys.argv[1:3])
branch = sys.argv[3]
sh_focal, mh_focal = map(float, sys.argv[4:6])
manifest = json.loads(manifest_path.read_text())
gt = json.loads(gt_path.read_text())
frames = int(gt["num_frames"])
if float(gt["fps"]) != 24.0:
    raise SystemExit(f"{gt_path}: GT FPS is not 24")
expected = {
    "primary_view": "MH",
    "auxiliary_view": "SH",
    "robot_overlay_view": "MH",
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(f"{manifest_path}: {key}={manifest.get(key)!r} != {value!r}")
if manifest.get("stereo_code_mapping") != {"camera_1": "SH", "camera_2": "MH"}:
    raise SystemExit(f"{manifest_path}: camera namespace mismatch")
if int(manifest.get("common_frames", -1)) != frames:
    raise SystemExit(f"{manifest_path}: common frame count mismatch")
if float(manifest.get("fps", -1)) != 24.0:
    raise SystemExit(f"{manifest_path}: manifest FPS is not 24")
intrinsics = manifest.get("intrinsics", {})
pixel_focal = intrinsics.get("pixel_focal_px", {})
if branch == "calibrated":
    if intrinsics.get("status") != "provided":
        raise SystemExit(f"{manifest_path}: calibrated intrinsics are not provided")
    for view, actual, wanted in (
        ("SH", pixel_focal.get("SH"), sh_focal),
        ("MH", pixel_focal.get("MH"), mh_focal),
    ):
        if actual is None or not math.isclose(float(actual), wanted, rel_tol=0.0, abs_tol=1e-3):
            raise SystemExit(f"{manifest_path}: {view} focal {actual} != {wanted}")
else:
    if intrinsics.get("status") != "not_provided":
        raise SystemExit(f"{manifest_path}: approx branch unexpectedly has intrinsics")
    if pixel_focal.get("SH") is not None or pixel_focal.get("MH") is not None:
        raise SystemExit(f"{manifest_path}: approx focal fields must be null")
print(frames)
' "$manifest" "$gt" "$branch" "$sh_focal" "$mh_focal"
}

validate_view_inputs() {
    local rgb_dir="$1"
    local hawor="$2"
    local frames="$3"
    local focal="$4"
    local view_name="$5"
    conda run -n "$ENVIRONMENT" python -c '
import math, sys
from pathlib import Path
import cv2
import numpy as np

rgb_dir, hawor_path = map(Path, sys.argv[1:3])
frames = int(sys.argv[3])
expected_focal = float(sys.argv[4])
view_name, expected_side = sys.argv[5:7]
expected_names = [f"rgb_frame{index:06d}.jpg" for index in range(frames)]
actual = sorted(path.name for path in rgb_dir.glob("rgb_frame*.jpg"))
if actual != expected_names:
    raise SystemExit(
        f"{rgb_dir}: RGB sequence is not exactly 0..{frames - 1} "
        f"({len(actual)} files)"
    )
for name in expected_names:
    path = rgb_dir / name
    if path.stat().st_size <= 0:
        raise SystemExit(f"{path}: empty RGB frame")
for name in (expected_names[0], expected_names[-1]):
    image = cv2.imread(str(rgb_dir / name), cv2.IMREAD_COLOR)
    if image is None or image.shape != (720, 1280, 3):
        raise SystemExit(f"{rgb_dir / name}: expected 1280x720 RGB")
with np.load(hawor_path, allow_pickle=False) as data:
    shapes = {
        "joints_left": (frames, 21, 3),
        "joints_right": (frames, 21, 3),
        "verts_left": (frames, 778, 3),
        "verts_right": (frames, 778, 3),
        "valid": (2, frames),
    }
    missing = sorted(set(shapes) - set(data.files))
    if missing:
        raise SystemExit(f"{hawor_path}: missing keys {missing}")
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise SystemExit(f"{hawor_path}: {key} shape {data[key].shape} != {shape}")
    if data["valid"].dtype != np.bool_:
        raise SystemExit(f"{hawor_path}: valid dtype must be bool")
    if not bool(np.asarray(data["frame_is_cam_space"]).item()):
        raise SystemExit(f"{hawor_path}: HaWoR must be camera-space")
    stored_focal = float(np.asarray(data["img_focal"]).item())
    if not math.isclose(stored_focal, expected_focal, rel_tol=0.0, abs_tol=1e-3):
        raise SystemExit(f"{hawor_path}: focal {stored_focal} != {expected_focal}")
    valid = np.asarray(data["valid"], dtype=bool)
    counts = valid.sum(axis=1)
    dominant = "left" if counts[0] > counts[1] else "right" if counts[1] > counts[0] else None
    if dominant != expected_side or int(counts[0]) <= 0:
        raise SystemExit(
            f"{hawor_path}: expected dominant {expected_side} hand, "
            f"valid L/R={int(counts[0])}/{int(counts[1])}"
        )
print(
    f"[{view_name}] frames={frames}, focal={stored_focal:.6f}, "
    f"side={dominant}, valid L/R={int(counts[0])}/{int(counts[1])}"
)
' "$rgb_dir" "$hawor" "$frames" "$focal" "$view_name" "$EXPECTED_SIDE"
}

validate_video() {
    local video="$1"
    local frames="$2"
    local metadata
    [ -s "$video" ] || return 1
    metadata=$(ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=width,height,avg_frame_rate,nb_read_frames \
        -of json "$video") || return 1
    python -c '
import json, sys
from fractions import Fraction
payload = json.loads(sys.argv[1])
expected_frames, expected_width, expected_height = map(int, sys.argv[2:5])
streams = payload.get("streams", [])
if len(streams) != 1:
    raise SystemExit(1)
stream = streams[0]
if int(stream.get("width", -1)) != expected_width:
    raise SystemExit(1)
if int(stream.get("height", -1)) != expected_height:
    raise SystemExit(1)
if int(stream.get("nb_read_frames", -1)) != expected_frames:
    raise SystemExit(1)
if Fraction(stream.get("avg_frame_rate", "0/1")) != Fraction(24, 1):
    raise SystemExit(1)
' "$metadata" "$frames" "$EXPECTED_WIDTH" "$EXPECTED_HEIGHT"
}

validate_injected() {
    local pd="$1"
    local hawor="$2"
    local frames="$3"
    conda run -n "$ENVIRONMENT" python -c '
import sys
from pathlib import Path
import numpy as np

pd, hawor_path = map(Path, sys.argv[1:3])
frames = int(sys.argv[3])
with np.load(hawor_path, allow_pickle=False) as source:
    valid = np.asarray(source["valid"], dtype=bool)
    joints = {
        "left": np.asarray(source["joints_left"], dtype=np.float64),
        "right": np.asarray(source["joints_right"], dtype=np.float64),
    }
    focal = float(np.asarray(source["img_focal"]).item())
expected_indices = np.arange(frames, dtype=np.int64)
for side_index, side in enumerate(("left", "right")):
    path = pd / "hand_processor" / f"hand_data_{side}.npz"
    with np.load(path, allow_pickle=False) as hand:
        if not np.array_equal(hand["frame_indices"], expected_indices):
            raise SystemExit(f"{path}: frame indices mismatch")
        if not np.array_equal(np.asarray(hand["hand_detected"], bool), valid[side_index]):
            raise SystemExit(f"{path}: {side} validity differs from HaWoR")
        if hand["kpts_2d"].shape != (frames, 21, 2):
            raise SystemExit(f"{path}: bad 2-D keypoint shape")
        if hand["kpts_3d"].shape != (frames, 21, 3):
            raise SystemExit(f"{path}: bad 3-D keypoint shape")
        if not np.array_equal(np.asarray(hand["kpts_3d"]), joints[side]):
            raise SystemExit(f"{path}: 3-D keypoints differ from HaWoR")
        xyz = joints[side]
        z = np.clip(xyz[..., 2], 1e-6, None)
        projected = np.stack(
            [focal * xyz[..., 0] / z + 640.0, focal * xyz[..., 1] / z + 360.0],
            axis=-1,
        )
        if not np.allclose(hand["kpts_2d"], projected, rtol=0.0, atol=1e-8):
            raise SystemExit(f"{path}: 2-D projection/focal mismatch")
bbox_path = pd / "bbox_processor" / "bbox_data.npz"
with np.load(bbox_path, allow_pickle=False) as bbox:
    if not np.array_equal(np.asarray(bbox["left_hand_detected"], bool), valid[0]):
        raise SystemExit(f"{bbox_path}: left validity mismatch")
    if not np.array_equal(np.asarray(bbox["right_hand_detected"], bool), valid[1]):
        raise SystemExit(f"{bbox_path}: right validity mismatch")
    for side_index, side in enumerate(("left", "right")):
        boxes = np.asarray(bbox[f"{side}_bboxes"])
        if boxes.shape != (frames, 4):
            raise SystemExit(f"{bbox_path}: {side} bbox shape mismatch")
        if np.any(boxes[~valid[side_index]] != 0):
            raise SystemExit(f"{bbox_path}: invalid {side} frames have nonzero boxes")
print(
    f"[SH injection] exact HaWoR validity L/R="
    f"{int(valid[0].sum())}/{int(valid[1].sum())}"
)
' "$pd" "$hawor" "$frames"
}

validate_mask() {
    local mask="$1"
    local frames="$2"
    local role="$3"
    conda run -n "$ENVIRONMENT" python -c '
import sys
from pathlib import Path
import numpy as np

path = Path(sys.argv[1])
frames, height, width = map(int, sys.argv[2:5])
role = sys.argv[5]
value = np.load(path, mmap_mode="r", allow_pickle=False)
expected = (frames, height, width)
if value.shape != expected or value.dtype != np.bool_:
    raise SystemExit(f"{path}: expected bool {expected}, got {value.shape}/{value.dtype}")
areas = np.count_nonzero(value, axis=(1, 2))
if np.any(areas <= 0):
    raise SystemExit(f"{path}: {int(np.sum(areas <= 0))} empty mask frames")
if np.any(areas >= height * width):
    raise SystemExit(f"{path}: full-frame mask detected")
print(
    f"[{role}] mask frames={frames}/{frames}, nonempty={int(np.sum(areas > 0))}, "
    f"area mean/min/max={float(areas.mean()):.0f}/{int(areas.min())}/{int(areas.max())}"
)
' "$mask" "$frames" "$EXPECTED_HEIGHT" "$EXPECTED_WIDTH" "$role"
}

video_cache_current() {
    local raw_video="$1"
    local processed_video="$2"
    local rgb_dir="$3"
    local frames="$4"
    validate_video "$raw_video" "$frames" >/dev/null 2>&1 || return 1
    validate_video "$processed_video" "$frames" >/dev/null 2>&1 || return 1
    cmp -s "$raw_video" "$processed_video" || return 1
    fresh_file "$raw_video" "$ROOT/src/inpainting/prepare_demo.py" || return 1
    fresh_file "$processed_video" "$raw_video" || return 1
    rgb_tree_not_newer_than "$rgb_dir" "$frames" "$raw_video" || return 1
}

run_prepare() {
    local rgb_dir="$1"
    local raw_root="$2"
    local processed_root="$3"
    local -a args=(
        --input "$rgb_dir"
        --data_root "$raw_root"
        --processed_root "$processed_root"
        --demo_name view
        --demo_num 0
        --fps "$EXPECTED_FPS"
        --glob 'rgb_frame*.jpg'
    )
    [ "$FORCE" = "1" ] && args+=(--overwrite)
    conda run -n "$ENVIRONMENT" --no-capture-output \
        python "$ROOT/src/inpainting/prepare_demo.py" "${args[@]}"
}

ensure_video_cache() {
    local branch="$1"
    local id="$2"
    local rgb_dir="$3"
    local raw_root="$4"
    local processed_root="$5"
    local frames="$6"
    local raw_video="$raw_root/view/0/video_L.mp4"
    local processed_video="$processed_root/view/0/video_L.mp4"
    if [ "$FORCE" = "1" ]; then
        echo "[$branch/$id] FORCE: rebuild SH 24-FPS prepared video"
        run_prepare "$rgb_dir" "$raw_root" "$processed_root"
    elif video_cache_current "$raw_video" "$processed_video" "$rgb_dir" "$frames"; then
        echo "[$branch/$id] SH prepared video is current"
    elif ! any_path_exists "$raw_video" "$processed_video"; then
        echo "[$branch/$id] prepare SH RGB sequence at 24 FPS"
        run_prepare "$rgb_dir" "$raw_root" "$processed_root"
    else
        die "[$branch/$id] existing SH prepared-video cache is stale, partial, or invalid; inspect it and rerun this stage with FORCE=1"
    fi
    validate_video "$raw_video" "$frames" || die "[$branch/$id] invalid raw SH video"
    validate_video "$processed_video" "$frames" || die "[$branch/$id] invalid processed SH video"
    cmp -s "$raw_video" "$processed_video" || \
        die "[$branch/$id] raw/processed SH videos are not byte-identical"
}

injected_cache_current() {
    local pd="$1"
    local hawor="$2"
    local video="$3"
    local frames="$4"
    local left="$pd/hand_processor/hand_data_left.npz"
    local right="$pd/hand_processor/hand_data_right.npz"
    local bbox="$pd/bbox_processor/bbox_data.npz"
    validate_injected "$pd" "$hawor" "$frames" >/dev/null 2>&1 || return 1
    fresh_file "$left" "$hawor" "$video" \
        "$ROOT/src/inpainting/inject_hawor_data.py" || return 1
    fresh_file "$right" "$hawor" "$video" \
        "$ROOT/src/inpainting/inject_hawor_data.py" || return 1
    fresh_file "$bbox" "$hawor" "$video" \
        "$ROOT/src/inpainting/inject_hawor_data.py" || return 1
}

run_inject() {
    local pd="$1"
    local hawor="$2"
    local -a args=(
        --processed_demo "$pd"
        --hawor_npz "$hawor"
        --skip_rgb_copy
    )
    [ "$FORCE" = "1" ] && args+=(--overwrite)
    conda run -n "$ENVIRONMENT" --no-capture-output \
        python "$ROOT/src/inpainting/inject_hawor_data.py" "${args[@]}"
}

ensure_injected_cache() {
    local branch="$1"
    local id="$2"
    local pd="$3"
    local hawor="$4"
    local frames="$5"
    local video="$pd/video_L.mp4"
    local left="$pd/hand_processor/hand_data_left.npz"
    local right="$pd/hand_processor/hand_data_right.npz"
    local bbox="$pd/bbox_processor/bbox_data.npz"
    if [ "$FORCE" = "1" ]; then
        echo "[$branch/$id] FORCE: inject exact SH HaWoR prompts"
        run_inject "$pd" "$hawor"
    elif injected_cache_current "$pd" "$hawor" "$video" "$frames"; then
        echo "[$branch/$id] SH prompt injection is current"
    elif ! any_path_exists "$left" "$right" "$bbox"; then
        echo "[$branch/$id] inject SH HaWoR hand/bbox prompts"
        run_inject "$pd" "$hawor"
    else
        die "[$branch/$id] existing SH prompt cache is stale, partial, or invalid; inspect it and rerun this stage with FORCE=1"
    fi
    validate_injected "$pd" "$hawor" "$frames"
}

mask_cache_current() {
    local mask="$1"
    local frames="$2"
    shift 2
    validate_mask "$mask" "$frames" "SH cached" >/dev/null 2>&1 || return 1
    fresh_file "$mask" "$@" || return 1
}

run_segment() {
    local pd="$1"
    local frames="$2"
    local target="$3"
    [ -s "$SAM2_CHECKPOINT" ] || die "SAM2 checkpoint is missing: $SAM2_CHECKPOINT"
    conda run -n "$ENVIRONMENT" python -c \
        'import torch; assert torch.cuda.is_available()' >/dev/null
    local staging
    staging=$(mktemp -d "$pd/.sh_visibility_segment.XXXXXX")
    (
        trap 'rm -rf -- "$staging"' EXIT
        ln -s "$(realpath "$pd/video_L.mp4")" "$staging/video_L.mp4"
        ln -s "$(realpath "$pd/hand_processor")" "$staging/hand_processor"
        ln -s "$(realpath "$pd/bbox_processor")" "$staging/bbox_processor"
        env MPLCONFIGDIR=/tmp/inpaint-mpl \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            conda run -n "$ENVIRONMENT" --no-capture-output \
            python "$ROOT/src/inpainting/segment_arms.py" \
            --processed_demo "$staging" \
            --output "$staging/masks_arm.npy"
        validate_mask "$staging/masks_arm.npy" "$frames" "SH staged"
        mkdir -p "$(dirname -- "$target")"
        if [ -e "$target" ]; then
            [ "$FORCE" = "1" ] || \
                die "refusing to replace existing SH mask without FORCE=1: $target"
            mv -f -- "$staging/masks_arm.npy" "$target"
        else
            mv -- "$staging/masks_arm.npy" "$target"
        fi
    )
}

ensure_mask_cache() {
    local branch="$1"
    local id="$2"
    local pd="$3"
    local hawor="$4"
    local frames="$5"
    local video="$pd/video_L.mp4"
    local left="$pd/hand_processor/hand_data_left.npz"
    local right="$pd/hand_processor/hand_data_right.npz"
    local bbox="$pd/bbox_processor/bbox_data.npz"
    local mask="$pd/segmentation_processor/masks_arm.npy"
    local -a sources=(
        "$video"
        "$hawor"
        "$left"
        "$right"
        "$bbox"
        "$ROOT/src/inpainting/segment_arms.py"
        "$ROOT/src/inpainting/_paths.py"
        "$SAM2_CHECKPOINT"
    )
    if [ "$FORCE" = "1" ]; then
        echo "[$branch/$id] FORCE: segment SH hand+arm visibility"
        run_segment "$pd" "$frames" "$mask"
    elif mask_cache_current "$mask" "$frames" "${sources[@]}"; then
        echo "[$branch/$id] SH visibility mask is current"
    elif [ ! -e "$mask" ]; then
        echo "[$branch/$id] SAM2 SH hand+arm visibility"
        run_segment "$pd" "$frames" "$mask"
    else
        die "[$branch/$id] existing SH visibility mask is stale or invalid; inspect it and rerun this stage with FORCE=1"
    fi
    validate_mask "$mask" "$frames" "SH visibility"
}

require_video_cache() {
    local branch="$1"
    local id="$2"
    local raw_video="$3"
    local processed_video="$4"
    local rgb_dir="$5"
    local frames="$6"
    video_cache_current "$raw_video" "$processed_video" "$rgb_dir" "$frames" || \
        die "[$branch/$id] SH prepared video is missing/stale; run STAGES=prepare first"
}

require_injected_cache() {
    local branch="$1"
    local id="$2"
    local pd="$3"
    local hawor="$4"
    local frames="$5"
    injected_cache_current "$pd" "$hawor" "$pd/video_L.mp4" "$frames" || \
        die "[$branch/$id] SH prompt cache is missing/stale; run STAGES=inject first"
}

cd "$ROOT"

for BRANCH in "${REQUESTED_BRANCHES[@]}"; do
    if [ "$BRANCH" = "approx" ]; then
        DATA_ROOT="$APPROX_ROOT"
        SH_FOCAL="$APPROX_SH_FOCAL"
        MH_FOCAL="$APPROX_MH_FOCAL"
    else
        DATA_ROOT="$CALIBRATED_ROOT"
        SH_FOCAL="$CALIBRATED_SH_FOCAL"
        MH_FOCAL="$CALIBRATED_MH_FOCAL"
    fi
    [ -d "$DATA_ROOT" ] || die "Missing $BRANCH root: $DATA_ROOT"

    for ID in "${EPISODES[@]}"; do
        EP="$DATA_ROOT/$ID"
        C1="$EP/camera_1"
        C2="$EP/camera_2"
        GT="$EP/gt_labels.json"
        MANIFEST="$EP/stereo_manifest.json"
        C1_RGB="$C1/rgb"
        C2_RGB="$C2/rgb"
        C1_HAWOR="$C1/rgb_hawor/retarget_input.npz"
        C2_HAWOR="$C2/rgb_hawor/retarget_input.npz"
        C1_RAW_ROOT="$C1/visibility/raw"
        C1_PROCESSED_ROOT="$C1/visibility/processed"
        C1_PD="$C1_PROCESSED_ROOT/view/0"
        C1_RAW_VIDEO="$C1_RAW_ROOT/view/0/video_L.mp4"
        C1_VIDEO="$C1_PD/video_L.mp4"
        C1_MASK="$C1_PD/segmentation_processor/masks_arm.npy"
        C2_PD="$C2/inpainting/processed/view/0"
        C2_VIDEO="$C2_PD/video_L.mp4"
        C2_MASK="$C2_PD/segmentation_processor/masks_arm.npy"

        for required in "$GT" "$MANIFEST" "$C1_RGB" "$C2_RGB" \
            "$C1_HAWOR" "$C2_HAWOR" "$C2_VIDEO" "$C2_MASK"; do
            [ -e "$required" ] || die "[$BRANCH/$ID] missing prerequisite: $required"
        done

        EXPECTED=$(validate_manifest \
            "$MANIFEST" "$GT" "$BRANCH" "$SH_FOCAL" "$MH_FOCAL")
        validate_view_inputs \
            "$C1_RGB" "$C1_HAWOR" "$EXPECTED" "$SH_FOCAL" "SH/camera_1"
        validate_view_inputs \
            "$C2_RGB" "$C2_HAWOR" "$EXPECTED" "$MH_FOCAL" "MH/camera_2"
        validate_video "$C2_VIDEO" "$EXPECTED" || \
            die "[$BRANCH/$ID] existing MH inpainting video is not exact $EXPECTED-frame 1280x720@24"
        validate_mask "$C2_MASK" "$EXPECTED" "MH reused (read-only)"
        echo "[$BRANCH/$ID] MH visibility asset validated and will not be modified: $C2_MASK"

        if stage_enabled prepare; then
            ensure_video_cache "$BRANCH" "$ID" "$C1_RGB" \
                "$C1_RAW_ROOT" "$C1_PROCESSED_ROOT" "$EXPECTED"
        fi

        if stage_enabled inject; then
            require_video_cache "$BRANCH" "$ID" "$C1_RAW_VIDEO" \
                "$C1_VIDEO" "$C1_RGB" "$EXPECTED"
            ensure_injected_cache \
                "$BRANCH" "$ID" "$C1_PD" "$C1_HAWOR" "$EXPECTED"
        fi

        if stage_enabled segment; then
            require_video_cache "$BRANCH" "$ID" "$C1_RAW_VIDEO" \
                "$C1_VIDEO" "$C1_RGB" "$EXPECTED"
            require_injected_cache \
                "$BRANCH" "$ID" "$C1_PD" "$C1_HAWOR" "$EXPECTED"
            ensure_mask_cache \
                "$BRANCH" "$ID" "$C1_PD" "$C1_HAWOR" "$EXPECTED"
        fi

        if stage_enabled validate; then
            require_video_cache "$BRANCH" "$ID" "$C1_RAW_VIDEO" \
                "$C1_VIDEO" "$C1_RGB" "$EXPECTED"
            require_injected_cache \
                "$BRANCH" "$ID" "$C1_PD" "$C1_HAWOR" "$EXPECTED"
            mask_cache_current "$C1_MASK" "$EXPECTED" \
                "$C1_VIDEO" "$C1_HAWOR" \
                "$C1_PD/hand_processor/hand_data_left.npz" \
                "$C1_PD/hand_processor/hand_data_right.npz" \
                "$C1_PD/bbox_processor/bbox_data.npz" \
                "$ROOT/src/inpainting/segment_arms.py" \
                "$ROOT/src/inpainting/_paths.py" \
                "$SAM2_CHECKPOINT" || \
                die "[$BRANCH/$ID] SH visibility mask is missing/stale/invalid"
            validate_mask "$C1_MASK" "$EXPECTED" "SH final validation"
        fi

        if [ -s "$C1_MASK" ]; then
            echo "[$BRANCH/$ID] SH visibility mask: $C1_MASK"
        fi
    done
done

echo "Completed 08-05 SH visibility stages '$STAGES' for branches '$BRANCHES'."
