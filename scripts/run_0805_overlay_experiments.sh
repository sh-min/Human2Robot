#!/usr/bin/env bash
set -euo pipefail

# Re-run the 08-04 RB5+XHand overlay ablations on the 08-05 data.
# Camera 2 (MH) always owns the rendered geometry and output pixels. Camera 1
# (SH) is used only for synchronized same-finger HaCo/visibility evidence.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BRANCHES="${BRANCHES:-calibrated,approx}"
STAGES="${STAGES:-visibility_assets,retarget,preview,render,raw,methods,stereo,derived,barrier,compare}"
FORCE="${FORCE:-0}"
RETARGET_ENV="${RETARGET_ENV:-RFM_retarget}"
RENDER_ENV="${RENDER_ENV:-inpaint-gpu}"
RENDER_SCALE="${RENDER_SCALE:-0.75}"
ARM_MODE="${ARM_MODE:-full}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/8-5/overlay_method_comparison}"

cd "$ROOT"

stage_enabled() {
    case ",$STAGES," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

any_stage_enabled() {
    local stage
    for stage in "$@"; do
        if stage_enabled "$stage"; then
            return 0
        fi
    done
    return 1
}

case "$FORCE" in 0|1) ;; *) echo "FORCE must be 0 or 1" >&2; exit 1 ;; esac
case "$ARM_MODE" in full|forearm|distal|wrist|hand_only) ;;
    *) echo "invalid ARM_MODE: $ARM_MODE" >&2; exit 1 ;;
esac

IFS=',' read -r -a REQUESTED_STAGES <<< "$STAGES"
for STAGE in "${REQUESTED_STAGES[@]}"; do
    case "$STAGE" in
        visibility_assets|retarget|preview|render|raw|methods|stereo|derived|barrier|compare) ;;
        *) echo "unknown stage: $STAGE" >&2; exit 1 ;;
    esac
done

IFS=',' read -r -a BRANCH_LIST <<< "$BRANCHES"
for BRANCH in "${BRANCH_LIST[@]}"; do
    case "$BRANCH" in approx|calibrated) ;;
        *) echo "BRANCHES accepts only approx,calibrated" >&2; exit 1 ;;
    esac
done

if [ "$#" -gt 0 ]; then
    EPISODES=("$@")
else
    EPISODES=(1 2)
fi
for ID in "${EPISODES[@]}"; do
    [[ "$ID" =~ ^[0-9]+$ ]] || {
        echo "episode names must be numeric, got '$ID'" >&2
        exit 1
    }
done

if stage_enabled retarget; then
    conda run -n "$RETARGET_ENV" python -c \
        'import numpy, scipy, pinocchio, dex_retargeting'
fi
if stage_enabled preview || stage_enabled render || stage_enabled raw || \
   stage_enabled methods || stage_enabled stereo || stage_enabled derived || \
   stage_enabled barrier || stage_enabled compare; then
    conda run -n "$RENDER_ENV" python -c \
        'import cv2, numpy, pyrender, trimesh; assert cv2.__version__'
fi
RETARGET_PYTHON="$(conda run -n "$RETARGET_ENV" python -c \
    'import sys; print(sys.executable)' | tail -n 1)"
RENDER_PYTHON="$(conda run -n "$RENDER_ENV" python -c \
    'import sys; print(sys.executable)' | tail -n 1)"
test -x "$RETARGET_PYTHON"
test -x "$RENDER_PYTHON"

expected_frames() {
    python - "$1" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1]))["num_frames"]))
PY
}

frame_offset() {
    python - "$1" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1]))["temporal_alignment"]["camera1_frame_offset"]))
PY
}

validate_video() {
    local path="$1" expected="$2" width="${3:-1280}" height="${4:-720}"
    "$RENDER_PYTHON" - "$path" "$expected" "$width" "$height" <<'PY'
import cv2, math, sys
path, expected = sys.argv[1], int(sys.argv[2])
expected_width, expected_height = int(sys.argv[3]), int(sys.argv[4])
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    raise SystemExit(f"cannot open video: {path}")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = float(cap.get(cv2.CAP_PROP_FPS))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
if (width, height, frames) != (expected_width, expected_height, expected):
    raise SystemExit(
        f"invalid video metadata {path}: {(width,height,frames)} != "
        f"{(expected_width,expected_height,expected)}"
    )
if not math.isclose(fps, 24.0, rel_tol=0.0, abs_tol=1e-3):
    raise SystemExit(f"invalid video fps {path}: {fps}")
PY
}

validate_retarget() {
    local pkl="$1" npz="$2" expected="$3"
    "$RETARGET_PYTHON" - "$pkl" "$npz" "$expected" <<'PY'
import pickle, numpy as np, sys
pkl, npz, expected = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(pkl, "rb") as stream:
    data = pickle.load(stream)
with np.load(npz) as source:
    valid = np.asarray(source["valid"])[0].astype(bool)
qpos = np.asarray(data["data"])
if data.get("hand") != "left" or data.get("embodiment", "xhand") != "xhand":
    raise SystemExit("retarget output must be left XHand")
if qpos.shape != (expected, 12) or not np.isfinite(qpos).all():
    raise SystemExit(f"invalid qpos: {qpos.shape}")
if not np.array_equal(np.asarray(data["valid"], dtype=bool), valid):
    raise SystemExit("retarget validity differs from HaWoR")
for key, shape in (("wrist_pos", (expected, 3)), ("wrist_quat", (expected, 4))):
    value = np.asarray(data[key])
    if value.shape != shape or not np.isfinite(value).all():
        raise SystemExit(f"invalid {key}: {value.shape}")
PY
}

validate_overlay_input() {
    local npz="$1" jn="$2" expected="$3" focal="$4"
    "$RETARGET_PYTHON" - "$npz" "$jn" "$expected" "$focal" <<'PY'
import json, math, numpy as np, sys
path, names_path, expected, focal = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
with np.load(path) as data:
    shapes = {
        "rb5_q": (expected, 6), "wrist_pos": (expected, 3),
        "wrist_rot": (expected, 3, 3), "qpos": (expected, 12),
        "valid": (expected,), "T_cam_base": (4, 4),
    }
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise SystemExit(f"{key} shape {data[key].shape} != {shape}")
    if str(np.asarray(data["side"]).item()) != "left":
        raise SystemExit("overlay input side must be left")
    if int(data["img_width"]) != 1280 or int(data["img_height"]) != 720:
        raise SystemExit("overlay input source size must be 1280x720")
    if not math.isclose(float(data["img_focal"]), focal, rel_tol=0.0, abs_tol=1e-3):
        raise SystemExit("overlay input focal mismatch")
meta = json.load(open(names_path))
if meta.get("side") != "left" or len(meta.get("joint_names", [])) != 12:
    raise SystemExit("invalid overlay joint-name sidecar")
PY
}

validate_overlay() {
    local directory="$1" expected="$2" focal="$3"
    "$RENDER_PYTHON" - "$directory" "$expected" "$focal" \
        "$RENDER_SCALE" "$ARM_MODE" <<'PY'
import json, math, numpy as np, sys
from pathlib import Path
root, expected, focal = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
scale, arm_mode = float(sys.argv[4]), sys.argv[5]
render_w, render_h = int(round(1280 * scale)), int(round(720 * scale))
render_focal = focal * render_w / 1280
report = json.load(open(root / "manifest.json"))
checks = {
    "side": "left", "frame_count": expected, "source_size": [1280,720],
    "render_size": [render_w,render_h], "arm_mode": arm_mode,
}
for key, value in checks.items():
    if report.get(key) != value:
        raise SystemExit(f"overlay manifest {key}: {report.get(key)!r} != {value!r}")
if not math.isclose(float(report["img_focal"]), render_focal, rel_tol=0.0, abs_tol=1e-3):
    raise SystemExit("overlay focal mismatch")
t, h, w = expected, render_h, render_w
spec = {
    "robot_rgb.npy": ((t,h,w,3), np.uint8),
    "robot_depth.npy": ((t,h,w), np.float16),
    "robot_mask.npy": ((t,h,w), np.bool_),
    "robot_finger_labels.npy": ((t,h,w), np.uint8),
    "robot_finger_surface_labels.npy": ((t,h,w), np.uint8),
    "robot_finger_mask.npy": ((t,h,w), np.bool_),
    "robot_hand_mask.npy": ((t,h,w), np.bool_),
}
arrays = {}
for name, (shape, dtype) in spec.items():
    value = np.load(root / name, mmap_mode="r")
    if value.shape != shape or value.dtype != dtype:
        raise SystemExit(f"invalid {name}: {value.shape}/{value.dtype}")
    arrays[name] = value
finger = arrays["robot_finger_mask.npy"]
hand = arrays["robot_hand_mask.npy"]
robot = arrays["robot_mask.npy"]
labels = arrays["robot_finger_labels.npy"]
surface = arrays["robot_finger_surface_labels.npy"]
for index in range(t):
    if np.any(finger[index] & ~hand[index]) or np.any(hand[index] & ~robot[index]):
        raise SystemExit("finger/hand/robot subset invariant failed")
    packed = np.asarray(surface[index])
    decoded = np.zeros_like(packed)
    active = packed > 0
    decoded[active] = ((packed[active].astype(np.int16)-1)//3+1).astype(np.uint8)
    if not np.array_equal(decoded, labels[index]):
        raise SystemExit("surface labels do not decode to finger labels")
PY
}

validate_contact() {
    local directory="$1" expected="$2" mode="$3" dual="$4" scale="$5" expand="$6" force_surface="$7" gap="$8" alignment="$9"
    "$RENDER_PYTHON" - \
        "$directory" "$expected" "$mode" "$dual" "$scale" "$expand" \
        "$force_surface" "$gap" "$alignment" "$OFFSET" "$BACKGROUND" \
        "$VIDEO" "$HAWOR" "$C2/contact" "$C1/contact" "$OVERLAY" \
        "$MODAL" "$RESTORE" "$SCENE_DEPTH" "$SURFACE" <<'PY' || return 1
import json, math, numpy as np, sys
from pathlib import Path
root = Path(sys.argv[1])
expected, mode = int(sys.argv[2]), sys.argv[3]
dual, scale, expand = bool(int(sys.argv[4])), float(sys.argv[5]), int(sys.argv[6])
force_surface, gap = bool(int(sys.argv[7])), int(sys.argv[8])
alignment, offset = sys.argv[9], int(sys.argv[10])
expected_sources = {
    "background": Path(sys.argv[11]).resolve(),
    "raw_video": Path(sys.argv[12]).resolve(),
    "hawor_npz": Path(sys.argv[13]).resolve(),
    "contact_dir": Path(sys.argv[14]).resolve(),
    "overlay_dir": Path(sys.argv[16]).resolve(),
    "object_mask": Path(sys.argv[17]).resolve(),
    "object_restore_mask": Path(sys.argv[18]).resolve(),
}
expected_aux = Path(sys.argv[15]).resolve()
expected_scene = Path(sys.argv[19]).resolve()
expected_surface = Path(sys.argv[20]).resolve()
report = json.load(open(root / "report.json"))
if report.get("frames") != expected or report.get("occlusion_mode") != mode:
    raise SystemExit("contact report frame/mode mismatch")
if report.get("side") != "left" or report.get("invariants", {}).get("occluded_subset_of_robot_fingers") is not True:
    raise SystemExit("contact finger invariant missing")
sources = report.get("sources", {})
if bool(sources.get("aux_contact_dir")) != dual:
    raise SystemExit("contact dual-camera source mismatch")
for key, expected_path in expected_sources.items():
    actual = sources.get(key)
    if actual is None or Path(actual).resolve() != expected_path:
        raise SystemExit(f"contact source mismatch for {key}: {actual}")
if dual:
    if Path(sources["aux_contact_dir"]).resolve() != expected_aux:
        raise SystemExit("contact auxiliary source mismatch")
    if int(report.get("aux_frame_offset", 0)) != offset:
        raise SystemExit("contact auxiliary frame offset mismatch")
else:
    if int(report.get("aux_frame_offset", 0)) != 0:
        raise SystemExit("MH-only result unexpectedly has an auxiliary offset")
if dual and report.get("invariants", {}).get("auxiliary_geometry_used") is not False:
    raise SystemExit("SH geometry must never be used")
for key in (
    "raw_rgb_restore_uses_object_restore_mask_only",
    "object_restore_mask_subset_of_object_mask",
):
    if report.get("invariants", {}).get(key) is not True:
        raise SystemExit(f"clean object restoration invariant missing: {key}")
config = report.get("config", {})
if not math.isclose(float(config.get("contact_depth_thickness_scale", 0.0)), scale, abs_tol=1e-9):
    raise SystemExit("contact thickness scale mismatch")
if int(config.get("contact_interior_expand_px", 0)) != expand:
    raise SystemExit("contact interior expansion mismatch")
if expand and not math.isclose(
    float(config.get("contact_interior_expand_cap_fraction", -1.0)),
    0.25,
    abs_tol=1e-9,
):
    raise SystemExit("contact interior expansion cap mismatch")
actual_scene = sources.get("scene_depth")
if mode == "ensemble":
    if actual_scene is None or Path(actual_scene).resolve() != expected_scene:
        raise SystemExit("scalar object-Z scene-depth source mismatch")
elif actual_scene is not None:
    raise SystemExit("unexpected scene-depth source")
if mode == "object3d":
    surface_source = sources.get("object_surface_depth")
    if surface_source is None or Path(surface_source).resolve() != expected_surface:
        raise SystemExit("2.5D object-surface source mismatch")
    if report.get("object_surface_3d", {}).get("alignment") != alignment:
        raise SystemExit("2.5D object-surface alignment mismatch")
    control = report.get("object3d_penetration_control", {})
    surface = control.get("surface_force", {})
    temporal = control.get("temporal_filter", {})
    if bool(surface.get("enabled", False)) != force_surface:
        raise SystemExit("surface-force mode mismatch")
    if int(temporal.get("max_gap_frames", 0)) != gap:
        raise SystemExit("object3d temporal-gap mismatch")
    if force_surface and not math.isclose(
        float(surface.get("force_margin_m", float("nan"))), 0.0, abs_tol=1e-9
    ):
        raise SystemExit("object3d force margin mismatch")
    if gap:
        if int(temporal.get("motion_radius_px_per_frame", -1)) != 6:
            raise SystemExit("object3d temporal motion radius mismatch")
        if not math.isclose(
            float(temporal.get("front_slack_m", float("nan"))),
            0.015,
            abs_tol=1e-9,
        ):
            raise SystemExit("object3d temporal front slack mismatch")
mask = np.load(root / "occluded_finger_mask.npy", mmap_mode="r")
if mask.shape != (expected, 720, 1280) or mask.dtype != np.bool_:
    raise SystemExit(f"invalid contact mask {mask.shape}/{mask.dtype}")
per_frame = np.asarray(
    [int(np.count_nonzero(mask[index])) for index in range(expected)],
    dtype=np.int64,
)
if report.get("occluded_pixel_count") != per_frame.tolist():
    raise SystemExit("contact report per-frame mask counts mismatch")
if int(report.get("occluded_pixels_total", -1)) != int(per_frame.sum()):
    raise SystemExit("contact report total mask count mismatch")
if int(report.get("frames_with_occlusion", -1)) != int(np.count_nonzero(per_frame)):
    raise SystemExit("contact report occluded-frame count mismatch")
for name in ("video_overlay_contact.mp4", "video_robot_only_contact.mp4"):
    if not (root / name).is_file():
        raise SystemExit(f"missing contact video: {name}")
PY
    validate_video "$directory/video_overlay_contact.mp4" "$expected"
}

validate_stereo() {
    local directory="$1" expected="$2" offset="$3"
    "$RENDER_PYTHON" - \
        "$directory" "$expected" "$offset" "$C1/rgb" "$C2/rgb" \
        "$C1/rgb_hawor/retarget_input.npz" "$HAWOR" \
        "$C1_VISIBLE" "$C2_VISIBLE" "$C1/contact" "$C2/contact" \
        "$BACKGROUND" "$OVERLAY" "$MODAL" "$RESTORE" <<'PY' || return 1
import json, numpy as np, sys
from pathlib import Path
root, expected, offset = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
expected_sources = {
    "camera1_rgb_dir": Path(sys.argv[4]).resolve(),
    "camera2_rgb_dir": Path(sys.argv[5]).resolve(),
    "camera1_hawor": Path(sys.argv[6]).resolve(),
    "camera2_hawor": Path(sys.argv[7]).resolve(),
    "camera1_visible_mask": Path(sys.argv[8]).resolve(),
    "camera2_visible_mask": Path(sys.argv[9]).resolve(),
    "camera1_contact_dir": Path(sys.argv[10]).resolve(),
    "camera2_contact_dir": Path(sys.argv[11]).resolve(),
    "background": Path(sys.argv[12]).resolve(),
    "overlay_dir": Path(sys.argv[13]).resolve(),
    "object_mask": Path(sys.argv[14]).resolve(),
    "object_restore_mask": Path(sys.argv[15]).resolve(),
}
report = json.load(open(root / "report.json"))
if report.get("frames") != expected or report.get("camera2_is_final_view") is not True:
    raise SystemExit("stereo report frame mismatch")
required = {"visibility", "haco_only", "visibility_haco"}
if not required <= set(report.get("output_modes", [])):
    raise SystemExit("stereo report modes missing")
alignment = report.get("temporal_alignment", {})
if alignment.get("camera1_frame_offset") != offset:
    raise SystemExit("stereo report offset mismatch")
if report.get("invariants", {}).get("dual_haco_uses_max_available_score") is not True:
    raise SystemExit("stereo dual-HaCo invariant missing")
for key in (
    "geometry_uses_modal_object_mask",
    "raw_rgb_restore_uses_object_restore_mask_only",
    "object_restore_mask_is_subset_of_modal_object_mask",
    "haco_only_ignores_visibility_and_depth",
):
    if report.get("invariants", {}).get(key) is not True:
        raise SystemExit(f"stereo invariant missing: {key}")
sources = report.get("sources", {})
for key, expected_path in expected_sources.items():
    actual = sources.get(key)
    if actual is None or Path(actual).resolve() != expected_path:
        raise SystemExit(f"stereo source mismatch for {key}: {actual}")
if Path(sources.get("contact_dir", "")).resolve() != expected_sources["camera2_contact_dir"]:
    raise SystemExit("stereo camera-2 contact alias mismatch")
if report.get("sides", {}).get("rendered") != "left" or report.get("sides", {}).get("camera2") != "left":
    raise SystemExit("stereo MH/rendered side mismatch")
counts = report.get("object_pixel_counts", {})
if int(counts.get("restore_pixels_outside_modal_total", -1)) != 0:
    raise SystemExit("stereo clean restore is not a modal-mask subset")
for mode in ("visibility", "haco_only", "visibility_haco"):
    mask = np.load(root / f"occluded_finger_mask_{mode}.npy", mmap_mode="r")
    if mask.shape != (expected,720,1280) or mask.dtype != np.bool_:
        raise SystemExit(f"invalid stereo {mode} mask")
PY
    validate_video "$directory/video_overlay_visibility_haco.mp4" "$expected"
}

validate_derived() {
    local directory="$1" expected="$2"
    "$RENDER_PYTHON" - \
        "$directory" "$expected" "$HACO_DUAL" "$HACO_HALF" \
        "$HACO_FULL" "$STEREO" "$OVERLAY" "$MODAL" "$RESTORE" \
        "$BACKGROUND" "$VIDEO" <<'PY' || return 1
import json, numpy as np, sys
from pathlib import Path
root, expected = Path(sys.argv[1]), int(sys.argv[2])
source_dirs = {
    "baseline": Path(sys.argv[3]).resolve(),
    "half_thickness": Path(sys.argv[4]).resolve(),
    "full_thickness": Path(sys.argv[5]).resolve(),
    "visibility_force": Path(sys.argv[6]).resolve(),
}
expected_paths = {
    "overlay_dir": Path(sys.argv[7]).resolve(),
    "object_mask": Path(sys.argv[8]).resolve(),
    "object_restore_mask": Path(sys.argv[9]).resolve(),
    "background": Path(sys.argv[10]).resolve(),
    "raw_video": Path(sys.argv[11]).resolve(),
}
report = json.load(open(root / "comparison_report.json"))
if report.get("comparison") != "xhand_thickness_strategies" or report.get("frames") != expected:
    raise SystemExit("derived strategy report metadata mismatch")
sources = report.get("sources", {})
for role, expected_path in source_dirs.items():
    actual = sources.get(role, {}).get("directory")
    if actual is None or Path(actual).resolve() != expected_path:
        raise SystemExit(f"derived source directory mismatch for {role}: {actual}")
for key, expected_path in expected_paths.items():
    actual = sources.get(key)
    if actual is None or Path(actual).resolve() != expected_path:
        raise SystemExit(f"derived source mismatch for {key}: {actual}")
for key in (
    "union_equals_baseline_or_force",
    "diagnostic_shell_is_union_superset",
    "object_restore_mask_subset_of_modal_object",
    "surface_labels_decode_to_finger_labels",
    "surface_side_half_uses_baseline_except_side_half",
    "surface_weighted_uses_front_zero_side_half_back_full",
):
    if report.get("invariants", {}).get(key) is not True:
        raise SystemExit(f"derived invariant missing: {key}")
for name in (
    "occluded_finger_mask_baseline_force_union.npy",
    "occluded_finger_mask_union_safety_shell_diagnostic.npy",
    "occluded_finger_mask_surface_front_side_half.npy",
    "occluded_finger_mask_surface_front_side_half_back_full.npy",
):
    mask = np.load(root / name, mmap_mode="r")
    if mask.shape != (expected, 720, 1280) or mask.dtype != np.bool_:
        raise SystemExit(f"invalid derived mask {name}: {mask.shape}/{mask.dtype}")
PY
    validate_video "$directory/video_compare_xhand_thickness_strategies_3x2.mp4" "$expected" 3840 1440
    validate_video "$directory/video_compare_xhand_surface_strategies_2x2.mp4" "$expected" 2560 1440
}

validate_barrier() {
    local directory="$1" expected="$2"
    "$RENDER_PYTHON" - "$directory" "$expected" <<'PY' || return 1
import json, numpy as np, sys
from pathlib import Path
root, expected = Path(sys.argv[1]), int(sys.argv[2])
report = json.load(open(root / "report.json"))
if report.get("frames") != expected or report.get("method") != "visual_camera_z_xhand_barrier":
    raise SystemExit("barrier report metadata mismatch")
required = (
    "baseline_subset_final", "final_occlusion_subset_of_xhand",
    "rb5_arm_excluded", "valid_surface_barrier_residual_is_zero",
    "trajectory_arrays_unchanged",
)
for key in required:
    if report.get("invariants", {}).get(key) is not True:
        raise SystemExit(f"barrier invariant missing: {key}")
if report.get("counts", {}).get("residual_violation_pixels") != 0:
    raise SystemExit("barrier retained valid surface violations")
mask = np.load(root / "occluded_hand_mask.npy", mmap_mode="r")
if mask.shape != (expected,720,1280) or mask.dtype != np.bool_:
    raise SystemExit("invalid whole-XHand barrier mask")
PY
    validate_video "$directory/video_overlay_hand_barrier.mp4" "$expected"
}

validate_comparison() {
    local directory="$1" expected="$2" approx_pd="$3" calibrated_pd="$4"
    "$RENDER_PYTHON" - \
        "$directory" "$expected" "$approx_pd" "$calibrated_pd" <<'PY' || return 1
import cv2, json, math, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
expected = int(sys.argv[2])
pd_by_branch = {
    "approx": Path(sys.argv[3]).resolve(),
    "calibrated": Path(sys.argv[4]).resolve(),
}
report_path = root / "comparison_report.json"
report = json.load(open(report_path))
if report.get("camera2_is_final_view") is not True:
    raise SystemExit("comparison final-view contract mismatch")
if report.get("dual_camera", {}).get("rendered") is not True:
    raise SystemExit("dual-camera grid was not rendered")
if report.get("extended_history", {}).get("rendered") is not True:
    raise SystemExit("required 4x4 history grid was not rendered")
if any(value is not True for value in report.get("invariants", {}).values()):
    raise SystemExit("comparison report contains a failed invariant")

dependencies = []
sources = report.get("sources", {})
for branch, pd in pd_by_branch.items():
    branch_sources = sources.get(branch, {})
    required_count = 4 if branch == "approx" else 16
    if len(branch_sources) < required_count:
        raise SystemExit(f"comparison branch {branch} is missing variants")
    for variant, record in branch_sources.items():
        directory = Path(record["directory"]).resolve()
        if not directory.is_relative_to(pd):
            raise SystemExit(
                f"comparison source escaped {branch} processed demo: "
                f"{variant} -> {directory}"
            )
        for key in ("video", "mask", "report"):
            value = record.get(key)
            if value is None:
                continue
            path = Path(value).resolve()
            if not path.is_file() or not path.is_relative_to(pd):
                raise SystemExit(
                    f"invalid comparison source {branch}/{variant}/{key}: "
                    f"{path}"
                )
            dependencies.append(path)
        metadata = record.get("metadata", {})
        if int(metadata.get("frames", -1)) != expected or not math.isclose(
            float(metadata.get("fps", 0.0)), 24.0, abs_tol=1e-3
        ):
            raise SystemExit(
                f"comparison input metadata mismatch: {branch}/{variant}"
            )

controlled = report.get("controlled_inputs", {})
for value in controlled.get("stereo_manifests", {}).values():
    dependencies.append(Path(value).resolve())
for key in (
    "approx_hawor", "calibrated_hawor",
    "approx_sh_hawor", "calibrated_sh_hawor",
):
    value = controlled.get(key)
    if value is not None:
        dependencies.append(Path(value).resolve())
if any(not path.is_file() for path in dependencies):
    raise SystemExit("comparison dependency disappeared")
if dependencies and report_path.stat().st_mtime_ns < max(
    path.stat().st_mtime_ns for path in dependencies
):
    raise SystemExit("comparison is older than one of its inputs")

videos = {
    "calibrated_methods_3x2": "video_compare_calibrated_methods_3x2.mp4",
    "camera_calibration_4x2": "video_compare_camera_calibration_4x2.mp4",
    "dual_camera_3x2": "video_compare_dual_camera_3x2.mp4",
    "overlay_history_4x4": "video_compare_overlay_history_4x4.mp4",
}
rendered = report.get("rendered_metadata", {})
for key, name in videos.items():
    expected_meta = rendered.get(key, {})
    path = root / name
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open comparison video: {path}")
    actual = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        float(capture.get(cv2.CAP_PROP_FPS)),
    )
    capture.release()
    wanted = (
        int(expected_meta.get("width", -1)),
        int(expected_meta.get("height", -1)),
        expected,
    )
    if actual[:3] != wanted or not math.isclose(
        actual[3], 24.0, abs_tol=1e-3
    ):
        raise SystemExit(
            f"comparison video metadata mismatch {name}: {actual} != {wanted}"
        )
PY
}

run_contact_variant() {
    local label="$1" directory="$2" expected="$3" mode="$4" dual="$5" scale="$6" expand="$7" force_surface="$8" gap="$9" alignment="${10}"
    shift 10
    if [ "$FORCE" != "1" ] && [ -s "$directory/report.json" ] && \
       validate_contact "$directory" "$expected" "$mode" "$dual" "$scale" "$expand" "$force_surface" "$gap" "$alignment"; then
        echo "    [skip] $label"
        return
    fi
    echo "    [run] $label"
    PYTHONPATH="$ROOT/src/inpainting" \
    conda run -n "$RENDER_ENV" --no-capture-output \
        python "$ROOT/src/inpainting/composite_rb5_contact_occlusion.py" \
        "${CONTACT_COMMON[@]}" "$@" --out_dir "$directory"
    validate_contact "$directory" "$expected" "$mode" "$dual" "$scale" "$expand" "$force_surface" "$gap" "$alignment"
}

if stage_enabled visibility_assets; then
    if [ ! -f "$ROOT/scripts/run_0805_stereo_visibility_assets.sh" ]; then
        echo "missing visibility asset runner" >&2
        exit 1
    fi
    BRANCHES="$BRANCHES" FORCE="$FORCE" ENVIRONMENT="$RENDER_ENV" \
        STAGES=all \
        bash "$ROOT/scripts/run_0805_stereo_visibility_assets.sh" \
        "${EPISODES[@]}"
fi

if any_stage_enabled retarget preview render raw methods stereo derived barrier; then
for BRANCH in "${BRANCH_LIST[@]}"; do
    DATA="$ROOT/data/cube_dataset/26.08.05_stereo_${BRANCH}"
    case "$BRANCH" in
        approx) MH_FOCAL="924.4444580078125" ;;
        calibrated) MH_FOCAL="1030.2115914516535" ;;
    esac
    test -d "$DATA"

    for ID in "${EPISODES[@]}"; do
        EP="$DATA/$ID"
        C1="$EP/camera_1"
        C2="$EP/camera_2"
        PD="$C2/inpainting/processed/view/0"
        HAWOR="$C2/rgb_hawor/retarget_input.npz"
        GT="$EP/gt_labels.json"
        STEREO_MANIFEST="$EP/stereo_manifest.json"
        EXPECTED="$(expected_frames "$GT")"
        OFFSET="$(frame_offset "$STEREO_MANIFEST")"
        VIDEO="$PD/video_L.mp4"
        COMPLETION="$PD/object_completion_dual_haco_e2fgvi"
        BACKGROUND="$COMPLETION/video_object_completed.mp4"
        MODAL="$PD/object_layer/object_mask_modal.npy"
        AMODAL="$COMPLETION/object_mask_amodal.npy"
        RESTORE="$COMPLETION/object_mask_observed_clean.npy"
        SCENE_DEPTH="$PD/depth_processor/depth_aligned_metric.npy"
        SURFACE="$COMPLETION/object_surface_depth_completed.npy"
        PKL="$C2/rgb_hawor/qpos_xhand_contact_left_smooth.pkl"
        OVERLAY_INPUT="$PD/rb5_overlay_input_left.npz"
        JOINT_NAMES="$PD/rb5_overlay_input_left_jointnames.json"
        PREVIEW="$PD/rb5_preview.png"
        OVERLAY="$PD/overlay_processor"
        RAW_DIR="$PD/overlay_method_raw"
        HACO_MH="$PD/overlay_haco_mh"
        HACO_DUAL="$PD/overlay_haco_dual"
        HACO_HALF="$PD/overlay_haco_dual_xhand_half"
        HACO_FULL="$PD/overlay_haco_dual_xhand_full"
        HACO_FILL="$PD/overlay_haco_dual_boundary_fill"
        OBJECT_SCALAR="$PD/overlay_object_scalar_dual"
        OBJECT_UNALIGNED="$PD/overlay_object3d_dual_unaligned"
        OBJECT_ALIGNED="$PD/overlay_object3d_dual_aligned"
        FORCE_TEMPORAL="$PD/overlay_object3d_force_temporal"
        STEREO="$PD/overlay_stereo_visibility"
        C1_VISIBLE="$C1/visibility/processed/view/0/segmentation_processor/masks_arm.npy"
        C2_VISIBLE="$PD/segmentation_processor/masks_arm.npy"
        SURFACE_DERIVED="$PD/overlay_xhand_surface_strategies"
        BARRIER="$PD/overlay_best_inpaint_barrier"

        echo
        echo "[$BRANCH/$ID] $EXPECTED frames, SH offset=$OFFSET"
        for REQUIRED in "$HAWOR" "$GT" "$STEREO_MANIFEST" "$VIDEO" \
            "$BACKGROUND" "$MODAL" "$AMODAL" "$RESTORE" "$SCENE_DEPTH" \
            "$SURFACE" "$C1/contact" "$C2/contact"; do
            [ -e "$REQUIRED" ] || { echo "missing: $REQUIRED" >&2; exit 1; }
        done
        validate_video "$VIDEO" "$EXPECTED"
        validate_video "$BACKGROUND" "$EXPECTED"

        if stage_enabled retarget; then
            if [ "$FORCE" != "1" ] && [ -s "$PKL" ] && \
               validate_retarget "$PKL" "$HAWOR" "$EXPECTED"; then
                echo "    [skip] contact-aware XHand retarget"
            else
                echo "    [run] contact-aware XHand retarget"
                RETARGET_TMP="$(mktemp -d "$C2/rgb_hawor/.overlay_retarget.XXXXXX")"
                conda run -n "$RETARGET_ENV" --no-capture-output \
                    python "$ROOT/src/retargeting/retarget_from_npz.py" \
                    --npz "$HAWOR" --hand left --out_dir "$RETARGET_TMP" \
                    --contact --contact_dir "$C2/contact" --smooth
                validate_retarget "$RETARGET_TMP/qpos_xhand_contact_left_smooth.pkl" "$HAWOR" "$EXPECTED"
                python - "$RETARGET_TMP/qpos_xhand_contact_left_smooth.pkl" "$PKL" <<'PY'
import os, sys
os.replace(sys.argv[1], sys.argv[2])
PY
                rm -r -- "$RETARGET_TMP"
            fi

            if [ "$FORCE" != "1" ] && [ -s "$OVERLAY_INPUT" ] && [ -s "$JOINT_NAMES" ] && \
               validate_overlay_input "$OVERLAY_INPUT" "$JOINT_NAMES" "$EXPECTED" "$MH_FOCAL"; then
                echo "    [skip] RB5 IK adapter"
            else
                echo "    [run] RB5 IK adapter"
                INPUT_TMP="$(mktemp -d "$PD/.overlay_input.XXXXXX")"
                conda run -n "$RETARGET_ENV" --no-capture-output \
                    python "$ROOT/src/inpainting/rb5_build_overlay_input.py" \
                    --hawor_npz "$HAWOR" --pkl "$PKL" --side left \
                    --out "$INPUT_TMP/rb5_overlay_input_left.npz" \
                    --img_w 1280 --img_h 720
                validate_overlay_input \
                    "$INPUT_TMP/rb5_overlay_input_left.npz" \
                    "$INPUT_TMP/rb5_overlay_input_left_jointnames.json" \
                    "$EXPECTED" "$MH_FOCAL"
                python - "$INPUT_TMP" "$PD" <<'PY'
import os, sys
source, target = sys.argv[1], sys.argv[2]
for name in ("rb5_overlay_input_left.npz", "rb5_overlay_input_left_jointnames.json"):
    os.replace(os.path.join(source, name), os.path.join(target, name))
PY
                rm -r -- "$INPUT_TMP"
            fi
        fi
        if any_stage_enabled retarget preview render raw methods stereo derived barrier; then
            validate_retarget "$PKL" "$HAWOR" "$EXPECTED"
            validate_overlay_input "$OVERLAY_INPUT" "$JOINT_NAMES" "$EXPECTED" "$MH_FOCAL"
        fi

        if stage_enabled preview; then
            if [ "$FORCE" != "1" ] && [ -s "$PREVIEW" ]; then
                echo "    [skip] placement preview"
            else
                echo "    [run] placement preview"
                PREVIEW_ARGS=(
                    --data "$OVERLAY_INPUT" --jn "$JOINT_NAMES" --out "$PD"
                    --background "$BACKGROUND" --render_scale "$RENDER_SCALE"
                    --arm_mode "$ARM_MODE" --preview --start 0 --n 6
                )
                [ "$FORCE" = "1" ] && PREVIEW_ARGS+=(--overwrite)
                PYOPENGL_PLATFORM=egl \
                conda run -n "$RENDER_ENV" --no-capture-output \
                    python "$ROOT/src/inpainting/render_rb5_pyrender_overlay.py" \
                    "${PREVIEW_ARGS[@]}"
            fi
            test -s "$PREVIEW"
            echo "    preview: $PREVIEW"
        fi

        if stage_enabled render; then
            if [ "$FORCE" != "1" ] && [ -s "$OVERLAY/manifest.json" ] && \
               validate_overlay "$OVERLAY" "$EXPECTED" "$MH_FOCAL"; then
                echo "    [skip] RB5+XHand render arrays"
            else
                echo "    [run] RB5+XHand render arrays"
                RENDER_ARGS=(
                    --data "$OVERLAY_INPUT" --jn "$JOINT_NAMES" --out "$PD"
                    --background "$BACKGROUND" --render_scale "$RENDER_SCALE"
                    --arm_mode "$ARM_MODE"
                )
                [ "$FORCE" = "1" ] && RENDER_ARGS+=(--overwrite)
                PYOPENGL_PLATFORM=egl \
                conda run -n "$RENDER_ENV" --no-capture-output \
                    python "$ROOT/src/inpainting/render_rb5_pyrender_overlay.py" \
                    "${RENDER_ARGS[@]}"
            fi
        fi
        if any_stage_enabled render raw methods stereo derived barrier; then
            validate_overlay "$OVERLAY" "$EXPECTED" "$MH_FOCAL"
        fi

        if stage_enabled raw; then
            mkdir -p "$RAW_DIR"
            if [ "$FORCE" != "1" ] && \
               validate_video "$RAW_DIR/video_overlay_robot_raw.mp4" "$EXPECTED" && \
               validate_video "$RAW_DIR/video_robot_only.mp4" "$EXPECTED"; then
                echo "    [skip] no-occlusion overlay"
            else
                echo "    [run] no-occlusion overlay"
                conda run -n "$RENDER_ENV" --no-capture-output \
                    python "$ROOT/src/inpainting/composite_robot_unclipped.py" \
                    --processed_demo "$PD" --background "$BACKGROUND" \
                    --out "$RAW_DIR/video_overlay_robot_raw.mp4" \
                    --robot_only_out "$RAW_DIR/video_robot_only.mp4" --fps 24
            fi
        fi

        CONTACT_COMMON=(
            --processed_demo "$PD" --episode_dir "$C2"
            --background "$BACKGROUND" --raw_video "$VIDEO"
            --hawor_npz "$HAWOR" --contact_dir "$C2/contact"
            --overlay_dir "$OVERLAY" --object_mask "$MODAL"
            --object_restore_mask "$RESTORE"
        )
        DUAL_ARGS=(
            --aux_contact_dir "$C1/contact" --aux_frame_offset "$OFFSET"
            --aux_side left
        )

        if stage_enabled methods; then
            run_contact_variant "MH HaCo" "$HACO_MH" "$EXPECTED" haco 0 0 0 0 0 na \
                --occlusion_mode haco
            run_contact_variant "MH+SH HaCo" "$HACO_DUAL" "$EXPECTED" haco 1 0 0 0 0 na \
                "${DUAL_ARGS[@]}" --occlusion_mode haco
            run_contact_variant "dual HaCo + XHand half thickness" "$HACO_HALF" "$EXPECTED" haco 1 0.5 0 0 0 na \
                "${DUAL_ARGS[@]}" --occlusion_mode haco \
                --contact_depth_thickness_scale 0.5
            run_contact_variant "dual HaCo + XHand full thickness" "$HACO_FULL" "$EXPECTED" haco 1 1.0 0 0 0 na \
                "${DUAL_ARGS[@]}" --occlusion_mode haco \
                --contact_depth_thickness_scale 1.0
            run_contact_variant "dual HaCo + boundary interior fill" "$HACO_FILL" "$EXPECTED" haco 1 0 3 0 0 na \
                "${DUAL_ARGS[@]}" --occlusion_mode haco \
                --contact_interior_expand_px 3 \
                --contact_interior_expand_cap_fraction 0.25
            run_contact_variant "dual HaCo + scalar object-Z" "$OBJECT_SCALAR" "$EXPECTED" ensemble 1 0 0 0 0 na \
                "${DUAL_ARGS[@]}" --scene_depth "$SCENE_DEPTH" \
                --object_depth_mask "$MODAL" --occlusion_mode ensemble
            run_contact_variant "dual HaCo + dense 2.5D surface" "$OBJECT_UNALIGNED" "$EXPECTED" object3d 1 0 0 0 0 none \
                "${DUAL_ARGS[@]}" --object_surface_depth "$SURFACE" \
                --object_surface_alignment none --occlusion_mode object3d
            run_contact_variant "dual HaCo + contact-aligned 2.5D surface" "$OBJECT_ALIGNED" "$EXPECTED" object3d 1 0 0 0 0 contact \
                "${DUAL_ARGS[@]}" --object_surface_depth "$SURFACE" \
                --object_surface_alignment contact --occlusion_mode object3d
            run_contact_variant "2.5D surface force + temporal suppression" "$FORCE_TEMPORAL" "$EXPECTED" object3d 1 0 0 1 2 contact \
                "${DUAL_ARGS[@]}" --object_surface_depth "$SURFACE" \
                --object_surface_alignment contact --occlusion_mode object3d \
                --object3d_force_surface --object3d_force_margin_m 0 \
                --object3d_temporal_max_gap_frames 2 \
                --object3d_temporal_motion_px 6 \
                --object3d_temporal_front_slack_m 0.015
        fi

        if stage_enabled stereo; then
            test -s "$C1_VISIBLE"
            test -s "$C2_VISIBLE"
            if [ "$FORCE" != "1" ] && [ -s "$STEREO/report.json" ] && \
               validate_stereo "$STEREO" "$EXPECTED" "$OFFSET"; then
                echo "    [skip] SH visibility + dual-HaCo"
            else
                echo "    [run] SH visibility + dual-HaCo"
                PYTHONPATH="$ROOT/src/inpainting" \
                conda run -n "$RENDER_ENV" --no-capture-output \
                    python "$ROOT/src/inpainting/composite_rb5_stereo_occlusion.py" \
                    --camera1_rgb_dir "$C1/rgb" --camera2_rgb_dir "$C2/rgb" \
                    --camera1_hawor "$C1/rgb_hawor/retarget_input.npz" \
                    --camera2_hawor "$HAWOR" \
                    --camera1_visible_mask "$C1_VISIBLE" \
                    --camera2_visible_mask "$C2_VISIBLE" \
                    --camera1_contact_dir "$C1/contact" \
                    --contact_dir "$C2/contact" --background "$BACKGROUND" \
                    --overlay_dir "$OVERLAY" --object_mask "$MODAL" \
                    --object_restore_mask "$RESTORE" \
                    --out_dir "$STEREO" --camera1_frame_offset "$OFFSET" \
                    --fps 24 --include_visibility_haco --include_haco_only
            fi
            validate_stereo "$STEREO" "$EXPECTED" "$OFFSET"
        fi

        if stage_enabled derived; then
            validate_contact "$HACO_DUAL" "$EXPECTED" haco 1 0 0 0 0 na
            validate_contact "$HACO_HALF" "$EXPECTED" haco 1 0.5 0 0 0 na
            validate_contact "$HACO_FULL" "$EXPECTED" haco 1 1.0 0 0 0 na
            validate_stereo "$STEREO" "$EXPECTED" "$OFFSET"
            if [ "$FORCE" != "1" ] && [ -s "$SURFACE_DERIVED/comparison_report.json" ] && \
               validate_derived "$SURFACE_DERIVED" "$EXPECTED"; then
                echo "    [skip] XHand surface-strategy derivations"
            else
                echo "    [run] XHand front/side/back strategy derivations"
                DERIVED_ARGS=(
                    --baseline_dir "$HACO_DUAL" --half_thickness_dir "$HACO_HALF"
                    --full_thickness_dir "$HACO_FULL" --force_dir "$STEREO"
                    --overlay_dir "$OVERLAY" --object_mask "$MODAL"
                    --object_restore_mask "$RESTORE"
                    --background "$BACKGROUND" --raw_video "$VIDEO"
                    --out_dir "$SURFACE_DERIVED"
                )
                [ "$FORCE" = "1" ] && DERIVED_ARGS+=(--overwrite)
                PYTHONPATH="$ROOT/src/inpainting" \
                conda run -n "$RENDER_ENV" --no-capture-output \
                    python "$ROOT/src/inpainting/compare_xhand_thickness_strategies.py" \
                    "${DERIVED_ARGS[@]}"
            fi
            validate_derived "$SURFACE_DERIVED" "$EXPECTED"
        fi

        if stage_enabled barrier; then
            validate_contact "$FORCE_TEMPORAL" "$EXPECTED" object3d 1 0 0 1 2 contact
            if [ "$FORCE" != "1" ] && [ -s "$BARRIER/report.json" ] && \
               validate_barrier "$BARRIER" "$EXPECTED"; then
                echo "    [skip] completed-object whole-XHand barrier"
            else
                echo "    [run] completed-object whole-XHand thickness barrier"
                PYTHONPATH="$ROOT/src/inpainting" \
                conda run -n "$RENDER_ENV" --no-capture-output \
                    python "$ROOT/src/inpainting/composite_xhand_object_barrier.py" \
                    --background "$BACKGROUND" --raw_video "$VIDEO" \
                    --overlay_dir "$OVERLAY" --object_mask "$AMODAL" \
                    --object_restore_mask "$RESTORE" \
                    --object_surface_depth "$SURFACE" \
                    --baseline_mask "$FORCE_TEMPORAL/occluded_finger_mask.npy" \
                    --thumb_shell_m 0.01958 --finger_shell_m 0.01465 \
                    --palm_shell_m 0.015 --temporal_max_gap_frames 2 \
                    --temporal_motion_px 6 --temporal_front_slack_m 0.015 \
                    --out_dir "$BARRIER"
            fi
            validate_barrier "$BARRIER" "$EXPECTED"
        fi
    done
done
fi

if stage_enabled compare; then
    for ID in "${EPISODES[@]}"; do
        APPROX_PD="$ROOT/data/cube_dataset/26.08.05_stereo_approx/$ID/camera_2/inpainting/processed/view/0"
        CALIBRATED_PD="$ROOT/data/cube_dataset/26.08.05_stereo_calibrated/$ID/camera_2/inpainting/processed/view/0"
        OUT="$OUTPUT_ROOT/episode_$ID"
        echo
        echo "[compare/$ID] synchronized method + dual-camera + calibration grids"
        COMPARE_ARGS=(
            --approx_pd "$APPROX_PD" --calibrated_pd "$CALIBRATED_PD"
            --out_dir "$OUT" --extended-grid required
        )
        if [ "$FORCE" != "1" ] && [ -s "$OUT/comparison_report.json" ] && \
           validate_comparison "$OUT" "$(expected_frames "$ROOT/data/cube_dataset/26.08.05_stereo_calibrated/$ID/gt_labels.json")" "$APPROX_PD" "$CALIBRATED_PD"; then
            echo "    [skip] current comparison grids"
        else
            [ -e "$OUT" ] && COMPARE_ARGS+=(--overwrite)
            PYTHONPATH="$ROOT/src/inpainting" \
            conda run -n "$RENDER_ENV" --no-capture-output \
                python "$ROOT/src/inpainting/compare_0805_overlay_experiments.py" \
                "${COMPARE_ARGS[@]}"
        fi
    done
fi

echo
echo "Completed 08-05 overlay stages '$STAGES'."
echo "Comparison root: $OUTPUT_ROOT"
