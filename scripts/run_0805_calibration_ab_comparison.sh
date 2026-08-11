#!/usr/bin/env bash
set -euo pipefail

# Read-only A/B analysis. This does not rerun HaWoR or HaCo.
#
# Environment overrides:
#   APPROX_ROOT, CALIBRATED_ROOT, OUT_DIR, EPISODES, VIEWS,
#   MAKE_VIDEOS (1/0), VIDEO_MAX_FRAMES, FPS, CONDA_ENV

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

APPROX_ROOT="${APPROX_ROOT:-$ROOT/data/kitchen_dataset/26.08.05_stereo_approx}"
CALIBRATED_ROOT="${CALIBRATED_ROOT:-$ROOT/data/kitchen_dataset/26.08.05_stereo_calibrated}"
OUT_DIR="${OUT_DIR:-$ROOT/8-5/calibration_ab}"
EPISODES="${EPISODES:-1,2}"
VIEWS="${VIEWS:-camera_1,camera_2}"
MAKE_VIDEOS="${MAKE_VIDEOS:-1}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-0}"
FPS="${FPS:-24}"
CONDA_ENV="${CONDA_ENV:-inpaint-gpu}"

case "$MAKE_VIDEOS" in
    0) VIDEO_ARGS=() ;;
    1) VIDEO_ARGS=(--make-videos) ;;
    *) echo "MAKE_VIDEOS must be 0 or 1" >&2; exit 1 ;;
esac

conda run -n "$CONDA_ENV" --no-capture-output \
    python "$ROOT/src/hand_estimation/compare_calibration_ab.py" \
    --approx-root "$APPROX_ROOT" \
    --calibrated-root "$CALIBRATED_ROOT" \
    --out-dir "$OUT_DIR" \
    --episodes "$EPISODES" \
    --views "$VIEWS" \
    --fps "$FPS" \
    --video-max-frames "$VIDEO_MAX_FRAMES" \
    "${VIDEO_ARGS[@]}" \
    "$@"
