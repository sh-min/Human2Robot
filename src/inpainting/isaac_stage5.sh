#!/bin/bash
# Stage 5 (Isaac backend): render RB5-850 arm + xhand hand(s) over the tracked
# wrist trajectory and emit overlay_processor/robot_{rgb,depth,mask}.npy — the
# same contract composite_layered.py consumes.
#
# Cross-env: the IK adapter runs in `RFM_retarget` (pinocchio), the renderer in
# `isaac_lab` (Isaac Sim). run_layered.py (uv env) shells out to this script.
#
# Usage: isaac_stage5.sh <PD> <HAWOR_NPZ> <RIGHT_PKL> <LEFT_PKL> <HAND>
#   HAND = left | right | both.  Render resolution is auto-detected from video_L.mp4.
#   GPU comes from the inherited CUDA_VISIBLE_DEVICES (default 0).
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <PD> <HAWOR_NPZ> <RIGHT_PKL> <LEFT_PKL> [left|right|both]" >&2
  exit 2
fi

PD="$1"; HAWOR_NPZ="$2"; RIGHT_PKL="$3"; LEFT_PKL="$4"; HAND="${5:-both}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
RETARGET_ENV="${SKILL2POLICY_RETARGET_ENV:-RFM_retarget}"
ISAAC_ENV="${SKILL2POLICY_ISAAC_ENV:-isaac_lab}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export SKILL2POLICY_ISAAC_ASSETS="${SKILL2POLICY_ISAAC_ASSETS:-$REPO/isaac_assets}"

command -v conda >/dev/null || { echo "[isaac5] conda is not on PATH" >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "[isaac5] ffprobe is required to read video resolution" >&2; exit 1; }
[ -f "$HAWOR_NPZ" ] || { echo "[isaac5] missing HaWoR npz: $HAWOR_NPZ" >&2; exit 1; }
[ -f "$PD/video_L.mp4" ] || { echo "[isaac5] missing source video: $PD/video_L.mp4" >&2; exit 1; }

# Resolve the interpreter once.  This keeps the merge step in the same NumPy
# environment as the Pinocchio adapter without depending on a user's conda
# installation path or an unrelated uv environment.
RETARGET_PY="$(conda run -n "$RETARGET_ENV" python -c 'import sys; print(sys.executable)' | tail -n 1)"
[ -x "$RETARGET_PY" ] || { echo "[isaac5] cannot resolve Python in env $RETARGET_ENV" >&2; exit 1; }

# Fail before starting a long render when an old editable IsaacLab install
# points at a checkout that has since been removed.
if ! conda run -n "$ISAAC_ENV" python -c 'import isaaclab' >/dev/null 2>&1; then
  echo "[isaac5] env '$ISAAC_ENV' cannot import isaaclab (check its editable IsaacLab checkout)" >&2
  exit 1
fi

cd "$HERE"

# Render resolution = source video resolution (matches HaWoR/focal projection).
WH="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
       -of csv=p=0:s=x "$PD/video_L.mp4" | head -n 1)"
if [[ ! "$WH" =~ ^[0-9]+x[0-9]+$ ]]; then
  echo "[isaac5] could not read a valid resolution from $PD/video_L.mp4: $WH" >&2
  exit 1
fi
W="${WH%x*}"; Hh="${WH#*x}"
echo "[isaac5] resolution ${W}x${Hh}  hand=$HAND  gpu=$CUDA_VISIBLE_DEVICES"

case "$HAND" in
  left)  SIDES=(left) ;;
  right) SIDES=(right) ;;
  both)  SIDES=(left right) ;;
  *) echo "[isaac5] bad HAND=$HAND"; exit 1 ;;
esac

for SIDE in "${SIDES[@]}"; do
  if [ "$SIDE" = "right" ]; then PKL="$RIGHT_PKL"; else PKL="$LEFT_PKL"; fi
  [ -f "$PKL" ] || { echo "[isaac5] missing $SIDE retarget pkl: $PKL" >&2; exit 1; }
  [ -f "$SKILL2POLICY_ISAAC_ASSETS/xhand_${SIDE}_urdf/xhand_${SIDE}.usd" ] || {
    echo "[isaac5] missing $SIDE XHand USD under $SKILL2POLICY_ISAAC_ASSETS" >&2; exit 1;
  }
done
[ -f "$SKILL2POLICY_ISAAC_ASSETS/rb5_850e_urdf/rb5_850e.usd" ] || {
  echo "[isaac5] missing RB5 USD under $SKILL2POLICY_ISAAC_ASSETS" >&2; exit 1;
}

RENDERED=()
for SIDE in "${SIDES[@]}"; do
  if [ "$SIDE" = "right" ]; then PKL="$RIGHT_PKL"; else PKL="$LEFT_PKL"; fi
  SMOOTH="${PKL%.pkl}_smooth.pkl"; [ -f "$SMOOTH" ] && PKL="$SMOOTH"
  NPZ="$PD/rb5_overlay_input_${SIDE}.npz"

  echo "[isaac5] adapter side=$SIDE  pkl=$(basename "$PKL")"
  # Default (RB5_BASE_PLACE unset) = the adapter's fixed per-hand off-frame base;
  # set RB5_BASE_PLACE=bottomright|... to opt into the corner search instead.
  BASE_ARG=(); [ -n "${RB5_BASE_PLACE:-}" ] && BASE_ARG=(--base_place "$RB5_BASE_PLACE")
  # the adapter aborts on a side with no valid frames; tolerate that for `both`
  if ! "$RETARGET_PY" "$HERE/rb5_build_overlay_input.py" \
        --hawor_npz "$HAWOR_NPZ" --pkl "$PKL" \
        --side "$SIDE" --img_w "$W" --img_h "$Hh" \
        "${BASE_ARG[@]}" --out "$NPZ"; then
    echo "[isaac5] side=$SIDE has no valid trajectory — skipping"; continue
  fi

  echo "[isaac5] render side=$SIDE"
  conda run -n "$ISAAC_ENV" --no-capture-output \
    python "$HERE/render_rb5_isaac_overlay.py" --headless --enable_cameras \
    --data "$NPZ" --jn "${NPZ%.npz}_jointnames.json" \
    --width "$W" --height "$Hh" --out "$PD/_rb5_${SIDE}"
  RENDERED+=("$SIDE")
done

[ "${#RENDERED[@]}" -gt 0 ] || { echo "[isaac5] nothing rendered"; exit 1; }

echo "[isaac5] merging sides: ${RENDERED[*]} (nearer robot depth wins per pixel)"
"$RETARGET_PY" - "$PD" "${RENDERED[@]}" <<'PY'
import json, sys, os, numpy as np
pd = sys.argv[1]; sides = sys.argv[2:]
rgb = dep = msk = None
for s in sides:
    o = f"{pd}/_rb5_{s}/overlay_processor"
    r = np.load(f"{o}/robot_rgb.npy")
    d = np.load(f"{o}/robot_depth.npy").astype(np.float32)
    m = np.load(f"{o}/robot_mask.npy")
    dd = np.where(m, d, np.inf)
    # Per-finger semantic labels are required by the stereo/contact compositor.
    fl_path = f"{o}/robot_finger_labels.npy"
    if not os.path.exists(fl_path):
        raise FileNotFoundError(f"missing Isaac finger semantics: {fl_path}")
    fl = np.load(fl_path)
    if r.shape[:3] != d.shape or d.shape != m.shape or m.shape != fl.shape:
        raise ValueError(
            f"overlay shape mismatch for {s}: rgb={r.shape}, depth={d.shape}, "
            f"mask={m.shape}, finger_labels={fl.shape}"
        )
    unknown = np.setdiff1d(np.unique(fl), np.arange(6, dtype=np.uint8))
    if unknown.size:
        raise ValueError(f"unexpected finger labels for {s}: {unknown.tolist()}")
    if rgb is None:
        rgb, dep, msk = r.copy(), dd.astype(np.float32), m.copy()
        flab = fl.copy()
    else:
        nearer = m & (dd < dep)
        rgb[nearer] = r[nearer]; dep[nearer] = dd[nearer]; msk |= m
        flab[nearer] = fl[nearer]   # nearer side's finger label wins
    del r, d, m
dep[~np.isfinite(dep)] = 0.0
out = f"{pd}/overlay_processor"; os.makedirs(out, exist_ok=True)
np.save(f"{out}/robot_rgb.npy", rgb)
np.save(f"{out}/robot_depth.npy", dep.astype(np.float16))
np.save(f"{out}/robot_mask.npy", msk)
np.save(f"{out}/robot_finger_labels.npy", flab)
np.save(f"{out}/robot_finger_mask.npy", flab > 0)
with open(f"{out}/manifest.json", "w") as stream:
    json.dump(
        {
            "sides": sides,
            "resolution": [int(rgb.shape[2]), int(rgb.shape[1])],
            "finger_mask": {
                "label_ids": {
                    "thumb": 1, "index": 2, "middle": 3, "ring": 4, "pinky": 5
                }
            },
        },
        stream,
    )
print(f"[isaac5] merged {len(sides)} side(s) -> {out}")
PY
echo "ISAAC5_DONE"
