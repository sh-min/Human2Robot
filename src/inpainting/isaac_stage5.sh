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
set -e
PD="$1"; HAWOR_NPZ="$2"; RIGHT_PKL="$3"; LEFT_PKL="$4"; HAND="${5:-both}"
REPO=/home/uhnam/workspace/skill2policy
UVPY=/home/uhnam/uv-envs/skill2policy/bin/python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMNI_KIT_ACCEPT_EULA=YES
source ~/miniconda3/etc/profile.d/conda.sh
cd "$REPO/src/inpainting"

# render resolution = source video resolution (matches HaWoR/focal projection)
WH="$("$UVPY" -c "import cv2; c=cv2.VideoCapture('$PD/video_L.mp4'); print(int(c.get(3)), int(c.get(4)))")"
W="${WH% *}"; Hh="${WH#* }"
echo "[isaac5] resolution ${W}x${Hh}  hand=$HAND  gpu=$CUDA_VISIBLE_DEVICES"

case "$HAND" in
  left)  SIDES="left" ;;
  right) SIDES="right" ;;
  both)  SIDES="left right" ;;
  *) echo "[isaac5] bad HAND=$HAND"; exit 1 ;;
esac

RENDERED=""
for SIDE in $SIDES; do
  if [ "$SIDE" = "right" ]; then PKL="$RIGHT_PKL"; else PKL="$LEFT_PKL"; fi
  SMOOTH="${PKL%.pkl}_smooth.pkl"; [ -f "$SMOOTH" ] && PKL="$SMOOTH"
  NPZ="$PD/rb5_overlay_input_${SIDE}.npz"

  echo "[isaac5] adapter side=$SIDE  pkl=$(basename "$PKL")"
  conda activate RFM_retarget
  # Default (RB5_BASE_PLACE unset) = the adapter's fixed per-hand off-frame base;
  # set RB5_BASE_PLACE=bottomright|... to opt into the corner search instead.
  BASE_ARG=(); [ -n "$RB5_BASE_PLACE" ] && BASE_ARG=(--base_place "$RB5_BASE_PLACE")
  # the adapter aborts on a side with no valid frames; tolerate that for `both`
  if ! python rb5_build_overlay_input.py --hawor_npz "$HAWOR_NPZ" --pkl "$PKL" \
        --side "$SIDE" --img_w "$W" --img_h "$Hh" \
        "${BASE_ARG[@]}" --out "$NPZ"; then
    echo "[isaac5] side=$SIDE has no valid trajectory — skipping"; continue
  fi

  echo "[isaac5] render side=$SIDE"
  conda activate isaac_lab
  python render_rb5_isaac_overlay.py --headless --enable_cameras \
    --data "$NPZ" --jn "${NPZ%.npz}_jointnames.json" \
    --width "$W" --height "$Hh" --out "$PD/_rb5_${SIDE}"
  RENDERED="$RENDERED $SIDE"
done

[ -z "$RENDERED" ] && { echo "[isaac5] nothing rendered"; exit 1; }

echo "[isaac5] merging sides:$RENDERED (nearer robot depth wins per pixel)"
"$UVPY" - "$PD" $RENDERED <<'PY'
import sys, os, numpy as np
pd = sys.argv[1]; sides = sys.argv[2:]
rgb = dep = msk = None
for s in sides:
    o = f"{pd}/_rb5_{s}/overlay_processor"
    r = np.load(f"{o}/robot_rgb.npy")
    d = np.load(f"{o}/robot_depth.npy").astype(np.float32)
    m = np.load(f"{o}/robot_mask.npy")
    dd = np.where(m, d, np.inf)
    # per-finger semantic labels (added by the semantic render); optional for old renders
    fl_path = f"{o}/robot_finger_labels.npy"
    fl = np.load(fl_path) if os.path.exists(fl_path) else None
    if rgb is None:
        rgb, dep, msk = r.copy(), dd.astype(np.float32), m.copy()
        flab = fl.copy() if fl is not None else None
    else:
        nearer = m & (dd < dep)
        rgb[nearer] = r[nearer]; dep[nearer] = dd[nearer]; msk |= m
        if fl is not None and flab is not None:
            flab[nearer] = fl[nearer]   # nearer side's finger label wins
    del r, d, m
dep[~np.isfinite(dep)] = 0.0
out = f"{pd}/overlay_processor"; os.makedirs(out, exist_ok=True)
np.save(f"{out}/robot_rgb.npy", rgb)
np.save(f"{out}/robot_depth.npy", dep.astype(np.float16))
np.save(f"{out}/robot_mask.npy", msk)
if flab is not None:
    np.save(f"{out}/robot_finger_labels.npy", flab)
    np.save(f"{out}/robot_finger_mask.npy", flab > 0)
    import shutil
    man = f"{pd}/_rb5_{sides[0]}/overlay_processor/manifest.json"
    if os.path.exists(man):
        shutil.copy2(man, f"{out}/manifest.json")
print(f"[isaac5] merged {len(sides)} side(s) -> {out}")
PY
echo "ISAAC5_DONE"
