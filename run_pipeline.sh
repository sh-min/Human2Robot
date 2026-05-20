#!/usr/bin/env bash
# ============================================================
#  Skill2Policy Pipeline
#  Stages:
#    1. Hand Estimation   (conda: hawor)        → rgb_hawor/retarget_input.npz
#    2. Contact Estimation(conda: haco)         → contact/*.npz
#    3. Retargeting       (conda: vjepa2-312) → rgb_hawor/qpos_xhand_*.pkl
#    4. Export Robot Traj (conda: vjepa2-312) → rgb_hawor/final_pose.pkl
#    5. V-JEPA Features   (conda: vjepa2-312) → features.pt
#    +. RGB Overlay       (conda: vjepa2-312) → rgb_hawor/overlay_*.mp4  [opt]
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Defaults -----------------------------------------------
DATA_DIR="/virtual_lab/ljw_rvlab/lab/cube_dataset/web_data_curated/Rubik_s_Cube__Finger_Tricks_Tutorial__Beginner_to__001"
IMG_FOCAL=497.77
IMG_GLOB=""            # empty = auto-detect from rgb dir
SKIP_SLAM=1          # 1=skip (default), 0=run SLAM
WITH_CONTACT=1       # 0=stage1 only, 1=stage1+stage2 contact-aware
RUN_OVERLAY=1        # generate RGB overlay video after retargeting
SKIP_HAND=0
SKIP_CONTACT=0
SKIP_RETARGET=0
SKIP_FEATURES=0
VJEPA_CKPT="/virtual_lab/ljw_rvlab/byeonggyeol/3dgs-visual-grounding/skill2policy/ckpt/v-jepa2/vitl.pt"
# -------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

  --data_dir PATH      Episode directory (default: $DATA_DIR)
  --img_focal FLOAT    Camera focal length in pixels (default: $IMG_FOCAL)
  --img_glob PATTERN   RGB frame filename glob (default: auto-detect)
  --with_slam          Run HaWoR SLAM stage (disabled by default)
  --contact/--no_contact   Stage-2 contact-aware retargeting (default: on)
  --overlay/--no_overlay   RGB overlay video after retargeting (default: on)
  --skip_hand          Skip stage 1 (hand estimation)
  --skip_contact       Skip stage 2 (contact estimation)
  --skip_retarget      Skip stage 3 (retargeting)
  --skip_features      Skip stage 5 (V-JEPA feature extraction)
  --vjepa_ckpt PATH    V-JEPA pretrained checkpoint (default: $VJEPA_CKPT)
  -h, --help           Show this message
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir)       DATA_DIR="$2";  shift 2 ;;
        --img_focal)      IMG_FOCAL="$2"; shift 2 ;;
        --img_glob)       IMG_GLOB="$2";  shift 2 ;;
        --with_slam)      SKIP_SLAM=0;    shift ;;
        --contact)        WITH_CONTACT=1; shift ;;
        --no_contact)     WITH_CONTACT=0; shift ;;
        --overlay)        RUN_OVERLAY=1;  shift ;;
        --no_overlay)     RUN_OVERLAY=0;  shift ;;
        --skip_hand)      SKIP_HAND=1;    shift ;;
        --skip_contact)   SKIP_CONTACT=1; shift ;;
        --skip_retarget)  SKIP_RETARGET=1; shift ;;
        --skip_features)  SKIP_FEATURES=1; shift ;;
        --vjepa_ckpt)     VJEPA_CKPT="$2"; shift 2 ;;
        -h|--help)        usage ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

RGB_DIR="$DATA_DIR/rgb"
HAWOR_DIR="$DATA_DIR/rgb_hawor"
NPZ="$HAWOR_DIR/retarget_input.npz"
CONTACT_DIR="$DATA_DIR/contact"

# Auto-detect IMG_GLOB from first image file in rgb dir
if [[ -z "$IMG_GLOB" ]]; then
    IMG_GLOB=$(python3 -c "
import os, re
from pathlib import Path
files = sorted(f for f in os.listdir('$RGB_DIR')
               if f.lower().endswith(('.jpg', '.jpeg', '.png')))
if not files:
    print('*.png')
else:
    ext = Path(files[0]).suffix
    m = re.match(r'^([A-Za-z_\-]+)', files[0])
    prefix = m.group(1) if m else ''
    print(f'{prefix}*{ext}')
" 2>/dev/null) || IMG_GLOB="*.png"
    GLOB_SOURCE="auto"
else
    GLOB_SOURCE="manual"
fi

PIPELINE_START=$(date +%s)
declare -A STAGE_ELAPSED   # stage name → seconds

echo "============================================================"
echo " Skill2Policy Pipeline"
echo "============================================================"
echo " DATA_DIR    : $DATA_DIR"
echo " ROOT_DIR    : $ROOT_DIR"
echo " IMG_FOCAL   : $IMG_FOCAL"
echo " IMG_GLOB    : $IMG_GLOB  ($GLOB_SOURCE)"
echo " SLAM        : $([ "$SKIP_SLAM" -eq 0 ] && echo enabled || echo skipped)"
echo " CONTACT     : $([ "$WITH_CONTACT" -eq 1 ] && echo stage1+2 || echo stage1-only)"
echo " OVERLAY     : $([ "$RUN_OVERLAY" -eq 1 ] && echo yes || echo no)"
echo " FEATURES    : $([ "$SKIP_FEATURES" -eq 0 ] && echo "yes ($VJEPA_CKPT)" || echo skipped)"
echo "============================================================"
echo

# ---- helpers -------------------------------------------------
_elapsed() { echo $(( $(date +%s) - $1 )); }
_fmt()     { printf "%dm%02ds" $(( $1/60 )) $(( $1%60 )); }

# ---- Stage 1: Hand Estimation --------------------------------
if [[ $SKIP_HAND -eq 0 ]]; then
    echo "[1/5] Hand Estimation  (conda: hawor)"
    _t=$(date +%s)

    HAND_ARGS=(
        --rgb_dir   "$RGB_DIR"
        --img_glob  "$IMG_GLOB"
        --img_focal "$IMG_FOCAL"
    )
    [[ $SKIP_SLAM -eq 1 ]] && HAND_ARGS+=(--skip_slam)

    conda run -n hawor --no-capture-output \
        python "$ROOT_DIR/src/hand_estimation/extract_for_retarget.py" \
        "${HAND_ARGS[@]}"

    # Convert NPZ -> per-frame result.json (required by stage 5 V-JEPA preprocess)
    conda run -n hawor --no-capture-output \
        python "$ROOT_DIR/src/hand_estimation/npz_to_result_json.py" \
        --npz      "$NPZ" \
        --rgb_dir  "$RGB_DIR" \
        --img_glob "$IMG_GLOB" \
        --out      "$DATA_DIR/result.json"

    STAGE_ELAPSED[hand]=$(_elapsed $_t)
    echo "[1/5] Done  →  $NPZ, $DATA_DIR/result.json  ($(_fmt ${STAGE_ELAPSED[hand]}))"
else
    echo "[1/5] Hand Estimation  SKIPPED"
fi
echo

# ---- Stage 2: Contact Estimation -----------------------------
if [[ $SKIP_CONTACT -eq 0 ]]; then
    echo "[2/5] Contact Estimation  (conda: haco)"
    _t=$(date +%s)

    conda run -n haco --no-capture-output \
        python "$ROOT_DIR/src/contact_estimation/extract_hand_contact.py" \
        --input_dir "$DATA_DIR"

    STAGE_ELAPSED[contact]=$(_elapsed $_t)
    echo "[2/5] Done  →  $CONTACT_DIR/  ($(_fmt ${STAGE_ELAPSED[contact]}))"
else
    echo "[2/5] Contact Estimation  SKIPPED"
fi
echo

# ---- Stage 3: Retargeting ------------------------------------
if [[ $SKIP_RETARGET -eq 0 ]]; then
    RETARGET_ARGS=(--npz "$NPZ" --smooth)
    [[ $WITH_CONTACT -eq 1 ]] && RETARGET_ARGS+=(--contact)

    if [[ $WITH_CONTACT -eq 1 ]]; then
        echo "[3/5] Retargeting  (conda: vjepa2-312)  [stage1 + stage2 contact-aware]"
    else
        echo "[3/5] Retargeting  (conda: vjepa2-312)  [stage1 vector-only]"
    fi
    _t=$(date +%s)

    # _paths.py uses __file__-relative imports; put src/retargeting on PYTHONPATH.
    PYTHONPATH="$ROOT_DIR/src/retargeting${PYTHONPATH:+:$PYTHONPATH}" \
        conda run -n vjepa2-312 --no-capture-output \
        python "$ROOT_DIR/src/retargeting/retarget_from_npz.py" \
        "${RETARGET_ARGS[@]}"

    STAGE_ELAPSED[retarget]=$(_elapsed $_t)
    if [[ $WITH_CONTACT -eq 1 ]]; then
        echo "[3/5] Done  →  $HAWOR_DIR/qpos_xhand_contact_{right,left}{,_smooth}.pkl  ($(_fmt ${STAGE_ELAPSED[retarget]}))"
    else
        echo "[3/5] Done  →  $HAWOR_DIR/qpos_xhand_{right,left}{,_smooth}.pkl  ($(_fmt ${STAGE_ELAPSED[retarget]}))"
    fi
else
    echo "[3/5] Retargeting  SKIPPED"
fi
echo

# ---- Stage 4: Export final_pose.pkl -------------------------
if [[ $SKIP_RETARGET -eq 0 ]]; then
    echo "[4/5] Export Robot Trajectory  (conda: vjepa2-312)"
    _t=$(date +%s)

    if [[ $WITH_CONTACT -eq 1 ]]; then
        RPKL="$HAWOR_DIR/qpos_xhand_contact_right_smooth.pkl"
        LPKL="$HAWOR_DIR/qpos_xhand_contact_left_smooth.pkl"
    else
        RPKL="$HAWOR_DIR/qpos_xhand_right_smooth.pkl"
        LPKL="$HAWOR_DIR/qpos_xhand_left_smooth.pkl"
    fi

    PYTHONPATH="$ROOT_DIR/src/retargeting${PYTHONPATH:+:$PYTHONPATH}" \
        conda run -n vjepa2-312 --no-capture-output \
        python "$ROOT_DIR/src/retargeting/export_robot_traj.py" \
        --npz       "$NPZ" \
        --right_pkl "$RPKL" \
        --left_pkl  "$LPKL" \
        --out       "$HAWOR_DIR/final_pose.pkl"

    STAGE_ELAPSED[export]=$(_elapsed $_t)
    echo "[4/5] Done  →  $HAWOR_DIR/final_pose.pkl  ($(_fmt ${STAGE_ELAPSED[export]}))"
else
    echo "[4/5] Export Robot Trajectory  SKIPPED"
fi
echo

# ---- Stage 5: V-JEPA Feature Extraction ---------------------
if [[ $SKIP_FEATURES -eq 0 ]]; then
    echo "[5/5] V-JEPA Feature Extraction  (conda: vjepa2-312)"
    _t=$(date +%s)

    if [[ ! -f "$VJEPA_CKPT" ]]; then
        echo "  ERROR: V-JEPA checkpoint not found: $VJEPA_CKPT" >&2
        echo "  Pass --vjepa_ckpt PATH or place a checkpoint at the default location." >&2
        exit 1
    fi

    REC_PARENT="$(dirname "$DATA_DIR")"
    REC_NAME="$(basename "$DATA_DIR")"

    PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
        conda run -n vjepa2-312 --no-capture-output \
        python -m data_preprocess.preprocess \
        --data_root "$REC_PARENT" \
        --recording_glob "$REC_NAME" \
        --checkpoint "$VJEPA_CKPT"

    STAGE_ELAPSED[features]=$(_elapsed $_t)
    echo "[5/5] Done  →  $DATA_DIR/features.pt  ($(_fmt ${STAGE_ELAPSED[features]}))"
else
    echo "[5/5] V-JEPA Feature Extraction  SKIPPED"
fi
echo

# ---- (Optional) RGB Overlay ----------------------------------
if [[ $RUN_OVERLAY -eq 1 && $SKIP_RETARGET -eq 0 ]]; then
    echo "[+] RGB Overlay  (conda: vjepa2-312)"
    _t=$(date +%s)

    if [[ $WITH_CONTACT -eq 1 ]]; then
        RPKL="$HAWOR_DIR/qpos_xhand_contact_right_smooth.pkl"
        LPKL="$HAWOR_DIR/qpos_xhand_contact_left_smooth.pkl"
        OUT="$HAWOR_DIR/overlay_stage2_contact.mp4"
    else
        RPKL="$HAWOR_DIR/qpos_xhand_right_smooth.pkl"
        LPKL="$HAWOR_DIR/qpos_xhand_left_smooth.pkl"
        OUT="$HAWOR_DIR/overlay_stage1.mp4"
    fi

    PYTHONPATH="$ROOT_DIR/src/retargeting${PYTHONPATH:+:$PYTHONPATH}" \
        conda run -n vjepa2-312 --no-capture-output \
        python "$ROOT_DIR/src/retargeting/overlay_on_rgb.py" \
        --npz       "$NPZ" \
        --rgb_dir   "$RGB_DIR" \
        --rgb_glob  "$IMG_GLOB" \
        --right_pkl "$RPKL" \
        --left_pkl  "$LPKL" \
        --out       "$OUT" \
        --img_focal "$IMG_FOCAL"

    STAGE_ELAPSED[overlay]=$(_elapsed $_t)
    echo "[+] Done  →  $OUT  ($(_fmt ${STAGE_ELAPSED[overlay]}))"
    echo
fi

# ---- Summary -------------------------------------------------
TOTAL=$(_elapsed $PIPELINE_START)
NUM_FRAMES=$(ls "$RGB_DIR"/$IMG_GLOB 2>/dev/null | wc -l)
TIMING_TXT="$DATA_DIR/pipeline_timing.txt"

{
    echo "============================================================"
    echo " Pipeline complete"
    echo "============================================================"
    printf " %-22s  %s\n" "date:"                "$(date '+%Y-%m-%d %H:%M:%S')"
    printf " %-22s  %s\n" "data_dir:"            "$DATA_DIR"
    printf " %-22s  %s\n" "frames processed:"    "$NUM_FRAMES"
    echo "------------------------------------------------------------"
    [[ -v STAGE_ELAPSED[hand]     ]] && printf " %-22s  %s  (%d frames/s)\n" "hand estimation:"    "$(_fmt ${STAGE_ELAPSED[hand]})"    $(( NUM_FRAMES / (STAGE_ELAPSED[hand] > 0 ? STAGE_ELAPSED[hand] : 1) ))
    [[ -v STAGE_ELAPSED[contact]  ]] && printf " %-22s  %s  (%d frames/s)\n" "contact estimation:" "$(_fmt ${STAGE_ELAPSED[contact]})"  $(( NUM_FRAMES / (STAGE_ELAPSED[contact] > 0 ? STAGE_ELAPSED[contact] : 1) ))
    [[ -v STAGE_ELAPSED[retarget] ]] && printf " %-22s  %s  (%d frames/s)\n" "retargeting:"        "$(_fmt ${STAGE_ELAPSED[retarget]})" $(( NUM_FRAMES / (STAGE_ELAPSED[retarget] > 0 ? STAGE_ELAPSED[retarget] : 1) ))
    [[ -v STAGE_ELAPSED[export]   ]] && printf " %-22s  %s  (%d frames/s)\n" "export final_pose:"  "$(_fmt ${STAGE_ELAPSED[export]})"   $(( NUM_FRAMES / (STAGE_ELAPSED[export]   > 0 ? STAGE_ELAPSED[export]   : 1) ))
    [[ -v STAGE_ELAPSED[features] ]] && printf " %-22s  %s  (%d frames/s)\n" "v-jepa features:"    "$(_fmt ${STAGE_ELAPSED[features]})" $(( NUM_FRAMES / (STAGE_ELAPSED[features] > 0 ? STAGE_ELAPSED[features] : 1) ))
    [[ -v STAGE_ELAPSED[overlay]  ]] && printf " %-22s  %s  (%d frames/s)\n" "rgb overlay:"        "$(_fmt ${STAGE_ELAPSED[overlay]})"  $(( NUM_FRAMES / (STAGE_ELAPSED[overlay] > 0 ? STAGE_ELAPSED[overlay] : 1) ))
    echo "------------------------------------------------------------"
    printf " %-22s  %s\n" "total:" "$(_fmt $TOTAL)"
    echo "============================================================"
} | tee "$TIMING_TXT"

echo "(timing saved → $TIMING_TXT)"
