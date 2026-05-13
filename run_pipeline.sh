#!/usr/bin/env bash
# ============================================================
#  Skill2Policy Pipeline
#  Stages:
#    1. Hand Estimation   (conda: hawor)        → rgb_hawor/retarget_input.npz
#    2. Contact Estimation(conda: haco)         → contact/*.npz
#    3. Retargeting       (conda: RFM_retarget) → rgb_hawor/qpos_xhand_*.pkl
#    +. RGB Overlay       (conda: RFM_retarget) → rgb_hawor/overlay_*.mp4  [opt]
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Defaults -----------------------------------------------
DATA_DIR="/virtual_lab/ljw_rvlab/lab/cube_dataset/web_data_curated/Rubik_s_Cube__Finger_Tricks_Tutorial__Beginner_to__001"
IMG_FOCAL=497.77
IMG_GLOB="frame_*.jpg"
SKIP_SLAM=1          # 1=skip (default), 0=run SLAM
WITH_CONTACT=1       # 0=stage1 only, 1=stage1+stage2 contact-aware
RUN_OVERLAY=1        # generate RGB overlay video after retargeting
SKIP_HAND=0
SKIP_CONTACT=0
SKIP_RETARGET=0
# -------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

  --data_dir PATH      Episode directory (default: $DATA_DIR)
  --img_focal FLOAT    Camera focal length in pixels (default: $IMG_FOCAL)
  --img_glob PATTERN   RGB frame filename glob (default: "frame_*.jpg")
  --with_slam          Run HaWoR SLAM stage (disabled by default)
  --contact/--no_contact   Stage-2 contact-aware retargeting (default: on)
  --overlay/--no_overlay   RGB overlay video after retargeting (default: on)
  --skip_hand          Skip stage 1 (hand estimation)
  --skip_contact       Skip stage 2 (contact estimation)
  --skip_retarget      Skip stage 3 (retargeting)
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
        -h|--help)        usage ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

RGB_DIR="$DATA_DIR/rgb"
HAWOR_DIR="$DATA_DIR/rgb_hawor"
NPZ="$HAWOR_DIR/retarget_input.npz"
CONTACT_DIR="$DATA_DIR/contact"

echo "============================================================"
echo " Skill2Policy Pipeline"
echo "============================================================"
echo " DATA_DIR    : $DATA_DIR"
echo " ROOT_DIR    : $ROOT_DIR"
echo " IMG_FOCAL   : $IMG_FOCAL"
echo " SLAM        : $([ "$SKIP_SLAM" -eq 0 ] && echo enabled || echo skipped)"
echo " CONTACT     : $([ "$WITH_CONTACT" -eq 1 ] && echo stage1+2 || echo stage1-only)"
echo " OVERLAY     : $([ "$RUN_OVERLAY" -eq 1 ] && echo yes || echo no)"
echo "============================================================"
echo

# ---- Stage 1: Hand Estimation --------------------------------
if [[ $SKIP_HAND -eq 0 ]]; then
    echo "[1/3] Hand Estimation  (conda: hawor)"

    HAND_ARGS=(
        --rgb_dir   "$RGB_DIR"
        --img_glob  "$IMG_GLOB"
        --img_focal "$IMG_FOCAL"
    )
    [[ $SKIP_SLAM -eq 1 ]] && HAND_ARGS+=(--skip_slam)

    conda run -n hawor --no-capture-output \
        python "$ROOT_DIR/src/hand_estimation/extract_for_retarget.py" \
        "${HAND_ARGS[@]}"

    echo "[1/3] Done  →  $NPZ"
else
    echo "[1/3] Hand Estimation  SKIPPED"
fi
echo

# ---- Stage 2: Contact Estimation -----------------------------
if [[ $SKIP_CONTACT -eq 0 ]]; then
    echo "[2/3] Contact Estimation  (conda: haco)"

    conda run -n haco --no-capture-output \
        python "$ROOT_DIR/src/contact_estimation/extract_hand_contact.py" \
        --input_dir "$DATA_DIR"

    echo "[2/3] Done  →  $CONTACT_DIR/"
else
    echo "[2/3] Contact Estimation  SKIPPED"
fi
echo

# ---- Stage 3: Retargeting ------------------------------------
if [[ $SKIP_RETARGET -eq 0 ]]; then
    RETARGET_ARGS=(--npz "$NPZ")
    [[ $WITH_CONTACT -eq 1 ]] && RETARGET_ARGS+=(--contact)

    if [[ $WITH_CONTACT -eq 1 ]]; then
        echo "[3/3] Retargeting  (conda: RFM_retarget)  [stage1 + stage2 contact-aware]"
    else
        echo "[3/3] Retargeting  (conda: RFM_retarget)  [stage1 vector-only]"
    fi

    # _paths.py uses __file__-relative imports; put src/retargeting on PYTHONPATH.
    PYTHONPATH="$ROOT_DIR/src/retargeting${PYTHONPATH:+:$PYTHONPATH}" \
        conda run -n vjepa2-312 --no-capture-output \
        python "$ROOT_DIR/src/retargeting/retarget_from_npz.py" \
        "${RETARGET_ARGS[@]}"

    if [[ $WITH_CONTACT -eq 1 ]]; then
        echo "[3/3] Done  →  $HAWOR_DIR/qpos_xhand_contact_{right,left}.pkl"
    else
        echo "[3/3] Done  →  $HAWOR_DIR/qpos_xhand_{right,left}.pkl"
    fi
else
    echo "[3/3] Retargeting  SKIPPED"
fi
echo

# ---- (Optional) RGB Overlay ----------------------------------
if [[ $RUN_OVERLAY -eq 1 && $SKIP_RETARGET -eq 0 ]]; then
    echo "[+] RGB Overlay  (conda: RFM_retarget)"

    if [[ $WITH_CONTACT -eq 1 ]]; then
        RPKL="$HAWOR_DIR/qpos_xhand_contact_right.pkl"
        LPKL="$HAWOR_DIR/qpos_xhand_contact_left.pkl"
        OUT="$HAWOR_DIR/overlay_stage2_contact.mp4"
    else
        RPKL="$HAWOR_DIR/qpos_xhand_right.pkl"
        LPKL="$HAWOR_DIR/qpos_xhand_left.pkl"
        OUT="$HAWOR_DIR/overlay_stage1.mp4"
    fi

    PYTHONPATH="$ROOT_DIR/src/retargeting${PYTHONPATH:+:$PYTHONPATH}" \
        conda run -n RFM_retarget --no-capture-output \
        python "$ROOT_DIR/src/retargeting/overlay_on_rgb.py" \
        --npz       "$NPZ" \
        --rgb_dir   "$RGB_DIR" \
        --right_pkl "$RPKL" \
        --left_pkl  "$LPKL" \
        --out       "$OUT" \
        --img_focal "$IMG_FOCAL"

    echo "[+] Done  →  $OUT"
    echo
fi

echo "============================================================"
echo " Pipeline complete"
echo "============================================================"
