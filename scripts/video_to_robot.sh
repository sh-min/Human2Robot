#!/usr/bin/env bash
# Turn one hand-manipulation video into the same clip performed by an RB5-850e
# arm with an XHAND1 hand, using the settings video 48/IMG_5393 settled on.
#
# Every stage below is the command that produced those results, with the paths
# lifted out. What is NOT parameterised is deliberate: the shadow, base and
# grasp-hiding numbers are the ones the measurements in
# docs/inpainting/ landed on, and changing them per clip is what the HaCo work
# was meant to stop.
#
# Usage:
#   scripts/video_to_robot.sh --video /path/to/clip.MOV --name myclip
#   scripts/video_to_robot.sh --video ... --name ... --from render   # resume
#
# The one thing a clip may still need is --overrides: a JSON of hand-picked SAM2
# prompts for objects the colour rule cannot separate from the table (a
# translucent container). See configs/inpainting/img5393_objects_overrides.json.
set -euo pipefail

VIDEO="" ; NAME="" ; SIDE="left" ; BASE_EDGE="left" ; OVERRIDES=""
FROM="frames" ; WIDTH=1280 ; HEIGHT=720 ; DIFFUERASER=0
PLATE_W=640 ; PLATE_H=360 ; BASE_FROM="" ; PASTE_SEGMENTS=""
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROCESSED_ROOT="${PROCESSED_ROOT:-/home/rkd02/s2p/inpaint_test/processed}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/rkd02/s2p/results}"
PY_INPAINT="${PY_INPAINT:-$HOME/venvs/inpaint/bin/python}"
PY_HAWOR="${PY_HAWOR:-$HOME/venvs/hawor/bin/python}"
PY_HACO="${PY_HACO:-$HOME/venvs/haco/bin/python}"
PY_RETARGET="${PY_RETARGET:-$HOME/venvs/retarget/bin/python}"
PY_DIFFUERASER="${PY_DIFFUERASER:-$HOME/venvs/diffueraser/bin/python}"
# GroundingDINO needs a current transformers, which the inpaint env does not
# carry; it runs in its own env and hands over a JSON of boxes.
PY_GDINO="${PY_GDINO:-$HOME/venvs/gdino/bin/python}"
# What the objects look like, not what they are called. Measured over nine
# frames of two clips: brand names score worst by a distance -- "chocobi" 0.26
# and found in only three of the nine, against 0.66 for "green snack carton" in
# all nine -- because the detector knows shapes and colours, not products.
# Korean works but lands 0.1-0.2 lower than the English for every object.
OBJECT_LABELS="${OBJECT_LABELS:-ceramic mug|green scrub sponge|green snack carton|red snack package|clear plastic container}"
export DEPTH_ANYTHING_CKPT_DIR="${DEPTH_ANYTHING_CKPT_DIR:-/home/rkd02/s2p/ckpt/depth_anything}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video)      VIDEO="$2"; shift 2 ;;
    --name)       NAME="$2"; shift 2 ;;
    --side)       SIDE="$2"; shift 2 ;;
    --base_edge)  BASE_EDGE="$2"; shift 2 ;;
    --overrides)  OVERRIDES="$2"; shift 2 ;;
    --from)       FROM="$2"; shift 2 ;;
    --diffueraser) DIFFUERASER=1; shift ;;
    --plate_size) PLATE_W="$2"; PLATE_H=$(( $2 * 9 / 16 )); shift 2 ;;
    --base_from)  BASE_FROM="$2"; shift 2 ;;
    --paste)      PASTE_SEGMENTS="$2"; shift 2 ;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$VIDEO" && -n "$NAME" ]] || { echo "--video and --name are required" >&2; exit 2; }

D="$PROCESSED_ROOT/$NAME/0"
R="$RESULTS_ROOT/$NAME"
CFG="$ROOT/configs/inpainting/${NAME}_objects_haco.json"
mkdir -p "$D" "$R"
cd "$ROOT"

# Stage ordering, so --from can resume after a failure without redoing hours.
STAGES=(frames hand contact retarget inject human depth objects plate robot composite)
stage_index() { local s; for i in "${!STAGES[@]}"; do [[ "${STAGES[$i]}" == "$1" ]] && { echo "$i"; return; }; done; echo -1; }
FROM_I=$(stage_index "$FROM"); [[ $FROM_I -ge 0 ]] || { echo "unknown stage: $FROM" >&2; exit 2; }
run_stage() { [[ $(stage_index "$1") -ge $FROM_I ]]; }
say() { printf '\n=== %s ===\n' "$1"; }

if run_stage frames; then
  say "frames  ->  $D/rgb"
  mkdir -p "$D/rgb"
  ffmpeg -v error -y -i "$VIDEO" -vf "scale=$WIDTH:$HEIGHT" -an -q:v 2 "$D/rgb/rgb_frame%05d.jpg"
  ffmpeg -v error -y -i "$VIDEO" -vf "scale=$WIDTH:$HEIGHT" -an -c:v libx264 -profile:v high \
         -pix_fmt yuv420p -crf 16 -movflags +faststart "$D/video_L.mp4"
  echo "extracted $(ls "$D/rgb" | wc -l) frames"
fi

if run_stage hand; then
  say "hand pose (HaWoR)"
  "$PY_HAWOR" -u src/hand_estimation/extract_for_retarget.py \
    --rgb_dir "$D/rgb" --img_glob "rgb_frame*.jpg" --skip_slam
  # The focal is estimated, and everything downstream projects with it. Check it
  # before trusting a run: src/contact_estimation/visualize_contact_overlay.py
  # draws the contact vertices on the frames.
fi

if run_stage contact; then
  say "contact (HaCo)"
  "$PY_HACO" -u src/contact_estimation/extract_hand_contact.py \
    --input_dir "$D" --img_glob "rgb_frame*.jpg" --no_viz
  "$PY_HACO" -u src/contact_estimation/aggregate_finger_contact.py \
    --contact_dir "$D/contact" --hawor_npz "$D/rgb_hawor/retarget_input.npz"
fi

if run_stage retarget; then
  say "retargeting (contact-aware)"
  PYTHONPATH="$ROOT/src/retargeting" "$PY_RETARGET" -u src/retargeting/retarget_from_npz.py \
    --npz "$D/rgb_hawor/retarget_input.npz" --hand "$SIDE" \
    --contact --contact_dir "$D/contact" --smooth
fi

if run_stage inject; then
  say "HaWoR -> processor layout"
  "$PY_INPAINT" -u src/inpainting/inject_hawor_data.py \
    --processed_demo "$D" --hawor_npz "$D/rgb_hawor/retarget_input.npz"
fi

if run_stage human || run_stage depth; then
  say "human mask (SAM2) + depth (Depth-Anything V2)"
  # Both fit on a 12 GB card together and neither needs the other, so they run
  # side by side; this is the longest pair in the pipeline.
  ( "$PY_INPAINT" -u src/inpainting/segment_arms.py --processed_demo "$D" ) &
  seg_pid=$!
  ( "$PY_INPAINT" -u src/inpainting/estimate_depth.py --processed_demo "$D" ) &
  depth_pid=$!
  wait $seg_pid; wait $depth_pid
  "$PY_INPAINT" -u src/inpainting/augment_hand_mask_from_keypoints.py \
    --mask "$D/segmentation_processor/masks_arm.npy" \
    --hand_data "$D/hand_processor/hand_data_${SIDE}.npz" \
    --output "$D/segmentation_processor/masks_arm_augmented.npy"
  "$PY_INPAINT" -u src/inpainting/align_depth.py \
    --processed_demo "$D" --hawor_npz "$D/rgb_hawor/retarget_input.npz"
fi

if run_stage objects; then
  say "objects: detect -> contact -> spec -> track -> complete"
  # Name the objects rather than hunt for them by colour. The arm mask spills
  # onto whatever the hand holds, which hides the held object from a colour
  # search exactly when the grip is firmest; a detector still sees it. Cached,
  # because it is a fixed cost per clip and unaffected by anything downstream.
  DETECTIONS="$D/interaction_objects/detections_gdino.json"
  DET_ARGS=()
  if [[ -x "$PY_GDINO" ]]; then
    if [[ ! -f "$DETECTIONS" ]]; then
      IFS='|' read -r -a labels <<<"$OBJECT_LABELS"
      "$PY_GDINO" -u src/inpainting/detect_objects_grounding_dino.py \
        --frames_dir "$D/rgb" --labels "${labels[@]}" --output "$DETECTIONS"
    fi
    DET_ARGS=(--detections "$DETECTIONS")
  else
    echo "[objects] no $PY_GDINO; falling back to the colour heuristic"
  fi
  "$PY_INPAINT" -u src/inpainting/build_object_segments_from_contact.py \
    --contact_dir "$D/contact" --hawor_npz "$D/rgb_hawor/retarget_input.npz" \
    --human_mask "$D/segmentation_processor/masks_arm.npy" --frames_dir "$D/rgb" \
    --side "$SIDE" "${DET_ARGS[@]}" \
    ${OVERRIDES:+--overrides "$OVERRIDES"} --output "$CFG"
  "$PY_INPAINT" -u src/inpainting/segment_interaction_objects.py \
    --processed_demo "$D" --segments_json "$CFG"
  # Fills only what the human hand covered, from the object's own pixels.
  "$PY_INPAINT" -u src/inpainting/complete_occluded_objects.py \
    --video "$D/video_L.mp4" \
    --merged_mask "$D/interaction_objects/object_mask.npy" \
    --base_object_mask "$D/interaction_objects/object_mask.npy" \
    --human_mask "$D/segmentation_processor/masks_arm.npy" \
    --segments_json "$CFG" \
    --output_video "$D/interaction_objects/objsrc_completed.mkv" \
    --output_mask "$D/interaction_objects/object_mask_completed.npy"
fi

if run_stage plate; then
  say "background plate (ProPainter 640 -> full res)"
  # Protect the tracked objects. The keypoint augmentation grows the human mask
  # past the hand silhouette and swallows whatever it is holding or brushing
  # past, and the inpainter then erases that object from the plate for good.
  # An earlier attempt at protection left skin fragments, but that was before
  # the object track had the human mask subtracted from it -- this mask cannot
  # contain hand pixels, so protecting it only keeps real object pixels.
  "$PY_INPAINT" -u src/inpainting/export_propainter_masks.py \
    --mask "$D/segmentation_processor/masks_arm_augmented.npy" \
    --protect_mask "$D/interaction_objects/object_mask.npy" \
    --output_dir "$D/segmentation_processor/propainter_masks"
  mkdir -p "$D/propainter_input_frames"; cp "$D/rgb"/*.jpg "$D/propainter_input_frames/"
  ( cd third_party/ProPainter && "$PY_INPAINT" -u inference_propainter.py \
      --video "$D/propainter_input_frames" \
      --mask "$D/segmentation_processor/propainter_masks" \
      --output "$D/propainter_results_640" --width "$PLATE_W" --height "$PLATE_H" \
      --subvideo_length 50 --fp16 --save_fps 30 )
  "$PY_INPAINT" -u src/inpainting/assemble_propainter_background.py \
    --source_video "$D/video_L.mp4" \
    --propainter_video "$D/propainter_results_640/propainter_input_frames/inpaint_out.mp4" \
    --mask "$D/segmentation_processor/masks_arm_augmented.npy" \
    --protect_mask "$D/interaction_objects/object_mask.npy" \
    --output "$D/inpaint_processor/plate_propainter.mkv"

  if [[ $DIFFUERASER -eq 1 ]]; then
    say "background plate (DiffuEraser refine)"
    # The ProPainter result above is the diffusion prior, so it is computed once
    # and used twice. --max_img_size 640 and the chunked decode are what fits a
    # 500+ frame clip on 12 GB.
    "$PY_INPAINT" - <<PYEOF
import numpy as np, cv2
m = np.load("$D/segmentation_processor/masks_arm_augmented.npy", mmap_mode="r")
T, H, W = m.shape
w = cv2.VideoWriter("$D/mask_for_diffueraser.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
for t in range(T):
    w.write(cv2.cvtColor(np.asarray(m[t]).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR))
w.release()
PYEOF
    # Run from inside the vendored clone so its own package imports resolve,
    # but keep the driver in this repo.
    ( cd third_party/DiffuEraser && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      DIFFUERASER_DECODE_CHUNK=16 "$PY_DIFFUERASER" -u "$ROOT/src/inpainting/diffueraser_lowvram.py" \
        --input_video "$D/video_L.mp4" --input_mask "$D/mask_for_diffueraser.mp4" \
        --priori "$D/propainter_results_640/propainter_input_frames/inpaint_out.mp4" \
        --video_length 20 --max_img_size 640 --save_path "$D/diffueraser_out" )
    "$PY_INPAINT" -u src/inpainting/assemble_propainter_background.py \
      --source_video "$D/video_L.mp4" \
      --propainter_video "$D/diffueraser_out/diffueraser_result.mp4" \
      --mask "$D/segmentation_processor/masks_arm_augmented.npy" \
      --output "$D/inpaint_processor/plate_diffueraser.mkv"
  fi
fi

if run_stage robot; then
  say "robot: base off-frame, hand bolted to the flange, render"
  # Reusing a base from another clip of the same rig skips the 864-candidate IK
  # search. Only the placement is reused -- this clip still solves its own joint
  # trajectory, because the arm has to follow this clip's wrist.
  BASE_ARGS=(--base_offscreen "$BASE_EDGE")
  if [[ -n "$BASE_FROM" ]]; then
    "$PY_INPAINT" -c "import sys,numpy as np; np.save(sys.argv[2], np.load(sys.argv[1])['T_cam_base'])" \
      "$BASE_FROM" "$D/base_reused.npy"
    echo "[base] reusing placement from $BASE_FROM"
    BASE_ARGS=(--base_override "$D/base_reused.npy")
  fi
  "$PY_RETARGET" -u src/inpainting/rb5_build_overlay_input.py \
    --hawor_npz "$D/rgb_hawor/retarget_input.npz" \
    --pkl "$D/rgb_hawor/qpos_xhand_contact_${SIDE}_smooth.pkl" --side "$SIDE" \
    --img_w "$WIDTH" --img_h "$HEIGHT" "${BASE_ARGS[@]}" \
    --smooth_win 5 --mount_hand --out "$D/rb5_mounted.npz"
  for extra in "" "--thumb_mask_only"; do
    PYOPENGL_PLATFORM=egl "$PY_INPAINT" -u src/inpainting/render_xhand_overlay_depth.py \
      --processed_demo "$D" --hawor_npz "$D/rgb_hawor/retarget_input.npz" \
      --right_pkl "$D/rgb_hawor/qpos_xhand_contact_${SIDE}_smooth.pkl" \
      --left_pkl "$D/rgb_hawor/qpos_xhand_contact_${SIDE}_smooth.pkl" \
      --hand "$SIDE" --arm rb5 --rb5_npz "$D/rb5_mounted.npz" \
      --left_embodiment xhand1 --output_subdir overlay_rb5_mnt $extra
  done
fi

if run_stage composite; then
  say "grasp hiding + composite"
  "$PY_INPAINT" -u src/inpainting/build_contact_force_front.py \
    --contact_dir "$D/contact" --hawor_npz "$D/rgb_hawor/retarget_input.npz" \
    --object_mask "$D/interaction_objects/object_mask_completed.npy" \
    --robot_mask "$D/overlay_rb5_mnt/robot_mask.npy" \
    --thumb_mask "$D/overlay_rb5_mnt/robot_thumb_mask.npy" \
    --output "$D/contact/force_front_haco.npy"

  OBJ_SOURCE="interaction_objects/objsrc_completed.mkv"
  FORCE_MASK="contact/force_front_haco.npy"
  if [[ "$PASTE_SEGMENTS" == "auto" ]]; then
    # Decide from the masks which grasps the completion failed to rebuild,
    # rather than naming them per clip.
    PASTE_SEGMENTS="$("$PY_INPAINT" -u src/inpainting/select_paste_segments.py \
      --segments_json "$CFG" \
      --modal_mask "$D/interaction_objects/object_mask.npy" \
      --completed_mask "$D/interaction_objects/object_mask_completed.npy" \
      --verbose | tail -1)"
    echo "[paste] auto-selected: ${PASTE_SEGMENTS:-(none)}"
  fi
  if [[ -n "$PASTE_SEGMENTS" ]]; then
    # An object whose visible pixels are only its rim and handle cannot be
    # completed from its own convex hull -- the hull spans the gap instead of
    # the body. These objects are rigid and fully visible before the hand
    # reaches them, so a clean frame is warped onto the occluded ones.
    say "paste from clean reference frames: $PASTE_SEGMENTS"
    "$PY_INPAINT" -u src/inpainting/paste_object_from_reference.py \
      --source_video "$D/video_L.mp4" \
      --object_source_video "$D/interaction_objects/objsrc_completed.mkv" \
      --object_mask "$D/interaction_objects/object_mask_completed.npy" \
      --modal_mask "$D/interaction_objects/object_mask.npy" \
      --robot_mask "$D/overlay_rb5_mnt/robot_mask.npy" \
      --thumb_mask "$D/overlay_rb5_mnt/robot_thumb_mask.npy" \
      --segments_json "$CFG" --segments $PASTE_SEGMENTS \
      --feather 3.0 --hull_smooth 9 \
      --output_video "$D/interaction_objects/objsrc_pasted.mkv" \
      --output_force_mask "$D/interaction_objects/force_front_pasted.npy"
    "$PY_INPAINT" -c "
import numpy as np, sys
a = np.load(sys.argv[1], mmap_mode='r'); b = np.load(sys.argv[2], mmap_mode='r')
out = np.lib.format.open_memmap(sys.argv[3], mode='w+', dtype=bool, shape=a.shape)
for t in range(len(a)):
    out[t] = np.asarray(a[t]) | (np.asarray(b[t]) if t < len(b) else False)
out.flush()
print(f'[ok] {sys.argv[3]}')
" "$D/contact/force_front_haco.npy" "$D/interaction_objects/force_front_pasted.npy" \
  "$D/contact/force_front_combined.npy"
    OBJ_SOURCE="interaction_objects/objsrc_pasted.mkv"
    FORCE_MASK="contact/force_front_combined.npy"
  fi

  for BG in propainter diffueraser; do
    plate="$D/inpaint_processor/plate_${BG}.mkv"
    [[ -f "$plate" ]] || continue

    # Inpainting takes the person out but leaves the shadow the person threw on
    # the desk, and once the robot is on top that shadow reads as the robot's --
    # while sweeping to a hand that is no longer there. Video 47 measured this
    # and 48 was composited on a cleaned plate; the batch pipeline was not, so
    # its results keep a human shadow the height-band contact shadow then has to
    # compete with. Cached, because it costs a full pass over the plate.
    clean="$D/inpaint_processor/plate_${BG}_noshadow.mkv"
    if [[ ! -f "$clean" ]]; then
      say "human cast shadow off the $BG plate"
      "$PY_INPAINT" -u src/inpainting/remove_cast_shadow.py \
        --plate "$plate" \
        --object_mask "$D/interaction_objects/object_mask.npy" \
        --robot_mask "$D/overlay_rb5_mnt/robot_mask.npy" \
        --output "$clean"
    fi

    out="$R/${NAME}_robot_${BG}.mp4"
    "$PY_INPAINT" -u src/inpainting/composite_interaction_objects.py --processed_demo "$D" \
      --hawor_npz "$D/rgb_hawor/retarget_input.npz" \
      --object_source_video "$OBJ_SOURCE" \
      --background_video "inpaint_processor/plate_${BG}_noshadow.mkv" \
      --object_mask interaction_objects/object_mask_completed.npy \
      --robot_dir overlay_rb5_mnt \
      --force_front_mask "$FORCE_MASK" \
      --force_robot_front_mask overlay_rb5_mnt/robot_thumb_mask.npy \
      --force_robot_front_dilate 2 \
      --shadow_depth depth_processor/depth_aligned.npy \
      --shadow_opacity 0.50 --shadow_blur 5 \
      --shadow_bands 5 --shadow_penumbra 70 --shadow_falloff 0.30 \
      --video_codec h264 --output "$out"
  done

  "$PY_INPAINT" -u src/inpainting/compare_composites.py \
    --video "source=$D/video_L.mp4" \
    --video "robot=$R/${NAME}_robot_propainter.mp4" \
    --fps 30 --output "$R/compare_source_vs_robot.mp4"
fi

say "done  ->  $R"
ls -la "$R"
