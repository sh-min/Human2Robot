#!/usr/bin/env bash
# Run the pipeline over a numbered list of Drive clips, unattended.
#
# Each clip is downloaded, processed with the same settings, and its result
# copied out under its number. Nothing here is tuned per clip: object prompts
# come from HaCo contact, the paste list is chosen from the masks, and the base
# placement is reused from a clip of the same rig. A clip that fails is logged
# and the batch moves on rather than stopping the night's work.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-/home/rkd02/s2p/결과영상}"
WORK="${WORK:-/tmp/claude-1000/-home-rkd02-s2p/fa9a59a3-db86-4b45-9b73-54d2f63771eb/scratchpad/drive}"
BASE_NPZ="${BASE_NPZ:-/home/rkd02/s2p/inpaint_test/processed/img5393/0/rb5_mounted.npz}"
LOG="${LOG:-$WORK/batch.log}"
mkdir -p "$OUT" "$WORK"

# number:drive_id:name
CLIPS=(
  "4:1KjixrQKEeroxMlOXzuBwcWdOmBAdzJMy:IMG_5397"
  "5:1QgWDEG7SvAdtE-Y-ewNqbbEAUfV0cOeO:IMG_5398"
  "6:1pk3mjxwV64zdUyigYqBOqxvHlA2RvHEv:IMG_5399"
  "7:1n6svbtF2QW9ACjiRLpvhJBq9bgi8zFOH:IMG_5400"
  "8:1ZootMgxIlG1edXpF3jXak-0_czFI1yBJ:IMG_5401"
  "9:1gDlGz4eI7BtlXpC_Yw_QpI6rihOCcv1Q:IMG_5402"
  "10:1sw5F-CB_vu6r7hAaSavYVgW3PDPUT0hc:IMG_5403"
  "11:1I3y9MS9TeRZwawg_Nf8zpUgY_DeB2w2m:IMG_5404"
  "12:1nVYdE-V44E4k1be_w1K7562-_XZwoNBm:IMG_5405"
  "13:1ckwTux7518kupTr1gbIpjPOqJ1bp2R-b:IMG_5406"
  "14:1Upt0dqDBBVWvTyxAsD91Id8D2Wu_octe:IMG_5407"
  "15:1wZvA5EAxOzNccTMm5F7X2NocQMksz_5V:IMG_5408"
  "16:1eUSU63wqIBNjYc3rgK-aNuSkMJgWy40j:IMG_5409"
  "17:1Sb8xngehRdhq7FQeBJG_6qY9aL5FrUmt:IMG_5411"
  "18:1qn1iTHkHqjCkWyrBfwqsuhWeqj5rVlrV:IMG_5412"
  "19:1997sMKjWW7AfcGelOEJkqiHzH-E_HHi3:IMG_5413"
  "20:13OtfiAddMHjHTHAN885taCtlRlybiJaY:IMG_5414"
  "21:13D4PAVZGj9Xof0iJpz2EuMvpykanbast:IMG_5415"
  "22:11YElmCleA6ZV-9RtiBK2aKlF9M9sjvWg:IMG_5416"
  "23:10mOugLiaazQmTHe_uLNa3GazxZOoQLRH:IMG_5417"
)

say() { printf '\n########## %s  %s ##########\n' "$(date +%H:%M)" "$1" | tee -a "$LOG"; }

for entry in "${CLIPS[@]}"; do
  IFS=: read -r num id name <<<"$entry"
  target="$OUT/${num}번_${name}.mp4"
  if [[ -f "$target" ]]; then say "$num $name  이미 있음, 건너뜀"; continue; fi

  say "$num $name  시작"
  video="$WORK/$name.MOV"
  if [[ ! -f "$video" ]]; then
    if ! gdown --no-cookies "$id" -O "$video" >>"$LOG" 2>&1; then
      say "$num $name  다운로드 실패"; continue
    fi
  fi

  if "$ROOT/scripts/video_to_robot.sh" --video "$video" --name "${num}_${name}" \
       --plate_size 512 --base_from "$BASE_NPZ" --paste auto >>"$LOG" 2>&1; then
    src="/home/rkd02/s2p/results/${num}_${name}/${num}_${name}_robot_propainter.mp4"
    if [[ -f "$src" ]]; then
      cp "$src" "$target"
      say "$num $name  완료 -> $(basename "$target")"
      # 10 GB of intermediates per clip would fill the disk over twenty of them.
      # The result and the plate stay; the arrays can be rebuilt from the clip.
      demo="/home/rkd02/s2p/inpaint_test/processed/${num}_${name}/0"
      rm -rf "$demo/overlay_rb5_mnt" "$demo/depth_processor" \
             "$demo/propainter_input_frames" "$demo/propainter_results_640" \
             "$demo/segmentation_processor/propainter_masks" \
             "$demo/interaction_objects/objsrc_completed.mkv" \
             "$demo/video_rgb_imgs.mkv" "$demo/rgb_hawor/extracted_images"
      say "$num $name  중간 산출물 정리, 남은 공간 $(df -h /home/rkd02/s2p | awk 'NR==2{print $4}')"
    else
      say "$num $name  합성 산출물 없음"
    fi
  else
    say "$num $name  파이프라인 실패 (로그 참조)"
  fi
  rm -f "$video"
done
say "배치 종료"
