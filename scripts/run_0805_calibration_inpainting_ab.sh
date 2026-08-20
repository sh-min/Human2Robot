#!/usr/bin/env bash
set -euo pipefail

# Build the minimum 08-05 MH object-aware inpainting stack for a controlled
# approximate-vs-calibrated focal A/B.  MH owns all pixels and geometry; the
# synchronized SH HaCo track contributes same-finger confidence only.
#
# Examples:
#   BRANCHES=calibrated STAGES=assets bash scripts/run_0805_calibration_inpainting_ab.sh 1
#   BRANCHES=calibrated STAGES=depth,surface,completion bash scripts/run_0805_calibration_inpainting_ab.sh 1
#   BRANCHES=approx,calibrated STAGES=all bash scripts/run_0805_calibration_inpainting_ab.sh 1 2

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
APPROX_ROOT="${APPROX_ROOT:-$ROOT/data/cube_dataset/26.08.05_stereo_approx}"
CALIBRATED_ROOT="${CALIBRATED_ROOT:-$ROOT/data/cube_dataset/26.08.05_stereo_calibrated}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/8-5/calibration_inpainting_ab}"
ENVIRONMENT="${ENVIRONMENT:-inpaint-gpu}"
CHECKPOINT="${CHECKPOINT:-$ROOT/weights/depth_anything/depth_anything_v2_metric_hypersim_vits.pth}"
E2FGVI_CHECKPOINT="$ROOT/third_party/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth"
BRANCHES="${BRANCHES:-approx,calibrated}"
STAGES="${STAGES:-all}"
FORCE="${FORCE:-0}"
INPAINT_BATCH_SIZE="${INPAINT_BATCH_SIZE:-4}"

if [ "$#" -gt 0 ]; then
    EPISODES=("$@")
else
    EPISODES=("1" "2")
fi

for ID in "${EPISODES[@]}"; do
    [[ "$ID" =~ ^[0-9]+$ ]] || {
        echo "Episode names must be numeric, got '$ID'" >&2
        exit 1
    }
done

stage_enabled() {
    [ "$STAGES" = "all" ] || [[ ",$STAGES," == *",$1,"* ]]
}

IFS=',' read -r -a REQUESTED_STAGES <<< "$STAGES"
for STAGE in "${REQUESTED_STAGES[@]}"; do
    case "$STAGE" in
        all|assets|depth|surface|completion|compare) ;;
        *)
            echo "Unknown stage '$STAGE'" >&2
            exit 1
            ;;
    esac
done

IFS=',' read -r -a REQUESTED_BRANCHES <<< "$BRANCHES"
for BRANCH in "${REQUESTED_BRANCHES[@]}"; do
    case "$BRANCH" in
        approx|calibrated) ;;
        *)
            echo "Unknown branch '$BRANCH'; use approx and/or calibrated" >&2
            exit 1
            ;;
    esac
done

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

validate_contact_tree() {
    local directory="$1"
    local frames="$2"
    [ -d "$directory" ] || return 1
    local contact_count
    contact_count=$(find "$directory" -maxdepth 1 -type f \
        -name 'rgb_frame*.npz' -printf '%f\n' | wc -l)
    [ "$contact_count" -eq "$frames" ] || return 1
    local frame_index path
    for ((frame_index = 0; frame_index < frames; frame_index++)); do
        printf -v path '%s/rgb_frame%06d.npz' "$directory" "$frame_index"
        [ -s "$path" ] || return 1
    done
    return 0
}

fresh_contact_tree() {
    local target="$1"
    local directory="$2"
    local frames="$3"
    [ -s "$target" ] || return 1
    validate_contact_tree "$directory" "$frames" || return 1
    local frame_index path
    for ((frame_index = 0; frame_index < frames; frame_index++)); do
        printf -v path '%s/rgb_frame%06d.npz' "$directory" "$frame_index"
        [ "$path" -nt "$target" ] && return 1
    done
    return 0
}

validate_array() {
    local path="$1"
    local frames="$2"
    local kind="$3"
    conda run -n "$ENVIRONMENT" python -c '
import numpy as np, sys
path, frames, kind = sys.argv[1], int(sys.argv[2]), sys.argv[3]
value = np.load(path, mmap_mode="r", allow_pickle=False)
if value.shape != (frames, 720, 1280):
    raise SystemExit(f"{path}: bad shape {value.shape}")
if kind == "bool" and value.dtype != np.bool_:
    raise SystemExit(f"{path}: expected bool, got {value.dtype}")
if kind == "float" and not np.issubdtype(value.dtype, np.floating):
    raise SystemExit(f"{path}: expected float, got {value.dtype}")
' "$path" "$frames" "$kind"
}

completion_matches() {
    local directory="$1"
    local offset="$2"
    local frames="$3"
    local hawor="$4"
    local contact="$5"
    local auxiliary_contact="$6"
    local source="$7"
    conda run -n "$ENVIRONMENT" python -c '
import json, sys
from pathlib import Path
import numpy as np
root, offset, frames = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
hawor, contact, auxiliary_contact, source = (
    Path(value).resolve() for value in sys.argv[4:8]
)
required = [
    root / "report.json",
    root / "video_hand_removed_modal_only.mp4",
    root / "video_object_completed.mp4",
    root / "debug_object_completion.mp4",
    root / "object_mask_observed_clean.npy",
    root / "object_mask_amodal.npy",
    root / "object_surface_depth_completed.npy",
    root / "haco_contact_support.npy",
    root / "haco_evidence.npz",
]
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    raise SystemExit(1)
report = json.loads(required[0].read_text())
if report.get("method") != "dual_haco_selected_hand_cleaned_object_constrained_e2fgvi":
    raise SystemExit(1)
if int(report.get("metadata", {}).get("frames", -1)) != frames:
    raise SystemExit(1)
config = report.get("config", {})
if int(config.get("aux_frame_offset", 999999)) != offset:
    raise SystemExit(1)
if config.get("aux_out_of_range_policy") != "primary_evidence_only":
    raise SystemExit(1)
try:
    reported_focal = float(config["primary_hawor_focal_px"])
except (KeyError, TypeError, ValueError):
    raise SystemExit(1)
with np.load(hawor, allow_pickle=False) as expected_hawor:
    expected_focal = float(np.asarray(expected_hawor["img_focal"]).item())
if not np.isfinite(reported_focal) or not np.isclose(
    reported_focal, expected_focal, rtol=1e-9, atol=1e-6
):
    raise SystemExit(1)
sources = report.get("sources", {})
for key, expected_path in (
    ("hawor_npz", hawor),
    ("contact_dir", contact),
    ("aux_contact_dir", auxiliary_contact),
    ("source", source),
):
    value = sources.get(key)
    if not isinstance(value, str) or Path(value).resolve() != expected_path:
        raise SystemExit(1)
invariants = report.get("invariants", {})
expected = (
    "trusted_modal_subset_input_modal",
    "trusted_modal_subset_amodal",
    "hand_contested_disjoint_trusted_modal",
    "hidden_disjoint_trusted_modal",
    "trusted_modal_rgb_has_priority",
    "haco_selected_hidden_subset_raw_hidden",
    "primary_view_owns_haco_projection",
    "auxiliary_haco_is_confidence_only",
)
if not all(invariants.get(name) is True for name in expected):
    raise SystemExit(1)
if int(report.get("counts", {}).get("hidden_pixels_without_completed_depth", -1)) != 0:
    raise SystemExit(1)
for path, kind in (
    (required[4], "bool"),
    (required[5], "bool"),
    (required[6], "float"),
    (required[7], "bool"),
):
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.shape != (frames, 720, 1280):
        raise SystemExit(1)
    if kind == "bool" and value.dtype != np.bool_:
        raise SystemExit(1)
    if kind == "float" and not np.issubdtype(value.dtype, np.floating):
        raise SystemExit(1)
with np.load(required[8], allow_pickle=False) as evidence:
    mapped = np.asarray(evidence["auxiliary_frame_indices"])
if mapped.shape != (frames,):
    raise SystemExit(1)
expected_mapped = np.arange(frames, dtype=np.int64) + offset
expected_mapped[(expected_mapped < 0) | (expected_mapped >= frames)] = -1
if not np.array_equal(mapped, expected_mapped):
    raise SystemExit(1)
' "$directory" "$offset" "$frames" "$hawor" "$contact" \
    "$auxiliary_contact" "$source" >/dev/null 2>&1
}

test -s "$CHECKPOINT"
if stage_enabled completion; then
    test -s "$E2FGVI_CHECKPOINT"
fi
conda run -n "$ENVIRONMENT" --no-capture-output python -c \
    'import cv2, mediapy, numpy, torch; assert torch.cuda.is_available()'
cd "$ROOT"

for BRANCH in "${REQUESTED_BRANCHES[@]}"; do
    if [ "$BRANCH" = "approx" ]; then
        DATA_ROOT="$APPROX_ROOT"
    else
        DATA_ROOT="$CALIBRATED_ROOT"
    fi
    for ID in "${EPISODES[@]}"; do
        EP="$DATA_ROOT/$ID"
        C1="$EP/camera_1"
        C2="$EP/camera_2"
        GT="$EP/gt_labels.json"
        MANIFEST="$EP/stereo_manifest.json"
        RGB="$C2/rgb"
        HAWOR="$C2/rgb_hawor/retarget_input.npz"
        CONTACT="$C2/contact"
        AUX_CONTACT="$C1/contact"
        EXPECTED=$(python -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["num_frames"])' \
            "$GT")
        OFFSET=$(python -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["temporal_alignment"]["camera1_frame_offset"])' \
            "$MANIFEST")
        mapfile -t HAND_CANDIDATES < <(
            rg --files --no-ignore "$C2/rgb_hawor" |
                rg '/tracks_[^/]+/model_masks\.npy$'
        )
        if [ "${#HAND_CANDIDATES[@]}" -ne 1 ]; then
            echo "[$BRANCH/$ID] expected one HaWoR hand mask, found ${#HAND_CANDIDATES[@]}" >&2
            exit 1
        fi
        HAND="${HAND_CANDIDATES[0]}"

        RAW_ROOT="$C2/inpainting/raw"
        PROCESSED_ROOT="$C2/inpainting/processed"
        PD="$PROCESSED_ROOT/view/0"
        VIDEO="$PD/video_L.mp4"
        RGB_COPY="$PD/video_rgb_imgs.mkv"
        ARM="$PD/segmentation_processor/masks_arm.npy"
        MODAL="$PD/object_layer/object_mask_modal.npy"
        DEPTH="$PD/depth_processor/depth_aligned_metric.npy"
        DEPTH_RAW="$PD/depth_processor/depth_metric_raw.npy"
        DEPTH_PARAMS="$PD/depth_processor/depth_metric_params.npz"
        SURFACE_DIR="$PD/object_surface_3d"
        SURFACE="$SURFACE_DIR/object_surface_depth.npy"
        COMPLETION="$PD/object_completion_dual_haco_e2fgvi"

        for required in "$GT" "$MANIFEST" "$HAWOR" "$CONTACT" \
            "$AUX_CONTACT" "$HAND"; do
            test -e "$required"
        done
        RGB_COUNT=$(rg --files --no-ignore "$RGB" |
            rg '/rgb_frame[0-9]{6}\.jpg$' | wc -l)
        if [ "$RGB_COUNT" -ne "$EXPECTED" ]; then
            echo "[$BRANCH/$ID] RGB count $RGB_COUNT != $EXPECTED" >&2
            exit 1
        fi
        validate_array "$HAND" "$EXPECTED" bool
        if ! validate_contact_tree "$CONTACT" "$EXPECTED"; then
            echo "[$BRANCH/$ID] incomplete MH HaCo contact tree" >&2
            exit 1
        fi
        if ! validate_contact_tree "$AUX_CONTACT" "$EXPECTED"; then
            echo "[$BRANCH/$ID] incomplete SH HaCo contact tree" >&2
            exit 1
        fi
        echo "[$BRANCH/$ID] frames=$EXPECTED, SH offset=$OFFSET"

        if stage_enabled assets; then
            if [ "$FORCE" = "1" ] || [ ! -s "$VIDEO" ]; then
                PREPARE_ARGS=(
                    --input "$RGB"
                    --data_root "$RAW_ROOT"
                    --processed_root "$PROCESSED_ROOT"
                    --demo_name view
                    --demo_num 0
                    --fps 24
                    --glob 'rgb_frame*.jpg'
                )
                [ "$FORCE" = "1" ] && PREPARE_ARGS+=(--overwrite)
                conda run -n "$ENVIRONMENT" --no-capture-output \
                    python src/inpainting/prepare_demo.py "${PREPARE_ARGS[@]}"
            fi
            VIDEO_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames \
                -show_entries stream=nb_read_frames -of default=nw=1:nk=1 \
                "$VIDEO")
            if [ "$VIDEO_FRAMES" -ne "$EXPECTED" ]; then
                echo "[$BRANCH/$ID] prepared video count $VIDEO_FRAMES != $EXPECTED" >&2
                exit 1
            fi

            if [ "$FORCE" = "1" ] || \
               ! fresh_file "$PD/hand_processor/hand_data_left.npz" "$HAWOR" "$VIDEO" || \
               ! fresh_file "$PD/hand_processor/hand_data_right.npz" "$HAWOR" "$VIDEO" || \
               ! fresh_file "$RGB_COPY" "$VIDEO"; then
                INJECT_ARGS=(--processed_demo "$PD" --hawor_npz "$HAWOR")
                [ "$FORCE" = "1" ] && INJECT_ARGS+=(--overwrite)
                conda run -n "$ENVIRONMENT" --no-capture-output \
                    python src/inpainting/inject_hawor_data.py "${INJECT_ARGS[@]}"
            fi

            if [ "$FORCE" = "1" ] || \
               ! fresh_file "$ARM" "$HAWOR" "$VIDEO" \
                   "$ROOT/src/inpainting/segment_arms.py"; then
                echo "[$BRANCH/$ID] SAM2 hand+arm mask"
                env MPLCONFIGDIR=/tmp/inpaint-mpl \
                    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                    conda run -n "$ENVIRONMENT" --no-capture-output \
                    python src/inpainting/segment_arms.py --processed_demo "$PD"
            fi
            validate_array "$ARM" "$EXPECTED" bool

            if [ "$FORCE" = "1" ] || \
               ! fresh_file "$MODAL" "$GT" "$HAWOR" "$VIDEO" \
                   "$ROOT/src/inpainting/segment_annotated_objects.py"; then
                echo "[$BRANCH/$ID] SAM2 annotated-object mask"
                env MPLCONFIGDIR=/tmp/inpaint-mpl \
                    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                    conda run -n "$ENVIRONMENT" --no-capture-output \
                    python src/inpainting/segment_annotated_objects.py \
                    --processed_demo "$PD" --labels_json "$GT"
            fi
            validate_array "$MODAL" "$EXPECTED" bool
        fi

        if stage_enabled depth; then
            test -s "$VIDEO"
            test -s "$PD/hand_processor/hand_data_left.npz"
            if [ "$FORCE" = "1" ] || \
               ! fresh_file "$DEPTH" "$VIDEO" "$HAWOR" "$CHECKPOINT" \
                   "$ROOT/src/inpainting/estimate_metric_depth.py" || \
               [ ! -s "$DEPTH_RAW" ] || [ ! -s "$DEPTH_PARAMS" ]; then
                echo "[$BRANCH/$ID] metric Depth Anything V2"
                conda run -n "$ENVIRONMENT" --no-capture-output \
                    python src/inpainting/estimate_metric_depth.py \
                    --processed_demo "$PD" --encoder vits --input_size 518 \
                    --checkpoint "$CHECKPOINT"
            fi
            validate_array "$DEPTH" "$EXPECTED" float
            validate_array "$DEPTH_RAW" "$EXPECTED" float
        fi

        if stage_enabled surface; then
            test -s "$DEPTH"
            test -s "$MODAL"
            if [ "$FORCE" = "1" ] || \
               ! fresh_file "$SURFACE_DIR/report.json" "$DEPTH" "$MODAL" \
                   "$HAWOR" "$VIDEO" \
                   "$ROOT/src/inpainting/build_object_surface_model.py" || \
               [ ! -s "$SURFACE" ]; then
                echo "[$BRANCH/$ID] robust visible-object camera-Z surface"
                env PYTHONPATH="$ROOT/src/inpainting" \
                    conda run -n "$ENVIRONMENT" --no-capture-output \
                    python src/inpainting/build_object_surface_model.py \
                    --scene_depth "$DEPTH" --object_mask "$MODAL" \
                    --hawor_npz "$HAWOR" --video "$VIDEO" \
                    --out_dir "$SURFACE_DIR"
            fi
            validate_array "$SURFACE" "$EXPECTED" float
        fi

        if stage_enabled completion; then
            for required in "$VIDEO" "$MODAL" "$ARM" "$HAND" "$SURFACE"; do
                test -s "$required"
            done
            if [ "$FORCE" != "1" ] && \
               completion_matches "$COMPLETION" "$OFFSET" "$EXPECTED" \
                   "$HAWOR" "$CONTACT" "$AUX_CONTACT" "$VIDEO" && \
               fresh_contact_tree "$COMPLETION/report.json" "$CONTACT" \
                   "$EXPECTED" && \
               fresh_contact_tree "$COMPLETION/report.json" "$AUX_CONTACT" \
                   "$EXPECTED" && \
               fresh_file "$COMPLETION/report.json" "$VIDEO" "$MODAL" \
                   "$ARM" "$HAND" "$SURFACE" "$HAWOR" "$GT" \
                   "$E2FGVI_CHECKPOINT" \
                   "$ROOT/src/inpainting/inpaint_object_completion.py" \
                   "$ROOT/src/inpainting/inpaint_hands.py" \
                   "$ROOT/src/inpainting/_paths.py" \
                   "$ROOT/src/inpainting/composite_rb5_contact_occlusion.py"; then
                echo "[$BRANCH/$ID] HaCo object completion is current"
            else
                echo "[$BRANCH/$ID] dual-view HaCo-selected E2FGVI completion"
                env PYTHONPATH="$ROOT/src/inpainting" \
                    INPAINT_BATCH_SIZE="$INPAINT_BATCH_SIZE" \
                    conda run -n "$ENVIRONMENT" --no-capture-output \
                    python src/inpainting/inpaint_object_completion.py \
                    --source "$VIDEO" --modal_mask "$MODAL" \
                    --arm_mask "$ARM" --hand_support "$HAND" \
                    --surface_depth "$SURFACE" --labels_json "$GT" \
                    --contact_dir "$CONTACT" \
                    --aux_contact_dir "$AUX_CONTACT" \
                    --hawor_npz "$HAWOR" --side left --aux_side left \
                    --aux_frame_offset "$OFFSET" \
                    --haco_modal_hand_exclusion_px 4 \
                    --haco_temporal_grace_frames 2 \
                    --inpaint_height 360 --out_dir "$COMPLETION"
                completion_matches "$COMPLETION" "$OFFSET" "$EXPECTED" \
                    "$HAWOR" "$CONTACT" "$AUX_CONTACT" "$VIDEO"
            fi
        fi
    done
done

if stage_enabled compare; then
    for ID in "${EPISODES[@]}"; do
        APPROX_PD="$APPROX_ROOT/$ID/camera_2/inpainting/processed/view/0"
        CALIBRATED_PD="$CALIBRATED_ROOT/$ID/camera_2/inpainting/processed/view/0"
        APPROX_ORIGINAL="$APPROX_PD/video_L.mp4"
        CALIBRATED_ORIGINAL="$CALIBRATED_PD/video_L.mp4"
        APPROX_HAWOR="$APPROX_ROOT/$ID/camera_2/rgb_hawor/retarget_input.npz"
        CALIBRATED_HAWOR="$CALIBRATED_ROOT/$ID/camera_2/rgb_hawor/retarget_input.npz"
        APPROX_COMPLETION="$APPROX_PD/object_completion_dual_haco_e2fgvi"
        CALIBRATED_COMPLETION="$CALIBRATED_PD/object_completion_dual_haco_e2fgvi"
        OUT="$OUTPUT_ROOT/episode_$ID"
        for required in "$APPROX_ORIGINAL" "$CALIBRATED_ORIGINAL" \
            "$APPROX_HAWOR" "$CALIBRATED_HAWOR" \
            "$APPROX_COMPLETION/report.json" \
            "$CALIBRATED_COMPLETION/report.json"; do
            test -s "$required"
        done
        echo "[$ID] rendering calibration inpainting A/B grid"
        env PYTHONPATH="$ROOT/src/inpainting" \
            conda run -n "$ENVIRONMENT" --no-capture-output \
            python src/inpainting/compare_calibration_inpainting_ab.py \
            --approx_original "$APPROX_ORIGINAL" \
            --calibrated_original "$CALIBRATED_ORIGINAL" \
            --expected_approx_hawor_npz "$APPROX_HAWOR" \
            --expected_calibrated_hawor_npz "$CALIBRATED_HAWOR" \
            --approx_completion_dir "$APPROX_COMPLETION" \
            --calibrated_completion_dir "$CALIBRATED_COMPLETION" \
            --out_dir "$OUT"
    done
fi

echo "Completed 08-05 calibration inpainting stages '$STAGES'."
