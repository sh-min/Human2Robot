#!/usr/bin/env bash
set -euo pipefail

# Run after the restored 07/24/27/28 overlay batch completes:
#   1. audit all video/annotation/trajectory contracts;
#   2. build the visual overlay QA gallery;
#   3. extract V-JEPA features from robot overlays and train the skill model;
#   4. visualize raw and classifier-adapted V-JEPA embedding clusters.
#
# Diffusion Policy and its LeRobot/trajectory preparation are deliberately
# excluded from this automatic workflow. They remain available as separate
# scripts for a later explicitly requested policy experiment.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$ROOT/data/robot_overlay/kitchen_0724_0728}"
POLICY_STAGE="$WORK_ROOT/policy_episodes"
VJEPA_STAGE="$ROOT/data/vjepa_training/kitchen_0724_0728_robot_overlay_no_choco"
VJEPA_CONFIG="$ROOT/src/skill_classifier/config/kitchen_0724_0728_robot_overlay_vjepa_only_no_choco.yaml"
VJEPA_OUTPUT="$ROOT/output/skill_classifier/kitchen_0724_0728_robot_overlay_vjepa_only_no_choco/seed_42"

cd "$ROOT"

conda run -n RFM_retarget --no-capture-output \
python "$SCRIPT_DIR/audit_restored_kitchen_overlays.py" \
    --work_root "$WORK_ROOT" \
    --source "0724=$ROOT/data/cube_dataset/26.07.24" \
    --source "0727=$ROOT/data/cube_dataset/26.07.27" \
    --source "0728=$ROOT/data/cube_dataset/26.07.28" \
    --policy_stage "$POLICY_STAGE" \
    --vjepa_stage "$VJEPA_STAGE" \
    --expected_total 45

python "$SCRIPT_DIR/build_restored_kitchen_visual_qc.py" \
    --manifest "$WORK_ROOT/policy_manifest.json" \
    --output_dir "$WORK_ROOT/visual_qc"

PYTHONPATH="$ROOT/src" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n vjepa2-312 --no-capture-output \
    python -m data_preprocess.preprocess \
        --data_root "$VJEPA_STAGE" \
        --recording_glob '*' \
        --checkpoint "$ROOT/weights/vjepa2/vitl.pt" \
        --batch_size 8 \
        --sampling_profile vjepa2_4fps \
        --action_labels Cup,Lock,Milk,Snack,Sweep,Trans

if [ -s "$VJEPA_OUTPUT/evaluation_summary.json" ]; then
    echo "[skip] V-JEPA overlay classifier already trained: $VJEPA_OUTPUT"
else
    PYTHONPATH="$ROOT/src" \
    conda run -n vjepa2-312 --no-capture-output \
        python -m skill_classifier.train \
            --config "$VJEPA_CONFIG" \
            --exp_id seed_42
fi

PYTHONPATH="$ROOT/src" \
conda run -n vjepa2-312 --no-capture-output \
    python -m skill_classifier.visualize_training_report \
        --experiment_dir "$VJEPA_OUTPUT"

PYTHONPATH="$ROOT/src" \
conda run -n vjepa2-312 --no-capture-output \
    python -m skill_classifier.visualize_vjepa_clusters \
        --config "$VJEPA_CONFIG" \
        --experiment_dir "$VJEPA_OUTPUT"

echo "[ok] restored kitchen V-JEPA workflow complete"
echo "[ok] manifest: $WORK_ROOT/policy_manifest.json"
echo "[ok] V-JEPA:   $VJEPA_OUTPUT"
echo "[ok] clusters: $VJEPA_OUTPUT/clustering/index.html"
echo "[skip] Diffusion Policy was not requested and was not started"
