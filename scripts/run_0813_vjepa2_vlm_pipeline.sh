#!/usr/bin/env bash
set -euo pipefail

cd /home/robin/shMin/skill2policy
export PYTHONPATH="$PWD:$PWD/src"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR=/tmp/skill-classifier-mpl

conda run --no-capture-output -n vjepa2-312 python -m data_preprocess.extract_vlm_sam_object_context \
  --data-root data/vjepa_training/kitchen_0813_human_vjepa2_color_jitter \
  --recording-glob 'No[1-8]__IMG_*,No9__IMG_*,No10__IMG_*' \
  --semantics src/skill_classifier/config/kitchen_action_semantics.yaml \
  --context-key vlm_sam_object_context \
  --grounding-model IDEA-Research/grounding-dino-base \
  --grounding-prompt-batch-size 8 \
  --sam2-root third_party/sam2 \
  --sam2-checkpoint third_party/sam2/checkpoints/sam2_hiera_large.pt \
  --sam2-config sam2_hiera_l.yaml \
  --device cuda

conda run --no-capture-output -n vjepa2-312 python -m skill_classifier.train \
  --config src/skill_classifier/config/kitchen_0813_human_vjepa2_vlm_sam_color_jitter.yaml \
  --exp_id dropout_04_seed_42

conda run --no-capture-output -n vjepa2-312 python -m skill_classifier.evaluate_validation_long_sequences \
  --checkpoint output/skill_classifier/kitchen_0813_human_vjepa2_vlm_sam_color_jitter/dropout_04_seed_42/best_object_mask_attention_mlp.pt \
  --feature-root data/vjepa_training/kitchen_0813_human_vjepa2_color_jitter \
  --recording-glob 'No9__IMG_*,No10__IMG_*' \
  --output-dir output/skill_classifier/kitchen_0813_human_vjepa2_vlm_sam_color_jitter/validation_auto_annotation_tolerance \
  --tolerance-frames 0,2,4,8,15,30 \
  --device cuda
