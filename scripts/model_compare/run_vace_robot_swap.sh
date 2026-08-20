#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRIAL="${TRIAL:-$ROOT/output/model_compare/vace/07_27-26_cup_f183_263}"
CHECKPOINT="${CHECKPOINT:-$ROOT/output/model_compare/vace/checkpoints/Wan2.1-VACE-1.3B}"
VACE_ROOT="${VACE_ROOT:-$ROOT/third_party/VACE}"
WAN_ROOT="${WAN_ROOT:-$ROOT/third_party/Wan2.1}"
FRAME_NUM="${FRAME_NUM:-81}"
OUTPUT="${OUTPUT:-$TRIAL/vace_robot_swap_${FRAME_NUM}f_seed2025.mp4}"
STEPS="${STEPS:-30}"
SEED="${SEED:-2025}"

PROMPT="A fixed first-person camera view in a bright real kitchen. A realistic gray bimanual robot arm with a compact industrial gripper reaches forward and grasps the blue cup, following the exact original hand trajectory. Preserve the table, cup, objects, lighting, camera, and all unmasked pixels. Photorealistic robot embodiment, temporally stable geometry, consistent joints and gripper."

test -s "$TRIAL/vace_src_video.mp4"
test -s "$TRIAL/vace_src_mask.mp4"
test -s "$TRIAL/robot_reference_white.png"
test -s "$CHECKPOINT/config.json"

export PYTHONPATH="$VACE_ROOT:$WAN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

conda run -n vace --no-capture-output \
    python "$VACE_ROOT/vace/vace_wan_inference.py" \
    --model_name vace-1.3B \
    --size 480p \
    --frame_num "$FRAME_NUM" \
    --ckpt_dir "$CHECKPOINT" \
    --src_video "$TRIAL/vace_src_video.mp4" \
    --src_mask "$TRIAL/vace_src_mask.mp4" \
    --src_ref_images "$TRIAL/robot_reference_white.png" \
    --prompt "$PROMPT" \
    --use_prompt_extend plain \
    --offload_model true \
    --t5_cpu \
    --sample_steps "$STEPS" \
    --base_seed "$SEED" \
    --save_file "$OUTPUT"

echo "[ok] VACE result: $OUTPUT"
