#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRIAL="$ROOT/output/model_compare/vace/No1__IMG_5367_f19_67"
PREPROCESSED="$TRIAL/official_swap_preprocess"
CHECKPOINT="$ROOT/output/model_compare/vace/checkpoints/Wan2.1-VACE-1.3B"
VACE_ROOT="$ROOT/third_party/VACE"
WAN_ROOT="$ROOT/third_party/Wan2.1"
OUTPUT="$TRIAL/vace_hangcup_49f_official_swap_seed2025.mp4"

PROMPT="A fixed overhead documentary-style video of a bright white tabletop. A single realistic matte-gray industrial robot arm enters from the left. Its compact mechanical two-finger gripper firmly holds a dark navy mug, moves the mug toward the metal cup rack, and hangs it on the rack. The robot has rigid articulated joints, metallic links, consistent mechanical geometry, and no human skin or human fingers. The dark navy mug, metal rack, containers, snack boxes, tabletop arrangement, bright lighting, and stationary camera remain visually consistent throughout the motion. Photorealistic materials and temporally stable movement."

test -e /dev/nvidia0 || {
    echo "CUDA device is not visible: /dev/nvidia0 is missing" >&2
    exit 2
}
test -s "$PREPROCESSED/src_video-swap_anything.mp4"
test -s "$PREPROCESSED/src_mask-swap_anything.mp4"
test -s "$PREPROCESSED/src_ref_image_0-swap_anything.png"
test -s "$CHECKPOINT/config.json"

export PYTHONPATH="$VACE_ROOT:$WAN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

conda run -n vace --no-capture-output \
    python "$VACE_ROOT/vace/vace_wan_inference.py" \
    --model_name vace-1.3B \
    --size 480p \
    --frame_num 49 \
    --ckpt_dir "$CHECKPOINT" \
    --src_video "$PREPROCESSED/src_video-swap_anything.mp4" \
    --src_mask "$PREPROCESSED/src_mask-swap_anything.mp4" \
    --src_ref_images "$PREPROCESSED/src_ref_image_0-swap_anything.png" \
    --prompt "$PROMPT" \
    --use_prompt_extend plain \
    --offload_model true \
    --t5_cpu \
    --sample_steps 30 \
    --base_seed 2025 \
    --save_file "$OUTPUT"

echo "[ok] $OUTPUT"
