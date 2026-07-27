#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-lerobot-312}"
DATASET="${DATASET:-$ROOT/data/lerobot_cube_26_07_24}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RUN_NAME="${RUN_NAME:-diffusion_xhand_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/train/$RUN_NAME}"

cd "$ROOT"

test -s "$DATASET/meta/info.json" || {
    echo "Missing LeRobot dataset: $DATASET" >&2
    echo "Run: bash scripts/prepare_policy_dataset.sh" >&2
    exit 1
}

HF_HOME="$ROOT/weights/huggingface" \
HF_DATASETS_CACHE="$ROOT/weights/huggingface/datasets" \
TORCH_HOME="$ROOT/weights/torch" \
  conda run -n "$ENV_NAME" --no-capture-output \
  lerobot-train \
    --config_path="$ROOT/src/policy/config/diffusion_xhand.yaml" \
    --dataset.root="$DATASET" \
    --dataset.video_backend=torchcodec \
    --steps="$STEPS" \
    --batch_size="$BATCH_SIZE" \
    --num_workers="$NUM_WORKERS" \
    --output_dir="$OUTPUT_DIR" \
    --wandb.enable=false

echo "Training output: $OUTPUT_DIR"
