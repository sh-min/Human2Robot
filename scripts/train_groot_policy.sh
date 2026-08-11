#!/usr/bin/env bash
set -euo pipefail

# Fine-tune official Isaac-GR00T N1.7 for RBY1 + XHand NEW_EMBODIMENT.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GROOT_ROOT="${GROOT_ROOT:-$ROOT/third_party/Isaac-GR00T}"
TASK_SPEC="${TASK_SPEC:-$ROOT/configs/tasks/kitchen.yaml}"
CONFIG="${CONFIG:-$ROOT/src/policy/config/groot_xhand_config.py}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
MAX_STEPS="${MAX_STEPS:-2000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_STEPS="${SAVE_STEPS:-500}"
RUN_NAME="${RUN_NAME:-groot_xhand_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/train/$RUN_NAME}"

test -x "$GROOT_ROOT/.venv/bin/python" || {
    echo "GR00T environment missing: bash scripts/bootstrap_groot.sh" >&2
    exit 1
}

DATASET="${DATASET:-$(
  PYTHONPATH="$ROOT/src" "$GROOT_ROOT/.venv/bin/python" \
    -m task_config get "$TASK_SPEC" dataset.groot_v21_root
)}"

for required in \
  "$DATASET/meta/info.json" \
  "$DATASET/meta/episodes.jsonl" \
  "$DATASET/meta/tasks.jsonl" \
  "$DATASET/meta/modality.json"; do
    test -s "$required" || {
        echo "Missing GR00T dataset file: $required" >&2
        echo "Run: TASK_SPEC=$TASK_SPEC bash scripts/prepare_policy_dataset.sh" >&2
        exit 1
    }
done

PYTHONPATH="$ROOT/src:$GROOT_ROOT" \
  "$GROOT_ROOT/.venv/bin/python" -m policy.validate_groot_setup \
    --dataset "$DATASET" \
    --modality_config "$CONFIG"

cd "$GROOT_ROOT"
HF_HOME="${HF_HOME:-$ROOT/weights/huggingface}" \
UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/weights/uv}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$GROOT_ROOT/.venv/bin/python" gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATASET" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$CONFIG" \
    --num-gpus 1 \
    --output-dir "$OUTPUT_DIR" \
    --save-total-limit 3 \
    --save-steps "$SAVE_STEPS" \
    --max-steps "$MAX_STEPS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --dataloader-num-workers "$NUM_WORKERS"

echo "GR00T training output: $OUTPUT_DIR"
