#!/usr/bin/env bash
set -euo pipefail

# End-to-end, future-data-aware policy dataset preparation.
# New IMG_* episodes are discovered by each stage.  A new episode must have
# completed smoothing/retargeting and the robot composite; otherwise this
# script fails instead of silently training on a partial dataset.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONVERTER_ENV="${CONVERTER_ENV:-vjepa2-312}"
TASK_SPEC="${TASK_SPEC:-$ROOT/configs/tasks/kitchen.yaml}"
OBJECT_SPEC="${OBJECT_SPEC:-}"

task_get() {
    PYTHONPATH="$ROOT/src" conda run -n "$CONVERTER_ENV" \
      python -m task_config get "$TASK_SPEC" "$1"
}

PYTHONPATH="$ROOT/src" conda run -n "$CONVERTER_ENV" \
  python -m task_config validate "$TASK_SPEC" --check-objects

DATA="${DATA:-$(task_get dataset.recordings_root)}"
OUT="${OUT:-$(task_get dataset.lerobot_v3_root)}"
GROOT_OUT="${GROOT_OUT:-$(task_get dataset.groot_v21_root)}"
TASK="${TASK:-$(task_get instruction)}"
EPISODE_GLOB="${EPISODE_GLOB:-$(task_get dataset.episode_glob)}"

OBJECT_ARGS=()
if [ -n "$OBJECT_SPEC" ]; then
  PYTHONPATH="$ROOT/src" conda run -n "$CONVERTER_ENV" \
    python -m object_config validate "$OBJECT_SPEC" --check-assets
  OBJECT_ARGS=(--object_spec "$OBJECT_SPEC")
fi

cd "$ROOT"

DATA="$DATA" EPISODE_GLOB="$EPISODE_GLOB" \
  bash scripts/export_policy_trajectories.sh
DATA="$DATA" EPISODE_GLOB="$EPISODE_GLOB" \
  bash scripts/validate_policy_trajectories.sh

conda run -n "$CONVERTER_ENV" python -c \
  'import numpy, pandas, pyarrow, scipy'

PYTHONPATH="$ROOT/src" \
conda run -n "$CONVERTER_ENV" --no-capture-output \
  python -m pkl_to_lerobot.convert_batch \
    --data_root "$DATA" \
    --out_dir "$OUT" \
    --visual_source robot \
    --task "$TASK" \
    --task_spec "$TASK_SPEC" \
    "${OBJECT_ARGS[@]}" \
    --episode_glob "$EPISODE_GLOB"

echo "Policy dataset ready: $OUT"

PYTHONPATH="$ROOT/src" \
conda run -n "$CONVERTER_ENV" --no-capture-output \
  python -m pkl_to_lerobot.export_groot_v21 \
    --source "$OUT" \
    --out "$GROOT_OUT" \
    --overwrite

echo "GR00T v2.1 dataset ready: $GROOT_OUT"
