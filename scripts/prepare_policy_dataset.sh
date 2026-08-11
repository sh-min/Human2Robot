#!/usr/bin/env bash
set -euo pipefail

# End-to-end, future-data-aware policy dataset preparation.
# New IMG_* episodes are discovered by each stage.  A new episode must have
# completed smoothing/retargeting and the robot composite; otherwise this
# script fails instead of silently training on a partial dataset.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONVERTER_ENV="${CONVERTER_ENV:-vjepa2-312}"
OBJECT_SPEC="${OBJECT_SPEC:-$ROOT/configs/objects/cube.yaml}"

spec_get() {
    PYTHONPATH="$ROOT/src" conda run -n "$CONVERTER_ENV" \
      python -m object_config get "$OBJECT_SPEC" "$1"
}

PYTHONPATH="$ROOT/src" conda run -n "$CONVERTER_ENV" \
  python -m object_config validate "$OBJECT_SPEC" --check-assets

DATA="${DATA:-$(spec_get dataset.recordings_root)}"
OUT="${OUT:-$(spec_get dataset.lerobot_v3_root)}"
GROOT_OUT="${GROOT_OUT:-$(spec_get dataset.groot_v21_root)}"
TASK="${TASK:-$(spec_get task.instruction)}"
EPISODE_GLOB="${EPISODE_GLOB:-$(spec_get dataset.episode_glob)}"

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
    --object_spec "$OBJECT_SPEC" \
    --episode_glob "$EPISODE_GLOB"

echo "Policy dataset ready: $OUT"

PYTHONPATH="$ROOT/src" \
conda run -n "$CONVERTER_ENV" --no-capture-output \
  python -m pkl_to_lerobot.export_groot_v21 \
    --source "$OUT" \
    --out "$GROOT_OUT" \
    --object_spec "$OBJECT_SPEC" \
    --overwrite

echo "GR00T v2.1 dataset ready: $GROOT_OUT"
