#!/usr/bin/env bash
set -euo pipefail

# End-to-end, future-data-aware policy dataset preparation.
# New IMG_* episodes are discovered by each stage.  A new episode must have
# completed smoothing/retargeting and the robot composite; otherwise this
# script fails instead of silently training on a partial dataset.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA="${DATA:-$ROOT/data/cube_dataset/26.07.24}"
OUT="${OUT:-$ROOT/data/lerobot_cube_26_07_24}"
CONVERTER_ENV="${CONVERTER_ENV:-vjepa2-312}"
TASK="${TASK:-manipulate cube}"

cd "$ROOT"

DATA="$DATA" bash scripts/export_policy_trajectories.sh
DATA="$DATA" bash scripts/validate_policy_trajectories.sh

conda run -n "$CONVERTER_ENV" python -c \
  'import numpy, pandas, pyarrow, scipy'

PYTHONPATH="$ROOT/src" \
conda run -n "$CONVERTER_ENV" --no-capture-output \
  python -m pkl_to_lerobot.convert_batch \
    --data_root "$DATA" \
    --out_dir "$OUT" \
    --visual_source robot \
    --task "$TASK"

echo "Policy dataset ready: $OUT"
