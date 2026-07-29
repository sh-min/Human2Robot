#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA="${DATA:-$ROOT/data/cube_dataset/26.07.24}"
STRIDE="${STRIDE:-1}"
EPISODE_GLOB="${EPISODE_GLOB:-IMG_*}"

cd "$ROOT"
MUJOCO_GL="${MUJOCO_GL:-egl}" PYTHONPATH="$ROOT/src" \
conda run -n RFM_retarget --no-capture-output \
  python -m sim.mujoco_sim.validate_retarget_dataset \
    --data_root "$DATA" \
    --episode_glob "$EPISODE_GLOB" \
    --stride "$STRIDE" \
    --out "$DATA/policy_trajectory_validation.json"
