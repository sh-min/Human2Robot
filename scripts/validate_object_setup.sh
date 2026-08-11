#!/usr/bin/env bash
set -euo pipefail

# Fast preflight before dataset preparation or policy training.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-lerobot-312}"
OBJECT_SPEC="${OBJECT_SPEC:-$ROOT/configs/objects/milk_carton.yaml}"
SCENE_OUT="$(mktemp --suffix=.xml /tmp/rby1_xhand_object.XXXXXX)"
trap 'rm -f "$SCENE_OUT"' EXIT

cd "$ROOT"

PYTHONPATH="$ROOT/src" conda run -n "$ENV_NAME" --no-capture-output \
  python -m object_config validate "$OBJECT_SPEC" --check-assets

PYTHONPATH="$ROOT/src" MUJOCO_GL="${MUJOCO_GL:-egl}" \
  conda run -n "$ENV_NAME" --no-capture-output \
  python -m sim.mujoco_sim.object_scene \
    --object_spec "$OBJECT_SPEC" \
    --out "$SCENE_OUT" \
    --check

for key in recordings_root lerobot_v3_root groot_v21_root; do
    path="$(PYTHONPATH="$ROOT/src" conda run -n "$ENV_NAME" \
      python -m object_config get "$OBJECT_SPEC" "dataset.$key")"
    state="will be created"
    test -e "$path" && state="exists"
    echo "$key: $path ($state)"
done
episode_glob="$(PYTHONPATH="$ROOT/src" conda run -n "$ENV_NAME" \
  python -m object_config get "$OBJECT_SPEC" dataset.episode_glob)"
echo "episode_glob: $episode_glob"

echo "Object configuration and MuJoCo scene are valid."
