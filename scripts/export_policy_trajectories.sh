#!/usr/bin/env bash
set -euo pipefail

# Export a unified final_pose.pkl for every ready episode. With no arguments,
# all IMG_* directories are discovered automatically, so newly added episodes
# are picked up on the next run. Existing outputs are refreshed only when an
# input is newer; use FORCE=1 to rebuild everything.
#
# Usage:
#   bash scripts/export_policy_trajectories.sh
#   bash scripts/export_policy_trajectories.sh IMG_5019
#   FORCE=1 bash scripts/export_policy_trajectories.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA="${DATA:-$ROOT/data/kitchen_dataset/26.07.24}"
FORCE="${FORCE:-0}"
CALIBRATION="${CALIBRATION:-$DATA/workspace_calibration_rby1.json}"
EPISODE_GLOB="${EPISODE_GLOB:-IMG_*}"

if [ "$#" -gt 0 ]; then
    EPISODE_IDS=("$@")
else
    EPISODE_IDS=()
    for EP in "$DATA"/$EPISODE_GLOB; do
        [ -d "$EP" ] || continue
        EPISODE_IDS+=("$(basename "$EP")")
    done
fi

if [ "${#EPISODE_IDS[@]}" -eq 0 ]; then
    echo "No $EPISODE_GLOB episodes found under $DATA" >&2
    exit 1
fi

cd "$ROOT"
conda run -n RFM_retarget python -c \
  'import mujoco, numpy, pinocchio, scipy'

# Fit once, then keep existing hand references frozen as episodes are added.
# If a previously unseen hand appears, only that hand is appended.
PYTHONPATH="$ROOT/src" \
conda run -n RFM_retarget --no-capture-output \
  python "$ROOT/src/retargeting/fit_workspace_calibration.py" \
    --data_root "$DATA" \
    --episode_glob "$EPISODE_GLOB" \
    --out "$CALIBRATION"

for ID in "${EPISODE_IDS[@]}"; do
    HAWOR="$DATA/$ID/rgb_hawor"
    NPZ="$HAWOR/retarget_input.npz"
    RIGHT="$HAWOR/qpos_xhand_right_smooth.pkl"
    LEFT="$HAWOR/qpos_xhand_left_smooth.pkl"
    OUT="$HAWOR/final_pose.pkl"

    for INPUT in "$NPZ" "$RIGHT" "$LEFT"; do
        if [ ! -s "$INPUT" ]; then
            echo "[$ID] missing: $INPUT" >&2
            exit 1
        fi
    done

    if [ "$FORCE" = "1" ] \
       || [ ! -s "$OUT" ] \
       || [ "$NPZ" -nt "$OUT" ] \
       || [ "$RIGHT" -nt "$OUT" ] \
       || [ "$LEFT" -nt "$OUT" ] \
       || [ "$CALIBRATION" -nt "$OUT" ]; then
        (
          cd "$ROOT/src/retargeting"
          conda run -n RFM_retarget --no-capture-output \
            python export_robot_traj.py \
              --npz "$NPZ" \
              --right_pkl "$RIGHT" \
              --left_pkl "$LEFT" \
              --calibration "$CALIBRATION" \
              --out "$OUT"
        )
    else
        echo "[$ID] up to date: $OUT"
    fi
done

ID_CSV=$(IFS=,; echo "${EPISODE_IDS[*]}")
conda run -n RFM_retarget --no-capture-output python -c "
import pickle
from pathlib import Path
import numpy as np
root = Path('$DATA')
ids = '$ID_CSV'.split(',')
for episode_id in ids:
    path = root / episode_id / 'rgb_hawor' / 'final_pose.pkl'
    with path.open('rb') as handle:
        pose = pickle.load(handle)
    T = int(pose['T'])
    assert T > 0, (episode_id, T)
    assert pose.get('hands'), (episode_id, pose.get('hands'))
    for side in ('right', 'left'):
        hand = pose[side]
        assert np.asarray(hand['qpos']).shape == (T, 12)
        assert np.asarray(hand['wrist_pos']).shape == (T, 3)
        assert np.asarray(hand['wrist_rot']).shape == (T, 3, 3)
        assert np.asarray(hand['wrist_pos_camera']).shape == (T, 3)
        assert np.asarray(hand['wrist_rot_camera']).shape == (T, 3, 3)
        assert np.asarray(hand['valid']).shape == (T,)
        assert hand.get('wrist_source') == 'pkl', (episode_id, side)
        assert hand.get('coordinate_frame') == 'rby1_base', (episode_id, side)
    assert pose.get('coordinate_frame') == 'rby1_base', episode_id
    print(f'[{episode_id}] validated T={T}')
"

echo "Exported ${#EPISODE_IDS[@]} episode trajectory file(s)."
