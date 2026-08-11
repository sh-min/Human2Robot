# Retarget trajectory to policy dataset

This module converts calibrated RBY1 + bimanual XHand trajectories and their robot
composite videos into LeRobot v3, then exports the result to GR00T-compatible
LeRobot v2.1.

## Recommended kitchen workflow

From the repository root:

```bash
TASK_SPEC=configs/tasks/kitchen.yaml bash scripts/prepare_policy_dataset.sh
```

The script discovers `IMG_*` recordings, exports and validates `final_pose.pkl`,
converts all complete episodes, and writes both configured dataset roots. The task
profile provides the instruction, episode glob, action labels and list of objects.

To call the converter directly:

```bash
PYTHONPATH=$PWD/src python -m pkl_to_lerobot.convert_batch \
  --data_root data/kitchen_dataset/26.07.24 \
  --out_dir data/lerobot_kitchen \
  --task_spec configs/tasks/kitchen.yaml \
  --visual_source robot

PYTHONPATH=$PWD/src python -m pkl_to_lerobot.export_groot_v21 \
  --source data/lerobot_kitchen \
  --out data/groot_kitchen \
  --overwrite
```

For a genuinely single-object dataset, pass `--object_spec` instead of a multi-object
task profile.

## Input

Every episode needs `rgb/` and `rgb_hawor/`:

```text
IMG_0001/
  rgb/frame_*.jpg
  rgb_hawor/
    final_pose.pkl
    retarget_input.npz
    qpos_xhand_contact_right_smooth.pkl
    qpos_xhand_contact_left_smooth.pkl
  video_overlay_rby1_xhand.mp4       optional preferred visual source
  object_pose.json                   optional episode metadata
```

Trajectory source priority is:

1. `final_pose.pkl` (calibrated wrists and finger joints)
2. contact-aware smoothed per-hand PKLs
3. contact-aware raw per-hand PKLs
4. vector-retargeted smoothed/raw PKLs

Legacy per-hand files do not carry a calibrated full-body trajectory and are rejected
unless `--allow_legacy_actions` is explicitly supplied.

## 38-D state and action

| Slice | Field | Dimension |
|---|---|---:|
| `0:12` | right XHand joints | 12 |
| `12:15` | right wrist position | 3 |
| `15:19` | right wrist quaternion `xyzw` | 4 |
| `19:31` | left XHand joints | 12 |
| `31:34` | left wrist position | 3 |
| `34:38` | left wrist quaternion `xyzw` | 4 |

Absolute actions are the default. GR00T applies its configured relative transform to
joint and wrist-position fields internally, so do not pre-delta a GR00T dataset.

## Output metadata

LeRobot v3 output contains chunked Parquet trajectories, 224×224 head-camera videos,
episode metadata and normalization statistics:

```text
data/lerobot_kitchen/
  data/chunk-000/file-*.parquet
  videos/observation.images.head_cam/chunk-000/file-*.mp4
  meta/
    info.json
    tasks.parquet
    episodes/chunk-000/file-000.parquet
    stats.json
    modality.json
    task_spec.json
```

`info.json` includes `task_id: kitchen` and every configured `object_id`.

## Training

```bash
TASK_SPEC=configs/tasks/kitchen.yaml bash scripts/train_diffusion_policy.sh
TASK_SPEC=configs/tasks/kitchen.yaml bash scripts/train_groot_policy.sh
```

Core conversion dependencies are `numpy`, `pandas`, `pyarrow`, `scipy` and system
`ffmpeg`.
