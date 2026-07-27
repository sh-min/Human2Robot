# Dataset Converter — Retarget PKL to LeRobot Format

Converts the output of the retargeting pipeline (`final_pose.pkl` + robot-composite video) into a
[LeRobot v3.0](https://huggingface.co/docs/lerobot) dataset that can be directly used for
training **Diffusion Policy** (LeRobot v3) or **GR00T N1.7** (exported
LeRobot v2.1).
Video frames are downscaled to 224×224 during encoding so they match the default
Diffusion Policy image encoder.

---

## Quick Start

```bash
# From the repo root
cd /path/to/skill2policy

# Convert a single episode
PYTHONPATH=$PWD/src python -m pkl_to_lerobot.convert_episode \
    --episode_dir data/raw/cube_001 \
    --out_dir data/lerobot_xhand_dataset \
    --episode_index 0

# Export and validate calibrated robot trajectories first.  With no episode
# arguments, future IMG_* directories are discovered automatically.
bash scripts/export_policy_trajectories.sh
bash scripts/validate_policy_trajectories.sh

# Convert all episodes in a directory.  "auto" prefers each episode's
# video_overlay_rby1_xhand.mp4 and only falls back to raw RGB.
PYTHONPATH=$PWD/src python -m pkl_to_lerobot.convert_batch \
    --data_root data/raw \
    --out_dir data/lerobot_xhand_dataset \
    --visual_source auto \
    --object_spec configs/objects/cube.yaml

# Or run trajectory export, validation, v3 conversion, and v2.1 export:
OBJECT_SPEC=configs/objects/cube.yaml \
  bash scripts/prepare_policy_dataset.sh
```

---

## Directory Layout

### Input: Raw Episode Data

Place your retargeted episodes under `data/raw/`. Each episode directory must contain
`rgb/` and `rgb_hawor/`:

```
skill2policy/
└── data/
    └── raw/                          ← put your episodes here
        ├── cube_001/
        │   ├── rgb/
        │   │   ├── frame_000001.jpg
        │   │   ├── frame_000002.jpg
        │   │   └── ...
        │   ├── rgb_hawor/
        │   │   ├── final_pose.pkl    ← PRIMARY (has wrist pose + fingers)
        │   │   ├── qpos_xhand_contact_right.pkl  ← fallback (fingers only)
        │   │   ├── qpos_xhand_contact_left.pkl
        │   │   └── retarget_input.npz
        │   └── contact/              (not used by converter)
        ├── cube_002/
        │   ├── rgb/ ...
        │   └── rgb_hawor/ ...
        └── ...
```

### Output: LeRobot v3.0 Dataset

```
skill2policy/
└── data/
    └── lerobot_xhand_dataset/        ← generated output
        ├── data/
        │   └── chunk-000/
        │       ├── file-000.parquet  ← episode 0 (one file per episode)
        │       ├── file-001.parquet
        │       └── ...
        ├── videos/
        │   └── observation.images.head_cam/
        │       └── chunk-000/
        │           ├── file-000.mp4  ← 224×224 @ fps
        │           └── ...
        └── meta/
            ├── info.json             ← v3.0 dataset metadata (features dict)
            ├── tasks.parquet         ← task table
            ├── episodes/
            │   └── chunk-000/
            │       └── file-000.parquet   ← per-episode metadata + stats
            ├── stats.json            ← aggregate normalization stats
            ├── modality.json         ← GR00T modality mapping
            └── object_spec.json      ← normalized object/task configuration
```

The GR00T v2.1 output root contains `meta/episodes.jsonl`,
`meta/tasks.jsonl`, and the official language mapping
`annotation.human.task_description`. Large parquet/video files are
hard-linked when both outputs share a filesystem.

---

## Data Source Priority

The converter searches for data in the following order:

1. **`final_pose.pkl`** (preferred) — contains both finger qpos AND wrist pose for both hands
2. **`qpos_xhand_contact_{hand}_smooth.pkl`** — contact-retargeted + smoothed (fingers only)
3. **`qpos_xhand_contact_{hand}.pkl`** — contact-retargeted (fingers only)
4. **`qpos_xhand_{hand}_smooth.pkl`** — basic retarget + smoothed (fingers only)
5. **`qpos_xhand_{hand}.pkl`** — basic retarget (fingers only)

When using `final_pose.pkl`, the full 38-D state vector is populated with wrist data.
When using per-hand pkl files (which lack wrist info), the wrist fields are zero-filled.

---

## State/Action Vector Layout (38-D)

| Index | Field | Dim | Source |
|-------|-------|-----|--------|
| 0:12 | `right_hand_joint` | 12 | finger joint angles (rad) |
| 12:15 | `right_wrist_pos` | 3 | xyz position (calibrated RBY1 base frame, meters) |
| 15:19 | `right_wrist_quat` | 4 | orientation quaternion (xyzw) |
| 19:31 | `left_hand_joint` | 12 | finger joint angles (rad) |
| 31:34 | `left_wrist_pos` | 3 | xyz position (calibrated RBY1 base frame) |
| 34:38 | `left_wrist_quat` | 4 | orientation quaternion (xyzw) |

---

## Action Modes

| Mode | Formula | Use Case |
|------|---------|----------|
| `absolute` (default) | `action[t] = state[t+1]` | Simpler; good for Diffusion Policy |
| `delta` | `action[t] = state[t+1] - state[t]` | Legacy/custom policies only |

GR00T should use the default absolute dataset. Its
`ActionRepresentation.RELATIVE` setting performs the relative transform
inside the official processor; pre-delta actions would apply it twice.

```bash
# Use delta actions
PYTHONPATH=$PWD/src python -m pkl_to_lerobot.convert_batch \
    --data_root data/raw \
    --out_dir data/lerobot_xhand_dataset_delta \
    --action_mode delta
```

---

## Training

### Diffusion Policy (LeRobot)

```bash
# Install lerobot (from the submodule)
pip install -e third_party/lerobot[training,diffusion]

# Train
lerobot-train --config_path src/policy/config/diffusion_xhand.yaml \
    --dataset.root=data/lerobot_xhand_dataset
```

### GR00T N1.7

```bash
bash scripts/bootstrap_groot.sh
OBJECT_SPEC=configs/objects/cube.yaml bash scripts/train_groot_policy.sh
```

---

## Evaluation (MuJoCo)

```bash
# Evaluate a trained policy in the RBY1+XHand simulation
MUJOCO_GL=egl PYTHONPATH=$PWD/src python -m policy.eval_mujoco \
    --backend lerobot \
    --checkpoint output/train/checkpoint \
    --n_episodes 10 \
    --save_video \
    --output_dir output/eval
```

---

## CLI Reference

### `convert_episode`

| Argument | Default | Description |
|----------|---------|-------------|
| `--episode_dir` | (required) | Path to one episode directory |
| `--out_dir` | (required) | Output dataset root |
| `--episode_index` | `0` | Index for this episode |
| `--fps` | `30.0` | Video frame rate |
| `--action_mode` | `absolute` | `absolute` or `delta` |
| `--img_glob` | `frame_*.jpg` | Glob pattern for RGB frames |
| `--task` | `manipulate object` | Task description string |
| `--visual_source` | `auto` | Prefer robot composite; `robot`, `rgb`, or a path also accepted |

### `convert_batch`

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | (required) | Root dir containing episode sub-dirs |
| `--out_dir` | (required) | Output dataset root |
| `--fps` | `30.0` | Video frame rate |
| `--action_mode` | `absolute` | `absolute` or `delta` |
| `--img_glob` | `frame_*.jpg` | Glob pattern for RGB frames |
| `--task` | object instruction or `manipulate object` | Task description string |
| `--object_spec` | none | Shared object/task YAML copied into metadata |
| `--episode_glob` | object spec or `*` | Episode directory name pattern |

---

## Dependencies

```
numpy
scipy
pyarrow
ffmpeg          (system, via brew/apt)
```

Install Python deps:
```bash
pip install numpy scipy pyarrow
```
