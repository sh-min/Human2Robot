# Dataset Converter — Retarget PKL to LeRobot Format

Converts the output of the retargeting pipeline (`final_pose.pkl` + RGB frames) into a
[LeRobot v2](https://huggingface.co/docs/lerobot) dataset that can be directly used for
training **Diffusion Policy** (via `lerobot-train`) or **GR00T N1** (via `gr00t_finetune.py`).

---

## Quick Start

```bash
# From the repo root
cd /path/to/skill2policy

# Convert a single episode
PYTHONPATH=$PWD/src python -m dataset_converter.convert_episode \
    --episode_dir data/raw/cube_001 \
    --out_dir data/lerobot_xhand_dataset \
    --episode_index 0

# Convert all episodes in a directory
PYTHONPATH=$PWD/src python -m dataset_converter.convert_batch \
    --data_root data/raw \
    --out_dir data/lerobot_xhand_dataset \
    --task "manipulate rubik's cube"
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

### Output: LeRobot v2 Dataset

```
skill2policy/
└── data/
    └── lerobot_xhand_dataset/        ← generated output
        ├── data/
        │   └── chunk-000/
        │       ├── episode_000000.parquet
        │       ├── episode_000001.parquet
        │       └── ...
        ├── videos/
        │   └── chunk-000/
        │       └── observation.images.head_cam/
        │           ├── episode_000000.mp4
        │           └── ...
        └── meta/
            ├── info.json             ← dataset metadata
            ├── episodes.jsonl        ← episode list
            ├── tasks.jsonl           ← task descriptions
            ├── stats.json            ← normalization statistics
            └── modality.json         ← GR00T N1 modality config
```

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
| 12:15 | `right_wrist_pos` | 3 | xyz position (camera frame, meters) |
| 15:19 | `right_wrist_quat` | 4 | orientation quaternion (xyzw) |
| 19:31 | `left_hand_joint` | 12 | finger joint angles (rad) |
| 31:34 | `left_wrist_pos` | 3 | xyz position |
| 34:38 | `left_wrist_quat` | 4 | orientation quaternion (xyzw) |

---

## Action Modes

| Mode | Formula | Use Case |
|------|---------|----------|
| `absolute` (default) | `action[t] = state[t+1]` | Simpler; good for Diffusion Policy |
| `delta` | `action[t] = state[t+1] - state[t]` | GR00T N1 relative mode |

```bash
# Use delta actions
PYTHONPATH=$PWD/src python -m dataset_converter.convert_batch \
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
lerobot-train --config_path src/policy_config/diffusion_xhand.yaml \
    --dataset.repo_id=data/lerobot_xhand_dataset \
    --dataset.local_files_only=true
```

### GR00T N1

```bash
# Install Isaac-GR00T (separate repo)
# See: https://github.com/NVIDIA/Isaac-GR00T

# Train
python gr00t_finetune.py \
    --dataset-path data/lerobot_xhand_dataset \
    --modality-config-path src/policy_config/groot_xhand_config.py \
    --embodiment-tag NEW_EMBODIMENT \
    --num-gpus 1
```

---

## Evaluation (MuJoCo)

```bash
# Evaluate a trained policy in the RBY1+XHand simulation
MUJOCO_GL=egl PYTHONPATH=$PWD/src python -m policy_config.eval_mujoco \
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
| `--task` | `manipulate cube` | Task description string |

### `convert_batch`

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | (required) | Root dir containing episode sub-dirs |
| `--out_dir` | (required) | Output dataset root |
| `--fps` | `30.0` | Video frame rate |
| `--action_mode` | `absolute` | `absolute` or `delta` |
| `--img_glob` | `frame_*.jpg` | Glob pattern for RGB frames |
| `--task` | `manipulate cube` | Task description string |

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
