# Policy Config — Training and Evaluation

Configuration files and scripts for training policies on the converted LeRobot
dataset and evaluating them in the MuJoCo simulation.

---

## Files

| File | Purpose |
|------|---------|
| `diffusion_xhand.yaml` | LeRobot Diffusion Policy training config |
| `groot_xhand_config.py` | GR00T N1 `NEW_EMBODIMENT` modality config |
| `eval_mujoco.py` | Evaluate trained policy in MuJoCo RBY1+XHand env |

---

## Training: Diffusion Policy

Prerequisites:
```bash
pip install -e third_party/lerobot[training,diffusion]
```

Train:
```bash
lerobot-train --config_path src/policy/config/diffusion_xhand.yaml \
    --dataset.root=data/lerobot_xhand_dataset
```

Override hyperparameters:
```bash
lerobot-train --config_path src/policy/config/diffusion_xhand.yaml \
    --dataset.root=data/lerobot_xhand_dataset \
    --batch_size=32 \
    --steps=200000
```

Offline action-MSE evaluation against a held-out val dataset:
```bash
MUJOCO_GL=egl PYTHONPATH=$PWD/src python -m policy.eval_offline \
    --checkpoint output/train/<run>/checkpoints/last/pretrained_model \
    --val_dataset data/lerobot_xhand_val \
    --output_dir output/eval_offline
```

---

## Training: GR00T N1

Prerequisites:
```bash
# Clone and install Isaac-GR00T
git clone https://github.com/NVIDIA/Isaac-GR00T.git
pip install -e Isaac-GR00T
```

Train:
```bash
python gr00t_finetune.py \
    --dataset-path data/lerobot_xhand_dataset \
    --modality-config-path src/policy/config/groot_xhand_config.py \
    --embodiment-tag NEW_EMBODIMENT \
    --num-gpus 1
```

---

## Evaluation: MuJoCo

Evaluate any trained checkpoint in the simulated RBY1+XHand environment:

```bash
# LeRobot policy
MUJOCO_GL=egl PYTHONPATH=$PWD/src python -m policy.eval_mujoco \
    --backend lerobot \
    --checkpoint /path/to/checkpoint \
    --n_episodes 10 \
    --save_video \
    --output_dir output/eval

# GR00T N1 policy
MUJOCO_GL=egl PYTHONPATH=$PWD/src python -m policy.eval_mujoco \
    --backend groot \
    --checkpoint /path/to/checkpoint \
    --modality_config src/policy/config/groot_xhand_config.py \
    --n_episodes 10 \
    --save_video
```

Output:
- `output/eval/eval_metrics.json` — episode lengths, rewards
- `output/eval/episode_*.mp4` — rollout videos (if `--save_video`)

---

## Environment Details

The MuJoCo environment (`src/policy/sim/mujoco_sim/env.py`) provides:

- **Robot**: Rainbow Robotics RBY1 + bimanual XHand
- **Action space**: 38-DOF absolute target qpos
  - Indices 0–6: right arm (7 joints)
  - Indices 7–13: left arm (7 joints)
  - Indices 14–25: right hand (12 finger joints)
  - Indices 26–37: left hand (12 finger joints)
- **Observation space**:
  - `observation.images.head_cam`: (224, 224, 3) uint8
  - `observation.state`: (38,) float32

Note: The dataset's 38-D vector maps finger joints directly to the env's hand joints.
Wrist pose from the dataset is converted to arm joint targets via inverse kinematics
during evaluation.
