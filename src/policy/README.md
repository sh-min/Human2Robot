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

## Training: GR00T N1.7

The official repository is pinned as a submodule. Create its isolated Python
3.12/CUDA 12.8 environment once:

```bash
conda install -n base -c conda-forge uv git-lfs
bash scripts/bootstrap_groot.sh
```

Before the first model load, accept access for
`nvidia/Cosmos-Reason2-2B` on Hugging Face and authenticate. Prepare the
official LeRobot v2.1 layout and train:

```bash
OBJECT_SPEC=configs/objects/<object_id>.yaml \
  bash scripts/prepare_policy_dataset.sh

OBJECT_SPEC=configs/objects/<object_id>.yaml \
  bash scripts/train_groot_policy.sh
```

The single-GPU default freezes the LLM and visual backbone, tunes the
projector and diffusion action model, and uses batch 1 with gradient
accumulation 8. NVIDIA recommends substantially more memory than a typical
RTX 5080 for N1.7 fine-tuning, so this conservative default does not guarantee
the full model will fit.

Before loading weights, the training script imports the custom embodiment
config and opens episode 0 with GR00T's own loader. Incompatible metadata
therefore fails early instead of after a multi-gigabyte model download.

---

## Evaluation: MuJoCo

Evaluate any trained checkpoint in the simulated RBY1+XHand environment:

```bash
# LeRobot policy
MUJOCO_GL=egl PYTHONPATH=$PWD/src python -m policy.eval_mujoco \
    --backend lerobot \
    --checkpoint /path/to/checkpoint \
    --object_spec configs/objects/<object_id>.yaml \
    --n_episodes 10 \
    --save_video \
    --output_dir output/eval

# GR00T N1 policy
MUJOCO_GL=egl PYTHONPATH=$PWD/src python -m policy.eval_mujoco \
    --backend groot \
    --checkpoint /path/to/checkpoint \
    --modality_config src/policy/config/groot_xhand_config.py \
    --object_spec configs/objects/<object_id>.yaml \
    --n_episodes 10 \
    --save_video
```

Output:
- `output/eval/eval_metrics.json` — episode lengths, rewards, success rate
- `output/eval/episode_*.mp4` — rollout videos (if `--save_video`)

---

## Environment Details

The MuJoCo environment (`src/sim/mujoco_sim/env.py`) provides:

- **Robot**: Rainbow Robotics RBY1 + bimanual XHand
- **Action space**: 38-D absolute finger + wrist targets
  - Indices 0–11: right XHand finger joints
  - Indices 12–14: right wrist xyz in the RBY1 base frame
  - Indices 15–18: right wrist quaternion `(x, y, z, w)`
  - Indices 19–30: left XHand finger joints
  - Indices 31–33: left wrist xyz in the RBY1 base frame
  - Indices 34–37: left wrist quaternion `(x, y, z, w)`
- **Observation space**:
  - `observation.images.head_cam`: (224, 224, 3) uint8
  - `observation.state`: (38,) float32
- **Object/task**: loaded from `configs/objects/*.yaml`; reset samples the
  configured pose range and lift success is reported in rollout metrics

The finger targets map directly to the XHand joints. Wrist targets are
converted to joint-limited RBY1 arm targets with inverse kinematics during
evaluation.

## Local Diffusion Policy workflow

The preparation command discovers every ready `IMG_*` episode, exports the
calibrated RBY1-base wrist trajectory, gates it with joint-limited IK, then
uses the robot-composite video for LeRobot:

```bash
bash scripts/prepare_policy_dataset.sh
```

Train in the dedicated Python 3.12 environment:

```bash
bash scripts/train_diffusion_policy.sh
```

Useful overrides:

```bash
STEPS=1000 BATCH_SIZE=8 NUM_WORKERS=2 \
  bash scripts/train_diffusion_policy.sh
```

The calibration profile is frozen after its first fit. Adding more episodes
does not shift existing action coordinates. If a previously unseen hand
appears, the exporter adds only that hand's reference to the profile.
