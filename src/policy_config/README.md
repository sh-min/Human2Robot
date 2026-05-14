# policy_config

LeRobot Diffusion Policy 및 GR00T N1 학습/평가 설정.

## 파일 구성

| 파일 | 역할 |
|------|------|
| `diffusion_xhand.yaml` | LeRobot Diffusion Policy 학습 설정 |
| `groot_xhand_config.py` | GR00T N1 `NEW_EMBODIMENT` modality config |
| `eval_mujoco.py` | MuJoCo 환경에서 학습된 정책 평가 |

## Diffusion Policy 학습

```bash
# 1. 데이터셋 변환 (dataset_converter 참고)
PYTHONPATH=$PWD/src python -m dataset_converter.convert_batch \
    --data_root /path/to/episodes \
    --out_dir data/lerobot_xhand_dataset

# 2. 학습
lerobot-train --config_path src/policy_config/diffusion_xhand.yaml \
    --dataset.repo_id=data/lerobot_xhand_dataset
```

## GR00T N1 학습

```bash
# 1. 데이터셋 변환 (위와 동일)

# 2. 학습 (Isaac-GR00T 환경 필요)
python gr00t_finetune.py \
    --dataset-path data/lerobot_xhand_dataset \
    --modality-config-path src/policy_config/groot_xhand_config.py \
    --embodiment-tag NEW_EMBODIMENT \
    --num-gpus 1
```

## 정책 평가 (MuJoCo)

```bash
# LeRobot 정책 평가
MUJOCO_GL=egl PYTHONPATH=$PWD/src python -m policy_config.eval_mujoco \
    --backend lerobot \
    --checkpoint /path/to/checkpoint \
    --n_episodes 10 \
    --save_video

# GR00T N1 정책 평가
MUJOCO_GL=egl PYTHONPATH=$PWD/src python -m policy_config.eval_mujoco \
    --backend groot \
    --checkpoint /path/to/checkpoint \
    --modality_config src/policy_config/groot_xhand_config.py \
    --n_episodes 10 \
    --save_video
```

## MuJoCo 환경 사양

`mujoco_sim.env.RBY1XHandEnv`는 아래 인터페이스를 제공합니다:

- **Action space**: 38-DOF absolute target qpos
  - `[0:7]` right arm, `[7:14]` left arm
  - `[14:26]` right hand (12 fingers), `[26:38]` left hand (12 fingers)
- **Observation space**:
  - `observation.images.head_cam`: (H, W, 3) uint8 이미지
  - `observation.state`: (38,) float32 현재 관절 위치

데이터셋 변환기의 38-D 벡터와 MuJoCo 환경의 38-D action space는
손가락 관절 부분이 동일한 순서를 사용합니다. 다만 변환기의 벡터에는
손목 pos/quat이 포함되어 있고 MuJoCo 환경에서는 IK로 처리됩니다.
