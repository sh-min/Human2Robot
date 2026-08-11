# Human2Robot

Human2Robot은 사람의 주방 물체 조작 영상을 RBY1/RB5 + XHand 로봇 궤적으로
변환하고, 접촉·차폐를 반영한 합성 영상과 정책 학습용 데이터셋을 만드는 연구
파이프라인입니다.

현재 기본 작업은 단일 큐브가 아니라 여러 주방 물체를 다루는 `kitchen`입니다.
행동 라벨은 `Cup`, `Lock`, `Choco`, `Snack`, `Sweep`, `Trans`이며, 물체별
형상·물성은 각각의 object spec으로 관리합니다.

## Pipeline

```text
RGB sequence
  ├─ HaWoR hand reconstruction
  ├─ HaCo contact estimation
  ├─ DexPilot XHand retargeting + smoothing
  ├─ V-JEPA feature extraction → skill classifier
  └─ SAM2 + E2FGVI + depth/amodal object layer
       └─ robot RGB-D rendering + occlusion-aware composition

final_pose.pkl + robot-composite video
  └─ LeRobot v3 / GR00T v2.1 dataset
       ├─ Diffusion Policy
       └─ NVIDIA GR00T
```

주 실행 스크립트인 `run_pipeline.sh`는 손 추정, 접촉 추정, 리타게팅,
`final_pose.pkl` 내보내기, V-JEPA 특징 추출과 기본 RGB 오버레이를 순서대로
실행합니다. 사람 손 제거와 객체 차폐를 포함한 최종 합성은
`src/inpainting/run_layered.py`가 담당합니다.

## Repository layout

```text
configs/
  tasks/kitchen.yaml        multi-object kitchen task and dataset paths
  objects/*.yaml            object geometry, physics, spawn and success rules
scripts/                    dataset, comparison, tracking and training entry points
src/
  hand_estimation/          HaWoR adapter and retarget input export
  contact_estimation/       HaCo contact extraction
  retargeting/              DexPilot XHand retargeting and overlay
  data_preprocess/          V-JEPA feature extraction
  skill_classifier/         temporal skill classification
  inpainting/               hand removal and layered object/robot composition
  pkl_to_lerobot/           LeRobot v3 and GR00T v2.1 conversion
  policy/                    policy configs, validation and MuJoCo evaluation
  sim/mujoco_sim/           RBY1 + bimanual XHand simulation
third_party/                pinned upstream repositories as Git submodules
```

Generated datasets, weights, videos, caches and simulation assets are intentionally
excluded from Git.

## Setup

Clone the submodules with the repository:

```bash
git clone --recurse-submodules git@github.com:sh-min/Human2Robot.git
cd Human2Robot
git submodule update --init --recursive
```

The full pipeline uses separate environments because upstream projects have
conflicting dependencies:

- `hawor`: HaWoR hand reconstruction
- `haco`: HaCo contact estimation
- `RFM_retarget`: DexPilot retargeting and lightweight overlay
- `vjepa2-312`: V-JEPA preprocessing and dataset conversion
- `inpaint`: SAM2, E2FGVI and layered composition
- `lerobot-312`: Diffusion Policy training
- `third_party/Isaac-GR00T/.venv`: GR00T fine-tuning

Install each upstream environment from the corresponding submodule or module README.
Large checkpoints belong under `weights/` or the paths documented by each module;
they are not committed.

## Kitchen task configuration

Validate the shared task profile and all referenced object specs:

```bash
PYTHONPATH=$PWD/src python -m task_config validate \
  configs/tasks/kitchen.yaml --check-objects
```

The default task profile contains these objects:

- blue/green cups
- chocolate milk carton
- Pringles can
- large/small lock boxes
- sponge
- trash bin

`configs/tasks/kitchen.yaml` defines dataset locations and the action vocabulary.
`configs/objects/*.yaml` defines one object's geometry and physics. In the inpainting
pipeline, `object_layer` means the segmented object pixels; it is not a task or a
specific shape.

Expected recording layout:

```text
data/kitchen_dataset/26.07.24/
  IMG_*/
    rgb/
      frame_*.jpg
    rgb_hawor/
      retarget_input.npz
      final_pose.pkl
    contact/
      *.npz
    gt_labels.json
```

## End-to-end trajectory pipeline

```bash
bash run_pipeline.sh \
  --data_dir data/kitchen_dataset/26.07.24/IMG_0001 \
  --vjepa_ckpt weights/vjepa2/vitl.pt
```

Useful switches:

```text
--no_contact       DexPilot vector retargeting only
--no_overlay       skip the basic RGB overlay
--with_slam        enable HaWoR SLAM
--skip_hand        reuse an existing retarget_input.npz
--skip_contact     reuse existing HaCo results
--skip_retarget    reuse existing robot trajectories
--skip_features    skip V-JEPA extraction
```

The main outputs are:

```text
rgb_hawor/retarget_input.npz
rgb_hawor/qpos_xhand_contact_{right,left}_smooth.pkl
rgb_hawor/final_pose.pkl
rgb_hawor/overlay_stage2_contact.mp4
features.pt
pipeline_timing.txt
```

## Inpainting and occlusion-aware overlay

The layered pipeline removes the human hand with SAM2 + E2FGVI, renders the robot as
RGB-D, and optionally restores an amodal object layer with Depth Anything V2 and
Diffusion-VAS before compositing it with the robot.

```bash
PYTHONPATH=$PWD/src python src/inpainting/run_layered.py \
  --input data/kitchen_dataset/26.07.24/IMG_0001/rgb \
  --hawor_npz data/kitchen_dataset/26.07.24/IMG_0001/rgb_hawor/retarget_input.npz \
  --right_pkl data/kitchen_dataset/26.07.24/IMG_0001/rgb_hawor/qpos_xhand_contact_right_smooth.pkl \
  --left_pkl data/kitchen_dataset/26.07.24/IMG_0001/rgb_hawor/qpos_xhand_contact_left_smooth.pkl \
  --data_root output/inpainting_raw \
  --processed_root output/inpainting \
  --object_layer
```

Omit `--object_layer` for the baseline hand-removal + robot-overlay result. The
object-layer run adds SAM2 modal tracking, monocular depth alignment and amodal object
segmentation. Detailed setup and output paths are in
[`src/inpainting/README.md`](src/inpainting/README.md).

## Policy dataset and training

Export validated trajectories and build both dataset formats:

```bash
TASK_SPEC=configs/tasks/kitchen.yaml bash scripts/prepare_policy_dataset.sh
```

This writes the task metadata into `meta/task_spec.json` and records `task_id` plus
all `object_ids` in `meta/info.json`.

Train a policy:

```bash
TASK_SPEC=configs/tasks/kitchen.yaml bash scripts/train_diffusion_policy.sh
TASK_SPEC=configs/tasks/kitchen.yaml bash scripts/train_groot_policy.sh
```

The dataset command accepts `OBJECT_SPEC=...` only for an explicit single-object run;
the kitchen default uses the multi-object task profile.

## Tests

```bash
PYTHONPATH=$PWD/src python -m pytest -q
bash -n run_pipeline.sh scripts/*.sh
```

## Module documentation

- [`src/hand_estimation/README.md`](src/hand_estimation/README.md)
- [`src/contact_estimation/README.md`](src/contact_estimation/README.md)
- [`src/retargeting/README.md`](src/retargeting/README.md)
- [`src/inpainting/README.md`](src/inpainting/README.md)
- [`docs/trex_haco_visibility.md`](docs/trex_haco_visibility.md) — T-Rex-style HaCo 차폐 실험
- [`src/data_preprocess/README.md`](src/data_preprocess/README.md)
- [`src/skill_classifier/README.md`](src/skill_classifier/README.md)
- [`src/pkl_to_lerobot/README.md`](src/pkl_to_lerobot/README.md)
- [`src/policy/README.md`](src/policy/README.md)
- [`src/sim/mujoco_sim/README.md`](src/sim/mujoco_sim/README.md)
