# skill2policy

## LLM pair-coding tools

This repo is built assuming you'll be pair-coding with an LLM (Claude Code or
similar) and ships with two helpers. Skim them first when adding a new module
or trying to understand code a teammate wrote.

### 1. [`CLAUDE.md`](CLAUDE.md) — coding guidelines

Four behavioral rules in the spirit of Andrej Karpathy's coding-with-LLM notes:
*Think before coding* / *Simplicity first* / *Surgical changes* /
*Goal-driven execution*. Claude Code reads this file automatically at the start
of each session. You shouldn't normally need to touch it — append project-
specific rules below the existing ones if you want to extend it.

### 2. `graphify` — codebase knowledge graph

Turns the whole `src/` tree into a node/edge/community graph so an LLM can
quickly grasp how modules connect. Useful when comparing structure before and
after merging a new module, or finding the seam between your code and a
collaborator's.

**Install (once):**

```bash
pip install graphifyy
graphify install --platform claude   # registers the /graphify slash command 
```

**Build the graph** (`src/` only — skip `third_party/`):

```bash
# Inside Claude Code
/graphify src

# Or directly from the shell
graphify build src
```

Outputs:
- `GRAPH_REPORT.md` — god nodes, surprising connections, suggested questions
  (the human-readable summary; read this first)
- `graphify-out/graph.html` — interactive visualization (open in a browser)
- `graphify-out/graph.json` — GraphRAG-ready index for LLM queries

**Query / explore:**

```bash
graphify query "where is features.pt produced and where is it consumed?"
graphify path "VJEPAFeatureExtractor" "SkillWindowDataset"
graphify explain "infer_long_horizon.main"
```

**After refactoring or adding a new module:**

```bash
graphify update      # re-extracts only changed files (uses cache)
```

Re-running this once after dropping in a new folder (e.g. `src/policy/`,
`src/sim/`) lets the LLM lock onto the overall structure faster in later
sessions, with less context burned on rediscovery.

---

## Cloning

This repository uses git submodules. A plain `git clone` will leave the `third_party/` directories empty.

**First-time clone:**

```bash
git clone --recurse-submodules https://github.com/<your-org>/skill2policy.git
```

**Already cloned without `--recurse-submodules`:**

```bash
git submodule update --init --recursive
```

### Third-party dependencies (`third_party/`)

| Directory | Repository | 용도 |
|---|---|---|
| `vjepa2` | https://github.com/facebookresearch/vjepa2 | V-JEPA 2 video feature backbone |
| `HACO_RELEASE` | https://github.com/dqj5182/HACO_RELEASE | hand-contact estimation |
| `HaWoR` | https://github.com/ThunderVVV/HaWoR | RGB → MANO hand pose |
| `dex-retargeting` | https://github.com/dexsuite/dex-retargeting | MANO → xhand qpos retargeting |
| `lerobot` | https://github.com/huggingface/lerobot | robot policy training infra |
| `sam2` | https://github.com/facebookresearch/sam2 | hand/arm segmentation for inpainting |
| `E2FGVI` | https://github.com/MCG-NKU/E2FGVI | flow-guided video inpainting |
| `Isaac-GR00T` | https://github.com/NVIDIA/Isaac-GR00T | GR00T N1.7 VLA fine-tuning |

### Updating submodules

To pull the latest commits from all submodules:

```bash
git submodule update --remote --recursive
```

---

## Object-ready policy setup

Object-dependent values are centralized in one YAML file. Copy
`configs/objects/template.yaml` to `configs/objects/<object_id>.yaml`, then
provide the task instruction, geometry, physics, spawn/randomization,
success condition, active hands, episode directory glob, and dataset paths.
Mesh assets belong under `assets/objects/<object_id>/`.

Geometry may be a primitive, visual/collision meshes, or a standalone MJCF.
The MuJoCo object branch is integrated as ready-to-use specs:
`cup_blue`, `cup_green`, `milk_carton`, `pringles`, `lock_box_large`,
`lock_box_small`, `sponge`, and `trash_bin`.

```bash
# Validate the spec, assets, and generated MuJoCo scene
OBJECT_SPEC=configs/objects/<object_id>.yaml \
  bash scripts/validate_object_setup.sh

# Export trajectories and build LeRobot v3 + GR00T v2.1 datasets
OBJECT_SPEC=configs/objects/<object_id>.yaml \
  bash scripts/prepare_policy_dataset.sh

# Train either backend
OBJECT_SPEC=configs/objects/<object_id>.yaml \
  bash scripts/train_diffusion_policy.sh

bash scripts/bootstrap_groot.sh
OBJECT_SPEC=configs/objects/<object_id>.yaml \
  bash scripts/train_groot_policy.sh
```

Future ready episodes are discovered automatically instead of using a fixed
recording list. The included `cube.yaml` preserves the current dataset as a
working example. A new object does not require policy-code changes, but still
requires demonstration recordings, calibrated trajectories, and realistic
geometry/physics values for closed-loop evaluation.

`spawn.randomization` randomizes MuJoCo evaluation; it does not invent new
training images. Object-position generalization therefore depends on
recordings that actually cover the intended workspace (and on the object
remaining visible in the robot-replacement observation).

### 물체 실측 무게

`physics.mass_kg` 값을 채울 때 쓰는 실측 무게 (2026-08-12 기록).

| 물체 | 무게 (g) | mass_kg |
|---|---:|---:|
| 허니버터 감자칩 통 | 21 | 0.021 |
| 우유팩 | 11 | 0.011 |
| 남색 컵 | 60 | 0.060 |
| 민트색 컵 | 74 | 0.074 |
| 컵걸이 | 247 | 0.247 |
| 쓰레기통 | 192 | 0.192 |
| 수세미 | 15 | 0.015 |
| 초코비 | 16 | 0.016 |
| 연청색 락앤락 | 44 | 0.044 |
| 스테인리스 락앤락 | 108 | 0.108 |
| 초코비 | 20 | 0.020 |

초코비가 두 항목(16 g / 20 g)으로 기록돼 있음 — 서로 다른 물체면 이름을 구분해야 함.

---

## ⚠️ Required Downloads (NOT in git)

라이선스 / 용량 문제로 다음 파일들은 git에 들어있지 않음. 코드를 돌리려면 직접 다운받아 정해진 위치에 둬야 함.

### MANO hand model (라이선스 등록 필요)
[MANO 공식 사이트](https://mano.is.tue.mpg.de) 가입 → `mano_v1_2.zip` 다운 → 압축 풀어서:

```
third_party/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl
third_party/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl
```

> `MANO_LEFT.pkl`은 일반적으로 `MANO_RIGHT.pkl`을 복사하고 사용 (smplx가 left hand의 shapedirs bug를 자동 fix). HaWoR README의 안내를 따르는 것이 안전.

### HaWoR / detector / SLAM 가중치 (≈ 3.5 GB)
[HaWoR README](third_party/HaWoR/README.md)의 "Pretrained Weights" 절 참고.

```
third_party/HaWoR/weights/hawor/checkpoints/hawor.ckpt
third_party/HaWoR/weights/hawor/checkpoints/infiller.pt
third_party/HaWoR/weights/external/droid.pth
third_party/HaWoR/weights/external/detector.pt
third_party/HaWoR/_DATA/data/mano_mean_params.npz
```

---

## Pipeline (`run_pipeline.sh`)

RGB 프레임 하나의 에피소드를 처음부터 끝까지 처리하는 end-to-end 스크립트.

```
rgb/  →  [1] hand estimation   →  rgb_hawor/retarget_input.npz
                                               ↓
      →  [2] contact estimation →  contact/*.npz
                                               ↓
      →  [3] retargeting (stage1+2)  →  rgb_hawor/qpos_xhand_contact_{right,left}.pkl
                                               ↓
                            rgb_hawor/overlay_stage2_contact.mp4
```

### 기본 실행

```bash
# 스크립트 내 기본 DATA_DIR 사용 (contact + overlay 포함, SLAM 생략)
bash run_pipeline.sh

# 에피소드 경로 직접 지정
bash run_pipeline.sh --data_dir /path/to/episode
```

### 자주 쓰는 조합

```bash
# contact/overlay 없이 stage1 retargeting만
bash run_pipeline.sh --data_dir /path/to/episode \
    --skip_contact --no_contact --no_overlay

# 앞 단계 결과가 이미 있을 때 retargeting + overlay만 재실행
bash run_pipeline.sh --data_dir /path/to/episode --skip_hand --skip_contact

# 프레임 파일명 패턴이 다른 경우
bash run_pipeline.sh --data_dir /path/to/episode --img_glob "rgb_frame*.png"
```

### 전체 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--data_dir PATH` | 스크립트 내 하드코딩 | 에피소드 루트 디렉터리 |
| `--img_focal FLOAT` | `497.77` | 카메라 focal length (픽셀) |
| `--img_glob PATTERN` | `frame_*.jpg` | RGB 프레임 파일명 glob |
| `--with_slam` | off | HaWoR SLAM 활성화 (world frame 출력) |
| `--contact` / `--no_contact` | **on** | stage-2 contact-aware retargeting |
| `--overlay` / `--no_overlay` | **on** | retargeting 결과 RGB overlay mp4 생성 |
| `--skip_hand` | off | stage 1 건너뜀 |
| `--skip_contact` | off | stage 2 건너뜀 |
| `--skip_retarget` | off | stage 3 건너뜀 |

### 단계별 출력

| 단계 | conda 환경 | 출력 경로 |
|---|---|---|
| 1. Hand Estimation | `hawor` | `<episode>/rgb_hawor/retarget_input.npz` |
| 2. Contact Estimation | `haco` | `<episode>/contact/<frame>.npz` |
| 3. Retargeting (stage1) | `vjepa2-312` | `<episode>/rgb_hawor/qpos_xhand_{right,left}.pkl` |
| 3. Retargeting (stage2) | `vjepa2-312` | `<episode>/rgb_hawor/qpos_xhand_contact_{right,left}.pkl` |
| + Overlay | `vjepa2-312` | `<episode>/rgb_hawor/overlay_stage{1,2_contact}.mp4` |

---

## Robot replacement → policy training workflow

The current training path uses a **robot-replacement observation**, not the
raw human video.  The human hand and arm are segmented and inpainted, the
RBY1 + XHand render is composited without clipping it to the human mask, and
V-JEPA / policy preprocessing consumes that completed video.

```
RGB
 ├─ HaWoR / HaCo ──→ MANO + contact ──→ XHand retargeting
 ├─ SAM2 hand+arm mask ──→ E2FGVI background inpainting
 └─ inpainted background + RBY1/XHand render
       ├─→ video_overlay_rby1_xhand.mp4 ──→ V-JEPA `vjepa_robot`
       └─→ LeRobot observation video

smoothed wrist + finger trajectory
 └─ camera-to-RBY1 workspace calibration
       └─ final_pose.pkl ──→ joint-limited IK validation
             └─ LeRobot 38-D state/action ──→ Diffusion Policy
```

### 1. Refresh SAM2 smoothing and visual comparisons

The refresh script preserves the unsmoothed baseline, applies the same
post-processing used by the production SAM2 stage, and rebuilds only the
dependent inpainting/composite outputs.  HaWoR, HaCo, retargeting, and the
robot-only render are reused.

```bash
# One episode
bash scripts/run_sam_smoothing_refresh.sh IMG_5019

# More episodes
bash scripts/run_sam_smoothing_refresh.sh IMG_5019 IMG_5020
```

Important outputs under
`<episode>/inpainting_processed/<episode>/0/`:

| Output | Meaning |
|---|---|
| `segmentation_processor/masks_arm.npy` | smoothed human hand+arm mask |
| `segmentation_processor/masks_arm_no_smooth.npy` | exact pre-smoothing baseline |
| `inpaint_processor/video_human_inpaint.mkv` | smoothed-mask background inpaint |
| `video_overlay_rby1_xhand.mp4` | final inpainted background + robot replacement |
| `sam2_smoothing_comparison.mp4` | mask/inpaint/final ON-vs-OFF comparison |
| `pipeline_required_components.mp4` | compact HaWoR/HaCo/segmentation/inpaint/render view |

`src/inpainting/inpaint_hands.py` accepts `--mask` and `--output` overrides so
an A/B run does not overwrite the production paths.

### 2. Export and validate robot trajectories

HaWoR wrist poses are in the external recording-camera frame.  They must not
be interpreted as RBY1 head-camera poses.  The exporter preserves relative
recorded motion while anchoring it in a reachable RBY1-base workspace.

```bash
# Discover every ready IMG_* episode
bash scripts/export_policy_trajectories.sh

# Validate every exported frame with joint-limited RBY1 arm IK
bash scripts/validate_policy_trajectories.sh
```

The calibration profile is fitted once at
`<data-root>/workspace_calibration_rby1.json`.  Existing hand references are
frozen when more episodes are added; only a previously unseen hand is
appended.  `final_pose.pkl` records the calibration hash and coordinate frame
for traceability.  Dataset conversion rejects legacy camera-frame actions by
default.

The policy state/action layout is 38-D:

| Slice | Value |
|---|---|
| `0:12` | right XHand finger joints |
| `12:15` | right wrist xyz in RBY1 base frame |
| `15:19` | right wrist quaternion `(x, y, z, w)` |
| `19:31` | left XHand finger joints |
| `31:34` | left wrist xyz in RBY1 base frame |
| `34:38` | left wrist quaternion `(x, y, z, w)` |

An absent hand is zero-filled and does not invalidate a single-hand episode.
After invalid frames are removed, next-step actions are rebuilt so an action
never targets a discarded state.

### 3. Prepare all available episodes and train

The preparation script discovers all `IMG_*` directories each time it runs.
A newly added episode is included only after its retargeting, robot composite,
calibration, and IK validation inputs are complete.

```bash
# Export → validate → convert to LeRobot v3
bash scripts/prepare_policy_dataset.sh

# Train LeRobot Diffusion Policy
bash scripts/train_diffusion_policy.sh
```

Useful overrides:

```bash
DATA=/path/to/recordings \
OUT=/path/to/lerobot_dataset \
TASK="manipulate object" \
  bash scripts/prepare_policy_dataset.sh

DATASET=/path/to/lerobot_dataset \
STEPS=100000 BATCH_SIZE=2 NUM_WORKERS=4 \
  bash scripts/train_diffusion_policy.sh
```

The converter prefers `video_overlay_rby1_xhand.mp4`, preserves aspect ratio
with letterboxing, records each source episode and visual source in metadata,
and fails fast on incomplete episodes unless `--skip_failed` is explicitly
requested.

### Changing from a cube to another object

The hand-estimation, contact, retargeting, hand/arm segmentation, background
inpainting, robot rendering, V-JEPA backbone, and 38-D robot schema are
object-agnostic.  The following parts must be reviewed for a new object:

1. Override `DATA`, `OUT`, `DATASET`, and `TASK`; the checked-in convenience
   scripts still use the current cube dataset as their local default.
2. Replace the cube-specific modal/amodal segmentation stages
   (`segment_cube.py`, `amodal_cube.py`, and related layer scripts) with an
   object mask.  Pass that mask through `inpaint_hands.py --protect_mask` when
   the manipulated object must be preserved under the human hand.
3. Replace the cube geometry, physical properties, and initial pose in the
   MuJoCo scene.  `RBY1XHandEnv` currently provides an imitation-learning
   rollout (`reward = 0`); object-specific success and reward logic must be
   added for task evaluation.
4. Regenerate `features.pt` from the new robot-composite videos and retrain or
   fine-tune the classifier.  If the skill taxonomy changes, update
   `src/utils/labels.py`, create matching `gt_labels.json` files, and retrain
   the classifier head.
5. Refit workspace calibration if the physical camera placement or robot
   workspace changes.  A different object alone does not require refitting
   when the camera and manipulation workspace remain fixed.

Generated datasets, videos, calibration results, model weights, and training
outputs are intentionally excluded from git.  A fresh clone must download the
required weights and generate its dataset-local calibration profile.

---

### 모듈별 사용법

각 모듈 폴더의 README 참고:
- [`src/data_preprocess/README.md`](src/data_preprocess/README.md) — 원본 데이터 정리 / 프레임 추출 (skill classifier용 features.pt)
- [`src/hand_estimation/README.md`](src/hand_estimation/README.md) — RGB → MANO
- [`src/contact_estimation/README.md`](src/contact_estimation/README.md) — RGB + MANO → contact mask
- [`src/retargeting/README.md`](src/retargeting/README.md) — MANO → xhand qpos + 시각화
- [`src/inpainting/README.md`](src/inpainting/README.md) — 사람 손/팔 inpaint + xhand 오버레이 합성
- [`src/skill_classifier/README.md`](src/skill_classifier/README.md) — V-JEPA 2 기반 skill 분류
- [`src/pkl_to_lerobot/README.md`](src/pkl_to_lerobot/README.md) — retarget pkl → LeRobot v3 dataset
- [`src/policy/README.md`](src/policy/README.md) — Diffusion Policy / GR00T 학습 + offline action-MSE eval
- [`src/sim/mujoco_sim/README.md`](src/sim/mujoco_sim/README.md) — MuJoCo RBY1+xhand 시뮬레이션 (retarget 검증 + policy eval)
- [`src/sim/isaac_lab/README.md`](src/sim/isaac_lab/README.md) — Isaac Lab 변형 (USD 변환 + replay)
- [`src/simulation_tool/README.md`](src/simulation_tool/README.md) — cube 6-DoF pose 최적화 도구
