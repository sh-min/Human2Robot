---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Pretendard', 'Apple SD Gothic Neo', sans-serif;
    font-size: 24px;
  }
  h1 { font-size: 36px; color: #1a1a2e; }
  h2 { font-size: 30px; color: #16213e; }
  h3 { font-size: 24px; color: #0f3460; }
  table { font-size: 20px; }
  code { font-size: 18px; }
  .columns { display: flex; gap: 40px; }
  .col { flex: 1; }
  .tag-done { background: #d4edda; color: #155724; padding: 2px 8px; border-radius: 4px; font-size: 18px; }
  .tag-todo { background: #fff3cd; color: #856404; padding: 2px 8px; border-radius: 4px; font-size: 18px; }
  .tag-blocked { background: #f8d7da; color: #721c24; padding: 2px 8px; border-radius: 4px; font-size: 18px; }
---

# Skill2Policy: LeRobot Pipeline Integration

**사람 손 시연 영상 → 로봇 정책 학습까지의 데이터 파이프라인**

<br>

Branch: `feature/lerobot-pipeline`
Date: 2026.05.14

---

# 프로젝트 목표

> 사람이 큐브를 조작하는 RGB 영상에서 **로봇 손(XHand)의 관절 궤적을 추출**하고,
> 이를 **Diffusion Policy / GR00T N1**으로 학습시켜
> **MuJoCo 시뮬레이션에서 정책을 평가**할 수 있는 구조를 만든다.

<br>

```
사람 손 영상 ──► 손 추정 ──► 리타겟팅 ──► 데이터셋 변환 ──► 정책 학습 ──► 시뮬 평가
  (기존)         (기존)       (기존)        (신규)           (신규)        (신규)
```

---

# 전체 파이프라인 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 1: Perception Pipeline (기존)                                    │
│                                                                         │
│  RGB frames ─► HaWoR ─► HACO ─► DexPilot ─► qpos_xhand.pkl            │
│               (MANO)   (접촉)   (리타겟)    (12-DOF + wrist 7D)        │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│  Stage 2: Dataset Conversion (신규 구현)                                │
│                                                                         │
│  pkl + RGB ──► convert_batch.py ──► LeRobot Dataset (Parquet + MP4)     │
│                                     + modality.json (GR00T용)           │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│  Stage 3: Training & Evaluation (신규 구현)                             │
│                                                                         │
│  LeRobot Dataset ──┬──► Diffusion Policy (lerobot-train)                │
│                    └──► GR00T N1 (gr00t_finetune) ──► MuJoCo Eval      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# 구현 완료: 데이터셋 변환 모듈

### `src/dataset_converter/`

<div class="columns">
<div class="col">

**schema.py** — 데이터 표현 정의
- 38-D state/action 벡터 레이아웃
- pkl → flat vector 변환 함수
- GR00T modality.json 생성

**convert_episode.py** — 단일 에피소드
- pkl 자동 탐색 (contact > smooth > 기본)
- valid 프레임 필터링
- ffmpeg MP4 인코딩 + Parquet 저장

**convert_batch.py** — 일괄 변환
- 에피소드 디렉토리 자동 탐색
- 메타데이터 파일 일괄 생성
- 통계(stats.json) 자동 계산

</div>
<div class="col">

**38-D 벡터 레이아웃**

| Index | Field | Dim |
|-------|-------|-----|
| 0:12 | right_hand_joint | 12 |
| 12:15 | right_wrist_pos | 3 |
| 15:19 | right_wrist_quat | 4 |
| 19:31 | left_hand_joint | 12 |
| 31:34 | left_wrist_pos | 3 |
| 34:38 | left_wrist_quat | 4 |

**Action mode**: absolute / delta 선택 가능

</div>
</div>

---

# 구현 완료: 정책 학습 설정

### `src/policy_config/`

<div class="columns">
<div class="col">

### Diffusion Policy
`diffusion_xhand.yaml`

- LeRobot `lerobot-train` CLI 호환
- 2-step observation, 16-step action horizon
- 224×224 이미지 + 38-D state 입력
- 38-D action 출력

```bash
lerobot-train \
  --config_path policy_config/diffusion_xhand.yaml
```

</div>
<div class="col">

### GR00T N1
`groot_xhand_config.py`

- `NEW_EMBODIMENT` 태그 등록
- 6개 modality key 매핑
- 손가락/위치: relative action
- 쿼터니언: absolute action

```bash
python gr00t_finetune.py \
  --modality-config-path groot_xhand_config.py \
  --embodiment-tag NEW_EMBODIMENT
```

</div>
</div>

---

# 구현 완료: MuJoCo 시뮬레이션 환경

### `src/mujoco_sim/` (mujoco_sim 브랜치에서 통합)

<div class="columns">
<div class="col">

**환경 구성**
- Rainbow Robotics **RBY1** 휴머노이드
- 양손 **XHand** (12-DOF × 2)
- 7-DOF 팔 × 2
- **총 38-DOF** action space

**주요 모듈**
- `env.py` — Gymnasium 래퍼 (LeRobot 규격)
- `ik_arm.py` — Pinocchio 기반 손목 IK
- `compose_rby1_xhand.py` — 씬 합성
- `web.py` — Gradio 인터랙티브 포저

</div>
<div class="col">

**평가 스크립트** (`eval_mujoco.py`)
- LeRobot / GR00T 두 백엔드 지원
- 에피소드별 메트릭 수집
- rollout 비디오 저장

```bash
python -m policy_config.eval_mujoco \
  --backend lerobot \
  --checkpoint /path/to/ckpt \
  --n_episodes 10 \
  --save_video
```

</div>
</div>

---

# 출력 데이터셋 구조 (LeRobot v2)

```
lerobot_dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet      ← state(38-D) + action(38-D) + timestamp
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       └── observation.images.head_cam/
│           ├── episode_000000.mp4      ← RGB 프레임 → MP4 인코딩
│           └── ...
└── meta/
    ├── info.json                       ← 데이터셋 기본 정보
    ├── episodes.jsonl                  ← 에피소드 목록
    ├── tasks.jsonl                     ← 태스크 설명
    ├── stats.json                      ← 정규화 통계
    └── modality.json                   ← GR00T N1 전용 메타데이터
```

---

# 진행 상황 요약

| 항목 | 상태 | 세부 |
|------|------|------|
| Perception Pipeline (HaWoR, HACO, DexPilot) | <span class="tag-done">완료 (기존)</span> | `run_pipeline.sh` |
| 데이터셋 변환 모듈 | <span class="tag-done">구현 완료</span> | `src/dataset_converter/` |
| Diffusion Policy 학습 설정 | <span class="tag-done">구현 완료</span> | `diffusion_xhand.yaml` |
| GR00T N1 modality 설정 | <span class="tag-done">구현 완료</span> | `groot_xhand_config.py` |
| MuJoCo 평가 환경 | <span class="tag-done">통합 완료</span> | `src/mujoco_sim/` |
| 정책 평가 스크립트 | <span class="tag-done">구현 완료</span> | `eval_mujoco.py` |
| 실제 데이터 변환 테스트 | <span class="tag-todo">다음 단계</span> | 서버에서 실행 필요 |
| Diffusion Policy 학습 실행 | <span class="tag-todo">다음 단계</span> | GPU 서버 필요 |
| GR00T N1 학습 실행 | <span class="tag-todo">다음 단계</span> | NVIDIA GPU 필요 |
| Inpainted RGB 지원 | <span class="tag-todo">추후</span> | 사람손 제거 + 로봇손 합성 |
| MuJoCo reward 함수 | <span class="tag-todo">추후</span> | 큐브 조작 성공 판정 |

---

# 다음 단계 (TODO)

### 1단계: 데이터 검증 (이번 주)
- [ ] 실제 리타겟팅 pkl 데이터로 `convert_batch` 실행
- [ ] 변환된 데이터셋이 LeRobot에서 정상 로드되는지 확인
- [ ] 통계 값 (stats.json) 분포 확인 및 정규화 검증

### 2단계: 학습 실험 (다음 주)
- [ ] Diffusion Policy 학습 실행 (lerobot-train)
- [ ] GR00T N1 finetuning 실행
- [ ] 학습 곡선(loss) 및 wandb 로그 확인

### 3단계: 평가 및 개선
- [ ] MuJoCo 환경에서 rollout 평가
- [ ] Inpainted RGB (로봇 손 시점) 데이터 추가
- [ ] Reward 함수 설계 (큐브 회전 각도 기반)
- [ ] 실제 로봇 배포 준비

---

# 기술 결정사항

<div class="columns">
<div class="col">

### 왜 LeRobot v2인가?
- GR00T N1이 **v2만 공식 지원**
- Diffusion Policy는 v2/v3 모두 가능
- v2로 통일하면 **양쪽 모두 호환**

### 왜 38-D 벡터인가?
- MuJoCo env의 action space와 일치 (38-DOF)
- 양손 각각 19-D (12 finger + 3 wrist pos + 4 quat)
- 한손만 있는 에피소드는 나머지 zero-fill

</div>
<div class="col">

### Action 표현: absolute vs delta
- **Absolute** (기본): 학습 초기 안정적
- **Delta**: GR00T N1 권장, 일반화 우수
- 변환기에서 `--action_mode` 옵션으로 선택

### 이미지 소스
- **현재**: 원본 RGB (사람 손 보임)
- **추후**: inpainting 모듈 연결
  (사람 손 제거 + 로봇 손 오버레이)

</div>
</div>

---

# 코드 구조 요약

```
skill2policy/
├── src/
│   ├── hand_estimation/        ← Stage 1: RGB → MANO (기존)
│   ├── contact_estimation/     ← Stage 1: 접촉 추정 (기존)
│   ├── retargeting/            ← Stage 1: MANO → xHand qpos (기존)
│   ├── inpainting/             ← 사람 손 제거 + 로봇 손 합성 (기존)
│   ├── skill_classifier/       ← 스킬 분류 (기존)
│   │
│   ├── dataset_converter/      ← ★ 신규: pkl → LeRobot Dataset
│   │   ├── schema.py
│   │   ├── convert_episode.py
│   │   └── convert_batch.py
│   │
│   ├── policy_config/          ← ★ 신규: 학습/평가 설정
│   │   ├── diffusion_xhand.yaml
│   │   ├── groot_xhand_config.py
│   │   └── eval_mujoco.py
│   │
│   └── mujoco_sim/             ← ★ 통합: MuJoCo 시뮬레이션
│       ├── env.py
│       ├── ik_arm.py
│       └── ...
│
└── third_party/
    └── lerobot/                ← 서브모듈 (학습 인프라)
```

---

# 감사합니다

<br>

**Branch**: `feature/lerobot-pipeline`
**Repository**: `gomduribo/skill2policy`

<br>

질문 / 피드백 환영합니다.
