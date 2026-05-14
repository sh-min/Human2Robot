# dataset_converter

리타겟팅 파이프라인 출력물(pkl + RGB)을 LeRobot 데이터셋 형식으로 변환하는 모듈.

## 데이터 흐름

```
run_pipeline.sh 출력                    이 모듈                         학습
─────────────────                    ────────                         ────
episode/
  rgb/frame_*.jpg ─────┐
  rgb_hawor/           │         ┌───────────────┐            ┌───────────────┐
    qpos_xhand_*.pkl ──┼────────►│ convert_batch │────────────► LeRobot v2    │
                       │         └───────────────┘            │ (Parquet+MP4) │
                       │                                      └───────┬───────┘
                       │                                              │
                       │                                    ┌─────────┴─────────┐
                       │                                    │                   │
                       │                              Diffusion Policy     GR00T N1
                       │                              (lerobot-train)    (gr00t_finetune)
```

## 사용법

### 단일 에피소드 변환

```bash
PYTHONPATH=$PWD/src python -m dataset_converter.convert_episode \
    --episode_dir /path/to/episode \
    --out_dir /path/to/output_dataset \
    --episode_index 0 \
    --fps 30 \
    --task "manipulate rubik's cube"
```

### 여러 에피소드 일괄 변환

```bash
PYTHONPATH=$PWD/src python -m dataset_converter.convert_batch \
    --data_root /path/to/all_episodes \
    --out_dir /path/to/lerobot_dataset \
    --fps 30 \
    --action_mode absolute \
    --task "manipulate rubik's cube"
```

`data_root` 아래에서 `rgb/`과 `rgb_hawor/`를 모두 포함하는 디렉토리를 자동 탐색합니다.

### 출력 디렉토리 구조 (LeRobot v2)

```
lerobot_dataset/
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
    ├── info.json
    ├── episodes.jsonl
    ├── tasks.jsonl
    ├── stats.json
    └── modality.json        ← GR00T N1 전용
```

## State/Action 벡터 레이아웃 (38-D)

| 인덱스 | 필드 | 차원 |
|--------|------|------|
| 0:12 | right_hand_joint | 12 (손가락 관절각) |
| 12:15 | right_wrist_pos | 3 (xyz 위치) |
| 15:19 | right_wrist_quat | 4 (xyzw 쿼터니언) |
| 19:31 | left_hand_joint | 12 |
| 31:34 | left_wrist_pos | 3 |
| 34:38 | left_wrist_quat | 4 |

## Action Mode

- `absolute` (기본값): action[t] = state[t+1] (다음 프레임의 절대 위치)
- `delta`: action[t] = state[t+1] - state[t] (프레임 간 변화량)

Diffusion Policy는 absolute/delta 모두 사용 가능.
GR00T N1은 자체 action config에서 relative/absolute를 지정 (groot_xhand_config.py 참고).

## pkl 파일 자동 탐색 우선순위

각 손(right/left)에 대해 아래 순서로 pkl을 탐색합니다:

1. `qpos_xhand_contact_{hand}_smooth.pkl` (접촉 + 스무딩)
2. `qpos_xhand_contact_{hand}.pkl` (접촉)
3. `qpos_xhand_{hand}_smooth.pkl` (스무딩)
4. `qpos_xhand_{hand}.pkl` (기본)
