# Contact Estimation

RGB 프레임에서 손의 contact vertex를 추출합니다. HACO 모델을 사용하며, HAWoR의 hand mesh 결과를 입력으로 받습니다.

## Setup

```bash
# conda 환경 설정은 HACO_RELEASE 참조
# https://github.com/dqj5182/HACO_RELEASE
```

## Input

episode 디렉토리 구조가 아래와 같아야 합니다:

```
<input_dir>/
├── rgb/                         # 원본 RGB 프레임 이미지
└── rgb_hawor/
    └── retarget_input.npz       # HAWoR 결과 (verts, joints, valid, start_idx)
```

## Usage

```bash
python extract_hand_contact.py \
    --input_dir /path/to/episode_dir \
    [--img_glob "*.jpg"] \
    [--backbone hamer] \
    [--checkpoint /path/to/checkpoint.ckpt]
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--input_dir` | (필수) | episode 디렉토리 경로 |
| `--img_glob` | `*.jpg` | 프레임당 하나의 원본 확장자만 선택 |
| `--img_focal` | HaWoR NPZ 값 | bbox 투영용 카메라 focal length (pixels) |
| `--backbone` | `hamer` | HACO backbone 종류 |
| `--checkpoint` | `HACO_RELEASE/release_checkpoint/haco_neurips_hamer_checkpoint.ckpt` | 모델 체크포인트 경로 |
| `--no_viz` | off | 접촉 시각화 PNG 생성을 생략 |

동일한 프레임의 JPG와 PNG가 함께 있는 데이터셋에서는 두 확장자를
동시에 읽으면 HaWoR 프레임 정렬이 깨진다. 따라서 `--img_glob`은 반드시
한 확장자만 선택하며, 같은 stem이 중복되면 추출기는 중단한다.

## Output

```
<input_dir>/contact/
├── <frame_key>.npz     # per-frame contact 데이터
│   ├── left_contact_mask        # (778,) bool
│   ├── left_contact_probability # (778,) float16, threshold 이전 sigmoid
│   ├── left_contact_indices     # contact vertex 인덱스
│   ├── left_contact_verts_3d    # contact vertex 3D 좌표
│   ├── left_valid               # 해당 frame의 HaWoR hand 유효 여부
│   └── right_* (동일 구조)
└── viz/
    └── <frame_key>.png  # [RGB | Left hand | Right hand] 시각화
```

각 NPZ에는 정렬 검증을 위한 `source_filename`, `hawor_frame_index`,
`img_focal`, `contact_threshold`도 함께 저장된다. 단순
`contact_mask.any()`는 접촉을 과검출할 수 있으므로, 합성 단계에서는
`contact_probability`를 손가락별로 집계하고 시간 히스테리시스를 적용한다.
