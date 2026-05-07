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
├── result.json                  # 프레임별 hand detection 결과 (kpts_2d, is_right)
└── rgb_hawor/
    └── retarget_input.npz       # HAWoR hand mesh (verts, valid, start_idx)
```

## Usage

```bash
python extract_hand_contact.py \
    --input_dir /path/to/episode_dir \
    [--backbone hamer] \
    [--checkpoint /path/to/checkpoint.ckpt]
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--input_dir` | (필수) | episode 디렉토리 경로 |
| `--backbone` | `hamer` | HACO backbone 종류 |
| `--checkpoint` | `HACO_RELEASE/release_checkpoint/haco_neurips_hamer_checkpoint.ckpt` | 모델 체크포인트 경로 |

## Output

```
<input_dir>/contact/
├── <frame_key>.npz     # per-frame contact 데이터
│   ├── left_contact_mask        # (778,) bool
│   ├── left_contact_indices     # contact vertex 인덱스
│   ├── left_contact_verts_3d    # contact vertex 3D 좌표
│   └── right_* (동일 구조)
└── viz/
    └── <frame_key>.png  # [RGB | Left hand | Right hand] 시각화
```
