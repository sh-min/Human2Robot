# Hand Estimation (HaWoR wrapper)

RGB 영상 시퀀스에서 양손 MANO pose / joint position 추정. 백엔드는 vendored
[HaWoR](https://github.com/ThunderVVV/HaWoR) (`third_party/HaWoR`).

## 환경 (conda env: `hawor`)

```bash
cd <repo_root>/third_party/HaWoR
conda env create -f environment.yml -n hawor       # 또는 README 기준 env
conda activate hawor
```

> 이 디렉터리 (`src/hand_estimation`)에 별도의 conda env 파일은 없음 —
> HaWoR이 자체 의존성을 관리하니 그쪽 env를 그대로 사용.

## 모델 가중치 / MANO 다운로드 (필수, git에 들어있지 않음)

이 파일들은 라이선스 / 용량 문제로 **이 repo에 올릴 수 없음**. 본인이 직접 다운받아 아래 위치에 두어야 함.

### 1. MANO 모델 (사용자 등록 + 라이선스 수락 필요)
- 가입: https://mano.is.tue.mpg.de
- `mano_v1_2.zip` 다운로드 후 압축 풀기
- 다음 위치에 배치:
  ```
  third_party/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl              ← right
  third_party/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl     ← left (mirror)
  ```

### 2. HaWoR 체크포인트 (≈3.5 GB)
[HaWoR README](../../third_party/HaWoR/README.md)의 "Pretrained Weights" 절 참고:
```
third_party/HaWoR/weights/hawor/checkpoints/hawor.ckpt
third_party/HaWoR/weights/hawor/checkpoints/infiller.pt
third_party/HaWoR/weights/external/droid.pth
third_party/HaWoR/weights/external/detector.pt
third_party/HaWoR/_DATA/data/mano_mean_params.npz   (보통 같이 옴)
```

## 실행

```bash
conda activate hawor
cd <repo_root>/src/hand_estimation

python extract_for_retarget.py \
    --rgb_dir   /path/to/rgb_frames \
    --img_glob  "rgb_frame*.png" \
    --img_focal 497.77 \
    --skip_slam
```

### 인자

| 인자 | 설명 |
|---|---|
| `--rgb_dir` | RGB 시퀀스가 들어있는 디렉터리 |
| `--img_glob` | 어떤 파일을 RGB로 쓸지 (default `rgb_frame*.png`) |
| `--img_focal` | 카메라 focal (pixel). 안 주면 default 600. 정확하면 추정 정확도 ↑ |
| `--skip_slam` | SLAM/infiller 건너뜀. 결과는 cam frame, 누락 프레임은 valid=False |
| `--workdir` | 작업/출력 폴더 override |
| `--video_path` | 캐시된 작업 폴더에 npz만 만들고 싶을 때. seq_folder 결정용 (실제 mp4 없어도 됨) |

### 출력

`<rgb_dir parent>/<rgb_dir basename>_hawor/retarget_input.npz`

| 키 | 모양 | 설명 |
|---|---|---|
| `joints_left`, `joints_right` | `(T, 21, 3)` | mediapipe-21 layout 손 joint 위치 (cam 또는 world frame) |
| `mano_trans` | `(2, T, 3)` | MANO `transl` 파라미터 (raw, canonical offset 미반영) |
| `mano_global_orient` | `(2, T, 3)` | wrist axis-angle |
| `mano_hand_pose` | `(2, T, 15, 3)` | 15-joint articulation axis-angle |
| `mano_betas` | `(2, T, 10)` | shape params |
| `valid` | `(2, T)` bool | 프레임별 detect 성공 여부 |
| `frame_is_cam_space` | bool scalar | True면 cam frame, False면 world frame (SLAM 적용) |
| `R_c2w`, `t_c2w` | `(T, 3, 3)`, `(T, 3)` | (skip_slam=False일 때만) per-frame SLAM 카메라 자세 |

> **주의**: 실제 손목 cam-frame 위치는 `mano_trans`가 아니라 `joints_*[t, 0]`. `mano_trans`는
> raw transl 파라미터로, MANO 모델 mean shape의 canonical wrist offset이 빠져있음.

## 작동 흐름

1. PNG 시퀀스 → `extracted_images/*.jpg` (lossless quality 95)
2. detect + track (Detectron2 + tracker) — hand bbox per frame
3. HaWoR motion estimation — chunk별 cam-frame MANO pose
4. (옵션) SLAM → world frame transform
5. (옵션) infiller — 누락 프레임 보간
6. MANO forward → 21 joint position
7. npz 저장

캐시된 중간 결과(`tracks_*/`, `cam_space/`, `SLAM/`, `world_space_res.pth`)가 있으면
재사용해서 추론 다시 안 돌림.
