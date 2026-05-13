# Inpainting + xhand overlay

RGB 영상에서 사람 손/팔을 SAM2 + E2FGVI로 inpaint 해서 지우고, 그 자리에
retargeting 결과(xhand qpos)로 렌더링한 로봇 손을 합성. 결과는
"인간 동작을 그대로 따라하는 xhand 데모 영상".

```
raw rgb frames + HaWoR(retarget_input.npz) + xhand qpos
        │
        ▼
[prepare_demo]                 frames → video_L.mp4, demo layout
        │
        ▼
[inject_hawor_data]            HaWoR 2D/3D kpts → bbox_data.npz, hand_data_{l,r}.npz,
                                                  video_rgb_imgs.mkv (ffv1)
        │
        ▼
[segment_arms]                 SAM2 — 양손+팔 mask propagation
        │
        ▼
[inpaint_hands]                E2FGVI — mask 영역 video inpainting
        │
        ▼
[render_xhand_overlay]         pyrender + xhand URDF — 로봇 손 합성
        │
        ▼
   video_overlay_xhand.mkv
```

## 환경 (전용 conda env 권장: `inpaint`)

```bash
conda create -n inpaint python=3.10 -y && conda activate inpaint
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -c nvidia/label/cuda-12.1.0 -y

# PyTorch (먼저, 다른 deps가 새 버전을 끌어오지 못하게)
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0 torchvision==0.16.0

# 핵심 deps
pip install numpy==1.26.4 opencv-python mediapy joblib tqdm scipy
pip install "PyOpenGL==3.1.4" "pyrender>=0.1.45" trimesh
# (`pyrender>=0.1.45` — 이전 버전은 `IntrinsicsCamera` 가 없어서 render_xhand_overlay
#  가 import 시점에 AttributeError로 죽는다. pip이 PyOpenGL 호환성 때문에 0.1.18을
#  고르는 경우가 있어서 명시적 pin.)
pip install gdown                  # E2FGVI weight 다운로드용
pip install "setuptools<81"        # mmcv가 import 시점에 pkg_resources를 씀

# vendored SAM2 (`--no-deps`: torch>=2.5.1 요구 무시)
pip install --no-deps -e third_party/sam2
# SAM2 runtime deps (`--no-deps` 때문에 직접 깔아야 함)
pip install "hydra-core>=1.3.2" "iopath>=0.1.10"

# pyrender가 networkx를 2.2로 silently 다운그레이드함 — 다시 올려야 함
# (Python 3.10에서 `from collections import Mapping`이 fail)
pip install "networkx>=3.0"

# E2FGVI는 install 없이 sys.path로 import — 별도 deps 추가
pip install timm einops "imageio[ffmpeg]"
# E2FGVI의 SPyNet flow component는 mmcv-full 필요 (cu121/torch2.1 prebuilt wheel)
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
# (mmcv-full이 numpy를 2.x로 올리는 경우 다시 1.26.4로 고정:
#  `pip install numpy==1.26.4`)
```

> 같은 머신에 `phantom` env가 있다면 그걸 그대로 써도 됨 — deps는 superset.

## Vendored submodules (`third_party/`)

```bash
git submodule update --init --recursive  # 이미 clone 되어 있어야 함
ls third_party/sam2   third_party/E2FGVI
```

| Submodule | Upstream | 용도 |
|---|---|---|
| `third_party/sam2`   | facebookresearch/sam2 | 양손+팔 segmentation |
| `third_party/E2FGVI` | MCG-NKU/E2FGVI         | flow-guided 영상 inpainting |

## 모델 가중치 (별도 다운로드, ~1.6 GB)

라이선스/용량 문제로 repo에 없음. 한 번만 받으면 됨.

### 1. SAM2 (~880 MB)
```bash
mkdir -p third_party/sam2/checkpoints
wget -P third_party/sam2/checkpoints \
    https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

### 2. E2FGVI (~440 MB)
```bash
mkdir -p third_party/E2FGVI/release_model
gdown 'https://drive.google.com/uc?id=10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3' \
    -O third_party/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth
```

xhand URDF / mesh / `R_mano_xhand_*.npy` 는 `src/retargeting/assets/` 의 것을 그대로 쓴다.

## 입력

| 입력 | 출처 | 비고 |
|---|---|---|
| `input` (dir 또는 video file) | 원본 RGB 시퀀스 | dir면 jpg/png 글롭 (`*.jpg` → `*.png` fallback), file이면 mp4/mkv 등 cv2/mediapy가 읽을 수 있는 모든 포맷 |
| `retarget_input.npz` | `src/hand_estimation/extract_for_retarget.py` | MANO joints + global_orient + valid + img_focal |
| `qpos_xhand_{left,right}.pkl` | `src/retargeting/retarget_from_npz.py` | 12-DOF DexPilot retargeted |

## 실행 (one-shot)

```bash
conda activate inpaint
cd src/inpainting

# `--input` 은 jpg/png 들어있는 디렉터리 OR raw video file 둘 다 받음.
# - dir → --fps 로 mp4 인코딩 (default 10)
# - file → 소스의 fps 그대로 사용 (--fps 로 덮어쓸 수 있음)

# (a) 프레임 디렉터리
python run.py \
    --input /data/RFM_proj/cam0_hawor/extracted_images \
    --hawor_npz  /data/RFM_proj/cam0_hawor/retarget_input.npz \
    --right_pkl  /data/RFM_proj/cam0_hawor/qpos_xhand_right.pkl \
    --left_pkl   /data/RFM_proj/cam0_hawor/qpos_xhand_left.pkl \
    --data_root      /data/RFM_proj/cam0_inpaint_raw \
    --processed_root /result/cam0_inpaint

# (b) raw 비디오 파일
python run.py \
    --input /data/cam0_raw.mp4 \
    --hawor_npz  /data/RFM_proj/cam0_hawor/retarget_input.npz \
    --right_pkl  /data/RFM_proj/cam0_hawor/qpos_xhand_right.pkl \
    --left_pkl   /data/RFM_proj/cam0_hawor/qpos_xhand_left.pkl \
    --data_root      /data/RFM_proj/cam0_inpaint_raw \
    --processed_root /result/cam0_inpaint
```

결과:
```
/result/cam0_inpaint/cam0/0/
├── video_L.mp4                                    # 입력 영상 (libx264)
├── video_rgb_imgs.mkv                             # ffv1 재인코딩 (E2FGVI 입력)
├── bbox_processor/bbox_data.npz                   # HaWoR 유래 bbox
├── hand_processor/hand_data_{left,right}.npz      # HaWoR 유래 keypoints
├── segmentation_processor/masks_arm.npy           # SAM2 mask
├── inpaint_processor/video_human_inpaint.mkv      # 손/팔 지운 배경
└── video_overlay_xhand.mkv                        # ★ 최종 출력 (로봇 손 합성)
```

각 stage는 출력이 이미 존재하면 skip. 부분 재실행하려면 해당 폴더만 지우면 됨.

## 단계별 호출 (디버깅용)

```bash
# 1) 데모 폴더 구조 만들기 (--input은 dir 또는 video file)
python prepare_demo.py \
    --input /data/RFM_proj/cam0_hawor/extracted_images \
    --data_root /data/RFM_proj/cam0_inpaint_raw \
    --processed_root /result/cam0_inpaint

# 2) HaWoR 데이터 주입 (Epic-Kitchens detector 우회)
python inject_hawor_data.py \
    --processed_demo /result/cam0_inpaint/cam0/0 \
    --hawor_npz /data/RFM_proj/cam0_hawor/retarget_input.npz

# 3) SAM2 arm segmentation
python segment_arms.py --processed_demo /result/cam0_inpaint/cam0/0

# 4) E2FGVI inpainting
python inpaint_hands.py --processed_demo /result/cam0_inpaint/cam0/0

# 5) xhand 오버레이
PYOPENGL_PLATFORM=egl python -u render_xhand_overlay.py \
    --processed_demo /result/cam0_inpaint/cam0/0 \
    --hawor_npz /data/RFM_proj/cam0_hawor/retarget_input.npz \
    --right_pkl /data/RFM_proj/cam0_hawor/qpos_xhand_right.pkl \
    --left_pkl  /data/RFM_proj/cam0_hawor/qpos_xhand_left.pkl
```

## 좌표계 메모

OpenCV cam (`+x right, +y down, +z forward`) → pyrender world (`+y up, -z forward`).
변환: `T_CV2GL = diag(1,-1,-1)`.

- **위치**:  `t_pr = T_CV2GL @ t_cam`
- **회전**:  `R_pr = T_CV2GL @ R_cam_xhand`
  `R_cam_xhand = R_mano @ R_MANO_XHAND[side]` (xhand local → MANO cam).
  `T_CV2GL`은 **출력 좌표계만** 바꾸는 것이라 양옆에 끼면 안 된다 — input frame
  (xhand local)은 좌표계 변환 대상이 아님. `@ T_CV2GL.T` 붙이면 손이 화면 아래로
  날아가서 0px 렌더된다.

## 알려진 함정

### 1. bbox 없으면 inpaint NO-OP
SAM2 prompt가 비면 `masks_arm.npy` 가 전부 False → E2FGVI 마스크 면적 0 → 원본 그대로 복사.
phantom의 Epic-Kitchens hand detector는 top-down view에서 손을 못 찾기 때문에, 이 모듈은
`inject_hawor_data.py` 에서 HaWoR 2D keypoints로 직접 bbox (`BBOX_PAD_RATIO = 0.4` forearm
포함) 를 만들어 사용한다.

### 2. SAM2 CUDA OOM (다른 사용자 GPU 공유 시)
`segment_arms.py` 가 `init_state(..., offload_video_to_cpu=True)` 로 호출한다. 안 그러면
SAM2가 1024×1024 float32 frame 전체 (~9.6 GB / 766 frames) 를 GPU에 올려서 fragmented
memory 에서 죽는다.

### 3. PyOpenGL 3.1.0 ≠ 3.1.4
pyrender 가 PyOpenGL을 3.1.0으로 silently downgrade 하는 경우가 있다. 3.1.0은
`OpenGL.EGL.EGLDeviceEXT` 가 없어서 EGL 백엔드가 실패한다.
```bash
pip install PyOpenGL==3.1.4
```

### 4. 프레임 수 mismatch
`retarget_input.npz` 의 T 가 `frames_dir` 보다 크면 inject/render 가 `min` 으로 잘라서 input
video 길이에 맞춘다. 반대로 npz가 짧으면 그만큼만 합성되고 나머지는 inpainted 배경 그대로.

## 파일 구조

```
src/inpainting/
├── README.md                    # 이 문서
├── _paths.py                    # SAM2_DIR, E2FGVI_DIR, xhand URDF, R_MANO_XHAND
├── prepare_demo.py              # frames → 데모 폴더 + 첫 copytree
├── inject_hawor_data.py         # HaWoR npz → bbox + hand_data + video_rgb_imgs
├── segment_arms.py              # SAM2 양손 forward/reverse propagation
├── inpaint_hands.py             # E2FGVI batched inpainting
├── render_xhand_overlay.py      # pyrender + xhand URDF
└── run.py                       # end-to-end orchestrator
```
