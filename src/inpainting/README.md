# Inpainting + xhand layered overlay

RGB 영상에서 사람 손/팔을 SAM2 + E2FGVI로 inpaint해서 지우고, retargeting
결과(xhand qpos)를 pyrender로 렌더링한 로봇 손을, **3-layer (behind-MCP robot
→ cube → front-MCP robot)** depth-aware composite로 raw bg 위에 합성한다.
결과는 "인간 동작을 그대로 따라하는 xhand 데모 영상" + 손에 잡고 있던 cube가
robot 손가락 사이로 보이게 occlusion 처리된 버전.

## 파이프라인 한눈에

```
raw rgb + HaWoR(retarget_input.npz) + xhand qpos
   │
   1.  prepare_demo                       video_L.mp4 (libx264 in demo layout)
   2.  inject_hawor_data                  HaWoR 2D/3D kpts → bbox + hand_data + video_rgb_imgs.mkv
   3.  segment_arms (SAM2)                segmentation_processor/masks_arm.npy   (M_hand)
   4.  inpaint_hands --mode legacy        inpaint_processor/video_human_inpaint.mkv   (hand-removed bg)
   5.  render_xhand_overlay_depth         overlay_processor/robot_{rgb,depth,mask}.npy
   6.  estimate_depth (Depth Anything V2) depth_processor/depth_raw.npy   (disparity)
   7.  align_depth                        depth_processor/depth_aligned.npy   (metric m)
   8.  crop_cube_layer                    cube_layer/cube_mask_raw.npy   (top-q + center CC)
   9.  regularize_and_cut_cube            cube_layer/cube_mask_clean.npy   (open/CC/close/hull/outlier/SDF)
  10.  composite_layered                  overlay_processor_layered/video_overlay.mp4    ★ 최종
```

## 환경 (전용 conda env 권장: `inpaint`)

```bash
conda create -n inpaint python=3.10 -y && conda activate inpaint
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -c nvidia/label/cuda-12.1.0 -y

# PyTorch (먼저, 다른 deps가 새 버전을 끌어오지 못하게)
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0 torchvision==0.16.0

# 핵심 deps
pip install numpy==1.26.4 opencv-python mediapy joblib tqdm scipy matplotlib
pip install "PyOpenGL==3.1.4" "pyrender>=0.1.45" trimesh
pip install gdown                  # E2FGVI weight 다운로드용
pip install "setuptools<81"        # mmcv가 import 시점에 pkg_resources를 씀

# vendored SAM2 (`--no-deps`: torch>=2.5.1 요구 무시)
pip install --no-deps -e third_party/sam2
pip install "hydra-core>=1.3.2" "iopath>=0.1.10"

# pyrender가 networkx를 silently 2.2로 떨어뜨리는 경우가 있어 다시 올림
pip install "networkx>=3.0"

# E2FGVI sys.path import 전용 deps
pip install timm einops "imageio[ffmpeg]"
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
# (mmcv-full이 numpy 2.x로 올리면 다시 `pip install numpy==1.26.4`)
```

## Vendored submodules (`third_party/`)

| Submodule | Upstream | 용도 |
|---|---|---|
| `third_party/sam2`              | facebookresearch/sam2     | 손/팔 segmentation (M_hand) |
| `third_party/E2FGVI`            | MCG-NKU/E2FGVI            | flow-guided 영상 inpainting |
| `third_party/Depth-Anything-V2` | DepthAnything/Depth-Anything-V2 | monocular depth |

## 모델 가중치 (별도 다운로드, ~2.9 GB)

```bash
# 1. SAM2 (~880 MB)
mkdir -p third_party/sam2/checkpoints
wget -P third_party/sam2/checkpoints \
    https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# 2. E2FGVI (~440 MB)
mkdir -p third_party/E2FGVI/release_model
gdown 'https://drive.google.com/uc?id=10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3' \
    -O third_party/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth

# 3. Depth Anything V2 vitl (~1.3 GB)
mkdir -p /ckpt/depth_anything
wget -P /ckpt/depth_anything \
    https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
```

xhand URDF / mesh / `R_mano_xhand_*.npy` 는 `src/retargeting/assets/` 의 것을 그대로 쓴다.

## 입력

| 입력 | 출처 | 비고 |
|---|---|---|
| `--input` (dir 또는 video file) | 원본 RGB 시퀀스 | dir면 `*.jpg` glob (`*.png` fallback), file이면 mp4/mkv/...등 cv2/mediapy가 읽을 수 있는 포맷 |
| `--hawor_npz` (retarget_input.npz) | `src/hand_estimation/extract_for_retarget.py` | MANO joints + global_orient + valid + img_focal |
| `--right_pkl` / `--left_pkl` (qpos_xhand_*) | `src/retargeting/retarget_from_npz.py` | 12-DOF DexPilot retargeted |

## 실행 (one-shot)

```bash
conda activate inpaint
cd src/inpainting

python run_layered.py \
    --input        /data/RFM_proj/cam0_hawor/extracted_images \
    --hawor_npz    /data/RFM_proj/cam0_hawor/retarget_input.npz \
    --right_pkl    /data/RFM_proj/cam0_hawor/qpos_xhand_right.pkl \
    --left_pkl     /data/RFM_proj/cam0_hawor/qpos_xhand_left.pkl \
    --data_root      /result/skill2policy/raw \
    --processed_root /result/skill2policy/processed
```

각 stage는 출력이 존재하면 `[skip]` — 부분 재실행하려면 해당 파일/폴더만 지우면 됨.

결과 디렉터리 (`/result/skill2policy/processed/cam0/0/`):

```
video_L.mp4                                        # 입력 영상 (libx264)
video_rgb_imgs.mkv                                 # ffv1 재인코딩 (E2FGVI 입력)
bbox_processor/bbox_data.npz                       # HaWoR 유래 bbox
hand_processor/hand_data_{left,right}.npz          # HaWoR 유래 2D/3D keypoints
segmentation_processor/masks_arm.npy               # M_hand (SAM2)
inpaint_processor/video_human_inpaint.mkv          # bg (E2FGVI on M_hand)
overlay_processor/robot_{rgb,depth,mask}.npy       # pyrender xhand RGBD
depth_processor/
  ├── depth_raw.npy                                # DA-V2 disparity on raw video
  ├── depth_aligned.npy                            # metric depth (m)
  └── depth_align_params.npz                       # per-frame (a, b)
cube_layer/
  ├── cube_mask_raw.npy                            # rough mask (top-q + center CC)
  ├── cube_cropped_raw.mp4                         # raw crop, 디버깅용
  ├── cube_mask_clean.npy                          # regularized + SDF-smoothed mask
  ├── cube_layer_final.mp4                         # bg cut by clean mask
  └── cube_edges_viz.mp4                           # bg + red outline
overlay_processor_layered/
  └── video_overlay.mp4                            # ★ 최종 결과
```

## 주요 knobs (run_layered.py)

```bash
# Depth Anything V2 encoder
--encoder {vits, vitb, vitl}      # 기본 vitl

# 8. crop_cube_layer
--cube_quantile FLOAT             # 0.25 (depth top quantile for cube isolation)

# 9. regularize_and_cut_cube
--sdf_sigma FLOAT                 # 8.0 (temporal smoothing sigma in frames)
--area_mad_k FLOAT                # 5.0 (area outlier MAD multiplier)
--centroid_max_jump FLOAT         # 120.0 (centroid jump in px)

# 10. composite_layered
--threshold_joint INT             # 5 = idx-MCP (front/behind split)
                                  # 0 = wrist (cube-heavy), 9 = mid-MCP (robot-heavy)
--zmcp_sigma_t FLOAT              # 8.0 (z-plane temporal smoothing in frames)
--edge_sigma FLOAT                # 1.5 px (alpha-blend edge feather)
```

## 단계별 호출 (디버깅용)

```bash
python prepare_demo.py            --input <...> --data_root <...> --processed_root <...>
python inject_hawor_data.py       --processed_demo <pd> --hawor_npz <...>
python segment_arms.py            --processed_demo <pd>
python inpaint_hands.py           --processed_demo <pd> --mode legacy
PYOPENGL_PLATFORM=egl python -u render_xhand_overlay_depth.py \
                                  --processed_demo <pd> --hawor_npz <...> \
                                  --right_pkl <...> --left_pkl <...>
python estimate_depth.py          --processed_demo <pd> --encoder vitl
python align_depth.py             --processed_demo <pd>
python crop_cube_layer.py         --processed_demo <pd> --quantile 0.25 --cc_pick center
python regularize_and_cut_cube.py --processed_demo <pd> --sdf_sigma 8.0
python composite_layered.py       --processed_demo <pd> --hawor_npz <...> \
                                  --cube_mask_npy cube_layer/cube_mask_clean.npy \
                                  --threshold_joint 5
```

## 디버그 시각화 도구 (visualize_*)

| Script | 출력 |
|---|---|
| `visualize_depth.py` | scene vs robot depth side-by-side (turbo colormap) |
| `visualize_depth_with_mask.py` | RGB \| depth, hand outline 빨강 |
| `visualize_depth_stats.py` | RGB/depth 왼쪽 + per-frame histogram 오른쪽 |
| `visualize_shallow_nonhand.py` | non-hand & shallow pixels만 raw RGB로 |
| `visualize_layers.py` | front-MCP/cube/behind-MCP을 색조 입혀 한 화면에 (debug) |
| `visualize_progressive_overlay.py` | 4단계 누적 overlay (L1 bg → L4 final) 각각 .mp4 |

## 좌표계 메모

OpenCV cam (`+x right, +y down, +z forward`) → pyrender world (`+y up, -z forward`).
`T_CV2GL = diag(1, -1, -1)`:

- **위치**:  `t_pr = T_CV2GL @ t_cam`
- **회전**:  `R_pr = T_CV2GL @ R_cam_xhand`
  `R_cam_xhand = R_mano @ R_MANO_XHAND[side]` (xhand local → MANO cam).
  `T_CV2GL`은 **출력 좌표계만** 바꿈 — input frame (xhand local)에는 적용하지 않는다.

## Depth alignment (Stage 7)

DA-V2 relative 출력은 disparity (higher = closer). HaWoR 21 joint × (양손)의 metric Z를
anchor로

```
d_pred ≈ a · (1 / Z_metric) + b
```

per-frame `(a, b)` LSQ. anchor 부족(<3) 프레임은 전체 median (a, b)로 fallback.
복원: `Z = a / (d_pred − b)`, `[0.05, 10] m` clip.

Cube isolation에는 metric 정확도가 필요하지 않지만 (top-q% disparity로 충분), composite
단계에서 cube 평면 깊이를 HaWoR MCP/wrist의 metric Z와 비교할 때 필요.

## Layered composite 로직 (Stage 10)

```
acc = bg (inpainted)
acc = α_behind · robot_rgb + (1-α_behind) · acc       # behind-MCP layer
acc = α_cube   · bg        + (1-α_cube)   · acc       # cube layer (= bg at cube_mask)
acc = α_front  · robot_rgb + (1-α_front)  · acc       # front-MCP layer
```

- `behind_mask = r_mask & (r_z + bias ≥ z_threshold_t)`  ← roomba pose pixels behind plane
- `front_mask  = r_mask & (r_z + bias <  z_threshold_t)`
- `z_threshold_t = mean(joints_*[t, threshold_joint, 2])`, valid hands only, 시간축
  Gaussian smoothing (sigma `--zmcp_sigma_t` frames)
- 각 binary mask는 Gaussian blur (sigma `--edge_sigma` px)로 0..1 alpha로 변환 후
  alpha-blend → 모서리 anti-alias

`--threshold_joint`로 cube가 robot을 occlude하는 양 조정:

| joint | z (median, m) | front : behind 비율 | 결과 |
|---|---|---|---|
| 0 (wrist)   | 0.244 | 0.86 | cube가 robot 위 많이 그려짐 |
| 5 (idx-MCP) | 0.279 | 2.29 | **기본값** — 균형 |
| 9 (mid-MCP) | 0.305 | 4.24 | cube가 robot 사이만 보임 |

## Cube mask 정제 흐름 (Stage 8 → 9)

```
1. depth_aligned[t][~M_hand]에서 top-q% (default q=0.25)  → rough mask
2. 가장 image center에 가까운 connected component만 keep (cube ≠ 가장 큰 blob, 종종 table)
3. MORPH_OPEN k=7 → table edge와 연결된 thin neck 절단
4. center-CC pick (다시) → cube 본체만 유지
5. MORPH_CLOSE k=5 → 작은 hole 메움 + 경계 smooth
6. cv2.convexHull → finger bite 영역까지 cube 전체 silhouette로 확장
7. dilate k=3 → 3px 마진
8. outlier rejection (MAD on area + centroid jump) → 망친 프레임은 temporally 가장
   가까운 valid 프레임으로 대체
9. SDF temporal smoothing (Gaussian sigma=8 frames) → 경계 떨림 제거
```

## 알려진 함정

### 1. SAM2 checkpoint 누락
경로: `third_party/sam2/checkpoints/sam2_hiera_large.pt`. 위 wget으로 받아둘 것.

### 2. SAM2 CUDA OOM (GPU 공유 시)
`segment_arms.py` 가 `init_state(..., offload_video_to_cpu=True)` 로 호출함.
안 그러면 1024×1024 float32 × 766 frames (~9.6 GB) 가 GPU에 올라감.

### 3. PyOpenGL 3.1.0 vs 3.1.4
pyrender가 PyOpenGL을 3.1.0으로 silently downgrade. 3.1.0은 `OpenGL.EGL.EGLDeviceEXT`가
없어 EGL 백엔드가 죽음. `pip install PyOpenGL==3.1.4`.

### 4. 프레임 수 mismatch
`retarget_input.npz` 의 T가 input frames보다 크면 `min`으로 자름. 반대로 npz가 짧으면
그만큼만 합성되고 나머지는 inpainted bg 그대로.

### 5. Cube isolation이 table을 잡아가는 경우
`crop_cube_layer.py`에서 quantile을 낮추거나 (`--quantile 0.10`), regularize 단계의
MORPH_OPEN 커널을 키워 (`--open_k 11`) 더 적극적으로 연결을 끊으면 됨.

### 6. Cube outline이 robot palm을 너무 많이 덮음
`composite_layered.py --threshold_joint 9` (mid-MCP) 로 cube 깊이 평면을 뒤로 밀거나,
`--depth_bias 0.02`로 robot z에 +2cm를 더해 robot을 더 자주 "앞"으로 분류.

## 파일 구조

```
src/inpainting/
├── README.md                          # 이 문서
├── _paths.py                          # SAM2/E2FGVI/DA-V2 paths, xhand URDF, R_MANO_XHAND
│
│   # core pipeline (run_layered.py 가 순서대로 호출)
├── prepare_demo.py                    # 1.  frames → video_L.mp4
├── inject_hawor_data.py               # 2.  HaWoR → bbox + hand_data + video_rgb_imgs.mkv
├── segment_arms.py                    # 3.  SAM2 → M_hand
├── inpaint_hands.py                   # 4.  E2FGVI legacy → inpainted bg
├── render_xhand_overlay_depth.py      # 5.  pyrender → robot RGBD
├── estimate_depth.py                  # 6.  Depth Anything V2 → disparity
├── align_depth.py                     # 7.  HaWoR anchors → metric depth
├── crop_cube_layer.py                 # 8.  rough cube mask
├── regularize_and_cut_cube.py         # 9.  cube mask cleanup + SDF smoothing
├── composite_layered.py               # 10. final 4-layer alpha-blend overlay
├── run_layered.py                     # end-to-end orchestrator (skips cached stages)
│
│   # legacy renderer + orchestrator (arm-mask clipped overlay; deprecated but kept)
├── render_xhand_overlay.py            #     pyrender + arm-mask clip
├── run.py                             #     legacy orchestrator
│
│   # debug visualizers
├── visualize_depth.py
├── visualize_depth_with_mask.py
├── visualize_depth_stats.py
├── visualize_shallow_nonhand.py
├── visualize_layers.py
└── visualize_progressive_overlay.py
```
