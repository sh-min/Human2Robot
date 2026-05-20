# Simulation Tool — Cube pose optimization

큐브 manipulation 시뮬레이션에 retargeting된 손을 올려놓기 위해, **각 skill segment 첫 프레임의 큐브 6-DOF 포즈**를 추정.

큐브는 RGB로 object tracking이 안 되므로 다음 세 단서를 함께 최소화:

1. **Contact SDF** — `손이 닿은 MANO vertex`(palmar + fingertip 영역)들이 큐브 표면에 있도록.
2. **Silhouette** — 큐브를 projection했을 때 RGB cube mask와 일치하도록.
3. **Axis alignment** — 큐브의 +Z축이 skill에 따라 정해진 카메라 좌표축과 정렬되도록 (회전 방향 제약).

## 환경

`RFM_retarget` (retargeting과 동일). assets/URDF는 `src/retargeting/assets/`를 그대로 재사용 — 별도 셋업 없음.

## 디렉터리

```
src/simulation_tool/
├── optimize_cube_pose.py   # ★ 단일 프레임 큐브 포즈 최적화 (Powell)
├── batch_cube_poses.py     # predictions.pt를 읽어 각 segment 첫 프레임에 대해 일괄 최적화
└── inspect_cube_pose.py    # MANO + xhand mesh + cube를 Open3D로 정적 3D 시각화
```

`R_mano_xhand_*.npy`, `palmar_mask_*.npy`, `fingertip_mask_*.npy`, `finger_part_*.npy`, `mano_faces_*.npy`, `xhand/` URDF는 모두 `src/retargeting/assets/`에서 자동으로 읽어옴.

## 입력

| 경로 | 내용 |
|---|---|
| `<episode>/predictions.pt` | skill classifier 결과. `labels`, `segments=[(start, end, label_idx), …]` 포함 |
| `<episode>/rgb_hawor/retarget_input.npz` | HaWoR MANO 결과 (`verts_{left,right}`, `joints_{left,right}`) |
| `<episode>/rgb_mask/rgb_frame{NNNNN}.png` | 각 프레임 큐브 마스크 (binary) |
| `<episode>/rgb/rgb_frame{NNNNN}.png` | RGB 원본 (overlay용) |

## Skill → 손 / 축 매핑

`batch_cube_poses.py`가 자동으로 적용:

| Skill letter | 잡는 손 | 큐브 +Z target axis (cam frame, sign-free) |
|---|---|---|
| `R` | left  | cam +x |
| `L` | right | cam +x |
| `U` | left  | cam +y |
| `D` | left  | cam +y |
| `F` | right | cam +z |
| `B` | left  | cam +z |

`TRANS` segment는 건너뜀.

## 사용법

### 1) 전체 segment 일괄 처리 (보통 이걸 씀)

```bash
conda activate RFM_retarget
cd <repo_root>/src/simulation_tool

python batch_cube_poses.py \
    --episode /media/.../saved_frames_YYYYMMDD_HHMMSS \
    [--silhouette_mode dt]          # 기본 dt. 손에 가려져 mask 경계가 깨질 때 line으로 교체 가능
    [--cube_size 0.05] [--img_focal 497.77] \
    [--alpha 1.0] [--beta 0.01] [--axis_weight 0.1]
```

출력: `<episode>/rgb_hawor/cube_poses/seg{KK}_f{NNNNN}_{SKILL}.npz` + `_viz.png`

각 `.npz`:

| key | shape / type | 설명 |
|---|---|---|
| `center` | (3,)   float64 | 큐브 중심 (cam frame, meters) |
| `rotvec` | (3,)   float64 | 큐브 회전 (scipy rotation vector) |
| `size`   | scalar         | 큐브 한 변 길이 (`--cube_size`와 동일) |
| `iou`    | scalar         | reporting용 silhouette IoU |
| `contact_loss` | scalar   | reporting용 contact SDF 합 |

### 2) 단일 프레임만 디버깅

```bash
python optimize_cube_pose.py \
    --npz   <episode>/rgb_hawor/retarget_input.npz \
    --mask  <episode>/rgb_mask/rgb_frame00120.png \
    --rgb   <episode>/rgb/rgb_frame00120.png \
    --hand  left          # R/U/D/B면 left, F/L이면 right
    --frame 120 \
    --target_axis y       # 위 표 참고
    [--silhouette_mode dt|line]   # 기본 dt
    --out_pose ./cube.npz --out_viz ./cube_viz.png
```

### 3) 3D로 검증

```bash
python inspect_cube_pose.py \
    --npz       <episode>/rgb_hawor/retarget_input.npz \
    --pkl       <episode>/rgb_hawor/qpos_xhand_contact_left.pkl \
    --cube_pose <episode>/rgb_hawor/cube_poses/seg03_f00120_RCW.npz \
    --frame     120 --hand left
```

Open3D 창에 MANO mesh (회색) + retargeted xhand mesh (파랑) + 최적화된 큐브 (주황) + 좌표축이 함께 뜸.

## 손실 함수

총 loss = `α * contact_SDF² + β * silhouette + γ * (1 - (cube_+Z · target_axis)²)`

- **contact_SDF** (`α`, 기본 1.0): MANO palmar+fingertip 영역 vertex 중 `손이 잡고 있는 손가락`의 vertex에 한해, 큐브 oriented-box SDF의 절댓값² 합. 박스 안쪽은 `lambda_inside=5.0` 배 가중치로 패널티 (큐브가 손 안쪽으로 너무 파고들지 않도록).
- **silhouette** (`β`, 기본 0.01): 두 가지 모드.
  - `dt` (기본) — 큐브 corners projection → convex hull 영역과 GT mask를 **양방향 distance transform**으로 매칭. mask 바깥으로 삐져나간 픽셀과 mask 안인데 큐브가 덮지 못한 픽셀 둘 다 패널티.
  - `line` — Hough로 큐브 마스크 직선 edge를 뽑아 큐브 edge와 매칭. 손에 큐브가 가려졌을 때 견고하지만 보통은 `dt`가 더 안정적.
- **axis** (`axis_weight=γ`, 기본 0.1): 큐브 +Z와 카메라 좌표계의 target axis(`x|y|z`) 내적 제곱이 1이 되도록 (부호 무관, 회전 방향만 맞춰주는 약한 prior).

## 메모

- 최적화는 `scipy.optimize.minimize(method="Powell")`. 6-DOF (center 3 + rotvec 3) 초기값은 MANO 손 contact 영역 centroid.
- 카메라 intrinsic은 `(fx, fy) = img_focal`, `(cx, cy) = (W/2, H/2)`를 가정. 다른 카메라면 `--img_focal`만 바꿔도 보통 충분.
- `predictions.pt`는 `src/skill_classifier`의 출력 포맷을 따름.
