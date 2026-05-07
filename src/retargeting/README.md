# Retargeting (xhand)

HaWoR이 만들어낸 양손 MANO 결과를 받아 [xhand](../../retarget/models/star1) 로봇 손
qpos sequence로 변환하고, 검증용 시각화를 제공.

백엔드: vendored [dex-retargeting](https://github.com/dexsuite/dex-retargeting)
(`third_party/dex-retargeting`) — DexPilot vector retargeting + pinocchio FK/IK.

## 환경 (conda env: `RFM_retarget`)

```bash
# 1) env 만들기 (numpy 2.x + pinocchio + sapien)
cd <repo_root>/third_party/dex-retargeting
conda env create -f environment.yml -n RFM_retarget   # 또는 README 참고
conda activate RFM_retarget

# 2) dex-retargeting을 editable install (이 repo의 third_party 경로 그대로)
pip install -e .[example]
```

> `pip install -e`는 **현재 폴더 절대경로를 기록**하므로, 폴더 위치가 바뀌면 재실행 필요.

## 디렉터리

```
src/retargeting/
├── _paths.py                # 모듈 공용 경로 상수 (수정 X)
├── assets/xhand/            # xhand URDF + STL meshes (양손)
├── configs/                 # DexPilot yml (양손)
│   ├── xhand_right_dexpilot.yml
│   └── xhand_left_dexpilot.yml
├── extract_urdf.py          # STAR1 full-body URDF → xhand 분리 (1회)
├── retarget_from_npz.py         # ★ MANO npz → qpos pkl
├── retarget_from_npz_contact.py # ★ MANO npz + HACO contact → qpos pkl
├── overlay_on_rgb.py        # qpos + RGB → overlay mp4 (sapien)
├── play_sequence.py         # 인터랙티브 3D 뷰어 (trimesh)
├── inspect_combined.py      # axis 검증용 (xhand + MANO skeleton 양손)
├── inspect_wrist_axes.py    # 단일 핸드 axis arrows (URDF 검증)
└── project_contact.py       # contact 점 RGB projection
```

## 실행

선행 조건: `src/hand_estimation/extract_for_retarget.py`로 `retarget_input.npz`가
이미 생성돼 있어야 함.

### 1. Retargeting (양손)

#### 1.1 Contact 없는 버전

```bash
conda activate RFM_retarget
cd <repo_root>/src/retargeting

python retarget_from_npz.py \
    --npz /path/to/<seq>_hawor/retarget_input.npz
```

출력: 같은 폴더에 `qpos_xhand_right.pkl`, `qpos_xhand_left.pkl`

| 키 | 설명 |
|---|---|
| `data` | `(T, 12)` xhand 12-DOF qpos |
| `joint_names` | qpos 순서대로 joint 이름 |
| `valid` | `(T,)` 프레임 유효성 |
| `config_path`, `hand`, `dof` | 메타 |

옵션:
- `--hand right|left|both` (default both)
- `--out_dir <path>`

#### 1.2 Contact 적용 버전

선행 조건: 위 1.1 npz + `src/contact_estimation/extract_hand_contact.py`로
`<episode>/contact/*.npz`가 생성돼 있어야 함.

```bash
# contact 추출 (conda activate haco)
cd <repo_root>/src/contact_estimation
python extract_hand_contact.py \
    --input_dir /path/to/<episode>

# contact retargeting (conda activate RFM_retarget)
cd <repo_root>/src/retargeting
python retarget_from_npz_contact.py \
    --npz /path/to/<episode>/rgb_hawor/retarget_input.npz
```

`--contact_dir` 미지정 시 `<episode>/contact` 로 자동 설정.

출력: npz와 같은 폴더에 `qpos_xhand_contact_right.pkl`, `qpos_xhand_contact_left.pkl`

contact 적용 방식: contact mask가 있는 손가락은 fingertip joint 위치를 해당 finger의
contact vertex centroid(물체 표면 근사점)로 대체한 뒤 retargeting. contact 없는
손가락은 기존과 동일.

옵션:
- `--hand right|left|both` (default both)
- `--contact_dir <path>`
- `--out_dir <path>`

### 2. RGB Overlay 영상

```bash
python overlay_on_rgb.py \
    --npz       /path/to/<seq>_hawor/retarget_input.npz \
    --rgb_dir   /path/to/rgb_frames \
    --right_pkl /path/to/<seq>_hawor/qpos_xhand_right.pkl \
    --left_pkl  /path/to/<seq>_hawor/qpos_xhand_left.pkl \
    --out       /path/to/<seq>_hawor/overlay.mp4 \
    --img_focal 497.77
```

원본 RGB 위에 cam-frame xhand 메쉬가 alpha-blended로 합성 (default α=0.7).

### 3. 인터랙티브 3D 플레이백

```bash
python play_sequence.py \
    --npz       /path/to/<seq>_hawor/retarget_input.npz \
    --right_pkl /path/to/<seq>_hawor/qpos_xhand_right.pkl \
    --left_pkl  /path/to/<seq>_hawor/qpos_xhand_left.pkl
```

trimesh 뷰어로 양손 xhand + MANO skeleton + camera frustum 함께 재생.

| 키 | 동작 |
|---|---|
| SPACE | pause / play |
| LEFT / RIGHT | 한 프레임씩 (자동 pause) |
| HOME / END | 처음 / 끝 |
| 마우스 드래그 / 스크롤 | 회전 / 줌 |

옵션: `--no_bones` (joint 점만), `--bone_radius 0.005`, `--fps 15`

### 4. Frame alignment 검증 (디버깅용)

```bash
# 양손 동시 inspector (xhand 메쉬 + MANO skeleton)
python inspect_combined.py \
    --npz /path/to/<seq>_hawor/retarget_input.npz --frame 0

# 단일 손 axis arrows
python inspect_wrist_axes.py --hand right --save /tmp/right_axes.png
```

### 5. xhand URDF 재생성 (일반적으로 불필요)

```bash
# extract_urdf.py 안의 SRC_URDF / SRC_MESH_DIR을 STAR1 위치로 수정 후
python extract_urdf.py
```

### 6. (선택) Contact projection

```bash
python project_contact.py \
    --rgb_dir <...>/rgb \
    --contact_dir <...>/contact \
    --out <...>/contact_projection.mp4 \
    --fx 497.77
```

## 핵심 정렬 정보

`R_MANO_XHAND` (in `retarget_from_npz.py`) — MANO canonical wrist frame을 xhand
wrist link frame으로 변환:

```python
R_MANO_XHAND = {
    "right": [[0,  0, 1], [0, -1, 0], [1, 0, 0]],  # MANO (x,y,z) -> xhand (z, -y, x)
    "left":  [[0,  0,-1], [0,  1, 0], [1, 0, 0]],  # MANO (x,y,z) -> xhand (-z, y, x)
}
```

검증: `inspect_combined.py`로 양손 동시 plot. 손가락별 색 (thumb=red,
index=blue, middle=green, ring=orange, pinky=purple)이 xhand 손가락과 일치하면 OK.

## 검증된 결과 (cube manipulation 예시)

DexPilot last distance:
- right: 0.007128
- left:  0.007490

→ 두 손 모두 사람 손 자세를 충실히 재현.
