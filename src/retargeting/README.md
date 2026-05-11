# Retargeting (xhand)

HaWoR이 만들어낸 양손 MANO 결과를 xhand 로봇 손 qpos sequence로 변환.

**Stage 1 (default)** — DexPilot vector retargeting (wrist + 5 fingertip 벡터 매칭)
**Stage 2 (`--contact`)** — Stage 1 위에 HACO contact 정보로 손가락 자세 미세 조정. palmar + fingertip 영역 contact vertex만 사용하고, **vertex outward normal까지 같이 매칭**해서 xhand의 손등쪽이 contact 점에 끌리는 문제를 방지.

백엔드: vendored [dex-retargeting](https://github.com/dexsuite/dex-retargeting) (`third_party/dex-retargeting`) — DexPilot + pinocchio FK + scipy.optimize.

## 환경 (conda env: `RFM_retarget`)

```bash
cd <repo_root>/third_party/dex-retargeting
conda env create -f environment.yml -n RFM_retarget
conda activate RFM_retarget
pip install -e .[example]
```

> `pip install -e`는 절대경로 기록 — 폴더 옮기면 재실행 필요.

## 디렉터리

```
src/retargeting/
├── _paths.py                    # 공용 경로 + R_MANO_XHAND 자동 로딩
├── assets/
│   ├── xhand/                   # xhand URDF + STL meshes
│   ├── R_mano_xhand_{r,l}.npy   # Procrustes-fit MANO↔xhand rotation
│   ├── palmar_mask_{r,l}.npy    # 778 bool, MANO palm-side vertex mask
│   ├── fingertip_mask_{r,l}.npy # 778 bool, MANO distal vertex mask
│   ├── finger_part_{r,l}.npy    # 778 int, MANO joint index per vertex
│   └── mano_faces_{r,l}.npy     # MANO mesh faces (vertex normal 계산용)
├── configs/                     # DexPilot yml (양손)
├── retarget_from_npz.py            # ★ MANO npz → xhand qpos (Stage 1 + Stage 2)
├── overlay_on_rgb.py               # qpos + RGB → overlay mp4
├── compare_stages.py               # Stage 1 vs Stage 2 인터랙티브 3D 뷰어 (오른손)
├── retarget_from_npz_contact.py    # (legacy) centroid 방식 contact retargeting
├── visualize_contact_retarget.py   # (legacy) retargeting 시각화
├── build_palmar_mask.py            # palmar mask 생성 (1회)
├── build_finger_parts.py           # finger-part / tip mask 생성 (1회)
├── compute_R_mano_xhand.py         # Procrustes R 계산 (1회)
├── extract_urdf.py                 # STAR1 풀바디 URDF → xhand 분리 (1회)
└── ...                             # 기타 디버그/검증 스크립트
```

## 1회 셋업 — assets 생성

`assets/`의 `.npy` 파일들은 repo에 포함되어 있음. 만약 새로 빌드해야 한다면:

```bash
# Procrustes R (MANO canonical MCP knuckle ↔ xhand MCP)
conda activate hawor     # MANO model 필요
cd <repo_root>/src/retargeting
python compute_R_mano_xhand.py
python build_palmar_mask.py
python build_finger_parts.py
```

세 스크립트 모두 1회 실행. 결과는 `assets/*.npy`에 저장되고, 그 후 `_paths.py`가 자동으로 로드.

## 실행

선행 조건: `src/hand_estimation/extract_for_retarget.py`로 `retarget_input.npz` 생성. `--contact` 사용 시 `src/contact_estimation/extract_hand_contact.py`로 `contact/*.npz`도 생성되어 있어야 함.

### Retargeting (양손)

**Stage 1 (vector only):**

```bash
conda activate RFM_retarget
cd <repo_root>/src/retargeting

python retarget_from_npz.py \
    --npz /path/to/<seq>_hawor/retarget_input.npz
```

출력: 같은 폴더에 `qpos_xhand_{right,left}.pkl`

**Stage 2 (contact-aware refinement):**

```bash
python retarget_from_npz.py \
    --npz /path/to/<seq>_hawor/retarget_input.npz \
    --contact
```

출력: `qpos_xhand_contact_{right,left}.pkl` (stage 1 pkl은 그대로 둠)

`--contact_dir <path>` 미지정 시 `<npz 부모>/../contact` 사용.

**옵션:**
- `--hand right|left|both` (default both)
- `--alpha 0.001` — anchor 강도 (stage1에서 이탈 패널티). 작게 → contact term 강함
- `--normal_thr 0.3` — normal 호환 임계값 (n_h · n_r > thr)
- `--out_dir <path>`

**Pkl 포맷:**

| 키 | 설명 |
|---|---|
| `data` | `(T, 12)` xhand 12-DOF qpos |
| `joint_names` | qpos joint 순서 |
| `valid` | `(T,)` bool, 프레임 유효성 |
| `config_path`, `hand`, `dof` | 메타 |

### RGB Overlay

```bash
SEQ=/path/to/<dataset>
HAWOR=$SEQ/rgb_hawor

# stage 1 (vector)
python overlay_on_rgb.py \
    --npz       $HAWOR/retarget_input.npz \
    --rgb_dir   $SEQ/rgb \
    --right_pkl $HAWOR/qpos_xhand_right.pkl \
    --left_pkl  $HAWOR/qpos_xhand_left.pkl \
    --out       $HAWOR/overlay_stage1.mp4 \
    --img_focal 497.77

# stage 2 (contact)
python overlay_on_rgb.py \
    --npz       $HAWOR/retarget_input.npz \
    --rgb_dir   $SEQ/rgb \
    --right_pkl $HAWOR/qpos_xhand_contact_right.pkl \
    --left_pkl  $HAWOR/qpos_xhand_contact_left.pkl \
    --out       $HAWOR/overlay_stage2_contact.mp4 \
    --img_focal 497.77
```

원본 RGB 위에 cam-frame xhand 메쉬가 alpha-blended로 합성 (default α=0.7). 두 영상 비교해서 stage 2가 contact 영역에서 손가락이 더 잘 닿는지 확인.

### Stage 1 vs Stage 2 인터랙티브 비교 (오른손)

```bash
python compare_stages.py \
    --npz  /path/to/<seq>_hawor/retarget_input.npz \
    --pkl1 /path/to/<seq>_hawor/qpos_xhand_right.pkl \
    --pkl2 /path/to/<seq>_hawor/qpos_xhand_contact_right.pkl
```

Open3D 뷰어로 같은 cam frame 위에 4종을 동시 표시 (오른손):

| 색 | 내용 |
|---|---|
| 회색 mesh | MANO (사람 손) |
| 연한 청록 mesh | xhand stage 1 (vector only) |
| 연한 빨강 mesh | xhand stage 2 (contact refined) |
| 노란 점 | palmar + fingertip 필터된 contact verts |

**키:**

| 키 | 동작 |
|---|---|
| `SPACE` | pause / play |
| `[` / `]` | 한 프레임 이전 / 다음 |
| `,` / `.` | 10 프레임씩 |
| `H` / `E` | 처음 / 끝 |
| `M` / `1` / `2` / `C` | MANO / xhand stage1 / xhand stage2 / contact 점 toggle |
| 마우스 | 드래그=회전, 스크롤=줌, Shift+드래그=pan |

## (legacy) retarget_from_npz_contact.py + visualize_contact_retarget.py

> 이전 contact-aware retargeting 구현. **centroid-replacement 방식**: contact 있는 손가락의 MANO fingertip keypoint를 해당 손가락 contact vertex의 centroid로 교체한 뒤 일반 retargeting 호출. `retarget_from_npz.py --contact` 의 normal-aware Chamfer 방식과 다름

### Centroid-방식 retargeting

```bash
conda activate RFM_retarget
cd <repo_root>/src/retargeting

python retarget_from_npz_contact.py \
    --npz /path/to/<episode>/rgb_hawor/retarget_input.npz
```

출력: 같은 폴더에 `qpos_xhand_contact_{right,left}.pkl`

> 같은 파일명을 `retarget_from_npz.py --contact` 도 사용. 둘 중 마지막에 실행한 게 디스크에 남음. 비교하려면 한쪽을 따로 보관.

옵션:
- `--hand right|left|both`
- `--contact_dir <path>` (default `<npz 부모>/../contact`)
- `--out_dir <path>`

### Retargeting 시각화 (디버그용 패널)

선행 조건: `retarget_from_npz.py` (Stage 1) + `retarget_from_npz_contact.py` (legacy Stage 2) 모두 완료되어 아래 4개 pkl 존재.
- `qpos_xhand_{right,left}.pkl`
- `qpos_xhand_contact_{right,left}.pkl`

```bash
conda activate vjepa2-312
cd <repo_root>/src/retargeting

python visualize_contact_retarget.py \
    --npz /path/to/<episode>/rgb_hawor/retarget_input.npz \
    --frame 10
```

출력 (default `src/retargeting/vis/`):

| 파일 | 내용 |
|---|---|
| `frame{N}_mano_2d.png` | MANO만 — original vs contact-adjusted |
| `frame{N}_mano_3d.html` | 위 인터랙티브 3D |
| `frame{N}_xhand_2d.png` | MANO + xhand — original vs contact-adjusted |
| `frame{N}_xhand_3d.html` | 위 인터랙티브 3D |

옵션:
- `--frame N` (default 10)
- `--pkl_dir`, `--contact_dir`, `--out_dir`

## R_MANO_XHAND (정렬)

`assets/R_mano_xhand_{right,left}.npy`에 저장된 3×3 rotation을 `_paths.py`가 모든 스크립트에 노출. **MANO canonical wrist frame → xhand wrist link frame** 변환.

계산 방식: `compute_R_mano_xhand.py`가 MANO 5 MCP knuckle (canonical T-pose) 위치와 xhand 5 MCP joint origin을 Procrustes (SVD) 로 정합. 평균 residual ~17mm (rigid rotation 한계).

## Stage 2 동작 원리

1. Human contact (`{hand}_contact_mask` from HACO) 에서 **palmar AND fingertip** 영역만 남김
2. 각 contact vertex h:
   - 위치 (xhand wrist frame)
   - outward normal (MANO mesh + face → trimesh로 매 프레임 계산)
3. xhand fingertip link mesh의 각 vertex r:
   - 위치는 FK로 계산
   - normal은 link-local frame에서 1회 계산 후 매 iter rotation 적용
4. **Normal-aware Chamfer**: 각 h에 대해 `n_h · n_r > normal_thr` 인 r들 중 위치 가장 가까운 것의 거리 제곱을 합산
5. anchor `||q − q_stage1||²` 추가하여 stage 1 결과에서 너무 멀어지지 않게 제약
6. scipy L-BFGS-B로 12 DOF 손가락 자세만 최적화 (손목 위치/회전은 HaWoR 그대로 유지)

## 검증된 결과 (cube manipulation 예시, Procrustes R 적용 후)

DexPilot last distance:
- right: **0.0011** (이전 R 0.0134 → 12× 개선)
- left:  **0.0025** (이전 R 0.0075 → 3× 개선)
