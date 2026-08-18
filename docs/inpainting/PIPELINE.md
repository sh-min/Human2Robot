# 사람 조작 영상 → 로봇 조작 영상 파이프라인

한 사람이 물체를 집는 영상을 넣으면, 같은 동작을 **RB5-850e 팔 + XHAND1 손**이 하는 영상으로
바꾼다. 48번(`a001_08051547_c056`)에서 굳힌 방식을 IMG_5393에 그대로 적용해 통과시킨 뒤
정리한 것이다.

실행:

```bash
scripts/video_to_robot.sh --video /path/to/clip.MOV --name myclip
scripts/video_to_robot.sh --video ... --name ... --from robot     # 중간 재개
scripts/video_to_robot.sh --video ... --name ... --diffueraser    # 배경 정제 추가
```

RTX 5070(12 GB)에서 574프레임 19초 클립 기준 **약 35분**.

---

## 전체 구조

```
 원본 영상
   │
   ├─ [1] 프레임 추출 1280×720
   │
   ├─ [2] HaWoR ──────────────► 손목 6-DoF + MANO 관절/버텍스 + 초점거리
   │                              │
   ├─ [3] HaCo ───────────────► 버텍스별 접촉 확률 (778개)
   │                              │
   │                              ├─► 손가락별 접촉 상태 (파지 구간 검출)
   │                              │
   ├─ [4] 리타게팅(접촉 반영) ─► XHAND1 관절각 12-DoF
   │
   ├─ [5] SAM2 ───────────────► 사람 손·팔 마스크 ──┐
   ├─ [6] Depth-Anything V2 ──► 메트릭 깊이         │
   │                                                │
   ├─ [7] 물체 ────────────────────────────────────┤
   │      HaCo 구간 → SAM2 추적 → 가려진 부분 복원  │
   │                                                │
   ├─ [8] 배경 ────────────────────────────────────┘
   │      ProPainter로 사람 제거 → (선택) DiffuEraser 정제
   │
   ├─ [9] 로봇 ─► 베이스 자동 배치 → IK → 손을 플랜지에 결합 → 렌더
   │
   └─ [10] 6단계 합성 + 높이대역 접촉 그림자 ──────► 결과 영상
```

---

## 단계별

### [1] 프레임 추출

원본을 1280×720으로 낮춰 `rgb/rgb_frame%05d.jpg`와 `video_L.mp4`를 만든다. 이후 모든 단계가
이 해상도를 기준으로 한다. 12 GB 카드에서 500프레임대 클립을 다루는 상한이다.

### [2] 손 자세 — HaWoR

`src/hand_estimation/extract_for_retarget.py`

프레임별 MANO 손목 위치·방향, 21개 관절, 778개 버텍스, 그리고 **초점거리 추정치**를 낸다.

> **반드시 확인할 것.** 초점거리가 틀리면 이후 모든 투영이 조용히 밀린다. IMG_5393에서는
> 추정치 600이 나왔는데, 48번 데이터의 925.3과 크게 달라 600/925/1200을 각각 투영해 눈으로
> 대조했다. 600만 손 위에 정확히 떨어졌다.
> 검증 도구: `src/contact_estimation/visualize_contact_overlay.py`

### [3] 접촉 — HaCo

`src/contact_estimation/extract_hand_contact.py` → `aggregate_finger_contact.py`

프레임마다 778개 버텍스 각각이 물체에 닿았는지 확률을 낸다. 이어서 손가락별 점수와
히스테리시스를 적용한 접촉 상태로 압축한다.

**이 파이프라인에서 HaCo는 네 곳에 쓰인다.** 원래 접촉 추정용이던 모델이 영상별 수작업을
없애는 축이 됐다.

| 쓰임 | 무엇을 대체했나 |
|---|---|
| 리타게팅 보정 | 손가락이 물체 표면에 실제로 닿게 접힘 |
| 파지 구간·박스 검출 | 사람이 영상 훑으며 타이핑하던 물체 JSON |
| 파지 가림 마스크 | 영상마다 손으로 칠하던 `force_front_v31…v40.npy` |
| 앞/뒤 split depth (39번) | 스칼라 평면 하나 — 현재 본선 미적용 |

### [4] 리타게팅 — 접촉 반영

`src/retargeting/retarget_from_npz.py --contact`

1단계 DexPilot 벡터 매칭 후, HaCo 접촉 버텍스를 향해 XHAND1 손끝을 당기는 2단계 보정.
IMG_5393에서 **574프레임 중 521프레임이 보정**됐다.

### [5] 사람 마스크 — SAM2

`segment_arms.py` → `augment_hand_mask_from_keypoints.py`

HaWoR 관절로 프롬프트를 만들어 SAM2로 손·팔을 추적하고, SAM2가 물체에 닿은 손끝을 잘라먹는
문제를 관절 기반 보강으로 메운다.

> 보강 반경(dilate 16 / bone 8 / tip 12)이 a001에 맞춰진 값이다. IMG_5393에서 팔이 수세미
> 옆을 지날 때 수세미까지 먹어서 배경에서 지워졌다. 클립에 따라 줄여야 할 수 있다.

### [6] 깊이 — Depth-Anything V2

`estimate_depth.py` → `align_depth.py --hawor_npz`

상대 깊이를 뽑고 HaWoR 관절 깊이에 맞춰 **미터 단위로 정렬**한다. `--hawor_npz`가 없으면
스케일이 3배 어긋나 접촉 그림자가 전 프레임 0이 된다. 그림자 단계의 유일한 입력이다.

### [7] 물체 — 접촉에서 스펙 생성 → 추적 → 복원

```
build_object_segments_from_contact.py   # HaCo 접촉 → 구간·시드·박스·점
      ↓
segment_interaction_objects.py          # SAM2 다중 물체 추적
      ↓
complete_occluded_objects.py            # 사람 손이 가렸던 부분만 채움
```

**스펙 생성**이 48번 대비 가장 큰 변화다. 엄지 제외 2손가락 이상이 접촉한 구간을 파지로 보고,
접촉 버텍스를 투영해 박스를 씌운다. 추적 구간은 접촉 전 30프레임(`--lead_frames`), 후
8프레임(`--trail_frames`)까지 넓힌다 — 손이 다가오는 동안에도 물체 레이어가 있어야 손가락을
물체 뒤로 보낼 수 있다.

IMG_5393에서 파지 구간 5개(머그·오뜨상자·초록팩·용기·수세미)와 박스 5개가 전부 맞았다.

**복원**은 보이는 물체에서 보수적 amodal 실루엣을 만들되, 새로 칠하는 픽셀을 사람이 가린
영역으로만 제한하고 같은 프레임의 진짜 물체 텍스처에서 가장 가까운 것을 복사한다. 빈 책상에
물체를 만들어내지 못하는 구조다.

> **유일하게 남은 수작업.** 색으로 물체를 책상과 구분할 수 없으면(반투명 용기) 자동 프롬프트가
> 빈 책상을 문다. `--overrides`에 그 물체의 박스·점만 손으로 넣는다. 별도 파일이라 스펙을
> 다시 생성해도 보정이 살아남는다.

### [8] 배경 — 사람 제거

```
export_propainter_masks.py → ProPainter (640×360) → assemble_propainter_background.py
                                    ↓ (선택)
                          DiffuEraser (prior로 재사용) → assemble
```

ProPainter를 640에서 돌리고 원본 해상도로 되붙인다. `--diffueraser`를 주면 **그 결과를
디퓨전 prior로 재사용해** 정제한다 — ProPainter를 두 번 돌리지 않는다.

DiffuEraser는 prior를 VAE로 인코딩해 시작 latent로 넣고 2스텝만 denoise하는 **refiner**다.
prior 없이는 성립하지 않는다. 빠르게 움직이는 물체가 가려졌다 드러나는 구간에서 차이가 크고
(ProPainter는 무지개 얼룩, DiffuEraser는 포장지 글씨까지 복원), 가릴 게 없는 구간은 동일하다.

### [9] 로봇 — 배치·IK·결합·렌더

```
rb5_build_overlay_input.py --base_offscreen left --mount_hand
      ↓
render_xhand_overlay_depth.py                      # RGB/깊이/마스크
render_xhand_overlay_depth.py --thumb_mask_only     # 엄지 마스크(스테이지 6용)
```

**베이스 자동 배치.** 45·46번에서 평면과 x·z를 손으로 맞추던 값을 격자 탐색이 대신한다.
조건은 세 가지 — 베이스 몸통이 화면 가장자리를 실제 반경(`--base_clear_m`, 미터)만큼 벗어날
것, 전 궤적이 리치 구 안에 들 것, 그리고 비용에 **최악 프레임 오차**를 넣을 것. 부분표본
p90만 보면 사람이 가장 멀리 뻗는 몇 프레임을 놓친다. 베이스는 전 프레임 하나의 고정 좌표다.

**손을 팔에 결합.** `--mount_hand`가 손을 팔의 실제 FK 플랜지에서 역산해 배치한다. 이전
`--snap_flange`는 link6를 목표로 순간이동시켜 손목3이 최대 79 mm 벌어진 채, 실제 로봇이 만들
수 없는 형상으로 그려졌다. 지금은 손이 팔에 의해 놓이므로 **구조적으로 분리될 수 없고**,
관절각은 IK가 관절 한계 안에서 푼 값 그대로다.

| | 이전 | 지금 |
|---|---|---|
| 손목3 벌어짐 | 최대 79.2 mm | 구조적으로 0 |
| 손이 사람 손목을 벗어난 거리 | — | 최대 2.7 mm |
| 지터 스냅 >0.3 rad | 4.7 % | 3.0 % |

### [10] 합성

```
build_contact_force_front.py   # 파지한 네 손가락을 물체 뒤로
      ↓
composite_interaction_objects.py
```

**파지 가림.** HaCo 접촉 버텍스 중 **엄지를 뺀** 네 손가락만 투영하고, 그 점이 닿은 물체
덩어리 **전체**를 마스크로 잡는다. 접촉은 "어느 물체를 언제" 가릴지만 고르고 범위는 물체
자신의 넓이가 정한다 — 접촉점 주변만 가리면 수동 마스크의 1/5밖에 안 된다.

접근 구간에서는 판정을 **렌더된 로봇 마스크**로 한다. 사람 버텍스와 화면 속 로봇은 접근 중
벌어져 있어서, 로봇이 이미 물체 뒤인데 사람 버텍스는 아직 물체 밖인 상황이 생긴다. 로봇의
엄지 제외 픽셀이 물체와 `--lead_overlap_px`(120 px) 이상 겹치면 **그 프레임에 바로** 가린다.

측정(가림 시작 − 겹침 시작, 음수가 미리 들어감): −8 / −4 / −10 / 0 / 0 프레임.

**6단계 합성** — 순서가 불변 조건이다:

```
1 배경  →  2 로봇(뒤)  →  3 물체  →  4 로봇(앞, 엄지 제외)
        →  5 물체 강제 전경(엄지 제외)  →  6 엄지 최종
```

엄지는 2·4·5에서 도려내고 6에서 마지막에 그린다. 이 순서가 깨지면 엄지가 물체 뒤로 넘어간다.

**그림자**는 깊이에서 RANSAC으로 지지평면을 잡고, 로봇 표면을 평면에 투영하되 높이로 5개
대역을 나눈다. 대역마다 반그림자가 넓어지고(`70 px/m`) 진하기가 감쇠한다(`exp(−t/0.30 m)`).
대역 안에서만 정규화하므로 책상에 붙은 손이 50 cm 위 팔의 그림자를 지워버리지 않는다.

---

## 자동/수동 경계

| 항목 | 상태 |
|---|---|
| 초점거리 | 자동 (검증은 눈으로) |
| 손 자세·접촉·리타게팅 | 자동 |
| 사람 마스크·깊이 | 자동 |
| 물체 구간·박스·추적 | 자동 |
| 물체 소실부 복원 | 자동 |
| 배경 복원 | 자동 |
| 로봇 베이스·IK·결합 | 자동 |
| 파지 가림 | 자동 |
| 그림자 | 자동 |
| **색 대비 없는 물체의 프롬프트** | **수동 (`--overrides`)** |

IMG_5393과 IMG_5394 모두 수작업은 반투명 용기 하나뿐이었다.

## 알려진 한계

- **손끝 마스크 보강 반경이 a001 기준**이다. 팔이 다른 물체 옆을 스치면 그 물체까지 배경에서
  지워진다.
- **추적되지 않는 손은 남는다.** HaWoR가 한쪽 손만 유효로 보면 화면에 들어오는 다른 손은
  마스크에도 제거에도 들어가지 않는다.
- **복원 텍스처에 손가락 자국**이 옅게 남는다. 가장 가까운 물체 픽셀을 복사하는 방식의 한계다.
- **팔이 화면 밖으로 나갔다 들어오면** 화면에서 두 조각으로 보인다(3D에서는 이어져 있다).
- **DiffuEraser는 12 GB에서 빠듯하다.** 640 해상도와 청크 디코딩이 없으면 OOM이다.

## 결과

| 영상 | 프레임 | 결과 |
|---|---|---|
| IMG_5393 | 574 | `results/0818/` |
| IMG_5394 | 537 | `results/0818_IMG5394/` |
| 48번 (a001) | 553 | `results/video/48_2026-08-14_팔그림자_높이대역/` |

---

## 부록 — DiffuEraser 설치 메모

이 저장소에 벤더링하지 않았다(가중치 15 GB). 별도로 클론·설치한다.

```bash
git clone --depth 1 https://github.com/lixiaowen-xw/DiffuEraser.git third_party/DiffuEraser
python3 -m venv ~/venvs/diffueraser
~/venvs/diffueraser/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
~/venvs/diffueraser/bin/pip install diffusers==0.29.2 transformers==4.41.1 accelerate==0.25.0 \
    peft==0.13.2 einops==0.8.0 opencv-python==4.9.0.80 imageio==2.34.1 av==14.0.1 \
    scipy==1.13.1 numpy==1.26.4 matplotlib tqdm
```

가중치는 `third_party/DiffuEraser/weights/` 아래 `diffuEraser`, `stable-diffusion-v1-5`,
`PCM_Weights`, `sd-vae-ft-mse`, `propainter`.

RTX 5070(sm_120)이라 리포가 요구하는 torch 2.3.1은 못 쓴다. 2.11+cu128을 쓰면서 벤더 코드에
두 곳을 고쳐야 했다.

1. **`torchvision.io.read_video` 제거됨** (torchvision 0.24). `propainter/inference.py`와
   `diffueraser/diffueraser.py`의 호출을 PyAV 디코더로 교체.
2. **마지막에 전 프레임을 한 번에 latent 디코딩** → 574프레임에서 OOM.
   `diffueraser/pipeline_diffueraser.py`에서 `DIFFUERASER_DECODE_CHUNK`(기본 24)만큼 나눠
   디코딩하고 각 청크를 CPU로 내리도록 수정.

그리고 SD1.5 부분 다운로드 시 `feature_extractor/preprocessor_config.json`이 빠지기 쉬우니
따로 받아야 한다.

`src/inpainting/diffueraser_lowvram.py`가 이 저장소의 드라이버다. 원본 `run_diffueraser.py`는
디퓨전 모델을 먼저 올린 뒤 prior를 돌려 VRAM이 겹치므로, 단계를 분리하고 `--priori`로 기존
ProPainter 결과를 받는다.
