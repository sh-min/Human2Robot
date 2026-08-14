# 48번 — 팔에도 그림자 (높이 대역 접촉 그림자)

47번 위에서 그림자 계산만 바꿨다. 베이스·손 모델·플레이트·나머지 합성 설정은 47번과 같다.

원본 결과 폴더: `results/48_2026-08-14_팔그림자_높이대역`

## 결과

`48_팔그림자_높이대역.mp4` — 1280×720, 24 fps, 553프레임, H.264 High / yuv420p.
패킷 553개, 디코드 정상.

## 왜 팔 그림자가 없었나

`contact_shadow.py`가 로봇 표면 전체를 지지평면에 투영해 하나의 누적기에 쌓은 뒤
**최댓값으로 정규화**하고 있었다:

```python
acc = gaussian_filter(acc, blur)
acc /= acc.max() + 1e-6          # <- 여기
```

책상에 붙어 있는 손은 좁은 면적에 점이 몰려서 항상 이 최댓값을 차지한다. 반면 50 cm 위에
있는 팔은 같은 점 수라도 훨씬 넓게 퍼지므로 픽셀당 밀도가 낮고, 정규화 후에는 거의 0으로
깎여 나간다. 팔이 그림자를 안 만든 게 아니라 **손 그림자에 의해 정규화로 지워지고 있었다.**

## 높이 대역 방식

투영점을 평면까지의 거리 `t`로 잘라 5개 대역으로 나누고, 각 대역을 독립적인 차폐물로
합성한다. 대역마다:

- **반그림자**가 높이에 비례해 넓어진다 — `blur + 70 px/m × t`
- **진하기**가 높이에 따라 감쇠한다 — `exp(−t / 0.30 m)`
- 정규화를 대역 안에서만 하므로 손이 팔을 깎아내지 못한다

합성은 `alpha = 1 − Π(1 − alpha_b)`. 결과적으로 손이 닿는 곳은 진하고 좁게, 팔 아래는
옅고 넓게 깔린다.

`--shadow_bands 0`이면 예전 단일 패스 그대로다. 47번까지 쓰던 넓은 2패스
(`--shadow_soft_opacity`)는 대역 방식과 역할이 겹쳐서 껐다.

## 검증

| frame | 단일 패스 | 대역 5개 | 배수 |
|---|---:|---:|---:|
| 0 | 26,059 px | 35,414 px | 1.36 |
| 96 | 17,007 px | 38,374 px | 2.26 |
| 192 | 30,241 px | 44,914 px | 1.49 |
| 288 | 19,747 px | 46,474 px | 2.35 |
| 384 | 9,644 px | 11,468 px | 1.19 |
| 480 | 17,715 px | 23,503 px | 1.33 |

전 구간 평균 **20,439 px → 36,722 px (1.80배)**. 늘어난 몫이 팔에서 나온 옅은 그림자다.
팔이 화면을 크게 가로지르는 구간(f96, f288)에서 2.3배로 가장 크게 늘고, 손만 나오는
구간(f384)에서는 1.2배로 거의 그대로다 — 의도한 대로 높이에 반응한다.

## 파라미터

```
--shadow_opacity 0.50 --shadow_blur 5
--shadow_bands 5 --shadow_penumbra 70 --shadow_falloff 0.30
```

`대역수_비교.png`에 단일 / bands=5 / bands=6(penumbra 95, falloff 0.42)을 나란히 뒀다.
bands=6 쪽은 팔 그림자가 더 넓고 진한데, 책상 절반이 어두워져서 5개 쪽을 택했다.
더 진하게 원하면 `--shadow_falloff`를 올리면 된다(0.30 → 0.42).

`비교_47_vs_48.png`는 47번(단일 패스 + 소프트 2패스)과 48번을 같은 프레임에서 비교한 것이다.

## 재현

48번은 44번의 합성 명령에서 45–47번의 델타만 누적한 것이다. 앞 단계(HaWoR,
retargeting, 세그멘테이션, ProPainter 플레이트, 물체 복원)의 산출물은 그대로 재사용한다.

```bash
D=/home/rkd02/s2p/inpaint_test/processed/a001_08051547_c056/0
PY=/home/rkd02/venvs/inpaint/bin/python
PY_RT=/home/rkd02/venvs/retarget/bin/python

# 1) RB5-850 베이스 — 화면 밖 안쪽 고정 (46번)
$PY_RT -u src/inpainting/rb5_build_overlay_input.py \
  --hawor_npz $D/rgb_hawor_full/retarget_input.npz \
  --pkl $D/rgb_hawor_full/qpos_xhand_left_smooth.pkl --side left \
  --img_w 1280 --img_h 720 --base_override $D/base_inward.npy \
  --smooth_win 5 --snap_flange --out $D/rb5_in2.npz

# 2) XHAND1 손 + RB5 팔 렌더 → overlay_rb5_x3_cut
PYOPENGL_PLATFORM=egl $PY -u src/inpainting/render_xhand_overlay_depth.py \
  --processed_demo $D --hawor_npz $D/rgb_hawor_full/retarget_input.npz \
  --right_pkl $D/rgb_hawor/qpos_xhand_right.pkl \
  --left_pkl $D/rgb_hawor_full/qpos_xhand_left_smooth.pkl \
  --hand left --arm rb5 --rb5_npz $D/rb5_in2.npz \
  --left_embodiment xhand1 --output_subdir overlay_rb5_x3

# 3) 사람 그림자 제거한 배경 플레이트 (47번)
$PY -u src/inpainting/remove_cast_shadow.py \
  --plate $D/inpaint_processor/plate_skinfix_24.mkv \
  --object_mask $D/interaction_objects/refined/object_mask_refined.npy \
  --robot_mask $D/overlay_rb5_x3_cut/robot_mask.npy \
  --output $D/inpaint_processor/plate_noshadow.mkv

# 4) 합성 — 47번에서 그림자 인자만 다름
$PY -u src/inpainting/composite_interaction_objects.py --processed_demo $D \
  --hawor_npz $D/rgb_hawor_full/retarget_input.npz \
  --object_source_video interaction_objects/objsrc_final.mkv \
  --background_video inpaint_processor/plate_noshadow.mkv \
  --object_mask interaction_objects/refined/object_mask_refined.npy \
  --robot_dir overlay_rb5_x3_cut \
  --force_front_mask interaction_objects/refined/force_front_final.npy \
  --force_robot_front_mask overlay_processor_hawor_reference/robot_thumb_mask.npy \
  --force_robot_front_dilate 2 \
  --behind_robot_object_mask interaction_objects/refined/behind_robot_objects_v40.npy \
  --shadow_depth depth_processor/depth_aligned.npy \
  --shadow_opacity 0.50 --shadow_blur 5 \
  --shadow_bands 5 --shadow_penumbra 70 --shadow_falloff 0.30 \
  --video_codec h264 --output 48_팔그림자_높이대역.mp4
```

`--shadow_depth`가 가리키는 `depth_processor/depth_aligned.npy`는 `align_depth.py`가
만든 metric depth다. 이 데이터에서는 `hand_data_*.npz`의 `kpts_3d`와 렌더러가 쓰는
joint의 스케일이 약 3배 달라서 `--hawor_npz`를 주고 다시 정렬해야 한다. 안 그러면
장면이 1.40 m로 잡히는데 로봇은 0.43 m에 렌더되어 그림자 alpha가 전 프레임 0이 된다.

```bash
$PY -u src/inpainting/align_depth.py --processed_demo $D \
  --hawor_npz $D/rgb_hawor_full/retarget_input.npz
```

## 관련 코드

| 파일 | 역할 |
|---|---|
| `src/inpainting/contact_shadow.py` | 지지평면 RANSAC + 높이 대역 접촉 그림자 |
| `src/inpainting/composite_interaction_objects.py` | 6단계 합성 진입점, `--shadow_*` 인자 |
| `src/inpainting/layered_compositor/stages.py` | stage 1에서 플레이트에 `(1 - shadow_alpha)` 곱 |
| `src/inpainting/layered_compositor/video.py` | `CompatibleVideoWriter` — ffmpeg 파이프 1회 인코딩 |
| `src/inpainting/align_depth.py` | `--hawor_npz` 전역 스케일 정렬 |
| `src/inpainting/remove_cast_shadow.py` | 배경 플레이트에서 사람 그림자 제거 |
