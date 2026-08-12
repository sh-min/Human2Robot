# 6-stage layered compositor

`layered_compositor`는 사람 손을 로봇 손으로 치환할 때 필요한 앞뒤 관계를 여섯
개의 누적 스테이지로 분리한다. 영상 입출력과 픽셀 합성 규칙을 분리했기 때문에
한 스테이지만 교체하거나 합성 전후 배열을 단위 테스트할 수 있다.

## 레이어별 의미와 개선 지점

| 단계 | 모듈 함수 | 의미 | 주요 입력 | 이 단계에서 확인할 결함 | 독립 개선 방법 |
|---|---|---|---|---|---|
| 1. Background | `stage_1_background` | 사람 손·팔을 지운 장면 바탕이다. 이후 레이어가 모두 이 영상 위에 쌓인다. | inpainted background | 손가락/피부 잔존, 팔 모양의 흐림, 테이블 선 왜곡 | 사람 마스크 확장, 접촉부 마스크 보강, ProPainter/E2FGVI 교체, temporal consistency 개선 |
| 2. Robot behind | `stage_2_robot_behind` | 기준 깊이보다 뒤에 있는 로봇 픽셀이다. 물체가 다음 단계에서 이 레이어를 덮는다. | robot RGB/depth/mask, split depth | 손가락 조각이 잘못 앞/뒤로 이동, 프레임별 깜빡임 | depth 정합, `depth_bias`, `threshold_joint`, `depth_sigma` 조정 또는 손가락별 분류기 적용 |
| 3. Object | `stage_3_object` | 사람 손에 가렸던 부분까지 복원한 조작 물체를 원본 RGB 계열에서 다시 올린다. | completed object RGB, refined object mask, behind-robot object mask | 손 피부 혼입, 물체 구멍, 움직임 잔상, 경계 손실, 손이 지나가기만 하는 정적 물체가 로봇을 관통 | modal/amodal mask 개선, 물체별 texture completion, `object_edge_sigma` 조정, `behind_robot_object_mask`로 비조작 정적 물체 제외 |
| 4. Robot front | `stage_4_robot_front` | 일반 깊이 판정상 물체 앞에 오는 로봇 픽셀이다. 강제 엄지는 여기서 제외된다. | robot RGB, ordinary-front mask | 검은 halo, 손가락 관통, 경계 떨림 | `robot_edge_mode`, `robot_edge_sigma`, 렌더 품질과 per-finger depth 개선 |
| 5. Object forced front | `stage_5_forced_object` | 컵·상자 같은 단단한 물체가 네 개의 말린 손가락을 반드시 덮어야 하는 영역이다. | grasp-specific force-front mask | 로봇 손가락이 물체 내부를 관통하거나, 반대로 물체가 엄지를 덮음 | 물체/파지별 force mask를 보수적으로 조정, `forced_object_edge_sigma` 조정 |
| 6. Robot forced front | `stage_6_forced_robot_front` | 의미론적으로 지정한 로봇 부품을 마지막에 올린다. 현재는 XHand 엄지이며 최종 결과다. | rendered semantic thumb mask | 엄지가 상자 뒤로 넘어감, 엄지 가장자리 틈 | thumb-link 렌더·depth agreement 개선, `forced_robot_front_dilate` 1–2 px 조정 |

## 반드시 유지하는 합성 불변 조건

- 6단계 강제 로봇 마스크는 2단계와 4단계 로봇 마스크에서 제외한다.
- 6단계 강제 로봇 마스크는 5단계 강제 물체 마스크에서도 제외한다.
- 따라서 네 손가락은 물체 뒤에 둘 수 있지만 엄지는 항상 마지막에 그릴 수 있다.
- `behind_robot_object_mask`로 표시한 물체 픽셀은 3단계와 5단계 물체 마스크에서
  모두 제외한다. 프레임당 하나뿐인 깊이 분할면은 파지 중인 물체를 기준으로
  잡히므로, 손이 위를 지나가기만 하는 정적 물체는 그 분할면으로 표현할 수 없다.
- 로봇 경계 alpha는 유효한 robot raster 밖으로 나가지 않아 검은 halo를 만들지 않는다.
- 각 스테이지는 누적 영상을 반환하므로 어느 단계에서 결함이 들어왔는지 바로 비교할 수 있다.

## 코드 구조

```text
layered_compositor/
├── models.py         # FrameInputs, LayerMasks, StageConfig, CompositeFrame
├── stages.py         # 마스크 분류 + 독립된 6개 pure stage 함수
├── visualization.py  # isolated/context/3×2 진단 영상 표현
└── video.py          # H.264/yuv420p/faststart 호환 출력
```

`compose_frame()`은 편의를 위한 기본 오케스트레이터일 뿐이다. 실험할 때는
`build_layer_masks()` 이후 원하는 `stage_N_*()` 함수만 다른 구현으로 바꿔 호출할
수 있다. 입력과 설정은 dataclass이므로 실험 간 차이도 명시적으로 남는다.

## CLI 출력

`composite_interaction_objects.py`는 이 모듈을 사용한다.

```bash
python src/inpainting/composite_interaction_objects.py \
  --processed_demo <processed-demo> \
  --hawor_npz <retarget_input.npz> \
  --background_video <human-free-background> \
  --object_mask <refined-object-mask.npy> \
  --force_front_mask <object-force-front.npy> \
  --force_robot_front_mask <robot-thumb-mask.npy> \
  --force_robot_front_dilate 2 \
  --behind_robot_object_mask <static-object-behind-robot.npy> \
  --layer_output_dir <isolated-layers> \
  --layer_context_videos \
  --progressive_output_dir <cumulative-stages> \
  --video_codec h264 \
  --output <final.mp4>
```

`--video_codec h264`가 기본값이다. OpenCV의 `mp4v` 중간 파일을 만든 다음
ffmpeg/libx264로 변환하고 원자적으로 게시하므로 VS Code와 브라우저에서 바로
미리 볼 수 있다. 빠른 내부 디버깅에서 변환이 필요 없을 때만 `mp4v`를 선택한다.

## 테스트

```bash
python -m unittest tests.inpainting.test_layered_compositor -v
```

테스트는 스테이지 순서, 마스크 상호 배타성, 엄지 최종 우선권, H.264 출력 코덱을
검증한다.
