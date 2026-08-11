# T-Rex-style HaCo visibility adapter

이 실험은 공식 T-Rex 체크포인트를 Human2Robot에 직접 실행하는 구성이 아닙니다.
공식 T-Rex는 전용 로봇의 열 개 촉각 영상과 손가락별 6축 wrench 이력을 입력으로
사용하지만, 현재 데이터에는 RGB, HaWoR/MANO 및 HaCo 접촉 확률만 있습니다.

대신 논문의 비동기 slow/fast 구조를 차폐 판단에 다음처럼 대응시켰습니다.

| T-Rex 개념 | Human2Robot 어댑터 |
| --- | --- |
| slow vision expert | MANO 앵커와 객체 차폐 비율을 저주기로 계산하고 유지 |
| 16-frame tactile history | HaCo 손가락별 접촉 확률의 과거 16프레임 통계 |
| fast tactile refinement | 매 프레임 새 HaCo 신호로 손가락별 차폐 비율 보정 |
| cascaded output | 기존 차폐 마스크를 보존하며 객체 지지 영역 안에서만 확장 |

모든 시간 통계는 현재와 과거 프레임만 사용하는 causal 계산입니다. 빠른 보정은
기존에 가려진 로봇 픽셀을 다시 보이게 하지 않으며, 새로 가리는 픽셀에는 현재
SAM2 객체 마스크가 있어야 합니다.

## 실행 예시

```bash
EP=/path/to/Hand
PD="$EP/inpainting_processed/Hand/0"

python scripts/refine_dense_visibility_trex_temporal.py \
  --clean_plate "$PD/object_completion_temporal_attention/video_object_completed.mp4" \
  --baseline_hidden_mask "$PD/dense_mano_anchor_xhand_visibility_object_override/dense_mapped_invisible_robot_mask.npy" \
  --dense_anchor_counts "$PD/dense_mano_anchor_xhand_visibility_object_override/dense_anchor_counts.npy" \
  --robot_rgb "$PD/overlay_processor_haco_stabilized/robot_rgb.npy" \
  --robot_mask "$PD/overlay_processor_haco_stabilized/robot_mask.npy" \
  --robot_finger_labels "$PD/overlay_processor_haco_stabilized/robot_finger_labels.npy" \
  --object_mask "$PD/sam2_contact_selected_objects/object_mask_modal.npy" \
  --contact_dir "$EP/contact" \
  --finger_parts src/retargeting/assets/finger_part_left.npy \
  --reference_hidden_mask "$PD/multiframe_dense_visibility_refinement/multiframe_refined_hidden_mask.npy" \
  --output_dir "$PD/trex_haco_slow_fast_v2" \
  --side left
```

주요 출력은 최종 영상 `video_trex_haco_slow_fast.mp4`, 3분할 비교 영상
`video_compare_visibility_methods.mp4`, 최종 차폐 마스크와 slow/fast 특징 파일,
그리고 공식 모델 사용 여부와 불변 조건을 기록한 `report.json`입니다.

## 참고 자료

- [T-Rex 논문](https://arxiv.org/abs/2606.17055)
- [공식 T-Rex 구현](https://github.com/ZhuoyangLiu2005/T-Rex)
