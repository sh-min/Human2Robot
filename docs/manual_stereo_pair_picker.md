# 수동 스테레오 캘리브레이션 프레임 매칭

두 카메라 영상을 독립적으로 탐색하고, 같은 순간이라고 판단한 프레임을 캘리브레이션 입력 사진으로 저장하는 로컬 도구입니다.

```bash
bash scripts/run_manual_stereo_pair_picker.sh
```

기본 주소는 `http://127.0.0.1:8012`입니다. 화면 상단에서 카메라 1 영상, 카메라 2 영상, 저장 루트 경로를 바꿀 수 있습니다.

저장 결과는 다음 규칙을 사용합니다.

```text
<저장 루트>/camera_1/1_Color.png
<저장 루트>/camera_2/1_Color.png
<저장 루트>/pairs.json
```

양쪽 파일명은 항상 같으며 `src/calibration/calibrate_stereo_checkerboard.py` 입력 규칙과 호환됩니다. `pairs.json`에는 각 이미지의 원본 영상 경로, 프레임 번호, 시간값이 기록됩니다.

주요 단축키:

- `A` / `D`: 카메라 1 이전/다음 프레임
- `J` / `L`: 카메라 2 이전/다음 프레임
- `Shift`와 함께 입력: 10프레임 이동
- `Space`: 양쪽 재생/정지
- `Ctrl+S` 또는 `Cmd+S`: 현재 프레임 쌍 저장

CLI에서 시작 경로를 직접 지정할 수도 있습니다.

```bash
bash scripts/run_manual_stereo_pair_picker.sh \
  --camera1 /path/to/camera1.mov \
  --camera2 /path/to/camera2.mov \
  --output /path/to/calibration_images \
  --port 8012
```
