# 서브모듈 로컬 패치

`third_party/` 아래 서브모듈은 업스트림 저장소를 그대로 가리키므로, 이 GPU에서
파이프라인을 돌리려고 손본 부분은 서브모듈 커밋으로 남길 수 없다. 그 수정을
여기에 패치로 보관한다. 체크아웃 직후 아래 순서대로 적용하면 결과영상을 만든
코드와 같은 상태가 된다.

| 패치 | 서브모듈 | 기준 커밋 | 내용 |
|---|---|---|---|
| `E2FGVI.patch` | `third_party/E2FGVI` | `709cbe3` | mmcv-full 없이 추론. SPyNet 초기화 체크포인트 다운로드를 건너뛴다 — 배포 체크포인트에 이미 전체 state가 들어 있다 |
| `HaWoR.patch` | `third_party/HaWoR` | `66c7d41` | PyTorch3D 스텁 환경에서도 MANO 추출이 돌게. 렌더링이 필요한 손 마스크 캐시만 건너뛴다 |
| `DiffuEraser.patch` | `third_party/DiffuEraser` | `8e6f279` | 12GB 카드용. ProPainter prior를 먼저 단독으로 돌리고 모델을 내린 뒤 디퓨전을 올린다. 새 파일 `run_diffueraser_lowvram.py`는 이미 있는 ProPainter 결과를 `--priori`로 받아 재계산을 건너뛴다 |

```sh
for m in E2FGVI HaWoR DiffuEraser; do
  git -C "third_party/$m" apply "../../patches/$m.patch"
done
```

`third_party/DiffuEraser`는 서브모듈이지만 업스트림에 `.gitmodules` 등록이
늦었다 — 클론 후 `git submodule update --init third_party/DiffuEraser`가 필요하다.
