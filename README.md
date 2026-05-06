# skill2policy

## Cloning

This repository uses git submodules. A plain `git clone` will leave the `third_party/` directories empty.

**First-time clone:**

```bash
git clone --recurse-submodules https://github.com/<your-org>/skill2policy.git
```

**Already cloned without `--recurse-submodules`:**

```bash
git submodule update --init --recursive
```

### Third-party dependencies (`third_party/`)

| Directory | Repository |
|---|---|
| `vjepa2` | https://github.com/facebookresearch/vjepa2 |
| `HACO_RELEASE` | https://github.com/dqj5182/HACO_RELEASE |
| `HaWoR` | https://github.com/ThunderVVV/HaWoR |
| `dex-retargeting` | https://github.com/dexsuite/dex-retargeting |

### Updating submodules

To pull the latest commits from all submodules:

```bash
git submodule update --remote --recursive
```

---

## ⚠️ Required Downloads (NOT in git)

라이선스 / 용량 문제로 다음 파일들은 git에 들어있지 않음. 코드를 돌리려면 직접 다운받아 정해진 위치에 둬야 함.

### MANO hand model (라이선스 등록 필요)
[MANO 공식 사이트](https://mano.is.tue.mpg.de) 가입 → `mano_v1_2.zip` 다운 → 압축 풀어서:

```
third_party/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl
third_party/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl
```

> `MANO_LEFT.pkl`은 일반적으로 `MANO_RIGHT.pkl`을 복사하고 사용 (smplx가 left hand의 shapedirs bug를 자동 fix). HaWoR README의 안내를 따르는 것이 안전.

### HaWoR / detector / SLAM 가중치 (≈ 3.5 GB)
[HaWoR README](third_party/HaWoR/README.md)의 "Pretrained Weights" 절 참고.

```
third_party/HaWoR/weights/hawor/checkpoints/hawor.ckpt
third_party/HaWoR/weights/hawor/checkpoints/infiller.pt
third_party/HaWoR/weights/external/droid.pth
third_party/HaWoR/weights/external/detector.pt
third_party/HaWoR/_DATA/data/mano_mean_params.npz
```

### 모듈별 사용법

각 모듈 폴더의 README 참고:
- [`src/hand_estimation/README.md`](src/hand_estimation/README.md) — RGB → MANO
- [`src/retargeting/README.md`](src/retargeting/README.md) — MANO → xhand qpos + 시각화