# skill2policy

## LLM pair-coding tools

This repo is built assuming you'll be pair-coding with an LLM (Claude Code or
similar) and ships with two helpers. Skim them first when adding a new module
or trying to understand code a teammate wrote.

### 1. [`CLAUDE.md`](CLAUDE.md) — coding guidelines

Four behavioral rules in the spirit of Andrej Karpathy's coding-with-LLM notes:
*Think before coding* / *Simplicity first* / *Surgical changes* /
*Goal-driven execution*. Claude Code reads this file automatically at the start
of each session. You shouldn't normally need to touch it — append project-
specific rules below the existing ones if you want to extend it.

### 2. `graphify` — codebase knowledge graph

Turns the whole `src/` tree into a node/edge/community graph so an LLM can
quickly grasp how modules connect. Useful when comparing structure before and
after merging a new module, or finding the seam between your code and a
collaborator's.

**Install (once):**

```bash
pip install graphifyy
graphify install --platform claude   # registers the /graphify slash command
```

**Build the graph** (`src/` only — skip `third_party/`):

```bash
# Inside Claude Code
/graphify src

# Or directly from the shell
graphify build src
```

Outputs:
- `GRAPH_REPORT.md` — god nodes, surprising connections, suggested questions
  (the human-readable summary; read this first)
- `graphify-out/graph.html` — interactive visualization (open in a browser)
- `graphify-out/graph.json` — GraphRAG-ready index for LLM queries

**Query / explore:**

```bash
graphify query "where is features.pt produced and where is it consumed?"
graphify path "VJEPAFeatureExtractor" "SkillWindowDataset"
graphify explain "infer_long_horizon.main"
```

**After refactoring or adding a new module:**

```bash
graphify update      # re-extracts only changed files (uses cache)
```

Re-running this once after dropping in a new folder (e.g. `src/policy/`,
`src/sim/`) lets the LLM lock onto the overall structure faster in later
sessions, with less context burned on rediscovery.

---

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

| Directory | Repository | 용도 |
|---|---|---|
| `vjepa2` | https://github.com/facebookresearch/vjepa2 | V-JEPA 2 video feature backbone |
| `HACO_RELEASE` | https://github.com/dqj5182/HACO_RELEASE | hand-contact estimation |
| `HaWoR` | https://github.com/ThunderVVV/HaWoR | RGB → MANO hand pose |
| `dex-retargeting` | https://github.com/dexsuite/dex-retargeting | MANO → xhand qpos retargeting |
| `lerobot` | https://github.com/huggingface/lerobot | robot policy training infra |
| `sam2` | https://github.com/facebookresearch/sam2 | hand/arm segmentation for inpainting |
| `E2FGVI` | https://github.com/MCG-NKU/E2FGVI | flow-guided video inpainting |

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
- [`src/data_preprocess/README.md`](src/data_preprocess/README.md) — 원본 데이터 정리 / 프레임 추출
- [`src/hand_estimation/README.md`](src/hand_estimation/README.md) — RGB → MANO
- [`src/contact_estimation/README.md`](src/contact_estimation/README.md) — hand contact 추정
- [`src/retargeting/README.md`](src/retargeting/README.md) — MANO → xhand qpos + 시각화
- [`src/inpainting/README.md`](src/inpainting/README.md) — 사람 손/팔 inpaint + xhand 오버레이 합성
- [`src/skill_classifier/README.md`](src/skill_classifier/README.md) — V-JEPA 2 기반 skill 분류