# data_preprocess

End-to-end preprocessing per recording. Takes raw frames and produces a single
bundled `features.pt` next to each recording, consumed by
[`skill_classifier`](../skill_classifier).

Steps:

1. **Hand pose extraction** — run [HaWoR](https://github.com/ThunderVVV/HaWoR)
   on `rgb/` → `result.json` *(friend's contribution, TBD)*
2. **V-JEPA features** — run V-JEPA encoder on `rgb/` (and optionally on
   `hand_object_mask_overlayed/`)
3. **MANO axis-angle** — convert HaWoR rotation matrices to axis-angle
4. **Per-token labels** — read `gt_labels.json` segments
5. **Bundle** all of the above into `features.pt`

Steps 2-5 are implemented in [`preprocess.py`](preprocess.py).

## Inputs (per recording dir)

```
{recording}/
├── rgb/                          # PNG frames (required)
├── result.json                   # HaWoR output (required, produced by step 1)
├── hand_object_mask_overlayed/     # masked frames for V-JEPA-masked variant (optional)
└── gt_labels.json                # skill segments (optional)
```

### `result.json` schema (HaWoR output)

Per-frame keys (`rgb_frame00000`, ...) → list of detected hands:

```
{
  "rgb_frame00000": [
    {
      "is_right":    0 | 1,
      "kpts_3d":     [21, 3],
      "mano_params": {
        "global_orient": [1, 3, 3],   # rotation matrix
        "hand_pose":     [15, 3, 3],  # 15 finger-joint rotation matrices
      },
      ...
    },
    ...
  ],
  ...
}
```

`preprocess.py` converts these rotation matrices to axis-angle (3-dim) via
Rodrigues' formula → 48 dims per hand × 2 hands = 96 dims per frame, then
downsamples to token rate.

## Output

```
{recording}/features.pt
    vjepa_orig:        [T, 1024]    V-JEPA on rgb/
    vjepa_orig_masked: [T, 1024]    V-JEPA on hand_object_mask_overlayed/  (optional)
    mano:              [T, 96]      MANO axis-angle, token-rate
    labels_per_token:  [T]          int skill label per token, -1 if no GT
    num_frames, num_tokens, recording
```

`T = num_frames // 2` (V-JEPA tubelet=2).

## Layout

```
data_preprocess/
├── preprocess.py            # V-JEPA + MANO + labels → features.pt (steps 2-5)
├── feature_extractor.py     # V-JEPA encoder loader + spatial-mean pooler
├── extract_pose.py          # HaWoR hand pose extraction (TBD, friend's part)
└── scripts/
    └── preprocess.sbatch    # SLURM driver (train + val splits)
```

## Run

Single command (all 0412 train + val):

```bash
sbatch src/data_preprocess/scripts/preprocess.sbatch
```

Or directly:

```bash
cd skill2policy
PYTHONPATH=$PWD/src python -m data_preprocess.preprocess \
    --data_root data/kitchen_dataset/0412_train \
    --recording_glob "saved_frames_*" \
    --checkpoint ckpt/v-jepa2/vitl.pt
```

Skips recordings that already have `features.pt`. Pass `--overwrite` to redo.

## Adding new datasets

Drop new recordings under `data/kitchen_dataset/<split>/<recording_name>/` with
the input layout above (run HaWoR first to generate `result.json`), then run
preprocess. The resulting `features.pt` plugs directly into
[`skill_classifier`](../skill_classifier).
