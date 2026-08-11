# skill_classifier

Per-frame skill classifier for cube manipulation videos. Trains a temporal
classifier (Transformer or MLP) over a sliding window of features to predict
the active skill at each frame.

## Inputs

Per-recording bundled features at `{recording}/features.pt`, produced by
[`../data_preprocess/preprocess.py`](../data_preprocess/preprocess.py)
(SLURM: `sbatch src/data_preprocess/scripts/preprocess.sbatch`):

```
{
    "vjepa_orig":        [T, 1024],   # V-JEPA features (orig ckpt)
    "vjepa_orig_masked": [T, 1024],   # optional, from masked frames
    "mano":              [T, 96],     # MANO axis-angle, downsampled to token rate
    "labels_per_token":  [T],         # int label, -1 if no GT
    "num_frames", "num_tokens", "recording",
}
```

`T = num_frames // 2` (V-JEPA tubelet=2).

## Variants

Choose which inputs feed the classifier via the `variant` config field:

| variant              | V-JEPA channel       | hand input |
|----------------------|----------------------|------------|
| `mano_only`          | (none)               | MANO       |
| `vjepa_orig`         | `vjepa_orig`         | MANO       |
| `masked_vjepa_orig`  | `vjepa_orig_masked`  | MANO       |

## Layout

```
skill_classifier/
├── skill_dataset.py        # SkillWindowDataset (sliding-window over recordings)
├── train.py                # Training loop
├── infer_long_horizon.py   # Long-horizon inference (re-extracts V-JEPA on-the-fly)
├── config/
│   ├── transformer.yaml
│   └── mlp.yaml
├── models/
│   ├── transformer.py
│   ├── mlp.py
│   └── ...
└── scripts/
    ├── train_skill_classifier.sbatch  # SLURM driver for train.py
    └── collect_results.py             # aggregate experiments + long-horizon results to CSV
```

## Training

```bash
cd skill2policy
PYTHONPATH=$PWD/src python -m skill_classifier.train \
    --config src/skill_classifier/config/transformer.yaml \
    --set window_size=8 variant=vjepa_orig
```

Or via SLURM:

```bash
sbatch --export=MODEL=transformer,VARIANT=vjepa_orig,W=8 \
    src/skill_classifier/scripts/train_skill_classifier.sbatch
```

Outputs go to `output/skill_classifier/{exp_id}/best_{model}.pt`.

## Long-horizon inference

```bash
PYTHONPATH=$PWD/src python -m skill_classifier.infer_long_horizon \
    --data_dir data/cube_dataset/0412_val \
    --vjepa_ckpt ckpt/v-jepa2/vitl.pt \
    --classifier_ckpt output/skill_classifier/{exp_id}/best_transformer.pt \
    --output_dir output/long_horizon/{exp_id}
```

Per-episode outputs:
- `predictions.pt` — frame-level preds + probs
- `eval_results.json` — overall + per-class accuracy
- `timeline.png` — predicted skill timeline vs GT
- `{exp_id}_{episode}.mp4` — annotated video (omit `--no_video` to generate)

## Labels

6 classes from [`utils/labels.py`](../utils/labels.py):
`Cup`, `Lock`, `Milk`, `Snack`, `Sweep`, and `Trans`.
