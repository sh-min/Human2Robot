# V-JEPA skill training

This branch contains the reproducible V-JEPA 2/2.1 skill-classification
pipeline used for the restored kitchen robot-overlay videos. Generated data,
features, checkpoints, and presentation media remain under ignored `data/`,
`weights/`, and `output/` directories.

## Selected experiment

The current best single-seed run uses:

- frozen V-JEPA 2.1 ViT-L/384 dense patch features;
- learned spatial attention over 24 x 24 patch tokens;
- an eight-token temporal window;
- train-only, temporally consistent Color Jitter;
- an MLP classifier with Dropout 0.4.

On the nine-recording held-out split (247 labelled tokens), the selected epoch
255 checkpoint reached 80.57% accuracy, 73.20% Macro F1, and 80.51% weighted
F1. This is a seed-42 result, not a multi-seed confidence estimate.

## Feature extraction

Initialize the V-JEPA submodule and place the official checkpoints under
`weights/`. The V-JEPA 2 download scripts validate both file integrity and the
checkpoint state-dict contract.

```bash
git submodule update --init third_party/vjepa2
bash scripts/download_vjepa2_vitl.sh
```

Build V-JEPA 2.1 dense features with Meta's evaluation crop and the exact
4-FPS temporal alignment:

```bash
PYTHONPATH=$PWD/src python -m data_preprocess.preprocess \
  --data_root data/vjepa_training/kitchen_dense \
  --recording_glob '*' \
  --checkpoint weights/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt \
  --backbone vjepa2_1_vitl384 \
  --crop_size 384 \
  --num_frames 16 \
  --sampling_profile vjepa2_4fps \
  --sample_fps 4 \
  --spatial_profile vjepa2_eval_center_crop \
  --store_dense_tokens \
  --allow_missing_mano \
  --batch_size 1
```

For one additional training view per recording, create train-only recording
directories with a `__cj1` suffix that point to the same RGB and annotation
inputs, then extract their features with:

```bash
PYTHONPATH=$PWD/src python -m data_preprocess.preprocess \
  --data_root data/vjepa_training/kitchen_dense_color_jitter \
  --recording_glob '*__cj1' \
  --checkpoint weights/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt \
  --backbone vjepa2_1_vitl384 \
  --crop_size 384 \
  --sampling_profile vjepa2_4fps \
  --sample_fps 4 \
  --store_dense_tokens \
  --allow_missing_mano \
  --color_jitter_brightness 0.2 \
  --color_jitter_contrast 0.2 \
  --color_jitter_saturation 0.2 \
  --color_jitter_hue 0.05 \
  --color_jitter_seed 142 \
  --batch_size 1
```

One jitter transform is sampled deterministically per recording and reused for
all of its frames, preventing augmentation-induced temporal flicker.

## Classifier training

The checked-in configuration keeps the validation recordings unaugmented.
Use an explicit experiment ID for each Dropout run:

```bash
PYTHONPATH=$PWD/src python -m skill_classifier.train \
  --config src/skill_classifier/config/kitchen_0724_0728_robot_overlay_vjepa21_spatial_attention_color_jitter_no_choco.yaml \
  --exp_id dropout_04_seed_42 \
  --set dropout=0.4
```

The training command saves the resolved configuration, learning history,
checkpoints, best-checkpoint confusion matrix, and evaluation summary under
the configured `output_dir`.

## Presentation and attention reports

```bash
PYTHONPATH=$PWD/src python src/skill_classifier/build_presentation_visuals.py \
  --experiment_dir output/skill_classifier/EXPERIMENT/dropout_04_seed_42 \
  --output_dir output/skill_classifier/vjepa21_d04_presentation

PYTHONPATH=$PWD/src python src/skill_classifier/visualize_spatial_attention_video.py \
  --experiment_dir output/skill_classifier/EXPERIMENT/dropout_04_seed_42 \
  --output_dir output/skill_classifier/vjepa21_d04_presentation/attention_videos
```

The video renderer preserves each full source sequence, places the ground-truth
label and prediction in a separate top-left header, and linearly interpolates
attention between adjacent V-JEPA token centers.

## Tests

```bash
PYTHONPATH=$PWD/src python -m unittest \
  tests.test_vjepa_preprocess \
  tests.test_vjepa_feature_extractor \
  tests.test_skill_dataset_sampling \
  tests.test_spatial_attention_mlp \
  tests.test_skill_classifier_labels
```

