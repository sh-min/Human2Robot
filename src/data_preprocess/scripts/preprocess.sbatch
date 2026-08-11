#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --partition=ljw_rvlab
#SBATCH --output=out/preprocess_%j.out
#SBATCH --error=out/preprocess_%j.err
#SBATCH --comment "Per-recording preprocess: V-JEPA + MANO + labels → features.pt"

# Generates {recording}/features.pt for both train and val splits.
# Skips recordings that already have features.pt (use --overwrite to redo).

PROJ_DIR=/virtual_lab/ljw_rvlab/byeonggyeol/3dgs-visual-grounding/skill2policy
cd $PROJ_DIR
mkdir -p out

source /virtual_lab/ljw_rvlab/byeonggyeol/miniconda3/etc/profile.d/conda.sh
conda activate vjepa2-312

CKPT=ckpt/v-jepa2/vitl.pt

echo "=== Train (saved_frames_*) ==="
PYTHONPATH=$PROJ_DIR/src python -m data_preprocess.preprocess \
    --data_root data/cube_dataset/0412_train \
    --recording_glob "saved_frames_*" \
    --checkpoint $CKPT

echo "=== Val (episode_*) ==="
PYTHONPATH=$PROJ_DIR/src python -m data_preprocess.preprocess \
    --data_root data/cube_dataset/0412_val \
    --recording_glob "episode_*" \
    --checkpoint $CKPT

echo "=== Done ==="
