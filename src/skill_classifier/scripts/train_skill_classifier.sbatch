#!/bin/bash
#SBATCH --job-name=skill_cls
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --partition=ljw_rvlab
#SBATCH --output=out/skill_cls_%A_%a.out
#SBATCH --error=out/skill_cls_%A_%a.err
#SBATCH --comment "skill classifier training"

# Usage:
#   sbatch --export=MODEL=transformer,VARIANT=mano_only,W=8 train_skill_classifier.sbatch
#
# MODEL:   mlp | transformer
# VARIANT: mano_only | vjepa_orig | masked_vjepa_orig
# W:       8 | 16 | 32
# VJEPA_DIFF: false (default) | true
#
# Reads per-recording bundled features from
#   data/kitchen_dataset/0412_train/saved_frames_*/features.pt        (train)
#   data/kitchen_dataset/0412_val/episode_*/features.pt               (val)
#
# Array usage:
#   sbatch --array=0-2 --export=MODEL=transformer,VARIANT=vjepa_orig train_skill_classifier.sbatch

PROJ_DIR=/virtual_lab/ljw_rvlab/byeonggyeol/3dgs-visual-grounding/skill2policy
cd $PROJ_DIR
mkdir -p out

source /virtual_lab/ljw_rvlab/byeonggyeol/miniconda3/etc/profile.d/conda.sh
conda activate vjepa2-312

# Defaults
: ${MODEL:=transformer}
: ${VARIANT:=mano_only}
: ${VJEPA_DIFF:=false}

WINDOWS=(8 16 32)
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    W=${WINDOWS[$SLURM_ARRAY_TASK_ID]}
else
    : ${W:=8}
fi

CFG=src/skill_classifier/config/${MODEL}.yaml
if [ ! -f "$CFG" ]; then
    echo "ERROR: config not found: $CFG"; exit 1
fi

case $VARIANT in
    mano_only|vjepa_orig|masked_vjepa_orig) ;;
    *) echo "ERROR: unknown VARIANT=$VARIANT"; exit 1 ;;
esac

echo "=== $MODEL / $VARIANT / window_size=$W / vjepa_diff=$VJEPA_DIFF ==="
PYTHONPATH=$PROJ_DIR/src python -m skill_classifier.train \
    --config $CFG \
    --set window_size=$W \
          variant=$VARIANT \
          vjepa_diff=$VJEPA_DIFF
