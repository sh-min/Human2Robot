#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA="${DATA:-$ROOT/data/kitchen_dataset/26.08.04_stereo}"
TRAIN_ENV="${TRAIN_ENV:-vjepa2-312}"
CONFIG="${CONFIG:-$ROOT/src/skill_classifier/config/0804_vjepa_mano_mlp.yaml}"

cd "$ROOT"
for ID in {1..24}; do
    FEATURE="$DATA/$ID/features.pt"
    if [ ! -s "$FEATURE" ]; then
        echo "Missing aligned feature bundle: $FEATURE" >&2
        exit 1
    fi
done

PYTHONPATH="$ROOT/src" MPLCONFIGDIR=/tmp/skill-classifier-mpl \
    conda run -n "$TRAIN_ENV" --no-capture-output \
    python -m skill_classifier.train --config "$CONFIG" "$@"
