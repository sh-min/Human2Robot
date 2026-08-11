#!/usr/bin/env bash
set -euo pipefail

# Download the raw Meta V-JEPA 2 ViT-L/16 (256 px) pretraining checkpoint.
# The feature loader consumes its EMA `target_encoder`; Hugging Face model
# directories and V-JEPA 2.1 checkpoints have a different contract.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
URL="${VJEPA_URL:-https://dl.fbaipublicfiles.com/vjepa2/vitl.pt}"
DEST="${VJEPA_CKPT:-$ROOT/weights/vjepa2/vitl.pt}"
EXPECTED_BYTES="${VJEPA_EXPECTED_BYTES:-5127726842}"
EXPECTED_SHA256="${VJEPA_EXPECTED_SHA256:-5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012}"
VJEPA_ENV="${VJEPA_ENV:-vjepa2-312}"
PART="${DEST}.part"

validate_checkpoint() {
    local checkpoint="$1"
    PYTHONPATH="$ROOT/src" conda run -n "$VJEPA_ENV" --no-capture-output \
        python -c '
import sys
from data_preprocess.feature_extractor import load_pretrained_encoder
model = load_pretrained_encoder(sys.argv[1], device="cpu")
print(f"validated embed_dim={model.backbone.embed_dim}")
' "$checkpoint"
}

case "$EXPECTED_BYTES" in
    ''|*[!0-9]*)
        echo "VJEPA_EXPECTED_BYTES must be a positive integer" >&2
        exit 1
        ;;
esac
if [ "$EXPECTED_BYTES" -le 0 ]; then
    echo "VJEPA_EXPECTED_BYTES must be a positive integer" >&2
    exit 1
fi

mkdir -p "$(dirname -- "$DEST")"
if [ -e "$DEST" ]; then
    ACTUAL_BYTES=$(stat -c '%s' "$DEST")
    if [ "$ACTUAL_BYTES" -eq "$EXPECTED_BYTES" ]; then
        ACTUAL_SHA256=$(sha256sum "$DEST" | awk '{print $1}')
        if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
            echo "Existing checkpoint has unexpected SHA-256: $ACTUAL_SHA256" >&2
            echo "Move it aside explicitly, then rerun this script." >&2
            exit 1
        fi
        echo "Validating existing V-JEPA checkpoint."
        if validate_checkpoint "$DEST"; then
            echo "V-JEPA checkpoint already complete: $DEST ($ACTUAL_BYTES bytes)"
            exit 0
        fi
        echo "Existing checkpoint has the expected size but failed model validation." >&2
        echo "Move it aside explicitly, then rerun this script." >&2
        exit 1
    fi
    echo "Refusing to replace existing checkpoint with unexpected size:" >&2
    echo "  $DEST ($ACTUAL_BYTES bytes; expected $EXPECTED_BYTES)" >&2
    echo "Move it aside explicitly, then rerun this script." >&2
    exit 1
fi

echo "Downloading official V-JEPA 2 ViT-L checkpoint (~4.8 GiB)."
echo "Partial downloads resume from: $PART"
wget -c --output-document "$PART" "$URL"

ACTUAL_BYTES=$(stat -c '%s' "$PART")
if [ "$ACTUAL_BYTES" -ne "$EXPECTED_BYTES" ]; then
    echo "Downloaded size $ACTUAL_BYTES != expected $EXPECTED_BYTES" >&2
    echo "The partial file is retained for inspection/resume: $PART" >&2
    exit 1
fi
ACTUAL_SHA256=$(sha256sum "$PART" | awk '{print $1}')
if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
    echo "Downloaded SHA-256 $ACTUAL_SHA256 != expected $EXPECTED_SHA256" >&2
    echo "The downloaded file is retained for inspection: $PART" >&2
    exit 1
fi

echo "Validating target_encoder against the local ViT-L architecture."
if ! validate_checkpoint "$PART"; then
    echo "Checkpoint structure/model validation failed." >&2
    echo "The downloaded file is retained for inspection: $PART" >&2
    exit 1
fi

mv "$PART" "$DEST"
echo "V-JEPA checkpoint ready: $DEST ($ACTUAL_BYTES bytes)"
