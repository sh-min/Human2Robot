#!/usr/bin/env bash
set -euo pipefail

# Parallel byte-range downloader for the official V-JEPA 2 ViT-L checkpoint.
# Each segment is size-checked, assembled at aligned offsets, then validated by
# the same local model loader used by download_vjepa2_vitl.sh.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
URL="${VJEPA_URL:-https://dl.fbaipublicfiles.com/vjepa2/vitl.pt}"
DEST="${VJEPA_CKPT:-$ROOT/weights/vjepa2/vitl.pt}"
EXPECTED_BYTES="${VJEPA_EXPECTED_BYTES:-5127726842}"
EXPECTED_SHA256="${VJEPA_EXPECTED_SHA256:-5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012}"
VJEPA_ENV="${VJEPA_ENV:-vjepa2-312}"
PARTS="${VJEPA_DOWNLOAD_PARTS:-8}"
SEGMENT_DIR="${DEST}.segments"
ASSEMBLED="${DEST}.parallel"
MIB=1048576

if [ -s "$DEST" ]; then
    ACTUAL_BYTES=$(stat -c '%s' "$DEST")
    if [ "$ACTUAL_BYTES" -ne "$EXPECTED_BYTES" ]; then
        echo "Existing destination has unexpected size: $ACTUAL_BYTES" >&2
        exit 1
    fi
    ACTUAL_SHA256=$(sha256sum "$DEST" | awk '{print $1}')
    if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
        echo "Existing destination has unexpected SHA-256: $ACTUAL_SHA256" >&2
        exit 1
    fi
    echo "Checkpoint already has the official size and SHA-256: $DEST"
    exit 0
fi
case "$PARTS" in
    ''|*[!0-9]*) echo "VJEPA_DOWNLOAD_PARTS must be an integer" >&2; exit 1 ;;
esac
if [ "$PARTS" -lt 1 ] || [ "$PARTS" -gt 32 ]; then
    echo "VJEPA_DOWNLOAD_PARTS must be between 1 and 32" >&2
    exit 1
fi

mkdir -p "$SEGMENT_DIR" "$(dirname -- "$DEST")"
CHUNK_MIB=$(( (EXPECTED_BYTES + PARTS * MIB - 1) / (PARTS * MIB) ))
CHUNK_BYTES=$(( CHUNK_MIB * MIB ))

download_segment() {
    local index="$1"
    local start=$(( index * CHUNK_BYTES ))
    local end=$(( start + CHUNK_BYTES - 1 ))
    local output
    local expected
    if [ "$start" -ge "$EXPECTED_BYTES" ]; then
        return 0
    fi
    if [ "$end" -ge "$EXPECTED_BYTES" ]; then
        end=$(( EXPECTED_BYTES - 1 ))
    fi
    output=$(printf '%s/segment_%02d.part' "$SEGMENT_DIR" "$index")
    expected=$(( end - start + 1 ))
    if [ -e "$output" ]; then
        actual=$(stat -c '%s' "$output")
        if [ "$actual" -eq "$expected" ]; then
            echo "[$index/$PARTS] reuse verified segment ($actual bytes)"
            return 0
        fi
        echo "Segment $index is partial: $actual != $expected" >&2
        echo "Move only that segment aside explicitly before retrying." >&2
        return 1
    fi
    echo "[$index/$PARTS] bytes $start-$end"
    curl --fail --location --silent --show-error \
        --retry 10 --retry-delay 2 --retry-all-errors \
        --range "$start-$end" \
        --output "$output" \
        "$URL"
    actual=$(stat -c '%s' "$output")
    if [ "$actual" -ne "$expected" ]; then
        echo "Segment $index size $actual != $expected" >&2
        return 1
    fi
}

export URL EXPECTED_BYTES PARTS SEGMENT_DIR CHUNK_BYTES
export -f download_segment
for ((index=0; index<PARTS; index++)); do
    download_segment "$index" &
done
wait

truncate -s "$EXPECTED_BYTES" "$ASSEMBLED"
for ((index=0; index<PARTS; index++)); do
    start=$(( index * CHUNK_BYTES ))
    if [ "$start" -ge "$EXPECTED_BYTES" ]; then
        break
    fi
    segment=$(printf '%s/segment_%02d.part' "$SEGMENT_DIR" "$index")
    dd if="$segment" of="$ASSEMBLED" bs=1M \
        seek=$(( start / MIB )) conv=notrunc status=none
done

ACTUAL_BYTES=$(stat -c '%s' "$ASSEMBLED")
if [ "$ACTUAL_BYTES" -ne "$EXPECTED_BYTES" ]; then
    echo "Assembled size $ACTUAL_BYTES != $EXPECTED_BYTES" >&2
    exit 1
fi
ACTUAL_SHA256=$(sha256sum "$ASSEMBLED" | awk '{print $1}')
if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
    echo "Assembled SHA-256 $ACTUAL_SHA256 != $EXPECTED_SHA256" >&2
    exit 1
fi

echo "Validating assembled target_encoder with the local ViT-L architecture."
PYTHONPATH="$ROOT/src" conda run -n "$VJEPA_ENV" --no-capture-output \
    python -c '
import sys
from data_preprocess.feature_extractor import load_pretrained_encoder
model = load_pretrained_encoder(sys.argv[1], device="cpu")
print(f"validated embed_dim={model.backbone.embed_dim}")
' "$ASSEMBLED"

mv "$ASSEMBLED" "$DEST"
echo "V-JEPA checkpoint ready: $DEST ($ACTUAL_BYTES bytes)"
