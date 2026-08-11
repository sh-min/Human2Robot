#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point for the retired clip/residual pipeline.
# The replacement defaults to IMG_5019; pass episode IDs explicitly or set
# ALL=1 to process every recording.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "[deprecated] run_all_xhand_overlay.sh -> run_all_rby1_xhand_overlay.sh"
exec bash "$SCRIPT_DIR/run_all_rby1_xhand_overlay.sh" "$@"
