#!/usr/bin/env bash
set -euo pipefail

# Frame-synchronized side-by-side comparison against the user-provided
# inpainted_video.mp4 reference. Usage:
#   bash scripts/model_compare/compare_with_inpainted_video.sh RESULT.mp4 [OUTPUT.mp4]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REFERENCE="${REFERENCE:-$ROOT/inpainted_video.mp4}"
LEFT_LABEL="${LEFT_LABEL:-EgoEngine}"
RIGHT_LABEL="${RIGHT_LABEL:-PICHU}"
RESULT="${1:?usage: $0 RESULT.mp4 [OUTPUT.mp4]}"
RESULT="$(realpath "$RESULT")"
if [[ $# -ge 2 ]]; then
    OUTPUT="$2"
else
    OUTPUT="${RESULT%.mp4}_vs_inpainted_video.mp4"
fi

test -s "$REFERENCE"
test -s "$RESULT"
mkdir -p "$(dirname "$OUTPUT")"

ffmpeg -nostdin -y -hide_banner -loglevel error \
    -i "$REFERENCE" -i "$RESULT" \
    -filter_complex \
    "[0:v]scale=960:544:flags=lanczos,setsar=1,drawtext=text='$LEFT_LABEL':x=24:y=24:fontsize=30:fontcolor=white:borderw=2:bordercolor=black[left];\
     [1:v]scale=960:540:flags=lanczos,pad=960:544:0:2:black,setsar=1,drawtext=text='$RIGHT_LABEL':x=24:y=24:fontsize=30:fontcolor=white:borderw=2:bordercolor=black[right];\
     [left][right]hstack=inputs=2[v]" \
    -map '[v]' -an -r 30 -frames:v 124 \
    -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart \
    "$OUTPUT"

echo "[done] $OUTPUT"
