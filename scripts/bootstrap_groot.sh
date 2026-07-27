#!/usr/bin/env bash
set -euo pipefail

# Initialize the pinned Isaac-GR00T submodule and create its official uv
# environment. Model weights are intentionally not downloaded here.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GROOT_ROOT="${GROOT_ROOT:-$ROOT/third_party/Isaac-GR00T}"

cd "$ROOT"
git submodule update --init --recursive third_party/Isaac-GR00T

if ! git lfs version >/dev/null 2>&1 &&
   test -x "$HOME/miniforge3/bin/git-lfs"; then
    export PATH="$HOME/miniforge3/bin:$PATH"
fi
if git lfs version >/dev/null 2>&1; then
    git -C "$GROOT_ROOT" lfs install --local
    git -C "$GROOT_ROOT" lfs pull \
      --include="scripts/deployment/dgpu/wheels/torchcodec-*.whl"
else
    echo "git-lfs is required by Isaac-GR00T's packaged wheels." >&2
    echo "Install it, then rerun this script." >&2
    exit 1
fi

if command -v uv >/dev/null; then
    UV_BIN="$(command -v uv)"
elif test -n "${CONDA_EXE:-}" &&
     test -x "${CONDA_EXE%/bin/conda}/bin/uv"; then
    UV_BIN="${CONDA_EXE%/bin/conda}/bin/uv"
elif test -x "$HOME/miniforge3/bin/uv"; then
    UV_BIN="$HOME/miniforge3/bin/uv"
else
    echo "uv is required. Install it from https://docs.astral.sh/uv/" >&2
    exit 1
fi

(
  cd "$GROOT_ROOT"
  UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/weights/uv}" \
    "$UV_BIN" sync --python 3.12
  .venv/bin/python -c \
    'import gr00t, torch; print("GR00T ready", torch.__version__, torch.cuda.is_available())'
)

cat <<'EOF'
GR00T environment is ready.
Before the first model load:
  1. Request access to nvidia/Cosmos-Reason2-2B on Hugging Face.
  2. Run: third_party/Isaac-GR00T/.venv/bin/hf auth login
EOF
