"""Apply the production SAM2 arm-mask denoiser to an existing mask sequence.

This is intentionally a post-processing-only entry point.  It lets an older
unsmoothed ``masks_arm.npy`` be converted with exactly the same ``_smooth_masks``
implementation used by ``segment_arms.py`` without running SAM2 inference a
second time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from segment_arms import _smooth_masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min_area_frac", type=float, default=2e-4)
    parser.add_argument("--close", type=int, default=5)
    parser.add_argument("--temporal_win", type=int, default=5)
    parser.add_argument("--boundary_sigma", type=float, default=1.5)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise ValueError("--input and --output must be different paths")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    masks = np.load(args.input, mmap_mode="r")
    if masks.ndim != 3:
        raise ValueError(f"Expected (T,H,W) masks, got {masks.shape}")

    smoothed, removed = _smooth_masks(
        masks,
        min_area_frac=args.min_area_frac,
        close_ksize=args.close,
        temporal_win=args.temporal_win,
        boundary_sigma=args.boundary_sigma,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, smoothed)

    before = np.asarray(masks).sum(axis=(1, 2), dtype=np.int64)
    after = smoothed.sum(axis=(1, 2), dtype=np.int64)
    changed = np.count_nonzero(
        np.asarray(masks, dtype=bool) != smoothed,
        axis=(1, 2),
    )
    print(
        "[smooth] "
        f"frames={len(smoothed)}, specks_removed={removed}, "
        f"mean_area={before.mean():.0f}->{after.mean():.0f}px, "
        f"mean_changed={changed.mean():.0f}px/frame"
    )
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
