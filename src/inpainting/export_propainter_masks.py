"""Export a ``(T,H,W)`` NumPy mask as a ProPainter PNG sequence."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--protect_mask", type=Path, default=None,
        help="Optional object mask removed from the inpainting support.",
    )
    args = parser.parse_args()

    masks = np.load(args.mask, mmap_mode="r")
    protect = (np.load(args.protect_mask, mmap_mode="r")
               if args.protect_mask is not None else None)
    frame_count = len(masks) if protect is None else min(len(masks), len(protect))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pixel_count = 0
    for frame_index in range(frame_count):
        current = np.asarray(masks[frame_index], dtype=bool).copy()
        if protect is not None:
            current &= ~np.asarray(protect[frame_index], dtype=bool)
        pixel_count += int(current.sum())
        output = args.output_dir / f"{frame_index:05d}.png"
        if not cv2.imwrite(str(output), current.astype(np.uint8) * 255):
            raise RuntimeError(f"cannot write {output}")
    print(f"[ok] wrote {frame_count} masks to {args.output_dir}")
    print(f"[info] masked pixels={pixel_count}")


if __name__ == "__main__":
    main()
