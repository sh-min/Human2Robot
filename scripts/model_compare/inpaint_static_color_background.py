#!/usr/bin/env python3
"""Remove human arms using a clean temporal plate for a fixed RGB camera.

This is the compact color-video adapter for the temporal-background method in
the latest Human2Robot inpainting branch.  Every plate pixel is estimated only
from frames where the SAM2 human mask is absent.  The plate is then blended
back strictly inside the current human mask, so manipulated objects excluded
by SAM2 remain in the source frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _decode(path: Path, count: int) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frames = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) != count:
        raise RuntimeError(f"decoded {len(frames)} frames, expected {count}")
    return np.stack(frames), fps


def _horizontal_repair(plate: np.ndarray, missing: np.ndarray) -> np.ndarray:
    repaired = plate.astype(np.float32)
    width = plate.shape[1]
    x_axis = np.arange(width, dtype=np.float32)
    for y in range(plate.shape[0]):
        row_missing = missing[y]
        if not row_missing.any():
            continue
        good = np.flatnonzero(~row_missing)
        if not len(good):
            continue
        for channel in range(3):
            repaired[y, row_missing, channel] = np.interp(
                x_axis[row_missing], good, repaired[y, good, channel]
            )
    return np.clip(repaired, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plate-output", type=Path, default=None)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--mask-dilate", type=int, default=10)
    parser.add_argument("--feather-sigma", type=float, default=2.0)
    parser.add_argument("--row-block", type=int, default=40)
    args = parser.parse_args()

    masks = np.load(args.mask, mmap_mode="r")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    count = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(masks))
    cap.release()
    frames, fps = _decode(args.video, count)
    height, width = frames.shape[1:3]

    radius = max(0, args.mask_dilate)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    expanded = np.empty((count, height, width), dtype=bool)
    for index in range(count):
        base = np.asarray(masks[index], dtype=np.uint8)
        expanded[index] = cv2.dilate(base, kernel).astype(bool)

    sample_indices = np.arange(0, count, max(1, args.sample_stride))
    samples = frames[sample_indices]
    invalid = expanded[sample_indices]
    plate = np.empty((height, width, 3), dtype=np.uint8)
    unsupported = np.zeros((height, width), dtype=bool)
    fallback = np.median(samples, axis=0)
    for y0 in range(0, height, max(1, args.row_block)):
        y1 = min(height, y0 + max(1, args.row_block))
        values = samples[:, y0:y1].astype(np.float32)
        bad = invalid[:, y0:y1, :, None]
        values[bad.repeat(3, axis=3)] = np.nan
        with np.errstate(all="ignore"):
            block = np.nanmedian(values, axis=0)
        missing = ~np.isfinite(block).all(axis=2)
        if missing.any():
            block[missing] = fallback[y0:y1][missing]
        plate[y0:y1] = np.clip(block, 0, 255).astype(np.uint8)
        unsupported[y0:y1] = missing
    if unsupported.any():
        plate = _horizontal_repair(plate, unsupported)

    plate_path = args.plate_output or args.output.with_name("background_plate.jpg")
    plate_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(plate_path), plate, [cv2.IMWRITE_JPEG_QUALITY, 95])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {args.output}")
    plate_float = plate.astype(np.float32)
    for index, frame in enumerate(frames):
        alpha = expanded[index].astype(np.float32)
        if args.feather_sigma > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), args.feather_sigma)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]
        composed = alpha * plate_float + (1.0 - alpha) * frame.astype(np.float32)
        writer.write(np.clip(composed, 0, 255).astype(np.uint8))
    writer.release()
    print(f"[info] frames={count}, samples={len(sample_indices)}, "
          f"unsupported={int(unsupported.sum())}px")
    print(f"[ok] wrote {args.output}")
    print(f"[ok] wrote {plate_path}")


if __name__ == "__main__":
    main()
