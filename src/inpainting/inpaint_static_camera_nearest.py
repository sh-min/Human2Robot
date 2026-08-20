"""High-fidelity fixed-camera inpainting from real temporal observations.

For every masked pixel, copy the temporally nearest frame in which that pixel
is not covered by the human (or by a supplied moving-object exclusion mask).
Unlike a temporal median, this preserves the original tabletop texture and
camera noise.  The human mask is dilated before filling, so the feathered seam
falls on clean background rather than on a skin-coloured silhouette edge.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _kernel(radius: int) -> np.ndarray:
    size = 2 * max(0, radius) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _load_video(path: Path, frame_count: int) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    frames = np.empty((frame_count, height, width, 3), dtype=np.uint8)
    written = 0
    while written < frame_count:
        ok, frame = cap.read()
        if not ok:
            break
        frames[written] = frame
        written += 1
    cap.release()
    if written != frame_count:
        raise RuntimeError(f"decoded {written} frames, expected {frame_count}")
    return frames, fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protect_mask", type=Path, default=None,
                        help="Current-frame pixels to preserve, such as visible objects.")
    parser.add_argument("--candidate_exclude_mask", type=Path, default=None,
                        help="Pixels excluded as temporal copy sources.")
    parser.add_argument("--fallback_plate", type=Path, default=None)
    parser.add_argument("--fill_dilate", type=int, default=18)
    parser.add_argument("--candidate_dilate", type=int, default=12)
    parser.add_argument("--protect_dilate", type=int, default=3)
    parser.add_argument("--feather_sigma", type=float, default=4.0)
    parser.add_argument("--row_block", type=int, default=32)
    args = parser.parse_args()

    human = np.load(args.mask, mmap_mode="r")
    protect = (np.load(args.protect_mask, mmap_mode="r")
               if args.protect_mask is not None else None)
    exclude = (np.load(args.candidate_exclude_mask, mmap_mode="r")
               if args.candidate_exclude_mask is not None else None)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(args.video)
    video_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    count = min(video_count, len(human))
    if protect is not None:
        count = min(count, len(protect))
    if exclude is not None:
        count = min(count, len(exclude))
    frames, fps = _load_video(args.video, count)
    height, width = frames.shape[1:3]
    print(f"[info] loaded {count} frames, {width}x{height}")

    fill = np.empty((count, height, width), dtype=bool)
    invalid = np.empty_like(fill)
    alpha = np.empty((count, height, width), dtype=np.uint8)
    fill_kernel = _kernel(args.fill_dilate)
    invalid_kernel = _kernel(args.candidate_dilate)
    protect_kernel = _kernel(args.protect_dilate)
    for frame_idx in range(count):
        base = np.asarray(human[frame_idx], dtype=np.uint8)
        fill_frame = cv2.dilate(base, fill_kernel, iterations=1).astype(bool)
        invalid_frame = cv2.dilate(base, invalid_kernel, iterations=1).astype(bool)
        if exclude is not None:
            invalid_frame |= np.asarray(exclude[frame_idx], dtype=bool)
        if protect is not None:
            protected = cv2.dilate(
                np.asarray(protect[frame_idx], dtype=np.uint8),
                protect_kernel, iterations=1,
            ).astype(bool)
            fill_frame &= ~protected
        fill[frame_idx] = fill_frame
        invalid[frame_idx] = invalid_frame
        a = fill_frame.astype(np.float32)
        if args.feather_sigma > 0:
            a = cv2.GaussianBlur(a, (0, 0), args.feather_sigma)
        alpha[frame_idx] = np.clip(a * 255.0, 0, 255).astype(np.uint8)
        if (frame_idx + 1) % 100 == 0:
            print(f"[mask] {frame_idx + 1}/{count}", flush=True)

    if args.fallback_plate is not None:
        fallback = cv2.imread(str(args.fallback_plate), cv2.IMREAD_COLOR)
        if fallback is None:
            raise RuntimeError(f"cannot read {args.fallback_plate}")
        if fallback.shape[:2] != (height, width):
            fallback = cv2.resize(fallback, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        sample = frames[::max(1, count // 80)].astype(np.float32)
        sample_invalid = invalid[::max(1, count // 80), ..., None]
        sample[sample_invalid.repeat(3, axis=3)] = np.nan
        with np.errstate(all="ignore"):
            fallback = np.nanmedian(sample, axis=0)
        missing = ~np.isfinite(fallback)
        fallback[missing] = np.median(frames[::max(1, count // 80)], axis=0)[missing]
        fallback = np.clip(fallback, 0, 255).astype(np.uint8)

    cache_path = args.output.with_suffix(args.output.suffix + ".nearest_frames.npy")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    nearest_frames = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.uint8, shape=frames.shape
    )
    time_axis = np.arange(count, dtype=np.int32)[:, None, None]
    for row_start in range(0, height, max(1, args.row_block)):
        row_end = min(height, row_start + max(1, args.row_block))
        valid = ~invalid[:, row_start:row_end]
        block_h = row_end - row_start
        previous = np.empty((count, block_h, width), dtype=np.int16)
        following = np.empty_like(previous)
        last = np.full((block_h, width), -1, dtype=np.int16)
        for frame_idx in range(count):
            last[valid[frame_idx]] = frame_idx
            previous[frame_idx] = last
        last.fill(-1)
        for frame_idx in range(count - 1, -1, -1):
            last[valid[frame_idx]] = frame_idx
            following[frame_idx] = last

        prev_dist = np.where(previous >= 0, time_axis - previous, count + 1)
        next_dist = np.where(following >= 0, following - time_axis, count + 1)
        chosen = np.where(prev_dist <= next_dist, previous, following)
        y_grid, x_grid = np.indices((block_h, width))
        source_block = frames[:, row_start:row_end]
        for frame_idx in range(count):
            choice = chosen[frame_idx]
            missing = choice < 0
            safe = np.maximum(choice, 0)
            copied = source_block[safe, y_grid, x_grid]
            if missing.any():
                copied[missing] = fallback[row_start:row_end][missing]
            nearest_frames[frame_idx, row_start:row_end] = copied
        nearest_frames.flush()
        print(f"[nearest] rows {row_start}:{row_end}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"FFV1"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {args.output}")
    for frame_idx in range(count):
        a = alpha[frame_idx].astype(np.float32)[..., None] / 255.0
        composed = (a * np.asarray(nearest_frames[frame_idx], dtype=np.float32) +
                    (1.0 - a) * frames[frame_idx].astype(np.float32))
        writer.write(np.clip(composed, 0, 255).astype(np.uint8))
        if (frame_idx + 1) % 100 == 0:
            print(f"[frame] {frame_idx + 1}/{count}", flush=True)
    writer.release()
    del nearest_frames
    cache_path.unlink(missing_ok=True)
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
