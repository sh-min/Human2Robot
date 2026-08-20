#!/usr/bin/env python3
"""Remove a human mask while preserving every usable source-video pixel.

For a fixed camera, pixels outside the current removal mask are copied from
the current frame unchanged.  Removed pixels are filled from a temporal plate
built only from frames where the same location is valid.  Spatial inpainting
is restricted to locations that are never valid anywhere in the sequence.
An optional protect mask (for example a SAM2 object track) always retains the
current-frame source RGB, including along feathered removal boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def decode_video(path: Path, count: int) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[np.ndarray] = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != count:
        raise RuntimeError(f"decoded {len(frames)} frames, expected {count}")
    return np.stack(frames), fps


def resize_masks(masks: np.ndarray, count: int, height: int, width: int) -> np.ndarray:
    output = np.empty((count, height, width), dtype=bool)
    for index in range(count):
        mask = np.asarray(masks[index], dtype=np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        output[index] = mask.astype(bool)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--human-mask", type=Path, required=True)
    parser.add_argument("--protect-mask", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plate-output", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--mask-dilate", type=int, default=25)
    parser.add_argument("--feather-sigma", type=float, default=2.0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--inpaint-radius", type=float, default=3.0)
    parser.add_argument("--row-block", type=int, default=40)
    args = parser.parse_args()

    human_source = np.load(args.human_mask, mmap_mode="r")
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(args.video)
    video_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    count = min(video_count, len(human_source))
    frames, fps = decode_video(args.video, count)
    height, width = frames.shape[1:3]
    human = resize_masks(human_source, count, height, width)

    if args.protect_mask is not None:
        protect_source = np.load(args.protect_mask, mmap_mode="r")
        if len(protect_source) < count:
            raise ValueError("protect mask has fewer frames than the video")
        protect = resize_masks(protect_source, count, height, width)
    else:
        protect = np.zeros_like(human)

    radius = max(0, args.mask_dilate)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    removal = np.empty_like(human)
    for index in range(count):
        expanded = cv2.dilate(human[index].astype(np.uint8), kernel).astype(bool)
        removal[index] = expanded & ~protect[index]

    sample_indices = np.arange(0, count, max(1, args.sample_stride))
    samples = frames[sample_indices]
    invalid = removal[sample_indices]
    plate_float = np.empty((height, width, 3), dtype=np.float32)
    unsupported = np.zeros((height, width), dtype=bool)
    block_height = max(1, args.row_block)
    for y0 in range(0, height, block_height):
        y1 = min(height, y0 + block_height)
        values = samples[:, y0:y1].astype(np.float32)
        values[np.repeat(invalid[:, y0:y1, :, None], 3, axis=3)] = np.nan
        with np.errstate(all="ignore"):
            block = np.nanmedian(values, axis=0)
        missing = ~np.isfinite(block).all(axis=2)
        block[missing] = 0.0
        plate_float[y0:y1] = block
        unsupported[y0:y1] = missing

    plate = np.clip(plate_float, 0, 255).astype(np.uint8)
    if unsupported.any():
        plate = cv2.inpaint(
            plate,
            unsupported.astype(np.uint8) * 255,
            args.inpaint_radius,
            cv2.INPAINT_TELEA,
        )

    plate_path = args.plate_output or args.output.with_name("copy_first_plate.jpg")
    plate_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(plate_path), plate, [cv2.IMWRITE_JPEG_QUALITY, 95])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {args.output}")
    plate_float = plate.astype(np.float32)
    temporal_samples = 0
    spatial_samples = 0
    protected_samples = int((protect & human).sum())
    for index, frame in enumerate(frames):
        alpha = removal[index].astype(np.float32)
        if args.feather_sigma > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), args.feather_sigma)
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha[protect[index]] = 0.0
        composed = (
            alpha[..., None] * plate_float
            + (1.0 - alpha[..., None]) * frame.astype(np.float32)
        )
        # Enforce bit-exact source ownership for protected object pixels.
        composed[protect[index]] = frame[protect[index]]
        writer.write(np.clip(composed, 0, 255).astype(np.uint8))
        spatial_samples += int((removal[index] & unsupported).sum())
        temporal_samples += int((removal[index] & ~unsupported).sum())
    writer.release()

    report = {
        "schema_version": 1,
        "method": "current-frame copy, temporal same-pixel donor, spatial inpaint only if never observed",
        "frames": count,
        "source_pixels_preserved_outside_removal": True,
        "protected_pixels_forced_to_current_frame_rgb": True,
        "temporal_donor_samples": temporal_samples,
        "spatial_inpaint_samples": spatial_samples,
        "never_observed_pixel_locations": int(unsupported.sum()),
        "protected_human_overlap_samples": protected_samples,
        "mask_dilate": radius,
        "sample_stride": int(args.sample_stride),
    }
    report_path = args.report_output or args.output.with_name("copy_first_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[ok] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
