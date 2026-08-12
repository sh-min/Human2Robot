"""Assemble a full-resolution background from a low-resolution ProPainter pass.

ProPainter pads its encoded output to a macroblock-aligned height.  This tool
center-crops that padding, restores the source resolution, and blends only the
requested human-removal mask back into the high-resolution source.  Pixels in
the optional protect mask remain untouched for later object-layer restoration.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _center_crop_to_aspect(frame: np.ndarray, width: int,
                           height: int) -> np.ndarray:
    source_h, source_w = frame.shape[:2]
    target_aspect = width / height
    source_aspect = source_w / source_h
    if abs(source_aspect - target_aspect) < 1e-6:
        return frame
    if source_aspect < target_aspect:
        crop_h = int(round(source_w / target_aspect))
        y0 = max(0, (source_h - crop_h) // 2)
        return frame[y0:y0 + crop_h]
    crop_w = int(round(source_h * target_aspect))
    x0 = max(0, (source_w - crop_w) // 2)
    return frame[:, x0:x0 + crop_w]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_video", type=Path, required=True)
    parser.add_argument("--propainter_video", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--protect_mask", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dilate", type=int, default=3)
    parser.add_argument("--edge_sigma", type=float, default=1.2)
    args = parser.parse_args()

    source_cap = cv2.VideoCapture(str(args.source_video))
    propainter_cap = cv2.VideoCapture(str(args.propainter_video))
    if not source_cap.isOpened() or not propainter_cap.isOpened():
        raise RuntimeError("cannot open source or ProPainter video")
    width = int(source_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(source_cap.get(cv2.CAP_PROP_FPS)) or 24.0
    masks = np.load(args.mask, mmap_mode="r")
    protect = (np.load(args.protect_mask, mmap_mode="r")
               if args.protect_mask is not None else None)

    frame_count = min(
        int(source_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(propainter_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        len(masks),
    )
    if protect is not None:
        frame_count = min(frame_count, len(protect))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"FFV1"),
        fps, (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {args.output}")

    kernel = None
    if args.dilate > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * args.dilate + 1, 2 * args.dilate + 1)
        )
    blended_pixels = 0
    for frame_idx in range(frame_count):
        ok_source, source = source_cap.read()
        ok_propainter, propainter = propainter_cap.read()
        if not ok_source or not ok_propainter:
            raise RuntimeError(f"video decode stopped at frame {frame_idx}")
        propainter = _center_crop_to_aspect(propainter, width, height)
        propainter = cv2.resize(
            propainter, (width, height), interpolation=cv2.INTER_CUBIC
        )

        mask = np.asarray(masks[frame_idx], dtype=bool).copy()
        if protect is not None:
            mask &= ~np.asarray(protect[frame_idx], dtype=bool)
        if kernel is not None:
            mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
            if protect is not None:
                mask &= ~np.asarray(protect[frame_idx], dtype=bool)
        alpha = mask.astype(np.float32)
        if args.edge_sigma > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), args.edge_sigma)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]
        result = (
            alpha * propainter.astype(np.float32) +
            (1.0 - alpha) * source.astype(np.float32)
        )
        writer.write(np.clip(result, 0, 255).astype(np.uint8))
        blended_pixels += int(mask.sum())
        if (frame_idx + 1) % 100 == 0:
            print(f"[frame] {frame_idx + 1}/{frame_count}", flush=True)

    source_cap.release()
    propainter_cap.release()
    writer.release()
    print(f"[ok] wrote {args.output}")
    print(f"[info] blended pixels={blended_pixels}")


if __name__ == "__main__":
    main()
