"""Render a (T, H, W) boolean mask over its source video as a review clip.

`segment_arms.py` writes masks_arm.npy and nothing else -- the upstream debug
videos were dropped from the vendored copy -- so there is no way to watch a
segmentation result without re-deriving it. This fills that gap for any mask
npy in the pipeline (hand+arm, object, forced-front, residual).

Panels: source | mask over source | mask alone, with per-frame coverage.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

LABEL_H = 38


def label_bar(width: int, text: str, accent: tuple[int, int, int]) -> np.ndarray:
    bar = np.full((LABEL_H, width, 3), 18, np.uint8)
    cv2.rectangle(bar, (0, 0), (6, LABEL_H), accent, -1)
    cv2.putText(bar, text, (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (240, 240, 240), 1, cv2.LINE_AA)
    return bar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_video", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="SAM2 hand + arm mask")
    parser.add_argument("--color", type=int, nargs=3, default=(90, 235, 90),
                        metavar=("B", "G", "R"))
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--panel_width", type=int, default=560)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--no_mask_panel", action="store_true")
    args = parser.parse_args()

    masks = np.load(args.mask, mmap_mode="r")
    cap = cv2.VideoCapture(str(args.source_video))
    if not cap.isOpened():
        raise FileNotFoundError(args.source_video)
    frame_count = min(len(masks), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_w = args.panel_width
    panel_h = int(round(src_h * panel_w / src_w))
    colour = np.array(args.color, dtype=np.float32)

    panels = 2 if args.no_mask_panel else 3
    total_w = panel_w * panels
    total_h = LABEL_H + panel_h + 30
    total_w += total_w % 2
    total_h += total_h % 2

    coverage = np.zeros(frame_count, dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (total_w, total_h))
        if not writer.isOpened():
            raise RuntimeError("cannot open temporary writer")

        for t in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                break
            mask = np.asarray(masks[t], dtype=bool)
            coverage[t] = 100.0 * mask.mean()

            blended = frame.astype(np.float32)
            blended[mask] = ((1.0 - args.alpha) * blended[mask]
                             + args.alpha * colour)
            blended = blended.astype(np.uint8)
            # A contour keeps thin structures (fingers) readable where the
            # translucent fill alone washes out against a bright table.
            edges = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                                     np.ones((3, 3), np.uint8)).astype(bool)
            blended[edges] = (40, 40, 235)

            views = [("source", frame), (args.label, blended)]
            if not args.no_mask_panel:
                flat = np.zeros_like(frame)
                flat[mask] = colour.astype(np.uint8)
                views.append(("mask only", flat))

            accents = [(90, 90, 90), (90, 200, 90), (60, 160, 255)]
            canvas = np.zeros((total_h, total_w, 3), np.uint8)
            for idx, (text, view) in enumerate(views):
                block = np.vstack([
                    label_bar(panel_w, text, accents[idx % len(accents)]),
                    cv2.resize(view, (panel_w, panel_h),
                               interpolation=cv2.INTER_AREA),
                ])
                canvas[:block.shape[0], idx * panel_w:(idx + 1) * panel_w] = block

            cv2.putText(canvas,
                        f"frame {t + 1}/{frame_count}   "
                        f"mask coverage {coverage[t]:.1f}% of frame",
                        (14, total_h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (200, 200, 200), 1, cv2.LINE_AA)
            writer.write(canvas)
        writer.release()
        cap.release()

        args.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
             "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-crf", "18", str(args.output)],
            check=True,
        )

    print(f"[ok] {args.output}  frames={frame_count}  {total_w}x{total_h}  "
          f"coverage mean={coverage.mean():.2f}% max={coverage.max():.2f}%")


if __name__ == "__main__":
    main()
