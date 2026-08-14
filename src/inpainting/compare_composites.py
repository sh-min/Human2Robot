"""Side-by-side comparison of composite variants, with an optional zoom row.

Layer-order changes are small in absolute pixel count and concentrated around
the grasp, so a full-frame A/B alone hides them. The zoom row crops the same
window from every variant and scales it to the panel width, which is where the
front/behind decision is actually visible.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

LABEL_H = 40


def label_bar(width: int, text: str, accent: tuple[int, int, int]) -> np.ndarray:
    bar = np.full((LABEL_H, width, 3), 18, np.uint8)
    cv2.rectangle(bar, (0, 0), (6, LABEL_H), accent, -1)
    cv2.putText(bar, text, (18, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (240, 240, 240), 1, cv2.LINE_AA)
    return bar


def tracked_centres(mask_paths, frame_count, frame_size, window, sigma):
    """Per-frame crop centre following the intersection of the given masks.

    Frames where the intersection is empty inherit the nearest populated
    frame's centre, then the whole track is Gaussian-smoothed so the zoom pans
    instead of jumping between grasps.
    """
    masks = [np.load(p, mmap_mode="r") for p in mask_paths]
    width, height = frame_size
    win_w, win_h = window
    centre = np.full((frame_count, 2), np.nan, dtype=np.float32)
    for t in range(frame_count):
        # A memmap slice is read-only, so copy before the in-place intersect.
        current = np.array(masks[0][t], dtype=bool)
        for extra in masks[1:]:
            current &= np.asarray(extra[t], dtype=bool)
        if current.sum() < 50:
            continue
        ys, xs = np.nonzero(current)
        centre[t] = ((xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0)

    good = np.flatnonzero(np.isfinite(centre[:, 0]))
    if not len(good):
        return np.tile([width / 2.0, height / 2.0], (frame_count, 1))
    for axis in (0, 1):
        centre[:, axis] = np.interp(np.arange(frame_count), good,
                                    centre[good, axis])
    if sigma > 0:
        centre = gaussian_filter1d(centre, sigma, axis=0, mode="nearest")

    centre[:, 0] = np.clip(centre[:, 0], win_w / 2.0, width - win_w / 2.0)
    centre[:, 1] = np.clip(centre[:, 1], win_h / 2.0, height - win_h / 2.0)
    return centre


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", required=True,
                        metavar="LABEL=PATH",
                        help="Repeatable. Panels appear in the given order.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel_width", type=int, default=640)
    parser.add_argument("--crop", type=int, nargs=4, default=None,
                        metavar=("X", "Y", "W", "H"),
                        help="Adds a zoom row showing this source-pixel window.")
    parser.add_argument("--track_mask", action="append", default=None,
                        help="Repeatable (T, H, W) mask npy. Their intersection "
                             "centroid recentres the zoom window each frame, so "
                             "the grasp stays framed while the arm moves.")
    parser.add_argument("--track_size", type=int, nargs=2, default=(460, 300),
                        metavar=("W", "H"))
    parser.add_argument("--track_sigma", type=float, default=12.0,
                        help="Temporal smoothing of the crop centre, in frames.")
    parser.add_argument("--fps", type=float, default=24.0)
    args = parser.parse_args()

    if args.track_mask and args.crop:
        raise ValueError("--crop and --track_mask are mutually exclusive")

    entries = []
    for item in args.video:
        label, _, path = item.partition("=")
        if not path:
            raise ValueError(f"expected LABEL=PATH, got {item!r}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(path)
        entries.append((label, cap))

    accents = [(90, 90, 90), (90, 200, 90), (60, 160, 255), (200, 120, 255)]
    frame_count = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                      for _, cap in entries)
    src_w = int(entries[0][1].get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(entries[0][1].get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_w = args.panel_width
    panel_h = int(round(src_h * panel_w / src_w))

    zoom_h = 0
    track_centres = None
    if args.crop is not None:
        cw, ch = args.crop[2], args.crop[3]
        zoom_h = int(round(ch * panel_w / cw))
    elif args.track_mask:
        cw, ch = args.track_size
        zoom_h = int(round(ch * panel_w / cw))
        track_centres = tracked_centres(
            [Path(p) for p in args.track_mask], frame_count,
            (src_w, src_h), (cw, ch), args.track_sigma,
        )

    total_w = panel_w * len(entries)
    total_h = LABEL_H + panel_h + (LABEL_H + zoom_h if zoom_h else 0)
    # libx264 needs even dimensions in yuv420p.
    total_w += total_w % 2
    total_h += total_h % 2

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (total_w, total_h))
        if not writer.isOpened():
            raise RuntimeError("cannot open temporary writer")

        for frame_idx in range(frame_count):
            if args.crop is not None:
                cx, cy, cw, ch = args.crop
            elif track_centres is not None:
                cw, ch = args.track_size
                cx = int(round(track_centres[frame_idx, 0] - cw / 2.0))
                cy = int(round(track_centres[frame_idx, 1] - ch / 2.0))
            top, zoom = [], []
            ok_all = True
            for idx, (label, cap) in enumerate(entries):
                ok, frame = cap.read()
                if not ok:
                    ok_all = False
                    break
                accent = accents[idx % len(accents)]
                top.append(np.vstack([
                    label_bar(panel_w, label, accent),
                    cv2.resize(frame, (panel_w, panel_h),
                               interpolation=cv2.INTER_AREA),
                ]))
                if zoom_h:
                    patch = frame[cy:cy + ch, cx:cx + cw]
                    zoom.append(np.vstack([
                        label_bar(panel_w, f"{label}  (zoom)", accent),
                        cv2.resize(patch, (panel_w, zoom_h),
                                   interpolation=cv2.INTER_CUBIC),
                    ]))
            if not ok_all:
                break

            canvas = np.zeros((total_h, total_w, 3), np.uint8)
            row = np.hstack(top)
            canvas[:row.shape[0], :row.shape[1]] = row
            if zoom_h:
                zrow = np.hstack(zoom)
                y0 = LABEL_H + panel_h
                canvas[y0:y0 + zrow.shape[0], :zrow.shape[1]] = zrow
            writer.write(canvas)
        writer.release()

        args.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
             "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-crf", "18", str(args.output)],
            check=True,
        )

    for _, cap in entries:
        cap.release()
    print(f"[ok] {args.output}  {total_w}x{total_h}  frames={frame_count}")


if __name__ == "__main__":
    main()
