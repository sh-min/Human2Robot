"""Replace hallucinated background with desk the camera actually recorded.

ProPainter fills the person's silhouette by inventing plausible texture, which
near the bottom-left corner comes out as a pink smear with a wobbly table edge —
the most obviously synthetic thing left in the shot. But the camera barely moves
(under 1 px of drift across the clip) and the arm does not cover every pixel in
every frame, so most of what was invented was also photographed at some other
moment.

For every pixel this takes the median of the frames where it was neither arm nor
interaction object, giving a true bare-desk plate, and pastes it back wherever
the plate was invented. Pixels with too few clean samples — the ones the arm
really did hide the whole time — keep the inpainted fill.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _read(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames, fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_video", type=Path, required=True,
                        help="Original footage (video_L.mp4).")
    parser.add_argument("--plate", type=Path, required=True,
                        help="Inpainted plate to repair.")
    parser.add_argument("--arm_mask", type=Path, required=True)
    parser.add_argument("--object_mask", type=Path, required=True)
    parser.add_argument("--min_samples", type=int, default=8,
                        help="Clean observations a pixel needs before its "
                             "median is trusted over the inpainted fill.")
    parser.add_argument("--dilate", type=int, default=30,
                        help="Grow the arm mask before pasting. The mask stops "
                             "at the crisp arm; the inpainter's damage extends "
                             "into the motion-blurred halo around it.")
    parser.add_argument("--shadow_drop", type=float, default=4.0,
                        help="Also paste where the plate is this many levels "
                             "darker than the clean plate: the person's cast "
                             "shadow, which inpainting never touched. Objects "
                             "keep their own contact shadows because those are "
                             "in the clean plate too, so it is not darker there.")
    parser.add_argument("--object_dilate", type=int, default=12,
                        help="Objects are held out of the paste by this margin.")
    parser.add_argument("--feather", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plate_png", type=Path, default=None,
                        help="Optional: also write the recovered clean plate.")
    args = parser.parse_args()

    source, fps = _read(args.source_video)
    plate, _ = _read(args.plate)
    arm = np.load(args.arm_mask, mmap_mode="r")
    objects = np.load(args.object_mask, mmap_mode="r")
    count = min(len(source), len(plate), len(arm), len(objects))
    height, width = source[0].shape[:2]

    # Per-pixel median over the frames where the pixel is bare desk. Running it
    # in row bands keeps the sample stack out of a 30 GB allocation.
    clean = np.zeros((height, width, 3), np.uint8)
    samples = np.zeros((height, width), np.int32)
    band = 90
    for top in range(0, height, band):
        bottom = min(top + band, height)
        stack, valid = [], []
        for idx in range(count):
            stack.append(source[idx][top:bottom])
            valid.append(~(np.asarray(arm[idx][top:bottom], dtype=bool)
                           | np.asarray(objects[idx][top:bottom], dtype=bool)))
        stack = np.stack(stack).astype(np.float32)
        valid = np.stack(valid)
        samples[top:bottom] = valid.sum(0)
        stack[~valid] = np.nan
        with np.errstate(all="ignore"):
            clean[top:bottom] = np.nan_to_num(
                np.nanmedian(stack, axis=0)).astype(np.uint8)
        print(f"[plate] rows {top}-{bottom}", flush=True)
    # The shadow test asks "darker than the clean plate", which is also true of
    # an object standing where the clean plate has bare desk. Objects are only
    # ever in places they visit at some point in the clip, so holding those out
    # keeps the test to genuine shadow.
    ever_object = np.zeros((height, width), dtype=bool)
    for idx in range(count):
        ever_object |= np.asarray(objects[idx], dtype=bool)
    shadow_ok = ~cv2.dilate(ever_object.astype(np.uint8),
                            np.ones((41, 41), np.uint8)).astype(bool)
    trusted = samples >= args.min_samples
    print(f"[info] clean plate: {trusted.sum()} px with >= {args.min_samples} "
          f"observations ({100 * trusted.mean():.1f}% of frame)")
    if args.plate_png:
        cv2.imwrite(str(args.plate_png), clean)

    kernel = np.ones((2 * args.dilate + 1,) * 2, np.uint8)
    obj_kernel = np.ones((2 * args.object_dilate + 1,) * 2, np.uint8)
    open_k, close_k = np.ones((9, 9), np.uint8), np.ones((25, 25), np.uint8)
    clean_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"FFV1"),
                             fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"cannot write {args.output}")
    pasted = 0
    for idx in range(count):
        invented = cv2.dilate(np.asarray(arm[idx], dtype=np.uint8), kernel).astype(bool)
        if args.shadow_drop > 0:
            gray = cv2.cvtColor(plate[idx], cv2.COLOR_BGR2GRAY).astype(np.float32)
            shade = (gray < clean_gray - args.shadow_drop).astype(np.uint8)
            shade = cv2.morphologyEx(shade, cv2.MORPH_OPEN, open_k)
            invented |= (cv2.morphologyEx(shade, cv2.MORPH_CLOSE, close_k).astype(bool)
                         & shadow_ok)
        held_out = cv2.dilate(np.asarray(objects[idx], dtype=np.uint8),
                              obj_kernel).astype(bool)
        paste = invented & trusted & ~held_out
        alpha = cv2.GaussianBlur(paste.astype(np.float32), (0, 0), args.feather)[..., None]
        out = np.clip(alpha * clean + (1 - alpha) * plate[idx], 0, 255).astype(np.uint8)
        writer.write(out)
        pasted += int(paste.sum())
        if idx % 100 == 0:
            print(f"[frame] {idx}/{count}", flush=True)
    writer.release()
    print(f"[info] pasted {pasted} px ({pasted / count:.0f} per frame)")
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
