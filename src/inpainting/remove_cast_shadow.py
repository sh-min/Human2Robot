"""Erase the human's cast shadow from the inpainted background plate.

Inpainting removes the person, not the shadow the person threw on the desk, so
the plate keeps a large soft shadow that sweeps with the *human* arm. Once the
robot is composited on top, that shadow reads as the robot's — except it does
not follow the robot hand, because it never belonged to it. Removing it leaves
the compositor's contact shadow (which is projected from the robot geometry
every frame) as the only shadow in the shot.

The desk is static and the shadow only passes over it, so the shadow-free level
of every pixel is its bright end over time (per-pixel 90th percentile). A frame
darker than that by a smooth, wide margin is in shadow, and dividing it back up
restores the desk.

Two things must survive: the objects themselves, and the small contact shadows
they cast where they sit. Both are handled by never touching a pixel that is
inside — or within `--guard` px of — any interaction-object mask, and by only
correcting pixels the reference says are bare desk (bright and unsaturated).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _read_video(path: Path) -> tuple[list[np.ndarray], float]:
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
    parser.add_argument("--plate", type=Path, required=True,
                        help="Inpainted background video to clean.")
    parser.add_argument("--object_mask", type=Path, required=True,
                        help="Per-frame interaction-object masks; their union "
                             "plus --guard px is left untouched so object "
                             "contact shadows survive.")
    parser.add_argument("--robot_mask", type=Path, default=None,
                        help="Optional; skipped so the robot's own footprint "
                             "is left to the compositor's contact shadow.")
    parser.add_argument("--guard", type=int, default=20)
    parser.add_argument("--ratio_lo", type=float, default=0.40,
                        help="Darker than this fraction of the reference is "
                             "taken to be an object edge, not a shadow.")
    parser.add_argument("--ratio_hi", type=float, default=0.97)
    parser.add_argument("--gain_max", type=float, default=2.4)
    parser.add_argument("--smooth", type=float, default=25.0,
                        help="Gaussian sigma (px) low-passing the gain field. "
                             "A cast shadow is smooth; this keeps texture and "
                             "edges from being corrected along with it.")
    parser.add_argument("--feather", type=float, default=12.0)
    parser.add_argument("--stride", type=int, default=4,
                        help="Frame stride for building the reference.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames, fps = _read_video(args.plate)
    height, width = frames[0].shape[:2]
    stack = np.stack(frames[::args.stride])
    reference = np.percentile(stack, 90, axis=0).astype(np.uint8)
    del stack
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    desk = (hsv[..., 2] > 110) & (hsv[..., 1] < 50)

    objects = np.load(args.object_mask, mmap_mode="r")
    robot = np.load(args.robot_mask, mmap_mode="r") if args.robot_mask else None
    ever = np.zeros((height, width), dtype=bool)
    for idx in range(0, min(len(objects), len(frames)), args.stride):
        ever |= np.asarray(objects[idx], dtype=bool)
    guard = cv2.dilate(ever.astype(np.uint8),
                       np.ones((2 * args.guard + 1,) * 2, np.uint8)).astype(bool)
    correctable = desk & ~guard
    print(f"[info] desk {desk.sum()} px, object guard {guard.sum()} px, "
          f"correctable {correctable.sum()} px", flush=True)

    open_k = np.ones((7, 7), np.uint8)
    close_k = np.ones((31, 31), np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"FFV1"),
                             fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"cannot write {args.output}")
    before = after = 0.0
    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ratio = gray / np.maximum(ref_gray, 1.0)
        shadow = correctable & (ratio > args.ratio_lo) & (ratio < args.ratio_hi)
        if robot is not None and idx < len(robot):
            shadow &= ~np.asarray(robot[idx], dtype=bool)
        shadow = cv2.morphologyEx(shadow.astype(np.uint8), cv2.MORPH_OPEN, open_k)
        shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, close_k).astype(bool)

        weight = shadow.astype(np.float32)
        raw = np.where(shadow, 1.0 / np.clip(ratio, args.ratio_lo, 1.0), 1.0)
        gain = (cv2.GaussianBlur(raw.astype(np.float32) * weight, (0, 0), args.smooth)
                / np.maximum(cv2.GaussianBlur(weight, (0, 0), args.smooth), 1e-3))
        gain = np.clip(gain, 1.0, args.gain_max)
        alpha = cv2.GaussianBlur(weight, (0, 0), args.feather)[..., None]
        out = np.clip(frame.astype(np.float32) * (1.0 + alpha * (gain[..., None] - 1.0)),
                      0, 255).astype(np.uint8)
        writer.write(out)

        before += float(ratio[correctable].mean())
        after += float((cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
                        / np.maximum(ref_gray, 1.0))[correctable].mean())
        if idx % 100 == 0:
            print(f"[frame] {idx}/{len(frames)}", flush=True)
    writer.release()
    n = len(frames)
    print(f"[info] desk brightness vs shadow-free reference: "
          f"{before / n:.4f} -> {after / n:.4f}")
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
