"""Restore a conservative completed-object margin onto a finished composite.

Only pixels newly introduced by an expanded object mask are copied. Robot
pixels and a configurable safety margin remain immutable, so this pass cannot
paint the restored object over visible robot fingers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composite_video", type=Path, required=True)
    parser.add_argument("--object_video", type=Path, required=True)
    parser.add_argument("--base_object_mask", type=Path, required=True)
    parser.add_argument("--expanded_object_mask", type=Path, required=True)
    parser.add_argument("--robot_mask", type=Path, required=True)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--robot_dilate", type=int, default=2)
    parser.add_argument(
        "--sponge_colors_only", action="store_true",
        help="Keep only green/yellow sponge texture in the restored margin.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = np.load(args.base_object_mask, mmap_mode="r")
    expanded = np.load(args.expanded_object_mask, mmap_mode="r")
    robot = np.load(args.robot_mask, mmap_mode="r")
    composite_cap = cv2.VideoCapture(str(args.composite_video))
    object_cap = cv2.VideoCapture(str(args.object_video))
    if not composite_cap.isOpened() or not object_cap.isOpened():
        raise RuntimeError("cannot open composite/object video")
    width = int(composite_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(composite_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(composite_cap.get(cv2.CAP_PROP_FPS)) or 24.0
    frame_count = min(
        int(composite_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(object_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        len(base), len(expanded), len(robot),
    )
    end_frame = frame_count - 1 if args.end_frame is None else min(
        args.end_frame, frame_count - 1
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * max(0, args.robot_dilate) + 1,) * 2
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"FFV1"), fps,
        (width, height),
    )
    restored_pixels = 0
    for frame_index in range(frame_count):
        ok_composite, composite = composite_cap.read()
        ok_object, object_frame = object_cap.read()
        if not ok_composite or not ok_object:
            raise RuntimeError(f"video decode stopped at frame {frame_index}")
        if args.start_frame <= frame_index <= end_frame:
            robot_safe = cv2.dilate(
                np.asarray(robot[frame_index], dtype=np.uint8), kernel,
                iterations=1,
            ).astype(bool)
            addition = (
                np.asarray(expanded[frame_index], dtype=bool)
                & ~np.asarray(base[frame_index], dtype=bool)
                & ~robot_safe
            )
            if args.sponge_colors_only:
                hsv = cv2.cvtColor(object_frame, cv2.COLOR_BGR2HSV)
                hue, saturation, value = cv2.split(hsv)
                green = ((hue >= 32) & (hue <= 100)
                         & (saturation >= 25) & (value >= 20))
                yellow = ((hue >= 15) & (hue <= 42)
                          & (saturation >= 50) & (value >= 65))
                addition &= green | yellow
            composite[addition] = object_frame[addition]
            restored_pixels += int(addition.sum())
        writer.write(composite)
        if (frame_index + 1) % 100 == 0:
            print(f"[frame] {frame_index + 1}/{frame_count}", flush=True)
    composite_cap.release()
    object_cap.release()
    writer.release()
    print(f"[info] restored object-margin pixels={restored_pixels}")
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
