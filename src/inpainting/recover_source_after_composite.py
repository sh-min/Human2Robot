"""Recover trustworthy source pixels after robot/object compositing.

This is deliberately a post-process.  It compares the final composite with the
original and the pre-robot reconstructed background, but operates *only* inside
the source-human mask plus the optional actual inpaint support.  Robot pixels
(with a safety margin) are immutable.

Two kinds of source pixels are restored:

1. visible manipulated-object pixels from the cleaned modal object mask; and
2. over-segmented hand-mask pixels whose original RGB already agrees with the
   clean reconstructed background and is not skin.

True hand pixels are never copied from the current source frame; they retain the
temporally reconstructed background/object content.  This prevents the common
failure mode where a final "copy source back" pass resurrects the human hand.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _skin_mask(frame: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    luminance, cr, cb = cv2.split(ycrcb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    return (
        (luminance >= 45) & (cr >= 132) & (cr <= 184)
        & (cb >= 72) & (cb <= 144) & (saturation >= 12)
        & (value >= 45)
    )


def _outline(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2
    )
    return cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel
    ).astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original_video", type=Path, required=True)
    parser.add_argument("--composite_video", type=Path, required=True)
    parser.add_argument("--background_video", type=Path, required=True)
    parser.add_argument("--human_mask", type=Path, required=True)
    parser.add_argument(
        "--inpaint_mask", type=Path, default=None,
        help="Optional actual inpaint support. It is unioned with the source "
             "human mask so expanded hand-boundary pixels can be repaired, "
             "while untouched regions remain immutable.",
    )
    parser.add_argument("--robot_mask", type=Path, required=True)
    parser.add_argument("--visible_object_mask", type=Path, required=True)
    parser.add_argument("--robot_dilate", type=int, default=3)
    parser.add_argument("--skin_dilate", type=int, default=2)
    parser.add_argument("--object_erode", type=int, default=1)
    parser.add_argument(
        "--agreement_threshold", type=float, default=17.0,
        help="Maximum CIE-Lab distance for direct background source recovery.",
    )
    parser.add_argument(
        "--minimum_change", type=float, default=2.5,
        help="Only copy source where final-vs-original Lab distance is at least "
             "this value; near-identical pixels are left untouched.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery_mask", type=Path, default=None)
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()

    human = np.load(args.human_mask, mmap_mode="r")
    inpaint = (np.load(args.inpaint_mask, mmap_mode="r")
               if args.inpaint_mask is not None else None)
    robot = np.load(args.robot_mask, mmap_mode="r")
    visible_objects = np.load(args.visible_object_mask, mmap_mode="r")
    original_cap = cv2.VideoCapture(str(args.original_video))
    composite_cap = cv2.VideoCapture(str(args.composite_video))
    background_cap = cv2.VideoCapture(str(args.background_video))
    if not all(cap.isOpened() for cap in
               (original_cap, composite_cap, background_cap)):
        raise RuntimeError("cannot open original/composite/background video")
    width = int(original_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(original_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(original_cap.get(cv2.CAP_PROP_FPS)) or 24.0
    frame_count = min(
        int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(composite_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(background_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        len(human), len(robot), len(visible_objects),
    )
    if inpaint is not None:
        frame_count = min(frame_count, len(inpaint))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"FFV1"), fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {args.output}")
    preview_writer = None
    if args.preview is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        preview_writer = cv2.VideoWriter(
            str(args.preview), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (width, height),
        )

    recovery_output = None
    if args.recovery_mask is not None:
        args.recovery_mask.parent.mkdir(parents=True, exist_ok=True)
        recovery_output = np.lib.format.open_memmap(
            args.recovery_mask, mode="w+", dtype=bool,
            shape=(frame_count, height, width),
        )

    robot_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max(0, args.robot_dilate) + 1,) * 2,
    )
    skin_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max(0, args.skin_dilate) + 1,) * 2,
    )
    object_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max(0, args.object_erode) + 1,) * 2,
    )
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    object_pixels = 0
    background_pixels = 0
    protected_robot_pixels = 0
    target_pixels = 0

    for frame_index in range(frame_count):
        ok_original, original = original_cap.read()
        ok_composite, composite = composite_cap.read()
        ok_background, background = background_cap.read()
        if not (ok_original and ok_composite and ok_background):
            raise RuntimeError(f"video decode stopped at frame {frame_index}")

        current_human = np.asarray(human[frame_index], dtype=bool)
        if inpaint is not None:
            current_human = current_human | np.asarray(
                inpaint[frame_index], dtype=bool
            )
        robot_safe = cv2.dilate(
            np.asarray(robot[frame_index], dtype=np.uint8),
            robot_kernel, iterations=1,
        ).astype(bool)
        target = current_human & ~robot_safe
        target_pixels += int(target.sum())
        protected_robot_pixels += int((current_human & robot_safe).sum())

        skin = cv2.dilate(
            _skin_mask(original).astype(np.uint8),
            skin_kernel, iterations=1,
        ).astype(bool)
        object_visible = np.asarray(
            visible_objects[frame_index], dtype=np.uint8
        )
        if args.object_erode > 0:
            object_visible = cv2.erode(
                object_visible, object_kernel, iterations=1
            )
        object_candidate = object_visible.astype(bool) & target

        original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
        background_lab = cv2.cvtColor(
            background, cv2.COLOR_BGR2LAB
        ).astype(np.float32)
        composite_lab = cv2.cvtColor(
            composite, cv2.COLOR_BGR2LAB
        ).astype(np.float32)
        delta = np.linalg.norm(original_lab - background_lab, axis=2)
        changed = (
            np.linalg.norm(original_lab - composite_lab, axis=2)
            >= args.minimum_change
        )
        object_restore = object_candidate & changed
        background_restore = (
            target & ~skin & ~object_restore
            & (delta <= args.agreement_threshold) & changed
        )
        # Remove isolated one-pixel decisions that would shimmer over time.
        background_restore = cv2.morphologyEx(
            background_restore.astype(np.uint8), cv2.MORPH_OPEN, open_kernel
        ).astype(bool)
        restore = object_restore | background_restore

        recovered = composite.copy()
        recovered[restore] = original[restore]
        writer.write(recovered)
        if recovery_output is not None:
            recovery_output[frame_index] = restore

        object_pixels += int(object_restore.sum())
        background_pixels += int(background_restore.sum())
        if preview_writer is not None:
            preview = composite.copy()
            remaining = target & ~restore
            preview[remaining] = (
                0.45 * preview[remaining]
                + 0.55 * np.array([25, 25, 230])
            ).astype(np.uint8)
            preview[background_restore] = (
                0.45 * original[background_restore]
                + 0.55 * np.array([25, 220, 235])
            ).astype(np.uint8)
            preview[object_restore] = (
                0.35 * original[object_restore]
                + 0.65 * np.array([35, 225, 35])
            ).astype(np.uint8)
            preview[_outline(robot_safe, 1)] = (235, 220, 25)
            preview_writer.write(preview)
        if (frame_index + 1) % 100 == 0:
            print(f"[frame] {frame_index + 1}/{frame_count}", flush=True)

    original_cap.release()
    composite_cap.release()
    background_cap.release()
    writer.release()
    if preview_writer is not None:
        preview_writer.release()
    if recovery_output is not None:
        recovery_output.flush()
    print(
        f"[info] source-hand target={target_pixels} px, "
        f"robot-protected={protected_robot_pixels} px, "
        f"source object recovered={object_pixels} px, "
        f"source background recovered={background_pixels} px"
    )
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
