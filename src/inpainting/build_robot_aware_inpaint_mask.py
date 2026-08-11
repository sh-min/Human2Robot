"""Limit human inpainting to pixels not safely covered by the final robot."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human_mask", type=Path, required=True)
    parser.add_argument("--robot_mask", type=Path, required=True)
    parser.add_argument("--object_mask", type=Path, default=None)
    parser.add_argument(
        "--object_dilate", type=int, default=1,
        help="Protect this margin around reconstructed object pixels from the "
             "residual inpainting mask.",
    )
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--robot_erode", type=int, default=5)
    parser.add_argument("--residual_dilate", type=int, default=1)
    parser.add_argument("--output_mask", type=Path, required=True)
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()

    human = np.load(args.human_mask, mmap_mode="r")
    robot = np.load(args.robot_mask, mmap_mode="r")
    objects = (np.load(args.object_mask, mmap_mode="r")
               if args.object_mask is not None else None)
    frame_count = min(len(human), len(robot))
    if objects is not None:
        frame_count = min(frame_count, len(objects))
    height, width = human.shape[1:]
    output = np.zeros((frame_count, height, width), dtype=bool)
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max(0, args.robot_erode) + 1,) * 2,
    )
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max(0, args.residual_dilate) + 1,) * 2,
    )
    original_pixels = 0
    skipped_pixels = 0
    output_pixels = 0
    object_skipped_pixels = 0
    object_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max(0, args.object_dilate) + 1,) * 2,
    )
    for index in range(frame_count):
        current_human = np.asarray(human[index], dtype=bool)
        current_robot = np.asarray(robot[index], dtype=np.uint8)
        robot_core = cv2.erode(
            current_robot, erode_kernel, iterations=1
        ).astype(bool)
        residual = current_human & ~robot_core
        if args.residual_dilate > 0:
            residual = cv2.dilate(
                residual.astype(np.uint8), dilate_kernel, iterations=1
            ).astype(bool)
            residual &= ~robot_core
        if objects is not None:
            protected_object = cv2.dilate(
                np.asarray(objects[index], dtype=np.uint8),
                object_kernel, iterations=1,
            ).astype(bool)
            object_skipped_pixels += int((residual & protected_object).sum())
            # Reconstructed object texture already supplies these pixels.  They
            # must not be sent to the background inpainter (or shown red in the
            # diagnostic preview), otherwise the object appears hollow at grip.
            residual &= ~protected_object
        output[index] = residual
        original_pixels += int(current_human.sum())
        skipped_pixels += int((current_human & robot_core).sum())
        output_pixels += int(residual.sum())

    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_mask, output)

    if args.preview is not None and args.video is not None:
        cap = cv2.VideoCapture(str(args.video))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
        writer = cv2.VideoWriter(
            str(args.preview), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (width, height),
        )
        for index in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                break
            overlay = frame.copy()
            mask = output[index]
            overlay[mask] = (
                0.35 * overlay[mask] + 0.65 * np.array([30, 30, 240])
            ).astype(np.uint8)
            robot_current = np.asarray(robot[index], dtype=bool)
            outline = cv2.morphologyEx(
                robot_current.astype(np.uint8), cv2.MORPH_GRADIENT,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            ).astype(bool)
            overlay[outline] = (240, 220, 20)
            if objects is not None:
                object_outline = cv2.morphologyEx(
                    np.asarray(objects[index], dtype=np.uint8),
                    cv2.MORPH_GRADIENT,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                ).astype(bool)
                overlay[object_outline] = (20, 230, 20)
            writer.write(overlay)
        cap.release()
        writer.release()

    reduction = 100.0 * (1.0 - output_pixels / max(1, original_pixels))
    print(
        f"[info] human={original_pixels} px, robot-covered skipped={skipped_pixels} px, "
        f"object-restored skipped={object_skipped_pixels} px, "
        f"inpaint={output_pixels} px, reduction={reduction:.1f}%"
    )
    print(f"[ok] wrote {args.output_mask}")


if __name__ == "__main__":
    main()
