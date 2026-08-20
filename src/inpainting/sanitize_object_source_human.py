"""Remove every known source-human pixel from an object RGB layer.

The compositor may draw the object layer after the robot. Therefore source RGB
inside the union human mask is never allowed through unchanged. Object-supported
pixels are filled from the nearest directly observed non-human object texture;
everything else falls back to the already inpainted background plate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object_source", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--object_mask", type=Path, required=True)
    parser.add_argument("--force_front_mask", type=Path, required=True)
    parser.add_argument("--human_mask", type=Path, action="append", required=True,
                        help="Repeat to supply every available human mask; the "
                             "union is removed from the object source.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    object_mask = np.load(args.object_mask, mmap_mode="r")
    force_front = np.load(args.force_front_mask, mmap_mode="r")
    human_masks = [np.load(path, mmap_mode="r") for path in args.human_mask]
    source_cap = cv2.VideoCapture(str(args.object_source))
    background_cap = cv2.VideoCapture(str(args.background))
    if not source_cap.isOpened() or not background_cap.isOpened():
        raise RuntimeError("cannot open object source or background")
    width = int(source_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(source_cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = min(
        int(source_cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(object_mask),
        len(force_front), *(len(mask) for mask in human_masks),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"FFV1"), fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {args.output}")

    removed = filled_object = background_fallback = 0
    for frame_index in range(frame_count):
        ok_source, source = source_cap.read()
        ok_background, background = background_cap.read()
        if not ok_source or not ok_background:
            raise RuntimeError(f"decode stopped at frame {frame_index}")
        human = np.zeros((height, width), dtype=bool)
        for mask in human_masks:
            human |= np.asarray(mask[frame_index], dtype=bool)
        objects = np.asarray(object_mask[frame_index], dtype=bool)
        forced = np.asarray(force_front[frame_index], dtype=bool)
        result = source.copy()
        result[human] = background[human]
        target = human & (objects | forced)
        safe_object = objects & ~human
        if target.any() and safe_object.any():
            _, indices = distance_transform_edt(
                ~safe_object, return_distances=True, return_indices=True)
            nearest = source[indices[0], indices[1]]
            result[target] = nearest[target]
            filled_object += int(target.sum())
        fallback = human & ~target
        removed += int(human.sum())
        background_fallback += int(fallback.sum())
        writer.write(result)
        if (frame_index + 1) % 100 == 0:
            print(f"[frame] {frame_index + 1}/{frame_count}", flush=True)

    source_cap.release()
    background_cap.release()
    writer.release()
    print(f"[ok] {args.output}")
    print(f"     human removed={removed} object-filled={filled_object} "
          f"background-fallback={background_fallback}")


if __name__ == "__main__":
    main()
