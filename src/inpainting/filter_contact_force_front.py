"""Suppress untrusted reconstruction fragments in a contact front-layer mask.

Stage 3 already draws the complete interaction object.  The forced-front mask
only needs to redraw object pixels that a later robot layer would overwrite.
This filter therefore keeps the robot neighbourhood and accepts synthesized
pixels only near directly observed support from the same object component.
All thresholds are component-scale relative and shared across videos.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def filter_frame(force: np.ndarray, objects: np.ndarray,
                 predicted: np.ndarray, robot: np.ndarray,
                 robot_margin_px: int, min_observed_pixels: int,
                 min_observed_fraction: float,
                 prediction_reach_ratio: float) -> tuple[np.ndarray, int, int]:
    mask = np.asarray(force, dtype=bool).copy()
    robot_support = np.asarray(robot, dtype=np.uint8)
    if robot_margin_px > 0:
        radius = robot_margin_px
        robot_support = cv2.dilate(
            robot_support,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2),
            iterations=1,
        )
    before = int(mask.sum())
    mask &= robot_support.astype(bool)
    rejected_remote = before - int(mask.sum())

    objects = np.asarray(objects, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    observed = objects & ~predicted
    trusted = np.zeros_like(mask)
    _, components = cv2.connectedComponents(
        objects.astype(np.uint8), connectivity=8)
    for label in np.unique(components[mask]):
        if label == 0:
            continue
        component = components == label
        component_area = int(component.sum())
        observed_component = component & observed
        observed_area = int(observed_component.sum())
        if (observed_area < min_observed_pixels or
                observed_area < min_observed_fraction * component_area):
            continue
        equivalent_radius = np.sqrt(component_area / np.pi)
        max_distance = max(6.0, prediction_reach_ratio * equivalent_radius)
        distance = cv2.distanceTransform(
            (~observed_component).astype(np.uint8),
            cv2.DIST_L2, cv2.DIST_MASK_3,
        )
        trusted |= component & (
            ~predicted | (distance <= max_distance))
    before = int(mask.sum())
    mask &= trusted
    return mask, rejected_remote, before - int(mask.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force_mask", type=Path, required=True)
    parser.add_argument("--object_mask", type=Path, required=True)
    parser.add_argument("--prediction_mask", type=Path, required=True)
    parser.add_argument("--robot_mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot_margin_px", type=int, default=3)
    parser.add_argument("--min_observed_pixels", type=int, default=150)
    parser.add_argument("--min_observed_fraction", type=float, default=0.12)
    parser.add_argument("--prediction_reach_ratio", type=float, default=0.45)
    args = parser.parse_args()

    force = np.load(args.force_mask, mmap_mode="r")
    objects = np.load(args.object_mask, mmap_mode="r")
    predicted = np.load(args.prediction_mask, mmap_mode="r")
    robot = np.load(args.robot_mask, mmap_mode="r")
    frame_count = min(len(force), len(objects), len(predicted), len(robot))
    output = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=bool, shape=force.shape)
    output[:] = False
    rejected_remote = 0
    rejected_prediction = 0
    for frame_index in range(frame_count):
        result, remote, prediction = filter_frame(
            force[frame_index], objects[frame_index], predicted[frame_index],
            robot[frame_index], args.robot_margin_px,
            args.min_observed_pixels, args.min_observed_fraction,
            args.prediction_reach_ratio,
        )
        output[frame_index] = result
        rejected_remote += remote
        rejected_prediction += prediction
    output.flush()
    print(f"[ok] {args.output} frames={frame_count} pixels={int(output.sum())}")
    print(f"     confidence filter: remote={rejected_remote} "
          f"synthesized={rejected_prediction} px rejected")


if __name__ == "__main__":
    main()
