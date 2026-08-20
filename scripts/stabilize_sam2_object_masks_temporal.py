#!/usr/bin/env python3
"""Repair short SAM2 object-mask collapses without video-specific tuning.

The detector/tracker output remains authoritative on normal frames.  Within a
single object track, abrupt area collapses are repaired by warping the nearest
reliable masks from both temporal directions.  RGB is deliberately not copied
between frames: downstream compositing reads current-frame source RGB through
the repaired mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


EXPECTED_METHOD = (
    "full-sequence HaCo peak discovery + HaWoR prompts + bidirectional SAM2"
)
EXPECTED_PARAMETERS = {
    "smooth_sigma": 5.0,
    "min_peak_distance": 45,
    "peak_prominence": 35.0,
    "max_objects": 8,
}


def local_track_median(
    areas: np.ndarray, track_ids: np.ndarray, radius: int
) -> np.ndarray:
    result = np.zeros_like(areas, dtype=np.float64)
    for index, track_id in enumerate(track_ids):
        lo = max(0, index - radius)
        hi = min(len(areas), index + radius + 1)
        same_track = track_ids[lo:hi] == track_id
        values = areas[lo:hi][same_track]
        result[index] = np.median(values) if len(values) else areas[index]
    return result


def contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if not len(indices):
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for value in map(int, indices[1:]):
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def expand_bad_runs(
    bad: np.ndarray,
    ratios: np.ndarray,
    track_ids: np.ndarray,
    recovery_ratio: float,
) -> np.ndarray:
    expanded = bad.copy()
    for start, end in contiguous_runs(np.flatnonzero(bad)):
        track_id = track_ids[start]
        left = start - 1
        while (
            left >= 0
            and track_ids[left] == track_id
            and ratios[left] < recovery_ratio
        ):
            expanded[left] = True
            left -= 1
        right = end + 1
        while (
            right < len(expanded)
            and track_ids[right] == track_id
            and ratios[right] < recovery_ratio
        ):
            expanded[right] = True
            right += 1
    return expanded


def read_gray(capture: cv2.VideoCapture, index: int, scale: float) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"video read failed at frame {index}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        gray = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    return gray


def warp_mask(
    source_mask: np.ndarray,
    source_gray: np.ndarray,
    target_gray: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    # Backward flow gives, for every target pixel, the source coordinate that
    # should be sampled by remap.
    flow = cv2.calcOpticalFlowFarneback(
        target_gray,
        source_gray,
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=25,
        iterations=4,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    height, width = target_gray.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    warped = cv2.remap(
        source_mask.astype(np.uint8),
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return cv2.resize(warped, output_size, interpolation=cv2.INTER_NEAREST).astype(bool)


def propagate(
    capture: cv2.VideoCapture,
    anchor: int,
    targets: list[int],
    anchor_mask: np.ndarray,
    scale: float,
) -> dict[int, np.ndarray]:
    if not targets:
        return {}
    output: dict[int, np.ndarray] = {}
    height, width = anchor_mask.shape
    current_index = anchor
    current_gray = read_gray(capture, current_index, scale)
    current_mask = cv2.resize(
        anchor_mask.astype(np.uint8),
        (current_gray.shape[1], current_gray.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    step = 1 if targets[0] > anchor else -1
    wanted = set(targets)
    last = targets[-1]
    while current_index != last:
        next_index = current_index + step
        next_gray = read_gray(capture, next_index, scale)
        full = warp_mask(current_mask, current_gray, next_gray, (width, height))
        current_mask = cv2.resize(
            full.astype(np.uint8),
            (next_gray.shape[1], next_gray.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        current_gray = next_gray
        current_index = next_index
        if current_index in wanted:
            output[current_index] = full
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--track_ids", type=Path, required=True)
    parser.add_argument("--segmentation_report", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--window_radius", type=int, default=5)
    parser.add_argument("--collapse_ratio", type=float, default=0.20)
    parser.add_argument("--recovery_ratio", type=float, default=0.75)
    parser.add_argument("--absolute_min_area", type=int, default=2000)
    parser.add_argument("--flow_scale", type=float, default=0.5)
    parser.add_argument(
        "--support_dilate_px",
        type=int,
        default=45,
        help=(
            "Limit flow-added pixels to a fixed neighborhood of the current "
            "SAM2 observation, preventing propagation into hands/background."
        ),
    )
    args = parser.parse_args()

    provenance = json.loads(args.segmentation_report.read_text())
    if provenance.get("method") != EXPECTED_METHOD:
        raise ValueError("mask is not from the approved in-house SAM2 pipeline")
    if provenance.get("parameters") != EXPECTED_PARAMETERS:
        raise ValueError("segmentation parameters differ from the fixed baseline")

    masks = np.load(args.mask, mmap_mode="r")
    track_ids = np.load(args.track_ids)
    if len(masks) != len(track_ids):
        raise ValueError("mask/track frame count mismatch")
    areas = masks.reshape(len(masks), -1).sum(axis=1).astype(np.float64)
    median = local_track_median(areas, track_ids, args.window_radius)
    ratios = areas / np.maximum(median, 1.0)
    bad = (ratios < args.collapse_ratio) | (areas < args.absolute_min_area)
    bad &= track_ids > 0
    repair = expand_bad_runs(bad, ratios, track_ids, args.recovery_ratio)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "object_mask_modal_temporal.npy"
    stabilized = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=bool, shape=masks.shape
    )
    stabilized[:] = masks[:]
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(args.video)

    repair_records = []
    try:
        for start, end in contiguous_runs(np.flatnonzero(repair)):
            track_id = int(track_ids[start])
            targets = list(range(start, end + 1))
            previous = start - 1
            while previous >= 0 and (
                track_ids[previous] != track_id or repair[previous]
            ):
                previous -= 1
            following = end + 1
            while following < len(masks) and (
                track_ids[following] != track_id or repair[following]
            ):
                following += 1
            if previous < 0 or track_ids[previous] != track_id:
                previous = None
            if following >= len(masks) or track_ids[following] != track_id:
                following = None
            forward = (
                propagate(
                    capture, previous, targets,
                    np.asarray(masks[previous], dtype=bool), args.flow_scale,
                )
                if previous is not None else {}
            )
            backward = (
                propagate(
                    capture, following, list(reversed(targets)),
                    np.asarray(masks[following], dtype=bool), args.flow_scale,
                )
                if following is not None else {}
            )
            for index in targets:
                observed = np.asarray(masks[index], dtype=bool)
                candidates = [observed]
                if index in forward:
                    candidates.append(forward[index])
                if index in backward:
                    candidates.append(backward[index])
                propagated = np.logical_or.reduce(candidates)
                if observed.any() and args.support_dilate_px > 0:
                    radius = args.support_dilate_px
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
                    )
                    support = cv2.dilate(
                        observed.astype(np.uint8), kernel
                    ).astype(bool)
                    propagated = observed | (propagated & support)
                stabilized[index] = propagated
            repair_records.append({
                "start": start,
                "end": end,
                "track_id": track_id,
                "previous_anchor": previous,
                "following_anchor": following,
            })
    finally:
        capture.release()
        stabilized.flush()

    repaired_areas = stabilized.reshape(len(stabilized), -1).sum(axis=1)
    report = {
        "schema_version": 1,
        "method": "in-house SAM2 plus bidirectional optical-flow mask persistence",
        "source_segmentation_method": EXPECTED_METHOD,
        "per_video_tuning": False,
        "parameters": {
            "window_radius": args.window_radius,
            "collapse_ratio": args.collapse_ratio,
            "recovery_ratio": args.recovery_ratio,
            "absolute_min_area": args.absolute_min_area,
            "flow_scale": args.flow_scale,
            "support_dilate_px": args.support_dilate_px,
        },
        "detected_collapse_frames": np.flatnonzero(bad).tolist(),
        "repaired_frames": np.flatnonzero(repair).tolist(),
        "repair_runs": repair_records,
        "pixels_added": int(np.maximum(repaired_areas - areas, 0).sum()),
        "output": str(output_path.resolve()),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
