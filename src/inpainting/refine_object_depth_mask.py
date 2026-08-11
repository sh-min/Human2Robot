"""Build a contact-local object mask for robust sensor-depth sampling.

The annotation-driven SAM2 mask is intentionally generous so an object is not
removed during hand inpainting.  A generous mask can include background
structures touching the object, which biases its median depth.  This utility
keeps the original mask unchanged and creates a second, depth-coherent mask:

* sample sensor depth only near projected HaWoR fingertips/pinch points,
* estimate the local held-object depth robustly, and
* retain SAM pixels within a conservative metric tolerance of that depth.

The output is for ``--object_depth_mask`` only.  The original modal SAM mask
should remain ``--object_mask`` for visual layering and inpaint protection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


FINGERTIPS = np.asarray([4, 8, 12, 16, 20], dtype=np.int32)


def _support(points: np.ndarray, shape: tuple[int, int], radius: int) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for point in np.asarray(points, dtype=np.float32):
        if not np.isfinite(point).all():
            continue
        x, y = np.round(point).astype(np.int32)
        if -radius <= x < width + radius and -radius <= y < height + radius:
            cv2.circle(mask, (int(x), int(y)), radius, 1, thickness=-1)
    return mask.astype(bool)


def _hand_prompt_points(data: np.lib.npyio.NpzFile, frame: int) -> np.ndarray:
    joints = np.asarray(data["kpts_2d"][frame], dtype=np.float32)
    tips = joints[FINGERTIPS]
    return np.concatenate(
        [
            tips,
            tips.mean(axis=0, keepdims=True),
            ((joints[4] + joints[8]) / 2.0)[None],
        ],
        axis=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_demo", type=Path, required=True)
    parser.add_argument("--object_mask", type=Path, default=None)
    parser.add_argument("--scene_depth", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--support_radius_px", type=int, default=35)
    parser.add_argument("--depth_tolerance_m", type=float, default=0.10)
    parser.add_argument("--min_support_samples", type=int, default=80)
    parser.add_argument("--close_px", type=int, default=5)
    args = parser.parse_args()

    processed = args.processed_demo.resolve()
    object_path = (
        args.object_mask.resolve()
        if args.object_mask is not None
        else processed / "object_layer" / "object_mask_modal.npy"
    )
    depth_path = (
        args.scene_depth.resolve()
        if args.scene_depth is not None
        else processed / "depth_processor" / "depth_sensor_aligned.npy"
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else processed / "object_layer" / "object_depth_mask_sensor.npy"
    )

    object_mask = np.load(object_path, mmap_mode="r")
    scene_depth = np.load(depth_path, mmap_mode="r")
    if object_mask.ndim != 3 or scene_depth.shape != object_mask.shape:
        raise ValueError(
            f"mask/depth mismatch: {object_mask.shape} vs {scene_depth.shape}"
        )
    frame_count, height, width = object_mask.shape

    hands: dict[str, np.lib.npyio.NpzFile] = {}
    for side in ("left", "right"):
        path = processed / "hand_processor" / f"hand_data_{side}.npz"
        if path.is_file():
            data = np.load(path)
            if len(data["hand_detected"]) != frame_count:
                data.close()
                raise ValueError(f"{side} hand/frame mismatch")
            hands[side] = data
    if not hands:
        raise FileNotFoundError("hand_processor/hand_data_{left,right}.npz")

    output.parent.mkdir(parents=True, exist_ok=True)
    refined = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=np.bool_,
        shape=object_mask.shape,
    )
    centers = np.full(frame_count, np.nan, dtype=np.float32)
    support_samples = np.zeros(frame_count, dtype=np.int32)
    retained_pixels = np.zeros(frame_count, dtype=np.int32)
    original_pixels = np.zeros(frame_count, dtype=np.int32)
    close_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * args.close_px + 1, 2 * args.close_px + 1),
        )
        if args.close_px > 0
        else None
    )
    try:
        for frame in range(frame_count):
            modal = np.asarray(object_mask[frame], dtype=bool)
            original_pixels[frame] = int(modal.sum())
            if not modal.any():
                refined[frame] = False
                continue
            depth = np.asarray(scene_depth[frame], dtype=np.float32)
            valid = np.isfinite(depth) & (depth > 0.02) & (depth < 5.0)
            points = []
            for data in hands.values():
                if bool(data["hand_detected"][frame]):
                    points.append(_hand_prompt_points(data, frame))
            if points:
                local_support = _support(
                    np.concatenate(points, axis=0),
                    (height, width),
                    args.support_radius_px,
                )
                samples = depth[modal & local_support & valid]
            else:
                samples = np.empty(0, dtype=np.float32)
            support_samples[frame] = len(samples)

            if len(samples) >= args.min_support_samples:
                lo, hi = np.quantile(samples, (0.1, 0.9))
                samples = samples[(samples >= lo) & (samples <= hi)]
                center = float(np.median(samples))
                centers[frame] = center
                keep = (
                    modal
                    & valid
                    & (np.abs(depth - center) <= args.depth_tolerance_m)
                )
            else:
                # No trustworthy hand-local depth: retain all valid modal
                # pixels.  Downstream erosion/trimmed median remains fail-open.
                keep = modal & valid
            if close_kernel is not None:
                keep = cv2.morphologyEx(
                    keep.astype(np.uint8),
                    cv2.MORPH_CLOSE,
                    close_kernel,
                ).astype(bool)
                keep &= modal
            retained_pixels[frame] = int(keep.sum())
            refined[frame] = keep
    finally:
        for data in hands.values():
            data.close()
        refined.flush()

    active = original_pixels > 0
    valid_center = np.isfinite(centers)
    ratios = retained_pixels[active] / np.maximum(original_pixels[active], 1)
    report = {
        "schema_version": 1,
        "processed_demo": str(processed),
        "object_mask": str(object_path),
        "scene_depth": str(depth_path),
        "output": str(output),
        "frames": frame_count,
        "support_radius_px": args.support_radius_px,
        "depth_tolerance_m": args.depth_tolerance_m,
        "min_support_samples": args.min_support_samples,
        "frames_with_object": int(active.sum()),
        "frames_with_contact_local_depth": int(valid_center.sum()),
        "retained_fraction": {
            "p05": float(np.quantile(ratios, 0.05)) if len(ratios) else 0.0,
            "median": float(np.median(ratios)) if len(ratios) else 0.0,
            "p95": float(np.quantile(ratios, 0.95)) if len(ratios) else 0.0,
        },
        "object_depth_m": [
            float(value) if np.isfinite(value) else None for value in centers
        ],
        "support_sample_count": support_samples.tolist(),
        "original_pixel_count": original_pixels.tolist(),
        "retained_pixel_count": retained_pixels.tolist(),
        "usage": {
            "object_mask": str(object_path),
            "object_depth_mask": str(output),
        },
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"[ok] object depth mask: {output}; "
        f"local_depth={valid_center.sum()}/{active.sum()} object frames; "
        f"retained median={report['retained_fraction']['median']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
