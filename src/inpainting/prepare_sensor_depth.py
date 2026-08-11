"""Align recorded 16-bit sensor depth to the processed RGB video.

The recorder stores RGB/depth timestamps separately and writes depth in
millimetres as 16-bit PNGs.  This utility:

1. pairs every RGB frame with its nearest depth timestamp,
2. rejects invalid/out-of-range sensor values,
3. applies a depth-to-RGB affine approximation in the unrotated camera view,
4. applies the same image rotation used by the RGB processing pipeline, and
5. writes a float16 metric-depth array for conservative occlusion gating.

Without camera intrinsics/extrinsics, the affine must not be treated as a
pixel-perfect reprojection.  Downstream code should sample only an eroded
object-mask interior and fail open when depth is unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _read_timestamps(path: Path) -> np.ndarray:
    values = np.asarray(
        [float(line.strip()) for line in path.read_text().splitlines() if line.strip()],
        dtype=np.float64,
    )
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"invalid timestamps: {path}")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"timestamps are not strictly increasing: {path}")
    return values


def _nearest_indices(source: np.ndarray, query: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source, query, side="left")
    right = np.clip(right, 0, len(source) - 1)
    left = np.clip(right - 1, 0, len(source) - 1)
    choose_left = np.abs(query - source[left]) <= np.abs(source[right] - query)
    return np.where(choose_left, left, right).astype(np.int32)


def _video_shape(path: Path) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open target video: {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if min(width, height, frames) <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError(
            f"invalid video metadata: {path} "
            f"({width}x{height}, frames={frames}, fps={fps})"
        )
    return width, height, frames, fps


def _rotated_shape(width: int, height: int, rotation: str) -> tuple[int, int]:
    if rotation in {"ccw", "cw"}:
        return height, width
    return width, height


def _rotate(image: np.ndarray, rotation: str) -> np.ndarray:
    if rotation == "none":
        return image
    if rotation == "ccw":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "cw":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "180":
        return cv2.rotate(image, cv2.ROTATE_180)
    raise ValueError(rotation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode_dir", type=Path, required=True)
    parser.add_argument("--processed_demo", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <processed_demo>/depth_processor/depth_sensor_aligned.npy",
    )
    parser.add_argument(
        "--rotation",
        choices=["none", "ccw", "cw", "180"],
        default="ccw",
        help="Rotation applied after unrotated depth-to-RGB alignment",
    )
    parser.add_argument(
        "--affine",
        type=float,
        nargs=4,
        metavar=("SX", "SY", "TX", "TY"),
        default=None,
        help=(
            "Unrotated depth-to-RGB map: "
            "x_rgb=SX*x_depth+TX, y_rgb=SY*y_depth+TY. "
            "Default is direct resolution scaling."
        ),
    )
    parser.add_argument("--min_depth_mm", type=float, default=200.0)
    parser.add_argument("--max_depth_mm", type=float, default=10000.0)
    args = parser.parse_args()

    episode = args.episode_dir.resolve()
    processed = args.processed_demo.resolve()
    target_video = processed / "video_L.mp4"
    target_width, target_height, frame_count, fps = _video_shape(target_video)

    rgb_timestamps = _read_timestamps(episode / "rgb_timestamps.txt")
    depth_timestamps = _read_timestamps(episode / "depth_timestamps.txt")
    depth_paths = sorted((episode / "depth_raw").glob("*.png"))
    if len(rgb_timestamps) != frame_count:
        raise ValueError(
            f"RGB timestamp/video mismatch: {len(rgb_timestamps)} != {frame_count}"
        )
    if len(depth_timestamps) != len(depth_paths):
        raise ValueError(
            "depth timestamp/image mismatch: "
            f"{len(depth_timestamps)} != {len(depth_paths)}"
        )
    mapping = _nearest_indices(depth_timestamps, rgb_timestamps)
    time_error_ms = (
        depth_timestamps[mapping] - rgb_timestamps
    ).astype(np.float64) * 1000.0

    first = cv2.imread(str(depth_paths[0]), cv2.IMREAD_UNCHANGED)
    if first is None or first.ndim != 2 or first.dtype != np.uint16:
        raise ValueError(
            f"expected uint16 depth PNG, got "
            f"{None if first is None else (first.shape, first.dtype)}"
        )
    depth_height, depth_width = first.shape

    if args.rotation in {"ccw", "cw"}:
        rgb_width, rgb_height = target_height, target_width
    else:
        rgb_width, rgb_height = target_width, target_height
    rotated_width, rotated_height = _rotated_shape(
        rgb_width, rgb_height, args.rotation
    )
    if (rotated_width, rotated_height) != (target_width, target_height):
        raise ValueError(
            "target dimensions are inconsistent with rotation: "
            f"unrotated={rgb_width}x{rgb_height}, rotation={args.rotation}, "
            f"target={target_width}x{target_height}"
        )

    if args.affine is None:
        sx = rgb_width / depth_width
        sy = rgb_height / depth_height
        tx = ty = 0.0
        alignment_mode = "resolution_scale"
    else:
        sx, sy, tx, ty = map(float, args.affine)
        alignment_mode = "user_affine"
    matrix = np.asarray([[sx, 0.0, tx], [0.0, sy, ty]], dtype=np.float32)

    output = (
        args.output.resolve()
        if args.output is not None
        else processed / "depth_processor" / "depth_sensor_aligned.npy"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    aligned = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=np.float16,
        shape=(frame_count, target_height, target_width),
    )
    valid_pixels = 0
    total_pixels = int(np.prod(aligned.shape))
    for frame_index, depth_index in enumerate(mapping):
        raw = cv2.imread(
            str(depth_paths[int(depth_index)]),
            cv2.IMREAD_UNCHANGED,
        )
        if raw is None or raw.shape != first.shape or raw.dtype != np.uint16:
            raise ValueError(
                f"invalid depth frame {depth_paths[int(depth_index)]}: "
                f"{None if raw is None else (raw.shape, raw.dtype)}"
            )
        metric = raw.astype(np.float32) * 0.001
        valid = (
            (raw >= args.min_depth_mm)
            & (raw <= args.max_depth_mm)
            & (raw != np.iinfo(np.uint16).max)
        )
        metric[~valid] = np.nan
        warped = cv2.warpAffine(
            metric,
            matrix,
            (rgb_width, rgb_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float("nan"),
        )
        warped = _rotate(warped, args.rotation)
        if warped.shape != (target_height, target_width):
            raise RuntimeError(
                f"aligned depth shape mismatch: {warped.shape} != "
                f"{(target_height, target_width)}"
            )
        valid_pixels += int(np.isfinite(warped).sum())
        aligned[frame_index] = warped.astype(np.float16)
    aligned.flush()

    report = {
        "schema_version": 1,
        "episode_dir": str(episode),
        "target_video": str(target_video),
        "output": str(output),
        "frames": frame_count,
        "width": target_width,
        "height": target_height,
        "fps": fps,
        "source_depth_width": depth_width,
        "source_depth_height": depth_height,
        "source_depth_unit": "millimetres",
        "output_depth_unit": "metres",
        "rotation": args.rotation,
        "alignment_mode": alignment_mode,
        "affine_depth_to_unrotated_rgb": matrix.tolist(),
        "min_depth_mm": args.min_depth_mm,
        "max_depth_mm": args.max_depth_mm,
        "unique_depth_frames_used": int(len(np.unique(mapping))),
        "timestamp_error_ms": {
            "min": float(time_error_ms.min()),
            "median": float(np.median(time_error_ms)),
            "max": float(time_error_ms.max()),
            "mean": float(time_error_ms.mean()),
        },
        "valid_pixel_fraction": valid_pixels / total_pixels,
        "limitations": [
            "Affine FOV alignment is not a calibrated 3-D reprojection.",
            "Use eroded object interiors and fail-open occlusion gating.",
        ],
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"[ok] sensor depth: {output} shape={aligned.shape} "
        f"valid={report['valid_pixel_fraction']:.3f} "
        f"time_error_ms={report['timestamp_error_ms']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
