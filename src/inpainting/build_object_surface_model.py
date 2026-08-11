"""Build an MH-camera object surface model from metric monocular depth.

This stage turns a dense scene-depth prediction plus the verified modal object
mask into the representation needed by the contact compositor:

* ``object_surface_depth.npy`` stores the visible object surface Z in the MH
  camera frame (zero means unknown/outside the modal object).
* ``object_surface_points.npy`` stores a deterministic, bounded point-cloud
  sample for every frame.  It is a diagnostic 3-D representation; the dense
  depth surface remains authoritative for pixel-accurate compositing.
* ``surface_stats.npz`` and ``report.json`` record robust filtering and camera
  assumptions so this model cannot be mistaken for calibrated sensor depth.

The input metric model may be anchored to HaWoR camera-space Z.  In that case
the result is an overlay-camera depth proxy, not independent metric ground
truth.  That distinction is intentional: it is still directly comparable to
the XHand renderer's camera-Z buffer, while ambiguous pixels stay unknown.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from atomic_directory_publish import publish_directory


@dataclass(frozen=True)
class SurfaceModelConfig:
    erode_px: int = 5
    bilateral_diameter: int = 5
    bilateral_sigma_depth_m: float = 0.025
    bilateral_sigma_space_px: float = 5.0
    trim_fraction: float = 0.02
    mad_scale: float = 6.0
    minimum_depth_band_m: float = 0.035
    maximum_depth_band_m: float = 0.30
    minimum_samples: int = 30
    minimum_depth_m: float = 0.05
    maximum_depth_m: float = 5.0
    point_count: int = 2048

    def validate(self) -> None:
        if self.erode_px < 0:
            raise ValueError("erode_px must be non-negative")
        if self.bilateral_diameter <= 0 or self.bilateral_diameter % 2 == 0:
            raise ValueError("bilateral_diameter must be a positive odd number")
        finite_positive = (
            self.bilateral_sigma_depth_m,
            self.bilateral_sigma_space_px,
            self.mad_scale,
            self.minimum_depth_band_m,
            self.maximum_depth_band_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in finite_positive):
            raise ValueError("surface filter scales must be finite and positive")
        if not 0.0 <= self.trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5)")
        if self.minimum_depth_band_m > self.maximum_depth_band_m:
            raise ValueError("minimum depth band must not exceed maximum")
        if self.minimum_samples <= 0 or self.point_count <= 0:
            raise ValueError("sample counts must be positive")
        if not 0.0 < self.minimum_depth_m < self.maximum_depth_m:
            raise ValueError("expected 0 < minimum_depth_m < maximum_depth_m")


def resize_depth(frame: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a positive camera-Z map without interpolating invalid zeros."""
    height, width = shape
    depth = np.asarray(frame, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth frame must be two-dimensional, got {depth.shape}")
    if depth.shape == (height, width):
        return depth.copy()
    valid = np.isfinite(depth) & (depth > 0.0)
    weighted = cv2.resize(
        np.where(valid, depth, 0.0),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    weights = cv2.resize(
        valid.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    out = np.zeros((height, width), dtype=np.float32)
    np.divide(weighted, weights, out=out, where=weights > 0.5)
    return out


def build_surface_frame(
    scene_depth: np.ndarray,
    object_mask: np.ndarray,
    *,
    output_shape: tuple[int, int],
    config: SurfaceModelConfig,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Return a robust visible object surface in the MH camera-Z frame."""
    config.validate()
    height, width = output_shape
    depth = resize_depth(scene_depth, output_shape)
    mask = np.asarray(object_mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"object mask must be two-dimensional, got {mask.shape}")
    if mask.shape != (height, width):
        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    mask = mask > 0
    valid_depth = (
        np.isfinite(depth)
        & (depth >= config.minimum_depth_m)
        & (depth <= config.maximum_depth_m)
    )
    if not np.any(mask & valid_depth):
        return np.zeros((height, width), dtype=np.float32), {
            "valid": False,
            "mask_pixels": int(mask.sum()),
            "trusted_samples": 0,
            "surface_pixels": 0,
            "median_depth_m": math.nan,
            "lower_depth_m": math.nan,
            "upper_depth_m": math.nan,
        }

    # Bilateral filtering reduces within-object shimmer while preserving the
    # depth discontinuity at the SAM modal boundary.
    fill_value = float(np.median(depth[mask & valid_depth]))
    # Never expose the bilateral kernel to a hand/table/background depth just
    # outside the modal boundary.  Unknown and non-object neighbours receive
    # the robust object centre, so the filter can smooth within the object
    # without importing another physical surface at contact edges.
    filter_input = np.where(
        mask & valid_depth,
        depth,
        fill_value,
    ).astype(np.float32)
    filtered = cv2.bilateralFilter(
        filter_input,
        config.bilateral_diameter,
        config.bilateral_sigma_depth_m,
        config.bilateral_sigma_space_px,
    )

    trusted = mask & valid_depth
    if config.erode_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * config.erode_px + 1, 2 * config.erode_px + 1),
        )
        eroded = cv2.erode(mask.astype(np.uint8), kernel) > 0
        if int((eroded & valid_depth).sum()) >= config.minimum_samples:
            trusted = eroded & valid_depth
    samples = filtered[trusted]
    if len(samples) < config.minimum_samples:
        return np.zeros((height, width), dtype=np.float32), {
            "valid": False,
            "mask_pixels": int(mask.sum()),
            "trusted_samples": int(len(samples)),
            "surface_pixels": 0,
            "median_depth_m": math.nan,
            "lower_depth_m": math.nan,
            "upper_depth_m": math.nan,
        }

    if config.trim_fraction > 0.0:
        quantile_low, quantile_high = np.quantile(
            samples,
            (config.trim_fraction, 1.0 - config.trim_fraction),
        )
        trimmed = samples[
            (samples >= quantile_low) & (samples <= quantile_high)
        ]
    else:
        quantile_low = float(samples.min())
        quantile_high = float(samples.max())
        trimmed = samples
    center = float(np.median(trimmed))
    mad = float(np.median(np.abs(trimmed - center)))
    robust_sigma = 1.4826 * mad
    half_band = float(np.clip(
        config.mad_scale * robust_sigma,
        config.minimum_depth_band_m,
        config.maximum_depth_band_m,
    ))
    lower = max(float(quantile_low), center - half_band)
    upper = min(float(quantile_high), center + half_band)
    if lower > upper:
        lower, upper = center - config.minimum_depth_band_m, center + config.minimum_depth_band_m

    surface_valid = (
        mask
        & valid_depth
        & np.isfinite(filtered)
        & (filtered >= lower)
        & (filtered <= upper)
    )
    if int(surface_valid.sum()) < config.minimum_samples:
        return np.zeros((height, width), dtype=np.float32), {
            "valid": False,
            "mask_pixels": int(mask.sum()),
            "trusted_samples": int(len(samples)),
            "surface_pixels": int(surface_valid.sum()),
            "median_depth_m": center,
            "lower_depth_m": lower,
            "upper_depth_m": upper,
        }
    surface = np.where(surface_valid, filtered, 0.0).astype(np.float32)
    return surface, {
        "valid": True,
        "mask_pixels": int(mask.sum()),
        "trusted_samples": int(len(samples)),
        "surface_pixels": int(surface_valid.sum()),
        "median_depth_m": center,
        "lower_depth_m": lower,
        "upper_depth_m": upper,
    }


def sample_surface_points(
    surface_depth: np.ndarray,
    *,
    focal_px: float,
    principal_point: tuple[float, float],
    point_count: int,
) -> tuple[np.ndarray, int]:
    """Back-project a deterministic bounded sample of one dense surface."""
    if not math.isfinite(focal_px) or focal_px <= 0.0:
        raise ValueError("focal_px must be finite and positive")
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    depth = np.asarray(surface_depth, dtype=np.float32)
    valid_flat = np.flatnonzero(np.isfinite(depth) & (depth > 0.0))
    out = np.full((point_count, 3), np.nan, dtype=np.float32)
    if not len(valid_flat):
        return out, 0
    if len(valid_flat) > point_count:
        sample_indices = np.linspace(
            0,
            len(valid_flat) - 1,
            point_count,
            dtype=np.int64,
        )
        valid_flat = valid_flat[sample_indices]
    y, x = np.unravel_index(valid_flat, depth.shape)
    z = depth[y, x]
    cx, cy = principal_point
    points = np.column_stack(
        ((x.astype(np.float32) - cx) * z / focal_px,
         (y.astype(np.float32) - cy) * z / focal_px,
         z)
    )
    out[:len(points)] = points
    return out, int(len(points))


def _debug_frame(frame: np.ndarray, surface: np.ndarray) -> np.ndarray:
    base = np.asarray(frame, dtype=np.uint8)
    valid = np.isfinite(surface) & (surface > 0.0)
    if not valid.any():
        return base
    values = surface[valid]
    low, high = np.quantile(values, (0.05, 0.95))
    denominator = max(float(high - low), 1.0e-5)
    normalized = np.zeros(surface.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        255.0 * (surface[valid] - low) / denominator,
        0.0,
        255.0,
    ).astype(np.uint8)
    colors = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    out = base.copy()
    out[valid] = np.clip(
        0.35 * base[valid].astype(np.float32)
        + 0.65 * colors[valid].astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_depth", type=Path, required=True)
    parser.add_argument("--object_mask", type=Path, required=True)
    parser.add_argument("--hawor_npz", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--erode_px", type=int, default=SurfaceModelConfig.erode_px)
    parser.add_argument("--point_count", type=int, default=SurfaceModelConfig.point_count)
    parser.add_argument("--minimum_samples", type=int, default=SurfaceModelConfig.minimum_samples)
    args = parser.parse_args()

    config = SurfaceModelConfig(
        erode_px=args.erode_px,
        point_count=args.point_count,
        minimum_samples=args.minimum_samples,
    )
    config.validate()
    scene_path = args.scene_depth.resolve()
    mask_path = args.object_mask.resolve()
    hawor_path = args.hawor_npz.resolve()
    out_dir = args.out_dir.resolve()
    depth = np.load(scene_path, mmap_mode="r")
    mask = np.load(mask_path, mmap_mode="r")
    if depth.ndim != 3 or mask.ndim != 3 or len(depth) != len(mask):
        raise ValueError("scene depth and object mask must be aligned (T,H,W)")
    frame_count, height, width = depth.shape
    with np.load(hawor_path) as hawor:
        focal_px = float(hawor["img_focal"])
    if not math.isfinite(focal_px) or focal_px <= 0.0:
        raise ValueError("invalid HaWoR image focal")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".object_surface.", dir=out_dir.parent))
    atexit.register(shutil.rmtree, staging, ignore_errors=True)
    surface_out = np.lib.format.open_memmap(
        staging / "object_surface_depth.npy",
        mode="w+",
        dtype=np.float16,
        shape=(frame_count, height, width),
    )
    points_out = np.lib.format.open_memmap(
        staging / "object_surface_points.npy",
        mode="w+",
        dtype=np.float16,
        shape=(frame_count, config.point_count, 3),
    )
    stats: dict[str, np.ndarray] = {
        "valid": np.zeros(frame_count, dtype=bool),
        "mask_pixels": np.zeros(frame_count, dtype=np.int64),
        "trusted_samples": np.zeros(frame_count, dtype=np.int64),
        "surface_pixels": np.zeros(frame_count, dtype=np.int64),
        "point_samples": np.zeros(frame_count, dtype=np.int32),
        "median_depth_m": np.full(frame_count, np.nan, dtype=np.float32),
        "lower_depth_m": np.full(frame_count, np.nan, dtype=np.float32),
        "upper_depth_m": np.full(frame_count, np.nan, dtype=np.float32),
    }
    capture = None
    writer = None
    if args.video is not None:
        capture = cv2.VideoCapture(str(args.video.resolve()))
        if not capture.isOpened():
            raise FileNotFoundError(args.video)
        video_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if video_frames != frame_count:
            raise ValueError(
                f"debug video frame count {video_frames} != depth {frame_count}"
            )
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
        writer = cv2.VideoWriter(
            str(staging / "video_object_surface_3d.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("could not open object-surface debug writer")
    try:
        for frame_index in range(frame_count):
            surface, frame_stats = build_surface_frame(
                depth[frame_index],
                mask[frame_index],
                output_shape=(height, width),
                config=config,
            )
            surface_out[frame_index] = surface.astype(np.float16)
            points, point_samples = sample_surface_points(
                surface,
                focal_px=focal_px,
                principal_point=((width - 1) / 2.0, (height - 1) / 2.0),
                point_count=config.point_count,
            )
            points_out[frame_index] = points.astype(np.float16)
            for key in stats:
                if key == "point_samples":
                    stats[key][frame_index] = point_samples
                else:
                    stats[key][frame_index] = frame_stats[key]
            if capture is not None and writer is not None:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"debug video read failed at {frame_index}")
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(_debug_frame(frame, surface))
            if (frame_index + 1) % 100 == 0:
                print(f"[object-surface] {frame_index + 1}/{frame_count}", flush=True)
    finally:
        surface_out.flush()
        points_out.flush()
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()

    np.savez(staging / "surface_stats.npz", **stats)
    valid_frames = stats["valid"]
    report = {
        "schema_version": 1,
        "representation": {
            "dense": "visible modal-object camera-Z surface",
            "sparse": "deterministic back-projected point-cloud sample",
            "not_watertight_mesh": True,
            "outside_or_unknown_depth_value": 0.0,
        },
        "frames": int(frame_count),
        "height": int(height),
        "width": int(width),
        "config": asdict(config),
        "camera": {
            "coordinate_frame": "MH HaWoR/overlay camera",
            "focal_px": focal_px,
            "principal_point_assumption": [(width - 1) / 2.0, (height - 1) / 2.0],
            "calibrated_phone_intrinsics": False,
        },
        "sources": {
            "scene_depth": str(scene_path),
            "object_mask": str(mask_path),
            "hawor_npz": str(hawor_path),
            "debug_video": str(args.video.resolve()) if args.video is not None else None,
        },
        "provenance_warning": (
            "Depth Anything V2 was scaled with HaWoR camera-Z anchors; this is "
            "an overlay-coordinate depth proxy, not independent sensor depth."
        ),
        "valid_frames": int(valid_frames.sum()),
        "frames_with_mask": int((stats["mask_pixels"] > 0).sum()),
        "surface_pixels_total": int(stats["surface_pixels"].sum()),
        "median_surface_depth_m": (
            float(np.nanmedian(stats["median_depth_m"][valid_frames]))
            if valid_frames.any() else None
        ),
    }
    (staging / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    del surface_out, points_out
    publish_directory(staging, out_dir)
    print(
        f"[ok] {out_dir}: valid={int(valid_frames.sum())}/{frame_count}, "
        f"surface_pixels={int(stats['surface_pixels'].sum())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
