#!/usr/bin/env python3
"""Register raw RealSense depth into the case camera-2 RGB image plane.

The converted stereo episodes intentionally keep the original depth images.
They are *not* pixel-aligned with RGB.  This module performs the missing
metric operation:

1. deproject each valid raw depth pixel in its depth-camera frame,
2. apply the calibrated rigid transform,
3. project into camera 2's colour image, and
4. retain the nearest sample at every RGB pixel (a metric z-buffer).

By default only camera 2 depth is registered to camera 2 RGB.  Passing
``--include-camera-1`` additionally transforms camera 1 depth through its
factory depth-to-colour extrinsic and the refined colour1-to-colour2 stereo
extrinsic.  ``--include-camera-1-native`` additionally emits camera 1 depth
in the native camera 1 RGB plane, while ``--only-camera-1-native`` avoids
rewriting the larger camera 2 products when only that semantic workspace is
needed.  The factory RealSense ``rotation=...`` array is parsed in the
column-major layout used by ``rs2_extrinsics``; treating it as row-major
silently applies the inverse-ish rotation and is a common source of error.

The output depth arrays contain metres as float32 and use NaN for missing
pixels.  A separate boolean valid mask is written for consumers that should
not depend on the NaN convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCRIPT_VERSION = 1


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in the unrotated sensor image convention."""

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    model: str
    coeffs: tuple[float, ...]

    @classmethod
    def from_manifest(cls, payload: dict[str, Any]) -> "Intrinsics":
        return cls(
            width=int(payload["width"]),
            height=int(payload["height"]),
            fx=float(payload["fx"]),
            fy=float(payload["fy"]),
            ppx=float(payload["ppx"]),
            ppy=float(payload["ppy"]),
            model=str(payload.get("model", "")),
            coeffs=tuple(float(value) for value in payload.get("coeffs", ())),
        )


def _parse_semicolon_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in text.rstrip("\x00;").split(";"):
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def parse_rs2_depth_to_color_transform(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a RealSense factory depth->colour rigid transform.

    ``rs2_extrinsics.rotation`` is serialized as nine consecutive values in
    column-major order.  NumPy therefore needs ``order="F"`` here.

    Returns matrices satisfying ``X_color = R_color_from_depth @ X_depth + t``.
    """

    fields = _parse_semicolon_fields(text)
    try:
        rotation_values = [float(value) for value in fields["rotation"].split(",")]
        translation_values = [
            float(value) for value in fields["translation"].split(",")
        ]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid RealSense extrinsic string: {text!r}") from exc
    if len(rotation_values) != 9 or len(translation_values) != 3:
        raise ValueError(
            "RealSense extrinsic must contain 9 rotation and 3 translation values"
        )

    rotation = np.asarray(rotation_values, dtype=np.float64).reshape(
        3, 3, order="F"
    )
    translation = np.asarray(translation_values, dtype=np.float64)
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("RealSense extrinsic contains non-finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=5.0e-4):
        raise ValueError("RealSense extrinsic rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=5.0e-4):
        raise ValueError("RealSense extrinsic rotation is not a proper rotation")
    return rotation, translation


def compose_rigid_transforms(
    rotation_target_from_middle: np.ndarray,
    translation_target_from_middle: np.ndarray,
    rotation_middle_from_source: np.ndarray,
    translation_middle_from_source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose ``source -> middle -> target`` rigid transforms."""

    rotation = rotation_target_from_middle @ rotation_middle_from_source
    translation = (
        rotation_target_from_middle @ translation_middle_from_source
        + translation_target_from_middle
    )
    return rotation, translation


def _validate_intrinsic_dimensions(intrinsics: Intrinsics, *, role: str) -> None:
    if intrinsics.width <= 0 or intrinsics.height <= 0:
        raise ValueError(f"{role} has invalid image dimensions: {intrinsics}")
    if intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0:
        raise ValueError(f"{role} has invalid focal length: {intrinsics}")


def _normalise_distortion_model(model: str) -> str:
    return " ".join(
        model.strip().lower().replace("_", " ").replace("-", " ").split()
    )


def _brown_coefficients(intrinsics: Intrinsics, *, role: str) -> np.ndarray:
    coefficients = np.asarray(intrinsics.coeffs, dtype=np.float64)
    if coefficients.size == 0:
        return np.zeros(5, dtype=np.float64)
    if coefficients.shape != (5,) or not np.isfinite(coefficients).all():
        raise ValueError(
            f"{role} requires five finite Brown-Conrady coefficients, got "
            f"{intrinsics.coeffs!r}"
        )
    return coefficients


def apply_brown_conrady(
    normalized_xy: np.ndarray,
    coefficients: Sequence[float],
) -> np.ndarray:
    """Apply the five-coefficient OpenCV/RealSense forward Brown model.

    Input coordinates are ideal normalized pinhole coordinates.  The output
    coordinates are distorted normalized image coordinates.
    """

    points = np.asarray(normalized_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"normalized points must have shape (N,2), got {points.shape}")
    coefficients_array = np.asarray(coefficients, dtype=np.float64)
    if coefficients_array.shape != (5,) or not np.isfinite(coefficients_array).all():
        raise ValueError("Brown-Conrady projection requires five finite coefficients")
    k1, k2, p1, p2, k3 = coefficients_array
    x = points[:, 0]
    y = points[:, 1]
    radius_squared = x * x + y * y
    radius_fourth = radius_squared * radius_squared
    radial = (
        1.0
        + k1 * radius_squared
        + k2 * radius_fourth
        + k3 * radius_fourth * radius_squared
    )
    distorted = np.empty_like(points)
    distorted[:, 0] = (
        x * radial
        + 2.0 * p1 * x * y
        + p2 * (radius_squared + 2.0 * x * x)
    )
    distorted[:, 1] = (
        y * radial
        + p1 * (radius_squared + 2.0 * y * y)
        + 2.0 * p2 * x * y
    )
    return distorted


def invert_brown_conrady(
    distorted_xy: np.ndarray,
    coefficients: Sequence[float],
    *,
    iterations: int = 12,
) -> np.ndarray:
    """Invert a Brown map with a vectorised Newton solve.

    RealSense ``Inverse Brown Conrady`` coefficients map distorted pixels to
    ideal rays in closed form.  Projection needs the inverse operation, which
    this function computes without depending on a connected RealSense device.
    """

    targets = np.asarray(distorted_xy, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError(f"normalized points must have shape (N,2), got {targets.shape}")
    coefficients_array = np.asarray(coefficients, dtype=np.float64)
    if coefficients_array.shape != (5,) or not np.isfinite(coefficients_array).all():
        raise ValueError("Brown-Conrady inversion requires five finite coefficients")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    estimate = targets.copy()
    k1, k2, p1, p2, k3 = coefficients_array
    for _ in range(iterations):
        x = estimate[:, 0]
        y = estimate[:, 1]
        radius_squared = x * x + y * y
        radius_fourth = radius_squared * radius_squared
        radial = (
            1.0
            + k1 * radius_squared
            + k2 * radius_fourth
            + k3 * radius_fourth * radius_squared
        )
        derivative_scale = (
            k1 + 2.0 * k2 * radius_squared + 3.0 * k3 * radius_fourth
        )
        radial_x = 2.0 * x * derivative_scale
        radial_y = 2.0 * y * derivative_scale

        residual_x = (
            x * radial
            + 2.0 * p1 * x * y
            + p2 * (radius_squared + 2.0 * x * x)
            - targets[:, 0]
        )
        residual_y = (
            y * radial
            + p1 * (radius_squared + 2.0 * y * y)
            + 2.0 * p2 * x * y
            - targets[:, 1]
        )
        jacobian_00 = radial + x * radial_x + 2.0 * p1 * y + 6.0 * p2 * x
        jacobian_01 = x * radial_y + 2.0 * p1 * x + 2.0 * p2 * y
        jacobian_10 = y * radial_x + 2.0 * p1 * x + 2.0 * p2 * y
        jacobian_11 = radial + y * radial_y + 6.0 * p1 * y + 2.0 * p2 * x
        determinant = jacobian_00 * jacobian_11 - jacobian_01 * jacobian_10
        safe = np.isfinite(determinant) & (np.abs(determinant) > 1.0e-12)
        if not np.all(safe):
            raise ValueError("Brown-Conrady projection did not have a finite inverse")
        delta_x = (
            jacobian_11 * residual_x - jacobian_01 * residual_y
        ) / determinant
        delta_y = (
            -jacobian_10 * residual_x + jacobian_00 * residual_y
        ) / determinant
        estimate[:, 0] -= delta_x
        estimate[:, 1] -= delta_y
        if max(float(np.max(np.abs(delta_x))), float(np.max(np.abs(delta_y)))) < 1.0e-12:
            break
    if not np.isfinite(estimate).all():
        raise ValueError("Brown-Conrady projection produced non-finite coordinates")
    return estimate


def projection_model_for_intrinsics(
    intrinsics: Intrinsics,
    *,
    projection_model_override: str | None = None,
) -> str:
    """Resolve the effective projection convention used for one RGB stream."""

    coefficients = _brown_coefficients(intrinsics, role="target colour intrinsics")
    if np.max(np.abs(coefficients)) <= 1.0e-12:
        return "pinhole_zero_distortion"
    model = _normalise_distortion_model(
        projection_model_override or intrinsics.model
    )
    if model in {
        "brown conrady",
        "modified brown conrady",
        "opencv brown forward",
        "brown conrady forward",
    }:
        return "brown_conrady_forward"
    if model in {"inverse brown conrady", "brown conrady inverse"}:
        return "inverse_brown_conrady"
    raise ValueError(
        f"unsupported target colour distortion model: "
        f"{projection_model_override or intrinsics.model!r}"
    )


def project_normalized_to_pixels(
    normalized_xy: np.ndarray,
    intrinsics: Intrinsics,
    *,
    projection_model_override: str | None = None,
) -> np.ndarray:
    """Project ideal normalized rays into a possibly distorted RGB image."""

    _validate_intrinsic_dimensions(intrinsics, role="target colour intrinsics")
    points = np.asarray(normalized_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"normalized points must have shape (N,2), got {points.shape}")
    coefficients = _brown_coefficients(intrinsics, role="target colour intrinsics")
    model = projection_model_for_intrinsics(
        intrinsics,
        projection_model_override=projection_model_override,
    )
    if model == "pinhole_zero_distortion":
        image_points = points
    elif model == "brown_conrady_forward":
        image_points = apply_brown_conrady(points, coefficients)
    elif model == "inverse_brown_conrady":
        image_points = invert_brown_conrady(points, coefficients)
    else:  # pragma: no cover - guarded by projection_model_for_intrinsics
        raise AssertionError(model)
    pixels = np.empty_like(image_points)
    pixels[:, 0] = image_points[:, 0] * intrinsics.fx + intrinsics.ppx
    pixels[:, 1] = image_points[:, 1] * intrinsics.fy + intrinsics.ppy
    return pixels


def _validate_supported_deprojection_intrinsics(
    intrinsics: Intrinsics,
    *,
    role: str,
) -> None:
    _validate_intrinsic_dimensions(intrinsics, role=role)

    # All case01 depth models and the camera-2 colour model have zero
    # coefficients.  Refuse silently-wrong deprojection if a future manifest
    # needs a distortion-aware depth ray model.
    coefficients = np.asarray(intrinsics.coeffs, dtype=np.float64)
    if coefficients.size and np.max(np.abs(coefficients)) > 1.0e-10:
        raise ValueError(
            f"{role} uses non-zero {intrinsics.model!r} distortion; "
            "source depth deprojection currently requires a zero-coefficient plane"
        )


def normalized_pinhole_rays(intrinsics: Intrinsics) -> np.ndarray:
    """Return one ``[x/z, y/z, 1]`` ray per depth pixel."""

    _validate_supported_deprojection_intrinsics(
        intrinsics, role="source depth intrinsics"
    )
    columns, rows = np.meshgrid(
        np.arange(intrinsics.width, dtype=np.float64),
        np.arange(intrinsics.height, dtype=np.float64),
    )
    rays = np.empty((intrinsics.height, intrinsics.width, 3), dtype=np.float64)
    rays[..., 0] = (columns - intrinsics.ppx) / intrinsics.fx
    rays[..., 1] = (rows - intrinsics.ppy) / intrinsics.fy
    rays[..., 2] = 1.0
    return rays


def register_depth_frame(
    raw_depth: np.ndarray,
    *,
    depth_units_m: float,
    source_intrinsics: Intrinsics,
    target_intrinsics: Intrinsics,
    rotation_target_from_depth: np.ndarray,
    translation_target_from_depth: np.ndarray,
    min_depth_m: float = 0.05,
    max_depth_m: float = 8.0,
    source_rays: np.ndarray | None = None,
    projection_model_override: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Register one raw depth frame into the target colour plane.

    Projection uses nearest-pixel sampling followed by ``minimum.at`` so that
    collisions always keep the geometrically nearest surface.
    """

    raw_depth = np.asarray(raw_depth)
    expected_shape = (source_intrinsics.height, source_intrinsics.width)
    if raw_depth.shape != expected_shape:
        raise ValueError(
            f"raw depth shape {raw_depth.shape} does not match {expected_shape}"
        )
    if not np.issubdtype(raw_depth.dtype, np.integer):
        raise TypeError(f"raw depth must have integer sensor units, got {raw_depth.dtype}")
    if depth_units_m <= 0.0 or not np.isfinite(depth_units_m):
        raise ValueError(f"invalid depth scale: {depth_units_m}")
    if not (0.0 <= min_depth_m < max_depth_m):
        raise ValueError("expected 0 <= min_depth_m < max_depth_m")

    _validate_intrinsic_dimensions(target_intrinsics, role="target colour intrinsics")
    projection_model_for_intrinsics(
        target_intrinsics,
        projection_model_override=projection_model_override,
    )
    rotation = np.asarray(rotation_target_from_depth, dtype=np.float64)
    translation = np.asarray(translation_target_from_depth, dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("rigid transform must have shapes (3,3) and (3,)")

    if source_rays is None:
        source_rays = normalized_pinhole_rays(source_intrinsics)
    else:
        source_rays = np.asarray(source_rays, dtype=np.float64)
        if source_rays.shape != (*expected_shape, 3):
            raise ValueError(
                f"source_rays shape {source_rays.shape} does not match "
                f"{(*expected_shape, 3)}"
            )

    depth_m = raw_depth.astype(np.float64) * depth_units_m
    valid_source = (
        (raw_depth > 0)
        & np.isfinite(depth_m)
        & (depth_m >= min_depth_m)
        & (depth_m <= max_depth_m)
    )
    target_flat = np.full(
        target_intrinsics.height * target_intrinsics.width,
        np.inf,
        dtype=np.float64,
    )
    if np.any(valid_source):
        points_depth = source_rays[valid_source] * depth_m[valid_source, None]
        points_target = points_depth @ rotation.T + translation
        target_z = points_target[:, 2]
        in_front = np.isfinite(points_target).all(axis=1) & (target_z > 0.0)
        points_target = points_target[in_front]
        target_z = target_z[in_front]

        if target_z.size:
            projected_pixels = project_normalized_to_pixels(
                points_target[:, :2] / target_z[:, None],
                target_intrinsics,
                projection_model_override=projection_model_override,
            )
            projected_x = projected_pixels[:, 0]
            projected_y = projected_pixels[:, 1]
            finite_projection = np.isfinite(projected_x) & np.isfinite(projected_y)
            pixel_x = np.rint(projected_x[finite_projection]).astype(np.int64)
            pixel_y = np.rint(projected_y[finite_projection]).astype(np.int64)
            projected_z = target_z[finite_projection]
            inside = (
                (pixel_x >= 0)
                & (pixel_x < target_intrinsics.width)
                & (pixel_y >= 0)
                & (pixel_y < target_intrinsics.height)
            )
            flat_indices = (
                pixel_y[inside] * target_intrinsics.width + pixel_x[inside]
            )
            np.minimum.at(target_flat, flat_indices, projected_z[inside])

    valid_target = np.isfinite(target_flat).reshape(
        target_intrinsics.height, target_intrinsics.width
    )
    metric_depth = target_flat.reshape(
        target_intrinsics.height, target_intrinsics.width
    ).astype(np.float32)
    metric_depth[~valid_target] = np.nan
    return metric_depth, valid_target


def _camera_by_id(manifest: dict[str, Any], camera_id: int) -> dict[str, Any]:
    matches = [
        camera
        for camera in manifest.get("cameras", [])
        if int(camera.get("camera_id", -1)) == camera_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"manifest must contain exactly one camera {camera_id}, found {len(matches)}"
        )
    return matches[0]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_depth_png(path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("OpenCV is required to read the raw depth PNG files") from exc
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"could not decode depth image: {path}")
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError(
            f"expected single-channel uint16 depth PNG, got {depth.shape} {depth.dtype}: {path}"
        )
    return depth


def _matrix_from_config(payload: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    transform = payload[key]
    rotation = np.asarray(transform["rotation"], dtype=np.float64)
    translation = np.asarray(transform["translation_m"], dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(f"invalid transform {key!r} in calibration config")
    return rotation, translation


def _require_allclose(
    actual: np.ndarray | Sequence[float],
    expected: np.ndarray | Sequence[float],
    *,
    label: str,
    atol: float,
) -> None:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual_array.shape != expected_array.shape or not np.allclose(
        actual_array, expected_array, atol=atol, rtol=0.0
    ):
        maximum_error = (
            float(np.max(np.abs(actual_array - expected_array)))
            if actual_array.shape == expected_array.shape
            else None
        )
        raise ValueError(
            f"{label} does not match the depth calibration binding "
            f"(max_abs_error={maximum_error})"
        )


def _check_manifest_binding(
    manifest: dict[str, Any], calibration: dict[str, Any]
) -> None:
    if manifest.get("image_transform") != "none":
        raise ValueError(
            "depth calibration is defined in unrotated sensor coordinates; "
            f"manifest image_transform is {manifest.get('image_transform')!r}"
        )
    if manifest.get("depth_registered_to_rgb") is not False:
        raise ValueError("expected unregistered raw depth in the conversion manifest")

    bindings = calibration.get("device_bindings", {})
    for camera_id in (1, 2):
        camera = _camera_by_id(manifest, camera_id)
        expected_serial = str(bindings[f"camera_{camera_id}"]["serial_number"])
        actual_serial = str(camera.get("device", {}).get("Serial Number", ""))
        if actual_serial != expected_serial:
            raise ValueError(
                f"camera {camera_id} serial mismatch: {actual_serial!r} != "
                f"{expected_serial!r}"
            )

        parsed_rotation, parsed_translation = parse_rs2_depth_to_color_transform(
            camera["color_tf_ref_raw"]
        )
        expected_factory = bindings[f"camera_{camera_id}"][
            "factory_depth_to_color"
        ]
        _require_allclose(
            parsed_rotation,
            np.asarray(expected_factory["rotation"], dtype=np.float64),
            label=f"camera {camera_id} factory rotation",
            atol=1.0e-9,
        )
        _require_allclose(
            parsed_translation,
            np.asarray(expected_factory["translation_m"], dtype=np.float64),
            label=f"camera {camera_id} factory translation",
            atol=1.0e-9,
        )
        for stream in ("depth", "color"):
            actual_intrinsics = camera[f"{stream}_info"]
            expected_intrinsics = bindings[f"camera_{camera_id}"][
                f"{stream}_intrinsics"
            ]
            for key in ("width", "height", "fx", "fy", "ppx", "ppy"):
                _require_allclose(
                    [actual_intrinsics[key]],
                    [expected_intrinsics[key]],
                    label=f"camera {camera_id} {stream} {key}",
                    atol=1.0e-9,
                )
            _require_allclose(
                actual_intrinsics.get("coeffs", ()),
                expected_intrinsics.get("coeffs", ()),
                label=f"camera {camera_id} {stream} distortion coefficients",
                atol=1.0e-9,
            )
        _require_allclose(
            [camera["depth_units_m"]],
            [bindings[f"camera_{camera_id}"]["depth_units_m"]],
            label=f"camera {camera_id} depth units",
            atol=1.0e-12,
        )


def source_to_camera2_color_transform(
    source_camera: int,
    manifest: dict[str, Any],
    calibration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return the depth-source -> camera2-colour transform."""

    camera = _camera_by_id(manifest, source_camera)
    rotation_color_from_depth, translation_color_from_depth = (
        parse_rs2_depth_to_color_transform(camera["color_tf_ref_raw"])
    )
    if source_camera == 2:
        return (
            rotation_color_from_depth,
            translation_color_from_depth,
            "camera2 factory rs2 depth_to_color",
        )
    if source_camera != 1:
        raise ValueError(f"unsupported source camera: {source_camera}")

    rotation_c2_from_c1, translation_c2_from_c1 = _matrix_from_config(
        calibration, "color2_from_color1"
    )
    rotation, translation = compose_rigid_transforms(
        rotation_c2_from_c1,
        translation_c2_from_c1,
        rotation_color_from_depth,
        translation_color_from_depth,
    )

    # The composition is the source of truth; this rounded direct transform is
    # persisted as an independent guard against direction/order mistakes.
    expected_rotation, expected_translation = _matrix_from_config(
        calibration, "depth1_to_color2_composed_reference"
    )
    _require_allclose(
        rotation,
        expected_rotation,
        label="composed depth1-to-color2 rotation",
        atol=1.0e-7,
    )
    _require_allclose(
        translation,
        expected_translation,
        label="composed depth1-to-color2 translation",
        atol=1.0e-7,
    )
    return (
        rotation,
        translation,
        "camera1 factory rs2 depth_to_color composed with refined color1_to_color2",
    )


def source_to_native_color_transform(
    source_camera: int,
    manifest: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return one device's factory depth -> native-colour transform."""

    camera = _camera_by_id(manifest, source_camera)
    rotation, translation = parse_rs2_depth_to_color_transform(
        camera["color_tf_ref_raw"]
    )
    return (
        rotation,
        translation,
        f"camera{source_camera} factory rs2 depth_to_color",
    )


def _native_color_projection_override(
    calibration: dict[str, Any],
    camera_id: int,
) -> str | None:
    binding = calibration.get("device_bindings", {}).get(f"camera_{camera_id}", {})
    projection = binding.get("native_color_projection", {})
    override = projection.get("model_override")
    if override is None:
        return None
    if not isinstance(override, str) or not override.strip():
        raise ValueError(
            f"camera {camera_id} native colour projection override must be a string"
        )
    return override


def _atomic_output_directory(output_dir: Path) -> tuple[Path, Path]:
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new path: {output_dir}"
        )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    return output_dir, staging


def _write_registered_sequence(
    *,
    episode_dir: Path,
    staging_dir: Path,
    source_camera: int,
    target_camera: int = 2,
    manifest: dict[str, Any],
    calibration: dict[str, Any],
    min_depth_m: float,
    max_depth_m: float,
    projection_model_override: str | None = None,
) -> dict[str, Any]:
    source_payload = _camera_by_id(manifest, source_camera)
    target_payload = _camera_by_id(manifest, target_camera)
    source_intrinsics = Intrinsics.from_manifest(source_payload["depth_info"])
    target_intrinsics = Intrinsics.from_manifest(target_payload["color_info"])
    _validate_supported_deprojection_intrinsics(
        source_intrinsics, role="source depth intrinsics"
    )
    _validate_intrinsic_dimensions(
        target_intrinsics, role=f"camera{target_camera} colour intrinsics"
    )
    effective_projection_model = projection_model_for_intrinsics(
        target_intrinsics,
        projection_model_override=projection_model_override,
    )

    if target_camera == 2:
        rotation, translation, transform_source = source_to_camera2_color_transform(
            source_camera, manifest, calibration
        )
    elif target_camera == source_camera:
        rotation, translation, transform_source = source_to_native_color_transform(
            source_camera, manifest
        )
    else:
        raise ValueError(
            f"unsupported depth registration camera_{source_camera} -> "
            f"camera_{target_camera}/color"
        )
    depth_paths = sorted(
        (episode_dir / f"camera_{source_camera}" / "depth_raw").glob("*.png")
    )
    frame_count = int(manifest["frame_count"])
    if len(depth_paths) != frame_count:
        raise ValueError(
            f"camera {source_camera} depth frame count {len(depth_paths)} != "
            f"manifest frame_count {frame_count}"
        )

    destination = (
        staging_dir
        / f"camera_{source_camera}_to_camera_{target_camera}_color"
    )
    destination.mkdir(parents=True)
    depth_output = destination / "depth_metric.npy"
    valid_output = destination / "valid_mask.npy"
    shape = (frame_count, target_intrinsics.height, target_intrinsics.width)
    depth_stack = np.lib.format.open_memmap(
        depth_output, mode="w+", dtype=np.float32, shape=shape
    )
    valid_stack = np.lib.format.open_memmap(
        valid_output, mode="w+", dtype=np.bool_, shape=shape
    )
    source_rays = normalized_pinhole_rays(source_intrinsics)

    input_valid_total = 0
    output_valid_total = 0
    frame_coverages: list[float] = []
    output_min_depth = np.inf
    output_max_depth = -np.inf
    for frame_index, depth_path in enumerate(depth_paths):
        raw_depth = _load_depth_png(depth_path)
        input_valid_total += int(np.count_nonzero(raw_depth))
        registered, valid = register_depth_frame(
            raw_depth,
            depth_units_m=float(source_payload["depth_units_m"]),
            source_intrinsics=source_intrinsics,
            target_intrinsics=target_intrinsics,
            rotation_target_from_depth=rotation,
            translation_target_from_depth=translation,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            source_rays=source_rays,
            projection_model_override=projection_model_override,
        )
        depth_stack[frame_index] = registered
        valid_stack[frame_index] = valid
        valid_count = int(np.count_nonzero(valid))
        output_valid_total += valid_count
        frame_coverages.append(valid_count / float(valid.size))
        if valid_count:
            output_min_depth = min(
                output_min_depth, float(np.nanmin(registered))
            )
            output_max_depth = max(
                output_max_depth, float(np.nanmax(registered))
            )

    depth_stack.flush()
    valid_stack.flush()
    del depth_stack, valid_stack
    report = {
        "source_camera": source_camera,
        "target": f"camera_{target_camera}/color",
        "frame_count": frame_count,
        "array_shape": list(shape),
        "depth_units": "meter",
        "invalid_depth_value": "NaN",
        "input": {
            "directory": str(
                (episode_dir / f"camera_{source_camera}" / "depth_raw").resolve()
            ),
            "dtype": "uint16",
            "depth_units_m": float(source_payload["depth_units_m"]),
            "nonzero_samples": input_valid_total,
        },
        "outputs": {
            "depth_metric_npy": str(depth_output.relative_to(staging_dir)),
            "valid_mask_npy": str(valid_output.relative_to(staging_dir)),
        },
        "source_depth_intrinsics": asdict(source_intrinsics),
        "target_color_intrinsics": asdict(target_intrinsics),
        "target_projection": {
            "manifest_model": target_intrinsics.model,
            "model_override": projection_model_override,
            "effective_model": effective_projection_model,
        },
        "transform": {
            "equation": (
                f"X_camera{target_camera}_color = R @ X_source_depth + t"
            ),
            "rotation": rotation.tolist(),
            "translation_m": translation.tolist(),
            "source": transform_source,
        },
        "filter": {"min_depth_m": min_depth_m, "max_depth_m": max_depth_m},
        "statistics": {
            "output_valid_samples": output_valid_total,
            "mean_rgb_coverage": float(np.mean(frame_coverages)),
            "min_frame_rgb_coverage": float(np.min(frame_coverages)),
            "max_frame_rgb_coverage": float(np.max(frame_coverages)),
            "output_min_depth_m": (
                None if not np.isfinite(output_min_depth) else output_min_depth
            ),
            "output_max_depth_m": (
                None if not np.isfinite(output_max_depth) else output_max_depth
            ),
        },
    }
    with (destination / "report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")
    return report


def register_episode(
    *,
    episode_dir: Path,
    output_dir: Path,
    calibration_path: Path,
    include_camera_1: bool = False,
    include_camera_1_native: bool = False,
    only_camera_1_native: bool = False,
    min_depth_m: float = 0.05,
    max_depth_m: float = 8.0,
) -> Path:
    episode_dir = episode_dir.resolve()
    manifest_path = episode_dir / "conversion_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    calibration_path = calibration_path.resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    manifest = _load_json(manifest_path)
    calibration = _load_json(calibration_path)
    _check_manifest_binding(manifest, calibration)

    final_dir, staging_dir = _atomic_output_directory(output_dir)
    try:
        source_reports = {}
        registrations: list[tuple[str, int, int, str | None]] = []
        if not only_camera_1_native:
            registrations.append(("camera_2", 2, 2, None))
            if include_camera_1:
                registrations.append(("camera_1", 1, 2, None))
        if include_camera_1_native or only_camera_1_native:
            registrations.append(
                (
                    "camera_1_native",
                    1,
                    1,
                    _native_color_projection_override(calibration, 1),
                )
            )
        for report_key, source_camera, target_camera, projection_override in registrations:
            source_reports[report_key] = _write_registered_sequence(
                episode_dir=episode_dir,
                staging_dir=staging_dir,
                source_camera=source_camera,
                target_camera=target_camera,
                manifest=manifest,
                calibration=calibration,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
                projection_model_override=projection_override,
            )

        report = {
            "schema_version": 1,
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "episode": str(episode_dir),
            "manifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
            },
            "calibration": {
                "path": str(calibration_path),
                "sha256": _sha256(calibration_path),
                "calibration_id": calibration.get("calibration_id"),
            },
            "provenance": {
                "script": str(Path(__file__).resolve()),
                "script_version": SCRIPT_VERSION,
                "python": sys.version,
                "numpy": np.__version__,
                "algorithm": (
                    "metric pinhole deprojection, rigid transform, nearest-pixel "
                    "projection, nearest-surface z-buffer"
                ),
                "rs2_rotation_storage": "column-major",
                "coordinate_convention": "x right, y down, z forward",
                "image_transform": "none",
            },
            "sources": source_reports,
        }
        with (staging_dir / "registration_report.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(report, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(staging_dir, final_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return final_dir


def _default_calibration_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "calibration"
        / "depth_registration_20260803.json"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode",
        required=True,
        type=Path,
        help="Converted episode containing conversion_manifest.json and camera_*/depth_raw",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New output directory (must not already exist)",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=_default_calibration_path(),
        help="Depth-consistent stereo calibration JSON",
    )
    parser.add_argument(
        "--include-camera-1",
        action="store_true",
        help="Also project camera-1 raw depth into camera-2 RGB",
    )
    parser.add_argument(
        "--include-camera-1-native",
        action="store_true",
        help="Also register camera-1 raw depth into its native camera-1 RGB",
    )
    parser.add_argument(
        "--only-camera-1-native",
        action="store_true",
        help=(
            "Write only camera-1 raw depth registered into native camera-1 RGB; "
            "the default camera-2 output is unchanged when this flag is absent"
        ),
    )
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=8.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = register_episode(
        episode_dir=args.episode,
        output_dir=args.output_dir,
        calibration_path=args.calibration,
        include_camera_1=args.include_camera_1,
        include_camera_1_native=args.include_camera_1_native,
        only_camera_1_native=args.only_camera_1_native,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
