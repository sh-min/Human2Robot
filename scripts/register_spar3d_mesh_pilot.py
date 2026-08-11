#!/usr/bin/env python3
"""Approximately register an SPAR3D object mesh to its selected MH frame.

SPAR3D produces a learned, canonical, non-metric single-image reconstruction.
This script estimates an *approximate* Sim(3) that places that reconstruction
in the MH overlay camera.  It uses the verified MH modal silhouette, calibrated
MH intrinsics/distortion, and the existing Depth Anything V2 Indoor Metric
depth map after the pipeline's HaWoR camera-Z scale anchoring.

The result is deliberately not called metric ground truth.  Its scale and
translation inherit monocular reconstruction, HaWoR, and depth-model error.
The SH checkerboard translation is in checker-square units and is never used as
a metric constraint here.

Run ``--preflight`` before the SPAR3D GLB exists.  Normal execution atomically
publishes a registered GLB, transforms, MH silhouette/depth diagnostics, a
canonical turntable contact sheet, and static/video before-after comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = (
    REPO_ROOT / "8-5" / "mesh_sota_pilot" / "episode_1" / "choco"
)
DEFAULT_INPUT_ROOT = PILOT_ROOT / "inputs"
DEFAULT_MANIFEST = DEFAULT_INPUT_ROOT / "manifest.json"
DEFAULT_IMAGE = DEFAULT_INPUT_ROOT / "mh_frame000187.jpg"
DEFAULT_MASK = DEFAULT_INPUT_ROOT / "mh_mask_modal_frame000187.png"
DEFAULT_MESH = PILOT_ROOT / "spar3d" / "mesh.glb"
DEFAULT_SPAR_REPORT = PILOT_ROOT / "spar3d" / "report.json"
DEFAULT_OUTPUT = PILOT_ROOT / "spar3d_registered_mh"
DEFAULT_DEPTH = (
    REPO_ROOT
    / "data"
    / "kitchen_dataset"
    / "26.08.05_stereo_calibrated"
    / "1"
    / "camera_2"
    / "inpainting"
    / "processed"
    / "view"
    / "0"
    / "depth_processor"
    / "depth_aligned_metric.npy"
)
DEFAULT_DEPTH_PARAMS = DEFAULT_DEPTH.parent / "depth_metric_params.npz"
DEFAULT_DEPTH_CHECKPOINT = (
    REPO_ROOT
    / "weights"
    / "depth_anything"
    / "depth_anything_v2_metric_hypersim_vits.pth"
)
DEFAULT_DEPTH_CHECKPOINT_SHA256 = (
    "b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545"
)

METHOD = "spar3d_canonical_to_mh_approximate_sim3_silhouette_depth_fit"
REPRESENTATION = "learned_single_view_mesh_approximately_registered_to_mh_camera"
T_CV_TO_GL = np.diag((1.0, -1.0, -1.0, 1.0))

OBJECTIVE_WEIGHTS = {
    "silhouette_iou": 0.48,
    "silhouette_centroid": 0.10,
    "silhouette_extent": 0.12,
    "silhouette_area": 0.05,
    "depth_median": 0.12,
    "depth_overlap_residual": 0.10,
    "canonical_view_prior": 0.03,
}

PROVENANCE_WARNINGS = (
    "The SPAR3D mesh and its hidden/backside surface are learned estimates, not observed geometry.",
    "The fitted Sim(3) is approximate and is not a metric calibration result.",
    "Depth Anything V2 is scaled by HaWoR camera-Z anchors, so depth is an overlay-coordinate proxy rather than independent sensor ground truth.",
    "The SH calibration translation is expressed in checker-square units and is not used as metric evidence.",
    "A single MH silhouette cannot identify all rotations, concavities, thicknesses, or absolute scale.",
    "The registered mesh is diagnostic and must not be treated as a validated physical collision boundary.",
)


class RegistrationInputError(ValueError):
    """Raised when the focused pilot contract is violated."""


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    camera_matrix: np.ndarray
    distortion: np.ndarray

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise RegistrationInputError("camera dimensions must be positive")
        if self.camera_matrix.shape != (3, 3):
            raise RegistrationInputError("MH camera matrix must have shape (3,3)")
        if self.distortion.shape != (5,):
            raise RegistrationInputError("MH distortion must have five coefficients")
        if not np.isfinite(self.camera_matrix).all() or not np.isfinite(
            self.distortion
        ).all():
            raise RegistrationInputError("MH calibration contains non-finite values")
        if self.camera_matrix[0, 0] <= 0 or self.camera_matrix[1, 1] <= 0:
            raise RegistrationInputError("MH focal lengths must be positive")


@dataclass(frozen=True)
class ObservationSummary:
    centroid_xy: tuple[float, float]
    bbox_xyxy_exclusive: tuple[int, int, int, int]
    foreground_pixels: int
    median_depth_proxy_m: float
    depth_mad_proxy_m: float
    projected_extent_proxy_m: tuple[float, float]


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _resolved_file(path: str | Path, description: str) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"missing {description}: {result}")
    return result


def _load_json(path: str | Path, description: str) -> tuple[Path, dict[str, Any]]:
    result = _resolved_file(path, description)
    try:
        payload = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistrationInputError(f"invalid {description} {result}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RegistrationInputError(f"{description} root must be an object")
    return result, payload


def load_pilot_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path, payload = _load_json(path, "mesh-pilot input manifest")
    if payload.get("schema_version") != 1 or payload.get("kind") != "mesh_sota_pilot_input_bundle":
        raise RegistrationInputError(
            "expected schema_version=1 mesh_sota_pilot_input_bundle manifest"
        )
    try:
        selection = payload["selection"]
        calibration = payload["calibration"]
        mh = calibration["intrinsics_by_view"]["MH"]
    except (KeyError, TypeError) as exc:
        raise RegistrationInputError(
            "input manifest lacks selection or calibration.intrinsics_by_view.MH"
        ) from exc
    object_label = selection.get("object_label")
    if not isinstance(object_label, str) or not object_label.strip():
        raise RegistrationInputError("manifest requires a non-empty object label")
    if str(selection.get("mh_role", "")).split("/")[0] != "primary":
        raise RegistrationInputError("manifest must declare MH as the primary view")
    checkerboard = calibration.get("checkerboard", {})
    if checkerboard.get("metric_scale_verified") is not False:
        raise RegistrationInputError(
            "expected checkerboard metric_scale_verified=false for this dataset"
        )
    if mh.get("calibration_camera") != "camera_1":
        raise RegistrationInputError("unexpected MH-to-calibration camera mapping")
    return manifest_path, payload


def validate_selected_input_records(
    manifest: dict[str, Any], *, image_path: Path, mask_path: Path
) -> None:
    """Bind the CLI paths to the small, hash-recorded pilot artifacts."""

    try:
        records = {
            "MH image": (manifest["outputs"]["mh_image"], image_path),
            "MH modal mask": (manifest["outputs"]["modal_mask"], mask_path),
        }
    except (KeyError, TypeError) as exc:
        raise RegistrationInputError(
            "pilot manifest lacks outputs.mh_image or outputs.modal_mask"
        ) from exc
    for label, (record, actual_path) in records.items():
        declared_path = Path(record["path"]).expanduser().resolve()
        if declared_path != actual_path:
            raise RegistrationInputError(
                f"{label} path differs from the manifest: {actual_path} != {declared_path}"
            )
        if int(record.get("bytes", -1)) != actual_path.stat().st_size:
            raise RegistrationInputError(f"{label} byte count differs from the manifest")
        if str(record.get("sha256")) != sha256_file(actual_path):
            raise RegistrationInputError(f"{label} SHA-256 differs from the manifest")


def parse_mh_calibration(
    manifest: dict[str, Any], image_shape: tuple[int, int]
) -> CameraCalibration:
    height, width = image_shape
    mh = manifest["calibration"]["intrinsics_by_view"]["MH"]
    camera = CameraCalibration(
        width=width,
        height=height,
        camera_matrix=np.asarray(mh["camera_matrix"], dtype=np.float64),
        distortion=np.asarray(
            mh["distortion_k1_k2_p1_p2_k3"], dtype=np.float64
        ),
    )
    camera.validate()
    return camera


def load_image_and_mask(
    image_path: str | Path, mask_path: str | Path
) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    image_file = _resolved_file(image_path, "selected MH image")
    mask_file = _resolved_file(mask_path, "selected MH modal mask")
    image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
    mask_u8 = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    if image is None or mask_u8 is None:
        raise RegistrationInputError("OpenCV could not decode the MH image or mask")
    if image.shape[:2] != mask_u8.shape:
        raise RegistrationInputError(
            f"MH image/mask shape mismatch: {image.shape[:2]} != {mask_u8.shape}"
        )
    mask = mask_u8 > 0
    if not mask.any() or mask.all():
        raise RegistrationInputError("MH modal mask must be nonempty and not full-frame")
    if not set(np.unique(mask_u8)).issubset({0, 255}):
        raise RegistrationInputError("MH modal mask must be binary 0/255")
    return image_file, mask_file, image, mask


def load_depth_frame(
    path: str | Path, *, frame_index: int, expected_shape: tuple[int, int]
) -> tuple[Path, np.ndarray, tuple[int, ...], str]:
    depth_path = _resolved_file(path, "HaWoR-anchored metric depth array")
    depth_array = np.load(depth_path, mmap_mode="r", allow_pickle=False)
    if depth_array.ndim == 2:
        frame = np.asarray(depth_array, dtype=np.float32)
    elif depth_array.ndim == 3:
        if not 0 <= frame_index < len(depth_array):
            raise RegistrationInputError(
                f"MH frame {frame_index} outside depth array length {len(depth_array)}"
            )
        frame = np.asarray(depth_array[frame_index], dtype=np.float32)
    else:
        raise RegistrationInputError(
            f"depth array must have shape (H,W) or (T,H,W), got {depth_array.shape}"
        )
    if frame.shape != expected_shape:
        raise RegistrationInputError(
            f"depth/image shape mismatch: {frame.shape} != {expected_shape}"
        )
    return depth_path, frame, tuple(depth_array.shape), str(depth_array.dtype)


def load_depth_anchor_record(
    path: str | Path, *, frame_index: int
) -> tuple[Path, dict[str, Any]]:
    params_path = _resolved_file(path, "metric-depth HaWoR anchor parameters")
    with np.load(params_path, allow_pickle=False) as payload:
        required = {"raw_scale", "scale", "valid_frames", "encoder", "checkpoint"}
        if not required.issubset(payload.files):
            raise RegistrationInputError(
                f"depth parameter archive lacks {sorted(required - set(payload.files))}"
            )
        scales = np.asarray(payload["scale"], dtype=np.float64)
        raw = np.asarray(payload["raw_scale"], dtype=np.float64)
        valid = np.asarray(payload["valid_frames"], dtype=np.uint8)
        if not 0 <= frame_index < len(scales):
            raise RegistrationInputError("selected MH frame is outside depth parameters")
        scale = float(scales[frame_index])
        if not math.isfinite(scale) or scale <= 0:
            raise RegistrationInputError("selected depth anchor scale is invalid")
        raw_scale = float(raw[frame_index])
        record = {
            "frame_index": int(frame_index),
            "temporal_scale": scale,
            "raw_frame_scale": raw_scale if math.isfinite(raw_scale) else None,
            "raw_frame_anchor_valid": bool(valid[frame_index]),
            "encoder": str(np.asarray(payload["encoder"]).item()),
            "checkpoint": str(np.asarray(payload["checkpoint"]).item()),
            "method": "Depth Anything V2 Indoor Metric HyperSim scaled by HaWoR camera-Z median ratios",
            "independent_metric_ground_truth": False,
        }
    return params_path, record


def validate_depth_source_provenance(
    *,
    depth_path: Path,
    params_path: Path,
    anchor: dict[str, Any],
) -> dict[str, Any]:
    """Record depth evidence and strictly bind the project's default source."""

    result = {
        "depth_array": {
            "path": str(depth_path),
            "bytes": depth_path.stat().st_size,
            "sha256": sha256_file(depth_path),
        },
        "anchor_params": {
            "path": str(params_path),
            "bytes": params_path.stat().st_size,
            "sha256": sha256_file(params_path),
        },
        "project_default_source_verified": False,
    }
    if depth_path == DEFAULT_DEPTH.resolve() or params_path == DEFAULT_DEPTH_PARAMS.resolve():
        if depth_path != DEFAULT_DEPTH.resolve() or params_path != DEFAULT_DEPTH_PARAMS.resolve():
            raise RegistrationInputError(
                "default depth array and default anchor parameters must be used together"
            )
        if anchor.get("encoder") != "vits":
            raise RegistrationInputError("default depth provenance must use encoder='vits'")
        checkpoint = Path(str(anchor.get("checkpoint", ""))).expanduser().resolve()
        if checkpoint != DEFAULT_DEPTH_CHECKPOINT.resolve() or not checkpoint.is_file():
            raise RegistrationInputError(
                "default depth anchor references an unexpected checkpoint"
            )
        checkpoint_sha = sha256_file(checkpoint)
        if checkpoint_sha != DEFAULT_DEPTH_CHECKPOINT_SHA256:
            raise RegistrationInputError("default depth checkpoint SHA-256 mismatch")
        result["depth_checkpoint"] = {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_sha,
        }
        result["project_default_source_verified"] = True
    return result


def weighted_remap_depth(
    depth: np.ndarray, map_x: np.ndarray, map_y: np.ndarray
) -> np.ndarray:
    """Remap camera-Z without allowing invalid values to bleed into the image."""

    values = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(values) & (values > 0)
    weighted = cv2.remap(
        np.where(valid, values, 0.0),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    weights = cv2.remap(
        valid.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    output = np.zeros_like(weighted, dtype=np.float32)
    np.divide(weighted, weights, out=output, where=weights > 0.5)
    return output


def undistort_observation(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    calibration: CameraCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Undistort all evidence to a pinhole image with the same MH K matrix."""

    height, width = image_bgr.shape[:2]
    new_k = calibration.camera_matrix.copy()
    map_x, map_y = cv2.initUndistortRectifyMap(
        calibration.camera_matrix,
        calibration.distortion,
        np.eye(3, dtype=np.float64),
        new_k,
        (width, height),
        cv2.CV_32FC1,
    )
    image_u = cv2.remap(
        image_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    mask_u = cv2.remap(
        mask.astype(np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    depth_u = weighted_remap_depth(depth, map_x, map_y)
    if not mask_u.any():
        raise RegistrationInputError("MH object mask vanished after undistortion")
    return image_u, mask_u, depth_u, new_k


def robust_object_surface(
    scene_depth: np.ndarray, object_mask: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reuse the project's robust modal-object surface filter."""

    source_root = REPO_ROOT / "src" / "inpainting"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from build_object_surface_model import (  # noqa: PLC0415
        SurfaceModelConfig,
        build_surface_frame,
    )

    config = SurfaceModelConfig(erode_px=3, minimum_samples=30, point_count=2048)
    surface, stats = build_surface_frame(
        scene_depth,
        object_mask,
        output_shape=object_mask.shape,
        config=config,
    )
    if not bool(stats["valid"]):
        raise RegistrationInputError(
            "selected MH mask has insufficient robust Depth Anything support"
        )
    return surface, {"config": asdict(config), "frame": stats}


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    y, x = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(x):
        raise RegistrationInputError("cannot compute a bounding box for an empty mask")
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1


def summarize_observation(
    mask: np.ndarray, surface_depth: np.ndarray, camera_matrix: np.ndarray
) -> ObservationSummary:
    foreground = np.asarray(mask, dtype=bool)
    values = np.asarray(surface_depth, dtype=np.float32)[foreground]
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 30:
        raise RegistrationInputError("too few valid object-depth samples")
    y, x = np.nonzero(foreground)
    bbox = mask_bbox(foreground)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    width_px = bbox[2] - bbox[0]
    height_px = bbox[3] - bbox[1]
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    return ObservationSummary(
        centroid_xy=(float(x.mean()), float(y.mean())),
        bbox_xyxy_exclusive=bbox,
        foreground_pixels=int(foreground.sum()),
        median_depth_proxy_m=median,
        depth_mad_proxy_m=mad,
        projected_extent_proxy_m=(width_px * median / fx, height_px * median / fy),
    )


def make_sim3(
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    canonical_center: np.ndarray | None = None,
) -> np.ndarray:
    """Return a matrix mapping original canonical vertices to MH camera XYZ."""

    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    center = (
        np.zeros(3, dtype=np.float64)
        if canonical_center is None
        else np.asarray(canonical_center, dtype=np.float64)
    )
    if rotation.shape != (3, 3) or translation.shape != (3,) or center.shape != (3,):
        raise ValueError("rotation, translation, and center have invalid shapes")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Sim(3) scale must be finite and positive")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("Sim(3) contains non-finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1.0e-6
    ):
        raise ValueError("rotation must be a proper orthonormal matrix")
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = scale * rotation
    output[:3, 3] = translation - scale * rotation @ center
    return output


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or matrix.shape != (4, 4):
        raise ValueError("points must be (N,3) and transform must be (4,4)")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def project_camera_points(points: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("camera points must have shape (N,3)")
    if np.any(points[:, 2] <= 0):
        raise ValueError("all camera points must have positive Z")
    normalized = points[:, :2] / points[:, 2:3]
    output = np.empty((len(points), 2), dtype=np.float64)
    output[:, 0] = normalized[:, 0] * camera_matrix[0, 0] + camera_matrix[0, 2]
    output[:, 1] = normalized[:, 1] * camera_matrix[1, 1] + camera_matrix[1, 2]
    return output


def proper_axis_rotations() -> list[np.ndarray]:
    """Return the 24 proper signed-axis rotations, identity first."""

    rotations: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3, dtype=np.float64)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            candidate = base @ np.diag(signs)
            if np.isclose(np.linalg.det(candidate), 1.0):
                rotations.append(candidate)
    rotations.sort(key=lambda item: float(np.linalg.norm(item - np.eye(3))))
    return rotations


def _initial_scale_translation(
    centered_vertices: np.ndarray,
    rotation: np.ndarray,
    observation: ObservationSummary,
    camera_matrix: np.ndarray,
) -> tuple[float, np.ndarray]:
    rotated = np.asarray(centered_vertices) @ np.asarray(rotation).T
    low, high = np.quantile(rotated, (0.005, 0.995), axis=0)
    extents = np.maximum(high - low, 1.0e-6)
    target_x, target_y = observation.projected_extent_proxy_m
    ratios = np.asarray((target_x / extents[0], target_y / extents[1]))
    scale = float(np.exp(np.median(np.log(np.maximum(ratios, 1.0e-8)))))
    # ``translation`` is the transformed canonical *centre*, whereas the
    # monocular evidence measures the visible/front surface.  Place the robust
    # 0.5-percent canonical front at the observed camera-Z instead of putting
    # the mesh centre there (which would bias half the object through the
    # observed surface before optimisation even starts).
    z = observation.median_depth_proxy_m - scale * float(low[2])
    u, v = observation.centroid_xy
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    translation = np.asarray(((u - cx) * z / fx, (v - cy) * z / fy, z))
    return scale, translation


def resize_depth_valid(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    values = np.asarray(depth, dtype=np.float32)
    if values.shape == (height, width):
        return values.copy()
    valid = np.isfinite(values) & (values > 0)
    weighted = cv2.resize(
        np.where(valid, values, 0.0), (width, height), interpolation=cv2.INTER_AREA
    )
    weights = cv2.resize(
        valid.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA
    )
    output = np.zeros((height, width), dtype=np.float32)
    np.divide(weighted, weights, out=output, where=weights > 0.5)
    return output


def scaled_camera_matrix(
    camera_matrix: np.ndarray,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> np.ndarray:
    source_height, source_width = source_shape
    target_height, target_width = target_shape
    sx, sy = target_width / source_width, target_height / source_height
    output = np.asarray(camera_matrix, dtype=np.float64).copy()
    output[0, :] *= sx
    output[1, :] *= sy
    output[2, :] = (0.0, 0.0, 1.0)
    return output


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    y, x = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(x):
        return None
    return float(x.mean()), float(y.mean())


def alignment_metrics(
    observed_mask: np.ndarray,
    observed_depth: np.ndarray,
    rendered_mask: np.ndarray,
    rendered_depth: np.ndarray,
) -> dict[str, float | int | None]:
    """Compute renderer-independent silhouette and robust camera-Z diagnostics."""

    observed = np.asarray(observed_mask, dtype=bool)
    rendered = np.asarray(rendered_mask, dtype=bool)
    if observed.shape != rendered.shape:
        raise ValueError("observed and rendered masks must have the same shape")
    intersection = observed & rendered
    union = observed | rendered
    obs_count = int(observed.sum())
    render_count = int(rendered.sum())
    intersection_count = int(intersection.sum())
    union_count = int(union.sum())
    iou = intersection_count / max(union_count, 1)
    dice = 2.0 * intersection_count / max(obs_count + render_count, 1)

    obs_center = _centroid(observed)
    render_center = _centroid(rendered)
    diagonal = math.hypot(*observed.shape)
    centroid_error = (
        math.dist(obs_center, render_center) / max(diagonal, 1.0)
        if obs_center is not None and render_center is not None
        else 1.0
    )
    if render_count:
        obs_box = mask_bbox(observed)
        render_box = mask_bbox(rendered)
        obs_extent = np.asarray((obs_box[2] - obs_box[0], obs_box[3] - obs_box[1]))
        render_extent = np.asarray(
            (render_box[2] - render_box[0], render_box[3] - render_box[1])
        )
        extent_log_error = float(
            np.mean(np.abs(np.log(np.maximum(render_extent, 1) / obs_extent)))
        )
        area_log_error = float(abs(math.log(max(render_count, 1) / obs_count)))
    else:
        extent_log_error = 5.0
        area_log_error = 5.0

    observed_z = np.asarray(observed_depth, dtype=np.float32)
    rendered_z = np.asarray(rendered_depth, dtype=np.float32)
    obs_valid = observed & np.isfinite(observed_z) & (observed_z > 0)
    render_valid = rendered & np.isfinite(rendered_z) & (rendered_z > 0)
    paired = obs_valid & render_valid
    observed_median = float(np.median(observed_z[obs_valid])) if obs_valid.any() else None
    rendered_median = (
        float(np.median(rendered_z[render_valid])) if render_valid.any() else None
    )
    median_error = (
        abs(rendered_median - observed_median)
        if observed_median is not None and rendered_median is not None
        else None
    )
    overlap_residual = (
        float(np.median(np.abs(rendered_z[paired] - observed_z[paired])))
        if paired.any()
        else None
    )
    return {
        "observed_pixels": obs_count,
        "rendered_pixels": render_count,
        "intersection_pixels": intersection_count,
        "iou": float(iou),
        "dice": float(dice),
        "centroid_error_fraction_of_image_diagonal": float(centroid_error),
        "extent_log_error": extent_log_error,
        "area_log_error": area_log_error,
        "observed_median_depth_proxy_m": observed_median,
        "rendered_median_camera_z_proxy_m": rendered_median,
        "median_depth_error_proxy_m": median_error,
        "overlap_depth_residual_median_proxy_m": overlap_residual,
        "paired_depth_pixels": int(paired.sum()),
    }


def registration_loss(
    metrics: dict[str, float | int | None],
    *,
    depth_reference_m: float,
    canonical_view_angle_rad: float,
) -> float:
    depth_norm = max(0.035, 0.08 * depth_reference_m)
    median_depth = metrics["median_depth_error_proxy_m"]
    overlap_depth = metrics["overlap_depth_residual_median_proxy_m"]
    median_term = 2.0 if median_depth is None else min(float(median_depth) / depth_norm, 2.0)
    overlap_term = 2.0 if overlap_depth is None else min(float(overlap_depth) / depth_norm, 2.0)
    terms = {
        "silhouette_iou": 1.0 - float(metrics["iou"]),
        "silhouette_centroid": min(
            float(metrics["centroid_error_fraction_of_image_diagonal"]) / 0.05, 2.0
        ),
        "silhouette_extent": min(float(metrics["extent_log_error"]), 2.0),
        "silhouette_area": min(float(metrics["area_log_error"]), 2.0),
        "depth_median": median_term,
        "depth_overlap_residual": overlap_term,
        "canonical_view_prior": min(abs(canonical_view_angle_rad) / math.pi, 1.0),
    }
    return float(sum(OBJECTIVE_WEIGHTS[key] * terms[key] for key in terms))


def cv_sim3_to_pyrender(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("Sim(3) matrix must have shape (4,4)")
    return T_CV_TO_GL @ matrix


class MeshRenderer:
    """Small pyrender wrapper for repeated camera-Z/silhouette evaluations."""

    def __init__(
        self,
        mesh: Any,
        *,
        width: int,
        height: int,
        camera_matrix: np.ndarray,
        textured: bool = False,
    ) -> None:
        import pyrender  # noqa: PLC0415
        import trimesh  # noqa: PLC0415

        self._pyrender = pyrender
        self.width = int(width)
        self.height = int(height)
        self.renderer = pyrender.OffscreenRenderer(self.width, self.height)
        self.scene = pyrender.Scene(
            bg_color=(0.025, 0.035, 0.055, 1.0),
            ambient_light=(0.45, 0.45, 0.45),
        )
        k = np.asarray(camera_matrix, dtype=np.float64)
        self.scene.add(
            pyrender.IntrinsicsCamera(
                fx=float(k[0, 0]),
                fy=float(k[1, 1]),
                cx=float(k[0, 2]),
                cy=float(k[1, 2]),
                znear=0.01,
                zfar=5.0,
            ),
            pose=np.eye(4),
        )
        if textured:
            try:
                render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
            except Exception:
                textured = False
        if not textured:
            plain = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.faces),
                process=False,
            )
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=(0.15, 0.72, 0.92, 1.0),
                metallicFactor=0.05,
                roughnessFactor=0.65,
            )
            render_mesh = pyrender.Mesh.from_trimesh(
                plain, material=material, smooth=True
            )
        self.node = self.scene.add(render_mesh, pose=np.eye(4))
        self.scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
            pose=np.eye(4),
        )
        fill_pose = np.eye(4)
        fill_pose[:3, 3] = (0.25, -0.2, 0.2)
        self.scene.add(
            pyrender.PointLight(color=(0.7, 0.8, 1.0), intensity=1.5),
            pose=fill_pose,
        )

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.delete()
            self.renderer = None

    def render(
        self, matrix: np.ndarray, *, color: bool = False
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
        self.scene.set_pose(self.node, cv_sim3_to_pyrender(matrix))
        if color:
            rgba, depth = self.renderer.render(
                self.scene, flags=self._pyrender.RenderFlags.RGBA
            )
            rgb: np.ndarray | None = np.asarray(rgba[..., :3], dtype=np.uint8)
        else:
            depth = self.renderer.render(
                self.scene, flags=self._pyrender.RenderFlags.DEPTH_ONLY
            )
            rgb = None
        depth = np.asarray(depth, dtype=np.float32)
        mask = np.isfinite(depth) & (depth > 0.01) & (depth < 5.0)
        depth = np.where(mask, depth, 0.0).astype(np.float32)
        return rgb, depth, mask


def load_canonical_mesh(path: str | Path) -> tuple[Any, dict[str, Any]]:
    import trimesh  # noqa: PLC0415

    mesh_path = _resolved_file(path, "SPAR3D mesh.glb")
    loaded = trimesh.load(mesh_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        parts = [loaded]
    else:
        dumped = loaded.dump(concatenate=False)
        parts = [part for part in np.atleast_1d(dumped) if isinstance(part, trimesh.Trimesh)]
    if not parts:
        raise RegistrationInputError("SPAR3D GLB contains no triangle mesh")
    mesh = parts[0].copy() if len(parts) == 1 else trimesh.util.concatenate(parts)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not len(faces):
        raise RegistrationInputError("SPAR3D GLB mesh is empty or malformed")
    if not np.isfinite(vertices).all():
        raise RegistrationInputError("SPAR3D mesh contains non-finite vertices")
    extents = np.asarray(mesh.extents, dtype=np.float64)
    if np.any(extents <= 1.0e-8):
        raise RegistrationInputError("SPAR3D mesh is degenerate")
    return mesh, {
        "path": str(mesh_path),
        "bytes": mesh_path.stat().st_size,
        "sha256": sha256_file(mesh_path),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "bounds_canonical": np.asarray(mesh.bounds).tolist(),
        "extents_canonical": extents.tolist(),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
    }


def validate_spar_mesh_record(
    spar_report: dict[str, Any], *, mesh_path: Path, mesh_sha256: str
) -> None:
    """Require the GLB to match the atomically published SPAR3D report."""

    try:
        record = spar_report["outputs"]["mesh_glb"]
        declared_path = Path(record["path"]).expanduser().resolve()
        declared_bytes = int(record["bytes"])
        declared_sha256 = str(record["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RegistrationInputError(
            "SPAR3D report lacks a complete outputs.mesh_glb record"
        ) from exc
    if declared_path != mesh_path:
        raise RegistrationInputError(
            f"SPAR3D mesh path differs from its report: {mesh_path} != {declared_path}"
        )
    if declared_bytes != mesh_path.stat().st_size:
        raise RegistrationInputError("SPAR3D mesh byte count differs from its report")
    if declared_sha256 != mesh_sha256:
        raise RegistrationInputError("SPAR3D mesh SHA-256 differs from its report")


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.acos(float(cosine)))


class RegistrationEvaluator:
    def __init__(
        self,
        renderer: MeshRenderer,
        observed_mask: np.ndarray,
        observed_depth: np.ndarray,
        *,
        depth_reference_m: float,
    ) -> None:
        self.renderer = renderer
        self.observed_mask = np.asarray(observed_mask, dtype=bool)
        self.observed_depth = np.asarray(observed_depth, dtype=np.float32)
        self.depth_reference_m = float(depth_reference_m)

    def evaluate(
        self, matrix: np.ndarray, rotation: np.ndarray
    ) -> tuple[float, dict[str, float | int | None]]:
        _rgb, rendered_depth, rendered_mask = self.renderer.render(matrix)
        metrics = alignment_metrics(
            self.observed_mask,
            self.observed_depth,
            rendered_mask,
            rendered_depth,
        )
        loss = registration_loss(
            metrics,
            depth_reference_m=self.depth_reference_m,
            canonical_view_angle_rad=_rotation_angle(rotation),
        )
        return loss, metrics


def select_fit_stage(
    *,
    naive: dict[str, Any],
    coarse: dict[str, Any],
    local: dict[str, Any],
    optimizer_success: bool,
) -> tuple[str, dict[str, Any], str | None]:
    """Select a finite non-regressing stage, excluding failed local fits."""

    eligible = [("naive", naive), ("coarse", coarse)]
    fallback_reason: str | None = None
    if optimizer_success:
        eligible.append(("local", local))
    else:
        fallback_reason = "local_optimizer_failed"
    finite = [
        (name, stage)
        for name, stage in eligible
        if math.isfinite(float(stage.get("loss", math.inf)))
    ]
    if not finite:
        raise RegistrationInputError("all registration stages produced non-finite loss")
    selected_name, selected = min(finite, key=lambda item: float(item[1]["loss"]))
    if optimizer_success and selected_name != "local":
        fallback_reason = "local_fit_regressed"
    return selected_name, selected, fallback_reason


def fit_approximate_sim3(
    mesh: Any,
    observed_mask: np.ndarray,
    observed_depth: np.ndarray,
    camera_matrix: np.ndarray,
    observation: ObservationSummary,
    *,
    optimization_height: int = 180,
    max_function_evaluations: int = 220,
    renderer_factory: Callable[..., MeshRenderer] = MeshRenderer,
) -> dict[str, Any]:
    """Coarse signed-axis search followed by bounded local silhouette/depth fit."""

    from scipy.optimize import minimize  # noqa: PLC0415
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    source_shape = observed_mask.shape
    ratio = optimization_height / source_shape[0]
    target_shape = (
        optimization_height,
        max(32, int(round(source_shape[1] * ratio))),
    )
    low_mask = cv2.resize(
        observed_mask.astype(np.uint8),
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    low_depth = resize_depth_valid(observed_depth, target_shape)
    low_k = scaled_camera_matrix(camera_matrix, source_shape, target_shape)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    center = np.asarray(mesh.bounds, dtype=np.float64).mean(axis=0)
    centered = vertices - center
    renderer = renderer_factory(
        mesh,
        width=target_shape[1],
        height=target_shape[0],
        camera_matrix=low_k,
        textured=False,
    )
    evaluator = RegistrationEvaluator(
        renderer,
        low_mask,
        low_depth,
        depth_reference_m=observation.median_depth_proxy_m,
    )
    try:
        candidates: list[dict[str, Any]] = []
        naive_rotation = np.eye(3)
        naive_scale, naive_translation = _initial_scale_translation(
            centered, naive_rotation, observation, camera_matrix
        )
        naive_matrix = make_sim3(
            naive_scale,
            naive_rotation,
            naive_translation,
            canonical_center=center,
        )
        naive_loss, naive_metrics = evaluator.evaluate(naive_matrix, naive_rotation)

        for index, rotation in enumerate(proper_axis_rotations()):
            scale, translation = _initial_scale_translation(
                centered, rotation, observation, camera_matrix
            )
            matrix = make_sim3(
                scale, rotation, translation, canonical_center=center
            )
            loss, metrics = evaluator.evaluate(matrix, rotation)
            candidates.append(
                {
                    "index": index,
                    "loss": loss,
                    "scale": scale,
                    "translation": translation,
                    "rotation": rotation,
                    "matrix": matrix,
                    "metrics": metrics,
                }
            )
        candidates.sort(key=lambda item: item["loss"])
        coarse = candidates[0]
        base_rotation = coarse["rotation"]
        base_scale = float(coarse["scale"])
        base_translation = np.asarray(coarse["translation"], dtype=np.float64)
        z0 = observation.median_depth_proxy_m
        center_z0 = float(base_translation[2])
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        u0, v0 = observation.centroid_xy
        bbox = observation.bbox_xyxy_exclusive
        width_px, height_px = bbox[2] - bbox[0], bbox[3] - bbox[1]

        def decode(parameters: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
            delta = Rotation.from_rotvec(parameters[:3]).as_matrix()
            rotation = delta @ base_rotation
            scale = base_scale * math.exp(float(parameters[3]))
            u = u0 + float(parameters[4])
            v = v0 + float(parameters[5])
            z = center_z0 + float(parameters[6])
            translation = np.asarray(((u - cx) * z / fx, (v - cy) * z / fy, z))
            matrix = make_sim3(
                scale, rotation, translation, canonical_center=center
            )
            return rotation, scale, translation, matrix

        def objective(parameters: np.ndarray) -> float:
            try:
                rotation, _scale, _translation, matrix = decode(parameters)
                loss, _metrics = evaluator.evaluate(matrix, rotation)
                return loss
            except Exception:
                return 10.0

        angle = math.radians(35.0)
        bounds = [
            (-angle, angle),
            (-angle, angle),
            (-angle, angle),
            (-math.log(2.0), math.log(2.0)),
            (-float(width_px), float(width_px)),
            (-float(height_px), float(height_px)),
            (-0.25 * z0, 0.25 * z0),
        ]
        optimization = minimize(
            objective,
            np.zeros(7, dtype=np.float64),
            method="Powell",
            bounds=bounds,
            options={
                "maxfev": int(max_function_evaluations),
                "xtol": 2.0e-3,
                "ftol": 2.0e-4,
            },
        )
        try:
            local_rotation, local_scale, local_translation, local_matrix = decode(
                optimization.x
            )
            local_loss, local_metrics = evaluator.evaluate(
                local_matrix, local_rotation
            )
        except Exception:
            local_rotation = base_rotation
            local_scale = base_scale
            local_translation = base_translation
            local_matrix = coarse["matrix"]
            local_loss = math.inf
            local_metrics = coarse["metrics"]

        naive_stage = {
            "matrix": naive_matrix,
            "rotation": naive_rotation,
            "scale": naive_scale,
            "translation": naive_translation,
            "loss": naive_loss,
            "metrics_low_resolution": naive_metrics,
        }
        coarse_stage = {
            "matrix": coarse["matrix"],
            "rotation": coarse["rotation"],
            "scale": coarse["scale"],
            "translation": coarse["translation"],
            "loss": coarse["loss"],
            "metrics_low_resolution": coarse["metrics"],
        }
        local_stage = {
            "matrix": local_matrix,
            "rotation": local_rotation,
            "scale": local_scale,
            "translation": local_translation,
            "loss": local_loss,
            "metrics_low_resolution": local_metrics,
        }
        selected_stage, selected, fallback_reason = select_fit_stage(
            naive=naive_stage,
            coarse=coarse_stage,
            local=local_stage,
            optimizer_success=bool(optimization.success),
        )
        return {
            "canonical_center": center,
            "naive": naive_stage,
            "coarse": coarse,
            "coarse_candidates": candidates,
            "local": local_stage,
            "final": {**selected, "selected_stage": selected_stage},
            "optimizer": {
                "method": "Powell bounded local refinement",
                "success": bool(optimization.success),
                "message": str(optimization.message),
                "function_evaluations": int(optimization.nfev),
                "iterations": int(getattr(optimization, "nit", 0)),
                "parameters": np.asarray(optimization.x).tolist(),
                "bounds": [list(item) for item in bounds],
                "optimization_shape_hw": list(target_shape),
                "selected_stage": selected_stage,
                "fallback_reason": fallback_reason,
            },
        }
    finally:
        renderer.close()


def _contours(mask: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(mask, dtype=np.uint8) * 255
    result = cv2.findContours(values, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return result[-2]


def draw_alignment_overlay(
    image_bgr: np.ndarray,
    observed_mask: np.ndarray,
    rendered_mask: np.ndarray,
    rendered_rgb: np.ndarray | None,
    *,
    title: str,
    metrics: dict[str, float | int | None],
) -> np.ndarray:
    output = np.asarray(image_bgr, dtype=np.uint8).copy()
    if rendered_rgb is not None:
        rendered_bgr = np.asarray(rendered_rgb)[..., ::-1]
        alpha = 0.32
        output[rendered_mask] = np.clip(
            (1 - alpha) * output[rendered_mask].astype(np.float32)
            + alpha * rendered_bgr[rendered_mask].astype(np.float32),
            0,
            255,
        ).astype(np.uint8)
    cv2.drawContours(output, _contours(observed_mask), -1, (70, 255, 70), 3)
    cv2.drawContours(output, _contours(rendered_mask), -1, (255, 220, 30), 3)
    cv2.rectangle(output, (8, 8), (690, 78), (16, 16, 16), -1)
    cv2.putText(
        output,
        title,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    text = (
        f"green=MH mask  cyan=mesh  IoU={float(metrics['iou']):.3f}  "
        f"depth err={float(metrics['median_depth_error_proxy_m'] or 0):.3f} proxy-m"
    )
    cv2.putText(
        output,
        text,
        (20, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return output


def colorize_depth(depth: np.ndarray, support: np.ndarray, title: str) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(support, dtype=bool) & np.isfinite(values) & (values > 0)
    canvas = np.zeros((*values.shape, 3), dtype=np.uint8)
    if valid.any():
        low, high = np.quantile(values[valid], (0.02, 0.98))
        normalized = np.zeros(values.shape, dtype=np.uint8)
        normalized[valid] = np.clip(
            255 * (values[valid] - low) / max(float(high - low), 1.0e-5),
            0,
            255,
        ).astype(np.uint8)
        canvas = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
        canvas[~valid] = 0
    cv2.putText(
        canvas,
        title,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def make_canonical_contact_sheet(
    mesh: Any,
    canonical_center: np.ndarray,
    *,
    cell_size: int = 320,
) -> np.ndarray:
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    k = np.asarray(
        ((0.9 * cell_size, 0, cell_size / 2),
         (0, 0.9 * cell_size, cell_size / 2),
         (0, 0, 1)),
        dtype=np.float64,
    )
    renderer = MeshRenderer(
        mesh,
        width=cell_size,
        height=cell_size,
        camera_matrix=k,
        textured=True,
    )
    extent = float(np.max(np.asarray(mesh.extents)))
    scale = 1.35 / max(extent, 1.0e-8)
    cells: list[np.ndarray] = []
    try:
        for azimuth in range(0, 360, 45):
            rotation = Rotation.from_euler(
                "yx", (azimuth, -12.0), degrees=True
            ).as_matrix()
            matrix = make_sim3(
                scale,
                rotation,
                np.asarray((0.0, 0.0, 3.0)),
                canonical_center=canonical_center,
            )
            rgb, _depth, _mask = renderer.render(matrix, color=True)
            assert rgb is not None
            cell = rgb[..., ::-1].copy()
            cv2.putText(
                cell,
                f"canonical {azimuth:03d} deg",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cells.append(cell)
    finally:
        renderer.close()
    return np.vstack((np.hstack(cells[:4]), np.hstack(cells[4:])))


def write_comparison_video(path: Path, before: np.ndarray, after: np.ndarray) -> bool:
    panel_width = 640
    panel_height = int(round(before.shape[0] * panel_width / before.shape[1]))
    before_small = cv2.resize(before, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    after_small = cv2.resize(after, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    canvas = np.hstack((before_small, after_small))
    fps = 24.0
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas.shape[1], canvas.shape[0])
    )
    if not writer.isOpened():
        return False
    try:
        for _ in range(int(3 * fps)):
            writer.write(canvas)
    finally:
        writer.release()
    return path.is_file() and path.stat().st_size > 0


def registration_contract() -> dict[str, Any]:
    """Stable provenance fields used by reports and weight-free tests."""

    return {
        "method": METHOD,
        "representation": REPRESENTATION,
        "metric_scale_verified": False,
        "camera_alignment": "approximate_MH_camera_Sim3",
        "physical_geometry_guarantee": False,
        "collision_ready": False,
        "uses_mh_calibrated_intrinsics": True,
        "uses_mh_lens_distortion": True,
        "uses_hawor_anchored_depth_proxy": True,
        "uses_sh_checker_square_translation_as_metric": False,
        "uses_sh_for_this_registration": False,
        "warnings": list(PROVENANCE_WARNINGS),
    }


def preflight_job(
    *,
    input_manifest: str | Path,
    mh_image: str | Path,
    mh_mask: str | Path,
    scene_depth: str | Path,
    depth_params: str | Path,
    mesh_glb: str | Path,
    spar_report: str | Path,
) -> dict[str, Any]:
    manifest_path, manifest = load_pilot_manifest(input_manifest)
    image_path, mask_path, image, mask = load_image_and_mask(mh_image, mh_mask)
    validate_selected_input_records(
        manifest, image_path=image_path, mask_path=mask_path
    )
    frame_index = int(manifest["selection"]["mh_frame_index"])
    camera = parse_mh_calibration(manifest, image.shape[:2])
    depth_path, depth, depth_shape, depth_dtype = load_depth_frame(
        scene_depth, frame_index=frame_index, expected_shape=image.shape[:2]
    )
    params_path, anchor = load_depth_anchor_record(depth_params, frame_index=frame_index)
    depth_provenance = validate_depth_source_provenance(
        depth_path=depth_path,
        params_path=params_path,
        anchor=anchor,
    )
    valid_depth_samples = int(
        np.count_nonzero(mask & np.isfinite(depth) & (depth > 0))
    )
    if valid_depth_samples < 30:
        raise RegistrationInputError("too few positive depth values in the MH mask")

    mesh_path = Path(mesh_glb).expanduser().resolve()
    report_path = Path(spar_report).expanduser().resolve()
    mesh_ready = mesh_path.is_file() and mesh_path.stat().st_size > 0
    spar_report_ready = report_path.is_file() and report_path.stat().st_size > 0
    if mesh_ready != spar_report_ready:
        status = "blocked_incomplete_spar3d_bundle"
    elif mesh_ready:
        status = "ready"
    else:
        status = "waiting_for_spar3d_mesh"
    result = {
        "schema_version": 1,
        "status": status,
        **registration_contract(),
        "selection": manifest["selection"],
        "inputs": {
            "manifest": str(manifest_path),
            "mh_image": str(image_path),
            "mh_modal_mask": str(mask_path),
            "scene_depth": str(depth_path),
            "depth_params": str(params_path),
            "mesh_glb_expected": str(mesh_path),
            "spar_report_expected": str(report_path),
        },
        "input_checks": {
            "image_shape_hwc": list(image.shape),
            "mask_pixels": int(mask.sum()),
            "depth_array_shape": list(depth_shape),
            "depth_array_dtype": depth_dtype,
            "positive_depth_samples_inside_mask": valid_depth_samples,
            "mesh_ready": mesh_ready,
            "spar_report_ready": spar_report_ready,
        },
        "mh_calibration": {
            "camera_matrix": camera.camera_matrix.tolist(),
            "distortion_k1_k2_p1_p2_k3": camera.distortion.tolist(),
            "fit_domain": "undistorted MH pinhole image using the same K matrix",
        },
        "depth_anchor": anchor,
        "depth_source_provenance": depth_provenance,
        "stereo_scale_guard": {
            "manifest_translation_unit": manifest["calibration"][
                "relative_extrinsics"
            ].get("translation_unit"),
            "checkerboard_metric_scale_verified": manifest["calibration"][
                "checkerboard"
            ].get("metric_scale_verified"),
            "translation_used": False,
        },
    }
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _output_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return true when either resolved path contains the other."""

    first = first.expanduser().resolve()
    second = second.expanduser().resolve()
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def validate_output_location(
    output_dir: str | Path,
    *,
    input_manifest: str | Path,
    mh_image: str | Path,
    mh_mask: str | Path,
    scene_depth: str | Path,
    depth_params: str | Path,
    mesh_glb: str | Path,
    spar_report: str | Path,
) -> Path:
    """Reject an overwrite target that can contain or replace source data."""

    output = Path(output_dir).expanduser().resolve()
    repo = REPO_ROOT.resolve()
    if output == repo or repo.is_relative_to(output):
        raise RegistrationInputError(
            f"registration output may not equal or contain the repository root: {output}"
        )

    protected_trees = {
        "pilot input bundle": Path(input_manifest).expanduser().resolve().parent,
        "depth processor": Path(scene_depth).expanduser().resolve().parent,
        "SPAR3D output bundle": Path(spar_report).expanduser().resolve().parent,
        "project weights": (repo / "weights").resolve(),
        "third-party repositories": (repo / "third_party").resolve(),
    }
    for label, protected in protected_trees.items():
        if _paths_overlap(output, protected):
            raise RegistrationInputError(
                f"registration output overlaps protected {label}: {output} vs {protected}"
            )

    protected_files = {
        "MH image": mh_image,
        "MH mask": mh_mask,
        "depth parameters": depth_params,
        "SPAR3D mesh": mesh_glb,
    }
    for label, value in protected_files.items():
        protected = Path(value).expanduser().resolve()
        if output == protected or protected.is_relative_to(output):
            raise RegistrationInputError(
                f"registration output would contain protected {label}: {protected}"
            )
    return output


def validate_spar_bundle_provenance(
    spar_report: dict[str, Any],
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    """Bind a SPAR result to this exact input bundle and selected frames."""

    try:
        report_manifest = spar_report["input_manifest"]
        report_input = spar_report["input"]
        report_selection = spar_report["selection"]
        rgba_record = manifest["outputs"]["spar3d_rgba_crop"]
    except (KeyError, TypeError) as exc:
        raise RegistrationInputError(
            "SPAR3D report lacks input/input_manifest/selection provenance"
        ) from exc

    current_manifest_sha = sha256_file(manifest_path)
    if Path(report_manifest.get("path", "")).expanduser().resolve() != manifest_path:
        raise RegistrationInputError("SPAR3D report references a different input manifest")
    if str(report_manifest.get("sha256")) != current_manifest_sha:
        raise RegistrationInputError("SPAR3D report input-manifest SHA-256 is stale")

    rgba_path = Path(rgba_record.get("path", "")).expanduser().resolve()
    if not rgba_path.is_file():
        raise RegistrationInputError("manifest SPAR3D RGBA input is missing")
    if Path(report_input.get("path", "")).expanduser().resolve() != rgba_path:
        raise RegistrationInputError("SPAR3D report used a different RGBA input")
    actual_rgba_sha = sha256_file(rgba_path)
    if str(rgba_record.get("sha256")) != actual_rgba_sha or str(
        report_input.get("sha256")
    ) != actual_rgba_sha:
        raise RegistrationInputError("SPAR3D RGBA SHA-256 provenance mismatch")
    if int(rgba_record.get("bytes", -1)) != rgba_path.stat().st_size or int(
        report_input.get("bytes", -1)
    ) != rgba_path.stat().st_size:
        raise RegistrationInputError("SPAR3D RGBA byte-count provenance mismatch")

    selected = manifest["selection"]
    for key in ("object_label", "mh_frame_index", "sh_frame_index"):
        if report_selection.get(key) != selected.get(key):
            raise RegistrationInputError(
                f"SPAR3D report selection differs for {key}"
            )


def _publish_directory(staging: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"output directory exists (pass --overwrite): {output}"
            )
        backup = output.with_name(f".{output.name}.backup")
        if backup.exists():
            raise FileExistsError(f"stale registration backup exists: {backup}")
        output.replace(backup)
        try:
            staging.replace(output)
        except BaseException:
            backup.replace(output)
            raise
        shutil.rmtree(backup)
    else:
        staging.replace(output)


def run_registration(
    *,
    input_manifest: str | Path,
    mh_image: str | Path,
    mh_mask: str | Path,
    scene_depth: str | Path,
    depth_params: str | Path,
    mesh_glb: str | Path,
    spar_report: str | Path,
    output_dir: str | Path,
    optimization_height: int = 180,
    max_function_evaluations: int = 220,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = validate_output_location(
        output_dir,
        input_manifest=input_manifest,
        mh_image=mh_image,
        mh_mask=mh_mask,
        scene_depth=scene_depth,
        depth_params=depth_params,
        mesh_glb=mesh_glb,
        spar_report=spar_report,
    )
    preflight = preflight_job(
        input_manifest=input_manifest,
        mh_image=mh_image,
        mh_mask=mh_mask,
        scene_depth=scene_depth,
        depth_params=depth_params,
        mesh_glb=mesh_glb,
        spar_report=spar_report,
    )
    if preflight["status"] != "ready":
        raise FileNotFoundError(
            f"SPAR3D bundle is not ready: {preflight['status']} "
            f"({preflight['inputs']['mesh_glb_expected']})"
        )
    manifest_path, manifest = load_pilot_manifest(input_manifest)
    image_path, mask_path, image, mask = load_image_and_mask(mh_image, mh_mask)
    frame_index = int(manifest["selection"]["mh_frame_index"])
    camera = parse_mh_calibration(manifest, image.shape[:2])
    depth_path, depth, _shape, _dtype = load_depth_frame(
        scene_depth, frame_index=frame_index, expected_shape=image.shape[:2]
    )
    params_path, anchor = load_depth_anchor_record(depth_params, frame_index=frame_index)
    spar_report_path, spar_payload = _load_json(spar_report, "SPAR3D report")
    if spar_payload.get("camera_alignment") != "none" or spar_payload.get(
        "metric_scale_verified"
    ) is not False:
        raise RegistrationInputError(
            "SPAR3D report must identify a canonical non-metric, unregistered mesh"
        )
    validate_spar_bundle_provenance(
        spar_payload,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    mesh, mesh_source = load_canonical_mesh(mesh_glb)
    validate_spar_mesh_record(
        spar_payload,
        mesh_path=Path(mesh_source["path"]),
        mesh_sha256=str(mesh_source["sha256"]),
    )

    image_u, mask_u, depth_u, undistorted_k = undistort_observation(
        image, mask, depth, camera
    )
    surface_u, surface_filter = robust_object_surface(depth_u, mask_u)
    observation = summarize_observation(mask_u, surface_u, undistorted_k)
    fit = fit_approximate_sim3(
        mesh,
        mask_u,
        surface_u,
        undistorted_k,
        observation,
        optimization_height=optimization_height,
        max_function_evaluations=max_function_evaluations,
    )

    full_renderer = MeshRenderer(
        mesh,
        width=image_u.shape[1],
        height=image_u.shape[0],
        camera_matrix=undistorted_k,
        textured=True,
    )
    try:
        before_rgb, before_depth, before_mask = full_renderer.render(
            fit["naive"]["matrix"], color=True
        )
        after_rgb, after_depth, after_mask = full_renderer.render(
            fit["final"]["matrix"], color=True
        )
    finally:
        full_renderer.close()
    before_metrics = alignment_metrics(
        mask_u, surface_u, before_mask, before_depth
    )
    after_metrics = alignment_metrics(mask_u, surface_u, after_mask, after_depth)

    before_overlay = draw_alignment_overlay(
        image_u,
        mask_u,
        before_mask,
        before_rgb,
        title="Before: canonical view + bbox/depth initialization",
        metrics=before_metrics,
    )
    after_overlay = draw_alignment_overlay(
        image_u,
        mask_u,
        after_mask,
        after_rgb,
        title="After: approximate Sim(3) silhouette + depth fit",
        metrics=after_metrics,
    )
    observed_depth_panel = colorize_depth(
        surface_u, mask_u, "Observed: DA-V2 + HaWoR camera-Z proxy"
    )
    residual = np.zeros_like(surface_u)
    paired = mask_u & after_mask & (surface_u > 0) & (after_depth > 0)
    residual[paired] = np.abs(surface_u[paired] - after_depth[paired])
    residual_panel = colorize_depth(
        residual, paired, "After: |mesh Z - observed proxy Z|"
    )
    diagnostic = np.vstack(
        (
            np.hstack((before_overlay, after_overlay)),
            np.hstack((observed_depth_panel, residual_panel)),
        )
    )
    before_after = np.hstack((before_overlay, after_overlay))
    contact_sheet = make_canonical_contact_sheet(mesh, fit["canonical_center"])

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output directory exists (pass --overwrite): {output}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        registered_mesh = mesh.copy()
        registered_mesh.apply_transform(fit["final"]["matrix"])
        registered_mesh.export(staging / "registered_mesh_mh.glb", include_normals=True)
        np.savez(
            staging / "registration_transform.npz",
            sim3_canonical_to_mh_camera=fit["final"]["matrix"].astype(np.float64),
            isotropic_scale_proxy_m_per_canonical_unit=np.float64(fit["final"]["scale"]),
            rotation_canonical_to_mh=fit["final"]["rotation"].astype(np.float64),
            canonical_center_position_mh_camera_proxy_m=fit["final"]["translation"].astype(np.float64),
            sim3_homogeneous_translation_term_proxy_m=fit["final"]["matrix"][:3, 3].astype(np.float64),
            canonical_center=fit["canonical_center"].astype(np.float64),
            metric_scale_verified=np.uint8(0),
        )
        np.save(staging / "registered_front_depth_proxy_m.npy", after_depth.astype(np.float32))
        cv2.imwrite(
            str(staging / "registered_silhouette.png"), after_mask.astype(np.uint8) * 255
        )
        cv2.imwrite(str(staging / "canonical_turntable_contact_sheet.png"), contact_sheet)
        cv2.imwrite(str(staging / "mh_silhouette_depth_alignment.png"), diagnostic)
        cv2.imwrite(str(staging / "before_after_registration.png"), before_after)
        video_written = write_comparison_video(
            staging / "before_after_registration.mp4", before_overlay, after_overlay
        )

        coarse_report = [
            {
                "rank": rank,
                "signed_axis_rotation_index": int(item["index"]),
                "loss": float(item["loss"]),
                "iou": float(item["metrics"]["iou"]),
                "scale_proxy_m_per_canonical_unit": float(item["scale"]),
            }
            for rank, item in enumerate(fit["coarse_candidates"][:8], start=1)
        ]
        report = {
            "schema_version": 1,
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            **registration_contract(),
            "selection": manifest["selection"],
            "source_mesh": mesh_source,
            "source_spar3d_report": {
                "path": str(spar_report_path),
                "sha256": sha256_file(spar_report_path),
                "source_representation": spar_payload.get("representation"),
                "hidden_geometry_is_learned_estimate": True,
            },
            "sources": {
                "input_manifest": _output_record(manifest_path),
                "mh_image": _output_record(image_path),
                "mh_modal_mask": _output_record(mask_path),
                "depth_aligned_proxy": preflight["depth_source_provenance"]["depth_array"],
                "depth_anchor_params": preflight["depth_source_provenance"]["anchor_params"],
            },
            "camera": {
                "coordinate_frame": "MH OpenCV camera: +X right, +Y down, +Z forward",
                "camera_matrix_original_and_undistorted": undistorted_k.tolist(),
                "distortion_k1_k2_p1_p2_k3": camera.distortion.tolist(),
                "fit_domain": "undistorted full-resolution MH image; K retained",
                "renderer": "pyrender EGL pinhole camera",
            },
            "depth_proxy": {
                **anchor,
                "surface_filter": surface_filter,
                "camera_z_not_euclidean_ray_depth": True,
            },
            "observation": asdict(observation),
            "sim3": {
                "matrix_canonical_to_mh_camera": fit["final"]["matrix"].tolist(),
                "rotation_canonical_to_mh": fit["final"]["rotation"].tolist(),
                "canonical_center_position_mh_camera_proxy_m": fit["final"]["translation"].tolist(),
                "sim3_homogeneous_translation_term_proxy_m": fit["final"]["matrix"][:3, 3].tolist(),
                "isotropic_scale_proxy_m_per_canonical_unit": float(fit["final"]["scale"]),
                "canonical_center": fit["canonical_center"].tolist(),
                "scale_semantics": "approximate overlay proxy metres per SPAR3D canonical unit",
                "metric_scale_verified": False,
            },
            "objective": {
                "weights": OBJECTIVE_WEIGHTS,
                "coarse_search": "24 proper signed-axis rotations",
                "coarse_top_candidates": coarse_report,
                "local_optimizer": fit["optimizer"],
                "selected_stage": fit["final"]["selected_stage"],
                "naive_low_resolution_loss": float(fit["naive"]["loss"]),
                "final_low_resolution_loss": float(fit["final"]["loss"]),
            },
            "comparison": {
                "before_definition": "identity canonical view with silhouette-bbox scale and depth-centroid translation",
                "after_definition": (
                    "lowest finite non-regressing stage among naive, coarse, and "
                    "successful bounded-local approximate Sim(3) fits"
                ),
                "before_full_resolution": before_metrics,
                "after_full_resolution": after_metrics,
                "iou_delta": float(after_metrics["iou"] - before_metrics["iou"]),
                "video_written": bool(video_written),
                "static_fallback_always_written": True,
            },
            "stereo_scale_guard": preflight["stereo_scale_guard"],
            "assumptions": [
                f"The selected MH frame {frame_index} is used because modal, clean, and amodal masks are exactly equal there.",
                "SPAR3D canonical topology is retained; only one global isotropic scale, rotation, and translation are fitted.",
                "The robust median Depth Anything/HaWoR camera-Z inside the modal mask initializes and constrains object Z.",
                "Lens distortion is handled by undistorting image, mask, and depth to the calibrated MH pinhole domain.",
                "SH is reserved for later held-out evidence; its non-metric checker-square translation is excluded from this fit.",
            ],
            "outputs": {},
        }
        output_names = (
            "registered_mesh_mh.glb",
            "registration_transform.npz",
            "registered_front_depth_proxy_m.npy",
            "registered_silhouette.png",
            "canonical_turntable_contact_sheet.png",
            "mh_silhouette_depth_alignment.png",
            "before_after_registration.png",
        )
        if video_written:
            output_names += ("before_after_registration.mp4",)
        for name in output_names:
            report["outputs"][name] = _output_record(staging / name)
            report["outputs"][name]["path"] = str(output / name)
        report["outputs"]["report.json"] = {"path": str(output / "report.json")}
        _write_json(staging / "report.json", report)
        _publish_directory(staging, output, overwrite=overwrite)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--mh-image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--mh-mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--scene-depth", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--depth-params", type=Path, default=DEFAULT_DEPTH_PARAMS)
    parser.add_argument("--mesh-glb", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--spar-report", type=Path, default=DEFAULT_SPAR_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--optimization-height", type=int, default=180)
    parser.add_argument("--max-function-evaluations", type=int, default=220)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    common = {
        "input_manifest": args.input_manifest,
        "mh_image": args.mh_image,
        "mh_mask": args.mh_mask,
        "scene_depth": args.scene_depth,
        "depth_params": args.depth_params,
        "mesh_glb": args.mesh_glb,
        "spar_report": args.spar_report,
    }
    if args.preflight:
        result = preflight_job(**common)
    else:
        result = run_registration(
            **common,
            output_dir=args.output_dir,
            optimization_height=args.optimization_height,
            max_function_evaluations=args.max_function_evaluations,
            overwrite=args.overwrite,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
