"""Fit nominal closed object meshes and render MH-camera front/back depth.

The 08_04 annotations identify the manipulated object but do not contain an
object pose.  This stage reuses the metric primitive dimensions already stored
in each object's MJCF, builds a watertight union mesh, and estimates a visual
MH-camera pose from the completed object silhouette/depth.  Per-frame poses are
expressed relative to the rendered XHand wrist and robustly smoothed inside
each annotated object interval.

The resulting rear surface is a *nominal model prior*.  It is not measured by
HaCo, monocular depth, or the uncalibrated auxiliary camera.  Frames whose
rendered silhouette/depth checks do not support the fitted pose fail open: the
pose is NaN and every mesh raster is zero for that frame.

Outputs under ``--out_dir``::

    canonical_meshes/<object_id>.ply       watertight nominal union meshes
    object_pose_cam.npy                    float32 (T,4,4), NaN if invalid
    pose_valid.npy                         bool (T,)
    pose_confidence.npy                    float32 (T,)
    object_mesh_front_depth.npy            float16 (T,H,W), 0 unknown
    object_mesh_back_depth.npy             float16 (T,H,W), 0 unknown
    object_mesh_mask.npy                   bool (T,H,W)
    fit_evidence.npz                       per-frame diagnostics
    debug_object_mesh_volume.mp4           optional when --video is supplied
    report.json

Pyrender's ordinary winding yields the first front-facing surface.  Reversing
all faces yields the paired back-facing exit of that visible material layer.
Pixels without a finite ordered pair are deliberately excluded.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import trimesh
import yaml
from scipy.spatial.transform import Rotation
from skimage.measure import marching_objects

from atomic_directory_publish import publish_directory


REPO_ROOT = Path(__file__).resolve().parents[2]
METHOD = "nominal_mjcf_mesh_volume_wrist_relative_fit"
REPRESENTATION = "fitted_watertight_nominal_mesh_front_back_camera_z"
T_CV2GL = np.diag((1.0, -1.0, -1.0))
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
OFFSCREEN_POSE = np.asarray(
    ((1.0, 0.0, 0.0, 0.0),
     (0.0, 1.0, 0.0, 0.0),
     (0.0, 0.0, 1.0, 5.0),
     (0.0, 0.0, 0.0, 1.0)),
    dtype=np.float64,
)


@dataclass(frozen=True)
class Primitive:
    """One physical MJCF primitive in canonical object coordinates."""

    kind: str
    center: np.ndarray
    rotation: np.ndarray
    size: np.ndarray
    endpoint_a: np.ndarray | None = None
    endpoint_b: np.ndarray | None = None
    name: str = ""

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Return the analytic signed distance (negative inside)."""
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("points must have shape (N,3)")
        if self.kind == "capsule" and self.endpoint_a is not None:
            a = self.endpoint_a
            b = self.endpoint_b
            ab = b - a
            denominator = float(ab @ ab)
            if denominator <= 1.0e-15:
                closest = np.broadcast_to(a, values.shape)
            else:
                fraction = np.clip(((values - a) @ ab) / denominator, 0.0, 1.0)
                closest = a + fraction[:, None] * ab
            return np.linalg.norm(values - closest, axis=1) - float(self.size[0])

        local = (values - self.center) @ self.rotation
        if self.kind == "box":
            q = np.abs(local) - self.size
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            inside = np.minimum(np.max(q, axis=1), 0.0)
            return outside + inside
        if self.kind == "sphere":
            return np.linalg.norm(local, axis=1) - float(self.size[0])
        if self.kind == "cylinder":
            radial = np.linalg.norm(local[:, :2], axis=1) - float(self.size[0])
            axial = np.abs(local[:, 2]) - float(self.size[1])
            q = np.column_stack((radial, axial))
            return (
                np.minimum(np.maximum(radial, axial), 0.0)
                + np.linalg.norm(np.maximum(q, 0.0), axis=1)
            )
        if self.kind == "capsule":
            half_length = float(self.size[1])
            closest_z = np.clip(local[:, 2], -half_length, half_length)
            closest = np.column_stack((np.zeros(len(local)), np.zeros(len(local)), closest_z))
            return np.linalg.norm(local - closest, axis=1) - float(self.size[0])
        raise ValueError(f"unsupported primitive kind {self.kind!r}")


@dataclass(frozen=True)
class Segment:
    label: str
    start: int
    end: int


def _parse_numbers(value: str | None, expected: int, default: tuple[float, ...]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    result = np.asarray([float(item) for item in value.split()], dtype=np.float64)
    if result.shape != (expected,):
        raise ValueError(f"expected {expected} values, got {value!r}")
    if not np.isfinite(result).all():
        raise ValueError(f"non-finite MJCF values: {value!r}")
    return result


def quaternion_wxyz_matrix(value: np.ndarray) -> np.ndarray:
    """Convert an MJCF wxyz quaternion to a proper rotation matrix."""
    quat = np.asarray(value, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError("quaternion must have four values")
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("quaternion norm must be positive")
    w, x, y, z = quat / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Load and validate the episode label-to-object mapping."""
    mapping_path = Path(path).expanduser().resolve()
    payload = json.loads(mapping_path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("mesh-volume mapping schema_version must be 1")
    labels = payload.get("labels")
    if not isinstance(labels, dict) or not labels:
        raise ValueError("mesh-volume mapping requires a non-empty labels map")
    seen_ids: set[str] = set()
    for label, item in labels.items():
        if not isinstance(label, str) or not isinstance(item, dict):
            raise ValueError("mapping labels must map strings to objects")
        object_id = item.get("object_id")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError(f"{label}: object_id is required")
        spec = item.get("object_spec")
        if not isinstance(spec, str) or not spec:
            raise ValueError(f"{label}: object_spec is required")
        spec_path = (mapping_path.parent / spec).resolve()
        if not spec_path.is_file():
            raise FileNotFoundError(spec_path)
        item["object_spec"] = str(spec_path)
        major = item.get("screen_major_axis")
        minor = item.get("screen_minor_axis")
        if major not in AXIS_INDEX or minor not in AXIS_INDEX or major == minor:
            raise ValueError(f"{label}: screen axes must be distinct x/y/z axes")
        sign = int(item.get("screen_minor_sign", 1))
        if sign not in (-1, 1):
            raise ValueError(f"{label}: screen_minor_sign must be -1 or 1")
        item["screen_minor_sign"] = sign
        seen_ids.add(object_id)
    pitch = float(payload.get("voxel_pitch_m", 0.001))
    if not math.isfinite(pitch) or not 0.00025 <= pitch <= 0.005:
        raise ValueError("voxel_pitch_m must be in [0.00025, 0.005]")
    payload["voxel_pitch_m"] = pitch
    payload["_path"] = str(mapping_path)
    return payload


def _load_object_mjcf(spec_path: str | Path, expected_id: str) -> Path:
    path = Path(spec_path).resolve()
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or payload.get("object_id") != expected_id:
        raise ValueError(f"object spec mismatch: {path} != {expected_id}")
    geometry = payload.get("geometry", {})
    mjcf = geometry.get("mjcf") if isinstance(geometry, dict) else None
    if not isinstance(mjcf, str) or not mjcf:
        raise ValueError(f"{path}: geometry.mjcf is required")
    result = (path.parent / mjcf).resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def parse_mjcf_primitives(path: str | Path) -> tuple[list[Primitive], dict[str, int]]:
    """Parse physical box/cylinder/capsule/sphere geoms from one MJCF.

    Class defaults are resolved before filtering.  A geom with ``contype=0``
    is visual-only and excluded; this removes the label/rim print geoms while
    retaining every physical wall, floor, handle, and body primitive.
    """
    mjcf_path = Path(path).resolve()
    root = ET.parse(mjcf_path).getroot()
    defaults: dict[str, dict[str, str]] = {}
    for node in root.findall("./default/default"):
        class_name = node.attrib.get("class")
        geom = node.find("geom")
        if class_name and geom is not None:
            defaults[class_name] = dict(geom.attrib)

    primitives: list[Primitive] = []
    excluded_visual = 0
    unsupported = 0
    for index, geom in enumerate(root.findall("./worldbody//geom")):
        attributes = dict(defaults.get(geom.attrib.get("class", ""), {}))
        attributes.update(geom.attrib)
        if int(float(attributes.get("contype", "1"))) == 0:
            excluded_visual += 1
            continue
        kind = attributes.get("type", "sphere")
        if kind not in {"box", "cylinder", "capsule", "sphere"}:
            unsupported += 1
            continue
        center = _parse_numbers(attributes.get("pos"), 3, (0.0, 0.0, 0.0))
        rotation = quaternion_wxyz_matrix(
            _parse_numbers(attributes.get("quat"), 4, (1.0, 0.0, 0.0, 0.0))
        )
        raw_size = attributes.get("size")
        if raw_size is None:
            raise ValueError(f"{mjcf_path}: physical geom {index} has no size")
        values = np.asarray([float(item) for item in raw_size.split()], dtype=np.float64)
        expected_sizes = {"box": 3, "cylinder": 2, "capsule": 1, "sphere": 1}
        minimum = expected_sizes[kind]
        if len(values) < minimum or not np.isfinite(values).all() or np.min(values) <= 0:
            raise ValueError(f"{mjcf_path}: invalid {kind} size {raw_size!r}")
        endpoint_a = endpoint_b = None
        if kind == "box":
            size = values[:3]
        elif kind == "cylinder":
            size = values[:2]
        elif kind == "sphere":
            size = values[:1]
        else:
            fromto = attributes.get("fromto")
            if fromto is not None:
                endpoints = _parse_numbers(fromto, 6, (0.0,) * 6)
                endpoint_a, endpoint_b = endpoints[:3], endpoints[3:]
                if np.linalg.norm(endpoint_b - endpoint_a) <= 1.0e-9:
                    raise ValueError(f"{mjcf_path}: zero-length capsule")
                size = values[:1]
            else:
                if len(values) < 2:
                    raise ValueError(f"{mjcf_path}: capsule needs size r h or fromto")
                size = values[:2]
        primitives.append(
            Primitive(
                kind=kind,
                center=center,
                rotation=rotation,
                size=size,
                endpoint_a=endpoint_a,
                endpoint_b=endpoint_b,
                name=attributes.get("name", f"geom_{index}"),
            )
        )
    if unsupported:
        raise ValueError(f"{mjcf_path}: {unsupported} unsupported physical geoms")
    if not primitives:
        raise ValueError(f"{mjcf_path}: no physical primitive geometry")
    return primitives, {
        "physical_primitives": len(primitives),
        "excluded_visual_only_geoms": excluded_visual,
    }


def _align_z_to(vector: np.ndarray) -> np.ndarray:
    # Copy: normalising a view here must not mutate the caller's segment and
    # silently turn a millimetre capsule into a one-metre capsule.
    z_axis = np.array(vector, dtype=np.float64, copy=True)
    z_axis /= np.linalg.norm(z_axis)
    helper = np.asarray((1.0, 0.0, 0.0)) if abs(z_axis[0]) < 0.9 else np.asarray((0.0, 1.0, 0.0))
    y_axis = np.cross(z_axis, helper)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def primitive_mesh(primitive: Primitive) -> trimesh.Trimesh:
    """Create a diagnostic triangle mesh for primitive bounds/QA."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = primitive.rotation
    transform[:3, 3] = primitive.center
    if primitive.kind == "box":
        return trimesh.creation.box(extents=2.0 * primitive.size, transform=transform)
    if primitive.kind == "cylinder":
        return trimesh.creation.cylinder(
            radius=float(primitive.size[0]),
            height=2.0 * float(primitive.size[1]),
            sections=48,
            transform=transform,
        )
    if primitive.kind == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=float(primitive.size[0]))
        mesh.apply_transform(transform)
        return mesh
    if primitive.endpoint_a is not None:
        direction = primitive.endpoint_b - primitive.endpoint_a
        capsule_transform = np.eye(4, dtype=np.float64)
        capsule_transform[:3, :3] = _align_z_to(direction)
        # trimesh centres the medial segment on the local origin even though
        # older docstrings described the first hemisphere as the origin.
        capsule_transform[:3, 3] = 0.5 * (
            primitive.endpoint_a + primitive.endpoint_b
        )
        return trimesh.creation.capsule(
            height=float(np.linalg.norm(direction)),
            radius=float(primitive.size[0]),
            count=(12, 24),
            transform=capsule_transform,
        )
    return trimesh.creation.capsule(
        height=2.0 * float(primitive.size[1]),
        radius=float(primitive.size[0]),
        count=(12, 24),
        transform=transform,
    )


def build_watertight_union_mesh(
    primitives: list[Primitive],
    *,
    voxel_pitch_m: float,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Voxel-union overlapping MJCF solids and extract one closed surface."""
    if not primitives:
        raise ValueError("at least one primitive is required")
    pitch = float(voxel_pitch_m)
    if not math.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("voxel pitch must be finite and positive")
    component_bounds = np.asarray([primitive_mesh(item).bounds for item in primitives])
    pad = 2.5 * pitch
    lower = np.floor((component_bounds[:, 0].min(axis=0) - pad) / pitch) * pitch
    upper = np.ceil((component_bounds[:, 1].max(axis=0) + pad) / pitch) * pitch
    shape = np.floor((upper - lower) / pitch + 0.5).astype(np.int64) + 1
    voxel_count = int(np.prod(shape))
    if voxel_count <= 0 or voxel_count > 20_000_000:
        raise ValueError(f"unsafe canonical voxel grid {tuple(shape)} ({voxel_count})")
    axes = [lower[axis] + pitch * np.arange(shape[axis]) for axis in range(3)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    union_sdf = np.full(len(points), np.inf, dtype=np.float64)
    for primitive in primitives:
        np.minimum(union_sdf, primitive.signed_distance(points), out=union_sdf)
    occupancy = (union_sdf <= 0.0).reshape(tuple(shape))
    if not occupancy.any() or occupancy.all():
        raise RuntimeError("canonical occupancy is empty or touches every voxel")
    vertices, faces, _, _ = marching_objects(
        occupancy.astype(np.float32),
        level=0.5,
        spacing=(pitch, pitch, pitch),
        allow_degenerate=False,
    )
    vertices += lower
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    if not mesh.is_watertight:
        raise RuntimeError("voxel-union canonical mesh is not watertight")
    if not np.isfinite(mesh.vertices).all() or len(mesh.faces) == 0:
        raise RuntimeError("canonical mesh is empty or non-finite")
    return mesh, {
        "voxel_pitch_m": pitch,
        "voxel_grid_shape": [int(value) for value in shape],
        "occupied_voxels": int(occupancy.sum()),
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "bounds_min_m": [float(value) for value in mesh.bounds[0]],
        "bounds_max_m": [float(value) for value in mesh.bounds[1]],
        "extent_m": [float(value) for value in mesh.extents],
        "volume_m3": float(abs(mesh.volume)),
    }


def load_segments(path: str | Path, frame_count: int, known_labels: set[str]) -> list[Segment]:
    payload = json.loads(Path(path).resolve().read_text())
    declared_frames = int(payload.get("num_frames", frame_count))
    if declared_frames != frame_count:
        raise ValueError(f"labels num_frames {declared_frames} != arrays {frame_count}")
    output: list[Segment] = []
    previous_end = -1
    for item in payload.get("segments", []):
        label = str(item["label"])
        start = int(item["start_frame"])
        end = int(item["end_frame"])
        if start < 0 or end >= frame_count or start > end:
            raise ValueError(f"invalid segment {label} {start}:{end}")
        if start <= previous_end:
            raise ValueError("segments must be ordered and non-overlapping")
        previous_end = end
        if label.casefold() == "trans":
            continue
        if label not in known_labels:
            raise ValueError(f"no nominal mesh mapping for label {label!r}")
        output.append(Segment(label, start, end))
    if not output:
        raise ValueError("labels contain no mapped object segments")
    return output


def validate_input_volumes(
    amodal: np.ndarray,
    front_depth: np.ndarray,
    wrist: np.lib.npyio.NpzFile,
) -> tuple[int, int, int, float, np.ndarray, np.ndarray]:
    """Validate aligned arrays and return metadata plus wrist poses."""
    if amodal.ndim != 3 or front_depth.ndim != 3 or amodal.shape != front_depth.shape:
        raise ValueError("amodal mask and completed front depth must share (T,H,W)")
    if amodal.dtype != np.bool_:
        raise TypeError(f"amodal mask must be bool, got {amodal.dtype}")
    frame_count, height, width = amodal.shape
    required = ("wrist_pos", "wrist_rot", "valid", "img_focal")
    missing = [name for name in required if name not in wrist.files]
    if missing:
        raise KeyError(f"wrist npz missing {missing}")
    wrist_pos = np.asarray(wrist["wrist_pos"], dtype=np.float64)
    wrist_rot = np.asarray(wrist["wrist_rot"], dtype=np.float64)
    wrist_valid = np.asarray(wrist["valid"], dtype=bool)
    if wrist_pos.shape != (frame_count, 3) or wrist_rot.shape != (frame_count, 3, 3):
        raise ValueError("wrist poses must align with the input frame count")
    if wrist_valid.shape != (frame_count,):
        raise ValueError("wrist valid must have shape (T,)")
    finite = np.isfinite(wrist_pos).all(axis=1) & np.isfinite(wrist_rot).all(axis=(1, 2))
    wrist_valid &= finite
    focal = float(wrist["img_focal"])
    if not math.isfinite(focal) or focal <= 0.0:
        raise ValueError("wrist img_focal must be finite and positive")
    if "img_width" in wrist.files and int(wrist["img_width"]) != width:
        raise ValueError("wrist img_width differs from input arrays")
    if "img_height" in wrist.files and int(wrist["img_height"]) != height:
        raise ValueError("wrist img_height differs from input arrays")
    determinants = np.linalg.det(wrist_rot[wrist_valid])
    if len(determinants) and np.max(np.abs(determinants - 1.0)) > 1.0e-3:
        raise ValueError("wrist_rot contains non-proper rotations")
    wrist_pose = np.full((frame_count, 4, 4), np.nan, dtype=np.float64)
    wrist_pose[wrist_valid] = np.eye(4, dtype=np.float64)
    wrist_pose[wrist_valid, :3, :3] = wrist_rot[wrist_valid]
    wrist_pose[wrist_valid, :3, 3] = wrist_pos[wrist_valid]
    return frame_count, height, width, focal, wrist_pose, wrist_valid


def mask_pca_axes(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centroid plus deterministic major/minor screen vectors."""
    binary = np.asarray(mask, dtype=bool)
    y, x = np.nonzero(binary)
    if len(x) < 3:
        raise ValueError("mask needs at least three pixels")
    points = np.column_stack((x, y)).astype(np.float64)
    centroid = points.mean(axis=0)
    covariance = np.cov(points - centroid, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    # Eigenvector sign is arbitrary; resolve it without frame history.
    if abs(major[0]) >= abs(major[1]):
        if major[0] < 0.0:
            major *= -1.0
    elif major[1] > 0.0:
        major *= -1.0
    major /= np.linalg.norm(major)
    minor = np.asarray((-major[1], major[0]), dtype=np.float64)
    return centroid, major, minor


def orientation_from_mask(mask: np.ndarray, label_config: dict[str, Any]) -> np.ndarray:
    """Construct a proper object-to-camera orientation from mask PCA."""
    _, major_2d, minor_2d = mask_pca_axes(mask)
    major_axis = AXIS_INDEX[str(label_config["screen_major_axis"])]
    minor_axis = AXIS_INDEX[str(label_config["screen_minor_axis"])]
    depth_axis = ({0, 1, 2} - {major_axis, minor_axis}).pop()
    rotation = np.zeros((3, 3), dtype=np.float64)
    rotation[:, major_axis] = (major_2d[0], major_2d[1], 0.0)
    minor_sign = int(label_config.get("screen_minor_sign", 1))
    rotation[:, minor_axis] = minor_sign * np.asarray((minor_2d[0], minor_2d[1], 0.0))
    rotation[:, depth_axis] = (0.0, 0.0, 1.0)
    # Preserve the two observable screen directions and resolve the unobserved
    # camera-depth sign so the result remains a proper rotation.
    if np.linalg.det(rotation) < 0.0:
        rotation[:, depth_axis] *= -1.0
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise RuntimeError("mask-derived orientation is not orthonormal")
    return rotation


def estimate_frame_pose(
    mask: np.ndarray,
    front_depth: np.ndarray,
    mesh: trimesh.Trimesh,
    label_config: dict[str, Any],
    *,
    focal_px: float,
    principal_point: tuple[float, float],
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Estimate a deterministic raw camera pose from silhouette and front Z."""
    binary = np.asarray(mask, dtype=bool)
    depth = np.asarray(front_depth, dtype=np.float32)
    if binary.shape != depth.shape:
        raise ValueError("frame mask and depth must share one shape")
    mask_pixels = int(binary.sum())
    valid_depth = binary & np.isfinite(depth) & (depth > 0.02) & (depth < 5.0)
    samples = depth[valid_depth]
    if mask_pixels < 3 or len(samples) < 3:
        return np.full((4, 4), np.nan), {
            "valid": False,
            "mask_pixels": mask_pixels,
            "depth_samples": int(len(samples)),
            "median_front_depth_m": math.nan,
            "depth_iqr_m": math.nan,
        }
    centroid, _, _ = mask_pca_axes(binary)
    rotation = orientation_from_mask(binary, label_config)
    rotated_vertices = np.asarray(mesh.vertices, dtype=np.float64) @ rotation.T
    local_centroid = np.asarray(mesh.centroid, dtype=np.float64)
    rotated_centroid = rotation @ local_centroid
    median_depth = float(np.median(samples))
    depth_iqr = float(np.quantile(samples, 0.75) - np.quantile(samples, 0.25))
    # A robust near-surface quantile is more stable than one extreme vertex on
    # thin handles/rims, while still putting the nominal solid behind the
    # completed visible surface.
    front_offset = float(np.quantile(rotated_vertices[:, 2], 0.10))
    translation_z = median_depth - front_offset
    centroid_z = translation_z + rotated_centroid[2]
    cx, cy = principal_point
    target_centroid = np.asarray(
        (
            (centroid[0] - cx) * centroid_z / focal_px,
            (centroid[1] - cy) * centroid_z / focal_px,
            centroid_z,
        ),
        dtype=np.float64,
    )
    translation = target_centroid - rotated_centroid
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    valid = bool(np.isfinite(pose).all() and centroid_z > 0.02)
    if not valid:
        pose[:] = np.nan
    return pose, {
        "valid": valid,
        "mask_pixels": mask_pixels,
        "depth_samples": int(len(samples)),
        "median_front_depth_m": median_depth,
        "depth_iqr_m": depth_iqr,
        "mask_centroid_x": float(centroid[0]),
        "mask_centroid_y": float(centroid[1]),
    }


def invert_pose(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -(value[:3, :3].T @ value[:3, 3])
    return result


def smooth_segment_wrist_relative(
    raw_pose_cam: np.ndarray,
    raw_valid: np.ndarray,
    wrist_pose_cam: np.ndarray,
    wrist_valid: np.ndarray,
    segment: Segment,
    *,
    observation_blend: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Robustly smooth object poses in the XHand wrist coordinate frame."""
    if not 0.0 <= observation_blend <= 1.0:
        raise ValueError("observation_blend must be in [0,1]")
    indices = np.arange(segment.start, segment.end + 1)
    anchors = indices[raw_valid[indices] & wrist_valid[indices]]
    output = np.full_like(raw_pose_cam, np.nan, dtype=np.float64)
    valid = np.zeros(len(raw_pose_cam), dtype=bool)
    if not len(anchors):
        return output, valid, {
            "anchor_frames": 0,
            "translation_mad_m": [None, None, None],
            "rotation_mad_deg": None,
        }
    relative = np.stack(
        [invert_pose(wrist_pose_cam[index]) @ raw_pose_cam[index] for index in anchors]
    )
    center_translation = np.median(relative[:, :3, 3], axis=0)
    center_rotation = Rotation.from_matrix(relative[:, :3, :3]).mean().as_matrix()
    translation_mad = np.median(
        np.abs(relative[:, :3, 3] - center_translation), axis=0
    )
    rotation_delta = Rotation.from_matrix(
        np.einsum("ij,njk->nik", center_rotation.T, relative[:, :3, :3])
    ).magnitude()
    center_relative = np.eye(4, dtype=np.float64)
    center_relative[:3, :3] = center_rotation
    center_relative[:3, 3] = center_translation

    for index in indices:
        if not wrist_valid[index]:
            continue
        relative_pose = center_relative.copy()
        if raw_valid[index]:
            raw_relative = invert_pose(wrist_pose_cam[index]) @ raw_pose_cam[index]
            relative_pose[:3, 3] += observation_blend * (
                raw_relative[:3, 3] - center_translation
            )
            delta = center_rotation.T @ raw_relative[:3, :3]
            delta_vector = Rotation.from_matrix(delta).as_rotvec()
            relative_pose[:3, :3] = center_rotation @ Rotation.from_rotvec(
                observation_blend * delta_vector
            ).as_matrix()
        output[index] = wrist_pose_cam[index] @ relative_pose
        valid[index] = np.isfinite(output[index]).all()
    return output, valid, {
        "anchor_frames": int(len(anchors)),
        "translation_mad_m": [float(value) for value in translation_mad],
        "rotation_mad_deg": float(np.degrees(np.median(rotation_delta))),
        "relative_translation_m": [float(value) for value in center_translation],
        "relative_rotation_matrix": center_rotation.tolist(),
    }


def cv_pose_to_pyrender(pose_cam: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_cam, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("camera pose must have shape (4,4)")
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = T_CV2GL @ pose[:3, :3]
    output[:3, 3] = T_CV2GL @ pose[:3, 3]
    return output


class FrontBackRenderer:
    """One EGL renderer with normal- and reversed-winding depth scenes."""

    def __init__(self, width: int, height: int, focal_px: float) -> None:
        if width <= 0 or height <= 0 or focal_px <= 0:
            raise ValueError("renderer dimensions and focal must be positive")
        self.width = int(width)
        self.height = int(height)
        self.focal_px = float(focal_px)
        self.renderer = pyrender.OffscreenRenderer(self.width, self.height)
        self.front_scene: pyrender.Scene | None = None
        self.back_scene: pyrender.Scene | None = None
        self.front_node: pyrender.Node | None = None
        self.back_node: pyrender.Node | None = None
        self.object_id: str | None = None

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.delete()
            self.renderer = None

    def _scene(self, mesh: trimesh.Trimesh) -> tuple[pyrender.Scene, pyrender.Node]:
        scene = pyrender.Scene(bg_color=(0.0, 0.0, 0.0, 0.0))
        scene.add(
            pyrender.IntrinsicsCamera(
                fx=self.focal_px,
                fy=self.focal_px,
                cx=self.width / 2.0,
                cy=self.height / 2.0,
                znear=0.01,
                zfar=5.0,
            ),
            pose=np.eye(4),
        )
        node = scene.add(
            pyrender.Mesh.from_trimesh(mesh, smooth=False),
            pose=OFFSCREEN_POSE,
        )
        return scene, node

    def set_mesh(self, object_id: str, mesh: trimesh.Trimesh) -> None:
        if self.object_id == object_id:
            return
        front = mesh.copy()
        back = mesh.copy()
        back.faces = np.asarray(back.faces)[:, ::-1]
        self.front_scene, self.front_node = self._scene(front)
        self.back_scene, self.back_node = self._scene(back)
        self.object_id = str(object_id)

    def render(self, pose_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if self.front_scene is None or self.back_scene is None:
            raise RuntimeError("set_mesh must be called before render")
        pose = cv_pose_to_pyrender(pose_cam)
        self.front_scene.set_pose(self.front_node, pose)
        self.back_scene.set_pose(self.back_node, pose)
        front = np.asarray(
            self.renderer.render(self.front_scene, flags=pyrender.RenderFlags.DEPTH_ONLY),
            dtype=np.float32,
        )
        back = np.asarray(
            self.renderer.render(self.back_scene, flags=pyrender.RenderFlags.DEPTH_ONLY),
            dtype=np.float32,
        )
        front_valid = np.isfinite(front) & (front > 0.02) & (front < 5.0)
        back_valid = np.isfinite(back) & (back > 0.02) & (back < 5.0)
        paired = front_valid & back_valid
        ordered = paired & (back + np.float32(5.0e-4) >= front)
        denominator = int(front_valid.sum())
        order_fraction = float(ordered.sum()) / max(denominator, 1)
        front = np.where(ordered, front, 0.0).astype(np.float32)
        back = np.where(ordered, np.maximum(back, front), 0.0).astype(np.float32)
        return front, back, ordered, order_fraction


def resize_depth_and_mask(
    front: np.ndarray,
    back: np.ndarray,
    mask: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if front.shape == (height, width):
        return front.copy(), back.copy(), mask.copy()
    output_mask = cv2.resize(
        mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    output_front = cv2.resize(front, (width, height), interpolation=cv2.INTER_NEAREST)
    output_back = cv2.resize(back, (width, height), interpolation=cv2.INTER_NEAREST)
    output_front[~output_mask] = 0.0
    output_back[~output_mask] = 0.0
    return output_front, output_back, output_mask


def _mask_centroid(mask: np.ndarray) -> np.ndarray | None:
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    return np.asarray((x.mean(), y.mean()), dtype=np.float64)


def fit_metrics(
    observed_mask: np.ndarray,
    observed_depth: np.ndarray,
    mesh_mask: np.ndarray,
    mesh_front_depth: np.ndarray,
    *,
    order_fraction: float,
) -> dict[str, float | int]:
    observed = np.asarray(observed_mask, dtype=bool)
    rendered = np.asarray(mesh_mask, dtype=bool)
    intersection = observed & rendered
    union = observed | rendered
    observed_pixels = int(observed.sum())
    mesh_pixels = int(rendered.sum())
    intersection_pixels = int(intersection.sum())
    iou = float(intersection_pixels) / max(int(union.sum()), 1)
    mesh_coverage = float(intersection_pixels) / max(mesh_pixels, 1)
    observation_coverage = float(intersection_pixels) / max(observed_pixels, 1)
    valid_depth = (
        intersection
        & np.isfinite(observed_depth)
        & (observed_depth > 0.02)
        & (observed_depth < 5.0)
        & np.isfinite(mesh_front_depth)
        & (mesh_front_depth > 0.02)
    )
    depth_samples = int(valid_depth.sum())
    depth_error = (
        float(np.median(np.abs(mesh_front_depth[valid_depth] - observed_depth[valid_depth])))
        if depth_samples else math.nan
    )
    observed_centroid = _mask_centroid(observed)
    mesh_centroid = _mask_centroid(rendered)
    centroid_error = (
        float(np.linalg.norm(observed_centroid - mesh_centroid))
        if observed_centroid is not None and mesh_centroid is not None
        else math.inf
    )
    return {
        "observed_pixels": observed_pixels,
        "mesh_pixels": mesh_pixels,
        "intersection_pixels": intersection_pixels,
        "iou": iou,
        "mesh_coverage": mesh_coverage,
        "observation_coverage": observation_coverage,
        "depth_samples": depth_samples,
        "median_front_depth_error_m": depth_error,
        "front_back_order_fraction": float(order_fraction),
        "centroid_error_px": centroid_error,
        "observation_over_mesh_area_ratio": float(observed_pixels) / max(mesh_pixels, 1),
    }


def pose_confidence(metrics: dict[str, float | int], fit_config: dict[str, Any]) -> float:
    """Combine complementary fit cues without letting contaminated IoU veto."""
    good_error = float(fit_config["good_depth_error_m"])
    bad_error = float(fit_config["bad_depth_error_m"])
    error = float(metrics["median_front_depth_error_m"])
    if not math.isfinite(error):
        depth_score = 0.0
    else:
        depth_score = float(np.clip((bad_error - error) / (bad_error - good_error), 0.0, 1.0))
    iou_score = float(np.clip(float(metrics["iou"]) / float(fit_config["good_iou"]), 0.0, 1.0))
    mesh_coverage = float(np.clip(float(metrics["mesh_coverage"]), 0.0, 1.0))
    order_score = float(np.clip(float(metrics["front_back_order_fraction"]), 0.0, 1.0))
    centroid_error = float(metrics["centroid_error_px"])
    centroid_score = (
        float(np.clip(1.0 - centroid_error / float(fit_config["maximum_centroid_error_px"]), 0.0, 1.0))
        if math.isfinite(centroid_error) else 0.0
    )
    # Mesh coverage dominates IoU when a SAM/amodal mask contains hand skin;
    # depth and ordered front/back evidence independently prevent a tiny mesh
    # wholly inside a contaminated mask from receiving high confidence.
    return float(
        0.30 * mesh_coverage
        + 0.15 * iou_score
        + 0.30 * depth_score
        + 0.15 * order_score
        + 0.10 * centroid_score
    )


def _debug_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    front: np.ndarray,
    back: np.ndarray,
    *,
    frame_index: int,
    label: str,
    confidence: float,
    valid: bool,
    iou: float,
    depth_error_m: float,
) -> np.ndarray:
    output = np.asarray(frame, dtype=np.uint8).copy()
    binary = np.asarray(mask, dtype=bool)
    if binary.any():
        thickness = np.zeros(front.shape, dtype=np.float32)
        thickness[binary] = np.maximum(back[binary] - front[binary], 0.0)
        normalized = np.clip(thickness / 0.12, 0.0, 1.0)
        colour = np.zeros((*front.shape, 3), dtype=np.float32)
        colour[..., 0] = 255.0 * normalized
        colour[..., 1] = 220.0 * (1.0 - normalized)
        colour[..., 2] = 255.0
        output[binary] = np.clip(
            0.45 * output[binary].astype(np.float32) + 0.55 * colour[binary],
            0,
            255,
        ).astype(np.uint8)
        outline = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        output[outline] = (40, 255, 255) if valid else (40, 40, 255)
    depth_text = f"{depth_error_m * 1000:.0f}mm" if math.isfinite(depth_error_m) else "n/a"
    text_value = (
        f"{frame_index:04d} {label or 'Trans'}  conf={confidence:.2f} "
        f"IoU={iou:.2f} dZ={depth_text} {'VALID' if valid else 'FAIL-OPEN'}"
    )
    cv2.putText(output, text_value, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(
        output,
        text_value,
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255) if valid else (80, 180, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--labels_json", type=Path, required=True)
    parser.add_argument("--amodal_mask", type=Path, required=True)
    parser.add_argument("--front_depth", type=Path, required=True)
    parser.add_argument("--wrist_npz", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument(
        "--render_scale",
        type=float,
        default=0.5,
        help="EGL raster size relative to output arrays; depth is nearest-upsampled",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=None,
        help="override mapping fit.minimum_pose_confidence",
    )
    args = parser.parse_args()

    if not math.isfinite(args.render_scale) or not 0.1 <= args.render_scale <= 1.0:
        raise ValueError("render_scale must be in [0.1,1.0]")
    mapping_path = args.mapping.resolve()
    labels_path = args.labels_json.resolve()
    amodal_path = args.amodal_mask.resolve()
    front_path = args.front_depth.resolve()
    wrist_path = args.wrist_npz.resolve()
    out_dir = args.out_dir.resolve()
    video_path = args.video.resolve() if args.video is not None else None
    for path in (mapping_path, labels_path, amodal_path, front_path, wrist_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if video_path is not None and not video_path.is_file():
        raise FileNotFoundError(video_path)

    mapping = load_mapping(mapping_path)
    fit_config = dict(mapping.get("fit", {}))
    required_fit = {
        "minimum_mask_pixels": 96,
        "minimum_depth_samples": 48,
        "minimum_segment_area_ratio": 0.35,
        "maximum_segment_area_ratio": 2.2,
        "maximum_depth_iqr_m": 0.35,
        "relative_observation_blend": 0.2,
        "minimum_pose_confidence": 0.4,
        "good_depth_error_m": 0.04,
        "bad_depth_error_m": 0.18,
        "good_iou": 0.45,
        "maximum_centroid_error_px": 120.0,
        "minimum_front_back_order_fraction": 0.65,
    }
    for name, default in required_fit.items():
        fit_config.setdefault(name, default)
    minimum_confidence = (
        float(args.min_confidence)
        if args.min_confidence is not None
        else float(fit_config["minimum_pose_confidence"])
    )
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum pose confidence must be in [0,1]")
    fit_config["minimum_pose_confidence"] = minimum_confidence
    if float(fit_config["bad_depth_error_m"]) <= float(fit_config["good_depth_error_m"]):
        raise ValueError("bad_depth_error_m must exceed good_depth_error_m")

    amodal = np.load(amodal_path, mmap_mode="r", allow_pickle=False)
    completed_front = np.load(front_path, mmap_mode="r", allow_pickle=False)
    with np.load(wrist_path, allow_pickle=False) as wrist:
        (
            frame_count,
            height,
            width,
            focal_px,
            wrist_pose_cam,
            wrist_valid,
        ) = validate_input_volumes(amodal, completed_front, wrist)
    segments = load_segments(labels_path, frame_count, set(mapping["labels"]))
    principal_point = (width / 2.0, height / 2.0)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".object_mesh_volume.", dir=out_dir.parent))
    atexit.register(shutil.rmtree, staging, ignore_errors=True)
    canonical_dir = staging / "canonical_meshes"
    canonical_dir.mkdir(parents=True)

    label_meshes: dict[str, trimesh.Trimesh] = {}
    canonical_report: dict[str, dict[str, Any]] = {}
    object_id_to_mesh: dict[str, trimesh.Trimesh] = {}
    for label, label_config in mapping["labels"].items():
        object_id = str(label_config["object_id"])
        mjcf_path = _load_object_mjcf(label_config["object_spec"], object_id)
        if object_id not in object_id_to_mesh:
            primitives, parse_stats = parse_mjcf_primitives(mjcf_path)
            mesh, mesh_stats = build_watertight_union_mesh(
                primitives,
                voxel_pitch_m=float(mapping["voxel_pitch_m"]),
            )
            mesh_path = canonical_dir / f"{object_id}.ply"
            mesh.export(mesh_path)
            object_id_to_mesh[object_id] = mesh
            canonical_report[object_id] = {
                "object_spec": str(Path(label_config["object_spec"]).resolve()),
                "mjcf": str(mjcf_path),
                **parse_stats,
                **mesh_stats,
                "mesh_file": str(Path("canonical_meshes") / mesh_path.name),
                "mesh_sha256": _sha256(mesh_path),
            }
        label_meshes[label] = object_id_to_mesh[object_id]

    # Build deterministic raw observations first.  Area outliers are not
    # anchors, but later receive the segment's wrist-relative fitted pose.
    raw_pose = np.full((frame_count, 4, 4), np.nan, dtype=np.float64)
    raw_valid = np.zeros(frame_count, dtype=bool)
    raw_mask_pixels = np.zeros(frame_count, dtype=np.int64)
    raw_depth_samples = np.zeros(frame_count, dtype=np.int64)
    raw_median_depth = np.full(frame_count, np.nan, dtype=np.float32)
    raw_depth_iqr = np.full(frame_count, np.nan, dtype=np.float32)
    segment_area_ratio = np.full(frame_count, np.nan, dtype=np.float32)
    frame_label_index = np.full(frame_count, -1, dtype=np.int16)
    label_names = list(mapping["labels"])
    segment_reports: list[dict[str, Any]] = []

    for segment_index, segment in enumerate(segments):
        label_config = mapping["labels"][segment.label]
        mesh = label_meshes[segment.label]
        indices = np.arange(segment.start, segment.end + 1)
        areas = np.asarray(
            [int(np.asarray(amodal[index], dtype=bool).sum()) for index in indices],
            dtype=np.float64,
        )
        positive = areas[areas > 0]
        median_area = float(np.median(positive)) if len(positive) else 0.0
        for index, area in zip(indices, areas):
            frame_label_index[index] = label_names.index(segment.label)
            ratio = float(area / median_area) if median_area > 0.0 else math.nan
            segment_area_ratio[index] = ratio
            pose, evidence = estimate_frame_pose(
                np.asarray(amodal[index]),
                np.asarray(completed_front[index]),
                mesh,
                label_config,
                focal_px=focal_px,
                principal_point=principal_point,
            )
            raw_mask_pixels[index] = int(evidence["mask_pixels"])
            raw_depth_samples[index] = int(evidence["depth_samples"])
            raw_median_depth[index] = float(evidence["median_front_depth_m"])
            raw_depth_iqr[index] = float(evidence["depth_iqr_m"])
            reliable = (
                bool(evidence["valid"])
                and wrist_valid[index]
                and int(evidence["mask_pixels"]) >= int(fit_config["minimum_mask_pixels"])
                and int(evidence["depth_samples"]) >= int(fit_config["minimum_depth_samples"])
                and math.isfinite(ratio)
                and float(fit_config["minimum_segment_area_ratio"])
                <= ratio
                <= float(fit_config["maximum_segment_area_ratio"])
                and float(evidence["depth_iqr_m"]) <= float(fit_config["maximum_depth_iqr_m"])
            )
            if reliable:
                raw_pose[index] = pose
                raw_valid[index] = True

        smoothed, candidate_valid, smoothing = smooth_segment_wrist_relative(
            raw_pose,
            raw_valid,
            wrist_pose_cam,
            wrist_valid,
            segment,
            observation_blend=float(fit_config["relative_observation_blend"]),
        )
        if segment_index == 0:
            fitted_pose_candidate = np.full_like(raw_pose, np.nan)
            candidate_pose_valid = np.zeros(frame_count, dtype=bool)
        fitted_pose_candidate[indices] = smoothed[indices]
        candidate_pose_valid[indices] = candidate_valid[indices]
        segment_reports.append(
            {
                "label": segment.label,
                "object_id": label_config["object_id"],
                "start": segment.start,
                "end": segment.end,
                "median_amodal_area_px": int(round(median_area)),
                "reliable_raw_pose_frames": int(raw_valid[indices].sum()),
                "candidate_pose_frames": int(candidate_pose_valid[indices].sum()),
                "wrist_relative_smoothing": smoothing,
            }
        )

    front_out = np.lib.format.open_memmap(
        staging / "object_mesh_front_depth.npy",
        mode="w+",
        dtype=np.float16,
        shape=(frame_count, height, width),
    )
    back_out = np.lib.format.open_memmap(
        staging / "object_mesh_back_depth.npy",
        mode="w+",
        dtype=np.float16,
        shape=(frame_count, height, width),
    )
    mask_out = np.lib.format.open_memmap(
        staging / "object_mesh_mask.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(frame_count, height, width),
    )
    front_out[:] = 0.0
    back_out[:] = 0.0
    mask_out[:] = False
    final_pose = np.full((frame_count, 4, 4), np.nan, dtype=np.float32)
    pose_valid = np.zeros(frame_count, dtype=bool)
    confidence = np.zeros(frame_count, dtype=np.float32)
    fit_iou = np.zeros(frame_count, dtype=np.float32)
    mesh_coverage = np.zeros(frame_count, dtype=np.float32)
    observation_coverage = np.zeros(frame_count, dtype=np.float32)
    depth_error = np.full(frame_count, np.nan, dtype=np.float32)
    fit_depth_samples = np.zeros(frame_count, dtype=np.int64)
    order_fraction = np.zeros(frame_count, dtype=np.float32)
    centroid_error = np.full(frame_count, np.inf, dtype=np.float32)
    mesh_pixels = np.zeros(frame_count, dtype=np.int64)
    observed_pixels = np.zeros(frame_count, dtype=np.int64)
    area_overcoverage = np.full(frame_count, np.nan, dtype=np.float32)

    render_width = max(32, int(round(width * args.render_scale)))
    render_height = max(32, int(round(height * args.render_scale)))
    render_focal = focal_px * render_width / width
    renderer = FrontBackRenderer(render_width, render_height, render_focal)
    capture = None
    writer = None
    if video_path is not None:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            renderer.close()
            raise FileNotFoundError(video_path)
        video_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if video_frames != frame_count:
            capture.release()
            renderer.close()
            raise ValueError(f"debug video frames {video_frames} != arrays {frame_count}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
        writer = cv2.VideoWriter(
            str(staging / "debug_object_mesh_volume.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            renderer.close()
            raise RuntimeError("could not open mesh-volume debug writer")

    current_label = ""
    try:
        for frame_index in range(frame_count):
            debug_source = None
            if capture is not None:
                ok, debug_source = capture.read()
                if not ok:
                    raise RuntimeError(f"debug video read failed at {frame_index}")
                if debug_source.shape[:2] != (height, width):
                    debug_source = cv2.resize(debug_source, (width, height), interpolation=cv2.INTER_AREA)

            label_index = int(frame_label_index[frame_index])
            local_front = np.zeros((height, width), dtype=np.float32)
            local_back = np.zeros((height, width), dtype=np.float32)
            local_mask = np.zeros((height, width), dtype=bool)
            metrics: dict[str, float | int] = {
                "observed_pixels": int(np.asarray(amodal[frame_index]).sum()),
                "mesh_pixels": 0,
                "iou": 0.0,
                "mesh_coverage": 0.0,
                "observation_coverage": 0.0,
                "depth_samples": 0,
                "median_front_depth_error_m": math.nan,
                "front_back_order_fraction": 0.0,
                "centroid_error_px": math.inf,
                "observation_over_mesh_area_ratio": math.nan,
            }
            label = "" if label_index < 0 else label_names[label_index]
            if label and candidate_pose_valid[frame_index]:
                if current_label != label:
                    renderer.set_mesh(
                        str(mapping["labels"][label]["object_id"]),
                        label_meshes[label],
                    )
                    current_label = label
                small_front, small_back, small_mask, ordered_fraction = renderer.render(
                    fitted_pose_candidate[frame_index]
                )
                local_front, local_back, local_mask = resize_depth_and_mask(
                    small_front,
                    small_back,
                    small_mask,
                    width=width,
                    height=height,
                )
                metrics = fit_metrics(
                    np.asarray(amodal[frame_index]),
                    np.asarray(completed_front[frame_index]),
                    local_mask,
                    local_front,
                    order_fraction=ordered_fraction,
                )
                confidence[frame_index] = pose_confidence(metrics, fit_config)
                hard_valid = (
                    confidence[frame_index] >= minimum_confidence
                    and int(metrics["mesh_pixels"]) >= int(fit_config["minimum_mask_pixels"])
                    and int(metrics["depth_samples"]) >= int(fit_config["minimum_depth_samples"])
                    and float(metrics["front_back_order_fraction"])
                    >= float(fit_config["minimum_front_back_order_fraction"])
                )
                if hard_valid:
                    # Cast after enforcing order.  Equal float16 values are
                    # allowed; an inverted pair is never published.
                    front16 = local_front.astype(np.float16)
                    back16 = np.maximum(local_back, local_front).astype(np.float16)
                    cast_valid = local_mask & (front16 > 0.02) & (back16 >= front16) & (back16 < 5.0)
                    front16[~cast_valid] = 0.0
                    back16[~cast_valid] = 0.0
                    if int(cast_valid.sum()) >= int(fit_config["minimum_mask_pixels"]):
                        front_out[frame_index] = front16
                        back_out[frame_index] = back16
                        mask_out[frame_index] = cast_valid
                        final_pose[frame_index] = fitted_pose_candidate[frame_index].astype(np.float32)
                        pose_valid[frame_index] = True

            fit_iou[frame_index] = float(metrics["iou"])
            mesh_coverage[frame_index] = float(metrics["mesh_coverage"])
            observation_coverage[frame_index] = float(metrics["observation_coverage"])
            depth_error[frame_index] = float(metrics["median_front_depth_error_m"])
            fit_depth_samples[frame_index] = int(metrics["depth_samples"])
            order_fraction[frame_index] = float(metrics["front_back_order_fraction"])
            centroid_error[frame_index] = float(metrics["centroid_error_px"])
            mesh_pixels[frame_index] = int(metrics["mesh_pixels"])
            observed_pixels[frame_index] = int(metrics["observed_pixels"])
            area_overcoverage[frame_index] = float(metrics["observation_over_mesh_area_ratio"])

            if writer is not None and debug_source is not None:
                writer.write(
                    _debug_frame(
                        debug_source,
                        local_mask,
                        local_front,
                        local_back,
                        frame_index=frame_index,
                        label=label,
                        confidence=float(confidence[frame_index]),
                        valid=bool(pose_valid[frame_index]),
                        iou=float(fit_iou[frame_index]),
                        depth_error_m=float(depth_error[frame_index]),
                    )
                )
            if (frame_index + 1) % 100 == 0:
                print(
                    f"[mesh-volume] {frame_index + 1}/{frame_count} "
                    f"valid={int(pose_valid[:frame_index + 1].sum())}",
                    flush=True,
                )
    finally:
        front_out.flush()
        back_out.flush()
        mask_out.flush()
        renderer.close()
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()

    np.save(staging / "object_pose_cam.npy", final_pose)
    np.save(staging / "pose_valid.npy", pose_valid)
    np.save(staging / "pose_confidence.npy", confidence)
    np.savez(
        staging / "fit_evidence.npz",
        frame_label_index=frame_label_index,
        raw_pose_cam=raw_pose.astype(np.float32),
        raw_pose_valid=raw_valid,
        fitted_pose_candidate_cam=fitted_pose_candidate.astype(np.float32),
        candidate_pose_valid=candidate_pose_valid,
        pose_valid=pose_valid,
        pose_confidence=confidence,
        raw_mask_pixels=raw_mask_pixels,
        raw_depth_samples=raw_depth_samples,
        raw_median_front_depth_m=raw_median_depth,
        raw_depth_iqr_m=raw_depth_iqr,
        segment_area_ratio=segment_area_ratio,
        fit_iou=fit_iou,
        mesh_coverage=mesh_coverage,
        observation_coverage=observation_coverage,
        median_front_depth_error_m=depth_error,
        fit_depth_samples=fit_depth_samples,
        front_back_order_fraction=order_fraction,
        centroid_error_px=centroid_error,
        mesh_pixels=mesh_pixels,
        observed_pixels=observed_pixels,
        observation_over_mesh_area_ratio=area_overcoverage,
        label_names=np.asarray(label_names),
    )

    valid_depth_errors = depth_error[pose_valid & np.isfinite(depth_error)]
    valid_ious = fit_iou[pose_valid]
    valid_confidence = confidence[pose_valid]
    total_mask_pixels = int(np.asarray(mask_out).sum())
    for report_segment, segment in zip(segment_reports, segments):
        indices = np.arange(segment.start, segment.end + 1)
        selected = pose_valid[indices]
        report_segment.update(
            {
                "valid_pose_frames": int(selected.sum()),
                "median_pose_confidence": (
                    float(np.median(confidence[indices][selected])) if selected.any() else None
                ),
                "median_fit_iou": (
                    float(np.median(fit_iou[indices][selected])) if selected.any() else None
                ),
                "median_front_depth_error_m": (
                    float(np.median(depth_error[indices][selected])) if selected.any() else None
                ),
            }
        )

    report = {
        "schema_version": 1,
        "method": METHOD,
        "representation": REPRESENTATION,
        "frames": int(frame_count),
        "width": int(width),
        "height": int(height),
        "coordinate_frame": "MH HaWoR/XHand overlay camera (OpenCV +Z forward)",
        "depth_unit": "metre camera-Z",
        "invalid_depth_value": 0.0,
        "pose_state_modified": False,
        "metric_collision_guarantee": False,
        "rear_surface_measured": False,
        "sources": {
            "mapping": str(mapping_path),
            "labels_json": str(labels_path),
            "amodal_mask": str(amodal_path),
            "completed_front_depth": str(front_path),
            "wrist_npz": str(wrist_path),
            "debug_video": str(video_path) if video_path is not None else None,
        },
        "camera": {
            "focal_px": float(focal_px),
            "principal_point_assumption": [float(principal_point[0]), float(principal_point[1])],
            "calibrated_phone_intrinsics": False,
            "render_scale": float(args.render_scale),
            "render_width": int(render_width),
            "render_height": int(render_height),
        },
        "fit": {
            **fit_config,
            "framewise_initialization": "amodal mask PCA + completed front-Z",
            "temporal_model": "segmentwise robust wrist-relative pose with bounded observation blend",
            "front_pass": "normal mesh winding",
            "back_pass": "reversed mesh winding; nearest paired material exit",
            "confidence_uses": [
                "mesh coverage",
                "IoU (non-vetoing under mask overcoverage)",
                "front-depth residual",
                "front/back order",
                "centroid residual",
            ],
        },
        "label_mapping": {
            label: {
                "object_id": item["object_id"],
                "object_spec": item["object_spec"],
            }
            for label, item in mapping["labels"].items()
        },
        "canonical_meshes": canonical_report,
        "segments": segment_reports,
        "counts": {
            "candidate_pose_frames": int(candidate_pose_valid.sum()),
            "valid_pose_frames": int(pose_valid.sum()),
            "fail_open_candidate_frames": int((candidate_pose_valid & ~pose_valid).sum()),
            "mesh_pixels": total_mask_pixels,
            "transition_mesh_pixels": int(np.asarray(mask_out)[frame_label_index < 0].sum()),
        },
        "summary": {
            "median_valid_confidence": float(np.median(valid_confidence)) if len(valid_confidence) else None,
            "median_valid_iou": float(np.median(valid_ious)) if len(valid_ious) else None,
            "median_valid_front_depth_error_m": (
                float(np.median(valid_depth_errors)) if len(valid_depth_errors) else None
            ),
        },
        "invariants": {
            "canonical_meshes_watertight": all(
                item["watertight"] for item in canonical_report.values()
            ),
            "wrist_relative_segment_smoothing_used": True,
            "invalid_pose_frames_have_empty_geometry": True,
            "transition_frames_invalid": not bool(pose_valid[frame_label_index < 0].any()),
            "valid_mesh_pixels_have_ordered_front_back": True,
            "mesh_mask_equals_positive_front_and_back": True,
            "auxiliary_camera_geometry_used": False,
            "robot_trajectory_arrays_unchanged": True,
        },
        "outputs": {
            "canonical_mesh_dir": "canonical_meshes",
            "pose": "object_pose_cam.npy",
            "pose_valid": "pose_valid.npy",
            "pose_confidence": "pose_confidence.npy",
            "front_depth": "object_mesh_front_depth.npy",
            "back_depth": "object_mesh_back_depth.npy",
            "mesh_mask": "object_mesh_mask.npy",
            "fit_evidence": "fit_evidence.npz",
            "debug_video": "debug_object_mesh_volume.mp4" if video_path is not None else None,
        },
        "provenance_warning": (
            "The closed meshes come from nominal MJCF dimensions. Their hidden/rear surfaces "
            "and fitted poses are model-derived, not measured by HaCo or calibrated stereo. "
            "The MH focal and monocular front depth are overlay-coordinate approximations, so "
            "this product supports visual comparison only and is not a metric collision guarantee."
        ),
    }
    (staging / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    del front_out, back_out, mask_out
    publish_directory(staging, out_dir)
    print(
        f"[ok] {out_dir}: valid={int(pose_valid.sum())}/{frame_count}, "
        f"mesh_pixels={total_mask_pixels}",
        flush=True,
    )


if __name__ == "__main__":
    main()
