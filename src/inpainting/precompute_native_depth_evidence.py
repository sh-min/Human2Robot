"""Precompute per-finger hand/object metric depths in one native RGB view.

The inputs to this utility must already share one camera's RGB pixel grid.
It deliberately performs no cross-camera projection and no resizing.  For
each frame and finger it rasterizes a local support around that camera's
HaWoR 2-D joint polyline, samples metric depth from the visible hand and
object masks, and stores robust trimmed/MAD medians plus sample counts.

``hand_data_{side}.npz`` keypoints are preferred because they are already in
the native RGB coordinates.  Camera-space MANO vertices are projected only as
a fallback for frames without usable native keypoints.

The visible hand is defined conservatively as::

    model_masks & masks_arm

Object samples use the refined mask when supplied (intersected with the modal
mask) and always exclude the visible-hand pixels.  Insufficient support stays
NaN so downstream depth ordering fails open.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_JOINTS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
JOINT_SCOPE_CHOICES = ("distal", "distal2", "full")
FINGER_PARTS = {
    "index": (1, 2, 3),
    "middle": (4, 5, 6),
    "pinky": (7, 8, 9),
    "ring": (10, 11, 12),
    "thumb": (13, 14, 15),
}

SUPPORT_NONE = np.uint8(0)
SUPPORT_NATIVE_KEYPOINTS = np.uint8(1)
SUPPORT_PROJECTED_VERTICES = np.uint8(2)
SUPPORT_SOURCE_LABELS = ("none", "native_kpts_2d", "projected_mano_vertices")


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def validate(self) -> None:
        numeric = np.asarray((self.fx, self.fy, self.cx, self.cy), dtype=float)
        if not np.isfinite(numeric).all() or self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera intrinsics must be finite with fx/fy > 0")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")


@dataclass(frozen=True)
class DepthEvidenceConfig:
    support_radius_px: int = 9
    # ``distal`` keeps the last three joints (PIP/IP through fingertip), which
    # targets the part of the finger that can actually pass behind a grasped
    # object.  ``full`` is retained only for diagnostics/backward comparison.
    joint_scope: str = "distal"
    min_projected_vertices: int = 6
    # Match the compositor's default count gate.  The robust inlier gate below
    # independently enforces these values before a finite depth is emitted.
    min_hand_samples: int = 6
    min_object_samples: int = 20
    trim_fraction: float = 0.10
    mad_scale: float = 3.5
    min_depth_m: float = 0.02
    max_depth_m: float = 5.0

    def validate(self) -> None:
        if self.support_radius_px < 0:
            raise ValueError("support_radius_px must be non-negative")
        if self.joint_scope not in JOINT_SCOPE_CHOICES:
            raise ValueError(
                f"joint_scope must be one of {JOINT_SCOPE_CHOICES}, "
                f"got {self.joint_scope!r}"
            )
        if self.min_projected_vertices <= 0:
            raise ValueError("min_projected_vertices must be positive")
        if self.min_hand_samples <= 0 or self.min_object_samples <= 0:
            raise ValueError("minimum sample counts must be positive")
        if not 0.0 <= self.trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5)")
        if not np.isfinite(self.mad_scale) or self.mad_scale <= 0.0:
            raise ValueError("mad_scale must be finite and positive")
        if (
            not np.isfinite(self.min_depth_m)
            or not np.isfinite(self.max_depth_m)
            or not 0.0 < self.min_depth_m < self.max_depth_m
        ):
            raise ValueError("expected 0 < min_depth_m < max_depth_m")


def robust_metric_depth(
    values: np.ndarray,
    *,
    min_samples: int,
    trim_fraction: float,
    mad_scale: float,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[float, int, int]:
    """Return ``(median_m, valid_count, robust_inlier_count)``.

    Invalid/out-of-range sensor samples are discarded first.  Symmetric
    quantile trimming removes long tails, followed by an optional MAD gate.
    The minimum count is enforced both before and after robust filtering;
    failure returns NaN instead of borrowing evidence from another frame.
    """

    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    samples = samples[
        np.isfinite(samples)
        & (samples >= float(min_depth_m))
        & (samples <= float(max_depth_m))
    ]
    valid_count = int(len(samples))
    if valid_count < min_samples:
        return float("nan"), valid_count, 0

    inliers = samples
    if trim_fraction > 0.0:
        low, high = np.quantile(
            inliers,
            (float(trim_fraction), 1.0 - float(trim_fraction)),
        )
        inliers = inliers[(inliers >= low) & (inliers <= high)]
    if len(inliers) < min_samples:
        return float("nan"), valid_count, int(len(inliers))

    center = float(np.median(inliers))
    mad = float(np.median(np.abs(inliers - center)))
    if np.isfinite(mad) and mad > np.finfo(np.float64).eps:
        robust_sigma = 1.4826 * mad
        inliers = inliers[
            np.abs(inliers - center) <= float(mad_scale) * robust_sigma
        ]
    inlier_count = int(len(inliers))
    if inlier_count < min_samples:
        return float("nan"), valid_count, inlier_count
    return float(np.median(inliers)), valid_count, inlier_count


def project_camera_points(
    points_camera: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Project CV-camera XYZ points using native RGB intrinsics."""

    intrinsics.validate()
    points = np.asarray(points_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_camera must have shape (N,3), got {points.shape}")
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1.0e-5)
    uv = np.full((len(points), 2), np.nan, dtype=np.float32)
    if valid.any():
        selected = points[valid]
        uv[valid, 0] = (
            intrinsics.fx * selected[:, 0] / selected[:, 2] + intrinsics.cx
        )
        uv[valid, 1] = (
            intrinsics.fy * selected[:, 1] / selected[:, 2] + intrinsics.cy
        )
    return uv, valid


def rasterize_joint_polyline(
    points_uv: np.ndarray,
    shape: tuple[int, int],
    radius_px: int,
) -> np.ndarray:
    """Rasterize a round support tube around one ordered finger polyline."""

    height, width = shape
    points = np.asarray(points_uv, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_uv must have shape (N,2), got {points.shape}")
    if radius_px < 0:
        raise ValueError("radius_px must be non-negative")
    out = np.zeros((height, width), dtype=np.uint8)
    if len(points) < 2 or not np.isfinite(points).all():
        return out.astype(bool)
    rounded = np.rint(points).astype(np.int32)
    thickness = max(1, 2 * radius_px + 1)
    cv2.polylines(
        out,
        [rounded.reshape(-1, 1, 2)],
        isClosed=False,
        color=1,
        thickness=thickness,
        lineType=cv2.LINE_8,
    )
    for x, y in rounded:
        cv2.circle(out, (int(x), int(y)), radius_px, 1, thickness=-1)
    return out.astype(bool)


def rasterize_vertex_support(
    points_uv: np.ndarray,
    shape: tuple[int, int],
    radius_px: int,
) -> np.ndarray:
    """Rasterize independent disks around unordered projected MANO vertices."""

    height, width = shape
    points = np.asarray(points_uv, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_uv must have shape (N,2), got {points.shape}")
    if radius_px < 0:
        raise ValueError("radius_px must be non-negative")
    out = np.zeros((height, width), dtype=np.uint8)
    for point in points:
        if not np.isfinite(point).all():
            continue
        x, y = np.rint(point).astype(np.int32)
        if (
            x < -radius_px
            or x >= width + radius_px
            or y < -radius_px
            or y >= height + radius_px
        ):
            continue
        cv2.circle(out, (int(x), int(y)), radius_px, 1, thickness=-1)
    return out.astype(bool)


def _validate_frame_array(
    name: str,
    value: np.ndarray,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != expected_shape:
        raise ValueError(
            f"{name} must share native depth shape {expected_shape}, got {array.shape}; "
            "resizing or cross-view masks is not allowed"
        )
    return array


def finger_joint_indices(finger: str, joint_scope: str) -> tuple[int, ...]:
    """Return ordered native keypoint indices for one support scope."""

    if finger not in FINGER_JOINTS:
        raise ValueError(f"unknown finger {finger!r}")
    joints = FINGER_JOINTS[finger]
    if joint_scope == "full":
        return joints
    if joint_scope == "distal":
        return joints[-3:]
    if joint_scope == "distal2":
        return joints[-2:]
    raise ValueError(
        f"joint_scope must be one of {JOINT_SCOPE_CHOICES}, got {joint_scope!r}"
    )


def _support_for_finger(
    *,
    frame_index: int,
    finger: str,
    image_shape: tuple[int, int],
    native_kpts_2d: np.ndarray | None,
    native_hand_detected: np.ndarray | None,
    vertices_camera: np.ndarray,
    hawor_valid: np.ndarray,
    finger_parts: np.ndarray,
    intrinsics: CameraIntrinsics,
    config: DepthEvidenceConfig,
) -> tuple[np.ndarray, np.uint8, int]:
    if native_kpts_2d is not None and native_hand_detected is not None:
        if bool(native_hand_detected[frame_index]):
            joint_indices = finger_joint_indices(finger, config.joint_scope)
            joints = np.asarray(
                native_kpts_2d[frame_index, joint_indices],
                dtype=np.float32,
            )
            if np.isfinite(joints).all():
                support = rasterize_joint_polyline(
                    joints,
                    image_shape,
                    config.support_radius_px,
                )
                if support.any():
                    return support, SUPPORT_NATIVE_KEYPOINTS, len(joints)

    if bool(hawor_valid[frame_index]):
        selection = np.isin(finger_parts, FINGER_PARTS[finger])
        points = np.asarray(vertices_camera[frame_index, selection], dtype=np.float32)
        uv, positive_z = project_camera_points(points, intrinsics)
        in_frame = (
            positive_z
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] < intrinsics.width)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < intrinsics.height)
        )
        uv = uv[in_frame]
        if len(uv) >= config.min_projected_vertices:
            support = rasterize_vertex_support(
                uv,
                image_shape,
                config.support_radius_px,
            )
            if support.any():
                return support, SUPPORT_PROJECTED_VERTICES, len(uv)

    return np.zeros(image_shape, dtype=bool), SUPPORT_NONE, 0


def estimate_native_depth_evidence(
    *,
    metric_depth_m: np.ndarray,
    modal_object_mask: np.ndarray,
    refined_object_mask: np.ndarray | None,
    model_hand_mask: np.ndarray,
    arm_hand_mask: np.ndarray,
    vertices_camera: np.ndarray,
    hawor_valid: np.ndarray,
    finger_parts: np.ndarray,
    intrinsics: CameraIntrinsics,
    config: DepthEvidenceConfig,
    native_kpts_2d: np.ndarray | None = None,
    native_hand_detected: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute independent per-frame/per-finger native-view depth evidence."""

    config.validate()
    intrinsics.validate()
    depth = np.asarray(metric_depth_m)
    if depth.ndim != 3:
        raise ValueError(f"metric_depth_m must have shape (T,H,W), got {depth.shape}")
    frame_count, height, width = depth.shape
    if (height, width) != (intrinsics.height, intrinsics.width):
        raise ValueError(
            "intrinsics dimensions do not match native depth grid: "
            f"{intrinsics.width}x{intrinsics.height} vs {width}x{height}"
        )
    modal = _validate_frame_array("modal_object_mask", modal_object_mask, depth.shape)
    model = _validate_frame_array("model_hand_mask", model_hand_mask, depth.shape)
    arm = _validate_frame_array("arm_hand_mask", arm_hand_mask, depth.shape)
    refined = (
        None
        if refined_object_mask is None
        else _validate_frame_array(
            "refined_object_mask",
            refined_object_mask,
            depth.shape,
        )
    )
    vertices = np.asarray(vertices_camera, dtype=np.float32)
    if vertices.shape != (frame_count, 778, 3):
        raise ValueError(
            f"vertices_camera must have shape ({frame_count},778,3), "
            f"got {vertices.shape}"
        )
    valid = np.asarray(hawor_valid, dtype=bool)
    if valid.shape != (frame_count,):
        raise ValueError(f"hawor_valid must have shape ({frame_count},)")
    parts = np.asarray(finger_parts, dtype=np.int32)
    if parts.shape != (778,):
        raise ValueError("finger_parts must have shape (778,)")

    if (native_kpts_2d is None) != (native_hand_detected is None):
        raise ValueError(
            "native_kpts_2d and native_hand_detected must be supplied together"
        )
    keypoints = None
    detected = None
    if native_kpts_2d is not None:
        keypoints = np.asarray(native_kpts_2d, dtype=np.float32)
        detected = np.asarray(native_hand_detected, dtype=bool)
        if keypoints.shape != (frame_count, 21, 2):
            raise ValueError(
                f"native_kpts_2d must have shape ({frame_count},21,2), "
                f"got {keypoints.shape}"
            )
        if detected.shape != (frame_count,):
            raise ValueError(
                f"native_hand_detected must have shape ({frame_count},)"
            )

    shape_tf = (frame_count, len(FINGER_NAMES))
    hand_depth = np.full(shape_tf, np.nan, dtype=np.float32)
    object_depth = np.full(shape_tf, np.nan, dtype=np.float32)
    hand_candidates = np.zeros(shape_tf, dtype=np.int32)
    object_candidates = np.zeros(shape_tf, dtype=np.int32)
    hand_samples = np.zeros(shape_tf, dtype=np.int32)
    object_samples = np.zeros(shape_tf, dtype=np.int32)
    hand_inliers = np.zeros(shape_tf, dtype=np.int32)
    object_inliers = np.zeros(shape_tf, dtype=np.int32)
    support_pixels = np.zeros(shape_tf, dtype=np.int32)
    support_points = np.zeros(shape_tf, dtype=np.int32)
    support_source = np.zeros(shape_tf, dtype=np.uint8)

    for frame_index in range(frame_count):
        frame_depth = np.asarray(depth[frame_index], dtype=np.float32)
        visible_hand = (
            np.asarray(model[frame_index], dtype=bool)
            & np.asarray(arm[frame_index], dtype=bool)
        )
        # ``modal`` is commonly a read-only mmap.  Take a private frame copy
        # before intersecting the refined/visible-hand masks in place.
        object_sampling = np.array(
            modal[frame_index],
            dtype=bool,
            copy=True,
        )
        if refined is not None:
            object_sampling &= np.asarray(refined[frame_index], dtype=bool)
        # A pixel cannot support both order hypotheses in one native view.
        object_sampling &= ~visible_hand

        for finger_index, finger in enumerate(FINGER_NAMES):
            support, source, point_count = _support_for_finger(
                frame_index=frame_index,
                finger=finger,
                image_shape=(height, width),
                native_kpts_2d=keypoints,
                native_hand_detected=detected,
                vertices_camera=vertices,
                hawor_valid=valid,
                finger_parts=parts,
                intrinsics=intrinsics,
                config=config,
            )
            support_source[frame_index, finger_index] = source
            support_points[frame_index, finger_index] = point_count
            support_pixels[frame_index, finger_index] = int(support.sum())
            if source == SUPPORT_NONE:
                continue

            hand_selection = support & visible_hand
            object_selection = support & object_sampling
            hand_candidates[frame_index, finger_index] = int(hand_selection.sum())
            object_candidates[frame_index, finger_index] = int(
                object_selection.sum()
            )
            h_depth, h_count, h_inliers = robust_metric_depth(
                frame_depth[hand_selection],
                min_samples=config.min_hand_samples,
                trim_fraction=config.trim_fraction,
                mad_scale=config.mad_scale,
                min_depth_m=config.min_depth_m,
                max_depth_m=config.max_depth_m,
            )
            o_depth, o_count, o_inliers = robust_metric_depth(
                frame_depth[object_selection],
                min_samples=config.min_object_samples,
                trim_fraction=config.trim_fraction,
                mad_scale=config.mad_scale,
                min_depth_m=config.min_depth_m,
                max_depth_m=config.max_depth_m,
            )
            hand_depth[frame_index, finger_index] = h_depth
            object_depth[frame_index, finger_index] = o_depth
            hand_samples[frame_index, finger_index] = h_count
            object_samples[frame_index, finger_index] = o_count
            hand_inliers[frame_index, finger_index] = h_inliers
            object_inliers[frame_index, finger_index] = o_inliers

    return {
        "hand_depth_m": hand_depth,
        "object_depth_m": object_depth,
        "hand_candidate_pixel_count": hand_candidates,
        "object_candidate_pixel_count": object_candidates,
        "hand_sample_count": hand_samples,
        "object_sample_count": object_samples,
        "hand_inlier_count": hand_inliers,
        "object_inlier_count": object_inliers,
        "support_pixel_count": support_pixels,
        "support_point_count": support_points,
        "support_source": support_source,
        "hawor_valid": valid,
        "finger_names": np.asarray(FINGER_NAMES),
        "support_source_labels": np.asarray(SUPPORT_SOURCE_LABELS),
    }


def _side_valid(valid: np.ndarray, side: str, frame_count: int) -> np.ndarray:
    values = np.asarray(valid, dtype=bool)
    side_index = 0 if side == "left" else 1
    if values.shape == (2, frame_count):
        return values[side_index]
    if values.shape == (frame_count, 2):
        return values[:, side_index]
    raise ValueError(
        f"HaWoR valid must have shape (2,{frame_count}) or ({frame_count},2), "
        f"got {values.shape}"
    )


def _load_camera_vertices(
    hawor: np.lib.npyio.NpzFile,
    *,
    side: str,
    frame_count: int,
) -> tuple[np.ndarray, str]:
    key = f"verts_{side}"
    if key not in hawor.files:
        raise ValueError(f"HaWoR file is missing {key}")
    vertices = np.asarray(hawor[key], dtype=np.float32)
    if vertices.shape != (frame_count, 778, 3):
        raise ValueError(
            f"{key} must have shape ({frame_count},778,3), got {vertices.shape}"
        )
    if "frame_is_cam_space" not in hawor.files:
        raise ValueError(
            "HaWoR file must declare frame_is_cam_space for safe projection"
        )
    if bool(np.asarray(hawor["frame_is_cam_space"]).item()):
        return vertices, "camera_space"
    if "R_c2w" not in hawor.files or "t_c2w" not in hawor.files:
        raise ValueError(
            "world-space HaWoR vertices require frame-aligned R_c2w and t_c2w"
        )
    rotation = np.asarray(hawor["R_c2w"], dtype=np.float32)
    translation = np.asarray(hawor["t_c2w"], dtype=np.float32)
    if rotation.shape != (frame_count, 3, 3) or translation.shape != (
        frame_count,
        3,
    ):
        raise ValueError("HaWoR camera transforms are not frame-aligned")
    camera = np.einsum(
        "tvi,tij->tvj",
        vertices - translation[:, None, :],
        rotation,
    )
    return camera.astype(np.float32), "world_to_camera"


def _file_provenance(path: Path, array: np.ndarray | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    result: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if array is not None:
        result["shape"] = list(array.shape)
        result["dtype"] = str(array.dtype)
    return result


def _finite_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float32)
    known = np.isfinite(array)
    return {
        "finite_count": int(known.sum()),
        "finite_fraction": float(known.mean()),
        "median_m": float(np.median(array[known])) if known.any() else None,
    }


def _report(
    *,
    args: argparse.Namespace,
    inputs: dict[str, tuple[Path, np.ndarray] | None],
    result: dict[str, np.ndarray],
    config: DepthEvidenceConfig,
    intrinsics: CameraIntrinsics,
    vertex_frame_conversion: str,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    source = result["support_source"]
    finger_summary = {}
    for index, finger in enumerate(FINGER_NAMES):
        hand_known = np.isfinite(result["hand_depth_m"][:, index])
        object_known = np.isfinite(result["object_depth_m"][:, index])
        finger_summary[finger] = {
            "hand_depth": _finite_summary(result["hand_depth_m"][:, index]),
            "object_depth": _finite_summary(result["object_depth_m"][:, index]),
            "median_hand_sample_count_when_known": (
                float(np.median(result["hand_sample_count"][hand_known, index]))
                if hand_known.any()
                else None
            ),
            "median_object_sample_count_when_known": (
                float(
                    np.median(result["object_sample_count"][object_known, index])
                )
                if object_known.any()
                else None
            ),
            "support_frames": {
                label: int(np.count_nonzero(source[:, index] == code))
                for code, label in enumerate(SUPPORT_SOURCE_LABELS)
            },
        }
    arrays = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in result.items()
    }
    return {
        "schema_version": 1,
        "tool": "precompute_native_depth_evidence.py",
        "camera": args.camera,
        "hand_side": args.side,
        "coordinate_contract": {
            "grid": "one camera native RGB pixel coordinates",
            "depth_value": "camera optical-axis z in metres",
            "cross_camera_projection": False,
            "mask_or_depth_resizing": False,
            "primary_support": "native hand_data kpts_2d finger polylines",
            "native_joint_scope": config.joint_scope,
            "fallback_support": "camera-space HaWoR MANO vertices",
            "visible_hand": "model_hand_mask AND arm_hand_mask",
            "object_sampling": (
                "modal_object_mask AND refined_object_mask AND NOT visible_hand"
                if inputs["refined_object_mask"] is not None
                else "modal_object_mask AND NOT visible_hand"
            ),
            "insufficient_samples": "NaN fail-open; no temporal/inter-view fill",
        },
        "parameters": asdict(config),
        "fallback_projection_intrinsics": asdict(intrinsics),
        "vertex_frame_conversion": vertex_frame_conversion,
        "inputs": {
            name: (
                _file_provenance(path, array)
                if item is not None
                else None
            )
            for name, item in inputs.items()
            for path, array in ([item] if item is not None else [])
        }
        | {
            name: None
            for name, item in inputs.items()
            if item is None
        },
        "outputs": {
            "npz": str(output_path.resolve()),
            "json": str(report_path.resolve()),
            "arrays": arrays,
            "array_semantics": {
                "hand_depth_m": "robust visible-hand camera-z, NaN if unsupported",
                "object_depth_m": "robust hand-local object camera-z, NaN if unsupported",
                "hand_sample_count": "valid metric samples before robust trimming",
                "object_sample_count": "valid metric samples before robust trimming",
                "hand_inlier_count": "samples retained by quantile/MAD filtering",
                "object_inlier_count": "samples retained by quantile/MAD filtering",
                "support_source": "integer index into support_source_labels",
            },
        },
        "summary": {
            "frames": int(result["hand_depth_m"].shape[0]),
            "fingers": finger_summary,
        },
    }


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", required=True, help="Provenance label, e.g. camera_1")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--modal_object_mask", type=Path, required=True)
    parser.add_argument("--refined_object_mask", type=Path, default=None)
    parser.add_argument("--model_hand_mask", type=Path, required=True)
    parser.add_argument("--arm_hand_mask", type=Path, required=True)
    parser.add_argument("--hand_data", type=Path, default=None)
    parser.add_argument("--hawor", type=Path, required=True)
    parser.add_argument("--finger_parts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--support_radius_px", type=int, default=9)
    parser.add_argument(
        "--joint_scope",
        choices=JOINT_SCOPE_CHOICES,
        default="distal",
        help=(
            "Native finger-polyline extent: distal=last 3 joints (default), "
            "distal2=last 2, full=all 4"
        ),
    )
    parser.add_argument("--min_projected_vertices", type=int, default=6)
    parser.add_argument("--min_hand_samples", type=int, default=6)
    parser.add_argument("--min_object_samples", type=int, default=20)
    parser.add_argument("--trim_fraction", type=float, default=0.10)
    parser.add_argument("--mad_scale", type=float, default=3.5)
    parser.add_argument("--min_depth_m", type=float, default=0.02)
    parser.add_argument("--max_depth_m", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_path = args.output.resolve()
    report_path = (
        args.report.resolve()
        if args.report is not None
        else output_path.with_suffix(".json")
    )
    if output_path == report_path:
        raise ValueError("NPZ output and JSON report paths must differ")

    paths = {
        "metric_depth_m": args.depth.resolve(),
        "modal_object_mask": args.modal_object_mask.resolve(),
        "refined_object_mask": (
            args.refined_object_mask.resolve()
            if args.refined_object_mask is not None
            else None
        ),
        "model_hand_mask": args.model_hand_mask.resolve(),
        "arm_hand_mask": args.arm_hand_mask.resolve(),
        "hand_data": args.hand_data.resolve() if args.hand_data is not None else None,
        "hawor": args.hawor.resolve(),
        "finger_parts": args.finger_parts.resolve(),
    }
    for name, path in paths.items():
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    depth = np.load(paths["metric_depth_m"], mmap_mode="r")
    modal = np.load(paths["modal_object_mask"], mmap_mode="r")
    refined = (
        np.load(paths["refined_object_mask"], mmap_mode="r")
        if paths["refined_object_mask"] is not None
        else None
    )
    model_mask = np.load(paths["model_hand_mask"], mmap_mode="r")
    arm_mask = np.load(paths["arm_hand_mask"], mmap_mode="r")
    finger_parts = np.load(paths["finger_parts"])
    if depth.ndim != 3:
        raise ValueError(f"depth must have shape (T,H,W), got {depth.shape}")
    frame_count, height, width = depth.shape

    hand_data_file: np.lib.npyio.NpzFile | None = None
    native_keypoints = None
    native_detected = None
    if paths["hand_data"] is not None:
        hand_data_file = np.load(paths["hand_data"])
        native_keypoints = np.asarray(hand_data_file["kpts_2d"], dtype=np.float32)
        native_detected = np.asarray(
            hand_data_file["hand_detected"], dtype=bool
        )
        if "frame_indices" in hand_data_file.files:
            indices = np.asarray(hand_data_file["frame_indices"], dtype=np.int64)
            if not np.array_equal(indices, np.arange(frame_count)):
                hand_data_file.close()
                raise ValueError("hand_data frame_indices are not dense 0..T-1")

    with np.load(paths["hawor"]) as hawor:
        vertices_camera, vertex_conversion = _load_camera_vertices(
            hawor,
            side=args.side,
            frame_count=frame_count,
        )
        hawor_valid = _side_valid(hawor["valid"], args.side, frame_count)
        if "img_focal" not in hawor.files and (args.fx is None or args.fy is None):
            raise ValueError("fallback projection requires HaWoR img_focal or fx/fy")
        focal = (
            float(np.asarray(hawor["img_focal"]).item())
            if "img_focal" in hawor.files
            else float("nan")
        )

    intrinsics = CameraIntrinsics(
        fx=float(args.fx if args.fx is not None else focal),
        fy=float(args.fy if args.fy is not None else focal),
        cx=float(args.cx if args.cx is not None else width / 2.0),
        cy=float(args.cy if args.cy is not None else height / 2.0),
        width=width,
        height=height,
    )
    config = DepthEvidenceConfig(
        support_radius_px=args.support_radius_px,
        joint_scope=args.joint_scope,
        min_projected_vertices=args.min_projected_vertices,
        min_hand_samples=args.min_hand_samples,
        min_object_samples=args.min_object_samples,
        trim_fraction=args.trim_fraction,
        mad_scale=args.mad_scale,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
    )
    try:
        result = estimate_native_depth_evidence(
            metric_depth_m=depth,
            modal_object_mask=modal,
            refined_object_mask=refined,
            model_hand_mask=model_mask,
            arm_hand_mask=arm_mask,
            vertices_camera=vertices_camera,
            hawor_valid=hawor_valid,
            finger_parts=finger_parts,
            intrinsics=intrinsics,
            config=config,
            native_kpts_2d=native_keypoints,
            native_hand_detected=native_detected,
        )
    finally:
        if hand_data_file is not None:
            hand_data_file.close()

    input_arrays: dict[str, tuple[Path, np.ndarray] | None] = {
        "metric_depth_m": (paths["metric_depth_m"], depth),
        "modal_object_mask": (paths["modal_object_mask"], modal),
        "refined_object_mask": (
            (paths["refined_object_mask"], refined)
            if paths["refined_object_mask"] is not None
            else None
        ),
        "model_hand_mask": (paths["model_hand_mask"], model_mask),
        "arm_hand_mask": (paths["arm_hand_mask"], arm_mask),
        "hand_data": (
            (paths["hand_data"], native_keypoints)
            if paths["hand_data"] is not None
            else None
        ),
        "hawor": (paths["hawor"], vertices_camera),
        "finger_parts": (paths["finger_parts"], finger_parts),
    }
    report = _report(
        args=args,
        inputs=input_arrays,
        result=result,
        config=config,
        intrinsics=intrinsics,
        vertex_frame_conversion=vertex_conversion,
        output_path=output_path,
        report_path=report_path,
    )
    _atomic_save_npz(output_path, result)
    _atomic_write_json(report_path, report)
    print(
        f"[ok] native depth evidence: {output_path} "
        f"(hand={np.isfinite(result['hand_depth_m']).sum()}, "
        f"object={np.isfinite(result['object_depth_m']).sum()} frame-fingers)",
        flush=True,
    )


if __name__ == "__main__":
    main()
