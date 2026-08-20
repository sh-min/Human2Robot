"""Compare approximate-focal and calibrated-focal HaWoR/HaCo outputs.

The current phone pipeline passes one scalar focal length to HaWoR.  HaCo then
reuses that focal length when it projects HaWoR joints to choose the hand crop.
This tool measures the resulting A/B sensitivity without pretending that the
rest of a stereo calibration was consumed.  In particular, principal points,
distortion, stereo extrinsics, rectification, and metric stereo fusion are not
part of this comparison.

Expected input layout::

    ROOT/1/camera_1/rgb/rgb_frame000000.jpg
    ROOT/1/camera_1/rgb_hawor/retarget_input.npz
    ROOT/1/camera_1/contact/rgb_frame000000.npz

The two roots must describe the same synchronized recordings.  The JSON and
CSV reports quantify HaWoR validity/geometry/projection changes and HaCo
probability/mask changes.  ``--make-videos`` additionally renders a direct 2x2
comparison from the saved arrays, so it does not depend on optional HaCo
visualization PNGs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np


SIDES = ("left", "right")
SIDE_INDEX = {"left": 0, "right": 1}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
CAMERA_ROLES = {
    "camera_1": "SH auxiliary",
    "camera_2": "MH primary",
}
CALIBRATION_SCOPE = {
    "tested": (
        "Per-view scalar focal length passed to HaWoR and reused by HaCo "
        "for hand-crop projection."
    ),
    "not_tested": [
        "calibrated principal point (cx, cy)",
        "lens-distortion correction",
        "stereo rotation/translation",
        "stereo rectification",
        "metric triangulation or cross-view fusion",
    ],
    "interpretation": (
        "These are A/B difference and stability measurements, not accuracy "
        "measurements. Ground-truth 2D/3D hand/contact labels are required "
        "to claim that either branch is more accurate."
    ),
}


def natural_key(value: str) -> tuple[object, ...]:
    """Return a deterministic human/numeric sorting key."""

    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def project_points(
    points: np.ndarray,
    focal_px: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Project camera-space points with the centered pinhole used by HaWoR."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if not math.isfinite(focal_px) or focal_px <= 0:
        raise ValueError(f"focal length must be finite and positive: {focal_px}")
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")

    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-8)
    result[valid, 0] = (
        focal_px * points[valid, 0] / points[valid, 2] + width / 2.0
    )
    result[valid, 1] = (
        focal_px * points[valid, 1] / points[valid, 2] + height / 2.0
    )
    return result


def _summary(arrays: Sequence[np.ndarray]) -> dict[str, int | float | None]:
    finite_parts: list[np.ndarray] = []
    for array in arrays:
        values = np.asarray(array, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            finite_parts.append(values)
    if not finite_parts:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    values = np.concatenate(finite_parts)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _bbox_iou(
    first: np.ndarray,
    second: np.ndarray,
    width: int,
    height: int,
) -> float | None:
    def bounds(points: np.ndarray) -> tuple[float, float, float, float] | None:
        finite = points[np.isfinite(points).all(axis=1)]
        if finite.size == 0:
            return None
        x0 = float(np.clip(finite[:, 0].min(), 0, width))
        y0 = float(np.clip(finite[:, 1].min(), 0, height))
        x1 = float(np.clip(finite[:, 0].max(), 0, width))
        y1 = float(np.clip(finite[:, 1].max(), 0, height))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    a = bounds(first)
    b = bounds(second)
    if a is None or b is None:
        return None
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else None


@dataclass
class HaWoRAccumulator:
    total_hand_frames: int = 0
    both_valid: int = 0
    approx_only_valid: int = 0
    calibrated_only_valid: int = 0
    neither_valid: int = 0
    joint_3d: list[np.ndarray] = field(default_factory=list)
    vertex_3d: list[np.ndarray] = field(default_factory=list)
    joint_projection: list[np.ndarray] = field(default_factory=list)
    vertex_projection: list[np.ndarray] = field(default_factory=list)
    bbox_iou: list[np.ndarray] = field(default_factory=list)

    def add_validity(self, approx_valid: bool, calibrated_valid: bool) -> None:
        self.total_hand_frames += 1
        if approx_valid and calibrated_valid:
            self.both_valid += 1
        elif approx_valid:
            self.approx_only_valid += 1
        elif calibrated_valid:
            self.calibrated_only_valid += 1
        else:
            self.neither_valid += 1

    def merge(self, other: "HaWoRAccumulator") -> None:
        for name in (
            "total_hand_frames",
            "both_valid",
            "approx_only_valid",
            "calibrated_only_valid",
            "neither_valid",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in (
            "joint_3d",
            "vertex_3d",
            "joint_projection",
            "vertex_projection",
            "bbox_iou",
        ):
            getattr(self, name).extend(getattr(other, name))

    def to_report(self) -> dict[str, object]:
        agreement = self.both_valid + self.neither_valid
        return {
            "total_hand_frames": self.total_hand_frames,
            "both_valid_hand_frames": self.both_valid,
            "approx_only_valid_hand_frames": self.approx_only_valid,
            "calibrated_only_valid_hand_frames": self.calibrated_only_valid,
            "neither_valid_hand_frames": self.neither_valid,
            "validity_agreement_rate": _safe_rate(
                agreement, self.total_hand_frames
            ),
            "joint_3d_displacement_camera_units": _summary(self.joint_3d),
            "vertex_3d_displacement_camera_units": _summary(self.vertex_3d),
            "projected_joint_displacement_px": _summary(
                self.joint_projection
            ),
            "projected_vertex_displacement_px": _summary(
                self.vertex_projection
            ),
            "projected_joint_bbox_iou": _summary(self.bbox_iou),
        }


@dataclass
class HaCoAccumulator:
    total_hand_frames: int = 0
    both_valid: int = 0
    approx_only_valid: int = 0
    calibrated_only_valid: int = 0
    neither_valid: int = 0
    probability_abs_delta: list[np.ndarray] = field(default_factory=list)
    mask_iou_all: list[np.ndarray] = field(default_factory=list)
    mask_iou_active: list[np.ndarray] = field(default_factory=list)
    contact_count_abs_delta: list[np.ndarray] = field(default_factory=list)
    approx_contact_count: list[np.ndarray] = field(default_factory=list)
    calibrated_contact_count: list[np.ndarray] = field(default_factory=list)
    exact_mask_pairs: int = 0
    both_empty_mask_pairs: int = 0
    flipped_vertices: int = 0
    compared_vertices: int = 0

    def add_record(
        self,
        approx_valid: bool,
        calibrated_valid: bool,
        approx_probability: np.ndarray | None = None,
        calibrated_probability: np.ndarray | None = None,
        approx_mask: np.ndarray | None = None,
        calibrated_mask: np.ndarray | None = None,
    ) -> None:
        self.total_hand_frames += 1
        if approx_valid and calibrated_valid:
            self.both_valid += 1
        elif approx_valid:
            self.approx_only_valid += 1
        elif calibrated_valid:
            self.calibrated_only_valid += 1
        else:
            self.neither_valid += 1

        if not (approx_valid and calibrated_valid):
            return
        if any(
            value is None
            for value in (
                approx_probability,
                calibrated_probability,
                approx_mask,
                calibrated_mask,
            )
        ):
            raise ValueError("both-valid HaCo records require probability and mask arrays")

        probability_a = np.asarray(approx_probability, dtype=np.float64).reshape(-1)
        probability_b = np.asarray(calibrated_probability, dtype=np.float64).reshape(-1)
        mask_a = np.asarray(approx_mask, dtype=bool).reshape(-1)
        mask_b = np.asarray(calibrated_mask, dtype=bool).reshape(-1)
        if probability_a.shape != probability_b.shape:
            raise ValueError("HaCo probability shapes do not match")
        if mask_a.shape != mask_b.shape or mask_a.shape != probability_a.shape:
            raise ValueError("HaCo mask/probability shapes do not match")

        finite = np.isfinite(probability_a) & np.isfinite(probability_b)
        self.probability_abs_delta.append(
            np.abs(probability_a[finite] - probability_b[finite])
        )

        xor = np.logical_xor(mask_a, mask_b)
        intersection = int(np.logical_and(mask_a, mask_b).sum())
        union = int(np.logical_or(mask_a, mask_b).sum())
        iou = float(intersection / union) if union else 1.0
        self.mask_iou_all.append(np.asarray([iou], dtype=np.float64))
        if union:
            self.mask_iou_active.append(np.asarray([iou], dtype=np.float64))
        else:
            self.both_empty_mask_pairs += 1
        if np.array_equal(mask_a, mask_b):
            self.exact_mask_pairs += 1
        self.flipped_vertices += int(xor.sum())
        self.compared_vertices += int(mask_a.size)

        count_a, count_b = int(mask_a.sum()), int(mask_b.sum())
        self.approx_contact_count.append(np.asarray([count_a], dtype=np.float64))
        self.calibrated_contact_count.append(
            np.asarray([count_b], dtype=np.float64)
        )
        self.contact_count_abs_delta.append(
            np.asarray([abs(count_a - count_b)], dtype=np.float64)
        )

    def merge(self, other: "HaCoAccumulator") -> None:
        for name in (
            "total_hand_frames",
            "both_valid",
            "approx_only_valid",
            "calibrated_only_valid",
            "neither_valid",
            "exact_mask_pairs",
            "both_empty_mask_pairs",
            "flipped_vertices",
            "compared_vertices",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in (
            "probability_abs_delta",
            "mask_iou_all",
            "mask_iou_active",
            "contact_count_abs_delta",
            "approx_contact_count",
            "calibrated_contact_count",
        ):
            getattr(self, name).extend(getattr(other, name))

    def to_report(self) -> dict[str, object]:
        agreement = self.both_valid + self.neither_valid
        return {
            "total_hand_frames": self.total_hand_frames,
            "both_valid_hand_frames": self.both_valid,
            "approx_only_valid_hand_frames": self.approx_only_valid,
            "calibrated_only_valid_hand_frames": self.calibrated_only_valid,
            "neither_valid_hand_frames": self.neither_valid,
            "validity_agreement_rate": _safe_rate(
                agreement, self.total_hand_frames
            ),
            "contact_probability_abs_delta": _summary(
                self.probability_abs_delta
            ),
            "contact_mask_iou_including_both_empty_as_one": _summary(
                self.mask_iou_all
            ),
            "contact_mask_iou_when_either_active": _summary(
                self.mask_iou_active
            ),
            "contact_mask_exact_agreement_rate": _safe_rate(
                self.exact_mask_pairs, self.both_valid
            ),
            "both_empty_contact_mask_pairs": self.both_empty_mask_pairs,
            "contact_vertex_flip_rate": _safe_rate(
                self.flipped_vertices, self.compared_vertices
            ),
            "contact_count_abs_delta": _summary(
                self.contact_count_abs_delta
            ),
            "approx_contact_vertex_count": _summary(
                self.approx_contact_count
            ),
            "calibrated_contact_vertex_count": _summary(
                self.calibrated_contact_count
            ),
        }


def _require_hawor_schema(data: Mapping[str, np.ndarray], label: str) -> int:
    required = {"valid", "img_focal"}
    required.update(f"joints_{side}" for side in SIDES)
    required.update(f"verts_{side}" for side in SIDES)
    missing = sorted(required - set(data.keys()))
    if missing:
        raise ValueError(f"{label} HaWoR NPZ is missing keys: {missing}")

    valid = np.asarray(data["valid"])
    if valid.ndim != 2 or valid.shape[0] != 2:
        raise ValueError(f"{label} valid must have shape (2, N), got {valid.shape}")
    frame_count = int(valid.shape[1])
    for side in SIDES:
        joints = np.asarray(data[f"joints_{side}"])
        vertices = np.asarray(data[f"verts_{side}"])
        if joints.shape != (frame_count, 21, 3):
            raise ValueError(
                f"{label} joints_{side} has invalid shape {joints.shape}"
            )
        if vertices.shape != (frame_count, 778, 3):
            raise ValueError(
                f"{label} verts_{side} has invalid shape {vertices.shape}"
            )
    focal = float(np.asarray(data["img_focal"]).item())
    if not math.isfinite(focal) or focal <= 0:
        raise ValueError(f"{label} img_focal is invalid: {focal}")
    return frame_count


def compare_hawor_arrays(
    approx: Mapping[str, np.ndarray],
    calibrated: Mapping[str, np.ndarray],
    width: int,
    height: int,
    frame_indices: Sequence[int] | None = None,
) -> tuple[HaWoRAccumulator, dict[str, HaWoRAccumulator]]:
    """Compare two aligned HaWoR archives."""

    count_a = _require_hawor_schema(approx, "approx")
    count_b = _require_hawor_schema(calibrated, "calibrated")
    if count_a != count_b:
        raise ValueError(f"HaWoR frame count mismatch: {count_a} != {count_b}")
    if frame_indices is None:
        selected_indices = list(range(count_a))
    else:
        selected_indices = [int(index) for index in frame_indices]
        if len(set(selected_indices)) != len(selected_indices):
            raise ValueError("HaWoR frame_indices must not contain duplicates")
        bad = [index for index in selected_indices if not 0 <= index < count_a]
        if bad:
            raise ValueError(f"HaWoR frame_indices are out of range: {bad[:8]}")
    focal_a = float(np.asarray(approx["img_focal"]).item())
    focal_b = float(np.asarray(calibrated["img_focal"]).item())

    per_side = {side: HaWoRAccumulator() for side in SIDES}
    for side in SIDES:
        accumulator = per_side[side]
        side_index = SIDE_INDEX[side]
        valid_a = np.asarray(approx["valid"], dtype=bool)[side_index]
        valid_b = np.asarray(calibrated["valid"], dtype=bool)[side_index]
        for frame_idx in selected_indices:
            is_valid_a = bool(valid_a[frame_idx])
            is_valid_b = bool(valid_b[frame_idx])
            accumulator.add_validity(is_valid_a, is_valid_b)
            if not (is_valid_a and is_valid_b):
                continue

            joints_a = np.asarray(approx[f"joints_{side}"][frame_idx])
            joints_b = np.asarray(calibrated[f"joints_{side}"][frame_idx])
            vertices_a = np.asarray(approx[f"verts_{side}"][frame_idx])
            vertices_b = np.asarray(calibrated[f"verts_{side}"][frame_idx])
            joint_3d = np.linalg.norm(joints_a - joints_b, axis=1)
            vertex_3d = np.linalg.norm(vertices_a - vertices_b, axis=1)
            accumulator.joint_3d.append(joint_3d)
            accumulator.vertex_3d.append(vertex_3d)

            projected_joints_a = project_points(
                joints_a, focal_a, width, height
            )
            projected_joints_b = project_points(
                joints_b, focal_b, width, height
            )
            projected_vertices_a = project_points(
                vertices_a, focal_a, width, height
            )
            projected_vertices_b = project_points(
                vertices_b, focal_b, width, height
            )
            joint_good = (
                np.isfinite(projected_joints_a).all(axis=1)
                & np.isfinite(projected_joints_b).all(axis=1)
            )
            vertex_good = (
                np.isfinite(projected_vertices_a).all(axis=1)
                & np.isfinite(projected_vertices_b).all(axis=1)
            )
            accumulator.joint_projection.append(
                np.linalg.norm(
                    projected_joints_a[joint_good]
                    - projected_joints_b[joint_good],
                    axis=1,
                )
            )
            accumulator.vertex_projection.append(
                np.linalg.norm(
                    projected_vertices_a[vertex_good]
                    - projected_vertices_b[vertex_good],
                    axis=1,
                )
            )
            iou = _bbox_iou(
                projected_joints_a,
                projected_joints_b,
                width,
                height,
            )
            if iou is not None:
                accumulator.bbox_iou.append(np.asarray([iou]))

    aggregate = HaWoRAccumulator()
    for accumulator in per_side.values():
        aggregate.merge(accumulator)
    return aggregate, per_side


def compare_contact_records(
    approx_records: Sequence[Mapping[str, np.ndarray]],
    calibrated_records: Sequence[Mapping[str, np.ndarray]],
) -> tuple[HaCoAccumulator, dict[str, HaCoAccumulator]]:
    """Compare aligned in-memory HaCo records (mainly useful for tests)."""

    if len(approx_records) != len(calibrated_records):
        raise ValueError("HaCo record counts do not match")
    per_side = {side: HaCoAccumulator() for side in SIDES}
    for record_a, record_b in zip(approx_records, calibrated_records):
        for side in SIDES:
            required = {
                f"{side}_valid",
                f"{side}_contact_probability",
                f"{side}_contact_mask",
            }
            missing_a = sorted(required - set(record_a.keys()))
            missing_b = sorted(required - set(record_b.keys()))
            if missing_a or missing_b:
                raise ValueError(
                    f"HaCo {side} keys missing: approx={missing_a}, "
                    f"calibrated={missing_b}"
                )
            valid_a = bool(np.asarray(record_a[f"{side}_valid"]).item())
            valid_b = bool(np.asarray(record_b[f"{side}_valid"]).item())
            per_side[side].add_record(
                valid_a,
                valid_b,
                np.asarray(record_a[f"{side}_contact_probability"]),
                np.asarray(record_b[f"{side}_contact_probability"]),
                np.asarray(record_a[f"{side}_contact_mask"]),
                np.asarray(record_b[f"{side}_contact_mask"]),
            )

    aggregate = HaCoAccumulator()
    for accumulator in per_side.values():
        aggregate.merge(accumulator)
    return aggregate, per_side


def _image_map(rgb_dir: Path) -> dict[str, Path]:
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory does not exist: {rgb_dir}")
    result: dict[str, Path] = {}
    for path in sorted(rgb_dir.iterdir(), key=lambda item: natural_key(item.name)):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in result:
            raise ValueError(f"duplicate RGB frame stem in {rgb_dir}: {path.stem}")
        result[path.stem] = path
    if not result:
        raise FileNotFoundError(f"no RGB frames found in {rgb_dir}")
    return result


def _npz_map(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"contact directory does not exist: {directory}")
    result = {path.stem: path for path in directory.glob("*.npz")}
    if not result:
        raise FileNotFoundError(f"no contact NPZ files found in {directory}")
    return result


def _matching_keys(
    first: Mapping[str, object],
    second: Mapping[str, object],
    label: str,
    allow_partial: bool,
) -> list[str]:
    first_keys, second_keys = set(first), set(second)
    missing_first = sorted(second_keys - first_keys, key=natural_key)
    missing_second = sorted(first_keys - second_keys, key=natural_key)
    if (missing_first or missing_second) and not allow_partial:
        raise ValueError(
            f"{label} sets differ; missing from approx={missing_first[:8]}, "
            f"missing from calibrated={missing_second[:8]}"
        )
    common = sorted(first_keys & second_keys, key=natural_key)
    if not common:
        raise ValueError(f"no matching {label} entries")
    return common


def _load_contact_record(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]).copy() for key in data.files}


def compare_contact_directories(
    approx_dir: Path,
    calibrated_dir: Path,
    expected_stems: Iterable[str] | None = None,
    allow_partial: bool = False,
) -> tuple[HaCoAccumulator, dict[str, HaCoAccumulator], int]:
    maps_a = _npz_map(approx_dir)
    maps_b = _npz_map(calibrated_dir)
    keys = _matching_keys(maps_a, maps_b, "contact frame", allow_partial)
    if expected_stems is not None:
        expected_ordered = list(expected_stems)
        expected = set(expected_ordered)
        available = set(keys)
        if not expected.issubset(available) and not allow_partial:
            missing = sorted(expected - available, key=natural_key)
            raise ValueError(
                "contact/RGB alignment differs; "
                f"missing={missing[:8]}"
            )
        keys = [key for key in expected_ordered if key in available]

    records_a: list[dict[str, np.ndarray]] = []
    records_b: list[dict[str, np.ndarray]] = []
    for key in keys:
        record_a = _load_contact_record(maps_a[key])
        record_b = _load_contact_record(maps_b[key])
        for label, record in (("approx", record_a), ("calibrated", record_b)):
            if "source_filename" in record:
                source = str(np.asarray(record["source_filename"]).item())
                if Path(source).stem != key:
                    raise ValueError(
                        f"{label} contact source mismatch for {key}: {source}"
                    )
        if "hawor_frame_index" in record_a and "hawor_frame_index" in record_b:
            index_a = int(np.asarray(record_a["hawor_frame_index"]).item())
            index_b = int(np.asarray(record_b["hawor_frame_index"]).item())
            if index_a != index_b:
                raise ValueError(
                    f"HaCo HaWoR index mismatch for {key}: {index_a} != {index_b}"
                )
        records_a.append(record_a)
        records_b.append(record_b)
    aggregate, per_side = compare_contact_records(records_a, records_b)
    return aggregate, per_side, len(keys)


def load_temporal_alignment(episode_dir: Path) -> dict[str, object]:
    """Load the explicit camera-1-to-camera-2 time-axis contract."""

    manifest_path = episode_dir / "stereo_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"stereo manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        temporal = manifest["temporal_alignment"]
        offset = int(temporal["camera1_frame_offset"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid temporal alignment in {manifest_path}"
        ) from exc
    if not isinstance(temporal, dict):
        raise ValueError(f"temporal_alignment must be an object: {manifest_path}")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "camera1_frame_offset": offset,
        "reference_view": temporal.get("reference_view"),
        "camera1_lookup": temporal.get("camera1_lookup"),
        "out_of_range_policy": temporal.get("out_of_range_policy"),
        "motion_correlation_audit": temporal.get("motion_correlation_audit"),
    }


def aligned_frame_selection(
    ordered_frame_keys: Sequence[str],
    camera1_frame_offset: int,
) -> list[tuple[int, int, str]]:
    """Map the immutable MH/reference axis to available SH source frames.

    ``camera1_frame_offset`` follows the manifest convention:
    ``SH source index = MH/reference index + offset``.  Out-of-range SH
    lookups are omitted from quantitative comparison (the fusion pipeline's
    corresponding behavior is fail-open).
    """

    selection: list[tuple[int, int, str]] = []
    frame_count = len(ordered_frame_keys)
    for reference_index in range(frame_count):
        source_index = reference_index + int(camera1_frame_offset)
        if 0 <= source_index < frame_count:
            selection.append(
                (reference_index, source_index, ordered_frame_keys[source_index])
            )
    return selection


@dataclass
class ViewComparison:
    episode: str
    view: str
    report: dict[str, object]
    hawor: HaWoRAccumulator
    haco: HaCoAccumulator


def compare_view(
    approx_view: Path,
    calibrated_view: Path,
    episode: str,
    view: str,
    allow_partial: bool = False,
) -> ViewComparison:
    temporal_a = load_temporal_alignment(approx_view.parent)
    temporal_b = load_temporal_alignment(calibrated_view.parent)
    offset_a = int(temporal_a["camera1_frame_offset"])
    offset_b = int(temporal_b["camera1_frame_offset"])
    if offset_a != offset_b:
        raise ValueError(
            f"temporal alignment is an A/B confound for {episode}: "
            f"approx offset {offset_a} != calibrated offset {offset_b}"
        )
    applied_offset = offset_a if view == "camera_1" else 0

    images_a = _image_map(approx_view / "rgb")
    images_b = _image_map(calibrated_view / "rgb")
    frame_keys = _matching_keys(images_a, images_b, "RGB frame", allow_partial)
    aligned = aligned_frame_selection(frame_keys, applied_offset)
    aligned_frame_keys = [item[2] for item in aligned]
    aligned_source_indices = [item[1] for item in aligned]
    if not aligned:
        raise ValueError(
            f"temporal offset {applied_offset} leaves no frames for {episode}/{view}"
        )
    first_a = cv2.imread(str(images_a[frame_keys[0]]))
    first_b = cv2.imread(str(images_b[frame_keys[0]]))
    if first_a is None or first_b is None:
        raise RuntimeError(f"could not read first RGB frame for {episode}/{view}")
    if first_a.shape[:2] != first_b.shape[:2]:
        raise ValueError(
            f"RGB geometry mismatch for {episode}/{view}: "
            f"{first_a.shape[:2]} != {first_b.shape[:2]}"
        )
    height, width = first_a.shape[:2]

    hawor_a_path = approx_view / "rgb_hawor" / "retarget_input.npz"
    hawor_b_path = calibrated_view / "rgb_hawor" / "retarget_input.npz"
    if not hawor_a_path.is_file() or not hawor_b_path.is_file():
        raise FileNotFoundError(
            f"missing HaWoR archive for {episode}/{view}: "
            f"{hawor_a_path}, {hawor_b_path}"
        )
    with np.load(hawor_a_path, allow_pickle=False) as hawor_a, np.load(
        hawor_b_path, allow_pickle=False
    ) as hawor_b:
        frame_count_a = _require_hawor_schema(hawor_a, "approx")
        frame_count_b = _require_hawor_schema(hawor_b, "calibrated")
        if not allow_partial and (
            frame_count_a != len(images_a) or frame_count_b != len(images_b)
        ):
            raise ValueError(
                f"RGB/HaWoR frame count mismatch for {episode}/{view}: "
                f"approx={len(images_a)}/{frame_count_a}, "
                f"calibrated={len(images_b)}/{frame_count_b}"
            )
        hawor, hawor_sides = compare_hawor_arrays(
            hawor_a,
            hawor_b,
            width,
            height,
            frame_indices=aligned_source_indices,
        )
        focal_a = float(np.asarray(hawor_a["img_focal"]).item())
        focal_b = float(np.asarray(hawor_b["img_focal"]).item())

    haco, haco_sides, contact_pairs = compare_contact_directories(
        approx_view / "contact",
        calibrated_view / "contact",
        expected_stems=aligned_frame_keys,
        allow_partial=allow_partial,
    )
    report: dict[str, object] = {
        "episode": episode,
        "view": view,
        "role": CAMERA_ROLES.get(view, "unspecified"),
        "image_size": [width, height],
        "rgb_frame_pairs": len(frame_keys),
        "aligned_reference_frames_compared": len(aligned),
        "temporal_boundary_frames_fail_open": len(frame_keys) - len(aligned),
        "contact_frame_pairs": contact_pairs,
        "temporal_alignment": {
            "approx_manifest": temporal_a,
            "calibrated_manifest": temporal_b,
            "offsets_match": True,
            "offset_applied_to_this_view": view == "camera_1",
            "applied_camera1_frame_offset": applied_offset,
            "lookup_convention": (
                "camera_1/SH source index = camera_2/MH reference index + offset"
            ),
            "quantitative_boundary_policy": (
                "omit out-of-range SH frames, matching fail-open fusion evidence"
            ),
        },
        "approx_focal_px": focal_a,
        "calibrated_focal_px": focal_b,
        "focal_delta_px": focal_b - focal_a,
        "focal_ratio_calibrated_over_approx": focal_b / focal_a,
        "hawor": hawor.to_report(),
        "hawor_by_side": {
            side: accumulator.to_report()
            for side, accumulator in hawor_sides.items()
        },
        "haco": haco.to_report(),
        "haco_by_side": {
            side: accumulator.to_report()
            for side, accumulator in haco_sides.items()
        },
    }
    return ViewComparison(episode, view, report, hawor, haco)


def _fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, int(round(image.shape[1] * scale)))
    resized_height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - resized_width) // 2
    y0 = (height - resized_height) // 2
    panel[y0:y0 + resized_height, x0:x0 + resized_width] = resized
    return panel


def _draw_points(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    radius: int,
    stride: int = 1,
) -> None:
    height, width = image.shape[:2]
    for point in points[::stride]:
        if not np.isfinite(point).all():
            continue
        x, y = int(round(point[0])), int(round(point[1]))
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)


def render_overlay(
    rgb: np.ndarray,
    hawor: Mapping[str, np.ndarray],
    frame_idx: int,
    mode: str,
    contact: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, str]:
    """Render one saved HaWoR or HaCo result over its RGB frame."""

    if mode not in {"hawor", "haco"}:
        raise ValueError("mode must be 'hawor' or 'haco'")
    output = rgb.copy()
    focal = float(np.asarray(hawor["img_focal"]).item())
    height, width = output.shape[:2]
    valid = np.asarray(hawor["valid"], dtype=bool)
    contact_counts: list[str] = []
    any_valid = False
    for side in SIDES:
        side_idx = SIDE_INDEX[side]
        if frame_idx >= valid.shape[1] or not bool(valid[side_idx, frame_idx]):
            continue
        any_valid = True
        vertices = project_points(
            np.asarray(hawor[f"verts_{side}"][frame_idx]),
            focal,
            width,
            height,
        )
        joints = project_points(
            np.asarray(hawor[f"joints_{side}"][frame_idx]),
            focal,
            width,
            height,
        )
        if mode == "hawor":
            color = (255, 225, 0) if side == "left" else (255, 0, 225)
            _draw_points(output, vertices, color, radius=1, stride=2)
            joint_color = (0, 180, 255)
        else:
            _draw_points(output, vertices, (105, 105, 105), radius=1, stride=4)
            mask_key = f"{side}_contact_mask"
            if contact is not None and mask_key in contact:
                mask = np.asarray(contact[mask_key], dtype=bool).reshape(-1)
                if mask.shape != (vertices.shape[0],):
                    raise ValueError(
                        f"{mask_key} shape {mask.shape} does not match vertices"
                    )
                contact_color = (
                    (40, 255, 40) if side == "left" else (0, 165, 255)
                )
                _draw_points(
                    output, vertices[mask], contact_color, radius=3, stride=1
                )
                contact_counts.append(f"{side[0].upper()}={int(mask.sum())}")
            joint_color = (210, 210, 210)

        for start, end in HAND_EDGES:
            p0, p1 = joints[start], joints[end]
            if np.isfinite(p0).all() and np.isfinite(p1).all():
                cv2.line(
                    output,
                    tuple(np.round(p0).astype(int)),
                    tuple(np.round(p1).astype(int)),
                    joint_color,
                    2,
                    cv2.LINE_AA,
                )
    if not any_valid:
        return output, "no valid hand"
    if mode == "haco":
        return output, "contact vertices " + (", ".join(contact_counts) or "missing")
    return output, "left=cyan, right=magenta"


def _tile(
    image: np.ndarray,
    title: str,
    subtitle: str,
    panel_width: int,
    panel_height: int,
    header_height: int = 52,
) -> np.ndarray:
    panel = _fit_panel(image, panel_width, panel_height)
    tile = np.zeros((header_height + panel_height, panel_width, 3), dtype=np.uint8)
    tile[header_height:] = panel
    cv2.putText(
        tile, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX,
        0.62, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        tile, subtitle, (12, 43), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (175, 205, 220), 1, cv2.LINE_AA,
    )
    return tile


def compose_ab_frame(
    rgb: np.ndarray,
    approx_hawor: Mapping[str, np.ndarray],
    calibrated_hawor: Mapping[str, np.ndarray],
    frame_idx: int,
    approx_contact: Mapping[str, np.ndarray] | None = None,
    calibrated_contact: Mapping[str, np.ndarray] | None = None,
    panel_width: int = 640,
    panel_height: int = 360,
    timeline_note: str = "",
) -> np.ndarray:
    """Compose the 2x2 visual used by the optional comparison video."""

    approx_focal = float(np.asarray(approx_hawor["img_focal"]).item())
    calibrated_focal = float(np.asarray(calibrated_hawor["img_focal"]).item())
    hawor_a, hawor_a_note = render_overlay(
        rgb, approx_hawor, frame_idx, "hawor"
    )
    hawor_b, hawor_b_note = render_overlay(
        rgb, calibrated_hawor, frame_idx, "hawor"
    )
    haco_a, haco_a_note = render_overlay(
        rgb, approx_hawor, frame_idx, "haco", approx_contact
    )
    haco_b, haco_b_note = render_overlay(
        rgb, calibrated_hawor, frame_idx, "haco", calibrated_contact
    )
    top = np.hstack(
        [
            _tile(
                hawor_a,
                "APPROX - HaWoR",
                f"fx={approx_focal:.2f}px | {hawor_a_note}",
                panel_width,
                panel_height,
            ),
            _tile(
                hawor_b,
                "CALIBRATED - HaWoR",
                f"fx={calibrated_focal:.2f}px | {hawor_b_note}",
                panel_width,
                panel_height,
            ),
        ]
    )
    bottom = np.hstack(
        [
            _tile(
                haco_a,
                "APPROX - HaCo",
                haco_a_note,
                panel_width,
                panel_height,
            ),
            _tile(
                haco_b,
                "CALIBRATED - HaCo",
                haco_b_note,
                panel_width,
                panel_height,
            ),
        ]
    )
    canvas = np.vstack([top, bottom])
    if timeline_note:
        text_size = cv2.getTextSize(
            timeline_note, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
        )[0]
        x0 = max(4, canvas.shape[1] - text_size[0] - 12)
        cv2.rectangle(
            canvas,
            (x0 - 5, 2),
            (canvas.shape[1] - 2, 22),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            canvas,
            timeline_note,
            (x0, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def compose_fail_open_frame(
    panel_width: int,
    panel_height: int,
    reference_index: int,
    source_index: int,
) -> np.ndarray:
    """Render an explicit empty SH cue at an offset boundary."""

    blank = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    note = (
        f"MH/reference {reference_index} -> SH {source_index}: "
        "out of range (fail-open)"
    )
    tiles = [
        _tile(blank, title, note, panel_width, panel_height)
        for title in (
            "APPROX - HaWoR",
            "CALIBRATED - HaWoR",
            "APPROX - HaCo",
            "CALIBRATED - HaCo",
        )
    ]
    return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])


def render_comparison_video(
    approx_view: Path,
    calibrated_view: Path,
    output_path: Path,
    fps: float = 24.0,
    panel_width: int = 640,
    panel_height: int = 360,
    max_frames: int = 0,
    allow_partial: bool = False,
    camera1_frame_offset: int = 0,
) -> tuple[Path, Path]:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if panel_width <= 0 or panel_height <= 0:
        raise ValueError("panel dimensions must be positive")
    if max_frames < 0:
        raise ValueError("max_frames must be non-negative")

    images_a = _image_map(approx_view / "rgb")
    images_b = _image_map(calibrated_view / "rgb")
    keys = _matching_keys(images_a, images_b, "RGB frame", allow_partial)
    reference_indices = list(range(len(keys)))
    if max_frames:
        reference_indices = reference_indices[:max_frames]
    contacts_a = _npz_map(approx_view / "contact")
    contacts_b = _npz_map(calibrated_view / "contact")
    if not allow_partial:
        _matching_keys(contacts_a, contacts_b, "contact frame", False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path = output_path.with_suffix(".jpg")
    header_height = 52
    output_width = panel_width * 2
    output_height = (panel_height + header_height) * 2
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{output_width}x{output_height}",
        "-r", f"{fps:.8g}", "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output_path),
    ]
    try:
        encoder = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg executable was not found") from exc

    preview: np.ndarray | None = None
    try:
        with np.load(
            approx_view / "rgb_hawor" / "retarget_input.npz",
            allow_pickle=False,
        ) as hawor_a, np.load(
            calibrated_view / "rgb_hawor" / "retarget_input.npz",
            allow_pickle=False,
        ) as hawor_b:
            _require_hawor_schema(hawor_a, "approx")
            _require_hawor_schema(hawor_b, "calibrated")
            for output_index, reference_index in enumerate(reference_indices):
                source_index = reference_index + int(camera1_frame_offset)
                if not 0 <= source_index < len(keys):
                    canvas = compose_fail_open_frame(
                        panel_width,
                        panel_height,
                        reference_index,
                        source_index,
                    )
                    if output_index == len(reference_indices) // 2:
                        preview = canvas.copy()
                    if encoder.stdin is None:
                        raise RuntimeError("ffmpeg input pipe was not created")
                    encoder.stdin.write(canvas.tobytes())
                    continue

                key = keys[source_index]
                rgb = cv2.imread(str(images_a[key]))
                if rgb is None:
                    raise RuntimeError(f"could not read RGB frame: {images_a[key]}")
                context_a = (
                    np.load(contacts_a[key], allow_pickle=False)
                    if key in contacts_a else nullcontext({})
                )
                context_b = (
                    np.load(contacts_b[key], allow_pickle=False)
                    if key in contacts_b else nullcontext({})
                )
                with context_a as contact_a, context_b as contact_b:
                    canvas = compose_ab_frame(
                        rgb,
                        hawor_a,
                        hawor_b,
                        source_index,
                        contact_a,
                        contact_b,
                        panel_width,
                        panel_height,
                        (
                            f"MH/reference {reference_index} -> "
                            f"SH/source {source_index}"
                            if camera1_frame_offset else f"frame {reference_index}"
                        ),
                    )
                if output_index == len(reference_indices) // 2:
                    preview = canvas.copy()
                if encoder.stdin is None:
                    raise RuntimeError("ffmpeg input pipe was not created")
                encoder.stdin.write(canvas.tobytes())
    except Exception:
        if encoder.stdin is not None:
            encoder.stdin.close()
        encoder.kill()
        encoder.wait()
        raise
    finally:
        if encoder.stdin is not None and not encoder.stdin.closed:
            encoder.stdin.close()

    return_code = encoder.wait()
    stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit {return_code}: {stderr.strip()}")
    if preview is not None:
        cv2.imwrite(str(preview_path), preview)
    return output_path, preview_path


def _glob_directories(root: Path, patterns: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_dir():
                result[path.name] = path
    return result


def discover_episode_views(
    approx_root: Path,
    calibrated_root: Path,
    episode_patterns: Sequence[str],
    view_patterns: Sequence[str],
    allow_partial: bool,
) -> list[tuple[str, str, Path, Path]]:
    episodes_a = _glob_directories(approx_root, episode_patterns)
    episodes_b = _glob_directories(calibrated_root, episode_patterns)
    episode_names = _matching_keys(
        episodes_a, episodes_b, "episode", allow_partial
    )
    pairs: list[tuple[str, str, Path, Path]] = []
    for episode in episode_names:
        views_a = _glob_directories(episodes_a[episode], view_patterns)
        views_b = _glob_directories(episodes_b[episode], view_patterns)
        view_names = _matching_keys(
            views_a, views_b, f"view in episode {episode}", allow_partial
        )
        for view in view_names:
            pairs.append((episode, view, views_a[view], views_b[view]))
    return sorted(pairs, key=lambda item: (natural_key(item[0]), natural_key(item[1])))


def _patterns(value: str) -> list[str]:
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result:
        raise ValueError("at least one non-empty glob pattern is required")
    return result


def _nested(report: Mapping[str, object], *keys: str) -> object:
    value: object = report
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _csv_rows(results: Sequence[ViewComparison]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        report = result.report
        rows.append(
            {
                "episode": result.episode,
                "view": result.view,
                "role": report["role"],
                "rgb_frames": report["rgb_frame_pairs"],
                "aligned_reference_frames": report[
                    "aligned_reference_frames_compared"
                ],
                "camera1_frame_offset_applied": _nested(
                    report,
                    "temporal_alignment",
                    "applied_camera1_frame_offset",
                ),
                "temporal_boundary_frames_fail_open": report[
                    "temporal_boundary_frames_fail_open"
                ],
                "approx_focal_px": report["approx_focal_px"],
                "calibrated_focal_px": report["calibrated_focal_px"],
                "hawor_validity_agreement": _nested(
                    report, "hawor", "validity_agreement_rate"
                ),
                "hawor_projected_joint_mean_px": _nested(
                    report, "hawor", "projected_joint_displacement_px", "mean"
                ),
                "hawor_projected_joint_p95_px": _nested(
                    report, "hawor", "projected_joint_displacement_px", "p95"
                ),
                "hawor_projected_vertex_mean_px": _nested(
                    report, "hawor", "projected_vertex_displacement_px", "mean"
                ),
                "hawor_projected_vertex_p95_px": _nested(
                    report, "hawor", "projected_vertex_displacement_px", "p95"
                ),
                "hawor_joint_bbox_iou_mean": _nested(
                    report, "hawor", "projected_joint_bbox_iou", "mean"
                ),
                "haco_probability_mae": _nested(
                    report, "haco", "contact_probability_abs_delta", "mean"
                ),
                "haco_probability_abs_delta_p95": _nested(
                    report, "haco", "contact_probability_abs_delta", "p95"
                ),
                "haco_mask_iou_active_mean": _nested(
                    report,
                    "haco",
                    "contact_mask_iou_when_either_active",
                    "mean",
                ),
                "haco_contact_vertex_flip_rate": _nested(
                    report, "haco", "contact_vertex_flip_rate"
                ),
            }
        )
    return rows


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_reports(
    output_dir: Path,
    approx_root: Path,
    calibrated_root: Path,
    results: Sequence[ViewComparison],
    videos: Sequence[Path],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_hawor = HaWoRAccumulator()
    aggregate_haco = HaCoAccumulator()
    for result in results:
        aggregate_hawor.merge(result.hawor)
        aggregate_haco.merge(result.haco)

    payload = {
        "schema_version": 1,
        "comparison": "approx_26mm_vs_calibrated_scalar_focal",
        "calibration_scope": CALIBRATION_SCOPE,
        "approx_root": str(approx_root.resolve()),
        "calibrated_root": str(calibrated_root.resolve()),
        "evaluated_episode_views": len(results),
        "aggregate": {
            "hawor": aggregate_hawor.to_report(),
            "haco": aggregate_haco.to_report(),
        },
        "episode_views": [result.report for result in results],
        "videos": [str(path.resolve()) for path in videos],
    }
    json_path = output_dir / "calibration_ab_report.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    rows = _csv_rows(results)
    csv_path = output_dir / "calibration_ab_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    markdown_path = output_dir / "calibration_ab_report.md"
    lines = [
        "# 08-05 calibration A/B report",
        "",
        "This report measures the effect of changing the scalar focal length "
        "used by HaWoR and HaCo. It does not exercise principal point, "
        "distortion, stereo extrinsics, rectification, or stereo fusion.",
        "",
        "The values show how much the outputs changed; without ground-truth "
        "hand/contact labels they do not prove that one branch is more accurate.",
        "",
        "For camera_1/SH, each episode manifest offset is applied on the "
        "camera_2/MH reference timeline. Out-of-range boundary cues are "
        "excluded from metrics and rendered as explicit fail-open panels.",
        "",
        "| Episode | View | offset | fx approx -> calibrated | HaWoR valid agree | "
        "joint shift mean / p95 (px) | HaCo prob. MAE | active mask IoU | "
        "mask flip rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {episode} | {view} ({role}) | {offset:+d} | {fa} -> {fb} | {valid} | "
            "{joint_mean} / {joint_p95} | {prob} | {iou} | {flip} |".format(
                episode=row["episode"],
                view=row["view"],
                role=row["role"],
                offset=int(row["camera1_frame_offset_applied"] or 0),
                fa=_fmt(row["approx_focal_px"], 2),
                fb=_fmt(row["calibrated_focal_px"], 2),
                valid=_fmt(row["hawor_validity_agreement"], 4),
                joint_mean=_fmt(row["hawor_projected_joint_mean_px"], 2),
                joint_p95=_fmt(row["hawor_projected_joint_p95_px"], 2),
                prob=_fmt(row["haco_probability_mae"], 4),
                iou=_fmt(row["haco_mask_iou_active_mean"], 4),
                flip=_fmt(row["haco_contact_vertex_flip_rate"], 4),
            )
        )
    aggregate_hawor_report = aggregate_hawor.to_report()
    aggregate_haco_report = aggregate_haco.to_report()
    lines.extend(
        [
            "",
            "Aggregate across all evaluated episodes and cameras:",
            "",
            f"- HaWoR validity agreement: "
            f"`{_fmt(aggregate_hawor_report['validity_agreement_rate'], 4)}`",
            f"- Projected joint displacement mean / p95: "
            f"`{_fmt(_nested(aggregate_hawor_report, 'projected_joint_displacement_px', 'mean'), 4)} / "
            f"{_fmt(_nested(aggregate_hawor_report, 'projected_joint_displacement_px', 'p95'), 4)} px`",
            f"- HaCo probability absolute delta mean / p95: "
            f"`{_fmt(_nested(aggregate_haco_report, 'contact_probability_abs_delta', 'mean'), 5)} / "
            f"{_fmt(_nested(aggregate_haco_report, 'contact_probability_abs_delta', 'p95'), 5)}`",
            f"- Active contact-mask IoU: "
            f"`{_fmt(_nested(aggregate_haco_report, 'contact_mask_iou_when_either_active', 'mean'), 5)}`",
            f"- Contact-vertex flip rate: "
            f"`{_fmt(aggregate_haco_report['contact_vertex_flip_rate'], 5)}`",
        ]
    )
    lines.extend(
        [
            "",
            f"Full machine-readable metrics: `{json_path.name}`",
            f"Flat per-view table: `{csv_path.name}`",
        ]
    )
    if videos:
        lines.extend(["", "Comparison videos:"])
        lines.extend(
            f"- `{path.relative_to(output_dir)}`" for path in videos
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantify approximate-vs-calibrated HaWoR/HaCo outputs"
    )
    parser.add_argument("--approx-root", type=Path, required=True)
    parser.add_argument("--calibrated-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--episodes", default="*",
        help="comma-separated episode globs (default: '*')",
    )
    parser.add_argument(
        "--views", default="camera_*",
        help="comma-separated view globs (default: 'camera_*')",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="compare intersections instead of rejecting missing frames/views",
    )
    parser.add_argument("--make-videos", action="store_true")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-height", type=int, default=360)
    parser.add_argument(
        "--video-max-frames", type=int, default=0,
        help="limit each video for a quick preview; 0 writes all frames",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    approx_root = args.approx_root.expanduser().resolve()
    calibrated_root = args.calibrated_root.expanduser().resolve()
    if not approx_root.is_dir():
        raise FileNotFoundError(f"approx root does not exist: {approx_root}")
    if not calibrated_root.is_dir():
        raise FileNotFoundError(
            f"calibrated root does not exist: {calibrated_root}"
        )
    pairs = discover_episode_views(
        approx_root,
        calibrated_root,
        _patterns(args.episodes),
        _patterns(args.views),
        args.allow_partial,
    )
    results: list[ViewComparison] = []
    video_paths: list[Path] = []
    for index, (episode, view, approx_view, calibrated_view) in enumerate(
        pairs, start=1
    ):
        result = compare_view(
            approx_view,
            calibrated_view,
            episode,
            view,
            args.allow_partial,
        )
        results.append(result)
        print(f"[{index}/{len(pairs)}] compared {episode}/{view}")
        if args.make_videos:
            video_path = (
                args.out_dir.expanduser().resolve()
                / "videos"
                / f"episode_{episode}_{view}_calibration_ab.mp4"
            )
            render_comparison_video(
                approx_view,
                calibrated_view,
                video_path,
                fps=args.fps,
                panel_width=args.panel_width,
                panel_height=args.panel_height,
                max_frames=args.video_max_frames,
                allow_partial=args.allow_partial,
                camera1_frame_offset=int(
                    _nested(
                        result.report,
                        "temporal_alignment",
                        "applied_camera1_frame_offset",
                    )
                    or 0
                ),
            )
            video_paths.append(video_path)
            print(f"    video: {video_path}")

    paths = write_reports(
        args.out_dir.expanduser().resolve(),
        approx_root,
        calibrated_root,
        results,
        video_paths,
    )
    print(f"JSON: {paths[0]}")
    print(f"CSV: {paths[1]}")
    print(f"Markdown: {paths[2]}")
    print("Scope: scalar focal only; full stereo calibration is not evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
