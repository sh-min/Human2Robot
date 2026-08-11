"""Compare HaCo XHand-thickness strategies and conservative derived masks.

The utility consumes completed baseline, half-thickness, full-thickness, and
stereo visibility-force compositor directories.  It never edits those source
directories.  In a sibling staging directory it derives two additional modes:

``baseline_force_union``
    The exact Boolean union of the baseline HaCo and visibility-force masks.

``union_safety_shell_diagnostic``
    The union plus an explicitly diagnostic screen-space shell.  For each
    visibility-selected finger, the modal object is dilated by an adaptive
    projected half-finger thickness.  Only the same semantic finger components
    touching the original force seed are eligible, and additions are capped per
    frame/finger.  This is a 2-D safety proxy, not calibrated metric depth.

All six modes are rendered into a synchronized 3x2 comparison video.  When the
renderer-provided packed XHand surface labels are present, a second 2x2 view
shows the surface map and a cumulative surface-aware comparison:

``surface_front_baseline``
    The existing zero-thickness baseline.  It establishes the unchanged
    palmar/front rule while the not-yet-specialized surfaces retain baseline.

``surface_front_side_half``
    The baseline with lateral/side pixels selected from the half-thickness
    result.

``surface_front_side_half_back_full``
    Palmar/front pixels use baseline, lateral/side pixels use half thickness,
    and dorsal/back pixels use full thickness.

Natural derived videos, masks, surface diagnostics, radius evidence, and
pairwise/per-finger statistics are published atomically after validation.
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
from typing import Any, Sequence

import cv2
import numpy as np

from atomic_directory_publish import publish_directory
from compare_contact_interior_expansion import (
    DEFAULT_FINGER_NAMES,
    FINAL_VIDEO_NAME,
    FPS_TOLERANCE,
    INPUT_REPORT_NAME,
    MASK_NAME,
    VideoMetadata,
    _binary_frame,
    _label_panel,
    _open_writer,
    _paths_overlap,
    _require_file,
    _source_path,
    _track_summary,
    _validate_report_metadata,
    compute_comparison_statistics,
    probe_video,
    validate_video_alignment,
)


FORCE_VIDEO_NAME = "video_overlay_visibility.mp4"
FORCE_MASK_NAME = "occluded_finger_mask_visibility.npy"
FORCE_REPORT_MODE = "visibility"

UNION_MASK_NAME = "occluded_finger_mask_baseline_force_union.npy"
UNION_VIDEO_NAME = "video_overlay_baseline_force_union.mp4"
SHELL_ADDED_MASK_NAME = "diagnostic_safety_shell_added.npy"
SHELL_MASK_NAME = "occluded_finger_mask_union_safety_shell_diagnostic.npy"
SHELL_VIDEO_NAME = "video_overlay_union_safety_shell_diagnostic.mp4"
SHELL_EVIDENCE_NAME = "diagnostic_safety_shell_evidence.npz"
OUTPUT_VIDEO_NAME = "video_compare_xhand_thickness_strategies_3x2.mp4"
OUTPUT_REPORT_NAME = "comparison_report.json"

SURFACE_LABELS_NAME = "robot_finger_surface_labels.npy"
SURFACE_LATERAL_MASK_NAME = (
    "occluded_finger_mask_surface_front_side_half.npy"
)
SURFACE_WEIGHTED_MASK_NAME = (
    "occluded_finger_mask_surface_front_side_half_back_full.npy"
)
SURFACE_LATERAL_VIDEO_NAME = "video_overlay_surface_front_side_half.mp4"
SURFACE_WEIGHTED_VIDEO_NAME = (
    "video_overlay_surface_front_side_half_back_full.mp4"
)
SURFACE_DEBUG_VIDEO_NAME = "video_xhand_surface_labels_debug.mp4"
SURFACE_COMPARISON_VIDEO_NAME = (
    "video_compare_xhand_surface_strategies_2x2.mp4"
)
SURFACE_NAMES = ("palmar_front", "lateral_side", "dorsal_back")
SURFACE_IDS = {name: index + 1 for index, name in enumerate(SURFACE_NAMES)}
SURFACE_STRATEGY_NAMES = (
    "surface_front_baseline",
    "surface_front_side_half",
    "surface_front_side_half_back_full",
)
SURFACE_PAIR_SPECS = (
    (
        "front_baseline_vs_side_half",
        "surface_front_baseline",
        "surface_front_side_half",
    ),
    (
        "side_half_vs_back_full",
        "surface_front_side_half",
        "surface_front_side_half_back_full",
    ),
    (
        "front_baseline_vs_surface_weighted",
        "surface_front_baseline",
        "surface_front_side_half_back_full",
    ),
)

SOURCE_MODE_NAMES = (
    "baseline",
    "half_thickness",
    "full_thickness",
    "visibility_force",
)
DERIVED_MODE_NAMES = (
    "baseline_force_union",
    "union_safety_shell_diagnostic",
)
MODE_NAMES = SOURCE_MODE_NAMES + DERIVED_MODE_NAMES
PAIR_SPECS = (
    ("baseline_vs_half_thickness", "baseline", "half_thickness"),
    ("baseline_vs_full_thickness", "baseline", "full_thickness"),
    ("half_vs_full_thickness", "half_thickness", "full_thickness"),
    ("baseline_vs_visibility_force", "baseline", "visibility_force"),
    ("baseline_vs_union", "baseline", "baseline_force_union"),
    (
        "union_vs_diagnostic_safety_shell",
        "baseline_force_union",
        "union_safety_shell_diagnostic",
    ),
)
EXPECTED_CONTACT_DEPTH_THICKNESS_SCALES = {
    "baseline": 0.0,
    "half_thickness": 0.5,
    "full_thickness": 1.0,
}


@dataclass(frozen=True)
class SafetyShellConfig:
    """Configuration for the opt-in screen-space safety-shell diagnostic."""

    enabled: bool = True
    min_radius_px: int = 3
    max_radius_px: int = 20
    temporal_median_window: int = 5
    radius_quantile: float = 0.85
    added_area_cap_fraction: float = 0.75

    def validate(self) -> None:
        if self.min_radius_px < 0:
            raise ValueError("min_radius_px must be non-negative")
        if self.max_radius_px < self.min_radius_px:
            raise ValueError("max_radius_px must be >= min_radius_px")
        if (
            self.temporal_median_window <= 0
            or self.temporal_median_window % 2 != 1
        ):
            raise ValueError("temporal_median_window must be a positive odd number")
        if not 0.0 <= self.radius_quantile <= 1.0:
            raise ValueError("radius_quantile must be in [0,1]")
        if not np.isfinite(self.added_area_cap_fraction):
            raise ValueError("added_area_cap_fraction must be finite")
        if self.added_area_cap_fraction < 0.0:
            raise ValueError("added_area_cap_fraction must be non-negative")


def _load_report(directory: Path, role: str) -> tuple[Path, dict[str, Any]]:
    path = directory / INPUT_REPORT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{role} report is missing: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse {role} report: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} report must contain a JSON object")
    return path.resolve(), value


def _common_finger_names(
    reports: dict[str, tuple[Path, dict[str, Any]]],
) -> tuple[str, ...]:
    reported: list[tuple[str, tuple[str, ...]]] = []
    for role, (_, report) in reports.items():
        raw = report.get("finger_names")
        if raw is None:
            continue
        if (
            not isinstance(raw, list)
            or not raw
            or not all(isinstance(name, str) and name for name in raw)
            or len(set(raw)) != len(raw)
        ):
            raise ValueError(f"{role} report has invalid finger_names")
        reported.append((role, tuple(raw)))
    if not reported:
        return DEFAULT_FINGER_NAMES
    reference_role, reference = reported[0]
    disagreement = [
        f"{role}={names}"
        for role, names in reported[1:]
        if names != reference
    ]
    if disagreement:
        raise ValueError(
            "source reports disagree on finger_names: "
            f"{reference_role}={reference}; " + "; ".join(disagreement)
        )
    return reference


def _resize_binary(
    raw: np.ndarray,
    width: int,
    height: int,
    role: str,
) -> np.ndarray:
    array = np.asarray(raw)
    if array.ndim != 2:
        raise ValueError(f"{role} frame must be 2-D")
    if array.dtype != np.bool_ and not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{role} contains non-binary values")
    value = array.astype(np.uint8, copy=False)
    if value.shape != (height, width):
        value = cv2.resize(
            value,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    return value.astype(bool, copy=False)


def _resize_labels(
    raw: np.ndarray,
    width: int,
    height: int,
    finger_count: int,
) -> np.ndarray:
    array = np.asarray(raw)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("robot finger-label frames must be 2-D integer arrays")
    if array.size and (int(array.min()) < 0 or int(array.max()) > finger_count):
        raise ValueError(f"robot finger labels must be within [0,{finger_count}]")
    value = array.astype(np.uint8, copy=False)
    if value.shape != (height, width):
        value = cv2.resize(
            value,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)
    return value


def _decode_packed_surface_labels(
    raw: np.ndarray,
    *,
    finger_count: int,
    role: str = "robot finger-surface labels",
) -> tuple[np.ndarray, np.ndarray]:
    """Decode ``3 * (finger_id - 1) + surface_id`` packed labels.

    Zero is reserved for non-finger pixels.  The returned finger and surface
    arrays are uint8 and use zero for the same non-finger pixels.
    """
    packed = np.asarray(raw)
    if packed.ndim != 2 or packed.dtype != np.uint8:
        raise ValueError(f"{role} frames must be 2-D uint8 arrays")
    maximum = 3 * finger_count
    if packed.size and int(packed.max()) > maximum:
        raise ValueError(f"{role} must be within [0,{maximum}]")
    finger = np.zeros_like(packed, dtype=np.uint8)
    surface = np.zeros_like(packed, dtype=np.uint8)
    active = packed > 0
    values = packed[active].astype(np.int16, copy=False) - 1
    finger[active] = (values // 3 + 1).astype(np.uint8)
    surface[active] = (values % 3 + 1).astype(np.uint8)
    return finger, surface


def _resize_surface_labels(
    raw: np.ndarray,
    width: int,
    height: int,
    finger_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize packed labels without interpolating IDs, then decode them."""
    packed = np.asarray(raw)
    # Validate the renderer contract before resizing so a corrupt source is
    # never hidden by nearest-neighbour sampling.
    _decode_packed_surface_labels(packed, finger_count=finger_count)
    if packed.shape != (height, width):
        packed = cv2.resize(
            packed,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)
    finger, surface = _decode_packed_surface_labels(
        packed,
        finger_count=finger_count,
    )
    return packed, finger, surface


def _validate_surface_finger_alignment(
    decoded_finger_labels: np.ndarray,
    finger_labels: np.ndarray,
    *,
    frame_index: int | None = None,
) -> None:
    decoded = np.asarray(decoded_finger_labels)
    fingers = np.asarray(finger_labels)
    if decoded.shape != fingers.shape:
        raise ValueError("surface and finger labels are not spatially aligned")
    if not np.array_equal(decoded, fingers):
        mismatch = int((decoded != fingers).sum())
        where = "" if frame_index is None else f" at frame {frame_index}"
        raise ValueError(
            "packed surface labels do not decode to robot finger labels"
            f"{where}: {mismatch} mismatched pixels"
        )


def build_surface_strategy_frame(
    *,
    baseline_mask: np.ndarray,
    half_thickness_mask: np.ndarray,
    full_thickness_mask: np.ndarray,
    surface_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative side-half and front/side/back-weighted masks.

    This is a selector over already-computed HaCo strategies; it does not
    reinterpret depth.  Front pixels remain at the baseline bias, side pixels
    select the half-thickness result, and back pixels select the full-thickness
    result.  The first cumulative stage leaves back pixels at baseline.
    """
    baseline = np.asarray(baseline_mask, dtype=bool)
    half = np.asarray(half_thickness_mask, dtype=bool)
    full = np.asarray(full_thickness_mask, dtype=bool)
    surfaces = np.asarray(surface_ids)
    if not (
        baseline.shape == half.shape == full.shape == surfaces.shape
    ):
        raise ValueError("surface-strategy inputs are not spatially aligned")
    if surfaces.ndim != 2 or (
        surfaces.size
        and (int(surfaces.min()) < 0 or int(surfaces.max()) > 3)
    ):
        raise ValueError("surface IDs must be a 2-D array within [0,3]")
    lateral = surfaces == SURFACE_IDS["lateral_side"]
    dorsal = surfaces == SURFACE_IDS["dorsal_back"]
    palmar = surfaces == SURFACE_IDS["palmar_front"]
    side_half = (baseline & ~lateral) | (half & lateral)
    weighted = (
        (baseline & palmar)
        | (half & lateral)
        | (full & dorsal)
    )
    return side_half, weighted


def _seed_connected_region(
    support: np.ndarray,
    seed: np.ndarray,
) -> np.ndarray:
    """Keep exactly the 8-connected support components touched by ``seed``."""
    support_mask = np.asarray(support, dtype=bool)
    seed_mask = np.asarray(seed, dtype=bool) & support_mask
    if support_mask.shape != seed_mask.shape:
        raise ValueError("support and seed must have the same shape")
    if not seed_mask.any():
        return np.zeros_like(support_mask)
    _, components = cv2.connectedComponents(
        support_mask.astype(np.uint8),
        connectivity=8,
    )
    touched = np.unique(components[seed_mask])
    touched = touched[touched > 0]
    if not len(touched):
        return np.zeros_like(support_mask)
    return np.isin(components, touched)


def estimate_adaptive_half_thickness_radii(
    force_masks: np.ndarray,
    finger_labels: np.ndarray,
    *,
    width: int,
    height: int,
    finger_count: int,
    quantile: float,
) -> np.ndarray:
    """Estimate projected half-width from each force-seeded finger component."""
    force = np.asanyarray(force_masks)
    labels_array = np.asanyarray(finger_labels)
    if force.ndim != 3 or labels_array.ndim != 3:
        raise ValueError("force mask and finger labels must have shape (T,H,W)")
    if len(force) != len(labels_array):
        raise ValueError("force mask and finger labels must align in time")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0,1]")
    raw = np.full((len(force), finger_count), np.nan, dtype=np.float32)
    for frame_index in range(len(force)):
        frame_force = _binary_frame(force, frame_index, "visibility force")
        labels = _resize_labels(
            labels_array[frame_index], width, height, finger_count
        )
        for finger_index in range(finger_count):
            finger = labels == finger_index + 1
            seed = frame_force & finger
            if not seed.any():
                continue
            connected = _seed_connected_region(finger, seed)
            distances = cv2.distanceTransform(
                connected.astype(np.uint8),
                cv2.DIST_L2,
                5,
            )
            samples = distances[connected]
            if samples.size:
                raw[frame_index, finger_index] = float(
                    np.quantile(samples, quantile)
                )
    return raw


def temporal_median_radii(
    raw_radii: np.ndarray,
    *,
    window: int,
    min_radius_px: int,
    max_radius_px: int,
) -> np.ndarray:
    """Median-smooth within contiguous active runs, then clamp and round."""
    raw = np.asarray(raw_radii, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError("raw radii must have shape (T,F)")
    if window <= 0 or window % 2 != 1:
        raise ValueError("window must be a positive odd number")
    if min_radius_px < 0 or max_radius_px < min_radius_px:
        raise ValueError("invalid radius clamp")
    smoothed = np.zeros(raw.shape, dtype=np.int16)
    half = window // 2
    for finger_index in range(raw.shape[1]):
        active = np.isfinite(raw[:, finger_index])
        start = 0
        while start < len(active):
            if not active[start]:
                start += 1
                continue
            stop = start + 1
            while stop < len(active) and active[stop]:
                stop += 1
            for frame_index in range(start, stop):
                lo = max(start, frame_index - half)
                hi = min(stop, frame_index + half + 1)
                radius = float(np.median(raw[lo:hi, finger_index]))
                smoothed[frame_index, finger_index] = int(
                    np.clip(np.rint(radius), min_radius_px, max_radius_px)
                )
            start = stop
    return smoothed


def build_safety_shell_frame(
    *,
    force_seed: np.ndarray,
    union_mask: np.ndarray,
    finger_labels: np.ndarray,
    object_mask: np.ndarray,
    smoothed_radii: np.ndarray,
    config: SafetyShellConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build one capped shell, retaining only seed-connected finger regions."""
    seed_all = np.asarray(force_seed, dtype=bool)
    union = np.asarray(union_mask, dtype=bool)
    labels = np.asarray(finger_labels)
    obj = np.asarray(object_mask, dtype=bool)
    radii = np.asarray(smoothed_radii)
    if not (seed_all.shape == union.shape == labels.shape == obj.shape):
        raise ValueError("safety-shell frame inputs are not spatially aligned")
    if radii.ndim != 1:
        raise ValueError("smoothed_radii must be a 1-D per-finger vector")
    shell = np.zeros_like(union)
    details: list[dict[str, Any]] = []
    dilated_object_cache: dict[int, np.ndarray] = {}
    outside_distance = cv2.distanceTransform(
        (~obj).astype(np.uint8), cv2.DIST_L2, 5
    )
    for finger_index in range(len(radii)):
        finger = labels == finger_index + 1
        seed = seed_all & finger
        seed_pixels = int(seed.sum())
        radius = int(radii[finger_index])
        detail: dict[str, Any] = {
            "seed_pixels": seed_pixels,
            "radius_px": radius,
            "eligible_pixels": 0,
            "cap_pixels": 0,
            "added_pixels": 0,
            "cap_limited": False,
            "seed_connected": True,
        }
        if not config.enabled or seed_pixels == 0 or radius <= 0:
            details.append(detail)
            continue
        dilated_object = dilated_object_cache.get(radius)
        if dilated_object is None:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * radius + 1, 2 * radius + 1),
            )
            dilated_object = cv2.dilate(
                obj.astype(np.uint8), kernel
            ).astype(bool)
            dilated_object_cache[radius] = dilated_object
        support = finger & dilated_object
        connected = _seed_connected_region(support, seed)
        candidates = connected & ~union
        candidate_pixels = int(candidates.sum())
        cap_pixels = int(
            math.ceil(seed_pixels * config.added_area_cap_fraction)
        )
        detail["eligible_pixels"] = candidate_pixels
        detail["cap_pixels"] = cap_pixels
        if candidate_pixels == 0 or cap_pixels == 0:
            detail["cap_limited"] = candidate_pixels > cap_pixels
            details.append(detail)
            continue

        accepted = candidates
        if candidate_pixels > cap_pixels:
            flat = np.flatnonzero(candidates)
            distance_to_seed = cv2.distanceTransform(
                (~seed).astype(np.uint8), cv2.DIST_L2, 5
            ).ravel()[flat]
            distance_to_object = outside_distance.ravel()[flat]
            order = np.lexsort((flat, distance_to_seed, distance_to_object))
            selected = flat[order[:cap_pixels]]
            accepted = np.zeros_like(candidates)
            accepted.ravel()[selected] = True
            detail["cap_limited"] = True
        shell |= accepted
        detail["added_pixels"] = int(accepted.sum())
        detail["seed_connected"] = bool(np.all(connected[accepted]))
        details.append(detail)
    return shell, details


def _resolve_report_video_source(
    explicit: Path | None,
    key: str,
    reports: dict[str, tuple[Path, dict[str, Any]]],
    *,
    required: bool,
    require_all_reports: bool = False,
) -> tuple[Path | None, list[str]]:
    candidates: list[Path] = []
    missing: list[str] = []
    missing_roles: list[str] = []
    for role, (report_path, report) in reports.items():
        candidate = _source_path(report, report_path, key)
        if candidate is None:
            missing_roles.append(role)
            continue
        if candidate.is_file():
            candidates.append(candidate.resolve())
        else:
            missing.append(str(candidate))
    unique = list(dict.fromkeys(candidates))
    if len(unique) > 1:
        raise ValueError(
            f"source reports disagree on {key}: "
            + ", ".join(str(path) for path in unique)
        )
    if require_all_reports and missing_roles:
        raise ValueError(
            f"source reports are missing sources.{key}: "
            + ", ".join(missing_roles)
        )
    if explicit is not None:
        selected = _require_file(explicit, key.replace("_", " "))
    else:
        selected = unique[0] if unique else None
    if required and selected is None:
        raise FileNotFoundError(
            f"could not infer {key}; pass --{key} explicitly"
        )
    return selected, missing


def _validate_source_report_counts(
    role: str,
    report: dict[str, Any],
    mask: np.ndarray,
) -> list[str]:
    per_frame = np.zeros(len(mask), dtype=np.int64)
    for frame_index in range(len(mask)):
        per_frame[frame_index] = int(
            _binary_frame(mask, frame_index, role).sum()
        )
    total = int(per_frame.sum())
    frames = int((per_frame > 0).sum())
    if role == "visibility_force":
        mode_statistics = report.get("mode_statistics")
        if not isinstance(mode_statistics, dict):
            raise ValueError("visibility-force report is missing mode_statistics")
        values = mode_statistics.get(FORCE_REPORT_MODE)
        if not isinstance(values, dict):
            raise ValueError(
                "visibility-force report is missing mode_statistics.visibility"
            )
        fields = values
        expected = (("pixels", total), ("frames", frames))
    else:
        fields = report
        expected = (
            ("occluded_pixels_total", total),
            ("frames_with_occlusion", frames),
        )
    checked: list[str] = []
    for key, value in expected:
        if key not in fields:
            continue
        try:
            actual = int(fields[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{role} report field {key} is invalid") from exc
        if actual != value:
            raise ValueError(
                f"{role} report/mask mismatch for {key}: {actual} != {value}"
            )
        checked.append(key)
    track_keys = (
        ("occluded_pixel_count",)
        if role != "visibility_force"
        else ("occluded_pixel_count", "pixel_count", "pixels_per_frame")
    )
    for key in track_keys:
        if key not in fields:
            continue
        raw = fields[key]
        if not isinstance(raw, list):
            raise ValueError(f"{role} report field {key} is not a list")
        try:
            actual_track = [int(value) for value in raw]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{role} report field {key} is invalid") from exc
        if actual_track != per_frame.tolist():
            raise ValueError(f"{role} report/mask mismatch for {key}")
        checked.append(key)
    return checked


def _validate_thickness_report_contract(
    role: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Verify that each named HaCo input carries the expected bias scale.

    The reviewed zero-bias pilot predates the thickness fields, so a missing
    scale is accepted only for ``baseline`` and is interpreted as the legacy
    zero-bias gate.  Half/full inputs must carry the new explicit report block.
    """
    if role not in EXPECTED_CONTACT_DEPTH_THICKNESS_SCALES:
        raise ValueError(f"no XHand thickness contract for role {role!r}")
    expected = EXPECTED_CONTACT_DEPTH_THICKNESS_SCALES[role]
    config = report.get("config")
    config_scale = config.get("contact_depth_thickness_scale") if isinstance(
        config, dict
    ) else None
    bias = report.get("xhand_contact_depth_bias")
    bias_scale = bias.get("scale") if isinstance(bias, dict) else None

    if role == "baseline" and config_scale is None and bias_scale is None:
        return {
            "expected_scale": expected,
            "actual_scale": 0.0,
            "legacy_missing_fields_interpreted_as_zero": True,
        }
    if config_scale is None or bias_scale is None:
        raise ValueError(
            f"{role} report is missing the XHand thickness scale contract"
        )
    try:
        actual_config = float(config_scale)
        actual_bias = float(bias_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{role} report has an invalid thickness scale") from exc
    if not np.isclose(actual_config, expected) or not np.isclose(
        actual_bias, expected
    ):
        raise ValueError(
            f"{role} XHand thickness scale {actual_config}/{actual_bias} "
            f"!= expected {expected}"
        )
    expected_enabled = expected > 0.0
    if bool(bias.get("enabled")) != expected_enabled:
        raise ValueError(f"{role} XHand thickness enabled flag is inconsistent")
    if bias.get("metric_object_depth_gate_modified") is not False:
        raise ValueError(f"{role} thickness bias modified the metric depth gate")
    return {
        "expected_scale": expected,
        "actual_scale": actual_config,
        "legacy_missing_fields_interpreted_as_zero": False,
    }


def _mode_statistics(
    masks: dict[str, np.ndarray],
    labels_array: np.ndarray,
    *,
    width: int,
    height: int,
    finger_names: Sequence[str],
) -> tuple[dict[str, Any], dict[str, int]]:
    frames = len(next(iter(masks.values())))
    tracks = {
        mode: np.zeros(frames, dtype=np.int64) for mode in masks
    }
    finger_tracks = {
        mode: np.zeros((frames, len(finger_names)), dtype=np.int64)
        for mode in masks
    }
    outside = {mode: 0 for mode in masks}
    for frame_index in range(frames):
        labels = _resize_labels(
            labels_array[frame_index], width, height, len(finger_names)
        )
        rendered_fingers = labels > 0
        for mode, mask_array in masks.items():
            mask = _binary_frame(mask_array, frame_index, mode)
            tracks[mode][frame_index] = int(mask.sum())
            outside[mode] += int((mask & ~rendered_fingers).sum())
            for finger_index in range(len(finger_names)):
                finger_tracks[mode][frame_index, finger_index] = int(
                    (mask & (labels == finger_index + 1)).sum()
                )
    statistics: dict[str, Any] = {}
    for mode in masks:
        summary = _track_summary(tracks[mode])
        summary["per_finger"] = {
            finger: _track_summary(finger_tracks[mode][:, finger_index])
            for finger_index, finger in enumerate(finger_names)
        }
        statistics[mode] = summary
    return statistics, outside


def _surface_statistics(
    masks: dict[str, np.ndarray],
    surface_labels_array: np.ndarray,
    finger_labels_array: np.ndarray,
    *,
    width: int,
    height: int,
    finger_names: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Summarize rendered and occluded pixels for each anatomical surface."""
    frames = len(next(iter(masks.values())))
    rendered_tracks = {
        name: np.zeros(frames, dtype=np.int64) for name in SURFACE_NAMES
    }
    mode_tracks = {
        mode: {
            name: np.zeros(frames, dtype=np.int64) for name in SURFACE_NAMES
        }
        for mode in masks
    }
    rendered_by_finger = {
        finger: {name: 0 for name in SURFACE_NAMES}
        for finger in finger_names
    }
    for frame_index in range(frames):
        fingers = _resize_labels(
            finger_labels_array[frame_index],
            width,
            height,
            len(finger_names),
        )
        _, decoded_fingers, surfaces = _resize_surface_labels(
            surface_labels_array[frame_index],
            width,
            height,
            len(finger_names),
        )
        _validate_surface_finger_alignment(
            decoded_fingers,
            fingers,
            frame_index=frame_index,
        )
        frame_masks = {
            mode: _binary_frame(mask, frame_index, mode)
            for mode, mask in masks.items()
        }
        for surface_index, surface_name in enumerate(SURFACE_NAMES, start=1):
            selector = surfaces == surface_index
            rendered_tracks[surface_name][frame_index] = int(selector.sum())
            for mode, mask in frame_masks.items():
                mode_tracks[mode][surface_name][frame_index] = int(
                    (mask & selector).sum()
                )
            for finger_index, finger_name in enumerate(finger_names, start=1):
                rendered_by_finger[finger_name][surface_name] += int(
                    (selector & (fingers == finger_index)).sum()
                )
    rendered_summary = {
        "packing": "3 * (finger_id - 1) + surface_id",
        "surface_ids": dict(SURFACE_IDS),
        "per_surface": {
            name: _track_summary(track)
            for name, track in rendered_tracks.items()
        },
        "per_finger_pixels": rendered_by_finger,
    }
    strategy_summary = {
        mode: {
            "per_surface": {
                name: _track_summary(track)
                for name, track in surface_tracks.items()
            }
        }
        for mode, surface_tracks in mode_tracks.items()
    }
    return rendered_summary, strategy_summary


def _resize_video_frame(
    frame: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _render_outputs(
    staging: Path,
    *,
    source_videos: dict[str, Path],
    background_video: Path,
    raw_video: Path | None,
    object_restore_masks: np.ndarray,
    surface_labels: np.ndarray,
    union_masks: np.ndarray,
    shell_added_masks: np.ndarray,
    shell_masks: np.ndarray,
    surface_lateral_masks: np.ndarray,
    surface_weighted_masks: np.ndarray,
    smoothed_radii: np.ndarray,
    shell_added_counts: np.ndarray,
    reference: VideoMetadata,
    finger_count: int,
) -> None:
    captures = {
        mode: cv2.VideoCapture(str(source_videos[mode]))
        for mode in SOURCE_MODE_NAMES
    }
    captures["background"] = cv2.VideoCapture(str(background_video))
    if raw_video is not None and raw_video != background_video:
        captures["raw"] = cv2.VideoCapture(str(raw_video))
    if not all(capture.isOpened() for capture in captures.values()):
        for capture in captures.values():
            capture.release()
        raise RuntimeError("could not open all source video streams")

    union_writer = _open_writer(
        staging / UNION_VIDEO_NAME,
        reference.fps,
        (reference.width, reference.height),
    )
    shell_writer = _open_writer(
        staging / SHELL_VIDEO_NAME,
        reference.fps,
        (reference.width, reference.height),
    )
    comparison_writer = _open_writer(
        staging / OUTPUT_VIDEO_NAME,
        reference.fps,
        (reference.width * 3, reference.height * 2),
    )
    surface_lateral_writer = _open_writer(
        staging / SURFACE_LATERAL_VIDEO_NAME,
        reference.fps,
        (reference.width, reference.height),
    )
    surface_weighted_writer = _open_writer(
        staging / SURFACE_WEIGHTED_VIDEO_NAME,
        reference.fps,
        (reference.width, reference.height),
    )
    surface_debug_writer = _open_writer(
        staging / SURFACE_DEBUG_VIDEO_NAME,
        reference.fps,
        (reference.width, reference.height),
    )
    surface_comparison_writer = _open_writer(
        staging / SURFACE_COMPARISON_VIDEO_NAME,
        reference.fps,
        (reference.width * 2, reference.height * 2),
    )
    try:
        for frame_index in range(reference.frames):
            decoded: dict[str, np.ndarray] = {}
            for name, capture in captures.items():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"{name} video read failed at frame {frame_index}"
                    )
                decoded[name] = _resize_video_frame(
                    frame, reference.width, reference.height
                )
            restore = decoded["background"].copy()
            object_restore_mask = _resize_binary(
                object_restore_masks[frame_index],
                reference.width,
                reference.height,
                "object restore mask",
            )
            if "raw" in decoded:
                restore[object_restore_mask] = decoded["raw"][
                    object_restore_mask
                ]

            union_mask = _binary_frame(
                union_masks, frame_index, "baseline-force union"
            )
            shell_mask = _binary_frame(
                shell_masks, frame_index, "diagnostic safety shell"
            )
            shell_added = _binary_frame(
                shell_added_masks, frame_index, "diagnostic shell addition"
            )
            union_frame = decoded["baseline"].copy()
            union_frame[union_mask] = restore[union_mask]
            shell_frame = decoded["baseline"].copy()
            shell_frame[shell_mask] = restore[shell_mask]
            union_writer.write(union_frame)
            shell_writer.write(shell_frame)

            shell_panel = shell_frame.copy()
            if shell_added.any():
                shell_panel[shell_added] = np.clip(
                    0.35 * shell_panel[shell_added].astype(np.float32)
                    + 0.65 * np.array([255, 0, 255], dtype=np.float32),
                    0,
                    255,
                ).astype(np.uint8)
            active_radii = smoothed_radii[frame_index]
            active_radii = active_radii[active_radii > 0]
            radius_detail = (
                f"r={int(active_radii.min())}-{int(active_radii.max())}px"
                if len(active_radii)
                else "r=inactive"
            )
            panels = (
                _label_panel(
                    decoded["baseline"], "Baseline HaCo (0 mm)", frame_index
                ),
                _label_panel(
                    decoded["half_thickness"],
                    "HaCo + XHand half thickness",
                    frame_index,
                ),
                _label_panel(
                    decoded["full_thickness"],
                    "HaCo + XHand full thickness",
                    frame_index,
                ),
                _label_panel(
                    decoded["visibility_force"],
                    "SH/MH visibility force",
                    frame_index,
                ),
                _label_panel(
                    union_frame,
                    "Baseline OR visibility force",
                    frame_index,
                ),
                _label_panel(
                    shell_panel,
                    "DIAGNOSTIC: union + 2-D shell",
                    frame_index,
                    f"magenta shell {int(shell_added_counts[frame_index])}px | "
                    + radius_detail,
                ),
            )
            top = np.concatenate(panels[:3], axis=1)
            bottom = np.concatenate(panels[3:], axis=1)
            comparison_writer.write(np.concatenate((top, bottom), axis=0))

            _, _, surfaces = _resize_surface_labels(
                surface_labels[frame_index],
                reference.width,
                reference.height,
                finger_count,
            )
            lateral_selector = surfaces == SURFACE_IDS["lateral_side"]
            dorsal_selector = surfaces == SURFACE_IDS["dorsal_back"]
            surface_lateral_frame = decoded["baseline"].copy()
            surface_lateral_frame[lateral_selector] = decoded[
                "half_thickness"
            ][lateral_selector]
            surface_weighted_frame = surface_lateral_frame.copy()
            surface_weighted_frame[dorsal_selector] = decoded[
                "full_thickness"
            ][dorsal_selector]
            surface_lateral_writer.write(surface_lateral_frame)
            surface_weighted_writer.write(surface_weighted_frame)

            surface_debug = np.clip(
                decoded["baseline"].astype(np.float32) * 0.25,
                0,
                255,
            ).astype(np.uint8)
            # OpenCV writers consume BGR: red=palmar/front,
            # yellow=lateral/side, blue=dorsal/back.
            surface_debug[surfaces == SURFACE_IDS["palmar_front"]] = (
                0,
                0,
                255,
            )
            surface_debug[lateral_selector] = (0, 255, 255)
            surface_debug[dorsal_selector] = (255, 0, 0)
            surface_debug_writer.write(surface_debug)

            lateral_count = int(
                _binary_frame(
                    surface_lateral_masks,
                    frame_index,
                    "surface side-half",
                ).sum()
            )
            weighted_count = int(
                _binary_frame(
                    surface_weighted_masks,
                    frame_index,
                    "surface weighted",
                ).sum()
            )
            surface_panels = (
                _label_panel(
                    surface_debug,
                    "XHand surface labels",
                    frame_index,
                    "red=front | yellow=side | blue=back",
                ),
                _label_panel(
                    decoded["baseline"],
                    "Front unchanged: baseline",
                    frame_index,
                    "0x contact-depth thickness",
                ),
                _label_panel(
                    surface_lateral_frame,
                    "+ Side half thickness",
                    frame_index,
                    f"selected mask {lateral_count}px",
                ),
                _label_panel(
                    surface_weighted_frame,
                    "+ Back full thickness",
                    frame_index,
                    f"selected mask {weighted_count}px",
                ),
            )
            surface_top = np.concatenate(surface_panels[:2], axis=1)
            surface_bottom = np.concatenate(surface_panels[2:], axis=1)
            surface_comparison_writer.write(
                np.concatenate((surface_top, surface_bottom), axis=0)
            )

        for name, capture in captures.items():
            ok, _ = capture.read()
            if ok:
                raise RuntimeError(
                    f"{name} video contains more than {reference.frames} frames"
                )
    finally:
        for capture in captures.values():
            capture.release()
        union_writer.release()
        shell_writer.release()
        comparison_writer.release()
        surface_lateral_writer.release()
        surface_weighted_writer.release()
        surface_debug_writer.release()
        surface_comparison_writer.release()


def build_comparison(
    baseline_dir: Path,
    half_thickness_dir: Path,
    full_thickness_dir: Path,
    force_dir: Path,
    *,
    overlay_dir: Path,
    object_mask: Path,
    object_restore_mask: Path | None = None,
    output_dir: Path,
    background: Path | None = None,
    raw_video: Path | None = None,
    surface_labels: Path | None = None,
    overwrite: bool = False,
    shell_config: SafetyShellConfig = SafetyShellConfig(),
) -> dict[str, Any]:
    """Validate sources, derive union/shell modes, and atomically publish."""
    shell_config.validate()
    source_dirs = {
        "baseline": Path(baseline_dir).expanduser().resolve(),
        "half_thickness": Path(half_thickness_dir).expanduser().resolve(),
        "full_thickness": Path(full_thickness_dir).expanduser().resolve(),
        "visibility_force": Path(force_dir).expanduser().resolve(),
    }
    for role, directory in source_dirs.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"{role} directory is missing: {directory}")
    if len(set(source_dirs.values())) != len(source_dirs):
        raise ValueError("all source directories must be distinct")
    overlay_dir = Path(overlay_dir).expanduser().resolve()
    if not overlay_dir.is_dir():
        raise FileNotFoundError(f"overlay directory is missing: {overlay_dir}")
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.is_symlink() or (
        output_dir.exists() and not output_dir.is_dir()
    ):
        raise ValueError(f"invalid comparison output path: {output_dir}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to replace existing comparison output without "
            f"--overwrite: {output_dir}"
        )
    for directory in (*source_dirs.values(), overlay_dir):
        if _paths_overlap(output_dir, directory):
            raise ValueError(
                "output directory must not overlap an input directory: "
                f"{directory}"
            )

    reports = {
        role: _load_report(directory, role)
        for role, directory in source_dirs.items()
    }
    finger_names = _common_finger_names(reports)
    source_videos = {
        "baseline": _require_file(
            source_dirs["baseline"] / FINAL_VIDEO_NAME, "baseline video"
        ),
        "half_thickness": _require_file(
            source_dirs["half_thickness"] / FINAL_VIDEO_NAME,
            "half-thickness video",
        ),
        "full_thickness": _require_file(
            source_dirs["full_thickness"] / FINAL_VIDEO_NAME,
            "full-thickness video",
        ),
        "visibility_force": _require_file(
            source_dirs["visibility_force"] / FORCE_VIDEO_NAME,
            "visibility-force video",
        ),
    }
    metadata = {
        role: probe_video(path) for role, path in source_videos.items()
    }
    reference = validate_video_alignment(metadata)
    report_metadata = {
        role: _validate_report_metadata(report, reference, role)
        for role, (_, report) in reports.items()
    }

    source_mask_paths = {
        "baseline": _require_file(
            source_dirs["baseline"] / MASK_NAME, "baseline mask"
        ),
        "half_thickness": _require_file(
            source_dirs["half_thickness"] / MASK_NAME,
            "half-thickness mask",
        ),
        "full_thickness": _require_file(
            source_dirs["full_thickness"] / MASK_NAME,
            "full-thickness mask",
        ),
        "visibility_force": _require_file(
            source_dirs["visibility_force"] / FORCE_MASK_NAME,
            "visibility-force mask",
        ),
    }
    source_masks = {
        role: np.load(path, mmap_mode="r")
        for role, path in source_mask_paths.items()
    }
    expected_shape = (reference.frames, reference.height, reference.width)
    for role, mask in source_masks.items():
        if mask.shape != expected_shape:
            raise ValueError(f"{role} mask shape {mask.shape} != {expected_shape}")

    labels_path = _require_file(
        overlay_dir / "robot_finger_labels.npy", "robot finger labels"
    )
    labels_array = np.load(labels_path, mmap_mode="r")
    if labels_array.ndim != 3 or len(labels_array) != reference.frames:
        raise ValueError("robot finger labels must align with source videos")
    surface_labels_path = _require_file(
        (
            Path(surface_labels).expanduser().resolve()
            if surface_labels is not None
            else overlay_dir / SURFACE_LABELS_NAME
        ),
        "robot finger-surface labels",
    )
    if _paths_overlap(output_dir, surface_labels_path):
        raise ValueError(
            "output directory must not overlap the surface-label input: "
            f"{surface_labels_path}"
        )
    surface_labels_array = np.load(surface_labels_path, mmap_mode="r")
    if surface_labels_array.dtype != np.uint8:
        raise ValueError("robot finger-surface labels must use uint8")
    if surface_labels_array.shape != labels_array.shape:
        raise ValueError(
            "robot finger-surface labels must exactly align with robot finger "
            f"labels: {surface_labels_array.shape} != {labels_array.shape}"
        )
    object_mask_path = _require_file(object_mask, "modal object mask")
    object_masks = np.load(object_mask_path, mmap_mode="r")
    if object_masks.ndim != 3 or len(object_masks) != reference.frames:
        raise ValueError("modal object mask must align with source videos")
    object_restore_mask_path = _require_file(
        (
            Path(object_restore_mask).expanduser().resolve()
            if object_restore_mask is not None
            else object_mask_path
        ),
        "clean observed-object restore mask",
    )
    object_restore_masks = np.load(
        object_restore_mask_path, mmap_mode="r"
    )
    if object_restore_masks.shape != object_masks.shape:
        raise ValueError(
            "object restore mask must exactly align with the modal object "
            f"mask: {object_restore_masks.shape} != {object_masks.shape}"
        )
    for frame_index in range(reference.frames):
        restore = np.asarray(
            object_restore_masks[frame_index], dtype=bool
        )
        modal = np.asarray(object_masks[frame_index], dtype=bool)
        if np.any(restore & ~modal):
            raise ValueError(
                "object restore mask is not a modal-object subset at frame "
                f"{frame_index}"
            )

    background_path, missing_background = _resolve_report_video_source(
        background,
        "background",
        reports,
        required=True,
        require_all_reports=True,
    )
    assert background_path is not None
    raw_video_path, missing_raw_video = _resolve_report_video_source(
        raw_video,
        "raw_video",
        reports,
        required=False,
    )
    auxiliary_metadata = {"background": probe_video(background_path)}
    if raw_video_path is not None and raw_video_path != background_path:
        auxiliary_metadata["raw_video"] = probe_video(raw_video_path)
    validate_video_alignment({"reference": reference, **auxiliary_metadata})

    checked_report_counts = {
        role: _validate_source_report_counts(
            role, reports[role][1], source_masks[role]
        )
        for role in SOURCE_MODE_NAMES
    }
    thickness_contract = {
        role: _validate_thickness_report_contract(role, reports[role][1])
        for role in EXPECTED_CONTACT_DEPTH_THICKNESS_SCALES
    }

    raw_radii = estimate_adaptive_half_thickness_radii(
        source_masks["visibility_force"],
        labels_array,
        width=reference.width,
        height=reference.height,
        finger_count=len(finger_names),
        quantile=shell_config.radius_quantile,
    )
    smoothed_radii = temporal_median_radii(
        raw_radii,
        window=shell_config.temporal_median_window,
        min_radius_px=shell_config.min_radius_px,
        max_radius_px=shell_config.max_radius_px,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".xhand_thickness_strategies.", dir=output_dir.parent
        )
    )
    cleanup = lambda: shutil.rmtree(staging, ignore_errors=True)
    atexit.register(cleanup)
    try:
        union_masks = np.lib.format.open_memmap(
            staging / UNION_MASK_NAME,
            mode="w+",
            dtype=bool,
            shape=expected_shape,
        )
        shell_added_masks = np.lib.format.open_memmap(
            staging / SHELL_ADDED_MASK_NAME,
            mode="w+",
            dtype=bool,
            shape=expected_shape,
        )
        shell_masks = np.lib.format.open_memmap(
            staging / SHELL_MASK_NAME,
            mode="w+",
            dtype=bool,
            shape=expected_shape,
        )
        surface_lateral_masks = np.lib.format.open_memmap(
            staging / SURFACE_LATERAL_MASK_NAME,
            mode="w+",
            dtype=bool,
            shape=expected_shape,
        )
        surface_weighted_masks = np.lib.format.open_memmap(
            staging / SURFACE_WEIGHTED_MASK_NAME,
            mode="w+",
            dtype=bool,
            shape=expected_shape,
        )
        shell_added_by_finger = np.zeros(
            (reference.frames, len(finger_names)), dtype=np.int64
        )
        shell_seed_by_finger = np.zeros_like(shell_added_by_finger)
        shell_cap_by_finger = np.zeros_like(shell_added_by_finger)
        shell_cap_limited = np.zeros_like(shell_added_by_finger, dtype=bool)
        seed_connected_violations = 0
        semantic_violations = 0
        cap_violations = 0
        force_outside_object = 0
        union_definition_violations = 0
        shell_definition_violations = 0
        surface_lateral_definition_violations = 0
        surface_weighted_definition_violations = 0

        for frame_index in range(reference.frames):
            baseline = _binary_frame(
                source_masks["baseline"], frame_index, "baseline"
            )
            half_thickness = _binary_frame(
                source_masks["half_thickness"],
                frame_index,
                "half thickness",
            )
            full_thickness = _binary_frame(
                source_masks["full_thickness"],
                frame_index,
                "full thickness",
            )
            force = _binary_frame(
                source_masks["visibility_force"],
                frame_index,
                "visibility force",
            )
            labels = _resize_labels(
                labels_array[frame_index],
                reference.width,
                reference.height,
                len(finger_names),
            )
            _, surface_fingers, surfaces = _resize_surface_labels(
                surface_labels_array[frame_index],
                reference.width,
                reference.height,
                len(finger_names),
            )
            _validate_surface_finger_alignment(
                surface_fingers,
                labels,
                frame_index=frame_index,
            )
            obj = _resize_binary(
                object_masks[frame_index],
                reference.width,
                reference.height,
                "object mask",
            )
            force_outside_object += int((force & ~obj).sum())
            union = baseline | force
            shell_added, details = build_safety_shell_frame(
                force_seed=force,
                union_mask=union,
                finger_labels=labels,
                object_mask=obj,
                smoothed_radii=smoothed_radii[frame_index],
                config=shell_config,
            )
            final_shell = union | shell_added
            union_masks[frame_index] = union
            shell_added_masks[frame_index] = shell_added
            shell_masks[frame_index] = final_shell
            surface_lateral, surface_weighted = build_surface_strategy_frame(
                baseline_mask=baseline,
                half_thickness_mask=half_thickness,
                full_thickness_mask=full_thickness,
                surface_ids=surfaces,
            )
            surface_lateral_masks[frame_index] = surface_lateral
            surface_weighted_masks[frame_index] = surface_weighted
            semantic_violations += int((shell_added & ~(labels > 0)).sum())
            union_definition_violations += int(
                np.logical_xor(union, baseline | force).sum()
            )
            shell_definition_violations += int(
                np.logical_xor(final_shell, union | shell_added).sum()
            )
            expected_surface_lateral, expected_surface_weighted = (
                build_surface_strategy_frame(
                    baseline_mask=baseline,
                    half_thickness_mask=half_thickness,
                    full_thickness_mask=full_thickness,
                    surface_ids=surfaces,
                )
            )
            surface_lateral_definition_violations += int(
                np.logical_xor(
                    surface_lateral,
                    expected_surface_lateral,
                ).sum()
            )
            surface_weighted_definition_violations += int(
                np.logical_xor(
                    surface_weighted,
                    expected_surface_weighted,
                ).sum()
            )
            for finger_index, detail in enumerate(details):
                shell_seed_by_finger[frame_index, finger_index] = int(
                    detail["seed_pixels"]
                )
                shell_added_by_finger[frame_index, finger_index] = int(
                    detail["added_pixels"]
                )
                shell_cap_by_finger[frame_index, finger_index] = int(
                    detail["cap_pixels"]
                )
                shell_cap_limited[frame_index, finger_index] = bool(
                    detail["cap_limited"]
                )
                if not detail["seed_connected"]:
                    seed_connected_violations += 1
                if int(detail["added_pixels"]) > int(detail["cap_pixels"]):
                    cap_violations += 1
        union_masks.flush()
        shell_added_masks.flush()
        shell_masks.flush()
        surface_lateral_masks.flush()
        surface_weighted_masks.flush()

        all_masks: dict[str, np.ndarray] = {
            **source_masks,
            "baseline_force_union": union_masks,
            "union_safety_shell_diagnostic": shell_masks,
        }
        surface_strategy_masks: dict[str, np.ndarray] = {
            "surface_front_baseline": source_masks["baseline"],
            "surface_front_side_half": surface_lateral_masks,
            "surface_front_side_half_back_full": surface_weighted_masks,
        }
        mode_statistics, outside_finger_pixels = _mode_statistics(
            all_masks,
            labels_array,
            width=reference.width,
            height=reference.height,
            finger_names=finger_names,
        )
        surface_mode_statistics, surface_outside_finger_pixels = (
            _mode_statistics(
                surface_strategy_masks,
                labels_array,
                width=reference.width,
                height=reference.height,
                finger_names=finger_names,
            )
        )
        if any(outside_finger_pixels.values()) or any(
            surface_outside_finger_pixels.values()
        ):
            raise ValueError(
                "an input or derived occlusion mask contains non-finger pixels: "
                + json.dumps(
                    {
                        **outside_finger_pixels,
                        **surface_outside_finger_pixels,
                    },
                    sort_keys=True,
                )
            )
        if (
            seed_connected_violations
            or semantic_violations
            or cap_violations
            or union_definition_violations
            or shell_definition_violations
            or surface_lateral_definition_violations
            or surface_weighted_definition_violations
        ):
            raise RuntimeError("derived mask invariant violation")
        if force_outside_object:
            raise ValueError(
                "visibility-force seed contains pixels outside the modal object: "
                f"{force_outside_object}"
            )

        pairwise: dict[str, Any] = {}
        for pair_name, first, second in PAIR_SPECS:
            pairwise[pair_name] = {
                "first": first,
                "second": second,
                "statistics": compute_comparison_statistics(
                    all_masks[first],
                    all_masks[second],
                    finger_labels=labels_array,
                    finger_names=finger_names,
                ),
            }

        surface_pairwise: dict[str, Any] = {}
        for pair_name, first, second in SURFACE_PAIR_SPECS:
            surface_pairwise[pair_name] = {
                "first": first,
                "second": second,
                "statistics": compute_comparison_statistics(
                    surface_strategy_masks[first],
                    surface_strategy_masks[second],
                    finger_labels=labels_array,
                    finger_names=finger_names,
                ),
            }
        rendered_surface_statistics, surface_breakdown = _surface_statistics(
            surface_strategy_masks,
            surface_labels_array,
            labels_array,
            width=reference.width,
            height=reference.height,
            finger_names=finger_names,
        )

        shell_added_counts = shell_added_by_finger.sum(axis=1)
        _render_outputs(
            staging,
            source_videos=source_videos,
            background_video=background_path,
            raw_video=raw_video_path,
            object_restore_masks=object_restore_masks,
            surface_labels=surface_labels_array,
            union_masks=union_masks,
            shell_added_masks=shell_added_masks,
            shell_masks=shell_masks,
            surface_lateral_masks=surface_lateral_masks,
            surface_weighted_masks=surface_weighted_masks,
            smoothed_radii=smoothed_radii,
            shell_added_counts=shell_added_counts,
            reference=reference,
            finger_count=len(finger_names),
        )
        output_metadata = {
            "union": probe_video(staging / UNION_VIDEO_NAME),
            "diagnostic_shell": probe_video(staging / SHELL_VIDEO_NAME),
            "comparison": probe_video(staging / OUTPUT_VIDEO_NAME),
            "surface_lateral": probe_video(
                staging / SURFACE_LATERAL_VIDEO_NAME
            ),
            "surface_weighted": probe_video(
                staging / SURFACE_WEIGHTED_VIDEO_NAME
            ),
            "surface_debug": probe_video(staging / SURFACE_DEBUG_VIDEO_NAME),
            "surface_comparison": probe_video(
                staging / SURFACE_COMPARISON_VIDEO_NAME
            ),
        }
        for role in (
            "union",
            "diagnostic_shell",
            "surface_lateral",
            "surface_weighted",
            "surface_debug",
        ):
            value = output_metadata[role]
            if (
                (value.width, value.height, value.frames)
                != (reference.width, reference.height, reference.frames)
                or not np.isclose(value.fps, reference.fps, atol=FPS_TOLERANCE)
            ):
                raise RuntimeError(f"{role} derived video failed validation")
        comparison_metadata = output_metadata["comparison"]
        if (
            (
                comparison_metadata.width,
                comparison_metadata.height,
                comparison_metadata.frames,
            )
            != (
                reference.width * 3,
                reference.height * 2,
                reference.frames,
            )
            or not np.isclose(
                comparison_metadata.fps,
                reference.fps,
                atol=FPS_TOLERANCE,
            )
        ):
            raise RuntimeError("3x2 comparison video failed validation")
        surface_comparison_metadata = output_metadata["surface_comparison"]
        if (
            (
                surface_comparison_metadata.width,
                surface_comparison_metadata.height,
                surface_comparison_metadata.frames,
            )
            != (
                reference.width * 2,
                reference.height * 2,
                reference.frames,
            )
            or not np.isclose(
                surface_comparison_metadata.fps,
                reference.fps,
                atol=FPS_TOLERANCE,
            )
        ):
            raise RuntimeError("2x2 surface comparison video failed validation")

        np.savez_compressed(
            staging / SHELL_EVIDENCE_NAME,
            raw_radius_px=raw_radii,
            smoothed_radius_px=smoothed_radii,
            seed_pixels_per_finger=shell_seed_by_finger,
            added_pixels_per_finger=shell_added_by_finger,
            cap_pixels_per_finger=shell_cap_by_finger,
            cap_limited=shell_cap_limited,
            finger_names=np.asarray(finger_names),
        )
        radius_summary: dict[str, Any] = {}
        shell_per_finger: dict[str, Any] = {}
        for finger_index, finger in enumerate(finger_names):
            active_raw = raw_radii[:, finger_index]
            active_raw = active_raw[np.isfinite(active_raw)]
            active_smoothed = smoothed_radii[:, finger_index]
            active_smoothed = active_smoothed[active_smoothed > 0]
            radius_summary[finger] = {
                "active_frames": int(len(active_smoothed)),
                "raw_median_px": (
                    float(np.median(active_raw)) if len(active_raw) else None
                ),
                "smoothed_median_px": (
                    float(np.median(active_smoothed))
                    if len(active_smoothed)
                    else None
                ),
                "smoothed_min_px": (
                    int(active_smoothed.min()) if len(active_smoothed) else None
                ),
                "smoothed_max_px": (
                    int(active_smoothed.max()) if len(active_smoothed) else None
                ),
            }
            shell_per_finger[finger] = {
                "seed_pixels": int(shell_seed_by_finger[:, finger_index].sum()),
                "added_pixels": int(
                    shell_added_by_finger[:, finger_index].sum()
                ),
                "frames_with_additions": int(
                    (shell_added_by_finger[:, finger_index] > 0).sum()
                ),
                "cap_limited_frames": int(
                    shell_cap_limited[:, finger_index].sum()
                ),
            }

        report: dict[str, Any] = {
            "schema_version": 2,
            "comparison": "xhand_thickness_strategies",
            "frames": reference.frames,
            "width": reference.width,
            "height": reference.height,
            "fps": reference.fps,
            "finger_names": list(finger_names),
            "panel_layout": [
                ["baseline", "half_thickness", "full_thickness"],
                [
                    "visibility_force",
                    "baseline_force_union",
                    "union_safety_shell_diagnostic",
                ],
            ],
            "surface_panel_layout": [
                ["surface_labels_debug", "surface_front_baseline"],
                [
                    "surface_front_side_half",
                    "surface_front_side_half_back_full",
                ],
            ],
            "source_directories_are_read_only_inputs": True,
            "sources": {
                role: {
                    "directory": str(source_dirs[role]),
                    "report": str(reports[role][0]),
                    "mask": str(source_mask_paths[role]),
                    "video": str(source_videos[role]),
                }
                for role in SOURCE_MODE_NAMES
            }
            | {
                "overlay_dir": str(overlay_dir),
                "finger_labels": str(labels_path),
                "finger_surface_labels": str(surface_labels_path),
                "object_mask": str(object_mask_path),
                "object_restore_mask": str(object_restore_mask_path),
                "background": str(background_path),
                "raw_video": (
                    str(raw_video_path) if raw_video_path is not None else None
                ),
            },
            "source_validation": {
                "report_metadata": report_metadata,
                "report_count_fields_checked": checked_report_counts,
                "thickness_contract": thickness_contract,
                "surface_label_contract": {
                    "dtype": "uint8",
                    "packing": "3 * (finger_id - 1) + surface_id",
                    "zero": "non-finger",
                    "surface_ids": dict(SURFACE_IDS),
                    "shape_matches_finger_labels": True,
                    "decoded_finger_ids_match_finger_labels": True,
                },
                "video_metadata": {
                    role: value.to_json() for role, value in metadata.items()
                },
                "background_metadata": auxiliary_metadata[
                    "background"
                ].to_json(),
                "missing_report_background_paths": missing_background,
                "missing_report_raw_video_paths": missing_raw_video,
            },
            "derived_definitions": {
                "baseline_force_union": "baseline OR visibility_force",
                "union_safety_shell_diagnostic": (
                    "baseline_force_union OR diagnostic_safety_shell_added"
                ),
                "surface_front_baseline": (
                    "existing zero-thickness baseline; palmar/front behavior "
                    "is unchanged and other surfaces retain baseline in this "
                    "first cumulative stage"
                ),
                "surface_front_side_half": (
                    "baseline with lateral/side pixels selected from the "
                    "half-thickness mask"
                ),
                "surface_front_side_half_back_full": (
                    "palmar/front selects baseline, lateral/side selects "
                    "half-thickness, dorsal/back selects full-thickness"
                ),
                "surface_video_rendering": (
                    "per-pixel selection from aligned baseline, half-thickness, "
                    "and full-thickness source composites; no source video is "
                    "edited"
                ),
                "restoration": (
                    "source background pixels; only clean observed-object "
                    "restore-mask pixels use the aligned raw video when a "
                    "distinct raw source is available"
                ),
            },
            "safety_shell": {
                "classification": (
                    "diagnostic screen-space proxy; not calibrated metric "
                    "XHand thickness and not a 3-D hand translation"
                ),
                "config": asdict(shell_config),
                "selection_rule": (
                    "same semantic finger AND dilated modal object AND a "
                    "connected component touching the visibility-force seed"
                ),
                "radius_estimator": (
                    "quantile of the 2-D distance transform over each "
                    "force-seeded semantic-finger component, contiguous-run "
                    "temporal median, then configured clamp"
                ),
                "area_cap": (
                    "ceil(force seed pixels * added_area_cap_fraction) per "
                    "frame/finger"
                ),
                "radius_summary": radius_summary,
                "per_finger": shell_per_finger,
                "total_added_pixels": int(shell_added_counts.sum()),
                "frames_with_additions": int((shell_added_counts > 0).sum()),
                "cap_limited_frame_fingers": int(shell_cap_limited.sum()),
                "evidence_file": SHELL_EVIDENCE_NAME,
            },
            "mode_statistics": mode_statistics,
            "pairwise": pairwise,
            "surface_labels": rendered_surface_statistics,
            "surface_strategy_statistics": surface_mode_statistics,
            "surface_strategy_breakdown": surface_breakdown,
            "surface_pairwise": surface_pairwise,
            "invariants": {
                "videos_are_frame_aligned": True,
                "masks_are_frame_aligned": True,
                "all_masks_are_semantic_finger_only": True,
                "force_seed_is_inside_modal_object": True,
                "union_equals_baseline_or_force": True,
                "diagnostic_shell_is_union_superset": True,
                "shell_additions_are_seed_component_limited": True,
                "shell_additions_respect_per_frame_finger_area_cap": True,
                "source_report_counts_match_masks": True,
                "object_restore_mask_subset_of_modal_object": True,
                "surface_labels_decode_to_finger_labels": True,
                "surface_side_half_uses_baseline_except_side_half": True,
                "surface_weighted_uses_front_zero_side_half_back_full": True,
            },
            "outputs": {
                "comparison_video": OUTPUT_VIDEO_NAME,
                "baseline_force_union_mask": UNION_MASK_NAME,
                "baseline_force_union_video": UNION_VIDEO_NAME,
                "diagnostic_shell_added_mask": SHELL_ADDED_MASK_NAME,
                "diagnostic_shell_final_mask": SHELL_MASK_NAME,
                "diagnostic_shell_video": SHELL_VIDEO_NAME,
                "diagnostic_shell_evidence": SHELL_EVIDENCE_NAME,
                "surface_side_half_mask": SURFACE_LATERAL_MASK_NAME,
                "surface_weighted_mask": SURFACE_WEIGHTED_MASK_NAME,
                "surface_side_half_video": SURFACE_LATERAL_VIDEO_NAME,
                "surface_weighted_video": SURFACE_WEIGHTED_VIDEO_NAME,
                "surface_labels_debug_video": SURFACE_DEBUG_VIDEO_NAME,
                "surface_comparison_video": SURFACE_COMPARISON_VIDEO_NAME,
                "report": OUTPUT_REPORT_NAME,
            },
            "note": (
                "These are strategy differences, not ground-truth occlusion "
                "accuracy.  The magenta shell in the comparison panel is an "
                "intentional diagnostic highlight; the standalone shell video "
                "is untinted."
            ),
        }
        (staging / OUTPUT_REPORT_NAME).write_text(
            json.dumps(report, indent=2) + "\n"
        )
        del (
            union_masks,
            shell_added_masks,
            shell_masks,
            surface_lateral_masks,
            surface_weighted_masks,
        )
        publish_directory(str(staging), str(output_dir))
        atexit.unregister(cleanup)
    except BaseException:
        cleanup()
        atexit.unregister(cleanup)
        raise

    print(f"[ok] XHand thickness strategy comparison: {output_dir}", flush=True)
    print(
        "[info] "
        + ", ".join(
            f"{mode}={mode_statistics[mode]['pixels']}px/"
            f"{mode_statistics[mode]['frames']}f"
            for mode in MODE_NAMES
        ),
        flush=True,
    )
    print(
        "[info] "
        + ", ".join(
            f"{mode}={surface_mode_statistics[mode]['pixels']}px/"
            f"{surface_mode_statistics[mode]['frames']}f"
            for mode in SURFACE_STRATEGY_NAMES
        ),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--half_thickness_dir", type=Path, required=True)
    parser.add_argument("--full_thickness_dir", type=Path, required=True)
    parser.add_argument("--force_dir", type=Path, required=True)
    parser.add_argument("--overlay_dir", type=Path, required=True)
    parser.add_argument(
        "--surface_labels",
        type=Path,
        default=None,
        help=(
            "Optional robot_finger_surface_labels.npy override; defaults to "
            f"OVERLAY_DIR/{SURFACE_LABELS_NAME}"
        ),
    )
    parser.add_argument("--object_mask", type=Path, required=True)
    parser.add_argument(
        "--object_restore_mask",
        type=Path,
        default=None,
        help=(
            "Optional clean observed-object mask used only for aligned raw-RGB "
            "restoration; defaults to --object_mask"
        ),
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing --out_dir",
    )
    parser.add_argument(
        "--background",
        type=Path,
        default=None,
        help="Optional restoration-background override; inferred from reports",
    )
    parser.add_argument(
        "--raw_video",
        type=Path,
        default=None,
        help="Optional raw-video override for restoring modal-object pixels",
    )
    parser.add_argument(
        "--disable_safety_shell",
        action="store_true",
        help="Emit an empty diagnostic shell while retaining union outputs",
    )
    parser.add_argument("--shell_min_radius_px", type=int, default=3)
    parser.add_argument("--shell_max_radius_px", type=int, default=20)
    parser.add_argument("--shell_temporal_median_window", type=int, default=5)
    parser.add_argument("--shell_radius_quantile", type=float, default=0.85)
    parser.add_argument(
        "--shell_added_area_cap_fraction", type=float, default=0.75
    )
    args = parser.parse_args()
    build_comparison(
        args.baseline_dir,
        args.half_thickness_dir,
        args.full_thickness_dir,
        args.force_dir,
        overlay_dir=args.overlay_dir,
        object_mask=args.object_mask,
        object_restore_mask=args.object_restore_mask,
        output_dir=args.out_dir,
        background=args.background,
        raw_video=args.raw_video,
        surface_labels=args.surface_labels,
        overwrite=args.overwrite,
        shell_config=SafetyShellConfig(
            enabled=not args.disable_safety_shell,
            min_radius_px=args.shell_min_radius_px,
            max_radius_px=args.shell_max_radius_px,
            temporal_median_window=args.shell_temporal_median_window,
            radius_quantile=args.shell_radius_quantile,
            added_area_cap_fraction=args.shell_added_area_cap_fraction,
        ),
    )


if __name__ == "__main__":
    main()
