"""Validate and compare the 08-05 robot-overlay experiments.

Two calibrated-focal views of the same episode are expected: ``approx`` and
``calibrated``.  Each processed-demo directory contains conventionally named
variant directories (all paths can be overridden with ``--source-dir``)::

    overlay_method_raw/
    overlay_haco_mh/
    overlay_haco_dual/
    overlay_object3d_dual_aligned/
    overlay_object3d_force_temporal/
    overlay_best_inpaint_barrier/
    overlay_stereo_visibility/

The command publishes two required synchronized grids and, when true stereo
visibility exists in both branches, one additional camera-evidence grid:

* calibrated methods, 3x2;
* approximate versus calibrated and MH versus dual evidence, 4x2;
* MH versus dual-HaCo versus stereo-visibility+HaCo, 3x2 (optional).
* calibrated 08-04-style method history, 4x4 (optional when complete).

Source directories are read-only.  Videos, masks, reports, temporal offsets,
and controlled lineage are validated before any output is atomically
published.  The comparison report records streamed mask differences; these
are method outputs, not accuracy measurements because pixel-level occlusion
ground truth is unavailable.
"""

from __future__ import annotations

import argparse
import contextvars
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from atomic_directory_publish import publish_directory
from compare_calibration_inpainting_ab import stream_exact_original_rgb_identity
from make_video_comparison_grid import (
    DEFAULT_DURATION_TOLERANCE_S,
    GridLayout,
    NamedVideo,
    VideoMetadata,
    probe_video,
    render_comparison_grid_layout,
    validate_grid_input_metadata,
)


BRANCHES = ("approx", "calibrated")
CONTACT_VARIANTS = (
    "haco_mh",
    "haco_dual",
    "haco_half_depth",
    "haco_full_depth",
    "boundary_fill",
    "scalar_object_z",
    "object3d_surface_unaligned",
    "object3d_dual",
    "object3d_force_temporal",
)
CORE_CALIBRATED_VARIANTS = (
    "raw",
    "haco_mh",
    "haco_dual",
    "object3d_dual",
    "object3d_force_temporal",
    "barrier",
)
CALIBRATED_LAYOUT = (
    ("raw", "1 Raw overlay (no occlusion)"),
    ("haco_mh", "2 MH HaCo"),
    ("haco_dual", "3 MH+SH HaCo"),
    ("object3d_dual", "4 Dual HaCo + 2.5D"),
    ("object3d_force_temporal", "5 2.5D force + temporal"),
    ("barrier", "6 Inpaint + whole-XHand barrier"),
)
CALIBRATED_GRID = GridLayout(columns=3, rows=2)
CAMERA_CALIBRATION_GRID = GridLayout(columns=4, rows=2)
DUAL_GRID = GridLayout(columns=3, rows=2)
EXTENDED_GRID = GridLayout(columns=4, rows=4)

EXTENDED_LAYOUT = (
    ("raw", "1 Raw overlay"),
    ("haco_mh", "2 MH HaCo"),
    ("haco_dual", "3 Dual HaCo baseline"),
    ("haco_half_depth", "4 HaCo + XHand depth 0.5x"),
    ("haco_full_depth", "5 HaCo + XHand depth 1.0x"),
    ("boundary_fill", "6 Contact boundary fill"),
    ("stereo_visibility", "7 Stereo visibility + HaCo"),
    ("haco_visibility_union", "8 HaCo OR stereo visibility"),
    ("union_safety_shell", "9 Union + 2D safety shell"),
    ("surface_front_side_half", "10 Front 0x + side 0.5x"),
    (
        "surface_front_side_half_back_full",
        "11 Front 0x + side 0.5x + back 1.0x",
    ),
    ("scalar_object_z", "12 HaCo + scalar object-Z"),
    ("object3d_surface_unaligned", "13 Dense 2.5D surface"),
    ("object3d_dual", "14 2.5D surface + HaCo register"),
    ("object3d_force_temporal", "15 2.5D force + temporal"),
    ("barrier", "16 Inpaint + whole-XHand barrier"),
)

CALIBRATED_VIDEO_NAME = "video_compare_calibrated_methods_3x2.mp4"
CAMERA_CALIBRATION_VIDEO_NAME = (
    "video_compare_camera_calibration_4x2.mp4"
)
DUAL_VIDEO_NAME = "video_compare_dual_camera_3x2.mp4"
EXTENDED_VIDEO_NAME = "video_compare_overlay_history_4x4.mp4"
REPORT_NAME = "comparison_report.json"
_ACTIVE_STAGING_PATHS: contextvars.ContextVar[list[Path] | None] = (
    contextvars.ContextVar("overlay_0805_staging_paths", default=None)
)


def _cleanup_staging_on_exit(function):
    """Clean registered staging trees immediately on success or failure."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        staging_paths: list[Path] = []
        token = _ACTIVE_STAGING_PATHS.set(staging_paths)
        try:
            return function(*args, **kwargs)
        finally:
            for path in reversed(staging_paths):
                shutil.rmtree(path, ignore_errors=True)
            _ACTIVE_STAGING_PATHS.reset(token)

    return wrapped


def _register_staging_path(path: Path) -> None:
    staging_paths = _ACTIVE_STAGING_PATHS.get()
    if staging_paths is None:
        raise RuntimeError("staging path registered outside cleanup scope")
    staging_paths.append(path)


@dataclass(frozen=True)
class VariantSpec:
    key: str
    directory_name: str
    video_name: str
    mask_name: str | None
    report_name: str | None
    report_kind: str


VARIANT_SPECS = {
    item.key: item
    for item in (
        VariantSpec(
            "raw",
            "overlay_method_raw",
            "video_overlay_robot_raw.mp4",
            None,
            None,
            "raw",
        ),
        VariantSpec(
            "haco_mh",
            "overlay_haco_mh",
            "video_overlay_contact.mp4",
            "occluded_finger_mask.npy",
            "report.json",
            "contact_haco_mh",
        ),
        VariantSpec(
            "haco_dual",
            "overlay_haco_dual",
            "video_overlay_contact.mp4",
            "occluded_finger_mask.npy",
            "report.json",
            "contact_haco_dual",
        ),
        VariantSpec(
            "haco_half_depth",
            "overlay_haco_dual_xhand_half",
            "video_overlay_contact.mp4",
            "occluded_finger_mask.npy",
            "report.json",
            "contact_haco_half_depth",
        ),
        VariantSpec(
            "haco_full_depth",
            "overlay_haco_dual_xhand_full",
            "video_overlay_contact.mp4",
            "occluded_finger_mask.npy",
            "report.json",
            "contact_haco_full_depth",
        ),
        VariantSpec(
            "boundary_fill",
            "overlay_haco_dual_boundary_fill",
            "video_overlay_contact.mp4",
            "occluded_finger_mask.npy",
            "report.json",
            "contact_boundary_fill",
        ),
        VariantSpec(
            "scalar_object_z",
            "overlay_object_scalar_dual",
            "video_overlay_contact.mp4",
            "occluded_finger_mask.npy",
            "report.json",
            "contact_scalar_object_z",
        ),
        VariantSpec(
            "object3d_surface_unaligned",
            "overlay_object3d_dual_unaligned",
            "video_overlay_contact.mp4",
            "occluded_finger_mask.npy",
            "report.json",
            "contact_object3d_unaligned",
        ),
        VariantSpec(
            "object3d_dual",
            "overlay_object3d_dual_aligned",
            "video_overlay_contact.mp4",
            "occluded_finger_mask.npy",
            "report.json",
            "contact_object3d",
        ),
        VariantSpec(
            "object3d_force_temporal",
            "overlay_object3d_force_temporal",
            "video_overlay_contact.mp4",
            "occluded_finger_mask.npy",
            "report.json",
            "contact_object3d_force_temporal",
        ),
        VariantSpec(
            "barrier",
            "overlay_best_inpaint_barrier",
            "video_overlay_hand_barrier.mp4",
            "occluded_hand_mask.npy",
            "report.json",
            "barrier",
        ),
        VariantSpec(
            "stereo_visibility",
            "overlay_stereo_visibility",
            "video_overlay_visibility_haco.mp4",
            "occluded_finger_mask_visibility_haco.npy",
            "report.json",
            "stereo_visibility",
        ),
        VariantSpec(
            "haco_visibility_union",
            "overlay_xhand_surface_strategies",
            "video_overlay_baseline_force_union.mp4",
            "occluded_finger_mask_baseline_force_union.npy",
            "comparison_report.json",
            "derived_haco_visibility_union",
        ),
        VariantSpec(
            "union_safety_shell",
            "overlay_xhand_surface_strategies",
            "video_overlay_union_safety_shell_diagnostic.mp4",
            "occluded_finger_mask_union_safety_shell_diagnostic.npy",
            "comparison_report.json",
            "derived_union_safety_shell",
        ),
        VariantSpec(
            "surface_front_side_half",
            "overlay_xhand_surface_strategies",
            "video_overlay_surface_front_side_half.mp4",
            "occluded_finger_mask_surface_front_side_half.npy",
            "comparison_report.json",
            "derived_surface_front_side_half",
        ),
        VariantSpec(
            "surface_front_side_half_back_full",
            "overlay_xhand_surface_strategies",
            "video_overlay_surface_front_side_half_back_full.mp4",
            "occluded_finger_mask_surface_front_side_half_back_full.npy",
            "comparison_report.json",
            "derived_surface_front_side_half_back_full",
        ),
    )
}


@dataclass
class LoadedVariant:
    branch: str
    spec: VariantSpec
    root: Path
    video: Path
    metadata: VideoMetadata
    mask_path: Path | None
    mask: np.ndarray | None
    report_path: Path | None
    report: dict[str, object] | None
    mask_statistics: dict[str, int]


def _metadata_dict(metadata: VideoMetadata) -> dict[str, object]:
    return {
        "width": metadata.width,
        "height": metadata.height,
        "frames": metadata.frame_count,
        "fps_fraction": (
            f"{metadata.fps.numerator}/{metadata.fps.denominator}"
        ),
        "fps": float(metadata.fps),
        "duration_s": metadata.duration_s,
        "codec": metadata.codec_name,
        "pixel_format": metadata.pixel_format,
    }


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON report: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"report must be a JSON object: {path}")
    return value


def parse_source_overrides(
    raw_values: Sequence[Sequence[str]] | None,
) -> dict[tuple[str, str], Path]:
    """Parse repeatable ``BRANCH VARIANT DIRECTORY`` override triples."""

    output: dict[tuple[str, str], Path] = {}
    for raw in raw_values or ():
        if len(raw) != 3:
            raise ValueError("each --source-dir needs BRANCH VARIANT DIRECTORY")
        branch, variant, directory = raw
        if branch not in BRANCHES:
            raise ValueError(
                f"unknown source branch {branch!r}; expected {BRANCHES}"
            )
        if variant not in VARIANT_SPECS:
            raise ValueError(
                f"unknown source variant {variant!r}; expected "
                f"{tuple(VARIANT_SPECS)}"
            )
        key = (branch, variant)
        if key in output:
            raise ValueError(f"duplicate source override for {branch}/{variant}")
        output[key] = Path(directory).expanduser().resolve()
    return output


def resolve_variant_directory(
    pd: Path,
    branch: str,
    variant: str,
    overrides: Mapping[tuple[str, str], Path],
) -> Path:
    if branch not in BRANCHES:
        raise ValueError(f"unknown branch {branch!r}")
    try:
        spec = VARIANT_SPECS[variant]
    except KeyError as exc:
        raise ValueError(f"unknown variant {variant!r}") from exc
    return overrides.get(
        (branch, variant),
        pd.resolve() / spec.directory_name,
    ).resolve()


def _variant_is_complete(
    pd: Path,
    branch: str,
    variant: str,
    overrides: Mapping[tuple[str, str], Path],
) -> bool:
    spec = VARIANT_SPECS[variant]
    root = resolve_variant_directory(pd, branch, variant, overrides)
    names = [spec.video_name]
    if spec.mask_name is not None:
        names.append(spec.mask_name)
    if spec.report_name is not None:
        names.append(spec.report_name)
    return all((root / name).is_file() and (root / name).stat().st_size > 0 for name in names)


def _find_episode_manifest(pd: Path) -> Path:
    for ancestor in (pd.resolve(), *pd.resolve().parents):
        candidate = ancestor / "stereo_manifest.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"could not find stereo_manifest.json above processed demo {pd}"
    )


def _manifest_offset(path: Path) -> int:
    payload = _load_json_object(path)
    try:
        return int(
            payload["temporal_alignment"]["camera1_frame_offset"]  # type: ignore[index]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"manifest lacks temporal_alignment.camera1_frame_offset: {path}"
        ) from exc


def _sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def validate_calibration_manifest_pair(
    approx: Mapping[str, object],
    calibrated: Mapping[str, object],
) -> dict[str, object]:
    """Require a controlled no-calibration/calibrated manifest pair."""

    controlled_fields = (
        "episode",
        "fps",
        "common_frames",
        "label_vocabulary",
        "primary_view",
        "auxiliary_view",
        "stereo_code_mapping",
        "training_view",
        "robot_overlay_view",
        "source_pairing",
        "sources",
        "raw_frame_counts",
        "tail_frames_dropped",
        "temporal_alignment",
        "frame_mapping",
    )
    for field in controlled_fields:
        if approx.get(field) != calibrated.get(field):
            raise ValueError(
                f"calibration manifests differ in controlled field {field}"
            )
    if (
        calibrated.get("primary_view") != "MH"
        or calibrated.get("auxiliary_view") != "SH"
        or calibrated.get("robot_overlay_view") != "MH"
    ):
        raise ValueError("manifest must define MH output geometry and SH auxiliary view")

    approx_calibration = approx.get("calibration")
    calibrated_calibration = calibrated.get("calibration")
    approx_intrinsics = approx.get("intrinsics")
    calibrated_intrinsics = calibrated.get("intrinsics")
    if not all(
        isinstance(value, dict)
        for value in (
            approx_calibration,
            calibrated_calibration,
            approx_intrinsics,
            calibrated_intrinsics,
        )
    ):
        raise ValueError("calibration manifests lack calibration/intrinsics objects")
    assert isinstance(approx_calibration, dict)
    assert isinstance(calibrated_calibration, dict)
    assert isinstance(approx_intrinsics, dict)
    assert isinstance(calibrated_intrinsics, dict)
    if (
        approx_calibration.get("status") != "not_provided"
        or approx_intrinsics.get("status") != "not_provided"
    ):
        raise ValueError("approx manifest unexpectedly contains calibration")
    if (
        calibrated_calibration.get("status") != "provided"
        or calibrated_intrinsics.get("status") != "provided"
    ):
        raise ValueError("calibrated manifest does not contain calibration")

    focal_values = calibrated_intrinsics.get("pixel_focal_px")
    intrinsic_views = calibrated_calibration.get("intrinsics_by_view")
    if not isinstance(focal_values, dict) or not isinstance(intrinsic_views, dict):
        raise ValueError("calibrated manifest lacks per-view intrinsics")
    calibrated_focals: dict[str, float] = {}
    for view in ("MH", "SH"):
        view_record = intrinsic_views.get(view)
        if not isinstance(view_record, dict):
            raise ValueError(f"calibrated manifest lacks {view} intrinsics")
        try:
            focal = float(focal_values[view])
            reference_focal = float(view_record["fx_px"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"calibrated manifest has invalid {view} focal") from exc
        if not math.isfinite(focal) or focal <= 0.0 or not math.isclose(
            focal,
            reference_focal,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(f"calibrated manifest has inconsistent {view} focal")
        calibrated_focals[view] = focal

    reference_value = calibrated_calibration.get("reference_json")
    expected_sha256 = calibrated_calibration.get("reference_sha256")
    if not isinstance(reference_value, str) or not reference_value:
        raise ValueError("calibrated manifest lacks calibration reference path")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("calibrated manifest lacks calibration reference digest")
    reference_path = Path(reference_value).expanduser().resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"calibration reference JSON is missing: {reference_path}"
        )
    actual_sha256 = _sha256_file(reference_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("calibration reference JSON digest has changed")

    checkerboard = calibrated_calibration.get("checkerboard")
    metric_scale_verified = (
        checkerboard.get("metric_scale_verified") is True
        if isinstance(checkerboard, dict)
        else False
    )
    return {
        "episode": str(calibrated.get("episode")),
        "common_frames": int(calibrated.get("common_frames", -1)),
        "fps": float(calibrated.get("fps", -1.0)),
        "approx_status": "not_provided",
        "calibrated_status": "provided",
        "calibration_reference": str(reference_path),
        "calibration_reference_sha256": actual_sha256,
        "calibrated_focal_px": calibrated_focals,
        "metric_scale_verified": metric_scale_verified,
        "controlled_fields": list(controlled_fields),
    }


def _mask_summary(mask: np.ndarray, *, chunk_frames: int = 8) -> dict[str, int]:
    if mask.ndim != 3 or mask.dtype != np.bool_:
        raise ValueError(f"mask must be bool (T,H,W), got {mask.shape}/{mask.dtype}")
    pixels = 0
    frames = 0
    maximum_pixels_per_frame = 0
    for start in range(0, len(mask), chunk_frames):
        block = np.asarray(mask[start : start + chunk_frames], dtype=bool)
        per_frame = np.count_nonzero(block, axis=(1, 2))
        pixels += int(per_frame.sum())
        frames += int(np.count_nonzero(per_frame))
        if len(per_frame):
            maximum_pixels_per_frame = max(
                maximum_pixels_per_frame,
                int(per_frame.max()),
            )
    return {
        "pixels": pixels,
        "frames": frames,
        "max_pixels_per_frame": maximum_pixels_per_frame,
    }


def _report_count_fields(
    spec: VariantSpec,
    report: Mapping[str, object],
) -> tuple[int, int]:
    if spec.report_kind.startswith("contact_"):
        return (
            int(report.get("occluded_pixels_total", -1)),
            int(report.get("frames_with_occlusion", -1)),
        )
    if spec.report_kind == "barrier":
        counts = report.get("counts")
        if not isinstance(counts, dict):
            raise ValueError("barrier report counts are missing")
        return (
            int(counts.get("final_occluded_pixels", -1)),
            int(counts.get("final_frames_with_occlusion", -1)),
        )
    if spec.report_kind == "stereo_visibility":
        mode_statistics = report.get("mode_statistics")
        if not isinstance(mode_statistics, dict):
            raise ValueError("stereo report mode_statistics are missing")
        statistics = mode_statistics.get("visibility_haco")
        if not isinstance(statistics, dict):
            raise ValueError("stereo report lacks visibility_haco statistics")
        return int(statistics.get("pixels", -1)), int(statistics.get("frames", -1))
    derived_mode_keys = {
        "derived_haco_visibility_union": (
            "mode_statistics",
            "baseline_force_union",
        ),
        "derived_union_safety_shell": (
            "mode_statistics",
            "union_safety_shell_diagnostic",
        ),
        "derived_surface_front_side_half": (
            "surface_strategy_statistics",
            "surface_front_side_half",
        ),
        "derived_surface_front_side_half_back_full": (
            "surface_strategy_statistics",
            "surface_front_side_half_back_full",
        ),
    }
    if spec.report_kind in derived_mode_keys:
        section_name, mode_name = derived_mode_keys[spec.report_kind]
        section = report.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"derived report lacks {section_name}")
        statistics = section.get(mode_name)
        if not isinstance(statistics, dict):
            raise ValueError(f"derived report lacks {mode_name} statistics")
        return int(statistics.get("pixels", -1)), int(statistics.get("frames", -1))
    raise ValueError(f"{spec.key} has no mask-count report contract")


def validate_variant_report(
    source: LoadedVariant,
    *,
    expected_offset: int,
) -> None:
    """Validate one report against its video, mask, and method definition."""

    spec = source.spec
    if spec.report_kind == "raw":
        if source.mask is not None or source.report is not None:
            raise ValueError("raw overlay must use the synthetic zero-occlusion contract")
        return
    report = source.report
    mask = source.mask
    if report is None or mask is None:
        raise ValueError(f"{spec.key} is missing its report or mask")
    expected_metadata = {
        "frames": source.metadata.frame_count,
        "width": source.metadata.width,
        "height": source.metadata.height,
    }
    for name, expected in expected_metadata.items():
        try:
            actual = int(report[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{spec.key} report lacks integer {name}") from exc
        if actual != expected:
            raise ValueError(
                f"{spec.key} report {name} {actual} != video {expected}"
            )
    try:
        report_fps = float(report["fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{spec.key} report lacks fps") from exc
    if not math.isclose(
        report_fps,
        float(source.metadata.fps),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise ValueError(f"{spec.key} report/video fps mismatch")
    if mask.shape != (
        source.metadata.frame_count,
        source.metadata.height,
        source.metadata.width,
    ):
        raise ValueError(
            f"{spec.key} mask/video geometry mismatch: {mask.shape}"
        )
    reported_pixels, reported_frames = _report_count_fields(spec, report)
    if reported_pixels != source.mask_statistics["pixels"]:
        raise ValueError(
            f"{spec.key} report/mask pixel-count mismatch: "
            f"{reported_pixels} != {source.mask_statistics['pixels']}"
        )
    if reported_frames != source.mask_statistics["frames"]:
        raise ValueError(
            f"{spec.key} report/mask frame-count mismatch: "
            f"{reported_frames} != {source.mask_statistics['frames']}"
        )

    sources = report.get("sources")
    if not isinstance(sources, dict):
        raise ValueError(f"{spec.key} report sources are missing")
    if spec.report_kind == "contact_haco_mh":
        if report.get("occlusion_mode") != "haco":
            raise ValueError("MH HaCo source is not in haco mode")
        if sources.get("aux_contact_dir") not in (None, ""):
            raise ValueError("MH HaCo source unexpectedly consumes auxiliary HaCo")
    elif spec.report_kind == "contact_haco_dual":
        if report.get("occlusion_mode") != "haco":
            raise ValueError("dual HaCo source is not in haco mode")
        _validate_contact_auxiliary(report, sources, expected_offset, spec.key)
    elif spec.report_kind in {
        "contact_haco_half_depth",
        "contact_haco_full_depth",
        "contact_boundary_fill",
    }:
        if report.get("occlusion_mode") != "haco":
            raise ValueError(f"{spec.key} is not in haco mode")
        _validate_contact_auxiliary(report, sources, expected_offset, spec.key)
        config = report.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"{spec.key} config is missing")
        if spec.report_kind == "contact_haco_half_depth" and not math.isclose(
            float(config.get("contact_depth_thickness_scale", -1.0)),
            0.5,
            abs_tol=1.0e-9,
        ):
            raise ValueError("haco_half_depth requires XHand thickness scale 0.5")
        if spec.report_kind == "contact_haco_full_depth" and not math.isclose(
            float(config.get("contact_depth_thickness_scale", -1.0)),
            1.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("haco_full_depth requires XHand thickness scale 1.0")
        if spec.report_kind == "contact_boundary_fill":
            expansion = report.get("contact_interior_expansion")
            if not isinstance(expansion, dict) or expansion.get("enabled") is not True:
                raise ValueError("boundary_fill lacks enabled interior expansion")
            if int(expansion.get("expand_px", 0)) <= 0:
                raise ValueError("boundary_fill has a non-positive expansion radius")
    elif spec.report_kind == "contact_scalar_object_z":
        if report.get("occlusion_mode") != "ensemble":
            raise ValueError("scalar_object_z is not in ensemble mode")
        _validate_contact_auxiliary(report, sources, expected_offset, spec.key)
        if not sources.get("scene_depth"):
            raise ValueError("scalar_object_z lacks scene-depth lineage")
    elif spec.report_kind in {
        "contact_object3d",
        "contact_object3d_unaligned",
        "contact_object3d_force_temporal",
    }:
        if report.get("occlusion_mode") != "object3d":
            raise ValueError(f"{spec.key} is not in object3d mode")
        _validate_contact_auxiliary(report, sources, expected_offset, spec.key)
        surface = report.get("object_surface_3d")
        expected_alignment = (
            "none"
            if spec.report_kind == "contact_object3d_unaligned"
            else "contact"
        )
        if not isinstance(surface, dict) or surface.get("alignment") != expected_alignment:
            raise ValueError(
                f"{spec.key} does not use {expected_alignment!r} 2.5D alignment"
            )
        invariants = report.get("invariants")
        if not isinstance(invariants, dict):
            raise ValueError(f"{spec.key} object3d invariants are missing")
        if not sources.get("object_surface_depth"):
            raise ValueError(f"{spec.key} lacks object surface lineage")
        config = report.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"{spec.key} config is missing")
        force_enabled = bool(config.get("object3d_force_surface", False))
        gap_frames = int(config.get("object3d_temporal_max_gap_frames", 0))
        if spec.report_kind in {
            "contact_object3d",
            "contact_object3d_unaligned",
        }:
            if invariants.get("object3d_haco_is_selector_only") is not True:
                raise ValueError(f"{spec.key} lacks selector-only HaCo invariant")
            if force_enabled or gap_frames != 0:
                raise ValueError(
                    f"{spec.key} must not include force/temporal ablations"
                )
        else:
            if not force_enabled or gap_frames != 2:
                raise ValueError(
                    "object3d_force_temporal requires surface force and a "
                    "2-frame gap"
                )
            if not all(
                invariants.get(name) is True
                for name in (
                    "object3d_force_bypasses_haco_selector",
                    "object3d_temporal_filter_only_adds_occlusion",
                )
            ):
                raise ValueError(
                    "object3d_force_temporal lacks force/temporal invariants"
                )
    elif spec.report_kind == "barrier":
        if report.get("method") != "visual_camera_z_xhand_barrier":
            raise ValueError("barrier report has the wrong method")
        if report.get("pose_state_modified") is not False:
            raise ValueError("barrier must not modify robot pose state")
        if report.get("metric_collision_guarantee") is not False:
            raise ValueError("barrier must not claim metric collision safety")
        counts = report.get("counts")
        if not isinstance(counts, dict) or int(
            counts.get("residual_violation_pixels", -1)
        ) != 0:
            raise ValueError("barrier retains camera-Z residual violations")
        invariants = report.get("invariants")
        required = (
            "baseline_subset_final",
            "final_occlusion_subset_of_xhand",
            "rb5_arm_excluded",
            "valid_surface_barrier_residual_is_zero",
            "trajectory_arrays_unchanged",
        )
        if not isinstance(invariants, dict) or not all(
            invariants.get(name) is True for name in required
        ):
            raise ValueError("barrier report invariants are incomplete")
        if not sources.get("baseline_mask") or not sources.get("object_surface_depth"):
            raise ValueError("barrier report lacks baseline/surface lineage")
    elif spec.report_kind == "stereo_visibility":
        modes = report.get("output_modes")
        if not isinstance(modes, list) or "visibility_haco" not in modes:
            raise ValueError("stereo source lacks visibility_haco output mode")
        if report.get("camera2_is_final_view") is not True:
            raise ValueError("stereo source does not keep camera 2 as final view")
        temporal = report.get("temporal_alignment")
        if not isinstance(temporal, dict) or int(
            temporal.get("camera1_frame_offset", 999999)
        ) != expected_offset:
            raise ValueError("stereo source has the wrong camera-1 frame offset")
        required_sources = (
            "camera1_hawor",
            "camera2_hawor",
            "camera1_contact_dir",
            "contact_dir",
            "camera1_visible_mask",
            "camera2_visible_mask",
            "overlay_dir",
        )
        if any(not sources.get(name) for name in required_sources):
            raise ValueError("stereo source lacks true dual-view lineage")
        invariants = report.get("invariants")
        if not isinstance(invariants, dict) or invariants.get(
            "dual_haco_uses_max_available_score"
        ) is not True:
            raise ValueError("stereo source lacks dual-HaCo fusion invariant")
    elif spec.report_kind.startswith("derived_"):
        if report.get("comparison") != "xhand_thickness_strategies":
            raise ValueError(f"{spec.key} has the wrong derived comparison report")
        invariants = report.get("invariants")
        if not isinstance(invariants, dict):
            raise ValueError(f"{spec.key} derived invariants are missing")
        required_invariants = {
            "derived_haco_visibility_union": ("union_equals_baseline_or_force",),
            "derived_union_safety_shell": (
                "union_equals_baseline_or_force",
                "diagnostic_shell_is_union_superset",
            ),
            "derived_surface_front_side_half": (
                "surface_labels_decode_to_finger_labels",
                "surface_side_half_uses_baseline_except_side_half",
            ),
            "derived_surface_front_side_half_back_full": (
                "surface_labels_decode_to_finger_labels",
                "surface_weighted_uses_front_zero_side_half_back_full",
            ),
        }[spec.report_kind]
        if not all(invariants.get(name) is True for name in required_invariants):
            raise ValueError(f"{spec.key} derived invariants are incomplete")


def _validate_contact_auxiliary(
    report: Mapping[str, object],
    sources: Mapping[str, object],
    expected_offset: int,
    variant: str,
) -> None:
    if not sources.get("aux_contact_dir"):
        raise ValueError(f"{variant} lacks auxiliary HaCo lineage")
    try:
        actual_offset = int(report["aux_frame_offset"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{variant} lacks auxiliary frame offset") from exc
    if actual_offset != expected_offset:
        raise ValueError(
            f"{variant} auxiliary offset {actual_offset} != {expected_offset}"
        )
    invariants = report.get("invariants")
    required = {
        "auxiliary_haco_is_confidence_only": True,
        "auxiliary_geometry_used": False,
        "primary_view_owns_contact_projection_and_depth": True,
    }
    if not isinstance(invariants, dict) or any(
        invariants.get(name) is not expected
        for name, expected in required.items()
    ):
        raise ValueError(
            f"{variant} lacks MH-geometry/SH-confidence invariants"
        )


def load_variant(
    *,
    pd: Path,
    branch: str,
    variant: str,
    overrides: Mapping[tuple[str, str], Path],
    expected_offset: int,
    ffprobe: str = "ffprobe",
) -> LoadedVariant:
    spec = VARIANT_SPECS[variant]
    root = resolve_variant_directory(pd, branch, variant, overrides)
    video = root / spec.video_name
    if not video.is_file() or video.stat().st_size <= 0:
        raise FileNotFoundError(f"{branch}/{variant} video is missing: {video}")
    metadata = probe_video(video, ffprobe)
    mask_path = root / spec.mask_name if spec.mask_name is not None else None
    report_path = root / spec.report_name if spec.report_name is not None else None
    mask = None
    report = None
    if mask_path is not None:
        if not mask_path.is_file() or mask_path.stat().st_size <= 0:
            raise FileNotFoundError(f"{branch}/{variant} mask is missing: {mask_path}")
        mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        mask_statistics = _mask_summary(mask)
    else:
        mask_statistics = {
            "pixels": 0,
            "frames": 0,
            "max_pixels_per_frame": 0,
        }
    if report_path is not None:
        if not report_path.is_file() or report_path.stat().st_size <= 0:
            raise FileNotFoundError(
                f"{branch}/{variant} report is missing: {report_path}"
            )
        report = _load_json_object(report_path)
    source = LoadedVariant(
        branch=branch,
        spec=spec,
        root=root,
        video=video.resolve(),
        metadata=metadata,
        mask_path=mask_path.resolve() if mask_path is not None else None,
        mask=mask,
        report_path=report_path.resolve() if report_path is not None else None,
        report=report,
        mask_statistics=mask_statistics,
    )
    validate_variant_report(source, expected_offset=expected_offset)
    return source


def _path_value(report: Mapping[str, object], field: str) -> Path | None:
    sources = report.get("sources")
    if not isinstance(sources, dict):
        return None
    value = sources.get(field)
    if not isinstance(value, str) or not value:
        return None
    return Path(value).expanduser().resolve()


def _same_source(
    sources: Mapping[str, LoadedVariant],
    variants: Sequence[str],
    field: str,
) -> Path:
    values: dict[str, Path | None] = {}
    for variant in variants:
        report = sources[variant].report
        if report is None:
            raise ValueError(f"{variant} has no report for source-lineage validation")
        values[variant] = _path_value(report, field)
    unique = {value for value in values.values() if value is not None}
    if len(unique) != 1 or any(value is None for value in values.values()):
        raise ValueError(f"variants use different {field} lineage: {values}")
    return next(iter(unique))


def _validate_derived_lineage(
    *,
    branch: str,
    sources: Mapping[str, LoadedVariant],
    overlay_dir: Path,
    common_contact_sources: Mapping[str, str],
) -> None:
    derived_variants = [
        name
        for name in (
            "haco_visibility_union",
            "union_safety_shell",
            "surface_front_side_half",
            "surface_front_side_half_back_full",
        )
        if name in sources
    ]
    if not derived_variants:
        return
    role_sources = {
        "baseline": "haco_dual",
        "half_thickness": "haco_half_depth",
        "full_thickness": "haco_full_depth",
        "visibility_force": "stereo_visibility",
    }
    missing = [name for name in role_sources.values() if name not in sources]
    if missing:
        raise ValueError(
            f"{branch} derived strategies lack source variants: {missing}"
        )
    for derived_name in derived_variants:
        report = sources[derived_name].report
        assert report is not None
        report_sources = report.get("sources")
        if not isinstance(report_sources, dict):
            raise ValueError(f"{branch}/{derived_name} lacks source lineage")
        expected_top_level = {
            "overlay_dir": overlay_dir,
            "object_mask": Path(common_contact_sources["object_mask"]),
            "background": Path(common_contact_sources["background"]),
            "raw_video": Path(common_contact_sources["raw_video"]),
        }
        for field, expected in expected_top_level.items():
            actual = report_sources.get(field)
            if not isinstance(actual, str) or Path(actual).resolve() != expected:
                raise ValueError(
                    f"{branch}/{derived_name} changes derived {field} lineage"
                )
        for role, source_name in role_sources.items():
            record = report_sources.get(role)
            if not isinstance(record, dict):
                raise ValueError(
                    f"{branch}/{derived_name} lacks derived {role} lineage"
                )
            expected_source = sources[source_name]
            expected_paths = {
                "directory": expected_source.root,
                "report": expected_source.report_path,
            }
            for field, expected in expected_paths.items():
                actual = record.get(field)
                if (
                    expected is None
                    or not isinstance(actual, str)
                    or Path(actual).resolve() != expected
                ):
                    raise ValueError(
                        f"{branch}/{derived_name} changes {role} {field} lineage"
                    )


def validate_branch_contract(
    *,
    branch: str,
    pd: Path,
    sources: Mapping[str, LoadedVariant],
    expected_offset: int,
    duration_tolerance_s: float = DEFAULT_DURATION_TOLERANCE_S,
) -> dict[str, object]:
    """Validate synchronized metadata and controlled compositor lineage."""

    metadata = [item.metadata for item in sources.values()]
    reference = validate_grid_input_metadata(
        metadata,
        len(metadata),
        duration_tolerance_s,
    )
    for variant, item in sources.items():
        if (item.metadata.width, item.metadata.height) != (
            reference.width,
            reference.height,
        ):
            raise ValueError(f"{branch}/{variant} has different video geometry")
    available_contacts = [name for name in CONTACT_VARIANTS if name in sources]
    if "haco_mh" not in available_contacts or "haco_dual" not in available_contacts:
        raise ValueError(f"{branch} requires both MH and dual HaCo sources")
    common_contact_sources: dict[str, str] = {}
    for field in (
        "processed_demo",
        "background",
        "raw_video",
        "hawor_npz",
        "contact_dir",
        "overlay_dir",
        "object_mask",
    ):
        common_contact_sources[field] = str(
            _same_source(sources, available_contacts, field)
        )
    if Path(common_contact_sources["processed_demo"]).resolve() != pd.resolve():
        raise ValueError(f"{branch} contact reports do not belong to {pd}")

    dual_variants = [
        name for name in available_contacts if name != "haco_mh"
    ]
    auxiliary_contact = _same_source(sources, dual_variants, "aux_contact_dir")
    reference_dual = sources["haco_dual"].report
    assert reference_dual is not None
    for variant in dual_variants[1:]:
        report = sources[variant].report
        assert report is not None
        for field in ("contact_score_fused", "hidden_fraction"):
            if report.get(field) != reference_dual.get(field):
                raise ValueError(
                    f"{branch}/{variant} changes controlled {field} evidence"
                )
        if int(report.get("aux_frame_offset", 999999)) != expected_offset:
            raise ValueError(f"{branch}/{variant} changes temporal alignment")

    surface_path = None
    surface_variants = [
        name
        for name in (
            "object3d_surface_unaligned",
            "object3d_dual",
            "object3d_force_temporal",
        )
        if name in sources
    ]
    if surface_variants:
        surface_path = _same_source(
            sources,
            surface_variants,
            "object_surface_depth",
        )
    overlay_dir = Path(common_contact_sources["overlay_dir"])
    barrier_baseline_check = None
    if "barrier" in sources:
        barrier_report = sources["barrier"].report
        assert barrier_report is not None
        barrier_overlay = _path_value(barrier_report, "overlay_dir")
        if barrier_overlay != overlay_dir:
            raise ValueError(f"{branch} barrier uses a different robot overlay")
        for field in ("background", "raw_video"):
            if _path_value(barrier_report, field) != Path(
                common_contact_sources[field]
            ):
                raise ValueError(
                    f"{branch} barrier uses different {field} input"
                )
        barrier_surface = _path_value(barrier_report, "object_surface_depth")
        if surface_path is not None and barrier_surface != surface_path:
            raise ValueError(f"{branch} barrier uses a different 2.5D surface")
        if "object3d_force_temporal" in sources:
            baseline = _path_value(barrier_report, "baseline_mask")
            force_source = sources["object3d_force_temporal"]
            expected_baseline = force_source.mask_path
            if baseline != expected_baseline:
                raise ValueError(
                    f"{branch} barrier baseline is not force+temporal mask"
                )
            counts = barrier_report.get("counts")
            if not isinstance(counts, dict) or int(
                counts.get("baseline_occluded_pixels", -1)
            ) != force_source.mask_statistics["pixels"]:
                raise ValueError(
                    f"{branch} barrier baseline count differs from current "
                    "force+temporal mask"
                )
            barrier_baseline_check = _mask_difference(
                force_source,
                sources["barrier"],
            )
            if int(barrier_baseline_check["removed_pixels"]) != 0:
                raise ValueError(
                    f"{branch} current force+temporal mask is not a subset "
                    "of the barrier mask"
                )
    if "stereo_visibility" in sources:
        stereo_report = sources["stereo_visibility"].report
        assert stereo_report is not None
        stereo_overlay = _path_value(stereo_report, "overlay_dir")
        if stereo_overlay != overlay_dir:
            raise ValueError(f"{branch} stereo mode uses a different robot overlay")
        expected_stereo_sources = {
            "camera2_hawor": Path(common_contact_sources["hawor_npz"]),
            "camera1_contact_dir": auxiliary_contact,
            "contact_dir": Path(common_contact_sources["contact_dir"]),
            "background": Path(common_contact_sources["background"]),
            "object_mask": Path(common_contact_sources["object_mask"]),
        }
        for field, expected in expected_stereo_sources.items():
            if _path_value(stereo_report, field) != expected:
                raise ValueError(
                    f"{branch} stereo mode changes controlled {field} input"
                )
    _validate_derived_lineage(
        branch=branch,
        sources=sources,
        overlay_dir=overlay_dir,
        common_contact_sources=common_contact_sources,
    )

    return {
        "branch": branch,
        "processed_demo": str(pd.resolve()),
        "metadata": _metadata_dict(reference),
        "camera1_frame_offset": expected_offset,
        "contact_sources": common_contact_sources,
        "aux_contact_dir": str(auxiliary_contact),
        "object_surface_depth": str(surface_path) if surface_path else None,
        "controlled_robot_overlay": str(overlay_dir),
        "barrier_baseline_check": barrier_baseline_check,
    }


def _mask_difference(
    first: LoadedVariant,
    second: LoadedVariant,
    *,
    chunk_frames: int = 8,
) -> dict[str, object]:
    """Stream directional mask differences from ``first`` to ``second``."""

    if first.metadata.frame_count != second.metadata.frame_count:
        raise ValueError("cannot compare masks with different frame counts")
    if (first.metadata.width, first.metadata.height) != (
        second.metadata.width,
        second.metadata.height,
    ):
        raise ValueError("cannot compare masks with different geometry")
    first_mask, second_mask = first.mask, second.mask
    if (
        first_mask is not None
        and second_mask is not None
        and first_mask.shape != second_mask.shape
    ):
        raise ValueError("cannot compare masks with different array shapes")
    added = removed = intersection = union = changed_frames = 0
    frames = first.metadata.frame_count
    for start in range(0, frames, chunk_frames):
        end = min(start + chunk_frames, frames)
        if first_mask is None:
            assert second_mask is not None or second.spec.report_kind == "raw"
            if second_mask is None:
                continue
            second_block = np.asarray(second_mask[start:end], dtype=bool)
            per_frame = np.count_nonzero(second_block, axis=(1, 2))
            block_pixels = int(per_frame.sum())
            added += block_pixels
            union += block_pixels
            changed_frames += int(np.count_nonzero(per_frame))
            continue
        if second_mask is None:
            first_block = np.asarray(first_mask[start:end], dtype=bool)
            per_frame = np.count_nonzero(first_block, axis=(1, 2))
            block_pixels = int(per_frame.sum())
            removed += block_pixels
            union += block_pixels
            changed_frames += int(np.count_nonzero(per_frame))
            continue
        first_block = np.asarray(first_mask[start:end], dtype=bool)
        second_block = np.asarray(second_mask[start:end], dtype=bool)
        block_added = second_block & ~first_block
        block_removed = first_block & ~second_block
        added += int(np.count_nonzero(block_added))
        removed += int(np.count_nonzero(block_removed))
        intersection += int(np.count_nonzero(first_block & second_block))
        union += int(np.count_nonzero(first_block | second_block))
        changed_frames += int(
            np.count_nonzero(np.any(block_added | block_removed, axis=(1, 2)))
        )
    return {
        "first": f"{first.branch}/{first.spec.key}",
        "second": f"{second.branch}/{second.spec.key}",
        "added_pixels": added,
        "removed_pixels": removed,
        "changed_pixels": added + removed,
        "changed_frames": changed_frames,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "jaccard": (intersection / union if union else None),
        "first_subset_second": removed == 0,
        "equal": added == 0 and removed == 0,
    }


def _within_branch_pairs(
    sources: Mapping[str, LoadedVariant],
) -> list[tuple[str, str]]:
    candidates = (
        ("raw", "haco_mh"),
        ("haco_mh", "haco_dual"),
        ("haco_dual", "haco_half_depth"),
        ("haco_half_depth", "haco_full_depth"),
        ("haco_dual", "boundary_fill"),
        ("haco_dual", "haco_visibility_union"),
        ("haco_visibility_union", "union_safety_shell"),
        ("haco_dual", "surface_front_side_half"),
        (
            "surface_front_side_half",
            "surface_front_side_half_back_full",
        ),
        ("haco_dual", "scalar_object_z"),
        ("haco_dual", "object3d_surface_unaligned"),
        ("object3d_surface_unaligned", "object3d_dual"),
        ("haco_dual", "object3d_dual"),
        ("object3d_dual", "object3d_force_temporal"),
        ("object3d_force_temporal", "barrier"),
        ("haco_dual", "stereo_visibility"),
    )
    return [pair for pair in candidates if all(name in sources for name in pair)]


def _source_record(source: LoadedVariant) -> dict[str, object]:
    return {
        "directory": str(source.root),
        "video": str(source.video),
        "mask": str(source.mask_path) if source.mask_path else None,
        "report": str(source.report_path) if source.report_path else None,
        "metadata": _metadata_dict(source.metadata),
        "mask_statistics": source.mask_statistics,
        "raw_uses_synthetic_zero_occlusion_mask": source.spec.key == "raw",
    }


def _hawor_focal_path(hawor_path: Path | None) -> tuple[Path, float]:
    if hawor_path is None or not hawor_path.is_file():
        raise FileNotFoundError(f"HaWoR source is missing: {hawor_path}")
    with np.load(hawor_path, allow_pickle=False) as data:
        try:
            focal = float(np.asarray(data["img_focal"]).item())
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid HaWoR img_focal: {hawor_path}") from exc
    if not math.isfinite(focal) or focal <= 0.0:
        raise ValueError(f"invalid HaWoR focal {focal}: {hawor_path}")
    return hawor_path, focal


def _hawor_focal(source: LoadedVariant) -> tuple[Path, float]:
    if source.report is None:
        raise ValueError(f"{source.spec.key} lacks HaWoR lineage")
    return _hawor_focal_path(_path_value(source.report, "hawor_npz"))


def _grid_videos(
    sources: Mapping[str, Mapping[str, LoadedVariant]],
    entries: Sequence[tuple[str, str, str]],
) -> list[NamedVideo]:
    return [
        NamedVideo(label=label, path=sources[branch][variant].video)
        for branch, variant, label in entries
    ]


def _select_camera_final_variant(
    requested: str,
    *,
    pd_by_branch: Mapping[str, Path],
    overrides: Mapping[tuple[str, str], Path],
) -> tuple[str, dict[str, bool]]:
    availability = {
        variant: all(
            _variant_is_complete(pd_by_branch[branch], branch, variant, overrides)
            for branch in BRANCHES
        )
        for variant in ("stereo_visibility", "barrier")
    }
    if requested == "auto":
        for candidate in ("stereo_visibility", "barrier"):
            if availability[candidate]:
                return candidate, availability
        raise FileNotFoundError(
            "camera/calibration grid needs stereo_visibility or barrier in both branches"
        )
    if not availability[requested]:
        raise FileNotFoundError(
            f"requested camera final variant {requested!r} is incomplete"
        )
    return requested, availability


def _validate_output_location(
    output: Path,
    *,
    pd_by_branch: Mapping[str, Path],
    sources: Mapping[str, Mapping[str, LoadedVariant]],
) -> None:
    output = output.resolve()
    for branch, pd in pd_by_branch.items():
        if output == pd or output in pd.parents:
            raise ValueError(
                f"output must not replace or contain the {branch} processed demo"
            )
    for branch, branch_sources in sources.items():
        for variant, source in branch_sources.items():
            root = source.root.resolve()
            if output == root or output in root.parents or root in output.parents:
                raise ValueError(
                    "comparison output must be separate from read-only source "
                    f"directory {branch}/{variant}: {root}"
                )


@_cleanup_staging_on_exit
def run_comparison(
    *,
    approx_pd: Path,
    calibrated_pd: Path,
    out_dir: Path,
    source_overrides: Mapping[tuple[str, str], Path] | None = None,
    camera_final_variant: str = "auto",
    extended_grid: str = "auto",
    validate_source_rgb_identity: bool = True,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    font_file: Path | None = None,
    duration_tolerance_s: float = DEFAULT_DURATION_TOLERANCE_S,
    crf: int = 18,
    preset: str = "medium",
    overwrite: bool = False,
) -> Path:
    if extended_grid not in {"auto", "required", "off"}:
        raise ValueError("extended_grid must be auto, required, or off")
    overrides = dict(source_overrides or {})
    pd_by_branch = {
        "approx": approx_pd.expanduser().resolve(),
        "calibrated": calibrated_pd.expanduser().resolve(),
    }
    for branch, pd in pd_by_branch.items():
        if not pd.is_dir():
            raise FileNotFoundError(f"{branch} processed demo is missing: {pd}")
        source_video = pd / "video_L.mp4"
        if not source_video.is_file():
            raise FileNotFoundError(f"{branch} source video is missing: {source_video}")

    manifests = {
        branch: _find_episode_manifest(pd)
        for branch, pd in pd_by_branch.items()
    }
    manifest_payloads = {
        branch: _load_json_object(path) for branch, path in manifests.items()
    }
    calibration_manifest_contract = validate_calibration_manifest_pair(
        manifest_payloads["approx"],
        manifest_payloads["calibrated"],
    )
    offsets = {branch: _manifest_offset(path) for branch, path in manifests.items()}
    if offsets["approx"] != offsets["calibrated"]:
        raise ValueError(f"calibration branches use different SH offsets: {offsets}")

    selected_final, final_availability = _select_camera_final_variant(
        camera_final_variant,
        pd_by_branch=pd_by_branch,
        overrides=overrides,
    )
    extended_variants = tuple(variant for variant, _label in EXTENDED_LAYOUT)
    extended_missing = [
        variant
        for variant in extended_variants
        if not _variant_is_complete(
            pd_by_branch["calibrated"],
            "calibrated",
            variant,
            overrides,
        )
    ]
    if extended_grid == "required" and extended_missing:
        raise FileNotFoundError(
            "required extended-grid sources are incomplete: "
            + ", ".join(extended_missing)
        )
    extended_enabled = extended_grid != "off" and not extended_missing
    required_by_branch = {
        "approx": {"raw", "haco_mh", "haco_dual", selected_final},
        "calibrated": set(CORE_CALIBRATED_VARIANTS) | {selected_final},
    }
    if selected_final == "barrier":
        required_by_branch["approx"].add("object3d_force_temporal")
    stereo_available = final_availability["stereo_visibility"]
    if stereo_available:
        required_by_branch["approx"].add("stereo_visibility")
        required_by_branch["calibrated"].add("stereo_visibility")
    if extended_enabled:
        required_by_branch["calibrated"].update(extended_variants)
    sources: dict[str, dict[str, LoadedVariant]] = {}
    for branch in BRANCHES:
        sources[branch] = {
            variant: load_variant(
                pd=pd_by_branch[branch],
                branch=branch,
                variant=variant,
                overrides=overrides,
                expected_offset=offsets[branch],
                ffprobe=ffprobe,
            )
            for variant in VARIANT_SPECS
            if variant in required_by_branch[branch]
        }

    branch_validation = {
        branch: validate_branch_contract(
            branch=branch,
            pd=pd_by_branch[branch],
            sources=sources[branch],
            expected_offset=offsets[branch],
            duration_tolerance_s=duration_tolerance_s,
        )
        for branch in BRANCHES
    }

    approx_hawor, approx_focal = _hawor_focal(sources["approx"]["haco_mh"])
    calibrated_hawor, calibrated_focal = _hawor_focal(
        sources["calibrated"]["haco_mh"]
    )
    if not approx_focal < calibrated_focal:
        raise ValueError(
            "expected approximate HaWoR focal to be smaller than calibrated focal"
        )
    calibrated_manifest_focals = calibration_manifest_contract[
        "calibrated_focal_px"
    ]
    assert isinstance(calibrated_manifest_focals, dict)
    if not math.isclose(
        calibrated_focal,
        float(calibrated_manifest_focals["MH"]),
        rel_tol=0.0,
        abs_tol=1.0e-3,
    ):
        raise ValueError("calibrated MH HaWoR focal differs from manifest")

    approx_sh_hawor = calibrated_sh_hawor = None
    approx_sh_focal = calibrated_sh_focal = None
    if stereo_available:
        approx_stereo_report = sources["approx"]["stereo_visibility"].report
        calibrated_stereo_report = sources["calibrated"][
            "stereo_visibility"
        ].report
        assert approx_stereo_report is not None
        assert calibrated_stereo_report is not None
        approx_sh_hawor, approx_sh_focal = _hawor_focal_path(
            _path_value(approx_stereo_report, "camera1_hawor")
        )
        calibrated_sh_hawor, calibrated_sh_focal = _hawor_focal_path(
            _path_value(calibrated_stereo_report, "camera1_hawor")
        )
        if not approx_sh_focal < calibrated_sh_focal:
            raise ValueError(
                "expected approximate SH HaWoR focal to be smaller than "
                "calibrated focal"
            )
        if not math.isclose(
            approx_sh_focal,
            approx_focal,
            rel_tol=0.0,
            abs_tol=1.0e-3,
        ):
            raise ValueError(
                "approx branch must use one default focal for MH and SH"
            )
        if not math.isclose(
            calibrated_sh_focal,
            float(calibrated_manifest_focals["SH"]),
            rel_tol=0.0,
            abs_tol=1.0e-3,
        ):
            raise ValueError("calibrated SH HaWoR focal differs from manifest")

    original_metadata = [
        probe_video(pd_by_branch[branch] / "video_L.mp4", ffprobe)
        for branch in BRANCHES
    ]
    source_reference = validate_grid_input_metadata(
        original_metadata,
        2,
        duration_tolerance_s,
    )
    if source_reference.frame_count != int(
        calibration_manifest_contract["common_frames"]
    ) or not math.isclose(
        float(source_reference.fps),
        float(calibration_manifest_contract["fps"]),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("manifest frame/fps metadata differs from source video")
    if (
        original_metadata[0].width,
        original_metadata[0].height,
    ) != (
        original_metadata[1].width,
        original_metadata[1].height,
    ):
        raise ValueError("approx/calibrated source videos have different geometry")
    for branch, original in zip(BRANCHES, original_metadata, strict=True):
        overlay_reference = sources[branch]["raw"].metadata
        if (original.width, original.height) != (
            overlay_reference.width,
            overlay_reference.height,
        ):
            raise ValueError(
                f"{branch} source and overlay videos have different geometry"
            )
    source_rgb_identity = (
        stream_exact_original_rgb_identity(
            pd_by_branch["approx"] / "video_L.mp4",
            pd_by_branch["calibrated"] / "video_L.mp4",
            metadata=source_reference,
        )
        if validate_source_rgb_identity
        else {
            "exact_equal": None,
            "skipped": True,
            "reason": "--skip-source-rgb-identity",
        }
    )

    # Selected overlay outputs must also share one synchronized frame axis.
    all_metadata = original_metadata + [
        item.metadata
        for branch in BRANCHES
        for item in sources[branch].values()
    ]
    validate_grid_input_metadata(
        all_metadata,
        len(all_metadata),
        duration_tolerance_s,
    )

    within_branch_statistics: dict[str, dict[str, object]] = {}
    for branch in BRANCHES:
        within_branch_statistics[branch] = {
            f"{first}_vs_{second}": _mask_difference(
                sources[branch][first], sources[branch][second]
            )
            for first, second in _within_branch_pairs(sources[branch])
        }
    cross_calibration_statistics = {
        variant: _mask_difference(
            sources["approx"][variant],
            sources["calibrated"][variant],
        )
        for variant in sorted(set(sources["approx"]) & set(sources["calibrated"]))
    }

    calibrated_entries = [
        ("calibrated", variant, label)
        for variant, label in CALIBRATED_LAYOUT
    ]
    final_label = (
        "Stereo visibility + dual HaCo"
        if selected_final == "stereo_visibility"
        else "Inpaint + whole-XHand barrier"
    )
    camera_entries = [
        ("approx", "raw", "Approx | raw"),
        ("approx", "haco_mh", "Approx | MH HaCo"),
        ("approx", "haco_dual", "Approx | MH+SH HaCo"),
        ("approx", selected_final, f"Approx | {final_label}"),
        ("calibrated", "raw", "Calibrated | raw"),
        ("calibrated", "haco_mh", "Calibrated | MH HaCo"),
        ("calibrated", "haco_dual", "Calibrated | MH+SH HaCo"),
        ("calibrated", selected_final, f"Calibrated | {final_label}"),
    ]
    dual_entries = [
        ("approx", "haco_mh", "Approx | MH HaCo"),
        ("approx", "haco_dual", "Approx | dual HaCo"),
        ("approx", "stereo_visibility", "Approx | stereo + HaCo"),
        ("calibrated", "haco_mh", "Calibrated | MH HaCo"),
        ("calibrated", "haco_dual", "Calibrated | dual HaCo"),
        ("calibrated", "stereo_visibility", "Calibrated | stereo + HaCo"),
    ]
    extended_entries = [
        ("calibrated", variant, label)
        for variant, label in EXTENDED_LAYOUT
    ]

    output = out_dir.expanduser().resolve()
    _validate_output_location(
        output,
        pd_by_branch=pd_by_branch,
        sources=sources,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError(f"output path must be a real directory: {output}")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output directory already exists (use --overwrite): {output}"
        )
    staging = Path(
        tempfile.mkdtemp(prefix=".overlay_0805_compare.", dir=output.parent)
    )
    _register_staging_path(staging)
    rendered: dict[str, VideoMetadata] = {}
    rendered["calibrated_methods_3x2"] = render_comparison_grid_layout(
        _grid_videos(sources, calibrated_entries),
        staging / CALIBRATED_VIDEO_NAME,
        layout=CALIBRATED_GRID,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        font_file=font_file,
        duration_tolerance_s=duration_tolerance_s,
        overwrite=True,
        crf=crf,
        preset=preset,
    )
    rendered["camera_calibration_4x2"] = render_comparison_grid_layout(
        _grid_videos(sources, camera_entries),
        staging / CAMERA_CALIBRATION_VIDEO_NAME,
        layout=CAMERA_CALIBRATION_GRID,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        font_file=font_file,
        duration_tolerance_s=duration_tolerance_s,
        overwrite=True,
        crf=crf,
        preset=preset,
    )
    dual_grid_report: dict[str, object]
    if stereo_available:
        rendered["dual_camera_3x2"] = render_comparison_grid_layout(
            _grid_videos(sources, dual_entries),
            staging / DUAL_VIDEO_NAME,
            layout=DUAL_GRID,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            font_file=font_file,
            duration_tolerance_s=duration_tolerance_s,
            overwrite=True,
            crf=crf,
            preset=preset,
        )
        dual_grid_report = {
            "rendered": True,
            "video": DUAL_VIDEO_NAME,
            "layout": [
                [entry[1] for entry in dual_entries[:3]],
                [entry[1] for entry in dual_entries[3:]],
            ],
        }
    else:
        dual_grid_report = {
            "rendered": False,
            "video": None,
            "reason": "stereo_visibility is not complete in both branches",
        }

    extended_grid_report: dict[str, object]
    if extended_enabled:
        rendered["overlay_history_4x4"] = render_comparison_grid_layout(
            _grid_videos(sources, extended_entries),
            staging / EXTENDED_VIDEO_NAME,
            layout=EXTENDED_GRID,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            font_file=font_file,
            duration_tolerance_s=duration_tolerance_s,
            overwrite=True,
            crf=crf,
            preset=preset,
        )
        extended_grid_report = {
            "rendered": True,
            "video": EXTENDED_VIDEO_NAME,
            "layout": [
                [entry[1] for entry in extended_entries[row : row + 4]]
                for row in range(0, len(extended_entries), 4)
            ],
            "terminology": (
                "dense object geometry is 2.5D camera-Z, not watertight 3-D "
                "collision geometry"
            ),
        }
    else:
        extended_grid_report = {
            "rendered": False,
            "video": None,
            "mode": extended_grid,
            "missing_variants": extended_missing,
            "reason": (
                "disabled by option"
                if extended_grid == "off"
                else "not all conventional extended sources are complete"
            ),
        }

    report = {
        "schema_version": 1,
        "comparison": "08-05 robot overlay methods, camera evidence, and focal calibration",
        "camera2_is_final_view": True,
        "camera_final_variant": selected_final,
        "calibrated_methods": {
            "video": CALIBRATED_VIDEO_NAME,
            "layout": [
                [entry[1] for entry in calibrated_entries[:3]],
                [entry[1] for entry in calibrated_entries[3:]],
            ],
        },
        "camera_calibration": {
            "video": CAMERA_CALIBRATION_VIDEO_NAME,
            "layout": [
                [entry[1] for entry in camera_entries[:4]],
                [entry[1] for entry in camera_entries[4:]],
            ],
        },
        "dual_camera": dual_grid_report,
        "extended_history": extended_grid_report,
        "rendered_metadata": {
            name: _metadata_dict(metadata)
            for name, metadata in rendered.items()
        },
        "sources": {
            branch: {
                variant: _source_record(source)
                for variant, source in branch_sources.items()
            }
            for branch, branch_sources in sources.items()
        },
        "controlled_inputs": {
            "branch_validation": branch_validation,
            "stereo_manifests": {
                branch: str(path) for branch, path in manifests.items()
            },
            "approx_hawor": str(approx_hawor),
            "calibrated_hawor": str(calibrated_hawor),
            "approx_focal_px": approx_focal,
            "calibrated_focal_px": calibrated_focal,
            "approx_sh_hawor": (
                str(approx_sh_hawor) if approx_sh_hawor is not None else None
            ),
            "calibrated_sh_hawor": (
                str(calibrated_sh_hawor)
                if calibrated_sh_hawor is not None
                else None
            ),
            "approx_sh_focal_px": approx_sh_focal,
            "calibrated_sh_focal_px": calibrated_sh_focal,
            "calibration_manifest_contract": calibration_manifest_contract,
            "source_video_metadata": _metadata_dict(source_reference),
            "source_rgb_identity": source_rgb_identity,
            "final_variant_availability": final_availability,
            "extended_grid_mode": extended_grid,
            "extended_grid_missing_variants": extended_missing,
        },
        "mask_statistics": {
            "within_branch": within_branch_statistics,
            "calibrated_minus_approx": cross_calibration_statistics,
            "directional_definition": (
                "added = second & ~first; removed = first & ~second"
            ),
        },
        "invariants": {
            "all_grid_inputs_share_frame_count_fps_and_duration": True,
            "all_variant_masks_match_video_shape_and_report_counts": True,
            "raw_variant_uses_zero_occlusion_mask_for_statistics": True,
            "all_contact_method_panels_share_robot_overlay_and_primary_inputs": True,
            "dual_contact_variants_share_auxiliary_source_and_temporal_offset": True,
            "object3d_ablation_and_force_temporal_share_surface": True,
            "barrier_uses_force_temporal_baseline": True,
            "barrier_does_not_modify_pose_or_claim_metric_collision": True,
            "stereo_visibility_requires_true_dual_masks_and_contacts": stereo_available,
            "calibration_manifests_share_episode_and_source_pairing": True,
            "calibrated_hawor_focals_match_manifest_per_view": (
                stereo_available
            ),
            "approx_and_calibrated_original_rgb_exact_equal": (
                source_rgb_identity.get("exact_equal") is True
                if validate_source_rgb_identity
                else None
            ),
            "source_directories_are_read_only": True,
            "publication_is_atomic": True,
        },
        "interpretation": {
            "mask_counts_are_not_accuracy": True,
            "object_3d_is_camera_z_2p5d_not_watertight_collision_geometry": True,
            "dual_haco_auxiliary_role": "same-finger confidence; MH owns output geometry",
            "calibration_scope": (
                "per-view HaWoR/overlay scalar focal; distortion and stereo "
                "R/T are not consumed"
            ),
        },
    }
    (staging / REPORT_NAME).write_text(json.dumps(report, indent=2) + "\n")
    publish_directory(staging, output)
    print(f"[ok] calibrated methods: {output / CALIBRATED_VIDEO_NAME}", flush=True)
    print(
        f"[ok] camera/calibration: {output / CAMERA_CALIBRATION_VIDEO_NAME}",
        flush=True,
    )
    if stereo_available:
        print(f"[ok] dual camera: {output / DUAL_VIDEO_NAME}", flush=True)
    if extended_enabled:
        print(f"[ok] extended history: {output / EXTENDED_VIDEO_NAME}", flush=True)
    print(f"[ok] report: {output / REPORT_NAME}", flush=True)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approx_pd", "--approx-pd", type=Path, required=True)
    parser.add_argument(
        "--calibrated_pd", "--calibrated-pd", type=Path, required=True
    )
    parser.add_argument("--out_dir", "--out-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing comparison output directory",
    )
    parser.add_argument(
        "--source-dir",
        action="append",
        nargs=3,
        metavar=("BRANCH", "VARIANT", "DIRECTORY"),
        help=(
            "override one conventional source directory; BRANCH is approx or "
            "calibrated and VARIANT is one of: " + ", ".join(VARIANT_SPECS)
        ),
    )
    parser.add_argument(
        "--camera-final-variant",
        choices=("auto", "stereo_visibility", "barrier"),
        default="auto",
        help="fourth camera/calibration column; auto prefers stereo visibility",
    )
    parser.add_argument(
        "--extended-grid",
        choices=("auto", "required", "off"),
        default="auto",
        help=(
            "4x4 history grid: auto renders when all conventional sources "
            "exist, required fails on missing sources, off disables it"
        ),
    )
    parser.add_argument(
        "--skip-source-rgb-identity",
        action="store_true",
        help="skip decoded equality validation of approx/calibrated video_L.mp4",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--font-file", type=Path, default=None)
    parser.add_argument(
        "--duration-tolerance",
        type=float,
        default=DEFAULT_DURATION_TOLERANCE_S,
    )
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        overrides = parse_source_overrides(args.source_dir)
        run_comparison(
            approx_pd=args.approx_pd,
            calibrated_pd=args.calibrated_pd,
            out_dir=args.out_dir,
            source_overrides=overrides,
            camera_final_variant=args.camera_final_variant,
            extended_grid=args.extended_grid,
            validate_source_rgb_identity=not args.skip_source_rgb_identity,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            font_file=args.font_file,
            duration_tolerance_s=args.duration_tolerance,
            crf=args.crf,
            preset=args.preset,
            overwrite=args.overwrite,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
