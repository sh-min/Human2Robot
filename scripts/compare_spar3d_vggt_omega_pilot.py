#!/usr/bin/env python3
"""Create a static, provenance-checked SPAR3D/VGGT-Omega comparison.

This tool deliberately compares *representations*, not geometric accuracy:

* A is a single-MH learned SPAR3D mesh with approximate MH proxy registration.
  Unseen/backside geometry remains a learned estimate.
* B is the official confidence/depth-edge rule at global p20, scoped to the
  MH+SH dual object-mask union (not the exact full-scene demo output).
* C is the existing custom safe adaptive-p50 MH+SH object-point path with
  per-view rescue.  Both VGGT variants remain relative-scale colored point
  sets, not meshes, calibrated geometry, or physical collision models.

Missing upstream bundles are a normal orchestration state.  Preflight reports
``waiting_for_upstream_artifacts`` and exits successfully until both bundles
exist.  Once primary reports exist, every declared upstream file is checked by
path, byte count, and SHA-256 before rendering and again immediately before an
atomic directory publish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = REPO_ROOT / "8-5" / "mesh_sota_pilot" / "episode_1" / "choco"
DEFAULT_SPAR_REPORT = PILOT_ROOT / "spar3d" / "report.json"
DEFAULT_REGISTRATION_REPORT = PILOT_ROOT / "spar3d_registered_mh" / "report.json"
DEFAULT_REGISTERED_MESH = (
    PILOT_ROOT / "spar3d_registered_mh" / "registered_mesh_mh.glb"
)
DEFAULT_VGGT_METADATA = PILOT_ROOT / "vggt_omega" / "metadata.json"
DEFAULT_OUTPUT = PILOT_ROOT / "spar3d_vggt_omega_comparison"

SPAR_METHOD = "spar3d_direct_prepared_rgba_low_vram"
REGISTRATION_METHOD = "spar3d_canonical_to_mh_approximate_sim3_silhouette_depth_fit"
REGISTRATION_REPRESENTATION = (
    "learned_single_view_mesh_approximately_registered_to_mh_camera"
)
VGGT_AGGREGATION_METHOD = "dual_view_mask_filtered_point_aggregation"
OUTPUT_NAMES = {
    "contact_sheet.png",
    "static_diagnostic.mp4",
    "report.json",
    "publish_manifest.json",
}


class ComparisonError(RuntimeError):
    """Base class for safe comparison failures."""


class ComparisonInputError(ComparisonError):
    """Raised when an upstream report or artifact violates its contract."""


class UnsafeOutputError(ComparisonError):
    """Raised when the requested output may overwrite upstream/user data."""


class PublishError(ComparisonError):
    """Raised when transactional publication cannot be completed safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


def file_identity_record(path: str | Path) -> dict[str, Any]:
    """Hash a regular file and reject mutation/replacement during hashing."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ComparisonInputError(f"required file is missing: {resolved}")
    before = _stat_identity(resolved)
    digest = sha256_file(resolved)
    after = _stat_identity(resolved)
    if before != after:
        raise ComparisonInputError(f"file changed while being hashed: {resolved}")
    return {
        "path": str(resolved),
        "bytes": before[2],
        "sha256": digest,
        "stat_identity": list(before),
    }


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonInputError(f"invalid {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ComparisonInputError(f"{description} root must be a JSON object")
    return value


def _declared_path(value: Any, *, base: Path, description: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ComparisonInputError(f"{description}.path must be a non-empty string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _declared_bytes(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ComparisonInputError(f"{description}.bytes must be a non-negative integer")
    return value


def _declared_sha(value: Any, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ComparisonInputError(f"{description}.sha256 must be a 64-character digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ComparisonInputError(f"{description}.sha256 is not hexadecimal") from exc
    return value.lower()


def validate_declared_file_record(
    record: Any,
    *,
    base: Path,
    description: str,
    expected_path: str | Path | None = None,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ComparisonInputError(f"{description} must be a file record object")
    path = _declared_path(record.get("path"), base=base, description=description)
    if expected_path is not None and path != Path(expected_path).expanduser().resolve():
        raise ComparisonInputError(
            f"{description} path is rebound: {path} != {Path(expected_path).expanduser().resolve()}"
        )
    declared_bytes = _declared_bytes(record.get("bytes"), description)
    declared_sha = _declared_sha(record.get("sha256"), description)
    actual = file_identity_record(path)
    if actual["bytes"] != declared_bytes:
        raise ComparisonInputError(
            f"{description} byte mismatch: {actual['bytes']} != {declared_bytes}"
        )
    if actual["sha256"] != declared_sha:
        raise ComparisonInputError(f"{description} SHA-256 mismatch: {path}")
    return actual


def _validate_path_sha_record(
    record: Any, *, base: Path, description: str, expected_path: Path | None = None
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ComparisonInputError(f"{description} must be an object")
    path = _declared_path(record.get("path"), base=base, description=description)
    if expected_path is not None and path != expected_path.resolve():
        raise ComparisonInputError(f"{description} path mismatch: {path} != {expected_path}")
    declared_sha = _declared_sha(record.get("sha256"), description)
    actual = file_identity_record(path)
    if actual["sha256"] != declared_sha:
        raise ComparisonInputError(f"{description} SHA-256 mismatch: {path}")
    return actual


def _require(value: bool, message: str) -> None:
    if not value:
        raise ComparisonInputError(message)


def _require_false(payload: Mapping[str, Any], key: str, description: str) -> None:
    if payload.get(key) is not False:
        raise ComparisonInputError(f"{description}.{key} must be explicitly false")


def _selection_tuple(selection: Any, description: str) -> tuple[str, str, int, int]:
    if not isinstance(selection, Mapping):
        raise ComparisonInputError(f"{description} selection must be an object")
    episode = str(selection.get("episode", ""))
    label = str(selection.get("object_label", ""))
    mh = selection.get("mh_frame_index")
    sh = selection.get("sh_frame_index")
    if isinstance(mh, bool) or not isinstance(mh, int):
        raise ComparisonInputError(f"{description} mh_frame_index must be an integer")
    if isinstance(sh, bool) or not isinstance(sh, int):
        raise ComparisonInputError(f"{description} sh_frame_index must be an integer")
    if label.casefold() != "choco":
        raise ComparisonInputError(f"{description} object_label must be Choco")
    return episode, label.casefold(), mh, sh


def _append_unique(records: list[dict[str, Any]], record: dict[str, Any]) -> None:
    if not any(item["path"] == record["path"] for item in records):
        records.append(record)


def _validate_spar_report(path: Path) -> dict[str, Any]:
    report_path = path.expanduser().resolve()
    report_identity = file_identity_record(report_path)
    report = _read_json(report_path, "SPAR3D report")
    _require(report.get("schema_version") == 1, "SPAR3D schema_version must be 1")
    _require(report.get("status") == "complete", "SPAR3D status must be complete")
    _require(report.get("method") == SPAR_METHOD, "unexpected SPAR3D method")
    _require(
        report.get("representation") == "learned_single_image_canonical_mesh_and_point_cloud",
        "unexpected SPAR3D representation",
    )
    _require_false(report, "metric_scale_verified", "SPAR3D report")
    _require_false(report, "physical_geometry_guarantee", "SPAR3D report")
    _require_false(report, "collision_ready", "SPAR3D report")
    _require(report.get("camera_alignment") == "none", "raw SPAR3D camera_alignment must be none")
    selection = _selection_tuple(report.get("selection"), "SPAR3D")

    records: list[dict[str, Any]] = [report_identity]
    input_record = validate_declared_file_record(
        report.get("input"), base=report_path.parent, description="SPAR3D input"
    )
    _append_unique(records, input_record)
    manifest_record = _validate_path_sha_record(
        report.get("input_manifest"),
        base=report_path.parent,
        description="SPAR3D input_manifest",
    )
    _append_unique(records, manifest_record)
    outputs = report.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ComparisonInputError("SPAR3D outputs must be an object")
    validated_outputs: dict[str, dict[str, Any]] = {}
    for key in ("mesh_glb", "points_ply"):
        validated_outputs[key] = validate_declared_file_record(
            outputs.get(key), base=report_path.parent, description=f"SPAR3D outputs.{key}"
        )
        _append_unique(records, validated_outputs[key])
    raw_self = outputs.get("report")
    if not isinstance(raw_self, str) or _declared_path(
        raw_self, base=report_path.parent, description="SPAR3D outputs.report"
    ) != report_path:
        raise ComparisonInputError("SPAR3D outputs.report does not bind the supplied report")
    return {
        "path": report_path,
        "report": report,
        "identity": report_identity,
        "selection": selection,
        "outputs": validated_outputs,
        "snapshot_records": records,
    }


def _validate_registration_report(
    path: Path, registered_mesh: Path, spar: Mapping[str, Any]
) -> dict[str, Any]:
    report_path = path.expanduser().resolve()
    report_identity = file_identity_record(report_path)
    report = _read_json(report_path, "SPAR3D registration report")
    _require(report.get("schema_version") == 1, "registration schema_version must be 1")
    _require(report.get("status") == "complete", "registration status must be complete")
    _require(report.get("method") == REGISTRATION_METHOD, "unexpected registration method")
    _require(
        report.get("representation") == REGISTRATION_REPRESENTATION,
        "unexpected registration representation",
    )
    _require_false(report, "metric_scale_verified", "registration report")
    _require_false(report, "physical_geometry_guarantee", "registration report")
    _require_false(report, "collision_ready", "registration report")
    _require(
        report.get("uses_sh_for_this_registration") is False,
        "registration must state that SH was not used",
    )
    _require(
        report.get("camera_alignment") == "approximate_MH_camera_Sim3",
        "registration alignment must remain explicitly approximate",
    )
    selection = _selection_tuple(report.get("selection"), "registration")
    _require(selection == spar["selection"], "SPAR3D and registration selections differ")

    source_mesh = validate_declared_file_record(
        report.get("source_mesh"),
        base=report_path.parent,
        description="registration source_mesh",
        expected_path=spar["outputs"]["mesh_glb"]["path"],
    )
    _require(
        source_mesh["sha256"] == spar["outputs"]["mesh_glb"]["sha256"],
        "registration source mesh is not the reported SPAR3D mesh",
    )
    source_report = report.get("source_spar3d_report")
    if not isinstance(source_report, Mapping):
        raise ComparisonInputError("registration source_spar3d_report must be an object")
    source_report_path = _declared_path(
        source_report.get("path"), base=report_path.parent, description="source_spar3d_report"
    )
    _require(source_report_path == spar["path"], "registration is bound to another SPAR3D report")
    _require(
        _declared_sha(source_report.get("sha256"), "source_spar3d_report")
        == spar["identity"]["sha256"],
        "registration SPAR3D report digest mismatch",
    )
    _require(
        source_report.get("hidden_geometry_is_learned_estimate") is True,
        "registration must identify hidden SPAR3D geometry as a learned estimate",
    )

    records: list[dict[str, Any]] = [report_identity, source_mesh]
    sources = report.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ComparisonInputError("registration sources must be a non-empty object")
    validated_sources: dict[str, dict[str, Any]] = {}
    for key, value in sources.items():
        validated_sources[str(key)] = validate_declared_file_record(
            value, base=report_path.parent, description=f"registration sources.{key}"
        )
        _append_unique(records, validated_sources[str(key)])

    outputs = report.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ComparisonInputError("registration outputs must be an object")
    validated_outputs: dict[str, dict[str, Any]] = {}
    for key, value in outputs.items():
        if key == "report.json":
            if not isinstance(value, Mapping):
                raise ComparisonInputError("registration outputs.report.json must be an object")
            self_path = _declared_path(
                value.get("path"), base=report_path.parent, description="registration report output"
            )
            _require(self_path == report_path, "registration report output path is rebound")
            continue
        validated_outputs[str(key)] = validate_declared_file_record(
            value,
            base=report_path.parent,
            description=f"registration outputs.{key}",
        )
        _append_unique(records, validated_outputs[str(key)])
    for required in (
        "registered_mesh_mh.glb",
        "canonical_turntable_contact_sheet.png",
        "before_after_registration.png",
    ):
        _require(required in validated_outputs, f"registration output is missing {required}")
    expected_mesh = registered_mesh.expanduser().resolve()
    _require(
        Path(validated_outputs["registered_mesh_mh.glb"]["path"]) == expected_mesh,
        "--registered-mesh is not the registration report's registered mesh",
    )
    sim3 = report.get("sim3")
    _require(isinstance(sim3, Mapping), "registration sim3 must be an object")
    _require_false(sim3, "metric_scale_verified", "registration sim3")
    return {
        "path": report_path,
        "report": report,
        "identity": report_identity,
        "selection": selection,
        "sources": validated_sources,
        "outputs": validated_outputs,
        "snapshot_records": records,
    }


def _load_npy(path: Path, description: str) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ComparisonInputError(f"cannot load {description} {path}: {exc}") from exc
    if not isinstance(value, np.ndarray):
        raise ComparisonInputError(f"{description} is not an ndarray")
    return value


def _load_point_evidence(path: Path, description: str) -> dict[str, Any]:
    required = {
        "points_relative",
        "colors_rgb",
        "depth_confidence",
        "input_depth_relative",
        "view_indices",
        "camera_order",
    }
    try:
        with np.load(path, allow_pickle=False) as loaded:
            missing = sorted(required - set(loaded.files))
            if missing:
                raise ComparisonInputError(
                    f"{description} is missing arrays: {', '.join(missing)}"
                )
            values = {key: np.array(loaded[key]) for key in required}
    except (OSError, ValueError) as exc:
        raise ComparisonInputError(
            f"cannot load {description} {path}: {exc}"
        ) from exc

    points = values["points_relative"]
    colors = values["colors_rgb"]
    confidence = values["depth_confidence"]
    depth = values["input_depth_relative"]
    view_indices = values["view_indices"]
    camera_order = values["camera_order"]
    _require(
        camera_order.shape == (2,) and camera_order.tolist() == ["MH", "SH"],
        f"{description} camera_order must be [MH, SH]",
    )
    _require(
        points.ndim == 2 and points.shape[1] == 3 and len(points) > 0,
        f"{description} points_relative must have non-empty shape (N,3)",
    )
    _require(
        np.issubdtype(points.dtype, np.floating),
        f"{description} points_relative must be floating point",
    )
    count = len(points)
    _require(colors.shape == (count, 3), f"{description} colors_rgb shape mismatch")
    _require(
        colors.dtype == np.uint8,
        f"{description} colors_rgb must use uint8",
    )
    _require(
        confidence.shape == (count,),
        f"{description} depth_confidence shape mismatch",
    )
    _require(
        np.issubdtype(confidence.dtype, np.floating),
        f"{description} depth_confidence must be floating point",
    )
    _require(
        depth.shape == (count,),
        f"{description} input_depth_relative shape mismatch",
    )
    _require(
        np.issubdtype(depth.dtype, np.floating),
        f"{description} input_depth_relative must be floating point",
    )
    _require(
        view_indices.shape == (count,),
        f"{description} view_indices shape mismatch",
    )
    _require(np.isfinite(points).all(), f"{description} contains non-finite points")
    _require(
        np.isfinite(confidence).all() and bool(np.all(confidence > 1.0e-5)),
        f"{description} confidence must be finite and positive",
    )
    _require(
        np.isfinite(depth).all() and bool(np.all(depth > 1.0e-6)),
        f"{description} input depth must be finite and positive",
    )
    _require(
        view_indices.dtype != np.bool_
        and np.issubdtype(view_indices.dtype, np.integer),
        f"{description} view_indices must be integers",
    )
    unique_views = set(view_indices.tolist())
    _require(
        unique_views.issubset({0, 1}),
        f"{description} view_indices contain an unknown camera",
    )
    counts = {
        camera: int(np.count_nonzero(view_indices == index))
        for index, camera in enumerate(("MH", "SH"))
    }
    return {
        "points": points.astype(np.float32, copy=False),
        "colors": colors.astype(np.uint8, copy=False),
        "confidence": confidence.astype(np.float32, copy=False),
        "depth": depth.astype(np.float32, copy=False),
        "view_indices": view_indices.astype(np.int16, copy=False),
        "camera_order": ["MH", "SH"],
        "count": count,
        "counts_by_view": counts,
    }


def _validate_vggt_metadata(path: Path, expected_selection: tuple[str, str, int, int]) -> dict[str, Any]:
    metadata_path = path.expanduser().resolve()
    identity = file_identity_record(metadata_path)
    metadata = _read_json(metadata_path, "VGGT-Omega metadata")
    _require(metadata.get("schema_version") == 1, "VGGT metadata schema_version must be 1")
    _require(metadata.get("status") == "completed", "VGGT metadata status must be completed")
    geometry = metadata.get("geometry_contract")
    if not isinstance(geometry, Mapping):
        raise ComparisonInputError("VGGT geometry_contract must be an object")
    expected_geometry = {
        "representation": "colored_point_cloud",
        "primitive": "points",
        "has_triangle_faces": False,
        "is_triangle_mesh": False,
        "is_watertight": False,
        "collision_ready": False,
        "scale": "relative_non_metric",
        "metric_scale_verified": False,
        "provided_calibration_applied": False,
    }
    for key, expected in expected_geometry.items():
        _require(
            geometry.get(key) == expected,
            f"VGGT geometry_contract.{key} must be {expected!r}",
        )
    aggregation = metadata.get("object_aggregation")
    if not isinstance(aggregation, Mapping):
        raise ComparisonInputError("VGGT object_aggregation must be an object")
    _require(
        aggregation.get("method") == VGGT_AGGREGATION_METHOD,
        "VGGT object aggregation method mismatch",
    )
    _require(aggregation.get("requested") is True, "VGGT object aggregation was not requested")
    _require(aggregation.get("performed") is True, "VGGT object aggregation was not performed")
    _require(aggregation.get("status") == "completed", "VGGT object aggregation is incomplete")
    official_reference = metadata.get("official_reference")
    if not isinstance(official_reference, Mapping):
        raise ComparisonInputError("VGGT official_reference must be an object")
    _require(
        official_reference.get("conversion_function")
        == "visual_util.predictions_to_glb",
        "VGGT official full-scene conversion function mismatch",
    )
    _require(
        official_reference.get("repository") == "facebook/VGGT-Omega"
        and isinstance(official_reference.get("local_code_commit"), str)
        and len(str(official_reference.get("local_code_commit"))) == 40,
        "VGGT official full-scene repository/commit provenance mismatch",
    )
    _require(
        official_reference.get("call_contract")
        == "predictions_to_glb(predictions_np) with unmodified official defaults",
        "VGGT official full-scene call contract mismatch",
    )
    _require(
        official_reference.get("exact_official_demo_output") is True,
        "VGGT official full-scene artifact must be exact official demo output",
    )
    official_full_parameters = official_reference.get("full_scene_parameters")
    if not isinstance(official_full_parameters, Mapping):
        raise ComparisonInputError(
            "VGGT official_reference.full_scene_parameters must be an object"
        )
    _require(
        official_full_parameters.get("confidence_percentile") == 20.0,
        "VGGT official full-scene confidence percentile must be 20",
    )
    _require(
        official_full_parameters.get("depth_edge_rtol") == 0.03,
        "VGGT official full-scene depth-edge rtol must be 0.03",
    )
    _require(
        official_full_parameters.get("show_cam") is True
        and official_full_parameters.get("scene_alignment")
        == "official_first_camera_opengl",
        "VGGT official full-scene camera/alignment semantics mismatch",
    )
    _require(
        official_full_parameters.get("depth_edge_filter") is True
        and official_full_parameters.get("mask_black_bg") is False
        and official_full_parameters.get("mask_white_bg") is False
        and official_full_parameters.get("mask_sky") is False
        and official_full_parameters.get("max_points") == 300000,
        "VGGT official full-scene default parameters mismatch",
    )

    official_object = aggregation.get("official_global_p20")
    if not isinstance(official_object, Mapping):
        raise ComparisonInputError(
            "VGGT object_aggregation.official_global_p20 must be an object"
        )
    _require(
        official_object.get("method")
        == "official_rule_global_p20_scoped_to_dual_object_mask_union",
        "VGGT official-rule p20 object method mismatch",
    )
    _require(
        official_object.get("status") == "completed",
        "VGGT official-rule p20 object status must be completed",
    )
    _require(
        official_object.get("exact_official_demo_output") is False,
        "VGGT object p20 must not claim to be exact official demo output",
    )
    _require(
        official_object.get("per_view_threshold_fallback") is False,
        "VGGT official-rule p20 object path must disable per-view fallback",
    )
    _require(
        official_object.get("threshold_population")
        == "finite dual object-mask union with depth-edge confidences zeroed",
        "VGGT official-rule p20 threshold population mismatch",
    )
    official_filter = official_object.get("point_filter")
    if not isinstance(official_filter, Mapping):
        raise ComparisonInputError("VGGT official-rule p20 point_filter missing")
    _require(
        official_filter.get("policy")
        == "official_rule_global_p20_scoped_to_dual_object_mask_union_v1",
        "VGGT official-rule p20 policy mismatch",
    )
    _require(
        official_filter.get("confidence_percentile") == 20.0,
        "VGGT official-rule object confidence percentile must be 20",
    )
    _require(
        official_filter.get("per_view_threshold_fallback") is False,
        "VGGT official-rule point filter must disable per-view fallback",
    )

    custom_object = aggregation.get("custom_adaptive_p50")
    if not isinstance(custom_object, Mapping):
        raise ComparisonInputError(
            "VGGT object_aggregation.custom_adaptive_p50 must be an object"
        )
    _require(
        custom_object.get("method") == "safe_per_view_adaptive_v1",
        "VGGT custom adaptive p50 method mismatch",
    )
    _require(
        custom_object.get("status") == "completed",
        "VGGT custom adaptive p50 status must be completed",
    )
    custom_filter = custom_object.get("point_filter")
    if not isinstance(custom_filter, Mapping):
        raise ComparisonInputError("VGGT custom adaptive p50 point_filter missing")
    _require(
        custom_filter.get("policy") == "safe_per_view_adaptive_v1"
        and custom_filter.get("confidence_percentile") == 50.0,
        "VGGT custom adaptive p50 policy mismatch",
    )

    input_info = metadata.get("input")
    if not isinstance(input_info, Mapping):
        raise ComparisonInputError("VGGT input must be an object")
    _require(str(input_info.get("object_label", "")).casefold() == "choco", "VGGT label must be Choco")
    _require(str(input_info.get("episode", "")) == expected_selection[0], "VGGT episode differs")
    _require(input_info.get("camera_order") == ["MH", "SH"], "VGGT camera_order must be [MH, SH]")
    views = input_info.get("views")
    if not isinstance(views, list) or len(views) != 2:
        raise ComparisonInputError("VGGT must contain exactly two MH/SH views")
    by_camera: dict[str, Mapping[str, Any]] = {}
    for view in views:
        if not isinstance(view, Mapping) or view.get("camera") not in ("MH", "SH"):
            raise ComparisonInputError("VGGT views must be named MH and SH")
        by_camera[str(view["camera"])] = view
    _require(set(by_camera) == {"MH", "SH"}, "VGGT views must contain one MH and one SH")
    _require(by_camera["MH"].get("frame_index") == expected_selection[2], "VGGT MH frame differs")
    _require(by_camera["SH"].get("frame_index") == expected_selection[3], "VGGT SH frame differs")

    records: list[dict[str, Any]] = [identity]
    manifest = validate_declared_file_record(
        input_info.get("manifest"), base=metadata_path.parent, description="VGGT input.manifest"
    )
    _append_unique(records, manifest)
    validated_views: dict[str, dict[str, dict[str, Any]]] = {}
    for camera in ("MH", "SH"):
        validated_views[camera] = {}
        for key in ("image", "source_image", "object_mask"):
            record = by_camera[camera].get(key)
            validated = validate_declared_file_record(
                record,
                base=metadata_path.parent,
                description=f"VGGT input.views.{camera}.{key}",
            )
            validated_views[camera][key] = validated
            _append_unique(records, validated)
    mask_paths = aggregation.get("mask_paths")
    if mask_paths is not None:
        if not isinstance(mask_paths, Mapping):
            raise ComparisonInputError("VGGT object_aggregation.mask_paths must be an object")
        for camera in ("MH", "SH"):
            declared_mask = _declared_path(
                mask_paths.get(camera),
                base=metadata_path.parent,
                description=f"VGGT object_aggregation.mask_paths.{camera}",
            )
            _require(
                declared_mask == Path(validated_views[camera]["object_mask"]["path"]),
                f"VGGT {camera} aggregation mask is rebound",
            )
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ComparisonInputError("VGGT artifacts must be a non-empty object")
    validated_artifacts: dict[str, dict[str, Any]] = {}
    for key, value in artifacts.items():
        validated_artifacts[str(key)] = validate_declared_file_record(
            value, base=metadata_path.parent, description=f"VGGT artifacts.{key}"
        )
        _append_unique(records, validated_artifacts[str(key)])
    required = {
        "world_points",
        "preprocessed_images",
        "object_masks_model_input",
        "official_full_scene_glb",
        "official_object_point_cloud_ply",
        "official_object_point_cloud_glb",
        "official_object_point_evidence",
        "object_point_cloud_ply",
        "object_point_cloud_glb",
        "object_point_evidence",
    }
    missing = sorted(required - set(validated_artifacts))
    _require(not missing, f"VGGT artifacts are missing: {', '.join(missing)}")

    world = _load_npy(Path(validated_artifacts["world_points"]["path"]), "VGGT world points")
    images = _load_npy(Path(validated_artifacts["preprocessed_images"]["path"]), "VGGT images")
    masks = _load_npy(Path(validated_artifacts["object_masks_model_input"]["path"]), "VGGT masks")
    official_evidence = _load_point_evidence(
        Path(validated_artifacts["official_object_point_evidence"]["path"]),
        "VGGT official-rule p20 evidence",
    )
    custom_evidence = _load_point_evidence(
        Path(validated_artifacts["object_point_evidence"]["path"]),
        "VGGT custom adaptive p50 evidence",
    )
    _require(world.ndim == 4 and world.shape[0] == 2 and world.shape[-1] == 3, "world points must have shape (2,H,W,3)")
    height, width = int(world.shape[1]), int(world.shape[2])
    _require(images.shape == (2, 3, height, width), "preprocessed images must have shape (2,3,H,W)")
    _require(masks.shape == (2, height, width), "object masks must have shape (2,H,W)")
    if masks.dtype != np.bool_:
        _require(np.issubdtype(masks.dtype, np.number), "object masks must be bool or binary numeric")
        _require(np.isfinite(masks).all(), "object masks contain non-finite values")
        unique = np.unique(masks)
        _require(set(unique.tolist()).issubset({0, 1}), "object masks are not binary")
    masks_bool = masks.astype(bool, copy=False)
    _require(np.isfinite(images).all(), "preprocessed images contain non-finite values")
    for index, camera in enumerate(("MH", "SH")):
        _require(bool(masks_bool[index].any()), f"VGGT {camera} object mask is empty")
        selected = world[index][masks_bool[index]]
        _require(selected.size > 0 and np.isfinite(selected).all(), f"VGGT {camera} masked points are empty or non-finite")

    def bind_inner_artifacts(
        payload: Mapping[str, Any],
        bindings: Mapping[str, str],
        description: str,
    ) -> None:
        inner = payload.get("artifacts")
        if not isinstance(inner, Mapping):
            raise ComparisonInputError(f"{description}.artifacts must be an object")
        for inner_key, top_key in bindings.items():
            value = inner.get(inner_key)
            _require(
                isinstance(value, str)
                and _declared_path(
                    value,
                    base=metadata_path.parent,
                    description=f"{description}.artifacts.{inner_key}",
                )
                == Path(validated_artifacts[top_key]["path"]),
                f"{description}.artifacts.{inner_key} is rebound",
            )

    bind_inner_artifacts(
        official_object,
        {
            "point_cloud_ply": "official_object_point_cloud_ply",
            "point_cloud_glb": "official_object_point_cloud_glb",
            "evidence": "official_object_point_evidence",
        },
        "VGGT official_global_p20",
    )
    bind_inner_artifacts(
        custom_object,
        {
            "point_cloud_ply": "object_point_cloud_ply",
            "point_cloud_glb": "object_point_cloud_glb",
            "evidence": "object_point_evidence",
        },
        "VGGT custom_adaptive_p50",
    )
    _require(
        isinstance(official_reference.get("full_scene_artifact"), str)
        and _declared_path(
            official_reference.get("full_scene_artifact"),
            base=metadata_path.parent,
            description="VGGT official_reference.full_scene_artifact",
        )
        == Path(validated_artifacts["official_full_scene_glb"]["path"]),
        "VGGT official full-scene artifact is rebound",
    )
    for description, evidence, point_filter in (
        ("official-rule p20", official_evidence, official_filter),
        ("custom adaptive p50", custom_evidence, custom_filter),
    ):
        reported_counts = point_filter.get("exported_points_by_view")
        _require(
            reported_counts == evidence["counts_by_view"],
            f"VGGT {description} evidence per-view counts mismatch metadata",
        )
        _require(
            point_filter.get("exported_count") == evidence["count"],
            f"VGGT {description} evidence count mismatch metadata",
        )
        _require(
            point_filter.get("all_views_contributed")
            is all(value > 0 for value in evidence["counts_by_view"].values()),
            f"VGGT {description} all_views_contributed mismatch",
        )
    _require(
        aggregation.get("point_filter") == custom_filter,
        "VGGT legacy object point_filter does not bind custom adaptive p50",
    )
    dual_contribution = aggregation.get("dual_view_contribution")
    if not isinstance(dual_contribution, Mapping):
        raise ComparisonInputError("VGGT dual_view_contribution must be an object")
    _require(
        dual_contribution.get("proven") is True
        and dual_contribution.get("exported_points_by_view")
        == custom_evidence["counts_by_view"],
        "VGGT dual_view_contribution does not bind custom adaptive evidence",
    )
    _require(
        isinstance(dual_contribution.get("evidence_artifact"), str)
        and _declared_path(
            dual_contribution.get("evidence_artifact"),
            base=metadata_path.parent,
            description="VGGT dual_view_contribution.evidence_artifact",
        )
        == Path(validated_artifacts["object_point_evidence"]["path"]),
        "VGGT dual_view_contribution evidence artifact is rebound",
    )
    compared_paths = {
        Path(validated_artifacts[key]["path"])
        for key in (
            "official_full_scene_glb",
            "official_object_point_cloud_ply",
            "official_object_point_cloud_glb",
            "official_object_point_evidence",
            "object_point_cloud_ply",
            "object_point_cloud_glb",
            "object_point_evidence",
        )
    }
    _require(
        len(compared_paths) == 7,
        "VGGT official full-scene, official-rule, and custom artifacts must be distinct files",
    )
    _require(
        all(value > 0 for value in custom_evidence["counts_by_view"].values()),
        "VGGT custom adaptive p50 evidence must contain both cameras",
    )
    metadata_self = metadata.get("metadata_path")
    _require(
        isinstance(metadata_self, str)
        and _declared_path(
            metadata_self,
            base=metadata_path.parent,
            description="VGGT metadata_path",
        )
        == metadata_path,
        "VGGT metadata_path does not bind the supplied metadata",
    )
    return {
        "path": metadata_path,
        "metadata": metadata,
        "identity": identity,
        "artifacts": validated_artifacts,
        "views": validated_views,
        "world_points": world.astype(np.float32, copy=False),
        "images": images.astype(np.float32, copy=False),
        "masks": masks_bool,
        "official_evidence": official_evidence,
        "custom_evidence": custom_evidence,
        "official_filter": official_filter,
        "custom_filter": custom_filter,
        "snapshot_records": records,
    }


def snapshot_digest(records: Sequence[Mapping[str, Any]]) -> str:
    canonical = [
        {"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"]}
        for item in sorted(records, key=lambda value: str(value["path"]))
    ]
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify_input_snapshot_unchanged(records: Sequence[Mapping[str, Any]]) -> None:
    for expected in records:
        actual = file_identity_record(expected["path"])
        if (
            actual["bytes"] != expected["bytes"]
            or actual["sha256"] != expected["sha256"]
            or actual["stat_identity"] != expected["stat_identity"]
        ):
            raise ComparisonInputError(f"upstream input changed during comparison: {expected['path']}")


def preflight_job(
    *,
    spar_report: str | Path = DEFAULT_SPAR_REPORT,
    registration_report: str | Path = DEFAULT_REGISTRATION_REPORT,
    registered_mesh: str | Path = DEFAULT_REGISTERED_MESH,
    vggt_metadata: str | Path = DEFAULT_VGGT_METADATA,
) -> dict[str, Any]:
    primary = {
        "spar_report": Path(spar_report).expanduser().resolve(),
        "registration_report": Path(registration_report).expanduser().resolve(),
        "registered_mesh": Path(registered_mesh).expanduser().resolve(),
        "vggt_metadata": Path(vggt_metadata).expanduser().resolve(),
    }
    missing = [str(path) for path in primary.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        return {
            "schema_version": 1,
            "status": "waiting_for_upstream_artifacts",
            "missing": missing,
            "message": "No output was created; run again after both upstream bundles are complete.",
        }
    spar = _validate_spar_report(primary["spar_report"])
    registration = _validate_registration_report(
        primary["registration_report"], primary["registered_mesh"], spar
    )
    vggt = _validate_vggt_metadata(primary["vggt_metadata"], spar["selection"])
    snapshot: list[dict[str, Any]] = []
    for group in (spar["snapshot_records"], registration["snapshot_records"], vggt["snapshot_records"]):
        for record in group:
            _append_unique(snapshot, record)
    return {
        "schema_version": 1,
        "status": "ready",
        "selection": {
            "episode": spar["selection"][0],
            "object_label": "Choco",
            "mh_frame_index": spar["selection"][2],
            "sh_frame_index": spar["selection"][3],
        },
        "spar": spar,
        "registration": registration,
        "vggt": vggt,
        "input_snapshot_records": snapshot,
        "input_snapshot_sha256": snapshot_digest(snapshot),
    }


def _to_uint8_rgb(chw: np.ndarray) -> np.ndarray:
    rgb = np.moveaxis(np.asarray(chw, dtype=np.float32), 0, -1)
    finite = np.isfinite(rgb)
    if not finite.all():
        raise ComparisonInputError("image tensor contains non-finite values")
    low, high = np.percentile(rgb, [1.0, 99.0])
    if high <= low + 1e-8:
        low, high = float(rgb.min()), float(rgb.max())
    if high <= low + 1e-8:
        return np.full(rgb.shape, 127, dtype=np.uint8)
    return np.clip((rgb - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def _fit_image(image: np.ndarray, size_wh: tuple[int, int], background: int = 22) -> np.ndarray:
    width, height = size_wh
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    if image is None or image.size == 0:
        return canvas
    scale = min(width / image.shape[1], height / image.shape[0])
    new_w = max(1, int(round(image.shape[1] * scale)))
    new_h = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def _put_lines(
    image: np.ndarray,
    lines: Iterable[str],
    origin: tuple[int, int],
    *,
    scale: float = 0.48,
    color: tuple[int, int, int] = (235, 235, 235),
    spacing: int = 21,
    thickness: int = 1,
) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(image, str(line), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += spacing


def _tinted_mask_view(image_rgb: np.ndarray, mask: np.ndarray, camera: str) -> np.ndarray:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    overlay[mask] = (
        0.35 * overlay[mask].astype(np.float32) + 0.65 * np.array([30, 190, 255], dtype=np.float32)
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1] - 1, overlay.shape[0] - 1), (90, 190, 255), 2)
    cv2.putText(overlay, f"{camera} mask-filtered evidence", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def _projection_bounds(
    evidence_items: Sequence[Mapping[str, Any]],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    points = np.concatenate(
        [np.asarray(item["points"], dtype=np.float32) for item in evidence_items],
        axis=0,
    )
    projections = ((0, 2), (0, 1))
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for axis_x, axis_y in projections:
        values = points[:, [axis_x, axis_y]]
        lows = np.percentile(values, 1.0, axis=0)
        highs = np.percentile(values, 99.0, axis=0)
        result.append((lows, np.maximum(highs, lows + 1.0e-6)))
    return tuple(result)


def _scatter_panel(
    evidence: Mapping[str, Any],
    size_wh: tuple[int, int],
    *,
    projection_bounds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    width, height = size_wh
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    margin = 38
    half = width // 2
    projections = ((0, 2, "relative X-Z"), (0, 1, "relative X-Y"))
    points_all = np.asarray(evidence["points"], dtype=np.float32)
    view_indices = np.asarray(evidence["view_indices"])
    selected: list[tuple[np.ndarray, int]] = []
    for view in range(2):
        points = points_all[view_indices == view]
        if len(points) > 40000:
            points = points[np.linspace(0, len(points) - 1, 40000, dtype=np.int64)]
        selected.append((points, view))
    colors = ((255, 165, 55), (80, 225, 125))  # BGR: MH, SH
    for panel_index, (axis_x, axis_y, label) in enumerate(projections):
        x0 = panel_index * half
        lows, highs = projection_bounds[panel_index]
        span = np.maximum(highs - lows, 1e-6)
        cv2.rectangle(canvas, (x0 + margin, margin), (x0 + half - margin, height - margin), (70, 70, 70), 1)
        for points, view in selected:
            if len(points) == 0:
                continue
            uv = (points[:, [axis_x, axis_y]] - lows) / span
            px = x0 + margin + np.clip(uv[:, 0], 0, 1) * (half - 2 * margin)
            py = height - margin - np.clip(uv[:, 1], 0, 1) * (height - 2 * margin)
            coords = np.stack([px, py], axis=1).astype(np.int32)
            canvas[coords[:, 1], coords[:, 0]] = colors[view]
        cv2.putText(canvas, label, (x0 + margin, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (235, 235, 235), 1, cv2.LINE_AA)
    _put_lines(canvas, ["MH", "SH"], (width - 104, 24), scale=0.43, spacing=20, color=(225, 225, 225))
    cv2.circle(canvas, (width - 122, 19), 4, colors[0], -1)
    cv2.circle(canvas, (width - 122, 39), 4, colors[1], -1)
    counts = evidence["counts_by_view"]
    cv2.putText(
        canvas,
        f"MH {counts['MH']:,}  SH {counts['SH']:,}",
        (margin, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _panel(title: str, subtitle: str, content: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    width, height = size_wh
    result = np.full((height, width, 3), 24, dtype=np.uint8)
    header = 62
    result[header:] = _fit_image(content, (width, height - header), background=18)
    cv2.rectangle(result, (0, 0), (width - 1, height - 1), (76, 76, 76), 1)
    cv2.putText(result, title, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, subtitle, (14, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 205, 215), 1, cv2.LINE_AA)
    return result


def make_contact_sheet(
    *,
    registration_before_after: str | Path,
    spar_turntable: str | Path,
    official_evidence: Mapping[str, Any],
    custom_evidence: Mapping[str, Any],
    selection: Mapping[str, Any],
    size_wh: tuple[int, int] = (1600, 1000),
) -> np.ndarray:
    """Render a four-panel static comparison with explicit truth labels."""

    width, height = size_wh
    if width < 800 or height < 500:
        raise ValueError("contact sheet must be at least 800x500")
    before_after = cv2.imread(str(registration_before_after), cv2.IMREAD_COLOR)
    turntable = cv2.imread(str(spar_turntable), cv2.IMREAD_COLOR)
    if before_after is None or turntable is None:
        raise ComparisonInputError("registration diagnostic PNG could not be decoded")
    shared_bounds = _projection_bounds((official_evidence, custom_evidence))

    canvas = np.full((height, width, 3), 14, dtype=np.uint8)
    top = 76
    footer = 44
    gap = 8
    panel_w = (width - gap) // 2
    panel_h = (height - top - footer - gap) // 2
    right_panel_w = width - panel_w - gap
    scatter_size = (right_panel_w, panel_h - 62)
    official_scatter = _scatter_panel(
        official_evidence,
        scatter_size,
        projection_bounds=shared_bounds,
    )
    custom_scatter = _scatter_panel(
        custom_evidence,
        scatter_size,
        projection_bounds=shared_bounds,
    )
    panels = (
        _panel("A1  SPAR3D -> approximate MH registration", "single MH; proxy Sim(3); non-metric", before_after, (panel_w, panel_h)),
        _panel(
            "B  VGGT-Omega official-rule p20 object points",
            "dual-mask-union percentile; no per-view fallback; shared plot bounds",
            official_scatter,
            (right_panel_w, panel_h),
        ),
        _panel("A2  Learned canonical mesh views", "backside/unseen surface = learned estimate", turntable, (panel_w, panel_h)),
        _panel(
            "C  VGGT-Omega custom adaptive p50 object points",
            "per-view rescue only when globally starved; shared plot bounds",
            custom_scatter,
            (right_panel_w, panel_h),
        ),
    )
    canvas[top : top + panel_h, :panel_w] = panels[0]
    canvas[top : top + panel_h, panel_w + gap :] = panels[1]
    canvas[top + panel_h + gap : top + 2 * panel_h + gap, :panel_w] = panels[2]
    canvas[top + panel_h + gap : top + 2 * panel_h + gap, panel_w + gap :] = panels[3]
    title = (
        f"STATIC REPRESENTATION DIAGNOSTIC | Choco | MH {selection['mh_frame_index']} | "
        f"SH {selection['sh_frame_index']}"
    )
    cv2.putText(canvas, title, (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.77, (248, 248, 248), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "A: learned mesh   B: official-rule p20 scoped to masks   C: custom adaptive p50",
        (18, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (170, 211, 255),
        1,
        cv2.LINE_AA,
    )
    footer_text = (
        "STATIC DIAGNOSTIC - none of these is metric physical geometry or collision ground truth"
    )
    cv2.putText(canvas, footer_text, (18, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (90, 210, 255), 1, cv2.LINE_AA)
    return canvas


def write_static_diagnostic_video(
    path: str | Path,
    contact_sheet: np.ndarray,
    *,
    fps: int = 10,
    duration_seconds: float = 3.0,
) -> dict[str, Any]:
    """Write a short animated pointer over one static diagnostic image."""

    if fps <= 0 or not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("fps and duration_seconds must be positive")
    output = Path(path)
    frame_count = max(1, int(round(fps * duration_seconds)))
    height, width = contact_sheet.shape[:2]
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise ComparisonError(f"cannot open MP4 writer: {output}")
    try:
        for index in range(frame_count):
            frame = contact_sheet.copy()
            fraction = (index + 1) / frame_count
            cv2.rectangle(frame, (0, 0), (max(1, int(width * fraction)), 5), (0, 190, 255), -1)
            active_panel = (index // max(1, fps)) % 3
            right_midpoint = (74 + height - 45) // 2
            if active_panel == 0:
                cv2.rectangle(frame, (2, 74), (width // 2 - 4, height - 45), (255, 178, 65), 3)
            elif active_panel == 1:
                cv2.rectangle(
                    frame,
                    (width // 2 + 4, 74),
                    (width - 3, right_midpoint - 4),
                    (85, 230, 135),
                    3,
                )
            else:
                cv2.rectangle(
                    frame,
                    (width // 2 + 4, right_midpoint + 4),
                    (width - 3, height - 45),
                    (80, 180, 255),
                    3,
                )
            writer.write(frame)
    finally:
        writer.release()
    if not output.is_file() or output.stat().st_size == 0:
        raise ComparisonError(f"MP4 writer produced no output: {output}")
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
        "semantics": "static_diagnostic_with_visual_pointer_not_temporal_reconstruction",
    }


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _has_symlink_component(path: Path) -> bool:
    candidate = path if path.is_absolute() else Path.cwd() / path
    for item in (candidate, *candidate.parents):
        # ``Path.exists`` is false for a dangling symlink, which is precisely a
        # path that must not become an output redirection later.
        if item.is_symlink():
            return True
    return False


def validate_output_path(output_dir: str | Path, protected_paths: Sequence[str | Path]) -> Path:
    raw = Path(output_dir).expanduser()
    if _has_symlink_component(raw):
        raise UnsafeOutputError(f"output path contains a symlink component: {raw}")
    output = raw.resolve()
    dangerous = {Path("/").resolve(), Path.home().resolve(), REPO_ROOT.resolve(), PILOT_ROOT.resolve()}
    if output in dangerous:
        raise UnsafeOutputError(f"refusing broad output directory: {output}")
    for item in protected_paths:
        protected = Path(item).expanduser().resolve()
        if _paths_overlap(output, protected):
            raise UnsafeOutputError(f"output overlaps upstream input {protected}: {output}")
    if output.exists() and not output.is_dir():
        raise UnsafeOutputError(f"output exists and is not a directory: {output}")
    return output


def _path_replace(source: Path, destination: Path) -> None:
    source.replace(destination)


def _publish_directory(
    staging: Path,
    output: Path,
    *,
    overwrite: bool,
    rename: Callable[[Path, Path], None] = _path_replace,
) -> None:
    """Publish staging atomically, restoring the old directory on failure."""

    backup = output.parent / f".{output.name}.backup-{os.getpid()}"
    if backup.exists():
        raise PublishError(f"stale backup prevents safe publication: {backup}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    moved_old = False
    try:
        if output.exists():
            rename(output, backup)
            moved_old = True
        try:
            rename(staging, output)
        except BaseException as publish_exc:
            if output.exists() and not staging.exists():
                try:
                    rename(output, staging)
                except BaseException:
                    pass
            if moved_old and backup.exists():
                try:
                    rename(backup, output)
                except BaseException as rollback_exc:
                    raise PublishError(
                        f"publish failed and rollback failed; backup retained at {backup}"
                    ) from rollback_exc
            raise PublishError("atomic comparison publish failed; previous output restored") from publish_exc
        if moved_old:
            shutil.rmtree(backup)
    except BaseException:
        if moved_old and backup.exists() and not output.exists():
            try:
                rename(backup, output)
            except BaseException:
                pass
        raise


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _final_output_record(staged_path: Path, final_path: Path) -> dict[str, Any]:
    identity = file_identity_record(staged_path)
    return {"path": str(final_path), "bytes": identity["bytes"], "sha256": identity["sha256"]}


def run_comparison(
    *,
    spar_report: str | Path = DEFAULT_SPAR_REPORT,
    registration_report: str | Path = DEFAULT_REGISTRATION_REPORT,
    registered_mesh: str | Path = DEFAULT_REGISTERED_MESH,
    vggt_metadata: str | Path = DEFAULT_VGGT_METADATA,
    output_dir: str | Path = DEFAULT_OUTPUT,
    overwrite: bool = False,
    fps: int = 10,
    duration_seconds: float = 3.0,
) -> dict[str, Any]:
    preflight = preflight_job(
        spar_report=spar_report,
        registration_report=registration_report,
        registered_mesh=registered_mesh,
        vggt_metadata=vggt_metadata,
    )
    if preflight["status"] != "ready":
        return preflight
    protected = [item["path"] for item in preflight["input_snapshot_records"]]
    output = validate_output_path(output_dir, protected)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        registration_outputs = preflight["registration"]["outputs"]
        sheet = make_contact_sheet(
            registration_before_after=registration_outputs["before_after_registration.png"]["path"],
            spar_turntable=registration_outputs["canonical_turntable_contact_sheet.png"]["path"],
            official_evidence=preflight["vggt"]["official_evidence"],
            custom_evidence=preflight["vggt"]["custom_evidence"],
            selection=preflight["selection"],
        )
        sheet_path = staging / "contact_sheet.png"
        if not cv2.imwrite(str(sheet_path), sheet):
            raise ComparisonError(f"failed to write contact sheet: {sheet_path}")
        video_path = staging / "static_diagnostic.mp4"
        video = write_static_diagnostic_video(
            video_path, sheet, fps=fps, duration_seconds=duration_seconds
        )
        report: dict[str, Any] = {
            "schema_version": 1,
            "kind": "spar3d_vggt_omega_official_custom_three_way_comparison",
            "status": "complete",
            "created_utc": _utc_now(),
            "method": "provenance_checked_static_diagnostic_contact_sheet",
            "selection": preflight["selection"],
            "input_snapshot": {
                "sha256": preflight["input_snapshot_sha256"],
                "records": [
                    {"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"]}
                    for item in preflight["input_snapshot_records"]
                ],
            },
            "comparison": {
                "A": {
                    "name": "SPAR3D single-MH learned mesh with approximate registration",
                    "representation": "learned_single_view_triangle_mesh",
                    "source_views": ["MH"],
                    "registration": "approximate_MH_camera_Sim3_using_proxy_depth",
                    "metric_scale_verified": False,
                    "physical_geometry_guarantee": False,
                    "collision_ready": False,
                    "backside_semantics": "unseen_and_backside_geometry_is_a_learned_estimate",
                    "report": preflight["registration"]["identity"],
                    "artifact": registration_outputs["registered_mesh_mh.glb"],
                },
                "B": {
                    "name": (
                        "VGGT-Omega official-rule p20 scoped to dual object-mask union"
                    ),
                    "representation": "colored_point_cloud",
                    "source_views": [
                        camera
                        for camera, count in preflight["vggt"][
                            "official_evidence"
                        ]["counts_by_view"].items()
                        if count > 0
                    ],
                    "aggregation_method": VGGT_AGGREGATION_METHOD,
                    "confidence_semantics": {
                        "percentile": 20.0,
                        "threshold_population": (
                            "finite dual object-mask union with depth-edge confidences zeroed"
                        ),
                        "per_view_threshold_fallback": False,
                        "exact_official_demo_output": False,
                        "rule_reference": (
                            "visual_util.predictions_to_glb confidence/depth-edge rule"
                        ),
                    },
                    "exported_points_by_view": preflight["vggt"][
                        "official_evidence"
                    ]["counts_by_view"],
                    "is_triangle_mesh": False,
                    "metric_scale_verified": False,
                    "provided_calibration_applied": False,
                    "physical_geometry_guarantee": False,
                    "collision_ready": False,
                    "report": preflight["vggt"]["identity"],
                    "artifact": preflight["vggt"]["artifacts"][
                        "official_object_point_cloud_glb"
                    ],
                    "evidence": preflight["vggt"]["artifacts"][
                        "official_object_point_evidence"
                    ],
                },
                "C": {
                    "name": "VGGT-Omega custom adaptive p50 dual-mask object points",
                    "representation": "colored_point_cloud",
                    "source_views": [
                        camera
                        for camera, count in preflight["vggt"][
                            "custom_evidence"
                        ]["counts_by_view"].items()
                        if count > 0
                    ],
                    "aggregation_method": VGGT_AGGREGATION_METHOD,
                    "confidence_semantics": {
                        "percentile": 50.0,
                        "threshold_population": (
                            "safe adaptive dual object-mask candidates; see recorded depth-edge modes"
                        ),
                        "per_view_threshold_fallback_enabled": True,
                        "fallback_only_when_view_globally_starved": True,
                        "fallback_views": preflight["vggt"]["custom_filter"]
                        .get("confidence_adaptation", {})
                        .get("fallback_views", []),
                        "confidence_mode_by_view": preflight["vggt"]
                        ["custom_filter"]
                        .get("confidence_adaptation", {})
                        .get("mode_by_view", {}),
                        "depth_edge_mode_by_view": preflight["vggt"]
                        ["custom_filter"]
                        .get("depth_edge_adaptation", {})
                        .get("mode_by_view", {}),
                        "depth_edge_valid_depth_fallback_views": preflight[
                            "vggt"
                        ]["custom_filter"]
                        .get("depth_edge_adaptation", {})
                        .get("finite_positive_depth_fallback_views", []),
                        "official_demo_semantics": False,
                    },
                    "exported_points_by_view": preflight["vggt"][
                        "custom_evidence"
                    ]["counts_by_view"],
                    "is_triangle_mesh": False,
                    "metric_scale_verified": False,
                    "provided_calibration_applied": False,
                    "physical_geometry_guarantee": False,
                    "collision_ready": False,
                    "report": preflight["vggt"]["identity"],
                    "artifact": preflight["vggt"]["artifacts"][
                        "object_point_cloud_glb"
                    ],
                    "evidence": preflight["vggt"]["artifacts"][
                        "object_point_evidence"
                    ],
                },
            },
            "exact_official_full_scene_reference": {
                "conversion_function": "visual_util.predictions_to_glb",
                "confidence_percentile": 20.0,
                "depth_edge_rtol": 0.03,
                "show_cam": True,
                "scene_alignment": "official_first_camera_opengl",
                "exact_official_demo_output": True,
                "artifact": preflight["vggt"]["artifacts"][
                    "official_full_scene_glb"
                ],
                "note": (
                    "This exact official full-scene GLB is provenance evidence; "
                    "panel B is the official rule scoped to the dual-mask union."
                ),
            },
            "interpretation_guardrails": [
                "This is a static representation diagnostic, not temporal reconstruction.",
                "SPAR3D hidden/backside surface is a learned single-image estimate.",
                "B adapts only the official confidence/depth-edge rule to a dual-mask-union population; it is not the exact official full-scene demo output.",
                "C is the custom p50 path with per-view rescue and is not official demo semantics.",
                "B and C are post-processing variants of the same VGGT-Omega inference and dual masks.",
                "p20 and p50 are confidence percentiles, not accuracy scores.",
                "Both VGGT-Omega object outputs are colored point sets, not meshes.",
                "VGGT-Omega scale is relative/non-metric and provided calibration was not applied.",
                "None of the three results is validated physical geometry or collision ground truth.",
            ],
            "video": video,
            "outputs": {},
        }
        report["outputs"]["contact_sheet"] = _final_output_record(
            sheet_path, output / sheet_path.name
        )
        report["outputs"]["static_diagnostic_video"] = _final_output_record(
            video_path, output / video_path.name
        )
        report["outputs"]["report"] = {
            "path": str(output / "report.json"),
            "integrity_recorded_by": "publish_manifest.json",
        }
        report["outputs"]["publish_manifest"] = {
            "path": str(output / "publish_manifest.json"),
            "self_hash_omitted_to_avoid_recursive_digest": True,
        }
        report_path = staging / "report.json"
        _write_json(report_path, report)
        publish_manifest = {
            "schema_version": 1,
            "kind": "atomic_publish_completeness_sentinel",
            "status": "complete",
            "created_utc": _utc_now(),
            "files": {
                name: _final_output_record(staging / name, output / name)
                for name in ("contact_sheet.png", "static_diagnostic.mp4", "report.json")
            },
            "self_hash_omitted_to_avoid_recursive_digest": True,
        }
        _write_json(staging / "publish_manifest.json", publish_manifest)
        if {item.name for item in staging.iterdir()} != OUTPUT_NAMES:
            raise ComparisonError("staging contains an unexpected or incomplete payload set")
        verify_input_snapshot_unchanged(preflight["input_snapshot_records"])
        _publish_directory(staging, output, overwrite=overwrite)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spar-report", type=Path, default=DEFAULT_SPAR_REPORT)
    parser.add_argument("--registration-report", type=Path, default=DEFAULT_REGISTRATION_REPORT)
    parser.add_argument("--registered-mesh", type=Path, default=DEFAULT_REGISTERED_MESH)
    parser.add_argument("--vggt-metadata", type=Path, default=DEFAULT_VGGT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", "--preflight", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--duration-seconds", type=float, default=3.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.preflight_only:
            result = preflight_job(
                spar_report=args.spar_report,
                registration_report=args.registration_report,
                registered_mesh=args.registered_mesh,
                vggt_metadata=args.vggt_metadata,
            )
        else:
            result = run_comparison(
                spar_report=args.spar_report,
                registration_report=args.registration_report,
                registered_mesh=args.registered_mesh,
                vggt_metadata=args.vggt_metadata,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                fps=args.fps,
                duration_seconds=args.duration_seconds,
            )
        printable = {key: value for key, value in result.items() if key not in {"spar", "registration", "vggt", "input_snapshot_records"}}
        print(json.dumps(printable, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ComparisonError, FileExistsError, ValueError) as exc:
        print(f"comparison failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
