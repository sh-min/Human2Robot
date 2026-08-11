#!/usr/bin/env python3
"""Compare the registered Choco SPAR3D mesh in the MH XHand compositor.

This is intentionally a *single-frame* diagnostic.  The learned SPAR3D mesh
is registered only at MH frame 187, so this script refuses to propagate that
pose to later frames.  It compares three views of the same robot pose:

1. the current completed 2.5-D object-depth/XHand-shell result;
2. the shared finger baseline plus SPAR3D front-depth z occlusion;
3. the same baseline plus a paired SPAR3D front/back volume, part-wise
   XHand camera-Z shell, and a bounded spatial gap-closing filter.

The auxiliary-camera HaCo stream is confidence-only.  Its score fusion,
active-frame increment, and primary-vs-dual final-mask difference are checked
from the persisted evidence instead of being inferred from directory presence.

The last method is a visual non-emergence filter.  It hides pixels; it does not
move the robot, solve mesh-mesh collision, or make the learned backside metric
ground truth.  A short MP4, when written, repeats the selected static frame and
is labelled as such rather than implying tracked object motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INPAINTING_ROOT = REPO_ROOT / "src" / "inpainting"
if str(INPAINTING_ROOT) not in sys.path:
    sys.path.insert(0, str(INPAINTING_ROOT))

from composite_rb5_contact_occlusion import composite_frame  # noqa: E402
from composite_xhand_mesh_volume import (  # noqa: E402
    CLASS_FRONT_OF,
    CLASS_FULLY_BEHIND,
    CLASS_INTERSECTING,
    classify_mesh_volume,
    combine_with_baseline,
    front_only_hidden,
    hidden_from_classification,
    mesh_temporal_eligibility,
)
from composite_xhand_object_barrier import (  # noqa: E402
    resize_overlay_frame,
    restore_raw_object_pixels,
    semantic_hand_labels,
    thickness_map,
)


PILOT_ROOT = REPO_ROOT / "8-5" / "mesh_sota_pilot" / "episode_1" / "choco"
DEFAULT_INPUT_MANIFEST = PILOT_ROOT / "inputs" / "manifest.json"
DEFAULT_MH_IMAGE = PILOT_ROOT / "inputs" / "mh_frame000187.jpg"
DEFAULT_REGISTERED_ROOT = PILOT_ROOT / "spar3d_registered_mh"
DEFAULT_REGISTERED_MESH = DEFAULT_REGISTERED_ROOT / "registered_mesh_mh.glb"
DEFAULT_REGISTRATION_REPORT = DEFAULT_REGISTERED_ROOT / "report.json"
DEFAULT_PROCESSED_ROOT = (
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
)
DEFAULT_OUTPUT_ROOT = PILOT_ROOT / "spar3d_xhand_occlusion_comparison"

METHOD = "static_registered_spar3d_xhand_occlusion_comparison"
REGISTRATION_METHOD = "spar3d_canonical_to_mh_approximate_sim3_silhouette_depth_fit"
CURRENT_METHOD = "visual_camera_z_xhand_barrier"
STATIC_VIDEO_NAME = "comparison_static_frame187_3s.mp4"

T_CV_TO_GL = np.diag((1.0, -1.0, -1.0, 1.0))


class ComparisonInputError(ValueError):
    """Raised when the focused static-comparison contract is violated."""


def sha256_file(path: str | Path, *, block_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def file_identity_record(path: str | Path) -> dict[str, Any]:
    """Return an exact content identity plus useful race diagnostics."""

    resolved = _require_file(path, "content-bound input")
    before = resolved.stat()
    digest = sha256_file(resolved)
    after = resolved.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise ComparisonInputError(f"input changed while hashing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": after.st_size,
        "sha256": digest,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def validate_declared_file_record(
    record: Any,
    *,
    expected_path: str | Path | None = None,
    description: str,
) -> dict[str, Any]:
    """Validate a report/manifest path+bytes+SHA record against disk."""

    if not isinstance(record, dict):
        raise ComparisonInputError(f"{description} record must be an object")
    try:
        declared_path = Path(record["path"]).expanduser().resolve()
        declared_bytes = int(record["bytes"])
        declared_hash = str(record["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ComparisonInputError(
            f"{description} record requires path, bytes, and sha256"
        ) from exc
    if expected_path is not None:
        expected = Path(expected_path).expanduser().resolve()
        if declared_path != expected:
            raise ComparisonInputError(
                f"{description} path differs: {declared_path} != {expected}"
            )
    actual = file_identity_record(declared_path)
    if actual["bytes"] != declared_bytes or actual["sha256"] != declared_hash:
        raise ComparisonInputError(
            f"{description} bytes/SHA-256 differs from its declared record"
        )
    return actual


def snapshot_digest(records: dict[str, dict[str, Any]]) -> str:
    canonical = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def verify_input_snapshot_unchanged(
    before: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Re-hash every bound input immediately before publication."""

    after = {
        name: file_identity_record(record["path"])
        for name, record in before.items()
    }
    if after != before:
        changed = sorted(name for name in before if after.get(name) != before[name])
        raise ComparisonInputError(
            "content-bound inputs changed during comparison: " + ", ".join(changed)
        )
    return after


def _require_file(path: str | Path, description: str) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_file() or result.stat().st_size <= 0:
        raise FileNotFoundError(f"missing {description}: {result}")
    return result


def _load_json(path: str | Path, description: str) -> tuple[Path, dict[str, Any]]:
    result = _require_file(path, description)
    try:
        payload = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonInputError(f"invalid {description}: {result}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ComparisonInputError(f"{description} must contain a JSON object")
    return result, payload


def source_paths(processed_root: str | Path) -> dict[str, Path]:
    root = Path(processed_root).expanduser().resolve()
    completion = root / "object_completion_dual_haco_e2fgvi"
    overlay = root / "overlay_processor"
    current = root / "overlay_best_inpaint_barrier"
    contact = root / "overlay_object3d_force_temporal"
    haco_mh = root / "overlay_haco_mh"
    haco_dual = root / "overlay_haco_dual"
    return {
        "processed_root": root,
        "raw_video": root / "video_L.mp4",
        "background_video": completion / "video_object_completed.mp4",
        "object_support_mask": completion / "object_mask_amodal.npy",
        "object_restore_mask": completion / "object_mask_observed_clean.npy",
        "object_surface_depth": completion / "object_surface_depth_completed.npy",
        "haco_completion_report": completion / "report.json",
        "haco_evidence": completion / "haco_evidence.npz",
        "object_modal_mask": root / "object_layer" / "object_mask_modal.npy",
        "overlay_rgb": overlay / "robot_rgb.npy",
        "overlay_depth": overlay / "robot_depth.npy",
        "overlay_robot_mask": overlay / "robot_mask.npy",
        "overlay_hand_mask": overlay / "robot_hand_mask.npy",
        "overlay_finger_labels": overlay / "robot_finger_labels.npy",
        "current_mask": current / "occluded_hand_mask.npy",
        "current_report": current / "report.json",
        "contact_baseline_mask": contact / "occluded_finger_mask.npy",
        "contact_baseline_report": contact / "report.json",
        "contact_penetration_evidence": contact
        / "object3d_penetration_evidence.npz",
        "haco_mh_mask": haco_mh / "occluded_finger_mask.npy",
        "haco_mh_report": haco_mh / "report.json",
        "haco_dual_mask": haco_dual / "occluded_finger_mask.npy",
        "haco_dual_report": haco_dual / "report.json",
    }


def load_pilot_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path, payload = _load_json(path, "mesh-pilot input manifest")
    if payload.get("schema_version") != 1 or payload.get("kind") != (
        "mesh_sota_pilot_input_bundle"
    ):
        raise ComparisonInputError("unexpected mesh-pilot manifest schema/kind")
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ComparisonInputError("mesh-pilot manifest has no selection")
    if str(selection.get("object_label", "")).casefold() != "choco":
        raise ComparisonInputError("this focused comparison only accepts Choco")
    if str(selection.get("mh_role", "")).split("/")[0] != "primary":
        raise ComparisonInputError("MH must be the primary/final view")
    calibration = payload.get("calibration")
    try:
        mh = calibration["intrinsics_by_view"]["MH"]
        checker = calibration["checkerboard"]
    except (KeyError, TypeError) as exc:
        raise ComparisonInputError("manifest lacks MH calibration metadata") from exc
    if checker.get("metric_scale_verified") is not False:
        raise ComparisonInputError("expected unverified checkerboard metric scale")
    camera_matrix = np.asarray(mh.get("camera_matrix"), dtype=np.float64)
    distortion = np.asarray(
        mh.get("distortion_k1_k2_p1_p2_k3"), dtype=np.float64
    )
    if camera_matrix.shape != (3, 3) or distortion.shape != (5,):
        raise ComparisonInputError("invalid MH camera calibration shape")
    if not np.isfinite(camera_matrix).all() or not np.isfinite(distortion).all():
        raise ComparisonInputError("non-finite MH camera calibration")
    return manifest_path, payload


def validate_manifest_output_records(
    manifest: dict[str, Any],
    *,
    expected_mh_image: str | Path,
) -> dict[str, dict[str, Any]]:
    """Bind every materialized pilot input to the manifest's bytes/SHA."""

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ComparisonInputError("mesh-pilot manifest has no output records")
    if "mh_image" not in outputs:
        raise ComparisonInputError("mesh-pilot manifest has no MH image record")
    records: dict[str, dict[str, Any]] = {}
    for name, record in sorted(outputs.items()):
        records[f"manifest_output:{name}"] = validate_declared_file_record(
            record,
            expected_path=expected_mh_image if name == "mh_image" else None,
            description=f"mesh-pilot output {name}",
        )
    return records


def comparison_contract() -> dict[str, Any]:
    """Stable truth-in-advertising fields used by reports and tests."""

    return {
        "method": METHOD,
        "experiment_scope": "one registered MH frame only",
        "static_frame_only": True,
        "object_pose_propagated_to_other_frames": False,
        "robot_pose_modified": False,
        "physical_collision_solver": False,
        "metric_collision_guarantee": False,
        "visual_compositing_only": True,
        "spar3d_hidden_surface_is_learned_estimate": True,
        "registered_sim3_metric_scale_verified": False,
        "current_0805_baseline_uses_nominal_primitive_mesh": False,
        "current_0805_baseline_geometry": "completed visible camera-Z height field",
    }


def registered_bundle_status(
    mesh_path: str | Path, report_path: str | Path
) -> str:
    mesh = Path(mesh_path).expanduser().resolve()
    report = Path(report_path).expanduser().resolve()
    mesh_ready = mesh.is_file() and mesh.stat().st_size > 0
    report_ready = report.is_file() and report.stat().st_size > 0
    if mesh_ready and report_ready:
        return "ready"
    if mesh_ready != report_ready:
        return "blocked_incomplete_registered_bundle"
    return "waiting_for_registered_spar3d_mesh"


def validate_registration_report(
    report_path: str | Path,
    mesh_path: str | Path,
    *,
    expected_frame_index: int,
    expected_manifest_path: str | Path | None = None,
    expected_mh_image_path: str | Path | None = None,
    expected_modal_mask_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path, report = _load_json(report_path, "SPAR3D MH registration report")
    mesh = _require_file(mesh_path, "registered SPAR3D MH mesh")
    if report.get("status") != "complete" or report.get("method") != REGISTRATION_METHOD:
        raise ComparisonInputError("registered mesh report is not a complete pilot fit")
    for key in (
        "metric_scale_verified",
        "physical_geometry_guarantee",
        "collision_ready",
    ):
        if report.get(key) is not False:
            raise ComparisonInputError(f"registration report overstates {key}")
    selection = report.get("selection")
    if not isinstance(selection, dict) or int(selection.get("mh_frame_index", -1)) != (
        expected_frame_index
    ):
        raise ComparisonInputError("registration report frame differs from the pilot")
    outputs = report.get("outputs")
    if not isinstance(outputs, dict):
        raise ComparisonInputError("registration report lacks output records")
    if "registered_mesh_mh.glb" not in outputs:
        raise ComparisonInputError("registration report lacks its registered mesh record")
    validated_outputs: dict[str, dict[str, Any]] = {}
    for name, record in sorted(outputs.items()):
        if name == "report.json":
            declared_report = Path(record.get("path", "")).expanduser().resolve()
            if declared_report != path:
                raise ComparisonInputError(
                    "registration report self path differs from its output record"
                )
            continue
        validated_outputs[name] = validate_declared_file_record(
            record,
            expected_path=mesh if name == "registered_mesh_mh.glb" else None,
            description=f"registration output {name}",
        )
    mesh_record = validated_outputs["registered_mesh_mh.glb"]
    source = report.get("source_spar3d_report")
    if not isinstance(source, dict) or source.get(
        "hidden_geometry_is_learned_estimate"
    ) is not True:
        raise ComparisonInputError("registration report lost learned-backside provenance")
    try:
        source_spar_path = Path(source["path"]).expanduser().resolve()
        source_spar_hash = str(source["sha256"])
    except (KeyError, TypeError) as exc:
        raise ComparisonInputError(
            "registration report lacks its source SPAR report path/SHA"
        ) from exc
    source_spar_record = file_identity_record(source_spar_path)
    if source_spar_record["sha256"] != source_spar_hash:
        raise ComparisonInputError("source SPAR report SHA-256 differs from registration")

    source_mesh = validate_declared_file_record(
        report.get("source_mesh"),
        description="registration canonical SPAR mesh",
    )
    validated_sources: dict[str, dict[str, Any]] = {
        "canonical_spar_mesh": source_mesh,
        "source_spar_report": source_spar_record,
    }
    registration_sources = report.get("sources")
    if not isinstance(registration_sources, dict):
        raise ComparisonInputError("registration report lacks source records")
    expected_sources = {
        "input_manifest": expected_manifest_path,
        "mh_image": expected_mh_image_path,
        "mh_modal_mask": expected_modal_mask_path,
    }
    for name, record in sorted(registration_sources.items()):
        validated_sources[f"registration_source:{name}"] = (
            validate_declared_file_record(
                record,
                expected_path=expected_sources.get(name),
                description=f"registration source {name}",
            )
        )
    return path, report, {
        **mesh_record,
        "validated_outputs": validated_outputs,
        "validated_sources": validated_sources,
    }


def _active_runs_by_finger(
    active: np.ndarray,
    finger_names: list[str],
) -> dict[str, list[list[int]]]:
    values = np.asarray(active, dtype=bool)
    if values.ndim != 2 or values.shape[1] != len(finger_names):
        raise ComparisonInputError("HaCo active evidence has an invalid shape")
    result: dict[str, list[list[int]]] = {}
    for finger_index, finger in enumerate(finger_names):
        indices = np.flatnonzero(values[:, finger_index])
        runs: list[list[int]] = []
        if indices.size:
            start = previous = int(indices[0])
            for raw_index in indices[1:]:
                index = int(raw_index)
                if index != previous + 1:
                    runs.append([start, previous])
                    start = index
                previous = index
            runs.append([start, previous])
        result[finger] = runs
    return result


def _mask_delta_summary(
    primary_path: Path,
    dual_path: Path,
    *,
    frame_index: int,
) -> dict[str, Any]:
    primary = np.load(
        _require_file(primary_path, "primary-only HaCo mask"),
        mmap_mode="r",
        allow_pickle=False,
    )
    dual = np.load(
        _require_file(dual_path, "dual-camera HaCo mask"),
        mmap_mode="r",
        allow_pickle=False,
    )
    if primary.shape != dual.shape or primary.ndim != 3:
        raise ComparisonInputError("primary/dual HaCo masks have different shapes")
    if primary.dtype != np.bool_ or dual.dtype != np.bool_:
        raise ComparisonInputError("primary/dual HaCo masks must be bool arrays")
    if not 0 <= frame_index < len(primary):
        raise ComparisonInputError("selected frame is outside the HaCo masks")

    primary_sha = sha256_file(primary_path)
    dual_sha = sha256_file(dual_path)
    changed_pixels = added_pixels = removed_pixels = 0
    if primary_sha != dual_sha:
        for index in range(len(primary)):
            primary_frame = np.asarray(primary[index], dtype=bool)
            dual_frame = np.asarray(dual[index], dtype=bool)
            changed_pixels += int(np.count_nonzero(primary_frame ^ dual_frame))
            added_pixels += int(np.count_nonzero(dual_frame & ~primary_frame))
            removed_pixels += int(np.count_nonzero(primary_frame & ~dual_frame))
    primary_frame = np.asarray(primary[frame_index], dtype=bool)
    dual_frame = np.asarray(dual[frame_index], dtype=bool)
    return {
        "shape": list(primary.shape),
        "primary_sha256": primary_sha,
        "dual_sha256": dual_sha,
        "changed_pixels": changed_pixels,
        "added_pixels": added_pixels,
        "removed_pixels": removed_pixels,
        "selected_frame_primary_pixels": int(primary_frame.sum()),
        "selected_frame_dual_pixels": int(dual_frame.sum()),
        "selected_frame_changed_pixels": int(
            np.count_nonzero(primary_frame ^ dual_frame)
        ),
    }


def validate_haco_auxiliary_effect(
    paths: dict[str, Path],
    contact: dict[str, Any],
    *,
    frame_index: int,
) -> dict[str, Any]:
    """Validate score fusion separately from SH activation, mask, and geometry."""

    _, completion = _load_json(
        paths["haco_completion_report"], "HaCo completion report"
    )
    _, primary_report = _load_json(
        paths["haco_mh_report"], "primary-only HaCo report"
    )
    _, dual_report = _load_json(paths["haco_dual_report"], "dual-camera HaCo report")

    contact_sources = contact.get("sources")
    if not isinstance(contact_sources, dict):
        raise ComparisonInputError("HaCo/contact report lacks source paths")
    auxiliary_dir_value = contact_sources.get("aux_contact_dir")
    if not auxiliary_dir_value:
        raise ComparisonInputError("HaCo/contact report lacks auxiliary-camera input")
    auxiliary_dir = Path(auxiliary_dir_value).expanduser().resolve()
    if not auxiliary_dir.is_dir():
        raise ComparisonInputError("auxiliary-camera HaCo directory is missing")
    if contact.get("contact_fusion") != (
        "per-finger maximum of primary/auxiliary HaCo scores"
    ):
        raise ComparisonInputError("HaCo/contact report does not declare score fusion")

    try:
        frame_count = int(contact["frames"])
        finger_names = [str(value) for value in contact["finger_names"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ComparisonInputError("HaCo/contact report lacks frame/finger metadata") from exc
    if frame_count <= 0 or len(finger_names) != 5 or not 0 <= frame_index < frame_count:
        raise ComparisonInputError("invalid HaCo frame/finger metadata")

    evidence_path = _require_file(paths["haco_evidence"], "HaCo evidence")
    required_evidence = {
        "finger_names",
        "primary_scores",
        "auxiliary_scores",
        "fused_scores",
        "primary_active",
        "active",
        "auxiliary_qualified",
        "auxiliary_frame_indices",
    }
    try:
        with np.load(evidence_path, allow_pickle=False) as evidence:
            missing = sorted(required_evidence - set(evidence.files))
            if missing:
                raise ComparisonInputError(
                    "HaCo evidence lacks arrays: " + ", ".join(missing)
                )
            evidence_fingers = [str(value) for value in evidence["finger_names"]]
            primary_scores = np.asarray(evidence["primary_scores"], dtype=np.float32)
            auxiliary_scores = np.asarray(
                evidence["auxiliary_scores"], dtype=np.float32
            )
            fused_scores = np.asarray(evidence["fused_scores"], dtype=np.float32)
            primary_active = np.asarray(evidence["primary_active"])
            active = np.asarray(evidence["active"])
            auxiliary_qualified = np.asarray(evidence["auxiliary_qualified"])
            auxiliary_frame_indices = np.asarray(
                evidence["auxiliary_frame_indices"]
            )
    except (OSError, ValueError) as exc:
        if isinstance(exc, ComparisonInputError):
            raise
        raise ComparisonInputError(f"invalid HaCo evidence: {evidence_path}: {exc}") from exc

    expected_matrix_shape = (frame_count, len(finger_names))
    if evidence_fingers != finger_names:
        raise ComparisonInputError("HaCo evidence finger order differs from its report")
    for name, value in (
        ("primary_scores", primary_scores),
        ("auxiliary_scores", auxiliary_scores),
        ("fused_scores", fused_scores),
        ("primary_active", primary_active),
        ("active", active),
        ("auxiliary_qualified", auxiliary_qualified),
    ):
        if value.shape != expected_matrix_shape:
            raise ComparisonInputError(f"HaCo evidence {name} has an invalid shape")
    if primary_active.dtype != np.bool_ or active.dtype != np.bool_:
        raise ComparisonInputError("HaCo active evidence must be bool")
    if auxiliary_qualified.dtype != np.bool_:
        raise ComparisonInputError("HaCo auxiliary-qualified evidence must be bool")
    if auxiliary_frame_indices.shape != (frame_count,) or not np.issubdtype(
        auxiliary_frame_indices.dtype, np.integer
    ):
        raise ComparisonInputError("HaCo auxiliary frame indices have an invalid shape")
    if np.any(np.isinf(primary_scores)) or np.any(np.isinf(auxiliary_scores)):
        raise ComparisonInputError("HaCo score evidence contains infinity")
    if np.any(~np.isfinite(primary_scores)) or np.any(~np.isfinite(fused_scores)):
        raise ComparisonInputError("primary/fused HaCo scores must be finite")

    expected_fused = np.fmax(primary_scores, auxiliary_scores)
    if not np.allclose(fused_scores, expected_fused, atol=1.0e-6, rtol=1.0e-6):
        raise ComparisonInputError(
            "persisted HaCo fused scores are not the primary/auxiliary maximum"
        )
    auxiliary_score_dominant = int(
        np.count_nonzero(auxiliary_scores > primary_scores)
    )
    fused_score_changes = auxiliary_score_dominant

    for report_key, evidence_value in (
        ("contact_score_primary", primary_scores),
        ("contact_score_auxiliary", auxiliary_scores),
        ("contact_score_fused", fused_scores),
    ):
        report_value = np.asarray(contact.get(report_key), dtype=np.float32)
        if report_value.shape != expected_matrix_shape or not np.allclose(
            report_value,
            evidence_value,
            atol=1.0e-6,
            rtol=1.0e-6,
            equal_nan=True,
        ):
            raise ComparisonInputError(
                f"HaCo evidence {report_key} differs from the contact report"
            )

    active_increment = int(np.count_nonzero(active & ~primary_active))
    active_removed = int(np.count_nonzero(primary_active & ~active))
    if active_removed:
        raise ComparisonInputError("auxiliary HaCo unexpectedly removed active fingers")
    contact_policy = contact.get("contact_activation_policy")
    if not isinstance(contact_policy, dict):
        raise ComparisonInputError("HaCo/contact report lacks activation policy")
    policy_counts = contact_policy.get("counts")
    if not isinstance(policy_counts, dict):
        raise ComparisonInputError("HaCo/contact report lacks activation counts")
    expected_counts = {
        "auxiliary_score_dominant_frame_fingers": auxiliary_score_dominant,
        "active_frame_fingers_added_vs_primary": active_increment,
    }
    for name, expected in expected_counts.items():
        if policy_counts.get(name) != expected:
            raise ComparisonInputError(
                f"HaCo/contact report {name} differs from persisted evidence"
            )
    if contact.get("active_runs") != _active_runs_by_finger(active, finger_names):
        raise ComparisonInputError("HaCo active runs differ from persisted evidence")
    if contact.get("active_runs_primary") != _active_runs_by_finger(
        primary_active, finger_names
    ):
        raise ComparisonInputError("primary HaCo active runs differ from evidence")

    completion_outputs = completion.get("outputs")
    completion_counts = completion.get("counts")
    completion_invariants = completion.get("invariants")
    if (
        not isinstance(completion_outputs, dict)
        or completion_outputs.get("haco_evidence") != evidence_path.name
        or not isinstance(completion_counts, dict)
        or not isinstance(completion_invariants, dict)
    ):
        raise ComparisonInputError("HaCo completion report does not bind its evidence")
    auxiliary_qualified_count = int(np.count_nonzero(auxiliary_qualified))
    if completion_counts.get("auxiliary_qualified_finger_frames") != (
        auxiliary_qualified_count
    ):
        raise ComparisonInputError(
            "completion auxiliary-qualified count differs from HaCo evidence"
        )

    geometry_claims = (
        contact_policy.get("auxiliary_geometry_used"),
        contact.get("invariants", {}).get("auxiliary_geometry_used"),
        completion_invariants.get("auxiliary_geometry_used"),
        dual_report.get("contact_activation_policy", {}).get(
            "auxiliary_geometry_used"
        ),
    )
    if any(value is not False for value in geometry_claims):
        raise ComparisonInputError("auxiliary HaCo geometry must be explicitly false")
    if completion_invariants.get("auxiliary_haco_is_confidence_only") is not True:
        raise ComparisonInputError("completion report lost confidence-only SH invariant")
    if primary_report.get("contact_fusion") != "primary HaCo scores only":
        raise ComparisonInputError("primary-only HaCo report is not primary-only")
    if dual_report.get("contact_fusion") != contact.get("contact_fusion"):
        raise ComparisonInputError("dual HaCo report fusion differs from shared baseline")
    if dual_report.get("active_runs") != contact.get("active_runs"):
        raise ComparisonInputError("dual HaCo active runs differ from shared baseline")
    if primary_report.get("active_runs") != contact.get("active_runs_primary"):
        raise ComparisonInputError("primary-only HaCo active runs differ from baseline")

    mask_delta = _mask_delta_summary(
        paths["haco_mh_mask"],
        paths["haco_dual_mask"],
        frame_index=frame_index,
    )

    penetration_path = _require_file(
        paths["contact_penetration_evidence"], "Object3D penetration evidence"
    )
    try:
        with np.load(penetration_path, allow_pickle=False) as penetration:
            required_penetration = {
                "finger_names",
                "force_candidate_pixels",
                "temporal_added_pixels",
            }
            missing = sorted(required_penetration - set(penetration.files))
            if missing:
                raise ComparisonInputError(
                    "Object3D penetration evidence lacks arrays: "
                    + ", ".join(missing)
                )
            penetration_fingers = [
                str(value) for value in penetration["finger_names"]
            ]
            force_pixels = np.asarray(penetration["force_candidate_pixels"])
            temporal_pixels = np.asarray(penetration["temporal_added_pixels"])
    except (OSError, ValueError) as exc:
        if isinstance(exc, ComparisonInputError):
            raise
        raise ComparisonInputError(
            f"invalid Object3D penetration evidence: {penetration_path}: {exc}"
        ) from exc
    if (
        penetration_fingers != finger_names
        or force_pixels.shape != expected_matrix_shape
        or temporal_pixels.shape != expected_matrix_shape
        or not np.issubdtype(force_pixels.dtype, np.integer)
        or not np.issubdtype(temporal_pixels.dtype, np.integer)
        or np.any(force_pixels < 0)
        or np.any(temporal_pixels < 0)
    ):
        raise ComparisonInputError("Object3D penetration evidence has invalid arrays")

    penetration_control = contact.get("object3d_penetration_control")
    if not isinstance(penetration_control, dict):
        raise ComparisonInputError("contact report lacks Object3D penetration control")
    surface_force = penetration_control.get("surface_force")
    temporal_filter = penetration_control.get("temporal_filter")
    if not isinstance(surface_force, dict) or not isinstance(temporal_filter, dict):
        raise ComparisonInputError("contact report lacks Object3D force/filter records")
    contact_invariants = contact.get("invariants")
    if (
        not isinstance(contact_invariants, dict)
        or contact_invariants.get("object3d_force_bypasses_haco_selector") is not True
    ):
        raise ComparisonInputError(
            "contact report does not verify that Object3D force bypasses HaCo"
        )
    if surface_force.get("haco_activation_used_for_added_branch") is not False:
        raise ComparisonInputError("Object3D force branch unexpectedly uses HaCo activation")
    if int(force_pixels.sum()) != surface_force.get("candidate_pixels"):
        raise ComparisonInputError("Object3D force pixel count differs from its report")
    if int(temporal_pixels.sum()) != temporal_filter.get("added_pixels"):
        raise ComparisonInputError("Object3D temporal pixel count differs from its report")

    baseline = np.load(
        _require_file(paths["contact_baseline_mask"], "shared baseline mask"),
        mmap_mode="r",
        allow_pickle=False,
    )
    if baseline.shape[0] != frame_count or baseline.ndim != 3 or baseline.dtype != np.bool_:
        raise ComparisonInputError("shared baseline mask has an invalid contract")
    baseline_frame_pixels = int(np.asarray(baseline[frame_index], dtype=bool).sum())
    report_pixel_counts = contact.get("occluded_pixel_count")
    if (
        not isinstance(report_pixel_counts, list)
        or len(report_pixel_counts) != frame_count
        or report_pixel_counts[frame_index] != baseline_frame_pixels
    ):
        raise ComparisonInputError("shared baseline frame count differs from its report")
    selected_force = [int(value) for value in force_pixels[frame_index]]
    selected_temporal = [int(value) for value in temporal_pixels[frame_index]]
    selected_active = int(np.count_nonzero(active[frame_index]))
    force_explains_selected_baseline = bool(
        selected_active == 0
        and sum(selected_temporal) == 0
        and baseline_frame_pixels == sum(selected_force)
        and surface_force.get("haco_activation_used_for_added_branch") is False
    )

    return {
        "auxiliary_haco_input_available": True,
        "auxiliary_scores_fused": True,
        "auxiliary_geometry_used": False,
        "auxiliary_score_dominant_frame_fingers": auxiliary_score_dominant,
        "auxiliary_score_changed_frame_fingers": fused_score_changes,
        "auxiliary_active_increment_frame_fingers": active_increment,
        "auxiliary_qualified_frame_fingers": auxiliary_qualified_count,
        "auxiliary_mask_increment_pixels": mask_delta["added_pixels"],
        "dual_camera_changed_final_mask": mask_delta["changed_pixels"] > 0,
        "dual_camera_changed_mask_pixels": mask_delta["changed_pixels"],
        "dual_camera_changed_final_mask_scope": (
            "primary-only versus dual-camera HaCo occluded_finger_mask.npy"
        ),
        "dual_camera_changed_final_mask_basis": (
            "exact mask-byte comparison; the retained Object3D force branch "
            "is separately verified to bypass HaCo activation"
        ),
        "primary_vs_dual_haco_mask": mask_delta,
        "selected_frame_baseline_attribution": {
            "frame_index": frame_index,
            "baseline_occluded_pixels": baseline_frame_pixels,
            "haco_active_fingers": selected_active,
            "object3d_force_candidate_pixels_by_finger": dict(
                zip(finger_names, selected_force)
            ),
            "object3d_force_candidate_pixels": sum(selected_force),
            "object3d_temporal_added_pixels": sum(selected_temporal),
            "haco_activation_used_for_object3d_force_branch": False,
            "all_pixels_explained_by_object3d_force_not_haco": (
                force_explains_selected_baseline
            ),
        },
        "evidence": {
            "haco_npz": str(evidence_path),
            "completion_report": str(paths["haco_completion_report"].resolve()),
            "contact_report": str(paths["contact_baseline_report"].resolve()),
            "primary_report": str(paths["haco_mh_report"].resolve()),
            "dual_report": str(paths["haco_dual_report"].resolve()),
            "object3d_penetration_npz": str(penetration_path),
        },
    }


def validate_current_method_reports(
    paths: dict[str, Path],
    *,
    frame_index: int,
) -> dict[str, Any]:
    current_path, current = _load_json(paths["current_report"], "current barrier report")
    contact_path, contact = _load_json(
        paths["contact_baseline_report"], "HaCo/contact baseline report"
    )
    if current.get("method") != CURRENT_METHOD:
        raise ComparisonInputError("current comparison source is not the XHand barrier")
    if current.get("representation") != "visible_camera_z_height_field":
        raise ComparisonInputError("current barrier is not the expected 2.5-D height field")
    if current.get("metric_collision_guarantee") is not False:
        raise ComparisonInputError("current barrier overstates metric collision provenance")
    current_sources = current.get("sources")
    if not isinstance(current_sources, dict):
        raise ComparisonInputError("current barrier report lacks source paths")
    expected_current_sources = {
        "background": paths["background_video"],
        "raw_video": paths["raw_video"],
        "overlay_dir": paths["overlay_rgb"].parent,
        "object_mask": paths["object_support_mask"],
        "object_restore_mask": paths["object_restore_mask"],
        "object_surface_depth": paths["object_surface_depth"],
        "baseline_mask": paths["contact_baseline_mask"],
    }
    for name, expected in expected_current_sources.items():
        actual = Path(current_sources.get(name, "")).expanduser().resolve()
        if actual != expected.resolve():
            raise ComparisonInputError(
                f"current barrier source {name} differs: {actual} != {expected}"
            )
    current_outputs = current.get("outputs")
    if not isinstance(current_outputs, dict) or current_outputs.get("mask") != (
        paths["current_mask"].name
    ):
        raise ComparisonInputError("current barrier report does not name its mask output")
    if current_path.parent / str(current_outputs["mask"]) != paths[
        "current_mask"
    ].resolve():
        raise ComparisonInputError("current barrier mask is outside its report bundle")
    expected_baseline = paths["contact_baseline_mask"].resolve()
    actual_baseline = Path(current_sources.get("baseline_mask", "")).resolve()
    if actual_baseline != expected_baseline:
        raise ComparisonInputError("current barrier and SPAR variants lack a shared baseline")
    contact_config = contact.get("config", {})
    if not isinstance(contact_config, dict) or contact_config.get(
        "object3d_force_surface"
    ) is not True:
        raise ComparisonInputError("unexpected HaCo/contact baseline configuration")
    contact_sources = contact.get("sources", {})
    if not isinstance(contact_sources, dict):
        raise ComparisonInputError("HaCo/contact report lacks source paths")
    expected_contact_sources = {
        "processed_demo": paths["processed_root"],
        "background": paths["background_video"],
        "raw_video": paths["raw_video"],
        "overlay_dir": paths["overlay_rgb"].parent,
        "object_mask": paths["object_modal_mask"],
        "object_restore_mask": paths["object_restore_mask"],
        "object_surface_depth": paths["object_surface_depth"],
    }
    for name, expected in expected_contact_sources.items():
        actual = Path(contact_sources.get(name, "")).expanduser().resolve()
        if actual != expected.resolve():
            raise ComparisonInputError(
                f"HaCo/contact source {name} differs: {actual} != {expected}"
            )
    haco_effect = validate_haco_auxiliary_effect(
        paths,
        contact,
        frame_index=frame_index,
    )
    return {
        "current_report": str(current_path),
        "contact_baseline_report": str(contact_path),
        "current_method": current.get("method"),
        "current_representation": current.get("representation"),
        "shared_contact_baseline": str(expected_baseline),
        "shared_baseline_has_auxiliary_camera_haco_input": haco_effect[
            "auxiliary_haco_input_available"
        ],
        "auxiliary_haco_effect": haco_effect,
        "current_config": current.get("config"),
        "binding_limitations": [
            "legacy current/contact reports contain source paths but no source hashes",
            "the exact reports and every directly consumed array are SHA-bound by this comparison",
        ],
    }


def _array_metadata(path: Path, frame_index: int) -> dict[str, Any]:
    array = np.load(_require_file(path, path.name), mmap_mode="r", allow_pickle=False)
    if array.ndim < 1 or not 0 <= frame_index < len(array):
        raise ComparisonInputError(f"frame {frame_index} is outside {path}")
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def build_input_snapshot(
    *,
    manifest_path: Path,
    manifest_output_records: dict[str, dict[str, Any]],
    image_path: Path,
    paths: dict[str, Path],
    registration_report_path: Path | None = None,
    registration_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Hash every directly consumed artifact and validated upstream record."""

    records: dict[str, dict[str, Any]] = {
        "input_manifest": file_identity_record(manifest_path),
        "mh_image_cli": file_identity_record(image_path),
        **manifest_output_records,
    }
    for name, path in sorted(paths.items()):
        if name != "processed_root":
            records[f"processed:{name}"] = file_identity_record(path)
    if registration_report_path is not None:
        records["registration:report"] = file_identity_record(
            registration_report_path
        )
    if registration_bundle is not None:
        for name, record in sorted(
            registration_bundle.get("validated_outputs", {}).items()
        ):
            records[f"registration_output:{name}"] = dict(record)
        for name, record in sorted(
            registration_bundle.get("validated_sources", {}).items()
        ):
            records[f"registration_upstream:{name}"] = dict(record)
    return {
        "algorithm": "SHA-256",
        "record_count": len(records),
        "records": records,
        "snapshot_sha256": snapshot_digest(records),
        "verified_unchanged_before_publish": False,
    }


def preflight_job(
    *,
    input_manifest: str | Path,
    mh_image: str | Path,
    registered_mesh: str | Path,
    registration_report: str | Path,
    processed_root: str | Path,
) -> dict[str, Any]:
    manifest_path, manifest = load_pilot_manifest(input_manifest)
    frame_index = int(manifest["selection"]["mh_frame_index"])
    image_path = _require_file(mh_image, "selected MH frame")
    manifest_output_records = validate_manifest_output_records(
        manifest,
        expected_mh_image=image_path,
    )
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ComparisonInputError(f"OpenCV could not read {image_path}")
    paths = source_paths(processed_root)
    for key, path in paths.items():
        if key != "processed_root":
            _require_file(path, key)
    method_sources = validate_current_method_reports(paths, frame_index=frame_index)
    arrays = {
        key: _array_metadata(paths[key], frame_index)
        for key in (
            "object_support_mask",
            "object_restore_mask",
            "overlay_rgb",
            "overlay_depth",
            "overlay_robot_mask",
            "overlay_hand_mask",
            "overlay_finger_labels",
            "current_mask",
            "contact_baseline_mask",
        )
    }
    status = registered_bundle_status(registered_mesh, registration_report)
    registration_summary = None
    if status == "ready":
        modal_mask_path = Path(
            manifest["outputs"]["modal_mask"]["path"]
        ).expanduser().resolve()
        report_path, report, mesh_record = validate_registration_report(
            registration_report,
            registered_mesh,
            expected_frame_index=frame_index,
            expected_manifest_path=manifest_path,
            expected_mh_image_path=image_path,
            expected_modal_mask_path=modal_mask_path,
        )
        registration_summary = {
            "report": str(report_path),
            "mesh": mesh_record,
            "method": report.get("method"),
        }
    input_snapshot = build_input_snapshot(
        manifest_path=manifest_path,
        manifest_output_records=manifest_output_records,
        image_path=image_path,
        paths=paths,
        registration_report_path=(
            Path(registration_report).expanduser().resolve()
            if status == "ready"
            else None
        ),
        registration_bundle=mesh_record if status == "ready" else None,
    )
    return {
        "schema_version": 1,
        "status": status,
        **comparison_contract(),
        "selection": manifest["selection"],
        "inputs": {
            "manifest": str(manifest_path),
            "mh_image": str(image_path),
            "processed_root": str(paths["processed_root"]),
            "registered_mesh_expected": str(Path(registered_mesh).resolve()),
            "registration_report_expected": str(Path(registration_report).resolve()),
        },
        "checks": {
            "image_shape_hwc": list(image.shape),
            "arrays": arrays,
            "registration": registration_summary,
            "current_methods": method_sources,
            "input_snapshot": input_snapshot,
        },
    }


def orient_depth_pair(
    normal_depth: np.ndarray,
    reversed_depth: np.ndarray,
    *,
    minimum_depth_m: float = 0.02,
    maximum_depth_m: float = 5.0,
    order_tolerance_m: float = 5.0e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Orient two winding passes into ordered front/back camera-Z rasters."""

    normal = np.asarray(normal_depth, dtype=np.float32)
    reverse = np.asarray(reversed_depth, dtype=np.float32)
    if normal.shape != reverse.shape or normal.ndim != 2:
        raise ValueError("normal/reversed depth must share one 2-D shape")
    if not (
        math.isfinite(minimum_depth_m)
        and math.isfinite(maximum_depth_m)
        and 0 < minimum_depth_m < maximum_depth_m
        and math.isfinite(order_tolerance_m)
        and order_tolerance_m >= 0
    ):
        raise ValueError("invalid depth-pair thresholds")
    normal_valid = (
        np.isfinite(normal)
        & (normal > np.float32(minimum_depth_m))
        & (normal < np.float32(maximum_depth_m))
    )
    reverse_valid = (
        np.isfinite(reverse)
        & (reverse > np.float32(minimum_depth_m))
        & (reverse < np.float32(maximum_depth_m))
    )
    paired = normal_valid & reverse_valid
    direct = paired & (reverse + np.float32(order_tolerance_m) >= normal)
    swapped = paired & (normal + np.float32(order_tolerance_m) >= reverse)
    use_swapped = int(swapped.sum()) > int(direct.sum())
    front_source, back_source = (reverse, normal) if use_swapped else (normal, reverse)
    ordered = paired & (
        back_source + np.float32(order_tolerance_m) >= front_source
    )
    front = np.where(ordered, front_source, 0.0).astype(np.float32)
    back = np.where(ordered, np.maximum(back_source, front_source), 0.0).astype(
        np.float32
    )
    return front, back, ordered, {
        "winding_orientation": "reversed_is_front" if use_swapped else "normal_is_front",
        "normal_valid_pixels": int(normal_valid.sum()),
        "reversed_valid_pixels": int(reverse_valid.sum()),
        "paired_before_order_pixels": int(paired.sum()),
        "direct_ordered_pixels": int(direct.sum()),
        "swapped_ordered_pixels": int(swapped.sum()),
        "ordered_pair_pixels": int(ordered.sum()),
        "ordered_pair_fraction_of_paired": float(ordered.sum()) / max(int(paired.sum()), 1),
    }


def _load_registered_triangle_mesh(path: Path) -> tuple[Any, dict[str, Any]]:
    import trimesh  # noqa: PLC0415

    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    else:
        parts = [
            value
            for value in np.atleast_1d(loaded.dump(concatenate=False))
            if isinstance(value, trimesh.Trimesh)
        ]
        if not parts:
            raise ComparisonInputError("registered GLB contains no triangle mesh")
        mesh = parts[0].copy() if len(parts) == 1 else trimesh.util.concatenate(parts)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not len(faces):
        raise ComparisonInputError("registered mesh is empty or malformed")
    if not np.isfinite(vertices).all():
        raise ComparisonInputError("registered mesh contains non-finite vertices")
    positive_z_fraction = float(np.mean(vertices[:, 2] > 0.02))
    if positive_z_fraction < 0.95:
        raise ComparisonInputError("registered mesh is not in positive-Z MH camera space")
    return mesh, {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "bounds_mh_camera_proxy_m": np.asarray(mesh.bounds).tolist(),
        "extents_proxy_m": np.asarray(mesh.extents).tolist(),
        "positive_z_vertex_fraction": positive_z_fraction,
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
    }


def render_registered_depth_pair(
    mesh_path: str | Path,
    *,
    camera_matrix: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Render a registered OpenCV-camera mesh with paired winding passes."""

    import pyrender  # noqa: PLC0415
    import trimesh  # noqa: PLC0415

    mesh_file = _require_file(mesh_path, "registered SPAR3D mesh")
    mesh, mesh_stats = _load_registered_triangle_mesh(mesh_file)
    k = np.asarray(camera_matrix, dtype=np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError("camera_matrix must be finite (3,3)")
    if width <= 0 or height <= 0:
        raise ValueError("render dimensions must be positive")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    def scene_for(reverse_winding: bool) -> Any:
        render_faces = faces[:, ::-1] if reverse_winding else faces
        plain = trimesh.Trimesh(
            vertices=vertices,
            faces=render_faces,
            process=False,
        )
        scene = pyrender.Scene(bg_color=(0.0, 0.0, 0.0, 0.0))
        scene.add(
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
        scene.add(
            pyrender.Mesh.from_trimesh(plain, smooth=False),
            pose=T_CV_TO_GL,
        )
        return scene

    renderer = pyrender.OffscreenRenderer(int(width), int(height))
    try:
        normal = np.asarray(
            renderer.render(scene_for(False), flags=pyrender.RenderFlags.DEPTH_ONLY),
            dtype=np.float32,
        )
        reversed_depth = np.asarray(
            renderer.render(scene_for(True), flags=pyrender.RenderFlags.DEPTH_ONLY),
            dtype=np.float32,
        )
    finally:
        renderer.delete()
    front, back, mask, pair_stats = orient_depth_pair(normal, reversed_depth)
    if int(mask.sum()) < 100:
        raise ComparisonInputError("registered mesh produced too few paired depth pixels")
    return front, back, mask, {**mesh_stats, **pair_stats}


def registered_front_raster_crosscheck(
    rendered_front: np.ndarray,
    persisted_front: np.ndarray,
) -> dict[str, Any]:
    """Confirm that GLB re-rasterization matches registration's saved front pass."""

    rendered = np.asarray(rendered_front, dtype=np.float32)
    persisted = np.asarray(persisted_front, dtype=np.float32)
    if rendered.shape != persisted.shape or rendered.ndim != 2:
        raise ComparisonInputError(
            "registered front-depth raster and comparison raster have different shapes"
        )
    rendered_valid = np.isfinite(rendered) & (rendered > 0.02) & (rendered < 5.0)
    persisted_valid = (
        np.isfinite(persisted) & (persisted > 0.02) & (persisted < 5.0)
    )
    intersection = rendered_valid & persisted_valid
    union = rendered_valid | persisted_valid
    iou = float(intersection.sum()) / max(int(union.sum()), 1)
    median_error = (
        float(np.median(np.abs(rendered[intersection] - persisted[intersection])))
        if intersection.any()
        else math.inf
    )
    result = {
        "rendered_positive_pixels": int(rendered_valid.sum()),
        "persisted_positive_pixels": int(persisted_valid.sum()),
        "intersection_pixels": int(intersection.sum()),
        "silhouette_iou": iou,
        "median_camera_z_difference_m": median_error,
    }
    if int(intersection.sum()) < 100 or iou < 0.95 or median_error > 0.005:
        raise ComparisonInputError(
            "registered GLB re-rasterization does not match its persisted front depth: "
            f"{result}"
        )
    return result


def bounded_spatial_close(
    hidden_mask: np.ndarray,
    eligibility_mask: np.ndarray,
    *,
    radius_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Close small image-space holes without escaping valid depth support."""

    hidden = np.asarray(hidden_mask, dtype=bool)
    eligibility = np.asarray(eligibility_mask, dtype=bool)
    if hidden.shape != eligibility.shape or hidden.ndim != 2:
        raise ValueError("hidden/eligibility masks must share one 2-D shape")
    if not 0 <= radius_px <= 12:
        raise ValueError("spatial close radius must be in 0..12 pixels")
    if np.any(hidden & ~eligibility):
        raise ValueError("raw hidden mask escaped spatial-filter eligibility")
    if radius_px == 0:
        return hidden.copy(), np.zeros_like(hidden)
    size = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(
        hidden.astype(np.uint8), cv2.MORPH_CLOSE, kernel
    ).astype(bool)
    added = closed & eligibility & ~hidden
    return hidden | added, added


def build_static_occlusion_masks(
    *,
    hand_mask: np.ndarray,
    finger_labels: np.ndarray,
    robot_depth: np.ndarray,
    object_support_mask: np.ndarray,
    mesh_mask: np.ndarray,
    front_depth: np.ndarray,
    back_depth: np.ndarray,
    current_mask: np.ndarray,
    contact_baseline_mask: np.ndarray,
    thumb_shell_m: float,
    finger_shell_m: float,
    palm_shell_m: float,
    spatial_close_radius_px: int,
    spatial_front_slack_m: float,
) -> dict[str, np.ndarray]:
    """Build the three fair static masks while retaining shared HaCo evidence."""

    hand = np.asarray(hand_mask, dtype=bool)
    labels = np.asarray(finger_labels, dtype=np.uint8)
    robot_z = np.asarray(robot_depth, dtype=np.float32)
    object_support = np.asarray(object_support_mask, dtype=bool)
    mesh = np.asarray(mesh_mask, dtype=bool)
    front = np.asarray(front_depth, dtype=np.float32)
    back = np.asarray(back_depth, dtype=np.float32)
    current = np.asarray(current_mask, dtype=bool)
    contact = np.asarray(contact_baseline_mask, dtype=bool)
    values = (hand, labels, robot_z, object_support, mesh, front, back, current, contact)
    if not all(value.shape == hand.shape for value in values) or hand.ndim != 2:
        raise ValueError("all static occlusion inputs must share one 2-D shape")
    if np.any(current & ~hand):
        raise ValueError("current barrier mask escaped XHand")
    if np.any(contact & ~hand):
        raise ValueError("shared HaCo/contact mask escaped XHand")
    shell_values = (thumb_shell_m, finger_shell_m, palm_shell_m)
    if any(not math.isfinite(value) or value < 0 for value in shell_values):
        raise ValueError("XHand shell values must be finite and non-negative")
    if not math.isfinite(spatial_front_slack_m) or spatial_front_slack_m < 0:
        raise ValueError("spatial front slack must be finite and non-negative")

    hand_labels = semantic_hand_labels(hand, labels)
    shell = thickness_map(
        hand_labels,
        thumb_shell_m=thumb_shell_m,
        finger_shell_m=finger_shell_m,
        palm_shell_m=palm_shell_m,
    )
    classification, support = classify_mesh_volume(
        hand_mask=hand,
        robot_depth=robot_z,
        object_support_mask=object_support,
        mesh_mask=mesh,
        front_depth=front,
        back_depth=back,
        pose_valid=True,
        shell_m=shell,
    )
    front_hidden = front_only_hidden(
        support=support,
        robot_depth=robot_z,
        front_depth=front,
    )
    volume_hidden = hidden_from_classification(classification)
    eligibility = mesh_temporal_eligibility(
        classification_support=support,
        robot_depth=robot_z,
        front_depth=front,
        shell_m=shell,
        front_slack_m=spatial_front_slack_m,
    )
    # Every shell-volume hidden pixel must be eligible; eligibility additionally
    # admits only a narrow front-slack band in which closing may fill pinholes.
    if np.any(volume_hidden & ~eligibility):
        raise RuntimeError("volume barrier escaped spatial-filter eligibility")
    filtered_hidden, spatial_added = bounded_spatial_close(
        volume_hidden,
        eligibility,
        radius_px=spatial_close_radius_px,
    )
    spar_front = combine_with_baseline(contact, front_hidden, hand)
    spar_volume_filter = combine_with_baseline(contact, filtered_hidden, hand)
    return {
        "current": current,
        "contact_baseline": contact,
        "classification": classification,
        "classification_support": support,
        "shell_m": shell,
        "spar_front_hidden": front_hidden,
        "spar_volume_hidden_raw": volume_hidden,
        "spatial_filter_eligibility": eligibility,
        "spatial_filter_added": spatial_added,
        "spar_front": spar_front,
        "spar_volume_filter": spar_volume_filter,
    }


def weighted_remap_depth(
    depth: np.ndarray, map_x: np.ndarray, map_y: np.ndarray
) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(values) & (values > 0)
    numerator = cv2.remap(
        np.where(valid, values, 0.0),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    denominator = cv2.remap(
        valid.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    output = np.zeros_like(numerator, dtype=np.float32)
    np.divide(numerator, denominator, out=output, where=denominator > 0.5)
    return output


def undistortion_maps(
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    return cv2.initUndistortRectifyMap(
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64),
        np.eye(3, dtype=np.float64),
        np.asarray(camera_matrix, dtype=np.float64),
        (int(width), int(height)),
        cv2.CV_32FC1,
    )


def _remap_image(value: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        np.asarray(value),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def _remap_mask(value: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        np.asarray(value, dtype=np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)


def _remap_labels(value: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        np.asarray(value, dtype=np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(np.uint8)


def _load_array_frame(path: Path, frame_index: int) -> np.ndarray:
    array = np.load(_require_file(path, path.name), mmap_mode="r", allow_pickle=False)
    if not 0 <= frame_index < len(array):
        raise ComparisonInputError(f"frame {frame_index} outside {path}")
    return np.asarray(array[frame_index]).copy()


def _read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(_require_file(path, path.name)))
    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index)):
            raise ComparisonInputError(f"could not seek {path} to frame {frame_index}")
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ComparisonInputError(f"could not decode {path} frame {frame_index}")
    return frame


def shared_square_roi(mask: np.ndarray, *, margin_px: int = 72) -> tuple[int, int, int, int]:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or not value.any():
        raise ValueError("ROI mask must be a nonempty 2-D mask")
    if margin_px < 0:
        raise ValueError("ROI margin must be non-negative")
    height, width = value.shape
    y, x = np.nonzero(value)
    center_x = 0.5 * (int(x.min()) + int(x.max()) + 1)
    center_y = 0.5 * (int(y.min()) + int(y.max()) + 1)
    side = max(int(x.max() - x.min() + 1), int(y.max() - y.min() + 1)) + 2 * margin_px
    side = min(max(side, 64), width, height)
    left = int(round(center_x - side / 2))
    top = int(round(center_y - side / 2))
    left = int(np.clip(left, 0, width - side))
    top = int(np.clip(top, 0, height - side))
    return left, top, left + side, top + side


def _annotated_panel(
    frame: np.ndarray,
    *,
    title: str,
    subtitle: str,
    width: int,
    height: int,
) -> np.ndarray:
    image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    header = np.full((72, width, 3), 18, dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        subtitle,
        (14, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (205, 220, 230),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, image))


def make_comparison_grid(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    *,
    titles: list[str],
    static_frame_index: int,
    cell_width: int,
    cell_height: int,
) -> np.ndarray:
    if not (len(frames) == len(masks) == len(titles) == 3):
        raise ValueError("comparison grid requires exactly three methods")
    panels = []
    for frame, mask, title in zip(frames, masks, titles):
        subtitle = (
            f"STATIC MH frame {static_frame_index} | hidden XHand pixels={int(mask.sum()):,}"
        )
        panels.append(
            _annotated_panel(
                frame,
                title=title,
                subtitle=subtitle,
                width=cell_width,
                height=cell_height,
            )
        )
    return np.hstack(panels)


def _classification_overlay(
    frame: np.ndarray,
    classification: np.ndarray,
    filter_added: np.ndarray,
) -> np.ndarray:
    output = np.asarray(frame, dtype=np.uint8).copy()
    classes = np.asarray(classification, dtype=np.uint8)
    colours = {
        int(CLASS_FRONT_OF): np.asarray((255, 255, 0), dtype=np.float32),
        int(CLASS_INTERSECTING): np.asarray((255, 0, 255), dtype=np.float32),
        int(CLASS_FULLY_BEHIND): np.asarray((0, 255, 255), dtype=np.float32),
    }
    for class_id, colour in colours.items():
        mask = classes == class_id
        output[mask] = np.clip(
            0.35 * output[mask].astype(np.float32) + 0.65 * colour,
            0,
            255,
        ).astype(np.uint8)
    added = np.asarray(filter_added, dtype=bool)
    output[added] = (30, 255, 30)
    cv2.rectangle(output, (8, 8), (920, 68), (12, 12, 12), -1)
    cv2.putText(
        output,
        "SPAR volume evidence: cyan=front, magenta=intersect, yellow=behind",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "green=bounded spatial-close addition | visual camera-ray classes only",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 230, 235),
        1,
        cv2.LINE_AA,
    )
    return output


def _depth_panel(
    front: np.ndarray, back: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    valid = np.asarray(mask, dtype=bool)
    thickness = np.zeros_like(front, dtype=np.float32)
    thickness[valid] = np.maximum(back[valid] - front[valid], 0.0)

    def colour(values: np.ndarray, title: str, inverse: bool = False) -> np.ndarray:
        canvas = np.zeros((*values.shape, 3), dtype=np.uint8)
        samples = values[valid]
        if len(samples):
            low, high = np.quantile(samples, (0.01, 0.99))
            normalized = np.zeros(values.shape, dtype=np.uint8)
            normalized[valid] = np.clip(
                255 * (values[valid] - low) / max(float(high - low), 1.0e-6),
                0,
                255,
            ).astype(np.uint8)
            if inverse:
                normalized[valid] = 255 - normalized[valid]
            canvas = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
            canvas[~valid] = 0
        cv2.putText(
            canvas,
            title,
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return cv2.resize(canvas, (640, 360), interpolation=cv2.INTER_AREA)

    return np.hstack(
        (
            colour(front, "SPAR front camera-Z", inverse=True),
            colour(back, "SPAR back camera-Z", inverse=True),
            colour(thickness, "learned ray thickness"),
        )
    )


def write_static_video(
    image_path: Path,
    output_path: Path,
    *,
    duration_s: float = 3.0,
    fps: int = 24,
) -> dict[str, Any]:
    """Encode a labelled repeated still as H.264; never imply a time interval."""

    if not math.isfinite(duration_s) or duration_s <= 0 or fps <= 0:
        raise ValueError("static-video duration/fps must be positive")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return {"written": False, "reason": "ffmpeg_not_found"}
    frames = int(round(duration_s * fps))
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(fps),
        "-i",
        str(image_path),
        "-frames:v",
        str(frames),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg static diagnostic failed: {result.stderr.strip()}")
    return {
        "written": True,
        "duration_s": duration_s,
        "fps": fps,
        "frames": frames,
        "content": "one labelled static comparison image repeated for every frame",
    }


def _output_record(path: Path) -> dict[str, Any]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.expanduser().resolve()
    second = second.expanduser().resolve()
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def validate_output_destination(
    output_dir: str | Path,
    *,
    input_records: dict[str, dict[str, Any]],
    processed_root: str | Path,
    repo_root: str | Path = REPO_ROOT,
    allowed_repo_output_root: str | Path = PILOT_ROOT,
) -> Path:
    """Reject broad, source-overlapping, weight, or repository-code targets."""

    output = Path(output_dir).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    allowed = Path(allowed_repo_output_root).expanduser().resolve()
    processed = Path(processed_root).expanduser().resolve()
    broad_roots = {
        Path("/").resolve(),
        Path.home().resolve(),
        *(Path(value).resolve() for value in ("/tmp", "/var", "/usr", "/etc", "/opt")),
    }
    if output in broad_roots or output.parent == Path("/"):
        raise ComparisonInputError(f"unsafe broad output directory: {output}")
    if output == repo or repo.is_relative_to(output):
        raise ComparisonInputError(
            f"comparison output may not equal or contain the repository: {output}"
        )
    if output.is_relative_to(repo) and not output.is_relative_to(allowed):
        raise ComparisonInputError(
            "in-repository comparison output must stay in the focused Choco "
            f"pilot tree: {output} not under {allowed}"
        )

    protected_trees = {
        "processed source tree": processed,
        "project weights": (repo / "weights").resolve(),
        "third-party repositories": (repo / "third_party").resolve(),
        "Git metadata": (repo / ".git").resolve(),
    }
    for name, record in input_records.items():
        protected_trees[f"input parent {name}"] = Path(record["path"]).resolve().parent
    for label, protected in protected_trees.items():
        if _paths_overlap(output, protected):
            raise ComparisonInputError(
                f"comparison output overlaps protected {label}: {output} vs {protected}"
            )
    return output


def stale_transaction_artifacts(output: Path) -> list[Path]:
    parent = output.parent
    result = []
    backup = output.with_name(f".{output.name}.backup")
    lock = output.with_name(f".{output.name}.lock")
    if backup.exists():
        result.append(backup)
    if lock.exists():
        result.append(lock)
    result.extend(sorted(parent.glob(f".{output.name}.staging.*")))
    return result


def validate_existing_output_bundle(output: Path) -> dict[str, Any]:
    """Allow overwrite only for an intact prior output of this exact method."""

    try:
        report_path, report = _load_json(
            output / "report.json", "existing comparison report"
        )
        manifest_path, manifest = _load_json(
            output / "publish_manifest.json", "existing comparison publish manifest"
        )
    except (FileNotFoundError, ComparisonInputError) as exc:
        raise ComparisonInputError(
            f"refusing to overwrite a directory without an intact prior bundle: {output}"
        ) from exc
    if report.get("method") != METHOD or report.get("status") != "complete":
        raise ComparisonInputError(
            f"refusing to overwrite unrelated/incomplete directory: {output}"
        )
    if (
        manifest.get("kind") != "atomic_comparison_publish_manifest"
        or manifest.get("status") != "complete"
        or manifest.get("method") != METHOD
        or Path(manifest.get("output_directory", "")).expanduser().resolve() != output
    ):
        raise ComparisonInputError("existing output publish manifest is not authoritative")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ComparisonInputError("existing output publish manifest has no file records")
    actual_content = {
        path.name for path in output.iterdir() if path.is_file()
    } - {manifest_path.name}
    if actual_content != set(files):
        raise ComparisonInputError(
            "existing output contains unrecorded or missing content artifacts"
        )
    for name, record in files.items():
        expected = output / name
        validate_declared_file_record(
            record,
            expected_path=expected,
            description=f"existing comparison output {name}",
        )
    if report_path.name not in files:
        raise ComparisonInputError("existing publish manifest does not bind report.json")
    return manifest


@contextmanager
def output_transaction_guard(
    output: Path,
    *,
    overwrite: bool,
) -> Iterator[None]:
    """Serialize one output and fail closed on stale transaction debris."""

    output.parent.mkdir(parents=True, exist_ok=True)
    stale = stale_transaction_artifacts(output)
    if stale:
        raise ComparisonInputError(
            "stale/concurrent comparison transaction artifacts exist: "
            + ", ".join(str(path) for path in stale)
        )
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {output}")
    if output.exists():
        if not output.is_dir() or output.is_symlink():
            raise ComparisonInputError(
                f"comparison output exists but is not a real directory: {output}"
            )
        validate_existing_output_bundle(output)
    lock = output.with_name(f".{output.name}.lock")
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise ComparisonInputError(f"comparison output is locked: {lock}") from exc
    try:
        owner = {
            "pid": os.getpid(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "output": str(output),
        }
        (lock / "owner.json").write_text(
            json.dumps(owner, sort_keys=True) + "\n", encoding="utf-8"
        )
        # A non-cooperating process could have created debris between the first
        # check and lock acquisition.  Recheck everything except our own lock.
        raced = [path for path in stale_transaction_artifacts(output) if path != lock]
        if raced:
            raise ComparisonInputError(
                "comparison transaction artifact appeared during lock acquisition: "
                + ", ".join(str(path) for path in raced)
            )
        if output.exists():
            if not overwrite:
                raise FileExistsError(f"output appeared while locking: {output}")
            if not output.is_dir() or output.is_symlink():
                raise ComparisonInputError(
                    f"output appeared as a non-directory while locking: {output}"
                )
            validate_existing_output_bundle(output)
        yield
    finally:
        shutil.rmtree(lock, ignore_errors=False)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_staging_tree(staging: Path) -> None:
    for path in sorted(staging.iterdir()):
        if path.is_file():
            _fsync_file(path)
    descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def finalize_staging_metadata(
    staging: Path,
    output: Path,
    *,
    report: dict[str, Any],
    input_snapshot_sha256: str,
) -> dict[str, Any]:
    """Write report after payloads, then a final completeness manifest."""

    report_path = staging / "report.json"
    manifest_path = staging / "publish_manifest.json"
    if report_path.exists() or manifest_path.exists():
        raise RuntimeError("staging control metadata already exists")
    declared_outputs = report.get("outputs")
    if not isinstance(declared_outputs, dict):
        raise RuntimeError("report outputs must be a mapping before finalization")
    payload_files = {
        path.name: path for path in staging.iterdir() if path.is_file()
    }
    if set(declared_outputs) != set(payload_files):
        raise RuntimeError("report output records do not cover the staged payload set")
    for name, path in payload_files.items():
        actual = _output_record(path)
        declared = declared_outputs[name]
        if (
            not isinstance(declared, dict)
            or int(declared.get("bytes", -1)) != actual["bytes"]
            or str(declared.get("sha256")) != actual["sha256"]
            or Path(declared.get("path", "")).expanduser().resolve()
            != (output / name).resolve()
        ):
            raise RuntimeError(f"stale report output record for {name}")
    report.setdefault("publication", {})
    report["publication"].update(
        {
            "atomic_directory_publish": True,
            "payloads_written_before_report": True,
            "report_written_before_publish_manifest": True,
            "publish_manifest": "publish_manifest.json",
            "publish_manifest_is_final_completeness_sentinel": True,
        }
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    content_files = sorted(
        path for path in staging.iterdir() if path.is_file() and path != manifest_path
    )
    records = {}
    for path in content_files:
        record = _output_record(path)
        record["path"] = str(output / path.name)
        records[path.name] = record
    manifest = {
        "schema_version": 1,
        "kind": "atomic_comparison_publish_manifest",
        "status": "complete",
        "method": METHOD,
        "output_directory": str(output),
        "input_snapshot_sha256": input_snapshot_sha256,
        "files": records,
        "file_count": len(records),
        "self_hash_note": (
            "publish_manifest.json is the final completeness sentinel and cannot "
            "embed its own content hash without recursion"
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # No staging content may be written after this point.  The manifest is the
    # last file and records every payload plus report.json by exact bytes/SHA.
    expected = {path.name for path in staging.iterdir() if path.is_file()} - {
        manifest_path.name
    }
    if set(records) != expected:
        raise RuntimeError("publish manifest does not cover every content artifact")
    return manifest


def _path_replace(source: Path, target: Path) -> None:
    source.replace(target)


def _publish_directory(
    staging: Path,
    output: Path,
    *,
    overwrite: bool,
    rename: Callable[[Path, Path], None] = _path_replace,
) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output exists (pass --overwrite): {output}")
        backup = output.with_name(f".{output.name}.backup")
        if backup.exists():
            raise FileExistsError(f"stale comparison backup exists: {backup}")
        rename(output, backup)
        fsync_directory(output.parent)
        new_output_materialized = False
        try:
            rename(staging, output)
            new_output_materialized = True
            fsync_directory(output.parent)
        except BaseException:
            if output.exists():
                try:
                    rename(output, staging)
                except BaseException as rollback_exc:
                    raise RuntimeError(
                        "publish failed and the new output could not be moved back; "
                        f"old output remains recoverable at {backup}"
                    ) from rollback_exc
            elif new_output_materialized:
                raise RuntimeError(
                    "publish reported a materialized output which then disappeared; "
                    f"old output remains recoverable at {backup}"
                )
            rename(backup, output)
            fsync_directory(output.parent)
            raise
        shutil.rmtree(backup)
        fsync_directory(output.parent)
    else:
        rename(staging, output)
        fsync_directory(output.parent)


def _mask_counts(masks: dict[str, np.ndarray]) -> dict[str, Any]:
    classification = masks["classification"]
    return {
        "current_hidden_xhand_pixels": int(masks["current"].sum()),
        "shared_object3d_force_finger_baseline_pixels": int(
            masks["contact_baseline"].sum()
        ),
        "valid_mesh_hand_support_pixels": int(masks["classification_support"].sum()),
        "front_of_pixels": int((classification == CLASS_FRONT_OF).sum()),
        "intersecting_pixels": int((classification == CLASS_INTERSECTING).sum()),
        "fully_behind_pixels": int((classification == CLASS_FULLY_BEHIND).sum()),
        "spar_front_hidden_raw_pixels": int(masks["spar_front_hidden"].sum()),
        "spar_front_final_pixels": int(masks["spar_front"].sum()),
        "spar_volume_hidden_raw_pixels": int(masks["spar_volume_hidden_raw"].sum()),
        "spatial_filter_eligible_pixels": int(masks["spatial_filter_eligibility"].sum()),
        "spatial_filter_added_pixels": int(masks["spatial_filter_added"].sum()),
        "spar_volume_filter_final_pixels": int(masks["spar_volume_filter"].sum()),
    }


def comparison_panel_titles(auxiliary_mask_increment_pixels: int) -> list[str]:
    """Label the panels without implying that SH improved the final mask."""

    increment = int(auxiliary_mask_increment_pixels)
    if increment < 0:
        raise ValueError("auxiliary mask increment must be non-negative")
    return [
        f"1 Current 2.5D/shell (SH mask delta {increment} px)",
        "2 SPAR front-Z + shared Object3D-force baseline",
        "3 SPAR volume/shell + bounded close",
    ]


def run_comparison(
    *,
    input_manifest: str | Path,
    mh_image: str | Path,
    registered_mesh: str | Path,
    registration_report: str | Path,
    processed_root: str | Path,
    output_dir: str | Path,
    thumb_shell_m: float = 0.01958,
    finger_shell_m: float = 0.01465,
    palm_shell_m: float = 0.015,
    spatial_close_radius_px: int = 3,
    spatial_front_slack_m: float = 0.003,
    overwrite: bool = False,
) -> dict[str, Any]:
    preflight = preflight_job(
        input_manifest=input_manifest,
        mh_image=mh_image,
        registered_mesh=registered_mesh,
        registration_report=registration_report,
        processed_root=processed_root,
    )
    if preflight["status"] != "ready":
        raise FileNotFoundError(
            f"registered SPAR3D mesh is not ready: {preflight['status']}"
        )
    input_records = preflight["checks"]["input_snapshot"]["records"]
    output = validate_output_destination(
        output_dir,
        input_records=input_records,
        processed_root=processed_root,
    )
    with output_transaction_guard(output, overwrite=overwrite):
        return _run_comparison_locked(
            preflight=preflight,
            output=output,
            input_manifest=input_manifest,
            mh_image=mh_image,
            registered_mesh=registered_mesh,
            registration_report=registration_report,
            processed_root=processed_root,
            thumb_shell_m=thumb_shell_m,
            finger_shell_m=finger_shell_m,
            palm_shell_m=palm_shell_m,
            spatial_close_radius_px=spatial_close_radius_px,
            spatial_front_slack_m=spatial_front_slack_m,
            overwrite=overwrite,
        )


def _run_comparison_locked(
    *,
    preflight: dict[str, Any],
    output: Path,
    input_manifest: str | Path,
    mh_image: str | Path,
    registered_mesh: str | Path,
    registration_report: str | Path,
    processed_root: str | Path,
    thumb_shell_m: float = 0.01958,
    finger_shell_m: float = 0.01465,
    palm_shell_m: float = 0.015,
    spatial_close_radius_px: int = 3,
    spatial_front_slack_m: float = 0.003,
    overwrite: bool = False,
) -> dict[str, Any]:
    if preflight["status"] != "ready":
        raise FileNotFoundError(
            f"registered SPAR3D mesh is not ready: {preflight['status']}"
        )
    manifest_path, manifest = load_pilot_manifest(input_manifest)
    frame_index = int(manifest["selection"]["mh_frame_index"])
    image_path = _require_file(mh_image, "selected MH image")
    raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if raw is None:
        raise ComparisonInputError(f"OpenCV could not decode {image_path}")
    height, width = raw.shape[:2]
    calibration = manifest["calibration"]["intrinsics_by_view"]["MH"]
    camera_matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(
        calibration["distortion_k1_k2_p1_p2_k3"], dtype=np.float64
    )
    map_x, map_y = undistortion_maps(
        camera_matrix, distortion, width=width, height=height
    )
    raw_u = _remap_image(raw, map_x, map_y)

    paths = source_paths(processed_root)
    method_sources = validate_current_method_reports(paths, frame_index=frame_index)
    report_path, registration, mesh_record = validate_registration_report(
        registration_report,
        registered_mesh,
        expected_frame_index=frame_index,
        expected_manifest_path=manifest_path,
        expected_mh_image_path=image_path,
        expected_modal_mask_path=Path(
            manifest["outputs"]["modal_mask"]["path"]
        ).expanduser().resolve(),
    )
    report_k = np.asarray(
        registration.get("camera", {}).get(
            "camera_matrix_original_and_undistorted"
        ),
        dtype=np.float64,
    )
    if report_k.shape != (3, 3) or not np.allclose(report_k, camera_matrix, atol=1.0e-9):
        raise ComparisonInputError("registration/comparison MH intrinsics differ")

    background = _read_video_frame(paths["background_video"], frame_index)
    if background.shape[:2] != (height, width):
        background = cv2.resize(background, (width, height), interpolation=cv2.INTER_AREA)
    background_u = _remap_image(background, map_x, map_y)
    object_support = _remap_mask(
        _load_array_frame(paths["object_support_mask"], frame_index), map_x, map_y
    )
    object_restore = _remap_mask(
        _load_array_frame(paths["object_restore_mask"], frame_index), map_x, map_y
    )
    if np.any(object_restore & ~object_support):
        raise ComparisonInputError("object restore mask escaped amodal support")

    resized = resize_overlay_frame(
        _load_array_frame(paths["overlay_rgb"], frame_index),
        _load_array_frame(paths["overlay_depth"], frame_index),
        _load_array_frame(paths["overlay_robot_mask"], frame_index),
        _load_array_frame(paths["overlay_hand_mask"], frame_index),
        _load_array_frame(paths["overlay_finger_labels"], frame_index),
        width=width,
        height=height,
    )
    robot_rgb, robot_depth, robot_mask, hand_mask, finger_labels = resized
    robot_rgb_u = _remap_image(robot_rgb, map_x, map_y)
    robot_depth_u = weighted_remap_depth(robot_depth, map_x, map_y)
    robot_mask_u = _remap_mask(robot_mask, map_x, map_y)
    hand_mask_u = _remap_mask(hand_mask, map_x, map_y)
    finger_labels_u = _remap_labels(finger_labels, map_x, map_y)
    current_mask_u = _remap_mask(
        _load_array_frame(paths["current_mask"], frame_index), map_x, map_y
    )
    contact_mask_u = _remap_mask(
        _load_array_frame(paths["contact_baseline_mask"], frame_index), map_x, map_y
    )

    front_depth, back_depth, mesh_mask, render_stats = render_registered_depth_pair(
        registered_mesh,
        camera_matrix=camera_matrix,
        width=width,
        height=height,
    )
    try:
        registered_front_record = registration["outputs"][
            "registered_front_depth_proxy_m.npy"
        ]
        registered_front_path = Path(
            registered_front_record["path"]
        ).expanduser().resolve()
        registered_front_bytes = int(registered_front_record["bytes"])
        registered_front_hash = str(registered_front_record["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ComparisonInputError(
            "registration report lacks its persisted front-depth record"
        ) from exc
    _require_file(registered_front_path, "registered front-depth raster")
    if (
        registered_front_path.stat().st_size != registered_front_bytes
        or sha256_file(registered_front_path) != registered_front_hash
    ):
        raise ComparisonInputError(
            "registered front-depth raster differs from its registration report"
        )
    persisted_front = np.load(registered_front_path, allow_pickle=False)
    raster_crosscheck = registered_front_raster_crosscheck(
        front_depth, persisted_front
    )
    comparison_support = mesh_mask & object_support
    mesh_outside_observed = mesh_mask & ~object_support
    if int(comparison_support.sum()) < 100:
        raise ComparisonInputError("too little SPAR/observed-object support overlap")
    front_depth = np.where(comparison_support, front_depth, 0.0).astype(np.float32)
    back_depth = np.where(comparison_support, back_depth, 0.0).astype(np.float32)

    masks = build_static_occlusion_masks(
        hand_mask=hand_mask_u,
        finger_labels=finger_labels_u,
        robot_depth=robot_depth_u,
        object_support_mask=object_support,
        mesh_mask=comparison_support,
        front_depth=front_depth,
        back_depth=back_depth,
        current_mask=current_mask_u,
        contact_baseline_mask=contact_mask_u,
        thumb_shell_m=thumb_shell_m,
        finger_shell_m=finger_shell_m,
        palm_shell_m=palm_shell_m,
        spatial_close_radius_px=spatial_close_radius_px,
        spatial_front_slack_m=spatial_front_slack_m,
    )
    counts = _mask_counts(masks)

    composite_background = restore_raw_object_pixels(
        background_u, raw_u, object_restore
    )
    method_masks = [
        masks["current"],
        masks["spar_front"],
        masks["spar_volume_filter"],
    ]
    composite_frames = []
    for mask in method_masks:
        final, _robot_only, _alpha = composite_frame(
            composite_background,
            robot_rgb_u,
            robot_mask_u,
            hand_mask_u,
            mask,
            robot_edge_sigma_px=0.6,
            occlusion_edge_sigma_px=0.0,
        )
        composite_frames.append(final)

    auxiliary_effect = method_sources["auxiliary_haco_effect"]
    auxiliary_mask_increment = int(
        auxiliary_effect["auxiliary_mask_increment_pixels"]
    )
    titles = comparison_panel_titles(auxiliary_mask_increment)
    full_grid = make_comparison_grid(
        composite_frames,
        method_masks,
        titles=titles,
        static_frame_index=frame_index,
        cell_width=640,
        cell_height=360,
    )
    roi_mask = object_support | (
        hand_mask_u
        & cv2.dilate(
            object_support.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)),
        ).astype(bool)
    )
    roi = shared_square_roi(roi_mask, margin_px=56)
    left, top, right, bottom = roi
    roi_frames = [value[top:bottom, left:right] for value in composite_frames]
    roi_masks = [value[top:bottom, left:right] for value in method_masks]
    roi_grid = make_comparison_grid(
        roi_frames,
        roi_masks,
        titles=titles,
        static_frame_index=frame_index,
        cell_width=480,
        cell_height=480,
    )
    evidence = _classification_overlay(
        raw_u, masks["classification"], masks["spatial_filter_added"]
    )
    depth_diagnostic = _depth_panel(front_depth, back_depth, comparison_support)

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        image_outputs = {
            "comparison_full_frame.png": full_grid,
            "comparison_roi.png": roi_grid,
            "spar_mesh_classification_evidence.png": evidence,
            "spar_mesh_depth_diagnostic.png": depth_diagnostic,
        }
        for name, image in image_outputs.items():
            if not cv2.imwrite(str(staging / name), image):
                raise RuntimeError(f"could not write {name}")
        np.save(staging / "spar_front_depth_proxy_m.npy", front_depth)
        np.save(staging / "spar_back_depth_proxy_m.npy", back_depth)
        np.save(staging / "spar_paired_object_support_mask.npy", comparison_support)
        np.save(staging / "spar_volume_classification.npy", masks["classification"])
        np.save(staging / "current_occluded_hand_mask.npy", masks["current"])
        np.save(staging / "spar_front_occluded_hand_mask.npy", masks["spar_front"])
        np.save(
            staging / "spar_volume_filter_occluded_hand_mask.npy",
            masks["spar_volume_filter"],
        )
        np.save(staging / "spatial_filter_added_mask.npy", masks["spatial_filter_added"])
        static_video = write_static_video(
            staging / "comparison_roi.png", staging / STATIC_VIDEO_NAME
        )

        bound_inputs = preflight["checks"]["input_snapshot"]
        before_records = bound_inputs["records"]
        verify_input_snapshot_unchanged(before_records)
        input_binding = {
            **bound_inputs,
            "verified_unchanged_before_publish": True,
            "verification_time_utc": datetime.now(timezone.utc).isoformat(),
        }

        output_names = list(image_outputs) + [
            "spar_front_depth_proxy_m.npy",
            "spar_back_depth_proxy_m.npy",
            "spar_paired_object_support_mask.npy",
            "spar_volume_classification.npy",
            "current_occluded_hand_mask.npy",
            "spar_front_occluded_hand_mask.npy",
            "spar_volume_filter_occluded_hand_mask.npy",
            "spatial_filter_added_mask.npy",
        ]
        if static_video["written"]:
            output_names.append(STATIC_VIDEO_NAME)
        report = {
            "schema_version": 1,
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            **comparison_contract(),
            "selection": {
                **manifest["selection"],
                "evaluated_frame_indices": [frame_index],
                "interval_video_generated": False,
                "reason": (
                    "the registered SPAR3D Sim(3) is valid only for its selected "
                    "MH frame; no tracked object transform was available"
                ),
            },
            "methods": {
                "current": {
                    "label": titles[0],
                    "geometry": "completed visible camera-Z height field",
                    "mask_source": str(paths["current_mask"]),
                    "uses_nominal_primitive_mesh": False,
                    "auxiliary_haco_effect": auxiliary_effect,
                },
                "spar_front": {
                    "label": titles[1],
                    "geometry": "registered SPAR3D front camera-Z raster",
                    "xhand_shell_m": 0.0,
                    "spatial_filter": False,
                    "shared_object3d_force_finger_baseline_retained": True,
                },
                "spar_volume_filter": {
                    "label": titles[2],
                    "geometry": "paired learned SPAR3D front/back camera-Z raster",
                    "xhand_shell_m": {
                        "thumb": thumb_shell_m,
                        "finger": finger_shell_m,
                        "palm": palm_shell_m,
                    },
                    "spatial_close_radius_px": spatial_close_radius_px,
                    "spatial_front_slack_m": spatial_front_slack_m,
                    "shared_object3d_force_finger_baseline_retained": True,
                    "filter_semantics": (
                        "image-space close restricted to valid hand/object/paired-mesh "
                        "support and a narrow front-depth slack band"
                    ),
                },
            },
            "registration": {
                "report": str(report_path),
                "mesh": mesh_record,
                "method": registration.get("method"),
                "metric_scale_verified": False,
                "uses_sh_for_registration": registration.get("uses_sh_for_this_registration"),
            },
            "camera": {
                "domain": "undistorted MH pinhole image",
                "camera_matrix": camera_matrix.tolist(),
                "distortion_k1_k2_p1_p2_k3": distortion.tolist(),
                "coordinate_frame": "OpenCV +X right, +Y down, +Z forward",
            },
            "dual_camera_scope": {
                "auxiliary_haco_input_available": auxiliary_effect[
                    "auxiliary_haco_input_available"
                ],
                "auxiliary_scores_fused": auxiliary_effect[
                    "auxiliary_scores_fused"
                ],
                "auxiliary_score_changed_frame_fingers": auxiliary_effect[
                    "auxiliary_score_changed_frame_fingers"
                ],
                "auxiliary_geometry_used": auxiliary_effect[
                    "auxiliary_geometry_used"
                ],
                "auxiliary_active_increment_frame_fingers": auxiliary_effect[
                    "auxiliary_active_increment_frame_fingers"
                ],
                "auxiliary_qualified_frame_fingers": auxiliary_effect[
                    "auxiliary_qualified_frame_fingers"
                ],
                "auxiliary_mask_increment_pixels": auxiliary_effect[
                    "auxiliary_mask_increment_pixels"
                ],
                "dual_camera_changed_final_mask": auxiliary_effect[
                    "dual_camera_changed_final_mask"
                ],
                "dual_camera_changed_mask_pixels": auxiliary_effect[
                    "dual_camera_changed_mask_pixels"
                ],
                "spar_mesh_registration_uses_mh_only": True,
                "sh_geometry_constraint_added": False,
            },
            "selected_frame_shared_baseline": auxiliary_effect[
                "selected_frame_baseline_attribution"
            ],
            "mesh_raster": {
                **render_stats,
                "registration_front_raster_crosscheck": raster_crosscheck,
                "paired_pixels_after_observed_object_clip": int(comparison_support.sum()),
                "mesh_pixels_outside_observed_object_clip": int(mesh_outside_observed.sum()),
                "front_back_semantics": (
                    "normal/reversed winding nearest hits, automatically ordered by camera-Z"
                ),
                "backside_measured": False,
            },
            "counts": counts,
            "roi_xyxy_exclusive": list(roi),
            "source_methods": method_sources,
            "input_binding": input_binding,
            "sources": {
                "input_manifest": str(manifest_path),
                "mh_image": str(image_path),
                **{key: str(value) for key, value in paths.items()},
            },
            "static_video": static_video,
            "invariants": {
                "single_registered_frame_only": True,
                "no_object_motion_fabricated": True,
                "same_robot_pose_for_all_methods": True,
                "same_shared_finger_baseline_for_spar_methods": True,
                "shared_baseline_subset_spar_front": bool(
                    np.all(~masks["contact_baseline"] | masks["spar_front"])
                ),
                "shared_baseline_subset_spar_volume_filter": bool(
                    np.all(~masks["contact_baseline"] | masks["spar_volume_filter"])
                ),
                "all_final_masks_subset_xhand": bool(
                    all(np.all(~value | hand_mask_u) for value in method_masks)
                ),
                "classification_partitions_valid_support": bool(
                    np.array_equal(
                        masks["classification"] > 0,
                        masks["classification_support"],
                    )
                ),
                "spatial_filter_does_not_escape_eligibility": bool(
                    np.all(
                        ~masks["spatial_filter_added"]
                        | masks["spatial_filter_eligibility"]
                    )
                ),
                "mesh_support_clipped_to_observed_object": True,
                "robot_pose_state_unchanged": True,
                "physical_collision_not_claimed": True,
                "all_inputs_bytes_sha_bound": True,
                "all_bound_inputs_reverified_before_publish": True,
                "output_destination_disjoint_from_protected_inputs": True,
                "atomic_sibling_directory_publish": True,
            },
            "limitations": [
                "SPAR3D hidden and rear geometry is a learned single-view estimate.",
                "The MH Sim(3) and Depth Anything/HaWoR camera-Z scale are approximate.",
                "The volume/shell/filter method only suppresses visible robot pixels; it does not change joint or object pose.",
                "The spatial close is an image-space pinhole filter, not a 3-D collision response.",
                "Only frame 187 is evaluated; the repeated-still MP4 is not a motion result.",
                "SH scores are fused as confidence evidence, but their active and final-mask increments are reported separately and may be zero.",
                "SH contributes no geometry and is not used by the SPAR3D registration.",
            ],
            "outputs": {},
        }
        if not all(report["invariants"].values()):
            raise RuntimeError("static comparison invariant failed")
        for name in output_names:
            report["outputs"][name] = _output_record(staging / name)
            report["outputs"][name]["path"] = str(output / name)
        publish_manifest = finalize_staging_metadata(
            staging,
            output,
            report=report,
            input_snapshot_sha256=input_binding["snapshot_sha256"],
        )
        if publish_manifest["file_count"] != len(output_names) + 1:
            raise RuntimeError("publish manifest file count mismatch")
        fsync_staging_tree(staging)
        _publish_directory(staging, output, overwrite=overwrite)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--mh-image", type=Path, default=DEFAULT_MH_IMAGE)
    parser.add_argument("--registered-mesh", type=Path, default=DEFAULT_REGISTERED_MESH)
    parser.add_argument(
        "--registration-report", type=Path, default=DEFAULT_REGISTRATION_REPORT
    )
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--thumb-shell-m", type=float, default=0.01958)
    parser.add_argument("--finger-shell-m", type=float, default=0.01465)
    parser.add_argument("--palm-shell-m", type=float, default=0.015)
    parser.add_argument("--spatial-close-radius-px", type=int, default=3)
    parser.add_argument("--spatial-front-slack-m", type=float, default=0.003)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    common = {
        "input_manifest": args.input_manifest,
        "mh_image": args.mh_image,
        "registered_mesh": args.registered_mesh,
        "registration_report": args.registration_report,
        "processed_root": args.processed_root,
    }
    if args.preflight:
        result = preflight_job(**common)
    else:
        result = run_comparison(
            **common,
            output_dir=args.output_dir,
            thumb_shell_m=args.thumb_shell_m,
            finger_shell_m=args.finger_shell_m,
            palm_shell_m=args.palm_shell_m,
            spatial_close_radius_px=args.spatial_close_radius_px,
            spatial_front_slack_m=args.spatial_front_slack_m,
            overwrite=args.overwrite,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
