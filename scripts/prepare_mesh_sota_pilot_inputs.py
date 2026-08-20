#!/usr/bin/env python3
"""Build a provenance-safe input bundle for the 08-05 Choco mesh pilot.

The bundle deliberately uses a frame where the observed modal, hand-cleaned,
and amodal supports are identical.  RGB is copied from the decoded camera
frames; completed videos and inferred hidden pixels are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODE_ROOT = (
    REPO_ROOT / "data" / "cube_dataset" / "26.08.05_stereo_calibrated" / "1"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "8-5" / "mesh_sota_pilot" / "episode_1" / "choco" / "inputs"
)
DEFAULT_MH_FRAME = 187
DEFAULT_LABEL = "Choco"
DEFAULT_CROP_SIZE = 512
DEFAULT_PADDING_RATIO = 0.20
EXPECTED_STEREO_SCHEMA = 3
EXPECTED_OBJECT_LAYER_SCHEMA = 1
EXPECTED_CALIBRATION_SCHEMA = 1
EXPECTED_MAPPING = {"camera_1": "SH", "camera_2": "MH"}
EXPECTED_CALIBRATION_MAPPING = {"camera_1": "MH", "camera_2": "SH"}
EXPECTED_PIPELINE_TO_CALIBRATION = {
    "camera_1": "camera_2",
    "camera_2": "camera_1",
}


def _json_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    """Return an actual JSON integer; never truncate a float such as 187.9."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {value}")
    return value


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _json_values_exact(value: Any, expected: Any) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(value, dict):
        return value.keys() == expected.keys() and all(
            _json_values_exact(value[key], expected[key]) for key in value
        )
    if isinstance(value, list):
        return len(value) == len(expected) and all(
            _json_values_exact(item, expected_item)
            for item, expected_item in zip(value, expected)
        )
    return value == expected


def _exact(value: Any, expected: Any, field: str) -> None:
    # JSON booleans compare equal to 0/1 in Python, so type is part of the
    # contract for all exact provenance comparisons.
    if not _json_values_exact(value, expected):
        raise ValueError(f"{field} must be {expected!r}, got {value!r}")


def _finite_number(
    value: Any, field: str, *, minimum: float | None = None
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite number")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {value!r}")
    return numeric


def _numeric_vector(value: Any, length: int, field: str) -> list[Any]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} numbers")
    for index, item in enumerate(value):
        _finite_number(item, f"{field}[{index}]")
    return value


def _numeric_matrix(value: Any, rows: int, columns: int, field: str) -> list[Any]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{field} must be a {rows}x{columns} numeric matrix")
    for index, row in enumerate(value):
        _numeric_vector(row, columns, f"{field}[{index}]")
    return value


def _declared_path(raw_path: Any, *, base: Path, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base / path
    return _required_file(path, field)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or _is_within(left, right) or _is_within(right, left)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"missing {description}: {path}")
    return path


def _load_json(path: Path, description: str) -> dict[str, Any]:
    path = _required_file(path, description)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description} JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return payload


def _file_record(path: Path, **metadata: Any) -> dict[str, Any]:
    path = _required_file(path, "source/output file")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **metadata,
    }


def _staged_output_record(
    staged_path: Path, final_path: Path, **metadata: Any
) -> dict[str, Any]:
    staged_path = _required_file(staged_path, "staged output file")
    final_path = final_path.expanduser().resolve()
    return {
        "path": str(final_path),
        "bytes": staged_path.stat().st_size,
        "sha256": sha256_file(staged_path),
        **metadata,
    }


def _validate_mapping_provenance(
    stereo: dict[str, Any], *, object_label: str
) -> tuple[dict[str, Any], int, int]:
    _exact(stereo.get("schema_version"), EXPECTED_STEREO_SCHEMA, "stereo schema_version")
    _exact(stereo.get("primary_view"), "MH", "stereo primary_view")
    _exact(stereo.get("auxiliary_view"), "SH", "stereo auxiliary_view")
    _exact(stereo.get("training_view"), "MH", "stereo training_view")
    _exact(stereo.get("robot_overlay_view"), "MH", "stereo robot_overlay_view")
    _exact(stereo.get("stereo_code_mapping"), EXPECTED_MAPPING, "stereo_code_mapping")
    _exact(
        stereo.get("frame_mapping"),
        "output frame k equals decoded source frame k",
        "stereo frame_mapping",
    )
    common_frames = _json_int(stereo.get("common_frames"), "stereo common_frames", minimum=1)
    label_vocabulary = stereo.get("label_vocabulary")
    if not isinstance(label_vocabulary, list) or not all(
        isinstance(label, str) and label for label in label_vocabulary
    ):
        raise ValueError("stereo label_vocabulary must be a list of non-empty strings")
    if len(set(label_vocabulary)) != len(label_vocabulary):
        raise ValueError("stereo label_vocabulary must not contain duplicates")
    if object_label not in label_vocabulary:
        raise ValueError(f"stereo label_vocabulary does not contain {object_label!r}")

    temporal = _mapping(stereo.get("temporal_alignment"), "temporal_alignment")
    _exact(
        temporal.get("reference_view"),
        "camera_2/MH/GT",
        "temporal_alignment.reference_view",
    )
    _exact(
        temporal.get("source_frames_reordered"),
        False,
        "temporal_alignment.source_frames_reordered",
    )
    _exact(
        temporal.get("apply_offset_only_during_dual_view_fusion"),
        True,
        "temporal_alignment.apply_offset_only_during_dual_view_fusion",
    )
    _exact(
        temporal.get("out_of_range_policy"),
        "fail_open",
        "temporal_alignment.out_of_range_policy",
    )
    offset = _json_int(
        temporal.get("camera1_frame_offset"),
        "temporal_alignment.camera1_frame_offset",
    )
    expected_lookup = (
        "camera1/SH source index = camera2/MH frame k + " f"({offset})"
    )
    _exact(
        temporal.get("camera1_lookup"),
        expected_lookup,
        "temporal_alignment.camera1_lookup",
    )
    return temporal, offset, common_frames


def _validate_annotation_provenance(
    *,
    stereo: dict[str, Any],
    stereo_manifest_path: Path,
    object_layer: dict[str, Any],
    object_layer_manifest_path: Path,
    object_label: str,
    frame_index: int,
    common_frames: int,
) -> tuple[Path, Path, dict[str, Any]]:
    _exact(
        object_layer.get("schema_version"),
        EXPECTED_OBJECT_LAYER_SCHEMA,
        "object-layer schema_version",
    )
    _exact(
        _json_int(object_layer.get("frame_count"), "object-layer frame_count"),
        common_frames,
        "object-layer frame_count",
    )
    _exact(
        object_layer.get("transition_policy"),
        "empty_mask",
        "object-layer transition_policy",
    )

    stereo_sources = _mapping(stereo.get("sources"), "stereo sources")
    annotation_path = _declared_path(
        stereo_sources.get("gt_labels"),
        base=stereo_manifest_path.parent,
        field="stereo sources.gt_labels",
    )
    layer_annotation_path = _declared_path(
        object_layer.get("labels_json"),
        base=object_layer_manifest_path.parent,
        field="object-layer labels_json",
    )
    if sha256_file(annotation_path) != sha256_file(layer_annotation_path):
        raise ValueError(
            "stereo and object-layer annotation files must be byte-identical"
        )
    annotation = _load_json(annotation_path, "ground-truth annotation")
    layer_annotation = _load_json(
        layer_annotation_path, "object-layer ground-truth annotation"
    )
    if annotation != layer_annotation:
        raise ValueError("stereo and object-layer annotation JSON values differ")

    episode = stereo.get("episode")
    if not isinstance(episode, str) or not episode:
        raise ValueError("stereo episode must be a non-empty string")
    _exact(annotation.get("episode"), episode, "annotation episode")
    _exact(
        _json_int(annotation.get("num_frames"), "annotation num_frames", minimum=1),
        common_frames,
        "annotation num_frames",
    )
    annotation_fps = annotation.get("fps")
    stereo_fps = stereo.get("fps")
    if (
        isinstance(annotation_fps, bool)
        or not isinstance(annotation_fps, (int, float))
        or not math.isfinite(float(annotation_fps))
    ):
        raise ValueError("annotation fps must be a finite number")
    _exact(annotation_fps, stereo_fps, "annotation fps")

    segments = annotation.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("annotation segments must be a non-empty list")
    expected_start = 0
    matches: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        segment = _mapping(segment, f"annotation segments[{index}]")
        start = _json_int(
            segment.get("start_frame"),
            f"annotation segments[{index}].start_frame",
            minimum=0,
        )
        end = _json_int(
            segment.get("end_frame"),
            f"annotation segments[{index}].end_frame",
            minimum=0,
        )
        if end < start:
            raise ValueError(f"annotation segment {index} has end before start")
        if start != expected_start:
            raise ValueError(
                f"annotation segments must be contiguous; expected start "
                f"{expected_start}, got {start}"
            )
        expected_start = end + 1
        label = segment.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"annotation segment {index} label must be a string")
        if label not in stereo["label_vocabulary"]:
            raise ValueError(
                f"annotation segment {index} label {label!r} is not in "
                "stereo label_vocabulary"
            )
        if label == object_label and start <= frame_index <= end:
            matches.append({"label": label, "start": start, "end": end})
    if expected_start != common_frames:
        raise ValueError(
            f"annotation segments end at {expected_start - 1}, expected "
            f"{common_frames - 1}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"MH frame {frame_index} must fall in exactly one annotation "
            f"{object_label!r} segment; found {len(matches)}"
        )
    return annotation_path, layer_annotation_path, matches[0]


def _validate_calibration_provenance(
    *,
    calibration: dict[str, Any],
    stereo_manifest_path: Path,
) -> tuple[Path, dict[str, Any]]:
    _exact(calibration.get("status"), "provided", "calibration status")
    _exact(
        calibration.get("schema_version"),
        EXPECTED_CALIBRATION_SCHEMA,
        "calibration schema_version",
    )
    _exact(
        calibration.get("calibration_camera_mapping"),
        EXPECTED_CALIBRATION_MAPPING,
        "calibration calibration_camera_mapping",
    )
    _exact(
        calibration.get("pipeline_camera_mapping"),
        EXPECTED_MAPPING,
        "calibration pipeline_camera_mapping",
    )
    _exact(
        calibration.get("pipeline_to_calibration_camera"),
        EXPECTED_PIPELINE_TO_CALIBRATION,
        "calibration pipeline_to_calibration_camera",
    )
    reference_path = _declared_path(
        calibration.get("reference_json"),
        base=stereo_manifest_path.parent,
        field="calibration reference_json",
    )
    declared_hash = calibration.get("reference_sha256")
    if (
        not isinstance(declared_hash, str)
        or len(declared_hash) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in declared_hash)
    ):
        raise ValueError("calibration reference_sha256 must be a SHA-256 digest")
    actual_hash = sha256_file(reference_path)
    if declared_hash.lower() != actual_hash:
        raise ValueError("calibration reference hash disagrees with stereo manifest")
    reference = _load_json(reference_path, "calibration reference")
    _exact(
        reference.get("schema_version"),
        EXPECTED_CALIBRATION_SCHEMA,
        "calibration reference schema_version",
    )
    created_utc = reference.get("created_utc")
    if not isinstance(created_utc, str) or not created_utc:
        raise ValueError("calibration reference created_utc must be a non-empty string")
    _exact(
        calibration.get("created_utc"),
        created_utc,
        "calibration created_utc",
    )
    reference_source = _mapping(reference.get("source"), "calibration reference source")
    calibration_size = calibration.get("image_size_wh")
    reference_size = reference_source.get("image_size_wh")
    for size, field in (
        (calibration_size, "calibration image_size_wh"),
        (reference_size, "calibration reference source.image_size_wh"),
    ):
        if not isinstance(size, list) or len(size) != 2:
            raise ValueError(f"{field} must be [width, height]")
        _json_int(size[0], f"{field}[0]", minimum=1)
        _json_int(size[1], f"{field}[1]", minimum=1)
    _exact(calibration_size, reference_size, "calibration image_size_wh")
    reference_checkerboard = _mapping(
        reference.get("checkerboard"), "calibration reference checkerboard"
    )
    checkerboard = _mapping(calibration.get("checkerboard"), "calibration checkerboard")
    square_size = reference_checkerboard.get("square_size_mm")
    if square_size is not None:
        _finite_number(
            square_size,
            "calibration reference checkerboard.square_size_mm",
            minimum=0.0,
        )
        if square_size <= 0:
            raise ValueError(
                "calibration reference checkerboard.square_size_mm must be positive"
            )
    length_unit = reference_checkerboard.get("length_unit")
    if not isinstance(length_unit, str) or not length_unit:
        raise ValueError(
            "calibration reference checkerboard.length_unit must be a string"
        )
    if not isinstance(reference_checkerboard.get("metric_scale_verified"), bool):
        raise ValueError(
            "calibration reference checkerboard.metric_scale_verified must be boolean"
        )
    for key in ("square_size_mm", "length_unit", "metric_scale_verified"):
        _exact(
            checkerboard.get(key),
            reference_checkerboard.get(key),
            f"calibration checkerboard.{key}",
        )

    intrinsics = _mapping(calibration.get("intrinsics_by_view"), "calibration intrinsics")
    for view, camera in (("MH", "camera_1"), ("SH", "camera_2")):
        view_intrinsics = _mapping(intrinsics.get(view), f"calibration intrinsics {view}")
        reference_camera = _mapping(reference.get(camera), f"calibration reference {camera}")
        _exact(
            view_intrinsics.get("calibration_camera"),
            camera,
            f"calibration intrinsics {view}.calibration_camera",
        )
        _numeric_matrix(
            reference_camera.get("camera_matrix"),
            3,
            3,
            f"calibration reference {camera}.camera_matrix",
        )
        _numeric_vector(
            reference_camera.get("distortion_k1_k2_p1_p2_k3"),
            5,
            f"calibration reference {camera}.distortion_k1_k2_p1_p2_k3",
        )
        _finite_number(
            reference_camera.get("rms_reprojection_px"),
            f"calibration reference {camera}.rms_reprojection_px",
            minimum=0.0,
        )
        for key in (
            "camera_matrix",
            "distortion_k1_k2_p1_p2_k3",
            "rms_reprojection_px",
        ):
            _exact(
                view_intrinsics.get(key),
                reference_camera.get(key),
                f"calibration intrinsics {view}.{key}",
            )

    relative = _mapping(
        calibration.get("relative_extrinsics"), "calibration relative_extrinsics"
    )
    reference_stereo = _mapping(reference.get("stereo"), "calibration reference stereo")
    _exact(relative.get("from_view"), "MH", "relative_extrinsics.from_view")
    _exact(relative.get("to_view"), "SH", "relative_extrinsics.to_view")
    _numeric_matrix(
        reference_stereo.get("T_camera2_from_camera1"),
        4,
        4,
        "calibration reference stereo.T_camera2_from_camera1",
    )
    translation_unit = reference_stereo.get("translation_unit")
    if not isinstance(translation_unit, str) or not translation_unit:
        raise ValueError(
            "calibration reference stereo.translation_unit must be a string"
        )
    for key in ("T_camera2_from_camera1", "translation_unit"):
        _exact(
            relative.get(key),
            reference_stereo.get(key),
            f"calibration relative_extrinsics.{key}",
        )
    return reference_path, reference


def validate_output_root(
    output_dir: Path,
    *,
    episode_root: Path,
    protected_paths: Sequence[Path],
) -> Path:
    """Reject output roots that could replace source, repo, or model trees."""

    expanded = output_dir.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"output root must not be a symbolic link: {expanded}")
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    parent = absolute.parent.resolve()
    if not parent.is_dir():
        raise ValueError(f"output root parent must already exist: {parent}")
    resolved = (parent / absolute.name).resolve()
    if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
        raise ValueError(f"refusing broad output root: {resolved}")
    if resolved.exists() and (resolved.is_symlink() or not resolved.is_dir()):
        raise ValueError(f"existing output root must be a real directory: {resolved}")

    episode_root = episode_root.resolve()
    if _paths_overlap(resolved, episode_root):
        raise ValueError(
            f"output root must not overlap source episode root: {resolved}"
        )
    repo_root = REPO_ROOT.resolve()
    if resolved == repo_root or _is_within(repo_root, resolved):
        raise ValueError(f"output root must not equal or contain repo root: {resolved}")
    for protected_root in (repo_root / "weights", repo_root / "third_party"):
        if _paths_overlap(resolved, protected_root.resolve()):
            raise ValueError(
                f"output root must not overlap protected model tree "
                f"{protected_root}: {resolved}"
            )
    for protected_path in protected_paths:
        protected_path = protected_path.expanduser().resolve()
        if _paths_overlap(resolved, protected_path):
            raise ValueError(
                f"output root overlaps protected source {protected_path}: {resolved}"
            )
    return resolved


def publish_bundle_directory(
    staged_dir: Path,
    output_dir: Path,
    *,
    replace_fn: Any = os.replace,
) -> None:
    """Atomically swap a complete staged bundle, restoring the old one on error."""

    staged_dir = staged_dir.resolve()
    output_dir = output_dir.resolve()
    if not staged_dir.is_dir():
        raise ValueError(f"missing staged bundle: {staged_dir}")
    if not (staged_dir / "manifest.json").is_file():
        raise ValueError("staged bundle has no manifest.json")
    if staged_dir.parent != output_dir.parent:
        raise ValueError("staged and final bundle must be sibling directories")
    backup_dir = staged_dir.with_name(staged_dir.name + ".previous")
    if backup_dir.exists():
        raise ValueError(f"unexpected existing bundle backup: {backup_dir}")

    moved_existing = False
    published_new = False
    try:
        if output_dir.exists():
            if output_dir.is_symlink() or not output_dir.is_dir():
                raise ValueError(f"refusing to replace unsafe output root: {output_dir}")
            replace_fn(output_dir, backup_dir)
            moved_existing = True
        replace_fn(staged_dir, output_dir)
        published_new = True
    except Exception as publication_error:
        rollback_errors: list[str] = []
        if published_new and output_dir.exists():
            try:
                shutil.rmtree(output_dir)
            except Exception as exc:  # pragma: no cover - filesystem failure
                rollback_errors.append(f"remove new bundle: {exc}")
        if moved_existing and backup_dir.exists():
            try:
                os.replace(backup_dir, output_dir)
            except Exception as exc:  # pragma: no cover - filesystem failure
                rollback_errors.append(f"restore previous bundle: {exc}")
        detail = (
            "; rollback errors: " + "; ".join(rollback_errors)
            if rollback_errors
            else ""
        )
        raise RuntimeError(
            f"bundle publication failed and was rolled back: "
            f"{publication_error}{detail}"
        ) from publication_error
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def _mask_frame(
    path: Path,
    frame_index: int,
    name: str,
    *,
    expected_frames: int | None = None,
) -> np.ndarray:
    path = _required_file(path, f"{name} mask array")
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load {name} mask array {path}: {error}") from error
    if array.ndim != 3:
        raise ValueError(
            f"{name} mask must have shape (frames, height, width), got {array.shape}"
        )
    if expected_frames is not None and array.shape[0] != expected_frames:
        raise ValueError(
            f"{name} mask has {array.shape[0]} frames, expected {expected_frames}"
        )
    if not 0 <= frame_index < array.shape[0]:
        raise ValueError(
            f"MH frame {frame_index} is outside {name} mask frame range "
            f"[0, {array.shape[0]})"
        )
    selected = np.asarray(array[frame_index])
    if selected.dtype != np.bool_:
        if not np.issubdtype(selected.dtype, np.number):
            raise ValueError(f"{name} mask has unsupported dtype {selected.dtype}")
        unique = np.unique(selected)
        if not np.all(np.isin(unique, (0, 1, 255))):
            raise ValueError(
                f"{name} mask must be binary, got values {unique[:10].tolist()}"
            )
    return selected.astype(bool, copy=True)


def _bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("selected object mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _square_crop_geometry(
    bbox: tuple[int, int, int, int], padding_ratio: float
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    width, height = x1 - x0, y1 - y0
    longest = max(width, height)
    pad = int(math.ceil(longest * padding_ratio))
    side = longest + 2 * pad
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    left = int(math.floor(center_x - side / 2.0))
    top = int(math.floor(center_y - side / 2.0))
    return left, top, left + side, top + side


def build_transparent_object_crop(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    bbox: tuple[int, int, int, int],
    padding_ratio: float,
    output_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a square BGRA array (encoded by OpenCV as an RGBA PNG)."""

    left, top, right, bottom = _square_crop_geometry(bbox, padding_ratio)
    side = right - left
    height, width = mask.shape
    source_x0, source_y0 = max(0, left), max(0, top)
    source_x1, source_y1 = min(width, right), min(height, bottom)
    if source_x0 >= source_x1 or source_y0 >= source_y1:
        raise ValueError("padded object crop does not intersect the MH image")

    canvas_bgr = np.zeros((side, side, 3), dtype=np.uint8)
    canvas_alpha = np.zeros((side, side), dtype=np.uint8)
    target_x0, target_y0 = source_x0 - left, source_y0 - top
    target_x1 = target_x0 + source_x1 - source_x0
    target_y1 = target_y0 + source_y1 - source_y0
    selected_mask = mask[source_y0:source_y1, source_x0:source_x1]
    selected_bgr = image_bgr[source_y0:source_y1, source_x0:source_x1]
    target_bgr = canvas_bgr[target_y0:target_y1, target_x0:target_x1]
    target_bgr[selected_mask] = selected_bgr[selected_mask]
    target_alpha = canvas_alpha[target_y0:target_y1, target_x0:target_x1]
    target_alpha[selected_mask] = 255

    interpolation = cv2.INTER_AREA if side > output_size else cv2.INTER_CUBIC
    resized_bgr = cv2.resize(
        canvas_bgr, (output_size, output_size), interpolation=interpolation
    )
    resized_alpha = cv2.resize(
        canvas_alpha, (output_size, output_size), interpolation=cv2.INTER_NEAREST
    )
    resized_bgr[resized_alpha == 0] = 0
    bgra = np.dstack((resized_bgr, resized_alpha))
    output_bbox = _bbox_xyxy(resized_alpha != 0)
    geometry = {
        "padding_ratio_each_side_of_longest_bbox_dimension": padding_ratio,
        "source_bbox_xyxy_exclusive": list(bbox),
        "source_square_xyxy_exclusive_unclipped": [left, top, right, bottom],
        "source_square_side_px": side,
        "source_intersection_xyxy_exclusive": [
            source_x0,
            source_y0,
            source_x1,
            source_y1,
        ],
        "output_size_wh": [output_size, output_size],
        "output_opaque_bbox_xyxy_exclusive": list(output_bbox),
        "rgb_resize_interpolation": (
            "opencv_inter_area" if side > output_size else "opencv_inter_cubic"
        ),
        "alpha_resize_interpolation": "opencv_inter_nearest",
        "transparent_rgb_value_bgr": [0, 0, 0],
    }
    return bgra, geometry


def _validate_label_frame(
    object_layer_manifest: dict[str, Any],
    label: str,
    frame_index: int,
    *,
    frame_count: int | None = None,
    allowed_labels: set[str] | None = None,
) -> dict[str, Any]:
    intervals = object_layer_manifest.get("intervals")
    if not isinstance(intervals, list):
        raise ValueError("object-layer manifest has no intervals list")
    matches: list[dict[str, Any]] = []
    for index, interval in enumerate(intervals):
        interval = _mapping(interval, f"object-layer intervals[{index}]")
        interval_label = interval.get("label")
        if not isinstance(interval_label, str) or not interval_label:
            raise ValueError(f"object-layer intervals[{index}].label must be a string")
        if allowed_labels is not None and interval_label not in allowed_labels:
            raise ValueError(
                f"object-layer interval {index} label {interval_label!r} is not in "
                "stereo label_vocabulary"
            )
        start = _json_int(
            interval.get("start"),
            f"object-layer intervals[{index}].start",
            minimum=0,
        )
        end = _json_int(
            interval.get("end"),
            f"object-layer intervals[{index}].end",
            minimum=0,
        )
        if end < start:
            raise ValueError(f"object-layer interval {index} has end before start")
        if frame_count is not None and end >= frame_count:
            raise ValueError(
                f"object-layer interval {index} ends outside frame range: {end}"
            )
        if interval_label == label and start <= frame_index <= end:
            matches.append({"label": interval_label, "start": start, "end": end})
    if len(matches) != 1:
        raise ValueError(
            f"MH frame {frame_index} must fall in exactly one {label!r} interval; "
            f"found {len(matches)}"
        )
    return matches[0]


def _write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write PNG: {path}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def prepare_pilot_inputs(
    *,
    episode_root: Path = DEFAULT_EPISODE_ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
    mh_frame_index: int = DEFAULT_MH_FRAME,
    object_label: str = DEFAULT_LABEL,
    crop_size: int = DEFAULT_CROP_SIZE,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    stereo_manifest_path: Path | None = None,
    object_layer_manifest_path: Path | None = None,
    modal_mask_path: Path | None = None,
    clean_mask_path: Path | None = None,
    amodal_mask_path: Path | None = None,
) -> dict[str, Any]:
    episode_root = episode_root.expanduser().resolve()
    if not episode_root.is_dir():
        raise ValueError(f"missing episode root: {episode_root}")
    output_candidate = output_dir.expanduser()
    mh_frame_index = _json_int(
        mh_frame_index, "MH frame index", minimum=0
    )
    crop_size = _json_int(crop_size, "crop size", minimum=1)
    if not isinstance(object_label, str) or not object_label:
        raise ValueError("object label must be a non-empty string")
    if (
        isinstance(padding_ratio, bool)
        or not isinstance(padding_ratio, (int, float))
        or not math.isfinite(float(padding_ratio))
        or padding_ratio < 0
    ):
        raise ValueError("padding ratio must be finite and non-negative")

    processed = episode_root / "camera_2" / "inpainting" / "processed" / "view" / "0"
    stereo_manifest_path = stereo_manifest_path or episode_root / "stereo_manifest.json"
    object_layer_manifest_path = (
        object_layer_manifest_path or processed / "object_layer" / "manifest.json"
    )
    modal_mask_path = (
        modal_mask_path or processed / "object_layer" / "object_mask_modal.npy"
    )
    completion = processed / "object_completion_dual_haco_e2fgvi"
    clean_mask_path = clean_mask_path or completion / "object_mask_observed_clean.npy"
    amodal_mask_path = amodal_mask_path or completion / "object_mask_amodal.npy"

    stereo_manifest_path = _required_file(stereo_manifest_path, "stereo manifest")
    object_layer_manifest_path = _required_file(
        object_layer_manifest_path, "object-layer manifest"
    )
    modal_mask_path = _required_file(modal_mask_path, "modal mask")
    clean_mask_path = _required_file(clean_mask_path, "clean mask")
    amodal_mask_path = _required_file(amodal_mask_path, "amodal mask")
    stereo = _load_json(stereo_manifest_path, "stereo manifest")
    object_layer = _load_json(object_layer_manifest_path, "object-layer manifest")

    temporal, offset, common_frames = _validate_mapping_provenance(
        stereo, object_label=object_label
    )
    if mh_frame_index >= common_frames:
        raise ValueError(
            f"MH frame {mh_frame_index} is outside common frame range "
            f"[0, {common_frames})"
        )
    sh_frame_index = mh_frame_index + offset
    if not 0 <= sh_frame_index < common_frames:
        raise ValueError(
            f"synchronized SH frame is outside common frame range: "
            f"{mh_frame_index} + ({offset}) = {sh_frame_index}, "
            f"expected [0, {common_frames})"
        )

    annotation_path, layer_annotation_path, annotation_segment = (
        _validate_annotation_provenance(
            stereo=stereo,
            stereo_manifest_path=stereo_manifest_path,
            object_layer=object_layer,
            object_layer_manifest_path=object_layer_manifest_path,
            object_label=object_label,
            frame_index=mh_frame_index,
            common_frames=common_frames,
        )
    )
    interval = _validate_label_frame(
        object_layer,
        object_label,
        mh_frame_index,
        frame_count=common_frames,
        allowed_labels=set(stereo["label_vocabulary"]),
    )
    _exact(
        interval["start"],
        annotation_segment["start"],
        "selected object-layer/annotation interval start",
    )
    _exact(
        interval["end"],
        annotation_segment["end"],
        "selected object-layer/annotation interval end",
    )

    calibration = _mapping(stereo.get("calibration"), "stereo calibration")
    calibration_reference, calibration_reference_json = (
        _validate_calibration_provenance(
            calibration=calibration,
            stereo_manifest_path=stereo_manifest_path,
        )
    )

    mh_image_path = _required_file(
        episode_root
        / "camera_2"
        / "rgb"
        / f"rgb_frame{mh_frame_index:06d}.jpg",
        "MH image",
    )
    sh_image_path = _required_file(
        episode_root
        / "camera_1"
        / "rgb"
        / f"rgb_frame{sh_frame_index:06d}.jpg",
        "synchronized SH image",
    )
    mh_image = cv2.imread(str(mh_image_path), cv2.IMREAD_COLOR)
    sh_image = cv2.imread(str(sh_image_path), cv2.IMREAD_COLOR)
    if mh_image is None or sh_image is None:
        raise ValueError("failed to decode MH or SH JPEG")
    if mh_image.shape != sh_image.shape:
        raise ValueError(
            f"MH/SH image shapes differ: {mh_image.shape} versus {sh_image.shape}"
        )
    layer_width = _json_int(
        object_layer.get("width"), "object-layer width", minimum=1
    )
    layer_height = _json_int(
        object_layer.get("height"), "object-layer height", minimum=1
    )
    _exact(layer_width, mh_image.shape[1], "object-layer width")
    _exact(layer_height, mh_image.shape[0], "object-layer height")
    _exact(
        calibration.get("image_size_wh"),
        [mh_image.shape[1], mh_image.shape[0]],
        "calibration/image dimensions",
    )

    masks = {
        "modal": _mask_frame(
            modal_mask_path,
            mh_frame_index,
            "modal",
            expected_frames=common_frames,
        ),
        "clean": _mask_frame(
            clean_mask_path,
            mh_frame_index,
            "clean",
            expected_frames=common_frames,
        ),
        "amodal": _mask_frame(
            amodal_mask_path,
            mh_frame_index,
            "amodal",
            expected_frames=common_frames,
        ),
    }
    mask_shapes = {name: mask.shape for name, mask in masks.items()}
    if len(set(mask_shapes.values())) != 1:
        raise ValueError(f"selected mask shapes differ: {mask_shapes}")
    if masks["modal"].shape != mh_image.shape[:2]:
        raise ValueError(
            f"mask shape {masks['modal'].shape} does not match MH image "
            f"shape {mh_image.shape[:2]}"
        )
    modal_clean_xor = int(np.count_nonzero(masks["modal"] ^ masks["clean"]))
    modal_amodal_xor = int(np.count_nonzero(masks["modal"] ^ masks["amodal"]))
    clean_amodal_xor = int(np.count_nonzero(masks["clean"] ^ masks["amodal"]))
    if modal_clean_xor or modal_amodal_xor or clean_amodal_xor:
        raise ValueError(
            "selected masks differ; this bundle forbids inferred/hidden support: "
            f"modal_clean_xor={modal_clean_xor}, "
            f"modal_amodal_xor={modal_amodal_xor}, "
            f"clean_amodal_xor={clean_amodal_xor}"
        )
    bbox = _bbox_xyxy(masks["modal"])

    crop_bgra, crop_geometry = build_transparent_object_crop(
        mh_image,
        masks["modal"],
        bbox=bbox,
        padding_ratio=padding_ratio,
        output_size=crop_size,
    )

    protected_paths = (
        stereo_manifest_path,
        object_layer_manifest_path,
        annotation_path,
        layer_annotation_path,
        calibration_reference,
        mh_image_path,
        sh_image_path,
        modal_mask_path,
        clean_mask_path,
        amodal_mask_path,
    )
    output_dir = validate_output_root(
        output_candidate,
        episode_root=episode_root,
        protected_paths=protected_paths,
    )
    final_output_paths = {
        "mh_image": output_dir / f"mh_frame{mh_frame_index:06d}.jpg",
        "sh_image": output_dir / f"sh_frame{sh_frame_index:06d}.jpg",
        "modal_mask": output_dir / f"mh_mask_modal_frame{mh_frame_index:06d}.png",
        "clean_mask": output_dir / f"mh_mask_clean_frame{mh_frame_index:06d}.png",
        "amodal_mask": output_dir / f"mh_mask_amodal_frame{mh_frame_index:06d}.png",
        "spar3d_rgba_crop": output_dir / f"spar3d_input_rgba_{crop_size}.png",
    }

    source_records = {
        "stereo_manifest": _file_record(
            stereo_manifest_path,
            provenance_role="camera_role_and_temporal_mapping",
            schema_version=EXPECTED_STEREO_SCHEMA,
        ),
        "object_layer_manifest": _file_record(
            object_layer_manifest_path,
            provenance_role="object_interval_and_mask_geometry",
            schema_version=EXPECTED_OBJECT_LAYER_SCHEMA,
        ),
        "stereo_ground_truth_annotation": _file_record(
            annotation_path,
            provenance_role="stereo_manifest_annotation",
            validated_contract="episode_num_frames_fps_contiguous_segments_v1",
        ),
        "object_layer_ground_truth_annotation": _file_record(
            layer_annotation_path,
            provenance_role="object_layer_manifest_annotation",
            validated_contract="episode_num_frames_fps_contiguous_segments_v1",
        ),
        "mh_image": _file_record(
            mh_image_path, view="MH", pipeline_camera="camera_2", frame_index=mh_frame_index
        ),
        "sh_image": _file_record(
            sh_image_path, view="SH", pipeline_camera="camera_1", frame_index=sh_frame_index
        ),
        "modal_mask": _file_record(
            modal_mask_path, selected_frame_index=mh_frame_index
        ),
        "clean_mask": _file_record(
            clean_mask_path, selected_frame_index=mh_frame_index
        ),
        "amodal_mask": _file_record(
            amodal_mask_path, selected_frame_index=mh_frame_index
        ),
        "calibration_reference": _file_record(
            calibration_reference,
            provenance_role="stereo_calibration_reference",
            schema_version=EXPECTED_CALIBRATION_SCHEMA,
        ),
    }
    staged_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.stage-", dir=str(output_dir.parent)
        )
    ).resolve()
    try:
        staged_output_paths = {
            name: staged_dir / final_path.name
            for name, final_path in final_output_paths.items()
        }
        shutil.copyfile(mh_image_path, staged_output_paths["mh_image"])
        shutil.copyfile(sh_image_path, staged_output_paths["sh_image"])
        for name in ("modal", "clean", "amodal"):
            _write_png(
                staged_output_paths[f"{name}_mask"],
                masks[name].astype(np.uint8) * 255,
            )
        _write_png(staged_output_paths["spar3d_rgba_crop"], crop_bgra)

        output_records = {
            name: _staged_output_record(staged_output_paths[name], final_path)
            for name, final_path in final_output_paths.items()
        }
        mh_byte_identical = (
            source_records["mh_image"]["sha256"]
            == output_records["mh_image"]["sha256"]
        )
        sh_byte_identical = (
            source_records["sh_image"]["sha256"]
            == output_records["sh_image"]["sha256"]
        )
        if not mh_byte_identical or not sh_byte_identical:
            raise RuntimeError(
                "copied full-frame images are not byte-identical to sources"
            )

        x0, y0, x1, y1 = bbox
        object_pixels = int(np.count_nonzero(masks["modal"]))
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "kind": "mesh_sota_pilot_input_bundle",
            "bundle": {
                "output_root": str(output_dir),
                "manifest_path": str(output_dir / "manifest.json"),
                "model_inference_performed": False,
                "publication": "complete_sibling_stage_then_directory_rename",
                "manifest_written_after_all_payload_outputs": True,
            },
            "selection": {
                "episode": stereo["episode"],
                "object_label": object_label,
                "frame_index_basis": "zero_based_decoded_source_frame",
                "mh_frame_index": mh_frame_index,
                "sh_frame_index": sh_frame_index,
                "mh_role": "primary/final",
                "sh_role": "auxiliary/evidence",
                "object_interval": {
                    "start": interval["start"],
                    "end": interval["end"],
                },
            },
            "image_geometry": {
                "width": int(mh_image.shape[1]),
                "height": int(mh_image.shape[0]),
                "channels": 3,
                "mh_sh_shapes_equal": True,
            },
            "stereo_alignment": {
                "reference_view": temporal["reference_view"],
                "camera1_frame_offset": offset,
                "lookup_convention": (
                    "camera_1/SH source index = camera_2/MH reference index + offset"
                ),
                "source_manifest_lookup": temporal["camera1_lookup"],
                "out_of_range_policy": temporal["out_of_range_policy"],
            },
            "camera_namespace": {
                "primary_view": stereo["primary_view"],
                "auxiliary_view": stereo["auxiliary_view"],
                "training_view": stereo["training_view"],
                "robot_overlay_view": stereo["robot_overlay_view"],
                "stereo_code_mapping": stereo["stereo_code_mapping"],
                "calibration_camera_mapping": calibration[
                    "calibration_camera_mapping"
                ],
                "pipeline_camera_mapping": calibration["pipeline_camera_mapping"],
                "pipeline_to_calibration_camera": calibration[
                    "pipeline_to_calibration_camera"
                ],
            },
            "calibration": {
                "status": calibration["status"],
                "reference": source_records["calibration_reference"],
                "checkerboard": calibration["checkerboard"],
                "intrinsics_by_view": calibration["intrinsics_by_view"],
                "relative_extrinsics": calibration["relative_extrinsics"],
            },
            "provenance_validation": {
                "records_contain_canonical_path_bytes_sha256": True,
                "stereo_manifest_schema_version": EXPECTED_STEREO_SCHEMA,
                "object_layer_manifest_schema_version": EXPECTED_OBJECT_LAYER_SCHEMA,
                "calibration_reference_schema_version": calibration_reference_json[
                    "schema_version"
                ],
                "annotation_files_byte_identical": True,
                "annotation_values_equal": True,
                "annotation_segment_matches_object_layer_interval": True,
                "camera_role_mapping_exact": True,
                "temporal_mapping_exact": True,
                "calibration_values_match_hashed_reference": True,
            },
            "mask_validation": {
                "modal_equals_clean_equals_amodal": True,
                "modal_clean_xor_pixels": modal_clean_xor,
                "modal_amodal_xor_pixels": modal_amodal_xor,
                "clean_amodal_xor_pixels": clean_amodal_xor,
                "foreground_pixels_each": object_pixels,
                "bbox_convention": "xyxy_exclusive",
                "bbox_xyxy_exclusive": [x0, y0, x1, y1],
                "bbox_xywh": [x0, y0, x1 - x0, y1 - y0],
            },
            "crop": {
                "purpose": "SPAR3D transparent object input",
                "format": "PNG RGBA with binary alpha",
                **crop_geometry,
            },
            "pixel_provenance": {
                "rgb_source": "decoded dataset MH JPEG at the selected frame",
                "alpha_source": "selected MH modal mask",
                "learned_models_run_by_this_builder": [],
                "operations": [
                    "modal masking",
                    "transparent square padding",
                    "resize",
                ],
                "modal_rgb_pixels_used": object_pixels,
                "hidden_amodal_pixels_used": 0,
                "inferred_pixels_used": 0,
                "inpainted_pixels_used": 0,
                "generated_pixels_used": 0,
                "completed_video_pixels_used": 0,
                "statement": (
                    "No inferred or inpainted RGB pixels are present. All opaque "
                    "RGB support originates from the selected decoded MH frame, "
                    "and the modal, clean, and amodal masks were exactly equal "
                    "before export."
                ),
            },
            "sources": source_records,
            "outputs": output_records,
            "invariants": {
                "synchronized_sh_index_derived_from_stereo_manifest": True,
                "modal_equals_clean_equals_amodal": True,
                "mask_nonempty": True,
                "output_masks_binary_0_255": True,
                "rgba_alpha_binary_0_255": True,
                "rgba_transparent_pixels_have_zero_rgb": True,
                "mh_full_image_byte_identical_to_source": mh_byte_identical,
                "sh_full_image_byte_identical_to_source": sh_byte_identical,
                "uses_inferred_or_inpainted_pixels": False,
                "model_inference_performed": False,
                "whole_bundle_replaces_prior_bundle_without_stale_payloads": True,
            },
        }
        # The manifest is deliberately the final file created in staging.  Only
        # then is the complete directory made visible at the final output path.
        _write_json_atomic(staged_dir / "manifest.json", manifest)
        publish_bundle_directory(staged_dir, output_dir)
        return manifest
    finally:
        if staged_dir.exists():
            shutil.rmtree(staged_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, default=DEFAULT_EPISODE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mh-frame", type=int, default=DEFAULT_MH_FRAME)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=DEFAULT_PADDING_RATIO,
        help="Padding added on each side as a fraction of the longer object bbox side.",
    )
    parser.add_argument("--stereo-manifest", type=Path)
    parser.add_argument("--object-layer-manifest", type=Path)
    parser.add_argument("--modal-mask", type=Path)
    parser.add_argument("--clean-mask", type=Path)
    parser.add_argument("--amodal-mask", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = prepare_pilot_inputs(
        episode_root=args.episode_root,
        output_dir=args.output,
        mh_frame_index=args.mh_frame,
        object_label=args.label,
        crop_size=args.crop_size,
        padding_ratio=args.padding_ratio,
        stereo_manifest_path=args.stereo_manifest,
        object_layer_manifest_path=args.object_layer_manifest,
        modal_mask_path=args.modal_mask,
        clean_mask_path=args.clean_mask,
        amodal_mask_path=args.amodal_mask,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output_root": manifest["bundle"]["output_root"],
                "mh_frame_index": manifest["selection"]["mh_frame_index"],
                "sh_frame_index": manifest["selection"]["sh_frame_index"],
                "object_label": manifest["selection"]["object_label"],
                "inferred_or_inpainted_pixels": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
