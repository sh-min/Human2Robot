"""Compare HaCo object inpainting with approximate and calibrated focal lengths.

The command binds both HaCo completion reports to their intended HaWoR focal
inputs, requires byte-exact decoded approximate/calibrated source RGB, computes
mask and decoded-RGB A/B metrics without retaining the full videos in memory,
and atomically publishes a labelled 3x2 H.264 grid plus ``report.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from atomic_directory_publish import publish_directory
from make_video_comparison_grid import (
    GridLayout,
    NamedVideo,
    VideoMetadata,
    probe_video,
    render_comparison_grid_layout,
    validate_grid_input_metadata,
)


HACO_COMPLETION_METHOD = (
    "dual_haco_selected_hand_cleaned_object_constrained_e2fgvi"
)
GRID = GridLayout(columns=3, rows=2)
VIDEO_NAME = "video_compare_calibration_inpainting_ab_3x2.mp4"
REPORT_NAME = "report.json"
COMPLETION_FILENAMES = {
    "hand_removed": "video_hand_removed_modal_only.mp4",
    "object_completed": "video_object_completed.mp4",
    "clean_mask": "object_mask_observed_clean.npy",
    "amodal_mask": "object_mask_amodal.npy",
    "report": "report.json",
}
PANEL_SPECS = (
    ("approx_original", "Approx: original"),
    ("approx_hand_removed", "Approx: hand removed"),
    ("approx_object_completed", "Approx: object completed"),
    ("calibrated_original", "Calibrated: original"),
    ("calibrated_hand_removed", "Calibrated: hand removed"),
    ("calibrated_object_completed", "Calibrated: object completed"),
)


def _require_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(resolved)
    return resolved


def _load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(_require_file(path).read_text())
    if not isinstance(value, dict):
        raise TypeError(f"JSON report must be an object: {path}")
    return value


def validate_haco_completion_report(
    report: Mapping[str, object],
    *,
    name: str,
) -> None:
    """Reject non-HaCo or internally inconsistent completion reports."""

    if report.get("method") != HACO_COMPLETION_METHOD:
        raise ValueError(
            f"{name} method {report.get('method')!r} is not dual-view HaCo "
            "object completion"
        )
    if report.get("generated_texture") is not True:
        raise ValueError(f"{name} does not declare generated texture")
    if report.get("physical_geometry_guarantee") is not False:
        raise ValueError(f"{name} overstates physical geometry provenance")

    invariants = report.get("invariants")
    if not isinstance(invariants, dict):
        raise TypeError(f"{name} invariants must be a dictionary")
    required_true = (
        "trusted_modal_subset_input_modal",
        "trusted_modal_subset_amodal",
        "hand_contested_disjoint_trusted_modal",
        "hidden_disjoint_trusted_modal",
        "trusted_modal_rgb_has_priority",
        "hand_contested_input_modal_is_not_rgb_protected",
        "trajectory_arrays_unchanged",
        "haco_selected_hidden_subset_raw_hidden",
        "haco_does_not_measure_object_rgb_or_depth",
        "primary_view_owns_haco_projection",
        "auxiliary_haco_is_confidence_only",
    )
    for invariant in required_true:
        if invariants.get(invariant) is not True:
            raise ValueError(f"{name} invariant failed: {invariant}")
    if invariants.get("auxiliary_geometry_used") is not False:
        raise ValueError(f"{name} unexpectedly used auxiliary image geometry")
    for invariant in (
        "preencode_trusted_modal_rgb_values_changed",
        "preencode_values_changed_outside_hidden",
    ):
        if int(invariants.get(invariant, -1)) != 0:
            raise ValueError(f"{name} invariant failed: {invariant}")

    counts = report.get("counts")
    if not isinstance(counts, dict):
        raise TypeError(f"{name} counts must be a dictionary")
    if int(counts.get("hidden_pixels_without_completed_depth", -1)) != 0:
        raise ValueError(f"{name} has hidden support without completed depth")

    outputs = report.get("outputs")
    if not isinstance(outputs, dict):
        raise TypeError(f"{name} outputs must be a dictionary")
    expected_outputs = {
        "baseline_video": COMPLETION_FILENAMES["hand_removed"],
        "completed_video": COMPLETION_FILENAMES["object_completed"],
        "clean_modal_mask": COMPLETION_FILENAMES["clean_mask"],
        "amodal_mask": COMPLETION_FILENAMES["amodal_mask"],
    }
    for key, expected in expected_outputs.items():
        if outputs.get(key) != expected:
            raise ValueError(
                f"{name} output {key!r} must be {expected!r}, got "
                f"{outputs.get(key)!r}"
            )


def _completion_report_focal_px(
    report: Mapping[str, object],
    *,
    name: str,
) -> float:
    config = report.get("config")
    if not isinstance(config, dict):
        raise TypeError(f"{name} config must be a dictionary")
    try:
        focal_px = float(config["primary_hawor_focal_px"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} report lacks a valid primary_hawor_focal_px"
        ) from exc
    if not np.isfinite(focal_px) or focal_px <= 0:
        raise ValueError(
            f"{name} primary_hawor_focal_px must be finite and positive"
        )
    return focal_px


def _hawor_focal_px(path: Path, *, name: str) -> tuple[Path, float]:
    resolved = _require_file(path)
    with np.load(resolved, allow_pickle=False) as hawor:
        if "img_focal" not in hawor.files:
            raise KeyError(f"{name} HaWoR input lacks img_focal: {resolved}")
        raw_focal = np.asarray(hawor["img_focal"])
        if raw_focal.size != 1 or raw_focal.dtype.kind not in "iuf":
            raise TypeError(f"{name} HaWoR img_focal must be one numeric scalar")
        focal_px = float(raw_focal.item())
    if not np.isfinite(focal_px) or focal_px <= 0:
        raise ValueError(f"{name} HaWoR img_focal must be finite and positive")
    return resolved, focal_px


def validate_calibration_provenance(
    approx_report: Mapping[str, object],
    calibrated_report: Mapping[str, object],
    *,
    expected_approx_hawor_npz: Path,
    expected_calibrated_hawor_npz: Path,
    expected_approx_source: Path,
    expected_calibrated_source: Path,
) -> dict[str, object]:
    """Bind each completion report to the intended calibration branch."""

    expected_approx_path, expected_approx_focal = _hawor_focal_px(
        expected_approx_hawor_npz,
        name="approx",
    )
    expected_calibrated_path, expected_calibrated_focal = _hawor_focal_px(
        expected_calibrated_hawor_npz,
        name="calibrated",
    )
    if expected_approx_path == expected_calibrated_path:
        raise ValueError("approx and calibrated HaWoR inputs resolve to one file")
    if np.isclose(
        expected_approx_focal,
        expected_calibrated_focal,
        rtol=1e-9,
        atol=1e-6,
    ):
        raise ValueError("approx and calibrated expected focals are identical")
    if expected_approx_focal >= expected_calibrated_focal:
        raise ValueError(
            "expected calibration identity requires approx focal < calibrated focal"
        )

    records: dict[str, dict[str, object]] = {}
    for name, report, expected_path, expected_focal, expected_source in (
        (
            "approx",
            approx_report,
            expected_approx_path,
            expected_approx_focal,
            expected_approx_source,
        ),
        (
            "calibrated",
            calibrated_report,
            expected_calibrated_path,
            expected_calibrated_focal,
            expected_calibrated_source,
        ),
    ):
        sources = report.get("sources")
        if not isinstance(sources, dict):
            raise TypeError(f"{name} sources must be a dictionary")
        source_value = sources.get("hawor_npz")
        if not isinstance(source_value, str) or not source_value:
            raise ValueError(f"{name} report lacks sources.hawor_npz")
        reported_path = Path(source_value).expanduser().resolve()
        if reported_path != expected_path:
            raise ValueError(
                f"{name} completion belongs to a different HaWoR branch: "
                f"{reported_path} != {expected_path}"
            )
        expected_source = _require_file(expected_source)
        reported_source_value = sources.get("source")
        if not isinstance(reported_source_value, str) or not reported_source_value:
            raise ValueError(f"{name} report lacks sources.source")
        reported_source = Path(reported_source_value).expanduser().resolve()
        if reported_source != expected_source:
            raise ValueError(
                f"{name} completion belongs to a different source video: "
                f"{reported_source} != {expected_source}"
            )
        reported_focal = _completion_report_focal_px(report, name=name)
        if not np.isclose(
            reported_focal,
            expected_focal,
            rtol=1e-9,
            atol=1e-6,
        ):
            raise ValueError(
                f"{name} report focal {reported_focal} != expected "
                f"{expected_focal}"
            )
        records[name] = {
            "reported_hawor_npz": str(reported_path),
            "expected_hawor_npz": str(expected_path),
            "reported_source": str(reported_source),
            "expected_source": str(expected_source),
            "reported_focal_px": reported_focal,
            "expected_focal_px": expected_focal,
        }

    if np.isclose(
        float(records["approx"]["reported_focal_px"]),
        float(records["calibrated"]["reported_focal_px"]),
        rtol=1e-9,
        atol=1e-6,
    ):
        raise ValueError("approx and calibrated reports use the same focal")
    approx_focal = float(records["approx"]["reported_focal_px"])
    calibrated_focal = float(records["calibrated"]["reported_focal_px"])
    if approx_focal >= calibrated_focal:
        raise ValueError(
            "completion identity requires approx focal < calibrated focal"
        )
    return {
        "branches": records,
        "focal_delta_px": calibrated_focal - approx_focal,
        "calibrated_over_approx_focal_ratio": calibrated_focal / approx_focal,
        "expected_order": "approx < calibrated",
    }


def _validate_report_metadata(
    report: Mapping[str, object],
    reference: VideoMetadata,
    *,
    name: str,
) -> None:
    metadata = report.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError(f"{name} metadata must be a dictionary")
    expected_ints = {
        "frames": reference.frame_count,
        "width": reference.width,
        "height": reference.height,
    }
    for field, expected in expected_ints.items():
        try:
            actual = int(metadata.get(field, -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} metadata has invalid {field}") from exc
        if actual != expected:
            raise ValueError(
                f"{name} report {field} {actual} != source {expected}"
            )
    try:
        report_fps = float(metadata.get("fps", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} metadata has invalid fps") from exc
    if not np.isclose(report_fps, float(reference.fps), rtol=0.0, atol=1e-9):
        raise ValueError(
            f"{name} report fps {report_fps} != source {float(reference.fps)}"
        )


def _load_bool_mask(
    path: Path,
    *,
    expected_shape: tuple[int, int, int],
    name: str,
) -> np.ndarray:
    mask = np.load(_require_file(path), mmap_mode="r", allow_pickle=False)
    if mask.shape != expected_shape or mask.dtype != np.bool_:
        raise ValueError(f"{name} must be bool {expected_shape}, got {mask.dtype} {mask.shape}")
    return mask


def stream_mask_comparison(
    approx: np.ndarray,
    calibrated: np.ndarray,
) -> dict[str, object]:
    """Compute binary-mask A/B overlap while materializing one frame at a time."""

    if approx.shape != calibrated.shape or approx.ndim != 3:
        raise ValueError("mask arrays must share (T,H,W)")
    if approx.dtype != np.bool_ or calibrated.dtype != np.bool_:
        raise TypeError("mask arrays must be boolean")

    approx_pixels = 0
    calibrated_pixels = 0
    intersection_pixels = 0
    union_pixels = 0
    symmetric_difference_pixels = 0
    for frame_index in range(approx.shape[0]):
        approx_frame = np.asarray(approx[frame_index], dtype=bool)
        calibrated_frame = np.asarray(calibrated[frame_index], dtype=bool)
        approx_pixels += int(approx_frame.sum())
        calibrated_pixels += int(calibrated_frame.sum())
        intersection_pixels += int(np.count_nonzero(approx_frame & calibrated_frame))
        union_pixels += int(np.count_nonzero(approx_frame | calibrated_frame))
        symmetric_difference_pixels += int(
            np.count_nonzero(approx_frame ^ calibrated_frame)
        )

    canvas_pixels = int(np.prod(approx.shape, dtype=np.int64))
    iou = 1.0 if union_pixels == 0 else intersection_pixels / union_pixels
    return {
        "frames": int(approx.shape[0]),
        "canvas_pixels": canvas_pixels,
        "approx_pixels": approx_pixels,
        "calibrated_pixels": calibrated_pixels,
        "intersection_pixels": intersection_pixels,
        "union_pixels": union_pixels,
        "iou": iou,
        "symmetric_difference_pixels": symmetric_difference_pixels,
        "changed_pixel_ratio": (
            symmetric_difference_pixels / max(canvas_pixels, 1)
        ),
    }


def validate_clean_subset_amodal(
    clean: np.ndarray,
    amodal: np.ndarray,
    *,
    name: str,
) -> None:
    if clean.shape != amodal.shape:
        raise ValueError(f"{name} clean and amodal mask shapes differ")
    for frame_index in range(clean.shape[0]):
        outside = np.asarray(clean[frame_index], dtype=bool) & ~np.asarray(
            amodal[frame_index], dtype=bool
        )
        if np.any(outside):
            raise ValueError(
                f"{name} clean mask is outside amodal support at frame "
                f"{frame_index}"
            )


def _empty_rgb_accumulator() -> dict[str, int]:
    return {
        "pixels": 0,
        "channels": 0,
        "absolute_error_sum": 0,
        "changed_pixels": 0,
        "changed_channels": 0,
        "max_absolute_error": 0,
    }


def _update_rgb_accumulator(
    accumulator: dict[str, int],
    difference: np.ndarray,
    support: np.ndarray | None = None,
) -> None:
    if support is None:
        selected = difference.reshape(-1, 3)
    else:
        if support.shape != difference.shape[:2]:
            raise ValueError("RGB metric support shape differs from video frame")
        selected = difference[support]
    pixel_count = int(selected.shape[0])
    if pixel_count == 0:
        return
    accumulator["pixels"] += pixel_count
    accumulator["channels"] += pixel_count * 3
    accumulator["absolute_error_sum"] += int(
        selected.sum(dtype=np.uint64)
    )
    accumulator["changed_pixels"] += int(
        np.count_nonzero(np.any(selected != 0, axis=1))
    )
    accumulator["changed_channels"] += int(np.count_nonzero(selected))
    accumulator["max_absolute_error"] = max(
        accumulator["max_absolute_error"], int(selected.max())
    )


def _finalize_rgb_accumulator(value: Mapping[str, int]) -> dict[str, object]:
    pixels = int(value["pixels"])
    channels = int(value["channels"])
    return {
        **{key: int(item) for key, item in value.items()},
        "mae_rgb_u8": (
            float(value["absolute_error_sum"]) / channels
            if channels
            else None
        ),
        "changed_pixel_ratio": (
            float(value["changed_pixels"]) / pixels if pixels else None
        ),
        "changed_channel_ratio": (
            float(value["changed_channels"]) / channels
            if channels
            else None
        ),
    }


def stream_completion_rgb_metrics(
    approx_video: Path,
    calibrated_video: Path,
    *,
    metadata: VideoMetadata,
    approx_amodal: np.ndarray,
    calibrated_amodal: np.ndarray,
) -> dict[str, object]:
    """Compare decoded completion frames using bounded, frame-sized memory."""

    approx_capture = cv2.VideoCapture(str(approx_video))
    calibrated_capture = cv2.VideoCapture(str(calibrated_video))
    if not approx_capture.isOpened() or not calibrated_capture.isOpened():
        approx_capture.release()
        calibrated_capture.release()
        raise RuntimeError("could not open both completed videos for RGB comparison")

    full = _empty_rgb_accumulator()
    support = _empty_rgb_accumulator()
    try:
        for frame_index in range(metadata.frame_count):
            approx_ok, approx_frame = approx_capture.read()
            calibrated_ok, calibrated_frame = calibrated_capture.read()
            if not approx_ok or not calibrated_ok:
                raise RuntimeError(
                    "completed video decode ended before expected frame "
                    f"{frame_index}"
                )
            expected_frame_shape = (metadata.height, metadata.width, 3)
            if (
                approx_frame.shape != expected_frame_shape
                or calibrated_frame.shape != expected_frame_shape
            ):
                raise ValueError(
                    f"decoded RGB frames must be {expected_frame_shape}"
                )
            difference = np.abs(
                approx_frame.astype(np.int16)
                - calibrated_frame.astype(np.int16)
            ).astype(np.uint8)
            _update_rgb_accumulator(full, difference)
            amodal_union = np.asarray(
                approx_amodal[frame_index], dtype=bool
            ) | np.asarray(calibrated_amodal[frame_index], dtype=bool)
            _update_rgb_accumulator(support, difference, amodal_union)

        approx_extra, _ = approx_capture.read()
        calibrated_extra, _ = calibrated_capture.read()
        if approx_extra or calibrated_extra:
            raise RuntimeError("completed video decode has more frames than metadata")
    finally:
        approx_capture.release()
        calibrated_capture.release()

    return {
        "compared_frames": metadata.frame_count,
        "comparison_space": "decoded BGR uint8 (channel order does not affect absolute error)",
        "full_frame": _finalize_rgb_accumulator(full),
        "amodal_union": _finalize_rgb_accumulator(support),
    }


def stream_exact_original_rgb_identity(
    approx_video: Path,
    calibrated_video: Path,
    *,
    metadata: VideoMetadata,
) -> dict[str, object]:
    """Require byte-exact decoded frames for a controlled calibration A/B."""

    approx_capture = cv2.VideoCapture(str(approx_video))
    calibrated_capture = cv2.VideoCapture(str(calibrated_video))
    if not approx_capture.isOpened() or not calibrated_capture.isOpened():
        approx_capture.release()
        calibrated_capture.release()
        raise RuntimeError("could not open both original videos for RGB validation")

    approx_digest = hashlib.sha256()
    calibrated_digest = hashlib.sha256()
    expected_frame_shape = (metadata.height, metadata.width, 3)
    try:
        for frame_index in range(metadata.frame_count):
            approx_ok, approx_frame = approx_capture.read()
            calibrated_ok, calibrated_frame = calibrated_capture.read()
            if not approx_ok or not calibrated_ok:
                raise RuntimeError(
                    "original video decode ended before expected frame "
                    f"{frame_index}"
                )
            if (
                approx_frame.shape != expected_frame_shape
                or calibrated_frame.shape != expected_frame_shape
            ):
                raise ValueError(
                    f"decoded original frames must be {expected_frame_shape}"
                )
            if not np.array_equal(approx_frame, calibrated_frame):
                changed_pixels = int(
                    np.count_nonzero(
                        np.any(approx_frame != calibrated_frame, axis=2)
                    )
                )
                raise ValueError(
                    "approx and calibrated original RGB differ at frame "
                    f"{frame_index} ({changed_pixels} pixels)"
                )
            approx_digest.update(approx_frame.tobytes(order="C"))
            calibrated_digest.update(calibrated_frame.tobytes(order="C"))

        approx_extra, _ = approx_capture.read()
        calibrated_extra, _ = calibrated_capture.read()
        if approx_extra or calibrated_extra:
            raise RuntimeError("original video decode has more frames than metadata")
    finally:
        approx_capture.release()
        calibrated_capture.release()

    approx_sha256 = approx_digest.hexdigest()
    calibrated_sha256 = calibrated_digest.hexdigest()
    if approx_sha256 != calibrated_sha256:
        raise RuntimeError("equal decoded originals produced unequal digests")
    return {
        "exact_equal": True,
        "compared_frames": metadata.frame_count,
        "comparison_space": "decoded BGR uint8",
        "decoded_rgb_sha256": approx_sha256,
    }


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


def validate_source_video_metadata(
    metadata: Mapping[str, VideoMetadata],
) -> VideoMetadata:
    """Require exact frame/FPS synchronization and native geometry equality."""

    if "original" not in metadata and "calibrated_original" not in metadata:
        raise ValueError("original video metadata is missing")
    values = list(metadata.values())
    reference = validate_grid_input_metadata(values, len(values))
    geometry_errors = [
        f"{name}: {value.width}x{value.height} != "
        f"{reference.width}x{reference.height}"
        for name, value in metadata.items()
        if (value.width, value.height) != (reference.width, reference.height)
    ]
    if geometry_errors:
        raise ValueError(
            "comparison video geometry differs:\n  "
            + "\n  ".join(geometry_errors)
        )
    return reference


def _validate_grid_output(
    output: VideoMetadata,
    reference: VideoMetadata,
) -> None:
    errors: list[str] = []
    if (output.width, output.height) != (GRID.output_width, GRID.output_height):
        errors.append(
            f"geometry {output.width}x{output.height} != "
            f"{GRID.output_width}x{GRID.output_height}"
        )
    if output.frame_count != reference.frame_count:
        errors.append(
            f"frames {output.frame_count} != {reference.frame_count}"
        )
    if output.fps != reference.fps:
        errors.append(f"fps {output.fps} != {reference.fps}")
    if output.codec_name != "h264" or output.pixel_format != "yuv420p":
        errors.append(
            f"codec/pixel format {output.codec_name}/{output.pixel_format} "
            "!= h264/yuv420p"
        )
    if errors:
        raise RuntimeError("rendered A/B grid failed validation: " + "; ".join(errors))


def _completion_paths(directory: Path) -> dict[str, Path]:
    root = directory.expanduser().resolve()
    return {
        key: _require_file(root / filename)
        for key, filename in COMPLETION_FILENAMES.items()
    }


def run_comparison(
    approx_original: Path,
    calibrated_original: Path,
    approx_completion_dir: Path,
    calibrated_completion_dir: Path,
    out_dir: Path,
    *,
    expected_approx_hawor_npz: Path,
    expected_calibrated_hawor_npz: Path,
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Validate, compare, and atomically publish one calibration A/B result."""

    approx_original = _require_file(approx_original)
    calibrated_original = _require_file(calibrated_original)
    approx_completion_root = approx_completion_dir.expanduser().resolve()
    calibrated_completion_root = calibrated_completion_dir.expanduser().resolve()
    if approx_completion_root == calibrated_completion_root:
        raise ValueError(
            "approx and calibrated completion directories resolve to one path"
        )
    approx_paths = _completion_paths(approx_completion_dir)
    calibrated_paths = _completion_paths(calibrated_completion_dir)
    approx_report = _load_json_object(approx_paths["report"])
    calibrated_report = _load_json_object(calibrated_paths["report"])
    validate_haco_completion_report(approx_report, name="approx")
    validate_haco_completion_report(calibrated_report, name="calibrated")
    calibration_provenance = validate_calibration_provenance(
        approx_report,
        calibrated_report,
        expected_approx_hawor_npz=expected_approx_hawor_npz,
        expected_calibrated_hawor_npz=expected_calibrated_hawor_npz,
        expected_approx_source=approx_original,
        expected_calibrated_source=calibrated_original,
    )

    video_paths = {
        "calibrated_original": calibrated_original,
        "approx_original": approx_original,
        "approx_hand_removed": approx_paths["hand_removed"],
        "approx_object_completed": approx_paths["object_completed"],
        "calibrated_hand_removed": calibrated_paths["hand_removed"],
        "calibrated_object_completed": calibrated_paths["object_completed"],
    }
    source_metadata = {
        name: probe_video(path) for name, path in video_paths.items()
    }
    reference = validate_source_video_metadata(source_metadata)
    _validate_report_metadata(approx_report, reference, name="approx")
    _validate_report_metadata(
        calibrated_report, reference, name="calibrated"
    )
    original_rgb_identity = stream_exact_original_rgb_identity(
        approx_original,
        calibrated_original,
        metadata=reference,
    )

    expected_shape = (
        reference.frame_count,
        reference.height,
        reference.width,
    )
    approx_clean = _load_bool_mask(
        approx_paths["clean_mask"],
        expected_shape=expected_shape,
        name="approx clean mask",
    )
    approx_amodal = _load_bool_mask(
        approx_paths["amodal_mask"],
        expected_shape=expected_shape,
        name="approx amodal mask",
    )
    calibrated_clean = _load_bool_mask(
        calibrated_paths["clean_mask"],
        expected_shape=expected_shape,
        name="calibrated clean mask",
    )
    calibrated_amodal = _load_bool_mask(
        calibrated_paths["amodal_mask"],
        expected_shape=expected_shape,
        name="calibrated amodal mask",
    )
    validate_clean_subset_amodal(
        approx_clean, approx_amodal, name="approx"
    )
    validate_clean_subset_amodal(
        calibrated_clean, calibrated_amodal, name="calibrated"
    )
    mask_metrics = {
        "clean_modal": stream_mask_comparison(
            approx_clean, calibrated_clean
        ),
        "amodal": stream_mask_comparison(approx_amodal, calibrated_amodal),
    }
    rgb_metrics = stream_completion_rgb_metrics(
        approx_paths["object_completed"],
        calibrated_paths["object_completed"],
        metadata=reference,
        approx_amodal=approx_amodal,
        calibrated_amodal=calibrated_amodal,
    )

    panel_paths = {
        "approx_original": approx_original,
        "approx_hand_removed": approx_paths["hand_removed"],
        "approx_object_completed": approx_paths["object_completed"],
        "calibrated_original": calibrated_original,
        "calibrated_hand_removed": calibrated_paths["hand_removed"],
        "calibrated_object_completed": calibrated_paths["object_completed"],
    }
    videos = [
        NamedVideo(label=label, path=panel_paths[name])
        for name, label in PANEL_SPECS
    ]

    output_dir = out_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".calibration_inpainting_ab.",
            dir=output_dir.parent,
        )
    )
    try:
        grid_metadata = render_comparison_grid_layout(
            videos,
            staging / VIDEO_NAME,
            layout=GRID,
            overwrite=False,
            crf=crf,
            preset=preset,
        )
        _validate_grid_output(grid_metadata, reference)
        report = {
            "schema_version": 2,
            "comparison": (
                "approximate-focal versus calibrated-focal dual-HaCo "
                "object inpainting"
            ),
            "layout": [
                ["approx_original", "approx_hand_removed", "approx_object_completed"],
                [
                    "calibrated_original",
                    "calibrated_hand_removed",
                    "calibrated_object_completed",
                ],
            ],
            "definitions": {
                name: label for name, label in PANEL_SPECS
            },
            "output_video": VIDEO_NAME,
            "metadata": {
                "reference_original": _metadata_dict(reference),
                "inputs": {
                    name: _metadata_dict(value)
                    for name, value in source_metadata.items()
                },
                "grid": _metadata_dict(grid_metadata),
            },
            "sources": {
                "approx_original": str(approx_original),
                "calibrated_original": str(calibrated_original),
                "approx_completion_dir": str(
                    approx_paths["report"].parent
                ),
                "calibrated_completion_dir": str(
                    calibrated_paths["report"].parent
                ),
            },
            "source_reports": {
                "approx": str(approx_paths["report"]),
                "calibrated": str(calibrated_paths["report"]),
            },
            "source_masks": {
                "approx_clean": str(approx_paths["clean_mask"]),
                "approx_amodal": str(approx_paths["amodal_mask"]),
                "calibrated_clean": str(calibrated_paths["clean_mask"]),
                "calibrated_amodal": str(calibrated_paths["amodal_mask"]),
            },
            "validated_methods": {
                "approx": approx_report["method"],
                "calibrated": calibrated_report["method"],
            },
            "validation": {
                "exact_video_frame_count_and_fps": True,
                "native_video_geometry_equal": True,
                "original_decoded_rgb_exact_equal": True,
                "reports_match_source_metadata": True,
                "completion_branches_and_focals_match_expected": True,
                "clean_masks_subset_amodal_masks": True,
                "output_h264_yuv420p": True,
            },
            "metrics": {
                "original_rgb_identity": original_rgb_identity,
                "masks": mask_metrics,
                "object_completed_rgb_ab": rgb_metrics,
            },
            "metric_definitions": {
                "mask_iou": "intersection / union; 1.0 when both masks are empty",
                "mask_changed_pixel_ratio": (
                    "mask symmetric-difference pixels / all T*H*W pixels"
                ),
                "mae_rgb_u8": (
                    "mean absolute decoded 8-bit channel difference; full "
                    "videos are streamed one frame at a time"
                ),
                "rgb_changed_pixel_ratio": (
                    "pixels with at least one unequal decoded channel / "
                    "compared pixels"
                ),
                "amodal_union": (
                    "per-frame union of approximate and calibrated amodal masks"
                ),
            },
            "completion_counts": {
                "approx": approx_report["counts"],
                "calibrated": calibrated_report["counts"],
            },
            "completion_invariants": {
                "approx": approx_report["invariants"],
                "calibrated": calibrated_report["invariants"],
            },
            "calibration_provenance": calibration_provenance,
            "provenance_warning": (
                "Both variants use inferred amodal support and generated "
                "E2FGVI texture. HaCo selects contact-connected support but "
                "does not measure hidden RGB, object depth, or physical geometry."
            ),
        }
        (staging / REPORT_NAME).write_text(
            json.dumps(report, indent=2) + "\n"
        )
        publish_directory(str(staging), str(output_dir))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approx_original", type=Path, required=True)
    parser.add_argument("--calibrated_original", type=Path, required=True)
    parser.add_argument("--approx_completion_dir", type=Path, required=True)
    parser.add_argument("--calibrated_completion_dir", type=Path, required=True)
    parser.add_argument("--expected_approx_hawor_npz", type=Path, required=True)
    parser.add_argument(
        "--expected_calibrated_hawor_npz",
        type=Path,
        required=True,
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be between 0 and 51")
    output = run_comparison(
        args.approx_original,
        args.calibrated_original,
        args.approx_completion_dir,
        args.calibrated_completion_dir,
        args.out_dir,
        expected_approx_hawor_npz=args.expected_approx_hawor_npz,
        expected_calibrated_hawor_npz=args.expected_calibrated_hawor_npz,
        crf=args.crf,
        preset=args.preset,
    )
    print(f"[ok] calibration inpainting A/B comparison: {output}", flush=True)


if __name__ == "__main__":
    main()
