"""Compare baseline and interior-expanded contact-occlusion outputs.

The utility is deliberately independent of the compositors.  It consumes two
completed output directories, validates their reports, masks, and final videos,
then atomically publishes a three-panel video and a quantitative JSON report:

    baseline final | expanded final | added/removed occlusion highlight

Raw and inpainted-background videos are optional inputs.  They are inferred
from the source reports when possible, metadata-checked, and can be selected as
the base of the difference panel.  Bright magenta denotes pixels occluded only
by the expanded result; cyan denotes pixels removed relative to the baseline.
"""

from __future__ import annotations

import argparse
import atexit
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from atomic_directory_publish import publish_directory


DEFAULT_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINAL_VIDEO_NAME = "video_overlay_contact.mp4"
MASK_NAME = "occluded_finger_mask.npy"
INPUT_REPORT_NAME = "report.json"
OUTPUT_VIDEO_NAME = "video_compare_contact_interior_expansion.mp4"
OUTPUT_REPORT_NAME = "comparison_report.json"
FPS_TOLERANCE = 0.1


@dataclass(frozen=True)
class VideoMetadata:
    """Minimal decoded-video contract used for synchronization checks."""

    path: Path
    width: int
    height: int
    frames: int
    fps: float

    def to_json(self) -> dict[str, Any]:
        values = asdict(self)
        values["path"] = str(self.path)
        return values


def probe_video(path: Path) -> VideoMetadata:
    """Return OpenCV metadata, rejecting missing or malformed videos."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    capture = cv2.VideoCapture(str(resolved))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {resolved}")
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or frames <= 0 or not np.isfinite(fps):
        raise ValueError(
            f"invalid video metadata for {resolved}: "
            f"{width}x{height}, {frames} frames, {fps} fps"
        )
    if fps <= 0:
        raise ValueError(f"invalid video fps for {resolved}: {fps}")
    return VideoMetadata(resolved, width, height, frames, fps)


def validate_video_alignment(
    metadata: dict[str, VideoMetadata],
    *,
    fps_tolerance: float = FPS_TOLERANCE,
) -> VideoMetadata:
    """Validate geometry, frame count, and fps for every named video."""
    if not metadata:
        raise ValueError("at least one video is required")
    reference_name = next(iter(metadata))
    reference = metadata[reference_name]
    mismatches: list[str] = []
    for name, current in metadata.items():
        if (current.width, current.height) != (
            reference.width,
            reference.height,
        ):
            mismatches.append(
                f"{name} geometry {current.width}x{current.height} != "
                f"{reference_name} {reference.width}x{reference.height}"
            )
        if current.frames != reference.frames:
            mismatches.append(
                f"{name} frame count {current.frames} != "
                f"{reference_name} {reference.frames}"
            )
        if not np.isclose(current.fps, reference.fps, atol=fps_tolerance):
            mismatches.append(
                f"{name} fps {current.fps:.6f} != "
                f"{reference_name} {reference.fps:.6f}"
            )
    if mismatches:
        raise ValueError("video alignment failed: " + "; ".join(mismatches))
    return reference


def _load_report(directory: Path, role: str) -> tuple[Path, dict[str, Any]]:
    path = directory / INPUT_REPORT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{role} report is missing: {path}")
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse {role} report: {path}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"{role} report must contain one JSON object: {path}")
    return path.resolve(), report


def _report_number(
    report: dict[str, Any],
    keys: Sequence[str],
    role: str,
) -> float:
    for key in keys:
        if key in report:
            value = report[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{role} report field {key!r} is not numeric")
            number = float(value)
            if not np.isfinite(number):
                raise ValueError(f"{role} report field {key!r} is not finite")
            return number
    raise ValueError(
        f"{role} report is missing required field "
        + "/".join(repr(key) for key in keys)
    )


def _validate_report_metadata(
    report: dict[str, Any],
    reference: VideoMetadata,
    role: str,
) -> dict[str, float | int]:
    frames = int(_report_number(report, ("frames", "frame_count"), role))
    width = int(_report_number(report, ("width",), role))
    height = int(_report_number(report, ("height",), role))
    fps = _report_number(report, ("fps",), role)
    mismatches = []
    if frames != reference.frames:
        mismatches.append(f"frames {frames} != {reference.frames}")
    if (width, height) != (reference.width, reference.height):
        mismatches.append(
            f"geometry {width}x{height} != "
            f"{reference.width}x{reference.height}"
        )
    if not np.isclose(fps, reference.fps, atol=FPS_TOLERANCE):
        mismatches.append(f"fps {fps:.6f} != {reference.fps:.6f}")
    if mismatches:
        raise ValueError(
            f"{role} report/video alignment failed: " + "; ".join(mismatches)
        )
    return {"frames": frames, "width": width, "height": height, "fps": fps}


def _source_path(
    report: dict[str, Any],
    report_path: Path,
    key: str,
) -> Path | None:
    sources = report.get("sources")
    if not isinstance(sources, dict):
        return None
    value = sources.get(key)
    if isinstance(value, dict):
        value = value.get("path") or value.get("video") or value.get("file")
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = report_path.parent / candidate
    return candidate.resolve()


def _require_file(path: Path, role: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} is missing: {resolved}")
    return resolved


def _infer_optional_video(
    explicit: Path | None,
    key: str,
    reports: Sequence[tuple[Path, dict[str, Any]]],
) -> tuple[Path | None, list[str]]:
    if explicit is not None:
        return _require_file(explicit, key.replace("_", " ")), []
    candidates = []
    missing = []
    for report_path, report in reports:
        candidate = _source_path(report, report_path, key)
        if candidate is None:
            continue
        if candidate.is_file():
            candidates.append(candidate)
        else:
            missing.append(str(candidate))
    unique = list(dict.fromkeys(candidates))
    if len(unique) > 1:
        raise ValueError(
            f"reports disagree on {key}: "
            + ", ".join(str(path) for path in unique)
            + f"; pass --{key} explicitly"
        )
    return (unique[0] if unique else None), missing


def _infer_finger_labels(
    explicit: Path | None,
    baseline_dir: Path,
    expanded_dir: Path,
    reports: Sequence[tuple[Path, dict[str, Any]]],
) -> tuple[Path | None, list[str]]:
    if explicit is not None:
        return _require_file(explicit, "finger-label array"), []
    candidates: list[Path] = []
    missing: list[str] = []
    for report_path, report in reports:
        direct = _source_path(report, report_path, "finger_labels")
        overlay = _source_path(report, report_path, "overlay_dir")
        processed = _source_path(report, report_path, "processed_demo")
        possible = [direct]
        if overlay is not None:
            possible.append(overlay / "robot_finger_labels.npy")
        if processed is not None:
            possible.append(
                processed / "overlay_processor" / "robot_finger_labels.npy"
            )
        for candidate in possible:
            if candidate is None:
                continue
            resolved = candidate.resolve()
            if resolved.is_file():
                candidates.append(resolved)
            else:
                missing.append(str(resolved))
    for directory in (baseline_dir, expanded_dir):
        candidate = directory.parent / "overlay_processor" / "robot_finger_labels.npy"
        if candidate.is_file():
            candidates.append(candidate.resolve())
    unique = list(dict.fromkeys(candidates))
    if len(unique) > 1:
        raise ValueError(
            "multiple finger-label arrays were inferred: "
            + ", ".join(str(path) for path in unique)
            + "; pass --finger_labels explicitly"
        )
    return (unique[0] if unique else None), missing


def _true_runs(values: np.ndarray) -> list[list[int]]:
    binary = np.asarray(values, dtype=bool)
    changes = np.diff(np.r_[False, binary, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [[int(start), int(end)] for start, end in zip(starts, ends)]


def _track_summary(pixel_count: np.ndarray) -> dict[str, Any]:
    values = np.asarray(pixel_count, dtype=np.int64)
    track = values > 0
    runs = _true_runs(track)
    run_lengths = [end - start + 1 for start, end in runs]
    nonzero = values[track]
    return {
        "pixels": int(values.sum()),
        "frames": int(track.sum()),
        "runs": runs,
        "run_count": len(runs),
        "median_run_frames": (
            float(np.median(run_lengths)) if run_lengths else 0.0
        ),
        "max_run_frames": max(run_lengths, default=0),
        "max_pixels_per_frame": int(values.max(initial=0)),
        "median_pixels_on_nonzero_frames": (
            float(np.median(nonzero)) if len(nonzero) else 0.0
        ),
    }


def _binary_frame(array: np.ndarray, frame_index: int, role: str) -> np.ndarray:
    raw = np.asarray(array[frame_index])
    if raw.dtype != np.bool_ and not np.all((raw == 0) | (raw == 1)):
        raise ValueError(f"{role} mask has non-binary values at frame {frame_index}")
    return raw.astype(bool, copy=False)


def compute_comparison_statistics(
    baseline_masks: np.ndarray,
    expanded_masks: np.ndarray,
    *,
    finger_labels: np.ndarray | None = None,
    finger_names: Sequence[str] = DEFAULT_FINGER_NAMES,
) -> dict[str, Any]:
    """Compute pixel, frame, run, and optional per-finger differences."""
    baseline = np.asanyarray(baseline_masks)
    expanded = np.asanyarray(expanded_masks)
    if baseline.ndim != 3 or expanded.ndim != 3:
        raise ValueError("occlusion masks must both have shape (T,H,W)")
    if baseline.shape != expanded.shape:
        raise ValueError(
            f"occlusion mask shape mismatch: {baseline.shape} != {expanded.shape}"
        )
    names = tuple(str(name) for name in finger_names)
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("finger names must be non-empty and unique")
    frames, height, width = baseline.shape
    labels_array = None if finger_labels is None else np.asanyarray(finger_labels)
    if labels_array is not None:
        if labels_array.ndim != 3 or len(labels_array) != frames:
            raise ValueError("finger labels must have shape (T,H,W) and align in time")
        if not np.issubdtype(labels_array.dtype, np.integer):
            raise ValueError("finger labels must use an integer dtype")

    tracks = {
        key: np.zeros(frames, dtype=np.int64)
        for key in ("baseline", "expanded", "added", "removed", "intersection", "union")
    }
    per_finger = None
    non_finger = None
    if labels_array is not None:
        per_finger = {
            key: np.zeros((frames, len(names)), dtype=np.int64)
            for key in ("baseline", "expanded", "added", "removed")
        }
        non_finger = {key: 0 for key in ("baseline", "expanded", "added", "removed")}

    for frame_index in range(frames):
        base = _binary_frame(baseline, frame_index, "baseline")
        grown = _binary_frame(expanded, frame_index, "expanded")
        added = grown & ~base
        removed = base & ~grown
        intersection = base & grown
        union = base | grown
        frame_masks = {
            "baseline": base,
            "expanded": grown,
            "added": added,
            "removed": removed,
            "intersection": intersection,
            "union": union,
        }
        for key, mask in frame_masks.items():
            tracks[key][frame_index] = int(mask.sum())

        if labels_array is None:
            continue
        labels_raw = np.asarray(labels_array[frame_index])
        if labels_raw.size and (
            int(labels_raw.min()) < 0 or int(labels_raw.max()) > len(names)
        ):
            raise ValueError(
                f"finger labels outside [0,{len(names)}] at frame {frame_index}"
            )
        labels = labels_raw.astype(np.uint8, copy=False)
        if labels.shape != (height, width):
            labels = cv2.resize(
                labels,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
        rendered_fingers = labels > 0
        assert per_finger is not None and non_finger is not None
        for key in per_finger:
            mask = frame_masks[key]
            non_finger[key] += int((mask & ~rendered_fingers).sum())
            for finger_index in range(len(names)):
                per_finger[key][frame_index, finger_index] = int(
                    (mask & (labels == finger_index + 1)).sum()
                )

    baseline_total = int(tracks["baseline"].sum())
    expanded_total = int(tracks["expanded"].sum())
    added_total = int(tracks["added"].sum())
    removed_total = int(tracks["removed"].sum())
    union_total = int(tracks["union"].sum())
    intersection_total = int(tracks["intersection"].sum())
    changed_track = tracks["added"] + tracks["removed"]
    statistics: dict[str, Any] = {
        "shape": {"frames": frames, "height": height, "width": width},
        "modes": {
            "baseline": _track_summary(tracks["baseline"]),
            "expanded": _track_summary(tracks["expanded"]),
        },
        "difference": {
            "added": _track_summary(tracks["added"]),
            "removed": _track_summary(tracks["removed"]),
            "changed": _track_summary(changed_track),
            "intersection_pixels": intersection_total,
            "union_pixels": union_total,
            "iou": (
                float(intersection_total / union_total) if union_total else 1.0
            ),
            "net_pixel_change": expanded_total - baseline_total,
            "added_percent_of_baseline": (
                float(100.0 * added_total / baseline_total)
                if baseline_total
                else None
            ),
            "removed_percent_of_baseline": (
                float(100.0 * removed_total / baseline_total)
                if baseline_total
                else None
            ),
        },
        "per_frame": {
            key: values.astype(int).tolist()
            for key, values in tracks.items()
            if key in ("baseline", "expanded", "added", "removed")
        },
        "per_finger": {
            "available": labels_array is not None,
            "names": list(names),
        },
        "invariants": {
            "baseline_subset_of_expanded": removed_total == 0,
            "net_change_equals_added_minus_removed": (
                expanded_total - baseline_total == added_total - removed_total
            ),
        },
    }

    if per_finger is not None and non_finger is not None:
        by_name = {}
        for finger_index, finger in enumerate(names):
            base_track = per_finger["baseline"][:, finger_index]
            expanded_track = per_finger["expanded"][:, finger_index]
            added_track = per_finger["added"][:, finger_index]
            removed_track = per_finger["removed"][:, finger_index]
            base_pixels = int(base_track.sum())
            by_name[finger] = {
                "baseline": _track_summary(base_track),
                "expanded": _track_summary(expanded_track),
                "added": _track_summary(added_track),
                "removed": _track_summary(removed_track),
                "net_pixel_change": int(expanded_track.sum() - base_track.sum()),
                "added_percent_of_baseline": (
                    float(100.0 * added_track.sum() / base_pixels)
                    if base_pixels
                    else None
                ),
            }
        statistics["per_finger"].update(
            {
                "values": by_name,
                "non_finger_pixels": non_finger,
            }
        )
        statistics["invariants"]["all_masks_are_finger_only"] = all(
            count == 0 for count in non_finger.values()
        )
    else:
        statistics["invariants"]["all_masks_are_finger_only"] = None
    return statistics


def _validate_report_counts(
    report: dict[str, Any],
    role: str,
    mode_statistics: dict[str, Any],
    per_frame: Sequence[int],
) -> list[str]:
    """Check every mask-derived count that the source report exposes."""
    checked = []
    expected_fields = {
        "occluded_pixels_total": int(mode_statistics["pixels"]),
        "frames_with_occlusion": int(mode_statistics["frames"]),
    }
    for key, expected in expected_fields.items():
        if key not in report:
            continue
        try:
            actual = int(report[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{role} report field {key!r} is invalid") from exc
        if actual != expected:
            raise ValueError(
                f"{role} report/mask mismatch for {key}: {actual} != {expected}"
            )
        checked.append(key)
    if "occluded_pixel_count" in report:
        actual_values = report["occluded_pixel_count"]
        if not isinstance(actual_values, list):
            raise ValueError(
                f"{role} report field 'occluded_pixel_count' is not a list"
            )
        try:
            actual = [int(value) for value in actual_values]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{role} report field 'occluded_pixel_count' is invalid"
            ) from exc
        expected = [int(value) for value in per_frame]
        if actual != expected:
            raise ValueError(
                f"{role} report/mask mismatch for occluded_pixel_count"
            )
        checked.append("occluded_pixel_count")
    return checked


def _finger_names_from_reports(
    baseline_report: dict[str, Any],
    expanded_report: dict[str, Any],
) -> tuple[str, ...]:
    values = []
    for report in (baseline_report, expanded_report):
        names = report.get("finger_names")
        if names is not None:
            if not isinstance(names, list) or not all(
                isinstance(name, str) and name for name in names
            ):
                raise ValueError("source report has invalid finger_names")
            values.append(tuple(names))
    if values and any(names != values[0] for names in values[1:]):
        raise ValueError("baseline and expanded reports disagree on finger_names")
    return values[0] if values else DEFAULT_FINGER_NAMES


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {path}")
    return writer


def _label_panel(
    frame: np.ndarray,
    title: str,
    frame_index: int,
    detail: str = "",
) -> np.ndarray:
    panel = np.asarray(frame, dtype=np.uint8).copy()
    header_height = min(58, panel.shape[0])
    cv2.rectangle(panel, (0, 0), (panel.shape[1], header_height), (0, 0, 0), -1)
    scale = 0.78 if panel.shape[0] >= 180 else 0.42
    thickness = 2 if panel.shape[0] >= 180 else 1
    cv2.putText(
        panel,
        title,
        (12, min(34, header_height - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    if detail and panel.shape[0] >= 80:
        cv2.putText(
            panel,
            detail,
            (12, min(53, header_height - 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    frame_text = f"frame {frame_index:04d}"
    (text_width, _), _ = cv2.getTextSize(
        frame_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        1,
    )
    if panel.shape[1] >= text_width + 18 and panel.shape[0] >= 60:
        cv2.putText(
            panel,
            frame_text,
            (panel.shape[1] - text_width - 10, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )
    return panel


def _difference_panel(
    base_frame: np.ndarray,
    added: np.ndarray,
    removed: np.ndarray,
) -> np.ndarray:
    panel = np.clip(
        np.asarray(base_frame, dtype=np.float32) * 0.42,
        0,
        255,
    ).astype(np.uint8)
    panel[added] = (255, 0, 255)  # bright magenta in BGR
    panel[removed] = (255, 255, 0)  # cyan in BGR
    kernel = np.ones((3, 3), dtype=np.uint8)
    for mask, color in ((added, (255, 255, 255)), (removed, (0, 255, 255))):
        if not mask.any():
            continue
        boundary = cv2.morphologyEx(
            mask.astype(np.uint8),
            cv2.MORPH_GRADIENT,
            kernel,
        ).astype(bool)
        panel[boundary] = color
    return panel


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return first == second or first in second.parents or second in first.parents


def _render_comparison(
    output_path: Path,
    *,
    baseline_video: Path,
    expanded_video: Path,
    base_video: Path | None,
    difference_base: str,
    baseline_masks: np.ndarray,
    expanded_masks: np.ndarray,
    reference: VideoMetadata,
) -> None:
    captures = {
        "baseline": cv2.VideoCapture(str(baseline_video)),
        "expanded": cv2.VideoCapture(str(expanded_video)),
    }
    if base_video is not None:
        captures["difference_base"] = cv2.VideoCapture(str(base_video))
    if not all(capture.isOpened() for capture in captures.values()):
        for capture in captures.values():
            capture.release()
        raise RuntimeError("could not open all comparison video streams")
    writer = _open_writer(
        output_path,
        reference.fps,
        (reference.width * 3, reference.height),
    )
    try:
        for frame_index in range(reference.frames):
            decoded = {}
            for name, capture in captures.items():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"{name} video read failed at frame {frame_index}"
                    )
                decoded[name] = frame
            baseline_mask = _binary_frame(
                baseline_masks, frame_index, "baseline"
            )
            expanded_mask = _binary_frame(
                expanded_masks, frame_index, "expanded"
            )
            added = expanded_mask & ~baseline_mask
            removed = baseline_mask & ~expanded_mask
            if difference_base == "baseline":
                diff_base = decoded["baseline"]
            elif difference_base == "expanded":
                diff_base = decoded["expanded"]
            else:
                diff_base = decoded["difference_base"]
            panels = [
                _label_panel(decoded["baseline"], "Baseline", frame_index),
                _label_panel(decoded["expanded"], "Interior expanded", frame_index),
                _label_panel(
                    _difference_panel(diff_base, added, removed),
                    "Occlusion difference",
                    frame_index,
                    f"magenta added {int(added.sum())} px | "
                    f"cyan removed {int(removed.sum())} px",
                ),
            ]
            writer.write(np.concatenate(panels, axis=1))
        for name, capture in captures.items():
            ok, _ = capture.read()
            if ok:
                raise RuntimeError(
                    f"{name} video contains more than {reference.frames} frames"
                )
    finally:
        for capture in captures.values():
            capture.release()
        writer.release()


def build_comparison(
    baseline_dir: Path,
    expanded_dir: Path,
    *,
    output_dir: Path | None = None,
    baseline_video: Path | None = None,
    expanded_video: Path | None = None,
    raw_video: Path | None = None,
    background_video: Path | None = None,
    finger_labels: Path | None = None,
    difference_base: str = "expanded",
) -> dict[str, Any]:
    """Validate inputs and atomically build the video and JSON comparison."""
    baseline_dir = Path(baseline_dir).expanduser().resolve()
    expanded_dir = Path(expanded_dir).expanduser().resolve()
    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"baseline directory is missing: {baseline_dir}")
    if not expanded_dir.is_dir():
        raise FileNotFoundError(f"expanded directory is missing: {expanded_dir}")
    if baseline_dir == expanded_dir:
        raise ValueError("baseline and expanded directories must differ")
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else expanded_dir.parent / "contact_interior_expansion_comparison"
    )
    for source in (baseline_dir, expanded_dir):
        if _paths_overlap(output_dir, source):
            raise ValueError(
                f"output directory must not overlap an input directory: {source}"
            )
    if difference_base not in {"baseline", "expanded", "raw", "background"}:
        raise ValueError(
            "difference_base must be baseline, expanded, raw, or background"
        )

    baseline_report_path, baseline_report = _load_report(
        baseline_dir, "baseline"
    )
    expanded_report_path, expanded_report = _load_report(
        expanded_dir, "expanded"
    )
    reports = (
        (baseline_report_path, baseline_report),
        (expanded_report_path, expanded_report),
    )
    baseline_video_path = _require_file(
        baseline_video or baseline_dir / FINAL_VIDEO_NAME,
        "baseline final video",
    )
    expanded_video_path = _require_file(
        expanded_video or expanded_dir / FINAL_VIDEO_NAME,
        "expanded final video",
    )
    raw_video_path, missing_raw = _infer_optional_video(
        raw_video, "raw_video", reports
    )
    background_video_path, missing_background = _infer_optional_video(
        background_video, "background", reports
    )
    finger_labels_path, missing_labels = _infer_finger_labels(
        finger_labels,
        baseline_dir,
        expanded_dir,
        reports,
    )

    metadata = {
        "baseline_final": probe_video(baseline_video_path),
        "expanded_final": probe_video(expanded_video_path),
    }
    if raw_video_path is not None:
        metadata["raw"] = probe_video(raw_video_path)
    if background_video_path is not None:
        metadata["background"] = probe_video(background_video_path)
    reference = validate_video_alignment(metadata)
    baseline_report_metadata = _validate_report_metadata(
        baseline_report, reference, "baseline"
    )
    expanded_report_metadata = _validate_report_metadata(
        expanded_report, reference, "expanded"
    )

    baseline_mask_path = _require_file(
        baseline_dir / MASK_NAME, "baseline occlusion mask"
    )
    expanded_mask_path = _require_file(
        expanded_dir / MASK_NAME, "expanded occlusion mask"
    )
    baseline_masks = np.load(baseline_mask_path, mmap_mode="r")
    expanded_masks = np.load(expanded_mask_path, mmap_mode="r")
    expected_shape = (reference.frames, reference.height, reference.width)
    if baseline_masks.shape != expected_shape:
        raise ValueError(
            f"baseline mask shape {baseline_masks.shape} != {expected_shape}"
        )
    if expanded_masks.shape != expected_shape:
        raise ValueError(
            f"expanded mask shape {expanded_masks.shape} != {expected_shape}"
        )
    labels_array = (
        np.load(finger_labels_path, mmap_mode="r")
        if finger_labels_path is not None
        else None
    )
    finger_names = _finger_names_from_reports(
        baseline_report, expanded_report
    )
    statistics = compute_comparison_statistics(
        baseline_masks,
        expanded_masks,
        finger_labels=labels_array,
        finger_names=finger_names,
    )
    statistics["per_finger"]["source"] = (
        str(finger_labels_path) if finger_labels_path is not None else None
    )
    statistics["per_finger"]["missing_inferred_candidates"] = missing_labels
    checked_counts = {
        "baseline": _validate_report_counts(
            baseline_report,
            "baseline",
            statistics["modes"]["baseline"],
            statistics["per_frame"]["baseline"],
        ),
        "expanded": _validate_report_counts(
            expanded_report,
            "expanded",
            statistics["modes"]["expanded"],
            statistics["per_frame"]["expanded"],
        ),
    }

    selected_base_video = None
    if difference_base == "raw":
        if raw_video_path is None:
            raise ValueError(
                "difference_base='raw' requires --raw_video or an existing "
                "sources.raw_video report entry"
            )
        selected_base_video = raw_video_path
    elif difference_base == "background":
        if background_video_path is None:
            raise ValueError(
                "difference_base='background' requires --background_video or "
                "an existing sources.background report entry"
            )
        selected_base_video = background_video_path

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".contact_interior_comparison.",
            dir=output_dir.parent,
        )
    )
    cleanup = lambda: shutil.rmtree(staging, ignore_errors=True)
    atexit.register(cleanup)
    try:
        video_output = staging / OUTPUT_VIDEO_NAME
        _render_comparison(
            video_output,
            baseline_video=baseline_video_path,
            expanded_video=expanded_video_path,
            base_video=selected_base_video,
            difference_base=difference_base,
            baseline_masks=baseline_masks,
            expanded_masks=expanded_masks,
            reference=reference,
        )
        rendered = probe_video(video_output)
        if (
            rendered.width != reference.width * 3
            or rendered.height != reference.height
            or rendered.frames != reference.frames
            or not np.isclose(rendered.fps, reference.fps, atol=FPS_TOLERANCE)
        ):
            raise RuntimeError(
                "rendered comparison failed validation: "
                f"{rendered.width}x{rendered.height}, {rendered.frames} frames, "
                f"{rendered.fps} fps"
            )

        report = {
            "schema_version": 1,
            "comparison": "contact_interior_expansion",
            "frames": reference.frames,
            "width": reference.width,
            "height": reference.height,
            "fps": reference.fps,
            "panel_layout": [
                "baseline_final",
                "expanded_final",
                "occlusion_difference",
            ],
            "difference_visualization": {
                "base": difference_base,
                "added_occlusion_bgr": [255, 0, 255],
                "removed_occlusion_bgr": [255, 255, 0],
                "mask_definition": {
                    "added": "expanded AND NOT baseline",
                    "removed": "baseline AND NOT expanded",
                },
            },
            "sources": {
                "baseline": {
                    "directory": str(baseline_dir),
                    "report": str(baseline_report_path),
                    "mask": str(baseline_mask_path),
                    "final_video": str(baseline_video_path),
                },
                "expanded": {
                    "directory": str(expanded_dir),
                    "report": str(expanded_report_path),
                    "mask": str(expanded_mask_path),
                    "final_video": str(expanded_video_path),
                },
                "raw_video": str(raw_video_path) if raw_video_path else None,
                "background_video": (
                    str(background_video_path) if background_video_path else None
                ),
                "finger_labels": (
                    str(finger_labels_path) if finger_labels_path else None
                ),
                "missing_inferred": {
                    "raw_video": missing_raw,
                    "background_video": missing_background,
                    "finger_labels": missing_labels,
                },
            },
            "input_metadata": {
                name: values.to_json() for name, values in metadata.items()
            },
            "source_report_validation": {
                "baseline_metadata": baseline_report_metadata,
                "expanded_metadata": expanded_report_metadata,
                "mask_count_fields_checked": checked_counts,
            },
            "statistics": statistics,
            "outputs": {
                "video": OUTPUT_VIDEO_NAME,
                "report": OUTPUT_REPORT_NAME,
            },
            "note": (
                "Added/removed masks measure implementation differences, not "
                "ground-truth occlusion accuracy."
            ),
        }
        (staging / OUTPUT_REPORT_NAME).write_text(
            json.dumps(report, indent=2) + "\n"
        )
        publish_directory(str(staging), str(output_dir))
        atexit.unregister(cleanup)
    except BaseException:
        cleanup()
        atexit.unregister(cleanup)
        raise
    print(f"[ok] interior-expansion comparison: {output_dir}", flush=True)
    print(
        "[info] "
        f"baseline={statistics['modes']['baseline']['pixels']}px, "
        f"expanded={statistics['modes']['expanded']['pixels']}px, "
        f"added={statistics['difference']['added']['pixels']}px, "
        f"removed={statistics['difference']['removed']['pixels']}px",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--expanded_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument(
        "--baseline_video",
        "--baseline_final_video",
        dest="baseline_video",
        type=Path,
        default=None,
        help=f"Defaults to BASELINE_DIR/{FINAL_VIDEO_NAME}",
    )
    parser.add_argument(
        "--expanded_video",
        "--expanded_final_video",
        dest="expanded_video",
        type=Path,
        default=None,
        help=f"Defaults to EXPANDED_DIR/{FINAL_VIDEO_NAME}",
    )
    parser.add_argument(
        "--raw_video",
        type=Path,
        default=None,
        help="Optional; inferred from reports when omitted",
    )
    parser.add_argument(
        "--background_video",
        "--background",
        dest="background_video",
        type=Path,
        default=None,
        help="Optional; inferred from reports when omitted",
    )
    parser.add_argument(
        "--finger_labels",
        type=Path,
        default=None,
        help="Optional robot_finger_labels.npy; inferred when possible",
    )
    parser.add_argument(
        "--difference_base",
        choices=("baseline", "expanded", "raw", "background"),
        default="expanded",
    )
    args = parser.parse_args()
    build_comparison(
        args.baseline_dir,
        args.expanded_dir,
        output_dir=args.out_dir,
        baseline_video=args.baseline_video,
        expanded_video=args.expanded_video,
        raw_video=args.raw_video,
        background_video=args.background_video,
        finger_labels=args.finger_labels,
        difference_base=args.difference_base,
    )


if __name__ == "__main__":
    main()
