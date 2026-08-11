"""Compare baseline, boundary-fill, and visibility-force occlusion outputs.

The three source directories are validated as frame-synchronized compositor
outputs before anything is published.  The resulting 2x2 video is laid out as

    baseline       | boundary fill
    visibility     | visibility minus baseline

where bright magenta marks occlusion added by the visibility-force result and
cyan marks baseline occlusion that it removes.  A JSON report contains a
summary for every mode and full pairwise pixel/frame/run/per-finger statistics.
"""

from __future__ import annotations

import argparse
import atexit
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from atomic_directory_publish import publish_directory
from compare_contact_interior_expansion import (
    DEFAULT_FINGER_NAMES,
    FINAL_VIDEO_NAME,
    INPUT_REPORT_NAME,
    MASK_NAME,
    FPS_TOLERANCE,
    VideoMetadata,
    _binary_frame,
    _difference_panel,
    _infer_finger_labels,
    _label_panel,
    _load_report,
    _open_writer,
    _paths_overlap,
    _require_file,
    _validate_report_counts,
    _validate_report_metadata,
    compute_comparison_statistics,
    probe_video,
    validate_video_alignment,
)


FORCE_VIDEO_NAME = "video_overlay_visibility.mp4"
FORCE_MASK_NAME = "occluded_finger_mask_visibility.npy"
FORCE_REPORT_MODE = "visibility"
OUTPUT_VIDEO_NAME = "video_compare_contact_occlusion_variants_2x2.mp4"
OUTPUT_REPORT_NAME = "comparison_report.json"
MODE_NAMES = ("baseline", "boundary", "force")
PAIR_SPECS = (
    ("baseline_vs_boundary", "baseline", "boundary"),
    ("baseline_vs_force", "baseline", "force"),
    ("boundary_vs_force", "boundary", "force"),
)


def _finger_names_from_reports(
    reports: Sequence[tuple[str, dict[str, Any]]],
) -> tuple[str, ...]:
    """Return one common finger-name contract across all source reports."""
    reported: list[tuple[str, tuple[str, ...]]] = []
    for role, report in reports:
        names = report.get("finger_names")
        if names is None:
            continue
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) and name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(f"{role} report has invalid finger_names")
        reported.append((role, tuple(names)))
    if reported:
        reference_role, reference = reported[0]
        mismatches = [
            f"{role}={names}"
            for role, names in reported[1:]
            if names != reference
        ]
        if mismatches:
            raise ValueError(
                "source reports disagree on finger_names: "
                f"{reference_role}={reference}; "
                + "; ".join(mismatches)
            )
        return reference
    return DEFAULT_FINGER_NAMES


def _validate_force_report_counts(
    report: dict[str, Any],
    mode_statistics: dict[str, Any],
    per_frame: Sequence[int],
    per_finger: dict[str, Any],
) -> list[str]:
    """Validate mask-derived counts exposed for stereo ``visibility`` mode."""
    output_modes = report.get("output_modes")
    if output_modes is not None:
        if not isinstance(output_modes, list) or FORCE_REPORT_MODE not in output_modes:
            raise ValueError(
                "force report output_modes does not contain 'visibility'"
            )

    all_modes = report.get("mode_statistics")
    if not isinstance(all_modes, dict):
        raise ValueError("force report is missing mode_statistics")
    values = all_modes.get(FORCE_REPORT_MODE)
    if not isinstance(values, dict):
        raise ValueError("force report is missing mode_statistics.visibility")

    checked: list[str] = []
    for key, expected in (
        ("pixels", int(mode_statistics["pixels"])),
        ("frames", int(mode_statistics["frames"])),
    ):
        try:
            actual = int(values[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"force report mode_statistics.visibility.{key} is invalid"
            ) from exc
        if actual != expected:
            raise ValueError(
                "force report/mask mismatch for "
                f"mode_statistics.visibility.{key}: {actual} != {expected}"
            )
        checked.append(f"mode_statistics.visibility.{key}")

    for key in ("occluded_pixel_count", "pixel_count", "pixels_per_frame"):
        if key not in values:
            continue
        raw = values[key]
        if not isinstance(raw, list):
            raise ValueError(
                f"force report mode_statistics.visibility.{key} is not a list"
            )
        try:
            actual = [int(value) for value in raw]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"force report mode_statistics.visibility.{key} is invalid"
            ) from exc
        expected = [int(value) for value in per_frame]
        if actual != expected:
            raise ValueError(
                "force report/mask mismatch for "
                f"mode_statistics.visibility.{key}"
            )
        checked.append(f"mode_statistics.visibility.{key}")

    report_fingers = values.get("per_finger")
    if report_fingers is not None:
        if not isinstance(report_fingers, dict):
            raise ValueError(
                "force report mode_statistics.visibility.per_finger is invalid"
            )
        for finger, expected_values in per_finger.items():
            actual_values = report_fingers.get(finger)
            if not isinstance(actual_values, dict):
                raise ValueError(
                    "force report is missing visibility per-finger statistics "
                    f"for {finger}"
                )
            for key in ("pixels", "frames"):
                try:
                    actual = int(actual_values[key])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "force report visibility per-finger field is invalid: "
                        f"{finger}.{key}"
                    ) from exc
                expected = int(expected_values[key])
                if actual != expected:
                    raise ValueError(
                        "force report/mask mismatch for visibility per-finger "
                        f"{finger}.{key}: {actual} != {expected}"
                    )
        checked.append("mode_statistics.visibility.per_finger")
    return checked


def _mode_summary(
    statistics: dict[str, Any],
    side: str,
    finger_names: Sequence[str],
) -> dict[str, Any]:
    """Extract a named mode summary from one pairwise-statistics result."""
    if side not in {"baseline", "expanded"}:
        raise ValueError(f"invalid pairwise side: {side}")
    summary = dict(statistics["modes"][side])
    by_finger = statistics["per_finger"].get("values")
    if not isinstance(by_finger, dict):
        raise ValueError("per-finger statistics are unavailable")
    summary["per_finger"] = {
        finger: dict(by_finger[finger][side]) for finger in finger_names
    }
    return summary


def _render_comparison(
    output_path: Path,
    *,
    video_paths: dict[str, Path],
    masks: dict[str, np.ndarray],
    reference: VideoMetadata,
) -> None:
    captures = {
        mode: cv2.VideoCapture(str(video_paths[mode])) for mode in MODE_NAMES
    }
    if not all(capture.isOpened() for capture in captures.values()):
        for capture in captures.values():
            capture.release()
        raise RuntimeError("could not open all comparison video streams")
    writer = _open_writer(
        output_path,
        reference.fps,
        (reference.width * 2, reference.height * 2),
    )
    try:
        for frame_index in range(reference.frames):
            decoded: dict[str, np.ndarray] = {}
            for mode, capture in captures.items():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"{mode} video read failed at frame {frame_index}"
                    )
                decoded[mode] = frame

            baseline_mask = _binary_frame(
                masks["baseline"], frame_index, "baseline"
            )
            force_mask = _binary_frame(masks["force"], frame_index, "force")
            added = force_mask & ~baseline_mask
            removed = baseline_mask & ~force_mask
            panels = {
                "baseline": _label_panel(
                    decoded["baseline"], "Baseline HaCo", frame_index
                ),
                "boundary": _label_panel(
                    decoded["boundary"], "Boundary fill (3 px)", frame_index
                ),
                "force": _label_panel(
                    decoded["force"], "Visibility force", frame_index
                ),
                "difference": _label_panel(
                    _difference_panel(decoded["force"], added, removed),
                    "Force - baseline",
                    frame_index,
                    f"magenta added {int(added.sum())} px | "
                    f"cyan removed {int(removed.sum())} px",
                ),
            }
            top = np.concatenate(
                (panels["baseline"], panels["boundary"]), axis=1
            )
            bottom = np.concatenate(
                (panels["force"], panels["difference"]), axis=1
            )
            writer.write(np.concatenate((top, bottom), axis=0))

        for mode, capture in captures.items():
            ok, _ = capture.read()
            if ok:
                raise RuntimeError(
                    f"{mode} video contains more than {reference.frames} frames"
                )
    finally:
        for capture in captures.values():
            capture.release()
        writer.release()


def build_comparison(
    baseline_dir: Path,
    boundary_dir: Path,
    force_dir: Path,
    *,
    output_dir: Path | None = None,
    finger_labels: Path | None = None,
) -> dict[str, Any]:
    """Validate three compositor outputs and atomically build their comparison."""
    source_dirs = {
        "baseline": Path(baseline_dir).expanduser().resolve(),
        "boundary": Path(boundary_dir).expanduser().resolve(),
        "force": Path(force_dir).expanduser().resolve(),
    }
    for mode, directory in source_dirs.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"{mode} directory is missing: {directory}")
    if len(set(source_dirs.values())) != len(source_dirs):
        raise ValueError("baseline, boundary, and force directories must differ")

    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else source_dirs["force"].parent
        / "contact_occlusion_variants_comparison"
    )
    for directory in source_dirs.values():
        if _paths_overlap(output_dir, directory):
            raise ValueError(
                "output directory must not overlap an input directory: "
                f"{directory}"
            )

    report_pairs = {
        mode: _load_report(directory, mode)
        for mode, directory in source_dirs.items()
    }
    reports_for_inference = tuple(report_pairs.values())
    named_reports = tuple(
        (mode, report_pairs[mode][1]) for mode in MODE_NAMES
    )
    finger_names = _finger_names_from_reports(named_reports)

    video_paths = {
        "baseline": _require_file(
            source_dirs["baseline"] / FINAL_VIDEO_NAME,
            "baseline final video",
        ),
        "boundary": _require_file(
            source_dirs["boundary"] / FINAL_VIDEO_NAME,
            "boundary final video",
        ),
        "force": _require_file(
            source_dirs["force"] / FORCE_VIDEO_NAME,
            "force final video",
        ),
    }
    metadata = {
        mode: probe_video(video_paths[mode]) for mode in MODE_NAMES
    }
    reference = validate_video_alignment(metadata)
    report_metadata = {
        mode: _validate_report_metadata(
            report_pairs[mode][1], reference, mode
        )
        for mode in MODE_NAMES
    }

    mask_paths = {
        "baseline": _require_file(
            source_dirs["baseline"] / MASK_NAME,
            "baseline occlusion mask",
        ),
        "boundary": _require_file(
            source_dirs["boundary"] / MASK_NAME,
            "boundary occlusion mask",
        ),
        "force": _require_file(
            source_dirs["force"] / FORCE_MASK_NAME,
            "force occlusion mask",
        ),
    }
    masks = {
        mode: np.load(mask_paths[mode], mmap_mode="r") for mode in MODE_NAMES
    }
    expected_shape = (reference.frames, reference.height, reference.width)
    for mode, mask in masks.items():
        if mask.shape != expected_shape:
            raise ValueError(
                f"{mode} mask shape {mask.shape} != {expected_shape}"
            )

    finger_labels_path, missing_labels = _infer_finger_labels(
        finger_labels,
        source_dirs["baseline"],
        source_dirs["boundary"],
        reports_for_inference,
    )
    if finger_labels_path is None:
        detail = (
            "; missing inferred candidates: " + ", ".join(missing_labels)
            if missing_labels
            else ""
        )
        raise FileNotFoundError(
            "could not infer robot_finger_labels.npy; pass --finger_labels"
            + detail
        )
    labels = np.load(finger_labels_path, mmap_mode="r")

    pairwise: dict[str, dict[str, Any]] = {}
    for pair_name, first, second in PAIR_SPECS:
        statistics = compute_comparison_statistics(
            masks[first],
            masks[second],
            finger_labels=labels,
            finger_names=finger_names,
        )
        if statistics["invariants"]["all_masks_are_finger_only"] is not True:
            raise ValueError(
                f"{pair_name} contains occlusion outside semantic fingers"
            )
        pairwise[pair_name] = {
            "first": first,
            "second": second,
            "statistics": statistics,
        }

    baseline_boundary = pairwise["baseline_vs_boundary"]["statistics"]
    baseline_force = pairwise["baseline_vs_force"]["statistics"]
    modes = {
        "baseline": _mode_summary(
            baseline_boundary, "baseline", finger_names
        ),
        "boundary": _mode_summary(
            baseline_boundary, "expanded", finger_names
        ),
        "force": _mode_summary(baseline_force, "expanded", finger_names),
    }
    checked_counts = {
        "baseline": _validate_report_counts(
            report_pairs["baseline"][1],
            "baseline",
            modes["baseline"],
            baseline_boundary["per_frame"]["baseline"],
        ),
        "boundary": _validate_report_counts(
            report_pairs["boundary"][1],
            "boundary",
            modes["boundary"],
            baseline_boundary["per_frame"]["expanded"],
        ),
        "force": _validate_force_report_counts(
            report_pairs["force"][1],
            modes["force"],
            baseline_force["per_frame"]["expanded"],
            modes["force"]["per_finger"],
        ),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".contact_occlusion_variants_comparison.",
            dir=output_dir.parent,
        )
    )
    cleanup = lambda: shutil.rmtree(staging, ignore_errors=True)
    atexit.register(cleanup)
    try:
        video_output = staging / OUTPUT_VIDEO_NAME
        _render_comparison(
            video_output,
            video_paths=video_paths,
            masks=masks,
            reference=reference,
        )
        rendered = probe_video(video_output)
        if (
            rendered.width != reference.width * 2
            or rendered.height != reference.height * 2
            or rendered.frames != reference.frames
            or not np.isclose(
                rendered.fps, reference.fps, atol=FPS_TOLERANCE
            )
        ):
            raise RuntimeError(
                "rendered comparison failed validation: "
                f"{rendered.width}x{rendered.height}, "
                f"{rendered.frames} frames, {rendered.fps} fps"
            )

        report = {
            "schema_version": 1,
            "comparison": "contact_occlusion_variants",
            "frames": reference.frames,
            "width": reference.width,
            "height": reference.height,
            "fps": reference.fps,
            "output_width": rendered.width,
            "output_height": rendered.height,
            "finger_names": list(finger_names),
            "panel_layout": [
                ["baseline", "boundary"],
                ["force", "force_vs_baseline_difference"],
            ],
            "difference_visualization": {
                "base": "force_final",
                "added_occlusion_bgr": [255, 0, 255],
                "removed_occlusion_bgr": [255, 255, 0],
                "mask_definition": {
                    "added": "force AND NOT baseline",
                    "removed": "baseline AND NOT force",
                },
            },
            "sources": {
                mode: {
                    "directory": str(source_dirs[mode]),
                    "report": str(report_pairs[mode][0]),
                    "mask": str(mask_paths[mode]),
                    "final_video": str(video_paths[mode]),
                }
                for mode in MODE_NAMES
            }
            | {"finger_labels": str(finger_labels_path)},
            "input_metadata": {
                mode: metadata[mode].to_json() for mode in MODE_NAMES
            },
            "source_report_validation": {
                "metadata": report_metadata,
                "mask_count_fields_checked": checked_counts,
                "missing_inferred_finger_label_candidates": missing_labels,
            },
            "modes": modes,
            "pairwise": pairwise,
            "invariants": {
                "videos_are_frame_aligned": True,
                "masks_are_frame_aligned": True,
                "all_masks_are_semantic_finger_only": True,
                "source_report_counts_match_masks": True,
            },
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

    force_difference = baseline_force["difference"]
    print(f"[ok] contact-occlusion variants comparison: {output_dir}", flush=True)
    print(
        "[info] "
        + ", ".join(
            f"{mode}={modes[mode]['pixels']}px/{modes[mode]['frames']}f"
            for mode in MODE_NAMES
        )
        + f", force_added={force_difference['added']['pixels']}px"
        + f", force_removed={force_difference['removed']['pixels']}px",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--boundary_dir", type=Path, required=True)
    parser.add_argument("--force_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--finger_labels",
        type=Path,
        default=None,
        help="Optional robot_finger_labels.npy override; inferred by default",
    )
    args = parser.parse_args()
    build_comparison(
        args.baseline_dir,
        args.boundary_dir,
        args.force_dir,
        output_dir=args.out_dir,
        finger_labels=args.finger_labels,
    )


if __name__ == "__main__":
    main()
