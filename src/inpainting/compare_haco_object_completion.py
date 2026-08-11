"""Compare broad hand-mask completion with dual-view HaCo selection.

The four synchronized panels are:

    original | broad HaWoR/hand-mask completion
    dual-HaCo-selected completion | HaCo completion + XHand barrier

Both the full-frame and shared dynamic-object-ROI grids are encoded as H.264
and published transactionally together with a provenance report.
"""

from __future__ import annotations

import argparse
import atexit
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from atomic_directory_publish import publish_directory
from compare_object_completion import _metadata_signature, _sha256
from compare_xhand_object_barriers import (
    dynamic_roi_centers,
    render_dynamic_roi_grid,
)
from make_video_comparison_grid import (
    GridLayout,
    NamedVideo,
    VideoMetadata,
    probe_video,
    render_comparison_grid_layout,
)


BROAD_METHOD = "hand_cleaned_modal_object_constrained_e2fgvi"
HACO_METHOD = "dual_haco_selected_hand_cleaned_object_constrained_e2fgvi"
MODE_SPECS = (
    ("original", "1 Original"),
    ("broad_completion", "2 HaWoR-only completion"),
    ("haco_completion", "3 Dual-HaCo-selected completion"),
    ("haco_barrier", "4 HaCo completion + XHand barrier"),
)
GRID = GridLayout(columns=2, rows=2)
FULL_VIDEO_NAME = "video_compare_haco_object_completion_2x2.mp4"
ROI_VIDEO_NAME = "video_compare_haco_object_completion_roi_2x2.mp4"
REPORT_NAME = "comparison_report.json"


def _require_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(resolved)
    return resolved


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(_require_file(path).read_text())
    if not isinstance(value, dict):
        raise TypeError(f"JSON report must be an object: {path}")
    return value


def _validate_completion_report(
    report: dict[str, object],
    *,
    expected_method: str,
    name: str,
) -> None:
    if report.get("method") != expected_method:
        raise ValueError(
            f"{name} completion method {report.get('method')!r} != "
            f"{expected_method!r}"
        )
    invariants = report.get("invariants", {})
    if not isinstance(invariants, dict):
        raise TypeError(f"{name} completion invariants must be a dictionary")
    expected_invariants = (
        "trusted_modal_subset_input_modal",
        "trusted_modal_subset_amodal",
        "hand_contested_disjoint_trusted_modal",
        "hidden_disjoint_trusted_modal",
        "trusted_modal_rgb_has_priority",
        "hand_contested_input_modal_is_not_rgb_protected",
        "trajectory_arrays_unchanged",
    )
    for invariant in expected_invariants:
        if invariants.get(invariant) is not True:
            raise ValueError(f"{name} completion invariant failed: {invariant}")
    if int(
        invariants.get("preencode_trusted_modal_rgb_values_changed", -1)
    ) != 0:
        raise ValueError(f"{name} completion changed trusted modal RGB")
    if int(invariants.get("preencode_values_changed_outside_hidden", -1)) != 0:
        raise ValueError(
            f"{name} completion changed RGB outside inferred hidden support"
        )
    counts = report.get("counts", {})
    if not isinstance(counts, dict):
        raise TypeError(f"{name} completion counts must be a dictionary")
    if int(counts.get("hidden_pixels_without_completed_depth", -1)) != 0:
        raise ValueError(f"{name} completion has hidden pixels without camera-Z")


def _validate_barrier_report(report: dict[str, object]) -> None:
    if report.get("method") != "visual_camera_z_xhand_barrier":
        raise ValueError("HaCo final overlay is not an XHand camera-Z barrier")
    if report.get("pose_state_modified") is not False:
        raise ValueError("HaCo barrier unexpectedly modified trajectory state")
    if report.get("metric_collision_guarantee") is not False:
        raise ValueError("HaCo barrier overstates physical collision provenance")
    counts = report.get("counts", {})
    if not isinstance(counts, dict):
        raise TypeError("HaCo barrier counts must be a dictionary")
    if int(counts.get("residual_violation_pixels", -1)) != 0:
        raise ValueError("HaCo barrier left camera-Z violations")
    invariants = report.get("invariants", {})
    if not isinstance(invariants, dict) or invariants.get(
        "valid_surface_barrier_residual_is_zero"
    ) is not True:
        raise ValueError("HaCo barrier zero-residual invariant is missing")
    config = report.get("config", {})
    if not isinstance(config, dict) or config.get(
        "object_restore_mask_explicit"
    ) is not True:
        raise ValueError("HaCo barrier did not separate modal RGB restoration")


def _validate_mask(
    path: Path,
    *,
    expected_shape: tuple[int, int, int],
    name: str,
) -> np.ndarray:
    mask = np.load(_require_file(path), mmap_mode="r", allow_pickle=False)
    if mask.shape != expected_shape or mask.dtype != np.bool_:
        raise ValueError(f"{name} mask must be bool {expected_shape}")
    return mask


def _metadata_dict(metadata: VideoMetadata) -> dict[str, object]:
    return {
        "width": metadata.width,
        "height": metadata.height,
        "frames": metadata.frame_count,
        "fps": float(metadata.fps),
        "duration_s": metadata.duration_s,
        "codec": metadata.codec_name,
        "pixel_format": metadata.pixel_format,
    }


def _validate_h264(metadata: VideoMetadata, *, name: str) -> None:
    if metadata.codec_name != "h264" or metadata.pixel_format != "yuv420p":
        raise RuntimeError(
            f"{name} output must be H.264/yuv420p, got "
            f"{metadata.codec_name}/{metadata.pixel_format}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--broad_completion_dir", type=Path, required=True)
    parser.add_argument("--haco_completion_dir", type=Path, required=True)
    parser.add_argument("--barrier_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    source = _require_file(args.source)
    broad_dir = args.broad_completion_dir.expanduser().resolve()
    haco_dir = args.haco_completion_dir.expanduser().resolve()
    barrier_dir = args.barrier_dir.expanduser().resolve()
    paths = {
        "original": source,
        "broad_completion": _require_file(
            broad_dir / "video_object_completed.mp4"
        ),
        "haco_completion": _require_file(
            haco_dir / "video_object_completed.mp4"
        ),
        "haco_barrier": _require_file(
            barrier_dir / "video_overlay_hand_barrier.mp4"
        ),
    }
    report_paths = {
        "broad_completion": _require_file(broad_dir / "report.json"),
        "haco_completion": _require_file(haco_dir / "report.json"),
        "haco_barrier": _require_file(barrier_dir / "report.json"),
    }
    mask_paths = {
        "broad_completion": _require_file(
            broad_dir / "object_mask_amodal.npy"
        ),
        "haco_completion": _require_file(
            haco_dir / "object_mask_amodal.npy"
        ),
    }

    reports = {
        name: _load_json(path)
        for name, path in report_paths.items()
    }
    _validate_completion_report(
        reports["broad_completion"],
        expected_method=BROAD_METHOD,
        name="broad",
    )
    _validate_completion_report(
        reports["haco_completion"],
        expected_method=HACO_METHOD,
        name="dual-HaCo",
    )
    _validate_barrier_report(reports["haco_barrier"])

    metadata = {name: probe_video(path) for name, path in paths.items()}
    reference_signature = _metadata_signature(metadata["original"])
    for name, value in metadata.items():
        if _metadata_signature(value) != reference_signature:
            raise ValueError(f"comparison video metadata differs for {name}")
    reference = metadata["original"]
    expected_shape = (
        reference.frame_count,
        reference.height,
        reference.width,
    )
    broad_mask = _validate_mask(
        mask_paths["broad_completion"],
        expected_shape=expected_shape,
        name="broad amodal object",
    )
    haco_mask = _validate_mask(
        mask_paths["haco_completion"],
        expected_shape=expected_shape,
        name="dual-HaCo amodal object",
    )
    # Keep the broad mask alive through validation so a truncated or malformed
    # baseline cannot pass merely because only the HaCo mask drives the crop.
    if len(broad_mask) != len(haco_mask):
        raise ValueError("completion mask frame counts differ")

    videos = [
        NamedVideo(label=label, path=paths[name])
        for name, label in MODE_SPECS
    ]
    output_dir = args.out_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".haco_object_completion_compare.",
            dir=output_dir.parent,
        )
    )
    atexit.register(shutil.rmtree, staging, ignore_errors=True)

    full_metadata = render_comparison_grid_layout(
        videos,
        staging / FULL_VIDEO_NAME,
        layout=GRID,
        overwrite=False,
        crf=18,
        preset="medium",
    )
    centres = dynamic_roi_centers(
        haco_mask,
        crop_width=640,
        crop_height=360,
        smooth_window=9,
    )
    roi_metadata = render_dynamic_roi_grid(
        videos,
        staging / ROI_VIDEO_NAME,
        centres=centres,
        crop_width=640,
        crop_height=360,
        fps=float(reference.fps),
    )
    _validate_h264(full_metadata, name="full-frame")
    _validate_h264(roi_metadata, name="dynamic-ROI")

    broad_report = reports["broad_completion"]
    haco_report = reports["haco_completion"]
    barrier_report = reports["haco_barrier"]
    report = {
        "schema_version": 1,
        "comparison": "broad HaWoR-only versus dual-HaCo-selected object completion",
        "layout": [
            [MODE_SPECS[0][0], MODE_SPECS[1][0]],
            [MODE_SPECS[2][0], MODE_SPECS[3][0]],
        ],
        "definitions": {name: label for name, label in MODE_SPECS},
        "videos": {
            "full_frame": FULL_VIDEO_NAME,
            "dynamic_roi": ROI_VIDEO_NAME,
        },
        "metadata": {
            "full_frame": _metadata_dict(full_metadata),
            "dynamic_roi": {
                **_metadata_dict(roi_metadata),
                "source_crop": [640, 360],
                "center_policy": (
                    "dual-HaCo amodal object bbox interpolation plus "
                    "9-frame moving average"
                ),
            },
        },
        "sources": {name: str(path) for name, path in paths.items()},
        "source_reports": {
            name: str(path) for name, path in report_paths.items()
        },
        "source_masks": {
            name: str(path) for name, path in mask_paths.items()
        },
        "source_hashes": {
            name: _sha256(path) for name, path in paths.items()
        },
        "validated_methods": {
            "broad_completion": broad_report["method"],
            "haco_completion": haco_report["method"],
            "haco_barrier": barrier_report["method"],
        },
        "completion_counts": {
            "broad": broad_report["counts"],
            "dual_haco": haco_report["counts"],
        },
        "completion_invariants": {
            "broad": broad_report["invariants"],
            "dual_haco": haco_report["invariants"],
        },
        "barrier_counts": barrier_report["counts"],
        "barrier_invariants": barrier_report["invariants"],
        "barrier_residual_violation_pixels": 0,
        "pose_state_modified": False,
        "physical_collision_solver": False,
        "metric_collision_guarantee": False,
        "provenance_warning": (
            "Dual-view HaCo selects contact-supported MH completion regions; "
            "SH contributes contact confidence only and no uncalibrated image "
            "geometry. Hidden object RGB and camera-Z remain inferred rather "
            "than measured texture or a watertight physical object model."
        ),
    }
    (staging / REPORT_NAME).write_text(json.dumps(report, indent=2) + "\n")
    publish_directory(str(staging), str(output_dir))
    print(f"[ok] dual-HaCo object-completion comparison: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
