"""Create synchronized object-completion and final-overlay comparison videos."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from atomic_directory_publish import publish_directory
from compare_xhand_object_barriers import (
    dynamic_roi_centers,
    render_dynamic_roi_grid,
)
from make_video_comparison_grid import (
    GridLayout,
    NamedVideo,
    probe_video,
    render_comparison_grid_layout,
)


MODE_SPECS = (
    ("original", "1 Original: human hand"),
    ("hand_removed", "2 Hand removed: cleaned modal"),
    ("object_completed", "3 Object-aware completion"),
    ("final_overlay", "4 Completion + XHand barrier"),
)
GRID = GridLayout(columns=2, rows=2)
FULL_VIDEO_NAME = "video_compare_object_completion_2x2.mp4"
ROI_VIDEO_NAME = "video_compare_object_completion_roi_2x2.mp4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _metadata_signature(value) -> tuple[object, ...]:
    return (
        value.width,
        value.height,
        value.frame_count,
        value.fps,
        round(value.duration_s, 6),
    )


def validate_reports(
    completion_report: dict[str, object],
    barrier_report: dict[str, object],
) -> None:
    if (
        completion_report.get("method")
        != "hand_cleaned_modal_object_constrained_e2fgvi"
    ):
        raise ValueError("completion report method is not object-constrained E2FGVI")
    completion_invariants = completion_report.get("invariants", {})
    if not isinstance(completion_invariants, dict):
        raise TypeError("completion invariants must be a dictionary")
    expected_completion = {
        "trusted_modal_subset_input_modal": True,
        "trusted_modal_subset_amodal": True,
        "hand_contested_disjoint_trusted_modal": True,
        "hidden_disjoint_trusted_modal": True,
        "trusted_modal_rgb_has_priority": True,
        "hand_contested_input_modal_is_not_rgb_protected": True,
        "trajectory_arrays_unchanged": True,
    }
    for name, expected in expected_completion.items():
        if completion_invariants.get(name) is not expected:
            raise ValueError(f"completion invariant failed: {name}")
    if int(
        completion_invariants.get(
            "preencode_trusted_modal_rgb_values_changed", -1
        )
    ) != 0:
        raise ValueError("completion changed trusted modal RGB")
    if int(
        completion_invariants.get("preencode_values_changed_outside_hidden", -1)
    ) != 0:
        raise ValueError("completion changed RGB outside inferred hidden support")
    counts = completion_report.get("counts", {})
    if not isinstance(counts, dict) or int(
        counts.get("hidden_pixels_without_completed_depth", -1)
    ) != 0:
        raise ValueError("completed object support has missing camera-Z depth")

    if barrier_report.get("method") != "visual_camera_z_xhand_barrier":
        raise ValueError("final overlay is not an XHand camera-Z barrier")
    if barrier_report.get("pose_state_modified") is not False:
        raise ValueError("final overlay unexpectedly modified trajectory state")
    if barrier_report.get("metric_collision_guarantee") is not False:
        raise ValueError("final overlay overstates physical collision provenance")
    barrier_counts = barrier_report.get("counts", {})
    if not isinstance(barrier_counts, dict) or int(
        barrier_counts.get("residual_violation_pixels", -1)
    ) != 0:
        raise ValueError("final overlay left camera-Z barrier violations")
    barrier_config = barrier_report.get("config", {})
    if not isinstance(barrier_config, dict) or not bool(
        barrier_config.get("object_restore_mask_explicit", False)
    ):
        raise ValueError("final overlay did not separate modal RGB restoration")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--completion_dir", type=Path, required=True)
    parser.add_argument("--barrier_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    completion_dir = args.completion_dir.expanduser().resolve()
    barrier_dir = args.barrier_dir.expanduser().resolve()
    paths = {
        "original": source,
        "hand_removed": completion_dir / "video_hand_removed_modal_only.mp4",
        "object_completed": completion_dir / "video_object_completed.mp4",
        "final_overlay": barrier_dir / "video_overlay_hand_barrier.mp4",
    }
    completion_report_path = completion_dir / "report.json"
    barrier_report_path = barrier_dir / "report.json"
    amodal_mask_path = completion_dir / "object_mask_amodal.npy"
    for path in (
        *paths.values(),
        completion_report_path,
        barrier_report_path,
        amodal_mask_path,
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)

    completion_report = json.loads(completion_report_path.read_text())
    barrier_report = json.loads(barrier_report_path.read_text())
    validate_reports(completion_report, barrier_report)
    metadata = {name: probe_video(path) for name, path in paths.items()}
    signature = _metadata_signature(metadata["original"])
    for name, value in metadata.items():
        if _metadata_signature(value) != signature:
            raise ValueError(f"comparison video metadata differs for {name}")

    amodal = np.load(amodal_mask_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (
        metadata["original"].frame_count,
        metadata["original"].height,
        metadata["original"].width,
    )
    if amodal.shape != expected_shape or amodal.dtype != np.bool_:
        raise ValueError(f"amodal object mask must be bool {expected_shape}")

    videos = [
        NamedVideo(label=label, path=paths[name])
        for name, label in MODE_SPECS
    ]
    output_dir = args.out_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".object_completion_compare.", dir=output_dir.parent)
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
        amodal,
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
        fps=float(metadata["original"].fps),
    )
    report = {
        "schema_version": 1,
        "comparison": "human removal and hand-cleaned object completion",
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
            "full_frame": {
                "width": full_metadata.width,
                "height": full_metadata.height,
                "frames": full_metadata.frame_count,
                "fps": float(full_metadata.fps),
            },
            "dynamic_roi": {
                "width": roi_metadata.width,
                "height": roi_metadata.height,
                "frames": roi_metadata.frame_count,
                "fps": float(roi_metadata.fps),
                "source_crop": [640, 360],
                "center_policy": "amodal object bbox interpolation plus 9-frame moving average",
            },
        },
        "sources": {name: str(path) for name, path in paths.items()},
        "source_hashes": {name: _sha256(path) for name, path in paths.items()},
        "completion_counts": completion_report["counts"],
        "completion_invariants": completion_report["invariants"],
        "barrier_counts": barrier_report["counts"],
        "barrier_invariants": barrier_report["invariants"],
        "pose_state_modified": False,
        "physical_collision_solver": False,
        "provenance_warning": (
            "Hidden object RGB and camera-Z are inferred. Trusted modal RGB "
            "outside the HaWoR hand exclusion is preserved; overlapping input "
            "modal pixels are treated as occluded. The completion is not "
            "measured texture or a watertight physical object model."
        ),
    }
    (staging / "comparison_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    publish_directory(str(staging), str(output_dir))
    print(f"[ok] object-completion comparison: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
