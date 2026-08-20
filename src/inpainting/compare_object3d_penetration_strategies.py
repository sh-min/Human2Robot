"""Compare Object3D surface-force and temporal penetration suppression.

Layout::

    contact-aligned baseline | full-finger surface-force
    temporal gap closing     | surface-force + temporal gap closing

All sources must share the same MH/SH/HaWoR/object-surface lineage.  The four
source result directories remain read-only and publication is atomic.
"""

from __future__ import annotations

import argparse
import atexit
import itertools
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from atomic_directory_publish import publish_directory
from make_video_comparison_grid import (
    GridLayout,
    NamedVideo,
    VideoMetadata,
    probe_video,
    render_comparison_grid_layout,
)


TEMPORAL_GAP_FRAMES = 2
MODE_SPECS = (
    ("baseline", "1 Baseline: contact-aligned Object3D"),
    ("surface_force", "2 Surface-force: full finger"),
    ("temporal", f"3 Temporal closing: <= {TEMPORAL_GAP_FRAMES}f"),
    ("force_temporal", "4 Surface-force + Temporal"),
)
GRID = GridLayout(columns=2, rows=2)
VIDEO_NAME = "video_overlay_contact.mp4"
MASK_NAME = "occluded_finger_mask.npy"
REPORT_NAME = "report.json"
OUTPUT_VIDEO_NAME = "video_compare_object3d_penetration_2x2.mp4"
PENETRATION_CONFIG_KEYS = {
    "object3d_force_surface",
    "object3d_force_margin_m",
    "object3d_temporal_max_gap_frames",
    "object3d_temporal_motion_px",
    "object3d_temporal_front_slack_m",
}


def _load_source(directory: Path, mode: str) -> dict[str, object]:
    root = directory.expanduser().resolve()
    video = root / VIDEO_NAME
    mask_path = root / MASK_NAME
    report_path = root / REPORT_NAME
    for path in (video, mask_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(f"{mode} source is incomplete: {path}")
    mask = np.load(mask_path, mmap_mode="r")
    if mask.ndim != 3 or mask.dtype != np.bool_:
        raise ValueError(f"{mode} mask must be bool (T,H,W), got {mask.shape}")
    report = json.loads(report_path.read_text())
    metadata = probe_video(video)
    if len(mask) != metadata.frame_count:
        raise ValueError(f"{mode} mask/video frame mismatch")
    if int(report.get("frames", -1)) != len(mask):
        raise ValueError(f"{mode} report/mask frame mismatch")
    if int(report.get("occluded_pixels_total", -1)) != int(mask.sum()):
        raise ValueError(f"{mode} report/mask pixel count mismatch")
    return {
        "root": root,
        "video": video,
        "mask": mask,
        "report": report,
        "metadata": metadata,
    }


def _normalized_config(report: dict[str, object]) -> dict[str, object]:
    config = dict(report.get("config", {}))
    for key in PENETRATION_CONFIG_KEYS:
        config.pop(key, None)
    return config


def _control(report: dict[str, object]) -> tuple[bool, int]:
    config = report.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("source report config must be an object")
    return (
        bool(config.get("object3d_force_surface", False)),
        int(config.get("object3d_temporal_max_gap_frames", 0)),
    )


def _metadata_signature(metadata: VideoMetadata) -> tuple[object, ...]:
    return (
        metadata.width,
        metadata.height,
        metadata.frame_count,
        metadata.fps,
        metadata.duration_s,
        metadata.codec_name,
        metadata.pixel_format,
    )


def _validate_strategy_lattice(masks: dict[str, np.ndarray]) -> None:
    required = {mode for mode, _ in MODE_SPECS}
    if set(masks) != required:
        raise ValueError("penetration comparison mask modes are incomplete")
    subset_pairs = (
        ("baseline", "surface_force"),
        ("baseline", "temporal"),
        ("surface_force", "force_temporal"),
        ("temporal", "force_temporal"),
    )
    for subset_name, superset_name in subset_pairs:
        subset = masks[subset_name]
        superset = masks[superset_name]
        if subset.shape != superset.shape:
            raise ValueError("penetration comparison mask shapes differ")
        for frame_index in range(len(subset)):
            if np.any(
                np.asarray(subset[frame_index], dtype=bool)
                & ~np.asarray(superset[frame_index], dtype=bool)
            ):
                raise ValueError(
                    f"strategy lattice failed: {subset_name} is not a subset "
                    f"of {superset_name} at frame {frame_index}"
                )


def _validate_contract(sources: dict[str, dict[str, object]]) -> None:
    reference = sources["baseline"]
    reference_report = reference["report"]
    reference_metadata = reference["metadata"]
    reference_mask = reference["mask"]
    assert isinstance(reference_report, dict)
    assert isinstance(reference_metadata, VideoMetadata)
    assert isinstance(reference_mask, np.ndarray)
    common_report_fields = (
        "frames",
        "width",
        "height",
        "fps",
        "side",
        "finger_names",
        "aux_frame_offset",
        "aux_side",
        "contact_score_fused",
        "hidden_fraction",
        "active_runs",
    )
    common_source_fields = (
        "processed_demo",
        "episode_dir",
        "background",
        "raw_video",
        "hawor_npz",
        "contact_dir",
        "aux_contact_dir",
        "overlay_dir",
        "object_mask",
        "object_surface_depth",
    )
    expected_controls = {
        "baseline": (False, 0),
        "surface_force": (True, 0),
        "temporal": (False, TEMPORAL_GAP_FRAMES),
        "force_temporal": (True, TEMPORAL_GAP_FRAMES),
    }
    for mode, source in sources.items():
        report = source["report"]
        metadata = source["metadata"]
        mask = source["mask"]
        assert isinstance(report, dict)
        assert isinstance(metadata, VideoMetadata)
        assert isinstance(mask, np.ndarray)
        if report.get("occlusion_mode") != "object3d":
            raise ValueError(f"{mode} is not an Object3D source")
        if report.get("object_surface_3d", {}).get("alignment") != "contact":
            raise ValueError(f"{mode} is not contact-aligned")
        if _control(report) != expected_controls[mode]:
            raise ValueError(
                f"{mode} penetration controls {_control(report)} != "
                f"{expected_controls[mode]}"
            )
        if _metadata_signature(metadata) != _metadata_signature(
            reference_metadata
        ):
            raise ValueError(f"{mode} video metadata differs from baseline")
        if mask.shape != reference_mask.shape:
            raise ValueError(f"{mode} mask shape differs from baseline")
        for field in common_report_fields:
            if report.get(field) != reference_report.get(field):
                raise ValueError(f"{mode} report field {field!r} differs")
        source_paths = report.get("sources", {})
        reference_paths = reference_report.get("sources", {})
        for field in common_source_fields:
            if source_paths.get(field) != reference_paths.get(field):
                raise ValueError(f"{mode} source {field!r} differs")
        if _normalized_config(report) != _normalized_config(reference_report):
            raise ValueError(f"{mode} non-penetration config differs")

    force_control = sources["surface_force"]["report"]
    temporal_control = sources["temporal"]["report"]
    combined_control = sources["force_temporal"]["report"]
    assert isinstance(force_control, dict)
    assert isinstance(temporal_control, dict)
    assert isinstance(combined_control, dict)
    if not force_control.get("invariants", {}).get(
        "object3d_force_bypasses_haco_selector"
    ):
        raise ValueError("surface-force does not bypass the HaCo selector")
    for mode, report in (
        ("temporal", temporal_control),
        ("force_temporal", combined_control),
    ):
        temporal_report = report.get("object3d_penetration_control", {}).get(
            "temporal_filter", {}
        )
        if not temporal_report.get("enabled"):
            raise ValueError(f"{mode} temporal filter is disabled")
        if not temporal_report.get("uses_future_frames"):
            raise ValueError(f"{mode} is not the bounded bidirectional filter")
    _validate_strategy_lattice(
        {
            mode: source["mask"]
            for mode, source in sources.items()
            if isinstance(source["mask"], np.ndarray)
        }
    )


def _statistics(sources: dict[str, dict[str, object]]) -> dict[str, object]:
    masks = {
        mode: source["mask"]
        for mode, source in sources.items()
    }
    mode_stats = {
        mode: {
            "pixels": int(mask.sum()),
            "frames": int(np.any(mask, axis=(1, 2)).sum()),
        }
        for mode, mask in masks.items()
        if isinstance(mask, np.ndarray)
    }
    pairs: dict[str, dict[str, int]] = {}
    for first_mode, second_mode in itertools.combinations(masks, 2):
        first = masks[first_mode]
        second = masks[second_mode]
        assert isinstance(first, np.ndarray) and isinstance(second, np.ndarray)
        added = removed = changed_frames = 0
        for frame_index in range(len(first)):
            first_frame = np.asarray(first[frame_index], dtype=bool)
            second_frame = np.asarray(second[frame_index], dtype=bool)
            frame_added = int((second_frame & ~first_frame).sum())
            frame_removed = int((first_frame & ~second_frame).sum())
            added += frame_added
            removed += frame_removed
            changed_frames += int(frame_added + frame_removed > 0)
        pairs[f"{first_mode}_vs_{second_mode}"] = {
            "added_pixels": added,
            "removed_pixels": removed,
            "changed_frames": changed_frames,
        }
    return {"modes": mode_stats, "comparisons": pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--surface_force_dir", type=Path, required=True)
    parser.add_argument("--temporal_dir", type=Path, required=True)
    parser.add_argument("--force_temporal_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    directories = {
        "baseline": args.baseline_dir,
        "surface_force": args.surface_force_dir,
        "temporal": args.temporal_dir,
        "force_temporal": args.force_temporal_dir,
    }
    sources = {
        mode: _load_source(directory, mode)
        for mode, directory in directories.items()
    }
    _validate_contract(sources)
    statistics = _statistics(sources)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".object3d_penetration_compare.", dir=out_dir.parent)
    )
    atexit.register(shutil.rmtree, staging, ignore_errors=True)
    videos = [
        NamedVideo(label=label, path=Path(sources[mode]["video"]))
        for mode, label in MODE_SPECS
    ]
    rendered = render_comparison_grid_layout(
        videos,
        staging / OUTPUT_VIDEO_NAME,
        layout=GRID,
        overwrite=False,
        crf=18,
        preset="medium",
    )
    report = {
        "schema_version": 1,
        "comparison": "Object3D penetration suppression 2x2 factor grid",
        "layout": [
            [MODE_SPECS[0][0], MODE_SPECS[1][0]],
            [MODE_SPECS[2][0], MODE_SPECS[3][0]],
        ],
        "definitions": {
            "baseline": "contact-aligned Object3D with HaCo finger/disk selector",
            "surface_force": (
                "baseline OR every semantic-finger pixel behind the raw dense "
                "surface; no HaCo selector for the added branch"
            ),
            "temporal": (
                "baseline plus bounded offline bidirectional closing of gaps "
                f"up to {TEMPORAL_GAP_FRAMES} frames"
            ),
            "force_temporal": "surface-force followed by the same temporal filter",
        },
        "video": OUTPUT_VIDEO_NAME,
        "frames": rendered.frame_count,
        "fps": float(rendered.fps),
        "width": rendered.width,
        "height": rendered.height,
        "sources": {
            mode: str(source["root"])
            for mode, source in sources.items()
        },
        "invariants": {
            "baseline_subset_surface_force": True,
            "baseline_subset_temporal": True,
            "surface_force_subset_combined": True,
            "temporal_subset_combined": True,
            "all_source_lineage_equal": True,
            "temporal_is_offline_bidirectional": True,
        },
        **statistics,
    }
    (staging / "comparison_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    publish_directory(staging, out_dir)
    print(f"[ok] {out_dir / OUTPUT_VIDEO_NAME}", flush=True)


if __name__ == "__main__":
    main()
