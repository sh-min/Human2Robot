"""Render every meaningful pre/post Object3D occlusion result in one grid.

The synchronized 3x4 layout keeps related strategies on the same row::

    HaCo baseline       | XHand half depth   | XHand full depth
    boundary fill       | SH visibility      | HaCo OR SH
    union safety shell  | front+side half    | front/side half + back full
    scalar object-Z     | dense surface      | surface + HaCo registration

Input/debug visualizations and exact duplicate baselines are intentionally not
panels.  The safety-shell result is retained because it was one of the compared
pre-Object3D strategies.  Original result videos are read-only; publication is
atomic.
"""

from __future__ import annotations

import argparse
import atexit
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from atomic_directory_publish import publish_directory
from make_video_comparison_grid import (
    GridLayout,
    NamedVideo,
    VideoMetadata,
    probe_video,
    render_comparison_grid_layout,
    validate_grid_input_metadata,
)


OUTPUT_VIDEO_NAME = "video_compare_all_before_after_3x4.mp4"
REPORT_NAME = "comparison_report.json"
GRID = GridLayout(columns=3, rows=4)


@dataclass(frozen=True)
class PanelSpec:
    key: str
    title: str
    relative_video: str
    relative_report: str
    phase: str
    definition: str


PANEL_SPECS = (
    PanelSpec(
        "haco_baseline",
        "BEFORE: HaCo contact-Z",
        "contact_occlusion_dual_haco_raw/video_overlay_contact.mp4",
        "contact_occlusion_dual_haco_raw/report.json",
        "before_object3d",
        "MH HaCo contact-Z gate with synchronized SH HaCo confidence support",
    ),
    PanelSpec(
        "xhand_half_depth",
        "BEFORE: XHand depth 0.5x",
        (
            "contact_occlusion_dual_haco_xhanddepth_s0p5_"
            "t39p16mm_f29p3mm_raw/video_overlay_contact.mp4"
        ),
        (
            "contact_occlusion_dual_haco_xhanddepth_s0p5_"
            "t39p16mm_f29p3mm_raw/report.json"
        ),
        "before_object3d",
        "HaCo contact-Z with half of the XHand finger thickness added",
    ),
    PanelSpec(
        "xhand_full_depth",
        "BEFORE: XHand depth 1.0x",
        (
            "contact_occlusion_dual_haco_xhanddepth_s1_"
            "t39p16mm_f29p3mm_raw/video_overlay_contact.mp4"
        ),
        (
            "contact_occlusion_dual_haco_xhanddepth_s1_"
            "t39p16mm_f29p3mm_raw/report.json"
        ),
        "before_object3d",
        "HaCo contact-Z with the full XHand finger thickness added",
    ),
    PanelSpec(
        "boundary_fill",
        "BEFORE: boundary interior fill",
        (
            "contact_occlusion_dual_haco_boundaryfill_3px_cap25_raw/"
            "video_overlay_contact.mp4"
        ),
        (
            "contact_occlusion_dual_haco_boundaryfill_3px_cap25_raw/"
            "report.json"
        ),
        "before_object3d",
        "three-pixel contact boundary seeds fill the enclosed finger interior",
    ),
    PanelSpec(
        "sh_visibility_force",
        "BEFORE: SH visibility force",
        "stereo_occlusion_visibility_force_raw/video_overlay_visibility.mp4",
        "stereo_occlusion_visibility_force_raw/report.json",
        "before_object3d",
        "SH-visible and MH-hidden evidence forces the matching finger behind",
    ),
    PanelSpec(
        "haco_sh_union",
        "BEFORE: HaCo OR SH force",
        (
            "contact_occlusion_compare_xhand_surface_strategies_raw/"
            "video_overlay_baseline_force_union.mp4"
        ),
        (
            "contact_occlusion_compare_xhand_surface_strategies_raw/"
            "comparison_report.json"
        ),
        "before_object3d",
        "pixelwise union of the HaCo baseline and SH visibility-force result",
    ),
    PanelSpec(
        "union_safety_shell",
        "BEFORE: union + safety shell",
        (
            "contact_occlusion_compare_xhand_surface_strategies_raw/"
            "video_overlay_union_safety_shell_diagnostic.mp4"
        ),
        (
            "contact_occlusion_compare_xhand_surface_strategies_raw/"
            "comparison_report.json"
        ),
        "before_object3d",
        "HaCo/SH union with the diagnostic two-dimensional safety shell",
    ),
    PanelSpec(
        "surface_front_side_half",
        "BEFORE: front 0x + side 0.5x",
        (
            "contact_occlusion_compare_xhand_surface_strategies_raw/"
            "video_overlay_surface_front_side_half.mp4"
        ),
        (
            "contact_occlusion_compare_xhand_surface_strategies_raw/"
            "comparison_report.json"
        ),
        "before_object3d",
        "XHand anatomical front uses baseline and side uses half thickness",
    ),
    PanelSpec(
        "surface_weighted",
        "BEFORE: front 0x + side 0.5x + back 1x",
        (
            "contact_occlusion_compare_xhand_surface_strategies_raw/"
            "video_overlay_surface_front_side_half_back_full.mp4"
        ),
        (
            "contact_occlusion_compare_xhand_surface_strategies_raw/"
            "comparison_report.json"
        ),
        "before_object3d",
        "front uses baseline, side half thickness, and back full thickness",
    ),
    PanelSpec(
        "scalar_object_depth",
        "AFTER: scalar object-Z",
        (
            "contact_occlusion_dual_haco_object3d_scalar_raw/"
            "video_overlay_contact.mp4"
        ),
        "contact_occlusion_dual_haco_object3d_scalar_raw/report.json",
        "after_object3d",
        "legacy HaCo gate intersected with one robust object depth per frame",
    ),
    PanelSpec(
        "dense_object_surface",
        "AFTER: dense object surface",
        (
            "contact_occlusion_dual_haco_object3d_surface_raw/"
            "video_overlay_contact.mp4"
        ),
        "contact_occlusion_dual_haco_object3d_surface_raw/report.json",
        "after_object3d",
        "dense per-pixel visible object surface owns front/behind ordering",
    ),
    PanelSpec(
        "registered_object_surface",
        "AFTER: surface + HaCo register",
        (
            "contact_occlusion_dual_haco_object3d_contact_aligned_raw/"
            "video_overlay_contact.mp4"
        ),
        (
            "contact_occlusion_dual_haco_object3d_contact_aligned_raw/"
            "report.json"
        ),
        "after_object3d",
        "dense surface after bounded local registration to MH HaCo contact-Z",
    ),
)


def _chunks(values: list[str], width: int) -> list[list[str]]:
    return [values[index : index + width] for index in range(0, len(values), width)]


def resolve_sources(processed_demo: Path) -> list[tuple[PanelSpec, Path, Path]]:
    root = processed_demo.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"processed demo does not exist: {root}")
    resolved: list[tuple[PanelSpec, Path, Path]] = []
    for spec in PANEL_SPECS:
        video = root / spec.relative_video
        report = root / spec.relative_report
        if not video.is_file():
            raise FileNotFoundError(f"{spec.key} video is missing: {video}")
        if not report.is_file():
            raise FileNotFoundError(f"{spec.key} report is missing: {report}")
        resolved.append((spec, video.resolve(), report.resolve()))
    return resolved


def validate_sources(
    sources: list[tuple[PanelSpec, Path, Path]],
    *,
    ffprobe: str = "ffprobe",
) -> tuple[list[VideoMetadata], VideoMetadata]:
    if len(sources) != GRID.video_count:
        raise ValueError(
            f"expected {GRID.video_count} comparison sources, got {len(sources)}"
        )
    keys = [spec.key for spec, _, _ in sources]
    if len(set(keys)) != len(keys):
        raise ValueError("comparison panel keys must be unique")
    metadata = [probe_video(video, ffprobe) for _, video, _ in sources]
    reference = validate_grid_input_metadata(metadata, GRID.video_count)
    geometry_errors = [
        f"{item.path}: {item.width}x{item.height}"
        for item in metadata
        if (item.width, item.height) != (reference.width, reference.height)
    ]
    if geometry_errors:
        raise ValueError(
            "comparison source geometry differs:\n  " + "\n  ".join(geometry_errors)
        )
    return metadata, reference


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


def build_report(
    processed_demo: Path,
    sources: list[tuple[PanelSpec, Path, Path]],
    metadata: list[VideoMetadata],
    rendered: VideoMetadata,
) -> dict[str, object]:
    keys = [spec.key for spec, _, _ in sources]
    return {
        "schema_version": 1,
        "comparison": "complete pre/post Object3D contact-occlusion history",
        "processed_demo": str(processed_demo.resolve()),
        "layout": _chunks(keys, GRID.columns),
        "panel_geometry": {
            "columns": GRID.columns,
            "rows": GRID.rows,
            "panel_width": GRID.panel_width,
            "panel_height": GRID.panel_height,
            "header_height": GRID.header_height,
        },
        "panels": {
            spec.key: {
                "number": index + 1,
                "label": f"{index + 1} {spec.title}",
                "phase": spec.phase,
                "definition": spec.definition,
                "video": str(video),
                "source_report": str(report),
                "metadata": _metadata_dict(item_metadata),
            }
            for index, ((spec, video, report), item_metadata) in enumerate(
                zip(sources, metadata, strict=True)
            )
        },
        "output": {
            "video": OUTPUT_VIDEO_NAME,
            **_metadata_dict(rendered),
        },
        "invariants": {
            "all_panels_frame_index_synchronized": True,
            "labels_use_separate_non_occluding_headers": True,
            "exact_duplicate_baselines_are_not_repeated": True,
            "diagnostic_input_visualizations_are_not_result_panels": True,
            "source_result_directories_are_read_only": True,
        },
        "note": (
            "Panel pixel counts are strategy outputs, not ground-truth accuracy. "
            "Object3D uses an MH camera-Z surface proxy; SH remains a synchronized "
            "HaCo confidence/visibility auxiliary without cross-camera projection."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_demo", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--font_file", type=Path, default=None)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    args = parser.parse_args()

    processed_demo = args.processed_demo.expanduser().resolve()
    sources = resolve_sources(processed_demo)
    metadata, _ = validate_sources(sources, ffprobe=args.ffprobe)
    videos = [
        NamedVideo(label=f"{index + 1} {spec.title}", path=video)
        for index, (spec, video, _) in enumerate(sources)
    ]

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".all_contact_compare.", dir=out_dir.parent)
    )
    atexit.register(shutil.rmtree, staging, ignore_errors=True)
    rendered = render_comparison_grid_layout(
        videos,
        staging / OUTPUT_VIDEO_NAME,
        layout=GRID,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        font_file=args.font_file,
        overwrite=False,
        crf=args.crf,
        preset=args.preset,
    )
    report = build_report(processed_demo, sources, metadata, rendered)
    (staging / REPORT_NAME).write_text(json.dumps(report, indent=2) + "\n")
    publish_directory(staging, out_dir)
    print(f"[ok] {out_dir / OUTPUT_VIDEO_NAME}", flush=True)


if __name__ == "__main__":
    main()
