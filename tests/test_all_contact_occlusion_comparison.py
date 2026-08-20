"""Contract tests for the complete 3x4 before/after comparison."""

from __future__ import annotations

import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INPAINTING_DIR = REPO_ROOT / "src" / "inpainting"
sys.path.insert(0, str(INPAINTING_DIR))

import compare_all_contact_occlusion_results as comparison  # noqa: E402
from make_video_comparison_grid import VideoMetadata  # noqa: E402


def _metadata(path: Path, *, output: bool = False) -> VideoMetadata:
    return VideoMetadata(
        path=path,
        width=1920 if output else 1280,
        height=1600 if output else 720,
        frame_count=695,
        fps=Fraction(24, 1),
        duration_s=28.958333,
        codec_name="h264" if output else "mpeg4",
        pixel_format="yuv420p",
    )


class CompleteComparisonTests(unittest.TestCase):
    def test_layout_contains_nine_before_and_three_after_results(self):
        specs = comparison.PANEL_SPECS
        self.assertEqual(len(specs), 12)
        self.assertEqual(comparison.GRID.columns, 3)
        self.assertEqual(comparison.GRID.rows, 4)
        self.assertEqual(
            (comparison.GRID.output_width, comparison.GRID.output_height),
            (1920, 1600),
        )
        self.assertEqual(
            [spec.phase for spec in specs].count("before_object3d"),
            9,
        )
        self.assertEqual(
            [spec.phase for spec in specs].count("after_object3d"),
            3,
        )
        self.assertEqual(
            [spec.key for spec in specs[-3:]],
            [
                "scalar_object_depth",
                "dense_object_surface",
                "registered_object_surface",
            ],
        )

    def test_source_resolution_preserves_panel_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for spec in comparison.PANEL_SPECS:
                for relative_path in (spec.relative_video, spec.relative_report):
                    path = root / relative_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()

            sources = comparison.resolve_sources(root)

        self.assertEqual(
            [spec.key for spec, _, _ in sources],
            [spec.key for spec in comparison.PANEL_SPECS],
        )

    def test_report_numbers_panels_and_keeps_after_row_together(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = []
            metadata = []
            for spec in comparison.PANEL_SPECS:
                video = root / spec.relative_video
                report = root / spec.relative_report
                sources.append((spec, video, report))
                metadata.append(_metadata(video))
            rendered = _metadata(root / comparison.OUTPUT_VIDEO_NAME, output=True)

            report = comparison.build_report(root, sources, metadata, rendered)

        self.assertEqual(
            report["layout"][-1],
            [
                "scalar_object_depth",
                "dense_object_surface",
                "registered_object_surface",
            ],
        )
        self.assertEqual(
            report["panels"]["haco_baseline"]["label"],
            "1 BEFORE: HaCo contact-Z",
        )
        self.assertEqual(
            report["panels"]["registered_object_surface"]["number"],
            12,
        )
        self.assertTrue(
            report["invariants"]["labels_use_separate_non_occluding_headers"]
        )


if __name__ == "__main__":
    unittest.main()
