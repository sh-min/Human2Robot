"""Tests for the reusable 4x2 video comparison builder."""

from __future__ import annotations

import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INPAINTING_DIR = REPO_ROOT / "src" / "inpainting"
sys.path.insert(0, str(INPAINTING_DIR))

import make_video_comparison_grid as grid  # noqa: E402


def _metadata(
    index: int,
    *,
    frame_count: int = 643,
    fps: Fraction = Fraction(30, 1),
    duration_s: float = 21.433333,
) -> grid.VideoMetadata:
    return grid.VideoMetadata(
        path=Path(f"video_{index}.mp4"),
        width=1280,
        height=720,
        frame_count=frame_count,
        fps=fps,
        duration_s=duration_s,
        codec_name="h264",
        pixel_format="yuv420p",
    )


class ComparisonGridTests(unittest.TestCase):
    def test_requires_exactly_eight_named_inputs(self):
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            grid.parse_named_videos([["mode", "video.mp4"]] * 7)

    def test_rejects_duplicate_labels(self):
        raw = [[f"mode-{index}", f"video-{index}.mp4"] for index in range(8)]
        raw[-1][0] = raw[0][0]
        with self.assertRaisesRegex(ValueError, "duplicate video label"):
            grid.parse_named_videos(raw)

    def test_matching_metadata_is_accepted(self):
        metadata = [_metadata(index) for index in range(8)]
        reference = grid.validate_input_metadata(metadata)
        self.assertEqual(reference, metadata[0])

    def test_all_sync_mismatches_are_reported(self):
        metadata = [_metadata(index) for index in range(8)]
        metadata[1] = _metadata(1, frame_count=642)
        metadata[2] = _metadata(2, fps=Fraction(30000, 1001))
        metadata[3] = _metadata(3, duration_s=21.45)

        with self.assertRaises(ValueError) as context:
            grid.validate_input_metadata(metadata)

        message = str(context.exception)
        self.assertIn("frame count", message)
        self.assertIn("fps", message)
        self.assertIn("duration", message)

    def test_duration_tolerance_handles_container_rounding_only(self):
        metadata = [_metadata(index) for index in range(8)]
        metadata[-1] = _metadata(7, duration_s=21.434000)
        grid.validate_input_metadata(metadata, duration_tolerance_s=0.001)
        with self.assertRaisesRegex(ValueError, "duration"):
            grid.validate_input_metadata(metadata, duration_tolerance_s=0.0001)

    def test_filter_graph_has_eight_unobstructed_panels_and_label_headers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            labels = [Path(temp_dir) / f"label_{index}.txt" for index in range(8)]
            graph = grid.build_filter_graph(
                labels,
                Path("/fonts/NotoSansCJK-Bold.ttc"),
                Fraction(30000, 1001),
            )

        self.assertEqual(graph.count("scale=640:360"), 8)
        self.assertEqual(graph.count("color=c=black:s=640x40:r=30000/1001"), 8)
        self.assertEqual(graph.count("drawtext="), 8)
        self.assertEqual(graph.count("fontsize=24"), 8)
        self.assertEqual(graph.count("fix_bounds=1"), 8)
        self.assertEqual(graph.count("vstack=inputs=2:shortest=1"), 8)
        self.assertEqual(graph.count("setpts=N*1001/(30000*TB)"), 16)
        self.assertNotIn("drawbox=", graph)
        self.assertIn("xstack=inputs=8", graph)
        self.assertIn(
            "layout=0_0|640_0|1280_0|1920_0|0_400|640_400|1280_400|1920_400",
            graph,
        )
        self.assertTrue(graph.endswith("format=yuv420p[vout]"))

    def test_output_geometry_preserves_source_and_header_areas(self):
        self.assertEqual(grid.PANEL_WIDTH, 640)
        self.assertEqual(grid.PANEL_HEIGHT, 360)
        self.assertEqual(grid.HEADER_HEIGHT, 40)
        self.assertEqual(grid.TILE_HEIGHT, 400)
        self.assertEqual((grid.OUTPUT_WIDTH, grid.OUTPUT_HEIGHT), (2560, 800))

        reference = _metadata(0)
        rendered = grid.VideoMetadata(
            path=Path("grid.mp4"),
            width=2560,
            height=800,
            frame_count=643,
            fps=Fraction(30, 1),
            duration_s=21.433333,
            codec_name="h264",
            pixel_format="yuv420p",
        )
        grid._validate_rendered_output(rendered, reference, 0.001)

    def test_custom_4x3_grid_has_twelve_synchronized_panels(self):
        layout = grid.GridLayout(columns=4, rows=3)
        self.assertEqual(layout.video_count, 12)
        self.assertEqual((layout.output_width, layout.output_height), (2560, 1200))

        with tempfile.TemporaryDirectory() as temp_dir:
            labels = [Path(temp_dir) / f"label_{index}.txt" for index in range(12)]
            graph = grid.build_grid_filter_graph(
                labels,
                Path("/fonts/NotoSansCJK-Bold.ttc"),
                Fraction(24, 1),
                layout,
            )

        self.assertEqual(graph.count("scale=640:360"), 12)
        self.assertEqual(graph.count("drawtext="), 12)
        self.assertIn("xstack=inputs=12", graph)
        self.assertIn("1920_800", graph)

        metadata = [_metadata(index) for index in range(12)]
        reference = grid.validate_grid_input_metadata(metadata, 12)
        self.assertEqual(reference, metadata[0])

    def test_rendered_output_contract_checks_codec_and_geometry(self):
        reference = _metadata(0)
        wrong = grid.VideoMetadata(
            path=Path("grid.mp4"),
            width=1920,
            height=1080,
            frame_count=643,
            fps=Fraction(30, 1),
            duration_s=21.433333,
            codec_name="hevc",
            pixel_format="yuv444p",
        )
        with self.assertRaisesRegex(RuntimeError, "geometry.*codec.*pixel format"):
            grid._validate_rendered_output(wrong, reference, 0.001)


if __name__ == "__main__":
    unittest.main()
