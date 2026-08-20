"""Tests for baseline/interior-expanded contact comparison outputs."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INPAINTING_DIR = REPO_ROOT / "src" / "inpainting"
sys.path.insert(0, str(INPAINTING_DIR))

import compare_contact_interior_expansion as comparison  # noqa: E402


def _write_video(
    path: Path,
    frames: list[np.ndarray],
    fps: float,
) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("test video writer did not open")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


class ComparisonStatisticsTests(unittest.TestCase):
    def test_counts_added_removed_and_per_finger_pixels(self):
        baseline = np.zeros((2, 3, 4), dtype=bool)
        expanded = np.zeros_like(baseline)
        labels = np.zeros_like(baseline, dtype=np.uint8)

        baseline[0, 1, 0:2] = True
        expanded[0, 1, 1:3] = True
        expanded[1, 2, 3] = True
        labels[0, 1, 0:3] = 1
        labels[1, 2, 3] = 2

        result = comparison.compute_comparison_statistics(
            baseline,
            expanded,
            finger_labels=labels,
            finger_names=("thumb", "index"),
        )

        self.assertEqual(result["modes"]["baseline"]["pixels"], 2)
        self.assertEqual(result["modes"]["expanded"]["pixels"], 3)
        self.assertEqual(result["difference"]["added"]["pixels"], 2)
        self.assertEqual(result["difference"]["removed"]["pixels"], 1)
        self.assertEqual(result["difference"]["changed"]["frames"], 2)
        self.assertFalse(
            result["invariants"]["baseline_subset_of_expanded"]
        )
        self.assertTrue(
            result["invariants"]["net_change_equals_added_minus_removed"]
        )
        self.assertEqual(
            result["per_finger"]["values"]["thumb"]["added"]["pixels"],
            1,
        )
        self.assertEqual(
            result["per_finger"]["values"]["index"]["added"]["pixels"],
            1,
        )
        self.assertTrue(result["invariants"]["all_masks_are_finger_only"])

    def test_rejects_non_binary_masks_and_out_of_range_labels(self):
        baseline = np.zeros((1, 2, 2), dtype=np.uint8)
        expanded = baseline.copy()
        expanded[0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, "non-binary"):
            comparison.compute_comparison_statistics(baseline, expanded)

        expanded[0, 0, 0] = 1
        labels = np.full((1, 2, 2), 6, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "outside"):
            comparison.compute_comparison_statistics(
                baseline,
                expanded,
                finger_labels=labels,
            )

    def test_video_alignment_reports_all_relevant_mismatches(self):
        reference = comparison.VideoMetadata(
            Path("baseline.mp4"), 1280, 720, 100, 24.0
        )
        mismatch = comparison.VideoMetadata(
            Path("expanded.mp4"), 640, 360, 99, 30.0
        )
        with self.assertRaises(ValueError) as context:
            comparison.validate_video_alignment(
                {"baseline": reference, "expanded": mismatch}
            )
        message = str(context.exception)
        self.assertIn("geometry", message)
        self.assertIn("frame count", message)
        self.assertIn("fps", message)


class ComparisonIntegrationTests(unittest.TestCase):
    def test_builds_atomic_three_panel_video_and_report_with_inference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_dir = root / "baseline"
            expanded_dir = root / "expanded"
            overlay_dir = root / "overlay_processor"
            output_dir = root / "comparison"
            baseline_dir.mkdir()
            expanded_dir.mkdir()
            overlay_dir.mkdir()

            frame_count, height, width, fps = 3, 24, 32, 12.0
            video_frames = [
                np.full((height, width, 3), (20 + index * 20), np.uint8)
                for index in range(frame_count)
            ]
            raw_video = root / "raw.mp4"
            background_video = root / "background.mp4"
            _write_video(raw_video, video_frames, fps)
            _write_video(background_video, video_frames, fps)
            _write_video(
                baseline_dir / comparison.FINAL_VIDEO_NAME,
                video_frames,
                fps,
            )
            _write_video(
                expanded_dir / comparison.FINAL_VIDEO_NAME,
                [np.flip(frame, axis=1) for frame in video_frames],
                fps,
            )

            baseline = np.zeros(
                (frame_count, height, width), dtype=bool
            )
            expanded = np.zeros_like(baseline)
            baseline[0, 8:11, 8:11] = True
            expanded[0, 8:11, 8:12] = True
            expanded[1, 12:14, 15:17] = True
            np.save(baseline_dir / comparison.MASK_NAME, baseline)
            np.save(expanded_dir / comparison.MASK_NAME, expanded)
            labels = np.zeros_like(baseline, dtype=np.uint8)
            labels[0, 8:11, 8:12] = 1
            labels[1, 12:14, 15:17] = 2
            np.save(overlay_dir / "robot_finger_labels.npy", labels)

            def write_report(directory: Path, mask: np.ndarray) -> None:
                per_frame = mask.sum(axis=(1, 2)).astype(int).tolist()
                report = {
                    "schema_version": 1,
                    "frames": frame_count,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "finger_names": list(comparison.DEFAULT_FINGER_NAMES),
                    "occluded_pixels_total": int(mask.sum()),
                    "frames_with_occlusion": int((mask.any(axis=(1, 2))).sum()),
                    "occluded_pixel_count": per_frame,
                    "sources": {
                        "raw_video": str(raw_video),
                        "background": str(background_video),
                        "overlay_dir": str(overlay_dir),
                    },
                }
                (directory / comparison.INPUT_REPORT_NAME).write_text(
                    json.dumps(report)
                )

            write_report(baseline_dir, baseline)
            write_report(expanded_dir, expanded)
            output_dir.mkdir()
            (output_dir / "stale.txt").write_text("old output")

            result = comparison.build_comparison(
                baseline_dir,
                expanded_dir,
                output_dir=output_dir,
            )

            self.assertFalse((output_dir / "stale.txt").exists())
            video_path = output_dir / comparison.OUTPUT_VIDEO_NAME
            report_path = output_dir / comparison.OUTPUT_REPORT_NAME
            self.assertTrue(video_path.is_file())
            self.assertTrue(report_path.is_file())
            metadata = comparison.probe_video(video_path)
            self.assertEqual(
                (metadata.width, metadata.height, metadata.frames),
                (width * 3, height, frame_count),
            )
            self.assertAlmostEqual(metadata.fps, fps, places=1)
            self.assertEqual(
                result["statistics"]["difference"]["added"]["pixels"],
                7,
            )
            self.assertTrue(result["statistics"]["per_finger"]["available"])
            self.assertEqual(
                result["sources"]["raw_video"], str(raw_video.resolve())
            )
            self.assertEqual(
                result["source_report_validation"][
                    "mask_count_fields_checked"
                ]["expanded"],
                [
                    "occluded_pixels_total",
                    "frames_with_occlusion",
                    "occluded_pixel_count",
                ],
            )


if __name__ == "__main__":
    unittest.main()
