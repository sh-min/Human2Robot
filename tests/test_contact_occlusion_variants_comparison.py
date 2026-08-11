"""Tests for the baseline/boundary/visibility-force comparison."""

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

import compare_contact_occlusion_variants as comparison  # noqa: E402


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
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


class VariantsComparisonTests(unittest.TestCase):
    def _build_fixture(
        self,
        root: Path,
        *,
        force_pixel_count_delta: int = 0,
    ) -> dict[str, Path | np.ndarray | int | float]:
        baseline_dir = root / "baseline"
        boundary_dir = root / "boundary"
        force_dir = root / "force"
        overlay_dir = root / "overlay_processor"
        for directory in (
            baseline_dir,
            boundary_dir,
            force_dir,
            overlay_dir,
        ):
            directory.mkdir()

        frame_count, height, width, fps = 3, 24, 32, 12.0
        frames = [
            np.full(
                (height, width, 3),
                (20 + frame_index * 30),
                dtype=np.uint8,
            )
            for frame_index in range(frame_count)
        ]
        _write_video(
            baseline_dir / comparison.FINAL_VIDEO_NAME,
            frames,
            fps,
        )
        _write_video(
            boundary_dir / comparison.FINAL_VIDEO_NAME,
            [np.flip(frame, axis=1) for frame in frames],
            fps,
        )
        _write_video(
            force_dir / comparison.FORCE_VIDEO_NAME,
            [np.flip(frame, axis=0) for frame in frames],
            fps,
        )

        baseline = np.zeros((frame_count, height, width), dtype=bool)
        boundary = np.zeros_like(baseline)
        force = np.zeros_like(baseline)
        labels = np.zeros_like(baseline, dtype=np.uint8)

        baseline[0, 10, 5:7] = True
        boundary[0, 10, 5:8] = True
        boundary[1, 12, 15:17] = True
        force[0, 10, [5, 7]] = True
        force[2, 15, 20:23] = True
        labels[0, 10, 5:8] = 1
        labels[1, 12, 15:17] = 2
        labels[2, 15, 20:23] = 5

        np.save(baseline_dir / comparison.MASK_NAME, baseline)
        np.save(boundary_dir / comparison.MASK_NAME, boundary)
        np.save(force_dir / comparison.FORCE_MASK_NAME, force)
        np.save(overlay_dir / "robot_finger_labels.npy", labels)

        def contact_report(mask: np.ndarray) -> dict[str, object]:
            return {
                "schema_version": 1,
                "frames": frame_count,
                "width": width,
                "height": height,
                "fps": fps,
                "finger_names": list(comparison.DEFAULT_FINGER_NAMES),
                "occluded_pixels_total": int(mask.sum()),
                "frames_with_occlusion": int(
                    mask.any(axis=(1, 2)).sum()
                ),
                "occluded_pixel_count": (
                    mask.sum(axis=(1, 2)).astype(int).tolist()
                ),
                "sources": {"overlay_dir": str(overlay_dir)},
            }

        (baseline_dir / comparison.INPUT_REPORT_NAME).write_text(
            json.dumps(contact_report(baseline))
        )
        (boundary_dir / comparison.INPUT_REPORT_NAME).write_text(
            json.dumps(contact_report(boundary))
        )

        force_per_finger = {}
        for finger_index, finger in enumerate(
            comparison.DEFAULT_FINGER_NAMES
        ):
            pixel_track = np.logical_and(
                force, labels == finger_index + 1
            ).sum(axis=(1, 2))
            force_per_finger[finger] = {
                "pixels": int(pixel_track.sum()),
                "frames": int((pixel_track > 0).sum()),
            }
        force_report = {
            "schema_version": 1,
            "frames": frame_count,
            "width": width,
            "height": height,
            "fps": fps,
            "finger_names": list(comparison.DEFAULT_FINGER_NAMES),
            "output_modes": ["visibility"],
            "mode_statistics": {
                "visibility": {
                    "pixels": int(force.sum()) + force_pixel_count_delta,
                    "frames": int(force.any(axis=(1, 2)).sum()),
                    "per_finger": force_per_finger,
                }
            },
            "sources": {"overlay_dir": str(overlay_dir)},
        }
        (force_dir / comparison.INPUT_REPORT_NAME).write_text(
            json.dumps(force_report)
        )
        return {
            "baseline_dir": baseline_dir,
            "boundary_dir": boundary_dir,
            "force_dir": force_dir,
            "overlay_dir": overlay_dir,
            "baseline": baseline,
            "boundary": boundary,
            "force": force,
            "frame_count": frame_count,
            "height": height,
            "width": width,
            "fps": fps,
        }

    def test_builds_atomic_2x2_video_and_three_pairwise_reports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = self._build_fixture(root)
            output_dir = root / "comparison"
            output_dir.mkdir()
            (output_dir / "stale.txt").write_text("old output")

            result = comparison.build_comparison(
                fixture["baseline_dir"],
                fixture["boundary_dir"],
                fixture["force_dir"],
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
                (
                    int(fixture["width"]) * 2,
                    int(fixture["height"]) * 2,
                    int(fixture["frame_count"]),
                ),
            )
            self.assertAlmostEqual(
                metadata.fps, float(fixture["fps"]), places=1
            )

            self.assertEqual(
                result["panel_layout"],
                [
                    ["baseline", "boundary"],
                    ["force", "force_vs_baseline_difference"],
                ],
            )
            self.assertEqual(set(result["modes"]), set(comparison.MODE_NAMES))
            self.assertEqual(result["modes"]["baseline"]["pixels"], 2)
            self.assertEqual(result["modes"]["boundary"]["pixels"], 5)
            self.assertEqual(result["modes"]["force"]["pixels"], 5)

            baseline_force = result["pairwise"]["baseline_vs_force"]
            self.assertEqual(baseline_force["first"], "baseline")
            self.assertEqual(baseline_force["second"], "force")
            stats = baseline_force["statistics"]
            self.assertEqual(stats["difference"]["added"]["pixels"], 4)
            self.assertEqual(stats["difference"]["removed"]["pixels"], 1)
            self.assertEqual(stats["difference"]["changed"]["frames"], 2)
            self.assertEqual(
                stats["per_finger"]["values"]["pinky"]["expanded"][
                    "pixels"
                ],
                3,
            )
            self.assertEqual(
                set(result["pairwise"]),
                {
                    "baseline_vs_boundary",
                    "baseline_vs_force",
                    "boundary_vs_force",
                },
            )
            self.assertTrue(
                result["invariants"]["all_masks_are_semantic_finger_only"]
            )
            self.assertIn(
                "mode_statistics.visibility.per_finger",
                result["source_report_validation"][
                    "mask_count_fields_checked"
                ]["force"],
            )
            self.assertEqual(
                json.loads(report_path.read_text())["pairwise"],
                result["pairwise"],
            )

    def test_force_report_mask_mismatch_does_not_replace_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = self._build_fixture(root, force_pixel_count_delta=1)
            output_dir = root / "comparison"
            output_dir.mkdir()
            sentinel = output_dir / "keep.txt"
            sentinel.write_text("preserved")

            with self.assertRaisesRegex(
                ValueError,
                "force report/mask mismatch",
            ):
                comparison.build_comparison(
                    fixture["baseline_dir"],
                    fixture["boundary_dir"],
                    fixture["force_dir"],
                    output_dir=output_dir,
                )
            self.assertEqual(sentinel.read_text(), "preserved")

    def test_rejects_disagreeing_finger_names(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            comparison._finger_names_from_reports(
                (
                    ("baseline", {"finger_names": ["thumb", "index"]}),
                    ("force", {"finger_names": ["thumb", "middle"]}),
                )
            )


if __name__ == "__main__":
    unittest.main()
