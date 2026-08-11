"""Tests for the calibrated-versus-approximate HaCo inpainting comparison."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INPAINTING_DIR = REPO_ROOT / "src" / "inpainting"
sys.path.insert(0, str(INPAINTING_DIR))

import compare_calibration_inpainting_ab as comparison  # noqa: E402


def _metadata(
    path: Path,
    *,
    width: int = 2,
    height: int = 2,
    frames: int = 2,
    fps: Fraction = Fraction(24, 1),
    codec: str = "h264",
    pixel_format: str = "yuv420p",
) -> comparison.VideoMetadata:
    return comparison.VideoMetadata(
        path=path,
        width=width,
        height=height,
        frame_count=frames,
        fps=fps,
        duration_s=frames / float(fps),
        codec_name=codec,
        pixel_format=pixel_format,
    )


def _valid_report(
    *,
    frames: int = 2,
    width: int = 2,
    height: int = 2,
    hawor_npz: str = "/data/retarget_input.npz",
    source: str = "/data/video_L.mp4",
    focal_px: float = 900.0,
) -> dict:
    return {
        "method": comparison.HACO_COMPLETION_METHOD,
        "generated_texture": True,
        "physical_geometry_guarantee": False,
        "metadata": {
            "frames": frames,
            "width": width,
            "height": height,
            "fps": 24.0,
        },
        "counts": {"hidden_pixels_without_completed_depth": 0},
        "config": {"primary_hawor_focal_px": focal_px},
        "sources": {"hawor_npz": hawor_npz, "source": source},
        "invariants": {
            "trusted_modal_subset_input_modal": True,
            "trusted_modal_subset_amodal": True,
            "hand_contested_disjoint_trusted_modal": True,
            "hidden_disjoint_trusted_modal": True,
            "trusted_modal_rgb_has_priority": True,
            "hand_contested_input_modal_is_not_rgb_protected": True,
            "trajectory_arrays_unchanged": True,
            "haco_selected_hidden_subset_raw_hidden": True,
            "haco_does_not_measure_object_rgb_or_depth": True,
            "primary_view_owns_haco_projection": True,
            "auxiliary_haco_is_confidence_only": True,
            "auxiliary_geometry_used": False,
            "preencode_trusted_modal_rgb_values_changed": 0,
            "preencode_values_changed_outside_hidden": 0,
        },
        "outputs": {
            "baseline_video": "video_hand_removed_modal_only.mp4",
            "completed_video": "video_object_completed.mp4",
            "clean_modal_mask": "object_mask_observed_clean.npy",
            "amodal_mask": "object_mask_amodal.npy",
        },
    }


def _write_completion(
    root: Path,
    clean: np.ndarray,
    amodal: np.ndarray,
    *,
    hawor_npz: Path,
    source: Path,
    focal_px: float,
) -> None:
    root.mkdir()
    for filename in (
        "video_hand_removed_modal_only.mp4",
        "video_object_completed.mp4",
    ):
        (root / filename).write_bytes(b"video")
    np.save(root / "object_mask_observed_clean.npy", clean)
    np.save(root / "object_mask_amodal.npy", amodal)
    (root / "report.json").write_text(
        json.dumps(
            _valid_report(
                hawor_npz=str(hawor_npz.resolve()),
                source=str(source.resolve()),
                focal_px=focal_px,
            )
        )
    )


def _write_hawor(path: Path, focal_px: float) -> None:
    np.savez(path, img_focal=np.asarray(focal_px, dtype=np.float32))


class CalibrationInpaintingComparisonTests(unittest.TestCase):
    def test_report_rejects_non_haco_method_and_failed_invariant(self):
        report = _valid_report()
        report["method"] = "hand_cleaned_modal_object_constrained_e2fgvi"
        with self.assertRaisesRegex(ValueError, "not dual-view HaCo"):
            comparison.validate_haco_completion_report(report, name="approx")

        report = _valid_report()
        report["invariants"]["auxiliary_haco_is_confidence_only"] = False
        with self.assertRaisesRegex(ValueError, "auxiliary_haco"):
            comparison.validate_haco_completion_report(report, name="calibrated")

    def test_calibration_provenance_binds_reports_to_expected_branches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            approx_hawor = root / "approx_hawor.npz"
            calibrated_hawor = root / "calibrated_hawor.npz"
            _write_hawor(approx_hawor, 924.4444580078125)
            _write_hawor(calibrated_hawor, 1030.2115478515625)
            approx_source = root / "approx.mp4"
            calibrated_source = root / "calibrated.mp4"
            approx_source.write_bytes(b"approx")
            calibrated_source.write_bytes(b"calibrated")
            approx_report = _valid_report(
                hawor_npz=str(approx_hawor),
                source=str(approx_source),
                focal_px=924.4444580078125,
            )
            calibrated_report = _valid_report(
                hawor_npz=str(calibrated_hawor),
                source=str(calibrated_source),
                focal_px=1030.2115478515625,
            )

            result = comparison.validate_calibration_provenance(
                approx_report,
                calibrated_report,
                expected_approx_hawor_npz=approx_hawor,
                expected_calibrated_hawor_npz=calibrated_hawor,
                expected_approx_source=approx_source,
                expected_calibrated_source=calibrated_source,
            )
            self.assertEqual(result["expected_order"], "approx < calibrated")
            self.assertAlmostEqual(
                result["branches"]["approx"]["reported_focal_px"],
                924.4444580078125,
            )
            self.assertGreater(result["focal_delta_px"], 100.0)

            with self.assertRaisesRegex(ValueError, "different HaWoR branch"):
                comparison.validate_calibration_provenance(
                    calibrated_report,
                    approx_report,
                    expected_approx_hawor_npz=approx_hawor,
                    expected_calibrated_hawor_npz=calibrated_hawor,
                    expected_approx_source=approx_source,
                    expected_calibrated_source=calibrated_source,
                )

            calibrated_report["config"]["primary_hawor_focal_px"] = 900.0
            with self.assertRaisesRegex(ValueError, "report focal"):
                comparison.validate_calibration_provenance(
                    approx_report,
                    calibrated_report,
                    expected_approx_hawor_npz=approx_hawor,
                    expected_calibrated_hawor_npz=calibrated_hawor,
                    expected_approx_source=approx_source,
                    expected_calibrated_source=calibrated_source,
                )

    def test_original_rgb_identity_is_streamed_and_rejects_first_difference(self):
        frames = [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.full((2, 2, 3), 7, dtype=np.uint8),
        ]
        registry = {
            "approx.mp4": frames,
            "calibrated.mp4": [frame.copy() for frame in frames],
        }

        class FakeCapture:
            def __init__(self, path):
                self.frames = registry[path]
                self.index = 0
                self.released = False

            def isOpened(self):
                return True

            def read(self):
                if self.index >= len(self.frames):
                    return False, None
                frame = self.frames[self.index].copy()
                self.index += 1
                return True, frame

            def release(self):
                self.released = True

        with mock.patch.object(comparison.cv2, "VideoCapture", FakeCapture):
            result = comparison.stream_exact_original_rgb_identity(
                Path("approx.mp4"),
                Path("calibrated.mp4"),
                metadata=_metadata(Path("original.mp4")),
            )
        self.assertTrue(result["exact_equal"])
        self.assertEqual(result["compared_frames"], 2)
        self.assertEqual(len(result["decoded_rgb_sha256"]), 64)

        registry["calibrated.mp4"][1][0, 1, 2] = 8
        with (
            mock.patch.object(comparison.cv2, "VideoCapture", FakeCapture),
            self.assertRaisesRegex(ValueError, "frame 1 \(1 pixels\)"),
        ):
            comparison.stream_exact_original_rgb_identity(
                Path("approx.mp4"),
                Path("calibrated.mp4"),
                metadata=_metadata(Path("original.mp4")),
            )

    def test_mask_metrics_are_streamed_and_exact(self):
        approx = np.array(
            [
                [[True, False], [False, False]],
                [[True, True], [False, False]],
            ],
            dtype=bool,
        )
        calibrated = np.array(
            [
                [[True, True], [False, False]],
                [[False, True], [False, False]],
            ],
            dtype=bool,
        )
        result = comparison.stream_mask_comparison(approx, calibrated)
        self.assertEqual(result["approx_pixels"], 3)
        self.assertEqual(result["calibrated_pixels"], 3)
        self.assertEqual(result["intersection_pixels"], 2)
        self.assertEqual(result["union_pixels"], 4)
        self.assertEqual(result["symmetric_difference_pixels"], 2)
        self.assertAlmostEqual(result["iou"], 0.5)
        self.assertAlmostEqual(result["changed_pixel_ratio"], 2 / 8)

    def test_empty_masks_have_perfect_iou(self):
        empty = np.zeros((3, 2, 2), dtype=bool)
        result = comparison.stream_mask_comparison(empty, empty)
        self.assertEqual(result["iou"], 1.0)
        self.assertEqual(result["changed_pixel_ratio"], 0.0)

    def test_clean_mask_must_be_subset_of_amodal(self):
        clean = np.zeros((2, 2, 2), dtype=bool)
        amodal = clean.copy()
        clean[1, 0, 1] = True
        with self.assertRaisesRegex(ValueError, "frame 1"):
            comparison.validate_clean_subset_amodal(
                clean, amodal, name="approx"
            )

    def test_video_metadata_requires_exact_fps_and_equal_geometry(self):
        values = {
            "original": _metadata(Path("original.mp4")),
            "approx": _metadata(
                Path("approx.mp4"), fps=Fraction(30000, 1001)
            ),
        }
        with self.assertRaisesRegex(ValueError, "fps"):
            comparison.validate_source_video_metadata(values)

        values["approx"] = _metadata(Path("approx.mp4"), width=3)
        with self.assertRaisesRegex(ValueError, "geometry"):
            comparison.validate_source_video_metadata(values)

    def test_run_rejects_same_completion_directory_before_rendering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            approx_original = root / "approx.mp4"
            calibrated_original = root / "calibrated.mp4"
            approx_original.write_bytes(b"approx")
            calibrated_original.write_bytes(b"calibrated")
            same = root / "same_completion"
            with self.assertRaisesRegex(ValueError, "resolve to one path"):
                comparison.run_comparison(
                    approx_original,
                    calibrated_original,
                    same,
                    same,
                    root / "output",
                    expected_approx_hawor_npz=root / "unused_approx.npz",
                    expected_calibrated_hawor_npz=root / "unused_calibrated.npz",
                )

    def test_rgb_metrics_decode_one_frame_at_a_time(self):
        approx_frames = [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.uint8),
        ]
        calibrated_frames = [frame.copy() for frame in approx_frames]
        calibrated_frames[0][0, 0] = [1, 2, 3]
        calibrated_frames[1][1, 1] = [10, 0, 0]
        frame_registry = {
            "approx.mp4": approx_frames,
            "calibrated.mp4": calibrated_frames,
        }

        class FakeCapture:
            def __init__(self, path):
                self.frames = frame_registry[path]
                self.index = 0

            def isOpened(self):
                return True

            def read(self):
                if self.index >= len(self.frames):
                    return False, None
                frame = self.frames[self.index].copy()
                self.index += 1
                return True, frame

            def release(self):
                return None

        approx_mask = np.zeros((2, 2, 2), dtype=bool)
        calibrated_mask = np.zeros((2, 2, 2), dtype=bool)
        approx_mask[0, 0, 0] = True
        calibrated_mask[1, 1, 1] = True
        with mock.patch.object(comparison.cv2, "VideoCapture", FakeCapture):
            result = comparison.stream_completion_rgb_metrics(
                Path("approx.mp4"),
                Path("calibrated.mp4"),
                metadata=_metadata(Path("original.mp4")),
                approx_amodal=approx_mask,
                calibrated_amodal=calibrated_mask,
            )

        full = result["full_frame"]
        self.assertEqual(result["compared_frames"], 2)
        self.assertEqual(full["pixels"], 8)
        self.assertEqual(full["absolute_error_sum"], 16)
        self.assertEqual(full["changed_pixels"], 2)
        self.assertAlmostEqual(full["mae_rgb_u8"], 16 / 24)
        self.assertAlmostEqual(full["changed_pixel_ratio"], 2 / 8)
        support = result["amodal_union"]
        self.assertEqual(support["pixels"], 2)
        self.assertEqual(support["absolute_error_sum"], 16)
        self.assertEqual(support["changed_pixel_ratio"], 1.0)

    def test_run_publishes_labelled_h264_grid_and_report_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            approx_original = root / "approx_original.mp4"
            calibrated_original = root / "calibrated_original.mp4"
            approx_original.write_bytes(b"approx-original")
            calibrated_original.write_bytes(b"calibrated-original")
            approx_hawor = root / "approx_hawor.npz"
            calibrated_hawor = root / "calibrated_hawor.npz"
            _write_hawor(approx_hawor, 924.4444580078125)
            _write_hawor(calibrated_hawor, 1030.2115478515625)
            clean = np.zeros((2, 2, 2), dtype=bool)
            clean[:, 0, 0] = True
            amodal = clean.copy()
            amodal[:, 0, 1] = True
            approx = root / "approx"
            calibrated = root / "calibrated"
            _write_completion(
                approx,
                clean,
                amodal,
                hawor_npz=approx_hawor,
                source=approx_original,
                focal_px=924.4444580078125,
            )
            _write_completion(
                calibrated,
                clean,
                amodal,
                hawor_npz=calibrated_hawor,
                source=calibrated_original,
                focal_px=1030.2115478515625,
            )
            output = root / "comparison"

            source_metadata = _metadata(calibrated_original.resolve())
            grid_metadata = _metadata(
                output / comparison.VIDEO_NAME,
                width=comparison.GRID.output_width,
                height=comparison.GRID.output_height,
            )

            captured: dict[str, object] = {}

            def fake_probe(path):
                return comparison.VideoMetadata(
                    path=Path(path).resolve(),
                    width=source_metadata.width,
                    height=source_metadata.height,
                    frame_count=source_metadata.frame_count,
                    fps=source_metadata.fps,
                    duration_s=source_metadata.duration_s,
                    codec_name="h264",
                    pixel_format="yuv420p",
                )

            def fake_render(videos, path, *, layout, **_kwargs):
                captured["labels"] = [video.label for video in videos]
                captured["paths"] = [video.path for video in videos]
                captured["layout"] = layout
                path.write_bytes(b"h264-grid")
                return comparison.VideoMetadata(
                    path=path,
                    width=grid_metadata.width,
                    height=grid_metadata.height,
                    frame_count=grid_metadata.frame_count,
                    fps=grid_metadata.fps,
                    duration_s=grid_metadata.duration_s,
                    codec_name=grid_metadata.codec_name,
                    pixel_format=grid_metadata.pixel_format,
                )

            rgb_metrics = {
                "compared_frames": 2,
                "full_frame": {"mae_rgb_u8": 0.0, "changed_pixel_ratio": 0.0},
                "amodal_union": {"mae_rgb_u8": 0.0, "changed_pixel_ratio": 0.0},
            }
            with (
                mock.patch.object(comparison, "probe_video", fake_probe),
                mock.patch.object(
                    comparison,
                    "render_comparison_grid_layout",
                    side_effect=fake_render,
                ),
                mock.patch.object(
                    comparison,
                    "stream_completion_rgb_metrics",
                    return_value=rgb_metrics,
                ),
                mock.patch.object(
                    comparison,
                    "stream_exact_original_rgb_identity",
                    return_value={
                        "exact_equal": True,
                        "compared_frames": 2,
                        "comparison_space": "decoded BGR uint8",
                        "decoded_rgb_sha256": "a" * 64,
                    },
                ),
            ):
                result = comparison.run_comparison(
                    approx_original,
                    calibrated_original,
                    approx,
                    calibrated,
                    output,
                    expected_approx_hawor_npz=approx_hawor,
                    expected_calibrated_hawor_npz=calibrated_hawor,
                )

            self.assertEqual(result, output.resolve())
            self.assertTrue((output / comparison.VIDEO_NAME).is_file())
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["schema_version"], 2)
            self.assertTrue(report["validation"]["output_h264_yuv420p"])
            self.assertTrue(
                report["validation"]["original_decoded_rgb_exact_equal"]
            )
            self.assertAlmostEqual(
                report["calibration_provenance"]["branches"]["calibrated"]
                ["reported_focal_px"],
                1030.2115478515625,
            )
            self.assertEqual(report["metrics"]["masks"]["amodal"]["iou"], 1.0)
            self.assertEqual(captured["layout"], comparison.GRID)
            self.assertEqual(
                captured["labels"],
                [label for _name, label in comparison.PANEL_SPECS],
            )
            self.assertEqual(captured["paths"][0], approx_original.resolve())
            self.assertEqual(
                captured["paths"][3], calibrated_original.resolve()
            )


if __name__ == "__main__":
    unittest.main()
