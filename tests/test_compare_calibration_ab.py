"""Tests for scalar-focal HaWoR/HaCo A/B reporting and visualization."""

from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
HAND_ESTIMATION_DIR = REPO_ROOT / "src" / "hand_estimation"
sys.path.insert(0, str(HAND_ESTIMATION_DIR))

import compare_calibration_ab as comparison  # noqa: E402


def _hawor(focal: float) -> dict[str, np.ndarray]:
    frames = 2
    joints = np.zeros((frames, 21, 3), dtype=np.float32)
    vertices = np.zeros((frames, 778, 3), dtype=np.float32)
    joints[..., 0] = 1.0
    joints[..., 2] = 10.0
    vertices[..., 0] = 1.0
    vertices[..., 2] = 10.0
    return {
        "valid": np.asarray([[True, True], [False, True]], dtype=bool),
        "img_focal": np.asarray(focal, dtype=np.float32),
        "joints_left": joints.copy(),
        "joints_right": joints.copy(),
        "verts_left": vertices.copy(),
        "verts_right": vertices.copy(),
    }


def _contact_record(
    left_mask: np.ndarray,
    left_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    zeros_bool = np.zeros(778, dtype=bool)
    zeros_probability = np.zeros(778, dtype=np.float32)
    return {
        "left_valid": np.asarray(True),
        "left_contact_mask": left_mask,
        "left_contact_probability": left_probability,
        "right_valid": np.asarray(False),
        "right_contact_mask": zeros_bool,
        "right_contact_probability": zeros_probability,
    }


class CalibrationComparisonTests(unittest.TestCase):
    def test_projection_matches_centered_hawor_camera(self):
        points = np.asarray([[1.0, 2.0, 10.0], [0.0, 0.0, -1.0]])
        projected = comparison.project_points(points, 100.0, 640, 360)
        np.testing.assert_allclose(projected[0], [330.0, 200.0])
        self.assertTrue(np.isnan(projected[1]).all())

    def test_hawor_metrics_use_each_branch_focal(self):
        approx = _hawor(100.0)
        calibrated = _hawor(200.0)
        calibrated["valid"][0, 1] = False

        aggregate, sides = comparison.compare_hawor_arrays(
            approx, calibrated, width=640, height=360
        )
        report = aggregate.to_report()

        self.assertEqual(report["total_hand_frames"], 4)
        self.assertEqual(report["both_valid_hand_frames"], 2)
        self.assertEqual(report["approx_only_valid_hand_frames"], 1)
        self.assertEqual(report["neither_valid_hand_frames"], 1)
        self.assertAlmostEqual(report["validity_agreement_rate"], 0.75)
        self.assertAlmostEqual(
            report["projected_joint_displacement_px"]["mean"], 10.0
        )
        self.assertAlmostEqual(
            report["joint_3d_displacement_camera_units"]["mean"], 0.0
        )
        self.assertEqual(sides["left"].both_valid, 1)
        self.assertEqual(sides["right"].both_valid, 1)

    def test_haco_probability_and_mask_metrics(self):
        mask_a = np.zeros(778, dtype=bool)
        mask_b = np.zeros(778, dtype=bool)
        mask_a[[0, 1]] = True
        mask_b[[0, 2]] = True
        record_a = _contact_record(mask_a, np.zeros(778, dtype=np.float32))
        record_b = _contact_record(mask_b, np.ones(778, dtype=np.float32))

        aggregate, sides = comparison.compare_contact_records(
            [record_a], [record_b]
        )
        report = aggregate.to_report()

        self.assertEqual(report["total_hand_frames"], 2)
        self.assertEqual(report["both_valid_hand_frames"], 1)
        self.assertEqual(report["neither_valid_hand_frames"], 1)
        self.assertAlmostEqual(
            report["contact_probability_abs_delta"]["mean"], 1.0
        )
        self.assertAlmostEqual(
            report["contact_mask_iou_when_either_active"]["mean"], 1.0 / 3.0
        )
        self.assertAlmostEqual(
            report["contact_vertex_flip_rate"], 2.0 / 778.0
        )
        self.assertEqual(report["contact_mask_exact_agreement_rate"], 0.0)
        self.assertEqual(sides["right"].neither_valid, 1)

    def test_both_empty_contact_masks_are_reported_explicitly(self):
        empty = np.zeros(778, dtype=bool)
        probability = np.zeros(778, dtype=np.float32)
        aggregate, _ = comparison.compare_contact_records(
            [_contact_record(empty, probability)],
            [_contact_record(empty, probability)],
        )
        report = aggregate.to_report()
        self.assertEqual(report["both_empty_contact_mask_pairs"], 1)
        self.assertEqual(
            report["contact_mask_iou_including_both_empty_as_one"]["mean"],
            1.0,
        )
        self.assertIsNone(
            report["contact_mask_iou_when_either_active"]["mean"]
        )

    def test_comparison_canvas_is_a_labelled_two_by_two_grid(self):
        rgb = np.zeros((72, 128, 3), dtype=np.uint8)
        approx = _hawor(100.0)
        calibrated = _hawor(200.0)
        mask = np.zeros(778, dtype=bool)
        mask[:5] = True
        contact = _contact_record(mask, np.zeros(778, dtype=np.float32))

        canvas = comparison.compose_ab_frame(
            rgb,
            approx,
            calibrated,
            frame_idx=0,
            approx_contact=contact,
            calibrated_contact=contact,
            panel_width=64,
            panel_height=36,
        )
        self.assertEqual(canvas.shape, (176, 128, 3))
        self.assertGreater(int(canvas[:52].sum()), 0)

    def test_manifest_offset_maps_sh_to_mh_axis_and_fails_open_at_tail(self):
        keys = [f"rgb_frame{index:06d}" for index in range(8)]
        selection = comparison.aligned_frame_selection(keys, 5)
        self.assertEqual(
            selection,
            [
                (0, 5, "rgb_frame000005"),
                (1, 6, "rgb_frame000006"),
                (2, 7, "rgb_frame000007"),
            ],
        )
        fail_open = comparison.compose_fail_open_frame(64, 36, 7, 12)
        self.assertEqual(fail_open.shape, (176, 128, 3))
        self.assertGreater(int(fail_open.sum()), 0)

    def test_temporal_alignment_is_loaded_from_each_episode_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            episode = Path(temp_dir)
            manifest = {
                "temporal_alignment": {
                    "reference_view": "camera_2/MH/GT",
                    "camera1_frame_offset": 5,
                    "camera1_lookup": "SH = MH + 5",
                    "out_of_range_policy": "fail_open",
                    "motion_correlation_audit": {"status": "accepted"},
                }
            }
            (episode / "stereo_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            temporal = comparison.load_temporal_alignment(episode)
        self.assertEqual(temporal["camera1_frame_offset"], 5)
        self.assertEqual(
            temporal["motion_correlation_audit"]["status"], "accepted"
        )

    def test_natural_sort_and_strict_alignment(self):
        self.assertEqual(
            sorted(["10", "2", "1"], key=comparison.natural_key),
            ["1", "2", "10"],
        )
        with self.assertRaisesRegex(ValueError, "sets differ"):
            comparison._matching_keys(
                {"frame0": object()},
                {"frame1": object()},
                "test frame",
                allow_partial=False,
            )


if __name__ == "__main__":
    unittest.main()
