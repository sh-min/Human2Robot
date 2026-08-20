import unittest

import numpy as np

from src.skill_classifier.evaluate_validation_long_sequences import (
    _edit_score,
    _metrics,
    _moving_average,
    _segment_f1,
    _segments,
    _boundary_ambiguous_mask,
    _tolerance_metrics,
    _tolerant_frame_correct,
)


class ValidationLongSequenceTests(unittest.TestCase):
    def test_segments_use_inclusive_annotation_bounds(self):
        self.assertEqual(
            _segments(np.array([0, 0, 1, 1, 0]), ["A", "B"]),
            [
                {"start_frame": 0, "end_frame": 1, "label": "A"},
                {"start_frame": 2, "end_frame": 3, "label": "B"},
                {"start_frame": 4, "end_frame": 4, "label": "A"},
            ],
        )

    def test_perfect_sequence_metrics(self):
        values = np.array([0, 0, 1, 1, 2, 2])
        result = _metrics(values, values, ["A", "B", "C"])
        self.assertEqual(result["frame_accuracy"], 1.0)
        self.assertEqual(result["edit_score"], 100.0)
        self.assertEqual(result["segment_f1_50"], 100.0)

    def test_edit_ignores_segment_duration(self):
        self.assertEqual(
            _edit_score(np.array([0, 0, 1, 2, 2]), np.array([0, 1, 1, 1, 2])),
            100.0,
        )

    def test_segment_f1_penalizes_fragmentation(self):
        gt = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        pred = np.array([0, 0, 1, 0, 1, 1, 1, 1])
        self.assertLess(_segment_f1(pred, gt, 0.5), 100.0)

    def test_centered_smoothing_keeps_shape(self):
        probability = np.eye(3, dtype=np.float32)
        smoothed = _moving_average(probability, width=3)
        self.assertEqual(smoothed.shape, probability.shape)
        np.testing.assert_allclose(smoothed.sum(axis=1), 1.0)

    def test_boundary_tolerance_accepts_either_adjacent_label(self):
        gt = np.array([0, 0, 1, 1])
        pred = np.array([0, 1, 0, 1])
        np.testing.assert_array_equal(
            _tolerant_frame_correct(pred, gt, radius=0),
            [True, False, False, True],
        )
        np.testing.assert_array_equal(
            _tolerant_frame_correct(pred, gt, radius=1),
            [True, True, True, True],
        )
        np.testing.assert_array_equal(
            _boundary_ambiguous_mask(gt, radius=1),
            [False, True, True, False],
        )

    def test_tolerance_zero_matches_strict_accuracy(self):
        gt = np.array([0, 0, 1, 1, 2])
        pred = np.array([0, 1, 1, 0, 2])
        result = _tolerance_metrics(pred, gt, radius=0)
        self.assertAlmostEqual(result["tolerant_accuracy"], 3 / 5)
        self.assertAlmostEqual(result["core_accuracy"], 3 / 5)
        self.assertEqual(result["core_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
