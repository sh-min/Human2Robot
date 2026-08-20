"""Focused checks for the whole-XHand visual camera-Z barrier."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INPAINTING_DIR = REPO_ROOT / "src" / "inpainting"
sys.path.insert(0, str(INPAINTING_DIR))

import composite_rb5_contact_occlusion as contact  # noqa: E402
import composite_xhand_object_barrier as barrier  # noqa: E402


class XHandObjectBarrierTests(unittest.TestCase):
    def test_object_restore_mask_defaults_to_barrier_mask(self):
        object_mask = np.asarray((((True, False),),), dtype=bool)

        selected = barrier.select_object_restore_mask(object_mask, None)

        self.assertIs(selected, object_mask)

    def test_object_restore_mask_controls_only_raw_rgb_restoration(self):
        background = np.zeros((1, 3, 3), dtype=np.uint8)
        raw = np.full((1, 3, 3), 200, dtype=np.uint8)
        restore_mask = np.asarray(((False, True, False),), dtype=bool)

        barrier_mask, _ = barrier.compute_visual_barrier(
            robot_mask=np.ones((1, 3), dtype=bool),
            hand_mask=np.ones((1, 3), dtype=bool),
            robot_depth=np.full((1, 3), 1.1, dtype=np.float32),
            object_mask=np.asarray(((True, False, True),), dtype=bool),
            object_surface_depth=np.ones((1, 3), dtype=np.float32),
            shell_m=np.zeros((1, 3), dtype=np.float32),
        )

        restored = barrier.restore_raw_object_pixels(
            background,
            raw,
            restore_mask,
        )

        np.testing.assert_array_equal(barrier_mask, ((True, False, True),))
        np.testing.assert_array_equal(restored[0, 0], (0, 0, 0))
        np.testing.assert_array_equal(restored[0, 1], (200, 200, 200))
        np.testing.assert_array_equal(restored[0, 2], (0, 0, 0))
        self.assertFalse(background.any(), "input background must remain unchanged")

    def test_object_restore_mask_volume_requires_video_shape_and_bool(self):
        expected = (3, 4, 5)
        barrier.validate_mask_volume(
            np.zeros(expected, dtype=bool),
            name="object_restore_mask",
            expected_shape=expected,
        )
        with self.assertRaisesRegex(ValueError, "differs from video"):
            barrier.validate_mask_volume(
                np.zeros((2, 4, 5), dtype=bool),
                name="object_restore_mask",
                expected_shape=expected,
            )
        with self.assertRaisesRegex(TypeError, "dtype bool"):
            barrier.validate_mask_volume(
                np.zeros(expected, dtype=np.uint8),
                name="object_restore_mask",
                expected_shape=expected,
            )

    def test_semantic_labels_add_palm_without_changing_fingers(self):
        hand = np.asarray(((1, 1, 1, 0),), dtype=bool)
        fingers = np.asarray(((0, 2, 5, 0),), dtype=np.uint8)

        result = barrier.semantic_hand_labels(hand, fingers)

        np.testing.assert_array_equal(result, np.asarray(((6, 2, 5, 0),)))

    def test_thickness_map_uses_part_specific_shells(self):
        labels = np.asarray(((0, 1, 2, 5, 6),), dtype=np.uint8)

        result = barrier.thickness_map(
            labels,
            thumb_shell_m=0.020,
            finger_shell_m=0.015,
            palm_shell_m=0.012,
        )

        np.testing.assert_allclose(result, ((0.0, 0.020, 0.015, 0.015, 0.012),))

    def test_barrier_includes_shell_intersection_and_excludes_arm(self):
        robot = np.ones((1, 4), dtype=bool)
        hand = np.asarray(((1, 1, 1, 0),), dtype=bool)
        depth = np.asarray(((0.970, 0.990, 1.010, 1.100),), dtype=np.float32)
        surface = np.ones((1, 4), dtype=np.float32)
        shell = np.asarray(((0.020, 0.020, 0.0, 0.020),), dtype=np.float32)

        result, support = barrier.compute_visual_barrier(
            robot_mask=robot,
            hand_mask=hand,
            robot_depth=depth,
            object_mask=np.ones((1, 4), dtype=bool),
            object_surface_depth=surface,
            shell_m=shell,
        )

        np.testing.assert_array_equal(result, ((False, True, True, False),))
        np.testing.assert_array_equal(support, ((True, True, True, False),))

    def test_unknown_surface_fails_open(self):
        result, support = barrier.compute_visual_barrier(
            robot_mask=np.ones((1, 2), dtype=bool),
            hand_mask=np.ones((1, 2), dtype=bool),
            robot_depth=np.asarray(((1.0, 1.0),), dtype=np.float32),
            object_mask=np.ones((1, 2), dtype=bool),
            object_surface_depth=np.asarray(((0.0, np.nan),), dtype=np.float32),
            shell_m=np.full((1, 2), 0.02, dtype=np.float32),
        )

        self.assertFalse(result.any())
        self.assertFalse(support.any())

    def test_temporal_eligibility_has_clear_front_veto(self):
        eligible = barrier.temporal_eligibility(
            hand_mask=np.ones((1, 3), dtype=bool),
            robot_depth=np.asarray(((0.90, 0.98, 1.0),), dtype=np.float32),
            object_mask=np.ones((1, 3), dtype=bool),
            object_surface_depth=np.asarray(((1.0, 1.0, 0.0),), dtype=np.float32),
            shell_m=np.asarray(((0.02, 0.02, 0.02),), dtype=np.float32),
            front_slack_m=0.015,
        )

        np.testing.assert_array_equal(eligible, ((False, True, True),))

    def test_temporal_bridge_supports_sixth_palm_label(self):
        raw = np.zeros((3, 1, 1), dtype=bool)
        raw[0, 0, 0] = True
        raw[2, 0, 0] = True
        labels = np.full((3, 1, 1), 6, dtype=np.uint8)

        filtered, diagnostics = contact.bridge_short_occlusion_gaps(
            raw,
            np.ones_like(raw),
            labels,
            max_gap_frames=1,
            motion_radius_px=0,
            label_count=6,
        )

        self.assertTrue(filtered[:, 0, 0].all())
        self.assertEqual(diagnostics["added_per_frame_finger"].shape, (3, 6))
        self.assertEqual(diagnostics["added_per_frame_finger"][1, 5], 1)


if __name__ == "__main__":
    unittest.main()
