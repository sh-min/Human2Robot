"""Unit tests for the HaCo-free sensor-depth occlusion primitive."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INPAINTING_DIR = REPO_ROOT / "src" / "inpainting"
sys.path.insert(0, str(INPAINTING_DIR))

import composite_rb5_depth_occlusion as depth_occlusion  # noqa: E402


class DepthOnlyOcclusionTests(unittest.TestCase):
    def test_missing_object_depth_fails_open(self):
        shape = (3, 4)
        hidden = depth_occlusion.compute_depth_only_occlusion(
            robot_mask=np.ones(shape, dtype=bool),
            finger_mask=np.ones(shape, dtype=bool),
            robot_depth=np.full(shape, 0.9, dtype=np.float32),
            depth_object_mask=np.ones(shape, dtype=bool),
            object_depth_m=np.nan,
            depth_margin_m=0.03,
        )
        self.assertFalse(hidden.any())

    def test_only_finger_object_overlap_behind_margin_is_hidden(self):
        robot = np.array([[True, True, True, False]], dtype=bool)
        fingers = np.array([[True, True, False, True]], dtype=bool)
        object_mask = np.array([[True, False, True, True]], dtype=bool)
        robot_depth = np.array(
            [[0.84, 0.90, 0.90, 0.90]],
            dtype=np.float32,
        )

        hidden = depth_occlusion.compute_depth_only_occlusion(
            robot_mask=robot,
            finger_mask=fingers,
            robot_depth=robot_depth,
            depth_object_mask=object_mask,
            object_depth_m=0.80,
            depth_margin_m=0.03,
        )

        np.testing.assert_array_equal(
            hidden,
            [[True, False, False, False]],
        )
        self.assertFalse(np.any(hidden & ~robot))
        self.assertFalse(np.any(hidden & ~fingers))
        self.assertFalse(np.any(hidden & ~object_mask))

    def test_depth_equal_to_margin_is_not_confidently_behind(self):
        hidden = depth_occlusion.compute_depth_only_occlusion(
            robot_mask=np.ones((1, 2), dtype=bool),
            finger_mask=np.ones((1, 2), dtype=bool),
            robot_depth=np.array([[0.83, 0.831]], dtype=np.float32),
            depth_object_mask=np.ones((1, 2), dtype=bool),
            object_depth_m=0.80,
            depth_margin_m=0.03,
        )
        np.testing.assert_array_equal(hidden, [[False, True]])


if __name__ == "__main__":
    unittest.main()
