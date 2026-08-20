"""Tests for whole-XHand barrier comparison invariants and ROI tracking."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INPAINTING_DIR = REPO_ROOT / "src" / "inpainting"
sys.path.insert(0, str(INPAINTING_DIR))

import compare_xhand_object_barriers as comparison  # noqa: E402


class XHandBarrierComparisonTests(unittest.TestCase):
    @staticmethod
    def _masks() -> dict[str, np.ndarray]:
        finger = np.zeros((2, 2, 3), dtype=bool)
        finger[0, 0, 0] = True
        whole = finger.copy()
        whole[0, 0, 1] = True
        shell = whole.copy()
        shell[1, 1, 1] = True
        temporal = shell.copy()
        temporal[1, 1, 2] = True
        return {
            "finger_best": finger,
            "whole_hand_zero": whole,
            "whole_hand_shell": shell,
            "whole_hand_shell_temporal": temporal,
        }

    def test_lattice_accepts_progressive_barriers(self):
        comparison.validate_lattice(self._masks())

    def test_lattice_rejects_removed_whole_hand_pixel(self):
        masks = self._masks()
        masks["whole_hand_shell"][0, 0, 1] = False
        with self.assertRaisesRegex(ValueError, "not a subset"):
            comparison.validate_lattice(masks)

    def test_statistics_cover_all_six_pairs(self):
        sources = {
            mode: {"mask": mask}
            for mode, mask in self._masks().items()
        }

        result = comparison._statistics(sources)

        self.assertEqual(len(result["comparisons"]), 6)
        self.assertEqual(
            result["comparisons"]["finger_best_vs_whole_hand_zero"],
            {"added_pixels": 1, "removed_pixels": 0, "changed_frames": 1},
        )

    def test_dynamic_roi_interpolates_missing_object_frames(self):
        mask = np.zeros((5, 20, 30), dtype=bool)
        mask[0, 4:8, 3:7] = True
        mask[4, 12:16, 23:27] = True

        centres = comparison.dynamic_roi_centers(
            mask,
            crop_width=10,
            crop_height=8,
            smooth_window=1,
        )

        np.testing.assert_allclose(centres[0], (5.0, 5.5))
        np.testing.assert_allclose(centres[2], (14.5, 9.5))
        np.testing.assert_allclose(centres[4], (24.5, 13.5))
        self.assertTrue(np.isfinite(centres).all())


if __name__ == "__main__":
    unittest.main()
