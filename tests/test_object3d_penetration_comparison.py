"""Tests for the Object3D penetration-suppression comparison contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INPAINTING_DIR = REPO_ROOT / "src" / "inpainting"
sys.path.insert(0, str(INPAINTING_DIR))

import compare_object3d_penetration_strategies as comparison  # noqa: E402


class PenetrationComparisonTests(unittest.TestCase):
    def _masks(self):
        baseline = np.zeros((2, 2, 3), dtype=bool)
        baseline[0, 0, 0] = True
        force = baseline.copy()
        force[0, 0, 1] = True
        temporal = baseline.copy()
        temporal[1, 1, 1] = True
        combined = force | temporal
        return {
            "baseline": baseline,
            "surface_force": force,
            "temporal": temporal,
            "force_temporal": combined,
        }

    def test_strategy_lattice_accepts_two_monotone_factors(self):
        comparison._validate_strategy_lattice(self._masks())

    def test_strategy_lattice_rejects_removed_baseline_pixel(self):
        masks = self._masks()
        masks["surface_force"][0, 0, 0] = False
        with self.assertRaisesRegex(ValueError, "not a subset"):
            comparison._validate_strategy_lattice(masks)

    def test_statistics_cover_all_six_pairs(self):
        sources = {
            mode: {"mask": mask}
            for mode, mask in self._masks().items()
        }
        statistics = comparison._statistics(sources)
        self.assertEqual(len(statistics["modes"]), 4)
        self.assertEqual(len(statistics["comparisons"]), 6)
        self.assertEqual(
            statistics["comparisons"]["baseline_vs_surface_force"],
            {"added_pixels": 1, "removed_pixels": 0, "changed_frames": 1},
        )

    def test_old_baseline_report_defaults_to_controls_off(self):
        self.assertEqual(comparison._control({"config": {}}), (False, 0))
        self.assertEqual(
            comparison._control(
                {
                    "config": {
                        "object3d_force_surface": True,
                        "object3d_temporal_max_gap_frames": 2,
                    }
                }
            ),
            (True, 2),
        )


if __name__ == "__main__":
    unittest.main()
