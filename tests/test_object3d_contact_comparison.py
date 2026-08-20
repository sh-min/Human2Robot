"""Contract tests for the object-surface 2x2 comparison."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "inpainting"))

import compare_object3d_contact_occlusion as comparison  # noqa: E402


class Object3dComparisonContractTests(unittest.TestCase):
    def _sources(self) -> dict[str, dict[str, object]]:
        common_sources = {
            "processed_demo": "/data/processed",
            "episode_dir": "/data/episode",
            "background": "/data/source.mov",
            "raw_video": "/data/source.mov",
            "hawor_npz": "/data/retarget_input.npz",
            "contact_dir": "/data/mh_contact",
            "aux_contact_dir": "/data/sh_contact",
            "overlay_dir": "/data/overlay",
            "object_mask": "/data/object_mask.npy",
        }
        common_report = {
            "frames": 2,
            "width": 3,
            "height": 2,
            "fps": 24.0,
            "side": "left",
            "finger_names": ["thumb", "index"],
            "config": {"depth_margin_m": 0.01},
            "aux_frame_offset": 0,
            "aux_side": "left",
            "contact_score_fused": [[0.8, 0.7], [0.9, 0.6]],
            "hidden_fraction": [[0.5, 0.4], [0.6, 0.3]],
            "active_runs": {"thumb": [[0, 1]], "index": []},
        }
        metadata = {"width": 3, "height": 2, "frames": 2, "fps": 24.0}
        modes = {
            "haco_proxy": "haco",
            "scalar_object_depth": "ensemble",
            "dense_surface": "object3d",
            "contact_aligned_surface": "object3d",
        }
        values: dict[str, dict[str, object]] = {}
        for mode, occlusion_mode in modes.items():
            report = copy.deepcopy(common_report)
            report["occlusion_mode"] = occlusion_mode
            report["sources"] = copy.deepcopy(common_sources)
            report["sources"].update(
                {
                    "scene_depth": (
                        "/data/scene_depth.npy"
                        if mode == "scalar_object_depth"
                        else None
                    ),
                    "object_surface_depth": (
                        "/data/object_surface.npy"
                        if mode in {
                            "dense_surface",
                            "contact_aligned_surface",
                        }
                        else None
                    ),
                }
            )
            report["object_surface_3d"] = {
                "alignment": (
                    "contact" if mode == "contact_aligned_surface" else "none"
                )
            }
            report["invariants"] = {
                "object3d_haco_is_selector_only": occlusion_mode == "object3d"
            }
            values[mode] = {
                "report": report,
                "metadata": copy.deepcopy(metadata),
                "mask": np.zeros((2, 2, 3), dtype=bool),
            }
        return values

    def test_equivalent_lineage_and_all_six_pairs_are_accepted(self):
        sources = self._sources()
        comparison._validate_contract(sources)
        sources["haco_proxy"]["mask"][0, 0, 0] = True
        sources["dense_surface"]["mask"][0, 0, 1] = True
        stats = comparison._statistics(sources)
        self.assertEqual(len(stats["comparisons"]), 6)
        self.assertIn(
            "dense_surface_vs_contact_aligned_surface",
            stats["comparisons"],
        )

    def test_mixed_surface_or_contact_lineage_is_rejected(self):
        surface_mismatch = self._sources()
        surface_mismatch["contact_aligned_surface"]["report"]["sources"][
            "object_surface_depth"
        ] = "/data/other_surface.npy"
        with self.assertRaisesRegex(ValueError, "different surfaces"):
            comparison._validate_contract(surface_mismatch)

        contact_mismatch = self._sources()
        contact_mismatch["dense_surface"]["report"]["contact_score_fused"][
            0
        ][0] = 0.1
        with self.assertRaisesRegex(ValueError, "contact_score_fused"):
            comparison._validate_contract(contact_mismatch)


if __name__ == "__main__":
    unittest.main()
