"""Tests for nominal-mesh volume comparison invariants."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INPAINTING_DIR = REPO_ROOT / "src" / "inpainting"
sys.path.insert(0, str(INPAINTING_DIR))

import compare_object_mesh_volume as comparison  # noqa: E402


class ObjectMeshVolumeComparisonTests(unittest.TestCase):
    @staticmethod
    def _write_mesh_volume(
        root: Path,
        *,
        front: np.ndarray | None = None,
        back: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        pose_valid: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        shape = (3, 2, 3)
        if mask is None:
            mask = np.zeros(shape, dtype=bool)
            mask[0, 0, 0] = True
            mask[1, 1, 1] = True
        if front is None:
            front = np.zeros(shape, dtype=np.float16)
            front[mask] = np.float16(0.8)
        if back is None:
            back = np.zeros(shape, dtype=np.float16)
            back[mask] = np.float16(0.9)
        if pose_valid is None:
            pose_valid = np.array([True, True, False], dtype=bool)
        pose = np.full((shape[0], 4, 4), np.nan, dtype=np.float32)
        pose[pose_valid] = np.eye(4, dtype=np.float32)
        confidence = np.zeros(shape[0], dtype=np.float32)
        confidence[pose_valid] = np.float32(0.8)
        np.save(root / "object_mesh_front_depth.npy", front)
        np.save(root / "object_mesh_back_depth.npy", back)
        np.save(root / "object_mesh_mask.npy", mask)
        np.save(root / "object_pose_cam.npy", pose)
        np.save(root / "pose_valid.npy", pose_valid)
        np.save(root / "pose_confidence.npy", confidence)
        (root / "report.json").write_text(
            json.dumps(
                {
                    "method": comparison.MESH_METHOD,
                    "representation": (
                        "fitted_watertight_nominal_mesh_front_back_camera_z"
                    ),
                    "metric_collision_guarantee": False,
                    "rear_surface_measured": False,
                    "pose_state_modified": False,
                    "frames": shape[0],
                    "counts": {
                        "valid_pose_frames": int(pose_valid.sum()),
                        "mesh_pixels": int(mask.sum()),
                    },
                    "invariants": {
                        "canonical_meshes_watertight": True,
                        "invalid_pose_frames_have_empty_geometry": True,
                        "valid_mesh_pixels_have_ordered_front_back": True,
                        "mesh_mask_equals_positive_front_and_back": True,
                        "robot_trajectory_arrays_unchanged": True,
                    },
                }
            )
        )
        return front, back, mask, pose_valid

    def test_mesh_volume_contract_accepts_ordered_front_and_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_mesh_volume(root)

            result = comparison.validate_mesh_volume_arrays(
                root,
                expected_shape=(3, 2, 3),
            )

        self.assertEqual(result["pose_valid_frames"], 2)
        self.assertEqual(result["pose_invalid_frames"], 1)
        self.assertEqual(result["valid_front_back_pixels"], 2)
        self.assertEqual(result["inverted_front_back_pixels"], 0)

    def test_mesh_volume_contract_rejects_back_in_front(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            front, back, mask, pose_valid = self._write_mesh_volume(root)
            back = back.copy()
            back[0, 0, 0] = np.float16(0.6)
            self._write_mesh_volume(
                root,
                front=front,
                back=back,
                mask=mask,
                pose_valid=pose_valid,
            )

            with self.assertRaisesRegex(ValueError, "back camera-Z"):
                comparison.validate_mesh_volume_arrays(root)

    def test_mesh_volume_contract_rejects_invalid_pose_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            front, back, mask, pose_valid = self._write_mesh_volume(root)
            mask = mask.copy()
            front = front.copy()
            back = back.copy()
            mask[2, 0, 2] = True
            front[2, 0, 2] = np.float16(0.7)
            back[2, 0, 2] = np.float16(0.95)
            self._write_mesh_volume(
                root,
                front=front,
                back=back,
                mask=mask,
                pose_valid=pose_valid,
            )

            with self.assertRaisesRegex(ValueError, "invalid object poses"):
                comparison.validate_mesh_volume_arrays(root)

    @staticmethod
    def _progressive_masks() -> dict[str, np.ndarray]:
        front = np.zeros((2, 2, 3), dtype=bool)
        front[0, 0, 0] = True
        shell = front.copy()
        shell[0, 0, 1] = True
        temporal = shell.copy()
        temporal[1, 1, 2] = True
        return {
            "mesh_front": front,
            "mesh_volume_shell": shell,
            "mesh_volume_shell_temporal": temporal,
        }

    def test_progressive_mesh_lattice_accepts_monotone_masks(self):
        comparison.validate_progressive_mesh_masks(self._progressive_masks())

    def test_progressive_mesh_lattice_rejects_removed_pixel(self):
        masks = self._progressive_masks()
        masks["mesh_volume_shell"][0, 0, 0] = False
        with self.assertRaisesRegex(ValueError, "not a subset"):
            comparison.validate_progressive_mesh_masks(masks)

    def test_statistics_cover_all_six_pairs(self):
        masks = self._progressive_masks()
        haco = np.zeros_like(masks["mesh_front"])
        haco[1, 0, 0] = True
        sources = {
            "haco_2p5d": {"mask": haco},
            **{name: {"mask": mask} for name, mask in masks.items()},
        }

        result = comparison._statistics(sources)

        self.assertEqual(len(result["comparisons"]), 6)
        self.assertEqual(
            result["comparisons"]["mesh_front_vs_mesh_volume_shell"],
            {"added_pixels": 1, "removed_pixels": 0, "changed_frames": 1},
        )

    def test_controlled_inputs_reject_mixed_panel_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_dir = root / "mesh"
            common = {
                "background": str(root / "background.mp4"),
                "raw_video": str(root / "raw.mov"),
                "overlay_dir": str(root / "overlay"),
                "object_support_mask": str(root / "support.npy"),
                "object_restore_mask": str(root / "restore.npy"),
                "baseline_mask": str(root / "baseline.npy"),
                "mesh_dir": str(mesh_dir),
            }
            haco = {
                **common,
                "object_mask": common["object_support_mask"],
            }
            sources = {
                "haco_2p5d": {"report": {"sources": haco}},
                **{
                    mode: {"report": {"sources": dict(common)}}
                    for mode in (
                        "mesh_front",
                        "mesh_volume_shell",
                        "mesh_volume_shell_temporal",
                    )
                },
            }
            result = comparison.validate_controlled_inputs(
                sources,
                mesh_dir=mesh_dir,
            )
            self.assertEqual(result["baseline_mask"], root / "baseline.npy")

            sources["mesh_volume_shell"]["report"]["sources"][
                "baseline_mask"
            ] = str(root / "other.npy")
            with self.assertRaisesRegex(ValueError, "baseline_mask differs"):
                comparison.validate_controlled_inputs(
                    sources,
                    mesh_dir=mesh_dir,
                )

    def test_builder_provenance_binds_exact_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                key: root / filename
                for key, filename in {
                    "mapping": "mapping.json",
                    "labels_json": "labels.json",
                    "amodal_mask": "amodal.npy",
                    "completed_front_depth": "front.npy",
                    "wrist_npz": "wrist.npz",
                    "debug_video": "raw.mp4",
                }.items()
            }
            for path in paths.values():
                path.write_bytes(b"test")
            report = {
                "sources": {
                    key: str(path)
                    for key, path in paths.items()
                }
            }

            result = comparison.validate_builder_provenance(
                report,
                mapping=paths["mapping"],
                wrist_npz=paths["wrist_npz"],
                amodal_mask=paths["amodal_mask"],
                completed_front_depth=paths["completed_front_depth"],
                debug_video=paths["debug_video"],
            )
            self.assertEqual(result["labels_json"], paths["labels_json"])

            with self.assertRaisesRegex(ValueError, "mapping differs"):
                comparison.validate_builder_provenance(
                    report,
                    mapping=root / "other.json",
                    wrist_npz=paths["wrist_npz"],
                    amodal_mask=paths["amodal_mask"],
                    completed_front_depth=paths["completed_front_depth"],
                    debug_video=paths["debug_video"],
                )


if __name__ == "__main__":
    unittest.main()
