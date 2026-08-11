"""Weight-free synthetic checks for Choco SPAR3D-to-MH registration."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "register_spar3d_mesh_pilot.py"
)
SPEC = importlib.util.spec_from_file_location(
    "register_spar3d_mesh_pilot", MODULE_PATH
)
registration = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = registration
SPEC.loader.exec_module(registration)


class Sim3Tests(unittest.TestCase):
    def test_center_maps_to_translation_and_offsets_receive_scale_rotation(self):
        center = np.asarray((2.0, -1.0, 0.5))
        rotation = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        translation = np.asarray((0.1, -0.2, 0.8))
        matrix = registration.make_sim3(
            0.25, rotation, translation, canonical_center=center
        )
        points = np.vstack((center, center + (1.0, 0.0, 0.0)))
        transformed = registration.transform_points(points, matrix)
        np.testing.assert_allclose(transformed[0], translation, atol=1.0e-12)
        np.testing.assert_allclose(
            transformed[1], translation + (0.0, 0.25, 0.0), atol=1.0e-12
        )
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)

    def test_projection_uses_positive_opencv_camera_z(self):
        k = np.asarray(((100.0, 0.0, 50.0), (0.0, 120.0, 40.0), (0.0, 0.0, 1.0)))
        points = np.asarray(((0.0, 0.0, 1.0), (0.1, -0.2, 2.0)))
        projected = registration.project_camera_points(points, k)
        np.testing.assert_allclose(projected, ((50.0, 40.0), (55.0, 28.0)))
        with self.assertRaisesRegex(ValueError, "positive Z"):
            registration.project_camera_points(np.asarray(((0.0, 0.0, -1.0),)), k)

    def test_signed_axis_rotation_group_has_24_proper_unique_members(self):
        rotations = registration.proper_axis_rotations()
        self.assertEqual(len(rotations), 24)
        self.assertTrue(np.allclose(rotations[0], np.eye(3)))
        fingerprints = {tuple(np.asarray(item, dtype=int).ravel()) for item in rotations}
        self.assertEqual(len(fingerprints), 24)
        for item in rotations:
            np.testing.assert_allclose(item.T @ item, np.eye(3))
            self.assertAlmostEqual(np.linalg.det(item), 1.0)


class MetricsAndProvenanceTests(unittest.TestCase):
    def test_identical_masks_and_depth_have_perfect_metrics(self):
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:7, 3:8] = True
        depth = np.zeros(mask.shape, dtype=np.float32)
        depth[mask] = 0.62
        metrics = registration.alignment_metrics(mask, depth, mask, depth)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertAlmostEqual(metrics["median_depth_error_proxy_m"], 0.0)
        self.assertAlmostEqual(
            metrics["overlap_depth_residual_median_proxy_m"], 0.0
        )
        self.assertEqual(
            registration.registration_loss(
                metrics, depth_reference_m=0.62, canonical_view_angle_rad=0.0
            ),
            0.0,
        )

    def test_contract_never_upgrades_proxy_to_metric_or_uses_sh_translation(self):
        contract = registration.registration_contract()
        self.assertFalse(contract["metric_scale_verified"])
        self.assertFalse(contract["collision_ready"])
        self.assertFalse(contract["uses_sh_checker_square_translation_as_metric"])
        self.assertFalse(contract["uses_sh_for_this_registration"])
        joined = " ".join(contract["warnings"])
        self.assertIn("checker-square", joined)
        self.assertIn("approximate", joined)

    def test_failed_or_regressing_local_fit_falls_back(self):
        stage = lambda loss: {"loss": loss, "matrix": np.eye(4)}
        name, selected, reason = registration.select_fit_stage(
            naive=stage(0.8),
            coarse=stage(0.4),
            local=stage(0.1),
            optimizer_success=False,
        )
        self.assertEqual(name, "coarse")
        self.assertEqual(selected["loss"], 0.4)
        self.assertEqual(reason, "local_optimizer_failed")

        name, selected, reason = registration.select_fit_stage(
            naive=stage(0.8),
            coarse=stage(0.4),
            local=stage(0.7),
            optimizer_success=True,
        )
        self.assertEqual(name, "coarse")
        self.assertEqual(selected["loss"], 0.4)
        self.assertEqual(reason, "local_fit_regressed")

    def test_spar_report_is_bound_to_exact_manifest_and_rgba(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgba = root / "input.png"
            rgba.write_bytes(b"rgba")
            manifest_path = root / "manifest.json"
            manifest = {
                "selection": {
                    "object_label": "Choco",
                    "mh_frame_index": 187,
                    "sh_frame_index": 192,
                },
                "outputs": {
                    "spar3d_rgba_crop": {
                        "path": str(rgba),
                        "bytes": rgba.stat().st_size,
                        "sha256": registration.sha256_file(rgba),
                    }
                },
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = {
                "input_manifest": {
                    "path": str(manifest_path),
                    "sha256": registration.sha256_file(manifest_path),
                },
                "input": {
                    "path": str(rgba),
                    "bytes": rgba.stat().st_size,
                    "sha256": registration.sha256_file(rgba),
                },
                "selection": dict(manifest["selection"]),
            }
            registration.validate_spar_bundle_provenance(
                report, manifest_path=manifest_path.resolve(), manifest=manifest
            )
            report["selection"]["mh_frame_index"] = 188
            with self.assertRaisesRegex(
                registration.RegistrationInputError, "selection differs"
            ):
                registration.validate_spar_bundle_provenance(
                    report, manifest_path=manifest_path.resolve(), manifest=manifest
                )

    def test_output_location_rejects_protected_bundle_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            depth = root / "depth"
            spar = root / "spar"
            inputs.mkdir()
            depth.mkdir()
            spar.mkdir()
            arguments = {
                "input_manifest": inputs / "manifest.json",
                "mh_image": inputs / "mh.jpg",
                "mh_mask": inputs / "mask.png",
                "scene_depth": depth / "depth.npy",
                "depth_params": depth / "params.npz",
                "mesh_glb": spar / "mesh.glb",
                "spar_report": spar / "report.json",
            }
            safe = registration.validate_output_location(
                root / "registered", **arguments
            )
            self.assertEqual(safe, (root / "registered").resolve())
            with self.assertRaisesRegex(
                registration.RegistrationInputError, "overlaps protected"
            ):
                registration.validate_output_location(inputs, **arguments)


class PreflightTests(unittest.TestCase):
    @staticmethod
    def _write_fixture(root: Path) -> dict[str, Path]:
        image_path = root / "mh.jpg"
        mask_path = root / "mask.png"
        depth_path = root / "depth.npy"
        params_path = root / "params.npz"
        manifest_path = root / "manifest.json"
        image = np.full((24, 32, 3), 127, dtype=np.uint8)
        mask = np.zeros((24, 32), dtype=np.uint8)
        mask[6:20, 9:24] = 255
        depth = np.full((3, 24, 32), 0.75, dtype=np.float32)
        self_ok = cv2.imwrite(str(image_path), image)
        mask_ok = cv2.imwrite(str(mask_path), mask)
        if not self_ok or not mask_ok:
            raise RuntimeError("failed to write synthetic fixture images")
        np.save(depth_path, depth)
        np.savez(
            params_path,
            raw_scale=np.asarray((0.9, 1.0, 1.1), dtype=np.float32),
            scale=np.asarray((0.95, 1.0, 1.05), dtype=np.float32),
            valid_frames=np.asarray((1, 1, 1), dtype=np.uint8),
            encoder="vits",
            checkpoint="synthetic-weight-free-checkpoint",
        )
        manifest = {
            "schema_version": 1,
            "kind": "mesh_sota_pilot_input_bundle",
            "selection": {
                "object_label": "Choco",
                "mh_frame_index": 1,
                "sh_frame_index": 6,
                "mh_role": "primary/final",
                "sh_role": "auxiliary/evidence",
            },
            "calibration": {
                "checkerboard": {
                    "length_unit": "checker_square",
                    "metric_scale_verified": False,
                },
                "intrinsics_by_view": {
                    "MH": {
                        "calibration_camera": "camera_1",
                        "camera_matrix": [
                            [100.0, 0.0, 16.0],
                            [0.0, 100.0, 12.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "distortion_k1_k2_p1_p2_k3": [0, 0, 0, 0, 0],
                    }
                },
                "relative_extrinsics": {
                    "translation_unit": "checker_square",
                    "T_camera2_from_camera1": np.eye(4).tolist(),
                },
            },
            "outputs": {
                "mh_image": {
                    "path": str(image_path.resolve()),
                    "bytes": image_path.stat().st_size,
                    "sha256": registration.sha256_file(image_path),
                },
                "modal_mask": {
                    "path": str(mask_path.resolve()),
                    "bytes": mask_path.stat().st_size,
                    "sha256": registration.sha256_file(mask_path),
                },
            },
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return {
            "input_manifest": manifest_path,
            "mh_image": image_path,
            "mh_mask": mask_path,
            "scene_depth": depth_path,
            "depth_params": params_path,
            "mesh_glb": root / "mesh.glb",
            "spar_report": root / "spar-report.json",
        }

    def test_preflight_succeeds_without_mesh_and_reports_waiting(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._write_fixture(Path(temporary))
            report = registration.preflight_job(**paths)
            self.assertEqual(report["status"], "waiting_for_spar3d_mesh")
            self.assertFalse(report["input_checks"]["mesh_ready"])
            self.assertFalse(report["stereo_scale_guard"]["translation_used"])
            self.assertEqual(
                report["stereo_scale_guard"]["manifest_translation_unit"],
                "checker_square",
            )
            self.assertFalse(report["metric_scale_verified"])

    def test_preflight_detects_half_published_spar_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._write_fixture(root)
            paths["mesh_glb"].write_bytes(b"synthetic")
            report = registration.preflight_job(**paths)
            self.assertEqual(report["status"], "blocked_incomplete_spar3d_bundle")


if __name__ == "__main__":
    unittest.main()
