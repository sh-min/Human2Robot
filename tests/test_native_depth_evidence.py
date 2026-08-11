"""Synthetic tests for same-camera per-finger metric depth evidence."""

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

import precompute_native_depth_evidence as evidence  # noqa: E402


def _intrinsics(width: int, height: int) -> evidence.CameraIntrinsics:
    return evidence.CameraIntrinsics(
        fx=40.0,
        fy=40.0,
        cx=width / 2.0,
        cy=height / 2.0,
        width=width,
        height=height,
    )


def _base_arrays(
    *,
    frames: int = 1,
    height: int = 48,
    width: int = 64,
) -> dict[str, np.ndarray]:
    vertices = np.zeros((frames, 778, 3), dtype=np.float32)
    vertices[..., 2] = -1.0
    return {
        "depth": np.full((frames, height, width), np.nan, dtype=np.float32),
        "modal": np.zeros((frames, height, width), dtype=bool),
        "refined": np.zeros((frames, height, width), dtype=bool),
        "model": np.zeros((frames, height, width), dtype=bool),
        "arm": np.zeros((frames, height, width), dtype=bool),
        "vertices": vertices,
        "valid": np.ones(frames, dtype=bool),
        "parts": np.zeros(778, dtype=np.int32),
        "kpts": np.full((frames, 21, 2), np.nan, dtype=np.float32),
        "detected": np.zeros(frames, dtype=bool),
    }


def _config(**overrides) -> evidence.DepthEvidenceConfig:
    values = {
        "support_radius_px": 4,
        "joint_scope": "distal",
        "min_projected_vertices": 4,
        "min_hand_samples": 5,
        "min_object_samples": 5,
        "trim_fraction": 0.0,
        "mad_scale": 3.5,
        "min_depth_m": 0.02,
        "max_depth_m": 5.0,
    }
    values.update(overrides)
    return evidence.DepthEvidenceConfig(**values)


class RobustMetricDepthTests(unittest.TestCase):
    def test_trim_and_mad_reject_outliers(self):
        samples = np.r_[np.full(20, 0.80), [0.05, 4.50, np.nan]]
        depth, valid_count, inlier_count = evidence.robust_metric_depth(
            samples,
            min_samples=10,
            trim_fraction=0.10,
            mad_scale=3.5,
            min_depth_m=0.02,
            max_depth_m=5.0,
        )

        self.assertAlmostEqual(depth, 0.80, places=6)
        self.assertEqual(valid_count, 22)
        self.assertGreaterEqual(inlier_count, 10)

    def test_insufficient_samples_return_nan(self):
        depth, valid_count, inlier_count = evidence.robust_metric_depth(
            np.array([0.7, 0.8, np.nan]),
            min_samples=3,
            trim_fraction=0.0,
            mad_scale=3.5,
            min_depth_m=0.02,
            max_depth_m=5.0,
        )

        self.assertTrue(np.isnan(depth))
        self.assertEqual(valid_count, 2)
        self.assertEqual(inlier_count, 0)


class NativeDepthEvidenceTests(unittest.TestCase):
    def test_distal_scope_excludes_proximal_mcp_depth(self):
        arrays = _base_arrays(height=24, width=48)
        # A long proximal->MCP segment at 0.50 m would dominate the full
        # polyline.  The distal three joints lie entirely on the 0.90 m strip.
        arrays["kpts"][0, [1, 2, 3, 4]] = [
            [5, 12],
            [25, 12],
            [30, 12],
            [35, 12],
        ]
        arrays["detected"][0] = True
        arrays["model"][0, 12, 5:36] = True
        arrays["arm"][0, 12, 5:36] = True
        arrays["depth"][0, 12, 5:25] = 0.50
        arrays["depth"][0, 12, 25:36] = 0.90

        common = dict(
            metric_depth_m=arrays["depth"],
            modal_object_mask=arrays["modal"],
            refined_object_mask=None,
            model_hand_mask=arrays["model"],
            arm_hand_mask=arrays["arm"],
            vertices_camera=arrays["vertices"],
            hawor_valid=arrays["valid"],
            finger_parts=arrays["parts"],
            intrinsics=_intrinsics(48, 24),
            native_kpts_2d=arrays["kpts"],
            native_hand_detected=arrays["detected"],
        )
        full = evidence.estimate_native_depth_evidence(
            **common,
            config=_config(
                support_radius_px=0,
                joint_scope="full",
                min_hand_samples=3,
            ),
        )
        distal = evidence.estimate_native_depth_evidence(
            **common,
            config=_config(
                support_radius_px=0,
                joint_scope="distal",
                min_hand_samples=3,
            ),
        )

        self.assertAlmostEqual(float(full["hand_depth_m"][0, 0]), 0.50, 5)
        self.assertAlmostEqual(float(distal["hand_depth_m"][0, 0]), 0.90, 5)
        self.assertEqual(full["support_point_count"][0, 0], 4)
        self.assertEqual(distal["support_point_count"][0, 0], 3)

    def test_native_joint_support_samples_disjoint_hand_and_refined_object(self):
        arrays = _base_arrays()
        # Thumb polyline in the native RGB grid.
        arrays["kpts"][0, [1, 2, 3, 4]] = [
            [20, 24],
            [24, 24],
            [28, 24],
            [32, 24],
        ]
        arrays["detected"][0] = True

        # model-only pixels at y=21 are deliberately not visible hand because
        # the arm mask does not include them.
        arrays["model"][0, 20:27, 18:35] = True
        arrays["arm"][0, 22:26, 18:35] = True
        visible_hand = arrays["model"] & arrays["arm"]
        arrays["depth"][visible_hand] = 0.80
        arrays["depth"][0, 21, 20:33] = 0.30

        # Modal includes a wrong-depth tail, while refined keeps the true
        # object strip.  Visible-hand overlap is excluded from object samples.
        arrays["modal"][0, 25:30, 18:35] = True
        # Refined deliberately overlaps the visible-hand row y=25.  The
        # preprocessor must remove that overlap from object sampling.
        arrays["refined"][0, 25:29, 18:35] = True
        object_only = arrays["refined"] & ~visible_hand
        arrays["depth"][object_only] = 1.20
        arrays["depth"][0, 29, 18:35] = 2.50

        result = evidence.estimate_native_depth_evidence(
            metric_depth_m=arrays["depth"],
            modal_object_mask=arrays["modal"],
            refined_object_mask=arrays["refined"],
            model_hand_mask=arrays["model"],
            arm_hand_mask=arrays["arm"],
            vertices_camera=arrays["vertices"],
            hawor_valid=arrays["valid"],
            finger_parts=arrays["parts"],
            intrinsics=_intrinsics(64, 48),
            config=_config(),
            native_kpts_2d=arrays["kpts"],
            native_hand_detected=arrays["detected"],
        )

        self.assertAlmostEqual(float(result["hand_depth_m"][0, 0]), 0.80, 5)
        self.assertAlmostEqual(float(result["object_depth_m"][0, 0]), 1.20, 5)
        self.assertEqual(
            result["support_source"][0, 0],
            evidence.SUPPORT_NATIVE_KEYPOINTS,
        )
        self.assertGreater(result["hand_sample_count"][0, 0], 5)
        self.assertGreater(result["object_sample_count"][0, 0], 5)
        self.assertTrue(np.isnan(result["hand_depth_m"][0, 1:]).all())
        self.assertTrue(np.isnan(result["object_depth_m"][0, 1:]).all())

    def test_projected_mano_vertices_are_fallback_only(self):
        arrays = _base_arrays()
        arrays["parts"][:8] = 13  # thumb part
        # Project to a compact cluster around native pixel (28, 24).
        arrays["vertices"][0, :8] = [
            [(28.0 - 32.0) / 40.0, 0.0, 1.0]
        ] * 8
        arrays["model"][0, 21:28, 24:33] = True
        arrays["arm"][0, 21:28, 24:33] = True
        arrays["depth"][0, 21:28, 24:33] = 0.90
        arrays["modal"][0, 28:31, 24:33] = True
        arrays["depth"][0, 28:31, 24:33] = 1.10

        result = evidence.estimate_native_depth_evidence(
            metric_depth_m=arrays["depth"],
            modal_object_mask=arrays["modal"],
            refined_object_mask=None,
            model_hand_mask=arrays["model"],
            arm_hand_mask=arrays["arm"],
            vertices_camera=arrays["vertices"],
            hawor_valid=arrays["valid"],
            finger_parts=arrays["parts"],
            intrinsics=_intrinsics(64, 48),
            config=_config(support_radius_px=6),
            native_kpts_2d=arrays["kpts"],
            native_hand_detected=arrays["detected"],
        )

        self.assertEqual(
            result["support_source"][0, 0],
            evidence.SUPPORT_PROJECTED_VERTICES,
        )
        self.assertAlmostEqual(float(result["hand_depth_m"][0, 0]), 0.90, 5)
        self.assertAlmostEqual(float(result["object_depth_m"][0, 0]), 1.10, 5)

    def test_minimum_count_is_per_finger_and_fails_open(self):
        arrays = _base_arrays()
        arrays["kpts"][0, [1, 2, 3, 4]] = [
            [20, 24],
            [24, 24],
            [28, 24],
            [32, 24],
        ]
        arrays["detected"][0] = True
        arrays["model"][0, 24, 20:22] = True
        arrays["arm"][0, 24, 20:22] = True
        arrays["depth"][0, 24, 20:22] = 0.80

        result = evidence.estimate_native_depth_evidence(
            metric_depth_m=arrays["depth"],
            modal_object_mask=arrays["modal"],
            refined_object_mask=None,
            model_hand_mask=arrays["model"],
            arm_hand_mask=arrays["arm"],
            vertices_camera=arrays["vertices"],
            hawor_valid=arrays["valid"],
            finger_parts=arrays["parts"],
            intrinsics=_intrinsics(64, 48),
            config=_config(min_hand_samples=3),
            native_kpts_2d=arrays["kpts"],
            native_hand_detected=arrays["detected"],
        )

        self.assertEqual(result["hand_sample_count"][0, 0], 2)
        self.assertTrue(np.isnan(result["hand_depth_m"][0, 0]))

    def test_mismatched_camera_grid_is_rejected_without_resize(self):
        arrays = _base_arrays()
        with self.assertRaisesRegex(ValueError, "resizing or cross-view"):
            evidence.estimate_native_depth_evidence(
                metric_depth_m=arrays["depth"],
                modal_object_mask=arrays["modal"][:, :, :-1],
                refined_object_mask=None,
                model_hand_mask=arrays["model"],
                arm_hand_mask=arrays["arm"],
                vertices_camera=arrays["vertices"],
                hawor_valid=arrays["valid"],
                finger_parts=arrays["parts"],
                intrinsics=_intrinsics(64, 48),
                config=_config(),
            )


class NativeDepthEvidenceCliTests(unittest.TestCase):
    def test_cli_writes_npz_and_json_provenance(self):
        arrays = _base_arrays()
        arrays["kpts"][0, [1, 2, 3, 4]] = [
            [20, 24],
            [24, 24],
            [28, 24],
            [32, 24],
        ]
        arrays["detected"][0] = True
        arrays["model"][0, 22:26, 18:35] = True
        arrays["arm"][0, 22:26, 18:35] = True
        arrays["depth"][0, 22:26, 18:35] = 0.80
        arrays["modal"][0, 26:29, 18:35] = True
        arrays["refined"][0, 26:29, 18:35] = True
        arrays["depth"][0, 26:29, 18:35] = 1.20

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name in ("depth", "modal", "refined", "model", "arm", "parts"):
                path = root / f"{name}.npy"
                np.save(path, arrays[name])
                paths[name] = path
            hand_data = root / "hand_data_left.npz"
            np.savez(
                hand_data,
                frame_indices=np.arange(1),
                hand_detected=arrays["detected"],
                kpts_2d=arrays["kpts"],
            )
            hawor = root / "retarget_input.npz"
            np.savez(
                hawor,
                verts_left=arrays["vertices"],
                valid=np.array([[True], [False]]),
                img_focal=np.float32(40.0),
                frame_is_cam_space=np.bool_(True),
            )
            output = root / "native_depth_evidence.npz"

            evidence.main(
                [
                    "--camera",
                    "synthetic_camera",
                    "--side",
                    "left",
                    "--depth",
                    str(paths["depth"]),
                    "--modal_object_mask",
                    str(paths["modal"]),
                    "--refined_object_mask",
                    str(paths["refined"]),
                    "--model_hand_mask",
                    str(paths["model"]),
                    "--arm_hand_mask",
                    str(paths["arm"]),
                    "--hand_data",
                    str(hand_data),
                    "--hawor",
                    str(hawor),
                    "--finger_parts",
                    str(paths["parts"]),
                    "--output",
                    str(output),
                    "--support_radius_px",
                    "4",
                    "--min_hand_samples",
                    "5",
                    "--min_object_samples",
                    "5",
                    "--trim_fraction",
                    "0",
                ]
            )

            report_path = output.with_suffix(".json")
            self.assertTrue(output.is_file())
            self.assertTrue(report_path.is_file())
            with np.load(output) as result:
                self.assertEqual(result["hand_depth_m"].shape, (1, 5))
                self.assertEqual(
                    result["support_source"][0, 0],
                    evidence.SUPPORT_NATIVE_KEYPOINTS,
                )
            report = json.loads(report_path.read_text())
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["camera"], "synthetic_camera")
            self.assertFalse(
                report["coordinate_contract"]["cross_camera_projection"]
            )
            self.assertEqual(
                report["inputs"]["metric_depth_m"]["shape"],
                [1, 48, 64],
            )
            self.assertIn("hand_depth_m", report["outputs"]["arrays"])


if __name__ == "__main__":
    unittest.main()
