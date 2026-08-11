"""Tests for estimated object-surface 3-D contact occlusion."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INPAINTING_DIR = REPO_ROOT / "src" / "inpainting"
sys.path.insert(0, str(INPAINTING_DIR))

import build_object_surface_model as surface_model  # noqa: E402
import composite_rb5_contact_occlusion as occlusion  # noqa: E402


class ObjectSurfaceModelTests(unittest.TestCase):
    def test_planar_surface_is_kept_and_background_is_zero(self):
        height, width = 24, 32
        x = np.arange(width, dtype=np.float32)[None]
        depth = np.broadcast_to(0.70 + 0.001 * x, (height, width)).copy()
        mask = np.zeros((height, width), dtype=bool)
        mask[4:20, 6:27] = True
        # A gross depth error inside the object should not survive the robust
        # visible-surface band.
        depth[10, 12] = 2.5
        config = surface_model.SurfaceModelConfig(
            erode_px=1,
            bilateral_diameter=3,
            minimum_samples=12,
            point_count=32,
        )

        surface, stats = surface_model.build_surface_frame(
            depth,
            mask,
            output_shape=(height, width),
            config=config,
        )

        self.assertTrue(stats["valid"])
        self.assertTrue(np.all(surface[~mask] == 0.0))
        self.assertEqual(surface[10, 12], 0.0)
        self.assertGreater(int((surface > 0.0).sum()), 250)
        self.assertAlmostEqual(float(np.median(surface[surface > 0])), 0.716, places=2)

    def test_missing_object_depth_fails_open(self):
        surface, stats = surface_model.build_surface_frame(
            np.zeros((8, 8), dtype=np.float32),
            np.ones((8, 8), dtype=bool),
            output_shape=(8, 8),
            config=surface_model.SurfaceModelConfig(minimum_samples=4),
        )
        self.assertFalse(stats["valid"])
        self.assertFalse(surface.any())

    def test_background_depth_cannot_bleed_across_modal_boundary(self):
        depth = np.full((16, 16), 0.74, dtype=np.float32)
        mask = np.zeros_like(depth, dtype=bool)
        mask[4:12, 4:12] = True
        depth[mask] = 0.80
        surface, stats = surface_model.build_surface_frame(
            depth,
            mask,
            output_shape=depth.shape,
            config=surface_model.SurfaceModelConfig(
                erode_px=0,
                bilateral_diameter=5,
                bilateral_sigma_depth_m=0.20,
                minimum_samples=8,
            ),
        )
        self.assertTrue(stats["valid"])
        self.assertGreater(float(surface[4, 4]), 0.795)
        self.assertEqual(float(surface[3, 4]), 0.0)

    def test_point_cloud_backprojection_preserves_camera_z(self):
        depth = np.zeros((3, 3), dtype=np.float32)
        depth[1, 1] = 2.0
        points, count = surface_model.sample_surface_points(
            depth,
            focal_px=100.0,
            principal_point=(1.0, 1.0),
            point_count=4,
        )
        self.assertEqual(count, 1)
        np.testing.assert_allclose(points[0], [0.0, 0.0, 2.0])
        self.assertTrue(np.isnan(points[1:]).all())


class ObjectSurfaceContactTests(unittest.TestCase):
    def test_default_contact_registration_never_requires_clamping(self):
        config = occlusion.OcclusionConfig()
        config.validate()
        self.assertEqual(
            config.object_surface_contact_max_shift_m,
            config.object_surface_contact_consistency_m,
        )

    def test_contact_alignment_uses_local_robust_depth_and_bounded_shift(self):
        surface = np.full((5, 6), 0.80, dtype=np.float32)
        surface[2, 2] = 1.8
        mask = np.ones_like(surface, dtype=bool)
        support = np.ones_like(surface, dtype=bool)
        result = occlusion.object_surface_contact_alignment(
            surface,
            mask,
            support,
            contact_depth_m=0.90,
            alignment="contact",
            min_samples=8,
            max_shift_m=0.06,
            consistency_m=0.12,
        )
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["local_surface_depth_m"], 0.80, places=5)
        self.assertAlmostEqual(result["contact_residual_m"], 0.10, places=5)
        self.assertAlmostEqual(result["applied_shift_m"], 0.06, places=5)

    def test_inconsistent_or_sparse_contact_fails_open(self):
        surface = np.full((4, 4), 0.80, dtype=np.float32)
        mask = np.ones_like(surface, dtype=bool)
        sparse = np.zeros_like(surface, dtype=bool)
        sparse[0, :2] = True
        sparse_result = occlusion.object_surface_contact_alignment(
            surface,
            mask,
            sparse,
            contact_depth_m=0.80,
            alignment="contact",
            min_samples=4,
            max_shift_m=0.05,
            consistency_m=0.10,
        )
        inconsistent = occlusion.object_surface_contact_alignment(
            surface,
            mask,
            np.ones_like(mask),
            contact_depth_m=1.10,
            alignment="contact",
            min_samples=4,
            max_shift_m=0.05,
            consistency_m=0.10,
        )
        self.assertFalse(sparse_result["valid"])
        self.assertFalse(inconsistent["valid"])
        self.assertTrue(np.isnan(inconsistent["applied_shift_m"]))

    def test_dense_surface_gate_is_pixelwise_and_finger_only(self):
        shape = (2, 4)
        surface = np.array(
            [[0.80, 0.85, 0.90, 0.95], [0.80, 0.85, 0.90, 0.95]],
            dtype=np.float32,
        )
        robot_depth = np.array(
            [[0.82, 0.86, 0.89, 1.00], [0.90, 0.90, 1.00, 1.00]],
            dtype=np.float32,
        )
        finger = np.array(
            [[True, True, True, True], [False, False, True, True]],
            dtype=bool,
        )
        result = occlusion.compute_occluded_fingers_surface(
            robot_mask=np.ones(shape, dtype=bool),
            finger_mask=finger,
            robot_depth=robot_depth,
            occluder_mask=np.ones(shape, dtype=bool),
            contact_support_mask=np.ones(shape, dtype=bool),
            object_surface_depth=surface,
            surface_shift_m=0.0,
            object_depth_margin_m=0.01,
        )
        expected = np.array(
            [[True, False, False, True], [False, False, True, True]],
            dtype=bool,
        )
        np.testing.assert_array_equal(result, expected)
        self.assertFalse(np.any(result & ~finger))

    def test_nan_alignment_shift_returns_empty_mask(self):
        result = occlusion.compute_occluded_fingers_surface(
            robot_mask=np.ones((2, 2), dtype=bool),
            finger_mask=np.ones((2, 2), dtype=bool),
            robot_depth=np.ones((2, 2), dtype=np.float32),
            occluder_mask=np.ones((2, 2), dtype=bool),
            contact_support_mask=np.ones((2, 2), dtype=bool),
            object_surface_depth=np.full((2, 2), 0.8, dtype=np.float32),
            surface_shift_m=float("nan"),
        )
        self.assertFalse(result.any())

    def test_object_surface_can_recover_a_contact_proxy_miss(self):
        common = {
            "robot_mask": np.ones((1, 1), dtype=bool),
            "finger_mask": np.ones((1, 1), dtype=bool),
            "robot_depth": np.array([[0.90]], dtype=np.float32),
            "occluder_mask": np.ones((1, 1), dtype=bool),
            "contact_support_mask": np.ones((1, 1), dtype=bool),
        }
        proxy = occlusion.compute_occluded_fingers(
            **common,
            contact_depth_m=1.0,
            contact_depth_tolerance_m=0.012,
        )
        surface = occlusion.compute_occluded_fingers_surface(
            **common,
            object_surface_depth=np.array([[0.80]], dtype=np.float32),
            object_depth_margin_m=0.01,
        )
        self.assertFalse(proxy.item())
        self.assertTrue(surface.item())

    def test_temporal_eligibility_vetoes_clear_front_but_allows_surface_hole(self):
        finger = np.ones((1, 4), dtype=bool)
        robot_depth = np.array([[0.70, 0.79, 0.82, np.nan]], np.float32)
        surface = np.array([[0.80, 0.80, 0.80, 0.80]], np.float32)
        surface[0, 1] = 0.0

        eligible = occlusion.object_surface_temporal_eligibility(
            finger_mask=finger,
            robot_depth=robot_depth,
            occluder_mask=np.ones_like(finger),
            object_surface_depth=surface,
            front_slack_m=0.015,
        )

        np.testing.assert_array_equal(
            eligible,
            [[False, True, True, False]],
        )

    def test_bidirectional_temporal_filter_bridges_only_same_finger_gap(self):
        raw = np.zeros((3, 3, 5), dtype=bool)
        labels = np.ones(raw.shape, dtype=np.uint8)
        eligible = np.ones_like(raw)
        raw[0, 1, 1] = True
        raw[2, 1, 3] = True

        filtered, diagnostics = occlusion.bridge_short_occlusion_gaps(
            raw,
            eligible,
            labels,
            max_gap_frames=1,
            motion_radius_px=1,
        )

        self.assertTrue(filtered[1, 1, 2])
        self.assertTrue(np.all(filtered[raw]))
        self.assertEqual(diagnostics["added_pixels"], 1)
        self.assertEqual(diagnostics["added_frames"], 1)

        different_finger = labels.copy()
        different_finger[2] = 2
        rejected, _ = occlusion.bridge_short_occlusion_gaps(
            raw,
            eligible,
            different_finger,
            max_gap_frames=1,
            motion_radius_px=1,
        )
        self.assertFalse(rejected[1].any())

    def test_temporal_filter_rejects_open_long_and_ineligible_gaps(self):
        raw = np.zeros((6, 1, 3), dtype=bool)
        labels = np.ones(raw.shape, dtype=np.uint8)
        eligible = np.ones_like(raw)
        raw[0, 0, 1] = True
        raw[4, 0, 1] = True
        raw[5, 0, 1] = True
        eligible[2, 0, 1] = False

        filtered, diagnostics = occlusion.bridge_short_occlusion_gaps(
            raw,
            eligible,
            labels,
            max_gap_frames=2,
            motion_radius_px=0,
        )

        # Frames 1..3 form a three-frame gap, longer than the configured two.
        self.assertFalse(filtered[1:4].any())
        # No future frame exists after frame 5, so an open edge cannot grow.
        self.assertTrue(filtered[5, 0, 1])
        self.assertEqual(diagnostics["added_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
