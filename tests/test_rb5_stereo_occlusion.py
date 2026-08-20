"""Focused tests for stereo, depth, and HaCo occlusion decisions."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INPAINTING_DIR = REPO_ROOT / "src" / "inpainting"
sys.path.insert(0, str(INPAINTING_DIR))

import composite_rb5_stereo_occlusion as stereo  # noqa: E402


class Camera1TemporalAlignmentTests(unittest.TestCase):
    def test_negative_offset_uses_previous_camera1_frame_and_fails_open(self):
        camera1 = np.array(
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
            dtype=np.float32,
        )

        aligned = stereo.align_camera1_to_camera2(
            camera1,
            -1,
            fill_value=np.nan,
        )

        np.testing.assert_allclose(
            aligned,
            [[np.nan, np.nan], [10.0, 11.0], [20.0, 21.0]],
            equal_nan=True,
        )

    def test_positive_offset_uses_next_camera1_frame_and_zero_fills_tail(self):
        camera1 = np.array(
            [[1, 2], [3, 4], [5, 6]],
            dtype=np.int32,
        )

        aligned = stereo.align_camera1_to_camera2(
            camera1,
            1,
            fill_value=0,
        )

        np.testing.assert_array_equal(aligned, [[3, 4], [5, 6], [0, 0]])
        self.assertEqual(aligned.dtype, camera1.dtype)

    def test_offset_magnitude_must_be_smaller_than_sequence_length(self):
        values = np.ones((3, 5), dtype=np.float32)

        for offset in (-3, 3):
            with self.subTest(offset=offset):
                with self.assertRaisesRegex(ValueError, "offset"):
                    stereo.align_camera1_to_camera2(
                        values,
                        offset,
                        fill_value=np.nan,
                    )

    def test_cli_exposes_zero_default_and_explicit_camera1_offset(self):
        required = [
            "composite_rb5_stereo_occlusion.py",
            "--camera1_rgb_dir",
            "c1_rgb",
            "--camera2_rgb_dir",
            "c2_rgb",
            "--camera1_hawor",
            "c1_hawor.npz",
            "--camera2_hawor",
            "c2_hawor.npz",
            "--contact_dir",
            "c2_contact",
            "--background",
            "background.mp4",
            "--overlay_dir",
            "overlay",
            "--object_mask",
            "object.npy",
            "--out_dir",
            "out",
        ]
        with mock.patch.object(sys, "argv", required):
            args = stereo._parse_args()
            self.assertEqual(args.camera1_frame_offset, 0)
            self.assertIsNone(args.object_restore_mask)
        with mock.patch.object(
            sys,
            "argv",
            required
            + [
                "--camera1_frame_offset",
                "-1",
                "--object_restore_mask",
                "object_clean.npy",
            ],
        ):
            args = stereo._parse_args()
            self.assertEqual(args.camera1_frame_offset, -1)
            self.assertEqual(args.object_restore_mask, Path("object_clean.npy"))


class ObjectRestoreMaskTests(unittest.TestCase):
    def test_clean_restore_mask_keeps_modal_geometry_independent(self):
        modal = np.array([[[True, True, False]]], dtype=bool)
        clean = np.array([[[True, False, False]]], dtype=bool)
        stereo.validate_object_restore_mask(modal, clean, frame_count=1)

        occluded = stereo.compute_visibility_haco_occlusion_mask(
            robot_mask=np.ones((1, 3), dtype=bool),
            finger_labels=np.ones((1, 3), dtype=np.uint8),
            object_mask=modal[0],
            visibility_haco_active_fingers=np.array(
                [True, False, False, False, False],
                dtype=bool,
            ),
        )
        background = np.zeros((1, 3, 3), dtype=np.uint8)
        raw = np.full((1, 3, 3), 200, dtype=np.uint8)
        restored = stereo.restore_camera2_object_pixels(
            background,
            raw,
            clean[0],
        )

        np.testing.assert_array_equal(occluded, [[True, True, False]])
        np.testing.assert_array_equal(restored[0, 0], [200, 200, 200])
        np.testing.assert_array_equal(restored[0, 1], [0, 0, 0])

    def test_restore_mask_must_be_modal_subset(self):
        modal = np.array([[[True, False]]], dtype=bool)
        restore = np.array([[[True, True]]], dtype=bool)

        with self.assertRaisesRegex(ValueError, "subset"):
            stereo.validate_object_restore_mask(modal, restore, frame_count=1)

    def test_restore_mask_must_match_shape_frames_and_dtype(self):
        modal = np.ones((2, 2, 3), dtype=bool)
        cases = (
            (
                np.ones((1, 2, 3), dtype=bool),
                "frame-aligned",
            ),
            (
                np.ones((2, 3, 2), dtype=bool),
                "shape must exactly match",
            ),
            (
                np.ones((2, 2, 3), dtype=np.uint8),
                "dtype must match",
            ),
        )
        for restore, message in cases:
            with self.subTest(shape=restore.shape, dtype=restore.dtype):
                with self.assertRaisesRegex(ValueError, message):
                    stereo.validate_object_restore_mask(
                        modal,
                        restore,
                        frame_count=2,
                    )

    def test_modal_mask_as_default_preserves_restore_behavior(self):
        modal = np.array([[[False, True]]], dtype=bool)
        stereo.validate_object_restore_mask(modal, modal, frame_count=1)
        background = np.zeros((1, 2, 3), dtype=np.uint8)
        raw = np.full((1, 2, 3), 127, dtype=np.uint8)

        restored = stereo.restore_camera2_object_pixels(
            background,
            raw,
            modal[0],
        )

        np.testing.assert_array_equal(restored[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(restored[0, 1], [127, 127, 127])


class StereoEvidenceTests(unittest.TestCase):
    def test_dual_camera_haco_uses_per_finger_maximum(self):
        camera1 = np.array(
            [[0.2, 0.9, np.nan, 0.3, 0.4]],
            dtype=np.float32,
        )
        camera2 = np.array(
            [[0.8, 0.1, 0.7, np.nan, 0.4]],
            dtype=np.float32,
        )

        fused = stereo.fuse_haco_scores(camera1, camera2)

        np.testing.assert_allclose(fused, [[0.8, 0.9, 0.7, 0.3, 0.4]])

    def test_camera2_only_haco_preserves_legacy_scores(self):
        camera2 = np.array(
            [[0.0, 0.25, 0.5, 0.75, 1.0]],
            dtype=np.float32,
        )

        fused = stereo.fuse_haco_scores(None, camera2)

        np.testing.assert_array_equal(fused, camera2)

    def test_dual_depth_is_camera2_primary_and_tracks_sources(self):
        camera1 = np.array(
            [0.84, 0.90, 0.80, np.nan, np.nan],
            dtype=np.float32,
        )
        camera2 = np.array(
            [0.83, 0.70, np.nan, 0.75, np.nan],
            dtype=np.float32,
        )

        fused, source = stereo.fuse_object_depth_tracks(
            camera1,
            camera2,
            agreement_tolerance_m=0.02,
        )

        # Agreeing views use the farther value; disagreement rejects C1 and
        # keeps C2.  C1-only and fully missing frames fail open.
        np.testing.assert_allclose(fused[[0, 1, 3]], [0.84, 0.70, 0.75])
        self.assertTrue(np.isnan(fused[2]))
        self.assertTrue(np.isnan(fused[4]))
        np.testing.assert_array_equal(
            source,
            [
                stereo.DEPTH_SOURCE_BOTH,
                stereo.DEPTH_SOURCE_CAMERA2_REJECTED_CAMERA1,
                stereo.DEPTH_SOURCE_CAMERA1_UNSUPPORTED,
                stereo.DEPTH_SOURCE_CAMERA2,
                stereo.DEPTH_SOURCE_NONE,
            ],
        )

    def test_depth_agreement_tolerance_is_inclusive(self):
        camera1 = np.array([0.82, 0.8201], dtype=np.float32)
        camera2 = np.array([0.80, 0.80], dtype=np.float32)

        fused, source = stereo.fuse_object_depth_tracks(
            camera1,
            camera2,
            agreement_tolerance_m=0.02,
        )

        np.testing.assert_allclose(fused, [0.82, 0.80], atol=1e-6)
        np.testing.assert_array_equal(
            source,
            [
                stereo.DEPTH_SOURCE_BOTH,
                stereo.DEPTH_SOURCE_CAMERA2_REJECTED_CAMERA1,
            ],
        )

    def test_invalid_depth_agreement_tolerance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "agreement_tolerance_m"):
            stereo.fuse_object_depth_tracks(
                np.array([0.8], dtype=np.float32),
                np.array([0.8], dtype=np.float32),
                agreement_tolerance_m=-0.01,
            )

    def test_camera2_only_depth_preserves_legacy_track(self):
        camera2 = np.array([0.7, np.nan, 0.9], dtype=np.float32)
        camera1 = np.full_like(camera2, np.nan)

        fused, source = stereo.fuse_object_depth_tracks(camera1, camera2)

        np.testing.assert_allclose(fused[[0, 2]], camera2[[0, 2]])
        self.assertTrue(np.isnan(fused[1]))
        np.testing.assert_array_equal(
            source,
            [
                stereo.DEPTH_SOURCE_CAMERA2,
                stereo.DEPTH_SOURCE_NONE,
                stereo.DEPTH_SOURCE_CAMERA2,
            ],
        )

    def test_visibility_requires_c1_observation_and_c2_nonobservation(self):
        camera1 = np.array([[0.9, 0.9, 0.1, np.nan]], dtype=np.float32)
        camera2 = np.array([[0.1, 0.8, 0.0, 0.0]], dtype=np.float32)

        evidence = stereo.stereo_visibility_evidence(camera1, camera2)

        np.testing.assert_allclose(
            evidence,
            [[0.81, 0.18, 0.1, 0.0]],
            atol=1e-6,
        )

    def test_unknown_c2_visibility_fails_open(self):
        evidence = stereo.stereo_visibility_evidence(
            np.array([0.9], dtype=np.float32),
            np.array([np.nan], dtype=np.float32),
        )
        np.testing.assert_array_equal(evidence, [0.0])

    def test_temporal_modes_are_per_finger_and_nested(self):
        config = stereo.StereoOcclusionConfig(
            visibility_on=0.6,
            visibility_off=0.3,
            visibility_min_on_frames=2,
            visibility_hold_frames=1,
            haco_on=0.7,
            haco_off=0.4,
            haco_min_on_frames=2,
            haco_hold_frames=1,
        )
        camera1 = np.ones((5, 5), dtype=np.float32)
        camera2 = np.ones((5, 5), dtype=np.float32)
        # Finger 0 is a strong stereo decision. Fingers 1 and 2 are ambiguous.
        camera2[1:4, 0] = 0.0
        camera2[1:4, 1] = 0.6
        camera2[1:4, 2] = 0.6
        haco = np.zeros((5, 5), dtype=np.float32)
        haco[1:4, 1] = 0.9

        active, _, strong_active, haco_active = stereo.temporal_mode_decisions(
            camera1,
            camera2,
            haco,
            config,
        )

        self.assertTrue(active["visibility"][:, 0].any())
        # HaCo may not erase strong stereo evidence.
        self.assertTrue(strong_active[:, 0].any())
        self.assertTrue(active["visibility_depth_haco"][:, 0].any())
        # HaCo admits the ambiguous finger 1, but not ambiguous finger 2.
        self.assertTrue(active["visibility_depth_haco"][:, 1].any())
        self.assertFalse(active["visibility_depth_haco"][:, 2].any())
        self.assertTrue(haco_active[:, 1].any())
        self.assertFalse(
            np.any(
                active["visibility_depth_haco"]
                & ~active["visibility_depth"]
            )
        )

    def test_projected_finger_fractions_follow_modal_mask(self):
        vertices = np.zeros((1, 778, 3), dtype=np.float32)
        vertices[..., 2] = 1.0
        parts = np.zeros(778, dtype=np.int32)
        part_and_x = (
            (13, -0.30),  # thumb -> u=20
            (1, -0.15),   # index -> u=35
            (4, 0.00),    # middle -> u=50
            (10, 0.15),   # ring -> u=65
            (7, 0.30),    # pinky -> u=80
        )
        for finger_index, (part, x_value) in enumerate(part_and_x):
            selection = slice(finger_index * 10, (finger_index + 1) * 10)
            parts[selection] = part
            vertices[0, selection, 0] = x_value
        mask = np.zeros((1, 100, 100), dtype=bool)
        mask[0, 50, 20] = True
        mask[0, 50, 35] = True

        fractions, counts = stereo.projected_finger_visible_fractions(
            vertices_camera=vertices,
            valid_frames=np.array([True]),
            visible_masks=mask,
            finger_parts=parts,
            focal_px=100.0,
            image_width=100,
            image_height=100,
            probe_radius_px=0,
            point_support_threshold=1.0,
            min_projected_vertices=5,
        )

        np.testing.assert_allclose(fractions, [[1.0, 1.0, 0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(counts, [[10, 10, 10, 10, 10]])

    def test_invalid_projection_stays_unknown_despite_detector_confidence(self):
        vertices = np.zeros((1, 778, 3), dtype=np.float32)
        parts = np.ones(778, dtype=np.int32)
        fractions, counts = stereo.projected_finger_visible_fractions(
            vertices_camera=vertices,
            valid_frames=np.array([False]),
            visible_masks=np.zeros((1, 10, 10), dtype=bool),
            finger_parts=parts,
            focal_px=10.0,
            image_width=10,
            image_height=10,
            probe_radius_px=1,
            point_support_threshold=0.2,
            min_projected_vertices=5,
        )
        self.assertTrue(np.isnan(fractions).all())
        self.assertFalse(counts.any())

        known_and_unknown = np.array(
            [[1.0, 0.0, np.nan, 0.5, np.nan]],
            dtype=np.float32,
        )
        fused = stereo.fuse_visible_fraction_with_detector(
            np.array([0.8], dtype=np.float32),
            known_and_unknown,
        )
        np.testing.assert_allclose(
            fused,
            [[0.8, 0.0, np.nan, 0.4, np.nan]],
            equal_nan=True,
        )

        # In particular, a missing C2 hand detection must not turn every
        # unprojectable finger into a confident "hidden" observation.
        hidden_unknown = stereo.fuse_visible_fraction_with_detector(
            np.array([0.0], dtype=np.float32),
            np.full((1, 5), np.nan, dtype=np.float32),
        )
        evidence = stereo.stereo_visibility_evidence(
            np.full((1, 5), 0.9, dtype=np.float32),
            hidden_unknown,
        )
        np.testing.assert_array_equal(evidence, np.zeros((1, 5)))


class MetricDepthOrderTests(unittest.TestCase):
    def test_visibility_haco_requires_dual_view_masks_and_contact(self):
        stereo.validate_visibility_haco_inputs(
            enabled=False,
            camera1_visible_mask=None,
            camera2_visible_mask=None,
            camera1_contact_dir=None,
        )
        with self.assertRaisesRegex(ValueError, "camera1_visible_mask"):
            stereo.validate_visibility_haco_inputs(
                enabled=True,
                camera1_visible_mask=None,
                camera2_visible_mask=Path("camera2.npy"),
                camera1_contact_dir=Path("camera1_contact"),
            )
        with self.assertRaisesRegex(ValueError, "camera2_visible_mask"):
            stereo.validate_visibility_haco_inputs(
                enabled=True,
                camera1_visible_mask=Path("camera1.npy"),
                camera2_visible_mask=None,
                camera1_contact_dir=Path("camera1_contact"),
            )
        with self.assertRaisesRegex(ValueError, "camera1_contact_dir"):
            stereo.validate_visibility_haco_inputs(
                enabled=True,
                camera1_visible_mask=Path("camera1.npy"),
                camera2_visible_mask=Path("camera2.npy"),
                camera1_contact_dir=None,
            )
        stereo.validate_visibility_haco_inputs(
            enabled=True,
            camera1_visible_mask=Path("camera1.npy"),
            camera2_visible_mask=Path("camera2.npy"),
            camera1_contact_dir=Path("camera1_contact"),
        )

    def test_visibility_haco_requires_complete_dual_view_contact_coverage(self):
        stereo.validate_visibility_haco_coverage(
            enabled=True,
            frame_count=10,
            camera1_missing_frames=0,
            camera2_missing_frames=0,
        )
        with self.assertRaisesRegex(ValueError, "complete per-frame HaCo"):
            stereo.validate_visibility_haco_coverage(
                enabled=True,
                frame_count=10,
                camera1_missing_frames=1,
                camera2_missing_frames=0,
            )
        stereo.validate_visibility_haco_coverage(
            enabled=False,
            frame_count=10,
            camera1_missing_frames=10,
            camera2_missing_frames=10,
        )

    def test_visibility_haco_is_independently_opt_in_without_metric_depth(self):
        modes, metric_enabled = stereo.resolve_output_modes(
            include_haco_only=True,
            include_visibility_haco=True,
            include_haco_priority=False,
            camera1_metric_depth_npz=None,
            camera2_metric_depth_npz=None,
        )
        self.assertEqual(
            modes,
            stereo.MODE_NAMES
            + (stereo.HACO_ONLY_MODE, stereo.VISIBILITY_HACO_MODE),
        )
        self.assertFalse(metric_enabled)

    def test_visibility_haco_mask_preserves_nonfinger_and_nonobject_pixels(self):
        robot = np.ones((2, 4), dtype=bool)
        labels = np.array([[1, 2, 0, 1], [1, 2, 1, 2]], dtype=np.uint8)
        object_mask = np.array(
            [[True, True, True, False], [False, True, True, True]],
            dtype=bool,
        )
        selected = np.array([True, False, False, False, False], dtype=bool)

        result = stereo.compute_visibility_haco_occlusion_mask(
            robot_mask=robot,
            finger_labels=labels,
            object_mask=object_mask,
            visibility_haco_active_fingers=selected,
        )

        expected = np.array(
            [[True, False, False, False], [False, False, True, False]],
            dtype=bool,
        )
        np.testing.assert_array_equal(result, expected)

    def test_haco_priority_is_opt_in_and_requires_both_metric_inputs(self):
        baseline_modes, metric_enabled = stereo.resolve_output_modes(
            include_haco_only=False,
            include_haco_priority=False,
            camera1_metric_depth_npz=None,
            camera2_metric_depth_npz=None,
        )
        self.assertEqual(baseline_modes, stereo.MODE_NAMES)
        self.assertFalse(metric_enabled)

        with self.assertRaisesRegex(ValueError, "include_haco_priority requires"):
            stereo.resolve_output_modes(
                include_haco_only=False,
                include_haco_priority=True,
                camera1_metric_depth_npz=None,
                camera2_metric_depth_npz=None,
            )

        priority_modes, metric_enabled = stereo.resolve_output_modes(
            include_haco_only=False,
            include_haco_priority=True,
            camera1_metric_depth_npz=Path("camera1.npz"),
            camera2_metric_depth_npz=Path("camera2.npz"),
        )
        self.assertEqual(
            priority_modes,
            stereo.MODE_NAMES
            + (stereo.METRIC_DEPTH_ORDER_MODE, stereo.HACO_PRIORITY_MODE),
        )
        self.assertTrue(metric_enabled)

    def test_metric_depth_npz_contract_and_count_gating(self):
        hand = np.full((2, 5), 0.70, dtype=np.float32)
        object_depth = np.full((2, 5), 0.80, dtype=np.float32)
        hand_count = np.full((2, 5), 10, dtype=np.int32)
        object_count = np.full((2, 5), 30, dtype=np.int32)
        hand_count[0, 0] = 5
        object_count[0, 1] = 19
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metric_depth.npz"
            np.savez(
                path,
                finger_names=np.asarray(stereo.FINGER_NAMES),
                hand_depth_m=hand,
                object_depth_m=object_depth,
                hand_sample_count=hand_count,
                object_sample_count=object_count,
            )

            evidence = stereo.load_metric_finger_depth_evidence(
                path,
                frame_count=2,
                min_hand_samples=6,
                min_object_samples=20,
            )

        self.assertTrue(np.isfinite(evidence.hand_depth_m_raw).all())
        self.assertTrue(np.isnan(evidence.hand_depth_m[0, 0]))
        self.assertTrue(np.isnan(evidence.object_depth_m[0, 1]))
        self.assertTrue(np.isfinite(evidence.hand_depth_m[1]).all())
        np.testing.assert_array_equal(evidence.hand_sample_count, hand_count)

    def test_metric_depth_npz_rejects_unaligned_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metric_depth.npz"
            np.savez(
                path,
                hand_depth_m=np.ones((2, 4), dtype=np.float32),
                object_depth_m=np.ones((2, 4), dtype=np.float32),
                hand_sample_count=np.ones((2, 4), dtype=np.int32),
                object_sample_count=np.ones((2, 4), dtype=np.int32),
            )

            with self.assertRaisesRegex(ValueError, "shape"):
                stereo.load_metric_finger_depth_evidence(
                    path,
                    frame_count=2,
                    min_hand_samples=1,
                    min_object_samples=1,
                )

    def test_metric_depth_order_uses_camera_z_and_uncertainty_margin(self):
        hand = np.array(
            [[0.70, 0.90, 0.81, np.nan, -1.0]],
            dtype=np.float32,
        )
        object_depth = np.full((1, 5), 0.80, dtype=np.float32)

        order = stereo.classify_hand_object_depth_order(
            hand,
            object_depth,
            separation_margin_m=0.02,
        )

        np.testing.assert_array_equal(
            order,
            [[
                stereo.DEPTH_ORDER_HAND_FRONT,
                stereo.DEPTH_ORDER_OBJECT_FRONT,
                stereo.DEPTH_ORDER_AMBIGUOUS,
                stereo.DEPTH_ORDER_AMBIGUOUS,
                stereo.DEPTH_ORDER_AMBIGUOUS,
            ]],
        )

    def test_c2_object_front_is_authoritative_but_contradicted_hand_is_not(self):
        c1 = np.array(
            [[
                stereo.DEPTH_ORDER_OBJECT_FRONT,
                stereo.DEPTH_ORDER_HAND_FRONT,
                stereo.DEPTH_ORDER_HAND_FRONT,
                stereo.DEPTH_ORDER_OBJECT_FRONT,
                stereo.DEPTH_ORDER_HAND_FRONT,
                stereo.DEPTH_ORDER_HAND_FRONT,
                stereo.DEPTH_ORDER_HAND_FRONT,
            ]],
            dtype=np.uint8,
        )
        c2 = np.array(
            [[
                stereo.DEPTH_ORDER_HAND_FRONT,
                stereo.DEPTH_ORDER_OBJECT_FRONT,
                stereo.DEPTH_ORDER_AMBIGUOUS,
                stereo.DEPTH_ORDER_AMBIGUOUS,
                stereo.DEPTH_ORDER_AMBIGUOUS,
                stereo.DEPTH_ORDER_HAND_FRONT,
                stereo.DEPTH_ORDER_HAND_FRONT,
            ]],
            dtype=np.uint8,
        )
        visibility = np.array(
            [[1.0, 1.0, 0.8, 0.8, 0.2, 0.8, 0.2]],
            dtype=np.float32,
        )

        resolved, source = stereo.resolve_camera2_depth_order(
            c1,
            c2,
            visibility,
            visibility_assist_threshold=0.55,
        )

        np.testing.assert_array_equal(
            resolved,
            [[
                stereo.DEPTH_ORDER_HAND_FRONT,
                stereo.DEPTH_ORDER_OBJECT_FRONT,
                stereo.DEPTH_ORDER_OBJECT_FRONT,
                stereo.DEPTH_ORDER_OBJECT_FRONT,
                stereo.DEPTH_ORDER_AMBIGUOUS,
                stereo.DEPTH_ORDER_OBJECT_FRONT,
                stereo.DEPTH_ORDER_HAND_FRONT,
            ]],
        )
        np.testing.assert_array_equal(
            source,
            [[
                stereo.DEPTH_ORDER_SOURCE_CAMERA2_METRIC,
                stereo.DEPTH_ORDER_SOURCE_CAMERA2_METRIC,
                stereo.DEPTH_ORDER_SOURCE_STEREO_VISIBILITY,
                stereo.DEPTH_ORDER_SOURCE_STEREO_VISIBILITY,
                stereo.DEPTH_ORDER_SOURCE_AMBIGUOUS,
                stereo.DEPTH_ORDER_SOURCE_STEREO_CONTRADICTION,
                stereo.DEPTH_ORDER_SOURCE_CAMERA2_METRIC,
            ]],
        )

    def test_contact_selector_cannot_change_depth_order(self):
        mask = stereo.compute_depth_order_occlusion_mask(
            robot_mask=np.ones((1, 5), dtype=bool),
            finger_labels=np.array([[1, 2, 3, 0, 1]], dtype=np.uint8),
            object_mask=np.array([[True, True, True, True, False]], dtype=bool),
            camera2_depth_order=np.array(
                [
                    stereo.DEPTH_ORDER_OBJECT_FRONT,
                    stereo.DEPTH_ORDER_OBJECT_FRONT,
                    stereo.DEPTH_ORDER_AMBIGUOUS,
                    stereo.DEPTH_ORDER_OBJECT_FRONT,
                    stereo.DEPTH_ORDER_OBJECT_FRONT,
                ],
                dtype=np.uint8,
            ),
            haco_active_fingers=np.array(
                [True, False, True, True, True],
                dtype=bool,
            ),
        )

        # Only the object-front thumb survives semantic/object-mask gating.
        np.testing.assert_array_equal(mask, [[True, False, False, False, False]])

    def test_haco_priority_accepts_ambiguous_and_object_front_only(self):
        order = np.array(
            [
                stereo.DEPTH_ORDER_AMBIGUOUS,
                stereo.DEPTH_ORDER_OBJECT_FRONT,
                stereo.DEPTH_ORDER_HAND_FRONT,
                stereo.DEPTH_ORDER_AMBIGUOUS,
                stereo.DEPTH_ORDER_OBJECT_FRONT,
            ],
            dtype=np.uint8,
        )
        active = np.array([True, True, True, False, True], dtype=bool)

        selected = stereo.select_haco_priority_fingers(order, active)

        np.testing.assert_array_equal(
            selected,
            [True, True, False, False, True],
        )

    def test_haco_priority_pixel_mask_preserves_semantic_boundaries(self):
        mask = stereo.compute_haco_priority_occlusion_mask(
            robot_mask=np.array(
                [[True, True, True, True, True, True, False]],
                dtype=bool,
            ),
            finger_labels=np.array(
                [[1, 2, 3, 4, 5, 0, 1]],
                dtype=np.uint8,
            ),
            object_mask=np.array(
                [[True, True, True, True, False, True, True]],
                dtype=bool,
            ),
            resolved_depth_order=np.array(
                [
                    stereo.DEPTH_ORDER_AMBIGUOUS,
                    stereo.DEPTH_ORDER_OBJECT_FRONT,
                    stereo.DEPTH_ORDER_HAND_FRONT,
                    stereo.DEPTH_ORDER_AMBIGUOUS,
                    stereo.DEPTH_ORDER_OBJECT_FRONT,
                ],
                dtype=np.uint8,
            ),
            haco_active_fingers=np.array(
                [True, True, True, False, True],
                dtype=bool,
            ),
        )

        # Ambiguous and object-front active fingers hide.  Confident
        # hand-front, inactive contact, palm, outside-object, and non-robot
        # pixels stay visible.
        np.testing.assert_array_equal(
            mask,
            [[True, True, False, False, False, False, False]],
        )

    def test_haco_priority_missing_contact_fails_open(self):
        mask = stereo.compute_haco_priority_occlusion_mask(
            robot_mask=np.ones((1, 5), dtype=bool),
            finger_labels=np.arange(1, 6, dtype=np.uint8)[None, :],
            object_mask=np.ones((1, 5), dtype=bool),
            resolved_depth_order=np.full(
                5,
                stereo.DEPTH_ORDER_AMBIGUOUS,
                dtype=np.uint8,
            ),
            haco_active_fingers=np.zeros(5, dtype=bool),
        )

        self.assertFalse(mask.any())


class StereoPixelDecisionTests(unittest.TestCase):
    def _active(self, *, contact: bool = True) -> dict[str, np.ndarray]:
        visible = np.array([True, False, False, False, False])
        combined = visible.copy() if contact else np.zeros(5, dtype=bool)
        return {
            "visibility": visible,
            "visibility_depth": visible,
            "visibility_depth_haco": combined,
        }

    def test_only_active_finger_object_overlap_behind_depth_is_hidden(self):
        masks = stereo.compute_mode_occlusion_masks(
            robot_mask=np.array([[True, True, True, False]], dtype=bool),
            finger_labels=np.array([[1, 1, 0, 1]], dtype=np.uint8),
            robot_depth=np.array([[0.85, 0.75, 0.90, 0.90]], dtype=np.float32),
            object_mask=np.array([[True, True, True, True]], dtype=bool),
            active_fingers=self._active(),
            object_depth_m=0.80,
            depth_margin_m=0.03,
        )

        np.testing.assert_array_equal(
            masks["visibility"],
            [[True, True, False, False]],
        )
        np.testing.assert_array_equal(
            masks["visibility_depth"],
            [[True, False, False, False]],
        )
        np.testing.assert_array_equal(
            masks["visibility_depth_haco"],
            [[True, False, False, False]],
        )

    def test_missing_depth_fails_open_only_for_depth_modes(self):
        masks = stereo.compute_mode_occlusion_masks(
            robot_mask=np.ones((1, 2), dtype=bool),
            finger_labels=np.ones((1, 2), dtype=np.uint8),
            robot_depth=np.full((1, 2), 0.9, dtype=np.float32),
            object_mask=np.ones((1, 2), dtype=bool),
            active_fingers=self._active(),
            object_depth_m=np.nan,
            depth_margin_m=0.03,
        )

        self.assertTrue(masks["visibility"].all())
        self.assertFalse(masks["visibility_depth"].any())
        self.assertFalse(masks["visibility_depth_haco"].any())

    def test_missing_contact_fails_open_for_combined_mode(self):
        masks = stereo.compute_mode_occlusion_masks(
            robot_mask=np.ones((1, 1), dtype=bool),
            finger_labels=np.ones((1, 1), dtype=np.uint8),
            robot_depth=np.full((1, 1), 0.9, dtype=np.float32),
            object_mask=np.ones((1, 1), dtype=bool),
            active_fingers=self._active(contact=False),
            object_depth_m=0.8,
            depth_margin_m=0.03,
        )
        self.assertTrue(masks["visibility_depth"].all())
        self.assertFalse(masks["visibility_depth_haco"].any())

    def test_haco_only_uses_object_active_finger_intersection(self):
        mask = stereo.compute_haco_only_occlusion_mask(
            robot_mask=np.array([[True, True, True, False, True]], dtype=bool),
            finger_labels=np.array([[1, 2, 0, 1, 1]], dtype=np.uint8),
            object_mask=np.array([[True, True, True, True, False]], dtype=bool),
            haco_active_fingers=np.array(
                [True, False, False, False, False],
                dtype=bool,
            ),
        )

        # Only active thumb geometry inside both robot and object masks is
        # hidden.  Palm(label 0), inactive fingers, and outside-object pixels
        # remain visible without consulting visibility or depth.
        np.testing.assert_array_equal(mask, [[True, False, False, False, False]])

    def test_haco_only_inactive_contact_fails_open(self):
        mask = stereo.compute_haco_only_occlusion_mask(
            robot_mask=np.ones((2, 2), dtype=bool),
            finger_labels=np.ones((2, 2), dtype=np.uint8),
            object_mask=np.ones((2, 2), dtype=bool),
            haco_active_fingers=np.zeros(5, dtype=bool),
        )

        self.assertFalse(mask.any())


class AblationModeTests(unittest.TestCase):
    def test_tuned_ablation_defaults_are_deterministic(self):
        config = stereo.AblationConfig()

        self.assertAlmostEqual(config.depth_separation_margin_m, 0.010)
        self.assertAlmostEqual(config.confidence_depth_saturation_m, 0.025)
        self.assertAlmostEqual(config.confidence_stereo_start, 0.30)
        self.assertAlmostEqual(config.confidence_stereo_saturation, 0.55)
        self.assertAlmostEqual(config.confidence_depth_weight, 0.65)
        self.assertAlmostEqual(config.confidence_stereo_weight, 0.35)
        self.assertAlmostEqual(config.confidence_contact_floor, 0.50)
        self.assertAlmostEqual(config.confidence_score_threshold, 0.18)

    def test_ablation_modes_are_opt_in_and_require_metric_inputs(self):
        with self.assertRaisesRegex(ValueError, "include_ablation_modes requires"):
            stereo.resolve_output_modes(
                include_haco_only=False,
                include_haco_priority=False,
                include_ablation_modes=True,
                camera1_metric_depth_npz=None,
                camera2_metric_depth_npz=None,
            )

        modes, metric_enabled = stereo.resolve_output_modes(
            include_haco_only=False,
            include_haco_priority=False,
            include_ablation_modes=True,
            camera1_metric_depth_npz=Path("camera1.npz"),
            camera2_metric_depth_npz=Path("camera2.npz"),
        )

        self.assertTrue(metric_enabled)
        self.assertEqual(
            modes,
            stereo.MODE_NAMES
            + (stereo.METRIC_DEPTH_ORDER_MODE,)
            + stereo.ABLATION_MODE_NAMES,
        )

    def test_camera2_depth_only_uses_only_object_front_class(self):
        selected = stereo.select_camera2_depth_only_fingers(
            np.array(
                [
                    stereo.DEPTH_ORDER_OBJECT_FRONT,
                    stereo.DEPTH_ORDER_HAND_FRONT,
                    stereo.DEPTH_ORDER_AMBIGUOUS,
                    stereo.DEPTH_ORDER_OBJECT_FRONT,
                    stereo.DEPTH_ORDER_AMBIGUOUS,
                ],
                dtype=np.uint8,
            )
        )

        np.testing.assert_array_equal(
            selected,
            [True, False, False, True, False],
        )

    def test_vote_2of3_keeps_fixed_denominator_for_abstentions(self):
        evidence = stereo.compute_vote_2of3_evidence(
            haco_active=np.array(
                [[True, True, False, False, True]],
                dtype=bool,
            ),
            camera2_depth_order=np.array(
                [[
                    stereo.DEPTH_ORDER_OBJECT_FRONT,
                    stereo.DEPTH_ORDER_AMBIGUOUS,
                    stereo.DEPTH_ORDER_OBJECT_FRONT,
                    stereo.DEPTH_ORDER_HAND_FRONT,
                    stereo.DEPTH_ORDER_AMBIGUOUS,
                ]],
                dtype=np.uint8,
            ),
            strong_stereo_active=np.array(
                [[False, True, True, True, False]],
                dtype=bool,
            ),
        )

        np.testing.assert_array_equal(
            evidence.positive_count,
            [[2, 2, 2, 1, 1]],
        )
        np.testing.assert_array_equal(
            evidence.depth_vote_state,
            [[1, 0, 1, -1, 0]],
        )
        np.testing.assert_array_equal(
            evidence.selected,
            [[True, True, True, False, False]],
        )

    def test_confidence_ensemble_haco_has_no_direction_of_its_own(self):
        config = stereo.AblationConfig()
        evidence = stereo.compute_confidence_ensemble_evidence(
            camera2_hand_depth_m=np.array(
                [[0.80, 0.88, 0.72, np.nan, 0.805]],
                dtype=np.float32,
            ),
            camera2_object_depth_m=np.full(
                (1, 5),
                0.80,
                dtype=np.float32,
            ),
            stereo_visibility=np.array(
                [[0.30, 0.0, 1.0, 1.0, 0.30]],
                dtype=np.float32,
            ),
            haco_confidence=np.array(
                [[1.0, 0.0, 1.0, 1.0, 1.0]],
                dtype=np.float32,
            ),
            depth_deadzone_m=0.02,
            config=config,
        )

        # Finger 0 has perfect HaCo but no directional evidence.  Strong C2
        # object-front depth (1) and strong stereo (3) can decide; a strong
        # hand-front depth (2) defeats the positive stereo cue.
        np.testing.assert_array_equal(
            evidence.selected,
            [[False, True, False, True, False]],
        )
        self.assertEqual(float(evidence.score[0, 0]), 0.0)
        self.assertLess(float(evidence.direction_score[0, 2]), 0.0)

    def test_ablation_pixel_masks_never_remove_palm_or_outside_object(self):
        selected = np.array([True, False, False, False, False], dtype=bool)
        mask = stereo.compute_selected_finger_occlusion_mask(
            robot_mask=np.array([[True, True, True, False]], dtype=bool),
            finger_labels=np.array([[1, 0, 1, 1]], dtype=np.uint8),
            object_mask=np.array([[True, True, False, True]], dtype=bool),
            selected_fingers=selected,
        )
        baseline = stereo.compute_no_occlusion_mask(
            np.ones((2, 2), dtype=bool)
        )

        np.testing.assert_array_equal(mask, [[True, False, False, False]])
        self.assertFalse(baseline.any())

    def test_confidence_saturation_must_exceed_depth_deadzone(self):
        config = stereo.AblationConfig(confidence_depth_saturation_m=0.02)
        with self.assertRaisesRegex(ValueError, "must exceed"):
            config.validate(depth_deadzone_m=0.02)


class HaWoRTrackTests(unittest.TestCase):
    def test_model_tracks_confidence_is_dense_and_side_specific(self):
        tracks = {
            11: [
                {
                    "frame": 1,
                    "det": True,
                    "det_box": np.array([[1, 2, 3, 4, 0.83]]),
                    "det_handedness": np.array([0]),
                }
            ],
            22: [
                {
                    "frame": 2,
                    "det": True,
                    "det_box": np.array([[1, 2, 3, 4, 0.91]]),
                    "det_handedness": np.array([1]),
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_tracks.npy"
            np.save(path, np.array(tracks, dtype=object))

            left = stereo.load_track_observation_confidence(
                path,
                side="left",
                frame_count=4,
            )
            right = stereo.load_track_observation_confidence(
                path,
                side="right",
                frame_count=4,
            )

        np.testing.assert_allclose(left, [0.0, 0.83, 0.0, 0.0])
        np.testing.assert_allclose(right, [0.0, 0.0, 0.91, 0.0])


if __name__ == "__main__":
    unittest.main()
