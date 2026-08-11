"""Unit tests for contact-conditioned, finger-only RB5 compositing."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INPAINTING_DIR = REPO_ROOT / "src" / "inpainting"
sys.path.insert(0, str(INPAINTING_DIR))

import composite_rb5_contact_occlusion as occlusion  # noqa: E402
import rb5_finger_semantics as semantics  # noqa: E402


class FingerSemanticContractTests(unittest.TestCase):
    def test_filtered_semantics_are_a_strict_robot_mask_subset(self):
        semantic_ids = np.array(
            [
                [0, 1, 2, 0],
                [3, 4, 0, 5],
            ],
            dtype=np.int32,
        )
        robot_mask = np.array(
            [
                [True, True, False, True],
                [False, True, True, False],
            ],
            dtype=bool,
        )

        semantic_info = {
            "idToLabels": {
                "0": "BACKGROUND",
                "1": "UNLABELLED",
                "2": {"rb5_finger": "thumb"},
                "3": '{"rb5_finger": "index"}',
                "4": {"rb5_finger": "middle"},
                "5": {"rb5_finger": "ring"},
            }
        }
        finger_labels = semantics.finger_labels_from_semantics(
            semantic_ids[..., None],
            robot_mask,
            semantic_info,
        )
        finger_mask = finger_labels > 0

        expected = np.array(
            [
                [False, False, False, False],
                [False, True, False, False],
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(finger_mask, expected)
        self.assertEqual(
            finger_labels[1, 1],
            semantics.FINGER_LABEL_IDS["middle"],
        )
        self.assertFalse(np.any(finger_mask & ~robot_mask))

    def test_unlabelled_nonzero_id_is_never_a_finger(self):
        labels = semantics.finger_labels_from_semantics(
            np.array([[1, 2]], dtype=np.int32),
            np.ones((1, 2), dtype=bool),
            {
                "idToLabels": {
                    "1": "UNLABELLED",
                    "2": {"rb5_finger": "pinky"},
                }
            },
        )
        np.testing.assert_array_equal(
            labels,
            [[0, semantics.FINGER_LABEL_IDS["pinky"]]],
        )

    def test_colorized_rgba_metadata_maps_only_exact_finger_color(self):
        thumb = np.array(
            semantics.FINGER_SEMANTIC_COLORS_RGBA["thumb"],
            dtype=np.uint8,
        )
        rgba = np.array(
            [[thumb, [1, 2, 3, 255]]],
            dtype=np.uint8,
        )
        labels = semantics.finger_labels_from_semantics(
            rgba,
            np.ones((1, 2), dtype=bool),
            {
                "idToLabels": {
                    str(tuple(int(value) for value in thumb)): {
                        "rb5_finger": "thumb"
                    },
                    "(1, 2, 3, 255)": "UNLABELLED",
                }
            },
        )
        np.testing.assert_array_equal(
            labels,
            [[semantics.FINGER_LABEL_IDS["thumb"], 0]],
        )

    def test_xhand_contract_contains_only_the_twelve_finger_links(self):
        for side in ("left", "right"):
            expected = semantics.expected_finger_link_names(side)
            self.assertEqual(len(expected), 12)
            self.assertTrue(
                all(name.startswith(f"{side}_hand_") for name in expected)
            )
            self.assertFalse(any("palm" in name for name in expected))
            self.assertEqual(
                semantics.validate_finger_link_names(side, expected),
                expected,
            )

            with self.assertRaisesRegex(RuntimeError, "contract mismatch"):
                semantics.validate_finger_link_names(
                    side,
                    set(expected) - {next(iter(expected))},
                )


class ContactOcclusionPrimitiveTests(unittest.TestCase):
    def test_contact_score_fusion_uses_per_finger_maximum(self):
        primary = np.array(
            [
                [0.20, 0.80, 0.10, 0.50, 0.40],
                [0.90, 0.10, 0.30, 0.20, 0.60],
            ],
            dtype=np.float32,
        )
        auxiliary = np.array(
            [
                [0.90, 0.30, 0.60, 0.70, 0.20],
                [0.40, 0.70, 0.20, 0.80, 0.50],
            ],
            dtype=np.float32,
        )

        fused = occlusion.fuse_contact_scores(primary, auxiliary)

        np.testing.assert_allclose(
            fused,
            [
                [0.90, 0.80, 0.60, 0.70, 0.40],
                [0.90, 0.70, 0.30, 0.80, 0.60],
            ],
        )

    def test_contact_score_fusion_treats_missing_or_nan_aux_as_no_evidence(self):
        primary = np.array(
            [[0.20, 0.50, 0.60, 0.40, 0.30]],
            dtype=np.float32,
        )

        without_aux = occlusion.fuse_contact_scores(primary, None)
        with_nan_aux = occlusion.fuse_contact_scores(
            primary,
            np.array(
                [[np.nan, 0.80, np.nan, 0.10, np.nan]],
                dtype=np.float32,
            ),
        )

        np.testing.assert_allclose(
            without_aux,
            primary,
        )
        np.testing.assert_allclose(
            with_nan_aux,
            [[0.20, 0.80, 0.60, 0.40, 0.30]],
        )

    def test_contact_score_fusion_rejects_shape_mismatch(self):
        primary = np.zeros((2, len(occlusion.FINGER_NAMES)), dtype=np.float32)
        auxiliary = np.zeros((1, len(occlusion.FINGER_NAMES)), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "shape"):
            occlusion.fuse_contact_scores(primary, auxiliary)

    def test_sh_rescue_requires_mh_local_support_and_cannot_veto_mh(self):
        primary = np.zeros((4, len(occlusion.FINGER_NAMES)), np.float32)
        auxiliary = np.zeros_like(primary)
        hidden_fraction = np.zeros_like(primary)
        config = occlusion.OcclusionConfig(hold_frames=0)

        # MH activates thumb. A low SH score cannot veto it.
        primary[:2, 0] = 0.90
        hidden_fraction[:2, 0] = 0.60
        # SH proposes index, but MH has no local object support.
        auxiliary[:2, 1] = 0.90
        hidden_fraction[:2, 1] = 0.10
        # SH proposes middle and MH supplies local object support.
        auxiliary[:2, 2] = 0.90
        hidden_fraction[:2, 2] = 0.60

        scores, evidence, active, gates = (
            occlusion.contact_activation_tracks(
                primary,
                auxiliary,
                hidden_fraction,
                config,
            )
        )

        np.testing.assert_allclose(scores[:2, :3], 0.90)
        self.assertTrue(gates["primary"][:2, 0].all())
        self.assertTrue(gates["auxiliary_proposal"][:2, 1:3].all())
        self.assertFalse(gates["auxiliary_qualified"][:2, 1].any())
        self.assertTrue(gates["auxiliary_qualified"][:2, 2].all())
        self.assertTrue(active[:2, 0].all())
        self.assertFalse(active[:, 1].any())
        self.assertTrue(active[:2, 2].all())
        self.assertFalse(active[2:, :].any())
        self.assertTrue(np.all(evidence[:2, 1] == 0.0))

    def test_contact_activation_rejects_hidden_fraction_shape_mismatch(self):
        primary = np.zeros((2, len(occlusion.FINGER_NAMES)), np.float32)

        with self.assertRaisesRegex(ValueError, "hidden fractions"):
            occlusion.contact_activation_tracks(
                primary,
                None,
                np.zeros((1, len(occlusion.FINGER_NAMES)), np.float32),
                occlusion.OcclusionConfig(),
            )

    def test_temporal_hysteresis_rejects_spikes_and_holds_short_dropouts(self):
        evidence = np.array(
            [0.0, 0.8, 0.9, 0.1, 0.1, 0.1, 0.0, 0.8],
            dtype=np.float32,
        )

        active = occlusion.temporal_hysteresis(
            evidence,
            on_threshold=0.7,
            off_threshold=0.3,
            min_on_frames=2,
            hold_frames=2,
        )

        np.testing.assert_array_equal(
            active,
            [False, True, True, True, True, False, False, False],
        )

    def test_short_occlusion_runs_are_removed_per_finger(self):
        tracks = np.array(
            [
                [False, True],
                [True, False],
                [True, True],
                [False, True],
            ],
            dtype=bool,
        )

        stable = occlusion.suppress_short_runs(tracks, min_frames=2)

        np.testing.assert_array_equal(
            stable,
            [
                [False, False],
                [True, False],
                [True, True],
                [False, True],
            ],
        )

    def test_depth_smoothing_does_not_fill_unbounded_missing_edges(self):
        values = np.array([np.nan, 0.5, np.nan, 0.7, np.nan], dtype=np.float32)

        filled = occlusion._median_fill_short_gaps(
            values,
            max_gap=1,
            window=3,
        )

        self.assertTrue(np.isnan(filled[0]))
        self.assertTrue(np.isnan(filled[-1]))
        self.assertTrue(np.isfinite(filled[2]))

    def test_camera_projection_marks_nonpositive_and_nonfinite_depth_invalid(self):
        points = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, -0.5, 2.0],
                [np.nan, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )

        uv, valid = occlusion.project_camera_points(
            points,
            focal_px=100.0,
            image_width=200,
            image_height=100,
        )

        np.testing.assert_array_equal(
            valid,
            [True, True, False, False, False],
        )
        np.testing.assert_allclose(
            uv[:2],
            [[100.0, 50.0], [150.0, 25.0]],
            atol=1e-6,
        )
        self.assertTrue(np.isnan(uv[~valid]).all())

    def test_missing_depth_evidence_fails_open(self):
        shape = (3, 4)
        occluded = occlusion.compute_occluded_fingers(
            robot_mask=np.ones(shape, dtype=bool),
            finger_mask=np.ones(shape, dtype=bool),
            robot_depth=np.full(shape, 0.8, dtype=np.float32),
            occluder_mask=np.ones(shape, dtype=bool),
            contact_support_mask=np.ones(shape, dtype=bool),
            object_depth_m=np.nan,
            contact_depth_m=np.nan,
        )

        self.assertFalse(occluded.any())

    def test_depth_gate_never_expands_beyond_candidate_finger_pixels(self):
        robot_mask = np.array([[True, True, True, False]], dtype=bool)
        finger_mask = np.array([[True, False, True, True]], dtype=bool)
        occluder_mask = np.array([[True, True, False, True]], dtype=bool)
        support_mask = np.ones((1, 4), dtype=bool)
        robot_depth = np.array([[0.8, 0.8, 0.8, 0.8]], dtype=np.float32)

        occluded = occlusion.compute_occluded_fingers(
            robot_mask=robot_mask,
            finger_mask=finger_mask,
            robot_depth=robot_depth,
            occluder_mask=occluder_mask,
            contact_support_mask=support_mask,
            object_depth_m=0.7,
            object_depth_margin_m=0.01,
        )

        np.testing.assert_array_equal(
            occluded,
            [[True, False, False, False]],
        )
        self.assertFalse(np.any(occluded & ~finger_mask))
        self.assertFalse(np.any(occluded & ~robot_mask))

    def test_zero_xhand_thickness_bias_is_exactly_the_legacy_gate(self):
        shape = (1, 5)
        depths = np.array(
            [[0.975, 0.987, 0.988, 0.999, np.nan]],
            dtype=np.float32,
        )
        common = {
            "robot_mask": np.ones(shape, dtype=bool),
            "finger_mask": np.ones(shape, dtype=bool),
            "robot_depth": depths,
            "occluder_mask": np.ones(shape, dtype=bool),
            "contact_support_mask": np.ones(shape, dtype=bool),
            "contact_depth_m": 1.0,
            "contact_depth_tolerance_m": 0.012,
        }

        legacy = occlusion.compute_occluded_fingers(**common)
        explicit_zero = occlusion.compute_occluded_fingers(
            **common,
            contact_depth_bias_m=0.0,
        )

        np.testing.assert_array_equal(explicit_zero, legacy)
        config = occlusion.OcclusionConfig()
        config.validate()
        self.assertEqual(config.contact_depth_thickness_scale, 0.0)
        self.assertEqual(
            config.xhand_thumb_thickness_m,
            occlusion.XHAND_THUMB_THICKNESS_M,
        )
        self.assertEqual(
            config.xhand_finger_thickness_m,
            occlusion.XHAND_FINGER_THICKNESS_M,
        )
        self.assertEqual(config.contact_depth_bias_m("thumb"), 0.0)
        self.assertEqual(config.contact_depth_bias_m("index"), 0.0)

    def test_half_and_full_xhand_thickness_expand_contact_proxy_gate(self):
        shape = (1, 3)
        common = {
            "robot_mask": np.ones(shape, dtype=bool),
            "finger_mask": np.ones(shape, dtype=bool),
            "robot_depth": np.array(
                [[0.96, 0.97, 0.98]],
                dtype=np.float32,
            ),
            "occluder_mask": np.ones(shape, dtype=bool),
            "contact_support_mask": np.ones(shape, dtype=bool),
            "contact_depth_m": 1.0,
            "contact_depth_tolerance_m": 0.012,
        }
        half = occlusion.OcclusionConfig(
            contact_depth_thickness_scale=0.5,
        )
        full = occlusion.OcclusionConfig(
            contact_depth_thickness_scale=1.0,
        )
        half.validate()
        full.validate()

        baseline = occlusion.compute_occluded_fingers(**common)
        half_finger = occlusion.compute_occluded_fingers(
            **common,
            contact_depth_bias_m=half.contact_depth_bias_m("index"),
        )
        half_thumb = occlusion.compute_occluded_fingers(
            **common,
            contact_depth_bias_m=half.contact_depth_bias_m("thumb"),
        )
        full_finger = occlusion.compute_occluded_fingers(
            **common,
            contact_depth_bias_m=full.contact_depth_bias_m("middle"),
        )

        np.testing.assert_array_equal(baseline, [[False, False, False]])
        np.testing.assert_array_equal(half_finger, [[False, False, True]])
        np.testing.assert_array_equal(half_thumb, [[False, True, True]])
        np.testing.assert_array_equal(full_finger, [[True, True, True]])
        self.assertAlmostEqual(half.contact_depth_bias_m("thumb"), 0.01958)
        self.assertAlmostEqual(half.contact_depth_bias_m("index"), 0.01465)

    def test_xhand_bias_does_not_modify_metric_object_depth_gate(self):
        shape = (1, 3)
        common = {
            "robot_mask": np.ones(shape, dtype=bool),
            "finger_mask": np.ones(shape, dtype=bool),
            "robot_depth": np.array(
                [[0.98, 0.981, 1.02]],
                dtype=np.float32,
            ),
            "occluder_mask": np.ones(shape, dtype=bool),
            "contact_support_mask": np.ones(shape, dtype=bool),
            "object_depth_m": 0.97,
            "contact_depth_m": 2.0,
            "object_depth_margin_m": 0.01,
        }

        baseline = occlusion.compute_occluded_fingers(**common)
        biased = occlusion.compute_occluded_fingers(
            **common,
            contact_depth_bias_m=0.5,
        )

        np.testing.assert_array_equal(biased, baseline)
        np.testing.assert_array_equal(biased, [[False, True, True]])

    def test_invalid_xhand_thickness_settings_are_rejected(self):
        invalid = (
            {"contact_depth_thickness_scale": -0.1},
            {"contact_depth_thickness_scale": float("nan")},
            {"contact_depth_thickness_scale": float("inf")},
            {"xhand_thumb_thickness_m": 0.0},
            {"xhand_thumb_thickness_m": float("nan")},
            {"xhand_finger_thickness_m": -0.01},
        )
        for settings in invalid:
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(ValueError, "thickness|XHand"):
                    occlusion.OcclusionConfig(**settings).validate()

        with self.assertRaisesRegex(ValueError, "contact depth bias"):
            occlusion.compute_occluded_fingers(
                robot_mask=np.ones((1, 1), dtype=bool),
                finger_mask=np.ones((1, 1), dtype=bool),
                robot_depth=np.ones((1, 1), dtype=np.float32),
                occluder_mask=np.ones((1, 1), dtype=bool),
                contact_support_mask=np.ones((1, 1), dtype=bool),
                contact_depth_m=1.0,
                contact_depth_bias_m=-0.01,
            )

        with self.assertRaisesRegex(ValueError, "unknown finger"):
            occlusion.OcclusionConfig().contact_depth_bias_m("palm")

    def test_contact_interior_expansion_is_off_by_default(self):
        config = occlusion.OcclusionConfig()
        config.validate()

        self.assertEqual(config.contact_interior_expand_px, 0)
        self.assertEqual(
            config.contact_interior_expand_cap_fraction,
            0.25,
        )
        candidate = np.zeros((5, 7), dtype=bool)
        candidate[2, 1:3] = True
        finger = np.zeros_like(candidate)
        finger[1:4, 1:6] = True

        expanded, diagnostics = (
            occlusion.expand_verified_contact_interior(
                candidate,
                eligible_mask=finger,
                finger_mask=finger,
                expand_px=config.contact_interior_expand_px,
                added_cap_fraction=(
                    config.contact_interior_expand_cap_fraction
                ),
            )
        )

        np.testing.assert_array_equal(expanded, candidate)
        self.assertEqual(diagnostics["added_pixels"], 0)

    def test_border_contact_grows_deterministically_with_a_strict_cap(self):
        finger = np.zeros((7, 10), dtype=bool)
        finger[1:6, 1:4] = True
        # A disconnected fragment of the same semantic finger must not be
        # reached by the bounded geodesic growth.
        finger[1:6, 7:9] = True
        candidate = np.zeros_like(finger)
        candidate[2:4, 1:3] = True

        first, diagnostics = occlusion.expand_verified_contact_interior(
            candidate,
            eligible_mask=finger,
            finger_mask=finger,
            expand_px=4,
            added_cap_fraction=0.5,
        )
        second, repeated = occlusion.expand_verified_contact_interior(
            candidate,
            eligible_mask=finger,
            finger_mask=finger,
            expand_px=4,
            added_cap_fraction=0.5,
        )

        np.testing.assert_array_equal(first, second)
        self.assertEqual(diagnostics, repeated)
        self.assertEqual(int(candidate.sum()), 4)
        self.assertEqual(diagnostics["added_cap_pixels"], 2)
        self.assertEqual(diagnostics["added_pixels"], 2)
        self.assertTrue(diagnostics["cap_limited"])
        self.assertTrue(diagnostics["expanded"])
        self.assertFalse(first[:, 7:9].any())
        self.assertFalse(np.any(first & ~finger))

    def test_interior_candidate_without_a_border_seed_is_not_expanded(self):
        finger = np.zeros((9, 9), dtype=bool)
        finger[1:8, 1:8] = True
        candidate = np.zeros_like(finger)
        candidate[4:6, 4:6] = True

        expanded, diagnostics = (
            occlusion.expand_verified_contact_interior(
                candidate,
                eligible_mask=finger,
                finger_mask=finger,
                expand_px=4,
                added_cap_fraction=1.0,
            )
        )

        np.testing.assert_array_equal(expanded, candidate)
        self.assertEqual(diagnostics["boundary_seed_pixels"], 0)
        self.assertEqual(diagnostics["added_pixels"], 0)

    def test_interior_growth_cannot_cross_an_unverified_pixel(self):
        finger = np.zeros((5, 9), dtype=bool)
        finger[1:4, 1:8] = True
        eligible = finger.copy()
        eligible[:, 4] = False
        candidate = np.zeros_like(finger)
        candidate[1:3, 1:3] = True

        expanded, _ = occlusion.expand_verified_contact_interior(
            candidate,
            eligible_mask=eligible,
            finger_mask=finger,
            expand_px=8,
            added_cap_fraction=1.0,
        )

        self.assertFalse(expanded[:, 5:].any())
        self.assertFalse(np.any(expanded & ~eligible))


class ContactOcclusionCompositeTests(unittest.TestCase):
    def test_explicit_object_mask_does_not_require_hand_visibility_masks(self):
        class FakeCapture:
            def __init__(self, _path):
                self._read = False

            def read(self):
                if self._read:
                    return False, None
                self._read = True
                return True, np.full((4, 4, 3), 32, dtype=np.uint8)

            def release(self):
                pass

        class FakeWriter:
            def __init__(self, path):
                self.path = Path(path)
                self.frames = 0

            def write(self, _frame):
                self.frames += 1

            def release(self):
                self.path.write_bytes(b"fake-video")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "processed"
            episode = root / "episode"
            overlay = processed / "overlay_processor"
            rgb = episode / "rgb"
            contact = episode / "contact"
            aux_contact = root / "aux_contact"
            for directory in (processed, overlay, rgb, contact, aux_contact):
                directory.mkdir(parents=True, exist_ok=True)

            (overlay / "manifest.json").write_text(
                json.dumps({"side": "right"})
            )
            robot_mask = np.ones((1, 4, 4), dtype=bool)
            finger_labels = np.zeros((1, 4, 4), dtype=np.uint8)
            finger_labels[:, :2, :2] = 1
            finger_mask = finger_labels > 0
            np.save(
                overlay / "robot_rgb.npy",
                np.zeros((1, 4, 4, 3), np.uint8),
            )
            np.save(
                overlay / "robot_depth.npy",
                np.ones((1, 4, 4), np.float32),
            )
            np.save(overlay / "robot_mask.npy", robot_mask)
            np.save(overlay / "robot_finger_mask.npy", finger_mask)
            np.save(overlay / "robot_finger_labels.npy", finger_labels)

            source_frame = rgb / "000000.jpg"
            source_frame.write_bytes(b"frame-placeholder")
            np.savez(contact / "000000.npz")
            aux_parts = np.load(
                REPO_ROOT / "src" / "retargeting" / "assets" /
                "finger_part_left.npy"
            ).astype(np.int32)
            aux_palmar = np.load(
                REPO_ROOT / "src" / "retargeting" / "assets" /
                "palmar_mask_left.npy"
            ).astype(bool)
            aux_probability = np.zeros(778, dtype=np.float32)
            aux_mask = np.zeros(778, dtype=bool)
            auxiliary_scores = np.array(
                [0.90, 0.30, 0.60, 0.70, 0.20],
                dtype=np.float32,
            )
            for finger_index, finger in enumerate(occlusion.FINGER_NAMES):
                eligible = (
                    aux_palmar
                    & np.isin(aux_parts, occlusion.FINGER_PARTS[finger])
                )
                aux_probability[eligible] = auxiliary_scores[finger_index]
                aux_mask[eligible] = True
            np.savez(
                aux_contact / "000000.npz",
                left_valid=np.bool_(True),
                left_contact_probability=aux_probability,
                left_contact_mask=aux_mask,
            )
            hawor = root / "retarget_input.npz"
            np.savez(
                hawor,
                verts_right=np.zeros((1, 778, 3), dtype=np.float32),
                img_focal=np.float32(100.0),
            )
            object_mask = root / "object_mask.npy"
            np.save(object_mask, np.ones((1, 4, 4), dtype=bool))
            object_restore_mask = root / "object_restore_mask.npy"
            clean_restore = np.zeros((1, 4, 4), dtype=bool)
            clean_restore[:, 1, 2] = True
            np.save(object_restore_mask, clean_restore)
            output = root / "result"

            points_uv = {
                finger: np.empty((0, 2), dtype=np.float32)
                for finger in occlusion.FINGER_NAMES
            }
            points_z = {
                finger: np.empty(0, dtype=np.float32)
                for finger in occlusion.FINGER_NAMES
            }
            argv = [
                "composite_rb5_contact_occlusion.py",
                "--processed_demo",
                str(processed),
                "--episode_dir",
                str(episode),
                "--out_dir",
                str(output),
                "--background",
                str(root / "background.mkv"),
                "--raw_video",
                str(root / "raw.mp4"),
                "--hawor_npz",
                str(hawor),
                "--contact_dir",
                str(contact),
                "--aux_contact_dir",
                str(aux_contact),
                "--aux_frame_offset",
                "0",
                "--aux_side",
                "left",
                "--overlay_dir",
                str(overlay),
                "--object_mask",
                str(object_mask),
                "--object_restore_mask",
                str(object_restore_mask),
                "--min_occlusion_run_frames",
                "1",
            ]

            def open_writer(path, _fps, _size):
                return FakeWriter(path)

            primary_scores = np.array(
                [0.20, 0.80, 0.10, 0.50, 0.40],
                dtype=np.float32,
            )

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    occlusion,
                    "_video_metadata",
                    return_value=(4, 4, 1, 24.0),
                ),
                mock.patch.object(occlusion.cv2, "VideoCapture", FakeCapture),
                mock.patch.object(occlusion, "_open_writer", open_writer),
                mock.patch.object(
                    occlusion,
                    "_contact_frame_features",
                    return_value=(
                        primary_scores,
                        points_uv,
                        points_z,
                    ),
                ),
            ):
                occlusion.main()

            self.assertFalse(
                (processed / "segmentation_processor" / "masks_arm.npy").exists()
            )
            self.assertEqual(
                list((episode / "rgb_hawor").glob("tracks_*/model_masks.npy")),
                [],
            )
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["mode"], "modal_object_mask")
            self.assertEqual(
                report["visibility_evidence"], "verified_modal_object_mask"
            )
            self.assertEqual(
                report["sources"]["aux_contact_dir"],
                str(aux_contact.resolve()),
            )
            self.assertEqual(
                report["sources"]["object_restore_mask"],
                str(object_restore_mask.resolve()),
            )
            self.assertEqual(report["raw_object_pixels_total"], 1)
            self.assertTrue(
                report["invariants"][
                    "raw_rgb_restore_uses_object_restore_mask_only"
                ]
            )
            self.assertTrue(
                report["invariants"][
                    "object_restore_mask_subset_of_object_mask"
                ]
            )
            self.assertEqual(
                report["contact_fusion"],
                "per-finger maximum of primary/auxiliary HaCo scores",
            )
            policy = report["contact_activation_policy"]
            self.assertEqual(
                policy["name"],
                "mh_geometry_with_sh_confidence_rescue",
            )
            self.assertEqual(policy["unit"], "five MANO fingers")
            self.assertFalse(policy["auxiliary_geometry_used"])
            self.assertEqual(policy["auxiliary_frame_lookup"], [0])
            self.assertEqual(
                policy["counts"]["auxiliary_only_threshold_proposals"],
                1,
            )
            self.assertEqual(
                policy["counts"][
                    "auxiliary_proposals_with_primary_local_support"
                ],
                0,
            )
            self.assertEqual(
                policy["counts"][
                    "active_frame_fingers_added_vs_primary"
                ],
                0,
            )
            expected_fused = np.maximum(primary_scores, auxiliary_scores)
            np.testing.assert_allclose(
                report["contact_score_primary"],
                [primary_scores],
            )
            np.testing.assert_allclose(
                report["contact_score_auxiliary"],
                [auxiliary_scores],
            )
            np.testing.assert_allclose(
                report["contact_score_fused"],
                [expected_fused],
            )
            np.testing.assert_allclose(
                report["contact_score"],
                report["contact_score_fused"],
            )
            interior = report["contact_interior_expansion"]
            self.assertFalse(interior["enabled"])
            self.assertEqual(interior["expand_px"], 0)
            self.assertEqual(interior["added_pixels_final"], 0)
            self.assertFalse(interior["auxiliary_geometry_used"])
            thickness = report["xhand_contact_depth_bias"]
            self.assertFalse(thickness["enabled"])
            self.assertEqual(thickness["scale"], 0.0)
            self.assertFalse(thickness["metric_object_depth_gate_modified"])
            self.assertTrue(
                all(
                    value == 0.0
                    for value in thickness["applied_bias_m"].values()
                )
            )
            self.assertTrue(
                report["invariants"]["occluded_subset_of_robot_fingers"]
            )
            self.assertTrue(
                report["invariants"][
                    "xhand_thickness_bias_is_contact_proxy_only"
                ]
            )
            self.assertTrue(
                report["invariants"][
                    "sensor_object_depth_gate_is_unbiased"
                ]
            )
            self.assertTrue(
                report["invariants"][
                    "contact_interior_expansion_respects_added_pixel_cap"
                ]
            )
            occluded = np.load(output / "occluded_finger_mask.npy")
            self.assertFalse(np.any(occluded & ~finger_mask))

    def test_overlay_resize_accepts_read_only_same_size_robot_mask(self):
        robot_mask = np.array(
            [[True, False], [False, False]],
            dtype=bool,
        )
        robot_mask.flags.writeable = False
        finger_labels = np.array(
            [[0, 1], [0, 0]],
            dtype=np.uint8,
        )

        _, _, robot, fingers, labels = occlusion._resize_overlay_frame(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2), dtype=np.float32),
            robot_mask,
            finger_labels,
            width=2,
            height=2,
        )

        np.testing.assert_array_equal(robot, [[True, True], [False, False]])
        np.testing.assert_array_equal(fingers, labels > 0)
        self.assertTrue(robot.flags.writeable)

    def test_occlusion_preserves_every_non_finger_pixel(self):
        background = np.full((3, 4, 3), [10, 20, 30], dtype=np.uint8)
        robot_rgb = np.full((3, 4, 3), [100, 150, 200], dtype=np.uint8)
        robot_mask = np.array(
            [
                [True, True, True, False],
                [True, True, True, False],
                [False, False, False, False],
            ],
            dtype=bool,
        )
        finger_mask = np.zeros((3, 4), dtype=bool)
        finger_mask[0, 0] = True
        occluded_mask = np.zeros((3, 4), dtype=bool)
        occluded_mask[0, 0] = True

        baseline, baseline_robot, baseline_alpha = occlusion.composite_frame(
            background,
            robot_rgb,
            robot_mask,
            finger_mask,
            np.zeros_like(occluded_mask),
            robot_edge_sigma_px=0.0,
            occlusion_edge_sigma_px=1.0,
        )
        final, robot_only, alpha = occlusion.composite_frame(
            background,
            robot_rgb,
            robot_mask,
            finger_mask,
            occluded_mask,
            robot_edge_sigma_px=0.0,
            occlusion_edge_sigma_px=1.0,
        )

        non_finger = ~finger_mask
        np.testing.assert_array_equal(final[non_finger], baseline[non_finger])
        np.testing.assert_array_equal(
            robot_only[non_finger],
            baseline_robot[non_finger],
        )
        np.testing.assert_array_equal(
            alpha[non_finger],
            baseline_alpha[non_finger],
        )
        self.assertLess(alpha[0, 0], baseline_alpha[0, 0])

    def test_non_finger_occlusion_input_is_rejected(self):
        shape = (2, 2)
        finger_mask = np.zeros(shape, dtype=bool)
        invalid_occlusion = np.zeros(shape, dtype=bool)
        invalid_occlusion[0, 0] = True

        with self.assertRaisesRegex(
            ValueError,
            "contains non-finger pixels",
        ):
            occlusion.composite_frame(
                np.zeros((*shape, 3), dtype=np.uint8),
                np.zeros((*shape, 3), dtype=np.uint8),
                np.ones(shape, dtype=bool),
                finger_mask,
                invalid_occlusion,
                robot_edge_sigma_px=0.0,
                occlusion_edge_sigma_px=0.0,
            )


if __name__ == "__main__":
    unittest.main()
