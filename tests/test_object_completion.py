import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "inpainting"))

import inpaint_object_completion as completion  # noqa: E402


class ObjectCompletionGeometryTests(unittest.TestCase):
    def test_auxiliary_frame_indices_apply_positive_offset_and_fail_open(self):
        mapped = completion.auxiliary_frame_indices(6, 2)
        np.testing.assert_array_equal(mapped, np.asarray([2, 3, 4, 5, -1, -1]))

    def test_auxiliary_frame_indices_apply_negative_offset_and_fail_open(self):
        mapped = completion.auxiliary_frame_indices(5, -2)
        np.testing.assert_array_equal(mapped, np.asarray([-1, -1, 0, 1, 2]))

    def test_auxiliary_frame_indices_reject_empty_timeline(self):
        with self.assertRaisesRegex(ValueError, "frame_count must be positive"):
            completion.auxiliary_frame_indices(0, 0)

    def test_clean_modal_observations_removes_hand_overlap_and_margin(self):
        modal = np.zeros((1, 32, 32), dtype=bool)
        modal[0, 8:24, 8:24] = True
        hand = np.zeros_like(modal)
        hand[0, 12:20, 12:20] = True
        trusted, contested = completion.clean_modal_observations(
            modal,
            hand,
            hand_dilate_px=2,
        )
        self.assertTrue(np.array_equal(trusted | contested, modal))
        self.assertFalse(np.any(trusted & contested))
        expected = modal[0] & completion.dilate_mask(hand[0], 2)
        self.assertTrue(np.array_equal(contested[0], expected))
        self.assertTrue(np.all(contested[0, 12:20, 12:20]))

    def test_contact_cleaning_only_contests_supported_hand_overlap(self):
        modal = np.zeros((1, 24, 24), dtype=bool)
        modal[0, 4:20, 4:20] = True
        hand = np.zeros_like(modal)
        hand[0, 9:15, 9:15] = True
        contact = np.zeros_like(modal)
        contact[0, 8:12, 8:12] = True
        # Contact outside both the modal object and dilated hand must not
        # contest otherwise trusted object observations.
        contact[0, 18:22, 18:22] = True

        trusted, contested = (
            completion.clean_modal_observations_with_contact(
                modal,
                hand,
                contact,
                hand_dilate_px=1,
            )
        )

        expected = (
            modal[0]
            & completion.dilate_mask(hand[0], 1)
            & contact[0]
        )
        self.assertTrue(np.array_equal(contested[0], expected))
        self.assertTrue(np.array_equal(trusted[0], modal[0] & ~expected))
        self.assertTrue(np.array_equal(trusted | contested, modal))
        self.assertFalse(np.any(trusted & contested))
        self.assertTrue(trusted[0, 13, 13])

    def test_haco_hidden_selection_keeps_only_seeded_eight_connected_component(self):
        raw_hidden = np.zeros((1, 16, 16), dtype=bool)
        # This component is connected only diagonally, so retaining all three
        # pixels verifies that component labelling uses 8-connectivity.
        raw_hidden[0, 2, 2] = True
        raw_hidden[0, 3, 3] = True
        raw_hidden[0, 4, 4] = True
        raw_hidden[0, 10:12, 10:12] = True
        contact = np.zeros_like(raw_hidden)
        contact[0, 2, 2] = True

        selected, direct, fallback = (
            completion.select_haco_hidden_components(
                raw_hidden,
                contact,
                [completion.Segment("test", 0, 0)],
                temporal_grace_frames=0,
            )
        )

        self.assertTrue(np.all(selected[0, (2, 3, 4), (2, 3, 4)]))
        self.assertFalse(np.any(selected[0, 10:12, 10:12]))
        self.assertFalse(np.any(selected & ~raw_hidden))
        np.testing.assert_array_equal(direct, np.asarray([True]))
        np.testing.assert_array_equal(fallback, np.asarray([False]))

    def test_haco_hidden_selection_uses_bounded_temporal_raw_fallback(self):
        raw_hidden = np.zeros((7, 14, 14), dtype=bool)
        raw_hidden[:, 2:5, 2:5] = True
        raw_hidden[:, 9:11, 9:11] = True
        contact = np.zeros_like(raw_hidden)
        contact[3, 3, 3] = True

        selected, direct, fallback = (
            completion.select_haco_hidden_components(
                raw_hidden,
                contact,
                [completion.Segment("test", 0, 6)],
                temporal_grace_frames=1,
            )
        )

        # The contact frame keeps only its seeded component.  Adjacent frames
        # deliberately fail open to the complete raw support, while distant
        # contact-free frames remain excluded.
        expected_direct = np.zeros_like(raw_hidden[3])
        expected_direct[2:5, 2:5] = True
        self.assertTrue(np.array_equal(selected[3], expected_direct))
        self.assertTrue(np.array_equal(selected[2], raw_hidden[2]))
        self.assertTrue(np.array_equal(selected[4], raw_hidden[4]))
        self.assertFalse(np.any(selected[[0, 1, 5, 6]]))
        self.assertFalse(np.any(selected & ~raw_hidden))
        np.testing.assert_array_equal(
            direct,
            np.asarray([False, False, False, True, False, False, False]),
        )
        np.testing.assert_array_equal(
            fallback,
            np.asarray([False, False, True, False, True, False, False]),
        )

    def test_haco_hidden_selection_does_not_cross_segment_boundary(self):
        raw_hidden = np.zeros((4, 10, 10), dtype=bool)
        raw_hidden[:, 3:7, 3:7] = True
        contact = np.zeros_like(raw_hidden)
        contact[1, 4, 4] = True

        selected, direct, fallback = (
            completion.select_haco_hidden_components(
                raw_hidden,
                contact,
                [
                    completion.Segment("first", 0, 1),
                    completion.Segment("second", 2, 3),
                ],
                temporal_grace_frames=2,
            )
        )

        self.assertTrue(np.array_equal(selected[0], raw_hidden[0]))
        self.assertTrue(np.array_equal(selected[1], raw_hidden[1]))
        self.assertFalse(np.any(selected[2:]))
        np.testing.assert_array_equal(
            direct,
            np.asarray([False, True, False, False]),
        )
        np.testing.assert_array_equal(
            fallback,
            np.asarray([True, False, False, False]),
        )

    def test_component_hulls_do_not_join_separate_components(self):
        mask = np.zeros((48, 64), dtype=bool)
        mask[10:20, 8:18] = True
        mask[10:20, 42:52] = True
        hull = completion.component_convex_hulls(mask)
        self.assertTrue(np.all(hull[10:20, 8:18]))
        self.assertTrue(np.all(hull[10:20, 42:52]))
        self.assertFalse(np.any(hull[:, 22:38]))

    def test_hidden_support_fills_only_hand_covered_hull_concavity(self):
        modal = np.zeros((64, 64), dtype=bool)
        modal[16:48, 16:48] = True
        modal[26:38, 16:30] = False
        hand = np.zeros_like(modal)
        hand[24:40, 10:32] = True
        amodal, hidden = completion.infer_hidden_support(
            modal,
            hand,
            hand_dilate_px=0,
            max_modal_distance_px=30,
        )
        self.assertTrue(np.all(amodal[modal]))
        self.assertGreater(int(hidden.sum()), 0)
        self.assertFalse(np.any(hidden & modal))
        self.assertFalse(np.any(hidden & ~hand))
        self.assertTrue(hidden[31, 20])

    def test_hand_pose_warp_tracks_translation(self):
        source_hand = np.zeros((80, 80), dtype=bool)
        source_hand[20:45, 15:35] = True
        target_hand = np.zeros_like(source_hand)
        target_hand[27:52, 24:44] = True
        source_object = np.zeros_like(source_hand)
        source_object[18:30, 32:48] = True
        warped = completion.warp_mask_by_hand_pose(
            source_object,
            source_hand,
            target_hand,
        )
        source_center = np.asarray(np.nonzero(source_object)).mean(axis=1)
        warped_center = np.asarray(np.nonzero(warped)).mean(axis=1)
        np.testing.assert_allclose(
            warped_center - source_center,
            np.asarray((7.0, 9.0)),
            atol=1.0,
        )

    def test_reference_prior_extends_visible_half_under_hand(self):
        visible = np.zeros((72, 72), dtype=bool)
        visible[20:50, 35:50] = True
        hand = np.zeros_like(visible)
        hand[18:52, 18:37] = True
        reference_hull = np.zeros_like(visible)
        reference_hull[20:50, 20:50] = True
        selected = completion.best_warped_reference_prior(
            visible,
            hand,
            [(0, reference_hull, hand)],
            target_frame=1,
        )
        self.assertIsNotNone(selected)
        prior, reference_frame, coverage = selected
        self.assertEqual(reference_frame, 0)
        self.assertGreater(coverage, 0.95)
        self.assertTrue(prior[30, 25])

    def test_build_masks_uses_temporal_fallback_for_collapsed_modal(self):
        modal = np.zeros((3, 72, 72), dtype=bool)
        modal[0, 20:50, 20:50] = True
        modal[0, 30:40, 20:34] = False
        modal[1, 39, 39] = True
        modal[2, 30:60, 30:60] = True
        modal[2, 40:50, 30:44] = False
        hand = np.zeros_like(modal)
        hand[0, 26:44, 12:36] = True
        hand[1, 31:49, 17:41] = True
        hand[2, 36:54, 22:46] = True
        arm = hand.copy()
        segment = completion.Segment("test", 0, 2)
        amodal, hidden, object_inpaint, removal, reports = (
            completion.build_completion_masks(
                modal,
                hand,
                arm,
                [segment],
                hand_dilate_px=2,
                arm_dilate_px=1,
                object_inpaint_dilate_px=2,
            )
        )
        self.assertGreater(int(hidden[0].sum()), 0)
        self.assertGreater(int(hidden[1].sum()), 0)
        self.assertTrue(np.all(amodal[modal]))
        self.assertFalse(np.any(hidden & modal))
        self.assertFalse(np.any(object_inpaint & modal))
        self.assertFalse(np.any(removal & modal))
        self.assertEqual(reports[0]["temporal_fallback_frames"], 1)

    def test_hand_removal_uses_hawor_when_arm_mask_misses(self):
        modal = np.zeros((1, 48, 48), dtype=bool)
        modal[0, 16:32, 24:38] = True
        hand = np.zeros_like(modal)
        hand[0, 12:36, 10:28] = True
        arm = np.zeros_like(modal)
        _amodal, _hidden, _object_inpaint, removal, _reports = (
            completion.build_completion_masks(
                modal,
                hand,
                arm,
                [completion.Segment("test", 0, 0)],
                hand_dilate_px=0,
                arm_dilate_px=0,
                object_inpaint_dilate_px=0,
            )
        )
        self.assertTrue(np.all(removal[hand & ~modal]))
        self.assertFalse(np.any(removal & modal))

    def test_surface_extension_uses_nearest_valid_modal_depth(self):
        surface = np.zeros((20, 20), dtype=np.float32)
        modal = np.zeros((20, 20), dtype=bool)
        hidden = np.zeros((20, 20), dtype=bool)
        modal[6:14, 6:10] = True
        surface[modal] = 0.75
        hidden[6:14, 10:14] = True
        result = completion.extend_surface_frame(surface, modal, hidden)
        self.assertTrue(np.allclose(result[hidden], 0.75))
        self.assertTrue(np.array_equal(result[modal], surface[modal]))
        self.assertFalse(np.any(result[~(modal | hidden)]))

    def test_surface_extension_fails_open_without_valid_depth(self):
        surface = np.zeros((12, 12), dtype=np.float32)
        modal = np.zeros((12, 12), dtype=bool)
        hidden = np.zeros((12, 12), dtype=bool)
        modal[3:6, 3:6] = True
        hidden[3:6, 6:9] = True
        result = completion.extend_surface_frame(surface, modal, hidden)
        self.assertTrue(np.array_equal(result, surface))

    def test_temporal_surface_fallback_interpolates_missing_frame(self):
        surface = np.zeros((3, 10, 10), dtype=np.float32)
        modal = np.zeros((3, 10, 10), dtype=bool)
        modal[:, 3:7, 3:7] = True
        surface[0, modal[0]] = 0.70
        surface[2, modal[2]] = 0.90
        values = completion.temporal_surface_fallbacks(
            surface,
            modal,
            [completion.Segment("test", 0, 2)],
        )
        np.testing.assert_allclose(values, (0.70, 0.80, 0.90), atol=1e-6)

    def test_object_colour_constraint_rejects_hand_coloured_candidate(self):
        raw = np.zeros((20, 20, 3), dtype=np.uint8)
        candidate = np.zeros_like(raw)
        modal = np.zeros((20, 20), dtype=bool)
        hidden = np.zeros((20, 20), dtype=bool)
        modal[5:15, 5:10] = True
        hidden[5:15, 10:15] = True
        raw[modal] = (30, 210, 40)
        candidate[hidden] = (25, 25, 25)
        constrained, weights = completion.constrain_object_candidate(
            raw,
            candidate,
            modal,
            hidden,
            minimum_modal_pixels=10,
        )
        self.assertLess(float(weights[hidden].max()), 0.05)
        np.testing.assert_array_equal(
            constrained[hidden],
            np.broadcast_to(np.asarray((30, 210, 40)), (int(hidden.sum()), 3)),
        )

    def test_object_colour_constraint_fails_open_on_collapsed_modal(self):
        raw = np.zeros((12, 12, 3), dtype=np.uint8)
        candidate = np.full_like(raw, 77)
        modal = np.zeros((12, 12), dtype=bool)
        hidden = np.zeros((12, 12), dtype=bool)
        modal[2, 2] = True
        hidden[4:8, 4:8] = True
        constrained, weights = completion.constrain_object_candidate(
            raw,
            candidate,
            modal,
            hidden,
        )
        np.testing.assert_array_equal(constrained, candidate)
        self.assertTrue(np.all(weights == 1.0))


if __name__ == "__main__":
    unittest.main()
