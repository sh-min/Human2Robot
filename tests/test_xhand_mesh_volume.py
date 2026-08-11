"""Focused tests for the pose-fitted XHand mesh-volume compositor."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INPAINTING_DIR = REPO_ROOT / "src" / "inpainting"
sys.path.insert(0, str(INPAINTING_DIR))

import composite_xhand_mesh_volume as volume  # noqa: E402


class MeshVolumeValidationTests(unittest.TestCase):
    def _arrays(self):
        shape = (3, 2, 4)
        front = np.full(shape, 0.8, dtype=np.float32)
        back = np.full(shape, 0.9, dtype=np.float32)
        mask = np.ones(shape, dtype=bool)
        pose_valid = np.asarray((True, False, True), dtype=bool)
        mask[1] = False
        front[1] = 0.0
        back[1] = 0.0
        return shape, front, back, mask, pose_valid

    def test_strict_front_back_contract_accepts_ordered_float_depth(self):
        shape, front, back, mask, pose_valid = self._arrays()

        result = volume.validate_depth_volume(
            front,
            back,
            mask,
            pose_valid,
            expected_shape=shape,
        )

        self.assertEqual(result["mesh_pixels"], 16)
        self.assertEqual(result["valid_pose_mesh_pixels"], 16)
        self.assertEqual(result["valid_pose_frames"], 2)

    def test_pose_invalid_frame_accepts_empty_zero_sentinel(self):
        shape, front, back, mask, pose_valid = self._arrays()

        volume.validate_depth_volume(
            front,
            back,
            mask,
            pose_valid,
            expected_shape=shape,
        )

    def test_pose_invalid_frame_and_outside_mask_require_zero_sentinel(self):
        shape, front, back, mask, pose_valid = self._arrays()
        mask[1] = True
        front[1] = 0.8
        back[1] = 0.9
        with self.assertRaisesRegex(ValueError, "pose-invalid"):
            volume.validate_depth_volume(
                front,
                back,
                mask,
                pose_valid,
                expected_shape=shape,
            )

        mask[1] = False
        front[1] = 0.0
        back[1] = 0.0
        mask[0, 0, 0] = False
        with self.assertRaisesRegex(ValueError, "outside-mask sentinel"):
            volume.validate_depth_volume(
                front,
                back,
                mask,
                pose_valid,
                expected_shape=shape,
            )

    def test_valid_pose_mask_rejects_nan_out_of_range_and_reversed_depth(self):
        corruptions = (
            (np.nan, 0.9),
            (0.01, 0.9),
            (0.8, 5.1),
            (1.0, 0.9),
        )
        for invalid_front, invalid_back in corruptions:
            with self.subTest(front=invalid_front, back=invalid_back):
                shape, front, back, mask, pose_valid = self._arrays()
                front[0, 0, 0] = invalid_front
                back[0, 0, 0] = invalid_back
                with self.assertRaisesRegex(ValueError, "invalid or unordered"):
                    volume.validate_depth_volume(
                        front,
                        back,
                        mask,
                        pose_valid,
                        expected_shape=shape,
                    )

    def test_depth_mask_and_pose_dtypes_are_strict(self):
        shape, front, back, mask, pose_valid = self._arrays()
        with self.assertRaisesRegex(TypeError, "floating"):
            volume.validate_depth_volume(
                front.astype(np.int16),
                back,
                mask,
                pose_valid,
                expected_shape=shape,
            )
        with self.assertRaisesRegex(TypeError, "mask must have dtype bool"):
            volume.validate_depth_volume(
                front,
                back,
                mask.astype(np.uint8),
                pose_valid,
                expected_shape=shape,
            )
        with self.assertRaisesRegex(TypeError, "pose_valid must have dtype bool"):
            volume.validate_depth_volume(
                front,
                back,
                mask,
                pose_valid.astype(np.uint8),
                expected_shape=shape,
            )

    def test_builder_report_requires_exact_method_and_aligned_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(
                json.dumps(
                    {
                        "method": volume.BUILDER_METHOD,
                        "representation": volume.BUILDER_REPRESENTATION,
                        "metadata": {"frames": 3, "width": 4, "height": 2},
                    }
                )
            )
            loaded = volume._builder_report(
                path,
                frame_count=3,
                width=4,
                height=2,
            )
            self.assertEqual(loaded["method"], volume.BUILDER_METHOD)

            path.write_text(
                json.dumps(
                    {
                        "method": "wrong",
                        "representation": volume.BUILDER_REPRESENTATION,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "builder method"):
                volume._builder_report(
                    path,
                    frame_count=3,
                    width=4,
                    height=2,
                )

            path.write_text(
                json.dumps(
                    {
                        "method": volume.BUILDER_METHOD,
                        "representation": volume.BUILDER_REPRESENTATION,
                        "metadata": {"frames": 4, "width": 4, "height": 2},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "frames"):
                volume._builder_report(
                    path,
                    frame_count=3,
                    width=4,
                    height=2,
                )


class MeshVolumeClassificationTests(unittest.TestCase):
    def _classify(self, depth, shell, *, pose_valid=True, **overrides):
        depth = np.asarray(depth, dtype=np.float32)
        shape = depth.shape
        inputs = {
            "hand_mask": np.ones(shape, dtype=bool),
            "robot_depth": depth,
            "object_support_mask": np.ones(shape, dtype=bool),
            "mesh_mask": np.ones(shape, dtype=bool),
            "front_depth": np.ones(shape, dtype=np.float32),
            "back_depth": np.full(shape, 1.2, dtype=np.float32),
            "pose_valid": pose_valid,
            "shell_m": np.broadcast_to(
                np.asarray(shell, dtype=np.float32), shape
            ).copy(),
        }
        inputs.update(overrides)
        return volume.classify_mesh_volume(**inputs)

    def test_front_intersecting_and_behind_are_an_exact_partition(self):
        classification, support = self._classify(
            [[0.80, 0.95, 1.10, 1.20, 1.30]],
            [[0.10, 0.10, 0.00, 0.00, 0.00]],
        )

        np.testing.assert_array_equal(
            classification,
            [[
                volume.CLASS_FRONT_OF,
                volume.CLASS_INTERSECTING,
                volume.CLASS_INTERSECTING,
                volume.CLASS_FULLY_BEHIND,
                volume.CLASS_FULLY_BEHIND,
            ]],
        )
        self.assertTrue(support.all())
        masks = [classification == value for value in (1, 2, 3)]
        np.testing.assert_array_equal(masks[0] | masks[1] | masks[2], support)
        self.assertFalse(np.any(masks[0] & masks[1]))
        self.assertFalse(np.any(masks[0] & masks[2]))
        self.assertFalse(np.any(masks[1] & masks[2]))

    def test_hidden_is_exactly_intersecting_or_fully_behind(self):
        classification = np.asarray([[0, 1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(
            volume.hidden_from_classification(classification),
            [[False, False, True, True]],
        )

    def test_front_mode_equals_zero_shell_volume_union(self):
        depth = np.asarray([[0.90, 1.00, 1.01, 1.19, 1.20]], np.float32)
        classification, support = self._classify(
            depth,
            np.zeros_like(depth),
        )

        front_hidden = volume.front_only_hidden(
            support=support,
            robot_depth=depth,
            front_depth=np.ones_like(depth),
        )

        np.testing.assert_array_equal(
            front_hidden,
            volume.hidden_from_classification(classification),
        )
        np.testing.assert_array_equal(
            front_hidden,
            [[False, False, True, True, True]],
        )

    def test_collapsed_front_back_equality_belongs_only_to_front(self):
        depth = np.asarray([[1.0, 1.0001]], dtype=np.float32)
        front = np.asarray([[1.0, 1.0]], dtype=np.float32)
        back = np.asarray([[1.0, 1.0]], dtype=np.float32)
        classification, support = self._classify(
            depth,
            np.zeros_like(depth),
            front_depth=front,
            back_depth=back,
        )

        np.testing.assert_array_equal(
            classification,
            [[volume.CLASS_FRONT_OF, volume.CLASS_FULLY_BEHIND]],
        )
        np.testing.assert_array_equal(classification > 0, support)

    def test_positive_shell_only_adds_to_zero_shell_barrier(self):
        depth = np.asarray([[0.94, 0.98, 1.01]], dtype=np.float32)
        zero, _ = self._classify(depth, np.zeros_like(depth))
        shell, _ = self._classify(depth, np.full_like(depth, 0.07))
        zero_hidden = volume.hidden_from_classification(zero)
        shell_hidden = volume.hidden_from_classification(shell)

        self.assertFalse(np.any(zero_hidden & ~shell_hidden))
        np.testing.assert_array_equal(zero_hidden, [[False, False, True]])
        np.testing.assert_array_equal(shell_hidden, [[True, True, True]])

    def test_back_depth_changes_state_but_not_the_hidden_union(self):
        depth = np.asarray([[1.10]], dtype=np.float32)
        intersecting, _ = self._classify(depth, [[0.0]])
        behind, _ = self._classify(
            depth,
            [[0.0]],
            back_depth=np.asarray([[1.05]], dtype=np.float32),
        )

        self.assertEqual(intersecting.item(), volume.CLASS_INTERSECTING)
        self.assertEqual(behind.item(), volume.CLASS_FULLY_BEHIND)
        self.assertTrue(volume.hidden_from_classification(intersecting).item())
        self.assertTrue(volume.hidden_from_classification(behind).item())

    def test_support_is_mesh_texture_pose_hand_and_finite_depth_intersection(self):
        depth = np.full((1, 6), 1.1, dtype=np.float32)
        hand = np.ones_like(depth, dtype=bool)
        texture = np.ones_like(hand)
        mesh = np.ones_like(hand)
        hand[0, 1] = False
        texture[0, 2] = False
        mesh[0, 3] = False
        depth[0, 4] = np.nan
        classification, support = self._classify(
            depth,
            np.zeros_like(depth),
            hand_mask=hand,
            object_support_mask=texture,
            mesh_mask=mesh,
        )

        np.testing.assert_array_equal(
            support,
            [[True, False, False, False, False, True]],
        )
        np.testing.assert_array_equal(classification > 0, support)

        invalid_pose, invalid_support = self._classify(
            np.asarray([[1.1]], np.float32),
            [[0.0]],
            pose_valid=False,
        )
        self.assertFalse(invalid_pose.any())
        self.assertFalse(invalid_support.any())

    def test_baseline_is_preserved_and_cannot_escape_xhand(self):
        baseline = np.asarray([[True, False, False]], dtype=bool)
        hidden = np.asarray([[False, True, False]], dtype=bool)
        hand = np.asarray([[True, True, False]], dtype=bool)
        combined = volume.combine_with_baseline(baseline, hidden, hand)

        np.testing.assert_array_equal(combined, [[True, True, False]])
        np.testing.assert_array_equal(combined[baseline], np.ones(1, dtype=bool))

        with self.assertRaisesRegex(ValueError, "baseline.*escaped"):
            volume.combine_with_baseline(
                np.asarray([[False, False, True]], dtype=bool),
                hidden,
                hand,
            )

    def test_temporal_eligibility_keeps_valid_support_and_vetoes_clear_front(self):
        support = np.ones((1, 4), dtype=bool)
        depth = np.asarray([[0.80, 0.97, 1.00, 1.10]], dtype=np.float32)
        front = np.ones_like(depth)
        shell = np.asarray([[0.05, 0.01, 0.00, 0.00]], dtype=np.float32)

        eligible = volume.mesh_temporal_eligibility(
            classification_support=support,
            robot_depth=depth,
            front_depth=front,
            shell_m=shell,
            front_slack_m=0.015,
        )

        np.testing.assert_array_equal(eligible, [[False, False, True, True]])

    def test_invalid_shell_is_rejected(self):
        for shell in (
            np.asarray([[-0.1]], np.float32),
            np.asarray([[np.nan]], np.float32),
        ):
            with self.subTest(shell=shell.item()):
                with self.assertRaisesRegex(ValueError, "shell"):
                    self._classify(np.asarray([[1.0]], np.float32), shell)


class MeshVolumeCliContractTests(unittest.TestCase):
    def test_required_cli_and_output_names_are_stable(self):
        parser = volume._build_parser()
        args = parser.parse_args(
            [
                "--background", "background.mp4",
                "--raw_video", "raw.mp4",
                "--overlay_dir", "overlay",
                "--mesh_dir", "mesh",
                "--object_support_mask", "support.npy",
                "--object_restore_mask", "restore.npy",
                "--baseline_mask", "baseline.npy",
                "--mode", "volume",
                "--out_dir", "output",
            ]
        )
        self.assertEqual(args.mode, "volume")
        self.assertEqual(volume.METHOD, "visual_xhand_mesh_volume_barrier")
        self.assertEqual(volume.FRONT_DEPTH_NAME, "object_mesh_front_depth.npy")
        self.assertEqual(volume.BACK_DEPTH_NAME, "object_mesh_back_depth.npy")
        self.assertEqual(volume.MESH_MASK_NAME, "object_mesh_mask.npy")
        self.assertEqual(volume.POSE_VALID_NAME, "pose_valid.npy")

    @staticmethod
    def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("test video writer did not open")
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()

    def test_small_end_to_end_run_publishes_atomic_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_count, height, width, fps = 3, 24, 32, 10.0
            shape = (frame_count, height, width)
            background = root / "background.mp4"
            raw = root / "raw.mp4"
            video_frames = [
                np.full((height, width, 3), 20 + index * 5, dtype=np.uint8)
                for index in range(frame_count)
            ]
            self._write_video(background, video_frames, fps)
            self._write_video(raw, video_frames, fps)

            overlay = root / "overlay"
            overlay.mkdir()
            hand = np.zeros(shape, dtype=bool)
            hand[:, 8:16, 8:24] = True
            robot_depth = np.full(shape, np.inf, dtype=np.float32)
            robot_depth[0, hand[0]] = 0.90
            robot_depth[1, hand[1]] = 1.10
            robot_depth[2, hand[2]] = 1.30
            labels = np.zeros(shape, dtype=np.uint8)
            labels[hand] = 1
            robot_rgb = np.zeros(shape + (3,), dtype=np.uint8)
            robot_rgb[hand] = (80, 120, 180)
            np.save(overlay / "robot_rgb.npy", robot_rgb)
            np.save(overlay / "robot_depth.npy", robot_depth)
            np.save(overlay / "robot_mask.npy", hand)
            np.save(overlay / "robot_hand_mask.npy", hand)
            np.save(overlay / "robot_finger_labels.npy", labels)

            mesh_dir = root / "mesh"
            mesh_dir.mkdir()
            mesh_mask = hand.copy()
            front = np.zeros(shape, dtype=np.float16)
            back = np.zeros(shape, dtype=np.float16)
            front[mesh_mask] = np.float16(1.0)
            back[mesh_mask] = np.float16(1.2)
            np.save(mesh_dir / volume.FRONT_DEPTH_NAME, front)
            np.save(mesh_dir / volume.BACK_DEPTH_NAME, back)
            np.save(mesh_dir / volume.MESH_MASK_NAME, mesh_mask)
            np.save(
                mesh_dir / volume.POSE_VALID_NAME,
                np.ones(frame_count, dtype=bool),
            )
            (mesh_dir / "report.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "method": volume.BUILDER_METHOD,
                        "representation": volume.BUILDER_REPRESENTATION,
                        "frames": frame_count,
                        "width": width,
                        "height": height,
                        "metric_collision_guarantee": False,
                    }
                )
            )
            support_path = root / "support.npy"
            restore_path = root / "restore.npy"
            baseline_path = root / "baseline.npy"
            np.save(support_path, mesh_mask)
            np.save(restore_path, np.zeros(shape, dtype=bool))
            baseline = np.zeros(shape, dtype=bool)
            baseline[:, 8, 8] = True
            np.save(baseline_path, baseline)
            output = root / "output"

            argv = [
                "composite_xhand_mesh_volume.py",
                "--background", str(background),
                "--raw_video", str(raw),
                "--overlay_dir", str(overlay),
                "--mesh_dir", str(mesh_dir),
                "--object_support_mask", str(support_path),
                "--object_restore_mask", str(restore_path),
                "--baseline_mask", str(baseline_path),
                "--mode", "front",
                "--out_dir", str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                volume.main()

            required = (
                "video_overlay_mesh_volume.mp4",
                "video_robot_only_mesh_volume.mp4",
                "debug_mesh_volume.mp4",
                "occluded_hand_mask.npy",
                "mesh_volume_classification.npy",
                "mesh_volume_evidence.npz",
                "report.json",
            )
            self.assertTrue(
                all((output / name).is_file() for name in required)
            )
            final = np.load(output / "occluded_hand_mask.npy")
            classification = np.load(
                output / "mesh_volume_classification.npy"
            )
            self.assertTrue(np.all(final[baseline]))
            self.assertFalse(np.any(final & ~hand))
            self.assertTrue(
                np.all(classification[0, hand[0]] == volume.CLASS_FRONT_OF)
            )
            self.assertTrue(
                np.all(
                    classification[1, hand[1]]
                    == volume.CLASS_INTERSECTING
                )
            )
            self.assertTrue(
                np.all(
                    classification[2, hand[2]]
                    == volume.CLASS_FULLY_BEHIND
                )
            )
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["method"], volume.METHOD)
            self.assertEqual(report["mode"], "front")
            self.assertEqual(
                report["counts"]["final_occluded_pixels"], int(final.sum())
            )
            self.assertEqual(
                report["counts"]["residual_violation_pixels"], 0
            )
            self.assertTrue(
                all(report["invariants"].values()), report["invariants"]
            )


if __name__ == "__main__":
    unittest.main()
