"""Tests for XHand-thickness strategy comparison and diagnostic shell."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INPAINTING_DIR = REPO_ROOT / "src" / "inpainting"
sys.path.insert(0, str(INPAINTING_DIR))

import compare_xhand_thickness_strategies as strategies  # noqa: E402


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


class XHandThicknessStrategiesTests(unittest.TestCase):
    def _build_fixture(self, root: Path) -> dict[str, object]:
        directories = {
            role: root / role
            for role in (
                "baseline",
                "half_thickness",
                "full_thickness",
                "visibility_force",
            )
        }
        overlay_dir = root / "overlay"
        for directory in (*directories.values(), overlay_dir):
            directory.mkdir()

        frame_count, height, width, fps = 3, 30, 40, 12.0
        background_path = root / "background.mp4"
        background_frames = [
            np.full(
                (height, width, 3),
                (15 + frame_index * 10, 50, 80),
                dtype=np.uint8,
            )
            for frame_index in range(frame_count)
        ]
        _write_video(background_path, background_frames, fps)

        labels = np.zeros((frame_count, height, width), dtype=np.uint8)
        labels[:, 12:18, 8:25] = 1
        surface_labels = np.zeros_like(labels)
        surface_labels[:, 12:14, 8:25] = 1  # thumb palmar/front
        surface_labels[:, 14:16, 8:25] = 2  # thumb lateral/side
        surface_labels[:, 16:18, 8:25] = 3  # thumb dorsal/back
        object_mask = np.zeros((frame_count, height, width), dtype=bool)
        object_mask[:, 10:21, 14:18] = True
        force = (labels == 1) & object_mask
        baseline = np.zeros_like(force)
        baseline[:, 13:17, 14:16] = True
        half = baseline | ((labels == 1) & object_mask & (np.indices(force.shape)[2] < 17))
        full = force.copy()
        masks = {
            "baseline": baseline,
            "half_thickness": half,
            "full_thickness": full,
            "visibility_force": force,
        }

        np.save(overlay_dir / "robot_finger_labels.npy", labels)
        surface_labels_path = overlay_dir / strategies.SURFACE_LABELS_NAME
        np.save(surface_labels_path, surface_labels)
        object_mask_path = root / "object_mask.npy"
        np.save(object_mask_path, object_mask)
        object_restore_mask = object_mask.copy()
        object_restore_mask[:, :, 17] = False
        object_restore_mask_path = root / "object_restore_mask.npy"
        np.save(object_restore_mask_path, object_restore_mask)

        source_frames = [
            np.full(
                (height, width, 3),
                (100, 30 + frame_index * 20, 160),
                dtype=np.uint8,
            )
            for frame_index in range(frame_count)
        ]
        for role, directory in directories.items():
            video_name = (
                strategies.FORCE_VIDEO_NAME
                if role == "visibility_force"
                else strategies.FINAL_VIDEO_NAME
            )
            mask_name = (
                strategies.FORCE_MASK_NAME
                if role == "visibility_force"
                else strategies.MASK_NAME
            )
            _write_video(directory / video_name, source_frames, fps)
            np.save(directory / mask_name, masks[role])
            per_frame = masks[role].sum(axis=(1, 2)).astype(int).tolist()
            common = {
                "schema_version": 1,
                "frames": frame_count,
                "width": width,
                "height": height,
                "fps": fps,
                "finger_names": list(strategies.DEFAULT_FINGER_NAMES),
                "sources": {
                    "background": str(background_path),
                    "raw_video": str(background_path),
                    "overlay_dir": str(overlay_dir),
                    "object_mask": str(object_mask_path),
                },
            }
            if role == "visibility_force":
                common.update(
                    {
                        "output_modes": ["visibility"],
                        "mode_statistics": {
                            "visibility": {
                                "pixels": int(masks[role].sum()),
                                "frames": int(
                                    masks[role].any(axis=(1, 2)).sum()
                                ),
                                "occluded_pixel_count": per_frame,
                            }
                        },
                    }
                )
            else:
                common.update(
                    {
                        "occluded_pixels_total": int(masks[role].sum()),
                        "frames_with_occlusion": int(
                            masks[role].any(axis=(1, 2)).sum()
                        ),
                        "occluded_pixel_count": per_frame,
                    }
                )
                if role != "baseline":
                    scale = 0.5 if role == "half_thickness" else 1.0
                    common["config"] = {
                        "contact_depth_thickness_scale": scale
                    }
                    common["xhand_contact_depth_bias"] = {
                        "enabled": True,
                        "scale": scale,
                        "metric_object_depth_gate_modified": False,
                    }
            (directory / strategies.INPUT_REPORT_NAME).write_text(
                json.dumps(common)
            )
        return {
            "directories": directories,
            "overlay_dir": overlay_dir,
            "object_mask_path": object_mask_path,
            "object_restore_mask_path": object_restore_mask_path,
            "background_path": background_path,
            "masks": masks,
            "labels": labels,
            "surface_labels": surface_labels,
            "surface_labels_path": surface_labels_path,
            "frame_count": frame_count,
            "height": height,
            "width": width,
            "fps": fps,
        }

    def test_seed_component_shell_is_semantic_and_area_capped(self):
        height, width = 24, 36
        labels = np.zeros((height, width), dtype=np.uint8)
        labels[8:14, 3:13] = 1
        labels[8:14, 24:34] = 1
        object_mask = np.zeros((height, width), dtype=bool)
        object_mask[6:17, 5:31] = True
        seed = np.zeros_like(object_mask)
        seed[9:13, 6:9] = True
        union = seed.copy()
        config = strategies.SafetyShellConfig(
            min_radius_px=3,
            max_radius_px=3,
            temporal_median_window=3,
            added_area_cap_fraction=0.75,
        )

        shell, details = strategies.build_safety_shell_frame(
            force_seed=seed,
            union_mask=union,
            finger_labels=labels,
            object_mask=object_mask,
            smoothed_radii=np.array([3], dtype=np.int16),
            config=config,
        )

        self.assertGreater(int(shell.sum()), 0)
        self.assertFalse(shell[:, 24:34].any())
        self.assertFalse((shell & (labels != 1)).any())
        self.assertFalse((shell & union).any())
        cap = math.ceil(int(seed.sum()) * 0.75)
        self.assertLessEqual(int(shell.sum()), cap)
        self.assertEqual(details[0]["cap_pixels"], cap)
        self.assertTrue(details[0]["seed_connected"])

    def test_temporal_median_does_not_bridge_inactive_gaps(self):
        raw = np.array(
            [[2.0], [10.0], [np.nan], [20.0], [4.0]],
            dtype=np.float32,
        )
        smoothed = strategies.temporal_median_radii(
            raw,
            window=3,
            min_radius_px=0,
            max_radius_px=30,
        )
        np.testing.assert_array_equal(
            smoothed[:, 0], np.array([6, 6, 0, 12, 12])
        )

    def test_surface_strategy_selects_front_zero_side_half_back_full(self):
        surfaces = np.array([[1, 1, 2, 2, 3, 3]], dtype=np.uint8)
        baseline = np.array([[1, 0, 1, 0, 1, 0]], dtype=bool)
        half = np.array([[0, 1, 0, 1, 0, 1]], dtype=bool)
        full = np.ones_like(baseline)

        side_half, weighted = strategies.build_surface_strategy_frame(
            baseline_mask=baseline,
            half_thickness_mask=half,
            full_thickness_mask=full,
            surface_ids=surfaces,
        )

        np.testing.assert_array_equal(
            side_half,
            np.array([[1, 0, 0, 1, 1, 0]], dtype=bool),
        )
        np.testing.assert_array_equal(
            weighted,
            np.array([[1, 0, 0, 1, 1, 1]], dtype=bool),
        )

    def test_packed_surface_labels_must_decode_to_finger_labels(self):
        packed = np.array([[0, 1, 2, 3, 4]], dtype=np.uint8)
        decoded, surfaces = strategies._decode_packed_surface_labels(
            packed,
            finger_count=2,
        )
        np.testing.assert_array_equal(
            decoded,
            np.array([[0, 1, 1, 1, 2]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            surfaces,
            np.array([[0, 1, 2, 3, 1]], dtype=np.uint8),
        )
        with self.assertRaisesRegex(ValueError, "decode to robot finger labels"):
            strategies._validate_surface_finger_alignment(
                decoded,
                np.array([[0, 1, 1, 1, 1]], dtype=np.uint8),
            )

    def test_builds_atomic_union_shell_and_3x2_comparison(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = self._build_fixture(root)
            directories = fixture["directories"]
            output_dir = root / "comparison"
            output_dir.mkdir()
            (output_dir / "stale.txt").write_text("old")

            result = strategies.build_comparison(
                directories["baseline"],
                directories["half_thickness"],
                directories["full_thickness"],
                directories["visibility_force"],
                overlay_dir=fixture["overlay_dir"],
                object_mask=fixture["object_mask_path"],
                object_restore_mask=fixture["object_restore_mask_path"],
                output_dir=output_dir,
                overwrite=True,
                shell_config=strategies.SafetyShellConfig(
                    min_radius_px=3,
                    max_radius_px=3,
                    temporal_median_window=3,
                    added_area_cap_fraction=0.75,
                ),
            )

            self.assertFalse((output_dir / "stale.txt").exists())
            for name in (
                strategies.UNION_MASK_NAME,
                strategies.UNION_VIDEO_NAME,
                strategies.SHELL_ADDED_MASK_NAME,
                strategies.SHELL_MASK_NAME,
                strategies.SHELL_VIDEO_NAME,
                strategies.SHELL_EVIDENCE_NAME,
                strategies.OUTPUT_VIDEO_NAME,
                strategies.SURFACE_LATERAL_MASK_NAME,
                strategies.SURFACE_WEIGHTED_MASK_NAME,
                strategies.SURFACE_LATERAL_VIDEO_NAME,
                strategies.SURFACE_WEIGHTED_VIDEO_NAME,
                strategies.SURFACE_DEBUG_VIDEO_NAME,
                strategies.SURFACE_COMPARISON_VIDEO_NAME,
                strategies.OUTPUT_REPORT_NAME,
            ):
                self.assertTrue((output_dir / name).is_file(), name)

            union = np.load(output_dir / strategies.UNION_MASK_NAME)
            shell_added = np.load(
                output_dir / strategies.SHELL_ADDED_MASK_NAME
            )
            shell = np.load(output_dir / strategies.SHELL_MASK_NAME)
            expected_union = (
                fixture["masks"]["baseline"]
                | fixture["masks"]["visibility_force"]
            )
            np.testing.assert_array_equal(union, expected_union)
            np.testing.assert_array_equal(shell, union | shell_added)
            self.assertGreater(int(shell_added.sum()), 0)
            self.assertFalse(
                (shell_added & (fixture["labels"] == 0)).any()
            )
            surface_ids = fixture["surface_labels"]
            expected_surface_lateral = (
                fixture["masks"]["baseline"] & (surface_ids != 2)
            ) | (fixture["masks"]["half_thickness"] & (surface_ids == 2))
            expected_surface_weighted = (
                (fixture["masks"]["baseline"] & (surface_ids == 1))
                | (
                    fixture["masks"]["half_thickness"]
                    & (surface_ids == 2)
                )
                | (
                    fixture["masks"]["full_thickness"]
                    & (surface_ids == 3)
                )
            )
            np.testing.assert_array_equal(
                np.load(output_dir / strategies.SURFACE_LATERAL_MASK_NAME),
                expected_surface_lateral,
            )
            np.testing.assert_array_equal(
                np.load(output_dir / strategies.SURFACE_WEIGHTED_MASK_NAME),
                expected_surface_weighted,
            )

            metadata = strategies.probe_video(
                output_dir / strategies.OUTPUT_VIDEO_NAME
            )
            self.assertEqual(
                (metadata.width, metadata.height, metadata.frames),
                (
                    int(fixture["width"]) * 3,
                    int(fixture["height"]) * 2,
                    int(fixture["frame_count"]),
                ),
            )
            self.assertEqual(
                result["panel_layout"],
                [
                    ["baseline", "half_thickness", "full_thickness"],
                    [
                        "visibility_force",
                        "baseline_force_union",
                        "union_safety_shell_diagnostic",
                    ],
                ],
            )
            self.assertEqual(
                result["surface_panel_layout"],
                [
                    ["surface_labels_debug", "surface_front_baseline"],
                    [
                        "surface_front_side_half",
                        "surface_front_side_half_back_full",
                    ],
                ],
            )
            surface_metadata = strategies.probe_video(
                output_dir / strategies.SURFACE_COMPARISON_VIDEO_NAME
            )
            self.assertEqual(
                (
                    surface_metadata.width,
                    surface_metadata.height,
                    surface_metadata.frames,
                ),
                (
                    int(fixture["width"]) * 2,
                    int(fixture["height"]) * 2,
                    int(fixture["frame_count"]),
                ),
            )
            self.assertTrue(
                result["invariants"]["all_masks_are_semantic_finger_only"]
            )
            self.assertGreater(
                result["safety_shell"]["total_added_pixels"], 0
            )
            self.assertEqual(
                set(result["mode_statistics"]), set(strategies.MODE_NAMES)
            )
            self.assertEqual(
                set(result["pairwise"]),
                {name for name, _, _ in strategies.PAIR_SPECS},
            )
            self.assertEqual(
                set(result["surface_strategy_statistics"]),
                set(strategies.SURFACE_STRATEGY_NAMES),
            )
            self.assertEqual(
                set(result["surface_pairwise"]),
                {name for name, _, _ in strategies.SURFACE_PAIR_SPECS},
            )
            self.assertEqual(
                result["sources"]["finger_surface_labels"],
                str(fixture["surface_labels_path"]),
            )
            self.assertEqual(
                result["sources"]["object_restore_mask"],
                str(fixture["object_restore_mask_path"]),
            )
            self.assertTrue(
                result["invariants"][
                    "object_restore_mask_subset_of_modal_object"
                ]
            )

    def test_disagreeing_report_backgrounds_preserve_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = self._build_fixture(root)
            directories = fixture["directories"]
            alternate = root / "alternate_background.mp4"
            _write_video(
                alternate,
                [
                    np.zeros(
                        (
                            int(fixture["height"]),
                            int(fixture["width"]),
                            3,
                        ),
                        dtype=np.uint8,
                    )
                    for _ in range(int(fixture["frame_count"]))
                ],
                float(fixture["fps"]),
            )
            report_path = (
                directories["full_thickness"] / strategies.INPUT_REPORT_NAME
            )
            report = json.loads(report_path.read_text())
            report["sources"]["background"] = str(alternate)
            report_path.write_text(json.dumps(report))
            output_dir = root / "comparison"
            output_dir.mkdir()
            sentinel = output_dir / "keep.txt"
            sentinel.write_text("preserved")

            with self.assertRaisesRegex(ValueError, "disagree on background"):
                strategies.build_comparison(
                    directories["baseline"],
                    directories["half_thickness"],
                    directories["full_thickness"],
                    directories["visibility_force"],
                    overlay_dir=fixture["overlay_dir"],
                    object_mask=fixture["object_mask_path"],
                    output_dir=output_dir,
                    overwrite=True,
                )
            self.assertEqual(sentinel.read_text(), "preserved")

    def test_surface_override_rejects_finger_mismatch_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = self._build_fixture(root)
            directories = fixture["directories"]
            invalid = np.array(fixture["surface_labels"], copy=True)
            invalid[0, 12, 8] = 4  # decodes as finger 2 over finger-label 1
            override_path = root / "surface_override.npy"
            np.save(override_path, invalid)
            output_dir = root / "comparison"
            output_dir.mkdir()
            sentinel = output_dir / "keep.txt"
            sentinel.write_text("preserved")

            with self.assertRaisesRegex(
                ValueError,
                "do not decode to robot finger labels",
            ):
                strategies.build_comparison(
                    directories["baseline"],
                    directories["half_thickness"],
                    directories["full_thickness"],
                    directories["visibility_force"],
                    overlay_dir=fixture["overlay_dir"],
                    surface_labels=override_path,
                    object_mask=fixture["object_mask_path"],
                    output_dir=output_dir,
                    overwrite=True,
                )
            self.assertEqual(sentinel.read_text(), "preserved")

    def test_existing_output_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = self._build_fixture(root)
            directories = fixture["directories"]
            output_dir = root / "comparison"
            output_dir.mkdir()
            sentinel = output_dir / "keep.txt"
            sentinel.write_text("preserved")

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                strategies.build_comparison(
                    directories["baseline"],
                    directories["half_thickness"],
                    directories["full_thickness"],
                    directories["visibility_force"],
                    overlay_dir=fixture["overlay_dir"],
                    object_mask=fixture["object_mask_path"],
                    output_dir=output_dir,
                )
            self.assertEqual(sentinel.read_text(), "preserved")

    def test_rejects_swapped_or_mislabeled_thickness_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = self._build_fixture(root)
            directories = fixture["directories"]
            report_path = (
                directories["half_thickness"] / strategies.INPUT_REPORT_NAME
            )
            report = json.loads(report_path.read_text())
            report["config"]["contact_depth_thickness_scale"] = 1.0
            report["xhand_contact_depth_bias"]["scale"] = 1.0
            report_path.write_text(json.dumps(report))

            with self.assertRaisesRegex(ValueError, "expected 0.5"):
                strategies.build_comparison(
                    directories["baseline"],
                    directories["half_thickness"],
                    directories["full_thickness"],
                    directories["visibility_force"],
                    overlay_dir=fixture["overlay_dir"],
                    object_mask=fixture["object_mask_path"],
                    output_dir=root / "comparison",
                )


if __name__ == "__main__":
    unittest.main()
