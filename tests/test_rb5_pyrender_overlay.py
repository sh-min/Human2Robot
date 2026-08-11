"""Focused, renderer-independent checks for the RB5 pyrender contract."""

from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INPAINTING_DIR = REPO_ROOT / "src" / "inpainting"
sys.path.insert(0, str(INPAINTING_DIR))

import render_rb5_pyrender_overlay as renderer  # noqa: E402


class FingerLinkLabelTests(unittest.TestCase):
    def test_fixed_tips_are_part_of_the_finger_segmentation(self):
        labels = renderer.finger_link_labels("left")

        self.assertEqual(len(labels), 17)
        self.assertEqual(labels["left_hand_thumb_rota_tip"], 1)
        self.assertEqual(labels["left_hand_index_rota_tip"], 2)
        self.assertEqual(labels["left_hand_mid_tip"], 3)
        self.assertEqual(labels["left_hand_ring_tip"], 4)
        self.assertEqual(labels["left_hand_pinky_tip"], 5)
        self.assertEqual(Counter(labels.values()), {1: 4, 2: 4, 3: 3, 4: 3, 5: 3})


class ArmLinkSelectionTests(unittest.TestCase):
    def test_full_mode_includes_every_rb5_link(self):
        self.assertEqual(
            renderer.arm_link_names("full"),
            tuple(f"link{index}" for index in range(7)),
        )

    def test_distal_mode_keeps_only_last_three_links(self):
        self.assertEqual(
            renderer.arm_link_names("distal"),
            ("link4", "link5", "link6"),
        )

    def test_forearm_mode_keeps_connected_last_four_links(self):
        self.assertEqual(
            renderer.arm_link_names("forearm"),
            ("link3", "link4", "link5", "link6"),
        )

    def test_wrist_mode_keeps_only_final_link(self):
        self.assertEqual(renderer.arm_link_names("wrist"), ("link6",))

    def test_hand_only_mode_hides_every_rb5_mesh(self):
        self.assertEqual(renderer.arm_link_names("hand_only"), ())

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported arm mode"):
            renderer.arm_link_names("unknown")


class FrameRangeTests(unittest.TestCase):
    def test_preview_defaults_to_one_frame(self):
        self.assertEqual(list(renderer.resolve_frame_range(10, 4, 0, True)), [4])

    def test_preview_range_is_clipped_to_input(self):
        self.assertEqual(list(renderer.resolve_frame_range(10, 8, 9, True)), [8, 9])

    def test_full_render_rejects_partial_arrays(self):
        with self.assertRaisesRegex(ValueError, "partial array"):
            renderer.resolve_frame_range(10, 3, 2, False)
        self.assertEqual(list(renderer.resolve_frame_range(10, 0, 0, False)), list(range(10)))


class CoordinateTests(unittest.TestCase):
    def test_cv_root_pose_maps_forward_z_to_opengl_minus_z(self):
        transform = np.eye(4)
        transform[:3, 3] = (0.25, 0.5, 1.0)

        pose = renderer._root_pose_cv_to_gl(transform)

        np.testing.assert_allclose(pose[:3, 3], (0.25, -0.5, -1.0))
        np.testing.assert_allclose(pose[:3, :3], np.diag((1.0, -1.0, -1.0)))

    def test_focal_scales_with_render_resolution(self):
        data = {
            "img_width": np.asarray(1280),
            "img_height": np.asarray(720),
            "img_focal": np.asarray(900.0),
        }

        result = renderer.resolve_image_size(data, None, None, None, 0.5)

        self.assertEqual(result[:4], (1280, 720, 640, 360))
        self.assertAlmostEqual(result[4], 450.0)


class FingerSurfaceClassificationTests(unittest.TestCase):
    @staticmethod
    def _box_labels(side: str, finger: str):
        mesh = renderer.trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        labels = renderer.classify_finger_face_surfaces(
            mesh,
            np.eye(4),
            side,
            finger,
        )
        return mesh, labels

    def test_right_non_thumb_uses_positive_y_as_palmar(self):
        mesh, labels = self._box_labels("right", "index")
        plus_y = np.isclose(mesh.face_normals[:, 1], 1.0)
        minus_y = np.isclose(mesh.face_normals[:, 1], -1.0)

        self.assertTrue(
            np.all(labels[plus_y] == renderer.SURFACE_LABEL_IDS["palmar"])
        )
        self.assertTrue(
            np.all(labels[minus_y] == renderer.SURFACE_LABEL_IDS["dorsal"])
        )
        self.assertEqual(
            Counter(labels.tolist()),
            {
                renderer.SURFACE_LABEL_IDS["palmar"]: 2,
                renderer.SURFACE_LABEL_IDS["lateral"]: 8,
                renderer.SURFACE_LABEL_IDS["dorsal"]: 2,
            },
        )

    def test_left_non_thumb_reverses_palmar_and_dorsal(self):
        mesh, labels = self._box_labels("left", "middle")
        plus_y = np.isclose(mesh.face_normals[:, 1], 1.0)
        minus_y = np.isclose(mesh.face_normals[:, 1], -1.0)

        self.assertTrue(
            np.all(labels[minus_y] == renderer.SURFACE_LABEL_IDS["palmar"])
        )
        self.assertTrue(
            np.all(labels[plus_y] == renderer.SURFACE_LABEL_IDS["dorsal"])
        )

    def test_thumb_uses_negative_z_as_palmar_on_both_sides(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                mesh, labels = self._box_labels(side, "thumb")
                minus_z = np.isclose(mesh.face_normals[:, 2], -1.0)
                plus_z = np.isclose(mesh.face_normals[:, 2], 1.0)
                self.assertTrue(
                    np.all(
                        labels[minus_z]
                        == renderer.SURFACE_LABEL_IDS["palmar"]
                    )
                )
                self.assertTrue(
                    np.all(
                        labels[plus_z]
                        == renderer.SURFACE_LABEL_IDS["dorsal"]
                    )
                )

    def test_visual_rotation_is_applied_before_classification(self):
        # This triangle has mesh-local +Z normal.  A -90 degree X rotation
        # maps it to link-local +Y, the right non-thumb palmar direction.
        mesh = renderer.trimesh.Trimesh(
            vertices=np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0))),
            faces=np.asarray(((0, 1, 2),)),
            process=False,
        )
        transform = renderer._make_transform(
            np.zeros(3),
            np.asarray((-np.pi / 2, 0.0, 0.0)),
        )

        labels = renderer.classify_finger_face_surfaces(
            mesh,
            transform,
            "right",
            "index",
        )

        np.testing.assert_array_equal(
            labels,
            np.asarray((renderer.SURFACE_LABEL_IDS["palmar"],), dtype=np.uint8),
        )

    def test_surface_split_preserves_faces_and_visual_colour(self):
        mesh = renderer.trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        colour = np.asarray((23, 47, 89, 255), dtype=np.uint8)
        mesh.visual.face_colors = np.tile(colour, (len(mesh.faces), 1))

        pieces = renderer.split_finger_mesh_surfaces(
            mesh,
            np.eye(4),
            "right",
            "index",
        )

        self.assertEqual(sum(len(piece.faces) for piece, _, _ in pieces), 12)
        self.assertEqual({surface for _, _, surface in pieces}, {1, 2, 3})
        for piece, _, _ in pieces:
            self.assertTrue(np.all(piece.visual.face_colors == colour))


class PackedFingerSurfaceContractTests(unittest.TestCase):
    def test_hand_segmentation_channel_includes_palm_and_fingers(self):
        palm = renderer.hand_segmentation_color(0, 0)
        finger = renderer.hand_segmentation_color(3, 8)

        np.testing.assert_array_equal(palm, np.asarray((0, 0, 1)))
        np.testing.assert_array_equal(finger, np.asarray((3, 8, 2)))

    def test_pack_and_decode_cover_all_fingers_and_surfaces(self):
        packed = [0]
        expected_fingers = [0]
        for finger in range(1, 6):
            for surface in range(1, 4):
                packed.append(renderer.pack_finger_surface_label(finger, surface))
                expected_fingers.append(finger)

        np.testing.assert_array_equal(
            renderer.decode_packed_finger_labels(
                np.asarray(packed, dtype=np.uint8)
            ),
            np.asarray(expected_fingers, dtype=np.uint8),
        )
        self.assertEqual(packed, list(range(16)))

    def test_validation_rejects_a_finger_mismatch(self):
        packed = np.asarray(((1, 4),), dtype=np.uint8)
        fingers = np.asarray(((1, 1),), dtype=np.uint8)

        with self.assertRaisesRegex(RuntimeError, "1 mismatched pixels"):
            renderer.validate_packed_surface_fingers(packed, fingers)

    def test_backfill_alignment_uses_lateral_when_finger_ids_disagree(self):
        # Render says finger 1 palmar, finger 2 dorsal, then background.
        rendered = np.asarray(((1, 6, 0, 4),), dtype=np.uint8)
        # Existing labels remove pixel 0, change pixel 1 to finger 3, expose
        # pixel 2, and retain pixel 3 as finger 2.
        existing = np.asarray(((0, 3, 5, 2),), dtype=np.uint8)

        aligned, stats = renderer.align_packed_surfaces_to_finger_labels(
            rendered,
            existing,
        )

        # Pixel 1 disagrees on finger identity and pixel 2 has no rendered
        # surface, so both conservatively fall back to lateral=2.
        np.testing.assert_array_equal(
            aligned,
            np.asarray(((0, 8, 14, 4),), dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            renderer.decode_packed_finger_labels(aligned),
            existing,
        )
        self.assertEqual(stats["raster_mismatch_pixels"], 3)
        self.assertEqual(stats["missing_surface_fallback_pixels"], 1)
        self.assertEqual(stats["finger_mismatch_fallback_pixels"], 1)
        self.assertEqual(stats["lateral_fallback_pixels"], 2)

    def test_manifest_documents_surface_order_and_decode(self):
        contract = renderer.finger_surface_manifest_contract()

        self.assertEqual(
            contract["surface_ids"],
            {"palmar": 1, "lateral": 2, "dorsal": 3},
        )
        self.assertEqual(contract["valid_range"], [0, 15])
        self.assertIn("packed_id - 1", contract["decode"]["surface_id"])

    def test_surface_backfill_uses_a_sibling_temp_and_preserves_target(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            fingers = np.asarray([[[0, 1], [5, 0]]], dtype=np.uint8)
            np.save(output_dir / "robot_finger_labels.npy", fingers)
            old_target = output_dir / "robot_finger_surface_labels.npy"
            old_target.write_bytes(b"old surface labels")

            array, temp_path, target_path, existing = (
                renderer._prepare_surface_backfill(
                    output_dir,
                    fingers.shape,
                    overwrite=True,
                )
            )
            try:
                self.assertEqual(target_path, old_target)
                self.assertEqual(old_target.read_bytes(), b"old surface labels")
                self.assertEqual(temp_path.parent, output_dir)
                self.assertNotEqual(temp_path, target_path)
                np.testing.assert_array_equal(existing, fingers)
            finally:
                del array
                temp_path.unlink(missing_ok=True)

    def test_surface_backfill_publish_restores_both_files_on_manifest_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            surface_target = output_dir / "robot_finger_surface_labels.npy"
            manifest_target = output_dir / "manifest.json"
            surface_target.write_bytes(b"old surface")
            manifest_target.write_text('{"old": true}\n')
            surface_temp = output_dir / ".new_surface.npy"
            surface_temp.write_bytes(b"new surface")
            manifest_temp, prepared_target = renderer._prepare_surface_manifest(
                output_dir,
                {"frame_count": 1},
            )
            self.assertEqual(prepared_target, manifest_target)
            real_replace = renderer.os.replace

            def fail_manifest_install(source, target):
                if Path(source) == manifest_temp and Path(target) == manifest_target:
                    raise OSError("injected manifest install failure")
                return real_replace(source, target)

            with mock.patch.object(renderer.os, "replace", side_effect=fail_manifest_install):
                with self.assertRaisesRegex(OSError, "injected manifest"):
                    renderer._publish_surface_backfill(
                        surface_temp,
                        surface_target,
                        manifest_temp,
                        manifest_target,
                    )

            self.assertEqual(surface_target.read_bytes(), b"old surface")
            self.assertEqual(manifest_target.read_text(), '{"old": true}\n')
            self.assertFalse(surface_temp.exists())
            self.assertFalse(manifest_temp.exists())


if __name__ == "__main__":
    unittest.main()
