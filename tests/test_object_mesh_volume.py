"""Focused tests for the nominal MJCF front/back mesh-volume builder."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[1]
INPAINTING_DIR = REPO_ROOT / "src" / "inpainting"
sys.path.insert(0, str(INPAINTING_DIR))

import build_object_mesh_volume as volume  # noqa: E402


MAPPING = REPO_ROOT / "configs" / "objects" / "0804_mesh_volume.json"


class ObjectMeshAssetTests(unittest.TestCase):
    def test_mapping_uses_the_requested_five_nominal_objects(self):
        mapping = volume.load_mapping(MAPPING)

        actual = {
            label: item["object_id"]
            for label, item in mapping["labels"].items()
        }

        self.assertEqual(
            actual,
            {
                "Lock": "lock_box_small",
                "Milk": "milk_carton",
                "Snack": "pringles",
                "Cup": "cup_green",
                "Sweep": "sponge",
            },
        )

    def test_mjcf_parser_excludes_print_geoms_and_preserves_expected_extents(self):
        mapping = volume.load_mapping(MAPPING)
        expected = {
            "Lock": ((26, 20), (0.1304, 0.0904, 0.0550)),
            "Milk": ((4, 1), (0.0550, 0.0550, 0.1150)),
            "Snack": ((2, 1), (0.0650, 0.0650, 0.0900)),
            "Cup": ((33, 0), (0.1138, 0.0800, 0.1050)),
            "Sweep": ((2, 0), (0.1300, 0.0800, 0.0300)),
        }
        for label, ((physical, excluded), extent) in expected.items():
            item = mapping["labels"][label]
            mjcf = volume._load_object_mjcf(item["object_spec"], item["object_id"])
            primitives, stats = volume.parse_mjcf_primitives(mjcf)
            bounds = np.asarray([volume.primitive_mesh(value).bounds for value in primitives])
            actual_extent = bounds[:, 1].max(axis=0) - bounds[:, 0].min(axis=0)
            with self.subTest(label=label):
                self.assertEqual(stats["physical_primitives"], physical)
                self.assertEqual(stats["excluded_visual_only_geoms"], excluded)
                np.testing.assert_allclose(actual_extent, extent, atol=6.0e-4)

    def test_capsule_component_bounds_match_fromto_not_one_metre(self):
        primitive = volume.Primitive(
            kind="capsule",
            center=np.zeros(3),
            rotation=np.eye(3),
            size=np.asarray((0.005,)),
            endpoint_a=np.asarray((-0.04, 0.0, 0.02)),
            endpoint_b=np.asarray((-0.05, 0.0, 0.03)),
        )

        mesh = volume.primitive_mesh(primitive)

        self.assertLess(float(mesh.extents.max()), 0.03)
        np.testing.assert_allclose(mesh.bounds.mean(axis=0), (-0.045, 0.0, 0.025), atol=3e-4)

    def test_voxel_union_is_watertight_and_removes_overlap_interior(self):
        first = volume.Primitive(
            "box", np.asarray((-0.02, 0.0, 0.0)), np.eye(3), np.asarray((0.03, 0.02, 0.02))
        )
        second = volume.Primitive(
            "box", np.asarray((0.02, 0.0, 0.0)), np.eye(3), np.asarray((0.03, 0.02, 0.02))
        )

        mesh, stats = volume.build_watertight_union_mesh(
            [first, second], voxel_pitch_m=0.004
        )

        self.assertTrue(mesh.is_watertight)
        self.assertTrue(stats["watertight"])
        np.testing.assert_allclose(mesh.extents, (0.10, 0.04, 0.04), atol=0.008)
        # The union volume is smaller than the sum only by their overlap and
        # must be much larger than either component alone.
        self.assertGreater(abs(mesh.volume), 0.00012)
        self.assertLess(abs(mesh.volume), 0.00020)

    def test_open_container_cavity_is_not_solid_in_analytic_union(self):
        # Base plus two walls: material points are inside, the open cavity is not.
        primitives = [
            volume.Primitive("box", np.asarray((0.0, 0.0, 0.002)), np.eye(3), np.asarray((0.05, 0.04, 0.002))),
            volume.Primitive("box", np.asarray((-0.048, 0.0, 0.027)), np.eye(3), np.asarray((0.002, 0.04, 0.025))),
            volume.Primitive("box", np.asarray((0.048, 0.0, 0.027)), np.eye(3), np.asarray((0.002, 0.04, 0.025))),
        ]
        points = np.asarray(((0.0, 0.0, 0.03), (0.048, 0.0, 0.03)))
        union_sdf = np.min(
            np.stack([primitive.signed_distance(points) for primitive in primitives]),
            axis=0,
        )

        self.assertGreater(union_sdf[0], 0.0)
        self.assertLessEqual(union_sdf[1], 0.0)


class ObjectMeshPoseTests(unittest.TestCase):
    def test_mask_pose_is_proper_and_places_front_near_observed_depth(self):
        mask = np.zeros((120, 160), dtype=bool)
        mask[30:100, 62:98] = True
        depth = np.zeros(mask.shape, dtype=np.float32)
        depth[mask] = 0.80
        mesh = trimesh.creation.box(extents=(0.04, 0.06, 0.12))
        config = {
            "screen_major_axis": "z",
            "screen_minor_axis": "x",
            "screen_minor_sign": 1,
        }

        pose, evidence = volume.estimate_frame_pose(
            mask,
            depth,
            mesh,
            config,
            focal_px=200.0,
            principal_point=(80.0, 60.0),
        )

        self.assertTrue(evidence["valid"])
        self.assertAlmostEqual(np.linalg.det(pose[:3, :3]), 1.0, places=6)
        transformed = np.asarray(mesh.vertices) @ pose[:3, :3].T + pose[:3, 3]
        self.assertAlmostEqual(float(np.quantile(transformed[:, 2], 0.10)), 0.80, places=5)

    def test_wrist_relative_smoothing_recovers_constant_relation_and_fills_gap(self):
        frame_count = 5
        wrist = np.repeat(np.eye(4)[None], frame_count, axis=0)
        wrist[:, 0, 3] = np.linspace(0.0, 0.04, frame_count)
        relative = np.eye(4)
        relative[:3, 3] = (0.03, -0.01, 0.12)
        raw = np.stack([wrist[index] @ relative for index in range(frame_count)])
        raw[3, :3, 3] += (0.20, -0.15, 0.10)  # excluded outlier
        raw_valid = np.asarray((True, True, False, False, True))

        smoothed, valid, diagnostics = volume.smooth_segment_wrist_relative(
            raw,
            raw_valid,
            wrist,
            np.ones(frame_count, dtype=bool),
            volume.Segment("test", 0, 4),
            observation_blend=0.0,
        )

        self.assertTrue(valid.all())
        self.assertEqual(diagnostics["anchor_frames"], 3)
        for index in range(frame_count):
            np.testing.assert_allclose(
                volume.invert_pose(wrist[index]) @ smoothed[index],
                relative,
                atol=1e-7,
            )

    def test_confidence_does_not_use_contaminated_iou_as_a_hard_veto(self):
        config = volume.load_mapping(MAPPING)["fit"]
        metrics = {
            "median_front_depth_error_m": 0.025,
            "iou": 0.22,
            "mesh_coverage": 0.96,
            "front_back_order_fraction": 0.98,
            "centroid_error_px": 12.0,
        }

        confidence = volume.pose_confidence(metrics, config)

        self.assertGreater(confidence, float(config["minimum_pose_confidence"]))


class ObjectMeshRenderTests(unittest.TestCase):
    def test_normal_and_reversed_winding_render_front_and_back_camera_z(self):
        mesh = trimesh.creation.box(extents=(0.20, 0.20, 0.20))
        pose = np.eye(4)
        pose[2, 3] = 1.0
        renderer = volume.FrontBackRenderer(96, 96, 120.0)
        try:
            renderer.set_mesh("box", mesh)
            front, back, mask, order_fraction = renderer.render(pose)
        finally:
            renderer.close()

        self.assertTrue(mask[48, 48])
        self.assertAlmostEqual(float(front[48, 48]), 0.90, places=3)
        self.assertAlmostEqual(float(back[48, 48]), 1.10, places=3)
        self.assertGreater(order_fraction, 0.95)
        self.assertTrue(np.all(back[mask] >= front[mask]))


if __name__ == "__main__":
    unittest.main()
