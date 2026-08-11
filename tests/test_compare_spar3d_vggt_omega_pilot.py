"""Weight-free adversarial tests for the SPAR3D/VGGT-Omega comparison."""

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
    Path(__file__).parents[1]
    / "scripts"
    / "compare_spar3d_vggt_omega_pilot.py"
)
SPEC = importlib.util.spec_from_file_location(
    "compare_spar3d_vggt_omega_pilot", MODULE_PATH
)
comparison = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def declared(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": comparison.sha256_file(path),
    }


class SyntheticBundle:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.spar_dir = root / "spar3d"
        self.registration_dir = root / "spar3d_registered_mh"
        self.vggt_dir = root / "vggt_omega"
        for directory in (self.spar_dir, self.registration_dir, self.vggt_dir):
            directory.mkdir(parents=True)
        self.selection = {
            "episode": "1",
            "object_label": "Choco",
            "frame_index_basis": "zero_based_decoded_source_frame",
            "mh_frame_index": 187,
            "sh_frame_index": 192,
        }
        self._build_spar()
        self._build_registration()
        self._build_vggt()

    def _build_spar(self) -> None:
        input_rgba = self.spar_dir / "input_rgba.png"
        image = np.zeros((12, 12, 4), dtype=np.uint8)
        image[2:10, 3:9] = (70, 120, 210, 255)
        self.assert_write_image(input_rgba, image)
        manifest = self.spar_dir / "manifest.json"
        write_json(manifest, {"selection": self.selection})
        mesh = self.spar_dir / "mesh.glb"
        points = self.spar_dir / "points.ply"
        mesh.write_bytes(b"synthetic canonical learned mesh")
        points.write_bytes(b"ply\nsynthetic learned points\n")
        self.spar_report = self.spar_dir / "report.json"
        payload = {
            "schema_version": 1,
            "status": "complete",
            "method": comparison.SPAR_METHOD,
            "selection": self.selection,
            "input": declared(input_rgba),
            "input_manifest": {
                "path": str(manifest.resolve()),
                "sha256": comparison.sha256_file(manifest),
            },
            "outputs": {
                "mesh_glb": declared(mesh),
                "points_ply": declared(points),
                "report": str(self.spar_report.resolve()),
            },
            "representation": "learned_single_image_canonical_mesh_and_point_cloud",
            "metric_scale_verified": False,
            "camera_alignment": "none",
            "physical_geometry_guarantee": False,
            "collision_ready": False,
            "warnings": ["Hidden and backside geometry is a learned estimate."],
        }
        write_json(self.spar_report, payload)
        self.canonical_mesh = mesh
        self.input_manifest = manifest

    def _build_registration(self) -> None:
        registered = self.registration_dir / "registered_mesh_mh.glb"
        transform = self.registration_dir / "registration_transform.npz"
        front = self.registration_dir / "registered_front_depth_proxy_m.npy"
        silhouette = self.registration_dir / "registered_silhouette.png"
        turntable = self.registration_dir / "canonical_turntable_contact_sheet.png"
        alignment = self.registration_dir / "mh_silhouette_depth_alignment.png"
        before_after = self.registration_dir / "before_after_registration.png"
        registered.write_bytes(b"synthetic approximately registered mesh")
        np.savez(transform, matrix=np.eye(4, dtype=np.float32))
        np.save(front, np.ones((16, 18), dtype=np.float32))
        visual = np.zeros((180, 280, 3), dtype=np.uint8)
        visual[:, :140] = (50, 80, 180)
        visual[:, 140:] = (70, 170, 80)
        cv2.putText(visual, "synthetic", (54, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for path in (silhouette, turntable, alignment, before_after):
            self.assert_write_image(path, visual)
        mh_image = self.registration_dir / "mh.png"
        mh_mask = self.registration_dir / "mh_mask.png"
        depth = self.registration_dir / "depth.npy"
        depth_params = self.registration_dir / "depth_params.npz"
        self.assert_write_image(mh_image, visual)
        self.assert_write_image(mh_mask, np.full((180, 280), 255, dtype=np.uint8))
        np.save(depth, np.ones((180, 280), dtype=np.float32))
        np.savez(depth_params, anchor=np.float32(1.0))
        self.registration_report = self.registration_dir / "report.json"
        source_report_identity = declared(self.spar_report)
        output_paths = (
            registered,
            transform,
            front,
            silhouette,
            turntable,
            alignment,
            before_after,
        )
        payload = {
            "schema_version": 1,
            "status": "complete",
            "method": comparison.REGISTRATION_METHOD,
            "representation": comparison.REGISTRATION_REPRESENTATION,
            "metric_scale_verified": False,
            "camera_alignment": "approximate_MH_camera_Sim3",
            "physical_geometry_guarantee": False,
            "collision_ready": False,
            "uses_sh_for_this_registration": False,
            "selection": self.selection,
            "source_mesh": declared(self.canonical_mesh),
            "source_spar3d_report": {
                "path": source_report_identity["path"],
                "sha256": source_report_identity["sha256"],
                "source_representation": "learned_single_image_canonical_mesh_and_point_cloud",
                "hidden_geometry_is_learned_estimate": True,
            },
            "sources": {
                "input_manifest": declared(self.input_manifest),
                "mh_image": declared(mh_image),
                "mh_modal_mask": declared(mh_mask),
                "depth_aligned_proxy": declared(depth),
                "depth_anchor_params": declared(depth_params),
            },
            "sim3": {
                "matrix_canonical_to_mh_camera": np.eye(4).tolist(),
                "metric_scale_verified": False,
            },
            "outputs": {path.name: declared(path) for path in output_paths},
        }
        payload["outputs"]["report.json"] = {
            "path": str(self.registration_report.resolve())
        }
        write_json(self.registration_report, payload)
        self.registered_mesh = registered
        self.before_after = before_after
        self.turntable = turntable

    def _build_vggt(self) -> None:
        height, width = 24, 32
        yy, xx = np.mgrid[:height, :width]
        world = np.empty((2, height, width, 3), dtype=np.float32)
        world[:, :, :, 0] = xx / width
        world[:, :, :, 1] = yy / height
        world[0, :, :, 2] = 0.8 + 0.1 * xx / width
        world[1, :, :, 2] = 0.9 + 0.1 * yy / height
        images = np.zeros((2, 3, height, width), dtype=np.float32)
        images[0, 0] = xx / width
        images[0, 1] = yy / height
        images[0, 2] = 0.35
        images[1, 0] = 0.2
        images[1, 1] = xx / width
        images[1, 2] = yy / height
        masks = np.zeros((2, height, width), dtype=bool)
        masks[0, 4:20, 6:26] = True
        masks[1, 3:21, 8:27] = True
        world_path = self.vggt_dir / "world_points_relative.npy"
        images_path = self.vggt_dir / "preprocessed_images.npy"
        masks_path = self.vggt_dir / "object_masks_model_input.npy"
        np.save(world_path, world)
        np.save(images_path, images)
        np.save(masks_path, masks)
        color_grid = np.moveaxis(images, 1, -1)
        color_grid = (color_grid * 255).clip(0, 255).astype(np.uint8)
        view_grid = np.broadcast_to(
            np.arange(2, dtype=np.int16)[:, None, None], masks.shape
        )
        masked_points = world[masks]
        masked_colors = color_grid[masks]
        masked_views = view_grid[masks]

        def save_evidence(path: Path, keep: np.ndarray, confidence_value: float) -> dict:
            points = masked_points[keep]
            colors = masked_colors[keep]
            views = masked_views[keep]
            np.savez_compressed(
                path,
                points_relative=points,
                colors_rgb=colors,
                depth_confidence=np.full(len(points), confidence_value, dtype=np.float32),
                input_depth_relative=np.full(len(points), 1.0, dtype=np.float32),
                view_indices=views,
                camera_order=np.asarray(["MH", "SH"]),
            )
            return {
                "exported_count": int(len(points)),
                "exported_points_by_view": {
                    "MH": int(np.count_nonzero(views == 0)),
                    "SH": int(np.count_nonzero(views == 1)),
                },
                "all_views_contributed": bool(
                    np.any(views == 0) and np.any(views == 1)
                ),
            }

        official_evidence = self.vggt_dir / "object_official_global_p20_evidence.npz"
        custom_evidence = self.vggt_dir / "object_dual_mask_filtered_evidence.npz"
        official_stats = save_evidence(
            official_evidence,
            np.ones(len(masked_points), dtype=bool),
            2.0,
        )
        custom_stats = save_evidence(
            custom_evidence,
            np.arange(len(masked_points)) % 2 == 0,
            5.0,
        )
        official_ply = self.vggt_dir / "object_official_global_p20_relative_point_cloud.ply"
        official_glb = self.vggt_dir / "object_official_global_p20_relative_point_cloud.glb"
        official_full_glb = self.vggt_dir / "official_full_scene_p20_aligned_with_cameras.glb"
        official_ply.write_bytes(b"ply\nsynthetic official-rule p20 points\n")
        official_glb.write_bytes(b"synthetic official-rule p20 point glb")
        official_full_glb.write_bytes(b"synthetic exact official full scene glb")
        object_ply = self.vggt_dir / "object_dual_mask_filtered_relative_point_cloud.ply"
        object_glb = self.vggt_dir / "object_dual_mask_filtered_relative_point_cloud.glb"
        object_ply.write_bytes(b"ply\nsynthetic dual-view object points\n")
        object_glb.write_bytes(b"synthetic dual-view point glb")
        camera_npz = self.vggt_dir / "cameras_relative.npz"
        np.savez(camera_npz, camera_order=np.asarray(["MH", "SH"]))
        view_records = []
        aggregation_masks = {}
        for index, camera in enumerate(("MH", "SH")):
            image_path = self.vggt_dir / f"{camera}_model_input.png"
            source_path = self.vggt_dir / f"{camera}_source.png"
            mask_path = self.vggt_dir / f"{camera}_object_mask.png"
            rgb = np.moveaxis((images[index] * 255).astype(np.uint8), 0, -1)
            self.assert_write_image(image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            self.assert_write_image(source_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            self.assert_write_image(mask_path, masks[index].astype(np.uint8) * 255)
            aggregation_masks[camera] = str(mask_path.resolve())
            view_records.append(
                {
                    "camera": camera,
                    "frame_index": (187, 192)[index],
                    "image": declared(image_path),
                    "source_image": declared(source_path),
                    "object_mask": declared(mask_path),
                }
            )
        vggt_manifest = self.vggt_dir / "manifest.json"
        write_json(vggt_manifest, {"selection": self.selection})
        self.vggt_metadata = self.vggt_dir / "metadata.json"
        artifacts = {
            "cameras": declared(camera_npz),
            "world_points": declared(world_path),
            "preprocessed_images": declared(images_path),
            "object_masks_model_input": declared(masks_path),
            "official_full_scene_glb": declared(official_full_glb),
            "official_object_point_cloud_ply": declared(official_ply),
            "official_object_point_cloud_glb": declared(official_glb),
            "official_object_point_evidence": declared(official_evidence),
            "object_point_cloud_ply": declared(object_ply),
            "object_point_cloud_glb": declared(object_glb),
            "object_point_evidence": declared(custom_evidence),
        }
        official_filter = {
            "policy": "official_rule_global_p20_scoped_to_dual_object_mask_union_v1",
            "confidence_percentile": 20.0,
            "confidence_threshold": 1.5,
            "confidence_population": (
                "finite_dual_object_mask_union_with_depth_edges_zeroed"
            ),
            "per_view_threshold_fallback": False,
            **official_stats,
        }
        custom_filter = {
            "policy": "safe_per_view_adaptive_v1",
            "confidence_percentile": 50.0,
            "confidence_threshold": 4.0,
            "confidence_adaptation": {
                "fallback_views": ["SH"],
                "mode_by_view": {
                    "MH": "global_percentile",
                    "SH": "per_view_percentile_fallback",
                },
            },
            "depth_edge_adaptation": {
                "mode_by_view": {"MH": "initial_rtol", "SH": "initial_rtol"},
                "finite_positive_depth_fallback_views": [],
            },
            **custom_stats,
        }
        payload = {
            "schema_version": 1,
            "status": "completed",
            "input": {
                "manifest": {**declared(vggt_manifest), "verified": True},
                "episode": "1",
                "object_label": "Choco",
                "camera_order": ["MH", "SH"],
                "views": view_records,
            },
            "geometry_contract": {
                "representation": "colored_point_cloud",
                "primitive": "points",
                "has_triangle_faces": False,
                "is_triangle_mesh": False,
                "is_watertight": False,
                "collision_ready": False,
                "coordinate_frame": "vggt_omega_predicted_world_opencv",
                "scale": "relative_non_metric",
                "metric_scale_verified": False,
                "provided_calibration_applied": False,
            },
            "official_reference": {
                "repository": "facebook/VGGT-Omega",
                "local_code_commit": "1" * 40,
                "conversion_function": "visual_util.predictions_to_glb",
                "full_scene_artifact": official_full_glb.name,
                "full_scene_parameters": {
                    "confidence_percentile": 20.0,
                    "depth_edge_filter": True,
                    "depth_edge_rtol": 0.03,
                    "show_cam": True,
                    "mask_black_bg": False,
                    "mask_white_bg": False,
                    "mask_sky": False,
                    "max_points": 300000,
                    "scene_alignment": "official_first_camera_opengl",
                },
                "call_contract": (
                    "predictions_to_glb(predictions_np) with unmodified official defaults"
                ),
                "exact_official_demo_output": True,
            },
            "object_aggregation": {
                "method": comparison.VGGT_AGGREGATION_METHOD,
                "requested": True,
                "performed": True,
                "status": "completed",
                "mask_paths": aggregation_masks,
                "official_global_p20": {
                    "method": (
                        "official_rule_global_p20_scoped_to_dual_object_mask_union"
                    ),
                    "status": "completed",
                    "exact_official_demo_output": False,
                    "threshold_population": (
                        "finite dual object-mask union with depth-edge confidences zeroed"
                    ),
                    "per_view_threshold_fallback": False,
                    "point_filter": official_filter,
                    "artifacts": {
                        "point_cloud_ply": official_ply.name,
                        "point_cloud_glb": official_glb.name,
                        "evidence": official_evidence.name,
                    },
                },
                "custom_adaptive_p50": {
                    "method": "safe_per_view_adaptive_v1",
                    "status": "completed",
                    "point_filter": custom_filter,
                    "artifacts": {
                        "point_cloud_ply": object_ply.name,
                        "point_cloud_glb": object_glb.name,
                        "evidence": custom_evidence.name,
                    },
                },
                "point_filter": custom_filter,
                "dual_view_contribution": {
                    "proven": True,
                    "exported_points_by_view": custom_stats[
                        "exported_points_by_view"
                    ],
                    "evidence_artifact": custom_evidence.name,
                },
            },
            "artifacts": artifacts,
            "metadata_path": str(self.vggt_metadata.resolve()),
        }
        write_json(self.vggt_metadata, payload)
        self.vggt_payload = payload
        self.world_path = world_path
        self.images_path = images_path
        self.masks_path = masks_path
        self.object_ply = object_ply
        self.official_evidence = official_evidence
        self.custom_evidence = custom_evidence

    @staticmethod
    def assert_write_image(path: Path, image: np.ndarray) -> None:
        if not cv2.imwrite(str(path), image):
            raise AssertionError(f"failed to create test image {path}")

    def preflight(self):
        return comparison.preflight_job(
            spar_report=self.spar_report,
            registration_report=self.registration_report,
            registered_mesh=self.registered_mesh,
            vggt_metadata=self.vggt_metadata,
        )

    def mutate_vggt(self, callback) -> None:
        payload = json.loads(self.vggt_metadata.read_text(encoding="utf-8"))
        callback(payload)
        write_json(self.vggt_metadata, payload)

    def refresh_vggt_artifact(self, key: str, path: Path) -> None:
        def update(payload: dict) -> None:
            payload["artifacts"][key] = declared(path)

        self.mutate_vggt(update)


class PreflightAndProvenanceTests(unittest.TestCase):
    def test_missing_upstream_is_successful_waiting_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = comparison.preflight_job(
                spar_report=root / "spar.json",
                registration_report=root / "registration.json",
                registered_mesh=root / "registered.glb",
                vggt_metadata=root / "metadata.json",
            )
            self.assertEqual(result["status"], "waiting_for_upstream_artifacts")
            self.assertEqual(len(result["missing"]), 4)
            self.assertEqual(
                comparison.main(
                    [
                        "--preflight-only",
                        "--spar-report",
                        str(root / "spar.json"),
                        "--registration-report",
                        str(root / "registration.json"),
                        "--registered-mesh",
                        str(root / "registered.glb"),
                        "--vggt-metadata",
                        str(root / "metadata.json"),
                    ]
                ),
                0,
            )

    def test_complete_bundle_is_ready_and_binds_both_cameras(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = SyntheticBundle(Path(temporary))
            result = bundle.preflight()
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["selection"]["mh_frame_index"], 187)
            self.assertEqual(result["selection"]["sh_frame_index"], 192)
            self.assertEqual(len(result["input_snapshot_sha256"]), 64)
            self.assertGreater(len(result["input_snapshot_records"]), 12)
            self.assertGreater(
                result["vggt"]["official_evidence"]["count"],
                result["vggt"]["custom_evidence"]["count"],
            )
            self.assertTrue(
                all(
                    value > 0
                    for value in result["vggt"]["custom_evidence"][
                        "counts_by_view"
                    ].values()
                )
            )

    def test_official_and_custom_policy_rebinding_is_rejected(self):
        cases = (
            (
                lambda payload: payload["object_aggregation"][
                    "official_global_p20"
                ]["point_filter"].__setitem__("confidence_percentile", 50.0),
                "official-rule object confidence percentile",
            ),
            (
                lambda payload: payload["object_aggregation"][
                    "custom_adaptive_p50"
                ]["point_filter"].__setitem__("policy", "official"),
                "custom adaptive p50 policy",
            ),
        )
        for mutate, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                bundle = SyntheticBundle(Path(temporary))
                bundle.mutate_vggt(mutate)
                with self.assertRaisesRegex(
                    comparison.ComparisonInputError, message
                ):
                    bundle.preflight()

    def test_official_evidence_tampering_and_count_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = SyntheticBundle(Path(temporary))
            original = bundle.official_evidence.read_bytes()
            bundle.official_evidence.write_bytes(
                bytes(value ^ 1 for value in original)
            )
            self.assertEqual(bundle.official_evidence.stat().st_size, len(original))
            with self.assertRaisesRegex(comparison.ComparisonInputError, "SHA-256"):
                bundle.preflight()
        with tempfile.TemporaryDirectory() as temporary:
            bundle = SyntheticBundle(Path(temporary))
            bundle.mutate_vggt(
                lambda payload: payload["object_aggregation"][
                    "official_global_p20"
                ]["point_filter"]["exported_points_by_view"].__setitem__(
                    "SH", 999
                )
            )
            with self.assertRaisesRegex(
                comparison.ComparisonInputError, "per-view counts mismatch"
            ):
                bundle.preflight()

    def test_bad_evidence_camera_order_and_custom_missing_view_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = SyntheticBundle(Path(temporary))
            with np.load(bundle.official_evidence, allow_pickle=False) as loaded:
                values = {key: np.array(loaded[key]) for key in loaded.files}
            values["camera_order"] = np.asarray(["SH", "MH"])
            np.savez_compressed(bundle.official_evidence, **values)
            bundle.refresh_vggt_artifact(
                "official_object_point_evidence", bundle.official_evidence
            )
            with self.assertRaisesRegex(
                comparison.ComparisonInputError, "camera_order"
            ):
                bundle.preflight()
        with tempfile.TemporaryDirectory() as temporary:
            bundle = SyntheticBundle(Path(temporary))
            with np.load(bundle.custom_evidence, allow_pickle=False) as loaded:
                values = {key: np.array(loaded[key]) for key in loaded.files}
            keep = values["view_indices"] == 0
            for key in (
                "points_relative",
                "colors_rgb",
                "depth_confidence",
                "input_depth_relative",
                "view_indices",
            ):
                values[key] = values[key][keep]
            np.savez_compressed(bundle.custom_evidence, **values)
            bundle.refresh_vggt_artifact(
                "object_point_evidence", bundle.custom_evidence
            )
            counts = {"MH": int(keep.sum()), "SH": 0}

            def update(payload: dict) -> None:
                custom = payload["object_aggregation"]["custom_adaptive_p50"][
                    "point_filter"
                ]
                custom["exported_count"] = int(keep.sum())
                custom["exported_points_by_view"] = counts
                custom["all_views_contributed"] = False
                payload["object_aggregation"]["point_filter"] = custom
                payload["object_aggregation"]["dual_view_contribution"][
                    "exported_points_by_view"
                ] = counts

            bundle.mutate_vggt(update)
            with self.assertRaisesRegex(
                comparison.ComparisonInputError,
                "custom adaptive p50 evidence must contain both cameras",
            ):
                bundle.preflight()

    def test_same_size_artifact_tampering_is_rejected_by_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = SyntheticBundle(Path(temporary))
            original = bundle.object_ply.read_bytes()
            bundle.object_ply.write_bytes(bytes(value ^ 1 for value in original))
            self.assertEqual(bundle.object_ply.stat().st_size, len(original))
            with self.assertRaisesRegex(comparison.ComparisonInputError, "SHA-256"):
                bundle.preflight()

    def test_registered_mesh_path_rebinding_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = SyntheticBundle(Path(temporary))
            other = bundle.root / "other.glb"
            other.write_bytes(bundle.registered_mesh.read_bytes())
            with self.assertRaisesRegex(comparison.ComparisonInputError, "registered mesh"):
                comparison.preflight_job(
                    spar_report=bundle.spar_report,
                    registration_report=bundle.registration_report,
                    registered_mesh=other,
                    vggt_metadata=bundle.vggt_metadata,
                )

    def test_vggt_mesh_metric_calibration_and_collision_overclaims_fail_closed(self):
        cases = (
            ("is_triangle_mesh", True),
            ("metric_scale_verified", True),
            ("provided_calibration_applied", True),
            ("collision_ready", True),
        )
        for key, value in cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                bundle = SyntheticBundle(Path(temporary))
                bundle.mutate_vggt(
                    lambda payload, key=key, value=value: payload["geometry_contract"].__setitem__(key, value)
                )
                with self.assertRaisesRegex(comparison.ComparisonInputError, key):
                    bundle.preflight()

    def test_spar_backside_and_physical_overclaims_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = SyntheticBundle(Path(temporary))
            registration = json.loads(bundle.registration_report.read_text(encoding="utf-8"))
            registration["source_spar3d_report"]["hidden_geometry_is_learned_estimate"] = False
            write_json(bundle.registration_report, registration)
            with self.assertRaisesRegex(comparison.ComparisonInputError, "hidden"):
                bundle.preflight()
        with tempfile.TemporaryDirectory() as temporary:
            bundle = SyntheticBundle(Path(temporary))
            spar = json.loads(bundle.spar_report.read_text(encoding="utf-8"))
            spar["physical_geometry_guarantee"] = True
            write_json(bundle.spar_report, spar)
            with self.assertRaisesRegex(comparison.ComparisonInputError, "physical_geometry"):
                bundle.preflight()

    def test_camera_frame_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = SyntheticBundle(Path(temporary))
            bundle.mutate_vggt(
                lambda payload: payload["input"]["views"][1].__setitem__("frame_index", 193)
            )
            with self.assertRaisesRegex(comparison.ComparisonInputError, "SH frame"):
                bundle.preflight()

    def test_bad_array_shape_and_empty_sh_mask_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = SyntheticBundle(Path(temporary))
            np.save(bundle.world_path, np.zeros((1, 3, 4, 3), dtype=np.float32))
            bundle.refresh_vggt_artifact("world_points", bundle.world_path)
            with self.assertRaisesRegex(comparison.ComparisonInputError, "shape"):
                bundle.preflight()
        with tempfile.TemporaryDirectory() as temporary:
            bundle = SyntheticBundle(Path(temporary))
            masks = np.load(bundle.masks_path, allow_pickle=False)
            masks[1] = False
            np.save(bundle.masks_path, masks)
            bundle.refresh_vggt_artifact("object_masks_model_input", bundle.masks_path)
            with self.assertRaisesRegex(comparison.ComparisonInputError, "SH object mask"):
                bundle.preflight()


class RenderingAndTransactionTests(unittest.TestCase):
    def test_full_run_writes_static_contact_video_report_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = SyntheticBundle(Path(temporary) / "upstream")
            output = Path(temporary) / "comparison"
            report = comparison.run_comparison(
                spar_report=bundle.spar_report,
                registration_report=bundle.registration_report,
                registered_mesh=bundle.registered_mesh,
                vggt_metadata=bundle.vggt_metadata,
                output_dir=output,
                fps=2,
                duration_seconds=0.5,
            )
            self.assertEqual(report["status"], "complete")
            self.assertEqual({item.name for item in output.iterdir()}, comparison.OUTPUT_NAMES)
            sheet = cv2.imread(str(output / "contact_sheet.png"), cv2.IMREAD_COLOR)
            self.assertIsNotNone(sheet)
            self.assertEqual(sheet.shape[:2], (1000, 1600))
            self.assertGreater(float(sheet.std()), 5.0)
            capture = cv2.VideoCapture(str(output / "static_diagnostic.mp4"))
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape[:2], (1000, 1600))
            disk_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(
                disk_report["video"]["semantics"],
                "static_diagnostic_with_visual_pointer_not_temporal_reconstruction",
            )
            self.assertEqual(
                disk_report["comparison"]["A"]["backside_semantics"],
                "unseen_and_backside_geometry_is_a_learned_estimate",
            )
            self.assertFalse(disk_report["comparison"]["B"]["is_triangle_mesh"])
            self.assertFalse(disk_report["comparison"]["B"]["provided_calibration_applied"])
            self.assertEqual(set(disk_report["comparison"]), {"A", "B", "C"})
            self.assertFalse(
                disk_report["comparison"]["B"]["confidence_semantics"][
                    "exact_official_demo_output"
                ]
            )
            self.assertEqual(
                disk_report["comparison"]["B"]["exported_points_by_view"],
                report["comparison"]["B"]["exported_points_by_view"],
            )
            self.assertEqual(
                disk_report["comparison"]["C"]["confidence_semantics"][
                    "fallback_views"
                ],
                ["SH"],
            )
            self.assertTrue(
                disk_report["exact_official_full_scene_reference"][
                    "exact_official_demo_output"
                ]
            )
            manifest = json.loads((output / "publish_manifest.json").read_text(encoding="utf-8"))
            for name, record in manifest["files"].items():
                path = output / name
                self.assertEqual(record["bytes"], path.stat().st_size)
                self.assertEqual(record["sha256"], comparison.sha256_file(path))

    def test_overwrite_replaces_whole_directory_and_removes_stale_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = SyntheticBundle(Path(temporary) / "upstream")
            output = Path(temporary) / "comparison"
            kwargs = dict(
                spar_report=bundle.spar_report,
                registration_report=bundle.registration_report,
                registered_mesh=bundle.registered_mesh,
                vggt_metadata=bundle.vggt_metadata,
                output_dir=output,
                fps=1,
                duration_seconds=1,
            )
            comparison.run_comparison(**kwargs)
            (output / "stale.txt").write_text("must disappear", encoding="utf-8")
            comparison.run_comparison(**kwargs, overwrite=True)
            self.assertFalse((output / "stale.txt").exists())
            self.assertEqual({item.name for item in output.iterdir()}, comparison.OUTPUT_NAMES)

    def test_output_overlap_and_symlink_are_rejected_even_for_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = SyntheticBundle(Path(temporary) / "upstream")
            ready = bundle.preflight()
            protected = [item["path"] for item in ready["input_snapshot_records"]]
            with self.assertRaises(comparison.UnsafeOutputError):
                comparison.validate_output_path(bundle.root, protected)
            with self.assertRaises(comparison.UnsafeOutputError):
                comparison.validate_output_path(bundle.spar_dir, protected)
            link = Path(temporary) / "linked-output"
            link.symlink_to(Path(temporary) / "actual-output", target_is_directory=True)
            with self.assertRaises(comparison.UnsafeOutputError):
                comparison.validate_output_path(link, protected)

    def test_publish_failure_rolls_back_previous_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            staging = root / "staging"
            output.mkdir()
            staging.mkdir()
            (output / "old.txt").write_text("old", encoding="utf-8")
            (staging / "new.txt").write_text("new", encoding="utf-8")
            calls = 0

            def fail_second(source: Path, destination: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected publish failure")
                source.replace(destination)

            with self.assertRaises(comparison.PublishError):
                comparison._publish_directory(
                    staging, output, overwrite=True, rename=fail_second
                )
            self.assertEqual((output / "old.txt").read_text(encoding="utf-8"), "old")
            self.assertTrue(staging.is_dir())
            self.assertEqual((staging / "new.txt").read_text(encoding="utf-8"), "new")
            self.assertFalse(any(root.glob(".output.backup-*")))


if __name__ == "__main__":
    unittest.main()
