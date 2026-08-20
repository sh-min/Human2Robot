"""Weight-free checks for the static SPAR3D/XHand comparison pilot."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "compare_spar3d_xhand_occlusion_pilot.py"
)
SPEC = importlib.util.spec_from_file_location(
    "compare_spar3d_xhand_occlusion_pilot", MODULE_PATH
)
comparison = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


class ProvenanceTests(unittest.TestCase):
    def test_contract_is_static_visual_and_never_claims_collision(self):
        contract = comparison.comparison_contract()
        self.assertTrue(contract["static_frame_only"])
        self.assertFalse(contract["object_pose_propagated_to_other_frames"])
        self.assertFalse(contract["physical_collision_solver"])
        self.assertFalse(contract["metric_collision_guarantee"])
        self.assertTrue(contract["spar3d_hidden_surface_is_learned_estimate"])
        self.assertFalse(contract["current_0805_baseline_uses_nominal_primitive_mesh"])

    def test_bundle_status_distinguishes_waiting_half_and_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mesh = root / "mesh.glb"
            report = root / "report.json"
            self.assertEqual(
                comparison.registered_bundle_status(mesh, report),
                "waiting_for_registered_spar3d_mesh",
            )
            mesh.write_bytes(b"mesh")
            self.assertEqual(
                comparison.registered_bundle_status(mesh, report),
                "blocked_incomplete_registered_bundle",
            )
            report.write_text("{}", encoding="utf-8")
            self.assertEqual(comparison.registered_bundle_status(mesh, report), "ready")

    def test_registered_report_binds_exact_mesh_and_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mesh = root / "registered_mesh_mh.glb"
            front = root / "registered_front_depth_proxy_m.npy"
            canonical_mesh = root / "canonical.glb"
            source_spar_report = root / "spar_report.json"
            manifest = root / "input_manifest.json"
            mh_image = root / "mh.jpg"
            mh_mask = root / "mask.png"
            report_path = root / "report.json"
            mesh.write_bytes(b"synthetic registered mesh record")
            front.write_bytes(b"synthetic front raster")
            canonical_mesh.write_bytes(b"synthetic canonical mesh")
            source_spar_report.write_text("{}", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            mh_image.write_bytes(b"image")
            mh_mask.write_bytes(b"mask")

            def declared(path: Path):
                record = comparison.file_identity_record(path)
                return {
                    "path": record["path"],
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                }

            payload = {
                "status": "complete",
                "method": comparison.REGISTRATION_METHOD,
                "metric_scale_verified": False,
                "physical_geometry_guarantee": False,
                "collision_ready": False,
                "selection": {"mh_frame_index": 187},
                "source_spar3d_report": {
                    "hidden_geometry_is_learned_estimate": True,
                    "path": str(source_spar_report.resolve()),
                    "sha256": comparison.sha256_file(source_spar_report),
                },
                "source_mesh": declared(canonical_mesh),
                "sources": {
                    "input_manifest": declared(manifest),
                    "mh_image": declared(mh_image),
                    "mh_modal_mask": declared(mh_mask),
                },
                "outputs": {
                    "registered_mesh_mh.glb": declared(mesh),
                    "registered_front_depth_proxy_m.npy": declared(front),
                    "report.json": {"path": str(report_path.resolve())},
                },
            }
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            _path, loaded, record = comparison.validate_registration_report(
                report_path,
                mesh,
                expected_frame_index=187,
                expected_manifest_path=manifest,
                expected_mh_image_path=mh_image,
                expected_modal_mask_path=mh_mask,
            )
            self.assertEqual(loaded["method"], comparison.REGISTRATION_METHOD)
            self.assertEqual(record["sha256"], comparison.sha256_file(mesh))
            with self.assertRaisesRegex(comparison.ComparisonInputError, "frame"):
                comparison.validate_registration_report(
                    report_path, mesh, expected_frame_index=188
                )

            mesh.write_bytes(b"tampered")
            with self.assertRaisesRegex(comparison.ComparisonInputError, "bytes/SHA"):
                comparison.validate_registration_report(
                    report_path, mesh, expected_frame_index=187
                )

    def test_manifest_output_record_rejects_same_size_content_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "mh.jpg"
            mask = root / "mask.png"
            image.write_bytes(b"AAAA")
            mask.write_bytes(b"MASK")

            def declared(path: Path):
                return {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": comparison.sha256_file(path),
                }

            manifest = {"outputs": {"mh_image": declared(image), "modal_mask": declared(mask)}}
            comparison.validate_manifest_output_records(
                manifest, expected_mh_image=image
            )
            image.write_bytes(b"BBBB")
            with self.assertRaisesRegex(comparison.ComparisonInputError, "SHA-256"):
                comparison.validate_manifest_output_records(
                    manifest, expected_mh_image=image
                )

    def test_input_snapshot_rehash_detects_mid_run_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            source.write_bytes(b"before")
            records = {"source": comparison.file_identity_record(source)}
            comparison.verify_input_snapshot_unchanged(records)
            source.write_bytes(b"after!")
            with self.assertRaisesRegex(comparison.ComparisonInputError, "changed"):
                comparison.verify_input_snapshot_unchanged(records)


class AuxiliaryHacoEvidenceTests(unittest.TestCase):
    @staticmethod
    def _fixture(
        root: Path,
        *,
        invalid_fusion: bool = False,
        dual_mask_change: bool = False,
    ) -> tuple[dict[str, Path], dict[str, object]]:
        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        frame_count = 3
        selected_frame = 1
        auxiliary_dir = root / "aux_contact"
        auxiliary_dir.mkdir()

        primary_scores = np.zeros((frame_count, 5), dtype=np.float32)
        auxiliary_scores = np.zeros_like(primary_scores)
        auxiliary_scores[0, 1] = 0.9
        fused_scores = np.fmax(primary_scores, auxiliary_scores)
        if invalid_fusion:
            fused_scores = primary_scores.copy()
        primary_active = np.zeros((frame_count, 5), dtype=bool)
        active = primary_active.copy()
        auxiliary_qualified = np.zeros_like(active)
        active_runs = {finger: [] for finger in finger_names}

        evidence_path = root / "haco_evidence.npz"
        np.savez(
            evidence_path,
            finger_names=np.asarray(finger_names),
            primary_scores=primary_scores,
            auxiliary_scores=auxiliary_scores,
            fused_scores=fused_scores,
            primary_active=primary_active,
            active=active,
            auxiliary_qualified=auxiliary_qualified,
            auxiliary_frame_indices=np.arange(frame_count, dtype=np.int64),
        )
        completion_report = root / "completion_report.json"
        completion_report.write_text(
            json.dumps(
                {
                    "outputs": {"haco_evidence": evidence_path.name},
                    "counts": {"auxiliary_qualified_finger_frames": 0},
                    "invariants": {
                        "auxiliary_haco_is_confidence_only": True,
                        "auxiliary_geometry_used": False,
                    },
                }
            ),
            encoding="utf-8",
        )

        fusion = "per-finger maximum of primary/auxiliary HaCo scores"
        primary_report = root / "primary_report.json"
        primary_report.write_text(
            json.dumps(
                {
                    "contact_fusion": "primary HaCo scores only",
                    "active_runs": active_runs,
                }
            ),
            encoding="utf-8",
        )
        dual_report = root / "dual_report.json"
        dual_report.write_text(
            json.dumps(
                {
                    "contact_fusion": fusion,
                    "active_runs": active_runs,
                    "contact_activation_policy": {
                        "auxiliary_geometry_used": False
                    },
                }
            ),
            encoding="utf-8",
        )

        primary_mask = np.zeros((frame_count, 2, 3), dtype=bool)
        dual_mask = primary_mask.copy()
        if dual_mask_change:
            dual_mask[0, 0, 0] = True
        primary_mask_path = root / "primary_mask.npy"
        dual_mask_path = root / "dual_mask.npy"
        np.save(primary_mask_path, primary_mask)
        np.save(dual_mask_path, dual_mask)

        force_pixels = np.zeros((frame_count, 5), dtype=np.int64)
        force_pixels[selected_frame, :2] = 1
        temporal_pixels = np.zeros_like(force_pixels)
        penetration_path = root / "penetration.npz"
        np.savez(
            penetration_path,
            finger_names=np.asarray(finger_names),
            force_candidate_pixels=force_pixels,
            temporal_added_pixels=temporal_pixels,
        )
        baseline = np.zeros((frame_count, 2, 3), dtype=bool)
        baseline[selected_frame, 0, :2] = True
        baseline_path = root / "baseline.npy"
        np.save(baseline_path, baseline)

        contact_report_path = root / "contact_report.json"
        contact = {
            "frames": frame_count,
            "finger_names": finger_names,
            "contact_fusion": fusion,
            "contact_score_primary": primary_scores.tolist(),
            "contact_score_auxiliary": auxiliary_scores.tolist(),
            "contact_score_fused": fused_scores.tolist(),
            "active_runs": active_runs,
            "active_runs_primary": active_runs,
            "contact_activation_policy": {
                "auxiliary_geometry_used": False,
                "counts": {
                    "auxiliary_score_dominant_frame_fingers": 1,
                    "active_frame_fingers_added_vs_primary": 0,
                },
            },
            "invariants": {
                "auxiliary_geometry_used": False,
                "object3d_force_bypasses_haco_selector": True,
            },
            "sources": {"aux_contact_dir": str(auxiliary_dir)},
            "object3d_penetration_control": {
                "surface_force": {
                    "haco_activation_used_for_added_branch": False,
                    "candidate_pixels": 2,
                },
                "temporal_filter": {"added_pixels": 0},
            },
            "occluded_pixel_count": [0, 2, 0],
        }
        contact_report_path.write_text(json.dumps(contact), encoding="utf-8")
        paths = {
            "haco_completion_report": completion_report,
            "haco_evidence": evidence_path,
            "haco_mh_report": primary_report,
            "haco_dual_report": dual_report,
            "haco_mh_mask": primary_mask_path,
            "haco_dual_mask": dual_mask_path,
            "contact_penetration_evidence": penetration_path,
            "contact_baseline_mask": baseline_path,
            "contact_baseline_report": contact_report_path,
        }
        return paths, contact

    def test_score_fusion_is_separate_from_zero_active_and_mask_increment(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, contact = self._fixture(Path(temporary))
            result = comparison.validate_haco_auxiliary_effect(
                paths, contact, frame_index=1
            )
            self.assertTrue(result["auxiliary_haco_input_available"])
            self.assertTrue(result["auxiliary_scores_fused"])
            self.assertEqual(result["auxiliary_score_changed_frame_fingers"], 1)
            self.assertFalse(result["auxiliary_geometry_used"])
            self.assertEqual(result["auxiliary_active_increment_frame_fingers"], 0)
            self.assertEqual(result["auxiliary_mask_increment_pixels"], 0)
            self.assertFalse(result["dual_camera_changed_final_mask"])
            attribution = result["selected_frame_baseline_attribution"]
            self.assertEqual(attribution["baseline_occluded_pixels"], 2)
            self.assertTrue(attribution["all_pixels_explained_by_object3d_force_not_haco"])

    def test_panel_titles_report_sh_delta_without_claiming_dual_haco_gain(self):
        titles = comparison.comparison_panel_titles(0)
        self.assertIn("SH mask delta 0 px", titles[0])
        self.assertIn("Object3D-force baseline", titles[1])
        self.assertTrue(all("dual HaCo" not in title for title in titles))

    def test_aux_directory_alone_cannot_validate_fused_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, contact = self._fixture(
                Path(temporary), invalid_fusion=True
            )
            with self.assertRaisesRegex(
                comparison.ComparisonInputError, "fused scores"
            ):
                comparison.validate_haco_auxiliary_effect(
                    paths, contact, frame_index=1
                )

    def test_primary_dual_mask_difference_is_reported_as_actual_effect(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, contact = self._fixture(
                Path(temporary), dual_mask_change=True
            )
            result = comparison.validate_haco_auxiliary_effect(
                paths, contact, frame_index=1
            )
            self.assertEqual(result["auxiliary_mask_increment_pixels"], 1)
            self.assertEqual(result["dual_camera_changed_mask_pixels"], 1)
            self.assertTrue(result["dual_camera_changed_final_mask"])


class OutputTransactionSafetyTests(unittest.TestCase):
    @staticmethod
    def _layout(root: Path):
        repo = root / "repo"
        allowed = repo / "8-5" / "pilot"
        processed = repo / "data" / "processed"
        input_dir = allowed / "inputs"
        input_dir.mkdir(parents=True)
        processed.mkdir(parents=True)
        source = input_dir / "source.bin"
        source.write_bytes(b"source")
        return repo, allowed, processed, source

    def test_destination_rejects_repo_processed_weights_and_input_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo, allowed, processed, source = self._layout(Path(temporary))
            records = {"source": comparison.file_identity_record(source)}
            safe = allowed / "comparison"
            self.assertEqual(
                comparison.validate_output_destination(
                    safe,
                    input_records=records,
                    processed_root=processed,
                    repo_root=repo,
                    allowed_repo_output_root=allowed,
                ),
                safe.resolve(),
            )
            unsafe = (
                repo,
                repo / "scripts" / "comparison",
                processed / "comparison",
                repo / "weights" / "comparison",
                source.parent / "comparison",
                allowed,
            )
            for target in unsafe:
                with self.subTest(target=target):
                    with self.assertRaises(comparison.ComparisonInputError):
                        comparison.validate_output_destination(
                            target,
                            input_records=records,
                            processed_root=processed,
                            repo_root=repo,
                            allowed_repo_output_root=allowed,
                        )

    def test_destination_resolves_symlink_alias_to_protected_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, allowed, processed, source = self._layout(root)
            alias = allowed / "processed_alias"
            alias.symlink_to(processed, target_is_directory=True)
            with self.assertRaises(comparison.ComparisonInputError):
                comparison.validate_output_destination(
                    alias / "comparison",
                    input_records={"source": comparison.file_identity_record(source)},
                    processed_root=processed,
                    repo_root=repo,
                    allowed_repo_output_root=allowed,
                )

    def test_guard_fails_closed_on_stale_staging_and_cleans_its_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result"
            stale = root / ".result.staging.dead"
            stale.mkdir()
            with self.assertRaisesRegex(comparison.ComparisonInputError, "stale"):
                with comparison.output_transaction_guard(output, overwrite=False):
                    pass
            stale.rmdir()
            lock = root / ".result.lock"
            with comparison.output_transaction_guard(output, overwrite=False):
                self.assertTrue(lock.is_dir())
                self.assertTrue((lock / "owner.json").is_file())
            self.assertFalse(lock.exists())

    def test_guard_refuses_overwrite_of_unrelated_existing_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            output.mkdir()
            (output / "user.txt").write_text("do not replace", encoding="utf-8")
            with self.assertRaises(comparison.ComparisonInputError):
                with comparison.output_transaction_guard(output, overwrite=True):
                    pass
            self.assertEqual((output / "user.txt").read_text(), "do not replace")

    def test_guard_accepts_intact_prior_bundle_of_same_method(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / ".result.staging.seed"
            output = root / "result"
            staging.mkdir()
            payload = staging / "payload.bin"
            payload.write_bytes(b"payload")
            report = {
                "schema_version": 1,
                "status": "complete",
                **comparison.comparison_contract(),
                "outputs": {
                    "payload.bin": {
                        **comparison._output_record(payload),
                        "path": str(output / "payload.bin"),
                    }
                },
            }
            comparison.finalize_staging_metadata(
                staging,
                output,
                report=report,
                input_snapshot_sha256="b" * 64,
            )
            staging.replace(output)
            with comparison.output_transaction_guard(output, overwrite=True):
                self.assertTrue((root / ".result.lock").is_dir())
            self.assertTrue((output / "payload.bin").is_file())

    def test_publish_failure_rolls_back_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result"
            staging = root / ".result.staging.test"
            output.mkdir()
            staging.mkdir()
            (output / "old.txt").write_text("old", encoding="utf-8")
            (staging / "new.txt").write_text("new", encoding="utf-8")
            calls = 0

            def fail_second(source: Path, target: Path):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected publish failure")
                source.replace(target)

            with self.assertRaisesRegex(OSError, "injected"):
                comparison._publish_directory(
                    staging,
                    output,
                    overwrite=True,
                    rename=fail_second,
                )
            self.assertEqual((output / "old.txt").read_text(), "old")
            self.assertTrue((staging / "new.txt").is_file())
            self.assertFalse((root / ".result.backup").exists())

    def test_publish_post_rename_failure_still_restores_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result"
            staging = root / ".result.staging.test"
            output.mkdir()
            staging.mkdir()
            (output / "old.txt").write_text("old", encoding="utf-8")
            (staging / "new.txt").write_text("new", encoding="utf-8")
            calls = 0

            def replace_then_fail(source: Path, target: Path):
                nonlocal calls
                calls += 1
                source.replace(target)
                if calls == 2:
                    raise OSError("injected failure after rename")

            with self.assertRaisesRegex(OSError, "after rename"):
                comparison._publish_directory(
                    staging,
                    output,
                    overwrite=True,
                    rename=replace_then_fail,
                )
            self.assertEqual((output / "old.txt").read_text(), "old")
            self.assertTrue((staging / "new.txt").is_file())
            self.assertFalse((root / ".result.backup").exists())

    def test_publish_manifest_is_last_control_and_hashes_report_and_payloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / ".result.staging.test"
            output = root / "result"
            staging.mkdir()
            payload = staging / "payload.bin"
            payload.write_bytes(b"payload")
            report = {
                "schema_version": 1,
                **comparison.comparison_contract(),
                "outputs": {
                    "payload.bin": {
                        **comparison._output_record(payload),
                        "path": str(output / "payload.bin"),
                    }
                },
            }
            manifest = comparison.finalize_staging_metadata(
                staging,
                output,
                report=report,
                input_snapshot_sha256="a" * 64,
            )
            self.assertEqual(set(manifest["files"]), {"payload.bin", "report.json"})
            for name, record in manifest["files"].items():
                self.assertEqual(record["bytes"], (staging / name).stat().st_size)
                self.assertEqual(record["sha256"], comparison.sha256_file(staging / name))
            persisted = json.loads((staging / "report.json").read_text())
            self.assertTrue(persisted["publication"]["payloads_written_before_report"])
            self.assertTrue(
                persisted["publication"]["publish_manifest_is_final_completeness_sentinel"]
            )

    def test_finalization_rejects_stale_payload_hash_in_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / ".result.staging.test"
            output = root / "result"
            staging.mkdir()
            payload = staging / "payload.bin"
            payload.write_bytes(b"before")
            declared = {
                **comparison._output_record(payload),
                "path": str(output / payload.name),
            }
            payload.write_bytes(b"after!")
            report = {
                "schema_version": 1,
                "status": "complete",
                **comparison.comparison_contract(),
                "outputs": {payload.name: declared},
            }
            with self.assertRaisesRegex(RuntimeError, "stale"):
                comparison.finalize_staging_metadata(
                    staging,
                    output,
                    report=report,
                    input_snapshot_sha256="c" * 64,
                )
            self.assertFalse((staging / "report.json").exists())
            self.assertFalse((staging / "publish_manifest.json").exists())


class FrontBackRasterTests(unittest.TestCase):
    def test_normal_front_orientation_is_ordered_and_zero_outside_pair(self):
        normal = np.asarray([[0.8, 0.0, 0.9]], dtype=np.float32)
        reverse = np.asarray([[1.1, 1.0, 0.85]], dtype=np.float32)
        front, back, mask, stats = comparison.orient_depth_pair(normal, reverse)
        np.testing.assert_array_equal(mask, [[True, False, False]])
        np.testing.assert_allclose(front, [[0.8, 0.0, 0.0]])
        np.testing.assert_allclose(back, [[1.1, 0.0, 0.0]])
        self.assertEqual(stats["winding_orientation"], "normal_is_front")

    def test_inward_winding_is_automatically_swapped(self):
        normal = np.asarray([[1.1, 1.2, 1.3]], dtype=np.float32)
        reverse = np.asarray([[0.8, 0.9, 1.0]], dtype=np.float32)
        front, back, mask, stats = comparison.orient_depth_pair(normal, reverse)
        self.assertTrue(mask.all())
        np.testing.assert_allclose(front, reverse)
        np.testing.assert_allclose(back, normal)
        self.assertEqual(stats["winding_orientation"], "reversed_is_front")

    def test_registered_front_crosscheck_accepts_small_export_noise(self):
        rendered = np.zeros((20, 20), dtype=np.float32)
        rendered[4:16, 5:17] = 0.62
        persisted = rendered.copy()
        persisted[4:16, 5:17] += 1.0e-4
        result = comparison.registered_front_raster_crosscheck(
            rendered, persisted
        )
        self.assertEqual(result["silhouette_iou"], 1.0)
        self.assertLess(result["median_camera_z_difference_m"], 0.001)

    def test_registered_front_crosscheck_rejects_coordinate_mismatch(self):
        rendered = np.zeros((20, 20), dtype=np.float32)
        persisted = np.zeros_like(rendered)
        rendered[2:14, 2:14] = 0.62
        persisted[6:18, 6:18] = 0.62
        with self.assertRaisesRegex(comparison.ComparisonInputError, "does not match"):
            comparison.registered_front_raster_crosscheck(rendered, persisted)


class StaticOcclusionTests(unittest.TestCase):
    @staticmethod
    def _fixture() -> dict[str, np.ndarray | float | int]:
        shape = (7, 9)
        hand = np.ones(shape, dtype=bool)
        labels = np.full(shape, 2, dtype=np.uint8)
        robot_depth = np.full(shape, 0.80, dtype=np.float32)
        robot_depth[2:5, 2:7] = 1.05
        robot_depth[3, 4] = 0.98
        front = np.ones(shape, dtype=np.float32)
        back = np.full(shape, 1.20, dtype=np.float32)
        support = np.ones(shape, dtype=bool)
        current = np.zeros(shape, dtype=bool)
        current[0, 0] = True
        contact = np.zeros(shape, dtype=bool)
        contact[0, 0] = True
        return {
            "hand_mask": hand,
            "finger_labels": labels,
            "robot_depth": robot_depth,
            "object_support_mask": support,
            "mesh_mask": support,
            "front_depth": front,
            "back_depth": back,
            "current_mask": current,
            "contact_baseline_mask": contact,
            "thumb_shell_m": 0.03,
            "finger_shell_m": 0.03,
            "palm_shell_m": 0.03,
            "spatial_close_radius_px": 1,
            "spatial_front_slack_m": 0.0,
        }

    def test_front_and_volume_variants_retain_shared_baseline(self):
        result = comparison.build_static_occlusion_masks(**self._fixture())
        self.assertTrue(np.all(~result["contact_baseline"] | result["spar_front"]))
        self.assertTrue(
            np.all(~result["contact_baseline"] | result["spar_volume_filter"])
        )
        self.assertEqual(result["spar_front_hidden"][3, 4], False)
        self.assertEqual(result["spar_volume_hidden_raw"][3, 4], True)
        np.testing.assert_array_equal(
            result["classification"] > 0,
            result["classification_support"],
        )

    def test_spatial_close_fills_only_an_eligible_hole(self):
        hidden = np.zeros((7, 9), dtype=bool)
        hidden[2:5, 2:7] = True
        hidden[3, 4] = False
        eligible = np.zeros_like(hidden)
        eligible[2:5, 2:7] = True
        closed, added = comparison.bounded_spatial_close(
            hidden, eligible, radius_px=1
        )
        self.assertTrue(added[3, 4])
        self.assertTrue(closed[3, 4])
        self.assertFalse(np.any(added & ~eligible))
        self.assertTrue(np.all(~hidden | closed))

    def test_spatial_close_rejects_raw_evidence_outside_eligibility(self):
        hidden = np.zeros((3, 3), dtype=bool)
        hidden[1, 1] = True
        with self.assertRaisesRegex(ValueError, "escaped"):
            comparison.bounded_spatial_close(
                hidden, np.zeros_like(hidden), radius_px=1
            )


class RoiTests(unittest.TestCase):
    def test_shared_roi_is_square_in_bounds_and_contains_support(self):
        mask = np.zeros((100, 160), dtype=bool)
        mask[20:40, 130:158] = True
        left, top, right, bottom = comparison.shared_square_roi(mask, margin_px=10)
        self.assertEqual(right - left, bottom - top)
        self.assertGreaterEqual(left, 0)
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(right, 160)
        self.assertLessEqual(bottom, 100)
        self.assertEqual(int(mask[top:bottom, left:right].sum()), int(mask.sum()))


if __name__ == "__main__":
    unittest.main()
