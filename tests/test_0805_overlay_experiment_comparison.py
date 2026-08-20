"""Contract tests for the 08-05 overlay experiment comparison."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "inpainting"))

import compare_0805_overlay_experiments as comparison  # noqa: E402
from make_video_comparison_grid import VideoMetadata  # noqa: E402


class Overlay0805ComparisonTests(unittest.TestCase):
    def _metadata(self, name: str = "video.mp4") -> VideoMetadata:
        return VideoMetadata(
            path=Path("/tmp") / name,
            width=3,
            height=2,
            frame_count=2,
            fps=Fraction(24, 1),
            duration_s=2 / 24,
            codec_name="h264",
            pixel_format="yuv420p",
        )

    def _mask(self, *pixels: tuple[int, int, int]) -> np.ndarray:
        value = np.zeros((2, 2, 3), dtype=bool)
        for frame, row, column in pixels:
            value[frame, row, column] = True
        return value

    def _common_contact_report(
        self,
        *,
        processed_demo: str = "/tmp/pd",
        auxiliary: bool = True,
    ) -> dict[str, object]:
        return {
            "frames": 2,
            "width": 3,
            "height": 2,
            "fps": 24.0,
            "occlusion_mode": "haco",
            "occluded_pixels_total": 2,
            "frames_with_occlusion": 2,
            "aux_frame_offset": 5,
            "contact_score_fused": [[0.9] * 5, [0.8] * 5],
            "hidden_fraction": [[0.5] * 5, [0.6] * 5],
            "config": {
                "contact_depth_thickness_scale": 0.0,
                "object3d_force_surface": False,
                "object3d_temporal_max_gap_frames": 0,
            },
            "sources": {
                "processed_demo": processed_demo,
                "background": "/tmp/background.mp4",
                "raw_video": "/tmp/video_L.mp4",
                "hawor_npz": "/tmp/retarget_input.npz",
                "contact_dir": "/tmp/mh_contact",
                "aux_contact_dir": "/tmp/sh_contact" if auxiliary else None,
                "overlay_dir": "/tmp/overlay",
                "object_mask": "/tmp/object_mask.npy",
                "object_surface_depth": None,
                "scene_depth": None,
            },
            "object_surface_3d": {"alignment": "none"},
            "invariants": {
                "auxiliary_haco_is_confidence_only": True,
                "auxiliary_geometry_used": False,
                "primary_view_owns_contact_projection_and_depth": True,
                "object3d_haco_is_selector_only": False,
            },
        }

    def _loaded(
        self,
        key: str,
        *,
        report: dict[str, object] | None = None,
        mask: np.ndarray | None = None,
        branch: str = "calibrated",
        mask_path: str | None = None,
    ) -> comparison.LoadedVariant:
        spec = comparison.VARIANT_SPECS[key]
        statistics = (
            comparison._mask_summary(mask)
            if mask is not None
            else {"pixels": 0, "frames": 0, "max_pixels_per_frame": 0}
        )
        return comparison.LoadedVariant(
            branch=branch,
            spec=spec,
            root=Path(f"/tmp/{key}"),
            video=Path(f"/tmp/{key}/{spec.video_name}"),
            metadata=self._metadata(f"{key}.mp4"),
            mask_path=Path(mask_path or f"/tmp/{key}/{spec.mask_name}")
            if spec.mask_name is not None
            else None,
            mask=mask,
            report_path=Path(f"/tmp/{key}/{spec.report_name}")
            if spec.report_name is not None
            else None,
            report=report,
            mask_statistics=statistics,
        )

    def test_override_parser_and_conventional_resolution(self):
        overrides = comparison.parse_source_overrides(
            [["approx", "haco_mh", "/tmp/custom"]]
        )
        self.assertEqual(overrides[("approx", "haco_mh")], Path("/tmp/custom"))
        self.assertEqual(
            comparison.resolve_variant_directory(
                Path("/tmp/pd"), "approx", "haco_mh", {}
            ),
            Path("/tmp/pd/overlay_haco_mh"),
        )
        self.assertEqual(
            comparison.VARIANT_SPECS["object3d_dual"].directory_name,
            "overlay_object3d_dual_aligned",
        )
        self.assertEqual(
            comparison.VARIANT_SPECS["barrier"].directory_name,
            "overlay_best_inpaint_barrier",
        )
        derived_directories = {
            comparison.VARIANT_SPECS[key].directory_name
            for key in (
                "haco_visibility_union",
                "union_safety_shell",
                "surface_front_side_half",
                "surface_front_side_half_back_full",
            )
        }
        self.assertEqual(
            derived_directories,
            {"overlay_xhand_surface_strategies"},
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            comparison.parse_source_overrides(
                [
                    ["approx", "haco_mh", "/tmp/one"],
                    ["approx", "haco_mh", "/tmp/two"],
                ]
            )
        with self.assertRaisesRegex(ValueError, "unknown source variant"):
            comparison.parse_source_overrides(
                [["approx", "not_a_variant", "/tmp/value"]]
            )
        parsed = comparison._build_parser().parse_args(
            [
                "--approx_pd",
                "/tmp/approx",
                "--calibrated_pd",
                "/tmp/calibrated",
                "--out_dir",
                "/tmp/out",
                "--overwrite",
                "--extended-grid",
                "required",
            ]
        )
        self.assertTrue(parsed.overwrite)
        self.assertEqual(parsed.extended_grid, "required")

    def test_calibration_manifests_are_a_controlled_per_view_focal_ab(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary) / "calibration.json"
            reference.write_text('{"schema_version": 1}\n')
            controlled = {
                "episode": "1",
                "fps": 24.0,
                "common_frames": 2,
                "label_vocabulary": ["Choco"],
                "primary_view": "MH",
                "auxiliary_view": "SH",
                "stereo_code_mapping": {
                    "camera_1": "SH",
                    "camera_2": "MH",
                },
                "training_view": "MH",
                "robot_overlay_view": "MH",
                "source_pairing": {"annotation_episode": "1"},
                "sources": {"MH": "/tmp/mh.mov", "SH": "/tmp/sh.mov"},
                "raw_frame_counts": {"MH": 2, "SH": 2},
                "tail_frames_dropped": {"MH": 0, "SH": 0},
                "temporal_alignment": {"camera1_frame_offset": 5},
                "frame_mapping": "output frame k equals decoded source frame k",
            }
            approx = copy.deepcopy(controlled)
            approx.update(
                {
                    "calibration": {"status": "not_provided"},
                    "intrinsics": {"status": "not_provided"},
                }
            )
            calibrated = copy.deepcopy(controlled)
            calibrated.update(
                {
                    "calibration": {
                        "status": "provided",
                        "reference_json": str(reference),
                        "reference_sha256": comparison._sha256_file(reference),
                        "checkerboard": {"metric_scale_verified": False},
                        "intrinsics_by_view": {
                            "MH": {"fx_px": 1030.0},
                            "SH": {"fx_px": 1070.0},
                        },
                    },
                    "intrinsics": {
                        "status": "provided",
                        "pixel_focal_px": {"MH": 1030.0, "SH": 1070.0},
                    },
                }
            )
            result = comparison.validate_calibration_manifest_pair(
                approx,
                calibrated,
            )
            self.assertEqual(
                result["calibrated_focal_px"],
                {"MH": 1030.0, "SH": 1070.0},
            )
            self.assertFalse(result["metric_scale_verified"])

            mixed = copy.deepcopy(calibrated)
            mixed["source_pairing"] = {"annotation_episode": "2"}
            with self.assertRaisesRegex(ValueError, "source_pairing"):
                comparison.validate_calibration_manifest_pair(approx, mixed)

    def test_contact_variant_contracts(self):
        mask = self._mask((0, 0, 0), (1, 1, 2))

        mh_report = self._common_contact_report(auxiliary=False)
        comparison.validate_variant_report(
            self._loaded("haco_mh", report=mh_report, mask=mask),
            expected_offset=5,
        )

        dual_report = self._common_contact_report()
        comparison.validate_variant_report(
            self._loaded("haco_dual", report=dual_report, mask=mask),
            expected_offset=5,
        )
        invalid_dual = copy.deepcopy(dual_report)
        invalid_dual["invariants"]["auxiliary_geometry_used"] = True
        with self.assertRaisesRegex(ValueError, "SH-confidence"):
            comparison.validate_variant_report(
                self._loaded("haco_dual", report=invalid_dual, mask=mask),
                expected_offset=5,
            )

        half_report = self._common_contact_report()
        half_report["config"]["contact_depth_thickness_scale"] = 0.5
        comparison.validate_variant_report(
            self._loaded("haco_half_depth", report=half_report, mask=mask),
            expected_offset=5,
        )

        full_report = self._common_contact_report()
        full_report["config"]["contact_depth_thickness_scale"] = 1.0
        comparison.validate_variant_report(
            self._loaded("haco_full_depth", report=full_report, mask=mask),
            expected_offset=5,
        )

        boundary_report = self._common_contact_report()
        boundary_report["contact_interior_expansion"] = {
            "enabled": True,
            "expand_px": 3,
        }
        comparison.validate_variant_report(
            self._loaded("boundary_fill", report=boundary_report, mask=mask),
            expected_offset=5,
        )

        scalar_report = self._common_contact_report()
        scalar_report["occlusion_mode"] = "ensemble"
        scalar_report["sources"]["scene_depth"] = "/tmp/depth.npy"
        comparison.validate_variant_report(
            self._loaded("scalar_object_z", report=scalar_report, mask=mask),
            expected_offset=5,
        )

    def test_object3d_and_force_contracts(self):
        mask = self._mask((0, 0, 0), (1, 1, 2))
        for key, alignment, force, gap in (
            ("object3d_surface_unaligned", "none", False, 0),
            ("object3d_dual", "contact", False, 0),
            ("object3d_force_temporal", "contact", True, 2),
        ):
            report = self._common_contact_report()
            report["occlusion_mode"] = "object3d"
            report["sources"]["object_surface_depth"] = "/tmp/surface.npy"
            report["object_surface_3d"] = {"alignment": alignment}
            object3d_invariants = (
                {
                    "object3d_haco_is_selector_only": False,
                    "object3d_force_bypasses_haco_selector": True,
                    "object3d_temporal_filter_only_adds_occlusion": True,
                }
                if force
                else {"object3d_haco_is_selector_only": True}
            )
            report["invariants"].update(object3d_invariants)
            report["config"].update(
                {
                    "object3d_force_surface": force,
                    "object3d_temporal_max_gap_frames": gap,
                }
            )
            comparison.validate_variant_report(
                self._loaded(key, report=report, mask=mask),
                expected_offset=5,
            )

        invalid = self._common_contact_report()
        invalid["occlusion_mode"] = "object3d"
        invalid["sources"]["object_surface_depth"] = "/tmp/surface.npy"
        invalid["object_surface_3d"] = {"alignment": "contact"}
        invalid["invariants"].update(
            {
                "object3d_force_bypasses_haco_selector": True,
                "object3d_temporal_filter_only_adds_occlusion": True,
            }
        )
        invalid["config"].update(
            {"object3d_force_surface": True, "object3d_temporal_max_gap_frames": 1}
        )
        with self.assertRaisesRegex(ValueError, "2-frame"):
            comparison.validate_variant_report(
                self._loaded(
                    "object3d_force_temporal", report=invalid, mask=mask
                ),
                expected_offset=5,
            )

    def test_barrier_and_stereo_contracts(self):
        mask = self._mask((0, 0, 0), (1, 1, 2))
        barrier = {
            "frames": 2,
            "width": 3,
            "height": 2,
            "fps": 24.0,
            "method": "visual_camera_z_xhand_barrier",
            "pose_state_modified": False,
            "metric_collision_guarantee": False,
            "counts": {
                "final_occluded_pixels": 2,
                "final_frames_with_occlusion": 2,
                "residual_violation_pixels": 0,
            },
            "sources": {
                "overlay_dir": "/tmp/overlay",
                "baseline_mask": "/tmp/force/mask.npy",
                "object_surface_depth": "/tmp/completed_surface.npy",
            },
            "invariants": {
                "baseline_subset_final": True,
                "final_occlusion_subset_of_xhand": True,
                "rb5_arm_excluded": True,
                "valid_surface_barrier_residual_is_zero": True,
                "trajectory_arrays_unchanged": True,
            },
        }
        comparison.validate_variant_report(
            self._loaded("barrier", report=barrier, mask=mask),
            expected_offset=5,
        )

        stereo = {
            "frames": 2,
            "width": 3,
            "height": 2,
            "fps": 24.0,
            "camera2_is_final_view": True,
            "output_modes": ["visibility_haco"],
            "mode_statistics": {
                "visibility_haco": {"pixels": 2, "frames": 2}
            },
            "temporal_alignment": {"camera1_frame_offset": 5},
            "sources": {
                "camera1_hawor": "/tmp/c1.npz",
                "camera2_hawor": "/tmp/c2.npz",
                "camera1_contact_dir": "/tmp/c1_contact",
                "contact_dir": "/tmp/c2_contact",
                "camera1_visible_mask": "/tmp/c1_mask.npy",
                "camera2_visible_mask": "/tmp/c2_mask.npy",
                "overlay_dir": "/tmp/overlay",
            },
            "invariants": {"dual_haco_uses_max_available_score": True},
        }
        comparison.validate_variant_report(
            self._loaded("stereo_visibility", report=stereo, mask=mask),
            expected_offset=5,
        )
        stereo["camera2_is_final_view"] = False
        with self.assertRaisesRegex(ValueError, "camera 2"):
            comparison.validate_variant_report(
                self._loaded("stereo_visibility", report=stereo, mask=mask),
                expected_offset=5,
            )
        invalid_fusion = copy.deepcopy(stereo)
        invalid_fusion["camera2_is_final_view"] = True
        invalid_fusion["invariants"]["dual_haco_uses_max_available_score"] = False
        with self.assertRaisesRegex(ValueError, "fusion invariant"):
            comparison.validate_variant_report(
                self._loaded(
                    "stereo_visibility",
                    report=invalid_fusion,
                    mask=mask,
                ),
                expected_offset=5,
            )

    def test_derived_strategy_contracts(self):
        mask = self._mask((0, 0, 0), (1, 1, 2))
        report = {
            "frames": 2,
            "width": 3,
            "height": 2,
            "fps": 24.0,
            "comparison": "xhand_thickness_strategies",
            "mode_statistics": {
                "baseline_force_union": {"pixels": 2, "frames": 2},
                "union_safety_shell_diagnostic": {"pixels": 2, "frames": 2},
            },
            "surface_strategy_statistics": {
                "surface_front_side_half": {"pixels": 2, "frames": 2},
                "surface_front_side_half_back_full": {
                    "pixels": 2,
                    "frames": 2,
                },
            },
            "sources": {"overlay_dir": "/tmp/overlay"},
            "invariants": {
                "union_equals_baseline_or_force": True,
                "diagnostic_shell_is_union_superset": True,
                "surface_labels_decode_to_finger_labels": True,
                "surface_side_half_uses_baseline_except_side_half": True,
                "surface_weighted_uses_front_zero_side_half_back_full": True,
            },
        }
        for key in (
            "haco_visibility_union",
            "union_safety_shell",
            "surface_front_side_half",
            "surface_front_side_half_back_full",
        ):
            comparison.validate_variant_report(
                self._loaded(key, report=copy.deepcopy(report), mask=mask),
                expected_offset=5,
            )

    def test_derived_lineage_is_tied_to_loaded_sources(self):
        role_sources = {
            "baseline": "haco_dual",
            "half_thickness": "haco_half_depth",
            "full_thickness": "haco_full_depth",
            "visibility_force": "stereo_visibility",
        }
        sources = {
            key: self._loaded(key)
            for key in set(role_sources.values())
        }
        report_sources: dict[str, object] = {
            "overlay_dir": "/tmp/overlay",
            "object_mask": "/tmp/object_mask.npy",
            "background": "/tmp/background.mp4",
            "raw_video": "/tmp/video_L.mp4",
        }
        for role, source_name in role_sources.items():
            source = sources[source_name]
            report_sources[role] = {
                "directory": str(source.root),
                "report": str(source.report_path),
            }
        sources["haco_visibility_union"] = self._loaded(
            "haco_visibility_union",
            report={"sources": report_sources},
        )
        common = {
            "object_mask": "/tmp/object_mask.npy",
            "background": "/tmp/background.mp4",
            "raw_video": "/tmp/video_L.mp4",
        }
        comparison._validate_derived_lineage(
            branch="calibrated",
            sources=sources,
            overlay_dir=Path("/tmp/overlay"),
            common_contact_sources=common,
        )

        report_sources["baseline"]["directory"] = "/tmp/mixed"
        with self.assertRaisesRegex(ValueError, "baseline directory"):
            comparison._validate_derived_lineage(
                branch="calibrated",
                sources=sources,
                overlay_dir=Path("/tmp/overlay"),
                common_contact_sources=common,
            )

    def test_output_location_cannot_overlap_read_only_inputs(self):
        pd_by_branch = {
            "approx": Path("/tmp/approx_pd"),
            "calibrated": Path("/tmp/calibrated_pd"),
        }
        sources = {
            "approx": {"raw": self._loaded("raw", branch="approx")},
            "calibrated": {"raw": self._loaded("raw")},
        }
        comparison._validate_output_location(
            Path("/var/tmp/overlay_comparison"),
            pd_by_branch=pd_by_branch,
            sources=sources,
        )
        with self.assertRaisesRegex(ValueError, "processed demo"):
            comparison._validate_output_location(
                Path("/tmp/approx_pd"),
                pd_by_branch=pd_by_branch,
                sources=sources,
            )
        with self.assertRaisesRegex(ValueError, "read-only source"):
            comparison._validate_output_location(
                Path("/tmp/raw/nested"),
                pd_by_branch=pd_by_branch,
                sources=sources,
            )

    def test_staging_cleanup_runs_immediately_on_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            created: list[Path] = []

            @comparison._cleanup_staging_on_exit
            def fail_after_staging() -> None:
                staging = Path(
                    tempfile.mkdtemp(prefix=".comparison.", dir=parent)
                )
                created.append(staging)
                comparison._register_staging_path(staging)
                (staging / "partial.mp4").write_bytes(b"partial")
                raise RuntimeError("injected render failure")

            with self.assertRaisesRegex(RuntimeError, "injected"):
                fail_after_staging()
            self.assertEqual(len(created), 1)
            self.assertFalse(created[0].exists())

    def test_streamed_mask_statistics_and_direction(self):
        first_mask = self._mask((0, 0, 0), (1, 0, 0))
        second_mask = self._mask((0, 0, 0), (0, 0, 1))
        self.assertEqual(
            comparison._mask_summary(first_mask),
            {"pixels": 2, "frames": 2, "max_pixels_per_frame": 1},
        )
        first = self._loaded("haco_mh", mask=first_mask)
        second = self._loaded("haco_dual", mask=second_mask)
        difference = comparison._mask_difference(first, second, chunk_frames=1)
        self.assertEqual(difference["added_pixels"], 1)
        self.assertEqual(difference["removed_pixels"], 1)
        self.assertEqual(difference["changed_frames"], 2)
        self.assertEqual(difference["intersection_pixels"], 1)
        self.assertEqual(difference["union_pixels"], 3)
        self.assertAlmostEqual(difference["jaccard"], 1 / 3)

        raw = self._loaded("raw", branch="calibrated")
        raw_to_first = comparison._mask_difference(raw, first)
        self.assertEqual(raw_to_first["added_pixels"], 2)
        self.assertEqual(raw_to_first["removed_pixels"], 0)

    def test_branch_lineage_accepts_controlled_core_and_rejects_mixing(self):
        mask = self._mask((0, 0, 0), (1, 1, 2))
        sources: dict[str, comparison.LoadedVariant] = {
            "raw": self._loaded("raw"),
        }
        for key in comparison.CONTACT_VARIANTS:
            if key not in {
                "haco_mh",
                "haco_dual",
                "object3d_dual",
                "object3d_force_temporal",
            }:
                continue
            report = self._common_contact_report(auxiliary=key != "haco_mh")
            if key.startswith("object3d"):
                report["occlusion_mode"] = "object3d"
                report["sources"]["object_surface_depth"] = "/tmp/surface.npy"
                report["object_surface_3d"] = {"alignment": "contact"}
                report["invariants"]["object3d_haco_is_selector_only"] = True
            if key == "object3d_force_temporal":
                report["invariants"].update(
                    {
                        "object3d_haco_is_selector_only": False,
                        "object3d_force_bypasses_haco_selector": True,
                        "object3d_temporal_filter_only_adds_occlusion": True,
                    }
                )
                report["config"].update(
                    {
                        "object3d_force_surface": True,
                        "object3d_temporal_max_gap_frames": 2,
                    }
                )
            sources[key] = self._loaded(
                key,
                report=report,
                mask=mask,
                mask_path=(
                    "/tmp/force/mask.npy"
                    if key == "object3d_force_temporal"
                    else None
                ),
            )
        barrier_report = {
            "counts": {"baseline_occluded_pixels": 2},
            "sources": {
                "overlay_dir": "/tmp/overlay",
                "baseline_mask": "/tmp/force/mask.npy",
                "background": "/tmp/background.mp4",
                "raw_video": "/tmp/video_L.mp4",
                "object_surface_depth": "/tmp/surface.npy",
            }
        }
        sources["barrier"] = self._loaded(
            "barrier", report=barrier_report, mask=mask
        )
        stereo_report = {
            "sources": {
                "camera2_hawor": "/tmp/retarget_input.npz",
                "camera1_contact_dir": "/tmp/sh_contact",
                "contact_dir": "/tmp/mh_contact",
                "background": "/tmp/background.mp4",
                "object_mask": "/tmp/object_mask.npy",
                "overlay_dir": "/tmp/overlay",
            }
        }
        sources["stereo_visibility"] = self._loaded(
            "stereo_visibility",
            report=stereo_report,
            mask=mask,
        )
        result = comparison.validate_branch_contract(
            branch="calibrated",
            pd=Path("/tmp/pd"),
            sources=sources,
            expected_offset=5,
        )
        self.assertEqual(result["controlled_robot_overlay"], "/tmp/overlay")

        mixed = dict(sources)
        mixed_report = copy.deepcopy(sources["object3d_dual"].report)
        mixed_report["sources"]["overlay_dir"] = "/tmp/other_overlay"
        mixed["object3d_dual"] = self._loaded(
            "object3d_dual", report=mixed_report, mask=mask
        )
        with self.assertRaisesRegex(ValueError, "overlay_dir lineage"):
            comparison.validate_branch_contract(
                branch="calibrated",
                pd=Path("/tmp/pd"),
                sources=mixed,
                expected_offset=5,
            )

        mixed_stereo = dict(sources)
        mixed_stereo_report = copy.deepcopy(stereo_report)
        mixed_stereo_report["sources"]["camera1_contact_dir"] = "/tmp/other_sh"
        mixed_stereo["stereo_visibility"] = self._loaded(
            "stereo_visibility",
            report=mixed_stereo_report,
            mask=mask,
        )
        with self.assertRaisesRegex(ValueError, "camera1_contact_dir"):
            comparison.validate_branch_contract(
                branch="calibrated",
                pd=Path("/tmp/pd"),
                sources=mixed_stereo,
                expected_offset=5,
            )

        stale_barrier = dict(sources)
        stale_barrier["barrier"] = self._loaded(
            "barrier",
            report=barrier_report,
            mask=self._mask(),
        )
        with self.assertRaisesRegex(ValueError, "not a subset"):
            comparison.validate_branch_contract(
                branch="calibrated",
                pd=Path("/tmp/pd"),
                sources=stale_barrier,
                expected_offset=5,
            )

    def test_extended_layout_is_complete_and_uses_2p5d_terminology(self):
        self.assertEqual(len(comparison.EXTENDED_LAYOUT), 16)
        self.assertEqual(comparison.EXTENDED_GRID.video_count, 16)
        self.assertEqual(
            {variant for variant, _label in comparison.EXTENDED_LAYOUT},
            {
                "raw",
                "haco_mh",
                "haco_dual",
                "haco_half_depth",
                "haco_full_depth",
                "boundary_fill",
                "stereo_visibility",
                "haco_visibility_union",
                "union_safety_shell",
                "surface_front_side_half",
                "surface_front_side_half_back_full",
                "scalar_object_z",
                "object3d_surface_unaligned",
                "object3d_dual",
                "object3d_force_temporal",
                "barrier",
            },
        )
        labels = " ".join(label.lower() for _variant, label in comparison.EXTENDED_LAYOUT)
        self.assertNotIn("mesh", labels)
        self.assertIn("2.5d", labels)

    def test_extended_auto_completeness_uses_conventional_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            pd = Path(temporary)
            spec = comparison.VARIANT_SPECS["haco_visibility_union"]
            root = pd / spec.directory_name
            root.mkdir()
            for name in (spec.video_name, spec.mask_name, spec.report_name):
                assert name is not None
                (root / name).write_bytes(b"x")
            self.assertTrue(
                comparison._variant_is_complete(
                    pd, "calibrated", "haco_visibility_union", {}
                )
            )
            (root / spec.mask_name).unlink()
            self.assertFalse(
                comparison._variant_is_complete(
                    pd, "calibrated", "haco_visibility_union", {}
                )
            )


if __name__ == "__main__":
    unittest.main()
