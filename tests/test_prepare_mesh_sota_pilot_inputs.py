"""Tests for the provenance-safe mesh pilot input builder."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import prepare_mesh_sota_pilot_inputs as builder  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MeshPilotInputBuilderTests(unittest.TestCase):
    def _fixture(
        self, root: Path, *, make_masks_differ: bool = False
    ) -> tuple[Path, dict[str, Path]]:
        episode = root / "episode"
        mh_frame, sh_frame = 2, 5
        for camera in ("camera_1", "camera_2"):
            (episode / camera / "rgb").mkdir(parents=True)
        processed = (
            episode
            / "camera_2"
            / "inpainting"
            / "processed"
            / "view"
            / "0"
        )
        object_layer = processed / "object_layer"
        completion = processed / "object_completion_dual_haco_e2fgvi"
        object_layer.mkdir(parents=True)
        completion.mkdir(parents=True)

        mh = np.zeros((20, 30, 3), dtype=np.uint8)
        mh[:, :, 0] = 17
        mh[6:14, 11:17] = (30, 110, 220)
        sh = np.full((20, 30, 3), (80, 40, 10), dtype=np.uint8)
        mh_path = episode / "camera_2" / "rgb" / f"rgb_frame{mh_frame:06d}.jpg"
        sh_path = episode / "camera_1" / "rgb" / f"rgb_frame{sh_frame:06d}.jpg"
        self.assertTrue(cv2.imwrite(str(mh_path), mh))
        self.assertTrue(cv2.imwrite(str(sh_path), sh))

        modal = np.zeros((8, 20, 30), dtype=bool)
        modal[mh_frame, 6:14, 11:17] = True
        clean = modal.copy()
        amodal = modal.copy()
        if make_masks_differ:
            amodal[mh_frame, 5, 11] = True
        paths = {
            "modal": object_layer / "object_mask_modal.npy",
            "clean": completion / "object_mask_observed_clean.npy",
            "amodal": completion / "object_mask_amodal.npy",
        }
        np.save(paths["modal"], modal)
        np.save(paths["clean"], clean)
        np.save(paths["amodal"], amodal)

        annotation = {
            "episode": "synthetic",
            "num_frames": 8,
            "fps": 24.0,
            "segments": [
                {"start_frame": 0, "end_frame": 0, "label": "Trans"},
                {"start_frame": 1, "end_frame": 4, "label": "Choco"},
                {"start_frame": 5, "end_frame": 7, "label": "Other"},
            ],
        }
        annotation_bytes = (
            json.dumps(annotation, indent=2, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        stereo_annotation = root / "annotations" / "gt_labels.json"
        stereo_annotation.parent.mkdir()
        stereo_annotation.write_bytes(annotation_bytes)
        layer_annotation = episode / "gt_labels.json"
        layer_annotation.write_bytes(annotation_bytes)

        layer_manifest = object_layer / "manifest.json"
        layer_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "labels_json": str(layer_annotation),
                    "frame_count": 8,
                    "height": 20,
                    "width": 30,
                    "transition_policy": "empty_mask",
                    "intervals": [
                        {"label": "Choco", "start": 1, "end": 4},
                        {"label": "Other", "start": 5, "end": 7},
                    ],
                }
            ),
            encoding="utf-8",
        )

        created_utc = "2026-08-05T08:05:19+00:00"
        checkerboard = {
            "square_size_mm": None,
            "length_unit": "checker_square",
            "metric_scale_verified": False,
        }
        camera_1 = {
            "camera_matrix": [
                [100.0, 0.0, 15.0],
                [0.0, 100.0, 10.0],
                [0.0, 0.0, 1.0],
            ],
            "distortion_k1_k2_p1_p2_k3": [0.1, 0.0, 0.0, 0.0, 0.0],
            "rms_reprojection_px": 0.2,
        }
        camera_2 = {
            "camera_matrix": [
                [101.0, 0.0, 15.0],
                [0.0, 101.0, 10.0],
                [0.0, 0.0, 1.0],
            ],
            "distortion_k1_k2_p1_p2_k3": [0.2, 0.0, 0.0, 0.0, 0.0],
            "rms_reprojection_px": 0.3,
        }
        transform = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        calibration_path = root / "calibration.json"
        calibration_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "created_utc": created_utc,
                    "source": {"image_size_wh": [30, 20]},
                    "checkerboard": checkerboard,
                    "camera_1": camera_1,
                    "camera_2": camera_2,
                    "stereo": {
                        "T_camera2_from_camera1": transform,
                        "translation_unit": "checker_square",
                    },
                }
            ),
            encoding="utf-8",
        )
        calibration_hash = _sha256(calibration_path)
        calibration = {
            "status": "provided",
            "schema_version": 1,
            "reference_json": str(calibration_path),
            "reference_sha256": calibration_hash,
            "created_utc": created_utc,
            "image_size_wh": [30, 20],
            "calibration_camera_mapping": {"camera_1": "MH", "camera_2": "SH"},
            "pipeline_camera_mapping": {"camera_1": "SH", "camera_2": "MH"},
            "pipeline_to_calibration_camera": {
                "camera_1": "camera_2",
                "camera_2": "camera_1",
            },
            "checkerboard": checkerboard,
            "intrinsics_by_view": {
                "MH": {"calibration_camera": "camera_1", **camera_1},
                "SH": {"calibration_camera": "camera_2", **camera_2},
            },
            "relative_extrinsics": {
                "from_view": "MH",
                "to_view": "SH",
                "T_camera2_from_camera1": transform,
                "translation_unit": "checker_square",
            },
        }
        stereo_manifest = episode / "stereo_manifest.json"
        stereo_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "episode": "synthetic",
                    "fps": 24.0,
                    "common_frames": 8,
                    "primary_view": "MH",
                    "auxiliary_view": "SH",
                    "training_view": "MH",
                    "robot_overlay_view": "MH",
                    "stereo_code_mapping": {"camera_1": "SH", "camera_2": "MH"},
                    "frame_mapping": "output frame k equals decoded source frame k",
                    "label_vocabulary": ["Trans", "Choco", "Other"],
                    "temporal_alignment": {
                        "reference_view": "camera_2/MH/GT",
                        "camera1_frame_offset": 3,
                        "camera1_lookup": (
                            "camera1/SH source index = camera2/MH frame k + (3)"
                        ),
                        "source_frames_reordered": False,
                        "apply_offset_only_during_dual_view_fusion": True,
                        "out_of_range_policy": "fail_open",
                    },
                    "sources": {"gt_labels": str(stereo_annotation)},
                    "calibration": calibration,
                }
            ),
            encoding="utf-8",
        )
        paths.update(
            {
                "mh": mh_path,
                "sh": sh_path,
                "layer_manifest": layer_manifest,
                "stereo_manifest": stereo_manifest,
                "calibration": calibration_path,
                "stereo_annotation": stereo_annotation,
                "layer_annotation": layer_annotation,
            }
        )
        return episode, paths

    def _prepare(self, episode: Path, output: Path) -> dict[str, object]:
        return builder.prepare_pilot_inputs(
            episode_root=episode,
            output_dir=output,
            mh_frame_index=2,
        )

    def test_cli_builds_synchronized_zero_inference_bundle_with_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode, sources = self._fixture(root)
            output = root / "bundle"

            result = builder.main(
                [
                    "--episode-root",
                    str(episode),
                    "--output",
                    str(output),
                    "--mh-frame",
                    "2",
                    "--crop-size",
                    "512",
                ]
            )

            self.assertEqual(result, 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["selection"]["mh_frame_index"], 2)
            self.assertEqual(manifest["selection"]["sh_frame_index"], 5)
            self.assertEqual(manifest["stereo_alignment"]["camera1_frame_offset"], 3)
            self.assertEqual(manifest["selection"]["mh_role"], "primary/final")
            self.assertEqual(manifest["selection"]["sh_role"], "auxiliary/evidence")
            self.assertEqual(manifest["image_geometry"]["width"], 30)
            self.assertEqual(manifest["image_geometry"]["height"], 20)
            self.assertEqual(
                manifest["mask_validation"]["bbox_xyxy_exclusive"], [11, 6, 17, 14]
            )
            self.assertTrue(
                manifest["mask_validation"]["modal_equals_clean_equals_amodal"]
            )
            self.assertEqual(
                manifest["camera_namespace"]["stereo_code_mapping"],
                {"camera_1": "SH", "camera_2": "MH"},
            )
            provenance = manifest["pixel_provenance"]
            self.assertEqual(provenance["inferred_pixels_used"], 0)
            self.assertEqual(provenance["inpainted_pixels_used"], 0)
            self.assertEqual(provenance["hidden_amodal_pixels_used"], 0)
            self.assertFalse(manifest["bundle"]["model_inference_performed"])
            self.assertTrue(
                manifest["bundle"]["manifest_written_after_all_payload_outputs"]
            )

            mh_output = Path(manifest["outputs"]["mh_image"]["path"])
            sh_output = Path(manifest["outputs"]["sh_image"]["path"])
            self.assertEqual(mh_output.read_bytes(), sources["mh"].read_bytes())
            self.assertEqual(sh_output.read_bytes(), sources["sh"].read_bytes())
            self.assertEqual(
                manifest["sources"]["calibration_reference"]["sha256"],
                _sha256(sources["calibration"]),
            )
            self.assertEqual(
                manifest["sources"]["stereo_ground_truth_annotation"]["sha256"],
                _sha256(sources["stereo_annotation"]),
            )
            self.assertEqual(
                manifest["sources"]["object_layer_ground_truth_annotation"][
                    "sha256"
                ],
                _sha256(sources["layer_annotation"]),
            )

            for records in (manifest["sources"], manifest["outputs"]):
                for record in records.values():
                    recorded_path = Path(record["path"])
                    self.assertTrue(recorded_path.is_file())
                    self.assertEqual(record["bytes"], recorded_path.stat().st_size)
                    self.assertEqual(record["sha256"], _sha256(recorded_path))

            expected_payload_names = {
                Path(record["path"]).name for record in manifest["outputs"].values()
            }
            self.assertEqual(
                {path.name for path in output.iterdir()},
                expected_payload_names | {"manifest.json"},
            )

            mask_images = []
            for key in ("modal_mask", "clean_mask", "amodal_mask"):
                mask_image = cv2.imread(
                    manifest["outputs"][key]["path"], cv2.IMREAD_UNCHANGED
                )
                self.assertEqual(set(np.unique(mask_image).tolist()), {0, 255})
                mask_images.append(mask_image)
            self.assertTrue(np.array_equal(mask_images[0], mask_images[1]))
            self.assertTrue(np.array_equal(mask_images[0], mask_images[2]))

            rgba = cv2.imread(
                manifest["outputs"]["spar3d_rgba_crop"]["path"],
                cv2.IMREAD_UNCHANGED,
            )
            self.assertEqual(rgba.shape, (512, 512, 4))
            self.assertEqual(set(np.unique(rgba[:, :, 3]).tolist()), {0, 255})
            self.assertGreater(np.count_nonzero(rgba[:, :, 3]), 0)
            self.assertTrue(np.all(rgba[rgba[:, :, 3] == 0, :3] == 0))

    def test_rejects_any_hidden_amodal_support_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode, _ = self._fixture(root, make_masks_differ=True)
            output = root / "bundle"

            with self.assertRaisesRegex(ValueError, "selected masks differ"):
                self._prepare(episode, output)

            self.assertFalse(output.exists())

    def test_rejects_camera_namespace_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode, paths = self._fixture(root)
            stereo = json.loads(paths["stereo_manifest"].read_text())
            stereo["stereo_code_mapping"] = {"camera_1": "MH", "camera_2": "SH"}
            paths["stereo_manifest"].write_text(json.dumps(stereo), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "stereo_code_mapping"):
                self._prepare(episode, root / "bundle")

    def test_rejects_fractional_frame_argument_without_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode, _ = self._fixture(root)
            output = root / "bundle"

            with self.assertRaisesRegex(ValueError, "MH frame index must be an integer"):
                builder.prepare_pilot_inputs(
                    episode_root=episode,
                    output_dir=output,
                    mh_frame_index=187.9,
                )

            self.assertFalse(output.exists())

    def test_rejects_fractional_manifest_indices_without_truncation(self):
        mutators = {
            "offset": lambda stereo, layer: stereo["temporal_alignment"].update(
                {"camera1_frame_offset": 3.9}
            ),
            "interval": lambda stereo, layer: layer["intervals"][0].update(
                {"start": 1.9}
            ),
        }
        for name, mutate in mutators.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                episode, paths = self._fixture(root)
                stereo = json.loads(paths["stereo_manifest"].read_text())
                layer = json.loads(paths["layer_manifest"].read_text())
                mutate(stereo, layer)
                paths["stereo_manifest"].write_text(
                    json.dumps(stereo), encoding="utf-8"
                )
                paths["layer_manifest"].write_text(
                    json.dumps(layer), encoding="utf-8"
                )
                output = root / "bundle"

                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    self._prepare(episode, output)
                self.assertFalse(output.exists())

    def test_rejects_annotation_byte_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode, paths = self._fixture(root)
            annotation = json.loads(paths["layer_annotation"].read_text())
            annotation["segments"][1]["end_frame"] = 3
            paths["layer_annotation"].write_text(
                json.dumps(annotation), encoding="utf-8"
            )
            output = root / "bundle"

            with self.assertRaisesRegex(ValueError, "byte-identical"):
                self._prepare(episode, output)
            self.assertFalse(output.exists())

    def test_rejects_calibration_value_that_differs_from_hashed_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode, paths = self._fixture(root)
            stereo = json.loads(paths["stereo_manifest"].read_text())
            stereo["calibration"]["intrinsics_by_view"]["MH"]["camera_matrix"][
                0
            ][0] = 999.0
            paths["stereo_manifest"].write_text(json.dumps(stereo), encoding="utf-8")
            output = root / "bundle"

            with self.assertRaisesRegex(ValueError, "MH.camera_matrix"):
                self._prepare(episode, output)
            self.assertFalse(output.exists())

    def test_rejects_calibration_reference_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode, paths = self._fixture(root)
            stereo = json.loads(paths["stereo_manifest"].read_text())
            stereo["calibration"]["reference_sha256"] = "0" * 64
            paths["stereo_manifest"].write_text(json.dumps(stereo), encoding="utf-8")
            output = root / "bundle"

            with self.assertRaisesRegex(ValueError, "reference hash disagrees"):
                self._prepare(episode, output)
            self.assertFalse(output.exists())

    def test_rejects_wrong_mapping_schema_type(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode, paths = self._fixture(root)
            stereo = json.loads(paths["stereo_manifest"].read_text())
            stereo["schema_version"] = 3.0
            paths["stereo_manifest"].write_text(json.dumps(stereo), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "schema_version"):
                self._prepare(episode, root / "bundle")

    def test_rejects_output_roots_overlapping_sources_repo_or_model_trees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode, paths = self._fixture(root)
            protected = [paths["stereo_manifest"], paths["calibration"]]
            candidates = (
                episode,
                root,
                REPO_ROOT,
                REPO_ROOT.parent,
                REPO_ROOT / "weights",
                REPO_ROOT / "third_party",
            )
            for candidate in candidates:
                with self.subTest(candidate=candidate), self.assertRaisesRegex(
                    ValueError, "output root"
                ):
                    builder.validate_output_root(
                        candidate,
                        episode_root=episode,
                        protected_paths=protected,
                    )

    def test_manifest_is_last_staged_file_before_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode, _ = self._fixture(root)
            output = root / "bundle"
            events: list[str] = []
            real_json_write = builder._write_json_atomic
            real_publish = builder.publish_bundle_directory

            def audit_json_write(path: Path, payload: dict[str, object]) -> None:
                self.assertEqual(path.name, "manifest.json")
                expected_names = {
                    Path(record["path"]).name
                    for record in payload["outputs"].values()
                }
                self.assertEqual(
                    {item.name for item in path.parent.iterdir()}, expected_names
                )
                events.append("manifest")
                real_json_write(path, payload)

            def audit_publish(staged: Path, final: Path) -> None:
                self.assertEqual(events, ["manifest"])
                self.assertTrue((staged / "manifest.json").is_file())
                events.append("publish")
                real_publish(staged, final)

            with mock.patch.object(
                builder, "_write_json_atomic", side_effect=audit_json_write
            ), mock.patch.object(
                builder, "publish_bundle_directory", side_effect=audit_publish
            ):
                self._prepare(episode, output)

            self.assertEqual(events, ["manifest", "publish"])

    def test_publication_failure_rolls_back_complete_old_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode, _ = self._fixture(root)
            output = root / "bundle"
            self._prepare(episode, output)
            sentinel = output / "old-only.txt"
            sentinel.write_text("keep old bundle", encoding="utf-8")
            old_manifest = (output / "manifest.json").read_bytes()
            real_publish = builder.publish_bundle_directory

            def fail_second_replace(staged: Path, final: Path) -> None:
                calls = 0

                def flaky_replace(source: Path, destination: Path) -> None:
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise OSError("injected publication failure")
                    os.replace(source, destination)

                real_publish(staged, final, replace_fn=flaky_replace)

            with mock.patch.object(
                builder,
                "publish_bundle_directory",
                side_effect=fail_second_replace,
            ):
                with self.assertRaisesRegex(RuntimeError, "rolled back"):
                    self._prepare(episode, output)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep old bundle")
            self.assertEqual((output / "manifest.json").read_bytes(), old_manifest)
            self.assertFalse(any(root.glob(".bundle.stage-*")))

    def test_successful_rebuild_removes_all_stale_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode, _ = self._fixture(root)
            output = root / "bundle"
            builder.prepare_pilot_inputs(
                episode_root=episode,
                output_dir=output,
                mh_frame_index=2,
                crop_size=256,
            )
            stale = output / "stale.txt"
            stale.write_text("must disappear", encoding="utf-8")
            old_crop = output / "spar3d_input_rgba_256.png"
            self.assertTrue(old_crop.exists())

            manifest = self._prepare(episode, output)

            self.assertFalse(stale.exists())
            self.assertFalse(old_crop.exists())
            self.assertTrue(Path(manifest["outputs"]["spar3d_rgba_crop"]["path"]).is_file())
            expected_names = {
                Path(record["path"]).name for record in manifest["outputs"].values()
            }
            self.assertEqual(
                {item.name for item in output.iterdir()},
                expected_names | {"manifest.json"},
            )


if __name__ == "__main__":
    unittest.main()
