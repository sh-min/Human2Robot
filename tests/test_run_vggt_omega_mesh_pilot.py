"""Focused tests for the VGGT-Omega two-view point-cloud pilot wrapper."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_vggt_omega_mesh_pilot as pilot  # noqa: E402


def _write_rgb(
    path: Path, *, width: int = 4, height: int = 3, offset: int = 0
) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.arange(width, dtype=np.uint8)[None] + offset
    Image.fromarray(image, mode="RGB").save(path)


def _write_mask(path: Path, *, width: int = 4, height: int = 3) -> None:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[1:, 1:3] = 255
    Image.fromarray(mask, mode="L").save(path)


def _record(path: Path, **metadata) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": pilot._sha256(path),
        **metadata,
    }


def _fixture_manifest(root: Path, *, include_sh_mask: bool) -> Path:
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    mh_image = inputs / "mh.jpg"
    sh_image = inputs / "sh.jpg"
    mh_mask = inputs / "mh_mask.png"
    sh_mask = inputs / "sh_mask.png"
    _write_rgb(mh_image, offset=0)
    _write_rgb(sh_image, offset=20)
    _write_mask(mh_mask)
    if include_sh_mask:
        _write_mask(sh_mask)

    outputs = {
        "mh_image": _record(mh_image),
        "sh_image": _record(sh_image),
        "modal_mask": _record(mh_mask),
    }
    if include_sh_mask:
        outputs["sh_modal_mask"] = _record(
            sh_mask,
            view="SH",
            pipeline_camera="camera_1",
            frame_index=192,
        )
    document = {
        "schema_version": 1,
        "kind": "mesh_sota_pilot_input_bundle",
        "bundle": {"output_root": str(inputs)},
        "selection": {
            "episode": "1",
            "object_label": "Choco",
            "mh_frame_index": 187,
            "sh_frame_index": 192,
            "mh_role": "primary/final",
            "sh_role": "auxiliary/evidence",
        },
        "image_geometry": {"width": 4, "height": 3},
        "camera_namespace": {
            "primary_view": "MH",
            "auxiliary_view": "SH",
            "stereo_code_mapping": {"camera_1": "SH", "camera_2": "MH"},
            "pipeline_camera_mapping": {"camera_1": "SH", "camera_2": "MH"},
        },
        "sources": {
            "mh_image": _record(
                mh_image,
                view="MH",
                pipeline_camera="camera_2",
                frame_index=187,
            ),
            "sh_image": _record(
                sh_image,
                view="SH",
                pipeline_camera="camera_1",
                frame_index=192,
            ),
            "modal_mask": {"selected_frame_index": 187},
        },
        "outputs": outputs,
        "calibration": {"status": "provided"},
    }
    path = inputs / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _manifest_document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest_document(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def _fixture_vggt_repository(root: Path) -> Path:
    repository = root / "VGGT-Omega"
    for relative_name in pilot.REQUIRED_REPOSITORY_FILES:
        path = repository / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            pilot.MODEL_LICENSE + "\nfixture license body\n"
            if relative_name == "LICENSE"
            else f"# fixture {relative_name}\n"
        )
        path.write_text(content, encoding="utf-8")
    git_directory = repository / ".git"
    reference = git_directory / "refs" / "heads" / "main"
    reference.parent.mkdir(parents=True)
    (git_directory / "HEAD").write_text(
        "ref: refs/heads/main\n", encoding="utf-8"
    )
    reference.write_text(pilot.EXPECTED_LOCAL_CODE_COMMIT + "\n", encoding="utf-8")
    return repository


def _invoke_main(arguments: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        return_code = pilot.main(arguments)
    return return_code, stdout.getvalue(), stderr.getvalue()


def _tree_snapshot(root: Path) -> dict[str, tuple]:
    snapshot: dict[str, tuple] = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            snapshot[relative] = ("symlink", str(path.readlink()))
        elif path.is_dir():
            snapshot[relative] = ("directory", path.stat().st_mtime_ns)
        else:
            snapshot[relative] = (
                "file",
                path.stat().st_size,
                path.stat().st_mtime_ns,
                pilot._sha256(path),
            )
    return snapshot


class ManifestContractTests(unittest.TestCase):
    def test_manifest_preserves_mh_first_order_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = _fixture_manifest(
                Path(directory), include_sh_mask=True
            )
            inputs = pilot.load_pilot_manifest(manifest)

            self.assertEqual([view.camera for view in inputs.views], ["MH", "SH"])
            self.assertEqual([view.frame_index for view in inputs.views], [187, 192])
            self.assertEqual(inputs.object_label, "Choco")
            self.assertEqual(inputs.mh.image_path.name, "mh.jpg")
            self.assertEqual(inputs.sh.object_mask_path.name, "sh_mask.png")
            pilot.validate_input_files(inputs, require_object_masks=True)

    def test_missing_sh_mask_rejects_object_aggregation_but_allows_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = _fixture_manifest(
                Path(directory), include_sh_mask=False
            )
            inputs = pilot.load_pilot_manifest(manifest)

            pilot.validate_input_files(inputs, require_object_masks=False)
            with self.assertRaisesRegex(
                pilot.MissingDualObjectMaskError,
                "requires an SH modal object mask.*only the MH mask",
            ):
                pilot.validate_input_files(inputs, require_object_masks=True)

    def test_fractional_string_and_boolean_integer_fields_are_rejected(self):
        cases = (
            (("selection", "mh_frame_index"), 187.9),
            (("selection", "sh_frame_index"), "192"),
            (("image_geometry", "width"), 4.0),
            (("schema_version",), True),
        )
        for key_path, value in cases:
            with self.subTest(key_path=key_path, value=value):
                with tempfile.TemporaryDirectory() as directory:
                    manifest = _fixture_manifest(
                        Path(directory), include_sh_mask=True
                    )
                    document = _manifest_document(manifest)
                    target = document
                    for key in key_path[:-1]:
                        target = target[key]
                    target[key_path[-1]] = value
                    _write_manifest_document(manifest, document)
                    with self.assertRaisesRegex(
                        pilot.PilotManifestError, "must be an integer"
                    ):
                        pilot.load_pilot_manifest(manifest)

    def test_schema_kind_choco_and_camera_mapping_are_strictly_bound(self):
        mutations = (
            ("schema", lambda doc: doc.update(schema_version=2), "schema_version"),
            ("kind", lambda doc: doc.update(kind="other_bundle"), "manifest kind"),
            (
                "label",
                lambda doc: doc["selection"].update(object_label="Milk"),
                "object_label='Choco'",
            ),
            (
                "camera mapping",
                lambda doc: doc["camera_namespace"].update(
                    pipeline_camera_mapping={"camera_1": "MH", "camera_2": "SH"}
                ),
                "camera_1=SH",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    manifest = _fixture_manifest(
                        Path(directory), include_sh_mask=True
                    )
                    document = _manifest_document(manifest)
                    mutate(document)
                    _write_manifest_document(manifest, document)
                    with self.assertRaisesRegex(pilot.PilotManifestError, message):
                        pilot.load_pilot_manifest(manifest)

    def test_source_and_mask_view_frame_bindings_cannot_be_swapped(self):
        mutations = (
            (
                "MH source frame",
                lambda doc: doc["sources"]["mh_image"].update(frame_index=188),
                "frame_index=188",
            ),
            (
                "SH source view",
                lambda doc: doc["sources"]["sh_image"].update(view="MH"),
                "does not match 'SH'",
            ),
            (
                "SH mask camera",
                lambda doc: doc["outputs"]["sh_modal_mask"].update(
                    pipeline_camera="camera_2"
                ),
                "does not match 'camera_1'",
            ),
            (
                "MH mask selected frame",
                lambda doc: doc["sources"]["modal_mask"].update(
                    selected_frame_index=186
                ),
                "selection.mh_frame_index=187",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    manifest = _fixture_manifest(
                        Path(directory), include_sh_mask=True
                    )
                    document = _manifest_document(manifest)
                    mutate(document)
                    _write_manifest_document(manifest, document)
                    with self.assertRaisesRegex(pilot.PilotManifestError, message):
                        pilot.load_pilot_manifest(manifest)

    def test_declared_path_bytes_and_sha_are_verified_against_files(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = _fixture_manifest(Path(directory), include_sh_mask=True)
            inputs = pilot.load_pilot_manifest(manifest)
            image_path = inputs.mh.image_path
            content = bytearray(image_path.read_bytes())
            content[-1] ^= 1
            image_path.write_bytes(content)

            with self.assertRaisesRegex(pilot.PilotManifestError, "SHA-256"):
                pilot.validate_input_files(inputs, require_object_masks=True)

        with tempfile.TemporaryDirectory() as directory:
            manifest = _fixture_manifest(Path(directory), include_sh_mask=True)
            document = _manifest_document(manifest)
            document["outputs"]["mh_image"]["bytes"] += 1
            document["sources"]["mh_image"]["bytes"] += 1
            _write_manifest_document(manifest, document)
            inputs = pilot.load_pilot_manifest(manifest)
            with self.assertRaisesRegex(pilot.PilotManifestError, "byte count"):
                pilot.validate_input_files(inputs, require_object_masks=True)

    def test_manifest_change_after_parse_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = _fixture_manifest(Path(directory), include_sh_mask=True)
            inputs = pilot.load_pilot_manifest(manifest)
            manifest.write_text(
                manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(
                pilot.PilotManifestError, "manifest changed after it was parsed"
            ):
                pilot.validate_input_files(inputs, require_object_masks=True)

    def test_output_image_record_cannot_be_rebound_to_the_other_view(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = _fixture_manifest(Path(directory), include_sh_mask=True)
            document = _manifest_document(manifest)
            document["outputs"]["mh_image"] = dict(
                document["outputs"]["sh_image"]
            )
            _write_manifest_document(manifest, document)
            with self.assertRaisesRegex(
                pilot.PilotManifestError, "MH output image bytes/SHA"
            ):
                pilot.load_pilot_manifest(manifest)

    def test_selected_output_paths_cannot_escape_the_bundle_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture_manifest(root, include_sh_mask=True)
            document = _manifest_document(manifest)
            original = Path(document["outputs"]["mh_image"]["path"])
            escaped = root / "escaped_mh.jpg"
            escaped.write_bytes(original.read_bytes())
            document["outputs"]["mh_image"] = _record(escaped)
            _write_manifest_document(manifest, document)

            with self.assertRaisesRegex(
                pilot.PilotManifestError, "escapes bundle root"
            ):
                pilot.load_pilot_manifest(manifest)

    def test_exact_checkpoint_filename_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrong = root / "model.pt"
            wrong.write_bytes(b"not a checkpoint")
            with self.assertRaisesRegex(
                pilot.CheckpointValidationError,
                pilot.EXACT_CHECKPOINT_FILENAME,
            ):
                pilot.validate_checkpoint_path(wrong)

            exact = root / pilot.EXACT_CHECKPOINT_FILENAME
            exact.write_bytes(b"fixture only")
            self.assertEqual(
                pilot.validate_checkpoint_path(
                    exact, require_official_content=False
                ),
                exact.resolve(),
            )

    def test_exact_huggingface_style_symlink_name_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "0123456789abcdef"
            blob.write_bytes(b"fixture only")
            snapshot = root / "snapshot"
            snapshot.mkdir()
            checkpoint = snapshot / pilot.EXACT_CHECKPOINT_FILENAME
            checkpoint.symlink_to(blob)

            self.assertEqual(
                pilot.validate_checkpoint_path(
                    checkpoint, require_official_content=False
                ),
                blob,
            )

    def test_partial_exactly_named_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / pilot.EXACT_CHECKPOINT_FILENAME
            checkpoint.write_bytes(b"partial")

            with self.assertRaisesRegex(
                pilot.CheckpointValidationError, "incomplete.*expected"
            ):
                pilot.validate_checkpoint_path(checkpoint)

    def test_same_size_wrong_checkpoint_sha_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / pilot.EXACT_CHECKPOINT_FILENAME
            checkpoint.write_bytes(b"same-size-wrong-checkpoint")

            with (
                mock.patch.object(
                    pilot, "EXPECTED_CHECKPOINT_BYTES", checkpoint.stat().st_size
                ),
                self.assertRaisesRegex(
                    pilot.CheckpointValidationError, "SHA-256 does not match"
                ),
            ):
                pilot.validate_checkpoint_path(checkpoint)


class PreflightTests(unittest.TestCase):
    def test_missing_checkpoint_waits_with_exit_zero_and_no_import_or_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture_manifest(root, include_sh_mask=True)
            repository = _fixture_vggt_repository(root)
            checkpoint = root / "weights" / pilot.EXACT_CHECKPOINT_FILENAME
            output = root / "new-output"
            before = _tree_snapshot(root)
            imported_model_modules: list[str] = []
            real_import = __import__

            def guarded_import(name, *args, **kwargs):
                if name == "torch" or name.startswith("vggt_omega"):
                    imported_model_modules.append(name)
                    raise AssertionError(f"preflight imported {name}")
                return real_import(name, *args, **kwargs)

            with mock.patch(
                "builtins.__import__", side_effect=guarded_import
            ), mock.patch.object(
                pilot,
                "run_pilot",
                side_effect=AssertionError("preflight invoked inference path"),
            ):
                return_code, stdout, stderr = _invoke_main(
                    [
                        "--preflight",
                        "--manifest",
                        str(manifest),
                        "--checkpoint",
                        str(checkpoint),
                        "--vggt-repository",
                        str(repository),
                        "--output-dir",
                        str(output),
                    ]
                )

            self.assertEqual(return_code, 0)
            self.assertEqual(stderr, "")
            result = json.loads(stdout)
            self.assertEqual(result["status"], "waiting_for_checkpoint")
            self.assertEqual(result["checkpoint"]["state"], "missing")
            self.assertFalse(result["checkpoint"]["exists"])
            self.assertTrue(result["input"]["dual_object_masks"]["verified"])
            self.assertFalse(result["model_imported"])
            self.assertFalse(result["gpu_checked"])
            self.assertFalse(output.exists())
            self.assertEqual(imported_model_modules, [])
            self.assertEqual(_tree_snapshot(root), before)

    def test_partial_checkpoint_waits_without_hashing_partial_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture_manifest(root, include_sh_mask=True)
            repository = _fixture_vggt_repository(root)
            checkpoint = root / "weights" / pilot.EXACT_CHECKPOINT_FILENAME
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"partial checkpoint")
            hashed_paths: list[Path] = []
            real_sha256 = pilot._sha256

            def audited_sha256(path: Path) -> str:
                hashed_paths.append(path.resolve())
                return real_sha256(path)

            with mock.patch.object(pilot, "_sha256", side_effect=audited_sha256):
                result = pilot.preflight_pilot(
                    manifest_path=manifest,
                    checkpoint_path=checkpoint,
                    vggt_repository=repository,
                    output_dir=root / "output",
                )

            self.assertEqual(result["status"], "waiting_for_checkpoint")
            self.assertEqual(result["checkpoint"]["state"], "partial")
            self.assertEqual(
                result["checkpoint"]["actual_bytes"], checkpoint.stat().st_size
            )
            self.assertNotIn(checkpoint.resolve(), hashed_paths)
            self.assertFalse((root / "output").exists())

    def test_verified_checkpoint_reports_ready_and_existing_default_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture_manifest(root, include_sh_mask=True)
            repository = _fixture_vggt_repository(root)
            checkpoint = root / "weights" / pilot.EXACT_CHECKPOINT_FILENAME
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"complete official fixture")
            expected_sha = pilot._sha256(checkpoint)
            default_output = root / "vggt_omega"
            default_output.mkdir()
            sentinel = default_output / "existing.txt"
            sentinel.write_text("do not modify", encoding="utf-8")
            before = _tree_snapshot(root)

            with mock.patch.object(
                pilot, "EXPECTED_CHECKPOINT_BYTES", checkpoint.stat().st_size
            ), mock.patch.object(
                pilot, "EXPECTED_CHECKPOINT_SHA256", expected_sha
            ):
                result = pilot.preflight_pilot(
                    manifest_path=manifest,
                    checkpoint_path=checkpoint,
                    vggt_repository=repository,
                )
                force_result = pilot.preflight_pilot(
                    manifest_path=manifest,
                    checkpoint_path=checkpoint,
                    vggt_repository=repository,
                    force=True,
                )

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["checkpoint"]["state"], "verified")
            self.assertEqual(result["checkpoint"]["actual_sha256"], expected_sha)
            self.assertTrue(result["checkpoint"]["official_content_verified"])
            self.assertEqual(result["output"]["path"], str(default_output))
            self.assertTrue(result["output"]["exists"])
            self.assertTrue(result["output"]["force_required_for_inference"])
            self.assertEqual(
                result["output"]["planned_publish_action"],
                "blocked_requires_force",
            )
            self.assertFalse(result["run_allowed_with_current_flags"])
            self.assertEqual(force_result["status"], "ready")
            self.assertTrue(force_result["run_allowed_with_current_flags"])
            self.assertEqual(
                force_result["output"]["planned_publish_action"],
                "replace_existing_directory",
            )
            self.assertEqual(
                result["model"]["official_huggingface_checkpoint_commit"],
                "05654241adc2f218dfb089c373a011f8a7040576",
            )
            self.assertEqual(
                result["repository"]["local_code_commit"],
                "39a0cb8af88554f15ddcb5354cd52bde588fa014",
            )
            self.assertEqual(result["model"]["license"], pilot.MODEL_LICENSE)
            self.assertEqual(
                result["repository"]["license"]["name"], pilot.MODEL_LICENSE
            )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not modify")
            self.assertEqual(_tree_snapshot(root), before)

    def test_complete_size_with_wrong_sha_is_hard_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture_manifest(root, include_sh_mask=True)
            repository = _fixture_vggt_repository(root)
            checkpoint = root / "weights" / pilot.EXACT_CHECKPOINT_FILENAME
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"same size but wrong digest")
            output = root / "output"

            with mock.patch.object(
                pilot, "EXPECTED_CHECKPOINT_BYTES", checkpoint.stat().st_size
            ), mock.patch.object(
                pilot, "EXPECTED_CHECKPOINT_SHA256", "0" * 64
            ):
                return_code, stdout, stderr = _invoke_main(
                    [
                        "--preflight",
                        "--manifest",
                        str(manifest),
                        "--checkpoint",
                        str(checkpoint),
                        "--vggt-repository",
                        str(repository),
                        "--output-dir",
                        str(output),
                    ]
                )

            self.assertEqual(return_code, 2)
            self.assertEqual(stdout, "")
            self.assertIn("SHA-256", stderr)
            self.assertFalse(output.exists())

    def test_preflight_requires_both_masks_even_with_full_scene_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture_manifest(root, include_sh_mask=False)
            repository = _fixture_vggt_repository(root)
            checkpoint = root / "weights" / pilot.EXACT_CHECKPOINT_FILENAME

            return_code, stdout, stderr = _invoke_main(
                [
                    "--preflight",
                    "--full-scene-only",
                    "--manifest",
                    str(manifest),
                    "--checkpoint",
                    str(checkpoint),
                    "--vggt-repository",
                    str(repository),
                    "--output-dir",
                    str(root / "output"),
                ]
            )

            self.assertEqual(return_code, 2)
            self.assertEqual(stdout, "")
            self.assertIn("requires an SH modal object mask", stderr)

    def test_repository_commit_structure_and_license_are_strict(self):
        cases = (
            (
                "commit",
                lambda repository: (
                    repository / ".git" / "refs" / "heads" / "main"
                ).write_text("0" * 40 + "\n", encoding="utf-8"),
                "local code commit mismatch",
            ),
            (
                "structure",
                lambda repository: (
                    repository / "vggt_omega" / "utils" / "pose_enc.py"
                ).unlink(),
                "missing regular VGGT-Omega repository file",
            ),
            (
                "license",
                lambda repository: (repository / "LICENSE").write_text(
                    "different license\n", encoding="utf-8"
                ),
                "license must be",
            ),
        )
        for name, mutate, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = _fixture_manifest(root, include_sh_mask=True)
                repository = _fixture_vggt_repository(root)
                mutate(repository)

                with self.assertRaisesRegex(pilot.VGGTOmegaPilotError, message):
                    pilot.preflight_pilot(
                        manifest_path=manifest,
                        checkpoint_path=(
                            root / "weights" / pilot.EXACT_CHECKPOINT_FILENAME
                        ),
                        vggt_repository=repository,
                        output_dir=root / "output",
                    )

    def test_preflight_rejects_unsafe_symlink_and_non_directory_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture_manifest(root, include_sh_mask=True)
            repository = _fixture_vggt_repository(root)
            checkpoint = root / "weights" / pilot.EXACT_CHECKPOINT_FILENAME
            target = root / "target-output"
            target.mkdir()
            linked_output = root / "linked-output"
            linked_output.symlink_to(target, target_is_directory=True)
            file_output = root / "file-output"
            file_output.write_text("not a directory", encoding="utf-8")
            cases = (
                (manifest.parent, "unsafe output directory"),
                (linked_output, "symbolic link"),
                (file_output, "not a directory"),
            )
            for output, message in cases:
                with self.subTest(output=output), self.assertRaisesRegex(
                    pilot.VGGTOmegaPilotError, message
                ):
                    pilot.preflight_pilot(
                        manifest_path=manifest,
                        checkpoint_path=checkpoint,
                        vggt_repository=repository,
                        output_dir=output,
                    )


class MaskTransformTests(unittest.TestCase):
    def test_nearest_resize_preserves_binary_regions(self):
        mask = np.zeros((2, 4), dtype=bool)
        mask[:, :2] = True

        resized = pilot.transform_mask_nearest(mask, (4, 8))

        self.assertEqual(resized.shape, (4, 8))
        self.assertEqual(resized.dtype, np.bool_)
        self.assertTrue(resized[:, :4].all())
        self.assertFalse(resized[:, 4:].any())

    def test_extreme_aspect_mask_uses_same_center_crop_rule(self):
        mask = np.zeros((2, 8), dtype=bool)
        mask[:, 2:6] = True

        resized = pilot.transform_mask_nearest(mask, (2, 4))

        # h/w=0.25 is cropped centrally to width=4 before resizing.
        self.assertTrue(resized.all())

    def test_binary_mask_loader_rejects_full_video_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stack.npy"
            np.save(path, np.ones((3, 2, 2), dtype=bool))
            with self.assertRaisesRegex(
                pilot.PilotManifestError, "selected-frame mask must be 2-D"
            ):
                pilot.load_binary_mask(path)

    def test_mask_loader_rejects_nonbinary_and_mixed_foreground_conventions(self):
        cases = (
            np.asarray([[0, 128], [255, 0]], dtype=np.uint8),
            np.asarray([[0, 1], [255, 0]], dtype=np.uint8),
            np.asarray([[0.0, 0.5], [1.0, 0.0]], dtype=np.float32),
        )
        for index, values in enumerate(cases):
            with self.subTest(index=index):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "mask.npy"
                    np.save(path, values)
                    with self.assertRaisesRegex(
                        pilot.PilotManifestError, "strict binary convention"
                    ):
                        pilot.load_binary_mask(path)

    def test_manifest_bound_nonbinary_mask_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = _fixture_manifest(Path(directory), include_sh_mask=True)
            document = _manifest_document(manifest)
            mask_path = Path(document["outputs"]["sh_modal_mask"]["path"])
            Image.fromarray(
                np.asarray(
                    [[0, 0, 0, 0], [0, 128, 128, 0], [0, 0, 0, 0]],
                    dtype=np.uint8,
                ),
                mode="L",
            ).save(mask_path)
            document["outputs"]["sh_modal_mask"].update(_record(mask_path))
            _write_manifest_document(manifest, document)
            inputs = pilot.load_pilot_manifest(manifest)

            with self.assertRaisesRegex(
                pilot.PilotManifestError, "strict binary convention"
            ):
                pilot.validate_input_files(inputs, require_object_masks=True)


class OutputPublicationTests(unittest.TestCase):
    def test_output_path_rejects_inputs_checkpoint_repo_and_repo_root_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture_manifest(root, include_sh_mask=True)
            inputs = pilot.load_pilot_manifest(manifest)
            checkpoint = root / "weights" / pilot.EXACT_CHECKPOINT_FILENAME
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            repository = root / "repository"
            vggt_repository = repository / "third_party" / "VGGT-Omega"
            vggt_repository.mkdir(parents=True)
            dedicated = repository / "results" / "vggt_omega"

            unsafe_paths = (
                inputs.manifest_path,
                inputs.bundle_root,
                inputs.bundle_root / "child-output",
                inputs.mh.image_path,
                inputs.mh.source_image.path.parent,
                checkpoint,
                checkpoint.parent,
                vggt_repository,
                vggt_repository / "child-output",
                repository,
                repository / "arbitrary-output",
                root,
            )
            for unsafe in unsafe_paths:
                with self.subTest(unsafe=unsafe), self.assertRaisesRegex(
                    pilot.VGGTOmegaPilotError, "unsafe output directory"
                ):
                    pilot.validate_output_directory(
                        unsafe,
                        inputs=inputs,
                        checkpoint_path=checkpoint,
                        vggt_repository=vggt_repository,
                        repo_root=repository,
                        dedicated_repo_output_root=dedicated,
                    )

            self.assertEqual(
                pilot.validate_output_directory(
                    dedicated / "run-1",
                    inputs=inputs,
                    checkpoint_path=checkpoint,
                    vggt_repository=vggt_repository,
                    repo_root=repository,
                    dedicated_repo_output_root=dedicated,
                ),
                (dedicated / "run-1").resolve(),
            )
            external = root.parent / f"{root.name}-external" / "run-1"
            self.assertEqual(
                pilot.validate_output_directory(
                    external,
                    inputs=inputs,
                    checkpoint_path=checkpoint,
                    vggt_repository=vggt_repository,
                    repo_root=repository,
                    dedicated_repo_output_root=dedicated,
                ),
                external.resolve(),
            )

    def test_run_rejects_overlap_before_checkpoint_or_force_handling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture_manifest(root, include_sh_mask=True)
            missing_checkpoint = root / "weights" / pilot.EXACT_CHECKPOINT_FILENAME
            missing_repository = root / "missing-vggt-repository"
            for force in (False, True):
                with self.subTest(force=force), self.assertRaisesRegex(
                    pilot.VGGTOmegaPilotError, "unsafe output directory"
                ):
                    pilot.run_pilot(
                        manifest_path=manifest,
                        checkpoint_path=missing_checkpoint,
                        vggt_repository=missing_repository,
                        output_dir=manifest.parent,
                        force=force,
                    )

    def test_atomic_force_publish_removes_stale_object_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            staging = root / "staging"
            output.mkdir()
            staging.mkdir()
            stale_names = (
                "object_dual_relative_point_cloud.ply",
                "object_dual_relative_point_cloud.glb",
                "object_dual_mask_filtered_relative_point_cloud.ply",
                "object_masks_model_input.npy",
            )
            for name in stale_names:
                (output / name).write_bytes(b"stale")
            (output / "metadata.json").write_text("old", encoding="utf-8")
            (staging / "metadata.json").write_text("new", encoding="utf-8")
            (staging / "full_scene_relative_point_cloud.ply").write_bytes(b"scene")

            pilot._publish_directory(staging, output, force=True)

            self.assertFalse(staging.exists())
            self.assertFalse((root / ".output.backup").exists())
            self.assertEqual(
                (output / "metadata.json").read_text(encoding="utf-8"), "new"
            )
            self.assertTrue((output / "full_scene_relative_point_cloud.ply").is_file())
            for name in stale_names:
                self.assertFalse((output / name).exists())

    def test_publish_without_force_preserves_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            staging = root / "staging"
            output.mkdir()
            staging.mkdir()
            (output / "keep.txt").write_text("keep", encoding="utf-8")
            (staging / "new.txt").write_text("new", encoding="utf-8")

            with self.assertRaisesRegex(
                pilot.VGGTOmegaPilotError, "pass --force"
            ):
                pilot._publish_directory(staging, output, force=False)

            self.assertEqual((output / "keep.txt").read_text(), "keep")
            self.assertTrue(staging.is_dir())

    def test_output_record_contains_final_path_bytes_and_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = root / "staged.bin"
            final = root / "published" / "artifact.bin"
            staged.write_bytes(b"artifact-content")

            record = pilot._output_record(staged, final)

            self.assertEqual(record["path"], str(final))
            self.assertEqual(record["bytes"], len(b"artifact-content"))
            self.assertEqual(record["sha256"], pilot._sha256(staged))

    def test_object_method_name_describes_aggregation_not_fusion(self):
        self.assertEqual(
            pilot.OBJECT_AGGREGATION_METHOD,
            "dual_view_mask_filtered_point_aggregation",
        )


class GeometryHookTests(unittest.TestCase):
    def test_point_cloud_export_writes_nonempty_ply_and_glb(self):
        selection = {
            "points": np.asarray(
                [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32
            ),
            "colors": np.asarray(
                [[255, 0, 0], [0, 255, 0]], dtype=np.uint8
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ply_path = root / "points.ply"
            glb_path = root / "points.glb"

            pilot.export_colored_point_cloud(
                selection, ply_path=ply_path, glb_path=glb_path
            )

            self.assertGreater(ply_path.stat().st_size, 0)
            self.assertGreater(glb_path.stat().st_size, 0)

    def test_identity_camera_unprojection(self):
        depth = np.ones((1, 2, 2, 1), dtype=np.float32)
        extrinsic = np.concatenate(
            [np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)],
            axis=1,
        )[None]
        intrinsic = np.eye(3, dtype=np.float32)[None]

        points = pilot.unproject_depth_map_to_world(
            depth, extrinsic, intrinsic
        )

        np.testing.assert_allclose(
            points[0],
            [
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ],
        )

    def test_official_global_p20_object_filter_uses_one_union_threshold(self):
        points = np.zeros((2, 1, 2, 3), dtype=np.float32)
        points[..., 2] = 1.0
        images = np.zeros((2, 3, 1, 2), dtype=np.float32)
        confidence = np.asarray(
            [[[100.0, 101.0]], [[1.0, 2.0]]], dtype=np.float32
        )
        depth = np.ones((2, 1, 2, 1), dtype=np.float32)
        masks = np.ones((2, 1, 2), dtype=bool)

        selected = pilot.filter_official_global_object_points(
            points,
            images,
            confidence,
            pixel_mask=masks,
            depth=depth,
        )

        stats = selected["stats"]
        self.assertAlmostEqual(stats["confidence_threshold"], 1.6)
        self.assertFalse(stats["per_view_threshold_fallback"])
        self.assertIsNone(stats["per_view_thresholds"])
        self.assertEqual(stats["exported_points_by_view"], {"MH": 2, "SH": 1})
        np.testing.assert_array_equal(selected["confidence"], [100.0, 101.0, 2.0])

    def test_official_global_p20_never_rescues_a_starved_view(self):
        points = np.zeros((2, 1, 2, 3), dtype=np.float32)
        points[..., 2] = 1.0
        images = np.zeros((2, 3, 1, 2), dtype=np.float32)
        confidence = np.asarray(
            [[[100.0, 101.0]], [[1.0, 2.0]]], dtype=np.float32
        )
        depth = np.asarray(
            [[[[1.0], [1.0]]], [[[1.0], [2.0]]]], dtype=np.float32
        )
        masks = np.ones((2, 1, 2), dtype=bool)

        selected = pilot.filter_official_global_object_points(
            points,
            images,
            confidence,
            pixel_mask=masks,
            depth=depth,
        )

        stats = selected["stats"]
        self.assertEqual(stats["confidence_threshold"], 0.0)
        self.assertEqual(stats["exported_points_by_view"], {"MH": 2, "SH": 0})
        self.assertFalse(stats["all_views_contributed"])
        self.assertFalse(stats["per_view_threshold_fallback"])

    def test_object_filter_tracks_points_from_both_views(self):
        points = np.zeros((2, 2, 2, 3), dtype=np.float32)
        points[..., 2] = 1.0
        images = np.zeros((2, 3, 2, 2), dtype=np.float32)
        images[0, 0] = 1.0
        images[1, 1] = 1.0
        confidence = np.ones((2, 2, 2), dtype=np.float32)
        depth = np.ones((2, 2, 2, 1), dtype=np.float32)
        masks = np.zeros((2, 2, 2), dtype=bool)
        masks[0, 0, 0] = True
        masks[1, 1, 1] = True

        selected = pilot.filter_point_cloud(
            points,
            images,
            confidence,
            pixel_mask=masks,
            depth=depth,
            confidence_percentile=0.0,
            max_points=10,
            require_all_views=True,
        )

        self.assertEqual(selected["points"].shape, (2, 3))
        self.assertEqual(
            selected["stats"]["exported_points_by_view"], {"MH": 1, "SH": 1}
        )
        self.assertTrue(selected["stats"]["all_views_required"])
        self.assertTrue(selected["stats"]["all_views_contributed"])
        self.assertEqual(
            selected["stats"]["stage_counts_by_view"]["positive_depth"],
            {"MH": 1, "SH": 1},
        )
        self.assertTrue(
            all(selected["stats"]["safety"]["invariants"].values())
        )
        self.assertEqual(
            selected["stats"]["confidence_adaptation"]["fallback_views"], []
        )
        for counts in selected["stats"]["stage_counts_by_view"].values():
            self.assertEqual(sum(counts.values()), 2)

    def test_global_confidence_starvation_uses_only_per_view_fallback(self):
        points = np.zeros((2, 1, 2, 3), dtype=np.float32)
        points[..., 2] = 1.0
        images = np.zeros((2, 3, 1, 2), dtype=np.float32)
        confidence = np.asarray(
            [[[100.0, 101.0]], [[1.0, 2.0]]], dtype=np.float32
        )
        depth = np.ones((2, 1, 2, 1), dtype=np.float32)

        selected = pilot.filter_point_cloud(
            points,
            images,
            confidence,
            depth=depth,
            confidence_percentile=75.0,
            max_points=10,
            require_all_views=True,
        )

        stats = selected["stats"]
        self.assertEqual(
            stats["stage_counts_by_view"]["after_global_confidence"],
            {"MH": 1, "SH": 0},
        )
        self.assertEqual(stats["confidence_adaptation"]["fallback_views"], ["SH"])
        self.assertEqual(
            stats["confidence_adaptation"]["mode_by_view"],
            {"MH": "global_percentile", "SH": "per_view_percentile_fallback"},
        )
        self.assertAlmostEqual(stats["confidence_threshold"], 100.25)
        self.assertAlmostEqual(
            stats["confidence_adaptation"]["per_view_thresholds"]["SH"], 1.75
        )
        self.assertEqual(stats["exported_points_by_view"], {"MH": 1, "SH": 1})

    def test_bounded_depth_edge_adaptation_recovers_only_starved_view(self):
        points = np.zeros((2, 1, 2, 3), dtype=np.float32)
        points[..., 2] = 1.0
        images = np.zeros((2, 3, 1, 2), dtype=np.float32)
        confidence = np.ones((2, 1, 2), dtype=np.float32)
        depth = np.asarray(
            [[[[1.0], [1.0]]], [[[1.0], [1.04]]]], dtype=np.float32
        )

        selected = pilot.filter_point_cloud(
            points,
            images,
            confidence,
            depth=depth,
            confidence_percentile=0.0,
            max_points=10,
            require_all_views=True,
        )

        stats = selected["stats"]
        self.assertEqual(
            stats["stage_counts_by_view"]["after_depth_edge_initial"],
            {"MH": 2, "SH": 0},
        )
        self.assertEqual(
            stats["depth_edge_adaptation"]["mode_by_view"],
            {"MH": "initial_rtol", "SH": "bounded_rtol_adaptation"},
        )
        self.assertLessEqual(
            stats["depth_edge_adaptation"]["selected_rtol_by_view"]["SH"],
            pilot.MAX_ADAPTIVE_DEPTH_EDGE_RTOL,
        )
        self.assertEqual(
            stats["depth_edge_adaptation"][
                "finite_positive_depth_fallback_views"
            ],
            [],
        )
        self.assertEqual(stats["exported_points_by_view"], {"MH": 2, "SH": 2})

    def test_depth_edge_terminal_fallback_preserves_positive_depth_gate(self):
        points = np.zeros((2, 1, 2, 3), dtype=np.float32)
        points[..., 2] = 1.0
        images = np.zeros((2, 3, 1, 2), dtype=np.float32)
        confidence = np.ones((2, 1, 2), dtype=np.float32)
        depth = np.asarray(
            [[[[1.0], [1.0]]], [[[1.0], [2.0]]]], dtype=np.float32
        )

        selected = pilot.filter_point_cloud(
            points,
            images,
            confidence,
            depth=depth,
            confidence_percentile=0.0,
            max_points=10,
            require_all_views=True,
        )

        stats = selected["stats"]
        policy = stats["depth_edge_adaptation"]
        self.assertEqual(
            policy["finite_positive_depth_fallback_views"], ["SH"]
        )
        self.assertEqual(
            policy["mode_by_view"]["SH"], "finite_positive_depth_fallback"
        )
        self.assertIsNone(policy["selected_rtol_by_view"]["SH"])
        self.assertEqual(policy["maximum_rtol"], 0.12)
        self.assertFalse(policy["unbounded_relaxation_allowed"])
        self.assertTrue(np.all(selected["input_depth"] > pilot.MIN_POSITIVE_DEPTH))
        self.assertEqual(stats["exported_points_by_view"], {"MH": 2, "SH": 2})

    def test_fallback_never_restores_nonfinite_or_nonpositive_samples(self):
        points = np.zeros((2, 1, 7, 3), dtype=np.float32)
        points[..., 2] = 1.0
        points[1, 0, 0] = np.nan
        images = np.zeros((2, 3, 1, 7), dtype=np.float32)
        confidence = np.ones((2, 1, 7), dtype=np.float32)
        confidence[1, 0, 4] = np.inf
        confidence[1, 0, 5] = 0.0
        depth = np.ones((2, 1, 7, 1), dtype=np.float32)
        depth[1, 0, :, 0] = [1.0, np.nan, 0.0, -1.0, 1.0, 1.0, 1.0]

        selected = pilot.filter_point_cloud(
            points,
            images,
            confidence,
            depth=depth,
            confidence_percentile=0.0,
            max_points=20,
            filter_depth_edges=False,
            require_all_views=True,
        )

        sh_selected = selected["view_indices"] == 1
        self.assertEqual(int(sh_selected.sum()), 1)
        np.testing.assert_array_equal(selected["input_depth"][sh_selected], [1.0])
        self.assertEqual(
            selected["stats"]["stage_counts_by_view"]["positive_depth"]["SH"],
            1,
        )
        self.assertTrue(
            all(selected["stats"]["safety"]["invariants"].values())
        )

    def test_required_view_with_only_invalid_depth_is_rejected(self):
        points = np.zeros((2, 1, 3, 3), dtype=np.float32)
        images = np.zeros((2, 3, 1, 3), dtype=np.float32)
        confidence = np.ones((2, 1, 3), dtype=np.float32)
        depth = np.ones((2, 1, 3, 1), dtype=np.float32)
        depth[1, 0, :, 0] = [np.nan, 0.0, -1.0]

        with self.assertRaisesRegex(
            pilot.VGGTOmegaPilotError,
            "retained no samples from: SH after finite/positive-depth safety checks",
        ):
            pilot.filter_point_cloud(
                points,
                images,
                confidence,
                depth=depth,
                confidence_percentile=0.0,
                max_points=10,
                require_all_views=True,
            )

    def test_invalid_depth_edge_tolerances_are_rejected(self):
        points = np.zeros((2, 1, 1, 3), dtype=np.float32)
        images = np.zeros((2, 3, 1, 1), dtype=np.float32)
        confidence = np.ones((2, 1, 1), dtype=np.float32)
        depth = np.ones((2, 1, 1, 1), dtype=np.float32)

        for bad_rtol in (np.nan, np.inf, 0.0, -0.1, 0.120001):
            with self.subTest(depth_edge_rtol=bad_rtol):
                with self.assertRaisesRegex(ValueError, "depth_edge_rtol"):
                    pilot.filter_point_cloud(
                        points,
                        images,
                        confidence,
                        depth=depth,
                        depth_edge_rtol=bad_rtol,
                        require_all_views=True,
                    )

    def test_stratified_cap_preserves_both_required_views(self):
        points = np.zeros((2, 1, 5, 3), dtype=np.float32)
        points[..., 2] = 1.0
        images = np.zeros((2, 3, 1, 5), dtype=np.float32)
        confidence = np.ones((2, 1, 5), dtype=np.float32)
        depth = np.ones((2, 1, 5, 1), dtype=np.float32)

        selected = pilot.filter_point_cloud(
            points,
            images,
            confidence,
            depth=depth,
            confidence_percentile=0.0,
            max_points=2,
            require_all_views=True,
        )

        self.assertEqual(
            selected["stats"]["exported_points_by_view"], {"MH": 1, "SH": 1}
        )
        self.assertTrue(selected["stats"]["stratified_max_points_limit_used"])

        with self.assertRaisesRegex(
            pilot.VGGTOmegaPilotError, "cannot retain 2 required views"
        ):
            pilot.filter_point_cloud(
                points,
                images,
                confidence,
                depth=depth,
                confidence_percentile=0.0,
                max_points=1,
                require_all_views=True,
            )

    def test_object_filter_rejects_zero_retained_sh_points(self):
        points = np.zeros((2, 1, 2, 3), dtype=np.float32)
        images = np.zeros((2, 3, 1, 2), dtype=np.float32)
        confidence = np.ones((2, 1, 2), dtype=np.float32)
        depth = np.ones((2, 1, 2, 1), dtype=np.float32)
        masks = np.zeros((2, 1, 2), dtype=bool)
        masks[0, 0, 0] = True

        with self.assertRaisesRegex(
            pilot.VGGTOmegaPilotError, "retained no samples from: SH"
        ):
            pilot.filter_point_cloud(
                points,
                images,
                confidence,
                pixel_mask=masks,
                depth=depth,
                confidence_percentile=0.0,
                max_points=10,
                require_all_views=True,
            )

    def test_geometry_metadata_never_claims_a_mesh_or_metric_scale(self):
        contract = pilot.geometry_contract()

        self.assertEqual(contract["representation"], "colored_point_cloud")
        self.assertFalse(contract["is_triangle_mesh"])
        self.assertFalse(contract["is_watertight"])
        self.assertFalse(contract["collision_ready"])
        self.assertEqual(contract["scale"], "relative_non_metric")
        self.assertFalse(contract["metric_scale_verified"])
        self.assertFalse(contract["provided_calibration_applied"])


if __name__ == "__main__":
    unittest.main()
