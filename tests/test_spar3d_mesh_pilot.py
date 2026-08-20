"""Focused, weight-free tests for the direct SPAR3D pilot runner."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "run_spar3d_mesh_pilot.py"
SPEC = importlib.util.spec_from_file_location("run_spar3d_mesh_pilot", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def write_rgba(path: Path, *, mode: str = "RGBA", size=(512, 512)) -> None:
    if mode == "RGBA":
        image = Image.new("RGBA", size, (40, 80, 120, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((150, 80, 360, 440), fill=(80, 210, 60, 255))
    else:
        image = Image.new(mode, size, (40, 80, 120))
    image.save(path)


def write_manifest(path: Path, input_path: Path, *, label: str = "Choco") -> None:
    payload = {
        "bundle": {"output_root": str(path.parent)},
        "selection": {
            "object_label": label,
            "mh_frame_index": 187,
            "sh_frame_index": 192,
            "mh_role": "primary_geometry_and_output",
            "sh_role": "auxiliary_evidence",
        },
        "outputs": {
            "spar3d_rgba_crop": {
                "path": str(input_path.resolve()),
                "sha256": runner.sha256_file(input_path),
                "bytes": input_path.stat().st_size,
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_weights(path: Path) -> None:
    path.mkdir()
    (path / "config.yaml").write_text("test: true\n", encoding="utf-8")
    with (path / "model.safetensors").open("wb") as handle:
        # Sparse file: validates the official byte contract without allocating 6.82 GiB.
        handle.truncate(runner.EXPECTED_MODEL_BYTES)


def write_fake_upstream(path: Path) -> None:
    (path / "spar3d").mkdir(parents=True)
    (path / "spar3d" / "system.py").write_text("# fake\n", encoding="utf-8")
    (path / "load" / "tets").mkdir(parents=True)
    (path / "load" / "tets" / "160_tets.npz").write_bytes(b"fake")


def write_existing_output_bundle(path: Path) -> None:
    path.mkdir()
    mesh = path / "mesh.glb"
    points = path / "points.ply"
    mesh.write_bytes(b"old-glb")
    points.write_bytes(b"old-ply")
    report = {
        "schema_version": 1,
        "status": "complete",
        "method": "spar3d_direct_prepared_rgba_low_vram",
        "outputs": {
            "mesh_glb": {
                "path": str(mesh.resolve()),
                "bytes": mesh.stat().st_size,
                "sha256": runner.sha256_file(mesh),
            },
            "points_ply": {
                "path": str(points.resolve()),
                "bytes": points.stat().st_size,
                "sha256": runner.sha256_file(points),
            },
            "report": str((path / "report.json").resolve()),
        },
    }
    (path / "report.json").write_text(json.dumps(report), encoding="utf-8")


def fake_runtime_asset_validation(alpha_clip_dir, hf_home):
    return (
        Path(alpha_clip_dir).resolve(),
        Path(hf_home).resolve(),
        {
            "network_access_allowed": False,
            "alpha_clip": {"sha256_verified": True},
            "dinov2": {"commit": runner.DINOV2_COMMIT},
        },
    )


class FakeMesh:
    vertices = np.asarray(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.0, 0.5, -0.5],
            [0.0, 0.0, 0.5],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    bounds = np.asarray([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    extents = np.asarray([1.0, 1.0, 1.0])
    is_watertight = True
    is_winding_consistent = True
    is_volume = True
    volume = 1.0 / 3.0

    def export(self, path, include_normals=False):
        assert include_normals
        Path(path).write_bytes(b"fake-glb")


class FakePointCloud:
    vertices = np.asarray([[-0.25, 0.0, 0.0], [0.25, 0.1, 0.2]])

    def export(self, path):
        Path(path).write_bytes(b"fake-ply")


class FakeModel:
    from_pretrained_args = None
    run_kwargs = None
    image_mode = None
    image_size = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def run_image(self, image, **kwargs):
        type(self).image_mode = image.mode
        type(self).image_size = image.size
        type(self).run_kwargs = kwargs
        return FakeMesh(), {"point_clouds": [FakePointCloud()]}


class FakeSPAR3D:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        FakeModel.from_pretrained_args = (args, kwargs)
        return FakeModel()


class FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def is_bf16_supported():
        return True

    @staticmethod
    def reset_peak_memory_stats(_device):
        return None

    @staticmethod
    def synchronize(_device):
        return None

    @staticmethod
    def mem_get_info(_device):
        return 12_000_000_000, 16_000_000_000

    @staticmethod
    def max_memory_allocated(_device):
        return 7_000_000_000

    @staticmethod
    def max_memory_reserved(_device):
        return 7_500_000_000

    @staticmethod
    def get_device_name(_device):
        return "Fake RTX 5080"

    @staticmethod
    def get_device_capability(_device):
        return (12, 0)


class FakeVersion:
    cuda = "12.8"


class FakeTorch:
    __version__ = "2.7.1+cu128"
    version = FakeVersion()
    cuda = FakeCuda()
    bfloat16 = "bfloat16"

    @staticmethod
    def no_grad():
        return nullcontext()

    @staticmethod
    def autocast(*, device_type, dtype):
        assert device_type == "cuda"
        assert dtype == FakeTorch.bfloat16
        return nullcontext()


class Spar3DInputValidationTests(unittest.TestCase):
    def test_official_provenance_constants_are_pinned(self):
        self.assertEqual(
            runner.SPAR3D_HF_COMMIT,
            "5699918cb34f55cd7d828493d2725f3038313761",
        )
        self.assertEqual(
            runner.SPAR3D_CONFIG_ETAG,
            "691b7b50f13599b03ea5eaaa5fdc01316c31bbf5",
        )
        self.assertEqual(runner.EXPECTED_ALPHA_CLIP_BYTES, 934_088_680)
        self.assertEqual(
            runner.EXPECTED_ALPHA_CLIP_SHA256,
            "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02",
        )
        self.assertEqual(
            runner.DINOV2_COMMIT, "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
        )
        self.assertEqual(runner.EXPECTED_DINOV2_MODEL_BYTES, 1_217_522_888)
        self.assertEqual(
            runner.EXPECTED_DINOV2_MODEL_SHA256,
            "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23",
        )
        self.assertEqual(runner.EXPECTED_DINOV2_CONFIG_BYTES, 549)
        self.assertEqual(
            runner.EXPECTED_DINOV2_CONFIG_SHA256,
            "12df51c069a2dc1305e34ba71ef58bc2407ea553b75f4722a1715c1bce3bbed0",
        )

    def test_requires_prepared_rgba_and_exact_512_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgb = root / "rgb.png"
            wrong_size = root / "small.png"
            write_rgba(rgb, mode="RGB")
            write_rgba(wrong_size, size=(256, 512))
            with self.assertRaisesRegex(runner.PilotInputError, "already be RGBA"):
                runner.validate_rgba_input(rgb)
            with self.assertRaisesRegex(runner.PilotInputError, "512x512"):
                runner.validate_rgba_input(wrong_size)

    def test_rejects_fully_opaque_alpha_instead_of_resegmenting(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "opaque.png"
            Image.new("RGBA", (512, 512), (30, 40, 50, 255)).save(path)
            with self.assertRaisesRegex(runner.PilotInputError, "fully opaque"):
                runner.validate_rgba_input(path)

    def test_same_size_wrong_model_sha_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "weights"
            weights.mkdir()
            config = weights / "config.yaml"
            model = weights / "model.safetensors"
            config.write_text("official-for-test: true\n", encoding="utf-8")
            model.write_bytes(b"same-size-but-wrong-checkpoint")
            config_sha = runner.sha256_file(config)

            with (
                mock.patch.object(runner, "EXPECTED_MODEL_BYTES", model.stat().st_size),
                mock.patch.object(runner, "EXPECTED_CONFIG_SHA256", config_sha),
                self.assertRaisesRegex(runner.PilotInputError, "model.safetensors SHA-256"),
            ):
                runner.validate_weights_dir(weights)

    def test_wrong_config_sha_is_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "weights"
            weights.mkdir()
            config = weights / "config.yaml"
            model = weights / "model.safetensors"
            config.write_text("wrong: config\n", encoding="utf-8")
            model.write_bytes(b"model")

            with (
                mock.patch.object(runner, "EXPECTED_MODEL_BYTES", model.stat().st_size),
                self.assertRaisesRegex(runner.PilotInputError, "config.yaml SHA-256"),
            ):
                runner.validate_weights_dir(weights)

    def test_verified_weight_records_report_actual_and_expected_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "weights"
            weights.mkdir()
            config = weights / "config.yaml"
            model = weights / "model.safetensors"
            config.write_text("verified: config\n", encoding="utf-8")
            model.write_bytes(b"verified-model")
            config_sha = runner.sha256_file(config)
            model_sha = runner.sha256_file(model)

            with (
                mock.patch.object(runner, "EXPECTED_MODEL_BYTES", model.stat().st_size),
                mock.patch.object(runner, "EXPECTED_CONFIG_SHA256", config_sha),
                mock.patch.object(runner, "EXPECTED_MODEL_SHA256", model_sha),
            ):
                _, records = runner.validate_weights_dir(weights)

            for key, expected_sha in (
                ("config", config_sha),
                ("model", model_sha),
            ):
                self.assertTrue(records[key]["sha256_verified"])
                self.assertEqual(records[key]["sha256"], expected_sha)
                self.assertEqual(records[key]["expected_sha256"], expected_sha)
            self.assertEqual(
                records["pinned_source"]["commit"], runner.SPAR3D_HF_COMMIT
            )
            self.assertEqual(
                records["pinned_source"]["config_etag"], runner.SPAR3D_CONFIG_ETAG
            )
            self.assertTrue(
                records["pinned_source"]["identity_bound_by_local_content_sha256"]
            )

    def test_weight_files_cannot_escape_local_weights_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "weights"
            weights.mkdir()
            outside_config = root / "outside-config.yaml"
            outside_model = root / "outside-model.safetensors"
            outside_config.write_text("outside: true\n", encoding="utf-8")
            outside_model.write_bytes(b"outside-model")
            (weights / "config.yaml").symlink_to(outside_config)
            (weights / "model.safetensors").symlink_to(outside_model)

            with self.assertRaisesRegex(runner.PilotInputError, "config.yaml escapes"):
                runner.validate_weights_dir(weights)

    def test_required_upstream_files_cannot_escape_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            upstream = root / "SPAR3D"
            outside = root / "outside-system.py"
            outside.write_text("# outside\n", encoding="utf-8")
            (upstream / "spar3d").mkdir(parents=True)
            (upstream / "spar3d" / "system.py").symlink_to(outside)
            (upstream / "load" / "tets").mkdir(parents=True)
            (upstream / "load" / "tets" / "160_tets.npz").write_bytes(b"fake")

            with self.assertRaisesRegex(runner.PilotInputError, "files escape"):
                runner.validate_spar3d_repo(upstream)


class Spar3DRuntimeAssetValidationTests(unittest.TestCase):
    def _write_assets(self, root: Path):
        alpha_dir = root / "alpha"
        hf_home = root / "hf"
        alpha_dir.mkdir()
        alpha_path = alpha_dir / runner.ALPHA_CLIP_FILENAME
        alpha_path.write_bytes(b"alpha-clip")

        cache = hf_home / "hub" / runner.DINOV2_CACHE_DIRECTORY
        snapshot = cache / "snapshots" / runner.DINOV2_COMMIT
        (cache / "refs").mkdir(parents=True)
        snapshot.mkdir(parents=True)
        (cache / "refs" / "main").write_text(runner.DINOV2_COMMIT)
        model = snapshot / runner.DINOV2_MODEL_FILENAME
        config = snapshot / runner.DINOV2_CONFIG_FILENAME
        model.write_bytes(b"dinov2-model")
        config.write_bytes(b"dinov2-config")
        return alpha_dir, hf_home, alpha_path, model, config

    def test_exact_auxiliary_assets_are_verified_and_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            alpha_dir, hf_home, alpha, model, config = self._write_assets(
                Path(temporary)
            )
            with (
                mock.patch.object(runner, "EXPECTED_ALPHA_CLIP_BYTES", alpha.stat().st_size),
                mock.patch.object(runner, "EXPECTED_ALPHA_CLIP_SHA256", runner.sha256_file(alpha)),
                mock.patch.object(runner, "EXPECTED_DINOV2_MODEL_BYTES", model.stat().st_size),
                mock.patch.object(
                    runner, "EXPECTED_DINOV2_MODEL_SHA256", runner.sha256_file(model)
                ),
                mock.patch.object(runner, "EXPECTED_DINOV2_CONFIG_BYTES", config.stat().st_size),
                mock.patch.object(
                    runner, "EXPECTED_DINOV2_CONFIG_SHA256", runner.sha256_file(config)
                ),
            ):
                _, _, records = runner.validate_runtime_assets(alpha_dir, hf_home)

            self.assertFalse(records["network_access_allowed"])
            self.assertTrue(records["alpha_clip"]["sha256_verified"])
            self.assertTrue(records["dinov2"]["model"]["sha256_verified"])
            self.assertTrue(records["dinov2"]["config"]["sha256_verified"])
            self.assertEqual(records["dinov2"]["commit"], runner.DINOV2_COMMIT)

    def test_standard_hf_snapshot_links_to_in_cache_blobs_are_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alpha_dir, hf_home, alpha, model, config = self._write_assets(root)
            cache = hf_home / "hub" / runner.DINOV2_CACHE_DIRECTORY
            blobs = cache / "blobs"
            blobs.mkdir()
            model_blob = blobs / "model-blob"
            config_blob = blobs / "config-blob"
            model.replace(model_blob)
            config.replace(config_blob)
            model.symlink_to(model_blob)
            config.symlink_to(config_blob)

            with (
                mock.patch.object(
                    runner, "EXPECTED_ALPHA_CLIP_BYTES", alpha.stat().st_size
                ),
                mock.patch.object(
                    runner, "EXPECTED_ALPHA_CLIP_SHA256", runner.sha256_file(alpha)
                ),
                mock.patch.object(
                    runner, "EXPECTED_DINOV2_MODEL_BYTES", model_blob.stat().st_size
                ),
                mock.patch.object(
                    runner,
                    "EXPECTED_DINOV2_MODEL_SHA256",
                    runner.sha256_file(model_blob),
                ),
                mock.patch.object(
                    runner,
                    "EXPECTED_DINOV2_CONFIG_BYTES",
                    config_blob.stat().st_size,
                ),
                mock.patch.object(
                    runner,
                    "EXPECTED_DINOV2_CONFIG_SHA256",
                    runner.sha256_file(config_blob),
                ),
            ):
                _, _, records = runner.validate_runtime_assets(alpha_dir, hf_home)

            self.assertEqual(
                Path(records["dinov2"]["model"]["path"]), model_blob.resolve()
            )
            self.assertEqual(
                Path(records["dinov2"]["config"]["path"]), config_blob.resolve()
            )

    def test_same_size_wrong_auxiliary_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            alpha_dir, hf_home, alpha, model, config = self._write_assets(
                Path(temporary)
            )
            expected_alpha_sha = runner.sha256_file(alpha)
            alpha.write_bytes(b"wrong-clip")
            self.assertEqual(alpha.stat().st_size, len(b"alpha-clip"))
            with (
                mock.patch.object(runner, "EXPECTED_ALPHA_CLIP_BYTES", alpha.stat().st_size),
                mock.patch.object(runner, "EXPECTED_ALPHA_CLIP_SHA256", expected_alpha_sha),
                self.assertRaisesRegex(runner.PilotInputError, "AlphaCLIP.*SHA-256"),
            ):
                runner.validate_runtime_assets(alpha_dir, hf_home)

    def test_alpha_checkpoint_link_cannot_escape_cache_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alpha_dir, hf_home, alpha, _model, _config = self._write_assets(root)
            alpha.unlink()
            outside = root / "outside-alpha"
            outside.write_bytes(b"alpha-clip")
            alpha.symlink_to(outside)

            with self.assertRaisesRegex(runner.PilotInputError, "AlphaCLIP.*escapes"):
                runner.validate_runtime_assets(alpha_dir, hf_home)

    def test_dinov2_ref_link_cannot_escape_cache_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alpha_dir, hf_home, _alpha, _model, _config = self._write_assets(root)
            ref = hf_home / "hub" / runner.DINOV2_CACHE_DIRECTORY / "refs" / "main"
            ref.unlink()
            outside = root / "outside-ref"
            outside.write_text(runner.DINOV2_COMMIT, encoding="utf-8")
            ref.symlink_to(outside)

            with self.assertRaisesRegex(runner.PilotInputError, "cache ref escapes"):
                runner.validate_runtime_assets(alpha_dir, hf_home)

    def test_dinov2_ref_and_cache_link_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alpha_dir, hf_home, alpha, model, config = self._write_assets(root)
            ref = (
                hf_home
                / "hub"
                / runner.DINOV2_CACHE_DIRECTORY
                / "refs"
                / "main"
            )
            ref.write_text("different-commit")
            with self.assertRaisesRegex(runner.PilotInputError, "not pinned"):
                runner.validate_runtime_assets(alpha_dir, hf_home)

            ref.write_text(runner.DINOV2_COMMIT)
            model.unlink()
            outside = root / "outside-model"
            outside.write_bytes(b"dinov2-model")
            model.symlink_to(outside)
            with self.assertRaisesRegex(runner.PilotInputError, "escapes"):
                runner.validate_runtime_assets(alpha_dir, hf_home)

    def test_offline_environment_hides_credentials_and_restores_on_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alpha_dir = root / "alpha"
            hf_home = root / "hf"
            alpha_dir.mkdir()
            hf_home.mkdir()
            before = {
                "ALPHA_CLIP_PATH": "before-alpha",
                "HF_HOME": "before-hf",
                "TRANSFORMERS_CACHE": "before-transformers-cache",
                "HF_TOKEN_PATH": "before-token-path",
                "HF_TOKEN": "before-hf-token",
                "HUGGING_FACE_HUB_TOKEN": "before-hub-token",
                "AWS_ACCESS_KEY_ID": "before-access-key",
                "AWS_SECRET_ACCESS_KEY": "before-secret-key",
                "AWS_SESSION_TOKEN": "before-session-token",
                "AWS_PROFILE": "before-profile",
                "AWS_SHARED_CREDENTIALS_FILE": "before-credentials-file",
                "AWS_CONFIG_FILE": "before-config-file",
                "AWS_CREDENTIAL_FILE": "before-legacy-credentials-file",
                "BOTO_CONFIG": "before-boto-config",
                "AWS_WEB_IDENTITY_TOKEN_FILE": "before-web-token",
                "AWS_CONTAINER_CREDENTIALS_FULL_URI": "before-container-uri",
                "AWS_EC2_METADATA_DISABLED": "false",
                "UNRELATED": "unchanged",
            }
            with mock.patch.dict(os.environ, before, clear=True):
                with self.assertRaisesRegex(RuntimeError, "forced failure"):
                    with runner.offline_runtime_environment(alpha_dir, hf_home):
                        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
                        for key in (
                            "HF_HUB_CACHE",
                            "HUGGINGFACE_HUB_CACHE",
                            "TRANSFORMERS_CACHE",
                            "PYTORCH_TRANSFORMERS_CACHE",
                            "PYTORCH_PRETRAINED_BERT_CACHE",
                        ):
                            self.assertEqual(os.environ[key], str(hf_home / "hub"))
                        self.assertEqual(
                            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1"
                        )
                        self.assertEqual(os.environ["AWS_EC2_METADATA_DISABLED"], "true")
                        self.assertFalse(Path(os.environ["HF_TOKEN_PATH"]).exists())
                        for key in (
                            "AWS_SHARED_CREDENTIALS_FILE",
                            "AWS_CONFIG_FILE",
                            "AWS_CREDENTIAL_FILE",
                            "BOTO_CONFIG",
                        ):
                            self.assertFalse(Path(os.environ[key]).exists())
                        for key in (
                            "HF_TOKEN",
                            "HUGGING_FACE_HUB_TOKEN",
                            "AWS_ACCESS_KEY_ID",
                            "AWS_SECRET_ACCESS_KEY",
                            "AWS_SESSION_TOKEN",
                            "AWS_PROFILE",
                            "AWS_WEB_IDENTITY_TOKEN_FILE",
                            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
                        ):
                            self.assertNotIn(key, os.environ)
                        raise RuntimeError("forced failure")
                self.assertEqual(dict(os.environ), before)


class Spar3DOutputPathSafetyTests(unittest.TestCase):
    def test_rejects_bidirectional_protected_path_overlaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            bundle = repository / "data" / "inputs"
            weights = repository / "weights"
            upstream = repository / "third_party" / "SPAR3D"
            bundle.mkdir(parents=True)
            weights.mkdir()
            upstream.mkdir(parents=True)
            rgba = bundle / "input.png"
            manifest = bundle / "manifest.json"
            rgba.write_bytes(b"rgba")
            manifest.write_text("{}", encoding="utf-8")
            dedicated = repository / "results" / "spar3d"

            unsafe_paths = (
                rgba,
                rgba / "child",
                manifest,
                bundle,
                bundle / "child-output",
                weights,
                weights / "child-output",
                upstream,
                upstream / "child-output",
                repository,
                repository / "arbitrary-output",
                root,
            )
            for unsafe in unsafe_paths:
                with self.subTest(unsafe=unsafe), self.assertRaisesRegex(
                    runner.PilotInputError, "unsafe output directory"
                ):
                    runner.validate_output_directory(
                        unsafe,
                        input_rgba=rgba,
                        input_manifest=manifest,
                        input_bundle=bundle,
                        weights_dir=weights,
                        spar3d_repo=upstream,
                        repo_root=repository,
                        dedicated_repo_output_root=dedicated,
                    )

            self.assertEqual(
                runner.validate_output_directory(
                    dedicated / "run-1",
                    input_rgba=rgba,
                    input_manifest=manifest,
                    input_bundle=bundle,
                    weights_dir=weights,
                    spar3d_repo=upstream,
                    repo_root=repository,
                    dedicated_repo_output_root=dedicated,
                ),
                (dedicated / "run-1").resolve(),
            )
            external = root / "external" / "run-1"
            self.assertEqual(
                runner.validate_output_directory(
                    external,
                    input_rgba=rgba,
                    input_manifest=manifest,
                    input_bundle=bundle,
                    weights_dir=weights,
                    spar3d_repo=upstream,
                    repo_root=repository,
                    dedicated_repo_output_root=dedicated,
                ),
                external.resolve(),
            )

    def test_rejects_broad_and_symbolic_link_output_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            bundle = repository / "inputs"
            weights = repository / "weights"
            upstream = repository / "SPAR3D"
            external = root / "external"
            bundle.mkdir(parents=True)
            weights.mkdir()
            upstream.mkdir()
            external.mkdir()
            rgba = bundle / "input.png"
            manifest = bundle / "manifest.json"
            rgba.write_bytes(b"rgba")
            manifest.write_text("{}", encoding="utf-8")
            output_link = root / "output-link"
            output_link.symlink_to(external, target_is_directory=True)

            common = {
                "input_rgba": rgba,
                "input_manifest": manifest,
                "input_bundle": bundle,
                "weights_dir": weights,
                "spar3d_repo": upstream,
                "repo_root": repository,
                "dedicated_repo_output_root": repository / "results",
            }
            with self.assertRaisesRegex(runner.PilotInputError, "broad output"):
                runner.validate_output_directory(Path(tempfile.gettempdir()), **common)
            with self.assertRaisesRegex(runner.PilotInputError, "symbolic link"):
                runner.validate_output_directory(output_link, **common)

    def test_existing_bundle_must_be_owned_and_hash_consistent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unrelated = root / "unrelated"
            unrelated.mkdir()
            (unrelated / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(runner.PilotInputError, "unrecognized"):
                runner.validate_existing_output_bundle(unrelated)

            output = root / "output"
            write_existing_output_bundle(output)
            (output / "mesh.glb").write_bytes(b"tampered-same-bundle")
            with self.assertRaisesRegex(runner.PilotInputError, "byte count"):
                runner.validate_existing_output_bundle(output)

    def test_output_cannot_be_nested_inside_an_existing_result_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "inputs"
            weights = root / "weights"
            upstream = root / "SPAR3D"
            existing = root / "existing-result"
            bundle.mkdir()
            weights.mkdir()
            upstream.mkdir()
            rgba = bundle / "input.png"
            manifest = bundle / "manifest.json"
            rgba.write_bytes(b"rgba")
            manifest.write_text("{}", encoding="utf-8")
            write_existing_output_bundle(existing)

            with self.assertRaisesRegex(runner.PilotInputError, "nested inside"):
                runner.validate_output_directory(
                    existing / "another-run",
                    input_rgba=rgba,
                    input_manifest=manifest,
                    input_bundle=bundle,
                    weights_dir=weights,
                    spar3d_repo=upstream,
                    repo_root=root / "unrelated-repository",
                    dedicated_repo_output_root=root / "dedicated",
                )


class Spar3DDirectRunnerTests(unittest.TestCase):
    def test_test_only_bypasses_are_not_reachable_with_production_runtime(self):
        arguments = {
            "input_rgba": "missing.png",
            "input_manifest": "missing.json",
            "weights_dir": "missing-weights",
            "spar3d_repo": "missing-repo",
            "output_dir": "missing-output",
        }
        with self.assertRaisesRegex(runner.PilotInputError, "injected test runtime"):
            runner.run_job(**arguments, allow_unverified_test_weights=True)
        with self.assertRaisesRegex(runner.PilotInputError, "injected test runtime"):
            runner.run_job(
                **arguments,
                runtime_assets_validator=fake_runtime_asset_validation,
            )

        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                runner.build_parser().parse_args(["--allow-unverified-test-weights"])

    def test_direct_low_vram_call_and_canonical_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            input_path = inputs / "spar3d_input_rgba_512.png"
            manifest = inputs / "manifest.json"
            weights = root / "weights"
            upstream = root / "SPAR3D"
            output = root / "output"
            write_rgba(input_path)
            write_manifest(manifest, input_path)
            write_weights(weights)
            write_fake_upstream(upstream)

            report = runner.run_job(
                input_rgba=input_path,
                input_manifest=manifest,
                weights_dir=weights,
                spar3d_repo=upstream,
                output_dir=output,
                allow_unverified_test_weights=True,
                runtime_loader=lambda _repo: (FakeTorch, FakeSPAR3D),
                runtime_assets_validator=fake_runtime_asset_validation,
            )

            args, kwargs = FakeModel.from_pretrained_args
            self.assertEqual(args, (str(weights.resolve()),))
            self.assertEqual(kwargs["config_name"], "config.yaml")
            self.assertEqual(kwargs["weight_name"], "model.safetensors")
            self.assertTrue(kwargs["low_vram_mode"])
            self.assertEqual(
                FakeModel.run_kwargs,
                {
                    "bake_resolution": 512,
                    "remesh": "none",
                    "vertex_count": -1,
                    "return_points": True,
                },
            )
            self.assertEqual(FakeModel.image_mode, "RGBA")
            self.assertEqual(FakeModel.image_size, (512, 512))
            self.assertEqual((output / "mesh.glb").read_bytes(), b"fake-glb")
            self.assertEqual((output / "points.ply").read_bytes(), b"fake-ply")
            saved = json.loads((output / "report.json").read_text())
            self.assertEqual(saved, report)
            self.assertFalse(saved["metric_scale_verified"])
            self.assertEqual(saved["camera_alignment"], "none")
            self.assertFalse(saved["collision_ready"])
            self.assertFalse(saved["runtime"]["background_remover_loaded"])
            self.assertFalse(saved["runtime"]["network_access_allowed"])
            self.assertTrue(saved["runtime"]["credential_environment_disabled"])
            self.assertFalse(saved["runtime_assets"]["network_access_allowed"])
            self.assertEqual(saved["runtime"]["random_seed"], 42)
            self.assertEqual(saved["runtime"]["peak_vram_allocated_bytes"], 7_000_000_000)
            self.assertFalse(saved["weights"]["config"]["sha256_verified"])
            self.assertFalse(saved["weights"]["model"]["sha256_verified"])
            self.assertEqual(
                saved["weights"]["pinned_source"]["commit"],
                runner.SPAR3D_HF_COMMIT,
            )
            self.assertTrue(any("Sim(3)" in warning for warning in saved["warnings"]))

    def test_runtime_asset_hash_failure_precedes_runtime_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            input_path = inputs / "spar3d_input_rgba_512.png"
            manifest = inputs / "manifest.json"
            weights = root / "weights"
            upstream = root / "SPAR3D"
            output = root / "output"
            alpha_dir = root / "alpha"
            hf_home = root / "hf"
            alpha_dir.mkdir()
            alpha = alpha_dir / runner.ALPHA_CLIP_FILENAME
            alpha.write_bytes(b"alpha-clip")
            expected_alpha_sha = runner.sha256_file(alpha)
            alpha.write_bytes(b"wrong-clip")
            cache = hf_home / "hub" / runner.DINOV2_CACHE_DIRECTORY
            snapshot = cache / "snapshots" / runner.DINOV2_COMMIT
            (cache / "refs").mkdir(parents=True)
            snapshot.mkdir(parents=True)
            (cache / "refs" / "main").write_text(
                runner.DINOV2_COMMIT, encoding="utf-8"
            )
            (snapshot / runner.DINOV2_MODEL_FILENAME).write_bytes(b"dino-model")
            (snapshot / runner.DINOV2_CONFIG_FILENAME).write_bytes(b"dino-config")
            write_rgba(input_path)
            write_manifest(manifest, input_path)
            write_weights(weights)
            write_fake_upstream(upstream)
            runtime_called = False

            def forbidden_runtime(_repo):
                nonlocal runtime_called
                runtime_called = True
                raise AssertionError("runtime loaded before asset hash validation")

            with (
                mock.patch.object(
                    runner, "EXPECTED_ALPHA_CLIP_BYTES", alpha.stat().st_size
                ),
                mock.patch.object(
                    runner, "EXPECTED_ALPHA_CLIP_SHA256", expected_alpha_sha
                ),
                self.assertRaisesRegex(runner.PilotInputError, "AlphaCLIP.*SHA-256"),
            ):
                runner.run_job(
                    input_rgba=input_path,
                    input_manifest=manifest,
                    weights_dir=weights,
                    spar3d_repo=upstream,
                    output_dir=output,
                    alpha_clip_dir=alpha_dir,
                    hf_home=hf_home,
                    allow_unverified_test_weights=True,
                    runtime_loader=forbidden_runtime,
                )
            self.assertFalse(runtime_called)

    def test_custom_validator_cannot_rebind_cache_into_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            input_path = inputs / "spar3d_input_rgba_512.png"
            manifest = inputs / "manifest.json"
            weights = root / "weights"
            upstream = root / "SPAR3D"
            output = root / "output"
            alpha_dir = root / "alpha"
            hf_home = root / "hf"
            write_rgba(input_path)
            write_manifest(manifest, input_path)
            write_weights(weights)
            write_fake_upstream(upstream)
            runtime_called = False

            def forbidden_runtime(_repo):
                nonlocal runtime_called
                runtime_called = True
                raise AssertionError("runtime must not load after cache-root rebinding")

            def rebinding_validator(_alpha, _hf):
                return output, hf_home, {"network_access_allowed": False}

            with self.assertRaisesRegex(runner.PilotInputError, "rebind"):
                runner.run_job(
                    input_rgba=input_path,
                    input_manifest=manifest,
                    weights_dir=weights,
                    spar3d_repo=upstream,
                    output_dir=output,
                    alpha_clip_dir=alpha_dir,
                    hf_home=hf_home,
                    allow_unverified_test_weights=True,
                    runtime_loader=forbidden_runtime,
                    runtime_assets_validator=rebinding_validator,
                )
            self.assertFalse(runtime_called)

    def test_overlap_is_rejected_before_overwrite_or_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            input_path = inputs / "spar3d_input_rgba_512.png"
            manifest = inputs / "manifest.json"
            weights = root / "weights"
            upstream = root / "SPAR3D"
            write_rgba(input_path)
            write_manifest(manifest, input_path)
            write_weights(weights)
            write_fake_upstream(upstream)

            runtime_called = False

            def forbidden_runtime(_repo):
                nonlocal runtime_called
                runtime_called = True
                raise AssertionError("runtime must not load for an unsafe output")

            for overwrite in (False, True):
                with self.subTest(overwrite=overwrite), self.assertRaisesRegex(
                    runner.PilotInputError, "unsafe output directory"
                ):
                    runner.run_job(
                        input_rgba=input_path,
                        input_manifest=manifest,
                        weights_dir=weights,
                        spar3d_repo=upstream,
                        output_dir=weights,
                        overwrite=overwrite,
                        allow_unverified_test_weights=True,
                        runtime_loader=forbidden_runtime,
                    )
            self.assertFalse(runtime_called)
            self.assertTrue((weights / "model.safetensors").is_file())

    def test_manifest_sha_mismatch_fails_before_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "spar3d_input_rgba_512.png"
            manifest = root / "manifest.json"
            write_rgba(input_path)
            write_manifest(manifest, input_path)
            payload = json.loads(manifest.read_text())
            payload["outputs"]["spar3d_rgba_crop"]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload))

            _, _, stats = runner.validate_rgba_input(input_path)
            with self.assertRaisesRegex(runner.PilotInputError, "SHA-256"):
                runner.validate_input_manifest(
                    manifest,
                    input_path=input_path.resolve(),
                    input_sha256=stats["sha256"],
                )

    def test_publish_directory_overwrite_replaces_then_removes_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            staging = root / "staging"
            write_existing_output_bundle(output)
            staging.mkdir()
            (staging / "new.txt").write_text("new")

            runner._publish_directory(staging, output, overwrite=True)

            self.assertFalse(staging.exists())
            self.assertFalse((root / ".output.backup").exists())
            self.assertFalse((output / "mesh.glb").exists())
            self.assertEqual((output / "new.txt").read_text(), "new")

    def test_publish_refuses_to_replace_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            staging = root / "staging"
            output.mkdir()
            staging.mkdir()
            (output / "keep.txt").write_text("keep", encoding="utf-8")
            (staging / "new.txt").write_text("new", encoding="utf-8")

            with self.assertRaisesRegex(runner.PilotInputError, "unrecognized"):
                runner._publish_directory(staging, output, overwrite=True)

            self.assertEqual((output / "keep.txt").read_text(), "keep")
            self.assertTrue(staging.is_dir())


if __name__ == "__main__":
    unittest.main()
