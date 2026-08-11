#!/usr/bin/env python3
"""Run SPAR3D from a provenance-recorded prepared RGBA object crop.

This runner intentionally does not import SPAR3D's ``run.py``.  The upstream
CLI constructs a separate background-removal network and has optional-remesher
argument handling that is irrelevant to this pilot.  Here the trusted alpha
channel prepared by ``prepare_mesh_sota_pilot_inputs.py`` is passed directly to
``SPAR3D.run_image``.

The exported GLB and PLY remain in SPAR3D's learned canonical coordinate frame.
They are not metric, are not registered to MH or SH, and are not collision-ready.
Those constraints are written explicitly to ``report.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import shutil
import sys
import tempfile
import time
import types
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = (
    REPO_ROOT / "8-5" / "mesh_sota_pilot" / "episode_1" / "choco" / "inputs"
)
DEFAULT_INPUT_RGBA = DEFAULT_INPUT_ROOT / "spar3d_input_rgba_512.png"
DEFAULT_INPUT_MANIFEST = DEFAULT_INPUT_ROOT / "manifest.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "8-5" / "mesh_sota_pilot" / "episode_1" / "choco" / "spar3d"
)
DEDICATED_OUTPUT_ROOT = REPO_ROOT / "8-5" / "mesh_sota_pilot" / "episode_1"
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "weights" / "spar3d"
DEFAULT_SPAR3D_REPO = REPO_ROOT / "third_party" / "SPAR3D"
DEFAULT_ALPHA_CLIP_DIR = (
    Path.home() / ".cache" / "skill2policy-models" / "alpha_clip"
)
DEFAULT_HF_HOME = Path.home() / ".cache" / "skill2policy-runtime-hf"

EXPECTED_INPUT_SIZE = (512, 512)
EXPECTED_MODEL_BYTES = 7_326_949_440
EXPECTED_MODEL_SHA256 = (
    "62673b63fd9dad425e74b213ffe8501262d9621d5174310ac33017747be31f58"
)
EXPECTED_CONFIG_SHA256 = (
    "2795e11a7a6cb381a442e07abfdb5a3c0cc20e711c6e7e349ba73bf343c2ebdd"
)
SPAR3D_HF_REPOSITORY = "stabilityai/stable-point-aware-3d"
SPAR3D_HF_COMMIT = "5699918cb34f55cd7d828493d2725f3038313761"
SPAR3D_CONFIG_ETAG = "691b7b50f13599b03ea5eaaa5fdc01316c31bbf5"
ALPHA_CLIP_FILENAME = "ViT-L-14-336px.pt"
EXPECTED_ALPHA_CLIP_BYTES = 934_088_680
EXPECTED_ALPHA_CLIP_SHA256 = (
    "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
)
DINOV2_REPOSITORY = "facebook/dinov2-large"
DINOV2_CACHE_DIRECTORY = "models--facebook--dinov2-large"
DINOV2_COMMIT = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
DINOV2_MODEL_FILENAME = "model.safetensors"
EXPECTED_DINOV2_MODEL_BYTES = 1_217_522_888
EXPECTED_DINOV2_MODEL_SHA256 = (
    "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23"
)
DINOV2_CONFIG_FILENAME = "config.json"
EXPECTED_DINOV2_CONFIG_BYTES = 549
EXPECTED_DINOV2_CONFIG_SHA256 = (
    "12df51c069a2dc1305e34ba71ef58bc2407ea553b75f4722a1715c1bce3bbed0"
)

CANONICAL_WARNINGS = (
    "SPAR3D output is a learned canonical reconstruction, not a metric object measurement.",
    "mesh.glb is not registered to MH or SH camera coordinates; validated Sim(3) "
    "registration is required before camera-space use.",
    "Hidden and backside geometry is a generative estimate, not observed ground truth.",
    "The exported mesh must not be used as a physical collision boundary until "
    "registration, watertightness, and held-out-view checks pass.",
)


class PilotInputError(ValueError):
    """Raised when prepared pilot inputs do not satisfy the fixed contract."""


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_file(path: str | Path, description: str) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"missing {description}: {result}")
    return result


def validate_rgba_input(path: str | Path) -> tuple[Path, Image.Image, dict[str, Any]]:
    """Load a 512-square, explicitly masked RGBA image.

    Requiring at least one transparent and one foreground pixel prevents a
    silent fallback to a full RGB scene that contains the hand or table.
    """

    image_path = _resolved_file(path, "SPAR3D RGBA input")
    with Image.open(image_path) as opened:
        opened.load()
        if opened.mode != "RGBA":
            raise PilotInputError(
                f"SPAR3D pilot input must already be RGBA, got {opened.mode!r}: "
                f"{image_path}"
            )
        if opened.size != EXPECTED_INPUT_SIZE:
            raise PilotInputError(
                f"SPAR3D pilot input must be {EXPECTED_INPUT_SIZE[0]}x"
                f"{EXPECTED_INPUT_SIZE[1]}, got {opened.size}: {image_path}"
            )
        image = opened.copy()

    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    foreground = alpha > 0
    foreground_count = int(np.count_nonzero(foreground))
    if foreground_count == 0:
        raise PilotInputError("SPAR3D RGBA alpha is empty")
    if foreground_count == alpha.size:
        raise PilotInputError(
            "SPAR3D RGBA alpha is fully opaque; the prepared object-only mask is required"
        )
    ys, xs = np.nonzero(foreground)
    stats = {
        "path": str(image_path),
        "sha256": sha256_file(image_path),
        "bytes": image_path.stat().st_size,
        "mode": image.mode,
        "size_wh": list(image.size),
        "alpha_min": int(alpha.min()),
        "alpha_max": int(alpha.max()),
        "foreground_pixels": foreground_count,
        "foreground_fraction": foreground_count / int(alpha.size),
        "alpha_bbox_xyxy_exclusive": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ],
        "background_remover_bypassed": True,
    }
    return image_path, image, stats


def validate_input_manifest(
    path: str | Path,
    *,
    input_path: Path,
    input_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    """Validate the handoff written by the pilot-input preparation script."""

    manifest_path = _resolved_file(path, "mesh-pilot input manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotInputError(f"invalid input manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PilotInputError("mesh-pilot manifest root must be an object")

    try:
        selection = payload["selection"]
        output_record = payload["outputs"]["spar3d_rgba_crop"]
        declared_path = Path(output_record["path"]).expanduser().resolve()
        declared_sha256 = str(output_record["sha256"])
    except (KeyError, TypeError) as exc:
        raise PilotInputError(
            "mesh-pilot manifest is missing selection or outputs.spar3d_rgba_crop"
        ) from exc

    object_label = selection.get("object_label")
    if not isinstance(object_label, str) or not object_label.strip():
        raise PilotInputError("mesh-pilot manifest requires a non-empty object_label")
    if declared_path != input_path:
        raise PilotInputError(
            f"manifest RGBA path does not match --input-rgba: {declared_path} != "
            f"{input_path}"
        )
    if declared_sha256 != input_sha256:
        raise PilotInputError(
            "manifest RGBA SHA-256 does not match the prepared input file"
        )
    declared_bytes = output_record.get("bytes")
    if declared_bytes is not None and int(declared_bytes) != input_path.stat().st_size:
        raise PilotInputError("manifest RGBA byte count does not match the input file")
    return manifest_path, payload


def _manifest_bundle_root(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    try:
        value = manifest["bundle"]["output_root"]
    except (KeyError, TypeError) as exc:
        raise PilotInputError("mesh-pilot manifest lacks bundle.output_root") from exc
    if not isinstance(value, str) or not value.strip():
        raise PilotInputError("mesh-pilot manifest bundle.output_root must be a path")
    result = Path(value).expanduser()
    if not result.is_absolute():
        result = manifest_path.parent / result
    return result.resolve()


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either resolved path contains the other."""

    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


def _has_completed_pilot_report(directory: Path) -> bool:
    report_path = directory / "report.json"
    if report_path.is_symlink() or not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(report, dict)
        and report.get("schema_version") == 1
        and report.get("status") == "complete"
        and report.get("method") == "spar3d_direct_prepared_rgba_low_vram"
    )


def validate_output_directory(
    path: str | Path,
    *,
    input_rgba: Path,
    input_manifest: Path,
    input_bundle: Path,
    weights_dir: Path,
    spar3d_repo: Path,
    runtime_asset_dirs: tuple[Path, ...] = (),
    repo_root: Path = REPO_ROOT,
    dedicated_repo_output_root: Path = DEDICATED_OUTPUT_ROOT,
) -> Path:
    """Reject output paths capable of replacing pilot inputs or source trees.

    Outputs outside the repository are allowed when they do not overlap any
    protected path.  Inside the repository, only the dedicated SPAR3D output
    subtree is allowed; this preserves the checked-in pilot layout without
    making an arbitrary repository directory an overwrite target.
    """

    requested_output = Path(path).expanduser()
    if requested_output.is_symlink():
        raise PilotInputError(
            f"unsafe output directory is a symbolic link: {requested_output}"
        )
    output = requested_output.resolve()
    filesystem_root = Path(output.anchor).resolve()
    if (
        output == filesystem_root
        or output.parent == filesystem_root
        or output == Path.home().resolve()
    ):
        raise PilotInputError(f"unsafe broad output directory: {output}")
    for ancestor in output.parents:
        if _has_completed_pilot_report(ancestor):
            raise PilotInputError(
                f"unsafe output directory {output} is nested inside an existing "
                f"SPAR3D result bundle {ancestor}"
            )
    protected = {
        "input RGBA": input_rgba.expanduser().resolve(),
        "input manifest": input_manifest.expanduser().resolve(),
        "input bundle": input_bundle.expanduser().resolve(),
        "weights directory": weights_dir.expanduser().resolve(),
        "SPAR3D repository": spar3d_repo.expanduser().resolve(),
    }
    for index, asset_dir in enumerate(runtime_asset_dirs):
        protected[f"runtime asset directory {index}"] = asset_dir.expanduser().resolve()
    for label, protected_path in protected.items():
        if _paths_overlap(output, protected_path):
            raise PilotInputError(
                f"unsafe output directory {output} overlaps protected {label} "
                f"{protected_path}"
            )

    repository = repo_root.expanduser().resolve()
    if _paths_overlap(output, repository):
        dedicated = dedicated_repo_output_root.expanduser().resolve()
        if output != dedicated and not output.is_relative_to(dedicated):
            raise PilotInputError(
                f"unsafe output directory {output} overlaps repository root "
                f"{repository}; repository-local output is restricted to the "
                f"dedicated subtree {dedicated}"
            )
    return output


def validate_existing_output_bundle(path: str | Path) -> Path:
    """Allow overwrite only for a complete bundle previously written here."""

    output = Path(path).expanduser().resolve()
    if not output.is_dir():
        raise PilotInputError(
            f"refusing to overwrite a non-directory SPAR3D output: {output}"
        )
    report_path = output / "report.json"
    if report_path.is_symlink() or not report_path.is_file():
        raise PilotInputError(
            f"refusing to overwrite an unrecognized directory without a regular "
            f"report.json: {output}"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotInputError(
            f"refusing to overwrite a directory with an invalid report.json: {output}"
        ) from exc
    if not isinstance(report, dict):
        raise PilotInputError(
            f"refusing to overwrite a directory with a non-object report: {output}"
        )
    if (
        report.get("schema_version") != 1
        or report.get("status") != "complete"
        or report.get("method") != "spar3d_direct_prepared_rgba_low_vram"
    ):
        raise PilotInputError(
            f"refusing to overwrite a directory not owned by this SPAR3D pilot: "
            f"{output}"
        )

    expected_entries = {"mesh.glb", "points.ply", "report.json"}
    try:
        actual_entries = {entry.name for entry in output.iterdir()}
    except OSError as exc:
        raise PilotInputError(f"failed to inspect existing output {output}: {exc}") from exc
    if actual_entries != expected_entries:
        raise PilotInputError(
            f"refusing to overwrite a SPAR3D directory with unexpected contents: "
            f"{output}"
        )

    try:
        outputs = report["outputs"]
        declared_report = Path(outputs["report"]).expanduser()
    except (KeyError, TypeError) as exc:
        raise PilotInputError(
            f"existing SPAR3D report has malformed output records: {report_path}"
        ) from exc
    if not declared_report.is_absolute() or declared_report.resolve() != report_path:
        raise PilotInputError(
            f"existing SPAR3D report points at a different report path: {report_path}"
        )

    for record_name, filename in (
        ("mesh_glb", "mesh.glb"),
        ("points_ply", "points.ply"),
    ):
        artifact = output / filename
        if artifact.is_symlink() or not artifact.is_file():
            raise PilotInputError(
                f"existing SPAR3D artifact is not a regular file: {artifact}"
            )
        try:
            record = outputs[record_name]
            declared_path = Path(record["path"]).expanduser()
            declared_bytes = record["bytes"]
            declared_sha256 = record["sha256"]
        except (KeyError, TypeError) as exc:
            raise PilotInputError(
                f"existing SPAR3D report has malformed {record_name}: {report_path}"
            ) from exc
        if not declared_path.is_absolute() or declared_path.resolve() != artifact:
            raise PilotInputError(
                f"existing SPAR3D report points outside its bundle for {record_name}"
            )
        if (
            isinstance(declared_bytes, bool)
            or not isinstance(declared_bytes, int)
            or declared_bytes != artifact.stat().st_size
        ):
            raise PilotInputError(
                f"existing SPAR3D byte count does not match {artifact}"
            )
        if not isinstance(declared_sha256, str) or declared_sha256 != sha256_file(
            artifact
        ):
            raise PilotInputError(
                f"existing SPAR3D SHA-256 does not match {artifact}"
            )
    return output


def validate_weights_dir(
    path: str | Path,
    *,
    require_official_size: bool = True,
    allow_unverified_test_weights: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Require a complete local checkpoint and never permit a remote model id."""

    if not isinstance(allow_unverified_test_weights, bool):
        raise TypeError("allow_unverified_test_weights must be a boolean")
    verify_sha256 = not allow_unverified_test_weights
    weights_dir = Path(path).expanduser().resolve()
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"missing local SPAR3D weights directory: {weights_dir}")
    config = _resolved_file(weights_dir / "config.yaml", "SPAR3D config.yaml")
    model = _resolved_file(weights_dir / "model.safetensors", "SPAR3D model.safetensors")
    for resolved_file, description in (
        (config, "SPAR3D config.yaml"),
        (model, "SPAR3D model.safetensors"),
    ):
        if not resolved_file.is_relative_to(weights_dir):
            raise PilotInputError(
                f"{description} escapes the local weights directory: {resolved_file}"
            )
    model_bytes = model.stat().st_size
    if require_official_size and model_bytes != EXPECTED_MODEL_BYTES:
        raise PilotInputError(
            "SPAR3D model.safetensors has the wrong size; expected official "
            f"checkpoint {EXPECTED_MODEL_BYTES} bytes, got {model_bytes}"
        )

    config_sha256 = sha256_file(config)
    actual_sha256: str | None = None
    if verify_sha256:
        if config_sha256 != EXPECTED_CONFIG_SHA256:
            raise PilotInputError(
                "SPAR3D config.yaml SHA-256 does not match the official config"
            )
        actual_sha256 = sha256_file(model)
        if actual_sha256 != EXPECTED_MODEL_SHA256:
            raise PilotInputError(
                "SPAR3D model.safetensors SHA-256 does not match the official checkpoint"
            )
    return weights_dir, {
        "directory": str(weights_dir),
        "pinned_source": {
            "repository": SPAR3D_HF_REPOSITORY,
            "commit": SPAR3D_HF_COMMIT,
            "config_etag": SPAR3D_CONFIG_ETAG,
            "identity_bound_by_local_content_sha256": verify_sha256,
        },
        "config": {
            "path": str(config),
            "bytes": config.stat().st_size,
            "sha256": config_sha256,
            "expected_sha256": EXPECTED_CONFIG_SHA256,
            "sha256_verified": verify_sha256,
        },
        "model": {
            "path": str(model),
            "bytes": model_bytes,
            "expected_bytes": EXPECTED_MODEL_BYTES,
            "sha256": actual_sha256,
            "expected_sha256": EXPECTED_MODEL_SHA256,
            "sha256_verified": verify_sha256,
        },
        "remote_download_allowed": False,
    }


def _verified_asset_record(
    path: Path,
    *,
    description: str,
    expected_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    """Verify one immutable runtime asset and return its provenance record."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing local {description}: {resolved}")
    actual_bytes = resolved.stat().st_size
    if actual_bytes != expected_bytes:
        raise PilotInputError(
            f"{description} has the wrong size; expected {expected_bytes} bytes, "
            f"got {actual_bytes}: {resolved}"
        )
    actual_sha256 = sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise PilotInputError(
            f"{description} SHA-256 does not match the pinned official asset: "
            f"{resolved}"
        )
    return {
        "path": str(resolved),
        "bytes": actual_bytes,
        "expected_bytes": expected_bytes,
        "sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "sha256_verified": True,
    }


def validate_runtime_assets(
    alpha_clip_dir: str | Path,
    hf_home: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Bind SPAR3D's auxiliary encoders to exact local, offline files."""

    alpha_root = Path(alpha_clip_dir).expanduser().resolve()
    hf_root = Path(hf_home).expanduser().resolve()
    if not alpha_root.is_dir():
        raise FileNotFoundError(f"missing local AlphaCLIP directory: {alpha_root}")
    if not hf_root.is_dir():
        raise FileNotFoundError(f"missing local Hugging Face cache: {hf_root}")

    alpha_path = alpha_root / ALPHA_CLIP_FILENAME
    if not alpha_path.resolve().is_relative_to(alpha_root):
        raise PilotInputError(
            f"AlphaCLIP checkpoint escapes its verified cache root: {alpha_path}"
        )

    dino_cache = (hf_root / "hub" / DINOV2_CACHE_DIRECTORY).resolve()
    if not dino_cache.is_dir() or not dino_cache.is_relative_to(hf_root):
        raise FileNotFoundError(
            f"missing pinned {DINOV2_REPOSITORY} cache under {hf_root}"
        )
    ref_path = dino_cache / "refs" / "main"
    resolved_ref = ref_path.resolve()
    if not resolved_ref.is_relative_to(dino_cache):
        raise PilotInputError(f"DINOv2 cache ref escapes its cache root: {ref_path}")
    if not resolved_ref.is_file():
        raise FileNotFoundError(f"missing DINOv2 cache ref: {ref_path}")
    try:
        ref_commit = resolved_ref.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise PilotInputError(f"failed to read DINOv2 cache ref {ref_path}: {exc}") from exc
    if ref_commit != DINOV2_COMMIT:
        raise PilotInputError(
            f"DINOv2 cache main ref is not pinned to {DINOV2_COMMIT}: {ref_commit!r}"
        )

    snapshot = (dino_cache / "snapshots" / DINOV2_COMMIT).resolve()
    if not snapshot.is_dir() or not snapshot.is_relative_to(dino_cache):
        raise FileNotFoundError(f"missing pinned DINOv2 snapshot: {snapshot}")
    model_path = snapshot / DINOV2_MODEL_FILENAME
    config_path = snapshot / DINOV2_CONFIG_FILENAME
    for asset_path, label in (
        (model_path, "DINOv2 model"),
        (config_path, "DINOv2 config"),
    ):
        resolved_asset = asset_path.resolve()
        if not resolved_asset.is_relative_to(dino_cache):
            raise PilotInputError(
                f"{label} cache link escapes the pinned repository cache: {asset_path}"
            )

    alpha_record = _verified_asset_record(
        alpha_path,
        description="AlphaCLIP ViT-L/14@336px base checkpoint",
        expected_bytes=EXPECTED_ALPHA_CLIP_BYTES,
        expected_sha256=EXPECTED_ALPHA_CLIP_SHA256,
    )
    dino_model_record = _verified_asset_record(
        model_path,
        description="DINOv2-large model.safetensors",
        expected_bytes=EXPECTED_DINOV2_MODEL_BYTES,
        expected_sha256=EXPECTED_DINOV2_MODEL_SHA256,
    )
    dino_config_record = _verified_asset_record(
        config_path,
        description="DINOv2-large config.json",
        expected_bytes=EXPECTED_DINOV2_CONFIG_BYTES,
        expected_sha256=EXPECTED_DINOV2_CONFIG_SHA256,
    )
    return alpha_root, hf_root, {
        "network_access_allowed": False,
        "alpha_clip": alpha_record,
        "dinov2": {
            "repository": DINOV2_REPOSITORY,
            "commit": DINOV2_COMMIT,
            "cache_root": str(dino_cache),
            "model": dino_model_record,
            "config": dino_config_record,
        },
    }


@contextmanager
def offline_runtime_environment(alpha_clip_dir: Path, hf_home: Path):
    """Temporarily force all auxiliary-model loading through verified caches."""

    credential_sentinels = {
        "HF_TOKEN_PATH": hf_home / ".offline-no-hf-token",
        "AWS_SHARED_CREDENTIALS_FILE": hf_home / ".offline-no-aws-credentials",
        "AWS_CONFIG_FILE": hf_home / ".offline-no-aws-config",
        "AWS_CREDENTIAL_FILE": hf_home / ".offline-no-legacy-aws-credentials",
        "BOTO_CONFIG": hf_home / ".offline-no-boto-config",
    }
    for variable, sentinel in credential_sentinels.items():
        if sentinel.exists() or sentinel.is_symlink():
            raise PilotInputError(
                f"offline credential sentinel for {variable} must not exist: "
                f"{sentinel}"
            )
    values = {
        "ALPHA_CLIP_PATH": str(alpha_clip_dir),
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
        "TRANSFORMERS_CACHE": str(hf_home / "hub"),
        "PYTORCH_TRANSFORMERS_CACHE": str(hf_home / "hub"),
        "PYTORCH_PRETRAINED_BERT_CACHE": str(hf_home / "hub"),
        **{key: str(value) for key, value in credential_sentinels.items()},
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "AWS_EC2_METADATA_DISABLED": "true",
        "AWS_SDK_LOAD_CONFIG": "0",
    }
    removed = (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_ACCESS_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SECRET_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "AWS_BEARER_TOKEN_BEDROCK",
    )
    previous = {key: os.environ.get(key) for key in (*values, *removed)}
    try:
        os.environ.update(values)
        for key in removed:
            os.environ.pop(key, None)
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def validate_spar3d_repo(path: str | Path) -> Path:
    repo = Path(path).expanduser().resolve()
    required = (repo / "spar3d" / "system.py", repo / "load" / "tets" / "160_tets.npz")
    escaped = [str(item) for item in required if not item.resolve().is_relative_to(repo)]
    if escaped:
        raise PilotInputError(
            f"official SPAR3D repository files escape {repo}: {escaped}"
        )
    missing = [str(item) for item in required if not item.resolve().is_file()]
    if missing:
        raise FileNotFoundError(
            f"invalid official SPAR3D repository {repo}; missing {missing}"
        )
    return repo


def load_runtime(spar3d_repo: Path) -> tuple[Any, Any]:
    """Import torch and SPAR3D only after all cheap input checks pass."""

    repo_string = str(spar3d_repo)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)

    # The official package imports its optional background-removal network at
    # module import time.  This pilot already has a provenance-checked alpha
    # channel and never calls it, so keep that second network out of memory and
    # fail loudly if upstream unexpectedly tries to instantiate it.
    if "transparent_background" not in sys.modules:
        background_stub = types.ModuleType("transparent_background")

        class DisabledBackgroundRemover:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError(
                    "background removal is disabled: the mesh pilot requires "
                    "the provenance-checked RGBA input"
                )

        background_stub.Remover = DisabledBackgroundRemover
        sys.modules["transparent_background"] = background_stub
    try:
        import torch
        from spar3d.system import SPAR3D
    except Exception as exc:  # pragma: no cover - depends on the isolated GPU env
        raise RuntimeError(
            "failed to import the isolated SPAR3D runtime; verify torch cu128, "
            "uv_unwrapper, and CUDA texture_baker installation"
        ) from exc

    implementation = Path(inspect.getfile(SPAR3D)).resolve()
    if not implementation.is_relative_to(spar3d_repo):
        raise RuntimeError(
            f"imported SPAR3D from unexpected location {implementation}; expected "
            f"under {spar3d_repo}"
        )
    return torch, SPAR3D


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bounds_list(values: Any) -> list[list[float]] | None:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (2, 3) or not np.isfinite(array).all():
        return None
    return array.tolist()


def mesh_report(mesh: Any) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    result = {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "bounds_canonical": _bounds_list(mesh.bounds),
        "extents_canonical": np.asarray(mesh.extents, dtype=np.float64).tolist(),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "is_volume": bool(mesh.is_volume),
        "volume_canonical_cubed": _finite_float(mesh.volume),
        "coordinate_frame": "SPAR3D learned canonical output",
        "metric_scale_verified": False,
        "camera_alignment": "none",
    }
    if result["vertices"] == 0 or result["faces"] == 0:
        raise RuntimeError("SPAR3D returned an empty mesh")
    return result


def point_cloud_report(point_cloud: Any) -> dict[str, Any]:
    vertices = np.asarray(point_cloud.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
        raise RuntimeError("SPAR3D returned an empty or malformed point cloud")
    return {
        "points": int(len(vertices)),
        "bounds_canonical": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "coordinate_frame": "SPAR3D learned canonical conditioning cloud",
        "metric_scale_verified": False,
    }


def _cuda_memory_info(torch_module: Any, device: str) -> dict[str, int | None]:
    try:
        free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
    except (AttributeError, RuntimeError, TypeError):
        return {"free_bytes": None, "total_bytes": None}
    return {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes)}


def _publish_directory(staging: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"output directory already exists (pass --overwrite to replace it): {output}"
            )
        # Recheck immediately before the destructive rename.  The earlier
        # run_job check avoids expensive inference for a bad target; this one
        # narrows the window in which the directory could have been replaced.
        validate_existing_output_bundle(output)
        backup = output.with_name(f".{output.name}.backup")
        if backup.exists():
            raise FileExistsError(f"stale SPAR3D backup directory exists: {backup}")
        output.replace(backup)
        try:
            staging.replace(output)
        except BaseException:
            backup.replace(output)
            raise
        shutil.rmtree(backup)
    else:
        staging.replace(output)


def run_job(
    *,
    input_rgba: str | Path,
    input_manifest: str | Path,
    weights_dir: str | Path,
    spar3d_repo: str | Path,
    output_dir: str | Path,
    alpha_clip_dir: str | Path = DEFAULT_ALPHA_CLIP_DIR,
    hf_home: str | Path = DEFAULT_HF_HOME,
    texture_resolution: int = 512,
    device: str = "cuda",
    seed: int = 42,
    overwrite: bool = False,
    allow_unverified_test_weights: bool = False,
    runtime_loader: Callable[[Path], tuple[Any, Any]] = load_runtime,
    runtime_assets_validator: Callable[
        [str | Path, str | Path], tuple[Path, Path, dict[str, Any]]
    ] = validate_runtime_assets,
) -> dict[str, Any]:
    """Run one prepared object image and atomically publish its mesh bundle."""

    if not isinstance(allow_unverified_test_weights, bool):
        raise TypeError("allow_unverified_test_weights must be a boolean")
    test_runtime_injected = runtime_loader is not load_runtime
    if allow_unverified_test_weights and not test_runtime_injected:
        raise PilotInputError(
            "unverified test weights require an explicitly injected test runtime"
        )
    if runtime_assets_validator is not validate_runtime_assets and not test_runtime_injected:
        raise PilotInputError(
            "a custom runtime-asset validator requires an explicitly injected "
            "test runtime"
        )
    if texture_resolution < 128 or texture_resolution > 2048:
        raise PilotInputError("texture_resolution must be in [128, 2048]")
    if not device.startswith("cuda"):
        raise PilotInputError("the focused pilot requires an NVIDIA CUDA device")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise PilotInputError("seed must be a non-negative integer")

    image_path, image, input_stats = validate_rgba_input(input_rgba)
    manifest_path, manifest = validate_input_manifest(
        input_manifest,
        input_path=image_path,
        input_sha256=input_stats["sha256"],
    )
    weights_candidate = Path(weights_dir).expanduser().resolve()
    upstream_candidate = Path(spar3d_repo).expanduser().resolve()
    alpha_candidate = Path(alpha_clip_dir).expanduser().resolve()
    hf_candidate = Path(hf_home).expanduser().resolve()
    output = validate_output_directory(
        output_dir,
        input_rgba=image_path,
        input_manifest=manifest_path,
        input_bundle=_manifest_bundle_root(manifest_path, manifest),
        weights_dir=weights_candidate,
        spar3d_repo=upstream_candidate,
        runtime_asset_dirs=(alpha_candidate, hf_candidate),
    )
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"output directory already exists (pass --overwrite to replace it): "
                f"{output}"
            )
        validate_existing_output_bundle(output)
    local_weights, weights_stats = validate_weights_dir(
        weights_candidate,
        allow_unverified_test_weights=allow_unverified_test_weights,
    )
    upstream_repo = validate_spar3d_repo(upstream_candidate)
    local_alpha_clip, local_hf_home, runtime_asset_stats = runtime_assets_validator(
        alpha_candidate,
        hf_candidate,
    )
    local_alpha_clip = Path(local_alpha_clip).expanduser().resolve()
    local_hf_home = Path(local_hf_home).expanduser().resolve()
    if local_alpha_clip != alpha_candidate or local_hf_home != hf_candidate:
        raise PilotInputError(
            "runtime-asset validator attempted to rebind a verified cache root"
        )
    if (
        not isinstance(runtime_asset_stats, dict)
        or runtime_asset_stats.get("network_access_allowed") is not False
    ):
        raise PilotInputError(
            "runtime-asset validator did not attest an offline asset bundle"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent)
    )

    try:
        with offline_runtime_environment(local_alpha_clip, local_hf_home):
            torch_module, spar3d_class = runtime_loader(upstream_repo)
            if not torch_module.cuda.is_available():
                raise RuntimeError("CUDA is unavailable in the isolated SPAR3D environment")
            if (
                hasattr(torch_module.cuda, "is_bf16_supported")
                and not torch_module.cuda.is_bf16_supported()
            ):
                raise RuntimeError("SPAR3D pilot requires CUDA bfloat16 support")

            # Point diffusion starts from random noise.  Fix all available RNGs so
            # rerunning the focused pilot does not silently compare a new sample.
            np.random.seed(seed)
            if hasattr(torch_module, "manual_seed"):
                torch_module.manual_seed(seed)
            if hasattr(torch_module.cuda, "manual_seed_all"):
                torch_module.cuda.manual_seed_all(seed)

            torch_module.cuda.reset_peak_memory_stats(device)
            before_memory = _cuda_memory_info(torch_module, device)
            overall_start = time.perf_counter()
            load_start = overall_start
            model = spar3d_class.from_pretrained(
                str(local_weights),
                config_name="config.yaml",
                weight_name="model.safetensors",
                low_vram_mode=True,
            )
            model.to(device)
            model.eval()
            torch_module.cuda.synchronize(device)
            load_seconds = time.perf_counter() - load_start

            inference_start = time.perf_counter()
            no_grad = torch_module.no_grad()
            autocast = torch_module.autocast(
                device_type="cuda", dtype=torch_module.bfloat16
            )
            with no_grad:
                with autocast:
                    mesh, global_values = model.run_image(
                        image,
                        bake_resolution=texture_resolution,
                        remesh="none",
                        vertex_count=-1,
                        return_points=True,
                    )
            torch_module.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - inference_start
            overall_seconds = time.perf_counter() - overall_start

        point_clouds = global_values.get("point_clouds")
        if not isinstance(point_clouds, (list, tuple)) or len(point_clouds) != 1:
            raise RuntimeError(
                "SPAR3D return_points=True did not return exactly one point cloud"
            )
        point_cloud = point_clouds[0]

        mesh_path = staging / "mesh.glb"
        points_path = staging / "points.ply"
        mesh.export(mesh_path, include_normals=True)
        point_cloud.export(points_path)
        if not mesh_path.is_file() or mesh_path.stat().st_size == 0:
            raise RuntimeError("SPAR3D mesh.glb export is empty")
        if not points_path.is_file() or points_path.stat().st_size == 0:
            raise RuntimeError("SPAR3D points.ply export is empty")

        peak_allocated = int(torch_module.cuda.max_memory_allocated(device))
        try:
            peak_reserved = int(torch_module.cuda.max_memory_reserved(device))
        except (AttributeError, RuntimeError, TypeError):
            peak_reserved = None
        after_memory = _cuda_memory_info(torch_module, device)

        try:
            device_name = str(torch_module.cuda.get_device_name(device))
        except (AttributeError, RuntimeError, TypeError):
            device_name = device
        try:
            capability = list(torch_module.cuda.get_device_capability(device))
        except (AttributeError, RuntimeError, TypeError):
            capability = None

        selection = manifest.get("selection", {})
        report = {
            "schema_version": 1,
            "status": "complete",
            "method": "spar3d_direct_prepared_rgba_low_vram",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection": selection,
            "input": input_stats,
            "input_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "weights": weights_stats,
            "runtime_assets": runtime_asset_stats,
            "runtime": {
                "upstream_repo": str(upstream_repo),
                "device": device,
                "device_name": device_name,
                "cuda_capability": capability,
                "torch_version": str(getattr(torch_module, "__version__", "unknown")),
                "torch_cuda_version": str(
                    getattr(getattr(torch_module, "version", None), "cuda", "unknown")
                ),
                "autocast_dtype": "bfloat16",
                "low_vram_mode": True,
                "background_remover_loaded": False,
                "network_access_allowed": False,
                "credential_environment_disabled": True,
                "random_seed": seed,
                "texture_resolution": texture_resolution,
                "remesh": "none",
                "vertex_count_argument": -1,
                "model_load_seconds": load_seconds,
                "inference_seconds": inference_seconds,
                "total_model_seconds": overall_seconds,
                "cuda_memory_before": before_memory,
                "cuda_memory_after": after_memory,
                "peak_vram_allocated_bytes": peak_allocated,
                "peak_vram_reserved_bytes": peak_reserved,
            },
            "mesh": mesh_report(mesh),
            "point_cloud": point_cloud_report(point_cloud),
            "outputs": {
                "mesh_glb": {
                    "path": str(output / "mesh.glb"),
                    "bytes": mesh_path.stat().st_size,
                    "sha256": sha256_file(mesh_path),
                },
                "points_ply": {
                    "path": str(output / "points.ply"),
                    "bytes": points_path.stat().st_size,
                    "sha256": sha256_file(points_path),
                },
                "report": str(output / "report.json"),
            },
            "representation": "learned_single_image_canonical_mesh_and_point_cloud",
            "metric_scale_verified": False,
            "camera_alignment": "none",
            "physical_geometry_guarantee": False,
            "collision_ready": False,
            "warnings": list(CANONICAL_WARNINGS),
        }
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        _publish_directory(staging, output, overwrite=overwrite)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input-rgba", type=Path, default=DEFAULT_INPUT_RGBA)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--spar3d-repo", type=Path, default=DEFAULT_SPAR3D_REPO)
    parser.add_argument(
        "--alpha-clip-dir", type=Path, default=DEFAULT_ALPHA_CLIP_DIR
    )
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--texture-resolution",
        type=int,
        default=512,
        help="UV texture resolution; 512 reduces the CUDA baking peak on a 16GB GPU.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_job(
        input_rgba=args.input_rgba,
        input_manifest=args.input_manifest,
        weights_dir=args.weights_dir,
        spar3d_repo=args.spar3d_repo,
        alpha_clip_dir=args.alpha_clip_dir,
        hf_home=args.hf_home,
        output_dir=args.output_dir,
        texture_resolution=args.texture_resolution,
        device=args.device,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(report["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
