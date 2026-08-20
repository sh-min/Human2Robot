#!/usr/bin/env python3
"""Run the two-view VGGT-Omega geometry pilot.

Despite the historical ``mesh_pilot`` directory name, VGGT-Omega predicts
cameras and dense depth.  This script exports colored point clouds; it does
not label them as a triangle mesh, watertight surface, or metric geometry.

The default object-aggregation mode is deliberately strict. Both the MH and SH
object masks must be supplied by the input manifest.  The current input
bundle contains an MH mask only, so it must either be extended with an SH
mask or run with ``--full-scene-only``.  The latter still uses both images for
camera/depth inference, but performs no mask-filtered object-point aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "8-5"
    / "mesh_sota_pilot"
    / "episode_1"
    / "choco"
    / "inputs"
    / "manifest.json"
)
DEFAULT_VGGT_REPOSITORY = REPO_ROOT / "third_party" / "VGGT-Omega"
EXACT_CHECKPOINT_FILENAME = "vggt_omega_1b_512.pt"
EXPECTED_CHECKPOINT_BYTES = 4_576_706_117
EXPECTED_CHECKPOINT_SHA256 = (
    "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "weights" / "vggt_omega" / EXACT_CHECKPOINT_FILENAME
)

MODEL_REPOSITORY = "facebook/VGGT-Omega"
MODEL_VARIANT = "VGGT-Omega-1B-512"
OFFICIAL_HF_CHECKPOINT_COMMIT = "05654241adc2f218dfb089c373a011f8a7040576"
EXPECTED_LOCAL_CODE_COMMIT = "39a0cb8af88554f15ddcb5354cd52bde588fa014"
MODEL_LICENSE = "FAIR Noncommercial Research License"
REQUIRED_REPOSITORY_FILES = (
    "pyproject.toml",
    "LICENSE",
    "vggt_omega/__init__.py",
    "vggt_omega/models/__init__.py",
    "vggt_omega/models/vggt_omega.py",
    "vggt_omega/utils/__init__.py",
    "vggt_omega/utils/load_fn.py",
    "vggt_omega/utils/pose_enc.py",
)
PREPROCESS_MODE = "balanced"
IMAGE_RESOLUTION = 512
PATCH_SIZE = 16
CAMERA_ORDER = ("MH", "SH")
OBJECT_AGGREGATION_METHOD = "dual_view_mask_filtered_point_aggregation"
MIN_POSITIVE_CONFIDENCE = 1.0e-5
MIN_POSITIVE_DEPTH = 1.0e-6
DEFAULT_DEPTH_EDGE_RTOL = 0.03
MAX_ADAPTIVE_DEPTH_EDGE_RTOL = 0.12
ADAPTIVE_DEPTH_EDGE_RTOLS = (0.05, 0.08, MAX_ADAPTIVE_DEPTH_EDGE_RTOL)
OFFICIAL_REFERENCE_CONFIDENCE_PERCENTILE = 20.0
OFFICIAL_REFERENCE_MAX_POINTS = 300_000


class VGGTOmegaPilotError(RuntimeError):
    """Base error for an invalid or incomplete pilot run."""


class PilotManifestError(VGGTOmegaPilotError):
    """Raised when the pilot input manifest violates its contract."""


class MissingDualObjectMaskError(PilotManifestError):
    """Raised when object aggregation is requested without both view masks."""


class CheckpointValidationError(VGGTOmegaPilotError):
    """Raised when the exact VGGT-Omega 512 checkpoint is unavailable."""


@dataclass(frozen=True)
class FileBinding:
    """One path/size/digest record selected from the input manifest."""

    field: str
    path: Path
    bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_field": self.field,
            "path": str(self.path),
            "bytes": self.bytes,
            "sha256": self.sha256,
            "verified": True,
        }


@dataclass(frozen=True)
class ViewInput:
    """One ordered camera input from the pilot manifest."""

    camera: str
    semantic_role: str
    frame_index: int
    image: FileBinding
    source_image: FileBinding
    object_mask: FileBinding | None

    @property
    def image_path(self) -> Path:
        return self.image.path

    @property
    def object_mask_path(self) -> Path | None:
        return self.object_mask.path if self.object_mask is not None else None


@dataclass(frozen=True)
class PilotInputs:
    """Normalized subset of the mesh-pilot input manifest."""

    manifest_path: Path
    manifest_bytes: int
    manifest_sha256: str
    schema_version: int
    episode: str
    object_label: str
    image_width: int
    image_height: int
    views: tuple[ViewInput, ViewInput]
    bundle_root: Path
    calibration: Mapping[str, Any] | None

    @property
    def mh(self) -> ViewInput:
        return self.views[0]

    @property
    def sh(self) -> ViewInput:
        return self.views[1]


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotManifestError(
            f"manifest field {field!r} must be an object, "
            f"got {type(value).__name__}"
        )
    return value


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PilotManifestError(f"manifest field {field!r} must be an integer")
    parsed = value
    if parsed < minimum:
        raise PilotManifestError(
            f"manifest field {field!r} must be >= {minimum}, got {parsed}"
        )
    return parsed


def _require_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PilotManifestError(
            f"manifest field {field!r} must be a non-empty string"
        )
    return value.strip()


def _record_path(record: Any, *, field: str, required: bool) -> str | None:
    if record is None:
        if required:
            raise PilotManifestError(f"manifest is missing required field {field!r}")
        return None
    if isinstance(record, str):
        value = record
    elif isinstance(record, Mapping):
        value = record.get("path")
    else:
        raise PilotManifestError(
            f"manifest field {field!r} must be a path string or path record"
        )
    if not isinstance(value, str) or not value.strip():
        if required:
            raise PilotManifestError(
                f"manifest field {field!r} must contain a non-empty path"
            )
        return None
    return value.strip()


def _resolve_manifest_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (manifest_path.parent / path).resolve()


def _first_record(
    containers: Sequence[tuple[str, Mapping[str, Any]]],
    keys: Sequence[str],
) -> tuple[Any, str] | tuple[None, str]:
    for container_name, container in containers:
        for key in keys:
            if key in container and container[key] is not None:
                return container[key], f"{container_name}.{key}"
    return None, f"{containers[0][0]}.{keys[0]}"


def _file_binding(
    record: Any, *, field: str, manifest_path: Path
) -> FileBinding:
    mapping = _require_mapping(record, field=field)
    raw_path = _record_path(mapping, field=field, required=True)
    assert raw_path is not None
    declared_bytes = _require_int(
        mapping.get("bytes"), field=f"{field}.bytes", minimum=1
    )
    declared_sha256 = _require_text(
        mapping.get("sha256"), field=f"{field}.sha256"
    )
    if len(declared_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in declared_sha256
    ):
        raise PilotManifestError(
            f"manifest field {field!r}.sha256 must be a lowercase SHA-256 digest"
        )
    return FileBinding(
        field=field,
        path=_resolve_manifest_path(raw_path, manifest_path),
        bytes=declared_bytes,
        sha256=declared_sha256,
    )


def _validate_view_record(
    record: Any,
    *,
    field: str,
    expected_view: str,
    expected_camera: str,
    expected_frame: int,
    require_metadata: bool,
) -> None:
    mapping = _require_mapping(record, field=field)
    expected = {
        "view": expected_view,
        "pipeline_camera": expected_camera,
        "frame_index": expected_frame,
    }
    for key, expected_value in expected.items():
        if key not in mapping:
            if require_metadata:
                raise PilotManifestError(
                    f"manifest field {field!r} is missing required {key!r} binding"
                )
            continue
        actual = mapping[key]
        if key == "frame_index":
            actual = _require_int(actual, field=f"{field}.{key}")
        if actual != expected_value:
            raise PilotManifestError(
                f"manifest field {field!r}.{key}={actual!r} does not match "
                f"{expected_value!r}"
            )


def _validate_selected_frame_record(
    record: Any, *, field: str, expected_frame: int
) -> None:
    mapping = _require_mapping(record, field=field)
    key = "selected_frame_index" if "selected_frame_index" in mapping else "frame_index"
    if key not in mapping:
        raise PilotManifestError(
            f"manifest field {field!r} lacks a selected-frame binding"
        )
    frame = _require_int(mapping[key], field=f"{field}.{key}")
    if frame != expected_frame:
        raise PilotManifestError(
            f"manifest field {field!r}.{key}={frame} does not match "
            f"selection.mh_frame_index={expected_frame}"
        )


def load_pilot_manifest(manifest_path: Path) -> PilotInputs:
    """Load and normalize the v1 mesh-pilot manifest without mutating it."""

    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise PilotManifestError(f"missing input manifest: {manifest_path}")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotManifestError(
            f"failed to read input manifest {manifest_path}: {exc}"
        ) from exc
    document = _require_mapping(document, field="<root>")

    schema_version = _require_int(
        document.get("schema_version"), field="schema_version", minimum=1
    )
    if schema_version != 1:
        raise PilotManifestError(
            f"unsupported mesh-pilot schema_version {schema_version}; expected 1"
        )
    kind = _require_text(document.get("kind"), field="kind")
    if kind != "mesh_sota_pilot_input_bundle":
        raise PilotManifestError(
            f"manifest kind must be 'mesh_sota_pilot_input_bundle', got {kind!r}"
        )

    selection = _require_mapping(document.get("selection"), field="selection")
    sources = _require_mapping(document.get("sources"), field="sources")
    outputs = _require_mapping(document.get("outputs"), field="outputs")
    image_geometry = _require_mapping(
        document.get("image_geometry"), field="image_geometry"
    )
    bundle = _require_mapping(document.get("bundle"), field="bundle")

    camera_namespace = _require_mapping(
        document.get("camera_namespace"), field="camera_namespace"
    )
    if camera_namespace.get("primary_view") != "MH" or camera_namespace.get(
        "auxiliary_view"
    ) != "SH":
        raise PilotManifestError("camera_namespace must bind MH primary and SH auxiliary")
    expected_pipeline_mapping = {"camera_1": "SH", "camera_2": "MH"}
    for key in ("stereo_code_mapping", "pipeline_camera_mapping"):
        if camera_namespace.get(key) != expected_pipeline_mapping:
            raise PilotManifestError(
                f"camera_namespace.{key} must be camera_1=SH, camera_2=MH"
            )

    object_label = _require_text(
        selection.get("object_label"), field="selection.object_label"
    )
    # The original pilot was introduced for Choco, but the inference and
    # dual-mask aggregation are object-agnostic.  Keep the manifest binding
    # strict while allowing every explicitly labelled object bundle.
    if not object_label.strip():
        raise PilotManifestError("selection.object_label must not be empty")
    mh_frame = _require_int(
        selection.get("mh_frame_index"), field="selection.mh_frame_index"
    )
    sh_frame = _require_int(
        selection.get("sh_frame_index"), field="selection.sh_frame_index"
    )

    mh_record = outputs.get("mh_image")
    sh_record = outputs.get("sh_image")
    mh_field = "outputs.mh_image"
    sh_field = "outputs.sh_image"
    mh_image = _file_binding(mh_record, field=mh_field, manifest_path=manifest_path)
    sh_image = _file_binding(sh_record, field=sh_field, manifest_path=manifest_path)
    mh_source_record = sources.get("mh_image")
    sh_source_record = sources.get("sh_image")
    mh_source = _file_binding(
        mh_source_record, field="sources.mh_image", manifest_path=manifest_path
    )
    sh_source = _file_binding(
        sh_source_record, field="sources.sh_image", manifest_path=manifest_path
    )
    _validate_view_record(
        mh_source_record,
        field="sources.mh_image",
        expected_view="MH",
        expected_camera="camera_2",
        expected_frame=mh_frame,
        require_metadata=True,
    )
    _validate_view_record(
        sh_source_record,
        field="sources.sh_image",
        expected_view="SH",
        expected_camera="camera_1",
        expected_frame=sh_frame,
        require_metadata=True,
    )
    _validate_view_record(
        mh_record,
        field=mh_field,
        expected_view="MH",
        expected_camera="camera_2",
        expected_frame=mh_frame,
        require_metadata=False,
    )
    _validate_view_record(
        sh_record,
        field=sh_field,
        expected_view="SH",
        expected_camera="camera_1",
        expected_frame=sh_frame,
        require_metadata=False,
    )
    for camera, output_record, source_record in (
        ("MH", mh_image, mh_source),
        ("SH", sh_image, sh_source),
    ):
        if (
            output_record.bytes != source_record.bytes
            or output_record.sha256 != source_record.sha256
        ):
            raise PilotManifestError(
                f"{camera} output image bytes/SHA do not match its bound source image"
            )

    # Modal pixels correspond to actually observed object surface samples.
    # Amodal/clean masks may contain inferred support and are intentionally not
    # substituted here.
    mh_mask_record, mh_mask_field = _first_record(
        (("outputs", outputs),),
        ("mh_modal_mask", "mh_mask_modal", "modal_mask"),
    )
    mh_mask = (
        _file_binding(mh_mask_record, field=mh_mask_field, manifest_path=manifest_path)
        if mh_mask_record is not None
        else None
    )
    if mh_mask_record is not None:
        _validate_view_record(
            mh_mask_record,
            field=mh_mask_field,
            expected_view="MH",
            expected_camera="camera_2",
            expected_frame=mh_frame,
            require_metadata=False,
        )
        mh_mask_source, mh_mask_source_field = _first_record(
            (("sources", sources),),
            ("mh_modal_mask", "mh_mask_modal", "modal_mask"),
        )
        if "frame_index" not in _require_mapping(mh_mask_record, field=mh_mask_field):
            if mh_mask_source is None:
                raise PilotManifestError(
                    f"{mh_mask_field} lacks frame_index and has no source frame binding"
                )
            _validate_selected_frame_record(
                mh_mask_source,
                field=mh_mask_source_field,
                expected_frame=mh_frame,
            )

    sh_mask_record, sh_mask_field = _first_record(
        (("outputs", outputs),),
        ("sh_modal_mask", "sh_mask_modal", "sh_object_mask"),
    )
    sh_mask = (
        _file_binding(sh_mask_record, field=sh_mask_field, manifest_path=manifest_path)
        if sh_mask_record is not None
        else None
    )
    if sh_mask_record is not None:
        _validate_view_record(
            sh_mask_record,
            field=sh_mask_field,
            expected_view="SH",
            expected_camera="camera_1",
            expected_frame=sh_frame,
            require_metadata=True,
        )

    bundle_root_text = _record_path(
        bundle.get("output_root"), field="bundle.output_root", required=True
    )
    assert bundle_root_text is not None
    bundle_root = _resolve_manifest_path(bundle_root_text, manifest_path)
    if not bundle_root.is_dir():
        raise PilotManifestError(f"manifest bundle root is not a directory: {bundle_root}")
    if manifest_path.parent != bundle_root and not manifest_path.is_relative_to(
        bundle_root
    ):
        raise PilotManifestError(
            f"input manifest {manifest_path} is outside bundle root {bundle_root}"
        )
    for binding in (mh_image, sh_image, mh_mask, sh_mask):
        if binding is not None and not binding.path.is_relative_to(bundle_root):
            raise PilotManifestError(
                f"manifest output {binding.field} escapes bundle root: {binding.path}"
            )

    episode_value = _require_text(
        document.get("episode", selection.get("episode", "1")), field="selection.episode"
    )

    mh = ViewInput(
        camera="MH",
        semantic_role=_require_text(
            selection.get("mh_role", "primary/final"),
            field="selection.mh_role",
        ),
        frame_index=mh_frame,
        image=mh_image,
        source_image=mh_source,
        object_mask=mh_mask,
    )
    sh = ViewInput(
        camera="SH",
        semantic_role=_require_text(
            selection.get("sh_role", "auxiliary/evidence"),
            field="selection.sh_role",
        ),
        frame_index=sh_frame,
        image=sh_image,
        source_image=sh_source,
        object_mask=sh_mask,
    )

    calibration = document.get("calibration")
    if calibration is not None:
        calibration = _require_mapping(calibration, field="calibration")

    return PilotInputs(
        manifest_path=manifest_path,
        manifest_bytes=manifest_path.stat().st_size,
        manifest_sha256=_sha256(manifest_path),
        schema_version=schema_version,
        episode=episode_value,
        object_label=object_label,
        image_width=_require_int(
            image_geometry.get("width"), field="image_geometry.width", minimum=1
        ),
        image_height=_require_int(
            image_geometry.get("height"), field="image_geometry.height", minimum=1
        ),
        views=(mh, sh),
        bundle_root=bundle_root,
        calibration=calibration,
    )


def require_dual_object_masks(inputs: PilotInputs) -> tuple[Path, Path]:
    """Return ordered MH/SH masks or reject an unsupported aggregation claim."""

    if inputs.mh.object_mask_path is None:
        raise MissingDualObjectMaskError(
            "dual-view mask-filtered point aggregation requires an MH modal object mask; "
            "add outputs.mh_modal_mask (or outputs.modal_mask) to the manifest"
        )
    if inputs.sh.object_mask_path is None:
        raise MissingDualObjectMaskError(
            "dual-view mask-filtered point aggregation requires an SH modal object mask, "
            "but the input manifest supplies only the MH mask. Add "
            "outputs.sh_modal_mask.path (or sources.sh_modal_mask.path). "
            "Use --full-scene-only only if a two-camera scene point cloud "
            "without object-level aggregation is acceptable."
        )
    return inputs.mh.object_mask_path, inputs.sh.object_mask_path


def _read_repository_head(repository: Path) -> tuple[str, Path]:
    """Read a normal/detached Git HEAD without running Git or taking locks."""

    git_marker = repository / ".git"
    if git_marker.is_symlink():
        raise VGGTOmegaPilotError(
            f"VGGT-Omega .git marker must not be a symbolic link: {git_marker}"
        )
    if git_marker.is_dir():
        git_directory = git_marker.resolve()
    elif git_marker.is_file():
        try:
            marker = git_marker.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise VGGTOmegaPilotError(
                f"failed to read VGGT-Omega .git marker {git_marker}: {exc}"
            ) from exc
        prefix = "gitdir:"
        if not marker.startswith(prefix):
            raise VGGTOmegaPilotError(
                f"invalid VGGT-Omega .git marker: {git_marker}"
            )
        git_directory = Path(marker[len(prefix) :].strip()).expanduser()
        if not git_directory.is_absolute():
            git_directory = git_marker.parent / git_directory
        git_directory = git_directory.resolve()
        if not git_directory.is_dir():
            raise VGGTOmegaPilotError(
                f"VGGT-Omega git directory is missing: {git_directory}"
            )
    else:
        raise VGGTOmegaPilotError(
            f"VGGT-Omega repository has no readable .git metadata: {repository}"
        )

    head_path = git_directory / "HEAD"
    if head_path.is_symlink() or not head_path.is_file():
        raise VGGTOmegaPilotError(
            f"VGGT-Omega HEAD must be a regular file: {head_path}"
        )
    try:
        head_value = head_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise VGGTOmegaPilotError(
            f"failed to read VGGT-Omega HEAD {head_path}: {exc}"
        ) from exc
    if head_value.startswith("ref:"):
        reference = head_value.removeprefix("ref:").strip()
        reference_path = (git_directory / reference).resolve()
        if not reference_path.is_relative_to(git_directory):
            raise VGGTOmegaPilotError(
                f"VGGT-Omega HEAD reference escapes git directory: {reference!r}"
            )
        if reference_path.is_file():
            try:
                commit = reference_path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError) as exc:
                raise VGGTOmegaPilotError(
                    f"failed to read VGGT-Omega HEAD reference {reference_path}: {exc}"
                ) from exc
        else:
            commit = ""
            packed_refs = git_directory / "packed-refs"
            if packed_refs.is_file():
                if packed_refs.is_symlink():
                    raise VGGTOmegaPilotError(
                        f"VGGT-Omega packed-refs must be a regular file: {packed_refs}"
                    )
                try:
                    packed_lines = packed_refs.read_text(encoding="utf-8").splitlines()
                except (OSError, UnicodeDecodeError) as exc:
                    raise VGGTOmegaPilotError(
                        f"failed to read VGGT-Omega packed-refs {packed_refs}: {exc}"
                    ) from exc
                for line in packed_lines:
                    if not line or line.startswith(("#", "^")):
                        continue
                    fields = line.split(" ", 1)
                    if len(fields) == 2 and fields[1] == reference:
                        commit = fields[0]
                        break
            if not commit:
                raise VGGTOmegaPilotError(
                    f"VGGT-Omega HEAD reference is unresolved: {reference!r}"
                )
    else:
        commit = head_value
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise VGGTOmegaPilotError(
            f"VGGT-Omega HEAD is not a full lowercase commit: {commit!r}"
        )
    return commit, git_directory


def validate_vggt_repository(repository_path: Path) -> dict[str, Any]:
    """Validate the pinned local code checkout without importing model code."""

    candidate = repository_path.expanduser()
    if candidate.is_symlink():
        raise VGGTOmegaPilotError(
            f"VGGT-Omega repository must not be a symbolic link: {candidate}"
        )
    repository = candidate.resolve()
    if not repository.is_dir():
        raise VGGTOmegaPilotError(
            f"missing official VGGT-Omega repository: {repository}"
        )
    head_commit, git_directory = _read_repository_head(repository)
    if head_commit != EXPECTED_LOCAL_CODE_COMMIT:
        raise VGGTOmegaPilotError(
            "VGGT-Omega local code commit mismatch: expected "
            f"{EXPECTED_LOCAL_CODE_COMMIT}, got {head_commit}"
        )

    required_files: dict[str, dict[str, Any]] = {}
    for relative_name in REQUIRED_REPOSITORY_FILES:
        path = repository / relative_name
        if path.is_symlink() or not path.is_file():
            raise VGGTOmegaPilotError(
                f"missing regular VGGT-Omega repository file: {path}"
            )
        try:
            file_bytes = path.stat().st_size
            file_sha256 = _sha256(path)
        except OSError as exc:
            raise VGGTOmegaPilotError(
                f"failed to hash VGGT-Omega repository file {path}: {exc}"
            ) from exc
        required_files[relative_name] = {
            "path": str(path.resolve()),
            "bytes": file_bytes,
            "sha256": file_sha256,
        }

    license_path = repository / "LICENSE"
    try:
        license_lines = license_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise VGGTOmegaPilotError(
            f"failed to read VGGT-Omega license {license_path}: {exc}"
        ) from exc
    if not license_lines or license_lines[0].strip() != MODEL_LICENSE:
        raise VGGTOmegaPilotError(
            f"VGGT-Omega license must be {MODEL_LICENSE!r}: {license_path}"
        )
    return {
        "path": str(repository),
        "git_directory": str(git_directory),
        "local_code_commit": head_commit,
        "expected_local_code_commit": EXPECTED_LOCAL_CODE_COMMIT,
        "local_code_commit_verified": True,
        "required_files": required_files,
        "structure_verified": True,
        "model_imported": False,
        "license": {
            "name": MODEL_LICENSE,
            **required_files["LICENSE"],
            "verified": True,
        },
    }


def inspect_checkpoint_for_preflight(checkpoint_path: Path) -> dict[str, Any]:
    """Classify a local checkpoint without downloading or loading it."""

    requested = checkpoint_path.expanduser().absolute()
    if requested.name != EXACT_CHECKPOINT_FILENAME:
        raise CheckpointValidationError(
            f"expected exact checkpoint filename {EXACT_CHECKPOINT_FILENAME!r}, "
            f"got {requested.name!r}. The 256 text-alignment checkpoint is not "
            "compatible with this pilot."
        )
    resolved = requested.resolve()
    record: dict[str, Any] = {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        "requested_path_is_symlink": requested.is_symlink(),
        "filename": EXACT_CHECKPOINT_FILENAME,
        "model_repository": MODEL_REPOSITORY,
        "official_huggingface_checkpoint_commit": OFFICIAL_HF_CHECKPOINT_COMMIT,
        "expected_bytes": EXPECTED_CHECKPOINT_BYTES,
        "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
        "download_performed": False,
    }
    if not requested.exists():
        return {
            **record,
            "state": "missing",
            "exists": False,
            "official_content_verified": False,
            "waiting_reason": "checkpoint_missing",
        }
    if not resolved.is_file():
        raise CheckpointValidationError(
            f"checkpoint path exists but is not a regular file: {requested}"
        )
    actual_bytes = resolved.stat().st_size
    if actual_bytes != EXPECTED_CHECKPOINT_BYTES:
        return {
            **record,
            "state": "partial",
            "exists": True,
            "actual_bytes": actual_bytes,
            "official_content_verified": False,
            "waiting_reason": "checkpoint_size_mismatch",
        }
    try:
        actual_sha256 = _sha256(resolved)
    except OSError as exc:
        raise CheckpointValidationError(
            f"failed to hash complete VGGT-Omega checkpoint {resolved}: {exc}"
        ) from exc
    if actual_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise CheckpointValidationError(
            "VGGT-Omega checkpoint has the complete expected byte count but its "
            "SHA-256 does not match the pinned official artifact"
        )
    return {
        **record,
        "state": "verified",
        "exists": True,
        "actual_bytes": actual_bytes,
        "actual_sha256": actual_sha256,
        "official_content_verified": True,
    }


def validate_checkpoint_path(
    checkpoint_path: Path,
    *,
    require_official_content: bool = True,
) -> Path:
    """Require the exact non-text VGGT-Omega 1B/512 checkpoint filename."""

    # Check the user-visible link name before ``resolve``: Hugging Face cache
    # snapshots expose the checkpoint as a correctly named symlink whose blob
    # target is hash-named.
    checkpoint_path = checkpoint_path.expanduser()
    if checkpoint_path.name != EXACT_CHECKPOINT_FILENAME:
        raise CheckpointValidationError(
            f"expected exact checkpoint filename {EXACT_CHECKPOINT_FILENAME!r}, "
            f"got {checkpoint_path.name!r}. The 256 text-alignment checkpoint "
            "is not compatible with this pilot."
        )
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise CheckpointValidationError(
            f"missing {MODEL_VARIANT} checkpoint: {checkpoint_path}. Download "
            f"{MODEL_REPOSITORY}/{EXACT_CHECKPOINT_FILENAME} after access is approved."
        )
    if require_official_content:
        actual_bytes = checkpoint_path.stat().st_size
        if actual_bytes != EXPECTED_CHECKPOINT_BYTES:
            raise CheckpointValidationError(
                "VGGT-Omega checkpoint is incomplete or not the official 1B/512 "
                f"artifact: expected {EXPECTED_CHECKPOINT_BYTES} bytes, got "
                f"{actual_bytes}"
            )
        actual_sha256 = _sha256(checkpoint_path)
        if actual_sha256 != EXPECTED_CHECKPOINT_SHA256:
            raise CheckpointValidationError(
                "VGGT-Omega checkpoint SHA-256 does not match the official "
                "1B/512 artifact"
            )
    return checkpoint_path


def _validate_file_binding(binding: FileBinding, *, label: str) -> None:
    if not binding.path.is_file():
        raise PilotManifestError(f"missing {label}: {binding.path}")
    actual_bytes = binding.path.stat().st_size
    if actual_bytes != binding.bytes:
        raise PilotManifestError(
            f"{label} byte count does not match manifest: "
            f"{actual_bytes} != {binding.bytes}: {binding.path}"
        )
    actual_sha256 = _sha256(binding.path)
    if actual_sha256 != binding.sha256:
        raise PilotManifestError(
            f"{label} SHA-256 does not match manifest: {binding.path}"
        )


def validate_input_files(inputs: PilotInputs, *, require_object_masks: bool) -> None:
    """Validate images, declared geometry, and optionally both object masks."""

    if inputs.manifest_path.stat().st_size != inputs.manifest_bytes:
        raise PilotManifestError("input manifest changed after it was parsed (byte count)")
    if _sha256(inputs.manifest_path) != inputs.manifest_sha256:
        raise PilotManifestError("input manifest changed after it was parsed (SHA-256)")
    for view in inputs.views:
        _validate_file_binding(view.image, label=f"{view.camera} input image")
        _validate_file_binding(
            view.source_image, label=f"{view.camera} bound source image"
        )
        try:
            with Image.open(view.image_path) as image:
                actual_size = image.size
        except OSError as exc:
            raise PilotManifestError(
                f"failed to read {view.camera} input image {view.image_path}: {exc}"
            ) from exc
        expected_size = (inputs.image_width, inputs.image_height)
        if actual_size != expected_size:
            raise PilotManifestError(
                f"{view.camera} image size {actual_size} does not match manifest "
                f"image_geometry {expected_size}: {view.image_path}"
            )

    if require_object_masks:
        require_dual_object_masks(inputs)
    expected_mask_shape = (inputs.image_height, inputs.image_width)
    for view in inputs.views:
        if view.object_mask is None:
            continue
        _validate_file_binding(
            view.object_mask, label=f"{view.camera} modal object mask"
        )
        mask = load_binary_mask(view.object_mask.path)
        if mask.shape != expected_mask_shape:
            raise PilotManifestError(
                f"{view.camera} mask shape {mask.shape} does not match original "
                f"image shape {expected_mask_shape}"
            )


def load_binary_mask(mask_path: Path) -> np.ndarray:
    """Load a selected-frame binary mask from NPY, NPZ, or an image file."""

    mask_path = mask_path.expanduser().resolve()
    if not mask_path.is_file():
        raise PilotManifestError(f"missing object mask: {mask_path}")

    suffix = mask_path.suffix.lower()
    try:
        if suffix == ".npy":
            mask = np.load(mask_path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(mask_path, allow_pickle=False) as archive:
                if "mask" in archive.files:
                    mask = np.asarray(archive["mask"])
                elif len(archive.files) == 1:
                    mask = np.asarray(archive[archive.files[0]])
                else:
                    raise PilotManifestError(
                        f"{mask_path}: NPZ mask must contain key 'mask' or one array"
                    )
        else:
            with Image.open(mask_path) as image:
                if image.mode not in {"1", "L", "I", "I;16", "F"}:
                    raise PilotManifestError(
                        f"{mask_path}: mask image mode {image.mode!r} is not single-channel"
                    )
                mask = np.asarray(image)
    except (OSError, ValueError) as exc:
        if isinstance(exc, PilotManifestError):
            raise
        raise PilotManifestError(f"failed to load mask {mask_path}: {exc}") from exc

    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise PilotManifestError(
            f"{mask_path}: selected-frame mask must be 2-D, got shape {mask.shape}"
        )
    if not (np.issubdtype(mask.dtype, np.number) or mask.dtype == np.bool_):
        raise PilotManifestError(f"{mask_path}: mask dtype {mask.dtype} is not numeric")
    if not np.isfinite(mask).all():
        raise PilotManifestError(f"{mask_path}: mask contains non-finite values")
    unique_values = set(np.unique(mask).tolist())
    if not (
        unique_values.issubset({0, 1})
        or unique_values.issubset({0, 255})
    ):
        raise PilotManifestError(
            f"{mask_path}: mask must use one strict binary convention "
            f"(0/1 or 0/255), got {sorted(unique_values)[:10]}"
        )
    result = mask.astype(bool, copy=False)
    if not result.any():
        raise PilotManifestError(f"{mask_path}: object mask is empty")
    return result


def transform_mask_nearest(
    mask: np.ndarray,
    target_hw: tuple[int, int],
    *,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
) -> np.ndarray:
    """Mirror VGGT-Omega's aspect crop and resize a mask with nearest sampling."""

    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    target_height, target_width = (int(target_hw[0]), int(target_hw[1]))
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"target_hw must be positive, got {target_hw}")

    height, width = mask.shape
    aspect_ratio = height / max(width, 1)
    if aspect_ratio < min_aspect_ratio:
        crop_width = min(
            width, max(1, int(round(height / min_aspect_ratio)))
        )
        left = max((width - crop_width) // 2, 0)
        mask = mask[:, left : left + crop_width]
    elif aspect_ratio > max_aspect_ratio:
        crop_height = min(
            height, max(1, int(round(width * max_aspect_ratio)))
        )
        top = max((height - crop_height) // 2, 0)
        mask = mask[top : top + crop_height, :]

    pil_mask = Image.fromarray(mask.astype(np.uint8, copy=False) * 255)
    resized = pil_mask.resize(
        (target_width, target_height), resample=Image.Resampling.NEAREST
    )
    return np.asarray(resized, dtype=np.uint8).astype(bool)


def unproject_depth_map_to_world(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """Use the official VGGT-Omega camera-from-world unprojection formula."""

    depth_map = np.asarray(depth_map)
    extrinsic = np.asarray(extrinsic)
    intrinsic = np.asarray(intrinsic)
    if depth_map.ndim != 4 or depth_map.shape[-1] != 1:
        raise ValueError(
            f"depth_map must have shape (S,H,W,1), got {depth_map.shape}"
        )
    num_frames, height, width, _ = depth_map.shape
    if extrinsic.shape != (num_frames, 3, 4):
        raise ValueError(
            f"extrinsic must have shape {(num_frames, 3, 4)}, "
            f"got {extrinsic.shape}"
        )
    if intrinsic.shape != (num_frames, 3, 3):
        raise ValueError(
            f"intrinsic must have shape {(num_frames, 3, 3)}, "
            f"got {intrinsic.shape}"
        )

    depth = depth_map[..., 0]
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    if np.any(np.isclose(fx, 0.0)) or np.any(np.isclose(fy, 0.0)):
        raise ValueError("intrinsic focal lengths must be non-zero")

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    ).astype(np.float32, copy=False)


def depth_edge_mask(
    depth: np.ndarray, *, rtol: float = 0.03, kernel_size: int = 3
) -> np.ndarray:
    """Mark relative depth discontinuities using the official demo rule."""

    depth = np.asarray(depth)
    if depth.ndim != 3:
        raise ValueError(f"depth must have shape (S,H,W), got {depth.shape}")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    original_shape = depth.shape
    flat = depth.reshape(-1, *original_shape[-2:])
    pad = kernel_size // 2
    padded = np.pad(flat, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(flat, -np.inf)
    depth_min = np.full_like(flat, np.inf)
    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[:, y : y + flat.shape[-2], x : x + flat.shape[-1]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)
    relative_jump = (depth_max - depth_min) / np.maximum(np.abs(flat), 1e-6)
    return (relative_jump > float(rtol)).reshape(original_shape)


def _images_to_uint8_rgb(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError(f"images must be 4-D, got shape {images.shape}")
    if images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    if images.shape[-1] != 3:
        raise ValueError(f"images must have three RGB channels, got {images.shape}")
    if images.dtype == np.uint8:
        return images
    return (images.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)


def _view_names(view_count: int) -> tuple[str, ...]:
    return tuple(
        CAMERA_ORDER[index] if index < len(CAMERA_ORDER) else str(index)
        for index in range(view_count)
    )


def _counts_by_view(mask: np.ndarray, view_names: Sequence[str]) -> dict[str, int]:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 3 or value.shape[0] != len(view_names):
        raise ValueError("per-view count mask has an invalid shape")
    return {
        camera: int(value[index].sum())
        for index, camera in enumerate(view_names)
    }


def _bounded_depth_edge_schedule(initial_rtol: float) -> tuple[float, ...]:
    initial = float(initial_rtol)
    if (
        not np.isfinite(initial)
        or initial <= 0.0
        or initial > MAX_ADAPTIVE_DEPTH_EDGE_RTOL
    ):
        raise ValueError(
            "depth_edge_rtol must be finite and in "
            f"(0, {MAX_ADAPTIVE_DEPTH_EDGE_RTOL}]"
        )
    values = [initial]
    values.extend(value for value in ADAPTIVE_DEPTH_EDGE_RTOLS if value > initial)
    return tuple(values)


def _stratified_keep_indices(
    view_indices: np.ndarray,
    *,
    max_points: int,
    required_view_count: int,
) -> np.ndarray:
    """Deterministically cap points while retaining every required view."""

    views = np.asarray(view_indices)
    if views.ndim != 1:
        raise ValueError("view_indices must be one-dimensional")
    if max_points < required_view_count:
        raise VGGTOmegaPilotError(
            f"max_points={max_points} cannot retain {required_view_count} required views"
        )
    counts = np.asarray(
        [np.count_nonzero(views == index) for index in range(required_view_count)],
        dtype=np.int64,
    )
    if np.any(counts <= 0):
        missing = [
            CAMERA_ORDER[index] if index < len(CAMERA_ORDER) else str(index)
            for index in np.flatnonzero(counts <= 0)
        ]
        raise VGGTOmegaPilotError(
            "cannot stratify a selection missing required views: " + ", ".join(missing)
        )
    allocation = np.ones(required_view_count, dtype=np.int64)
    remaining = int(max_points - allocation.sum())
    while remaining > 0:
        capacity = counts - allocation
        available = np.flatnonzero(capacity > 0)
        if not len(available):
            break
        capacity_sum = int(capacity[available].sum())
        proposed = np.floor(
            remaining * capacity[available].astype(np.float64) / capacity_sum
        ).astype(np.int64)
        proposed = np.minimum(proposed, capacity[available])
        if not np.any(proposed):
            chosen = int(available[np.argmax(capacity[available])])
            allocation[chosen] += 1
            remaining -= 1
            continue
        allocation[available] += proposed
        remaining -= int(proposed.sum())

    selected: list[np.ndarray] = []
    for view_index, count in enumerate(allocation):
        positions = np.flatnonzero(views == view_index)
        if int(count) == len(positions):
            selected.append(positions)
        else:
            offsets = np.linspace(0, len(positions) - 1, int(count)).astype(np.int64)
            selected.append(positions[offsets])
    return np.sort(np.concatenate(selected))


def filter_official_global_object_points(
    world_points: np.ndarray,
    images: np.ndarray,
    depth_confidence: np.ndarray,
    *,
    pixel_mask: np.ndarray,
    depth: np.ndarray,
    confidence_percentile: float = OFFICIAL_REFERENCE_CONFIDENCE_PERCENTILE,
    max_points: int = OFFICIAL_REFERENCE_MAX_POINTS,
    depth_edge_rtol: float = DEFAULT_DEPTH_EDGE_RTOL,
) -> dict[str, Any]:
    """Apply the official visualization confidence rule to one dual-mask union.

    ``visual_util.predictions_to_glb`` zeroes confidence at depth edges, computes
    one global percentile, applies that threshold, and finally rejects
    confidence at or below ``1e-5``.  The official utility has no object-mask
    argument, so this reference branch changes only the percentile population:
    it is the union of the supplied MH/SH masks.  It never adds a per-view
    threshold or fallback.
    """

    world_points = np.asarray(world_points)
    depth_confidence = np.asarray(depth_confidence)
    pixel_mask = np.asarray(pixel_mask, dtype=bool)
    depth_values = np.asarray(depth)
    if world_points.ndim != 4 or world_points.shape[-1] != 3:
        raise ValueError(
            f"world_points must have shape (S,H,W,3), got {world_points.shape}"
        )
    shape_shw = world_points.shape[:3]
    if shape_shw[0] != len(CAMERA_ORDER):
        raise ValueError(
            f"official dual-mask filter expects {len(CAMERA_ORDER)} views, "
            f"got {shape_shw[0]}"
        )
    if depth_confidence.shape != shape_shw:
        raise ValueError(
            f"depth_confidence must have shape {shape_shw}, "
            f"got {depth_confidence.shape}"
        )
    if pixel_mask.shape != shape_shw:
        raise ValueError(
            f"pixel_mask must have shape {shape_shw}, got {pixel_mask.shape}"
        )
    if depth_values.shape == shape_shw + (1,):
        depth_values = depth_values[..., 0]
    if depth_values.shape != shape_shw:
        raise ValueError(f"depth must have shape {shape_shw}, got {depth_values.shape}")
    if not 0.0 <= confidence_percentile <= 100.0:
        raise ValueError("confidence_percentile must be in [0, 100]")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    depth_schedule = _bounded_depth_edge_schedule(depth_edge_rtol)
    if len(depth_schedule) == 0:
        raise RuntimeError("official depth-edge schedule is unexpectedly empty")

    colors = _images_to_uint8_rgb(images)
    if colors.shape[:3] != shape_shw:
        raise ValueError(
            f"image grid {colors.shape[:3]} does not match points {shape_shw}"
        )
    view_names = _view_names(shape_shw[0])
    edge_mask = depth_edge_mask(depth_values, rtol=float(depth_edge_rtol))
    effective_confidence = depth_confidence.copy()
    effective_confidence[edge_mask] = 0.0
    finite = np.isfinite(world_points).all(axis=-1) & np.isfinite(
        effective_confidence
    )
    scoped_finite = pixel_mask & finite
    if not np.any(scoped_finite):
        raise VGGTOmegaPilotError(
            "official global-p20 object filter has no finite dual-mask candidates"
        )
    confidence_threshold = float(
        np.percentile(
            effective_confidence[scoped_finite], confidence_percentile
        )
    )
    after_global_percentile = scoped_finite & (
        effective_confidence >= confidence_threshold
    )
    final_valid = after_global_percentile & (
        effective_confidence > MIN_POSITIVE_CONFIDENCE
    )
    if not np.any(final_valid):
        raise VGGTOmegaPilotError(
            "official global-p20 object filter retained no samples"
        )

    view_grid = np.broadcast_to(
        np.arange(shape_shw[0], dtype=np.int16)[:, None, None], shape_shw
    )
    points = world_points[final_valid].astype(np.float32, copy=False)
    point_colors = colors[final_valid]
    confidence = depth_confidence[final_valid].astype(np.float32, copy=False)
    selected_depth = depth_values[final_valid].astype(np.float32, copy=False)
    view_indices = view_grid[final_valid]
    selected_before_limit = int(len(points))
    if len(points) > max_points:
        keep = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[keep]
        point_colors = point_colors[keep]
        confidence = confidence[keep]
        selected_depth = selected_depth[keep]
        view_indices = view_indices[keep]

    exported_by_view = {
        camera: int(np.count_nonzero(view_indices == index))
        for index, camera in enumerate(view_names)
    }
    stage_masks = {
        "masked": pixel_mask,
        "finite_world_and_effective_confidence": scoped_finite,
        "after_depth_edge": scoped_finite
        & ~edge_mask
        & (effective_confidence > MIN_POSITIVE_CONFIDENCE),
        "after_global_percentile": after_global_percentile,
        "after_official_positive_confidence": final_valid,
    }
    stage_counts = {
        name: _counts_by_view(mask, view_names)
        for name, mask in stage_masks.items()
    }
    return {
        "points": points,
        "colors": point_colors,
        "confidence": confidence,
        "input_depth": selected_depth,
        "view_indices": view_indices,
        "stats": {
            "policy": "official_rule_global_p20_scoped_to_dual_object_mask_union_v1",
            "rule_reference": "visual_util.predictions_to_glb confidence/depth-edge rule",
            "exact_official_demo_output": False,
            "confidence_percentile": float(confidence_percentile),
            "confidence_threshold": confidence_threshold,
            "confidence_population": "finite_dual_object_mask_union_with_depth_edges_zeroed",
            "depth_edge_rtol": float(depth_edge_rtol),
            "depth_edge_mode": "official_confidence_zeroing",
            "per_view_threshold_fallback": False,
            "per_view_thresholds": None,
            "stage_counts_by_view": stage_counts,
            "selected_before_max_points": selected_before_limit,
            "max_points": int(max_points),
            "max_points_sampling": "official_global_linspace",
            "exported_count": int(len(points)),
            "exported_points_by_view": exported_by_view,
            "all_views_contributed": all(
                count > 0 for count in exported_by_view.values()
            ),
        },
    }


def filter_point_cloud(
    world_points: np.ndarray,
    images: np.ndarray,
    depth_confidence: np.ndarray,
    *,
    pixel_mask: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    confidence_percentile: float = 50.0,
    max_points: int = 300_000,
    filter_depth_edges: bool = True,
    depth_edge_rtol: float = DEFAULT_DEPTH_EDGE_RTOL,
    require_all_views: bool = False,
) -> dict[str, Any]:
    """Filter full-scene or object points without changing their relative frame."""

    world_points = np.asarray(world_points)
    depth_confidence = np.asarray(depth_confidence)
    if world_points.ndim != 4 or world_points.shape[-1] != 3:
        raise ValueError(
            f"world_points must have shape (S,H,W,3), got {world_points.shape}"
        )
    shape_shw = world_points.shape[:3]
    if depth_confidence.shape != shape_shw:
        raise ValueError(
            f"depth_confidence must have shape {shape_shw}, "
            f"got {depth_confidence.shape}"
        )
    colors = _images_to_uint8_rgb(images)
    if colors.shape[:3] != shape_shw:
        raise ValueError(
            f"image grid {colors.shape[:3]} does not match points {shape_shw}"
        )
    if pixel_mask is None:
        pixel_mask = np.ones(shape_shw, dtype=bool)
    else:
        pixel_mask = np.asarray(pixel_mask, dtype=bool)
        if pixel_mask.shape != shape_shw:
            raise ValueError(
                f"pixel_mask must have shape {shape_shw}, got {pixel_mask.shape}"
            )
    if not 0.0 <= confidence_percentile <= 100.0:
        raise ValueError("confidence_percentile must be in [0, 100]")
    if max_points <= 0:
        raise ValueError("max_points must be positive")

    if require_all_views and shape_shw[0] != len(CAMERA_ORDER):
        raise ValueError(
            f"require_all_views expects {len(CAMERA_ORDER)} views, got {shape_shw[0]}"
        )
    view_names = _view_names(shape_shw[0])
    if depth is None:
        if filter_depth_edges or require_all_views:
            raise ValueError(
                "depth is required for edge filtering and required-view safety"
            )
        depth_values = None
    else:
        depth_values = np.asarray(depth)
        if depth_values.shape == shape_shw + (1,):
            depth_values = depth_values[..., 0]
        if depth_values.shape != shape_shw:
            raise ValueError(
                f"depth must have shape {shape_shw}, got {depth_values.shape}"
            )

    stage_masks: dict[str, np.ndarray] = {}
    stage_masks["masked"] = pixel_mask.copy()
    stage_masks["finite_world"] = stage_masks["masked"] & np.isfinite(
        world_points
    ).all(axis=-1)
    stage_masks["finite_confidence"] = stage_masks[
        "finite_world"
    ] & np.isfinite(depth_confidence)
    stage_masks["positive_confidence"] = stage_masks[
        "finite_confidence"
    ] & (depth_confidence > MIN_POSITIVE_CONFIDENCE)
    if depth_values is None:
        stage_masks["finite_depth"] = stage_masks["positive_confidence"].copy()
        stage_masks["positive_depth"] = stage_masks["finite_depth"].copy()
    else:
        stage_masks["finite_depth"] = stage_masks[
            "positive_confidence"
        ] & np.isfinite(depth_values)
        stage_masks["positive_depth"] = stage_masks["finite_depth"] & (
            depth_values > MIN_POSITIVE_DEPTH
        )
    safe_valid = stage_masks["positive_depth"]

    safe_counts = _counts_by_view(safe_valid, view_names)
    if require_all_views:
        empty_safe_views = [
            camera for camera, count in safe_counts.items() if count == 0
        ]
        if empty_safe_views:
            raise VGGTOmegaPilotError(
                "dual-view point filtering retained no samples from: "
                + ", ".join(empty_safe_views)
                + " after finite/positive-depth safety checks"
            )

    depth_schedule: tuple[float, ...] = ()
    selected_depth_rtol: dict[str, float | None] = {
        camera: None for camera in view_names
    }
    depth_modes = {camera: "disabled" for camera in view_names}
    depth_fallback_views: list[str] = []
    if filter_depth_edges:
        assert depth_values is not None
        depth_schedule = _bounded_depth_edge_schedule(depth_edge_rtol)
        initial_edges = depth_edge_mask(depth_values, rtol=depth_schedule[0])
        initial_edge_valid = safe_valid & ~initial_edges
        adaptive_edge_valid = initial_edge_valid.copy()
        for view_index, camera in enumerate(view_names):
            selected_depth_rtol[camera] = depth_schedule[0]
            depth_modes[camera] = "initial_rtol"
            if not require_all_views or np.any(initial_edge_valid[view_index]):
                continue
            for candidate_rtol in depth_schedule[1:]:
                candidate_edges = depth_edge_mask(
                    depth_values[view_index : view_index + 1],
                    rtol=candidate_rtol,
                )[0]
                candidate = safe_valid[view_index] & ~candidate_edges
                if np.any(candidate):
                    adaptive_edge_valid[view_index] = candidate
                    selected_depth_rtol[camera] = candidate_rtol
                    depth_modes[camera] = "bounded_rtol_adaptation"
                    break
            else:
                # This explicit fallback drops only the discontinuity veto.  It
                # retains every finite-world, finite/positive-confidence, and
                # finite/positive-camera-depth condition above.
                adaptive_edge_valid[view_index] = safe_valid[view_index]
                selected_depth_rtol[camera] = None
                depth_modes[camera] = "finite_positive_depth_fallback"
                depth_fallback_views.append(camera)
    else:
        initial_edge_valid = safe_valid.copy()
        adaptive_edge_valid = safe_valid.copy()
    stage_masks["after_depth_edge_initial"] = initial_edge_valid
    stage_masks["after_depth_edge_adaptive"] = adaptive_edge_valid

    candidate_count = int(adaptive_edge_valid.sum())
    if candidate_count == 0:
        raise VGGTOmegaPilotError(
            "no safe point-cloud samples remain before confidence filtering"
        )
    confidence_threshold = float(
        np.percentile(
            depth_confidence[adaptive_edge_valid], confidence_percentile
        )
    )
    global_confidence_valid = adaptive_edge_valid & (
        depth_confidence >= confidence_threshold
    )
    stage_masks["after_global_confidence"] = global_confidence_valid

    final_valid = global_confidence_valid.copy()
    confidence_fallback_views: list[str] = []
    confidence_thresholds_by_view = {
        camera: confidence_threshold for camera in view_names
    }
    confidence_modes = {camera: "global_percentile" for camera in view_names}
    if require_all_views:
        for view_index, camera in enumerate(view_names):
            if np.any(global_confidence_valid[view_index]):
                continue
            view_candidates = adaptive_edge_valid[view_index]
            if not np.any(view_candidates):
                raise VGGTOmegaPilotError(
                    "dual-view point filtering retained no samples from: "
                    + camera
                    + " after bounded depth-edge adaptation"
                )
            per_view_threshold = float(
                np.percentile(
                    depth_confidence[view_index][view_candidates],
                    confidence_percentile,
                )
            )
            per_view_valid = view_candidates & (
                depth_confidence[view_index] >= per_view_threshold
            )
            if not np.any(per_view_valid):
                # Finite candidates make this branch defensive only.  Selecting
                # the finite maximum never weakens the positive-confidence gate.
                finite_confidence = np.where(
                    view_candidates,
                    depth_confidence[view_index],
                    -np.inf,
                )
                per_view_valid = np.zeros_like(view_candidates)
                per_view_valid.flat[int(np.argmax(finite_confidence))] = True
                confidence_modes[camera] = "finite_maximum_fallback"
            else:
                confidence_modes[camera] = "per_view_percentile_fallback"
            final_valid[view_index] = per_view_valid
            confidence_thresholds_by_view[camera] = per_view_threshold
            confidence_fallback_views.append(camera)
    stage_masks["after_adaptive_confidence"] = final_valid

    selected_before_limit = int(final_valid.sum())
    if selected_before_limit == 0:
        raise VGGTOmegaPilotError(
            "no point-cloud samples remain after confidence filtering"
        )
    pre_limit_counts = _counts_by_view(final_valid, view_names)
    if require_all_views:
        empty_views = [
            camera for camera, count in pre_limit_counts.items() if count == 0
        ]
        if empty_views:
            raise VGGTOmegaPilotError(
                "dual-view point filtering retained no samples from: "
                + ", ".join(empty_views)
            )

    view_grid = np.broadcast_to(
        np.arange(shape_shw[0], dtype=np.int16)[:, None, None], shape_shw
    )
    points = world_points[final_valid].astype(np.float32, copy=False)
    point_colors = colors[final_valid]
    confidence = depth_confidence[final_valid].astype(np.float32, copy=False)
    selected_depth = (
        depth_values[final_valid].astype(np.float32, copy=False)
        if depth_values is not None
        else None
    )
    view_indices = view_grid[final_valid]
    stratified_limit_used = False
    if len(points) > max_points:
        if require_all_views:
            keep = _stratified_keep_indices(
                view_indices,
                max_points=max_points,
                required_view_count=shape_shw[0],
            )
            stratified_limit_used = True
        else:
            keep = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[keep]
        point_colors = point_colors[keep]
        confidence = confidence[keep]
        view_indices = view_indices[keep]
        if selected_depth is not None:
            selected_depth = selected_depth[keep]

    view_counts = {
        camera: int(np.count_nonzero(view_indices == index))
        for index, camera in enumerate(view_names)
    }
    all_views_contributed = all(count > 0 for count in view_counts.values())
    if require_all_views and not all_views_contributed:
        empty_views = [camera for camera, count in view_counts.items() if count == 0]
        raise VGGTOmegaPilotError(
            "dual-view point filtering retained no samples from: "
            + ", ".join(empty_views)
            + " after max_points limiting"
        )
    safety_invariants = {
        "exported_world_points_finite": bool(np.isfinite(points).all()),
        "exported_confidence_finite": bool(np.isfinite(confidence).all()),
        "exported_confidence_strictly_positive": bool(
            np.all(confidence > MIN_POSITIVE_CONFIDENCE)
        ),
        "exported_input_depth_finite": bool(
            selected_depth is None or np.isfinite(selected_depth).all()
        ),
        "exported_input_depth_strictly_positive": bool(
            selected_depth is None or np.all(selected_depth > MIN_POSITIVE_DEPTH)
        ),
    }
    if not all(safety_invariants.values()):
        raise RuntimeError("point-filter safety invariant failed")

    stage_counts = {
        name: _counts_by_view(mask, view_names)
        for name, mask in stage_masks.items()
    }
    per_view = {
        camera: {
            **{
                stage_name: counts[camera]
                for stage_name, counts in stage_counts.items()
            },
            "before_max_points": pre_limit_counts[camera],
            "exported": view_counts[camera],
            "depth_edge_mode": depth_modes[camera],
            "selected_depth_edge_rtol": selected_depth_rtol[camera],
            "confidence_mode": confidence_modes[camera],
            "selected_confidence_threshold": confidence_thresholds_by_view[camera],
        }
        for camera in view_names
    }
    return {
        "points": points,
        "colors": point_colors,
        "confidence": confidence,
        "input_depth": selected_depth,
        "view_indices": view_indices,
        "stats": {
            "policy": "safe_per_view_adaptive_v1",
            "candidate_count": candidate_count,
            "selected_before_max_points": selected_before_limit,
            "exported_count": int(len(points)),
            "confidence_percentile": float(confidence_percentile),
            "confidence_threshold": confidence_threshold,
            "max_points": int(max_points),
            "depth_edge_filter": bool(filter_depth_edges),
            "depth_edge_rtol": float(depth_edge_rtol),
            "stage_counts_by_view": stage_counts,
            "per_view": per_view,
            "depth_edge_adaptation": {
                "bounded_schedule": list(depth_schedule),
                "maximum_rtol": MAX_ADAPTIVE_DEPTH_EDGE_RTOL,
                "selected_rtol_by_view": selected_depth_rtol,
                "mode_by_view": depth_modes,
                "finite_positive_depth_fallback_views": depth_fallback_views,
                "unbounded_relaxation_allowed": False,
            },
            "confidence_adaptation": {
                "global_percentile": float(confidence_percentile),
                "global_threshold": confidence_threshold,
                "per_view_thresholds": confidence_thresholds_by_view,
                "mode_by_view": confidence_modes,
                "fallback_views": confidence_fallback_views,
            },
            "safety": {
                "minimum_positive_confidence": MIN_POSITIVE_CONFIDENCE,
                "minimum_positive_depth": MIN_POSITIVE_DEPTH,
                "conditions_never_relaxed": [
                    "pixel_mask",
                    "finite_world_point",
                    "finite_confidence",
                    "strictly_positive_confidence",
                    "finite_input_depth",
                    "strictly_positive_input_depth",
                ],
                "invariants": safety_invariants,
            },
            "exported_points_by_view": view_counts,
            "all_views_required": bool(require_all_views),
            "all_views_contributed": all_views_contributed,
            "stratified_max_points_limit_used": stratified_limit_used,
        },
    }


def export_colored_point_cloud(
    selection: Mapping[str, Any], *, ply_path: Path, glb_path: Path
) -> None:
    """Export the same relative-world colored point set as PLY and GLB."""

    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - dependency preflight
        raise VGGTOmegaPilotError(
            "trimesh is required to export VGGT-Omega GLB/PLY point clouds"
        ) from exc

    points = np.asarray(selection["points"], dtype=np.float32)
    colors = np.asarray(selection["colors"], dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise VGGTOmegaPilotError(
            f"cannot export invalid/empty point array with shape {points.shape}"
        )
    if colors.shape != (len(points), 3):
        raise VGGTOmegaPilotError(
            f"point colors must have shape {(len(points), 3)}, got {colors.shape}"
        )

    point_cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
    ply_path.write_bytes(point_cloud.export(file_type="ply"))
    scene = trimesh.Scene()
    scene.add_geometry(point_cloud, geom_name="vggt_omega_relative_point_cloud")
    glb_path.write_bytes(scene.export(file_type="glb"))


def _tensor_predictions_to_numpy(predictions: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    converted: dict[str, Any] = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            array = value.detach().float().cpu().numpy()
            if array.shape[0] == 1:
                array = array[0]
            converted[key] = array
    return converted


def _validate_prediction_shapes(
    predictions: Mapping[str, np.ndarray], *, num_views: int
) -> tuple[int, int]:
    required = (
        "pose_enc",
        "depth",
        "depth_conf",
        "images",
        "extrinsic",
        "intrinsic",
    )
    missing = [key for key in required if key not in predictions]
    if missing:
        raise VGGTOmegaPilotError(
            f"VGGT-Omega predictions are missing required keys: {missing}"
        )
    depth = predictions["depth"]
    if depth.ndim != 4 or depth.shape[0] != num_views or depth.shape[-1] != 1:
        raise VGGTOmegaPilotError(
            f"unexpected depth shape {depth.shape}; expected (2,H,W,1)"
        )
    height, width = int(depth.shape[1]), int(depth.shape[2])
    expected = {
        "pose_enc": (num_views, 9),
        "depth_conf": (num_views, height, width),
        "images": (num_views, 3, height, width),
        "extrinsic": (num_views, 3, 4),
        "intrinsic": (num_views, 3, 3),
    }
    for key, shape in expected.items():
        if predictions[key].shape != shape:
            raise VGGTOmegaPilotError(
                f"unexpected {key} shape {predictions[key].shape}; expected {shape}"
            )
    return height, width


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_output_dir(inputs: PilotInputs) -> Path:
    bundle_root = inputs.bundle_root
    parent = bundle_root.parent if bundle_root.name == "inputs" else bundle_root
    # Preserve the final component until output safety has checked whether it
    # is a symlink.  ``bundle_root`` is already canonical and absolute.
    return parent / "vggt_omega"


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


def validate_output_directory(
    path: Path,
    *,
    inputs: PilotInputs,
    checkpoint_path: Path,
    vggt_repository: Path,
    repo_root: Path = REPO_ROOT,
    dedicated_repo_output_root: Path | None = None,
) -> Path:
    """Return a dedicated output path that cannot replace inputs or source trees."""

    requested = path.expanduser()
    if requested.is_symlink():
        raise VGGTOmegaPilotError(
            f"unsafe output directory is a symbolic link: {requested}"
        )
    output = requested.resolve()
    protected: dict[str, Path] = {
        "input manifest": inputs.manifest_path,
        "input bundle": inputs.bundle_root,
        "VGGT-Omega checkpoint": checkpoint_path.expanduser().resolve(),
        "VGGT-Omega repository": vggt_repository.expanduser().resolve(),
    }
    for view in inputs.views:
        protected[f"{view.camera} input image"] = view.image.path
        protected[f"{view.camera} source image"] = view.source_image.path
        if view.object_mask is not None:
            protected[f"{view.camera} object mask"] = view.object_mask.path
    for label, protected_path in protected.items():
        if _paths_overlap(output, protected_path):
            raise VGGTOmegaPilotError(
                f"unsafe output directory {output} overlaps protected {label} "
                f"{protected_path}"
            )

    repository = repo_root.expanduser().resolve()
    if _paths_overlap(output, repository):
        dedicated = (
            dedicated_repo_output_root.expanduser().resolve()
            if dedicated_repo_output_root is not None
            else _default_output_dir(inputs)
        )
        if output != dedicated and not output.is_relative_to(dedicated):
            raise VGGTOmegaPilotError(
                f"unsafe output directory {output} overlaps repository root "
                f"{repository}; repository-local output is restricted to "
                f"{dedicated}"
            )
    return output


def inspect_output_for_preflight(output: Path, *, force: bool) -> dict[str, Any]:
    """Describe publication state without creating, replacing, or deleting paths."""

    output = output.expanduser().resolve()
    nearest_parent = output.parent
    while not nearest_parent.exists() and nearest_parent != nearest_parent.parent:
        nearest_parent = nearest_parent.parent
    if not nearest_parent.is_dir():
        raise VGGTOmegaPilotError(
            f"output parent chain is blocked by a non-directory: {nearest_parent}"
        )
    if output.exists() and not output.is_dir():
        raise VGGTOmegaPilotError(
            f"preflight output path exists but is not a directory: {output}"
        )
    exists = output.is_dir()
    try:
        existing_entry_count = sum(1 for _ in output.iterdir()) if exists else 0
    except OSError as exc:
        raise VGGTOmegaPilotError(
            f"cannot inspect existing output directory {output}: {exc}"
        ) from exc
    return {
        "path": str(output),
        "exists": exists,
        "path_type": "directory" if exists else "missing",
        "existing_entry_count": existing_entry_count,
        "existing_metadata_json": (output / "metadata.json").is_file()
        if exists
        else False,
        "parent_exists": output.parent.is_dir(),
        "nearest_existing_parent": str(nearest_parent),
        "force_requested": bool(force),
        "force_required_for_inference": exists,
        "publish_allowed_with_current_flags": not exists or bool(force),
        "planned_publish_action": (
            "replace_existing_directory"
            if exists and force
            else "blocked_requires_force"
            if exists
            else "create_new_directory"
        ),
        "preflight_read_only": True,
        "modified_by_preflight": False,
    }


def _output_record(staged_path: Path, final_path: Path) -> dict[str, Any]:
    if not staged_path.is_file():
        raise VGGTOmegaPilotError(f"missing staged output artifact: {staged_path}")
    return {
        "path": str(final_path),
        "bytes": staged_path.stat().st_size,
        "sha256": _sha256(staged_path),
    }


def _publish_directory(staging: Path, output: Path, *, force: bool) -> None:
    if output.exists():
        if not output.is_dir():
            raise VGGTOmegaPilotError(
                f"refusing to replace non-directory output path: {output}"
            )
        if not force:
            raise VGGTOmegaPilotError(
                f"output directory already exists: {output}; pass --force to replace it"
            )
        backup = output.with_name(f".{output.name}.backup")
        if backup.exists():
            raise VGGTOmegaPilotError(f"stale VGGT-Omega backup exists: {backup}")
        output.replace(backup)
        try:
            staging.replace(output)
        except BaseException:
            backup.replace(output)
            raise
        shutil.rmtree(backup)
    else:
        staging.replace(output)


def geometry_contract() -> dict[str, Any]:
    """Machine-readable guardrails for all artifacts produced here."""

    return {
        "representation": "colored_point_cloud",
        "primitive": "points",
        "has_triangle_faces": False,
        "is_triangle_mesh": False,
        "is_watertight": False,
        "collision_ready": False,
        "coordinate_frame": "vggt_omega_predicted_world_opencv",
        "scale": "relative_non_metric",
        "metric_scale_verified": False,
        "camera_extrinsic_convention": "camera_from_world_3x4_opencv",
        "camera_parameters_source": "vggt_omega_prediction",
        "provided_calibration_applied": False,
        "warning": (
            "VGGT-Omega GLB/PLY artifacts are point clouds at arbitrary relative "
            "scale. The provided checkerboard calibration is provenance only in "
            "this stage. Register scale/pose and build a validated surface before "
            "using the points for hand-object collision or occlusion."
        ),
    }


def model_provenance_contract() -> dict[str, str]:
    """Pinned model/code/license identity shared by preflight and run metadata."""

    return {
        "repository": MODEL_REPOSITORY,
        "variant": MODEL_VARIANT,
        "official_huggingface_checkpoint_commit": OFFICIAL_HF_CHECKPOINT_COMMIT,
        "local_code_commit": EXPECTED_LOCAL_CODE_COMMIT,
        "license": MODEL_LICENSE,
    }


def _run_pilot_to_staging(
    *,
    inputs: PilotInputs,
    checkpoint_path: Path,
    vggt_repository: Path,
    staging_dir: Path,
    final_output_dir: Path,
    full_scene_only: bool = False,
    confidence_percentile: float = 50.0,
    max_points: int = 300_000,
) -> dict[str, Any]:
    """Run inference and write one complete, unpublished staging bundle."""

    validate_input_files(inputs, require_object_masks=not full_scene_only)
    package_root = vggt_repository / "vggt_omega"
    if not package_root.is_dir():
        raise VGGTOmegaPilotError(
            f"missing official VGGT-Omega repository/package: {package_root}"
        )
    if str(vggt_repository) not in sys.path:
        sys.path.insert(0, str(vggt_repository))

    try:
        import torch
        from visual_util import predictions_to_glb
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera
    except ImportError as exc:
        raise VGGTOmegaPilotError(
            "failed to import official VGGT-Omega or its dependencies; use the "
            "haco environment and the official third_party/VGGT-Omega checkout"
        ) from exc

    if not torch.cuda.is_available():
        raise VGGTOmegaPilotError("VGGT-Omega pilot requires a CUDA GPU")
    if tuple(torch.cuda.get_device_capability(0)) < (8, 0):
        raise VGGTOmegaPilotError(
            "VGGT-Omega 1B/512 requires a CUDA GPU with bf16 support"
        )

    image_paths = [str(view.image_path) for view in inputs.views]
    images = load_and_preprocess_images(
        image_paths,
        mode=PREPROCESS_MODE,
        image_resolution=IMAGE_RESOLUTION,
        patch_size=PATCH_SIZE,
    )
    if images.ndim != 4 or images.shape[0] != len(CAMERA_ORDER):
        raise VGGTOmegaPilotError(
            f"unexpected preprocessed image shape {tuple(images.shape)}"
        )

    # PyTorch 2.7's restricted loader and mmap avoid an unnecessary duplicate
    # checkpoint copy.  ``strict=True`` is the content-level architecture check.
    model = VGGTOmega().eval()
    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        model.load_state_dict(state_dict, strict=True, assign=True)
    except Exception as exc:
        raise CheckpointValidationError(
            f"failed strict load of {EXACT_CHECKPOINT_FILENAME}: {exc}"
        ) from exc
    del state_dict
    model = model.to("cuda")
    images = images.to("cuda", non_blocking=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        predictions = model(images)
        extrinsic, intrinsic = encoding_to_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
    peak_gpu_bytes = int(torch.cuda.max_memory_allocated())

    predictions_np = _tensor_predictions_to_numpy(predictions)
    height, width = _validate_prediction_shapes(
        predictions_np, num_views=len(CAMERA_ORDER)
    )
    world_points = unproject_depth_map_to_world(
        predictions_np["depth"],
        predictions_np["extrinsic"],
        predictions_np["intrinsic"],
    )
    predictions_np["world_points_from_depth"] = world_points

    official_full_scene_glb = (
        staging_dir / "official_full_scene_p20_aligned_with_cameras.glb"
    )
    official_scene = predictions_to_glb(predictions_np)
    official_scene.export(file_obj=str(official_full_scene_glb))
    if (
        not official_full_scene_glb.is_file()
        or official_full_scene_glb.stat().st_size <= 0
    ):
        raise VGGTOmegaPilotError(
            "official visual_util conversion produced no full-scene GLB"
        )
    del official_scene

    np.save(staging_dir / "depth_relative.npy", predictions_np["depth"])
    np.save(
        staging_dir / "depth_confidence.npy", predictions_np["depth_conf"]
    )
    np.save(staging_dir / "world_points_relative.npy", world_points)
    np.save(staging_dir / "preprocessed_images.npy", predictions_np["images"])
    np.savez_compressed(
        staging_dir / "cameras_relative.npz",
        pose_encoding=predictions_np["pose_enc"],
        extrinsic_camera_from_world=predictions_np["extrinsic"],
        intrinsic_model_pixels=predictions_np["intrinsic"],
        camera_order=np.asarray(CAMERA_ORDER),
        frame_indices=np.asarray(
            [view.frame_index for view in inputs.views], dtype=np.int64
        ),
        input_paths=np.asarray(image_paths),
        original_size_wh=np.asarray(
            [inputs.image_width, inputs.image_height], dtype=np.int64
        ),
        model_size_hw=np.asarray([height, width], dtype=np.int64),
        scale_status=np.asarray("relative_non_metric"),
    )

    full_scene = filter_point_cloud(
        world_points,
        predictions_np["images"],
        predictions_np["depth_conf"],
        depth=predictions_np["depth"],
        confidence_percentile=confidence_percentile,
        max_points=max_points,
    )
    full_scene_ply = staging_dir / "full_scene_relative_point_cloud.ply"
    full_scene_glb = staging_dir / "full_scene_relative_point_cloud.glb"
    export_colored_point_cloud(
        full_scene, ply_path=full_scene_ply, glb_path=full_scene_glb
    )

    object_aggregation: dict[str, Any]
    object_artifact_names: dict[str, str] = {}
    if full_scene_only:
        object_aggregation = {
            "method": OBJECT_AGGREGATION_METHOD,
            "requested": False,
            "status": "not_run_full_scene_only",
            "performed": False,
            "reason": (
                "--full-scene-only was selected; both cameras informed scene "
                "geometry, but no mask-filtered object point aggregation was performed"
            ),
        }
    else:
        mask_paths = require_dual_object_masks(inputs)
        original_masks = [load_binary_mask(path) for path in mask_paths]
        expected_mask_shape = (inputs.image_height, inputs.image_width)
        for camera, mask in zip(CAMERA_ORDER, original_masks):
            if mask.shape != expected_mask_shape:
                raise PilotManifestError(
                    f"{camera} mask shape {mask.shape} does not match original "
                    f"image shape {expected_mask_shape}"
                )
        model_masks = np.stack(
            [transform_mask_nearest(mask, (height, width)) for mask in original_masks]
        )
        np.save(staging_dir / "object_masks_model_input.npy", model_masks)
        official_object_points = filter_official_global_object_points(
            world_points,
            predictions_np["images"],
            predictions_np["depth_conf"],
            pixel_mask=model_masks,
            depth=predictions_np["depth"],
        )
        official_object_evidence = (
            staging_dir / "object_official_global_p20_evidence.npz"
        )
        np.savez_compressed(
            official_object_evidence,
            points_relative=official_object_points["points"],
            colors_rgb=official_object_points["colors"],
            depth_confidence=official_object_points["confidence"],
            input_depth_relative=official_object_points["input_depth"],
            view_indices=official_object_points["view_indices"],
            camera_order=np.asarray(CAMERA_ORDER),
        )
        official_object_ply = (
            staging_dir / "object_official_global_p20_relative_point_cloud.ply"
        )
        official_object_glb = (
            staging_dir / "object_official_global_p20_relative_point_cloud.glb"
        )
        export_colored_point_cloud(
            official_object_points,
            ply_path=official_object_ply,
            glb_path=official_object_glb,
        )
        object_points = filter_point_cloud(
            world_points,
            predictions_np["images"],
            predictions_np["depth_conf"],
            pixel_mask=model_masks,
            depth=predictions_np["depth"],
            confidence_percentile=confidence_percentile,
            max_points=max_points,
            require_all_views=True,
        )
        object_filter_stats = object_points["stats"]
        exported_by_view = object_filter_stats["exported_points_by_view"]
        if (
            not object_filter_stats["all_views_contributed"]
            or any(exported_by_view[camera] <= 0 for camera in CAMERA_ORDER)
        ):
            raise VGGTOmegaPilotError(
                "object aggregation did not retain points from both required views"
            )
        object_evidence = (
            staging_dir / "object_dual_mask_filtered_evidence.npz"
        )
        np.savez_compressed(
            object_evidence,
            points_relative=object_points["points"],
            colors_rgb=object_points["colors"],
            depth_confidence=object_points["confidence"],
            input_depth_relative=object_points["input_depth"],
            view_indices=object_points["view_indices"],
            camera_order=np.asarray(CAMERA_ORDER),
        )
        object_ply = (
            staging_dir
            / "object_dual_mask_filtered_relative_point_cloud.ply"
        )
        object_glb = (
            staging_dir
            / "object_dual_mask_filtered_relative_point_cloud.glb"
        )
        export_colored_point_cloud(
            object_points, ply_path=object_ply, glb_path=object_glb
        )
        object_artifact_names = {
            "official_object_point_cloud_ply": official_object_ply.name,
            "official_object_point_cloud_glb": official_object_glb.name,
            "official_object_point_evidence": official_object_evidence.name,
            "object_point_cloud_ply": object_ply.name,
            "object_point_cloud_glb": object_glb.name,
            "object_masks_model_input": "object_masks_model_input.npy",
            "object_point_evidence": object_evidence.name,
        }
        object_aggregation = {
            "method": OBJECT_AGGREGATION_METHOD,
            "requested": True,
            "status": "completed",
            "performed": True,
            "mask_paths": {
                camera: str(path) for camera, path in zip(CAMERA_ORDER, mask_paths)
            },
            "official_global_p20": {
                "method": "official_rule_global_p20_scoped_to_dual_object_mask_union",
                "status": "completed",
                "rule_reference": (
                    "visual_util.predictions_to_glb confidence/depth-edge rule"
                ),
                "exact_official_demo_output": False,
                "threshold_population": (
                    "finite dual object-mask union with depth-edge confidences zeroed"
                ),
                "per_view_threshold_fallback": False,
                "point_filter": official_object_points["stats"],
                "artifacts": {
                    "point_cloud_ply": official_object_ply.name,
                    "point_cloud_glb": official_object_glb.name,
                    "evidence": official_object_evidence.name,
                },
            },
            "custom_adaptive_p50": {
                "method": "safe_per_view_adaptive_v1",
                "status": "completed",
                "point_filter": object_filter_stats,
                "artifacts": {
                    "point_cloud_ply": object_ply.name,
                    "point_cloud_glb": object_glb.name,
                    "evidence": object_evidence.name,
                },
            },
            "point_filter": object_filter_stats,
            "dual_view_contribution": {
                "required": True,
                "proven": True,
                "camera_order": list(CAMERA_ORDER),
                "exported_points_by_view": exported_by_view,
                "evidence_artifact": object_evidence.name,
            },
        }

    artifact_names = {
        "cameras": "cameras_relative.npz",
        "depth": "depth_relative.npy",
        "depth_confidence": "depth_confidence.npy",
        "world_points": "world_points_relative.npy",
        "preprocessed_images": "preprocessed_images.npy",
        "official_full_scene_glb": official_full_scene_glb.name,
        "full_scene_point_cloud_ply": full_scene_ply.name,
        "full_scene_point_cloud_glb": full_scene_glb.name,
        **object_artifact_names,
    }
    artifact_records = {
        key: _output_record(staging_dir / name, final_output_dir / name)
        for key, name in artifact_names.items()
    }
    manifest_record = {
        "path": str(inputs.manifest_path),
        "bytes": inputs.manifest_bytes,
        "sha256": inputs.manifest_sha256,
        "verified": True,
    }
    metadata = {
        "schema_version": 1,
        "status": "completed",
        "model": {
            **model_provenance_contract(),
            "checkpoint_filename": EXACT_CHECKPOINT_FILENAME,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_bytes": EXPECTED_CHECKPOINT_BYTES,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "checkpoint_official_content_verified": True,
            "official_code_path": str(vggt_repository),
            "local_code_commit_verified_before_import": True,
        },
        "official_reference": {
            "repository": MODEL_REPOSITORY,
            "local_code_commit": EXPECTED_LOCAL_CODE_COMMIT,
            "conversion_function": "visual_util.predictions_to_glb",
            "full_scene_artifact": official_full_scene_glb.name,
            "full_scene_parameters": {
                "confidence_percentile": OFFICIAL_REFERENCE_CONFIDENCE_PERCENTILE,
                "depth_edge_filter": True,
                "depth_edge_rtol": DEFAULT_DEPTH_EDGE_RTOL,
                "show_cam": True,
                "mask_black_bg": False,
                "mask_white_bg": False,
                "mask_sky": False,
                "max_points": OFFICIAL_REFERENCE_MAX_POINTS,
                "scene_alignment": "official_first_camera_opengl",
            },
            "call_contract": (
                "predictions_to_glb(predictions_np) with unmodified official defaults"
            ),
            "exact_official_demo_output": True,
        },
        "input": {
            "manifest": manifest_record,
            "episode": inputs.episode,
            "object_label": inputs.object_label,
            "camera_order": list(CAMERA_ORDER),
            "views": [
                {
                    "camera": view.camera,
                    "semantic_role": view.semantic_role,
                    "frame_index": view.frame_index,
                    "image": view.image.as_dict(),
                    "source_image": view.source_image.as_dict(),
                    "object_mask": (
                        view.object_mask.as_dict()
                        if view.object_mask is not None
                        else None
                    ),
                }
                for view in inputs.views
            ],
            "calibration_record_present": inputs.calibration is not None,
        },
        "preprocessing": {
            "mode": PREPROCESS_MODE,
            "image_resolution": IMAGE_RESOLUTION,
            "patch_size": PATCH_SIZE,
            "original_size_wh": [inputs.image_width, inputs.image_height],
            "model_size_hw": [height, width],
            "mask_interpolation": "nearest",
        },
        "geometry_contract": geometry_contract(),
        "full_scene": {
            "dual_view_inference": True,
            "point_filter": full_scene["stats"],
        },
        "object_aggregation": object_aggregation,
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
            "peak_gpu_memory_bytes": peak_gpu_bytes,
        },
        "artifacts": artifact_records,
        "metadata_path": str(final_output_dir / "metadata.json"),
    }
    (staging_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    del model, predictions, images
    torch.cuda.empty_cache()
    return metadata


def preflight_pilot(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    vggt_repository: Path = DEFAULT_VGGT_REPOSITORY,
    output_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Validate complete pilot readiness without imports, GPU use, or writes."""

    inputs = load_pilot_manifest(manifest_path)
    # Preflight deliberately checks the object-level dual-camera contract even
    # when a later run might request --full-scene-only.
    validate_input_files(inputs, require_object_masks=True)
    mh_mask, sh_mask = require_dual_object_masks(inputs)

    checkpoint_candidate = checkpoint_path.expanduser()
    repository_candidate = vggt_repository.expanduser()
    requested_output = (
        output_dir.expanduser()
        if output_dir is not None
        else _default_output_dir(inputs)
    )
    output = validate_output_directory(
        requested_output,
        inputs=inputs,
        checkpoint_path=checkpoint_candidate,
        vggt_repository=repository_candidate,
    )
    output_record = inspect_output_for_preflight(output, force=force)
    repository_record = validate_vggt_repository(repository_candidate)
    checkpoint_record = inspect_checkpoint_for_preflight(checkpoint_candidate)
    status = (
        "ready"
        if checkpoint_record["state"] == "verified"
        else "waiting_for_checkpoint"
    )
    run_allowed = bool(
        status == "ready"
        and output_record["publish_allowed_with_current_flags"]
    )
    manifest_record = {
        "path": str(inputs.manifest_path),
        "bytes": inputs.manifest_bytes,
        "sha256": inputs.manifest_sha256,
        "verified": True,
    }
    return {
        "schema_version": 1,
        "mode": "preflight",
        "status": status,
        "preflight_ok": True,
        "run_allowed_with_current_flags": run_allowed,
        "inference_performed": False,
        "download_performed": False,
        "gpu_checked": False,
        "model_imported": False,
        "model": {
            **model_provenance_contract(),
            "checkpoint_filename": EXACT_CHECKPOINT_FILENAME,
            "local_code_commit_verified": True,
        },
        "input": {
            "manifest": manifest_record,
            "episode": inputs.episode,
            "object_label": inputs.object_label,
            "camera_order": list(CAMERA_ORDER),
            "frame_indices": {
                view.camera: view.frame_index for view in inputs.views
            },
            "images": {
                view.camera: view.image.as_dict() for view in inputs.views
            },
            "dual_object_masks": {
                "required": True,
                "verified": True,
                "MH": inputs.mh.object_mask.as_dict(),
                "SH": inputs.sh.object_mask.as_dict(),
                "ordered_paths": [str(mh_mask), str(sh_mask)],
            },
        },
        "repository": repository_record,
        "checkpoint": checkpoint_record,
        "output": output_record,
        "read_only_guards": {
            "output_mutated": False,
            "checkpoint_loaded": False,
            "model_code_imported": False,
            "cuda_queried": False,
            "network_accessed": False,
        },
    }


def run_pilot(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    vggt_repository: Path = DEFAULT_VGGT_REPOSITORY,
    output_dir: Path | None = None,
    full_scene_only: bool = False,
    confidence_percentile: float = 50.0,
    max_points: int = 300_000,
    force: bool = False,
) -> dict[str, Any]:
    """Validate inputs, stage a complete bundle, and atomically publish it."""

    inputs = load_pilot_manifest(manifest_path)
    validate_input_files(inputs, require_object_masks=not full_scene_only)
    checkpoint_candidate = checkpoint_path.expanduser()
    repository_candidate = vggt_repository.expanduser()
    requested_output = (
        output_dir.expanduser()
        if output_dir is not None
        else _default_output_dir(inputs)
    )
    output = validate_output_directory(
        requested_output,
        inputs=inputs,
        checkpoint_path=checkpoint_candidate,
        vggt_repository=repository_candidate,
    )
    if output.exists():
        if not output.is_dir():
            raise VGGTOmegaPilotError(
                f"refusing to replace non-directory output path: {output}"
            )
        if not force:
            raise VGGTOmegaPilotError(
                f"output directory already exists: {output}; pass --force to replace it"
            )

    repository_record = validate_vggt_repository(repository_candidate)
    repository = Path(repository_record["path"])
    checkpoint = validate_checkpoint_path(checkpoint_candidate)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent)
    )
    try:
        metadata = _run_pilot_to_staging(
            inputs=inputs,
            checkpoint_path=checkpoint,
            vggt_repository=repository,
            staging_dir=staging,
            final_output_dir=output,
            full_scene_only=full_scene_only,
            confidence_percentile=confidence_percentile,
            max_points=max_points,
        )
        _publish_directory(staging, output, force=force)
        return metadata
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run VGGT-Omega 1B/512 on the synchronized MH/SH pilot pair and "
            "export explicit relative-scale colored point clouds."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--vggt-repository", type=Path, default=DEFAULT_VGGT_REPOSITORY
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Read-only validation of the manifest, both object masks, output "
            "safety, pinned repository, license, and checkpoint. Missing or "
            "partial checkpoints report waiting_for_checkpoint with exit 0."
        ),
    )
    parser.add_argument(
        "--full-scene-only",
        action="store_true",
        help=(
            "Allow full-scene two-camera reconstruction without object masks. "
            "The metadata will state that mask-filtered aggregation was not run."
        ),
    )
    parser.add_argument(
        "--confidence-percentile", type=float, default=50.0
    )
    parser.add_argument("--max-points", type=int, default=300_000)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace the complete existing output directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.preflight:
            metadata = preflight_pilot(
                manifest_path=args.manifest,
                checkpoint_path=args.checkpoint,
                vggt_repository=args.vggt_repository,
                output_dir=args.output_dir,
                force=args.force,
            )
        else:
            metadata = run_pilot(
                manifest_path=args.manifest,
                checkpoint_path=args.checkpoint,
                vggt_repository=args.vggt_repository,
                output_dir=args.output_dir,
                full_scene_only=args.full_scene_only,
                confidence_percentile=args.confidence_percentile,
                max_points=args.max_points,
                force=args.force,
            )
    except VGGTOmegaPilotError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
