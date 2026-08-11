#!/usr/bin/env python3
"""Infer the SH Choco modal mask for the two-view mesh pilot with SAM2.

This is intentionally a single-frame, explicit-box utility.  The resulting
mask is recorded as model-inferred evidence, never as annotation or ground
truth.  The input manifest is replaced atomically only after both image
artifacts have been written successfully.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    REPO_ROOT / "8-5" / "mesh_sota_pilot" / "episode_1" / "choco" / "inputs"
)
DEFAULT_MANIFEST = DEFAULT_INPUT_DIR / "manifest.json"
DEFAULT_IMAGE = DEFAULT_INPUT_DIR / "sh_frame000192.jpg"
DEFAULT_MASK = DEFAULT_INPUT_DIR / "sh_mask_modal_sam2_frame000192.png"
DEFAULT_OVERLAY = DEFAULT_INPUT_DIR / "sh_mask_modal_sam2_overlay_frame000192.png"
DEFAULT_SAM2_ROOT = REPO_ROOT / "third_party" / "sam2"
DEFAULT_CHECKPOINT = DEFAULT_SAM2_ROOT / "checkpoints" / "sam2_hiera_large.pt"
DEFAULT_CONFIG = "sam2_hiera_l.yaml"
DEFAULT_BOX = (590, 230, 690, 390)
MODEL_NAME = "SAM2 Hiera Large"
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_MANIFEST_KIND = "mesh_sota_pilot_input_bundle"
EXPECTED_OBJECT_LABEL = "Choco"
EXPECTED_MH_FRAME_INDEX = 187
EXPECTED_SH_FRAME_INDEX = 192
EXPECTED_MH_ROLE = "primary/final"
EXPECTED_SH_ROLE = "auxiliary/evidence"


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"manifest field {field!r} must be an object")
    return value


def _require_exact(value: Any, expected: Any, field: str) -> None:
    if value != expected or isinstance(value, bool) != isinstance(expected, bool):
        raise ValueError(
            f"manifest field {field!r} must be {expected!r}, got {value!r}"
        )


def _record_path(record: dict[str, Any], field: str, bundle_root: Path) -> Path:
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"manifest field {field!r} must contain a non-empty path")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = bundle_root / path
    return path.resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or _is_within(left, right) or _is_within(right, left)


def load_and_bind_manifest(
    manifest_path: Path,
    image_path: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Bind the supplied SH image to the immutable v1 pilot input contract.

    This check intentionally runs before SAM2 is imported or its checkpoint is
    loaded.  A same-sized but different image, a stale manifest record, or a
    manifest from another frame/view/object cannot consume inference resources.
    """

    manifest_path = manifest_path.expanduser().resolve()
    image_path = image_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"missing manifest: {manifest_path}")
    if not image_path.is_file():
        raise ValueError(f"missing SH image: {image_path}")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read manifest {manifest_path}: {exc}") from exc
    document = _require_mapping(document, "<root>")

    _require_exact(
        document.get("schema_version"), EXPECTED_SCHEMA_VERSION, "schema_version"
    )
    _require_exact(document.get("kind"), EXPECTED_MANIFEST_KIND, "kind")
    bundle = _require_mapping(document.get("bundle"), "bundle")
    raw_bundle_root = bundle.get("output_root")
    if not isinstance(raw_bundle_root, str) or not raw_bundle_root.strip():
        raise ValueError("manifest field 'bundle.output_root' must be a path string")
    bundle_root = Path(raw_bundle_root).expanduser().resolve()
    if not bundle_root.is_dir():
        raise ValueError(f"missing declared bundle root: {bundle_root}")
    if manifest_path.parent != bundle_root:
        raise ValueError(
            "manifest must be the direct manifest file of its declared bundle root: "
            f"manifest={manifest_path}, bundle_root={bundle_root}"
        )

    selection = _require_mapping(document.get("selection"), "selection")
    for field, expected in (
        ("object_label", EXPECTED_OBJECT_LABEL),
        ("mh_frame_index", EXPECTED_MH_FRAME_INDEX),
        ("sh_frame_index", EXPECTED_SH_FRAME_INDEX),
        ("mh_role", EXPECTED_MH_ROLE),
        ("sh_role", EXPECTED_SH_ROLE),
    ):
        _require_exact(selection.get(field), expected, f"selection.{field}")

    camera_namespace = _require_mapping(
        document.get("camera_namespace"), "camera_namespace"
    )
    _require_exact(
        camera_namespace.get("primary_view"), "MH", "camera_namespace.primary_view"
    )
    _require_exact(
        camera_namespace.get("auxiliary_view"),
        "SH",
        "camera_namespace.auxiliary_view",
    )
    stereo_mapping = _require_mapping(
        camera_namespace.get("stereo_code_mapping"),
        "camera_namespace.stereo_code_mapping",
    )
    _require_exact(
        stereo_mapping.get("camera_1"),
        "SH",
        "camera_namespace.stereo_code_mapping.camera_1",
    )
    _require_exact(
        stereo_mapping.get("camera_2"),
        "MH",
        "camera_namespace.stereo_code_mapping.camera_2",
    )

    outputs = _require_mapping(document.get("outputs"), "outputs")
    output_image_record = _require_mapping(outputs.get("sh_image"), "outputs.sh_image")
    recorded_image_path = _record_path(
        output_image_record, "outputs.sh_image.path", bundle_root
    )
    if not _is_within(recorded_image_path, bundle_root):
        raise ValueError(
            "manifest outputs.sh_image.path must stay inside declared bundle root: "
            f"{recorded_image_path}"
        )
    if recorded_image_path != image_path:
        raise ValueError(
            "supplied SH image does not match manifest outputs.sh_image.path: "
            f"supplied={image_path}, recorded={recorded_image_path}"
        )
    recorded_bytes = output_image_record.get("bytes")
    if isinstance(recorded_bytes, bool) or not isinstance(recorded_bytes, int):
        raise ValueError("manifest outputs.sh_image.bytes must be an integer")
    actual_bytes = image_path.stat().st_size
    if recorded_bytes != actual_bytes:
        raise ValueError(
            "supplied SH image byte size does not match manifest outputs.sh_image: "
            f"actual={actual_bytes}, recorded={recorded_bytes}"
        )
    recorded_sha = output_image_record.get("sha256")
    if not isinstance(recorded_sha, str) or len(recorded_sha) != 64:
        raise ValueError("manifest outputs.sh_image.sha256 must be a SHA-256 hex digest")
    actual_sha = sha256_file(image_path)
    if recorded_sha.lower() != actual_sha:
        raise ValueError(
            "supplied SH image SHA-256 does not match manifest outputs.sh_image: "
            f"actual={actual_sha}, recorded={recorded_sha}"
        )

    # The source record carries the explicit view/frame namespace that the
    # copied output record intentionally omits.
    sources = _require_mapping(document.get("sources"), "sources")
    source_image_record = _require_mapping(sources.get("sh_image"), "sources.sh_image")
    source_image_path = _record_path(
        source_image_record, "sources.sh_image.path", bundle_root
    )
    for field, expected in (
        ("view", "SH"),
        ("pipeline_camera", "camera_1"),
        ("frame_index", EXPECTED_SH_FRAME_INDEX),
        ("bytes", actual_bytes),
        ("sha256", actual_sha),
    ):
        actual = source_image_record.get(field)
        if field == "sha256" and isinstance(actual, str):
            actual = actual.lower()
        _require_exact(actual, expected, f"sources.sh_image.{field}")
    if not source_image_path.is_file():
        raise ValueError(f"missing manifest sources.sh_image file: {source_image_path}")
    if source_image_path.stat().st_size != actual_bytes:
        raise ValueError("manifest sources.sh_image file no longer matches its byte record")
    if sha256_file(source_image_path) != actual_sha:
        raise ValueError("manifest sources.sh_image file no longer matches its SHA-256 record")

    image_geometry = _require_mapping(document.get("image_geometry"), "image_geometry")
    width = image_geometry.get("width")
    height = image_geometry.get("height")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (width, height)):
        raise ValueError("manifest image_geometry width/height must be integers")

    bound_record = {
        **copy.deepcopy(output_image_record),
        "path": str(image_path),
        "bytes": actual_bytes,
        "sha256": actual_sha,
        "view": "SH",
        "pipeline_camera": "camera_1",
        "frame_index": EXPECTED_SH_FRAME_INDEX,
    }
    return document, bundle_root, bound_record


def validate_publication_paths(
    *,
    bundle_root: Path,
    mask_path: Path,
    overlay_path: Path,
    image_path: Path,
    checkpoint_path: Path,
    sam2_root: Path,
    manifest_path: Path,
) -> tuple[Path, Path]:
    """Resolve safe final paths for the two generated PNG artifacts."""

    bundle_root = bundle_root.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    image_path = image_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    sam2_root = sam2_root.expanduser().resolve()

    resolved_outputs: list[Path] = []
    for raw_path, label in ((mask_path, "mask"), (overlay_path, "overlay")):
        expanded = raw_path.expanduser()
        if expanded.is_symlink():
            raise ValueError(f"{label} output must not be a symbolic link: {expanded}")
        if expanded.suffix.lower() != ".png":
            raise ValueError(f"{label} output must use a .png suffix: {expanded}")
        absolute = expanded if expanded.is_absolute() else bundle_root / expanded
        if absolute.is_symlink():
            raise ValueError(f"{label} output must not be a symbolic link: {absolute}")
        parent = absolute.parent.resolve()
        if not parent.is_dir():
            raise ValueError(f"{label} output parent must already exist: {parent}")
        resolved = (parent / absolute.name).resolve()
        if not _is_within(resolved, bundle_root):
            raise ValueError(
                f"{label} output must stay inside declared bundle root "
                f"{bundle_root}: {resolved}"
            )
        if resolved.exists() and resolved.is_dir():
            raise ValueError(f"{label} output is a directory: {resolved}")
        resolved_outputs.append(resolved)

    resolved_mask, resolved_overlay = resolved_outputs
    same_existing_file = (
        resolved_mask.exists()
        and resolved_overlay.exists()
        and os.path.samefile(resolved_mask, resolved_overlay)
    )
    if resolved_mask == resolved_overlay or same_existing_file:
        raise ValueError(
            f"mask and overlay outputs must be distinct: {resolved_mask}"
        )

    protected_files = {
        "SH input image": image_path,
        "SAM2 checkpoint": checkpoint_path,
        "manifest": manifest_path,
    }
    for label, output in (("mask", resolved_mask), ("overlay", resolved_overlay)):
        for protected_label, protected_path in protected_files.items():
            same_existing_file = (
                output.exists()
                and protected_path.exists()
                and os.path.samefile(output, protected_path)
            )
            if _paths_overlap(output, protected_path) or same_existing_file:
                raise ValueError(
                    f"{label} output overlaps protected {protected_label}: {output}"
                )
        if _paths_overlap(output, sam2_root):
            raise ValueError(
                f"{label} output overlaps protected SAM2 repository: {output}"
            )
    return resolved_mask, resolved_overlay


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, **metadata: Any) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **metadata,
    }


def validate_box(
    box: Sequence[int], *, width: int, height: int
) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError("box must contain x0 y0 x1 y1")
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in box):
        raise ValueError("box coordinates must be integers")
    x0, y0, x1, y1 = (int(value) for value in box)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(
            f"box {(x0, y0, x1, y1)} is outside image {width}x{height} "
            "or has non-positive area"
        )
    return x0, y0, x1, y1


def mask_metrics(mask: np.ndarray, box: Sequence[int] | None = None) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("SAM2 returned an empty mask")
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(component_areas.max()) if component_areas.size else 0
    pixels = int(mask.sum())
    result: dict[str, Any] = {
        "foreground_pixels": pixels,
        "foreground_fraction_of_image": float(pixels / mask.size),
        "bbox_convention": "xyxy_exclusive",
        "bbox_xyxy_exclusive": bbox,
        "connected_components": int(component_count - 1),
        "largest_component_pixels": largest,
        "largest_component_fraction": float(largest / pixels),
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
    }
    if box is not None:
        x0, y0, x1, y1 = validate_box(
            box, width=mask.shape[1], height=mask.shape[0]
        )
        inside = np.zeros_like(mask)
        inside[y0:y1, x0:x1] = True
        outside_pixels = int(np.count_nonzero(mask & ~inside))
        result.update(
            {
                "prompt_box_area_pixels": int((x1 - x0) * (y1 - y0)),
                "foreground_outside_prompt_box_pixels": outside_pixels,
                "foreground_outside_prompt_box_fraction": float(
                    outside_pixels / pixels
                ),
            }
        )
    return result


def select_mask_candidate(
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    candidate_index: int | None = None,
) -> tuple[np.ndarray, int, float]:
    masks = np.asarray(masks)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if masks.ndim != 3 or masks.shape[0] != scores.size or scores.size == 0:
        raise ValueError(
            f"expected masks (N,H,W) and scores (N,), got {masks.shape} and {scores.shape}"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("SAM2 candidate scores must be finite")
    if candidate_index is None:
        selected = int(np.argmax(scores))
    else:
        if isinstance(candidate_index, bool) or not isinstance(
            candidate_index, (int, np.integer)
        ):
            raise ValueError("candidate index must be an integer")
        selected = int(candidate_index)
        if not 0 <= selected < scores.size:
            raise ValueError(
                f"candidate index {selected} is outside [0, {scores.size})"
            )
    mask = np.asarray(masks[selected])
    if mask.dtype != np.bool_:
        mask = mask > 0
    else:
        mask = mask.copy()
    if not np.any(mask):
        raise ValueError(f"selected SAM2 candidate {selected} is empty")
    return mask, selected, float(scores[selected])


def make_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    prompt_box: Sequence[int],
    selected_index: int,
    selected_score: float,
) -> np.ndarray:
    overlay = image_bgr.copy()
    tint = np.empty_like(overlay)
    tint[:] = (40, 220, 40)
    foreground = np.asarray(mask, dtype=bool)
    overlay[foreground] = cv2.addWeighted(
        image_bgr[foreground], 0.42, tint[foreground], 0.58, 0
    )
    x0, y0, x1, y1 = (int(value) for value in prompt_box)
    cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 220, 255), 2)
    metrics = mask_metrics(foreground)
    bx0, by0, bx1, by1 = metrics["bbox_xyxy_exclusive"]
    cv2.rectangle(overlay, (bx0, by0), (bx1 - 1, by1 - 1), (255, 0, 255), 2)
    text = f"SAM2 inferred (not GT)  candidate={selected_index}  score={selected_score:.4f}"
    cv2.putText(
        overlay,
        text,
        (max(8, x0 - 4), max(24, y0 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        text,
        (max(8, x0 - 4), max(24, y0 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def write_png_atomic(path: Path, image: np.ndarray) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        if not cv2.imwrite(str(temporary), image):
            raise OSError(f"failed to encode PNG: {temporary}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def staged_file_record(
    staged_path: Path,
    published_path: Path,
    **metadata: Any,
) -> dict[str, Any]:
    """Describe staged bytes using the path they will have after commit."""

    staged_path = staged_path.expanduser().resolve()
    published_path = published_path.expanduser().resolve()
    if not staged_path.is_file():
        raise ValueError(f"missing staged file: {staged_path}")
    return {
        "path": str(published_path),
        "bytes": staged_path.stat().st_size,
        "sha256": sha256_file(staged_path),
        **metadata,
    }


def publish_staged_transaction(
    publications: Sequence[tuple[Path, Path]],
    *,
    replace_fn: Any = os.replace,
) -> None:
    """Publish staged files in order and restore every original on failure.

    The caller places the manifest last, so readers never observe a manifest
    pointing at only one new image.  Existing artifacts are moved into sibling
    backups first.  Any exception restores the complete pre-run state.

    ``replace_fn`` is injectable solely for deterministic rollback tests.
    """

    if not publications:
        raise ValueError("at least one staged publication is required")
    staged_paths = [Path(staged).resolve() for staged, _ in publications]
    final_paths = [Path(final).resolve() for _, final in publications]
    if len(set(staged_paths)) != len(staged_paths):
        raise ValueError("staged publication paths must be distinct")
    if len(set(final_paths)) != len(final_paths):
        raise ValueError("final publication paths must be distinct")
    stage_root = staged_paths[0].parent
    if any(path.parent != stage_root for path in staged_paths):
        raise ValueError("all staged files must share one staging directory")
    for path in staged_paths:
        if not path.is_file():
            raise ValueError(f"missing staged publication file: {path}")

    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for index, (staged_path, final_path) in enumerate(
            zip(staged_paths, final_paths)
        ):
            if final_path.exists():
                if final_path.is_dir():
                    raise ValueError(f"refusing to replace directory: {final_path}")
                backup = stage_root / f"backup-{index}-{final_path.name}"
                replace_fn(final_path, backup)
                backups[final_path] = backup
            replace_fn(staged_path, final_path)
            published.append(final_path)
    except Exception as publication_error:
        rollback_errors: list[str] = []
        for final_path in reversed(published):
            try:
                if final_path.exists() or final_path.is_symlink():
                    final_path.unlink()
            except Exception as exc:  # pragma: no cover - filesystem failure
                rollback_errors.append(f"remove {final_path}: {exc}")
        for final_path, backup in reversed(list(backups.items())):
            try:
                if final_path.exists() or final_path.is_symlink():
                    final_path.unlink()
                if backup.exists():
                    os.replace(backup, final_path)
            except Exception as exc:  # pragma: no cover - filesystem failure
                rollback_errors.append(f"restore {final_path}: {exc}")
        detail = (
            "; rollback errors: " + "; ".join(rollback_errors)
            if rollback_errors
            else ""
        )
        raise RuntimeError(
            f"artifact publication failed and was rolled back: "
            f"{publication_error}{detail}"
        ) from publication_error


def stage_and_publish_outputs(
    *,
    document: dict[str, Any],
    bundle_root: Path,
    manifest_path: Path,
    mask_path: Path,
    overlay_path: Path,
    mask_image: np.ndarray,
    overlay_image: np.ndarray,
    image_record: dict[str, Any],
    checkpoint_record: dict[str, Any],
    sam2_root: Path,
    config_name: str,
    prompt_box: Sequence[int],
    candidate_scores: Sequence[float],
    selected_index: int,
    selected_score: float,
    selection_policy: str,
    selected_metrics: dict[str, Any],
    candidate_metrics: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Stage mask, overlay, and manifest then commit them as one transaction."""

    bundle_root = bundle_root.resolve()
    manifest_path = manifest_path.resolve()
    mask_path = mask_path.resolve()
    overlay_path = overlay_path.resolve()
    stage_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{bundle_root.name}.sh-mask-stage-",
            dir=str(bundle_root.parent),
        )
    ).resolve()
    try:
        staged_mask = stage_dir / "mask.png"
        staged_overlay = stage_dir / "overlay.png"
        staged_manifest = stage_dir / "manifest.json"
        write_png_atomic(staged_mask, mask_image)
        write_png_atomic(staged_overlay, overlay_image)

        sh_frame = document["selection"]["sh_frame_index"]
        mask_record = staged_file_record(
            staged_mask,
            mask_path,
            view="SH",
            pipeline_camera="camera_1",
            frame_index=sh_frame,
            representation="binary PNG with values 0 and 255",
            provenance_class="model_inferred_not_ground_truth",
        )
        overlay_record = staged_file_record(
            staged_overlay,
            overlay_path,
            view="SH",
            frame_index=sh_frame,
            purpose=(
                "visual sanity check; green=mask, yellow=prompt, "
                "magenta=mask bbox"
            ),
        )
        updated = update_manifest_with_sh_mask(
            copy.deepcopy(document),
            image_record=image_record,
            mask_record=mask_record,
            overlay_record=overlay_record,
            checkpoint_record=checkpoint_record,
            sam2_root=sam2_root,
            config_name=config_name,
            prompt_box=prompt_box,
            candidate_scores=candidate_scores,
            selected_index=selected_index,
            selected_score=selected_score,
            selection_policy=selection_policy,
            selected_metrics=selected_metrics,
            candidate_metrics=candidate_metrics,
        )
        write_json_atomic(staged_manifest, updated)

        # Manifest publication is last.  If any replace fails, the helper
        # restores all pre-existing files before this staging directory is
        # removed.
        publish_staged_transaction(
            (
                (staged_mask, mask_path),
                (staged_overlay, overlay_path),
                (staged_manifest, manifest_path),
            )
        )
        return updated, mask_record, overlay_record
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def update_manifest_with_sh_mask(
    document: dict[str, Any],
    *,
    image_record: dict[str, Any],
    mask_record: dict[str, Any],
    overlay_record: dict[str, Any],
    checkpoint_record: dict[str, Any],
    sam2_root: Path,
    config_name: str,
    prompt_box: Sequence[int],
    candidate_scores: Sequence[float],
    selected_index: int,
    selected_score: float,
    selection_policy: str,
    selected_metrics: dict[str, Any],
    candidate_metrics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Return the manifest extension; callers perform the atomic write."""

    if not isinstance(document, dict):
        raise ValueError("manifest root must be an object")
    selection = document.get("selection")
    outputs = document.get("outputs")
    provenance = document.get("pixel_provenance")
    bundle = document.get("bundle")
    invariants = document.get("invariants")
    for name, value in (
        ("selection", selection),
        ("outputs", outputs),
        ("pixel_provenance", provenance),
        ("bundle", bundle),
        ("invariants", invariants),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"manifest field {name!r} must be an object")
    sh_frame = selection.get("sh_frame_index")
    if isinstance(sh_frame, bool) or not isinstance(sh_frame, int):
        raise ValueError("manifest selection.sh_frame_index must be an integer")
    if image_record.get("frame_index", sh_frame) != sh_frame:
        raise ValueError("SH image record frame index disagrees with manifest")
    if mask_record.get("frame_index") != sh_frame:
        raise ValueError("SH mask record frame index disagrees with manifest")

    outputs["sh_modal_mask"] = mask_record
    outputs["sh_modal_mask_overlay"] = overlay_record
    provenance["sh_modal_mask"] = {
        "provenance_class": "model_inferred_not_ground_truth",
        "is_model_inferred": True,
        "is_ground_truth": False,
        "is_human_annotated": False,
        "semantic_scope": "observed/modal Choco support in one synchronized SH frame",
        "source_image": image_record,
        "model": {
            "family": "Segment Anything 2 (SAM2)",
            "architecture": MODEL_NAME,
            "implementation_root": str(sam2_root.expanduser().resolve()),
            "config_name": config_name,
            "checkpoint": checkpoint_record,
        },
        "prompt": {
            "type": "explicit_box",
            "coordinate_system": "original_SH_image_pixels",
            "box_convention": "xyxy_exclusive",
            "box_xyxy_exclusive": [int(value) for value in prompt_box],
            "origin": "manually_selected_for_Choco_pilot",
        },
        "candidate_selection": {
            "policy": selection_policy,
            "candidate_scores": [float(value) for value in candidate_scores],
            "selected_candidate_index": int(selected_index),
            "selected_predicted_iou_score": float(selected_score),
            "candidate_metrics": list(candidate_metrics),
        },
        "selected_mask_metrics": selected_metrics,
        "inferred_foreground_pixels": int(selected_metrics["foreground_pixels"]),
        "statement": (
            "This binary SH modal mask is a SAM2 prediction from one RGB frame "
            "and an explicit box prompt. It is auxiliary model evidence, not "
            "ground truth, a human annotation, or an amodal reconstruction."
        ),
    }
    bundle["model_inference_performed"] = True
    invariants["model_inference_performed"] = True
    invariants["sh_modal_mask_is_model_inferred"] = True
    invariants["sh_modal_mask_is_ground_truth"] = False
    invariants["sh_modal_mask_binary_0_255"] = True
    return document


def run(
    *,
    manifest_path: Path,
    image_path: Path,
    mask_path: Path,
    overlay_path: Path,
    sam2_root: Path,
    checkpoint_path: Path,
    config_name: str,
    prompt_box: Sequence[int],
    device: str,
    candidate_index: int | None,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    image_path = image_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    sam2_root = sam2_root.expanduser().resolve()
    # Bind every immutable input and every publication path before importing
    # torch/SAM2 or loading the model checkpoint.
    document, bundle_root, image_record = load_and_bind_manifest(
        manifest_path, image_path
    )
    if not checkpoint_path.is_file():
        raise ValueError(f"missing SAM2 checkpoint: {checkpoint_path}")
    if not sam2_root.is_dir():
        raise ValueError(f"missing SAM2 implementation root: {sam2_root}")
    mask_path, overlay_path = validate_publication_paths(
        bundle_root=bundle_root,
        mask_path=mask_path,
        overlay_path=overlay_path,
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        sam2_root=sam2_root,
        manifest_path=manifest_path,
    )

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"failed to decode SH image: {image_path}")
    height, width = image_bgr.shape[:2]
    geometry = document["image_geometry"]
    expected_size = (int(geometry["width"]), int(geometry["height"]))
    if (width, height) != expected_size:
        raise ValueError(
            "decoded SH image dimensions do not match manifest image_geometry: "
            f"decoded={(width, height)}, recorded={expected_size}"
        )
    box = validate_box(prompt_box, width=width, height=height)
    checkpoint_record = file_record(checkpoint_path)

    sys.path.insert(0, str(sam2_root))
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    model = build_sam2(config_name, str(checkpoint_path), device=device)
    predictor = SAM2ImagePredictor(model)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(
            box=np.asarray(box, dtype=np.float32), multimask_output=True
        )
    selected_mask, selected_index, selected_score = select_mask_candidate(
        masks, scores, candidate_index=candidate_index
    )
    candidate_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    candidate_metrics = [
        mask_metrics(np.asarray(mask) > 0, box) for mask in np.asarray(masks)
    ]
    selected_metrics = mask_metrics(selected_mask, box)
    centroid_x, centroid_y = selected_metrics["centroid_xy"]
    if not (box[0] <= centroid_x < box[2] and box[1] <= centroid_y < box[3]):
        raise RuntimeError("selected mask centroid lies outside the Choco prompt box")
    if selected_metrics["largest_component_fraction"] < 0.90:
        raise RuntimeError(
            "selected mask is too fragmented for the focused Choco pilot: "
            f"largest component fraction={selected_metrics['largest_component_fraction']:.4f}"
        )

    mask_u8 = selected_mask.astype(np.uint8) * 255
    overlay = make_overlay(
        image_bgr,
        selected_mask,
        prompt_box=box,
        selected_index=selected_index,
        selected_score=selected_score,
    )
    updated, mask_record, overlay_record = stage_and_publish_outputs(
        document=document,
        bundle_root=bundle_root,
        manifest_path=manifest_path,
        mask_path=mask_path,
        overlay_path=overlay_path,
        mask_image=mask_u8,
        overlay_image=overlay,
        image_record=image_record,
        checkpoint_record=checkpoint_record,
        sam2_root=sam2_root,
        config_name=config_name,
        prompt_box=box,
        candidate_scores=candidate_scores.tolist(),
        selected_index=selected_index,
        selected_score=selected_score,
        selection_policy=(
            "maximum_predicted_iou_score"
            if candidate_index is None
            else "explicit_candidate_index_override"
        ),
        selected_metrics=selected_metrics,
        candidate_metrics=candidate_metrics,
    )
    return {
        "status": "ok",
        "manifest": str(manifest_path),
        "mask": mask_record,
        "overlay": overlay_record,
        "prompt_box_xyxy_exclusive": list(box),
        "selected_candidate_index": selected_index,
        "selected_predicted_iou_score": selected_score,
        "selected_mask_metrics": selected_metrics,
        "provenance_class": "model_inferred_not_ground_truth",
        "device": device,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    result.add_argument("--mask-output", type=Path, default=DEFAULT_MASK)
    result.add_argument("--overlay-output", type=Path, default=DEFAULT_OVERLAY)
    result.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM2_ROOT)
    result.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    result.add_argument("--config", default=DEFAULT_CONFIG)
    result.add_argument("--box", nargs=4, type=int, default=DEFAULT_BOX)
    result.add_argument("--device", default="auto")
    result.add_argument(
        "--candidate-index",
        type=int,
        help="Override the default maximum predicted-IoU candidate selection.",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = run(
        manifest_path=args.manifest,
        image_path=args.image,
        mask_path=args.mask_output,
        overlay_path=args.overlay_output,
        sam2_root=args.sam2_root,
        checkpoint_path=args.checkpoint,
        config_name=args.config,
        prompt_box=args.box,
        device=args.device,
        candidate_index=args.candidate_index,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
