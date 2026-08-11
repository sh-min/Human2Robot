"""Render a provenance-checked comparison of 2.5-D and mesh-volume barriers.

The synchronized panels are::

    HaCo 2.5-D shell + temporal | nominal mesh front only
    nominal mesh volume + shell | nominal mesh volume + shell + temporal

This utility does not estimate geometry or change robot/object poses.  It
validates the independently generated mesh-volume and compositor artifacts,
renders full-frame and shared dynamic-ROI H.264 grids, and publishes the two
videos and their provenance report transactionally.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import itertools
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from atomic_directory_publish import publish_directory
from compare_xhand_object_barriers import (
    dynamic_roi_centers,
    render_dynamic_roi_grid,
)
from make_video_comparison_grid import (
    GridLayout,
    NamedVideo,
    VideoMetadata,
    probe_video,
    render_comparison_grid_layout,
)


MESH_METHOD = "nominal_mjcf_mesh_volume_wrist_relative_fit"
MODE_SPECS = (
    ("haco_2p5d", "1 HaCo 2.5D: shell + temporal"),
    ("mesh_front", "2 Nominal mesh: front only"),
    ("mesh_volume_shell", "3 Mesh volume + XHand shell"),
    (
        "mesh_volume_shell_temporal",
        "4 Mesh volume + shell + temporal",
    ),
)
GRID = GridLayout(columns=2, rows=2)
FULL_VIDEO_NAME = "video_compare_object_mesh_volume_2x2.mp4"
ROI_VIDEO_NAME = "video_compare_object_mesh_volume_roi_2x2.mp4"
REPORT_NAME = "comparison_report.json"
MESH_REQUIRED_FILES = (
    "object_mesh_front_depth.npy",
    "object_mesh_back_depth.npy",
    "object_mesh_mask.npy",
    "object_pose_cam.npy",
    "pose_valid.npy",
    "pose_confidence.npy",
    "report.json",
)


def _require_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(resolved)
    return resolved


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(_require_file(path).read_text())
    if not isinstance(value, dict):
        raise TypeError(f"JSON report must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _require_file(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, hash_file: bool = True) -> dict[str, object]:
    resolved = _require_file(path)
    record: dict[str, object] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
    }
    if hash_file:
        record["sha256"] = _sha256(resolved)
    return record


def _output_record(path: Path) -> dict[str, object]:
    """Record a staged output without persisting its temporary directory."""

    resolved = _require_file(path)
    return {
        "filename": resolved.name,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _metadata_signature(value: VideoMetadata) -> tuple[object, ...]:
    return (
        value.width,
        value.height,
        value.frame_count,
        round(float(value.fps), 9),
        round(float(value.duration_s), 6),
    )


def _metadata_dict(value: VideoMetadata) -> dict[str, object]:
    return {
        "width": value.width,
        "height": value.height,
        "frames": value.frame_count,
        "fps": float(value.fps),
        "duration_s": float(value.duration_s),
        "codec": value.codec_name,
        "pixel_format": value.pixel_format,
    }


def _validate_h264(value: VideoMetadata, *, name: str) -> None:
    if value.codec_name != "h264" or value.pixel_format != "yuv420p":
        raise RuntimeError(
            f"{name} must be H.264/yuv420p, got "
            f"{value.codec_name}/{value.pixel_format}"
        )


def validate_mesh_volume_arrays(
    mesh_dir: Path,
    *,
    expected_shape: tuple[int, int, int] | None = None,
) -> dict[str, object]:
    """Validate the persisted front/back raster and pose-valid contract.

    Camera-Z is positive forward.  On every valid volume pixel, the back hit
    must therefore be no nearer than the front hit.  Invalid poses must not
    publish a mesh mask or either depth surface.
    """

    root = mesh_dir.expanduser().resolve()
    paths = {name: _require_file(root / name) for name in MESH_REQUIRED_FILES}
    front = np.load(paths["object_mesh_front_depth.npy"], mmap_mode="r")
    back = np.load(paths["object_mesh_back_depth.npy"], mmap_mode="r")
    mask = np.load(paths["object_mesh_mask.npy"], mmap_mode="r")
    pose = np.load(paths["object_pose_cam.npy"], mmap_mode="r")
    pose_valid = np.load(paths["pose_valid.npy"], mmap_mode="r")
    pose_confidence = np.load(paths["pose_confidence.npy"], mmap_mode="r")
    if front.ndim != 3 or front.dtype not in (np.float16, np.float32):
        raise ValueError("mesh front depth must be float16/float32 (T,H,W)")
    if back.shape != front.shape or back.dtype not in (np.float16, np.float32):
        raise ValueError("mesh back depth must match front depth shape/type family")
    if mask.shape != front.shape or mask.dtype != np.bool_:
        raise ValueError("object mesh mask must be bool and match depth shape")
    if pose_valid.shape != (len(front),) or pose_valid.dtype != np.bool_:
        raise ValueError("pose_valid must be bool (T,)")
    if pose.shape != (len(front), 4, 4) or pose.dtype != np.float32:
        raise ValueError("object_pose_cam must be float32 (T,4,4)")
    if (
        pose_confidence.shape != (len(front),)
        or pose_confidence.dtype != np.float32
        or not np.isfinite(pose_confidence).all()
        or np.any(pose_confidence < 0.0)
        or np.any(pose_confidence > 1.0)
    ):
        raise ValueError("pose_confidence must be finite float32 (T,) in [0,1]")
    if expected_shape is not None and front.shape != expected_shape:
        raise ValueError(
            f"mesh volume shape {front.shape} != expected {expected_shape}"
        )

    mask_pixels = 0
    valid_volume_pixels = 0
    front_without_mask = 0
    back_without_mask = 0
    invalid_pose_pixels = 0
    inverted_pixels = 0
    nonfinite_pixels = 0
    mask_without_depth = 0
    invalid_pose_state_frames = 0
    for frame_index in range(len(front)):
        front_frame = np.asarray(front[frame_index], dtype=np.float32)
        back_frame = np.asarray(back[frame_index], dtype=np.float32)
        mask_frame = np.asarray(mask[frame_index], dtype=bool)
        front_positive = np.isfinite(front_frame) & (front_frame > 0.0)
        back_positive = np.isfinite(back_frame) & (back_frame > 0.0)
        volume = mask_frame & front_positive & back_positive
        mask_pixels += int(mask_frame.sum())
        valid_volume_pixels += int(volume.sum())
        front_without_mask += int(np.sum(front_positive & ~mask_frame))
        back_without_mask += int(np.sum(back_positive & ~mask_frame))
        nonfinite_pixels += int(
            np.sum(mask_frame & (~np.isfinite(front_frame) | ~np.isfinite(back_frame)))
        )
        mask_without_depth += int(
            np.sum(mask_frame & ~(front_positive & back_positive))
        )
        inverted_pixels += int(
            np.sum(volume & (back_frame + 1.0e-4 < front_frame))
        )
        if not bool(pose_valid[frame_index]):
            invalid_pose_pixels += int(
                mask_frame.sum() + front_positive.sum() + back_positive.sum()
            )
            invalid_pose_state_frames += int(
                not np.isnan(np.asarray(pose[frame_index])).all()
            )
        else:
            pose_frame = np.asarray(pose[frame_index], dtype=np.float64)
            valid_transform = (
                np.isfinite(pose_frame).all()
                and np.allclose(
                    pose_frame[3],
                    np.asarray((0.0, 0.0, 0.0, 1.0)),
                    atol=1.0e-5,
                )
                and np.allclose(
                    pose_frame[:3, :3].T @ pose_frame[:3, :3],
                    np.eye(3),
                    atol=2.0e-3,
                )
                and abs(np.linalg.det(pose_frame[:3, :3]) - 1.0) <= 2.0e-3
            )
            if not valid_transform:
                raise ValueError(
                    f"valid object pose is not a finite rigid transform at "
                    f"frame {frame_index}"
                )
    if front_without_mask or back_without_mask:
        raise ValueError("positive mesh depth exists outside object_mesh_mask")
    if nonfinite_pixels:
        raise ValueError("mesh mask contains non-finite front/back depth")
    if mask_without_depth:
        raise ValueError("mesh mask contains missing/non-positive front/back depth")
    if inverted_pixels:
        raise ValueError("mesh back camera-Z is in front of mesh front camera-Z")
    if invalid_pose_pixels:
        raise ValueError("invalid object poses published mesh pixels/depth")
    if invalid_pose_state_frames:
        raise ValueError("invalid object poses must use an all-NaN pose sentinel")
    if int(pose_valid.sum()) <= 0 or valid_volume_pixels <= 0:
        raise ValueError("mesh builder published no valid fitted object volume")

    report = _load_json(paths["report.json"])
    if report.get("method") != MESH_METHOD:
        raise ValueError(
            f"mesh builder method {report.get('method')!r} != {MESH_METHOD!r}"
        )
    if report.get("representation") != (
        "fitted_watertight_nominal_mesh_front_back_camera_z"
    ):
        raise ValueError("mesh builder representation is not front/back camera-Z")
    if report.get("metric_collision_guarantee") is not False:
        raise ValueError("nominal mesh report overstates metric collision provenance")
    if report.get("rear_surface_measured") is not False:
        raise ValueError("nominal mesh report overstates rear-surface measurement")
    if report.get("pose_state_modified") is not False:
        raise ValueError("mesh builder unexpectedly reports trajectory modification")
    report_frames = int(report.get("frames", len(front)))
    if report_frames != len(front):
        raise ValueError("mesh builder report/array frame count mismatch")
    counts = report.get("counts", {})
    if not isinstance(counts, Mapping):
        raise TypeError("mesh builder counts must be a mapping")
    if int(counts.get("valid_pose_frames", -1)) != int(pose_valid.sum()):
        raise ValueError("mesh builder report/pose-valid count mismatch")
    if int(counts.get("mesh_pixels", -1)) != mask_pixels:
        raise ValueError("mesh builder report/mask pixel count mismatch")
    invariants = report.get("invariants", {})
    if not isinstance(invariants, Mapping) or not all(
        invariants.get(key) is True
        for key in (
            "canonical_meshes_watertight",
            "invalid_pose_frames_have_empty_geometry",
            "valid_mesh_pixels_have_ordered_front_back",
            "mesh_mask_equals_positive_front_and_back",
            "robot_trajectory_arrays_unchanged",
        )
    ):
        raise ValueError("mesh builder report is missing required invariants")

    return {
        "root": root,
        "paths": paths,
        "report": report,
        "shape": front.shape,
        "pose_valid_frames": int(pose_valid.sum()),
        "pose_invalid_frames": int(len(pose_valid) - pose_valid.sum()),
        "mesh_mask_pixels": mask_pixels,
        "valid_front_back_pixels": valid_volume_pixels,
        "front_without_mask_pixels": front_without_mask,
        "back_without_mask_pixels": back_without_mask,
        "inverted_front_back_pixels": inverted_pixels,
        "invalid_pose_published_pixels": invalid_pose_pixels,
        "invalid_pose_state_frames": invalid_pose_state_frames,
    }


def _load_source(directory: Path, mode: str) -> dict[str, object]:
    root = directory.expanduser().resolve()
    if mode == "haco_2p5d":
        video = _require_file(root / "video_overlay_hand_barrier.mp4")
    else:
        video = _require_file(root / "video_overlay_mesh_volume.mp4")
    mask_path = _require_file(root / "occluded_hand_mask.npy")
    report_path = _require_file(root / "report.json")
    report = _load_json(report_path)
    expected_method = (
        "visual_camera_z_xhand_barrier"
        if mode == "haco_2p5d"
        else "visual_xhand_mesh_volume_barrier"
    )
    if report.get("method") != expected_method:
        raise ValueError(
            f"{mode} method {report.get('method')!r} != {expected_method!r}"
        )
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    if mask.ndim != 3 or mask.dtype != np.bool_:
        raise ValueError(f"{mode} occlusion mask must be bool (T,H,W)")
    metadata = probe_video(video)
    if mask.shape != (metadata.frame_count, metadata.height, metadata.width):
        raise ValueError(f"{mode} mask/video shape mismatch")
    if int(report.get("frames", -1)) != len(mask):
        raise ValueError(f"{mode} report/mask frame mismatch")
    counts = report.get("counts", {})
    if not isinstance(counts, dict):
        raise TypeError(f"{mode} counts must be a dictionary")
    reported_pixels = int(counts.get("final_occluded_pixels", -1))
    if reported_pixels != int(mask.sum()):
        raise ValueError(f"{mode} report/mask occluded-pixel mismatch")
    if int(counts.get("residual_violation_pixels", -1)) != 0:
        raise ValueError(f"{mode} left residual camera-Z violations")
    if report.get("pose_state_modified") is not False:
        raise ValueError(f"{mode} unexpectedly changed the robot trajectory")
    if report.get("metric_collision_guarantee") is not False:
        raise ValueError(f"{mode} overstates physical collision provenance")
    invariants = report.get("invariants", {})
    if not isinstance(invariants, dict):
        raise TypeError(f"{mode} invariants must be a dictionary")
    residual_invariant = (
        "valid_surface_barrier_residual_is_zero"
        if mode == "haco_2p5d"
        else "valid_volume_barrier_residual_is_zero"
    )
    if invariants.get(residual_invariant) is not True:
        raise ValueError(f"{mode} zero-residual invariant is missing")
    return {
        "root": root,
        "video": video,
        "mask_path": mask_path,
        "report_path": report_path,
        "report": report,
        "mask": mask,
        "metadata": metadata,
    }


def _shell_controls(report: Mapping[str, object]) -> tuple[float, float, float, int]:
    config = report.get("config", {})
    if not isinstance(config, Mapping):
        raise TypeError("barrier config must be a mapping")
    return (
        float(config.get("thumb_shell_m", -1.0)),
        float(config.get("finger_shell_m", -1.0)),
        float(config.get("palm_shell_m", -1.0)),
        int(config.get("temporal_max_gap_frames", -1)),
    )


def validate_controls(sources: Mapping[str, Mapping[str, object]]) -> None:
    """Verify that the panel labels match the actual compositor controls."""

    expected = {
        "haco_2p5d": (0.01958, 0.01465, 0.015, 2),
        "mesh_front": (0.0, 0.0, 0.0, 0),
        "mesh_volume_shell": (0.01958, 0.01465, 0.015, 0),
        "mesh_volume_shell_temporal": (0.01958, 0.01465, 0.015, 2),
    }
    for mode, wanted in expected.items():
        report = sources[mode]["report"]
        if not isinstance(report, Mapping):
            raise TypeError(f"{mode} report must be a mapping")
        actual = _shell_controls(report)
        if not np.allclose(actual[:3], wanted[:3], rtol=0.0, atol=1.0e-9):
            raise ValueError(f"{mode} shell controls {actual[:3]} != {wanted[:3]}")
        if actual[3] != wanted[3]:
            raise ValueError(f"{mode} temporal control {actual[3]} != {wanted[3]}")
        if mode != "haco_2p5d":
            config = report.get("config", {})
            assert isinstance(config, Mapping)
            wanted_mode = "front" if mode == "mesh_front" else "volume"
            actual_mode = report.get("mode", config.get("mode"))
            if actual_mode != wanted_mode:
                raise ValueError(
                    f"{mode} compositor mode {actual_mode!r} != {wanted_mode!r}"
                )


def validate_controlled_inputs(
    sources: Mapping[str, Mapping[str, object]],
    *,
    mesh_dir: Path,
) -> dict[str, Path]:
    """Require all panels to differ only in their declared barrier strategy."""

    haco_report = sources["haco_2p5d"].get("report", {})
    if not isinstance(haco_report, Mapping):
        raise TypeError("HaCo report must be a mapping")
    haco_inputs = haco_report.get("sources", {})
    if not isinstance(haco_inputs, Mapping):
        raise TypeError("HaCo report sources must be a mapping")

    def source_path(values: Mapping[str, object], key: str) -> Path:
        value = values.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"missing controlled source {key!r}")
        return Path(value).expanduser().resolve()

    controlled = {
        "background": source_path(haco_inputs, "background"),
        "raw_video": source_path(haco_inputs, "raw_video"),
        "overlay_dir": source_path(haco_inputs, "overlay_dir"),
        "object_support_mask": source_path(haco_inputs, "object_mask"),
        "object_restore_mask": source_path(
            haco_inputs,
            "object_restore_mask",
        ),
        "baseline_mask": source_path(haco_inputs, "baseline_mask"),
    }
    expected_mesh_dir = mesh_dir.expanduser().resolve()
    for mode in (
        "mesh_front",
        "mesh_volume_shell",
        "mesh_volume_shell_temporal",
    ):
        report = sources[mode].get("report", {})
        if not isinstance(report, Mapping):
            raise TypeError(f"{mode} report must be a mapping")
        inputs = report.get("sources", {})
        if not isinstance(inputs, Mapping):
            raise TypeError(f"{mode} report sources must be a mapping")
        for key, expected in controlled.items():
            actual = source_path(inputs, key)
            if actual != expected:
                raise ValueError(
                    f"{mode} controlled source {key} differs: "
                    f"{actual} != {expected}"
                )
        actual_mesh_dir = source_path(inputs, "mesh_dir")
        if actual_mesh_dir != expected_mesh_dir:
            raise ValueError(
                f"{mode} mesh source {actual_mesh_dir} != {expected_mesh_dir}"
            )
    return controlled


def validate_builder_provenance(
    report: Mapping[str, object],
    *,
    mapping: Path,
    wrist_npz: Path,
    amodal_mask: Path,
    completed_front_depth: Path,
    debug_video: Path,
) -> dict[str, Path]:
    """Bind the fitted mesh report to this comparison's exact inputs."""

    raw_sources = report.get("sources", {})
    if not isinstance(raw_sources, Mapping):
        raise TypeError("mesh builder sources must be a mapping")

    def reported_path(key: str) -> Path:
        value = raw_sources.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"mesh builder source {key!r} is missing")
        path = Path(value).expanduser().resolve()
        _require_file(path)
        return path

    actual = {
        key: reported_path(key)
        for key in (
            "mapping",
            "labels_json",
            "amodal_mask",
            "completed_front_depth",
            "wrist_npz",
            "debug_video",
        )
    }
    expected = {
        "mapping": mapping.expanduser().resolve(),
        "amodal_mask": amodal_mask.expanduser().resolve(),
        "completed_front_depth": completed_front_depth.expanduser().resolve(),
        "wrist_npz": wrist_npz.expanduser().resolve(),
        "debug_video": debug_video.expanduser().resolve(),
    }
    for key, expected_path in expected.items():
        if actual[key] != expected_path:
            raise ValueError(
                f"mesh builder source {key} differs: "
                f"{actual[key]} != {expected_path}"
            )
    return actual


def validate_progressive_mesh_masks(masks: Mapping[str, np.ndarray]) -> None:
    """Require monotone protection within the three nominal-mesh panels."""

    required = {
        "mesh_front",
        "mesh_volume_shell",
        "mesh_volume_shell_temporal",
    }
    if set(masks) != required:
        raise ValueError("progressive mesh masks are incomplete")
    shapes = {mask.shape for mask in masks.values()}
    if len(shapes) != 1:
        raise ValueError("progressive mesh mask shapes differ")
    for smaller_name, larger_name in (
        ("mesh_front", "mesh_volume_shell"),
        ("mesh_volume_shell", "mesh_volume_shell_temporal"),
    ):
        smaller = masks[smaller_name]
        larger = masks[larger_name]
        for frame_index in range(len(smaller)):
            violation = np.asarray(smaller[frame_index]) & ~np.asarray(
                larger[frame_index]
            )
            if np.any(violation):
                raise ValueError(
                    f"{smaller_name} is not a subset of {larger_name} at "
                    f"frame {frame_index}: {int(violation.sum())} pixels"
                )


def _statistics(sources: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    masks = {
        mode: source["mask"]
        for mode, source in sources.items()
        if isinstance(source.get("mask"), np.ndarray)
    }
    modes = {
        mode: {
            "pixels": int(mask.sum()),
            "frames": int(
                sum(bool(np.any(mask[index])) for index in range(len(mask)))
            ),
        }
        for mode, mask in masks.items()
    }
    comparisons: dict[str, dict[str, int]] = {}
    order = [mode for mode, _label in MODE_SPECS]
    for left_name, right_name in itertools.combinations(order, 2):
        left, right = masks[left_name], masks[right_name]
        added = removed = changed_frames = 0
        for frame_index in range(len(left)):
            left_frame = np.asarray(left[frame_index], dtype=bool)
            right_frame = np.asarray(right[frame_index], dtype=bool)
            frame_added = int(np.sum(right_frame & ~left_frame))
            frame_removed = int(np.sum(left_frame & ~right_frame))
            added += frame_added
            removed += frame_removed
            changed_frames += int(frame_added > 0 or frame_removed > 0)
        comparisons[f"{left_name}_vs_{right_name}"] = {
            "added_pixels": added,
            "removed_pixels": removed,
            "changed_frames": changed_frames,
        }
    return {"modes": modes, "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--haco_2p5d_dir", type=Path, required=True)
    parser.add_argument("--mesh_front_dir", type=Path, required=True)
    parser.add_argument("--mesh_volume_shell_dir", type=Path, required=True)
    parser.add_argument(
        "--mesh_volume_shell_temporal_dir", type=Path, required=True
    )
    parser.add_argument("--mesh_dir", type=Path, required=True)
    parser.add_argument("--object_mask", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--overlay_input", type=Path, required=True)
    parser.add_argument("--joint_names", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    directories = {
        "haco_2p5d": args.haco_2p5d_dir,
        "mesh_front": args.mesh_front_dir,
        "mesh_volume_shell": args.mesh_volume_shell_dir,
        "mesh_volume_shell_temporal": args.mesh_volume_shell_temporal_dir,
    }
    sources = {
        mode: _load_source(directory, mode)
        for mode, directory in directories.items()
    }
    reference_metadata = sources["haco_2p5d"]["metadata"]
    reference_mask = sources["haco_2p5d"]["mask"]
    assert isinstance(reference_metadata, VideoMetadata)
    assert isinstance(reference_mask, np.ndarray)
    expected_shape = (
        reference_metadata.frame_count,
        reference_metadata.height,
        reference_metadata.width,
    )
    for mode, source in sources.items():
        metadata = source["metadata"]
        mask = source["mask"]
        assert isinstance(metadata, VideoMetadata)
        assert isinstance(mask, np.ndarray)
        if _metadata_signature(metadata) != _metadata_signature(reference_metadata):
            raise ValueError(f"{mode} video metadata differs")
        if mask.shape != expected_shape:
            raise ValueError(f"{mode} mask shape differs")

    validate_controls(sources)
    controlled_inputs = validate_controlled_inputs(
        sources,
        mesh_dir=args.mesh_dir,
    )
    validate_progressive_mesh_masks(
        {
            mode: sources[mode]["mask"]
            for mode in (
                "mesh_front",
                "mesh_volume_shell",
                "mesh_volume_shell_temporal",
            )
            if isinstance(sources[mode]["mask"], np.ndarray)
        }
    )
    mesh_validation = validate_mesh_volume_arrays(
        args.mesh_dir,
        expected_shape=expected_shape,
    )

    object_mask_path = _require_file(args.object_mask)
    object_mask = np.load(object_mask_path, mmap_mode="r", allow_pickle=False)
    if object_mask.shape != expected_shape or object_mask.dtype != np.bool_:
        raise ValueError(f"dynamic ROI object mask must be bool {expected_shape}")

    mapping = _require_file(args.mapping)
    overlay_input = _require_file(args.overlay_input)
    joint_names = _require_file(args.joint_names)
    haco_report = sources["haco_2p5d"]["report"]
    assert isinstance(haco_report, Mapping)
    haco_report_sources = haco_report.get("sources", {})
    if not isinstance(haco_report_sources, Mapping):
        raise TypeError("HaCo report sources must be a mapping")
    completed_front_value = haco_report_sources.get("object_surface_depth")
    if not isinstance(completed_front_value, str):
        raise ValueError("HaCo report is missing object_surface_depth")
    mesh_report = mesh_validation["report"]
    assert isinstance(mesh_report, dict)
    builder_inputs = validate_builder_provenance(
        mesh_report,
        mapping=mapping,
        wrist_npz=overlay_input,
        amodal_mask=object_mask_path,
        completed_front_depth=Path(completed_front_value),
        debug_video=controlled_inputs["raw_video"],
    )
    videos = [
        NamedVideo(label=label, path=Path(sources[mode]["video"]))
        for mode, label in MODE_SPECS
    ]
    output_dir = args.out_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".object_mesh_volume_compare.", dir=output_dir.parent)
    )
    atexit.register(shutil.rmtree, staging, ignore_errors=True)

    full_metadata = render_comparison_grid_layout(
        videos,
        staging / FULL_VIDEO_NAME,
        layout=GRID,
        overwrite=False,
        crf=18,
        preset="medium",
    )
    centres = dynamic_roi_centers(
        object_mask,
        crop_width=640,
        crop_height=360,
        smooth_window=9,
    )
    roi_metadata = render_dynamic_roi_grid(
        videos,
        staging / ROI_VIDEO_NAME,
        centres=centres,
        crop_width=640,
        crop_height=360,
        fps=float(reference_metadata.fps),
    )
    _validate_h264(full_metadata, name="full-frame comparison")
    _validate_h264(roi_metadata, name="dynamic-ROI comparison")

    statistics = _statistics(sources)
    source_records = {
        mode: {
            "directory": str(source["root"]),
            "video": _file_record(Path(source["video"])),
            "mask": _file_record(Path(source["mask_path"]), hash_file=False),
            "report": _file_record(Path(source["report_path"])),
            "method": source["report"].get("method"),
        }
        for mode, source in sources.items()
    }
    mesh_paths = mesh_validation["paths"]
    assert isinstance(mesh_paths, dict)
    mesh_artifacts = {
        name: _file_record(path, hash_file=name in {"pose_valid.npy", "report.json"})
        for name, path in mesh_paths.items()
        if isinstance(path, Path)
    }
    report = {
        "schema_version": 1,
        "comparison": "HaCo 2.5-D versus fitted nominal object mesh volume",
        "layout": [
            [MODE_SPECS[0][0], MODE_SPECS[1][0]],
            [MODE_SPECS[2][0], MODE_SPECS[3][0]],
        ],
        "definitions": {
            "common_baseline": (
                "all four panels retain the same dual-HaCo finger-only "
                "occlusion mask and completed-object background"
            ),
            "haco_2p5d": (
                "existing dual-HaCo inferred front camera-Z with XHand "
                "half-thickness shell and temporal <=2f"
            ),
            "mesh_front": (
                "fitted nominal watertight mesh front camera-Z only; "
                "zero XHand shell and no temporal closing"
            ),
            "mesh_volume_shell": (
                "fitted nominal mesh front/back camera-Z classification; "
                "both intersecting and fully-behind states are hidden, with "
                "the XHand half-thickness shell"
            ),
            "mesh_volume_shell_temporal": (
                "same nominal mesh volume and shell plus temporal <=2f"
            ),
        },
        "physical_collision_solver": False,
        "pose_state_modified": False,
        "metric_collision_guarantee": False,
        "interpretation_note": (
            "For a watertight ray interval, the zero-shell hidden union of "
            "intersecting and fully-behind states equals the front-depth "
            "gate. The back surface distinguishes penetration from fully "
            "behind for evidence/debugging; the panel-2 to panel-3 visual "
            "difference also includes the declared XHand shell."
        ),
        "videos": {
            "full_frame": _output_record(staging / FULL_VIDEO_NAME),
            "dynamic_roi": _output_record(staging / ROI_VIDEO_NAME),
        },
        "metadata": {
            "source": _metadata_dict(reference_metadata),
            "full_frame": _metadata_dict(full_metadata),
            "dynamic_roi": {
                **_metadata_dict(roi_metadata),
                "source_crop": [640, 360],
                "center_policy": (
                    "amodal object bbox interpolation plus 9-frame moving average"
                ),
            },
        },
        "sources": source_records,
        "controlled_inputs": {
            key: (
                {"path": str(path)}
                if key == "overlay_dir"
                else _file_record(
                    path,
                    hash_file=key == "background",
                )
            )
            for key, path in controlled_inputs.items()
        },
        "mesh_builder": {
            "directory": str(mesh_validation["root"]),
            "method": mesh_report.get("method"),
            "representation": mesh_report.get("representation"),
            "mapping": _file_record(mapping),
            "inputs": {
                key: _file_record(
                    path,
                    hash_file=key in {"mapping", "labels_json", "wrist_npz"},
                )
                for key, path in builder_inputs.items()
            },
            "artifacts": mesh_artifacts,
            "validation": {
                key: value
                for key, value in mesh_validation.items()
                if key not in {"root", "paths", "report", "shape"}
            },
            "shape": list(mesh_validation["shape"]),
        },
        "trajectory_provenance": {
            "overlay_input": _file_record(overlay_input),
            "joint_names": _file_record(joint_names),
            "unchanged_for_all_panels": True,
        },
        "invariants": {
            "all_video_metadata_aligned": True,
            "all_panels_share_background_raw_overlay_masks_and_baseline": True,
            "mesh_front_subset_volume_shell": True,
            "mesh_volume_shell_subset_temporal": True,
            "mesh_back_not_in_front_of_mesh_front": True,
            "zero_shell_front_gate_equals_volume_hidden_union": True,
            "invalid_pose_frames_publish_no_mesh_pixels": True,
            "all_barrier_residual_violations_zero": True,
            "robot_trajectory_unchanged": True,
        },
        **statistics,
        "provenance_warning": (
            "The front/back surfaces come from object-specific watertight "
            "nominal or scan-regularized priors fitted with approximate phone "
            "intrinsics, HaWoR/Depth-Anything camera-Z, masks, and wrist motion. "
            "Unseen surfaces and object poses are inferred, not measured. "
            "This is a visual camera-Z exclusion volume, not calibrated metric "
            "reconstruction, a physical collision solver, or proof of contact."
        ),
    }
    (staging / REPORT_NAME).write_text(json.dumps(report, indent=2) + "\n")
    publish_directory(str(staging), str(output_dir))
    print(f"[ok] object mesh-volume comparison: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
