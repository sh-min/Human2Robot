"""Composite XHand with a pose-fitted object-mesh volume barrier.

The mesh builder is authoritative for the MH-camera front/back depth pair.
This stage never changes robot or object pose.  It classifies every valid
XHand/object pixel as front-of, intersecting, or fully-behind, then hides the
intersecting/behind pixels (or applies the equivalent zero-shell front gate in
``front`` mode).  A pre-existing finger-only mask is always retained.

This remains a visual camera-ray constraint, not a mesh-mesh collision solve:
the rendered XHand contributes its visible front depth plus a configured
part-wise camera-Z shell.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from atomic_directory_publish import publish_directory
from composite_rb5_contact_occlusion import (
    bridge_short_occlusion_gaps,
    composite_frame,
)
from composite_xhand_object_barrier import (
    PART_NAMES,
    _open_writer,
    _video_metadata,
    resize_overlay_frame,
    restore_raw_object_pixels,
    semantic_hand_labels,
    thickness_map,
    validate_mask_volume,
)


BUILDER_METHOD = "nominal_mjcf_mesh_volume_wrist_relative_fit"
BUILDER_REPRESENTATION = "fitted_watertight_nominal_mesh_front_back_camera_z"
METHOD = "visual_xhand_mesh_volume_barrier"
FRONT_DEPTH_NAME = "object_mesh_front_depth.npy"
BACK_DEPTH_NAME = "object_mesh_back_depth.npy"
MESH_MASK_NAME = "object_mesh_mask.npy"
POSE_VALID_NAME = "pose_valid.npy"
BUILDER_REPORT_NAME = "report.json"

CLASS_INVALID = np.uint8(0)
CLASS_FRONT_OF = np.uint8(1)
CLASS_INTERSECTING = np.uint8(2)
CLASS_FULLY_BEHIND = np.uint8(3)
CLASS_NAMES = ("invalid", "front_of", "intersecting", "fully_behind")


def validate_pose_valid(pose_valid: np.ndarray, frame_count: int) -> None:
    """Validate the strict per-frame mesh-pose contract."""
    if pose_valid.shape != (frame_count,):
        raise ValueError(
            f"pose_valid must have shape ({frame_count},), got "
            f"{pose_valid.shape}"
        )
    if pose_valid.dtype != np.bool_:
        raise TypeError(f"pose_valid must have dtype bool, got {pose_valid.dtype}")


def validate_depth_volume(
    front_depth: np.ndarray,
    back_depth: np.ndarray,
    mesh_mask: np.ndarray,
    pose_valid: np.ndarray,
    *,
    expected_shape: tuple[int, int, int],
    minimum_depth_m: float = 0.02,
    maximum_depth_m: float = 5.0,
) -> dict[str, int]:
    """Validate front/back camera-Z arrays and return compact diagnostics.

    Invalid-pose frames are allowed to retain builder diagnostics, but every
    masked pixel in a pose-valid frame must have a finite, ordered metric depth
    interval.  No interpolation or dtype conversion is performed here.
    """
    if front_depth.shape != expected_shape or back_depth.shape != expected_shape:
        raise ValueError("mesh front/back depth must have video shape (T,H,W)")
    if mesh_mask.shape != expected_shape:
        raise ValueError("object mesh mask must have video shape (T,H,W)")
    if not np.issubdtype(front_depth.dtype, np.floating):
        raise TypeError("mesh front depth must have a floating dtype")
    if not np.issubdtype(back_depth.dtype, np.floating):
        raise TypeError("mesh back depth must have a floating dtype")
    if mesh_mask.dtype != np.bool_:
        raise TypeError(f"object mesh mask must have dtype bool, got {mesh_mask.dtype}")
    validate_pose_valid(pose_valid, expected_shape[0])
    if (
        not math.isfinite(minimum_depth_m)
        or not math.isfinite(maximum_depth_m)
        or not 0.0 < minimum_depth_m < maximum_depth_m
    ):
        raise ValueError("invalid mesh depth range")

    mesh_pixels = 0
    valid_pose_mesh_pixels = 0
    for frame_index in range(expected_shape[0]):
        mask = np.asarray(mesh_mask[frame_index], dtype=bool)
        front = np.asarray(front_depth[frame_index], dtype=np.float32)
        back = np.asarray(back_depth[frame_index], dtype=np.float32)
        mesh_pixels += int(mask.sum())
        outside = ~mask
        invalid_sentinel = outside & ((front != 0.0) | (back != 0.0))
        if np.any(invalid_sentinel):
            raise ValueError(
                "mesh front/back outside-mask sentinel must be 0.0 at frame "
                f"{frame_index}: {int(invalid_sentinel.sum())} pixels"
            )
        if not bool(pose_valid[frame_index]):
            if np.any(mask):
                raise ValueError(
                    f"pose-invalid frame {frame_index} has {int(mask.sum())} "
                    "mesh pixels"
                )
            continue
        valid_pose_mesh_pixels += int(mask.sum())
        if not np.any(mask):
            continue
        valid = (
            np.isfinite(front)
            & np.isfinite(back)
            & (front > np.float32(minimum_depth_m))
            & (back < np.float32(maximum_depth_m))
            & (front <= back)
        )
        invalid = mask & ~valid
        if np.any(invalid):
            raise ValueError(
                "mesh front/back depth is invalid or unordered at frame "
                f"{frame_index}: {int(invalid.sum())} pixels"
            )
    return {
        "mesh_pixels": mesh_pixels,
        "valid_pose_mesh_pixels": valid_pose_mesh_pixels,
        "valid_pose_frames": int(np.asarray(pose_valid, dtype=bool).sum()),
    }


def classify_mesh_volume(
    *,
    hand_mask: np.ndarray,
    robot_depth: np.ndarray,
    object_support_mask: np.ndarray,
    mesh_mask: np.ndarray,
    front_depth: np.ndarray,
    back_depth: np.ndarray,
    pose_valid: bool,
    shell_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify valid XHand pixels into a disjoint camera-ray volume state."""
    hand = np.asarray(hand_mask, dtype=bool)
    robot_z = np.asarray(robot_depth, dtype=np.float32)
    texture = np.asarray(object_support_mask, dtype=bool)
    mesh = np.asarray(mesh_mask, dtype=bool)
    front = np.asarray(front_depth, dtype=np.float32)
    back = np.asarray(back_depth, dtype=np.float32)
    shell = np.asarray(shell_m, dtype=np.float32)
    if not (
        hand.shape
        == robot_z.shape
        == texture.shape
        == mesh.shape
        == front.shape
        == back.shape
        == shell.shape
    ):
        raise ValueError("mesh-volume classification inputs must share one shape")
    if np.any(~np.isfinite(shell)) or np.any(shell < 0.0):
        raise ValueError("XHand shell must be finite and non-negative")

    valid_interval = (
        np.isfinite(front)
        & np.isfinite(back)
        & (front > 0.02)
        & (back < 5.0)
        & (front <= back)
    )
    support = (
        hand
        & texture
        & mesh
        & bool(pose_valid)
        & np.isfinite(robot_z)
        & valid_interval
    )
    hand_back = robot_z + shell
    front_of = support & (hand_back <= front)
    intersecting = support & (hand_back > front) & (robot_z < back)
    # Quantised float16 mesh rasters can collapse a very thin material layer
    # so that ``front == back``.  Assign the shared equality boundary to
    # ``front_of`` first; a pixel is fully behind only after the hand (including
    # its shell) has crossed the front surface.  This keeps the three states a
    # strict partition and preserves the zero-shell ``robot_z > front`` gate.
    fully_behind = support & (hand_back > front) & (robot_z >= back)

    # Ordered front/back intervals and a non-negative shell make these masks a
    # complete, mutually exclusive partition of support.
    overlap = (
        (front_of & intersecting)
        | (front_of & fully_behind)
        | (intersecting & fully_behind)
    )
    if np.any(overlap):
        raise RuntimeError("mesh-volume classes overlap")
    covered = front_of | intersecting | fully_behind
    if np.any(covered != support):
        raise RuntimeError("mesh-volume classes do not partition valid support")

    classification = np.zeros(hand.shape, dtype=np.uint8)
    classification[front_of] = CLASS_FRONT_OF
    classification[intersecting] = CLASS_INTERSECTING
    classification[fully_behind] = CLASS_FULLY_BEHIND
    return classification, support


def hidden_from_classification(classification: np.ndarray) -> np.ndarray:
    """Return the intersecting-or-behind volume barrier."""
    value = np.asarray(classification, dtype=np.uint8)
    return (value == CLASS_INTERSECTING) | (value == CLASS_FULLY_BEHIND)


def front_only_hidden(
    *,
    support: np.ndarray,
    robot_depth: np.ndarray,
    front_depth: np.ndarray,
) -> np.ndarray:
    """Return the zero-shell visible-front z-buffer barrier."""
    valid = np.asarray(support, dtype=bool)
    robot_z = np.asarray(robot_depth, dtype=np.float32)
    front = np.asarray(front_depth, dtype=np.float32)
    if not (valid.shape == robot_z.shape == front.shape):
        raise ValueError("front-only barrier inputs must share one shape")
    return valid & (robot_z > front)


def combine_with_baseline(
    baseline: np.ndarray,
    mesh_hidden: np.ndarray,
    hand_mask: np.ndarray,
) -> np.ndarray:
    """Retain the finger baseline while guaranteeing an XHand-only result."""
    base = np.asarray(baseline, dtype=bool)
    hidden = np.asarray(mesh_hidden, dtype=bool)
    hand = np.asarray(hand_mask, dtype=bool)
    if not (base.shape == hidden.shape == hand.shape):
        raise ValueError("baseline, mesh barrier, and hand mask must align")
    if np.any(base & ~hand):
        raise ValueError("baseline finger mask escaped the XHand mask")
    if np.any(hidden & ~hand):
        raise ValueError("mesh barrier escaped the XHand mask")
    return base | hidden


def mesh_temporal_eligibility(
    *,
    classification_support: np.ndarray,
    robot_depth: np.ndarray,
    front_depth: np.ndarray,
    shell_m: np.ndarray,
    front_slack_m: float,
) -> np.ndarray:
    """Permit bounded closing only on valid mesh/texture/pose support."""
    if not math.isfinite(front_slack_m) or front_slack_m < 0.0:
        raise ValueError("temporal front slack must be finite and non-negative")
    support = np.asarray(classification_support, dtype=bool)
    robot_z = np.asarray(robot_depth, dtype=np.float32)
    front = np.asarray(front_depth, dtype=np.float32)
    shell = np.asarray(shell_m, dtype=np.float32)
    if not (support.shape == robot_z.shape == front.shape == shell.shape):
        raise ValueError("mesh temporal inputs must share one shape")
    clearly_front = (
        np.isfinite(robot_z)
        & np.isfinite(front)
        & ((robot_z + shell) <= front - np.float32(front_slack_m))
    )
    return support & ~clearly_front


def _part_counts(values: np.ndarray) -> dict[str, int]:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != len(PART_NAMES):
        raise ValueError("part counts must have shape (T,6)")
    return {
        name: int(array[:, index].sum())
        for index, name in enumerate(PART_NAMES)
    }


def _builder_report(
    path: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
) -> dict[str, object]:
    report = json.loads(path.read_text())
    if report.get("method") != BUILDER_METHOD:
        raise ValueError(
            f"mesh builder method {report.get('method')!r} != {BUILDER_METHOD!r}"
        )
    if report.get("representation") != BUILDER_REPRESENTATION:
        raise ValueError(
            "mesh builder representation "
            f"{report.get('representation')!r} != {BUILDER_REPRESENTATION!r}"
        )
    metadata = report.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("mesh builder metadata must be an object")
    expected = {"frames": frame_count, "width": width, "height": height}
    for key, value in expected.items():
        actual = metadata.get(key, report.get(key))
        if actual is not None and int(actual) != value:
            raise ValueError(
                f"mesh builder {key} {actual} differs from video {value}"
            )
    return report


def _paint_classification(
    frame: np.ndarray,
    classification: np.ndarray,
    temporal_added: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    output = np.asarray(frame, dtype=np.uint8).copy()
    labels = np.asarray(classification, dtype=np.uint8)
    temporal = np.asarray(temporal_added, dtype=bool)
    colours = {
        int(CLASS_FRONT_OF): np.asarray((255, 255, 0), dtype=np.float32),
        int(CLASS_INTERSECTING): np.asarray((255, 0, 255), dtype=np.float32),
        int(CLASS_FULLY_BEHIND): np.asarray((0, 255, 255), dtype=np.float32),
    }
    for class_id, colour in colours.items():
        mask = labels == class_id
        if np.any(mask):
            output[mask] = np.clip(
                output[mask].astype(np.float32) * 0.40 + colour * 0.60,
                0,
                255,
            ).astype(np.uint8)
    if np.any(temporal):
        colour = np.asarray((0, 255, 0), dtype=np.float32)
        output[temporal] = np.clip(
            output[temporal].astype(np.float32) * 0.25 + colour * 0.75,
            0,
            255,
        ).astype(np.uint8)
    cv2.putText(
        output,
        f"mesh {mode}: cyan=front magenta=intersect yellow=behind green=temporal",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--raw_video", type=Path, required=True)
    parser.add_argument("--overlay_dir", type=Path, required=True)
    parser.add_argument("--mesh_dir", type=Path, required=True)
    parser.add_argument("--object_support_mask", type=Path, required=True)
    parser.add_argument("--object_restore_mask", type=Path, required=True)
    parser.add_argument("--baseline_mask", type=Path, required=True)
    parser.add_argument("--mode", choices=("front", "volume"), required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--thumb_shell_m", type=float, default=0.0)
    parser.add_argument("--finger_shell_m", type=float, default=0.0)
    parser.add_argument("--palm_shell_m", type=float, default=0.0)
    parser.add_argument("--temporal_max_gap_frames", type=int, default=0)
    parser.add_argument("--temporal_motion_px", type=int, default=6)
    parser.add_argument("--temporal_front_slack_m", type=float, default=0.015)
    parser.add_argument("--robot_edge_sigma_px", type=float, default=0.6)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not 0 <= args.temporal_max_gap_frames <= 2:
        parser.error("--temporal_max_gap_frames must be in 0..2")
    if args.temporal_motion_px < 0:
        parser.error("--temporal_motion_px must be non-negative")
    shell_values = (
        float(args.thumb_shell_m),
        float(args.finger_shell_m),
        float(args.palm_shell_m),
    )
    if any(not math.isfinite(value) or value < 0.0 for value in shell_values):
        parser.error("XHand shell values must be finite and non-negative")
    if args.mode == "front" and (
        any(value != 0.0 for value in shell_values)
        or args.temporal_max_gap_frames != 0
    ):
        parser.error("front mode requires zero shell and temporal filtering off")

    background_path = args.background.expanduser().resolve()
    raw_video_path = args.raw_video.expanduser().resolve()
    overlay_dir = args.overlay_dir.expanduser().resolve()
    mesh_dir = args.mesh_dir.expanduser().resolve()
    object_support_path = args.object_support_mask.expanduser().resolve()
    object_restore_path = args.object_restore_mask.expanduser().resolve()
    baseline_path = args.baseline_mask.expanduser().resolve()
    output_dir = args.out_dir.expanduser().resolve()
    mesh_paths = {
        "front": mesh_dir / FRONT_DEPTH_NAME,
        "back": mesh_dir / BACK_DEPTH_NAME,
        "mask": mesh_dir / MESH_MASK_NAME,
        "pose_valid": mesh_dir / POSE_VALID_NAME,
        "report": mesh_dir / BUILDER_REPORT_NAME,
    }
    overlay_paths = {
        "rgb": overlay_dir / "robot_rgb.npy",
        "depth": overlay_dir / "robot_depth.npy",
        "robot": overlay_dir / "robot_mask.npy",
        "hand": overlay_dir / "robot_hand_mask.npy",
        "labels": overlay_dir / "robot_finger_labels.npy",
    }
    for path in (
        background_path,
        raw_video_path,
        object_support_path,
        object_restore_path,
        baseline_path,
        *mesh_paths.values(),
        *overlay_paths.values(),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)

    width, height, frame_count, fps = _video_metadata(background_path)
    raw_width, raw_height, raw_frames, raw_fps = _video_metadata(raw_video_path)
    if (
        (raw_width, raw_height, raw_frames) != (width, height, frame_count)
        or not np.isclose(raw_fps, fps, atol=1.0e-6)
    ):
        raise ValueError("background/raw video metadata mismatch")
    expected_shape = (frame_count, height, width)

    front_depth = np.load(mesh_paths["front"], mmap_mode="r", allow_pickle=False)
    back_depth = np.load(mesh_paths["back"], mmap_mode="r", allow_pickle=False)
    mesh_mask = np.load(mesh_paths["mask"], mmap_mode="r", allow_pickle=False)
    pose_valid = np.load(
        mesh_paths["pose_valid"], mmap_mode="r", allow_pickle=False
    )
    object_support = np.load(
        object_support_path, mmap_mode="r", allow_pickle=False
    )
    object_restore = np.load(
        object_restore_path, mmap_mode="r", allow_pickle=False
    )
    baseline_mask = np.load(baseline_path, mmap_mode="r", allow_pickle=False)
    validate_mask_volume(
        object_support,
        name="object_support_mask",
        expected_shape=expected_shape,
    )
    validate_mask_volume(
        object_restore,
        name="object_restore_mask",
        expected_shape=expected_shape,
    )
    validate_mask_volume(
        baseline_mask,
        name="baseline_mask",
        expected_shape=expected_shape,
    )
    restore_outside_support = 0
    for frame_index in range(frame_count):
        restore_outside_support += int(
            np.sum(
                np.asarray(object_restore[frame_index], dtype=bool)
                & ~np.asarray(object_support[frame_index], dtype=bool)
            )
        )
    if restore_outside_support:
        raise ValueError(
            "object restore mask escaped renderable object support: "
            f"{restore_outside_support} pixels"
        )
    mesh_diagnostics = validate_depth_volume(
        front_depth,
        back_depth,
        mesh_mask,
        pose_valid,
        expected_shape=expected_shape,
    )
    builder_report = _builder_report(
        mesh_paths["report"],
        frame_count=frame_count,
        width=width,
        height=height,
    )

    robot_rgb = np.load(overlay_paths["rgb"], mmap_mode="r", allow_pickle=False)
    robot_depth = np.load(
        overlay_paths["depth"], mmap_mode="r", allow_pickle=False
    )
    robot_mask = np.load(
        overlay_paths["robot"], mmap_mode="r", allow_pickle=False
    )
    robot_hand = np.load(
        overlay_paths["hand"], mmap_mode="r", allow_pickle=False
    )
    finger_labels = np.load(
        overlay_paths["labels"], mmap_mode="r", allow_pickle=False
    )
    overlay_shape = robot_depth.shape
    if len(overlay_shape) != 3 or overlay_shape[0] != frame_count:
        raise ValueError("overlay depth must have aligned shape (T,H,W)")
    if robot_rgb.shape != overlay_shape + (3,) or not (
        robot_mask.shape
        == robot_hand.shape
        == finger_labels.shape
        == overlay_shape
    ):
        raise ValueError("overlay arrays differ in shape")
    if robot_rgb.dtype != np.uint8:
        raise TypeError("robot RGB must have dtype uint8")
    if robot_mask.dtype != np.bool_ or robot_hand.dtype != np.bool_:
        raise TypeError("robot/hand masks must have dtype bool")
    if finger_labels.dtype != np.uint8:
        raise TypeError("finger labels must have dtype uint8")
    if not np.issubdtype(robot_depth.dtype, np.floating):
        raise TypeError("robot depth must have a floating dtype")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".xhand_mesh_volume.", dir=output_dir.parent)
    )
    atexit.register(shutil.rmtree, staging, ignore_errors=True)
    classification_path = staging / "mesh_volume_classification.npy"
    classification_buffer = np.lib.format.open_memmap(
        classification_path,
        mode="w+",
        dtype=np.uint8,
        shape=expected_shape,
    )
    raw_mask_path = staging / ".raw_occluded_hand_mask.npy"
    raw_buffer = np.lib.format.open_memmap(
        raw_mask_path,
        mode="w+",
        dtype=bool,
        shape=expected_shape,
    )
    eligibility_path = staging / ".temporal_eligibility.npy"
    semantic_path = staging / ".temporal_hand_labels.npy"
    eligibility_buffer = None
    semantic_buffer = None
    if args.temporal_max_gap_frames > 0:
        eligibility_buffer = np.lib.format.open_memmap(
            eligibility_path,
            mode="w+",
            dtype=bool,
            shape=expected_shape,
        )
        semantic_buffer = np.lib.format.open_memmap(
            semantic_path,
            mode="w+",
            dtype=np.uint8,
            shape=expected_shape,
        )

    class_counts = np.zeros((frame_count, 3), dtype=np.int64)
    class_part_counts = np.zeros(
        (frame_count, 3, len(PART_NAMES)), dtype=np.int64
    )
    raw_hide_counts = np.zeros(frame_count, dtype=np.int64)
    raw_added_part_counts = np.zeros(
        (frame_count, len(PART_NAMES)), dtype=np.int64
    )
    baseline_counts = np.zeros(frame_count, dtype=np.int64)
    valid_support_counts = np.zeros(frame_count, dtype=np.int64)
    mesh_outside_support_counts = np.zeros(frame_count, dtype=np.int64)
    source_presence = np.zeros((frame_count, len(PART_NAMES)), dtype=bool)

    for frame_index in range(frame_count):
        (_rgb, depth, _robot, hand, labels) = resize_overlay_frame(
            robot_rgb[frame_index],
            robot_depth[frame_index],
            robot_mask[frame_index],
            robot_hand[frame_index],
            finger_labels[frame_index],
            width=width,
            height=height,
        )
        hand_labels = semantic_hand_labels(hand, labels)
        shell = thickness_map(
            hand_labels,
            thumb_shell_m=args.thumb_shell_m,
            finger_shell_m=args.finger_shell_m,
            palm_shell_m=args.palm_shell_m,
        )
        support_mask = np.asarray(object_support[frame_index], dtype=bool)
        frame_mesh = np.asarray(mesh_mask[frame_index], dtype=bool)
        classification, support = classify_mesh_volume(
            hand_mask=hand,
            robot_depth=depth,
            object_support_mask=support_mask,
            mesh_mask=frame_mesh,
            front_depth=front_depth[frame_index],
            back_depth=back_depth[frame_index],
            pose_valid=bool(pose_valid[frame_index]),
            shell_m=shell,
        )
        classification_buffer[frame_index] = classification
        volume_hidden = hidden_from_classification(classification)
        if args.mode == "front":
            mesh_hidden = front_only_hidden(
                support=support,
                robot_depth=depth,
                front_depth=front_depth[frame_index],
            )
            if np.any(mesh_hidden != volume_hidden):
                raise RuntimeError(
                    "zero-shell front and volume barriers differ at frame "
                    f"{frame_index}"
                )
        else:
            mesh_hidden = volume_hidden
        baseline = np.asarray(baseline_mask[frame_index], dtype=bool)
        combined = combine_with_baseline(baseline, mesh_hidden, hand)
        raw_buffer[frame_index] = combined
        baseline_counts[frame_index] = int(baseline.sum())
        raw_hide_counts[frame_index] = int(mesh_hidden.sum())
        valid_support_counts[frame_index] = int(support.sum())
        mesh_outside_support_counts[frame_index] = int(
            np.sum(frame_mesh & ~support_mask)
        )
        for class_index, class_id in enumerate(
            (CLASS_FRONT_OF, CLASS_INTERSECTING, CLASS_FULLY_BEHIND)
        ):
            class_mask = classification == class_id
            class_counts[frame_index, class_index] = int(class_mask.sum())
            for part_index in range(len(PART_NAMES)):
                class_part_counts[frame_index, class_index, part_index] = int(
                    np.sum(class_mask & (hand_labels == part_index + 1))
                )
        added = mesh_hidden & ~baseline
        for part_index in range(len(PART_NAMES)):
            part = hand_labels == part_index + 1
            raw_added_part_counts[frame_index, part_index] = int(
                np.sum(added & part)
            )
            source_presence[frame_index, part_index] = bool(
                np.any(combined & part)
            )
        if eligibility_buffer is not None:
            assert semantic_buffer is not None
            eligibility_buffer[frame_index] = mesh_temporal_eligibility(
                classification_support=support,
                robot_depth=depth,
                front_depth=front_depth[frame_index],
                shell_m=shell,
                front_slack_m=args.temporal_front_slack_m,
            )
            semantic_buffer[frame_index] = hand_labels
        if (frame_index + 1) % 100 == 0:
            print(f"[mesh-volume-mask] {frame_index + 1}/{frame_count}", flush=True)

    classification_buffer.flush()
    raw_buffer.flush()
    if eligibility_buffer is not None:
        eligibility_buffer.flush()
    if semantic_buffer is not None:
        semantic_buffer.flush()
    if not np.array_equal(class_counts.sum(axis=1), valid_support_counts):
        raise RuntimeError("mesh-volume class counts do not partition support")
    if not np.array_equal(
        class_counts[:, 1] + class_counts[:, 2], raw_hide_counts
    ):
        raise RuntimeError("mesh-volume hide counts differ from intersect+behind")

    final_mask_path = staging / "occluded_hand_mask.npy"
    temporal_diagnostics: dict[str, object] = {
        "added_pixels": 0,
        "added_frames": 0,
        "added_frame_fingers": 0,
    }
    temporal_added_part_counts = np.zeros_like(raw_added_part_counts)
    if eligibility_buffer is None:
        del raw_buffer
        os.replace(raw_mask_path, final_mask_path)
    else:
        assert semantic_buffer is not None
        final_buffer = np.lib.format.open_memmap(
            final_mask_path,
            mode="w+",
            dtype=bool,
            shape=expected_shape,
        )
        _, temporal_diagnostics = bridge_short_occlusion_gaps(
            raw_buffer,
            eligibility_buffer,
            semantic_buffer,
            max_gap_frames=args.temporal_max_gap_frames,
            motion_radius_px=args.temporal_motion_px,
            label_count=len(PART_NAMES),
            source_presence=source_presence,
            output=final_buffer,
        )
        temporal_added_part_counts[:] = np.asarray(
            temporal_diagnostics["added_per_frame_finger"], dtype=np.int64
        )
        final_buffer.flush()
        del final_buffer
        del raw_buffer
        del eligibility_buffer
        del semantic_buffer
        raw_mask_path.unlink(missing_ok=True)
        eligibility_path.unlink(missing_ok=True)
        semantic_path.unlink(missing_ok=True)
    del classification_buffer

    final_mask = np.load(final_mask_path, mmap_mode="r", allow_pickle=False)
    classification_volume = np.load(
        classification_path, mmap_mode="r", allow_pickle=False
    )
    final_counts = np.zeros(frame_count, dtype=np.int64)
    final_added_part_counts = np.zeros_like(raw_added_part_counts)
    residual_counts = np.zeros(frame_count, dtype=np.int64)
    raw_capture = cv2.VideoCapture(str(raw_video_path))
    background_capture = cv2.VideoCapture(str(background_path))
    final_writer = _open_writer(
        staging / "video_overlay_mesh_volume.mp4", fps, (width, height)
    )
    robot_writer = _open_writer(
        staging / "video_robot_only_mesh_volume.mp4", fps, (width, height)
    )
    debug_writer = _open_writer(
        staging / "debug_mesh_volume.mp4", fps, (width, height)
    )
    try:
        for frame_index in range(frame_count):
            ok_raw, raw_frame = raw_capture.read()
            ok_background, background = background_capture.read()
            if not ok_raw or not ok_background:
                raise RuntimeError(f"video read failed at frame {frame_index}")
            if raw_frame.shape[:2] != (height, width):
                raw_frame = cv2.resize(
                    raw_frame, (width, height), interpolation=cv2.INTER_AREA
                )
            if background.shape[:2] != (height, width):
                background = cv2.resize(
                    background, (width, height), interpolation=cv2.INTER_AREA
                )
            (rgb, _depth, robot, hand, labels) = resize_overlay_frame(
                robot_rgb[frame_index],
                robot_depth[frame_index],
                robot_mask[frame_index],
                robot_hand[frame_index],
                finger_labels[frame_index],
                width=width,
                height=height,
            )
            hand_labels = semantic_hand_labels(hand, labels)
            classification = np.asarray(
                classification_volume[frame_index], dtype=np.uint8
            )
            raw_mesh_hidden = hidden_from_classification(classification)
            baseline = np.asarray(baseline_mask[frame_index], dtype=bool)
            raw_combined = baseline | raw_mesh_hidden
            occluded = np.asarray(final_mask[frame_index], dtype=bool)
            if np.any(raw_combined & ~occluded):
                raise RuntimeError(
                    f"final mask removed raw evidence at frame {frame_index}"
                )
            if np.any(occluded & ~hand):
                raise RuntimeError(
                    f"mesh-volume mask escaped XHand at frame {frame_index}"
                )
            residual = raw_mesh_hidden & ~occluded
            residual_counts[frame_index] = int(residual.sum())
            if residual_counts[frame_index]:
                raise RuntimeError(
                    f"mesh-volume barrier left {residual.sum()} violations at "
                    f"frame {frame_index}"
                )
            final_counts[frame_index] = int(occluded.sum())
            added = occluded & ~baseline
            for part_index in range(len(PART_NAMES)):
                final_added_part_counts[frame_index, part_index] = int(
                    np.sum(added & (hand_labels == part_index + 1))
                )

            composite_background = restore_raw_object_pixels(
                background,
                raw_frame,
                np.asarray(object_restore[frame_index], dtype=bool),
            )
            final, robot_only, _alpha = composite_frame(
                composite_background,
                rgb,
                robot,
                hand,
                occluded,
                robot_edge_sigma_px=args.robot_edge_sigma_px,
                occlusion_edge_sigma_px=0.0,
            )
            final_writer.write(final)
            robot_writer.write(robot_only)
            debug_writer.write(
                _paint_classification(
                    final,
                    classification,
                    occluded & ~raw_combined,
                    mode=args.mode,
                )
            )
            if (frame_index + 1) % 100 == 0:
                print(
                    f"[mesh-volume-composite] {frame_index + 1}/{frame_count}",
                    flush=True,
                )
    finally:
        raw_capture.release()
        background_capture.release()
        final_writer.release()
        robot_writer.release()
        debug_writer.release()

    if np.any(residual_counts):
        raise RuntimeError("final mesh-volume barrier has residual violations")
    baseline_total = int(baseline_counts.sum())
    final_total = int(final_counts.sum())
    if final_total < baseline_total:
        raise RuntimeError("mesh-volume barrier removed baseline occlusion")

    np.savez(
        staging / "mesh_volume_evidence.npz",
        class_names=np.asarray(CLASS_NAMES),
        part_names=np.asarray(PART_NAMES),
        pose_valid=np.asarray(pose_valid, dtype=bool),
        front_of_pixels=class_counts[:, 0],
        intersecting_pixels=class_counts[:, 1],
        fully_behind_pixels=class_counts[:, 2],
        front_of_pixels_by_part=class_part_counts[:, 0],
        intersecting_pixels_by_part=class_part_counts[:, 1],
        fully_behind_pixels_by_part=class_part_counts[:, 2],
        raw_mesh_hidden_pixels=raw_hide_counts,
        raw_added_pixels_by_part=raw_added_part_counts,
        temporal_added_pixels_by_part=temporal_added_part_counts,
        final_added_pixels_by_part=final_added_part_counts,
        valid_support_pixels=valid_support_counts,
        mesh_outside_object_support_pixels=mesh_outside_support_counts,
        residual_violation_pixels=residual_counts,
    )
    report = {
        "schema_version": 1,
        "method": METHOD,
        "mode": args.mode,
        "representation": {
            "object": "pose-fitted nominal MJCF mesh front/back camera-Z volume",
            "hand": "visible XHand front depth plus part-wise camera-Z shell",
            "builder_method": BUILDER_METHOD,
        },
        "pose_state_modified": False,
        "physical_collision_solver": False,
        "metric_collision_guarantee": False,
        "frames": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "config": {
            "mode": args.mode,
            "thumb_shell_m": shell_values[0],
            "finger_shell_m": shell_values[1],
            "palm_shell_m": shell_values[2],
            "temporal_max_gap_frames": int(args.temporal_max_gap_frames),
            "temporal_motion_px": int(args.temporal_motion_px),
            "temporal_front_slack_m": float(args.temporal_front_slack_m),
            "robot_edge_sigma_px": float(args.robot_edge_sigma_px),
            "front_mode_requires_zero_shell": True,
        },
        "sources": {
            "background": str(background_path),
            "raw_video": str(raw_video_path),
            "overlay_dir": str(overlay_dir),
            "mesh_dir": str(mesh_dir),
            "mesh_front_depth": str(mesh_paths["front"]),
            "mesh_back_depth": str(mesh_paths["back"]),
            "mesh_mask": str(mesh_paths["mask"]),
            "pose_valid": str(mesh_paths["pose_valid"]),
            "mesh_report": str(mesh_paths["report"]),
            "object_support_mask": str(object_support_path),
            "object_restore_mask": str(object_restore_path),
            "baseline_mask": str(baseline_path),
        },
        "mesh_builder": {
            "method": builder_report.get("method"),
            "representation": builder_report.get("representation"),
            "schema_version": builder_report.get("schema_version"),
        },
        "classification": {
            "invalid": int(CLASS_INVALID),
            "front_of": int(CLASS_FRONT_OF),
            "intersecting": int(CLASS_INTERSECTING),
            "fully_behind": int(CLASS_FULLY_BEHIND),
            "hide_definition": "intersecting OR fully_behind",
        },
        "counts": {
            **mesh_diagnostics,
            "baseline_occluded_pixels": baseline_total,
            "valid_comparison_support_pixels": int(valid_support_counts.sum()),
            "front_of_pixels": int(class_counts[:, 0].sum()),
            "intersecting_pixels": int(class_counts[:, 1].sum()),
            "fully_behind_pixels": int(class_counts[:, 2].sum()),
            "raw_mesh_hidden_pixels": int(raw_hide_counts.sum()),
            "raw_mesh_added_pixels": int(raw_added_part_counts.sum()),
            "temporal_added_pixels": int(temporal_added_part_counts.sum()),
            "final_occluded_pixels": final_total,
            "final_frames_with_occlusion": int((final_counts > 0).sum()),
            "mesh_outside_object_support_pixels": int(
                mesh_outside_support_counts.sum()
            ),
            "residual_violation_pixels": int(residual_counts.sum()),
            "front_of_by_part": _part_counts(class_part_counts[:, 0]),
            "intersecting_by_part": _part_counts(class_part_counts[:, 1]),
            "fully_behind_by_part": _part_counts(class_part_counts[:, 2]),
            "raw_added_by_part": _part_counts(raw_added_part_counts),
            "temporal_added_by_part": _part_counts(temporal_added_part_counts),
            "final_added_by_part": _part_counts(final_added_part_counts),
        },
        "temporal_filter": {
            "enabled": bool(args.temporal_max_gap_frames > 0),
            "offline_bidirectional": bool(args.temporal_max_gap_frames > 0),
            "added_frames": int(temporal_diagnostics.get("added_frames", 0)),
            "added_frame_parts": int(
                temporal_diagnostics.get("added_frame_fingers", 0)
            ),
        },
        "invariants": {
            "builder_method_validated": True,
            "mesh_front_back_order_valid": True,
            "classification_is_mutually_exclusive": True,
            "classification_partitions_valid_support": True,
            "volume_hide_is_intersecting_or_fully_behind": True,
            "front_mode_is_zero_shell_front_gate": bool(
                args.mode != "front"
                or (
                    all(value == 0.0 for value in shell_values)
                    and args.temporal_max_gap_frames == 0
                )
            ),
            "front_mode_controls_are_valid": bool(
                args.mode != "front"
                or (
                    all(value == 0.0 for value in shell_values)
                    and args.temporal_max_gap_frames == 0
                )
            ),
            "baseline_subset_final": True,
            "final_occlusion_subset_of_xhand": True,
            "rb5_arm_excluded": True,
            "valid_volume_barrier_residual_is_zero": True,
            "mesh_support_clipped_to_object_texture_support": True,
            "raw_rgb_restore_uses_restore_mask_only": True,
            "unknown_mesh_fails_open_except_existing_baseline": True,
            "trajectory_arrays_unchanged": True,
            "pose_state_unchanged": True,
        },
        "outputs": {
            "final_video": "video_overlay_mesh_volume.mp4",
            "robot_only_video": "video_robot_only_mesh_volume.mp4",
            "debug_video": "debug_mesh_volume.mp4",
            "mask": "occluded_hand_mask.npy",
            "classification": "mesh_volume_classification.npy",
            "evidence": "mesh_volume_evidence.npz",
        },
        "provenance_warning": (
            "The object mesh pose and MH intrinsics are inferred, and XHand "
            "thickness is a camera-Z shell. This is a visual non-emergence "
            "constraint, not calibrated physical mesh-mesh collision."
        ),
    }
    (staging / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    publish_directory(str(staging), str(output_dir))
    print(
        f"[ok] XHand mesh-volume barrier: {output_dir} mode={args.mode} "
        f"pixels={final_total} frames={(final_counts > 0).sum()}/{frame_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
