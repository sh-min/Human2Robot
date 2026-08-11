"""Composite RB5 + XHand with a whole-hand camera-Z object barrier.

This is deliberately a *visual* non-penetration constraint.  The object input
is a visible camera-Z height field, not a watertight mesh/SDF, so the script
does not modify wrist, finger, or RB5 joint state.  It extends the established
finger-only Object3D result to every rendered XHand pixel (palm/base included),
optionally tests a conservative hand-thickness shell, and can bridge bounded
one/two-frame holes offline.

The pre-existing finger result is always OR-ed into the new mask.  Unknown
surface depth fails open.  RB5 arm pixels are never hidden by this barrier.
An optional object-restore mask can be narrower than the barrier/object support:
it controls only which source RGB pixels are restored after background
inpainting and never changes the camera-Z barrier decision.
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
    FINGER_NAMES,
    bridge_short_occlusion_gaps,
    composite_frame,
)


PART_NAMES = (*FINGER_NAMES, "palm_base")
PALM_BASE_LABEL = 6


def _video_metadata(path: Path) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    if width <= 0 or height <= 0 or frames <= 0 or fps <= 0.0:
        raise ValueError(f"invalid video metadata: {path}")
    return width, height, frames, fps


def _open_writer(
    path: Path,
    fps: float,
    size: tuple[int, int],
) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {path}")
    return writer


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    value = np.asarray(mask, dtype=np.uint8)
    if value.shape != (height, width):
        value = cv2.resize(
            value,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    return value.astype(bool)


def _resize_depth(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    value = np.asarray(depth, dtype=np.float32)
    if value.shape != (height, width):
        value = cv2.resize(
            value,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    return value


def validate_mask_volume(
    mask: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, int, int],
) -> None:
    """Validate the strict bool ``(T,H,W)`` contract for a video mask."""
    if mask.ndim != 3:
        raise ValueError(f"{name} must have shape (T,H,W), got {mask.shape}")
    if mask.shape != expected_shape:
        raise ValueError(
            f"{name} shape {mask.shape} differs from video {expected_shape}"
        )
    if mask.dtype != np.bool_:
        raise TypeError(f"{name} must have dtype bool, got {mask.dtype}")


def select_object_restore_mask(
    object_mask: np.ndarray,
    object_restore_mask: np.ndarray | None,
) -> np.ndarray:
    """Use the barrier mask for RGB restoration unless an override is given."""
    return object_mask if object_restore_mask is None else object_restore_mask


def restore_raw_object_pixels(
    background: np.ndarray,
    raw_frame: np.ndarray,
    restore_mask: np.ndarray,
) -> np.ndarray:
    """Restore source RGB only at the explicitly selected object pixels."""
    base = np.asarray(background, dtype=np.uint8)
    raw = np.asarray(raw_frame, dtype=np.uint8)
    mask = np.asarray(restore_mask, dtype=bool)
    if base.shape != raw.shape or mask.shape != base.shape[:2]:
        raise ValueError("raw-object restoration inputs are not aligned")
    output = base.copy()
    output[mask] = raw[mask]
    return output


def resize_overlay_frame(
    robot_rgb: np.ndarray,
    robot_depth: np.ndarray,
    robot_mask: np.ndarray,
    hand_mask: np.ndarray,
    finger_labels: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resize one overlay frame while retaining hand/robot subset contracts."""
    rgb = np.asarray(robot_rgb, dtype=np.uint8)
    depth = np.asarray(robot_depth, dtype=np.float32)
    robot = np.asarray(robot_mask, dtype=np.uint8)
    hand = np.asarray(hand_mask, dtype=np.uint8)
    labels = np.asarray(finger_labels, dtype=np.uint8)
    if not (
        rgb.shape[:2]
        == depth.shape
        == robot.shape
        == hand.shape
        == labels.shape
    ):
        raise ValueError("overlay arrays are not spatially aligned")
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(
            depth,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        robot = cv2.resize(
            robot,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        hand = cv2.resize(
            hand,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        labels = cv2.resize(
            labels,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    robot_bool = robot.astype(bool)
    hand_bool = hand.astype(bool)
    finger_bool = labels > 0
    # Nearest sampling can choose opposite sides of a one-pixel edge.  The
    # semantic render remains authoritative for these subset repairs.
    hand_bool |= finger_bool
    robot_bool |= hand_bool
    return rgb, depth, robot_bool, hand_bool, labels


def semantic_hand_labels(
    hand_mask: np.ndarray,
    finger_labels: np.ndarray,
) -> np.ndarray:
    """Return labels 1..5 for fingers and 6 for palm/XHand base."""
    hand = np.asarray(hand_mask, dtype=bool)
    labels = np.asarray(finger_labels, dtype=np.uint8)
    if hand.shape != labels.shape:
        raise ValueError("hand mask and finger labels must share one shape")
    if labels.size and int(labels.max()) > len(FINGER_NAMES):
        raise ValueError("finger labels are outside 0..5")
    output = labels.copy()
    output[hand & (output == 0)] = PALM_BASE_LABEL
    output[~hand] = 0
    return output


def thickness_map(
    hand_labels: np.ndarray,
    *,
    thumb_shell_m: float,
    finger_shell_m: float,
    palm_shell_m: float,
) -> np.ndarray:
    """Return the configured camera-Z safety shell for each XHand part."""
    values = (thumb_shell_m, finger_shell_m, palm_shell_m)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("hand shell values must be finite and non-negative")
    labels = np.asarray(hand_labels, dtype=np.uint8)
    shell = np.zeros(labels.shape, dtype=np.float32)
    shell[labels == 1] = np.float32(thumb_shell_m)
    shell[(labels >= 2) & (labels <= 5)] = np.float32(finger_shell_m)
    shell[labels == PALM_BASE_LABEL] = np.float32(palm_shell_m)
    return shell


def compute_visual_barrier(
    *,
    robot_mask: np.ndarray,
    hand_mask: np.ndarray,
    robot_depth: np.ndarray,
    object_mask: np.ndarray,
    object_surface_depth: np.ndarray,
    shell_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return barrier mask and valid XHand/object comparison support."""
    robot = np.asarray(robot_mask, dtype=bool)
    hand = np.asarray(hand_mask, dtype=bool)
    depth = np.asarray(robot_depth, dtype=np.float32)
    object_pixels = np.asarray(object_mask, dtype=bool)
    surface = np.asarray(object_surface_depth, dtype=np.float32)
    shell = np.asarray(shell_m, dtype=np.float32)
    if not (
        robot.shape
        == hand.shape
        == depth.shape
        == object_pixels.shape
        == surface.shape
        == shell.shape
    ):
        raise ValueError("camera-Z barrier inputs must share one shape")
    valid_surface = (
        np.isfinite(surface)
        & (surface > 0.02)
        & (surface < 5.0)
    )
    support = (
        robot
        & hand
        & object_pixels
        & np.isfinite(depth)
        & valid_surface
    )
    # robot_depth is the visible/front XHand surface.  A positive shell tests
    # a conservative back surface, preventing visible breakthrough before the
    # front raster itself has crossed the object surface.
    barrier = support & ((depth + shell) > surface)
    return barrier, support


def temporal_eligibility(
    *,
    hand_mask: np.ndarray,
    robot_depth: np.ndarray,
    object_mask: np.ndarray,
    object_surface_depth: np.ndarray,
    shell_m: np.ndarray,
    front_slack_m: float,
) -> np.ndarray:
    """Allow a bounded gap unless valid depth proves the hand clearly front."""
    if not math.isfinite(front_slack_m) or front_slack_m < 0.0:
        raise ValueError("front slack must be finite and non-negative")
    hand = np.asarray(hand_mask, dtype=bool)
    depth = np.asarray(robot_depth, dtype=np.float32)
    object_pixels = np.asarray(object_mask, dtype=bool)
    surface = np.asarray(object_surface_depth, dtype=np.float32)
    shell = np.asarray(shell_m, dtype=np.float32)
    if not (
        hand.shape
        == depth.shape
        == object_pixels.shape
        == surface.shape
        == shell.shape
    ):
        raise ValueError("temporal barrier inputs must share one shape")
    finite_robot = np.isfinite(depth)
    valid_surface = (
        np.isfinite(surface)
        & (surface > 0.02)
        & (surface < 5.0)
    )
    clearly_front = (
        finite_robot
        & valid_surface
        & ((depth + shell) <= surface - np.float32(front_slack_m))
    )
    return hand & object_pixels & finite_robot & ~clearly_front


def _part_counts(values: np.ndarray) -> dict[str, int]:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != len(PART_NAMES):
        raise ValueError("part count array must have shape (T,6)")
    return {
        name: int(array[:, index].sum())
        for index, name in enumerate(PART_NAMES)
    }


def _paint_evidence(
    frame: np.ndarray,
    raw_added: np.ndarray,
    temporal_added: np.ndarray,
) -> np.ndarray:
    output = np.asarray(frame, dtype=np.uint8).copy()
    raw = np.asarray(raw_added, dtype=bool)
    temporal = np.asarray(temporal_added, dtype=bool)
    if np.any(raw):
        colour = np.asarray((255, 0, 255), dtype=np.float32)
        output[raw] = np.clip(
            output[raw].astype(np.float32) * 0.35 + colour * 0.65,
            0,
            255,
        ).astype(np.uint8)
    if np.any(temporal):
        colour = np.asarray((255, 255, 0), dtype=np.float32)
        output[temporal] = np.clip(
            output[temporal].astype(np.float32) * 0.35 + colour * 0.65,
            0,
            255,
        ).astype(np.uint8)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--raw_video", type=Path, required=True)
    parser.add_argument("--overlay_dir", type=Path, required=True)
    parser.add_argument("--object_mask", type=Path, required=True)
    parser.add_argument(
        "--object_restore_mask",
        type=Path,
        default=None,
        help=(
            "Optional bool (T,H,W) mask selecting raw-video object pixels to "
            "restore. It affects RGB compositing only and defaults to --object_mask."
        ),
    )
    parser.add_argument("--object_surface_depth", type=Path, required=True)
    parser.add_argument("--baseline_mask", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--thumb_shell_m", type=float, default=0.0)
    parser.add_argument("--finger_shell_m", type=float, default=0.0)
    parser.add_argument("--palm_shell_m", type=float, default=0.0)
    parser.add_argument("--temporal_max_gap_frames", type=int, default=0)
    parser.add_argument("--temporal_motion_px", type=int, default=6)
    parser.add_argument("--temporal_front_slack_m", type=float, default=0.015)
    parser.add_argument("--robot_edge_sigma_px", type=float, default=0.6)
    args = parser.parse_args()

    if args.temporal_max_gap_frames < 0:
        parser.error("--temporal_max_gap_frames must be non-negative")
    if args.temporal_motion_px < 0:
        parser.error("--temporal_motion_px must be non-negative")
    for key in ("thumb_shell_m", "finger_shell_m", "palm_shell_m"):
        value = float(getattr(args, key))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{key} must be finite and non-negative")

    background_path = args.background.expanduser().resolve()
    raw_video_path = args.raw_video.expanduser().resolve()
    overlay_dir = args.overlay_dir.expanduser().resolve()
    object_mask_path = args.object_mask.expanduser().resolve()
    object_restore_mask_path = (
        args.object_restore_mask.expanduser().resolve()
        if args.object_restore_mask is not None
        else None
    )
    surface_path = args.object_surface_depth.expanduser().resolve()
    baseline_path = args.baseline_mask.expanduser().resolve()
    output_dir = args.out_dir.expanduser().resolve()
    for path in (
        background_path,
        raw_video_path,
        object_mask_path,
        surface_path,
        baseline_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (
        object_restore_mask_path is not None
        and not object_restore_mask_path.is_file()
    ):
        raise FileNotFoundError(object_restore_mask_path)

    array_paths = {
        "rgb": overlay_dir / "robot_rgb.npy",
        "depth": overlay_dir / "robot_depth.npy",
        "robot": overlay_dir / "robot_mask.npy",
        "hand": overlay_dir / "robot_hand_mask.npy",
        "labels": overlay_dir / "robot_finger_labels.npy",
    }
    for path in array_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    width, height, frame_count, fps = _video_metadata(background_path)
    raw_metadata = _video_metadata(raw_video_path)
    if raw_metadata != (width, height, frame_count, fps):
        raise ValueError("background/raw video metadata mismatch")

    robot_rgb = np.load(array_paths["rgb"], mmap_mode="r", allow_pickle=False)
    robot_depth = np.load(array_paths["depth"], mmap_mode="r", allow_pickle=False)
    robot_mask = np.load(array_paths["robot"], mmap_mode="r", allow_pickle=False)
    robot_hand = np.load(array_paths["hand"], mmap_mode="r", allow_pickle=False)
    finger_labels = np.load(array_paths["labels"], mmap_mode="r", allow_pickle=False)
    object_mask = np.load(object_mask_path, mmap_mode="r", allow_pickle=False)
    explicit_object_restore_mask = (
        np.load(
            object_restore_mask_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        if object_restore_mask_path is not None
        else None
    )
    object_restore_mask = select_object_restore_mask(
        object_mask,
        explicit_object_restore_mask,
    )
    object_surface = np.load(surface_path, mmap_mode="r", allow_pickle=False)
    baseline_mask = np.load(baseline_path, mmap_mode="r", allow_pickle=False)

    overlay_shape = robot_depth.shape
    if robot_rgb.shape != overlay_shape + (3,) or not (
        robot_mask.shape
        == robot_hand.shape
        == finger_labels.shape
        == overlay_shape
    ):
        raise ValueError("overlay array shapes differ")
    if len(overlay_shape) != 3 or overlay_shape[0] != frame_count:
        raise ValueError("overlay frame count differs from the video")
    expected_output_shape = (frame_count, height, width)
    validate_mask_volume(
        object_mask,
        name="object_mask",
        expected_shape=expected_output_shape,
    )
    validate_mask_volume(
        object_restore_mask,
        name="object_restore_mask",
        expected_shape=expected_output_shape,
    )
    if object_surface.shape != expected_output_shape:
        raise ValueError("object surface shape differs from the video")
    if baseline_mask.shape != expected_output_shape or baseline_mask.dtype != np.bool_:
        raise ValueError("baseline mask must be bool with video shape")
    if robot_mask.dtype != np.bool_ or robot_hand.dtype != np.bool_:
        raise TypeError("robot/hand masks must be bool")
    if finger_labels.dtype != np.uint8:
        raise TypeError("finger labels must be uint8")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".xhand_object_barrier.", dir=output_dir.parent)
    )
    atexit.register(shutil.rmtree, staging, ignore_errors=True)
    raw_mask_path = staging / ".raw_occluded_hand_mask.npy"
    raw_buffer = np.lib.format.open_memmap(
        raw_mask_path,
        mode="w+",
        dtype=bool,
        shape=expected_output_shape,
    )
    eligibility_path = staging / ".temporal_eligibility.npy"
    labels_path = staging / ".temporal_hand_labels.npy"
    eligibility_buffer = None
    semantic_buffer = None
    if args.temporal_max_gap_frames > 0:
        eligibility_buffer = np.lib.format.open_memmap(
            eligibility_path,
            mode="w+",
            dtype=bool,
            shape=expected_output_shape,
        )
        semantic_buffer = np.lib.format.open_memmap(
            labels_path,
            mode="w+",
            dtype=np.uint8,
            shape=expected_output_shape,
        )

    raw_candidate_pixels = np.zeros((frame_count, len(PART_NAMES)), dtype=np.int64)
    raw_added_pixels = np.zeros_like(raw_candidate_pixels)
    source_presence = np.zeros((frame_count, len(PART_NAMES)), dtype=bool)
    valid_support_pixels = np.zeros(frame_count, dtype=np.int64)
    for frame_index in range(frame_count):
        (
            _rgb,
            depth,
            robot,
            hand,
            labels,
        ) = resize_overlay_frame(
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
        object_pixels = _resize_mask(object_mask[frame_index], width, height)
        surface = _resize_depth(object_surface[frame_index], width, height)
        barrier, support = compute_visual_barrier(
            robot_mask=robot,
            hand_mask=hand,
            robot_depth=depth,
            object_mask=object_pixels,
            object_surface_depth=surface,
            shell_m=shell,
        )
        baseline = np.asarray(baseline_mask[frame_index], dtype=bool)
        if np.any(baseline & ~hand):
            raise RuntimeError(
                f"baseline finger mask escaped XHand at frame {frame_index}"
            )
        raw = baseline | barrier
        raw_buffer[frame_index] = raw
        valid_support_pixels[frame_index] = int(support.sum())
        for part_index in range(len(PART_NAMES)):
            part = hand_labels == part_index + 1
            raw_candidate_pixels[frame_index, part_index] = int(
                np.sum(barrier & part)
            )
            raw_added_pixels[frame_index, part_index] = int(
                np.sum(barrier & ~baseline & part)
            )
            source_presence[frame_index, part_index] = bool(
                np.any(raw & part)
            )
        if eligibility_buffer is not None:
            assert semantic_buffer is not None
            eligibility_buffer[frame_index] = temporal_eligibility(
                hand_mask=hand,
                robot_depth=depth,
                object_mask=object_pixels,
                object_surface_depth=surface,
                shell_m=shell,
                front_slack_m=args.temporal_front_slack_m,
            )
            semantic_buffer[frame_index] = hand_labels
        if (frame_index + 1) % 100 == 0:
            print(f"[barrier-mask] {frame_index + 1}/{frame_count}", flush=True)
    raw_buffer.flush()
    if eligibility_buffer is not None:
        eligibility_buffer.flush()
    if semantic_buffer is not None:
        semantic_buffer.flush()

    final_mask_path = staging / "occluded_hand_mask.npy"
    temporal_added_pixels = np.zeros_like(raw_candidate_pixels)
    temporal_diagnostics: dict[str, object] = {
        "added_pixels": 0,
        "added_frames": 0,
        "added_frame_fingers": 0,
    }
    if eligibility_buffer is None:
        del raw_buffer
        os.replace(raw_mask_path, final_mask_path)
    else:
        assert semantic_buffer is not None
        final_buffer = np.lib.format.open_memmap(
            final_mask_path,
            mode="w+",
            dtype=bool,
            shape=expected_output_shape,
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
        temporal_added_pixels[:] = np.asarray(
            temporal_diagnostics["added_per_frame_finger"],
            dtype=np.int64,
        )
        final_buffer.flush()
        del final_buffer
        del raw_buffer
        del eligibility_buffer
        del semantic_buffer
        raw_mask_path.unlink(missing_ok=True)
        eligibility_path.unlink(missing_ok=True)
        labels_path.unlink(missing_ok=True)

    final_mask = np.load(final_mask_path, mmap_mode="r", allow_pickle=False)
    final_counts = np.zeros(frame_count, dtype=np.int64)
    residual_violations = np.zeros(frame_count, dtype=np.int64)
    final_added_pixels = np.zeros_like(raw_candidate_pixels)
    raw_capture = cv2.VideoCapture(str(raw_video_path))
    background_capture = cv2.VideoCapture(str(background_path))
    final_writer = _open_writer(
        staging / "video_overlay_hand_barrier.mp4",
        fps,
        (width, height),
    )
    robot_writer = _open_writer(
        staging / "video_robot_only_hand_barrier.mp4",
        fps,
        (width, height),
    )
    debug_writer = _open_writer(
        staging / "debug_hand_barrier.mp4",
        fps,
        (width, height),
    )
    try:
        for frame_index in range(frame_count):
            ok_raw, raw_frame = raw_capture.read()
            ok_background, background = background_capture.read()
            if not ok_raw or not ok_background:
                raise RuntimeError(f"video read failed at frame {frame_index}")
            if raw_frame.shape[:2] != (height, width):
                raw_frame = cv2.resize(
                    raw_frame,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )
            if background.shape[:2] != (height, width):
                background = cv2.resize(
                    background,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )
            (
                rgb,
                depth,
                robot,
                hand,
                labels,
            ) = resize_overlay_frame(
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
            object_pixels = _resize_mask(object_mask[frame_index], width, height)
            restore_pixels = _resize_mask(
                object_restore_mask[frame_index],
                width,
                height,
            )
            surface = _resize_depth(object_surface[frame_index], width, height)
            raw_barrier, _support = compute_visual_barrier(
                robot_mask=robot,
                hand_mask=hand,
                robot_depth=depth,
                object_mask=object_pixels,
                object_surface_depth=surface,
                shell_m=shell,
            )
            baseline = np.asarray(baseline_mask[frame_index], dtype=bool)
            raw_combined = baseline | raw_barrier
            occluded = np.asarray(final_mask[frame_index], dtype=bool)
            if np.any(raw_combined & ~occluded):
                raise RuntimeError(
                    f"temporal barrier removed raw evidence at frame {frame_index}"
                )
            if np.any(occluded & ~hand):
                raise RuntimeError(
                    f"barrier escaped the XHand mask at frame {frame_index}"
                )
            residual = raw_barrier & ~occluded
            residual_violations[frame_index] = int(residual.sum())
            if residual_violations[frame_index]:
                raise RuntimeError(
                    f"camera-Z barrier left {residual.sum()} violations at "
                    f"frame {frame_index}"
                )
            final_counts[frame_index] = int(occluded.sum())
            added = occluded & ~baseline
            for part_index in range(len(PART_NAMES)):
                final_added_pixels[frame_index, part_index] = int(
                    np.sum(added & (hand_labels == part_index + 1))
                )

            # Restore only the selected source-object pixels.  This may be a
            # conservative subset of the wider mask used for barrier support.
            # Robot pixels are then drawn except where the barrier removes them.
            composite_background = restore_raw_object_pixels(
                background,
                raw_frame,
                restore_pixels,
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
                _paint_evidence(
                    final,
                    raw_barrier & ~baseline,
                    occluded & ~raw_combined,
                )
            )
            if (frame_index + 1) % 100 == 0:
                print(
                    f"[barrier-composite] {frame_index + 1}/{frame_count} "
                    f"hidden={final_counts[frame_index]}",
                    flush=True,
                )
    finally:
        raw_capture.release()
        background_capture.release()
        final_writer.release()
        robot_writer.release()
        debug_writer.release()

    if np.any(residual_violations):
        raise RuntimeError("final camera-Z barrier has residual violations")
    baseline_total = int(np.asarray(baseline_mask, dtype=bool).sum())
    final_total = int(final_counts.sum())
    if final_total < baseline_total:
        raise RuntimeError("whole-hand barrier removed baseline occlusion")
    np.savez(
        staging / "barrier_evidence.npz",
        part_names=np.asarray(PART_NAMES),
        raw_candidate_pixels=raw_candidate_pixels,
        raw_added_pixels=raw_added_pixels,
        temporal_added_pixels=temporal_added_pixels,
        final_added_pixels=final_added_pixels,
        valid_support_pixels=valid_support_pixels,
        residual_violation_pixels=residual_violations,
    )
    report = {
        "schema_version": 1,
        "method": "visual_camera_z_xhand_barrier",
        "representation": "visible_camera_z_height_field",
        "watertight_object": False,
        "pose_state_modified": False,
        "metric_collision_guarantee": False,
        "scope": "visible XHand including palm/base and all five fingers; RB5 excluded",
        "frames": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "config": {
            "thumb_shell_m": float(args.thumb_shell_m),
            "finger_shell_m": float(args.finger_shell_m),
            "palm_shell_m": float(args.palm_shell_m),
            "temporal_max_gap_frames": int(args.temporal_max_gap_frames),
            "temporal_motion_px": int(args.temporal_motion_px),
            "temporal_front_slack_m": float(args.temporal_front_slack_m),
            "robot_edge_sigma_px": float(args.robot_edge_sigma_px),
            "object_restore_mask_explicit": bool(
                object_restore_mask_path is not None
            ),
        },
        "sources": {
            "background": str(background_path),
            "raw_video": str(raw_video_path),
            "overlay_dir": str(overlay_dir),
            "object_mask": str(object_mask_path),
            "object_restore_mask": str(
                object_restore_mask_path or object_mask_path
            ),
            "object_surface_depth": str(surface_path),
            "baseline_mask": str(baseline_path),
        },
        "counts": {
            "baseline_occluded_pixels": baseline_total,
            "raw_barrier_candidate_pixels": int(raw_candidate_pixels.sum()),
            "raw_barrier_added_pixels": int(raw_added_pixels.sum()),
            "temporal_added_pixels": int(temporal_added_pixels.sum()),
            "final_occluded_pixels": final_total,
            "final_frames_with_occlusion": int((final_counts > 0).sum()),
            "valid_comparison_support_pixels": int(valid_support_pixels.sum()),
            "residual_violation_pixels": int(residual_violations.sum()),
            "raw_candidate_by_part": _part_counts(raw_candidate_pixels),
            "raw_added_by_part": _part_counts(raw_added_pixels),
            "temporal_added_by_part": _part_counts(temporal_added_pixels),
            "final_added_by_part": _part_counts(final_added_pixels),
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
            "baseline_subset_final": True,
            "final_occlusion_subset_of_xhand": True,
            "rb5_arm_excluded": True,
            "barrier_support_uses_object_mask_only": True,
            "raw_rgb_restore_uses_object_restore_mask_only": True,
            "object_restore_defaults_to_object_mask_when_omitted": True,
            "valid_surface_barrier_residual_is_zero": True,
            "unknown_surface_fails_open_except_existing_baseline_or_bounded_temporal": True,
            "trajectory_arrays_unchanged": True,
            "auxiliary_geometry_used": False,
        },
        "outputs": {
            "final_video": "video_overlay_hand_barrier.mp4",
            "robot_only_video": "video_robot_only_hand_barrier.mp4",
            "debug_video": "debug_hand_barrier.mp4",
            "mask": "occluded_hand_mask.npy",
            "evidence": "barrier_evidence.npz",
        },
        "provenance_warning": (
            "The object surface is HaWoR-Z anchored monocular depth with "
            "uncalibrated phone intrinsics. This guarantees only MH-view "
            "visual non-emergence where the visible surface is valid; it is "
            "not a physical 3D collision solve."
        ),
    }
    (staging / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    publish_directory(str(staging), str(output_dir))
    print(
        f"[ok] XHand camera-Z barrier: {output_dir} "
        f"pixels={final_total} frames={(final_counts > 0).sum()}/{frame_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
