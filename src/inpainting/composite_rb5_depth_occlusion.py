"""Sensor-depth-only finger occlusion for an RB5 + XHand overlay.

This compositor is the geometry-only counterpart to
``composite_rb5_contact_occlusion.py``.  It intentionally does not read HaCo
scores or projected HaCo contact points.  A rendered robot-finger pixel is
hidden only when it overlaps the verified modal object mask and lies farther
from the camera than the robust sensor-depth estimate of that object.

The raw visible object is restored over the hand-inpainted background before
the robot is composited, matching the HaCo and ensemble compositors.  Missing
or ambiguous object depth fails open and leaves the robot visible.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from atomic_directory_publish import publish_directory
from composite_rb5_contact_occlusion import (
    FINGER_NAMES,
    _open_writer,
    _resize_mask,
    _resize_overlay_frame,
    _true_runs,
    _video_metadata,
    composite_frame,
    estimate_object_depth_track,
    suppress_short_runs,
)


def compute_depth_only_occlusion(
    *,
    robot_mask: np.ndarray,
    finger_mask: np.ndarray,
    robot_depth: np.ndarray,
    depth_object_mask: np.ndarray,
    object_depth_m: float,
    depth_margin_m: float,
) -> np.ndarray:
    """Return finger pixels confidently behind the sensor-depth object."""
    robot = np.asarray(robot_mask, dtype=bool)
    fingers = np.asarray(finger_mask, dtype=bool)
    depth = np.asarray(robot_depth, dtype=np.float32)
    object_pixels = np.asarray(depth_object_mask, dtype=bool)
    if not (robot.shape == fingers.shape == depth.shape == object_pixels.shape):
        raise ValueError("depth-only occlusion inputs must share one shape")
    if depth_margin_m < 0:
        raise ValueError("depth_margin_m must be non-negative")
    if not np.isfinite(object_depth_m):
        return np.zeros_like(robot)
    return (
        robot
        & fingers
        & object_pixels
        & np.isfinite(depth)
        & (depth > float(object_depth_m) + float(depth_margin_m))
    )


def _debug_grid(
    raw: np.ndarray,
    object_mask: np.ndarray,
    robot_rgb: np.ndarray,
    robot_mask: np.ndarray,
    finger_mask: np.ndarray,
    occluded: np.ndarray,
    final: np.ndarray,
    object_depth_m: float,
) -> np.ndarray:
    """Build a compact 2x2 diagnostic frame at the output resolution."""
    height, width = raw.shape[:2]
    object_panel = raw.copy()
    object_panel[object_mask] = (
        0.35 * object_panel[object_mask]
        + 0.65 * np.array([0, 255, 255])
    ).astype(np.uint8)

    decision_panel = raw.copy()
    decision_panel[finger_mask] = (
        0.55 * decision_panel[finger_mask]
        + 0.45 * np.array([255, 180, 0])
    ).astype(np.uint8)
    decision_panel[occluded] = (0, 0, 255)

    robot_panel = np.zeros_like(raw)
    robot_bgr = np.asarray(robot_rgb, dtype=np.uint8)[..., ::-1]
    visible = np.asarray(robot_mask, dtype=bool) & ~occluded
    robot_panel[visible] = robot_bgr[visible]
    robot_panel[occluded] = (0, 0, 255)

    depth_text = (
        f"object z={object_depth_m:.3f}m"
        if np.isfinite(object_depth_m)
        else "object z=missing"
    )
    labels = (
        (object_panel, "depth-coherent object"),
        (decision_panel, f"depth decision | {depth_text}"),
        (robot_panel, "visible / hidden"),
        (final, "depth-only final"),
    )
    panels: list[np.ndarray] = []
    for panel, label in labels:
        resized = cv2.resize(
            panel,
            (width // 2, height // 2),
            interpolation=cv2.INTER_AREA,
        )
        cv2.putText(
            resized,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(resized)
    return np.concatenate(
        [
            np.concatenate(panels[:2], axis=1),
            np.concatenate(panels[2:], axis=1),
        ],
        axis=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_demo", type=Path, required=True)
    parser.add_argument("--episode_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--background", type=Path, default=None)
    parser.add_argument("--raw_video", type=Path, default=None)
    parser.add_argument("--overlay_dir", type=Path, default=None)
    parser.add_argument("--object_mask", type=Path, default=None)
    parser.add_argument("--object_depth_mask", type=Path, default=None)
    parser.add_argument("--scene_depth", type=Path, default=None)
    parser.add_argument("--depth_margin_m", type=float, default=0.030)
    parser.add_argument("--object_depth_erode_px", type=int, default=18)
    parser.add_argument("--min_occlusion_run_frames", type=int, default=2)
    parser.add_argument("--robot_edge_sigma_px", type=float, default=0.6)
    args = parser.parse_args()

    if args.depth_margin_m < 0:
        parser.error("--depth_margin_m must be non-negative")
    if args.object_depth_erode_px < 0:
        parser.error("--object_depth_erode_px must be non-negative")
    if args.min_occlusion_run_frames <= 0:
        parser.error("--min_occlusion_run_frames must be positive")

    processed = args.processed_demo.resolve()
    episode = args.episode_dir.resolve()
    output_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else processed / "depth_occlusion"
    )
    background_path = (
        args.background.resolve()
        if args.background is not None
        else processed / "inpaint_processor" / "video_human_inpaint.mkv"
    )
    raw_video_path = (
        args.raw_video.resolve()
        if args.raw_video is not None
        else processed / "video_L.mp4"
    )
    overlay_dir = (
        args.overlay_dir.resolve()
        if args.overlay_dir is not None
        else processed / "overlay_processor"
    )
    object_mask_path = (
        args.object_mask.resolve()
        if args.object_mask is not None
        else processed / "object_layer" / "object_mask_modal.npy"
    )
    default_depth_mask = (
        processed / "object_layer" / "object_depth_mask_sensor.npy"
    )
    object_depth_mask_path = (
        args.object_depth_mask.resolve()
        if args.object_depth_mask is not None
        else (
            default_depth_mask
            if default_depth_mask.is_file()
            else object_mask_path
        )
    )
    scene_depth_path = (
        args.scene_depth.resolve()
        if args.scene_depth is not None
        else processed / "depth_processor" / "depth_sensor_aligned.npy"
    )

    width, height, frame_count, fps = _video_metadata(background_path)
    raw_width, raw_height, raw_frames, raw_fps = _video_metadata(raw_video_path)
    if raw_frames != frame_count or not np.isclose(raw_fps, fps, atol=0.1):
        raise ValueError(
            "raw/background video mismatch: "
            f"frames {raw_frames}/{frame_count}, fps {raw_fps}/{fps}"
        )
    if not np.isclose(
        width / raw_width,
        height / raw_height,
        atol=1e-3,
    ):
        raise ValueError("raw/background resize must preserve aspect ratio")

    robot_rgb = np.load(overlay_dir / "robot_rgb.npy", mmap_mode="r")
    robot_depth = np.load(overlay_dir / "robot_depth.npy", mmap_mode="r")
    robot_mask = np.load(overlay_dir / "robot_mask.npy", mmap_mode="r")
    finger_labels = np.load(
        overlay_dir / "robot_finger_labels.npy",
        mmap_mode="r",
    )
    if robot_mask.ndim != 3:
        raise ValueError(f"robot_mask must be (T,H,W), got {robot_mask.shape}")
    overlay_height, overlay_width = robot_mask.shape[1:3]
    expected_shapes = {
        "robot_rgb": (frame_count, overlay_height, overlay_width, 3),
        "robot_depth": (frame_count, overlay_height, overlay_width),
        "robot_mask": (frame_count, overlay_height, overlay_width),
        "robot_finger_labels": (
            frame_count,
            overlay_height,
            overlay_width,
        ),
    }
    for name, array in (
        ("robot_rgb", robot_rgb),
        ("robot_depth", robot_depth),
        ("robot_mask", robot_mask),
        ("robot_finger_labels", finger_labels),
    ):
        if array.shape != expected_shapes[name]:
            raise ValueError(
                f"{name} shape mismatch: {array.shape} != "
                f"{expected_shapes[name]}"
            )
    if not np.isclose(
        overlay_width / overlay_height,
        width / height,
        atol=1e-3,
    ):
        raise ValueError(
            "Isaac/background aspect mismatch: "
            f"{overlay_width}x{overlay_height} vs {width}x{height}"
        )

    object_mask = np.load(object_mask_path, mmap_mode="r")
    object_depth_mask = np.load(object_depth_mask_path, mmap_mode="r")
    scene_depth = np.load(scene_depth_path, mmap_mode="r")
    for name, array in (
        ("object_mask", object_mask),
        ("object_depth_mask", object_depth_mask),
        ("scene_depth", scene_depth),
    ):
        if array.ndim != 3 or len(array) != frame_count:
            raise ValueError(
                f"{name} must have {frame_count} (T,H,W) frames, "
                f"got {array.shape}"
            )

    object_depth_track = estimate_object_depth_track(
        scene_depth,
        object_depth_mask,
        output_shape=(height, width),
        erode_px=args.object_depth_erode_px,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".depth_occlusion.",
            dir=output_dir.parent,
        )
    )
    atexit.register(shutil.rmtree, staging, ignore_errors=True)
    occluded_buffer = np.lib.format.open_memmap(
        staging / "occluded_finger_mask.npy",
        mode="w+",
        dtype=bool,
        shape=(frame_count, height, width),
    )
    candidate_presence = np.zeros(
        (frame_count, len(FINGER_NAMES)),
        dtype=bool,
    )

    for frame_index in range(frame_count):
        depth_object = _resize_mask(
            object_depth_mask[frame_index],
            width,
            height,
        )
        (
            _frame_robot_rgb,
            frame_robot_depth,
            frame_robot_mask,
            frame_finger_mask,
            frame_finger_labels,
        ) = _resize_overlay_frame(
            robot_rgb[frame_index],
            robot_depth[frame_index],
            robot_mask[frame_index],
            finger_labels[frame_index],
            width=width,
            height=height,
        )
        object_z = float(object_depth_track[frame_index])
        frame_occluded = compute_depth_only_occlusion(
            robot_mask=frame_robot_mask,
            finger_mask=frame_finger_mask,
            robot_depth=frame_robot_depth,
            depth_object_mask=depth_object,
            object_depth_m=object_z,
            depth_margin_m=args.depth_margin_m,
        )
        for finger_index in range(len(FINGER_NAMES)):
            candidate_presence[frame_index, finger_index] = bool(
                np.any(
                    frame_occluded
                    & (frame_finger_labels == finger_index + 1)
                )
            )
        if np.any(frame_occluded & ~frame_finger_mask):
            raise RuntimeError(
                f"non-finger occlusion invariant failed at {frame_index}"
            )
        occluded_buffer[frame_index] = frame_occluded
        if (frame_index + 1) % 100 == 0:
            print(
                f"[depth-mask] {frame_index + 1}/{frame_count}",
                flush=True,
            )
    occluded_buffer.flush()

    stable_presence = suppress_short_runs(
        candidate_presence,
        min_frames=args.min_occlusion_run_frames,
    )
    occluded_counts = np.zeros(frame_count, dtype=np.int64)
    for frame_index in range(frame_count):
        frame_occluded = np.asarray(
            occluded_buffer[frame_index],
            dtype=bool,
        ).copy()
        _, _, _, _, labels = _resize_overlay_frame(
            robot_rgb[frame_index],
            robot_depth[frame_index],
            robot_mask[frame_index],
            finger_labels[frame_index],
            width=width,
            height=height,
        )
        for finger_index in range(len(FINGER_NAMES)):
            if not stable_presence[frame_index, finger_index]:
                frame_occluded[labels == finger_index + 1] = False
        occluded_buffer[frame_index] = frame_occluded
        occluded_counts[frame_index] = int(frame_occluded.sum())
    occluded_buffer.flush()

    final_writer = _open_writer(
        staging / "video_overlay_depth.mp4",
        fps,
        (width, height),
    )
    robot_writer = _open_writer(
        staging / "video_robot_only_depth.mp4",
        fps,
        (width, height),
    )
    debug_writer = _open_writer(
        staging / "debug_depth_occlusion.mp4",
        fps,
        (width, height),
    )
    raw_capture = cv2.VideoCapture(str(raw_video_path))
    bg_capture = cv2.VideoCapture(str(background_path))
    raw_object_counts = np.zeros(frame_count, dtype=np.int64)
    try:
        for frame_index in range(frame_count):
            ok_raw, raw = raw_capture.read()
            ok_bg, background = bg_capture.read()
            if not ok_raw or not ok_bg:
                raise RuntimeError(
                    f"video read failed during composite pass at {frame_index}"
                )
            if raw.shape[:2] != (height, width):
                raw = cv2.resize(
                    raw,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )
            core_object = _resize_mask(
                object_mask[frame_index],
                width,
                height,
            )
            depth_object = _resize_mask(
                object_depth_mask[frame_index],
                width,
                height,
            )
            (
                frame_robot_rgb,
                _frame_robot_depth,
                frame_robot_mask,
                frame_finger_mask,
                _frame_finger_labels,
            ) = _resize_overlay_frame(
                robot_rgb[frame_index],
                robot_depth[frame_index],
                robot_mask[frame_index],
                finger_labels[frame_index],
                width=width,
                height=height,
            )
            frame_occluded = np.asarray(
                occluded_buffer[frame_index],
                dtype=bool,
            )
            composite_background = background.copy()
            composite_background[core_object] = raw[core_object]
            raw_object_counts[frame_index] = int(core_object.sum())
            final, robot_only, _ = composite_frame(
                composite_background,
                frame_robot_rgb,
                frame_robot_mask,
                frame_finger_mask,
                frame_occluded,
                robot_edge_sigma_px=args.robot_edge_sigma_px,
                occlusion_edge_sigma_px=0.0,
            )
            final_writer.write(final)
            robot_writer.write(robot_only)
            debug_writer.write(
                _debug_grid(
                    raw,
                    depth_object,
                    frame_robot_rgb,
                    frame_robot_mask,
                    frame_finger_mask,
                    frame_occluded,
                    final,
                    float(object_depth_track[frame_index]),
                )
            )
            if (frame_index + 1) % 100 == 0:
                print(
                    f"[composite] {frame_index + 1}/{frame_count} "
                    f"hidden_px={occluded_counts[frame_index]}",
                    flush=True,
                )
    finally:
        raw_capture.release()
        bg_capture.release()
        final_writer.release()
        robot_writer.release()
        debug_writer.release()
        occluded_buffer.flush()
        del occluded_buffer

    report = {
        "schema_version": 1,
        "occlusion_mode": "depth",
        "frames": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "config": {
            "depth_margin_m": args.depth_margin_m,
            "object_depth_erode_px": args.object_depth_erode_px,
            "min_occlusion_run_frames": args.min_occlusion_run_frames,
            "robot_edge_sigma_px": args.robot_edge_sigma_px,
        },
        "sources": {
            "processed_demo": str(processed),
            "episode_dir": str(episode),
            "background": str(background_path),
            "raw_video": str(raw_video_path),
            "overlay_dir": str(overlay_dir),
            "object_mask": str(object_mask_path),
            "object_depth_mask": str(object_depth_mask_path),
            "scene_depth": str(scene_depth_path),
        },
        "finger_names": list(FINGER_NAMES),
        "candidate_occlusion_runs": {
            finger: _true_runs(candidate_presence[:, index])
            for index, finger in enumerate(FINGER_NAMES)
        },
        "stable_occlusion_runs": {
            finger: _true_runs(stable_presence[:, index])
            for index, finger in enumerate(FINGER_NAMES)
        },
        "active_frame_count": {
            finger: int(stable_presence[:, index].sum())
            for index, finger in enumerate(FINGER_NAMES)
        },
        "suppressed_short_finger_frames": int(
            candidate_presence.sum() - stable_presence.sum()
        ),
        "object_depth_m": [
            float(value) if np.isfinite(value) else None
            for value in object_depth_track
        ],
        "valid_object_depth_frames": int(
            np.isfinite(object_depth_track).sum()
        ),
        "occluded_pixel_count": occluded_counts.tolist(),
        "occluded_pixels_total": int(occluded_counts.sum()),
        "frames_with_occlusion": int((occluded_counts > 0).sum()),
        "raw_object_pixel_count": raw_object_counts.tolist(),
        "raw_object_pixels_total": int(raw_object_counts.sum()),
        "invariants": {
            "haco_used_for_occlusion_decision": False,
            "occluded_subset_of_robot_fingers": True,
            "explicit_object_mask_must_be_modal": True,
            "missing_object_depth_fails_open": True,
        },
        "compositing_order": (
            "inpainted_background_then_raw_object_then_depth_occluded_robot"
        ),
    }
    (staging / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    publish_directory(str(staging), str(output_dir))
    print(f"[ok] depth-only overlay: {output_dir}", flush=True)
    print(
        f"[info] mode=depth, "
        f"occluded pixels={int(occluded_counts.sum())}, "
        f"frames={int((occluded_counts > 0).sum())}/{frame_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
