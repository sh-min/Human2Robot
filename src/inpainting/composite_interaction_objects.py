"""Composite a robot with source-RGB object occlusion in a 2.5D layer stack.

Layer order, from back to front::

    inpainted background -> behind-MCP robot -> source object -> front-MCP robot

The important difference from the old cube compositor is that the object layer
uses pixels from the *source RGB*, not from the inpainted background.  This
prevents inpaint damage at hand-object contact from being copied into the final
shot while still letting front-side robot fingers wrap over the object.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from contact_shadow import contact_shadow_alpha, fit_support_plane


def _video_info(capture: cv2.VideoCapture) -> tuple[int, int, float]:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    return width, height, fps


def _alpha(mask: np.ndarray, sigma: float,
           mode: str = "gaussian",
           support: np.ndarray | None = None) -> np.ndarray:
    value = mask.astype(np.float32)
    if sigma > 0 and mode == "gaussian":
        value = cv2.GaussianBlur(value, (0, 0), sigma)
    elif sigma > 0 and mode == "clamped":
        value = cv2.GaussianBlur(value, (0, 0), sigma)
        if support is None:
            raise ValueError("clamped alpha requires a support mask")
        # Preserve Gaussian antialiasing on the rendered side of the edge, but
        # never let alpha reach RGB pixels outside the valid robot raster.
        value[~np.asarray(support, dtype=bool)] = 0.0
    elif sigma > 0 and mode == "inside":
        # Robot RGB is only defined inside its raster mask.  Blurring a binary
        # robot mask outwards mixes the near-black, undefined pixels outside
        # the render into the scene and creates a dark moving halo.  A distance
        # transform gives us the same sub-pixel softening on the *inside* while
        # keeping alpha exactly zero outside the rendered robot support.
        distance = cv2.distanceTransform(
            mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_3
        )
        value = np.clip(distance / max(1.0, 1.08 * sigma), 0.0, 1.0)
    elif mode not in {"gaussian", "clamped", "inside"}:
        raise ValueError(f"unknown alpha mode: {mode}")
    return np.clip(value, 0.0, 1.0)[..., None]


def _blend(acc: np.ndarray, content: np.ndarray, mask: np.ndarray, sigma: float,
           mode: str = "gaussian",
           support: np.ndarray | None = None) -> np.ndarray:
    alpha = _alpha(mask, sigma, mode, support)
    return alpha * content.astype(np.float32) + (1.0 - alpha) * acc


def _fill_and_smooth_depth(joints_left: np.ndarray,
                           joints_right: np.ndarray,
                           valid: np.ndarray,
                           frame_count: int,
                           joint: int,
                           sigma: float) -> np.ndarray:
    z = np.full(frame_count, np.nan, dtype=np.float32)
    for idx in range(frame_count):
        values = []
        if idx < joints_left.shape[0] and valid[0, idx]:
            values.append(float(joints_left[idx, joint, 2]))
        if idx < joints_right.shape[0] and valid[1, idx]:
            values.append(float(joints_right[idx, joint, 2]))
        if values:
            z[idx] = float(np.mean(values))
    good = np.flatnonzero(np.isfinite(z))
    if not len(good):
        return np.full(frame_count, np.inf, dtype=np.float32)
    z = np.interp(np.arange(frame_count), good, z[good]).astype(np.float32)
    if sigma > 0:
        radius = max(1, int(np.ceil(3 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        z = np.convolve(np.pad(z, (radius, radius), mode="edge"), kernel,
                        mode="valid").astype(np.float32)
    return z


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_demo", type=Path, required=True)
    parser.add_argument("--hawor_npz", type=Path, required=True)
    parser.add_argument("--source_video", default="video_L.mp4")
    parser.add_argument(
        "--object_source_video", default=None,
        help="Optional source video containing completed object texture. The "
             "ordinary source video remains the geometry/timing reference.",
    )
    parser.add_argument("--background_video",
                        default="inpaint_processor/video_human_inpaint.mkv")
    parser.add_argument("--object_mask", default="interaction_objects/object_mask.npy")
    parser.add_argument("--robot_dir", default="overlay_processor",
                        help="Directory under --processed_demo containing "
                             "robot_rgb.npy, robot_depth.npy, and robot_mask.npy.")
    parser.add_argument("--force_front_mask", default=None,
                        help="Optional source-RGB mask drawn after the complete robot "
                             "layer, used to prevent visible rigid-object interiors "
                             "from being penetrated by robot links.")
    parser.add_argument(
        "--force_robot_behind_mask",
        default=None,
        help="Optional robot mask forced into the behind-object layer.",
    )
    parser.add_argument(
        "--thumb_mask",
        default=None,
        help="Optional semantic thumb mask kept in front on forced-behind frames.",
    )
    parser.add_argument(
        "--contact_split_depth",
        default=None,
        help="Optional per-pixel object/contact depth split array.",
    )
    parser.add_argument("--output",
                        default="interaction_objects/video_overlay_object_occlusion.mp4")
    parser.add_argument("--threshold_joint", type=int, default=5)
    parser.add_argument("--depth_bias", type=float, default=0.0)
    parser.add_argument("--depth_sigma", type=float, default=8.0)
    parser.add_argument("--robot_edge_sigma", type=float, default=1.2)
    parser.add_argument(
        "--robot_edge_mode", choices=("clamped", "inside", "gaussian"),
        default="clamped",
        help="clamped keeps Gaussian antialiasing only inside the valid robot "
             "raster; inside uses a distance feather; gaussian is the legacy "
             "outward blur that can create a dark halo over undefined RGB.",
    )
    parser.add_argument("--object_edge_sigma", type=float, default=0.8)
    parser.add_argument("--force_front_edge_sigma", type=float, default=0.45)
    parser.add_argument(
        "--shadow_depth", default=None,
        help="Optional metric scene-depth array for the accepted height-band shadow.",
    )
    parser.add_argument("--shadow_opacity", type=float, default=0.50)
    parser.add_argument("--shadow_blur", type=float, default=5.0)
    parser.add_argument("--shadow_bands", type=int, default=5)
    parser.add_argument("--shadow_penumbra", type=float, default=70.0)
    parser.add_argument("--shadow_falloff", type=float, default=0.30)
    args = parser.parse_args()

    processed = args.processed_demo
    source_cap = cv2.VideoCapture(str(processed / args.source_video))
    object_source_cap = (
        cv2.VideoCapture(str(processed / args.object_source_video))
        if args.object_source_video is not None else None
    )
    background_cap = cv2.VideoCapture(str(processed / args.background_video))
    if (not source_cap.isOpened() or not background_cap.isOpened()
            or (object_source_cap is not None
                and not object_source_cap.isOpened())):
        raise RuntimeError("cannot open source/background video")
    width, height, fps = _video_info(source_cap)

    robot_dir = processed / args.robot_dir
    robot_rgb = np.load(robot_dir / "robot_rgb.npy", mmap_mode="r")
    robot_depth = np.load(robot_dir / "robot_depth.npy", mmap_mode="r")
    robot_mask = np.load(robot_dir / "robot_mask.npy", mmap_mode="r")
    object_mask = np.load(processed / args.object_mask, mmap_mode="r")
    force_front = (np.load(processed / args.force_front_mask, mmap_mode="r")
                   if args.force_front_mask is not None else None)
    force_robot_behind = (
        np.load(processed / args.force_robot_behind_mask, mmap_mode="r")
        if args.force_robot_behind_mask is not None else None
    )
    thumb_mask = (
        np.load(processed / args.thumb_mask, mmap_mode="r")
        if args.thumb_mask is not None else None
    )
    contact_split = (
        np.load(processed / args.contact_split_depth, mmap_mode="r")
        if args.contact_split_depth is not None else None
    )
    scene_depth = (
        np.load(processed / args.shadow_depth, mmap_mode="r")
        if args.shadow_depth is not None else None
    )
    pose = np.load(args.hawor_npz)
    frame_count = min(
        len(robot_rgb), len(robot_depth), len(robot_mask), len(object_mask),
        int(source_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(background_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        pose["joints_left"].shape[0],
    )
    if force_front is not None:
        frame_count = min(frame_count, len(force_front))
    if force_robot_behind is not None:
        frame_count = min(frame_count, len(force_robot_behind))
    if thumb_mask is not None:
        frame_count = min(frame_count, len(thumb_mask))
    if contact_split is not None:
        frame_count = min(frame_count, len(contact_split))
    if scene_depth is not None:
        frame_count = min(frame_count, len(scene_depth))
    if object_source_cap is not None:
        frame_count = min(
            frame_count,
            int(object_source_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    z_split = _fill_and_smooth_depth(
        pose["joints_left"], pose["joints_right"], pose["valid"], frame_count,
        args.threshold_joint, args.depth_sigma,
    )
    shadow_plane = None
    focal = float(pose["img_focal"])

    output = processed / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create output: {output}")

    object_overlap = 0
    front_pixels = 0
    behind_pixels = 0
    forced_pixels = 0
    for idx in range(frame_count):
        ok_source, source = source_cap.read()
        if object_source_cap is not None:
            ok_object_source, object_source = object_source_cap.read()
        else:
            ok_object_source, object_source = ok_source, source
        ok_bg, background = background_cap.read()
        if not ok_source or not ok_object_source or not ok_bg:
            raise RuntimeError(f"video decode stopped at frame {idx}")
        if source.shape[:2] != (height, width):
            source = cv2.resize(source, (width, height), interpolation=cv2.INTER_LINEAR)
        if background.shape[:2] != (height, width):
            background = cv2.resize(background, (width, height), interpolation=cv2.INTER_LINEAR)
        if object_source.shape[:2] != (height, width):
            object_source = cv2.resize(
                object_source, (width, height), interpolation=cv2.INTER_LINEAR
            )

        visible_robot = np.asarray(robot_mask[idx], dtype=bool)
        depth = np.asarray(robot_depth[idx], dtype=np.float32) + args.depth_bias
        split = (
            np.asarray(contact_split[idx], dtype=np.float32)
            if contact_split is not None else z_split[idx]
        )
        behind = visible_robot & (depth >= split)
        front = visible_robot & (depth < split)
        if force_robot_behind is not None:
            forced_behind = (
                visible_robot
                & np.asarray(force_robot_behind[idx], dtype=bool)
            )
            behind |= forced_behind
            front &= ~forced_behind
            if thumb_mask is not None and forced_behind.any():
                thumb_front = (
                    visible_robot & np.asarray(thumb_mask[idx], dtype=bool)
                )
                behind &= ~thumb_front
                front |= thumb_front
        obj = np.asarray(object_mask[idx], dtype=bool)

        if scene_depth is not None:
            depth_frame = np.asarray(scene_depth[idx], dtype=np.float32)
            shadow_plane = fit_support_plane(
                depth_frame, visible_robot, focal, width / 2.0, height / 2.0,
                prev=shadow_plane,
            )
            shadow = contact_shadow_alpha(
                depth_frame, np.asarray(robot_depth[idx], dtype=np.float32),
                visible_robot, shadow_plane, focal, width / 2.0, height / 2.0,
                opacity=args.shadow_opacity, blur=args.shadow_blur,
                bands=args.shadow_bands, penumbra=args.shadow_penumbra,
                falloff=args.shadow_falloff,
            )
            acc = background.astype(np.float32) * (1.0 - shadow[..., None])
        else:
            acc = background.astype(np.float32)
        acc = _blend(
            acc, np.asarray(robot_rgb[idx]), behind, args.robot_edge_sigma,
            args.robot_edge_mode, visible_robot,
        )
        acc = _blend(acc, object_source, obj, args.object_edge_sigma)
        acc = _blend(
            acc, np.asarray(robot_rgb[idx]), front, args.robot_edge_sigma,
            args.robot_edge_mode, visible_robot,
        )
        if force_front is not None:
            forced = np.asarray(force_front[idx], dtype=bool)
            acc = _blend(acc, object_source, forced, args.force_front_edge_sigma)
            forced_pixels += int((forced & visible_robot).sum())
        writer.write(np.clip(acc, 0, 255).astype(np.uint8))

        object_overlap += int((obj & visible_robot).sum())
        front_pixels += int(front.sum())
        behind_pixels += int(behind.sum())
        if (idx + 1) % 100 == 0:
            print(f"[frame] {idx + 1}/{frame_count}")

    source_cap.release()
    if object_source_cap is not None:
        object_source_cap.release()
    background_cap.release()
    writer.release()
    print(f"[ok] wrote {output}")
    print(f"[info] object/robot overlap={object_overlap} px, "
          f"front robot={front_pixels} px, behind robot={behind_pixels} px, "
          f"forced-object-over-robot={forced_pixels} px")


if __name__ == "__main__":
    main()
