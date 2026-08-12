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

from layered_compositor import (
    STAGE_SPECS,
    FrameInputs,
    StageConfig,
    compose_frame,
)
from layered_compositor.visualization import (
    checkerboard,
    context_layer,
    grid_3x2,
    isolated_layer,
    label,
)
from layered_compositor.video import CompatibleVideoWriter


def _video_info(capture: cv2.VideoCapture) -> tuple[int, int, float]:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    return width, height, fps


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
        "--force_robot_front_mask", default=None,
        help="Optional robot-part mask that is always classified as the front "
             "robot layer. It is also carved out of --force_front_mask so the "
             "forced object layer cannot cover it.",
    )
    parser.add_argument(
        "--behind_robot_object_mask", default=None,
        help="Optional mask of object pixels that lie behind the robot, such as "
             "a static table object the hand only passes over. It is carved out "
             "of both object layers so the robot cannot be penetrated by an "
             "object the frame's single depth split does not describe.",
    )
    parser.add_argument(
        "--force_robot_front_dilate", type=int, default=0,
        help="Dilate the forced-front robot mask by this many pixels, then "
             "clip it to the rendered robot support. A small value closes "
             "one-pixel object seams around the thumb.",
    )
    parser.add_argument(
        "--layer_output_dir", default=None,
        help="Optional directory for isolated background/robot/object layer "
             "videos and a six-panel visualization.",
    )
    parser.add_argument(
        "--layer_context_videos", action="store_true",
        help="Also write sparse layers over a 20%%-brightness scene reference "
             "so their position and edge quality are easier to inspect.",
    )
    parser.add_argument(
        "--progressive_output_dir", default=None,
        help="Optional directory for six cumulative videos that show the "
             "composite being assembled one layer at a time, plus a six-panel "
             "cumulative comparison.",
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
        "--video_codec", choices=("h264", "mp4v"), default="h264",
        help="Output codec. h264 is the default because VS Code and browser "
             "previews generally cannot decode OpenCV's mp4v output.",
    )
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
    force_robot_front = (
        np.load(processed / args.force_robot_front_mask, mmap_mode="r")
        if args.force_robot_front_mask is not None else None
    )
    behind_robot_object = (
        np.load(processed / args.behind_robot_object_mask, mmap_mode="r")
        if args.behind_robot_object_mask is not None else None
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
    if force_robot_front is not None:
        frame_count = min(frame_count, len(force_robot_front))
    if behind_robot_object is not None:
        frame_count = min(frame_count, len(behind_robot_object))
    if object_source_cap is not None:
        frame_count = min(
            frame_count,
            int(object_source_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    z_split = _fill_and_smooth_depth(
        pose["joints_left"], pose["joints_right"], pose["valid"], frame_count,
        args.threshold_joint, args.depth_sigma,
    )

    output = processed / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    def make_writer(path: Path) -> CompatibleVideoWriter:
        return CompatibleVideoWriter(
            path, fps, (width, height), codec=args.video_codec
        )

    writer = make_writer(output)
    if not writer.isOpened():
        raise RuntimeError(f"cannot create output: {output}")

    layer_writers = None
    layer_grid_writer = None
    context_writers = None
    progressive_writers = None
    progressive_grid_writer = None
    transparency_plate = None
    if args.layer_output_dir is not None:
        layer_dir = processed / args.layer_output_dir
        layer_dir.mkdir(parents=True, exist_ok=True)
        layer_paths = {
            "background": layer_dir / "01_background_inpaint.mp4",
            "behind_robot": layer_dir / "02_robot_behind.mp4",
            "object": layer_dir / "03_object.mp4",
            "front_robot": layer_dir / "04_robot_front.mp4",
            "forced_object": layer_dir / "05_object_forced_front.mp4",
            "forced_thumb": layer_dir / "06_thumb_forced_front.mp4",
        }
        layer_writers = {
            name: make_writer(path)
            for name, path in layer_paths.items()
        }
        if not all(item.isOpened() for item in layer_writers.values()):
            raise RuntimeError(f"cannot create layer videos in {layer_dir}")
        layer_grid_writer = make_writer(layer_dir / "layers_6panel.mp4")
        if not layer_grid_writer.isOpened():
            raise RuntimeError(f"cannot create layer grid in {layer_dir}")
        if args.layer_context_videos:
            context_paths = {
                "behind_robot": layer_dir / "02_robot_behind_context.mp4",
                "object": layer_dir / "03_object_context.mp4",
                "front_robot": layer_dir / "04_robot_front_context.mp4",
                "forced_object": layer_dir / "05_object_forced_front_context.mp4",
                "forced_thumb": layer_dir / "06_thumb_forced_front_context.mp4",
            }
            context_writers = {
                name: make_writer(path)
                for name, path in context_paths.items()
            }
            if not all(item.isOpened() for item in context_writers.values()):
                raise RuntimeError(f"cannot create context videos in {layer_dir}")
        transparency_plate = checkerboard(height, width)

    if args.progressive_output_dir is not None:
        progressive_dir = processed / args.progressive_output_dir
        progressive_dir.mkdir(parents=True, exist_ok=True)
        progressive_paths = {
            "background": progressive_dir / "01_background.mp4",
            "behind_robot": progressive_dir / "02_add_robot_behind.mp4",
            "object": progressive_dir / "03_add_object.mp4",
            "front_robot": progressive_dir / "04_add_robot_front.mp4",
            "forced_object": progressive_dir / "05_add_object_forced_front.mp4",
            "forced_thumb": progressive_dir / "06_add_thumb_forced_front_final.mp4",
        }
        progressive_writers = {
            name: make_writer(path)
            for name, path in progressive_paths.items()
        }
        if not all(item.isOpened() for item in progressive_writers.values()):
            raise RuntimeError(
                f"cannot create progressive videos in {progressive_dir}"
            )
        progressive_grid_writer = make_writer(
            progressive_dir / "progressive_layers_6panel.mp4"
        )
        if not progressive_grid_writer.isOpened():
            raise RuntimeError(
                f"cannot create progressive grid in {progressive_dir}"
            )

    object_overlap = 0
    front_pixels = 0
    behind_pixels = 0
    forced_pixels = 0
    stage_config = StageConfig(
        robot_edge_sigma=args.robot_edge_sigma,
        robot_edge_mode=args.robot_edge_mode,
        object_edge_sigma=args.object_edge_sigma,
        forced_object_edge_sigma=args.force_front_edge_sigma,
        forced_robot_front_dilate=args.force_robot_front_dilate,
    )
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

        robot_frame = np.asarray(robot_rgb[idx])
        visible_robot = np.asarray(robot_mask[idx], dtype=bool)
        frame_inputs = FrameInputs(
            background=background,
            robot_rgb=robot_frame,
            object_rgb=object_source,
            robot_mask=visible_robot,
            robot_depth=(np.asarray(robot_depth[idx], dtype=np.float32)
                         + args.depth_bias),
            object_mask=np.asarray(object_mask[idx], dtype=bool),
            forced_object_mask=(
                np.asarray(force_front[idx], dtype=bool)
                if force_front is not None else np.zeros_like(visible_robot)
            ),
            forced_robot_front_mask=(
                np.asarray(force_robot_front[idx], dtype=bool)
                if force_robot_front is not None else np.zeros_like(visible_robot)
            ),
            split_depth=float(z_split[idx]),
            behind_robot_object_mask=(
                np.asarray(behind_robot_object[idx], dtype=bool)
                if behind_robot_object is not None else None
            ),
        )
        composite = compose_frame(frame_inputs, stage_config)
        progressive_stages = [
            np.clip(value, 0, 255).astype(np.uint8)
            for value in composite.stages
        ]
        writer.write(progressive_stages[-1])

        masks = composite.masks
        behind = masks.robot_behind
        obj = masks.object_visible
        ordinary_front = masks.robot_front
        forced = masks.object_forced_front
        forced_robot = masks.robot_forced_front
        front = ordinary_front | forced_robot

        if progressive_writers is not None:
            labelled_progressive = [
                label(stage, spec.label)
                for stage, spec in zip(progressive_stages, STAGE_SPECS)
            ]
            for progressive_writer, stage in zip(
                progressive_writers.values(), labelled_progressive
            ):
                progressive_writer.write(stage)
            progressive_grid_writer.write(
                grid_3x2(labelled_progressive, width, height)
            )

        if layer_writers is not None:
            background_layer = np.asarray(background, dtype=np.uint8)
            behind_layer = isolated_layer(
                robot_frame, behind, transparency_plate, (255, 160, 32)
            )
            object_layer = isolated_layer(
                object_source, obj, transparency_plate, (32, 220, 255)
            )
            front_layer = isolated_layer(
                robot_frame, ordinary_front, transparency_plate, (255, 64, 220)
            )
            forced_object_layer = isolated_layer(
                object_source, forced, transparency_plate, (40, 40, 255)
            )
            forced_thumb_layer = isolated_layer(
                robot_frame, forced_robot, transparency_plate, (40, 255, 40)
            )
            layers = [
                background_layer, behind_layer, object_layer, front_layer,
                forced_object_layer, forced_thumb_layer,
            ]
            for layer_writer, layer_frame in zip(layer_writers.values(), layers):
                layer_writer.write(layer_frame)
            if context_writers is not None:
                context_layers = {
                    "behind_robot": context_layer(
                        background, robot_frame, behind, (255, 160, 32)
                    ),
                    "object": context_layer(
                        background, object_source, obj, (32, 220, 255)
                    ),
                    "front_robot": context_layer(
                        background, robot_frame, ordinary_front, (255, 64, 220)
                    ),
                    "forced_object": context_layer(
                        background, object_source, forced, (40, 40, 255)
                    ),
                    "forced_thumb": context_layer(
                        background, robot_frame, forced_robot, (40, 255, 40)
                    ),
                }
                for name, context_writer in context_writers.items():
                    context_writer.write(context_layers[name])
            labelled = [
                label(layer, name) for layer, name in zip(
                    layers,
                    ("01 BACKGROUND", "02 ROBOT BEHIND", "03 OBJECT",
                     "04 ROBOT FRONT", "05 OBJECT FORCED FRONT",
                     "06 THUMB FORCED FRONT"),
                )
            ]
            layer_grid_writer.write(grid_3x2(labelled, width, height))

        object_overlap += int((obj & visible_robot).sum())
        front_pixels += int(front.sum())
        behind_pixels += int(behind.sum())
        forced_pixels += int((forced & visible_robot).sum())
        if (idx + 1) % 100 == 0:
            print(f"[frame] {idx + 1}/{frame_count}")

    source_cap.release()
    if object_source_cap is not None:
        object_source_cap.release()
    background_cap.release()
    writer.release()
    if layer_writers is not None:
        for layer_writer in layer_writers.values():
            layer_writer.release()
        layer_grid_writer.release()
        if context_writers is not None:
            for context_writer in context_writers.values():
                context_writer.release()
    if progressive_writers is not None:
        for progressive_writer in progressive_writers.values():
            progressive_writer.release()
        progressive_grid_writer.release()
    print(f"[ok] wrote {output}")
    print(f"[info] object/robot overlap={object_overlap} px, "
          f"front robot={front_pixels} px, behind robot={behind_pixels} px, "
          f"forced-object-over-robot={forced_pixels} px")


if __name__ == "__main__":
    main()
