"""Build a synchronized 2x2 RGB/depth preview for a stereo episode.

The expected converted episode layout is::

    camera_1/rgb/*.png        camera_1/depth_raw/*.png
    camera_2/rgb/*.png        camera_2/depth_raw/*.png

Rows correspond to cameras and columns correspond to RGB/depth.  Every tile
keeps a dedicated black header so labels never cover source pixels.  Raw
uint16 RealSense depth is interpreted using a configurable metres-per-unit
scale and rendered with a fixed range shared by both cameras.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def discover_frames(directory: Path, prefix: str) -> list[Path]:
    frames = sorted(directory.glob(f"{prefix}*.png"))
    if not frames:
        raise FileNotFoundError(f"no {prefix}*.png frames in {directory}")
    return frames


def colorize_depth(
    raw_depth: np.ndarray,
    *,
    depth_units_m: float,
    near_m: float,
    far_m: float,
) -> np.ndarray:
    if raw_depth.ndim != 2 or raw_depth.dtype != np.uint16:
        raise ValueError(
            f"depth must be uint16 HxW, got {raw_depth.dtype} {raw_depth.shape}"
        )
    depth_m = raw_depth.astype(np.float32) * float(depth_units_m)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    clipped = np.clip(depth_m, near_m, far_m)
    # OpenCV Turbo is blue at 0 and red at 255.  Invert metric depth so near
    # surfaces are warm and far surfaces are cool.
    normalized = (far_m - clipped) / (far_m - near_m)
    image_u8 = np.round(normalized * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(image_u8, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def make_tile(
    image: np.ndarray,
    *,
    label: str,
    frame_index: int,
    content_size: tuple[int, int],
    header_px: int,
) -> np.ndarray:
    width, height = content_size
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    tile = np.zeros((height + header_px, width, 3), dtype=np.uint8)
    tile[header_px:] = resized
    cv2.putText(
        tile,
        label,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    frame_text = f"F{frame_index:04d}"
    text_width = cv2.getTextSize(
        frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1
    )[0][0]
    cv2.putText(
        tile,
        frame_text,
        (width - text_width - 12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return tile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--depth_units_m", type=float, default=0.001)
    parser.add_argument("--near_m", type=float, default=0.35)
    parser.add_argument("--far_m", type=float, default=2.0)
    parser.add_argument("--tile_width", type=int, default=640)
    parser.add_argument("--tile_height", type=int, default=360)
    parser.add_argument("--header_px", type=int, default=40)
    parser.add_argument("--crf", type=int, default=18)
    args = parser.parse_args()

    if not (0.0 < args.near_m < args.far_m):
        raise ValueError("expected 0 < near_m < far_m")
    if args.fps <= 0.0 or args.depth_units_m <= 0.0:
        raise ValueError("fps and depth_units_m must be positive")
    if min(args.tile_width, args.tile_height, args.header_px) <= 0:
        raise ValueError("tile dimensions and header_px must be positive")

    episode = args.episode.resolve()
    streams: dict[tuple[int, str], list[Path]] = {}
    for camera in (1, 2):
        camera_dir = episode / f"camera_{camera}"
        streams[(camera, "rgb")] = discover_frames(camera_dir / "rgb", "rgb_frame")
        streams[(camera, "depth")] = discover_frames(
            camera_dir / "depth_raw", "depth_frame"
        )
    counts = {key: len(value) for key, value in streams.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"stream frame-count mismatch: {counts}")
    frame_count = next(iter(counts.values()))

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=output.suffix, dir=output.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink(missing_ok=True)

    tile_size = (args.tile_width, args.tile_height)
    output_width = args.tile_width * 2
    output_height = (args.tile_height + args.header_px) * 2
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{output_width}x{output_height}",
        "-r",
        str(args.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for frame_index in range(frame_count):
            rows = []
            for camera, model, view in (
                (1, "D455", "SIDE"),
                (2, "D435I", "EGO"),
            ):
                rgb = cv2.imread(
                    str(streams[(camera, "rgb")][frame_index]),
                    cv2.IMREAD_COLOR,
                )
                raw_depth = cv2.imread(
                    str(streams[(camera, "depth")][frame_index]),
                    cv2.IMREAD_UNCHANGED,
                )
                if rgb is None or raw_depth is None:
                    raise RuntimeError(
                        f"failed to read camera {camera} frame {frame_index}"
                    )
                depth = colorize_depth(
                    raw_depth,
                    depth_units_m=args.depth_units_m,
                    near_m=args.near_m,
                    far_m=args.far_m,
                )
                rgb_tile = make_tile(
                    rgb,
                    label=f"CAMERA {camera} | RGB | {model} {view}",
                    frame_index=frame_index,
                    content_size=tile_size,
                    header_px=args.header_px,
                )
                depth_tile = make_tile(
                    depth,
                    label=(
                        f"CAMERA {camera} | RAW DEPTH {args.near_m:.2f}-"
                        f"{args.far_m:.2f}m | NEAR=RED"
                    ),
                    frame_index=frame_index,
                    content_size=tile_size,
                    header_px=args.header_px,
                )
                rows.append(np.hstack([rgb_tile, depth_tile]))
            frame = np.vstack(rows)
            process.stdin.write(frame.tobytes())
            if (frame_index + 1) % 100 == 0:
                print(f"[preview] {frame_index + 1}/{frame_count}", flush=True)
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")
        os.replace(temporary, output)
    except BaseException:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise

    print(
        f"[ok] {output} ({output_width}x{output_height}, "
        f"{frame_count} frames, {args.fps:g} fps)",
        flush=True,
    )


if __name__ == "__main__":
    main()
