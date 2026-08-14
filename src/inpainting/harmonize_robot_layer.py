"""Give the rendered robot the two things the camera would have given it.

A pyrender frame is a perfect sample of a static pose: infinitely sharp and
noise-free. The footage it is dropped into is neither, and the mismatch is what
reads as "composited" even when the geometry is right.

*Motion blur.*  At 24 fps the wrist crosses up to 49 px between frames, so a
real 180-degree shutter would smear it over ~25 px. The render does not, and a
razor-sharp hand against a motion-blurred scene is the strongest tell of all.
Per-pixel velocity comes from optical flow on the rendered robot itself, and the
frame is resampled along it over the open-shutter interval.

*Grain.*  Static background pixels in the source carry 1.3 levels of temporal
noise; the render carries none, so the robot sits dead still inside a shot that
is quietly fizzing everywhere else. Matching noise is added back.

Both are applied through the robot's own coverage, feathered, so the effect
fades out into the plate instead of stopping at the silhouette.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from layered_compositor.video import CompatibleVideoWriter


def _flow(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 21, 3,
                                        5, 1.2, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composite", type=Path, required=True)
    parser.add_argument("--robot_dir", type=Path, required=True,
                        help="Overlay directory: robot_rgb.npy drives the flow, "
                             "robot_mask.npy the coverage.")
    parser.add_argument("--shutter", type=float, default=0.5,
                        help="Open fraction of the frame interval; 0.5 is a "
                             "180-degree shutter.")
    parser.add_argument("--samples", type=int, default=7,
                        help="Resamples across the open shutter.")
    parser.add_argument("--deadzone", type=float, default=1.5,
                        help="Flow below this many px is treated as noise, so a "
                             "stationary arm stays sharp.")
    parser.add_argument("--max_shift", type=float, default=20.0,
                        help="Clamp on per-pixel travel (px), so a flow "
                             "blow-up cannot smear the frame.")
    parser.add_argument("--grain", type=float, default=1.3,
                        help="Sigma of the noise added over the robot, in "
                             "levels; measure it off the source footage.")
    parser.add_argument("--gain", type=float, nargs=3, default=None,
                        metavar=("B", "G", "R"),
                        help="Per-channel gain on the robot, mapping its white "
                             "onto the scene's. The renderer's own exposure is "
                             "arbitrary, so its shell comes out brighter than "
                             "anything the camera actually saw and reads as a "
                             "sticker. Measure the ratio between the robot's "
                             "bright shell and a white surface in the plate.")
    parser.add_argument("--gain_feather", type=float, default=1.0,
                        help="A colour change that bleeds past the silhouette "
                             "shows as a halo, so this is much tighter than "
                             "--feather.")
    parser.add_argument("--feather", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.composite))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.composite}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    robot_rgb = np.load(args.robot_dir / "robot_rgb.npy", mmap_mode="r")
    robot_mask = np.load(args.robot_dir / "robot_mask.npy", mmap_mode="r")
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                 np.arange(height, dtype=np.float32))
    rng = np.random.default_rng(args.seed)
    writer = CompatibleVideoWriter(args.output, fps, (width, height), codec="h264")

    previous_gray = previous_mask = None
    blurred_px = 0
    travel = []
    for idx in range(len(robot_mask)):
        ok, frame = capture.read()
        if not ok:
            break
        mask = np.asarray(robot_mask[idx], dtype=bool)
        gray = cv2.cvtColor(np.asarray(robot_rgb[idx]), cv2.COLOR_BGR2GRAY)
        out = frame.astype(np.float32)

        # Grade before blurring, so the smear carries the graded colour out
        # past the silhouette the way a real one would.
        if args.gain is not None and mask.any():
            tight = cv2.GaussianBlur(mask.astype(np.float32), (0, 0),
                                     args.gain_feather)[..., None]
            out *= 1.0 - tight + tight * np.asarray(args.gain, np.float32)

        if previous_mask is not None and mask.sum() > 500:
            flow = _flow(previous_gray, gray)
            # Flow is a correspondence, so it only means anything where the
            # robot is in both frames. At a silhouette that just appeared there
            # is nothing to match and Farneback invents a large vector, which
            # would smear the plate; hold the field to the overlap and let it
            # decay outward so the blur still has somewhere to go.
            overlap = (mask & previous_mask).astype(np.float32)
            weight = np.clip(cv2.GaussianBlur(overlap, (0, 0), 9.0), 0, 1)
            flow *= weight[..., None]
            speed = np.linalg.norm(flow, axis=2)
            flow[speed < args.deadzone] = 0.0        # flow noise on a still arm
            flow = np.clip(flow * args.shutter, -args.max_shift, args.max_shift)
            travel.append(float(np.abs(flow[mask]).max()) if mask.any() else 0.0)

            accumulator = np.zeros_like(out)
            offsets = np.linspace(-0.5, 0.5, args.samples)
            for step in offsets:
                map_x = grid_x + step * flow[..., 0]
                map_y = grid_y + step * flow[..., 1]
                accumulator += cv2.remap(out, map_x, map_y, cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
            accumulator /= len(offsets)
            alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0),
                                     args.feather)[..., None]
            out = alpha * accumulator + (1 - alpha) * out
            blurred_px += int(mask.sum())

        if args.grain > 0 and mask.any():
            noise = rng.normal(0.0, args.grain, size=(height, width, 1))
            alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0),
                                     args.feather)[..., None]
            out += alpha * noise

        writer.write(np.clip(out, 0, 255).astype(np.uint8))
        previous_gray, previous_mask = gray, mask
        if idx % 100 == 0:
            print(f"[frame] {idx}/{len(robot_mask)}", flush=True)
    capture.release()
    writer.release()
    if travel:
        print(f"[info] shutter travel: median {np.median(travel):.1f} px, "
              f"p90 {np.percentile(travel, 90):.1f} px, max {max(travel):.1f} px")
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
