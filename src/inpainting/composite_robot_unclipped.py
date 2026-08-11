"""Composite the depth-rendered robot over the hand/arm-inpainted background.

This is the simple, object-agnostic composite used when an amodal object mask
is not available yet.  The robot mask is deliberately independent of the
human hand/arm segmentation mask, so no robot links or fingers are clipped by
the removed human silhouette.

Inputs:
    <processed_demo>/inpaint_processor/video_human_inpaint.mkv
    <processed_demo>/overlay_processor/robot_rgb.npy
    <processed_demo>/overlay_processor/robot_mask.npy

Outputs:
    --out
    --robot_only_out
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def _writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--robot_only_out", type=Path, required=True)
    ap.add_argument("--background", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--edge_sigma", type=float, default=0.6,
                    help="Gaussian feathering after resizing the robot mask.")
    args = ap.parse_args()

    pd = args.processed_demo
    bg_path = args.background or (
        pd / "inpaint_processor" / "video_human_inpaint.mkv"
    )
    robot_rgb = np.load(
        pd / "overlay_processor" / "robot_rgb.npy", mmap_mode="r"
    )
    robot_mask = np.load(
        pd / "overlay_processor" / "robot_mask.npy", mmap_mode="r"
    )

    cap = cv2.VideoCapture(str(bg_path))
    if not cap.isOpened():
        raise FileNotFoundError(bg_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0

    final_writer = _writer(args.out, fps, (width, height))
    robot_writer = _writer(args.robot_only_out, fps, (width, height))

    n = min(len(robot_rgb), len(robot_mask))
    written = 0
    for t in range(n):
        ok, bg = cap.read()
        if not ok:
            break

        # Renderer arrays are RGB at raw-video resolution; OpenCV expects BGR.
        robot = np.asarray(robot_rgb[t])[..., ::-1]
        mask = np.asarray(robot_mask[t]).astype(np.float32)
        if robot.shape[:2] != (height, width):
            robot = cv2.resize(
                robot, (width, height), interpolation=cv2.INTER_AREA
            )
            mask = cv2.resize(
                mask, (width, height), interpolation=cv2.INTER_AREA
            )
        if args.edge_sigma > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), args.edge_sigma)
        alpha = np.clip(mask, 0.0, 1.0)[..., None]

        robot_only = np.clip(
            robot.astype(np.float32) * alpha, 0, 255
        ).astype(np.uint8)
        final = np.clip(
            bg.astype(np.float32) * (1.0 - alpha)
            + robot.astype(np.float32) * alpha,
            0, 255,
        ).astype(np.uint8)

        robot_writer.write(robot_only)
        final_writer.write(final)
        written += 1
        if written % 100 == 0:
            print(f"{written}/{n}")

    cap.release()
    final_writer.release()
    robot_writer.release()
    print(f"[ok] final: {args.out} ({written} frames)")
    print(f"[ok] robot only: {args.robot_only_out} ({written} frames)")


if __name__ == "__main__":
    main()
