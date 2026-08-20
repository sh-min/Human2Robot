#!/usr/bin/env python3
"""Prepare a short user-video clip for local masked VACE inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def fit_480p(frame: np.ndarray, width: int = 832, height: int = 480) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max(width / w, height / h)
    resized = cv2.resize(
        frame,
        (round(w * scale), round(h * scale)),
        interpolation=cv2.INTER_NEAREST if frame.ndim == 2 else cv2.INTER_AREA,
    )
    y = (resized.shape[0] - height) // 2
    x = (resized.shape[1] - width) // 2
    return resized[y : y + height, x : x + width]


def arm_mask(frame: np.ndarray, use_hull: bool = True) -> np.ndarray:
    """Extract the skin-colored arm entering from the left edge."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(ycrcb, (35, 128, 72), (255, 184, 137))
    skin &= cv2.inRange(hsv, (0, 18, 35), (28, 190, 255))

    # The actor enters exclusively from the left in this fixed-camera episode.
    roi = np.zeros_like(skin)
    roi[:, : int(frame.shape[1] * 0.72)] = 255
    skin &= roi
    skin = cv2.morphologyEx(
        skin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    )
    skin = cv2.morphologyEx(
        skin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(skin)
    chosen = np.zeros_like(skin)
    candidates = []
    for idx in range(1, count):
        x, y, w, h, area = stats[idx]
        # Reject the beige wall band at the top; the arm is connected to the
        # left image edge and extends well into the tabletop region.
        if area >= 1500 and x < 90 and y + h > frame.shape[0] * 0.30:
            candidates.append((area, idx))
    if candidates:
        chosen[labels == max(candidates)[1]] = 255
    # The tabletop begins below the fixed beige wall band. On a few frames the
    # wall and arm meet at the left border after morphology, so cut the known
    # non-editable band explicitly.
    chosen[: int(frame.shape[0] * 0.22)] = 0
    points = cv2.findNonZero(chosen)
    if use_hull and points is not None and len(points) >= 3:
        # Skin thresholding can break along the comparatively desaturated
        # forearm. Its silhouette is the convex connection between the hand
        # and the left-edge arm entry in this fixed-camera sequence.
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(chosen, hull, 255)
    chosen = cv2.dilate(
        chosen, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    )
    chosen[: int(frame.shape[0] * 0.22)] = 0
    # Keep the dark mug intact even where it lies inside the hand's hull.
    dark_object = cv2.inRange(hsv, (0, 0, 0), (179, 255, 72))
    dark_object = cv2.dilate(
        dark_object, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    chosen[dark_object > 0] = 0
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-frame", type=int, default=1)
    parser.add_argument("--frame-num", type=int, default=81)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    if args.frame_num % 4 != 1:
        raise ValueError("VACE frame count must be 4n+1")

    args.output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise FileNotFoundError(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.first_frame)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    paths = {
        "raw": args.output / "raw_clip_480p.mp4",
        "src": args.output / "vace_src_video.mp4",
        "mask": args.output / "vace_src_mask.mp4",
        "overlay": args.output / "mask_overlay.mp4",
    }
    writers = {key: cv2.VideoWriter(str(path), fourcc, fps, (832, 480)) for key, path in paths.items()}
    coverages = []
    for frame_idx in range(args.frame_num):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to decode relative frame {frame_idx}")
        fitted = fit_480p(frame)
        if frame_idx == 0:
            seed_mask = fit_480p(arm_mask(frame, use_hull=False))
            cv2.imwrite(str(args.output / "swap_seed_mask_frame0.png"), seed_mask)
        mask = fit_480p(arm_mask(frame))
        masked = fitted.copy()
        masked[mask > 0] = 127
        overlay = fitted.copy()
        red = np.zeros_like(overlay)
        red[:, :, 2] = 255
        selected = mask > 0
        overlay[selected] = cv2.addWeighted(overlay, 0.35, red, 0.65, 0)[selected]
        writers["raw"].write(fitted)
        writers["src"].write(masked)
        writers["mask"].write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        writers["overlay"].write(overlay)
        coverages.append(float(np.mean(selected)))
    cap.release()
    for writer in writers.values():
        writer.release()

    reference = cv2.imread(str(args.reference))
    if reference is None:
        raise FileNotFoundError(args.reference)
    cv2.imwrite(str(args.output / "robot_reference_white.png"), fit_480p(reference))
    manifest = {
        "source": str(args.input.resolve()),
        "first_frame": args.first_frame,
        "last_frame": args.first_frame + args.frame_num - 1,
        "frame_num": args.frame_num,
        "fps": fps,
        "label": "HangCup",
        "mask_method": "skin-color arm component entering from left; 25px dilation before 480p resize",
        "mean_editable_fraction": float(np.mean(coverages)),
        "reference": str(args.reference.resolve()),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
