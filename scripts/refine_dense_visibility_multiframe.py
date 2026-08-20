#!/usr/bin/env python3
"""Refine per-frame robot visibility using only neighboring video frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return np.asarray(mask, dtype=bool)
    return cv2.resize(
        np.asarray(mask, dtype=np.uint8), (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def open_writer(path: Path, fps: float, size: tuple[int, int]):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer: {path}")
    return writer


def temporal_consensus(signal: np.ndarray, window: int) -> np.ndarray:
    """Centered robust consensus; uses no contact/model signal."""
    before = window // 2
    after = window - before
    output = np.zeros_like(signal, dtype=np.float32)
    for frame in range(len(signal)):
        values = signal[max(0, frame - before):min(len(signal), frame + after)]
        output[frame] = np.median(values, axis=0)
    return output


def expand_near_current_mask(
    current: np.ndarray,
    finger: np.ndarray,
    object_support: np.ndarray,
    desired_count: int,
) -> np.ndarray:
    output = current & finger
    need = min(
        max(desired_count - int(output.sum()), 0),
        int((finger & object_support & ~output).sum()),
    )
    if need <= 0:
        return output
    candidates = np.argwhere(finger & object_support & ~output)
    if output.any():
        distance = cv2.distanceTransform((~output).astype(np.uint8), cv2.DIST_L2, 3)
        score = distance[candidates[:, 0], candidates[:, 1]]
    else:
        center = candidates.mean(axis=0)
        score = ((candidates - center) ** 2).sum(axis=1)
    chosen = candidates[np.argpartition(score, need - 1)[:need]]
    output[chosen[:, 0], chosen[:, 1]] = True
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean_plate", type=Path, required=True)
    parser.add_argument("--baseline_hidden_mask", type=Path, required=True)
    parser.add_argument("--dense_anchor_counts", type=Path, required=True)
    parser.add_argument("--robot_rgb", type=Path, required=True)
    parser.add_argument("--robot_mask", type=Path, required=True)
    parser.add_argument("--robot_finger_labels", type=Path, required=True)
    parser.add_argument("--object_mask", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--object_dilate_px", type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = np.load(args.baseline_hidden_mask, mmap_mode="r")
    anchor_counts = np.asarray(np.load(args.dense_anchor_counts), dtype=np.float32)
    robot_rgb = np.load(args.robot_rgb, mmap_mode="r")
    robot_mask = np.load(args.robot_mask, mmap_mode="r")
    robot_labels = np.load(args.robot_finger_labels, mmap_mode="r")
    object_mask = np.load(args.object_mask, mmap_mode="r")
    cap = cv2.VideoCapture(str(args.clean_plate))
    if not cap.isOpened():
        raise FileNotFoundError(args.clean_plate)
    frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if not (
        len(baseline) == len(anchor_counts) == len(robot_rgb) == len(robot_mask)
        == len(robot_labels) == len(object_mask) == frames
    ):
        raise ValueError("frame count mismatch")
    low_h, low_w = robot_mask.shape[1:]

    anchor_ratio = np.divide(
        anchor_counts[:, :, 1], anchor_counts[:, :, 0],
        out=np.zeros((frames, 5), np.float32),
        where=anchor_counts[:, :, 0] > 0,
    )
    baseline_ratio = np.zeros((frames, 5), np.float32)
    for frame in range(frames):
        labels = np.asarray(robot_labels[frame], dtype=np.uint8)
        hidden = np.asarray(baseline[frame], dtype=bool)
        for finger in range(5):
            part = labels == finger + 1
            if part.any():
                baseline_ratio[frame, finger] = float((hidden & part).sum() / part.sum())

    consensus = temporal_consensus(anchor_ratio, args.window)
    missing = np.maximum(consensus - anchor_ratio, 0.0)
    desired_ratio = np.clip(
        baseline_ratio + args.strength * missing, 0.0, 1.0
    )

    refined_all = np.zeros((frames, low_h, low_w), dtype=bool)
    added_by_finger = np.zeros(5, np.int64)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (args.object_dilate_px * 2 + 1,) * 2
    )
    final_writer = open_writer(
        args.output_dir / "video_multiframe_refined.mp4v.mp4",
        fps, (width, height),
    )
    panel_w, panel_h, header = width // 2, height // 2, 70
    compare_writer = open_writer(
        args.output_dir / "video_compare_perframe_vs_multiframe.mp4v.mp4",
        fps, (panel_w * 2, panel_h + header),
    )

    try:
        for frame in range(frames):
            ok, clean = cap.read()
            if not ok:
                raise RuntimeError(f"clean plate ended at frame {frame}")
            labels = np.asarray(robot_labels[frame], dtype=np.uint8)
            base_hidden = np.asarray(baseline[frame], dtype=bool)
            refined = base_hidden.copy()
            obj = resize_mask(object_mask[frame], low_w, low_h)
            obj = cv2.dilate(obj.astype(np.uint8), kernel).astype(bool)
            for finger in range(5):
                part = labels == finger + 1
                before_mask = refined & part
                desired = int(round(desired_ratio[frame, finger] * part.sum()))
                after_mask = expand_near_current_mask(
                    before_mask, part, obj, desired
                )
                refined[part] = after_mask[part]
                added_by_finger[finger] += int(
                    after_mask.sum() - before_mask.sum()
                )
            refined_all[frame] = refined

            rmask = np.asarray(robot_mask[frame], dtype=bool)
            rrgb = cv2.resize(
                np.asarray(robot_rgb[frame]), (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
            base_alpha = cv2.resize(
                (rmask & ~base_hidden).astype(np.float32), (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            refined_alpha = cv2.resize(
                (rmask & ~refined).astype(np.float32), (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            base_final = np.clip(
                clean.astype(np.float32) * (1 - base_alpha[..., None])
                + rrgb.astype(np.float32) * base_alpha[..., None], 0, 255,
            ).astype(np.uint8)
            refined_final = np.clip(
                clean.astype(np.float32) * (1 - refined_alpha[..., None])
                + rrgb.astype(np.float32) * refined_alpha[..., None], 0, 255,
            ).astype(np.uint8)
            final_writer.write(refined_final)
            canvas = np.full((panel_h + header, panel_w * 2, 3), 22, np.uint8)
            canvas[header:, :panel_w] = cv2.resize(base_final, (panel_w, panel_h))
            canvas[header:, panel_w:] = cv2.resize(refined_final, (panel_w, panel_h))
            cv2.putText(
                canvas, "Single-frame dense visibility", (18, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, f"{args.window}-frame temporal consensus", (panel_w + 18, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 230, 255), 2, cv2.LINE_AA,
            )
            compare_writer.write(canvas)
            if (frame + 1) % 100 == 0:
                print(f"[multiframe] {frame + 1}/{frames}", flush=True)
    finally:
        cap.release()
        final_writer.release()
        compare_writer.release()

    np.save(args.output_dir / "multiframe_refined_hidden_mask.npy", refined_all)
    np.savez_compressed(
        args.output_dir / "multiframe_features.npz",
        finger_names=np.asarray(FINGER_NAMES),
        perframe_anchor_ratio=anchor_ratio,
        temporal_consensus=consensus,
        baseline_ratio=baseline_ratio,
        desired_ratio=desired_ratio,
    )
    report = {
        "schema_version": 1,
        "method": "multi-frame per-finger visibility consensus",
        "frames": frames,
        "window_frames": args.window,
        "uses_haco_contact": False,
        "uses_trex_model_or_architecture": False,
        "uses_learned_temporal_model": False,
        "added_hidden_pixels_lowres": int(added_by_finger.sum()),
        "added_hidden_pixels_by_finger": {
            name: int(value) for name, value in zip(FINGER_NAMES, added_by_finger)
        },
        "rule": "when current occlusion is lower than the neighboring-frame median, add only object-supported hidden pixels",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
