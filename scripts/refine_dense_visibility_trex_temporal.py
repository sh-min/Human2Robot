#!/usr/bin/env python3
"""T-Rex-inspired fast temporal residual for dense robot visibility masks.

This is not the released T-Rex VLA.  It borrows the two-clock design: a slow
dense MANO/object visibility estimate and a fast per-finger residual updated
from a 16-frame temporal window plus HaCo contact probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
PART_GROUPS = ((13, 14, 15), (1, 2, 3), (4, 5, 6), (10, 11, 12), (7, 8, 9))


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


def compute_contact_features(
    contact_paths: list[Path], parts: np.ndarray, side: str
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros((len(contact_paths), 5), np.float32)
    maximum = np.zeros_like(mean)
    for frame, path in enumerate(contact_paths):
        with np.load(path) as data:
            probability = np.asarray(
                data[f"{side}_contact_probability"], dtype=np.float32
            )
        for finger, group in enumerate(PART_GROUPS):
            selected = probability[np.isin(parts, group)]
            if len(selected):
                mean[frame, finger] = float(selected.mean())
                maximum[frame, finger] = float(selected.max())
    return mean, maximum


def centered_temporal_prior(signal: np.ndarray, window: int) -> np.ndarray:
    radius_before = window // 2
    radius_after = window - radius_before
    output = np.zeros_like(signal, dtype=np.float32)
    for frame in range(len(signal)):
        start = max(0, frame - radius_before)
        stop = min(len(signal), frame + radius_after)
        values = signal[start:stop]
        median = np.median(values, axis=0)
        upper = np.quantile(values, 0.70, axis=0)
        output[frame] = 0.7 * median + 0.3 * upper
    return output


def expand_from_seed(
    seed: np.ndarray,
    part: np.ndarray,
    allowed: np.ndarray,
    desired_count: int,
) -> np.ndarray:
    output = seed & part
    current = int(output.sum())
    if current >= desired_count:
        return output
    candidates = part & allowed & ~output
    need = min(desired_count - current, int(candidates.sum()))
    if need <= 0:
        return output
    if output.any():
        distance_input = (~output).astype(np.uint8)
        distance = cv2.distanceTransform(distance_input, cv2.DIST_L2, 3)
        positions = np.argwhere(candidates)
        scores = distance[positions[:, 0], positions[:, 1]]
        selected = positions[np.argpartition(scores, need - 1)[:need]]
    else:
        positions = np.argwhere(candidates)
        # With no spatial seed, start from the object-overlap pixels nearest
        # the candidate centroid. This is a conservative fallback.
        center = positions.mean(axis=0)
        scores = ((positions - center) ** 2).sum(axis=1)
        selected = positions[np.argpartition(scores, need - 1)[:need]]
    output[selected[:, 0], selected[:, 1]] = True
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
    parser.add_argument("--contact_dir", type=Path, required=True)
    parser.add_argument("--finger_parts", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--temporal_window", type=int, default=16)
    parser.add_argument("--slow_stride", type=int, default=4)
    parser.add_argument("--contact_weight", type=float, default=0.25)
    parser.add_argument("--residual_strength", type=float, default=0.85)
    parser.add_argument("--object_dilate_px", type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = np.load(args.baseline_hidden_mask, mmap_mode="r")
    anchor_counts = np.asarray(np.load(args.dense_anchor_counts), dtype=np.float32)
    robot_rgb = np.load(args.robot_rgb, mmap_mode="r")
    robot_mask = np.load(args.robot_mask, mmap_mode="r")
    robot_labels = np.load(args.robot_finger_labels, mmap_mode="r")
    object_mask = np.load(args.object_mask, mmap_mode="r")
    parts = np.asarray(np.load(args.finger_parts), dtype=np.int32)
    contacts = sorted(args.contact_dir.glob("*.npz"))
    cap = cv2.VideoCapture(str(args.clean_plate))
    if not cap.isOpened():
        raise FileNotFoundError(args.clean_plate)
    frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if not (
        len(baseline) == len(anchor_counts) == len(robot_rgb) == len(robot_mask)
        == len(robot_labels) == len(object_mask) == len(contacts) == frames
    ):
        raise ValueError("frame count mismatch")
    low_h, low_w = robot_mask.shape[1:]

    anchor_ratio = np.divide(
        anchor_counts[:, :, 1], anchor_counts[:, :, 0],
        out=np.zeros((frames, 5), np.float32),
        where=anchor_counts[:, :, 0] > 0,
    )
    baseline_ratio = np.zeros((frames, 5), np.float32)
    total_pixels = np.zeros((frames, 5), np.int32)
    for frame in range(frames):
        labels = np.asarray(robot_labels[frame], dtype=np.uint8)
        hidden = np.asarray(baseline[frame], dtype=bool)
        for finger in range(5):
            part = labels == finger + 1
            total_pixels[frame, finger] = int(part.sum())
            if part.any():
                baseline_ratio[frame, finger] = float((hidden & part).sum() / part.sum())

    contact_mean, contact_max = compute_contact_features(contacts, parts, args.side)
    temporal_prior = centered_temporal_prior(anchor_ratio, args.temporal_window)
    slow_signal = np.zeros_like(anchor_ratio)
    for frame in range(frames):
        slow_signal[frame] = anchor_ratio[(frame // args.slow_stride) * args.slow_stride]
    contact_gate = np.clip(
        1.0 - args.contact_weight + args.contact_weight * contact_max,
        0.0, 1.0,
    )
    desired_ratio = np.maximum(
        baseline_ratio,
        baseline_ratio + args.residual_strength
        * np.maximum(temporal_prior - slow_signal, 0.0) * contact_gate,
    )
    desired_ratio = np.clip(desired_ratio, 0.0, 1.0)

    refined_all = np.zeros((frames, low_h, low_w), dtype=bool)
    added_by_finger = np.zeros(5, np.int64)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (args.object_dilate_px * 2 + 1,) * 2
    )
    final_writer = open_writer(
        args.output_dir / "video_trex_temporal_refined.mp4v.mp4",
        fps, (width, height),
    )
    panel_w, panel_h, header = width // 2, height // 2, 70
    compare_writer = open_writer(
        args.output_dir / "video_compare_baseline_vs_trex_temporal.mp4v.mp4",
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
                before = refined & part
                desired = int(round(desired_ratio[frame, finger] * part.sum()))
                after = expand_from_seed(before, part, obj, desired)
                refined[part] = after[part]
                added_by_finger[finger] += int(after.sum() - before.sum())
            refined_all[frame] = refined

            rmask_low = np.asarray(robot_mask[frame], dtype=bool)
            rrgb = cv2.resize(
                np.asarray(robot_rgb[frame]), (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
            base_alpha = cv2.resize(
                (rmask_low & ~base_hidden).astype(np.float32),
                (width, height), interpolation=cv2.INTER_NEAREST,
            )
            refined_alpha = cv2.resize(
                (rmask_low & ~refined).astype(np.float32),
                (width, height), interpolation=cv2.INTER_NEAREST,
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
                canvas, "Dense anchors - per frame", (18, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.67, (245, 245, 245), 2, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, "T-Rex-inspired 16-frame fast residual", (panel_w + 18, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (120, 230, 255), 2, cv2.LINE_AA,
            )
            compare_writer.write(canvas)
            if (frame + 1) % 100 == 0:
                print(f"[trex-temporal] {frame + 1}/{frames}", flush=True)
    finally:
        cap.release()
        final_writer.release()
        compare_writer.release()

    np.save(args.output_dir / "trex_temporal_refined_hidden_mask.npy", refined_all)
    np.savez_compressed(
        args.output_dir / "temporal_features.npz",
        finger_names=np.asarray(FINGER_NAMES),
        anchor_ratio=anchor_ratio,
        baseline_ratio=baseline_ratio,
        slow_signal=slow_signal,
        temporal_prior=temporal_prior,
        contact_mean=contact_mean,
        contact_max=contact_max,
        desired_ratio=desired_ratio,
    )
    report = {
        "schema_version": 1,
        "method": "T-Rex-inspired asynchronous per-finger temporal visibility residual",
        "not_official_trex_model": True,
        "frames": frames,
        "slow_stride_frames": args.slow_stride,
        "fast_stride_frames": 1,
        "temporal_window_frames": args.temporal_window,
        "added_hidden_pixels_lowres": int(added_by_finger.sum()),
        "added_hidden_pixels_by_finger": {
            name: int(value) for name, value in zip(FINGER_NAMES, added_by_finger)
        },
        "features": [
            "dense_MANO_anchor_occlusion_ratio",
            "baseline_robot_removed_ratio",
            "HaCo_mean_contact_probability",
            "HaCo_max_contact_probability",
        ],
        "invariant": "fast residual only adds object-supported hidden pixels; it never removes baseline occlusion",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
