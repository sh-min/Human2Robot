#!/usr/bin/env python3
"""Causal T-Rex-style slow/fast refinement for robot-hand visibility.

This adapter does not run the official T-Rex checkpoint. Official T-Rex needs
ten fingertip tactile streams and 6-axis wrench histories from its Vega-1 /
Sharpa hardware. Human2Robot instead maps the reusable system idea as follows:

* slow expert: dense MANO/object visibility, sampled and held at a lower rate;
* fast expert: fresh per-finger HaCo contact plus a causal 16-frame history;
* safety rule: the fast expert may hide robot pixels only where the current
  segmented object supports them, and may never reveal a baseline-hidden pixel.

The output report records this distinction so results cannot be mistaken for
inference from the official T-Rex model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
PART_GROUPS = ((13, 14, 15), (1, 2, 3), (4, 5, 6), (10, 11, 12), (7, 8, 9))
OFFICIAL_REPOSITORY = "https://github.com/ZhuoyangLiu2005/T-Rex"
OFFICIAL_PAPER = "https://arxiv.org/abs/2606.17055"


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return np.asarray(mask, dtype=bool)
    return cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def open_writer(path: Path, fps: float, size: tuple[int, int]):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer: {path}")
    return writer


def compute_contact_features(
    contact_paths: list[Path], parts: np.ndarray, side: str
) -> tuple[np.ndarray, np.ndarray]:
    """Convert HaCo vertex probabilities to five per-finger signals."""
    parts = np.asarray(parts)
    if parts.ndim != 1:
        raise ValueError("finger_parts must be a one-dimensional array")
    mean = np.zeros((len(contact_paths), len(FINGER_NAMES)), np.float32)
    maximum = np.zeros_like(mean)
    for frame, path in enumerate(contact_paths):
        with np.load(path) as data:
            valid_key = f"{side}_valid"
            if valid_key in data and not bool(data[valid_key]):
                continue
            probability = np.asarray(
                data[f"{side}_contact_probability"], dtype=np.float32
            )
        if probability.shape != parts.shape:
            raise ValueError(
                f"{path}: contact probability shape {probability.shape} "
                f"does not match finger_parts {parts.shape}"
            )
        probability = np.clip(probability, 0.0, 1.0)
        for finger, group in enumerate(PART_GROUPS):
            selected = probability[np.isin(parts, group)]
            if selected.size:
                mean[frame, finger] = float(selected.mean())
                maximum[frame, finger] = float(selected.max())
    return mean, maximum


def causal_window_statistics(
    signal: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return causal median, 70th percentile, and maximum histories."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2 or signal.shape[1] != len(FINGER_NAMES):
        raise ValueError("signal must have shape (frames, 5)")
    if window <= 0:
        raise ValueError("window must be positive")
    median = np.zeros_like(signal)
    upper = np.zeros_like(signal)
    maximum = np.zeros_like(signal)
    for frame in range(len(signal)):
        values = signal[max(0, frame - window + 1):frame + 1]
        median[frame] = np.median(values, axis=0)
        upper[frame] = np.quantile(values, 0.70, axis=0)
        maximum[frame] = values.max(axis=0)
    return median, upper, maximum


def sample_and_hold(signal: np.ndarray, stride: int) -> np.ndarray:
    """Cache a slow-expert output and reuse it between update ticks."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    signal = np.asarray(signal, dtype=np.float32)
    output = np.empty_like(signal)
    for frame in range(len(signal)):
        output[frame] = signal[(frame // stride) * stride]
    return output


def discrete_contact_state(contact_gate: np.ndarray) -> np.ndarray:
    """Diagnostic four-state code: none, onset, sustained, release."""
    active = np.asarray(contact_gate) >= 0.5
    previous = np.zeros_like(active)
    previous[1:] = active[:-1]
    state = np.zeros(active.shape, dtype=np.uint8)
    state[active & ~previous] = 1
    state[active & previous] = 2
    state[~active & previous] = 3
    return state


def build_slow_fast_signals(
    anchor_ratio: np.ndarray,
    baseline_ratio: np.ndarray,
    contact_max: np.ndarray,
    *,
    temporal_window: int,
    slow_stride: int,
    contact_threshold: float,
    contact_weight: float,
    residual_strength: float,
) -> dict[str, np.ndarray]:
    """Build causal slow/fast signals without accessing a future frame."""
    anchor_ratio = np.asarray(anchor_ratio, dtype=np.float32)
    baseline_ratio = np.asarray(baseline_ratio, dtype=np.float32)
    contact_max = np.asarray(contact_max, dtype=np.float32)
    if not (
        anchor_ratio.shape == baseline_ratio.shape == contact_max.shape
        and anchor_ratio.ndim == 2
        and anchor_ratio.shape[1] == len(FINGER_NAMES)
    ):
        raise ValueError("anchor, baseline, and contact arrays must share (T, 5)")
    if not 0.0 <= contact_threshold < 1.0:
        raise ValueError("contact_threshold must be in [0, 1)")
    if not 0.0 <= contact_weight <= 1.0:
        raise ValueError("contact_weight must be in [0, 1]")
    if not 0.0 <= residual_strength <= 1.0:
        raise ValueError("residual_strength must be in [0, 1]")

    anchor_median, anchor_upper, _ = causal_window_statistics(
        anchor_ratio, temporal_window
    )
    contact_median, contact_upper, contact_peak = causal_window_statistics(
        contact_max, temporal_window
    )
    visual_prior = 0.7 * anchor_median + 0.3 * anchor_upper
    contact_history = (
        0.50 * contact_max + 0.30 * contact_median + 0.20 * contact_peak
    )
    contact_gate = np.clip(
        (contact_history - contact_threshold) / (1.0 - contact_threshold),
        0.0,
        1.0,
    )

    slow_anchor = sample_and_hold(anchor_ratio, slow_stride)
    slow_hidden = sample_and_hold(baseline_ratio, slow_stride)
    slow_contact = sample_and_hold(contact_gate, slow_stride)
    fast_contact_residual = np.maximum(contact_gate - slow_contact, 0.0)

    # A cached slow hidden decision is held only while fresh contact supports
    # it. This avoids trailing occlusion after the hand releases the object.
    held_slow_hidden = baseline_ratio + np.maximum(
        slow_hidden - baseline_ratio, 0.0
    ) * contact_gate
    visual_deficit = np.maximum(visual_prior - slow_anchor, 0.0)
    fast_gate = np.clip(
        (1.0 - contact_weight)
        + contact_weight * contact_gate
        + 0.5 * contact_weight * fast_contact_residual,
        0.0,
        1.0,
    )
    desired_ratio = np.maximum(
        baseline_ratio,
        held_slow_hidden + residual_strength * visual_deficit * fast_gate,
    )
    desired_ratio = np.clip(desired_ratio, 0.0, 1.0)

    return {
        "visual_prior": visual_prior.astype(np.float32),
        "slow_anchor": slow_anchor.astype(np.float32),
        "slow_hidden": slow_hidden.astype(np.float32),
        "contact_history": contact_history.astype(np.float32),
        "contact_gate": contact_gate.astype(np.float32),
        "slow_contact": slow_contact.astype(np.float32),
        "fast_contact_residual": fast_contact_residual.astype(np.float32),
        "contact_state": discrete_contact_state(contact_gate),
        "visual_deficit": visual_deficit.astype(np.float32),
        "desired_ratio": desired_ratio.astype(np.float32),
    }


def expand_from_seed(
    seed: np.ndarray,
    part: np.ndarray,
    allowed: np.ndarray,
    desired_count: int,
) -> np.ndarray:
    """Grow a hidden region locally while staying inside object support."""
    output = np.asarray(seed, dtype=bool) & np.asarray(part, dtype=bool)
    current = int(output.sum())
    if current >= desired_count:
        return output
    candidates = part & allowed & ~output
    need = min(desired_count - current, int(candidates.sum()))
    if need <= 0:
        return output
    positions = np.argwhere(candidates)
    if output.any():
        distance = cv2.distanceTransform(
            (~output).astype(np.uint8), cv2.DIST_L2, 3
        )
        scores = distance[positions[:, 0], positions[:, 1]]
    else:
        center = positions.mean(axis=0)
        scores = ((positions - center) ** 2).sum(axis=1)
    selected = positions[np.argpartition(scores, need - 1)[:need]]
    output[selected[:, 0], selected[:, 1]] = True
    return output


def _finger_ratios(
    hidden_masks: np.ndarray, robot_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    frames = len(hidden_masks)
    ratios = np.zeros((frames, len(FINGER_NAMES)), np.float32)
    totals = np.zeros((frames, len(FINGER_NAMES)), np.int32)
    for frame in range(frames):
        labels = np.asarray(robot_labels[frame], dtype=np.uint8)
        hidden = np.asarray(hidden_masks[frame], dtype=bool)
        for finger in range(len(FINGER_NAMES)):
            part = labels == finger + 1
            totals[frame, finger] = int(part.sum())
            if part.any():
                ratios[frame, finger] = float((hidden & part).sum() / part.sum())
    return ratios, totals


def _put_header(canvas: np.ndarray, text: str, x: int, color) -> None:
    cv2.putText(
        canvas,
        text,
        (x + 16, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )


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
    parser.add_argument("--reference_hidden_mask", type=Path, default=None)
    parser.add_argument("--reference_name", default="Multi-frame only")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--temporal_window", type=int, default=16)
    parser.add_argument("--slow_stride", type=int, default=4)
    parser.add_argument("--contact_threshold", type=float, default=0.20)
    parser.add_argument("--contact_weight", type=float, default=0.65)
    parser.add_argument("--residual_strength", type=float, default=0.85)
    parser.add_argument("--object_dilate_px", type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = np.load(args.baseline_hidden_mask, mmap_mode="r")
    anchor_counts = np.asarray(
        np.load(args.dense_anchor_counts), dtype=np.float32
    )
    robot_rgb = np.load(args.robot_rgb, mmap_mode="r")
    robot_mask = np.load(args.robot_mask, mmap_mode="r")
    robot_labels = np.load(args.robot_finger_labels, mmap_mode="r")
    object_mask = np.load(args.object_mask, mmap_mode="r")
    reference = (
        np.load(args.reference_hidden_mask, mmap_mode="r")
        if args.reference_hidden_mask is not None
        else None
    )
    parts = np.asarray(np.load(args.finger_parts), dtype=np.int32)
    contacts = sorted(args.contact_dir.glob("*.npz"))
    cap = cv2.VideoCapture(str(args.clean_plate))
    if not cap.isOpened():
        raise FileNotFoundError(args.clean_plate)
    frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    inputs = (
        baseline,
        anchor_counts,
        robot_rgb,
        robot_mask,
        robot_labels,
        object_mask,
    )
    if not all(len(value) == frames for value in inputs) or len(contacts) != frames:
        raise ValueError("frame count mismatch")
    if reference is not None and len(reference) != frames:
        raise ValueError("reference hidden-mask frame count mismatch")
    low_h, low_w = robot_mask.shape[1:]
    if baseline.shape[1:] != (low_h, low_w):
        raise ValueError("baseline and robot masks must share resolution")
    if anchor_counts.shape != (frames, len(FINGER_NAMES), 2):
        raise ValueError("dense anchor counts must have shape (frames, 5, 2)")
    if robot_rgb.shape != (frames, low_h, low_w, 3):
        raise ValueError("robot RGB must have shape (frames, height, width, 3)")
    if robot_labels.shape != robot_mask.shape:
        raise ValueError("robot labels and mask must share shape")
    if reference is not None and reference.shape != baseline.shape:
        raise ValueError("reference and baseline hidden masks must share shape")
    if np.setdiff1d(np.unique(robot_labels), np.arange(6)).size:
        raise ValueError("robot finger labels must be in [0, 5]")

    anchor_ratio = np.divide(
        anchor_counts[:, :, 1],
        anchor_counts[:, :, 0],
        out=np.zeros((frames, len(FINGER_NAMES)), np.float32),
        where=anchor_counts[:, :, 0] > 0,
    )
    baseline_ratio, total_pixels = _finger_ratios(baseline, robot_labels)
    contact_mean, contact_max = compute_contact_features(
        contacts, parts, args.side
    )
    signals = build_slow_fast_signals(
        anchor_ratio,
        baseline_ratio,
        contact_max,
        temporal_window=args.temporal_window,
        slow_stride=args.slow_stride,
        contact_threshold=args.contact_threshold,
        contact_weight=args.contact_weight,
        residual_strength=args.residual_strength,
    )

    refined_all = np.zeros((frames, low_h, low_w), dtype=bool)
    added_by_finger = np.zeros(len(FINGER_NAMES), np.int64)
    kernel_size = args.object_dilate_px * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    final_writer = open_writer(
        args.output_dir / "video_trex_haco_slow_fast.mp4",
        fps,
        (width, height),
    )
    panel_count = 3 if reference is not None else 2
    panel_w, panel_h, header = width // 3, height // 3, 70
    compare_writer = open_writer(
        args.output_dir / "video_compare_visibility_methods.mp4",
        fps,
        (panel_w * panel_count, panel_h + header),
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
            for finger in range(len(FINGER_NAMES)):
                part = labels == finger + 1
                before = refined & part
                desired = int(
                    round(signals["desired_ratio"][frame, finger] * part.sum())
                )
                after = expand_from_seed(before, part, obj, desired)
                refined[part] = after[part]
                added_by_finger[finger] += int(after.sum() - before.sum())
            if np.any(base_hidden & ~refined):
                raise AssertionError("fast refinement revealed a baseline-hidden pixel")
            if np.any((refined & ~base_hidden) & ~obj):
                raise AssertionError("fast refinement escaped object support")
            refined_all[frame] = refined

            rmask = np.asarray(robot_mask[frame], dtype=bool)
            rrgb = cv2.resize(
                np.asarray(robot_rgb[frame]),
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

            def composite(hidden: np.ndarray) -> np.ndarray:
                alpha = cv2.resize(
                    (rmask & ~hidden).astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                return np.clip(
                    clean.astype(np.float32) * (1 - alpha[..., None])
                    + rrgb.astype(np.float32) * alpha[..., None],
                    0,
                    255,
                ).astype(np.uint8)

            base_final = composite(base_hidden)
            refined_final = composite(refined)
            final_writer.write(refined_final)
            panels = [base_final]
            if reference is not None:
                panels.append(composite(np.asarray(reference[frame], dtype=bool)))
            panels.append(refined_final)
            canvas = np.full(
                (panel_h + header, panel_w * panel_count, 3), 22, np.uint8
            )
            for index, panel in enumerate(panels):
                start = index * panel_w
                canvas[header:, start:start + panel_w] = cv2.resize(
                    panel, (panel_w, panel_h)
                )
            _put_header(canvas, "Per-frame baseline", 0, (245, 245, 245))
            if reference is not None:
                _put_header(canvas, args.reference_name, panel_w, (190, 220, 255))
            _put_header(
                canvas,
                "Slow vision + fast HaCo",
                panel_w * (panel_count - 1),
                (120, 230, 255),
            )
            compare_writer.write(canvas)
            if (frame + 1) % 100 == 0:
                print(f"[trex-haco] {frame + 1}/{frames}", flush=True)
    finally:
        cap.release()
        final_writer.release()
        compare_writer.release()

    np.save(args.output_dir / "trex_haco_hidden_mask.npy", refined_all)
    np.savez_compressed(
        args.output_dir / "slow_fast_features.npz",
        finger_names=np.asarray(FINGER_NAMES),
        anchor_ratio=anchor_ratio,
        baseline_ratio=baseline_ratio,
        total_robot_pixels=total_pixels,
        contact_mean=contact_mean,
        contact_max=contact_max,
        **signals,
    )
    baseline_pixels = int(np.count_nonzero(baseline))
    refined_pixels = int(np.count_nonzero(refined_all))
    reference_metrics = None
    if reference is not None:
        reference_pixels = int(np.count_nonzero(reference))
        intersection = int(np.count_nonzero(refined_all & reference))
        union = int(np.count_nonzero(refined_all | reference))
        reference_metrics = {
            "name": args.reference_name,
            "hidden_pixels_lowres": reference_pixels,
            "intersection_pixels_lowres": intersection,
            "union_pixels_lowres": union,
            "jaccard": float(intersection / union) if union else 1.0,
            "disagreement_pixels_lowres": int(
                np.count_nonzero(refined_all != reference)
            ),
        }
    report = {
        "schema_version": 2,
        "method": "causal slow-vision fast-HaCo visibility adapter",
        "official_trex_model_used": False,
        "official_trex_checkpoint_used": False,
        "official_reference": {
            "paper": OFFICIAL_PAPER,
            "repository": OFFICIAL_REPOSITORY,
        },
        "reason_official_checkpoint_is_incompatible": (
            "Human2Robot has RGB, MANO and HaCo but not T-Rex's ten tactile "
            "image streams and per-fingertip 6-axis wrench history"
        ),
        "mapped_concepts": {
            "slow_expert": "dense MANO/object visibility sampled and held",
            "fast_expert": "per-frame HaCo contact with causal history",
            "tactile_history_frames": args.temporal_window,
            "cascaded_output": "per-finger hidden-pixel ratio",
        },
        "causal": True,
        "uses_future_frames": False,
        "frames": frames,
        "slow_stride_frames": args.slow_stride,
        "fast_stride_frames": 1,
        "contact_threshold": args.contact_threshold,
        "contact_weight": args.contact_weight,
        "residual_strength": args.residual_strength,
        "baseline_hidden_pixels_lowres": baseline_pixels,
        "final_hidden_pixels_lowres": refined_pixels,
        "added_hidden_pixels_lowres": int(added_by_finger.sum()),
        "added_hidden_pixels_by_finger": {
            name: int(value)
            for name, value in zip(FINGER_NAMES, added_by_finger)
        },
        "invariants": {
            "baseline_hidden_pixels_preserved": True,
            "new_hidden_pixels_require_object_support": True,
            "robot_pixels_outside_object_support_unchanged": True,
        },
        "reference_comparison": reference_metrics,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
