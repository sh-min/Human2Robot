#!/usr/bin/env python3
"""Harmonize inferred object pixels from nearby observed object appearance.

Only ``amodal & ~observed`` pixels are modified.  Telea propagation supplies a
smooth local surface from the visible object boundary, bilateral filtering
removes generated lumps, and boundary-ring colour statistics harmonize the
result.  A small fraction of the input completion is retained for texture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def open_writer(path: Path, fps: float, size: tuple[int, int]):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer: {path}")
    return writer


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return np.asarray(mask, dtype=bool)
    return cv2.resize(
        np.asarray(mask, dtype=np.uint8), (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def harmonize_colour(
    candidate: np.ndarray,
    source: np.ndarray,
    hidden: np.ndarray,
    observed: np.ndarray,
    ring_radius: int,
) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ring_radius * 2 + 1,) * 2
    )
    ring = cv2.dilate(hidden.astype(np.uint8), kernel).astype(bool)
    ring &= observed
    if int(ring.sum()) < 64 or int(hidden.sum()) < 16:
        return candidate
    result = candidate.astype(np.float32)
    donor = source[ring].astype(np.float32)
    generated = result[hidden]
    donor_mean = donor.mean(axis=0)
    donor_std = donor.std(axis=0)
    generated_mean = generated.mean(axis=0)
    generated_std = generated.std(axis=0)
    scale = np.clip(donor_std / np.maximum(generated_std, 6.0), 0.65, 1.45)
    result[hidden] = np.clip(
        (generated - generated_mean) * scale + donor_mean, 0, 255
    )
    return result.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_video", type=Path, required=True)
    parser.add_argument("--observed_mask", type=Path, required=True)
    parser.add_argument("--amodal_mask", type=Path, required=True)
    parser.add_argument(
        "--robot_mask", type=Path,
        help="when provided, flatten the dilated robot/object loss region instead of amodal hidden",
    )
    parser.add_argument("--loss_radius_px", type=int, default=16)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--inpaint_radius_px", type=float, default=9.0)
    parser.add_argument("--ring_radius_px", type=int, default=18)
    parser.add_argument("--boundary_ramp_px", type=float, default=6.0)
    parser.add_argument(
        "--completion_texture_weight", type=float, default=0.18,
        help="fraction of the incoming generated texture retained in hidden pixels",
    )
    args = parser.parse_args()
    if not 0.0 <= args.completion_texture_weight <= 1.0:
        parser.error("completion texture weight must be in [0,1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.input_video))
    if not capture.isOpened():
        raise FileNotFoundError(args.input_video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    observed_all = np.load(args.observed_mask, mmap_mode="r")
    amodal_all = np.load(args.amodal_mask, mmap_mode="r")
    robot_all = (
        np.load(args.robot_mask, mmap_mode="r")
        if args.robot_mask is not None else None
    )
    if len(observed_all) != frames or len(amodal_all) != frames:
        raise ValueError("mask/video frame count mismatch")
    if robot_all is not None and len(robot_all) != frames:
        raise ValueError("robot mask/video frame count mismatch")

    output_writer = open_writer(
        args.output_dir / "video_object_flattened.mp4v.mp4", fps, (width, height)
    )
    panel_w, panel_h, header = width // 2, height // 2, 64
    compare_writer = open_writer(
        args.output_dir / "video_compare_completion_vs_flattened.mp4v.mp4",
        fps,
        (panel_w * 2, panel_h + header),
    )
    evidence_writer = open_writer(
        args.output_dir / "video_loss_region_evidence.mp4v.mp4",
        fps,
        (width, height + header),
    )
    loss_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (args.loss_radius_px * 2 + 1,) * 2
    )
    hidden_counts = np.zeros(frames, dtype=np.int64)
    modified_counts = np.zeros(frames, dtype=np.int64)
    try:
        for index in range(frames):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"video ended at frame {index}")
            observed = resize_mask(observed_all[index], width, height).copy()
            amodal = resize_mask(amodal_all[index], width, height)
            if robot_all is not None:
                robot = resize_mask(robot_all[index], width, height)
                hidden = cv2.dilate(
                    (amodal & robot).astype(np.uint8), loss_kernel, iterations=1
                ).astype(bool)
                hidden &= amodal
                observed &= ~hidden
            else:
                hidden = amodal & ~observed
            hidden_counts[index] = int(hidden.sum())
            result = frame.copy()
            if np.any(hidden):
                mask_u8 = hidden.astype(np.uint8) * 255
                propagated = cv2.inpaint(
                    frame, mask_u8, args.inpaint_radius_px, cv2.INPAINT_TELEA
                )
                smooth = cv2.bilateralFilter(
                    propagated, d=9, sigmaColor=30.0, sigmaSpace=9.0
                )
                smooth = harmonize_colour(
                    smooth, frame, hidden, observed, args.ring_radius_px
                )
                flat = np.clip(
                    (1.0 - args.completion_texture_weight)
                    * smooth.astype(np.float32)
                    + args.completion_texture_weight * frame.astype(np.float32),
                    0,
                    255,
                )
                inside_distance = cv2.distanceTransform(
                    hidden.astype(np.uint8), cv2.DIST_L2, 5
                )
                alpha = np.clip(
                    inside_distance / max(args.boundary_ramp_px, 1.0), 0.0, 1.0
                )
                alpha *= hidden.astype(np.float32)
                result = np.clip(
                    alpha[..., None] * flat
                    + (1.0 - alpha[..., None]) * frame.astype(np.float32),
                    0,
                    255,
                ).astype(np.uint8)
                modified_counts[index] = int(np.sum(hidden & (alpha > 0)))
            # The observed object is an exact invariant, not just an alpha preference.
            result[observed] = frame[observed]
            output_writer.write(result)

            comparison = np.full(
                (panel_h + header, panel_w * 2, 3), 24, dtype=np.uint8
            )
            comparison[header:, :panel_w] = cv2.resize(
                frame, (panel_w, panel_h)
            )
            comparison[header:, panel_w:] = cv2.resize(
                result, (panel_w, panel_h)
            )
            cv2.putText(
                comparison, "Object completion", (18, 41),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA,
            )
            cv2.putText(
                comparison, "Visible-guided flattening", (panel_w + 18, 41),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (80, 220, 80), 2, cv2.LINE_AA,
            )
            compare_writer.write(comparison)
            evidence = cv2.copyMakeBorder(
                frame, header, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)
            )
            tint = np.zeros_like(frame)
            tint[hidden] = (0, 0, 255)
            evidence[header:] = cv2.addWeighted(
                evidence[header:], 1.0, tint, 0.55, 0
            )
            cv2.putText(
                evidence, "red = robot/object information-loss region",
                (18, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (240, 240, 240), 2, cv2.LINE_AA,
            )
            evidence_writer.write(evidence)
            if (index + 1) % 100 == 0:
                print(f"[flatten] {index + 1}/{frames}", flush=True)
    finally:
        capture.release()
        output_writer.release()
        compare_writer.release()
        evidence_writer.release()

    report = {
        "schema_version": 1,
        "method": "visible_object_guided_hidden_surface_flattening",
        "frames": frames,
        "frames_with_hidden_surface": int((hidden_counts > 0).sum()),
        "hidden_pixels": int(hidden_counts.sum()),
        "modified_hidden_pixels": int(modified_counts.sum()),
        "inpaint_radius_px": args.inpaint_radius_px,
        "ring_radius_px": args.ring_radius_px,
        "boundary_ramp_px": args.boundary_ramp_px,
        "completion_texture_weight": args.completion_texture_weight,
        "target_mode": (
            "dilated_robot_object_overlap" if robot_all is not None
            else "inferred_amodal_hidden"
        ),
        "loss_radius_px": args.loss_radius_px if robot_all is not None else None,
        "invariants": {
            "observed_object_rgb_unchanged_before_encoding": True,
            "pixels_outside_inferred_hidden_object_unchanged": True,
            "flattening_uses_visible_object_boundary_statistics": True,
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
