#!/usr/bin/env python3
"""Hide robot-hand pixels mapped from inferred occluded human-hand pixels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return np.asarray(mask, dtype=bool)
    return cv2.resize(
        np.asarray(mask, dtype=np.uint8), (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def open_writer(path: Path, fps: float, size: tuple[int, int]):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer: {path}")
    return writer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean_plate", type=Path, required=True)
    parser.add_argument("--robot_rgb", type=Path, required=True)
    parser.add_argument("--robot_mask", type=Path, required=True)
    parser.add_argument("--robot_finger_mask", type=Path, required=True)
    parser.add_argument("--robot_finger_labels", type=Path)
    parser.add_argument("--human_occluded_mask", type=Path, required=True)
    parser.add_argument("--human_finger_labels", type=Path)
    parser.add_argument("--object_mask", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mapping_dilate_px", type=int, default=1)
    parser.add_argument("--mapping_close_px", type=int, default=2)
    parser.add_argument("--semantic_finger_mapping", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    robot_rgb = np.load(args.robot_rgb, mmap_mode="r")
    robot_mask = np.load(args.robot_mask, mmap_mode="r")
    robot_finger = np.load(args.robot_finger_mask, mmap_mode="r")
    robot_labels = (
        np.load(args.robot_finger_labels, mmap_mode="r")
        if args.robot_finger_labels is not None else None
    )
    human_hidden = np.load(args.human_occluded_mask, mmap_mode="r")
    human_labels = (
        np.load(args.human_finger_labels, mmap_mode="r")
        if args.human_finger_labels is not None else None
    )
    object_mask = np.load(args.object_mask, mmap_mode="r")
    cap = cv2.VideoCapture(str(args.clean_plate))
    if not cap.isOpened():
        raise FileNotFoundError(args.clean_plate)
    frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if not (
        len(robot_rgb) == len(robot_mask) == len(robot_finger)
        == len(human_hidden) == len(object_mask) == frames
    ):
        raise ValueError("frame count mismatch")
    if args.semantic_finger_mapping and (
        robot_labels is None or human_labels is None
    ):
        parser.error(
            "--semantic_finger_mapping requires --robot_finger_labels and "
            "--human_finger_labels"
        )

    low_h, low_w = robot_mask.shape[1:]
    mapped_masks = np.zeros((frames, low_h, low_w), dtype=bool)
    hidden_counts = np.zeros(frames, dtype=np.int64)
    final_writer = open_writer(
        args.output_dir / "video_human_visibility_mapped_robot.mp4v.mp4",
        fps, (width, height),
    )
    panel_w, panel_h, header = width // 3, height // 3, 68
    compare_writer = open_writer(
        args.output_dir / "video_compare_mapping.mp4v.mp4",
        fps, (panel_w * 3, panel_h + header),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (args.mapping_close_px * 2 + 1,) * 2,
    )
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (args.mapping_dilate_px * 2 + 1,) * 2,
    )

    try:
        for index in range(frames):
            ok, clean = cap.read()
            if not ok:
                raise RuntimeError(f"clean plate ended at frame {index}")
            rmask_low = np.asarray(robot_mask[index], dtype=bool)
            finger_low = np.asarray(robot_finger[index], dtype=bool)
            hidden_low = resize_mask(human_hidden[index], low_w, low_h)
            obj_low = resize_mask(object_mask[index], low_w, low_h)
            if args.semantic_finger_mapping:
                mapped = np.zeros((low_h, low_w), dtype=bool)
                # Preserve integer identities after any required resize.
                raw_hlabels = np.asarray(human_labels[index], dtype=np.uint8)
                if raw_hlabels.shape != (low_h, low_w):
                    raw_hlabels = cv2.resize(
                        raw_hlabels, (low_w, low_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                raw_rlabels = np.asarray(robot_labels[index], dtype=np.uint8)
                if raw_rlabels.shape != (low_h, low_w):
                    raw_rlabels = cv2.resize(
                        raw_rlabels, (low_w, low_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                for label in range(1, 6):
                    human_part = raw_hlabels == label
                    robot_part = raw_rlabels == label
                    if not human_part.any() or not robot_part.any():
                        continue
                    hy, hx = np.where(human_part)
                    ry, rx = np.where(robot_part)
                    hya, hyb = int(hy.min()), int(hy.max()) + 1
                    hxa, hxb = int(hx.min()), int(hx.max()) + 1
                    rya, ryb = int(ry.min()), int(ry.max()) + 1
                    rxa, rxb = int(rx.min()), int(rx.max()) + 1
                    human_pattern = hidden_low[hya:hyb, hxa:hxb] & human_part[hya:hyb, hxa:hxb]
                    warped = cv2.resize(
                        human_pattern.astype(np.uint8),
                        (rxb - rxa, ryb - rya),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    mapped[rya:ryb, rxa:rxb] |= (
                        warped & robot_part[rya:ryb, rxa:rxb]
                    )
                mapped &= finger_low
            else:
                mapped = hidden_low & finger_low & obj_low
            if args.mapping_close_px > 0:
                mapped = cv2.morphologyEx(
                    mapped.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel
                ).astype(bool)
                mapped &= finger_low
                if not args.semantic_finger_mapping:
                    mapped &= obj_low
            if args.mapping_dilate_px > 0:
                mapped = cv2.dilate(
                    mapped.astype(np.uint8), dilate_kernel
                ).astype(bool)
                mapped &= finger_low
                if not args.semantic_finger_mapping:
                    mapped &= obj_low
            mapped_masks[index] = mapped
            hidden_counts[index] = int(mapped.sum())

            rmask = resize_mask(rmask_low, width, height)
            mapped_hi = resize_mask(mapped, width, height)
            rrgb = cv2.resize(
                np.asarray(robot_rgb[index]), (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
            alpha_simple = rmask.astype(np.float32)
            alpha_mapped = alpha_simple.copy()
            alpha_mapped[mapped_hi] = 0.0
            simple = np.clip(
                clean.astype(np.float32) * (1.0 - alpha_simple[..., None])
                + rrgb.astype(np.float32) * alpha_simple[..., None], 0, 255,
            ).astype(np.uint8)
            final = np.clip(
                clean.astype(np.float32) * (1.0 - alpha_mapped[..., None])
                + rrgb.astype(np.float32) * alpha_mapped[..., None], 0, 255,
            ).astype(np.uint8)
            final_writer.write(final)

            evidence = simple.copy()
            red = np.zeros_like(evidence); red[..., 2] = 255
            evidence[mapped_hi] = (
                evidence[mapped_hi].astype(np.float32) * 0.25
                + red[mapped_hi].astype(np.float32) * 0.75
            ).astype(np.uint8)
            canvas = np.full(
                (panel_h + header, panel_w * 3, 3), 22, dtype=np.uint8
            )
            for column, image in enumerate((simple, evidence, final)):
                canvas[header:, column * panel_w:(column + 1) * panel_w] = (
                    cv2.resize(image, (panel_w, panel_h))
                )
            labels = (
                "1 Robot over restored object",
                "2 Mapped hidden robot pixels",
                "3 Invisible human part hidden",
            )
            for column, label in enumerate(labels):
                cv2.putText(
                    canvas, label, (column * panel_w + 12, 41),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53, (245, 245, 245), 2,
                    cv2.LINE_AA,
                )
            compare_writer.write(canvas)
            if (index + 1) % 100 == 0:
                print(f"[map] {index + 1}/{frames}", flush=True)
    finally:
        cap.release()
        final_writer.release()
        compare_writer.release()

    np.save(args.output_dir / "mapped_occluded_robot_finger_mask.npy", mapped_masks)
    report = {
        "schema_version": 1,
        "method": "camera_space_human_MANO_occlusion_to_robot_finger_mapping",
        "frames": frames,
        "mapped_hidden_robot_pixels_lowres": int(hidden_counts.sum()),
        "frames_with_hidden_robot_pixels": int((hidden_counts > 0).sum()),
        "mapping_rule": (
            "per-finger normalized human-occlusion pattern transferred to matching robot finger"
            if args.semantic_finger_mapping
            else "human_occluded_MANO & robot_finger & restored_object"
        ),
        "semantic_finger_mapping": args.semantic_finger_mapping,
        "semantic_correspondence": (
            "thumb/index/middle/ring/pinky normalized-bbox mask transfer"
            if args.semantic_finger_mapping else None
        ),
        "mapping_dilate_px_lowres": args.mapping_dilate_px,
        "mapping_close_px_lowres": args.mapping_close_px,
        "robot_arm_is_never_hidden_by_this_mapping": True,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
