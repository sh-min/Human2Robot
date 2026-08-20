#!/usr/bin/env python3
"""Compare rendered XHand pixels with the SAM2 modal human-hand mask.

Green pixels are supported by a hand/arm pixel that is actually visible in the
source frame.  Red pixels belong to the rendered hand estimate but have no
nearby modal hand evidence, so they are candidates for object occlusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def open_writer(path: Path, fps: float, size: tuple[int, int]):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer: {path}")
    return writer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--visible_human_mask", type=Path, required=True)
    parser.add_argument("--robot_finger_mask", type=Path, required=True)
    parser.add_argument("--robot_finger_labels", type=Path, required=True)
    parser.add_argument(
        "--hand_bbox_npz", type=Path,
        help="optional HaWoR bbox_data.npz used to remove the forearm",
    )
    parser.add_argument("--bbox_side", choices=("left", "right"), default="left")
    parser.add_argument("--bbox_expand_ratio", type=float, default=0.18)
    parser.add_argument(
        "--hand_keypoints_npz", type=Path,
        help="optional hand_data_<side>.npz; its 21-joint hull is preferred over bbox",
    )
    parser.add_argument("--keypoint_hull_expand_ratio", type=float, default=0.16)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--support_dilation_px",
        type=int,
        default=3,
        help="dilation in the low-resolution robot-render coordinate system",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    visible_all = np.load(args.visible_human_mask, mmap_mode="r")
    finger_all = np.load(args.robot_finger_mask, mmap_mode="r")
    labels_all = np.load(args.robot_finger_labels, mmap_mode="r")
    bbox_all = None
    if args.hand_bbox_npz is not None:
        with np.load(args.hand_bbox_npz) as bbox_data:
            bbox_all = np.asarray(
                bbox_data[f"{args.bbox_side}_bboxes"], dtype=np.float32
            )
    keypoints_all = None
    if args.hand_keypoints_npz is not None:
        with np.load(args.hand_keypoints_npz) as hand_data:
            keypoints_all = np.asarray(hand_data["kpts_2d"], dtype=np.float32)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if not (len(visible_all) == len(finger_all) == len(labels_all) == frames):
        raise ValueError("frame count mismatch")
    if bbox_all is not None and len(bbox_all) != frames:
        raise ValueError("hand bbox frame count mismatch")
    if keypoints_all is not None and keypoints_all.shape != (frames, 21, 2):
        raise ValueError("hand keypoints must have shape (frames,21,2)")
    low_h, low_w = finger_all.shape[1:]
    if labels_all.shape != finger_all.shape:
        raise ValueError("finger mask/label shape mismatch")

    hidden_all = np.zeros(finger_all.shape, dtype=bool)
    supported_all = np.zeros_like(hidden_all)
    fractions = np.full((frames, len(FINGER_NAMES)), np.nan, np.float32)
    counts = np.zeros((frames, len(FINGER_NAMES)), np.int32)
    header = 74
    panel_w, panel_h = width // 2, height // 2
    writer = open_writer(
        args.output_dir / "video_hand_visibility_detection.mp4v.mp4",
        fps,
        (panel_w * 2, panel_h + header),
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (args.support_dilation_px * 2 + 1,) * 2,
    )

    try:
        for index in range(frames):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"video ended at frame {index}")
            modal_hi = np.asarray(visible_all[index], dtype=bool).copy()
            if modal_hi.shape != (height, width):
                modal_hi = cv2.resize(
                    modal_hi.astype(np.uint8), (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if keypoints_all is not None:
                points = keypoints_all[index]
                finite = np.isfinite(points).all(axis=1)
                hand_roi = np.zeros((height, width), dtype=np.uint8)
                if int(finite.sum()) >= 3:
                    valid_points = points[finite]
                    hull = cv2.convexHull(np.rint(valid_points).astype(np.int32))
                    cv2.fillConvexPoly(hand_roi, hull, 1)
                    span = max(
                        float(np.ptp(valid_points[:, 0])),
                        float(np.ptp(valid_points[:, 1])),
                        1.0,
                    )
                    radius = max(1, int(round(args.keypoint_hull_expand_ratio * span)))
                    kernel_hi = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
                    )
                    hand_roi = cv2.dilate(hand_roi, kernel_hi)
                modal_hi &= hand_roi.astype(bool)
            elif bbox_all is not None:
                x1, y1, x2, y2 = bbox_all[index]
                span = max(float(x2 - x1), float(y2 - y1), 1.0)
                margin = args.bbox_expand_ratio * span
                xa = int(np.clip(np.floor(x1 - margin), 0, width))
                ya = int(np.clip(np.floor(y1 - margin), 0, height))
                xb = int(np.clip(np.ceil(x2 + margin), 0, width))
                yb = int(np.clip(np.ceil(y2 + margin), 0, height))
                hand_roi = np.zeros((height, width), dtype=bool)
                hand_roi[ya:yb, xa:xb] = True
                modal_hi &= hand_roi
            modal_low = cv2.resize(
                modal_hi.astype(np.uint8),
                (low_w, low_h),
                interpolation=cv2.INTER_NEAREST,
            )
            if args.support_dilation_px > 0:
                modal_low = cv2.dilate(modal_low, kernel)
            modal_low = modal_low.astype(bool)
            finger = np.asarray(finger_all[index], dtype=bool)
            labels = np.asarray(labels_all[index], dtype=np.uint8)
            supported = finger & modal_low
            hidden = finger & ~modal_low
            supported_all[index] = supported
            hidden_all[index] = hidden

            for finger_index in range(len(FINGER_NAMES)):
                part = finger & (labels == finger_index + 1)
                count = int(part.sum())
                counts[index, finger_index] = count
                if count:
                    fractions[index, finger_index] = float(
                        (part & modal_low).sum() / count
                    )

            supported_hi = cv2.resize(
                supported.astype(np.uint8), (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            hidden_hi = cv2.resize(
                hidden.astype(np.uint8), (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            left = frame.copy()
            contours, _ = cv2.findContours(
                modal_hi.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(left, contours, -1, (255, 255, 0), 3, cv2.LINE_AA)
            right = frame.astype(np.float32)
            green = np.zeros_like(frame)
            green[..., 1] = 255
            red = np.zeros_like(frame)
            red[..., 2] = 255
            right[supported_hi] = right[supported_hi] * 0.35 + green[supported_hi] * 0.65
            right[hidden_hi] = right[hidden_hi] * 0.25 + red[hidden_hi] * 0.75
            right = np.clip(right, 0, 255).astype(np.uint8)

            canvas = np.full(
                (panel_h + header, panel_w * 2, 3), 22, dtype=np.uint8
            )
            canvas[header:, :panel_w] = cv2.resize(left, (panel_w, panel_h))
            canvas[header:, panel_w:] = cv2.resize(right, (panel_w, panel_h))
            cv2.putText(
                canvas, "SAM2 visible hand only (cyan)", (18, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas, "GREEN visible | RED estimated but invisible",
                (panel_w + 18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                (245, 245, 245), 2, cv2.LINE_AA,
            )
            values = fractions[index]
            summary = "  ".join(
                f"{name[0].upper()}:{value * 100:.0f}%"
                if np.isfinite(value) else f"{name[0].upper()}:?"
                for name, value in zip(FINGER_NAMES, values)
            )
            cv2.putText(
                canvas, summary, (panel_w + 18, 61),
                cv2.FONT_HERSHEY_SIMPLEX, 0.49, (170, 230, 255), 1,
                cv2.LINE_AA,
            )
            writer.write(canvas)
            if (index + 1) % 100 == 0:
                print(f"[visibility] {index + 1}/{frames}", flush=True)
    finally:
        cap.release()
        writer.release()

    np.save(args.output_dir / "estimated_invisible_robot_hand_lowres.npy", hidden_all)
    np.save(args.output_dir / "observed_supported_robot_hand_lowres.npy", supported_all)
    np.savez_compressed(
        args.output_dir / "finger_visibility.npz",
        finger_names=np.asarray(FINGER_NAMES),
        visible_fraction=fractions,
        projected_pixel_count=counts,
    )
    valid = np.isfinite(fractions)
    mean_fraction = np.divide(
        np.nansum(fractions, axis=0), valid.sum(axis=0),
        out=np.zeros(len(FINGER_NAMES), np.float64), where=valid.sum(axis=0) > 0,
    )
    report = {
        "schema_version": 1,
        "method": "SAM2_modal_hand_support_on_rendered_XHand_pixels",
        "frames": frames,
        "support_dilation_lowres_px": args.support_dilation_px,
        "hand_only_bbox": bbox_all is not None,
        "bbox_expand_ratio": args.bbox_expand_ratio if bbox_all is not None else None,
        "hand_only_keypoint_hull": keypoints_all is not None,
        "keypoint_hull_expand_ratio": (
            args.keypoint_hull_expand_ratio if keypoints_all is not None else None
        ),
        "visible_projected_pixels": int(supported_all.sum()),
        "estimated_but_invisible_projected_pixels": int(hidden_all.sum()),
        "mean_visible_fraction_by_finger": {
            name: float(value) for name, value in zip(FINGER_NAMES, mean_fraction)
        },
        "interpretation": {
            "green": "rendered hand pixel supported by actually visible SAM2 hand/arm",
            "red": "rendered hand estimate without nearby visible hand evidence",
            "warning": "red is an occlusion candidate, not geometric ground truth",
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
