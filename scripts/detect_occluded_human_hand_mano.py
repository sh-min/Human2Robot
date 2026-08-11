#!/usr/bin/env python3
"""Visualize visible versus inferred-occluded HUMAN hand pixels.

This does not use the robot render.  The complete human-hand silhouette comes
from the HaWoR MANO mesh; the modal/actually visible hand comes from SAM2,
restricted to the HaWoR 21-joint hand hull.  Their difference is an occlusion
candidate mask.
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


def hand_roi(points: np.ndarray, height: int, width: int, expand: float):
    finite = np.isfinite(points).all(axis=1)
    roi = np.zeros((height, width), dtype=np.uint8)
    if int(finite.sum()) < 3:
        return roi.astype(bool)
    points = points[finite]
    hull = cv2.convexHull(np.rint(points).astype(np.int32))
    cv2.fillConvexPoly(roi, hull, 1)
    span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0)
    radius = max(1, int(round(expand * span)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    return cv2.dilate(roi, kernel).astype(bool)


def project_mano_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    focal: float,
    width: int,
    height: int,
) -> np.ndarray:
    valid_vertex = np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 1e-5)
    uv = np.zeros((len(vertices), 2), dtype=np.float32)
    uv[valid_vertex, 0] = focal * vertices[valid_vertex, 0] / vertices[valid_vertex, 2] + width / 2
    uv[valid_vertex, 1] = focal * vertices[valid_vertex, 1] / vertices[valid_vertex, 2] + height / 2
    valid_face = valid_vertex[faces].all(axis=1)
    polys = np.rint(uv[faces[valid_face]]).astype(np.int32)
    # Ignore fully offscreen triangles while retaining triangles crossing an edge.
    if len(polys):
        xmin = polys[..., 0].min(axis=1)
        xmax = polys[..., 0].max(axis=1)
        ymin = polys[..., 1].min(axis=1)
        ymax = polys[..., 1].max(axis=1)
        polys = polys[(xmax >= 0) & (xmin < width) & (ymax >= 0) & (ymin < height)]
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(polys):
        cv2.fillPoly(mask, list(polys), 1, lineType=cv2.LINE_8)
    return mask.astype(bool)


def mano_vertex_finger_labels(parts: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(parts), dtype=np.uint8)
    groups = ((13, 14, 15), (1, 2, 3), (4, 5, 6), (10, 11, 12), (7, 8, 9))
    for label, group in enumerate(groups, start=1):
        labels[np.isin(parts, group)] = label
    return labels


def project_mano_finger_labels(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_labels: np.ndarray,
    focal: float,
    width: int,
    height: int,
) -> np.ndarray:
    valid = np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 1e-5)
    uv = np.zeros((len(vertices), 2), dtype=np.float32)
    uv[valid, 0] = focal * vertices[valid, 0] / vertices[valid, 2] + width / 2
    uv[valid, 1] = focal * vertices[valid, 1] / vertices[valid, 2] + height / 2
    output = np.zeros((height, width), dtype=np.uint8)
    for label in range(1, 6):
        selected = valid[faces].all(axis=1) & (
            (vertex_labels[faces] == label).sum(axis=1) >= 2
        )
        polys = np.rint(uv[faces[selected]]).astype(np.int32)
        if len(polys):
            cv2.fillPoly(output, list(polys), label, lineType=cv2.LINE_8)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--visible_human_mask", type=Path, required=True)
    parser.add_argument("--hand_keypoints_npz", type=Path, required=True)
    parser.add_argument("--hawor_npz", type=Path, required=True)
    parser.add_argument("--mano_faces", type=Path, required=True)
    parser.add_argument("--finger_parts", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--keypoint_hull_expand_ratio", type=float, default=0.16)
    parser.add_argument("--visible_support_dilation_px", type=int, default=12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    visible_all = np.load(args.visible_human_mask, mmap_mode="r")
    with np.load(args.hand_keypoints_npz) as hand_data:
        keypoints_all = np.asarray(hand_data["kpts_2d"], dtype=np.float32)
    with np.load(args.hawor_npz) as hawor:
        vertices_all = np.asarray(hawor[f"verts_{args.side}"], dtype=np.float32)
        focal = float(np.asarray(hawor["img_focal"]).item())
        valid_raw = np.asarray(hawor["valid"], dtype=bool)
        side_index = 0 if args.side == "left" else 1
        valid_all = (
            valid_raw[side_index]
            if valid_raw.shape[0] == 2 else valid_raw[:, side_index]
        )
    faces = np.asarray(np.load(args.mano_faces), dtype=np.int32)
    vertex_labels = mano_vertex_finger_labels(
        np.asarray(np.load(args.finger_parts), dtype=np.int32)
    )
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if not (
        len(visible_all) == len(keypoints_all) == len(vertices_all)
        == len(valid_all) == frames
    ):
        raise ValueError("frame count mismatch")

    low_w, low_h = 270, 480
    visible_low = np.zeros((frames, low_h, low_w), dtype=bool)
    occluded_low = np.zeros_like(visible_low)
    full_low = np.zeros_like(visible_low)
    finger_labels_low = np.zeros((frames, low_h, low_w), dtype=np.uint8)
    finger_visible_fraction = np.full((frames, 5), np.nan, dtype=np.float32)
    counts = np.zeros((frames, 3), dtype=np.int64)
    header = 74
    panel_w, panel_h = width // 2, height // 2
    writer = open_writer(
        args.output_dir / "video_human_hand_visibility.mp4v.mp4",
        fps, (panel_w * 2, panel_h + header),
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (args.visible_support_dilation_px * 2 + 1,) * 2,
    )

    try:
        for index in range(frames):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"video ended at frame {index}")
            sam = np.asarray(visible_all[index], dtype=bool).copy()
            if sam.shape != (height, width):
                sam = cv2.resize(
                    sam.astype(np.uint8), (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            sam &= hand_roi(
                keypoints_all[index], height, width,
                args.keypoint_hull_expand_ratio,
            )
            full = (
                project_mano_mask(
                    vertices_all[index], faces, focal, width, height
                ) if valid_all[index] else np.zeros((height, width), dtype=bool)
            )
            finger_labels = (
                project_mano_finger_labels(
                    vertices_all[index], faces, vertex_labels,
                    focal, width, height,
                ) if valid_all[index] else np.zeros((height, width), np.uint8)
            )
            support = cv2.dilate(sam.astype(np.uint8), kernel).astype(bool)
            visible = full & support
            occluded = full & ~support
            counts[index] = (int(full.sum()), int(visible.sum()), int(occluded.sum()))
            full_low[index] = cv2.resize(
                full.astype(np.uint8), (low_w, low_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            visible_low[index] = cv2.resize(
                visible.astype(np.uint8), (low_w, low_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            occluded_low[index] = cv2.resize(
                occluded.astype(np.uint8), (low_w, low_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            finger_labels_low[index] = cv2.resize(
                finger_labels, (low_w, low_h), interpolation=cv2.INTER_NEAREST
            )
            for finger_index in range(5):
                part = finger_labels == finger_index + 1
                if part.any():
                    finger_visible_fraction[index, finger_index] = float(
                        (part & support).sum() / part.sum()
                    )

            left = frame.copy()
            contours, _ = cv2.findContours(
                sam.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(left, contours, -1, (255, 255, 0), 3, cv2.LINE_AA)
            right = frame.astype(np.float32)
            green = np.zeros_like(frame); green[..., 1] = 255
            red = np.zeros_like(frame); red[..., 2] = 255
            right[visible] = right[visible] * 0.35 + green[visible] * 0.65
            right[occluded] = right[occluded] * 0.25 + red[occluded] * 0.75
            right = np.clip(right, 0, 255).astype(np.uint8)

            canvas = np.full((panel_h + header, panel_w * 2, 3), 22, np.uint8)
            canvas[header:, :panel_w] = cv2.resize(left, (panel_w, panel_h))
            canvas[header:, panel_w:] = cv2.resize(right, (panel_w, panel_h))
            cv2.putText(
                canvas, "Visible HUMAN hand (SAM2)", (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.67, (245, 245, 245), 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas, "HUMAN MANO: GREEN visible | RED occluded",
                (panel_w + 18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (245, 245, 245), 2, cv2.LINE_AA,
            )
            fraction = counts[index, 2] / max(counts[index, 0], 1)
            cv2.putText(
                canvas, f"inferred occluded hand area: {fraction * 100:.1f}%",
                (panel_w + 18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (150, 220, 255), 2, cv2.LINE_AA,
            )
            writer.write(canvas)
            if (index + 1) % 100 == 0:
                print(f"[human-hand] {index + 1}/{frames}", flush=True)
    finally:
        cap.release()
        writer.release()

    np.save(args.output_dir / "human_mano_full_mask_lowres.npy", full_low)
    np.save(args.output_dir / "human_hand_visible_mask_lowres.npy", visible_low)
    np.save(args.output_dir / "human_hand_occluded_mask_lowres.npy", occluded_low)
    np.save(args.output_dir / "human_mano_finger_labels_lowres.npy", finger_labels_low)
    np.save(args.output_dir / "human_finger_visible_fraction.npy", finger_visible_fraction)
    np.save(args.output_dir / "pixel_counts.npy", counts)
    report = {
        "schema_version": 1,
        "method": "HaWoR_MANO_human_hand_minus_SAM2_modal_human_hand",
        "robot_render_used": False,
        "frames": frames,
        "full_mano_pixels": int(counts[:, 0].sum()),
        "visible_mano_pixels": int(counts[:, 1].sum()),
        "inferred_occluded_mano_pixels": int(counts[:, 2].sum()),
        "mean_occluded_fraction": float(
            np.mean(counts[:, 2] / np.maximum(counts[:, 0], 1))
        ),
        "warning": "Occluded MANO pixels are model inference, not directly observable ground truth.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
