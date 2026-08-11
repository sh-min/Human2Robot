#!/usr/bin/env python3
"""Apply ObMan joint hand/object contact surfaces to an existing robot overlay.

The official ObMan object mesh is z-buffered in the same camera coordinate
system as the HaWoR-aligned XHand render.  On confidence-gated frames, robot
finger pixels are removed only where they are (a) locally supported by an
ObMan contact-surface vertex and (b) behind the predicted object surface.
Rejected ObMan frames are left exactly as the existing HaCo overlay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def project(points: np.ndarray, focal: float, width: int, height: int):
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1.0e-4)
    uv = np.full((len(points), 2), np.nan, dtype=np.float32)
    uv[valid, 0] = focal * points[valid, 0] / points[valid, 2] + width / 2
    uv[valid, 1] = focal * points[valid, 1] / points[valid, 2] + height / 2
    return uv, valid


def rasterize_surface_depth(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    focal: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a painter-z-buffer approximation and its mesh silhouette."""
    uv, valid = project(vertices, focal, width, height)
    face_valid = valid[faces].all(axis=1)
    face_ids = np.flatnonzero(face_valid)
    # Draw far to near so the nearest triangle owns overlapping pixels.
    face_ids = face_ids[np.argsort(vertices[faces[face_ids], 2].mean(axis=1))[::-1]]
    depth = np.zeros((height, width), dtype=np.float32)
    silhouette = np.zeros((height, width), dtype=np.uint8)
    for face_id in face_ids:
        triangle = np.rint(uv[faces[face_id]]).astype(np.int32)
        if (
            triangle[:, 0].max() < 0
            or triangle[:, 1].max() < 0
            or triangle[:, 0].min() >= width
            or triangle[:, 1].min() >= height
        ):
            continue
        z = float(vertices[faces[face_id], 2].mean())
        cv2.fillConvexPoly(depth, triangle, z, lineType=cv2.LINE_8)
        cv2.fillConvexPoly(silhouette, triangle, 1, lineType=cv2.LINE_8)
    return depth, silhouette.astype(bool)


def contact_support_mask(
    vertices: np.ndarray,
    contact: np.ndarray,
    *,
    focal: float,
    width: int,
    height: int,
    radius_px: int,
) -> np.ndarray:
    uv, valid = project(vertices, focal, width, height)
    support = np.zeros((height, width), dtype=np.uint8)
    for point in uv[valid & contact]:
        center = tuple(np.rint(point).astype(int))
        cv2.circle(support, center, radius_px, 1, -1, cv2.LINE_8)
    return support.astype(bool)


def open_writer(path: Path, fps: float, size: tuple[int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {path}")
    return writer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--obman_report", type=Path, required=True)
    parser.add_argument("--hawor_npz", type=Path, required=True)
    parser.add_argument("--overlay_arrays", type=Path, required=True)
    parser.add_argument("--baseline_overlay", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--baseline_occlusion", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_alignment_rmse_mm", type=float, default=15.0)
    parser.add_argument("--max_penetrating_vertices", type=int, default=100)
    parser.add_argument("--contact_radius_px", type=int, default=7)
    parser.add_argument("--object_depth_margin_m", type=float, default=0.005)
    parser.add_argument("--thumb_half_thickness_m", type=float, default=0.01958)
    parser.add_argument("--finger_half_thickness_m", type=float, default=0.01465)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence = np.load(args.sequence)
    vertices = np.asarray(sequence["object_vertices"], dtype=np.float32)
    faces = np.asarray(sequence["object_faces"], dtype=np.int32)
    contact = np.asarray(sequence["object_contact"], dtype=bool)
    frame_indices = np.asarray(sequence["frame_indices"], dtype=np.int32)
    report_items = {
        int(item["frame_index"]): item
        for item in json.loads(args.obman_report.read_text())["frames"]
    }
    with np.load(args.hawor_npz) as hawor:
        source_focal = float(hawor["img_focal"])

    robot_depth = np.load(args.overlay_arrays / "robot_depth.npy", mmap_mode="r")
    finger_labels = np.load(
        args.overlay_arrays / "robot_finger_labels.npy", mmap_mode="r"
    )
    baseline_occlusion = np.load(args.baseline_occlusion, mmap_mode="r")
    frames, render_height, render_width = robot_depth.shape
    if len(frame_indices) != frames or len(vertices) != frames:
        raise ValueError("ObMan sequence and robot render frame counts differ")
    render_focal = source_focal * render_width / 1920.0

    baseline_capture = cv2.VideoCapture(str(args.baseline_overlay))
    background_capture = cv2.VideoCapture(str(args.background))
    if not baseline_capture.isOpened() or not background_capture.isOpened():
        raise FileNotFoundError("could not open baseline overlay or background")
    width = int(baseline_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(baseline_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(baseline_capture.get(cv2.CAP_PROP_FPS) or 30.0)
    if int(round(baseline_capture.get(cv2.CAP_PROP_FRAME_COUNT))) != frames:
        raise ValueError("baseline overlay frame count mismatch")

    overlay_tmp = output_dir / "video_overlay_obman_contact.mp4v.mp4"
    compare_tmp = output_dir / "video_compare_haco_vs_obman.mp4v.mp4"
    debug_tmp = output_dir / "video_obman_contact_debug.mp4v.mp4"
    overlay_writer = open_writer(overlay_tmp, fps, (width, height))
    panel_width, panel_height, header = width // 2, height // 2, 64
    compare_writer = open_writer(
        compare_tmp, fps, (panel_width * 2, panel_height + header)
    )
    debug_writer = open_writer(debug_tmp, fps, (width, height + header))

    reliable = np.zeros(frames, dtype=bool)
    added_counts = np.zeros(frames, dtype=np.int64)
    total_counts = np.zeros(frames, dtype=np.int64)
    try:
        for sequence_index, frame_index in enumerate(frame_indices):
            ok_base, baseline = baseline_capture.read()
            ok_bg, background = background_capture.read()
            if not ok_base or not ok_bg:
                raise RuntimeError(f"video read failed at frame {frame_index}")
            if background.shape[:2] != (height, width):
                background = cv2.resize(background, (width, height))
            item = report_items[int(frame_index)]
            is_reliable = (
                float(item["alignment_rmse_mm"])
                <= args.max_alignment_rmse_mm
                and int(item["hand_penetrating_vertices"])
                <= args.max_penetrating_vertices
            )
            reliable[frame_index] = is_reliable
            occluded_low = np.zeros((render_height, render_width), dtype=bool)
            surface_depth = np.zeros_like(occluded_low, dtype=np.float32)
            support = np.zeros_like(occluded_low)
            silhouette = np.zeros_like(occluded_low)
            if is_reliable and contact[sequence_index].any():
                surface_depth, silhouette = rasterize_surface_depth(
                    vertices[sequence_index], faces,
                    focal=render_focal,
                    width=render_width,
                    height=render_height,
                )
                support = contact_support_mask(
                    vertices[sequence_index], contact[sequence_index],
                    focal=render_focal,
                    width=render_width,
                    height=render_height,
                    radius_px=args.contact_radius_px,
                )
                labels = np.asarray(finger_labels[frame_index])
                depth = np.asarray(robot_depth[frame_index])
                thickness = np.where(
                    labels == 1,
                    args.thumb_half_thickness_m,
                    args.finger_half_thickness_m,
                )
                occluded_low = (
                    (labels > 0)
                    & (depth > 0)
                    & silhouette
                    & support
                    & (depth + thickness > surface_depth + args.object_depth_margin_m)
                )

            occluded_full = cv2.resize(
                occluded_low.astype(np.uint8),
                (width, height), interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            baseline_mask = np.asarray(baseline_occlusion[frame_index], dtype=bool)
            if baseline_mask.shape != (height, width):
                baseline_mask = cv2.resize(
                    baseline_mask.astype(np.uint8),
                    (width, height), interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            added = occluded_full & ~baseline_mask
            total_counts[frame_index] = int(occluded_full.sum())
            added_counts[frame_index] = int(added.sum())
            alpha = cv2.GaussianBlur(
                added.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0
            )[..., None]
            result = np.clip(
                baseline.astype(np.float32) * (1.0 - alpha)
                + background.astype(np.float32) * alpha,
                0, 255,
            ).astype(np.uint8)
            overlay_writer.write(result)

            left = cv2.resize(baseline, (panel_width, panel_height))
            right = cv2.resize(result, (panel_width, panel_height))
            compare = np.full(
                (panel_height + header, panel_width * 2, 3), 24, np.uint8
            )
            compare[header:, :panel_width] = left
            compare[header:, panel_width:] = right
            cv2.putText(compare, "HaCo baseline", (24, 41),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2,
                        cv2.LINE_AA)
            state = "ObMan contact + HaCo" if is_reliable else "HaCo fallback"
            cv2.putText(compare, state, (panel_width + 24, 41),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (80, 220, 80) if is_reliable else (0, 190, 255), 2,
                        cv2.LINE_AA)
            compare_writer.write(compare)

            debug = cv2.copyMakeBorder(
                result, header, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)
            )
            color_mask = np.zeros_like(result)
            color_mask[occluded_full] = (0, 0, 255)
            debug[header:] = cv2.addWeighted(debug[header:], 1.0, color_mask, 0.55, 0)
            cv2.putText(
                debug,
                f"frame {frame_index:04d} | {'RELIABLE' if is_reliable else 'FALLBACK'} | red=ObMan hidden pixels",
                (24, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (80, 220, 80) if is_reliable else (0, 190, 255), 2,
                cv2.LINE_AA,
            )
            debug_writer.write(debug)
            if (sequence_index + 1) % 100 == 0:
                print(f"[overlay] {sequence_index + 1}/{frames}", flush=True)
    finally:
        baseline_capture.release()
        background_capture.release()
        overlay_writer.release()
        compare_writer.release()
        debug_writer.release()

    np.save(output_dir / "obman_occluded_pixel_count.npy", total_counts)
    summary = {
        "schema_version": 1,
        "method": "official ObMan joint MANO+AtlasNet contact surface depth gate",
        "frames": int(frames),
        "reliable_frames": int(reliable.sum()),
        "frames_with_obman_occlusion": int((total_counts > 0).sum()),
        "frames_with_added_occlusion_vs_haco": int((added_counts > 0).sum()),
        "obman_occluded_pixels_total": int(total_counts.sum()),
        "added_pixels_vs_haco_total": int(added_counts.sum()),
        "fallback": "existing HaCo overlay on rejected ObMan frames",
        "gates": {
            "max_alignment_rmse_mm": args.max_alignment_rmse_mm,
            "max_penetrating_vertices": args.max_penetrating_vertices,
            "contact_radius_render_px": args.contact_radius_px,
            "object_depth_margin_m": args.object_depth_margin_m,
            "thumb_half_thickness_m": args.thumb_half_thickness_m,
            "finger_half_thickness_m": args.finger_half_thickness_m,
        },
        "sources": {key: str(value.resolve()) for key, value in {
            "sequence": args.sequence,
            "obman_report": args.obman_report,
            "hawor_npz": args.hawor_npz,
            "overlay_arrays": args.overlay_arrays,
            "baseline_overlay": args.baseline_overlay,
            "background": args.background,
            "baseline_occlusion": args.baseline_occlusion,
        }.items()},
    }
    (output_dir / "report.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
