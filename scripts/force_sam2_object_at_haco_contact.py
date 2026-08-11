#!/usr/bin/env python3
"""Force the SAM2 modal object in front of robot pixels at HaCo contacts.

HaCo MANO contact vertices are projected into the output image.  Within a
local disk around those projections, any rendered robot pixel that overlaps
the SAM2 modal object mask is replaced by the observed object RGB.  This is a
deliberately depth-free ablation: contact-supported object pixels always win.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def project(points: np.ndarray, focal: float, width: int, height: int):
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1.0e-5)
    uv = np.full((len(points), 2), np.nan, dtype=np.float32)
    uv[valid, 0] = focal * points[valid, 0] / points[valid, 2] + width / 2
    uv[valid, 1] = focal * points[valid, 1] / points[valid, 2] + height / 2
    return uv, valid


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return np.asarray(mask, dtype=bool)
    return cv2.resize(
        np.asarray(mask, dtype=np.uint8), (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def open_writer(path: Path, fps: float, size: tuple[int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer: {path}")
    return writer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--raw_video", type=Path, required=True)
    parser.add_argument("--sam2_mask", type=Path, required=True)
    parser.add_argument("--robot_mask", type=Path, required=True)
    parser.add_argument("--contact_dir", type=Path, required=True)
    parser.add_argument("--hawor_npz", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--contact_radius_px", type=int, default=30)
    parser.add_argument("--edge_sigma_px", type=float, default=1.0)
    parser.add_argument(
        "--force_scope",
        choices=("local-contact", "full-object-overlap"),
        default="local-contact",
        help=(
            "local-contact only restores pixels near projected HaCo vertices; "
            "full-object-overlap uses that local intersection as a trigger and "
            "restores the complete SAM2-object/robot overlap"
        ),
    )
    parser.add_argument(
        "--min_trigger_pixels", type=int, default=1,
        help="minimum local object/robot/contact intersection needed to trigger a frame",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_capture = cv2.VideoCapture(str(args.overlay))
    raw_capture = cv2.VideoCapture(str(args.raw_video))
    if not overlay_capture.isOpened() or not raw_capture.isOpened():
        raise FileNotFoundError("could not open overlay/raw video")
    width = int(overlay_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(overlay_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(round(overlay_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(overlay_capture.get(cv2.CAP_PROP_FPS) or 30.0)
    masks = np.load(args.sam2_mask, mmap_mode="r")
    robot_masks = np.load(args.robot_mask, mmap_mode="r")
    contact_files = sorted(args.contact_dir.glob("*.npz"))
    if len(masks) != frames or len(robot_masks) != frames or len(contact_files) != frames:
        raise ValueError("frame count mismatch among video, SAM2, robot, and HaCo")
    with np.load(args.hawor_npz) as hawor:
        focal = float(hawor["img_focal"])

    output_tmp = output_dir / "video_overlay_contact_object_front.mp4v.mp4"
    compare_tmp = output_dir / "video_compare_depth_vs_forced_contact.mp4v.mp4"
    debug_tmp = output_dir / "video_forced_contact_mask.mp4v.mp4"
    output_writer = open_writer(output_tmp, fps, (width, height))
    panel_width, panel_height, header = width // 2, height // 2, 64
    compare_writer = open_writer(
        compare_tmp, fps, (panel_width * 2, panel_height + header)
    )
    debug_writer = open_writer(debug_tmp, fps, (width, height + header))
    forced_counts = np.zeros(frames, dtype=np.int64)
    contact_counts = np.zeros(frames, dtype=np.int64)
    trigger_counts = np.zeros(frames, dtype=np.int64)

    try:
        for frame_index, contact_path in enumerate(contact_files):
            ok_overlay, overlay = overlay_capture.read()
            ok_raw, raw = raw_capture.read()
            if not ok_overlay or not ok_raw:
                raise RuntimeError(f"video read failed at {frame_index}")
            if raw.shape[:2] != (height, width):
                raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_AREA)
            object_mask = resize_mask(masks[frame_index], width, height)
            robot_mask = resize_mask(robot_masks[frame_index], width, height)
            support = np.zeros((height, width), dtype=np.uint8)
            with np.load(contact_path) as contact:
                is_valid = bool(contact[f"{args.side}_valid"])
                vertices = np.asarray(
                    contact[f"{args.side}_contact_verts_3d"], dtype=np.float32
                )
            if is_valid and len(vertices):
                uv, valid = project(vertices, focal, width, height)
                for point in uv[valid]:
                    cv2.circle(
                        support, tuple(np.rint(point).astype(int)),
                        args.contact_radius_px, 1, -1, cv2.LINE_8,
                    )
            contact_counts[frame_index] = int(len(vertices))
            local_contact = object_mask & robot_mask & support.astype(bool)
            trigger_counts[frame_index] = int(local_contact.sum())
            if (
                args.force_scope == "full-object-overlap"
                and trigger_counts[frame_index] >= args.min_trigger_pixels
            ):
                forced = object_mask & robot_mask
            else:
                forced = local_contact
            forced_counts[frame_index] = int(forced.sum())
            alpha = forced.astype(np.float32)
            if args.edge_sigma_px > 0 and forced.any():
                alpha = cv2.GaussianBlur(
                    alpha, (0, 0), args.edge_sigma_px, args.edge_sigma_px
                )
            alpha = np.clip(alpha, 0.0, 1.0)[..., None]
            result = np.clip(
                overlay.astype(np.float32) * (1.0 - alpha)
                + raw.astype(np.float32) * alpha,
                0, 255,
            ).astype(np.uint8)
            output_writer.write(result)

            left = cv2.resize(overlay, (panel_width, panel_height))
            right = cv2.resize(result, (panel_width, panel_height))
            compare = np.full(
                (panel_height + header, panel_width * 2, 3), 24, dtype=np.uint8
            )
            compare[header:, :panel_width] = left
            compare[header:, panel_width:] = right
            cv2.putText(
                compare, "Depth-gated SAM2 + HaCo", (18, 41),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                compare, "Forced object front at HaCo vertices",
                (panel_width + 18, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.66,
                (80, 220, 80), 2, cv2.LINE_AA,
            )
            compare_writer.write(compare)

            debug = cv2.copyMakeBorder(
                result, header, 0, 0, 0, cv2.BORDER_CONSTANT,
                value=(24, 24, 24),
            )
            tint = np.zeros_like(result)
            tint[forced] = (0, 0, 255)
            debug[header:] = cv2.addWeighted(debug[header:], 1.0, tint, 0.6, 0)
            cv2.putText(
                debug,
                f"frame {frame_index:04d} | HaCo vertices {len(vertices)} | red=robot removed/object restored",
                (18, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (80, 220, 80), 2, cv2.LINE_AA,
            )
            debug_writer.write(debug)
            if (frame_index + 1) % 100 == 0:
                print(f"[force] {frame_index + 1}/{frames}", flush=True)
    finally:
        overlay_capture.release()
        raw_capture.release()
        output_writer.release()
        compare_writer.release()
        debug_writer.release()

    np.save(output_dir / "forced_object_front_mask_count.npy", forced_counts)
    report = {
        "schema_version": 1,
        "method": "projected HaCo contact vertices force SAM2 modal object in front",
        "depth_gate_used": False,
        "frames": frames,
        "frames_with_forced_object_front": int((forced_counts > 0).sum()),
        "forced_pixels_total": int(forced_counts.sum()),
        "contact_vertices_total": int(contact_counts.sum()),
        "contact_radius_px": args.contact_radius_px,
        "edge_sigma_px": args.edge_sigma_px,
        "force_scope": args.force_scope,
        "min_trigger_pixels": args.min_trigger_pixels,
        "frames_triggered_by_local_contact": int(
            (trigger_counts >= args.min_trigger_pixels).sum()
        ),
        "invariants": {
            "forced_pixels_are_inside_sam2_object": True,
            "forced_pixels_are_inside_rendered_robot": True,
            "forced_pixels_have_projected_haco_contact_support": (
                args.force_scope == "local-contact"
            ),
            "full_overlap_is_gated_by_projected_haco_contact": (
                args.force_scope == "full-object-overlap"
            ),
            "object_rgb_is_restored_from_raw_video": True,
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
