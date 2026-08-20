#!/usr/bin/env python3
"""Composite saved robot render layers after object/background restoration.

This deliberately avoids patching an already composited overlay.  The clean
plate is the base, the robot layer is added once, and contact-triggered object
occlusion is applied directly to the robot alpha before compositing.
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
    parser.add_argument(
        "--robot_thumb_mask", type=Path,
        help="optional XHand thumb mask; the thumb remains in front at contact",
    )
    parser.add_argument("--object_mask", type=Path, required=True)
    parser.add_argument(
        "--observed_object_mask", type=Path,
        help="trusted visible-object mask required by observed-contact-only policy",
    )
    parser.add_argument(
        "--visible_human_mask", type=Path,
        help=(
            "original-frame visible hand/arm mask required by visibility-aware; "
            "estimated robot pixels outside it can be placed behind restored object"
        ),
    )
    parser.add_argument("--visibility_dilation_px", type=int, default=8)
    parser.add_argument("--contact_dir", type=Path, required=True)
    parser.add_argument("--hawor_npz", type=Path, required=True)
    parser.add_argument("--old_overlay", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--contact_radius_px", type=int, default=30)
    parser.add_argument("--min_trigger_pixels", type=int, default=1)
    parser.add_argument("--robot_edge_sigma_px", type=float, default=0.6)
    parser.add_argument(
        "--robot_priority", action="store_true",
        help=(
            "make every non-occluded robot-mask pixel fully opaque; restored "
            "object pixels are never allowed to blend over the robot"
        ),
    )
    parser.add_argument(
        "--occlusion_policy",
        choices=(
            "full-amodal",
            "observed-contact-only",
            "amodal-contact-only",
            "visibility-aware",
        ),
        default="full-amodal",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_cap = cv2.VideoCapture(str(args.clean_plate))
    old_cap = (
        cv2.VideoCapture(str(args.old_overlay))
        if args.old_overlay is not None else None
    )
    if not clean_cap.isOpened() or (old_cap is not None and not old_cap.isOpened()):
        raise FileNotFoundError("could not open clean plate or old overlay")
    width = int(clean_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(clean_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(round(clean_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(clean_cap.get(cv2.CAP_PROP_FPS) or 30.0)
    robot_rgb = np.load(args.robot_rgb, mmap_mode="r")
    robot_mask = np.load(args.robot_mask, mmap_mode="r")
    robot_thumb_mask = (
        np.load(args.robot_thumb_mask, mmap_mode="r")
        if args.robot_thumb_mask is not None else None
    )
    object_mask = np.load(args.object_mask, mmap_mode="r")
    observed_object_mask = (
        np.load(args.observed_object_mask, mmap_mode="r")
        if args.observed_object_mask is not None else None
    )
    visible_human_mask = (
        np.load(args.visible_human_mask, mmap_mode="r")
        if args.visible_human_mask is not None else None
    )
    if args.occlusion_policy in ("observed-contact-only", "visibility-aware") and observed_object_mask is None:
        parser.error(f"{args.occlusion_policy} requires --observed_object_mask")
    if args.occlusion_policy == "visibility-aware" and visible_human_mask is None:
        parser.error("visibility-aware requires --visible_human_mask")
    contacts = sorted(args.contact_dir.glob("*.npz"))
    if not (
        len(robot_rgb) == len(robot_mask) == len(object_mask) == len(contacts) == frames
    ):
        raise ValueError("frame count mismatch")
    if observed_object_mask is not None and len(observed_object_mask) != frames:
        raise ValueError("observed object mask frame count mismatch")
    if robot_thumb_mask is not None and len(robot_thumb_mask) != frames:
        raise ValueError("robot thumb mask frame count mismatch")
    if visible_human_mask is not None and len(visible_human_mask) != frames:
        raise ValueError("visible human mask frame count mismatch")
    with np.load(args.hawor_npz) as hawor:
        focal = float(hawor["img_focal"])

    final_writer = open_writer(
        args.output_dir / "video_restored_first_robot_overlay.mp4v.mp4",
        fps,
        (width, height),
    )
    panel_w, panel_h, header = width // 2, height // 2, 64
    comparison_writer = open_writer(
        args.output_dir / "video_compare_patch_after_vs_restore_before.mp4v.mp4",
        fps,
        (panel_w * 2, panel_h + header),
    )
    pipeline_writer = open_writer(
        args.output_dir / "video_cleanplate_then_robot.mp4v.mp4",
        fps,
        (panel_w * 2, panel_h + header),
    )
    triggered = np.zeros(frames, dtype=bool)
    hidden_counts = np.zeros(frames, dtype=np.int64)
    invisible_restored_counts = np.zeros(frames, dtype=np.int64)

    try:
        for index, contact_path in enumerate(contacts):
            ok, clean = clean_cap.read()
            if not ok:
                raise RuntimeError(f"clean plate ended at frame {index}")
            old = None
            if old_cap is not None:
                ok_old, old = old_cap.read()
                if not ok_old:
                    raise RuntimeError(f"old overlay ended at frame {index}")
            obj = resize_mask(object_mask[index], width, height)
            observed_obj = (
                resize_mask(observed_object_mask[index], width, height)
                if observed_object_mask is not None else obj
            )
            visible_human = (
                resize_mask(visible_human_mask[index], width, height)
                if visible_human_mask is not None
                else np.ones((height, width), dtype=bool)
            )
            if args.visibility_dilation_px > 0:
                radius = args.visibility_dilation_px
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
                )
                visible_human = cv2.dilate(
                    visible_human.astype(np.uint8), kernel
                ).astype(bool)
            rmask = resize_mask(robot_mask[index], width, height)
            thumb = (
                resize_mask(robot_thumb_mask[index], width, height)
                if robot_thumb_mask is not None
                else np.zeros((height, width), dtype=bool)
            )
            rrgb = cv2.resize(
                np.asarray(robot_rgb[index]), (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
            support = np.zeros((height, width), dtype=np.uint8)
            with np.load(contact_path) as contact:
                valid_hand = bool(contact[f"{args.side}_valid"])
                vertices = np.asarray(
                    contact[f"{args.side}_contact_verts_3d"], dtype=np.float32
                )
            if valid_hand and len(vertices):
                uv, valid = project(vertices, focal, width, height)
                for point in uv[valid]:
                    cv2.circle(
                        support, tuple(np.rint(point).astype(int)),
                        args.contact_radius_px, 1, -1, cv2.LINE_8,
                    )
            trigger_object = (
                observed_obj
                if args.occlusion_policy == "observed-contact-only" else obj
            )
            trigger_count = int(
                np.sum(trigger_object & rmask & support.astype(bool))
            )
            triggered[index] = trigger_count >= args.min_trigger_pixels

            alpha = rmask.astype(np.float32)
            if args.robot_edge_sigma_px > 0:
                alpha = cv2.GaussianBlur(
                    alpha, (0, 0), args.robot_edge_sigma_px,
                    args.robot_edge_sigma_px,
                )
            alpha = np.clip(alpha, 0.0, 1.0)
            if triggered[index] or args.occlusion_policy == "visibility-aware":
                if args.occlusion_policy == "observed-contact-only":
                    # Inferred/restored pixels never hide the robot.  Only
                    # measured object RGB near projected HaCo contacts can put
                    # the corresponding robot pixels behind the object.
                    behind = observed_obj & support.astype(bool)
                elif args.occlusion_policy == "amodal-contact-only":
                    # Treat both measured and newly restored object pixels as
                    # a valid front surface, but only near HaCo contacts.
                    behind = obj & support.astype(bool)
                elif args.occlusion_policy == "visibility-aware":
                    restored_obj = obj & ~observed_obj
                    invisible_estimated_robot = rmask & ~visible_human
                    invisible_restored = (
                        restored_obj & invisible_estimated_robot
                    )
                    # Measured object at a HaCo contact retains the original
                    # contact-ordering rule.  In addition, any robot hand
                    # inferred where no human hand was visible goes behind the
                    # newly restored object surface.
                    behind = (
                        observed_obj & support.astype(bool) & triggered[index]
                    ) | invisible_restored
                    invisible_restored_counts[index] = int(
                        invisible_restored.sum()
                    )
                else:
                    behind = obj
                # The opposition thumb is the explicit front layer; only the
                # four rear fingers/hand surface are sent behind the object.
                behind &= ~thumb
                alpha[behind] = 0.0
            else:
                behind = np.zeros_like(obj)
            if args.robot_priority:
                # Hard ownership rule: the robot wins at every robot pixel
                # except trusted, observed object pixels explicitly selected
                # by the HaCo contact support.  This also removes clean-plate
                # bleed caused by alpha feathering along the robot boundary.
                allowed_behind = (
                    behind & observed_obj
                    if args.occlusion_policy == "observed-contact-only"
                    else behind
                )
                alpha[rmask & ~allowed_behind] = 1.0
                alpha[~rmask] = 0.0
                behind = allowed_behind
            hidden_counts[index] = int(np.sum(rmask & behind))
            final = np.clip(
                clean.astype(np.float32) * (1.0 - alpha[..., None])
                + rrgb.astype(np.float32) * alpha[..., None],
                0,
                255,
            ).astype(np.uint8)
            final_writer.write(final)

            clean_panel = cv2.resize(clean, (panel_w, panel_h))
            final_panel = cv2.resize(final, (panel_w, panel_h))
            pipeline = np.full(
                (panel_h + header, panel_w * 2, 3), 24, dtype=np.uint8
            )
            pipeline[header:, :panel_w] = clean_panel
            pipeline[header:, panel_w:] = final_panel
            cv2.putText(
                pipeline, "1 Restored clean plate", (18, 41),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2, cv2.LINE_AA,
            )
            cv2.putText(
                pipeline, "2 Render robot once", (panel_w + 18, 41),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 220, 80), 2, cv2.LINE_AA,
            )
            pipeline_writer.write(pipeline)

            comparison = np.full_like(pipeline, 24)
            if old is None:
                old = clean
            comparison[header:, :panel_w] = cv2.resize(old, (panel_w, panel_h))
            comparison[header:, panel_w:] = final_panel
            cv2.putText(
                comparison, "Patch after rendering", (18, 41),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (240, 240, 240), 2, cv2.LINE_AA,
            )
            cv2.putText(
                comparison, "Restore before rendering", (panel_w + 18, 41),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (80, 220, 80), 2, cv2.LINE_AA,
            )
            comparison_writer.write(comparison)
            if (index + 1) % 100 == 0:
                print(f"[composite] {index + 1}/{frames}", flush=True)
    finally:
        clean_cap.release()
        if old_cap is not None:
            old_cap.release()
        final_writer.release()
        comparison_writer.release()
        pipeline_writer.release()

    report = {
        "schema_version": 1,
        "method": "restore_object_first_then_single_robot_layer_composite",
        "frames": frames,
        "frames_with_haco_trigger": int(triggered.sum()),
        "robot_pixels_hidden_by_restored_object": int(hidden_counts.sum()),
        "robot_edge_sigma_px": args.robot_edge_sigma_px,
        "robot_priority": args.robot_priority,
        "thumb_preserved_in_front": robot_thumb_mask is not None,
        "visibility_dilation_px": args.visibility_dilation_px,
        "occlusion_policy": args.occlusion_policy,
        "robot_pixels_behind_restored_object_due_to_invisibility": int(
            invisible_restored_counts.sum()
        ),
        "ordering": [
            "restored_object_clean_plate",
            "saved_robot_rgb_and_alpha",
            "contact_triggered_object_occlusion_on_robot_alpha",
            "single_final_composite",
        ],
        "invariants": {
            "already_composited_overlay_not_used_as_render_base": True,
            "restored_object_exists_before_robot_composite": True,
            "object_occlusion_is_applied_to_robot_alpha": True,
            "restored_hidden_object_never_occludes_robot": (
                args.occlusion_policy == "observed-contact-only"
            ),
            "haco_contact_support_limits_observed_object_occlusion": (
                args.occlusion_policy == "observed-contact-only"
            ),
            "restored_object_overlap_with_visible_robot_is_zero": (
                args.robot_priority
                and args.occlusion_policy == "observed-contact-only"
            ),
            "restored_object_can_occlude_robot_near_haco_contact": (
                args.occlusion_policy == "amodal-contact-only"
            ),
            "invisible_estimated_robot_goes_behind_restored_object": (
                args.occlusion_policy == "visibility-aware"
            ),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
