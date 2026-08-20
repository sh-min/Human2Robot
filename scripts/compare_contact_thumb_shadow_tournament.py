#!/usr/bin/env python3
"""Build a four-way D/thumbnail-priority/contact-shadow tournament.

The accepted D contact compositor remains the immutable baseline.  Two
orthogonal post-composite cues are toggled independently:

* thumb priority: restore only D-occluded semantic-thumb pixels over the
  manipulated object's modal mask;
* contact shadow: use Human2Robot's geometry-grounded shadow implementation,
  evaluated on the compact robot RGB-D buffers and upsampled to the video.

This produces two first-round matches without changing the retargeted pose or
the D occlusion mask.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "inpainting"))
from contact_shadow import contact_shadow_alpha, fit_support_plane  # noqa: E402


def writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not value.isOpened():
        raise RuntimeError(f"could not open video writer: {path}")
    return value


def panel(frame: np.ndarray, label: str, size: tuple[int, int]) -> np.ndarray:
    value = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    cv2.putText(
        value, label, (16, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.82,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-video", type=Path, required=True)
    parser.add_argument("--scene-depth", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--object-mask", type=Path, required=True)
    parser.add_argument("--occluded-mask", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thumb-alpha", type=float, default=1.0)
    parser.add_argument("--shadow-opacity", type=float, default=0.6)
    parser.add_argument(
        "--shadow-blur-fullres", type=float, default=6.0,
        help="Desired full-resolution Gaussian sigma; scaled for compact RGB-D",
    )
    args = parser.parse_args()
    if not 0.0 <= args.thumb_alpha <= 1.0:
        parser.error("--thumb-alpha must be in [0, 1]")
    if not 0.0 <= args.shadow_opacity <= 1.0:
        parser.error("--shadow-opacity must be in [0, 1]")

    capture = cv2.VideoCapture(str(args.base_video))
    if not capture.isOpened():
        raise FileNotFoundError(args.base_video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)

    overlay = args.overlay_dir
    robot_rgb = np.load(overlay / "robot_rgb.npy", mmap_mode="r")
    robot_depth = np.load(overlay / "robot_depth.npy", mmap_mode="r")
    robot_mask = np.load(overlay / "robot_mask.npy", mmap_mode="r")
    finger_labels = np.load(
        overlay / "robot_finger_labels.npy", mmap_mode="r"
    )
    scene_depth = np.load(args.scene_depth, mmap_mode="r")
    object_mask = np.load(args.object_mask, mmap_mode="r")
    occluded_mask = np.load(args.occluded_mask, mmap_mode="r")
    arrays = {
        "robot_rgb": robot_rgb,
        "robot_depth": robot_depth,
        "robot_mask": robot_mask,
        "finger_labels": finger_labels,
        "scene_depth": scene_depth,
        "object_mask": object_mask,
        "occluded_mask": occluded_mask,
    }
    for name, value in arrays.items():
        if len(value) != frames:
            raise ValueError(f"{name} frame mismatch: {len(value)} != {frames}")

    low_h, low_w = robot_depth.shape[1:3]
    with np.load(args.hawor_npz) as hawor:
        focal_full = float(hawor["img_focal"])
    focal_low = focal_full * low_w / width
    blur_low = args.shadow_blur_fullres * low_w / width

    out = args.output_dir.resolve()
    outputs = {
        "thumb": out / "D_plus_thumb_priority.mp4",
        "shadow": out / "D_plus_github_contact_shadow.mp4",
        "both": out / "D_plus_thumb_and_shadow.mp4",
        "match1": out / "ROUND1_match1_D_vs_thumb.mp4",
        "match2": out / "ROUND1_match2_shadow_vs_both.mp4",
        "grid": out / "ROUND1_all_four_2x2.mp4",
        "shadow_debug": out / "shadow_alpha_debug.mp4",
    }
    full_writers = {
        key: writer(outputs[key], fps, (width, height))
        for key in ("thumb", "shadow", "both")
    }
    match_size = (width // 2, height // 2)
    match_writers = {
        key: writer(outputs[key], fps, (width, height // 2))
        for key in ("match1", "match2")
    }
    grid_writer = writer(outputs["grid"], fps, (width, height))
    shadow_debug_writer = writer(
        outputs["shadow_debug"], fps, (low_w, low_h)
    )

    plane = None
    thumb_counts = np.zeros(frames, dtype=np.int64)
    shadow_counts = np.zeros(frames, dtype=np.int64)
    shadow_alpha_sums = np.zeros(frames, dtype=np.float64)
    try:
        for index in range(frames):
            ok, base = capture.read()
            if not ok:
                raise RuntimeError(f"base video read failed at frame {index}")

            depth_low = cv2.resize(
                np.asarray(scene_depth[index], dtype=np.float32),
                (low_w, low_h), interpolation=cv2.INTER_NEAREST,
            )
            rmask_low = np.asarray(robot_mask[index], dtype=bool)
            rdepth_low = np.asarray(robot_depth[index], dtype=np.float32)
            plane = fit_support_plane(
                depth_low, rmask_low, focal_low, low_w / 2.0, low_h / 2.0,
                prev=plane, rng_seed=index,
            )
            shadow_low = contact_shadow_alpha(
                depth_low, rdepth_low, rmask_low, plane,
                focal_low, low_w / 2.0, low_h / 2.0,
                opacity=args.shadow_opacity, blur=blur_low,
            )
            shadow = cv2.resize(
                shadow_low, (width, height), interpolation=cv2.INTER_LINEAR
            )
            shadow_counts[index] = int(np.count_nonzero(shadow > 1.0e-3))
            shadow_alpha_sums[index] = float(shadow.sum())
            shadow_frame = np.clip(
                base.astype(np.float32) * (1.0 - shadow[..., None]), 0, 255
            ).astype(np.uint8)

            labels = cv2.resize(
                np.asarray(finger_labels[index], dtype=np.uint8),
                (width, height), interpolation=cv2.INTER_NEAREST,
            )
            obj = np.asarray(object_mask[index], dtype=bool)
            hidden = np.asarray(occluded_mask[index], dtype=bool)
            thumb_front = (labels == 1) & obj & hidden
            thumb_counts[index] = int(thumb_front.sum())
            thumb_bgr = cv2.resize(
                np.asarray(robot_rgb[index], dtype=np.uint8)[..., ::-1],
                (width, height), interpolation=cv2.INTER_LINEAR,
            )

            thumb_frame = base.copy()
            both_frame = shadow_frame.copy()
            if thumb_front.any() and args.thumb_alpha > 0:
                alpha = args.thumb_alpha
                for value in (thumb_frame, both_frame):
                    value[thumb_front] = np.clip(
                        value[thumb_front].astype(np.float32) * (1.0 - alpha)
                        + thumb_bgr[thumb_front].astype(np.float32) * alpha,
                        0, 255,
                    ).astype(np.uint8)

            full_writers["thumb"].write(thumb_frame)
            full_writers["shadow"].write(shadow_frame)
            full_writers["both"].write(both_frame)
            match_writers["match1"].write(np.hstack((
                panel(base, "D BASELINE", match_size),
                panel(thumb_frame, "D + THUMB PRIORITY", match_size),
            )))
            match_writers["match2"].write(np.hstack((
                panel(shadow_frame, "D + GITHUB SHADOW", match_size),
                panel(both_frame, "D + THUMB + SHADOW", match_size),
            )))
            grid_writer.write(np.vstack((
                np.hstack((
                    panel(base, "A  D BASELINE", match_size),
                    panel(thumb_frame, "B  D + THUMB PRIORITY", match_size),
                )),
                np.hstack((
                    panel(shadow_frame, "C  D + GITHUB SHADOW", match_size),
                    panel(both_frame, "D  D + THUMB + SHADOW", match_size),
                )),
            )))
            debug = np.clip(shadow_low * 255.0 / max(args.shadow_opacity, 1e-6), 0, 255)
            shadow_debug_writer.write(cv2.applyColorMap(
                debug.astype(np.uint8), cv2.COLORMAP_INFERNO
            ))
            if (index + 1) % 100 == 0:
                print(f"[tournament] {index + 1}/{frames}", flush=True)
    finally:
        capture.release()
        for value in full_writers.values():
            value.release()
        for value in match_writers.values():
            value.release()
        grid_writer.release()
        shadow_debug_writer.release()

    report = {
        "schema_version": 1,
        "frames": frames,
        "fps": fps,
        "baseline": str(args.base_video.resolve()),
        "github_source": {
            "repository": "https://github.com/sh-min/Human2Robot",
            "contact_shadow": "src/inpainting/contact_shadow.py",
            "fetched_commit": "97ce2bdd15e35c207bd637846c9a56499f3b8080",
        },
        "thumb_priority_source": "local extension of force_sam2_object_at_haco_contact.py",
        "thumb_priority": {
            "policy": "restore only D-occluded semantic thumb pixels inside modal object",
            "alpha": args.thumb_alpha,
            "pixels": int(thumb_counts.sum()),
            "frames": int(np.count_nonzero(thumb_counts)),
        },
        "contact_shadow": {
            "policy": "Human2Robot support-plane projected robot shadow",
            "opacity": args.shadow_opacity,
            "full_resolution_blur_sigma_px": args.shadow_blur_fullres,
            "compact_blur_sigma_px": blur_low,
            "pixels_over_1e-3": int(shadow_counts.sum()),
            "frames": int(np.count_nonzero(shadow_counts)),
            "alpha_sum": float(shadow_alpha_sums.sum()),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
