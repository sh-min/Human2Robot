#!/usr/bin/env python3
"""Diffusion-inpaint only the object pixels used to occlude the robot hand.

The diffusion hole is the rendered robot/object intersection dilated by a
small radius and clipped to the inferred amodal object.  The generated RGB is
never used outside that support.  The final overlay restores only the exact
robot/object intersection, so unrelated source pixels remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return np.asarray(mask, dtype=bool)
    return cv2.resize(
        np.asarray(mask, dtype=np.uint8), (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def square_crop(mask: np.ndarray, margin: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, mask.shape[1], mask.shape[0]
    x0, x1 = int(xs.min()) - margin, int(xs.max()) + 1 + margin
    y0, y1 = int(ys.min()) - margin, int(ys.max()) + 1 + margin
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0, y0 = cx - side // 2, cy - side // 2
    x1, y1 = x0 + side, y0 + side
    height, width = mask.shape
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return max(0, x0), max(0, y0), x1, y1


def open_writer(path: Path, fps: float, size: tuple[int, int]):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer: {path}")
    return writer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_video", type=Path, required=True)
    parser.add_argument("--base_overlay", type=Path, required=True)
    parser.add_argument("--object_mask", type=Path, required=True)
    parser.add_argument("--robot_mask", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--model", default="stable-diffusion-v1-5/stable-diffusion-inpainting"
    )
    parser.add_argument("--radius_px", type=int, default=16)
    parser.add_argument("--crop_margin_px", type=int, default=96)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int)
    parser.add_argument(
        "--prompt",
        default=(
            "an intact photorealistic white snack package, preserve the exact "
            "package shape, printed label, red and navy diamond pattern, "
            "consistent lighting, no hand, no robot"
        ),
    )
    parser.add_argument(
        "--negative_prompt",
        default=(
            "hand, fingers, robot, transparent, translucent, hole, missing "
            "surface, distorted package, extra object, text artifacts"
        ),
    )
    args = parser.parse_args()
    if args.radius_px < 0 or args.crop_margin_px < 0:
        parser.error("radius and crop margin must be non-negative")
    if args.crop_size % 8:
        parser.error("crop size must be divisible by 8")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_cap = cv2.VideoCapture(str(args.source_video))
    overlay_cap = cv2.VideoCapture(str(args.base_overlay))
    if not source_cap.isOpened() or not overlay_cap.isOpened():
        raise FileNotFoundError("could not open source or overlay")
    width = int(source_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(round(source_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(source_cap.get(cv2.CAP_PROP_FPS) or 30.0)
    end_frame = frames - 1 if args.end_frame is None else args.end_frame
    if not 0 <= args.start_frame <= end_frame < frames:
        parser.error("invalid frame interval")
    objects = np.load(args.object_mask, mmap_mode="r")
    robots = np.load(args.robot_mask, mmap_mode="r")
    if len(objects) != frames or len(robots) != frames:
        raise ValueError("mask/video frame count mismatch")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_attention_slicing()
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"

    out_size = (width, height)
    patch_writer = open_writer(
        args.output_dir / "video_object_diffusion_patch.mp4v.mp4", fps, out_size
    )
    final_writer = open_writer(
        args.output_dir / "video_overlay_diffusion_object_front.mp4v.mp4",
        fps,
        out_size,
    )
    header = 64
    panel_w, panel_h = width // 2, height // 2
    compare_writer = open_writer(
        args.output_dir / "video_compare_temporal_vs_diffusion.mp4v.mp4",
        fps,
        (panel_w * 2, panel_h + header),
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (args.radius_px * 2 + 1,) * 2
    )
    core_counts = []
    diffusion_counts = []

    try:
        for frame_index in range(frames):
            ok_source, source = source_cap.read()
            ok_overlay, overlay = overlay_cap.read()
            if not ok_source or not ok_overlay:
                raise RuntimeError(f"video read failed at frame {frame_index}")
            if overlay.shape[:2] != (height, width):
                overlay = cv2.resize(overlay, out_size, interpolation=cv2.INTER_AREA)
            object_mask = resize_mask(objects[frame_index], width, height)
            robot_mask = resize_mask(robots[frame_index], width, height)
            core = object_mask & robot_mask
            expanded = cv2.dilate(
                core.astype(np.uint8), kernel, iterations=1
            ).astype(bool)
            expanded &= object_mask
            core_counts.append(int(core.sum()))
            diffusion_counts.append(int(expanded.sum()))
            diffused = source.copy()

            if args.start_frame <= frame_index <= end_frame and np.any(core):
                x0, y0, x1, y1 = square_crop(expanded, args.crop_margin_px)
                crop = source[y0:y1, x0:x1]
                mask_crop = expanded[y0:y1, x0:x1].astype(np.uint8) * 255
                image_pil = Image.fromarray(
                    cv2.cvtColor(
                        cv2.resize(crop, (args.crop_size, args.crop_size)),
                        cv2.COLOR_BGR2RGB,
                    )
                )
                mask_pil = Image.fromarray(
                    cv2.resize(
                        mask_crop, (args.crop_size, args.crop_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                )
                generator = torch.Generator(device=generator_device).manual_seed(
                    args.seed
                )
                generated = pipe(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    image=image_pil,
                    mask_image=mask_pil,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    height=args.crop_size,
                    width=args.crop_size,
                ).images[0]
                generated = cv2.cvtColor(np.asarray(generated), cv2.COLOR_RGB2BGR)
                generated = cv2.resize(
                    generated, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LANCZOS4
                )
                region = diffused[y0:y1, x0:x1]
                local_mask = expanded[y0:y1, x0:x1]
                region[local_mask] = generated[local_mask]
                diffused[y0:y1, x0:x1] = region

            final = overlay.copy()
            final[core] = diffused[core]
            patch_writer.write(diffused)
            final_writer.write(final)
            left = cv2.resize(source, (panel_w, panel_h))
            right = cv2.resize(diffused, (panel_w, panel_h))
            compare = np.full(
                (panel_h + header, panel_w * 2, 3), 24, dtype=np.uint8
            )
            compare[header:, :panel_w] = left
            compare[header:, panel_w:] = right
            cv2.putText(
                compare, "Temporal completion", (18, 41),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA,
            )
            cv2.putText(
                compare, f"Local diffusion (+{args.radius_px}px)",
                (panel_w + 18, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (80, 220, 80), 2, cv2.LINE_AA,
            )
            compare_writer.write(compare)
            if (frame_index + 1) % 10 == 0:
                print(f"[diffusion] {frame_index + 1}/{frames}", flush=True)
    finally:
        source_cap.release()
        overlay_cap.release()
        patch_writer.release()
        final_writer.release()
        compare_writer.release()

    report = {
        "schema_version": 1,
        "method": "localized_stable_diffusion_inpaint_on_robot_object_overlap",
        "model": args.model,
        "frames": frames,
        "diffused_interval": [args.start_frame, end_frame],
        "radius_px": args.radius_px,
        "crop_margin_px": args.crop_margin_px,
        "crop_size": args.crop_size,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed_reused_for_temporal_consistency": args.seed,
        "robot_object_overlap_pixels": int(sum(core_counts)),
        "diffusion_support_pixels": int(sum(diffusion_counts)),
        "invariants": {
            "diffusion_support_clipped_to_amodal_object": True,
            "final_overlay_changes_only_robot_object_intersection": True,
            "pixels_outside_diffusion_support_unchanged": True,
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
