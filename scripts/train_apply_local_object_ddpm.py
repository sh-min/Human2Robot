#!/usr/bin/env python3
"""Train and apply a lightweight object-conditioned diffusion inpainter.

Training uses normalized crops from the current clip.  The clean temporal
object video is the target, while the rendered robot/object intersection plus
a configurable radius is hidden from the conditioning image.  At inference,
generated RGB is accepted only inside that local support, and the final robot
overlay is changed only at the exact robot/object intersection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return np.asarray(mask, dtype=bool)
    return cv2.resize(
        np.asarray(mask, dtype=np.uint8), (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def square_box(mask: np.ndarray, margin: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    if not len(xs):
        return 0, 0, width, height
    x0, x1 = int(xs.min()) - margin, int(xs.max()) + 1 + margin
    y0, y1 = int(ys.min()) - margin, int(ys.max()) + 1 + margin
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0, y0 = cx - side // 2, cy - side // 2
    x1, y1 = x0 + side, y0 + side
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


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames, fps


def make_model(size: int) -> UNet2DModel:
    return UNet2DModel(
        sample_size=size,
        in_channels=7,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(32, 64, 128, 128),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"
        ),
        norm_num_groups=8,
    )


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
    parser.add_argument("--radius_px", type=int, default=16)
    parser.add_argument("--crop_margin_px", type=int, default=48)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--train_steps", type=int, default=1600)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--inference_steps", type=int, default=30)
    parser.add_argument(
        "--strength", type=float, default=0.35,
        help="fraction of the DDIM trajectory used for local refinement",
    )
    parser.add_argument(
        "--diffusion_weight", type=float, default=0.35,
        help="generated RGB weight inside the local support",
    )
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    if args.crop_size % 16:
        parser.error("crop size must be divisible by 16")
    if not 0.0 < args.strength <= 1.0:
        parser.error("strength must be in (0, 1]")
    if not 0.0 < args.diffusion_weight <= 1.0:
        parser.error("diffusion weight must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    source_frames, fps = read_video(args.source_video)
    overlay_frames, overlay_fps = read_video(args.base_overlay)
    if len(source_frames) != len(overlay_frames):
        raise ValueError("source/overlay frame count mismatch")
    if abs(fps - overlay_fps) > 0.1:
        raise ValueError("source/overlay FPS mismatch")
    frame_count = len(source_frames)
    height, width = source_frames[0].shape[:2]
    objects = np.load(args.object_mask, mmap_mode="r")
    robots = np.load(args.robot_mask, mmap_mode="r")
    if len(objects) != frame_count or len(robots) != frame_count:
        raise ValueError("mask/video frame count mismatch")
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * args.radius_px + 1,) * 2
    )

    crops = np.empty(
        (frame_count, args.crop_size, args.crop_size, 3), dtype=np.uint8
    )
    holes = np.empty(
        (frame_count, args.crop_size, args.crop_size), dtype=np.uint8
    )
    boxes: list[tuple[int, int, int, int]] = []
    cores: list[np.ndarray] = []
    expanded_masks: list[np.ndarray] = []
    for index, frame in enumerate(source_frames):
        object_mask = resize_mask(objects[index], width, height)
        robot_mask = resize_mask(robots[index], width, height)
        core = object_mask & robot_mask
        expanded = cv2.dilate(
            core.astype(np.uint8), kernel, iterations=1
        ).astype(bool)
        expanded &= object_mask
        box = square_box(object_mask, args.crop_margin_px)
        x0, y0, x1, y1 = box
        crops[index] = cv2.resize(
            frame[y0:y1, x0:x1],
            (args.crop_size, args.crop_size),
            interpolation=cv2.INTER_AREA,
        )
        holes[index] = cv2.resize(
            expanded[y0:y1, x0:x1].astype(np.uint8),
            (args.crop_size, args.crop_size),
            interpolation=cv2.INTER_NEAREST,
        )
        boxes.append(box)
        cores.append(core)
        expanded_masks.append(expanded)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args.crop_size).to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    crop_tensor = torch.from_numpy(crops).permute(0, 3, 1, 2).float() / 127.5 - 1
    hole_tensor = torch.from_numpy(holes[:, None]).float()
    losses = []
    model.train()
    for step in range(args.train_steps):
        indices = torch.randint(0, frame_count, (args.batch_size,))
        clean = crop_tensor[indices].to(device, non_blocking=True)
        hole = hole_tensor[indices].to(device, non_blocking=True)
        # Randomly vary the learned occlusion radius without changing object RGB.
        if step % 3 == 1:
            hole = F.max_pool2d(hole, kernel_size=5, stride=1, padding=2)
        elif step % 3 == 2:
            hole = F.max_pool2d(hole, kernel_size=9, stride=1, padding=4)
        known = clean * (1.0 - hole) - hole
        noise = torch.randn_like(clean)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (args.batch_size,), device=device, dtype=torch.long,
        )
        noisy = noise_scheduler.add_noise(clean, noise, timesteps)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            prediction = model(
                torch.cat((noisy, known, hole), dim=1), timesteps
            ).sample
            weight = 0.25 + 1.75 * hole
            loss = ((prediction - noise).square() * weight).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 100 == 0:
            print(
                f"[train] {step + 1}/{args.train_steps} "
                f"loss={np.mean(losses[-100:]):.5f}", flush=True,
            )

    checkpoint = args.output_dir / "object_local_ddpm.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "crop_size": args.crop_size,
            "train_steps": args.train_steps,
            "seed": args.seed,
        },
        checkpoint,
    )

    ddim = DDIMScheduler.from_config(noise_scheduler.config)
    ddim.set_timesteps(args.inference_steps, device=device)
    active_step_count = max(1, int(round(args.inference_steps * args.strength)))
    active_timesteps = ddim.timesteps[-active_step_count:]
    fixed_noise = torch.randn(
        (1, 3, args.crop_size, args.crop_size),
        generator=torch.Generator(device=device).manual_seed(args.seed),
        device=device,
    )
    patch_writer = open_writer(
        args.output_dir / "video_object_local_ddpm.mp4v.mp4", fps, (width, height)
    )
    final_writer = open_writer(
        args.output_dir / "video_overlay_local_ddpm_object_front.mp4v.mp4",
        fps,
        (width, height),
    )
    header = 64
    panel_w, panel_h = width // 2, height // 2
    compare_writer = open_writer(
        args.output_dir / "video_compare_temporal_vs_local_ddpm.mp4v.mp4",
        fps,
        (panel_w * 2, panel_h + header),
    )
    model.eval()
    generated_pixel_count = 0
    with torch.no_grad():
        for index in range(frame_count):
            clean = crop_tensor[index:index + 1].to(device)
            hole = hole_tensor[index:index + 1].to(device)
            known = clean * (1.0 - hole) - hole
            first_timestep = active_timesteps[0].reshape(1)
            sample = ddim.add_noise(clean, fixed_noise, first_timestep)
            for step_index, timestep in enumerate(active_timesteps):
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    prediction = model(
                        torch.cat((sample, known, hole), dim=1), timestep
                    ).sample
                sample = ddim.step(prediction, timestep, sample).prev_sample
                if step_index + 1 < len(active_timesteps):
                    next_timestep = active_timesteps[step_index + 1].reshape(1)
                    known_noisy = ddim.add_noise(clean, fixed_noise, next_timestep)
                    sample = sample * hole + known_noisy * (1.0 - hole)
                else:
                    sample = sample * hole + clean * (1.0 - hole)
            generated = (
                sample.clamp(-1, 1).add(1).mul(127.5)[0]
                .permute(1, 2, 0).float().cpu().numpy().astype(np.uint8)
            )
            x0, y0, x1, y1 = boxes[index]
            generated = cv2.resize(
                generated, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LANCZOS4
            )
            patch = source_frames[index].copy()
            local = patch[y0:y1, x0:x1]
            local_mask = expanded_masks[index][y0:y1, x0:x1]
            blended = np.clip(
                args.diffusion_weight * generated.astype(np.float32)
                + (1.0 - args.diffusion_weight) * local.astype(np.float32),
                0,
                255,
            ).astype(np.uint8)
            local[local_mask] = blended[local_mask]
            patch[y0:y1, x0:x1] = local
            final = overlay_frames[index].copy()
            final[cores[index]] = patch[cores[index]]
            generated_pixel_count += int(expanded_masks[index].sum())
            patch_writer.write(patch)
            final_writer.write(final)
            comparison = np.full(
                (panel_h + header, panel_w * 2, 3), 24, dtype=np.uint8
            )
            comparison[header:, :panel_w] = cv2.resize(
                source_frames[index], (panel_w, panel_h)
            )
            comparison[header:, panel_w:] = cv2.resize(
                patch, (panel_w, panel_h)
            )
            cv2.putText(
                comparison, "Temporal completion", (18, 41),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA,
            )
            cv2.putText(
                comparison, f"Object DDPM (+{args.radius_px}px)",
                (panel_w + 18, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (80, 220, 80), 2, cv2.LINE_AA,
            )
            compare_writer.write(comparison)
            if (index + 1) % 25 == 0:
                print(f"[infer] {index + 1}/{frame_count}", flush=True)
    patch_writer.release()
    final_writer.release()
    compare_writer.release()

    report = {
        "schema_version": 1,
        "method": "clip_specific_conditional_ddpm_local_object_inpainting",
        "pretrained": False,
        "training_frames": frame_count,
        "train_steps": args.train_steps,
        "inference_steps": args.inference_steps,
        "active_inference_steps": active_step_count,
        "strength": args.strength,
        "diffusion_weight": args.diffusion_weight,
        "crop_size": args.crop_size,
        "radius_px": args.radius_px,
        "generated_pixels": generated_pixel_count,
        "final_training_loss_mean_last_100": float(np.mean(losses[-100:])),
        "conditioning": "masked object crop plus robot/object overlap mask",
        "invariants": {
            "generated_pixels_clipped_to_amodal_object": True,
            "final_overlay_changes_only_robot_object_intersection": True,
            "source_pixels_outside_local_radius_unchanged": True,
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
