"""Estimate positive metric scene depth with Depth Anything V2 Indoor Metric.

The official Hypersim metric model predicts depth in metres.  A monocular
metric model can still have a sequence-level scale bias, so each frame is
anchored with the median ratio between HaWoR camera-space joint Z and the
predicted depth at the corresponding 2D joints.  Only this positive scale is
adjusted; no affine offset or sign-changing slope is fitted.

Outputs:
    <processed_demo>/depth_processor/depth_metric_raw.npy
    <processed_demo>/depth_processor/depth_aligned_metric.npy
    <processed_demo>/depth_processor/depth_metric_params.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mediapy as media
import numpy as np
import torch
from tqdm import tqdm

from _paths import DEPTH_ANYTHING_CONFIGS, DEPTH_ANYTHING_CKPT_DIR, REPO_ROOT

METRIC_SOURCE = (
    Path(REPO_ROOT) / "third_party" / "Depth-Anything-V2" / "metric_depth"
)
if str(METRIC_SOURCE) not in sys.path:
    sys.path.insert(0, str(METRIC_SOURCE))

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402


def _sample_bilinear(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    height, width = image.shape
    u = np.clip(uv[:, 0], 0, width - 1.001)
    v = np.clip(uv[:, 1], 0, height - 1.001)
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = u0 + 1
    v1 = v0 + 1
    du = u - u0
    dv = v - v0
    return (
        (1 - du) * (1 - dv) * image[v0, u0]
        + du * (1 - dv) * image[v0, u1]
        + (1 - du) * dv * image[v1, u0]
        + du * dv * image[v1, u1]
    )


def _temporal_scale(raw_scale: np.ndarray, window: int) -> np.ndarray:
    scale = np.asarray(raw_scale, dtype=np.float32).copy()
    valid = np.isfinite(scale) & (scale > 0)
    if not valid.any():
        raise RuntimeError("no valid HaWoR metric-depth scale anchors")
    indices = np.arange(len(scale))
    scale[~valid] = np.interp(indices[~valid], indices[valid], scale[valid])
    radius = max(0, window // 2)
    smoothed = np.array(
        [
            np.median(scale[max(0, t - radius):min(len(scale), t + radius + 1)])
            for t in range(len(scale))
        ],
        dtype=np.float32,
    )
    global_median = float(np.median(smoothed))
    return np.clip(smoothed, 0.5 * global_median, 2.0 * global_median)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_demo", type=Path, required=True)
    parser.add_argument("--encoder", choices=["vits", "vitb"], default="vits")
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--max_depth", type=float, default=20.0)
    parser.add_argument("--scale_window", type=int, default=7)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    processed = args.processed_demo.resolve()
    checkpoint = args.checkpoint or (
        Path(DEPTH_ANYTHING_CKPT_DIR)
        / f"depth_anything_v2_metric_hypersim_{args.encoder}.pth"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnythingV2(
        **DEPTH_ANYTHING_CONFIGS[args.encoder],
        max_depth=args.max_depth,
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device).eval()
    print(f"[info] metric Depth Anything V2 ({args.encoder}) on {device}")

    video_path = processed / "video_L.mp4"
    frames = media.read_video(str(video_path))
    frame_count, height, width = frames.shape[:3]
    hands = {}
    for side in ("left", "right"):
        path = processed / "hand_processor" / f"hand_data_{side}.npz"
        if path.is_file():
            hands[side] = np.load(path)
    if not hands:
        raise FileNotFoundError("hand_processor/hand_data_{left,right}.npz")

    out_dir = processed / "depth_processor"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "depth_metric_raw.npy"
    raw = np.lib.format.open_memmap(
        raw_path,
        mode="w+",
        dtype=np.float16,
        shape=(frame_count, height, width),
    )
    raw_scale = np.full(frame_count, np.nan, dtype=np.float32)
    try:
        for frame_index in tqdm(range(frame_count), desc="DA-V2 metric"):
            bgr = cv2.cvtColor(frames[frame_index], cv2.COLOR_RGB2BGR)
            with torch.inference_mode():
                prediction = model.infer_image(bgr, args.input_size).astype(
                    np.float32
                )
            raw[frame_index] = prediction.astype(np.float16)
            ratios = []
            for data in hands.values():
                if not data["hand_detected"][frame_index]:
                    continue
                uv = np.asarray(data["kpts_2d"][frame_index], dtype=np.float32)
                metric_z = np.asarray(
                    data["kpts_3d"][frame_index, :, 2],
                    dtype=np.float32,
                )
                predicted_z = _sample_bilinear(prediction, uv)
                valid = (
                    np.isfinite(metric_z)
                    & np.isfinite(predicted_z)
                    & (metric_z > 0.05)
                    & (metric_z < 10.0)
                    & (predicted_z > 0.05)
                    & (predicted_z < args.max_depth)
                )
                ratios.extend((metric_z[valid] / predicted_z[valid]).tolist())
            if ratios:
                raw_scale[frame_index] = float(np.median(ratios))
    finally:
        for data in hands.values():
            data.close()
        raw.flush()

    scale = _temporal_scale(raw_scale, args.scale_window)
    aligned_path = out_dir / "depth_aligned_metric.npy"
    aligned = np.lib.format.open_memmap(
        aligned_path,
        mode="w+",
        dtype=np.float16,
        shape=raw.shape,
    )
    for frame_index in tqdm(range(frame_count), desc="metric scale"):
        aligned[frame_index] = np.clip(
            np.asarray(raw[frame_index], dtype=np.float32) * scale[frame_index],
            0.05,
            10.0,
        ).astype(np.float16)
    aligned.flush()
    np.savez(
        out_dir / "depth_metric_params.npz",
        raw_scale=raw_scale,
        scale=scale,
        valid_frames=np.isfinite(raw_scale).astype(np.uint8),
        encoder=args.encoder,
        max_depth=args.max_depth,
        checkpoint=str(checkpoint.resolve()),
    )
    print(
        f"[ok] {aligned_path}: frames={frame_count}, "
        f"scale p05/median/p95="
        f"{np.percentile(scale, [5, 50, 95]).round(3).tolist()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
