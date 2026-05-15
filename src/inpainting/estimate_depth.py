"""Stage A — monocular depth estimation with Depth Anything V2.

Runs DA-V2 per-frame over <processed_demo>/video_L.mp4 and writes the raw
model output. DA-V2 emits *relative inverse depth* (disparity-like, higher =
closer) for the non-metric checkpoints (vits / vitb / vitl). The alignment
stage (`align_depth.py`) turns this into metric meters using HaWoR's hand
keypoint depths as per-frame anchors, so we don't bake in any scale here.

Outputs:
    <processed_demo>/depth_processor/depth_raw.npy           (T,H,W) float16
    <processed_demo>/depth_processor/depth_meta.npz          encoder, input_size

Usage:
    python estimate_depth.py --processed_demo /result/cam0_inpaint/cam0/0 \
        --encoder vitl --input_size 518
"""
import argparse
import sys
from pathlib import Path

import cv2
import mediapy as media
import numpy as np
import torch
from tqdm import tqdm

from _paths import (
    DEPTH_ANYTHING_CKPTS,
    DEPTH_ANYTHING_CONFIGS,
    ensure_depth_anything_importable,
)

ensure_depth_anything_importable()
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--encoder", choices=list(DEPTH_ANYTHING_CONFIGS), default="vitl")
    ap.add_argument("--input_size", type=int, default=518,
                    help="DA-V2 inference size (multiple of 14)")
    ap.add_argument("--input_video", type=str, default="video_L.mp4",
                    help="Filename under processed_demo to estimate depth on")
    ap.add_argument("--out_name", type=str, default="depth_raw.npy",
                    help="Output filename under depth_processor/")
    args = ap.parse_args()

    ckpt = Path(DEPTH_ANYTHING_CKPTS[args.encoder])
    if not ckpt.exists():
        sys.exit(
            f"Depth Anything V2 checkpoint missing: {ckpt}\n"
            f"Download with: mkdir -p {ckpt.parent} && \\\n"
            f"  wget -P {ckpt.parent} "
            f"https://huggingface.co/depth-anything/Depth-Anything-V2-Large/"
            f"resolve/main/depth_anything_v2_{args.encoder}.pth"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DEPTH_ANYTHING_CONFIGS[args.encoder]
    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    model = model.to(device).eval()
    print(f"[info] Depth Anything V2 ({args.encoder}) loaded on {device}")

    video_path = args.processed_demo / args.input_video
    frames = media.read_video(str(video_path))   # (T,H,W,3) uint8 RGB
    T, H, W, _ = frames.shape
    print(f"[info] T={T}, {W}x{H}, source={video_path}")

    out = np.zeros((T, H, W), dtype=np.float16)
    for t in tqdm(range(T), desc="DA-V2"):
        # DA-V2 expects BGR uint8 (it calls cv2 inside infer_image)
        bgr = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        with torch.no_grad():
            d = model.infer_image(bgr, args.input_size)   # (H,W) float32, disparity-like
        out[t] = d.astype(np.float16)

    out_dir = args.processed_demo / "depth_processor"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / args.out_name, out)
    meta_name = Path(args.out_name).stem + "_meta.npz"
    np.savez(out_dir / meta_name,
             encoder=args.encoder, input_size=args.input_size,
             source_video=str(video_path),
             # DA-V2 relative ckpts emit disparity-like values: higher = closer.
             # The alignment stage assumes disparity convention.
             convention="disparity")
    print(f"[ok] wrote {out_dir/args.out_name} (T,H,W)={out.shape}")


if __name__ == "__main__":
    main()
