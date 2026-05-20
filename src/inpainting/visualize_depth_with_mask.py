"""Side-by-side viz: raw RGB | colorized scene depth, both with the hand mask
outlined in red. Useful for eyeballing how depth varies inside the hand
silhouette vs. just outside it (i.e. where the cube would sit).

Output:
    <processed_demo>/depth_processor/depth_with_mask.mp4

Usage:
    python visualize_depth_with_mask.py --processed_demo /result/cam0_inpaint/cam0/0
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
import mediapy as media
import numpy as np


OUTLINE_RGB = np.array([255, 0, 0], dtype=np.uint8)   # red


def _colorize(depth: np.ndarray, vmin: float, vmax: float, cmap) -> np.ndarray:
    z = np.clip(depth, vmin, vmax)
    norm = (z - vmin) / max(vmax - vmin, 1e-6)
    return (cmap(norm)[..., :3] * 255).astype(np.uint8)


def _outline(mask: np.ndarray, thickness: int) -> np.ndarray:
    """(H,W) bool → (H,W) bool: pixels on the boundary of `mask`, `thickness` wide."""
    m = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dil = cv2.dilate(m, kernel, iterations=thickness)
    ero = cv2.erode(m,  kernel, iterations=thickness)
    return (dil != ero)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--thickness", type=int, default=2,
                    help="hand-mask outline thickness (px)")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    args = ap.parse_args()

    pd = args.processed_demo
    rgb     = media.read_video(str(pd / "video_L.mp4"))            # (T,H,W,3) uint8
    depth   = np.load(pd / "depth_processor" / "depth_aligned.npy").astype(np.float32)
    m_hand  = np.load(pd / "segmentation_processor" / "masks_arm.npy").astype(bool)

    T = min(rgb.shape[0], depth.shape[0], m_hand.shape[0])
    rgb, depth, m_hand = rgb[:T], depth[:T], m_hand[:T]
    H, W = rgb.shape[1], rgb.shape[2]

    vmin = float(np.percentile(depth, 5))  if args.vmin is None else args.vmin
    vmax = float(np.percentile(depth, 95)) if args.vmax is None else args.vmax
    print(f"[info] T={T}, depth range [{vmin:.2f}, {vmax:.2f}] m, cmap={args.cmap}")
    cmap = matplotlib.colormaps[args.cmap]

    gap = 10
    out = np.zeros((T, H, 2 * W + gap, 3), dtype=np.uint8)
    out[:, :, W:W + gap] = 255

    for t in range(T):
        d_col = _colorize(depth[t], vmin, vmax, cmap)
        outline = _outline(m_hand[t], args.thickness)
        left  = rgb[t].copy()
        right = d_col
        left[outline]  = OUTLINE_RGB
        right[outline] = OUTLINE_RGB
        out[t, :, :W]            = left
        out[t, :, W + gap:]      = right
        if (t + 1) % 100 == 0:
            print(f"  {t+1}/{T}")

    dst = pd / "depth_processor" / "depth_with_mask.mp4"
    media.write_video(str(dst), out, fps=args.fps, codec="libx264")
    print(f"[ok] wrote {dst}  (RGB | depth, both with hand outline)")


if __name__ == "__main__":
    main()
