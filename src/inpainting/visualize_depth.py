"""Side-by-side depth visualization: scene depth | robot depth.

Both `depth_aligned.npy` (scene, meters) and `robot_depth.npy` (robot z-buffer,
meters) are in MANO cam frame, so they share the same scale and can be
colorized with a shared per-frame range. Non-robot pixels in the right panel
are rendered black.

Output:
    <processed_demo>/depth_processor/depth_compare.mp4

Usage:
    python visualize_depth.py --processed_demo /result/cam0_inpaint/cam0/0
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.cm
import mediapy as media
import numpy as np


def _colorize(depth: np.ndarray, mask: np.ndarray, vmin: float, vmax: float,
              cmap) -> np.ndarray:
    """(H,W) float depth + (H,W) bool mask → (H,W,3) uint8 RGB. mask=False → black."""
    z = np.clip(depth, vmin, vmax)
    norm = (z - vmin) / max(vmax - vmin, 1e-6)
    rgba = cmap(norm)              # (H,W,4) float in [0,1]
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    rgb[~mask] = 0
    return rgb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--cmap", default="turbo",
                    help="matplotlib colormap (turbo, plasma, viridis, magma, …)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--vmin", type=float, default=None,
                    help="depth min (m). Default: 5th percentile of scene depth across all frames")
    ap.add_argument("--vmax", type=float, default=None,
                    help="depth max (m). Default: 95th percentile of scene depth across all frames")
    args = ap.parse_args()

    pd = args.processed_demo
    scene = np.load(pd / "depth_processor" / "depth_aligned.npy").astype(np.float32)
    robot_z = np.load(pd / "overlay_processor" / "robot_depth.npy").astype(np.float32)
    robot_m = np.load(pd / "overlay_processor" / "robot_mask.npy").astype(bool)

    T = min(scene.shape[0], robot_z.shape[0], robot_m.shape[0])
    scene = scene[:T]; robot_z = robot_z[:T]; robot_m = robot_m[:T]

    vmin = float(np.percentile(scene, 5)) if args.vmin is None else args.vmin
    vmax = float(np.percentile(scene, 95)) if args.vmax is None else args.vmax
    print(f"[info] T={T}, depth range used: [{vmin:.2f}, {vmax:.2f}] m, cmap={args.cmap}")
    cmap = matplotlib.cm.get_cmap(args.cmap)

    H, W = scene.shape[1], scene.shape[2]
    gap = 10
    out = np.zeros((T, H, W * 2 + gap, 3), dtype=np.uint8)
    out[:, :, W:W + gap] = 255

    scene_mask = np.ones((H, W), dtype=bool)
    for t in range(T):
        out[t, :, :W]            = _colorize(scene[t],   scene_mask,  vmin, vmax, cmap)
        out[t, :, W + gap:]      = _colorize(robot_z[t], robot_m[t],  vmin, vmax, cmap)
        if (t + 1) % 100 == 0:
            print(f"  {t+1}/{T}")

    dst = pd / "depth_processor" / "depth_compare.mp4"
    media.write_video(str(dst), out, fps=args.fps, codec="libx264")
    print(f"[ok] wrote {dst}  (scene | robot side-by-side, range [{vmin:.2f}, {vmax:.2f}] m)")


if __name__ == "__main__":
    main()
