"""Per-frame depth stats video (excluding hand-mask pixels).

For each frame `t`, compute the depth distribution over `~M_hand` (i.e. every
pixel NOT in the SAM2 arm/hand mask) and render an info panel showing:
  - histogram of non-hand depths (fixed x-axis = global 1st/99th percentile)
  - vertical lines for that frame's median + mean
  - text overlay: min/p05/median/mean/p95/max + non-hand pixel count

Layout: raw RGB (with hand outline) | stats panel.

Output:
    <processed_demo>/depth_processor/depth_stats.mp4

Usage:
    python visualize_depth_stats.py --processed_demo /result/cam0_inpaint/cam0/0
"""
import argparse
import io
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
from PIL import Image


OUTLINE_RGB = np.array([255, 0, 0], dtype=np.uint8)


def _outline(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    m = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return (cv2.dilate(m, k, iterations=thickness) != cv2.erode(m, k, iterations=thickness))


def _render_hist_panel(depths: np.ndarray, edges: np.ndarray,
                       global_ymax: float, t: int, T: int,
                       size_wh: tuple[int, int]) -> np.ndarray:
    """Render the histogram + stats panel as RGB at exactly `size_wh` pixels."""
    w, h = size_wh
    dpi = 100
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.hist(depths, bins=edges, color="#3a7ed1", edgecolor="none")

    if depths.size:
        p05, med, mean, p95 = np.percentile(depths, [5, 50, 95]).tolist()[0], \
                              float(np.median(depths)), float(depths.mean()), \
                              np.percentile(depths, 95)
        ax.axvline(med,  color="orange", lw=1.5, label=f"median {med:.2f} m")
        ax.axvline(mean, color="red",    lw=1.5, ls="--", label=f"mean {mean:.2f} m")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        txt = (f"frame {t:4d}/{T}\n"
               f"N    {depths.size}\n"
               f"min  {depths.min():.2f}\n"
               f"p05  {p05:.2f}\n"
               f"med  {med:.2f}\n"
               f"mean {mean:.2f}\n"
               f"p95  {p95:.2f}\n"
               f"max  {depths.max():.2f}")
    else:
        txt = f"frame {t}/{T}\nno non-hand pixels"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=8,
            family="monospace", va="top", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, global_ymax)
    ax.set_xlabel(_render_hist_panel.xlabel)
    ax.set_ylabel("pixel count")
    ax.set_title(_render_hist_panel.title)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))
    if img.shape[:2] != (h, w):
        img = np.array(Image.fromarray(img).resize((w, h)))
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--vmin", type=float, default=None,
                    help="histogram x-min (default: 1st percentile across all frames)")
    ap.add_argument("--vmax", type=float, default=None,
                    help="histogram x-max (default: 99th percentile across all frames)")
    ap.add_argument("--rgb_video", default="video_L.mp4",
                    help="RGB video (left panel, used when --left_panel=rgb)")
    ap.add_argument("--left_panel", choices=["rgb", "depth"], default="rgb",
                    help="What to show on the left: raw RGB or a colorized depth map.")
    ap.add_argument("--cmap", default="turbo",
                    help="matplotlib colormap when --left_panel=depth")
    ap.add_argument("--depth_npy", default="depth_processor/depth_aligned.npy",
                    help="(T,H,W) depth or disparity array")
    ap.add_argument("--mask_npy", default="segmentation_processor/masks_arm.npy",
                    help="(T,H,W) bool — pixels EXCLUDED from the histogram. "
                         "Pass an empty string to disable.")
    ap.add_argument("--xlabel", default="depth (m)",
                    help="x-axis label for the histogram")
    ap.add_argument("--out_name", default="depth_stats.mp4",
                    help="output filename under depth_processor/")
    args = ap.parse_args()

    pd = args.processed_demo
    depth = np.load(pd / args.depth_npy).astype(np.float32)
    if args.left_panel == "rgb":
        rgb = media.read_video(str(pd / args.rgb_video))
    else:
        # Build a colorized depth video to use as the "left panel".
        cmap = matplotlib.colormaps[args.cmap]
        d_lo = float(np.percentile(depth, 5))
        d_hi = float(np.percentile(depth, 95))
        rng = max(d_hi - d_lo, 1e-6)
        norm = np.clip((depth - d_lo) / rng, 0, 1)
        rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
        print(f"[info] left panel = colorized depth, "
              f"range [{d_lo:.2f}, {d_hi:.2f}] (5/95 percentile), cmap={args.cmap}")
    if args.mask_npy:
        m_hand = np.load(pd / args.mask_npy).astype(bool)
    else:
        m_hand = np.zeros(depth.shape, dtype=bool)
    T = min(rgb.shape[0], depth.shape[0], m_hand.shape[0])
    rgb, depth, m_hand = rgb[:T], depth[:T], m_hand[:T]
    H, W = rgb.shape[1], rgb.shape[2]

    # Global x-range + y-max — sample to keep memory bounded.
    flat = depth[~m_hand]
    sample = flat[::max(1, flat.size // 5_000_000)]   # at most ~5M values
    vmin = float(np.percentile(sample, 1))  if args.vmin is None else args.vmin
    vmax = float(np.percentile(sample, 99)) if args.vmax is None else args.vmax
    edges = np.linspace(vmin, vmax, args.bins + 1)

    # Estimate global histogram y-max so the y-axis is comparable across frames.
    global_ymax = 0
    for t in range(T):
        d = depth[t][~m_hand[t]]
        if d.size:
            c, _ = np.histogram(d, bins=edges)
            if c.max() > global_ymax:
                global_ymax = int(c.max())
    global_ymax = int(global_ymax * 1.05)
    print(f"[info] T={T}, x-range used for hist: [{vmin:.2f}, {vmax:.2f}], "
          f"global y-max ≈ {global_ymax} px/bin")

    # Push title/xlabel into the inner render function so we don't thread args through.
    _render_hist_panel.xlabel = args.xlabel
    _render_hist_panel.title = ("non-hand " if args.mask_npy else "") + "depth distribution"

    gap = 10
    out = np.zeros((T, H, 2 * W + gap, 3), dtype=np.uint8)
    out[:, :, W:W + gap] = 255

    for t in range(T):
        d = depth[t][~m_hand[t]]
        left = rgb[t].copy()
        left[_outline(m_hand[t])] = OUTLINE_RGB
        right = _render_hist_panel(d, edges, global_ymax, t, T, (W, H))
        out[t, :, :W]            = left
        out[t, :, W + gap:]      = right
        if (t + 1) % 50 == 0:
            print(f"  {t+1}/{T}")

    dst = pd / "depth_processor" / args.out_name
    media.write_video(str(dst), out, fps=args.fps, codec="libx264")
    print(f"[ok] wrote {dst}")


if __name__ == "__main__":
    main()
