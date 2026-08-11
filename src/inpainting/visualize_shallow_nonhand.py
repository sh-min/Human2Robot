"""Keep only RGB pixels that are
   (1) outside the hand mask AND
   (2) at depth ≤ that frame's median non-hand depth.
Everything else is set to black. Idea: the table sits near the per-frame
median depth, so this sweeps away the table+background and leaves only the
shallow non-hand stuff in front of it — which on this scene is mostly the
object.

Output:
    <processed_demo>/depth_processor/shallow_nonhand.mp4

Usage:
    python visualize_shallow_nonhand.py --processed_demo /result/cam0_inpaint/cam0/0
"""
import argparse
from pathlib import Path

import mediapy as media
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--quantile", type=float, default=0.5,
                    help="Per-frame threshold on non-hand depth. 0.5 = median. "
                         "Lower = stricter (keep only very-close pixels).")
    args = ap.parse_args()

    pd = args.processed_demo
    rgb    = media.read_video(str(pd / "video_L.mp4"))
    depth  = np.load(pd / "depth_processor" / "depth_aligned.npy").astype(np.float32)
    m_hand = np.load(pd / "segmentation_processor" / "masks_arm.npy").astype(bool)
    T = min(rgb.shape[0], depth.shape[0], m_hand.shape[0])
    rgb, depth, m_hand = rgb[:T], depth[:T], m_hand[:T]
    H, W = rgb.shape[1], rgb.shape[2]

    out = np.zeros_like(rgb)
    thresh = np.zeros(T, dtype=np.float32)
    kept = np.zeros(T, dtype=np.int64)
    for t in range(T):
        non_hand = ~m_hand[t]
        d_nh = depth[t][non_hand]
        if d_nh.size == 0:
            continue
        thresh[t] = float(np.quantile(d_nh, args.quantile))
        keep = non_hand & (depth[t] <= thresh[t])
        out[t][keep] = rgb[t][keep]
        kept[t] = int(keep.sum())
        if (t + 1) % 100 == 0:
            print(f"  {t+1}/{T}  thresh={thresh[t]:.2f}m  kept={kept[t]} px")

    dst = pd / "depth_processor" / "shallow_nonhand.mp4"
    media.write_video(str(dst), out, fps=args.fps, codec="libx264")
    print(f"[ok] wrote {dst}")
    print(f"[info] quantile={args.quantile}, median per-frame thresh "
          f"= {np.median(thresh):.2f} m (range [{thresh.min():.2f}, {thresh.max():.2f}])")
    print(f"[info] kept pixels per frame: median {int(np.median(kept))}, "
          f"max {kept.max()}, min {kept.min()}")


if __name__ == "__main__":
    main()
