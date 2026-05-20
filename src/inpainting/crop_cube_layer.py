"""Step 1-2: Crop the cube out of the RAW video using depth-based layer separation.

Pipeline:
    cube_mask_t = ~M_hand[t]  AND  depth_aligned[t] ≤ quantile(depth, q)
    keep_largest_cc(cube_mask_t)               # drop gantry, table-edge leakage
    cropped[t] = raw[t]  where cube_mask_t  else  0

The raw-video depth (depth_aligned.npy) is used because, as we observed, depth
on the inpainted bg ranks the gantry as "close" and we can't tell the cube
from it. On the raw video, the hand mask still gates out the hand depths
properly.

Inputs:
    <pd>/video_L.mp4
    <pd>/depth_processor/depth_aligned.npy
    <pd>/segmentation_processor/masks_arm.npy

Outputs:
    <pd>/cube_layer/cube_cropped_raw.mp4    raw cropped to cube_mask
    <pd>/cube_layer/cube_mask_raw.npy       (T,H,W) bool — the cube mask used

Usage:
    python crop_cube_layer.py --processed_demo /result/cam0_inpaint/cam0/0 \
        --quantile 0.25
"""
import argparse
from pathlib import Path

import cv2
import mediapy as media
import numpy as np


def _keep_largest_cc(mask: np.ndarray) -> np.ndarray:
    """Connected-components: return mask with only the single largest CC kept."""
    if not mask.any():
        return mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8),
                                                             connectivity=8)
    if num <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return labels == largest


def _keep_center_cc(mask: np.ndarray, min_area: int = 200) -> np.ndarray:
    """Connected-components: keep the CC whose centroid is closest to image
    center, ignoring CCs smaller than `min_area`. The cube is the most-central
    foreground blob in this scene."""
    if not mask.any():
        return mask
    H, W = mask.shape
    cy_img, cx_img = H / 2.0, W / 2.0
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask
    best, best_d = None, float("inf")
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        cx, cy = centroids[i]
        d = (cx - cx_img) ** 2 + (cy - cy_img) ** 2
        if d < best_d:
            best_d, best = d, i
    if best is None:
        return mask
    return labels == best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--quantile", type=float, default=0.25,
                    help="depth quantile threshold on non-hand pixels (smaller "
                         "= keep only very close pixels)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--cc_pick", choices=["largest", "center", "none"],
                    default="center",
                    help="Connected-component selection. 'center' picks the CC "
                         "whose centroid is closest to image center (cube usually). "
                         "'largest' picks the biggest CC. 'none' keeps every CC.")
    ap.add_argument("--min_area", type=int, default=200,
                    help="Skip CCs smaller than this when picking with cc_pick=center.")
    args = ap.parse_args()

    pd = args.processed_demo
    rgb    = media.read_video(str(pd / "video_L.mp4"))
    depth  = np.load(pd / "depth_processor" / "depth_aligned.npy").astype(np.float32)
    m_hand = np.load(pd / "segmentation_processor" / "masks_arm.npy").astype(bool)
    T = min(rgb.shape[0], depth.shape[0], m_hand.shape[0])
    rgb, depth, m_hand = rgb[:T], depth[:T], m_hand[:T]
    H, W = rgb.shape[1], rgb.shape[2]

    cropped = np.zeros_like(rgb)
    cube_mask = np.zeros((T, H, W), dtype=bool)
    n_px = np.zeros(T, dtype=np.int64)
    thr_arr = np.zeros(T, dtype=np.float32)

    print(f"[info] T={T}, {W}x{H}, q={args.quantile}, cc_pick={args.cc_pick}")

    for t in range(T):
        non_hand = ~m_hand[t]
        d_nh = depth[t][non_hand]
        if d_nh.size == 0:
            continue
        thr = float(np.quantile(d_nh, args.quantile))
        thr_arr[t] = thr
        mask_t = non_hand & (depth[t] <= thr)
        if args.cc_pick == "largest":
            mask_t = _keep_largest_cc(mask_t)
        elif args.cc_pick == "center":
            mask_t = _keep_center_cc(mask_t, min_area=args.min_area)
        cube_mask[t] = mask_t
        n_px[t] = int(mask_t.sum())
        cropped[t][mask_t] = rgb[t][mask_t]
        if (t + 1) % 100 == 0:
            print(f"  {t+1}/{T}  thr={thr:.2f}m  cube_px={n_px[t]}")

    out_dir = pd / "cube_layer"
    out_dir.mkdir(parents=True, exist_ok=True)
    media.write_video(str(out_dir / "cube_cropped_raw.mp4"), cropped,
                      fps=args.fps, codec="libx264")
    np.save(out_dir / "cube_mask_raw.npy", cube_mask)
    print(f"[ok] wrote {out_dir/'cube_cropped_raw.mp4'}")
    print(f"[ok] wrote {out_dir/'cube_mask_raw.npy'}")
    print(f"[info] cube px per frame: median {int(np.median(n_px))}, "
          f"max {n_px.max()}, min {n_px.min()}")
    print(f"[info] threshold per frame: median {np.median(thr_arr):.2f}m, "
          f"range [{thr_arr.min():.2f}, {thr_arr.max():.2f}] m")


if __name__ == "__main__":
    main()
