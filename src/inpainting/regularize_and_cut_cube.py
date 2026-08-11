"""Steps 3-5: regularize the rough cube mask, then cut the legacy-inpainted bg
along those clean edges to produce a clean cube layer.

Regularization (per frame):
    1. Morphological OPEN with a small kernel — severs thin connections to the
       table edge (the appendage we saw bottom-left in the rough crop).
    2. Pick the largest connected component — drops residual specks.
    3. Morphological CLOSE — smooths the boundary and fills small interior holes.
    4. Optional small dilation — gives the cube a few-pixel margin so cutting
       doesn't shave the sticker edges.

We use `inpaint_processor/video_human_inpaint.mkv` as the "inpainted video"
(legacy E2FGVI on M_hand), so the human hands are already gone inside the
cube area. Cutting that along the clean edges gives a finger-contamination-
free cube cutout.

Inputs:
    <pd>/inpaint_processor/video_human_inpaint.mkv     legacy-inpainted bg
    <pd>/cube_layer/cube_mask_raw.npy                  rough mask (from crop_cube_layer)

Outputs:
    <pd>/cube_layer/cube_mask_clean.npy                regularized mask
    <pd>/cube_layer/cube_layer_final.mp4               cube cutout on inpainted bg
    <pd>/cube_layer/cube_edges_viz.mp4                 inpainted bg with red edge outline

Usage:
    python regularize_and_cut_cube.py --processed_demo /result/cam0_inpaint/cam0/0
"""
import argparse
from pathlib import Path

import cv2
import mediapy as media
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, uniform_filter1d


def _center_cc(mask: np.ndarray, min_area: int = 200) -> np.ndarray:
    """Keep the CC whose centroid is closest to image center."""
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


def _convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    """Replace `mask` with its convex hull (fills in finger-bite concavities)."""
    if not mask.any():
        return mask
    ys, xs = np.where(mask)
    pts = np.stack([xs, ys], axis=1)
    if len(pts) < 3:
        return mask
    hull = cv2.convexHull(pts.astype(np.int32))
    out = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(out, [hull], 1)
    return out.astype(bool)


def _regularize(mask_t: np.ndarray, open_k: int, close_k: int, dilate_k: int,
                min_area: int, hull: bool) -> np.ndarray:
    m = mask_t.astype(np.uint8)
    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = _center_cc(m.astype(bool), min_area=min_area).astype(np.uint8)
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if hull:
        m = _convex_hull_mask(m.astype(bool)).astype(np.uint8)
    if dilate_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        m = cv2.dilate(m, k, iterations=1)
    return m.astype(bool)


def _outline(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    m = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(m, k, iterations=thickness) != cv2.erode(m, k, iterations=thickness)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--open_k",   type=int, default=7,
                    help="kernel size for MORPH_OPEN (sever appendages)")
    ap.add_argument("--close_k",  type=int, default=5,
                    help="kernel size for MORPH_CLOSE (smooth + fill holes)")
    ap.add_argument("--dilate_k", type=int, default=3,
                    help="kernel size for final MORPH_DILATE (edge margin)")
    ap.add_argument("--min_area", type=int, default=200,
                    help="min CC area when picking center CC")
    ap.add_argument("--no_hull", action="store_true",
                    help="Skip the convex-hull step (keep finger bites).")
    ap.add_argument("--temporal_window", type=int, default=7,
                    help="Box-filter window (odd) on the binary mask. "
                         "Ignored when --sdf_sigma > 0. 1 = off.")
    ap.add_argument("--temporal_thresh", type=float, default=0.5,
                    help="Threshold on the temporally-averaged float mask.")
    ap.add_argument("--sdf_sigma", type=float, default=0.0,
                    help="Gaussian sigma (in frames) for SDF temporal smoothing. "
                         "0 = use box filter on the binary mask instead. "
                         "Typical values: 2 (mild), 4 (noticeable), 8 (very stiff).")
    ap.add_argument("--sdf_clip", type=float, default=20.0,
                    help="Clip SDF values to ±this many pixels before smoothing. "
                         "Smaller = boundary stiffer near object; larger = "
                         "smoother but pulls in distant context.")
    ap.add_argument("--area_mad_k", type=float, default=5.0,
                    help="Outlier: |area - median(area)| > k * MAD(area). Set "
                         "<=0 to disable. Outlier frames are replaced by the "
                         "temporally nearest non-outlier mask.")
    ap.add_argument("--centroid_max_jump", type=float, default=120.0,
                    help="Outlier: centroid >this many pixels from temporal "
                         "median centroid. Set <=0 to disable.")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--bg_video", default="inpaint_processor/video_human_inpaint.mkv")
    ap.add_argument("--raw_mask", default="cube_layer/cube_mask_raw.npy",
                    help="input mask to regularize, relative to processed_demo. "
                         "Point at cube_layer/cube_mask_amodal.npy when running "
                         "after the Diffusion-VAS amodal stage.")
    args = ap.parse_args()

    pd = args.processed_demo
    bg = media.read_video(str(pd / args.bg_video))
    raw_mask = np.load(pd / args.raw_mask).astype(bool)
    T = min(bg.shape[0], raw_mask.shape[0])
    bg, raw_mask = bg[:T], raw_mask[:T]
    H, W = bg.shape[1], bg.shape[2]

    smoothing_desc = (f"SDF sigma={args.sdf_sigma} clip=±{args.sdf_clip}px"
                      if args.sdf_sigma > 0
                      else f"box window={args.temporal_window} thresh={args.temporal_thresh}")
    print(f"[info] T={T}, {W}x{H}, open_k={args.open_k}, "
          f"close_k={args.close_k}, dilate_k={args.dilate_k}, "
          f"hull={not args.no_hull}, smoothing=[{smoothing_desc}]")

    clean_mask = np.zeros_like(raw_mask)
    cut_frames = np.zeros_like(bg)
    edge_frames = bg.copy()

    n_before, n_after = 0, 0
    # Phase 1: per-frame regularization (open, CC, close, hull, dilate)
    for t in range(T):
        mt = _regularize(raw_mask[t], args.open_k, args.close_k, args.dilate_k,
                         args.min_area, hull=not args.no_hull)
        clean_mask[t] = mt
        n_before += int(raw_mask[t].sum())
        if (t + 1) % 200 == 0:
            print(f"  per-frame regularize {t+1}/{T}")

    # Phase 1.5: knock out outlier frames (replace with nearest non-outlier).
    areas = clean_mask.sum(axis=(1, 2)).astype(np.float32)
    cents = np.full((T, 2), np.nan, dtype=np.float32)
    for t in range(T):
        if areas[t] > 0:
            ys, xs = np.where(clean_mask[t])
            cents[t] = [xs.mean(), ys.mean()]

    area_outlier = np.zeros(T, dtype=bool)
    if args.area_mad_k > 0:
        med = float(np.median(areas))
        mad = float(np.median(np.abs(areas - med))) + 1e-6
        area_outlier = np.abs(areas - med) > args.area_mad_k * mad
        print(f"[outliers] area median={med:.0f} MAD={mad:.0f} → "
              f"{area_outlier.sum()} frames flagged by area")

    centroid_outlier = np.zeros(T, dtype=bool)
    if args.centroid_max_jump > 0:
        valid = ~np.isnan(cents[:, 0])
        if valid.any():
            cx_med = float(np.nanmedian(cents[valid, 0]))
            cy_med = float(np.nanmedian(cents[valid, 1]))
            dist = np.where(valid,
                            np.hypot(cents[:, 0] - cx_med, cents[:, 1] - cy_med),
                            np.inf)
            centroid_outlier = dist > args.centroid_max_jump
            print(f"[outliers] centroid median=({cx_med:.0f},{cy_med:.0f}) → "
                  f"{centroid_outlier.sum()} frames flagged by centroid "
                  f"(jump > {args.centroid_max_jump}px)")

    outlier = area_outlier | centroid_outlier | (areas == 0)
    if outlier.any():
        good = np.where(~outlier)[0]
        if good.size == 0:
            print("[outliers] WARN: all frames flagged; keeping originals")
        else:
            for t in np.where(outlier)[0]:
                nearest = good[np.argmin(np.abs(good - t))]
                clean_mask[t] = clean_mask[nearest]
            print(f"[outliers] replaced {outlier.sum()} outlier frames "
                  f"(area only: {(area_outlier & ~centroid_outlier).sum()}, "
                  f"centroid only: {(centroid_outlier & ~area_outlier).sum()}, "
                  f"both: {(area_outlier & centroid_outlier).sum()})")

    # Phase 2: temporal smoothing.
    if args.sdf_sigma > 0:
        # SDF approach: per-frame signed distance field → temporally Gaussian-smooth → threshold at 0.
        print(f"[info] computing per-frame SDFs (sigma_t={args.sdf_sigma} frames, "
              f"clip=±{args.sdf_clip}px)…")
        sdf = np.zeros(clean_mask.shape, dtype=np.float32)
        for t in range(T):
            m = clean_mask[t]
            if m.all() or not m.any():
                sdf[t] = args.sdf_clip if m.all() else -args.sdf_clip
                continue
            d_in  = distance_transform_edt(m)
            d_out = distance_transform_edt(~m)
            sdf[t] = np.clip(d_in - d_out, -args.sdf_clip, args.sdf_clip)
            if (t + 1) % 200 == 0:
                print(f"  SDF {t+1}/{T}")
        print(f"[info] gaussian smoothing SDF along time axis…")
        sdf = gaussian_filter1d(sdf, sigma=args.sdf_sigma, axis=0, mode="nearest")
        clean_mask = sdf > 0
        del sdf
    elif args.temporal_window > 1:
        w = args.temporal_window
        if w % 2 == 0:
            w += 1
        print(f"[info] box-filter temporal smoothing: window={w}")
        smoothed = uniform_filter1d(clean_mask.astype(np.float32),
                                    size=w, axis=0, mode="nearest")
        clean_mask = smoothed >= args.temporal_thresh
        del smoothed

    # Phase 3: write cut frames + edges
    for t in range(T):
        mt = clean_mask[t]
        cut_frames[t][mt] = bg[t][mt]
        ed = _outline(mt, thickness=2)
        edge_frames[t][ed] = (255, 0, 0)
        n_after += int(mt.sum())

    out_dir = pd / "cube_layer"
    np.save(out_dir / "cube_mask_clean.npy", clean_mask)
    media.write_video(str(out_dir / "cube_layer_final.mp4"), cut_frames,
                      fps=args.fps, codec="libx264")
    media.write_video(str(out_dir / "cube_edges_viz.mp4"), edge_frames,
                      fps=args.fps, codec="libx264")
    print(f"[ok] wrote {out_dir/'cube_mask_clean.npy'}")
    print(f"[ok] wrote {out_dir/'cube_layer_final.mp4'}")
    print(f"[ok] wrote {out_dir/'cube_edges_viz.mp4'}")
    print(f"[info] total cube px: raw={n_before}, clean={n_after} "
          f"({100*(n_after-n_before)/max(n_before,1):+.1f}%)")
    per_clean = clean_mask.sum(axis=(1, 2))
    print(f"[info] clean cube_px per frame: median {int(np.median(per_clean))}, "
          f"max {per_clean.max()}, min {per_clean.min()}")


if __name__ == "__main__":
    main()
