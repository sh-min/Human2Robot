"""Stage 8 (SAM2 variant): segment the cube across the video with SAM2,
bootstrapped from a single depth-derived seed frame.

Replaces the depth-quantile + per-frame center-CC isolation in
crop_cube_layer.py. Thresholding depth on *every* frame leaks the table;
here depth is used only to *locate* the cube on one good seed frame, then
SAM2's video predictor tracks the object through the whole clip.

Bootstrap:
    1. Per frame, compute the rough depth mask (non-hand pixels whose
       aligned depth is in the closest `quantile`), keep the center-most CC.
    2. Pick the seed frame: the rough mask with the largest distance-
       transform peak — the most compact, solid blob, i.e. the cleanest
       SAM2 prompt.
    3. From that frame derive a box (mask extent) + one positive point
       (distance-transform peak, guaranteed interior even when finger
       bites make the blob concave).
    4. SAM2 propagates forward + reverse from the seed; union the passes.

Inputs:
    <pd>/video_L.mp4
    <pd>/depth_processor/depth_aligned.npy
    <pd>/segmentation_processor/masks_arm.npy

Outputs:
    <pd>/cube_layer/cube_mask_raw.npy       (T,H,W) bool — SAM2 cube mask
    <pd>/cube_layer/cube_cropped_raw.mp4    raw cropped to the mask (debug)

Stage 9 (regularize_and_cut_cube.py) consumes cube_mask_raw.npy unchanged.

Usage:
    python segment_cube.py --processed_demo /result/skill2policy/processed/cam0/0 \
        --quantile 0.25
"""
import argparse
import shutil
import sys
from pathlib import Path

import mediapy as media
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from _paths import SAM2_CHECKPOINT, SAM2_CONFIG_NAME, ensure_sam2_importable
from crop_cube_layer import _keep_center_cc
from segment_arms import _dump_frames_as_jpegs, _segment_one_pass

ensure_sam2_importable()
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _depth_rough_masks(depth: np.ndarray, m_hand: np.ndarray,
                       quantile: float, min_area: int) -> np.ndarray:
    """Per-frame depth-quantile cube mask (closest-`quantile` non-hand
    pixels, center CC). Used only to locate a SAM2 seed frame."""
    T = depth.shape[0]
    rough = np.zeros(depth.shape, dtype=bool)
    for t in range(T):
        non_hand = ~m_hand[t]
        d_nh = depth[t][non_hand]
        if d_nh.size == 0:
            continue
        thr = float(np.quantile(d_nh, quantile))
        mt = non_hand & (depth[t] <= thr)
        rough[t] = _keep_center_cc(mt, min_area=min_area)
    return rough


# Seed-frame area band, as a multiple of the median rough-mask area. The
# cube is a small, stable blob (~median area); table-leak frames blow the
# mask up to many times that. Candidate seed frames must fall inside this
# band so a leaky frame is never chosen, however "solid" it looks.
_SEED_AREA_BAND = (0.5, 2.0)


def _pick_seed(rough: np.ndarray, min_area: int):
    """Seed = frame whose rough mask has the largest distance-transform
    peak (most compact, solid blob), restricted to frames whose area is
    near the median (rejects table-leak frames). Returns (seed_idx, box,
    point) where box is (x0,y0,x1,y1) and point is (1,2); (None,)*3 if
    no usable frame."""
    areas = rough.reshape(rough.shape[0], -1).sum(axis=1).astype(np.float64)
    in_dist = areas[areas >= min_area]
    if in_dist.size == 0:
        return None, None, None
    med = float(np.median(in_dist))
    lo = max(float(min_area), _SEED_AREA_BAND[0] * med)
    hi = _SEED_AREA_BAND[1] * med
    print(f"[info] rough-mask area median={med:.0f}px → "
          f"seed candidates within [{lo:.0f}, {hi:.0f}]px")

    best_idx, best_score, best_dt = None, -1.0, None
    for t in range(rough.shape[0]):
        if not (lo <= areas[t] <= hi):
            continue
        dt = distance_transform_edt(rough[t])
        score = float(dt.max())
        if score > best_score:
            best_idx, best_score, best_dt = t, score, dt
    if best_idx is None:
        return None, None, None
    ys, xs = np.where(rough[best_idx])
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
    py, px = np.unravel_index(int(np.argmax(best_dt)), best_dt.shape)
    point = np.array([[px, py]], dtype=np.float32)
    return best_idx, box, point


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--quantile", type=float, default=0.25,
                    help="depth top-quantile for the bootstrap seed mask "
                         "(smaller = only the very closest pixels)")
    ap.add_argument("--min_area", type=int, default=200,
                    help="ignore rough-mask CCs / candidate seed frames "
                         "smaller than this many pixels")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--keep_tmp", action="store_true",
                    help="keep the original_images/ JPEG dump for debugging")
    args = ap.parse_args()

    if not Path(SAM2_CHECKPOINT).exists():
        sys.exit(f"SAM2 checkpoint missing: {SAM2_CHECKPOINT}\n"
                 f"Download with: wget -P {Path(SAM2_CHECKPOINT).parent} "
                 f"https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt")

    pd = args.processed_demo
    video_path = pd / "video_L.mp4"
    rgb    = media.read_video(str(video_path))
    depth  = np.load(pd / "depth_processor" / "depth_aligned.npy").astype(np.float32)
    m_hand = np.load(pd / "segmentation_processor" / "masks_arm.npy").astype(bool)
    Tb = min(rgb.shape[0], depth.shape[0], m_hand.shape[0])
    H, W = rgb.shape[1], rgb.shape[2]

    print(f"[info] depth bootstrap: rough masks over {Tb} frames (q={args.quantile})")
    rough = _depth_rough_masks(depth[:Tb], m_hand[:Tb], args.quantile, args.min_area)
    seed_idx, seed_box, seed_pt = _pick_seed(rough, args.min_area)
    if seed_idx is None:
        sys.exit("[err] depth bootstrap found no cube blob — try a different "
                 "--quantile, or seed SAM2 manually.")
    print(f"[info] seed frame={seed_idx} box={seed_box.round(1).tolist()} "
          f"point={seed_pt[0].round(1).tolist()}")

    frames_dir = pd / "original_images"
    n_frames = _dump_frames_as_jpegs(video_path, frames_dir)
    print(f"[info] T={n_frames}, {W}x{H}")

    video_predictor = build_sam2_video_predictor(SAM2_CONFIG_NAME, SAM2_CHECKPOINT,
                                                 device=DEVICE)

    cube_mask = np.zeros((n_frames, H, W), dtype=bool)
    for reverse in (False, True):
        out = _segment_one_pass(video_predictor, frames_dir,
                                seed_box, seed_pt, seed_idx, reverse=reverse)
        for idx, m in out.items():
            cube_mask[idx] |= m[0]

    per_frame = cube_mask.sum(axis=(1, 2))
    print(f"[info] frames with cube mask: {(per_frame > 0).sum()}/{n_frames}, "
          f"median {int(np.median(per_frame))} px, max {per_frame.max()} px")

    out_dir = pd / "cube_layer"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cube_mask_raw.npy", cube_mask)

    Tc = min(n_frames, rgb.shape[0])
    cropped = np.zeros_like(rgb[:Tc])
    for t in range(Tc):
        cropped[t][cube_mask[t]] = rgb[t][cube_mask[t]]
    media.write_video(str(out_dir / "cube_cropped_raw.mp4"), cropped,
                      fps=args.fps, codec="libx264")
    print(f"[ok] wrote {out_dir / 'cube_mask_raw.npy'}")
    print(f"[ok] wrote {out_dir / 'cube_cropped_raw.mp4'}")

    if not args.keep_tmp:
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
