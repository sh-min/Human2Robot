"""Stage 8b: video amodal segmentation of the cube with Diffusion-VAS.

The SAM2 mask from segment_cube.py (stage 8a) is *modal* — only the
visible cube pixels, so gripping fingers bite chunks out of it. Diffusion-
VAS (CVPR 2025) is a Stable-Video-Diffusion model that completes the full
*amodal* silhouette.

Diffusion-VAS processes exactly 25 frames per call, so a long clip is cut
into 25-frame windows; this script runs each window and stitches the
results back together.

Post-processing pipeline (applied after all windows are stitched):
    1. Per-frame top-percentile threshold on the raw RGB channel sum
    2. Union with SAM2 modal mask (never lose visible pixels)
    3. Clip to expanded modal bbox (reject far-flung noise)
    4. Morph open/close + largest CC
    5. SDF temporal smoothing (Gaussian along time axis)

Content completion is a separate step: see content_completion.py (stage 8c).

This script must run in the `diffusion_vas` conda env (NOT `inpaint`):
    conda run -n diffusion_vas python amodal_cube.py --processed_demo <pd>

Inputs:
    <pd>/video_L.mp4
    <pd>/cube_layer/cube_mask_raw.npy        (T,H,W) bool — SAM2 modal mask

Outputs:
    <pd>/cube_layer/cube_rawsum.npy          (T,H,W) float16 — raw channel sums
    <pd>/cube_layer/cube_mask_amodal.npy     (T,H,W) bool — amodal silhouette
    <pd>/cube_layer/cube_amodal_overlay.mp4  raw + amodal outline (debug)

Per-window checkpoints are saved under <pd>/cube_layer/_amodal_ckpt/ so the
script can resume after OOM crashes without re-running completed windows.

Usage:
    python amodal_cube.py --processed_demo /result/skill2policy/processed/cam0/0 \\
        --top_percentile 1 --smooth_sigma 2.0
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter1d, distance_transform_edt
from torchvision import transforms

from _paths import (DEPTH_ANYTHING_CKPT_DIR, DIFFUSION_VAS_MASK_CKPT,
                     ensure_diffusion_vas_importable)

ensure_diffusion_vas_importable()
import demo as dvas  # noqa: E402

NUM_FRAMES = 25  # Diffusion-VAS / SVD backbone is fixed to 25-frame clips


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_video(path: Path) -> np.ndarray:
    """Read an mp4 to (T,H,W,3) uint8 RGB without mediapy (not in this env)."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def _write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    H, W = frames.shape[1:3]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def _outline(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    m = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(m, k, iterations=thickness).astype(bool) ^ \
        cv2.erode(m, k, iterations=thickness).astype(bool)


# ---------------------------------------------------------------------------
# Tensor conversion
# ---------------------------------------------------------------------------

def _masks_to_tensor(window: np.ndarray, res) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize(res), transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    out = []
    for m in window:
        img = Image.fromarray((m.astype(np.uint8) * 255)).convert("L")
        img = img.point(lambda p: 255 if p > 128 else 0)
        out.append(tf(img))
    return torch.stack(out).unsqueeze(0)


def _rgbs_to_tensor(window: np.ndarray, res) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize(res), transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    out = [tf(Image.fromarray(f).convert("RGB")) for f in window]
    return torch.stack(out).unsqueeze(0)


def _infer(pipeline, cond_a, cond_b, res, generator):
    """One Diffusion-VAS 25-frame pass. Returns (25,h,w,3) uint8."""
    frames = pipeline(
        cond_a, cond_b,
        height=res[0], width=res[1], num_frames=NUM_FRAMES,
        decode_chunk_size=8, motion_bucket_id=127, fps=8,
        noise_aug_strength=0.02, min_guidance_scale=1.5, max_guidance_scale=1.5,
        generator=generator,
    ).frames[0]
    return [np.array(im).astype(np.uint8) for im in frames]


# ---------------------------------------------------------------------------
# Post-processing: top-percentile + morph + largest-CC + SDF smoothing
# ---------------------------------------------------------------------------

def _smooth_amodal(rawsum: np.ndarray, modal: np.ndarray,
                   top_pct: float, sigma_t: float,
                   bbox_margin: int = 25, clip_px: float = 15.0) -> np.ndarray:
    """Per-frame top-percentile threshold → morph cleanup → SDF temporal smooth.

    Args:
        rawsum:     (T,H,W) float — average RGB channel sum per pixel.
        modal:      (T,H,W) bool  — SAM2 modal mask.
        top_pct:    percentile (e.g. 1 = top 1% brightest pixels per frame).
        sigma_t:    Gaussian sigma in frames for SDF temporal smoothing.
        bbox_margin: px margin around modal bbox for clipping noise.
        clip_px:    SDF clip distance in pixels.

    Returns:
        (T,H,W) bool — smoothed amodal mask.
    """
    T, H, W = rawsum.shape
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    smooth = np.zeros((T, H, W), dtype=bool)
    for t in range(T):
        # Top-percentile threshold + modal union
        thr_t = np.percentile(rawsum[t], 100 - top_pct)
        m = np.logical_or(rawsum[t] >= thr_t, modal[t]).astype(np.uint8)

        # Clip to expanded modal bbox
        if modal[t].any():
            ys, xs = np.where(modal[t])
            y0 = max(0, ys.min() - bbox_margin)
            y1 = min(H, ys.max() + bbox_margin)
            x0 = max(0, xs.min() - bbox_margin)
            x1 = min(W, xs.max() + bbox_margin)
            bbox = np.zeros((H, W), dtype=np.uint8)
            bbox[y0:y1, x0:x1] = 1
            m = m & bbox

        # Morph open/close
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)

        # Largest connected component
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m)
        if n_labels > 1:
            biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            m = (labels == biggest).astype(np.uint8)

        # No convex hull: Diffusion-VAS already amodal-completes the
        # finger-occluded silhouette, so a convexity prior here would only
        # destroy genuinely non-convex object shapes.
        smooth[t] = m.astype(bool)

    # SDF temporal smoothing
    if sigma_t > 0:
        sdf = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            if smooth[t].any():
                d_in = distance_transform_edt(smooth[t])
                d_out = distance_transform_edt(~smooth[t])
                sdf[t] = np.clip(d_in - d_out, -clip_px, clip_px)
            else:
                sdf[t] = -clip_px
        sdf = gaussian_filter1d(sdf, sigma=sigma_t, axis=0, mode="nearest")
        smooth = sdf >= 0

    return smooth


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--overlap", type=int, default=0,
                    help="frames shared between consecutive 25-frame windows "
                         "(0 = non-overlapping; 4-6 smooths window seams)")
    ap.add_argument("--pred_res", default="256,512",
                    help="Diffusion-VAS inference resolution 'H,W'")
    ap.add_argument("--depth_encoder", default="vitl",
                    choices=["vits", "vitb", "vitl"])
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max_windows", type=int, default=0,
                    help="process only the first N windows (smoke test; 0 = all)")
    # Post-processing
    ap.add_argument("--top_percentile", type=float, default=1.0,
                    help="per-frame top-N%% brightest pixels as amodal mask")
    ap.add_argument("--smooth_sigma", type=float, default=2.0,
                    help="SDF temporal smoothing sigma in frames (0 = off)")
    ap.add_argument("--bbox_margin", type=int, default=25,
                    help="px margin around modal bbox for noise clipping")
    # Parallel sharding across GPUs (windows are independent)
    ap.add_argument("--shard_id", type=int, default=-1,
                    help="if >=0, only compute windows where w %% num_shards == "
                         "shard_id, then exit (run one process per GPU)")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--assemble_only", action="store_true",
                    help="skip inference; load all window checkpoints, aggregate "
                         "and post-process into cube_mask_amodal.npy")
    args = ap.parse_args()

    if not (0 <= args.overlap < NUM_FRAMES):
        sys.exit(f"--overlap must be in [0, {NUM_FRAMES})")
    res = tuple(int(x) for x in args.pred_res.split(","))

    pd = args.processed_demo
    rgb = _read_video(pd / "video_L.mp4")
    modal = np.load(pd / "cube_layer" / "cube_mask_raw.npy").astype(bool)
    T = min(rgb.shape[0], modal.shape[0])
    rgb, modal = rgb[:T], modal[:T]
    H, W = rgb.shape[1:3]

    stride = NUM_FRAMES - args.overlap
    starts = list(range(0, T, stride))
    if args.max_windows > 0:
        starts = starts[:args.max_windows]
        print(f"[info] SMOKE TEST: first {len(starts)} window(s) only")
    print(f"[info] T={T}, {W}x{H}, res={res}")
    print(f"[info] {len(starts)} windows of {NUM_FRAMES} frames "
          f"(stride={stride}, overlap={args.overlap})")

    depth_ckpt = f"{DEPTH_ANYTHING_CKPT_DIR}/depth_anything_v2_{args.depth_encoder}.pth"
    if args.assemble_only:
        pipe_mask = depth_model = None
        print("[info] assemble-only: skipping model load", flush=True)
    else:
        print("[info] loading Diffusion-VAS amodal segmentation pipeline...",
              flush=True)
        pipe_mask = dvas.init_amodal_segmentation_model(DIFFUSION_VAS_MASK_CKPT)
        depth_model = dvas.init_depth_model(depth_ckpt, args.depth_encoder)
        pipe_mask.set_progress_bar_config(disable=False)

    # -- Per-window inference with checkpointing --
    ckpt_dir = pd / "cube_layer" / "_amodal_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    accum_rawsum = np.zeros((T, H, W), dtype=np.float32)
    count = np.zeros(T, dtype=np.int32)

    # Which windows does this process compute? (windows are independent)
    if args.shard_id >= 0:
        my_windows = [w for w in range(len(starts))
                      if w % args.num_shards == args.shard_id]
        print(f"[shard {args.shard_id}/{args.num_shards}] assigned windows "
              f"{[w + 1 for w in my_windows]}", flush=True)
    elif args.assemble_only:
        my_windows = []
    else:
        my_windows = list(range(len(starts)))

    for w in my_windows:
        cp = ckpt_dir / f"w{w:03d}.npy"
        if cp.exists():
            print(f"  [skip] window {w+1}/{len(starts)} already checkpointed",
                  flush=True)
            continue
        s = starts[w]
        idx = list(range(s, min(s + NUM_FRAMES, T)))
        widx = idx + [idx[-1]] * (NUM_FRAMES - len(idx))
        modal_t = _masks_to_tensor(modal[widx], res)
        rgb_t = _rgbs_to_tensor(rgb[widx], res)
        depth_t = dvas.rgb_to_depth(rgb_t, depth_model)

        print(f"[window {w+1}/{len(starts)}] frames [{idx[0]},{idx[-1]}]  "
              f"amodal segmentation (25 SVD steps)...", flush=True)
        amodal_raw = np.stack(_infer(pipe_mask, modal_t, depth_t, res,
                                     torch.manual_seed(args.seed)))
        rawsum = amodal_raw.sum(axis=-1).astype(np.float32)  # (25,h,w)
        rawsum_full = np.stack([cv2.resize(rs, (W, H),
                                           interpolation=cv2.INTER_LINEAR)
                                for rs in rawsum])
        np.save(cp, rawsum_full[:len(idx)])
        print(f"  [ckpt] saved window {w+1}/{len(starts)}", flush=True)

    # Sharded workers stop after their windows; a separate --assemble_only run
    # (or the non-sharded path below) aggregates once all checkpoints exist.
    if args.shard_id >= 0:
        done = sum((ckpt_dir / f"w{w:03d}.npy").exists()
                   for w in range(len(starts)))
        print(f"[shard {args.shard_id}] finished; "
              f"{done}/{len(starts)} total checkpoints present", flush=True)
        return

    # Aggregate from ALL window checkpoints
    missing = [w + 1 for w in range(len(starts))
               if not (ckpt_dir / f"w{w:03d}.npy").exists()]
    if missing:
        sys.exit(f"[error] cannot assemble: missing window checkpoints {missing}")
    for w, s in enumerate(starts):
        rawsum_w = np.load(ckpt_dir / f"w{w:03d}.npy")
        idx = list(range(s, min(s + NUM_FRAMES, T)))
        for j, fi in enumerate(idx):
            accum_rawsum[fi] += rawsum_w[j]
            count[fi] += 1

    # -- Aggregate and post-process --
    cov = np.maximum(count, 1)
    avg_rawsum = accum_rawsum / cov[:, None, None]

    out_dir = pd / "cube_layer"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cube_rawsum.npy", avg_rawsum.astype(np.float16))
    print(f"[ok] wrote {out_dir / 'cube_rawsum.npy'}")

    print(f"[info] post-processing: top {args.top_percentile}%, "
          f"sigma={args.smooth_sigma}, bbox_margin={args.bbox_margin}px",
          flush=True)
    amodal_mask = _smooth_amodal(avg_rawsum, modal,
                                 top_pct=args.top_percentile,
                                 sigma_t=args.smooth_sigma,
                                 bbox_margin=args.bbox_margin)
    np.save(out_dir / "cube_mask_amodal.npy", amodal_mask)

    # Debug overlay video
    overlay = rgb.copy()
    for t in range(T):
        overlay[t][_outline(amodal_mask[t])] = (255, 0, 0)
    _write_video(out_dir / "cube_amodal_overlay.mp4", overlay, args.fps)

    per = amodal_mask.sum(axis=(1, 2))
    modal_per = modal.sum(axis=(1, 2))
    print(f"[ok] wrote {out_dir / 'cube_mask_amodal.npy'}")
    print(f"[ok] wrote {out_dir / 'cube_amodal_overlay.mp4'}")
    print(f"[info] amodal px/frame: median {int(np.median(per))}, "
          f"std {int(np.std(per))} "
          f"(modal was {int(np.median(modal_per))}, "
          f"{100*(np.median(per)-np.median(modal_per))/max(np.median(modal_per),1):+.0f}%)")


if __name__ == "__main__":
    main()
