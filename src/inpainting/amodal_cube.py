"""Stage 8b: video amodal segmentation of the cube with Diffusion-VAS.

The SAM2 mask from segment_cube.py (stage 8a) is *modal* — only the
visible cube pixels, so gripping fingers bite chunks out of it. Diffusion-
VAS (CVPR 2025) is a Stable-Video-Diffusion model that completes the full
*amodal* silhouette, and optionally synthesises the occluded cube RGB
(content completion).

Diffusion-VAS processes exactly 25 frames per call, so a long clip is cut
into 25-frame windows; this script runs each window and stitches the
results back together.

  --overlap K            windows step by 25-K frames and the amodal masks
                         are averaged in the overlap (smoother seams).
                         K=0 (default) → non-overlapping, fastest.
  --no_content_completion  skip the amodal-RGB pass (mask only, ~2x faster).

This script must run in the `diffusion_vas` conda env (NOT `inpaint`):
    conda run -n diffusion_vas python amodal_cube.py --processed_demo <pd>

Inputs:
    <pd>/video_L.mp4
    <pd>/cube_layer/cube_mask_raw.npy        (T,H,W) bool — SAM2 modal mask

Outputs:
    <pd>/cube_layer/cube_mask_amodal.npy     (T,H,W) bool — amodal silhouette
    <pd>/cube_layer/cube_amodal_overlay.mp4  raw + amodal outline (debug)
    <pd>/cube_layer/cube_amodal_rgb.npy      (T,H,W,3) uint8 — amodal cube RGB
    <pd>/cube_layer/cube_amodal_rgb.mp4      amodal RGB cut to mask (debug)
        (the last two only when content completion is enabled)
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from _paths import (DEPTH_ANYTHING_CKPT_DIR, DIFFUSION_VAS_MASK_CKPT,
                     DIFFUSION_VAS_RGB_CKPT, ensure_diffusion_vas_importable)

ensure_diffusion_vas_importable()
import demo as dvas  # noqa: E402  third_party/diffusion-vas/demo.py — reuse its model loaders

NUM_FRAMES = 25  # Diffusion-VAS / SVD backbone is fixed to 25-frame clips


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


def _masks_to_tensor(window: np.ndarray, res) -> torch.Tensor:
    """(N,H,W) bool → [1,N,3,h,w] in [-1,1], mirroring demo.load_and_transform_masks."""
    tf = transforms.Compose([
        transforms.Resize(res),
        transforms.ToTensor(),
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
    """(N,H,W,3) uint8 → [1,N,3,h,w] in [-1,1], mirroring demo.load_and_transform_rgbs."""
    tf = transforms.Compose([
        transforms.Resize(res),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    out = [tf(Image.fromarray(f).convert("RGB")) for f in window]
    return torch.stack(out).unsqueeze(0)


def _infer(pipeline, cond_a, cond_b, res, generator):
    """One Diffusion-VAS 25-frame pass. Returns a list of 25 (h,w,3) uint8 arrays."""
    frames = pipeline(
        cond_a, cond_b,
        height=res[0], width=res[1], num_frames=NUM_FRAMES,
        decode_chunk_size=8, motion_bucket_id=127, fps=8,
        noise_aug_strength=0.02, min_guidance_scale=1.5, max_guidance_scale=1.5,
        generator=generator,
    ).frames[0]
    return [np.array(im).astype(np.uint8) for im in frames]


def _outline(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    m = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(m, k, iterations=thickness).astype(bool) ^ \
        cv2.erode(m, k, iterations=thickness).astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--overlap", type=int, default=0,
                    help="frames shared between consecutive 25-frame windows "
                         "(0 = non-overlapping; 4-6 smooths window seams)")
    ap.add_argument("--no_content_completion", action="store_true",
                    help="skip the amodal-RGB pass; emit the amodal mask only")
    ap.add_argument("--pred_res", default="256,512",
                    help="Diffusion-VAS inference resolution 'H,W' "
                         "(512,1024 is sharper but much slower)")
    ap.add_argument("--depth_encoder", default="vitl", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max_windows", type=int, default=0,
                    help="process only the first N windows (smoke test; "
                         "0 = all). Frames past the covered range stay empty.")
    args = ap.parse_args()

    if not (0 <= args.overlap < NUM_FRAMES):
        sys.exit(f"--overlap must be in [0, {NUM_FRAMES})")
    res = tuple(int(x) for x in args.pred_res.split(","))
    do_cc = not args.no_content_completion

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
    print(f"[info] T={T}, {W}x{H}, res={res}, content_completion={do_cc}")
    print(f"[info] {len(starts)} windows of {NUM_FRAMES} frames "
          f"(stride={stride}, overlap={args.overlap})")

    depth_ckpt = f"{DEPTH_ANYTHING_CKPT_DIR}/depth_anything_v2_{args.depth_encoder}.pth"
    print("[info] loading Diffusion-VAS pipelines (~7.6 GB each)...", flush=True)
    pipe_mask = dvas.init_amodal_segmentation_model(DIFFUSION_VAS_MASK_CKPT)
    depth_model = dvas.init_depth_model(depth_ckpt, args.depth_encoder)
    pipe_rgb = dvas.init_rgb_model(DIFFUSION_VAS_RGB_CKPT) if do_cc else None
    # upstream init disables the progress bar — re-enable so the per-window
    # 25-step denoise visibly ticks (otherwise a live run looks hung).
    pipe_mask.set_progress_bar_config(disable=False)
    if pipe_rgb is not None:
        pipe_rgb.set_progress_bar_config(disable=False)

    accum_mask = np.zeros((T, H, W), dtype=np.float32)
    accum_rgb = np.zeros((T, H, W, 3), dtype=np.float32) if do_cc else None
    count = np.zeros(T, dtype=np.int32)

    for w, s in enumerate(starts):
        idx = list(range(s, min(s + NUM_FRAMES, T)))
        widx = idx + [idx[-1]] * (NUM_FRAMES - len(idx))  # pad short tail window
        modal_t = _masks_to_tensor(modal[widx], res)
        rgb_t = _rgbs_to_tensor(rgb[widx], res)
        depth_t = dvas.rgb_to_depth(rgb_t, depth_model)

        print(f"[window {w+1}/{len(starts)}] frames [{idx[0]},{idx[-1]}]  "
              f"amodal segmentation (25 SVD steps)...", flush=True)
        amodal = _infer(pipe_mask, modal_t, depth_t, res,
                        torch.manual_seed(args.seed))
        amodal = (np.stack(amodal).sum(axis=-1) > 600).astype(np.uint8)  # (25,h,w)
        modal_union = (modal_t[0, :, 0].cpu().numpy() > 0).astype(np.uint8)
        amodal = np.logical_or(amodal, modal_union)
        amodal_full = np.stack([cv2.resize(a.astype(np.uint8), (W, H),
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                                for a in amodal])

        if do_cc:
            amodal_cond = torch.from_numpy(np.where(amodal == 0, -1, 1)).float()
            amodal_cond = amodal_cond.unsqueeze(0).unsqueeze(2).repeat(1, 1, 3, 1, 1)
            modal_obj = (modal_t > 0).float()
            modal_rgb = (rgb_t + 1) / 2 * modal_obj + (1 - modal_obj)  # cube on white bg
            modal_rgb = modal_rgb * 2 - 1
            print(f"[window {w+1}/{len(starts)}] content completion "
                  f"(25 SVD steps)...", flush=True)
            arGB = _infer(pipe_rgb, modal_rgb, amodal_cond, res,
                          torch.manual_seed(args.seed))
            arGB_full = np.stack([cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
                                  for a in arGB])

        for j, fi in enumerate(idx):  # real frames only — drop padding repeats
            accum_mask[fi] += amodal_full[j]
            count[fi] += 1
            if do_cc:
                accum_rgb[fi] += arGB_full[j]

    cov = np.maximum(count, 1)  # uncovered frames (smoke test) → divide by 1, stay empty
    amodal_mask = (accum_mask / cov[:, None, None]) >= 0.5
    out_dir = pd / "cube_layer"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cube_mask_amodal.npy", amodal_mask)

    overlay = rgb.copy()
    for t in range(T):
        overlay[t][_outline(amodal_mask[t])] = (255, 0, 0)
    _write_video(out_dir / "cube_amodal_overlay.mp4", overlay, args.fps)

    per = amodal_mask.sum(axis=(1, 2))
    modal_per = modal.sum(axis=(1, 2))
    print(f"[ok] wrote {out_dir / 'cube_mask_amodal.npy'}")
    print(f"[ok] wrote {out_dir / 'cube_amodal_overlay.mp4'}")
    print(f"[info] amodal cube px/frame: median {int(np.median(per))} "
          f"(modal was {int(np.median(modal_per))}, "
          f"{100*(np.median(per)-np.median(modal_per))/max(np.median(modal_per),1):+.0f}%)")

    if do_cc:
        amodal_rgb = (accum_rgb / cov[:, None, None, None]).astype(np.uint8)
        np.save(out_dir / "cube_amodal_rgb.npy", amodal_rgb)
        cropped = np.zeros_like(rgb)
        for t in range(T):
            cropped[t][amodal_mask[t]] = amodal_rgb[t][amodal_mask[t]]
        _write_video(out_dir / "cube_amodal_rgb.mp4", cropped, args.fps)
        print(f"[ok] wrote {out_dir / 'cube_amodal_rgb.npy'}")
        print(f"[ok] wrote {out_dir / 'cube_amodal_rgb.mp4'}")


if __name__ == "__main__":
    main()
