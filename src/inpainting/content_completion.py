"""Stage 8c: Diffusion-VAS content completion using a pre-computed amodal mask.

Takes the SAM2 modal mask + RGB + the amodal mask from stage 8b, and
generates the amodal cube RGB (fills in occluded cube faces).

This script must run in the `diffusion_vas` conda env (NOT `inpaint`):
    conda run -n diffusion_vas python content_completion.py --processed_demo <pd>

Per-window checkpoints are saved under <pd>/cube_layer/_cc_ckpt/ so the
script can resume after OOM crashes.

Inputs:
    <pd>/video_L.mp4
    <pd>/cube_layer/cube_mask_raw.npy        (T,H,W) bool — SAM2 modal mask
    <pd>/cube_layer/cube_mask_amodal.npy     (T,H,W) bool — amodal mask (or --amodal_mask)

Outputs:
    <pd>/cube_layer/cube_amodal_rgb.npy      (T,H,W,3) uint8 — amodal cube RGB
    <pd>/cube_layer/cube_amodal_rgb.mp4      amodal RGB cut to mask (debug)
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from _paths import DIFFUSION_VAS_RGB_CKPT, ensure_diffusion_vas_importable

ensure_diffusion_vas_importable()
import demo as dvas  # noqa: E402

NUM_FRAMES = 25


def _read_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def _write_video(path, frames, fps):
    H, W = frames.shape[1:3]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def _masks_to_tensor(window, res):
    tf = transforms.Compose([
        transforms.Resize(res), transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    out = []
    for m in window:
        img = Image.fromarray((m.astype(np.uint8) * 255)).convert("L")
        img = img.point(lambda p: 255 if p > 128 else 0)
        out.append(tf(img))
    return torch.stack(out).unsqueeze(0)


def _rgbs_to_tensor(window, res):
    tf = transforms.Compose([
        transforms.Resize(res), transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    out = [tf(Image.fromarray(f).convert("RGB")) for f in window]
    return torch.stack(out).unsqueeze(0)


def _infer(pipeline, cond_a, cond_b, res, generator):
    frames = pipeline(
        cond_a, cond_b,
        height=res[0], width=res[1], num_frames=NUM_FRAMES,
        decode_chunk_size=8, motion_bucket_id=127, fps=8,
        noise_aug_strength=0.02, min_guidance_scale=1.5, max_guidance_scale=1.5,
        generator=generator,
    ).frames[0]
    return [np.array(im).astype(np.uint8) for im in frames]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--amodal_mask", default="cube_layer/cube_mask_amodal.npy",
                    help="path (relative to processed_demo) to the amodal mask")
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--pred_res", default="256,512")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    res = tuple(int(x) for x in args.pred_res.split(","))
    pd = args.processed_demo

    rgb = _read_video(pd / "video_L.mp4")
    modal = np.load(pd / "cube_layer" / "cube_mask_raw.npy").astype(bool)
    amodal = np.load(pd / args.amodal_mask).astype(bool)
    T = min(rgb.shape[0], modal.shape[0], amodal.shape[0])
    H, W = rgb.shape[1:3]

    stride = NUM_FRAMES - args.overlap
    starts = list(range(0, T, stride))

    print(f"[info] T={T}, {W}x{H}, res={res}, {len(starts)} windows", flush=True)
    print("[info] loading content completion pipeline...", flush=True)
    pipe_rgb = dvas.init_rgb_model(DIFFUSION_VAS_RGB_CKPT)
    pipe_rgb.set_progress_bar_config(disable=False)

    # Checkpoint support
    ckpt_dir = pd / "cube_layer" / "_cc_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    accum_rgb = np.zeros((T, H, W, 3), dtype=np.float32)
    count = np.zeros(T, dtype=np.int32)

    resume_from = 0
    for w in range(len(starts)):
        cp = ckpt_dir / f"w{w:03d}.npy"
        if cp.exists():
            d = np.load(cp)
            s = starts[w]
            idx = list(range(s, min(s + NUM_FRAMES, T)))
            for j, fi in enumerate(idx):
                accum_rgb[fi] += d[j]
                count[fi] += 1
            resume_from = w + 1
        else:
            break
    if resume_from > 0:
        print(f"[resume] loaded {resume_from}/{len(starts)} checkpoints",
              flush=True)

    for w, s in enumerate(starts):
        if w < resume_from:
            continue
        idx = list(range(s, min(s + NUM_FRAMES, T)))
        widx = idx + [idx[-1]] * (NUM_FRAMES - len(idx))

        modal_t = _masks_to_tensor(modal[widx], res)
        rgb_t = _rgbs_to_tensor(rgb[widx], res)

        # Amodal condition: amodal mask as [-1, 1]
        amodal_small = np.stack([cv2.resize(amodal[fi].astype(np.uint8),
                                            (res[1], res[0]),
                                            interpolation=cv2.INTER_NEAREST)
                                 for fi in widx])
        amodal_cond = torch.from_numpy(
            np.where(amodal_small == 0, -1, 1)).float()
        amodal_cond = amodal_cond.unsqueeze(0).unsqueeze(2).repeat(1, 1, 3, 1, 1)

        # Modal RGB on white background
        modal_obj = (modal_t > 0).float()
        modal_rgb = (rgb_t + 1) / 2 * modal_obj + (1 - modal_obj)
        modal_rgb = modal_rgb * 2 - 1

        print(f"[window {w+1}/{len(starts)}] frames [{idx[0]},{idx[-1]}] "
              f"content completion (25 SVD steps)...", flush=True)
        arGB = _infer(pipe_rgb, modal_rgb, amodal_cond, res,
                      torch.manual_seed(args.seed))
        arGB_full = np.stack([cv2.resize(a, (W, H),
                                         interpolation=cv2.INTER_LINEAR)
                              for a in arGB])

        np.save(ckpt_dir / f"w{w:03d}.npy", arGB_full[:len(idx)])
        print(f"  [ckpt] saved window {w+1}/{len(starts)}", flush=True)

        for j, fi in enumerate(idx):
            accum_rgb[fi] += arGB_full[j]
            count[fi] += 1

    cov = np.maximum(count, 1)
    amodal_rgb = (accum_rgb / cov[:, None, None, None]).astype(np.uint8)

    out_dir = pd / "cube_layer"
    np.save(out_dir / "cube_amodal_rgb.npy", amodal_rgb)

    cropped = np.zeros_like(rgb[:T])
    for t in range(T):
        cropped[t][amodal[t]] = amodal_rgb[t][amodal[t]]
    _write_video(out_dir / "cube_amodal_rgb.mp4", cropped, args.fps)

    print(f"[ok] wrote {out_dir / 'cube_amodal_rgb.npy'}")
    print(f"[ok] wrote {out_dir / 'cube_amodal_rgb.mp4'}")


if __name__ == "__main__":
    main()
