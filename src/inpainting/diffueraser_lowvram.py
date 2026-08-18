"""DiffuEraser on a 12 GB card, with the ProPainter prior supplied or run alone.

``run_diffueraser.py`` loads the diffusion model first and then runs the
ProPainter prior, so RAFT computes optical flow for the whole clip with ~6 GB
already taken. On this card that is an out-of-memory error at 574 frames even at
640 px.

Two changes. The prior runs by itself and its model is released before the
diffusion model is built, so each stage gets the whole card. And ``--priori``
accepts a ProPainter result that already exists -- ours does, from the pipeline's
own background stage -- which skips the recompute and, more usefully, makes the
comparison exact: the same prior goes in, so any difference in the output is the
diffusion refinement and nothing else.
"""
import argparse
import os
import time

import torch

from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--input_mask", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--priori", default=None,
                        help="Existing ProPainter result to use as the prior. "
                             "Omit to compute it here.")
    parser.add_argument("--video_length", type=int, default=20,
                        help="Seconds of video to process.")
    parser.add_argument("--mask_dilation_iter", type=int, default=8)
    parser.add_argument("--max_img_size", type=int, default=640)
    parser.add_argument("--ref_stride", type=int, default=10)
    parser.add_argument("--neighbor_length", type=int, default=10)
    parser.add_argument("--subvideo_length", type=int, default=50)
    parser.add_argument("--base_model_path", default="weights/stable-diffusion-v1-5")
    parser.add_argument("--vae_path", default="weights/sd-vae-ft-mse")
    parser.add_argument("--diffueraser_path", default="weights/diffuEraser")
    parser.add_argument("--propainter_model_dir", default="weights/propainter")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    priori_path = os.path.join(args.save_path, "priori.mp4")
    output_path = os.path.join(args.save_path, "diffueraser_result.mp4")
    device = get_device()
    start = time.time()

    if args.priori:
        import shutil
        shutil.copyfile(args.priori, priori_path)
        print(f"[prior] reusing {args.priori}")
    else:
        propainter = Propainter(args.propainter_model_dir, device=device)
        propainter.forward(args.input_video, args.input_mask, priori_path,
                           video_length=args.video_length,
                           ref_stride=args.ref_stride,
                           neighbor_length=args.neighbor_length,
                           subvideo_length=args.subvideo_length,
                           mask_dilation=args.mask_dilation_iter)
        del propainter
        torch.cuda.empty_cache()
        print("[prior] computed, ProPainter released")

    video_inpainting_sd = DiffuEraser(device, args.base_model_path, args.vae_path,
                                      args.diffueraser_path, ckpt="2-Step")
    video_inpainting_sd.forward(args.input_video, args.input_mask, priori_path,
                                output_path, max_img_size=args.max_img_size,
                                video_length=args.video_length,
                                mask_dilation_iter=args.mask_dilation_iter,
                                guidance_scale=None)
    print(f"[ok] {output_path}  ({time.time() - start:.1f} s)")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
