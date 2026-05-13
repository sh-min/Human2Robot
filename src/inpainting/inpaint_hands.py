"""E2FGVI video inpainting over SAM2 arm masks.

Minimal port of phantom's `HandInpaintProcessor`, keeping only the batched
inference + temporal context loop. Drops the BaseProcessor framework and
the epic-mode resolution path.

Inputs:
    <processed_demo>/video_rgb_imgs.mkv               (ffv1, produced by inject_hawor_data.py)
    <processed_demo>/segmentation_processor/masks_arm.npy   (bool, produced by segment_arms.py)

Output:
    <processed_demo>/inpaint_processor/video_human_inpaint.mkv  (ffv1)

Usage:
    python inpaint_hands.py --processed_demo /result/cam0_inpaint/cam0/0
"""
import argparse
import gc
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapy as media
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from _paths import E2FGVI_CHECKPOINT, ensure_e2fgvi_importable

ensure_e2fgvi_importable()
from model.e2fgvi_hq import InpaintGenerator  # noqa: E402
from core.utils import to_tensors  # noqa: E402

# E2FGVI HQ defaults
REF_LENGTH = 20
NUM_REF = -1
NEIGHBOR_STRIDE = 5
BATCH_SIZE = 10
MOD_H, MOD_W = 60, 108


def _read_frames(video_path: Path) -> List[Image.Image]:
    return [Image.fromarray(f) for f in media.read_video(str(video_path))]


def _read_masks(mask_path: Path, size: Tuple[int, int]) -> List[Image.Image]:
    """Load arm masks, resize, and dilate (matches phantom's read_mask)."""
    out = []
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for frame in np.load(mask_path, allow_pickle=True):
        m = Image.fromarray(frame).resize(size, Image.NEAREST)
        binary = (np.array(m.convert("L")) > 0).astype(np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=4)
        out.append(Image.fromarray(dilated * 255))
    return out


def _create_binary_masks(masks: List[Image.Image], start: int, end: int) -> List[np.ndarray]:
    """Per-frame binary mask in (H, W, 1) for compositing."""
    out = []
    for m in masks[start:end]:
        arr = np.array(m)
        bm = (arr != 0).astype(np.uint8)
        out.append(bm[..., None])
    return out


def _pad_images(img_tensor: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Reflect-pad to nearest multiple of (60, 108) so E2FGVI accepts the input."""
    h_pad = (MOD_H - h % MOD_H) % MOD_H
    w_pad = (MOD_W - w % MOD_W) % MOD_W
    img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [3])], 3)[:, :, :, :h + h_pad, :]
    return torch.cat([img_tensor, torch.flip(img_tensor, [4])], 4)[:, :, :, :, :w + w_pad]


def _get_ref_index(neighbor_ids: List[int], length: int) -> List[int]:
    """Reference frame indices for temporal consistency (matches phantom)."""
    return [i for i in range(0, length, REF_LENGTH) if i not in neighbor_ids]


def _clear_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _inpaint_video(
    model,
    device: torch.device,
    frames: List[Image.Image],
    masks: List[Image.Image],
) -> List[np.ndarray]:
    """Run E2FGVI in temporal batches and return the composited frames."""
    video_length = len(frames)
    w, h = frames[0].size
    comp_frames: List[Optional[np.ndarray]] = [None] * video_length

    for batch_start in tqdm(range(0, video_length, BATCH_SIZE * NEIGHBOR_STRIDE),
                            desc="Processing batches"):
        batch_end = min(batch_start + BATCH_SIZE * NEIGHBOR_STRIDE + NEIGHBOR_STRIDE,
                        video_length)
        batch_frames = frames[batch_start:batch_end]
        batch_imgs = to_tensors()(batch_frames).unsqueeze(0).to(device) * 2 - 1
        batch_masks = to_tensors()(masks[batch_start:batch_end]).unsqueeze(0).to(device)
        binary_masks = _create_binary_masks(masks, batch_start, batch_end)

        stride = NEIGHBOR_STRIDE if (batch_start + BATCH_SIZE * NEIGHBOR_STRIDE
                                     < video_length) else 1
        stride = max(1, stride)

        for frame_idx in range(batch_start, batch_end, stride):
            neighbor_ids = list(range(
                max(batch_start, frame_idx - NEIGHBOR_STRIDE),
                min(batch_end, frame_idx + NEIGHBOR_STRIDE + 1)
            ))
            if not neighbor_ids:
                continue
            batch_neighbor_ids = [i - batch_start for i in neighbor_ids]
            ref_ids = _get_ref_index(neighbor_ids, video_length)
            batch_ref_ids = [i - batch_start for i in ref_ids
                             if batch_start <= i < batch_end]

            with torch.no_grad():
                sel_imgs = batch_imgs[:, batch_neighbor_ids + batch_ref_ids]
                sel_masks = batch_masks[:, batch_neighbor_ids + batch_ref_ids]
                masked = sel_imgs * (1 - sel_masks)
                masked = _pad_images(masked, h, w)
                pred, _ = model(masked, len(batch_neighbor_ids))
                pred = (pred[:, :, :h, :w] + 1) / 2
                pred = (pred.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

                for i, idx in enumerate(neighbor_ids):
                    bm = binary_masks[idx - batch_start]
                    orig = np.array(frames[idx])
                    inp = pred[i] * bm + orig * (1 - bm)
                    if comp_frames[idx] is None:
                        comp_frames[idx] = inp
                    else:
                        comp_frames[idx] = ((comp_frames[idx].astype(np.float32) +
                                             inp.astype(np.float32)) / 2).astype(np.uint8)

            _clear_gpu_memory()

        del batch_imgs, batch_masks
        _clear_gpu_memory()

    # Handle stride-skipped frames by copying the nearest filled frame
    for idx in range(video_length):
        if comp_frames[idx] is None:
            for offset in range(1, video_length):
                lo, hi = idx - offset, idx + offset
                if lo >= 0 and comp_frames[lo] is not None:
                    comp_frames[idx] = comp_frames[lo]; break
                if hi < video_length and comp_frames[hi] is not None:
                    comp_frames[idx] = comp_frames[hi]; break
            else:
                raise RuntimeError(f"frame {idx} never inpainted")

    return [f for f in comp_frames]  # type: ignore[misc]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--output_resolution", type=int, default=None,
                    help="Resize frames to this height before inpainting "
                         "(default: keep original)")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    if not Path(E2FGVI_CHECKPOINT).exists():
        sys.exit(f"E2FGVI checkpoint missing: {E2FGVI_CHECKPOINT}\n"
                 f"Download with: gdown 'https://drive.google.com/uc?id=10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3' "
                 f"-O {E2FGVI_CHECKPOINT}")

    pd = args.processed_demo
    video_path = pd / "video_rgb_imgs.mkv"
    mask_path  = pd / "segmentation_processor" / "masks_arm.npy"
    for p in (video_path, mask_path):
        if not p.exists():
            sys.exit(f"missing input: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InpaintGenerator().to(device)
    model.load_state_dict(torch.load(E2FGVI_CHECKPOINT, map_location=device))
    model.eval()
    print(f"[info] E2FGVI loaded on {device}")

    frames = _read_frames(video_path)
    w0, h0 = frames[0].size
    if args.output_resolution and h0 != args.output_resolution:
        new_w = int(w0 / h0 * args.output_resolution)
        size = (new_w, args.output_resolution)
        frames = [f.resize(size) for f in frames]
        print(f"[info] resized to {size}")
    size = frames[0].size  # (w, h)
    masks = _read_masks(mask_path, size)
    print(f"[info] T={len(frames)}, frame size {size[0]}x{size[1]}")

    comp = _inpaint_video(model, device, frames, masks)

    out_dir = pd / "inpaint_processor"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "video_human_inpaint.mkv"
    media.write_video(str(out), comp, fps=args.fps, codec="ffv1")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
