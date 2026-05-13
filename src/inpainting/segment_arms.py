"""SAM2 arm/hand segmentation given precise HaWoR-derived bbox prompts.

Minimal port of phantom's `ArmSegmentationProcessor`, dropping:
  - Detectron2 bbox refinement (we already inject precise bboxes via
    inject_hawor_data.py, so refinement is dead weight)
  - BaseProcessor / Hydra framework (we use argparse like the rest of the repo)
  - Debug visualization videos (`video_sam_arm`, `video_masks_arm`)
  - Annotation video composition

Inputs (already produced by prepare_demo.py + inject_hawor_data.py):
    <processed_demo>/video_L.mp4
    <processed_demo>/bbox_processor/bbox_data.npz
    <processed_demo>/hand_processor/hand_data_{left,right}.npz

Output:
    <processed_demo>/segmentation_processor/masks_arm.npy   (T, H, W) bool

Per hand: SAM2 propagates from the highest-quality seed frame (max distance to
edge), both forward and reverse. Then the two passes are unioned. Left and
right hand masks are unioned again to produce a single bimanual mask.

Usage:
    python segment_arms.py --processed_demo /result/cam0_inpaint/cam0/0
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import mediapy as media
import numpy as np
import torch
from PIL import Image

from _paths import SAM2_CHECKPOINT, SAM2_CONFIG_NAME, ensure_sam2_importable

ensure_sam2_importable()
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _dump_frames_as_jpegs(video_path: Path, dst_dir: Path) -> int:
    """SAM2's init_state requires a directory of sequentially-named JPEGs."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in dst_dir.iterdir() if p.suffix == ".jpg")
    frames = media.read_video(str(video_path))
    if len(existing) == len(frames):
        return len(frames)
    for p in existing:
        p.unlink()
    for i, f in enumerate(frames):
        Image.fromarray(f).save(dst_dir / f"{i:05d}.jpg", quality=95)
    return len(frames)


def _segment_one_pass(
    video_predictor,
    video_dir: Path,
    bbox: np.ndarray,
    kpts_2d: np.ndarray,    # (21, 2) for the seed frame
    seed_frame_idx: int,
    reverse: bool,
) -> dict:
    """One SAM2 propagation pass (forward or backward in time) from a single
    seed frame. Returns {frame_idx: mask (1, H, W) bool}. Always reads
    forward-ordered JPEGs; `reverse` flips propagation direction in SAM2."""
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        state = video_predictor.init_state(video_path=str(video_dir),
                                           offload_video_to_cpu=True)
        video_predictor.reset_state(state)
        video_predictor.add_new_points_or_box(
            state,
            frame_idx=int(seed_frame_idx),
            obj_id=0,
            box=np.asarray(bbox, dtype=np.float32),
            points=np.asarray(kpts_2d, dtype=np.float32),
            labels=np.ones(len(kpts_2d), dtype=np.int32),
        )
        segments = {}
        for out_frame_idx, _, out_mask_logits in video_predictor.propagate_in_video(
            state, reverse=reverse,
        ):
            segments[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()
    torch.cuda.empty_cache()
    return segments


def _segment_hand(
    video_predictor,
    frames_dir: Path,
    n_frames: int,
    bboxes: np.ndarray,
    bbox_min_dist: np.ndarray,
    hand_detected: np.ndarray,
    kpts_2d: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Run forward+reverse SAM2 propagation for one hand. Returns (T,H,W) bool."""
    masks = np.zeros((n_frames, img_h, img_w), dtype=bool)
    if not hand_detected.any() or bbox_min_dist.max() == 0:
        return masks

    seed_idx = int(np.argmax(bbox_min_dist))
    seed_bbox = bboxes[seed_idx]
    seed_kpts = kpts_2d[seed_idx]
    print(f"  seed frame={seed_idx} bbox={seed_bbox.round(1).tolist()}")

    for reverse in (False, True):
        out = _segment_one_pass(video_predictor, frames_dir,
                                seed_bbox, seed_kpts, seed_idx, reverse=reverse)
        for idx, m in out.items():
            masks[idx] |= m[0]
    return masks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--keep_tmp", action="store_true",
                    help="Keep original_images/ and original_images_reverse/ for debugging")
    args = ap.parse_args()

    if not Path(SAM2_CHECKPOINT).exists():
        sys.exit(f"SAM2 checkpoint missing: {SAM2_CHECKPOINT}\n"
                 f"Download with: wget -P {Path(SAM2_CHECKPOINT).parent} "
                 f"https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt")

    pd = args.processed_demo
    video_path = pd / "video_L.mp4"
    bbox = np.load(pd / "bbox_processor" / "bbox_data.npz")
    hd_l = np.load(pd / "hand_processor" / "hand_data_left.npz")
    hd_r = np.load(pd / "hand_processor" / "hand_data_right.npz")

    print(f"[info] dumping JPEGs for SAM2 init_state...")
    frames_dir = pd / "original_images"
    n_frames = _dump_frames_as_jpegs(video_path, frames_dir)
    sample = np.array(Image.open(frames_dir / "00000.jpg"))
    img_h, img_w = sample.shape[:2]
    print(f"[info] T={n_frames}, {img_w}x{img_h}")

    video_predictor = build_sam2_video_predictor(SAM2_CONFIG_NAME, SAM2_CHECKPOINT, device=DEVICE)

    print("[left]  SAM2 propagation")
    left_masks = _segment_hand(
        video_predictor, frames_dir, n_frames,
        bbox["left_bboxes"], bbox["left_bbox_min_dist_to_edge"],
        bbox["left_hand_detected"], hd_l["kpts_2d"][:n_frames],
        img_h, img_w,
    )
    print("[right] SAM2 propagation")
    right_masks = _segment_hand(
        video_predictor, frames_dir, n_frames,
        bbox["right_bboxes"], bbox["right_bbox_min_dist_to_edge"],
        bbox["right_hand_detected"], hd_r["kpts_2d"][:n_frames],
        img_h, img_w,
    )

    masks = left_masks | right_masks
    per_frame = masks.sum(axis=(1, 2))
    print(f"[info] frames with mask: {(per_frame > 0).sum()}/{n_frames}, "
          f"avg {per_frame.mean():.0f} px, max {per_frame.max()} px")

    out_dir = pd / "segmentation_processor"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "masks_arm.npy", masks)
    print(f"[ok] wrote {out_dir / 'masks_arm.npy'}")

    if not args.keep_tmp:
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
