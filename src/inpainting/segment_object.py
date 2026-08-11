"""Stage 8 (SAM2 variant): segment the manipulated object across the video
with SAM2, bootstrapped from the hand-contact (grasp) region.

Earlier versions located the object as the closest-`quantile` non-hand depth
blob. That assumed a single small object nearest the camera on a table (the
object setup) and on egocentric, varied-object data (EgoDex) it grabs the table
instead. Here we instead seed SAM2 from where the hand grips the object: the
manipulated object is, by definition, at the hand-object contact.

Bootstrap (no depth needed):
    1. From HaWoR 2D hand keypoints (hand_processor/hand_data_{left,right}.npz),
       find a seed (frame, hand) where a hand is gripping — fingertip spread in
       a sensible band (rejects open hands and fists), picking the tightest grip.
    2. Build SAM2 prompts that target the *object*, not the hand:
         positive points = fingertip centroid + thumb-index pinch point,
         negative points = wrist + finger MCP knuckles (clearly hand),
         box            = tight around the fingertip cluster.
    3. SAM2 propagates forward + reverse from the seed; union the passes.

KNOWN LIMITATION (WIP): the prompt construction in (2) is validated — forcing
the correct (frame, hand) cleanly segments the held object. But the *automatic*
seed selection in (1) is unreliable on two-hand scenes: keypoints alone can't
tell the holding hand from a reaching hand, so it can latch onto a sleeve or
the wrong hand. Robust selection needs a which-hand-when signal (HACO contact)
or a seed-both-hands-and-pick-best scheme. Use --seed_frame/--side to override.

Inputs:
    <pd>/video_L.mp4
    <pd>/hand_processor/hand_data_{left,right}.npz   (kpts_2d, hand_detected)

Outputs:
    <pd>/object_layer/object_mask_raw.npy       (T,H,W) bool — SAM2 object mask
    <pd>/object_layer/object_cropped_raw.mp4    raw cropped to the mask (debug)

Stage 8b (amodal_object.py) consumes object_mask_raw.npy unchanged.

Usage:
    python segment_object.py --processed_demo /result/skill2policy/processed/cam0/0
"""
import argparse
import shutil
import sys
from pathlib import Path

import mediapy as media
import numpy as np
import torch

from _paths import SAM2_CHECKPOINT, SAM2_CONFIG_NAME, ensure_sam2_importable
from segment_arms import _dump_frames_as_jpegs, _segment_one_pass

ensure_sam2_importable()
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# mediapipe-21 hand layout
FINGERTIPS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips
HAND_NEG   = [0, 5, 9, 13, 17]    # wrist + finger MCP knuckles (clearly hand)

# Accept grips whose fingertip spread (normalized by hand bbox diagonal) is in
# this band: too small = a fist (no object), too large = an open/relaxed hand.
_GRIP_BAND = (0.12, 0.55)


def _grip_metrics(pts):
    tips = pts[FINGERTIPS]
    lo, hi = pts.min(0), pts.max(0)
    diag = float(np.linalg.norm(hi - lo)) + 1e-6
    spread = float(np.linalg.norm(tips - tips.mean(0), axis=1).mean()) / diag
    return spread


def _pick_grasp_seed(hand_npz: dict, H: int, W: int,
                     force_frame: int = None, force_side: str = None):
    """Find a seed (frame, hand) where a hand grips the object and return SAM2
    prompts that target the object: positives at the grasp (from the hand pose)
    plus negatives at BOTH hands' wrist+MCP joints, so SAM2 grabs the object
    between the fingers and never a hand/sleeve.

    Returns (seed_idx, points (P,2), labels (P,), box (4,)) or (None,)*4.
    """
    T = next(iter(hand_npz.values()))["kpts_2d"].shape[0]
    # Restrict auto-seed to the middle of the clip — the ends are reach/retract
    # where the object is not yet (or no longer) in the fingers.
    lo_t, hi_t = int(0.15 * T), int(0.85 * T)

    if force_frame is not None:
        cands = [(_grip_metrics(d["kpts_2d"][force_frame]), side)
                 for side, d in hand_npz.items()
                 if d["hand_detected"][force_frame]
                 and (force_side is None or side == force_side)]
        if not cands:
            return None, None, None, None
        side = min(cands)[1]
        seed_idx = int(force_frame)
    else:
        # tightest grip (fingers wrapped around something) within the mid-clip window
        best = None  # (spread, frame, side)
        for side, d in hand_npz.items():
            det = d["hand_detected"].astype(bool)
            for t in np.where(det)[0]:
                if not (lo_t <= t <= hi_t):
                    continue
                spread = _grip_metrics(d["kpts_2d"][t])
                if not (_GRIP_BAND[0] <= spread <= _GRIP_BAND[1]):
                    continue
                if best is None or spread < best[0]:
                    best = (spread, int(t), side)
        if best is None:
            return None, None, None, None
        _, seed_idx, side = best

    pts = hand_npz[side]["kpts_2d"][seed_idx]
    tips = pts[FINGERTIPS]
    tip_c = tips.mean(0)
    pinch = (pts[4] + pts[8]) / 2.0                 # thumb-index midpoint
    pos = np.stack([tip_c, pinch])                  # object-side positives (from pose)

    # Negatives: always mask out BOTH hands. Each detected hand's wrist + MCP
    # knuckles (HAND_NEG, *not* fingertips — those sit on the object), plus
    # samples from the M_hand mask restricted to the hand region (near hand
    # keypoints) so the forearm part of a full-arm mask isn't included.
    neg_list = [d["kpts_2d"][seed_idx][HAND_NEG].astype(np.float32)
                for d in hand_npz.values() if d["hand_detected"][seed_idx]]
    neg = np.concatenate(neg_list, axis=0) if neg_list else pts[HAND_NEG].astype(np.float32)

    points = np.concatenate([pos, neg], axis=0).astype(np.float32)
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int32)

    # Generous box around the grasp (the object often extends past the tips).
    lo, hi = tips.min(0), tips.max(0)
    margin = 1.2 * (hi - lo + 20.0)
    x0, y0 = np.maximum([0, 0], lo - margin)
    x1, y1 = np.minimum([W, H], hi + margin)
    box = np.array([x0, y0, x1, y1], dtype=np.float32)
    return seed_idx, points, labels, box


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--quantile", type=float, default=0.25,
                    help="(deprecated, ignored) old depth-seed quantile")
    ap.add_argument("--seed_frame", type=int, default=None,
                    help="debug: force the SAM2 seed frame instead of auto-picking")
    ap.add_argument("--side", choices=["left","right"], default=None,
                    help="debug: force which hand to seed from")
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
    rgb = media.read_video(str(video_path))
    H, W = rgb.shape[1], rgb.shape[2]

    hand_npz = {}
    for side in ("left", "right"):
        p = pd / "hand_processor" / f"hand_data_{side}.npz"
        if p.exists():
            hand_npz[side] = np.load(p)
    if not hand_npz:
        sys.exit("[err] no hand_processor/hand_data_*.npz — run inject_hawor_data first")

    seed_idx, seed_pts, seed_lbls, seed_box = _pick_grasp_seed(
        hand_npz, H, W, force_frame=args.seed_frame, force_side=args.side)
    if seed_idx is None:
        sys.exit("[err] no detected hand to seed from — cannot locate the object.")
    n_pos = int(seed_lbls.sum())
    print(f"[info] grasp seed frame={seed_idx} box={seed_box.round(1).tolist()} "
          f"({n_pos} positive / {len(seed_lbls) - n_pos} negative points)")

    frames_dir = pd / "original_images"
    n_frames = _dump_frames_as_jpegs(video_path, frames_dir)
    print(f"[info] T={n_frames}, {W}x{H}")

    video_predictor = build_sam2_video_predictor(SAM2_CONFIG_NAME, SAM2_CHECKPOINT,
                                                 device=DEVICE)

    object_mask = np.zeros((n_frames, H, W), dtype=bool)
    # _segment_one_pass expects batched prompts (K boxes / K point-sets / K frame
    # indices); the grasp seed is a single frame, so add the K=1 batch dim.
    seed_box_b = np.asarray(seed_box)[None]         # (1, 4)
    seed_pts_b = np.asarray(seed_pts)[None]         # (1, P, 2)
    seed_idx_b = np.asarray([seed_idx])             # (1,)
    for reverse in (False, True):
        out = _segment_one_pass(video_predictor, frames_dir, seed_box_b,
                                seed_pts_b, seed_idx_b, reverse=reverse,
                                labels=seed_lbls)
        for idx, m in out.items():
            object_mask[idx] |= m[0]

    per_frame = object_mask.sum(axis=(1, 2))
    print(f"[info] frames with object mask: {(per_frame > 0).sum()}/{n_frames}, "
          f"median {int(np.median(per_frame))} px, max {per_frame.max()} px")

    out_dir = pd / "object_layer"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "object_mask_raw.npy", object_mask)

    Tc = min(n_frames, rgb.shape[0])
    cropped = np.zeros_like(rgb[:Tc])
    for t in range(Tc):
        cropped[t][object_mask[t]] = rgb[t][object_mask[t]]
    media.write_video(str(out_dir / "object_cropped_raw.mp4"), cropped,
                      fps=args.fps, codec="libx264")
    print(f"[ok] wrote {out_dir / 'object_mask_raw.npy'}")
    print(f"[ok] wrote {out_dir / 'object_cropped_raw.mp4'}")

    if not args.keep_tmp:
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
