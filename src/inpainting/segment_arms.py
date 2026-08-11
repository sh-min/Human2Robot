"""SAM2 full hand/arm segmentation from HaWoR hand prompts.

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

Per hand, the HaWoR wrist-to-palm direction is extrapolated onto the forearm.
Those extra positive points keep SAM2 on the connected human limb rather than
returning only the hand inside the original detector box. SAM2 is propagated
forward and backward and the component connected to the hand prompts is kept.
Left and right masks are finally unioned into a bimanual hand+arm mask, then
denoised (`_smooth_masks`, default on): detached specks removed, holes closed,
per-pixel temporal flicker suppressed by a majority vote, and jagged edges
rounded. Disable with --no_smooth.

Usage:
    python segment_arms.py --processed_demo /result/cam0_inpaint/cam0/0
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import mediapy as media
import cv2
import numpy as np
import torch
from PIL import Image

from _paths import SAM2_CHECKPOINT, SAM2_CONFIG_NAME, ensure_sam2_importable

ensure_sam2_importable()
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PALM_IDXS = (5, 9, 13, 17)


def _augment_with_forearm_points(kpts_2d: np.ndarray,
                                 img_h: int,
                                 img_w: int,
                                 scales: tuple[float, ...]) -> np.ndarray:
    """Append positive prompts extending from palm through wrist into forearm.

    MANO joint 0 is the wrist, while the four MCP joints provide a stable palm
    centre.  Extrapolating wrist - palm stays on the forearm even when the arm
    bends farther away from the hand.
    """
    out = []
    for points in np.asarray(kpts_2d, dtype=np.float32):
        wrist = points[0]
        palm = points[list(PALM_IDXS)].mean(axis=0)
        direction = wrist - palm
        length = float(np.linalg.norm(direction))
        if not np.isfinite(length) or length < 1.0:
            extra = np.repeat(wrist[None], len(scales), axis=0)
        else:
            extra = np.stack([
                wrist + direction * scale
                for scale in scales
            ])
        augmented = np.concatenate([points, extra], axis=0)
        augmented[:, 0] = np.clip(augmented[:, 0], 0, img_w - 1)
        augmented[:, 1] = np.clip(augmented[:, 1], 0, img_h - 1)
        out.append(augmented.astype(np.float32))
    return np.stack(out)


FINGERTIP_IDXS = (4, 8, 12, 16, 20)   # MANO thumb..pinky tips


def _object_negative_points(kpts_2d: np.ndarray,
                            img_h: int,
                            img_w: int,
                            scales: tuple[float, ...] = (0.6, 1.2)) -> np.ndarray:
    """Negative SAM2 prompts for a grasped object.

    A held object protrudes past the fingers, so we extrapolate outward from the
    palm through each fingertip and drop negative points beyond the tips. When
    the hand grips something those land on the object and SAM2 carves it out of
    the hand mask; when the hand is empty they fall on background the mask does
    not reach anyway (harmless). Fixed count per frame so it stacks cleanly.
    """
    out = []
    for points in np.asarray(kpts_2d, dtype=np.float32):
        palm = points[list(PALM_IDXS)].mean(axis=0)
        negs = []
        for ti in FINGERTIP_IDXS:
            tip = points[ti]
            d = tip - palm
            if not np.isfinite(d).all() or np.linalg.norm(d) < 1.0:
                d = np.zeros(2, np.float32)   # degenerate -> keep the tip itself
            for s in scales:
                negs.append(tip + d * s)
        negs = np.asarray(negs, np.float32)
        negs[:, 0] = np.clip(negs[:, 0], 0, img_w - 1)
        negs[:, 1] = np.clip(negs[:, 1], 0, img_h - 1)
        out.append(negs)
    return np.stack(out)


def _expand_boxes_to_prompts(bboxes: np.ndarray,
                             points: np.ndarray,
                             img_h: int,
                             img_w: int) -> np.ndarray:
    """Expand each hand box just enough to include its forearm prompts."""
    expanded = np.asarray(bboxes, dtype=np.float32).copy()
    for i, pts in enumerate(points):
        x1, y1, x2, y2 = expanded[i]
        hand_size = max(float(x2 - x1), float(y2 - y1), 1.0)
        margin = 0.12 * hand_size
        expanded[i] = [
            max(0.0, min(x1, float(pts[:, 0].min())) - margin),
            max(0.0, min(y1, float(pts[:, 1].min())) - margin),
            min(float(img_w - 1), max(x2, float(pts[:, 0].max())) + margin),
            min(float(img_h - 1), max(y2, float(pts[:, 1].max())) + margin),
        ]
    return expanded


def _component_at_prompts(mask: np.ndarray,
                          points: np.ndarray,
                          previous: np.ndarray | None = None) -> np.ndarray:
    """Keep the SAM component attached to the hand/forearm prompts.

    This removes disconnected table/object leaks without imposing the old hand
    bounding-box crop, which was the reason the human forearm disappeared.
    """
    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary.astype(bool)

    scores = np.zeros(count, dtype=np.float64)
    h, w = binary.shape
    for x, y in np.asarray(points):
        xi = int(np.clip(round(float(x)), 0, w - 1))
        yi = int(np.clip(round(float(y)), 0, h - 1))
        label = labels[yi, xi]
        if label > 0:
            scores[label] += 1_000_000.0
    if previous is not None and previous.any():
        overlap_labels, overlap_counts = np.unique(labels[previous], return_counts=True)
        for label, overlap in zip(overlap_labels, overlap_counts):
            if label > 0:
                scores[label] += float(overlap) * 100.0
    scores[1:] += stats[1:, cv2.CC_STAT_AREA]
    keep = int(np.argmax(scores[1:]) + 1)
    selected = (labels == keep).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, kernel)
    return selected.astype(bool)


def _repair_temporal_mask_outliers(masks: np.ndarray,
                                   ratio_threshold: float = 2.0,
                                   radius: int = 3) -> tuple[np.ndarray, list[int]]:
    """Replace isolated SAM leaks with adjacent, temporally stable masks.

    A connected SAM component can occasionally jump from the arm onto the
    similarly coloured table for exactly one frame.  Genuine arm motion is
    gradual in this dataset, whereas those leaks make the mask area more than
    twice the local median.  Using the union of the nearest stable neighbours
    preserves the complete hand/arm silhouette without retaining the table.
    """
    repaired = np.asarray(masks, dtype=bool).copy()
    if len(repaired) < 3:
        return repaired, []

    areas = repaired.reshape(len(repaired), -1).sum(axis=1).astype(np.float64)
    local_median = np.array([
        np.median(areas[max(0, idx - radius):min(len(areas), idx + radius + 1)])
        for idx in range(len(areas))
    ])
    outlier = areas > ratio_threshold * np.maximum(local_median, 1.0)
    repaired_indices: list[int] = []

    for idx in np.flatnonzero(outlier):
        previous = next(
            (j for j in range(idx - 1, max(-1, idx - radius - 1), -1)
             if not outlier[j]),
            None,
        )
        following = next(
            (j for j in range(idx + 1, min(len(repaired), idx + radius + 1))
             if not outlier[j]),
            None,
        )
        if previous is None and following is None:
            continue
        if previous is None:
            replacement = repaired[following]
        elif following is None:
            replacement = repaired[previous]
        else:
            replacement = repaired[previous] | repaired[following]
        repaired[idx] = replacement
        repaired_indices.append(int(idx))

    return repaired, repaired_indices


def _smooth_masks(masks: np.ndarray,
                  min_area_frac: float = 2e-4,
                  close_ksize: int = 5,
                  temporal_win: int = 5,
                  boundary_sigma: float = 1.5) -> tuple[np.ndarray, int]:
    """Denoise the SAM arm/hand masks so the silhouette is clean and stable.

    SAM masks come out with three kinds of noise: small detached fragments
    (specks near the fingers), jagged/holey boundaries, and per-pixel flicker
    of the edge from frame to frame. This applies, in order:

      1. speck removal  — drop connected components smaller than *min_area_frac*
                          of the frame (keeps both arms; kills scattered bits).
      2. morphological close — fill pin-holes and concave notches (*close_ksize*).
      3. temporal majority — a pixel stays ON only if ON in a strict majority of
                          a centred *temporal_win*-frame window; removes edge
                          flicker while following genuine motion. Memory-safe
                          sliding window (one (H,W) accumulator, no float volume).
      4. boundary round  — Gaussian-blur the binary edge and re-threshold at 0.5
                          (*boundary_sigma*) to take the staircase off the edge.

    Returns (smoothed_masks, n_specks_removed). Any stage is skipped when its
    parameter is <= its no-op value (frac<=0, ksize<=1, win<=1, sigma<=0).
    """
    m = np.asarray(masks, dtype=bool)
    T, H, W = m.shape
    min_area = int(min_area_frac * H * W) if min_area_frac > 0 else 0
    ck = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
          if close_ksize > 1 else None)

    # 1-2) per-frame: speck removal + close
    out = np.empty_like(m)
    n_specks = 0
    for t in range(T):
        b = m[t].astype(np.uint8)
        if min_area > 0:
            cnt, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
            keep = np.zeros_like(b)
            for i in range(1, cnt):
                if st[i, cv2.CC_STAT_AREA] >= min_area:
                    keep[lab == i] = 1
                else:
                    n_specks += 1
            b = keep
        if ck is not None:
            b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, ck)
        out[t] = b.astype(bool)

    # 3) temporal majority vote over a centred window (sliding running sum)
    if temporal_win > 1:
        r = temporal_win // 2
        o8 = out.astype(np.uint8)
        acc = np.zeros((H, W), dtype=np.int32)
        for i in range(min(r + 1, T)):
            acc += o8[i]
        tmp = np.empty_like(out)
        for t in range(T):
            if t > 0:
                add, rem = t + r, t - r - 1
                if add < T:
                    acc += o8[add]
                if rem >= 0:
                    acc -= o8[rem]
            win = min(t + r, T - 1) - max(0, t - r) + 1
            tmp[t] = (acc * 2 > win)
        out = tmp

    # 4) per-frame boundary rounding
    if boundary_sigma > 0:
        for t in range(T):
            fb = cv2.GaussianBlur(out[t].astype(np.float32), (0, 0), boundary_sigma)
            out[t] = fb >= 0.5

    return out, n_specks


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
    prompt_bboxes: np.ndarray,
    prompt_kpts_2d: np.ndarray,  # (K, P, 2) point prompts
    prompt_frame_indices: np.ndarray,
    reverse: bool,
    labels: np.ndarray = None,   # (P,) 1=positive, 0=negative; default all positive
) -> dict:
    """One SAM2 propagation pass (forward or backward in time) from a single
    seed frame. Returns {frame_idx: mask (1, H, W) bool}. Always reads
    forward-ordered JPEGs; `reverse` flips propagation direction in SAM2."""
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        state = video_predictor.init_state(video_path=str(video_dir),
                                           offload_video_to_cpu=True)
        video_predictor.reset_state(state)
        for bbox, kpts_2d, frame_idx in zip(
            prompt_bboxes, prompt_kpts_2d, prompt_frame_indices
        ):
            frame_labels = (
                np.ones(len(kpts_2d), dtype=np.int32)
                if labels is None else np.asarray(labels, dtype=np.int32)
            )
            video_predictor.add_new_points_or_box(
                state,
                frame_idx=int(frame_idx),
                obj_id=0,
                box=np.asarray(bbox, dtype=np.float32),
                points=np.asarray(kpts_2d, dtype=np.float32),
                labels=frame_labels,
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
    forearm_scales: tuple[float, ...],
    seed_stride: int = 20,
    reanchor_rounds: int = 2,
    collapse_frac: float = 0.15,
    use_negatives: bool = True,
    neg_scales: tuple[float, ...] = (0.6, 1.2),
) -> np.ndarray:
    """Run forward+reverse SAM2 propagation for one hand. Returns (T,H,W) bool.

    Seeds every `seed_stride`-th detected frame, then re-anchors: any detected
    frame whose propagated mask collapsed to < collapse_frac of the median area
    (drift / lost track) is re-prompted with its own hand keypoints and
    re-propagated, recovering frames the single-direction track dropped.
    """
    masks = np.zeros((n_frames, img_h, img_w), dtype=bool)
    if not hand_detected.any() or bbox_min_dist.max() == 0:
        return masks

    seed_idx = int(np.argmax(bbox_min_dist))
    valid_indices = np.flatnonzero(hand_detected)
    prompt_indices = np.unique(np.concatenate([
        valid_indices[::max(1, seed_stride)],
        np.array([seed_idx, valid_indices[-1]], dtype=np.int64),
    ]))
    # Positive prompts (hand keypoints + forearm). Used for the seed box AND the
    # connected-component keep — negatives must NOT influence either.
    arm_points = _augment_with_forearm_points(
        kpts_2d, img_h, img_w, forearm_scales,
    )
    arm_boxes = _expand_boxes_to_prompts(bboxes, arm_points, img_h, img_w)
    # Add negative prompts on the grasped object so SAM2 returns hand+arm only.
    if use_negatives:
        neg_points = _object_negative_points(kpts_2d, img_h, img_w, neg_scales)
        prompt_points = np.concatenate([arm_points, neg_points], axis=1)
        prompt_labels = np.concatenate([
            np.ones(arm_points.shape[1], np.int32),
            np.zeros(neg_points.shape[1], np.int32)])
    else:
        prompt_points = arm_points
        prompt_labels = None
    print(f"  seed frame={seed_idx} bbox={arm_boxes[seed_idx].round(1).tolist()} "
          f"({len(prompt_indices)} temporal prompts, "
          f"{'negatives on' if use_negatives else 'no negatives'})")

    def _fwd_rev(pidx):
        acc = np.zeros((n_frames, img_h, img_w), dtype=bool)
        for reverse in (False, True):
            out = _segment_one_pass(video_predictor, frames_dir,
                                    arm_boxes[pidx], prompt_points[pidx], pidx,
                                    reverse=reverse, labels=prompt_labels)
            for idx, m in out.items():
                acc[idx] |= m[0]
        return acc

    masks |= _fwd_rev(prompt_indices)
    for _ in range(max(0, reanchor_rounds)):
        areas = masks.reshape(n_frames, -1).sum(1)
        va = areas[valid_indices]
        med = np.median(va[va > 0]) if (va > 0).any() else 0.0
        collapsed = valid_indices[areas[valid_indices] < collapse_frac * med]
        collapsed = np.setdiff1d(collapsed, prompt_indices)
        if med <= 0 or len(collapsed) == 0:
            break
        add = collapsed[::max(1, len(collapsed) // 20)]   # cap ~20 new seeds/round
        prompt_indices = np.unique(np.concatenate([prompt_indices, add]))
        masks |= _fwd_rev(add)
        print(f"  [reanchor] +{len(add)} seeds for collapsed frames "
              f"(area<{collapse_frac:.2f}x median)")

    # Keep only the component attached to the hand/forearm prompts.  Crucially,
    # do not crop to the hand bbox: that old crop amputated the forearm.
    previous = None
    for idx in range(n_frames):
        prompt_idx = int(valid_indices[np.argmin(np.abs(valid_indices - idx))])
        masks[idx] = _component_at_prompts(
            masks[idx], arm_points[prompt_idx], previous,
        )
        previous = masks[idx]

    # Continuity guarantee: any frame the propagation left empty or badly collapsed
    # is filled from its nearest well-covered neighbour, so the hand/arm mask is
    # present across the ENTIRE clip (no dropout holes that leak the human hand into
    # the inpaint). This is what makes the mask "propagate throughout".
    areas = masks.reshape(n_frames, -1).sum(1)
    med = float(np.median(areas[areas > 0])) if (areas > 0).any() else 0.0
    good = areas > max(1.0, collapse_frac * med)
    if good.any() and not good.all():
        gi = np.flatnonzero(good)
        for idx in np.flatnonzero(~good):
            masks[idx] = masks[gi[np.argmin(np.abs(gi - idx))]]
        print(f"  [fill] filled {int((~good).sum())} empty/collapsed frames "
              f"from nearest covered frame (mask now continuous over {n_frames})")
    return masks


def _segment_arms_egodex(video_predictor, frames_dir: Path, arm_kpts,
                         n_frames: int, img_h: int, img_w: int) -> np.ndarray:
    """EgoDex path: seed SAM2 from projected arm-chain keypoints (elbow ->
    forearm -> wrist -> fingers, interpolated along the sleeve) plus background
    negatives (grid points far from any arm joint, so SAM2 doesn't eat the
    table). Segments BOTH full arms+hands in one pass. Returns (T,H,W) bool."""
    def frame_pts(fr, side):
        P = arm_kpts[side][fr]
        v = arm_kpts[side + "_valid"][fr]
        pts = list(P[v])
        for a, b in [(0, 1), (1, 2)]:           # arm->forearm->hand densify
            if v[a] and v[b]:
                for t in np.linspace(0.2, 0.8, 3):
                    pts.append(P[a] * (1 - t) + P[b] * t)
        return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 2), np.float32)

    seed = max((len(frame_pts(fr, "left")) + len(frame_pts(fr, "right")), fr)
               for fr in range(n_frames))[1]
    pos = np.concatenate([frame_pts(seed, "left"), frame_pts(seed, "right")], 0)
    masks = np.zeros((n_frames, img_h, img_w), dtype=bool)
    if len(pos) == 0:
        print("  [warn] no projected arm keypoints — empty arm mask")
        return masks

    gx, gy = np.meshgrid(np.linspace(20, img_w - 20, 9),
                         np.linspace(20, img_h - 20, 7))
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    far = grid[np.sqrt(((grid[:, None] - pos[None]) ** 2).sum(-1)).min(1) > 70]
    pts = np.concatenate([pos, far], 0).astype(np.float32)
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(far))]).astype(np.int32)
    box = np.array([0, img_h * 0.15, img_w, img_h], dtype=np.float32)
    print(f"  EgoDex arm seed frame={seed}: {len(pos)} arm pts, {len(far)} bg negatives")

    for reverse in (False, True):
        out = _segment_one_pass(
            video_predictor, frames_dir, box[None], pts[None],
            np.array([seed]), reverse=reverse, labels=labels
        )
        for idx, m in out.items():
            masks[idx] |= m[0]
    return masks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--arm_kpts", type=Path, default=None,
                    help="EgoDex arm_kpts_2d.npz (projected arm-chain joints). "
                         "If given, SAM2 is seeded along the whole arm; else "
                         "falls back to hand-bbox seeding.")
    ap.add_argument("--keep_tmp", action="store_true",
                    help="Keep original_images/ and original_images_reverse/ for debugging")
    ap.add_argument("--forearm_scales", type=float, nargs="+",
                    default=[0.75, 1.5, 2.5, 4.0, 6.0, 8.0],
                    help="Wrist-minus-palm extrapolation scales used as SAM2 "
                         "positive prompts. Default reaches up the whole arm to the "
                         "frame edge; shorten (e.g. 0.75 1.5 2.5) for short/cropped arms.")
    ap.add_argument("--no_object_negatives", dest="object_negatives",
                    action="store_false", default=True,
                    help="Disable SAM2 negative prompts on the grasped object. By "
                         "default negatives are placed beyond the fingertips so a held "
                         "object is carved out and the mask stays hand+arm only.")
    ap.add_argument("--object_neg_scales", type=float, nargs="+", default=[0.6, 1.2],
                    help="Fingertip-minus-palm extrapolation scales for the object "
                         "negative prompts.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output mask path. Default: "
                         "<processed_demo>/segmentation_processor/masks_arm.npy")
    # mask denoising (see _smooth_masks)
    ap.add_argument("--smooth", dest="smooth", action="store_true", default=True,
                    help="Denoise masks: speck removal + close + temporal-majority "
                         "flicker removal + edge rounding (default on).")
    ap.add_argument("--no_smooth", dest="smooth", action="store_false",
                    help="Disable mask denoising.")
    ap.add_argument("--smooth_min_area_frac", type=float, default=2e-4,
                    help="Drop connected components smaller than this fraction of "
                         "the frame (speck removal). 0 = keep all.")
    ap.add_argument("--smooth_close", type=int, default=5,
                    help="Morphological close kernel (px) to fill holes/notches. 1 = off.")
    ap.add_argument("--smooth_temporal_win", type=int, default=5,
                    help="Temporal majority window (odd frames) for flicker removal. 1 = off.")
    ap.add_argument("--smooth_boundary_sigma", type=float, default=1.5,
                    help="Gaussian sigma (px) to round jagged mask edges. 0 = off.")
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

    arm_kpts = None
    if args.arm_kpts is not None and args.arm_kpts.exists():
        arm_kpts = np.load(args.arm_kpts)

    if arm_kpts is not None:
        print("[arms] EgoDex arm-keypoint seeding (whole hand+arm)")
        masks = _segment_arms_egodex(video_predictor, frames_dir, arm_kpts,
                                     n_frames, img_h, img_w)
    else:
        print("[arms] hand-bbox seeding (fallback; no EgoDex arm keypoints)")
        left_masks = _segment_hand(
            video_predictor, frames_dir, n_frames,
            bbox["left_bboxes"], bbox["left_bbox_min_dist_to_edge"],
            bbox["left_hand_detected"], hd_l["kpts_2d"][:n_frames],
            img_h, img_w, tuple(args.forearm_scales),
            use_negatives=args.object_negatives, neg_scales=tuple(args.object_neg_scales),
        )
        right_masks = _segment_hand(
            video_predictor, frames_dir, n_frames,
            bbox["right_bboxes"], bbox["right_bbox_min_dist_to_edge"],
            bbox["right_hand_detected"], hd_r["kpts_2d"][:n_frames],
            img_h, img_w, tuple(args.forearm_scales),
            use_negatives=args.object_negatives, neg_scales=tuple(args.object_neg_scales),
        )
        masks = left_masks | right_masks
    masks, repaired_indices = _repair_temporal_mask_outliers(masks)
    if repaired_indices:
        print(f"[repair] temporal SAM leak frames: {repaired_indices}")
    if args.smooth:
        masks, n_specks = _smooth_masks(
            masks, args.smooth_min_area_frac, args.smooth_close,
            args.smooth_temporal_win, args.smooth_boundary_sigma)
        print(f"[smooth] specks removed={n_specks}, close={args.smooth_close}px, "
              f"temporal_win={args.smooth_temporal_win}, "
              f"boundary_sigma={args.smooth_boundary_sigma}px")
    per_frame = masks.sum(axis=(1, 2))
    print(f"[info] frames with mask: {(per_frame > 0).sum()}/{n_frames}, "
          f"avg {per_frame.mean():.0f} px, max {per_frame.max()} px")

    out_dir = pd / "segmentation_processor"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or (out_dir / "masks_arm.npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, masks)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[ok] wrote {output}")

    if not args.keep_tmp:
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
