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
Left and right masks are finally unioned into a bimanual hand+arm mask.

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
    arm_points = _augment_with_forearm_points(
        kpts_2d, img_h, img_w, forearm_scales,
    )
    arm_boxes = _expand_boxes_to_prompts(bboxes, arm_points, img_h, img_w)
    print(f"  seed frame={seed_idx} bbox={arm_boxes[seed_idx].round(1).tolist()} "
          f"({len(prompt_indices)} temporal prompts)")

    def _fwd_rev(pidx):
        acc = np.zeros((n_frames, img_h, img_w), dtype=bool)
        for reverse in (False, True):
            out = _segment_one_pass(video_predictor, frames_dir,
                                    arm_boxes[pidx], arm_points[pidx], pidx, reverse=reverse)
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
                    default=[0.75, 1.5, 2.5],
                    help="Wrist-minus-palm extrapolation scales used as SAM2 "
                         "positive prompts. Use e.g. 0.75 1.5 2.5 4 6 to "
                         "continue through a long sleeve to the frame edge.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output mask path. Default: "
                         "<processed_demo>/segmentation_processor/masks_arm.npy")
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
        )
        right_masks = _segment_hand(
            video_predictor, frames_dir, n_frames,
            bbox["right_bboxes"], bbox["right_bbox_min_dist_to_edge"],
            bbox["right_hand_detected"], hd_r["kpts_2d"][:n_frames],
            img_h, img_w, tuple(args.forearm_scales),
        )
        masks = left_masks | right_masks
    masks, repaired_indices = _repair_temporal_mask_outliers(masks)
    if repaired_indices:
        print(f"[repair] temporal SAM leak frames: {repaired_indices}")
    per_frame = masks.sum(axis=(1, 2))
    print(f"[info] frames with mask: {(per_frame > 0).sum()}/{n_frames}, "
          f"avg {per_frame.mean():.0f} px, max {per_frame.max()} px")

    out_dir = pd / "segmentation_processor"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or (out_dir / "masks_arm.npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, masks)
    print(f"[ok] wrote {output}")

    if not args.keep_tmp:
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
