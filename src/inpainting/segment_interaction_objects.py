"""Track several manipulated objects and merge them into one visible-object mask.

Unlike ``segment_cube.py``, this stage does not assume that one object is present
for the whole clip.  A JSON file describes each interaction interval and gives
one SAM2 prompt inside that interval.  SAM2 is propagated only to the interval
boundaries, then all interval masks are merged into a single ``(T,H,W)`` mask.

The mask is *modal*: it contains only object pixels visible in the source RGB.
That is exactly what the 2.5D compositor needs to put source object pixels back
in front of robot links without hallucinating hidden object texture.

Example::

    python segment_interaction_objects.py \
      --processed_demo /path/to/processed/demo/0 \
      --segments_json ../../configs/inpainting/v0729_01_objects.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from _paths import SAM2_CHECKPOINT, SAM2_CONFIG_NAME, ensure_sam2_importable

ensure_sam2_importable()
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402


def _video_info(path: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    cap.release()
    return count, height, width, fps


def _dump_jpegs(video_path: Path, frames_dir: Path, expected: int) -> None:
    existing = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
    if len(existing) == expected:
        return
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in existing:
        old.unlink()
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(
            frames_dir / f"{idx:05d}.jpg", quality=95
        )
        idx += 1
    cap.release()
    if idx != expected:
        raise RuntimeError(f"decoded {idx} frames, expected {expected}")


def _parse_segments(path: Path, frame_count: int) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload["segments"] if isinstance(payload, dict) else payload
    if not segments:
        raise ValueError("segments JSON is empty")
    parsed = []
    for idx, item in enumerate(segments):
        seg = dict(item)
        seg.setdefault("name", f"object_{idx}")
        start = int(seg["start_frame"])
        end = int(seg["end_frame"])
        seed = int(seg["seed_frame"])
        if not (0 <= start <= seed <= end < frame_count):
            raise ValueError(
                f"{seg['name']}: require 0 <= start <= seed <= end < {frame_count}"
            )
        box = np.asarray(seg["box"], dtype=np.float32)
        if box.shape != (4,) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"{seg['name']}: invalid box {box.tolist()}")
        positive = np.asarray(seg.get("positive_points", []), dtype=np.float32).reshape(-1, 2)
        negative = np.asarray(seg.get("negative_points", []), dtype=np.float32).reshape(-1, 2)
        if len(positive) == 0:
            positive = np.asarray([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]],
                                  dtype=np.float32)
        seg["start_frame"], seg["end_frame"], seg["seed_frame"] = start, end, seed
        seg["box"], seg["positive_points"], seg["negative_points"] = box, positive, negative
        parsed.append(seg)
    return parsed


def _keep_component(mask: np.ndarray,
                    previous: np.ndarray | None,
                    seed_points: np.ndarray | None = None) -> np.ndarray:
    """Remove detached SAM leaks while following the component through time."""
    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary.astype(bool)
    scores = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    scores[0] = -np.inf
    if previous is not None and previous.any():
        overlap_labels, overlap_counts = np.unique(labels[previous], return_counts=True)
        for label, overlap in zip(overlap_labels, overlap_counts):
            if label:
                scores[label] += 1000.0 * float(overlap)
    if seed_points is not None:
        h, w = binary.shape
        for x, y in seed_points:
            label = labels[
                int(np.clip(round(float(y)), 0, h - 1)),
                int(np.clip(round(float(x)), 0, w - 1)),
            ]
            if label:
                scores[label] += 1e9
    return labels == int(np.argmax(scores))


def _project_mano_core(verts: np.ndarray, valid: np.ndarray, frame_idx: int,
                       focal: float, width: int, height: int,
                       dilate_px: int) -> np.ndarray | None:
    """Rasterise the MANO surface without filling between grasping fingers."""
    if frame_idx >= len(verts) or not valid[frame_idx]:
        return None
    xyz = np.asarray(verts[frame_idx], dtype=np.float32)
    z = xyz[:, 2]
    keep = z > 1e-3
    if not keep.any():
        return None
    xyz, z = xyz[keep], z[keep]
    u = np.rint(focal * xyz[:, 0] / z + width / 2.0).astype(np.int32)
    v = np.rint(focal * xyz[:, 1] / z + height / 2.0).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not inside.any():
        return None
    core = np.zeros((height, width), dtype=np.uint8)
    core[v[inside], u[inside]] = 1
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
        core = cv2.dilate(core, kernel, iterations=1)
    return core.astype(bool)


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return float(xs.mean()), float(ys.mean())


def _translate_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        mask.astype(np.uint8), matrix, (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)


def _repair_collapsed_frames(track: np.ndarray, seg: dict,
                             collapse_ratio: float,
                             search_radius: int) -> np.ndarray:
    """Bridge short mask collapses from neighbouring frames of the same track."""
    start, end = seg["start_frame"], seg["end_frame"]
    area = track[start:end + 1].sum(axis=(1, 2)).astype(np.float64)
    if not len(area) or collapse_ratio <= 0 or search_radius <= 0:
        return track
    repaired = 0
    # Decide all bad frames from the original areas so repairing one frame does
    # not recursively make a long, genuinely absent interval look trustworthy.
    bad = np.zeros(len(area), dtype=bool)
    for local in range(len(area)):
        lo = max(0, local - search_radius)
        hi = min(len(area), local + search_radius + 1)
        positive = area[lo:hi][area[lo:hi] >= 250]
        if len(positive) >= 3:
            reference = float(np.median(positive))
            bad[local] = area[local] < collapse_ratio * reference

    for local in np.flatnonzero(bad):
        left = next((j for j in range(local - 1,
                                      max(-1, local - search_radius - 1), -1)
                     if not bad[j] and area[j] >= 250), None)
        right = next((j for j in range(local + 1,
                                       min(len(area), local + search_radius + 1))
                      if not bad[j] and area[j] >= 250), None)
        if left is None or right is None:
            continue
        frame_idx = start + local
        left_idx, right_idx = start + left, start + right
        left_c = _centroid(track[left_idx])
        right_c = _centroid(track[right_idx])
        current_c = _centroid(track[frame_idx])
        if left_c is None or right_c is None:
            continue
        alpha = (local - left) / float(right - left)
        target = current_c if area[local] >= 100 else (
            (1.0 - alpha) * left_c[0] + alpha * right_c[0],
            (1.0 - alpha) * left_c[1] + alpha * right_c[1],
        )
        from_left = _translate_mask(
            track[left_idx], target[0] - left_c[0], target[1] - left_c[1])
        from_right = _translate_mask(
            track[right_idx], target[0] - right_c[0], target[1] - right_c[1])
        candidate = track[frame_idx] | from_left | from_right
        candidate = _keep_component(candidate, track[frame_idx])
        if candidate.sum() <= 2.5 * max(area[left], area[right]):
            track[frame_idx] = candidate
            repaired += 1
    if repaired:
        print(f"[repair] {seg['name']}: restored {repaired} collapsed frames")
    return track


def _clean_track(track: np.ndarray, seg: dict, human: np.ndarray | None,
                 mano_verts: np.ndarray | None = None,
                 mano_valid: np.ndarray | None = None,
                 focal: float | None = None,
                 mano_core_dilate_px: int = 8,
                 collapse_ratio: float = 0.22,
                 collapse_search_frames: int = 8) -> np.ndarray:
    """Component continuity, tiny-hole closing, and conservative hand removal."""
    start, end, seed = seg["start_frame"], seg["end_frame"], seg["seed_frame"]
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    seed_mask = _keep_component(track[seed], None, seg["positive_points"])
    track[seed] = seed_mask
    previous = seed_mask
    for frame_idx in range(seed + 1, end + 1):
        current = _keep_component(track[frame_idx], previous)
        track[frame_idx] = current
        if current.any():
            previous = current
    previous = seed_mask
    for frame_idx in range(seed - 1, start - 1, -1):
        current = _keep_component(track[frame_idx], previous)
        track[frame_idx] = current
        if current.any():
            previous = current

    for frame_idx in range(start, end + 1):
        current = cv2.morphologyEx(track[frame_idx].astype(np.uint8),
                                   cv2.MORPH_CLOSE, close_kernel)
        # SAM's propagated arm mask frequently absorbs the grasped object as
        # soon as hand and object become one connected silhouette. Subtracting
        # that whole mask deletes the object exactly at contact. Restrict the
        # subtraction to pixels supported by projected MANO geometry. The
        # sparse/dilated surface covers real fingers but deliberately does not
        # fill the space between them, where the held object lives.
        if human is not None and frame_idx < len(human):
            human_frame = np.asarray(human[frame_idx], dtype=bool)
            mano_core = None
            if (mano_verts is not None and mano_valid is not None and
                    focal is not None):
                mano_core = _project_mano_core(
                    mano_verts, mano_valid, frame_idx, focal,
                    current.shape[1], current.shape[0], mano_core_dilate_px)
            rejection = human_frame if mano_core is None else human_frame & mano_core
            current[rejection] = 0
        track[frame_idx] = current.astype(bool)

    # Removing the true hand core cuts an occasional SAM bridge between object
    # and forearm. Run continuity once more so the detached arm fragment cannot
    # survive merely because it was connected before hand removal.
    seed_mask = _keep_component(track[seed], None, seg["positive_points"])
    track[seed] = seed_mask
    previous = seed_mask
    for frame_idx in range(seed + 1, end + 1):
        current = _keep_component(track[frame_idx], previous)
        track[frame_idx] = current
        if current.any():
            previous = current
    previous = seed_mask
    for frame_idx in range(seed - 1, start - 1, -1):
        current = _keep_component(track[frame_idx], previous)
        track[frame_idx] = current
        if current.any():
            previous = current
    return _repair_collapsed_frames(
        track, seg, collapse_ratio, collapse_search_frames)


def _track_is_reliable(track: np.ndarray, seg: dict) -> tuple[bool, str]:
    """Content-independent rejection for collapsed or explosive SAM tracks."""
    area = track[seg["start_frame"]:seg["end_frame"] + 1].sum(axis=(1, 2))
    positive = area[area > 0]
    if len(positive) < max(3, int(0.35 * len(area))):
        return False, "mask absent for most of interval"
    median = float(np.median(area))
    seed_area = float(area[seg["seed_frame"] - seg["start_frame"]])
    if median < 250 or seed_area < 250:
        return False, "mask collapsed below 250 pixels"
    if np.percentile(positive, 95) > 8.0 * max(seed_area, 1.0):
        return False, "mask area exploded relative to seed"
    return True, "ok"


def _write_preview(video_path: Path, masks: np.ndarray, output: Path, fps: float) -> None:
    cap = cv2.VideoCapture(str(video_path))
    height, width = masks.shape[1:]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create preview: {output}")
    idx = 0
    while idx < len(masks):
        ok, frame = cap.read()
        if not ok:
            break
        overlay = frame.copy()
        overlay[masks[idx]] = (
            0.35 * overlay[masks[idx]] + 0.65 * np.array([40, 220, 40])
        ).astype(np.uint8)
        contours, _ = cv2.findContours(masks[idx].astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
        writer.write(overlay)
        idx += 1
    cap.release()
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_demo", type=Path, required=True)
    parser.add_argument("--segments_json", type=Path, required=True)
    parser.add_argument("--video", default="video_L.mp4")
    parser.add_argument("--human_mask", default="segmentation_processor/masks_arm.npy")
    parser.add_argument("--output", default="interaction_objects/object_mask.npy")
    parser.add_argument("--preview", default="interaction_objects/object_mask_preview.mp4")
    parser.add_argument("--hawor_npz", type=Path, default=None,
                        help="HaWoR retarget_input.npz. When supplied, only the "
                             "SAM human pixels supported by projected MANO are "
                             "removed from the object track.")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--mano_core_dilate_px", type=int, default=8)
    parser.add_argument("--collapse_ratio", type=float, default=0.22,
                        help="Repair a frame when its mask area falls below "
                             "this fraction of its local temporal reference.")
    parser.add_argument("--collapse_search_frames", type=int, default=8)
    parser.add_argument("--keep_frames", action="store_true")
    args = parser.parse_args()

    if not Path(SAM2_CHECKPOINT).exists():
        sys.exit(f"SAM2 checkpoint missing: {SAM2_CHECKPOINT}")
    if not torch.cuda.is_available():
        sys.exit("CUDA is required for this SAM2 video pass")

    processed = args.processed_demo
    video_path = processed / args.video
    frame_count, height, width, fps = _video_info(video_path)
    segments = _parse_segments(args.segments_json, frame_count)
    frames_dir = processed / "interaction_objects" / "sam2_frames"
    _dump_jpegs(video_path, frames_dir, frame_count)

    human_path = processed / args.human_mask
    human = np.load(human_path, mmap_mode="r") if human_path.exists() else None
    mano_verts = mano_valid = None
    mano_focal = None
    if args.hawor_npz is not None:
        hawor = np.load(args.hawor_npz)
        side_idx = 0 if args.side == "left" else 1
        mano_verts = hawor[f"verts_{args.side}"]
        mano_valid = hawor["valid"][side_idx]
        mano_focal = float(hawor["img_focal"])
    masks = np.zeros((frame_count, height, width), dtype=bool)

    print(f"[info] {frame_count} frames, {width}x{height}, {len(segments)} object intervals")
    predictor = build_sam2_video_predictor(
        SAM2_CONFIG_NAME, SAM2_CHECKPOINT, device="cuda"
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False,
        )
        for obj_id, seg in enumerate(segments):
            predictor.reset_state(state)
            points = np.concatenate(
                [seg["positive_points"], seg["negative_points"]], axis=0
            ).astype(np.float32)
            labels = np.concatenate(
                [np.ones(len(seg["positive_points"]), np.int32),
                 np.zeros(len(seg["negative_points"]), np.int32)]
            )
            seed = seg["seed_frame"]
            predictor.add_new_points_or_box(
                state,
                frame_idx=seed,
                obj_id=obj_id,
                box=seg["box"],
                points=points,
                labels=labels,
            )
            track = np.zeros_like(masks)
            for reverse, distance in (
                (False, seg["end_frame"] - seed),
                (True, seed - seg["start_frame"]),
            ):
                for frame_idx, _, logits in predictor.propagate_in_video(
                    state,
                    start_frame_idx=seed,
                    max_frame_num_to_track=distance,
                    reverse=reverse,
                ):
                    if seg["start_frame"] <= frame_idx <= seg["end_frame"]:
                        track[frame_idx] = (logits[0].squeeze() > 0).cpu().numpy()
            track = _clean_track(
                track, seg, human, mano_verts, mano_valid, mano_focal,
                args.mano_core_dilate_px, args.collapse_ratio,
                args.collapse_search_frames)
            reliable, reason = _track_is_reliable(track, seg)
            if reliable:
                masks[seg["start_frame"]:seg["end_frame"] + 1] |= \
                    track[seg["start_frame"]:seg["end_frame"] + 1]
            else:
                print(f"[reject] {seg['name']}: {reason}")
            area = track[seg["start_frame"]:seg["end_frame"] + 1].sum(axis=(1, 2))
            print(f"[object] {seg['name']}: frames {seg['start_frame']}..{seg['end_frame']}, "
                  f"seed={seed}, area median={int(np.median(area))}, max={int(area.max())}")
            torch.cuda.empty_cache()

    output = processed / args.output
    preview = processed / args.preview
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, masks)
    _write_preview(video_path, masks, preview, fps)
    print(f"[ok] wrote {output}")
    print(f"[ok] wrote {preview}")
    if not args.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
