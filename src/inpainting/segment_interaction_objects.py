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


def _clean_track(track: np.ndarray, seg: dict, human: np.ndarray | None) -> np.ndarray:
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
        # Never restore source human pixels over the robot.  A soft object edge
        # in the compositor hides the possible one-pixel notch at contact much
        # better than leaving a skin-coloured halo in this mask.
        if human is not None and frame_idx < len(human):
            current[np.asarray(human[frame_idx], dtype=bool)] = 0
        track[frame_idx] = current.astype(bool)
    return track


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
            track = _clean_track(track, seg, human)
            masks[seg["start_frame"]:seg["end_frame"] + 1] |= \
                track[seg["start_frame"]:seg["end_frame"] + 1]
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
