"""Restore clean object texture underneath a hand during static contact.

For each configured interaction, this utility extracts the object mask and RGB
from a clean pre-contact reference frame.  The object is pasted only until its
first motion interval ends, so hand-occluded pixels contain the original object
instead of a temporal background plate without leaving a duplicate after lift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from refine_interaction_object_masks import _track_interval


def _solid_mask(mask: np.ndarray) -> np.ndarray:
    points = np.column_stack(np.nonzero(mask))
    if not len(points):
        return np.asarray(mask, dtype=bool)
    hull = cv2.convexHull(points[:, ::-1].astype(np.int32))
    result = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(result, hull, 1)
    return result.astype(bool)


def _read_frames(video: Path, indices: set[int]) -> tuple[dict[int, np.ndarray],
                                                           int, int, float, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: dict[int, np.ndarray] = {}
    for index in sorted(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read reference frame {index}")
        frames[index] = frame
    cap.release()
    return frames, width, height, fps, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--merged_mask", type=Path, required=True)
    parser.add_argument("--segments_json", type=Path, required=True)
    parser.add_argument("--completion_json", type=Path, required=True)
    parser.add_argument("--base_object_mask", type=Path, required=True)
    parser.add_argument(
        "--reference_mask_override", action="append", default=[],
        metavar="NAME=MASK_NPY",
        help="Use MASK_NPY at the configured reference frame for NAME instead "
             "of extracting the reference from the merged interaction mask. "
             "This is useful for an object prompted in a clean earlier frame.",
    )
    parser.add_argument("--output_video", type=Path, required=True)
    parser.add_argument("--output_mask", type=Path, required=True)
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()

    merged = np.load(args.merged_mask, mmap_mode="r")
    base_mask = np.load(args.base_object_mask, mmap_mode="r")
    segment_payload = json.loads(args.segments_json.read_text(encoding="utf-8"))
    completion_payload = json.loads(
        args.completion_json.read_text(encoding="utf-8")
    )
    segments = {item["name"]: item for item in segment_payload["segments"]}
    references = {int(item["reference_frame"])
                  for item in completion_payload["completions"]}
    reference_frames, width, height, fps, video_count = _read_frames(
        args.video, references
    )
    frame_count = min(video_count, len(merged), len(base_mask))

    override_paths: dict[str, Path] = {}
    for value in args.reference_mask_override:
        if "=" not in value:
            raise ValueError(
                f"--reference_mask_override requires NAME=MASK_NPY: {value}"
            )
        name, path = value.split("=", 1)
        override_paths[name] = Path(path)
    override_masks = {
        name: np.load(path, mmap_mode="r")
        for name, path in override_paths.items()
    }

    layers = []
    for item in completion_payload["completions"]:
        name = item["name"]
        if name not in segments:
            raise KeyError(f"completion segment not found: {name}")
        reference_index = int(item["reference_frame"])
        if name in override_masks:
            reference_mask = np.asarray(
                override_masks[name][reference_index], dtype=bool
            )
            print(
                f"[reference] {name}@{reference_index} from "
                f"{override_paths[name]}"
            )
        else:
            track = _track_interval(merged, segments[name])
            reference_mask = np.asarray(track[reference_index], dtype=bool)
        if item.get("solid_mask", False):
            reference_mask = _solid_mask(reference_mask)
        if not reference_mask.any():
            raise RuntimeError(f"empty reference mask: {name}@{reference_index}")
        layers.append({
            "name": name,
            "start": int(item["start_frame"]),
            "end": int(item["end_frame"]),
            "rgb": reference_frames[reference_index],
            "mask": reference_mask,
        })

    completed_mask = np.asarray(base_mask, dtype=bool).copy()
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_video), cv2.VideoWriter_fourcc(*"FFV1"), fps,
        (width, height),
    )
    preview_writer = None
    if args.preview is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        preview_writer = cv2.VideoWriter(
            str(args.preview), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (width, height),
        )

    cap = cv2.VideoCapture(str(args.video))
    restored_pixels = 0
    for frame_index in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"video decode stopped at {frame_index}")
        completed = frame.copy()
        active_mask = np.zeros((height, width), dtype=bool)
        for layer in layers:
            if layer["start"] <= frame_index <= layer["end"]:
                mask = layer["mask"]
                completed[mask] = layer["rgb"][mask]
                completed_mask[frame_index] |= mask
                active_mask |= mask
                restored_pixels += int(mask.sum())
        writer.write(completed)
        if preview_writer is not None:
            preview = completed.copy()
            outline = cv2.morphologyEx(
                active_mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            ).astype(bool)
            preview[outline] = (20, 20, 245)
            preview_writer.write(preview)
        if (frame_index + 1) % 100 == 0:
            print(f"[frame] {frame_index + 1}/{frame_count}", flush=True)
    cap.release()
    writer.release()
    if preview_writer is not None:
        preview_writer.release()
    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_mask, completed_mask)
    print(f"[info] restored static-object pixels={restored_pixels}")
    print(f"[ok] wrote {args.output_video}")
    print(f"[ok] wrote {args.output_mask}")


if __name__ == "__main__":
    main()
