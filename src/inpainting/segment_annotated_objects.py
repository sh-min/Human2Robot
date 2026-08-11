"""Segment each annotated manipulated-object interval with an independent SAM2 track.

The annotation JSON is expected to contain inclusive frame intervals such as
``Cup``, ``Choco`` and ``Sweep``, separated by ``Trans`` intervals. A separate
SAM2 state is used for every non-transition interval so an object from one
task cannot leak into the next task.

The seed is selected from the central part of each interval using the tightest
HaWoR fingertip configuration.  Positive prompts lie at the fingertip centroid
and thumb/index pinch point; wrist and MCP joints are negative prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import mediapy as media
import numpy as np
import torch

from _paths import SAM2_CHECKPOINT, SAM2_CONFIG_NAME, ensure_sam2_importable
from segment_arms import _dump_frames_as_jpegs, _segment_one_pass
from segment_object import _GRIP_BAND, _grip_metrics, _pick_grasp_seed

ensure_sam2_importable()
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_intervals(labels_path: Path, frame_count: int) -> list[dict]:
    payload = json.loads(labels_path.read_text())
    intervals = []
    previous_end = -1
    for item in payload.get("segments", []):
        label = str(item["label"])
        start = max(0, int(item["start_frame"]))
        end = min(frame_count - 1, int(item["end_frame"]))
        if start > end:
            continue
        if start <= previous_end:
            raise ValueError(
                f"overlapping/non-monotonic annotation at {label}: "
                f"{start} <= {previous_end}"
            )
        previous_end = end
        if label.casefold() == "trans":
            continue
        intervals.append({"label": label, "start": start, "end": end})
    if not intervals:
        raise ValueError(f"no non-transition intervals in {labels_path}")
    return intervals


def _select_seed(
    hand_data: dict[str, np.lib.npyio.NpzFile],
    start: int,
    end: int,
) -> tuple[int, str, float]:
    length = end - start + 1
    margin = min(max(1, int(round(0.15 * length))), max(0, length // 3))
    lo, hi = start + margin, end - margin
    candidates = []
    fallback = []
    for side, data in hand_data.items():
        detected = np.asarray(data["hand_detected"], dtype=bool)
        for frame in range(lo, hi + 1):
            if not detected[frame]:
                continue
            spread = _grip_metrics(data["kpts_2d"][frame])
            fallback.append((abs(spread - np.mean(_GRIP_BAND)), frame, side, spread))
            if _GRIP_BAND[0] <= spread <= _GRIP_BAND[1]:
                candidates.append((spread, frame, side))
    if candidates:
        spread, frame, side = min(candidates)
        return int(frame), str(side), float(spread)
    if fallback:
        _, frame, side, spread = min(fallback)
        return int(frame), str(side), float(spread)
    raise RuntimeError(f"no detected hand in annotated interval {start}:{end}")


def _link_interval_frames(
    all_frames: Path,
    interval_dir: Path,
    start: int,
    end: int,
) -> None:
    interval_dir.mkdir(parents=True, exist_ok=True)
    for local, global_index in enumerate(range(start, end + 1)):
        source = all_frames / f"{global_index:05d}.jpg"
        target = interval_dir / f"{local:05d}.jpg"
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def _component_near_grasp(
    mask: np.ndarray,
    hand_data: dict[str, np.lib.npyio.NpzFile],
    side: str,
    frame: int,
) -> np.ndarray:
    """Drop disconnected SAM distractors, keeping the component at the grasp."""
    binary = np.asarray(mask, dtype=np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 2:
        return binary.astype(bool)

    points = np.asarray(hand_data[side]["kpts_2d"][frame], dtype=np.float32)
    fingertips = points[[4, 8, 12, 16, 20]]
    grasp_points = np.concatenate(
        [
            fingertips,
            fingertips.mean(axis=0, keepdims=True),
            ((points[4] + points[8]) / 2.0)[None],
        ],
        axis=0,
    )
    height, width = binary.shape
    best_label = 0
    best_score = -np.inf
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        local = components[y:y + h, x:x + w] == label
        ys, xs = np.nonzero(local)
        pixels = np.stack([xs + x, ys + y], axis=1).astype(np.float32)
        if not len(pixels):
            continue
        distance_sq = (
            (pixels[:, None] - grasp_points[None]) ** 2
        ).sum(axis=2)
        min_distance_sq = float(distance_sq.min())
        prompt_hits = 0
        for px, py in grasp_points:
            xi = int(np.clip(round(float(px)), 0, width - 1))
            yi = int(np.clip(round(float(py)), 0, height - 1))
            prompt_hits += int(components[yi, xi] == label)
        # Prompt containment dominates, followed by distance.  A tiny area
        # preference breaks ties without allowing a large table/bin component
        # to beat the held object.
        score = 1_000_000.0 * prompt_hits - min_distance_sq + 1e-3 * area
        if score > best_score:
            best_score = score
            best_label = label
    return components == best_label


def _write_debug_video(
    video_path: Path,
    masks: np.ndarray,
    intervals: list[dict],
    output_path: Path,
    fps: float,
) -> None:
    frames = media.read_video(str(video_path))
    owner = np.full(len(masks), "", dtype=object)
    for interval in intervals:
        owner[interval["start"]:interval["end"] + 1] = interval["label"]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (frames.shape[2], frames.shape[1]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open debug writer: {output_path}")
    try:
        for index, rgb in enumerate(frames):
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            mask = masks[index]
            colored = bgr.copy()
            colored[mask] = (
                0.35 * colored[mask] + 0.65 * np.array([40, 220, 40])
            ).astype(np.uint8)
            cv2.putText(
                colored,
                f"{index:04d} {owner[index]}",
                (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                colored,
                f"{index:04d} {owner[index]}",
                (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(colored)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_demo", type=Path, required=True)
    parser.add_argument("--labels_json", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--keep_tmp", action="store_true")
    args = parser.parse_args()

    if not Path(SAM2_CHECKPOINT).is_file():
        sys.exit(f"SAM2 checkpoint missing: {SAM2_CHECKPOINT}")

    processed = args.processed_demo.resolve()
    video_path = processed / "video_L.mp4"
    frames = media.read_video(str(video_path))
    frame_count, height, width = frames.shape[:3]
    intervals = _load_intervals(args.labels_json.resolve(), frame_count)

    hand_data = {}
    for side in ("left", "right"):
        path = processed / "hand_processor" / f"hand_data_{side}.npz"
        if path.is_file():
            hand_data[side] = np.load(path)
    if not hand_data:
        sys.exit("[err] no HaWoR hand data; run inject_hawor_data first")

    frames_dir = processed / "original_images"
    _dump_frames_as_jpegs(video_path, frames_dir)
    temp_root = Path(tempfile.mkdtemp(prefix=".object_segments.", dir=processed))
    masks = np.zeros((frame_count, height, width), dtype=bool)
    reports = []
    try:
        predictor = build_sam2_video_predictor(
            SAM2_CONFIG_NAME,
            SAM2_CHECKPOINT,
            device=DEVICE,
        )
        for interval_index, interval in enumerate(intervals):
            start, end = interval["start"], interval["end"]
            seed, side, spread = _select_seed(hand_data, start, end)
            _, points, labels, box = _pick_grasp_seed(
                hand_data,
                height,
                width,
                force_frame=seed,
                force_side=side,
            )
            segment_dir = temp_root / f"{interval_index:02d}_{interval['label']}"
            _link_interval_frames(frames_dir, segment_dir, start, end)
            local_seed = seed - start
            segment_mask = np.zeros(
                (end - start + 1, height, width),
                dtype=bool,
            )
            for reverse in (False, True):
                output = _segment_one_pass(
                    predictor,
                    segment_dir,
                    box[None],
                    points[None],
                    np.array([local_seed]),
                    reverse=reverse,
                    labels=labels,
                )
                for local_index, value in output.items():
                    segment_mask[local_index] |= value[0]
            for local_index in range(len(segment_mask)):
                segment_mask[local_index] = _component_near_grasp(
                    segment_mask[local_index],
                    hand_data,
                    side,
                    start + local_index,
                )
            masks[start:end + 1] = segment_mask
            areas = segment_mask.reshape(len(segment_mask), -1).sum(axis=1)
            report = {
                **interval,
                "seed_frame": seed,
                "seed_side": side,
                "grip_spread": spread,
                "frames_with_mask": int((areas > 0).sum()),
                "median_area_px": int(np.median(areas)),
                "max_area_px": int(areas.max(initial=0)),
            }
            reports.append(report)
            print(
                f"[segment] {interval['label']} {start}:{end}, "
                f"seed={seed}/{side}, visible={report['frames_with_mask']}/"
                f"{len(segment_mask)}, median={report['median_area_px']} px",
                flush=True,
            )
    finally:
        for data in hand_data.values():
            data.close()
        if not args.keep_tmp:
            shutil.rmtree(temp_root, ignore_errors=True)
            shutil.rmtree(frames_dir, ignore_errors=True)
        torch.cuda.empty_cache()

    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else processed / "object_layer"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / "object_mask_modal.npy"
    temporary_mask = mask_path.with_name(f".{mask_path.name}.tmp")
    try:
        with temporary_mask.open("wb") as stream:
            np.save(stream, masks)
        temporary_mask.replace(mask_path)
    finally:
        temporary_mask.unlink(missing_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    finally:
        capture.release()
    _write_debug_video(
        video_path,
        masks,
        intervals,
        out_dir / "debug_object_mask.mp4",
        fps,
    )
    manifest = {
        "schema_version": 1,
        "labels_json": str(args.labels_json.resolve()),
        "frame_count": frame_count,
        "height": height,
        "width": width,
        "transition_policy": "empty_mask",
        "intervals": reports,
    }
    manifest_path = out_dir / "manifest.json"
    temporary_manifest = out_dir / ".manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(manifest_path)
    print(f"[ok] modal object masks: {mask_path}", flush=True)


if __name__ == "__main__":
    main()
