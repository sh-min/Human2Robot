#!/usr/bin/env python3
"""Discover manipulated-object episodes from HaCo and track them with SAM2.

This is an offline, annotation-free pilot.  HaCo contact density is smoothed
over the complete sequence; separated local maxima become grasp seeds and
their temporal midpoints define non-overlapping object episodes.  HaWoR hand
keypoints produce positive grasp points and hand-negative prompts for SAM2.
Each seed is then propagated in both temporal directions inside its episode.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]
INPAINT = ROOT / "src/inpainting"
sys.path.insert(0, str(INPAINT))

from _paths import SAM2_CHECKPOINT, SAM2_CONFIG_NAME, ensure_sam2_importable  # noqa: E402
from segment_annotated_objects import (  # noqa: E402
    _component_near_grasp,
    _link_interval_frames,
)
from segment_arms import _dump_frames_as_jpegs, _segment_one_pass  # noqa: E402
from segment_cube import _pick_grasp_seed  # noqa: E402

ensure_sam2_importable()
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402


def discover_episodes(
    contact_dir: Path,
    frame_count: int,
    *,
    side: str,
    smooth_sigma: float,
    min_peak_distance: int,
    peak_prominence: float,
    max_objects: int,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    files = sorted(contact_dir.glob("*.npz"))
    if len(files) != frame_count:
        raise ValueError(
            f"HaCo contact count {len(files)} != video frames {frame_count}"
        )
    density = np.zeros(frame_count, dtype=np.float32)
    valid = np.zeros(frame_count, dtype=bool)
    for index, path in enumerate(files):
        with np.load(path) as contact:
            valid[index] = bool(contact[f"{side}_valid"])
            density[index] = float(contact[f"{side}_contact_mask"].sum())
    smoothed = gaussian_filter1d(density, smooth_sigma)
    peaks, properties = find_peaks(
        smoothed,
        distance=min_peak_distance,
        prominence=peak_prominence,
    )
    peaks = peaks[valid[peaks]]
    if not len(peaks):
        raise RuntimeError("HaCo produced no separated contact episode peaks")
    if len(peaks) > max_objects:
        prominence = properties["prominences"]
        keep = np.argsort(prominence)[-max_objects:]
        peaks = np.sort(peaks[keep])
    boundaries = ((peaks[:-1] + peaks[1:]) // 2).astype(int)
    starts = np.r_[0, boundaries + 1]
    ends = np.r_[boundaries, frame_count - 1]
    episodes = [
        {
            "track_id": int(track_id + 1),
            "start": int(start),
            "end": int(end),
            "seed_frame": int(seed),
            "haco_contact_vertices": int(density[seed]),
            "haco_smoothed_density": float(smoothed[seed]),
        }
        for track_id, (start, end, seed) in enumerate(zip(starts, ends, peaks))
    ]
    return episodes, density, smoothed


def write_debug_video(
    video_path: Path,
    masks: np.ndarray,
    track_ids: np.ndarray,
    output: Path,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not capture.isOpened() or not writer.isOpened():
        raise RuntimeError("could not open SAM2 debug video")
    colors = np.asarray(
        [(0, 0, 0), (40, 220, 40), (40, 180, 240), (220, 80, 80),
         (220, 80, 220), (80, 220, 220), (220, 160, 40)],
        dtype=np.float32,
    )
    try:
        for frame_index in range(len(masks)):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"video read failed at {frame_index}")
            mask = np.asarray(masks[frame_index], dtype=bool)
            track_id = int(track_ids[frame_index])
            color = colors[track_id % len(colors)]
            frame[mask] = (
                0.35 * frame[mask].astype(np.float32) + 0.65 * color
            ).astype(np.uint8)
            cv2.putText(
                frame, f"frame {frame_index:04d} | auto object track {track_id}",
                (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"frame {frame_index:04d} | auto object track {track_id}",
                (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        capture.release()
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_demo", type=Path, required=True)
    parser.add_argument("--episode_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--smooth_sigma", type=float, default=5.0)
    parser.add_argument("--min_peak_distance", type=int, default=45)
    parser.add_argument("--peak_prominence", type=float, default=35.0)
    parser.add_argument("--max_objects", type=int, default=8)
    parser.add_argument("--keep_frames", action="store_true")
    args = parser.parse_args()

    processed = args.processed_demo.resolve()
    episode_dir = args.episode_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = processed / "video_L.mp4"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    episodes, density, smoothed = discover_episodes(
        episode_dir / "contact", frame_count,
        side=args.side,
        smooth_sigma=args.smooth_sigma,
        min_peak_distance=args.min_peak_distance,
        peak_prominence=args.peak_prominence,
        max_objects=args.max_objects,
    )
    print("[discover]", [(e["start"], e["seed_frame"], e["end"]) for e in episodes])

    hand_data = {}
    for side in ("left", "right"):
        path = processed / "hand_processor" / f"hand_data_{side}.npz"
        if path.is_file():
            hand_data[side] = np.load(path)
    if args.side not in hand_data:
        raise FileNotFoundError(f"missing HaWoR {args.side} hand data")

    frames_dir = processed / "original_images_contact_auto"
    _dump_frames_as_jpegs(video_path, frames_dir)
    temp_root = Path(tempfile.mkdtemp(prefix=".sam2_contact_auto.", dir=processed))
    mask_path = output_dir / "object_mask_modal.npy"
    masks = np.lib.format.open_memmap(
        mask_path, mode="w+", dtype=bool,
        shape=(frame_count, height, width),
    )
    masks[:] = False
    track_ids = np.zeros(frame_count, dtype=np.int16)
    predictor = build_sam2_video_predictor(
        SAM2_CONFIG_NAME, SAM2_CHECKPOINT, device="cuda"
    )
    arm_mask_path = processed / "segmentation_processor/masks_arm.npy"
    arm_masks = (
        np.load(arm_mask_path, mmap_mode="r") if arm_mask_path.is_file() else None
    )
    try:
        for episode in episodes:
            start, end, seed = (
                episode["start"], episode["end"], episode["seed_frame"]
            )
            _, points, labels, box = _pick_grasp_seed(
                hand_data, height, width,
                force_frame=seed, force_side=args.side,
            )
            if points is None:
                episode["status"] = "rejected_no_hawor_prompt"
                continue
            segment_dir = temp_root / f"track_{episode['track_id']:02d}"
            _link_interval_frames(frames_dir, segment_dir, start, end)
            local_seed = seed - start
            segment_masks = np.zeros(
                (end - start + 1, height, width), dtype=bool
            )
            for reverse in (False, True):
                output = _segment_one_pass(
                    predictor, segment_dir, box[None], points[None],
                    np.asarray([local_seed]), reverse=reverse, labels=labels,
                )
                for local_index, value in output.items():
                    segment_masks[local_index] |= value[0]
            for local_index in range(len(segment_masks)):
                segment_masks[local_index] = _component_near_grasp(
                    segment_masks[local_index], hand_data, args.side,
                    start + local_index,
                )
            masks[start:end + 1] = segment_masks
            track_ids[start:end + 1] = episode["track_id"]
            areas = segment_masks.reshape(len(segment_masks), -1).sum(axis=1)
            clean_score = areas.astype(np.float64)
            if arm_masks is not None:
                overlap = np.zeros(len(segment_masks), dtype=np.float64)
                for local_index, mask in enumerate(segment_masks):
                    arm = np.asarray(arm_masks[start + local_index], dtype=bool)
                    if arm.shape != mask.shape:
                        arm = cv2.resize(
                            arm.astype(np.uint8), (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    overlap[local_index] = (mask & arm).sum() / max(mask.sum(), 1)
                clean_score *= np.clip(1.0 - overlap, 0.05, 1.0)
            clean_local = int(np.argmax(clean_score))
            episode.update({
                "status": "tracked",
                "frames_with_mask": int((areas > 0).sum()),
                "median_area_px": int(np.median(areas)),
                "max_area_px": int(areas.max(initial=0)),
                "clean_reference_frame": int(start + clean_local),
            })
            print(
                f"[track {episode['track_id']}] {start}:{end} seed={seed} "
                f"clean={start + clean_local} median={np.median(areas):.0f}",
                flush=True,
            )
            del segment_masks
            torch.cuda.empty_cache()
    finally:
        masks.flush()
        for data in hand_data.values():
            data.close()
        if not args.keep_frames:
            shutil.rmtree(temp_root, ignore_errors=True)
            shutil.rmtree(frames_dir, ignore_errors=True)
        torch.cuda.empty_cache()

    np.save(output_dir / "object_track_id.npy", track_ids)
    np.savez_compressed(
        output_dir / "haco_episode_signal.npz",
        contact_density=density,
        smoothed_density=smoothed,
    )
    write_debug_video(
        video_path,
        np.load(mask_path, mmap_mode="r"),
        track_ids,
        output_dir / "video_sam2_auto_object_tracks.mp4",
    )
    report = {
        "schema_version": 1,
        "method": "full-sequence HaCo peak discovery + HaWoR prompts + bidirectional SAM2",
        "annotation_free": True,
        "frame_count": frame_count,
        "episodes": episodes,
        "parameters": {
            "smooth_sigma": args.smooth_sigma,
            "min_peak_distance": args.min_peak_distance,
            "peak_prominence": args.peak_prominence,
            "max_objects": args.max_objects,
        },
        "outputs": {
            "modal_mask": str(mask_path.resolve()),
            "track_ids": str((output_dir / "object_track_id.npy").resolve()),
            "debug_video": str((output_dir / "video_sam2_auto_object_tracks.mp4").resolve()),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[ok] {output_dir}")


if __name__ == "__main__":
    main()
