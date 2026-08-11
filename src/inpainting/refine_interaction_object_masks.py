"""Extract contact-specific masks from a merged interaction-object track.

The Chocobi mask is used as a final foreground interior so robot links cannot
render through the visible box.  The sponge mask is augmented with conservative
green/yellow colour cues, restoring source sponge pixels that were removed by
the human inpainting mask.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _component(mask: np.ndarray, previous: np.ndarray | None,
               seed: tuple[float, float] | None = None) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), 8
    )
    if count <= 1:
        return np.asarray(mask, dtype=bool)
    score = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    score[0] = -np.inf
    if previous is not None and previous.any():
        ids, overlap = np.unique(labels[previous], return_counts=True)
        for label, value in zip(ids, overlap):
            if label:
                score[label] += 1000.0 * float(value)
    if seed is not None:
        x, y = seed
        label = labels[int(np.clip(round(y), 0, labels.shape[0] - 1)),
                       int(np.clip(round(x), 0, labels.shape[1] - 1))]
        if label:
            score[label] += 1e12
    return labels == int(np.argmax(score))


def _track_interval(merged: np.ndarray, segment: dict) -> np.ndarray:
    start, end, seed_frame = (int(segment[k]) for k in
                              ("start_frame", "end_frame", "seed_frame"))
    track = np.zeros_like(merged, dtype=bool)
    points = segment.get("positive_points") or []
    seed_point = tuple(points[0]) if points else None
    seed = _component(merged[seed_frame], None, seed_point)
    track[seed_frame] = seed
    previous = seed
    for frame_idx in range(seed_frame + 1, end + 1):
        current = _component(merged[frame_idx], previous)
        track[frame_idx] = current
        if current.any():
            previous = current
    previous = seed
    for frame_idx in range(seed_frame - 1, start - 1, -1):
        current = _component(merged[frame_idx], previous)
        track[frame_idx] = current
        if current.any():
            previous = current
    return track


def _fill_small_holes(mask: np.ndarray, max_area: float = 350.0) -> np.ndarray:
    """Restore dark printed details without filling a hand-sized occlusion."""
    result = np.asarray(mask, dtype=np.uint8).copy()
    contours, hierarchy = cv2.findContours(
        result, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return result.astype(bool)
    for contour, relation in zip(contours, hierarchy[0]):
        if relation[3] >= 0 and cv2.contourArea(contour) <= max_area:
            cv2.drawContours(result, [contour], -1, 1, thickness=cv2.FILLED)
    return result.astype(bool)


def _extract_visible_sponge(
    track: np.ndarray, video: Path, start: int, end: int,
    static_box: tuple[int, int, int, int],
) -> np.ndarray:
    """Keep only source pixels belonging to the green/yellow sponge material.

    SAM's object component can grow onto fingers at close contact.  Colour-based
    extraction avoids restoring those skin pixels after human inpainting.
    """
    result = np.zeros_like(track, dtype=bool)
    cap = cv2.VideoCapture(str(video))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    support_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    x0, y0, x1, y1 = static_box
    static_support = np.zeros(track.shape[1:], dtype=bool)
    static_support[
        max(0, y0 - 30):min(track.shape[1], y1 + 30),
        max(0, x0 - 30):min(track.shape[2], x1 + 30),
    ] = True
    frame_idx = 0
    while frame_idx < len(track):
        ok, frame = cap.read()
        if not ok:
            break
        if track[frame_idx].any():
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(hsv)
            green = (hue >= 32) & (hue <= 100) & (sat >= 25) & (val >= 20)
            yellow = (hue >= 15) & (hue <= 42) & (sat >= 50) & (val >= 65)
            support = cv2.dilate(track[frame_idx].astype(np.uint8),
                                 support_kernel, iterations=1).astype(bool)
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(hsv)
            green = (hue >= 32) & (hue <= 100) & (sat >= 25) & (val >= 20)
            yellow = (hue >= 15) & (hue <= 42) & (sat >= 50) & (val >= 65)
            support = static_support
        if support.any():
            colour = ((green | yellow) & support).astype(np.uint8)
            colour = cv2.morphologyEx(colour, cv2.MORPH_CLOSE, close_kernel)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(colour, 8)
            visible = np.zeros_like(colour, dtype=bool)
            for label in range(1, count):
                if stats[label, cv2.CC_STAT_AREA] >= 12:
                    visible |= labels == label
            result[frame_idx] = _fill_small_holes(visible)
        frame_idx += 1
    cap.release()
    return result


def _preview(video: Path, green: np.ndarray, sponge: np.ndarray,
             output: Path, fps: float) -> None:
    cap = cv2.VideoCapture(str(video))
    height, width = green.shape[1:]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    for frame_idx in range(len(green)):
        ok, frame = cap.read()
        if not ok:
            break
        overlay = frame.copy()
        overlay[green[frame_idx]] = (
            0.35 * overlay[green[frame_idx]] + 0.65 * np.array([40, 40, 240])
        ).astype(np.uint8)
        overlay[sponge[frame_idx]] = (
            0.35 * overlay[sponge[frame_idx]] + 0.65 * np.array([240, 220, 20])
        ).astype(np.uint8)
        writer.write(overlay)
    cap.release()
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--merged_mask", type=Path, required=True)
    parser.add_argument("--segments_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--chocobi_erode", type=int, default=3)
    args = parser.parse_args()

    merged = np.load(args.merged_mask, mmap_mode="r")
    payload = json.loads(args.segments_json.read_text(encoding="utf-8"))
    segments = {item["name"]: item for item in payload["segments"]}
    green = _track_interval(merged, segments["green_snack_box"])
    sponge_segment = segments["sponge"]
    sponge_track = _track_interval(merged, sponge_segment)
    sponge = _extract_visible_sponge(
        sponge_track, args.video, int(sponge_segment["start_frame"]),
        int(sponge_segment["end_frame"]), tuple(sponge_segment["box"]),
    )

    refined = np.asarray(merged, dtype=bool).copy()
    # Replace the original SAM component so close-contact skin cannot survive
    # through the object-protection/compositing path.
    refined &= ~sponge_track
    refined |= sponge
    force = np.zeros_like(green)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max(0, args.chocobi_erode) + 1,) * 2,
    )
    for frame_idx in range(len(force)):
        if green[frame_idx].any():
            force[frame_idx] = cv2.erode(
                green[frame_idx].astype(np.uint8), kernel, iterations=1
            ).astype(bool)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "object_mask_refined.npy", refined)
    np.save(args.output_dir / "chocobi_force_front.npy", force)
    np.save(args.output_dir / "sponge_visible_mask.npy", sponge)
    cap = cv2.VideoCapture(str(args.video))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    cap.release()
    _preview(args.video, green, sponge,
             args.output_dir / "contact_mask_refinement_preview.mp4", fps)
    print(f"[info] chocobi px={int(green.sum())}, force-front px={int(force.sum())}")
    print(
        f"[info] sponge px={int(sponge.sum())}, "
        f"skin/object-track removed={int((sponge_track & ~sponge).sum())}, "
        f"added={int((sponge & ~merged).sum())}"
    )
    print(f"[ok] wrote refined masks to {args.output_dir}")


if __name__ == "__main__":
    main()
