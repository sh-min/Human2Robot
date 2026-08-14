"""Paste a grasped object from a clean reference frame instead of smearing it.

Nearest-texture filling reconstructs an occluded object from whatever pixels
touch the hole, which reads as a smear once the hole is hand-sized. These
objects are rigid and fully visible in other frames, so a clean frame is warped
onto the current one instead:

    reference silhouette --(similarity from mask moments, refined by ECC)-->
    current silhouette

Only the covered part is replaced; genuinely visible object pixels stay as
shot. The silhouette itself is the object's convex hull, smoothed, and clipped
to the robot so it never spills onto the background.

Also writes the force-front mask for those silhouettes, with the rendered thumb
carved out so a power grasp keeps the thumb in front.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from refine_interaction_object_masks import _track_interval


def _smooth_hull(mask: np.ndarray, smooth: int, min_area: int = 200) -> np.ndarray:
    """Convex hull per component, with the corners taken off.

    A raw hull gives a cup straight edges where its wall is round; closing then
    opening with an elliptical kernel restores a rounded silhouette.
    """
    hull = np.zeros(mask.shape, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            continue
        contours, _ = cv2.findContours((labels == label).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(hull, [cv2.convexHull(contour)], -1, 1, -1)
    if smooth > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * smooth + 1,) * 2)
        hull = cv2.morphologyEx(hull, cv2.MORPH_OPEN, kernel)
        hull = cv2.GaussianBlur(hull.astype(np.float32), (0, 0), smooth / 2.0)
        hull = (hull > 0.5).astype(np.uint8)
    return hull.astype(bool)


def _moments_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Similarity transform mapping the *src* mask onto the *dst* mask.

    Built from area (scale), centroid (translation) and the principal axis
    (rotation), which is enough of a starting point for ECC to refine.
    """
    ys, xs = np.nonzero(src)
    yd, xd = np.nonzero(dst)
    if len(xs) < 50 or len(xd) < 50:
        return None
    src_c = np.array([xs.mean(), ys.mean()])
    dst_c = np.array([xd.mean(), yd.mean()])
    scale = float(np.sqrt(len(xd) / max(1, len(xs))))

    def axis(x, y):
        pts = np.stack([x - x.mean(), y - y.mean()])
        cov = pts @ pts.T / max(1, pts.shape[1])
        values, vectors = np.linalg.eigh(cov)
        return vectors[:, int(np.argmax(values))]

    a_src, a_dst = axis(xs, ys), axis(xd, yd)
    angle = np.arctan2(a_dst[1], a_dst[0]) - np.arctan2(a_src[1], a_src[0])
    if angle > np.pi / 2:
        angle -= np.pi
    elif angle < -np.pi / 2:
        angle += np.pi
    cos, sin = np.cos(angle) * scale, np.sin(angle) * scale
    matrix = np.array([[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=np.float32)
    matrix[:, 2] = dst_c - matrix[:, :2] @ src_c
    return matrix


def _refine_ecc(src_gray, dst_gray, matrix, iterations=60):
    try:
        _, refined = cv2.findTransformECC(
            dst_gray, src_gray, matrix.copy(), cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-4),
            None, 5,
        )
        return refined
    except cv2.error:
        return matrix


def _match_exposure(patch, target_pixels):
    """Scale the pasted patch to the brightness of the object as shot."""
    if target_pixels.size < 200:
        return patch
    src_mean = patch.reshape(-1, 3).mean(0) + 1e-6
    dst_mean = target_pixels.reshape(-1, 3).mean(0)
    return np.clip(patch * (dst_mean / src_mean), 0, 255)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_video", type=Path, required=True,
                        help="Clean source RGB (video_L.mp4): the reference "
                             "frames are taken from here, not from a composite.")
    parser.add_argument("--object_source_video", type=Path, required=True,
                        help="Object-layer video the compositor reads.")
    parser.add_argument("--object_mask", type=Path, required=True)
    parser.add_argument("--modal_mask", type=Path, required=True)
    parser.add_argument("--robot_mask", type=Path, required=True)
    parser.add_argument("--thumb_mask", type=Path, default=None)
    parser.add_argument("--segments_json", type=Path, required=True)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--reference_frames", default="",
                        help="Comma-separated per-segment reference frame. "
                             "Only used with --reference_mode fixed.")
    parser.add_argument("--reference_mode", choices=("nearest", "fixed"),
                        default="nearest",
                        help="nearest re-picks the closest clean frame for every "
                             "frame, which tracks an object whose pose drifts "
                             "through the grasp; fixed warps one frame onto the "
                             "whole segment.")
    parser.add_argument("--clean_area_frac", type=float, default=0.6,
                        help="A frame counts as clean when its robot-free "
                             "silhouette is at least this fraction of the "
                             "segment's largest one.")
    parser.add_argument("--hull_smooth", type=int, default=5)
    parser.add_argument("--no_extend", action="store_true",
                        help="Repaint inside the tracked silhouette only. A "
                             "thin object the hand nearly covers has too little "
                             "mask left for a hull to be its shape.")
    parser.add_argument("--feather", type=float, default=1.5)
    parser.add_argument("--output_video", type=Path, required=True)
    parser.add_argument("--output_force_mask", type=Path, required=True)
    args = parser.parse_args()

    refined = np.load(args.object_mask, mmap_mode="r")
    modal = np.load(args.modal_mask, mmap_mode="r")
    robot = np.load(args.robot_mask, mmap_mode="r")
    thumb = np.load(args.thumb_mask, mmap_mode="r") if args.thumb_mask else None
    segments = {item["name"]: item for item in json.loads(
        args.segments_json.read_text(encoding="utf-8"))["segments"]}
    names = [n.strip() for n in args.segments.split(",") if n.strip()]
    overrides = [int(v) for v in args.reference_frames.split(",") if v.strip()]

    source = cv2.VideoCapture(str(args.source_video))
    frames = []
    while True:
        ok, frame = source.read()
        if not ok:
            break
        frames.append(frame)
    source.release()
    frame_count = min(len(frames), len(refined), len(modal), len(robot))

    tracks, spans, refs = [], [], []
    for position, name in enumerate(names):
        if name not in segments:
            raise SystemExit(f"unknown segment: {name}")
        track = _track_interval(refined, segments[name])
        start = int(segments[name]["start_frame"])
        end = min(int(segments[name]["end_frame"]), frame_count - 1)
        areas = {}
        for idx in range(max(0, start - 40), end + 1):
            visible = (np.asarray(track[idx], dtype=bool)
                       & np.asarray(modal[idx], dtype=bool)
                       & ~np.asarray(robot[idx], dtype=bool))
            areas[idx] = int(visible.sum())
        largest = max(areas.values()) if areas else 0
        clean = [idx for idx, area in areas.items()
                 if area >= args.clean_area_frac * largest and area > 0]
        if args.reference_mode == "fixed":
            ref = (overrides[position] if position < len(overrides)
                   else max(areas, key=areas.get))
            clean = [ref]
        tracks.append(track)
        spans.append((start, end))
        refs.append(clean)
        print(f"[info] {name}: frames {start}-{end}, {len(clean)} clean "
              f"reference frames ({clean[:3]}{'...' if len(clean) > 3 else ''})",
              flush=True)

    capture = cv2.VideoCapture(str(args.object_source_video))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
    height, width = frames[0].shape[:2]
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output_video),
                             cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    force = np.lib.format.open_memmap(args.output_force_mask, mode="w+",
                                      dtype=bool, shape=(frame_count, height, width))

    references = []
    for track, clean in zip(tracks, refs):
        entry = {}
        for ref in clean:
            mask = (np.asarray(track[ref], dtype=bool)
                    & np.asarray(modal[ref], dtype=bool))
            entry[ref] = (frames[ref], mask,
                          cv2.cvtColor(frames[ref], cv2.COLOR_BGR2GRAY))
        references.append(entry)

    pasted_px = 0
    for idx in range(frame_count):
        ok, frame = capture.read()
        if not ok:
            break
        robot_frame = np.asarray(robot[idx], dtype=bool)
        visible_all = np.asarray(modal[idx], dtype=bool)
        frame_force = np.zeros((height, width), dtype=bool)
        for track, (start, end), entry in zip(tracks, spans, references):
            if not start <= idx <= end or not entry:
                continue
            ref = min(entry, key=lambda candidate: abs(candidate - idx))
            ref_img, ref_mask, ref_gray = entry[ref]
            component = np.asarray(track[idx], dtype=bool)
            if component.sum() < 200:
                continue
            if args.no_extend:
                silhouette = component
            else:
                silhouette = (_smooth_hull(component, args.hull_smooth)
                              & robot_frame) | component
            kept = component & visible_all & ~robot_frame      # trustworthy pixels
            missing = silhouette & ~kept
            frame_force |= silhouette
            if not missing.any():
                continue
            matrix = _moments_similarity(ref_mask, silhouette)
            if matrix is None:
                continue
            matrix = _refine_ecc(ref_gray,
                                 cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), matrix)
            warped = cv2.warpAffine(ref_img, matrix, (width, height),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
            warped = _match_exposure(warped.astype(np.float32), frame[kept])
            alpha = cv2.GaussianBlur(missing.astype(np.float32), (0, 0),
                                     args.feather)[..., None]
            frame[:] = np.clip(alpha * warped + (1 - alpha) * frame, 0,
                               255).astype(np.uint8)
            pasted_px += int(missing.sum())
        if thumb is not None and idx < len(thumb):
            frame_force &= ~np.asarray(thumb[idx], dtype=bool)
        force[idx] = frame_force
        writer.write(frame)
    capture.release()
    writer.release()
    force.flush()
    print(f"[info] pasted {pasted_px} px from reference frames")
    print(f"[ok] wrote {args.output_video}")
    print(f"[ok] wrote {args.output_force_mask}")


if __name__ == "__main__":
    main()
