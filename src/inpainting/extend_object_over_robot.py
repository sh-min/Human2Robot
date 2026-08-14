"""Keep a grasped object's silhouette whole where the robot hand covers it.

The human hand and the rendered robot hand never occupy exactly the same
pixels, so a grasped object ends up with a bite taken out of it: the object
mask stops at the human silhouette, and the robot hand is drawn over the gap.
Two things are needed there, and both are produced here:

1. *Shape.*  The object's own convex hull, clipped to the robot mask, extends
   the silhouette over the robot without spilling onto the background.
2. *Colour.*  Those pixels hold robot RGB in the object source video, so they
   are refilled from the object's own visible texture (robot pixels excluded
   from the donor, which is what made earlier fills come out white or brown).

Outputs a new object-source video plus the extended force-front mask; the
compositor draws the object over the robot from those two.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from complete_occluded_objects import _nearest_texture
from refine_interaction_object_masks import _track_interval


def _hull(mask: np.ndarray, min_area: int = 200) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            continue
        contours, _ = cv2.findContours((labels == label).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(out, [cv2.convexHull(contour)], -1, 1, -1)
    return out.astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object_source_video", type=Path, required=True)
    parser.add_argument("--object_mask", type=Path, required=True,
                        help="Refined (completed) object mask.")
    parser.add_argument("--modal_mask", type=Path, required=True,
                        help="Pre-completion mask; the trustworthy texture.")
    parser.add_argument("--robot_mask", type=Path, required=True)
    parser.add_argument("--thumb_mask", type=Path, default=None,
                        help="Rendered thumb; excluded from the force mask so a "
                             "power grasp keeps the thumb in front.")
    parser.add_argument("--segments_json", type=Path, required=True)
    parser.add_argument("--segments", required=True,
                        help="Comma-separated segments to extend.")
    parser.add_argument("--donor_erode", type=int, default=3)
    parser.add_argument("--no_extend", action="store_true",
                        help="Recolour inside the tracked silhouette only, "
                             "leaving its shape alone.")
    parser.add_argument("--reject_warm", action="store_true",
                        help="Drop pixels warmer than the object (R > B + 4) "
                             "from the donor: shadowed skin left in a cool "
                             "object's mask otherwise wins the nearest-texture "
                             "vote.")
    parser.add_argument("--output_video", type=Path, required=True)
    parser.add_argument("--output_force_mask", type=Path, required=True)
    args = parser.parse_args()

    refined = np.load(args.object_mask, mmap_mode="r")
    modal = np.load(args.modal_mask, mmap_mode="r")
    robot = np.load(args.robot_mask, mmap_mode="r")
    thumb = np.load(args.thumb_mask, mmap_mode="r") if args.thumb_mask else None
    segments = {item["name"]: item for item in json.loads(
        args.segments_json.read_text(encoding="utf-8"))["segments"]}

    spans, tracks = [], []
    for name in filter(None, (n.strip() for n in args.segments.split(","))):
        if name not in segments:
            raise SystemExit(f"unknown segment: {name}")
        spans.append((int(segments[name]["start_frame"]),
                      int(segments[name]["end_frame"])))
        tracks.append(_track_interval(refined, segments[name]))
        print(f"[info] {name}: frames {spans[-1][0]}-{spans[-1][1]}", flush=True)

    capture = cv2.VideoCapture(str(args.object_source_video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.object_source_video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = min(len(refined), len(modal), len(robot))
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output_video),
                             cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"cannot write {args.output_video}")
    force = np.lib.format.open_memmap(args.output_force_mask, mode="w+",
                                      dtype=bool, shape=(frame_count, height, width))

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * args.donor_erode + 1,) * 2)
    extended_px = refilled_px = 0
    for idx in range(frame_count):
        ok, frame = capture.read()
        if not ok:
            break
        robot_frame = np.asarray(robot[idx], dtype=bool)
        visible = np.asarray(modal[idx], dtype=bool)
        warm = None
        if args.reject_warm:
            warm = (frame[..., 2].astype(np.int16)
                    - frame[..., 0].astype(np.int16)) > 4
        frame_force = np.zeros((height, width), dtype=bool)
        for track, (start, end) in zip(tracks, spans):
            if not start <= idx <= end:
                continue
            component = np.asarray(track[idx], dtype=bool)
            if component.sum() < 200:
                continue
            # Grow the silhouette over the robot only.
            extended = (component if args.no_extend
                        else (_hull(component) & robot_frame) | component)
            # Robot pixels hold robot RGB in this video, so they cannot be
            # donors; the same goes for skin left inside the mask.
            donor = component & visible & ~robot_frame
            if warm is not None:
                donor &= ~warm
            missing = extended & ~donor
            if args.donor_erode > 0 and donor.any():
                donor = cv2.erode(donor.astype(np.uint8), kernel,
                                  iterations=1).astype(bool) & ~missing
            if donor.any() and missing.any():
                filled = _nearest_texture(frame, donor, missing)
                frame[missing] = filled[missing]
                refilled_px += int(missing.sum())
            extended_px += int((extended & ~component).sum())
            frame_force |= extended
        if thumb is not None and idx < len(thumb):
            frame_force &= ~np.asarray(thumb[idx], dtype=bool)
        force[idx] = frame_force
        writer.write(frame)
    capture.release()
    writer.release()
    force.flush()
    print(f"[info] extended {extended_px} px over the robot, refilled {refilled_px} px")
    print(f"[ok] wrote {args.output_video}")
    print(f"[ok] wrote {args.output_force_mask}")


if __name__ == "__main__":
    main()
