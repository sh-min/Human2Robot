"""Export the object pixels that must stay behind the robot on each frame.

The compositor splits robot pixels against a single depth plane per frame, and
that plane comes from the grasping hand, so it describes the object being
manipulated and nothing else.  Every pixel of the object layer inherits that
ordering, which puts two kinds of pixel in front of the robot where they do not
belong:

1. *Amodal completions.*  `complete_occluded_objects.py` rebuilds the part of an
   object the human hand was covering.  That is exactly the region the robot
   hand now occupies, so a fabricated completion must never be drawn over it —
   it reads as the robot being sliced by the object it is holding.
2. *Static objects nobody is holding.*  A sponge lying on the table is restored
   on every frame, because the human inpainting mask erases it whenever an arm
   sweeps above.  It still has to be restored, just underneath the robot.

Both are unions of masks the earlier stages already produce, and the result is
written for the compositor's ``--behind_robot_object_mask``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _component_hulls(visible: np.ndarray, min_area: int = 200) -> np.ndarray:
    """Union of the convex hull of every visible object component.

    Completion *inside* an object's own hull is the object closing back over
    the fingers that grip it — the cup rim the four fingers curl behind.
    Completion reaching *outside* the hull is the smear left where the human
    arm used to be, which is where the robot arm now is.  The first must stay
    in front, the second must go behind, and the hull separates them without a
    per-object distance threshold.
    """
    hull = np.zeros(visible.shape, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        visible.astype(np.uint8), 8
    )
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            continue
        contours, _ = cv2.findContours(
            (labels == label).astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            cv2.drawContours(hull, [cv2.convexHull(contour)], -1, 1, -1)
    return hull.astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object_mask", type=Path, required=True,
                        help="Refined object mask the compositor draws.")
    parser.add_argument("--modal_mask", type=Path, required=True,
                        help="Object mask before amodal completion; whatever is "
                             "in the refined mask but not here was hidden by "
                             "the human hand and is a completion.")
    parser.add_argument("--static_mask", type=Path, default=None,
                        help="Optional mask of a static object that is only an "
                             "interaction object inside its own segment.")
    parser.add_argument("--segments_json", type=Path, default=None)
    parser.add_argument("--static_object_name", default=None,
                        help="Segment whose interaction interval keeps "
                             "--static_mask in front of the robot.")
    parser.add_argument("--trust_completion_segments", default="",
                        help="Comma-separated segments whose completion is kept "
                             "in front of the robot for the duration of their "
                             "own grasp. Use where the robot's fingers are "
                             "rendered outside the object's visible silhouette, "
                             "so only the completion can hide them.")
    parser.add_argument("--keep_completion_in_hull", action="store_true",
                        help="Keep completion that lies inside the convex hull "
                             "of the object component it belongs to. Without "
                             "it a grasped object loses the silhouette its own "
                             "fingers grip into, and the robot shows through.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.static_mask is not None and (args.segments_json is None
                                         or args.static_object_name is None):
        raise SystemExit("--static_mask needs --segments_json and "
                         "--static_object_name")

    refined = np.load(args.object_mask, mmap_mode="r")
    modal = np.load(args.modal_mask, mmap_mode="r")
    frame_count = min(len(refined), len(modal))

    all_segments = {}
    if args.segments_json is not None:
        all_segments = {item["name"]: item for item in json.loads(
            args.segments_json.read_text(encoding="utf-8"))["segments"]}
    trusted = []
    for name in filter(None, (n.strip() for n in
                              args.trust_completion_segments.split(","))):
        if name not in all_segments:
            raise SystemExit(f"unknown segment: {name}")
        span = (int(all_segments[name]["start_frame"]),
                int(all_segments[name]["end_frame"]))
        trusted.append(span)
        print(f"[info] {name}: completion trusted over frames {span[0]}-{span[1]}")

    static = None
    hold_start = hold_end = -1
    if args.static_mask is not None:
        static = np.load(args.static_mask, mmap_mode="r")
        frame_count = min(frame_count, len(static))
        segments = {item["name"]: item for item in json.loads(
            args.segments_json.read_text(encoding="utf-8"))["segments"]}
        if args.static_object_name not in segments:
            raise SystemExit(f"unknown segment: {args.static_object_name}")
        segment = segments[args.static_object_name]
        hold_start = int(segment["start_frame"])
        hold_end = int(segment["end_frame"])
        print(f"[info] {args.static_object_name} held over frames "
              f"{hold_start}-{hold_end}; behind the robot elsewhere")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=bool, shape=(frame_count,) + refined.shape[1:]
    )
    completion_px = static_px = 0
    for frame_idx in range(frame_count):
        frame = np.asarray(refined[frame_idx], dtype=bool)
        visible = np.asarray(modal[frame_idx], dtype=bool)
        held = None
        if static is not None:
            held = np.asarray(static[frame_idx], dtype=bool)
            visible = visible | held
        if any(start <= frame_idx <= end for start, end in trusted):
            behind = np.zeros_like(frame)
        else:
            behind = frame & ~visible
            if args.keep_completion_in_hull and behind.any():
                behind &= ~_component_hulls(visible)
        completion_px += int(behind.sum())
        if held is not None and not (hold_start <= frame_idx <= hold_end):
            loose = frame & held
            static_px += int((loose & ~behind).sum())
            behind = behind | loose
        result[frame_idx] = behind
    result.flush()
    print(f"[info] amodal-completion px={completion_px}, "
          f"unheld-static px={static_px}, frames={frame_count}")
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
