"""Repaint an object's completed pixels from that object's own visible texture.

`complete_occluded_objects.py` fills a completion from the nearest visible
*interaction-object* pixel, and the merged mask holds every object at once.  A
mug gripped in front of the steel tray therefore borrows the tray's grey, and
once the compositor is allowed to draw that completion over the fingers it
hides, the grey band reads worse than the penetration it fixes.  Filling per
connected component keeps each object's own colour.

Only the requested segments' frames are rewritten; every other frame is copied
through untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from complete_occluded_objects import _nearest_texture
from refine_interaction_object_masks import _track_interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object_source_video", type=Path, required=True)
    parser.add_argument("--object_mask", type=Path, required=True,
                        help="Refined (completed) object mask.")
    parser.add_argument("--modal_mask", type=Path, required=True,
                        help="Object mask before completion; the donor texture.")
    parser.add_argument("--segments_json", type=Path, required=True)
    parser.add_argument("--segments", required=True,
                        help="Comma-separated segment names to repaint.")
    parser.add_argument("--min_area", type=int, default=200)
    parser.add_argument("--skip_mask", type=Path, default=None,
                        help="Completion the compositor never draws "
                             "(behind_robot_objects). Repainting it "
                             "cannot help and can only spill onto a "
                             "neighbour the amodal hull reached.")
    parser.add_argument("--protect_mask", type=Path, default=None,
                        help="Pixels never repainted, e.g. the restored sponge. "
                             "A conservative amodal hull can reach across a "
                             "neighbouring object, and those pixels are real "
                             "texture the compositor still draws.")
    parser.add_argument("--reject_warm", action="store_true",
                        help="Also treat pixels warmer than the object "
                             "(R > B + margin) as missing. The mug's mask keeps "
                             "a thin fringe of shadowed skin, and because that "
                             "fringe lines the completion it wins the nearest-"
                             "texture vote and paints the whole fill brown.")
    parser.add_argument("--warm_margin", type=int, default=8)
    parser.add_argument("--donor_erode", type=int, default=0,
                        help="Shrink the donor by this many pixels first. The "
                             "boundary ring is where the mask leaks onto the "
                             "hand, and it is exactly the ring the nearest-"
                             "texture search reaches first.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    refined = np.load(args.object_mask, mmap_mode="r")
    modal = np.load(args.modal_mask, mmap_mode="r")
    protect = (np.load(args.protect_mask, mmap_mode="r")
               if args.protect_mask is not None else None)
    skip = (np.load(args.skip_mask, mmap_mode="r")
            if args.skip_mask is not None else None)
    segments = {item["name"]: item for item in json.loads(
        args.segments_json.read_text(encoding="utf-8"))["segments"]}
    spans, tracks = [], []
    for name in filter(None, (n.strip() for n in args.segments.split(","))):
        if name not in segments:
            raise SystemExit(f"unknown segment: {name}")
        spans.append((int(segments[name]["start_frame"]),
                      int(segments[name]["end_frame"])))
        # Only this object's own component is repainted.  A frame holds several
        # objects, and a rule tuned for one of them (a cool-coloured mug) would
        # otherwise repaint a red snack box out of existence.
        tracks.append(_track_interval(refined, segments[name]))
        print(f"[info] {name}: repainting frames {spans[-1][0]}-{spans[-1][1]}")

    capture = cv2.VideoCapture(str(args.object_source_video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.object_source_video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"FFV1"),
                             fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"cannot write {args.output}")

    frame_idx = repainted_px = repainted_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if (frame_idx < len(refined) and frame_idx < len(modal)
                and any(start <= frame_idx <= end for start, end in spans)):
            whole = np.asarray(refined[frame_idx], dtype=bool)
            visible = np.asarray(modal[frame_idx], dtype=bool)
            if protect is not None and frame_idx < len(protect):
                visible = visible | np.asarray(protect[frame_idx], dtype=bool)
            warm = None
            if args.reject_warm:
                warm = (frame[..., 2].astype(np.int16)
                        - frame[..., 0].astype(np.int16)) > args.warm_margin
            changed = 0
            for track, (start, end) in zip(tracks, spans):
                if not start <= frame_idx <= end:
                    continue
                component = np.asarray(track[frame_idx], dtype=bool) & whole
                if component.sum() < args.min_area:
                    continue
                donor = component & visible
                missing = component & ~visible
                if warm is not None:
                    donor = donor & ~warm
                    missing = missing | (component & warm)
                if skip is not None and frame_idx < len(skip):
                    missing = missing & ~np.asarray(skip[frame_idx], dtype=bool)
                if args.donor_erode > 0 and donor.any():
                    radius = args.donor_erode
                    donor = cv2.erode(
                        donor.astype(np.uint8),
                        cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2
                        ), iterations=1,
                    ).astype(bool) & ~missing
                if not donor.any() or not missing.any():
                    continue
                filled = _nearest_texture(frame, donor, missing)
                frame[missing] = filled[missing]
                changed += int(missing.sum())
            if changed:
                repainted_px += changed
                repainted_frames += 1
        writer.write(frame)
        frame_idx += 1
    capture.release()
    writer.release()
    print(f"[info] repainted {repainted_px} px over {repainted_frames} frames "
          f"of {frame_idx}")
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
