"""HaCo contact -> the interaction-object spec that SAM2 tracking needs.

``segment_interaction_objects.py`` takes a JSON listing every grasp: when it
starts and ends, one seed frame, a box around the object and a few point
prompts. That file was written by hand for each video -- someone scrubbed the
clip, found each grasp and typed pixel coordinates. It is the second per-video
manual asset, after the forced-object mask.

HaCo already knows all of it. Per-finger contact state gives the intervals, the
frame with the strongest grip is a good seed, and the contact vertices project
onto the object being held, which bounds it. What HaCo cannot say is which
pixels are object rather than hand, so positive points are picked inside the box
from pixels the human mask does not claim and whose colour departs from the
surface the scene rests on; negatives come from the human mask itself.

Output: JSON in the schema ``segment_interaction_objects.py --segments_json``
expects, with a ``source`` field recording that it was generated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "contact_estimation"))
from aggregate_finger_contact import FINGERS, finger_labels  # noqa: E402


def episodes(state: np.ndarray, min_fingers: int, min_frames: int,
             bridge: int) -> list[tuple[int, int]]:
    """Contiguous runs where enough non-thumb fingers report contact.

    The thumb is ignored for detection: it rests against something during most
    of a reach, so including it merges every grasp into one run.
    """
    held = state[:, 1:].sum(axis=1) >= min_fingers
    runs, start = [], None
    for t, flag in enumerate(held):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            runs.append((start, t - 1))
            start = None
    if start is not None:
        runs.append((start, len(held) - 1))

    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] - 1 <= bridge:
            merged[-1] = (merged[-1][0], run[1])
        else:
            merged.append(run)
    return [r for r in merged if r[1] - r[0] + 1 >= min_frames]


def project(verts: np.ndarray, focal: float, width: int, height: int
            ) -> tuple[np.ndarray, np.ndarray]:
    z = verts[:, 2]
    forward = z > 1e-3
    verts, z = verts[forward], z[forward]
    u = np.round(focal * verts[:, 0] / z + width / 2.0).astype(np.int64)
    v = np.round(focal * verts[:, 1] / z + height / 2.0).astype(np.int64)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[inside], v[inside]


def object_points(frame: np.ndarray, box: tuple[int, int, int, int],
                  human: np.ndarray, count: int, colour_margin: float,
                  contact_uv: tuple[np.ndarray, np.ndarray] | None = None
                  ) -> list[list[int]]:
    """Points inside *box* that read as object rather than hand or table.

    The support surface is measured in a ring just outside the box rather than
    at the frame border, which may show a wall or floor instead of the table the
    objects sit on. Nothing here assumes the table is white -- only that the
    object differs from whatever it rests on, which is what makes it visible.
    """
    x0, y0, x1, y1 = box
    height, width = human.shape
    pad = 60
    ring = np.zeros((height, width), dtype=bool)
    ring[max(0, y0 - pad):min(height, y1 + pad + 1),
         max(0, x0 - pad):min(width, x1 + pad + 1)] = True
    ring[y0:y1 + 1, x0:x1 + 1] = False
    ring &= ~human
    if not ring.any():
        return []
    surface = np.median(frame[ring].astype(np.float32), axis=0)

    crop = frame[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    distance = np.linalg.norm(crop - surface, axis=2)
    candidate = (distance > colour_margin) & ~human[y0:y1 + 1, x0:x1 + 1]
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN,
                                 np.ones((5, 5), np.uint8)).astype(bool)
    if not candidate.any():
        return []

    labelled = cv2.connectedComponents(candidate.astype(np.uint8))[1]
    counts = np.bincount(labelled.ravel())
    counts[0] = 0

    # Prefer the blob the fingers are on. Inside a box drawn around a grasp
    # there is often more non-table colour behind the object than in it -- a
    # dish rack, another item on the bench -- and taking the largest blob then
    # seeds SAM2 on the background, which tracks the wrong thing for the whole
    # interval. Contact says which blob is held.
    chosen = None
    if contact_uv is not None and len(contact_uv[0]):
        u, v = contact_uv
        inside = (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)
        if inside.any():
            hits = np.bincount(
                labelled[np.clip(v[inside] - y0, 0, labelled.shape[0] - 1),
                         np.clip(u[inside] - x0, 0, labelled.shape[1] - 1)],
                minlength=len(counts))
            hits[0] = 0
            if hits.max() > 0:
                chosen = int(np.argmax(hits))
    if chosen is None:
        chosen = int(np.argmax(counts))
    component = labelled == chosen

    # Prompt well inside the object. A point on the boundary is ambiguous
    # between the object and the hand holding it, and SAM2 answers accordingly.
    depth = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    picks = []
    for _ in range(count):
        y, x = np.unravel_index(int(np.argmax(depth)), depth.shape)
        if depth[y, x] <= 2.0:
            break
        picks.append([int(x + x0), int(y + y0)])
        cv2.circle(depth, (int(x), int(y)), int(max(12, depth[y, x])), 0, -1)
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact_dir", type=Path, required=True,
                        help="Holds finger_contact.npz from "
                             "aggregate_finger_contact.py.")
    parser.add_argument("--hawor_npz", type=Path, required=True)
    parser.add_argument("--human_mask", type=Path, required=True,
                        help="(T, H, W) bool from segment_arms.py.")
    parser.add_argument("--frames_dir", type=Path, required=True)
    parser.add_argument("--frame_glob", default="rgb_frame*.jpg")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--min_fingers", type=int, default=2,
                        help="Non-thumb fingers that must report contact.")
    parser.add_argument("--min_frames", type=int, default=8,
                        help="Shorter runs are brushes, not grasps.")
    parser.add_argument("--bridge_frames", type=int, default=6,
                        help="Gaps this short are closed; a grip that flickers "
                             "for a few frames is still one grasp.")
    parser.add_argument("--lead_frames", type=int, default=30,
                        help="Start each tracked interval this many frames "
                             "before contact. The object has to exist as a "
                             "layer while the hand is still closing on it, or "
                             "the compositor has nothing to draw over the "
                             "approaching robot and the fingers cross in front.")
    parser.add_argument("--trail_frames", type=int, default=8,
                        help="Keep tracking this long after contact ends, so "
                             "the object does not pop as the hand lets go.")
    parser.add_argument("--margin_px", type=int, default=45,
                        help="Grown around the projected contact points. The "
                             "object extends well past where fingers touch it.")
    parser.add_argument("--positive_points", type=int, default=2)
    parser.add_argument("--negative_points", type=int, default=2)
    parser.add_argument("--colour_margin", type=float, default=45.0,
                        help="Distance from the support-surface colour before a "
                             "pixel counts as object.")
    parser.add_argument("--overrides", type=Path, default=None,
                        help="JSON mapping a segment name to fields that "
                             "replace the generated ones, e.g. "
                             '{\"object_04\": {\"box\": [...], '
                             '\"positive_points\": [[x, y]]}}. Colour cannot '
                             "separate an object from a table it matches, so "
                             "those few need prompts by hand; keeping them in "
                             "their own file means regenerating does not "
                             "silently drop the correction.")
    args = parser.parse_args()

    hawor = np.load(args.hawor_npz)
    side_idx = 0 if args.side == "left" else 1
    valid = hawor["valid"][side_idx]
    focal = float(hawor["img_focal"])
    verts = hawor[f"verts_{args.side}"]

    contact = np.load(args.contact_dir / "finger_contact.npz")
    state = contact["state"][:, side_idx, :]
    score = contact["score"][:, side_idx, :]

    human = np.load(args.human_mask, mmap_mode="r")
    frame_count, height, width = human.shape
    frames = sorted(args.frames_dir.glob(args.frame_glob))

    labels = finger_labels(hawor[f"joints_{args.side}"][valid].mean(axis=0),
                           verts[valid].mean(axis=0))
    non_thumb = labels > 0

    runs = episodes(state[:frame_count], args.min_fingers, args.min_frames,
                    args.bridge_frames)
    segments = []
    for index, (start, end) in enumerate(runs):
        seed = start + int(np.argmax(score[start:end + 1, 1:].sum(axis=1)))
        # Widen the tracked interval around the contact. SAM2 propagates both
        # ways from the seed, so this costs only tracking time, and the seed
        # stays where the grip is strongest.
        tracked_start = max(0, start - args.lead_frames)
        tracked_end = min(frame_count - 1, end + args.trail_frames)
        if index and segments:
            tracked_start = max(tracked_start, segments[-1]["end_frame"] + 1)
        u, v = project(verts[seed][non_thumb], focal, width, height)
        if not len(u):
            print(f"[warn] f{start}-{end}: no contact vertex projects into "
                  f"frame, skipped")
            continue
        box = (max(0, int(u.min()) - args.margin_px),
               max(0, int(v.min()) - args.margin_px),
               min(width - 1, int(u.max()) + args.margin_px),
               min(height - 1, int(v.max()) + args.margin_px))

        frame = cv2.imread(str(frames[seed]))
        human_seed = np.asarray(human[seed], dtype=bool)
        positive = object_points(frame, box, human_seed, args.positive_points,
                                 args.colour_margin, contact_uv=(u, v))
        ys, xs = np.nonzero(human_seed[box[1]:box[3] + 1, box[0]:box[2] + 1])
        negative = []
        if len(xs):
            picks = np.linspace(0, len(xs) - 1, args.negative_points + 2)
            negative = [[int(xs[int(i)] + box[0]), int(ys[int(i)] + box[1])]
                        for i in picks[1:-1]]

        segments.append({
            "name": f"object_{index + 1:02d}",
            "start_frame": int(tracked_start),
            "end_frame": int(tracked_end),
            "contact_frames": [int(start), int(end)],
            "seed_frame": int(seed),
            "box": [int(c) for c in box],
            "positive_points": positive,
            "negative_points": negative,
        })
        print(f"[seg] {segments[-1]['name']}: track f{tracked_start}-{tracked_end} "
              f"(contact f{start}-{end}) seed={seed} box={box} "
              f"+{len(positive)} -{len(negative)}")

    if args.overrides is not None:
        edits = json.loads(args.overrides.read_text(encoding="utf-8"))
        for segment in segments:
            patch = edits.get(segment["name"])
            if patch:
                segment.update(patch)
                print(f"[override] {segment['name']}: "
                      f"{', '.join(sorted(patch))}")
        unknown = set(edits) - {s["name"] for s in segments}
        for name in sorted(unknown):
            print(f"[warn] override for {name!r} matches no segment")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "source": "build_object_segments_from_contact.py",
        "notes": (f"HaCo contact intervals, {args.side} hand, "
                  f"min_fingers={args.min_fingers}"),
        "segments": segments,
    }, indent=2))
    print(f"[ok] {args.output}  segments={len(segments)}")


if __name__ == "__main__":
    main()
