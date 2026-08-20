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
from scipy.ndimage import gaussian_filter1d

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
                  contact_uv: tuple[np.ndarray, np.ndarray] | None = None,
                  hand_projection: np.ndarray | None = None,
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
    if hand_projection is not None:
        candidate &= ~hand_projection[y0:y1 + 1, x0:x1 + 1]
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
            # Contact vertices lie on the hand surface, which has deliberately
            # been removed from ``candidate``.  Therefore an exact label lookup
            # usually reads background (0).  Select the component nearest to
            # the true contact vertices instead.  This is still fully automatic
            # and, unlike choosing the largest coloured blob, does not jump to
            # a rack or another object elsewhere in the box.
            cu = np.clip(u[inside] - x0, 0, labelled.shape[1] - 1)
            cv = np.clip(v[inside] - y0, 0, labelled.shape[0] - 1)
            best = None
            for label in np.flatnonzero(counts):
                if counts[label] < 100:
                    continue
                distance_to_component = cv2.distanceTransform(
                    (labelled != label).astype(np.uint8), cv2.DIST_L2, 5)
                distances = distance_to_component[cv, cu]
                near = float(np.exp(-distances / 24.0).sum())
                # A weak area term breaks ties but cannot let a large remote
                # background component beat a small object at the fingertips.
                component_score = near + 0.05 * np.log1p(counts[label])
                if best is None or component_score > best[0]:
                    best = (component_score, float(distances.min()), int(label))
            if best is not None and best[1] <= 55.0:
                chosen = best[2]
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
    parser.add_argument("--margin_px", type=int, default=100,
                        help="Grown around the projected contact points. The "
                             "object extends well past where fingers touch it.")
    parser.add_argument("--positive_points", type=int, default=2)
    parser.add_argument("--negative_points", type=int, default=8)
    parser.add_argument("--colour_margin", type=float, default=45.0,
                        help="Distance from the support-surface colour before a "
                             "pixel counts as object.")
    parser.add_argument("--contact_threshold", type=float, default=0.35,
                        help="Per-vertex HaCo probability used to localise the "
                             "actual touch surface. This replaces the old "
                             "whole-hand projection.")
    parser.add_argument("--prob_sigma_t", type=float, default=2.0,
                        help="Temporal Gaussian smoothing for HaCo vertex "
                             "probabilities.")
    parser.add_argument("--max_track_frames", type=int, default=96,
                        help="Automatically re-seed longer tracks in chunks "
                             "of at most this size. Applied identically to "
                             "every video.")
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

    # Load the dense HaCo probabilities. The previous implementation used all
    # non-thumb MANO vertices here, even those nowhere near contact; that made
    # prompt boxes depend on the whole hand pose and frequently selected scene
    # background. Numeric filenames are aligned to the source frame index.
    probability = np.zeros((frame_count, verts.shape[1]), dtype=np.float32)
    for path in args.contact_dir.glob("[0-9]*.npz"):
        try:
            frame_index = int(path.stem)
        except ValueError:
            continue
        if frame_index >= frame_count:
            continue
        data = np.load(path)
        key = f"{args.side}_contact_probability"
        if key in data and data[key].shape == probability[frame_index].shape:
            probability[frame_index] = data[key]
    if args.prob_sigma_t > 0:
        probability = gaussian_filter1d(
            probability, args.prob_sigma_t, axis=0, mode="nearest")

    runs = episodes(state[:frame_count], args.min_fingers, args.min_frames,
                    args.bridge_frames)
    segments = []
    for index, (start, end) in enumerate(runs):
        # Widen the tracked interval around the contact. SAM2 propagates both
        # ways from the seed, so this costs only tracking time, and the seed
        # stays where the grip is strongest.
        tracked_start = max(0, start - args.lead_frames)
        tracked_end = min(frame_count - 1, end + args.trail_frames)
        if index and segments:
            tracked_start = max(tracked_start, segments[-1]["end_frame"] + 1)
        if tracked_start > tracked_end:
            continue

        # Fixed-length chunks are a content-independent re-seeding policy.
        # It prevents long propagations from drifting without encoding any
        # video-specific split frame.
        part = 0
        for chunk_start in range(tracked_start, tracked_end + 1,
                                 args.max_track_frames):
            chunk_end = min(tracked_end,
                            chunk_start + args.max_track_frames - 1)
            contact_start = max(start, chunk_start)
            contact_end = min(end, chunk_end)
            if contact_start > contact_end:
                # Lead/trail-only chunks use their frame nearest the genuine
                # contact interval so the seed remains inside the track.
                candidate_frames = np.array([
                    min(max(start, chunk_start), chunk_end)], dtype=np.int64)
            else:
                candidate_frames = np.arange(contact_start, contact_end + 1)
            contact_strength = probability[candidate_frames][:, non_thumb].sum(axis=1)
            seed = int(candidate_frames[int(np.argmax(contact_strength))])
            touching = non_thumb & (probability[seed] >= args.contact_threshold)
            if touching.sum() < 8:
                wanted = np.flatnonzero(non_thumb)
                top = wanted[np.argsort(probability[seed, wanted])[-32:]]
                touching = np.zeros_like(non_thumb)
                touching[top] = True
            u, v = project(verts[seed][touching], focal, width, height)
            if not len(u):
                print(f"[warn] f{chunk_start}-{chunk_end}: no true contact "
                      "vertex projects into frame, skipped")
                continue
            box = (max(0, int(u.min()) - args.margin_px),
                   max(0, int(v.min()) - args.margin_px),
                   min(width - 1, int(u.max()) + args.margin_px),
                   min(height - 1, int(v.max()) + args.margin_px))

            frame = cv2.imread(str(frames[seed]))
            human_seed = np.asarray(human[seed], dtype=bool)
            hand_u, hand_v = project(verts[seed], focal, width, height)
            hand_projection = np.zeros((height, width), dtype=np.uint8)
            hand_projection[hand_v, hand_u] = 1
            hand_projection = cv2.dilate(
                hand_projection, np.ones((9, 9), np.uint8), iterations=1
            ).astype(bool)
            positive = object_points(
                frame, box, human_seed, args.positive_points,
                args.colour_margin, contact_uv=(u, v),
                hand_projection=hand_projection)
            if not positive:
                print(f"[warn] f{chunk_start}-{chunk_end}: no object candidate "
                      "near HaCo contact, skipped")
                continue
            # Explicitly tell SAM that projected MANO pixels are not object.
            # The arm mask often misses fingers exactly where an object
            # occludes them, while MANO still gives a stable hand location.
            # Farthest-point sampling spreads the negatives over the visible
            # hand and is identical for every clip.
            in_box = ((hand_u >= box[0]) & (hand_u <= box[2]) &
                      (hand_v >= box[1]) & (hand_v <= box[3]))
            candidates = np.stack([hand_u[in_box], hand_v[in_box]], axis=1)
            positive_array = np.asarray(positive, dtype=np.float32)
            negative = []
            if len(candidates):
                candidates = np.unique(candidates, axis=0).astype(np.float32)
                # Avoid contradictory labels at a positive prompt.
                gap = np.linalg.norm(
                    candidates[:, None] - positive_array[None], axis=2).min(axis=1)
                candidates = candidates[gap >= 10.0]
            if len(candidates):
                centre = positive_array.mean(axis=0)
                first = int(np.argmin(np.linalg.norm(candidates - centre, axis=1)))
                selected = [first]
                while (len(selected) < args.negative_points and
                       len(selected) < len(candidates)):
                    distance = np.linalg.norm(
                        candidates[:, None] - candidates[selected][None], axis=2
                    ).min(axis=1)
                    distance[selected] = -1
                    selected.append(int(np.argmax(distance)))
                negative = candidates[selected].round().astype(int).tolist()

            part += 1
            segments.append({
                "name": f"object_{index + 1:02d}_part_{part:02d}",
                "start_frame": int(chunk_start),
                "end_frame": int(chunk_end),
                "contact_frames": [int(start), int(end)],
                "seed_frame": int(seed),
                "box": [int(c) for c in box],
                "positive_points": positive,
                "negative_points": negative,
            })
            print(f"[seg] {segments[-1]['name']}: "
                  f"track f{chunk_start}-{chunk_end} "
                  f"(contact f{start}-{end}) seed={seed} box={box} "
                  f"touch={int(touching.sum())} "
                  f"+{len(positive)} -{len(negative)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "source": "build_object_segments_from_contact.py/generalized-v2",
        "notes": (f"HaCo contact intervals, {args.side} hand, "
                  f"min_fingers={args.min_fingers}, no per-video overrides, "
                  f"max_track_frames={args.max_track_frames}"),
        "segments": segments,
    }, indent=2))
    print(f"[ok] {args.output}  segments={len(segments)}")


if __name__ == "__main__":
    main()
