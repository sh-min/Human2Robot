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


def object_component(frame: np.ndarray, box: tuple[int, int, int, int],
                     human: np.ndarray, colour_margin: float,
                     contact_uv: tuple[np.ndarray, np.ndarray] | None = None
                     ) -> tuple[np.ndarray, int] | None:
    """The blob inside *box* that reads as the held object, and its contact hits.

    The support surface is measured in a ring just outside the box rather than
    at the frame border, which may show a wall or floor instead of the table the
    objects sit on. Nothing here assumes the table is white -- only that the
    object differs from whatever it rests on, which is what makes it visible.

    Returns the component in crop coordinates together with the number of
    projected contact points that landed on it. A hit count of zero means the
    caller is looking at a frame where the object is not visibly touched --
    usually because the human mask has swallowed it -- and the component is
    then only a guess from size. Callers should prefer a frame that scores
    above zero rather than trust that guess.
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
        return None
    surface = np.median(frame[ring].astype(np.float32), axis=0)

    crop = frame[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    distance = np.linalg.norm(crop - surface, axis=2)
    candidate = (distance > colour_margin) & ~human[y0:y1 + 1, x0:x1 + 1]
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN,
                                 np.ones((5, 5), np.uint8)).astype(bool)
    if not candidate.any():
        return None

    labelled = cv2.connectedComponents(candidate.astype(np.uint8))[1]
    counts = np.bincount(labelled.ravel())
    counts[0] = 0

    # Prefer the blob the fingers are on. Inside a box drawn around a grasp
    # there is often more non-table colour behind the object than in it -- a
    # dish rack, another item on the bench -- and taking the largest blob then
    # seeds SAM2 on the background, which tracks the wrong thing for the whole
    # interval. Contact says which blob is held.
    chosen, hit_total = None, 0
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
                hit_total = int(hits.sum())
    if chosen is None:
        chosen = int(np.argmax(counts))
    return labelled == chosen, hit_total


def detected_box(detections: dict, frame: int, u: np.ndarray, v: np.ndarray,
                 min_score: float, near_px: float = 60.0
                 ) -> tuple[tuple[int, int, int, int], str, float] | None:
    """The detection the fingers are inside, at the frame nearest *frame*.

    Scored by how many projected contact points a box contains, not by
    detection confidence. Confidence says the model is sure it found *a* mug;
    only the contact points say which object this hand is on. Ties -- a box
    inside another, the mug and the rack it hangs from -- go to the smaller
    box, which is the object rather than the furniture around it.
    """
    if not detections:
        return None
    stride = max(1, int(detections.get("stride", 1)))
    nearest = str(int(round(frame / stride)) * stride)
    if nearest not in detections["frames"]:
        nearest = min(detections["frames"],
                      key=lambda k: abs(int(k) - frame), default=None)
        if nearest is None:
            return None
    best = None
    closest = None
    for entry in detections["frames"][nearest]:
        if entry["score"] < min_score:
            continue
        x0, y0, x1, y1 = entry["box"]
        held = int(((u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)).sum())
        area = max(1, (x1 - x0) * (y1 - y0))
        if held:
            key = (held, -area)
            if best is None or key > best[0]:
                best = (key, (x0, y0, x1, y1), entry["label"], entry["score"])
            continue
        # Nothing inside is not the same as nothing there. Wiping a table puts
        # the sponge flat under the palm, so its box hugs the object while the
        # contact points sit on the hand a little outside it. The nearest box
        # is still the right object; only a box far away is a different one.
        gap = float(np.hypot(np.clip(u, x0, x1) - u, np.clip(v, y0, y1) - v).min())
        if gap <= near_px and (closest is None or gap < closest[0]):
            closest = (gap, (x0, y0, x1, y1), entry["label"], entry["score"])
    if best is not None:
        return best[1], best[2], best[3]
    if closest is not None:
        return closest[1], closest[2], closest[3]
    return None


def _same_object(frame_a: np.ndarray, box_a: tuple[int, int, int, int],
                 comp_a: np.ndarray, frame_b: np.ndarray,
                 box_b: tuple[int, int, int, int], comp_b: np.ndarray,
                 margin: float) -> bool:
    """Whether two components are plausibly the same object, by median colour.

    Compared on the median rather than the mean so a few pixels of hand or
    background inside the component cannot drag the answer.
    """
    a = frame_a[box_a[1]:box_a[3] + 1, box_a[0]:box_a[2] + 1][comp_a]
    b = frame_b[box_b[1]:box_b[3] + 1, box_b[0]:box_b[2] + 1][comp_b]
    if not len(a) or not len(b):
        return False
    return bool(np.linalg.norm(np.median(a.astype(np.float32), axis=0)
                               - np.median(b.astype(np.float32), axis=0)) <= margin)


def clean_seed(frames: list, human, box: tuple[int, int, int, int],
               first: int, last: int, detections: dict | None, label: str | None,
               min_score: float) -> tuple[int, tuple[int, int, int, int]] | None:
    """The frame before the grasp where the object is least covered by the hand.

    Every frame inside a grasp shows the object with a hand on it, so whatever
    the seed picks there is a partial object and the prompt has to thread
    between fingers. Before the hand arrives, the object sits on the table
    whole. It has not moved yet either -- it is about to be picked up, not
    already moving -- so the box found at contact still bounds it, and a
    detection of the same label is used in preference when one is there.

    Scored by how much of the box the human mask claims, lowest first, so the
    frame chosen is the one where the arm has not yet reached across the
    object. SAM2 propagates in both directions from the seed, so seeding here
    costs nothing and tracks the grasp from a clean start.
    """
    best = None
    for t in range(last, first - 1, -1):
        if t < 0 or t >= len(frames):
            continue
        candidate = box
        if detections is not None and label is not None:
            stride = max(1, int(detections.get("stride", 1)))
            key = str(int(round(t / stride)) * stride)
            same = [e for e in detections["frames"].get(key, [])
                    if e["label"] == label and e["score"] >= min_score]
            overlapping = []
            for entry in same:
                a0, b0, a1, b1 = entry["box"]
                ix = max(0, min(a1, box[2]) - max(a0, box[0]))
                iy = max(0, min(b1, box[3]) - max(b0, box[1]))
                if ix * iy > 0:
                    overlapping.append((ix * iy, entry["box"]))
            if overlapping:
                candidate = tuple(max(overlapping)[1])
        x0, y0, x1, y1 = candidate
        area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
        covered = float(np.asarray(human[t], dtype=bool)
                        [y0:y1 + 1, x0:x1 + 1].sum()) / area
        if best is None or covered < best[0]:
            best = (covered, t, candidate)
        if covered == 0.0:
            break
    if best is None:
        return None
    return best[1], best[2]


def anchor_chain(detections: dict, label: str, seed: int,
                 seed_box: tuple[int, int, int, int], first: int, last: int,
                 min_score: float) -> list[list]:
    """Same-label detection boxes across the interval, chained for continuity.

    SAM2 gets one prompt per segment and propagates from it, so a frame where
    the hand buries the object ends the track for every frame after it. These
    boxes are the evidence needed to restart it: the detector still sees the
    object on the frames either side of the burial.

    Chained from the seed outward, each step taking the same-label box nearest
    the previous one, because a scene holds two mugs and the nearest is the one
    this track is on.
    """
    if not detections or label is None:
        return []
    stride = max(1, int(detections.get("stride", 1)))
    frames = detections["frames"]
    anchors = {}
    for direction in (1, -1):
        previous = seed_box
        step = seed
        while True:
            step += direction * stride
            if step < first or step > last:
                break
            key = str(int(round(step / stride)) * stride)
            same = [e["box"] for e in frames.get(key, [])
                    if e["label"] == label and e["score"] >= min_score]
            if not same:
                continue
            centre = ((previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2)
            nearest = min(same, key=lambda b: (
                ((b[0] + b[2]) / 2 - centre[0]) ** 2
                + ((b[1] + b[3]) / 2 - centre[1]) ** 2))
            anchors[step] = [int(c) for c in nearest]
            previous = nearest
    return [[t, anchors[t]] for t in sorted(anchors)]


def object_points(component: np.ndarray, box: tuple[int, int, int, int],
                  count: int,
                  contact_uv: tuple[np.ndarray, np.ndarray] | None = None,
                  reach_px: float = 70.0) -> list[list[int]]:
    """Prompt points well inside the object, near where the fingers touch it.

    A point on the boundary is ambiguous between the object and the hand
    holding it, and SAM2 answers accordingly -- hence the distance transform.
    But depth alone is not enough: the colour test only says "not the support
    surface", so a component can run from the object straight into a wall or a
    stretch of table the ring failed to characterise, and the deepest point then
    sits in that background. Contact points that landed *on* the component are
    known object pixels, so points are taken from within ``reach_px`` of those.
    Contact points that missed are no help here -- they sit on the hand, which
    is as close to the background as it is to the object -- and using them all
    puts the reach back over the background. Falls back to the whole component
    when contact gives nothing.
    """
    x0, y0 = box[0], box[1]
    depth = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    if contact_uv is not None and len(contact_uv[0]):
        u, v = contact_uv
        inside = (u >= x0) & (u <= box[2]) & (v >= y0) & (v <= box[3])
        if inside.any():
            ys = np.clip(v[inside] - y0, 0, component.shape[0] - 1)
            xs = np.clip(u[inside] - x0, 0, component.shape[1] - 1)
            on_object = component[ys, xs]
            if on_object.any():
                seeds = np.zeros(component.shape, np.uint8)
                seeds[ys[on_object], xs[on_object]] = 1
                reach = cv2.distanceTransform((seeds == 0).astype(np.uint8),
                                              cv2.DIST_L2, 5) <= reach_px
                depth = np.where(reach, depth, 0.0)
    picks = []
    for _ in range(count):
        y, x = np.unravel_index(int(np.argmax(depth)), depth.shape)
        if depth[y, x] <= 2.0:
            break
        picks.append([int(x + x0), int(y + y0)])
        cv2.circle(depth, (int(x), int(y)), int(max(12, depth[y, x])), 0, -1)
    return picks


def hand_points(human_crop: np.ndarray, component: np.ndarray,
                box: tuple[int, int, int, int], count: int) -> list[list[int]]:
    """Negative points on the hand, taken as far from the object as possible.

    Sampling the hand mask evenly puts points right at the grip, where the mask
    routinely spills a few pixels onto the object it is holding -- a negative
    point on the object is exactly the prompt that makes SAM2 drop it. Points
    deep in the forearm say "not this" just as well and cannot land on the
    object by accident.
    """
    x0, y0 = box[0], box[1]
    if not human_crop.any():
        return []
    away = cv2.distanceTransform((~component).astype(np.uint8), cv2.DIST_L2, 5)
    away[~human_crop] = -1.0
    picks = []
    for _ in range(count):
        y, x = np.unravel_index(int(np.argmax(away)), away.shape)
        if away[y, x] < 0:
            break
        picks.append([int(x + x0), int(y + y0)])
        cv2.circle(away, (int(x), int(y)), 40, -1.0, -1)
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
    parser.add_argument("--detections", type=Path, default=None,
                        help="JSON from detect_objects_grounding_dino.py. When "
                             "given, the grasp box comes from the detection "
                             "holding the most contact points instead of from "
                             "the colour blob, which is what lets a segment "
                             "survive the arm mask swallowing the object.")
    parser.add_argument("--detection_near_px", type=float, default=60.0,
                        help="When no detection contains the contact points, "
                             "accept the nearest one within this many pixels. "
                             "A hand pressing a sponge flat sits just outside "
                             "its box; a different object is much further.")
    parser.add_argument("--detection_score", type=float, default=0.30,
                        help="Ignore detections weaker than this.")
    parser.add_argument("--seed_colour_margin", type=float, default=40.0,
                        help="How far the pre-grasp object's median colour may "
                             "sit from the held object's before the clean seed "
                             "is rejected as a different object.")
    parser.add_argument("--seed_in_grasp", action="store_true",
                        help="Seed inside the grasp instead of on the clean "
                             "frame before it. The clean frame is better when "
                             "the object is on the table beforehand; this is "
                             "the escape hatch for a clip where it is not, "
                             "e.g. an object carried in from off-screen.")
    parser.add_argument("--seed_tries", type=int, default=25,
                        help="Frames to consider as the seed, in order of grip "
                             "strength, before giving up and taking the "
                             "largest blob at the firmest one.")
    parser.add_argument("--seed_min_px", type=int, default=600,
                        help="A seed blob smaller than this is a sliver of the "
                             "object showing between fingers; prompting SAM2 "
                             "there tracks the sliver, not the object.")
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

    detections = (json.loads(args.detections.read_text(encoding="utf-8"))
                  if args.detections is not None else None)
    if detections is not None:
        print(f"[gdino] {sum(len(v) for v in detections['frames'].values())} "
              f"boxes over {len(detections['frames'])} frames, "
              f"labels={detections['labels']}")

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
        # Widen the tracked interval around the contact. SAM2 propagates both
        # ways from the seed, so this costs only tracking time.
        tracked_start = max(0, start - args.lead_frames)
        tracked_end = min(frame_count - 1, end + args.trail_frames)
        if index and segments:
            tracked_start = max(tracked_start, segments[-1]["end_frame"] + 1)

        # Seed where the grip is strongest AND the object is still visibly
        # touched. The strongest grip alone is the frame where the hand covers
        # the object most, which is also where the arm segmentation is likeliest
        # to have swallowed it; the object is then absent from the candidate
        # blobs and the seed silently lands on whatever else is largest nearby
        # -- a sponge, a tray -- which SAM2 then tracks for the whole interval.
        # Walking down the grip ranking and taking the first frame whose contact
        # points fall on visible object pixels costs a few JPEG reads and keeps
        # the seed inside the grasp.
        order = start + np.argsort(-score[start:end + 1, 1:].sum(axis=1))
        seed = box = positive = negative = None
        hit_label = hit_score = None
        fallback = None
        for candidate in order[:args.seed_tries]:
            candidate = int(candidate)
            u, v = project(verts[candidate][non_thumb], focal, width, height)
            if not len(u):
                continue
            candidate_box = (max(0, int(u.min()) - args.margin_px),
                             max(0, int(v.min()) - args.margin_px),
                             min(width - 1, int(u.max()) + args.margin_px),
                             min(height - 1, int(v.max()) + args.margin_px))
            # A detected box is the object's own outline; the projected-contact
            # box is only the span the fingers touch, grown by a guess.
            hit = detected_box(detections, candidate, u, v, args.detection_score,
                               args.detection_near_px)
            candidate_label = candidate_score = None
            if hit is not None:
                candidate_box, candidate_label, candidate_score = hit
            frame = cv2.imread(str(frames[candidate]))
            if frame is None:
                continue
            human_seed = np.asarray(human[candidate], dtype=bool)
            found = object_component(frame, candidate_box, human_seed,
                                     args.colour_margin, contact_uv=(u, v))
            if found is None:
                continue
            component, hits = found
            if fallback is None:
                fallback = (candidate, candidate_box, component, human_seed,
                            (u, v))
            if hits > 0 and int(component.sum()) >= args.seed_min_px:
                seed, box = candidate, candidate_box
                hit_label, hit_score = candidate_label, candidate_score
                contact_seed = (u, v)

                # The grasp told us which object this is; now seed it where it
                # is whole rather than where it is gripped.
                if not args.seed_in_grasp:
                    clean = clean_seed(frames, human, candidate_box,
                                       tracked_start, start - 1, detections,
                                       candidate_label, args.detection_score)
                    if clean is not None:
                        clean_frame = cv2.imread(str(frames[clean[0]]))
                        clean_human = np.asarray(human[clean[0]], dtype=bool)
                        found_clean = object_component(
                            clean_frame, clean[1], clean_human,
                            args.colour_margin, contact_uv=None)
                        # Only move the seed if the earlier frame shows the SAME
                        # object. "Before contact" is not "before the pick": in
                        # a place-and-release clip the object is already in the
                        # hand when HaCo first calls it contact, and the box then
                        # frames wherever it is headed -- an empty rack, or the
                        # other mug of the pair, which carries the right label
                        # and the wrong identity. Colour is enough to tell those
                        # apart and cannot be fooled by the label.
                        if (found_clean is not None
                                and int(found_clean[0].sum()) >= args.seed_min_px
                                and _same_object(frame, candidate_box, component,
                                                 clean_frame, clean[1],
                                                 found_clean[0],
                                                 args.seed_colour_margin)):
                            seed, box = clean[0], clean[1]
                            component, _ = found_clean
                            human_seed = clean_human
                            contact_seed = None

                x0, y0, x1, y1 = box
                positive = object_points(component, box,
                                         args.positive_points,
                                         contact_uv=contact_seed)
                negative = hand_points(human_seed[y0:y1 + 1, x0:x1 + 1],
                                       component, box,
                                       args.negative_points)
                break
        if seed is None:
            if fallback is None:
                print(f"[warn] f{start}-{end}: no contact vertex projects into "
                      f"frame, skipped")
                continue
            seed, box, component, human_seed, contact = fallback
            x0, y0, x1, y1 = box
            positive = object_points(component, box, args.positive_points,
                                     contact_uv=contact)
            negative = hand_points(human_seed[y0:y1 + 1, x0:x1 + 1],
                                   component, box, args.negative_points)
            print(f"[warn] f{start}-{end}: the fingers never touch visible "
                  f"object pixels in the {args.seed_tries} firmest frames; "
                  f"seeding f{seed} on the largest blob instead. Check this "
                  f"segment, and override it if SAM2 tracks the wrong thing.")

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
        if hit_label is not None:
            segments[-1]["detected_label"] = hit_label
            segments[-1]["detected_score"] = hit_score
            segments[-1]["anchor_boxes"] = anchor_chain(
                detections, hit_label, seed, box, tracked_start, tracked_end,
                args.detection_score)
        print(f"[seg] {segments[-1]['name']}: track f{tracked_start}-{tracked_end} "
              f"(contact f{start}-{end}) seed={seed} box={box} "
              f"+{len(positive)} -{len(negative)}"
              + (f" [{hit_label} {hit_score:.2f}]" if hit_label else " [colour]"))

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
