"""HaCo contact vertices -> forced-object mask that hides the grasping fingers.

When a hand wraps a cup, the four curled fingers pass *behind* it and only the
thumb stays in front. The compositor expresses that with ``--force_front_mask``:
object interior that must be drawn over the robot in stage 5. Until now that
mask was painted by hand, once per video (``force_front_v31 … v40.npy``), which
is the single biggest reason the pipeline does not transfer to a new clip.

HaCo already predicts which MANO vertices touch the object. Those vertices are
exactly the ones that should disappear behind it. Projecting the non-thumb
contact vertices, growing them into a patch and clipping to the object interior
reproduces the hand-authored mask from a measurement instead of a brush.

The thumb is excluded twice over: its vertices are dropped here, and the
compositor carves ``--force_robot_front_mask`` out of the forced-object layer
anyway, so stage 6 keeps final authority over thumb pixels.

Caveat: the contact vertices sit on the *human* MANO hand, while the pixels
being hidden belong to the *retargeted robot* hand. The two agree only as well
as the retargeting does, which is why the patch is grown by ``--patch_px``
rather than used at vertex resolution. Check the projection with
``src/contact_estimation/visualize_contact_overlay.py`` before trusting it.

Output: ``(T, H, W) bool``, consumable by
``composite_interaction_objects.py --force_front_mask``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "contact_estimation"))
from aggregate_finger_contact import FINGERS, SIDES, finger_labels  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact_dir", type=Path, required=True)
    parser.add_argument("--hawor_npz", type=Path, required=True)
    parser.add_argument("--object_mask", type=Path, required=True,
                        help="(T, H, W) bool. The forced layer draws object "
                             "RGB, so the mask is clipped to object interior.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot_mask", type=Path, default=None,
                        help="(T, H, W) bool. Reports how much robot the mask "
                             "hides, and drives the lead-in: during the approach "
                             "the trigger is the rendered robot overlapping the "
                             "object, not the human vertices, because those two "
                             "part company before the grasp closes.")
    parser.add_argument("--thumb_mask", type=Path, default=None,
                        help="(T, H, W) bool, subtracted from --robot_mask so "
                             "the thumb never triggers the lead-in.")
    parser.add_argument("--lead_overlap_px", type=int, default=120,
                        help="Robot-on-object pixels that count as occluded. "
                             "Once the fingers are this far behind the object "
                             "they are hidden on that frame, with no wait.")
    parser.add_argument("--fingers", nargs="+", default=list(FINGERS[1:]),
                        choices=list(FINGERS),
                        help="Fingers whose contact hides the robot. The thumb "
                             "is excluded by default and should stay that way.")
    parser.add_argument("--contact_threshold", type=float, default=0.5,
                        help="Vertex contact probability accepted as a touch.")
    parser.add_argument("--prob_sigma_t", type=float, default=2.0,
                        help="Temporal Gaussian on per-vertex probability, in "
                             "frames, so one-frame flips do not pop a finger "
                             "in and out of the object.")
    parser.add_argument("--mode", choices=("component", "patch"),
                        default="component",
                        help="component: a touched object is hidden over its "
                             "whole connected area, which is what a cup does to "
                             "the fingers wrapped around it. patch: only near "
                             "the contact points, for a partial occluder.")
    parser.add_argument("--patch_px", type=float, default=14.0,
                        help="Radius the projected contact points are grown by. "
                             "Covers the offset between the MANO hand HaCo saw "
                             "and the robot hand being composited. In component "
                             "mode this is the reach used to decide whether a "
                             "contact point belongs to an object component.")
    parser.add_argument("--close_px", type=float, default=9.0,
                        help="Morphological close, to fuse neighbouring "
                             "fingertips into one patch instead of blobs.")
    parser.add_argument("--lead_frames", type=int, default=30,
                        help="Start hiding this many frames before contact is "
                             "detected. Fingers pass behind an object while "
                             "they are still closing on it, so a mask that "
                             "waits for the contact probability leaves them "
                             "drawn over the object during the approach. Lead "
                             "frames only take effect where the projected "
                             "fingers already overlap the object, so this "
                             "cannot hide anything the hand has not reached.")
    parser.add_argument("--lead_min_fingers", type=int, default=2,
                        help="Non-thumb fingers whose contact state marks a "
                             "grasp, for locating where the lead-in starts. "
                             "Match build_object_segments_from_contact.py.")
    parser.add_argument("--min_px", type=int, default=150,
                        help="Frames whose mask is smaller than this are "
                             "cleared; a handful of pixels is noise, not a "
                             "grasp.")
    parser.add_argument("--img_focal", type=float, default=None,
                        help="Default: img_focal stored in the HaWoR npz.")
    args = parser.parse_args()

    hawor = np.load(args.hawor_npz)
    valid = hawor["valid"]                                  # (2, T)
    focal = float(args.img_focal if args.img_focal is not None
                  else hawor["img_focal"])

    objects = np.load(args.object_mask, mmap_mode="r")
    frame_count, height, width = objects.shape
    if valid.shape[1] < frame_count:
        frame_count = valid.shape[1]
    cx, cy = width / 2.0, height / 2.0

    frames = sorted(p for p in args.contact_dir.glob("*.npz")
                    if p.name != "finger_contact.npz")
    if len(frames) < frame_count:
        raise ValueError(
            f"contact frames ({len(frames)}) < composited frames ({frame_count})"
        )

    keep_fingers = {FINGERS.index(name) for name in args.fingers}
    if FINGERS.index("thumb") in keep_fingers:
        print("[warn] the thumb is in --fingers; stage 6 will draw it back on "
              "top anyway, so this only softens its edge.")

    # Vertex -> finger once, from the mean hand: MANO topology is fixed, so a
    # per-frame assignment would only add noise.
    labels = {}
    for side_idx, side in enumerate(SIDES):
        side_valid = valid[side_idx]
        if not side_valid.any():
            labels[side] = None
            continue
        labels[side] = finger_labels(
            hawor[f"joints_{side}"][side_valid].mean(axis=0),
            hawor[f"verts_{side}"][side_valid].mean(axis=0),
        )

    prob = np.zeros((frame_count, 2, 778), dtype=np.float32)
    for t in range(frame_count):
        data = np.load(frames[t])
        for side_idx, side in enumerate(SIDES):
            if bool(data[f"{side}_valid"]):
                prob[t, side_idx] = data[f"{side}_contact_probability"]
    if args.prob_sigma_t > 0:
        prob = gaussian_filter1d(prob, args.prob_sigma_t, axis=0, mode="nearest")

    verts = {side: hawor[f"verts_{side}"] for side in SIDES}
    robot = (np.load(args.robot_mask, mmap_mode="r")
             if args.robot_mask is not None else None)
    thumb = (np.load(args.thumb_mask, mmap_mode="r")
             if args.thumb_mask is not None else None)
    patch = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(2 * args.patch_px) + 1,) * 2)
    closer = (cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(2 * args.close_px) + 1,) * 2)
        if args.close_px > 0 else None)

    out = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=bool,
        shape=(frame_count, height, width))

    # Where a grasp begins, and the run of frames just before it: during the
    # approach the contact probability has not crossed yet, but the fingers are
    # closing and already pass behind the object. Grasp starts come from the
    # same per-finger state the object spec uses, so the two agree on when a
    # grasp is; a bare "any vertex over threshold" test is true almost always
    # and would find no starts at all.
    lead = np.zeros(frame_count, dtype=bool)
    held = np.zeros(frame_count, dtype=bool)
    state_path = args.contact_dir / "finger_contact.npz"
    if state_path.exists():
        state = np.load(state_path)["state"][:frame_count, :, 1:]
        held = (state.sum(axis=2) >= args.lead_min_fingers).any(axis=1)
        if args.lead_frames > 0:
            starts = np.flatnonzero(held & ~np.roll(held, 1))
            for start in starts:
                lead[max(0, start - args.lead_frames):start] = True
            lead &= ~held
    elif args.lead_frames > 0:
        raise FileNotFoundError(
            f"{state_path} is needed for --lead_frames; run "
            f"aggregate_finger_contact.py first, or pass --lead_frames 0")

    hidden = np.zeros(frame_count, dtype=np.int64)
    seeds_per_frame = np.zeros(frame_count, dtype=np.int64)
    lead_used = 0
    for t in range(frame_count):
        seeds = np.zeros((height, width), dtype=np.uint8)
        for side_idx, side in enumerate(SIDES):
            if labels[side] is None or not valid[side_idx, t]:
                continue
            wanted = np.isin(labels[side], list(keep_fingers))
            keep = wanted & (prob[t, side_idx] >= args.contact_threshold)
            if lead[t]:
                # No vertex clears the threshold yet, so take the whole finger
                # and let the object-overlap test below decide.
                keep = wanted
            if not keep.any():
                continue
            pts = verts[side][t][keep]
            z = pts[:, 2]
            forward = z > 1e-3
            if not forward.any():
                continue
            pts, z = pts[forward], z[forward]
            u = np.round(focal * pts[:, 0] / z + cx).astype(np.int64)
            v = np.round(focal * pts[:, 1] / z + cy).astype(np.int64)
            inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            if not inside.any():
                continue
            seeds[v[inside], u[inside]] = 1

        if (lead[t] or held[t]) and robot is not None:
            # Ask the robot itself, not just the human vertices. The vertices
            # decide which object is held, but they only land on the object mask
            # when the two agree closely; a mask that is slightly off, or a hand
            # that has drifted from the retargeted robot, leaves the projection
            # outside the object and nothing gets hidden -- while HaCo is
            # reporting a firm four-finger grasp. Wherever the robot's non-thumb
            # pixels sit on the object, those fingers are behind it, so the
            # overlap seeds the mask directly.
            arm = np.array(robot[t], dtype=bool)
            if thumb is not None:
                arm &= ~np.asarray(thumb[t], dtype=bool)
            behind = arm & np.asarray(objects[t], dtype=bool)
            if behind.sum() >= args.lead_overlap_px:
                seeds = np.maximum(seeds, behind.astype(np.uint8))

        seeds_per_frame[t] = int(seeds.sum())
        if not seeds.any():
            out[t] = False
            continue

        region = cv2.dilate(seeds, patch)
        if closer is not None:
            region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, closer)
        objects_t = np.asarray(objects[t], dtype=bool)

        if args.mode == "patch":
            mask = region.astype(bool) & objects_t
        else:
            # A grasped object hides the wrapped fingers over its whole body,
            # not only where a vertex happens to touch. So contact only elects
            # which object is being held; the object's own extent does the
            # occluding.
            count, comps = cv2.connectedComponents(
                objects_t.astype(np.uint8), connectivity=8)
            touched = np.unique(comps[region.astype(bool) & objects_t])
            touched = touched[touched != 0]
            mask = (np.isin(comps, touched) if touched.size
                    else np.zeros_like(objects_t))
        if mask.sum() < args.min_px:
            mask[:] = False
        out[t] = mask
        hidden[t] = int(mask.sum())
        if lead[t] and mask.any():
            lead_used += 1

    out.flush()

    active = hidden > 0
    print(f"[ok] {args.output}  frames={frame_count}  "
          f"grasp frames={int(active.sum())}/{frame_count}  "
          f"(of which {lead_used} are approach frames added by --lead_frames "
          f"{args.lead_frames})")
    if active.any():
        print(f"     contact seeds/frame: mean={seeds_per_frame[active].mean():.0f}"
              f"  mask px/frame: mean={hidden[active].mean():.0f} "
              f"max={hidden.max()}  total={hidden.sum()}")
    if args.robot_mask is not None:
        robot = np.load(args.robot_mask, mmap_mode="r")
        covered = sum(int((np.asarray(out[t]) & np.asarray(robot[t])).sum())
                      for t in range(frame_count))
        print(f"     robot px hidden behind the object: {covered}")


if __name__ == "__main__":
    main()
