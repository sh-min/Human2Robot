"""HaCo per-vertex contact -> per-finger contact score with temporal hysteresis.

``extract_hand_contact.py`` writes a 778-vertex contact probability per frame.
A bare ``contact_mask.any()`` over-detects, so the compositor needs a compact,
temporally stable signal instead: one score per finger per hand per frame.

Vertices are assigned to a finger once, from the mean MANO hand over all valid
frames -- MANO topology is fixed, so a per-frame assignment only adds noise.
The score is the mean of the top fraction of that finger's probabilities, which
matches the aggregation ``composite_rb5_contact_occlusion.py`` already uses.

Output: ``<contact_dir>/finger_contact.npz``

    score        (T, 2, 5) float32   sides (left, right), fingers thumb..pinky
    state        (T, 2, 5) bool      after on/off hysteresis + min-duration
    valid        (T, 2)    bool      HaWoR hand validity
    finger_label (778,)    int8      0=thumb .. 4=pinky
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SIDES = ("left", "right")

# MANO 21-keypoint layout -> finger index (0=thumb .. 4=pinky). Keypoint 0 is
# the wrist and belongs to no finger; palm vertices nearest to it are dropped
# from every finger set rather than being forced into one.
_KPT_TO_FINGER = {0: -1}
for _f, _start in enumerate((1, 5, 9, 13, 17)):
    for _k in range(_start, _start + 4):
        _KPT_TO_FINGER[_k] = _f


def finger_labels(joints: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Assign each of the 778 MANO vertices to a finger by nearest keypoint."""
    dists = np.linalg.norm(verts[:, None, :] - joints[None, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    return np.array([_KPT_TO_FINGER[int(k)] for k in nearest], dtype=np.int8)


def hysteresis(score: np.ndarray, on: float, off: float,
               min_on: int, min_off: int) -> np.ndarray:
    """Schmitt trigger over time, then drop runs shorter than the minimum.

    A single-frame contact flip makes the compositor swap a finger's layer for
    one frame, which reads as a flicker. Rising above *on* starts contact and
    only falling below *off* ends it; short runs of either state are absorbed
    into their neighbour.
    """
    state = np.zeros(score.shape[0], dtype=bool)
    active = False
    for t, value in enumerate(score):
        if active:
            active = value >= off
        else:
            active = value >= on
        state[t] = active

    for target, min_len in ((True, min_on), (False, min_off)):
        if min_len <= 1:
            continue
        start = 0
        while start < len(state):
            end = start
            while end < len(state) and state[end] == state[start]:
                end += 1
            if state[start] == target and (end - start) < min_len:
                # Absorb the short run into whichever neighbour exists.
                state[start:end] = not target
            start = end
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact_dir", type=Path, required=True)
    parser.add_argument("--hawor_npz", type=Path, required=True)
    parser.add_argument("--top_fraction", type=float, default=0.25,
                        help="Fraction of a finger's vertices averaged, highest "
                             "probability first.")
    parser.add_argument("--on_threshold", type=float, default=0.72)
    parser.add_argument("--off_threshold", type=float, default=0.55)
    parser.add_argument("--min_on_frames", type=int, default=3)
    parser.add_argument("--min_off_frames", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    hawor = np.load(args.hawor_npz)
    valid = hawor["valid"]                       # (2, T)
    frames = sorted(args.contact_dir.glob("*.npz"))
    frames = [p for p in frames if p.name != "finger_contact.npz"]
    if not frames:
        raise FileNotFoundError(f"no per-frame contact npz under {args.contact_dir}")
    num_frames = len(frames)
    if num_frames != valid.shape[1]:
        raise ValueError(
            f"contact frames ({num_frames}) != HaWoR frames ({valid.shape[1]})"
        )

    labels = {}
    for side_idx, side in enumerate(SIDES):
        side_valid = valid[side_idx]
        if not side_valid.any():
            labels[side] = None
            continue
        joints = hawor[f"joints_{side}"][side_valid].mean(axis=0)
        verts = hawor[f"verts_{side}"][side_valid].mean(axis=0)
        labels[side] = finger_labels(joints, verts)

    score = np.zeros((num_frames, 2, 5), dtype=np.float32)
    for t, path in enumerate(frames):
        data = np.load(path)
        for side_idx, side in enumerate(SIDES):
            if labels[side] is None or not bool(data[f"{side}_valid"]):
                continue
            prob = data[f"{side}_contact_probability"].astype(np.float32)
            for f_idx in range(5):
                values = prob[labels[side] == f_idx]
                if values.size == 0:
                    continue
                keep = max(1, int(round(values.size * args.top_fraction)))
                score[t, side_idx, f_idx] = np.sort(values)[-keep:].mean()

    state = np.zeros_like(score, dtype=bool)
    for side_idx in range(2):
        for f_idx in range(5):
            state[:, side_idx, f_idx] = hysteresis(
                score[:, side_idx, f_idx],
                args.on_threshold, args.off_threshold,
                args.min_on_frames, args.min_off_frames,
            )

    out = args.output or (args.contact_dir / "finger_contact.npz")
    np.savez(
        out,
        score=score,
        state=state,
        valid=valid.T.copy(),
        finger_label=np.stack([
            labels[s] if labels[s] is not None else np.full(778, -1, np.int8)
            for s in SIDES
        ]),
        fingers=np.array(FINGERS),
        sides=np.array(SIDES),
        on_threshold=np.float32(args.on_threshold),
        off_threshold=np.float32(args.off_threshold),
        top_fraction=np.float32(args.top_fraction),
    )

    print(f"[ok] {out}  frames={num_frames}")
    for side_idx, side in enumerate(SIDES):
        if not valid[side_idx].any():
            print(f"  {side:5s}: no valid HaWoR frames")
            continue
        print(f"  {side:5s}:")
        for f_idx, name in enumerate(FINGERS):
            s = score[:, side_idx, f_idx]
            on = state[:, side_idx, f_idx]
            print(f"    {name:7s} mean={s.mean():.3f} max={s.max():.3f} "
                  f"contact_frames={int(on.sum()):4d}/{num_frames}")


if __name__ == "__main__":
    main()
