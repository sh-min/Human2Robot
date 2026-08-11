"""Build per-vertex finger-part labels for MANO right/left.

Uses the MANO skinning weights matrix (778, 16) — for each vertex, the joint
that drives it the most is its "part". MANO joint indices:
    0 = wrist
    1,2,3   = index1 / index2 / index3 (tip-side)
    4,5,6   = middle
    7,8,9   = pinky  ('little' in MANO)
    10,11,12 = ring
    13,14,15 = thumb
Joints {3,6,9,12,15} are the distal (tip-side) ones — use them for "fingertip
only" masks.

Outputs (saved to src/retargeting/assets/):
    finger_part_{right,left}.npy    (778,) int   joint-index per vertex (0..15)
    fingertip_mask_{right,left}.npy (778,) bool  True if joint in {3,6,9,12,15}

Run once (in `hawor` env because MANO .pkl needs chumpy):
    conda activate hawor
    cd <repo_root>/src/retargeting
    python build_finger_parts.py
"""
import os
import pickle

import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MANO_PKL = {
    "right": os.path.join(REPO_ROOT, "third_party", "HaWoR",
                          "_DATA", "data", "mano", "MANO_RIGHT.pkl"),
    "left":  os.path.join(REPO_ROOT, "third_party", "HaWoR",
                          "_DATA", "data_left", "mano_left", "MANO_LEFT.pkl"),
}

TIP_JOINTS = (3, 6, 9, 12, 15)   # index, middle, pinky, ring, thumb (distal joints)
FINGER_NAME = {1: "index", 2: "index", 3: "index",
               4: "middle", 5: "middle", 6: "middle",
               7: "pinky", 8: "pinky", 9: "pinky",
               10: "ring", 11: "ring", 12: "ring",
               13: "thumb", 14: "thumb", 15: "thumb",
               0: "wrist"}


def main():
    os.makedirs(ASSETS, exist_ok=True)
    for hand, pkl in MANO_PKL.items():
        with open(pkl, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        W = np.asarray(d["weights"], dtype=np.float64)   # (778, 16)
        if W.shape != (778, 16):
            raise RuntimeError(f"unexpected weights shape {W.shape} for {hand}")
        parts = W.argmax(axis=1).astype(np.int32)        # (778,)
        tip_mask = np.isin(parts, TIP_JOINTS)

        np.save(os.path.join(ASSETS, f"finger_part_{hand}.npy"), parts)
        np.save(os.path.join(ASSETS, f"fingertip_mask_{hand}.npy"), tip_mask)

        counts = {FINGER_NAME[j]: 0 for j in range(16)}
        for j in parts:
            counts[FINGER_NAME[int(j)]] += 1
        print(f"[{hand}] tip_verts={int(tip_mask.sum())}/778")
        print(f"         per-finger counts: {counts}")


if __name__ == "__main__":
    main()
