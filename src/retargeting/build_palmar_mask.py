"""Build a palmar (manipulation-relevant) vertex mask for MANO right/left.

Loads each MANO .pkl, takes its mean-shape vertices (v_template, canonical
T-pose), computes per-vertex normals, and flags vertices whose normal points
toward the palm side. The result is saved as a (778,) bool array per hand,
plus a 4-view PNG for visual verification.

Run once:
    conda activate RFM_retarget
    cd <repo_root>/src/retargeting
    python build_palmar_mask.py
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from _paths import REPO_ROOT

# MANO models live inside the HaWoR submodule.
MANO_RIGHT_PKL = os.path.join(REPO_ROOT, "third_party", "HaWoR",
                              "_DATA", "data", "mano", "MANO_RIGHT.pkl")
MANO_LEFT_PKL = os.path.join(REPO_ROOT, "third_party", "HaWoR",
                             "_DATA", "data_left", "mano_left", "MANO_LEFT.pkl")

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# Palm-facing direction in MANO canonical frame.
# Empirically: right hand palm faces -y; left is mirrored along x so its palm
# also faces -y in its own canonical (left MANO models are mirrors).
# If the rendered PNG shows the wrong half lit up, flip the sign here.
PALM_DIR = {
    "right": np.array([0.0, -1.0, 0.0]),
    "left":  np.array([0.0, -1.0, 0.0]),
}
# vertex.normal · palm_dir > THRESHOLD  → palmar.
# 0.0 = exactly half-space; raise to be stricter, lower to include side flanks.
THRESHOLD = 0.0


def _to_np(x):
    """MANO pkl stores chumpy arrays; cast to numpy."""
    try:
        return np.asarray(x, dtype=np.float64)
    except Exception:
        return np.array(x.r, dtype=np.float64)


def build_mask(pkl_path, palm_dir, threshold=THRESHOLD):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    v = _to_np(d["v_template"])           # (778, 3)
    faces = np.asarray(d["f"], dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)
    normals = np.asarray(mesh.vertex_normals)
    mask = (normals @ palm_dir) > threshold
    return v, faces, normals, mask


def visualize(v, mask, out_png, title):
    palmar = v[mask]
    dorsal = v[~mask]
    fig = plt.figure(figsize=(14, 10))
    views = [("palm view",     0, -90),
             ("dorsal view",   0,  90),
             ("side view",     0,   0),
             ("isometric",    30,  45)]
    for k, (name, el, az) in enumerate(views):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        ax.scatter(*dorsal.T, s=2, c="lightgray", alpha=0.4,
                   label=f"dorsal ({len(dorsal)})")
        ax.scatter(*palmar.T, s=4, c="crimson", alpha=0.9,
                   label=f"palmar ({len(palmar)})")
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=el, azim=az)
        ax.set_title(name)
        if k == 0:
            ax.legend(loc="upper right")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(ASSETS, exist_ok=True)
    for hand, pkl in [("right", MANO_RIGHT_PKL), ("left", MANO_LEFT_PKL)]:
        if not os.path.exists(pkl):
            print(f"[{hand}] MANO .pkl not found at {pkl}, skip")
            continue
        v, faces, normals, mask = build_mask(pkl, PALM_DIR[hand])
        out_npy = os.path.join(ASSETS, f"palmar_mask_{hand}.npy")
        out_png = os.path.join(ASSETS, f"palmar_mask_{hand}.png")
        np.save(out_npy, mask)
        visualize(v, mask, out_png,
                  f"MANO {hand}  palm_dir={PALM_DIR[hand].tolist()}  "
                  f"thr={THRESHOLD}  palmar={int(mask.sum())}/{len(mask)}")
        print(f"[{hand}] palmar={int(mask.sum())}/{len(mask)}")
        print(f"         saved {out_npy}")
        print(f"         saved {out_png}")


if __name__ == "__main__":
    main()
