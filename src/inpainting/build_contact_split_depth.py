"""HaCo contact vertices -> per-pixel front/behind split depth for the compositor.

The layered compositor decides whether a robot pixel is drawn in front of or
behind the manipulated object by comparing the rendered robot depth against a
single scalar per frame: the depth of one MANO joint (index-MCP by default).
One plane for the whole hand cannot be right everywhere -- the thumb and the
curled fingers sit at genuinely different depths around a grasped cup -- so the
missing occlusion has been patched per video with hand-authored
``force_front_*.npy`` masks.

HaCo predicts which hand vertices touch the object. Those vertices lie *on* the
object surface, so projecting them gives a sparse, per-pixel measurement of the
contact surface depth. Densifying it with a distance-weighted fill produces a
split-depth *map* instead of a plane, and it reverts to the original scalar
plane wherever no contact was predicted, so behaviour away from the grasp is
unchanged.

Output: ``(T, H, W) float16`` metric depth, consumable by
``composite_interaction_objects.py --contact_split_depth``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d

SIDES = ("left", "right")


def scalar_split_depth(joints_left, joints_right, valid, frame_count,
                       joint, sigma):
    """Reproduce the compositor's scalar plane, used as the fallback value."""
    z = np.full(frame_count, np.nan, dtype=np.float32)
    for idx in range(frame_count):
        values = []
        if idx < joints_left.shape[0] and valid[0, idx]:
            values.append(float(joints_left[idx, joint, 2]))
        if idx < joints_right.shape[0] and valid[1, idx]:
            values.append(float(joints_right[idx, joint, 2]))
        if values:
            z[idx] = float(np.mean(values))
    good = np.flatnonzero(np.isfinite(z))
    if not len(good):
        return np.full(frame_count, np.inf, dtype=np.float32)
    z = np.interp(np.arange(frame_count), good, z[good]).astype(np.float32)
    if sigma > 0:
        radius = max(1, int(np.ceil(3 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        z = np.convolve(np.pad(z, (radius, radius), mode="edge"), kernel,
                        mode="valid").astype(np.float32)
    return z


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact_dir", type=Path, required=True)
    parser.add_argument("--hawor_npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--img_focal", type=float, default=None,
                        help="Default: img_focal stored in the HaWoR npz.")
    parser.add_argument("--contact_threshold", type=float, default=0.5,
                        help="Vertex contact probability accepted as a surface "
                             "sample.")
    parser.add_argument("--prob_sigma_t", type=float, default=2.0,
                        help="Temporal Gaussian on per-vertex probability, in "
                             "frames. Stops one-frame contact flips from "
                             "reshaping the split surface.")
    parser.add_argument("--influence_px", type=float, default=90.0,
                        help="Distance at which the contact surface fades back "
                             "to the scalar plane.")
    parser.add_argument("--surface_bias", type=float, default=0.0,
                        help="Metres added to the contact surface. Positive "
                             "pushes the split plane away from the camera, "
                             "leaving more robot in front.")
    parser.add_argument("--spatial_sigma", type=float, default=9.0,
                        help="Gaussian blur (px) on the densified surface.")
    parser.add_argument("--threshold_joint", type=int, default=5)
    parser.add_argument("--depth_sigma", type=float, default=8.0)
    args = parser.parse_args()

    hawor = np.load(args.hawor_npz)
    valid = hawor["valid"]
    frame_count = valid.shape[1]
    focal = float(args.img_focal if args.img_focal is not None
                  else hawor["img_focal"])
    cx, cy = args.width / 2.0, args.height / 2.0

    frames = sorted(p for p in args.contact_dir.glob("*.npz")
                    if p.name != "finger_contact.npz")
    if len(frames) != frame_count:
        raise ValueError(
            f"contact frames ({len(frames)}) != HaWoR frames ({frame_count})"
        )

    # Load every per-vertex probability first so the temporal filter can run
    # before anything is projected; 553 x 778 x 2 floats is small.
    prob = np.zeros((frame_count, 2, 778), dtype=np.float32)
    for t, path in enumerate(frames):
        data = np.load(path)
        for side_idx, side in enumerate(SIDES):
            if bool(data[f"{side}_valid"]):
                prob[t, side_idx] = data[f"{side}_contact_probability"]
    if args.prob_sigma_t > 0:
        prob = gaussian_filter1d(prob, args.prob_sigma_t, axis=0, mode="nearest")

    fallback = scalar_split_depth(
        hawor["joints_left"], hawor["joints_right"], valid, frame_count,
        args.threshold_joint, args.depth_sigma,
    )
    verts = {side: hawor[f"verts_{side}"] for side in SIDES}

    out = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=np.float16,
        shape=(frame_count, args.height, args.width),
    )

    covered_px = np.zeros(frame_count, dtype=np.int64)
    for t in range(frame_count):
        plane = float(fallback[t])
        sparse = np.full((args.height, args.width), np.inf, dtype=np.float32)

        for side_idx, side in enumerate(SIDES):
            if not valid[side_idx, t]:
                continue
            keep = prob[t, side_idx] >= args.contact_threshold
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
            inside = ((u >= 0) & (u < args.width)
                      & (v >= 0) & (v < args.height))
            if not inside.any():
                continue
            u, v, z = u[inside], v[inside], z[inside]
            # Several vertices land on one pixel; the nearest one is the
            # visible surface.
            order = np.argsort(-z)
            sparse[v[order], u[order]] = z[order]

        seeds = np.isfinite(sparse)
        if not seeds.any():
            out[t] = np.float16(plane)
            continue
        covered_px[t] = int(seeds.sum())

        dist, (iy, ix) = distance_transform_edt(
            ~seeds, return_distances=True, return_indices=True
        )
        surface = sparse[iy, ix] + args.surface_bias
        if args.spatial_sigma > 0:
            surface = cv2.GaussianBlur(surface, (0, 0), args.spatial_sigma)
        weight = np.clip(1.0 - dist / args.influence_px, 0.0, 1.0).astype(np.float32)
        # Smoothstep keeps the seam between measured surface and scalar plane
        # from showing up as a visible depth edge in the composite.
        weight = weight * weight * (3.0 - 2.0 * weight)
        out[t] = (weight * surface + (1.0 - weight) * plane).astype(np.float16)

    out.flush()
    used = covered_px > 0
    print(f"[ok] {args.output}  frames={frame_count}  "
          f"contact frames={int(used.sum())}/{frame_count}")
    if used.any():
        print(f"     seed px/frame: mean={covered_px[used].mean():.0f} "
              f"max={covered_px.max()}")
        print(f"     scalar plane z: mean={fallback.mean():.3f} m "
              f"[{fallback.min():.3f}, {fallback.max():.3f}]")


if __name__ == "__main__":
    main()
