"""Project per-frame hand-contact MANO vertices onto RGB and save mp4.

Each contact npz has:
    {right,left}_contact_mask    (778,) bool       full MANO vertex mask
    {right,left}_contact_indices (N,) int64        indices of contacting vertices
    {right,left}_contact_verts_3d (N, 3) float32   3D positions in cam frame

Usage:
    conda activate RFM_retarget
    python project_contact.py \
        --rgb_dir /media/.../rgb \
        --contact_dir /media/.../contact \
        --out /media/.../contact_overlay.mp4
"""
import argparse
import os
from glob import glob

import cv2
import numpy as np
import tqdm
from natsort import natsorted


def project_points(pts, fx, fy, cx, cy):
    """pts: (N, 3) cam-frame 3D. Returns (N, 2) image pixel coords + depth."""
    pts = np.asarray(pts, dtype=np.float64)
    z = pts[:, 2]
    valid = z > 1e-3
    u = fx * pts[:, 0] / np.clip(z, 1e-3, None) + cx
    v = fy * pts[:, 1] / np.clip(z, 1e-3, None) + cy
    return np.stack([u, v], axis=1), z, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_dir", required=True)
    ap.add_argument("--contact_dir", required=True)
    ap.add_argument("--rgb_glob", default="rgb_frame*.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fx", type=float, default=497.77)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dot_radius", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.85)
    args = ap.parse_args()

    rgb_files = natsorted(glob(os.path.join(args.rgb_dir, args.rgb_glob)))
    if not rgb_files:
        raise RuntimeError(f"No RGB matched in {args.rgb_dir}")
    sample = cv2.imread(rgb_files[0])
    H, W = sample.shape[:2]
    fy = args.fy if args.fy is not None else args.fx
    cx = args.cx if args.cx is not None else W / 2
    cy = args.cy if args.cy is not None else H / 2

    print(f"image {W}x{H}  fx={args.fx} fy={fy} cx={cx} cy={cy}")
    print(f"{len(rgb_files)} RGB frames")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H)
    )

    color_right = (0, 80, 255)    # BGR: red-ish (right)
    color_left = (255, 80, 0)     # BGR: blue-ish (left)

    for rgb_path in tqdm.tqdm(rgb_files):
        bgr = cv2.imread(rgb_path)
        if bgr.shape[:2] != (H, W):
            bgr = cv2.resize(bgr, (W, H))

        contact_path = os.path.join(args.contact_dir,
                                    os.path.basename(rgb_path).replace(".png", ".npz"))
        if not os.path.exists(contact_path):
            writer.write(bgr)
            continue

        d = np.load(contact_path)
        overlay = bgr.copy()
        for prefix, color in (("right", color_right), ("left", color_left)):
            verts = d[f"{prefix}_contact_verts_3d"]
            if verts.size == 0:
                continue
            uv, z, ok = project_points(verts, args.fx, fy, cx, cy)
            for (u, v), good in zip(uv, ok):
                if not good:
                    continue
                ui, vi = int(round(u)), int(round(v))
                if 0 <= ui < W and 0 <= vi < H:
                    cv2.circle(overlay, (ui, vi), args.dot_radius, color, -1, cv2.LINE_AA)

        if args.alpha < 1.0:
            bgr = cv2.addWeighted(overlay, args.alpha, bgr, 1 - args.alpha, 0)
        else:
            bgr = overlay
        writer.write(bgr)

    writer.release()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
