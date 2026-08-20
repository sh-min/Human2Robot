"""Geometry-grounded contact shadow for the robot overlay (compositor stage).

The robot hand otherwise floats: nothing ties it to the surface it rests on.
Here the support surface is recovered as a plane from the scene depth, the
robot's visible surface is projected onto it along the shared scene light
direction, and the resulting footprint darkens the background before the robot
is composited on top.

Needs the metric scene depth (depth_processor/depth_aligned.npy, stage 7) and
the robot depth buffer (stage 5), both in the OpenCV camera frame, metres, at
the same intrinsics. Light direction comes from scene_lighting so the shadow
agrees with the render-stage shading.
"""
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from scene_lighting import light_dir_cam


def _backproject(depth, F, cx, cy):
    H, W = depth.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    # non-finite depth (e.g. +inf off the robot) -> 0; callers mask it out.
    z = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
    return np.stack([(uu - cx) * z / F, (vv - cy) * z / F, z], -1)


def fit_support_plane(scene_depth, robot_mask, F, cx, cy, *,
                      z_lo=0.31, z_hi=1.5, iters=140, thresh=0.012,
                      prev=None, ema=0.6, rng_seed=0):
    """RANSAC dominant plane on scene points, robot region and upper frame
    excluded. Returns (n, d) with n·X = d, n facing the camera, EMA-smoothed
    against *prev* for temporal stability under a moving egocentric camera.
    Returns *prev* unchanged if too few points this frame.
    """
    H, W = scene_depth.shape
    valid = np.isfinite(scene_depth) & (scene_depth > z_lo) & (scene_depth < z_hi)
    valid &= ~cv2.dilate(robot_mask.astype(np.uint8),
                         np.ones((15, 15), np.uint8)).astype(bool)
    valid[: int(0.25 * H)] = False          # wall / backrest
    P = _backproject(scene_depth, F, cx, cy)[valid]
    if len(P) < 500:
        return prev
    rng = np.random.RandomState(rng_seed)
    P = P[rng.choice(len(P), min(5000, len(P)), replace=False)]

    best_n, best_d, best_cnt = None, None, 0
    for _ in range(iters):
        tri = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n /= nn
        d = n @ tri[0]
        c = int((np.abs(P @ n - d) < thresh).sum())
        if c > best_cnt:
            best_n, best_d, best_cnt = n, d, c
    if best_n is None:
        return prev

    inl = np.abs(P @ best_n - best_d) < thresh
    Q = P[inl]
    c = Q.mean(0)
    _, _, Vt = np.linalg.svd(Q - c)
    n = Vt[-1]
    if n[2] > 0:                             # orient toward camera (−z)
        n = -n
    d = n @ c

    if prev is not None:
        pn, pd = prev
        if pn @ n < 0:
            n, d = -n, -d
        n = ema * pn + (1 - ema) * n
        n /= np.linalg.norm(n) + 1e-12
        d = ema * pd + (1 - ema) * d
    return n, d


def contact_shadow_alpha(scene_depth, robot_depth, robot_mask, plane,
                         F, cx, cy, *, light_dir=None, opacity=0.6,
                         blur=6.0, plane_tol=0.06, gain=3.0,
                         bands=0, band_max=0.9, penumbra=70.0, falloff=0.30):
    """Per-pixel darkening in [0, opacity]: the robot's footprint projected onto
    *plane* along *light_dir*, restricted to pixels where the visible surface
    really is that plane (not the wall, an object, or the robot itself).

    With ``bands`` > 1, robot points are split by height above the support
    plane. Higher slices receive a wider, weaker penumbra, reproducing the
    accepted soft full-arm shadow instead of one hard silhouette.
    """
    H, W = scene_depth.shape
    if plane is None:
        return np.zeros((H, W), np.float32)
    n, d = plane
    L = light_dir_cam(light_dir)
    nL = n @ L
    if abs(nL) < 1e-4:
        return np.zeros((H, W), np.float32)

    rmask = np.isfinite(robot_depth) & robot_mask
    P = _backproject(robot_depth, F, cx, cy)[rmask]
    t = (d - P @ n) / nL
    keep = t > 0
    P, t = P[keep], t[keep]
    Sp = P + t[:, None] * L                  # points on the plane
    z = Sp[:, 2]
    us = np.round(Sp[:, 0] * F / z + cx).astype(int)
    vs = np.round(Sp[:, 1] * F / z + cy).astype(int)
    ok = (us >= 0) & (us < W) & (vs >= 0) & (vs < H) & (z > 0)
    if bands > 1:
        edges = np.linspace(0.0, band_max, bands + 1)
        height = np.clip(t[ok], 0.0, band_max)
        transmit = np.ones((H, W), np.float32)
        for lo, hi in zip(edges[:-1], edges[1:]):
            selected = (height >= lo) & (height < hi)
            if selected.sum() < 50:
                continue
            acc = np.zeros((H, W), np.float32)
            np.add.at(acc, (vs[ok][selected], us[ok][selected]), 1.0)
            midpoint = 0.5 * (lo + hi)
            acc = gaussian_filter(acc, blur + penumbra * midpoint)
            acc /= acc.max() + 1e-6
            strength = np.exp(-midpoint / falloff)
            transmit *= 1.0 - np.clip(acc * gain, 0, 1.0) * strength
        alpha = (1.0 - transmit) * opacity
    else:
        acc = np.zeros((H, W), np.float32)
        np.add.at(acc, (vs[ok], us[ok]), 1.0)
        acc = gaussian_filter(acc, blur)
        acc /= acc.max() + 1e-6
        alpha = np.clip(acc * opacity * gain, 0, opacity)

    # z where each pixel's ray meets the plane; keep shadow only where the
    # visible surface sits there (so it lands on the table, not the far wall).
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    ray = np.stack([(uu - cx) / F, (vv - cy) / F, np.ones((H, W))], -1)
    plane_z = d / (ray @ n)
    on_plane = np.isfinite(scene_depth) & np.isfinite(plane_z) \
        & (np.abs(scene_depth - plane_z) < plane_tol)
    alpha *= on_plane
    alpha[cv2.dilate(robot_mask.astype(np.uint8),
                     np.ones((3, 3), np.uint8)).astype(bool)] = 0
    return alpha.astype(np.float32)
