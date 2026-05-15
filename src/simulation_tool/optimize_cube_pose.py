"""Estimate the 6-DOF pose of a fixed-size cube at the first frame, given:
  1. MANO hand verts (palmar+fingertip on chosen fingers) → cube surface should
     touch them (contact distance² minimized).
  2. A 2-D cube mask (binary image) → projected cube silhouette should match
     it (1 - IoU minimized).

Output: cube_pose.npz + visualization PNG.

Usage:
    conda activate RFM_retarget
    cd <repo_root>/src/simulation_tool
    python optimize_cube_pose.py \
        --npz   /path/to/<seq>_hawor/retarget_input.npz \
        --mask  /path/to/cube_mask_first_frame.png \
        --rgb   /path/to/rgb/rgb_frame00000.png \
        --hand  left
"""
import argparse
import os

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as Rscipy

# Re-use the masks / faces / R_MANO_XHAND that the retargeting module already
# built (palmar_mask, fingertip_mask, finger_part, mano_faces).
_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.abspath(os.path.join(_SIM_DIR, "..", "retargeting", "assets"))

# MANO joint indices that belong to each finger:
#   thumb=13-15, index=1-3, middle=4-6, ring=10-12, pinky=7-9
FINGER_JOINTS = {
    "thumb":  {13, 14, 15},
    "index":  {1, 2, 3},
    "middle": {4, 5, 6},
    "ring":   {10, 11, 12},
    "pinky":  {7, 8, 9},
}


def cube_corners(center, rotvec, size):
    h = size / 2.0
    local = np.array(
        [[i * h, j * h, k * h] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)],
        dtype=np.float64,
    )
    R = Rscipy.from_rotvec(rotvec).as_matrix()
    return local @ R.T + center


def contact_loss(p_world, center, rotvec, size, lambda_inside=5.0):
    """Σ (signed distance from p to cube surface)² with extra penalty inside.

    p_local = R.T @ (p - c)  -- in box frame.
    SDF = ||max(|p_local|-h, 0)||   if outside (≥0)
        + min(max(|p_local|-h), 0)  if inside  (≤0)
    """
    R = Rscipy.from_rotvec(rotvec).as_matrix()
    p_local = (p_world - center) @ R   # row convention == R.T @ (p-c)
    h = size / 2.0
    q = np.abs(p_local) - h
    out_d = np.linalg.norm(np.maximum(q, 0), axis=1)
    in_d = np.minimum(np.max(q, axis=1), 0)         # ≤ 0 inside
    return float(np.sum(out_d ** 2 + lambda_inside * in_d ** 2))


def project(p, fx, fy, cx, cy):
    z = np.clip(p[:, 2], 1e-6, None)
    u = fx * p[:, 0] / z + cx
    v = fy * p[:, 1] / z + cy
    return np.stack([u, v], axis=1)


def silhouette_iou(corners_3d, K, mask_gt):
    """Plain IoU — kept for reporting only."""
    if (corners_3d[:, 2] <= 0).any():
        return 0.0
    fx, fy, cx, cy = K
    uv = project(corners_3d, fx, fy, cx, cy)
    H, W = mask_gt.shape
    try:
        hull = ConvexHull(uv)
    except Exception:
        return 0.0
    poly = uv[hull.vertices].astype(np.int32)
    pred = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(pred, [poly], 1)
    pred = pred.astype(bool)
    inter = (pred & mask_gt).sum()
    union = (pred | mask_gt).sum()
    return float(inter) / max(union, 1)


def _rasterize_cube_silhouette(corners_3d, K, mask_shape):
    if (corners_3d[:, 2] <= 0).any():
        return None
    fx, fy, cx, cy = K
    uv = project(corners_3d, fx, fy, cx, cy)
    try:
        hull = ConvexHull(uv)
    except Exception:
        return None
    poly = uv[hull.vertices].astype(np.int32)
    pred = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(pred, [poly], 1)
    return pred.astype(bool)


def detect_mask_lines(mask_gt, min_length=20, max_gap=5, threshold=20):
    """Detect long straight line segments along the mask boundary via Hough.

    Returns array (N, 4): x1, y1, x2, y2 — pixel-space line segments. These
    are the parts of the mask outline that are TRUE cube silhouette
    boundaries (the curved / hand-occluded parts don't pass the Hough
    threshold and are dropped).
    """
    edge = cv2.Canny((mask_gt.astype(np.uint8) * 255), 50, 150)
    lines = cv2.HoughLinesP(
        edge, 1, np.pi / 180.0, threshold=threshold,
        minLineLength=min_length, maxLineGap=max_gap,
    )
    if lines is None:
        return np.zeros((0, 4), dtype=np.float64)
    return lines[:, 0].astype(np.float64)


def _point_to_segment_distance(p, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-9:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float((p - a) @ ab / L2)))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def line_alignment_loss(corners_3d, K, mask_lines, num_samples=10):
    """Avg distance (pixels) from sampled points along the detected mask
    straight lines to the nearest cube silhouette polygon edge.

    Hand-occluded curved parts of the mask boundary do not appear in
    mask_lines (Hough filters them out) so they never penalise the cube.
    """
    if len(mask_lines) == 0:
        return 0.0
    if (corners_3d[:, 2] <= 0).any():
        return 1e3
    fx, fy, cx, cy = K
    uv = project(corners_3d, fx, fy, cx, cy)
    try:
        hull = ConvexHull(uv)
    except Exception:
        return 1e3
    hi = hull.vertices
    n = len(hi)
    cube_segs = [(uv[hi[i]], uv[hi[(i + 1) % n]]) for i in range(n)]

    total, n_pts = 0.0, 0
    for x1, y1, x2, y2 in mask_lines:
        a = np.array([x1, y1], dtype=np.float64)
        b = np.array([x2, y2], dtype=np.float64)
        for t in np.linspace(0.0, 1.0, num_samples):
            p = a + t * (b - a)
            d = min(_point_to_segment_distance(p, sa, sb) for sa, sb in cube_segs)
            total += d
            n_pts += 1
    return total / max(n_pts, 1)


def silhouette_dt_loss(corners_3d, K, mask_gt, dt_to_mask):
    """Symmetric distance-transform silhouette loss.

    L = mean(distance to mask over pred-but-not-mask pixels)
      + mean(distance to pred over mask-but-not-pred pixels)

    dt_to_mask: precomputed per-pixel distance to the mask interior
                (0 inside mask, >0 outside).

    Provides a non-flat gradient even when pred and mask don't overlap,
    so the cube can't 'escape' the mask region.
    """
    pred = _rasterize_cube_silhouette(corners_3d, K, mask_gt.shape)
    if pred is None:
        return 1e3
    # Pred pixels outside the mask → penalized by distance to mask
    out_pen = (pred.astype(np.float32) * dt_to_mask).sum() / max(pred.sum(), 1)
    # Mask pixels outside pred → penalized by distance to pred
    # cv2.distanceTransform measures distance to nearest ZERO pixel,
    # so the foreground (pred==1) must be inverted to be the "zero" target.
    dt_to_pred = cv2.distanceTransform(
        (~pred).astype(np.uint8), cv2.DIST_L2, 3
    )
    unc_pen = (mask_gt.astype(np.float32) * dt_to_pred).sum() / max(mask_gt.sum(), 1)
    return float(out_pen + unc_pen)


_AXIS_VEC = {"x": np.array([1.0, 0.0, 0.0]),
             "y": np.array([0.0, 1.0, 0.0]),
             "z": np.array([0.0, 0.0, 1.0])}


def _axis_align_loss(rv, target_axis):
    """1 - (cube_z · target_axis)²   (sign-free)."""
    if target_axis is None:
        return 0.0
    R = Rscipy.from_rotvec(rv).as_matrix()
    z = R[:, 2]                          # cube +Z in cam frame
    d = float(z @ target_axis)
    return 1.0 - d * d


def total_loss(params, p_world, size, K, mask_gt, silhouette_aux,
               alpha, beta, gamma=0.0, target_axis=None,
               silhouette_mode="line"):
    """silhouette_aux:
        silhouette_mode='dt'    → precomputed dt_to_mask  (H, W) float32
        silhouette_mode='line'  → mask_lines              (N, 4)  float64
    """
    c, rv = params[:3], params[3:6]
    contact = contact_loss(p_world, c, rv, size)
    silhouette = 0.0
    if beta > 0.0:
        corners = cube_corners(c, rv, size)
        if silhouette_mode == "dt":
            silhouette = silhouette_dt_loss(corners, K, mask_gt, silhouette_aux)
        elif silhouette_mode == "line":
            silhouette = line_alignment_loss(corners, K, silhouette_aux)
    axis = gamma * _axis_align_loss(rv, target_axis) if gamma > 0.0 else 0.0
    return alpha * contact + beta * silhouette + axis


def viz(rgb_path, mask_gt, contact_verts, corners, K, out_path,
        cube_center=None, cube_rotvec=None, cube_size=0.05):
    fx, fy, cx, cy = K
    img = cv2.imread(rgb_path)
    H, W = img.shape[:2]
    overlay = img.copy()
    overlay[mask_gt] = (
        overlay[mask_gt] * 0.5 + np.array([0, 0, 200]) * 0.5
    ).astype(np.uint8)
    uv = project(corners, fx, fy, cx, cy).astype(np.int32)
    edges = [(0,1),(0,2),(1,3),(2,3),
             (4,5),(4,6),(5,7),(6,7),
             (0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        cv2.line(overlay, tuple(uv[i]), tuple(uv[j]), (0, 255, 255), 2, cv2.LINE_AA)
    cv_uv = project(contact_verts, fx, fy, cx, cy).astype(np.int32)
    for u, v in cv_uv:
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(overlay, (u, v), 3, (0, 255, 0), -1, cv2.LINE_AA)
    # Cube +Z arrow
    if cube_center is not None and cube_rotvec is not None:
        R = Rscipy.from_rotvec(cube_rotvec).as_matrix()
        arrow_len = cube_size * 1.5
        ends = np.stack([cube_center, cube_center + R[:, 2] * arrow_len], axis=0)
        if (ends[:, 2] > 1e-6).all():
            puv = project(ends, fx, fy, cx, cy).astype(np.int32)
            cv2.arrowedLine(overlay, tuple(puv[0]), tuple(puv[1]),
                            (255, 0, 255), 3, cv2.LINE_AA, tipLength=0.25)
            cv2.putText(overlay, "Z", tuple(puv[1] + np.array([4, -4])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, overlay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--mask", required=True, help="binary cube mask image")
    ap.add_argument("--rgb", default=None, help="optional RGB for visualization")
    ap.add_argument("--hand", default="left", choices=["left", "right"])
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--cube_size", type=float, default=0.05)
    ap.add_argument("--img_focal", type=float, default=497.77)
    ap.add_argument("--img_cx", type=float, default=None)
    ap.add_argument("--img_cy", type=float, default=None)
    ap.add_argument("--fingers", default="thumb,index,middle",
                    help="comma-separated; subset of {thumb,index,middle,ring,pinky}")
    ap.add_argument("--full_palmar", action="store_true",
                    help="Use all palmar verts on chosen fingers (default: tip+palmar).")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="contact-loss weight")
    ap.add_argument("--beta", type=float, default=0.01,
                    help="silhouette term weight (loss is in pixel units; "
                         "larger → stronger silhouette pull)")
    ap.add_argument("--silhouette_mode", choices=["line", "dt"], default="dt",
                    help="line: Hough-line alignment of mask straight edges "
                         "→ cube silhouette (robust to hand occlusion); "
                         "dt:   symmetric distance-transform loss (penalises "
                         "both 'cube outside mask' and 'mask uncovered').")
    ap.add_argument("--line_min_length", type=int, default=20,
                    help="(line mode) Hough min line length (px)")
    ap.add_argument("--line_threshold", type=int, default=20,
                    help="(line mode) Hough accumulator threshold")
    ap.add_argument("--line_max_gap", type=int, default=5,
                    help="(line mode) Hough max gap between line points")
    ap.add_argument("--target_axis", choices=["x", "y", "z", "none"],
                    default="none",
                    help="If set, soft-align cube +Z with this cam axis "
                         "(sign-free).  Use 'x' for R/L, 'y' for U/D, "
                         "'z' for F/B; 'none' disables.")
    ap.add_argument("--axis_weight", type=float, default=0.1,
                    help="weight on (1 - (cube_z·target_axis)^2) term")
    ap.add_argument("--out_pose", default=None)
    ap.add_argument("--out_viz", default=None)
    args = ap.parse_args()

    data = np.load(args.npz)
    verts = data[f"verts_{args.hand}"][args.frame]                # (778, 3) cam frame
    valid = data["valid"][0 if args.hand == "left" else 1][args.frame]
    if not valid:
        raise RuntimeError(f"frame {args.frame} invalid for {args.hand}")

    palmar = np.load(os.path.join(ASSETS, f"palmar_mask_{args.hand}.npy")).astype(bool)
    tip    = np.load(os.path.join(ASSETS, f"fingertip_mask_{args.hand}.npy")).astype(bool)
    parts  = np.load(os.path.join(ASSETS, f"finger_part_{args.hand}.npy"))

    chosen = set()
    for f in args.fingers.split(","):
        chosen |= FINGER_JOINTS[f.strip()]
    in_chosen = np.isin(parts, list(chosen))
    sel = palmar & in_chosen
    if not args.full_palmar:
        sel &= tip
    pts = verts[sel]
    if len(pts) == 0:
        raise RuntimeError("no contact verts after filtering")
    print(f"[{args.hand}] contact verts: {len(pts)} from fingers={args.fingers}")

    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(args.mask)
    mask_gt = mask > 127
    H, W = mask_gt.shape
    fx = fy = args.img_focal
    cx = args.img_cx if args.img_cx is not None else W / 2
    cy = args.img_cy if args.img_cy is not None else H / 2
    K = (fx, fy, cx, cy)
    print(f"image {W}x{H}  fx={fx}  cx={cx} cy={cy}  cube_size={args.cube_size}")

    target_vec = None if args.target_axis == "none" else _AXIS_VEC[args.target_axis]
    gamma = args.axis_weight if target_vec is not None else 0.0

    if args.silhouette_mode == "dt":
        # 0 inside mask, positive outside. cv2.distanceTransform measures
        # distance to nearest ZERO pixel, so feed ~mask.
        silhouette_aux = cv2.distanceTransform(
            (~mask_gt).astype(np.uint8), cv2.DIST_L2, 3
        )
        aux_info = "DT"
    else:
        silhouette_aux = detect_mask_lines(
            mask_gt,
            min_length=args.line_min_length,
            max_gap=args.line_max_gap,
            threshold=args.line_threshold,
        )
        aux_info = f"line ({len(silhouette_aux)} segments)"

    params0 = np.concatenate([pts.mean(axis=0), np.zeros(3)])

    print("Stage 1 — contact only ...")
    r1 = minimize(total_loss, params0,
                  args=(pts, args.cube_size, K, mask_gt, silhouette_aux,
                        args.alpha, 0.0, 0.0, None, args.silhouette_mode),
                  method="Powell", options={"maxiter": 500, "xtol": 1e-6, "ftol": 1e-6})
    print(f"  contact loss = {r1.fun:.6f}")

    print(f"Stage 2 — + silhouette [{aux_info}, β={args.beta}]  "
          f"axis target = {args.target_axis}, γ = {gamma} ...")
    r2 = minimize(total_loss, r1.x,
                  args=(pts, args.cube_size, K, mask_gt, silhouette_aux,
                        args.alpha, args.beta, gamma, target_vec,
                        args.silhouette_mode),
                  method="Powell", options={"maxiter": 1000, "xtol": 1e-6, "ftol": 1e-6})
    c, rv = r2.x[:3], r2.x[3:6]
    quat = Rscipy.from_rotvec(rv).as_quat()
    cont = contact_loss(pts, c, rv, args.cube_size)
    iou = silhouette_iou(cube_corners(c, rv, args.cube_size), K, mask_gt)
    print(f"  total = {r2.fun:.6f}  |  contact = {cont:.6f}  IoU = {iou:.4f}")

    out_pose = args.out_pose or os.path.join(
        os.path.dirname(os.path.abspath(args.npz)), "cube_pose.npz"
    )
    np.savez(
        out_pose,
        center=c, rotvec=rv, quat_xyzw=quat,
        size=args.cube_size,
        contact_loss=cont, iou=iou,
        hand=args.hand, frame=args.frame, img_focal=fx, cx=cx, cy=cy,
    )
    print(f"saved {out_pose}")

    if args.rgb is not None:
        out_viz = args.out_viz or os.path.splitext(out_pose)[0] + "_viz.png"
        viz(args.rgb, mask_gt, pts, cube_corners(c, rv, args.cube_size), K, out_viz,
            cube_center=c, cube_rotvec=rv, cube_size=args.cube_size)
        print(f"saved {out_viz}")


if __name__ == "__main__":
    main()
