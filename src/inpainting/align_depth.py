"""Stage B — turn DA-V2's relative depth into metric depth using HaWoR anchors.

DA-V2 emits disparity-like values `d_pred` where higher = closer. We model

    d_pred(u, v)  ≈  a / Z_metric(u, v)  +  b              [per frame]

and solve `(a, b)` per frame by least squares against the HaWoR 21-joint
keypoints, whose cam-frame Z is metric (meters). Both hands contribute jointly
when both are valid. Frames where neither hand is valid fall back to the
global median `(a, b)` so the metric scale is at least consistent.

The inverted depth map

    Z_metric(u, v) =  a / (d_pred(u, v) - b)

is clipped to a sensible (0.05 m, 10 m) range, then stored as float16.

Inputs:
    <processed_demo>/depth_processor/depth_raw.npy
    <processed_demo>/hand_processor/hand_data_{left,right}.npz   (kpts_2d, kpts_3d)

Outputs:
    <processed_demo>/depth_processor/depth_aligned.npy   (T,H,W) float16 meters
    <processed_demo>/depth_processor/depth_align_params.npz   per-frame (a,b)

Usage:
    python align_depth.py --processed_demo /result/cam0_inpaint/cam0/0
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

Z_MIN, Z_MAX = 0.05, 10.0   # meters
EPS = 1e-6
A_TOL = 3.0                 # per-frame `a` must stay within this factor of global
SMOOTH_WIN = 15             # frames, temporal median over the fitted (a, b)


def _sample_bilinear(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sample a (H,W) float map at float (N,2) (u,v) pixel coords."""
    H, W = img.shape
    u = np.clip(uv[:, 0], 0, W - 1.001)
    v = np.clip(uv[:, 1], 0, H - 1.001)
    u0 = np.floor(u).astype(np.int64); u1 = u0 + 1
    v0 = np.floor(v).astype(np.int64); v1 = v0 + 1
    du = u - u0; dv = v - v0
    s = ((1 - du) * (1 - dv) * img[v0, u0]
         + du   * (1 - dv) * img[v0, u1]
         + (1 - du) * dv   * img[v1, u0]
         + du   * dv       * img[v1, u1])
    return s


def _fit_ab(d_pred: np.ndarray, inv_z: np.ndarray) -> tuple[float, float]:
    """LSQ for d_pred = a * inv_z + b. Returns (a, b)."""
    A = np.stack([inv_z, np.ones_like(inv_z)], axis=1)
    sol, *_ = np.linalg.lstsq(A, d_pred, rcond=None)
    return float(sol[0]), float(sol[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument(
        "--hawor_npz", type=Path, default=None,
        help="Anchor on this retargeting npz (joints_left/joints_right and "
             "img_focal) instead of hand_processor/hand_data_*.npz. Required "
             "when the depth feeds the compositor: the robot render and the "
             "depth split are built on these joints, and the two hand sources "
             "do not share a metric scale.",
    )
    args = ap.parse_args()

    pd = args.processed_demo
    depth_raw = np.load(pd / "depth_processor" / "depth_raw.npy").astype(np.float32)
    T, H, W = depth_raw.shape

    if args.hawor_npz is not None:
        pose = np.load(args.hawor_npz)
        joints = (pose["joints_left"], pose["joints_right"])
        hand_valid = pose["valid"].astype(bool)
        focal = float(pose["img_focal"])
        cx, cy = W / 2.0, H / 2.0
        T_use = min(T, joints[0].shape[0], joints[1].shape[0])
        print(f"[info] anchoring on {args.hawor_npz.name} (focal={focal:.1f})")

        def frame_anchors(t: int) -> tuple[list, list]:
            uv_list, z_list = [], []
            for hand in (0, 1):
                if not hand_valid[hand, t]:
                    continue
                joint = joints[hand][t]
                z = joint[:, 2]
                usable = z > Z_MIN
                if not usable.any():
                    continue
                uv_list.append(np.stack([
                    focal * joint[usable, 0] / z[usable] + cx,
                    focal * joint[usable, 1] / z[usable] + cy,
                ], axis=1))
                z_list.append(z[usable])
            return uv_list, z_list
    else:
        hd_l = np.load(pd / "hand_processor" / "hand_data_left.npz")
        hd_r = np.load(pd / "hand_processor" / "hand_data_right.npz")
        T_use = min(T, hd_l["kpts_2d"].shape[0], hd_r["kpts_2d"].shape[0])

        def frame_anchors(t: int) -> tuple[list, list]:
            uv_list, z_list = [], []
            for hd in (hd_l, hd_r):
                if hd["hand_detected"][t]:
                    uv_list.append(hd["kpts_2d"][t])
                    z_list.append(hd["kpts_3d"][t, :, 2])
            return uv_list, z_list

    a_arr = np.full(T_use, np.nan, dtype=np.float32)
    b_arr = np.full(T_use, np.nan, dtype=np.float32)
    pooled_d: list[np.ndarray] = []
    pooled_inv_z: list[np.ndarray] = []

    for t in range(T_use):
        uv_list, z_list = frame_anchors(t)
        if not uv_list:
            continue
        uv = np.concatenate(uv_list, axis=0)
        z = np.concatenate(z_list, axis=0)
        keep = (z > Z_MIN) & (z < Z_MAX) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) \
               & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
        if keep.sum() < 3:
            continue
        d = _sample_bilinear(depth_raw[t], uv[keep])
        a_arr[t], b_arr[t] = _fit_ab(d, 1.0 / z[keep])
        pooled_d.append(d)
        pooled_inv_z.append(1.0 / z[keep])

    if not pooled_d:
        raise RuntimeError("No frame had ≥3 valid hand anchors — cannot align depth.")

    # One frame's 21 joints span a narrow inverse-depth range, so the two
    # parameters are barely identifiable and noise can even flip the sign of
    # `a`.  Pooling every frame's anchors spans the whole trajectory instead,
    # which conditions the fit; the per-frame solution is then only kept where
    # it is physical and close to that global one.
    a_glob, b_glob = _fit_ab(np.concatenate(pooled_d),
                             np.concatenate(pooled_inv_z))
    valid = (np.isfinite(a_arr) & (a_arr > 0)
             & (a_arr > a_glob / A_TOL) & (a_arr < a_glob * A_TOL))
    a_arr[~valid] = a_glob
    b_arr[~valid] = b_glob
    print(f"[info] global (a, b) = ({a_glob:.3f}, {b_glob:.3f}); "
          f"kept {int(valid.sum())}/{T_use} per-frame fits, "
          f"{int((~valid).sum())} fell back to global")

    # DA-V2's relative scale drifts slowly for a static camera, so a temporal
    # median removes the remaining single-frame outliers without lagging.
    if T_use >= SMOOTH_WIN:
        a_arr = median_filter(a_arr, size=SMOOTH_WIN, mode="nearest")
        b_arr = median_filter(b_arr, size=SMOOTH_WIN, mode="nearest")

    # Z_metric = a / (d_pred - b). Where (d_pred - b) ≤ 0 we get nonsense — clip.
    inv_z_pred = (depth_raw[:T_use] - b_arr[:, None, None]) / np.maximum(a_arr[:, None, None], EPS)
    z_metric = 1.0 / np.maximum(inv_z_pred, 1.0 / Z_MAX)
    z_metric = np.clip(z_metric, Z_MIN, Z_MAX).astype(np.float16)

    out_dir = pd / "depth_processor"
    np.save(out_dir / "depth_aligned.npy", z_metric)
    np.savez(out_dir / "depth_align_params.npz", a=a_arr, b=b_arr,
             valid_frames=valid.astype(np.uint8))
    print(f"[ok] wrote {out_dir/'depth_aligned.npy'}  "
          f"(meters, range [{z_metric.min():.2f}, {z_metric.max():.2f}])")


if __name__ == "__main__":
    main()
