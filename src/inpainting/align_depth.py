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

Z_MIN, Z_MAX = 0.05, 10.0   # meters
EPS = 1e-6


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
    args = ap.parse_args()

    pd = args.processed_demo
    depth_raw = np.load(pd / "depth_processor" / "depth_raw.npy").astype(np.float32)
    T, H, W = depth_raw.shape

    hd_l = np.load(pd / "hand_processor" / "hand_data_left.npz")
    hd_r = np.load(pd / "hand_processor" / "hand_data_right.npz")
    T_use = min(T, hd_l["kpts_2d"].shape[0], hd_r["kpts_2d"].shape[0])

    a_arr = np.full(T_use, np.nan, dtype=np.float32)
    b_arr = np.full(T_use, np.nan, dtype=np.float32)

    for t in range(T_use):
        uv_list, z_list = [], []
        if hd_l["hand_detected"][t]:
            uv_list.append(hd_l["kpts_2d"][t])
            z_list.append(hd_l["kpts_3d"][t, :, 2])
        if hd_r["hand_detected"][t]:
            uv_list.append(hd_r["kpts_2d"][t])
            z_list.append(hd_r["kpts_3d"][t, :, 2])
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

    valid = ~np.isnan(a_arr)
    if not valid.any():
        raise RuntimeError("No frame had ≥3 valid hand anchors — cannot align depth.")
    a_med = float(np.median(a_arr[valid]))
    b_med = float(np.median(b_arr[valid]))
    a_arr[~valid] = a_med
    b_arr[~valid] = b_med
    print(f"[info] aligned {valid.sum()}/{T_use} frames; "
          f"median (a, b) = ({a_med:.3f}, {b_med:.3f})")

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
