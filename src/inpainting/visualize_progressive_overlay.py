"""Progressive 4-stage layered overlay: dump one video per cumulative layer.

For each frame we write 4 versions in lockstep:

    L1  bg only                                = inpainted bg
    L2  bg + behind-MCP robot                 (alpha blend)
    L3  L2 + cube                              (cube layer = inpainted bg at cube_mask)
    L4  L3 + front-MCP robot                   = final composite

Same alpha-blend / soft-edge / z_MCP-smoothing knobs as composite_layered.py.

Outputs (under <pd>/overlay_processor_layered/):
    progressive_L1_bg.mp4
    progressive_L2_behind.mp4
    progressive_L3_cube.mp4
    progressive_L4_front.mp4    # same as video_overlay.mkv

Usage:
    python visualize_progressive_overlay.py \
        --processed_demo /result/cam0_inpaint/cam0/0 \
        --hawor_npz /data/RFM_proj/cam0_hawor/retarget_input.npz \
        --cube_mask_npy cube_layer/cube_mask_clean.npy
"""
import argparse
from pathlib import Path

import cv2
import mediapy as media
import numpy as np
from scipy.ndimage import gaussian_filter1d

MCP_JOINT = 9


def _soft_alpha(mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return mask.astype(np.float32)
    k = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(mask.astype(np.float32), (k, k), sigma)


def _blend(acc: np.ndarray, content: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return a * content.astype(np.float32) + (1.0 - a) * acc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--hawor_npz", type=Path, required=True)
    ap.add_argument("--depth_bias", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--bg_video", default="inpaint_processor/video_human_inpaint.mkv")
    ap.add_argument("--cube_mask_npy", default="overlay_processor_cube_v2/cube_mask.npy")
    ap.add_argument("--zmcp_sigma_t", type=float, default=8.0)
    ap.add_argument("--edge_sigma", type=float, default=1.5)
    args = ap.parse_args()

    pd = args.processed_demo
    bg     = media.read_video(str(pd / args.bg_video))
    r_rgb  = np.load(pd / "overlay_processor" / "robot_rgb.npy")
    r_z    = np.load(pd / "overlay_processor" / "robot_depth.npy").astype(np.float32)
    r_mask = np.load(pd / "overlay_processor" / "robot_mask.npy").astype(bool)
    cube_m = np.load(pd / args.cube_mask_npy).astype(bool)

    ri = np.load(args.hawor_npz)
    joints_l = ri["joints_left"].astype(np.float64)
    joints_r = ri["joints_right"].astype(np.float64)
    valid = ri["valid"]

    T = min(bg.shape[0], r_rgb.shape[0], cube_m.shape[0], joints_l.shape[0])
    bg, r_rgb, r_z, r_mask, cube_m = bg[:T], r_rgb[:T], r_z[:T], r_mask[:T], cube_m[:T]
    H, W = bg.shape[1], bg.shape[2]

    # smooth z_MCP_t
    z_mcp = np.full(T, np.nan, dtype=np.float32)
    for t in range(T):
        zs = []
        if valid[0, t]:
            zs.append(joints_l[t, MCP_JOINT, 2])
        if valid[1, t]:
            zs.append(joints_r[t, MCP_JOINT, 2])
        if zs:
            z_mcp[t] = float(np.mean(zs))
    if args.zmcp_sigma_t > 0:
        valid_z = ~np.isnan(z_mcp)
        if valid_z.any():
            nearest = np.where(valid_z)[0]
            for i in range(T):
                if np.isnan(z_mcp[i]):
                    z_mcp[i] = z_mcp[nearest[np.argmin(np.abs(nearest - i))]]
            z_mcp = gaussian_filter1d(z_mcp.astype(np.float32),
                                      sigma=args.zmcp_sigma_t, mode="nearest")
            print(f"[smooth] z_MCP sigma_t={args.zmcp_sigma_t} → "
                  f"range [{z_mcp.min():.3f}, {z_mcp.max():.3f}] m")

    L1 = bg.copy()                                  # just bg
    L2 = np.zeros_like(bg)
    L3 = np.zeros_like(bg)
    L4 = np.zeros_like(bg)

    print(f"[info] T={T}, {W}x{H}, edge_sigma={args.edge_sigma}px")
    for t in range(T):
        z_t = float(z_mcp[t]) if np.isfinite(z_mcp[t]) else np.inf
        behind_robot = r_mask[t] & ((r_z[t] + args.depth_bias) >= z_t)
        front_robot  = r_mask[t] & ((r_z[t] + args.depth_bias) <  z_t)

        a_behind = _soft_alpha(behind_robot, args.edge_sigma)
        a_cube   = _soft_alpha(cube_m[t],    args.edge_sigma)
        a_front  = _soft_alpha(front_robot,  args.edge_sigma)

        s1 = bg[t].astype(np.float32)
        s2 = _blend(s1, r_rgb[t], a_behind)
        s3 = _blend(s2, bg[t],    a_cube)
        s4 = _blend(s3, r_rgb[t], a_front)

        L2[t] = np.clip(s2, 0, 255).astype(np.uint8)
        L3[t] = np.clip(s3, 0, 255).astype(np.uint8)
        L4[t] = np.clip(s4, 0, 255).astype(np.uint8)

        if (t + 1) % 100 == 0:
            print(f"  {t+1}/{T}")

    out_dir = pd / "overlay_processor_layered"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "progressive_L1_bg.mp4":     L1,
        "progressive_L2_behind.mp4": L2,
        "progressive_L3_cube.mp4":   L3,
        "progressive_L4_front.mp4":  L4,
    }
    for name, arr in paths.items():
        media.write_video(str(out_dir / name), arr, fps=args.fps, codec="libx264")
        print(f"[ok] wrote {out_dir/name}")


if __name__ == "__main__":
    main()
