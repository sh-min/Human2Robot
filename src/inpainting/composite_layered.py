"""Explicit 4-layer compositor:

    layer order (top → bottom):
        1. front-MCP robot   = robot pixels with r_z <  z_MCP_t
        2. cube              = bg content at cube_mask pixels (cube isolated layer)
        3. behind-MCP robot  = robot pixels with r_z >= z_MCP_t
        4. bg                = inpainted background

Pixel rule (applied in reverse so top layers overwrite):
    final = bg
    final[behind_robot] = robot_rgb
    final[cube_mask]    = bg              # cube is "in" the inpainted bg already
    final[front_robot]  = robot_rgb

Inputs:
    <pd>/inpaint_processor/video_human_inpaint.mkv
    <pd>/overlay_processor/robot_{rgb,depth,mask}.npy
    <pd>/overlay_processor_cube_v2/cube_mask.npy          (from composite_cube_inpaint_first.py)
    --hawor_npz                                            joints_*[t, MCP, 2], valid

Outputs:
    <pd>/overlay_processor_layered/video_overlay.mkv      final 4-layer composite
    <pd>/overlay_processor_layered/debug_layers.mkv       4-up: front | cube | behind | final
    <pd>/overlay_processor_layered/z_mcp.npy
"""
import argparse
from pathlib import Path

import cv2
import mediapy as media
import numpy as np
from scipy.ndimage import gaussian_filter1d

from contact_shadow import contact_shadow_alpha, fit_support_plane

MCP_JOINT = 9   # default; --threshold_joint can override


def _soft_alpha(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Binary mask → float alpha in [0,1] with Gaussian-blurred edges."""
    if sigma <= 0:
        return mask.astype(np.float32)
    k = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(mask.astype(np.float32), (k, k), sigma)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--hawor_npz", type=Path, required=True)
    ap.add_argument("--depth_bias", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--bg_video", default="inpaint_processor/video_human_inpaint.mkv")
    ap.add_argument("--cube_mask_npy",
                    default="cube_layer/cube_mask_amodal.npy")
    ap.add_argument("--debug", action="store_true",
                    help="also emit a 4-up debug video showing each layer")
    ap.add_argument("--zmcp_sigma_t", type=float, default=8.0,
                    help="Gaussian sigma (frames) on the per-frame z_MCP_t time "
                         "series. Damps front/behind threshold jitter. 0 = off.")
    ap.add_argument("--edge_sigma", type=float, default=1.5,
                    help="Gaussian sigma (px) on each layer's binary mask, "
                         "producing soft alpha for alpha-blending. 0 = hard edges.")
    ap.add_argument("--contact_shadow", dest="contact_shadow",
                    action="store_true", default=True,
                    help="Cast a geometry-grounded contact shadow of the robot "
                         "onto the support surface (default on).")
    ap.add_argument("--no_contact_shadow", dest="contact_shadow",
                    action="store_false",
                    help="Disable the contact shadow.")
    ap.add_argument("--shadow_depth",
                    default="depth_processor/depth_aligned.npy",
                    help="Metric scene depth (T,H,W) for the plane fit.")
    ap.add_argument("--shadow_opacity", type=float, default=0.6,
                    help="Max darkening of the contact shadow, 0..1.")
    ap.add_argument("--shadow_blur", type=float, default=6.0,
                    help="Gaussian sigma (px) softening the shadow footprint.")
    ap.add_argument("--light_dir", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="Light travel direction in the camera frame; must "
                         "match the render stage. Default: scene_lighting.")
    ap.add_argument("--threshold_joint", type=int, default=5,
                    help="MANO joint used as the front/behind depth threshold. "
                         "Default 5=idx-MCP (~0.28 m, ~2:1 front:behind split). "
                         "Common alternatives: 0=wrist (~0.24 m, cube-heavy), "
                         "9=mid-MCP (~0.30 m, robot-heavy).")
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

    # Contact shadow setup. Needs metric scene depth (stage 7); if absent we
    # warn and skip rather than fail the whole composite over an optional cue.
    scene_depth = None
    if args.contact_shadow:
        sd_path = pd / args.shadow_depth
        if sd_path.exists():
            scene_depth = np.load(sd_path)[:T]
            F = float(ri["img_focal"])
            cx_s, cy_s = W / 2.0, H / 2.0
            print(f"[shadow] on: depth={sd_path.name}, opacity={args.shadow_opacity}, "
                  f"blur={args.shadow_blur}px")
        else:
            print(f"[shadow] WARN: {sd_path} not found — skipping contact shadow")
    plane = None

    z_mcp = np.full(T, np.nan, dtype=np.float32)
    for t in range(T):
        zs = []
        if valid[0, t]:
            zs.append(joints_l[t, args.threshold_joint, 2])
        if valid[1, t]:
            zs.append(joints_r[t, args.threshold_joint, 2])
        if zs:
            z_mcp[t] = float(np.mean(zs))

    if args.zmcp_sigma_t > 0:
        valid_z = ~np.isnan(z_mcp)
        if valid_z.any():
            # fill nans with nearest valid before smoothing
            idx = np.where(valid_z, np.arange(T), -1)
            for i in range(T):
                if idx[i] == -1:
                    nearest = np.where(valid_z)[0]
                    if len(nearest):
                        z_mcp[i] = z_mcp[nearest[np.argmin(np.abs(nearest - i))]]
            z_mcp = gaussian_filter1d(z_mcp.astype(np.float32),
                                      sigma=args.zmcp_sigma_t, mode="nearest")
            print(f"[smooth] z_MCP gauss sigma_t={args.zmcp_sigma_t} frames "
                  f"→ range now [{z_mcp.min():.3f}, {z_mcp.max():.3f}] m "
                  f"(std {z_mcp.std()*100:.2f} cm)")

    out_frames = bg.copy()
    if args.debug:
        dbg = np.zeros((T, H, 4 * W + 3 * 6, 3), dtype=np.uint8)
        for i in range(1, 4):
            dbg[:, :, i * W + (i - 1) * 6:i * W + i * 6] = 255

    print(f"[info] T={T}, {W}x{H}, threshold_joint={args.threshold_joint}, "
          f"depth_bias={args.depth_bias}, edge_sigma={args.edge_sigma}px")

    n_front_total, n_cube_total, n_behind_total = 0, 0, 0
    for t in range(T):
        z_t = float(z_mcp[t]) if np.isfinite(z_mcp[t]) else np.inf

        behind_robot = r_mask[t] & ((r_z[t] + args.depth_bias) >= z_t)
        front_robot  = r_mask[t] & ((r_z[t] + args.depth_bias) <  z_t)

        # Alpha-blend layered composite. Each layer's binary mask is blurred
        # into an alpha map (0..1), then the layer is alpha-blended over the
        # accumulating image. Order: bg → behind robot → cube → front robot.
        acc = bg[t].astype(np.float32)

        # Contact shadow darkens the surface (bg) before the robot is drawn on
        # top; the alpha is already zero under the robot mask.
        if scene_depth is not None:
            plane = fit_support_plane(scene_depth[t].astype(np.float32),
                                      r_mask[t], F, cx_s, cy_s, prev=plane)
            sh = contact_shadow_alpha(
                scene_depth[t].astype(np.float32), r_z[t], r_mask[t], plane,
                F, cx_s, cy_s, light_dir=args.light_dir,
                opacity=args.shadow_opacity, blur=args.shadow_blur)
            acc *= (1.0 - sh[..., None])

        for layer_mask, content in [
            (behind_robot, r_rgb[t]),
            (cube_m[t],    bg[t]),
            (front_robot,  r_rgb[t]),
        ]:
            alpha = _soft_alpha(layer_mask, args.edge_sigma)[..., None]
            acc = alpha * content.astype(np.float32) + (1.0 - alpha) * acc
        out_frames[t] = np.clip(acc, 0, 255).astype(np.uint8)

        n_front_total  += int(front_robot.sum())
        n_cube_total   += int(cube_m[t].sum())
        n_behind_total += int(behind_robot.sum())

        if args.debug:
            # Panel 1: bg with only front-MCP robot
            p1 = bg[t].copy(); p1[front_robot] = r_rgb[t][front_robot]
            # Panel 2: bg with only cube
            p2 = bg[t].copy()  # bg already has the cube
            p2_mask = np.zeros((H, W, 3), dtype=np.uint8)
            p2_mask[cube_m[t]] = bg[t][cube_m[t]]
            # Panel 3: bg with only behind-MCP robot
            p3 = bg[t].copy(); p3[behind_robot] = r_rgb[t][behind_robot]
            # Panel 4: final
            p4 = out_frames[t]
            dbg[t, :,           :W]        = p1
            dbg[t, :,   W +  6: 2 * W +  6] = p2_mask
            dbg[t, :, 2 * W + 12: 3 * W + 12] = p3
            dbg[t, :, 3 * W + 18:]          = p4

        if (t + 1) % 100 == 0:
            print(f"  {t+1}/{T}  z_MCP={z_mcp[t]:.3f}m  "
                  f"front={int(front_robot.sum())}  cube={int(cube_m[t].sum())}  "
                  f"behind={int(behind_robot.sum())}")

    out_dir = pd / "overlay_processor_layered"
    out_dir.mkdir(parents=True, exist_ok=True)
    media.write_video(str(out_dir / "video_overlay.mp4"), out_frames,
                      fps=args.fps, codec="libx264")
    np.save(out_dir / "z_mcp.npy", z_mcp)
    print(f"[ok] wrote {out_dir/'video_overlay.mp4'}")
    if args.debug:
        media.write_video(str(out_dir / "debug_layers.mkv"), dbg,
                          fps=args.fps, codec="libx264")
        print(f"[ok] wrote {out_dir/'debug_layers.mkv'}  "
              f"(panels: front-MCP | cube isolated | behind-MCP | final)")
    print(f"[info] totals (sum over all frames): "
          f"front-MCP robot px={n_front_total}, cube px={n_cube_total}, "
          f"behind-MCP robot px={n_behind_total}")


if __name__ == "__main__":
    main()
