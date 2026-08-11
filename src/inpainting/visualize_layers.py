"""Visualize the three composite layers in one frame.

Each layer keeps its actual content (robot rgb / object from bg) but gets a
color tint so you can tell them apart:
    front-MCP robot  →  blue tint
    object (isolated)  →  green tint
    behind-MCP robot →  red tint
    rest of bg       →  shown at half brightness

Paint order matches the prescribed composite (front-MCP on top):
    bg* → behind-MCP → object → front-MCP

Output:
    <pd>/overlay_processor_layered/layers_tinted.mp4

Usage:
    python visualize_layers.py --processed_demo /result/cam0_inpaint/cam0/0 \
        --hawor_npz /data/RFM_proj/cam0_hawor/retarget_input.npz
"""
import argparse
from pathlib import Path

import mediapy as media
import numpy as np

MCP_JOINT = 9
TINT_FRONT  = np.array([ 60, 120, 255], dtype=np.float32)  # blue
TINT_OBJECT   = np.array([ 60, 255, 120], dtype=np.float32)  # green
TINT_BEHIND = np.array([255,  80,  80], dtype=np.float32)  # red
TINT_ALPHA  = 0.45     # how much of the tint to mix in
BG_DIM      = 0.5      # multiply un-layered bg by this


def _tint(rgb: np.ndarray, mask: np.ndarray, tint: np.ndarray) -> None:
    """In-place: mix `tint` into rgb at `mask=True`."""
    if not mask.any():
        return
    rgb[mask] = (rgb[mask].astype(np.float32) * (1 - TINT_ALPHA)
                 + tint * TINT_ALPHA).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--hawor_npz", type=Path, required=True)
    ap.add_argument("--depth_bias", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--bg_video", default="inpaint_processor/video_human_inpaint.mkv")
    ap.add_argument("--object_mask_npy",
                    default="overlay_processor_object_v2/object_mask.npy")
    args = ap.parse_args()

    pd = args.processed_demo
    bg     = media.read_video(str(pd / args.bg_video))
    r_rgb  = np.load(pd / "overlay_processor" / "robot_rgb.npy")
    r_z    = np.load(pd / "overlay_processor" / "robot_depth.npy").astype(np.float32)
    r_mask = np.load(pd / "overlay_processor" / "robot_mask.npy").astype(bool)
    object_m = np.load(pd / args.object_mask_npy).astype(bool)

    ri = np.load(args.hawor_npz)
    joints_l = ri["joints_left"].astype(np.float64)
    joints_r = ri["joints_right"].astype(np.float64)
    valid = ri["valid"]

    T = min(bg.shape[0], r_rgb.shape[0], object_m.shape[0], joints_l.shape[0])
    bg, r_rgb, r_z, r_mask, object_m = bg[:T], r_rgb[:T], r_z[:T], r_mask[:T], object_m[:T]
    H, W = bg.shape[1], bg.shape[2]

    out_frames = np.zeros_like(bg)

    print(f"[info] T={T}, {W}x{H}, depth_bias={args.depth_bias}, "
          f"tints: front=blue, object=green, behind=red")

    for t in range(T):
        zs = []
        if valid[0, t]:
            zs.append(joints_l[t, MCP_JOINT, 2])
        if valid[1, t]:
            zs.append(joints_r[t, MCP_JOINT, 2])
        z_t = float(np.mean(zs)) if zs else np.inf

        behind_robot = r_mask[t] & ((r_z[t] + args.depth_bias) >= z_t)
        front_robot  = r_mask[t] & ((r_z[t] + args.depth_bias) <  z_t)

        # bg dimmed
        frame = (bg[t].astype(np.float32) * BG_DIM).astype(np.uint8)
        # layer 1 (bottom): behind-MCP robot, red tint
        if behind_robot.any():
            frame[behind_robot] = r_rgb[t][behind_robot]
            _tint(frame, behind_robot, TINT_BEHIND)
        # layer 2: object (bg at object_mask pixels), green tint
        if object_m[t].any():
            frame[object_m[t]] = bg[t][object_m[t]]
            _tint(frame, object_m[t], TINT_OBJECT)
        # layer 3 (top): front-MCP robot, blue tint
        if front_robot.any():
            frame[front_robot] = r_rgb[t][front_robot]
            _tint(frame, front_robot, TINT_FRONT)

        out_frames[t] = frame

        if (t + 1) % 100 == 0:
            print(f"  {t+1}/{T}  z_MCP={z_t:.3f}m  "
                  f"front={int(front_robot.sum())}  "
                  f"object={int(object_m[t].sum())}  "
                  f"behind={int(behind_robot.sum())}")

    out_dir = pd / "overlay_processor_layered"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "layers_tinted.mp4"
    media.write_video(str(dst), out_frames, fps=args.fps, codec="libx264")
    print(f"[ok] wrote {dst}")
    print("    layer key:  blue=front-MCP robot   green=object   red=behind-MCP robot")


if __name__ == "__main__":
    main()
