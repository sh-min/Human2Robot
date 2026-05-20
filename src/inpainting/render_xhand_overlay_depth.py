"""Stage D — render the xhand robot and emit RGB + depth + mask separately.

This is the depth-aware variant of `render_xhand_overlay.py`. It does NOT
composite the robot into the raw video and does NOT clip to the arm mask —
that's the depth-aware compositor's job (`composite_depth_aware.py`).

Outputs (all length-T arrays, T = min(video, hawor)):
    <processed_demo>/overlay_processor/robot_rgb.npy    (T,H,W,3) uint8
    <processed_demo>/overlay_processor/robot_depth.npy  (T,H,W)   float16
                                                        meters in MANO cam frame.
                                                        +inf where the robot is not drawn.
    <processed_demo>/overlay_processor/robot_mask.npy   (T,H,W)   bool

Coordinate sanity:
    pyrender camera at identity (looks along -Z in OpenGL world)
    T_CV2GL = diag(1, -1, -1) maps MANO cam (OpenCV) → pyrender world.
    With the camera at origin, pyrender's returned depth value equals the
    original MANO cam Z (forward, meters), so no extra rescale.

Usage:
    PYOPENGL_PLATFORM=egl python -u render_xhand_overlay_depth.py \
        --processed_demo /result/cam0_inpaint/cam0/0 \
        --hawor_npz /data/RFM_proj/cam0_hawor/retarget_input.npz \
        --right_pkl /data/RFM_proj/cam0_hawor/qpos_xhand_right.pkl \
        --left_pkl  /data/RFM_proj/cam0_hawor/qpos_xhand_left.pkl
"""
import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mediapy as media
import numpy as np
import pyrender
from scipy.spatial.transform import Rotation

from _paths import XHAND_URDF_LEFT, XHAND_URDF_RIGHT, R_MANO_XHAND
from render_xhand_overlay import compute_fk, parse_urdf, T_CV2GL


def _render_robot_rgbd(hands_data, camera, renderer):
    """Return (rgb (H,W,3) uint8, depth (H,W) float32 meters, mask (H,W) bool)."""
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0., 0., 0., 0.])
    scene.add(camera, pose=np.eye(4))
    scene.add(pyrender.DirectionalLight(color=[1., 1., 1.], intensity=3.0), pose=np.eye(4))
    pl_pose = np.eye(4); pl_pose[:3, 3] = [0.3, -0.3, -0.5]
    scene.add(pyrender.PointLight(color=[0.8, 0.8, 0.8], intensity=2.0), pose=pl_pose)

    for side, wrist_pos, R_mano, qpos_dict, joints_tree, link_meshes in hands_data:
        R_cam_xhand = R_mano @ R_MANO_XHAND[side]
        R_pr = T_CV2GL @ R_cam_xhand
        t_pr = T_CV2GL @ wrist_pos

        T_root = np.eye(4); T_root[:3, :3] = R_pr; T_root[:3, 3] = t_pr
        link_T = compute_fk(joints_tree, qpos_dict, T_root)

        for lname, items in link_meshes.items():
            if lname not in link_T:
                continue
            for mesh, T_vis in items:
                pr_mesh = pyrender.Mesh.from_trimesh(mesh.copy(), smooth=False)
                scene.add(pr_mesh, pose=link_T[lname] @ T_vis)

    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    mask = depth > 0
    return color[..., :3], depth.astype(np.float32), mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--hawor_npz", type=Path, required=True)
    ap.add_argument("--right_pkl", type=Path, required=True)
    ap.add_argument("--left_pkl",  type=Path, required=True)
    ap.add_argument("--hand", choices=["left", "right", "both"], default="both")
    args = ap.parse_args()

    ri = np.load(args.hawor_npz)
    joints_left  = ri["joints_left"].astype(np.float64)
    joints_right = ri["joints_right"].astype(np.float64)
    go    = ri["mano_global_orient"]
    valid = ri["valid"]
    focal = float(ri["img_focal"])

    qr = pickle.load(open(args.right_pkl, "rb"))
    ql = pickle.load(open(args.left_pkl,  "rb"))
    right_data  = np.asarray(qr["data"])
    left_data   = np.asarray(ql["data"])
    right_jname = qr["joint_names"]
    left_jname  = ql["joint_names"]

    bg_path = args.processed_demo / "video_L.mp4"
    bg_frames = media.read_video(str(bg_path))
    T_vid = bg_frames.shape[0]
    img_h, img_w = bg_frames.shape[1], bg_frames.shape[2]

    T_use = min(T_vid, joints_left.shape[0])
    print(f"[info] video T={T_vid}, npz T={joints_left.shape[0]}, "
          f"using T={T_use}, hand={args.hand}")

    cx, cy = img_w / 2.0, img_h / 2.0
    camera = pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=cx, cy=cy,
                                       znear=0.01, zfar=10.0)
    renderer = pyrender.OffscreenRenderer(img_w, img_h)

    side_cfg = {}
    for s, urdf in (("right", XHAND_URDF_RIGHT), ("left", XHAND_URDF_LEFT)):
        if args.hand not in (s, "both"):
            continue
        side_cfg[s] = parse_urdf(Path(urdf))
    print("URDF loaded for:", list(side_cfg.keys()))

    rgb_buf   = np.zeros((T_use, img_h, img_w, 3), dtype=np.uint8)
    depth_buf = np.full((T_use, img_h, img_w), np.inf, dtype=np.float32)
    mask_buf  = np.zeros((T_use, img_h, img_w), dtype=bool)

    for t in range(T_use):
        hands_data = []
        for s in ("right", "left"):
            if s not in side_cfg:
                continue
            h_idx = 1 if s == "right" else 0
            if not valid[h_idx, t]:
                continue
            joints_tree, link_meshes = side_cfg[s]
            jnames = right_jname if s == "right" else left_jname
            qdata  = right_data[t] if s == "right" else left_data[t]
            qpos_dict = {jn: float(qdata[i]) for i, jn in enumerate(jnames)}
            wrist_pos = (joints_right if s == "right" else joints_left)[t, 0, :]
            R_mano = Rotation.from_rotvec(go[h_idx, t]).as_matrix()
            hands_data.append((s, wrist_pos, R_mano, qpos_dict, joints_tree, link_meshes))

        if not hands_data:
            continue

        rgb, depth, mask = _render_robot_rgbd(hands_data, camera, renderer)
        rgb_buf[t] = rgb
        mask_buf[t] = mask
        depth_buf[t, mask] = depth[mask]   # leave non-robot pixels at +inf

        if (t + 1) % 100 == 0:
            print(f"  {t+1}/{T_use}")

    renderer.delete()

    out_dir = args.processed_demo / "overlay_processor"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "robot_rgb.npy",   rgb_buf)
    np.save(out_dir / "robot_depth.npy", depth_buf.astype(np.float16))
    np.save(out_dir / "robot_mask.npy",  mask_buf)
    per_frame = mask_buf.sum(axis=(1, 2))
    print(f"[ok] wrote robot_rgb/depth/mask to {out_dir}  "
          f"(robot avg {per_frame.mean():.0f} px / max {per_frame.max()} px)")


if __name__ == "__main__":
    main()
