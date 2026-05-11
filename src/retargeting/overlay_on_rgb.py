"""Overlay both retargeted xhands onto the original RGB at HaWoR's estimated
wrist pose (cam frame).

Usage:
    conda activate RFM_retarget
    python overlay_on_rgb.py \
        --npz       <...>/cam0_hawor/retarget_input.npz \
        --rgb_dir   <...>/cam0 \
        --right_pkl <...>/cam0_hawor/qpos_xhand_right.pkl \
        --left_pkl  <...>/cam0_hawor/qpos_xhand_left.pkl \
        --out       <...>/cam0_hawor/overlay.mp4
"""
import argparse
import os
import pickle
import subprocess
import tempfile
from glob import glob

import cv2
import numpy as np
import sapien
import tqdm
from natsort import natsorted
from scipy.spatial.transform import Rotation as Rscipy

from dex_retargeting.retargeting_config import RetargetingConfig

from _paths import URDF_ROOT, R_MANO_XHAND as _R_DICT

R_MANO_XHAND = {h: R.astype(np.float64) for h, R in _R_DICT.items()}

# Sapien camera convention vs OpenCV camera (HaWoR cam):
#   sapien cam local: +x forward, +y left, +z up
#   OpenCV cam world: +z forward, +y down, +x right
# Rotation that places sapien cam at world origin behaving like the OpenCV cam.
R_SAPIEN_CAM = np.array([
    [0, -1,  0],
    [0,  0, -1],
    [1,  0,  0],
], dtype=np.float64)


def matrix_to_pose(R, t):
    quat_xyzw = Rscipy.from_matrix(R).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return sapien.Pose(t.astype(np.float64), quat_wxyz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--rgb_dir", required=True)
    ap.add_argument("--rgb_glob", default="rgb_frame*.png")
    ap.add_argument("--right_pkl", required=True)
    ap.add_argument("--left_pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img_focal", type=float, default=600.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="Robot overlay blending alpha (1.0=opaque)")
    args = ap.parse_args()

    RetargetingConfig.set_default_urdf_dir(URDF_ROOT)

    data = np.load(args.npz)
    joints_left = data["joints_left"]
    joints_right = data["joints_right"]
    mano_root = data["mano_global_orient"]
    valid = data["valid"]

    right_d = pickle.load(open(args.right_pkl, "rb"))
    left_d = pickle.load(open(args.left_pkl, "rb"))
    right_qpos = np.asarray(right_d["data"])
    left_qpos = np.asarray(left_d["data"])

    rgb_files = natsorted(glob(os.path.join(args.rgb_dir, args.rgb_glob)))
    T = min(len(rgb_files), right_qpos.shape[0], left_qpos.shape[0])
    print(f"Using T={T} (rgb={len(rgb_files)}, qpos_r={right_qpos.shape[0]}, qpos_l={left_qpos.shape[0]})")

    sample = cv2.imread(rgb_files[0])
    H, W = sample.shape[:2]
    fovy = 2 * np.arctan2(H / 2.0, args.img_focal)
    print(f"image {W}x{H}  fx={args.img_focal}  fovy={fovy:.4f} rad ({np.degrees(fovy):.2f}°)")

    sapien.render.set_camera_shader_dir("default")
    scene = sapien.Scene()
    scene.add_directional_light(np.array([0.3, 0.3, 1]), np.array([1.5, 1.5, 1.5]))
    scene.add_point_light(np.array([0.3, -0.3, 0.5]), np.array([0.5, 0.5, 0.5]), shadow=False)
    scene.add_point_light(np.array([-0.3, 0.3, 0.5]), np.array([0.5, 0.5, 0.5]), shadow=False)

    cam = scene.add_camera(name="cam", width=W, height=H,
                           fovy=fovy, near=0.05, far=10.0)
    cam.set_local_pose(matrix_to_pose(R_SAPIEN_CAM, np.zeros(3)))

    loader = scene.create_urdf_loader()
    loader.load_multiple_collisions_from_file = True
    robot_r = loader.load(os.path.join(URDF_ROOT, "xhand/xhand_right.urdf"))
    robot_l = loader.load(os.path.join(URDF_ROOT, "xhand/xhand_left.urdf"))

    sj_r = [j.get_name() for j in robot_r.get_active_joints()]
    sj_l = [j.get_name() for j in robot_l.get_active_joints()]
    map_r = np.array([right_d["joint_names"].index(n) for n in sj_r], dtype=int)
    map_l = np.array([left_d["joint_names"].index(n) for n in sj_l], dtype=int)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (W, H))
    far_away = sapien.Pose([0, 0, 100])

    for t in tqdm.tqdm(range(T)):
        robot_r.set_qpos(right_qpos[t][map_r])
        robot_l.set_qpos(left_qpos[t][map_l])
        for robot, h_idx, joints_arr, hand_name in [
            (robot_l, 0, joints_left, "left"),
            (robot_r, 1, joints_right, "right"),
        ]:
            if not valid[h_idx, t]:
                robot.set_pose(far_away)
                continue
            R_cam_mano = Rscipy.from_rotvec(mano_root[h_idx, t]).as_matrix()
            t_cam = joints_arr[t, 0]
            R_cam_xhand = R_cam_mano @ R_MANO_XHAND[hand_name]
            robot.set_pose(matrix_to_pose(R_cam_xhand, t_cam))

        scene.update_render()
        cam.take_picture()
        rendered = cam.get_picture("Color")[..., :3]
        rendered = np.clip(rendered, 0, 1).astype(np.float32)

        pos = cam.get_picture("Position")
        if pos.shape[-1] >= 3:
            mask = np.linalg.norm(pos[..., :3], axis=-1) > 1e-3
        else:
            mask = rendered.sum(-1) > 0.02

        raw_bgr = cv2.imread(rgb_files[t])
        if raw_bgr.shape[:2] != (H, W):
            raw_bgr = cv2.resize(raw_bgr, (W, H))
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        a = float(args.alpha)
        m = mask[..., None].astype(np.float32)
        out_rgb = raw_rgb * (1 - m * a) + rendered * (m * a)

        out_bgr = cv2.cvtColor((out_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(out_bgr)

    writer.release()
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_path,
         "-vcodec", "libx264", "-pix_fmt", "yuv420p",
         args.out],
        check=True,
    )
    os.remove(tmp_path)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
