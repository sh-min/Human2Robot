"""Render a full retargeted episode (final_pose.pkl) as an mp4.

Loops every frame: solve bimanual IK + copy finger qpos via apply_frame,
render head_cam and front_view side-by-side, write an mp4.

Usage:
    MUJOCO_GL=egl PYTHONPATH=$PWD/src python -m mujoco_sim.replay_retarget \
        --pkl data/cube_dataset/0412_val/episode_0/rgb_hawor/final_pose.pkl \
        --out output/episode_0_retarget.mp4
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import pinocchio as pin

from mujoco_sim.ik_arm import SCENE, apply_frame, head_cam_world

REPO = Path(__file__).resolve().parent.parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/cube_dataset/0412_val/episode_0/rgb_hawor/final_pose.pkl")
    ap.add_argument("--out", default="output/episode_0_retarget.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--head-pitch", type=float, default=0.6)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    pkl_path = REPO / args.pkl
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pkl_path.open("rb") as f:
        pose = pickle.load(f)

    pin_model = pin.buildModelFromMJCF(str(SCENE))
    pin_data = pin_model.createData()
    muj_model = mujoco.MjModel.from_xml_path(str(SCENE))
    muj_data = mujoco.MjData(muj_model)

    q_home = np.zeros(pin_model.nq)
    hjid = mujoco.mj_name2id(muj_model, mujoco.mjtObj.mjOBJ_JOINT, "head_1")
    q_home[muj_model.jnt_qposadr[hjid]] = args.head_pitch
    muj_data.qpos[:] = q_home
    mujoco.mj_forward(muj_model, muj_data)

    T_world_cam = head_cam_world(pin_model, pin_data, q_home)
    renderer = mujoco.Renderer(muj_model, height=args.height, width=args.width)

    T = int(pose["T"])
    print(f"rendering {T} frames -> {out_path}")
    tmp = Path(tempfile.mkdtemp(prefix="replay_retarget_"))
    t0 = time.time()
    try:
        for t in range(T):
            apply_frame(pin_model, pin_data, muj_model, muj_data, pose, t, T_world_cam)
            renderer.update_scene(muj_data, camera="head_cam")
            img_head = renderer.render()
            renderer.update_scene(muj_data, camera="front_view")
            img_front = renderer.render()
            imageio.imwrite(tmp / f"frame_{t:05d}.png", np.concatenate([img_head, img_front], axis=1))
            if (t + 1) % 100 == 0:
                dt = time.time() - t0
                print(f"  {t+1}/{T}  ({(t+1)/dt:.1f} fps)")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(args.fps),
            "-i", str(tmp / "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out_path),
        ], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
