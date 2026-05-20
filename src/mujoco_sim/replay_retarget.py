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

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import pinocchio as pin

from mujoco_sim.ik_arm import (
    SCENE,
    apply_frame,
    cam_to_world,
    head_cam_world,
    solve_arm_ik,
    wrist_to_arm6,
)

REPO = Path(__file__).resolve().parent.parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/cube_dataset/0412_val/episode_0/rgb_hawor/final_pose.pkl")
    ap.add_argument("--rgb-dir", default="data/cube_dataset/0412_val/episode_0/rgb_hawor/extracted_images",
                    help="Directory with NNNN.jpg originals; if missing the panel is dropped.")
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
    # Robot joints come first in MuJoCo's qpos/qvel; cube state follows.
    # Pinocchio sees only the robot, so its nq/nv give the slice into
    # MuJoCo's state that corresponds to the robot.
    ROBOT_NQ = pin_model.nq
    ROBOT_NV = pin_model.nv

    q_home = muj_model.qpos0.copy()
    hjid = mujoco.mj_name2id(muj_model, mujoco.mjtObj.mjOBJ_JOINT, "head_1")
    q_home[muj_model.jnt_qposadr[hjid]] = args.head_pitch
    # Bias arm joints to the middle of their range so the 7-DOF IK
    # redundancy is resolved into the natural elbow-down branch instead
    # of the elbow-inverted (out-of-range) branch that zero-warm-start
    # converges to.
    for side in ("right", "left"):
        for i in range(7):
            jid = mujoco.mj_name2id(muj_model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_arm_{i}")
            lo, hi = muj_model.jnt_range[jid]
            q_home[muj_model.jnt_qposadr[jid]] = 0.5 * (lo + hi)
    muj_data.qpos[:] = q_home
    mujoco.mj_forward(muj_model, muj_data)

    T_world_cam = head_cam_world(pin_model, pin_data, q_home[:ROBOT_NQ])
    renderer = mujoco.Renderer(muj_model, height=args.height, width=args.width)

    # Drive the robot through its position actuators: each frame compute
    # the IK target via pinocchio, push to ctrl, let mj_step's PD track.
    # Cube physics and any contact reactions emerge naturally from the
    # same mj_step. n_substeps makes sim time advance ~1/fps per frame.
    n_substeps = max(1, int(round(1.0 / (args.fps * muj_model.opt.timestep))))

    # Sync actuator targets to the seeded home pose so the first mj_step
    # doesn't snap the robot toward zero.
    for ai in range(muj_model.nu):
        jid = int(muj_model.actuator_trnid[ai, 0])
        if jid >= 0:
            muj_data.ctrl[ai] = muj_data.qpos[muj_model.jnt_qposadr[jid]]

    rgb_dir = REPO / args.rgb_dir
    use_rgb = rgb_dir.exists()
    if use_rgb:
        # Resize originals to match render height; preserve aspect.
        sample = imageio.imread(rgb_dir / "0000.jpg")
        rgb_target_w = int(round(sample.shape[1] * args.height / sample.shape[0]))
        print(f"using rgb originals from {rgb_dir} ({sample.shape[1]}x{sample.shape[0]} -> {rgb_target_w}x{args.height})")
    else:
        print(f"rgb dir {rgb_dir} missing; skipping originals panel")

    T = int(pose["T"])
    print(f"rendering {T} frames -> {out_path}")
    tmp = Path(tempfile.mkdtemp(prefix="replay_retarget_"))
    t0 = time.time()
    try:
        for t in range(T):
            apply_frame(pin_model, pin_data, muj_model, muj_data, pose, t, T_world_cam)
            robot_q = muj_data.qpos[:ROBOT_NQ].copy()
            for _ in range(n_substeps):
                mujoco.mj_step(muj_model, muj_data)
                muj_data.qpos[:ROBOT_NQ] = robot_q
                muj_data.qvel[:ROBOT_NV] = 0
            renderer.update_scene(muj_data, camera="head_cam")
            img_head = renderer.render()
            renderer.update_scene(muj_data, camera="front_view")
            img_front = renderer.render()
            panels = []
            if use_rgb:
                rgb = imageio.imread(rgb_dir / f"{t:04d}.jpg")
                rgb = cv2.resize(rgb, (rgb_target_w, args.height), interpolation=cv2.INTER_AREA)
                panels.append(rgb)
            panels.extend([img_head, img_front])
            imageio.imwrite(tmp / f"frame_{t:05d}.png", np.concatenate(panels, axis=1))
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
