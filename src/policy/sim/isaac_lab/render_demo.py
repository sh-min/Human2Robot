"""Render demo: load the RBY1+XHand USD, attach a 3rd-person camera, step
the sim for ~1 s, and save the camera RGB frames as PNGs + an mp4.

Run (from repo root, in the isaac_lab env):
    OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/src \\
        python -m isaac_lab.render_demo --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--n-frames", type=int, default=60)
parser.add_argument("--out", type=str, default="output/isaac_render_demo.mp4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import imageio.v2 as imageio
import numpy as np

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from .scene import build_scene

REPO = Path(__file__).resolve().parents[4]
USD_PATH = REPO / "src/isaac_lab/assets/rby1_xhand/rby1_xhand.usd"


def main():
    sim_cfg = SimulationCfg(dt=1.0 / 60.0)
    sim = SimulationContext(sim_cfg)

    cube = build_scene()

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(USD_PATH)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=1000.0, damping=50.0)
        },
    )
    robot = Articulation(robot_cfg)

    # Third-person camera matching the mujoco front_view ray
    # (eye=(1.0, 0, 1.65), aim=(0, 0, 1.0)).
    cam_cfg = CameraCfg(
        prim_path="/World/FrontCam",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 50.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )
    camera = Camera(cfg=cam_cfg)

    sim.reset()
    # Aim camera at the robot's torso post-init (set_world_poses_from_view
    # handles the look-at quaternion for us).
    cam_device = camera.device
    camera.set_world_poses_from_view(
        eyes=torch.tensor([[2.2, 0.0, 1.7]], device=cam_device),
        targets=torch.tensor([[0.0, 0.0, 1.1]], device=cam_device),
    )
    print(f"[render_demo] sim ready  n_joints={robot.num_joints}  cam={camera.image_shape}")

    tmp = Path(tempfile.mkdtemp(prefix="isaac_render_"))
    print(f"  frames -> {tmp}")
    try:
        for i in range(args.n_frames):
            sim.step()
            robot.update(sim_cfg.dt)
            camera.update(sim_cfg.dt)
            rgb = camera.data.output["rgb"][0].cpu().numpy()  # (H, W, 4) RGBA uint8
            imageio.imwrite(tmp / f"frame_{i:05d}.png", rgb[..., :3])
            if i % 10 == 0:
                print(f"  step {i:>3}  t={i * sim_cfg.dt:.2f}s")

        out_path = REPO / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(int(1.0 / sim_cfg.dt)),
                "-i", str(tmp / "frame_%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(out_path),
            ],
            check=True,
        )
        print(f"wrote {out_path}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import os

    main()
    # Kit's simulation_app.close() reliably hangs on plugin shutdown
    # (replicator/physx/syntheticdata); skip it and let the OS reap.
    os._exit(0)
