"""Load the converted RBY1+XHand USD as an Isaac Lab Articulation and verify
basic state: joint count, body count, initial joint positions, and that a
few simulation steps run without exploding.

Run (from repo root):
    OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/src \\
        python -m isaac_lab.hello_robot --headless
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext

REPO = Path(__file__).resolve().parents[3]
USD_PATH = REPO / "src/isaac_lab/assets/rby1_xhand/rby1_xhand.usd"


def main():
    if not USD_PATH.exists():
        raise FileNotFoundError(
            f"{USD_PATH} not found. Run `python -m isaac_lab.convert_robot_usd` first."
        )

    sim_cfg = SimulationCfg(dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.5, 2.5, 1.5), target=(0.0, 0.0, 1.0))

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg())

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(USD_PATH)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=1000.0,
                damping=50.0,
            )
        },
    )
    robot = Articulation(robot_cfg)

    sim.reset()
    print(f"[hello_robot] num_joints = {robot.num_joints}")
    print(f"[hello_robot] num_bodies = {robot.num_bodies}")
    print(f"[hello_robot] joint names: {robot.joint_names}")

    n_steps = 60  # 0.5 s
    for i in range(n_steps):
        sim.step()
        robot.update(sim_cfg.dt)
        if i % 12 == 0:
            joint_pos = robot.data.joint_pos[0]  # [num_joints]
            print(f"  step {i:>3}  t={i * sim_cfg.dt:.2f}s  "
                  f"joint_pos[:3]={[f'{q:+.3f}' for q in joint_pos[:3].cpu().tolist()]}")

    print("[hello_robot] done")


if __name__ == "__main__":
    import os

    main()
    os._exit(0)
