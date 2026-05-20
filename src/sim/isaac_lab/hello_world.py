"""Isaac Lab hello world: spawn a ground plane + a falling cube, step the
sim for ~1 s of sim time, and print the cube's z position over time.

Run (from repo root):
    OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/src python -m isaac_lab.hello_world --headless

The first invocation pulls a few hundred MB of Isaac Sim extension caches;
later runs start in ~5 s.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac modules can only be imported AFTER AppLauncher starts Kit.
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationCfg, SimulationContext


def main():
    sim_cfg = SimulationCfg(dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.5, 2.5, 1.5), target=(0.0, 0.0, 0.0))

    # Ground plane.
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())

    # Light so the scene isn't black if we ever render it.
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg())

    # Cube as a RigidObject so we can read its physics-driven world pose.
    cube_cfg = RigidObjectCfg(
        prim_path="/World/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    cube = RigidObject(cube_cfg)

    sim.reset()
    print("[hello_world] sim started")

    # Step for ~1 s of sim time, log the cube z every 0.1 s.
    n_steps = int(1.0 / sim_cfg.dt)
    for i in range(n_steps):
        sim.step()
        cube.update(sim_cfg.dt)
        if i % (n_steps // 10) == 0:
            z = float(cube.data.root_pos_w[0, 2])
            print(f"  step {i:>4}  t={i * sim_cfg.dt:.2f}s  cube_z={z:+.3f} m")

    print("[hello_world] sim done")


if __name__ == "__main__":
    import os

    main()
    os._exit(0)
