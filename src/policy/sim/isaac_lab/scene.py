"""Reusable scene setup: ground, walls/ceiling, 3-point lighting, table, cube.

Used by ``render_demo`` and any future eval/replay script that shares the
robot + cube + table layout. Must be imported AFTER ``AppLauncher`` brings
Kit up (it touches ``isaaclab.sim``).
"""

from __future__ import annotations

import math
from typing import Tuple

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg


# Default DistantLight shines along -Z; rotate that to an arbitrary direction.
def _dir_to_quat(direction: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
    x, y, z = direction
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    # axis = (0,0,-1) x (x,y,z) = (y, -x, 0)
    ax, ay = y, -x
    an = math.sqrt(ax * ax + ay * ay)
    cos_t = max(-1.0, min(1.0, -z))
    if an < 1e-8:
        return (1.0, 0.0, 0.0, 0.0) if cos_t > 0.0 else (0.0, 1.0, 0.0, 0.0)
    angle = math.acos(cos_t)
    s = math.sin(angle * 0.5)
    return (math.cos(angle * 0.5), ax / an * s, ay / an * s, 0.0)


def build_scene(cube_pos: Tuple[float, float, float] = (0.55, 0.0, 1.10)) -> RigidObject:
    """Spawn ground, walls, lights, table, and cube. Returns the cube."""
    # --- Ground plane (warm gray) ---
    ground = sim_utils.GroundPlaneCfg(size=(40.0, 40.0), color=(0.45, 0.42, 0.40))
    ground.func("/World/defaultGroundPlane", ground)

    # --- Walls + ceiling (kinematic, neutral off-white) ---
    wall_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.70, 0.66), roughness=0.9)
    walls = (
        ("back", (0.1, 6.0, 3.5), (-1.5, 0.0, 1.75)),  # behind robot
        ("left", (5.0, 0.1, 3.5), (0.0, 2.5, 1.75)),
        ("right", (5.0, 0.1, 3.5), (0.0, -2.5, 1.75)),
        ("ceil", (5.0, 5.0, 0.1), (0.0, 0.0, 3.5)),
    )
    for name, size, pos in walls:
        cfg = sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=wall_mat,
        )
        cfg.func(f"/World/Walls/{name}", cfg, translation=pos)

    # --- Lighting: low ambient + 3-point (key warm, fill cool, rim cool/back) ---
    # CRITICAL: pass the configured cfg as the second arg to .func — passing a
    # default-constructed one (as the earlier inline code did) silently drops
    # the configured intensity/color.
    ambient = sim_utils.DomeLightCfg(intensity=800.0, color=(0.85, 0.88, 1.0))
    ambient.func("/World/Lights/ambient", ambient)

    # Direction vectors are where the light *shines* (sun goes that way), so
    # the source is on the opposite side. Z components are intentionally small
    # to keep most of the light coming in from the side, not straight down.
    key = sim_utils.DistantLightCfg(intensity=3500.0, color=(1.0, 0.96, 0.90), angle=2.0)
    key.func("/World/Lights/key", key, orientation=_dir_to_quat((-0.4, 0.8, -0.4)))

    fill = sim_utils.DistantLightCfg(intensity=1500.0, color=(0.85, 0.92, 1.0), angle=4.0)
    fill.func("/World/Lights/fill", fill, orientation=_dir_to_quat((0.3, -0.9, -0.3)))

    rim = sim_utils.DistantLightCfg(intensity=2500.0, color=(0.95, 0.95, 1.0), angle=1.0)
    rim.func("/World/Lights/rim", rim, orientation=_dir_to_quat((1.0, 0.0, -0.15)))

    # --- Table ---
    table = sim_utils.CuboidCfg(
        size=(1.0, 2.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.30), roughness=0.6),
    )
    table.func("/World/Table", table, translation=(0.9, 0.0, 0.5))

    # --- Cube (dynamic; returned to caller) ---
    cube_cfg = RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.034),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.18, 0.18), roughness=0.4),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=cube_pos),
    )
    return RigidObject(cube_cfg)
