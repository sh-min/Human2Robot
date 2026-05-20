"""Convert the robot-only MJCF (built by ``sim.mujoco_sim.build_robot_mjcf``) to
a USD asset that Isaac Lab can load as an Articulation.

Outputs ``src/isaac_lab/assets/rby1_xhand/rby1_xhand.usd`` plus any materials/
meshes the converter generates.

Run (from repo root, in the isaac_lab env):
    OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/src python -m isaac_lab.convert_robot_usd
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

# Isaac Lab modules require Kit to be up.
from isaacsim.core.utils.extensions import enable_extension

# IsaacSim 5.x doesn't auto-load the MJCF importer in headless apps; without
# this enable_extension call the converter's MJCFCreateImportConfig command
# is missing and import_config comes back None.
enable_extension("isaacsim.asset.importer.mjcf")

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
MJCF_PATH = REPO / "src/isaac_lab/assets/rby1_xhand.xml"
USD_DIR = REPO / "src/isaac_lab/assets/rby1_xhand"


def main():
    if not MJCF_PATH.exists():
        raise FileNotFoundError(
            f"{MJCF_PATH} not found. Run `python -m sim.mujoco_sim.build_robot_mjcf` "
            "first (in the mujoco_sim env)."
        )

    USD_DIR.mkdir(parents=True, exist_ok=True)
    cfg = MjcfConverterCfg(
        asset_path=str(MJCF_PATH),
        usd_dir=str(USD_DIR),
        usd_file_name="rby1_xhand.usd",
        force_usd_conversion=True,
        fix_base=True,            # base is locked to world in the MJCF too
        import_inertia_tensor=True,
        self_collision=False,
        make_instanceable=False,
    )
    converter = MjcfConverter(cfg)
    print(f"wrote {converter.usd_path}")

    # MJCF importer tags both the implicit worldBody Xform and the real robot
    # base body with ArticulationRootAPI; Isaac Lab's Articulation refuses to
    # load when more than one root exists under prim_path. Keep the deepest
    # path (the real robot base body) and strip the API from the worldBody
    # placeholder. NOTE: even with set_fix_base(True) creating a FixedJoint
    # from world->base, PhysX won't flag the articulation as fixed_base
    # (Articulation.is_fixed_base stays False). The FixedJoint still holds
    # the base in place kinematically; callers that care should hardcode the
    # fixed-base assumption rather than read is_fixed_base.
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(converter.usd_path)
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print(f"articulation roots before fixup: {[str(p.GetPath()) for p in roots]}")
    if len(roots) > 1:
        keep = max(roots, key=lambda p: len(str(p.GetPath()).split("/")))
        for p in roots:
            if p == keep:
                continue
            p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        stage.GetRootLayer().Save()
        print(f"kept articulation root: {keep.GetPath()}")
    else:
        print("no fixup needed")


if __name__ == "__main__":
    import os

    main()
    os._exit(0)
