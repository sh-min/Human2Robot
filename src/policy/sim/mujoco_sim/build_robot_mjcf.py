"""Compose RBY1 + bimanual XHand as a robot-only MJCF (no table, cube, or
debug cameras) and write to ``src/policy/sim/isaac_lab/assets/rby1_xhand.xml``.

Mirrors ``compose_rby1_xhand.py`` but strips world objects so the output is
a clean robot definition ready to feed Isaac Lab's MjcfConverter (which
runs in the separate isaac_lab conda env).

Run (from repo root, in the mujoco_sim env):
    PYTHONPATH=$PWD/src python -m mujoco_sim.build_robot_mjcf
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco

from .compose_rby1_xhand import attach_hand

REPO = Path(__file__).resolve().parents[4]
RBY1_SCENE = REPO / "third_party/mujoco_menagerie/rainbow_robotics_rby1/scene_rby1m_1.2_no_gripper.xml"
XHAND_R = REPO / "src/mujoco_sim/assets/xhand_right/xhand_right.xml"
XHAND_L = REPO / "src/mujoco_sim/assets/xhand_left/xhand_left.xml"
OUT = REPO / "src/policy/sim/isaac_lab/assets/rby1_xhand.xml"


def main():
    spec = mujoco.MjSpec.from_file(str(RBY1_SCENE))

    attach_hand(spec, "link_right_arm_6", XHAND_R, prefix="rh_")
    attach_hand(spec, "link_left_arm_6",  XHAND_L, prefix="lh_")

    # Lock everything below the shoulder; same set as the mujoco_sim scene.
    LOWER_BODY_JOINTS = (
        "world_j",
        "wheel_fr", "wheel_fl", "wheel_rr", "wheel_rl",
        "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
        "head_0",
    )
    for jname in LOWER_BODY_JOINTS:
        joint = next((j for j in spec.joints if j.name == jname), None)
        if joint is not None:
            spec.delete(joint)
    LOWER_BODY_ACTUATORS = (
        "front_right_wheel_act", "front_left_wheel_act",
        "rear_right_wheel_act",  "rear_left_wheel_act",
        "link1_act", "link2_act", "link3_act", "link4_act", "link5_act", "link6_act",
        "head_0_act",
    )
    for aname in LOWER_BODY_ACTUATORS:
        actuator = next((a for a in spec.actuators if a.name == aname), None)
        if actuator is not None:
            spec.delete(actuator)

    # Restore a sane torque budget on head_1 (stock values are too small).
    j = next((j for j in spec.joints if j.name == "head_1"), None)
    if j is not None:
        j.actfrcrange = [-500.0, 500.0]
    a = next((a for a in spec.actuators if a.name == "head_1_act"), None)
    if a is not None:
        a.forcerange = [-500.0, 500.0]

    # Absolutize mesh paths (spec.attach collapses meshdirs), then rewrite to
    # be repo-relative so the saved XML stays portable.
    rby1_meshdir = (RBY1_SCENE.parent / "assets").resolve()
    xhand_r_meshdir = (XHAND_R.parent / "meshes").resolve()
    xhand_l_meshdir = (XHAND_L.parent / "meshes").resolve()
    repo_root = REPO.resolve()
    for mesh in spec.meshes:
        if mesh.name.startswith("rh_"):
            src = xhand_r_meshdir
        elif mesh.name.startswith("lh_"):
            src = xhand_l_meshdir
        else:
            src = rby1_meshdir
        mesh.file = str(src / Path(mesh.file).name)
    spec.meshdir = ""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    xml = spec.to_xml()
    out_to_repo = os.path.relpath(repo_root, OUT.parent.resolve())
    xml = xml.replace(str(repo_root), out_to_repo)
    OUT.write_text(xml)

    m = mujoco.MjModel.from_xml_path(str(OUT))
    print(f"wrote {OUT}")
    print(f"robot-only model: nq={m.nq} nu={m.nu} nbody={m.nbody}")


if __name__ == "__main__":
    main()
