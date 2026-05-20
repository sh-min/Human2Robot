"""Compose RBY1 (no-gripper) + both XHands into a single MJCF scene.

Loads:
- third_party/mujoco_menagerie/rainbow_robotics_rby1/scene_rby1m_1.2_no_gripper.xml
- src/sim/mujoco_sim/assets/xhand_right/xhand_right.xml
- src/sim/mujoco_sim/assets/xhand_left/xhand_left.xml

Attaches each XHand at the corresponding wrist (link_right_arm_6 /
link_left_arm_6) via the MuJoCo 3.x MjSpec API at z=-0.1261 (RBY1's
original EE mating surface). Writes the composed scene to
src/sim/mujoco_sim/scenes/rby1_xhand.xml.

Run:
    PYTHONPATH=$PWD/src python -m mujoco_sim.compose_rby1_xhand
"""

import os
from pathlib import Path

import mujoco

REPO = Path(__file__).resolve().parents[3]
RBY1_SCENE = REPO / "third_party/mujoco_menagerie/rainbow_robotics_rby1/scene_rby1m_1.2_no_gripper.xml"
XHAND_R = REPO / "src/sim/mujoco_sim/assets/xhand_right/xhand_right.xml"
XHAND_L = REPO / "src/sim/mujoco_sim/assets/xhand_left/xhand_left.xml"
CUBE = REPO / "src/sim/mujoco_sim/assets/cube/cube.xml"
OUT = REPO / "src/sim/mujoco_sim/scenes/rby1_xhand.xml"

# RBY1's original gripper mounts at z=-0.1261 in link_*_arm_6 frame.
EE_OFFSET = [0.0, 0.0, -0.1261]


def attach_hand(spec, wrist_body_name, hand_xml_path, prefix):
    """Attach an XHand spec under the given wrist body at the EE offset."""
    hand = mujoco.MjSpec.from_file(str(hand_xml_path))
    wrist = spec.body(wrist_body_name)
    frame = wrist.add_frame()
    frame.pos = EE_OFFSET
    spec.attach(hand, prefix=prefix, frame=frame)


def main():
    spec = mujoco.MjSpec.from_file(str(RBY1_SCENE))

    # Increase offscreen framebuffer to allow HD720 rendering (1280x720).
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    attach_hand(spec, "link_right_arm_6", XHAND_R, prefix="rh_")
    attach_hand(spec, "link_left_arm_6",  XHAND_L, prefix="lh_")

    # Lock everything below the shoulder: base freejoint, 4 wheels, and
    # the 6 torso joints. Only the two arms (incl. wrists), both hands,
    # and the 2 head joints remain articulated.
    LOWER_BODY_JOINTS = (
        "world_j",
        "wheel_fr", "wheel_fl", "wheel_rr", "wheel_rl",
        "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
        # head_0 (yaw) is removed because the RBY1 actuator can't fully hold
        # it against numerical drift — the head_cam slowly rotates even
        # at rest. Only head_1 (pitch) is needed for tabletop manipulation.
        "head_0",
    )
    for jname in LOWER_BODY_JOINTS:
        joint = next((j for j in spec.joints if j.name == jname), None)
        if joint is not None:
            spec.delete(joint)
    # Drop the actuators that drove those joints.
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

    # Original RBY1 limits on head_1 are too small (actuator forcerange
    # ~7.3 N*m and joint actuatorfrcrange=0,0), so the commanded pitch
    # silently drifts under gravity / numerical error. Restore a sane
    # torque budget on both the joint and the actuator.
    j = next((j for j in spec.joints if j.name == "head_1"), None)
    if j is not None:
        j.actfrcrange = [-500.0, 500.0]
    a = next((a for a in spec.actuators if a.name == "head_1_act"), None)
    if a is not None:
        a.forcerange = [-500.0, 500.0]

    # Head-mounted egocentric camera on link_head_2.
    # MuJoCo camera convention: looks along -z, +y is up, +x is right.
    # In head frame, +x is forward, +z is up, so we orient the camera with
    # xaxis=-head_y (camera-right) and yaxis=+head_z (camera-up).
    head = spec.body("link_head_2")
    # fovy=60 matches the ZED Mini's vertical FOV (HFOV ~90, VFOV ~60) when
    # rendered at 16:9 (e.g. 1280x720 HD720 mode).
    head.add_camera(
        name="head_cam",
        pos=[0.12, 0.0, 0.05],
        xyaxes=[0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        fovy=60.0,
    )

    # Static table in front of the robot. Box half-extents (0.5, 1.0, 0.5)
    # = (1m depth, 2m width, 1m height). Top surface at z=1.0.
    table = spec.worldbody.add_body(name="table", pos=[0.9, 0.0, 0.5])
    table.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.5, 1.0, 0.5],
        rgba=[0.7, 0.5, 0.35, 1.0],
    )

    # Third-person debug camera, parented to the (static) table so it
    # never moves when the robot's joints change during IK / sim.
    # World pos (1.0, 0, 1.65) -> table-local (0.1, 0, 1.15), 0.6 m
    # along the look ray (-0.832, 0, -0.555) from the original
    # (1.5, 0, 2.0); aimed near robot torso z~1.0.
    table.add_camera(
        name="front_view",
        pos=[0.10, 0.0, 1.15],
        xyaxes=[0.0, 1.0, 0.0, -0.555, 0.0, 0.832],
        fovy=65.0,
    )

    # Side-view debug cameras with fixed xyaxes. Look mainly along ±y but
    # tilted down 15° to favor the table-top scene. World positions:
    #   side_left  -> (0.5, +1.6, 1.25)  (robot's left,  viewer-right)
    #   side_right -> (0.5, -1.6, 1.25)  (robot's right, viewer-left)
    # Table at world (0.9, 0, 0.5) so table-local = world - table_pos.
    table.add_camera(
        name="side_left",
        pos=[-0.4, 1.6, 0.75],
        xyaxes=[-1.0, 0.0, 0.0, 0.0, -0.259, 0.966],
        fovy=70.0,
    )
    table.add_camera(
        name="side_right",
        pos=[-0.4, -1.6, 0.75],
        xyaxes=[1.0, 0.0, 0.0, 0.0, 0.259, 0.966],
        fovy=70.0,
    )

    # 2x2 cube as a free body, initially floating ~7.5 cm above the
    # table top so we can see whether physics actually integrates.
    cube = mujoco.MjSpec.from_file(str(CUBE))
    cube_frame = spec.worldbody.add_frame()
    cube_frame.pos = [0.55, 0.0, 1.10]
    spec.attach(cube, prefix="cube_", frame=cube_frame)

    # spec.attach merges specs under a single meshdir, so XHand meshes
    # (originally relative to xhand_{right,left}/) get resolved under RBY1's
    # assets/ and fail. Set each mesh's file to an absolute path so
    # spec.to_xml() validation passes, then text-substitute the repo root
    # prefix with a relative path from OUT so the saved XML stays portable.
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
    print(f"composed model: nq={m.nq} nu={m.nu} nbody={m.nbody}")


if __name__ == "__main__":
    main()
