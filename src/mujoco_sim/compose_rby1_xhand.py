"""Compose RBY1 (no-gripper) + both XHands into a single MJCF scene.

Loads:
- third_party/mujoco_menagerie/rainbow_robotics_rby1/scene_rby1m_1.2_no_gripper.xml
- src/mujoco_sim/assets/xhand_right/xhand_right.xml
- src/mujoco_sim/assets/xhand_left/xhand_left.xml

Attaches each XHand at the corresponding wrist (link_right_arm_6 /
link_left_arm_6) via the MuJoCo 3.x MjSpec API at z=-0.1261 (RBY1's
original EE mating surface). Writes the composed scene to
src/mujoco_sim/scenes/rby1_xhand.xml.

Run:
    PYTHONPATH=$PWD/src python -m mujoco_sim.compose_rby1_xhand
"""

import os
from pathlib import Path

import mujoco

REPO = Path(__file__).resolve().parent.parent.parent
RBY1_SCENE = REPO / "third_party/mujoco_menagerie/rainbow_robotics_rby1/scene_rby1m_1.2_no_gripper.xml"
XHAND_R = REPO / "src/mujoco_sim/assets/xhand_right/xhand_right.xml"
XHAND_L = REPO / "src/mujoco_sim/assets/xhand_left/xhand_left.xml"
OUT = REPO / "src/mujoco_sim/scenes/rby1_xhand.xml"

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

    # Third-person debug camera, parented to the (static) table so it never
    # moves when the robot's joints change during IK / sim. Camera at
    # world (2.0, 0, 2.0) -> table-local (1.1, 0, 1.5). Orientation baked
    # via xyaxes (no targetbody) so the view never drifts: look-at is
    # link_torso_2's default world pos (0, 0, 0.6305).
    table.add_camera(
        name="front_view",
        pos=[1.1, 0.0, 1.5],
        xyaxes=[0.0, 1.0, 0.0, -0.566, 0.0, 0.826],
        fovy=60.0,
    )

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
