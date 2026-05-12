"""Compose RBY1 (no-gripper) + right XHand into a single MJCF scene.

Loads:
- third_party/mujoco_menagerie/rainbow_robotics_rby1/scene_rby1m_1.2_no_gripper.xml
- src/mujoco_sim/assets/xhand_right/xhand_right.xml

Attaches the XHand at the right wrist body (link_right_arm_6) via the
MuJoCo 3.x MjSpec API, and writes the composed scene to
src/mujoco_sim/scenes/rby1_xhand_right.xml.

Run:
    PYTHONPATH=$PWD/src python -m mujoco_sim.compose_rby1_xhand
"""

import os
from pathlib import Path

import mujoco

REPO = Path(__file__).resolve().parent.parent.parent
RBY1_SCENE = REPO / "third_party/mujoco_menagerie/rainbow_robotics_rby1/scene_rby1m_1.2_no_gripper.xml"
XHAND_R = REPO / "src/mujoco_sim/assets/xhand_right/xhand_right.xml"
OUT = REPO / "src/mujoco_sim/scenes/rby1_xhand_right.xml"


def main():
    spec = mujoco.MjSpec.from_file(str(RBY1_SCENE))
    xhand = mujoco.MjSpec.from_file(str(XHAND_R))

    # Increase offscreen framebuffer to allow HD720 rendering (1280x720).
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    wrist = spec.body("link_right_arm_6")
    frame = wrist.add_frame()
    # RBY1's original gripper mounts at z=-0.1261 in link_right_arm_6 frame.
    # Match that so XHand sits at the arm's end-effector mating surface
    # instead of being inset into the wrist link.
    frame.pos = [0.0, 0.0, -0.1261]
    spec.attach(xhand, prefix="rh_", frame=frame)

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

    # Third-person debug camera using targetbody so we just specify position;
    # orientation auto-tracks the robot. Pelvis would be ideal but use a
    # mid-body link.
    spec.worldbody.add_camera(
        name="front_view",
        pos=[2.0, 0.0, 2.0],
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
        targetbody="link_torso_2",
        fovy=55.0,
    )

    # Static table in front of the robot. Box half-extents (0.5, 1.0, 0.5)
    # = (1m depth, 2m width, 1m height). Top surface at z=1.0.
    table = spec.worldbody.add_body(name="table", pos=[0.9, 0.0, 0.5])
    table.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.5, 1.0, 0.5],
        rgba=[0.7, 0.5, 0.35, 1.0],
    )

    # spec.attach merges both specs under a single meshdir, so XHand meshes
    # (originally relative to xhand_right/) get resolved under RBY1's
    # assets/ and fail. Set each mesh's file to an absolute path so
    # spec.to_xml() validation passes, then text-substitute the repo root
    # prefix with a relative path from OUT so the saved XML stays portable.
    rby1_meshdir = (RBY1_SCENE.parent / "assets").resolve()
    xhand_meshdir = (XHAND_R.parent / "meshes").resolve()
    repo_root = REPO.resolve()
    for mesh in spec.meshes:
        src = xhand_meshdir if mesh.name.startswith("rh_") else rby1_meshdir
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
