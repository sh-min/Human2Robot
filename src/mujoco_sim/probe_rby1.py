"""Headless sanity-check render of the RBY1 scene from MuJoCo Menagerie.

Loads scene_rby1m_1.3.xml, holds the robot in qpos=0, and renders a short
mp4 with the camera rotating around the robot. Output: output/rby1_probe.mp4.

Run on a GPU (or CPU) node with:
    MUJOCO_GL=osmesa python -m mujoco_sim.probe_rby1
"""

from pathlib import Path

import imageio.v2 as imageio
import mujoco

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MJCF = REPO_ROOT / "third_party/mujoco_menagerie/rainbow_robotics_rby1/scene_rby1m_1.3.xml"
OUT = REPO_ROOT / "output/rby1_probe.mp4"

W, H = 480, 360
N_FRAMES = 90
FPS = 30


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 3.5
    cam.elevation = -15.0
    cam.lookat[:] = [0.0, 0.0, 0.9]

    renderer = mujoco.Renderer(model, height=H, width=W)
    frames = []
    for i in range(N_FRAMES):
        cam.azimuth = (i / N_FRAMES) * 360.0
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

    imageio.mimsave(str(OUT), frames, fps=FPS, codec="libx264")
    print(f"wrote {OUT} ({len(frames)} frames, {W}x{H}@{FPS}fps)")
    print(f"nq={model.nq} nu={model.nu} nbody={model.nbody}")


if __name__ == "__main__":
    main()
