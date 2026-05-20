# mujoco_sim

MuJoCo simulation environment for evaluating LeRobot-trained policies on a
Rainbow Robotics body + bimanual XHand setup. Consumes LeRobot dataset format;
outputs rollout videos and metrics.

> Folder is named `mujoco_sim` (not `mujoco`) to avoid shadowing the pip
> `mujoco` package on `PYTHONPATH`.

## Setup

```bash
conda env create -f src/sim/mujoco_sim/environment.yml   # run from repo root
conda activate mujoco_sim
```

Installs (Python 3.12):
- `lerobot[training,diffusion]` (editable, from `third_party/lerobot`)
- `mujoco`, `imageio`, `gradio`

## Layout

```
src/sim/mujoco_sim/
├── assets/
│   ├── xhand_right/{xhand_right.xml, meshes/}    # right XHand MJCF + mesh symlink
│   └── xhand_left/{xhand_left.xml, meshes/}      # left XHand MJCF + mesh symlink
├── scenes/
│   └── rby1_xhand.xml                            # composed scene (generated)
├── compose_rby1_xhand.py     # build composed scene from RBY1 + both XHands
├── ik.py                     # damped least-squares wrist IK (single-arm)
├── web.py                    # gradio MVP for interactive pose tuning
├── probe_rby1.py             # legacy sanity-check render of bare RBY1
├── environment.yml
└── README.md
```

## Composing the scene

`scenes/rby1_xhand.xml` is generated from the Menagerie RBY1 (no-gripper)
plus both XHand MJCFs. Rebuild after editing the compose script or hand
XMLs:

```bash
PYTHONPATH=$PWD/src python -m mujoco_sim.compose_rby1_xhand
```

What it produces:
- RBY1 base (mobile + 6-DOF torso + 7-DOF arms × 2 + head)
- Right + left XHand attached at each wrist's EE mating surface
  (`link_{side}_arm_6` frame, z=-0.1261 — matches RBY1's original gripper)
- 12 position actuators per hand (kp=50, force range = joint limit)
- Cameras: `head_cam` (ZED-Mini-matched, fovy=60° on head_2), `front_view`
  (table-mounted, baked xyaxes — stays put under IK / sim)
- Static table at world (0.9, 0, 0.5), 1 m deep × 2 m wide × 1 m tall
- HD720 offscreen framebuffer for 1280×720 rendering

Model totals: nq=57, nu=50, nbody=54.

## Inverse kinematics

`solve_wrist_ik(model, qpos, "link_right_arm_6", pos, quat)` runs a damped
least-squares Jacobian IK that updates **only the 7 arm joints on the named
side**; every other DOF (base freejoint, torso, head, other arm, hand
fingers) is mechanically untouched by construction.

```python
from sim.mujoco_sim.ik import solve_wrist_ik
q = solve_wrist_ik(model, qpos, "link_right_arm_6", [0.55, -0.08, 1.05],
                   [0.7071, 0.0, -0.7071, 0.0])  # palm-inward
```

Caveat: collisions are not enforced during IK (kinematics-only). Use sim
dynamics or future collision-aware variants when interference matters.

## Interactive poser (gradio)

```bash
PYTHONPATH=$PWD/src MUJOCO_GL=osmesa python -m mujoco_sim.web
```

Listens on `0.0.0.0:7860`. From your laptop:

```bash
ssh -L 7860:localhost:7860 <worker-node>
# open http://localhost:7860
```

Sliders: per-arm wrist target (x, y, z), head pitch, hand close.
Dropdown: palm orientation (inward / down / up). Radio: camera.

## Rendering notes

- **`MUJOCO_GL=osmesa`** — headless software rendering. No display or GPU
  driver needed. Slower than EGL but reliable.
- **`MUJOCO_GL=egl`** — hardware-accelerated headless on a GPU node.
  Currently blocked by a PyOpenGL ↔ `mujoco.egl` incompatibility
  (`module 'OpenGL.EGL' has no attribute 'EGLDeviceEXT'`); fix if throughput
  becomes a bottleneck.
- Interactive `mujoco.viewer` (X11 GUI) is not used here.
