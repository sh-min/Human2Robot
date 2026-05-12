# mujoco_sim

MuJoCo simulation environment for evaluating LeRobot-trained policies on a
Rainbow Robotics body + XHand setup. Consumes LeRobot dataset format; outputs
rollout videos and metrics.

> Folder is named `mujoco_sim` (not `mujoco`) to avoid shadowing the pip
> `mujoco` package on `PYTHONPATH`.

## Setup

```bash
conda env create -f src/mujoco_sim/environment.yml   # run from repo root
conda activate mujoco_sim
```

Installs (Python 3.12):
- `lerobot[training,diffusion]` (editable, from `third_party/lerobot`)
- `mujoco`, `imageio`

## Layout

```
src/mujoco_sim/
├── probe_rby1.py        # sanity-check render of RBY1 (Menagerie scene)
├── environment.yml     # conda env spec
└── README.md
```

## Sanity check

Renders 90 frames of the RBY1 scene with a rotating camera to
`output/rby1_probe.mp4`:

```bash
PYTHONPATH=$PWD/src MUJOCO_GL=osmesa python -m mujoco_sim.probe_rby1
```

## Rendering notes

- **`MUJOCO_GL=osmesa`** — headless software rendering. Works on any node,
  no display or GPU driver needed. Slower than EGL but fine for eval videos.
- **`MUJOCO_GL=egl`** — hardware-accelerated headless rendering on a GPU
  node. Currently blocked by a PyOpenGL <-> `mujoco.egl` incompatibility
  (`module 'OpenGL.EGL' has no attribute 'EGLDeviceEXT'`); fix if eval
  throughput becomes a bottleneck.
- Interactive `mujoco.viewer` requires an X display (`ssh -X` or VNC) — not
  used here.
