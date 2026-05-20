# isaac_lab

Isaac Sim / Isaac Lab pipeline for replaying retargeted bimanual hand
data on the RBY1 + XHand robot, and for training / evaluating IL policies on
top of the same scene.

## Setup

```bash
conda create -y -n isaac_lab python=3.11 pip
conda activate isaac_lab

# Isaac Sim 5.1 (pulls torch 2.7 / cuda 12.6 / numpy<2 as deps).
# torchaudio matches torch's version but isaacsim-core doesn't declare it.
pip install --extra-index-url https://pypi.nvidia.com \
    "isaacsim[all,extscache]==5.1.0" "torchaudio==2.7.0"

# Isaac Lab editable install.
git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
cd third_party/IsaacLab
OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh --install none   # editable install, no RL extras
cd ../..

# Pinocchio for arm IK -- AFTER isaacsim so its numpy<2 constraint is already
# in place; otherwise pin's cmeel-boost dep pulls numpy>=2.3 and breaks
# isaacsim's camera pipeline.
pip install pin
```

The first Python that imports `isaacsim` triggers the Omniverse Kit EULA. Set
`OMNI_KIT_ACCEPT_EULA=YES` to accept non-interactively (matches CI behavior).

## Layout

(planned — under construction)

```
src/sim/isaac_lab/
├── assets/                  # USD: RBY1 + XHand + table + cube (TBD)
├── envs/                    # Isaac Lab Articulation / ManagerBased envs
├── replay_retarget.py       # mirror of sim/mujoco_sim/replay_retarget.py
├── environment.yml
└── README.md
```

## Smoke test

```bash
OMNI_KIT_ACCEPT_EULA=YES python -c "from isaaclab.app import AppLauncher; print('IsaacLab OK')"
```

Should print `IsaacLab OK` after a few seconds of Kit boot.
