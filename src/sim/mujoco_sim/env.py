"""Gymnasium env wrapper for the RBY1 + bimanual XHand scene.

Standalone (not registered with gym or lerobot). Instantiate directly:

    from mujoco_sim.env import RBY1XHandEnv, EnvConfig
    env = RBY1XHandEnv(EnvConfig(image_size=224, max_episode_steps=200))
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

Action space: 38-DOF absolute target qpos
    [right_arm_0..6, left_arm_0..6,                  # 14 arm
     right_hand_12_fingers, left_hand_12_fingers]    # 24 hand

Observation space: dict matching LeRobot conventions
    "observation.images.head_cam": (H, W, 3) uint8
    "observation.state":           (38,) float32  -- current qpos of action joints

Reward / termination are placeholders for IL eval; truncated by step count.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

REPO = Path(__file__).resolve().parents[3]
SCENE = REPO / "src/sim/mujoco_sim/scenes/rby1_xhand.xml"

# Per-hand finger joint suffixes in the order used by both XHands.
_HAND_FINGER_NAMES = (
    "thumb_bend_joint",
    "thumb_rota_joint1",
    "thumb_rota_joint2",
    "index_bend_joint",
    "index_joint1",
    "index_joint2",
    "mid_joint1",
    "mid_joint2",
    "ring_joint1",
    "ring_joint2",
    "pinky_joint1",
    "pinky_joint2",
)

# State-vector joint names (38 hinges, same names as the joints in MJCF).
_ARM_JOINTS = tuple(f"{side}_arm_{i}" for side in ("right", "left") for i in range(7))
_HAND_JOINTS = tuple(
    f"{prefix}_{side}_hand_{n}"
    for prefix, side in (("rh", "right"), ("lh", "left"))
    for n in _HAND_FINGER_NAMES
)
STATE_JOINTS = _ARM_JOINTS + _HAND_JOINTS  # 14 + 24 = 38

# Action-space actuator names (same order as STATE_JOINTS). RBY1 arm
# actuators are 1-indexed (right_arm_1_act drives joint right_arm_0), hand
# actuators share the joint name.
_ARM_ACTUATORS = tuple(f"{side}_arm_{i}_act" for side in ("right", "left") for i in range(1, 8))
ACTION_ACTUATORS = _ARM_ACTUATORS + _HAND_JOINTS  # 14 + 24 = 38
assert len(ACTION_ACTUATORS) == 38


@dataclass
class EnvConfig:
    image_size: int = 224
    control_freq: float = 30.0
    max_episode_steps: int = 200
    camera: str = "head_cam"
    home_qpos: np.ndarray | None = None  # if None, uses _default_home_qpos()


class RBY1XHandEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, config: EnvConfig | None = None, render_mode: str = "rgb_array"):
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model,
            height=self.cfg.image_size,
            width=self.cfg.image_size,
        )

        # Resolve indices once.
        self._action_act_idx = np.array(
            [self._actuator_id(n) for n in ACTION_ACTUATORS], dtype=np.int32
        )
        self._state_qadr = np.array(
            [self._joint_qpos_addr(n) for n in STATE_JOINTS], dtype=np.int32
        )

        # Substeps per env.step() call.
        sim_dt = float(self.model.opt.timestep)
        ctrl_dt = 1.0 / float(self.cfg.control_freq)
        self._n_substeps = max(1, int(round(ctrl_dt / sim_dt)))

        # Spaces.
        lo = self.model.actuator_ctrlrange[self._action_act_idx, 0].astype(np.float32)
        hi = self.model.actuator_ctrlrange[self._action_act_idx, 1].astype(np.float32)
        self.action_space = spaces.Box(low=lo, high=hi, dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation.images.head_cam": spaces.Box(
                low=0, high=255,
                shape=(self.cfg.image_size, self.cfg.image_size, 3),
                dtype=np.uint8,
            ),
            "observation.state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(38,), dtype=np.float32,
            ),
        })

        self._step_count = 0

    def _actuator_id(self, name: str) -> int:
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise ValueError(f"actuator {name!r} not found in scene")
        return aid

    def _joint_qpos_addr(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name!r} not found in scene")
        return int(self.model.jnt_qposadr[jid])

    def _default_home_qpos(self) -> np.ndarray:
        """Default home: zeros for everything, head tilted to view table."""
        q = np.zeros(self.model.nq)
        head1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "head_1")
        q[self.model.jnt_qposadr[head1]] = 0.6
        return q

    def _sync_ctrl_to_qpos(self) -> None:
        """Set every position-actuator's ctrl to its joint's current qpos so
        the controller doesn't snap the robot to 0 on the first step."""
        for ai in range(self.model.nu):
            jid = int(self.model.actuator_trnid[ai, 0])
            if jid >= 0:
                self.data.ctrl[ai] = self.data.qpos[self.model.jnt_qposadr[jid]]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        home = self.cfg.home_qpos if self.cfg.home_qpos is not None else self._default_home_qpos()
        self.data.qpos[:] = home
        self._sync_ctrl_to_qpos()
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[self._action_act_idx] = action
        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = self._step_count >= self.cfg.max_episode_steps
        return obs, reward, terminated, truncated, {}

    def _get_obs(self) -> dict:
        self.renderer.update_scene(self.data, camera=self.cfg.camera)
        img = self.renderer.render()
        state = self.data.qpos[self._state_qadr].astype(np.float32)
        return {
            "observation.images.head_cam": img,
            "observation.state": state,
        }

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        self.renderer.update_scene(self.data, camera=self.cfg.camera)
        return self.renderer.render()

    def close(self):
        pass
