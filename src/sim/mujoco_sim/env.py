"""Gymnasium RBY1 + XHand environment using the policy's 38-D schema.

The policy dataset does *not* contain arm joint targets.  Its per-hand layout
is finger qpos (12), wrist position in the RBY1 base frame (3), and wrist
quaternion xyzw (4).  ``step`` therefore solves joint-limited arm IK before
driving the MuJoCo position actuators.  Observations reconstruct the same
38-D schema from the simulated robot state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import pinocchio as pin
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from pkl_to_lerobot.schema import BIMANUAL_DIM, FINGER_DOF, PER_HAND_DIM

from .ik_arm import (
    _SAFE_ARM_QPOS,
    _T_WRIST_ARM6,
    _arm_dof_idx,
    solve_arm_ik,
)

REPO = Path(__file__).resolve().parents[3]
SCENE = REPO / "src/sim/mujoco_sim/scenes/rby1_xhand.xml"

# This order exactly matches final_pose.pkl's qpos/joint_names and therefore
# pkl_to_lerobot.schema.  It is not the URDF declaration order.
_FINGER_SUFFIXES = (
    "index_bend_joint",
    "index_joint1",
    "index_joint2",
    "mid_joint1",
    "mid_joint2",
    "pinky_joint1",
    "pinky_joint2",
    "ring_joint1",
    "ring_joint2",
    "thumb_bend_joint",
    "thumb_rota_joint1",
    "thumb_rota_joint2",
)


def _finger_joint_names(side: str) -> tuple[str, ...]:
    prefix = "rh" if side == "right" else "lh"
    return tuple(
        f"{prefix}_{side}_hand_{suffix}" for suffix in _FINGER_SUFFIXES
    )


@dataclass
class EnvConfig:
    image_size: int = 224
    control_freq: float = 30.0
    max_episode_steps: int = 200
    camera: str = "head_cam"
    active_hands: tuple[str, ...] = ("left",)
    home_qpos: np.ndarray | None = None


class RBY1XHandEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str = "rgb_array",
    ):
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        unknown = set(self.cfg.active_hands) - {"right", "left"}
        if unknown:
            raise ValueError(f"Unknown active hands: {sorted(unknown)}")

        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.pin_model = pin.buildModelFromMJCF(str(SCENE))
        self.pin_data = self.pin_model.createData()
        self.renderer = mujoco.Renderer(
            self.model,
            height=self.cfg.image_size,
            width=self.cfg.image_size,
        )

        self._arm_dof = {
            side: _arm_dof_idx(self.pin_model, side)
            for side in ("right", "left")
        }
        self._arm_act = {
            side: np.array([
                self._actuator_id(f"{side}_arm_{index}_act")
                for index in range(1, 8)
            ])
            for side in ("right", "left")
        }
        self._finger_names = {
            side: _finger_joint_names(side) for side in ("right", "left")
        }
        self._finger_act = {
            side: np.array([
                self._actuator_id(name) for name in self._finger_names[side]
            ])
            for side in ("right", "left")
        }
        self._finger_qadr = {
            side: np.array([
                self._joint_qpos_addr(name) for name in self._finger_names[side]
            ])
            for side in ("right", "left")
        }

        sim_dt = float(self.model.opt.timestep)
        ctrl_dt = 1.0 / float(self.cfg.control_freq)
        self._n_substeps = max(1, int(round(ctrl_dt / sim_dt)))

        low = np.empty(BIMANUAL_DIM, dtype=np.float32)
        high = np.empty(BIMANUAL_DIM, dtype=np.float32)
        for side in ("right", "left"):
            offset = 0 if side == "right" else PER_HAND_DIM
            finger_range = self.model.actuator_ctrlrange[
                self._finger_act[side]
            ]
            low[offset : offset + FINGER_DOF] = finger_range[:, 0]
            high[offset : offset + FINGER_DOF] = finger_range[:, 1]
            low[offset + 12 : offset + 15] = [-0.5, -1.0, 0.3]
            high[offset + 12 : offset + 15] = [1.5, 1.0, 2.0]
            low[offset + 15 : offset + 19] = -1.0
            high[offset + 15 : offset + 19] = 1.0
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation.images.head_cam": spaces.Box(
                low=0,
                high=255,
                shape=(self.cfg.image_size, self.cfg.image_size, 3),
                dtype=np.uint8,
            ),
            "observation.state": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(BIMANUAL_DIM,),
                dtype=np.float32,
            ),
        })
        self._step_count = 0
        self._last_ik_errors: dict[str, tuple[float, float]] = {}

    def _actuator_id(self, name: str) -> int:
        actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        if actuator_id < 0:
            raise ValueError(f"actuator {name!r} not found in scene")
        return actuator_id

    def _joint_qpos_addr(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise ValueError(f"joint {name!r} not found in scene")
        return int(self.model.jnt_qposadr[joint_id])

    def _default_home_qpos(self) -> np.ndarray:
        qpos = self.model.qpos0.copy()
        head = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "head_1"
        )
        qpos[self.model.jnt_qposadr[head]] = 0.6
        for side in ("right", "left"):
            for index, value in enumerate(_SAFE_ARM_QPOS[side]):
                joint = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"{side}_arm_{index}",
                )
                qpos[self.model.jnt_qposadr[joint]] = value
        return qpos

    def _sync_ctrl_to_qpos(self) -> None:
        for actuator in range(self.model.nu):
            joint = int(self.model.actuator_trnid[actuator, 0])
            if joint >= 0:
                qadr = self.model.jnt_qposadr[joint]
                self.data.ctrl[actuator] = self.data.qpos[qadr]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        home = (
            self.cfg.home_qpos
            if self.cfg.home_qpos is not None
            else self._default_home_qpos()
        )
        self.data.qpos[:] = home
        self._sync_ctrl_to_qpos()
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._last_ik_errors = {}
        return self._get_obs(), {
            "coordinate_frame": "rby1_base",
            "active_hands": self.cfg.active_hands,
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (BIMANUAL_DIM,):
            raise ValueError(
                f"Expected action shape {(BIMANUAL_DIM,)}, got {action.shape}"
            )
        action = np.clip(action, self.action_space.low, self.action_space.high)
        robot_q = self.data.qpos[: self.pin_model.nq].copy()
        errors = {}

        for side in ("right", "left"):
            if side not in self.cfg.active_hands:
                continue
            offset = 0 if side == "right" else PER_HAND_DIM
            fingers = action[offset : offset + FINGER_DOF]
            wrist_pos = action[offset + 12 : offset + 15]
            wrist_quat = action[offset + 15 : offset + 19].astype(np.float64)
            quat_norm = float(np.linalg.norm(wrist_quat))
            if quat_norm < 1e-6:
                raise ValueError(f"{side} wrist quaternion has zero norm")
            wrist_rot = Rotation.from_quat(
                wrist_quat / quat_norm
            ).as_matrix()
            target_wrist = pin.SE3(wrist_rot, wrist_pos.astype(np.float64))
            robot_q, pos_error, ori_error = solve_arm_ik(
                self.pin_model,
                self.pin_data,
                robot_q,
                side,
                target_wrist * _T_WRIST_ARM6,
            )
            errors[side] = (pos_error, ori_error)
            self.data.ctrl[self._arm_act[side]] = robot_q[
                self._arm_dof[side]
            ]
            self.data.ctrl[self._finger_act[side]] = fingers

        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)
        self._last_ik_errors = errors
        self._step_count += 1
        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = self._step_count >= self.cfg.max_episode_steps
        info = {
            "ik_error": {
                side: {
                    "position_mm": float(values[0] * 1000.0),
                    "orientation_deg": float(np.degrees(values[1])),
                }
                for side, values in errors.items()
            }
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> dict:
        self.renderer.update_scene(self.data, camera=self.cfg.camera)
        image = self.renderer.render()
        state = np.zeros(BIMANUAL_DIM, dtype=np.float32)
        robot_q = self.data.qpos[: self.pin_model.nq]
        pin.forwardKinematics(self.pin_model, self.pin_data, robot_q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        for side in ("right", "left"):
            if side not in self.cfg.active_hands:
                continue
            offset = 0 if side == "right" else PER_HAND_DIM
            state[offset : offset + FINGER_DOF] = self.data.qpos[
                self._finger_qadr[side]
            ]
            arm6 = self.pin_data.oMf[
                self.pin_model.getFrameId(f"link_{side}_arm_6")
            ]
            wrist = arm6 * _T_WRIST_ARM6.inverse()
            state[offset + 12 : offset + 15] = wrist.translation
            state[offset + 15 : offset + 19] = Rotation.from_matrix(
                wrist.rotation
            ).as_quat()
        return {
            "observation.images.head_cam": image,
            "observation.state": state,
        }

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        self.renderer.update_scene(self.data, camera=self.cfg.camera)
        return self.renderer.render()

    def close(self):
        self.renderer.close()
