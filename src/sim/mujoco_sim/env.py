"""Gymnasium RBY1 + XHand environment using the policy's 38-D schema.

The policy dataset does *not* contain arm joint targets.  Its per-hand layout
is finger qpos (12), wrist position in the RBY1 base frame (3), and wrist
quaternion xyzw (4).  ``step`` therefore solves joint-limited arm IK before
driving the MuJoCo position actuators.  Observations reconstruct the same
38-D schema from the simulated robot state.

Each reset scatters the tabletop objects at random positions and yaws inside
SPAWN_X x SPAWN_Y; pass ``reset(seed=...)`` to make it repeatable.
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


# Region the tabletop objects' *footprints* must stay inside on reset, in
# world coords.  The table is 1 m deep x 2 m wide centered at (0.9, 0) with
# its top at z=1.0, so this box keeps every object clear of the edges and
# inside the arms' reach.  Each object's center is drawn from this box
# shrunk by its own footprint radius, so a big container never overhangs
# the way a fixed center-region would let it.
SPAWN_X = (0.47, 0.92)
SPAWN_Y = (-0.55, 0.55)
SPAWN_Z = 1.0        # table top; each object's origin is its bottom center
SPAWN_GAP = 0.02     # clear space left between two object footprints
SPAWN_TRIES = 400    # rejection-sampling attempts per object
SPAWN_RESTARTS = 4   # whole-layout restarts before giving up


def _footprint_radius(model, body_id: int) -> float:
    """Radius of a circle around the body's geoms seen from above, centered
    on the body origin.  Rotation-invariant about z, so it stays valid for
    any yaw we spawn the object at."""
    radius = 0.0
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body_id:
            continue
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, model.geom_quat[geom])
        mat = mat.reshape(3, 3)
        size, geom_type = model.geom_size[geom], model.geom_type[geom]
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            extent = np.abs(mat) @ size
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            axis = mat[:, 2]
            extent = (
                np.abs(axis) * size[1]
                + size[0] * np.sqrt(np.maximum(0, 1 - axis**2))
            )
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent = np.abs(mat[:, 2]) * size[1] + size[0]
        else:
            extent = np.full(3, model.geom_rbound[geom])
        pos = model.geom_pos[geom]
        radius = max(
            radius,
            float(np.hypot(abs(pos[0]) + extent[0], abs(pos[1]) + extent[1])),
        )
    return radius


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
        # Free joints belong to the tabletop objects only -- the robot's
        # base freejoint is deleted when the scene is composed.  Biggest
        # first: placing the bulky objects while the table is still empty
        # is what makes rejection sampling converge.
        self._objects = sorted(
            (
                (
                    int(self.model.jnt_qposadr[joint]),
                    _footprint_radius(
                        self.model, int(self.model.jnt_bodyid[joint])
                    ),
                )
                for joint in range(self.model.njnt)
                if self.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
            ),
            key=lambda obj: -obj[1],
        )

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

    def _randomize_objects(self, qpos: np.ndarray) -> None:
        """Scatter the tabletop objects over SPAWN_X x SPAWN_Y, upright and
        with random yaw.  A sample is rejected if the object's footprint
        would come within SPAWN_GAP of an already-placed one.

        Getting stuck is an unlucky *early* placement boxing in a later
        object, not the last object being unlucky, so the whole layout
        restarts rather than grinding more tries on the one that stuck.
        With nine objects a single pass fails about one reset in 500;
        restarting clears 20k seeds without a miss."""
        for _ in range(SPAWN_RESTARTS):
            placed: list[tuple[float, float, float]] = []
            for _qadr, radius in self._objects:
                for _ in range(SPAWN_TRIES):
                    x = float(self.np_random.uniform(
                        SPAWN_X[0] + radius, SPAWN_X[1] - radius
                    ))
                    y = float(self.np_random.uniform(
                        SPAWN_Y[0] + radius, SPAWN_Y[1] - radius
                    ))
                    if all(
                        (x - px) ** 2 + (y - py) ** 2
                        >= (radius + pr + SPAWN_GAP) ** 2
                        for px, py, pr in placed
                    ):
                        break
                else:
                    break                      # stuck -- restart the layout
                placed.append((x, y, radius))
            if len(placed) == len(self._objects):
                break
        else:
            raise RuntimeError(
                f"could not lay out {len(self._objects)} objects in "
                f"{SPAWN_RESTARTS} restarts of {SPAWN_TRIES} tries each: "
                f"the spawn region is too small for them"
            )

        for (qadr, _), (x, y, _) in zip(self._objects, placed):
            yaw = float(self.np_random.uniform(-np.pi, np.pi))
            qpos[qadr : qadr + 3] = (x, y, SPAWN_Z)
            qpos[qadr + 3 : qadr + 7] = (
                np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)
            )

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
        self._randomize_objects(self.data.qpos)
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
