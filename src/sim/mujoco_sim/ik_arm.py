"""Pinocchio-based IK + camera-frame helpers for the RBY1 + XHand scene.

Public API:
    solve_arm_ik(model, data, q0, side, target_SE3) -> (q_new, p_err, o_err)
        Damped-LS IK on the 7 arm joints of ``side`` (right/left). All
        other dofs in q0 are preserved.
    head_cam_world(model, data, q) -> pin.SE3
        World pose of the ``head_cam`` (pinocchio's MJCF loader doesn't
        surface cameras, so we compose link_head_2's pose with the
        camera mount transform).
    cam_to_world(p_cv, R_cv, T_world_mjcam) -> pin.SE3
        CV-camera-frame pose -> world SE3.

qpos layout matches MuJoCo exactly (39 hinges, same order).

Camera convention: HaWoR/CV is +x right, +y DOWN, +z INTO scene; MuJoCo
camera is +x right, +y UP, +z OUT of scene. diag(1, -1, -1) maps between.

Running the module as a script loads ``rgb_hawor/final_pose.pkl`` from
the default episode, solves the bimanual IK at the middle frame, and
renders head_cam + front_view to ``output/`` for visual inspection.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[3]
SCENE = REPO / "src/sim/mujoco_sim/scenes/rby1_xhand.xml"

# CV camera frame -> MuJoCo camera frame.
_CV_TO_MJ = pin.SE3(np.diag([1.0, -1.0, -1.0]), np.zeros(3))

# head_cam mount on link_head_2 (from compose_rby1_xhand.py):
#   pos=(0.12, 0, 0.05), xyaxes=(0,-1,0, 0,0,1)
# -> cam_x=(0,-1,0), cam_y=(0,0,1), cam_z = cam_x x cam_y = (-1,0,0)
_T_HEAD2_CAM = pin.SE3(
    np.array([[0.0, 0.0, -1.0],
              [-1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0]]),
    np.array([0.12, 0.0, 0.05]),
)

# Friend's wrist_pos is the MANO/XHand wrist-link origin. In our composed
# scene the XHand was attached to link_right_arm_6 at EE_OFFSET = (0,0,-0.1261)
# (compose_rby1_xhand.py:30), so the IK target for link_*_arm_6 is the wrist
# pose shifted by +0.1261 along its own local z.
_T_WRIST_ARM6 = pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.1261]))

# Reachable, elbow-down configurations used only as deterministic fallback
# starts when a warm-started nonlinear IK solve becomes trapped.
_SAFE_ARM_QPOS = {
    "left": np.array([-1.2, 0.4, -1.2, -1.0, 0.0, 1.2, 0.8]),
    "right": np.array([-1.2, -0.4, 1.2, -1.0, 0.0, 1.2, -0.8]),
}


def _arm_dof_idx(model: pin.Model, side: str) -> np.ndarray:
    """v-index of the 7 arm joints for ``side`` (right/left). Hinges so
    idx_v == idx_q for these."""
    idx = []
    for i in range(7):
        jname = f"{side}_arm_{i}"
        jid = model.getJointId(jname)
        idx.append(model.idx_vs[jid])
    return np.array(idx, dtype=int)


def head_cam_world(model: pin.Model, data: pin.Data, q: np.ndarray) -> pin.SE3:
    """World pose of head_cam at configuration ``q`` (any head pitch)."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    fid = model.getFrameId("link_head_2")
    return data.oMf[fid] * _T_HEAD2_CAM


def cam_to_world(p_cv: np.ndarray, R_cv: np.ndarray, T_world_mjcam: pin.SE3) -> pin.SE3:
    """CV-cam-frame wrist-link pose -> world SE3 (still at the wrist link,
    not yet shifted to link_*_arm_6)."""
    T_cvcam_wrist = pin.SE3(R_cv, p_cv)
    return T_world_mjcam * _CV_TO_MJ * T_cvcam_wrist


def wrist_to_arm6(T_world_wrist: pin.SE3) -> pin.SE3:
    """Shift the wrist-link target onto link_*_arm_6 (the body our IK
    drives) so the visible XHand wrist lands at T_world_wrist."""
    return T_world_wrist * _T_WRIST_ARM6


def solve_arm_ik(
    model: pin.Model, data: pin.Data, q0: np.ndarray,
    side: str, target: pin.SE3,
    max_iters: int = 80, damping: float = 1e-2, step_scale: float = 0.5,
    tol_pos: float = 1e-3, tol_ori: float = 1e-2,
) -> tuple[np.ndarray, float, float]:
    """Joint-limited IK on the 7 arm joints of ``side``.

    ``damping`` and ``step_scale`` remain in the signature for compatibility
    with earlier callers.  The bounded nonlinear solve is more reliable than
    the old unconstrained DLS loop, which could return multi-turn, physically
    impossible arm angles.  Other DoFs in ``q0`` are preserved.
    """
    q = q0.copy()
    fid = model.getFrameId(f"link_{side}_arm_6")
    dof = _arm_dof_idx(model, side)
    lower = model.lowerPositionLimit[dof].astype(np.float64) + 1e-7
    upper = model.upperPositionLimit[dof].astype(np.float64) - 1e-7
    x0 = np.clip(q[dof], lower, upper)
    orientation_weight = 0.12

    def residual(x: np.ndarray) -> np.ndarray:
        q[dof] = x
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current = data.oMf[fid]
        position = current.translation - target.translation
        orientation = pin.log3(current.rotation.T @ target.rotation)
        return np.concatenate([position, orientation_weight * orientation])

    def run(start: np.ndarray, evaluations: int):
        return least_squares(
            residual,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            max_nfev=evaluations,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
        )

    solutions = [run(x0, max_iters)]
    first_error = residual(solutions[0].x)
    if (
        np.linalg.norm(first_error[:3]) > max(tol_pos, 5e-3)
        or np.linalg.norm(first_error[3:]) / orientation_weight
        > max(tol_ori, np.radians(3.0))
    ):
        # A 7-DoF arm has multiple IK branches.  Try two fixed starts only
        # when the temporally warm-started branch is demonstrably bad.
        solutions.extend([
            run(_SAFE_ARM_QPOS[side], max_iters * 2),
            run(0.5 * (lower + upper), max_iters * 2),
        ])

    # Prefer target accuracy; use closeness to the previous frame only as a
    # tiny tie-breaker between equivalent redundant-arm solutions.
    def score(solution) -> float:
        return float(
            np.linalg.norm(residual(solution.x))
            + 1e-7 * np.linalg.norm(solution.x - x0)
        )

    solution = min(solutions, key=score)
    q[dof] = solution.x
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    current = data.oMf[fid]
    pos_err = float(np.linalg.norm(current.translation - target.translation))
    ori_err = float(
        np.linalg.norm(pin.log3(current.rotation.T @ target.rotation))
    )
    return q, pos_err, ori_err


def apply_frame(
    pin_model: pin.Model, pin_data: pin.Data,
    muj_model: mujoco.MjModel, muj_data: mujoco.MjData,
    pose: dict, t: int, T_world_cam: pin.SE3,
) -> dict[str, tuple[float, float]]:
    """Solve IK + copy fingers for frame t. Mutates ``muj_data.qpos`` and
    runs mj_forward. Returns per-side (pos_err, ori_err).

    MuJoCo's qpos may carry extra DOFs that pinocchio's model doesn't see
    (e.g. a free-joint object). Those live past pin_model.nq; we slice to
    the robot part for IK and merge back."""
    robot_nq = pin_model.nq
    q_full = muj_data.qpos.copy()
    q = q_full[:robot_nq].copy()
    errors = {}
    for side, prefix in (("right", "rh_"), ("left", "lh_")):
        s = pose[side]
        if not bool(s["valid"][t]):
            continue
        if s.get("coordinate_frame", pose.get("coordinate_frame")) == "rby1_base":
            T_wrist = pin.SE3(
                s["wrist_rot"][t].astype(float),
                s["wrist_pos"][t].astype(float),
            )
        else:
            T_wrist = cam_to_world(
                s["wrist_pos"][t].astype(float),
                s["wrist_rot"][t].astype(float),
                T_world_cam,
            )
        q, perr, oerr = solve_arm_ik(pin_model, pin_data, q, side, wrist_to_arm6(T_wrist))
        errors[side] = (perr, oerr)
        for jname, qval in zip(s["joint_names"], s["qpos"][t]):
            jid = mujoco.mj_name2id(muj_model, mujoco.mjtObj.mjOBJ_JOINT, prefix + jname)
            if jid >= 0:
                q[muj_model.jnt_qposadr[jid]] = float(qval)
    q_full[:robot_nq] = q
    muj_data.qpos[:] = q_full
    mujoco.mj_forward(muj_model, muj_data)
    return errors


def main():
    pkl_path = REPO / "data/kitchen_dataset/0412_val/episode_0/rgb_hawor/final_pose.pkl"
    with pkl_path.open("rb") as f:
        pose = pickle.load(f)

    pin_model = pin.buildModelFromMJCF(str(SCENE))
    pin_data = pin_model.createData()
    muj_model = mujoco.MjModel.from_xml_path(str(SCENE))
    muj_data = mujoco.MjData(muj_model)
    # Robot joints come first in MuJoCo's qpos; the object's free joint and
    # optional articulation follow. Pinocchio sees only the robot, so
    # ROBOT_NQ is the slice into MuJoCo's state corresponding to its
    # kinematic tree.
    ROBOT_NQ = pin_model.nq

    # Home pose: MJCF defaults (including object placement), override head pitch.
    # Also bias arm joints to mid-range so the 7-DOF IK redundancy is
    # resolved into the natural elbow-down branch rather than the
    # elbow-inverted (out-of-range) one that zero-warm-start picks.
    q_home = muj_model.qpos0.copy()
    hjid_muj = mujoco.mj_name2id(muj_model, mujoco.mjtObj.mjOBJ_JOINT, "head_1")
    q_home[muj_model.jnt_qposadr[hjid_muj]] = 0.6
    for side in ("right", "left"):
        for i in range(7):
            jid = mujoco.mj_name2id(muj_model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_arm_{i}")
            lo, hi = muj_model.jnt_range[jid]
            q_home[muj_model.jnt_qposadr[jid]] = 0.5 * (lo + hi)
    muj_data.qpos[:] = q_home
    mujoco.mj_forward(muj_model, muj_data)

    T_world_cam = head_cam_world(pin_model, pin_data, q_home[:ROBOT_NQ])
    print(f"head_cam world pos={T_world_cam.translation}")

    t = pose["T"] // 2
    errs = apply_frame(pin_model, pin_data, muj_model, muj_data, pose, t, T_world_cam)
    for side, (pe, oe) in errs.items():
        print(f"{side}: pos_err={pe*1000:.2f} mm, ori_err={np.degrees(oe):.2f} deg")

    out_dir = REPO / "output"
    out_dir.mkdir(exist_ok=True)
    r = mujoco.Renderer(muj_model, height=720, width=1280)
    for cam in ("head_cam", "front_view"):
        r.update_scene(muj_data, camera=cam)
        img = r.render()
        imageio.imwrite(str(out_dir / f"retarget_frame_{t}_{cam}.png"), img)
        print(f"wrote output/retarget_frame_{t}_{cam}.png")


if __name__ == "__main__":
    main()
