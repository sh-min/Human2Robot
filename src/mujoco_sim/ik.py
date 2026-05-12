"""Inverse kinematics for the RBY1 + XHand bimanual scene.

Damped least-squares Jacobian IK that updates only the 7 arm joints on
one side; every other DOF (base freejoint, torso, head, other arm, hand
fingers) is mechanically untouched.

Usage:
    new_qpos = solve_wrist_ik(model, qpos, "link_right_arm_6",
                              target_pos, target_quat)
"""

from __future__ import annotations

import numpy as np

import mujoco


def _quat_err(quat_cur: np.ndarray, quat_tgt: np.ndarray) -> np.ndarray:
    """Rotation-vector error that rotates quat_cur to quat_tgt (3-vector)."""
    qe = np.zeros(4)
    qcinv = np.zeros(4)
    mujoco.mju_negQuat(qcinv, quat_cur)
    mujoco.mju_mulQuat(qe, quat_tgt, qcinv)
    rotvec = np.zeros(3)
    mujoco.mju_quat2Vel(rotvec, qe, 1.0)
    return rotvec


def solve_wrist_ik(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    wrist_body: str,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    max_iters: int = 200,
    damping: float = 0.05,
    step_scale: float = 0.5,
    tol_pos: float = 1e-3,
    tol_ori: float = 1e-2,
) -> np.ndarray:
    """Solve IK to place ``wrist_body`` at the given world target pose.

    Only the 7 arm joints on the same side as ``wrist_body`` (right/left)
    are updated. Damped least-squares step with explicit Jacobian column
    selection — no soft regularization, no base/torso/other-arm drift.

    target_quat is (w, x, y, z) in MuJoCo convention.
    Returns a new qpos array of the same length as input.
    """
    if "right" in wrist_body:
        side = "right"
    elif "left" in wrist_body:
        side = "left"
    else:
        raise ValueError(f"can't infer side from wrist_body={wrist_body!r}")

    # qpos / v indices for the 7 arm joints (all hinges).
    qpos_idx = []
    dof_idx = []
    for i in range(7):
        jname = f"{side}_arm_{i}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise ValueError(f"joint {jname} not found")
        qpos_idx.append(int(model.jnt_qposadr[jid]))
        dof_idx.append(int(model.jnt_dofadr[jid]))
    qpos_idx = np.array(qpos_idx)
    dof_idx = np.array(dof_idx)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wrist_body)
    if body_id < 0:
        raise ValueError(f"body {wrist_body} not found")

    data = mujoco.MjData(model)
    data.qpos[:] = qpos.copy()

    target_pos = np.asarray(target_pos, dtype=float)
    target_quat = np.asarray(target_quat, dtype=float)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)

        cur_pos = data.xpos[body_id]
        cur_quat = data.xquat[body_id]
        err_p = target_pos - cur_pos
        err_r = _quat_err(cur_quat, target_quat)
        if np.linalg.norm(err_p) < tol_pos and np.linalg.norm(err_r) < tol_ori:
            break

        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        J = np.vstack([jacp[:, dof_idx], jacr[:, dof_idx]])  # 6 x 7
        err = np.concatenate([err_p, err_r])

        # Damped least squares: dq = J^T (J J^T + lam^2 I)^-1 err
        dq = J.T @ np.linalg.solve(J @ J.T + (damping**2) * np.eye(6), err)
        data.qpos[qpos_idx] += step_scale * dq

    return data.qpos.copy()
