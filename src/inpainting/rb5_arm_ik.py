"""RB5-850e 6-DOF arm IK for the robot-overlay.

There is no real RB5 in the source video — the arm is synthesised. So we (a)
place a synthetic base relative to the camera, auto-fit to the tracked wrist
trajectory (overridable), and (b) solve per-frame joint angles so the tool
flange (`tcp`) reaches the tracked wrist. Pinocchio damped-least-squares IK,
warm-started across frames. Unreachable targets are clamped to the reachable
sphere so the arm stops extending instead of jittering.

Frames: wrist poses are in the OpenCV camera frame (x-right, y-down, z-forward),
metres. Scene "up" is -y_cam, so the arm base mounts with its Z axis along -y.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pinocchio as pin
from scipy.signal import savgol_filter

RB5_URDF = str(Path(__file__).resolve().parents[2] / "third_party" / "rb5_850e" / "rb5_850e.urdf")
EE_FRAME = "tcp"
JOINT_NAMES = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")


def load_model():
    model = pin.buildModelFromUrdf(RB5_URDF)
    data = model.createData()
    fid = model.getFrameId(EE_FRAME)
    return model, data, fid


def reach_radius(model, data, fid) -> float:
    """Max flange distance from base (fully extended), a bit conservative."""
    q = pin.neutral(model)
    # sweep shoulder/elbow to the straight-out configuration and measure
    best = 0.0
    for sh in np.linspace(-1.5, 1.5, 7):
        q2 = q.copy(); q2[1] = sh
        pin.forwardKinematics(model, data, q2); pin.updateFramePlacements(model, data)
        best = max(best, float(np.linalg.norm(data.oMf[fid].translation)))
    return best


def _R_cam_base() -> np.ndarray:
    """Base mounted vertically: base-Z along scene-up (-y_cam)."""
    x = np.array([1.0, 0.0, 0.0]); z = np.array([0.0, -1.0, 0.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def auto_fit_base(wrist_pos_cam: np.ndarray, model, data, fid,
                  frac: float = 0.7, override=None) -> pin.SE3:
    """T_cam_base placing the base below+behind the wrist centroid so the whole
    trajectory sits within `frac` of the reach radius. `override` (4x4) wins.
    """
    if override is not None:
        return pin.SE3(np.asarray(override)[:3, :3], np.asarray(override)[:3, 3])
    R = _R_cam_base()
    c = np.mean(wrist_pos_cam, axis=0)
    reach = reach_radius(model, data, fid)
    # target distance from base to the farthest wrist = frac*reach; mount along
    # scene-up (-y_cam => base sits +y i.e. below the wrists) plus a little +z (back).
    up_cam = np.array([0.0, 1.0, 0.0])   # +y = down in cam => base below wrists
    dmax = float(np.max(np.linalg.norm(wrist_pos_cam - c, axis=1)))
    d = max(0.15, frac * reach)          # base ~d below centroid
    base_pos = c + up_cam * d + np.array([0.0, 0.0, 0.10])
    return pin.SE3(R, base_pos)


def solve_ik(model, data, fid, q0, target: pin.SE3, *, w_ori=0.0,
             iters=120, damping=5e-2, step=0.4, tol_p=1e-3, tol_o=3e-2):
    """DLS IK with joint-limit clamping. w_ori weights orientation error
    (default 0 = position-only, which a 6-DOF arm can track smoothly; a free
    human wrist's full orientation cannot be followed without singularity jitter).
    """
    lo, hi = model.lowerPositionLimit, model.upperPositionLimit
    W = np.diag([1., 1., 1., w_ori, w_ori, w_ori])
    q = q0.copy(); err = np.zeros(6)
    for _ in range(iters):
        pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
        err = W @ pin.log6(data.oMf[fid].inverse() * target).vector
        if np.linalg.norm(err[:3]) < tol_p and np.linalg.norm(err[3:]) < tol_o:
            break
        J = W @ pin.computeFrameJacobian(model, data, q, fid, pin.LOCAL)
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), err)
        q = np.clip(pin.integrate(model, q, step * dq), lo, hi)
    return q, float(np.linalg.norm(err[:3]))


def solve_sequence(wrist_poses_cam, valid, T_cam_base, model, data, fid, *,
                   w_ori=0.0, smooth_win=11):
    """wrist_poses_cam: (T,4,4) flange target in cam frame. Position-only IK by
    default + savgol smoothing of the joint trajectory. Unreachable targets are
    clamped onto the reach sphere so the arm stops extending (no jitter).
    Returns (q[T,6], pos_err[T], reachable[T]).
    """
    T_base_cam = T_cam_base.inverse()
    reach = reach_radius(model, data, fid) * 0.99
    N = len(wrist_poses_cam)
    q = np.zeros((N, model.nq)); perr = np.full(N, np.nan); reachable = np.zeros(N, bool)
    q_prev = pin.neutral(model)
    for t in range(N):
        if not valid[t]:
            q[t] = q_prev; continue
        tgt = T_base_cam * pin.SE3(wrist_poses_cam[t][:3, :3], wrist_poses_cam[t][:3, 3])
        p = tgt.translation; r = float(np.linalg.norm(p))
        reachable[t] = r <= reach
        if r > reach:
            tgt = pin.SE3(tgt.rotation, p * (reach / r))
        q_sol, e = solve_ik(model, data, fid, q_prev, tgt, w_ori=w_ori)
        q[t] = q_sol; perr[t] = e; q_prev = q_sol
    # temporal smoothing on the valid joint trajectory (overlay needs visual, not
    # dynamic, smoothness)
    v = np.asarray(valid, bool)
    if smooth_win and v.sum() > smooth_win:
        q[v] = np.clip(savgol_filter(q[v], smooth_win, 2, axis=0),
                       model.lowerPositionLimit, model.upperPositionLimit)
    return q, perr, reachable
