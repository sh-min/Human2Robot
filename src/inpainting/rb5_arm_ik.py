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

# Drive IK with the physical flange (link6) as the end effector, NOT "tcp". tcp is a
# virtual tool point 9.67cm beyond the flange (tcp_joint xyz="0 -0.0967 0"), so the
# visible flange plate sits at link6's -y end (measured mesh y in [-0.097, -0.061]).
# rb5_build_overlay_input.py builds the link6 target so the flange face mates flush
# with the xhand mount plate (see its FLANGE_TCP mate construction); we solve for
# link6 directly so that mate is exact.
RB5_URDF = str(Path(__file__).resolve().parents[2] / "third_party" / "rb5_850e" / "rb5_850e.urdf")
EE_FRAME = "link6"
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


def _R_cam_base(mount: str = "floor") -> np.ndarray:
    """Base-Z (the mount/joint-0 axis). floor: points scene-up (-y_cam), arm
    rises from below. ceiling: points down (+y_cam), arm hangs from above."""
    x = np.array([1.0, 0.0, 0.0])
    z = np.array([0.0, -1.0, 0.0]) if mount == "floor" else np.array([0.0, 1.0, 0.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def auto_fit_base(wrist_pos_cam: np.ndarray, model, data, fid,
                  frac: float = 0.7, override=None, shift=(0.0, 0.0, 0.0),
                  mount: str = "floor") -> pin.SE3:
    """T_cam_base placing the base `frac`*reach from the wrist centroid so the
    whole trajectory is reachable. mount="floor" puts the base below (arm rises
    from the bottom); mount="ceiling" puts it above (arm hangs from the top).
    `override` (4x4) wins; `shift` (x,y,z, CV cam frame) nudges the base.
    """
    if override is not None:
        return pin.SE3(np.asarray(override)[:3, :3], np.asarray(override)[:3, 3])
    R = _R_cam_base(mount)
    c = np.mean(wrist_pos_cam, axis=0)
    reach = reach_radius(model, data, fid)
    # offset from centroid to base: +y (down) for floor => base below wrists;
    # -y (up) for ceiling => base above wrists. Plus a little +z (further away).
    off = np.array([0.0, 1.0, 0.0]) if mount == "floor" else np.array([0.0, -1.0, 0.0])
    d = max(0.15, frac * reach)
    base_pos = c + off * d + np.array([0.0, 0.0, 0.10]) + np.asarray(shift, float)
    return pin.SE3(R, base_pos)


def optimize_base(flange_poses, wrist_pos_cam, valid, model, data, fid, *,
                  w_ori=1.0, n_sub=120, mount="floor", verbose=True):
    """Search base placements and return the T_cam_base that reaches the wrist
    trajectory with the least position error + jitter (orientation-weighted by
    w_ori). Sweeps a lateral(x)/depth(z)/distance(frac) grid on a subsampled
    trajectory. Returns (T_cam_base, info dict)."""
    vi = np.flatnonzero(valid)
    sub = vi[:: max(1, len(vi) // n_sub)]
    sub_fl = flange_poses[sub]
    sub_valid = np.ones(len(sub), bool)
    reach = reach_radius(model, data, fid) * 0.99
    best = None
    for frac in (0.5, 0.6, 0.7):
        for dx in np.linspace(-0.10, 0.30, 5):
            for dz in np.linspace(-0.15, 0.25, 5):
                Tcb = auto_fit_base(wrist_pos_cam[valid], model, data, fid,
                                    frac=frac, shift=(dx, 0.0, dz), mount=mount)
                q, perr, rok = solve_sequence(sub_fl, sub_valid, Tcb, model, data, fid,
                                              w_ori=w_ori, smooth_win=0)
                if rok.mean() < 0.99:
                    continue
                p90 = float(np.nanpercentile(perr, 90))
                jit = float((np.abs(np.diff(q, axis=0)).sum(1) > 0.3).mean())
                cost = p90 * 1000.0 + 25.0 * jit    # mm + jitter%
                if best is None or cost < best["cost"]:
                    best = dict(cost=cost, Tcb=Tcb, frac=frac, dx=float(dx), dz=float(dz),
                                p90_mm=p90 * 1000, jitter=jit)
    if best is None:   # nothing fully reachable -> plain auto-fit
        return auto_fit_base(wrist_pos_cam[valid], model, data, fid, mount=mount), {"note": "no reachable base found"}
    if verbose:
        print(f"[opt-base] frac={best['frac']} dx={best['dx']:.3f} dz={best['dz']:.3f} "
              f"-> p90-err {best['p90_mm']:.1f}mm jitter {best['jitter']*100:.1f}% "
              f"base(cam)={best['Tcb'].translation.round(3)}")
    return best["Tcb"], best


# which image corner -> (x-sign, y-sign, base mount). Bottom corners rise from a
# floor base; top corners hang from a ceiling base.
_CORNER = {
    "bottomright": (+1, +1, "floor"),
    "bottomleft":  (-1, +1, "floor"),
    "topright":    (+1, -1, "ceiling"),
    "topleft":     (-1, -1, "ceiling"),
}


def place_corner(flange_poses, wrist_pos_cam, valid, model, data, fid, *,
                 corner="bottomright", focal, img_w=1920, img_h=1080,
                 w_ori=1.0, n_sub=100, verbose=True):
    """Place the base so it PROJECTS into the given image `corner`, generalised from the
    wrist trajectory. Searches an x/y/deeper-z grid biased toward that corner, keeps
    placements that (a) project into the corner box and (b) reach the whole trajectory,
    and returns the deepest reachable (slimmer arm) with the lowest reach-error+jitter.
    Mount is floor for bottom corners (arm rises), ceiling for top corners (arm hangs).
    Falls back to a plain auto-fit if nothing in the corner is reachable."""
    if corner not in _CORNER:
        raise ValueError(f"corner must be one of {list(_CORNER)}")
    sx, sy, mount = _CORNER[corner]
    vi = np.flatnonzero(valid)
    sub = vi[:: max(1, len(vi) // n_sub)]
    sub_fl = flange_poses[sub]; sub_valid = np.ones(len(sub), bool)
    R = _R_cam_base(mount)
    c = np.mean(wrist_pos_cam[valid], axis=0)
    u_lo, u_hi = (0.75 * img_w, 0.98 * img_w) if sx > 0 else (0.02 * img_w, 0.25 * img_w)
    v_lo, v_hi = (0.78 * img_h, 1.00 * img_h) if sy > 0 else (0.02 * img_h, 0.30 * img_h)
    best = None
    for adx in np.linspace(0.2, 0.7, 6):
        for ady in np.linspace(0.2, 0.5, 4):
            for dz in np.linspace(0.3, 1.1, 6):  # deeper, so the arm can span
                bp = c + np.array([sx * adx, sy * ady, dz])
                if bp[2] <= 1e-3:
                    continue
                u, v = focal * bp[0] / bp[2] + img_w / 2, focal * bp[1] / bp[2] + img_h / 2
                if not (u_lo < u < u_hi and v_lo < v < v_hi):
                    continue
                Tcb = pin.SE3(R, bp)
                q, perr, rok = solve_sequence(sub_fl, sub_valid, Tcb, model, data, fid,
                                              w_ori=w_ori, smooth_win=0)
                if rok.mean() < 0.99:
                    continue
                p90 = float(np.nanpercentile(perr, 90))
                jit = float((np.abs(np.diff(q, axis=0)).sum(1) > 0.3).mean())
                # reward depth: a deeper base reads as a slimmer arm from farther back.
                cost = p90 * 1000.0 + 25.0 * jit - 15.0 * float(bp[2])
                if best is None or cost < best[0]:
                    best = (cost, Tcb, bp, (u, v), p90 * 1000, jit)
    if best is None:
        if verbose:
            print(f"[place-{corner}] no reachable base in corner; auto-fit fallback")
        return auto_fit_base(wrist_pos_cam[valid], model, data, fid, mount=mount)
    if verbose:
        print(f"[place-{corner}] base(cam)={best[2].round(3)} px=({best[3][0]:.0f},{best[3][1]:.0f}) "
              f"p90-err {best[4]:.1f}mm jitter {best[5] * 100:.1f}%")
    return best[1]


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
