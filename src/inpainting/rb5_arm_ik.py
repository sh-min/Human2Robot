"""RB5-850e 6-DOF arm IK for the robot-overlay.

There is no real RB5 in the source video — the arm is synthesised at the single
canonical camera-relative base pose defined by ``rb5_build_overlay_input``.
This module only solves per-frame joint angles so the tool flange reaches the
tracked wrist. Pinocchio damped-least-squares IK is warm-started across frames.
Unreachable targets are clamped to the reachable sphere so the arm stops
extending instead of jittering.

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
DEFAULT_POSITION_MARGIN_RAD = 0.02
DEFAULT_VELOCITY_SCALE = 0.90


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


def safe_position_bounds(model, margin=DEFAULT_POSITION_MARGIN_RAD):
    """Return finite joint bounds inset from the URDF hard stops."""
    lo = np.asarray(model.lowerPositionLimit, dtype=np.float64).copy()
    hi = np.asarray(model.upperPositionLimit, dtype=np.float64).copy()
    finite = np.isfinite(lo) & np.isfinite(hi)
    if margin < 0:
        raise ValueError("joint position margin must be non-negative")
    lo[finite] += margin
    hi[finite] -= margin
    if np.any(lo > hi):
        raise ValueError("joint position margin collapses a URDF joint range")
    return lo, hi


def rate_limit_trajectory(q, velocity_limit, fps, *, velocity_scale=1.0,
                          lower=None, upper=None):
    """Project a trajectory onto per-frame position and velocity bounds.

    The causal pass is intentional: frame zero is the commanded starting pose,
    and every later command must be reachable from the command immediately
    before it. This is a command-safety constraint, not merely a diagnostic.
    """
    out = np.asarray(q, dtype=np.float64).copy()
    velocity_limit = np.broadcast_to(
        np.asarray(velocity_limit, dtype=np.float64), out.shape[1:]
    )
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("fps must be finite and positive")
    if not 0 < velocity_scale <= 1:
        raise ValueError("velocity_scale must be in (0, 1]")
    if np.any(~np.isfinite(velocity_limit)) or np.any(velocity_limit <= 0):
        raise ValueError("URDF velocity limits must be finite and positive")
    if lower is not None or upper is not None:
        if lower is None or upper is None:
            raise ValueError("both lower and upper position bounds are required")
        out = np.clip(out, lower, upper)
    max_step = velocity_limit * velocity_scale / float(fps)
    for frame in range(1, len(out)):
        out[frame] = np.clip(
            out[frame], out[frame - 1] - max_step, out[frame - 1] + max_step
        )
        if lower is not None:
            out[frame] = np.clip(out[frame], lower, upper)
    return out


def assert_trajectory_limits(q, model, fps, *,
                             position_margin=DEFAULT_POSITION_MARGIN_RAD,
                             velocity_scale=DEFAULT_VELOCITY_SCALE,
                             atol=1e-7):
    """Fail closed if a trajectory violates the enforced URDF contract."""
    values = np.asarray(q, dtype=np.float64)
    lo, hi = safe_position_bounds(model, position_margin)
    if not np.isfinite(values).all():
        raise ValueError("trajectory contains non-finite joint commands")
    if np.any(values < lo - atol) or np.any(values > hi + atol):
        raise ValueError("trajectory violates safe URDF position bounds")
    if len(values) > 1:
        velocity = np.abs(np.diff(values, axis=0)) * float(fps)
        allowed = np.asarray(model.velocityLimit) * velocity_scale
        if np.any(velocity > allowed + atol):
            raise ValueError("trajectory violates safe URDF velocity bounds")


def assert_no_self_collision(model, data, urdf_path, q, *, label="robot"):
    """Check all non-neighbouring collision links for every command frame."""
    geometry = build_self_collision_geometry(model, urdf_path)
    geometry_data = pin.GeometryData(geometry)
    for frame, command in enumerate(np.asarray(q, dtype=np.float64)):
        pairs = self_collision_pairs(
            model, data, geometry, geometry_data, command
        )
        if pairs:
            raise ValueError(
                f"{label} self-collision at frame {frame}: {', '.join(pairs[:5])}"
            )
    return len(geometry.collisionPairs)


def build_self_collision_geometry(model, urdf_path):
    """Build collision pairs while excluding meshes that meet by design."""
    geometry = pin.buildGeomFromUrdf(
        model,
        str(urdf_path),
        pin.GeometryType.COLLISION,
        package_dirs=[str(Path(urdf_path).resolve().parent)],
    )
    for first in range(len(geometry.geometryObjects)):
        joint_a = geometry.geometryObjects[first].parentJoint
        for second in range(first + 1, len(geometry.geometryObjects)):
            joint_b = geometry.geometryObjects[second].parentJoint
            # Same-link and parent-child meshes meet by construction.
            neighbours = (
                joint_a == joint_b
                or model.parents[joint_a] == joint_b
                or model.parents[joint_b] == joint_a
            )
            if not neighbours:
                geometry.addCollisionPair(pin.CollisionPair(first, second))
    return geometry


def self_collision_pairs(model, data, geometry, geometry_data, command):
    pin.updateGeometryPlacements(
        model, data, geometry, geometry_data,
        np.asarray(command, dtype=np.float64),
    )
    pin.computeCollisions(geometry, geometry_data, False)
    pairs = []
    for pair_index, result in enumerate(geometry_data.collisionResults):
        if not result.isCollision():
            continue
        pair = geometry.collisionPairs[pair_index]
        pairs.append(
            f"{geometry.geometryObjects[pair.first].name}<->"
            f"{geometry.geometryObjects[pair.second].name}"
        )
    return pairs


def project_collision_free_trajectory(model, data, urdf_path, q, *, samples=80,
                                      max_step=None):
    """Minimally retract colliding commands toward the last safe command.

    The input must already satisfy position and velocity limits. A convex blend
    with the preceding safe command preserves both constraints. Frame zero uses
    the neutral/open hand as its safe reference. If a collision-free point
    cannot be found along that segment, generation fails closed.
    """
    if samples < 2:
        raise ValueError("collision projection needs at least two samples")
    out = np.asarray(q, dtype=np.float64).copy()
    geometry = build_self_collision_geometry(model, urdf_path)
    geometry_data = pin.GeometryData(geometry)
    neutral = np.clip(
        pin.neutral(model), model.lowerPositionLimit, model.upperPositionLimit
    )
    if self_collision_pairs(model, data, geometry, geometry_data, neutral):
        raise ValueError("neutral robot configuration is self-colliding")
    adjusted = 0
    for frame in range(len(out)):
        desired = out[frame].copy()
        if frame > 0 and max_step is not None:
            desired = np.clip(
                desired,
                out[frame - 1] - np.asarray(max_step),
                out[frame - 1] + np.asarray(max_step),
            )
            desired = np.clip(
                desired, model.lowerPositionLimit, model.upperPositionLimit
            )
            if not np.allclose(desired, out[frame], atol=1e-12, rtol=0):
                adjusted += 1
            out[frame] = desired
        if not self_collision_pairs(
            model, data, geometry, geometry_data, desired
        ):
            continue
        reference = neutral if frame == 0 else out[frame - 1]
        if self_collision_pairs(
            model, data, geometry, geometry_data, reference
        ):
            raise RuntimeError("collision projection reference is not safe")
        safe = None
        # Search from the desired command toward the safe reference, retaining
        # the greatest possible fraction of the retargeted finger pose.
        for alpha in np.linspace(1.0, 0.0, samples + 1)[1:]:
            candidate = reference + alpha * (desired - reference)
            if not self_collision_pairs(
                model, data, geometry, geometry_data, candidate
            ):
                safe = candidate
                break
        if safe is None:
            raise RuntimeError(
                f"could not project self-collision out of frame {frame}"
            )
        out[frame] = safe
        adjusted += int(np.allclose(desired, q[frame], atol=1e-12, rtol=0))
    return out, adjusted, len(geometry.collisionPairs)


def solve_ik(model, data, fid, q0, target: pin.SE3, *, w_ori=0.0,
             iters=120, damping=5e-2, step=0.4, tol_p=1e-3, tol_o=3e-2,
             lower=None, upper=None):
    """DLS IK with joint-limit clamping. w_ori weights orientation error
    (default 0 = position-only, which a 6-DOF arm can track smoothly; a free
    human wrist's full orientation cannot be followed without singularity jitter).
    """
    lo = model.lowerPositionLimit if lower is None else np.asarray(lower)
    hi = model.upperPositionLimit if upper is None else np.asarray(upper)
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
                   w_ori=0.0, smooth_win=11, initial_q=None, fps=30.0,
                   position_margin=DEFAULT_POSITION_MARGIN_RAD,
                   velocity_scale=DEFAULT_VELOCITY_SCALE):
    """wrist_poses_cam: (T,4,4) flange target in cam frame. Position-only IK by
    default + savgol smoothing of the joint trajectory. Unreachable targets are
    clamped onto the reach sphere so the arm stops extending (no jitter).
    Returns (q[T,6], pos_err[T], reachable[T]).
    """
    T_base_cam = T_cam_base.inverse()
    reach = reach_radius(model, data, fid) * 0.99
    N = len(wrist_poses_cam)
    q = np.zeros((N, model.nq)); perr = np.full(N, np.nan); reachable = np.zeros(N, bool)
    lo, hi = safe_position_bounds(model, position_margin)
    q_prev = (
        pin.neutral(model)
        if initial_q is None
        else np.clip(
            np.asarray(initial_q, dtype=np.float64).reshape(model.nq),
            lo,
            hi,
        )
    )
    max_step = np.asarray(model.velocityLimit) * velocity_scale / float(fps)
    have_command = False
    for t in range(N):
        if not valid[t]:
            q[t] = q_prev; continue
        tgt = T_base_cam * pin.SE3(wrist_poses_cam[t][:3, :3], wrist_poses_cam[t][:3, 3])
        p = tgt.translation; r = float(np.linalg.norm(p))
        reachable[t] = r <= reach
        if r > reach:
            tgt = pin.SE3(tgt.rotation, p * (reach / r))
        frame_lo, frame_hi = lo, hi
        if have_command:
            frame_lo = np.maximum(lo, q_prev - max_step)
            frame_hi = np.minimum(hi, q_prev + max_step)
        q_sol, e = solve_ik(
            model, data, fid, q_prev, tgt, w_ori=w_ori,
            lower=frame_lo, upper=frame_hi,
        )
        # Translation is the hard task: the XHand mount must remain at the
        # tracked wrist. Orientation is followed only with the remaining DOF.
        # A final position-only correction prevents a fast human wrist twist
        # from pulling the whole flange centimetres away while its joints are
        # correctly rate limited.
        if w_ori > 0:
            q_sol, e = solve_ik(
                model, data, fid, q_sol, tgt, w_ori=0.0,
                lower=frame_lo, upper=frame_hi,
            )
        q[t] = q_sol; perr[t] = e; q_prev = q_sol
        have_command = True
    # temporal smoothing on the valid joint trajectory (overlay needs visual, not
    # dynamic, smoothness)
    v = np.asarray(valid, bool)
    if smooth_win and v.sum() > smooth_win:
        q[v] = np.clip(savgol_filter(q[v], smooth_win, 2, axis=0), lo, hi)
    # Savitzky-Golay is not constraint preserving. Project the final commands
    # again and validate what is actually written to disk.
    q = rate_limit_trajectory(
        q, model.velocityLimit, fps, velocity_scale=velocity_scale,
        lower=lo, upper=hi,
    )
    assert_trajectory_limits(
        q, model, fps, position_margin=position_margin,
        velocity_scale=velocity_scale,
    )
    # Report error for the final executable command, not the pre-smoothing IK.
    for t in np.flatnonzero(v):
        target = T_base_cam * pin.SE3(
            wrist_poses_cam[t][:3, :3], wrist_poses_cam[t][:3, 3]
        )
        pin.forwardKinematics(model, data, q[t])
        pin.updateFramePlacements(model, data)
        perr[t] = np.linalg.norm(data.oMf[fid].translation - target.translation)
    return q, perr, reachable
