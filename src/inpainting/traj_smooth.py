"""Temporal smoothing of the robot-hand trajectory, applied at render time.

The depth renderer re-derives the wrist pose and global orientation from the
raw HaWoR npz (not from the retargeted pkl), so the most visible jitter — the
wrist and arm — is not touched by the retargeter's own --smooth. Smoothing here
covers everything the renderer actually draws: finger qpos, wrist position, and
global orientation, each over the time axis, interpolating across invalid
frames first so gaps don't smear.

Ported from src/retargeting/retarget_from_npz.py:smooth_trajectory.
"""
import numpy as np
from scipy.signal import medfilt, savgol_filter
from scipy.spatial.transform import Rotation


def _odd(n, cap):
    n = min(int(n), int(cap))
    return n - 1 if n % 2 == 0 else n


def _interp_invalid(arr, valid):
    """Linear-interpolate per channel where *valid* is False."""
    arr = np.asarray(arr, dtype=np.float64)
    if valid.all():
        return arr.copy()
    out = arr.copy()
    t = np.arange(len(out))
    vi = np.where(valid)[0]
    if len(vi) < 2:
        return out
    for c in range(out.shape[1]):
        out[~valid, c] = np.interp(t[~valid], vi, out[vi, c])
    return out


def smooth_channels(arr, valid, win=15, poly=3, med_win=5):
    """Smooth a (T, C) series: interp invalid -> median(med_win) -> savgol(win).

    Returns float32 the same shape as *arr*.
    """
    arr = np.asarray(arr, dtype=np.float64)
    flat = arr.reshape(len(arr), -1)
    x = _interp_invalid(flat, valid)
    mw = _odd(med_win, len(x))
    if mw >= 3:
        x = medfilt(x, kernel_size=(mw, 1))
    w = _odd(win, len(x))
    if w >= poly + 2:
        x = savgol_filter(x, w, poly, axis=0)
    return x.reshape(arr.shape).astype(np.float32)


def smooth_rotvec(rotvec, valid, win=21, poly=3):
    """Smooth an axis-angle (T, 3) orientation series via its quaternion.

    Median filtering is skipped (nonsensical on rotations); instead the
    quaternions are antipodally unwrapped, savgol-smoothed, and renormalised.
    Returns float32 (T, 3) axis-angle.
    """
    rotvec = np.asarray(rotvec, dtype=np.float64)
    q = Rotation.from_rotvec(_interp_invalid(rotvec, valid)).as_quat()  # xyzw
    for k in range(1, len(q)):                                          # unwrap
        if np.dot(q[k - 1], q[k]) < 0:
            q[k] = -q[k]
    w = _odd(win, len(q))
    if w >= poly + 2:
        q = savgol_filter(q, w, poly, axis=0)
    q /= np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-9, None)
    return Rotation.from_quat(q).as_rotvec().astype(np.float32)
