"""External-camera wrist poses -> RBY1 base-frame workspace poses.

HaWoR reconstructs a wrist in the recording camera frame.  That camera is
not the RBY1 head camera, so composing it with the robot head-camera pose
produces physically invalid arm targets.  A workspace calibration preserves
the recorded *relative* motion while anchoring it at a reachable robot pose.

The calibration is intentionally stored next to a dataset root.  It is fitted
once for a fixed camera setup, then reused unchanged when more episodes are
added.  Missing hands can be appended later without changing an existing
hand's reference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

CALIBRATION_SCHEMA_VERSION = 1
RBY1_BASE_FRAME = "rby1_base"


def load_calibration(path: str | Path) -> dict:
    path = Path(path)
    profile = json.loads(path.read_text())
    if profile.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported workspace calibration schema in {path}: "
            f"{profile.get('schema_version')!r}"
        )
    if profile.get("coordinate_frame") != RBY1_BASE_FRAME:
        raise ValueError(
            f"Calibration {path} targets {profile.get('coordinate_frame')!r}, "
            f"expected {RBY1_BASE_FRAME!r}"
        )

    axes = np.asarray(profile["camera_axes_to_world"], dtype=np.float64)
    if axes.shape != (3, 3) or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6):
        raise ValueError(f"camera_axes_to_world in {path} is not a rotation")
    if not np.isclose(np.linalg.det(axes), 1.0, atol=1e-6):
        raise ValueError(f"camera_axes_to_world in {path} must be right-handed")
    return profile


def calibration_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform_wrist_sequence(
    wrist_pos_camera: np.ndarray,
    wrist_rot_camera: np.ndarray,
    valid: np.ndarray,
    hand: str,
    profile: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a camera-frame wrist sequence into the calibrated RBY1 workspace."""
    if hand not in profile.get("hands", {}):
        if np.asarray(valid, dtype=bool).any():
            raise ValueError(
                f"Calibration has no {hand!r} reference. Re-run "
                "fit_workspace_calibration.py to extend the profile."
            )
        T = len(valid)
        return (
            np.zeros((T, 3), dtype=np.float32),
            np.tile(np.eye(3, dtype=np.float32), (T, 1, 1)),
        )

    cfg = profile["hands"][hand]
    pos = np.asarray(wrist_pos_camera, dtype=np.float64)
    rot = np.asarray(wrist_rot_camera, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if pos.shape != (len(valid), 3) or rot.shape != (len(valid), 3, 3):
        raise ValueError(
            f"Invalid {hand} wrist shapes: pos={pos.shape}, rot={rot.shape}, "
            f"valid={valid.shape}"
        )

    axes = np.asarray(profile["camera_axes_to_world"], dtype=np.float64)
    pos_scale = float(profile["position_scale"])
    ori_scale = float(profile["orientation_scale"])
    ref_pos = np.asarray(cfg["reference_position_camera"], dtype=np.float64)
    ref_rot = Rotation.from_quat(
        np.asarray(cfg["reference_quaternion_xyzw"], dtype=np.float64)
    ).as_matrix()
    anchor_pos = np.asarray(cfg["anchor_position_world"], dtype=np.float64)
    anchor_rot = Rotation.from_quat(
        np.asarray(cfg["anchor_quaternion_xyzw"], dtype=np.float64)
    ).as_matrix()

    out_pos = np.zeros_like(pos)
    out_rot = np.tile(np.eye(3, dtype=np.float64), (len(valid), 1, 1))
    idx = np.flatnonzero(valid)
    if len(idx):
        out_pos[idx] = (
            anchor_pos
            + pos_scale * (pos[idx] - ref_pos) @ axes.T
        )
        relative = ref_rot.T[None, :, :] @ rot[idx]
        relative_rotvec = Rotation.from_matrix(relative).as_rotvec()
        scaled_relative = Rotation.from_rotvec(
            ori_scale * relative_rotvec
        ).as_matrix()
        out_rot[idx] = anchor_rot[None, :, :] @ scaled_relative

    return out_pos.astype(np.float32), out_rot.astype(np.float32)
