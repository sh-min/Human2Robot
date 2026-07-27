"""Fit/freeze an external-camera workspace profile for RBY1 trajectories.

With no ``--force``, existing hand entries are never changed.  This is the
important incremental-data property: newly added episodes use the same
mapping, while a hand that was not previously observed can be added later.
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from sim.mujoco_sim.ik_arm import SCENE, _T_WRIST_ARM6, _arm_dof_idx
from workspace_calibration import (
    CALIBRATION_SCHEMA_VERSION,
    RBY1_BASE_FRAME,
    load_calibration,
)

# Comfortable arm configurations whose wrist origins sit on either side of
# the cube workspace.  They are used only to define the workspace anchor.
NATURAL_ARM_QPOS = {
    "left": [-1.2, 0.4, -1.2, -1.0, 0.0, 1.2, 0.8],
    "right": [-1.2, -0.4, 1.2, -1.0, 0.0, 1.2, -0.8],
}

# Overhead recording camera convention:
#   camera +x -> robot +y, camera +y -> robot +x, camera depth -> robot -z.
CAMERA_AXES_TO_WORLD = [
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
]


def _discover_hand_samples(data_root: Path, hand: str):
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    episodes: list[str] = []
    pattern = f"IMG_*/rgb_hawor/qpos_xhand_{hand}_smooth.pkl"
    for path in sorted(data_root.glob(pattern)):
        with path.open("rb") as handle:
            data = pickle.load(handle)
        valid = np.asarray(data["valid"], dtype=bool)
        if not valid.any():
            continue
        pos = np.asarray(data["wrist_pos"], dtype=np.float64)
        quat = np.asarray(data["wrist_quat"], dtype=np.float64)
        positions.append(pos[valid])
        rotations.append(Rotation.from_quat(quat[valid]).as_matrix())
        episodes.append(path.parents[1].name)
    if not positions:
        return None
    return np.concatenate(positions), np.concatenate(rotations), episodes


def _robot_anchor(hand: str) -> tuple[np.ndarray, np.ndarray]:
    muj_model = mujoco.MjModel.from_xml_path(str(SCENE))
    pin_model = pin.buildModelFromMJCF(str(SCENE))
    pin_data = pin_model.createData()
    q = muj_model.qpos0[: pin_model.nq].copy()
    q[_arm_dof_idx(pin_model, hand)] = NATURAL_ARM_QPOS[hand]
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    arm6 = pin_data.oMf[pin_model.getFrameId(f"link_{hand}_arm_6")]
    wrist = arm6 * _T_WRIST_ARM6.inverse()
    return wrist.translation.copy(), wrist.rotation.copy()


def _new_hand_config(hand: str, samples) -> dict:
    positions, rotations, episodes = samples
    anchor_pos, anchor_rot = _robot_anchor(hand)
    reference_rot = Rotation.from_matrix(rotations).mean()
    return {
        "reference_position_camera": np.median(positions, axis=0).tolist(),
        "reference_quaternion_xyzw": reference_rot.as_quat().tolist(),
        "anchor_position_world": anchor_pos.tolist(),
        "anchor_quaternion_xyzw": Rotation.from_matrix(anchor_rot).as_quat().tolist(),
        "natural_arm_qpos": NATURAL_ARM_QPOS[hand],
        "fit_valid_frames": int(len(positions)),
        "fit_episodes": episodes,
    }


def fit_or_extend(
    data_root: Path,
    out_path: Path,
    *,
    position_scale: float,
    orientation_scale: float,
    force: bool,
) -> dict:
    if out_path.exists() and not force:
        profile = load_calibration(out_path)
        print(f"Reusing frozen calibration: {out_path}")
    else:
        profile = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "name": f"{data_root.name}_rby1_external_camera",
            "coordinate_frame": RBY1_BASE_FRAME,
            "camera_axes_to_world": CAMERA_AXES_TO_WORLD,
            "position_scale": position_scale,
            "orientation_scale": orientation_scale,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "hands": {},
        }

    added = []
    for hand in ("right", "left"):
        samples = _discover_hand_samples(data_root, hand)
        if samples is None:
            print(f"  {hand}: no valid samples; leaving profile unchanged")
            continue
        if hand not in profile["hands"]:
            profile["hands"][hand] = _new_hand_config(hand, samples)
            added.append(hand)
            print(
                f"  {hand}: added from "
                f"{profile['hands'][hand]['fit_valid_frames']} valid frames"
            )
        else:
            positions = samples[0]
            ref = np.asarray(
                profile["hands"][hand]["reference_position_camera"],
                dtype=np.float64,
            )
            distance = np.linalg.norm(positions - ref, axis=1)
            print(
                f"  {hand}: frozen; current camera-reference distance "
                f"p95/max={np.percentile(distance, 95):.3f}/"
                f"{distance.max():.3f} m"
            )

    if not profile["hands"]:
        raise RuntimeError(f"No valid hand trajectory found under {data_root}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if added or force or not out_path.exists():
        out_path.write_text(json.dumps(profile, indent=2) + "\n")
        print(f"Wrote calibration: {out_path}")
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--position_scale", type=float, default=0.3)
    parser.add_argument("--orientation_scale", type=float, default=0.5)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the profile, allowing existing hand references to change.",
    )
    args = parser.parse_args()
    fit_or_extend(
        args.data_root.resolve(),
        args.out.resolve(),
        position_scale=args.position_scale,
        orientation_scale=args.orientation_scale,
        force=args.force,
    )


if __name__ == "__main__":
    main()
