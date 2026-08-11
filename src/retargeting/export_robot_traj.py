"""Combine wrist pose (retarget_input.npz) + finger qpos (qpos_xhand_contact_*.pkl)
into a single final_pose.pkl for robot execution.

Output schema (per hand):
    wrist_pos   (T, 3)      cam-frame wrist position
    wrist_rot   (T, 3, 3)   cam-frame wrist rotation in xhand frame
    qpos        (T, 12)     finger joint angles
    joint_names list[str]   qpos joint order
    valid       (T,)  bool  frame is valid for both HaWoR and retargeting

Usage:
    python export_robot_traj.py --npz /path/to/rgb_hawor/retarget_input.npz
"""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rscipy

from _paths import R_MANO_XHAND as _R_DICT
from workspace_calibration import (
    RBY1_BASE_FRAME,
    calibration_sha256,
    load_calibration,
    transform_wrist_sequence,
)

R_MANO_XHAND = {h: R.astype(np.float64) for h, R in _R_DICT.items()}


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def export(npz_path, right_pkl, left_pkl, out_path, calibration_path=None):
    data = np.load(npz_path)
    joints = {"left": data["joints_left"], "right": data["joints_right"]}
    mano_orient = {"left": data["mano_global_orient"][0],   # (T, 3) axis-angle
                   "right": data["mano_global_orient"][1]}
    npz_valid = {"left": data["valid"][0], "right": data["valid"][1]}

    pkl_paths = {"right": right_pkl, "left": left_pkl}
    pkls = {hand: _load_pkl(path) for hand, path in pkl_paths.items()}

    lengths = []
    for hand in ("right", "left"):
        d = pkls[hand]
        lengths.extend([
            len(d["data"]),
            len(d["valid"]),
            len(joints[hand]),
            len(mano_orient[hand]),
            len(npz_valid[hand]),
        ])
        if "wrist_pos" in d:
            lengths.append(len(d["wrist_pos"]))
        if "wrist_quat" in d:
            lengths.append(len(d["wrist_quat"]))
    T = min(lengths)
    if T <= 0:
        raise ValueError(f"No aligned frames across inputs: {lengths}")

    profile = (
        load_calibration(calibration_path)
        if calibration_path is not None
        else None
    )
    result = {}
    for hand in ("right", "left"):
        d = pkls[hand]
        qpos = np.asarray(d["data"][:T], dtype=np.float32)
        pkl_valid = np.asarray(d["valid"][:T], dtype=bool)

        # Smooth retarget PKLs carry a temporally filtered wrist trajectory.
        # Prefer it so final_pose exactly matches the rendered replacement.
        if "wrist_pos" in d and "wrist_quat" in d:
            wrist_pos = np.asarray(d["wrist_pos"][:T], dtype=np.float32)
            wrist_quat = np.asarray(d["wrist_quat"][:T], dtype=np.float32)
            R_cam_xhand = Rscipy.from_quat(wrist_quat).as_matrix()
            wrist_source = "pkl"
        else:
            wrist_pos = joints[hand][:T, 0, :]
            # axis-angle → rotation matrix, then apply MANO→xhand alignment
            R_cam_mano = Rscipy.from_rotvec(
                mano_orient[hand][:T]
            ).as_matrix()
            R_cam_xhand = R_cam_mano @ R_MANO_XHAND[hand]
            wrist_source = "npz"

        valid = npz_valid[hand][:T] & pkl_valid
        finite = (
            np.isfinite(qpos).all(axis=1)
            & np.isfinite(wrist_pos).all(axis=1)
            & np.isfinite(R_cam_xhand).all(axis=(1, 2))
        )
        valid &= finite

        wrist_pos_camera = wrist_pos.astype(np.float32)
        wrist_rot_camera = R_cam_xhand.astype(np.float32)
        if profile is not None:
            wrist_pos, R_cam_xhand = transform_wrist_sequence(
                wrist_pos_camera,
                wrist_rot_camera,
                valid,
                hand,
                profile,
            )
            coordinate_frame = RBY1_BASE_FRAME
        else:
            coordinate_frame = "camera_cv"

        result[hand] = {
            "wrist_pos":   wrist_pos.astype(np.float32),
            "wrist_rot":   R_cam_xhand.astype(np.float32),
            "wrist_pos_camera": wrist_pos_camera,
            "wrist_rot_camera": wrist_rot_camera,
            "qpos":        qpos,
            "joint_names": d["joint_names"],
            "valid":       valid,
            "wrist_source": wrist_source,
            "coordinate_frame": coordinate_frame,
        }
        if profile is not None and hand in profile["hands"]:
            result[hand]["natural_arm_qpos"] = profile["hands"][hand][
                "natural_arm_qpos"
            ]

    result["T"] = T
    result["hands"] = [
        hand for hand in ("right", "left")
        if bool(np.asarray(result[hand]["valid"]).any())
    ]
    if not result["hands"]:
        raise ValueError("Neither hand has a valid frame")
    result["source"] = {
        "npz": str(Path(npz_path).resolve()),
        "right_pkl": str(Path(right_pkl).resolve()),
        "left_pkl": str(Path(left_pkl).resolve()),
    }
    result["coordinate_frame"] = (
        RBY1_BASE_FRAME if profile is not None else "camera_cv"
    )
    if profile is not None:
        result["workspace_calibration"] = profile
        result["source"]["workspace_calibration"] = str(
            Path(calibration_path).resolve()
        )
        result["source"]["workspace_calibration_sha256"] = calibration_sha256(
            calibration_path
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(
        f"saved {out_path}  (T={result['T']}, "
        f"hands={','.join(result['hands'])})"
    )
    for hand in ("right", "left"):
        v = result[hand]["valid"]
        print(
            f"  {hand:5s}  valid={v.sum()}/{len(v)} "
            f"wrist={result[hand]['wrist_source']}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="retarget_input.npz path")
    ap.add_argument("--right_pkl", default=None,
                    help="qpos pkl for right hand (default: <npz_dir>/qpos_xhand_contact_right.pkl)")
    ap.add_argument("--left_pkl", default=None,
                    help="qpos pkl for left hand (default: <npz_dir>/qpos_xhand_contact_left.pkl)")
    ap.add_argument("--out", default=None,
                    help="output path (default: <npz_dir>/final_pose.pkl)")
    ap.add_argument(
        "--calibration",
        default=None,
        help="External-camera -> RBY1 workspace calibration JSON.",
    )
    args = ap.parse_args()

    npz_dir = os.path.dirname(os.path.abspath(args.npz))
    right_pkl = args.right_pkl or os.path.join(npz_dir, "qpos_xhand_contact_right.pkl")
    left_pkl  = args.left_pkl  or os.path.join(npz_dir, "qpos_xhand_contact_left.pkl")
    out_path  = args.out       or os.path.join(npz_dir, "final_pose.pkl")

    export(
        args.npz,
        right_pkl,
        left_pkl,
        out_path,
        calibration_path=args.calibration,
    )


if __name__ == "__main__":
    main()
