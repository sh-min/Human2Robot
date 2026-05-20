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

import numpy as np
from scipy.spatial.transform import Rotation as Rscipy

from _paths import R_MANO_XHAND as _R_DICT

R_MANO_XHAND = {h: R.astype(np.float64) for h, R in _R_DICT.items()}


def export(npz_path, right_pkl, left_pkl, out_path):
    data = np.load(npz_path)
    joints = {"left": data["joints_left"], "right": data["joints_right"]}
    mano_orient = {"left": data["mano_global_orient"][0],   # (T, 3) axis-angle
                   "right": data["mano_global_orient"][1]}
    npz_valid = {"left": data["valid"][0], "right": data["valid"][1]}

    pkl_paths = {"right": right_pkl, "left": left_pkl}

    result = {}
    for hand, pkl_path in pkl_paths.items():
        d = pickle.load(open(pkl_path, "rb"))
        qpos = np.asarray(d["data"])          # (T, 12)
        pkl_valid = np.asarray(d["valid"])    # (T,)
        T = qpos.shape[0]

        wrist_pos = joints[hand][:T, 0, :]   # (T, 3)

        # axis-angle → rotation matrix, then apply MANO→xhand alignment
        R_cam_mano = Rscipy.from_rotvec(mano_orient[hand][:T]).as_matrix()   # (T,3,3)
        R_cam_xhand = R_cam_mano @ R_MANO_XHAND[hand]                        # (T,3,3)

        valid = npz_valid[hand][:T] & pkl_valid

        result[hand] = {
            "wrist_pos":   wrist_pos.astype(np.float32),
            "wrist_rot":   R_cam_xhand.astype(np.float32),
            "qpos":        qpos,
            "joint_names": d["joint_names"],
            "valid":       valid,
        }

    result["T"] = result["right"]["qpos"].shape[0]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"saved {out_path}  (T={result['T']})")
    for hand in ("right", "left"):
        v = result[hand]["valid"]
        print(f"  {hand:5s}  valid={v.sum()}/{len(v)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="retarget_input.npz path")
    ap.add_argument("--right_pkl", default=None,
                    help="qpos pkl for right hand (default: <npz_dir>/qpos_xhand_contact_right.pkl)")
    ap.add_argument("--left_pkl", default=None,
                    help="qpos pkl for left hand (default: <npz_dir>/qpos_xhand_contact_left.pkl)")
    ap.add_argument("--out", default=None,
                    help="output path (default: <npz_dir>/final_pose.pkl)")
    args = ap.parse_args()

    npz_dir = os.path.dirname(os.path.abspath(args.npz))
    right_pkl = args.right_pkl or os.path.join(npz_dir, "qpos_xhand_contact_right.pkl")
    left_pkl  = args.left_pkl  or os.path.join(npz_dir, "qpos_xhand_contact_left.pkl")
    out_path  = args.out       or os.path.join(npz_dir, "final_pose.pkl")

    export(args.npz, right_pkl, left_pkl, out_path)


if __name__ == "__main__":
    main()
